// SPDX-License-Identifier: Apache-2.0
//
// Debug.swift — Debug tab: live log pane fed by the C trace ring +
// "Run Tests" button that invokes slm_run_all_tests on a detached
// task.
//
// The C side fires `on_trace` once per trace() call from whatever
// thread is currently executing inside slm runtime. We bridge to the
// MainActor via DispatchQueue.main.async so SwiftUI sees state
// changes on its own runloop. Lines are capped to a recent window
// (lines.count > maxLines drops the oldest in one shot).

import SwiftUI
import Observation
import Foundation
import Darwin    // for posix_strerror etc. if ever needed
#if os(macOS)
import AppKit
#else
import UIKit
#endif

@Observable
@MainActor
final class TraceLog {

    /// Most-recent log lines (newest at the end). Capped to maxLines.
    var lines: [String] = []

    /// Joined-text view of `lines` (one big newline-separated String).
    /// The DebugView renders this as a single Text so the user can
    /// drag-select across multiple lines (the prior per-line LazyVStack
    /// limited selection to one line at a time). Lazily recomputed on
    /// access — SwiftUI's diffing notices when it changes.
    var joinedText: String { lines.joined(separator: "\n") }

    /// Cap; trimming happens in a single removeFirst(N) so SwiftUI
    /// observes one change per arrival burst, not N changes.
    private static let maxLines = 2000

    /// Total number of lines ever observed (for diagnostics).
    private(set) var totalObserved: UInt64 = 0

    func append(_ line: String) {
        lines.append(line)
        totalObserved &+= 1
        if lines.count > Self.maxLines {
            lines.removeFirst(lines.count - Self.maxLines)
        }
    }

    func clear() {
        lines.removeAll()
    }

}

/// Bridge plate: a singleton that the C trace observer pokes. The C
/// callback runs OFF the MainActor; we hop via DispatchQueue.main.
/// We use a class-static `current` so the @_cdecl function pointer
/// has somewhere stable to write through.
final class TraceBridge: @unchecked Sendable {

    static let shared = TraceBridge()

    /// Set by the SwiftUI side; nil = no UI consumer attached.
    /// Reads + writes serialized on MainActor.
    @MainActor private(set) var log: TraceLog?

    @MainActor
    func attach(_ log: TraceLog) {
        self.log = log
    }

    private var subscribed = false

    /// Call once at app startup to wire the C observer. Idempotent.
    func subscribeOnce() {
        if subscribed { return }
        subscribed = true
        var observer = trace_observer(
            that:     Unmanaged.passUnretained(self).toOpaque(),
            on_trace: traceCallbackCFn)
        trace_subscribe(&observer)
    }

}

// Free function so it has C-compatible calling convention. The
// observer carries `that` (= TraceBridge.shared opaque ptr) so we
// can find the bridge back.
private func traceCallbackCFn(obsPtr: UnsafePointer<trace_observer>?,
                              entryPtr: UnsafePointer<trace_entry>?) {
    guard let obsPtr   = obsPtr,
          let entryPtr = entryPtr,
          let thatPtr  = obsPtr.pointee.that else {
        return
    }
    let entry = entryPtr.pointee
    let timestamp = entry.timestamp
    let file = entry.file.map { String(cString: $0) } ?? ""
    let function = entry.function.map { String(cString: $0) } ?? ""
    let line = Int(entry.line)
    // The message is a heap pointer + length on the C side now (no
    // fixed cap). trace_message() returns the raw bytes; we decode
    // to a Swift String via UTF-8.
    var msgLen: size_t = 0
    let message: String
    if let p = trace_message(entryPtr, &msgLen), msgLen > 0 {
        let buf = UnsafeBufferPointer(
            start: UnsafeRawPointer(p).assumingMemoryBound(to: UInt8.self),
            count: Int(msgLen))
        message = String(decoding: buf, as: UTF8.self)
    } else {
        message = ""
    }
    // Strip trailing newline so SwiftUI Text doesn't render an empty
    // gap line beneath each entry.
    let trimmed = message.hasSuffix("\n")
        ? String(message.dropLast()) : message
    let formatted = String(format: "%10.6f %@:%d @%@ %@",
                           timestamp, file, line, function, trimmed)
    // Hop to the MainActor via DispatchQueue — same pattern the chat
    // stream callback uses; runloop-integrated, one render tick per
    // line, no executor coalescing.
    DispatchQueue.main.async {
        TraceBridge.shared.log?.append(formatted)
    }
    _ = thatPtr  // silence unused; keeps observer.that referenced
}

/// The Debug tab body. Renders the trace log in a scroll view and
/// gives a button to kick the C self-test battery.
struct DebugView: View {

    @Bindable var log: TraceLog

    @State private var runningTests = false
    @State private var lastResult: String? = nil

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Button {
                    Task { await runTests() }
                } label: {
                    if runningTests {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 16, height: 16)
                    } else {
                        Image(systemName: "play.fill")
                    }
                    Text(runningTests ? "Running…" : "Run Tests")
                }
                .buttonStyle(.borderedProminent)
                .disabled(runningTests)

                Button("Copy") { copyLogToPasteboard() }
                    .buttonStyle(.bordered)
                    .disabled(log.lines.isEmpty)

                Button("Clear log") { log.clear() }
                    .buttonStyle(.bordered)

                Spacer()

                if let r = lastResult {
                    Text(r)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Text("\(log.lines.count) lines (of \(log.totalObserved) total)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    // Single Text node carrying all lines joined by \n
                    // so the user can drag-select across multiple lines
                    // and Cmd-C them (the prior per-line LazyVStack
                    // restricted selection to one line at a time).
                    // Tail sentinel sits below so scrollTo("tail")
                    // can pin the view to the bottom on append.
                    VStack(alignment: .leading, spacing: 0) {
                        Text(log.joinedText)
                            .font(.system(size: 11,
                                          weight: .regular,
                                          design: .monospaced))
                            .textSelection(.enabled)
                            .lineLimit(nil)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity,
                                   alignment: .leading)
                        Color.clear.frame(height: 1).id("tail")
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }
                .onChange(of: log.lines.count) {
                    withAnimation(.linear(duration: 0.1)) {
                        proxy.scrollTo("tail", anchor: .bottom)
                    }
                }
            }
        }
    }

    private func runTests() async {
        runningTests = true
        lastResult = nil
        let failures = await Task.detached(priority: .userInitiated) {
            Int(slm_run_all_tests())
        }.value
        runningTests = false
        lastResult = (failures == 0)
            ? "all PASS" : "\(failures) FAIL"
    }

    private func copyLogToPasteboard() {
        let payload = log.lines.joined(separator: "\n")
        #if os(macOS)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(payload, forType: .string)
        #else
        UIPasteboard.general.string = payload
        #endif
    }

}
