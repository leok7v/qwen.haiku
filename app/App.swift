import SwiftUI
import Observation
import Foundation
#if os(macOS)
import AppKit
#endif

// One chat bubble's worth of state. Stored on SLMViewModel; the C
// side accumulates its own UTF-8 message fragments on slm_ctx, but
// the UI wants a turn-grained view with reasoning split out, so we
// keep our own slim list here.
struct ChatMessage: Identifiable, Equatable, Sendable {
    enum Role: String, Sendable {
        case user
        case assistant
    }
    let id:        UUID
    var role:      Role
    var content:   String
    var reasoning: String?
    init(role: Role, content: String, reasoning: String? = nil) {
        self.id        = UUID()
        self.role      = role
        self.content   = content
        self.reasoning = reasoning
    }
}

#if os(macOS)
final class AppDelegate: NSObject, NSApplicationDelegate {

    func applicationShouldTerminateAfterLastWindowClosed(
        _ sender: NSApplication
    ) -> Bool {
        return true
    }

}
#endif

@main
struct App: SwiftUI.App {

    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self)
    var appDelegate
    #endif

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .commands {
            CommandGroup(replacing: .newItem) { }
        }
        #endif
    }

}

@Observable
@MainActor
final class SLMViewModel {

    enum State: Equatable {
        case idle                                    // initial, nothing checked yet
        case needsDownload(size: Int)                // cache empty, awaiting user OK
        case downloading(done: Int, total: Int)      // user pressed Download
        case loading                                 // model on disk, mmap+parse
        case ready                                   // ctx alive, can generate
        case generating
        case error(String)
    }

    var state:        State        = .idle
    var systemPrompt: String       = "You are a helpful assistant.\n"
    var messages:     [ChatMessage] = []
    var think:        Bool          = false
    var tools:        Bool          = false
    var debug:        Bool          = true

    // Throughput from the most recent completed generation.
    // Zero before the first one finishes.
    var lastPP:        Double = 0
    var lastTG:        Double = 0
    var lastNPrefill:  Int    = 0
    var lastNGen:      Int    = 0

    // True between the moment a turn starts and the C side fires
    // its `.prefilled` chunk (prompt prefill done, decode begins).
    // Used to show a "thinking..." indicator in the status line.
    // No granular per-token progress — at this model size prefill
    // is ~1s and a fine-grained bar is more noise than value.
    var prefilling:    Bool = false

    @ObservationIgnored private var slm:        SLM?
    @ObservationIgnored private var downloader: Downloader?
    @ObservationIgnored
    nonisolated(unsafe) private var stopRequested = false

    func checkCache() {
        do {
            let dl = try Downloader()
            self.downloader = dl
            if dl.isCached {
                self.state = .loading
                Task { await self.loadModel() }
            } else {
                self.state = .needsDownload(size: dl.model.expectedSize)
            }
        } catch {
            self.state = .error(String(describing: error))
        }
    }

    func beginDownload() async {
        if let dl = self.downloader {
            self.state = .downloading(done: 0, total: dl.model.expectedSize)
            do {
                _ = try await dl.ensureAvailable { [weak self] done, total in
                    Task { @MainActor in
                        self?.state = .downloading(done: done, total: total)
                    }
                    return false
                }
                self.state = .loading
                await self.loadModel()
            } catch {
                self.state = .error(String(describing: error))
            }
        }
    }

    private func loadModel() async {
        if let dl = self.downloader {
            let path = dl.localURL
            let sampler = SLM.Sampler(temperature:       0.7,
                                      topK:              40,
                                      topP:              0.9,
                                      minP:              0.05,
                                      repetitionPenalty: 1.25,
                                      repetitionWindow:  64,
                                      maxNew:            512,
                                      minNew:            8)
            let toolsOn = self.tools
            let thinkOn = self.think
            let debugLv: Int32 = self.debug ? 1 : 0
            let sysText = self.systemPrompt
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    () throws -> SLM in
                    let slm = try SLM(modelPath: path, sampler: sampler)
                    // Set ctrl BEFORE prefilling the system block —
                    // tools/think get baked into the in-context framing.
                    slm.ctrl.pointee.tools  = toolsOn
                    slm.ctrl.pointee.think  = thinkOn
                    slm.ctrl.pointee.effort = nil
                    slm.ctrl.pointee.debug  = debugLv
                    try slm.prefillSystem(text: sysText)
                    return slm
                }.value
                self.slm   = loaded
                self.state = .ready
            } catch {
                self.state = .error(String(describing: error))
            }
        }
    }

    func send(_ text: String) async {
        let trimmedUser = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if slm != nil, state == .ready, !trimmedUser.isEmpty {
            self.messages.append(
                ChatMessage(role: .user, content: trimmedUser))
            self.messages.append(
                ChatMessage(role: .assistant, content: ""))
            let assistantIdx = self.messages.count - 1
            self.state         = .generating
            self.stopRequested = false
            self.prefilling    = true
            // Debug is the only ctrl knob safe to flip mid-conversation
            // (tools/think are frozen after prefillSystem). Push it
            // down so the next generate() sees the current toggle.
            self.slm?.ctrl.pointee.debug = self.debug ? 1 : 0
            await self.streamAssistant(prompt: trimmedUser,
                                       at: assistantIdx)
        }
    }

    private func streamAssistant(prompt: String, at idx: Int) async {
        if let slm = self.slm {
            let debugOn = self.debug
            let thinkOn = self.think
            await Task.detached(priority: .userInitiated) { [weak self] in
                // The C side already strips <think>...</think> /
                // <tool_call>...</tool_call> markers; chunks land in
                // the right enum case. No client-side state machine.
                _ = slm.generate(prompt: prompt) { chunk in
                    var keepGoing = true
                    switch chunk {
                    case .content(let s):
                        Task { @MainActor in
                            self?.appendAssistant(s, at: idx)
                        }
                    case .reasoning(let s):
                        if thinkOn {
                            Task { @MainActor in
                                self?.appendReasoning(s, at: idx)
                            }
                        }
                    case .toolCall(let body):
                        if debugOn {
                            let line = "\n[tool: " + body + "]\n"
                            Task { @MainActor in
                                self?.appendAssistant(line, at: idx)
                            }
                        }
                    case .toolResponse(let body):
                        if debugOn {
                            let n = min(body.count, 400)
                            let prefix = String(body.prefix(n))
                            let line = "\n[result: " + prefix + "]\n"
                            Task { @MainActor in
                                self?.appendAssistant(line, at: idx)
                            }
                        }
                    case .prefilled:
                        Task { @MainActor in
                            self?.prefilling = false
                        }
                    case .pp:
                        // Per-token prefill progress chunks. The view
                        // model just shows a "thinking…" indicator;
                        // ignore the position counter.
                        break
                    }
                    if keepGoing {
                        keepGoing = !(self?.stopRequested ?? true)
                    }
                    return keepGoing
                }
                let pp = slm.ppPerSec
                let tg = slm.tgPerSec
                let np = slm.nPrefill
                let ng = slm.nGenerated
                Task { @MainActor in
                    self?.finishTurn(pp: pp, tg: tg, np: np, ng: ng)
                }
            }.value
        }
    }

    private func appendAssistant(_ piece: String, at idx: Int) {
        if idx < self.messages.count {
            self.messages[idx].content += piece
        }
    }

    private func appendReasoning(_ piece: String, at idx: Int) {
        if idx < self.messages.count {
            let prev = self.messages[idx].reasoning ?? ""
            self.messages[idx].reasoning = prev + piece
        }
    }

    private func finishTurn(pp: Double, tg: Double, np: Int, ng: Int) {
        self.lastPP       = pp
        self.lastTG       = tg
        self.lastNPrefill = np
        self.lastNGen     = ng
        self.state        = .ready
    }

    func stopGeneration() { self.stopRequested = true }

    func clearChat() {
        self.messages.removeAll()
        if let slm = self.slm {
            let sysText = self.systemPrompt
            let toolsOn = self.tools
            let thinkOn = self.think
            let debugLv: Int32 = self.debug ? 1 : 0
            do {
                try slm.newConversation()
                slm.ctrl.pointee.tools  = toolsOn
                slm.ctrl.pointee.think  = thinkOn
                slm.ctrl.pointee.effort = nil
                slm.ctrl.pointee.debug  = debugLv
                try slm.prefillSystem(text: sysText)
            } catch {
                self.state = .error(String(describing: error))
            }
        }
    }

}

/// Compact capsule-style on/off chip. Replaces the cross-platform
/// `Toggle(...).toggleStyle(...)` row in the system bar — iOS's
/// UISwitch is too wide for a three-toggle row on phone width, and
/// `.checkbox` is macOS-only. This renders the same on both: a
/// monospaced caption inside a capsule whose tint shifts when on.
struct ChipToggle: View {
    let title: String
    @Binding var isOn: Bool
    init(_ title: String, isOn: Binding<Bool>) {
        self.title = title
        self._isOn = isOn
    }
    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            Text(title)
                .font(.caption.monospaced())
                .padding(.horizontal, 10)
                .padding(.vertical,    5)
                .background(
                    Capsule()
                        .fill(isOn
                              ? Color.accentColor.opacity(0.35)
                              : Color.secondary.opacity(0.18)))
                .overlay(
                    Capsule()
                        .stroke(isOn
                                ? Color.accentColor.opacity(0.7)
                                : Color.secondary.opacity(0.30),
                                lineWidth: 1))
                .foregroundStyle(isOn ? Color.primary : Color.secondary)
        }
        .buttonStyle(.plain)
        .contentShape(Capsule())
    }
}

struct ContentView: View {

    @State private var vm    = SLMViewModel()
    @State private var input: String = ""
    @State private var showSystem    = false

    private static let suggestions: [String] = [
//      "Search Internet to find what rock band from which country"
//          + " recorded HelloWorld album?",
//      "What is bitcoin price today?",
        "Write me a haiku.",
    ]

    var body: some View {
        VStack(spacing: 0) {
            statusView
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            Divider()
            systemBar
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            Divider()
            if case .needsDownload(let size) = vm.state {
                downloadPrompt(size: size)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
            }
            if vm.messages.isEmpty {
                suggestionsView
            } else {
                messagesView
            }
            Divider()
            inputBar
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
        }
        .task { vm.checkCache() }
    }

    @ViewBuilder
    private var systemBar: some View {
        HStack(alignment: .top, spacing: 8) {
            Text("system")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .padding(.top, 4)
            if showSystem {
                TextField("system prompt",
                          text: $vm.systemPrompt,
                          axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1 ... 4)
                    .disabled(vm.state == .generating)
            } else {
                Text(vm.systemPrompt.isEmpty ? "(none)" : vm.systemPrompt)
                    .font(.callout)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .foregroundStyle(vm.systemPrompt.isEmpty
                                     ? .secondary : .primary)
                Spacer()
            }
            Button(showSystem ? "done" : "edit") { showSystem.toggle() }
                .buttonStyle(.bordered)
                .controlSize(.small)
            // iOS UISwitch toggles eat way too much horizontal space
            // for a three-toggle row on phone width; on macOS the
            // .checkbox style is the right one but it's iOS-
            // unavailable. ChipToggle below renders a compact tap-
            // able capsule on both platforms — same width budget on
            // iPhone as a `Text("Tools")`.
            ChipToggle("Tools", isOn: $vm.tools)
                .disabled(vm.state == .generating)
            ChipToggle("Think", isOn: $vm.think)
                .disabled(vm.state == .generating)
            ChipToggle("Debug", isOn: $vm.debug)
        }
    }

    @ViewBuilder
    private var suggestionsView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                Text("Try one of these")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 4)
                ForEach(Self.suggestions, id: \.self) { s in
                    Button {
                        if vm.state == .ready { Task { await vm.send(s) } }
                    } label: {
                        Text(s)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .background(.background.secondary)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                    .disabled(vm.state != .ready)
                }
            }
            .padding(16)
        }
    }

    @ViewBuilder
    private var messagesView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(vm.messages) { m in
                        messageRow(m).id(m.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: vm.messages.last?.content) {
                if let last = vm.messages.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    @ViewBuilder
    private func messageRow(_ m: ChatMessage) -> some View {
        if m.role == .user {
            HStack {
                Spacer(minLength: 40)
                Text(m.content)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.tint.opacity(0.25))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .textSelection(.enabled)
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                if let r = m.reasoning, !r.isEmpty {
                    Text(r)
                        .font(.callout.italic())
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.background.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .textSelection(.enabled)
                }
                if m.content.isEmpty {
                    ShinyWhimsyView()
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text(m.content)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
            }
        }
    }

    @ViewBuilder
    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Ask anything", text: $input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1 ... 5)
                .disabled(vm.state != .ready)
                .onSubmit { trySend() }
            if vm.state == .generating {
                Button("Stop") { vm.stopGeneration() }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
            } else {
                Button("Send") { trySend() }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.state != .ready ||
                              input.trimmingCharacters(in: .whitespacesAndNewlines)
                                   .isEmpty)
            }
            Button("Clear") { vm.clearChat() }
                .buttonStyle(.bordered)
                .disabled(vm.messages.isEmpty || vm.state == .generating)
        }
    }

    private func trySend() {
        let text    = input
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, vm.state == .ready {
            input = ""
            Task { await vm.send(text) }
        }
    }

    @ViewBuilder
    private func downloadPrompt(size: Int) -> some View {
        let mb = Double(size) / 1_048_576.0
        HStack(spacing: 12) {
            Text(String(format: "Model not cached (~%.0f MB).", mb))
                .font(.callout)
            Spacer()
            Button("Download model") {
                Task { await vm.beginDownload() }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(8)
        .background(.background.secondary)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var statusView: some View {
        TimelineView(.periodic(from: .now, by: 1.0)) { _ in
            statusLine
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        let rss = formatMB(residentBytes())
        switch vm.state {
        case .idle:
            Text("starting...")
        case .needsDownload:
            Text("model not downloaded")
        case .downloading(let done, let total):
            let mb      = Double(done)  / 1_048_576.0
            let totalMb = Double(total) / 1_048_576.0
            Text(String(format: "downloading: %.1f / %.1f MB", mb, totalMb))
        case .loading:
            Text("loading weights...  |  \(rss)")
        case .ready:
            if vm.lastNGen > 0 {
                let stats = String(format: "pp %.1f  tg %.1f tok/s  (%d+%d tok)",
                                   vm.lastPP, vm.lastTG,
                                   vm.lastNPrefill, vm.lastNGen)
                Text("ready  |  \(stats)  |  \(rss)")
            } else {
                Text("ready  |  \(rss)")
            }
        case .generating:
            if vm.prefilling {
                Text("prefilling...  |  \(rss)")
            } else {
                Text("generating...  |  \(rss)")
            }
        case .error(let m):
            Text("error: \(m)").foregroundStyle(.red)
        }
    }

}

// ---------------------------------------------------------------------------
// Resident memory (`phys_footprint`) for the status line. Was
// app/utils/Memory.swift; folded in here as fileprivate since the
// status line is the only consumer.
// ---------------------------------------------------------------------------

import Darwin.Mach

fileprivate func residentBytes() -> UInt64 {
    var info  = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info>.size /
        MemoryLayout<integer_t>.size
    )
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { p in
        p.withMemoryRebound(to: integer_t.self,
                            capacity: Int(count)) { iptr in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO),
                      iptr, &count)
        }
    }
    return kerr == KERN_SUCCESS ? info.phys_footprint : 0
}

fileprivate func formatMB(_ bytes: UInt64) -> String {
    String(format: "%.0f MB", Double(bytes) / 1024.0 / 1024.0)
}
