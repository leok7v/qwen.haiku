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

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Disable AppKit's system-wide window tabbing — we don't
        // support multiple chat tabs at the OS level (each window
        // would have its own SLMViewModel, KV cache, etc., which is
        // confusing). This removes "View > Show Tab Bar" and the
        // auto-generated "+ tab" toolbar button.
        NSWindow.allowsAutomaticWindowTabbing = false
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
                .frame(minWidth: 600, minHeight: 540)
        }
        #if os(macOS)
        .defaultSize(width: 760, height: 640)
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
    // Graduated trace verbosity 0..9. 0 = silent, 1 = high-level
    // events, 3 = tool args, 5 = top hits, 7 = prompt/preamble text,
    // 9 = raw web bodies. Mutable on the fly; changes take effect on
    // the next slm_generate call.
    var debug:        Int32         = 1

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
        // Idempotent — fires once on first appearance. Subsequent
        // SwiftUI .task / .onAppear re-fires (e.g. when a tab is
        // unmounted and remounted) hit this guard and bail without
        // re-mmap'ing the GGUF or re-allocating the KV cache.
        guard state == .idle else { return }
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
                                      maxNew:            2048,
                                      minNew:            8)
            let toolsOn = self.tools
            let thinkOn = self.think
            let debugLv = self.debug
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
            let assistantID  = self.messages[assistantIdx].id
            self.state         = .generating
            self.stopRequested = false
            self.prefilling    = true
            // Debug is the only ctrl knob safe to flip mid-conversation
            // (tools/think are frozen after prefillSystem). Push it
            // down so the next generate() sees the current level.
            self.slm?.ctrl.pointee.debug = self.debug
            await self.streamAssistant(prompt: trimmedUser,
                                       at: assistantIdx,
                                       id: assistantID)
        }
    }

    private func streamAssistant(prompt: String, at idx: Int,
                                 id: UUID) async {
        if let slm = self.slm {
            let debugLv = self.debug
            let thinkOn = self.think
            await Task.detached(priority: .userInitiated) { [weak self] in
                // The C side already strips <think>...</think> /
                // <tool_call>...</tool_call> markers; chunks land in
                // the right enum case. No client-side state machine.
                //
                // Per-chunk UI updates dispatch via DispatchQueue.main
                // rather than `Task { @MainActor in ... }` — the
                // unstructured-Task path gets coalesced by Swift's
                // executor when the detached task is hot (every token
                // dispatches a new Task and the MainActor batches them,
                // producing visible "no-streaming" jumps). DispatchQueue
                // .main is FIFO + runloop-integrated, so each block
                // becomes its own SwiftUI render tick.
                //
                // Each dispatch carries the assistant message's UUID
                // so a stale block from a prior session (interrupted
                // by Stop / Clear) can't accidentally write to the new
                // assistant bubble that ends up at the same idx. The
                // appendAssistant/Reasoning guards re-check id.
                _ = slm.generate(prompt: prompt) { chunk in
                    var keepGoing = true
                    switch chunk {
                    case .content(let s):
                        DispatchQueue.main.async {
                            self?.appendAssistant(s, at: idx, id: id)
                        }
                    case .reasoning(let s):
                        if thinkOn {
                            DispatchQueue.main.async {
                                self?.appendReasoning(s, at: idx,
                                                      id: id)
                            }
                        }
                    case .toolCall(let body):
                        // tool-call surfaced from debug level 1+.
                        if debugLv >= 1 {
                            let line = "\n[tool: " + body + "]\n"
                            DispatchQueue.main.async {
                                self?.appendAssistant(line, at: idx,
                                                      id: id)
                            }
                        }
                    case .toolResponse(let body):
                        // Higher debug level -> show more of the body
                        // inline. Sized to roughly match the trace
                        // ring's per-level budget.
                        if debugLv >= 1 {
                            let cap: Int = debugLv >= 9 ? 4000
                                         : debugLv >= 7 ? 1200
                                         : debugLv >= 5 ?  400
                                         : debugLv >= 3 ?  120
                                                        :   40
                            let n = min(body.count, cap)
                            let prefix = String(body.prefix(n))
                            let line = "\n[result: " + prefix + "]\n"
                            DispatchQueue.main.async {
                                self?.appendAssistant(line, at: idx,
                                                      id: id)
                            }
                        }
                    case .prefilled:
                        DispatchQueue.main.async {
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
                DispatchQueue.main.async {
                    self?.finishTurn(pp: pp, tg: tg, np: np, ng: ng)
                }
            }.value
        }
    }

    // Per-chunk mutators carry the assistant message's UUID so stale
    // dispatch blocks from a prior session (interrupted by Stop /
    // Clear) can't accidentally land on the new assistant bubble that
    // happens to occupy the same idx after Clear. We compare against
    // the live message's id; mismatch → discard.
    private func appendAssistant(_ piece: String, at idx: Int,
                                 id: UUID) {
        if idx < self.messages.count && self.messages[idx].id == id {
            self.messages[idx].content += piece
        }
    }

    private func appendReasoning(_ piece: String, at idx: Int,
                                 id: UUID) {
        if idx < self.messages.count && self.messages[idx].id == id {
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
        // Reset transient flags before tearing down the C ctx so any
        // in-flight chunk callback (already past its abort check) sees
        // clean state. The per-chunk UUID guard catches the rest.
        self.stopRequested = false
        self.prefilling    = false
        self.messages.removeAll()
        if let slm = self.slm {
            let sysText = self.systemPrompt
            let toolsOn = self.tools
            let thinkOn = self.think
            let debugLv = self.debug
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
///
/// Single-line label with tail truncation so a "Debug" / "Tools" /
/// "Think" label never wraps to a second line on a tight layout —
/// the bug-screenshot showed "De-\nbug" and "Too-\nls" stacks when
/// the chip was squeezed by the system-prompt textfield in the same
/// HStack.
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
                .lineLimit(1)
                .truncationMode(.tail)
                .fixedSize(horizontal: true, vertical: false)
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

/// Top-level shell: TabView wrapping the chat experience and the
/// Debug pane. Both SLMViewModel and TraceLog live here so their
/// lifetime spans the whole app session. If the vm lived on
/// ChatView, switching to the Debug tab and back would unmount and
/// remount ChatView, destroy the @State vm, and trigger a fresh
/// GGUF mmap + KV-cache realloc on every tab switch. Hoisting the
/// vm one level up keeps the chat session alive across tab flips.
struct ContentView: View {

    @State private var vm       = SLMViewModel()
    @State private var traceLog = TraceLog()

    var body: some View {
        TabView {
            ChatView(vm: vm)
                .tabItem { Label("Chat", systemImage: "bubble.left") }
            DebugView(log: traceLog)
                .tabItem { Label("Debug", systemImage: "ant") }
        }
        .onAppear {
            TraceBridge.shared.attach(traceLog)
            TraceBridge.shared.subscribeOnce()
        }
    }

}

/// Chat experience. The vm is owned by ContentView and passed in
/// here as a @Bindable so any tab cycling doesn't reset its state.
struct ChatView: View {

    @Bindable var vm: SLMViewModel
    @State private var input: String = ""
    @State private var showSystem    = false

    private static let suggestions: [String] = [
//      "Search Internet to find what rock band from which country"
//          + " recorded HelloWorld album?",
//      "What is bitcoin price today?",
        "Compose a haiku.",
    ]

    var body: some View {
        VStack(spacing: 0) {
            statusView
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            Divider()
            systemRow
                .padding(.horizontal, 16)
                .padding(.top,    8)
                .padding(.bottom, 4)
            toggleRow
                .padding(.horizontal, 16)
                .padding(.top,    4)
                .padding(.bottom, 8)
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
        .toolbar {
            // Top-right of the window title bar, alongside the
            // Chat|Debug tab toggle. Same semantic the in-bar "Clear"
            // button used to have: drop messages, recreate the C
            // ctx (fresh KV + SSM state), re-prefill the system
            // block. NOT a new window.
            ToolbarItem(placement: .primaryAction) {
                Button {
                    vm.clearChat()
                } label: {
                    Label("New chat", systemImage: "plus")
                }
                .disabled(vm.messages.isEmpty || vm.state == .generating)
            }
        }
    }

    // Row 1: system-prompt label + textfield/preview + edit button.
    // Separated from the toggle row below so a long system prompt
    // doesn't squeeze the chips into ugly two-line wraps.
    @ViewBuilder
    private var systemRow: some View {
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
        }
    }

    // Row 2: tools / think chips + debug-level slider. Their own row
    // so they get a full-width budget and the labels never wrap.
    // Debug is a 10-notch (0..9) horizontal slider so the user can
    // gradually open up the trace fire-hose: 1 = events only, 9 =
    // raw HTML bodies + full distill output. step:1 snaps to integer
    // notches without needing custom tick-mark drawing.
    @ViewBuilder
    private var toggleRow: some View {
        HStack(spacing: 8) {
            ChipToggle("Tools", isOn: $vm.tools)
                .disabled(vm.state == .generating)
            ChipToggle("Think", isOn: $vm.think)
                .disabled(vm.state == .generating)
            Text("Debug")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            Slider(value: Binding(
                       get: { Double(vm.debug) },
                       set: { vm.debug = Int32($0.rounded()) }),
                   in: 0 ... 9, step: 1)
                .controlSize(.small)
                .frame(width: 160)
            Text("\(vm.debug)")
                .font(.caption.monospaced().weight(.semibold))
                .foregroundStyle(.primary)
                .frame(width: 14, alignment: .leading)
            Spacer()
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
            // "New chat" lives in the toolbar (the system-provided
            // "+" button at the top-right) — fires the .newDocument
            // command we expose below. No duplicate in-bar button.
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

// Whimsy - thinking-state placeholder UI.
//
// "..." was boring. Whimsy loads `whimsicals.txt` once at startup
// (~370 emoji-flanked verbs like "🧽 Absorbing 🧽") and renders a
// shimmering, sparkling view that cycles through a fresh random
// verb every ~2s while the model is prefilling or decoding the
// first token of a turn.

import Combine

enum Whimsy {

    /// All loaded entries. Loaded lazily on first access from the
    /// app bundle. One verb per line; blank lines are skipped. Falls
    /// back to a tiny built-in set if the resource is missing — the
    /// app should never show a sad empty placeholder.
    static let all: [String] = {
        let fallback = ["🧽 Absorbing 🧽",
                        "🧠 Cogitating 🧠",
                        "🎬 Actioning 🎬"]
        if let url = Bundle.main.url(forResource: "Whimsicals",
                                     withExtension: "txt"),
           let raw = try? String(contentsOf: url, encoding: .utf8) {
            let lines = raw
                .split(whereSeparator: \.isNewline)
                .map { String($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            if !lines.isEmpty { return lines }
        }
        return fallback
    }()

    /// One random verb, or the first fallback if the list is empty.
    static func pick() -> String {
        Whimsy.all.randomElement() ?? "🧠 Cogitating 🧠"
    }

}

/// Animated "thinking" placeholder: cycling random whimsy verb with
/// a horizontal shimmer sweep through the text. Replaces the
/// pre-2026 "..." string while the assistant bubble is still empty.
struct ShinyWhimsyView: View {

    /// Picks a fresh verb on each tick. The body re-renders on
    /// `.id(whim)` so the .transition fires.
    @State private var whim: String = Whimsy.pick()
    /// 0 -> 1 cycle that drives the shimmer position.
    @State private var phase: CGFloat = 0
    /// Re-pick a verb roughly every 2 seconds. Slightly randomised
    /// so multiple thinking bubbles on screen don't lock-step.
    private let pickEvery = Timer.publish(every: 2.0, on: .main,
                                          in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 6) {
            // Sparkle on the left — SF Symbols' built-in `sparkle`
            // effect pulses without us writing any animation code.
            Image(systemName: "sparkles")
                .symbolRenderingMode(.multicolor)
                .symbolEffect(.variableColor.iterative, isActive: true)
                .imageScale(.medium)
            Text(whim)
                .font(.callout.weight(.medium))
                .foregroundStyle(shimmer)
                .id(whim)                            // restart transition
                .transition(.opacity.combined(with:
                    .scale(scale: 0.96, anchor: .leading)))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(.background.secondary)
                .overlay(
                    // Subtle outer gradient pulse for "alive" feel.
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(LinearGradient(
                            colors: [.purple.opacity(0.35),
                                     .blue.opacity(0.35),
                                     .pink.opacity(0.35)],
                            startPoint: .leading,
                            endPoint:   .trailing),
                                lineWidth: 1)))
        .onAppear {
            withAnimation(.linear(duration: 1.6)
                            .repeatForever(autoreverses: false)) {
                phase = 1.0
            }
        }
        .onReceive(pickEvery) { _ in
            withAnimation(.easeInOut(duration: 0.35)) {
                whim = Whimsy.pick()
            }
        }
        .accessibilityLabel("Thinking: \(whim)")
    }

    /// Diagonal moving gradient mask used as the text foreground —
    /// a light highlight sweeps from left to right across the verb.
    private var shimmer: LinearGradient {
        let mid = CGFloat(phase)
        return LinearGradient(
            stops: [
                .init(color: .secondary,                location: 0.00),
                .init(color: .secondary,                location: max(0, mid - 0.20)),
                .init(color: .primary,                  location: mid),
                .init(color: .secondary,                location: min(1, mid + 0.20)),
                .init(color: .secondary,                location: 1.00),
            ],
            startPoint: .leading,
            endPoint:   .trailing)
    }

}
