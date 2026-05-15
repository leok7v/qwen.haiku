// SPDX-License-Identifier: Apache-2.0
//
// App.swift - SwiftUI entry point for QwenHaiku (iOS + macOS).
//
// Chat surface over the C runner: a system-prompt header, a
// scrollable message list, and an input bar. The view model frames
// turns with `ChatTemplate.apply`, streams the assistant's reply
// through `ChatStreamFilter`, and commits each completed turn back
// to the message list so the next send carries the full history.
//
// On first launch the app downloads the default GGUF (~504 MB) into
// Library/Caches/Qwen/. Subsequent launches reuse the cached file.
//
// State machine:
//   .idle                  -> not loaded yet, no model on disk
//   .needsDownload(size)   -> cache empty, awaiting user OK
//   .downloading(done,tot) -> fetching the GGUF
//   .loading               -> model on disk, mmap+parse running
//   .ready                 -> Qwen ctx open, ready to generate
//   .generating            -> forward pass running, output streaming
//   .error(msg)            -> load/download/generation failed
//
// The model load and generation both run off-main; UI state lives on
// @MainActor.

import SwiftUI
import Observation
#if os(macOS)
import AppKit
#endif

// macOS single-window behavior: the default SwiftUI App lifecycle
// on macOS leaves the dock icon running after the last window
// closes (the "open recent" Mac convention). For QwenHaiku's
// single-purpose chat surface that's confusing — the model stays
// resident, eating ~500 MB of RSS, with no UI to interact with it.
// AppDelegate's applicationShouldTerminateAfterLastWindowClosed
// flips that to "terminate when the window closes", and a
// CommandGroup .newItem suppression in the body() removes the
// File -> New Window menu item so the user can't open a second
// chat that would share Qwen ctx state. iOS is untouched (no
// equivalent concept).
#if os(macOS)
final class QwenAppDelegate: NSObject, NSApplicationDelegate {

    func applicationShouldTerminateAfterLastWindowClosed(
        _ sender: NSApplication
    ) -> Bool {
        return true
    }

}
#endif

@main
struct QwenHaikuApp: App {

    #if os(macOS)
    @NSApplicationDelegateAdaptor(QwenAppDelegate.self)
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
final class QwenViewModel {

    enum State: Equatable {
        case idle                                    // initial, nothing checked yet
        case needsDownload(size: Int64)              // cache empty, awaiting user OK
        case downloading(done: Int64, total: Int64)  // user pressed Download
        case loading                                 // model on disk, mmap+parse
        case ready                                   // ctx alive, can generate
        case generating
        case error(String)
    }

    var state:        State        = .idle
    var systemPrompt: String       =
        "You are a helpful assistant.\n"
    var messages:     [ChatMessage] = []
    // Tools toggle drives both (a) the first-turn frame: when off,
    // the system block emitted by `ChatTemplate.applyDelta` has no
    // `# Tools` advertisement, so the model doesn't know any tools
    // exist; and (b) the sampler's tools flag — the embedded agent
    // loop in llm_generate ignores `<tool_call>` markers when off.
    // Default true. The setting only takes effect on the FIRST turn
    // of a conversation; once the system block is committed to KV
    // the model "remembers" whichever choice was in force. Toggle
    // before sending the first message, or Clear to start fresh.
    var tools:        Bool          = true
    // Debug toggle drives both (a) the sampler's debug flag — i.e.
    // whether llm_generate emits chunk->tool_call / tool_response
    // visibility chunks — and (b) the UI rendering: when on, those
    // chunks land as muted bracketed lines in the assistant bubble
    // ("[tool: …]" / "[result: …]") so you can watch the agent
    // poking the network. Default true while we iterate; later this
    // becomes a settings toggle the user controls.
    var debug:        Bool          = true

    // Throughput from the most recent completed generation.
    // Zero before the first one finishes.
    var lastPP:        Double = 0
    var lastTG:        Double = 0
    var lastNPrefill:  Int    = 0
    var lastNGen:      Int    = 0

    // Live prefill progress for the in-flight turn. Updated by the
    // chunk callback; UI shows a progress bar while prefillTotal > 0
    // and prefillDone < prefillTotal. Reset to 0/0 once decode starts.
    var prefillDone:   Int = 0
    var prefillTotal:  Int = 0

    @ObservationIgnored private var qwen:       Qwen?
    @ObservationIgnored private var downloader: ModelDownloader?
    // `nonisolated(unsafe)`: read on the C-callback worker, written
    // from MainActor (stopGeneration()). Bool reads/writes are atomic
    // on aligned memory; no lock needed.
    @ObservationIgnored
    nonisolated(unsafe) private var stopRequested = false

    /// Called once on view appearance. Cheap: only checks for the
    /// cached file and decides what state to be in. Never starts a
    /// download or load on its own; that requires a user-initiated
    /// `beginDownload()` call.
    func checkCache() {
        do {
            let dl = try ModelDownloader()
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

    /// User pressed the Download button. Kicks off the ~508 MB
    /// fetch; on success, automatically transitions to loading.
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

    /// Internal: parse the cached GGUF on a background priority task
    /// (llm_create mmaps and walks ~500 MB of weights and would
    /// otherwise block the main actor for ~1 s).
    private func loadModel() async {
        if let dl = self.downloader {
            let path = dl.localURL
            // Chat sampling. im.ai's defaults on the same Qwen3.5-0.8B
            // GGUF - empirically tuned to produce clean haikus where
            // the Qwen3.5 page's T=1.0/top_p=1.0/presence=2.0 spec
            // happily loops on "/no /no /no" or hallucinates fake
            // close-tags like `</thrank>`. Notable departures from the
            // page: temperature 0.7 (less randomness), repPenalty
            // 1.25 (more aggressive than llama.cpp's 1.05 default),
            // minP 0.05 (drops the long-tail bad-token cliff the
            // 0.8B model otherwise samples from). minNew=8 suppresses
            // eos / `<|im_end|>` for the first 8 sampling steps.
            let opts = Qwen.Options(temperature:       0.7,
                                    topK:              40,
                                    topP:              0.9,
                                    minP:              0.05,
                                    repetitionPenalty: 1.25,
                                    repetitionWindow:  64,
                                    tools:             self.tools,
                                    think:             false,
                                    debug:             self.debug,
                                    maxNew:            512,
                                    minNew:            8)
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try Qwen(modelPath: path, options: opts)
                }.value
                self.qwen  = loaded
                self.state = .ready
            } catch {
                self.state = .error(String(describing: error))
            }
        }
    }

    /// Send a user message: append it to the history, frame the
    /// whole conversation through `ChatTemplate.apply`, stream the
    /// assistant's reply into a placeholder message, then commit.
    /// `systemPrompt` is prepended to the FIRST user message body
    /// (no `<|im_start|>system` block). This matches im.ai's
    /// observed `formatChat` log byte-for-byte on the same GGUF:
    /// they emit no system block and put the rule inline with the
    /// first user turn. A real system block (via llama-cli --jinja
    /// -sys) is the documented Jinja path but produces worse output
    /// here than user-prefix - probably because the Qwen3.5 chat
    /// template's `if messages[0].role == 'system'` arm emits the
    /// block but the 0.8B model under our forward pass doesn't
    /// follow short system blocks well at all. Match what's
    /// demonstrably working in the wild over what the spec says.
    func send(_ text: String) async {
        let trimmedSys  = systemPrompt
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedUser = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if qwen != nil, state == .ready, !trimmedUser.isEmpty {
            let isFirstTurn = !self.messages.contains(
                where: { $0.role == .user })
            // Persistent KV: send only the DELTA each turn (the new
            // user envelope + assistant gen header). The Qwen ctx
            // holds the conversation's KV cache across calls. On the
            // first turn, the system prompt rides inline with the
            // user body (matches im.ai's framing - no system block).
            self.messages.append(
                ChatMessage(role: .user, content: trimmedUser))
            self.messages.append(
                ChatMessage(role: .assistant, content: ""))
            let assistantIdx = self.messages.count - 1
            let delta = ChatTemplate.applyDelta(
                userMessage: trimmedUser,
                systemPrefix: isFirstTurn ? trimmedSys : nil,
                tools: self.tools)
            self.state         = .generating
            self.stopRequested = false
            self.prefillDone   = 0
            self.prefillTotal  = 0
            // Sync per-turn flags onto the live Qwen ctx so the next
            // generate() reads the current toggle state. `tools`
            // also gates the embedded agent loop in llm_generate —
            // when off, `<tool_call>` markers stream as plain content.
            self.qwen?.options.debug = self.debug
            self.qwen?.options.tools = self.tools
            await self.streamAssistant(prompt: delta, at: assistantIdx)
        }
    }

    /// Detached generation; appends streamed pieces (filtered for
    /// turn markers) into `messages[idx].content` on the main actor.
    /// Listens to the full typed-chunk stream so debug tool_call /
    /// tool_response pieces can also reach the UI when the Debug
    /// toggle is on.
    private func streamAssistant(prompt: String, at idx: Int) async {
        if let qwen = self.qwen {
            let debugOn = self.debug
            await Task.detached(priority: .userInitiated) { [weak self] in
                var filter = ChatStreamFilter()
                do {
                    try qwen.generate(prompt: prompt) { chunk in
                        var keepGoing = true
                        switch chunk {
                        case .content(let s):
                            let visible = filter.push(s)
                            if !visible.isEmpty {
                                Task { @MainActor in
                                    self?.appendAssistant(visible,
                                                          at: idx)
                                }
                            }
                            if filter.done { keepGoing = false }
                        case .reasoning(_):
                            // Drop — could go to a thinking pane
                            // in a future iteration.
                            break
                        case .toolCall(let body):
                            if debugOn {
                                let line = "\n[tool: " + body + "]\n"
                                Task { @MainActor in
                                    self?.appendAssistant(line,
                                                          at: idx)
                                }
                            }
                        case .toolResponse(let body):
                            if debugOn {
                                let n = min(body.count, 400)
                                let prefix = String(body.prefix(n))
                                let line = "\n[result: "
                                         + prefix + "]\n"
                                Task { @MainActor in
                                    self?.appendAssistant(line,
                                                          at: idx)
                                }
                            }
                        case .prefill(let done, let total):
                            // Mirror to the view model so the UI bar
                            // tracks live. Clear on completion so the
                            // bar disappears once decode starts.
                            let isDone = (done >= total)
                            Task { @MainActor in
                                self?.prefillDone  = isDone ? 0 : done
                                self?.prefillTotal = isDone ? 0 : total
                            }
                        }
                        if keepGoing {
                            keepGoing = !(self?.stopRequested ?? true)
                        }
                        return keepGoing
                    }
                    let tail = filter.finish()
                    if !tail.isEmpty {
                        Task { @MainActor in
                            self?.appendAssistant(tail, at: idx)
                        }
                    }
                    let pp = qwen.ppPerSec
                    let tg = qwen.tgPerSec
                    let np = qwen.nPrefill
                    let ng = qwen.nGenerated
                    Task { @MainActor in
                        // Normalize the committed assistant content to
                        // history-clean shape (strip any mid-stream
                        // <think>...</think>) before the next turn's
                        // `ChatTemplate.apply` re-frames the history.
                        self?.cleanCommittedAssistant(at: idx)
                        self?.finishTurn(pp: pp, tg: tg, np: np, ng: ng)
                    }
                } catch {
                    Task { @MainActor in
                        self?.state = .error(String(describing: error))
                    }
                }
            }.value
        }
    }

    private func appendAssistant(_ piece: String, at idx: Int) {
        if idx < self.messages.count {
            self.messages[idx].content += piece
        }
    }

    /// Once a turn has finished streaming, scrub the committed text
    /// so that re-feeding it as history on the next turn matches the
    /// Qwen3 Jinja's `content.split('</think>')[-1].lstrip('\n')`
    /// rule. The streaming filter swallows the LEADING think block as
    /// it arrives; this catches mid-stream blocks the filter doesn't
    /// touch, so past-turn history never carries reasoning tokens.
    private func cleanCommittedAssistant(at idx: Int) {
        if idx < self.messages.count {
            let raw = self.messages[idx].content
            let cleaned = ChatStreamFilter.contentOnly(raw)
            if cleaned != raw {
                self.messages[idx].content = cleaned
            }
        }
    }

    private func finishTurn(pp: Double, tg: Double, np: Int, ng: Int) {
        self.lastPP       = pp
        self.lastTG       = tg
        self.lastNPrefill = np
        self.lastNGen     = ng
        self.state        = .ready
    }

    /// User pressed Stop while a generation was in flight. The
    /// running detached task will see this flag in its next token
    /// callback and tell llm_generate to abort.
    func stopGeneration() { self.stopRequested = true }

    /// Clear the conversation; system prompt is kept. Also resets
    /// the persistent KV / SSM state so the next `send(...)` starts
    /// a fresh prefill (matches a "new conversation" semantically).
    func clearChat() {
        self.messages.removeAll()
        self.qwen?.reset()
    }

}

struct ContentView: View {

    @State private var vm    = QwenViewModel()
    @State private var input: String = ""
    @State private var showSystem    = false

    // The first two are the showcase prompts the C-side --ask agent
    // loop uses (see llm/tools.c and llm/agent.c): they exercise
    // websearch / fetch / distill against the live web. In the
    // iOS/macOS app the tools are not yet wired through the Swift
    // bridge, so the model will answer from training data alone —
    // useful nonetheless for testing chat framing and inspecting
    // what the small model knows without web access.
    //
    // The trailing six are open-ended "deep question" prompts that
    // build on each other in multi-turn context (note "But how..." /
    // "Or how..." follow-ons that only make sense as continuations
    // of the previous reply).
    private static let suggestions: [String] = [
        "Search Internet to find what rock band from which country"
            + " recorded HelloWorld album?",
        "What is bitcoin price today?",
        "Why do some obscure ideas go viral instantly?",
        "What makes constructive criticism easy to absorb?",
        "But how can one receive criticism easily?",
        "How do we systematically overcome creative blocks?",
        "Or how can we stop our own blocks from growing?",
        "How does remote work alter long-term team culture?",
        "Or how can we build new teams that thrive online?",
        "How does rapid failure accelerate technological innovation?",
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
            // Tools toggle: when off, the first-turn frame has no
            // tool advertisement and the embedded agent loop is
            // dormant. Only takes effect at the start of a new
            // conversation (the system block is KV-cached after
            // turn 1).
            Toggle("Tools", isOn: $vm.tools)
                .toggleStyle(.checkbox)
                .font(.caption.monospaced())
                .disabled(vm.state == .generating)
            // Debug toggle: when on, agent tool_call / tool_response
            // chunks land as bracketed lines in the assistant bubble
            // so the user can watch the model's web activity.
            Toggle("Debug", isOn: $vm.debug)
                .toggleStyle(.checkbox)
                .font(.caption.monospaced())
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
            Text(m.content.isEmpty ? "..." : m.content)
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(m.content.isEmpty ? .secondary : .primary)
                .textSelection(.enabled)
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
    private func downloadPrompt(size: Int64) -> some View {
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
        let rss = Memory.formatMB(Memory.residentBytes())
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
            if vm.prefillTotal > 0 {
                let pct = Int(Double(vm.prefillDone) /
                              Double(max(vm.prefillTotal, 1)) * 100.0)
                Text(String(format: "prefill %d/%d (%d%%)  |  %@",
                            vm.prefillDone, vm.prefillTotal, pct, rss))
            } else {
                Text("generating...  |  \(rss)")
            }
        case .error(let m):
            Text("error: \(m)").foregroundStyle(.red)
        }
    }

}
