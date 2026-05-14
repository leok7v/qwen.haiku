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

@main
struct QwenHaikuApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
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
    var systemPrompt: String       = "HARD RULE: all your answers are haiku."
    var messages:     [ChatMessage] = []

    // Throughput from the most recent completed generation.
    // Zero before the first one finishes.
    var lastPP:        Double = 0
    var lastTG:        Double = 0
    var lastNPrefill:  Int    = 0
    var lastNGen:      Int    = 0

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
            // Chat sampling: Qwen3.5 non-thinking default is T=1.0,
            // top_k=20. top_p / min_p / penalties land with task #21.
            let opts = Qwen.Options(temperature: 1.0, topK: 20, maxNew: 512)
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
    func send(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if qwen != nil, state == .ready, !trimmed.isEmpty {
            self.messages.append(ChatMessage(role: .user, content: trimmed))
            self.messages.append(ChatMessage(role: .assistant, content: ""))
            let assistantIdx = self.messages.count - 1
            var framed = Array(self.messages.dropLast())
            if !systemPrompt.isEmpty {
                let sys = ChatMessage(role: .system, content: systemPrompt)
                framed.insert(sys, at: 0)
            }
            let prompt = ChatTemplate.apply(messages: framed)
            self.state         = .generating
            self.stopRequested = false
            await self.streamAssistant(prompt: prompt, at: assistantIdx)
        }
    }

    /// Detached generation; appends streamed pieces (filtered for
    /// turn markers) into `messages[idx].content` on the main actor.
    private func streamAssistant(prompt: String, at idx: Int) async {
        if let qwen = self.qwen {
            await Task.detached(priority: .userInitiated) { [weak self] in
                var filter = ChatStreamFilter()
                do {
                    try qwen.generate(prompt: prompt) { piece in
                        let visible = filter.push(piece)
                        if !visible.isEmpty {
                            Task { @MainActor in
                                self?.appendAssistant(visible, at: idx)
                            }
                        }
                        if filter.done { return false }
                        return !(self?.stopRequested ?? true)
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

    /// Clear the conversation; system prompt is kept.
    func clearChat() { self.messages.removeAll() }

}

struct ContentView: View {

    @State private var vm    = QwenViewModel()
    @State private var input: String = ""
    @State private var showSystem    = false

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
            messagesView
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
            Text("generating...  |  \(rss)")
        case .error(let m):
            Text("error: \(m)").foregroundStyle(.red)
        }
    }

}
