// SPDX-License-Identifier: Apache-2.0
//
// App.swift - SwiftUI entry point for QwenHaiku (iOS + macOS).
//
// Single screen: a prompt field, a streaming output area, a Run
// button. On first launch the app downloads the default GGUF
// (~504 MB) into Library/Caches/Qwen/. Subsequent launches reuse
// the cached file.
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

    var state:  State  = .idle
    var output: String = ""

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
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try Qwen(modelPath: path)
                }.value
                self.qwen  = loaded
                self.state = .ready
            } catch {
                self.state = .error(String(describing: error))
            }
        }
    }

    func generate(prompt: String) async {
        if let qwen = self.qwen, state == .ready {
            self.output        = prompt
            self.state         = .generating
            self.stopRequested = false
            await Task.detached(priority: .userInitiated) { [weak self] in
                do {
                    try qwen.generate(prompt: prompt) { piece in
                        Task { @MainActor in self?.output += piece }
                        // Return false from the C callback to abort.
                        // The C runner reads the result and stops the
                        // next forward pass.
                        return !(self?.stopRequested ?? true)
                    }
                    let pp  = qwen.ppPerSec
                    let tg  = qwen.tgPerSec
                    let np  = qwen.nPrefill
                    let ng  = qwen.nGenerated
                    Task { @MainActor in
                        self?.lastPP        = pp
                        self?.lastTG        = tg
                        self?.lastNPrefill  = np
                        self?.lastNGen      = ng
                        self?.state         = .ready
                    }
                } catch {
                    Task { @MainActor in
                        self?.state = .error(String(describing: error))
                    }
                }
            }.value
        }
    }

    /// User pressed Stop while a generation was in flight. The
    /// running detached task will see this flag in its next token
    /// callback and tell llm_generate to abort.
    func stopGeneration() { self.stopRequested = true }

}

struct ContentView: View {

    @State private var vm = QwenViewModel()

    @State private var prompt: String =
        "One word answer: Which city is the capital of France?"

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            statusView
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
            if case .needsDownload(let size) = vm.state {
                downloadPrompt(size: size)
            }
            TextField("Prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2 ... 5)
                .disabled(vm.state == .generating)
            HStack {
                if vm.state == .generating {
                    Button("Stop") { vm.stopGeneration() }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                } else {
                    Button("Run") {
                        Task { await vm.generate(prompt: prompt) }
                    }
                    .disabled(vm.state != .ready || prompt.isEmpty)
                }
                Spacer()
                Button("Clear") { vm.output = "" }
                    .disabled(vm.output.isEmpty || vm.state == .generating)
            }
            ScrollView {
                Text(vm.output.isEmpty ? "-" : vm.output)
                    .font(.system(.body, design: .serif))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .background(.background.secondary)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding(16)
        .task { vm.checkCache() }
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
            Text("starting…")
        case .needsDownload:
            Text("model not downloaded")
        case .downloading(let done, let total):
            let mb      = Double(done)  / 1_048_576.0
            let totalMb = Double(total) / 1_048_576.0
            Text(String(format: "downloading: %.1f / %.1f MB", mb, totalMb))
        case .loading:
            Text("loading weights…  |  \(rss)")
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
            Text("generating…  |  \(rss)")
        case .error(let m):
            Text("error: \(m)").foregroundStyle(.red)
        }
    }

}
