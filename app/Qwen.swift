// SPDX-License-Identifier: Apache-2.0
//
// Qwen.swift - Swift bridge to the llm C runner.
//
// Calls llm_* C functions directly through the Xcode bridging header
// (app/bridge.h -> llm/llm.h), so it works on iOS as well as macOS,
// no spawned binary, no IPC.
//
// Lifecycle parallels the C side: instantiate once with a path to a
// GGUF, run any number of generations on the same instance. Each
// `Qwen` owns a single `llm_ctx`, which holds the streaming KV/SSM
// state. That state mutates on every forward pass, so a `Qwen` is
// NOT thread-safe; create one per concurrent stream.

import Foundation

public enum QwenError: Error, CustomStringConvertible {

    case loadFailed(String)
    case generationFailed(String)

    public var description: String {
        switch self {
        case .loadFailed(let s):       return "llm load failed: \(s)"
        case .generationFailed(let s): return "llm generation failed: \(s)"
        }
    }

}

// `@unchecked Sendable`: the C ctx is not concurrency-safe within a
// single Qwen instance (every forward pass mutates the KV cache and
// SSM state), but the object itself can be MOVED across isolation
// boundaries: bootstrap loads on a detached task and hands the Qwen
// back to MainActor. Callers must not share one Qwen across
// concurrent generation streams.
public final class Qwen: @unchecked Sendable {

    public struct Options {
        public var temperature:       Float
        public var topK:              Int
        public var topP:              Float
        public var minP:              Float
        public var repetitionPenalty: Float
        public var repetitionWindow:  Int
        public var maxNew:            Int
        public var minNew:            Int
        public var seed:              UInt64
        public init(temperature:       Float  = 0.0,
                    topK:              Int    = 40,
                    topP:              Float  = 0.0,
                    minP:              Float  = 0.0,
                    repetitionPenalty: Float  = 1.0,
                    repetitionWindow:  Int    = 64,
                    maxNew:            Int    = 256,
                    minNew:            Int    = 0,
                    seed:              UInt64 = 0) {
            self.temperature       = temperature
            self.topK              = topK
            self.topP              = topP
            self.minP              = minP
            self.repetitionPenalty = repetitionPenalty
            self.repetitionWindow  = repetitionWindow
            self.maxNew            = maxNew
            self.minNew            = minNew
            self.seed              = seed
        }
    }

    public let modelPath: URL
    public let options:   Options

    // `struct llm_ctx;` in llm.h is forward-declared (opaque), so the
    // Clang importer maps `struct llm_ctx *` to OpaquePointer here.
    // No wrapping, no UnsafeMutablePointer<...> needed.
    private let ctx: OpaquePointer

    /// Load the model at `modelPath`. Throws QwenError.loadFailed if
    /// the GGUF can't be parsed.
    public init(modelPath: URL, options: Options = Options()) throws {
        guard let raw = llm_create(modelPath.path) else {
            throw QwenError.loadFailed("llm_create returned NULL")
        }
        if llm_loaded(raw) == 0 {
            let msg = String(cString: llm_get_error(raw))
            llm_destroy(raw)
            throw QwenError.loadFailed(msg)
        }
        self.ctx       = raw
        self.modelPath = modelPath
        self.options   = options
    }

    deinit { llm_destroy(ctx) }

    /// Clear the persistent KV / SSM state. Call when starting a new
    /// conversation in the chat surface; subsequent `generate(...)`
    /// calls within the same conversation should NOT reset because
    /// the C runner keeps the conversation's KV cache populated
    /// across calls (task #36 - persistent KV chat).
    public func reset() { llm_reset(ctx) }

    /// Blocking: generate a full completion for `prompt` and return
    /// it as a single string.
    public func generate(prompt: String) throws -> String {
        var collected = ""
        try generate(prompt: prompt) { piece in
            collected += piece
            return true
        }
        return collected
    }

    /// Streaming: call `onToken` once per generated UTF-8 piece.
    /// Return false from the callback to stop early. Concatenating
    /// all pieces yields the full completion text.
    public func generate(prompt: String,
                         onToken: @escaping (String) -> Bool) throws {
        let bufSize = 4096
        var ids = [Int32](repeating: 0, count: bufSize)
        let n = ids.withUnsafeMutableBufferPointer { buf -> Int32 in
            return Int32(llm_tokenize(ctx, prompt, buf.baseAddress,
                                      Int32(bufSize)))
        }
        if n < 0 {
            throw QwenError.generationFailed("tokenize overflow")
        }
        // Bridge the Swift closure across the C callback boundary
        // through an Unmanaged box so the callback `user` pointer
        // stays alive for the duration of llm_generate().
        let box = ContinuationBox(onToken: onToken)
        let boxRef = Unmanaged.passRetained(box)
        defer { boxRef.release() }
        var sampler = llm_sampler(
            temperature:        options.temperature,
            top_k:              Int32(options.topK),
            top_p:              options.topP,
            min_p:              options.minP,
            repetition_penalty: options.repetitionPenalty,
            repetition_window:  Int32(options.repetitionWindow))
        _ = ids.withUnsafeBufferPointer { buf -> Int32 in
            return withUnsafePointer(to: &sampler) { sp in
                Int32(llm_generate(
                    ctx,
                    buf.baseAddress, n,
                    Int32(options.maxNew),
                    Int32(options.minNew),
                    sp,
                    options.seed,
                    qwenTokenTrampoline,
                    UnsafeMutableRawPointer(boxRef.toOpaque())))
            }
        }
    }

    // Metadata.
    public var vocabSize: Int { Int(llm_vocab_size(ctx)) }
    public var eosId:     Int { Int(llm_eos_id    (ctx)) }
    public var bosId:     Int { Int(llm_bos_id    (ctx)) }

    // Throughput stats from the most recent generate() call. Zero
    // before the first call completes.
    public var ppPerSec:   Double { llm_pp_per_sec(ctx) }
    public var tgPerSec:   Double { llm_tg_per_sec(ctx) }
    public var nPrefill:   Int    { Int(llm_n_prefill(ctx)) }
    public var nGenerated: Int    { Int(llm_n_generated(ctx)) }

}

// Bridge helpers, extracted so the call-site stays readable.

private final class ContinuationBox {
    let onToken: (String) -> Bool
    init(onToken: @escaping (String) -> Bool) { self.onToken = onToken }
}

private func qwenTokenTrampoline(
    chunk: UnsafePointer<llm_stream_chunk>?,
    user: UnsafeMutableRawPointer?
) -> Int32 {
    var rc: Int32 = 1
    if let chunk = chunk, let user = user {
        let box = Unmanaged<ContinuationBox>.fromOpaque(user)
                       .takeUnretainedValue()
        // For the existing single-stream Swift API we only surface
        // `chunk.content`. Reasoning (chunk.reasoning) is currently
        // discarded — a future onToken-with-reasoning variant can
        // expose it directly if Chat.swift wants to render the
        // scratchpad separately.
        let c = chunk.pointee
        var rcSwift = true
        if let cPtr = c.content {
            let piece = String(cString: cPtr)
            rcSwift = box.onToken(piece)
        }
        // Reasoning piece — silently drop (still iterate so the
        // generator advances normally).
        rc = rcSwift ? 0 : 1
    }
    return rc
}
