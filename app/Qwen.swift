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
        public var tools:             Bool  // enable agent dispatch
        public var think:             Bool  // enable <think> in gen prompt
        public var debug:             Bool  // surface tool_call /
                                            // tool_response chunks
        public var maxNew:            Int
        public var minNew:            Int
        public var seed:              UInt64
        public init(temperature:       Float  = 0.7,
                    topK:              Int    = 40,
                    topP:              Float  = 0.9,
                    minP:              Float  = 0.05,
                    repetitionPenalty: Float  = 1.25,
                    repetitionWindow:  Int    = 64,
                    tools:             Bool   = true,
                    think:             Bool   = false,
                    debug:             Bool   = true,
                    maxNew:            Int    = 256,
                    minNew:            Int    = 0,
                    seed:              UInt64 = 0) {
            self.temperature       = temperature
            self.topK              = topK
            self.topP              = topP
            self.minP              = minP
            self.repetitionPenalty = repetitionPenalty
            self.repetitionWindow  = repetitionWindow
            self.tools             = tools
            self.think             = think
            self.debug             = debug
            self.maxNew            = maxNew
            self.minNew            = minNew
            self.seed              = seed
        }
    }

    /// One streamed piece from llm_generate. Each case corresponds to
    /// exactly one signal the C runtime can emit:
    ///   .content        — visible chat bubble
    ///   .reasoning      — muted "thinking" panel (when shown)
    ///   .toolCall       — debug-only "tool: …" trace line
    ///   .toolResponse   — debug-only "result: …" trace line
    ///   .prefill        — prompt prefill progress (done / total
    ///                     prefilled tokens). Fires periodically
    ///                     during long prefill so the UI can show a
    ///                     progress bar. Returning false from the
    ///                     callback ABORTS prefill (same Stop
    ///                     semantics as during decode).
    public enum Chunk {
        case content(String)
        case reasoning(String)
        case toolCall(String)
        case toolResponse(String)
        case prefill(done: Int, total: Int)
    }

    public let modelPath: URL
    // `var` so callers can flip per-generate flags (debug / think /
    // tools) without rebuilding the Qwen instance. Costs nothing —
    // the next generate() call just reads the current value.
    public var options:   Options

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

    /// Streaming (content-only): call `onToken` once per generated
    /// UTF-8 piece of visible reply text. Reasoning, tool_call, and
    /// tool_response chunks are silently dropped. Convenience for
    /// callers that just want the bubble text.
    public func generate(prompt: String,
                         onToken: @escaping (String) -> Bool) throws {
        try generate(prompt: prompt) { chunk in
            var keepGoing = true
            switch chunk {
            case .content(let s):
                keepGoing = onToken(s)
            case .reasoning, .toolCall, .toolResponse, .prefill:
                break  // dropped by the content-only convenience
            }
            return keepGoing
        }
    }

    /// Streaming (typed): call `onChunk` once per emission. The
    /// chunk's case identifies the stream (content / reasoning /
    /// toolCall / toolResponse). Return false to abort generation.
    public func generate(prompt: String,
                         onChunk: @escaping (Chunk) -> Bool) throws {
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
        let box = ContinuationBox(onChunk: onChunk)
        let boxRef = Unmanaged.passRetained(box)
        defer { boxRef.release() }
        var sampler = llm_sampler(
            temperature:        options.temperature,
            top_k:              Int32(options.topK),
            top_p:              options.topP,
            min_p:              options.minP,
            repetition_penalty: options.repetitionPenalty,
            repetition_window:  Int32(options.repetitionWindow),
            tools:              options.tools,
            think:              options.think,
            debug:              options.debug)
        _ = ids.withUnsafeBufferPointer { buf -> Int32 in
            return withUnsafePointer(to: &sampler) { sp in
                Int32(llm_generate(
                    ctx,
                    buf.baseAddress, n,
                    Int32(options.maxNew),
                    Int32(options.minNew),
                    sp,
                    options.seed,
                    qwenChunkTrampoline,
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
    let onChunk: (Qwen.Chunk) -> Bool
    init(onChunk: @escaping (Qwen.Chunk) -> Bool) {
        self.onChunk = onChunk
    }
}

private func qwenChunkTrampoline(
    chunk: UnsafePointer<llm_stream_chunk>?,
    user: UnsafeMutableRawPointer?
) -> Int32 {
    var rc: Int32 = 1
    if let chunk = chunk, let user = user {
        let box = Unmanaged<ContinuationBox>.fromOpaque(user)
                       .takeUnretainedValue()
        let c = chunk.pointee
        var keepGoing = true
        // Exactly one signal is set per emission per llm.h contract.
        // Prefill progress (prefill_total > 0) is checked first
        // because the four string fields are NULL in that case.
        if c.prefill_total > 0 {
            keepGoing = box.onChunk(.prefill(done:  Int(c.prefill_done),
                                             total: Int(c.prefill_total)))
        } else if let p = c.content {
            keepGoing = box.onChunk(.content(String(cString: p)))
        } else if let p = c.reasoning {
            keepGoing = box.onChunk(.reasoning(String(cString: p)))
        } else if let p = c.tool_call {
            keepGoing = box.onChunk(.toolCall(String(cString: p)))
        } else if let p = c.tool_response {
            keepGoing = box.onChunk(.toolResponse(String(cString: p)))
        }
        rc = keepGoing ? 0 : 1
    }
    return rc
}
