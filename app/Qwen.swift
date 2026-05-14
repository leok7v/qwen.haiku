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
        public var temperature: Float
        public var topK:        Int
        public var maxNew:      Int
        public var minNew:      Int
        public init(temperature: Float = 0.0, topK: Int = 40,
                    maxNew: Int = 256, minNew: Int = 0) {
            self.temperature = temperature
            self.topK        = topK
            self.maxNew      = maxNew
            self.minNew      = minNew
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
        _ = ids.withUnsafeBufferPointer { buf -> Int32 in
            return Int32(llm_generate(
                ctx,
                buf.baseAddress, n,
                Int32(options.maxNew),
                Int32(options.minNew),
                options.temperature,
                Int32(options.topK),
                qwenTokenTrampoline,
                UnsafeMutableRawPointer(boxRef.toOpaque())))
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

private func qwenTokenTrampoline(utf8: UnsafePointer<CChar>?,
                                 user: UnsafeMutableRawPointer?) -> Int32 {
    var rc: Int32 = 1
    if let utf8 = utf8, let user = user {
        let box = Unmanaged<ContinuationBox>.fromOpaque(user)
                       .takeUnretainedValue()
        let piece = String(cString: utf8)
        rc = box.onToken(piece) ? 0 : 1
    }
    return rc
}
