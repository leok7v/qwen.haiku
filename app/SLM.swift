// SPDX-License-Identifier: Apache-2.0
//
// SLM.swift - Swift bridge to the slm_* C runner.
//
// Calls slm_* C functions directly through the Xcode bridging header
// (app/bridge.h -> llm/slm.h). Works on iOS as well as macOS, no
// spawned binary, no IPC.
//
// Lifecycle parallels the C side (canonical Qwen3 framing):
//   1. SLM(modelPath:) — open the GGUF.
//   2. slm.ctrl.tools / slm.ctrl.think — flip BEFORE step 3.
//   3. try slm.prefillSystem(text:) — sync prefill of system block.
//   4. try slm.generate(prompt:) — repeat for each user turn.
//
// One `SLM` owns one `slm_ctx`. The ctx mutates on every forward
// pass, so an SLM is NOT thread-safe — create one per concurrent
// stream. Tools/think become read-only after prefillSystem returns.

import Foundation

public enum SLMError: Error, CustomStringConvertible {

    case loadFailed(String)
    case ctxCreateFailed
    case systemPromptFailed
    case generationFailed(String)

    public var description: String {
        switch self {
        case .loadFailed(let s):       return "slm load failed: \(s)"
        case .ctxCreateFailed:         return "slm_ctx_create returned NULL"
        case .systemPromptFailed:      return "slm_ctx_system_prompt aborted"
        case .generationFailed(let s): return "slm generation failed: \(s)"
        }
    }

}

// `@unchecked Sendable`: the C ctx is not concurrency-safe within a
// single SLM instance (every forward pass mutates the KV cache and
// SSM state), but the object itself can be MOVED across isolation
// boundaries — bootstrap loads on a detached task and hands the SLM
// back to MainActor. Callers must not share one SLM across concurrent
// generation streams.
public final class SLM: @unchecked Sendable {

    /// Sampling parameters (passed to every slm_generate call).
    public struct Sampler {
        public var temperature:       Float
        public var topK:              Int
        public var topP:              Float
        public var minP:              Float
        public var repetitionPenalty: Float
        public var repetitionWindow:  Int
        public var maxNew:            Int
        public var minNew:            Int
        public var seed:              UInt64
        public init(temperature:       Float  = 0.7,
                    topK:              Int    = 40,
                    topP:              Float  = 0.9,
                    minP:              Float  = 0.05,
                    repetitionPenalty: Float  = 1.25,
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

    /// One streamed piece from slm_generate / slm_ctx_system_prompt.
    ///   .content     — visible reply text. Display in chat bubble.
    ///   .reasoning   — text inside <think>...</think>. Muted.
    ///   .toolCall    — debug-only "tool: …" trace.
    ///   .toolResponse — debug-only "result: …" trace.
    ///   .prefilled   — prompt prefill finished; decode begins.
    ///   .pp(pos,total) — per-token prefill progress.
    public enum Chunk {
        case content(String)
        case reasoning(String)
        case toolCall(String)
        case toolResponse(String)
        case prefilled
        case pp(Int, Int)
    }

    public let modelPath: URL
    public var sampler:   Sampler

    // `struct slm_model;` and `struct slm_ctx;` in slm.h are forward-
    // declared (opaque), so the Clang importer maps the pointers to
    // OpaquePointer here.
    private let model: OpaquePointer
    private var ctx:   OpaquePointer

    /// Load the model at `modelPath` and open a fresh ctx. Throws
    /// SLMError.loadFailed if the GGUF can't be parsed,
    /// SLMError.ctxCreateFailed if the ctx couldn't be allocated.
    public init(modelPath: URL,
                sampler:   Sampler = Sampler()) throws {
        guard let m = slm_model_load(modelPath.path) else {
            throw SLMError.loadFailed("slm_model_load returned NULL")
        }
        if !slm_model_loaded(m) {
            let msg = String(cString: slm_model_error(m))
            slm_model_unload(m)
            throw SLMError.loadFailed(msg)
        }
        guard let c = slm_ctx_create(m) else {
            slm_model_unload(m)
            throw SLMError.ctxCreateFailed
        }
        self.model     = m
        self.ctx       = c
        self.modelPath = modelPath
        self.sampler   = sampler
    }

    deinit {
        slm_ctx_destroy(ctx)
        slm_model_unload(model)
    }

    /// Direct read-write access to the ctx's ctrl. Mutate fields
    /// BEFORE prefillSystem() — tools / think get locked in.
    ///
    ///     slm.ctrl.pointee.tools = false
    ///     slm.ctrl.pointee.think = true
    public var ctrl: UnsafeMutablePointer<slm_ctrl> {
        slm_ctx_ctrl(ctx)
    }

    /// Start a new conversation: drop the current ctx, open a fresh
    /// one against the same model. After this you must call
    /// prefillSystem() again before generate().
    public func newConversation() throws {
        slm_ctx_destroy(ctx)
        guard let c = slm_ctx_create(model) else {
            throw SLMError.ctxCreateFailed
        }
        ctx = c
    }

    /// Sync prefill of the system block. Must be called once per ctx
    /// before any generate(). After it returns, ctrl.tools / .think
    /// are frozen. Throws SLMError.systemPromptFailed if cancelled.
    public func prefillSystem(text: String = "",
                              onChunk: ((Chunk) -> Bool)? = nil) throws {
        let box = ContinuationBox(onChunk: onChunk)
        let opaque = Unmanaged.passUnretained(box).toOpaque()
        var cb = slm_stream_callback(
            that:      opaque,
            content:   nil, reasoning: nil, call: nil, response: nil,
            prefilled: false,
            pp_pos:    0, pp_total: 0,
            callback:  slmStreamTrampoline)
        let ok = withUnsafeMutablePointer(to: &cb) { p -> Bool in
            text.withCString { tp in
                slm_ctx_system_prompt(ctx, tp, p)
            }
        }
        if !ok {
            throw SLMError.systemPromptFailed
        }
    }

    /// Run one user-turn chat round. Streams emissions through the
    /// callback closure; return false from the closure to abort.
    /// Returns the count of decode tokens emitted.
    @discardableResult
    public func generate(prompt:  String,
                         onChunk: @escaping (Chunk) -> Bool) -> Int {
        let box = ContinuationBox(onChunk: onChunk)
        let opaque = Unmanaged.passUnretained(box).toOpaque()
        var cb = slm_stream_callback(
            that:      opaque,
            content:   nil, reasoning: nil, call: nil, response: nil,
            prefilled: false,
            pp_pos:    0, pp_total: 0,
            callback:  slmStreamTrampoline)
        var sp = slm_sampler(
            temperature:        sampler.temperature,
            top_k:              Int32(sampler.topK),
            top_p:              sampler.topP,
            min_p:              sampler.minP,
            repetition_penalty: sampler.repetitionPenalty,
            repetition_window:  Int32(sampler.repetitionWindow))
        let n = withUnsafePointer(to: &sp) { spp -> Int32 in
            withUnsafeMutablePointer(to: &cb) { cbp -> Int32 in
                prompt.withCString { pp in
                    slm_generate(ctx, pp,
                                 Int32(sampler.maxNew),
                                 Int32(sampler.minNew),
                                 spp,
                                 sampler.seed,
                                 cbp)
                }
            }
        }
        return Int(n)
    }

    /// Blocking convenience: collect content-only into a single
    /// string. Reasoning, tool dialogue, and prefill chunks are
    /// dropped. Returns the full reply text.
    public func generate(prompt: String) -> String {
        var collected = ""
        _ = generate(prompt: prompt) { chunk in
            if case .content(let s) = chunk {
                collected += s
            }
            return true
        }
        return collected
    }

    /// Streaming content-only convenience: callback per visible piece.
    public func generate(prompt:  String,
                         onToken: @escaping (String) -> Bool) {
        _ = generate(prompt: prompt) { chunk in
            var keepGoing = true
            if case .content(let s) = chunk {
                keepGoing = onToken(s)
            }
            return keepGoing
        }
    }

    // Metadata + stats.
    public var vocabSize: Int { Int(slm_vocab_size(ctx)) }
    public var eosId:     Int { Int(slm_eos_id    (ctx)) }
    public var bosId:     Int { Int(slm_bos_id    (ctx)) }
    public var ppPerSec:  Double { slm_pp_per_sec(ctx) }
    public var tgPerSec:  Double { slm_tg_per_sec(ctx) }
    public var nPrefill:  Int    { Int(slm_n_prefill(ctx)) }
    public var nGenerated: Int   { Int(slm_n_generated(ctx)) }

    /// Number of UTF-8 message fragments accumulated on the ctx.
    public var messageCount: Int { Int(slm_ctx_messages_count(ctx)) }

    /// i-th message fragment, or nil if out of range.
    public func message(at i: Int) -> String? {
        guard let p = slm_ctx_message(ctx, Int32(i)) else { return nil }
        return String(cString: p)
    }

}

// Bridge helpers, extracted so the call sites above stay readable.

private final class ContinuationBox {
    let onChunk: ((SLM.Chunk) -> Bool)?
    init(onChunk: ((SLM.Chunk) -> Bool)?) {
        self.onChunk = onChunk
    }
}

// Trampoline for the new functor: the runtime hands us the same cb
// struct it's invoking, with that == our box.
private func slmStreamTrampoline(
    cbPtr: UnsafePointer<slm_stream_callback>?
) -> Int32 {
    var rc: Int32 = 0
    if let cbPtr = cbPtr, let opaque = cbPtr.pointee.that {
        let box = Unmanaged<ContinuationBox>.fromOpaque(opaque)
                       .takeUnretainedValue()
        let cb = cbPtr.pointee
        var keepGoing = true
        if cb.prefilled {
            keepGoing = box.onChunk?(.prefilled) ?? true
        } else if cb.pp_total > 0 {
            keepGoing = box.onChunk?(.pp(Int(cb.pp_pos),
                                         Int(cb.pp_total))) ?? true
        } else if let p = cb.content {
            keepGoing = box.onChunk?(.content(String(cString: p))) ?? true
        } else if let p = cb.reasoning {
            keepGoing = box.onChunk?(.reasoning(String(cString: p))) ?? true
        } else if let p = cb.call {
            keepGoing = box.onChunk?(.toolCall(String(cString: p))) ?? true
        } else if let p = cb.response {
            keepGoing = box.onChunk?(.toolResponse(String(cString: p))) ?? true
        }
        rc = keepGoing ? 0 : 1
    }
    return rc
}
