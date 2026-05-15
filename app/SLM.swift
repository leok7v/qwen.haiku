// SPDX-License-Identifier: Apache-2.0
//
// SLM.swift - Swift bridge to the slm_* C runner.
//
// (Renamed from Qwen.swift on 2026-05-15 along with the C side's
// llm_* → slm_* rename. "SLM" stands for Small Language Model — the
// runner is no longer Qwen-specific in name, even though Qwen3.5 is
// still the only architecture wired up.)
//
// Calls slm_* C functions directly through the Xcode bridging header
// (app/bridge.h -> llm/slm.h), so it works on iOS as well as macOS,
// no spawned binary, no IPC.
//
// Lifecycle parallels the C side: instantiate once with a path to a
// GGUF, run any number of generations on the same instance. Each
// `SLM` owns a single `slm_ctx`, which holds the streaming KV/SSM
// state. That state mutates on every forward pass, so an `SLM` is
// NOT thread-safe; create one per concurrent stream.

import Foundation

public enum SLMError: Error, CustomStringConvertible {

    case loadFailed(String)
    case generationFailed(String)

    public var description: String {
        switch self {
        case .loadFailed(let s):       return "slm load failed: \(s)"
        case .generationFailed(let s): return "slm generation failed: \(s)"
        }
    }

}

// `@unchecked Sendable`: the C ctx is not concurrency-safe within a
// single SLM instance (every forward pass mutates the KV cache and
// SSM state), but the object itself can be MOVED across isolation
// boundaries: bootstrap loads on a detached task and hands the SLM
// back to MainActor. Callers must not share one SLM across
// concurrent generation streams.
public final class SLM: @unchecked Sendable {

    /// Sampling parameters (passed to every slm_generate call).
    /// Token-pick math only — conversation-shape knobs (tools,
    /// reasoning, debug verbosity) moved to `Ctrl` below.
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

    /// Per-conversation behavior knobs. Stored on the C ctx via
    /// `slm_set_ctrl`. The C side documents tools / think / effort as
    /// "set once at conversation start"; debug is mutable on the fly.
    public struct Ctrl {
        public var tools:  Bool      // enable embedded agent dispatch
        public var think:  Bool      // open <think> in gen prompt
        public var effort: String    // "low" / "medium" / "high"
        public var debug:  Int32     // 0 quiet, 9 chatty
        public init(tools:  Bool   = true,
                    think:  Bool   = false,
                    effort: String = "medium",
                    debug:  Int32  = 1) {
            self.tools  = tools
            self.think  = think
            self.effort = effort
            self.debug  = debug
        }
    }

    /// One streamed piece from slm_generate. Each case corresponds to
    /// exactly one signal the C runtime can emit:
    ///   .content        — visible chat bubble
    ///   .reasoning      — muted "thinking" panel (when shown)
    ///   .toolCall       — debug-only "tool: …" trace line
    ///   .toolResponse   — debug-only "result: …" trace line
    ///   .prefilled      — prompt prefill finished; decode begins
    public enum Chunk {
        case content(String)
        case reasoning(String)
        case toolCall(String)
        case toolResponse(String)
        case prefilled
    }

    public let modelPath: URL
    // Sampler is per-call — mutate freely between generate() calls.
    public var sampler:   Sampler
    // Ctrl is per-conversation — set once, mostly. Changing tools /
    // think / effort mid-conversation works mechanically but mixes
    // history; flip debug freely.
    public var ctrl:      Ctrl { didSet { syncCtrl() } }

    // `struct slm_model;` and `struct slm_ctx;` in slm.h are forward-
    // declared (opaque), so the Clang importer maps the pointers to
    // OpaquePointer here. No wrapping, no UnsafeMutablePointer needed.
    // The model is loaded once; the ctx is replaced on
    // `newConversation()` (the new-API replacement for the old reset).
    private let model: OpaquePointer
    private var ctx:   OpaquePointer
    private var systemPromptCopy: String?  // remembered for new ctxs

    /// Load the model at `modelPath` and open a first ctx. Throws
    /// SLMError.loadFailed if the GGUF can't be parsed.
    public init(modelPath:     URL,
                systemPrompt:  String? = nil,
                sampler:       Sampler = Sampler(),
                ctrl:          Ctrl    = Ctrl()) throws {
        guard let m = slm_model_load(modelPath.path) else {
            throw SLMError.loadFailed("slm_model_load returned NULL")
        }
        if !slm_model_loaded(m) {
            let msg = String(cString: slm_model_error(m))
            slm_model_unload(m)
            throw SLMError.loadFailed(msg)
        }
        let initialCtx = SLM.makeCtx(model:        m,
                                     systemPrompt: systemPrompt,
                                     ctrl:         ctrl)
        guard let c = initialCtx else {
            slm_model_unload(m)
            throw SLMError.loadFailed("slm_ctx_create returned NULL")
        }
        self.model            = m
        self.ctx              = c
        self.modelPath        = modelPath
        self.sampler          = sampler
        self.ctrl             = ctrl
        self.systemPromptCopy = systemPrompt
    }

    deinit {
        slm_ctx_destroy(ctx)
        slm_model_unload(model)
    }

    /// Push the current Swift-side ctrl values down to the C ctx so
    /// the next slm_generate sees them. Called automatically from
    /// the `ctrl` didSet observer; callers can also invoke directly
    /// after mutating ctrl in place via, e.g., `slm.ctrl.debug = 9`.
    public func syncCtrl() {
        ctrl.effort.withCString { effortPtr in
            var c = slm_ctrl(tools:  ctrl.tools,
                             think:  ctrl.think,
                             effort: effortPtr,
                             debug:  ctrl.debug)
            slm_set_ctrl(ctx, &c)
        }
    }

    /// Start a fresh conversation: destroy the current ctx and open
    /// a new one against the same model. Replaces the old `reset()`
    /// — but explicit about WHAT it clears (everything stored on the
    /// ctx; the model and the loaded weights survive). `systemPrompt
    /// = nil` reuses whatever system prompt was passed at init time.
    public func newConversation(systemPrompt: String? = nil) {
        slm_ctx_destroy(ctx)
        let nextSys = systemPrompt ?? systemPromptCopy
        guard let c = SLM.makeCtx(model:        model,
                                  systemPrompt: nextSys,
                                  ctrl:         ctrl) else {
            // ctx_create only fails on unloaded model; the model was
            // already validated at init time, so this branch is
            // unreachable. Trap to surface the bug if it ever fires.
            fatalError("slm_ctx_create returned NULL on loaded model")
        }
        ctx               = c
        systemPromptCopy  = nextSys
        syncCtrl()
    }

    /// Build a fresh ctx; helper for init + newConversation so the
    /// ctrl-bridge dance lives in one place.
    private static func makeCtx(model:        OpaquePointer,
                                systemPrompt: String?,
                                ctrl:         Ctrl) -> OpaquePointer? {
        ctrl.effort.withCString { effortPtr -> OpaquePointer? in
            var c = slm_ctrl(tools:  ctrl.tools,
                             think:  ctrl.think,
                             effort: effortPtr,
                             debug:  ctrl.debug)
            if let sys = systemPrompt {
                return sys.withCString { sysPtr in
                    slm_ctx_create(model, sysPtr, &c)
                }
            }
            return slm_ctx_create(model, nil, &c)
        }
    }

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
            case .reasoning, .toolCall, .toolResponse, .prefilled:
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
        // slm_tokenize allocates the id array (worst-case sized);
        // we free it on the way out. No more hard-coded 4096 cap.
        var idsPtr: UnsafeMutablePointer<Int32>? = nil
        let n = Int32(slm_tokenize(ctx, prompt, &idsPtr))
        defer { free(idsPtr) }
        guard let ids = idsPtr, n >= 0 else {
            throw SLMError.generationFailed("tokenize failed")
        }
        // Bridge the Swift closure across the C callback boundary
        // through an Unmanaged box so the callback `user` pointer
        // stays alive for the duration of slm_generate().
        let box = ContinuationBox(onChunk: onChunk)
        let boxRef = Unmanaged.passRetained(box)
        defer { boxRef.release() }
        var s = slm_sampler(
            temperature:        sampler.temperature,
            top_k:              Int32(sampler.topK),
            top_p:              sampler.topP,
            min_p:              sampler.minP,
            repetition_penalty: sampler.repetitionPenalty,
            repetition_window:  Int32(sampler.repetitionWindow))
        _ = withUnsafePointer(to: &s) { sp -> Int32 in
            Int32(slm_generate(
                ctx,
                ids, n,
                Int32(sampler.maxNew),
                Int32(sampler.minNew),
                sp,
                sampler.seed,
                slmChunkTrampoline,
                UnsafeMutableRawPointer(boxRef.toOpaque())))
        }
    }

    // Metadata.
    public var vocabSize: Int { Int(slm_vocab_size(ctx)) }
    public var eosId:     Int { Int(slm_eos_id    (ctx)) }
    public var bosId:     Int { Int(slm_bos_id    (ctx)) }

    // Throughput stats from the most recent generate() call. Zero
    // before the first call completes.
    public var ppPerSec:   Double { slm_pp_per_sec(ctx) }
    public var tgPerSec:   Double { slm_tg_per_sec(ctx) }
    public var nPrefill:   Int    { Int(slm_n_prefill(ctx)) }
    public var nGenerated: Int    { Int(slm_n_generated(ctx)) }

}

// Bridge helpers, extracted so the call-site stays readable.

private final class ContinuationBox {
    let onChunk: (SLM.Chunk) -> Bool
    init(onChunk: @escaping (SLM.Chunk) -> Bool) {
        self.onChunk = onChunk
    }
}

private func slmChunkTrampoline(
    chunk: UnsafePointer<slm_stream_chunk>?,
    user: UnsafeMutableRawPointer?
) -> Int32 {
    var rc: Int32 = 1
    if let chunk = chunk, let user = user {
        let box = Unmanaged<ContinuationBox>.fromOpaque(user)
                       .takeUnretainedValue()
        let c = chunk.pointee
        var keepGoing = true
        // Exactly one signal is set per emission per slm.h contract.
        // The `prefilled` bool is checked first because the four
        // string fields are NULL on that emission.
        if c.prefilled {
            keepGoing = box.onChunk(.prefilled)
        } else if let p = c.content {
            keepGoing = box.onChunk(.content(String(cString: p)))
        } else if let p = c.reasoning {
            keepGoing = box.onChunk(.reasoning(String(cString: p)))
        } else if let p = c.call {
            keepGoing = box.onChunk(.toolCall(String(cString: p)))
        } else if let p = c.response {
            keepGoing = box.onChunk(.toolResponse(String(cString: p)))
        }
        rc = keepGoing ? 0 : 1
    }
    return rc
}
