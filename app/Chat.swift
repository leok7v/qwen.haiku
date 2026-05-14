// SPDX-License-Identifier: Apache-2.0
//
// Chat.swift - Qwen3 chat-template state machine + streaming filter.
//
// The Qwen3 GGUF stores its chat template as a ~5 KB Jinja string
// under the KV `tokenizer.chat_template`; the C runner exposes it
// through `llm_chat_template(ctx)`. Rather than embed a Jinja parser
// we mirror the template as a small Swift state machine here, where
// the conversation model lives anyway. The structural form is the
// `<|im_start|>role\n...content...<|im_end|>\n` envelope; the file
// `docs/DESIGN.md` (section "Chat-template state machine") gives the
// algorithm. Reasoning mode is off by default - Qwen3.5-0.8B's
// documented default - and is wired to insert the empty
// `<think>\n\n</think>\n\n` block after the generation prompt so the
// model jumps straight to content.
//
// The streaming side is symmetric: `ChatStreamFilter` watches the
// token stream coming back from `llm_generate`, strips the
// `<|im_end|>` / `<|endoftext|>` marker once it appears, and signals
// end-of-turn so the view model can stop generation cleanly without
// the marker leaking into the visible bubble.

import Foundation

public struct ChatMessage: Identifiable, Equatable, Sendable {

    public enum Role: String, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    public let id:        UUID
    public var role:      Role
    public var content:   String
    public var reasoning: String?

    public init(role: Role, content: String, reasoning: String? = nil) {
        self.id        = UUID()
        self.role      = role
        self.content   = content
        self.reasoning = reasoning
    }

}

public enum ReasoningEffort: String, CaseIterable, Sendable {
    case low
    case medium
    case high
}

public enum ChatTemplate {

    /// Frame `messages` into the Qwen3 `<|im_start|>role ... <|im_end|>`
    /// envelope. The first message may be a `system` message; the rest
    /// alternate `user` and `assistant` (with optional `tool` turns
    /// framed as `<tool_response>` blocks per the template).
    /// `reasoning=false` is Qwen3.5's documented default: the
    /// generation prompt contains an empty `<think>\n\n</think>\n\n`
    /// block so the model skips its scratchpad. `reasoning=true`
    /// opens a `<think>\n` block instead so the model emits its
    /// reasoning before content. `effort` prepends a hint to the
    /// system message; `addGenerationPrompt` appends the trailing
    /// `<|im_start|>assistant\n` so `llm_generate` continues from
    /// the assistant's slot.
    public static func apply(messages: [ChatMessage],
                             reasoning: Bool = false,
                             effort: ReasoningEffort = .medium,
                             addGenerationPrompt: Bool = true) -> String {
        var out  = ""
        var rest = ArraySlice(messages)
        if let first = rest.first, first.role == .system {
            out += "<|im_start|>system\n"
            out += effortPrefix(effort) + first.content
            out += "<|im_end|>\n"
            rest = rest.dropFirst()
        }
        for m in rest {
            switch m.role {
            case .system:
                out += "<|im_start|>system\n" + m.content + "<|im_end|>\n"
            case .user:
                out += "<|im_start|>user\n" + m.content + "<|im_end|>\n"
            case .assistant:
                if reasoning, let r = m.reasoning, !r.isEmpty {
                    out += "<|im_start|>assistant\n<think>\n" + r
                    out += "\n</think>\n\n" + m.content + "<|im_end|>\n"
                } else {
                    out += "<|im_start|>assistant\n" + m.content
                    out += "<|im_end|>\n"
                }
            case .tool:
                out += "<|im_start|>user\n<tool_response>\n" + m.content
                out += "\n</tool_response><|im_end|>\n"
            }
        }
        if addGenerationPrompt {
            // Matches the GGUF's `tokenizer.chat_template` exactly:
            //   if enable_thinking: open a <think> block for the
            //     model to fill in;
            //   else: pre-fill an empty <think>\n\n</think>\n\n block
            //     so the model jumps straight to content.
            // `ChatStreamFilter` defensively strips a leading <think>
            // block on the way out (so any close tag the model still
            // emits anyway does not leak into the visible bubble or
            // contaminate the framed history on the next turn).
            out += "<|im_start|>assistant\n"
            if reasoning {
                out += "<think>\n"
            } else {
                out += "<think>\n\n</think>\n\n"
            }
        }
        return out
    }

    /// Single-turn delta for persistent-KV chat: the bytes to append
    /// to an already-warm Qwen ctx so the model sees one new user
    /// message + the assistant generation header. Pass non-nil
    /// `systemPrefix` ONLY on the first turn of a conversation; it
    /// is rendered inline with the user message (no
    /// `<|im_start|>system` block) matching im.ai's framing.
    public static func applyDelta(userMessage: String,
                                  systemPrefix: String? = nil,
                                  reasoning: Bool = false) -> String {
        var out  = "<|im_start|>user\n"
        if let sys = systemPrefix, !sys.isEmpty {
            out += sys + "\n\n"
        }
        out += userMessage
        out += "<|im_end|>\n<|im_start|>assistant\n"
        if reasoning {
            out += "<think>\n"
        } else {
            out += "<think>\n\n</think>\n\n"
        }
        return out
    }

    private static func effortPrefix(_ e: ReasoningEffort) -> String {
        switch e {
        case .low:    return "(brief: 1-2 sentences)\n\n"
        case .medium: return ""
        case .high:   return "(think carefully step-by-step before answering)\n\n"
        }
    }

}

/// Streaming-side companion to `ChatTemplate`. Accepts UTF-8 pieces
/// emitted by `llm_generate`'s token callback and produces clean
/// user-visible text:
///
/// 1. Skip an optional leading `<think>...</think>` block (Qwen3.5
///    emits one even in non-thinking mode); discard surrounding
///    whitespace so content starts at the first real character.
/// 2. Stream visible content with a 32-byte holdback so an
///    `<|im_end|>` or `<|endoftext|>` marker split across two pieces
///    is still caught. When such a marker arrives, flip `done` so
///    the view model can stop generation.
public struct ChatStreamFilter {

    private static let endMarkers = ["<|im_end|>", "<|endoftext|>"]
    // Holdback wide enough for end markers, and for "</think>" to be
    // detected before any of its bytes leak as visible content.
    private static let holdback = 32
    // Give up looking for a leading think block once this many bytes
    // of non-think content have arrived without a `<` opener.
    private static let leadingThinkScanLimit = 48

    public private(set) var visible:     String = ""
    public private(set) var done:        Bool   = false
    private var buf:                     String = ""
    private var contentStarted:          Bool   = false

    public init() {}

    /// Push the next streamed piece and return the slice that is
    /// safe to display (turn markers and any leading think block
    /// stripped, no partial marker in the trailing bytes). When
    /// `done` flips to true, the caller should cancel the
    /// generation; any subsequent `push` calls are no-ops.
    public mutating func push(_ piece: String) -> String {
        if done { return "" }
        buf += piece
        if !contentStarted {
            if let consumed = consumeLeadingThinkPrefix() {
                buf = consumed
                contentStarted = true
            } else {
                return ""
            }
        }
        for m in Self.endMarkers {
            if let r = buf.range(of: m) {
                let head = String(buf[..<r.lowerBound])
                buf  = ""
                done = true
                visible += head
                return head
            }
        }
        let emit: String
        if buf.count > Self.holdback {
            let idx = buf.index(buf.endIndex, offsetBy: -Self.holdback)
            emit = String(buf[..<idx])
            buf  = String(buf[idx...])
        } else {
            emit = ""
        }
        visible += emit
        return emit
    }

    /// Flush whatever bytes are still held back and return them. Call
    /// once when generation has stopped without hitting an end marker
    /// (e.g. `max_new` reached, user pressed Stop).
    public mutating func finish() -> String {
        let tail = buf
        buf = ""
        if !contentStarted { contentStarted = true }
        visible += tail
        return tail
    }

    /// Strip reasoning content for chat-history storage. Mirrors the
    /// Qwen3 Jinja's
    ///     content = content.split('</think>')[-1].lstrip('\n')
    /// rule: past-turn assistant messages in chat history must NOT
    /// contain `<think>`/`</think>` markers nor any reasoning text
    /// between them, only the content that comes after the LAST
    /// `</think>`. The streaming filter swallows a LEADING think
    /// block as it streams; this static helper handles the (rarer)
    /// case of a mid-stream block leaking through, plus normalizes
    /// the committed text so re-feeding history matches what
    /// `llama_jinja_template` would have produced.
    public static func contentOnly(_ s: String) -> String {
        let trim: (Substring) -> Substring = { sub in
            var out = sub
            while let c = out.first,
                  c == "\n" || c == "\r" || c == " " || c == "\t" {
                out = out.dropFirst()
            }
            return out
        }
        if let r = s.range(of: "</think>", options: .backwards) {
            return String(trim(s[r.upperBound...]))
        }
        return String(trim(Substring(s)))
    }

    /// Inspect `buf` for the once-per-turn leading think pattern:
    /// optional whitespace, optional `<think>...</think>` (or just
    /// `</think>` if our prompt opened one), then content. Returns
    /// the remaining buffer with the prefix removed once content
    /// boundary is identified, or nil if more data is needed to
    /// decide. Switches to "content started" once the boundary is
    /// crossed; subsequent pushes go through the streaming path.
    private func consumeLeadingThinkPrefix() -> String? {
        var s = Substring(buf)
        while let c = s.first, c == "\n" || c == "\r" || c == " " || c == "\t" {
            s = s.dropFirst()
        }
        if s.hasPrefix("<think>") || s.hasPrefix("</think>") {
            if let r = s.range(of: "</think>") {
                var rest = s[r.upperBound...]
                while let c = rest.first,
                      c == "\n" || c == "\r" || c == " " || c == "\t" {
                    rest = rest.dropFirst()
                }
                return String(rest)
            }
            return nil
        }
        // Looks like content but with too few bytes to be sure no
        // `<think>` opener is on the way; wait for more.
        if s.hasPrefix("<") && s.count < 8 {
            return nil
        }
        // Buf hasn't started with a think tag and has accumulated
        // enough bytes to be confident; treat as plain content.
        if buf.count >= Self.leadingThinkScanLimit || !s.isEmpty {
            return buf
        }
        return nil
    }

}
