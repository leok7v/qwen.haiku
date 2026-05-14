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
            out += "<|im_start|>assistant\n"
            if reasoning {
                out += "<think>\n"
            } else {
                out += "<think>\n\n</think>\n\n"
            }
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
/// emitted by `llm_generate`'s token callback, strips Qwen3 turn
/// markers, and signals `done == true` once `<|im_end|>` or
/// `<|endoftext|>` is seen so the view model can stop generation.
/// Uses a small holdback buffer (longer than the longest marker) so
/// markers that arrive split across two pieces are still caught.
public struct ChatStreamFilter {

    private static let endMarkers = ["<|im_end|>", "<|endoftext|>"]
    // Longest marker is "<|endoftext|>" = 13 bytes; round up.
    private static let holdback = 16

    public private(set) var visible: String = ""
    public private(set) var done:    Bool   = false
    private var buf:                 String = ""

    public init() {}

    /// Push the next streamed piece and return the slice that is
    /// safe to display (markers stripped, no partial marker in the
    /// trailing bytes). When `done` flips to true, the caller should
    /// cancel the generation; any subsequent `push` calls are no-ops.
    public mutating func push(_ piece: String) -> String {
        if done { return "" }
        buf += piece
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
        visible += tail
        return tail
    }

}
