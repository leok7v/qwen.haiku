// SPDX-License-Identifier: Apache-2.0
//
// main.swift - headless Swift CLI driving the same Qwen / ChatTemplate
// code path the iOS/macOS app uses. Built from `swift-cli/Makefile`
// with a single swiftc invocation that:
//
//   1. Compiles llm/llm.c (which #include-s tensor.c + neon.c +
//      chunked.c) as a single C TU without LLM_CLI, producing the
//      library-half of the bridge.
//   2. Imports app/bridge.h so every llm_* symbol is visible in Swift.
//   3. Compiles app/Qwen.swift + app/Chat.swift (the exact files
//      Xcode builds for the GUI app) alongside this main.swift.
//
// Net effect: this binary exercises the same Swift bridge + chat
// template state machine the iOS/macOS app uses, but headlessly so
// it can be run from the terminal and from CI.
//
// Modes:
//   qwen-swift-cli single "<prompt>"           one-shot completion
//   qwen-swift-cli chat-test                   three-turn parity test
//                                              (mirrors --chat-test
//                                              in the C CLI; hashes
//                                              should match the C
//                                              hashes for the same
//                                              seed + preset)

import Foundation

let modelPath: URL = {
    let env = ProcessInfo.processInfo.environment["QWEN_GGUF"]
        ?? "/Users/leo/Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf"
    return URL(fileURLWithPath: env)
}()

// im.ai conversational preset (mirrors llm_sampler_defaults() in llm.c).
// Kept in sync by hand for now; if these ever drift, --chat-test
// hashes will diverge between this binary and the C binary.
let imAi = Qwen.Options(temperature:       0.7,
                        topK:              40,
                        topP:              0.9,
                        minP:              0.05,
                        repetitionPenalty: 1.25,
                        repetitionWindow:  64,
                        maxNew:            30,
                        minNew:            0,
                        seed:              42)

func runSingle(prompt: String) -> Int32 {
    var rc: Int32 = 0
    do {
        let q = try Qwen(modelPath: modelPath, options: imAi)
        let framed = ChatTemplate.applyDelta(userMessage: prompt)
        var filter = ChatStreamFilter()
        try q.generate(prompt: framed) { piece in
            let visible = filter.push(piece)
            if !visible.isEmpty {
                FileHandle.standardOutput.write(Data(visible.utf8))
            }
            return !filter.done
        }
        let tail = filter.finish()
        if !tail.isEmpty {
            FileHandle.standardOutput.write(Data(tail.utf8))
        }
        FileHandle.standardOutput.write(Data("\n".utf8))
        FileHandle.standardError.write(Data(
            "pp: \(q.ppPerSec) tok/s tg: \(q.tgPerSec) tok/s\n".utf8))
    } catch {
        FileHandle.standardError.write(Data(
            "qwen-swift-cli: \(error)\n".utf8))
        rc = 1
    }
    return rc
}

// Same three turns as run_chat_test in llm.c. Hashes are FNV-1a 64
// over the visible (post-filter) UTF-8 stream of each turn. If the
// Swift filter matches the C output byte-for-byte, hashes equal
// the C side's at the same seed+preset.
let chatTestTurns: [String] = [
    "Hi! Just say hello back.",
    "What did I just ask?",
    "Thanks!"
]

func fnv1a64(_ s: String) -> UInt64 {
    var h: UInt64 = 0xcbf29ce484222325
    for b in s.utf8 {
        h ^= UInt64(b)
        h = h &* 0x100000001b3
    }
    return h
}

func runChatTest() -> Int32 {
    var rc: Int32 = 0
    do {
        let q = try Qwen(modelPath: modelPath, options: imAi)
        var passA = [UInt64](repeating: 0, count: chatTestTurns.count)
        var passB = [UInt64](repeating: 0, count: chatTestTurns.count)
        var textsA = [String](repeating: "", count: chatTestTurns.count)
        var textsB = [String](repeating: "", count: chatTestTurns.count)
        for pass in 0...1 {
            if pass == 1 { q.reset() }
            for (t, turn) in chatTestTurns.enumerated() {
                let framed = ChatTemplate.applyDelta(userMessage: turn)
                var filter = ChatStreamFilter()
                var visible = ""
                try q.generate(prompt: framed) { piece in
                    let v = filter.push(piece)
                    visible += v
                    return !filter.done
                }
                visible += filter.finish()
                let cleaned = ChatStreamFilter.contentOnly(visible)
                let h = fnv1a64(cleaned)
                if pass == 0 {
                    passA[t]  = h
                    textsA[t] = cleaned
                } else {
                    passB[t]  = h
                    textsB[t] = cleaned
                }
                let line = "chat-test pass \(pass) turn \(t + 1):"
                    + " hash=\(String(format: "%016llx", h))\n"
                FileHandle.standardError.write(Data(line.utf8))
            }
        }
        var firstMismatch = -1
        for t in 0..<chatTestTurns.count {
            if passA[t] != passB[t] && firstMismatch < 0 {
                firstMismatch = t
            }
        }
        if firstMismatch < 0 {
            print("chat-test: PASS"
                  + " (3 turns x 2 passes, hashes match across reset)")
        } else {
            let t = firstMismatch
            print("chat-test: FAIL at turn \(t + 1)")
            print("  user:   \(chatTestTurns[t])")
            print("  passA:  \(textsA[t])")
            print("  passB:  \(textsB[t])")
            rc = 1
        }
    } catch {
        FileHandle.standardError.write(Data(
            "qwen-swift-cli: \(error)\n".utf8))
        rc = 1
    }
    return rc
}

func usage() {
    print("usage:")
    print("  qwen-swift-cli single \"<prompt>\"")
    print("  qwen-swift-cli chat-test")
    print("")
    print("env: QWEN_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"
          + " (default: ~/Library/Caches/Qwen/...)")
}

let args = CommandLine.arguments
var exitCode: Int32 = 0
if args.count < 2 {
    usage()
} else if args[1] == "single" && args.count >= 3 {
    exitCode = runSingle(prompt: args[2])
} else if args[1] == "chat-test" {
    exitCode = runChatTest()
} else {
    usage()
    exitCode = 1
}
exit(exitCode)
