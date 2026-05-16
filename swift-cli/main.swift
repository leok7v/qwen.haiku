// SPDX-License-Identifier: Apache-2.0
//
// main.swift - headless Swift CLI driving the slm_* C runner via the
// SLM.swift bridge. Same lifecycle the iOS/macOS app uses, just
// terminal-driven so CI / sweep.sh can run it.
//
//   qwen-swift-cli single "<prompt>"  one-shot completion
//   qwen-swift-cli chat-test          three-turn parity probe

import Foundation

let modelPath: URL = {
    let env = ProcessInfo.processInfo.environment["QWEN_GGUF"]
        ?? "/Users/leo/Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf"
    return URL(fileURLWithPath: env)
}()

// im.ai conversational preset (mirrors slm_sampler_defaults() in slm.c).
let imAi = SLM.Sampler(temperature:       0.7,
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
        let slm = try SLM(modelPath: modelPath, sampler: imAi)
        // The C side strips chat-frame markers; just print content.
        try slm.prefillSystem(text: "")
        slm.generate(prompt: prompt) { piece in
            FileHandle.standardOutput.write(Data(piece.utf8))
            return true
        }
        FileHandle.standardOutput.write(Data("\n".utf8))
        FileHandle.standardError.write(Data(
            "pp: \(slm.ppPerSec) tok/s tg: \(slm.tgPerSec) tok/s\n".utf8))
    } catch {
        FileHandle.standardError.write(Data(
            "qwen-swift-cli: \(error)\n".utf8))
        rc = 1
    }
    return rc
}

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
        let slm = try SLM(modelPath: modelPath, sampler: imAi)
        var passA = [UInt64](repeating: 0, count: chatTestTurns.count)
        var passB = [UInt64](repeating: 0, count: chatTestTurns.count)
        for pass in 0...1 {
            if pass == 1 { try slm.newConversation() }
            try slm.prefillSystem(text: "")
            for (t, turn) in chatTestTurns.enumerated() {
                var visible = ""
                _ = slm.generate(prompt: turn) { chunk in
                    if case .content(let s) = chunk { visible += s }
                    return true
                }
                let h = fnv1a64(visible)
                if pass == 0 { passA[t] = h } else { passB[t] = h }
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
            print("chat-test: FAIL at turn \(firstMismatch + 1)")
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
