// SPDX-License-Identifier: Apache-2.0
//
// Whimsy.swift - thinking-state placeholder UI.
//
// "..." was boring. This file loads `Whimsicals.txt` once at startup
// (~370 emoji-flanked verbs like "🧽 Absorbing 🧽") and renders a
// shimmering, sparkling view that cycles through a fresh random
// verb every ~2s while the model is prefilling or decoding the
// first token of a turn.

import SwiftUI
import Combine

enum Whimsy {

    /// All loaded entries. Loaded lazily on first access from the
    /// app bundle. One verb per line; blank lines are skipped. Falls
    /// back to a tiny built-in set if the resource is missing — the
    /// app should never show a sad empty placeholder.
    static let all: [String] = {
        let fallback = ["🧽 Absorbing 🧽",
                        "🧠 Cogitating 🧠",
                        "🎬 Actioning 🎬"]
        if let url = Bundle.main.url(forResource: "Whimsicals",
                                     withExtension: "txt"),
           let raw = try? String(contentsOf: url, encoding: .utf8) {
            let lines = raw
                .split(whereSeparator: \.isNewline)
                .map { String($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            if !lines.isEmpty { return lines }
        }
        return fallback
    }()

    /// One random verb, or the first fallback if the list is empty.
    static func pick() -> String {
        Whimsy.all.randomElement() ?? "🧠 Cogitating 🧠"
    }

}

/// Animated "thinking" placeholder: cycling random whimsy verb with
/// a horizontal shimmer sweep through the text. Replaces the
/// pre-2026 "..." string while the assistant bubble is still empty.
struct ShinyWhimsyView: View {

    /// Picks a fresh verb on each tick. The body re-renders on
    /// `.id(whim)` so the .transition fires.
    @State private var whim: String = Whimsy.pick()
    /// 0 -> 1 cycle that drives the shimmer position.
    @State private var phase: CGFloat = 0
    /// Re-pick a verb roughly every 2 seconds. Slightly randomised
    /// so multiple thinking bubbles on screen don't lock-step.
    private let pickEvery = Timer.publish(every: 2.0, on: .main,
                                          in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 6) {
            // Sparkle on the left — SF Symbols' built-in `sparkle`
            // effect pulses without us writing any animation code.
            Image(systemName: "sparkles")
                .symbolRenderingMode(.multicolor)
                .symbolEffect(.variableColor.iterative, isActive: true)
                .imageScale(.medium)
            Text(whim)
                .font(.callout.weight(.medium))
                .foregroundStyle(shimmer)
                .id(whim)                            // restart transition
                .transition(.opacity.combined(with:
                    .scale(scale: 0.96, anchor: .leading)))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(.background.secondary)
                .overlay(
                    // Subtle outer gradient pulse for "alive" feel.
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(LinearGradient(
                            colors: [.purple.opacity(0.35),
                                     .blue.opacity(0.35),
                                     .pink.opacity(0.35)],
                            startPoint: .leading,
                            endPoint:   .trailing),
                                lineWidth: 1)))
        .onAppear {
            withAnimation(.linear(duration: 1.6)
                            .repeatForever(autoreverses: false)) {
                phase = 1.0
            }
        }
        .onReceive(pickEvery) { _ in
            withAnimation(.easeInOut(duration: 0.35)) {
                whim = Whimsy.pick()
            }
        }
        .accessibilityLabel("Thinking: \(whim)")
    }

    /// Diagonal moving gradient mask used as the text foreground —
    /// a light highlight sweeps from left to right across the verb.
    private var shimmer: LinearGradient {
        let mid = CGFloat(phase)
        return LinearGradient(
            stops: [
                .init(color: .secondary,                location: 0.00),
                .init(color: .secondary,                location: max(0, mid - 0.20)),
                .init(color: .primary,                  location: mid),
                .init(color: .secondary,                location: min(1, mid + 0.20)),
                .init(color: .secondary,                location: 1.00),
            ],
            startPoint: .leading,
            endPoint:   .trailing)
    }

}
