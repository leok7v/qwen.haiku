import Foundation
import os.log

nonisolated private let logger = Logger()

nonisolated func trace(_ items: Any...,
                       separator: String = " ",
                       terminator: String = "\n",
                       _ file: StaticString = #file,
                       line: Int = #line,
                       _ function: StaticString = #function) {
    let name = (String(describing: file) as NSString).lastPathComponent
    let fun  = String(describing: function).components(separatedBy: "(").first!
    if !Trace.muted(name, fun) {
        let body = items.map { e in String(describing: e) }
                   .joined(separator: separator)
        let text = "\(name):\(line) @\(fun) \(body)"
        print(text, terminator: terminator)
        logger.info("\(text, privacy: .public)")
        Trace.t.add(text)
    }
}

nonisolated final class Trace: @unchecked Sendable {

    static let t = Trace()

    private var lock = os_unfair_lock()
    private var _muted: Set<String> = []
    private var _only:  Set<String> = []
    private var _history: [String] = []

    static func muted(_ name: String, _ fun: String) -> Bool {
        t.locked {
            t._muted.contains(name) || t._muted.contains(fun) ||
            (!t._only.isEmpty &&
             !t._only.contains(name) && !t._only.contains(fun))
        }
    }

    static func mute(_ s: String)   { _ = t.locked { t._muted.insert(s) } }
    static func unmute(_ s: String) { _ = t.locked { t._muted.remove(s) } }
    static func only(_ s: String)   { _ = t.locked { t._only.insert(s)  } }
    static func all()               { t.locked { t._only.removeAll() } }

    private func locked<T>(_ body: () -> T) -> T {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return body()
    }

    func add(_ text: String) {
        locked {
            _history.append(text)
            if _history.count > 100 { _history.removeFirst() }
        }
    }

    static func getHistory() -> String {
        t.locked { t._history.joined(separator: "\n") }
    }

}
