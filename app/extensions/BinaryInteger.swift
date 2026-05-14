import Foundation

extension BinaryInteger {

    nonisolated var underscored: String {
        let digits = String(describing: magnitude)
        let chunked = digits.reversed().enumerated().map { i, d in
            i > 0 && i % 3 == 0 ? "_\(d)" : "\(d)"
        }
        let formatted = String(chunked.joined().reversed())
        return self < 0 ? "-\(formatted)" : formatted
    }

}
