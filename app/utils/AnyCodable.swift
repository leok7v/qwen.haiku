import Foundation

nonisolated struct AnyCodable: Codable {

    let value: Any

    init(_ value: Any) { self.value = value }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self.value = NSNull()
        } else if let v = try? c.decode(Bool.self) {
            self.value = v
        } else if let v = try? c.decode(Int64.self) {
            self.value = v
        } else if let v = try? c.decode(Int.self) {
            self.value = v
        } else if let v = try? c.decode(Double.self) {
            self.value = v
        } else if let v = try? c.decode(String.self) {
            self.value = v
        } else if let v = try? c.decode([AnyCodable].self) {
            self.value = v.map { e in e.value }
        } else if let v = try? c.decode([String: AnyCodable].self) {
            self.value = v.mapValues { v in v.value }
        } else {
            throw DecodingError.dataCorruptedError(in: c,
                                     debugDescription: "Invalid JSON")
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
            case let v as Bool:              try c.encode(v)
            case let v as Double:            try c.encode(v)
            case let v as String:            try c.encode(v)
            case let v as [Any]:
                     try c.encode(v.map { e in AnyCodable(e) })
            case let v as any BinaryInteger: try c.encode(Int64(v))
            case let v as [String: Any]:
                     try c.encode(v.mapValues { v in AnyCodable(v) })
            default:                         try c.encodeNil()
        }
    }

}
