import Foundation

extension URL {

    /// Returns the local file size in bytes, or -1 if the file does
    /// not exist.
    nonisolated func fileSize() -> Int64 {
        assert(isFileURL)
        let fm = FileManager.default
        let attrs = try? fm.attributesOfItem(atPath: path)
        return attrs?[.size] as? Int64 ?? -1
    }

    /// Returns true if a local file or folder exists at this URL.
    nonisolated func exist() -> Bool {
        assert(isFileURL)
        let p = self.path
        return FileManager.default.fileExists(atPath: p, isDirectory: nil)
    }

    /// Returns true if this URL is a local folder.
    nonisolated func isFolder() -> Bool {
        assert(isFileURL)
        let p = self.path
        var d = ObjCBool(false)
        return FileManager.default.fileExists(atPath: p, isDirectory: &d) &&
               d.boolValue
    }

    /// Async HEAD request to get the remote file size without
    /// downloading the body.
    nonisolated func contentLength(_ timeout: TimeInterval = 30.0) async
        -> Result<Int64, URLError> {
        var request = URLRequest(url: self)
        request.httpMethod = "HEAD"
        request.timeoutInterval = timeout
        var result: Result<Int64, URLError> =
            .failure(URLError(.badServerResponse))
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse {
                result = httpResult(http)
            }
        } catch {
            let urlError = error as? URLError ??
                URLError(.cannotConnectToHost)
            result = .failure(urlError)
        }
        return result
    }

    nonisolated func download(_ folder: URL, _ fname: String, _ known: Int64,
        _ progress: @escaping @Sendable (_: Int64, _: Int64) -> Bool) async throws {
        let file = folder.appendingPathComponent(fname)
        let size = file.fileSize()
        var meta = file.readMeta() ?? [:]
        let stored = meta["size"] as? Int64 ?? -1
        if !downloaded(size, stored, known) {
            let expected = try await expectedContentLength(known, self)
            meta["size"] = Int64(expected)
            try file.writeMeta(meta)
            try await performDownload(file, expected, self, progress)
        }
    }

    /// Sidecar path for this file: hidden, suffixed `.meta`, sibling
    /// to self. e.g. `.../Caches/Qwen/Q4_K_M.gguf`'s meta sits at
    /// `.../Caches/Qwen/.Q4_K_M.gguf.meta`.
    fileprivate func metaFile() -> URL {
        let folder = self.deletingLastPathComponent()
        let fname  = self.lastPathComponent
        return folder.appendingPathComponent(".\(fname).meta")
    }

    /// Read this file's meta sidecar. `self` is the actual file URL;
    /// the meta sidecar path is computed via `metaFile()`.
    nonisolated func readMeta() -> [String: Any]? {
        var result: [String: Any]? = nil
        if let data = try? Data(contentsOf: metaFile()),
           let wrapper = try? JSONDecoder().decode(AnyCodable.self, from: data),
           let dict = wrapper.value as? [String: Any] {
            result = dict
        }
        return result
    }

    /// Write the meta sidecar atomically.
    nonisolated func writeMeta(_ meta: [String: Any]) throws {
        let url = metaFile()
        let data = try JSONEncoder().encode(AnyCodable(meta))
        try data.write(to: url, options: .atomic)
    }

}

// download() helpers, kept file-private outside the extension to keep
// the public API small.

private nonisolated func httpResult(_ http: HTTPURLResponse)
        -> Result<Int64, URLError> {
    var r: Result<Int64, URLError>
    switch http.statusCode {
        case 200: r = .success(http.expectedContentLength)
        case 404: r = .failure(URLError(.fileDoesNotExist))
        case 403: r = .failure(URLError(.noPermissionsToReadFile))
        default:  r = .failure(URLError(.resourceUnavailable))
    }
    return r
}

private nonisolated func downloaded(_ size: Int64, _ stored: Int64,
                                    _ known: Int64) -> Bool {
    (stored > 0 && size == stored) || (known > 0 && size == known)
}

private func expectedContentLength(_ s: Int64, _ url: URL)
                                   async throws -> Int64 {
    var expected = s
    if expected <= 0 {
        let result = await url.contentLength()
        switch result {
            case .success(let bytes):
                if bytes > 0 {
                    trace("\(bytes.underscored) url: \(url)")
                    expected = bytes
                } else {
                    throw URLError(.zeroByteResource)
                }
            case .failure(let error):
                trace(url)
                throw error
        }
    }
    return expected
}

private func performDownload(_ file: URL, _ expected: Int64, _ url: URL,
    _ progress: @escaping @Sendable (_: Int64, _: Int64) -> Bool)
    async throws {
    try await withCheckedThrowingContinuation {
        (c: CheckedContinuation<Void, Error>) in
        // OnceContinuation collapses any number of delegate callbacks
        // into a single continuation resume. `nonisolated(unsafe)`
        // plus `os_unfair_lock` is the minimum-cost serialization;
        // delegate methods come in on URLSession's own queue and
        // resume() must be called exactly once.
        let box = OnceContinuation(c)
        let finish: @Sendable (Error?) -> Void = { error in
            box.fire(error)
        }
        let delegate = Delegate(file, expected, progress, finish)
        let config = URLSessionConfiguration.default
        let session = URLSession(configuration: config, delegate: delegate,
                                 delegateQueue: nil)
        let task = session.downloadTask(with: url)
        task.resume()
    }
}

private final class OnceContinuation: @unchecked Sendable {

    private let c: CheckedContinuation<Void, Error>
    private var fired = false
    private var lock  = os_unfair_lock()
    init(_ c: CheckedContinuation<Void, Error>) { self.c = c }
    func fire(_ error: Error?) {
        os_unfair_lock_lock(&lock)
        let already = fired
        fired = true
        os_unfair_lock_unlock(&lock)
        if !already {
            if let error = error { c.resume(throwing: error) }
            else                 { c.resume(returning: ()) }
        }
    }

}

private final class Delegate: NSObject, URLSessionDownloadDelegate,
                               @unchecked Sendable {

    let dest: URL
    let expected: Int64
    let progress: @Sendable (_ done: Int64, _ total: Int64) -> Bool
    let completion: @Sendable (_ error: Error?) -> Void
    nonisolated(unsafe) var session: URLSession?

    init(_ dest: URL, _ expected: Int64,
        _ progress: @escaping @Sendable (_: Int64, _: Int64) -> Bool,
        _ completion: @escaping @Sendable (_ error: Error?) -> Void) {
        self.dest = dest
        self.expected = expected
        self.progress = progress
        self.completion = completion
    }

    func urlSession(_ session: URLSession,
        downloadTask task: URLSessionDownloadTask,
        didWriteData bytes: Int64,
        totalBytesWritten written: Int64,
        totalBytesExpectedToWrite expected: Int64) {
        if expected > 0 {
            Task { @MainActor in
                let cancel = progress(written, expected)
                if cancel { task.cancel() }
            }
        }
    }

    func urlSession(_ session: URLSession,
        downloadTask task: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL) {
        var error: Error? = nil
        do {
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: location, to: dest)
            if dest.fileSize() != expected {
                error = URLError(.cannotDecodeContentData)
            } else {
                Task { @MainActor in _ = progress(expected, expected) }
            }
        } catch let e {
            error = e
        }
        cleanup(session, error)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
        didCompleteWithError error: Error?) {
        if let error = error { cleanup(session, error) }
    }

    private func cleanup(_ session: URLSession, _ error: Error?) {
        session.invalidateAndCancel()
        completion(error)
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        // Essential: tell the system the UI is updated so the app can
        // sleep again.
        DispatchQueue.main.async { backgroundSessionCompletionHandler() }
    }

}
