// SPDX-License-Identifier: Apache-2.0
//
// Downloader.swift -- resumable single-file fetch into the per-app
// sandbox Application-Support folder. Holds everything the download
// path needs (URL helpers, sidecar .meta read/write, URLSession
// delegate) as fileprivate detail. Only `QwenModel`, `Downloader`
// and `DownloaderError` are public.
//
// `.meta` sidecar format: plain-text, one decimal Swift Int followed
// by a newline, holding the expected content length in bytes. No
// JSON, no codable. (Migrated 2026-05-15 from a JSON dict; old
// sidecars fail Int() parse and the downloader simply re-probes the
// HEAD as if they weren't there.)

import Foundation

public struct QwenModel: Sendable {

    public let name:         String  // local filename
    public let remoteURL:    URL
    public let expectedSize: Int     // -1 if unknown (HEAD will discover)

    /// Default model: unsloth's Q4_K_M GGUF of Qwen3.5-0.8B. The exact
    /// file the runner was developed and bug-fixed against. Change
    /// only after re-validating the runner on a new release.
    ///
    /// expectedSize was measured against the unsloth release as of
    /// 2026-05-13. If HuggingFace re-quantizes the file the size
    /// changes and the size-check at the end of the download fails
    /// with NSURLErrorCannotDecodeContentData (-1016). Either update
    /// this constant after re-validating, or pass -1 to force a HEAD
    /// probe on every cold start.
    public static let `default` = QwenModel(
        name: "Qwen3.5-0.8B-Q4_K_M.gguf",
        remoteURL: URL(string:
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/" +
            "Qwen3.5-0.8B-Q4_K_M.gguf")!,
        expectedSize: 532_517_120  // ~508 MiB
    )

    public init(name: String, remoteURL: URL, expectedSize: Int) {
        self.name         = name
        self.remoteURL    = remoteURL
        self.expectedSize = expectedSize
    }

}

public enum DownloaderError: Error, CustomStringConvertible {

    case cacheDirectoryUnavailable
    case downloadFailed(Error)

    public var description: String {
        switch self {
        case .cacheDirectoryUnavailable:
            return "FileManager.cachesDirectory unavailable on this platform"
        case .downloadFailed(let e):
            return "model download failed: \(e)"
        }
    }

}

public final class Downloader {

    public typealias Progress = @Sendable (_ done: Int, _ total: Int) -> Bool

    public let model:  QwenModel
    public let folder: URL

    /// Initializes a downloader pointed at the per-app sandbox
    /// Application-Support dir. Throws only if the OS won't give us
    /// one, which on iOS/macOS only happens for severely broken
    /// sandboxes.
    public init(model: QwenModel = .default) throws {
        self.model = model
        let fm = FileManager.default
        guard let base = fm.urls(for: .applicationSupportDirectory,
                                 in: .userDomainMask).first else {
            throw DownloaderError.cacheDirectoryUnavailable
        }
        let dir = base.appendingPathComponent("Qwen", isDirectory: true)
        if !dir.fileExist() {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        self.folder = dir
    }

    /// Local URL where the GGUF will be (or already is) stored.
    public var localURL: URL { folder.appendingPathComponent(model.name) }

    /// True if the model is already on disk with the expected size.
    /// Cheap check, no network.
    public var isCached: Bool {
        let u = localURL
        return u.fileExist() && model.expectedSize > 0 &&
               u.fileSize() == model.expectedSize
    }

    /// Download the model if not cached; otherwise no-op. Progress
    /// callback fires on the main actor; return true from it to
    /// cancel. The returned URL points to the on-disk GGUF.
    public func ensureAvailable(progress: @escaping Progress = { _, _ in false })
        async throws -> URL
    {
        if !isCached {
            do {
                try await downloadFile(model.remoteURL,
                                       folder, model.name,
                                       model.expectedSize, progress)
            } catch {
                throw DownloaderError.downloadFailed(error)
            }
        }
        // Mark the GGUF as not-for-backup so Time Machine and iCloud
        // Documents don't include the 508 MB blob in user backups.
        // Idempotent: the OS just rewrites the xattr each launch.
        var url = localURL
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
        return localURL
    }

}

// ---------------------------------------------------------------------------
// File-private implementation: URL helpers (file size / exists),
// HEAD probe, .meta sidecar text format, URLSession delegate.
// Was previously its own `extension URL` in app/utils/URL.swift; only
// the Downloader needed it, so it lives here as fileprivate now.
// ---------------------------------------------------------------------------

fileprivate extension URL {

    /// Local file size in bytes, or -1 if absent. Uses Swift `Int`
    /// uniformly — qwen.haiku targets 64-bit only, so Int == Int64
    /// in width and the bookkeeping is one type instead of two.
    func fileSize() -> Int {
        assert(isFileURL)
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        if let n = attrs?[.size] as? Int      { return n }
        if let n = attrs?[.size] as? Int64    { return Int(n) }
        if let n = attrs?[.size] as? NSNumber { return n.intValue }
        return -1
    }

    func fileExist() -> Bool {
        assert(isFileURL)
        return FileManager.default.fileExists(atPath: path, isDirectory: nil)
    }

    /// Async HEAD probe for the remote content length.
    func contentLength(_ timeout: TimeInterval = 30.0) async
        -> Result<Int, URLError> {
        var request = URLRequest(url: self)
        request.httpMethod = "HEAD"
        request.timeoutInterval = timeout
        var result: Result<Int, URLError> =
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

    /// Sidecar path: hidden, suffixed `.meta`, sibling to self.
    /// e.g. `.../Qwen/Q4_K_M.gguf` → `.../Qwen/.Q4_K_M.gguf.meta`.
    func metaFile() -> URL {
        let folder = self.deletingLastPathComponent()
        let fname  = self.lastPathComponent
        return folder.appendingPathComponent(".\(fname).meta")
    }

    /// Read expected content length from this file's `.meta` sidecar.
    /// Plain-text decimal Int + optional whitespace. Returns nil if
    /// the sidecar is absent or unparseable; old JSON sidecars from
    /// the previous format fail Int() parse and the caller re-probes
    /// the HEAD as if no sidecar existed (no explicit migration).
    func readMeta() -> Int? {
        var result: Int? = nil
        if let data = try? Data(contentsOf: metaFile()),
           let text = String(data: data, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            result = Int(trimmed)
        }
        return result
    }

    /// Write the expected content length to the `.meta` sidecar
    /// atomically. Plain-text: `"<size>\n"`.
    func writeMeta(_ size: Int) throws {
        let url = metaFile()
        let data = Data("\(size)\n".utf8)
        try data.write(to: url, options: .atomic)
    }

}

fileprivate func httpResult(_ http: HTTPURLResponse)
        -> Result<Int, URLError> {
    var r: Result<Int, URLError>
    switch http.statusCode {
        case 200: r = .success(Int(http.expectedContentLength))
        case 404: r = .failure(URLError(.fileDoesNotExist))
        case 403: r = .failure(URLError(.noPermissionsToReadFile))
        default:  r = .failure(URLError(.resourceUnavailable))
    }
    return r
}

fileprivate func downloaded(_ size: Int, _ stored: Int, _ known: Int) -> Bool {
    (stored > 0 && size == stored) || (known > 0 && size == known)
}

fileprivate func downloadFile(_ url: URL,
                              _ folder: URL,
                              _ fname: String,
                              _ known: Int,
                              _ progress: @escaping @Sendable (Int, Int) -> Bool)
                              async throws {
    let file = folder.appendingPathComponent(fname)
    let size = file.fileSize()
    let stored = file.readMeta() ?? -1
    if !downloaded(size, stored, known) {
        let expected = try await expectedContentLength(known, url)
        try file.writeMeta(expected)
        try await performDownload(file, expected, url, progress)
    }
}

fileprivate func expectedContentLength(_ s: Int, _ url: URL) async throws -> Int {
    var expected = s
    if expected <= 0 {
        let result = await url.contentLength()
        switch result {
            case .success(let bytes):
                if bytes > 0 {
                    expected = bytes
                } else {
                    throw URLError(.zeroByteResource)
                }
            case .failure(let error):
                throw error
        }
    }
    return expected
}

fileprivate func performDownload(_ file: URL, _ expected: Int, _ url: URL,
    _ progress: @escaping @Sendable (_: Int, _: Int) -> Bool)
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
        let delegate = DownloadDelegate(file, expected, progress, finish)
        let config = URLSessionConfiguration.default
        let session = URLSession(configuration: config, delegate: delegate,
                                 delegateQueue: nil)
        let task = session.downloadTask(with: url)
        task.resume()
    }
}

fileprivate final class OnceContinuation: @unchecked Sendable {

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

fileprivate final class DownloadDelegate: NSObject, URLSessionDownloadDelegate,
                                          @unchecked Sendable {

    let dest: URL
    let expected: Int
    let progress: @Sendable (_ done: Int, _ total: Int) -> Bool
    let completion: @Sendable (_ error: Error?) -> Void
    nonisolated(unsafe) var session: URLSession?

    init(_ dest: URL, _ expected: Int,
        _ progress: @escaping @Sendable (_: Int, _: Int) -> Bool,
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
            let writtenInt  = Int(written)
            let expectedInt = Int(expected)
            Task { @MainActor in
                let cancel = progress(writtenInt, expectedInt)
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

    // No urlSessionDidFinishEvents override: we use
    // URLSessionConfiguration.default (foreground), never
    // .background(withIdentifier:), so the system never wakes the
    // app up to finish downloads and the callback never fires.

}
