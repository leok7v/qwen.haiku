// SPDX-License-Identifier: Apache-2.0
//
// ModelDownloader.swift - fetch the Qwen3.5-0.8B Q4_K_M GGUF into a
// per-app sandbox cache directory, resumable across launches.
//
// Storage location:
//   * macOS sandboxed:    ~/Library/Containers/<bundle>/Data/Library/Caches/Qwen/
//   * macOS unsandboxed:  ~/Library/Caches/Qwen/
//   * iOS:                <app sandbox>/Library/Caches/Qwen/
// We use `FileManager.urls(for: .cachesDirectory, in: .userDomainMask)`,
// same code path on both platforms; the OS picks the right container.
// `.cachesDirectory` is intentionally chosen over `.documentDirectory`
// so iCloud doesn't try to back up the ~500 MB model file and the OS
// is allowed to evict it under memory pressure (we re-download on the
// next launch).
//
// Resumability: the URL.swift extension writes a sidecar `.<name>.meta`
// file holding the expected size; subsequent calls notice the partial
// file matches that size and skip re-download. The download itself is
// a single URLSession download task, no chunking yet.

import Foundation

public struct QwenModel: Sendable {

    public let name:         String  // local filename, e.g. "Qwen3.5-0.8B-Q4_K_M.gguf"
    public let remoteURL:    URL
    public let expectedSize: Int64   // -1 if unknown (HEAD will discover)

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
            "Qwen3.5-0.8B-Q4_K_M.gguf?download=true")!,
        expectedSize: 532_517_120  // ~508 MiB
    )

    public init(name: String, remoteURL: URL, expectedSize: Int64) {
        self.name         = name
        self.remoteURL    = remoteURL
        self.expectedSize = expectedSize
    }

}

public enum ModelDownloaderError: Error, CustomStringConvertible {

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

public final class ModelDownloader {

    public typealias Progress = @Sendable (_ done: Int64, _ total: Int64) -> Bool

    public let model:  QwenModel
    public let folder: URL

    /// Initializes a downloader pointed at the per-app sandbox cache
    /// dir. Throws only if the OS won't give us a caches directory,
    /// which on iOS/macOS only happens for severely broken sandboxes.
    public init(model: QwenModel = .default) throws {
        self.model = model
        let fm = FileManager.default
        guard let base = fm.urls(for: .cachesDirectory,
                                 in: .userDomainMask).first else {
            throw ModelDownloaderError.cacheDirectoryUnavailable
        }
        let dir = base.appendingPathComponent("Qwen", isDirectory: true)
        if !dir.exist() {
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
        return u.exist() && model.expectedSize > 0 &&
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
                try await model.remoteURL.download(
                    folder, model.name, model.expectedSize, progress)
            } catch {
                throw ModelDownloaderError.downloadFailed(error)
            }
        }
        return localURL
    }

}
