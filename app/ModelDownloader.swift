// SPDX-License-Identifier: Apache-2.0
//
// ModelDownloader.swift - fetch the Qwen3.5-0.8B Q4_K_M GGUF into a
// per-app sandbox Application-Support directory, resumable across
// launches AND not purgeable by macOS.
//
// Storage location:
//   * macOS sandboxed:   ~/Library/Containers/<bundle>/Data/Library/
//                         Application Support/Qwen/
//   * macOS unsandboxed: ~/Library/Application Support/Qwen/
//   * iOS:               <app sandbox>/Library/Application Support/Qwen/
// We use `FileManager.urls(for: .applicationSupportDirectory, ...)`,
// same code path on both platforms; the OS picks the right container.
//
// History: this used to be `.cachesDirectory`, which IS sandboxed but
// is also explicitly eviction-eligible — macOS purges it under disk
// pressure and the user has to re-download 508 MB. The motivating
// rationale ("OS evicts; we re-download") proved hostile in practice.
// We now use `.applicationSupportDirectory` (not purgeable) AND set
// `isExcludedFromBackupKey` on the file so Time Machine / iCloud
// doesn't haul the GGUF around either. A one-shot migration moves
// any leftover GGUF out of the old Caches location on first launch
// so existing users don't re-download.
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
        guard let base = fm.urls(for: .applicationSupportDirectory,
                                 in: .userDomainMask).first else {
            throw ModelDownloaderError.cacheDirectoryUnavailable
        }
        let dir = base.appendingPathComponent("Qwen", isDirectory: true)
        if !dir.exist() {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        self.folder = dir
        ModelDownloader.migrateFromCaches(model: model,
                                          destFolder: dir,
                                          fm: fm)
    }

    /// One-shot migration: if a fully-downloaded GGUF still exists in
    /// the old `.cachesDirectory/Qwen/` location, move it into the
    /// new Application-Support folder so the user doesn't have to
    /// re-download. Silent best-effort; failures fall through and
    /// the normal download path kicks in. Does nothing once the new
    /// folder already holds the file.
    private static func migrateFromCaches(model:      QwenModel,
                                          destFolder: URL,
                                          fm:         FileManager) {
        let dest = destFolder.appendingPathComponent(model.name)
        if dest.exist() { return }
        guard let cachesBase = fm.urls(for: .cachesDirectory,
                                       in: .userDomainMask).first
        else { return }
        let src = cachesBase
            .appendingPathComponent("Qwen", isDirectory: true)
            .appendingPathComponent(model.name)
        if src.exist() && src.fileSize() == model.expectedSize {
            try? fm.moveItem(at: src, to: dest)
        }
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
