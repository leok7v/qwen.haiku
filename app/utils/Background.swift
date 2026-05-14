// SPDX-License-Identifier: Apache-2.0
//
// Background.swift - holder for the system-provided completion handler
// that AppDelegate.application(_:handleEventsForBackgroundURLSession:
// completionHandler:) hands off to URL.swift's URLSessionDelegate.
//
// App targets reassign this on startup. The no-op default lets the
// CLI build link without an AppDelegate.

import Foundation

nonisolated(unsafe) public var backgroundSessionCompletionHandler: () -> Void = {}
