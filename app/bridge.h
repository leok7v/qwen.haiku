// SPDX-License-Identifier: Apache-2.0
//
// bridge.h - Swift bridging header for the QwenHaiku Xcode target.
//
// Exposes the llm C API (implemented in llm/slm.c, which #includes
// llm/tensor.c) to Swift via the SWIFT_OBJC_BRIDGING_HEADER build
// setting. No module map needed - Xcode treats every C symbol
// declared in (or transitively included by) this header as visible
// to every Swift file in the target.

#ifndef BRIDGE_H
#define BRIDGE_H

#include "slm.h"

// trace.c is single-file (stb-style): declarations always visible,
// implementation gated by TRACE_IMPLEMENTATION (defined exactly
// once in model.c). Here we just pull in the declarations so Swift
// can see trace_subscribe / trace_observer / trace_entry.
#include "../slm/utils/trace.c"

#endif // BRIDGE_H
