# swift-cli Makefile - builds a headless Swift CLI binary that
# exercises the same SLM.swift code path the iOS/macOS Xcode app
# uses. Single swiftc invocation: the C side (llm/slm.c + its
# #include-d translation-unit-mates tensor.c, simd/, chunked.c) is
# compiled inline via -import-objc-header. Chat framing moved to C
# in the 2026 phase-A refactor; no more Chat.swift on either side.

SWIFTC   = swiftc
ROOT     = ..
LLM      = $(ROOT)/llm/slm.c
SWIFT_SRCS = main.swift \
             $(ROOT)/app/SLM.swift
BRIDGE   = $(ROOT)/app/bridge.h

# Mirror llm/Makefile's clang flags for the C side. -DLLM_CLI is
# DELIBERATELY omitted: that gate compiles the C runner's own
# main() + run_chat/run_repl helpers; we want only the library half
# so the Swift CLI can drive llm_create / llm_generate from main.swift.
CFLAGS = -O3 -std=c17 -Wno-unused-parameter -Wno-missing-braces \
         -fvectorize -fslp-vectorize \
         -I$(ROOT)/llm

ifeq ($(shell uname -m),arm64)
CFLAGS += -mcpu=apple-m1
endif

BUILD = ../Build/cli
BIN   = $(BUILD)/qwen-swift-cli

all: $(BIN)

$(BIN): $(SWIFT_SRCS) $(LLM) $(BRIDGE) | $(BUILD)
	$(SWIFTC) -O \
	    -import-objc-header $(BRIDGE) \
	    $(addprefix -Xcc ,$(CFLAGS)) \
	    $(SWIFT_SRCS) $(LLM) \
	    -framework Foundation \
	    -lcurl \
	    -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -f $(BIN)

run: $(BIN)
	$(BIN) chat-test

.PHONY: all clean run
