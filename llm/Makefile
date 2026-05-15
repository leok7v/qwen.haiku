# Makefile - standalone CLI build of the llm runner.
#
# Out-of-source: all artifacts go under ../Build/cli/ relative to
# this directory (i.e. <repo>/Build/cli/), matching where the Xcode
# project also writes its outputs. The source tree under llm/ stays
# clean - nothing appears or disappears next to the C files. `make
# clean` removes only Build/cli/.
#
# Building defines LLM_CLI so llm.c emits main() and the CLI
# helpers. The Xcode build leaves LLM_CLI undefined and treats this
# directory as the library half of the bridge.

CC      = clang
CFLAGS  = -O3 -std=c17 -Wall -Wextra \
          -Wno-unused-parameter -Wno-missing-braces \
          -DLLM_CLI \
          -fvectorize -fslp-vectorize
# libcurl is always linked: tools.c (websearch / fetch / distill)
# is compiled unconditionally so the model can hit the network the
# moment it emits a <tool_call>. Future per-llm_ctx flags will
# gate runtime activation.
LDFLAGS = -lm -lcurl

ifeq ($(shell uname -m),arm64)
CFLAGS += -mcpu=apple-m1
endif

BUILD := ../Build/cli
BIN   := $(BUILD)/llm

all: $(BIN)

# Single TU rooted at llm.c, but every `#include`-d single-file
# library needs to be a prerequisite or edits won't trigger a
# rebuild (made/llm.c-only deps gave a stale binary against the
# new qwen.c path on 2026-05-15).
$(BIN): llm.c llm.h qwen.c gguf.c tokenizer.c sampler.c \
        tensor.c neon.c chunked.c \
        tools.c agent.c \
        utils/arrays.c utils/chars.c utils/maps.c utils/jinja.c \
        | $(BUILD)
	$(CC) $(CFLAGS) -o $@ llm.c $(LDFLAGS)

$(BUILD):
	mkdir -p $@

run: $(BIN)
	$(BIN) --self-test

clean:
	rm -rf $(BUILD)

.PHONY: all clean run
