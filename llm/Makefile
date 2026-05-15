# Makefile - standalone CLI build of the llm runner.
#
# Out-of-source: all artifacts go under ../Build/cli/ relative to
# this directory (i.e. <repo>/Build/cli/), matching where the Xcode
# project also writes its outputs. The source tree under llm/ stays
# clean - nothing appears or disappears next to the C files. `make
# clean` removes only Build/cli/.
#
# Building defines LLM_CLI so slm.c emits main() and the CLI
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
# Standalone qwen.c test binary: compiles qwen.c on its own (own
# header guards + #include's) with -DQWEN_TESTS so its trailing
# main() runs `qwen_self_test` / `chunked_self_test` without the
# slm.c TU.
QWEN_TEST := $(BUILD)/qwen-test

all: $(BIN) $(QWEN_TEST)

# Single TU rooted at slm.c, but every `#include`-d single-file
# library needs to be a prerequisite or edits won't trigger a
# rebuild (made/slm.c-only deps gave a stale binary against the
# new qwen.c path on 2026-05-15).
$(BIN): slm.c slm.h qwen.c gguf.c tokenizer.c sampler.c jinja.c \
        tensor.c neon.c chunked.c \
        tools.c agent.c \
        utils/arrays.c utils/chars.c utils/maps.c \
        | $(BUILD)
	$(CC) $(CFLAGS) -o $@ slm.c $(LDFLAGS)

# qwen.c standalone: not LLM_CLI (no slm.c main()), but QWEN_TESTS
# (qwen.c's own main() that runs qwen_self_test + chunked_self_test).
$(QWEN_TEST): qwen.c slm.h gguf.c tensor.c neon.c chunked.c \
              utils/arrays.c utils/chars.c utils/maps.c \
              | $(BUILD)
	$(CC) -O3 -std=c17 -Wall -Wextra \
	      -Wno-unused-parameter -Wno-missing-braces \
	      -DQWEN_TESTS \
	      -fvectorize -fslp-vectorize \
	      $(if $(filter arm64,$(shell uname -m)),-mcpu=apple-m1,) \
	      -o $@ qwen.c -lm

$(BUILD):
	mkdir -p $@

run: $(BIN)
	$(BIN) --self-test

clean:
	rm -rf $(BUILD)

.PHONY: all clean run
