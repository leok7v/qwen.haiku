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
LDFLAGS = -lm

ifeq ($(shell uname -m),arm64)
CFLAGS += -mcpu=apple-m1
endif

# Opt-in: build with agent tools (websearch / fetch / distill, see
# llm/tools.c). Adds a libcurl link dep — the system libcurl on
# macOS / Linux works out of the box; `curl-config --libs` should
# print `-lcurl`. Invocation:
#   LLM_WITH_TOOLS=1 make
# Off by default so the core inference binary stays dep-free.
ifeq ($(LLM_WITH_TOOLS),1)
CFLAGS  += -DLLM_WITH_TOOLS
LDFLAGS += -lcurl
endif

BUILD := ../Build/cli
BIN   := $(BUILD)/llm

all: $(BIN)

$(BIN): llm.c tensor.c llm.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ llm.c $(LDFLAGS)

$(BUILD):
	mkdir -p $@

run: $(BIN)
	$(BIN) --self-test

clean:
	rm -rf $(BUILD)

.PHONY: all clean run
