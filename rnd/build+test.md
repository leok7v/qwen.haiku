# Cross-host build + test sweep

Drives the `qwen.haiku` CLI (`llm`) and the standalone `qwen-test` on every
host in the fleet so SIMD-dispatcher changes can be validated end-to-end
on real silicon, not just the synthetic `simd_test` self-test.

Output goes to `rnd/sweep-results/<host>.out`; the sweep script lives at
[`rnd/sweep.sh`](./sweep.sh).

## Source files driving the build

| File           | Role                                                         |
|----------------|--------------------------------------------------------------|
| `simd/quants.h`| Block layouts (q4k/q5k/q6k/q8_0/q8k), fp16 helpers, scalar refs |
| `simd/neon.c`  | ARM tiers: `+dotprod` (Apple Silicon, A76+) and baseline (A53/A72/A73) |
| `simd/avx.c`   | x86 tiers: AVX-VNNI, AVX2-FMA, AVX1, F16C HGEMM              |
| `simd/simd.c`  | Dispatcher (CPUID/HWCAP) + tiled SGEMM/HGEMM + `-DSIMD_TEST` self-tests |
| `llm/*.c`      | The actual Qwen3.5 runner (slm.c TU root, includes simd via tensor.c) |
| Tests invoked  | `--self-test`, `--think-test`, `--jinja-test`, `--chunked-test`, `--agent-test`, `--chat-test`, plus `qwen-test` standalone |

`simd.c` `#include`s `neon.c` or `avx.c` based on `__aarch64__` /
`__x86_64__`; one binary per (OS, arch) covers every ISA tier on that
architecture via `__attribute__((target("...")))`.

## What "PASS" means per host

The `llm` binary's `--chat-test` hashes are the load-bearing gate:

- C CLI hashes (`--chat-test`):  `d2ba984b228fff7a / f12add080343c386 / a99b9d0705cc0220`
- Swift CLI hashes (`chat-test`): `d2ba984b228fff7a / 3e68bb5f53261a05 / 041f1a443d077ca9`

A host is "PASS" if it reproduces these hashes verbatim. Other tests
(`--self-test`, `qwen-test`, …) just need to print PASS — they're
synthetic and don't gate the cross-host story.

ISAs that use different reduction trees (AVX-VNNI vs AVX2-FMA vs AVX1
vs NEON dotprod vs NEON baseline) produce identical token outputs at
fp32 precision because the sampler argmax is stable under the
~1.5e-05 per-element drift the kernels can introduce. If a host's
`--chat-test` hash diverges, the diff between its per-layer trace and
Apple Silicon's is the first thing to look at (see `slm_trace_*` in
`model.c`).

## Test fleet

The same source tree on each host, with the toolchain noted. The sweep
script syncs `simd/` + `llm/` + `app/bridge.h` + the model GGUF (lazy
push — only if missing remotely), builds, and runs the test battery.

### 1. Apple M-series — local dev Mac (this machine)

- **Chip / OS**: Apple M-series, macOS 26 arm64.
- **Connect**: localhost (this Mac is where I run the master script).
- **Toolchain**: system `clang` (Apple Xcode).
- **ISA tier**: NEON + `dotprod` (`hw.optional.arm.FEAT_DotProd`).
- **Build**: `make -C llm && (cd swift-cli && make)`
- **Test**:
  ```
  ../Build/cli/llm --self-test
  ../Build/cli/llm --think-test
  ../Build/cli/llm --jinja-test
  ../Build/cli/llm --chunked-test
  ../Build/cli/llm --agent-test
  ../Build/cli/llm --chat-test          # hash gate
  ../Build/cli/qwen-test                # standalone qwen.c
  QWEN_GGUF=...  ../Build/cli/qwen-swift-cli chat-test   # swift hash gate
  ```
- Authoritative — every other host's hashes must agree with this Mac.

### 2. AMD Zen 5 — `halo2` (AMD Ryzen AI Max+ 395, Strix Halo)

- **Chip / OS**: AMD Ryzen AI Max+ 395 (Zen 5), Ubuntu 24.04 x86_64.
- **Connect**: `ssh halo2` (`amd@halo2.local`).
- **Toolchain**: Ubuntu `clang-18`.
- **ISA tier**: AVX-VNNI (`_mm256_dpbusd_avx_epi32`).
- **Sync**: `rsync -az --delete --exclude=Build/ ./ halo2:~/qwen.haiku/`
- **Build & test**:
  ```
  ssh halo2 'cd ~/qwen.haiku && make -C llm \
      && ../Build/cli/llm --chat-test \
      && ../Build/cli/llm --self-test \
      && ../Build/cli/qwen-test'
  ```
- Highest-end x86 test target. AVX-VNNI is the preferred tier when
  both VNNI and AVX2 are present.

### 3. Intel Haswell — `x.local` (i7-4578U, 2014 mini)

- **Chip / OS**: Intel Core i7-4578U (Haswell, 3.0 GHz), macOS 26 x86_64.
- **Connect**: `ssh x` (`leo@x-2.local`; hostname reclaimed from `x-2`
  back to `x` via `/Users/Shared/rename.sh`).
- **Toolchain**: Apple `clang` 17 (Xcode CLT).
- **ISA tier**: AVX2 + FMA + F16C.
- **Sync + build + test**: same `ssh x …` shape as halo2.
- First chip with FMA in the fleet — honest "baseline AVX2" distinct
  from Zen 5 / VNNI.

### 4. Intel Ivy Bridge — `mbp15` (i7-3615QM, MacBook Pro 15" 2012)

- **Chip / OS**: Intel Core i7-3615QM (Ivy Bridge, 2.3 GHz), macOS 26 x86_64.
- **Connect**: `ssh mbp15` (`agi@mbp15.local`; the box has no `leo`
  user, only `agi`).
- **Toolchain**: Apple `clang` 17.
- **ISA tier**: AVX1 + F16C (no AVX2, no FMA).
- **Sync + build + test**: same shape.
- Travel laptop, sleep prevention NOT applied — available best-effort.

### 5. Intel Ivy Bridge — `agi` (i7-3720QM, late-2012 quad-core)

- **Chip / OS**: Intel Core i7-3720QM (Ivy Bridge, 2.6 GHz), macOS 26 x86_64.
- **Connect**: `ssh agi` (`agi.iam@agi.local`).
- **Toolchain**: Apple `clang` 17.
- **ISA tier**: AVX1 + F16C (same tier as mbp15, higher clock).
- **Sync + build + test**: same shape.
- Long-running. Sleep prevention via `/Users/Shared/stay_awake.sh`.
  When agi naps despite this it's unreachable and the sweep skips it.

### 6. Intel Ivy Bridge / Windows — `mb-air-2012` (i7-3667U, MacBook Air 2012)

- **Chip / OS**: Intel Core i7-3667U (Ivy Bridge, 2.0 GHz dual-core),
  Windows 10 19045 x86_64.
- **Connect**: `ssh mb-air-2012.local` (MSYS2 sshd on Windows).
- **Toolchain**: MSYS2 `mingw-w64-x86_64-clang` (LLVM 22). Shell is
  MSYS2 bash; resulting `.exe` is native Windows.
- **ISA tier**: AVX1 + F16C.
- **Sync to Windows path**:
  ```
  rsync -az --delete --exclude=Build/ ./ mb-air-2012.local:/c/Users/leo/qwen.haiku/
  ```
- **Build**:
  ```
  ssh mb-air-2012.local 'cd /c/Users/leo/qwen.haiku/llm \
      && CC=/mingw64/bin/clang LDFLAGS="-static -lm" make'
  ```
  `-static` rolls libwinpthread / libgcc into the .exe; without it the
  binary segfaults under bare `cmd /c …` because `/mingw64/bin/` isn't
  on PATH at exec time.
- **Run**: PE stdout under SSH+MSYS bash is flaky, so redirect:
  `ssh mb-air-2012.local 'cd /c/Users/leo/qwen.haiku && Build/cli/llm.exe --chat-test > out.txt 2>&1 && cat out.txt'`.

### 7. Qualcomm Snapdragon 765G — Pixel 5 (Cortex-A76 + A55)

- **Chip / OS**: Qualcomm SM7250 (Snapdragon 765G), Android 14 aarch64.
- **Connect**: USB + `adb`. `brew install --cask android-platform-tools`
  on the dev Mac, then `adb devices -l` confirms it.
- **Toolchain**: Android NDK r29 (`brew install --cask android-ndk`;
  path
  `/opt/homebrew/share/android-ndk/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang`).
- **ISA tier**: NEON + dotprod + asimdhp (FEAT_DotProd, FEAT_FP16
  arithmetic).
- **Cross-compile**: see `rnd/sweep.sh` for the NDK invocation. Builds
  an Android target that skips libcurl (`tools.c` compiled out via
  `-DLLM_NO_TOOLS` once that flag exists; until then, an Android-only
  curl tarball is pushed alongside the binary).
- **Push & run**:
  ```
  adb push Build/cli/llm-android /data/local/tmp/llm
  adb shell 'taskset -a f0 /data/local/tmp/llm --chat-test'
  ```
  `taskset -a f0` pins to the big (A76) cores; without it Android's
  scheduler often parks the process on a LITTLE (A55) core (~3× slower).

### 8. Amlogic A311D — `khadas` (Khadas VIM3, Cortex-A73 + A53)

- **Chip / OS**: Amlogic A311D (Cortex-A73 + A53, ARMv8.0-A),
  Ubuntu 24.04 aarch64.
- **Connect**: `ssh khadas` (`leo@khadas.local`).
- **Toolchain**: Ubuntu `clang-18`.
- **ISA tier**: NEON **baseline** — `/proc/cpuinfo` Features line is
  `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid`, **no `asimddp`**.
  The dispatcher picks the `vmull_s8 + vpadalq_s16` fallback
  automatically.
- **Sync + build + test**: same shape as halo2.
- The "minimum ARM target" in this fleet — the canary that the
  no-dotprod NEON code path actually works on real silicon. The
  pre-simd-dispatcher `llm/neon.c` does NOT have this path; swapping
  to the simd dispatcher is what enables khadas to run at all.

### Sleeping (not in current sweep)

- `agi` autosleeps despite `pmset`; sweep skips it when unreachable.
- `mbp15` plugged in only opportunistically (personal laptop).
- The other six hosts stay on for the test campaign.

## Results format

`rnd/sweep-results/<host>.out` holds the captured stdout+stderr of one
sweep. The first line is the host label; subsequent lines are the test
outputs. The penultimate test block is `--- bench` which captures the
end-to-end prefill (pp) and token-generation (tg) throughput in tok/s
along with which ISA tier the dispatcher picked. A trailing summary
line aggregates everything:

    summary: <host>: PASS|FAIL | pp=<float> tg=<float> n_pp=<int> n_tg=<int> dispatch=<label> | hashes=<3>

`grep -h '^summary:' rnd/sweep-results/*.out` shows the whole fleet at
a glance.

## --bench details

`./Build/cli/llm --bench` runs a fixed ~30-token prompt through prefill
+ 64-token greedy decode (temperature=0, deterministic) three times,
then prints the median pp/tg. Greedy decode removes sampler RNG noise
from the timing. The fixed prompt is `"Explain in five short sentences
why benchmarking large language models across diverse hardware
matters. Avoid vague generalities; be concrete."` — identical bytes on
every host, identical token IDs after BPE.

The dispatch label is whatever `simd_init()` picked at startup
(`NEON+dotprod`, `NEON-baseline`, `AVX-VNNI`, `AVX2-FMA`, `AVX1`,
or `scalar`). For ARM hosts this is gated on `FEAT_DotProd`
(`hw.optional.arm.FEAT_DotProd` on Apple, `HWCAP_ASIMDDP` on Linux);
for x86 hosts it's the CPUID-derived tier.

## Throughput results

Populated by the sweep. Update after each successful run.

| Chip                         | OS / arch        | Dispatch tier   | pp tok/s | tg tok/s | n_pp | n_tg |
|------------------------------|------------------|-----------------|---------:|---------:|-----:|-----:|
| Apple M-series               | macOS arm64      | NEON+dotprod    |    24.43 |    24.29 |   25 |   64 |
| AMD Zen 5 (Strix Halo)       | Linux x86_64     | AVX-VNNI        |        — |        — |    — |    — |
| Intel Haswell i7-4578U       | macOS x86_64     | AVX2-FMA        |        — |        — |    — |    — |
| Intel Ivy Bridge i7-3615QM   | macOS x86_64     | AVX1            |        — |        — |    — |    — |
| Intel Ivy Bridge i7-3720QM   | macOS x86_64     | AVX1            |        — |        — |    — |    — |
| Intel Ivy Bridge i7-3667U    | Windows x86_64   | AVX1            |        — |        — |    — |    — |
| Qualcomm SD 765G (A76 + A55) | Android arm64    | NEON+dotprod    |        — |        — |    — |    — |
| Amlogic A311D (A73 + A53)    | Ubuntu arm64     | NEON-baseline   |        — |        — |    — |    — |
