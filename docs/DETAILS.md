# Strata - the details

The technical side of Strata: every measured number, the API, images, all settings and how the engine works.
New here? Start with the [README](../README.md); installing step by step is in [INSTALL.md](INSTALL.md), the models in
[MODELS.md](MODELS.md), common problems in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

> **On this page:** [Speed](#speed-measured) · [Other GPUs](#other-gpus-estimated) · [Which model?](#which-model) ·
> [Requirements](#before-you-start) · [Windows](#windows) · [Linux](#linux) · [API](#using-it) ·
> [MCP tools](#tools-from-mcp-servers) · [MCP server](#manage-strata-from-your-ai-assistant-mcp-server) ·
> [Images](#images-vision) ·
> [Troubleshooting](#troubleshooting) · [How it works](#how-it-works)

---

## Upstream v0.1.36 in the V100 fork

This update includes upstream releases 0.1.32 through 0.1.36. The upstream source is
[`36fa455`](https://github.com/Niko1221/Strata/commit/36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca).
See the [0.1.36 release](https://github.com/Niko1221/Strata/releases/tag/v0.1.36).

The fork keeps support for Tesla V100 (`sm_70`). Compile the engine locally with
CUDA 12.x. CUDA 13 cannot compile code for `sm_70`. A normal V100 build does not
enable the experimental Pascal option.

The V100 prompt path keeps FP16 tensor-core products, compact expert buffers,
and the fork's two-kernel hyper-connection window. The BF16x2 option has no effect
on V100 or Turing cards. The new fused prompt experts need `sm_80` or newer.
The new cluster decode kernels need `sm_90` or newer. Older cards keep their
previous kernels.

The update also includes setup and server fixes, saved expert profiles,
request-cancel records, update scripts, low-RAM options, and the MCP server.
The Linux asynchronous file reader now uses the correct completion-array offset.
The PLE reader records its row count before it accepts read requests. These two
corrections prevent invalid completions and incorrect PLE rows.

**Verification on two Tesla V100 16 GB cards, CUDA 12.8, and an AVX2 CPU:**

- The Release build for `sm_70` passed.
- Tools and setup: 256 Python tests passed.
- Server: 158 Python tests ran; nine were skipped. The suite passed.
- CTest: 60 tests passed. Two fused-expert tests were skipped because V100 does
  not support those kernels.
- The AVX512-only expert test was excluded on this CPU. The legacy PLE capture
  test was excluded because its fixed capture files were not available.
  The synthetic PLE reader tests and live PLE inference passed.
- The Q2_0 model, native PLE table, MTP draft, int8 KV cache, and 20/28 layer split
  ran with `STRATA_PREFILL_BF16X2=1`. Short replies, a 6,063-token prompt, native
  tool calls, and streamed replies passed.
- The live OpenAI checks passed all five cases. The complete Hermes tool
  workflow passed.
- With the default settings, a 30,063-token prompt and short replies passed.
- The rebuilt CUDA vision encoder identified a red test image correctly.
- After a client disconnected during prefill, the next request completed.
  That request waited 16.11 seconds in the queue. This check does not establish
  immediate cancellation.
- The opt-in API monitor showed completed and disconnected requests in a real
  browser. Request inspection and rendering passed without browser errors.

The 6,063-token prompt ran at 621.1 prompt tokens/s and 63.9 output tokens/s.
These values are a smoke check, not a controlled speed comparison. The earlier
benchmark tables below still identify the builds that produced them.
AMD, Windows, mixed-generation GPU splits, and the new `sm_80`/`sm_90` kernels
were not tested on this machine.

## Upstream v0.1.39 in the V100 fork

This update integrates the upstream releases through
[v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39). The fork keeps its V100
tensor-core experts (the native FP16 prompt path with the compact 192-slot streamed ring),
the attention changes, the asymmetric GPU split and the disk conversation cache. New
upstream kernels with higher architecture requirements stay gated: the fused prompt
experts need `sm_80` or newer, the cluster decode kernels need `sm_90`, and older cards
keep their previous kernels. The experimental Pascal option stays off.

The CUDA and HIP engines drop the upstream `--trim-stage-weights` flag and
`STRATA_STAGE_TRIM`: an explicit `--layer-split` already loads only each stage's own
layers, so the trimming is automatic and those settings are no longer accepted. The
separate experimental Intel (SYCL) engine keeps its own opt-in path.

Batch slot sessions are allocated after the main session, the native head, the logits
buffer and the MTP draft layer, so the engine's fit check sees the memory really left on
each card. A count that does not fit falls back to fewer slots, or to no batch slots (one
request at a time), at startup: the HTTP server stays ready and the configuration is
untouched. Observed on this rack at the 524,288-token context with the 20/28 split.
See [BATCHING.md](BATCHING.md) for the allocation order, the fallback and the per-slot
memory costs; the final admission-time figures are in the benchmark report.

**Verification on two Tesla V100 16 GB cards, CUDA 12.8, and an AVX2 CPU:**

- The Release build for `sm_70` passed.
- CTest: 81 supported tests ran after all fixes; 79 passed and the two `sm_80`-only
  kernel tests were skipped on V100 (100 % of the runnable set, 76.8 s; two IQ-fixture
  cases need the pack tooling's Python path set, then they pass). The AVX-512-only
  expert tests were excluded on this Ryzen CPU. The legacy PLE capture test was excluded
  because its fixed capture files were not available; the live PLE path passed.
- Server: 275 Python tests ran; seven were skipped. The suite passed.
- Tools: 396 Python tests ran; one was skipped. The suite passed.
- Targeted checks on the final build: `tools.test_setup_parallel` (setup's Volta
  `"parallel": 2` default and its recommendations) and the concurrent per-request
  history tests passed.
- HTTP smoke at the full 524,288-token context on the fallback server (requested 2 slots,
  serving one at a time): Chat Completions, the Responses API non-stream and its streamed
  SSE events, a real-model function tool call (`add_integers(17, 25)` answered 42) with a
  stateless follow-up, the Anthropic endpoint, a malformed-tools request answered HTTP
  400, strict JSON-schema output (`{"ok": true}`), and a real red PNG vision input
  identified as red. All passed.
- Final HTTP concurrency at a 32,768-token context: four independently marked streamed
  requests completed through two serving slots. Metrics showed two active slots and
  excess requests waiting. Token-emission intervals overlapped by 6.1 s and 6.2 s.
  The live web monitor also showed both slots decoding with one request waiting.
- Batch exactness at a 32,768-token context with two slots (`tools/batch_test.py`,
  greedy, `--adapt-every 1000000 --pcie-frac 0`): each slot's 128 generated tokens were
  identical to its solo run, token for token; the two admissions took 1.3 s, and the two
  128-token answers (256 tokens together) decoded in 5.8 s at a 44.0 tok/s aggregate
  (49.6 tok/s counted from the first token).
- The L3 disk tier with three batch slots passed all nine exact-token-parity checks in
  126.1 s: A2 reused exactly 152 tokens and the next request exactly 299 - the longest valid
  prefix wins, so the 25-token and 172-token disk records present could not displace the
  longer 152- and 299-token states. A 4,021-token request read a 512-token first prompt
  part, gave way at 1,024 (`BYIELD`), and continued with 1,024 reused plus 2,997 read, with
  the first checkpoint preserved.

**Prefill on the two V100s (4 October 2026, final build).** A controlled before/after
run of the integrated build against the previously installed runtime, one request at a time,
cold, Q2_0 with int8 KV at the 524,288-token context, the 20/28 layer split and MTP (`--spec 8`)
unchanged; two fresh requests per size, medians:

| Target prompt | Before (tok/s) | Integrated (tok/s) | Change |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Each value is the median of two fresh requests (`reused=0`); with an even sample the median is
the mean of the two repeats. The gains are small and, at 2K and 32K, sit within the baseline's
own repeat spread (5.2% and 1.26% of its median); only the 8K gain clearly exceeds that
spread (1.09%). Two repeats per size are not a statistical claim. These are the final build's
measured values (engine SHA-256 `e9ad337a…`).

All six paired first tokens matched the baseline's. No full-output parity is claimed: the
adaptive expert tier keeps a different mix of experts on the GPU and the CPU for the
integrated build, and the two paths round differently. No decode or quality change is
claimed. The full report is
[`benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md`](../benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).
Per-request time-to-first-token reporting now follows the same rule on the solo and the
slot path: the wall time from the request's start to its first generated token, recorded
per request in `/metrics` (`ttft_ms`).

## Retained CPU, PLE and Volta changes

These changes are retained in the fork. They keep the model formats, the int8 KV
cache at a 524,288-token context, vision, MTP, the L2/L3 conversation tiers and
the NVMe-backed n-gram table. No 20% end-to-end gain is proved yet, and no
performance guarantee follows. The final measurements are still pending.

- **AVX-2 i-quant rows (IQ3_XXS, IQ3_S, IQ2_S).** On a CPU with AVX2, no AVX-512
  and no AVX-VNNI, the AVX-2 expert kernel serves one-token (NT1) gate/up rows
  for these three formats by default. NT means the tokens in one group; NT1 is a
  group of one token. The IQ3_S and IQ2_S decodes use 32-bit loads to read packed
  index bytes, which shortens the grid assembly. The rows reduce in ggml-cpu's
  own float order and apply the format normalization factor after the reduction,
  so a row is ggml's dot product bit for bit, including subnormal products. The
  IQ4_NL down rows are unchanged. An explicit `STRATA_IQ_MT_MIN` keeps its rule
  and overrides this default.
- **CPU pool workers on SMT siblings.** After each non-host physical core has a
  worker, the pool pins additional workers to the allowed SMT siblings of the
  worker cores. SMT (simultaneous multithreading) means a physical core's second
  logical processor. The host core's sibling comes last. A CPU outside the
  process's allowed set, and any CPU a worker or the host already holds, is
  skipped. The default placement is unchanged: a pool no larger than the physical
  core count, which is five workers and the host thread on the measured six-core
  CPU. On Windows the extras stay unpinned, as before.
- **L3 background-writer affinity.** The asynchronous direct L3 fallback writer
  restores the allowed CPUs at the start of its thread. A new thread inherits the serving thread's
  pin, which the session loop holds to the reserved host core (the first physical
  core, logical processor 0), so before this change each writer ran on one
  logical processor. The pool captures the allowed CPUs at construction, before
  that pin; each writer, created later on the pinned thread, captures a copy of
  that saved set and restores it at entry. The GPU host thread stays pinned. The
  L2 parking, the snapshot format, the FNV hash, the disk budgets and the order
  of the future writes are unchanged. In a strace smoke on two NVMe conversation
  files, the old writer inherits mask [0] and restores nothing; each new writer
  restores [0..11] before its file open. Each restore returns 0 on the machine (six
  cores, 12 threads). The RAM A/B/A reuse was 4615 of 4620 tokens in both arms.
  The trace is not timing evidence. The matched HTTP study does not meet the
  20% target. See the [CPU and SSD study](../benchmarks/v100-iq3_s-prefill-wmma.md#cpu-and-ssd-study-2026-10-10).
- **SSD-PLE page cache and boundary rows.** The n-gram (PLE) reader's bounded
  page cache now copies a row that crosses a page boundary from both cached
  pages. Before, such a row always re-read the SSD. The reader copies the head of
  the row from the first page and the tail from the second. The destination stays
  unchanged when a page is missing, so a partial hit is still a miss. This covers
  the 90-byte IQ4_NL row (the production default) and the 110-byte Q5_0 row.
- **Volta prompt top-k bound.** On sm_70 the prompt path's QSA top-k may use the
  active-block bound of a prefill. The bound selects the register kernel when the
  prompt's active blocks fit it, even when a long context capacity makes the
  score-row stride large. Decode and captured graphs pass no bound and keep the
  capacity rule. Other CUDA devices keep the capacity rule.

## Speed (measured)

### Tesla V100 fork benchmark

The current Strata-V100 numbers were measured on 2026-10-02 on a single, cooled card
(below). They use Qwen3.8-Flash-Next Q2_0 on a Tesla V100-PCIE-16GB (`sm_70`), Ryzen 5
3600 (6 cores, 12 threads), 48 GB DDR4-3200, CUDA 12.8, and Linux. The engine is the
fork's version 0.1.31 (CMake project version), installed build commit `78417ea`
(SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`).
Runtime settings: 262,144-token context, int8 KV cache, automatic prefill,
MTP speculative decoding (`--spec 8`, draft floor 0.70), the paired
expert variant, and a 700 MiB vision reserve. Prompt and decode timings come from the
server's `/metrics` endpoint, not from client wall clocks. The expert cache held 4,198
slots at startup.

**Single-card cooled benchmark (2 October 2026)**

Measured on physical GPU 1 only: a single Tesla V100-PCIE-16GB on a PCIe Gen3 x16 link.
The engine and vision encoder ran on GPU1. GPU0 was not used for inference.
Eight target sizes ran, from 1K to near 256K (the configured capacity is 262,144 tokens).
Each size had three fresh requests. Every request wrote exactly 256 output tokens. The
table gives medians.

| Target prompt | Median prompt tokens | Prefill (tok/s) | Decode (tok/s) | Max GPU temp |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

† Software thermal slowdown active during the prompt (84 °C).

Before every measured request, GPU1 cooled for at least 120 seconds. The gate was:
temperature at or below 55 °C, stable for 15 seconds, without software thermal
slowdown. The gate checked every 3 seconds. Separate telemetry ran at 1 Hz.
Actual start temperatures were 51-55 °C. Actual waits were
121.4-588.7 seconds (about 2-9.8 minutes; the 128K and 256K requests needed the
longest waits). Seed 20261002. All 24 requests were fresh (`reused=0`).

The card showed no software thermal slowdown through 64K; its maximum there was
81 °C. The 128K and 256K prompts heated the passively cooled card to 84 °C, and it
throttled during those prompts despite the cooled start. Long requests may throttle
after a cooled start. All requests were kept; none were removed.

Raw data:
[`matrix.json`](../bench/results/2026-10-02-v100-single-cooled/matrix.json),
[`completed-requests.json`](../bench/results/2026-10-02-v100-single-cooled/completed-requests.json),
[`protocol.json`](../bench/results/2026-10-02-v100-single-cooled/protocol.json),
[`summary.json`](../bench/results/2026-10-02-v100-single-cooled/summary.json), and
[`gpu.csv`](../bench/results/2026-10-02-v100-single-cooled/gpu.csv) (continuous
1 Hz telemetry).

Repeat the sweep with `bench/run_v100_bench.py`: targets
`1024,4096,8192,16384,32768,65536,131072,256073`, `--max-tokens 256`,
`--seed 20261002`, `--repeats 3`, and the same cooldown gate before every
measured request.
The harness does not pause automatically. Apply the cooldown gate between
individual measured requests, as in this run.

**Older measurements.** The September sections used 64 output tokens and older
engine builds. The 1 October comparison used two cards and 256 output tokens.
These tables are kept for reference. They are not controlled comparisons with
the 2 October single-card run because the protocols differ.

The prefill fast path (2026-09-28) keeps everything else unchanged but replaces the BF16 projection GEMMs with
FP16 tensor-core products (a BF16 value is exact in FP16 - the scalar 16x16 fallback is gone), the PLE row
reads with io_uring O_DIRECT (previously one synchronous pread at a time), and moves the model files to the
256 GB NVMe. Fresh uncached requests measured 2026-09-28 (engine log, `reused=0`):

| Prompt size | Exact tokens | Prompt time | Prompt speed | Output speed | Expert-cache hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ~4K | 4,143 | 4.19 s | **989.1 tok/s** | 49.3 tok/s | 84.4% |
| ~4K | 4,376 | 4.34 s | 1007.5 tok/s | 52.1 tok/s | 86.5% |
| ~8K | 8,785 | 9.09 s | **966.5 tok/s** | 50.6 tok/s | 88.0% |
| 32K | 29,527 | 25.97 s | **1136.8 tok/s** | 50.5 tok/s | 88.7% |
| 128K | 117,837 | 178.72 s | 659.3 tok/s | 44.9 tok/s | 88.4% |
| 256K | 256,076 | 618.39 s | 414.1 tok/s | 38.8 tok/s | 84.3% |

The 8K row was re-measured three times from a cool card (967.0 / 965.0 / 966.5 tok/s). The V100-PCIE-16GB is
passively cooled: after the long rows it idles at ~79 C and throttles from the 1380 MHz boost to ~960-1280 MHz
at ~83 C under load, dropping the same prompt by up to ~25% (the 8K prompt spans 616-967 tok/s depending on
when it runs). The table above is the cool-idle measurement. The raw rows are in
[`bench/results/2026-09-28-v100-fastpath`](../bench/results/2026-09-28-v100-fastpath), and the whole table is
reproduced by [`bench/run_v100_bench.py`](../bench/run_v100_bench.py).

The PLE row gather for a fresh request dropped from ~3.6-8 s to ~7 ms, and the hyper-connection GEMMs (the
largest BF16 projection block) from ~5.2 s to ~0.5 s at 8K. Decode is unchanged.

An explicit layer split loads each card's canonical weights and native dense
projections for its own layers only. Automatic placements keep the full tables.
The configured context capacity and vision support are unchanged.

**Opt-in decode kernel variant (`STRATA_EXPERT_PAIR=1`, off by default):** with
a native expert pack, the grouped expert kernels compute two rows per warp
instead of one. The GPU parity fixtures produced byte-identical results on
both V100 cards. Set this variable in the installed config's environment
to enable the variant. The service file does not need a change.

The prompt path's FP16 experts use per-expert scratch (at most one chunk's
rows per expert) by default. The decode path quantizes the shared expert's
activation once per layer. CPU activation quantization runs only for tokens
with CPU expert jobs. A window with no CPU expert jobs does not wait for the
CPU worker barrier. Cache refills start on both cards before either card waits.


**V100 follow-up measurement (1 October 2026; earlier, dual-GPU A/B)**

This run used two V100 PCIe 16GB cards. GPU0 had a Gen3 x2 link. GPU1 had a
Gen3 x16 link. Each size had three fresh requests and 256 output tokens.
The table gives the median speed in tokens per second.

| Target prompt | Original prefill | Installed prefill | Change | Original decode | Installed decode | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 442.3 | 525.0 | +18.7% | 66.9 | 69.9 | +4.5% |
| 8K | 620.8 | 705.9 | +13.7% | 66.8 | 64.5 | -3.4% |
| 16K | 820.2 | 955.6 | +16.5% | 64.0 | 66.5 | +3.9% |
| 32K | 976.3 | 1105.9 | +13.3% | 66.2 | 70.0 | +5.7% |

The 20-percent prefill and decode target was not met at all four sizes.
Decode results are mixed. Do not use the kernel fixture speed as a runtime
speed claim. Recorded samples reached 84 C on GPU0 and 83 C on GPU1.
Both cards reported software thermal slowdown during this run.

The installed settings keep the 20/28 layer split, 262144-token context,
int8 KV, automatic prefill chunks, spec 8, draft confidence 0.70, and vision.
The paired expert variant is enabled. On GPU0, owned-layer loading reduced
native dense weights from 1376.20 MiB to 541.27 MiB. The expert cache increased
from 6750 to 7918 slots.

Raw measurements:
[original](../bench/results/2026-10-01-v100-pass2/baseline/matrix.json),
[installed](../bench/results/2026-10-01-v100-pass2/final-installed/matrix.json),
[summary](../bench/results/2026-10-01-v100-pass2/final-summary.json), and
[thermal samples](../bench/results/2026-10-01-v100-pass2/final-installed-gpu.csv).
Use `bench/run_v100_bench.py` with targets `4096,8192,16384,32768`,
`--max-tokens 256`, `--seed 20261002`, and `--repeats 3` to repeat the sweep.

**V100 int8-KV prefill with FP16 tensor-core experts (3 October 2026; dual-GPU A/B)**

This run used the same two V100 PCIe 16GB cards and the same settings as the
1 October A/B above: layer split 20, int8 KV, 524,288-token context, YaRN scale 2,
speculation window 8 with a 0.70 draft floor, automatic prefill and expert cache,
and a 700 MiB vision reserve. Prompt checkpoints and the conversation cache were
off, so every request read its whole prompt (`reused=0`, `resume=0`). Each size had
three fresh requests with exact token ids (seed 20261003), and each request
generated one token. The table gives the median prompt time and speed.

| Target prompt | Baseline prompt time | Candidate prompt time | Less time | Baseline prefill | Candidate prefill | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 3,663.4 ms | 2,910.7 ms | 20.6% | 559.0 tok/s | 703.6 tok/s | +25.9% |
| 4K | 5,039.0 ms | 3,794.8 ms | 24.7% | 812.9 tok/s | 1,079.4 tok/s | +32.8% |
| 8K | 7,810.4 ms | 5,767.0 ms | 26.2% | 1,048.9 tok/s | 1,420.5 tok/s | +35.4% |
| 16K | 11,052.3 ms | 8,517.8 ms | 22.9% | 1,482.4 tok/s | 1,923.5 tok/s | +29.8% |
| 32K | 18,078.8 ms | 14,506.0 ms | 19.8% | 1,812.5 tok/s | 2,258.9 tok/s | +24.6% |

All five sizes gain at least 20% more tokens per second (24.6% to 35.4%). The
"Change" column is the speed change; "Less time" is the fall in prompt time.
Decode was not measured, because each request generated one token.

At 2K the batched prompt chunk is 2,047 tokens, which is below the compact walk's
2,048-token stream floor. That row therefore keeps the int8 small-chunk path; its
gain comes from the new staging, buffer-sizing, and refill policies.

The candidate moves the routed prompt experts of the native Q2_0 pack to FP16
tensor-core products with FP32 accumulation. The gate/up and hidden buffers then
hold a compact row capacity instead of one row per routed token; the last small
chunk keeps the int8 path. On this path the compact walk also selects a 192-slot
streamed ring (`STRATA_PREFILL_RING` overrides it). The kernels select on compute
capability 7.0 (Volta), not on a product name. `STRATA_PREFILL_FP16TC=0` keeps the
int8 reference on every device. The candidate also starts the refill of the
expert-cache slots that a non-last stage lent the prompt path before that stage
waits for the last stage (`STRATA_PREFILL_REFILL_OVERLAP=0` disables it; the serial
and blocking refill modes bypass it).

Before every measured request, both cards were at or below 55 °C for 10 seconds.
Every request met the gate. The highest sampled temperatures were 51 °C (GPU 0)
and 57 °C (GPU 1) in the baseline, and 50 °C and 57 °C in the candidate. Neither
card reported a thermal or hardware slowdown.

The new path is not bit-exact. The parity test `prefill_fp16tc_test` passed on both
cards: against a double-precision reference the new path measured 1.073e-02
relative RMS and the int8 MMQ path 1.074e-02. The first generated token was
identical in 14 of the 15 paired requests. A one-time semantic smoke passed all
eight checks in both arms (retrieval at 10%, 50%, and 90% depth at 2K and 32K, and
one arithmetic question at each length), and both arms produced the same output
tokens. This is a smoke check, not a quality measurement. Do not use these results
to claim unchanged perplexity or long-generation quality.

Only V100 cards were tested.

Raw measurements: [the result directory](../bench/results/2026-10-03-v100-prefill/).
Repeat the sweep with [`bench/run_v100_prefill.py`](../bench/run_v100_prefill.py),
targets `2048,4096,8192,16384,32768`, `--repeats 3`, `--seed 20261003`,
`--max-new 1`, and the same cooldown gate before every request.

The same harness measures a chain of agent turns with `--chain`. It reads one prefix prompt. Then it
sends one turn for each `--chain-diffs` value (default `0,1,32,64,128,256,512,1024,2048`). Each turn
adds that exact number of tokens. Thus every prompt is a prefix of the next one. The engine's prompt
cache stays on, and `--chain-cache-every` sets its checkpoint interval (default: a quarter of the
prefix). A conversation disk store that the config names goes to a private directory under `--out`.
The harness drops `--expert-profile-save`. The first diff is 0. That turn sends the same prompt again.
Each row records the requested diff, the tokens the engine reused (`reused`), and the tokens it read
(`prompt_read`). The engine resumes at most one token short of the prompt, so the two counts differ.
Do not report the requested diff as a read count. Add an arm's own knobs with `--xargs`, `--env`,
`--short-read N`, or `--pipeline-windows N`. Every request sends `pin=n-2` by default: the engine
ignores `pin >= n-1`, so the next turn re-reads the last two tokens instead of the tokens above its
newest periodic checkpoint. `--no-chain-pin-tail` turns the pin off and shows that gap. Example:
`python bench/run_v100_prefill.py --chain --arm-label chain-baseline --exe /tmp/engine --out /tmp/chain`.
Compare two chained runs with `--compare`.

Long-context baseline for comparison (engine 0.1.20, models on the portable SSD, before the fast path):
| Prompt size | Exact tokens | Prompt time | Prompt speed | Output speed | Expert-cache hit rate | Total request |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1,031 | 3.54 s | 291.0 tok/s | 33.6 tok/s | 81.7% | 5.4 s |
| 8K | 7,431 | 16.25 s | 457.4 tok/s | 41.2 tok/s | 90.1% | 17.8 s |
| 32K | 29,512 | 63.52 s | **464.6 tok/s** | 42.5 tok/s | 92.8% | 65.0 s |
| 128K | 117,833 | 363.06 s | 324.6 tok/s | **44.0 tok/s** | 94.4% | 364.5 s |
| Full context | 256,073 | 987.69 s | 259.3 tok/s | 40.3 tok/s | 96.2% | 989.4 s |

The separate hardware calibrator generated 256 tokens while tuning and measured **45.5 tok/s** at the selected
0.28 PCIe share and 0.70 draft floor. The tables above deliberately include prompt processing and use
uncached prefixes (`reused=0` for every row); the 256K row reaches within 6,068 tokens of the configured
262,144-token limit.

RTX 5070 **12 GB**, Ryzen 5 7600 (6 cores), 64 GB DDR5-5200, Windows, engine 0.1.26 with the settings setup writes
(`--prefill auto`, 8-bit KV above 4K, KV streaming from 64K). One code-agent prompt per length, 256 generated tokens,
MTP speculative decoding on. "262K" is the model's full context window (a 259,943-token prompt). The IQ2_XS row was
measured with Swift 1.5's IQ2_XS, which runs at the original's speed.

**Engine 0.1.36 (#136), the same PC:** Q2_0's prompt experts run on fused int8 tensor-core kernels (RTX 30 and newer):
4K 1,294 -> 1,570, 32K 2,170 -> 2,653, 128K 2,123 -> 2,468 tokens/s (+16-22%), as close to an FP16 reference as the
previous kernels (closer at 32K: teacher-forced KL 0.009 vs 0.012). The decode path's block selection and greedy
argmax run on thread-block clusters (RTX 50, sm_90+; other cards keep the previous kernels; the same tokens): Q2_0 output at 4K 89 -> 93.5, at 128K
64.5 -> 76.4 tokens/s. `STRATA_PF_FUSED=0` keeps the previous prompt kernels (byte-identical answers to 0.1.35);
`STRATA_PF_FUSED=1` also runs the native IQ packs' fused kernels (opt-in: IQ2_XS prompts +12% at 4K, +3% at 32K, the
IQ3 packs about even; on HIP they run on gfx11 and on gfx1200 / gfx1201, see AMD_HIP.md); `STRATA_QSA_CLUSTER=0` / `STRATA_ARGMAX_MULTI=0` turn the decode kernels off. The tables
below are 0.1.26's.

### Prompt processing (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 536 | 1,299 | 2,171 | 2,126 | 2,107 | 1,304† |
| **IQ2_XS** | 534 | 1,256 | 2,092 | 1,754 | 1,752 | 1,181*† |
| **IQ3_XXS** | 482 | 1,007 | 1,745 | 1,609 | 1,602 | - |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 | - |
| **Coder** | 656 | 1,583 | 2,177 | 2,236 | 2,208 | 1,034** |
| **IQ3_S** (AMD RX 7900 XTX, gfx1100) | 760 | 1,275 | 1,641 | 1,594 | 1,494 | - |

Engine 0.1.26; `bench/results/2026-09-29-speed-0126`. The AMD RX 7900 XTX row: engine 0.1.38 on a ROCm
10.2 nightly with a tuned gfx1100 hipBLASLt-100500 table (upstream as [#755](https://github.com/Niko1221/Strata/pull/755)),
the median of 3 clean cells per tier
(Ryzen 9 7900X, 96 GB; 1K-128K one-shot runs, 256 generated tokens,
greedy). At 32K-128K that is 8-28% faster than 0.1.22. † not measured
again: 0.1.22. \* measured with images on (the image encoder's VRAM reserve leaves fewer experts cached). \*\* not
measured again: 0.1.14.

### Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 87.3 | 93.0 | 81.8 | 76.2 | 73.7 | 60.3† |
| **IQ2_XS** | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 | 52.8† |
| **IQ3_XXS** | 61.9 | 61.6 | 58.5 | 57.2 | 49.0 | - |
| **IQ3_S** | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 | - |
| **Coder** | 58.9 | 55.1 | 54.9 | 53.2 | 43.0 | 42.8† |
| **IQ3_S** (AMD RX 7900 XTX, gfx1100) | 65.1 | 60.4 | 64.4 | 61.8 | 59.2 | - |

Engine 0.1.26, the same runs. The AMD RX 7900 XTX row: engine 0.1.38 on a ROCm 10.2 nightly (tuned
gfx1100 hipBLASLt-100500 table, upstream as [#755](https://github.com/Niko1221/Strata/pull/755)), median of 3 clean cells per tier (same box; decode is flat in
context - the GDN linear attention is O(1) per token; 1K/4K prefill ran slightly under the 0.1.31
packaged-ROCm numbers, 32K-128K ahead). † not measured again: 0.1.14.

Output speed depends on the text as well: speculative decoding runs faster when more of the drafted tokens are
accepted, so a different answer to the same prompt moves it by several percent. Run back to back on the 4K prompt,
0.1.14 writes 88.5 tokens/s and 0.1.12 85.7. The numbers before 0.1.13 (prompts about half as fast):
[`bench/results/2026-09-24-final`](../bench/results/2026-09-24-final/matrix.md); these:
[`bench/results/2026-09-28-speed-0114`](../bench/results/2026-09-28-speed-0114/README.md).

IQ3_XXS and IQ3_S at 262K are not measured: with their 43 / 50 GB of experts, a 260K-token context brings a 64 GB PC
to its memory limit by setup's estimate (the experts + the context's KV cache + 24 GB), so setup recommends up to 128K
with them on 64 GB. A longer context you choose (`--context 262144`, or a pick in its list) is kept, with a note: users
ran IQ3_S at 256K on 64 GB with RAM to spare (#406). In the low-RAM mode the KV cache stays in VRAM and the context
does not count against RAM. IQ3_S (engine 0.1.4 or newer) is published for the original model and for Swift 1.5
(the same shard layout as Swift's other sizes).

**KV streaming (engine 0.1.5):** at 64K and more, setup keeps the context's KV cache in RAM and only the part the
attention reads in VRAM (`--kv-resident 32768`), so more experts fit on the GPU. Q2_0 at 262K: 50.9 -> 62.6 tokens/s
(1,589 -> 3,872 experts in VRAM); at 128K about +6%. The attention reads exactly the same values (only where the KV lives changes); it
costs ~13.7 KB of RAM per context token (1.7 GB at 128K). Existing installs: run `START-HERE.bat --setup` once to turn
it on. Setup turns it on when the RAM has room for it; `--kv-streaming on|off` overrides that (on past the RAM test with a
note; never under WSL, which cannot stream).

Under WSL2 the driver caps what `cudaHostAlloc` may pin at about 1 GiB, so the KV host copy is pinned with `cudaHostRegister` of ordinary memory there (detected through `/dev/dxg`); `STRATA_KV_HOST_REGISTER=1` forces that order on any system and `=0` forces `cudaHostAlloc` first (the other method is the fallback in both cases). On Linux without WSL nothing changes by default; with the switch the engine prints `KV streaming: the host copy is pinned with cudaHostRegister` (#1498 by yjamil8; WSL not tested here, the forced path was run on a Tesla P100: `--kv int8 --kv-resident 32768`, answers and decode speed unchanged).

**4-bit KV cache (engine 0.1.8, optional):** `START-HERE.bat --setup` asks above 8K context (or pass `--kv q4_0`). It
halves the KV cache's memory with a Hadamard rotation before 4-bit rounding (PR #21), about 4% faster at 128K, but it
is measurably less precise on long documents (perplexity +8-12%; needle tests still pass). 8-bit stays the default.
Details: [`bench/results/2026-09-27-kv-q4`](../bench/results/2026-09-27-kv-q4/README.md).

**Hybrid K8V4 KV cache (engine 0.1.25, optional, PR #120):** `--kv k8v4` (`START-HERE.bat --setup --kv k8v4`) keeps
the keys at 8 bits and stores the values as rotated 4-bit: 23% less KV memory than 8-bit, so more experts fit in
VRAM. RTX 3090, the Coder at 198K context: 99 instead of 85 tokens/s output, the same needle results, prompts 2-5%
slower. It streams its KV cache like the other formats (`--kv-resident N`): on an RTX 2060 SUPER 8 GB at 128K with
20,480 resident cells, +780 expert slots over resident K8V4, and it scores better than 4-bit KV on long documents.

**TurboQuant KV cache (opt-in, `--kv tq3|tq4|tq6|tq8`):** each 256-value head is rotated with a Hadamard
transform and quantized against a Lloyd-Max codebook fitted to the rotated coordinates, with one FP16 norm per head;
the width is 3, 4, 6 or 8 bits per value; the fit is minimum squared error only (this cut has no QJL residual).
It is a resident format: the K/V stays in VRAM (mode 0, no host copy), so it does not stream, and `--kv-resident` is
refused. The main layers use it; the MTP draft layer keeps its plain INT8 windowed ring, as it does under
K8V4 (its attention and verify paths handle whole formats only), so `--mtp`/`--spec` keep an INT8 draft cache.
Setup does not offer these values yet.

| `--kv` | Bytes/cell | 524,288-token KV, 12 layers |
|---|---:|---:|
| int8 | 1,056 | 6.1875 GiB |
| tq8 | 1,032 | 6.046875 GiB |
| tq6 | 776 | 4.546875 GiB |
| tq4 | 520 | 3.046875 GiB |
| tq3 | 392 | 2.296875 GiB |

Those are calculated capacities (524,288 cells x 12 layers), not measured savings; a smaller cache only becomes more
expert slots with the elastic K/V (`--kv-grow`), which gives the free VRAM to the expert cache. The context capacity
and the CPU memory defaults do not change; only the bytes per KV cell change.

Measured (before the opt-in tensor reader below, 2026-10-25): two Tesla V100 16 GB cards at 175 W, Ryzen 5 3600, IQ3_S, `--mtp --spec 4`,
`--layer-split 16`, `--pipeline-windows 2`, one run per format, prompts of 2,071 and 51,746 tokens (the "2K"/"49K"
labels were nominal). Wall seconds for the prompt request: int8 2.926 / 24.289, tq4 3.297 / 35.190, tq8 3.401 /
41.085, tq6 3.431 / 39.272, tq3 3.173 / 33.643. Every arm ran without a crash. These runs show a prompt slowdown,
not a speed win. Static analysis attributes the long-prompt cost to the prompt path using the decode-attention
fallback (the batched reader below is off by default). Total GPU memory stayed near 15,656/15,684 MiB:
the expert cache fills the freed space, so total GPU usage is not a measurement of KV savings. This exploratory
five-format run is not part of the published evidence file.

All five formats pass the four OpenAI-style recall and native-tool scenarios. The old fifth check required the
literal word "tool" in a plain answer; it was removed because that tests wording, not tool behavior. Hermes passes
for int8, tq4, tq8 and tq6; tq3 still calls the nonexistent `list_jobs` instead of `list_scheduled_jobs`.
An earlier int8 run also made that error, so these small checks do not establish quality parity or a TQ-specific
regression. Perplexity was not measured. A needle test (`tools/needle_bench.py`, unmodified) found 15 of 15 cells for
both int8 and tq6 at 3 lengths and 5 depths, up to an actual 122,289 prompt tokens (the "128K" labels are nominal).
The daily configuration hash was identical before and after. This is one synthetic retrieval test, not a general
quality result. The final tq6 log confirms conversation parking at 2,071 and 51,746 tokens and checkpoint-archive
parking, without the earlier draft-ring snapshot refusal. The GPU layout/append/gather/attention regression
(`kv_tq_parity`) passes, including under CUDA Compute Sanitizer with zero memory errors. The
[full report](../benchmarks/v100-turboquant-2026-10-25.md) and its
[sanitized evidence](../bench/results/2026-10-25-turboquant-published/v100-turboquant-2026-10-25.json) hold the
measured rows and settings.

**Volta TQ prompt attention (experimental, off by default):** `STRATA_PROMPT_ATTN_TQ=1` enables a tensor-core
prompt reader for `tq3`, `tq4`, `tq6` and `tq8` on sm_70. Unset or `0` keeps the existing decode fallback;
other GPU architectures are unchanged. It decodes selected compressed tiles into shared memory, not a persistent
FP16 cache. The per-head norm is applied once in FP32. KV allocation and stored bytes do not change.

Same-day measurements on the two V100s above, with `STRATA_PREFILL_TIMING=1`, fresh service per arm:

| Arm | 2,071-token request | 51,746-token request |
|---|---:|---:|
| tq6, reader off (baseline) | 3.400 s | 40.357 s |
| tq6, reader on | 3.194 s | 31.721 s |
| tq6, reader off (control) | 3.549 s | 40.253 s |
| tq6, reader on again | 3.105 s | 31.852 s |
| int8 control | 3.025 s | 25.827 s |

The new reader cuts tq6's long-request wall time by 20.9–21.4% (about 1.27x throughput), but remains 22.8–23.3%
slower than int8. CUDA0's long-prompt attention phase falls from 8,238–8,246 ms to 4,308/4,325 ms; the inter-stage
wait reported under `embed+steps` also falls from 12,768–12,821 ms to 9,640/9,712 ms. Total GPU memory is unchanged
between the tq6 arms (15,618/15,608 MiB). These are limited measurements, not a general speed guarantee. The report
has the tq8 reader measurements and the qsa-attention totals.

The GPU parity checks pass for all four widths, including causal tails and shuffled pages, with zero Compute
Sanitizer memory errors. `conversation_snapshot_test` passes 6,709 checks and also has zero Compute Sanitizer
errors: real state allocation at every width (3, 4, 6 and 8 bits), a snapshot A/B/A round trip, a restore from
another width rejected on payload size, and the mixed INT8 draft layer. Both live opt-in runs pass the four
OpenAI scenarios; Hermes fails once on `list_jobs` and passes once. Quality parity is therefore not established,
and this remains opt-in. The daily service stays on int8. The CUDA full build passes; the CPU-only build passes
with 55 CTest targets passing and 4 skipping (59 in total). The full HIP build passes with hipcc 7.2.4 for
gfx1100, as a compile and link check: there is no AMD device here, so `kv_tq_parity` skips, the host-side
conversation validation passes, and no HIP TQ run was measured. `kv_tq_parity` passes on the CUDA build, and
its no-device exit code 77 is recorded as a skip in both builds. The HIP build needed four build fixes: a
backend-neutral runtime target instead of `CUDA::cudart` in CMake, a portable `std::bit_cast` half bit read in
the GEMM parity test, the test reference arm's stale warp launch replaced with the production scorer's grid,
and a per-device VMM record (the device id and allocation granularity are captured with the range; HIP uses
`hipMemCreate`/`hipMemMap`, and `STRATA_HIP_VMM=0` turns it off). Eighteen CUDA target regression tests ran
while the live int8 service held the cards: 12 passed and 6 ended on out-of-memory errors
(`qsa_slots_parity`, `qsa_slots_parity_lanecell`, `qsa_merge_daily_parity_default`,
`qsa_merge_daily_parity_daily`, `kv_stream_parity`, `kv_window_parity`). Those six failed at allocation and
do not establish a regression verdict; they are not claimed as passes. The SYCL compiler is not installed here, SYCL has no TurboQuant
implementation (its separate port is unchanged), and no AMD hardware was available for a validation run. Other
GPU architectures were not tested.

**Reproducible greedy output (0.1.30, opt-in, `STRATA_IQ_MT_MIN=1`):** with the IQ models, the CPU computes an
expert for one token with ggml's dot product and for several tokens with Strata's multi-token kernels, which round
slightly differently. How many tokens share an expert depends on the drafts in a verify window, so the same prompt
at temperature 0 can end in a different (equally good) answer when the drafting, the cache state or a resumed
conversation differ (issue #152). `STRATA_IQ_MT_MIN=1` (in the config's `env`) uses the multi-token kernels for
every group: the answer then no longer depends on the drafting. Measured on a Ryzen 7600 (AVX-512): IQ3_S decode
-1..-3%, the other models the same; the default stays the fastest rule. On a CPU with AVX2 but no AVX-512 and no
AVX-VNNI, the default instead serves IQ3_XXS, IQ3_S and IQ2_S one-token (NT1) gate/up rows with the AVX-2 kernel;
those rows are ggml's dot product bit for bit, so a lone token's rows no longer depend on the drafting there.
`STRATA_IQ_MT_MIN` set still overrides this default. On an Intel CPU of Alder Lake or later
without AVX-512, where the AVX-2 kernel gathers the IQ3_S grid, `STRATA_IQ3S_MT1=1` (opt-in) gives IQ3_S the multi-token
kernel for one token, which is the faster one there; it changes a lone token's rounding, so it is off by default. Through the server, two more things carry
over from one request to the next (#410): the adaptive tier moves experts between RAM and VRAM (the GPU and the CPU
round an expert differently), and the prompt cache resumes a repeated prompt and reads only its tail through the
decode path. For byte-identical repeats add `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0` to the engine's args
as well (#410): the PCIe share of the missed experts (computed on the GPU instead of the CPU) still made the first
answer after a start differ from the next ones. Measured here (IQ3_XXS, a 3.6K-token prompt, 4 repeats): with all
three switches 1 answer of 4, without `--pcie-frac 0` 2 of 4 (the first one differs), with the defaults 2 of 4.
`--pcie-frac 0` costs decode speed (the missed experts all run on the CPU), so keep it for A/B runs.

**Coupled drafts with Gumbel-max picks (opt-in, `STRATA_SPEC_COUPLED=1` and `STRATA_SPEC_GUMBEL=1`):** for a request
that samples (temperature above 0), `STRATA_SPEC_COUPLED=1` lets the draft layer sample its guesses with the target's own
chain and random draw instead of taking its most likely token. `STRATA_SPEC_GUMBEL=1` changes how both of them pick from
that chain: the token with the largest p / E, where E is exponential noise keyed by the seed, the position and the token
id (the Gumbel-max trick; it is still an exact sample of the same distribution). With the default pick, one random number
walked over the candidates sorted by probability, a draft and a target whose candidate lists differ in one token tend to
land on different tokens; with noise keyed by the token, a token gets the same noise on both sides, so they agree on the
tokens they share. Greedy requests are unchanged, and without the variables nothing changes. Measured on a Ryzen AI
Max+ 395 (Radeon 8060S, Linux, ROCm 7.14.1, the iGPU alone), UD-Q4_K_XL, `--spec 4 --mtp` (the base model's draft
layer), temperature 1.0 / top_p 0.95 / top_k 20, a 1.3K-token prompt and 512 output tokens, 12-13 requests per arm:
drafts accepted 52.8% -> 59.9%, tokens per verify window 2.65 -> 2.88, output 41.1 -> 44.9 tokens/s (+9%); a window
costs the same (draft 8.8 -> 9.1 ms of 64). `sampler_parity` checks the pick against a host reference on every sampled
path, and that its frequencies match the softmax. On an RTX 3060 (IQ3_XXS, `--spec 4 --mtp`, temperature 1.0 / top_p 0.95 / top_k 20, 12 interleaved pairs of 200-token story and code requests) `STRATA_SPEC_COUPLED=1` raised the accepted drafts from 62.1% to 66.9% and adding `STRATA_SPEC_GUMBEL=1` to 68.0%, but decode speed did not move beyond the run-to-run spread (median 43.3 tok/s with coupled alone, 42.8 with Gumbel as well): try it on your own card before relying on it.

**The draft layer's tokens (0.1.27, `--draft-vocab`):** the MTP draft layer can only propose tokens from a subset
of the vocabulary (`mtp/rt/draft_vocab.bin`). Since 0.1.27 the subset includes every Chinese, Japanese and Korean
token (106,299 ids), so answers in those languages are 15-38% faster (Q2_0, RTX 5070). Its head takes ~180 MiB of
VRAM, which the expert cache leaves free for it (0.1.28). `START-HERE.bat --setup --draft-vocab en` keeps the
English/code subset from before (40,525 ids, ~110 MiB less VRAM, English answers 1-2% faster; CJK answers get
almost no drafts). `--draft-vocab cyrillic` takes the English/code subset plus the whole Cyrillic script (58,963
ids): the shipped subsets hold 142 of the vocabulary's 18,580 Cyrillic tokens, so Ukrainian or Russian answers got
1.4 tokens a round; with it 2.1, and 83 -> 109 tokens/s (RTX 5090, the NVFP4 fork), English unchanged.
`--draft-vocab fr` (0.1.39, #597) takes the English/code subset plus the 5,686 tokens that cover 99% of a French
Wikipedia corpus (46,211 ids, `tools/draft_vocab.py --corpus`): 23.6% of French text's tokens were outside the
English/code subset, 0.7% are outside this one. Drafts accepted in French answers 0.51 -> 0.60 (IQ3_XXS, RTX 5070,
8 prompts x 2 passes; English 0.61 -> 0.63 and code 0.77 -> 0.78, no loss), and 141 -> 158 tok/s in French on an
RTX 5090 (IQ3_S, the reporter's measurement).
`tools/draft_vocab.py` builds and inspects subsets. When the start stops with "the draft head does not fit" (a
12 GB card with a long context, #474), the engine says how much the head needs, how much VRAM is free and which
smaller subset fits, and the server's start error repeats it; setup suggests `--draft-vocab en` on cards under
14 GB (only a suggestion: nothing changes unless you pass it).

**Serving without the draft layer (`--mtp` is optional):** `serve` runs without `--mtp`. Drafts then come from the
suffix/prompt-lookup drafter only (or one token per round), every token is still verified against the model, and the
draft layer's VRAM (~0.7-1 GiB with its head) goes to the expert cache: 1,000 -> 1,678 slots in one A/B on an 8 GB
card. The conversation cache (`--conversation-cache-mib`) stays on: a parked conversation then carries no draft K/V.

**Low-RAM mode (engine 0.1.26, chosen by setup):** normally all of a model's experts are copied into RAM (23-50 GB,
pinned) and the GPU holds a copy of the most-used ones. On a PC whose RAM cannot hold them beside the system (the
experts plus ~10 GB), setup instead maps them from one file in the model's folder (`--mmap-experts`, the pack's
`experts.bin`, +23-50 GB of disk). The OS file cache holds what the GPU does not, and it can give that memory back.
On the Coder the engine's committed memory drops from 36 to ~13 GB, with the same answers. With a big GPU (an RTX
5090 holds all of the Coder's experts, most of Q2_0's) it runs at nearly the usual speed. With a small one, most
experts come from the SSD and it is much slower (setup says so). `START-HERE.bat --setup --low-ram on|off` overrides
the choice.

**Low-RAM mode, resident (engine 0.1.30):** when the experts the GPU does not hold fit the RAM (with the same ~10 GB
beside them), setup picks the resident variant instead (`--resident-experts`): at start the engine copies exactly those
experts from `experts.bin` into RAM (page-locked when the driver allows, else locked in RAM), so while it answers
nothing is read from the SSD, however little RAM the OS leaves for its file cache. Examples with setup's context: a
32 GB PC with a 24 GB GPU runs Q2_0, IQ2_XS and the Coder this way (~16-18 GB of experts in RAM, the GPU holds the
other ~18 GB), a 32 GB PC with a 12-16 GB GPU the Coder; IQ3_XXS on a 32 GB PC stays mapped. The details:
- The prompt path borrows room in the GPU's expert cache for its buffers and puts those experts back after the prompt;
  as far as the RAM allows, their experts are kept in RAM too (so a prompt reads nothing from the SSD either).
- The cache still follows the conversation (`--adapt-every`): a swap copies the evicted expert back from VRAM into the
  RAM place of the one that replaces it, so the RAM copy keeps holding exactly what the GPU does not.
- `STRATA_RAM_ADAPT=<n>` (opt-in, `--serve`, one GPU): the RAM copy follows the conversation too. Without it the
  copy is chosen once, at start, so when the RAM cannot hold every expert the GPU does not, an expert outside the copy
  is read from the SSD every time the conversation routes to it. With it, before a decode window and while no adaptive
  round is in flight, up to `n` experts the windows keep routing to (the adaptive tier's decayed count, 2 or more) take
  the RAM places of experts of their own layer that are not routed to (count below 0.5): one pair per layer, the
  largest difference first, one blob read each. The RAM stays the same size; the routing, the weights and the VRAM
  tier's rule are untouched, and an expert that left the copy is read from the file again when it is needed. It needs
  the adaptive tier (`--adapt-every`), and does nothing with `STRATA_EXCHANGE_ROTATE=1`, a layer split, a peer or
  helper GPU, or `--batch`. Prompt routing is not used to choose replacements, and no selection runs during prefill.
  Later prompts can read from a RAM copy changed by earlier decode requests. The log reports how many experts moved.
  Measured on an RTX 3060 12 GB with 32 GB of RAM (Linux, IQ2_XS, 20.5 GiB of experts in RAM and about 6,000 experts on
  the file tier, `--adapt-async 1`, `--spec 3`, power-saver), against v0.1.43: 47 frozen Korean requests from
  my own use (tools, follow-ups and background tasks, 281-33,241 prompt tokens), replayed in order in fresh engines:
  release, this option, release. Each starts from the same held learned profile; the private prompts are not attached.
  With default tail skip (7), decode is 43.26 -> 50.40 tok/s (+16.5%) against the closing release, computed as total
  generated tokens / total decode time; the two release arms are within 0.2%. Short/long new-input requests gain
  16.3%/16.9%; total prompt speed differs by 2.5%, within run variation. Decode file-tier demand reads per generated
  token fall from 14.15 -> 2.67; 8,056 successful RAM promotions read another blob each, giving 3.16 with those added.
  These are file-tier counts, not physical SSD I/O. RAM residency does not exempt an expert from tail skip. One machine,
  one quantization; tail skip off, blocking adaptation, `--spec 2` and the soak were not repeated on v0.1.43.
- `--adapt-async 1` (opt-in, `--serve`): the swaps of a round advance between decode windows on a helper thread
  (copy back, copy in, move into RAM) instead of one window waiting for the whole round. With `--batch` slots it
  advances between their windows too (docs/BATCHING.md); not with `--batch-groups` or `--peer-device` (the blocking
  tier runs there). With `--pipeline-windows 2` (two windows in flight, see
  [MULTI_GPU.md](MULTI_GPU.md)) the copies into the evicted slots and the moves into RAM also wait until the windows
  that were in flight when they were decided have completed (`STRATA_PIPELINE_ADAPT_ASYNC=0`: the blocking tier
  there). It is not bit-exact from run to run: which window first computes a swapped-in expert on the GPU (which
  rounds differently from the CPU) depends on when its copy lands.
  It stays on the blocking tier (said in the log) when the exchange buffers are not page-locked; the stats line
  reports ms per round.
- **The arena too** (one GPU, no layer split): with every expert in RAM `--adapt-async 1` runs the same tier copy-in only:
  nothing to copy back or move into RAM. The evicted slot is marked non-resident before the round's copies are posted, and an
  expert becomes resident only once its copy has landed; `STRATA_ADAPT_ASYNC_WAIT=1` (the default there) waits for the round at
  the next tick, so the window an expert turns GPU-resident does not depend on timing. The pick runs on the host thread
  (`STRATA_ADAPT_PICK_INLINE=1`, one round per window); `STRATA_ADAPT_ASYNC_BATCH=N` posts at most N copies per tick (default 0
  = all; 16 measured slower). Every K/V grow and the `VRAM` command land the round in flight first. RTX 3090 + Ryzen 5 5600X,
  cold prompts: x1.07-1.11 decode, 12 of 12 prompts faster
  ([report](../bench/results/2026-10-11-community-rtx3090-5600x-arena-async/README.md)).
- **Pin by need** (`STRATA_PIN_BY_NEED=1`, opt-in, the arena on one GPU): Windows page-locks about half the RAM at most (the
  WDDM shared budget), so the arena registers its first ~30 GiB and the layers past it get no PCIe share. With the variable
  every blob gets a page-aligned stride (+0.15 % RAM) and the expert profile's non-GPU-resident experts are registered by rank
  instead, up to `STRATA_ARENA_PIN_GIB` (29 by default on Windows). `STRATA_PCIE_MIN1=1` sends one miss over PCIe as soon as a
  layer has two (the share otherwise rounds to 0 at 1-2 misses). RTX 3090 + Ryzen 5 5600X: prompt read +5-7 % at 56K-117K
  tokens; decode level alone, +2-3 % with `--adapt-async 1` and `--pcie-frac 0.6`
  ([report](../bench/results/2026-10-11-community-rtx3090-5600x-arena-pin-by-need/README.md)).
- **Pinned-memory headroom** (Windows CUDA, on by default; `STRATA_PIN_RESERVE_MIB=N`, `0` = off, sets it on any OS): the arena registers
  slice by slice until the driver refuses one, and on a 64 GB PC that left no page-locked memory at all for the few small buffers the
  engine allocates after it (the verify window's staging: `verify: mapped staging allocation failed`, exit 1, on Swift 1.5 IQ2_XS). The
  engine now holds 640 MiB (given back as soon as the arena is placed) and 160 MiB (kept until a required buffer is refused, or READY) while
  the arena pins, so the arena stops about one layer earlier; a refused required buffer gives both back and tries once more. If that still
  fails the error ends with `[pinned-memory-exhausted]` and names the `STRATA_ARENA_PIN_GIB` to start with. Linux has no such ceiling
  (the NVIDIA driver pins past `ulimit -l`), so the reserve is off there. Test knobs: `STRATA_TEST_PIN_BUDGET_MIB=N` makes the engine's own
  pinned allocations fail beyond N MiB, `STRATA_TEST_PIN_LEFT_KIB=N` lowers that ceiling after the arena to what is charged plus N KiB.
- `STRATA_EXCHANGE_ROTATE=1` (opt-in): an adaptive swap hands buffer ownership over instead of copying the evicted
  blob into the RAM copy (fully page-locked copy; layers of different blob sizes keep spare buffers per size). Same
  tokens, fewer host copies; it works with `--adapt-async 1` too. Swift 1.5 IQ3_S on an RTX 5090 with 64 GB RAM
  (Windows, about 30 swaps per verify window): 164 -> 188 tok/s on one forced 1,500-token text, 155 -> 178 over 23
  coding-agent turns. Details and the measurements: [EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md). Measured by us on an RTX 3060 12 GB (IQ3_XXS, resident RAM, default flags, 5 interleaved pairs): decode 58.96 -> 62.03 tok/s (+5.2%, 5/5 pairs), prompt speed unchanged; tokens identical with real adaptive swaps (Tesla P100, resident, blocking adaptation). It adds about 1.4 GiB of page-locked RAM, so leave it off when the RAM is already tight.
- The answers are the plain mapped mode's for the same expert placement: the bytes are the file's. With a page-locked
  copy the GPU also takes its usual share of the misses over PCIe (`--pcie-frac`), as with enough RAM; `--pcie-frac 0`
  (or `STRATA_RESIDENT_PIN=0`) gives the mapped mode's exact tokens.
- The engine leaves 4 GB of the RAM it finds free (`STRATA_RESIDENT_HEADROOM_GIB`); when even the experts the GPU does
  not hold do not fit, it says so and runs the plain mapped mode. The server log shows, per request, how many expert
  reads went to the file (`resident RAM: ... blob reads from the file`: 0 in steady use).
- `--low-ram resident|mmap` forces one variant (also on a PC with enough RAM, e.g. to try it).
- Several GPUs (#364, #384): setup recommends one GPU in the low-RAM mode (the resident variant has no layer split
  yet), and asks; `--gpus 0,1` (or answering 2) shares the model across them with the mapped variant
  (`--mmap-experts`): the cards together hold more of the experts, and two users measured it 1.3-1.6x faster than
  one card, but the OS file cache can fill the RAM to 0 free during long prompts. `--yes` keeps one GPU. A config
  with `--resident-experts` started with `--gpus` switches to `--mmap-experts` with a note, and the engine runs that
  pair as `--mmap-experts` with a warning instead of refusing it. **Since 0.1.40 (#642, #848)** setup requires an
  engine that runs the resident variant on a split, so it keeps both cards, resident, as for any config:
  [MULTI_GPU.md](MULTI_GPU.md#using-it).

**A mapped arena for small RAM (Linux, opt-in, 0.1.39, PR #640):** `STRATA_ARENA_MMAP=1` maps a native pack's expert
arena read-only from the pack's `experts.bin` instead of reading it into locked RAM, for a PC whose GPUs hold most
experts but whose RAM is small (2x 16 GB GPUs with 32 GB of RAM: ~1 GB -> 25 GB available while serving). The first
start writes `experts.bin` (when the drive has room for it), later starts map it; the pages of the experts a GPU holds
are handed back to the OS. Run it with `--pcie-frac 0` (the GPUs get no mapped alias). Without the variable nothing
changes.

**Releasing mapped expert pages on Windows (opt-in):** `STRATA_FILE_RELEASE=1` lets `FileExpertSource` trim the
full file-backed pages of experts after their GPU uploads complete, including the slots lent to prefill and then
refilled. It works with `experts.bin` and the direct GGUF views; shared boundary pages and private/pinned buffers
are left alone. Unset or `0` keeps the previous behavior. On one 32 GB Windows 11 laptop with an RTX 4080 Laptop
and a Thunderbolt RTX 3090, IQ2_XS with `--mmap-experts --layer-split 12 --trim-stage-weights` raised median available
RAM from 0.51 to 12.19 GiB across 1K/4K/16K prompt trials, with 3.7-4.4% lower prefill throughput. It did not reduce
committed memory or physical SSD reads. This is a working-set hint, not an unmap or a guarantee that the OS drops
its file cache. See the [configuration, measurements and limits](../bench/results/2026-10-04-windows-mapped-release/README.md).

**Low-RAM mode without `experts.bin` (engine 0.1.31):** for the native packs (IQ2_XS, IQ3_XXS, IQ3_S, the Coder, Swift,
Q2_0 packed by `tools/iq_pack.py`; not the canonical Q2_0 pack setup makes for AVX-512 CPUs) the mapped mode no longer
needs the pack's `experts.bin`: when the pack has none, the engine
maps the model's GGUF files themselves and reads each expert's gate, up and down rows from where `native_experts.txt`
says they are (the files are checked against it first: every tensor's name, type, shape, offset and bounds). That
saves the 23-50 GB copy on the disk. The answers are the same: on the Coder, 64 greedy tokens from `experts.bin` and
from the GGUF gave identical tokens and logits. An expert read from the GGUF is three reads instead of one, so the
engine fetches a layer's missing experts on 8 threads (`STRATA_FETCH_THREADS`) with one batched page request
(Windows `PrefetchVirtualMemory`). With an `experts.bin` in the pack, nothing changes. Setup does not use this yet.

**Linux file tier, 0.2.0 defaults (zero-copy page-cached experts, io prefetch, the decode gate):** when experts are read
from a GGUF read in place (the native packs, also with `--mmap-experts`), Linux now runs three things by default (an
`experts.bin` pack, such as the NVFP4 ones, keeps them off unless you set them: not measured there yet).
(1) **Zero-copy** (`STRATA_COLD_OVERLAP=4`): an expert of a GGUF read in place whose three tensors' pages are all in the page
cache (mincore) is computed straight from the mapping; before, every such expert was first assembled into a staging buffer
(a copy of ~1.5 MB per Q2_0 expert, on the decode's critical path). Same bytes, same rows, same output (identity gates
IDENTICAL). (2) **io prefetch** (`STRATA_IO_PREFETCH=1`, the path described below): whole-blob `pread`s and the router-
predicted experts of the next layers read ahead. (3) **the decode gate** (`STRATA_IO_PF_GATE=1`): every 16 decode windows the
decode's read-ahead is judged - it runs only while the page cache has room for the experts (the memory available, the
tightest cgroup limit included, over the experts' bytes: off below 0.55, on above 0.65, `STRATA_IO_PF_GATE_LO` / `_HI`) and
while it pays (at least 128 MB read ahead in 16 windows and 10% of it used, `STRATA_IO_PF_GATE_USEFUL`; a probe every 256
windows when it closed for that). Closed, the decode reads as it did without io prefetch (the look-ahead predicts one layer
and warms with an madvise hint); the prompt path keeps its preads either way. Measured on an RTX PRO 5000 emulating a 12 GB
card (`--expert-cache 4448`, Q2_0 GGUF in place, page cache dropped before each engine, RAM limited by a cgroup), the 0.2.0
defaults against the 0.1.4x ones, 3 interleaved pairs, story / code / doc8k decode and 4K / 20K prompts: 32 GB of RAM
1.70 / 2.56 / 2.88x decode (34.0 / 28.0 / 21.7 -> 57.6 / 71.7 / 62.6 tok/s), 1.54 / 1.91x prompts; 16 GB 1.16 / 1.25 / 1.25x
decode, 1.63 / 1.68x prompts (the gate stays closed there); `--resident-budget-gib 18` in 32 GB 1.16 / 1.10 / 1.30x decode,
1.55 / 1.91x prompts; greedy token ids equal on story and code (doc8k varies run to run with the adaptive tier in both). On a
real RTX 3060 12 GB (32 GB cgroup, 3 pairs) zero-copy alone gave 1.14 / 1.44 / 1.47x decode, 1.06 / 1.02x prompts. With them,
the container RAM check counts a cgroup's clean page cache (active and inactive) as available, as MemAvailable does on a bare
host: before, the 18 GiB RAM tier in a 32 GiB cgroup re-checked "1.2 GiB available" once io prefetch's startup reads had left
their cache in the group, and its file tier switched to unbuffered reads (which also left zero-copy off). Off switches: `STRATA_COLD_OVERLAP=0`, `STRATA_IO_PREFETCH=0`,
`STRATA_IO_PF_GATE=0`; the startup log names what is on. Windows and macOS are unchanged.

**Linux file-tier I/O path (`STRATA_IO_PREFETCH=1`; 0.1.40.2, on by default on Linux since 0.2.0 - see above):** the mapped file tier reads experts through
page faults. With this on, the layer's uncached experts and the router-predicted ones of the next layer (`STRATA_LOOKAHEAD_K`,
`STRATA_IO_PREFETCH_DEPTH`, default 2 when on) are read by I/O threads (`STRATA_IO_PF_THREADS`, default 8) with whole-blob
`pread`s into the page cache, and the CPU kernels read the mapping as before. The bytes are the same, so the output is the
same (greedy IQ3_XXS and Q2_0 hashes identical, prefetch off vs on). `STRATA_IO_PF_STAGE=1` instead stages every uncached
expert in a buffer and makes the layer wait for them; `STRATA_IO_PF_AHEAD=0` keeps only the demand reads. The per-request
log line "file tier I/O" shows what the OS read from the drive (/proc/self/io, major faults) against the engine's expert
reads; `STRATA_IO_STATS=1` (or prefetch on) adds the page-cache hit / miss split (mincore) and the read-ahead used / unused counts.
Measured (interleaved A/B, 10 pairs, 200-token greedy medians, page cache dropped before each run, memory limit by cgroup):

| box, lane | default (fill) | `STRATA_IO_PF_STAGE=1` |
|---|---|---|
| RTX 3060, 32 GB, IQ3_XXS | -3.4% | **+25.7%** |
| RTX 3060, 16 GB, IQ3_XXS | -1.8% | -32.6% |
| Radeon 780M iGPU, 16 GB, Q2_0 (adaptive tier off) | -0.7% (story +3.8%, code +4.1%) | -19% |

With the experts warm in the page cache (46 GB on the 3060, 60 GB on the iGPU box) the file tier already runs at the
resident speed, so there is nothing to win; the gap is the cold start and the low-RAM lanes. So it is off by default, and
the staging variant is worth trying only on a 32 GB-class box. **iGPU caveat:** on a Radeon 780M, prefetch together with the
adaptive tier (`--adapt-every`, on by default) under memory pressure reset the GPU in about 7 of 9 runs (the engine
prints a warning and runs as asked); with `--adapt-every 100000` there were no resets in ~40 runs. The cause is not found.

**A RAM budget (engine 0.1.31, `--resident-budget-gib N`):** the resident variant for a model whose experts do not all
fit: the N GiB of experts the GPU cache does not hold that the expert profile ranks hottest are copied into RAM at
start (locked; page-locked when the driver allows the whole budget), and the rest are read from the files.
It implies `--mmap-experts` and leaves 4 GiB of headroom. On Windows, available commit capacity also
limits the budget; a larger N is clamped to the smaller limit less 4 GiB and a 256 MiB margin, with a message.
A clamped budget no longer fails the safety check that follows (#403). A budget that cannot be kept at all is a
warning, with every expert read from the files. Setup sets N with `--resident-budget-gib N`. When the budget has room
left after the ranked experts, the slots the prompt path borrows keep their experts in RAM too, as they do without a
budget (the engine prints `the prompt path's lend region: N of its M slots keep their experts in RAM too`); before, a
budget always left them out, and every prompt read the lent slots' experts from the model files again (RX 6900 XT,
UD-Q4_K_XL, budget 71 GiB with 7.5 GiB unused: 0.5-2.6 GB of file reads per agent turn and 1.9-2.8 GB from the SSD
per fresh 8K-16K prompt went to zero; fresh prompts read 3-8% faster, the prompt path's host staging fell from
0.8-2.1 s to 0.06-0.5 s per request). With
the GGUF read in place it also warms the next layer's likely experts: while the CPU works on a layer, a thread applies
the next layer's router to this layer's input and asks the OS for the pages of the predicted experts that neither the
GPU nor the RAM budget holds (only pages - the experts computed are the same; `STRATA_LOOKAHEAD=0` turns it off). This
is what runs [Unsloth's UD-Q4_K_XL](UNSLOTH_Q4.md) (72 GiB of experts) on a 64 GB PC: 7-8.5 tokens/s at N = 40 on an
RTX 5070, against ~3 tokens/s before these changes.

**Opt-in switches measured on a hybrid Intel CPU under Windows (Core i7-13700K, 8 P + 8 E cores, RTX 5090, 64 GB RAM,
Windows 11, Swift 1.5 IQ3_XXS at 128K):**
- `--pool-affinity measure` (hybrid CPUs, native packs): the `all` and `p-cores` layouts are timed at start on the
  pack's own experts (about a second) and the faster one is taken. Engine 0.1.41 built from source, the same prompt:
  `all` (15 workers) spent 6.7 ms per verify window in the CPU pool and decoded 148 tok/s, `p-cores` (7 workers) 2.2 ms
  and 188-191 tok/s; in another run of the same day `all` took 2.4 ms, so the difference comes and goes. The
  measurement is taken once, at start: over four server starts of one day it read `all` at 14.8, 21.9 and 25.3 ms
  against 6.1 to 6.2 ms for `p-cores` and took `p-cores`, and once read `all` at 6.6 ms against 8.6 ms and took `all`.
  It does not look again during a session, so where the E-cores' speed changes while the engine runs,
  `--pool-affinity p-cores` is the steady choice. Not measured: other CPUs, Linux, AMD cards.

**Exact one-token expert rows (0.1.45, opt-in, `STRATA_CPU_IQ_ALL_EXACT1=1`, #1415 #1544):** below the #152 rule
(a lone token in an expert's group) an AVX-2 CPU without AVX-512 runs ggml-cpu's `vec_dot` for the i-quant gate/up rows.
With the switch, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S take Strata's AVX-2 decode instead, with ggml's float order
(the same per-lane integer sums, one fmadd per block, the format constant after `hsum_float_8`), so the output does not
change, only the time (`iq_avx2_parity`: 0 of 5,120 outputs per format differ from ggml's; the identity gate with the
switch on is 10/10 plus exact_pp 4K/20K identical on RTX 3060, 2x A4000 and R9700 boxes).  It is used only where
ggml-cpu itself is the AVX2 build (not an ISA-floor build, not AVX-512) and not with `STRATA_IQ_MT_MIN=1`.  Per expert,
gate/up, one thread: IQ3_S -32% on a Threadripper PRO 3975WX (Zen 2) and -11% on a Ryzen 5 5600X (Zen 3), IQ2_XXS -5..-6%,
IQ3_XXS -5% (Zen 2) / +1% (Zen 3), IQ2_S 0% (Zen 2) / +7% (Zen 3, slower); on a Core Ultra 7 265 (VNNI, gather) IQ3_S -60%,
IQ3_XXS -39%, IQ2_XXS -34%, IQ2_XS -24%, IQ2_S -3% on a P-core.  End to end, with the engine's decode timing: the CPU part of a
window falls 5% on a Xeon E5-2643 v3 + RTX A4000 (22.2 -> 21.1 ms of a 46.9 ms window, decode +0.5% over 10 interleaved pairs,
+2.5% on the story and code prompts) and on a Core Ultra 7 265 + RTX 3060 (8.2 -> 7.8 ms of 39.5 ms, decode +0.4%), and does not
move on a Threadripper PRO 3975WX + R9700 (14.5 ms, +0.4%, 5 pairs); no case reached +3% end to end, so the switch stays off.

**Switches added in 0.1.40 (all off unless noted; none changes the default output):**
- `--kv-grow` (or `STRATA_KV_GROW=1`; `--no-kv-grow` turns it off): allocate physical GPU K/V memory only for the
  context in use. The expert cache uses the remaining memory and releases slots when the context grows.
  The logical `--max-context` capacity does not change. With a layer split, each GPU stage manages its own
  K/V pools and expert cache. This option requires VMM support on each GPU, an expert profile, GPU-only K/V
  (no KV streaming), and all experts in RAM (not the resident low-RAM mode). It does not support remote experts,
  `--batch`, `--vram-elastic`, or `--peer-device`. The engine reports when these conditions disable the option.
  During an active series of requests, keep a recent K/V target of at most 65,536 cells.
  After 30 seconds without a completed request, return surplus K/V memory to the expert cache.
  The current request can still use the full logical context. Vision and reserve-pressure paths
  can reclaim memory immediately. This rule avoids repeated shrink, expert refill, and growth
  operations between short chat switches. The existing trim hysteresis still applies.
- `--host-core last` (or `STRATA_HOST_CORE=last`, Windows): the host thread runs on the last physical core and the
  workers take the first. Windows sends a GPU's interrupts to the first core, where a host spinning on the GPU's flags
  waits for them (`--host-core first` is the default; the startup log names the cores). It leaves a hybrid CPU as it
  is. `--host-core sibling` (or `STRATA_HOST_CORE=sibling`) keeps the host on the first core but on its other
  hardware thread (SMT), so the interrupts keep the first logical processor and the workers keep every core; it works
  on hybrid CPUs too, and is `first` on a core without SMT. On a 4060 Ti + 5080 layer split with an i9-14900KF it
  turned a decode that swung by ±6% from run to run into a steady one, 8% faster on average.
- `STRATA_ADAPT_LAG=2` (#764): a window takes the adaptive tier's swaps once they are two windows old (default 1,
  as 0.1.39). `STRATA_PREFILL_EQUAL=1` (#693): a prompt segment is read in chunks of equal size, not full chunks and a
  short last one (changes the rounding). `STRATA_OWNED_PRICE=exact` (#796): the cache sizing prices the prompt path's
  own buffers by their real allocation instead of 0.1.39's rule.
- `STRATA_Q2_BITPLANE=1`: a bit-plane row kernel for Q2_0 on CPUs with AVX2 and no AVX-512 (changes the last bits).
  `STRATA_NO_AVXVNNI=1` turns the AVX-VNNI forms of the i-quant, IQ4_NL and Q2_0 rows off on CPUs that have them
  (Alder Lake, Sapphire Rapids and later); they give the same bits as the AVX2 forms, 4-25% faster rows.
- Draft layer: `--lookup-chain-min M` (the shortest context match `--lookup-chain` extends, default 3),
  `--mtp-hnorm stream` (one norm per stream, as llama.cpp's MTP graph) and `--mtp-draft-vocab FILE` (a token subset
  of your own instead of `<mtp>/draft_vocab.bin`).
- Elsewhere: HIP `STRATA_DENSE_MMQ=1` and `STRATA_HIP_ADAPT_KERNEL_COPY=1` in [AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration);
  `STRATA_KEEP_EMPTY_TURNS=1` and `STRATA_TOPK_STREAM=0` in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).


**Read-ahead at start (Linux):** the weights, the native dense matrices, the GPU cache's fill from the profile, the
resident RAM copy and the MTP draft files are asked for ahead of their reads (madvise / posix_fadvise WILLNEED in
128 KiB steps), so the drive sees a deep queue instead of one page fault at a time. Measured on a Gen3 NVMe (RTX 5090,
32 GB, Q2_0 resident at 262K): ready in 70 s instead of ~920 s; the fill went from 39 MB/s to 3.2 GB/s.
`STRATA_READ_AHEAD=0` turns it off; `STRATA_FILL_AHEAD=N` sets how many fill pairs are asked for ahead (default 256).

**How much came from where:** with `--stats` the engine prints the tiers of the decode (`expert tiers`: blobs from the
RAM copy, blobs and MB from the files, the time spent reading them; `routing prefetch`: how many of the file reads had
been warmed). The server log has the same per request (`expert tiers: GPU ... hits ...; RAM ... blobs, files ...
blobs ... MB read`), and `GET /metrics` lists `ram_blobs`, `file_blobs` and `file_mb` for each recent request (with
engine 0.1.31 or newer). It also lists each request's speculative drafts, `drafts_offered` and `drafts_accepted`
(`null` when the engine did not report them), and their sums since the server started in `totals` (#457).
Its `hit_rate` is the VRAM share of the experts looked up while answering: experts the GPU reads over PCIe
(`--pcie-frac`) are not in it, so a higher `--pcie-frac` raises it even when decoding gets slower. `pcie_share`
(engine 0.1.39 or newer, #588) is their share of all routed experts, and the server log and the Monitor tab show it
beside the hit rate.

**Where a decode window's time goes (profiling, #610):** start the server with `STRATA_DECODE_TIMING=1` (and
`STRATA_VERIFY_PROFILE=1` for the GPU's side) in the environment. After each request the engine log then has one
`strata decode timing:` line - windows, tokens per window, and per window the verify time split into the wait for the
GPU, the CPU expert pool (plan, activation quantization, jobs) and the stage, plus commit and draft - and one
`strata decode GPU stages (ms/window):` line with the GPU time of each stage (GDN and QSA layers, the VRAM expert
hits, the router, the head, ...). The GPU profile times every stage with events, so it slows the decode a little:
use it to compare, not to measure speed. This works with every pack; `--gpu-stages` (a one-token replay of
per-layer graphs) refuses a native (IQ) pack, which has no such graphs.

**How good the next-layer expert prediction is (measurement only):** `STRATA_LOOKAHEAD_STATS=1` runs the router look-ahead on any expert tier and
scores it against the routing of the next layer: every 100 windows the engine log has a `strata lookahead stats:` line with the experts per layer
outside the GPU cache, the recall and precision of the top-k / k+4 / k+10 guesses and of the vote-ranked top 4 / 8 / 12, how much of the PCIe share they
would cover, and the predictor thread's time per window. It costs CPU (the predictor is a thread beside the expert pool), so use it to measure, not to
time. `tools/replay_cache_policy.py` replays a `--dump-routing` trace through the adaptive tier's swap policy offline.

Time to first token is prompt length / prompt speed: with Q2_0 about 4 s at 4K, 25 s at 32K, under 2 minutes at 128K
and 4.5 minutes at 262K (engine 0.1.13 made long prompts about twice as fast, below).

**Faster prompts (engine 0.1.13):** the prompt is read in chunks of up to 8,192 tokens instead of 2,048 (`--prefill
auto`: the largest chunk whose buffers fit in the expert-cache slots it borrows, and a request borrows only what its
prompt needs); the experts are multiplied by llama.cpp's quantized MMQ kernels instead of being expanded to FP16
first; the next layer's experts stream over PCIe while the current layer's attention runs; the PLE block runs for the
whole chunk at once; unpinned experts are copied by helper threads. Measured on the RTX 5070 12 GB, 64 GB RAM,
32K-token prompt: Q2_0 572 -> 1,290 tokens/s, IQ3_S 383 -> 1,208. Through the server (Q2_0, 128K context): 999 tokens
353 -> 438 tokens/s, 6,927 tokens 529 -> 1,077, 28,584 tokens 584 -> 1,249. Output speed is unchanged. Needles 5/5
(1K-262K). Details and the quality check:
[`bench/results/2026-09-28-prefill-speed`](../bench/results/2026-09-28-prefill-speed/README.md). Existing installs
switch to `--prefill auto` the next time START-HERE / setup.sh starts them. The raw numbers:
[`bench/results/`](../bench/results/). The [paper](paper/Strata-Paper.pdf) explains every number.

## Other GPUs (estimated)

Not measured - estimated from the runs above (same CPU and 64 GB RAM): the GPU part scaled by memory bandwidth, the CPU
part by how many more experts the card's VRAM holds. Treat as **±20%**. Numbers are *prompt / output* tokens/s.
The prompt figures predate engine 0.1.13, which about doubled prompt speed on the measured card; how much of that a
card gains depends on its PCIe link (the experts stream over it), so they are still the older estimates.

| GPU | Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti 16GB | Q2_0 | ~341 / ~80 | ~472 / ~87 | ~501 / ~81 | ~492 / ~72 | ~476 / ~62 | ~435 / ~53 |
|  | IQ2_XS | ~291 / ~80 | ~406 / ~77 | ~434 / ~63 | ~426 / ~62 | ~413 / ~51 | ~383 / ~47 |
|  | IQ3_XXS | ~249 / ~66 | ~359 / ~65 | ~381 / ~56 | ~374 / ~54 | ~363 / ~45 | - |
| RTX 3090 24GB | Q2_0 | ~355 / ~128 | ~491 / ~140 | ~521 / ~130 | ~512 / ~115 | ~495 / ~100 | ~453 / ~85 |
|  | IQ2_XS | ~303 / ~131 | ~422 / ~128 | ~451 / ~103 | ~444 / ~102 | ~430 / ~85 | ~398 / ~78 |
|  | IQ3_XXS | ~260 / ~106 | ~374 / ~103 | ~396 / ~89 | ~390 / ~85 | ~378 / ~71 | - |

More VRAM matters more than a faster GPU: every extra GB holds ~700 more experts, and every expert on the GPU is one the
CPU does not have to compute. A 3090's 24 GB takes most of the CPU work away. (Since 0.1.14 the expert profile ranks
all 24,576 experts; before, the cache stopped at 8,000, about 10-14 GB. `tools/make_profile.py` builds a profile from
your own prompts: run the engine once with `--dump-routing trace.bin`, see the tool's help. The shipped profile already
ranks every pair, so a trace only changes the order with `--reorder` (0.1.39, #587): your trace's pairs first, then the
base's, then the rest.)

## Which model?

All three are [ISTA-DASLab's GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

| Model | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | ~34 GB experts + ~6 GB | fastest | good |
| **IQ2_XS** | 68 GB | ~36 GB experts + ~6 GB | close to Q2_0 | a bit better |
| **IQ3_XXS** | 76 GB | ~43 GB experts + ~6 GB | slower (more CPU work) | best |

With 64 GB of RAM all three fit (close the browser for IQ3_XXS, and keep its context at 128K or less). With 48 GB only Q2_0 / IQ2_XS may fit. With 32 GB: the Coder (below).

### Or: the Coder (half the experts, for code)

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** is
ISTA-DASLab's expert-pruned release: 256 of each layer's 512 experts are kept (still 10 active per token), chosen with
RCO on code, agentic and vision calibration data; its authors report 91.3% of the full model's SWE-bench Verified and
98.7% of LiveCodeBench v6. One size, named IQ1_M for its 1.89 bits per *original* parameter; the kept experts are
stored like IQ3_S (IQ2_S-IQ4_XS gate/up, IQ4_NL/Q2_0 down). Shard 1 is 29.6 GB (experts: 23 GB of RAM), so it runs
on **32 GB of RAM**, and at 262K on 64 GB. Its shard 2 and its vision encoder are the original's files: with the
original installed, setup downloads only shard 1. Strata ships its expert profile (`data/expert-profile-coder.bin`,
the shipped ranking mapped onto the kept experts through the release's `rco-allocation.txt`: 72% of the expert
reads hit the GPU on a 12 GB card). Images work; the experimental speed projection loads and runs on it (it was made
for the full model).

```
START-HERE.bat --setup --family coder
```

### Or: Swift 1.5 (a fine-tune that thinks shorter)

The setup's first question also offers **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**,
UkisAI's fine-tune of Qwen3.8-Flash-Next, trained to reach the answer with much less thinking (its authors: 63% fewer
thinking tokens, 1.8x sooner answers, under 1% accuracy loss). Same architecture, the original's sizes except Q2_0,
its own
vision encoder; Strata runs it at the same speed (4K, IQ2_XS: 465 prompt / 78.7 output tokens/s, vs 467 / 78.3 for
the original). Its authors recommend **IQ2_XS** (their Q2_0 is marked experimental). Its license is the Swift Open
License 1.0 - read it on the model page.

Our small check (8 reasoning questions, default thinking, IQ2_XS): both models got **8/8**; Swift used **1,234**
output tokens in 28 s, the original **2,682** in 46 s - most of the difference from one question the original
thought about for 1,524 tokens. Not a benchmark, but consistent with the claim.

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

### Experimental: Unsloth's UD-Q4_K_XL

A 4-bit quantization of the same model (111 GB, 72 GiB of experts). Setup offers it from engine 0.1.32
(`--family unsloth --model UD-Q4_K_XL`: a RAM budget of your RAM less 24 GB, the rest read from the SSD); what it
does and the manual workflow are in **[docs/UNSLOTH_Q4.md](UNSLOTH_Q4.md)**. On a 64 GB PC with a 12 GB RTX 5070 it writes 7-8.5
tokens/s, most experts read from the SSD; it picks the same tokens as llama.cpp on the same file at 97.5-99% of
the positions of short greedy answers, 90-91% after a 16K prompt, differing mostly at near-ties
([measured](UNSLOTH_Q4.md#quality-against-llamacpp-on-the-same-file)).

## Before you start

You need **only a graphics driver**: NVIDIA 580 or newer (update it with the NVIDIA App or from
[nvidia.com/drivers](https://www.nvidia.com/drivers)), or for AMD the one in [INSTALL.md](INSTALL.md#what-you-need).
Everything else is installed for you the first time.

| | |
| --- | --- |
| GPU | NVIDIA compute capability **7.0 or newer**, **12 GB VRAM or more** (8 GB runs, slowly). RTX 30/40/50 use the ready-made engine. Volta cards such as the 16 GB Tesla V100 compile locally with CUDA 12.x; Ampere-only optional kernels are disabled and prompt GEMM uses a portable FP32 fallback. Or AMD **Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700, RX 6800 / 6900 series**: [AMD_HIP.md](AMD_HIP.md). |
| RAM | **64 GB** recommended (see the table above). |
| CPU | x86-64 with AVX2 (any Intel/AMD desktop CPU from the last ~8 years). AVX-512 (Ryzen 7000/9000) is a bit faster. Older CPUs without AVX2 are experimental and slow: [Older CPUs](INSTALL.md#older-cpus-experimental). |
| Disk | ~70-80 GB free for the model, ~6 GB for the MTP layer (+1 GB with images). **Q2_0 on an AVX-512 CPU** also writes a one-time ~40 GB pack for the fast CPU kernel (34 GB of experts in its layout, plus the dense weights). An NVMe SSD is strongly recommended. |
| OS | Windows 10/11, or Linux (Ubuntu 22.04/24.04 and Debian 12/13 get everything installed automatically). |

What the first start installs: in this folder `.venv/`, `engine/` and `third_party/`; the model files (`models/`,
`packs/`, `mtp/`, 70-120 GB) in **`Strata-data` next to this folder**, so a new copy of Strata (an update unzipped
elsewhere) finds them and sets itself up the same way. The place is remembered per user (`%APPDATA%\Strata\settings.json`,
`~/.config/strata/settings.json`); `--data-dir` chooses another. Installs from before 0.1.16 are moved there by the next
start (a rename on the same drive; files on another drive are used where they are).
Python 3.12 if you have none (for your user account, no admin), a private Python environment, NVIDIA's CUDA libraries
(from pip, ~0.4 GB), the ready-made Strata engine for RTX 20/30/40/50, the model and the MTP draft layer. If no
ready-made engine fits your PC, it offers to install the build tools (Visual Studio Build Tools + CUDA Toolkit on
Windows, `build-essential` + CUDA on Ubuntu) and compiles the engine for your GPU (asks first; 20-40 minutes once).
CUDA 12.x is required when targeting Volta because CUDA 13 no longer emits sm_70 code.

---

## Windows

### Double-click `START-HERE.bat`

**The first time** it asks four questions and does the rest:

1. **Which model?** Qwen3.8-Flash-Next (the original) or Swift 1.5 (the fine-tune that thinks shorter).
2. **Which size?** Q2_0, IQ2_XS or IQ3_XXS (it recommends one for your RAM).
3. **How much context?** 8K to 256K tokens (it recommends one for your VRAM).
4. **Images?** yes / no (see [Images](#images-vision)).

Then it downloads and prepares everything (the model is 66-76 GB, so the first start takes a while; an interrupted
download continues where it stopped) and **starts the model**: your browser opens `http://127.0.0.1:8080`, the Strata
app. It has three tabs:
- **Chat:** streaming answers, the model's thinking (folded away once it answers), code with a copy button, pictures when
  images are on, and sampling and thinking-level settings. Chats stay in your browser.
- **Monitor:** what the model is doing (reading the prompt, with progress, or writing, at how many tokens/s); GPU load,
  VRAM, temperature, power and PCIe traffic; CPU, RAM and disk; the context in use; the last requests.

On a Mac (Metal backend) the Monitor's GPU load and VRAM come from IOKit's IOAccelerator nodes - the driver's Device
Utilization % and its in-use system memory, against the device's Metal working set - and the card says the rest: there
is no temperature or power reading without root (`powermetrics`), and an Apple Silicon GPU sits on the die, so there is
no PCIe link. The engine's own VRAM figures are in its log.
- **About:** the model and engine settings, and the addresses to connect other apps.

`http://127.0.0.1:8080/?q=your question` opens it with a new chat already asking. The API is at
`http://127.0.0.1:8080/v1` for your apps.

**Every time after that**, `START-HERE.bat` just starts the model (30-90 s to load 34-43 GB into RAM). Nothing is
downloaded again. Closing the window stops the model.

```
START-HERE.bat --setup                          install another model, or change context / images
SETUP.bat                                       the same (double-click it)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --gguf-dir D:\models\IQ2_XS       use GGUF files you already have
START-HERE.bat --data-dir E:\Strata-data         keep the model files somewhere else
START-HERE.bat --port 8081                      another port
START-HERE.bat --gpu 1                          another GPU (numbered as nvidia-smi; setup picks the one with the most VRAM)
START-HERE.bat --calibrate                      tune the engine for this PC (about 15-30 minutes, longer on a slow card), then start
```

With more than one model installed, it asks which one to start. `run-<model>.bat` starts a model directly.

**Tuning for your PC (`--calibrate`, engine 0.1.19).** These engine settings depend on the PC more than on the model:
- the share of the experts missing from VRAM that are copied to the GPU instead of computed by the CPU
  (`--pcie-frac`: a fast PCIe link and a slower CPU want more, a laptop's narrower link less);
- how sure the draft layer must be to add another guess to a check (`--spec-min-p`);
- how many CPU threads compute experts (`--pool-workers`: on CPUs with efficiency cores, fewer can be faster);
- how busily the VRAM expert cache follows the conversation (`--adapt-every`, `--adapt-swaps`, `--adapt-decay`): a PC
  whose CPU reads the missed experts slowly (DDR3 or DDR4 in few channels) gains from swapping more. On a Xeon
  E5-2673 v3 with DDR3 and an RTX 4060 Ti on PCIe 3.0 x8, 160 swaps every window measured +7.9% over the default
  (Q2_0, with the swaps taking effect a window later, #764); see #906;
- on two or more cards (a layer split), the prompt chunks that keep the cards reading at once (`--prefill-pipe`, the
  prompt read speed of fresh prompts rather than the output speed; see docs/MULTI_GPU.md).

The defaults were measured on a Ryzen 5 7600 with an RTX 5070. Setup offers to measure them on your PC after an
install; `START-HERE.bat --calibrate` (Linux: `./setup.sh --calibrate`) does it any time. It measures the output
speed with each setting and keeps one only when it is more than 3% faster. The result is remembered per PC and model
(in the settings file next to the data folder's record), so updates keep it.

Measuring the worker count needs a fresh engine, so the model is loaded more than once: the PC is busy, and can
stop responding for a minute or two, once per restart. It then starts the model, like a plain `START-HERE.bat`.

**Manual CPU task granularity.** `--pool-tasks N` sets the target total number of row tasks in each batched
CPU expert Gate/Up and Down phase, not the number of threads or tasks per expert. The default `0` keeps
three tasks per participating thread (including the host when enabled). Values `1..4096` are capped by
each phase's row count; native batches larger than the pool's capacity apply the target to each sub-batch.
For example, `--pool-workers 13 --pool-tasks 192` uses 14 participating threads with the default host worker.
More tasks can reduce imbalance between cores, but also add scheduling overhead: compare against `0` with
the same worker count and workload. This does not change kernels, phase barriers, or PCIe placement, and
does not affect the legacy single-token/oracle fallback. Setup's `--calibrate` does not tune it yet.
For the server, add `"--pool-tasks", "192"` to the existing `args` list in its configuration, then restart it.

**How many CPU threads the engine starts (`--pool-workers`).** Without the setting the engine picks the number of
CPU expert workers from the CPU, not from the count of logical threads: one worker per *physical* core (never an SMT
sibling) except the core of the main thread, one core less on a CPU with 17 or more cores, only the cores of the main
thread's CPU socket on a multi-socket machine, at most 32, no more than a container's CPU limit (docker `--cpus`) allows,
and on a hybrid CPU with more efficiency than performance cores the performance cores but one plus half of the efficiency
cores. When the OS does not say which logical processors share a core (some containers), it counts half of them. The start
line says what bounded the number (`CPU pool: 31 workers by default (32 physical cores, 64 logical processors; bounded by ...)`).
The workers wait by spinning, so a worker without a core of its own takes time from the others: on a 20-core CPU 24
workers decoded 44% slower than the default, on a 32-core EPYC 63 workers 35% slower. A count in the config
(`--pool-workers N`) is always used as given; when it is above what the CPU uses well the engine prints a plain warning
with the recommended number at start (`strata cpu pool: warning: ...`). Setup writes no worker count except the hybrid
CPU's, capped the same way, and gives the CPU vision encoder at most 16 threads (`--threads` in the `vision` entry of the
config changes that).

### Running it at startup (Task Scheduler)

To have the model up at logon, people start the serve from **Task Scheduler** (or a service). Beware: Windows
throttles such contexts, and the model's ~40 GB expert load then crawls at **~0.05 GiB/s (13-14 minutes)**
instead of **~1.4-1.5 GiB/s (~35 seconds)** - a 24x slower start. Measured on an RTX 5070 Ti + Ryzen 7 9800X3D
+ NVMe, same binary, same args, same cache state:

| How the serve starts | Expert load |
| --- | ---: |
| Double-click / terminal / SSH | 1.42-1.52 GiB/s (~35 s) |
| Task Scheduler with its defaults | 0.05 GiB/s (821-841 s) |
| Task Scheduler with the two settings below | 1.42 GiB/s (35 s) |

In the task's properties set both of these (the defaults are the opposite):

- **Priority level: Normal** (Options tab; the default is Below normal), and
- **Run with highest privileges** (General tab; without it the task runs with a limited user token - which
  also strips `SeLockMemoryPrivilege`, the privilege Windows large pages need).

(Both were changed at once, so the isolated effect of each is not measured.) If the model still starts
slowly, the engine prints a hint under its `loaded ... GiB at ...` line naming this cause.

**Large pages need a new logon (#1412).** Granting "Lock pages in memory" (`secpol.msc`, User Rights Assignment) to the account that runs the engine takes effect at the next logon: Windows puts the privilege in the access token when the session starts, so log off and on (or reboot) after granting it. Until then the startup log still says the large pages were refused. `tools\windows\enable_large_pages.ps1` (run as Administrator) grants it to the current account with `secedit`.

### Chat in the terminal (optional)

```
.venv\Scripts\python chat.py
```

---

## Linux

```bash
./setup.sh
```

The same questions, the same automatic install (it uses `sudo apt` for Python and, only if it has to compile,
for the build tools), and the same start: `http://127.0.0.1:8080`. Later runs of `./setup.sh` (or `./run-<model>.sh`)
start the model directly. Options as on Windows (`./setup.sh --setup`, `--model Q2_0 --yes`, `--gguf-dir /data/Q2_0`).
Terminal chat: `.venv/bin/python chat.py`.

- **Updating:** `git pull`, then `./setup.sh`: it compiles the engine again when its source changed (a minute or
  two for the changed files). If that compile fails, it says so and starts the engine you had.
- **Other distributions** (Arch, Fedora, ...): install the C++ compiler and the CUDA Toolkit 13 with your package
  manager first (Arch: `sudo pacman -S base-devel cuda`); setup finds `nvcc` on PATH, in `/usr/local/cuda*` and in
  `/opt/cuda*`, and does the rest.
- **WSL** works (Ubuntu 24.04 tested), with one limit: the NVIDIA driver pins only about 1 GB of RAM there, so KV
  streaming (`--kv-resident`) is off and the KV cache stays in VRAM, and the experts are copied to the GPU from
  unpinned RAM (slower prompts than native Linux).

### Starting the V100 IQ3_S server as a user service

The repository's `strata-v100.service` template starts the server as a user service on two Tesla
V100 cards. It is separate from the default Q2_0 setup above; use it only on the IQ3_S deployment.

Before the first start:

1. Make the checkout available at `~/strata-v100`. A symbolic link to the checkout is sufficient.
   Check that `strata-iq3_s-arena.json` exists in the checkout and uses the native IQ3_S expert
   pack, with `STRATA_SPLIT_RING=192` in its `env` object. Keep this configuration and the
   required `.strata-service.env` file private. Do not commit authentication keys.
2. Check that the model storage is mounted. The unit runs `/usr/bin/mountpoint --quiet
   /mnt/strata-models` before the server. When the mount is absent, the start fails and the unit
   retries after 10 seconds instead of loading a model from an empty disk path.
3. Install the template and reload the user manager:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp strata-v100.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   ```

4. Stop an old manually started server first, so that the new service does not fail on the port that
   the manual server owns.
5. Start and enable the service: `systemctl --user enable --now strata-v100.service`.
6. To start the service at boot before a login, enable lingering: `loginctl enable-linger "$USER"`.
   The command can need administrator rights; the privilege policy on the machine decides.

Daily commands:

```bash
systemctl --user restart strata-v100.service
systemctl --user status strata-v100.service
curl -fsS http://127.0.0.1:8088/health
journalctl --user -u strata-v100.service -f
```

The unit restarts the server after any exit, 10 seconds later, and keeps the model mount as a
precondition. A missing disk stops the start; it does not load an incomplete model.

---

## Sharing the GPU with other programs (optional)

By default the model stays loaded until you close Strata. On a PC that also games, renders or runs another model
server, three server options (all off by default; also as keys in `strata-<model>.json`) give the VRAM back:

| Option | Config key | What it does |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | unload the model after 600 s without requests; the next request loads it again |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | load an unloaded model only when that much VRAM is free (it waits up to 15 s for memory being given back), else answer **503** "the GPU is in use by another program" instead of starting into what a game left (with several GPUs it checks the first one) |
| `--before-load "cmd"` | `"before_load": "cmd"` or `["cmd", "arg"]` | a command run before the model is loaded again, e.g. one that unloads another server's model |
| | `"before_load_timeout_s": 120` | how long that command may run (default 120 s). When it exits non-zero or times out (a peer that is not running makes `curl` exit 7), the free VRAM decides: with `min_free_vram_mib` set and not enough free, the model is **not** loaded and the request gets **503** with `Retry-After`; with room (or no limit set) it loads, with a warning in the log. `"before_load_strict": true` always refuses on a failure; `"before_load_ignore_failure": true` always loads |

`POST /unload` unloads it now (`409` while a request is running; `POST /unload?wait=1` waits for the running turn first, up to 300 s, or `?wait=<seconds>`, so a second server can really free the card) and `POST /load` loads it ahead of a request (both
with `Content-Type: application/json`, e.g. `curl -X POST -H "Content-Type: application/json" localhost:8080/unload`);
`/health` says `"loaded"`, `/v1/models` lists it as `unloaded` (like llama.cpp's router), `/props` sets
`is_sleeping` and the Monitor shows the state. Unloading ends the engine process - and the image encoder, when images
are on; it is started again first, as at a start - so their VRAM and RAM go straight back. The model files stay in
the OS file cache, so loading again takes seconds while that RAM is not needed elsewhere. Measured on an RTX 5060 Ti
16 GB with Q2_0 in the low-RAM mode: unloading takes ~0.3 s, and a request to an unloaded model answered after
4.6 s (text) or 14.7 s (a picture, image encoder on the CPU).

**Giving part of the VRAM back while it keeps serving (#533, opt-in, one NVIDIA GPU; every request and answer:
[VRAM_ELASTIC.md](VRAM_ELASTIC.md)).** With `"vram_elastic": true`
in the config (the engine flag `--vram-elastic`), the expert cache is allocated in 512 MiB segments
(`"vram_segment_mib"`), and `POST /v1/vram` with `{"reserve_mib": 8000}` shrinks it between requests until that much
VRAM is free for another program; `{"reserve_mib": null}` grows it back towards its full size, keeping the reserve
the engine started with (`--vram-reserve-mib`), and `{"reserve_mib": 0}` takes all of it back. A request that is
running finishes first. The experts of the segments given back are computed on the CPU, like any expert outside the
cache, so answers keep coming, slower; growing back puts the same experts in the same slots. Nothing resizes on its
own. Not with a layer split, the helper caches, `--peer-device` or the resident low-RAM mode. Measured on an RTX 5070
12 GB with Q2_0 (4.8 GiB cache): `{"reserve_mib": 6000}` took 78 ms and freed 4.3 GiB (the cache keeps 0.5 GiB for
the prompt path), decode 44 -> 33 tok/s; growing back took 92 ms and the answers were token for token the ones before
the shrink. The segmented-cache behavior requires `--vram-elastic`.

GPU-only KV growth also supports `POST /v1/vram`, without `--vram-elastic`. This mode releases whole physical
chunks from each GPU's expert cache. It keeps the KV mappings, cache addresses, and prompt-loan region fixed.
After an external allocation is released, `{"reserve_mib": null}` maps available chunks back and fills the
new slots from the expert profile. It keeps the startup VRAM reserve on each GPU. Memory still used by KV
cannot return to the expert cache until the KV pools shrink. A requested reserve can fail if the remaining
cache chunks cannot supply enough memory.


**Keep what the expert cache learned across restarts (opt-in, engine 0.1.36, #477):** a start fills the GPU's expert
cache from the shipped profile, and the adaptive tier (`--adapt-every`) then moves in the experts your requests use.
With `"expert_profile_save": "expert-profile-learned.bin"` in `strata-<model>.json` the engine saves that as a
profile - the experts in VRAM first, then the routing it counted since the start, then the shipped order - on a
clean exit and every 10 minutes between requests (`"expert_profile_save_every": 5` for another interval, `0` for
exit only), written to a temporary file and renamed, so a crash never leaves half a file. The next start begins from
it instead of the config's `--expert-profile` when it is a profile of the same model (else from the config's, as
before). A relative path is in the Strata folder; one file per model, and a profile per project works the same way
(point the key at another file). The file is a fingerprint of what you used the model for: it stays on your PC.
Without the key nothing is counted or written. Setup rewrites the config when run again: add the key again then.
On an Intel Arc card with `STRATA_VERIFY_NO_HOST` (the SYCL default on Battlemage, no adaptive tier) the profile is ranked from the device's own routing counts and applies at the next start; a profile learned from one kind of text speeds that kind up (code, +13-19% decode on the Arc Pro B70, same tokens) and slows other kinds (prose, -18%), so keep one file per kind of work.

**The expert cache size is a budget, and on Windows an over-sized one pages instead of failing.** `--expert-cache N`
(`"expert_cache": N`) is `N` times the largest expert blob, and the engine compares it only with the free VRAM it
reads **before** the slots are written. `--expert-cache auto` (the default) sizes from that same figure but **checks
again once the slots are actually written** and shrinks if they do not fit; that second check is auto-only (the loop
breaks on `!auto_cache`). Under WDDM an allocation is not resident until it is touched, and the free figure read
before it can be about a gigabyte too high, and the driver's default sysmem fallback puts an over-committed
allocation in system memory instead of failing - so the start looks normal, the banner still reports the slot count,
and only the speed collapses. The startup line is the tell: `... MiB of VRAM free with everything loaded`, and below
about 256 MiB it adds `LOW: requests may stall; add --vram-reserve-mib N` with the value that would have left room.

Measured on an RTX 4080 SUPER 32 GB, IQ3_S, engine 0.1.39, `--max-context 1048576` (yarn factor 4), fresh
43,969 / 45,670 / 47,956-token prompts with 200 generated each, one engine start per row (the rest of the
configuration is the one in `bench/results/2026-10-04-community-rtx-4080s-iq3s`):

| expert cache | slots | VRAM free at start | decode tok/s |
| --- | ---: | --- | --- |
| `--expert-cache 8900` (11,631 slots, 22.07 GiB) | 11,631 | 0 MiB (`LOW`, suggests `--vram-reserve-mib 1212`) | 13.7 / 14.7 |
| `auto` | 11,178 | 217 MiB (`LOW`) | 102.1 / 122.3 |
| `auto` with `--vram-reserve-mib 1500` | 10,766 | 1,075 MiB | 88.9 / 96.5 / 101.4 |
| `--expert-cache 7000` | 9,148 | 4,130 MiB | 94.1 / 97.3 |
| the same 11,631-slot cache at `--max-context 524288` | 11,631 | 299 MiB | 113.2 / 107.9 |

453 slots (0.86 GiB) separate 13.7 from 102 tok/s, and the slower run had the **higher** cache hit rate (91.4 / 94.7%
against 90.4 / 93.9%), so this is not misses: being at 0 MiB free is what costs 7x. Two ways to stay out of it: leave
the cache on `auto`, or keep an explicit size and raise `--vram-reserve-mib`, which the engine deducts **before** it
sizes the cache (700 -> 1000 -> 1500 took the free figure from 217 to 575 to 1,075 MiB and cost about 400 slots).

**Checking it on Windows.** `\GPU Process Memory(<pid>)\Shared Usage` and `Dedicated Usage` (Performance Monitor, or
`Get-Counter`) look like the right instrument, but on this machine they did **not** separate a 13.7 tok/s run from a
102 tok/s one: `Shared` read about 55.5 GB at 262/524K and 62.4 GB at 1M in every run, because it counts the
deliberately pinned memory (the expert arena, 50.3 GB here, plus the pinned K/V, 6.19 -> 12.38 GiB at 1M). The free
line in the log is the reliable tell, and on NVIDIA hardware `CUDA - Sysmem Fallback Policy` set to
`Prefer No Sysmem Fallback` for the engine's executable is the other way to make an over-sized cache fail instead of
paging (not measured here). Reported with the measurements in
[issue #781](https://github.com/Niko1221/Strata/issues/781).

The same shape appears on a second, smaller card. On a 16 GB RTX 5080 at 786,432, UD-IQ4_XS, INT8 KV, with
`Prefer No Sysmem Fallback` set and an explicit `--expert-cache 4000`, taking `--vram-reserve-mib` down from 770 to 0
grew the cache from 2,782 to 3,115 slots (768 MiB) while the card's used memory moved 129 MiB, the hit rate stayed
flat (59.9% to 63.0%) and decode fell from 40.9 to 16.6 tok/s. Every step there reached READY with 0 MiB free, so
with the fallback off it still starts and degrades instead of failing - the slack left for the rest of the card is
worth more than the extra slots, which is the same ordering measured above (enkynakamura, #781; one 256-token sample
per step, so the upper rows are noisy).

**The free figure moves after startup.** The `... MiB of VRAM free with everything loaded` line is read before the
verify windows are captured (each captured window, and each `--batch` slot layout, takes some tens of MiB: 538 MiB at
that line was 395 MiB after the last capture in #2065, an RTX 4090 D with `--batch 2` at 262,144 context). Below about
a gigabyte on Windows the driver can page GPU memory to system RAM instead of failing: decode fell from 91 to 3 tok/s
at 70K tokens and then a window never finished, the watchdog stopping the engine after 60 s. From 0.1.43.1 each request
re-reads the free VRAM, says `WARNING: only N MiB of VRAM free at the start of this request` once it is under 1 GiB
(512 MiB elsewhere) with the `--vram-reserve-mib` that would leave room, and the stall report prints the figure. The
engine only says it; the fix is a larger `--vram-reserve-mib`, a smaller `--max-context` or `--batch`, or
`Prefer No Sysmem Fallback` for the engine.

**A stall named "request" can be the OS paging the engine's own memory.** Before a request that switches
conversation, the engine parks the outgoing one in RAM: copies of its running checkpoints and K/V, which then sit in
ordinary memory until the conversation comes back. On a 64 GB PC with `--resident-budget-gib 40`, an 8 GiB conversation
cache and a 256K context, the working set was 8-30 GB of 56 GB committed and the engine took about 190 page faults
a second (under 1 MB/s if each is a pagefile read) in #2049: one silent step of more than 60 s, five times, on 0.1.41
and 0.1.43 alike (thread dumps: the main thread inside the copy of the checkpoints, at the same instruction each time). The log's `verify window (last window, not the
current stage) ... host at layer step 48; the GPU rang 48` line in that report is the previous request's last window,
not the stalled one. From 0.1.43.1 the park is named in the stage (`request: parking the previous conversation`), its
copies report progress to the watchdog, and Windows counts page faults at 8 a unit instead of 64 when it decides
whether the file tier is busy. The cure for the slowness itself is memory: a smaller `--resident-budget-gib`, a smaller
`--conversation-cache-mib`, or the pagefile on an SSD.

---

## Using it

The server listens on `http://127.0.0.1:8080` (change with `--port` in setup, or edit the run script).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#the-responses-api-and-codex-cli)) | `POST /v1/responses` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /health` |
| Model properties | `GET /props` (also accepts `?model=<loaded-model-id>`) |
| What the model is doing right now | `GET /status`, `GET /slots` (busy or idle, with `n_prompt_tokens`; one entry per slot with `--batch`) |
| Save / restore the conversation to a file (session files, below) | `POST /slots/0?action=save\|restore` |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The same for Prometheus, with vLLM's metric names (asked with `Accept: text/plain` or `?format=prometheus`) | `GET /metrics` |
| Request totals and what is running in Prometheus' text format, under llama-server's metric names | `GET /metrics/prometheus` |
| The MCP servers, their state and tools ([below](#tools-from-mcp-servers)) | `GET /mcp` |

`GET /metrics` answers a Prometheus scrape (`Accept: text/plain` or `application/openmetrics-text`) in the text
format with vLLM's names - `vllm:num_requests_running` / `_waiting`, `vllm:kv_cache_usage_perc`, the token and
request counters, `vllm:prefix_cache_queries_total` / `_hits_total` (prompt tokens read / reused),
`vllm:spec_decode_num_draft_tokens_total` / `_accepted_tokens_total` (the MTP drafts), and the histograms
`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds` and `vllm:e2e_request_latency_seconds` - so the
dashboards and alerts written for a vLLM server read this one. With an API key, the scraper sends it as a bearer
token. Any other request keeps the JSON.

The JSON's own facts that vLLM has no name for come in the same scrape under `strata:`, named after their JSON key,
so a Monitor-tab panel and a Grafana panel read the same value: `strata:live_state{state="..."}`, `strata:live_tok_s`,
`strata:live_prefill_tok_s_mean`, `strata:live_prompt_read`, `strata:engine_max_context`,
`strata:totals_prompt_seconds_total` / `strata:totals_decode_seconds_total`, `strata:last_hit_rate` and
`strata:last_decode_tok_s` (the last request's), and the hardware - `strata:gpu_util`, `strata:gpu_mem_used_bytes`,
`strata:gpu_mem_total_bytes`, `strata:gpu_temp_celsius`, `strata:gpu_power_watts` and `strata:gpu_energy_wh` (the
card's energy used since its driver loaded; one sample per card, label `gpu`), `strata:cpu`, `strata:ram_used_bytes`, `strata:ram_total_bytes`. A value the server does not have (no GPU
telemetry, an older engine) has no sample rather than a zero.

Ready to use, in `docs/monitoring/`: `prometheus.yml` (a scrape config, the API key as a bearer token),
`servicemonitor.yaml` (the same for the Prometheus Operator) and `grafana-strata.json`, a Grafana dashboard to
import (it asks for the Prometheus data source): requests running and waiting, tokens per second, time to first
token, time between tokens and request duration (p50 / p95), the prompt cache and MTP draft rates, and the engine's
own panels (state, batch slots, rates, expert cache hit rate, GPU and RAM). Its first row uses only vLLM's names, so
it also reads a vLLM server.

`/models` and `/v1/models` list only the loaded model, with its context limit and input modalities. `/props` exposes the original chat template, context limit, configured generation defaults (shared settings take precedence), model path and engine version when available. Context means the full engine context, not the resident KV window. `n_predict: -1` means no fixed output cap. Unconfigured sampling fields are omitted. `/slots` uses llama.cpp's names: `n_ctx` and `n_prompt_tokens` (the running request's prompt size, kept after it ends: what a front-end's context meter divides by `n_ctx`). `total_slots` in `/props` is the number of `--batch` slots (1 without). `autoload` has no effect; an unknown `model` returns 404. These metadata endpoints and `/slots` require the API key when one is configured. They do not load, unload or restart models.

`chat_template_caps` uses llama.cpp's field names. Strata checks them when loading the active template by rendering
small requests; a rejected or omitted feature is `false`, and a template that fails a probe in any way only turns that
feature off (the server still starts). `supports_tools` checks tool definitions and XML call
instructions; `supports_tool_calls` checks that calls in the history use Strata's XML format and keep tool results.
`supports_parallel_tool_calls` checks multiple calls in one assistant turn, and `supports_system_role` checks a
leading system message. `supports_preserve_reasoning` means older assistant turns keep their `reasoning_content`
in the rendered prompt. It does not describe whether a new reply returns reasoning. These are compatibility hints
for clients, not a guarantee that the model will follow a request. Zed's llama.cpp provider reads this object to
offer tools.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

- **Thinking levels: none, low, medium, high.** The model thinks before it answers (streamed as
  `reasoning_content`, Anthropic: `thinking` blocks). Choose how much per request - in the chat page (the "Thinking"
  menu), in `chat.py` (`/think low`), or over the API:

  | API | how |
  | --- | --- |
  | OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
  | Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

  Without a setting the model uses its own default, **high**. `none` answers at once (fastest); `low` keeps the thinking
  short. The levels are instructions the model was trained with, not a hard token limit: on easy questions all three
  think briefly, on hard ones `high` thinks longest and is most accurate.
  To start with another level for every request that names none, put `"reasoning_effort": "low"` (or `none`, `medium`,
  `high`) in `strata-<model>.json`, outside `"args"` (#1641): the engine has no such option, and `--reasoning-effort` in
  `"args"` is moved to this key with a note at startup. The web app's shared Chat settings ("Use for other apps too") and a
  request's own level win over it.
- **Keep Windows awake while it works (opt-in, #1727).** `"prevent_sleep": true` in `strata-<model>.json` keeps the PC
  from going to sleep while at least one request is running or waiting (Windows' `SetThreadExecutionState` with
  `ES_SYSTEM_REQUIRED`; the screen may still turn off), and gives the normal sleep rules back when the server is idle
  and when it exits. Default off. On Linux and macOS it does nothing (one line at startup says so).
- **A hard thinking budget (opt-in).** `"reasoning_budget_tokens": N` in a request (OpenAI or Anthropic) caps the
  thinking at N tokens: when it gets there the server ends it with a short wrap-up line and `</think>`, and the model
  answers from there (the engine continues from what it already holds, so nothing is read again). The wrap-up is
  part of the thinking the client sees and counts as output tokens. `"reasoning_budget_tokens": N` in
  `strata-<model>.json` sets it for every request; a request's own value wins, and `0` means no budget. Off by default;
  Anthropic's `"thinking": {"budget_tokens": N}` chooses the level, as above, and since 0.1.45 is also the budget (a
  `reasoning_budget_tokens` in the same request wins).
- **max_tokens counts the thinking, so the answer keeps its room (0.1.45).** A client that caps the whole reply (the
  desktop agent's harness sends `max_completion_tokens: 8192` on every request) and a task the model thinks long about
  used to be cut off in the middle of the thinking, with no answer. Now a request with thinking on, no budget of its
  own and `max_tokens` of 1,024 or more gets one: the thinking is wrapped up when only a quarter of the cap (at least
  512 tokens) is left, and the model answers in that room (8,192: thinking up to 6,144, answer 2,048). Requests without
  `max_tokens` (the rest of the context) get a budget of three quarters of it. `"answer_room": false` in
  `strata-<model>.json` turns it off. Qwen recommends up to 262,144 tokens of reasoning and 131,072 of answer: a client
  should not cap the reply far below what the context allows.
- **Past thinking is not part of the next prompt by default.** The server removes
  historical `reasoning_content`, including in tool-call history. It still generates
  and streams thinking for the current reply. To include supplied historical
  reasoning, set `"chat_template_kwargs": {"preserve_thinking": true}`. This is an
  explicit request option, not the default.
- **A reply that ends inside its thinking (#1053, opt-in).** Some turns write a sentence of reasoning and then the
  end-of-turn token with no `</think>`: the content is empty and an agent stops. `"reasoning_close_retry": true` in
  `strata-<model>.json` closes the thinking once (as the thinking budget does) and continues, once per request, only for
  a reply that ended that way with no answer and no tool call. Off by default.
- **A reply stuck on one token is ended (0.1.39, #606).** When a reply repeats the same token 256 times in a row, the
  server ends it there with `finish_reason` `"length"` and says so in its window: a model in a loop, or a broken
  state that answers one token forever (#606 saw 36,689 tokens of `!`). `"repeat_stop_tokens": N` in
  `strata-<model>.json` sets the run length; `0` turns it off (for a request that really wants one token many times).
- **Non-finite logits end the request and drop the cached state (#879, on by default).** Rarely, a window's logits
  become NaN and the reply degenerates into token 0 (`!`) repeated (#606); the poison sits in the conversation's K/V
  and recurrent state, so with the prompt cache every later turn of that conversation stays poisoned. The q8_1 clamps
  of 0.1.39 and later remove the overflows found so far; this guard catches whatever cause is left. Every verify window
  of `--serve` (the decode windows, the `--pipeline-windows` decode windows of a layer split, the prompt's short
  tail, batch slots) checks its head logits on the GPU beside the greedy pick (one small kernel into mapped flags, read
  after the sync the window already does). A flagged window emits none of its tokens, the engine prints one
  `non-finite logits (#879)` line with the position, ends the request with `DONE ... nonfinite` (a slot:
  `BDONE <slot> <n> nonfinite`), and the next request first drops the live session, every checkpoint, every parked
  conversation and every idle slot (a slot still running finishes its reply and is not kept after it) and reads its
  prompt from token 0. The server retries a request that had sent nothing yet once, transparently; a reply that had
  started ends as `"length"` with a line in the server window; a second fire on the retry ends the request with an
  error (503 before an answer, an error event in a stream). With replicas, the server forgets what that replica held.
  `repeat_stop_tokens` stays as the backstop. The kernel writes only its flags and the host only reads them, so an
  answer without a NaN is the one the engine gives with `STRATA_NAN_GUARD=0`, which records no kernel: on an RTX
  3080 with the RVN IQ3_S config at fixed expert sets (a chat replay of 8 turns on a 16K-token system prompt), the
  guard on and off gave byte-identical answers in 3 pairs (one of them with `STRATA_ROUTE_TAIL_SKIP=0`, one with
  `STRATA_SH_STREAM=0`); one more guard-off run differed from all the others from its first answer on, while a second
  guard-off run matched the guard-on runs, so that was a difference between runs, not the guard. Its cost there: 4.8 us of GPU time a window on average (7.8 us at most), 0.009% of the decode
  time (Nsight Systems, 424 windows). The one-shot command line (`strata` without `--serve`) and the Intel (SYCL) port
  in `sycl/` do not check.
  Debug switch: `STRATA_NAN_INJECT=<n>[,<n>...]` reports the n-th window the guard checks in the engine process
  (counted from 1 over prompt-tail, decode and batch windows) as non-finite, to exercise the whole path without a real
  fire. With `=1,60` on that 3080: a new chat's first window was flagged and the server sent the request again from
  token 0 (the client saw one normal answer); the 60th, mid-reply in the next turn, ended that reply as `"length"`
  after 58 tokens; the turn after it reused 0 cached tokens and answered normally.
- **Repeated reasoning (opt-in, #728).** The single-token guard above does not see a model that repeats whole
  passages. `"reasoning_loop_recovery"` in `strata-<model>.json` is `false` (the default), `"stop"` or `"recover"`
  (`true` means `"recover"`). Every 512 output tokens, at a complete character and parser boundary, the reasoning is
  measured over its last 2,000 words and punctuation marks (counting passages over the last 30,000 words), and its
  last 2,048 characters are checked for a period of at most 64 characters (#1753: a loop written as one long word,
  such as a 24k-digit string cycling an 11-digit pattern, is a single word to the count and never repeats one
  token); a periodic tail counts as fully repeated. If at
  least 25% belong to 12-word passages seen three times, `"stop"` ends the reply there as `"length"` and says so in
  the server window. `"recover"` stops and drains that generation, then goes on once from all its generated token
  ids with the template's low-effort sentence in place of the xhigh one in the first system message (a splice of
  token ids: the rest of the prompt is not decoded or re-encoded). The task stays the same; no answer or `</think>`
  is inserted, and both passes share the original output limit. On recovery only, temperature is raised to at least
  1.0 and presence penalty to at least 1.5; a client's top-p, top-k and seed are never changed. `/metrics` records
  `reasoning_recoveries` and the coverage. This is a policy, not a numerical engine fix or a guarantee of an answer.
  `"recover"` needs the exact xhigh sentence in the first system message and skips requests with images; re-reading
  the changed prefix costs prompt time. Either mode can mistake repeated useful code or checks for a loop, so keep
  the recorded answer quality alongside the completion rate when testing it.
- **Changing the effort without re-reading the prompt (opt-in, 0.1.39, #458).** The effort's instruction is the
  first thing in the prompt, so a request that only changes the effort (an agent's "think harder" switch, `none` for
  a quick tool step) reads the whole conversation again. `"effort_position": "end"` in `strata-<model>.json` renders
  every request's prompt start as the default effort's and puts a `low` / `medium` effort in a short system turn
  right before the answer (no thinking: the empty thinking block, as always); the engine (0.1.39+, which the server
  checks) keeps its checkpoint in front of that turn, so the next request reuses the conversation whatever its
  effort. The default (`"start"`) prompt is unchanged. The model sees a level that is not the default in another
  place than it was trained with; how well it follows it there is not measured yet.
- **Anthropic requests that don't ask for thinking (opt-in, 0.1.32, #278).** By default a `/v1/messages` request
  with no `"thinking"`, effort or budget thinks as the model's template does. `"anthropic_thinking": "on_request"` in
  `strata-<model>.json` renders such a request without thinking - Anthropic's own rule, and what Claude Code's short
  helper calls (a session title in a few dozen tokens) need; its real turns ask for thinking when it is on there.
- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece, like OpenAI and Anthropic do). While the model reads a
  long prompt the stream sends keep-alives, so agents do not time out; the server window prints progress every
  15 s, and `GET /status` says what it is doing (`reading the prompt`, `answering`, tokens so far). Closing the
  connection or pressing stop in your app really stops the model, so the next request starts at once.
- **Tool calls in other forms (opt-in).** `"tool_call_recovery": true` in `strata-<model>.json` reads a call of a
  tool the request declared also when the model writes it as `<parameter=NAME>` instead of `<function=NAME>`, as JSON
  (`{"name": ..., "arguments": ...}`) inside `<tool_call>`, as a `<function=NAME>` block at the start of a line
  without `<tool_call>` (outside code), or as a second call inside the same `<tool_call>` (without the switch the
  second call's parameters merge into the first). Anything else in those forms stays the text it is. Measured on
  1,462 agent turns (Qwen3.8 under Claude Code, from signalnine/q27's drift corpus): 98.3% read as intended with it,
  92.9% without. Off by default, so what a client gets back is unchanged unless you turn it on.
  This also repairs consecutive `<tool_call>` openers before a complete function call inside the thinking.
  The call must name a declared tool, start on its own line outside code, and be followed only by whitespace
  or more calls up to `</think>` or a natural stop. A quoted call or one cut by a token limit, cancellation,
  or error stays text. Parameter values are not rewritten.
- **Generation safeguard counters.** With `"api_monitor": true`, open `/api-monitor` to see calls recovered from
  thinking, duplicate openers repaired, candidate calls kept as thinking, passage-loop stops, repeated-token stops,
  and reasoning retries. The same counters are in `/metrics` under `totals`: `tool_calls_from_reasoning`,
  `tool_call_openers_repaired`, `tool_calls_kept_as_reasoning`, `reasoning_loop_stops`, `repeated_token_stops`,
  and `reasoning_recoveries`. Counts reset when the server starts. Categories overlap: a repaired opener also
  counts as a recovered call. Calls kept as thinking counts recognized candidates rejected at settlement,
  not every tag in quoted code. These are intervention counts, not a measured model error rate.
  For request-level passage-loop stopping, set `"reasoning_loop_recovery": "stop"`; it ends that reply as
  `"length"` and leaves the server available for the next request. This switch remains off by default.
- **opencode's environment block in the user turn (opt-in).** opencode writes an `<env>` block (working directory,
  today's date) near the start of its system message, ahead of ~40K characters of instructions and tool definitions that
  are the same in every session. The prefix cache only reuses a prompt's start, so every new folder and every new day
  read ~14K tokens again. `"env_to_user": true` moves that block to the front of the first user message: the system
  turn is then the same for every session and the disk prefix cache restores it. Measured (RTX 5070 Ti Laptop, Swift 1.5
  IQ3_XXS, a new opencode session in a second folder): first request 14,073 tokens with 13,693 from the cache, 1.1 s,
  against 14,753 read, 11.9 s, without it; the model still answered the folder and date correctly (3/3). Off by default.
- **Prefill progress in the stream (opt-in).** `"return_progress": true` puts that progress on the stream instead of
  sending only the keep-alive, as one extra field on a chunk with an empty delta: `prompt_progress` with `total`,
  `cache`, `processed` and `time_ms`. Those are llama.cpp's four fields and mean the same there (`time_ms` is the time
  since the prompt started reading, and the work still to do is `(total-cache) - (processed-cache)`), so a client that
  draws a prefill bar for llama.cpp draws one here too. It is off unless the request asks, as it is in llama.cpp.
  The engine says one line per `--prefill` chunk, so that chunk is the step: measured on two RTX 3090s with
  `--prefill auto` (8192 tokens), a 42,131 token prompt read in 15 s sent six of them. A prompt shorter than one chunk
  sends nothing, and that is on purpose: its only line arrives once the prompt is read, because the last tokens go
  through the verify windows rather than the batched path, so it stops up to `--short-read` tokens short of the end.
  The server drops a PP that close to the end, and reads `--short-read` the way the engine does (the last one in the
  args wins, a negative is 0), so the dropped line follows the configured threshold.
- **Chat apps.** Any app with an "OpenAI-compatible" provider works: base URL `http://127.0.0.1:8080/v1`, any API key.
  Zed's llama.cpp provider reads `/props` and offers tools from `chat_template_caps`. Its generic OpenAI-compatible
  provider does not: add `"capabilities": {"tools": true, "chat_completions": true}` on that model in Zed's settings.
- **OpenCode** (#543). A starting point for `opencode.jsonc` (in your project, or `~/.config/opencode/`); the field
  names are OpenCode's, so check its config docs if your version differs:

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "strata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Strata (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "strata": {
            "name": "Qwen3.8-Flash-Next (Strata)",
            // context: what you chose in setup; output: what one reply may use (prompt + output must fit)
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },                  // sent as reasoning_effort
            "variants": {                                              // switch between them in OpenCode
              "low": { "reasoningEffort": "low" },
              "medium": { "reasoningEffort": "medium" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "strata/strata"
  }
  ```

  Set `limit.context` to the context you chose in setup: OpenCode compacts the conversation before it gets there.
  Keep `limit.output` well under it: a request whose prompt plus `max_tokens` runs past the context has its `max_tokens`
  shortened to the room left (see **Context** below). For a hard cap on the thinking, add
  `"reasoning_budget_tokens": N` to `strata-<model>.json` (see above).
- **Claude Code** (Strata 0.1.17 or newer): set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` and
  `ANTHROPIC_MODEL` to a Claude model name it knows (it refuses names it doesn't; Strata ignores the name), plus any
  `ANTHROPIC_AUTH_TOKEN` (or your `api_key`, if you set one).
- **Codex CLI** (0.1.39): see [the Responses API](#the-responses-api-and-codex-cli) below.
- **Context.** Chosen in setup (8K-262K). Requests longer than that are refused, never silently cut. A request whose
  `max_tokens` would run past the context is answered with `max_tokens` shortened to the room left (agents such as
  Claude Code ask for 32,000 on every request); the thinking budget follows the shortened cap. For the strict llama.cpp
  behaviour (400) set `"fit_max_tokens": false` in `strata-<model>.json` or pass `--no-fit-max-tokens` to
  `serve/server.py`. A prompt that leaves no room at all is always refused.
- **Model aliases** (0.1.32). `"aliases": ["qwen", "local-model"]` in `strata-<model>.json` lists the model under
  those names too in `/v1/models` (each with its own `id`, and in the model's `aliases`), like llama-server's
  `--alias`; a request naming one is answered under that name. Any other name is still served, as before.
- **Model settings in the web page (0.1.39, #564).** The About tab's Model settings card shows and changes a few of
  the keys above in `strata-<model>.json`: the `sampling` defaults (temperature, top_p, top_k, min_p),
  `reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`,
  `lazy_load`, `engine_silence_s`, `api_monitor`, `open_browser` and `--vram-reserve-mib`. An empty field removes the
  key (its default). Every other key of the file stays as it is, the earlier file is kept as
  `strata-<model>.json.bak`, and the model uses the change from its next start. Only Strata's own page can save
  (JSON, the API key when one is set, as for the Chat settings); the network, key, MCP and program keys are not
  editable there.
- **From other devices on your network.** The server listens on your PC only (`127.0.0.1`) unless you say otherwise:
  run setup with `START-HERE.bat --setup --host 0.0.0.0 --api-key some-long-secret` (or add `"host": "0.0.0.0"` and
  `"api_key": "..."` to `strata-<model>.json`). The server window then prints this PC's addresses
  (`from other devices: http://192.168.x.x:8080/`); open that on the other device, or use `.../v1` as an API base URL.
  On Windows the firewall blocks it until you allow it: accept its prompt for Python (private networks), or run
  `New-NetFirewallRule -DisplayName "Strata 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private`
  in an admin PowerShell, and make sure the network is set to Private.
- **From the internet.** Put a tunnel in front of it, for example [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):
  `cloudflared tunnel --url http://127.0.0.1:8080`. **Set a key first**, or anyone with the link can use your PC:
  add `"api_key": "some-long-secret"` to `strata-<model>.json` (or set the `STRATA_API_KEY` environment variable);
  clients then send it as their API key. Several keys (one per client, so one can be withdrawn alone): separate them
  with commas, `--api-key key1,key2` or `"api_key": "key1,key2"` as llama.cpp does, or give a list,
  `"api_key": ["key1", "key2"]` (the way to use a key that contains a comma); a request passes with any one of them.
  Streamed answers carry `X-Accel-Buffering: no`, so nginx-style proxies pass
  each token on at once. The web app's settings and MCP tools only answer Strata's own page: when you open it through
  a proxy or tunnel whose address differs, add that address, e.g. `"trusted_origins": ["https://strata.example.com"]`.
  With the key set, any `Host` name reaches the server (see Host names below).
- **From web apps in a browser (CORS).** Off by default. `"cors_origins": ["https://chat.example.com"]` lets pages of
  those origins call `/v1/*` from the browser (Open WebUI's direct connections, browser extensions); `["*"]` lets any
  page do it - only sensible with an API key. It never opens `/settings`, `/unload` or the MCP tools.
- **Host names (DNS rebinding).** A web page of another site can point its own name at `127.0.0.1` and then reach
  this server as if it were its own, so without an API key the server answers only requests whose `Host` is a name
  it knows (with a key the check is off: such a page cannot send the key, and tunnels and proxies that pass their
  own name on keep working):
  `localhost` (and `*.localhost`), any IP address (`127.0.0.1`, `[::1]`, `192.168.x.x`, ...), the address it
  listens on and, when it listens beyond this PC (`0.0.0.0` or a LAN address), this PC's name (`mypc`, `mypc.local`)
  and `host.docker.internal`; any port. Others get **403** naming the setting, and the server window prints one line
  for each. Reach it under another name (a reverse proxy that keeps the name, a tunnel, a DNS name on your network,
  another container's name for it)? Add the name: `"allowed_hosts": ["strata.example.com"]` in
  `strata-<model>.json` or `STRATA_ALLOWED_HOSTS=strata.example.com` (comma-separated); `".example.com"` allows that
  name and every name below it, and `["*"]` turns the check off (so does setting `api_key`). The hosts of
  `trusted_origins` count as allowed. Requests without a `Host` header (HTTP/1.0 clients) pass.
- **Web pages without an API key.** Without `api_key`, a `POST` to `/v1/*` that carries an `Origin` header (a
  browser page sent it) is answered only for Strata's own page, pages on `localhost` or an allowed host name (any
  port), the origins in `trusted_origins` or `cors_origins`, and browser extensions and desktop apps
  (`chrome-extension://`, `moz-extension://`, `app://`: no web site can send those), and only with a JSON body; any
  other page, and `Origin: null`, gets **403**. Clients that send no `Origin` (curl, the OpenAI and Anthropic SDKs,
  other servers) are not affected. With
  an API key, the key decides. `POST /unload` and `POST /load` take `Content-Type: application/json` from Strata's
  own page (or no `Origin`), like `/settings`. `POST /slots/0?action=save|restore` keeps the Host and API-key checks
  and also takes only JSON from no `Origin`, Strata's own page or a trusted origin - also when an API key is set.

**Conversation cache.** A continuation uses the live session or a compatible saved checkpoint.
The active checkpoint limit is 6 by default. Checkpoints contain recurrent state, PLE history,
and indexer state. Their size depends on the model, not on the token count.
The engine bounds ordinary archive history as well as the active checkpoint list. It keeps
required prefix anchors and explicit pins. A checkpoint must match the prompt's tokens and pictures.
A later recurrent state cannot substitute for an earlier checkpoint. The engine restores a retained
anchor before the branch point and processes the remaining tokens.
The active list also admits an anchor before the final input token. The first decode window
processes that token. A reply rewrite can use this resident input anchor without loading a newer
state from the obsolete generated tail.
A bounded checkpoint archive does not remove a full parked source. Later archive compaction
can thus discard temporary states without removing the original continuation's resume state.
A branch inside the saved input also freezes its source archive. New archives can borrow its
K/V, but they do not rotate the source's original resume state.

L2 branches share their unchanged K/V allocations. An append or rewrite allocates only its new
or changed ranges. The cache charges each shared payload allocation once. A short request that
shares a long conversation's prefix does not remove the long continuation. A system-prompt
checkpoint can also supply a common root for separate chats. Its threshold is set by
`--prompt-cache-root`. Claude Code stamps its system prompt with a billing header that changes on every
request (`cch=...`) and every session (the 4th part of `cc_version=`); the server pins both stamps (to `f`s, as
llama.cpp does), so the system prompt and the tool list in front of it are the same prompt on every turn. Engine options: `--prompt-cache N` (0 = off), `--prompt-cache-every N`,
`--prompt-cache-root N` (0 = no system-prompt checkpoint), `--turn-token ID`.
A one-shot request that no later request continues (a classification call, a probe) can send
`"strata_checkpoint": false` in its body: it saves no checkpoint at its last turn nor every 16K tokens, so what
follows the reused prefix (or the root) is read in one run, and its session is not kept or parked for a next request.
It still starts from a checkpoint it matches, and still saves the system-prompt root when that reaches
`--prompt-cache-root`. Without the field (or with `true`) nothing changes.

**Answer-only history in GPU memory.** A client can omit the previous reply's
thinking. The generated GPU state then differs from the next prompt. That state
cannot be reused as if the thinking were absent.

After a complete reply, a serial server prepares the exact answer-only history
with a decode-free idle prefill. It includes completed tool calls and the original
image embeddings. It does not store hidden thinking in the next prompt or generate
an extra token. A following request reuses the prepared prefix only if its tokens,
images, and control-vector mode match. The server encodes the prefix with the same
rules as the request, so a control token inside a message stays text.

A request cancels idle prefill before it waits for the engine. The engine stops at
a safe prompt-read boundary and drains the command before it accepts the request.
There is one exception. When the prefill reads the start of that request's own
prompt, the request waits for the read instead. The request then reuses the read
and processes fewer tokens. A prefill for a different conversation always gives
way.

A text request does not cancel idle prefill when it loads an engine that is already
loaded or prepares its prompt. These steps do not need the engine queue. Image
encoding, engine loading, and a prefill that does not match the request still cancel
idle prefill before they use the engine queue.

The rebuild still uses GPU time. If the next request arrives before it completes,
or changes the history or prompt settings, some prompt processing remains on the
request path. The last session stays in GPU memory; L2 and L3 parking is for a
conflicting session, not for every completed turn.

The engine reports `INFO prefill_only=1` when it supports this path.
`GEN 0 <ids>` and `GENI 0 <embeddings> <ids>` process all but the final prompt
token, emit no generated tokens, and return `DONE` with a generated count of zero.
The deferred token is processed by the next generation request.
**Checkpoints spread back from the end (opt-in, `--prompt-cache-spread`).** Least recent use keeps the newest points:
every turn adds a checkpoint at its start, so after a few short turns the 6 sit within the last 1-2K tokens. An agent
client that rewrites its history a few thousand tokens back (it trims an old tool output, drops a step) then matches
none of them and resumes from the system prompt - the whole conversation is read again. With the flag the checkpoint
that leaves is the one whose removal leaves the smallest gap for its distance from the newest, so the kept points thin
out geometrically: after 200 turns of 1,000 tokens they sit 0, 1K, 3K, 8K, 43K and 200K tokens before the end, and an
edit d tokens back reads about d again (6K back: 8K instead of 200K; 42K back: 43K). The same number of checkpoints, no
more RAM; the system-prompt root, a pinned prefix and the newest point are never dropped, and the resume is the usual
exact-prefix match. Measured on 2x CMP 50HX (layer split, IQ3_XXS) with the pi coding agent: before, a 210,528-token
prompt that was the same as the session for 204,349 tokens resumed at 9,525 and read 201,003 again (100 s). With the
flag, an 82,593-token conversation grown turn by turn with a turn ~6 steps back rewritten resumed at 73,610 tokens
(10.2 s against 40.0 s read in full), and with the expert placement fixed (`--adapt-swaps 0`, fixed `--pcie-frac`) the
greedy answer was the same text as the full read. Without the flag nothing changes.

**One long document, many questions: a pinned shared prefix (opt-in, 0.1.40.2, [RESEARCH_RUNS.md](RESEARCH_RUNS.md)).**
A request can say where the document ends: `"strata_prefix": {"messages": 1}` (the first message is the document),
`{"message": 0, "chars": 210000}` (the first 210,000 characters of message 0, for a client that sends the document and
the question in one message) or `{"tokens": 12345}`. The engine reads the prompt in two parts there, keeps the checkpoint
at the boundary pinned (retention never evicts it, a parked conversation holding it stays parked, `SAVE` keeps it) and,
when a later question resumes from it, does not park the question it leaves. So the next question starts at the end of
the document, and 20 questions cost one read of it. Without the field the server's checkpoints are where they were: the
last turn's start, the system prompt's end and every `--prompt-cache-every` tokens, so a question that follows a long
document in the same chat re-reads up to 16K tokens of it (55 s at 262K on a Tesla P100). The field is checked against the
prompt's own ids (the prefix is always the longest common start), a prefix that cannot be marked is said in the server's
window and ignored, and a malformed field is a 400. One pinned prefix at a time per engine (a new one replaces it);
it needs `--prompt-cache 3` or more. `tools/research_run.py` runs a document and a question list against a server with and
without it. The engine's own key is `pin=N` on the `GEN` line.

**Multiple conversations (opt-in).** Add `--conversation-cache-mib 8192
--conversation-cache-slots 4` to the engine arguments to park up to four conversations
in a bounded 8 GiB host-RAM cache. This preserves controller/worker histories when
their requests alternate; it does not execute requests concurrently. No client session
ID is required: only exact token/image prefixes with matching steering mode are reused.
The default budget is 0 (disabled); `--prompt-cache 0` also disables parking.
With `--layer-split`, a parked conversation holds one image per stage (each later stage's
session and its share of the conversation checkpoints), every image is validated before
anything is overwritten, and each is restored on its own GPU. Verified on a 4-GPU split:
a 9,276-token conversation restored in 51 ms, and its follow-up decodes exactly the tokens
it decodes when the conversation never left (`tools/parking_test.py`). FP16,
INT8, Q4_0 and identity-layout K8V4 snapshots are supported; the K8V4 draft ring
remains INT8, as in upstream. Windows/HIP and multi-GPU runtime coverage must be
reported separately from Linux/CUDA evidence.

The web page's Monitor tab has a **RAM tier (L2)** card. The `/api-monitor` page also
shows L2 RAM and L3 disk use. The L2 display shows parked conversations, the slot
limit, RAM use, the byte budget, parks, restores, evictions, and prompt-token reuse.
Root parks are included in the counters. `/metrics` supplies the same data in
`"conversation_cache"` and `"l3"`.

Snapshots contain running state, checkpoints, used K/V pages, and draft-layer K/V.
They add host RAM, not another model or VRAM allocation. The byte budget also counts
an incoming snapshot during a switch. After a restore, unchanged K/V pages can be
retained for the next parking operation; growth appends storage without copying
the existing pages, with room for the next turns of at most an eighth of each
buffer (and at most 16 MiB), which the budget counts like the rest of the snapshot.
Rewinds refresh the affected pages, and running state and
checkpoints are captured again. Retained active K/V counts against the same byte
budget and is discarded before evicting parked entries under memory pressure.
Oldest parked entries are evicted first. A new conversation that starts with a
checkpoint inside a parked one (subagents that share a system prompt and tool
list, a compacted history) restores that prefix and leaves the parked
conversation where it is, so its next turn still resumes in full; the engine
log says "borrowed". A match on a parked conversation's live state or its newest
checkpoint is that conversation going on and moves it into the session as
before; so does a match that reaches a pinned shared prefix (`pin=N`), which
keeps one conversation holding the pin. When the outgoing conversation fits only
in the parked one's room, it keeps its place and the parked one is moved in
whole, as before.
Oversized snapshots or host allocation failures fall back to ordinary prompt processing.
`--conversation-cache-min-tokens N` (default 0 = park every conversation) parks only conversations of at least N
tokens. A parked conversation costs one of the slots whatever its length, and parking is a switch-time decision, so a
client that interleaves short side requests with one long conversation spends the cache on the side requests and, once
it is full, evicts the long one oldest-first - exactly the reuse it wanted. Setting the minimum above the side
requests' length keeps the long conversation parked. The check runs before `make_room()`, so a skipped park evicts
nothing and stores nothing; the log line is `conversation cache: skip parking (N tokens, below the M minimum;
parked=P)`. 0 is the pre-flag behaviour and the flag leaves the answers untouched - it only decides which
conversations keep a slot.
`--conversation-cache-park-gain N` (opt-in, default 0 = park every turn) parks a conversation again only after it
advanced N tokens past the copy it would supersede; a turn that grew by a few hundred tokens keeps the older copy
instead of writing a whole snapshot that the next turn drops. The cost is that a resume from the older copy reads up
to N tokens again (#2265).
`--conversation-cache-min-free-mib N` (default 2560) additionally requires that
RAM headroom remain available: the engine checks before allocation (against the
snapshot's estimated size, less the K/V it already retains) and again after capture.
Unknown telemetry or insufficient RAM skips parking, and the same figure and floor
decide whether a session file may be saved or restored (below). An 8 GiB budget
is a cap, not a recommendation for every machine.
The slot limit and the byte budget apply together. For example,
`--conversation-cache-mib 15360 --conversation-cache-slots 32` permits up to
32 parked entries within a 15 GiB RAM budget. The engine does not allocate the
whole budget at startup. Root images also use slots and RAM. Increase the slot
limit if slot eviction occurs before the byte limit is reached. Keep the
physical-RAM headroom check enabled.

**The RAM figure (`available_host_bytes()`).** The headroom is the smallest of three
numbers, sampled each time it is asked and not a reservation (other writers can take the
room afterwards):

1. `MemAvailable` (Windows: `GlobalMemoryStatusEx`'s available physical memory).
2. The room under each memory limit of the engine's own cgroup and of every ancestor
   visible under `/sys/fs/cgroup` (the group named in `/proc/self/cgroup`, or the deepest
   part of that path this mount shows): cgroup v2 `memory.max` and `memory.high`, each
   minus `memory.current`; cgroup v1 `memory.limit_in_bytes` minus
   `memory.usage_in_bytes` (both less the clean inactive file cache, below). `max` and v1's "unlimited" count as no limit; a file that is
   absent (no memory controller at that level) is skipped; a file that is there but
   cannot be read or parsed makes the cgroup terms unusable: without `--memory-limit-mib` the guard then uses
   `MemAvailable` alone, as 0.1.41 did, and prints one warning per process; with the flag the sample is unknown, and unknown skips parking.
3. `--memory-limit-mib N` (or `STRATA_MEMORY_LIMIT_MIB=N`; the flag wins, `0` = none):
   N MiB is the total this engine's container or cgroup may use, minus what that
   container uses now. Use it when the container cannot see its own limit. The usage
   is `memory.current` of the top cgroup this mount shows (the container's own, in a
   container with its own cgroup namespace); if that file cannot be read it is
   `MemTotal - MemAvailable`, which is a weaker figure (it is only as true as the
   container's `/proc/meminfo`), and if `MemTotal` is missing too the sample is
   unknown. The cap is taken in addition to 1 and 2, never instead of them: whichever
   number is smallest decides. It is an engine argument, so the server config carries it in
   `args` (`"--memory-limit-mib", "102400"`); the variable also works from the config's
   `"env"`.

Usage is `memory.current` as the kernel reports it, **less one reclaimable kind: clean
inactive file cache**. That is the group's own `memory.stat` `inactive_file` (cgroup v1:
`total_inactive_file`), less its dirty and writeback pages when `memory.stat` lists them
(`file_dirty`, `file_writeback`; v1 `total_dirty`, `total_writeback`), and never more than
`memory.current`. The kernel gives such pages back at `memory.high` or the limit before
it kills anything, and a loaded model's file reads leave a lot of them behind. Nothing
else is credited: `active_file` (it holds the mlocked pages), `shmem` (pinned expert
complements, the KV pool) and anonymous memory stay charged in full, since a kill at a
hard limit is worse than a skipped park. For each limit the room is
`limit - (memory.current - credit)`, so it never exceeds the limit. A `memory.stat` that is missing or cannot be
parsed gives no credit; it does not make the sample unknown. The operator's cap uses the
`memory.stat` of the same top group whose `memory.current` it reads. A parked
conversation is part of the engine's own usage once captured; the check before
capture is what stops it from growing past the cap.

The startup log has one line for it, `strata generate: memory guard: limit X GiB
(source: flag|cgroup|meminfo), current Y GiB, reclaimable cache credited Z GiB;
available ...`, where `source` names the
number that was smallest at that moment (`meminfo`: MemTotal and `MemTotal -
MemAvailable`), and a skipped park or a refused save/restore says how much it had and
from which source. The line is printed before the model is loaded, so its credit is the
cache at that moment, not at the time of a park. Without a cgroup limit and without `--memory-limit-mib` the figure is
plain `MemAvailable`, as before.

What it does not touch: the startup sizing of the expert arena, the file tier's resident budget, the Windows commit
check and `STRATA_PIN_GUARD` (#1250: whether to page-lock the expert copy in steps follows `MemFree`, and each step
is checked against the cgroup-aware `host_available_memory`) keep their own probes (`--resident-budget-gib`, #633,
#1250), and `--memory-limit-mib` does not reach them. This figure is for what runs after the start: parking a
conversation and saving or restoring a session file, which read `MemAvailable` alone. `host_available_memory`
(expert_source.cpp) and this figure both read the cgroup limits; this one also reads `memory.high` and the operator's
cap; an unreadable cgroup file makes it fall back to `MemAvailable` (one warning), or unknown when a cap is set.

Earlier measurement on a Proxmox LXC where the engine's container cannot see its limit (a 100 GiB cap on the parent
cgroup, outside the container's namespace; the container's own `memory.max` and `memory.high` read `max`):
`/proc/meminfo` showed about 25 GiB `MemAvailable` while about 1 GiB was really left, and right after the engine had
loaded the container's `memory.current` was 98.29 GiB with `anon` 1.29 GiB, `shmem` 65.51 GiB, `inactive_file`
3.81 GiB (clean page cache left from reading the model files) and `active_file` 0.01 GiB, of which the 3.8 GiB of
clean cache is what the credit gives back. There `--memory-limit-mib 102400` is the cap, and the floor
(`--conversation-cache-min-free-mib`) is measured against what is left under it: with the engine already near the cap,
a floor of several GiB refuses every park; a few hundred MiB keeps a margin for the allocator and the kernel while
letting a snapshot that really fits go through.

Measured on the test machine (2x RTX 3080 20 GB, UD-Q4_K_XL, a 90 GiB container, resident RAM mode, 2 slots), one restart
per arm, `--conversation-cache-min-free-mib 2048`. With `--memory-limit-mib 101376` (the cap our deployment runs with)
the engine parked on every attempt: 135 parks and 0 skips over one run of 308 requests. With `--memory-limit-mib 78000`
the start line read `limit 76.2 GiB (source: flag) ... available 73.0 GiB (MemAvailable 88.6 GiB)`, and 8 of 8 parking
attempts were skipped (`skip parking (physical RAM admission; need 226 MiB plus 2048 MiB floor, 0 MiB available, source
flag)`), none parked. The container's `memory.current` sat at 89.7-89.9 GiB, above the 76.2 GiB cap, so the headroom
under the cap is 0, while `MemAvailable` read 15.7-15.9 GiB in the same samples; a check on `MemAvailable` alone would
have let those parks through (the old check was not run). The requests were served normally (median of 6 decodes
79.6 tok/s); only parking is refused, and the log says so. The unit test `memory_guard_test` (CPU only, no model) passes
130 checks. Not measured: a cap between 78000 and 101376, a run without the flag, any effect on throughput.

The shared snapshot core validates all layers and checkpoints before applying any
state. Invalid entries are discarded; transfer/synchronization failure is fatal
rather than permission to continue with partial state. Indexer spare keys and the
moving spare row are preserved, including checkpoint rewinds.
The engine log reports parking, restoration, bytes, evictions, individual snapshot
sizes and K/V bytes reused during capture. `STRATA_SNAPSHOT_FULL_CAPTURE=1` disables
retention for diagnostic comparisons. RAM-tier snapshots are not persisted across restarts.

**Conversation cache on disk (opt-in, L3).** The conversation state has three tiers. Tier 1
(L1) is the live session in GPU memory; it always exists. Tier 2 (L2) is the optional
host-RAM cache above (`--conversation-cache-mib`). Tier 3 (L3) is the optional disk store
below. L2 and L3 are optional. L1 always exists.

Add `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` to the
engine arguments. The engine then keeps parked conversations as files in the
`strata-conversations` directory, and a restarted server can resume them. The path and a
positive GiB budget are required together. The store is off by default, it needs `--serve`,
and it works with `--conversation-cache-mib 0`. Both the disk and RAM caches
support `--layer-split`. The engine creates the directory when it is absent. A store that cannot be
created stops startup with an error.

The engine first parks an outgoing conversation in L2. A successful L2 admission
does not write a copy to L3. If the L2 byte budget, slot limit, or physical RAM
floor requires eviction, the engine moves the selected snapshot to L3.
The spill uses the existing host buffers. It does not copy the payload or read
the snapshot from the GPU again. Each spill creates a complete disk record.
If L2 is disabled or cannot admit a conversation, the engine parks it directly
in L3. The same rule applies to shared root prefixes.
L2-only entries do not survive a service restart. Only entries already in L3
can be restored after a restart.

Two optional flags tune the store. `--conversation-cache-disk-slots N` caps the number of
records (default 0 = no cap). `--conversation-cache-disk-min-free-mib N` keeps that many MiB
free on the store's filesystem (default 0 = no floor). Both need the path and the GiB budget.

Two more flags pin the shared system prompt. `--conversation-cache-keep-root` (default on) pins the
system-prompt prefix image in the RAM cache. `--conversation-cache-disk-keep-root` (default on) gives the
prefix its own 'p'-class record on disk. Their `--no-` variants turn each pin off.

A request reuses a record only when its prompt starts with exactly the record's tokens and
images, and its control-vector mode matches. The store keeps the longest matching prefix; an equal-length
tie prefers the smaller record: the system-prompt prefix over the conversation record that contains it.
Across the live session, the RAM cache, the disk tier and the idle batch slots,
the longest valid prefix wins. A disk record is used only when its prefix is
strictly longer than each available resident prefix. An equal-length RAM
match uses L2, not L3. An equal-length live or idle-slot match stays resident.
L3 can still supply a longer prefix, or restore a conversation after L2 eviction.
No client session ID is needed.

When a validated disk root is restored, the engine can retain its existing host
image in L2. The RAM slot limit, byte budget, physical-RAM floor, and root setting
still apply. The engine does not capture the root from the GPU again.
A later RAM restore keeps the pinned root available for another chat.
Only token and image identity metadata is copied into the active session.
The cached K/V buffers remain separate from active reuse buffers.

A direct L3 fallback captures the outgoing state before the GPU session changes.
Its disk write can run in the background. A lookup, a store change, or another
park waits for that writer before it accesses the store. L2 eviction spills
finish before the victim buffers are released. A continuation with the full
useful resident prefix does not wait for a disk lookup that cannot improve reuse.

**The shared system prompt (captured once).** When a fresh chat's prompt is read from token 0, the engine
checkpoints the end of the system prompt (the first turn boundary) if that is `--prompt-cache-root` tokens
or more. The engine captures that prefix once in L2, or in L3 when L2 is unavailable.
It stores the prefix as a system-prompt (root) image. The root outlives the conversations that share it: the RAM cache and the disk
LRU evict it only after every unpinned record, and an equal-length match prefers the root because restoring
it reads the shared prefix alone instead of the conversation record that contains it. The bytes of one
system prompt are written once, not once per chat. Options: `--conversation-cache-keep-root` and
`--conversation-cache-disk-keep-root` (both default on).

The RAM cache removes ordinary conversations before root images. If only roots remain,
it removes the oldest root to make room within the byte budget or slot limit.
Old roots cannot occupy all slots and prevent a new conversation from entering L2.
An explicit shared-prefix checkpoint pin remains protected from this eviction.
The physical RAM admission check and free-memory floor remain active.

One record holds one conversation and one image per `--layer-split` stage. A two-card run
parks and resumes as one unit, and every stage must hold the same token and image chain. A
record carries a compatibility key that covers the engine version, the model and MTP sources,
the geometry, the KV format and rotation, the context, the rope configuration, the per-stage
layer carve and devices, and the control-vector mode. A store is reused only by a run with the
same key.

The files are checksummed and versioned. Each file has a fixed header with a format version, a
metadata checksum, and a payload checksum. The engine writes a record to a temporary file and
renames it over the target file (an atomic replacement). On startup the engine scans the directory, rebuilds the
index, and removes files that fail validation, including files from another version or another
compatibility key. It also removes leftover temporary files. The startup line reports the
recovered record count and the byte total.

The store is bounded. It evicts the least recently used record when the byte budget or the
record cap is exceeded, and when the filesystem free space would fall below the free-space
floor. A record larger than the whole budget is refused.

The engine validates the whole record before a restore. It first checks that all stages agree
on the token and image chain, then validates every stage against the current geometry. It
makes no CUDA write before this validation passes. A corrupt, incompatible, or invalid record
is removed, and the request falls back to normal prompt processing. After validation, a
transfer or synchronization failure is fatal; the engine never decodes from a partially
restored session.

GPU capture and restore are synchronous. The engine captures or restores one stage at a time,
and it synchronizes each stage's device. After capture, the file write runs asynchronously
while the next request uses the GPUs. A disk lookup or a later park waits for that write before
it accesses the store index. During a save or a load the record is staged in host RAM, so the
operation needs temporary host memory about the size of the record. The engine releases that
memory after the file operation. Restore stays on the serving thread.

The disk tier preserves parked conversations across alternation and across restarts.
It does not itself add concurrent execution. Set `"parallel": N` to enable batch slots;
see [BATCHING.md](BATCHING.md) for their memory costs and limits.

**Disk endurance and sizing.** A conversation uses a `.conversation` chain for token metadata
and K/V panels, and a `.states` sidecar for retained running states. A compatible append writes
only the token and K/V delta. It can repeat a boundary page and the moving indexer row.
New panels do not repeat full checkpoint token lists or inline running states.
If the longest token prefix has different K/V bytes, the writer checks earlier input and checkpoint
boundaries. A difference in reply cells must not force a second copy of an unchanged input prefix.
Replay can also change the final input cells. A replacement before the saved input boundary must
prove that the remaining original input tokens and image keys are unchanged. An input edit keeps
the original branch.

The sidecar has fixed state slots for ordinary anchors. New states replace anchors that the
retention policy removes. Explicit pins can require extra slots. Ordinary turns do not add
one permanent recurrent-state payload per turn. If state publication fails, the cache write
fails; it does not use an unbounded inline-state fallback.
When later input consumes a stored generated end, the sidecar can retain that original resume
state in one ordinary slot. Dense input checkpoints do not rotate it away. The marker and state
survive restart. This does not add a state slot on each turn.

A seed uses a temporary chain file and a rename. Appends write a new panel and update the
chain header. The reader checks panel checksums and truncates a torn tail to the last complete
panel. A state-slot update writes its payload before its slot header. An invalid slot cannot
supply a restore state. A torn sidecar header can make all its anchors unavailable.
The engine then needs another valid anchor or a prompt read.

Version-2 files with inline states remain readable. The loader selects retained states before
it allocates their payloads. It verifies discarded payload bytes without retaining them.
Legacy inline states remain on disk until their chain is evicted or replaced.

The engine checks the live GPU prefix before it selects a saved conversation. A disk restore
must offer a longer compatible state anchor than the selected resident state.
A live continuation needs no conversation-cache transfer. Before a rewind overwrites a different
continuation, the engine preserves its eligible checkpoints. This also applies to pinned-prefix
requests. A pin does not authorize the engine to discard the rest of a long conversation.
L2 can remain disabled while L3 is enabled.

During decode, the engine keeps a checkpoint of the state before the latest decode window on each
owning GPU. At response end, it copies this state to one host checkpoint. This checkpoint contains
the recurrent state, PLE history, and indexer state; it does not contain a copy of the K/V cells.
If the next prompt differs near the end of the reply, the engine can resume before that final
window instead of before the entire reply. A mismatch earlier in the history still needs an
earlier compatible checkpoint. The engine limits speculative commits at an end-of-sequence token
or the requested output limit, so the saved live history does not include tokens past that limit.

No conversation-cache transfer is required for a compatible live continuation. This does not mean
that the drive is idle: the model's PLE table has a separate disk-read path. In a prompt log,
"read" counts tokens processed, not bytes read from the conversation cache. Conversation-disk
read and restore messages identify L3 loads. A conversation switch can still take seconds to load
a large record.

New K/V storage grows with the changed suffix, the KV format, and the number of split stages.
Retained running-state storage is bounded separately. The byte budget includes both the chain
and its sidecar. Cache size and write cost depend on the model and workload; a byte budget does
not imply a fixed number of conversations. Flash endurance still depends on the bytes written.
Stop the server before you clear the cache directory. The engine builds an empty index at its
next start.

The store selects shared K/V backing separately from a recurrent-state restore anchor.
A chain can lend a shorter prefix even when it has no running state at that exact position.
The engine verifies retained K/V bytes and completed indexer rows before it borrows them.
Equal token IDs alone are not proof of equal K/V payloads. Replayed boundary cells can require
a shorter shared extent; the changed suffix then starts at that verified boundary.

A branch references the unchanged prefix of an existing chain and stores its own suffix.
It does not truncate the source chain's continuation. A reference can point to a prefix inside
a panel or to another branch. The store checks content, image keys, stage layout, and base
dependencies. It refuses a rewind or replacement that would strand a dependent branch.
The finite cache budget can still evict a chain and its dependents.

The engine records the input-prompt boundary separately from the generated reply.
It can replace an obsolete reply tail after that boundary. This operation does not add a new
conversation or a permanent state file on each turn. A branch before that boundary retains
the source continuation. Pins and dependent prefix references still prevent destructive replacement.
If the stored input boundary is unknown, the engine does not infer it from a nearby checkpoint.

A prefix load skips panels that cannot contribute to the requested extent. It reads the panel
that contains the cut in full to verify its checksum. Thus, a short load from a large legacy
panel can still require a large disk read. A matching token position without a valid running
state is not a restore point. Keep each chain and its sidecar together.

Older duplicate records are not removed by a background deduplication operation. The store
evicts least-recently-used conversation records when a save needs space within the byte budget,
record limit, or free-space limit. System-prompt records have a separate eviction priority.
Thus old unused copies can be removed during later saves, but an idle cache does not shrink.
The fix does not compact obsolete payloads inside an existing chain. No manual cache reset is
required. To reclaim all cache space immediately, stop the service before clearing its cache
directory; this discards saved conversations and makes later requests rebuild their state.

Size the budget for the number of conversations that you want to keep across restarts, not for
one conversation's repeated full snapshots.  A 25 GiB budget is an example: it holds a few large
conversations at a long context, or many smaller ones.  Set `--conversation-cache-disk-min-free-mib`
to protect the free space of the filesystem, and `--conversation-cache-disk-slots` when you want
a hard chain count.
The engine log reports each park and restore with its byte size and its time, and the hit,
miss, eviction, and corruption counts. Read it to size the budget for your workload.
retention for diagnostic comparisons. Parked snapshots are not
persisted across restarts; the session files below are.

**Session files (disk).** The conversation the engine holds can be saved to a file and restored later, also after a
restart of the same engine version, so a long prompt is not read again. The server exposes the save and restore
requests of llama-server's slot API, for its single slot 0, when started with `--slot-save-path DIR` (also
`"slot_save_path"` in the config); NAME must be a plain file name inside DIR. There is no erase action and the file
format is Strata's own, not llama.cpp's:

```bash
curl -X POST "http://127.0.0.1:8080/slots/0?action=save"    -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_saved": 63025, "n_written": 1198691396, "timings": {"save_ms": 709.5}}
curl -X POST "http://127.0.0.1:8080/slots/0?action=restore" -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_restored": 63025, "n_read": 1198691396, "timings": {"restore_ms": 898.2}}
```

DIR becomes one absolute path at start (`--slot-save-path` relative to the server's working directory, the config's
`slot_save_path` relative to the config's `"cwd"`), created with mode 0700 when missing. It should be private to the
user that runs Strata: the files hold the conversation's token IDs and state, are created with mode 0600 on Linux
(on Windows they inherit the folder's permissions), and nothing deletes them - about 1.2 GB per 63K-token
conversation. Before it writes, a save checks that the disk has room for the whole new file plus
`--session-min-free-mib` (an engine argument, in the config's `args`; default 4096, 0 = no check) - also when it
replaces a file, whose space comes back only after the rename. This is a preflight, not a quota or a reservation.
NAME may
not contain a path, a drive, a stream (`:`), a Windows device name (`NUL`, `CON.bin`, `COM1`...), a control character,
a leading dot or a trailing dot or space.

`--session-save-reclaim` (an engine argument in the config's `args`, off by default) lets a SAVE that fails its
RAM preflight first release reconstructible host caches: retained K/V buffers, oldest unpinned parked
conversations, then unpinned checkpoints other than the one selected for the file. It stops as soon as measured
available RAM meets the original allocation estimate plus `--conversation-cache-min-free-mib`; it does not lower
that floor or assume that freed allocations have reached the OS. Pinned prefixes and the selected checkpoint
stay intact, including their order and tie preference. Unknown RAM telemetry or an unknown allocation estimate
does not evict anything. The usual admission check still runs before copying state.

With this option the HTTP server also allows up to 30 seconds for physical RAM accounting to catch up after
cache reclamation. Only an unpublished `memory` refusal from the SAVE RAM preflight is retried, at most three
times, when measured available RAM reaches the reported estimate plus the original floor and 2 MiB for rounding.
The same live engine process and service FIFO are held throughout; a process change, death, unknown telemetry
or missing estimate ends the wait. This does not unload a model or trim another process's working set. Persistent
pressure still returns the last refusal. The wait is outside the engine's reported `save_ms`.

This can trade later prefix-cache hits for room to save. Released caches stay released even if admission or file
I/O later fails; the current tokens, images, steering and live device state are unchanged. It cannot guarantee a
SAVE under arbitrary memory pressure. With the option off, or enough RAM at the first probe, cache retention and
the session file format are unchanged. RESTORE is unchanged.

The request must be `Content-Type: application/json` (else `415`) and come from no browser page, Strata's own or a
trusted origin (another site's `Origin` gets `403`, also with an API key); the Host and API-key checks apply as
everywhere. Errors: `501` without `--slot-save-path` or with parallel requests; `400` for a slot other than 0, an unknown action, a refused
file name, or a file the engine refuses as invalid (not a session file, corrupt, another model or configuration, over
this session's limits; the session is as it was); `404` for a restore of a missing file; `503` while the model is not
loaded, when the RAM to read the file is not there or an allocation failed; `507` when the disk has no room (the
free-space reserve, or the OS reports no space or quota); `500` for any other I/O failure (permissions, read, write,
flush, rename) and when the engine ended (a restore transfer failure, below, an engine that said nothing for
`engine_silence_s`, or one that answered out of protocol) - the next request starts it again. The status follows the
engine's category (`error.kind`: `invalid`, `memory`, `storage`, `io`), never the words of the message. A save that
failed after its new file had replaced the old one says so with `error.published: true` (below). A save or restore waits for the running request
(the same queue), shows in `/status` and counts as activity for the idle unload. A later request whose messages
continue the restored conversation reuses the restored state or its checkpoint; a short tail may be read again (35
tokens in the measurement below). Only the deepest checkpoint is saved, so an edit further back reads more again.
Clients still send their messages (and images): the file holds engine state and token/image identity, not a chat
export. Underneath, `strata --serve` takes `SAVE <path>` and `RESTORE <path>` on stdin between requests and answers
`SAVED <tokens> <bytes> <ms>`, `RESTORED <tokens> <bytes> <ms>`, `SERR <kind> <published 0|1> <reason>` (failed,
the engine and the session as they were) or `FATAL <reason>` (then exits). On the way, `SESSION <done> <total>` follows
every block of the file that moved (at most 16 MiB, the last partial block and a small file included), and
`SWAIT <phase> <seconds>` comes before a step that blocks in one call (`fingerprint`, `capture`, `flush`, `publish`,
`validate`, `transfer`): that step is allowed those seconds - 60 plus one per 4 MiB it concerns, at most 3600 - by the
engine's watchdog and by the server, then it counts as stuck. The next line clears the allowance. The server checks
every line: a malformed or unknown one, counts that are negative or go back, an allowance outside 1..3600 or a time
that is not a finite number end the engine as out of step.

One file holds the running state, the deepest checkpoint, every QSA layer's K/V up to the conversation's length and
the draft layer's K/V. Format v1, little-endian, fixed-width integers, IEEE-754 floats (a big-endian build does not
compile): a 64-byte header (magic `STRSESS\x01`, version u32, header size u32, model fingerprint u64, config
fingerprint u64, payload length u64, two reserved u64 that must be 0, a hash of the first 56 bytes), the payload
(geometry, layer range, cvec flag, the live state, the checkpoints, the K/V layers; every array preceded by its u64
count), the payload hash and the end marker `STRSEND\x01`. The hash is a 64-bit function with xxHash64-style rounds,
not the standard XXH64 stream; it detects accidental corruption and does not authenticate a file: restore only files
this engine wrote. An unknown version is refused; a new format gets a new version number.

A file is bound to the model inputs and to the settings that change what the saved bytes mean. The model fingerprint
samples (size, first and last MiB) every file the engine loads, by its role: the GGUF shards (also
`--native-dense-gguf`, the head shards and `--embd-gguf`), the PLE shard, the pack's index/dense/embedding files and
`native_experts.txt`, every file the expert source resolved - the pack's `experts.bin`, or for a pack read in place each
layer's gate/up/down GGUF as `native_experts.txt` names it, also one outside the CLI shards - and the MTP's files; not
other files in those folders, and not the path, so a moved model folder still matches. A
change in the middle of a file that keeps its size is not seen: do not change model files while their sessions are
kept. The config fingerprint covers the engine version string (another version is refused; two builds of the same version
are not told apart - there is no build hash), CUDA or HIP,
`--kv`, `STRATA_KV_ROT`, `--kv-resident`, `--max-context`, `--mtp-window`, the resolved rope configuration (type,
base, factor, freq scale, original context, the YaRN knobs - the cached K is post-RoPE), the loaded control vector (a
digest of the tables uploaded: every file's content times its exact scale, the mode, the layer range and the
direction) and the arithmetic switches (`--native-*`, `--no-ple`, the A/B arms). Sampling, seeds, draft tuning and
the expert tier are not in it: they change what comes next, not what the saved cells hold.

A save writes a temporary file with a new hidden name beside `path` (created exclusively, never an existing file or
link), flushes it to the disk, renames it over `path` (`MoveFileExW` with write-through on Windows, which promises no transaction
on every filesystem) and, on POSIX, flushes the folder (Windows has no folder flush; the file flush and the
write-through rename are all it does). A save that fails before the rename keeps the old file at `path` and removes
only its own temporary file. Once the rename is done the old file is gone: if the folder flush then fails, the save
fails with `published` set - the new file's bytes are complete and flushed, but its name may not survive a power loss.
A filesystem that cannot flush a folder (`EINVAL`) is not a failure; the engine logs it. A restore opens `path` without following a symbolic link (or a Windows reparse point) and refuses
anything but a regular file with one name; it checks the size, the header, both fingerprints (before the payload is
parsed; the first 16 MiB block, header included, is already read), that the parse's peak (the image, the read buffer, the per-segment overhead) fits in RAM above the parking
floor (`--conversation-cache-min-free-mib`), the file's size against the largest this session can restore, the
geometry and layer range before any state array, and every count against the bytes left and this session's exact
limits (context and cells, checkpoints, layers, each running-state array, each K/V part) before allocating it, the payload hash
and then the usual snapshot validation - all before any device write, and a refusal leaves the current session as it
was. A transfer failure after the device writes began ends the engine (`FATAL`) rather than decode from a partial
state; the server reports `500` and starts it again. A restore does not park the outgoing session. Not supported with
`--layer-split`, `--peer-device`, `--batch` (the config's `"parallel"`, #465; the server answers `501`) or
`--prompt-cache 0` (the RAM conversation cache need not be on). On Linux the file
moves with `O_DIRECT` in 16 MiB blocks when the filesystem takes it (buffered I/O otherwise, or with
`STRATA_SESSION_BUFFERED=1`); on Windows with buffered I/O. The engine has been run on Linux/CUDA only. An earlier
revision's CPU file-I/O test passed as a 32-bit Windows executable under Wine; the current code has not been built
for Windows, and the Windows engine, HIP and AMD cards have not been run.

A session file saves conversation state, not all of the process's execution history. Exact future token replay
across restarts is not guaranteed: expert residency and CPU/GPU rounding can change later output. In a 63K test the
first 32-token continuation after a restore matched the process that kept running; on the next continuation three
restored processes agreed with one another but differed from that process from the 28th token on. A refused,
corrupted restore in between did not change the restored processes' continuation. The cause of this divergence has
not been isolated.

Measured on an RTX 4070 Ti (12 GB), Ryzen 9 5900X, 64 GB RAM, NVMe ext4, IQ3_XXS, a 63,025-token conversation
(engine 0.1.38 with this change, binary sha256 `3bbe4fc3...`): file 1.20 GB, save 0.65 s including the flushes to
disk (10 more saves of the 1.20 GB state after a further turn: 0.59-0.61 s replacing one file, 0.80-1.18 s to new
names), restore 1.05 s in a new engine process, then the next 32-token turn in 1.14 s with the same 32 output tokens as
the same turn without a restart (1.00 s); the cold first turn takes 25.5 s. The save reported 73 `SESSION` lines, never
more than 16 MiB apart. A symbolic link, a second hard link and a file with one flipped payload byte were refused
(the last also while a restored conversation was live, which then went on to answer); a fresh engine with another
`--rope-freq-base` refused the file and then answered; a save with a RAM floor above the machine's RAM was refused as
`memory` before any copy, the file it would have replaced untouched. The times are single runs.

**Current limits (v1):** one request at a time unless `"parallel": N` is set (opt-in batch slots, up to N requests
decoded together: [BATCHING.md](BATCHING.md)), and one conversation cached at a time (switching between two chats
re-reads the other one unless the opt-in cache above is enabled, or each conversation keeps its own batch slot); images only when set up with them (below); no video. **Temperature / top_p / top_k / min_p /
seed** are honored per request (OpenAI and Anthropic fields), and so are stop strings (OpenAI `stop`, a string or up
to 4; Anthropic `stop_sequences`): the answer ends before the first one, which is not sent, and the engine stops
there (`finish_reason` "stop"; `stop_reason` "stop_sequence" with `stop_sequence` set to the one found); with the default adaptive expert tier a sampled result
is not reproducible run to run - for seed-reproducible output add `--adapt-every 100000` (static residency) to the
engine arguments. The run config's optional `sampling` block sets the defaults for requests that leave the fields out
(`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`); a request's own fields always win. **With no
block at all (0.1.45) a request without sampling keys gets the sampling the model card recommends**, picked by the
request's thinking state: Qwen3.8-Flash-Next (and Gyro) thinking temperature 1.0, top_p 0.95, top_k 20; not thinking temperature 0.7,
top_p 0.8, top_k 20, presence_penalty 1.5. Qwen3.6-35B-A3B thinking 1.0 / 0.95 / 20 with presence_penalty 1.5, not thinking
0.7 / 0.8 / 20 with 1.5 (min_p 0 and repetition_penalty 1 are "off"). Order of precedence: the request, the web page's shared Chat settings,
the config's `sampling` block, the card. Greedy is only for who asks for it: `temperature: 0` in the request, or `"sampling": {"temperature": 0}` in the config
(before 0.1.45 a config without a block ran greedy, and greedy decoding is what makes small local models loop; #2685). The web page sends no sampling of its own until you change its Sampling settings (it showed and sent 0.6 before); an old stored 0.6 with the default top_p and top_k counts as not set. The penalties (`presence_penalty`, `frequency_penalty`,
`repetition_penalty`, with `penalty_last_n` capping how many recent tokens they count over, default 64 when any
penalty is set; llama.cpp's names `repeat_penalty` and `repeat_last_n` work too, in the config's `sampling` block and in a request, #1819) ride the same path; they count the tokens the request has consumed, so a repetition penalty
suppresses what the model itself just said, not the prompt alone. Since engine 0.1.19 they apply to every token
the speculative decoding checks at once, exactly as if it decoded one token at a time (before, only the first of
each batch got them). That makes requests with penalties 1-11% slower than in 0.1.18: the draft layer guesses
without penalties, so more of its guesses are now rejected. Requests without penalties are unchanged. `top_k` keeps at most 64 candidates: `0` ("off") or anything above 64 uses all 64.

**Batch MTP on a layer split (`--batch-mtp`).** Our measurement on 2x R9700 with all experts in VRAM: 80.2 -> 62.7 tok/s (-21.9%, 0/5 pairs faster) against the pipelined default, so it pays only when the experts do not all fit in VRAM (docs/BATCHING.md).

**Token biases (opt-in).** The chat API supports request-scoped `logit_bias` objects and token/bias pair lists on supporting CUDA/HIP engines without continuous batching. A value of -100 (or `false` in a pair list) excludes a token. Unsupported engines return an error. See [LOGIT_BIAS.md](LOGIT_BIAS.md) for formats, limits and validation.

**Sampled drafting (opt-in, 0.1.40.2).** With temperature above 0 the engine keeps a draft only when it equals the token the model sampled for that position, and the draft layer proposes its best guess. Two opt-in switches change how the draft layer drafts a sampled request; greedy requests (temperature 0) are never touched and stay byte-identical. `STRATA_SPEC_COUPLED=1` (or `--coupled-draft`) drafts by sampling with the request's own settings and the random number the checking row will use; the text for a seed does not change. `STRATA_SPEC_PROB=1` is speculative rejection sampling: the draft layer samples its guess from its own distribution q, the check accepts it with probability min(1, p/q) (p is the model's distribution after penalties, top_k, top_p, min_p and temperature) and otherwise samples the replacement from the leftover distribution, so every token is distributed exactly as without drafts (the text for a seed then depends on the drafts, so it is reproducible only for the same engine, flags and prompt). Guesses without a distribution (prompt lookup, suffix drafts) keep the exact-match rule, which is already exact. Neither is faster in a way that holds up: on an RTX 3060 (IQ3_XXS, 200-token story and code answers, 10 interleaved pairs) the default's median output speed was 42.4 / 43.2 / 43.3 tok/s at temperature 0.3 / 0.7 / 1.0, coupled drafting 40.6 / 42.4 / 43.3 and rejection sampling 41.0 / 42.6 / 42.6, with run-to-run differences of up to 5% between repeats; drafts kept per round fell at 0.3 and rose by about 5 points at 1.0, which the cost of the longer sampled windows ate. `STRATA_SPEC_DEPTH=1` prints the acceptance by draft depth for each request; `STRATA_SPEC_MIN_TEMP=0.9` limits sampled drafting to hotter requests. Proof that the rejection rule keeps the distribution: `spec_prob_test` (chi-square, 2 million trials per case) and `spec_verify_parity` (GPU against the host reference).

---

### The Responses API and Codex CLI

`POST /v1/responses` (0.1.39, #451) speaks OpenAI's newer Responses API, which Codex CLI uses (it no longer
speaks Chat Completions). It runs on the same path as `/v1/chat/completions`, so the thinking levels, the thinking
budget, the conversation cache and the same API key, Host and Origin checks apply.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex does,
with `store: false`). `previous_response_id`, `conversation`, `background` and the retrieve/delete/cancel endpoints
are refused with an error that says so.

| Request | What Strata does |
| --- | --- |
| `input` as a string, or as items | `message` items (`user`, `assistant`, `system`, `developer`; text and images), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)` |
| `instructions` | The system message (with leading `developer` messages; later ones become user messages, as on the chat path) |
| `tools` | `function` tools, `namespace` tools (the model sees `namespace.name`; calls come back with `namespace` and `name`), `custom` tools (one free-form `input` string). Hosted tools (`web_search`, `file_search`, ...) are left out: the model cannot run them |
| `tool_choice` | `"none"` hides the tools; anything else lets the model choose (it cannot be forced) |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; without it the config's `reasoning_effort` (0.1.44.1; 0.1.44 ignored it here), else the model's default (high) |
| `max_output_tokens` | The output cap (thinking included). Running out ends the response `incomplete` (`max_output_tokens`) |
| `text.format` | `json_schema` and `json_object` use the [JSON response formats](#json-response-formats) (checked, not constrained) |
| `thread_source` `thread_title` in `client_metadata["x-codex-turn-metadata"]` | Codex's thread-title request (opt-in: `"codex_thread_titles": true` in `strata-<model>.json`, off by default), sent beside each user turn from another session with `tools: []`: answered without the engine as `{"title": "<the user's line, cut to 36 characters>"}`, so the conversation keeps the prompt cache. The title is the user's line, not one the model wrote |
| `temperature`, `top_p`, `reasoning_budget_tokens`, ... | As on the chat path |

The model's thinking comes back as a `reasoning` output item with `reasoning_text` content (streamed as
`response.reasoning_text.delta`). This model writes no separate summaries, so `summary` is empty. With
`"include": ["reasoning.encrypted_content"]` the item also carries `encrypted_content`: an opaque string (base64, not
encrypted; the client already holds the text). Send the reasoning items back with the rest of the conversation, as
Codex does: their thinking goes back into the prompt, so it matches what the model wrote and the conversation cache
is reused.

With `"stream": true` the events are the official ones, in order: `response.created`, `response.in_progress`, then
for each output item `response.output_item.added`, its deltas (`response.reasoning_text.delta`,
`response.output_text.delta`, `response.function_call_arguments.delta`), its `...done` events and
`response.output_item.done`, and at the end `response.completed`, `response.incomplete` or `response.failed`.
While a long prompt is read, a `response.in_progress` event goes out every 15 s, which Codex counts as activity.
Errors before the answer starts have the Responses form (`{"error": {"message", "type", "param", "code"}}`); after
it started they arrive as `response.failed`.

**Codex CLI.** In `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
model = "strata"                        # any name; Strata answers with its model
model_provider = "strata"
model_context_window = 32768            # the context you chose in setup: Codex compacts before it gets there
show_raw_agent_reasoning = true         # show the model's thinking (it writes no summaries)
# model_reasoning_effort = "medium"     # none, low, medium or high; default: the model's (high)

[model_providers.strata]
name = "Strata (local)"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # a first, long prompt can take minutes to read
# env_key = "STRATA_API_KEY"            # only if the server has an api_key: the variable holding it
```

Then run `codex` (or `codex exec "..."`) as usual. Codex warns `Model metadata for ... not found` for a local model
name; that is expected. Measured with Codex CLI 0.160.0 and Q2_0 on an RTX 5070: its first prompt (instructions and
tool descriptions) was 9,443 tokens, read in 10 s; in a tool loop, each later turn reused about 96% of the prompt from
the cache and read only the new part in 1-2 s. On Windows, Codex's sandbox rejected every shell command in that test
until it was started with `-c 'windows.sandbox="unelevated"'` (a Codex setting, not Strata's).

**Codex's compaction (opt-in: `"codex_compaction_cache": true` in `strata-<model>.json`).** When the context fills up (or
on `/compact`), Codex sends the conversation once more with a request to summarize it, and with `tools: []`. The template
writes the tools at the top of the prompt, so that prompt would share only its first few tokens with the conversation the
engine holds and be read again from the start, at its longest. With the option on, Strata keeps which Codex conversation
sent the last prompt other than a compaction (`session_id` and `thread_id` from `client_metadata["x-codex-turn-metadata"]`)
and the tools that prompt was rendered with; a request Codex marks `"request_kind": "compaction"`, from that same
conversation and without tools of its own, is rendered with them. Only the prompt: for the output parser and in the
response the request's tools stay what Codex sent, none, so a tool call the model writes comes back as a plain function
call without its namespace or custom-tool form. It is one entry, replaced by each such prompt; a compaction of another
conversation, a request without that metadata (Codex before 0.140) or a restart renders the request as sent, and so does
one that would not fit the context with the kept tools. Off by default: the prompt of a compaction differs from the
one the client sent. The author measured, with Codex CLI 0.160.0 and an engine that only counts the shared prompt start,
that a compaction after an 85,000-token conversation reused 84,895 of its 84,997 tokens and read 102; without it, 41 of
80,683.

## Tools from MCP servers

The chat page can give the model tools from [MCP](https://modelcontextprotocol.io) servers, as LM Studio and Claude
Desktop do: reading your files, fetching web pages, searching, anything an MCP server offers. List the servers in
`strata-<model>.json` under `"mcp_servers"` - the same shape as Claude Desktop's `mcpServers` block, which you can
also paste as it is (key `"mcpServers"`):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me\\Documents\\notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

Or keep them in their own file and start the server with `--mcp-config path\to\claude_desktop_config.json` (a file
with an `mcpServers` block; add it to the `serve/server.py` line of your run script). Restart Strata after a change.

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by Strata and spoken to over its
  stdin/stdout; `npx`, `uvx`, `python` and friends are found on `PATH` as usual (Node.js is needed for `npx`
  servers). **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport (the older SSE-only
  transport is not supported). `"disabled": true` leaves an entry out.
- The servers start with Strata, in the background; the server window says what each one offers
  (`MCP server 'files': 14 tools (...)`), or why it did not start - its tools are then left out and the chat works
  without them. The Monitor tab lists them, and the Sampling drawer has **Use tools from MCP servers** (on by
  default). A server that stops later is started again at its next call.
- In the chat each call shows as a small block (tool, arguments, result); the model reads the result and goes on,
  up to `max_rounds` calls in a row per answer. A tool that fails or takes longer than `timeout_s` (default 60 s)
  gives the model an `error: ...` result instead of ending the chat. Results longer than `max_result_chars`
  (default 20,000 characters) are cut, with a note, before the model reads them. Stop stops a running tool too.
- Only the chat page uses them. API clients (omp, Claude Code, OpenAI and Anthropic SDKs) see the API exactly as
  before and keep their own tools; a request to `/v1/chat/completions` opts in with `"strata_mcp": true` (it then
  gets `strata_mcp` tool events in the stream).

**Security.** MCP tools run on your PC with your user's rights, and **the model decides when to call them** - also
because of what it reads (a web page or a file can contain instructions). Give a filesystem server only the folders
it needs, prefer read-only tools, and don't add servers you don't trust. The tools can only be used from the chat
page itself (a request with another site's Origin or without a JSON content type is refused); if Strata is reachable
from other devices, set an API key.

**Context extension past 262K (rope scaling, EXPERIMENTAL, off unless you pick it).** The model was trained on
262,144 positions (rotary base 1e7). Rope scaling rescales the rotation angles so that longer contexts stay usable,
with llama.cpp's types and flag names. `linear` is Position Interpolation: every angle is shrunk by the factor.
`yarn` keeps the high-frequency angles, interpolates the low-frequency ones, and adds the magnitude correction
that keeps the attention temperature where training put it. **Without the flags nothing changes:** an unscaled
run computes exactly what it did before the feature existed, bit for bit. Scaled contexts need proportionally
more VRAM/RAM for the KV cache and the rope tables (~13 KB and ~0.26 KB per token).

What was measured (contributors' runs, RTX 5080 + IQ3_S, native path, in PR #84): per-position perplexity on the
same tokens, with only the scaling flag changed. At 293K tokens (1.12x the trained length), `yarn` with factor 2
lowered the NLL by 0.18 nats against both `none` and `linear` 2 (2.04 vs 2.22 / 2.22). That is 3-4x the path noise
measured at the same length. `linear` 2 was indistinguishable from `none`. At 2.7K and 32K no arm separated from the
noise. Needle tests do not tell the arms apart: the unscaled model also finds a needle at 413K. Long real-document
Q&A worked with `yarn` 2 at 421K and `yarn` 4 at 714K (8/8 each), and a 1M-token `yarn` 4 run read end to end.
Taken together, use **yarn**. It is still experimental: the numbers come from one machine and one quant.

- setup: `START-HERE.bat --setup --context 393216` asks nothing extra - it picks the method (yarn; one
  question when run interactively) and derives the factor from the final context for you (final context /
  262,144, at least 1: 1.5 at 393K, 2 at 512K, 1 inside the trained range; `--rope-scaling`/`--rope-scale`
  override; an explicit `--rope-scale` is kept as given even when it is too small for the context actually
  served, so check it if you set one). An explicit `--rope-scaling none` for a context past
  262,144 is refused: the setup will not configure a run with the stock angles past the trained range. If
  the RAM check reduces a chosen 384K/512K back inside the trained range, an omitted method adds no
  scaling, and an explicitly chosen one stays at factor 1 - the trained angles, no expansion (not a
  switch for rope as a whole: explicitly supplied rope settings keep their behavior).
- engine: `--rope-scaling none|linear|yarn`, `--rope-scale F`, and the raw ggml knobs `--rope-freq-base`,
  `--rope-freq-scale`, `--yarn-orig-ctx` (default 262,144), `--yarn-ext-factor`, `--yarn-attn-factor`,
  `--yarn-beta-fast` (32), `--yarn-beta-slow` (1). The model file's `rope.scaling.*` keys, when a
  fine-tune ships them, are the defaults the flags override.

The scaling is fixed for the whole run - the engine stores keys in its cache after rotating them, so one
cache must never mix two scalings, and there is no per-request form. Within the trained 262,144 a scaled
run is a slightly different model: the rescaled angles, and `yarn`'s magnitude correction, apply at every
position, not only past the trained end. That is why the setup turns scaling on only for a context past
262,144. Pictures read the same scaled table (their (t, h, w) positions feed it). That should work, but it is
unmeasured: all the runs above are text.

---

## Manage Strata from your AI assistant (MCP server)

`tools/strata_mcp.py` is an MCP server for Claude Code, Claude Desktop, Cursor, VS Code, Codex and other assistants.
Once it is added, you can ask your assistant "install Strata for this PC", "start Strata" or "is Strata running?".
In Claude Code, add it with:

```bash
claude mcp add strata -- python C:\Users\you\Strata\tools\strata_mcp.py
```

It has eight tools: status (the running model, what is installed, the hardware, a recommended size), the model
list, install, start, stop, logs, a speed test, and connection settings for other apps.

Install runs `setup.py` with `--yes` in the background. Before it downloads anything, it shows the plan and waits
for your OK. Start and stop work like the run scripts and the server's own unload. The MCP server only ends
processes it started itself. It uses only Python's standard library, so it works before `.venv` exists.

The config snippets for every client, the tool arguments and the safety rules are in
[docs/MCP_SERVER.md](MCP_SERVER.md). This is the opposite direction from
[Tools from MCP servers](#tools-from-mcp-servers) above, where the Strata model calls *your* MCP tools.

---

## Images (vision)

(Qwen3.6-35B-A3B reads pictures the same way, with its own `mmproj-F16.gguf`: [QWEN36.md](QWEN36.md).)

The model has a vision encoder: [`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(0.9 GB, a 27-layer ViT plus the projector into the language model). Pictures are **on by default** wherever a model can read
them. The app, `strata launch` and setup use one rule for the reader: the **CPU reader under 24 GB** of graphics memory (nothing is
taken from the card's memory, so the experts keep their room) and the **graphics card's reader from 24 GB**, on NVIDIA and AMD
cards. You can switch pictures off (`--vision no`, or "No" in the app and in `strata launch`) or pick the reader per model
(`--vision gpu` or `--vision cpu`). The setup downloads the encoder and puts a small helper (`strata-vision`, from llama.cpp's
`mtmd` library) beside the engine; it adds it to your start script. Nothing else changes. Where a reader is missing, pictures
stay off and the screen says so: the Mac app and the Mac Terminal have no picture reader yet, and neither has the Intel Arc
engine. On AMD the card's reader is `strata-vision-hip`, a second program in the engine package; if it cannot start, the server
starts the CPU reader by itself and prints one line ([AMD_HIP.md](AMD_HIP.md)).

| Encoder on | Time per picture | Cost |
| --- | --- | --- |
| **GPU** (the default from 24 GB) | **0.1-0.5 s** (up to 1,024 image tokens) | ~1.4 GB of VRAM is kept free for it, so the expert cache is smaller: text output is a few % slower (table below) |
| CPU (the default under 24 GB) | about 3 s at 300 image tokens on 8 cores (6-13 s on 4 threads); more tokens take longer, in proportion (#767, #625) | nothing on the GPU |

A picture becomes up to 1,024 tokens of the context (a 640x480 photo: 300). The same picture sent again, as chat apps
do on every turn, is encoded only once.

**Lazy vision (the encoder's VRAM goes to the expert caches until a picture arrives):** add `"lazy": true` to the
`"vision"` section of `strata-<model>.json`. Use GPU-only KV growth with `"vram_elastic": false`, or use
`"vram_elastic": true` without GPU-only KV growth. Do not enable batch inference for lazy vision.
The encoder then does not start at startup: the ~1.4 GB of VRAM it would hold stays with the expert caches, so text
runs without the small vision penalty.  The first image request gives that VRAM back - the engine's `VRAM` command
takes one reserve per GPU - and starts the encoder; `"vram_mib"` sets how much per GPU (`[1700, 600]` for a layer
split, or `[1800]` for one GPU; the defaults match this model's encoder).  After `"idle_s"` seconds without an image
request the encoder stops and the VRAM goes back to the caches (its encoded pictures stay cached on disk).  The
first picture after an idle gap answers a few seconds later, so the encoder can start again.

**Lazy vision with the resident low-RAM mode or `--peer-device`:** neither path above exists there. The engine
refuses the `VRAM` command with the resident low-RAM mode (its caches hold experts RAM does not) and with
`--peer-device`, and GPU-only KV growth needs every expert in RAM and no `--vram-elastic`, so a resident config
cannot have it either. Started anyway, the encoder finds no VRAM to take: the first image request answers 400
with the engine's refusal, on every request after it, while `/health` stays ok. The launcher therefore turns
`"lazy"` off in such a config and starts the encoder with the model, so the expert caches size around it at
boot (the same shape as `"lazy": false`).

**Lazy GPU vision with GPU-only KV growth:** Keep `"vram_elastic": false` in the main configuration.
Set `"gpu": true`, `"lazy": true`, and `"idle_s": 900` in the `"vision"` section. The encoder loads on an
image request and unloads after 900 idle seconds. The runtime releases expert-cache chunks before the
encoder starts. When the encoder stops, the runtime restores available cache chunks and fills their
slots from the expert profile. The image embeddings remain cached on disk.

This path uses the same per-GPU controller as KV growth. It does not shrink live KV state or move the
prompt buffers. Do not enable `"vram_elastic": true` for this path: segmented expert-cache allocation
disables GPU-only KV growth.

The dual-V100 IQ3_S service was checked with a five-second vision idle timeout. GPU process observations
confirmed that the encoder started for an image request, exited after the idle timeout, and started again
for another image. Both image requests returned correct answers. A conversation screenshot with 27,978
prompt tokens returned the correct summary while both GPUs' KV pools grew to 32,768 cells. A subsequent
short text request trimmed both pools to 8,192 cells. The prompt chunk remained 8,192 tokens throughout
the VRAM transitions. The service was then returned to the 900-second idle timeout. These checks did not
exercise the full 524,288-token context.

**More image tokens (0.1.39, #625):** `--vision-tokens N` at setup (`START-HERE.bat --setup --vision cpu
--vision-tokens 768`) sets the most tokens a picture becomes - `"max_tokens"` in the `"vision"` section of
`strata-<model>.json`, which you can also edit by hand. More tokens keep more detail (small text, charts, screenshots)
and take longer to encode, on the CPU most of all; a setup run again keeps the value.

**A minimum number of image tokens (0.1.40, #767):** `"min_tokens": N` in the `"vision"` section of
`strata-<model>.json` (edit it by hand; a setup run again keeps it) is passed to the encoder as `--min-tokens N`
(mtmd's `image_min_tokens`): a small picture is scaled up to at least N tokens. llama.cpp's mtmd prints that Qwen-VL
models want at least 1,024 for grounding tasks (pointing, counting small items); the CPU default stays at 300 at most
and no minimum, because a larger minimum changes the image answers and costs encode time (about 3 s at 300 tokens on
8 cores, in proportion to the tokens).

**A Q8_0 encoder (#625):** `"mmproj"` in the `"vision"` section can point to another mmproj file of this model, for
example a Q8_0 one (llama.cpp's `convert_hf_to_gguf.py --mmproj --outtype q8_0` makes one): the encoder's library
reads quantized weights, the file is smaller (590 MiB against 865 MiB for the BF16 one, from `llama-quantize
mmproj-Qwen3.8-Flash-Next-BF16.gguf mmproj-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0`), and on the CPU it uses less RAM
(about 280 MiB less in the encoder) and can encode faster than BF16 at the default 300 tokens. Setup downloads the
BF16 file, and a setup run again keeps a file of your own that still exists. **Recommended for `--vision cpu`.**
Accuracy against BF16, per image token: the cosine of the embeddings is 0.997-0.999 on average (#625's report, with
a community Q8_0 file, at 300, 768 and 1,024 tokens), and 0.9988 and 0.9987 on two pictures with the file made by
the command above (300 tokens; the worst single token 0.94-0.96). Encode time above 768 tokens is the same as BF16's
within about 3%. Numbers on more pictures (charts, small text) are welcome in #625.

**Flash attention on the CPU (#660):** ggml's fast CPU kernel for flash attention needs the encoder's head size (72)
to be a multiple of the vector width: 8 floats with AVX2, which the release builds use, but 16 with AVX-512, which a
build from source gets on a CPU that has it (Zen 4 and newer, some Intel). There ggml falls back to a kernel that is
several times slower and accumulates in FP16, so `strata-vision` turns flash attention off on the CPU when its ggml has
AVX-512 and leaves it at `auto` (on) otherwise; `--flash-attn on|off|auto` overrides that. A 1024x1024 picture
(1,024 image tokens) on a Ryzen 7 7700X, 8 threads, an F16 mmproj, one encode per fresh process:

| Build | Flash attention | Encode | Peak RSS | Output vs FP32 attention |
| --- | --- | --- | --- | --- |
| AVX2 (as released) | on (default) | 8.3 / 9.9 s | 1,152 MiB | 1.2% |
| AVX2 | off | 15.2 / 14.0 s | 2,194 MiB | 1.1% |
| AVX-512 (from source) | on | 43.5 s | 1,152 MiB | 26% |
| AVX-512 (from source) | off (default) | 13.3 s | 2,195 MiB | - |

The last column is the relative L2 distance of the embeddings from the AVX-512 run without flash attention; two builds
differ by about 1% anyway.

**A spare GPU for the encoder (0.1.33, #408):** with a card the engine doesn't use, add `"cuda_device": 2` (numbered
like `nvidia-smi`) to the `"vision"` section of `strata-<model>.json`: the encoder then runs on that card alone. Lower
`--vram-reserve-mib` in `"args"` to 700 as well, so the engine's cards keep that VRAM for the expert cache. The
encoder's card needs code in the ready-made encoder (RTX 20/30/40/50).

**The encoder only while there are pictures (opt-in, `"elastic": true` in the `"vision"` section, with the config's
`"vram_elastic": true`; [VRAM_ELASTIC.md](VRAM_ELASTIC.md#the-image-encoder-on-demand)):** the GPU encoder does not start
with the server, so the expert cache keeps all of the VRAM. The first picture's turn makes the cache keep
`"elastic_mib"` (default 2048) free and starts the encoder there (a few seconds); later pictures use it as it is, and
after `"elastic_idle_s"` (default 120) without a picture it stops and the cache takes the VRAM back. Measured on an R9700
32 GB (gfx1201, Linux, UD-IQ4_XS, the BF16 encoder at up to 1,024 tokens, #2189 for HIP's virtual memory): the encoder
took 1,339 MiB and started in 4.7 s (2.7 s the next time); a request with one picture took 8.6 s with the start and
5.0 s on a later cold start, and 2 / 5 / 10 pictures with the encoder running took 3.8 / 5.6 / 9.0 s (~0.45 s to encode
each, the same as on a second card). Without the key nothing changes.

### Sending a picture

**Terminal chat:** type `/image <path to a picture>`, press Enter, then type your question.

```
you> /image C:\Users\me\Pictures\receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API** (an `image_url` part: a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API:** an `image` block with a `base64` (or `url`) source, as usual.

JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work (the last ones are converted to PNG first; agents such as omp send
WebP). Chat apps with image upload work the same way.

### Speed with images on (4K context, measured)

| Model | Prompt tok/s, images off | Prompt tok/s, images on | Output tok/s, images off | Output tok/s, images on |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 541 | 531 | 90.7 | 86.9 |
| **IQ2_XS** | 468 | 458 | 77.1 | 74.0 |
| **IQ3_XXS** | 411 | 401 | 64.7 | 59.5 |

These numbers date from 2026-09-24 (an engine before 0.1.13). "Off" is the published setup; "on" runs with the encoder loaded on the GPU and its VRAM kept free, so ~1,000 fewer
experts fit in VRAM and the CPU computes a few more per token: 2-8% slower. For text, turning images on changes
nothing else (the output is bit-identical when the VRAM is the same).

**On a 16 GB card (0.1.45, #2406).** The encoder's room is ~850 expert slots of a 16 GB card's ~4,500, and where most experts are
computed on the CPU that moves the hit rate. Measured on Linux with the config setup writes (IQ3_S on a Tesla P100, IQ3_XXS on an RTX
A4000, 65K context, KV 8-bit, 5 rounds of fresh engine starts, medians): GPU encoder vs CPU encoder, chat decode 31.1 vs 32.6 tok/s
(P100, +4.8%) and 46.0 vs 48.4 (A4000, +5.2%), 4K prompt 315.5 vs 316.0 and 1,022 vs 1,053 tok/s, 16K prompt unchanged / +0.8%;
expert slots 3,819 vs 4,522 (P100), 4,686 vs 5,435 (A4000). The "elastic" encoder (`"vision": {"elastic": true}` with
`"vram_elastic": true`; it frees the encoder after 120 s idle, but turns `--kv-grow` off) gave the CPU encoder's text speed at 65K context (32.5, 47.7
tok/s) and 2.7 s per picture once loaded (first picture 4.5-5.8 s); at 196K context with q4_0 KV (where it cannot keep `--kv-grow`) it did not
(A4000: GPU 44.7, CPU 47.4, elastic 44.9 tok/s; +6% for the CPU encoder). A report on an RTX 4080 SUPER under Windows measured 35-44% of decode and
36-62% of prompt speed with the GPU encoder (VRAM free 61 MiB); that size did not reproduce on the Linux cards, so it may come from
Windows' VRAM management when the card is full (not measured by us). A picture on the CPU encoder takes 4-6 s (a 300-token picture) against 2.7 s on the P100's
GPU encoder (1.7 s on the A4000). Since 0.1.45 setup gives "images: yes" the CPU encoder on an NVIDIA card under 24 GB (pictures read at up to
300 tokens; `--vision-tokens 1024` for more detail); `--vision gpu` is kept with a note.

A question about a picture (a 640x480 newspaper page = 300 image tokens, a 328-token prompt; the whole request, encoder
on the GPU, measured through the API):

| Model | Answer ("MEN WALK ON MOON") | Picture encoded | Prompt | Output |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 1.9 s | ~0.1 s | 207 tok/s | 65-73 tok/s |
| **IQ2_XS** | 2.3 s | ~0.1 s | 173 tok/s | 50-60 tok/s |
| **IQ3_XXS** | 2.7 s | ~0.1 s | 144 tok/s | 41-50 tok/s |

(Short prompts run below the 4K prompt speed: a 2,048-token chunk is where the prompt path is efficient. Short answers
run below the long-output speed: the first rounds have no draft yet.)

**How it works inside:** the encoder turns the picture into rows of the same width as the model's word embeddings;
Strata puts them where the prompt has `<|image_pad|>` tokens and gives each one its 2-D position (row and column in the
picture; the model uses interleaved M-RoPE). Answers match llama.cpp's multimodal implementation token for token on our
test images.

---

**Stager waits (0.1.40.2, `STRATA_STAGER_SLEEP`):** the prompt-staging threads sleep while they wait on Linux and spin on Windows, because spinning is a little faster there (a 5070 read an 8K Q2_0 prompt 1.2% faster). `STRATA_STAGER_SLEEP=1` makes Windows sleep too, which is the choice when sharing the machine matters more than speed: on a Ryzen 9 7940HS laptop with an RTX 4070 (IQ3_S, 64K context) sleeping waits read 5,914 and 22,305 token prompts 5-6% slower while the whole-machine CPU use fell from 77-90% to 21-25% (issue #1101, thanks to midhatn). `STRATA_STAGER_SLEEP=0` forces spinning on Linux. The output is the same either way.

**Stager threads for a GGUF read in place (0.1.41, #1353):** a native pack read from its GGUF shards in place (UD-Q4_K_XL, a Q8_0 pack) copies each expert with 32 threads and a 128-deep ring when the bytes are page faults on an SSD. If the shards' pages are already in the page cache (a big-RAM box, a warm start) those copies are memory copies and the same profile ran 4-7x slower than the 4-thread default (a 124K-token Q8 prompt: 57 -> 350-410 tok/s with the default). The engine now checks a sample of the experts with `mincore` at the first prompt (Linux) and takes the RAM profile when 90% or more of their pages are resident, saying so in its log; a cold or partly cached GGUF keeps the SSD profile. `STRATA_STAGER_SSD=1` / `0` forces either profile; `STRATA_STAGER_THREADS` still wins. The bytes copied are the same.

## Short prompts: the CPU shares the experts (on by default on one NVIDIA GPU, `STRATA_PREFILL_CPU_SHARE`)

A prompt chunk of a few thousand tokens (an agent's tool result, a test's output, a short follow-up) streams every routed
expert that is not in VRAM over PCIe, while the CPU pool that decodes sits idle and RAM already holds those experts.
`STRATA_PREFILL_CPU_SHARE=auto` hands the pool the experts few of the chunk's tokens route to, measures per layer how
long each side takes and gives the CPU the share at which both end together (`STRATA_PREFILL_CPU_SHARE=0.4` fixes a
share). `auto` also checks that sharing pays: it times layers with the share and without it (the first ones alternate until three comparisons are in, then one in 29 runs the other way) and shares only while the layers that share are the faster ones. Where sharing is slower (every share was, on an RX 7900 GRE with a Ryzen 7 5700X3D under ROCm, #1282), the experts stay on the GPU apart from those measuring layers. It is off unless you set it, and then the output is byte-identical to the build without it.

**0.1.41: on by default** on CUDA builds with one GPU and no `--batch` slots, for prompt chunks below 1,024 tokens: unset behaves as `auto` with `STRATA_PREFILL_CPU_SHARE_MAX=1024`, and the engine prints one line at start saying so. Measured (interleaved whole-engine runs, 8 rounds, medians): 512 tokens -28..-34% and 1,000 tokens -20..-26% prompt time on an RTX 3060, a Tesla P100 and an RTX 5070, 8/8 pairs each; mean KL against the share off 0.004 (max 0.025). The answers can differ slightly from 0.1.40.3. **`STRATA_PREFILL_CPU_SHARE=0` turns it off and gives 0.1.40.3's exact bytes.** The layer split, batch slots and AMD (HIP) builds keep it off unless you set the variable; `STRATA_PREFILL_CPU_SHARE_MAX=3072` (the extension to larger chunks) also stays opt-in: it lost 2% at 2,048 tokens on the 5070.

With it on, chunks below 3,072 tokens are staged after their routing (only those can hand the CPU a share) instead of
streaming every expert from 1,024 tokens on; `STRATA_PREFILL_CPU_SHARE_MAX=1024` keeps the old limit. The CPU takes
experts it reads from RAM as they are: the arena, the page-locked copy of the resident RAM mode, or the mapped
`experts.bin`'s pages in the file cache (`--mmap-experts`), not only page-locked ones. Serve without `--batch`; on a
layer split every stage has the pool and one stage at a time takes it for a chunk.

**0.1.42: not armed by default when memory is short.** A 0.1.41 report (RTX 5090, RAM already tight) had chats failing and the engine restarting until `STRATA_PREFILL_CPU_SHARE=0` was set. The default is now skipped, with one line in the log saying why, when the RAM this process can get (the cgroup limit counts on Linux) is below the share's own buffers plus 3 GiB, when on Windows the commit limit (RAM + page file) has less than 4 GiB left beyond them, or when the experts it would read are file pages (the mapped `experts.bin` outside the GPU cache and the RAM copy) that the free RAM plus 3 GiB cannot keep. An explicit `STRATA_PREFILL_CPU_SHARE=auto` (or a number) is kept and only warns. The share's two page-locked buffers (about 50 MB) are now taken once, at their largest, and a failed allocation ends the share for that run instead of failing the request.

When on, the CPU's rows are computed in the CPU's own activation format, so the output changes in the last bits (first
token KL against off: mean 0.006, max 0.026 nats over 22 prompts; about half of the 32-token greedy answers on 500 and
1,000-token prompts are identical, the rest part at a near tie after about 23 tokens). Chunks of 3,072 tokens and more
read the same either way. `auto` picks the share from wall-clock timings, so the same request can be split
differently - and answer differently - from one run to the next (a resumed conversation's re-read differed in 31 of
1,536 teacher-forced argmax tokens, #1684); a fixed share repeats, `0` gives the exact bytes of a run without it.

**Compact CPU activation download.** A sharing layer downloads only the rows its CPU experts read. The layer used to copy its whole activation
tensor (T x N floats) to the host on the compute stream, before the layer's host sync, whether or not any expert went
to the CPU. At 2,048 tokens that copy is about 21 MB a layer, and 48 of them queue in front of the sync. The layer now
collects the token rows its CPU experts read, gathers them on the device and copies the compact result on its own
stream, beside the expert GEMMs. A layer that sends no expert to the CPU copies nothing. Every buffer of the share is
now taken for the share the layer has, not for the chunk's worst case. The CPU quantizes the same FP32 row bytes.
Automatic sharing still depends on timings. Identical model answers are not guaranteed.

**First Volta expert GEMM tile.** When the second tile does not select a launch,
`STRATA_PF_WMMA=1` uses a compact tile for IQ3_XXS, IQ3_S, and IQ2_S gate/up and IQ4_NL down
when the largest expert in the group has at most 128 rows.
The tile uses 128-value weight chunks and two 64-row output tiles. IQ3_XXS, IQ3_S, and IQ4_NL
use 64-value activation chunks and 44,032 bytes of dynamic shared memory. Static shared storage
adds 1,024, 2,048, and 16 bytes respectively. IQ2_S uses 32-value activation chunks and 39,936
bytes of dynamic shared memory. Its codebook adds 8,192 bytes. Each total fits the shared-memory
budget for two blocks on one V100 SM. The short kernels have a 128-register limit per thread.
Larger groups keep the original 256-row output tile to avoid extra weight decoding. Q2_0 and
the other quantized formats keep their existing tile selection.

**Prefetched Volta expert GEMM tile.** The sm_70 defaults enable `STRATA_PF_WMMA2`.
It supports IQ3_XXS, IQ3_S, and IQ2_S gate/up and IQ4_NL down.
It reads the same stored weight blocks through contiguous or indirect weight pointers.
It also supports gathered input rows.
Groups with at least 128 rows in their largest expert use a 128-token by 64-output tile.
Shorter groups use a 64-token by 128-output tile.
Both orientations use the same FP16 operand decode and FP32 accumulation order.
All variants fit the 48 KiB static shared-memory limit.
Set `STRATA_PF_WMMA2_N64=0` to retain the 64-token orientation.
Set it to `1` to force the 128-token orientation.
Set `STRATA_PF_WMMA2=0` to use only the first tile.
Unsupported types, directions, shapes, and alignments keep the first tile.

Use `pf_wmma_parity` to check short and wide groups, sparse input-row indices, indirect weights,
and output guards. Use `pf_wmma_parity --bench` on an idle GPU to measure validated IQ3_XXS,
IQ3_S, IQ2_S, and IQ4_NL shapes. This mode measures eight launches in a CUDA graph and reports
the median of seven CUDA-event windows. It excludes request setup, weight transfer, and token generation.

**Early loan refill on split V100s.** Native packs with Volta WMMA format pairs can now use
the existing early refill path by default. All stages must use Volta GPUs. A layer can use
IQ3_XXS, IQ3_S, IQ2_S, or IQ4_XS gate/up with IQ4_NL or Q2_0 down. A Q2_0 gate/up and down
pair is also supported. These pairs can differ between layers. An unsupported pair keeps the
normal post-run refill path.
Only a non-last stage can issue its refill early. Its compute streams must have finished a
batched segment whose next segment returns the loan: either prompt completion or a short-read
window. Checkpoint cuts before a short header window are eligible. A segment followed by another
batched segment keeps its scratch loan. Explicit `STRATA_PREFILL_REFILL_OVERLAP=0`, serial refill,
and blocking refill disable early refill.
The last stage keeps the normal refill path. The refills copy the same expert-weight bytes.
The stage also waits for KV-prefix prefetch DMA before it calls the refill hook. This prevents
refill writes into a staging pool while a KV copy still uses that pool.

Prompt time, medians of 10 interleaved pairs (off / auto, ms, `--expert-cache 1500`):

| machine | 512 tokens | 1,000 tokens | 2K / 4K / 16K |
|---|---|---|---|
| RTX 5070, Ryzen 5 7600, Q2_0 | 1,376 / 1,019 (-26%) | 1,788 / 1,392 (-22%) | unchanged |
| RTX 3060, Core Ultra 7 265, IQ3_XXS | 1,620 / 1,159 (-28%) | 2,196 / 1,770 (-19%) | unchanged |
| Tesla P100, Xeon E5-2690 v4, IQ3_XXS | 4,805 / 3,126 (-35%) | 6,821 / 5,085 (-25%) | unchanged |

Up to 3,072 tokens and with mapped experts, fresh prompts read after an 8K prewarm, medians of 5 interleaved rounds
(off / auto, ms; Ryzen 9 5900XT, 96 GB DDR4-3200, Windows 11, `--mmap-experts`, q4_0 KV; before this the share took no
expert here: none was page-locked):

| machine | 512 tokens | 1,000 tokens | 2,000 tokens | 3,000 tokens |
|---|---|---|---|---|
| RTX 5070 Ti (PCIe 3.0 x8), Swift 1.5 IQ3_XXS (huihui-ai's build) | 3,322 / 1,706 (-49%) | 3,990 / 2,357 (-41%) | 5,413 / 3,354 (-38%) | 5,525 / 4,309 (-22%) |
| RTX 3060 + RTX 5070 Ti (layers 0-11 / 12-47, both PCIe 3.0 x8), IQ3_S | 3,452 / 2,037 (-41%) | 4,506 / 2,680 (-41%) | 6,376 / 3,869 (-39%) | 6,733 / 5,138 (-24%) |

A coding agent's recorded conversation on the two cards (a 100K-token start, then 8 turns of 1-5K tokens of code, 128
tokens written a turn, the same tokens read by both): 137.4 s off, 130.6 s auto (reading 121.1 -> 114.3 s).

## Missed experts in a verify window: the tail is skipped (on by default on CUDA, `STRATA_ROUTE_TAIL_SKIP`)

When a card cannot hold every routed expert, most of a decode token's waiting time is spent on the experts that are not in VRAM:
their copy over PCIe or their rows on the CPU pool. In a verify window of several tokens, an expert that **every** token of the window
routes at the bottom of its top 10 (rank 7 or lower, the smallest weights) adds little to any of them. **0.1.42: on by default on CUDA builds**:
such a missed expert is neither copied nor computed, its contribution is zero, and the other experts keep the weight the router gave them
(no renormalisation). An expert that any token ranks above 7, and every expert already in VRAM, is served as before.

**`STRATA_ROUTE_TAIL_SKIP=0` turns it off** and restores the answers of a build without it byte for byte (checked: `mg_norepeat` 10 of 10 and `exact_pp` 4096 and 20000
tokens identical to 0.1.42 without the change, on the RTX 3060, the RTX A4000 alone and as a 2-GPU split, and the Tesla P100). Another rank
(`STRATA_ROUTE_TAIL_SKIP=8`, `=9`; 1-9 are accepted) is taken as given. The start-up line
`STRATA_ROUTE_TAIL_SKIP=7 is ON (default)` names the opt-out.
Since 0.1.43 the same default holds on HIP for gfx1200 / gfx1201 (RDNA4), under the same rules and with the same `STRATA_ROUTE_TAIL_SKIP=0` opt-out; other AMD parts are unchanged.

Where it applies: serve on a CUDA build with one GPU or a layer split (every stage skips by itself). Not applied, and no line printed: when every expert is
in VRAM (nothing is missed), with `--batch` / `parallel` slots (a window then mixes requests, and one request's skip would depend on another's
tokens: kept off until measured), with a peer-expert tier or a helper GPU, on HIP and SYCL builds, and (0.1.42.1, #1884) when 90% or more
of all the experts are in the GPU cache: the few misses are cheap there and the skip costs more draft acceptance than it saves (a Tesla V100 32 GB
with Coder IQ1_M, 94.8% of the experts cached, 99.9% decode hits: draft acceptance 70.1% to 68.2%, decode 1-2% slower than 0.1.41). A start-up line
says so; `STRATA_ROUTE_TAIL_SKIP=7` turns it on anyway, and `STRATA_TAIL_SKIP_MAX_CACHED=<percent>` moves the line (101 = never). On an AMD layer split the line is 60% (2x R9700 at 16 GB-equivalent: Q2_0 with 73% cached lost 6.1% story decode with the skip, IQ3_S with 45% cached gained 7-10%). The request log gets
`route tail skip: N missed experts skipped` (cumulative) while it is active.

**Answers differ slightly from 0.1.41** (`STRATA_ROUTE_TAIL_SKIP=0` brings them back). Teacher-forced KL of the decode path (the default's own greedy text and the default sampled at T=0.8, 28 chat prompts of code,
reasoning, chat, six languages and long-document summaries, 220 tokens each, about 6,000 scored tokens per set, through the real verify windows with the MTP drafts):

| | RTX 3060 12 GB, IQ3_XXS | RTX A4000 16 GB, Q2_0 |
|---|---|---|
| KL(0.1.41 behaviour \|\| tail skip), mean, greedy / sampled text | 0.0058 / 0.0056 | 0.0032 / 0.0030 |
| p99, max (greedy) | 0.061, 0.29 | 0.035, 0.32 |
| top-1 token agreement | 97.7% / 97.5% | 98.2% / 98.2% |
| perplexity of the forced text, skip / no skip | x1.011 / x1.007 | x1.005 / x1.003 |
| KL of the first token after the prompt | 0.024 / 0.033 | 0.015 / 0.016 |
| the same engine run twice (the floor; adaptive tier and CPU/GPU split depend on timing) | 0.0006-0.0007 | 0.0007 |
| `STRATA_PREFILL_CPU_SHARE` on / off (already the default) | 0.0006-0.0007 (first token 0.002-0.0035) | 0.0005-0.0007 (first token 0.002-0.004) |
| Q2_0 against IQ3_XXS, the same text | - | 0.1006 (p99 1.08, top-1 90.6%) |

The effect is not even: code and reasoning prompts have a KL of 0.002-0.004, chat and non-English prompts 0.004-0.009.
Task check, greedy, thinking off (HumanEval every second problem, 82; 30 generated reasoning tasks), default / tail skip:
RTX 3060 76 and 78 of 82 / 78 of 82, reasoning 27 and 27 / 29; RTX A4000 76 and 76 / 76, reasoning 27 and 30 / 27. The runs do not separate the settings
(one run each; a change of 3 reasoning tasks is noise).

Decode speed, `STRATA_ROUTE_TAIL_SKIP` default against `=0`, 5 interleaved rounds on fresh engines, a story, a code answer and a 6K-token document, 200 tokens each,
median over the rounds (ratio of tokens/s, default over `=0`; rounds in which the default was faster):

| machine | story | code | 6K document | all three |
|---|---|---|---|---|
| RTX 3060 12 GB, IQ3_XXS, resident experts | 1.104 (5/5) | 1.197 (5/5) | 1.241 (5/5) | **1.156 (5/5)** |
| RTX A4000 16 GB alone, Q2_0 | 1.151 (5/5) | 1.135 (5/5) | 1.101 (5/5) | **1.129 (5/5)** |
| Tesla P100 16 GB, IQ3_XXS, `--pcie-frac 0.29` | 1.149 (5/5) | 1.182 (5/5) | 1.118 (5/5) | **1.146 (5/5)** |
| two RTX A4000, layer split (hit rate 99.7%) | 0.987 (0/5) | 1.004 (5/5) | 1.002 (3/5) | 0.998 (1/5) |

The gain follows how much of a token is spent waiting for missed experts (hit rate 75-90% on the single cards in the profile); a 2-GPU
split that holds nearly every expert (hit rate 99.7%) gains nothing. It does not touch prompt reading. Switches that stay opt-in and are not part of this:
`STRATA_ROUTE_PRIOR` (adds speed, doubles the KL), `STRATA_ROUTE_RESIDENT` and `STRATA_ROUTE_RESIDENT_MTP` (the latter is not faster than the former).

## A verify window's expert set can be capped (opt-in, `STRATA_VERIFY_ECAP`)

`STRATA_VERIFY_ECAP=N` (off by default) caps the experts a speculative verify window reads. Token 0 keeps all K picks.
The speculative tail (rows 1..T-1) keeps the top `N-K` entries by routing weight; every other entry joins a kept expert
round-robin with weight 0, so the plan reads no expert for it and every combine path (each multiplies by the routing
weight) adds exactly zero. Rank is weight descending, then index ascending, so the result is deterministic.

One expert holds at most `T` rows, the window's own token count. That is what the uncapped window gives it, because a
token's top-K names different experts. Two kept entries can name the same expert, so the keep list holds different
experts only and the fold skips an expert that is full; if no expert has room, the entry keeps its id and only the
weight becomes 0. The limit matters: the CPU expert pool keeps `MAXT` rows per expert (`cpu/expert.hpp`,
`ExpertJobMulti::act/out/nact`), and the engine already refuses a window wider than `MAXT` at start-up.

The cap has a width floor too, checked at start-up against the draft window (`--spec` after the suffix/lookup
adjustment - the widest window the drafter can offer, not the pipeline buffer's capacity): `N` keeps `N-K` tail
entries for `spec-1` draft rows, so a configuration under one entry per draft row (`N < K + spec - 1`) is refused,
and one under two per row prints a warning naming both numbers. Below that floor a draft row's picks are all folded
to weight 0, so the row is computed from a fraction of its experts and the window's output changes far beyond the
low-weight flip above. `--spec 8` with `N=16` is such a configuration: 6 tail entries for 7 draft rows (70 picks).
In a paired agentic A/B on two V100s (same 448de7bb binary and prompt, only this option changed, 5 seeds x 4 turns),
the capped arm finished 12 of 20 turns as truncated reasoning with no tool call and no text, against 0 of 20
uncapped; the capped arm also accepted more drafts (62.3% against 56.7%).

The option applies to decode windows only (a window no wider than `kVerifyMaxT`). A prompt read through wider windows
commits every row, so a cap there would drop real experts with no draft to save. `N <= K` and a one-token window are
no-ops, and the default path is unchanged.

**The cap only acts when the window's tail is longer than `N-K`.** The keep list is token 0's K picks plus the best
`N-K` tail entries; when the tail holds `N-K` entries or fewer, nothing is dropped and the option is a no-op. The
window's width is not fixed: `--spec` sets a ceiling and `--spec-min-p` trims it per token. On this rig's daily
service (the server's default sampling, temperature 1.0) the engine offers `avg T` 1.82, 1.84 and 2.01 over three
512-token requests, so with K=10 the tail holds 8-10 entries and **`N=20` keeps all of them and does nothing**.
`N=16` keeps 6 and drops 2-4. Choose `N` from the workload's measured `avg T`, not from the `--spec` ceiling.

Measured on the live service at that window width, `bench/run_v100_bench.py --only 32K --seed 42 --max-tokens 256
--repeats 4`, the service restarted per arm:

| Arm | decode, four 32K requests (tok/s) | mean |
|---|---:|---:|
| off | 63.2 / 72.4 / 75.9 / 74.3 | 71.5 |
| `=16` | 99.9 / 79.9 / 74.3 / 64.5 | 79.7 |
| off again | 60.2 / 63.6 / 72.4 / 68.8 | 66.2 |
| `=16` again | 67.7 / 68.8 / 65.7 / 93.4 | 73.9 |

The two drift-paired comparisons give +11.5 % and +11.5 %. The two control arms drifted 8 % apart over the run and
the per-request spread is 60 to 100 tok/s, so read that as about +11 % with a wide error bar: the pooled medians of
the same eight samples move only +2.8 %. The prompt path is unchanged in every arm.
[The rows](../benchmarks/v100-verify-ecap-window-width-2026-10-29.json) hold both sets.

**It changes the output.** A dropped low-weight expert of a draft row can flip that row, and a flipped row can accept a
draft the exact model would have rejected. Measured on 2026-10-29 on two Tesla V100-PCIE-16GB (GPU0 on PCIe Gen3 x4,
GPU1 on x16, Ryzen 5 3600), IQ3_S, INT8 KV, 524288-token capacity, `--layer-split 16`, `--spec 4`,
`--pipeline-windows 2`, the daily service environment, one fresh server per arm and the same binary.

End to end, three independent 32K prompts per arm and 512 greedy tokens each (`bench/run_v100_prefill.py`):

| Arm | decode, three prompts (tok/s) | median | paired change |
|---|---:|---:|---:|
| off | 90.1 / 115.1 / 91.8 | 91.8 | - |
| `=20` | 110.5 / 118.1 / 121.5 | 118.1 | +20.4 / +3.0 / +29.7 % |
| `=16` | 105.7 / 127.9 / 111.9 | 111.9 | +15.6 / +12.8 / +20.1 % |

The mean paired change is +17.7 % at `=20` and +16.2 % at `=16`; the same arms read their prompts in 14051, 14071 and
14048 ms, so the prompt path does not move. The engine's own per-window report over a 32K read and a 256-token output
shows where the gain comes from: the window's own time does not fall (35.50, 36.06 and 36.88 ms/window) while the
committed tokens per window rise (2.49, 2.81 and 3.24), because the cap relaxes the verification of the draft rows.
The window's missed-expert PCIe wait does fall with it (3.92 to 2.47 and 2.19 ms/window on the second stage's GDN
layers). The end-to-end rows of `bench/run_v100_bench.py` are too noisy at 256 output tokens to separate a 17 % effect:
one arm's own repeats span 59 to 94 tok/s at 32K.

## The PCIe share of the missed experts is set from the CPU's speed (CUDA, single GPU, no `--pcie-frac`)

Of the experts missing from VRAM, a share goes over PCIe to the GPU and the rest is computed by the CPU pool; `--pcie-frac` sets the share.
Up to 0.1.41 the default came from the PCIe probe only (0.55 from 20 GB/s up), which ignores how fast the CPU is: with the route tail skip on, the
best share was 0.15-0.20 on an RTX 3060 (default 0.55), 0.20 on a Tesla P100 (default 0.29), and the RTX A4000 is flat from 0.15 to 0.40.

Since 0.1.42, on CUDA with one GPU and no `--pcie-frac`, the share starts at the old value and is refined in the first decode windows from what the engine
already times: the wall time of the CPU pool per missed expert. With `R` = expert bytes / that time (GB/s) and `L` = the probe's link speed, the share is
`L / (L + 1.85 R)`, rounded to 0.05 and kept between 0.10 and 0.60; it is changed at most three times (after 48, 96 and 96 more windows, about 10 s of
decoding), and only by a full step. Each change prints one `PCIe share:` line. A request that sets its own `pcie_frac` is left alone.
**`--pcie-frac N` always wins; `STRATA_PCIE_FRAC_DEFAULT=old` keeps the 0.1.41 rule.** Not used with a layer split, `--batch`, a peer GPU, when every
expert is in VRAM, or on SYCL or on HIP other than gfx1200 / gfx1201 (RDNA4, where the measured share is the default since 0.1.43; `STRATA_PCIE_FRAC_DEFAULT=old` opts out; on the other HIP parts `STRATA_PCIE_FRAC_DEFAULT=auto` turns it on: an RX 7900 XTX showed no regression, no gain was claimed, #2224). The canonical Q2_0 pack uses it on CUDA since 0.1.43, on Linux by default and on Windows with `STRATA_PCIE_AUTO_CANON=1` (an RTX 5070 there: code +2.5%, a short answer -3.3%). The constant 1.85 is a fit to three machines, so it lands within a step of each machine's best, not on it. On HIP, `STRATA_PCIE_AUTO_SPLIT=1` (opt-in, 0.1.43) also refines a two-stage layer split: the pool speed is read once and each stage takes the share its own probed link gives (2x R9700 at 16 GB-eq, tail skip on: Q2_0 story +4.5%, code +9.9%, 4K +8.8%; IQ3_S story -2.1%, code +8.0%, 4K +23.3%; with three or more stages the pipelined loop does not run it).

Measured (interleaved pairs of whole engine runs, new default against `STRATA_PCIE_FRAC_DEFAULT=old`, medians, route tail skip on):

| card | the pool takes | share (old -> new) | decode, new / old | pairs faster |
|---|---|---|---|---|
| RTX 3060 12 GB, IQ3_XXS | 32-37 us per expert | 0.55 -> 0.20 | **1.300** | 5/5 |
| Tesla P100 16 GB, IQ3_XXS | 84-97 us | 0.29 -> 0.20-0.25 | 1.011 | 4/5 |
| RTX A4000 16 GB, Q2_0 | 97-115 us | 0.30 -> 0.30 (unchanged) | 0.999 (same setting) | - |

Explicit-share runs are byte-identical to 0.1.41 (`mg_norepeat` 10 of 10, `exact_pp` 4096 and 20000). With the new default the answers differ a little
from the old default, as any change of the share does (CPU and GPU round differently): teacher-forced KL against the old default 0.0015 on the 3060
(run-to-run floor 0.0013) and 0.0011 on the A4000 (floor 0.0012; at that time the share was 0.25).

`STRATA_PCIE_BALANCE=1` (opt-in, CUDA, one GPU, no `--batch`, no `--pcie-frac`; #1548 by sergiywith / stuchapin909): instead of one share for every layer, each verify layer
reads over PCIe the number of its missed experts that balances the two sides. The spin kernels stamp the GPU's clock (the plan's arrival, the PCIe part's start, the wait for the CPU rows),
the host times its pool, and least squares with forgetting fit the pool's `a + c*n_cpu + d*n_pcie` and the GPU's `g0 + g*n_pcie`; the engine prints the fit (`pcie model`) at the end of `generate`.
Every 64th layer with a miss moves one expert to the other side so both fits stay measured. Unset, nothing changes (the logits equal the build before). Measured by us (5 interleaved pairs, box default flags, IQ3_XXS): RTX A4000 arena decode 48.45 -> 51.15 tok/s (+5.6%, 5/5 pairs), RTX 3060 resident 59.12 -> 58.15 (-1.6%), Tesla P100 37.95 -> 37.48 (-1.3%): opt-in, worth trying on A4000-class cards.

## The prompt chunk follows the host's memory (Linux, file tier; on by default, `STRATA_PREFILL_HOST_PLAN=0` turns it off)

When the experts that do not fit in VRAM are read from the model files through the page cache, the size of the prompt chunk decides how many times a long
prompt walks over those files, and how big the working set is each time (the chunk's activation buffers and the lent slots take RAM and VRAM from the
cache). Before engine 0.1.45.3 the chunk was always today's pick (8192 on most cards). The planner prices the host from what it can measure at start: the RAM copy and its
lent slots, the page cache that is left, the drive class (read from the GGUF path) and the PCIe link, and picks the chunk (and the ring) with the fewest
predicted bytes read per prompt. It prints one `chunk 8192 -> 3840, ring 96 -> 96: ...` line when it changes the pick, and `chunk 8192 kept: ...` when it does not.
A prompt that fits today's chunk keeps one chunk; a longer one is read in equal chunks of at most the plan's size (no stub at the end). No weight or expert
changes; only the rounding of the batched prompt path moves (first-window logits KL 0.002-0.012 against the fixed chunk, same top-1; the text matched in 30 of 30
requests; a run at a fixed chunk is bit-identical to itself).

Where it applies: a Linux file tier whose page cache is smaller than the file set at today's chunk. It does nothing, and says `kept`, when the GPU holds every
expert, when the file set fits the cache (the owner-class RTX 5070 box: 14.8 GiB set against 41.2 GiB of cache, 2024 vs 2035 tok/s at 20K), when the set can
never fit (a 16 GiB cgroup, 23.1 GiB set: 99.9 vs 98.9 tok/s at 20K, the fewest passes win), and on a fast NVMe where the bigger chunk wins even though it thrashes (P2003, 8192 kept).
Windows reads the file tier unbuffered (no page cache), so it keeps today's pick. The plan is made for a prompt of `--max-context` tokens (at most 32768, at least 4096). `STRATA_SSD_GBPS` overrides the drive speed, `STRATA_PREFILL_PLAN_TOKS`
the prompt speed (tokens/s) it assumes; `--prefill N` and `STRATA_PREFILL_RING` always win.

Measured (RTX 3060 12 GB, IQ2_XS, 20 GiB of experts in RAM, cgroup 34 GB, SATA speed emulated at 450 MB/s, page cache dropped and a fresh engine per run, 5 interleaved
pairs, plan off against the planner's `8192 -> 3840`, medians):

| prompt (tokens) | plan off, tok/s | planner, tok/s | ratio | pairs faster | drive read, GB (off -> planner) | decode after, tok/s |
|---|---|---|---|---|---|---|
| 4,101 | 138.1 | 138.2 | 1.001 (one chunk either way) | 3/5 (tie) | 12.8 -> 12.7 | 27.0 -> 28.2 |
| 20,400 | 334.0 | 822.3 | **2.46x** | 5/5 | 26.1 -> 2.7 | 25.4 -> 42.6 |
| 28,887 | 418.8 | 910.4 | **2.17x** | 5/5 | 29.5 -> 0.6 | 28.6 -> 45.7 |

The same box with its real NVMe and a 32 GB cap picks 8192 again (bigger chunk 480 vs 327 tok/s at 3328, even though it thrashes), and the 16 GiB cap is unchanged. Identity gate (fixed flags,
`--expert-cache 1500`): the plan line says `kept`. The host model is `include/strata/program/prefill_plan.hpp` with `tests/program/prefill_plan_test.cpp`.

## More opt-in switches measured for 0.1.41 (all off unless you set them)

None of these changes the default answers; each was measured against the default with interleaved off/on pairs of whole
engine runs (a fresh engine per side, warmed, 200-token greedy answers on three fixed prompts, medians; "pairs faster" counts
the pairs in which the switched arm won). They are here so you can try them on your own card.

- **`STRATA_GDN_CHUNKED=1`: the DeltaNet recurrence of the prompt, opt-in**. On CUDA it runs in chunks of 32 tokens
  (sergiywith, #1372; WY form, FP32,
  other bits than the default: first-token KL against the default was 0.002-0.009 nats on average, the same top token in
  every prompt tried). The kernel needs 128 SMs in one wave, so by itself it runs only on a card with 128 or more
  (RTX 5090: 1.8x on the recurrence). On smaller cards `STRATA_GDN_CHUNKED=2` forces it for a measurement, and it is
  slower there: 0.73x on an RTX 3060 (28 SMs) and 0.87x on an RTX 5070 (48 SMs) at 2K-32K tokens (`gdn_rec_parity --bench`),
  and the prompt time did not improve: RTX 5070 2K 1.01x, 8K 1.00x of the default's time; RTX 3060 2K, 8K, 16K 1.01x, no pair faster.
  On HIP, gfx1100-gfx1102, gfx1150 and gfx1151 use 16-token WMMA chunks with FP16 operands and a fused output norm; other cards and short
  chunks keep the existing path. It changes rounding too (first-token KL against the default 0.006-0.007 on average, the same top
  token in 12 of 12 prompts); on a Radeon 8060S (gfx1151) the layer's recurrence is 1.6x faster and a 4K-128K prompt after a warm-up prompt 1.4-2.1%; see
  [the HIP conditions and numerical checks](AMD_HIP.md#opt-in-chunked-prompt-gdn-on-gfx11).
- **`STRATA_FS_SLOTS=N`: the Foresight swap space** (q8atnight, #1348): N VRAM slots per layer, refilled on a copy stream
  with the experts that just missed and with the ones the next layer's router predicts (`STRATA_FS_AHEAD=0` for the
  misses only), so that an expert the cache lacks can be computed on the GPU instead of by the CPU. The model's own router
  decides everything; the slots only hold copies of the same bytes (`STRATA_FS_VERIFY=1` reads every landed slot back and
  compares it: 0 bad in 2,300+ reads on an RTX 5070). The slots are taken from the VRAM the expert cache would have
  (raise `--vram-reserve-mib` by about N x 80 MiB), and in every box measured the smaller cache cost more than the slots
  returned: decode 4 slots, median tok/s B/A (pairs faster): RTX 3060 0.95 (0/6) with misses only, 0.91 (0/6) with the
  look-ahead; Tesla P100 0.97 (2/6) and 1.00 (4/6); 16 slots on the P100 0.96 (0/6); RTX 5070 0.98 (1/6). The reporter
  measured the same on 2x RTX 3090 (the misses per layer are below one there, so the CPU round trip stays). Left off.
- **`STRATA_STAGE_PIN`: pinned stage buffers** (Zhong Uncle, #1237; **opt-in, `STRATA_STAGE_PIN=1`**. It was on by default in
  development, but the 0.1.41 release gate found IQ3_S decode corrupted after a 4096-token prompt with it on, likely a stage
  buffer recycled before its asynchronous copy finished; it stays off until that is fixed). Where the experts are served from the GGUF in
  place (`--mmap-experts`, or too little RAM for the arena) the buffers the cache fill copies from are page-locked, so the
  copy no longer goes through the driver's bounce buffer (about 0.4-0.6 GiB of pinned RAM, one buffer falls back to
  pageable when the driver refuses). Decode, median tok/s on, off, 6 pairs of whole runs: Tesla P100 with the RAM capped at
  24 GB (cgroup) 13.9 -> 15.0 (+8%, 6/6 pairs faster); RTX 5070 `--mmap-experts` +1.2% (5/6); RTX 3060 `--mmap-experts`
  -0.1% (5/6). Boxes that hold all experts in RAM never use the stage buffers.
- **`STRATA_ADAPT_LAG=2`: the adaptive tier's copies are waited for one window later** (#764). Decode, 6 pairs: Tesla P100
  (PCIe 3.0 x16) +3.5% (6/6 pairs faster), RTX 5070 +0.2% (4/6), RTX 3060 -1.3% (0/6), so it stays opt-in.
  Where the tier does not wait for its copies (`STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG` above 1) the device's residency table now follows each eviction before the next window (a small patch kernel before every verifier launch; #1517 by sergiywith): before, the pool could compute an evicted expert on the CPU from a stale activation (`STRATA_ADAPT_CHECKRES=1` counts such layers; `STRATA_ADAPT_EVICT_SYNC=0` turns the fix off). The default (the tier waits) is untouched.
- **`STRATA_ADAPT_OVERLAP` (`=1` turns it on, on every backend; the startup log names the state and, when it cannot run, why): the adaptive tier's copies land while the next
  window runs.** With a fixed swap schedule, the window takes the swaps at the same moment as with #463's wait (so the same experts run
  on the GPU and the CPU in every window), but the copies go in layer order and the verify window holds a layer's plan back
  until that layer's copies have landed, instead of the whole window waiting for the round (about 10 ms every fourth window on
  an A4000, PCIe 3.0 x16). One GPU, the whole arena in RAM, no `--adapt-async`, `--pipeline-windows`, `STRATA_ADAPT_LAG`,
  `STRATA_ADAPT_NOWAIT` or `STRATA_DMA_BATCH`. A layer split (two or more GPUs) never runs it, in any version: the startup log says
  `off: not on a layer split` (0.1.43 printed that line only when the variable was set, and ran without the overlap on a split just the
  same, so 0.1.43.1's opt-in default changes nothing there; measured on 2x A4000, #2281). On by default on CUDA on Linux since 0.1.43, opt-in on Windows and HIP;
  opt-in on every backend since #2010, after an RTX 5090 Laptop lost 21% decode with it (one Volta card is on by default again, see
  the end of this entry). What it is worth per box: an RTX A4000 (6
  interleaved pairs each, decode median tok/s: Q2_0 16 GB 62.5 -> 66.2 (+5.9%), Q2_0 as a 12 GB card 51.3 -> 54.2 (+5.7%),
  IQ3_XXS 16 GB 48.6 -> 52.6 (+8.3%), IQ3_XXS as 12 GB 39.3 -> 41.5 (+5.5%), 6/6 pairs faster every time) and a Tesla P100
  (+4.4%) gain, but an RTX 5070 on Windows loses 2.6% on Q2_0 (IQ3_XXS no faster) and an RTX 5090 Laptop (WSL2, Core Ultra 9
  275HX, native IQ3_XXS pack, `--expert-cache-per-layer`) loses 21%: 84.6 vs 106.6 tok/s medians of 5 generations of 256
  tokens from one build, the slowest of the fast run above the fastest of the slow one (#2010).
  **One Volta card on Linux CUDA: on by default** (`include/strata/core/nv_defaults.hpp`, `STRATA_SM70_PERF=0` or `STRATA_ADAPT_OVERLAP=0`
  turns it off; a layer split still never runs it). Without `--adapt-swaps` the round's count follows the recent verify
  window (`include/strata/core/adapt_budget.hpp`). Counts above the legacy minimum of 96 must fit half of that
  window's work time at the measured host-to-device rate. The hard ceiling is 384. The time excludes the overlap
  gate's wait, so a slow copy round cannot raise the budget that caused it. The budget runs only in the normal
  serial single-GPU serve path. A batch window, a helper cache, a resident RAM source, `--adapt-async`,
  `--pipeline-windows`, `STRATA_ADAPT_LAG`/`STRATA_ADAPT_NOWAIT` or an explicit `--adapt-swaps N` keeps the fixed count.
  Without a valid measured rate or window the count stays 96, also when the measured budget is below that minimum.
  The time-based count can change cache residency between runs. It does not guarantee identical token IDs.
  An explicit `--adapt-swaps N` fixes the count; `0` disables the copies.
- **`STRATA_ADAPT_CALLBACK_WARMUP` (Linux, on by default since 0.1.43.1; =0 turns it off): start the GPU host-callback thread before the CPU
  pool pins the host thread.** A callback thread created later can inherit the host's one-CPU affinity and compete
  with the verifier spinning for its signal (#1975). The warm-up runs and synchronizes one no-op host callback before
  constructing the pool; swap selection, per-layer callbacks and residency updates stay unchanged. On an RTX 5090 /
  Ryzen 9 9950X, a paired 4K three-prompt test averaged 142.82 -> 207.75 tok/s. Other GPUs and drivers remain untested;
  see the [reproducer, settings and limitations](../bench/results/2026-10-11-rtx-5090-callback-affinity/README.md).
- **`STRATA_ADAPT_SIGNAL=host`: the overlap's per-layer "landed" signal as a host function** (the 0.1.43 way). From 0.1.43.1 the default is a
  4-byte stream-ordered copy into pinned memory, which needs no driver callback thread: on a 20-core desktop (RTX 3060) the host
  functions still cost ~7 ms per window with the thread started early, because the pool's workers occupy every other CPU.
  **On Windows the host function stays the default** (0.1.44, #2188): under WDDM the copy and its signal waited in the
  driver's command buffer while the window spun on the pinned int, and the first request stopped with "the adaptive
  tier's copies for layer 45 never landed" (RTX 4070 SUPER, driver 617.42). `STRATA_ADAPT_SIGNAL=copy` asks for the copy
  there; the engine now flushes the copy stream after queueing the swaps, which is meant to fix it but has not been run
  on Windows yet.
- **`--prefill-head-alias` (`--serve`, one GPU): the output head's VRAM holds the prompt buffers while a prompt is read**,
  and the head is read back in before the first generated token. It was not measured with the paired method above. On
  an 8 GB RTX PRO 2000 Laptop (Windows, IQ3_XXS, 590 MiB reserve) it let the prompt path use 2,048-token chunks
  instead of 512: prefill 292.5 -> 667.1 tok/s on a 29,854-token prompt (medians of 12 and 6 fresh-engine trials).
  Reads of up to 512 fresh tokens cost about 0.6-0.75 s more per request, and the card had 0 MiB free while decoding.
  Details, limits and the measurement: [PREFILL_HEAD_ALIAS.md](PREFILL_HEAD_ALIAS.md).
  `STRATA_ADAPT_NOWAIT` or `STRATA_DMA_BATCH`; otherwise the log says why it is off. RTX A4000, 6 interleaved pairs each,
  decode median tok/s: Q2_0 16 GB 62.5 -> 66.2 (+5.9%), Q2_0 as a 12 GB card 51.3 -> 54.2 (+5.7%), IQ3_XXS 16 GB 48.6 -> 52.6 (+8.3%),
  IQ3_XXS as 12 GB 39.3 -> 41.5 (+5.5%), 6/6 pairs faster every time.
- **`STRATA_MMQ_DIRECT=1`: the prompt's MMQ reads resident experts in their cache slots** (CUDA, native packs; needs an
  engine built with `-DSTRATA_MMQ_TAB=ON`, which is off by default). The prompt path multiplies its experts in groups of
  16 with llama.cpp's MMQ, which wants a group's weights in one buffer at a fixed stride, so every expert a chunk routes
  to is copied there first (2.3-3.1 MB each, per chunk and layer), resident ones too. With this on, a Strata-side copy of
  MMQ's outer scheduler (`src/prefill/moe_mmq_tab.cuh`) takes one pointer per expert instead, and a resident expert is
  read where it is; streamed, peer-GPU and other experts are still copied. The launch (tile size, stream-k split,
  fixup) is the one MMQ makes for the copied group and the tile code is llama.cpp's own, so the answers are the same
  bits (`prefill_mmq_tab_test`: 874 products bitwise equal to the copied path, all nine types - Q2_0, IQ2_XXS, IQ2_XS,
  IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0 - and every tile size; the optional Q5_0 and K-quant experts get no
  pointer table and keep the copy). Each sized cache slot keeps 256 bytes after its expert, because MMQ reads up to
  136 bytes past a 640-wide down matrix; they are zeroed before each group. That is about 2.3 MiB over 9,300 slots (one
  slot fewer out of 6,998 on one RTX 3090). `STRATA_MMQ_DIRECT=2` keeps that layout but copies as before (an A/B
  reference); 0 or unset is the default path, byte for byte. `STRATA_MMQ_DIRECT_LOG=1` adds each stage's count of experts
  read in place to the log after every prompt. Under HIP, on a build without `STRATA_MMQ_TAB`, or when CMake turned
  that option off (it does so, with a message, when the llama.cpp tree's `mmq.cuh` differs from the one the copy
  mirrors), the variable is ignored.
  **Why the build option is off:** the engine loads every kernel at start (`CUDA_MODULE_LOADING=EAGER`), and the nine
  pointer-table instances (+15 MB of binary) take 36-38 MiB of VRAM on every GPU (sm_86), with the variable unset too -
  expert slots every model would lose, as with `STRATA_MMQ_KQUANTS`. They also cost ~70% of the upstream MMQ instances'
  compile time per CUDA arch (175 s of CPU at sm_86). Without the option the device code is byte for byte that of a
  build without the feature.
  Measured on 4x RTX 3090, UD-IQ4_XS, split `12,24,36` with `--trim-stage-weights` and `--batch 16` (every expert
  resident), the engine driven directly, its "read in" time, medians of 8: a 170-token prompt 331 -> 251 ms (-24%) and
  369 -> 290 ms (-21%) in two series, a 1,217-token prompt 773 -> 634 ms (-18%) and 838 -> 675 ms (-19%). A hack that
  skipped the copies altogether ran 225 and 577 ms, but it is no bound for an exact implementation: it changes the
  routing (12,003 expert-layer entries fell to 777 on one prompt) and so does less work. On one RTX 3090 with most
  experts streamed from RAM the copies of streamed experts stay, and the prompt time did not move (within 1.5%).
  Exactness, with `STRATA_PREFILL_CPU_SHARE=0` and no adaptive swaps: the first decode window's logits after the prompt
  (`STRATA_DUMP_FIRST_LOGITS`, fp32) and 32 greedy tokens were byte-identical between 2 and 1 (and 0) for prompts of
  180-2,619 tokens - on the 4-GPU split with `--prefill 2048` (chunks of 1,024 tokens and more: the streamed walk),
  the same with 9 GB less expert cache per GPU (20-31% of the expert reads streamed, so groups mix in-place, copied and
  mid-group flushed experts), and on one GPU with `--expert-cache-per-layer`; the 150 greedy tokens of each of
  `tools/batch_test.py --n 16` prompts (solo and batch) were identical too, and to those of a build without the
  option. The uniform cache (no profile) holds no resident expert when a native pack's prompt is read (serve needs the
  profile, and a one-shot run fills the cache only while it decodes), so `expert_cache_slot_guard_test` covers it.

## Thread placement: `--aux-cpus` (Linux, off by default)

The CPU pool pins its workers (one logical CPU per physical core) and the host thread pins itself to the pool's reserved
core. A thread the engine starts after that inherits the host's one CPU: the adaptive tier's job thread, the prefill
helpers, the router look-ahead. CUDA driver threads created before pinning can run on any allowed CPU, the workers'
included; a lazily created host-callback thread can instead inherit the pinned host's one CPU (see the warm-up above).
`--aux-cpus` puts every thread that is neither a pool worker nor the host thread on spare CPUs. It moves threads only and
never changes a result.

- **Which CPUs.** `--aux-cpus auto` (or `STRATA_AUX_CPUS=auto`): the SMT siblings of the host's core first, then the CPUs of
  physical cores that have no pool worker. A CPU that shares a core with a worker is never taken; when no CPU is spare
  the feature does nothing and the start-up line says so. A list (`--aux-cpus 24,26` or `24-27`) is used as given, minus
  the host's CPU and the workers' CPUs, and only CPUs the process may run on. `--aux-cpus off` is the default.
- **How.** Threads the engine starts pin themselves first thing: the adaptive tier's job thread and its per-round
  threads, the prefill copy threads and helpers, the router look-ahead, the PLE reader, the file readers, the stdin reader
  and the watchdog. The threads someone else started (the CUDA driver's) are moved by a sweep over `/proc/self/task`
  once the pool exists and again at the start of every request. The pool's workers and the host thread are never moved,
  and neither is a thread somebody pinned to one CPU. An explicit `STRATA_ADAPT_JOB_CPU` still wins for the adaptive tier's
  job thread. `STRATA_AUX_STAGER=0` leaves the prefill copy threads where they were started: with a one-CPU spare set up
  to 32 of them (`STRATA_STAGER_THREADS`) would share that CPU, which can slow the read of a long prompt.
- **Per request.** `"strata_tune": {"aux_cpus": 1}` (`serve/server.py` forwards `1`, `0`, `true` and `false`) moves the
  threads to the set; `0` puts every moved thread back to the placement it had. A request without the key goes back to the
  start-up setting. With `--batch` the most recently admitted request's value holds, as for the other `strata_tune` keys.
  An engine started without `--aux-cpus` plans the set anyway, so `1` takes `auto`.
- **What it prints.** At start: `strata aux cpus: threads that are neither pool workers nor the host go to CPUs 24 (on;
  N threads placed so far)`, or `(off at start; a request's aux_cpus=1 turns it on)`. A request that changes the state
  prints `strata serve: aux cpus ON/off for this request`.
- **Not covered.** Other processes (the Python launcher, a container runtime) and the engine's host thread.
- **Measured.** One machine: 2x RTX 3080 20 GB at 220 W, Xeon E5-2696 v4 (24 logical CPUs in the container, 16 pool
  workers each on its own CPU), UD-Q4_K_XL, layer split 23, `--batch-mtp`, two slots, adaptive tier on. `auto` put the
  host thread on CPU 2 and the helper threads on CPU 24. Same binary in every arm, only the flag differs. Two pairs of
  whole engine restarts in the order `auto`, off, `auto`, off (12 solo decodes and 10 two-stream rounds per restart,
  medians): solo decode 79.7 and 80.4 tok/s with `auto` against 72.1 and 72.4 off (off / `auto` = 0.904 and 0.900), two
  streams in aggregate 90.8 and 90.7 against 80.8 and 82.8 (0.890 and 0.912). The same configuration restarted twice
  moved solo by 1.009 and 1.005 and two streams by 0.999 and 1.024. Live on one engine, switching per request with
  `strata_tune` and rotating the order (20 solo requests per arm, 16 two-stream rounds): off / on = 0.888 solo and
  0.917 two streams, against 0.985 and 0.990 for an A/A control of the same setting. The host thread's involuntary
  context switches over a bench of about ten minutes were 43.8k and 42.8k with the flag off and 3.9k and 2.8k with
  `auto`. Prompt reads (24K, 25K, 51K, 104K) do not depend on the flag: off is +0.7% to +1.5% against `auto`, inside the
  restart noise. The mean and the p99 of the doorbell-to-flag time are not worse without the flag; what appears is rare
  stalls above 100 us, in 7 and 8 of 12 requests off against 0 or 1 with `auto`. That preemption of the host causes the
  loss is an inference; the counters do not prove it. The warm-up requests of each run were discarded.

---

## Experimental speed projection (EXPERIMENTAL, off by default)

**This is an experiment, not a finished feature.** It ships with Strata but stays off unless you turn it on.

A 480 KB control vector for Qwen3.8-Flash-Next (`data/experimental-speed-projection/`, see its README). After each
of layers 4-44 the engine removes one direction from every hyper-connection stream of the residual: `h -= (h . v) v`,
one unit vector `v` per layer, exactly as llama.cpp does with the package's `--cvec-mode project` patches.

**What it changes.** The vector's own package describes it as a **refusal-direction projection**: with it the model
declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set), and removing refusals removes a safety
behaviour - you are responsible for what the model writes with it on. It also shifts ordinary answers a little
(measured below). It is not an optimization in the engine: on the same text it costs 0.2-0.4% per token. What a
chat's tokens/s does with it on depends on the text the model writes (length, repetition, how well the drafts land),
so measure it on your own prompts; the Monitor marks every request ESP or stock.

**Turning it on (at setup).** `START-HERE.bat --setup` asks "Turn on the experimental speed projection?" (default:
no), or pass `--experimental-speed-projection on` (`off`, or a path to another vector GGUF). Only for the original
Qwen3.8-Flash-Next, not Swift 1.5. It writes these engine flags (llama.cpp's) into `strata-<model>.json`:

```
--control-vector-scaled <Strata>\data\experimental-speed-projection\Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

The engine log then says `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`, and the web
app's About tab lists it. (`--cvec-mode add` is llama.cpp's stock additive mode, for additive vectors.)

**Per request.** A loaded vector is on for every request unless it says otherwise: the web app's Sampling drawer has
a switch, and the API takes `"experimental_speed_projection": false` in the request body (OpenAI and Anthropic; a
config default goes in `"sampling": {"experimental_speed_projection": false}`). Switching drops the conversation
cache once, since the model state was computed the other way. Switched off, the output is token-for-token the stock
model's.

**Measured here** (Q2_0, fixed experts, 2,557 teacher-forced tokens of code, a document and a chat): the top-1 token
changes at 10% of positions, mean KL from the stock model 0.063 nats (max 4.1), perplexity +15% on code, +2.3% on
the document, +0.4% on the chat. Details: `bench/results/2026-09-27-esp/`.

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | Update the driver (NVIDIA App or nvidia.com/drivers), restart, run `START-HERE.bat` again. |
| Python or the build tools could not be installed | Install what it names (links are printed), then run it again. Everything already done is kept. |
| `port 8080 is already in use` | Strata is already running (look for its window), or another program uses the port: `START-HERE.bat --port 8081`. |
| `cudaHostRegister ... out of memory` in the log | Normal on Windows: the engine pins the experts in per-layer slices instead. Only a problem if the load then fails. |
| `ExpertCache: cudaMalloc(...) failed: out of memory` although VRAM is free | Windows' page file is off or tiny: every allocation on the graphics card is also charged to Windows' commit (RAM + page file). Set a fixed page file size, the same initial and maximum size (e.g. 65536 MB: System > About > Advanced system settings > Performance > Advanced > Virtual memory > Custom size), and restart: a page file Windows grows on demand ("System managed", or an initial size below the maximum) may not grow in time while the card's memory is charged (#60: "System managed" and 4096-32768 MB still failed, a fixed 64 GB worked). Since 0.1.19 the engine retries with a smaller cache instead of stopping, and setup warns about a page file under 4 GB (issue #60). |
| The first start takes minutes | It is reading 34-55 GB into RAM; the second start is faster while the files are in the OS cache. |
| The PC freezes for a few minutes at the start | Normal, most of all the first time (the server window says when it happens): the engine loads the experts into RAM, pins part of it for the GPU and sizes the expert cache. Wait; don't close the window. Still frozen after 10 minutes: restart the PC, close other programs, try again, or pick a smaller size. |
| `the engine stopped unexpectedly (exit code ...)` | The engine process ended mid-answer - usually out of RAM (Linux ends the biggest program: `sudo dmesg \| grep -i -E 'killed process\|out of memory'`). The next request starts it again by itself. If it repeats: close other programs or pick a smaller size. The server also warns at start when the model's experts leave less than ~6 GB of RAM for everything else. |
| Slow output, disk light busy | Not enough free RAM: close other programs, or choose Q2_0 / IQ2_XS. |
| `prompt ... exceeds the context` | The request is longer than the context you chose: run setup again with a bigger `--context`. |
| `the setup refuses --rope-scaling none for a past-trained context` | A context past the trained 262,144 needs the rotary angles rescaled (experimental rope scaling), and the setup will not configure one with the stock angles there. Let it pick (`START-HERE.bat --setup --context 393216` adds yarn and a covering factor), or pass `--rope-scaling linear` or `yarn` yourself. |
| Slower than the tables | The monitor plugged into the GPU and other GPU programs take VRAM from the expert cache; RAM running below its rated speed (enable EXPO/XMP in the BIOS) slows the CPU half. |
| `this server was started without the vision encoder` | The model was set up for text only: run setup again with `--vision gpu`. |
| Setup puts the model on an old drive after you moved or reinstalled Strata (#1070) | Setup remembers the data folder you used last in its settings file (`%APPDATA%\Strata\settings.json` on Windows, `~/.config/strata/settings.json` on Linux; the `data_dir` entry) and uses it again. Choose another place with `--data-dir DIR` (model files, packs, MTP layer) or `--models-dir DIR` (the GGUF files only), or delete the `data_dir` line; `--gguf-dir DIR` uses GGUF files you already have without copying. To move a model, copy its folder and run setup with the new `--data-dir`. |
| A picture is refused or `cannot read the image` | The file is not a picture Pillow can open (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF work). |
| Pictures are slow (3-30 s) | The encoder runs on the CPU, which is the default under 24 GB of graphics memory (the card's reader is the default from 24 GB). To use the card's reader: run setup again with `--vision gpu` (needs ~1.4 GB of VRAM; it takes room from the expert cache). |
| A request never finishes: "reading the prompt", GPU "100%" at low power | The GPU ran out of VRAM (engines before 0.1.9 could end with ~30 MiB free at large contexts). Run `START-HERE.bat` once to get engine 0.1.9 or newer; the log then says `... MiB of VRAM free with everything loaded` (a few hundred) and names the `--vram-reserve-mib` to add if it is low. |
| Output much slower than the tables above, only with a large `--context`, and the start says a small `... MiB of VRAM free with everything loaded` | The expert cache does not fit the card at that context. With `--expert-cache auto` the engine shrinks it to fit; with a fixed number it does not check again after the slots are written, and on Windows the extra is put in system memory instead of failing, so only the speed shows it. Use `auto`, a smaller number, or raise `--vram-reserve-mib` (it is deducted before the cache is sized). See "The expert cache size is a budget" under Sharing the GPU. |
| Generation stops mid-answer, GPU "100%", one CPU core busy | Fixed in engine 0.1.12 (issue #29, a race in the CPU expert pool on big-VRAM cards). Since then a request that stops moving ends with an error instead of hanging (after 2 minutes; 1 minute from 0.1.13): the log says `no progress for ... s ... (issue #29)` with where it stopped, and the next request starts the engine again. If you see that line, please open an issue with it. Engine 0.1.13 adds a stall report under it (what every expert-pool thread and the GPU handshake were doing, memory and page faults) and, on Windows, a `strata-stall-<pid>.dmp` file with every thread's stack: attach both. (`STRATA_WATCHDOG_S` sets the time in seconds; 0 turns it off.) Engine 0.1.14 fixes the stall those reports found (issue #31: with the IQ packs the host could wait forever inside the NVIDIA driver while copying experts in a verify window; the experts are now copied by a GPU kernel, `--pcie-mode dma` restores the old way). |
| `no progress for 60 s ... reading the prompt` on Linux, and the stall report says `threads waiting on the disk (state D): 16 ...` | The engine waits for the drive, not a deadlock: the n-gram table is read at random (`--ple-io direct`), which a rotational disk cannot keep up with (#605). The engine warns at start when the table is on one; `--ple-io ram` (Linux, needs RAM for the table) or the model on an SSD fixes it. Setup adds `--ple-io ram` itself on a rotational disk when the RAM holds the table (0.1.39). |
| `the engine said nothing for ... s during the request` or `... did not finish the request after it was stopped (STOP)` | Issue #481: the engine and the server lost step (the engine waits for its next command, the server for the request's end; GPU at 0 %, nothing in the log). The server ends the engine after 300 s without a line from it during a request (while a prompt is read: each chunk may take three times the previous one's time, the first one up to its tokens at 50 tok/s more), the request ends with an error and the next request starts the engine again. `"engine_silence_s": 600` in `strata-<model>.json` sets the time (0 = wait forever, as before). If you see it, please add the end of the engine log to #481. |
| `the engine said nothing for ... s and used no CPU or disk in that time (frozen ...)` | Issue #1317: a quicker version of the check above for an engine that is not slow but stopped (a deadlock inside a CUDA call, a driver stall): after 90 s without a line, if the engine process has also used no CPU time and moved no disk bytes in that time, the server ends it and the next request starts it again. An engine that is silent but still working is never ended by this check, the server prints one note (`... but is still working; it is not ended`) and leaves it to `engine_silence_s`. `STRATA_ENGINE_STALL_S=180` sets the time, `0` turns the check off (it needs `psutil`, which setup installs). |
| `out of memory: cudaFuncSetAttribute` in the log (IQ3_XXS, long prompt) | Fixed in engine 0.1.15: CUDA loaded a kernel's code when it was first needed, and mid-prompt there was no VRAM left for it. Run `START-HERE.bat` (Windows) or `./setup.sh` (Linux) once to update. |
| The desktop app or the Terminal says it could not find an engine, or that a model needs a newer engine; a Mac says an app is "damaged" | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-desktop-app-and-the-terminal). |
| Anything else | The engine log is `strata-<model>.log` in this folder. In the app: Settings, Engine, Diagnostics, View logs, Save logs. In the Terminal: `strata doctor`. |

---

## How it works

<p align="center"><img src="paper/tiers.svg" width="760" alt="memory tiers"></p>

- **GPU (VRAM):** attention and DeltaNet mixers, the gated-residual weights, routers, shared experts, output head, the MTP
  draft layer, the KV cache (from 64K: only its most-read part, the rest streams from RAM), and an **expert cache** that fills the rest of VRAM with the most-used experts (it adapts to
  the conversation while you chat).
- **RAM:** all 24,576 experts, pinned. The CPU computes the experts that are not on the GPU **in place**, at the same time
  as the GPU works on the cached ones (AVX-512 / AVX2 kernels, ggml's for the i-quants).
- **SSD:** the 28.8 GB n-gram table, a few rows per token, read unbuffered past the OS cache (`--ple-io direct`, the
  default, made for SSDs; on a rotational disk `--ple-io ram` keeps the table in RAM, #605). A start-up probe warns when direct reads measure slow (under 15K rows/s) but never switches by itself (since 0.1.43.1; `STRATA_PLE_AUTO_MMAP=1` brings back the 0.1.42 switch to `--ple-io mmap`): the bar is calibrated for NVMe, a SATA SSD tops out at 10-14K rows/s with direct's 16 reads in flight, and on one (Linux, 77.8K-token prompt) mmap read 2.1K rows/s against direct's 14.4K (#2040), while on a Windows box with a slow mapped file mmap stalled prompts (#1984). The engine reads the table
  in the format the GGUF has it: IQ4_NL (the default table), Q4_0, Q5_0, Q5_1, Q8_0, FP8 (E4M3 with a scale) or BF16.
  Measured on 4,000 random rows against the checkpoint's own BF16 table, the mean per-row error is Q8_0 0.53%, FP8 2.64%,
  Q5_1 3.78%, Q5_0 4.25%, IQ4_NL 7.60%, Q4_1 7.80%, Q4_0 8.55% (a Q8_0 table is 54 GB, an IQ4_NL one 28.8 GB). On the Q2_0 model, 700 teacher-forced tokens, the mean KL against the BF16 table is 0.0117 (Q5_0), 0.0121 (Q5_1), 0.0125
  (Q8_0), 0.0128 (FP8), 0.0140 (Q4_0) and 0.0146 (IQ4_NL), with the perplexity within 1% of BF16's either way: any change to the
  table moves the 2-bit model by about 0.012, so the formats are hard to tell apart. Setup does not offer another table;
  it only changes which GGUF the engine is given.
- **Speculation:** the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them. 2.4-3.2
  tokens per pass on average. When the reply repeats the context (code edits, quoted text), **prompt lookup** (engine
  0.1.7) drafts up to 5 tokens from the earlier copy, but only where its measured acceptance and cost say it pays:
  code edits 6-11% faster, other text unchanged. The drafts are checked like the MTP's, so the output is the same.
- **Prompts** are processed in chunks of up to 8,192 tokens (`--prefill auto`; 32,768 opt-in) with the experts
  streamed to the GPU over PCIe. Since engine 0.1.39b (#583) the streamed ring is sized in bytes for the pack (a
  native pack with bigger blobs gets fewer ring slots and keeps more cache slots) and `--prefill auto` picks the
  largest chunk that keeps that ring full. It only gives up ring slots where 0.1.39's rule held the chunk at 4,096
  tokens or less; from 6,144 on it keeps 0.1.39's ring and only looks for a bigger chunk beside it (shrinking the
  ring there measured slower: the Coder -12%), and a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one
  chunk: a smaller ring only slowed it). RTX 5070, 32K prompts against `STRATA_RING_BYTES=0`, 2-3 interleaved
  rounds: IQ3_XXS +18.5% at a 1,500-slot cache and +9% in another memory state, the Coder +3%, IQ3_S unchanged; 4K
  prompts unchanged (-0.7% to +1.1%, bit-identical). Q2_0 (a fixed `--prefill 2048`) is not affected. It changes
  the bits of a long prompt against 0.1.39 (the experts go through a different
  mix of cached and streamed groups). Measured quality, teacher-forced over the next 2,001 tokens of a long document
  against the FP16 prompt path (IQ3_XXS, `--prefill auto`): 8K prompt KL 0.042 (0.1.39: 0.054), top-1 agreement
  93.5% (92.2%); 32K prompt KL 0.020 (0.019), top-1 95.3% (95.0%) - the same band as before. `STRATA_RING_BYTES=0`
  restores 0.1.39's ring, loan and chunk choice.

- **The startup VRAM plan (#765):** before the expert cache takes what is left of the VRAM, the engine prices
  everything that must still fit beside it - the draft head, the prompt path's own buffers when it cannot borrow
  (`Prefill::bytes_needed`, exact, in place of the old `160 + chunk * 680 / 1024` estimate that ran ~7x high at a
  24,576-token chunk) and a slot for `--spec`'s residency table - and prints the plan (`strata generate: VRAM
  plan:`). A requested `--prefill` the planned cache cannot lend is booked as its own buffers before the cache is
  committed (the cache shrinks first); a chunk that fits nowhere is reduced to the largest 256-token size that
  does, said plainly; and a configuration nothing can satisfy is refused at startup with its budget and the knobs
  that make room, instead of dying between the model load and READY. Part of that price was hidden before: the
  streamed ring's ~384 separate 2.7 MiB allocations each rounded up to a 2 MiB page, ~0.5 GiB the count never saw -
  the ring is one allocation now. Measured on an RTX 4070 Ti SUPER (16 GB): at 32K context with `--prefill auto`
  the plan is exactly what the first prompt borrows (8,192 tokens, 2,487 cache slots, 1,950 tok/s prefill - the
  numbers the pre-plan engine reached); the 524K int8-KV boot with `--prefill 24576`, which died at its first
  prompt before the plan, now runs end to end (a 17,664-token chunk on its own buffers, 2,139 tok/s prefill, the
  cache keeps the slot `--spec` needs).

On this rack's production split (two V100 16 GB, 524,288-token int8 KV, `--prefill auto`,
layer-split 20) the plan prints the same deal: an 8,192-token chunk lent from the planned
caches, and the caches opened exactly as planned (8,255 slots on CUDA0, 5,750 on CUDA1).
The batch-slot fallback below still triggers on the same boot when the slots do not fit.

The full story, with measurements, bottlenecks and what comes next: **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)**.

---

## Credits and licenses

Strata itself: [MIT](../LICENSE). The model files are not part of it; their licenses apply to them (below).

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; quantizations:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
  The Coder: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (Apache-2.0 per its card); its support in Strata came from @pjgmobile's PR #54.
  Swift 1.5: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  by UkisAI. Their licenses apply to the weights.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the i-quant formats, the GPU dot products and
  dequantizers transcribed in `src/kernels/cuda/iq_kernels.cu`, the CPU backend linked for the i-quant experts, the
  `mtmd` library behind the image encoder (`tools/vision/`), and `gguf-py` used by the tools. See
  `third_party/ggml/LICENSE`.
- Ideas from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen); references in the paper.
- The web app's font: [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1, see
  `serve/web/fonts/OFL.txt`). Its Monitor tab started from @code-martin's dashboard idea (PR #22).
- The experimental speed projection's vector (`data/experimental-speed-projection/`): Qwen Community License 1.0,
  made from the model's activations (see its README).

### Start the API without loading the model

`serve/server.py --engine strata --config strata-<model>.json --lazy` (or `"lazy_load": true` in that
config) starts the lightweight HTTP API without starting the native engine or the vision encoder. The first
generation request, or `POST /load`, starts the vision encoder first when configured, then loads the engine through
the existing reload path, including `before_load` and `min_free_vram_mib`. This also works with image requests:
the load finishes before Strata encodes the image. `POST /unload` stops both processes. Eager startup remains the
default.

`POST /v1/load` and `/v1/unload` are JSON control aliases for integrations, accepting `{}` or
`{"model":"<configured model>"}` and returning model status. They require the configured API key,
`application/json`, and no foreign browser Origin. They return **409** while a request is active or queued,
and **404** for an unknown model. Existing `/load` and `/unload` behavior is preserved. `/api/health` aliases
`/health`; `/v1/status` exposes `loaded` and `auto_load`. The unloaded model remains discoverable.

Unloading and shutdown close the native engine's stdin after sending `QUIT`, allowing Windows' detached
stdin reader to see EOF. Cleanup waits for process exit before releasing handles; if forced shutdown still
times out, the server keeps ownership and reports an error rather than claiming the model was unloaded.

### JSON response formats

`POST /v1/chat/completions` accepts `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}`.
The schema must accept only JSON objects at its root: `"type":"object"`, or an `anyOf`/`oneOf` whose branches are all
object schemas (an `allOf` with an object member, or a local `$ref` to one, also counts), as apps written for
llama.cpp's `json_schema` send. A root that also allows an array, string, number, boolean or null is refused. Local
`#` references work; remote references are refused.
`json_schema` is checked with the Python package `jsonschema` when it is installed (`python -m pip install
"jsonschema>=4.23,<5"`; setup does not add it); without it the answer is only checked to be one JSON object, and the
server says so once.

This is **schema prompting followed by server validation**, not grammar-constrained decoding. One generation
is made per request, with no hidden retry. Successful responses contain a validated JSON object. Malformed JSON,
duplicate keys, non-finite numbers, schema violations and incomplete generations return **502** with
`error.code: structured_output_failed`; invalid request schemas return **400**. JSON formats combined with
tools/MCP are refused explicitly. Without `response_format`, ordinary text and tool behavior stays the same.

Structured SSE buffers the answer while sending keep-alive comments. It emits content only after validation,
then usage/timings and `[DONE]`; failures emit an SSE error and `[DONE]` without invalid content deltas.
`/v1/status.structured_output` advertises the formats, validation method and buffered streaming behavior.

### API request monitor

Off by default, since it keeps prompts and answers in memory: turn it on with `"api_monitor": true` in
`strata-<model>.json` (or `serve/server.py --api-monitor`); otherwise nothing is recorded and the two endpoints below
answer 404.
Open `/api-monitor` to inspect API traffic without opening a chat. It shows the model state, safe
load/unload controls, active/queued requests, original request bodies, output, separate reasoning and
non-stream response bodies. Total wall-clock includes FIFO waits and automatic loading; load, queue,
first-token, prompt/output tokens and engine decode timing are shown separately.

`GET /api/requests` returns compact summaries; `GET /api/requests?id=<id>` returns one retained request.
Both use the existing API-key check. The monitor retains the newest **100 requests in memory** until restart,
with **262,144 characters per input/output/reasoning/response field** and visible truncation flags. The actual API
responses are unaffected. Headers are not recorded, and the monitor key is kept in this tab's session storage.
Treat request history as sensitive input/output when exposing Strata on a network: set an API key as above.
The page uses relative URLs and works through the existing host binding or a reverse proxy.

### Prompt buffers: `bo` shares `emb` (#1454)

The prompt path's half-output buffer `bo` reuses the embedding buffer `emb`, which is dead after the first hyper-connection broadcast: T x 2560 floats less VRAM per chunk (320 MiB at 32768 rows). The planner still counts those bytes by default, so the auto chunk and the cache slots the prompt path borrows are exactly those of 0.1.40.3 and the output bits are unchanged. `STRATA_EMB_REUSE_ACCOUNT=1` lets the planner use the saved bytes: where VRAM limits the chunk it grows (RTX 3060, IQ3_XXS: 6400 to 6656 tokens, 1652 to 1670 borrowed slots, prompt about +3.9%). A different chunk changes the prompt path's rounding, so prompt residuals are not byte-identical to the default; greedy output matched in our runs. Opt-in until it has a KL measurement.

**Fused int8 prompt experts on gfx12 (RDNA4, R9700): on by default for IQ3_S and IQ3_XXS packs.** `STRATA_PF_FUSED=0` turns them off (back to the MMQ prompt kernels, the previous bits). R9700, 1 card: prompt 4K +15.7%, 16K +13..+17%, 32K +19..+20%, a 96K prompt +16% (answers identical text); first-token KL against the FP16 prompt path over 96 prompts is equal to MMQ within noise (IQ3_XXS 0.00613 against 0.00683, IQ3_S 0.00416 against 0.00372). Other quants and other GPUs are unchanged; `STRATA_PF_FUSED=1` still opts in every covered format.

**gfx12 (Linux): a bare `--prefill auto` may take 16384-token chunks.** Explicit `--prefill N`, `auto:N` and `STRATA_PREFILL_AUTO_MAX` are untouched; `STRATA_GFX12_CHUNK=0` or `STRATA_GFX12_DEFAULTS=0` keeps 8192.

**sm_70: larger automatic prompt chunks on one Volta card.** A native pack whose
arena is loaded whole in RAM can use 16384-token chunks with bare `--prefill auto`.
Mapped arenas, shared file-backed arenas, resident complements and layer splits
keep the previous ceiling. The startup VRAM plan prices the buffers with the same
loan cap and 128-slot floor. The rule keeps 8192 when the available RAM does not
cover the larger loan plus 6 GiB. Explicit `--prefill N`, `auto:N` and
`STRATA_PREFILL_AUTO_MAX` are not changed. `STRATA_SM70_CHUNK=0` keeps 8192;
`STRATA_SM70_CHUNK=1` skips the RAM-headroom check in this eligible mode.
A larger chunk changes the prompt path's rounding. Prompt residuals can differ
from an 8192-token chunk.

### CUDA diagnostics: a host thread for the verify-window launch

`STRATA_VERIFY_ASYNC_LAUNCH=1` runs `Verifier::run`'s CUDA graph launch on a separate
host thread. This lets its caller serve the CPU expert doorbells while an
instrumented `cudaGraphLaunch` blocks waiting for the graph. The caller joins the
launch and checks its real error before returning a frontier; an early service
failure releases the mapped GPU waits before joining. The switch is off by
default and applies to this single-request verifier entry point. Direct batch
and pipelined graph launches and the HIP/SYCL paths retain their existing scheduling.

For Compute Sanitizer, set this variable before running the usual model command
under `compute-sanitizer --tool memcheck --error-exitcode 86`. Build with
`STRATA_BUILD_TESTS=ON` and run `host_graph_launch_test` for the CUDA graph/CPU
signal, early cleanup and `--errors` error-propagation checks. Its `--synchronous`
mode is the negative control under memcheck: an eight-second watchdog releases
the original launch/service deadlock and the test fails.

## RTX 30 series (sm_86) and Tesla P100 (sm_60): bitwise decode switches on by default

The 0.1.45 switch set that is bit-exact on every card was measured switch by switch on a 12 GB RTX 3060, a 16 GB RTX A4000 (one card)
and a 16 GB Tesla P100, IQ3_XXS, default flags (adaptive tier on), fresh engine per run, 8 interleaved pairs, equal expert-cache slots,
with a noise arm: the whole 16-switch set gave +9.6% (3060), +14.6% (A4000) and +17.8% (P100) decode, 8/8 pairs each, and the identity
gate (10 prompts + exact_pp 4K/20K, fixed flags) is IDENTICAL with it on. Removing one switch at a time (4 rounds each) kept the ones that
paid on their architecture. They are on by default now, per architecture (`include/strata/core/nv_defaults.hpp`, set before the
engine reads its switches; a switch you set yourself, to anything, is kept):

- sm_86 (RTX 3050-3090, RTX A2000-A6000): `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1`,
  `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`, `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1`, `STRATA_QSA_ROWS=1`, `STRATA_B6_MMVQ_ROWS=1`.
  `STRATA_SM86_PERF=0` turns the set off.
- sm_60 (Tesla P100): `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_SIDE_MIXER=2`, `STRATA_FUSE_WAIT=2`, `STRATA_B6_MMVQ_ROWS=1`,
  `STRATA_FUSE_DECODE=1`, `STRATA_GR_UP75=1`. `STRATA_SM60_PERF=0` turns the set off.

The table is used when every visible card has that one compute capability. `STRATA_ADAPT_OVERLAP` (on by default on sm_70,
opt-in elsewhere) does not run beside
`STRATA_ADAPT_LAG` / `STRATA_ADAPT_NOWAIT`: set `STRATA_ADAPT_LAG=1 STRATA_ADAPT_NOWAIT=0` with it. Left out because they did not pay or
were mixed: `STRATA_EXPDEC`, `STRATA_GR_NORM_FAST`, `STRATA_RAM_ADAPT`, `STRATA_KV_HOST_REGISTER`, `STRATA_HOST_CORE`,
`STRATA_GR_DOWN_TG`, `STRATA_DEQUANT_WIDE`; `STRATA_EXCHANGE_ROTATE` never reaches READY on the 3060 and `STRATA_PCIE_BALANCE` is slower on a
single card.

The per-architecture sets themselves (no environment variables), against the engine without them, IQ3_XXS, default flags, 8
interleaved pairs of fresh engines, the same number of expert-cache slots in every arm (engine 0.1.45.3 and its test builds):

| Card | Decode before -> after | Gain | Pairs faster | Same engine against itself | With `STRATA_SM86_PERF=0` / `STRATA_SM60_PERF=0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 12 GB (sm_86) | 60.36 -> 66.52 tok/s | +10.2% | 8/8 | 0.999 | 1.002 (equal to before) |
| RTX A4000 16 GB (sm_86) | 51.16 -> 57.82 tok/s | +13.0% | 8/8 | 0.979 | 0.994 |
| Tesla P100 16 GB (sm_60) | 36.56 -> 41.76 tok/s | +14.2% | 8/8 | 0.975 | 0.991 |

Two things looked at and **not changed**. A prompt chunk above 8192 (`--prefill auto:16384` or `auto:32768`) changes prompt speed by -3% to +8%
on 12-16 GB cards at 20K tokens (at 16384 / 32768: RTX 3060 +4.5% / +4.5%, Tesla P100 +3.1% / +7.7%, RTX A4000 -3.4% / +6.0%), but it is not bit-exact on
CUDA (the residual dump differs on the P100 and the A4000; 32768 also changed the output tokens on the P100), so the default
stays 8192 on NVIDIA, except for one Volta card whose native-pack experts are whole in RAM (the sm_70 rule above raises a bare
`auto` to 16384; the same rounding caveat applies). It is exact on gfx12, where 16384 is the default. And `STRATA_ROUTE_PRIOR=0.5` stays opt-in: on an RTX 3060
(IQ3_XXS) its teacher-forced KL is 0.0126 against a run-to-run floor of 0.0012 (top-1 agreement 96.6% against 99.1%); on an RTX A4000
HumanEval scored 154/164 against 157 and 159 for the default twice, and reasoning 25/30 against 24 and 26 (agent 6/6, needles 3/3, CJK
8/8 and tool calls 5/5 for all), which is inside the run-to-run spread.

## Compute capability 12.0 cards: faster decode (bitwise) and fused prompt experts by default

Measured on an RTX PRO 5000 Blackwell 72 GB (110 SMs, sm_120, Linux, CUDA 13.2), IQ3_S with every expert in VRAM, setup's
flags (`--spec 4 --spec-min-p 0.5`), 6 interleaved pairs of fresh engines, 300-token greedy answers, medians:

| | story | code | 8K document | 4K / 20K / 98K prompt (tok/s) |
|---|---|---|---|---|
| before (tok/s) | 168.8 | 224.6 | 186.6 | 3829 / 4613 / 4767 |
| now | 192.0 (+13.8%, 6/6 pairs) | 260.0 (+15.8%, 6/6) | 211.9 (+13.6%, 6/6) | 4212 / 4870 / 4929 (+9.9% / +5.4% / +3.4%, 6/6 each) |

Decode is bitwise: with `STRATA_PF_FUSED=0` the answers are the same tokens as before and the identity gate passes (10 prompts
plus the 4K and 20K prompt dumps, with the fixed small cache and with every expert resident). The prompts change at rounding
level because of the fused prompt experts (below). What changed:

- On a 12.0 card these bitwise switches are on unless set: `STRATA_SIDE_MIXER=2` (the mixer's input-only work on graph
  branches), `STRATA_ATTN_DAILY=1` (the Daily's decode attention), `STRATA_GR_UP75=1` and `STRATA_GR_DOWN_TG=1` (the
  hyper-connection read's up projection with its loads hoisted, and its down projection with a block row per token; token
  pairs, `=2`, below 80 SMs),
  `STRATA_GU_SWIGLU=1` (new: the routed experts' gate/up, SwiGLU and q8_1 in one launch). A fixed 4-token verify round went
  13.70 -> 11.80 ms with them. `STRATA_SM120_PERF=0` turns the set off; each switch's own variable still wins in both directions.
- The interleaved small-batch matvec (`native_mmvq_il`, docs/MMVQ_IL_TABLE.md) also runs 5 to 8 tokens (it stopped at 4, and a
  5-token window cost 2 ms more than a 4-token one), with a 12.0 table for 5-8 tokens and, on 12.0 cards with 80+ SMs, its own
  2-4 token table. Bitwise (`mmvq_il_parity`, 1-8 tokens). `STRATA_MMVQ_IL_WIDE=0` / `STRATA_MMVQ_IL_120=0` turn them off.
- On 12.0 cards with 80 or more SMs the native packs' prompt experts run on the fused int8 kernels (`STRATA_PF_FUSED`, the rule
  gfx12 has) for IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS gate/up; IQ2_XXS / IQ2_XS keep MMQ until measured. An RTX 5070 (48 SMs, IQ2_XS
  pack, experts partly in RAM) was 4.7% slower on 20K prompts with them (0/6 pairs), so smaller 12.0 cards keep MMQ. Quality on IQ3_S: teacher-forced KL against MMQ 0.0002
  (5998 positions; MMQ against itself 0.0000) and 0.0010 after ~6K-token documents (1760 positions); first-token KL against
  the FP16 prompt path 0.01181 vs MMQ's 0.01236 (48 prompts of 1.5K-24K tokens); HumanEval 156/164, reasoning 26/30, agent
  6/6, needles 6/6, CJK 8/8, tool calls 4/5, the same as MMQ. `STRATA_PF_FUSED=0` keeps MMQ.
- With the cheaper rows, deeper speculation was measured again (3 pairs): `--spec 3` 1.145 / 1.020 / 1.123 of before,
  `--spec 4` 1.122 / 1.141 / 1.132, `--spec 5` 1.079 / 1.101 / 1.074, `--spec 6` 1.060 / 1.177 / 1.097, `--spec 8` 1.027 /
  1.105 / 1.069 (story / code / document). `--spec 4 --spec-min-p 0.5` stays the default.
- Setup (Linux, not WSL): when the RAM covers the model's need + the n-gram table + 16 GB (Q2_0 93 GB, IQ3_S 107 GB), the
  table is kept in RAM (`--ple-io ram`), so a new prompt does not wait for its rows from the SSD: on this card 4K -18%, 20K
  -9%, 98K -2% with a fresh engine per prompt, the same bits. On an R9700 the read was already hidden; with RAM short (a 48 GB
  cap) ram mode cost +20% prompt and +14 s of start-up, which is why smaller PCs keep the SSD reads.

Not measured: other 12.0 cards (RTX 5070), Windows, partly-resident caches, the IQ3_XXS and IQ2_XS packs.

## CUDA opt-in switches measured for 0.1.45 (RTX 3060 and RTX A4000, sm_86; all off unless you set them)

Ported from open pull requests and A/B'd as whole engine runs against the default (a fresh engine per side, IQ3_XXS, 5 pairs, medians;
bit-exact means the 10-prompt + 4K/20K identity gate with the switch on equals the gate with it off). Decode after a 30K-token prompt
unless noted; the gain is small on these two cards and larger on the Turing cards the authors measured.

| Switch | What | Answers | RTX 3060 / A4000 |
|---|---|---|---|
| `STRATA_ATTN_DAILY=1` (#2075) | decode attention: two cells per step, a merge that skips empty chunks | bit-exact | +3.2% (5/5) / +0.7% |
| `STRATA_QSA_ROWS=1` (#2059) | block scores of a verify window, queries in registers ([QSAROWS.md](QSAROWS.md)) | bit-exact | -0.3% / +0.3% |
| `STRATA_EXPDEC=1` (#2058) | gate/up of the window's resident experts with the q8_1 rows in shared memory ([EXPERTDEC.md](EXPERTDEC.md)) | bit-exact | +0.5% / +1 to +5% (short context) |
| `STRATA_SIDE_MIXER=1/2` (#1889) | the mixer's input-only work on graph branches | bit-exact | +0.5% / -0.6% (short context) |
| `STRATA_B6_MMVQ_ROWS=1` (#1418) | exact multi-column MMVQ reusing the activation fragment across rows (K-quants, IQ4_XS) | bit-exact | +0.2..1% / +0.5..2% |
| `STRATA_QSA_SPLIT_ATTN=1` (#1888) | decode attention split across blocks on the int8 mma path (sm_80+, int8 KV) | rounding-level (teacher-forced KL 0.0016 against a floor of 0.0012) | +2.8% / +3.5% |
| `STRATA_HC_UPMIX=1` (#1370) | the hyper-connection up projection and `gr_mix_r` in one kernel (sm_80+) | bit-exact here (gate identical) | prompt +0.6% / +0.4% |
| `STRATA_PROMPT_ATTN_IMMA=1` (#1367) | int8-KV prompt attention on INT8 tensor cores | rounding-level (KL at the floor) | prompt read +2.5% |
| `STRATA_HC_REQ8=1` (#1262) | the hyper-connection projections as int8 + a scale per 32 (more expert-cache slots) | rounding-level (KL 0.0018) | decode -0.2% / +2.2% |
| `STRATA_QUANT_GATHER=1` (#1368) | turns the once-per-token MoE input quantization back to the per-row gather | n/a | no difference measured |
| `STRATA_PREFILL_GROUP_GATHER=0` (#1107) | turns the short-prompt grouped gather off | n/a | on is 0.6-1.3% faster |

`STRATA_DEQUANT_WIDE` (the wide dequantization of Q4_K / Q5_K / Q6_K / IQ4_XS for the prompt's dense weights) is now on by default on
sm_86 as well as sm_61: bit-exact, and the time to the first token of a 150 / 300 / 450-token prompt fell 11-15% on the RTX 3060 and 6-10% on the RTX A4000 (5 of 5 pairs each).
`STRATA_DEQUANT_WIDE=0` turns it off.

### RTX 20 (Turing, sm_75) kernel series (opt-in, 0.1.45)

Kernels by rafatxf (#2069-#2072, #2093, #2220-#2222), imanu86 (#2081) and xrip (#1821) for compute capability 7.5. They were measured
on one RTX 2080 Ti (decode +5..6 % at 4K-32K, prompts at depth up to +30 %, per the PRs) and are bitwise equal to the kernels they
replace, but nobody on the team has an sm_75 card to confirm them, so they are off until asked for:

`STRATA_SM75_PERF=1` turns on every bitwise Turing kernel below on a 7.5 card (no other card is ever affected). Each has its own
variable, which wins in both directions (`=0` off, `=1` on): `STRATA_SM75_IL` (interleaved small-batch matvec table),
`STRATA_SM75_EXP` (expert layout mode 4), `STRATA_GR_UP75` / `STRATA_GR_DOWN_TG` (hyper-connection read), `STRATA_SCORES_HALVE`
and `STRATA_TOPK_SPLIT` (decode QSA selection), `STRATA_TOPK_SAMPLE` (prompt top-k, one read of the scores), `STRATA_ATTN_V3`
(decode attention), `STRATA_Q2_WARP` (Q2_0 matvec), `STRATA_FUSE_WAIT=2` and `STRATA_QIL` (fewer launches per decode layer),
`STRATA_TC_SM75` (BF16 dequant, 16-byte stores), `STRATA_PA_V1_PREFETCH` (prompt attention v1 prefetch),
`STRATA_SCORES_THREAD` (#1825: the prompt path's thread-per-head block scorer, bitwise the warp kernel's scores; `=1` takes it on any
CUDA card, where it measured +1..3 % prompt speed on a Tesla P100 at 30-60K tokens, so it is not a default there).

Options that change answers (other summation order) are separate and explicit: `STRATA_SELECT_SIMT=1|2` (FP32 tiled block scorer,
KL 0.01 against the warp scorer per #2071), `STRATA_PA_V2T=1` (v2 prompt attention, #2092/#2146), `STRATA_PF_FUSED=1
STRATA_PF_FUSED_SM75=1` (fused int8 prompt experts for the Q2_0 pack, #2145; only on a 7.5 card, sm_80+ is unchanged), `STRATA_BF16_TC=1` with `STRATA_XN_F16` (#2091).
