# Strata-V100

**A community-maintained V100-focused fork of [Strata](https://github.com/Niko1221/Strata).** This fork keeps NVIDIA Volta (`sm_70`) support working, develops and measures performance changes on Tesla V100 hardware, and periodically merges upstream improvements for features and fixes beyond V100.

Strata is an open-source local runtime for [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It combines GPU, system memory, and storage to run this large model on a personal computer. This fork retains the broader Strata project: other supported NVIDIA and AMD hardware, the web interface, OpenAI-compatible API, tools, image input, and upstream features remain important too.

> **Huge credit to Niko ([@Niko1221](https://github.com/Niko1221)) and the original Strata contributors.** This project builds on their work: Strata's model runtime, architecture, features, and the work that made running this model locally possible all come from the upstream project. The V100 fork would not exist without that foundation. Thank you for making it possible, and for continuing to develop Strata.

[Upstream Strata](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussions](https://github.com/jmnargi/Strata-V100/discussions)

## What this fork focuses on

- **Tesla V100 / Volta support (`sm_70`)**: preserve normal V100 builds and runtime support. V100 builds use a CUDA 12.x toolkit; CUDA 13 does not compile `sm_70` code.
- **Test machine:** two Tesla V100-PCIE-16GB cards; GPU 0 uses PCIe Gen3 x4 and GPU 1 uses Gen3 x16. The pipeline measurements below use both cards. The historical single-card benchmark uses only the x16 card. A system with two full Gen3 x16 links has not been measured here.
- **Upstream integration**: this is not a separate replacement for Strata. Upstream changes are merged periodically, with V100 compatibility checked and corrected where needed. New upstream features may have hardware requirements above `sm_70`; they do not automatically accelerate V100.
- **The wider Strata community**: improvements, bug reports, testing, documentation, and contributions for other supported platforms are welcome.

Support for V100 does not imply that every upstream feature or every other GPU configuration has been tested on V100. Check the relevant code, release notes, and measurement reports before relying on a hardware-specific feature.

## Selected upstream sync — 19 October 2026

This branch imports shared CUDA engine and server changes from upstream
commit `3d1fb480`. It keeps the fork's Volta kernels, pipeline K/V growth,
and conversation-cache changes. It excludes unrelated desktop packages and
separate Metal, AMD, and Intel backend updates. See the
[sync scope](improvements/RELEASE-2026-10-19-V100.md).
Use CUDA 12.8 with `sm_70`. An imported option does not establish a V100
performance gain.


## Upstream v0.1.39 integration

This update integrates [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39), including the intervening releases. It keeps the fork's V100 tensor-core experts, attention changes, asymmetric GPU split, and disk conversation cache.

- **Parallel requests:** set `"parallel": 2` in the model configuration, or use `./setup.sh --parallel 2`. Setup writes this default for a new Volta configuration and preserves an existing setting. The engine can decode separate conversations together. Extra requests wait for a free slot. Slots use additional GPU memory and can reduce single-request speed; if two slot sessions do not fit, the engine says so and serves one request at a time, with the configuration unchanged. See [batch slots](docs/BATCHING.md) for exactness settings, memory costs, and limits.
- **APIs and interface:** the update adds the OpenAI Responses API, model settings, conversation-cache monitoring, and per-slot metrics. See [API details](docs/DETAILS.md#using-it).
- **Performance and memory:** the update includes verify-pass launch reductions, byte-sized prompt rings, Linux huge pages, RAM-budget corrections, and multi-GPU options. An explicit `--layer-split` now loads only each stage's own layers' dense weights automatically (the upstream `--trim-stage-weights` flag and `STRATA_STAGE_TRIM` are gone); the freed VRAM goes to the expert cache. An optimization for newer GPUs does not imply a V100 speed improvement. Optional paths that change rounding must be measured before use.
- **Other platforms:** the upstream Intel Arc, older CPU, and additional AMD paths remain available. This machine does not provide hardware validation for those platforms.

V100 requires CUDA 12.x and a build that includes `sm_70`. Do not install the CUDA 13 engine on a V100. Explicit parallel configuration is recommended for existing installations; an update does not overwrite the model configuration.

**Measured on the two V100 cards (4 October 2026, final build).** A controlled before/after
run of the integrated build against the previously installed runtime, one request at a time
(no batch slots), cold, Qwen3.8-Flash-Next Q2_0 with int8 KV at the 524,288-token context,
the 20/28 layer split and MTP (`--spec 8`) unchanged; two fresh requests per size, medians:

| Target prompt | Before (tok/s) | Integrated (tok/s) | Change |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Each value is the median of two fresh requests (`reused=0`). The gains are small; two repeats
per size are not a statistical claim. All six paired first tokens matched the baseline's.
The full replies are not claimed to be identical: the adaptive expert tier keeps a different
mix of experts on the GPU and the CPU for the integrated build, and the two paths round
differently. No decode or quality change is claimed. These are the final build's measured
values (engine SHA-256 `e9ad337a…`), reported in the
[full report](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).

Parallel slots at the 524,288-token context depend on the split and the memory left on the
last stage; the measured fallback (requested 2, served one request at a time, configuration
untouched) is described in [docs/BATCHING.md](docs/BATCHING.md).

## Dual-V100 pipeline performance — 15 October 2026

The measured service profile enables `--pipeline-windows 2` with GPU order
`0,1` and layer split `16`. It uses two 16-GiB V100 cards at 175 W,
a Ryzen 5 3600, the native IQ3_S mixed-format model, int8 KV, and MTP
with `--spec 8`. The configured context capacity remains 524,288 tokens.
The largest measured prompt is 32,768 tokens.

**Long-response safety:** use a build with the
[pipeline KV-growth repair](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)
when combining this profile with `STRATA_KV_GROW=1`. The earlier pipeline
loop could access unmapped KV cells during a long response. The repair
drains speculative work before growing the mapping and then resumes
pipelining. The short-output speed measurements below did not exercise
that boundary.

### Fresh-prompt prefill through 32K

Each value is the median of three exact-token requests with no prompt reuse
and a 16-token output cap. Prefill time excludes response generation.
Throughput is prompt tokens divided by median prefill time.

| Prompt tokens | Previous prefill time | Pipeline prefill time | Previous tokens/s | Pipeline tokens/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

Fresh-prompt prefill is approximately unchanged. These small differences
do not establish a meaningful regression.

### Conversation decode

The separate HTTP comparison uses an 8,828-token initial prompt and six
continuation turns. Context reaches approximately 11.9K tokens. Each value
is the median of four requests, with arm order reversed on alternate rounds
and a 256-token output cap. New prompt tokens are the tokens processed on
that turn, not the total context size.

| Turn | New prompt tokens | Previous decode tokens/s | Pipeline decode tokens/s | Decode change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 121 | 70.20 | 79.45 | +13.18% |
| 2 | 321 | 70.60 | 82.05 | +16.22% |
| 3 | 1,086 | 76.60 | 79.00 | +3.13% |
| 4 | 128 | 73.00 | 79.90 | +9.45% |
| 5 | 313 | 74.35 | 86.55 | +16.41% |
| 6 | 1,089 | 74.15 | 80.75 | +8.90% |

The median of the six decode throughput ratios is **+11.31%**.
This is not a measured decode gain at every context size through 32K.
Generated lengths and MTP work differ with automatic CPU sharing and
resident routing. Whole-request latency is mixed; one short-turn HTTP
median regresses. Do not interpret the decode gain as a guaranteed
whole-request speedup or a matched-output comparison.

A separate fixed-setting control matches all three complete 16-token output
sequences at 2K, 8K, and 32K prompt sizes. It does not prove identical output
for every production request.

See the [evaluation report](benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
for exact timings, build provenance, rejected trials, and reproduction
commands. See the [service profile guide](docs/NVIDIA_V100.md#dual-pcie-pipeline-service-profile)
before enabling the option on another installation.

## TurboQuant KV cache — 25 October 2026

TurboQuant is an opt-in KV cache format. It rotates each 256-value head with a Hadamard transform, then
quantizes the rotated values against a Lloyd-Max codebook. Each head keeps one FP16 norm. The widths are
3, 4, 6 and 8 bits per value. The fit is minimum squared error only; this cut has no QJL residual.

Use `--kv tq6` (or `tq3`, `tq4`, `tq8`). The cache is resident only: it stays in VRAM, there is no host
copy, and `--kv-resident` is refused. The MTP draft layer keeps its INT8 cache. Setup does not offer these
values. Nothing is a default, and the daily service stays on int8.

At the configured 524,288-token capacity, with the model's 12 attention layers, the KV cache is:

| `--kv` | Bytes per cell, one layer | Full-capacity KV, 12 layers | Change from int8 |
| --- | ---: | ---: | ---: |
| int8 | 1,056 | 6.1875 GiB | — |
| tq8 | 1,032 | 6.046875 GiB | −0.14 GiB |
| tq6 | 776 | 4.546875 GiB | −1.64 GiB |
| tq4 | 520 | 3.046875 GiB | −3.14 GiB |
| tq3 | 392 | 2.296875 GiB | −3.89 GiB |

The tq6 main saving is 1.64 GiB at full capacity. These are calculated capacities, not measured
`nvidia-smi` reductions: the expert cache fills the freed VRAM, so total GPU memory stays near 15,600 MiB.

The V100 prompt reader is a second opt-in: `STRATA_PROMPT_ATTN_TQ=1` on sm_70. It decodes compressed tiles
for the prompt path; unset or `0` keeps the decode-attention fallback. It does not change the KV bytes.

Measured on two Tesla V100 16 GB cards at 175 W with IQ3_S, `--spec 4`, `--layer-split 16`,
`--pipeline-windows 2`, a cold 2,048-token output cap, and one run per size per arm. The tq6 arm had
the prompt reader on. At 32,768 prompt tokens, tq6 prefilled at 1,687 tokens/s against int8's 2,117,
and decoded at 120.2 tokens/s against 125.5. With the reader on, tq6's 51,746-token request wall time
fell from 40.3–40.4 s to 31.7–31.9 s, still slower than int8's 25.8 s. These measurements do not show a
speed gain. A needle test found 15 of 15 cells in both arms through 122,289 prompt tokens; that is not a
general quality result.

See the [full report](benchmarks/v100-turboquant-2026-10-25.md) and
[details](docs/DETAILS.md#tesla-v100-fork-benchmark).

## Gyro-L pair trellis on two V100 — 31 October 2026

This branch reads agentionai's `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` (108,323,559,808 bytes, Hugging Face
revision `05892758a67efff0895d23c6df10a6fdb6be8cf4`). Its routed experts are stored as the pair trellis types
TQP5, TQP6 and TQP7 (ggml types 147 to 149) in a Hadamard-rotated basis. The pack step
(`tools/iq_pack.py --compat-bf16`) and the kernels are described in [docs/GYRO.md](docs/GYRO.md). CPU and CUDA
are supported; Metal and SYCL have no pair-trellis kernels. This needs engine 0.1.45.6 or newer.

Two engine defects had to be repaired first. The expert kernels themselves were already bit-exact against the
authors' ggml:

- **The decode window lost the rotation.** For a folded pack the window rotated the expert input, then
  overwrote that rotated copy with the unrotated one before the experts read it, so the gate/up projections
  ran in the wrong basis on every decoded token. One-shot output looked plausible and then looped; the
  server was wrong from the first token. The window now keeps one "image written" mark per layer.
- **The layer split's prompt helper** (opt-in) rotated only its SwiGLU output and wrote a group's rows into a
  buffer laid out for the stage's WMMA tiles. It now rotates the input it is given and owns its H.

Measured on this fork's pair: two Tesla V100-PCIE-16GB (GPU 0 on PCIe Gen3 x4, GPU 1 on x16) at 175 W, a
Ryzen 5 3600, layer split 14, `--spec 4 --spec-min-p 0.70`, int8 KV with KV grow, `--pipeline-windows 2`,
262,144-token context, the service's own environment, interleaved repeats:

| Prompt tokens | Prefill (tokens/s) |
| ---: | ---: |
| 2,328 | 632 |
| 8,199 | 1,036-1,084 |
| 16,371 | 1,344-1,411 |
| 32,679 | 1,597-1,642 |

Replies decode at 68 to 75 tokens/s from 2K to 32K context (medians of repeated 520-token replies); the split
point does not move that beyond run-to-run noise. A read is limited by the experts that are not in VRAM: the
x16 card streams 15.01 GiB over PCIe for a 2.3K-token read (profile at layer split 16) and the two stages run
one after the other, so the x4 card idles about 700 ms of every read.

The pack serves through the OpenAI-compatible API: "Paris" for the capital question, a follow-up turn that
reuses 86 prompt tokens, a code request, a `get_weather` tool call and a correct answer from a tool result,
and a 442-token passage question answered from the passage.

**Layer split 14 instead of 16.** Fourteen layers on the x4 card and 34 on the x16 card read 7.9% faster at
2.3K tokens (632 against 586 tokens/s) and 6% faster at 8.2K, the same at 16.3K, and 1.5% slower at 32.7K.
15 and 13 sit between, and 20 is 23% slower at 2.3K and 13% at 8.2K. A read is what an agentic turn spends
its time on, so this profile uses 14.

**What was measured and did not pay on this pair:** `--spec 6` and `--spec 8`, `--spec-min-p 0.50` and
`0.85`, a fixed `STRATA_PREFILL_CPU_SHARE`, `STRATA_PREFILL_CPU_SHARE_MAX` above 6,144, `--prefill-pipe`,
`--max-context 131072`, `STRATA_HC_Q8=1`, and a larger `--vram-reserve-mib`. The prompt helper itself is
slower here (484 against 583 tokens/s at 2.3K): it can only take the first stage's experts, and on this pair
that stage is the x4 card, which streams 1.77 GiB of the 16.8 GiB a read moves. `--mtp` and the default
`STRATA_ROUTE_TAIL_SKIP=7` are worth keeping: decode falls to 35.5 and 48 tokens/s without them.

## V100 benchmark

The table below records a single-card run on one Tesla V100-PCIE-16GB using PCIe Gen3 x16. The test machine also has a second V100 on PCIe Gen3 x4; it was not used for this run. Results from a two-card setup or from a machine with two full-width links may differ and could be better. It is a reproducible measurement, not a promise of speed on other systems.

| Target prompt | Median prompt tokens | Prefill (tokens/s) | Decode (tokens/s) | Maximum GPU temperature |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**Test setup:** Qwen3.8-Flash-Next Q2_0; one V100 only (GPU1, PCIe Gen3 x16); Ryzen 5 3600; 48 GB DDR4-3200; CUDA 12.8; 262,144-token context; int8 KV cache; automatic prefill; MTP with `--spec 8` and draft floor 0.70; paired expert variant; 700 MiB vision reserve. Each value is the median of three fresh, uncached requests (`reused=0`) with exactly 256 output tokens. Timings are from the engine's `/metrics` endpoint.
- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

Before each request, the passively cooled card idled for at least 120 seconds and reached 55 °C or below without software thermal slowdown for 15 seconds. The starting temperatures were 51–55 °C. All requests are included. The 128K and 256K prompts reached 84 °C and experienced software thermal slowdown during prompt processing, despite the cooled start; long runs can throttle. † indicates slowdown during the prompt.

Measured 2026-10-02. Engine: fork version 0.1.31, build commit `78417ea` (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). The benchmark was recorded before the later upstream v0.1.36 integration and V100 attention PRs; it is not a measurement of the current `main` build. The historical table is kept with its original provenance. Do not interpret it as a controlled comparison against earlier tables or current builds.

Raw data: [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json), [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json), [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json), [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json), and [`GPU telemetry`](bench/results/2026-10-02-v100-single-cooled/gpu.csv). See the [full method and history](docs/DETAILS.md#tesla-v100-fork-benchmark), [benchmark instructions](benchmarks/README.md), and [V100 performance reports](benchmarks/).

Recent V100 pull requests include measured kernel-level changes and explicitly report when end-to-end gains are not established. For example, the prompt-attention port from upstream PR #600 reported 27.9% less int8 attention kernel time at its measured shape, while model prefill changes ranged from +1.3% to +3.0% in the reported runs; it did not establish a reliable decode gain. Read the [full report](benchmarks/v100-q2_0-pr600-2026-10-03.md) before comparing results.

## V100 performance work

The V100 work spans the prompt path, decode path, memory use, storage reads, and multi-GPU execution. The changes below are in the fork's merged history; each link includes its own test method, hardware details, and limitations. Kernel timing improvements do not necessarily produce the same percentage gain in full-model throughput.

### Prompt processing

- **BF16-to-FP16 tensor-core GEMMs on Volta:** Volta does not have native BF16 GEMM support. The fork converts BF16 weights exactly to FP16 and uses V100 tensor cores for prompt GEMMs instead of the scalar fallback. The associated FP16 prompt activation images use the right bit representation.
- **Faster PLE table reads:** switch POSIX PLE reads to queued `io_uring` O_DIRECT operations, with a synchronous fallback. One measured fresh 8K PLE gather dropped from about 6.9 seconds to 7 milliseconds.
- **Q2_0 MMQ prefill:** enable the existing llama.cpp Q2_0 matrix-matrix quantized path for Strata's internal expert type. A measured 14,486-token prompt improved 6.5% in prefill throughput (926.1 to 986.2–986.9 tokens/s).
- **Score only live QSA blocks:** avoid scoring blocks beyond the current prompt's live range. On a measured 15,423-token prompt, QSA selection time fell 91.1%; end-to-end prefill throughput rose from 961.0 to 1,111.3–1,112.9 tokens/s in the reported tests.
- **Tensor-core prompt attention for Volta:** use `mma.m8n8k4` on the V100 instead of the FP32 fallback. An earlier PR measured roughly 1.6–1.9× speed in the attention kernel and higher prompt throughput in its tests. The later upstream PR #600 port reduced int8 attention kernel time 27.9% at one measured shape; reported end-to-end prefill deltas were smaller (+1.3% to +3.0% in its 4K–32K tests).

### Decode and multi-GPU execution

- **Batched verification window:** run multiple speculative decode positions together, reducing per-token launch overhead. Paired historical tests reported faster output generation, but results vary by prompt and runtime conditions.
- **Tiled QSA block scoring:** reuse selected key data across query tiles instead of repeatedly reading it. The measured 256-query / 65K-block kernel fell from 3.5 ms to 1.3 ms; its per-score arithmetic remained bit-identical.
- **Fused Volta GR and batched KV append:** combine work across verification positions and append KV data in fewer launches. On the tested dual-V100 setup, decode increased 5.6–17.3% across 4K–32K prompts, while prefill was approximately unchanged.
- **Asymmetric dual-V100 tuning:** improve host-row handling, device-plan dispatch, and memory ownership for a two-card configuration. One earlier installed configuration showed prefill gains of 13.3–18.7% at 4K–32K, but decode results were mixed and both GPUs thermally slowed during that run. These values do not isolate each code change.
- **Reduce weight and dispatch overhead:** load only the layers owned by each card when using an explicit split, avoid unnecessary CPU activation quantization and empty dispatch barriers, and optionally run paired-row expert kernels. GPU0's measured native dense allocation fell from 1,376.20 MiB to 541.27 MiB. The paired-row mode is opt-in; the reported whole-model results did not establish a decode improvement.

### Attention kernels and memory traffic

- **Decode attention (upstream PR #540 port):** reduce score shuffles and shared-memory traffic while preserving accumulation order. The measured kernel time fell from 52.90 to 31.13 microseconds on GPU0 for M=1, and from 180.42 to 111.85 microseconds for M=8. The reported whole-model decode changes were small; prefill was effectively unchanged.
- **Prompt attention (upstream PR #600 port):** use the Volta tensor-core implementation with four independent MMA computations. The PR reports correctness checks and kernel-level / model-level measurements; the 27.9% kernel reduction should not be read as a 27.9% increase in full-model speed.
- **KV gather and GDN recurrence (upstream PR #627 ports):** use wider aligned int8 KV loads with a safe original-width fallback, and use multiple accumulators for the V100 prompt recurrence. Measured kernel times improved about 10% for a 32K KV gather and about 7% for a 4,096-token recurrence. Paired model prefill differed by less than 0.2%; no reliable end-to-end decode gain was established.

### Build support and evidence

The fork has also repaired CUDA configuration and linking for `sm_70`, maintained V100 device admission through upstream merges, and adapted upstream changes when their architecture requirements exclude Volta. These compatibility fixes keep the optimized paths buildable; they are not themselves performance claims.

For the full evidence and implementation scope, see [PR #1](https://github.com/jmnargi/Strata-V100/pull/1), [#2](https://github.com/jmnargi/Strata-V100/pull/2), [#5](https://github.com/jmnargi/Strata-V100/pull/5), [#6](https://github.com/jmnargi/Strata-V100/pull/6), [#11](https://github.com/jmnargi/Strata-V100/pull/11), [#12](https://github.com/jmnargi/Strata-V100/pull/12), [#15](https://github.com/jmnargi/Strata-V100/pull/15), [#16](https://github.com/jmnargi/Strata-V100/pull/16), and [#17](https://github.com/jmnargi/Strata-V100/pull/17). The results use different dates, builds, prompts, GPU placements, and protocols. They are historical measurements, not a single controlled before/after comparison or a guarantee of gains on every V100 system.

## Get started

This fork tracks the upstream installation experience. For complete and current platform requirements, model choices, and options, see the [installation guide](docs/INSTALL.md), [model guide](docs/MODELS.md), and [upstream setup guide](docs/AI_SETUP.md).

- **Windows:** download or clone this repository and run `START-HERE.bat`.
- **Linux:** clone this repository and run `./setup.sh`.
- Follow the prompts to select a model and context size. Setup downloads model data and starts the local service.
- Open the address printed by setup (normally `http://127.0.0.1:8080`).

For a Tesla V100, use the CUDA 12 engine with `sm_70` support. Setup selects the older-GPU engine for Volta; Linux builds it locally. The upstream Windows CUDA 12 archive includes Volta code, but it was not tested on this Linux machine. See the [older-GPU guide](docs/OLDER_GPUS.md) for toolkit and driver requirements. For Docker, follow the [Docker instructions](docs/INSTALL.md) and build for the target GPU architecture.

## Use Strata

- **Web app:** use the local URL printed by setup for chat and the live monitor.
- **OpenAI-compatible API:** set your app's base URL to `http://127.0.0.1:8080/v1`.
- **Anthropic-compatible API:** use `http://127.0.0.1:8080/v1/messages`.
- **OpenAI Responses API:** use `http://127.0.0.1:8080/v1/responses`. This stateless endpoint supports Codex-style tool calls; it does not store `previous_response_id`.
- **Images, MCP, multi-GPU, and configuration:** see [details](docs/DETAILS.md), [MCP server](docs/MCP_SERVER.md), and [multi-GPU guide](docs/MULTI_GPU.md).

The server normally listens on localhost. If you expose it to other machines, configure an API key and use a trusted network. Do not publish secrets or private configuration in issues, benchmark results, or pull requests.

### Conversation cache disk (opt-in)

The conversation state has three tiers. L1 is the active session and its resident prompt checkpoints. L2 is the optional host-RAM cache (`--conversation-cache-mib`). L3 is an optional disk store that keeps saved conversations and checkpoint groups across restarts. Streamed KV layers use their authoritative host pools when the engine captures state.

To use the disk store, add `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` to the engine arguments. The path and a positive GiB budget are required together. The store is off by default, it needs `--serve`, it works with `--conversation-cache-mib 0`, and it supports `--layer-split` on multiple GPUs. Optional: `--conversation-cache-disk-slots N` caps the record count, and `--conversation-cache-disk-min-free-mib N` keeps free space on the filesystem.

The engine preserves an active checkpoint before retention removes it or a request overwrites its branch. Each saved checkpoint includes the GDN state, PLE history, QSA indexer state, and the required KV backing. Token KV data alone is not sufficient to resume this model.

Related checkpoints share one KV image per archive group. The engine moves and extends that image instead of copying the full prefix for each checkpoint. It copies only new or changed KV pages when the backing is still valid. A group stops growing before it reaches the disk metadata limit. A later group has its own backing. The engine does not combine a saved checkpoint with KV bytes from a different computation of the prefix.

Archive groups use the same L2 byte and slot limits as saved conversations. L2 capacity victims move to L3 when the disk tier is enabled. A successful L2 admission does not also write the image to disk. If an image cannot fit in L2, the engine attempts to store it in L3. Physical-RAM and disk free-space limits still apply. The minimum-token setting also applies to checkpoint archives.

Preserving a checkpoint is an optimization, not a requirement. If the host RAM floor or the disk budget refuses the image, the engine drops the checkpoint, logs the drop, and continues serving. It does not stop the server. The session stays correct; only the reuse of that checkpoint is lost.

Lookup selects the longest compatible prefix across the active session, L2, L3, and idle batch slots. Tokens, image identity, and control-vector mode must match. For an edit to an earlier message, reuse ends at a saved checkpoint before the first changed token. The engine must process the suffix again; it cannot resume at an arbitrary matching token without the recurrent state at that position. Increasing `--prompt-cache` is not a substitute for checkpoint preservation.

An L2 archive remains available after restore. A valid L3 image is promoted to L2 when the byte budget and physical-RAM floor permit it. Only a bounded subset of its checkpoints becomes resident in the active session. Restoring one branch does not remove other checkpoints from the saved archive.

The disk store supports append-only conversation chains. A continuing chain can append new token IDs, checkpoints, running state, and KV pages. An L2 eviction writes a complete seed image; it does not require a GPU capture. A rewind starts a separate branch instead of replacing the source chain. The file format does not change for checkpoint archives.

Disk panels have checksums and version fields. A seed uses a temporary file and a rename. An incomplete append is recovered to the last complete panel. Invalid or incompatible records are removed. The disk store evicts least-recently-used chains to enforce its byte budget, record limit, and free-space floor. Thus, neither L2 nor L3 provides unlimited history. If both tiers have evicted the required checkpoint, the request must process the missing prefix again.

The engine captures a shared system-prompt root once into L2 when root retention is enabled. L3 receives the root on L2 eviction or failed L2 admission. With L2 disabled, an enabled L3 can receive the root directly. `--conversation-cache-keep-root` and `--conversation-cache-disk-keep-root` control root retention in each tier. Their `--no-` variants disable it. Roots have lower eviction priority than ordinary images; explicit request checkpoint pins remain protected.

The disk tier does not add concurrent execution. See [batch slots](docs/BATCHING.md) for concurrent-request memory costs. Capture must finish before a request overwrites the required session bytes. Restore is synchronous. L2 eviction writes use host snapshots and can add disk I/O latency. Size both cache budgets for the conversations and checkpoint history that you need. See [the V100 guide](docs/NVIDIA_V100.md) for the checkpoint-retention verification and [the disk benchmark report](benchmarks/v100-l3-conversation-cache-2026-10-03.md) for earlier measurements.

#### Measured V100 result

On this fork's two-V100, layer-split configuration, resuming a 33,725-token conversation from a 25 GiB NVMe cache took 1,614.6 ms to read and 315.8 ms to restore. The complete resumed prompt phase, including 22 new tokens, took 3,015.1 ms. A cold read of the same 33,725-token prefix took 20,332.7 ms. This is a **6.74x speed-up** and an **85.2% prompt-latency reduction** for the resumed request. The 943.6 MiB record restored byte-exact main-model state and identical output across both GPUs. Restart recovery passed byte-exact parity, and corrupt-record fallback produced matching output. See [the benchmark report](benchmarks/v100-l3-conversation-cache-2026-10-03.md).

Measured on this fork's two-V100 layer-split rig (Qwen3.8-Flash-Next Q2_0, int8 KV): two
conversations alternating through a scratch cache served correctly across 8 requests with
`corruptions=0 evictions=0`; one chain file per conversation grew 238 MiB -> 594 MiB across
three parks of +12 new tokens each (a full-snapshot store would have written a fresh ~2.4 GiB
file per park at 100k+ tokens).  The first park of a fresh short chat seeds the chain
(~118 MiB: one K/V page per layer plus the model's fixed running state); afterwards every park
is the delta (+12 tokens ~ 118 MiB) and the read-back/restore path is unchanged (~0.4 s to read,
~0.04-0.08 s to restore).

### Lazy vision (opt-in)

The image encoder (`strata-vision`) holds about 1.74 GiB of VRAM (1.43 GiB on the first GPU, 0.31 GiB on the second) for as long as it runs. With lazy vision it does not run between image requests: that VRAM stays with the expert caches, so text-only work keeps the full resident-expert count.

To enable it, add to the model configuration:

- `"vram_elastic": true` (the engine flag `--vram-elastic`): the expert caches are segmented, so the engine can give VRAM back and take it again between requests.
- `"vision": {"lazy": true, "idle_s": 900}`: the encoder starts on the first image request and stops after `idle_s` seconds without one (900 = 15 minutes). The encoder needs the engine's one-request-at-a-time mode, so do not combine it with `"parallel"` (or give the encoder its VRAM another way, for example `POST /v1/vram`).

How it behaves:

- A text request never starts the encoder. An image request first tells the engine to hand the GPUs' VRAM back - one reserve per GPU, `[1700, 600]` on a layer split and `[1800]` on one GPU, override with `"vram_mib"` - then starts the encoder and encodes the picture while the engine is idle.
- After `idle_s` seconds without an image request the encoder stops, and the expert caches grow back to their full size. Encoded images stay cached on disk, so sending the same picture again does not re-encode it.
- Measured on this fork's two-V100 layer split: the caches hold **14,005** resident experts while the encoder is idle (12,688 with the encoder resident), a cold image request answered in **5.9 s**, and both caches returned to 14,005 after the encoder unloaded. See [details](docs/DETAILS.md).

## Contributing

Contributions are welcome. You do not need a V100 to help: documentation, tests, setup, server behavior, API compatibility, and improvements for other supported devices are useful. V100-specific code and performance results benefit from validation on real Volta hardware.

1. Check [open issues](https://github.com/jmnargi/Strata-V100/issues) and [pull requests](https://github.com/jmnargi/Strata-V100/pulls) to avoid duplicate work.
2. For broad upstream features, check whether the change belongs in [Niko's upstream repository](https://github.com/Niko1221/Strata). This fork periodically integrates upstream changes; focused V100 fixes and measurements can be proposed here.
3. Make a focused change, describe the hardware and exact steps used to test it, and include relevant tests.
4. For performance claims, report the baseline and candidate, full runtime settings, workload, number of repetitions, and limits. Include raw data or a reproducible command when possible. Separate kernel timing from whole-model throughput and do not claim a gain that the measurements do not show.
5. Open a pull request against this repository's `main` branch. Explain whether the change is V100-specific, an upstream integration, or a general Strata improvement. Keep credentials, personal configuration, and unrelated local files out of the change.

See [`benchmarks/README.md`](benchmarks/README.md) and [`docs/DETAILS.md`](docs/DETAILS.md) for the current measurement approach. Contributions remain subject to the project license and the licenses of included components and model files.

## Project history

This fork develops V100 support and V100-specific work while retaining Strata's upstream development. Recent integration and performance pull requests illustrate that process:

- [#14 — Merge upstream v0.1.36 while preserving V100 support](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — Port measured V100 KV gather and GDN recurrence changes from upstream PR #627](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — Reduce V100 decode-attention shuffles and shared-memory traffic](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — Port upstream PR #600's Volta prompt-attention kernel](https://github.com/jmnargi/Strata-V100/pull/17)
- [#88 — Read Gyro-L's pair-trellis experts (TQP5/6/7), fix the folded decode window and the layer split's prompt helper, and move the V100 profile to `layer_split 14`](https://github.com/jmnargi/Strata-V100/pull/88)

These reports describe specific tested changes; they are not a blanket claim of faster performance across models or GPUs. For more detail, browse the [complete pull request history](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged) and [commit history](https://github.com/jmnargi/Strata-V100/commits/main).

## Credits and license

This repository is a fork of [Strata by Niko1221](https://github.com/Niko1221/Strata). **We are deeply grateful to Niko and all upstream contributors.** Their original work is the reason this fork, its V100 support, and this local model runtime are possible. We aim to credit and follow upstream work as we periodically bring in its changes.

Strata incorporates [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) and other open-source components. The project is licensed under the [MIT License](LICENSE); components and model files may have separate terms. See the [credits and license notes](docs/HOW_IT_WORKS.md#credits).
