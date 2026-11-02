# ReVolta-style request segments on two V100s: parity, full-context memory, and concurrency

**Date:** 2026-11-02
**Status:** completed evidence only. This report uses the installed baseline runs, the eight nonce-prompt `revolta-qsa-v1` A/B arms, the four plain-prompt A/S arms, the eight final plain-prompt A/B arms, two serial plain reference arms, one four-arm 39,812-token KV-stream smoke, a semantic HTTP smoke, and an edge-final structural smoke. No other runs are included.
**Compact data:** [v100-revolta-concurrency-2026-11-02.json](v100-revolta-concurrency-2026-11-02.json)

Two opt-in switches were added to the batch-MTP verify path:

- `STRATA_QSA_SEGMENTS=1` runs a contiguous run of rows of one slot through the existing multi-row QSA indexer append, block scores, top-k and attention calls, instead of one call per row.
- `STRATA_BATCH_VERIFY_WIDTH=1` lets a slot's verify group use the available row budget on serial windows.

Both are environment-variable opt-ins. Neither changes a runtime default. The A/B runs below use **A** = both off, **B** = both on, with the same candidate binary and the same server configuration. The plain-prompt isolation run uses **A** = off and **S** = `STRATA_QSA_SEGMENTS=1` only, so it tests segments without the width switch.

## Source idea: ReVolta #59 versus the existing Strata path

The idea comes from [ReVolta PR #59, "SM70: score multi-request decode batches by request segments"](https://github.com/Leonccaa/ReVolta/pull/59) ([ReVolta repository](https://github.com/Leonccaa/ReVolta)). On SM70 its paged QSA scorer loads every 64-key tile once per query row. The PR groups rows of one request into a segment, loads a tile once per segment of up to four rows, and multiplies the tile by those rows together. Segment descriptors are built on the device, so the route is CUDA-graph safe. Single-row-per-request batches keep the paged scorer. The PR reports its own validation on 4 x V100 PCIe cards at 250 W and decode on/off ratios of 1.01 to 1.24 across its DFlash2/MTP4 shapes, with C1 unchanged.

Strata already had window-row batching (`STRATA_QSA_SLOTS`, `STRATA_BATCH_ATTN_ROWS`). This change adds the per-slot run: in a `--batch-mtp` window, a run of contiguous rows of **one slot** shares one indexer key set and one K/V pool, so the run's indexer keys are read once. It reuses the existing int8/fp16 multi-row kernels and asks for bitwise equality with the per-row loop. The two implementations are not the same experiment:

| | ReVolta #59 | This Strata change |
|---|---|---|
| Target | vLLM SM70 paged QSA scorer, multi-request decode batch | Strata batch-MTP verify window |
| Grouping | Request segments, up to 4 rows; other rows' lanes zeroed | Contiguous rows of one slot; parity cases at 1, 2, 4, 8 rows |
| Data format | FP16 H4/D128, int32 metadata | int8 and fp16 K/V pools |
| Graph handling | Segment descriptors on device; CUDA-graph safe | Reuses the captured window graph; parity replays A/B/A |
| Streamed KV | Not part of the PR | Scores/top-k/attention fall back to per-row; the indexer append still runs over the run |
| Default | ReVolta SM70 profile enables it; explicit setting wins | Opt-in only; no default changed |
| Measurement host | 4 x V100 PCIe at 250 W | 2 x V100 PCIe; power cap not measured |

The ReVolta numbers are therefore a source claim from a different runtime and host. They are not compared numerically with the Strata timings below.

The Strata details are in [docs/BATCHING.md](../docs/BATCHING.md) ("One call per slot run") and [docs/QSAROWS.md](../docs/QSAROWS.md).

## Hardware and environment

| Item | Value |
|---|---|
| GPUs | 2 x Tesla V100-PCIE-16GB, 16384 MiB each |
| GPU interconnect probe | PCIe, 13.1 GB/s reported by the engine |
| CPU | AMD Ryzen 5 3600 6-Core Processor; 6 physical cores, 12 logical processors |
| Host RAM | 94.22 GiB total |
| CUDA / arch | CUDA 12.8, `sm_70` |
| Engine | 0.1.45.6 |
| Model | Gyro-L Qwen3.8-Flash-Next, TQ3_0 native pack |
| K/V | `--kv int8`; full-context candidate arms use `--kv-resident 32768` |

The only power measurements are the initial and final status snapshots across the measured arms: a combined 84.396 to 227.827 W for the two-GPU machine, at 48 to 69 C. The status endpoint reports one power value for the pair. **No enforced power limit or clock lock was recorded, so the power cap is unknown.** The 250 W figure in ReVolta #59 is that host's setup, not a measurement here.

## Configuration

Both engines read the same service-derived arguments, except for the arm-controlled options:

```text
--pack /mnt/strata-models/packs/gyro_l
--native /mnt/strata-models/models/Gyro-L/Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf
--expert-profile /mnt/strata-models/data/expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --mtp /mnt/strata-models/mtp/rt
--spec-min-p 0.70 --vision --vram-reserve-mib 750
--conversation-cache-mib 25600 --conversation-cache-slots 32
--pipeline-windows 2 --conversation-cache-park-gain 1024 --kv int8
```

Shared environment: `STRATA_EXPERT_PAIR=1`, `STRATA_SPLIT_RING=192`, `STRATA_KV_GROW=1`. The server adds `--serve`, `--layer-split 14` and `--batch N` from its config.

| Mode | Context | Batch arg | Batch-MTP | Batch groups | KV resident | Arm env |
|---|---:|---:|---|---:|---:|---|
| installed serial | 262144 or 32768 | 1 | off | engine default | none | shared |
| installed batch-MTP, one group | 262144 or 32768 | 4 | on | 1 | none | shared |
| installed batch-MTP, parallel 2 | 262144 | 2 | on | 1 | none | shared |
| installed pipelined | 262144 or 32768 | 4 | off | auto | none | shared |
| A/B full | 262144 | 2 | on | 1 | 32768 | A: none; B: `STRATA_QSA_SEGMENTS=1 STRATA_BATCH_VERIFY_WIDTH=1` |
| A/B short | 32768 | 4 | on | 1 | none | A: none; B: `STRATA_QSA_SEGMENTS=1 STRATA_BATCH_VERIFY_WIDTH=1` |
| plain A/S short | 32768 | 4 | on | 1 | none | A: none; S: `STRATA_QSA_SEGMENTS=1` |

The derived configs are sanitized: no API key is stored. The compact JSON repeats these values per arm under `configs`. The final plain A/B uses the same full and short configurations with `--plain-prompts`; A is off and B has both switches.

## Build and provenance

| Item | Value |
|---|---|
| Baseline source HEAD | `a9e06cb3feff8df423b823862fa7222259a6f4b9` (2026-10-31 17:20 -04:00) |
| Installed baseline binary | `engine/strata`, SHA-256 `1b4c32cd77361ad834f436cc34abccdc97ac2336f04660252b7cc2ef5aa20163`, 98,154,944 bytes, mtime 2026-10-31 17:23 -04:00, version 0.1.45.6 |
| Candidate base | `a9e06cb3feff8df423b823862fa7222259a6f4b9` |
| Candidate diff that produced the binary | 5 files, +443/-43: `src/core/verify.cpp`, `src/kernels/batch_rows_parity.cpp`, `src/program/generate.cpp`, `docs/BATCHING.md`, `docs/QSAROWS.md` |
| Candidate binary | `build/strata`, SHA-256 `d615147e71192210154d892aff2c41caa1537be7f6739b9cc861beb4f9622904`, 98,159,000 bytes, built 2026-11-02 14:30 -05:00, `sm_70` |

The baseline source HEAD was `a9e06cb3`, the same base as the candidate. The installed binary is the prebuilt `1b4c32` file; its mtime at 17:23 is after the 17:20 `a9e06cb3` merge, which is consistent with a build of that source, but no fresh baseline build was run. The candidate is that base plus the uncommitted five-file diff, built as `d615147e`. So the comparison is the installed 0.1.45.6 binary against a candidate build of the same base plus the two opt-in changes.

### Exercised build/test boundaries

- **CUDA:** the candidate `sm_70` build and link completed. The binary exists and the parity binaries ran on a V100.
- **HIP:** a ROCm 7.2.4/clang++ build was configured with HIP on. Only three host objects were compiled (`src/core/verify.cpp.o`, `src/program/generate.cpp.o` and `src/kernels/batch_rows_parity.cpp.o`). There was no link and no AMD GPU runtime. This is a partial compile, not HIP validation.
- **SYCL:** the separate `sycl/` implementation was untouched. `STRATA_ENABLE_SYCL=OFF`; no SYCL build or test was run.
- **Server tests:** `python -m pytest serve/test_parallel.py -q` passed all 39 tests in 61.46 seconds.

## Parity

All parity binaries ran on one V100 (`sm_70`) with the daily service stopped.

| Check | Result |
|---|---|
| `batch_rows_parity`, four environment combinations | `bitwise` in every case; `batch_rows_parity: ok` in all four |
| `qsa_parity --selftest` | `qsa: 0 failures`, PASS |
| `gdn_parity --selftest` | `gdn: 0 failures`, PASS |
| `verify_batch_parity` | `verify_batch_parity: PASS` |

The four batch-rows environments were the combinations of `STRATA_QSA_ROWS` 0/1, `STRATA_SCORES_MULTI` 0/1 and `STRATA_ATTN_LANECELL` 0/1 used by the runner. The exact kernel cases were `qsa_block_scores_rows`, `qsa_block_topk`, `qsa_decode_attn_batch_rows`, `native_qsa_indexer_append`/`native_qsa_indexer_append_steps`, and `gdn_step_norm_groups` window and commit. The segmentation cases covered runs of 1, 2, 4 and 8 rows, int8 and fp16 pools. Each case captured the grouped scores/top-k/attention calls once, instantiated once, and replayed the graph three times over two device step sets (A, B, A) with changed step contents; the result was bitwise equal to the per-row loop. `native_qsa_indexer_append_steps` was also bitwise equal, including a run that starts on the partial tail at 2046 and the block the run completes.

This is kernel and captured-graph parity. It is not a claim that `STRATA_BATCH_VERIFY_WIDTH` leaves output unchanged: wider verify groups can change arithmetic by design, and the HTTP runs below are timing experiments, not a quality comparison. No AMD or SYCL parity was exercised.

## Method

The runs used the OpenAI-compatible endpoint with streaming:

- Request body: `model=strata`, `stream=true`, `max_tokens=256`, `temperature=0`, `reasoning_effort=none`, `chat_template_kwargs.enable_thinking=false`, one user message containing the prompt. The compact JSON stores this body and the exact prompt texts.
- Two rounds per arm and client count. C1, C2 and C4 were measured.
- The nonce runs use eight code tasks. Each request is `[run baseline20261102 client <c> round <r>] <code task> Write at least 220 words.` This is a natural-output run: the model can stop at EOS before 256 tokens. Seven answers are the 39-byte nonce echoed back.
- The plain runs use eight fixed prompts with no prefix. Each asks for a complete implementation and at least 1000 words. The full texts are in the compact JSON.
- One or two warm-up bursts (up to four clients, 16 tokens each) per arm were excluded from the medians.
- A long-context probe used a 12,792-token prompt and 32 output tokens. Only six arms ran it: installed serial 262144, installed batch-MTP 262144, and the four full A/B arms. The 32768, pipelined and short arms ran `--long-tokens 0`.
- TTFT is the first content chunk. Latency is the full call. Table TTFT and latency are medians of the per-round means. Aggregate wall throughput is total tokens divided by the burst wall time. Aggregate decode throughput is `(tokens - clients) / (max_end - min_first)`. Per-client decode uses `(tokens - 1) / (end - first)`.
- The tables use per-arm medians over that arm's two rounds; the A/B summary medians use all four available rounds (both repeats). The A/B paired ratios are B/A for all four same-round pairs: A1r0/B1r0, A1r1/B1r1, A2r0/B2r0 and A2r1/B2r1. The plain A/S ratios are S1/A1 and S2/A2. Ranges and medians cover all four A/B pairs, or the two A/S ratios.
- The nonce A/B tables each contribute 4 arms x 3 client counts x 2 rounds = 24 rounds and 56 requests; the long probe and staged transition are extra. The plain A/S isolation contributes 4 arms x 3 client counts x 2 rounds = 24 rounds and 56 requests. The final plain A/B contributes 8 arms (full and short, A/B/A/B) x 3 client counts x 2 rounds = 48 rounds and 112 requests. The baseline has 7 arms from two local runs, with the client counts and rounds shown below.

No fixed-length `logit_bias` was used. Fixing output length is rejected with `--parallel > 1` by the server and engine, so natural EOS has to be recorded and treated as a limitation.

## Full-context memory: the decisive result

At 262,144 tokens, the expert cache is what concurrency costs.

| Arm | Context | Batch slots admitted/requested | CUDA0 cache slots | CUDA1 cache slots | CUDA1 cache GiB | Prompt chunk | Resident experts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Installed serial | 262144 | 1/1 | 7120 | **4758** | 8.82 | 8192 | 48% |
| Installed batch-MTP, one group | 262144 | 2/4 | 5754 | **658** | 1.22 | 1024 | 26% |
| Installed batch-MTP, parallel 2 | 262144 | 2/2 | 5754 | **658** | 1.22 | 1024 | 26% |
| Candidate full A/B, KV streamed | 262144 | 2/2 | 6877 | **3932** | 7.29 | 8192 | 44% |

The installed batch-MTP arm at full context is the 658-slot case. The same context with one serial slot keeps 4758 slots. The batch arm's CUDA1 is down to 1.22 GiB, and the engine warns that the prompt chunk has to fall from 8192 to 1024 tokens because CUDA1 must keep 128 slots and lend at most 90%; this is the `#448` warning in the log.

The candidate full A/B configuration adds KV streaming: `--kv-resident 32768` keeps 32,768 of 262,144 cells per QSA layer in VRAM and the K/V in 0.77 GiB of pinned RAM. That restores the CUDA1 expert cache to **3932 slots (7.29 GiB)**, CUDA0 to 6877 slots (12.30 GiB), keeps 44% of experts resident, and returns the prompt chunk to 8192, while retaining the 262,144-token context. It is 3274 slots above the unstreamed two-slot arm and 826 slots below the one-slot serial arm. The 1.53 GiB cache gap matches the two 0.72 GiB CUDA1 slot sessions and two 50 MiB drafters that the candidate full log reports (1.54 GiB before rounding).

This is a memory-layout result of the candidate build and its `--kv-resident` configuration, not an effect of the two A/B flags. A and B use the same layout and both show 6877/3932.

There is a compute consequence: under a streamed KV cache, `STRATA_QSA_SEGMENTS` groups the indexer append but leaves block scores, top-k and attention per-row (see the fallback documented in `docs/BATCHING.md`). The full-context B arm is therefore not a pure test of the segmented scorer. The short-context mode below, with no KV streaming, is the mode where the segmented path is fully active.

## Baseline: installed engine

All throughput values are medians in tokens/s, shown as wall/decode. "Slots" is admitted/requested. A dash means that client count was not run for that arm.

| Arm | Slots | Cache CUDA0/1 | C1 | C2 | C4 |
|---|---:|---:|---:|---:|---:|
| serial, 262144 | 1/1 | 7120/4758 | 48.2/61.6* | 55.2/59.5 | 51.0/51.2 |
| serial, 32768 | 1/1 | 7168/4925 | 52.7/64.5 | 49.8/50.3 | 50.6/50.9 |
| batch-MTP group 1, 262144 | 2/4 | 5754/658 | 22.8/26.3 | 27.3/27.5 | 25.0/25.0 |
| batch-MTP group 1, 32768 | 4/4 | 6943/3918 | — | 46.7/52.2 | 61.6/61.9 |
| batch-MTP group 1, parallel 2, 262144 | 2/2 | 5754/658 | 24.5/28.7 | 30.3/30.5 | — |
| pipelined, 262144 | 2/4 | 5754/712 | — | 12.8/13.3 | 13.3/13.3 |
| pipelined, 32768 | 4/4 | 6943/4037 | — | 18.4/19.1 | 25.9/25.9 |

`*` The serial 262144 C1 rounds produced 256 and 169 tokens, so its 48.2 wall value is not a clean 256-token rate; its 61.6 decode value also mixes the two output lengths. The clean single-stream baseline is the 32768 serial arm: 52.7 wall / 64.5 decode with 256 tokens in both rounds. The 262144 serial arm's long probe (12,792 prompt tokens, 32 output tokens) reached 10.82 s TTFT and 35.5 decode tokens/s; the 262144 batch-MTP group-1 arm's long probe reached 26.51 s TTFT and 17.2 decode tokens/s.

The installed data shows the same memory effect from a different direction: at 262144 the batch-MTP group drops single-stream decode from 61.6 to 26.3 tokens/s and does not recover at C4 (25.0). At 32768, where four slots fit and the cache stays near 6943/3918, the batch arm reaches 61.9 decode tokens/s at C4. Pipelining without batch-MTP is slower in both contexts.

The installed serial numbers are context, not a like-for-like candidate comparison: the candidate A/B single-client runs use batch-MTP with two or four configured slots, while the installed serial arm uses one slot.

## Nonce-prompt A/B, full context (262144, KV streamed, exploratory)

Configuration: context 262144, `--kv-resident 32768`, parallel 2, batch-MTP, one batch group. A = both switches off; B = `STRATA_QSA_SEGMENTS=1` and `STRATA_BATCH_VERIFY_WIDTH=1`. Both arms start with 2/2 slots and the 6877/3932 cache layout.

| Clients | A wall/decode | A TTFT/latency | B wall/decode | B TTFT/latency |
|---:|---:|---:|---:|---:|
| 1 | 42.44 / 52.69 | 1.17 / 5.36 s | 37.55 / 45.36 | 1.19 / 6.85 s |
| 2 | 49.52 / 50.01 | 1.02 / 9.79 s | 55.97 / 56.60 | 1.02 / 8.57 s |
| 4 | 47.06 / 47.22 | 6.10 / 15.21 s | 55.83 / 56.10 | 4.52 / 11.99 s |

The wall/decode columns are medians of four rounds. TTFT and latency are medians of the per-round means. The paired columns use all four same-round pairs: A1r0/B1r0, A1r1/B1r1, A2r0/B2r0 and A2r1/B2r1.

Paired B/A ratios and the matched token totals (four pairs; min-max):

| Clients | Wall | Decode | TTFT | Latency | Tokens A / B by pair |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.859–1.011 | **0.837–0.928** | 1.005–1.036 | 1.134–1.575 | 256/256, 256/256, 176/256, 169/256 |
| 2 | 0.681–1.376 | 0.681–1.376 | 0.894–1.103 | 0.530–1.551 | 312/512, 512/512, 512/512, 512/275 |
| 4 | 1.116–1.296 | 1.118–1.297 | 0.686–0.928 | 0.753–0.905 | 902/965, 1024/947, 996/955, 1024/1024 |

**C1: a measured decode regression, but not a uniform 16%.** B's decode is lower in all four same-round pairs: B/A 0.8367, 0.8574, 0.9282 and 0.8388. The range is 0.837–0.928 and the median is 0.848, so B is 7.2–16.3% lower, with a median of 15.2%. The four-round medians ratio is 0.861, but the same-round pairs are the estimate. Wall is 0.859–1.011 (median 0.922) and is not the regression signal: A2 produced 176 and 169 tokens while B produced 256 in both rounds. Decode is cleaner. The logs for the first repeat show B1's single-client windows carry four rows of slot 0 (`0,0,0,0`) while A1's carry two (`0,0`); that is consistent with the wider verify group adding per-step work when only one request is active, but no isolated experiment attributes the range to width. This is an inference from the window shapes, not a separate measurement. This is an exploratory nonce-prompt result, not the final C1 verdict.

**C2: noisy, no claim.** The four same-round full C2 pairs have ratios 0.681–1.376. Token totals are A 312/512/512/512 and B 512/512/512/275. A1 contains a 56-token answer; B2 contains a 19-token answer. No throughput claim.

**C4: suggestive, not proven.** All four same-round full-context pairs favor B: decode 1.118–1.297 (median 1.177), TTFT 0.686–0.928 (median 0.735), latency 0.753–0.905 (median 0.765). Token totals differ by up to 77 tokens (A 902/1024/996/1024, B 965/947/955/1024). Four pairs from two repeats are not four independent samples. Do not treat it as a demonstrated C4 gain.

## Nonce-prompt A/B, short context (32768, no KV stream, exploratory)

Configuration: context 32768, parallel 4, batch-MTP, one batch group. Four slots are admitted. A and B both start with 6943/3918 cache slots. Because KV is not streamed, the segmented path is eligible in B, and the logs show same-slot multi-row windows (for example `0,0,0,0`).

| Clients | A wall/decode | A TTFT/latency | B wall/decode | B TTFT/latency |
|---:|---:|---:|---:|---:|
| 1 | 42.21 / 52.27 | 1.02 / 5.81 s | 42.31 / 50.88 | 1.01 / 6.06 s |
| 2 | 45.86 / 46.43 | 1.01 / 7.09 s | 53.43 / 54.00 | 1.01 / 8.65 s |
| 4 | 55.41 / 55.75 | 1.66 / 10.47 s | 71.90 / 72.38 | 1.46 / 13.11 s |

The wall/decode columns are medians of four rounds; TTFT and latency are medians of the per-round means. The paired columns use all four same-round pairs: A1r0/B1r0, A1r1/B1r1, A2r0/B2r0 and A2r1/B2r1.

Paired B/A ratios and the matched token totals (four pairs; min-max):

| Clients | Wall | Decode | TTFT | Latency | Tokens A / B by pair |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.967–1.051 | 0.937–1.023 | 0.982–1.007 | 0.981–1.384 | 256/256, 176/256, 256/256, 237/256 |
| 2 | 0.942–1.781 | 0.941–1.791 | 0.974–1.022 | 0.905–1.894 | 251/512, 512/512, 512/512, 249/469 |
| 4 | 1.119–1.446 | 1.119–1.448 | **0.776–0.937** | 0.978–1.335 | 655/846, 935/994, 784/1024, 669/992 |

**C1: neutral within noise.** Wall is 0.967–1.051 and decode is 0.937–1.023 (median 0.987). Unlike the full-context C1, this B arm does not show a uniform decode loss; the range crosses 1.

**C2: strongly confounded, no claim.** The four same-round ratios span 0.941–1.791. Two A rounds are 19-token nonce echoes (A totals 251/512/512/249); B totals 512/512/512/469. The apparent gain is output-budget-confounded; no claim.

**C4: suggestive, not proven.** All four same-round pairs favor B on wall/decode: 1.119–1.446 (decode 1.119–1.448, median 1.292). TTFT is 0.776–0.937 (median 0.916). But B has more output in every round (A 655/935/784/669, B 846/994/1024/992), so the wall/decode ratio is confounded. Suggestive, not proven.

## Plain-prompt segments-only isolation (short context)

A separate run isolates `STRATA_QSA_SEGMENTS` from the width switch. It uses the same candidate binary at 32768, parallel 4, batch-MTP and one batch group, with A = off and S = `STRATA_QSA_SEGMENTS=1`; the width switch is off in both. All arms start with 4/4 slots and 6943/3918 cache slots. The sequence is A1, S1, A2, S2. This run uses the eight fixed plain code prompts with no run/client/round prefix. Each asks for a complete implementation and at least 1000 words. `fixed_length` is false, so this is a natural-output run.

| Clients | A1 wall/decode | S1 wall/decode | A2 wall/decode | S2 wall/decode | S/A wall | S/A decode |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.55 / 43.88 | 45.66 / 45.92 | 47.83 / 48.21 | 52.21 / 52.66 | 1.048–1.092 | 1.047–1.092 |
| 2 | 64.49 / 64.83 | 58.81 / 59.07 | 56.47 / 56.95 | 65.90 / 66.21 | 0.912–1.167 | 0.911–1.163 |
| 4 | 78.33 / 78.67 | 84.46 / 84.66 | 79.55 / 79.68 | 78.92 / 79.01 | 0.992–1.078 | 0.992–1.076 |

The S/A columns compare S1 with A1 and S2 with A2; each range is min–max of those two ratios. Values are medians of two rounds per arm and client.

The output budgets are much cleaner than the nonce A/B: C1 is 256 tokens in every round, C2 is 512, and C4 is 1024 except S1's first round at 799 (one 31-token answer plus three 256-token answers). Every request completed with `errors=0`. These prompts have no nonce prefix, so the 39-byte echo artifact is absent.

TTFT is 0.05–0.62 s, but every request reused all but one prompt token (`cached_tokens` 69/70, 74/75, 76/77 and 82/83). These TTFTs are prompt-cache hits, not cold prefill. The nonce runs have a different mix (their C1 requests start at `cached_tokens: 0` and about 1.0–1.2 s TTFT, while their C2/C4 values mix cached and cold requests), so the plain TTFT cannot be compared with them.

What the run shows:

- **C1:** S is 4.8% and 9.2% faster in the two repeats by wall time (decode 4.7% and 9.2%). This is not credited to segments: C1 is a single request, the A1/A2 values themselves span 43.5–47.8 and S1/S2 span 45.7–52.2, and the run cannot separate thermal or host variance from the switch. The single-request case is not the concurrency case the switch targets.
- **C2:** the signs are mixed. S1 is 8.8% slower than A1; S2 is 16.7% faster than A2. No consistent C2 gain.
- **C4:** the signs are mixed. S1 is 7.8% faster than A1 but its first round is 799 tokens; S2 is 0.8% slower than A2 with full 1024-token rounds. No consistent C4 gain.

The plain run is the cleanest comparison in this report because almost every output reaches 256 tokens. It still does not show a consistent concurrency gain from `STRATA_QSA_SEGMENTS` alone. The C1 direction is not evidence for segments, and the C2/C4 directions disagree between repeats. Do not promote the switch on this result.

## Final plain-prompt A/B (full and short)

The final run uses the same eight plain prompts as the A/S isolation, no nonce prefix, and the A/B/A/B sequence. A is off; B is both `STRATA_QSA_SEGMENTS=1` and `STRATA_BATCH_VERIFY_WIDTH=1`. The full mode is 262144 with `--kv-resident 32768`, 2/2 slots and 6877/3932 cache slots. The short mode is 32768, no KV streaming, 4/4 slots and 6943/3918 cache slots. All requests completed with `errors=0`.

C1 is 256 tokens in every round. C2 is 512 in every round. C4 is mostly 1024; the exceptions are full A1 round 0 at 858 (one 90-token answer), short B1 round 1 at 790 (one 22-token answer), and short B2 round 0 at 1003 (one 235-token answer).

| Mode | Clients | A1 wall/decode | B1 wall/decode | A2 wall/decode | B2 wall/decode | B/A decode median (range) | B/A wall median (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 1 | 41.22 / 41.50 | 44.36 / 44.71 | 41.03 / 41.29 | 44.12 / 44.41 | 1.074 (1.022–1.138) | 1.074 (1.022–1.135) |
| full | 2 | 56.06 / 56.35 | 66.18 / 66.53 | 55.04 / 55.24 | 59.62 / 60.15 | 1.135 (1.012–1.254) | 1.134 (1.004–1.253) |
| full | 4 | 50.06 / 50.18 | 53.89 / 55.31 | 49.74 / 49.83 | 57.47 / 57.61 | 1.157 (1.022–1.205) | 1.141 (1.027–1.171) |
| short | 1 | 43.07 / 43.32 | 47.04 / 47.33 | 46.16 / 46.48 | 47.90 / 48.22 | 1.037 (0.988–1.205) | 1.037 (0.988–1.205) |
| short | 2 | 61.24 / 61.61 | 60.94 / 61.59 | 65.87 / 66.30 | 63.73 / 64.41 | 0.988 (0.957–1.011) | 0.989 (0.949–1.000) |
| short | 4 | 80.61 / 80.79 | 82.11 / 82.26 | 79.47 / 79.65 | 84.47 / 85.14 | 1.043 (0.924–1.188) | 1.041 (0.924–1.181) |

The B/A columns use all four same-round pairs: A1r0/B1r0, A1r1/B1r1, A2r0/B2r0 and A2r1/B2r1. Values are medians and min–max ranges over those four pairs.

### Runtime levels (all four rounds)

The same binary also ran two serial arms: `--parallel 1 --no-batch-mtp --context 262144`, no KV streaming, 7120/4758 cache slots, no extra `STRATA_*` arm flags, with the same plain prompts. These are runtime levels, not a paired A/B comparison. The serial arms ran after the A/B arms, not interleaved, so cross-config thermal noise is explicit.

| Path | C1 wall | C2 wall | C4 wall | C4 round tokens |
|---|---:|---:|---:|---|
| serial, parallel 1, no KV stream | 48.92 | 50.97 | 47.71 | 784, 784, 784, 784 |
| stream A, parallel 2, KV stream, flags off | 41.12 | 55.38 | 49.74 | 858, 1024, 1024, 1024 |
| stream B, parallel 2, KV stream, both on | 44.36 | 62.79 | 56.04 | 1024, 1024, 1024, 1024 |

C1 is 256 tokens in all four rounds for all three paths. C2 is 512 in all four rounds. Those C1/C2 budgets are consistent. C4 is not: serial is 784 in every round (one 16-token answer per round), stream A has one 858-token round, and stream B is 1024 in every round. Do not compare the C4 wall values as fixed-output rates.

The levels give these ratios:

- C2 stream B is **+13.4%** over stream A and **+23.2%** over serial, by wall time.
- C1 stream B is **-9.3%** vs serial and **+7.9%** over stream A, by wall time. The batch window also changes from two to four rows with the width switch, so this comparison does not isolate the effect of segments.
- The serial path gives the best lone-user C1 and the best per-client C2, at 50.7/53.6 per-client decode p50 vs 34.9/32.0 for stream B. Stream B gives the highest C2 aggregate.

The serial path is a runtime reference, not a default recommendation. It was recorded after the A/B arms and has only four rounds per level.

What the final run shows:

Full mode streams KV. The segmented scores/top-k/attention path is not active there; only the indexer append is grouped. The full-mode A/B differences are therefore not segments-only effects.

- **Full C2 is the strongest positive signal.** B is higher in all four pairs. The medians are 1.134 wall and 1.135 decode, with a decode range of 1.012–1.254. The A2r0/B2r0 pair is near-neutral (1.004 wall, 1.012 decode). The prompts and output budgets are the same: 512 tokens in every round. This is the most credible concurrent result in the report, but it is still four pairs from two repeats.
- **Full C4 also favors B in all four pairs.** Decode median is 1.157 (range 1.022–1.205); wall median is 1.141 (1.027–1.171). One A round is 858 tokens while B is 1024, so one pair has an output-budget difference. The rest are 1024. The result is suggestive but not proven.
- **Full C1 favors B by 7.4% median.** All four decode ratios are above 1 (1.022–1.138). C1 is one request, not the concurrency case. The gain is not attributed to segments or width, and the earlier nonce-run C1 regression is not the final verdict.
- **Short C1 is +3.7% median but mixed.** The four decode ratios are 0.988–1.205; two are below 1. The gain is not attributed to the switches.
- **Short C2 is neutral to slower.** The four decode ratios are 0.957–1.011; the median is 0.988. No gain.
- **Short C4 is small and EOS-confounded.** The four decode ratios are 0.924–1.188; the median is 1.043. B1r1 has a 22-token answer, and B2r0 has a 235-token answer. No speedup claim.

The full arms also ran a 39,812-token prompt, beyond the 32,768 resident cells. All four completed.

| Arm | TTFT | 32-token decode | Output |
|---|---:|---:|---:|
| full A1 | 24.026 s | 20.965 tok/s | 32 |
| full B1 | 24.355 s | 26.061 tok/s | 32 |
| full A2 | 24.381 s | 20.981 tok/s | 32 |
| full B2 | 24.581 s | 19.129 tok/s | 32 |

The KV stream crossed the 32,768 resident cells and worked. The TTFTs are flat at 24.03–24.58 s. The 32-token decode is too short for a speedup claim.

The final plain run does not establish a general concurrency gain. Full C2 is the best candidate signal; full C4 is secondary. Short C2 is neutral or slower. Short C4 is small and EOS-confounded. C1 differences are not attributed to the switches. Do not promote either default.

## API and structural smokes

### Semantic HTTP smoke (not a quality benchmark)

The final service smoke used the actual `/v1/chat/completions` endpoint at 262144 context, streamed, `--parallel 2`, one batch group, `max_tokens=96` for the first turn and `max_tokens=16` for the follow-up. Temperature was 0 and thinking was off. Each arm launched two conversations at the same time.

| Arm | Country | Capital | Follow-up | Initial chars | Capital check | Follow-up check |
|---|---|---|---|---:|---|---|
| off | France | Paris | France | 510 | pass | pass |
| off | Japan | Tokyo | Japan | 487 | pass | pass |
| both | France | Paris | France | 231 | pass | pass |
| both | Japan | Tokyo | Japan | 322 | pass | pass |

All eight checks passed: two countries, two checks per conversation, two arms. The initial descriptions differ in length and wording between off and both. This is a dialogue smoke, not a general quality benchmark. The service was restored active and loaded at 262144 after the run.

### Edge-final structural smoke

The corrected edge-final smoke ran `off`, `segments`, `both` and `off` repeat with zero harness failures. It covered count budgets 1, 2, 3 and 4 tokens and a 64-token case, plus a 1/1/64/64 transition. The cancel and reuse checks passed on all four arms: `cancelled=true` and `reuse_ok=true`; the cancel path produced 4, 4, 5 and 4 tokens respectively.

Segments-only was byte-identical to off at all five lengths (1, 2, 3 and 4 tokens and the 64-token case). Off and segments-only first diverged from solo at token 14; both first diverged from solo at token 25. This is a structural token-stream check. It does not claim width identity, quality, or general correctness.

## What the outputs show

No compared nonce A/B request produced byte-identical text. On the sampled pairs, the first differing character ranged from character 0 to character 438. That is expected when the verify window and arithmetic change, but it is also why these runs cannot be used as a quality comparison: no semantic scoring was done.

Seven answer files are exactly 39 bytes: the bracketed run/client/round nonce, echoed by the model, followed by EOS. These are the 19-token outliers in C2 and the C4 cases. They are not model answers. The 39-byte files come from both installed and candidate arms, so they are a prompt/harness artifact, not a candidate regression. Their presence is the main reason the nonce C2 ratios are unusable and part of why some nonce C4 rounds are short.

All measured nonce A/B, final plain A/B, serial plain and plain A/S requests completed with `errors=0`; the timing tables above exclude errored requests (there were none). The absence of transport errors is not a correctness result.

Treat every throughput number here as exploratory. The final plain A/B has clean C1/C2 output budgets, but it is still two repeats and four same-round pairs. The initial nonce-prompt C1 regression is not the final verdict.

## What is supported

- The full-context memory result is clear: two unstreamed batch-MTP slots leave **658** CUDA1 expert-cache slots versus **4758** for one serial slot. KV streaming restores **3932** slots at the same two-slot layout while retaining **262144** context, 826 below serial.
- Final plain full C2 is the strongest positive signal. B is higher in all four same-round pairs: decode median 1.135 (range 1.012–1.254), wall median 1.134 (1.004–1.253). The prompts and 512-token budgets are the same; one pair is near-neutral. Full mode streams KV, so this is not a segments-only effect. Not proven.
- Final plain full C4 also favors B in all four pairs: decode median 1.157 (1.022–1.205), wall median 1.141 (1.027–1.171). One A round is 858 tokens versus 1024. It is a full-mode result, not segments-only. Suggestive, not proven.
- Final plain full C1 is +7.4% median (decode 1.022–1.138) in B, but C1 is a single request and the gain is not attributed to segments or width. The final short C1 is +3.7% median with mixed signs (0.988–1.205).
- Final plain short C2 is neutral to slower: decode 0.957–1.011, median 0.988. Final short C4 is small and mixed: decode 0.924–1.188, median 1.043; two pairs favor B and two favor A. Its short rounds are EOS-confounded.
- Runtime levels on the same binary: serial C1 48.92, stream A 41.12, stream B 44.36; C2 serial 50.97, stream A 55.38, stream B 62.79. C2 stream B is +13.4% over A and +23.2% over serial by wall; C1 stream B is -9.3% vs serial by wall. Serial is a non-interleaved reference.
- The nonce-prompt A/B is exploratory only. Its C1 decode is lower in all four pairs (0.837–0.928, median 0.848), but that is a configuration-specific result, not the final C1 verdict.
- The plain-prompt A/S isolation has full output budgets in almost every round. C1 favors S, but it is not credited to segments; C2 and C4 have mixed signs. No consistent segments-only concurrency gain.
- Kernel and captured-graph parity for the listed cases is bitwise on sm_70: `qsa_block_scores_rows`, `qsa_block_topk`, `qsa_decode_attn_batch_rows`, `native_qsa_indexer_append_steps` and `gdn_step_norm_groups`, with int8 and fp16 pools and runs of 1/2/4/8 rows. This does not claim that the width switch leaves output unchanged.
- No runtime default changed. The switches are opt-in.

## What is not supported

- A general "ReVolta segments improve concurrent throughput" claim. Full C2 is the best signal, but it is four pairs from two repeats and one pair is near-neutral; the short modes do not confirm it.
- A proven full C2, full C4 or short C4 speedup. Full C2 and full C4 medians are positive; short C4 is mixed. The samples are small and C4 has EOS-confounded rounds.
- A short C2 gain. The final plain short C2 median is 0.988 with a range of 0.957–1.011.
- A segments-only concurrency gain. The plain A/S run has full output budgets but its C2 and C4 signs disagree between repeats; its C1 difference is not credited to segments.
- A claim that the two switches together are fully validated for serving. Parity covers the kernels and captured graphs; the HTTP runs are timing experiments and produce different text.
- A claim about the segmented scorer from the full-context C1 result: that mode streams KV, so the segmented scores/top-k/attention path is not active there.
- A single-switch attribution from the nonce A/B rows: they compare both switches together. The plain A/S run isolates `STRATA_QSA_SEGMENTS` on the short-context path, and its C2/C4 signs are mixed; its C1 difference is not credited to segments.
- A long-context speedup claim. The 39,812-token smoke completed, but its 32 output tokens are too short for a decode comparison.
- A default recommendation from the serial/stream comparison. The serial arms ran after the A/B arms, not interleaved, and each level has four rounds.
- A fixed-output C4 comparison across paths. Serial C4 is 784 tokens, stream A has one 858-token round, and stream B is 1024.
- A general quality result from the semantic smoke. It checked two countries and two follow-ups; the initial descriptions differ in length and wording.
- Width identity from the edge-final smoke. Segments-only matched off at 1/2/3/4/64 tokens; both first diverged from solo at token 25.
- AMD/HIP performance or correctness. HIP only compiled three host objects and never linked or ran.
- SYCL behavior. The separate implementation was unchanged and not built.
- A power-normalized comparison with ReVolta #59. This host's power cap is unknown, and the model, engine and KV format differ.

Do not change the defaults of `STRATA_QSA_SEGMENTS` or `STRATA_BATCH_VERIFY_WIDTH` based on these runs.

## Reproduction

The rows came from local scripts under `/tmp/strata-concurrency-bench`. They are temporary harnesses, not committed code. They stopped the daily user service, waited for both GPUs to drop below 700 MiB, ran isolated arms, and restarted the service in an `EXIT` trap. The service was restored after each completed job. No API key is written to the derived configs or this report.

The exact scripts were archived locally as `/tmp/strata-revolta-benchmark-harness-2026-11-02.tar.gz` before the temporary script copies were removed. Raw measurement JSON and logs remain in the measurement directory. Extract the archive there before using the local commands below; the archive is not part of this PR.

The committed `tools/serve_load.py` is the closest tracked load generator. It uses different prompts and a different metric surface. It did not produce these rows and must not be cited as their source.

Set generic roots and the local unit name:

```bash
CAND=/path/to/candidate-build
TMP=/tmp/strata-concurrency-bench
PY=/path/to/venv/bin/python
SERVICE=strata.service
```

The baseline table is two local runs. `run_baseline.sh` ran four arms: serial 262144, batch-MTP 262144, pipelined 262144 and batch-MTP 32768. `run_extra.sh` ran the other three: serial 32768, pipelined 32768 and batch-MTP parallel-2 262144.

```bash
/bin/bash "$TMP/run_baseline.sh"
/bin/bash "$TMP/run_extra.sh"
```

The first log was `run-20261102-140213.log`; the extra log was `run-extra-20261102-141703.log`.

Candidate A/B, the exact log header was tag `revolta-qsa-v1`, arms `A B A B`, modes `full short`, with the parity preflight:

```bash
RUN_TAG=revolta-qsa-v1 \
ARMS="A B A B" \
MODES="full short" \
PREFLIGHT="$TMP/run_parity.sh $CAND/build" \
"$TMP/run_ab.sh" "$CAND/build/strata"
```

The plain-prompt segments-only isolation used the same wrapper with sequence A1 S1 A2 S2:

```bash
RUN_TAG=revolta-plain-segments \
ARMS="A S A S" \
MODES=short \
"$TMP/run_ab.sh" "$CAND/build/strata" --plain-prompts
```

The wrapper's `S` arm sets only `STRATA_QSA_SEGMENTS=1`; the width switch is off. The log header confirms tag `revolta-plain-segments`, arms `A S A S`, mode `short`, and `--plain-prompts` in each arm command.

The final plain A/B run used both modes, A/B/A/B, and `LONG_FULL=25600`:

```bash
RUN_TAG=revolta-plain-final \
ARMS="A B A B" \
MODES="full short" \
LONG_FULL=25600 \
"$TMP/run_ab.sh" "$CAND/build/strata" --plain-prompts
```

The full arms then ran a 39,812-token prompt; the short arms ran `--long-tokens 0`. The final run also used the corrected edge-smoke preflight before the arms; that preflight is omitted here because it does not generate the timing rows.

The serial reference ran after the A/B arms with the same binary and plain prompts:

```bash
RUN_TAG=revolta-plain-serial \
ARMS="A A" \
MODES=serial \
"$TMP/run_ab.sh" "$CAND/build/strata" --plain-prompts
```

One arm, exactly as the wrapper constructs it (full B1):

```bash
"$PY" "$TMP/bench.py" --label revolta-qsa-v1-full-B1 \
  --exe "$CAND/build/strata" --context 262144 --kv-resident 32768 \
  --parallel 2 --batch-mtp --batch-groups 1 \
  --clients 1,2,4 --rounds 2 --transition --long-tokens 8192 \
  --env STRATA_QSA_SEGMENTS=1 --env STRATA_BATCH_VERIFY_WIDTH=1
```

Short B1 differs only in `--context 32768` (no `--kv-resident`) and `--parallel 4 --long-tokens 0`. A arms omit both `--env` flags. The parity preflight was:

```bash
"$TMP/run_parity.sh" "$CAND/build"
```

The daily service is a user unit; the wrapper did, in effect:

```bash
systemctl --user stop "$SERVICE"
# run the arms with both GPUs free
systemctl --user start "$SERVICE"
```

The full shared engine arguments are in the compact JSON under `configs.shared_args`. The model and pack paths are the `/mnt/strata-models/...` paths recorded there. The per-arm differences are the context, `--kv-resident`, parallel/batch count, batch group and environment flags.

## Limitations

1. n=2 rounds per arm/client. A/B ranges are min/max of four same-round pairs; A/S ranges are min/max of two ratios. None is a confidence interval.
2. Natural outputs stop at EOS; wall throughput and latency are not comparable across unequal output lengths.
3. Seven 39-byte nonce echoes produce 19-token outliers; nonce C2 and part of nonce C4 are unusable for throughput.
4. Full B streams KV, so the segmented scores/top-k/attention path is not active; only the indexer append is grouped.
5. The baseline is the installed `1b4c32` binary; the captured source baseline is `a9e06cb3`. No fresh baseline build was run for this report.
6. The power cap and clocks were not measured; power values are initial/final aggregate snapshots.
7. HIP is a partial host compile only; no AMD link, runtime or parity. SYCL is separate and unchanged.
8. Parity is kernel/graph level. It does not prove serving equivalence for the combined switches.
9. No semantic quality scoring; nonce output hashes differ and no arm-to-arm text equality is claimed. The final plain A/B has no nonce.
10. ReVolta #59 is a different engine and host; its reported ratios are not numerically compared here.
11. Staged transition rows exist in the raw arm JSON but are budget-confounded and are not used for a throughput claim.
12. The plain A/S run is n=2 per arm; only one C4 round is short (S1 round 0, 799 tokens).
13. Plain-prompt TTFT is a prompt-cache hit time (all but one prompt token cached); it is not a cold-prefill measurement and is not comparable with the nonce-run TTFT.
14. Final plain C4 has three short rounds: full A1r0 858, short B1r1 790 and short B2r0 1003. Its C4 ratios are EOS-confounded.
15. The two serial plain arms ran after the A/B arms, not interleaved. The cross-config comparison has explicit thermal noise and four rounds per level.
16. C4 is not fixed across paths: serial is 784 in every round, stream A has one 858-token round, and stream B is 1024.

The raw JSON and engine logs are in `/tmp/strata-concurrency-bench` on the benchmark host. The committed [compact JSON](v100-revolta-concurrency-2026-11-02.json) is the sanitized measurement copy.
