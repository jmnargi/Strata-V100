# NVIDIA Tesla V100 / Titan V (Volta, sm_70): the community build

Strata's ready-made engine and the support matrix start at compute capability 7.5 (RTX 20). A V100 (7.0) runs the same
engine compiled for it, through the experimental build that issue #236 added. It is **community-tested, not part of the
supported cards**: no ready-made engine is published for it and the numbers below come from one machine.

## Build

CUDA 13 dropped Volta, so the build needs a **CUDA 12.x** toolkit (12.8 was used) and a host GCC that toolkit accepts.

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata -j
```

Setup does the same by itself for a model on a V100 (the experimental CUDA 12 engine, in `engine-cuda12/`: ready-made
on Windows, compiled with a CUDA 12.x toolkit on Linux; [OLDER_GPUS.md](OLDER_GPUS.md)). For a PC with a V100 and a newer card, a build for both architectures (`-DCMAKE_CUDA_ARCHITECTURES="70;86"`) is the
natural choice (a community report in issue #509 ran an RTX 3080 beside a V100 that way; it was not repeated here). On the
measuring PC a Quadro RTX 4000 (sm_75) sits beside the V100s and was hidden with `CUDA_VISIBLE_DEVICES` because the sm_70-only
build has no code for it.

## What runs on Volta

The flag only lowers the floor (CMake and the runtime device check) and replaces two intrinsics Volta lacks. It does not
select "old" kernels everywhere: which kernel runs is decided per architecture when the engine is built.

| Part of the prompt path | On sm_70 |
| --- | --- |
| MoE experts | opt-in (`STRATA_PF_WMMA=1`, only when every visible card is compute capability 7+; MMQ otherwise): `gemm_iq_f16_grouped`: a group of experts per launch on the FP16 tensor cores, the weights dequantized in shared memory (+10% prompt speed against MMQ's `dp4a` kernels, [bench](../bench/results/2026-10-07-v100-prompt-experts/README.md); FP16 activations instead of MMQ's q8_1, so the answers differ) |
| Dense projections (dequantized weights) | FP16 tensor-core GEMMs (cuBLAS / CUTLASS `s884`) |
| BF16 projections (hyper-connection, router, indexer, ...) | converted to FP16 and run on the FP16 tensor cores (#655, #540; `STRATA_BF16_TC=0`: cuBLAS BF16, an FP32 SIMT kernel on Volta) |
| QSA attention for decode and verify windows (and prompts with `STRATA_PROMPT_ATTN_OLD=1`) | #540's kernel (fewer shuffles, bit-exact; `STRATA_ATTN_PRE75=0`: the one other cards run) |
| QSA prompt attention, int8 / FP16 / K8V4 KV | `prompt_attn_v70_kernel` on `mma.m8n8k4` (`STRATA_PROMPT_ATTN_OLD=1`: the decode kernel, one query at a time) |
| QSA prompt attention, Q4_0 KV | the decode kernel |
| QSA block scores | the warp kernel (the tensor-core scorer needs sm_80) |

| Part of a decode / verify window | On sm_70 |
| --- | --- |
| Routed experts in VRAM | gfx906's expert mode 8: the codebook grid and the group's activations in shared memory, SwiGLU fused (`STRATA_EXP_MODE=0`: the CUDA layout other cards run) |
| Dense 2-4 column GEMVs and the head | the interleaved `native_mmvq_il` with a Volta rows table (`STRATA_MMVQ_IL=0`: `native_mmvq`) |
| Hyper-connection read, up projection | gfx906's latency-hidden up kernel (`STRATA_GR_FAST=0`: the plain one) |

All three are bitwise the kernels they replace; on a V100-SXM2 they take a verify window's GPU work from 22.8 to 21.2 ms
([bench/results/2026-10-07-v100-decode-kernels](../bench/results/2026-10-07-v100-decode-kernels/README.md)).

## Setup default for UD-Q4_K_XL

On sm_70, the FP16 prompt path returns non-finite logits (#879) from layer 0 for `UD-Q4_K_XL`. The first non-finite
values are in the down projection, which uses Q5_1 in this file. The K-quant MMQ kernels give finite output for this
file. Therefore, setup writes `STRATA_MMQ_KQ=1` for this file on the CUDA 12 engine only. CUDA 13 and HIP keep the flag
opt-in. `UD-IQ4_XS` keeps the FP16 path because its down projection is IQ4_NL; its default does not change. Run setup
again to add the entry to an existing config.

## Measured

One V100-PCIE-32GB, i9-7960X, Unsloth `UD-IQ4_XS` (not a setup model), int8 KV, MTP `--spec 2`; prompt read speed of random-word prompts, tokens/s,
with and without the Volta attention kernel (`STRATA_PROMPT_ATTN_OLD=1`):

| Prompt tokens | old attention path | `prompt_attn_v70_kernel` |
| --- | ---: | ---: |
| 7,194 | 1,010 | 1,164 |
| 28,650 | 1,063 | 1,251 |
| 114,338 | 973 | 1,123 |

Decode was 38-51 tok/s in every run and is not affected. The profile, the parity numbers, the needle test and the limits are in
[bench/results/2026-10-03-v100-prompt-attn](../bench/results/2026-10-03-v100-prompt-attn/README.md).

What the kernel does and does not guarantee: against an FP64 reference its error is about 2-3e-6 (output scale 3.6; the FP32 kernel it replaces: 2e-6; 5e-6 before the hi and lo halves got separate accumulator chains); 15 of 15
needles were found; greedy output was identical on the 28,650- and 114,338-token prompts and differed late in the answer on the 7,194-token one (another
summation order).

## Checking and benchmarking the Volta kernels

```sh
cmake -S . -B build -DSTRATA_PARITY_PROMPT_ATTN=ON ... && cmake --build build --target qsa_prompt_attn_parity
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5     # synthetic, no model; exit 0 = PASS
```

It compares the tensor-core attention with the FP32 kernel and an FP64 host reference and times both. The Q4_0 cases are skipped below
sm_80.

## Limits

- A layer split over two V100 was not measured with this kernel. The cards work in turn on a single request, so a split adds expert-cache room, not prompt speed.
- sm_60 (Pascal) is covered by the same flag, and community runs exist: two Tesla P40 with a layer split (#1028, bench
  [2026-10-05-community-2x-p40](../bench/results/2026-10-05-community-2x-p40/README.md)), a Tesla P100 alone and with an RTX 2070 SUPER as
  the expert-helper card (#1069: 30.5 tok/s alone, 35.8-38.4 with the helper), and a V100 + P100 pair (#1079: +6-11% decode from 0.1.39 to
  0.1.40 with the P100 as helper cache). One reporter's numbers each; this page's own kernel measurements are V100 only.
- Volta's tensor cores take FP16 only: a model's BF16 weights are converted, and a weight outside FP16's range would saturate (none did in
  the check above).

## Selected v0.1.41 kernels

Set `STRATA_SM70_TABLE=1` to select the interleaved dense-matvec table,
native expert mode 8, and latency-hidden HC up projection on Volta.
The interleaved 2-4 column matvec with a Volta rows table and the
latency-hidden norm/up are bitwise the default kernels' output (GPU work
per verify window 22.8 -> 21.2 ms on a V100-SXM2 in the contributor's
measurement). The Volta defaults table now selects this master setting.
The individual settings `STRATA_MMVQ_IL`, `STRATA_EXP_MODE`, and
`STRATA_GR_FAST` take precedence. Set `STRATA_SM70_TABLE=0` to select
the previous decode path explicitly.

The Volta defaults also select `STRATA_DEQUANT_WIDE=1`. This setting uses
the existing vector-store dequantization kernels. The CUDA parity test
checks that their output is bitwise equal to the narrow kernels.
Set `STRATA_DEQUANT_WIDE=0` to select the narrow control.

Set `STRATA_EMB_REUSE_ACCOUNT=1` to let the prompt planner use the embedding
buffer space reused by half outputs. Default accounting remains conservative.
On the measured 8K chunk, this option reduces each stage's prompt loan by
about 80 MiB. It does not establish a general prompt-speed gain.

For the measured dual-V100 IQ3_S arena, `STRATA_PREFILL_CPU_SHARE=auto`
and `STRATA_PREFILL_STREAM_MIN=3072` select CPU-assisted staging below
3,072 tokens. The larger-chunk option can change prompt rounding.
Do not enable it for batch slots or assume the same gain on other models.

**Retained CPU and Volta changes.** These changes keep the model formats, the
int8 KV cache at a 524,288-token context, vision, MTP, the L2/L3 conversation
tiers and the NVMe-backed n-gram table. No 20% end-to-end gain is proved yet,
and no performance guarantee follows.

On sm_70 the prompt path's QSA top-k may use the active-block bound of a
prefill. The bound selects the register kernel when the prompt's active blocks
fit it, even when a long context capacity makes the score-row stride large.
Decode and captured graphs pass no bound and keep the capacity rule.

On a CPU with AVX2, no AVX-512 and no AVX-VNNI, the AVX-2 expert kernel serves
IQ3_XXS, IQ3_S and IQ2_S rows for one token (NT1) by default. NT means the
tokens in one group; NT1 is a group of one token. The IQ3_S and IQ2_S decodes
use 32-bit loads to read packed index bytes, which shortens the grid assembly.
These rows reduce in ggml's own float order and apply the format normalization
factor after the reduction, so a row is ggml's dot product bit for bit,
including subnormal products. IQ4_NL down rows are unchanged. An explicit
`STRATA_IQ_MT_MIN` keeps its rule.

After each non-host physical core has a worker, the pool pins additional
workers to the allowed SMT siblings of the worker cores. SMT (simultaneous
multithreading) means a physical core's second logical processor. The host
core's sibling comes last. The default placement is unchanged: five workers and
the host thread on the measured six-core CPU.

See the [release selection](../improvements/RELEASE-v0.1.41-V100.md)
and [prefill/decode measurements](../benchmarks/v100-iq3_s-v0141-2026-10-08.md).
The options do not change an installed service until its environment and
engine are changed. Do not enable stage-buffer pinning for IQ3_S.

## Shared-root return latency

The dual-V100 protected study measures repeat shared-root HTTP latency
at 1,799.7 ms before the cache changes and 902.6 ms after them.
The engine retains a disk-restored root in the bounded RAM cache.
The unrelated small-chat and live-continuation cases do not improve
in this sample. Do not treat this result as a general throughput gain.
See the [agentic latency measurements](../benchmarks/v100-iq3_s-prefill-wmma.md#agentic-short-turn-latency-2026-10-10).

## Selected v0.1.43 performance changes

The [release review](../improvements/RELEASE-v0.1.43-V100.md) lists the selected
ports and the changes that do not apply to Volta.

- `STRATA_ROUTE_RESIDENT=1` selects resident-biased routing. Its route swap now
  uses one warp per token. The option remains off by default.
- Linux CUDA can overlap adaptive expert copies with the next decode window.
  Layer splits remain excluded. On one Volta card the option is now on by
  default (`STRATA_SM70_PERF=0` or `STRATA_ADAPT_OVERLAP=0` turns it off).
  Resident RAM sources require `STRATA_ADAPT_OVERLAP_RESIDENT=1`.
- In the normal serial serve path, an implicit adaptive-copy count can grow
  from 96 to 384 swaps. Counts above 96 must fit half of the recent verify
  work time at the measured host-to-device rate. The legacy minimum is 96,
  also when a valid measurement is not available. An explicit `--adapt-swaps N`
  fixes the count; `0` disables the copies. The time-based count can change
  cache residency between runs. It does not guarantee identical token IDs.
- The automatic route tail skip stays off at 90% expert-cache residency or
  more. `STRATA_TAIL_SKIP_MAX_CACHED` changes this limit. An explicit
  `STRATA_ROUTE_TAIL_SKIP` takes precedence.
- Linux CUDA uses the measured PCIe share for the canonical Q2_0 pack.
  Native packs already used a measured share.

These controls do not establish a speed gain for every V100 configuration.
See the [verification report](../benchmarks/v100-upstream-v0.1.43-2026-10-11.md).

## The measured Volta profile (2026-10-20)

This section is the fork's own measurement on a dual V100 PCIe 16 GB rig (GPU 0 on a PCIe 3.0
**x4** link, GPU 1 on **x16**, a Ryzen 5 3600 with six cores, 94 GiB of RAM), the IQ3_S pack,
the 524,288-token context, the int8 KV cache, the MTP draft, the layer split `16` and
`--pipeline-windows 2`. The raw arms and the method are in
[the report](../bench/results/2026-10-20-v100-sm70-speed-stack/README.md).

### Layer-split prompt path: keep the 8,192-token chunk

`STRATA_PREFILL_TIMING=1` shows where a prompt chunk goes. On that rig one 8,192-token chunk
costs about 2.1 s of PCIe pass over the experts and about 1.96 s of compute, and the chunk
takes 3.9 s: the rig's b/a is about 8,800 tokens.

- A **bigger** chunk borrows more of the expert cache for its buffers, so the cache holds fewer
  experts and the PCIe pass grows: `auto:16384` fell back to 10,240 tokens and read the
  32,768-token prompt 7.5 % slower (1,607 against 1,736 tok/s).
- A **smaller** chunk pays the fixed pass again for every chunk: 4,096 tokens read it 23 %
  slower, 2,048 tokens 54 % slower.
- A **deeper streamed ring** prefetches further over PCIe but is VRAM the expert cache does not
  keep: 384 slots and 512 slots both read the same prompt no faster than 192.

These measurements apply to the layer-split configuration. Keep its 8,192-token
ceiling. The automatic single-card rule below does not apply to a layer split.

One card is different. A native pack with an arena loaded whole in RAM can use
16,384-token chunks on one Volta card. Mapped arenas, shared file-backed arenas
and resident complements keep the previous ceiling.
The startup VRAM plan prices those buffers with the same loan cap, the 128-slot floor and the
low-RAM check, and the rule keeps 8,192 when the available RAM does not cover the larger loan
plus 6 GiB. An explicit `--prefill N`, `auto:N` or `STRATA_PREFILL_AUTO_MAX` is not changed.
`STRATA_SM70_CHUNK=0` keeps 8,192 and `STRATA_SM70_CHUNK=1` takes 16,384 anyway. A larger chunk
changes the prompt path's rounding, so a prompt's residual bits can differ from an 8,192 chunk
(the same caveat the general 16,384 option has on CUDA).

### The decode window waits on three things, and the profile names them

`STRATA_VERIFY_PROFILE=1` prints the GPU's own stamps per window. On the rig above one
verify window (2.4 to 2.5 tokens) takes about 40 ms, and the second stage's time is:

| Stamp | What it waits for | Stage 1 (ms per window) |
| --- | --- | ---: |
| the dense projections, the head, the VRAM experts | the real work | about 20 |
| `waitB` | the PCIe share of the missed experts, copied into the staging buffer inside the layer's chain | 6.5 - 8.0 |
| `waitCPU` | the CPU pool's rows for the layer | 4.2 - 5.2 |
| `waitA` | the host's plan for the layer (the doorbell handshake) | 1.4 - 1.6 |

A decode rate of about 70 tok/s against a bandwidth floor of roughly 10 ms per window is not a
bandwidth problem: it is these three waits, and the PCIe traffic behind `waitB` (about 82 MiB
per window for the missed experts, plus about 45 MiB for the adaptive tier's swaps).

The candidates measured on this rig did not move it: upstream's six- and thirteen-switch Volta
stacks, `--pcie-mode dma` (worse: the DMA's landing signal is a host callback, and the callback's
latency on a six-core host is larger than the copy it waits for), `--pcie-frac 0`,
`--pool-workers 10`, `--adapt-swaps 24`, `--adapt-every 8` and `--adapt-every 0` were all
inside the noise of two baseline arms.

### The DMA path with the device-raised flag

`--pcie-mode dma` copies the PCIe share of the missed experts on a second stream, beside the layer's
compute, so the wait the copy kernel causes inside the layer's chain disappears. Its landing signal is a
one-thread kernel on the copy stream: the kernel runs after the copy (the stream orders them) and writes the
mapped flag with a system fence, so the graph that spins on the flag sees the value without the host calling
into the driver inside a window. The engine already uses the same pattern for the split hand-off and for the
doorbell.

On the rig above: **15.39 / 15.79 ms per token with the copy kernel, 14.58 / 14.53 with the DMA path**
(-6.6 %), and 66.5 / 65.8 against 72.9 / 70.2 pooled tok/s (+8.1 %). The path with the old host callback
measured 19.07 ms per token.

**The default stays the copy kernel.** The DMA path queues a copy to a card whose window spins on a host
flag, and that is the pattern of upstream issue #31 (a copy that blocks in the driver until the window ends).
The device-raised flag removes the host callback from the window; the host's `cudaMemcpyAsync` remains, so the
pattern is not gone. Use it with the stall watchdog (`STRATA_WATCHDOG_S=300`) as the backstop.

### The Volta defaults table

`src/core/nv_defaults.cpp` has a table per architecture for the switches that are **bitwise
the default engine's output** and faster on that card. The sm_70 table holds the Volta
master switch (`STRATA_SM70_TABLE=1`: the interleaved dense table, the expert mode 8 layout
and the latency-hidden hyper-connection up projection) and
`STRATA_DEQUANT_WIDE=1` for native weight conversion.
`STRATA_SM70_PERF=0` turns the table off. A setting supplied by the user
always takes precedence. The measured service profile is in
[service/strata-v100.service.d](../service/strata-v100.service.d/README.md).

## Dual-PCIe pipeline service profile

The [recent upstream PR review](../benchmarks/v100-upstream-pr-inventory-2026-10-15.md)
includes the V100 pipeline report in upstream PR #2395. The tested local
profile uses GPU 0 on PCIe x4 and GPU 1 on PCIe x16. Keep the GPU order
`0,1` and the explicit layer split `16`. Do not use an equal-link assumption.

With `STRATA_KV_GROW=1`, use a build with the
[pipeline KV-growth repair](../benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md).
The original pipelined decode loop did not grow the mapped cells during a
long response. It could fail with a CUDA illegal memory access even when
the configured context had space left. The repair waits for speculative
rollback and adaptive copies before growing both stages, then resumes
pipelining. Short-output speed checks alone do not test this boundary.

The service template uses a separate private configuration,
`strata-iq3_s-pipeline.json`. Prepare it from the existing configuration:

```sh
python3 - <<'PY'
import json
from pathlib import Path

source = Path("strata-iq3_s-arena.json")
target = Path("strata-iq3_s-pipeline.json")
config = json.loads(source.read_text())
args = config["args"]
if "--pipeline-windows" in args:
    args[args.index("--pipeline-windows") + 1] = "2"
else:
    args.extend(["--pipeline-windows", "2"])
target.write_text(json.dumps(config, indent=2) + "\n")
target.chmod(0o600)
PY
```

Keep both configurations private. Do not commit credentials, model paths,
or conversation data. Install the service template only after the new
configuration exists. Keep the previous engine and service settings for
rollback. This profile does not change the model, quantization, context
capacity, vision settings, or cache budgets.

`--pipeline-windows 2` needs two GPU stages and MTP. It is not a batch-slot
profile. It reserves extra memory for a second verifier and stage snapshots.
Do not apply it to a different machine without measurements. See the
[local evaluation report](../benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
for results, limits, and reproduction commands.

## Prefix-aware conversation storage

L2 keeps one shared K/V backing for matching conversation prefixes.
A short branch has its own token metadata and changed suffix. It does not
remove the original long continuation. The cache charges shared payload
allocations once, including when it admits a branch or evicts another owner.
Writes detach only the ranges that change. The allocation forecast includes
buffer-directory growth and shared suffix writes.

K/V reuse and recurrent-state restore are separate operations. The engine
can share K/V through a branch point without having a recurrent state there.
It must restore a valid earlier state and process the remaining prompt.
An explicit pin can retain an exact boundary. Ordinary history uses bounded
anchors instead of one permanent state per turn. The same anchor selection
applies to every split stage.
A branch inside the saved input freezes its source archive. A later append
can share that source's K/V without rotating away its original resume state.
The active list also admits a predecode input anchor. Reply rewrites can
resume there instead of loading a parked state from the obsolete reply.

Before it shares saved backing after a rewrite, the engine compares the
retained prefix with the active K/V bytes. This includes live cells in a
partial page and completed indexer rows. It excludes unused cells and the
moving spare row. Equal token IDs alone do not permit reuse. If replay
changes boundary cells, the engine can share a shorter verified prefix.

L3 uses prefix references and K/V delta panels. A branch can reference a
shorter extent of a longer chain without truncating that chain. Bounded
`.states` slots hold retained recurrent, PLE, indexer, and draft state.
New panels do not add an inline state copy on every turn. A sidecar write
failure fails the cache write instead of using an unbounded fallback.
The first stored generated end that later input consumes can retain one
ordinary state slot. Dense input checkpoints do not rotate away this original
resume point. Its marker survives restart without increasing the slot limit.
Legacy inline states remain readable. They occupy disk space until their
chain is evicted or replaced. A cut inside a legacy panel still requires
the reader to verify that complete panel's checksum.

The L2 byte limit, slot limit, and free-RAM floor still apply. L2 capacity
victims can move to L3. Disk limits include chain and sidecar bytes.
Pins consume state storage. Finite budgets can still evict conversations.
These changes do not increase the active context limit or remove long-context
decode costs.

The saved input-prompt boundary identifies an obsolete generated tail.
A client can replace that tail without creating a new conversation on each
turn. A branch before that boundary keeps its source continuation.
A nearby checkpoint is not proof of a terminal rewrite. Records without
boundary provenance use the conservative branch path.
If replay changes cells before the input boundary, replacement requires the
remaining original input tokens and image keys to match. An input edit
still creates a separate branch.

### Current native branch check

A native V100 check used a 300,000-token parent and a 20,008-token branch.
No exact boundary pin was set. The first branch reused 16,384 tokens and
processed the remaining prefix. Twenty growing-input turns then reused the
resident input anchor. Their prompt time was 356 to 407 ms.

After those turns, the short branch reused 20,007 tokens. The original long
continuation reused 300,015 tokens and produced the same output as its first
return. Peak L2 storage was 8,508,704,232 bytes across four parked groups.
No capacity eviction occurred. Archive history stayed at six checkpoints.
These figures describe cache allocations, not process RSS or GPU memory.

A 300,000-token L3-only retention check also passed. This check preceded the
final replay-boundary correction. The engine restarted before the 20
growing-input turns. Both long returns
reused 300,015 tokens. The final short return reused 20,007 tokens.
The store held two chains and 6,189,411,036 bytes in total, including
1,298,695,712 bytes of bounded state sidecars. Allocated disk space was
6,189,428,736 bytes.

Disk-only append prompt time was 7.84 to 9.73 seconds in this check.
Disk-only operation still pays for prefix verification and state writes.
The storage and recovery result does not imply RAM-speed disk reuse.

The final writer was also exercised with K/V and running states captured
from that 300,000-token run. The short branch referenced 16,384 source tokens.
Its chain held 55,581,873 bytes, not a second copy of the matching prefix.
The source chain held 4,580,646,417 bytes. Total storage was 5,226,544,706
bytes with the states retained for this replay. The original 300,015-token
state and the 20,023-token branch restored with exact state payloads and K/V.

The final native writer also passed a 40,000-token parent check with a
20,008-token branch, restart, and 20 growing-input turns. It retained two
chains. After repeated parking, the short chain still referenced 16,384
source tokens and held 55,889,222 bytes. Total storage was 1,851,520,352
bytes, including 1,180,632,496 bytes of bounded state sidecars.
The original parent return reused 40,014 tokens and produced the same output.

A separate final check pinned the 19,998-token boundary. The short chain
referenced all 19,998 tokens and held 429,625 bytes. Generated token IDs and
all nine canonical main-state hash fields matched the RAM reference in
four request cases.

All nine focused conversation and session tests passed. All 84 cache-tool
and prefill-tool tests passed. AddressSanitizer and UndefinedBehaviorSanitizer
checks passed 4,394 RAM assertions and 1,863 disk assertions. The disk checks
include branch preservation, bounded state slots,
restart recovery, missing sidecars, and state-publication failures. A dedicated
check replaces 20 generated tails as the input grows. It verifies the latest
image after restart, the original input anchor, and historical-branch refusal.

### Earlier unbounded-archive checks

The following results used the earlier retention policy. They are historical
evidence, not acceptance results for the bounded policy described above.

A deterministic single-GPU engine check used a 12,293-token synthetic
prompt, eight chat continuations with replaced reply tails, and an unrelated
full prompt read between continuations. One archive retained the chain.
After that full read, the archive grew from 28 to 35 checkpoints. The last
three extensions reused 189,330,432 to 189,578,880 KV bytes and took 193.1
to 193.9 ms. No capacity eviction or host-RAM admission failure occurred.
This check does not establish decode speed at a 340,000-token context.

A single-V100 smoke run used a 12,440-token synthetic prompt, eight short
continuations, and edits before the active checkpoint window. The run used
these limits:

```text
--prompt-cache 3 --prompt-cache-every 512
--conversation-cache-mib 8192 --conversation-cache-slots 2
--conversation-cache-min-tokens 0 --conversation-cache-min-free-mib 1024
--conversation-cache-disk <scratch-directory>
--conversation-cache-disk-gib 32
--conversation-cache-disk-min-free-mib 128
```

The L2 archive retained 31 checkpoints after the eight continuations.
An edit at token 6,144 restored the historical L2 checkpoint. The engine
processed the remaining 6,296 tokens. Three unrelated requests then forced
L2 evictions. An edit at token 7,168 restored an L3 checkpoint. The engine
processed the remaining 5,272 tokens. Both requests produced the same output
and main-model state hashes as their cold-reference requests. The logs showed
L2-to-L3 spills and L3-to-L2 promotion. L2 occupancy stayed below 8 GiB and
two slots.

The installed service also passed an HTTP smoke check with its existing
settings: a 524,288-token context, a 25 GiB L2, 32 cache slots, and the
default active checkpoint count. Ten short chat turns completed. A change
to the fifth user message then restored 124 tokens from L2 for a 293-token
prompt. The earlier continuation reference reused 48 tokens. The runtime
log confirmed an L2 restore, not an L3 restore.

The first restart checks failed because of the host memory, not the cache
code. The host uses non-ECC memory and has no error log, so the fault made
no hardware record. A CPU-only pattern test found one bit that always read
back wrong. Two cold engine runs of the same prompt then gave different
model-state fingerprints, so no state comparison was possible.

The operator removed the two 16 GB modules and kept the two 32 GB modules.
A 54 GiB pattern scan of the 62.7 GiB of installed memory found no fault.
Two cold engine runs then gave identical model-state fingerprints, and the
restart and promotion parity checks passed:

- the L3 restore after a full engine restart matched the cold reference;
- the L2 promotion matched the cold reference;
- the model-state fingerprints and the outputs were identical.

The defective module is one of the two removed 16 GB modules. That pair
carries no part number and serial number 00000000. Test each module alone to
confirm which one is defective.

Finite cache budgets still limit history. A checkpoint that both L2 and L3
have evicted is not available for reuse. The engine must process that prefix
again.

## Incremental time to first output

The engine saves an input checkpoint before it changes the prompt state.
The new path captures this state on the GPU. It copies the state to host
memory after the first output window is flushed. The cache receives the
same input checkpoint before the request finishes.

`STRATA_INPUT_ANCHOR_DEFER=0` selects the synchronous control path.
Prefill-only requests keep the synchronous path. A failed GPU snapshot
allocation also uses that path while the input state is still unchanged.
The serial loop reuses its existing snapshot. The two-stage pipeline uses
about 113 MiB of temporary GPU memory across both stages. It releases this
memory after the host copy.

The measured service profile also enables `STRATA_CKPT_POOL=1` and
`STRATA_CKPT_THREADS=1`. These settings reuse checkpoint storage and copy
the stage checkpoints in parallel. They do not change the KV capacity,
KV format, or model quantization.

See the [2026-10-21 TTFT report](../benchmarks/v100-incremental-ttft-2026-10-21.md).
It measures client-observed streaming output, not full response time.
The copy still occurs during the request. Do not interpret a lower
prompt time as an equal increase in decode throughput.

### Expert cache admission after prefill

`STRATA_PREFILL_PROMOTE=N` permits up to `N` expert replacements per stage
when prefill returns a cache loan. The engine default is `0` (off).
The measured service profile selects `2048`. This setting
keeps frequently used experts in GPU slots that later loans do not use.
It replaces less-used experts in the same layer. Route counts reset for
each new loan.

Each replacement first copies the cold expert to a loan slot on the GPU.
It then copies the hot expert from host memory into the stable slot.
The host-copy budget and GPU cache capacity do not increase. Cache maps
change only after these copies finish. Failed preconditions leave the
maps unchanged.

This path requires a full host arena and no batch slots. It is off with
a RAM complement, a peer or helper tier, `STRATA_PREFILL_HELP`, a remote
tier, or a prefill head alias. The engine logs the reason when it rejects
an explicit setting.
Growable KV is supported. Entries with inconsistent active slot maps
are skipped.

This option changes expert placement. Existing resident-aware routing
and CPU/GPU rounding can change the model output. It does not change
expert weights, model quantization, KV format, or KV capacity.

See the [cache admission and dequantization report](../benchmarks/v100-prefill-cache-admission-2026-10-21.md)
for the measured read counts, four-arm TTFT results, and full-request costs.

## Smaller prompt buffers and batched draft K/V

`STRATA_PF_WMMA_LEAN=1` removes unused gathered FP16 input rows,
FP16 expert intermediates, and the fallback dequantization ring on the
direct Volta WMMA path. The expert products already read the original
input through row indices. The option also skips the unused gather.
It leaves the executed expert kernels and their arithmetic unchanged.

This is opt-in. It requires `STRATA_PF_WMMA=1`, compute capability 7.0,
and a native pack whose every layer has a supported direct WMMA format.
Other devices and packs keep the existing allocation. The input and
intermediate buffers needed by other expert paths remain unchanged.

Measured on 2026-10-25: two Tesla V100-PCIE-16GB at 175 W, PCIe Gen3
x4/x16, Ryzen 5 3600, IQ3_S, INT8 KV, 524288-token capacity, split 16,
spec 4, pipeline windows 2, and 8192-token prompt chunks. With lean
buffers, each stage's prompt loan fell from 3.24 to 2.73 GiB.
CUDA0 borrowed 1561 rather than 1845 cache slots; CUDA1 borrowed
1381 rather than 1640. The cache capacity itself did not change.

Two cold prompts per size, in off/on/off restart order, gave these
median prompt times. Both arms used batched draft K/V:

| Prompt tokens | Lean off | Lean on | Lean off again |
| ---: | ---: | ---: | ---: |
| 2048 | 1933.0 ms | 1914.0 ms | 1934.9 ms |
| 8192 | 5258.6 ms | 5082.9 ms | 5270.1 ms |
| 32768 | 14340.9 ms | 14066.4 ms | 14363.8 ms |

The prompt throughput gain was about 3.5% at 8K and 2.0% at 32K.
The 2K difference was small. This is not a decode-speed claim:
256-token decode medians at 8K were 107.5/97.4/103.4 tok/s.
Answers and draft acceptance varied, including between control restarts.
There is no bitwise-output or quality-equivalence claim. Smaller loans
change expert placement, and the service uses residency-biased routing.

A second lean-on restart with a 512-token output cap reproduced the
prompt times (5086.1 ms at 8K, 14058.8 ms at 32K). Its matched
lean-off control read those prompts in 5279.2 and 14363.2 ms.
Decode at 8K was 121.7 tok/s on versus 119.7 off, but the two
2K samples were slower with lean buffers (80.0/118.6 versus
101.5/122.8 tok/s). One 32K control ended at EOS after 440 tokens.
Keep this option off for the daily service until its short-prompt
decode tradeoff is acceptable for the workload. It is a measured
long-prefill option, not a general speed default.

The existing `STRATA_SPLIT_MTP_BATCH=1` option separately improved
prompt time on the installed engine. Its off/on/off medians were
5524.5/5265.4/5538.0 ms at 8K and
15383.4/14348.5/15423.7 ms at 32K. It changes draft K/V arithmetic
and requires a restart. See [the split-draft requirements](MULTI_GPU.md).
Do not interpret either prefill improvement as an equal decode gain.

The [measurement rows](../benchmarks/v100-prefill-2026-10-25.json)
include binary and prompt hashes, decode lengths, and acceptance counts.
The harness kept the daily spec-4 and 750-MiB reserve rather than its
strict spec-8/700-MiB profile. Prompt reuse was disabled.
A threaded short-chunk copy issuer was also built and measured; it
gave no gain and was removed. CUDA sm_70 built and ran. The full HIP
engine also built for gfx1100 with ROCm 7.2.4. HIP execution was not
tested: this machine has no AMD GPU or loaded AMD kernel driver.
The earlier report that the HIP compiler was unavailable was incorrect.
SYCL's separate source was not changed.

`prefill_plan_test`, `draft_kv_plan_test`, `pf_wmma_parity`,
`pf_wmma2_parity`, and `pf_wmma2_n64_parity` passed. The N64 case
completed in 166.59 seconds when run separately after the aggregate
CTest command had reached its 600-second deadline.

To repeat the allocation experiment, export the service's environment
into a private harness config, then run `bench/run_v100_prefill.py`
with `--targets 2048,8192,32768 --repeats 2 --max-new 256
--no-settings-strict --env STRATA_SPLIT_MTP_BATCH=1` and the same
`--exe` in both arms. Set `--env STRATA_PF_WMMA_LEAN=0` or `=1`
and use a separate `--out` and `--arm-label` for each restart.

## Tiled prompt selection

The daily two-V100 profile selects `STRATA_SELECT_SIMT=1`.
On pre-sm_80 CUDA cards this uses a tiled FP32 scorer for prompt
attention selection instead of the older warp scorer. The engine
default stays unchanged; `STRATA_SELECT_SIMT=0` keeps the warp scorer.
No model weights, KV format, cache capacity, or chunk size change.

Measured on 2026-10-25 on the same two-V100 system described above:
IQ3_S, 175 W, INT8 KV, 524288-token capacity, split 16, spec 4,
pipeline windows 2, and 8192-token chunks. Batched MTP was enabled
in every arm; lean buffers were disabled. Each restart read two
uncached prompts per size and generated up to 256 tokens.
The three arms used the same binary and matching prompt hashes:

| Prompt tokens | Scorer off | Tiled scorer | Off again | Throughput gain vs first off |
| ---: | ---: | ---: | ---: | ---: |
| 65536 | 27.609 s | 26.212 s | 27.767 s | 5.3% |
| 131072 | 58.216 s | 51.909 s | 58.235 s | 12.2% |

The longer prompt saved about 6.3 seconds. This is an additional
gain over batched MTP, not a comparison against a slower MTP baseline.
A separate short-prompt run was effectively neutral at 8K and about
2.7% faster at 32K. Do not expect the long-context gain at every size.

The scorer changes FP32 summation order. In the synthetic comparison
against FP64, the maximum score error was 4.36e-5 at 32K and 5.02e-5
at 128K. All 256 query selections matched at 32K; 255 of 256 matched
at 128K (0.0002% of selected cells differed). These checks do not
establish model-quality equivalence. Decode continuations and draft
acceptance varied; no fixed-work decode speedup is claimed.
The alternate tiled scorer (`STRATA_SELECT_SIMT=2`) had the same
score hash at 32K but did not improve its microbenchmark time.

The [measurement rows](../benchmarks/v100-qsa-simt-2026-10-25.json)
record the binary and prompt hashes, decode counts, numerical checks,
and rejected alternatives. A 16384-token chunk was slower at 32K
(14.87 s versus about 14.06 s with lean buffers and 8192-token chunks).
`STRATA_GDN_CHUNK=1` also lost (17.27 s versus about 14.36 s at 32K).
Neither is enabled.

The long-run driver imported `bench/run_v100_prefill.py` and extended
`ALLOWED_TARGETS` with `(65536, 131072)`. Prompt generation, the cold
request protocol, and cooldown settings were unchanged. The normal
harness CLI still permits only its original sizes. For reproduction,
run it through this Python entry point, then append the normal
`--config`, `--exe`, `--arm-label`, and `--out` arguments:

```sh
python -c 'from bench import run_v100_prefill as h; h.ALLOWED_TARGETS += (65536, 131072); raise SystemExit(h.main())' \
  --targets 65536,131072 --repeats 2 --max-new 256 --no-settings-strict \
  --env STRATA_SPLIT_MTP_BATCH=1 --env STRATA_PF_WMMA_LEAN=0 \
  --env STRATA_SELECT_SIMT=1
```

Use `STRATA_SELECT_SIMT=0` for the matched control and separate output
directories for every restart. Preserve the service environment in the
private harness config.

## Small-prompt prefill

The measured two-V100 profile keeps automatic CPU-assisted prefill
below 6144 read rows. It sets `STRATA_PREFILL_STREAM_MIN=6144` and
`STRATA_PREFILL_CPU_SHARE_MAX=6144`; the previous profile used 3072.
This keeps the hybrid path where it helps without extending it through
8K, where it lost to the streamed path.

The engine's new opt-in `STRATA_PREFILL_LOAN_EXACT=1` sizes a fresh
single-chunk loan for its actual read length. Previously, rounding to
256 rows could cross the staged/streamed cutoff and price a larger ring
than the read used. The extra loan evicted useful experts. Exact loans
keep those slots available. Planned continuations and fixed
`--prefill-head-alias` layouts retain their original capacity.
Unset or `0` preserves the old layout.

Install the matching engine before using the new service profile.
Without exact loans, the higher cutoff can still over-borrow near its
boundary. The engine can hold back a token for the first output, so the
read segment may be shorter than the total prompt length.

Measured on 2026-10-25: two V100-PCIE-16GB at 175 W, PCIe Gen3 x4/x16,
Ryzen 5 3600, IQ3_S, INT8 KV, 524288-token capacity, split 16, spec 4,
pipeline windows 2. Batched MTP and tiled selection were enabled in
every arm; lean buffers were off. All arms used the same binary and
matching prompt hashes. Three cold prompts per size were run in the
order old profile, new profile, old profile again (90 requests).

The baseline below pools the six control samples per size. The new
column is the median of its three samples. These are prefill times,
not client time to first output or decode throughput.

| Prompt tokens | Previous profile | New profile | Observed throughput change |
| ---: | ---: | ---: | ---: |
| 1024 | 1.182 s | 1.198 s | -1.3% |
| 2048 | 1.701 s | 1.702 s | -0.1% |
| 3072 | 2.639 s | 2.265 s | +16.5% |
| 4096 | 3.558 s | 2.876 s | +23.7% |
| 5120 | 3.908 s | 3.607 s | +8.3% |
| 6144 | 4.295 s | 4.231 s | +1.5% |
| 7168 | 4.750 s | 4.753 s | -0.1% |
| 8192 | 5.163 s | 5.156 s | +0.1% |
| 9216 | 6.208 s | 6.243 s | -0.6% |
| 10240 | 6.643 s | 6.634 s | +0.1% |

The clear gains are at 3K–5K. Treat the small differences elsewhere as
roughly unchanged, not additional wins. The 4K prompt saves about
0.68 seconds relative to the already-tuned previous profile.

In a separate same-binary comparison at the new cutoff, a 6143-token
prompt took 4.773 s with rounded loans and 4.212 s with exact loans
(three samples each, +13.3% throughput). The trace confirmed a
6142-row segment was laid out for 6142 rather than 6144 rows.
Instrumented smoke requests at 64, 255, 256, 257, and 6143 tokens
completed. The final CUDA sm_70 and HIP gfx1100 engines built and
linked; HIP execution was not tested without an AMD device.

The [measurement rows](../benchmarks/v100-small-prefill-2026-10-25.json)
include binary/prompt hashes, overrides, boundary tests, and rejected
arms. Larger hybrid cutoffs did not improve the 6K–8K range. Fixed CPU
shares of 0.30 and 0.75 were slower at 1K; automatic sharing stays on.
An eight-row CPU down kernel passed bitwise parity and improved its
microbenchmark but gave no full-engine gain, so it was removed.
The idle-stage helper arm was slower at 1K and failed at 2K; it is off.

The scheduling and loan changes affect expert placement and CPU/GPU
rounding. Outputs and draft acceptance can differ. No full-output
bitwise, model-quality-equivalence, or fixed-work decode claim is made.
The engine-wide defaults stay unchanged; the measured crossover is
selected only by this service profile.

## Decode mixer overlap

Set `STRATA_SIDE_MIXER=2` to test input-only mixer work on graph branches.
This setting remains opt-in on Volta. The default stays unchanged.
The indexer branch takes its tail snapshot before it appends speculative rows.
The main stream must not overwrite that snapshot.
The commit graph then restores the accepted prefix without rejected keys.

Set `STRATA_SIDE_MIXER=0` to use the inline path.
Use the same model, KV settings, cache state, and workload in both arms.
Normal MTP throughput can differ from a fixed-work decode measurement.
The setting does not change stored weights, model quantization, KV format,
or logical context capacity.
The [IQ3_S study](../benchmarks/v100-iq3_s-prefill-decode-2026-10-23.md)
records the boundary regression, performance measurements, and limitations.

## Verify window expert cap

`STRATA_VERIFY_ECAP=N` (opt-in, off by default) caps the expert set of a speculative verify window.
The window keeps all K picks of token 0. The speculative tail keeps the best `N-K` picks by routing weight.
A dropped pick joins a kept expert with weight 0. The engine reads no expert for that pick.
Every combine path multiplies by the routing weight, so a dropped pick adds nothing.

The option changes the output. A dropped low-weight expert of a draft row can flip that row.
The engine rejects most draft rows, so the read saving is larger than the loss.
Do not enable it where the answer must be byte-identical.

One expert holds at most `N/K` rows, where `N` is the window's token count.
The uncapped window gives it the same limit, because a token's top-K names different experts.
The CPU expert pool holds `MAXT` rows for each expert, and the engine refuses a wider window at startup.
The kernel keeps that limit. Two kept entries can name the same expert, so the keep list holds different
experts only and the fold skips an expert that is full.

The option applies to decode windows only. A prompt read through wider windows commits every row,
so a cap there would drop real experts with no draft to save.
`N <= K` and a one-token window are no-ops. The default path is unchanged.

The cap only acts when the window's tail is longer than `N-K`: the keep list is token 0's K picks plus the best
`N-K` tail entries, so a window with `N-K` tail entries or fewer loses nothing. The window's width is not fixed;
`--spec-min-p` trims it per token. The daily service, which samples at temperature 1.0, offers `avg T` 1.82, 1.84
and 2.01 over three 512-token requests, so with K=10 the tail holds 8-10 entries and `N=20` is a no-op there.
Choose `N` from the workload's measured `avg T`. At that width on the live service, `N=16` gave +11.5 % and
+11.5 % in two drift-paired comparisons of four 32K requests each (the two control arms drifted 8 % apart, and the
pooled medians move +2.8 %, so the error bar is wide).

Measured on 2026-10-29 on the two-V100 rig below: IQ3_S, INT8 KV, 524288-token capacity, split 16,
spec 4, pipeline windows 2, the daily service environment, one fresh server per arm, the same binary.

End to end, three independent 32K prompts per arm and 512 greedy tokens each
(`bench/run_v100_prefill.py`, `temperature=0`):

| Arm | decode, three prompts (tok/s) | median | paired change |
| --- | ---: | ---: | ---: |
| off | 90.1 / 115.1 / 91.8 | 91.8 | - |
| `=20` | 110.5 / 118.1 / 121.5 | 118.1 | +20.4 / +3.0 / +29.7 % |
| `=16` | 105.7 / 127.9 / 111.9 | 111.9 | +15.6 / +12.8 / +20.1 % |

The mean paired change is +17.7 % at `=20` and +16.2 % at `=16`.
The same three arms read their 32K prompts in 14051, 14071 and 14048 ms (medians, -0.1 % to +0.1 %),
so the prompt path does not move.

The engine's own per-window report over a 32K read and a 256-token output shows where the gain comes from:

| Arm | tokens/window | ms/window | decode |
| --- | ---: | ---: | ---: |
| off | 2.49 | 35.50 | 70.1 tok/s |
| `=20` | 2.81 | 36.06 | 77.9 tok/s |
| `=16` | 3.24 | 36.88 | 87.9 tok/s |

The window's own time does not fall. The committed tokens per window rise, because the cap relaxes the
verification of the draft rows. The window's missed-expert PCIe wait does fall with it: 3.92 to 2.47
and 2.19 ms/window on the second stage's GDN layers.

The end-to-end rows of `bench/run_v100_bench.py` are too noisy at 256 output tokens to separate a
17 % effect: one arm's own repeats span 59 to 94 tok/s at 32K.
The [measurement rows](../benchmarks/v100-verify-ecap-2026-10-29.json) hold both sets.

The prompt path does not move with the driver knobs either: a 32K read measured 14281 ms off against
14259 and 14263 ms with `CUDA_SCALE_LAUNCH_QUEUES=4x` and `CUDA_DEVICE_MAX_CONNECTIONS=32`, which is
the run-to-run spread.

## More opt-in switches (CUDA 12 experimental build)

Unless stated below, these switches are off by default. The sm_70 table sets `STRATA_PF_WMMA2=1`; each switch was
measured by its contributor on a V100, not by the maintainers.

`STRATA_PF_WMMA2` is on by default on sm_70 (`STRATA_PF_WMMA2=0` keeps the first tile only, and
`STRATA_SM70_PERF=0` turns the whole sm_70 table off).
It still requires the experimental CUDA build and the WMMA prompt path; set `STRATA_PF_WMMA=1` as well.
The second tile takes IQ3_XXS, IQ3_S and IQ2_S gate/up and IQ4_NL down.
It reads contiguous weights or the grouped resident/streamed weight table.
It also supports gathered activation rows.
Groups whose largest expert has at least 128 rows use a 128-token by 64-output tile.
Smaller groups use a 64-token by 128-output tile.
The wider token tile reuses each decoded weight across more input rows.
Set `STRATA_PF_WMMA2_N64=0` to keep the 64-token tile, or `=1` to force the 128-token tile.
Other formats, directions, devices or unsupported alignments keep the first tile.
The option does not change stored model quantization, KV format, or KV capacity.
Like the first WMMA tile, it uses FP16 operands and FP32 accumulation.
Do not require bitwise agreement with the CPU or integer MMQ path.
See the [IQ3_S prefill and decode study](../benchmarks/v100-iq3_s-prefill-decode-2026-10-23.md)
for measured gains, fixed settings, and numerical checks.

| Switch | What it does | Contributor's number |
| --- | --- | --- |
| `STRATA_FUSE_DECODE=1` | the decode GEMV / MoE operator fusions of #1097 (gate/up + SwiGLU in one kernel, SwiGLU + quantizer in one kernel, all activation images of a vector in one pass, two BF16 projections in one kernel); bitwise the unfused kernels | 4-16 us per token saved, not visible end to end |
| `STRATA_PF_WMMA=1` | the FP16 tensor-core prompt experts (above); also takes IQ4_NL down projections (K=640, #1771) | +10% prompt speed (#1401) |
| `STRATA_PF_WMMA2` | A second prompt-expert tile for IQ3_XXS, IQ3_S and IQ2_S gate/up and IQ4_NL down. Supports grouped resident and streamed weights. On by default on sm_70; `=0` keeps the first tile only. | Measure on your workload. The original contributor reported +2.5-4.3%. |
| `STRATA_ARGMAX_ROWS_SM70=1` | the multi-block greedy pick on sm_7x (#2357), bitwise the one-block pick | +0.7-0.8% decode |
| `STRATA_GR_NORM_FAST=1` | the register-split hyper-connection norm on CUDA (#2358) | +0.5-0.6% decode |
| `STRATA_Q8_GATHER_WIDE=0` | turns off the 8-wide int8 KV gather (#1095) that the experimental build uses below compute capability 7.5 | +11% on the gather kernel |
| `STRATA_GDN_REC_WARP=2` | Select the chain-split DeltaNet recurrence explicitly. `1` selects the sync-free control. The fork's normal dispatch already uses `STRATA_GDN_REC_FAST` on Volta. For the original pipelined control, set both `STRATA_GDN_REC_WARP=0` and `STRATA_GDN_REC_FAST=0`. `STRATA_GDN_CHUNK=1` selects the chunked path. | Contributor: +6-9% on the recurrence; not a new gain over this fork |
| `STRATA_VERIFY_ECAP=N` | Caps the verify window's expert set: token 0 keeps all K picks, the tail keeps the top `N-K` by weight, and a dropped pick joins a kept expert at weight 0. Opt-in, decode windows only. It changes the output. A cap under one entry per draft row (`N < K + spec - 1`, `spec` being the draft window after the suffix/lookup adjustment) is refused at start-up; under two per row the engine warns. | 32K decode, three paired prompts of 512 greedy tokens: +17.7 % at `N=20` and +16.2 % at `N=16`. 2.49 -> 2.81 and 3.24 tokens/window at a flat ms/window. Prompt unchanged. |
