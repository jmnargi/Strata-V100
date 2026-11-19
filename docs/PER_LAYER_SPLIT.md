# Per-layer split (`--split-mode per-layer`): design, and why a flag alone is not enough

Status: **design, not implemented.** This document exists so the implementation is unambiguous and
so the flag below has a fixed meaning before code lands. The measured budget it is built on is
[2026-11-19-v100-prompt-path-decode-budget.md](2026-11-19-v100-prompt-path-decode-budget.md).

## What the layer split costs a single request

A layer split gives each card its own layers, so a request visits the cards **in turn**: stage 0
(layers 0..K-1) then stage 1 (K..N). Measured on two V100-PCIE-16GB (1,024-token prompt, 256 greedy
tokens, 162 verify windows, `nsys`):

| | ms per window |
| --- | ---: |
| stage 0 GPU work | 6.5 |
| stage 1 GPU work | 15.8 |
| wall | ~20 |
| overlap between the stages | ~15 % |

The stages are serial because the same card is not two cards: stage 1 cannot start until stage 0
hands over, and the next window cannot start until this one is verified. `--layer-split` exists for
**VRAM room**, not for speed - the engine's own docs say so ("the cards work in turn on a single
request, so a split adds expert-cache room, not prompt speed").

A per-layer split changes exactly that: every card holds a slice of **every** layer, so both cards
work on the same layer at once and a request uses both cards' bandwidth together.

## The four slices, and which of them need a reduction

| part | how it splits | needs a sum? |
| --- | --- | --- |
| routed experts (gate/up/down) | each expert's FF width in 32-value blocks, n ways | **no** - each card's rows are a disjoint slice of the FF output; the *down* projection consumes the whole width, so `down` is split by **output** rows instead and needs no sum either |
| dense projections (attn qkv/gate, shared expert, output head) | output rows (vocabulary for the head) | **no** - disjoint outputs |
| attention K/V | by position, blocks dealt round-robin | **yes**, at the output projection (its input is the concatenated heads) |
| hyper-connection mixes | output rows | **no** |

Only the attention output projection needs a reduction per layer, plus the shared residual adds that
already exist. That is why this design is smaller than it looks: the expert and dense slices - 13 %
and 29 % of the window's measured GPU work - are **column-parallel** and need no collective at all,
which is exactly the part NCCL is worst at and the part sfnx builds a hand-written ring for.

## What each slice can win

From the per-window budget (22.3 ms of GPU work in a ~20 ms window):

| slice | ms/window | share |
| --- | ---: | ---: |
| dense matvecs (Q6_K, MMVQ-IL, B6 rows, small, multi) | 6.5 | 29 % |
| staging and doorbell waits | 4.2 | 19 % |
| hyper-connection read (down + up) | 3.5 | 16 % |
| MoE experts (gate/up + down) | 3.0 | 13 % |
| attention, GDN, router, K/V, quantization | 5.1 | 23 % |

If every slice were split 50/50 and the hand-off stayed at one per layer, the window's work halves
and the wall follows: **1.6-1.9x on decode**, which is what sfnx measures (209-211 against 131-135
tok/s on the same class of box). Splitting only the experts would move the window by about 6 % -
which is why the flag below must cover the dense and hyper-connection slices too, not just the MoE.

## The reduction, and why it is not NCCL

Two sums per layer (attention output projection, residual), 96 per window. sfnx measures their
hand-written ring at ~4.5 ms of a 22.5 ms six-row window on three cards, and NCCL at 85-110 µs per
small all-reduce, i.e. ~10 ms a token for the same 96. This fork would write the sum the same way:
GPU threads storing into the peer's memory inside the window's graph, with a device-raised flag
instead of a host callback - the pattern already used here for the stage hand-off and the doorbell
(`wait_flag_ge`, `doorbell_publish`).

## The flag

```
--split-mode layer|per-layer        STRATA_SPLIT_MODE=layer|per-layer
```

- `layer` (default): today's behaviour, byte-identical.
- `per-layer`: requires an even split of every shape (experts' FF width in 32-value blocks, the
  vocabulary, the K/V positions), one primary that owns the sampler and the logits, and at least
  two visible cards of the same compute capability. It refuses, with a named reason, when the
  pack's shapes, the KV type, or `--batch`/`--peer-device`/helper caches make it impossible.

Startup must print the plan it will run - slices per card, the sums per window, and the VRAM each
card keeps for experts - so a measurement is comparable to another one.

## Phased implementation

1. **Placement only** (no code beyond the plan): `per-layer` computes and prints the slice plan, and
   *refuses to serve*. This is the phase that fixes what the numbers mean. Not a usable mode.
2. **Dense + HC slices** (no reduction): the biggest win per unit of risk (45 % of the window), and
   it makes the residual and the sampler agreement testable on its own.
3. **Expert slices** (no reduction): per-card FF-width slices, the cache becomes "half of every
   expert of every layer" instead of "all experts of half the layers".
4. **K/V by position + the two sums per layer**: the only part that needs the ring, and the only
   part that moves attention off one card.

Each phase lands behind the same flag with its own parity test: with `per-layer` off, every output
must stay bit-identical to today; with it on, acceptance is measured against the layer split's
tokens (top-1 agreement and a KL bound), as sfnx does for its own split.

## What exists today, and why it is not this

`--peer-device N` is the closest existing thing: a primary card keeps the dense layers, KV, MTP and
its own expert cache, and a secondary card holds **only** an expert cache and computes its rows.
Measured on this rig with this model (2x V100-PCIE-16GB, UD-IQ3_XXS), it is not usable:

- **Expert residency falls**: 8,913 of 24,576 `(layer, expert)` pairs on the GPUs (960 on the
  primary, 7,953 on the peer) against **11,663** in the layer split. Less residency means more
  expert traffic over PCIe, which is the opposite of what a per-layer split is for.
- **It faults on the prompt**: `prefill: the PLE rows' upload failed: an illegal memory access` once
  the PLE table is in play, with `--ple-io mmap` and without it. This model needs PLE, so the path
  cannot serve it today. The engine now refuses that combination at start-up instead of faulting
  mid-prefill (`--peer-device` with `--ple-gguf`).
- It disables `STRATA_ADAPT_OVERLAP`, `STRATA_PREFILL_PROMOTE` and the two extra Volta switches that
  the layer split keeps, so even a working peer would start from a slower baseline.

So a per-layer split is a new data flow for the window, not a flag over the peer tier. The repo's
contributing rules ask for the scope to be agreed before a change this size; this document is that
scope.

## Risks

- **VRAM**: half of every expert needs both halves of every layer resident somewhere; the packing
  must be a clean partition or the cache shrinks (see the peer measurement above).
- **Correctness**: slices change summation order for the attention output projection. The layer
  split's answers are already "the GPU's order, not the CPU's"; a per-layer split needs its own
  documented tolerance and a parity test per phase, not per release.
- **The x4 card**: this rig's GPU 0 is on PCIe Gen3 x4 (2.9 GB/s). A split weighted evenly would
  stall on it; the plan must weight slices by the measured link, which the engine already probes
  (`pcie_frac`).
- **CUDA graphs**: the window is captured; peer writes must be device-visible inside the graph, as
  the existing hand-off is, or the capture must fall back.
