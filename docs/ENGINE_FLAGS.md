# Engine flags

The engine is one binary, `engine/strata` (source `src/program/generate.cpp`). It has no
tokenizer: the prompt arrives as token ids. The binary has two start modes.

- One-shot run: `engine/strata --pack DIR --tokens "1,2,3" [flags]`. The usage text labels
  this form `strata generate`.
- Server mode: `engine/strata --serve --pack DIR [flags]`. The process stays resident and
  reads requests on stdin, one `GEN <max_new> <id,...>` line each. See [SERVER.md](SERVER.md).

`--tokens` or `--tokens-file` is required for a one-shot run. `--serve` supplies its own
default prompt (token `248045`, `--max-new 1`) when no token input is given.

A flag whose description begins with `serve` has an effect only in server mode. The server
passes these flags from the run config's `"args"` list; see [How to set a flag](#how-to-set-a-flag).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success; also `--help`, `--version`, `--families`, `--graph-only`, `--gpu-only-full`, and `--gpu-stages` |
| `1` | the run failed after start, for example a model load, a CUDA fault, or a profile read |
| `2` | usage error: an unknown flag, a missing value, a missing `--tokens`, or a value out of range |

These three flags exit before any GPU work:

- `--version` (or `-V`) prints `strata 0.1.45.8` and exits `0`.
- `--families` prints the GGUF architectures this build runs (`qwen4exp`, and `qwen35moe`
  when the Qwen3.6 path is built) and exits `0`.
- `--help` (or `-h`) prints the usage text to stderr and exits `0`.

## Model, pack, and shards

| Flag | Value | Default | Description |
|---|---|---|---|
| `--pack` | `DIR` | `pack/full` | the directory of the converted expert pack that the engine reads |
| `--model` | `PATH` | none | the model's first shard; alias of `--native`; the Qwen engine requires it; the main engine reads it only in the family switch |
| `--native` | `SHARD1` | none | one model shard in native layout; turns on every native path at once |
| `--native-dense-gguf` | `PATH` | none | native gated delta-net (GDN), full-attention (QSA), and shared projections; repeat for each source shard |
| `--native-head-gguf` | `PATH` | the `--native` model | native Q5_K output head; requires `--stream-token` |
| `--embd-gguf` | `PATH` | the `--native` model | the token embedding from this GGUF, in mapped host memory |
| `--ple-gguf` | `PATH` | none | the per-layer embedding (PLE) table from the original second GGUF shard |
| `--no-ple` | no value | off | explicit diagnostic ablation of the PLE layer |
| `--families` | no value | none | print the GGUF architectures this build runs, then exit |
| `--gpu` | `LIST` | every visible GPU | run on these PCI indices; sets `CUDA_VISIBLE_DEVICES` (or `HIP_VISIBLE_DEVICES`) and PCI order |

`--model` is an alias only where the family switch runs: the Qwen3.6 engine requires it, and
the main engine rejects it after the switch. A native (IQ) pack needs `--native SHARD1`,
`--spec T` with `T >= 2`, and `--prefill CHUNK` (from the code). If no shard holds the PLE
table, `--native` needs `--ple-gguf` or `--no-ple`.

NOTE: a missing `--ple-gguf` makes the engine skip layer 1's PLE without an error. Every
number downstream changes. Pass it for any real run.

## Context and RoPE

The engine applies rotary position embeddings (RoPE) to the attention layers.

| Flag | Value | Default | Description |
|---|---|---|---|
| `--max-context` | `N` | `4096` | KV and state capacity in tokens |
| `--rope-scaling` | `none\|linear\|yarn` | the model file's keys, else `none` | extend the context past the trained one; llama.cpp's types and names |
| `--rope-scale` | `F` | `0` (the model file's factor, else `1` = off) | the extension factor for linear or yarn |
| `--rope-freq-base` | `N` | `0` (the model's `1e7`) | the raw ggml frequency base |
| `--rope-freq-scale` | `F` | `0` (`1 / --rope-scale`) | the raw ggml angle shrink |
| `--yarn-orig-ctx` | `N` | `0` (the model's, else `262144`) | the trained context that the YaRN correction targets |
| `--yarn-ext-factor` | `F` | `-1` (auto: `1` for yarn, else `0`) | the YaRN extrapolation mix |
| `--yarn-attn-factor` | `F` | `1` | the YaRN attention magnitude factor |
| `--yarn-beta-fast` | `F` | `32` | the YaRN fast-rotation correction threshold |
| `--yarn-beta-slow` | `F` | `1` | the YaRN slow-rotation correction threshold |

An explicit flag wins over the model file's rope keys, and the model file wins over the
struct defaults. The scaling is fixed at startup: K sits in the cache after RoPE, so one run
uses one scaling. For `--rope-scale`, `0` is the absent value, so the model file's factor
applies first, then `1`.

## KV cache

The engine keeps the key-value (KV) state of each attention layer in VRAM by default.

| Flag | Value | Default | Description |
|---|---|---|---|
| `--kv` | `fp16\|int8\|q4_0\|k8v4\|tq3\|tq4\|tq6\|tq8` | `fp16` | KV storage; int8 halves the VRAM, q4_0 halves it again, tqN uses rotated codebooks; all but fp16 change the answers |
| `--kv-resident` | `N` | `0` (all in VRAM) | keep N cells of each full-attention (QSA) layer in VRAM; the rest streams from pinned RAM |
| `--kv-grow` | no value | off | the K/V takes VRAM only for the cells requests reach; needs `--expert-profile` |
| `--no-kv-grow` | no value | off | turn the elastic K/V off; the same as the default |

`--kv-resident` takes a minimum of 20480 cells. `--kv tq3|tq4|tq6|tq8` cannot be combined
with `--kv-resident` (exit `2`). `--kv-grow` also reads `STRATA_KV_GROW=1`.

NOTE: `--kv-grow` turns itself off with a message unless four conditions hold. The profile is
set, `--kv-resident` is `0`, the expert cache is on, and virtual memory management (VMM) is
available on every GPU. It also excludes `--batch`, `--vram-elastic`, and `--peer-device`.

NOTE: a TurboQuant K/V has no host copy to snapshot, so the MTP draft layer's ring stays INT8
under `--mtp`.

## Prompt processing

| Flag | Value | Default | Description |
|---|---|---|---|
| `--prefill` | `CHUNK\|auto[\:N]` | `0` (token path) | batched prompt chunks; `auto` takes the largest chunk whose buffers the cache can lend |
| `--prefill-pipe` | `R` | `-1` (not given; off) | a layer split: size prompt chunks so the stages read at once; `1` only evens the chunk count |
| `--prefill-until` | `P` | `0` (whole prompt but the last position) | validation: batch positions `0..P-1`, then run the rest through the token path |
| `--prefill-head-alias` | no value | off | serve: unload the native output head while a fresh prompt is read, then reload it |
| `--short-read` | `N` | `64` | serve: read up to N fresh text tokens through the decode windows; `0` = batched path |
| `--no-prefill-borrow` | no value | off (borrow on) | reserve the prompt buffers' VRAM for the session instead of lending cache slots |

`--prefill auto` caps at `8192` tokens by default; `auto:16384` or `auto:32768` raises the
cap. A named chunk is never capped. `--prefill-pipe` also reads `STRATA_PREFILL_PIPE`.

NOTE: with `--no-prefill-borrow` or no `--expert-profile`, the engine turns `--prefill auto`
off and uses `2048`-token chunks, because no cache can lend the buffers.

## Speculative decoding and the draft layer

The multi-token prediction (MTP) draft layer proposes the tokens of a verify window.

| Flag | Value | Default | Description |
|---|---|---|---|
| `--mtp` | `DIR` | none | runtime directory of the MTP draft layer |
| `--spec` | `N` | `0` (plain decode; `2` on Metal) | verify window size: the last accepted token plus `N-1` drafts |
| `--spec-min-p` | `P` | `0` (always `--spec-1` drafts) | the draft layer's probability floor for another guess; setup writes `0.5` |
| `--spec-split` | no value | off | opt-in split verify window; exact, and slower (comment in `generate.cpp`) |
| `--no-spec-split` | no value | off | turn the split verify window off; the same as the default |
| `--suffix-draft` | `N` | `3` | draft from an earlier repeat of the last N+ tokens when the draft policy expects a gain; `0` = MTP only |
| `--mtp-window` | `N` | `32768` | the draft layer attends to the last N cells; `0` = every cell |
| `--mtp-max-t` | `M` | `0` (= `--spec`) | cap the MTP's windows at M tokens; longer windows come from suffix matches |
| `--mtp-hnorm` | `pooled\|stream` | `pooled` | one RMS over all four streams, or one per stream (llama.cpp's graph) |
| `--mtp-q4` | `all\|proj\|head` | none (off) | 4-bit copies of the draft projections and/or the draft head |
| `--mtp-draft-vocab` | `PATH` | `<mtp>/draft_vocab.bin` | the draft head's token subset file |
| `--lookup-chain` | `K` | `0` (off) | append up to K prompt-lookup drafts after the MTP's drafts; the window grows to at most 8 |
| `--lookup-chain-min` | `M` | `3` | the shortest context match that `--lookup-chain` extends on |
| `--coupled-draft` | no value | off | the draft layer samples with the target's chain instead of proposing the argmax; `STRATA_SPEC_COUPLED=1` also turns it on |
| `--no-coupled-draft` | no value | off | propose argmax drafts; turns coupled sampling off |
| `--batch-mtp` | no value | off | each batch slot verifies one MTP proposal per window; needs `--mtp`, `--spec`, and `--batch` |

NOTE: with `--suffix-draft` above `0` and `--mtp-max-t 0`, the engine sets the MTP cap to
`--spec`. It then raises the window by 2 tokens, up to 8. A wider window costs decode time
when the draft layer is unsure, so measure `--spec` on the target text.

`--batch-mtp` also reads `STRATA_BATCH_MTP=1`. It does not combine with pipelined
`--batch-groups` unless `--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` is also given.

## Expert placement in VRAM and RAM

| Flag | Value | Default | Description |
|---|---|---|---|
| `--expert-cache` | `N\|auto` | `0` (off) | keep N expert blobs in VRAM and compute their rows on the GPU; `auto` sizes from free VRAM |
| `--expert-cache-cpu-order` | no value | off | experimental: GPU expert reduction in the CPU order |
| `--expert-cache-device1` | `N\|auto` | `0` (off) | fill CUDA1 with N experts or as many as fit |
| `--expert-cache-device2` | `N\|auto` | `0` (off) | fill CUDA2 with more experts |
| `--expert-cache-device3` | `N\|auto` | `0` (off) | fill CUDA3 with more experts |
| `--expert-cache-per-layer` | no value | off | give each layer its own slots instead of one shared counter |
| `--expert-cache-remote-placement` | `stripe\|layer` | `stripe` | distribute expert ranks or whole layers across CUDA1..3 |
| `--expert-profile` | `PATH` | none | pre-load the VRAM tier from a `profile.bin` instead of admitting on first use |
| `--expert-profile-save` | `PATH` | none | serve: save the tier's experts and routing counts as a profile on quit and on a timer |
| `--expert-profile-save-every` | `MIN` | `10` | minutes between profile saves; `0` = on quit only |
| `--adapt-every` | `N` | `4` | adaptive tier: swap experts every N rounds; `0` = static residency |
| `--adapt-swaps` | `N` | `96` | experts swapped per adaptation step |
| `--adapt-decay` | `F` | `0.7` | the routing counts are multiplied by this after each adaptation |
| `--adapt-min-gain` | `F` | `1.5` | swap an expert in only when its decayed count beats the replaced one by more than F |
| `--adapt-async` | `0\|1` | `0` (blocking) | serve with the resident RAM mode or arena: swaps advance between windows; not bit-exact |
| `--resident-cpu-experts` | no value | off | with mmap and a static profile, keep cache misses in ordinary RAM; requires both |
| `--resident-experts` | no value | off | the low-RAM mode: mmap, resident CPU experts, page-locked copies, 4 GiB headroom |
| `--resident-budget-gib` | `N` | `0` (whole complement) | the N GiB of non-resident experts, ranked by the profile, copied into RAM |
| `--mmap-experts` | no value | off (arena on) | opt out of the resident expert arena and map the expert files instead |
| `--shared-expert-arena` | `PATH` | none | Linux: back the resident arena with one `MAP_SHARED` file; put it on `/dev/shm` |
| `--vram-elastic` | no value | off | serve, NVIDIA: expert caches in segments so `POST /v1/vram` can lend VRAM between requests |
| `--vram-segment-mib` | `N` | `512` | the segment size for `--vram-elastic` |
| `--vram-reserve-mib` | `N` | `700` | VRAM left free for graphs, hit scratch, and the head when `--expert-cache auto` sizes the tier |
| `--vram-reserve-later-mib` | `N` | `-1` (same as the first card) | the reserve on a layer split's later cards |
| `--pcie-frac` | `F` | `-1` (model default: `0.2` direct for Q2_0, `0.55` DMA for native) | share of each layer's missed experts that the GPU reads over PCIe |
| `--pcie-mode` | `auto\|dma\|kernel\|direct` | `auto` | the PCIe read path |
| `--peer-device` | `N` | `-1` (off) | a second GPU as an adaptive expert tier; not with `--layer-split` or helper caches |
| `--peer-reserve-mib` | `N` | `600` | VRAM that the peer tier leaves free |
| `--peer-slots` | `N` | `0` (as many as fit) | expert slots on the peer |
| `--peer-adapt-swaps` | `N` | `-1` (= `--adapt-swaps`) | peer cache swaps per adaptation step |
| `--peer-prefill-rows` | `N` | `-1` (half of chunk x top-k) | prompt rows per layer that the peer computes; `0` = the prompt path stays on the primary |
| `--remote-expert-opt` | no value | off | enable remote-expert decode optimizations in serve mode |
| `--no-hit-poke` | no value | off | A/B arm: skip the driver poke after the hit launch, so the hit work does not overlap the pool |
| `--split-skip-if-fits` | no value | off | with `--layer-split auto`, stay on CUDA0 alone when it holds every profiled pair and the session |

NOTE: when `--expert-profile` is given, `--expert-cache 0` means `auto`, not off (from the
code). `--expert-profile` also decides residency instead of the compulsory-miss policy.
`--mmap-experts` cannot combine with `--shared-expert-arena`, and `--resident-cpu-experts`
needs `--mmap-experts` plus a static `--expert-profile`.

## CPU pool and thread placement

| Flag | Value | Default | Description |
|---|---|---|---|
| `--pool-workers` | `N` | `0` (every physical core except the host's) | the worker count of the CPU expert pool |
| `--pool-tasks` | `N` | `0` (3 per participating thread) | batched CPU expert tasks per GU/Down phase; range `0..4096` |
| `--pool-affinity` | `all\|auto\|p-cores\|measure` | `all` | worker CPU layout; `measure` times `all` against `p-cores` at start |
| `--host-core` | `first\|last\|sibling` | `first` | the host thread's core; moves threads only, never a result |
| `--aux-cpus` | `off\|auto\|LIST` | `off` | where non-pool, non-host threads run, for example `24,26` |
| `--no-host-worker` | no value | off | keep the host thread out of the drain; the A/B arm of the default |
| `--no-pool` | no value | off | skip the CPU expert pool: the GPU-only floor |
| `--no-split-rows` | no value | off | A/B arm: one whole expert per pool thread instead of split rows |

`--host-core` and `--aux-cpus` also read `STRATA_HOST_CORE` and `STRATA_AUX_CPUS`. The Qwen3.6
engine has its own `--cpu-threads`; see the Qwen section.

## Multi-GPU

| Flag | Value | Default | Description |
|---|---|---|---|
| `--layer-split` | `K[,K2..]\|auto` | none (one GPU) | serve: later stages run from layer K on their own GPU; needs `--expert-profile` across GPUs |
| `--split-device` | `D1[,D2..]` | the next visible GPUs | the later stages' devices; `0` with one K keeps both stages on one GPU |
| `--pipeline-windows` | `0\|1\|2` | `-1` (auto: off, or `2` for a qualifying gfx12 split) | `1` overlaps the prompt's short reads; `2` overlaps decode too |
| `--batch-groups` | `G\|auto` | `1` (auto for a layer split with `--batch`) | the batch slots in G groups pipelined through the stages |

An explicit `--pipeline-windows` is turned off with a message when it cannot run. The reasons
are: no `--serve`, no layer split, no MTP with `--spec 2` or more, `--batch`, `--peer-device`,
or helper caches.
See [MULTI_GPU.md](MULTI_GPU.md) and [BATCHING.md](BATCHING.md).

## Conversation cache, prompt cache, and sessions

| Flag | Value | Default | Description |
|---|---|---|---|
| `--conversation-cache-mib` | `N` | `0` (off) | serve: RAM budget for parked conversations |
| `--conversation-cache-slots` | `N` | `4` | at most N parked conversations |
| `--conversation-cache-min-free-mib` | `N` | `2560` | RAM floor under `MemAvailable`, the cgroup limits, and `--memory-limit-mib` |
| `--conversation-cache-min-tokens` | `N` | `0` (park any) | park only conversations of at least N tokens |
| `--conversation-cache-park-gain` | `N` | `0` (park every turn) | park only after the conversation advanced N tokens past the superseded copy |
| `--conversation-cache-disk` | `DIR` | none | serve: L3 disk store for parked conversations; needs `--conversation-cache-disk-gib` |
| `--conversation-cache-disk-gib` | `N` | `0` (off) | the disk store's file budget in GiB; must be above `0` with a directory |
| `--conversation-cache-disk-slots` | `N` | `0` (no cap) | at most N conversation records on disk |
| `--conversation-cache-disk-min-free-mib` | `N` | `0` (no floor) | free-space floor that the disk store keeps |
| `--conversation-cache-keep-root` | no value | on | serve: pin the shared system-prompt prefix in the RAM cache |
| `--no-conversation-cache-keep-root` | no value | off | do not pin the shared system-prompt prefix in the RAM cache |
| `--conversation-cache-disk-keep-root` | no value | on | keep the system-prompt prefix record on disk after conversation records |
| `--no-conversation-cache-disk-keep-root` | no value | off | do not keep the system-prompt prefix record on disk |
| `--memory-limit-mib` | `N` | none | total RAM that the container can use; also `STRATA_MEMORY_LIMIT_MIB`; the flag wins |
| `--prompt-cache` | `N` | `6` | serve: keep N conversation checkpoints between requests; each uses about 118 MB of RAM |
| `--prompt-cache-every` | `N` | `16384` | also checkpoint every N fresh prompt tokens; `0` = turn boundary only |
| `--prompt-cache-root` | `N` | `2048` | checkpoint the first turn boundary when it is at least N tokens; `0` = never |
| `--prompt-cache-tail` | no value | off | one extra checkpoint near the prompt's end, at an existing chunk boundary |
| `--prompt-cache-spread` | no value | off | keep checkpoints spread back from the end instead of most recently used |
| `--session-min-free-mib` | `N` | `4096` | serve: disk space that a SAVE must leave free; `0` = no check |
| `--session-save-reclaim` | no value | off | serve: reclaim unpinned host caches if SAVE needs RAM |
| `--turn-token` | `ID` | `248045` | serve: the token that opens a chat turn (`<\|im_start\|>`) |
| `--tail-role-token` | `ID` | `-1` (off) | serve: leave a trailing turn of this role out of the conversation checkpoint |

The disk tier is off unless both `--conversation-cache-disk` and a positive
`--conversation-cache-disk-gib` are given, and it needs `--serve`. `--prompt-cache 0` or
`--conversation-cache-slots 0` disables conversation caching. See
[PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md).

## Batch slots

| Flag | Value | Default | Description |
|---|---|---|---|
| `--batch` | `N` | `0` (off) | serve: up to N requests decode together, one token per window; each slot has its own session |
| `--slots` | `N` | `0` (off) | alias of `--batch`; the same variable |

`--batch` needs `--serve` and at least 2 slots. A window holds at most 8 rows, so a larger
`--batch` is reduced unless pipelined groups on a layer split carry it. See
[BATCHING.md](BATCHING.md).

## Sampling

| Flag | Value | Default | Description |
|---|---|---|---|
| `--greedy` | no value | on | argmax token selection; the default |
| `--seed` | `S` | `0` | sample with this Philox seed instead of the argmax |
| `--top-k` | `N` | `20` | keep the N best tokens at each step when the run samples |
| `--top-p` | `F` | `0.95` | the nucleus cutoff when the run samples |
| `--temperature` | `F` | `1.0` | the token temperature when the run samples |
| `--stop-eos` | no value | off | stop at an end-of-turn token |
| `--eos-ids` | `LIST` | `248044,248046` | the stop token ids; giving this flag also turns `--stop-eos` on |

A run is greedy until `--seed` is given. The server sends sampling per request; these flags
set the engine default for a one-shot run.

## Control vectors

| Flag | Value | Default | Description |
|---|---|---|---|
| `--control-vector` | `FILE` | none | a control vector GGUF at scale `1.0` on the residual stream |
| `--control-vector-scaled` | `FILE:S[,FILE:S..]` | none | control vectors with a scale per file |
| `--control-vector-layer-range` | `A B` | `1` to the last layer | the inclusive layer range that the vector follows |
| `--cvec-mode` | `add\|project` | `add` | `add` does `h += s v`; `project` does `h -= s (h.v) v` |
| `--cvec-dir` | `per-layer\|single:L` | `per-layer` | each layer's direction, or layer L's direction everywhere (project) |

`--serve` switches a loaded control vector per request (`cvec=0|1`). These flags change the
answers.

## Vision

| Flag | Value | Default | Description |
|---|---|---|---|
| `--vision` | no value | off | serve takes images too (GENI requests); embeddings come from `strata-vision` |

See [IMAGES.md](IMAGES.md).

## Native and experimental kernel switches

| Flag | Value | Default | Description |
|---|---|---|---|
| `--native-bf16` | no value | off | experimental: BF16 projections for the state-space model (SSM), router, and indexer |
| `--native-bf16-extra` | no value | off | experimental: BF16 projections for the PLE value and shared expert gate |
| `--native-gdn` | no value | off | experimental: pinned CUDA GDN norms, gates, and recurrence |
| `--native-moe-combine` | no value | off | experimental: pinned CUDA routed and shared expert combination |
| `--native-ple-key` | no value | off | experimental: native PLE key; requires `--native-dense-gguf` |
| `--native-ple-postops` | no value | off | experimental: pinned PLE post-projection arithmetic |
| `--native-qsa` | no value | off | experimental: pinned QSA normalization and output gate |
| `--native-qsa-indexer` | no value | off | experimental: pinned indexer key cache and pooling |
| `--native-rope` | no value | off | experimental: pinned text-only CUDA rotary arithmetic |
| `--native-router` | no value | off | experimental: pinned CUDA 512-expert top-10 routing |
| `--native-flash-attn-short` | no value | off | diagnostic: pinned vector attention; requires `--max-context` of 256 or less |
| `--gr-fp32-activations` | no value | off | experimental: CUDA-oracle GR activation precision |
| `--gr-native-mmvf` | no value | off | experimental: pinned GR norm and projections; implies FP32 activations |
| `--cpu-oracle-q8-0` | no value | off | experimental: pinned x86 activation scales and codes at both expert stages |
| `--no-fast-attn` | no value | off | A/B: gather plus one-block-per-head QSA attention |
| `--no-fast-select` | no value | off | A/B: FP64 row scores and bit-serial cell top-k |
| `--no-fused-gr` | no value | off | A/B: six-kernel hyper-connection read and a separate write |
| `--no-fused-gdn` | no value | off | A/B: llama.cpp-layout GDN step and a separate output norm |
| `--no-token-graph` | no value | off | A/B: two graphs per layer instead of one graph per token |
| `--no-capture` | no value | off | run the layers directly without a graph replay |
| `--no-publish-kernel` | no value | off | A/B: memcpy nodes for the doorbell and QSA step |
| `--shared-late` | no value | off | A/B: the shared expert runs after the CPU pool instead of overlapped with it |
| `--keep-canonical` | no value | off | A/B: also load canonical copies of natively served tensors; more VRAM |

These flags select a kernel path. They do not change the model file. Most are experimental
A/B arms, and the pinned paths change the arithmetic order, so cross-build output can differ.

NOTE: `--native-flash-attn-short` cannot run with `--gpu-only-full`, `--graph-only`, or
`--gpu-stages`. `--no-capture` refuses `--dump-layers` and `--dump-halves`.

## Diagnostics and measurement

| Flag | Value | Default | Description |
|---|---|---|---|
| `--dump-logits` | `PATH` | none | write one line of raw logits per stored position |
| `--logits-stride` | `N` | `1` | store every Nth logits row; `N > 1` needs `--max-new 1` and `--dump-logits` |
| `--dump-residual` | `PATH` | none | write the final residual (`hc x n_embd`, f32) |
| `--dump-mixed` | `PATH` | none | write the post-attention residual that the head reads |
| `--dump-layers` | `PATH` | none | write the residual after every layer and position; refused with `--no-capture` |
| `--dump-halves` | `PATH` | none | write both halves' block outputs and injections per layer; needs the captured path |
| `--dump-routing` | `PATH` | none | write the routed expert ids and weights per layer and position; refuses `--no-pool` |
| `--dump-final-r` | `PATH` | none | append the residual after the last layer for every position; token path only |
| `--stats` | no value | off | print the per-stage breakdown and the pool numbers |
| `--stage-timing` | no value | off | per-stage kernel-count shares; requires `--no-capture`; not a time profile |
| `--graph-only` | no value | off | measure: replay the 48 `pre` graphs only, then exit; not the GPU floor |
| `--gpu-only-full` | no value | off | measure: `pre`, `post`, and the head with no pool, then exit; the GPU floor for one token |
| `--gpu-stages` | no value | off | measure: capture each layer as three graphs and time them from outside, then exit |
| `--window-hashes` | `PATH` | none | per verify window: index, position, size, a residual hash, and the argmaxes |
| `--spec-follow` | `PATH` | none | emit this continuation and accept matching drafts, so speculation settings compare on one text |
| `--spec-oracle` | `PATH` | none | draft from this token file (the expected continuation) for the exactness test |
| `--spec-corrupt` | `N` | `0` (off) | replace every Nth draft with a wrong token |
| `--stream-token` | no value | off | experimental: enqueue token work on the session stream |
| `--check-logits` | no value | off | copy and check all logits in the stream-token path |
| `--sync-every-layer` | no value | off | debug: synchronize after every layer |

NOTE: quote `--gpu-only-full` as the GPU floor. `--graph-only` omits the 48 `post` graphs and
the LM head, so it is not the floor. `--window-hashes` is meaningful with
`--spec-follow --adapt-every 0`.

## Qwen3.6-35B-A3B flags

These flags belong to the Qwen3.6-35B-A3B engine path in `src/qwen35/q36_engine.cu`. The main
binary switches to that path from the model header. `--model` names the model and is required
there.

| Flag | Value | Default | Description |
|---|---|---|---|
| `--chunk` | `N\|auto` | `0` (auto) | prompt chunk in tokens; `auto` picks the largest chunk that fits beside every expert |
| `--ctx` | `N` | `32768` | KV capacity in tokens; the same as `--max-context` |
| `--emb` | `PATH` | none | pictures file for `--tokens`; requires `--vision` |
| `--n` | `N` | `64` | tokens to generate; the same as `--max-new` |
| `--expert-vram` | `F` | `-1` (free VRAM minus 768 MiB) | GiB budget for the routed experts |
| `--hybrid-rows` | `N` | `128` | hybrid mode: up to N rows use the VRAM slots and the CPU pool together |
| `--expert-slots` | `N` | `0` (all that fit) | experts resident in VRAM per layer in hybrid mode |
| `--draft-vocab` | `N` | `-1` (all ids) | the MTP drafts pick among the first N token ids |
| `--bench-pp` | `N` | `0` (off) | measure a prompt of N tokens, then exit |
| `--bench-tg` | `N` | `0` (off) | measure N generated tokens, then exit |
| `--no-graph` | no value | off | decode one token at a time without a captured graph |
| `--no-warmup` | no value | off | skip the warmup pass |
| `--force` | `IDS` | none | emit this continuation without token selection; `--n` is set from its length |
| `--dump-full-logits` | `PATH` | none | write full logits in binary |
| `--cpu-threads` | `N` | `0` (every physical core except the first) | the thread count of the CPU expert pool in hybrid mode |

`--ctx` accepts the range from 256, and `--chunk` accepts `0..4096`. `--draft-vocab` also
reads `STRATA_Q36_DRAFT_VOCAB`. That engine also accepts the shared names `--native`,
`--max-context`, `--prefill`, `--max-new`, `--pool-workers`, `--serve`, and `--gpu`. It also
accepts `--greedy`, `--batch`, `--slots`, `--batch-mtp`, `--vision`, `--no-token-graph`, and
`--mtp N` (a draft count). In serve mode it ignores a flag it does not know, with a message. Defaults that
differ there are listed in the table. See [QWEN36.md](QWEN36.md).

## Other flags

| Flag | Value | Default | Description |
|---|---|---|---|
| `--tokens` | `LIST` | none | the prompt as comma-separated token ids; required unless `--serve` |
| `--tokens-file` | `PATH` | none | the prompt from a file, commas or whitespace; one token input only |
| `--max-new` | `N` | `16` | tokens to generate |
| `--serve` | no value | off | stay resident and read requests on stdin (`GEN` lines) |
| `--help` | no value | none | print the usage text to stderr and exit `0` |
| `--version` | no value | none | print `strata 0.1.45.8` and exit `0` |
| `--ple-io` | `direct\|mmap\|ram` | `direct` | PLE table reads: unbuffered SSD, mmap, or the whole table locked in RAM |
| `--ple-row-cache` | `N` | `1048576` rows | cache budget in rows of 90 B, held as whole pages; `0` = off |
| `--ple-inflight` | `N` | `256` | outstanding PLE SSD reads; range `1..1024` |
| `--ple-delay-us` | `U` | `0` (off) | fault injection: a row read completes no earlier than U microseconds |
| `--ple-sync-submit` | no value | off | A/B: submit table reads on the token thread instead of an I/O thread |
| `--no-ple-prefetch` | no value | off | A/B: read the table's sixteen rows one at a time instead of one prefetch call |

`--serve` sets `--max-new 1` and `--stop-eos` when no token input is given. `--tokens` and
`--tokens-file` are mutually exclusive. See [KV_PREFETCH.md](KV_PREFETCH.md).

## How to set a flag

A flag can be written in three places.

1. The command line, for a one-shot run or a server start:

   ```sh
   engine/strata --pack pack/full --tokens "1,2,3" --spec 4
   ```

2. The run config's `"args"` list, which the server passes to the engine. The server
   (`serve/server.py`) runs `engine/strata --serve` followed by these elements, so do not repeat
   `--serve`:

   ```json
   {
     "exe": "engine/strata",
     "args": ["--pack", "pack/full", "--max-context", "16384", "--kv", "int8",
              "--expert-cache", "auto"]
   }
   ```

   Write a flag and its value as two elements. A flag with no value is one element. The
   server also adds flags from config keys, for example `--vision`, `--split-skip-if-fits`,
   `--vram-elastic`, and `--expert-profile-save`. See [SERVER.md](SERVER.md).

3. `strata tune`, which measures candidates and writes the result into the run config:

   ```sh
   strata tune run --engine engine/strata --backend cuda --model <model>
   strata tune apply <job-id>
   ```

   See [TERMINAL.md](TERMINAL.md) and [TUNING.md](TUNING.md).

The main engine treats a flag that it does not know as an error (exit `2`), not a warning.
A typo then stops the run instead of silently falling back to the default. The Qwen3.6 engine
in serve mode ignores an unknown flag with a message.
