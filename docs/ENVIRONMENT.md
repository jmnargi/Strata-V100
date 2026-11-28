# Environment variables

This page lists every environment variable that the engine, the server and the `strata`
command read. It gives the default and the effect of each one.

The lists come from the code: `std::getenv` in `src/` and `include/`, and `os.environ` in
`serve/`, `strata_cli/`, `strata_control/`, `tools/` and `setup.py`. There are 617 runtime
names. Section [Compile-time macros](#compile-time-macros) lists 61 names that a build sets,
not a person.

## How settings are read

Four layers set a value. The order is: the request, then the command-line flag, then the
environment, then the default. The first layer that sets a value wins.

| Layer | Where it is read | Example |
|---|---|---|
| request | the `strata_tune` object of an API request body (`strata_control/tune_api.py`) | `"strata_tune": {"spec_min_p": 0.7}` |
| command line | `src/program/generate.cpp`, `serve/server.py` | `--memory-limit-mib 8192` |
| environment | `std::getenv` in the engine, `os.environ` in the server and the terminal | `STRATA_MEMORY_LIMIT_MIB=8192` |
| default | the engine, the server or the terminal | `0` (no limit) |

- The engine reads most variables once, at start or at the first use, into a `static`. A
  change to the environment after that point has no effect.
- Many engine settings have a flag. If the flag is given, the engine does not read the
  variable. `--memory-limit-mib` is one example: the code reads `STRATA_MEMORY_LIMIT_MIB`
  only when the flag value is negative (`src/program/generate.cpp:3049`).
- If the variable is not set, the engine keeps its own default. The tables below give that
  default. `engine default` means the value comes from the device, the pack or a measured
  rule in the code, not from a constant.
- The server passes its own environment to the engine process, then adds the settings of
  the active profile (`serve/server.py:3425`).
- The `strata_tune` object of an API request sets a few of these settings for that request
  only, without a restart. Section [How to set a variable](#how-to-set-a-variable) gives the
  details.

### Architecture tables that set variables for you

Some speed switches are on by default per architecture. The engine writes them into the
environment only if the person did not set them (`src/core/nv_defaults.cpp`,
`src/core/arch_defaults.cpp`). An explicit setting always wins, whatever its value.

| Table | Card | Off switch |
|---|---|---|
| `nv_defaults.cpp` | sm_86 (RTX 3060, RTX A4000) | `STRATA_SM86_PERF=0` |
| `nv_defaults.cpp` | sm_70 (Tesla V100) | `STRATA_SM70_PERF=0` |
| `nv_defaults.cpp` | sm_60 (Tesla P100) | `STRATA_SM60_PERF=0` |
| `arch_defaults.cpp` | gfx1151 (Strix Halo) | `STRATA_GFX1151_DEFAULTS=0` |
| `arch_defaults.cpp` | gfx12 (RDNA4) | `STRATA_GFX12_DEFAULTS=0` |

The engine prints the list it set on stderr at start. The measured gains of these tables
are in [NVIDIA_V100.md](NVIDIA_V100.md), [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md)
and [STRIX_HALO.md](STRIX_HALO.md).

## Settings you are likely to change

| Flag or engine default | Variable | Values | Effect |
|---|---|---|---|
| engine default: `auto` where measured (CUDA, one GPU, no batch slots), else off | `STRATA_PREFILL_CPU_SHARE` | `0`..`1`, `auto`, `0` = off | share of a short chunk's experts the idle CPU pool computes. -20..-35% prompt time at 512 and 1000 tokens on RTX 3060, Tesla P100 and RTX 5070; mean KL 0.004 against off, so the answers can change |
| 3072 (1024 in the measured range) | `STRATA_PREFILL_CPU_SHARE_MAX` | tokens, at least 1024 | chunks below this size are staged so the CPU share can act |
| engine default | `STRATA_PREFILL_RING` | slots | depth of the prompt path's streamed ring |
| off | `STRATA_PREFILL_HOST_PLAN` | `0` turns it off | price the auto prompt chunk against host RAM and drive speed. Advisory: only the chunk shape changes |
| `7` on CUDA and HIP gfx1200/gfx1201 (0 = off); off with NVFP4 experts, with batch slots, with a peer tier, or at a 90% or higher cache hit rate | `STRATA_ROUTE_TAIL_SKIP` | `0` off, `1`..`9` | skip a missed expert whose rank every token of the window is at or above. Changes which expert computes a row |
| off | `STRATA_VERIFY_ECAP` | `0` off, above the top-k | cap the unique experts of a decode window (relaxed MTP, upstream #3545). Changes the output where a dropped low-weight expert flips a draft |
| off | `STRATA_TAIL_SKIP_MAX_CACHED` | percent, 101 = never | share of the expert cache that routing must cover before the tail skip applies |
| engine default | `STRATA_SPLIT_RING` | slots | the layer split's streamed ring, from the stages' residency |
| off unless set | `STRATA_SPLIT_MTP_BATCH` | nonzero = on | the split's draft K/V through the prompt path (Q8_1 x Q8_0 MMQ). The drafts can differ; the target's tokens do not |
| RAM-priced | `STRATA_SM70_CHUNK` | `0` = 8192, `1` = 16384 | prompt chunk ceiling on one Volta card with a native pack in RAM |
| `--kv-grow`: off | `STRATA_KV_GROW` | `1` / `0` | the K/V takes VRAM only for the cells requests reach; needs `--expert-profile`; GPU only |
| 8192 | `STRATA_KV_GROW_STEP` | cells | cells the elastic K/V adds per step |
| 128 | `STRATA_KV_GROW_FLOOR` | cells | cache slots kept below the elastic K/V (tests) |
| 20000 | `STRATA_POOL_SPIN_US` | microseconds | spin before a CPU pool worker sleeps. `0` sleeps at once |
| `--memory-limit-mib`: 0 = none | `STRATA_MEMORY_LIMIT_MIB` | MiB | RAM budget of the container or cgroup; parking and session save/restore keep their floor under it. The flag wins |
| `--host-core`: first | `STRATA_HOST_CORE` | `first`, `last`, `sibling` | core the host thread takes |
| engine default (`off`) | `STRATA_AUX_CPUS` | `off`, `auto`, `LIST` | where the non-pool, non-host threads run |
| off | `STRATA_PIN_BY_NEED` | `1` = on | pin the arena by predicted need instead of by layer prefix. Needs `--expert-profile`; single GPU |
| profile size | `STRATA_PIN_BY_NEED_SKIP` | pairs | forces the count of pairs taken as GPU-resident |
| off | `STRATA_ARENA_MMAP` | `1` = on | map `experts.bin` read-only instead of locking it (Linux). Run with `--pcie-frac 0` |
| 0 | `STRATA_DMA_BATCH` | `0`, `1`, `2` | mode of `cudaMemcpyBatchAsync` for a window's uploads |
| off | `STRATA_PARTIAL_PIN` | `1` = on | pin a partial arena on a card that shares PCIe |
| 24 | `STRATA_PARTIAL_PIN_GIB` | GiB | cap on the arena pinned slice by slice |
| no cap | `STRATA_ARENA_PIN_GIB` | GiB, `auto` | cap on the whole-arena registration |
| 6 | `STRATA_UMA_HEADROOM_GIB` | GiB, 0..1024 | VRAM left free on a unified-memory device |
| 640 on Windows CUDA, else 0 | `STRATA_PIN_RESERVE_MIB` | MiB | soft reserve of the pin budget |
| `--batch-mtp`: off | `STRATA_BATCH_MTP` | nonzero = on | each batch slot verifies one MTP proposal per window |
| `--coupled-draft`: off | `STRATA_SPEC_COUPLED` | nonzero = on | coupled draft sampling: the chain samples over the draft head with the penalty ring. Changes the drafts |
| 0 | `STRATA_SPEC_MIN_TEMP` | temperature | coupled or probabilistic drafting only at or above this temperature |
| off | `STRATA_MMQ_KQ` | `1` = on | load the K-quant prompt kernel library next to the engine |
| 4 (Windows) / 16 (Linux) | `STRATA_IO_THREADS` | 1..64 | issuing threads for direct reads |
| on for a GGUF read in place on Linux | `STRATA_IO_PREFETCH` | `0` / `1` | whole-blob preads and read-ahead of the predicted experts |
| 8 | `STRATA_IO_PF_THREADS` | count | I/O threads of the prefetch |
| 4 (Linux file tier) | `STRATA_COLD_OVERLAP` | `0`..`4` | compute page-cached experts from the mapping |
| 60 | `STRATA_WATCHDOG_S` | seconds, `0` = off | heartbeat limit before the engine ends itself |
| 10x the limit | `STRATA_WATCHDOG_IO_S` | seconds, `0` = none | time allowed while the file tier still makes progress |
| detected drive class | `STRATA_SSD_GBPS` | GB/s | drive speed the prompt planner prices |
| 190 | `STRATA_SPLIT_MISS_MS` | milliseconds | miss cost in the split's traffic fit; raise it on a slow CPU |
| 30000 | `STRATA_SPLIT_WAIT_MS` | milliseconds, `0` = no bound | hand-off wait of the overlapped split |
| 16 | `STRATA_STAGER_RING` | 2..256 | depth of the pinned staging ring |
| 2..4, or 32 for a file set | `STRATA_STAGER_THREADS` | 1..32 | host copy threads of unpinned blobs |
| architecture table | `STRATA_ADAPT_OVERLAP` | `0` / `1` | overlap the adaptive tier's copies with verification |
| architecture table | `STRATA_ADAPT_NOWAIT` | `0` / `1` | do not wait for the tier's copies |
| architecture table | `STRATA_ADAPT_LAG` | windows | how far behind the tier runs |
| off | `STRATA_ROUTE_RESIDENT` | margin in logits | residency-biased routing. Changes the output |
| 6-9 | `STRATA_ROUTE_RESIDENT_RANKS` | `lo-hi` | the ranks the margin applies to |
| off | `STRATA_KV_ROT` | `1` = on | Hadamard rotation over INT8 K/V |
| 2000 | `STRATA_TG_FLUSH_US` | microseconds | driver flush after each token graph |
| off, on for gfx1200/gfx1201 | `STRATA_MTP_HIP_PRIO` | `0` / `1` | draft stream at high priority on HIP |
| server: `""` | `STRATA_API_KEY` | keys | keys the server accepts |
| server: 0 = 256 MiB | `STRATA_MAX_BODY_MIB` | MiB | largest request body read into memory |
| server: 256 | `STRATA_HTTP_BACKLOG` | count, minimum 5 | listen backlog |
| server: 3 | `STRATA_START_RETRIES` | count, `0` = never | retries of a start that fails in the GPU runtime |
| server: 900 | `STRATA_ENGINE_READY_S` | seconds, `0` = for ever | wait for the engine's READY line |
| server: 90 | `STRATA_ENGINE_STALL_S` | seconds, `0` = off | frozen-engine watchdog; needs `psutil` |
| server: 300 / 300 | `STRATA_VISION_READY_S`, `STRATA_VISION_ENCODE_S` | seconds, `0` = for ever | image encoder start and encode timeouts |
| server: 0 (off) | `STRATA_ADMIT_SHORT_FIRST` | tokens | short prompts take the control lines first |
| server: 30 | `STRATA_ADMIT_MAX_WAIT_S` | seconds | a waiter beyond this counts as a short prompt |
| server: config | `STRATA_ALLOWED_HOSTS` | names | extra `Host` names the server answers to (DNS rebinding) |
| off | `STRATA_NO_PRESET` | `1` = on | a tune's own worker measures without the preset |
| on | `STRATA_PARALLEL_SOLO` | `0` = off | keep a solo parallel request in its slot |
| auto | `STRATA_CUDA` | `12`, `13`, `auto` | CUDA toolkit `setup.py` picks |
| auto | `STRATA_ISA_FLOOR` | `avx`, `none` | oldest CPU tier the build keeps |
| auto | `STRATA_GLYPHS` | `ascii`, `unicode` | glyph set of the terminal |

## Every runtime variable

The groups follow the subsystem that reads the variable. `off unless set` means the code
tests for the variable and does nothing when it is absent. `on` means the code treats an
absent variable as enabled. `engine default` means a value the code computes.

### Prefill and the prompt path

| Variable | Default | Description |
|---|---|---|
| `STRATA_PREFILL_AUTO_MAX` | engine default (8192; 16384 on sm_70 when RAM allows) | ceiling of an `auto` prompt chunk, never past the context |
| `STRATA_PREFILL_BF16X2` | off unless set | add each activation's BF16 remainder as a second pass |
| `STRATA_PREFILL_CPU_SHARE` | `auto` where measured, else off | share `0`..`1` of a short chunk's experts the CPU pool computes |
| `STRATA_PREFILL_CPU_SHARE_MAX` | 3072 (1024 in the measured range) | chunks below this size are staged for the CPU share |
| `STRATA_PREFILL_DUMP_R` | off unless set | write the final residuals, every 64th row, to the named file |
| `STRATA_PREFILL_DUMP_R_ALL` | off unless set | write every row of the final residuals |
| `STRATA_PREFILL_EQUAL` | off unless set | even out the chunk count for `--prefill-pipe 1` |
| `STRATA_PREFILL_FP16TC` | on | tensor-core prompt GEMMs in FP16; `0` keeps the MMQ reference |
| `STRATA_PREFILL_GROUP_GATHER` | on | one gather and one record per expert group; `0` = per expert |
| `STRATA_PREFILL_HELP` | off | lend the idle stage's GPU the prompt rows (layer split) |
| `STRATA_PREFILL_HELPER_SOURCE` | off | the prompt can take a blob a helper's VRAM cache already holds |
| `STRATA_PREFILL_HELP_FRAC` | engine default | share of the prompt the helper stage takes, for every chunk size |
| `STRATA_PREFILL_HELP_MIN` | engine default (the stream-all floor) | the helper's own floor in tokens |
| `STRATA_PREFILL_HOST_PLAN` | on | price the auto chunk against RAM and drive speed; `0` turns it off |
| `STRATA_PREFILL_ISSUER` | on | copies issued by a dedicated issuer; `0` issues inline |
| `STRATA_PREFILL_LEND_PCT` | engine default | percent of the expert cache the prompt can borrow |
| `STRATA_PREFILL_LOAN_EXACT` | engine default | exact loan arithmetic instead of the rounded rule |
| `STRATA_PREFILL_MMQ` | engine default (built) | quantized matrix-multiply prompt kernels; `0` takes the FP16 path |
| `STRATA_PREFILL_NVFP4` | off unless set | `w4a4`: NVFP4 prompt MMQ with FP4 activations (Blackwell) |
| `STRATA_PREFILL_PIPE` | off unless set | a layer split reads the prompt as a pipeline (same as `--prefill-pipe`) |
| `STRATA_PREFILL_PLAN_TOKS` | engine default | prompt compute rate, tokens/s, that the host plan prices |
| `STRATA_PREFILL_PROMOTE` | engine default | the prompt's routing counts for this part's own layers |
| `STRATA_PREFILL_REFILL_OVERLAP` | engine default | a split's non-last stage refills while it waits for the next chunk |
| `STRATA_PREFILL_RING` | engine default | slots in the prompt path's streamed ring |
| `STRATA_PREFILL_STREAM_AHEAD` | on | upload every expert ahead of use; `0` keeps the routed-only schedule |
| `STRATA_PREFILL_STREAM_MIN` | 1024 (2048 on the tensor-core path) | chunk size from which every expert streams |
| `STRATA_PREFILL_TIMING` | off | print the prompt path's GPU time by phase |
| `STRATA_PREFILL_WARM` | on | warm the prompt path at start; `0` turns it off |
| `STRATA_GROUP_COPY` | off unless set | group the staging copies into batches (A/B arm) |
| `STRATA_HANDOFF_PIPE` | off (unset, `0` and `1` = one copy) | `2`..`16` row pieces a split hands a chunk on in |
| `STRATA_HANDOFF_PIPE_WORKER` | on | a worker per row piece; `0` = a thread per chunk |
| `STRATA_EMB_REUSE_ACCOUNT` | off | let the planner count the bytes `bo` shares with `emb` |
| `STRATA_RING_BYTES` | on | price the ring by bytes; `0` keeps 0.1.39's slot list |
| `STRATA_STAGER_BATCH` | 1 (8 for a file set) | blobs a stager thread claims in one request |
| `STRATA_STAGER_RING` | 16 (4 x threads for a file set) | depth of the pinned staging ring, 2..256 |
| `STRATA_STAGER_SLEEP` | on | sleep when no copy is queued; `1` / `0` forces either |
| `STRATA_STAGER_SSD` | engine default (from the page-cache sample) | `1` / `0` forces the SSD or the RAM copy profile |
| `STRATA_STAGER_THREADS` | 2..4, or 32 for a file set | host copy threads of unpinned blobs, 1..32 |
| `STRATA_AUX_STAGER` | on | copy threads follow `--aux-cpus`; `0` leaves them where they were |
| `STRATA_COLD_TRACE` | off | print where the first prompt's cold reads go |
| `STRATA_KV_HOST_DMA` | off | append to the staging pool and resident slots, then DMA the block |
| `STRATA_F16_RANGE` | off unless set | clamp the values that reach FP16's range |
| `STRATA_XN_F16` | on | keep a pure FP16 image; `0` keeps the BF16 one |
| `STRATA_PROMPT_DIRECT` | off | unbuffered batched prompt reads while the read exceeds the page cache |
| `STRATA_INDEXER_PER_TOKEN` | off unless set | append the indexer's keys per token (old path) |
| `STRATA_IDX_FP16_CHECK` | off unless set | check whether FP16 pooled indexer keys select the same cells |

### Prefill kernels and tiles

| Variable | Default | Description |
|---|---|---|
| `STRATA_PF_BK64` | on | BK 64 tile kernel for the wide shapes; `0` takes BK 32 |
| `STRATA_PF_F16` | off unless set | FP16 expert prompt path (needs `STRATA_PF_FUSED_KQ=1`) |
| `STRATA_PF_F16_BN` | engine default | `64` or `128` forces the FP16 kernel's block count |
| `STRATA_PF_FUSED` | off unless set | launch a layer's streamed experts together (fewer launches) |
| `STRATA_PF_FUSED_KQ` | off unless set | fused K-quant prompt experts |
| `STRATA_PF_FUSED_NATIVE` | on | native packs keep MMQ under `STRATA_PF_FUSED=1`; `0` drops it |
| `STRATA_PF_FUSED_SM75` | off | Turing fused experts filled by plain copies instead of cp.async |
| `STRATA_PF_FUSED_TILE` | engine default | `64` or `128` forces the fused prompt tile |
| `STRATA_PF_GEMM` | off unless set | FP16 GEMM for the two K 6144 prompt projections |
| `STRATA_PF_GROUP_CAP` | engine default | cap on the experts in one prompt group |
| `STRATA_PF_GROUP_SCRATCH` | engine default | scratch bytes per prompt group |
| `STRATA_PF_HCDOWN` | off | fused hyper-connection down projection on large chunks |
| `STRATA_PF_HC_REGION` | off | one scratch region for the hyper-connection read |
| `STRATA_PF_OCC` | engine default | blocks per WGP for the fused experts (experiment knob) |
| `STRATA_PF_OCC_W` | engine default | the same for every occupancy variant (experiment knob) |
| `STRATA_PF_PAD` | off | padded GEMM row strides; acts only with `STRATA_PF_GEMM` |
| `STRATA_PF_PEER_COMPACT` | on | layer-sized peer buffers; `0` takes the A/B form |
| `STRATA_PF_PEER_F16` | on | the peer's sums mode transfers in FP16; `0` = FP32 |
| `STRATA_PF_PEER_GATHER` | on | one launch per group for the peer's sums; `0` = per expert |
| `STRATA_PF_PEER_HOST` | on | allow the host route for the peer's prompt rows; `0` refuses |
| `STRATA_PF_PEER_OUT_PIPE` | on | pipelined output copies; `0` = one copy after the last group |
| `STRATA_PF_PEER_RING` | engine default | ring slots of the peer prompt stream |
| `STRATA_PF_PEER_STREAM` | engine default (share, 0 = off) | share of the prompt the peer stage takes |
| `STRATA_PF_PEER_SUMS` | on (host route) | the peer adds rows into one weighted sum; `0` = rows |
| `STRATA_PF_PEER_XSTREAM` | on | the host route's copies on their own stream; `0` = the compute stream |
| `STRATA_PF_SKIP_UNROUTED` | off | skip a pinned blob whose routing did not reach it (WMMA walk) |
| `STRATA_PF_STEP_SYNC` | off | wait for both streams after each prompt step (diagnosis) |
| `STRATA_PF_SWITCH_MIN_T` | 0 (off) | chunk size from which the rounding-level prompt switches act |
| `STRATA_PF_WMMA` | off | WMMA prompt path with the tensor-core expert tiles. On a dual V100-PCIE-16GB with UD-IQ3_XXS (IQ2_S/IQ3_XXS gate/up, IQ4_NL down) it reads prompts **8.6-20 % faster** than MMQ's dp4a from 2K to 64K tokens (A-B-A-B, [NVIDIA_V100.md](NVIDIA_V100.md#the-fp16-prompt-expert-path-on-ud-iq3_xxs-2026-11-18)); it changes the prompt's FP16 activations, so a reply can differ in its last bits |
| `STRATA_PF_WMMA2` | architecture table | the second prompt-expert tile; `0` keeps the first tile only |
| `STRATA_PF_WMMA2_N64` | off | alternate tile orientation, half the weight rows per block |
| `STRATA_PF_WMMA_LEAN` | off | lean buffers on the direct WMMA plan: it stops allocating the MMQ-only prompt buffers the direct plan never reads. With `STRATA_PF_WMMA=1` on a dual V100-PCIE-16GB / UD-IQ3_XXS it read prompts **+4.0 % at 8K, +2.2 % at 32K, +1.7 % at 64K** (A-B-A-B, [NVIDIA_V100.md](NVIDIA_V100.md#the-fp16-prompt-expert-path-on-ud-iq3_xxs-2026-11-18)) |
| `STRATA_PF_XMX` | off | grouped FP16 XMX GEMM for the A770 prompt path |
| `STRATA_PF_XMX_GU_MAX` | 250 | mean rows per expert below which gate/up runs on XMX |

### Expert routing and the adaptive tier

| Variable | Default | Description |
|---|---|---|
| `STRATA_ADAPT_ASYNC_BATCH` | 0 | copies in flight per tick for the async adaptive tier |
| `STRATA_ADAPT_ASYNC_WAIT` | 1 | wait for the tier's copies instead of polling |
| `STRATA_ADAPT_CALLBACK_WARMUP` | on | warm the tier's callback threads; `0` turns it off |
| `STRATA_ADAPT_CHECKRES` | off | check the tier's resident table in serial windows |
| `STRATA_ADAPT_EVICT_SYNC` | on | wait for the evicted copies; `0` does not |
| `STRATA_ADAPT_JOB_CPU` | engine default | logical CPU (or `-1` for none) the tier's job thread pins to |
| `STRATA_ADAPT_LAG` | architecture table (`2` on sm_86) | verify windows the tier runs behind |
| `STRATA_ADAPT_NOWAIT` | architecture table | the tier does not wait for its copies |
| `STRATA_ADAPT_OVERLAP` | architecture table (Volta on Linux: `1`) | overlap the tier's copies with verification |
| `STRATA_ADAPT_OVERLAP_RESIDENT` | unset | `1` allows the overlapped tier in the resident RAM mode; unset refuses it |
| `STRATA_ADAPT_OVERLAP_SLOW_US` | 0 (off) | test hook: make the link slow so the windows wait |
| `STRATA_ADAPT_PICK_INLINE` | 1 | the tier's choice runs inline in the tick; `0` moves it off the thread |
| `STRATA_ADAPT_SIGNAL` | engine default | `copy` asks for the copy's arrival rather than the step's |
| `STRATA_DISJOINT_ADAPT` | off | a helper GPU's expert stays out of the primary's adaptive tier |
| `STRATA_ROUTE_TAIL_SKIP` | 7 on CUDA; 0 (off) elsewhere and with NVFP4 experts | rank at or below which a missed expert that every window token routes that low is skipped |
| `STRATA_TAIL_SKIP_MAX_CACHED` | 90 | percent of the cache routing must cover before the tail skip moves; `101` = never |
| `STRATA_ROUTE_RESIDENT` | off (value = margin in logits) | residency-biased routing. Changes the answer |
| `STRATA_ROUTE_RESIDENT_RANKS` | 6-9 | rank range the residency margin applies to |
| `STRATA_ROUTE_RESIDENT_MTP` | off | the same bias in the MTP draft layer (experimental, changes answers) |
| `STRATA_ROUTE_PRIOR` | 0 (off) | logit bonus for the previous token's experts when the top-10 is picked. Changes the output |
| `STRATA_ROUTE_PER_TOKEN` | on | route each row on its own; the variable forces the per-token table |
| `STRATA_ROUTE_PROJ_PER_TOKEN` | off | project the router input per token instead of per window |
| `STRATA_ROUTER_BLOCK` | off | force the block router on packs where the single-block path fits |
| `STRATA_HIP_ROUTER_FAST` | off | the fast HIP router (measured per card before it becomes a default) |
| `STRATA_HIP_ROUTER_OLD` | off | the old HIP router kernel |
| `STRATA_LOOKAHEAD` | on | predict the next layers' experts and warm their pages; `0` turns it off |
| `STRATA_LOOKAHEAD_K` | 10 | experts per token the predictor asks for |
| `STRATA_LOOKAHEAD_STATS` | off | score the predictor against the real routing (measurement only) |
| `STRATA_RAM_ADAPT` | 0 (off) | promotions staged per window on the RAM tier |
| `STRATA_RAM_ADAPT2` | off | staged promotions with the second queue |
| `STRATA_RAM_ADAPT_CACHED_W` | engine default | weight of the cached share in the RAM adaptor |
| `STRATA_RAM_ADAPT_PER_LAYER` | engine default | layers the RAM adaptor plans at once |
| `STRATA_RAM_ADAPT_PROMPT_W` | engine default | weight of the prompt's traffic in the RAM adaptor |
| `STRATA_RAM_ADAPT_QUEUE` | engine default | queue depth of the RAM tier's promotions |
| `STRATA_RAM_ADAPT_SKIP_TOP` | engine default | top-ranked pairs the RAM adaptor leaves alone |
| `STRATA_MMQ_DIRECT` | off | resident experts read in their cache slots by the pointer-table MMQ |
| `STRATA_MMQ_GROUP_SORT` | engine default | sort the 16-expert products of both directions per MMQ layer |
| `STRATA_MMQ_RESIDENT_SORT_NE` | engine default | the same sort without the product split |
| `STRATA_MMQ_DIRECT_LOG` | off | print each stage's MMQ totals (a check) |
| `STRATA_GROUPED_PAIR_MIN_HITS` | engine default | fewest hits that justify the paired grouped kernels |

### VRAM slots, PCIe and the file tier

| Variable | Default | Description |
|---|---|---|
| `STRATA_PCIE_AUTO_CANON` | off | canonicalize the PCIe window automatically (opt-in on Windows) |
| `STRATA_PCIE_AUTO_SPLIT` | off | refine a layer split's shares from each stage's probed link and the pool's measured speed (CUDA and HIP); a later stage's share is recomputed whenever a reading lands, even when the first stage's own share does not move |
| `STRATA_PCIE_BALANCE` | off | take the expert count from measured CPU and GPU costs |
| `STRATA_PCIE_FRAC_DEFAULT` | engine default (share form) | default PCIe share of the expert cache; `auto` refines it |
| `STRATA_PCIE_MIN1` | off | give 1-2 misses per layer at least one slot on the PCIe lane |
| `STRATA_PEER_DIRECT` | on | the peer writes its rows directly into the output; `0` copies |
| `STRATA_PEER_HOT` | 0 (off) | share of the HOT pairs the peer card takes |
| `STRATA_PEER_HOT_AT` | 8700 | pair rank from which the peer's share is counted |
| `STRATA_PEER_NO_P2P` | off | test knob: pretend the GPU pair has no peer-to-peer link |
| `STRATA_REMOTE_SPIN` | engine default | spin value of the remote experts' copies; `0` keeps the driver default |
| `STRATA_REMOTE_ZEROCOPY` | on | remote experts read the host buffer directly; `0` copies |
| `STRATA_ARENA_MMAP` | off | map `experts.bin` read-only instead of locking it (Linux only) |
| `STRATA_ARENA_PIN_GIB` | no cap | GiB cap on the whole-arena registration; `auto` uses the shared-memory budget |
| `STRATA_ARENA_LOCK` | engine default | lock mode of the arena's pinned memory |
| `STRATA_ARENA_RELEASE` | on | hand back the arena pages a GPU holds; `0` keeps them |
| `STRATA_ARENA_THP` | off | ask for transparent huge pages on the arena |
| `STRATA_NO_ARENA_THP` | off | skip the huge-page request (a 25 s start once became 432 s) |
| `STRATA_NO_LARGEPAGES` | off | skip the large-page attempt (A/B, same run and boot) |
| `STRATA_PARTIAL_PIN` | off | pin a partial arena when the card shares PCIe |
| `STRATA_PARTIAL_PIN_GIB` | 24 | GiB cap of the sliced registration |
| `STRATA_PIN_BY_NEED` | off | pin by predicted need rather than by layer prefix |
| `STRATA_PIN_BY_NEED_SKIP` | profile size | forces the resident-pair count |
| `STRATA_PIN_GUARD` | engine default | `0` locks the pages in one go, `2` always registers in steps (a test) |
| `STRATA_PIN_RESERVE_GIB` | engine default | GiB kept back from the pin budget |
| `STRATA_PIN_RESERVE_MIB` | 640 on Windows CUDA, else 0 | MiB reserve of the soft pin budget |
| `STRATA_RESIDENT_HEADROOM_GIB` | engine default (RAM left free at start) | RAM the resident copy must leave free |
| `STRATA_RESIDENT_PIN` | on | pin the resident copy; `0` keeps it pageable |
| `STRATA_STAGE_PIN` | engine default (off) | pin the staging buffers; `1` = on, `0` = off |
| `STRATA_DEFERRED_REGISTER` | off | defer the arena registration to its first use (RTX 5070, Windows) |
| `STRATA_UNBUFFERED_LOAD` | Windows: automatic; Linux: off | `1` forces unbuffered loading. On Linux this applies to an aligned, packed `experts.bin` arena; unsupported or unaligned reads use the file cache. `0` forces buffered loading. |
| `STRATA_STARTUP_UNBUFFERED` | on | read the startup fill unbuffered; `0` reads through the mapping |
| `STRATA_DROP_CACHE_ON_EXIT` | off | release the model's file pages when the engine stops |
| `STRATA_FILE_RELEASE` | engine default | release a file's pages after its last read |
| `STRATA_FILE_CACHE_OLD_RULE` | engine default | `1` / `0` forces the old page-cache rule |
| `STRATA_KEEP_MAPPING` | off | keep the read-only mapping while reading unbuffered (A/B) |
| `STRATA_DEC_BATCH` | on | batch the decode's file reads; `0` reads one at a time |
| `STRATA_EXCHANGE_ROTATE` | engine default | rotate the exchange buffer between windows |
| `STRATA_FETCH_PVM` | on | prefetch cached pages with `PrefetchVirtualMemory`; `0` is the A/B arm |
| `STRATA_FETCH_THREADS` | engine default | threads that fetch the GPU cache's fill |
| `STRATA_IO_PREFETCH` | on for a GGUF read in place on Linux | whole-blob preads plus read-ahead of predicted experts |
| `STRATA_IO_PF_AHEAD` | on | read the predicted experts ahead; `0` turns it off |
| `STRATA_IO_PF_BUDGET_FRAC` | engine default | multiplier on the bytes the workers can read in one window |
| `STRATA_IO_PF_BUDGET_MB` | engine default (off) | fixed speculative byte budget for a decode window |
| `STRATA_IO_PF_GATE` | on | gate the decode read-ahead on page-cache room and gain |
| `STRATA_IO_PF_GATE_HI` | engine default | upper gate threshold |
| `STRATA_IO_PF_GATE_LO` | engine default | lower gate threshold |
| `STRATA_IO_PF_GATE_USEFUL` | engine default | smallest share of the read-ahead that must be used |
| `STRATA_IO_PF_MINCONF` | engine default | smallest prediction confidence the scheduler admits |
| `STRATA_IO_PF_SCHED` | off | deadline, confidence and byte-budget scheduling of the read-ahead |
| `STRATA_IO_PF_STAGE` | off | stage the experts from the workers, kernels read the mapping |
| `STRATA_IO_PF_TARGET` | engine default | page-cache target the budget aims at |
| `STRATA_IO_PF_THREADS` | 8 | I/O threads of the prefetch |
| `STRATA_IO_PREFETCH_DEPTH` | engine default | layers of read-ahead in the decode |
| `STRATA_IO_STATS` | off | print what the page cache held when the kernels got the blob |
| `STRATA_IO_THREADS` | 4 (Windows) / 16 (Linux) | issuing threads for direct reads, 1..64 |
| `STRATA_FS_ADMIT` | 1 | admit a slot to the file store's admission set |
| `STRATA_FS_AHEAD` | on | feed the swap space's predictions from the file store |
| `STRATA_FS_BUDGET` | 32 | file store's byte budget, in MiB |
| `STRATA_FS_DEPTH` | 4 | depth of the file store's read queue |
| `STRATA_FS_SLOTS` | engine default | slots of the file store |
| `STRATA_FS_VERIFY` | 0 | verify the file store's reads |
| `STRATA_SSD_GBPS` | detected class (NVMe 2.5, SATA 0.45, rotational 0.15, else 1.5) | GB/s the prompt planner prices |
| `STRATA_SSD_KEEPALIVE` | 100 | ms without a read before one page is read anyway; `0` = off |
| `STRATA_SSD_KEEPALIVE_WINDOW` | 60 | seconds after the last row request that the keepalive goes on |
| `STRATA_READ_AHEAD` | engine default | read-ahead distance of the memory-mapped arena |
| `STRATA_FILL_AHEAD` | on | advise the next pairs so their mapped reads overlap; `0` off |

### KV cache and attention

| Variable | Default | Description |
|---|---|---|
| `STRATA_KV_GROW` | off | the K/V takes VRAM only for the cells requests reach (same as `--kv-grow`) |
| `STRATA_KV_GROW_STEP` | 8192 | cells the elastic K/V adds per step |
| `STRATA_KV_GROW_FLOOR` | 128 | cache slots the elastic K/V keeps; tests set it |
| `STRATA_KV_GROW_INIT` | engine default | cells the elastic K/V allocates at start |
| `STRATA_KV_HOST_REGISTER` | engine default | `1` registers the host buffer first, `0` uses `cudaHostAlloc` first |
| `STRATA_KV_PREFETCH` | off | stage the next streamed layer's KV prefix on its own stream |
| `STRATA_KV_MAIN_OFF` | off | A/B: keep the main layers out of the resident draft ring |
| `STRATA_KV_RING_OFF` | off | A/B: keep the drafter fully resident |
| `STRATA_KV_ROT` | off | Hadamard rotation over the INT8 K/V |
| `STRATA_KV_STAGE_OWN` | off | A/B: the staging pool gets its own allocation |
| `STRATA_ATTN_DAILY` | architecture table | the daily decode attention kernel; part of the sm_86, sm_70 and sm_60 tables |
| `STRATA_ATTN_LANECELL` | architecture table (gfx1151: `1`) | decode attention scores, one cell per thread |
| `STRATA_ATTN_MERGE_V2` | engine default | `1` / `0` picks the merge kernel; unset follows the device |
| `STRATA_ATTN_PRE75` | on | chunk attention kernel; `0` turns it off (A/B) |
| `STRATA_ATTN_V3` | engine default | `0` keeps the previous chunk attention kernel |
| `STRATA_QSA_CLUSTER` | on | the cluster form of the QSA selection; `0` takes the old paths |
| `STRATA_QSA_ROWS` | architecture table (sm_86, sm_70: `1`) | the row form of the QSA selection |
| `STRATA_QSA_SLOTS` | off | in a batch window, one launch for scores, top-k and attention |
| `STRATA_QSA_SPLIT_ATTN` | off | the draft layer's window attention split the same way |
| `STRATA_QSA_WARP` | off | A/B arm: `1` or `select` runs the pre-sm_80 kernels on any card |
| `STRATA_QSA_EARLY_EXIT` | on | load the queries first; `0` keeps the old staging |
| `STRATA_QSA_DUMP` | off | append every QSA layer's selected cells to the named file |
| `STRATA_QSA_DUMP_LAST` | 4096 | positions of the prompt that the QSA dump covers |
| `STRATA_QSA_DUMP_LAYER` | -1 (all layers) | dump one QSA layer only |
| `STRATA_QSA_DUMP_STEPS` | off | end each dump record with the step data |
| `STRATA_TOPK_ACTIVE_ANY` | off | test arm: let any active cell win the top-k |
| `STRATA_TOPK_CAPACITY_GUARD` | off | guard the top-k capacity check |
| `STRATA_TOPK_COMPACT` | engine default | compact form of the top-k output; `0` takes the wide form |
| `STRATA_TOPK_OLD` | off | the original top-k kernel |
| `STRATA_TOPK_REG_MIN_BLOCKS` | 4096 (gfx1151 table) | fewest blocks for the register top-k |
| `STRATA_TOPK_SAMPLE` | on | sample inside the wide top-k kernel; `0` keeps the wide kernel alone |
| `STRATA_TOPK_SPLIT` | on | split top-k; `0` keeps the wide kernel |
| `STRATA_TOPK_STREAM` | on | the coalesced streaming top-k above ~90K cells; `0` = as before |
| `STRATA_SCORES_HALVE` | engine default | half-precision block scores; `0` keeps the old kernel |
| `STRATA_SCORES_MULTI` | on | the multi-block score kernel; `0` takes the old path |
| `STRATA_SCORES_PROMPT_HALVE` | engine default | half form of the prompt block scores |
| `STRATA_SCORES_THREAD` | engine default | per-thread score branch of the gfx1100 kernel |
| `STRATA_SCORES_THREAD_TRACE` | off | print the score branch the kernel took |
| `STRATA_SELECT_OLD` | off | the old warp selection kernel |
| `STRATA_SELECT_SIMT` | engine default | `1` / `2` force the two SIMT selection forms |
| `STRATA_SELECT_WMMA` | engine default (on where measured) | WMMA form of the selection; `0` keeps the old kernel |
| `STRATA_SEL_OVERLAP` | off | debug: how much neighbouring queries' selections share |
| `STRATA_SIMT2_TILE` | 1 | `2` takes 64 x 64 tiles (measured slower) |
| `STRATA_PROMPT_ATTN_IMMA` | off | the INT8 MMA prompt attention kernel |
| `STRATA_PROMPT_ATTN_OLD` | off | the old prompt attention kernel (Q4_0 KV) |
| `STRATA_PROMPT_ATTN_Q4` | on | the Q4_0 KV prompt attention kernel; `0` takes the old one |
| `STRATA_PROMPT_ATTN_Q4_TC` | off | tensor-core form of the Q4_0 prompt attention |
| `STRATA_PROMPT_ATTN_TQ` | engine default | the TurboQuant prompt attention kernel for the KV type |
| `STRATA_PROMPT_ATTN_V1` | off | debug: the first prompt attention version |
| `STRATA_PA_FAST` | off | the fast prompt attention path |
| `STRATA_PA_V1_PREFETCH` | engine default | `0` / `1` forces off / on for the v1 prefetch |
| `STRATA_PA_V2T` | off | Turing: v2 with register staging instead of cp.async |
| `STRATA_HIP_WMMA` | auto (gfx12: on) | matrix-core prompt attention; `1` forces it, `0` keeps the FP32 kernel |
| `STRATA_HIP_WMMA_MIN_CAP` | 0 | cell count above which the matrix-core attention runs |
| `STRATA_HIP_WMMA_MIN_Q` | 64 | queries above which the matrix-core attention runs |
| `STRATA_HIP_PROMPT_F16` | off | the stage's 16-bit GEMMs run FP16 in and out |
| `STRATA_HIP_SWIGLU_FUSED` | off | the fused HIP SwiGLU kernel (bitwise the same) |
| `STRATA_HIP_VMM` | on | virtual memory management on HIP; `0` turns it off, so `--kv-grow` stays off |
| `STRATA_HIP_ADAPT_KERNEL_COPY` | off | an adaptive swap's host-to-device copy as a kernel on the refill stream |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE` | off | query swizzle of the gfx906 decode attention |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE` | off | print the gfx906 query swizzle choice |
| `STRATA_GFX906_ATTN_REDUCE12` | engine default | the 12-lane reduction of the gfx906 attention |
| `STRATA_GFX906_ATTN_REDUCE12_TRACE` | off | print the gfx906 reduction choice |
| `STRATA_Q8_GATHER_WIDE` | on | the wide Q8 gather on cards below sm_75; `0` turns it off |

### Speculation and MTP

| Variable | Default | Description |
|---|---|---|
| `STRATA_SPEC_COUPLED` | off | coupled draft sampling: the chain samples over the draft head with the penalty ring |
| `STRATA_SPEC_PROB` | off | probabilistic draft acceptance; switches the coupled machinery on |
| `STRATA_SPEC_PROB_GATE` | engine default | gate on the draft's top probability instead of the ratio |
| `STRATA_SPEC_PROB_DT` | engine default | scale of the draft temperature in the probabilistic mode |
| `STRATA_SPEC_MIN_TEMP` | 0 (every request) | coupled or probabilistic drafting only at or above this temperature |
| `STRATA_SPEC_DEPTH` | off | report MTP drafts tested and accepted by depth |
| `STRATA_SPEC_GUMBEL` | engine default | Gumbel-max noise in the sampler path |
| `STRATA_SPLIT_MTP_BATCH` | off | the split's draft K/V through the prompt path (MMQ) |
| `STRATA_MTP_BATCH` | on | the drafter's K/V batched through the prompt path; `0` turns it off |
| `STRATA_MTP_BATCH_F16` | off | the batch's GEMMs in FP16 (the A/B) |
| `STRATA_MTP_BATCH_RING` | on | the batch's K/V in the ring; `0` keeps it out |
| `STRATA_MTP_CATCHUP_ALL` | off | catch the drafter's K/V up for the whole window, rejected rows included |
| `STRATA_MTP_FULL_HEAD` | off | diagnostic: draft over the whole vocabulary |
| `STRATA_MTP_HIP_PRIO` | on for gfx1200/gfx1201 | the draft stream at high priority on HIP |
| `STRATA_MTP_PREFILL_SYNC` | off | loop synchronisation in the prefill capture |
| `STRATA_MTP_PRIORITY` | on | the draft stream gets the highest priority on stage 0; `0` keeps the default |
| `STRATA_MTP_PROFILE` | off | GPU stamps between the draft layer's stages (measurement only) |
| `STRATA_MTP_SHARED_BRANCH` | on (CUDA), off (HIP) | the shared expert's branch joined before the combine; `0` = one stream |
| `STRATA_MTP_SIDE_STREAM` | engine default (gfx103x only) | `1` / `0` forces the side stream |
| `STRATA_MTP_STAGE0` | engine default | `0` / `1` decides whether stage 0 drafts |
| `STRATA_MTP_TOP2` | off | two draft candidates per step |
| `STRATA_HEAD_MIX_MULTI` | off | HIP: multi-row head mix |
| `STRATA_FUSE_HEAD_GR` | engine default | fused head with the GR read; `0` separates them |
| `STRATA_FUSE_HITS` | architecture table | `moe_hit_add` folded into the combine; `0` / `1` forces off / on |
| `STRATA_FUSE_WAIT` | architecture table | fused wait form of the decode |
| `STRATA_FUSE_DECODE` | architecture table | fused decode kernels |
| `STRATA_FUSE_CONVAB` | off | GDN conv with the a/b projections in one launch |
| `STRATA_SIDE_MIXER` | architecture table | `0`..`2` mixer form; part of the sm_86, sm_70 and sm_60 tables |
| `STRATA_COMMIT_BATCH` | on | one barrier for every GDN layer's commit; `0` keeps the per-layer loop |
| `STRATA_COMMIT_SYNC` | off | forces the commit path to stay synchronous |
| `STRATA_ONE_TOKEN_COMMIT` | engine default | HIP keeps the commit graph after every window; `0` does too |
| `STRATA_VERIFY_ASYNC_LAUNCH` | engine default | asynchronous launch of the verify graph |
| `STRATA_VERIFY_ALL_RESIDENT` | engine default | the window's all-resident graph |
| `STRATA_VERIFY_RESIDENT` | engine default (on since 0.1.39) | resident form of the verify window |
| `STRATA_VERIFY_COHERENT` | off | handshake words and rows in explicitly coherent host memory |
| `STRATA_VERIFY_QDEDUP` | on | deduplicate the window's quantizations |
| `STRATA_VERIFY_ECAP` | 0 (off) | cap the unique experts of a decode window (relaxed MTP) |
| `STRATA_VERIFY_DEVICE_PLAN` | off | build the window's plan on the device |
| `STRATA_VERIFY_COMPLEMENT` | off | check the resident complement against the window |
| `STRATA_VERIFY_ARENA` | off | tests: checksum the loaded arena |
| `STRATA_VERIFY_NODES` | off | list the draft graphs' nodes |
| `STRATA_WAVE_MMVQ` | engine default | the wave form of the mmvq kernel |
| `STRATA_MULTI_BLOCK_ARGMAX` | engine default | the multi-block argmax of a window |
| `STRATA_ARGMAX_MULTI` | engine default | multi-row argmax; `0` takes the scalar path |
| `STRATA_ARGMAX_ROWS_SM70` | architecture table (sm_70: `1`) | the row argmax on Volta |
| `STRATA_DOORBELL_STORE` | off | the GPU stores each step's ring instead of read-modify-writing it over PCIe |
| `STRATA_CKPT_POOL` | off | the serve loop keeps the running-state vectors of parked conversations |
| `STRATA_CKPT_REREAD` | off | read the checkpoint's prefix again instead of restoring it |
| `STRATA_CKPT_THREADS` | off | a split's checkpoint saves every stage's part at once |
| `STRATA_SNAPSHOT_FULL_CAPTURE` | engine default | capture the full graph for a snapshot |
| `STRATA_SNAPSHOT_VERIFY` | off | verify the snapshot against a replayed window |
| `STRATA_BATCH_ATTN_ROWS` | off | the window's attention rows in one launch per step |
| `STRATA_BATCH_GDN_GROUPS` | off | all batch slots' GDN in one launch |
| `STRATA_BATCH_SELF_COMMIT` | off | a batch window keeps its rows when each is another slot's token |
| `STRATA_PIPED_CONCURRENT` | off | a prompt runs on the GPU while the batch windows run |
| `STRATA_PIPED_CONCURRENT_PRIO` | on | high priority for that prompt; `0` keeps the default priority |
| `STRATA_SH_FORK_LATE` | engine default | fork the shared-expert stream late; `0` forks at the top |
| `STRATA_SH_STREAM` | architecture table (gfx1151: `1`) | the shared expert on its own stream |
| `STRATA_COPY_HIP_PRIO` | off | the copy stream at high priority on HIP |
| `STRATA_NAN_GUARD` | on | the non-finite guard in the verify graph; `0` turns it off |
| `STRATA_NAN_INJECT` | off | debug: windows the guard reports as non-finite |
| `STRATA_DUMMY_NODES` | 0 | measurement: empty kernel nodes per layer, to price one node |
| `STRATA_TEST_VERIFY_STALL` | off | test hook: withhold the last layer's flag in the N-th window |
| `STRATA_TEST_HANDOFF_DROP` | off | test hook: the writing stage's N-th window leaves its flag down |
| `STRATA_DF_BRANCH` | off | the mixer's side branches |
| `STRATA_DF_PDL` | off | programmatic dependent launch in the window (sm_90+) |
| `STRATA_DF_PLE` | off | the window's post-projection steps in one pass |
| `STRATA_DF_QB_Q4` | on since 0.1.45 | Q4_0 KV batch path; `0` gives the token-by-token path |
| `STRATA_QIL` | architecture table (sm_70) | the interleaved layout of the decode kernels |
| `STRATA_LFUSE` | off | fewer launches around the shared expert and the KV append |
| `STRATA_LFUSE_GATE` | on | the gate half of that fusion; `0` off |
| `STRATA_LFUSE_PAIR` | on | the paired half of that fusion; `0` off |
| `STRATA_QFUSE` | architecture table (gfx1151) | activation q8_1 images written by their producers |
| `STRATA_QFUSE_HC` | architecture table (gfx12) | only the hyper-connection read writes its consumer's image |
| `STRATA_QFUSE_HC_LOCAL` | on | the 32-column up kernel under `STRATA_QFUSE_HC`; `0` restores the atomic-tail kernel |
| `STRATA_HC_V100` | off | `experimental`, sm_70 only. `1` reads the hyper-connection down matrix once per window (K-split). Rounding changes can produce different greedy answers. |
| `STRATA_HC_V100_FUSED` | off | under `STRATA_HC_V100=1`: `1` runs down and up in one launch for windows of 6 tokens or fewer. It ends in a grid-wide spin that needs all 160 blocks resident, so it can hang a window when another stream runs on the same card (the MTP drafter, the forked shared expert); `0` keeps the two launches, which have no such barrier |
| `STRATA_MMVQ_IL` | on | the interleaved mmvq kernels; `0` keeps native_mmvq's multi-column kernels |
| `STRATA_MMVQ_IL_ROWS` | engine default | row table of the interleaved mmvq |
| `STRATA_MMVQ_IL_120` | engine default | `0` gives a 12.0 card the shared table again |
| `STRATA_MMVQ_IL_Q3K` | engine default | the Q3_K row of the interleaved table |
| `STRATA_MMVQ_IL_T1` | on | the one-column interleaved kernel; `0` keeps native_mmvq's |
| `STRATA_MMVQ_IL_WIDE` | on | the 5-8 column interleaved path; `0` turns it off |
| `STRATA_MMVQ_WAVE` | engine default | `0` runs the layout parity test without the wave form |
| `STRATA_MMVF_ROWS` | architecture table (gfx1151) | bf16 multi-row GEMV, 4 rows per block |
| `STRATA_MMVF_PF` | engine default | the router GEMV's prefetch form; the bench compares `0` / `1` |
| `STRATA_SAMPLER_ONE_BLOCK` | engine default | the one-block sampler; part of the sampler parity test |
| `STRATA_OLD_SAMPLER` | off | the old sampler kernel |
| `STRATA_OLD_IQ_MMVQ` | off | the old native gate/up and down kernels |
| `STRATA_OLD_GROUPED` | off | the previous grouped expert kernels |
| `STRATA_GROUPED_V1` | off | the first grouped expert kernel |
| `STRATA_EXPERT_V2` | architecture table (gfx1151) | grouped decode experts (IQ3_S gate/up, IQ4_NL down) |
| `STRATA_EXPERT_V2K` | engine default | `0` / `1` picks the split path's kernel form |
| `STRATA_EXPERT_PAIR` | engine default | row-pair variants: one warp per pair |
| `STRATA_EXPERT_QPN2` | engine default | NVFP4 expert decode path |
| `STRATA_EXP_MODE` | engine default | expert kernel mode of the native expert bench |
| `STRATA_KQ_SPLIT` | off | the split K-quant path, to compare against the old kernels |
| `STRATA_Q2_WARP` | engine default | the Q2 warp kernel; `0` keeps the old kernels, `1` takes it on any card |
| `STRATA_Q6_PACKED` | architecture table (gfx1151) | the packed Q6_K heads |
| `STRATA_Q6P_GRID` | engine default | grid of the packed Q6_K kernel |
| `STRATA_Q6P_SELFTEST` | off | `1` / `2` run the packed Q6_K self-test and grid sweep |
| `STRATA_Q8_PACKED` | architecture table (gfx1151) | the packed Q8_0 decode layout |
| `STRATA_Q8_SM60` | off | the sm_60 Q8 kernel (a no-op on other types) |
| `STRATA_Q8_SM60_BPS` | engine default | bytes per step of the sm_60 Q8 kernel |
| `STRATA_DEQUANT_WIDE` | architecture table (sm_70: `1`) | vector stores for native weights |
| `STRATA_GR_FAST` | engine default | the AMD fast hyper-connection kernels |
| `STRATA_GR_NORM_FAST` | architecture table | the fast norm of the hyper-connection read |
| `STRATA_GR_UP75` | architecture table | the up form of that read |
| `STRATA_GR_DOWN_MAX4` | off | up to 4 tokens' sums per thread in the down read |
| `STRATA_GR_DOWN_TG` | architecture table (sm_70) | thread-group form of the down read |
| `STRATA_GR_SPLIT` | off | gfx906: latency-hidden norm/up with a K-split down |
| `STRATA_GR_UNFUSED` | off | A/B: keep the FP32 copy of the normalized rows |
| `STRATA_GR_V3` | engine default | the split read of the hyper-connection kernels |
| `STRATA_HC_PERSIST` | off | `3` enables the persistent HIP kernel |
| `STRATA_HC_Q8` | engine default | Q8_0 copy of the hyper-connection matrices |
| `STRATA_HC_Q8_INJECT` | off | ask for the Q8_0 copy explicitly |
| `STRATA_HC_REQ8` | off | expand the int8 codes into one reused scratch |
| `STRATA_HC_SPLIT` | engine default | the split hyper-connection form |
| `STRATA_HC_UPMIX` | off | the up projection and `gr_mix_r` in one kernel |
| `STRATA_HC_UPMIX_CHECK` | 0 (off) | compare the fused up-mix against `gr_upmix_parity` |
| `STRATA_GU_SWIGLU` | architecture table | the fused SwiGLU gate/up kernel |
| `STRATA_FUSED_SWIGLU_Q81` | engine default | the shared expert's fused SwiGLU with q8_1 output |
| `STRATA_S2_SWIGLU_Q8` | on | the S2 activation written back once; `0` keeps the two launches |
| `STRATA_QUANT_GATHER` | off | gather the quantized experts before the MMQ |
| `STRATA_V2K` | engine default | the s27 launch form of the grouped experts |
| `STRATA_B6_MMVQ_ROWS` | architecture table | the 6-row mmvq form |
| `STRATA_NO_MMVQ_ROWS2` | engine default | `1` forces one row, `0` keeps two rows everywhere |
| `STRATA_TQ_SUB8` | off | the sub-8-bit trellis form |
| `STRATA_SHFL_W` | engine default | warp width of the HIP shuffle compatibility macro |
| `STRATA_PDL_RESTRICT` | engine default | `__restrict__` on the mmvq pointers below sm_70 |

### CPU pool and placement

| Variable | Default | Description |
|---|---|---|
| `STRATA_POOL_SPIN_US` | 20000 (20 ms) | spin before a pool worker sleeps; `0` sleeps at once |
| `STRATA_POOL_DEP` | off (`1` turns it on) | one batch per native CPU layer call: a down row range waits only for its own expert's gate/up ranges instead of the phase barrier, and the first worker there quantizes that expert's intermediate once. Bit-exact. Measured with `--max-new` A/B (see [NVIDIA_V100.md](NVIDIA_V100.md)) |
| `STRATA_POOL_TRACE` | off | print who waits on the expert pool |
| `STRATA_POOL_UMWAIT` | off | use `WAITPKG` waits in the pool (Intel) |
| `STRATA_POOL_UMWAIT_C02` | 1 | C0.2 wait form; `1` takes it, any other value keeps the plain spin |
| `STRATA_POOL_UMWAIT_ESC` | 0 (off) | TSC cycles after which the wait escalates from C0.1 to C0.2 (experimental) |
| `STRATA_CPU_IQ_ALL_EXACT1` | architecture table (sm_86) | exact IQ kernels for every CPU expert |
| `STRATA_CPU_YMM` | off | use 256-bit registers instead of 512-bit in the CPU expert kernels |
| `STRATA_FORCE_ISA` | engine default | forces a CPU tier, for example `avx` |
| `STRATA_FORCE_AVX2` | off | `1` skips the AVX-512 layout and keeps AVX-2 |
| `STRATA_NO_AVXVNNI` | off | keep the AVX-VNNI kernels out |
| `STRATA_NO_IQ128` | off | an AVX-1 CPU keeps ggml's scalar reference |
| `STRATA_NO_IQ256` | off | drop the IQ256 kernel on AVX-2 CPUs |
| `STRATA_NO_IQ4NL` | off | fall back to ggml's single-token IQ4_NL dot |
| `STRATA_NO_IQ512` | off | drop an AVX-512 CPU to the AVX-2 tier |
| `STRATA_NO_NVFP4_256` | off | fall back to ggml-cpu for NVFP4 on AVX-2 |
| `STRATA_NO_NVFP4_512` | off | fall back to ggml-cpu for NVFP4 on AVX-512 |
| `STRATA_NO_Q8K_AVX1` | off | keep ggml's scalar reference on an AVX-1 CPU |
| `STRATA_NO_Q8K_AVX2` | off | keep ggml's Q8_K reference |
| `STRATA_NO_TQ_KERNELS` | off | do not use the CPU trellis kernels |
| `STRATA_IQ256_GATHER` | engine default | gather form of the IQ256 kernel |
| `STRATA_IQ3S_MT1` | off | multi-token IQ3_S rounding on AVX-2 CPUs |
| `STRATA_IQ_GATHER` | off | gather-based IQ lookups on AVX-512 |
| `STRATA_IQ_MT_MIN` | engine default | fewest tokens for the multi-token IQ path |
| `STRATA_IQ_PREFETCH` | engine default | software prefetch distance of the IQ kernels |
| `STRATA_KQ256` | off | the KQ256 kernel form |
| `STRATA_NVFP4_PREFETCH` | 0 (off) | prefetch distance in bytes for the NVFP4 AVX-2 rows |
| `STRATA_TQ_ISA` | engine default | CPU tier the trellis kernels use |
| `STRATA_TSUM` | architecture table (gfx1151) | several warp sums as one transposed butterfly |
| `STRATA_Q2_BITPLANE` | off | bit-plane row kernel for Q2 on AVX-1 |
| `STRATA_PRIMARY_DEVICE` | engine default | device index the engine treats as primary |
| `STRATA_NO_BATCH_KV_STEP` | off | append the drafter's K/V per token instead of in one step |
| `STRATA_NO_MULTI_GR` | off | keep the per-token GR calls |
| `STRATA_NO_NORM_ROPE` | off | keep the norm and RoPE as two kernels |
| `STRATA_NORM_ROPE` | engine default (off on HIP) | the fused norm and RoPE kernel |

### Layer split and the pipeline

| Variable | Default | Description |
|---|---|---|
| `STRATA_SPLIT_RING` | engine default | streamed ring slots of a layer split, from the stages' residency |
| `STRATA_SPLIT_CHUNK16K` | engine default | `1` keeps the gfx12 16384 ceiling on a split |
| `STRATA_SPLIT_COVER_B` | 3.0 | curve fit constant of the split's traffic model |
| `STRATA_SPLIT_LINK_ABS` | off | `1` takes 0.1.41's fixed 20 GB/s rule |
| `STRATA_SPLIT_MISS_MS` | 190 | miss cost of the traffic fit; raise it on a slow CPU |
| `STRATA_SPLIT_NEAR_TIE_PCT` | 4.0 | percent band in which two stage shares count as a tie |
| `STRATA_SPLIT_SPEED_SCALE` | none | per-stage speed factors, for example `1,0.55` (test seam) |
| `STRATA_SPLIT_UNIFORM` | off | `1` removes the per-link scaling (pre-0.1.41 rule) |
| `STRATA_SPLIT_OWN` | off | prompt experts of every stage on its own card; `0` = the default |
| `STRATA_SPLIT_OWN_BUFFERS` | engine default | the split's first card keeps its own prompt buffers |
| `STRATA_SPLIT_OWN_HEADROOM_MIB` | engine default | VRAM kept free beside those buffers |
| `STRATA_SPLIT_OWN_PLACE` | engine default | `reserve` places the split's own buffers by reservation |
| `STRATA_SPLIT_OWN_PRIMARY` | 0 (borrow) | chunk the split's first card keeps its own prompt buffers for |
| `STRATA_SPLIT_SMALL_MAX` | engine default | a prompt of at most M tokens reads in S-token chunks on those stages |
| `STRATA_SPLIT_SMALL_OWN` | off | the same small-prompt rule for the split's own buffers |
| `STRATA_SPLIT_TIMING` | off | print where each verify stage's host time went |
| `STRATA_SPLIT_OVERLAP` | off unless set | the overlapped split; the serial order otherwise. `--pipeline-windows` overlaps the stages itself |
| `STRATA_SPLIT_WAIT_MS` | 30000 | hand-off wait bound; `0` = no bound |
| `STRATA_PIPELINE_AUTO` | on for 2+ gfx12 stages | `0` turns the automatic pipelined windows off |
| `STRATA_PIPELINE_AUTO_MIN_STAGES` | engine default | fewest stages that get the automatic pipeline |
| `STRATA_PIPELINE_DEBUG` | off | allow the tuning variables below (they are ignored without it) |
| `STRATA_PIPELINE_ADAPT_ASYNC` | on | the async adaptive tier beside the pipeline; `0` keeps the blocking tier |
| `STRATA_PIPELINE_AGREE` | on | the old chain's probabilities too; `0` takes only them |
| `STRATA_PIPELINE_DOOM_SKIP` | on | skip the doomed rows; `0` serves them in full |
| `STRATA_PIPELINE_FORCE_MISS` | off | debug: force a miss on the named window |
| `STRATA_PIPELINE_LOG` | off | log the pipeline's decisions |
| `STRATA_PIPELINE_LOOKUP_PON` | engine default | speculate behind the copied stretch |
| `STRATA_PIPELINE_PRESTAGE` | on | stage B at its launch; `0` stages it earlier |
| `STRATA_PIPELINE_SNAP_OVERLAP` | on | two snapshot buffers overlapped; `0` = one buffer |
| `STRATA_PIPELINE_SPEC_DEPTH` | engine default | debug: B runs speculatively on the first k front stages |
| `STRATA_PIPELINE_SWITCH` | engine default | per-request `short_read=` switch |
| `STRATA_PIPELINE_THETA` | engine default | pipeline share of the two stages |
| `STRATA_PIPELINE_TRACE` | off | append every event of the loop with its time to the file |
| `STRATA_PIPE_BYIELD_NODRAIN` | off | a BYIELD that reaches the main thread does not drain |
| `STRATA_PIPED_READ_PIECE` | engine default | tokens per read piece on the pipelined slot groups |
| `STRATA_PIPED_DECODE_MIN_MS` | engine default | floor of a piece's decode time |
| `STRATA_PIPED_DECODE_MAX_MS` | engine default | cap of a piece's decode time |
| `STRATA_PIPED_CONCURRENT_UNPIN` | on | the concurrent prompt gives its CPUs back; `0` keeps them |
| `STRATA_REFILL_BLOCKING` | off | refill lent slots with one wait per copy |
| `STRATA_REFILL_PARTS` | off | refill a lent slot from the pieces in the mapping |
| `STRATA_REFILL_SERIAL` | off | refill stage after stage, as 0.1.30 did |
| `STRATA_LAYER_SPLIT_GATE` | off | debug: print the layer split's gate decisions |
| `STRATA_LAYER_SPLIT_PROBE` | on | probe the links at start; `0` skips it |
| `STRATA_GFX12_CHUNK` | 8192 | prompt chunk ceiling on gfx12; `0` keeps 8192 |
| `STRATA_EARLY_REMOTE_CONTEXTS` | off | create every secondary context at start (WSL2 workaround) |
| `STRATA_INPUT_ANCHOR_DEFER` | on | deferred anchor save; `0` = the synchronous save |
| `STRATA_EMULATE_CC` | engine default | report another compute capability (tests) |

### Batch slots and the conversation cache

| Variable | Default | Description |
|---|---|---|
| `STRATA_BATCH_MTP` | off | each batch slot verifies one MTP proposal per window |
| `STRATA_BATCH_MTP_SERIAL` | off | one batch group because of `--batch-mtp` |
| `STRATA_BATCH_DECODE_SHARE` | 0.5 | share of a chunk's time that decode keeps |
| `STRATA_BATCH_YIELD_CHUNKS` | 8 (`0` = one run) | chunks a yieldable decode runs in |
| `STRATA_BATCH_TELEMETRY` | off | print the batch's per-window telemetry |
| `STRATA_ADMIT_MERGE_FIRST` | off | a batch admission reads the last prompt tokens of its last request first |
| `STRATA_CACHE_MESSAGE_BOUNDARY` | off | checkpoint at message boundaries |
| `STRATA_SESSION_BUFFERED` | engine default | buffered session file; `1` forces it |
| `STRATA_SESSION_GOLDEN_PRINT` | off | tests: print the golden values of the session file |
| `STRATA_SLOT_CKPT_MOVE` | off | a batch admission moves its own turn checkpoint into the slot |
| `STRATA_SLOT_COPY_DIRECT` | off | direct slot copy |
| `STRATA_SLOT_COPY_BATCHSYNC` | off | the direct slot copy only queues its copies |
| `STRATA_TEST_SERVE_DROP_LOAN` | off | drop the accepted loan before the contract check |
| `STRATA_TEST_SLOW_COPY_US` | 0 (off) | tests: microseconds added per MiB of a checkpoint copy |
| `STRATA_KEEP_EMPTY_TURNS` | off | render empty turns as before |
| `STRATA_KEEP_LEFTOVERS` | off | server: keep the leftover state of a failed request |
| `STRATA_PARALLEL_SOLO` | on | keep a solo parallel request in its slot; `0` = off |

### PLE and the n-gram table

| Variable | Default | Description |
|---|---|---|
| `STRATA_PLE_BATCH` | on | the verify window's PLE projections at once; `0` = per token |
| `STRATA_PLE_AUTO_MMAP` | on | fall back to mapped reads when the probe is slow |
| `STRATA_PLE_PROBE` | off | keep the direct read mode after a slow probe |
| `STRATA_PLE_GGUF` | engine default | the PLE table's shard, when no `--ple-gguf` is given |

### Vision

| Variable | Default | Description |
|---|---|---|
| `STRATA_VISION_ARCH_CHECK` | off | refuse to run if the vision binary's GPU architecture does not match |

The vision binary's own build sets `STRATA_VISION_CUDA`, `STRATA_VISION_HIP` and
`STRATA_VISION_HIP_ARCHS` (`tools/vision/strata_vision.cpp`). The engine and the server do not
read them.

### Platform, kernels and tuning tables

| Variable | Default | Description |
|---|---|---|
| `STRATA_SM60_TABLE` | on | the P100 decode table; `0` keeps native_mmvq's kernels |
| `STRATA_SM61_TABLE` | on | the sm_61 decode table; `0` keeps native_mmvq's kernels |
| `STRATA_SM70_TABLE` | architecture table | the Volta master switch of the interleaved dense table |
| `STRATA_SM70_PERF` | on (sm_70) | `0` turns the whole Volta table off |
| `STRATA_SM70_CHUNK` | RAM-priced | prompt chunk ceiling on one Volta card: `0` = 8192, `1` = 16384 |
| `STRATA_SM75_PERF` | engine default (Turing) | the Turing series switches |
| `STRATA_SM75_IL` | engine default | `0` / `1` forces the interleaved table off / on |
| `STRATA_SM86_PERF` | on (sm_86) | `0` turns the whole sm_86 table off |
| `STRATA_SM120_PERF` | on for a 12.0 card with 80+ SMs | `0` turns the sm_120 switches off |
| `STRATA_ROPE_TABLE` | engine default | angles from the session's float64 table |
| `STRATA_BF16_TC` | off | Turing tensor-core BF16 prompt GEMMs (Volta has them by default) |
| `STRATA_DENSE_F32` | off | dense prompt projections as `cublasSgemm` on Pascal |
| `STRATA_DENSE_MMQ` | engine default | quantized dense MMQ kernels |
| `STRATA_CVEC_FUSE` | off | a steered layer's write, control vector and next norm in one pass |
| `STRATA_CVEC_FUSE_CHECK` | 0 (off) | on the first N fused writes, also run the three-kernel path |
| `STRATA_HCD_EXACT` | off | the hyper-connection down projection by a WMMA kernel |
| `STRATA_RDNA2_SGEMM` | off | the `SGEMM` path for gfx103x |
| `STRATA_WMMA_GEMM` | off | RDNA3 / RDNA3.5 WMMA dense GEMM |
| `STRATA_WMMA_BF16` | on | `0` excludes bf16 from `STRATA_WMMA_GEMM` |
| `STRATA_HIPBLASLT_TUNING` | none | hipBLASLt solution table file or directory |
| `STRATA_HIPBLASLT_VERBOSE` | off | print the hipBLASLt solution chosen |
| `STRATA_ROCBLAS_TUNING` | none | rocBLAS solution table file or directory |
| `STRATA_ROCBLAS_VERBOSE` | off | print the rocBLAS solution chosen |
| `STRATA_IQ_STAGE_GRID` | on | stage the 64-bit i-quant codebook tables in shared memory; `0` off |
| `STRATA_IQ_STAGE_GRID_MMVQ` | off | the same staging for mmvq |
| `STRATA_IQ_STAGE_TABLES` | off | stage the tables for the single-matrix mmvq too |
| `STRATA_GDN_HEAD` | architecture table | the GDN recurrence with four lanes a column |
| `STRATA_GDN_PP` | architecture table (2 on AMD) | columns per lane of the GDN recurrence |
| `STRATA_GDN_CONVL2` | architecture table | conv with the q/k L2 norm in one kernel (AMD only) |
| `STRATA_GDN_NOY` | architecture table | drop the dead FP32 store of the norm |
| `STRATA_GDN_SPLIT` | architecture table (gfx1151) | the GDN step over 4 blocks per head |
| `STRATA_GDN_KEYHEAD` | on | the key-head recurrence kernel; `0` takes `gdn_rec_cols_pipe_kernel` |
| `STRATA_GDN_PIPELINE` | on | the pipelined recurrence; `0` takes `gdn_rec_cols_kernel` |
| `STRATA_GDN_CHUNK` | engine default | `2` selects the FP32-only FMA scan of the chunked recurrence |
| `STRATA_GDN_CHUNKED` | off | `2` or more enables the chunked recurrence (WY form) |
| `STRATA_GDN_CHUNK_PROF` | off | per-stage milliseconds of the chunked GDN |
| `STRATA_GDN_COMMIT_SPLIT` | engine default | split form of the GDN commit |
| `STRATA_GDN_CONV_SERIAL` | off | the old serial conv walk (A/B) |
| `STRATA_GDN_REC_FAST` | engine default (cc 7.0 only) | `0` baseline, `1` opts in |
| `STRATA_GDN_REC_HEADS` | off | the one-block-per-head recurrence kernel (A/B) |
| `STRATA_GDN_REC_WARP` | engine default | `1` selects the sync-free bit-exact recurrence |
| `STRATA_GDN_XE` | engine default (on for Alchemist) | `0` off, `1` on any Intel GPU |

### Qwen3.6 engine

| Variable | Default | Description |
|---|---|---|
| `STRATA_Q36_CHUNK` | engine default | forces a prompt chunk, 256..4096 (512 is the old default) |
| `STRATA_Q36_READ_WINDOWS` | 1 | decode windows of other slots between two chunks of a prompt read |
| `STRATA_Q36_BLAS_F32` | off | the hipBLAS / cuBLAS SGEMM (A/B) |
| `STRATA_Q36_DRAFT_VOCAB` | engine default | draft head vocabulary subset |
| `STRATA_Q36_FIXED_SLOTS` | off | tests: place the slots as given |
| `STRATA_Q36_KEEP_PACKED` | off | keep the packed layout instead of the hybrid mode |
| `STRATA_Q36_NO_EXT` | off | keep the extended GEMM path out |
| `STRATA_Q36_NO_MMQ` | off | keep the MMQ kernels out |
| `STRATA_Q36_NO_MTP_CPU` | off | keep the MTP head off the CPU |
| `STRATA_Q36_OLD_KERNELS` | off | the old kernels |
| `STRATA_Q36_STREAM_ALL` | off | stream the whole layer from one source |
| `STRATA_Q36_VERIFY` | off | compare each new kernel against the old one |

### Image pipeline

| Variable | Default | Description |
|---|---|---|
| `STRATA_QI_TIMING` | off | per-step timings of the image pipeline |
| `STRATA_QI_QSPLIT` | off | split attention (measured: no accuracy gain, +7% step time) |
| `STRATA_QI_EXACT_ATTN` | off | exact attention in the image pipeline |
| `STRATA_QI_NO_QKV_FUSE` | off | keep the QKV projections separate |
| `STRATA_QI_FAST_LN` | off | fast layer norm (measured: no gain on the RTX 3060) |
| `STRATA_QI_GGML_ATTN` | off | keep ggml's attention |
| `STRATA_QI_OLD_QKV` | off | the old QKV kernel |
| `STRATA_QI_GFX12_ATTN` | on | `0` keeps ggml's attention on gfx12 |

### Memory and allocation

| Variable | Default | Description |
|---|---|---|
| `STRATA_MEMORY_LIMIT_MIB` | 0 (none) | RAM budget of the container or cgroup; the flag wins |
| `STRATA_UMA_HEADROOM_GIB` | 6 | VRAM left free on a unified-memory device |
| `STRATA_WDDM_BUDGET` | engine default | WDDM budget the engine plans against |
| `STRATA_TEST_PIN_BUDGET_MIB` | engine default | tests: soft pin budget |
| `STRATA_TEST_PIN_LEFT_KIB` | engine default | tests: free KiB of the pin budget |
| `STRATA_TEST_PINNED_SHARE` | engine default | tests: the Windows-like ring rule |
| `STRATA_TEST_POSTTOUCH_SHORTFALL` | 0 | tests: report N MiB less free RAM after the post-touch read |
| `STRATA_TEST_UNMAP_GGUF` | off | tests: close the GGUF views |

### Server and networking

| Variable | Default | Description |
|---|---|---|
| `STRATA_API_KEY` | `""` | keys the server accepts |
| `STRATA_ALLOWED_HOSTS` | config | extra `Host` names the server answers to |
| `STRATA_MAX_BODY_MIB` | 0 = 256 MiB | largest request body read into memory |
| `STRATA_HTTP_BACKLOG` | 256 (minimum 5) | listen backlog |
| `STRATA_START_RETRIES` | 3 | retries of a start that fails in the GPU runtime |
| `STRATA_ENGINE_READY_S` | 900 | seconds to wait for the engine's READY line; `0` = for ever |
| `STRATA_ENGINE_STALL_S` | 90 | frozen-engine watchdog; `0` = off; needs `psutil` |
| `STRATA_VISION_READY_S` | 300 | image encoder start timeout; `0` = for ever |
| `STRATA_VISION_ENCODE_S` | 300 | image encode timeout; `0` = for ever |
| `STRATA_ADMIT_SHORT_FIRST` | 0 (off) | short prompts take the control lines first |
| `STRATA_ADMIT_MAX_WAIT_S` | 30 | a waiter beyond this counts as a short prompt |
| `STRATA_DEBUG` | off | dump the raw model text of each request |
| `STRATA_REQUEST_LINES` | off | one stdout line per finished request |
| `STRATA_BATCH_TRACE` | off | trace the batch admission |
| `STRATA_TELEMETRY_KEY` | config | read key of the telemetry endpoint |
| `STRATA_REPLICA_START` | parallel | `serial` starts the replicas one after the other |
| `STRATA_NO_PRESET` | off | a tune's own worker measures without the preset |
| `STRATA_SYCL_BIN` | `build-sycl-aot/strata` | the SYCL engine binary the server starts |
| `STRATA_LIFELINE` | auto | `watcher` forces the portable lifeline path (tests) |

### Terminal, control and setup

| Variable | Default | Description |
|---|---|---|
| `STRATA_URL` | the built-in default URL | server the `strata` command talks to |
| `STRATA_CONNECTION` | saved default | connection name to use |
| `STRATA_CONNECTION_DIR` | `~/.config/strata/connections` (Windows: `%APPDATA%\Strata\connections`) | connection files; must be absolute and owner-only |
| `STRATA_SETTINGS_FILE` | `$XDG_CONFIG_HOME/strata/settings.json` | settings file; the desktop app sets its own |
| `STRATA_DOCUMENTS_DIR` | the person's Documents folder | where the app keeps `Strata` |
| `STRATA_INSTALL_DIR` | the repository root | where `setup.py` is |
| `STRATA_STATE_DIR` | the per-user state folder | state files of the terminal |
| `STRATA_PROFILE_DIR` | the state folder | saved profiles |
| `STRATA_PRESET_DIR` | next to the profiles | presets |
| `STRATA_RUNTIME_DIR` | the state folder's `runtime` | must be absolute |
| `STRATA_RUNTIME_PROFILE` | none | profile name written into the lifeline's process record |
| `STRATA_LAUNCH_PRESET_DIR` | the state folder | launch presets |
| `STRATA_TUNE_DIR` | the state folder | tune results |
| `STRATA_TIMESKIP_DIR` | the state folder | where timeskip runs are recorded |
| `STRATA_TIMESKIP_PROMPT` | none | prompt file a timeskip run replays |
| `STRATA_TIMESKIP_ATTEMPT` | none | attempt number a timeskip run writes |
| `STRATA_COMPONENTS_DIR` | the per-user folder | component folder of the terminal |
| `STRATA_TUNE_GPU_LOCK` | none | GPU lock file for `strata tune` |
| `STRATA_GLYPHS` | auto | `ascii` or `unicode` glyph set |
| `STRATA_CLOUDFLARED` | from `PATH` | where a `cloudflared` binary is |
| `STRATA_NEED_ENGINE` | off | `1` makes setup stop when the engine is missing |
| `STRATA_ALLOW_NO_ENGINE` | off | `1` lets setup go on without an engine |
| `STRATA_EXECV` | off | POSIX: replace the process with the engine |
| `STRATA_CONTROL_INSTALL` | off | `1` runs the tools against the installed control module |
| `STRATA_ALLOW_UNPINNED` | off | `1` lets the fetch tool take an unpinned revision |
| `STRATA_MTP_REVISION` | pinned revision | model revision the fetch tool downloads |
| `STRATA_SOURCE` | Hugging Face | `ms` or `modelscope` takes ModelScope |
| `STRATA_DL_CONNECTIONS` | 8 | download connections of the fetch tool |
| `STRATA_SKIP_SHA256` | off | skip the engine's hash check |
| `STRATA_UPDATE_FEED` | the project feed | signed update feed for engine downloads |
| `STRATA_UPDATE_TRUSTED_KEYS` | `update_keys.json` | keys a feed must be signed with |
| `STRATA_UPDATE_ALLOW_HTTP` | off | allow an `http` update feed |
| `STRATA_APP_UPDATE_PUBKEY` | built-in key | public key of the application updater |
| `STRATA_REQUIRE_SIGNED_COMPONENTS` | off | `1` refuses unsigned components |
| `STRATA_PREBUILT_URL` | the project release URL | where setup takes prebuilt engines |
| `STRATA_CUDA` | auto | `12`, `13` or `auto` |
| `STRATA_NVCC` | from `PATH` | the `nvcc` setup uses |
| `STRATA_VCVARS` | from `vswhere` | `vcvars64.bat` path |
| `STRATA_ROCM_INDEX` | per architecture | pip index of the ROCm wheels |
| `STRATA_ROCM_VERSION` | per architecture | ROCm wheel version |
| `STRATA_ISA_FLOOR` | auto | `avx` or `none` |
| `STRATA_EXPERIMENTAL_GFX1103` | off (experimental) | `1` opts a gfx1103 card in |
| `STRATA_EXPERIMENTAL_GFX1150` | off (experimental) | `1` opts a gfx1150 card in |
| `STRATA_EXPERIMENTAL_SM60` | off | `1` enables the experimental sm_60 kernels |
| `STRATA_GGUF_PY` | the bundled `gguf-py` | where the converter finds `gguf-py` |
| `STRATA_REPO` | the repository root | tools: where the tree is |
| `STRATA_EXE` | the config's engine path | tools: engine binary to start |
| `STRATA_EXTRA_ARGS` | none | tools: extra engine arguments |
| `STRATA_KEY` | `""` | tools: API key for the load tests |
| `STRATA_SHARD1` | the development layout | converter: first shard of the model |
| `STRATA_TOKENIZER` | the pack's tokenizer | tests: tokenizer for the detokenize test |

### Diagnostics and tracing

| Variable | Default | Description |
|---|---|---|
| `STRATA_TRACE` | off | trace the prompt path's steps |
| `STRATA_TRACE_ADAPT` | off | report whether a round's copies had landed |
| `STRATA_TRACE_RESUME` | off | trace the resume path |
| `STRATA_DECODE_TIMING` | off | where a decode window's time goes (with `--serve`) |
| `STRATA_DRAFT_TIMING` | off | where the draft pass's time goes |
| `STRATA_COMMIT_TIMING` | off | GPU time of the commit graph, from an event pair |
| `STRATA_VERIFY_DEBUG` | off | verbose verify output |
| `STRATA_VERIFY_PROFILE` | off | verify profile hooks |
| `STRATA_VERIFY_TRACE` | off | trace every verify window |
| `STRATA_LOGPOS` | off | read every window's logits (the serial loop) |
| `STRATA_LOGPOS_EXTRA` | 248046 | column whose log-probability is added to each position |
| `STRATA_LOGPOS_TOPK` | 0 (off) | also report the K most likely tokens and their log-probabilities |
| `STRATA_DUMP_FIRST_LOGITS` | off | write each request's first logits to the path |
| `STRATA_DUMP_MOE_INPUT` | off | write the first chunk's MoE input to the path |
| `STRATA_DUMP_MOE_LAYER` | 0 | layer the MoE dump writes |
| `STRATA_DBG_FEAT` | off | research hook: dump routing and router input to the path |
| `STRATA_DBG_FEAT_LAYERS` | none | layers the routing dump covers |
| `STRATA_DBG_ROUTE_OVERRIDE` | off | replace the routing of the layers in the file |
| `STRATA_DBG_FORCE_SLOW_LAYER` | off | test the backstop: layer 0 counts as a slow layer |
| `STRATA_DBG_GDN` | off | check the GDN window limit |
| `STRATA_DBG_NAN` | off | check for non-finite values |
| `STRATA_PARITY_DUMP` | off | dump the route-resident parity data |
| `STRATA_RSS_TRACE` | off | print the mapped file pages at a startup step |
| `STRATA_STATE_HASH` | off | hash the running state of each window |
| `STRATA_STATE_HASH_GDN` | off | hash the GDN states after the prompt |
| `STRATA_Q36_PLACE_DEBUG` | off | print the slot placement of the Qwen3.6 engine |
| `STRATA_Q36_VDEBUG` | off | print a mismatching Qwen3.6 block |

### Measurement A/B arms and test hooks

| Variable | Default | Description |
|---|---|---|
| `STRATA_BENCH_QMIXED` | off | the fused GR bench also runs the q-mixed form |
| `STRATA_TEST_PAGEABLE` | off | make every pinned allocation fall back to pageable |
| `STRATA_TEST_RING_FAIL` | off | make the ring's allocation fail like an out-of-memory `cudaMalloc` |
| `STRATA_TEST_CACHE_FAIL` | 0 | fail the cache's allocation after N tries |
| `STRATA_TEST_BELL_SLEEP_US` | 0 | sleep N us instead of the bell wait |
| `STRATA_TEST_CONC_ON_HOST_CPU` | off | run the concurrent prompt on the host CPU |
| `STRATA_LOOKUP_CHAIN_FIXED` | off | no policy gate on the lookup chain |

## Compile-time macros

These 61 names are build settings, not run-time settings. A person cannot change them with a
shell export or with `strata tune`. To change one, change the build: `CMakeLists.txt` or the
`-D` argument of the compiler. The engine and the server do not read them with `getenv`,
except the four names marked below: those set a build default and also accept a run-time
override, which the tables above describe.

| Macro | Where it is used | Meaning |
|---|---|---|
| `STRATA_AM_CLUSTER` | `src/kernels/cuda/sampler.cu` | builds the cluster form of the sampler |
| `STRATA_ATOMIC_WAIT` | `src/prefill/prefill.cpp` | uses `std::atomic::wait`; a GCC 10 build spins instead |
| `STRATA_ATTN_PRE75_BUILT` | `src/kernels/cuda/qsa_decode_attn.cu` | the pre-sm_75 attention chunk kernel is built |
| `STRATA_AVXVNNI` | `src/kernels/cpu/q2_avx2.cpp` | the compiler has the AVX-VNNI intrinsics |
| `STRATA_DEQUANT_SELFTEST` | `src/artifact/dequant.cpp` | builds the dequantization self-test |
| `STRATA_EC_NO_VMM` | `src/core/expert_cache.cpp` | the expert cache is built without virtual memory management |
| `STRATA_EXPDEC_MINB` | `src/kernels/cuda/iq_kernels.cu` | block floor of the EXPDEC kernel |
| `STRATA_EXPERIMENTAL_SM60` | `src/prefill/gdn_chunk.cu` | builds the experimental sm_60 GDN kernels; also read by `setup.py` |
| `STRATA_EXP_LAYOUTS` | `src/kernels/cuda/iq_kernels.cu` | builds the expert kernel layout variants |
| `STRATA_F16_DEVICE` | `src/prefill/moe_fused_iq.cu` | the FP16 fused-expert device pass |
| `STRATA_FSEEK64` | `src/core/expert_source.cpp` | name of the 64-bit seek wrapper |
| `STRATA_FUSED_W11` | `src/prefill/moe_fused.cu` | builds the gfx11 fused prompt expert |
| `STRATA_GDN_CP_ASYNC` | `src/prefill/kernels.cu` | the GDN kernels use `cp.async` |
| `STRATA_GDN_WMMA` | `src/prefill/kernels.cu` | the GDN device pass uses the gfx11 WMMA intrinsic |
| `STRATA_GFX1012_PORTABLE_DOT` | `include/strata/hip_compat/intrinsics.hpp` | portable dot product on gfx1012 |
| `STRATA_GGUF_MAIN_DISABLED` | `src/artifact/gguf_reader.cpp` | the GGUF reader has no `main()` |
| `STRATA_GPU_METAL` | `src/prefill/prefill.cpp` | the build has the Metal backend |
| `STRATA_GR_CP_ASYNC` | `src/kernels/cuda/fused_gr.cu` | the fused GR kernel uses `cp.async` |
| `STRATA_GR_FAST_BUILD` | `src/kernels/cuda/fused_gr.cu` | builds the fast variant of the fused GR kernel |
| `STRATA_HAVE_QWEN35` | `src/program/generate.cpp` | the Qwen3.5/3.6 engine is compiled in |
| `STRATA_HIP_ARCHS` | `src/core/device.cu` | the HIP architecture list of the build |
| `STRATA_HIPBLASLT_AVAILABLE` | `src/prefill/gemm.cu` | hipBLASLt is available |
| `STRATA_HIP_GFX906` | `src/prefill/prefill.cpp` | the build targets gfx906 |
| `STRATA_IQ_PARITY_DISPATCH_ONLY` | `src/kernels/cpu/iq_avx2_parity.cpp` | the parity test calls only public entry points |
| `STRATA_ISA_FLOOR_AVX` | `src/kernels/cpu/expert_layout.cpp` | the CPU floor of the build is AVX |
| `STRATA_ISA_FLOOR_NONE` | `src/kernels/cpu/expert_layout.cpp` | the build has no CPU instruction floor |
| `STRATA_KQ_DSO` | `src/prefill/moe_mmq.cu` | the K-quant library is a shared object |
| `STRATA_MMQ_KQUANTS` | `src/prefill/moe_mmq.cu` | the MMQ has the K-quant types |
| `STRATA_MMQ_TAB` | `src/prefill/moe_mmq_tab.cuh` | the MMQ table kernel; CMake verifies a source hash first |
| `STRATA_MMQ_TQ` | `src/prefill/moe_mmq.cu` | the MMQ has the trellis types |
| `STRATA_MMQ_TQP` | `src/prefill/moe_mmq.cu` | the MMQ has the pair-trellis types |
| `STRATA_NATIVE_EXPERTS` | `src/prefill/prefill.cpp` | the llama.cpp sources are present, so the native experts are built |
| `STRATA_NAT_W11` | `src/prefill/moe_fused_iq.cu` | the native W11 kernel |
| `STRATA_NAT_W12` | `src/prefill/moe_fused_iq.cu` | the native W12 kernel |
| `STRATA_NAT_WMMA` | `src/prefill/moe_fused_iq.cu` | `NAT_W11` or `NAT_W12` |
| `STRATA_NO_RLIMIT_TEST` | `src/core/conversation_file_test.cpp` | the conversation test skips the rlimit case |
| `STRATA_ORCA_Q4KS_MMQ` | `src/prefill/moe_mmq.cu` | builds the ORCA Q4_K_S MMQ kernel |
| `STRATA_PA_GFX11` | `src/kernels/cuda/qsa_prompt_attn.cu` | the gfx11 prompt attention variant |
| `STRATA_PA_SM80` | `src/kernels/cuda/qsa_prompt_attn.cu` | the sm_80 prompt attention variant |
| `STRATA_PA_VOLTA` | `src/kernels/cuda/qsa_prompt_attn.cu` | the Volta prompt attention variant |
| `STRATA_PA_WMMA` | `src/kernels/cuda/qsa_prompt_attn.cu` | the WMMA prompt attention variant |
| `STRATA_POOL_SPIN_ZERO` | `src/kernels/cpu/pool.cpp` | the pool's spin time is zero in this build |
| `STRATA_PREFILL_FP16TC` | `src/prefill/prefill.cpp` | sets the default of the FP16 tensor-core prompt GEMMs; also read at run time |
| `STRATA_PREFILL_FUSED` | `src/prefill/prefill.cpp` | the fused prompt experts are built |
| `STRATA_PREFILL_MMQ` | `src/prefill/prefill.cpp` | builds the quantized prompt MMQ kernels; also read at run time |
| `STRATA_Q36_CPU_POOL` | `src/qwen35/q36_engine.cu` | the Qwen3.6 engine has the CPU pool |
| `STRATA_Q6K_EXPERTS` | `src/qwen35/q36_engine.cu` | the Qwen3.6 engine takes Q6_K experts |
| `STRATA_ROCBLAS_AVAILABLE` | `src/prefill/gemm.cu` | rocBLAS is available |
| `STRATA_ROWS_VNNI` | `src/kernels/cpu/q2_avx2.cpp` | the VNNI row kernel |
| `STRATA_SEL_CLUSTER` | `src/kernels/cuda/qsa_select.cu` | the cluster form of the selection |
| `STRATA_SEL_GFX11` | `src/kernels/cuda/qsa_select.cu` | the gfx11 selection variant |
| `STRATA_SEL_GFX12` | `src/kernels/cuda/qsa_select.cu` | the gfx12 selection variant |
| `STRATA_SEL_SM80` | `src/kernels/cuda/qsa_select.cu` | the sm_80 selection variant |
| `STRATA_TQ_X86` | `src/kernels/cpu/tq_cpu.cpp` | the x86 trellis kernels are built |
| `STRATA_USE_HIP` | `src/prefill/prefill.cpp` | the build is the HIP one |
| `STRATA_VERIFY_NODES` | `src/core/mtp.cpp` | builds the node listing of the draft graphs; also read at run time |
| `STRATA_W_LB` | `src/prefill/moe_fused_iq.cu` | occupancy cap of the native prompt kernel |
| `STRATA_WMMA_GFX11` | `src/prefill/wmma_gemm.cu` | set by CMake from `CMAKE_HIP_ARCHITECTURES`, never inferred |
| `STRATA_W_NO_COMB` | `src/prefill/kernels.cu` | disables the combined variant of the W kernel |
| `STRATA_W_NO_OCC` | `src/prefill/moe_fused_iq.cu` | disables the occupancy variant of the W kernel |
| `STRATA_W_NO_T16` | `src/prefill/moe_fused_iq.cu` | disables the T16 variant of the W kernel |

## How to set a variable

Set the variable in the environment of the process that starts the engine. The server passes
its own environment to the engine, so one export covers both.

Linux and macOS, one command:

```bash
STRATA_WATCHDOG_S=120 ./engine/strata generate --pack pack/full --tokens "1,2,3"
```

Linux and macOS, the whole shell session:

```bash
export STRATA_KV_GROW=1
export STRATA_PREFILL_CPU_SHARE=0
```

PowerShell:

```powershell
$env:STRATA_WATCHDOG_S = "120"
.\engine\strata.exe generate --pack pack\full --tokens "1,2,3"
```

PowerShell, only for one command:

```powershell
$env:STRATA_ROUTE_TAIL_SKIP = "8"; strata serve --port 8080
```

systemd, a drop-in for the service:

```ini
# /etc/systemd/system/strata.service.d/env.conf
[Service]
Environment=STRATA_WATCHDOG_S=120
Environment=STRATA_KV_GROW=1
```

```bash
systemctl daemon-reload
systemctl restart strata
```

The `strata_tune` object of an API request sets a few settings for that request only. The
request layer wins over the environment for that request, and the engine does not restart.
The knob names and their ranges are in `strata_control/tune_api.py` and
[SERVER.md](SERVER.md#per-request-tuning). `strata tune` is different: it measures and
writes the run config, and it sets `STRATA_NO_PRESET` in its worker, so it measures without
the saved preset.

A value the engine wrote itself, from an architecture table, counts as set. To override it,
export the variable before the engine starts: an explicit setting always wins
(`src/core/nv_defaults.cpp`, `apply_nv_defaults`).

## Related pages

- [HOW_IT_WORKS.md](HOW_IT_WORKS.md): the parts of the engine and where each setting acts.
- [ENGINE_FLAGS.md](ENGINE_FLAGS.md): the command-line flags, which take precedence over these variables.
- [TUNING.md](TUNING.md): how to measure a change on your own hardware.
- [SERVER.md](SERVER.md): the server, its API and its request settings.
- [TERMINAL.md](TERMINAL.md): the `strata` command.
- [DETAILS.md](DETAILS.md): the measured numbers behind the switches.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): what to try when a start fails or a run is slow.
