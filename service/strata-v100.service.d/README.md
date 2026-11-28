# The measured dual-V100 service profile

These drop-ins define the fork's dual-V100 service profile. Install them under
`~/.config/systemd/user/strata-v100.service.d/` only after you install the matching engine.
They contain no credentials or private paths. Keep the model paths, API key, and
conversation cache in the private configuration and `.strata-service.env`.

| File | What it selects |
| --- | --- |
| `90-cpu-prefill.conf` | the CPU-assisted prompt staging (`STRATA_PREFILL_CPU_SHARE=auto`) |
| `95-v0141-profile.conf` | the v0.1.41 profile: the CPU prefill share, the staged-chunk limit, the embedding-buffer accounting and the Volta master switch (`STRATA_SM70_TABLE=1`) |
| `96-prefill-performance.conf` | FP16 tensor-core experts, skipped unrouted experts, elastic KV growth, pooled checkpoint buffers, parallel stage checkpoint saves, exact wide dequantization, up to 2048 expert-cache admissions per stage and input loan, the first card's own prompt buffers, tiled FP32 QSA selection, the measured small-read crossover, exact fresh-segment loans, and one batch per native CPU layer call (`STRATA_POOL_DEP=1`) |
| `97-resident-routing.conf` | the residency-biased routing (`STRATA_ROUTE_RESIDENT=1`); it can change the model's answers |
| `98-verify-ecap.conf` | the verify window's expert cap (`STRATA_VERIFY_ECAP=16`, about +11 % decode); it can accept a draft the exact model would reject |
| `99-dual-gpu.conf` | the two cards' layer split: the streamed ring, split draft K/V through the prompt path, and the 8,192-token prompt chunk |
| `99-hc-v100.conf` | the K-split hyper-connection read (`STRATA_HC_V100=1`) with the fused variant off (`STRATA_HC_V100_FUSED=0`); the rounding differs, so greedy text can differ |

`STRATA_POOL_DEP=1` is bit-exact (`pool_tasks_test` compares it with the two-phase batch) and
measured at -1.76 % prompt time on 257-2,049-token reads, the daily agentic shape
([NVIDIA_V100.md](../docs/NVIDIA_V100.md#one-batch-per-native-cpu-layer-call-strata_pool_dep-2026-11-21)).
`STRATA_HC_V100_FUSED=0` is also the engine default since 0.1.45.14: the fused read's grid-wide spin
needs all 160 blocks resident and hangs a window when another stream shares the card.

Install them with:

```sh
install -d ~/.config/systemd/user/strata-v100.service.d
install -m 644 service/strata-v100.service.d/*.conf ~/.config/systemd/user/strata-v100.service.d/
systemctl --user daemon-reload
systemctl --user restart strata-v100.service
```

`STRATA_SM70_TABLE=1` is the engine's own default on a compute capability 7.0 card since the Volta defaults
table (`src/core/nv_defaults.cpp`); the line stays here so the profile is explicit. `STRATA_SM70_PERF=0`
turns the table off.

The measurements behind these switches, and the ones that did **not** help on this rig, are in
[the Volta report](../bench/results/2026-10-20-v100-sm70-speed-stack/README.md).

The checkpoint settings were measured with the deferred input-anchor engine.
The engine copies the input anchor to host memory after the first output window.
This reduces time to first output. It does not remove the copy from the request.
See the [incremental TTFT report](../../benchmarks/v100-incremental-ttft-2026-10-21.md)
for the A/B results and limits. The test did not install this revised profile in
the daily service.

The revised profile also selects `STRATA_DEQUANT_WIDE=1` and
`STRATA_PREFILL_PROMOTE=2048`. Wide dequantization preserves the narrow
kernel output bit for bit. Cache admission changes expert placement.
Resident-aware routing and CPU/GPU rounding can then change an answer.
Set `STRATA_PREFILL_PROMOTE=0` to keep the static cache refill policy.

Cache admission requires a full host arena. It is off with batch slots,
a RAM complement, a peer or helper tier, `STRATA_PREFILL_HELP`, or a
prefill head alias. The engine logs the reason for an explicit refusal.
Do not combine these paths or install this profile with an older engine.
The cache size, KV capacity, KV format, and model quantization stay unchanged.

See the [cache admission report](../../benchmarks/v100-prefill-cache-admission-2026-10-21.md)
for the four-arm comparison and full-request limits.

The 2026-10-25 profile also selects `STRATA_SPLIT_MTP_BATCH=1` and
`STRATA_SELECT_SIMT=1`. Both remain opt-in in the engine. On the two
V100 cards, tiled selection improved cold prompt throughput by about
5.3% at 64K and 12.2% at 128K, with batched MTP already enabled.
It was effectively neutral at 8K. The scorer uses a different FP32
summation order; near-tied scores can select different attention cells.
This is not an output-equivalence or decode-speed claim.
See [the measured V100 results](../../docs/NVIDIA_V100.md#tiled-prompt-selection).
The lean-buffer option stays off because of its short-prompt tradeoff.

For short prompts, the profile keeps automatic CPU-assisted prefill below
6144 read rows (`STRATA_PREFILL_STREAM_MIN` and
`STRATA_PREFILL_CPU_SHARE_MAX`). The streamed path remains in use above
the cutoff. `STRATA_PREFILL_LOAN_EXACT=1` prices a fresh single-chunk
loan at its actual read length instead of rounding it up across that
cutoff. Planned continuations and fixed head-alias layouts are unchanged.
Install the matching engine before these settings. An older engine
without exact loans can borrow an unused larger ring near the cutoff.

In the measured 1K–10K comparison, prefill throughput improved by 16.5%
at 3K, 23.7% at 4K, and 8.3% at 5K. The other sizes were roughly
unchanged. These are additional gains over batched MTP and tiled
selection. CPU/GPU placement can change answers; there is no
bitwise-output, model-quality-equivalence, or decode-speed claim.
See [the small-prompt results](../../docs/NVIDIA_V100.md#small-prompt-prefill).
