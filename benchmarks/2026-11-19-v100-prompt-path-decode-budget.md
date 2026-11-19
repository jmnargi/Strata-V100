# Dual V100-PCIE-16GB: the FP16 prompt-expert path, and where a decode window's time goes

Measured on 2026-11-18/19 on the fork's own rig. Two Tesla V100-PCIE-16GB (GPU 0 on a PCIe Gen3
**x4** link, GPU 1 on **x16**), Ryzen 5 3600 (6C/12T, AVX-2), 94 GiB RAM, one NVIDIA driver and
CUDA 12.8, `sm_70` build of this fork at source `7567e2d5` plus this PR's documentation only.

Model: Unsloth **UD-IQ3_XXS v3.1** (`d9620119`) — IQ2_S / IQ3_XXS gate/up and IQ4_NL down in a
native expert pack of 46.84 GiB (512 experts a layer), Q2_0 MTP draft layer, 524,288-token
context with yarn 2.

Engine arguments (the daily service's, from `strata-v100-pipeline.json`):

```
--pack /mnt/strata-models/packs/unsloth_v31_iq3_xxs
--native .../Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
--ple-gguf .../Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf
--mtp /mnt/strata-models/mtp/rt --expert-profile .../expert-profile-iq3xxs.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.70
--max-context 524288 --kv int8 --kv-grow --rope-scaling yarn --rope-scale 2
--vram-reserve-mib 800 --layer-split 16 --pipeline-windows 2 --ple-io mmap --vision
```

Environment (the six daily drop-ins, `service/strata-v100.service.d/`):

```
STRATA_PREFILL_CPU_SHARE=auto   STRATA_PREFILL_CPU_SHARE_MAX=6144
STRATA_PREFILL_STREAM_MIN=6144  STRATA_PREFILL_PROMOTE=2048
STRATA_PREFILL_LOAN_EXACT=1     STRATA_SPLIT_OWN_PRIMARY=1
STRATA_SPLIT_OWN_HEADROOM_MIB=256
STRATA_EMB_REUSE_ACCOUNT=1      STRATA_SM70_TABLE=1      STRATA_PF_WMMA=1
STRATA_PF_SKIP_UNROUTED=1       STRATA_KV_GROW=1         STRATA_CKPT_POOL=1
STRATA_CKPT_THREADS=1           STRATA_DEQUANT_WIDE=1    STRATA_SELECT_SIMT=1
STRATA_ROUTE_RESIDENT=1         STRATA_VERIFY_ECAP=16    STRATA_SPLIT_RING=192
STRATA_SPLIT_MTP_BATCH=1        STRATA_SM70_CHUNK=0
```

## Method

**Prompt rows.** `bench/run_v100_prefill.py`: one cold, uncached request per row through the
engine's own `--serve` protocol with exact token ids (no template, no calibration), 2K / 4K / 8K /
16K / 32K / 64K, `--max-new 128`, two repeats an arm; arms were also run **A-B-A-B** (2K/8K/32K/64K) with an
engine restart between arms, which reproduced every point to 0.2 %. Conversation reuse is structurally off (`--prompt-cache 0`,
`--conversation-cache-mib 0`). Prompt throughput is the engine's own `prompt_ms`; decode is
`generated / decode_ms` from the same `DONE` line.

**Windows (decode).** A 3,000-token prompt, 192 greedy tokens, one engine start an arm, arms in
pairs. A single sample's spread on this rig is 15 %, so only paired, repeated arms are read.

**Profile.** `nsys` on the daily service (1,024-token prompt, 256 decoded tokens = 162 verify
windows), summed over both cards and divided by the window count.

## Limits

- One rig, one model, one quant. The two cards' links differ (x4 / x16), which is not the common
  case and is why the layer split is unbalanced.
- Prompt rows are a repeated-text corpus built from the harness pool, not natural prompts; they say
  nothing about answer quality.
- Decode numbers are from the engine's own counters, not from a wall clock on the client.

## The prompt path: MMQ dp4a against the FP16 tensor-core tiles

`STRATA_PF_WMMA` picks how the prompt multiplies its routed experts: MMQ's `dp4a` integer
kernels (the engine default) or `gemm_iq_f16_grouped`, which dequantizes each weight superblock
once into shared memory and multiplies on the FP16 tensor cores with FP16 activations and FP32
sums. IQ2_S / IQ3_XXS gate/up and IQ4_NL down are covered, which is exactly this pack, so the
whole prompt path takes the second.

| prompt tokens | MMQ dp4a | FP16 tiles | prompt gain |
| ---: | ---: | ---: | ---: |
| 2,048 | 972 / 1,052 tok/s | 1,060 / 1,173 tok/s | **+10.3 %** |
| 4,096 | 1,315 / 1,343 tok/s | 1,461 / 1,474 tok/s | **+10.4 %** |
| 8,192 | 1,339 / 1,323 tok/s | 1,550 / 1,552 tok/s | **+16.5 %** |
| 16,384 | 1,707 / 1,709 tok/s | 2,023 / 2,025 tok/s | **+18.5 %** |
| 32,768 | 1,971 / 1,968 tok/s | 2,357 / 2,354 tok/s | **+19.6 %** |
| 65,536 | 2,121 / 2,124 tok/s | 2,543 / 2,549 tok/s | **+20.0 %** |

Decode on the same rows was 89-116 tok/s in both arms (128 tokens a row); the switch does not
touch the token path.

The FP16 path changes the prompt's arithmetic, so a reply can differ in its last bits from the
MMQ path's. The daily profile already sets it (`96-prefill-performance.conf`); the engine default
stays off.

## Decode: every arm measured, none adopted

3,000-token prompt, 192 greedy tokens, one engine start an arm, arms in pairs, production
profile. A single sample's spread on this rig is 15 %, so each arm is read against the base that
ran beside it in the same round.

| round | arm | decode tok/s | against its base |
| --- | --- | --- | ---: |
| 1 | production | 105.3 / 101.1 | - |
| 1 | `--pcie-mode dma` | 112.9 | +8.6 % (one sample) |
| 1 | `STRATA_ROUTE_TAIL_SKIP=9` | 115.2 | +11.3 % (one sample) |
| 1 | `--mtp-max-t 6` | 109.7 | +5.7 % (one sample) |
| 1 | `--lookup-chain 8` + `STRATA_VERIFY_ECAP=18` | 106.2 / 108.2 | +0.8 / +7.0 % |
| 1 | `--mtp-q4 head` | 97.5 | -7.4 % |
| 1 | `--mtp-q4 all` | 112.4 | +6.7 % (one sample) |
| 1 | `STRATA_VERIFY_DEVICE_PLAN=1` | 97.5 | -7.4 % |
| 1 | `STRATA_VERIFY_ECAP=20` | 97.2 | -7.7 % |
| 2 | production | 108.8 / 114.2 | - |
| 2 | `STRATA_SPLIT_RING=384` | 95.6 / 93.8 | -12.1 / -17.9 % |
| 2 | `STRATA_SPLIT_RING=512` | 109.0 / 104.8 | +0.2 / -8.2 % |
| 2 | `STRATA_EXPERT_PAIR=1` | 106.1 / 99.1 | -2.5 / -13.2 % |
| 2 | `STRATA_HC_SPLIT=1` | 102.5 / 105.3 | -5.8 / -7.8 % |
| 2 | `STRATA_MTP_BATCH_F16=1` | 110.7 / 102.1 | +1.7 / -10.6 % |
| 2 | `--spec 3` (5-row window) | 96.4 / 96.3 | -11.4 / -15.7 % |
| 3 | production | 116.2 / 101.2 | - |
| 4 | production | 118.1 / 116.9 | - |
| 4 | `STRATA_EXPDEC_MINB=4` | 114.9 / 108.6 | -2.7 / -7.1 % |
| 4 | `STRATA_EXPDEC_D20=1` | 104.5 / 114.7 | -11.5 / -1.9 % |
| 4 | `STRATA_EXPDEC_DR=2` | 107.7 / 107.3 | -8.8 / -8.2 % |
| 4 | `STRATA_EXPDEC_MINBLK=1200` | 98.4 / 102.8 | -16.7 / -12.1 % |
| 4 | `STRATA_EXPDEC_R=2` | 109.1 / 116.2 | -7.6 / -0.6 % |

Every decode arm is at or below production. Two are worth a longer sample - `--pcie-mode dma` and
`STRATA_ROUTE_TAIL_SKIP=9` each read above 110 tok/s in the round they ran - but neither is above
production in a second sample, so nothing here is adopted.

`--spec 3` is 11-16 % slower: the draft chain's length is not where the window's time goes, and the
shorter chain loses more acceptance than it saves. `STRATA_VERIFY_ECAP=14` refuses to start at all
("cannot serve a 6-token window: it keeps 4 of the tail's 50 picks"), so 16 - the profile's value -
is the tightest cap that runs.

## The prompt path: other knobs, all neutral

8,192 and 32,768 tokens, one run an arm, production profile:

| arm | 8,192 tok/s | 32,768 tok/s |
| --- | ---: | ---: |
| production | 1,488 | 2,348 |
| `STRATA_GDN_CHUNKED=2` | 1,484 | 2,337 |
| `STRATA_GDN_CHUNKED=4` | 1,482 | 2,343 |
| `STRATA_SM70_CHUNK=1` (16,384-row chunk) | 1,487 | 2,346 |
| `STRATA_PREFILL_PROMOTE=4096` | 1,483 | 2,345 |
| `STRATA_ATTN_MERGE_V2=1` | 1,487 | 2,348 |

With the FP16 tiles in place a 16,384-row prompt chunk no longer costs anything (it measured 7.5 %
slower against MMQ), but it buys nothing either, so `STRATA_SM70_CHUNK=0` stays.

## Two opt-in switches that do not work on Volta

Both are off by default, and the daily profile does not set either.

- **`STRATA_PROMPT_ATTN_IMMA=1`** (the INT8-MMA prompt attention kernel) **hangs at start-up.**
  The engine's log stops at `GPU 0: Tesla V100-PCIE-16GB, compute capability 7.0` and the process
  never reaches `READY` (five minutes, killed). Volta has no `s8` MMA (`mma.m8n8k16.s8` is sm_75+),
  so there is nothing for that kernel to bind to. This is the one part of 1Cat-vLLM's Volta work
  that does not carry over: they prepack weights in MMA fragment order and use `mma.m8n8k4` at fp16,
  or dp4a at int8, never int8 tensor cores.
- **`STRATA_PA_V70_PREFETCH=1`** (the register prefetch for int8 prompt attention, merged
  2026-11-18) **closed the engine on a 32,768-token prompt** in the production profile: the prompt
  row ended with `EOFError: the engine closed its stdout`, and the engine log ends immediately after
  `strata: K/V on CUDA1 grown to 40960 cells ...` with no error line. The 8,192-token row of the
  same arm completed (1,493 tok/s, i.e. no gain). Its own measurement notes already said the switch
  did not move whole-model throughput; this run says it is not safe to enable on this configuration.

## `STRATA_PF_WMMA_LEAN=1`: +4 % at 8K, +2 % at 32K

On the direct WMMA plan the prompt's experts read `mixed_h` by `src_dev`, so the MMQ-only prompt
buffers (`Xs` = `T*K*N` uint16, `Hh` = `T*K*640` uint16, and the dq ring) are never touched. The
switch stops allocating them, and the expert cache keeps the VRAM instead. At an 8,192-row chunk
that is ~420 MB more cache, which is why the gain is largest where the prompt streams the most
experts. It changes no arithmetic: the buffers are unread on that plan, and every row's first
generated token id matched across the four arms.

A-B-A-B, one engine start an arm, `--max-new 64`, no cool-down gate between arms (same-day):

| prompt tokens | profile | + `STRATA_PF_WMMA_LEAN=1` | |
| ---: | ---: | ---: | ---: |
| 2,048 | 1,071 / 1,041 tok/s | 1,051 / 1,080 | +1.3 % / +3.8 % (mean 1,056 vs 1,056: neutral) |
| 8,192 | 1,523 / 1,508 | 1,572 / 1,580 | **+3.9 % / +4.8 %** |
| 32,768 | 2,350 / 2,326 | 2,396 / 2,383 | **+2.0 % / +2.4 %** |
| 65,536 | 2,535 / 2,504 | 2,574 / 2,550 | **+1.5 % / +1.8 %** |

Adopted in `service/strata-v100.service.d/96-prefill-performance.conf`.
