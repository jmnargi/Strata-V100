# Strata documentation

This directory holds all Strata documentation. The pages describe engine version
0.1.45.8 and the server in `serve/`.

## Start here

Read these pages in this order:

1. [README](../README.md) - what Strata is, and what it does on this hardware.
2. [INSTALL.md](INSTALL.md) - install, start, update and configure.
3. [MODELS.md](MODELS.md) - which model to use, and which one fits your PC.
4. [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - how the engine fits a large model in a small PC.
5. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - when something goes wrong.

## Reference

These pages are the complete reference. Each setting appears with its default and its
effect.

| Page | What it answers |
|---|---|
| [ENGINE_FLAGS.md](ENGINE_FLAGS.md) | Every engine command-line flag: value, default, effect. |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Every environment variable the engine, server and terminal read. |
| [SERVER.md](SERVER.md) | How to start the API server: options, run config keys, endpoints. |
| [TERMINAL.md](TERMINAL.md) | Every `strata` command and subcommand. |
| [TUNING.md](TUNING.md) | How to make it faster: settings by goal, optimizations, measured and rejected work. |
| [GLOSSARY.md](GLOSSARY.md) | The terms the docs use, and where each one is explained. |
| [STYLE.md](STYLE.md) | The writing rules for these pages. |
| [DETAILS.md](DETAILS.md) | All measured numbers, the request formats, and the settings history. |

## By task

| Task | Page |
|---|---|
| Install on Windows, Linux, or Mac | [INSTALL.md](INSTALL.md), [AI_SETUP.md](AI_SETUP.md) |
| Choose a model or a quantization | [MODELS.md](MODELS.md), [QUANTS.md](QUANTS.md) |
| Run on a Tesla V100 or Titan V | [NVIDIA_V100.md](NVIDIA_V100.md), [OLDER_GPUS.md](OLDER_GPUS.md) |
| Run Inkling 2 Mini on V100 | [INKLING2_V100.md](INKLING2_V100.md) |
| Run Hy4-Flash Maya-S on V100 | [NVIDIA_V100.md](NVIDIA_V100.md#hy4-flash-maya-s) |
| Run on two or more NVIDIA cards | [MULTI_GPU.md](MULTI_GPU.md), [SECOND_GPU.md](SECOND_GPU.md) |
| Serve several requests at once | [BATCHING.md](BATCHING.md) |
| Run NVFP4 experts | [NVFP4.md](NVFP4.md) |
| Run Gyro trellis quants | [GYRO.md](GYRO.md) |
| Run Unsloth K-quants | [QUANTS.md](QUANTS.md), [UNSLOTH_Q4.md](UNSLOTH_Q4.md), [UNSLOTH_Q6.md](UNSLOTH_Q6.md) |
| Run Qwen3.6-35B-A3B | [QWEN36.md](QWEN36.md) |
| Run on AMD Radeon | [AMD_HIP.md](AMD_HIP.md), [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) |
| Run on Intel Arc | [INTEL.md](INTEL.md), [INTEL_ARC.md](INTEL_ARC.md) |
| Run on Strix Halo | [STRIX_HALO.md](STRIX_HALO.md) |
| Read images | [IMAGES.md](IMAGES.md) |
| Make images | [IMAGES.md](IMAGES.md) |
| Let an AI assistant install and run Strata | [MCP_SERVER.md](MCP_SERVER.md), [AI_SETUP.md](AI_SETUP.md) |
| Run behind llama-swap | [LLAMA_SWAP.md](LLAMA_SWAP.md) |
| Check what Strata downloads | [SAFETY.md](SAFETY.md) |
| Report a benchmark or ask for a test | [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md), [TEST_REQUESTS.md](TEST_REQUESTS.md) |
| Monitor a server with Prometheus | [monitoring/METRICS.md](monitoring/METRICS.md) |

## Optimization pages

Each page below explains one opt-in optimization, its switch, and its measurement.
Read the switch first, then the measurement.

| Page | What it covers |
|---|---|
| [EXPERTDEC.md](EXPERTDEC.md) | The verify window's routed experts read the group's activations from shared memory. |
| [QSAROWS.md](QSAROWS.md) | The decode window's QSA scores stay in registers. |
| [BATCHED_DMA.md](BATCHED_DMA.md) | Expert uploads of one verifier group as one batched copy. |
| [KV_PREFETCH.md](KV_PREFETCH.md) | Streamed KV uploads overlap the prefill of the layer that needs them. |
| [PREFILL_HEAD_ALIAS.md](PREFILL_HEAD_ALIAS.md) | The prompt read borrows the output head's VRAM. |
| [VRAM_ELASTIC.md](VRAM_ELASTIC.md) | Give VRAM back to other programs between requests. |
| [DISJOINT_EXPERT_CACHE.md](DISJOINT_EXPERT_CACHE.md) | No expert is in two caches on the same PC. |
| [EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md) | The resident RAM tier swaps an expert without the extra host copy. |
| [MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md) | The interleaved small-batch matvec, and how to measure the table for a card. |
| [PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md) | One extra checkpoint near the end of a prompt. |
| [MESSAGE_BOUNDARY_CACHE.md](MESSAGE_BOUNDARY_CACHE.md) | Reuse a long history when the last user message changes. |
| [LOGIT_BIAS.md](LOGIT_BIAS.md) | Per-request token biases. |

## Operations and internals

| Page | What it covers |
|---|---|
| [CLOSING.md](CLOSING.md) | What ends the engine, and when. |
| [MANAGEMENT_PROTOCOL.md](MANAGEMENT_PROTOCOL.md) | The local management bridge that the Desktop app and the terminal share. |
| [SECOND_GPU.md](SECOND_GPU.md) | CUDA1 to CUDA3 as expert-cache tiers. |
| [RESEARCH_RUNS.md](RESEARCH_RUNS.md) | One long document, many questions: the state plan. |
| [THIRD_PARTY_QUANTS.md](THIRD_PARTY_QUANTS.md) | A PLE table that is not IQ4_NL. |
| [ORCA.md](ORCA.md), [ORCA_Q4_K_S.md](ORCA_Q4_K_S.md) | OrcaRouter checkpoints. |
| [A770_SPEED_SPEC.md](A770_SPEED_SPEC.md) | The Arc A770 decode objective and its criteria. |

## Measurement logs

These pages record measurements as they were taken. They keep their original text and
their original numbers. A log is evidence, not current advice: read the build and the
hardware in each table before you compare it with a run of your own.

- [NVIDIA_V100.md](NVIDIA_V100.md) - the Volta build, the two-V100 service profiles, and their numbers.
- [DETAILS.md](DETAILS.md) - the long measurement and settings record.
- [benchmarks/](benchmarks/) and [../benchmarks/](../benchmarks/) - benchmark data and reports.
- [../improvements/](../improvements/) - the upstream-sync and release selection records, from
  v0.1.40.2 to 2026-10-19, with the changes that were taken and the ones that were left out.

## Conventions

- The English pages are authoritative. The root README files (`README.md` and the six
  translations) mirror each other one to one: the same sections, the same tables, the same
  numbers, translated. [zh_CN/](zh_CN/) holds translations of the pages in this directory
  and can lag behind them.
- `--flag value` is an engine or server option. `NAME=value` is an environment variable.
  `+N tokens` or `tokens/s` is the measured value.
- A number belongs to the hardware in its table. Do not compare a number from one table
  with a number from another table unless both tables name the same build and the same
  hardware.
- "Measured" means a run of the code. "From the code" means a reading of the source,
  with the file named. Tests do not replace a measurement.
- The pages follow [STYLE.md](STYLE.md).
