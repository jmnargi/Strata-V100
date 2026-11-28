# A CUDA 13 engine on a V100: volta-forever

CUDA 13 cannot compile for Volta. A CUDA 13 engine is built for compute capability 7.5 or newer, and
[volta-forever](https://github.com/nbald/volta-forever) (`vf`) retargets that image's PTX to sm_70 when the module
loads, so a V100 runs it. Strata's ready-made CUDA 12 engine still runs natively on the card and is the default for it
([OLDER_GPUS.md](OLDER_GPUS.md)); this page is for a CUDA 13 toolkit only, or for a machine that wants to run the
CUDA 13 engine on its Volta card.

Two things go wrong without `-DSTRATA_VOLTA_PTX=ON`, and both are the same cause: the image was compiled for a Turing
card, so it holds Turing's instructions and not Volta's.

- **The Volta kernel bodies are traps.** The prompt attention's `mma.m8n8k4` kernel, the shorter `__dp4a` dots of the
  i-quants and the shuffle-based `ldmatrix` replacements are selected by `__CUDA_ARCH__`. In a 7.5-only image that code
  is a `__trap()` whose comment says "the host only launches this kernel on sm_70". Under volta-forever the host does
  launch it - the card is a V100 - and the engine dies on the first prompt (`dequant launch: unspecified launch
  failure`; `VF_FAULT_TRACE=1` names the kernel).
- **The kernels that do run are emulated.** Without the flag the engine takes its Turing paths: `ldmatrix`,
  `cp.async`, `mma.m16n8k8` and int8 tensor-core products. A V100 has none of them, so the layer computes them with
  other instructions, and it says so: `128 FP16 tensor-core products, 2048 integer tensor-core products, 64 ldmatrix`
  emulated.

## The build

```sh
cmake -S . -B build-vf -DSTRATA_ENABLE_CUDA=ON -DSTRATA_VOLTA_PTX=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc
cmake --build build-vf --target strata -j
```

`STRATA_VOLTA_PTX` does three things:

- It compiles the Volta kernel set from the same sources (`STRATA_EXPERIMENTAL_SM60`'s kernels) and defines
  `STRATA_VOLTA_PTX`, which makes `STRATA_VOLTA_ISA` / `STRATA_PRE_TURING_ISA`
  ([include/strata/kernels/volta_isa.hpp](../include/strata/kernels/volta_isa.hpp)) choose the Volta sequences for the
  sites that otherwise test `__CUDA_ARCH__`: `mma.m8n8k4` in the prompt attention, two `__dp4a` instead of three in
  the i-quant dot, shuffles instead of `ldmatrix`, plain copies instead of `cp.async`.
- It leaves the image as **PTX only** (`<arch>-virtual`) and warns nothing: those Volta instructions are not valid
  for a ptxas that targets sm_75, so nvcc stops at PTX and the layer's own ptxas (sm_70) assembles them when the
  module loads.
- It patches the llama.cpp sources the engine fetches ([cmake/strata_volta_ptx_ggml.cmake](../cmake/strata_volta_ptx_ggml.cmake))
  so ggml's MMQ kernels take their **dp4a** path instead of Turing's int8 tensor cores and `ldmatrix`. Without it the
  layer emulates those kernels (512 integer tensor-core products and 32 `ldmatrix` per `mul_mat_q` instance), and one
  of them was measured to fault (`mul_mat_q<GGML_TYPE_Q8_0, 64, false>`, `CUDA_ERROR_ILLEGAL_ADDRESS` on the first
  prompt read).
- It defines `STRATA_HOST_VOLTA_ISA` for the host side, so the host only selects a Volta kernel where the image
  really holds one. A sm_75-only build (no flag) run on a V100 - through the layer, or with `STRATA_EMULATE_CC=70` -
  then keeps the generic kernel instead of launching a trap.

## Running it

```sh
vf build-vf/strata --pack ... --native ...      # the same arguments the CUDA 12 engine takes
```

The first start after a build spends a minute or more compiling the image's PTX (once per build: the layer caches
it). Give the engine time: `STRATA_WATCHDOG_S=900` on that start. `CUDA_MODULE_LOADING=LAZY` is what the layer wants
(EAGER translates every kernel of the binary before the banner prints); Strata sets it itself when it detects the
layer.

## Measurements

Two Tesla V100-PCIE-16GB, IQ3_XXS native pack, this repository's daily config and drop-in environment, greedy, exact
token counts from `bench/run_v100_prefill.py`. "Native" is the installed sm_70 engine with CUDA 12.8. Each row is the
median of two prompts of that size; the table's last column is the same engine behind `vf` (its kernels run as they
were compiled, no retargeting) and shows what the layer itself costs.

| prompt tokens | native sm_70, CUDA 12.8 | sm_75 build, no flag (emulated) | sm_75 build with `STRATA_VOLTA_PTX` |
| ---: | ---: | ---: | ---: |
| 2,048 | 1,991 ms | 2,846 ms | 1,794 ms² |
| 8,192 | 4,908 ms | 5,624 ms (1.15x) | 4,880 ms (0.99x) |
| 16,384 | 7,481 ms | 8,821 ms (1.18x) | 7,435 ms (0.99x) |
| 32,768 | 12,890 ms | 15,542 ms (1.21x) | 12,829 ms (1.00x) |
| 65,536 | 24,307 ms | 31,113 ms (1.28x) | 23,840 ms (0.98x) |

² the first request in a fresh process pays the layer's compilation of the kernels that size uses first; the second
request of the same size is the number quoted (at 16K: 8,982-9,043 ms first, 7,477-7,495 after, against the native
engine's 7,533-7,820). It is per size, once per process, and it does not depend on `VF_LOG_LEVEL`.

Decode (an 8K prompt, 128 greedy tokens, the same six prompts per arm, paired): native 99.3 tok/s, `STRATA_VOLTA_PTX`
100.5 tok/s, per-prompt ratios 0.87-1.12 (the draft acceptance moves more than the layer does). The naive sm_75 build
read the same six prompts at 86.2 tok/s against the native 103.3 of that session (0.84x).

The 8K and 64K replies were the native engine's, first token included: the flag costs no accuracy, where the naive
sm_75 build diverged at 32K (its first token was `16401`, the native engine's `279`).

**What the flag is worth, on two Tesla V100-PCIE-16GB with the IQ3_XXS native pack and this repository's daily
config:** a CUDA 13-style engine goes from 15-35% slower than native to native speed (0.98-1.00x prefill, decode
within noise). It does not make the card faster than the CUDA 12 engine - the layer adds no throughput of its own; it
removes the penalty of a build that was compiled for another card.

### What was measured and found neutral or worse

The same two cards, the same prompts, alternating arms in one session:

| tried | result |
| --- | --- |
| a newer code generator: the engine built `70-virtual` and JIT-compiled by the 580 driver, against the `sm_70` SASS ptxas 12.8 makes | 7,724 / 7,742 ms at 16K against 7,734 / 7,738 - neutral |
| CUDA 12.9's cuBLAS in front of the native engine (`libcublas.so.12` 12.9.2.10) | 7,748 / 7,746 ms at 16K, neutral; one 64K session with it degraded over repeats (24.8 -> 37.1 s), cause not established |
| `STRATA_PREFILL_AUTO_MAX=16384` (the `--prefill auto:16384` of #282) | 32K 13,757 / 13,879 ms against 13,140 / 13,238 (4.7% slower); 64K neutral |
| `--mtp-hnorm stream` | decode, 6 prompts x 2 rounds: 1.01x / 0.96x paired medians, inside the acceptance noise |

The layer is a passthrough once the image carries the card's own instructions: it costs nothing to use and adds
nothing. What a prompt read costs on this machine - the expert stream over PCIe and from the SSD, and the serial
stage pipeline - is not something the layer touches.

## Limits

- A prompt read of 64K tokens streams most of the expert pack from the SSD on this machine, so it is disk-bound and
  swings between runs; read it as "slower", not as a factor.
- The layer compiles the whole image at the first load. With an EAGER module-loading setting that happens before the
  banner and looks like a hang; Strata defaults to LAZY when it sees the layer.
- Under volta-forever the engine treats its cards as Volta (the layer only exists to run Volta images). A machine
  that also has a newer card behind the same layer is not a configuration this build supports.
- Not measured here: a real CUDA 13 toolkit build. The numbers above are a CUDA 12.8 build of the same sources with
  `75-virtual`, which is the same PTX; a CUDA 13 build additionally enables the `cudaMemcpyBatchAsync` expert uploads
  ([dma_batch.hpp](../include/strata/core/dma_batch.hpp), `CUDART_VERSION >= 13000`), which no measurement here
  covers.
- The ik2 and inkling2 kernel families took the same switch mechanically. Their models were not run.
