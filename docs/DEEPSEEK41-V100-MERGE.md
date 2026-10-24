# DeepSeek-V4.1 support in the V100 fork

This page records the merge of `upstream/main` (version 0.2.4.2) into the V100 fork, and the V100
performance checks after the merge.

## What the merge adds

Upstream added a second engine for DeepSeek-V4.1-Flash. The merge brings these parts into the fork:

| Part | Path |
| --- | --- |
| DeepSeek engine (CUDA and HIP) | `src/dsv41/`, `include/strata/dsv41/` |
| Model pack tool | `tools/dsv41/pack_gguf.py` |
| Server paths and tests | `serve/test_dsv41.py`, `serve/fixtures/deepseek41/` |
| Expert profiles | `data/dsv41/profile-2.00.bin`, `profile-2.25.bin` |
| Benchmark command | `strata bench run`, `strata bench report`, `strata bench compare` |
| Documentation | `docs/DEEPSEEK41.md`, `docs/BENCHMARK_PROTOCOL.md` |

The engine builds as the `strata-dsv41` target when `-DSTRATA_BUILD_DSV41=ON` is set.

## What the merge keeps

The fork's V100 work stays in the engine. The conflicts were resolved as follows:

| File | Kept from the fork | Added from upstream |
| --- | --- | --- |
| `src/kernels/cuda/iq_kernels.cu` | pair-kernel launches, rows-per-block grids | trellis (TQ2/TQK) staged paths, `tq_sub8` gate/up branch |
| `src/kernels/cpu/pool.cpp`, `pool.hpp` | submitter affinity, `worker_cpu` | NUMA slices, `cpu_lowpower` E-core set, `placement_line` |
| `src/kernels/cpu/iq_avx2_rows.inl` | ggml bit-exact order for IQ3_XXS/IQ3_S/IQ2_S | `EX` path for the DeepSeek CPU lane |
| `src/program/generate.cpp` | startup VRAM plan, sm70 16384-token prefill raise, idle prefill | MTP pre-allocation, DFlash2 accounting, expert-cache sizing |
| `serve/server.py` | idle prefill worker | `declared_only` parameter, encoder directory recovery |
| `CMakeLists.txt` | fork source list | `dflash2.cpp`, DSV41 and gfx906 targets |

The fork removed the desktop app, the host app, the packaging scripts, the Metal backend and the
Terminal documentation. These paths stay removed. Two files come back because the merged Terminal
command imports them: `strata_cli/bench.py` and `strata_control/benchmarking.py`.

## Build

```
cmake -S . -B build-merge -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_BUILD_TESTS=ON
cmake --build build-merge -j10
```

The build is clean on this PC: CUDA 12.8, `sm_70`, both targets `strata` and `strata-dsv41`.

## Tests

`ctest --test-dir build-merge -j6` passes all tests except `ple_parity`. That test needs the Q2_0
PLE shard, which is not on this PC. The failure is environmental, not a code failure.

## V100 performance after the merge

The daily IQ3_S profile runs on two Tesla V100-PCIE-16GB cards. Both engines ran the same profile,
one after the other, with the same pack and the same prompts.

| Measurement | Fork main (deployed) | Merged engine | Delta |
| --- | ---: | ---: | ---: |
| Decode, 256 tokens | 74.0 tok/s | 72.9 tok/s | -1.5% |
| Prefill, 2464 tokens | 647.1 tok/s | 648.2 tok/s | +0.2% |
| Drafts accepted | 135 of 179 | 134 of 171 | - |
| Expert-cache hit | 92.4% | 91.8% | - |

The decode difference is inside the run-to-run variation of the MTP drafter. The prefill rate is
equal. The merge does not remove the V100 optimizations.

## Limits of this check

The DeepSeek engine is compiled and unit-tested only. A run needs the smallest published set:
121 GB of model, 111 GB of Engram table and 8 GB of drafter (239 GB of disk). This PC has 46 GB
free on the model NVMe. Upstream also lists RTX 30 and A100 as the supported cards, so a V100 run is
not validated and no speed number exists for it.
