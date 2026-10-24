# OrcaRouter Uncensored Q4_K_S（手动 setup）
<a id="orcarouter-uncensored-q4_k_s-manual-setup"></a>

这条路径直接使用 `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` 原有的三个 Q4_K_S 分片，不改写其中专家层（expert）、embedding 或 PLE 表的字节。它独立于安装程序的模型菜单。三个分片的文件名要放在一起。

普通 GGUF 会量化掉一些小投影，而 Strata 把这些投影当 BF16 读。`--compat-bf16` 会在单独的模型包（pack）里把那些投影反量化并舍入到 BF16；它不会恢复它们原来的精度。Q5_0 的 PLE 表仍留在源 GGUF 里，目前走映射读取（`--ple-io mmap`）。Q4_K 的 gate/up 专家层和 Q5_0/Q5_1 的 down 专家层保持 GGUF 形式，包括那三个张量跨越分片边界的层（每个角色从自己的分片读取）。GPU 的专家内核接受 Q4_K 配 Q5_1（或 Q8_0）的 down；Q5_0 的 down 还不在其中，遇到带这种层的模型包，引擎会拒绝启动（并指出是哪一层）。

## 构建与准备
<a id="build-and-prepare"></a>

用 CUDA 构建 Strata，启用原生专家层和 `STRATA_ORCA_Q4KS_MMQ=ON`（要跑下面两个一致性程序还需 `STRATA_BUILD_TESTS=ON`）。这个可选项为 CUDA 或 HIP 编译 Q5_0 的 GGML MMQ 实例（Q4_K 和 Q5_1 实例随引擎在 CUDA 上的 K-quant MMQ 构建一起提供）；通常的 Q8_0 草稿层实例在两种构建里都有。HIP 还需要 `STRATA_PREFILL_MMQ=ON`。选择你显卡的计算能力，例如 L40S 上是 `89`。在 Windows 上用 Ninja 和 CUDA 12.6 时，请使用 Visual Studio 开发者环境，并加 `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`，以保持 CUDA runtime 的链接方式一致。项目构建支持默认固定的 ggml checkout。

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_ORCA_Q4KS_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strata native_expert_parity ple_q5_parity
python tools/iq_pack.py --gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf --out packs/orca-q4_k_s --compat-bf16
```

`iq_pack.py` 需要 NumPy，以及来自固定 llama.cpp 依赖的 `gguf-py` 模块（`STRATA_GGUF_PY` 可以指向该目录）。它会写出约 1.43 GiB 的 dense 模型包和一份原生专家清单。112 GB 的原始模型分片被原地读取；`experts.bin` 是可选的，这条命令不会写它。

在启动服务之前，先用真实文件校验改动过的内核：

```sh
build/native_expert_parity /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00002-of-00003.gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00003-of-00003.gguf 0 5 6 20 47
build/ple_q5_parity /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf
```

## 引擎参数
<a id="engine-flags"></a>

把同一个首分片同时传给 `--native` 和 `--ple-gguf`；Strata 会按文件名找到另外两个分片。第一次纯文本运行时，用 32K 上下文和适中的缓存。跑通之后，再在目标机器上实测内存，逐步加大上下文和缓存。

```sh
--pack packs/orca-q4_k_s --native /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf --ple-gguf /models/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf --ple-io mmap --expert-profile data/expert-profile.bin --expert-cache auto --prefill 512 --spec 4 --mtp /path/to/mtp/rt --max-context 32768 --kv int8
```

要让两块 GPU 以持久服务模式运行，加上 `--layer-split auto --split-device 1`（并让 CUDA0 和 CUDA1 可见）。常规的 Strata MTP runtime 由原模型的草稿层构建；Uncensored 微调上的草稿接受率必须实测。请保留该模型自己的 tokenizer（从模型包里取）。在原生模型包上，`--expert-cache-per-layer` 让每层拥有相同数量的槽位，每个槽位的大小是该层自己数据块的大小。`--expert-cache N` 仍然是 N 个最大数据块的预算。

这次移植并不声称 Q4_K_S 在某台电脑上胜过 llama.cpp。比较时请用相同的模型字节、提示词、上下文、输出长度和 GPU 负载；prefill 和 decode 分开报告。
