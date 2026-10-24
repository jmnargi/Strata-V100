# Gyro（agentionai 的 rotor 量化）在 Strata 上
<a id="gyro-agentionai-rotor-quants-on-strata"></a>

Gyro-S 和 Gyro-M 是 agentionai 对 Qwen3.8-Flash-Next 做的量化
（[agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)）：被路由的
专家层以 trellis 编码块（ggml 类型 TQ2_T、TQK6、TQK7）存放在 Hadamard 旋转基下。Strata 直接读取发布出来的文件，
不做任何改动。**Setup 把它们列为独立的一个模型系列**（"Qwen3.8-Flash-Next (Gyro)"）；两者都不是任何人的默认选项。

| | Gyro-S | Gyro-M |
| --- | --- | --- |
| 文件 | `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` | `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` |
| **下载**（n-gram 表就在文件里） | **58.5 GB** | **92.0 GB** |
| **所需内存**（显存 + 内存合计） | **27.6 GB** | **35 GB** |
| 专家层（trellis） | TQK6 gate/up，TQK7 down | TQ2_T |
| 通俗地说 | 最小的选项，适合内存少的 PC；质量低于 2-3 bit 的几个尺寸 | 质量大致相当于 IQ3_XXS，GPU 内存少约 20%；在 12 GB 卡上 decode 比 Q2_0 慢约 20% |

每个文件大约一半是 n-gram（PLE）表：它留在 SSD 上，每个 token 都要逐行读取，所以请把文件放在本地的快盘上。表里的
内存数字是权重加载后占用的部分。

按实测，它们都不是默认选项（我们的评估，2026-10）：在相同的 GPU 字节数下，Gyro-M 把 KL 散度减半（128 个 chunk 上
0.159 对 0.312），与 IQ3_XXS 相当（0.157），但在 12 GB 卡（RTX 3060）和一块 R9700 上 decode 慢 17-21%，而且它的
推理题集得分低于 Q2_0（49/60 对 57/60）。Gyro-S 的 KL 比 IQ2_XS 差 13%（0.268 对 0.236），字节数少 24%。所以：
放不下更大尺寸的地方选 Gyro-S，在意比 IQ3_XXS 省 20% 内存时选 Gyro-M。Setup 和 MCP server 只在其他尺寸放不进这台
PC 时才提到它们；它们只提示，从不拒绝。Gyro-L（agentionai 的 50 GiB 文件，2026-10-12）不受支持：它用的是另一套
trellis 家族（类型 147-149），引擎没有。

## 安装
<a id="install"></a>

```sh
./setup.sh --setup --family gyro --model Gyro-S      # or Gyro-M;  Windows: START-HERE.bat --setup ...
```

Setup 会下载该文件固定的 revision（校验大小与 SHA-256），准备模型包（`tools/iq_pack.py
--compat-bf16`：Hadamard 表和 dense 权重，1.5 GB）、草稿头（Strata 自己的 MTP 头，和每个 Qwen3.8-Flash-Next 文件
一样从原模型准备）以及启动脚本。需要引擎 0.1.45.1 或更新版本：模型包是 native_experts.txt v5（Hadamard 折叠），
引擎自 APR 移植起就能读取，而发布的 0.1.45 只到 v4。现成的引擎若版本更旧，会被一个用本源码编译出来的引擎替换
（setup）；或者应用提示该模型需要更新的引擎，并提供 Update engine。

**支持：**NVIDIA（CUDA，Linux 和 Windows）与 AMD（HIP，实验性：作者验证了 gfx1151；我们在 gfx1201 上跑过，输出
与 rc2 完全一致）。**不支持：**Intel Arc（SYCL）和 Apple Metal：这两处的引擎没有 trellis 内核；Metal 引擎在加载时
拒绝该文件并给出原因。

**图片。**Gyro 用原模型的图片读取器文件来读图（每个 Qwen3.8-Flash-Next 模型用的都是同一个；setup 随模型一起下载）。
该读取器只打开 Gyro 文件的词表和头部，所以 trellis 张量对它没有影响。图片默认开启：显存低于 24 GB 时用 CPU
读取器，24 GB 起用显卡上的读取器。这需要引擎 0.1.45.1 或更新版本，和 Gyro 文件本身的要求一样。

**RTX 50 显卡（sm_120）。**用引擎 0.1.45.3 时 Gyro 在它们上面输出正确。融合的 gate/up SwiGLU 内核在 compute
capability 12.0 的卡上是默认路径，但它只处理 80 个点积的行；Gyro 的 trellis 行有 160 个，修复之前这些行照样走了
它。于是 Gyro-S 的回答会出现循环、胡话或什么都不输出：在一台 RTX PRO 5000 上，修复前 8 个测试问题答对 0 个，
修复后 8 个全对（贪心解码和用应用的采样都是）。现在这些行走全 warp 内核，identity gate（10 条提示词，4K 和 20K
提示词）与关掉融合内核的运行完全一致。如果在 RTX 50 卡上 Gyro 表现异常，请更新引擎。

**提示词速度与 ggml 构建。**trellis 的提示词内核（MMQ）是 agentionai 的，出自他们的 llama.cpp fork（MIT）。
Strata 对上游 ggml 构建，上游没有这些内核，所以它在 configure 阶段自己加进去：
`cmake/strata_ggml_tq_overlay.cmake` 把固定版本的 ggml-cuda 头文件副本写进 build 目录，并把三种 trellis 类型按
IQ2_XS 的方式路由（tile loader 用 fork 的，`third_party/ggml-tq/`）；只有 Strata 自己的 MMQ 库会看到这份副本，
其他所有类型都从未改动的文件编译（见下面的 identity gate）。现成引擎通过它以和 rc2 一样快的速度读取 Gyro 提示词，
而且逐位相同。如果固定的 ggml 变了、脚本的某个锚点不再匹配，overlay 会自行关闭并给出消息，提示词改走较慢的 FP16
路径。对 fork 构建时（`-DSTRATA_GGML_DIR=<agentionai/llama.cpp>`，rc2 就是这么做的），改用 fork 自己的实例。

## 致谢与许可
<a id="credit-and-licenses"></a>

trellis 类型、Hadamard 旋转、CPU 与 GPU 内核以及打包步骤都是 Laurent Zuijdwijk（agentionai）在 agentionai/Strata
`rc2`（9ff02c9，MIT，Strata v0.1.40.3 加 Gyro）里写的，并逐个 commit 移植到这里，保留了他的作者身份；
`third_party/ggml/ggml-common-tq.h` 和 MMQ 实例来自 agentionai/llama.cpp（MIT，ggml 作者）。权重是 agentionai 在
Qwen Community License 1.0 下对 Qwen3.8-Flash-Next 的量化
（<https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/blob/main/LICENSE>）。

## 它怎么跑（手工）
<a id="how-it-runs-by-hand"></a>

本页其余部分是作者自己写的手工安装指南，以及他们在自己机器上的测量（RTX 5090、3090、4090 和一台 Strix Halo，
跑的是 Strata rc2）。那些数字是他们的；我们的在下一节。

## 1. 构建
<a id="1-build"></a>

Setup 会替你做完这些步骤；手工做的话，从这份 checkout 开始。为了走快的提示词路径，Strata 的 GGML 依赖是
agentionai 的 llama.cpp fork，它带 trellis 类型：

```sh
git clone --depth 1 https://github.com/agentionai/llama.cpp ../agention-llama.cpp
# NVIDIA. RTX 50 series: use CUDA 13 (CUDA 12.8 crashes the prompt path on sm_120: issues #220, #224)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
# AMD (experimental: kernel parity and an end-to-end run validated on gfx1151; in a non-root container add
# --group-add $(getent group render | cut -d: -f3), otherwise HIP finds no device)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 2. 下载与打包
<a id="2-download-and-pack"></a>

```sh
.venv/bin/pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --local-dir models
.venv/bin/python tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --out packs/gyro-s --compat-bf16
build/native_expert_parity --load-check packs/gyro-s models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf
```

模型包是 1.5 GB（dense 权重、tokenizer、Hadamard 表和专家层表）；专家层从 GGUF 里读。请把 GGUF 放在本地的高速
SSD 上：n-gram 表每个 token 都要从它读取。`--load-check` 必须以 `0 failures` 结束。

## 3. 草稿头
<a id="3-draft-head"></a>

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

## 4. 起服务
<a id="4-serve"></a>

在仓库根目录下保存为 `strata-gyro-s.json`：

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/gyro-s",
    "--native", "models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "131072", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/gyro-s/tokenizer",
  "model_name": "gyro-s-strata",
  "log": "strata-gyro-s.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json --port 8080
```

网页界面在 `http://127.0.0.1:8080`；API 客户端用 `http://127.0.0.1:8080/v1`（OpenAI）或
`/v1/messages`（Anthropic）。`--expert-cache auto` 按空闲显存决定专家缓存的大小：32 GB 卡、128k 上下文时，它容纳
24,576 个专家层中的 23,464 个（22.9 GiB）。卡更小时容纳得更少，其余的从内存流式读取；其他都不变。

## 在这里测到的（本移植版，2026-10-15/16）
<a id="measured-here-the-port-2026-10-1516"></a>

一块 AMD Radeon AI PRO R9700（gfx1201，ROCm 7），四张卡中的一张，整机独占锁，`--prefill auto --spec 4 --spec-min-p 0.5
--mtp --kv int8`，22K 上下文，交替配对（先跑的那一组轮换），取中位数，用的是第一个 Gyro 发布（2026-10-07）的文件。
A = agentionai/Strata rc2 9ff02c9，B = 本移植版。

| B 用的 ggml | 文件 | decode tok/s A / B (B/A) | B 更快的配对 | prompt 4K ms A / B | prompt 20K ms A / B |
| --- | --- | ---: | --- | ---: | ---: |
| agentionai fork | Gyro-S | 85.4 / 89.7 (1.051) | 5/5 | 5418 / 5313 (0.98) | 14358 / 13859 (0.97) |
| agentionai fork | Gyro-M | 72.7 / 82.7 (1.136) | 5/5 | 5183 / 5520 (1.07) | 13421 / 14164 (1.06) |
| 上游 + overlay（现成引擎） | Gyro-S | 85.3 / 89.6 (1.050) | 5/5 | 5413 / 5298 (0.98) | 14372 / 13854 (0.96) |
| 上游 + overlay | Gyro-M | 72.6 / 82.6 (1.138) | 5/5 | 5682 / 5524 (0.97) | 15005 / 14160 (0.94) |
| 上游，无 overlay（FP16 提示词路径） | Gyro-S | 85.3 / 89.6 (1.050) | 5/5 | 5425 / 6939 (1.28) | 14364 / 21677 (1.51) |

decode 比 rc2 快，是因为移植版建在更新的 Strata（0.1.45）上；同一处引擎改动让 IQ3_XXS 快了 7%（rc2 86.9，移植版
93.1 tok/s，2 组配对）。ISTA 文件的提示词也因此快了 11-16%（IQ3_XXS 4K 从 6184 降到 5480 ms，20K 从 16587 降到
13864 ms）。Gyro-M 对 rc2 的第一次运行显示它的提示词慢 5-7%；第二次时 rc2 自己的 Gyro-M 提示词用了 15.0 s 而不是
13.4 s（92 GB 文件的 page cache 状态），移植版仍是 14.2 s，于是 Gyro-M 的提示词比 rc2 快 3-6%：那个差异是 rc2 的
baseline，不是回归。没有 overlay（纯上游 ggml）时缺少 trellis 提示词内核，提示词走 FP16 路径：比 rc2 慢 28%（4K）
到 51%（20K），而且舍入不同。decode 两种情况下都不受影响。

一致性验证（固定的 gate 参数，10 条提示词，exact_pp 4096 和 20000 个 token，两组用同一张卡）：

- Q2_0 和 IQ3_XXS，f-split 对移植版（上游 ggml）：10/10 完全一致，残差 dump 和输出 IDENTICAL。
- Gyro-S 和 Gyro-M，rc2 对移植版（分别用 fork 的 ggml 和上游 + overlay）：10/10 完全一致；exact_pp 4096 /
  20000 的残差 dump 和输出 IDENTICAL（单独运行；有一次和其他引擎同机运行时出现过不同的 dump）。
- Q2_0 和 IQ3_XXS，f-split 对开着 overlay 的移植版，在 AMD gfx1201 和 NVIDIA sm_86 上：IDENTICAL。
- 打包步骤（`tools/iq_pack.py --compat-bf16`）对 Gyro-S 写出的文件与 rc2 的工具逐字节相同。

## 测量（作者的，rc2）
<a id="measured-the-authors-rc2"></a>

所有运行：贪心解码或文中注明的采样，提示词 token 与 llama.cpp 相同，MTP 草稿 `--spec 4 --spec-min-p 0.5`，
int8 KV。「llama.cpp」指带 CUDA trellis 修复的 agentionai fork（自 2026-10-04 起在其 `main` 里）。

### RTX 5090（32 GB），Gyro-S，全部专家层在 GPU 上（2026-10-03）
<a id="rtx-5090-32-gb-gyro-s-all-experts-on-the-gpu-2026-10-03"></a>

decode tokens/s，单条 34-177 个 token 的提示词，生成 384 个新 token，`--max-context 16384`（全部 24,576 个专家层
都缓存）：

| | 散文 | JSON | 代码 |
|---|---:|---:|---:|
| Strata | 139.9 | 221.3 | 209.0 |

Server 模式，rc2 的 `gyro_setup.sh` 配置（128k 上下文，缓存 23,463 个专家层），关闭思考，temperature 0.7，
top-k 20，top-p 0.95，min-p 0.05，512 个新 token，重复 3 次：

| | 散文 | JSON | 代码 | copy |
|---|---:|---:|---:|---:|
| Strata | 131.7 | 200.1 | 195.2 | 201.8 |
| llama.cpp，MTP 草稿 | 116.8 | 213.0 | 179.0 | 220.7 |
| llama.cpp，无草稿 | 121.8 | 123.2 | 122.5 | 121.2 |

长真实文本提示词，prefill tokens/s：4k token 3,456（llama.cpp 2,520），16k token 4,883（llama.cpp 2,471）。

### RTX 5090，Gyro-M（2026-10-03）
<a id="rtx-5090-gyro-m-2026-10-03"></a>

专家层（29.9 GiB）无法和 KV 缓存一起全部放下；`auto` 缓存 24,576 个中的 18,007 个。同样的 server 协议：
散文 113.8、JSON 170.8、代码 153.0、copy 175.7 tokens/s；prefill 4k 3,369、16k 4,854 tokens/s。在 32 GB 卡上，
llama.cpp 跑这个文件需要 `-ncmoe 8`，没有草稿时 decode 70.8 tokens/s。

### RTX 3090（24 GB），2026-10-03
<a id="rtx-3090-24-gb-2026-10-03"></a>

同样的 server 协议，128k 上下文，`auto` 缓存：

| | 缓存的专家层 | 散文 | JSON | 代码 | copy | prefill 4k | prefill 16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gyro-S | 15,756 | 31.8 | 45.2 | 42.2 | 45.1 | 1,402 | 1,863 |
| Gyro-M | 11,811 | 27.8 | 39.4 | 34.4 | 37.7 | 1,363 | 1,910 |
| llama.cpp，Gyro-S，`-ncmoe 10`，32k 上下文，无草稿 | （10 层在 CPU 上） | 16.0 | 16.2 | 16.2 | 16.5 | 468 (`-ncmoe 11`) | 479 |

在 24 GB 卡上，缓存未命中要落到 CPU 上跑，所以 CPU 很关键：这台机器是 Zen 2 EPYC（AVX2，无 AVX-512）。
llama.cpp 那一行是 CUDA trellis 修复之前的构建；这些修复在 24 GB（CPU 层占主导）时带来约 3 %。

Strata 的那几行早于 Strata 为 trellis 类型自己实现的 CPU 内核（`src/kernels/cpu/tq_*.cpp`，AVX2 和 AVX-512，
运行时选择）。在它们之前，缓存未命中走的是 ggml-cpu 的 `vec_dot`，一次一个 token；新内核在 AVX2 上每个专家层只花
它约 45 % 的时间，并且对一个验证窗口的所有 token 只解码一次块（`build/tq_cpu_parity --bench` 可以在你的 CPU 上
测它们）。`STRATA_NO_TQ_KERNELS=1` 回到 ggml-cpu；`STRATA_TQ_ISA=avx2` 或 `scalar` 限制指令集（用于对比）。

### 更小的缓存（RTX 5090，2026-10-02，int8 decode 内核之前）
<a id="smaller-caches-rtx-5090-2026-10-02-before-the-int8-decode-kernels"></a>

| 专家缓存 | 散文 | JSON | 代码 |
|---|---:|---:|---:|
| 24,576 个槽位（全部专家层） | 116.4 | 194.3 | 173.5 |
| 22,000 个槽位 | 111.4 | 175.7 | 166.1 |
| 9,000 个槽位（约相当于 16 GB 卡的空间） | 81.4 | 74.0 | 96.8 |
| 3,000 个槽位（约相当于 8 GB 卡的空间） | 48.0 | 52.4 | 57.9 |

较小的缓存是在 5090 上模拟出来的（快 GPU，PCIe 5.0 x16，约 51 GB/s）。真实的 8 GB 或 16 GB 卡 GPU 更慢，链路
往往也更慢：这些行只当「仍然可用」来看，不要当成对某张具体卡的预测。decode 出的输出被读过，通顺且正确；与
llama.cpp 的 token 级一致性尚未核对。128k 上下文的真实对话通过网页界面跑通了。

### 两块 GPU（2026-10-04，2× RTX 4090，无 NVLink/P2P）
<a id="two-gpus-2026-10-04-2-rtx-4090-no-nvlinkp2p"></a>

- **层切分**（Strata 自动把层分到两张卡上）在 Gyro 上可用，包括空闲卡上的提示词助手
  （`STRATA_PREFILL_HELP=1`）。重复运行输出一致。
- **用第二块卡 decode**（`--peer-device 1` 或 `--expert-cache-device1 auto`）可用，且与单 GPU 输出一致。
- **在对等卡上处理提示词**需要 GPU 间直连（NVLink 或数据中心 GPU）；没有它，Strata 把提示词留在第一块 GPU 上，
  并记录 `no P2P`。

单块 RTX 4090 的快速验证（Gyro-S，MTP 草稿，贪心解码）：代码 107.5、散文 84.3、JSON 76.3 tokens/s。

## 数值处理
<a id="numerics"></a>

- **decode**（Strata 自己的专家层内核）通过码本的一份 int8 副本、配一个全局 scale 来读取 trellis 权重
  （`tq_lut_i8`，相对 RMS 误差为码本标准差的 0.85 %，远低于量化误差本身），对 q8_1 激活值计算。
- **提示词处理**则用另一种方式解码它们：FP16 路径走 fp16 码本（`tq_lut_f16`，GGML 自己的反量化器读的就是它）；
  MMQ 路径走 llama.cpp fork 的内核，在 fork 当前的 `main` 上它们读的也是 int8 码本。
- 所以 decode 与提示词的结果在容差内一致，而不是逐位一致。`native_expert_parity` 把 decode 内核上的一个专家层
  与 GGML 的 float 反量化器及 float SwiGLU 对比：相对 L1 误差低于 3e-2，这是那里每种专家层类型都能达到的界限。
  它的「bitwise equal」几行比较的是 Strata 的 decode-once 内核和逐条目 decode 内核彼此之间，而不是与 float 参考
  对比。

## 已知限制
<a id="known-limits"></a>

- **本移植版没有**（rc2 有；见本页末尾的清单）：针对 Gyro 模型包的在第二块 GPU 上的提示词助手。两块卡的 setup 仍
  会做层切分和 decode 对等；提示词对等会提示它对 Hadamard 折叠的模型包不使用，提示词留在第一块卡上。Gyro-L、
  Intel Arc（SYCL）和 Apple Metal 不受支持。
- **源码构建：**SYCL 树用这些源码可以构建（P1006，0 个错误），但没有 trellis GPU 内核；Metal 引擎能构建，但在
  加载时拒绝该文件（`native_expert_supported`）。

- **尚未测量：**与 llama.cpp 的 token 一致性。
- **仅基准模式：**`strata --tokens-file ... --max-new N` 会在答案结束后继续 decode N 个 token；只在答案结束之前
  比较速度。
- **AMD：**trellis 内核在 gfx1151（ROCm 7.2.1）上通过一致性验证，那里的一次短端到端运行生成正确
  （约 28-31 tokens/s）；尚未做速度调优，也还没在 RDNA4 上跑过。
- **Gyro-M：**29.9 GiB 的专家层无法和 KV 缓存一起全部放进 32 GB 卡；`auto` 把其余的流式读取。
- **尚未做端到端验证**（Strata 会运行它们并记录警告）：gfx1151 之外的 AMD GPU，以及通过 P2P 在对等 GPU 上处理
  提示词。
