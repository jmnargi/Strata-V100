# 较老的 GPU（实验性）
<a id="older-gpus-experimental"></a>

Strata 受支持的显卡是 NVIDIA RTX 20 / 30 / 40 / 50 以及 [AMD_HIP.md](AMD_HIP.md) 中列出的 AMD 显卡。下面这些显卡通过
社区成员在自己机器上编写并实测的**可选开启、实验性**路径运行。维护者们持有其中一块显卡——Tesla P100——并在其上测试；
其余型号他们都没有：每条路径都在这里通过编译检查和单元测试，现成引擎及其输出保持原样不动。除文中说明由维护者们实测的
之外，数字都是报告者本人在各自一台机器上测得的。

## 支持矩阵
<a id="support-matrix"></a>

### NVIDIA
<a id="nvidia"></a>

| 显卡 | 计算能力 | 如何运行 | 其上有什么不同 | 报告数据 |
| --- | --- | --- | --- | --- |
| Tesla P100 | 6.0 | CUDA 12 引擎 | `__dp4a` 以模拟实现（逐位精确）；BF16 投影经 fp32；可选开启：`STRATA_Q8_SM60=1` 用 GP100 内核运行 Q8_0 稠密投影与输出头（[下文](#pascal-gp100-a-q8_0-decode-kernel-opt-in)） | 2x P100，IQ3_S，引擎 0.1.39：提示词 526-532 tok/s，128K 提示词 token 下 decode 28-32 tok/s，122K 处 needle 检查 3/3 通过（[报告](../../bench/results/2026-10-06-community-2x-p100/README.md)，#1157）；PH402（4 个 GP100 die），IQ3_XXS：用 `STRATA_Q8_SM60=1` 加一份 Q8_0 稠密分片时 decode 28.3 -> 38.9 tok/s |
| Tesla P40 / P4、GTX 10 系列 | 6.1 | CUDA 12 引擎 | BF16 投影经 fp32（那里的 cuBLAS 没有 BF16 GEMM，#395）；稠密权重用宽内核反量化、专家模式 8 及其自己的交错 matvec 表；可选开启 `STRATA_DENSE_F32=1`：把大的稠密 GEMM 作为 `cublasSgemm`（[下文](#tesla-p40-sm_61-what-was-measured-and-changed)） | P40，IQ3_S，引擎 0.1.30：提示词 217-374 tok/s，decode 30-33 tok/s（#395）；带 sm_61 改动的 0.1.43.1：565 个 token 时提示词 255 tok/s（原为 205），decode 37-38 tok/s（原为 34-35）；带 `STRATA_DENSE_F32=1` 时 7K 个 token 下 464-474 tok/s（原为 360-432） |
| Tesla V100、Titan V | 7.0 | CUDA 12 引擎 | BF16 投影跑在 FP16 张量核心上（#655、#540）；提示词注意力用 `mma.m8n8k4`（#600）；更精简的注意力内核（#540） | V100-PCIE-32GB，UD-IQ4_XS：提示词 1,123-1,251 tok/s（#600）；V100 32GB，IQ2_XS：来自 #540 的提示词 +22% |
| RTX 20（Turing） | 7.5 | **受支持**，现成引擎 | 可选开启：`STRATA_BF16_TC=1` 把 BF16 投影跑在 FP16 张量核心上 | RTX 2080 Ti，Q2_0：提示词 +15-18%（#655） |

一块显卡需要足够的显存才有实用价值：和所有显卡一样推荐 12 GB 或以上（2 GB 的 GT 1030 计算能力也是 6.1，
但它装不下模型的任何部分）。

### AMD
<a id="amd"></a>

| 显卡 | 架构 | 如何运行 | 报告数据 |
| --- | --- | --- | --- |
| RX 6800 / 6900 系列 | gfx1030 | setup（`--backend hip`），未验证；#540 的注意力内核在那里是默认值，每步 8 个 cell 并使用 DPP lane 交换（逐位精确；`STRATA_ATTN_PRE75=0` 运行标准内核） | [AMD_HIP.md](AMD_HIP.md#rdna2-gfx1030)；RX 6900 XT，IQ3_S：单独提示词 +4-6%，加上 #835 为 +7-12%（[bench/results/2026-10-04-rdna2-pre75-attention](../../bench/results/2026-10-04-rdna2-pre75-attention/README.md)） |
| RX 6700 XT | gfx1031 | setup（`--backend hip`），未验证（#524） | 其报告者日常使用，单块卡 |
| RX 5500 XT（RDNA1） | gfx1012 | 手工构建：`-DCMAKE_HIP_ARCHITECTURES=gfx1012`（HIP 5.7 或 7） | 8 GB 显卡，IQ3_S，8K 提示词：decode 15.3 tok/s（#442） |
| RX 5700 XT（RDNA1） | gfx1010 | 手工构建：`-DCMAKE_HIP_ARCHITECTURES=gfx1010`，用 ROCm 7.14 的 `gfx101X-dgpu` wheel；在一台同时有 AMD 核显的 PC 上需要 `ROCR_VISIBLE_DEVICES=0` | 8 GB 显卡，Coder IQ1_M，32K 上下文：4K / 16K / 30K 提示词 token 下提示词 115 / 144 / 149 tok/s、decode 18.0 / 20.3 / 23.3 tok/s，8K 和 30K 下 needle 检查 6/6，在该卡上 ctest 61/61，显存中 607 / 12,288 个专家层 |
| RX 5700 与 6 GB 的 RX 5600（RDNA1） | gfx1010 | 同上 | 与 RX 5700 XT 相同的 Navi 10 芯片，因此用同一构建；6 GB 显卡只剩约 50 个专家槽位，预期应看 RX 5500 XT 的区间而不是这个 |
| Radeon PRO V520 / Pro 5600M（RDNA1） | gfx1011 | 手工构建：`-DCMAKE_HIP_ARCHITECTURES=gfx1011` | 未在硬件上测试：它与 gfx1010 是同一 RDNA1 指令集（Navi 12），且引擎用同样的 wheel 构建它时 0 错误 |
| Instinct MI50 / MI60、Radeon VII | gfx906（wave64） | 手工构建：`-DSTRATA_HIP_GFX906=ON` | 2x MI50，Coder IQ1_M，128K 上下文：4K / 32K / 128K 提示词 token 下 decode 50.1 / 47.8 / 45.7 tok/s，提示词约 520 tok/s（#677） |

## NVIDIA：CUDA 12 引擎
<a id="nvidia-the-cuda-12-engine"></a>

CUDA 13 放弃了对 Pascal 和 Volta 的支持：它无法为它们编译。因此 setup 保留**第二个引擎**，用 CUDA 12.9 加
`-DSTRATA_EXPERIMENTAL_SM60=ON` 构建，放在它自己的文件夹中（`engine-cuda12\`，与 `engine\` 并列）。每个模型跑一个引擎，
所以这个选择按模型做出，依据是该模型所跑的最老的那块卡：

- **模型使用的每块卡都是 RTX 20 或更新：** 一如往常使用现成的 CUDA 13 引擎。
- **某块卡是 Pascal 或 Volta：** CUDA 12 引擎。setup 会说明（`CUDA 12: sm_70 is older than CUDA 13 supports ...`）。
  在 Windows 上它会下载 `strata-windows-x64-cuda12.zip` 以及 NVIDIA 的 CUDA 12 库（和 CUDA 13 的一样来自 pip）；
  在 Linux 上，或加 `--build` 时，它会用 CUDA 12.x 工具包编译引擎。

### 选择加入
<a id="opting-in"></a>

只有当你选择使用较老的显卡时才会用它；配有更新显卡的 PC 仍会推荐较新的那块。

| 你的情况 | setup 的行为 |
| --- | --- |
| 只有 Pascal / Volta 的 NVIDIA 显卡（且没有它可用的 AMD 显卡） | 使用它们，配 CUDA 12 引擎 |
| 指名显卡：`START-HERE.bat --setup --gpu 1`，或与较新显卡一起用 `--gpus 0,1` | 使用它；该模型得到 CUDA 12 引擎 |
| `--cuda 12`（或 `STRATA_CUDA=12`） | 该模型使用 CUDA 12 引擎，在任何显卡上 |
| `--cuda 13` | 即使是较老显卡也用 CUDA 13 引擎（会有警告：它没有该显卡的代码） |
| `STRATA_EXPERIMENTAL_SM60=1` | 较老的显卡被列为可用（#295 的那个设置仍然有效） |

这个选择保存在模型的配置里（`"cuda": 12`）：它的每次启动和 `UPDATE.bat` 都会保留，其他模型保留各自的引擎。
重新配置该模型会重新选择（依据其显卡；加上 `--cuda 12` 可保持强制选择）。在某次启动时用 `--gpus` 给模型加一块
Pascal / Volta 显卡，会把该模型移到 CUDA 12 引擎上。

`--cuda 12` 也是在低于 580 的 NVIDIA 驱动上运行 Strata 的办法：CUDA 12 在 Windows 上需要 528 或更新
（527.41，NVIDIA 的小版本兼容性），Linux 上需要 525；setup 的"驱动过旧"停止提示会说明这一点。这样老的驱动
未在这里测试过。

### 混合显卡
<a id="mixed-cards"></a>

一个把 V100 与 RTX 30 / 40 显卡共享的模型，两者都跑在 CUDA 12 引擎上（它有 sm_60 到 sm_89 的代码，外加 PTX）。
一块 RTX 50 显卡（sm_120）出现在 CUDA 12 引擎里是警告而非停止：CUDA 12.8 及更新能为它编译，但用 12.8 构建的引擎
在长提示词上崩溃过（#220、#224）。请把 RTX 50 显卡留给它自己的模型（`--gpu N`），在那里它跑 CUDA 13 引擎。

### 自己构建
<a id="building-it-yourself"></a>

```sh
# Linux (Windows: the same with the toolkit's nvcc.exe)
cmake -S . -B build-cuda12 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON \
      -DCMAKE_CUDA_ARCHITECTURES="61;70" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build-cuda12 --target strata -j
```

setup 在编译时做的是同样的事：它会寻找最新的 CUDA 12.x 工具包（`STRATA_NVCC=<path to nvcc>` 可指定一个，#601；
在 glibc 2.43 上用 12.8，见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)）。

在 GCC 12.3 配 nvcc（openEuler 24.03、CUDA 12.8、2x V100）上，构建需要在 host 标志里加上 `-D_BITS_OPT_RANDOM_H`
（#1074；一份报告，此处未能复现）。同一报告遇到的 `vmm.hpp` 中的 `size_t` 错误已在 0.1.40.2 中修复。

### 该标志改变了什么、没改变什么
<a id="what-the-flag-changes-and-what-it-does-not"></a>

`-DSTRATA_EXPERIMENTAL_SM60=ON` 把运行时下限降到计算能力 6.0，并且只在那个构建里编译较老显卡的代码路径：
Volta 的提示词注意力（#600）、#540 的注意力内核（在 sm_75 之下使用），以及 Pascal 的 BF16 路径。现成的 CUDA 13 引擎
完全没有这些，所以它的内核和输出都不变。BF16 投影的 FP16 路径在每个构建里都有，但默认只在 sm_75 之下运行；
RTX 20 用户可以用 `STRATA_BF16_TC=1` 试试（在 V100 上用 `=0` 关闭）。它与 cuBLAS 的 BF16 内核并非逐位相同
（FP16 张量核心求和的舍入不同）：#540 在 V100 上测得 24 个代码提示词的下一 token 分布平均 KL 为 8.4e-3
（23 个的 top-1 相同），量级与其他求和顺序改动相当；#655 在 RTX 2080 Ti 上测得每个乘积最坏相对差 3.5e-5。

### Tesla P100（sm_60）：默认开启的七个 decode 开关（引擎 0.1.45.3）
<a id="tesla-p100-sm_60-seven-decode-switches-on-by-default-engine-01453"></a>

在 Tesla P100 上，引擎现在会开启一组七个只改变工作调度方式的开关，因此答案逐位保持不变：`STRATA_ADAPT_NOWAIT=1`、
`STRATA_ADAPT_LAG=2`、`STRATA_SIDE_MIXER=2`、`STRATA_FUSE_WAIT=2`、`STRATA_B6_MMVQ_ROWS=1`、`STRATA_FUSE_DECODE=1`
和 `STRATA_GR_UP75=1`。`STRATA_SM60_PERF=0` 关闭整组；你自己设置的任何开关（无论取什么值）都会被保留。
当每块可见显卡的计算能力都是 6.0 时使用该组。由维护者在 Tesla P100 16 GB、IQ3_XXS、默认标志下实测，
8 组全新引擎的交错配对：decode 36.56 -> 41.76 tokens/s（+14.2%，8/8 组都更快；与自身对跑得到 0.975），
专家缓存槽位数相同；精确性门（10 个提示词与 4K/20K 提示词转储）在开启该组时完全一致。完整列表、RTX 30 的那组
以及逐个开关的结果：[DETAILS.md](DETAILS.md#rtx-30-series-sm_86-and-tesla-p100-sm_60-bitwise-decode-switches-on-by-default)。

### Pascal GP100：一个 Q8_0 的 decode 内核（可选开启）
<a id="pascal-gp100-a-q8_0-decode-kernel-opt-in"></a>

GP100（P100、PH402）是唯一没有 `__dp4a` 的 Pascal 芯片，而 decode 内核都是为有它的显卡写的：在一个 GP100 die 上，
verify window 的稠密投影读取权重只能达到其约 730 GB/s 中的 40-80 GB/s，逐个 int 地解码 K- 与 i-quant 块
（Q6_K、IQ4_XS、Q4_K、IQ3_S……）。两处改动，都只在 PH402 上实测过：

- **`STRATA_Q8_SM60=1`** 把每个 Q8_0 稠密矩阵和输出头打包进 `STRATA_Q8_PACKED` 的平面（任意形状），并在
  `src/kernels/cuda/q8_sm60.cuh` 上运行它们：每行一个 warp（窄行则半个或四分之一个 warp）、16 字节的权重载入、
  该窗口的 q8_1 激活值在每个工作群组只拷贝一次进共享内存成为一个 int8 平面、点积用 `STRATA_DP4A`（vmad）、
  一次 shuffle 求和——每行不需要屏障。它拒绝的形状（宽度 6144、7-8 列的行）仍用常规内核。并非逐位精确的内核
  （每 32 个乘积以另一顺序相加）。打包副本在提示词路径读取的 GGUF 布局之外还要占用其大小的显存
  （Flash-Next：跨各级 2.9 GiB 稠密投影，输出头 0.6 GiB）。
- GSQ-RCO 模型包没有 Q8_0 稠密矩阵，所以该内核需要一份其稠密投影和输出头为 Q8_0 的模型副本：每个持有这类矩阵的
  分片都会被转换（Swift 1.5 在它的两个分片里都保留它们），其余分片以硬链接放在旁边（+3.5 GiB；专家层保持其偏移量，
  因此原生模型包可以复用）。把 `--native` 和 `--ple-gguf` 指向新文件夹。重新量化的代价最多是每个张量 |w|max 的 0.4%
  （Q8_0 比源格式更细）。
- **可选开启 `STRATA_IQ_STAGE_TABLES=1`**：IQ3_XXS 和 IQ3_S 的码本与符号掩码表由分组专家内核暂存进共享内存
  （IQ3_XXS 的从 shinbunbun/llama-cpp-p100-patches 的 29 和 30 移植而来，IQ3_S 的对其符号字节做同样处理；输出逐位相同）。
  IQ3_S 是那些内核仍从全局内存读取码本的唯一 gate/up 格式：贡献者在 GP100 die 上测得一个 verify window 的 IQ3_S 专家层
  （20 个专家层、24 个条目）快 24%（配 Q2_0 down 投影时 517 -> 393 us，配 IQ4_NL 时 492 -> 369 us），
  而 Swift 1.5 IQ3_XXS（其 48 层中有 13 层是 IQ3_S）在 1M 上下文下快 3.2%（39.3 -> 40.5 tok/s）；
  只有 IQ3_XXS 的表时在噪声范围内（+0-4%）。维护者在 Tesla P100（PCIe，IQ3_XXS 模型包与 IQ3_S 模型包，
  各 5-6 组）上的交错 A/B 显示端到端没有变化（decode 0.97-1.03x，在该机器的噪声之内），所以它默认关闭
  （贡献者的版本在计算能力 6.0 上默认开启）。在 RTX 3090 Ti 上强制开启时，同样的 IQ3_S 层耗时多 3-13%
  （那些内核在那里本来就以 490-665 GB/s 读取）。
- 两处改动都没有在计算能力 6.1（P40、P4、GTX 10 系列）上实测，它们有 `__dp4a`。两者都能为 sm_61 构建，
  而在有 `__dp4a` 的显卡上 Q8_0 内核的点积就是那条硬件指令；来自某块 6.1 显卡配 `STRATA_Q8_SM60=1` 或
  `STRATA_IQ_STAGE_TABLES=1` 的数字可以看出它们在那里是否有帮助。

在 PH402 SKU 200（两块板，四个各 48 SM、32 GB HBM2 的 GP100 die）上实测，应用时钟锁定在 1050 MHz，Windows 11，
驱动 581.80（TCC），CUDA 12.9，引擎 0.1.40.2；Flash-Next IQ3_XXS，32K，`--layer-split 12,24,36 --prefill 2048`；
贪心，256 个 token，2-3 次交错启动的中位数，decode tok/s：

| | 约 10.7K-token 文档 | 同样的内容，已缓存 | 短推理 | 代码 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSQ-RCO 分片，精确内核 | 28.8 | 29.4 | 28.4 | 26.8 | 28.3 |
| Q8_0 稠密分片，`STRATA_Q8_PACKED=1` | 27.5 | 31.0 | 34.4 | 31.8 | 31.1 |
| Q8_0 稠密分片，`STRATA_Q8_SM60=1` | 37.5 | 38.4 | 41.2 | 38.5 | 38.9 |

每个 verify window 的 GPU 时间（`STRATA_VERIFY_PROFILE`）：稠密投影 19.2 -> 11.2 ms，输出头 4.25 -> 2.1 ms；
随后草稿策略选择更长的窗口（每个窗口 2.10 -> 2.31 个 token）。提示词读取不变（10.7K 下 437-438 tok/s）。

在这块卡上有两件事的影响同样大：它的默认应用时钟是 759 MHz，而层切分会让每个 die 在一个窗口的大部分时间空闲，
所以除非锁定时钟，各 die 会在 1050 MHz 之下解码（以管理员身份运行 `nvidia-smi -i <dies> -ac 715,1050`；
10.7K 上下文下 25.0 -> 29.1 tok/s）。以及同一台 PC 上第二个把线程固定到相同逻辑 CPU 的引擎，会在其运行期间
使本引擎的 decode 减半（host 线程被时间片轮转）；把两者放到不同的逻辑 CPU（SMT 兄弟线程）可以避免。

用于 A/B 的开关：`STRATA_DEQUANT_WIDE=0`、`STRATA_SM61_TABLE=0`（sm_61：0.1.43.1 所跑的内核）、
`STRATA_DENSE_F32=1`（Pascal，可选开启）、`STRATA_Q8_SM60=0|1`（Pascal GP100，可选开启）、`STRATA_BF16_TC=0|1`、
`STRATA_PROMPT_ATTN_OLD=1`（用于提示词的 decode 内核）、`STRATA_ATTN_PRE75=0`（关闭 #540 的内核；在 HIP 上的 gfx103x
上该内核是默认值，见上文 AMD 表，`=1` 会为另一块 wave32 的 AMD 显卡开启它）。[NVIDIA_V100.md](NVIDIA_V100.md) 有 V100
的构建、其实测数据和一致性测试。

### Tesla P40（sm_61）：实测与改动
<a id="tesla-p40-sm_61-what-was-measured-and-changed"></a>

一块 Tesla P40 24 GB，位于 20 vCPU 的虚拟机（Xeon E5-2695 v4、98 GB 内存），CUDA 12.4（cuBLAS 12.4.5.8），
带 MTP 的 IQ3_S（`--spec 4`），int8 KV，`--max-context 262144`，引擎 0.1.43.1；数字和每次运行都在
[bench/results/2026-10-12-p40-sm61-kernels](../../bench/results/2026-10-12-p40-sm61-kernels/README.md)。
`STRATA_PREFILL_TIMING=1`、`STRATA_DECODE_TIMING=1` 和 `STRATA_VERIFY_PROFILE=1` 显示这块卡在处处都是瓶颈：
提示词的 GPU 时间线等于其墙钟时间（一个 7K-token 提示词中，等待专家拷贝占 0.2%，宿主分组占 0.2%），
一个 decode 窗口是 49 ms 的 GPU 阶段加 1-3 ms 的 CPU 专家层。在一个 7K-token 提示词中，稠密投影的 GEMM 占 36%，
专家层占 28%（ggml 的 MMQ，它有自己的 Pascal dp4a 配置），hyper-connection 读取占 13%，注意力占 12%。

在计算能力 6.1 上有三件事默认运行方式不同，第四件是可选开启。每一项都与其替换的对象给出相同的位：

| 改动 | 它替换了什么 | 在 P40 上的实测 | 开关 |
| --- | --- | --- | --- |
| 为显存中的专家层（decode 与 verify window）做宽反量化（每 8 个值一个线程、16 位载入、16 字节存储），适用于 Q4_K / Q5_K / Q6_K / IQ4_XS | 每 32 个值一个线程并用字节载入：不论提示词分块多长，它都要花掉每个分块的 0.5 s | Q6_K 10240 x 2560：7.2 -> 0.33 ms（10 -> 227 GB/s）；Q4_K 5.2x，Q5_K 3.8x，IQ4_XS 3.6x；`dequant_wide_parity` 360 个用例逐位 | 默认开启；`STRATA_DEQUANT_WIDE=0` |
| 显存中专家层用专家模式 8（decode 与 verify window） | CUDA 布局 | 一次 18 个专家层、2 个 token 的调用：IQ3_S 390 -> 255 us，IQ3_XXS 345 -> 252，IQ2_S 290 -> 238，IQ4_XS 328 -> 261；`native_expert_bench` 逐位 | 默认开启；`STRATA_SM61_TABLE=0`（或 `STRATA_EXP_MODE=0`） | |
| 使用在 P40 上实测的表的交错小批量 matvec（`kIlRows_61`） | `native_mmvq` | IQ4_XS 2560 x 10240 在 2 列时 112 -> 65 us，Q6_K 12288 x 2560 在 4 列时 189 -> 127 us，Q5_K 输出头在 3 列时 2,545 -> 1,766 us；`mmvq_il_parity` 逐位 | 默认开启；`STRATA_SM61_TABLE=0`（或 `STRATA_MMVQ_IL=0`） |
| 把大的稠密 GEMM（qkv、query、gate、输出投影）作为 FP16 操作数的 FP32 副本上的 `cublasSgemm` | cuBLAS 的 FP16 操作数内核，它们在多数分块大小下只跑到 4.3 TFLOPS 而不是 7.6（一个 7K-token 提示词按其精确 token 数需要 16.3 或 19.5 s 读取） | 从 512 个 token 起的每种规模都有 8.4-9.8 TFLOPS；7,041 个 token 时 qkv 85 -> 45 ms；`gemm_f32_parity` 5,201 种分块大小逐位 | **可选开启**，`STRATA_DENSE_F32=1` |

这条 GEMM 路线之所以是可选开启，是因为它的相等性是 cuBLAS 构建的属性，而非代码的属性：它只用 cuBLAS 12.4.5.8
实测过。在你设置它之前，先用你工具包的库运行 `gemm_f32_parity`（一个带 `-DSTRATA_BUILD_TESTS=ON` 的构建；
退出码 0 = 每种规模都逐位）。

经由服务器，每次运行取 3 个请求的中位数，每个构建两次运行交替，每次运行都从显卡 65 C 开始（见下文）。默认情况下：
565 个 token 的提示词 203 / 205 -> 256 / 255 tok/s，其后的 decode 35.0 / 34.2 -> 38.2 / 37.2 tok/s，
7,047 个 token 的提示词在相同 token 数下快 1-3%。同时带 `STRATA_DENSE_F32=1` 时：7,047 个 token 的提示词
430 / 364 -> 474 / 473 tok/s（旧构建读取它约为 361 或 431 tok/s，取决于分块是否超过 7,040 个 token；
用了这条路线后每次读取都是 464-474），其后的 decode 32.1 / 33.4 -> 35.9 / 34.0。用
`STRATA_DEQUANT_WIDE=0 STRATA_SM61_TABLE=0` 且关闭该路线时，新构建测得与旧构建一致（210 / 365 tok/s，33.3 tok/s）。

实测过但**没有**改动的：延迟隐藏的 hyper-connection 读取（`STRATA_GR_FAST=1`）在 P40 上于 5 个 token 以下更慢
（1 个 token 时 82 -> 88 us）；`STRATA_PREFILL_CPU_SHARE=auto` 和 `STRATA_ADAPT_OVERLAP=1` 没有任何收益。

在这类显卡上做测量时要知道两件事：

- **温度。** P40 没有自己的风扇。这一处在几分钟工作后达到 87-89 C；89 C 时驱动报告 `SW Thermal Slowdown: Active`，
  时钟从 1,531 降到约 1,300 MHz（`nvidia-smi -q -d PERFORMANCE`、`nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv -l 2`）。
  让一次 A/B 的每个分支都从相同温度开始，否则后来的那个分支必输。
- **在没有 AVX-512 的 CPU 上，贪心答案逐次运行并不相同**，即使加上 `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`：
  在 CPU 上计算的专家层，对孤立 token 的舍入与在多个 token 的窗口中不同（`native_gu_mt_min`），
  而一个窗口装多少个草稿取决于时序。加上 `STRATA_IQ_MT_MIN=1`（以及 `STRATA_PREFILL_CPU_SHARE=0`、
  `STRATA_ROUTE_TAIL_SKIP=0`）才能逐字节比较两个构建的答案。

### Quadro RTX 8000：对既有开关的实测
<a id="quadro-rtx-8000-measurements-of-existing-switches"></a>

Quadro RTX 8000 是 Turing（sm_75）：它有 FP16 张量核心，但没有原生 BF16 或 TF32 张量核心。两个**既有的、可选开启**的
CUDA 开关值得在你自己的工作负载上比较：

- `STRATA_BF16_TC=1` 把 BF16 投影操作数转换为 FP16，并使用带 FP32 累加的 FP16 张量核心。sm_75 上默认关闭；
  `=0` 恢复 cuBLAS 的 BF16 路径。该开关适用于 7.x 显卡，不适用于 sm_80 及更新上的原生 BF16 路径。
  转换会把 FP16 范围之外的有限值钳制到 ±65504；很小的值以及顺序不同的求和可能舍入不同。
- `STRATA_SELECT_SIMT=1` 在 pre-sm_80 的 CUDA 路径上使用分块的 FP32 QSA 块打分器，而不是默认的 warp 打分器。
  `=0` 保留 warp 打分器。它不会在 Turing 上启用 TF32，且其求和顺序不同。

这些既有路径在 [gemm.cu](../../src/prefill/gemm.cu) 和 [qsa_select.cu](../../src/kernels/cuda/qsa_select.cu) 中。

在启动引擎之前设置环境变量，或把开关写进模型配置的 `env` 对象，然后重启。例如，这段配置片段同时启用两者
（保留配置的其他设置）：

```json
"env": {
  "STRATA_BF16_TC": "1",
  "STRATA_SELECT_SIMT": "1"
}
```

本节记录的是一次调查，**不是一个新内核，也不是对默认值的改动**。2026-10-07 的实测使用一块 48 GiB 的
Quadro RTX 8000（物理 GPU 3）、260 W 功耗上限、一颗 Xeon W-2295、CUDA 12.4 和 GCC 13。源码构建的引擎基于
v0.1.40.3 并整合了其他本地改动；这些测试分支中禁用了无关的可选开启改动。这些不是在一个干净仅有文档改动的检出上
的测量，不是启动器 200 W 设置下的结果，也不是对每块 sm_75 显卡的速度承诺。

模型是带 MTP 的 IQ3_S（`--spec 4`）、`--max-context 262144`、带 `--kv-resident 32768` 的 INT8 KV、
`--pcie-frac 0.20`、固定请求 `--expert-cache 19000`，以及 `--prefill auto`（8,192 个 token 上限）。
引擎报告 21,189 个缓存槽位（含提示词借用）。提示词缓存与自适应换入换出都关闭（`--prompt-cache 0 --adapt-swaps 0`）。
每个规模使用三个全新、带种子的合成提示词，每个请求 128 个输出 token；表中给出中位数的引擎报告速率，
不是端到端的服务器延迟。

| 既有开关设置 | 537 个 token：提示词 / decode tok/s | 4,057 个 token：提示词 / decode tok/s | 32,057 个 token：提示词 / decode tok/s |
| --- | --- | --- | --- |
| BF16_TC=0、SELECT_SIMT=0 | 493 / 86.8 | 1,055 / 79.9 | 1,135 / 70.6 |
| BF16_TC=1、SELECT_SIMT=0 | 661 / 83.7 | 1,262 / 75.4 | 1,416 / 68.4 |
| BF16_TC=1、SELECT_SIMT=1 | 679 / 87.1 | 1,290 / 78.2 | 1,501 / 71.6 |

BF16_TC 把提示词吞吐提升了约 20–34%，但在这些比较中没有改善 decode。在 BF16_TC 之上再加 SELECT_SIMT，
使 32K 提示词速率相比只用 BF16_TC 提升约 6%；这不是普遍的 decode 收益。BF16_TC 在 9 个配对请求中的 3 个里改变了
回复，所以只有在能接受不同舍入时才使用它。五个基础算术/JSON 检查在每个分支中都通过，这是一次冒烟检查，
**不是模型质量研究**。另一次在 131,072 上下文 token 上做的合成 QSA 打分器检查，对全部 256 次查询都匹配了所选 ID，
并通过了它的 FP64 精度门。该检查并不能证明在任意提示词上选择完全一致，也不证明模型质量。

另外两个既有的调优选项在这台机器上不足以构成推荐：

- `--prefill auto:16384`（BF16_TC 开、SELECT_SIMT 关）把提示词的缓存借用从 4.38 GiB 增加到 7.38 GiB，
  却只带来约 0.6% 的 32K 提示词吞吐提升（1,425 对 1,416 tok/s）。最后一次图捕获时只剩 24 MiB 显存。
  为这么小的收益不值得花掉那点内存余量，请保留 8,192 的上限。
- `STRATA_PREFILL_CPU_SHARE=auto` 在两个测试规模（537 和 4,057 个 token）上没有收益：提示词速率为 647 和
  1,253 tok/s，而只用 BF16_TC 时是 661 和 1,262；decode 也更低。更大的提示词未针对该设置测试。

## AMD：构建 gfx906 与 gfx1012
<a id="amd-building-gfx906-and-gfx1012"></a>

setup 不构建这些；请手工构建，并像在其他显卡上那样用配置运行 `serve/server.py`。

- **gfx906**（MI50 / MI60 / Radeon VII，wave64）：一个单独的可选开启构建，`-DSTRATA_HIP_GFX906=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx906`（不是 `STRATA_ENABLE_HIP`）。当前的 ROCm 不再提供 gfx906 库；报告者使用了一个
  社区的 ROCm 7.14 镜像。做法、内核与实测数据：[AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source)。
- **gfx1012**（RX 5500 XT）：wave32 后端，`-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1012`。
  HIP 5.7（Ubuntu 的软件包）可用：较旧的 hipBLAS（0.x）通过 rocBLAS 使用，旧的 HIP 名称与缺失的 `__syncwarp`
  都按版本门控，而 RDNA1 缺失的带符号 dot4 使用 llama.cpp 的 SDWA 序列
  （`-DSTRATA_GFX1012_PORTABLE_DOT=ON`：可移植的那个）。

## 欢迎报告
<a id="reports-welcome"></a>

在这些路径被更多人运行之前，它们会保持实验性状态。在 issue 中给出一份包含显卡、驱动 / ROCm 版本、模型和引擎日志
（`strata-<model>.log`）的报告会有帮助；[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) 有实测数据的格式。
