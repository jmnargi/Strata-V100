# Unsloth 的 K-quants 与 Q8_0：格式、内核与质量
<a id="unsloths-k-quants-and-q8_0-formats-kernels-and-quality"></a>

本页介绍 Unsloth 为该模型提供的文件（[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)，
修订版 `38bb39e`；仓库的 head `766911a` 有完全相同的文件，逐字节一致）：每个文件在
每一层里存了什么、Strata 的哪些内核在各类硬件上运行这些格式，以及每个文件离
Q8_0 参照有多近。安装与逐文件细节见：[UD-Q4_K_XL](UNSLOTH_Q4.md)、[UD-Q6_K_XL](UNSLOTH_Q6.md)、
[MODELS.md](MODELS.md)。

## 每个文件存了什么
<a id="what-each-file-stores"></a>

文件之间的差别在于路由专家层（gate、up、down；每层 512 个，每个 token 用 10 个）：

| 文件 | 下载大小 | 路由 gate / up | 路由 down | 路由专家层 | 每个 token（10 个专家层 x 48 层） |
|---|---|---:|---|---|---:|---:|
| UD-IQ4_XS | 93.7 GB | IQ3_S（第 2 层为 IQ4_XS） | IQ4_NL（第 2、4、30、46、47 层为 Q8_0） | 55.4 GiB | 1.16 GB |
| UD-Q4_K_XL | 111.3 GB | Q4_K（第 2 层为 Q5_K） | Q5_1（第 2、4、30、46、47 层为 Q8_0） | 71.7 GiB | 1.50 GB |
| UD-Q5_K_XL | 158.3 GB | Q5_K（第 2 层为 Q6_K） | Q8_0 | 91.6 GiB | 1.92 GB |
| UD-Q6_K_XL | 169.2 GB | Q6_K（第 2 层为 Q8_0） | Q8_0 | 101.7 GiB | 2.13 GB |
| Q8_0 | 188.2 GB | Q8_0 | Q8_0 | 119.5 GiB | 2.51 GB |

其余部分在五个文件里都一样：attention、SSM、hyper-connection 和共享专家层投影，PLE 的
key/value 以及 token embedding 都是 Q8_0（小向量、norm 和 router 为 F32；QSA indexer 为 BF16），
输出头是 Q8_0（UD-IQ4_XS 里为 Q6_K）。PLE 表（`per_layer_token_embd`）在 UD-IQ4_XS
和 UD-Q4_K_XL 里是 IQ4_NL（26.8 GiB），在 UD-Q5_K_XL、UD-Q6_K_XL 和 Q8_0 里是 Q8_0（50.7 GiB）。GPU 每个 verify
窗口读一遍的稠密部分在每个文件里都约为 4.8 GB。

UD-Q5_K_XL、UD-Q6_K_XL 和 Q8_0 的分片 2（输出头）和分片 3（PLE 表）逐字节相同（SHA-256 一致），
所以已经有其中一个的 PC 可以直接链接这两个分片，不必重新下载。

## 每种格式与后端的内核
<a id="kernels-per-format-and-backend"></a>

| 格式（角色） | CUDA decode（分组专家层） | CUDA prompt | CPU 池 | AMD（HIP）decode | AMD prompt | Intel（SYCL）decode | Intel prompt |
|---|---|---|---|---|---|---|---|
| Q4_K（gate/up） | 是 | FP16 路径；`STRATA_MMQ_KQ=1` 时用 MMQ | ggml AVX2 点积；多 token AVX2（`STRATA_KQ256=1`） | 是（CUDA 布局；`STRATA_EXPERT_V2K=1` 时按位一致） | FP16 路径；`STRATA_MMQ_KQ=1` 时用 MMQ；gfx11 融合（`STRATA_PF_FUSED_KQ=1`） | 内核已构建，未用这些文件运行 | FP16 路径 |
| Q5_K（gate/up） | 是 | 同 Q4_K | ggml AVX2 点积 | 同 Q4_K | 同 Q4_K | 内核已构建，未用这些文件运行 | FP16 路径 |
| Q6_K（gate/up） | 是（自本次改动起；此前为 `-DSTRATA_Q6K_EXPERTS=ON`） | FP16 路径；`STRATA_MMQ_KQ=1` 时用 MMQ | ggml AVX2 点积 | 是（CUDA 布局） | FP16 路径；`STRATA_MMQ_KQ=1` 时用 MMQ | 内核已构建，未用这些文件运行 | FP16 路径 |
| Q5_1（down） | 是 | 同 Q4_K | ggml AVX2 点积；多 token AVX2（`STRATA_KQ256=1`） | 是（CUDA 布局；V2K） | 同 Q4_K | 内核已构建，未用这些文件运行 | FP16 路径 |
| Q8_0（gate/up，down） | 是 | MMQ | ggml AVX2 点积；多 token AVX2 down（`STRATA_KQ256=1`） | 是（CUDA 布局；V2K down） | MMQ | 内核已构建，未用这些文件运行 | FP16 路径 |

- “FP16 路径”：把 prompt 的专家层反量化为 FP16，再用 BLAS 库相乘。它对上面每种格式
  都可用，但每个专家层要读约 10 MB 的 FP16。
- **格式不用就不产生代价。** 引擎启动时会加载自己二进制里的每个内核（这样之后在 prompt 进行中
  就不再需要显存），这也是过去模型没有的格式的内核会占用每个模型专家槽位的原因。Q6_K 的专家层内核
  现在已进入每个构建：每个分组内核多一个实例、sm_60 代码 +19 KB，实测专家缓存槽位数不变
  （IQ3_XXS：P100 上 6,028 个槽位，R9700 上 15,878 和 1,980 个）。K-quant 的 MMQ prompt 内核
  （Q4_K、Q5_K、Q5_1、Q6_K：约 3.8 MB sm_60 代码，放在引擎里时在 P100 上占 3 个 IQ3_XXS
  槽位）在 CUDA 上是引擎旁边的独立库（`libstrata_kq.so` / `strata_kq.dll`），只对带这些
  专家层且设了 `STRATA_MMQ_KQ=1` 的模型打开；没有这个文件时，这些专家层走 FP16 prompt
  路径。在 AMD 上代码对象在首次使用时加载，所以它们留在引擎里。
- `STRATA_MMQ_KQ=1` 是选择性启用的，因为 MMQ 对 prompt 乘积的舍入方式与 FP16 路径不同（输出
  在舍入层面就有差异）；在成为默认值之前会先测量它的速度和 KL。

## 速度与上界
<a id="speed-and-the-bound"></a>

在 AMD 机器上测得：一块 Radeon AI PRO R9700（32 GB，功耗上限 225 W）、EPYC 7452（32 核，8 通道
DDR4-2400：32 线程下 `rdbw` 读到 106 GB/s），所有专家层都在 RAM（见下面 setup 的规则），`--expert-cache auto`，
40K 上下文，8 位 K/V，MTP 草稿；取两次请求中的第二次；这台机器上 prompt 速度在多次运行之间
最多相差 2x（已给出范围）。

| 文件 | 短故事（tok/s） | 代码（tok/s） | 6K 文档之后 | Prompt 32K（tok/s） | llama.cpp 8bb3cf0d2（同一张卡，`-ngl 99 --n-cpu-moe 38-48`）：decode / prompt 4K |
|---|---:|---:|---:|---:|---:|
| IQ3_S（对照） | 75 | 95 | 77 | 1,555 | - |
| UD-Q4_K_XL | 63-64 | 85-88 | 70-72 | 1,110-1,320 | 18-20 / 198-257 |
| UD-Q5_K_XL | 57-60 | 81-83 | 62 | 810-1,260 | 17-18 / 226-270 |
| UD-Q6_K_XL | 54 | 76 | 61 | 1,020 | - |
| Q8_0 | 53-57 | 66-74 | 54-55 | 1,230-1,300 | 16-18 / 186-225 |

**一个 decode 窗口的时间花在哪里**（`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`，UD-Q4_K_XL，短故事，每个窗口
2.0 个 token，每个窗口 34.3 ms）：一个窗口必须读取的数据的带宽下界约为 12.5 ms —— 显存里的路由
专家层（约 2.7 GB，按 644 GB/s 计：下界 4.2 ms，实测 6.0）、稠密的 Q8_0 投影（2.84 GB：下界 4.4 ms，
实测 7.2）、hyper-connection（0.67 GB：下界 1.0 ms，实测 4.0）、router 与共享专家层（0.5 GB：下界 0.8 ms，
实测 3.7）、输出头（下界 1.05，实测 1.26）、并行以 RAM 带宽运行的 CPU 池专家层
（1.2 ms），再加上 MTP 草稿（实测 5.8 ms）。对每个文件（包括 IQ3_S），decode 都跑在该下界的约 35-40%：
各格式自己的内核已接近其带宽（放在显存里的专家层约 70%，CPU 池为 RAM 带宽的 84-97%，
输出头 83%）；其余是小内核和草稿的每窗口延迟，对所有模型都一样。

**稠密 Q8_0 GEMV**（引擎的 shape，2-4 列）：在测得变快的块上每块处理一行，位宽与两行相同
（R9700：2560 -> 10240 / 6144 / 12288，达到 644 GB/s 的 76-86% 而不是 51-80%；RTX A4000：
+1-4%，且 6 列时 6144 -> 2560 从 70% -> 76%）。在 R9700 上，同样的文本下 UD-Q4_K_XL 的 decode +1.0-2.7%。

**NVIDIA 上使用 RAM 预算**（RTX A4000 16 GB、Xeon E5-2643 v3、62 GB：UD-Q4_K_XL 有 39 GiB 的专家层放在 RAM，
其余以 2.1 GB/s 从 NVMe SSD 读取）：14-20 token/s，4K prompt 为 134-144 token/s，受 SSD 限制；
同一台 PC 上的 llama.cpp 用同一个文件为 14.8-16.5 / 117-123。`STRATA_PROMPT_DIRECT=1`（Linux，选择性启用）让
prompt 路径的专家层走无缓冲读取，而 decode 保留文件缓存：输出相同（固定参数下 6/6 个 prompt 完全相同），
4K prompt 152 -> 315 token/s，20K prompt 217 -> 500，decode +2-15%（5 组交替对比，每次运行前丢弃
页缓存）。在另一台文件缓存空间更大的 PC 上（R9700 机器限制到 64 GB）结果有好有坏：6K 文档的 prompt +8%，
4K prompt -15%，文档之后的回答 -38%，因为缓冲读取把该文档的专家层留在了文件缓存里。
所以它仍是选择性启用。全部改成无缓冲读取（`STRATA_UNBUFFERED_LOAD=1`）同样让 prompt 翻倍，但 decode
损失 9-13%；更大或更小的 RAM 预算、更多取数线程、更深的预读或更宽的提前查找都没有
帮助。

**多张显卡**（R9700 机器，所有专家层都在 RAM，层切分，每种各跑 2 次）：UD-Q4_K_XL 在 1 / 2 / 3 / 4 张卡上
短故事为 63 / 64 / 55 / 52 token/s，代码为 88 / 96 / 123 / 96，读 32K prompt 为 1,110 /
1,275 / 1,987 / 2,381 token/s；UD-Q5_K_XL 为 57 / 60 / 62 / 46 和 83 / 94 / 101 / 89，prompt 为 1,264 / 1,177 / 1,767 /
2,067。llama.cpp 在 3 / 4 张卡上：UD-Q4_K_XL 为 24.5 / 28.2 token/s 和 497 / 800（4K prompt）。卡越多读
prompt 越快；对短回答来说，层切分每阶段的成本超过了它放进显存的专家层收益，因为一张卡把所有专家层
放在 RAM 上已经很快（`--pipeline-windows 0` 在 3-4 张卡时对短回答有帮助，x1.3，代价是代码慢
6-20%）。

**所有专家层都在 RAM 对比旧的 RAM 预算**（setup 的新规则，R9700 机器，UD-Q4_K_XL，5 组交替对比，
旧配置用 `--resident-budget-gib 71` 原地从 GGUF 读取同样的专家层）：decode x1.34-1.40（短故事
39.5 -> 53.3 tok/s，代码 60.1 -> 80.4，6K 文档之后 49.0 -> 68.8，每项 5/5），prompt 4K x2.07、6K x1.58、
32K x1.03。

**route tail skip**（`STRATA_ROUTE_TAIL_SKIP=7`，自 0.1.42 起的默认值：窗口内每个 token 若漏掉的专家层
在 rank 7 或更靠后，就跳过它）对这些文件来说代价高于 2-3 位模型包：UD-Q4_K_XL 相对
Q8_0 的 KL，在所有专家层都在 RAM 时（R9700，auto 缓存）从 0.056 升到 0.086，在 RAM
预算模式下（A4000，1,500 个 GPU 槽位）从 0.053 升到 0.115，换来 RAM 模式下 +4-10% 的 decode、
预算模式下 +26-67%（各 5 组对比）。在配置的 `"env"` 里设 `STRATA_ROUTE_TAIL_SKIP=0` 可以关掉它。

**NVIDIA 上的 hyper-connection：** 模型包里 `--compat-bf16` 生成的 Q8_0 HC 投影副本会把它们舍入成 BF16
（字节数也翻倍）。`STRATA_HC_Q8=1` 改为读取文件自带的 Q8_0（在 AMD gfx12 上这是默认值）：在
A4000 上 UD-Q4_K_XL 于是更接近 Q8_0（KL 0.0562 -> 0.0526，p99 0.925 -> 0.716），也更接近同一文件在
AMD 上的结果（KL 0.038 -> 0.033）。在做出决定前仍是选择性启用。

## 图像
<a id="images"></a>

Unsloth 的每个文件都用模型自带的编码器读图：setup 的 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（来自
ISTA-DASLab 的仓库）与 Unsloth 的 `mmproj-BF16.gguf` 有完全相同的 334 个张量，逐字节一致。用一张测试图片
（一行文字和三个彩色图形）端到端运行，通过 `/v1/chat/completions` 问两个问题：
在带 GPU 编码器的 RTX A4000 上（UD-Q4_K_XL、IQ3_XXS）以及在带 CPU 编码器的 R9700 上（UD-Q4_K_XL、UD-Q5_K_XL、
IQ3_XXS），每个回答都把文字和三个图形答对了，与 llama.cpp 的 `mtmd-cli` 在同一 GGUF 和
mmproj 上的表现相同。GPU 编码器会保留约 1.3 GB 的显存余量（16 GB 上的 UD-Q4_K_XL：2,331 个专家槽位而不是 2,741 个）；CPU
编码器则不占。对这些文件，setup 在 16 GB 或以上的 NVIDIA 卡上默认打开图像，在 AMD 上
（Linux，CPU 编码器，所有专家层都留在 RAM）也默认打开。
