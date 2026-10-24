# NVIDIA Tesla V100 / Titan V（Volta，sm_70）：社区构建
<a id="nvidia-tesla-v100-titan-v-volta-sm_70-the-community-build"></a>

Strata 的现成引擎与支持矩阵从计算能力 7.5（RTX 20）起步。V100（7.0）通过 issue #236 加入的实验性构建，运行**为它
编译的同一引擎**。它**由社区测试，不属于受支持的显卡**：没有为它发布的现成引擎，下文数字也来自一台机器。

## 构建
<a id="build"></a>

CUDA 13 放弃了对 Volta 的支持，因此构建需要一个 **CUDA 12.x** 工具包（用的是 12.8）以及该工具包接受的宿主 GCC。

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata -j
```

对于跑在 V100 上的模型，setup 自己就会做同样的事（实验性的 CUDA 12 引擎，位于 `engine-cuda12/`：Windows 上取现成的，
Linux 上用 CUDA 12.x 工具包编译；[OLDER_GPUS.md](OLDER_GPUS.md)）。对于一块 V100 加一块较新显卡的 PC，为两种架构
构建（`-DCMAKE_CUDA_ARCHITECTURES="70;86"`）是自然的选择（issue #509 中的一份社区报告以这种方式让一块 RTX 3080 与
一块 V100 并存；此处未重复该实验）。在测量用的 PC 上，一块 Quadro RTX 4000（sm_75）与 V100 们并排，因为仅为 sm_70
的构建没有适用于它的代码，所以用 `CUDA_VISIBLE_DEVICES` 把它隐藏了。

## Volta 上运行什么
<a id="what-runs-on-volta"></a>

该标志只降低下限（CMake 与运行时的设备检查），并替换 Volta 缺失的两个内在函数。它并不会处处选择"旧"内核：运行哪个
内核是在构建引擎时按架构决定的。

| 提示词路径的组成部分 | 在 sm_70 上 |
| --- | --- |
| MoE 专家层 | 可选开启（`STRATA_PF_WMMA=1`，仅当每块可见显卡都是计算能力 7 或以上；否则用 MMQ）：`gemm_iq_f16_grouped`：一次启动处理一组专家层，跑在 FP16 张量核心上，权重在共享内存中反量化（相比 MMQ 的 `dp4a` 内核提示词速度 +10%，[bench](../../bench/results/2026-10-07-v100-prompt-experts/README.md)；用 FP16 激活值而非 MMQ 的 q8_1，因此答案会有差异） |
| 稠密投影（已反量化的权重） | FP16 张量核心 GEMM（cuBLAS / CUTLASS `s884`） |
| BF16 投影（hyper-connection、路由器、索引器……） | 转换为 FP16 并跑在 FP16 张量核心上（#655、#540；`STRATA_BF16_TC=0`：cuBLAS BF16，在 Volta 上是一个 FP32 SIMT 内核） |
| decode 与 verify window 的 QSA 注意力（以及用 `STRATA_PROMPT_ATTN_OLD=1` 的提示词） | #540 的内核（更少的 shuffle，逐位精确；`STRATA_ATTN_PRE75=0`：其他显卡运行的那个） |
| QSA 提示词注意力，int8 / FP16 / K8V4 KV | 跑在 `mma.m8n8k4` 上的 `prompt_attn_v70_kernel`（`STRATA_PROMPT_ATTN_OLD=1`：decode 内核，一次一个查询） |
| QSA 提示词注意力，Q4_0 KV | decode 内核 |
| QSA 块打分 | warp 内核（张量核心打分器需要 sm_80） |

| decode / verify window 的组成部分 | 在 sm_70 上 |
| --- | --- |
| 显存中的路由专家层 | gfx906 的专家模式 8：码本网格与该组的激活值放共享内存，SwiGLU 融合（`STRATA_EXP_MODE=0`：其他显卡运行的 CUDA 布局） |
| 稠密的 2-4 列 GEMV 与输出头 | 带 Volta 行数表的交错 `native_mmvq_il`（`STRATA_MMVQ_IL=0`：`native_mmvq`） |
| hyper-connection 读取、up 投影 | gfx906 的隐藏延迟 up 内核（`STRATA_GR_FAST=0`：普通的那个） |

这三者都与它们所替换的内核逐位相同；在 V100-SXM2 上，它们把一个 verify window 的 GPU 工作从 22.8 ms 降到 21.2 ms
（[bench/results/2026-10-07-v100-decode-kernels](../../bench/results/2026-10-07-v100-decode-kernels/README.md)）。

## 实测
<a id="measured"></a>

一块 V100-PCIE-32GB、i9-7960X、Unsloth 的 `UD-IQ4_XS`（不是 setup 安装的模型）、int8 KV、MTP `--spec 2`；随机词
提示词的读取速度，单位 tokens/s，分别在关闭与开启 Volta 注意力内核（`STRATA_PROMPT_ATTN_OLD=1`）下测得：

| 提示词 token 数 | 旧的注意力路径 | `prompt_attn_v70_kernel` |
| --- | ---: | ---: |
| 7,194 | 1,010 | 1,164 |
| 28,650 | 1,063 | 1,251 |
| 114,338 | 973 | 1,123 |

每次运行的 decode 都是 38-51 tok/s，不受影响。性能剖析、一致性数字、needle 测试与限制都在
[bench/results/2026-10-03-v100-prompt-attn](../../bench/results/2026-10-03-v100-prompt-attn/README.md) 中。

该内核保证与不保证的内容：相比 FP64 参考，其误差约为 2-3e-6（输出尺度 3.6；被它替换的 FP32 内核：2e-6；在高、低两半
获得各自的累加链之前是 5e-6）；15 个 needle 全部找到；在 28,650 与 114,338 token 的提示词上贪心输出完全相同，在
7,194 token 的提示词上于回答后段出现差异（另一种求和顺序）。

## 检查与基准测试 Volta 内核
<a id="checking-and-benchmarking-the-volta-kernels"></a>

```sh
cmake -S . -B build -DSTRATA_PARITY_PROMPT_ATTN=ON ... && cmake --build build --target qsa_prompt_attn_parity
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5     # synthetic, no model; exit 0 = PASS
```

它把张量核心注意力与 FP32 内核以及一个 FP64 宿主参考作比较，并对两者计时。sm_80 以下的 Q4_0 用例会跳过。

## 限制
<a id="limits"></a>

- 两块 V100 的层切分未用该内核测量过。单请求时各卡轮流工作，所以切分增加的是专家缓存的容量，不是提示词速度。
- 同一个标志也覆盖 sm_60（Pascal），且有社区运行记录：两块 Tesla P40 做层切分（#1028，bench
  [2026-10-05-community-2x-p40](../../bench/results/2026-10-05-community-2x-p40/README.md)）、一块 Tesla P100 单独运行
  以及搭配一块 RTX 2070 SUPER 作为专家助手卡（#1069：单独 30.5 tok/s，带助手 35.8-38.4），以及一对 V100 + P100
  （#1079：从 0.1.39 到 0.1.40，把 P100 作为助手缓存后 decode +6-11%）。每条都只是某一位报告者的数字；本页自己的
  内核测量都只针对 V100。
- Volta 的张量核心只接受 FP16：模型的 BF16 权重会被转换，而超出 FP16 范围的权重会被钳到饱和值（上述检查中没有出现
  这样的权重）。

## 选定的 v0.1.41 内核
<a id="selected-v0141-kernels"></a>

设置 `STRATA_SM70_TABLE=1` 以在 Volta 上选用交错的稠密 matvec 表、专家模式 8 以及隐藏延迟的 HC up 投影。
带 Volta 行数表的交错 2-4 列 matvec 与隐藏延迟的 norm/up 都逐位等于默认内核的输出（贡献者的测量中，
V100-SXM2 上每次 verify window 的 GPU 工作 22.8 -> 21.2 ms）。Volta 默认表现在会选定这一主控设置。
单独的设置 `STRATA_MMVQ_IL`、`STRATA_EXP_MODE` 与 `STRATA_GR_FAST` 优先级更高。设置
`STRATA_SM70_TABLE=0` 可以显式选用原先的 decode 路径。

Volta 默认值还会选择 `STRATA_DEQUANT_WIDE=1`。该设置使用现有的向量存储反量化内核。CUDA
一致性测试检查它们的输出与窄内核逐位相等。设置 `STRATA_DEQUANT_WIDE=0`
选用窄的内核作为对照。

设置 `STRATA_EMB_REUSE_ACCOUNT=1` 让提示词规划器使用被半数输出复用的嵌入缓冲区空间。默认的计量方式
仍然保守。在实测的 8K 分块上，该选项把每一级的提示词借用减少约 80 MiB。它并不构成
普遍意义上的提示词速度收益。

对于实测的双 V100 IQ3_S arena，`STRATA_PREFILL_CPU_SHARE=auto` 与
`STRATA_PREFILL_STREAM_MIN=3072` 会在低于 3,072 个 token 时选用 CPU 辅助的暂存。
更大分块的选项可能改变提示词的舍入。不要为批量槽位启用它，也不要假定在其他模型上有同样收益。

**保留的 CPU 与 Volta 改动。** 这些改动保留模型格式、524,288 token 上下文下的 int8 KV
缓存、视觉、MTP、L2/L3 对话层级以及基于 NVMe 的 n-gram 表。目前尚未证明 20%
的端到端收益，也不据此作出任何性能保证。

在 sm_70 上，提示词路径的 QSA top-k 可以使用一次 prefill 的活跃块上界。当提示词的
活跃块能放进该内核时，该上界就会选用寄存器内核，即使很长的上下文容量使打分行的
步长很大。decode 与已捕获的计算图不传上界，沿用容量规则。

在一颗具备 AVX2、没有 AVX-512 也没有 AVX-VNNI 的 CPU 上，AVX-2 专家内核默认
为单个 token（NT1）服务 IQ3_XXS、IQ3_S 与 IQ2_S 的行。NT 指一组中的 token
数；NT1 就是每组一个 token。IQ3_S 与 IQ2_S 的反量化用 32 位载入读取打包的索引字节，
从而缩短网格装配。这些行按 ggml 自身的浮点顺序归约，并在归约之后才施加该格式的归一化
因子，因此一行结果逐位等于 ggml 的点积，包括次正规乘积。IQ4_NL 的 down 行不变。显式的
`STRATA_IQ_MT_MIN` 保留其自身的规则。

在每个非宿主的物理核心都有一个 worker 之后，线程池会把额外的 worker 固定到 worker
核心所允许的 SMT 兄弟线程上。SMT（同时多线程）指一个物理核心的第二个逻辑处理器。
宿主核心的兄弟线程排在最后。默认的放置方式不变：在实测的六核 CPU 上是五个 worker
加宿主线程。

参见[发布版选定项](../../improvements/RELEASE-v0.1.41-V100.md)
与 [prefill/decode 实测数据](../../benchmarks/v100-iq3_s-v0141-2026-10-08.md)。
这些选项不会改变已安装的服务，除非修改它的环境与引擎。不要为 IQ3_S 启用暂存缓冲区锁定。

## 共享根的返回延迟
<a id="shared-root-return-latency"></a>

双 V100 的受控研究实测：在缓存改动之前，重复共享根的 HTTP 延迟为 1,799.7 ms，
改动之后为 902.6 ms。引擎会把一个从磁盘恢复的根保留在有界的 RAM 缓存中。
无关的小对话与实时续写用例在这次样本中没有改善。不要把这个结果当作普遍的吞吐收益。
参见 [agentic 延迟实测数据](../../benchmarks/v100-iq3_s-prefill-wmma.md#agentic-short-turn-latency-2026-10-10)。

## 选定的 v0.1.43 性能改动
<a id="selected-v0143-performance-changes"></a>

[发布版评审](../../improvements/RELEASE-v0.1.43-V100.md) 列出了选定的移植项，以及不适用于 Volta 的改动。

- `STRATA_ROUTE_RESIDENT=1` 选用偏向常驻的路由。它的路由换入换出现在每 token 用一个 warp。
  该选项默认仍为关闭。
- Linux CUDA 可以让自适应的专家层拷贝与下一个 decode window 重叠。层切分仍被排除。用
  `STRATA_ADAPT_OVERLAP=0` 关闭它。常驻 RAM 源需要 `STRATA_ADAPT_OVERLAP_RESIDENT=1`。
- 自动的路由尾部跳过在专家缓存驻留率达到 90% 或以上时保持关闭。`STRATA_TAIL_SKIP_MAX_CACHED`
  改变该上限。显式的 `STRATA_ROUTE_TAIL_SKIP` 优先级更高。
- Linux CUDA 对规范的 Q2_0 模型包使用实测的 PCIe 份额。原生模型包本来就使用实测份额。

这些开关并不能为每一种 V100 配置确立速度收益。
参见[验证报告](../../benchmarks/v100-upstream-v0.1.43-2026-10-11.md)。

## 实测的 Volta 性能剖面（2026-10-20）
<a id="the-measured-volta-profile-2026-10-20"></a>

本节是本 fork 在自己的双 V100 PCIe 16 GB 机器（GPU 0 在 PCIe 3.0 **x4** 链路上，GPU 1 在
**x16** 上；Ryzen 5 3600 六核；94 GiB 内存）上的实测，使用 IQ3_S 模型包、524,288 token 上下文、
int8 KV 缓存、MTP 草稿、层切分 `16` 以及 `--pipeline-windows 2`。各分支的原始数据与方法在
[报告](../../bench/results/2026-10-20-v100-sm70-speed-stack/README.md) 中。

### 提示词路径处于局部最优：保留 8,192 token 的分块
<a id="the-prompt-path-is-at-a-local-optimum-keep-the-8192-token-chunk"></a>

`STRATA_PREFILL_TIMING=1` 显示一个提示词分块的时间去了哪里。在那台机器上，一个 8,192 token 的分块
花在专家上的 PCIe 往返约 2.1 s，计算约 1.96 s，整个分块耗时 3.9 s：该机器的 b/a 约为 8,800 个 token。

- **更大**的分块会为它的缓冲区借用更多专家缓存，于是缓存装下的专家更少、PCIe 往返变长：
  `auto:16384` 回落到 10,240 个 token，读取 32,768 token 的提示词慢 7.5 %（1,607 对 1,736 tok/s）。
- **更小**的分块要为每个分块重新付出固定的往返成本：4,096 个 token 读得慢 23 %，2,048 个 token 慢 54 %。
- **更深的流式环形缓冲**能沿 PCIe 预取更远，但那是专家缓存不再保有的显存：384 槽位与 512 槽位读取
  同一提示词都不比 192 更快。

**不要在这台机器上上调 `STRATA_PREFILL_AUTO_MAX`。** 通用的 8,192 上限就是此处实测得出的那个，
本 fork 的服务保留它。

### decode window 在等待三件事，性能剖面指出了它们
<a id="the-decode-window-waits-on-three-things-and-the-profile-names-them"></a>

`STRATA_VERIFY_PROFILE=1` 会打印每个 window 的 GPU 自身时间戳。在上述机器上，一个
verify window（2.4 到 2.5 个 token）约花 40 ms，第一级的时间构成是：

| 时间戳 | 它在等什么 | 第 1 级（每 window 毫秒） |
| --- | --- | ---: |
| 稠密投影、输出头、显存中的专家层 | 真正的工作 | 约 20 |
| `waitB` | 未命中专家中走 PCIe 的那部分，在本层的调用链内拷入暂存缓冲区 | 6.5 - 8.0 |
| `waitCPU` | CPU 池为本层计算的那些行 | 4.2 - 5.2 |
| `waitA` | 宿主为本层制定的计划（门铃握手） | 1.4 - 1.6 |

约 70 tok/s 的 decode 速率，对应的带宽下限大约是每 window 10 ms——这不是带宽问题：而是这三项等待，
以及 `waitB` 背后的 PCIe 流量（未命中的专家每 window 约 82 MiB，加上自适应层换入换出约 45 MiB）。

在这台机器上实测过的候选项都没能动它：上游的六个与十三个开关的 Volta 组合、`--pcie-mode dma`（更差：
DMA 的落位信号是一个宿主回调，而六核宿主上回调的延迟大于它所等待的那次拷贝）、`--pcie-frac 0`、
`--pool-workers 10`、`--adapt-swaps 24`、`--adapt-every 8` 与 `--adapt-every 0` 都在两次基线分支的噪声之内。

### 由设备置位的 DMA 路径
<a id="the-dma-path-with-the-device-raised-flag"></a>

`--pcie-mode dma` 在第二条流上拷贝未命中专家中走 PCIe 的那部分，与本层的计算并行，于是拷贝内核
在本层调用链内部造成的等待消失。它的落位信号是拷贝流上一个单线程内核：该内核在拷贝之后运行
（流为它们定序），并用一次系统栅栏写入映射的置位标志，因此自旋等待该标志的计算图无需宿主在 window
内部调用驱动就能看到该值。引擎对切分交接与门铃本来就使用同样的模式。

在上述机器上：**带拷贝内核为每 token 15.39 / 15.79 ms，带 DMA 路径为 14.58 / 14.53**
（-6.6 %），合并统计的 tok/s 为 66.5 / 65.8 对 72.9 / 70.2（+8.1 %）。使用旧的宿主回调的那条路径
测得每 token 19.07 ms。

**默认值仍是拷贝内核。** DMA 路径会向一块其 window 自旋等待宿主标志的显卡排入一次拷贝，而这正是
上游 issue #31 的那个模式（一次拷贝会阻塞在驱动里直到 window 结束）。设备置位的标志把宿主回调从
window 中移除了；宿主的 `cudaMemcpyAsync` 仍然存在，所以该模式并未消失。请配合停顿看门狗
（`STRATA_WATCHDOG_S=300`）作为兜底使用。

### Volta 默认值表
<a id="the-volta-defaults-table"></a>

`src/core/nv_defaults.cpp` 为每种架构维护一张表，收录那些**逐位等于默认引擎输出**且在该显卡上更快的开关。
sm_70 的表保存 Volta 的主控开关（`STRATA_SM70_TABLE=1`：交错的稠密表、专家模式 8 布局，以及隐藏延迟的
hyper-connection up 投影）以及用于原生权重转换的 `STRATA_DEQUANT_WIDE=1`。
`STRATA_SM70_PERF=0` 关闭该表。用户提供的任何设置始终优先。实测的服务配置见
[service/strata-v100.service.d](../../service/strata-v100.service.d/README.md)。

## 双 PCIe 流水线服务配置
<a id="dual-pcie-pipeline-service-profile"></a>

[近期的上游 PR 评审](../../benchmarks/v100-upstream-pr-inventory-2026-10-15.md)
把 V100 流水线报告纳入了上游 PR #2395。经过测试的本地配置使用位于 PCIe x4 的
GPU 0 与位于 PCIe x16 的 GPU 1。请保持 `0,1` 的 GPU 顺序与显式的层切分 `16`。
不要假定链路等宽。

使用 `STRATA_KV_GROW=1` 时，请采用带[流水线 KV 增长修复](../../benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)
的构建。原先的流水线 decode 循环在长回复期间不会增长映射的 cell。即使配置的上下文
还有剩余空间，它也可能因 CUDA 非法内存访问而失败。该修复会先等待投机回滚与自适应
拷贝，再增长两级缓冲区，然后恢复流水线。只看短输出的速度检查测不到这一边界。

服务模板使用一份独立的私有配置，`strata-iq3_s-pipeline.json`。请从现有配置准备它：

```sh
python3 - <<'PY'
import json
from pathlib import Path

source = Path("strata-iq3_s-arena.json")
target = Path("strata-iq3_s-pipeline.json")
config = json.loads(source.read_text())
args = config["args"]
if "--pipeline-windows" in args:
    args[args.index("--pipeline-windows") + 1] = "2"
else:
    args.extend(["--pipeline-windows", "2"])
target.write_text(json.dumps(config, indent=2) + "\n")
target.chmod(0o600)
PY
```

请让两份配置都保持私有。不要提交凭据、模型路径或对话数据。只有在新配置存在之后再安装
服务模板。请保留先前的引擎与服务设置以便回滚。该配置不改变模型、量化、上下文容量、
视觉设置或缓存预算。

`--pipeline-windows 2` 需要两级 GPU 与 MTP。它不是批量槽位配置。它会为第二个校验器与
各级快照预留额外内存。没有实测数据就不要把它套用到别的机器。结果、限制与复现命令见
[本地评估报告](../../benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)。

## 前缀感知的对话存储
<a id="prefix-aware-conversation-storage"></a>

L2 为匹配的对话前缀保留一份共享的 K/V 底层存储。一条短分支拥有自己的 token 元数据与
发生变化的后缀。它不会移除原来的长续写。缓存在记账时只计入共享负载的分配一次，
包括它接纳一条分支或驱逐另一个属主的时候。写入只会分离发生变化的区间。
分配预估包含缓冲区目录的增长与共享后缀的写入。

K/V 复用与递归状态恢复是两个独立的操作。引擎可以跨一个分支点共享 K/V，而无需在
那个位置拥有递归状态。它必须恢复一个更早的有效状态并处理剩余的提示词。
显式的固定（pin）可以保留一个精确的边界。普通历史使用有界的锚点，而不是每轮
保留一个永久状态。同一套锚点选择规则适用于每一个切分级。保存在输入内的分支会冻结
其源归档。之后的追加可以共享那份源的 K/V，而不会轮换掉它原始的续写状态。
活跃列表也会接纳一个 predecode 输入锚点。回复改写可以从那里续写，而不必加载
被废弃回复中一个已停放的停放状态。

在改写之后共享已保存的底层存储之前，引擎会把保留的前缀与当前的 K/V 字节作比较。
这包括部分页中的活跃 cell 以及已完成的索引器行。它不包括未使用的 cell 与那个
移动的备用行。仅 token ID 相同并不允许复用。如果重放在边界 cell 上产生了变化，
引擎可以共享一个更短的、已验证过的前缀。

L3 使用前缀引用与 K/V 增量面板。一条分支可以引用一条更长链的较短区段，而不截断
那条链。有界的 `.states` 槽位保存保留的递归、PLE、索引器与草稿状态。新面板不会在
每一轮额外内联一份状态拷贝。附属文件写入失败会让整次缓存写入失败，而不是使用
一个无界的回退方案。首个被存储、且后续输入会消费的生成结尾可以保留一个普通
状态槽位。密集的输入检查点不会轮换掉这一原始续写点。它的标记能在重启后存活，
且不增加槽位上限。遗留的内联状态仍可读取。它们会占用磁盘空间，直到所属的链被
驱逐或替换。遗留面板内部的截断仍要求读取方校验那个完整面板的校验和。

L2 的字节上限、槽位上限与空闲内存下限仍然适用。L2 容量驱逐的牺牲者可以移入 L3。
磁盘限额包含链与附属文件的字节数。固定（pin）会消耗状态存储。有限的预算
仍可能驱逐对话。这些改动不会提高活跃上下文上限，也不会消除长上下文 decode 的开销。

已保存的输入提示词边界能识别一个已被废弃的生成尾部。客户端可以替换该尾部，
而不必在每一轮新建一个对话。该边界之前的分支会保留其源的续写。附近存在一个
检查点并不能证明这是一次终止性改写。缺少边界来源记录的记录使用保守的分支路径。
如果重放在输入边界之前改变了 cell，则替换要求剩余的原始输入 token 与图片 key
都匹配。一次输入编辑仍会创建独立的分支。

### 当前的原生分支检查
<a id="current-native-branch-check"></a>

一次原生 V100 检查使用 300,000 token 的父对话与 20,008 token 的分支。未设置
精确的边界固定。首次分支复用了 16,384 个 token，并处理了剩余的前缀。随后二十轮
输入不断增长的对话复用了常驻的输入锚点。它们的提示词耗时为 356 到 407 ms。

在这些轮次之后，短分支复用了 20,007 个 token。原始的长续写复用了 300,015 个 token，
并产出与其首次返回相同的输出。L2 存储峰值为 8,508,704,232 字节，分布在四个停放组上。
未发生容量驱逐。归档历史保持为六个检查点。这些数字描述的是缓存分配，不是进程 RSS 或显存。

一次 300,000 token 的仅 L3 保留检查也通过了。该检查早于最终的 replay 边界修正。
引擎在 20 轮输入不断增长的对话之前重启。两次长返回都复用了 300,015 个 token。
最后的短返回复用了 20,007 个 token。存储中持有两条链，合计 6,189,411,036 字节，
其中包括 1,298,695,712 字节的有界状态附属文件。已分配的磁盘空间为 6,189,428,736 字节。

在该检查中，仅磁盘模式的追加提示词耗时为 7.84 到 9.73 秒。仅磁盘运行仍要为前缀
校验与状态写入付出成本。这次存储与恢复的结果并不意味着磁盘复用能达到 RAM 速度。

最终版的写入器还用那次 300,000 token 运行中捕获的 K/V 与运行状态进行了实测。
短分支引用了 16,384 个源 token。它的链持有 55,581,873 字节，而不是匹配前缀的
第二份拷贝。源链持有 4,580,646,417 字节。总存储为 5,226,544,706 字节，其中包含
为这次重放保留的状态。原始的 300,015 token 状态与 20,023 token 的分支都以精确的
状态负载与 K/V 得到恢复。

最终版的原生写入器还通过了一次 40,000 token 父对话的检查，包含一个 20,008 token
的分支、一次重启以及 20 轮输入不断增长的对话。它保留了两条链。在反复停放之后，
短链仍引用 16,384 个源 token 并持有 55,889,222 字节。总存储为 1,851,520,352 字节，
其中包括 1,180,632,496 字节的有界状态附属文件。原始的父对话返回复用了 40,014 个
token 并产出相同的输出。

另一次最终检查固定了 19,998 token 的边界。短链引用了全部 19,998 个 token 并持有
429,625 字节。在四种请求情形下，生成的 token ID 与全部九个规范的主状态哈希字段
都与 RAM 参考一致。

全部九项专注的对话与会话测试都通过。全部 84 项缓存工具与 prefill 工具测试都通过。
AddressSanitizer 与 UndefinedBehaviorSanitizer 检查通过了 4,394 项 RAM 断言与
1,863 项磁盘断言。磁盘检查包括分支保留、有界状态槽位、重启恢复、缺失的附属文件
以及状态发布失败。一项专门的检查会在输入不断增长时替换 20 个生成的尾部。它校验
重启后的最新图片、原始输入锚点，以及对历史分支的拒绝。

### 较早的无界归档检查
<a id="earlier-unbounded-archive-checks"></a>

以下结果使用的是较早的保留策略。它们是历史证据，不是上文所述有界策略的验收结果。

一次确定性的单 GPU 引擎检查使用 12,293 token 的合成提示词、八次替换了回复尾部的
对话续写，并在续写之间插入一次无关的完整提示词读取。一份归档保留了该链。在那次
完整读取之后，归档从 28 个检查点增长到 35 个。最后三次扩展复用了 189,330,432 到
189,578,880 个 KV 字节，耗时 193.1 到 193.9 ms。未发生容量驱逐，也没有宿主 RAM
接纳失败。该检查并不能确立 340,000 token 上下文下的 decode 速度。

一次单块 V100 的冒烟运行使用 12,440 token 的合成提示词、八次短续写，以及在活跃
检查点窗口之前的编辑。该运行使用了这些限额：

```text
--prompt-cache 3 --prompt-cache-every 512
--conversation-cache-mib 8192 --conversation-cache-slots 2
--conversation-cache-min-tokens 0 --conversation-cache-min-free-mib 1024
--conversation-cache-disk <scratch-directory>
--conversation-cache-disk-gib 32
--conversation-cache-disk-min-free-mib 128
```

L2 归档在八次续写之后保留了 31 个检查点。在 token 6,144 处的一次编辑恢复了历史的
L2 检查点。引擎处理了剩余的 6,296 个 token。随后三个无关请求迫使 L2 发生驱逐。
在 token 7,168 处的一次编辑恢复了一个 L3 检查点。引擎处理了剩余的 5,272 个 token。
两次请求都产出与其冷启动参考请求相同的输出与主模型状态哈希。日志显示了 L2 到 L3
的溢出以及 L3 到 L2 的提升。L2 占用始终低于 8 GiB 与两个槽位。

已安装的服务还用其现有设置通过了一次 HTTP 冒烟检查：524,288 token 上下文、
25 GiB 的 L2、32 个缓存槽位以及默认的活跃检查点数量。十轮短对话完成。随后对
第五条用户消息的一处改动，为一个 293 token 的提示词从 L2 恢复了 124 个 token。
此前的续写引用复用了 48 个 token。运行日志确认这是一次 L2 恢复，而不是 L3 恢复。

最早几次的重启检查失败是因为宿主内存，不是缓存代码。这台宿主使用非 ECC 内存且
没有错误日志，所以这次故障没有留下硬件记录。一项仅 CPU 的图案测试发现有一个 bit
总是回读出错的值。随后同一提示词的两次冷启动引擎运行给出了不同的模型状态指纹，
因此无法进行状态比较。

操作员取出了两条 16 GB 内存条，保留两条 32 GB 内存条。对已安装的 62.7 GiB 内存
中 54 GiB 做的图案扫描未发现故障。随后两次冷启动引擎运行给出完全相同的模型状态
指纹，重启与提升的一致性检查也都通过：

- 整个引擎重启之后的 L3 恢复与冷启动参考一致；
- L2 提升与冷启动参考一致；
- 模型状态指纹与输出完全相同。

有缺陷的内存条是被取下的两条 16 GB 内存条中的一条。那一组没有料号，序列号为
00000000。请逐条单独测试以确认哪一条有缺陷。

有限的缓存预算仍会限制历史长度。一个同时被 L2 与 L3 驱逐的检查点无法再用于复用。
引擎必须重新处理那段前缀。

## 增量式首 token 输出时间
<a id="incremental-time-to-first-output"></a>

引擎在改变提示词状态之前保存一个输入检查点。新路径在 GPU 上捕获该状态。
它在第一个输出 window 刷出之后才把状态拷贝到宿主内存。缓存在请求结束之前
收到同一个输入检查点。

`STRATA_INPUT_ANCHOR_DEFER=0` 选用同步的对照路径。仅 prefill 的请求保留同步路径。
GPU 快照分配失败时，只要输入状态尚未改变，也走那条路径。串行循环复用其已有的
快照。两级流水线在两级合计使用约 113 MiB 的临时显存。它在宿主拷贝结束后释放
这块内存。

实测的服务配置还启用了 `STRATA_CKPT_POOL=1` 与 `STRATA_CKPT_THREADS=1`。
这些设置复用检查点存储，并并行拷贝各级检查点。它们不改变 KV 容量、KV 格式
或模型量化。

参见 [2026-10-21 TTFT 报告](../../benchmarks/v100-incremental-ttft-2026-10-21.md)。
它测量的是客户端所见的流式输出，不是完整响应时间。拷贝仍发生在请求期间。
不要把更低的提示词时间解读为 decode 吞吐的等量提升。

### prefill 之后的专家缓存接纳
<a id="expert-cache-admission-after-prefill"></a>

`STRATA_PREFILL_PROMOTE=N` 允许在 prefill 归还缓存借用时，每一级最多进行 `N` 次
专家替换。引擎默认值为 `0`（关闭）。实测的服务配置选用 `2048`。该设置把频繁使用的
专家保留在后续借用不会用到的 GPU 槽位中。它替换同一层中使用较少的专家。路由计数在
每次新的借用时重置。

每一次替换先把冷专家拷贝到 GPU 上的一个借用槽位，然后把热专家从宿主内存拷入那个
稳定的槽位。宿主拷贝预算与 GPU 缓存容量都不增加。缓存映射只在拷贝全部完成之后才改变。
前置条件失败时映射保持不变。

该路径需要完整的宿主 arena 且没有批量槽位。在存在 RAM 补集、对等层或助手层、
`STRATA_PREFILL_HELP`、远程层或 prefill 头别名时它是关闭的。当它拒绝一个显式设置时，
引擎会记录原因。支持可增长的 KV。活跃槽位映射不一致的条目会被跳过。

该选项改变专家的放置方式。既有的常驻感知路由与 CPU/GPU 舍入可能改变模型输出。
它不改变专家权重、模型量化、KV 格式或 KV 容量。

实测的读取次数、四分支 TTFT 结果与完整请求开销见
[缓存接纳与反量化报告](../../benchmarks/v100-prefill-cache-admission-2026-10-21.md)。

## 更多可选开启的开关（CUDA 12 实验性构建）
<a id="more-opt-in-switches-cuda-12-experimental-build"></a>

这些开关都不是默认开启的。每一个都由其贡献者在 V100 上实测，而非由维护者们实测。

| 开关 | 它做什么 | 贡献者的数字 |
| --- | --- | --- |
| `STRATA_FUSE_DECODE=1` | #1097 的 decode GEMV / MoE 算子融合（gate/up + SwiGLU 合成一个内核，SwiGLU + 量化器合成一个内核，一个向量的全部激活值镜像一次遍历完成，两个 BF16 投影合成一个内核）；逐位等于未融合的内核 | 每 token 省 4-16 us，端到端看不出来 |
| `STRATA_PF_WMMA=1` | FP16 张量核心的提示词专家内核（见上文）；也接管 IQ4_NL 的 down 投影（K=640，#1771） | 提示词速度 +10%（#1401） |
| `STRATA_PF_WMMA2=1` | 第二个提示词专家内核，用于 IQ2_S 的 gate/up 与 IQ4_NL 的 down（#2356） | 提示词速度 +2.5-4.3% |
| `STRATA_ARGMAX_ROWS_SM70=1` | sm_7x 上的多块贪心选取（#2357），逐位等于单块选取 | decode +0.7-0.8% |
| `STRATA_GR_NORM_FAST=1` | CUDA 上的寄存器分裂 hyper-connection norm（#2358） | decode +0.5-0.6% |
| `STRATA_Q8_GATHER_WIDE=0` | 关闭实验性构建在计算能力 7.5 以下使用的 8 宽 int8 KV gather（#1095） | 该 gather 内核 +11% |
| `STRATA_GDN_REC_WARP=2` | 显式选用链分裂的 DeltaNet 递归。`1` 选用无同步的对照。本 fork 的常规分派在 Volta 上已经使用 `STRATA_GDN_REC_FAST`。若要原先的流水线对照，请同时设置 `STRATA_GDN_REC_WARP=0` 与 `STRATA_GDN_REC_FAST=0`。`STRATA_GDN_CHUNK=1` 选用分块路径。 | 贡献者：递归部分 +6-9%；相对本 fork 不是新的收益 |
