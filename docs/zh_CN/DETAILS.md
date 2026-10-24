# Strata - 细节

<a id="strata---the-details"></a>

Strata 的技术面：每一个实测数字、API、图像、所有设置以及引擎如何工作。初来乍到？从
[README](../../README.md) 开始；一步步安装在 [INSTALL.md](INSTALL.md)，模型在
[MODELS.md](MODELS.md)，常见问题在 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

<a id="the-technical-side-of-strata"></a>

> **本页内容：** [速度](#speed-measured) · [其他 GPU](#other-gpus-estimated) · [选哪个模型？](#which-model) ·
> [要求](#before-you-start) · [Windows](#windows) · [Linux](#linux) · [API](#using-it) ·
> [MCP 工具](#tools-from-mcp-servers) · [MCP 服务器](#manage-strata-from-your-ai-assistant-mcp-server) ·
> [图像](#images-vision) ·
> [疑难排查](#troubleshooting) · [工作原理](#how-it-works)

---

## V100 fork 中的 upstream v0.1.36

<a id="upstream-v0136-in-the-v100-fork"></a>

本次更新包含 upstream 的 0.1.32 到 0.1.36。Upstream 源码是
[`36fa455`](https://github.com/Niko1221/Strata/commit/36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca)。
见 [0.1.36 发布](https://github.com/Niko1221/Strata/releases/tag/v0.1.36)。

<a id="this-update-includes-upstream-releases"></a>

Fork 保留对 Tesla V100（`sm_70`）的支持。用 CUDA 12.x 本地编译引擎。CUDA 13 无法编译
`sm_70` 代码。普通 V100 构建不启用实验性 Pascal 选项。

<a id="the-fork-keeps-support-for-tesla-v100"></a>

V100 的 prompt 路径保留 FP16 tensor-core 乘积、紧凑 expert 缓冲区，以及 fork 的两内核
超连接窗口。BF16x2 选项在 V100 或 Turing 卡上无效果。新的融合 prompt expert 需要 `sm_80`
或更新。新的 cluster decode 内核需要 `sm_90` 或更新。更旧的卡保持它们之前的内核。

<a id="the-v100-prompt-path-keeps-fp16"></a>

本次更新还包含 setup 与服务器修复、保存的 expert profile、请求取消记录、更新脚本、低 RAM
选项，以及 MCP 服务器。Linux 异步文件读取器现在用正确的完成数组偏移。PLE 读取器在接受读请求
之前记录它的行数。这两处更正防止无效完成与错误的 PLE 行。

<a id="the-update-also-includes-setup"></a>

**在两块 Tesla V100 16 GB、CUDA 12.8 和一台 AVX2 CPU 上的验证：**

<a id="verification-on-two-tesla-v100-16-gb-cards"></a>

- `sm_70` 的 Release 构建通过。
- 工具与 setup：256 个 Python 测试通过。
- 服务器：运行 158 个 Python 测试；跳过九个。整套通过。
- CTest：60 个测试通过。两个融合 expert 测试因 V100 不支持那些内核被跳过。
- 仅限 AVX512 的 expert 测试在这台 CPU 上被排除。传统 PLE 捕获测试因其固定捕获文件不可用
  被排除。合成 PLE 读取器测试和实时 PLE 推理通过。
- Q2_0 模型、原生 PLE 表、MTP 草稿、int8 KV 缓存、20/28 层级切分，带
  `STRATA_PREFILL_BF16X2=1` 运行。短回答、6,063-token prompt、原生工具调用、流式回复都通过。
- 实时 OpenAI 检查五个用例全部通过。完整的 Hermes 工具工作流通过。
- 默认设置下，30,063-token prompt 和短回答通过。
- 重新构建的 CUDA 视觉编码器正确识别了一张红色测试图像。
- 客户端在 prefill 期间断开之后，下一个请求完成了。该请求在队列里等了 16.11 秒。这项检查
  并未确立即时取消。
- 需显式开启的 API 监视器在真实浏览器里显示了已完成与已断开的请求。请求检查与渲染通过，
  浏览器无错误。

6,063-token prompt 以 621.1 prompt token/s、63.9 输出 token/s 运行。这些数值是冒烟检查，
不是受控的速度对比。下面更早的基准表仍标明产生它们的构建。AMD、Windows、跨代 GPU 切分以及
新的 `sm_80`/`sm_90` 内核在这台机器上没有测。

<a id="the-6063-token-prompt-ran-at"></a>

## V100 fork 中的 upstream v0.1.39

<a id="upstream-v0139-in-the-v100-fork"></a>

本次更新集成到 [v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39) 为止的
upstream 发布。Fork 保留它的 V100 tensor-core expert（带紧凑 192 槽流式环的原生 FP16 prompt
路径）、attention 改动、非对称 GPU 切分和磁盘对话缓存。要求更高架构的 upstream 新内核保持受
门控：融合 prompt expert 需要 `sm_80` 或更新，cluster decode 内核需要 `sm_90`，更旧的卡保持
它们之前的内核。实验性 Pascal 选项保持关闭。

<a id="this-update-integrates-the-upstream-releases"></a>

CUDA 和 HIP 引擎去掉 upstream 的 `--trim-stage-weights` 参数和 `STRATA_STAGE_TRIM`：显式的
`--layer-split` 本来就只加载每个分页自己的层，所以修剪是自动的，那些设置不再被接受。独立的
实验性 Intel（SYCL）引擎保留它自己的需显式开启路径。

<a id="the-cuda-and-hip-engines-drop"></a>

批处理槽会话在主会话、原生 head、logits 缓冲区和 MTP 草稿层之后分配，所以引擎的 fit 检查看
到的才是每块卡上真正剩下的内存。装不下的数量在启动时回退到更少的槽，或不带批处理槽（一次一个
请求）：HTTP 服务器保持就绪，配置保持不变。在本机柜的 524,288-token 上下文、20/28 切分上观察
到此现象。分配顺序、回退和每槽内存代价见 [BATCHING.md](BATCHING.md)；最终的准入时刻数字在
基准报告中。

<a id="batch-slot-sessions-are-allocated"></a>

**在两块 Tesla V100 16 GB、CUDA 12.8 和一台 AVX2 CPU 上的验证：**

<a id="verification-on-two-tesla-v100-16-gb-cards-1"></a>

- `sm_70` 的 Release 构建通过。
- CTest：所有修复后运行 81 个受支持测试；79 个通过，两个仅限 `sm_80` 的内核测试在 V100 上
  被跳过（可运行集的 100%，76.8 秒；两个 IQ-fixture 用例需要设好打包工具的 Python 路径，
  然后它们通过）。仅限 AVX-512 的 expert 测试在这台 Ryzen CPU 上被排除。传统 PLE 捕获测试
  因其固定捕获文件不可用被排除；实时 PLE 路径通过。
- 服务器：运行 275 个 Python 测试；跳过七个。整套通过。
- 工具：运行 396 个 Python 测试；跳过一个。整套通过。
- 最终构建上的定向检查：`tools.test_setup_parallel`（setup 的 Volta `"parallel": 2` 默认及其
  推荐）以及并发的每请求历史测试通过。
- 在回退服务器上完整 524,288-token 上下文的 HTTP 冒烟（请求 2 个槽，一次服务一个）：
  Chat Completions、Responses API 非流式及其流式 SSE 事件、一次真实模型的函数工具调用
  （`add_integers(17, 25)` 答 42）带一次无状态追问、Anthropic 端点、一个畸形工具请求以 HTTP
  400 应答、严格 JSON-schema 输出（`{"ok": true}`），以及一个真实红色 PNG 视觉输入被识别为
  红色。全部通过。
- 32,768-token 上下文下的最终 HTTP 并发：四个各自标记的流式请求通过两个服务槽完成。指标
  显示两个活动槽和排队等待的超额请求。token 发出间隔重叠了 6.1 秒和 6.2 秒。实时网页监视器
  也显示两个槽在解码、一个请求在等。
- 32,768-token 上下文、两个槽下的批处理精确性（`tools/batch_test.py`，贪心，
  `--adapt-every 1000000 --pcie-frac 0`）：每个槽生成的 128 个 token 与它单独运行时逐 token
  一致；两次准入花 1.3 秒，两个 128-token 回答（合计 256 token）在 5.8 秒内解完，聚合
  44.0 tok/s（从首 token 起算是 49.6 tok/s）。
- 带三个批处理槽的 L3 磁盘层在 126.1 秒内通过全部九项精确 token 一致检查：A2 精确复用 152
  个 token、下一个请求精确 299——最长有效前缀胜出，所以磁盘上存在的 25-token 和 172-token
  记录没能取代更长的 152 和 299 token 状态。一个 4,021-token 请求读了 512-token 的首段
  prompt，在 1,024 处让位（`BYIELD`），以复用 1,024 加读取 2,997 继续，首个检查点被保留。

<a id="-the-release-build-for-sm_70-passed"></a>

**两块 V100 上的 prefill（2026 年 10 月 4 日，最终构建）。** 集成构建对照此前安装运行时的一
次受控前后运行，一次一个请求，冷，524,288-token 上下文下 int8 KV 的 Q2_0，20/28 层级切分和
MTP（`--spec 8`）不变；每个尺寸两次全新请求，中位数：

<a id="prefill-on-the-two-v100s"></a>

| Target prompt | Before (tok/s) | Integrated (tok/s) | Change |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

每个数值是两次全新请求的中位数（`reused=0`）；样本为偶数时中位数取两次重复的平均。增益很
小，在 2K 和 32K 上落在基线自身的重复散布之内（其 5.2% 和 1.26% 的中位数）；只有 8K 增益
明确超过该散布（1.09%）。每尺寸两次重复不是统计结论。这些是最终构建的实测值（引擎 SHA-256
`e9ad337a…`）。

<a id="each-value-is-the-median"></a>

六个配对的首 token 全部与基线一致。不主张完整输出一致：自适应 expert 层为集成构建在 GPU 和
CPU 上保留不同的 expert 组合，两条路径舍入不同。不主张任何 decode 或质量变化。完整报告在
[`benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md`](../../benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md)。
每请求的首 token 时间报告现在在 solo 与槽路径上遵循同一规则：从请求开始到其首个生成 token 的
墙钟时间，在 `/metrics` 里按请求记录（`ttft_ms`）。

<a id="all-six-paired-first-tokens-matched"></a>

## 保留的 CPU、PLE 与 Volta 改动

<a id="retained-cpu-ple-and-volta-changes"></a>

这些改动在 fork 中保留。它们保留模型格式、524,288-token 上下文下的 int8 KV 缓存、视觉、MTP、
L2/L3 对话分层，以及 NVMe 支撑的 n-gram 表。尚未证明 20% 的端到端增益，也不附带任何性能保证。
最终测量仍在等待。

<a id="these-changes-are-retained-in-the-fork"></a>

- **AVX-2 i-quant 行（IQ3_XXS、IQ3_S、IQ2_S）。** 在有 AVX2、无 AVX-512、无 AVX-VNNI 的
  CPU 上，AVX-2 expert 内核默认以 NT1 为这三种格式服务单 token（NT1）的 gate/up 行。NT 指
  一组中的 token 数；NT1 是每行一个 token 的分组。IQ3_S 和 IQ2_S 的 decode 用 32 位加载读取
  打包索引字节，缩短了网格组装。这些行按 ggml-cpu 自己的 float 顺序归约，归约之后才施加格式
  归一化因子，所以每一行与 ggml 的点积逐位相同，包括非规格化乘积。IQ4_NL 下行不变。显式的
  `STRATA_IQ_MT_MIN` 保持其规则并覆盖此默认。
- **CPU 池 worker 在 SMT 兄弟上。** 在每个非 host 物理核都有一个 worker 之后，池把额外的
  worker 绑定到 worker 核所允许 SMT 兄弟超线程上。SMT（同时多线程）指物理核的第二个逻辑
  处理器。host 核的兄弟排最后。进程允许集合之外的 CPU，以及 worker 或 host 已占的任何 CPU，
  都跳过。默认放置不变：池不超过物理核数，即在实测六核 CPU 上是五个 worker 加 host 线程。
  Windows 上额外的 worker 像以前一样不绑定。
- **L3 后台写入器的亲和性。** 异步直接 L3 回退写入器在其线程开头恢复允许的 CPU。新线程继承
  服务线程的绑定，而会话循环把它保持在保留 host 核（第一个物理核、逻辑处理器 0），所以改动
  前每个写入器在一个逻辑处理器上跑。池在构造时（那次绑定之前）捕获允许的 CPU；每个写入器
  稍后在已绑定的线程上创建，进入时捕获那份保存集合的副本并在入口处恢复它。GPU host 线程保持
  绑定。L2 暂存、快照格式、FNV 哈希、磁盘预算和后续写入的顺序不变。两个 NVMe 对话文件上的
  strace 冒烟测试：旧写入器继承掩码 [0] 且不恢复；每个新写入器在打开文件前恢复 [0..11]。在
  实测机（六核 12 线程）上每次恢复都返回 0。RAM A/B/A 复用两臂都是 4,620 个 token 中复用
  4,615。trace 不是计时证据。匹配的 HTTP 研究未达到 20% 目标。见 [CPU 与 SSD
  研究](../../benchmarks/v100-iq3_s-prefill-wmma.md#cpu-and-ssd-study-2026-10-10)。
- **SSD-PLE 页缓存与跨界行。** n-gram（PLE）读取器的有界页缓存现在从两个已缓存页拷贝一个
  跨页边界的行。此前这种行总要重读 SSD。读取器从第一页拷贝行的头部、从第二页拷贝尾部。某个
  页缺失时目标保持不变，所以部分命中仍是未命中。这覆盖 90 字节的 IQ4_NL 行（生产默认）和
  110 字节的 Q5_0 行。
- **Volta prompt top-k 界限。** 在 sm_70 上，prompt 路径的 QSA top-k 可以用一次 prefill 的
  活动块界限。该界限在 prompt 的活动块装得下时选择寄存器内核，即使很长的上下文容量让分数行
  跨步很大。Decode 和捕获的图不传界限、保持容量规则。其他 CUDA 设备保持容量规则。

<a id="-avx-2-i-quant-rows-iq3_xxs-iq3_s-iq2_s"></a>

## 速度（实测）

<a id="speed-measured"></a>

### Tesla V100 fork 基准

<a id="tesla-v100-fork-benchmark"></a>

当前的 Strata-V100 数字于 2026-10-02 在一块单卡、散热良好的机器上测得（见下）。用
Qwen3.8-Flash-Next Q2_0，Tesla V100-PCIE-16GB（`sm_70`）、Ryzen 5 3600（六核十二线程）、
48 GB DDR4-3200、CUDA 12.8、Linux。引擎是 fork 的 0.1.31 版（CMake 工程版本），安装构建提交
`78417ea`（SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`）。
运行时设置：262,144-token 上下文，int8 KV 缓存，自动 prefill，MTP 推测解码（`--spec 8`，草稿
下限 0.70），配对 expert 变体，700 MiB 视觉预留。Prompt 与 decode 计时取自服务器的
`/metrics` 端点，不是客户端墙钟。Expert 缓存启动时持 4,198 个槽。

<a id="the-current-strata-v100-numbers"></a>

**单卡散热良好基准（2026 年 10 月 2 日）**

<a id="single-card-cooled-benchmark"></a>

只在物理 GPU 1 上测：一块 Tesla V100-PCIE-16GB，PCIe Gen3 x16 链路。引擎和视觉编码器跑在
GPU1。GPU0 不用于推理。跑了八个目标尺寸，从 1K 到接近 256K（配置的容量是 262,144 token）。
每个尺寸三次全新请求。每个请求正好写 256 个输出 token。表格给中位数。

<a id="measured-on-physical-gpu-1-only"></a>

| Target prompt | Median prompt tokens | Prefill (tok/s) | Decode (tok/s) | Max GPU temp |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

† prompt 期间软件热降频处于活动状态（84 °C）。

<a id="-software-thermal-slowdown-active"></a>

每次受测请求之前，GPU1 至少冷却 120 秒。闸门是：温度不高于 55 °C、稳定 15 秒、且没有软件热
降频。闸门每 3 秒检查一次。独立的遥测以 1 Hz 运行。实际起始温度为 51-55 °C。实际等待为
121.4-588.7 秒（约 2-9.8 分钟；128K 和 256K 请求需要最长等待）。种子 20261002。全部 24 个
请求都是全新的（`reused=0`）。

<a id="before-every-measured-request"></a>

这张卡直到 64K 都没有软件热降频；那里的最高温是 81 °C。128K 和 256K 的 prompt 把无风扇的卡
加热到 84 °C，尽管冷却起步，那些 prompt 期间它仍降频了。长请求可能在冷却起步之后降频。所有
请求都保留；没有一个被剔除。

<a id="the-card-showed-no-software-thermal-slowdown"></a>

原始数据：
[`matrix.json`](../../bench/results/2026-10-02-v100-single-cooled/matrix.json)、
[`completed-requests.json`](../../bench/results/2026-10-02-v100-single-cooled/completed-requests.json)、
[`protocol.json`](../../bench/results/2026-10-02-v100-single-cooled/protocol.json)、
[`summary.json`](../../bench/results/2026-10-02-v100-single-cooled/summary.json)，以及
[`gpu.csv`](../../bench/results/2026-10-02-v100-single-cooled/gpu.csv)（连续 1 Hz 遥测）。

<a id="raw-data"></a>

用 `bench/run_v100_bench.py` 重复这一扫描：目标
`1024,4096,8192,16384,32768,65536,131072,256073`、`--max-tokens 256`、`--seed 20261002`、
`--repeats 3`，以及每次受测请求之前同样的冷却闸门。测试框架不会自动暂停。请在各个受测请求
之间施加冷却闸门，与本次运行相同。

<a id="repeat-the-sweep-with"></a>

**更早的测量。** 九月的各节用了 64 个输出 token 和更旧的引擎构建。10 月 1 日的对比用两块卡、
256 个输出 token。这些表格作为参考保留。因为协议不同，它们不是与 10 月 2 日单卡运行的受控
对比。

<a id="older-measurements"></a>

Prefill 快路径（2026-09-28）其他都不变，但把 BF16 投影 GEMM 换成 FP16 tensor-core 乘积（一个
BF16 值在 FP16 里是精确的——标量 16x16 回退没有了），把 PLE 行读取换成 io_uring O_DIRECT
（此前一次一个同步 pread），并把模型文件移到 256 GB NVMe。2026-09-28 测量的全新未缓存请求
（引擎日志，`reused=0`）：

<a id="the-prefill-fast-path-2026-09-28"></a>

| Prompt size | Exact tokens | Prompt time | Prompt speed | Output speed | Expert-cache hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ~4K | 4,143 | 4.19 s | **989.1 tok/s** | 49.3 tok/s | 84.4% |
| ~4K | 4,376 | 4.34 s | 1007.5 tok/s | 52.1 tok/s | 86.5% |
| ~8K | 8,785 | 9.09 s | **966.5 tok/s** | 50.6 tok/s | 88.0% |
| 32K | 29,527 | 25.97 s | **1136.8 tok/s** | 50.5 tok/s | 88.7% |
| 128K | 117,837 | 178.72 s | 659.3 tok/s | 44.9 tok/s | 88.4% |
| 256K | 256,076 | 618.39 s | 414.1 tok/s | 38.8 tok/s | 84.3% |

8K 那一行从一块凉的卡重测三次（967.0 / 965.0 / 966.5 tok/s）。V100-PCIE-16GB 是无风扇散热：
长行之后它空闲在约 79 C，负载下在约 83 C 从 1380 MHz 加速频率降到约 960-1280 MHz，同一
prompt 最多掉约 25%（8K prompt 跨 616-967 tok/s，取决于何时运行）。上表是凉机空闲时的测量。
原始行在 [`bench/results/2026-09-28-v100-fastpath`](../../bench/results/2026-09-28-v100-fastpath)，
整张表可由 [`bench/run_v100_bench.py`](../../bench/run_v100_bench.py) 复现。

<a id="the-8k-row-was-re-measured"></a>

一次全新请求的 PLE 行 gather 从约 3.6-8 秒降到约 7 毫秒，超连接 GEMM（最大的 BF16 投影块）
在 8K 时从约 5.2 秒降到约 0.5 秒。Decode 不变。

<a id="the-ple-row-gather-for-a-fresh-request"></a>

显式的层级切分为每块卡只加载它自己那些层的规范权重和原生 dense 投影。自动放置保留完整的表。
配置的上下文容量与视觉支持不变。

<a id="an-explicit-layer-split-loads"></a>

**需显式开启的 decode 内核变体（`STRATA_EXPERT_PAIR=1`，默认关）：** 带原生 expert 包时，
分组 expert 内核每 warp 计算两行而不是一行。GPU 等期性 fixture 在两块 V100 上都产出逐字节
相同的结果。在已安装配置的环境里设这个变量来启用该变体。服务文件不需要改动。

<a id="opt-in-decode-kernel-variant"></a>

Prompt 路径的 FP16 expert 默认用每 expert 的 scratch（每个 expert 最多一个分块的行）。Decode
路径每层把 shared expert 的激活量化一次。CPU 激活量化只为有 CPU expert 任务的 token 运行。
没有 CPU expert 任务的窗口不等 CPU worker 屏障。两块卡的缓存回补在任一块等待之前就开始。

<a id="the-prompt-paths-fp16-experts"></a>

**V100 后续测量（2026 年 10 月 1 日；更早，双 GPU A/B）**

<a id="v100-follow-up-measurement"></a>

这次运行用两块 V100 PCIe 16GB。GPU0 是 Gen3 x2 链路。GPU1 是 Gen3 x16 链路。每个尺寸三次
全新请求、256 个输出 token。表格给 token/秒的速度中位数。

<a id="this-run-used-two-v100-pcie-16gb-cards"></a>

| Target prompt | Original prefill | Installed prefill | Change | Original decode | Installed decode | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 442.3 | 525.0 | +18.7% | 66.9 | 69.9 | +4.5% |
| 8K | 620.8 | 705.9 | +13.7% | 66.8 | 64.5 | -3.4% |
| 16K | 820.2 | 955.6 | +16.5% | 64.0 | 66.5 | +3.9% |
| 32K | 976.3 | 1105.9 | +13.3% | 66.2 | 70.0 | +5.7% |

四个尺寸上都没有达到 prefill 与 decode 提升 20% 的目标。Decode 结果好坏参半。不要把内核
fixture 的速度当作运行时速度声明。记录的样本在 GPU0 上达到 84 C、GPU1 上 83 C。这次运行中
两块卡都报告了软件热降频。

<a id="the-20-percent-prefill-and-decode-target"></a>

安装的设置保持 20/28 层级切分、262144-token 上下文、int8 KV、自动 prefill 分块、spec 8、
草稿置信 0.70 和视觉。配对 expert 变体已启用。在 GPU0 上，拥有层加载把原生 dense 权重从
1376.20 MiB 降到 541.27 MiB。Expert 缓存从 6750 增加到 7918 个槽。

<a id="the-installed-settings-keep"></a>

原始测量：[原始](../../bench/results/2026-10-01-v100-pass2/baseline/matrix.json)、
[已安装](../../bench/results/2026-10-01-v100-pass2/final-installed/matrix.json)、
[汇总](../../bench/results/2026-10-01-v100-pass2/final-summary.json)，以及
[温度样本](../../bench/results/2026-10-01-v100-pass2/final-installed-gpu.csv)。
用 `bench/run_v100_bench.py`，目标 `4096,8192,16384,32768`、`--max-tokens 256`、
`--seed 20261002`、`--repeats 3` 重复这一扫描。

<a id="raw-measurements-original"></a>

**带 FP16 tensor-core expert 的 V100 int8-KV prefill（2026 年 10 月 3 日；双 GPU A/B）**

<a id="v100-int8-kv-prefill-with-fp16"></a>

这次运行用与上面 10 月 1 日 A/B 相同的两块 V100 PCIe 16GB 和相同设置：层级切分 20，int8 KV，
524,288-token 上下文，YaRN scale 2，窗口 8 带 0.70 草稿下限的推测，自动 prefill 与 expert
缓存，700 MiB 视觉预留。Prompt 检查点和对话缓存关闭，所以每个请求读它完整的 prompt
（`reused=0`、`resume=0`）。每个尺寸三次带精确 token id 的全新请求（种子 20261003），每个
请求生成一个 token。表格给 prompt 时间与速度的中位数。

<a id="this-run-used-the-same-two-v100"></a>

| Target prompt | Baseline prompt time | Candidate prompt time | Less time | Baseline prefill | Candidate prefill | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 3,663.4 ms | 2,910.7 ms | 20.6% | 559.0 tok/s | 703.6 tok/s | +25.9% |
| 4K | 5,039.0 ms | 3,794.8 ms | 24.7% | 812.9 tok/s | 1,079.4 tok/s | +32.8% |
| 8K | 7,810.4 ms | 5,767.0 ms | 26.2% | 1,048.9 tok/s | 1,420.5 tok/s | +35.4% |
| 16K | 11,052.3 ms | 8,517.8 ms | 22.9% | 1,482.4 tok/s | 1,923.5 tok/s | +29.8% |
| 32K | 18,078.8 ms | 14,506.0 ms | 19.8% | 1,812.5 tok/s | 2,258.9 tok/s | +24.6% |

全部五个尺寸每秒 token 数都至少多 20%（24.6% 到 35.4%）。"Change" 列是速度变化；"Less time"
是 prompt 时间的降幅。Decode 未测，因为每个请求只生成一个 token。

<a id="all-five-sizes-gain-at-least"></a>

2K 时批处理的 prompt 分块是 2,047 个 token，低于紧凑遍历的 2,048-token 流式下限。所以那一行
保持 int8 小分块路径；它的增益来自新的暂存、缓冲区定大小与回补策略。

<a id="at-2k-the-batched-prompt-chunk"></a>

候选方案把原生 Q2_0 包的被路由 prompt expert 移到 FP16 tensor-core 乘积加 FP32 累加。gate/up
和 hidden 缓冲区于是装紧凑的行容量，而不是每个被路由 token 一行；最后的小分块保持 int8 路径。
在这条路上紧凑遍历还选一个 192 槽的流式环（`STRATA_PREFILL_RING` 覆盖它）。内核按算力 7.0
（Volta）选择，不按产品名。`STRATA_PREFILL_FP16TC=0` 在所有设备上保持 int8 参考。候选方案
还在非最后分页等待最后分页之前，就启动它借给 prompt 路径的 expert 缓存槽的回补
（`STRATA_PREFILL_REFILL_OVERLAP=0` 禁用；串行和阻塞回补模式绕过它）。

<a id="the-candidate-moves-the-routed"></a>

每次受测请求之前，两块卡都不高于 55 °C 并保持 10 秒。每个请求都通过闸门。基线中最高采样温度
是 51 °C（GPU 0）和 57 °C（GPU 1），候选方案中是 50 °C 和 57 °C。两块卡都没有报告热或硬件
降频。

<a id="before-every-measured-request-both"></a>

新路径不是逐位精确的。等期性测试 `prefill_fp16tc_test` 在两块卡上都通过：对照双精度参考，新
路径测得 1.073e-02 的相对 RMS，int8 MMQ 路径 1.074e-02。15 个配对请求中 14 个的首个生成
token 相同。一次性的语义冒烟在两臂都通过全部八项检查（10%、50%、90% 深度在 2K 和 32K 上的
检索，以及每个长度一道算术题），两臂产出相同的输出 token。这是冒烟检查，不是质量测量。不要
用这些结果宣称困惑度或长生成长度质量不变。

<a id="the-new-path-is-not-bit-exact"></a>

只测了 V100 卡。

<a id="only-v100-cards-were-tested"></a>

原始测量：[结果目录](../../bench/results/2026-10-03-v100-prefill/)。用
[`bench/run_v100_prefill.py`](../../bench/run_v100_prefill.py) 重复这一扫描，目标
`2048,4096,8192,16384,32768`、`--repeats 3`、`--seed 20261003`、`--max-new 1`，以及每个请求
之前同样的冷却闸门。

<a id="raw-measurements-the-result-directory"></a>

同一测试框架用 `--chain` 测量一连串 agent 轮次。它读取一个前缀 prompt。然后为每个
`--chain-diffs` 值（默认 `0,1,32,64,128,256,512,1024,2048`）发一轮。每一轮加上那个确切的
token 数。于是每个 prompt 都是下一个的前缀。引擎的 prompt 缓存保持开启，
`--chain-cache-every` 设它的检查点间隔（默认：前缀的四分之一）。配置点名的对话磁盘存储会被
放到 `--out` 之下的一个私有目录。测试框架丢弃 `--expert-profile-save`。第一个 diff 是 0。
那一轮把同一个 prompt 再发一次。每行记录请求的 diff、引擎复用的 token（`reused`）和它读取的
token（`prompt_read`）。引擎最多在距 prompt 一个 token 处恢复，所以两个计数不同。不要把请求
的 diff 报告成读取计数。用 `--xargs`、`--env`、`--short-read N` 或 `--pipeline-windows N` 给
某一臂加上它自己的旋钮。每个请求默认发 `pin=n-2`：引擎忽略 `pin >= n-1`，所以下一轮重读最后
两个 token，而不是其最新周期检查点之上的 token。`--no-chain-pin-tail` 关掉这个钉并展示那个
缺口。示例：`python bench/run_v100_prefill.py --chain --arm-label chain-baseline --exe
/tmp/engine --out /tmp/chain`。用 `--compare` 比较两次链式运行。

<a id="the-same-harness-measures"></a>

供对比的长上下文基线（引擎 0.1.20，模型在便携 SSD 上，快路径之前）：

<a id="long-context-baseline-for-comparison"></a>

| Prompt size | Exact tokens | Prompt time | Prompt speed | Output speed | Expert-cache hit rate | Total request |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1,031 | 3.54 s | 291.0 tok/s | 33.6 tok/s | 81.7% | 5.4 s |
| 8K | 7,431 | 16.25 s | 457.4 tok/s | 41.2 tok/s | 90.1% | 17.8 s |
| 32K | 29,512 | 63.52 s | **464.6 tok/s** | 42.5 tok/s | 92.8% | 65.0 s |
| 128K | 117,833 | 363.06 s | 324.6 tok/s | **44.0 tok/s** | 94.4% | 364.5 s |
| Full context | 256,073 | 987.69 s | 259.3 tok/s | 40.3 tok/s | 96.2% | 989.4 s |

独立的硬件校准器在调优时生成 256 个 token，在选定的 0.28 PCIe 份额和 0.70 草稿下限下测得
**45.5 tok/s**。上面的表格刻意包含 prompt 处理并使用未缓存前缀（每行都是 `reused=0`）；256K
那一行距配置的 262,144-token 上限只差 6,068 个 token。

<a id="the-separate-hardware-calibrator"></a>

RTX 5070 **12 GB**、Ryzen 5 7600（六核）、64 GB DDR5-5200、Windows、引擎 0.1.26 配 setup
写的设置（`--prefill auto`、4K 以上 8 位 KV、64K 起 KV 流式）。每个长度一个代码 agent
prompt，256 个生成 token，MTP 推测解码开。"262K" 是模型的完整上下文窗口（一个 259,943-token
prompt）。IQ2_XS 那一行用 Swift 1.5 的 IQ2_XS 测，它以原版的速度运行。

<a id="rtx-5070-12-gb"></a>

**引擎 0.1.36（#136），同一台 PC：** Q2_0 的 prompt expert 跑在融合 int8 tensor-core 内核上
（RTX 30 及更新）：4K 1,294 -> 1,570、32K 2,170 -> 2,653、128K 2,123 -> 2,468 tokens/s
（+16-22%），与之前的内核同样接近 FP16 参考（32K 上更近：强制教师 KL 0.009 对 0.012）。Decode
路径的块选择与贪心 argmax 跑在线程块 cluster 上（RTX 50、sm_90+；其他卡保持之前的内核；同样的
token）：Q2_0 输出在 4K 89 -> 93.5、在 128K 64.5 -> 76.4 tokens/s。`STRATA_PF_FUSED=0` 保持
之前的 prompt 内核（与 0.1.35 逐字节一致的回答）；`STRATA_PF_FUSED=1` 也跑原生 IQ 包的融合
内核（需显式开启：IQ2_XS prompt 在 4K +12%、32K +3%，IQ3 各包基本持平；HIP 上它们在 gfx11 和
gfx1200 / gfx1201 上运行，见 AMD_HIP.md）；`STRATA_QSA_CLUSTER=0` / `STRATA_ARGMAX_MULTI=0`
关掉那些 decode 内核。下面的表格是 0.1.26 的。

<a id="engine-0136-136-the-same-pc"></a>

### Prompt 处理（tokens/s）

<a id="prompt-processing-tokenss"></a>

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 536 | 1,299 | 2,171 | 2,126 | 2,107 | 1,304† |
| **IQ2_XS** | 534 | 1,256 | 2,092 | 1,754 | 1,752 | 1,181*† |
| **IQ3_XXS** | 482 | 1,007 | 1,745 | 1,609 | 1,602 | - |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 | - |
| **Coder** | 656 | 1,583 | 2,177 | 2,236 | 2,208 | 1,034** |
| **IQ3_S**（AMD RX 7900 XTX，gfx1100） | 760 | 1,275 | 1,641 | 1,594 | 1,494 | - |

引擎 0.1.26；`bench/results/2026-09-29-speed-0126`。AMD RX 7900 XTX 那一行：引擎 0.1.38 跑在
ROCm 10.2 nightly、配调好的 gfx1100 hipBLASLt-100500 表（作为
[#755](https://github.com/Niko1221/Strata/pull/755) 提交 upstream），每档 3 个干净单元格的
中位数（Ryzen 9 7900X，96 GB；1K-128K 一次性运行，256 个生成 token，贪心）。32K-128K 上比
0.1.22 快 8-28%。† 没有再测：0.1.22。\* 带图像开启测（图像编码器的 VRAM 预留让缓存的 expert
更少）。\*\* 没有再测：0.1.14。

<a id="engine-0126-benchresults2026-09-29-speed-0126"></a>

### 输出（tokens/s）

<a id="output-tokenss"></a>

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 87.3 | 93.0 | 81.8 | 76.2 | 73.7 | 60.3† |
| **IQ2_XS** | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 | 52.8† |
| **IQ3_XXS** | 61.9 | 61.6 | 58.5 | 57.2 | 49.0 | - |
| **IQ3_S** | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 | - |
| **Coder** | 58.9 | 55.1 | 54.9 | 53.2 | 43.0 | 42.8† |
| **IQ3_S**（AMD RX 7900 XTX，gfx1100） | 65.1 | 60.4 | 64.4 | 61.8 | 59.2 | - |

引擎 0.1.26，同一批运行。AMD RX 7900 XTX 那一行：引擎 0.1.38 跑在 ROCm 10.2 nightly（调好的
gfx1100 hipBLASLt-100500 表，作为 [#755](https://github.com/Niko1221/Strata/pull/755) 提交
upstream），每档 3 个干净单元格的中位数（同一台机器；decode 对上下文不敏感——GDN 线性
attention 每 token O(1)；1K/4K prefill 略低于 0.1.31 打包版 ROCm 的数字，32K-128K 领先）。
† 没有再测：0.1.14。

<a id="engine-0126-the-same-runs"></a>

输出速度也取决于文本：推测解码在被接受的草稿 token 更多时跑得更快，所以同一 prompt 的不同
回答会让它变动几个百分点。在 4K prompt 上背靠背运行，0.1.14 写 88.5 tokens/s、0.1.12 写
85.7。0.1.13 之前的数字（prompt 约慢一半）：
[`bench/results/2026-09-24-final`](../../bench/results/2026-09-24-final/matrix.md)；这些：
[`bench/results/2026-09-28-speed-0114`](../../bench/results/2026-09-28-speed-0114/README.md)。

<a id="output-speed-depends-on-the-text"></a>

IQ3_XXS 和 IQ3_S 在 262K 上没有测：它们 43 / 50 GB 的 expert 让 260K-token 上下文按 setup 的
估计把 64 GB 的 PC 逼到内存极限（expert + 上下文的 KV 缓存 + 24 GB），所以 setup 在 64 GB 上
配它们最多推荐 128K。你选的更长上下文（`--context 262144`，或其列表中的选择）会被保留，附带
一条说明：有用户在 64 GB 上把 IQ3_S 跑到 256K 且 RAM 有富余（#406）。低 RAM 模式下 KV 缓存
留在 VRAM、上下文不计入 RAM。IQ3_S（引擎 0.1.4 或更新）为原模型和 Swift 1.5 发布（与 Swift
其他尺寸相同的 shard 布局）。

<a id="iq3_xxs-and-iq3_s-at-262k"></a>

**KV 流式（引擎 0.1.5）：** 64K 及以上，setup 把上下文的 KV 缓存放在 RAM、只在 VRAM 放
attention 读到的那部分（`--kv-resident 32768`），于是 GPU 上装得下更多 expert。Q2_0 在 262K：
50.9 -> 62.6 tokens/s（VRAM 内 1,589 -> 3,872 个 expert）；128K 上约 +6%。Attention 读的正是
同样的值（只是 KV 住的地方变了）；代价是每上下文 token 约 13.7 KB 的 RAM（128K 时 1.7 GB）。
已有的安装：跑一次 `START-HERE.bat --setup` 打开它。RAM 有余量时 setup 会打开它；
`--kv-streaming on|off` 覆盖该判断（越过 RAM 测试开启时附说明；WSL 下绝不开启，它无法流式）。

<a id="kv-streaming-engine-015"></a>

WSL2 下驱动把 `cudaHostAlloc` 能锁页的量限制在约 1 GiB，所以那里 KV 的 host 副本改用
`cudaHostRegister` 锁普通内存（通过 `/dev/dxg` 检测）；`STRATA_KV_HOST_REGISTER=1` 在任何
系统上强制那个顺序，`=0` 强制先 `cudaHostAlloc`（两种情况下另一种方法都是回退）。非 WSL 的
Linux 上默认什么都不变；带该开关时引擎打印 `KV streaming: the host copy is pinned with
cudaHostRegister`（#1498，作者 yjamil8；WSL 这里没测，强制路径在 Tesla P100 上跑过：
`--kv int8 --kv-resident 32768`，回答与 decode 速度不变）。

<a id="under-wsl2-the-driver-caps"></a>

**4 位 KV 缓存（引擎 0.1.8，可选）：** `START-HERE.bat --setup` 在 8K 以上上下文时会问（或传
`--kv q4_0`）。它在 4 位舍入前用 Hadamard 旋转把 KV 缓存内存减半（PR #21），128K 上约快 4%，
但长文档上可测地不够精确（困惑度 +8-12%；大海捞针测试仍通过）。8 位保持默认。细节：
[`bench/results/2026-09-27-kv-q4`](../../bench/results/2026-09-27-kv-q4/README.md)。

<a id="4-bit-kv-cache-engine-018-optional"></a>

**混合 K8V4 KV 缓存（引擎 0.1.25，可选，PR #120）：** `--kv k8v4`（`START-HERE.bat --setup
--kv k8v4`）把键保持 8 位、值存为旋转后的 4 位：比 8 位少 23% 的 KV 内存，于是 VRAM 里装得下
更多 expert。RTX 3090、Coder 在 198K 上下文：输出 99 对 85 tokens/s，相同的大海捞针结果，
prompt 慢 2-5%。它像其他格式一样流式它的 KV 缓存（`--kv-resident N`）：RTX 2060 SUPER 8 GB
在 128K、20,480 个常驻 cell 时，比常驻 K8V4 多 780 个 expert 槽，且长文档上得分优于 4 位 KV。

<a id="hybrid-k8v4-kv-cache-engine-0125"></a>

**可复现的贪心输出（0.1.30，需显式开启，`STRATA_IQ_MT_MIN=1`）：** 带 IQ 模型时，CPU 对单个
token 用 ggml 的点积计算 expert、对多个 token 用 Strata 的多 token 内核，后者舍入略有不同。多
少 token 共享一个 expert 取决于 verify 窗口里的草稿，所以同一 prompt 在温度 0 下，当草稿、
缓存状态或恢复的对话不同时会以不同的（同样好的）回答结束（issue #152）。
`STRATA_IQ_MT_MIN=1`（在配置的 `env` 里）对每个分组都用多 token 内核：回答于是不再取决于草稿。
在 Ryzen 7600（AVX-512）上实测：IQ3_S decode -1..-3%，其他模型不变；默认保持最快的规则。在
有 AVX2、无 AVX-512、无 AVX-VNNI 的 CPU 上，默认改为用 AVX-2 内核为 IQ3_XXS、IQ3_S 和 IQ2_S
服务单 token（NT1）的 gate/up 行；这些行与 ggml 的点积逐位相同，所以那里单个 token 的行不再
取决于草稿。设了 `STRATA_IQ_MT_MIN` 仍覆盖此默认。在没有 AVX-512 的 Alder Lake 或更新的 Intel
CPU 上，AVX-2 内核要为 IQ3_S  gather 网格时，`STRATA_IQ3S_MT1=1`（需显式开启）让 IQ3_S 对单个
token 用多 token 内核，那里它更快；它改变单个 token 的舍入，所以默认关闭。经服务器，还有两件
事从一个请求带到下一个（#410）：自适应层在 RAM 与 VRAM 之间移动 expert（GPU 和 CPU 对一个
expert 舍入不同），以及 prompt 缓存恢复重复的 prompt、只把它的尾部走 decode 路径读。要逐字节
一致的重放，还要在引擎参数里加 `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`（#410）：缺失
expert 的 PCIe 份额（改在 GPU 而非 CPU 上算）仍会让启动后的第一个回答与之后的不同。这里实测
（IQ3_XXS，3.6K-token prompt，4 次重复）：三个开关全带时 4 次里 1 个回答，不带
`--pcie-frac 0` 时 2/4（第一个不同），默认时 2/4。`--pcie-frac 0` 牺牲 decode 速度（缺失
expert 全在 CPU 上跑），所以只把它留给 A/B 运行。

<a id="reproducible-greedy-output-0130"></a>

**带 Gumbel-max 选择的耦合草稿（需显式开启，`STRATA_SPEC_COUPLED=1` 和
`STRATA_SPEC_GUMBEL=1`）：** 对采样的请求（温度大于 0），`STRATA_SPEC_COUPLED=1` 让草稿层用目标
自己的链和随机抽取来采样它的猜测，而不是取它最可能的 token。`STRATA_SPEC_GUMBEL=1` 改变两者从
那条链选取的方式：取 p / E 最大的 token，其中 E 是以种子、位置和 token id 为键的指数噪声
（Gumbel-max 技巧；它仍是同一分布的精确采样）。默认选取是让一个随机数走过按概率排序的候选，
候选列表只差一个 token 的草稿和目标往往落在不同的 token 上；而按 token 定键的噪声让同一个 token
在两边拿到同样的噪声，于是它们在共享的 token 上一致。贪心请求不变，不设这些变量时什么都不变。
在 Ryzen AI Max+ 395（Radeon 8060S，Linux，ROCm 7.14.1，仅核显）上实测，UD-Q4_K_XL，
`--spec 4 --mtp`（基础模型的草稿层），温度 1.0 / top_p 0.95 / top_k 20，1.3K-token prompt 和
512 个输出 token，每臂 12-13 个请求：草稿接受率 52.8% -> 59.9%，每个 verify 窗口的 token 数
2.65 -> 2.88，输出 41.1 -> 44.9 tokens/s（+9%）；一个窗口代价相同（64 毫秒中草稿 8.8 -> 9.1
毫秒）。`sampler_parity` 在每个采样路径上对照 host 参考检查选取，并检查其频率与 softmax 相符。
在 RTX 3060 上（IQ3_XXS，`--spec 4 --mtp`，温度 1.0 / top_p 0.95 / top_k 20，200-token 故事与
代码请求的 12 个交错对）`STRATA_SPEC_COUPLED=1` 把接受的草稿从 62.1% 提到 66.9%，再加
`STRATA_SPEC_GUMBEL=1` 到 68.0%，但 decode 速度没有超出运行间散布而变动（仅 coupled 中位数
43.3 tok/s，加 Gumbel 42.8）：先在你自己的卡上试再依赖它。

<a id="coupled-drafts-with-gumbel-max-picks"></a>

**草稿层的 token（0.1.27，`--draft-vocab`）：** MTP 草稿层只能提议词表的一个子集里的 token
（`mtp/rt/draft_vocab.bin`）。自 0.1.27 起子集包含每个中文、日文、韩文 token（106,299 个
id），所以那些语言的回答快 15-38%（Q2_0，RTX 5070）。它的头占约 180 MiB VRAM，expert 缓存为它
空出那块（0.1.28）。`START-HERE.bat --setup --draft-vocab en` 保持之前的英文/代码子集（40,525
个 id，少约 110 MiB VRAM，英文回答快 1-2%；CJK 回答几乎拿不到草稿）。`--draft-vocab cyrillic`
取英文/代码子集加整个西里尔文字（58,963 个 id）：随带的子集只装着词表 18,580 个西里尔 token
中的 142 个，所以乌克兰语或俄语回答每轮 1.4 个 token；带上它 2.1 个，83 -> 109 tokens/s（RTX
5090，NVFP4 fork），英文不变。`--draft-vocab fr`（0.1.39，#597）取英文/代码子集加覆盖一个法语
维基百科语料 99% 的 5,686 个 token（46,211 个 id，`tools/draft_vocab.py --corpus`）：法语文本
23.6% 的 token 在英文/代码子集之外，0.7% 在这个之外。法语回答的草稿接受率 0.51 -> 0.60
（IQ3_XXS，RTX 5070，8 个 prompt × 2 遍；英文 0.61 -> 0.63、代码 0.77 -> 0.78，没有损失），
RTX 5090 上法语 141 -> 158 tok/s（IQ3_S，报告者的测量）。`tools/draft_vocab.py` 构建并检查
子集。当启动以 "the draft head does not fit"（一块 12 GB 卡配长上下文，#474）停止时，引擎说明
头需要多少、还剩多少 VRAM、哪个更小的子集装得下，服务器的启动错误会复述它；setup 对小于 14 GB
的卡建议 `--draft-vocab en`（只是建议：不传就什么都不变）。

<a id="the-draft-layers-tokens-0127"></a>

**不带草稿层的服务（`--mtp` 可选）：** `serve` 可以不带 `--mtp` 运行。草稿于是只来自
后缀/prompt-lookup 草稿器（或每轮一个 token），每个 token 仍然对照模型校验，而草稿层的 VRAM
（连同它的头约 0.7-1 GiB）给 expert 缓存：一块 8 GB 卡上的一次 A/B 里 1,000 -> 1,678 个槽。
对话缓存（`--conversation-cache-mib`）保持开启：被暂存的对话那时不带草稿 K/V。

<a id="serving-without-the-draft-layer"></a>

**低 RAM 模式（引擎 0.1.26，setup 选择）：** 通常模型的全部 expert 都被拷进 RAM（23-50 GB，
锁页），GPU 保留最常用的那份的副本。在 RAM 装不下它们加系统的 PC 上（expert 加约 10 GB），
setup 改为从模型文件夹里的一个文件映射它们（`--mmap-experts`，包的 `experts.bin`，磁盘
+23-50 GB）。操作系统文件缓存装着 GPU 没装的那些，而且它能交还那块内存。在 Coder 上引擎的已
提交内存从 36 降到约 13 GB，回答相同。配一块大 GPU（RTX 5090 装得下 Coder 的全部 expert、
Q2_0 的大部分）它以接近平常的速度运行。配小的，多数 expert 来自 SSD、慢得多（setup 会说明）。
`START-HERE.bat --setup --low-ram on|off` 覆盖该选择。

<a id="low-ram-mode-engine-0126"></a>

**低 RAM 模式，常驻版（引擎 0.1.30）：** 当 GPU 没装的 expert 装得进 RAM（旁边同样留约 10 GB）
时，setup 改选常驻变体（`--resident-experts`）：启动时引擎把正是那些 expert 从 `experts.bin`
拷进 RAM（驱动允许时锁页，否则锁在 RAM 里），所以回答期间不从 SSD 读任何东西，无论操作系统
给文件缓存留多么少的 RAM。用 setup 的上下文的例子：32 GB 的 PC 配 24 GB 的 GPU 就这样跑 Q2_0、
IQ2_XS 和 Coder（RAM 里约 16-18 GB 的 expert，GPU 装另外的约 18 GB），32 GB 的 PC 配 12-16 GB
的 GPU 跑 Coder；32 GB PC 上的 IQ3_XXS 保持映射。细节：

<a id="low-ram-mode-resident-engine-0130"></a>

- Prompt 路径把 GPU expert 缓存里的空间借给自己的缓冲区，prompt 结束后把那些 expert 放回去；
  在 RAM 允许的范围内，它们的 expert 也保留在 RAM（所以 prompt 也不从 SSD 读任何东西）。
- 缓存仍跟随对话（`--adapt-every`）：一次交换把被逐出的 expert 从 VRAM 拷回、放进取代它的
  expert 的 RAM 位置，所以 RAM 副本恰好一直装着 GPU 没装的那些。
- `STRATA_RAM_ADAPT=<n>`（需显式开启，`--serve`，一块 GPU）：RAM 副本也跟随对话。不带它时副本
  在启动时选定一次，所以当 RAM 装不下 GPU 没装的每个 expert 时，对话每次路由到副本之外的
  expert 都要从 SSD 读它。带上它时，在 decode 窗口之前、且没有自适应轮在途时，最多 `n` 个窗口
  持续路由到的 expert（自适应层的衰减计数，2 或以上）取代它们所在层没有被路由到的 expert 的 RAM
  位置（计数低于 0.5）：每层一对，差值大的先来，每次一个 blob 读取。RAM 大小不变；路由、权重和
  VRAM 层的规则都不动，离开副本的 expert 再次需要时从文件重新读。它需要自适应层
  （`--adapt-every`），在 `STRATA_EXCHANGE_ROTATE=1`、层级切分、peer 或辅助 GPU、`--batch` 下
  不起作用。Prompt 的路由不用于选择替换，prefill 期间不运行选择。之后的 prompt 可能从被先前
  decode 请求改过的 RAM 副本读取。日志报告移动了多少 expert。在 RTX 3060 12 GB、32 GB RAM 上
  实测（Linux，IQ2_XS，RAM 里 20.5 GiB expert、文件层上约 6,000 个 expert，`--adapt-async 1`、
  `--spec 3`、节能模式），对照 v0.1.43：47 个我日常使用的冻结韩语请求（工具、追问和后台任务，
  281-33,241 个 prompt token），按序在新引擎里重放：release、本选项、release。每次从同一份保存
  的学习 profile 起步；私有 prompt 不随附。带默认尾部跳过（7），对照收尾的 release decode 是
  43.26 -> 50.40 tok/s（+16.5%），按总生成 token / 总 decode 时间计算；两个 release 臂相差
  0.2% 以内。短/长新输入请求分别增益 16.3%/16.9%；总 prompt 速度相差 2.5%，在运行散布内。每
  生成 token 的 decode 文件层按需读取从 14.15 -> 2.67；8,056 次成功的 RAM 提升各多读一个 blob，
  加上后是 3.16。这些是文件层计数，不是物理 SSD I/O。RAM 常驻不能让 expert 免于尾部跳过。一台
  机器、一种量化；尾部跳过关闭、阻塞式自适应、`--spec 2` 和耐久测试没有在 v0.1.43 上重复。
- `--adapt-async 1`（需显式开启，`--serve`）：一轮的交换在 decode 窗口之间由辅助线程推进（拷回、
  拷入、移进 RAM），而不是一个窗口等整轮。配 `--batch` 槽时也在它们的窗口之间推进
  （docs/BATCHING.md）；配 `--batch-groups` 或 `--peer-device` 不行（那里跑阻塞层）。配
  `--pipeline-windows 2`（两个窗口在途，见 [MULTI_GPU.md](MULTI_GPU.md)）时，拷进被逐出槽和
  移进 RAM 也等它们被决定时在途的窗口完成（`STRATA_PIPELINE_ADAPT_ASYNC=0`：那里退回阻塞层）。
  它跨运行不是逐位一致的：哪个窗口最先在 GPU 上算一个换入的 expert（与 CPU 舍入不同）取决于它的
  拷贝何时落地。交换缓冲区没有锁页时它保持在阻塞层（日志说明）；统计行报告每轮毫秒。
- **区也一样**（一块 GPU，无层级切分）：所有 expert 都在 RAM 时 `--adapt-async 1` 跑只拷入的
  同层：没有要拷回或移进 RAM 的东西。被逐出的槽在该轮的拷贝发出之前被标记为非常驻，expert 只在
  它的拷贝落地后才变常驻；`STRATA_ADAPT_ASYNC_WAIT=1`（那里的默认）在下一个 tick 等这一轮，所以
  expert 变 GPU 常驻的窗口不依赖时机。选择在 host 线程上跑（`STRATA_ADAPT_PICK_INLINE=1`，每
  窗口一轮）；`STRATA_ADAPT_ASYNC_BATCH=N` 每个 tick 最多发出 N 次拷贝（默认 0 = 全部；16 实测
  更慢）。每次 K/V 增长和 `VRAM` 命令都先让在途的轮落地。RTX 3090 + Ryzen 5 5600X，冷 prompt：
  decode ×1.07-1.11，12 个 prompt 里 12 个更快
  （[报告](../../bench/results/2026-10-11-community-rtx3090-5600x-arena-async/README.md)）。
- **按需锁页**（`STRATA_PIN_BY_NEED=1`，需显式开启，一块 GPU 上的区）：Windows 最多锁页约一半
  RAM（WDDM 共享预算），所以区注册它的前约 30 GiB、其后的层拿不到 PCIe 份额。带该变量时每个
  blob 获得页对齐的跨步（RAM +0.15%），expert profile 的非 GPU 常驻 expert 改为按排名注册，最多
  `STRATA_ARENA_PIN_GIB`（Windows 上默认 29）。`STRATA_PCIE_MIN1=1` 让一层只要有两次缺失就送
  一次走 PCIe（份额在 1-2 次缺失时否则舍入为 0）。RTX 3090 + Ryzen 5 5600X：56K-117K token 时
  prompt 读取 +5-7%；decode 单独持平，配 `--adapt-async 1` 和 `--pcie-frac 0.6` +2-3%
  （[报告](../../bench/results/2026-10-11-community-rtx3090-5600x-arena-pin-by-need/README.md)）。
- **锁页内存余量**（Windows CUDA，默认开；`STRATA_PIN_RESERVE_MIB=N`，`0` = 关，在任何 OS 上
  设置）：区一块一块地注册直到驱动拒绝某一块，而在 64 GB 的 PC 上那不给引擎之后分配的几块小
  缓冲区（verify 窗口的暂存：Swift 1.5 IQ2_XS 上 `verify: mapped staging allocation failed`、
  退出码 1）留下任何锁页内存。引擎现在在区锁页期间扣留 640 MiB（区放好就交还）和 160 MiB（保留
  到某个必需缓冲区被拒绝或 READY），所以区大约早一层停下；被拒绝的必需缓冲区把两块都交还并再试
  一次。若仍失败，错误以 `[pinned-memory-exhausted]` 结尾并点名应从哪个 `STRATA_ARENA_PIN_GIB`
  起步。Linux 没有这种上限（NVIDIA 驱动越过 `ulimit -l` 也能锁页），那里的预留是关的。测试旋钮：
  `STRATA_TEST_PIN_BUDGET_MIB=N` 让引擎自己的锁页分配超过 N MiB 就失败，
  `STRATA_TEST_PIN_LEFT_KIB=N` 在区之后把该上限降到所计额度加 N KiB。
- `STRATA_EXCHANGE_ROTATE=1`（需显式开启）：自适应交换交出缓冲区所有权，而不是把被逐出的 blob
  拷进 RAM 副本（完全锁页的副本；blob 大小不同的层按大小保留备用缓冲区）。token 相同、host 拷贝
  更少；它也能与 `--adapt-async 1` 同用。RTX 5090、64 GB RAM（Windows，每个 verify 窗口约 30 次
  交换）上的 Swift 1.5 IQ3_S：一段强制的 1,500-token 文本 164 -> 188 tok/s，23 轮编码 agent
  对话 155 -> 178。细节与测量：[EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md)。我们在 RTX 3060
  12 GB 上实测（IQ3_XXS，常驻 RAM，默认参数，5 个交错对）：decode 58.96 -> 62.03 tok/s（+5.2%，
  5/5 对），prompt 速度不变；带真实自适应交换时 token 相同（Tesla P100，常驻，阻塞式自适应）。
  它增加约 1.4 GiB 锁页 RAM，RAM 本来就紧时请保持关闭。
- 在相同的 expert 放置下回答就是普通映射模式的：字节来自文件。带锁页副本时 GPU 也经 PCIe 拿走
  缺失中它 usual 的份额（`--pcie-frac`），与 RAM 充足时一样；`--pcie-frac 0`（或
  `STRATA_RESIDENT_PIN=0`）给出映射模式的精确 token。
- 引擎把它发现空闲的 RAM 留下 4 GB（`STRATA_RESIDENT_HEADROOM_GIB`）；当连 GPU 没装的 expert
  都装不下时它说明原因并跑普通映射模式。服务器日志按请求显示多少次 expert 读取去了文件
  （`resident RAM: ... blob reads from the file`：稳定使用中为 0）。
- `--low-ram resident|mmap` 强制一种变体（也在 RAM 足够的 PC 上，例如为了试用）。
- 多块 GPU（#364、#384）：setup 在低 RAM 模式下推荐一块 GPU（常驻变体还没有层级切分），并询问；
  `--gpus 0,1`（或答 2）用映射变体（`--mmap-experts`）在它们之间共享模型：两块卡合计装下更多
  expert，两位用户测得比单卡快 1.3-1.6 倍，但操作系统文件缓存可能在长 prompt 期间把 RAM 填到
  0 空闲。`--yes` 保持一块 GPU。带 `--resident-experts` 的配置以 `--gpus` 启动时带说明切换到
  `--mmap-experts`，引擎也以警告而不是拒绝的方式把那块对子按 `--mmap-experts` 跑。**自 0.1.40
  起（#642、#848）** setup 要求在切分上能跑常驻变体的引擎，所以它像对任何配置一样把两块卡和
  常驻都保留：[MULTI_GPU.md](MULTI_GPU.md#using-it)。

<a id="-the-prompt-path-borrows-room"></a>

**面向小 RAM 的映射区（Linux，需显式开启，0.1.39，PR #640）：** `STRATA_ARENA_MMAP=1` 把原生包
的 expert 区从包的 `experts.bin` 只读映射，而不是读进锁页 RAM，给那种 GPU 装下大多数 expert 但
RAM 很小的 PC（2 块 16 GB 的 GPU 配 32 GB RAM：服务期间约 1 GB -> 25 GB 可用）。首次启动写
`experts.bin`（磁盘装得下的话），之后的启动映射它；GPU 所装 expert 的页交还给操作系统。配
`--pcie-frac 0` 运行（GPU 不拿映射别名）。不带该变量时什么都不变。

<a id="a-mapped-arena-for-small-ram"></a>

**Windows 上释放映射的 expert 页（需显式开启）：** `STRATA_FILE_RELEASE=1` 让
`FileExpertSource` 在 expert 的 GPU 上传完成后裁剪它们完整的文件支撑页，包括借给 prefill 随后
回补的槽。它对 `experts.bin` 和直接 GGUF 视图都有效；共享的边界页与私有/锁页缓冲区不动。不设
或 `0` 保持之前的行为。在一台配 RTX 4080 Laptop 和雷电口 RTX 3090 的 32 GB Windows 11 笔记本上，
IQ2_XS 带 `--mmap-experts --layer-split 12 --trim-stage-weights`，在 1K/4K/16K prompt 试验中把
可用 RAM 中位数从 0.51 提到 12.19 GiB，prefill 吞吐低 3.7-4.4%。它没有减少已提交内存或物理 SSD
读取。这是一个工作集提示，不是解除映射，也不保证操作系统丢弃它的文件缓存。见
[配置、测量与限制](../../bench/results/2026-10-04-windows-mapped-release/README.md)。

<a id="releasing-mapped-expert-pages-on-windows"></a>

**不带 `experts.bin` 的低 RAM 模式（引擎 0.1.31）：** 对原生包（IQ2_XS、IQ3_XXS、IQ3_S、Coder、
Swift、`tools/iq_pack.py` 打包的 Q2_0；不是 setup 为 AVX-512 CPU 做的规范 Q2_0 包）映射模式不再
需要包的 `experts.bin`：包里没有它时，引擎映射模型自己的 GGUF 文件，并按 `native_experts.txt`
标注的位置读每个 expert 的 gate、up 和 down 行（文件先对照它校验：每个张量的名字、类型、形状、
偏移和边界）。这省掉磁盘上那 23-50 GB 的拷贝。回答相同：在 Coder 上，`experts.bin` 与 GGUF 各
64 个贪心 token 给出相同的 token 和 logits。从 GGUF 读一个 expert 是三次读取而不是一次，所以
引擎用 8 个线程（`STRATA_FETCH_THREADS`）加一次批量页请求（Windows
`PrefetchVirtualMemory`）取一层缺失的 expert。包里有 `experts.bin` 时什么都不变。setup 还没有
用这条路。

<a id="low-ram-mode-without-expertsbin"></a>

**Linux 文件层，0.2.0 默认（零拷贝页缓存 expert、io 预读、decode 闸门）：** 当 expert 从就地
读取的 GGUF 读取（原生包，`--mmap-experts` 也是），Linux 现在默认跑三件事（一个 `experts.bin`
包，例如 NVFP4 那些，除非你设置否则它们保持关闭：那里还没测）。(1) **零拷贝**
（`STRATA_COLD_OVERLAP=4`）：就地读取的 GGUF 的 expert，若其三个张量的页都在页缓存里（mincore），
就直接从映射计算；此前每个这样的 expert 都要先组装进暂存缓冲区（每个 Q2_0 expert 约 1.5 MB 的
拷贝，在 decode 的关键路径上）。字节相同、行相同、输出相同（同一性门 IDENTICAL）。(2) **io
预读**（`STRATA_IO_PREFETCH=1`，下面描述的路径）：整 blob 的 `pread`，并提前读 router 预测的下几层
expert。(3) **decode 闸门**（`STRATA_IO_PF_GATE=1`）：每 16 个 decode 窗口评判一次 decode 的
预读——它只在页缓存为 expert 留有余地时运行（可用内存含最紧的 cgroup 限制除以 expert 字节：低于
0.55 关、高于 0.65 开，`STRATA_IO_PF_GATE_LO` / `_HI`），且只在它划算时运行（16 个窗口里预读至少
128 MB 且用掉其中 10%，`STRATA_IO_PF_GATE_USEFUL`；为此关闭时每 256 个窗口探测一次）。关闭时
decode 像没有 io 预读那样读（前瞻预测一层并用 madvise 提示暖机）；两种情况下 prompt 路径都保持它
的 pread。在模拟 12 GB 卡的 RTX PRO 5000 上实测（`--expert-cache 4448`、就地 Q2_0 GGUF、每个引擎
之前清页缓存、RAM 由 cgroup 限制），0.2.0 默认对照 0.1.4x 的，3 个交错对，故事 / 代码 / doc8k
decode 和 4K / 20K prompt：32 GB RAM 时 decode 1.70 / 2.56 / 2.88 倍（34.0 / 28.0 / 21.7 ->
57.6 / 71.7 / 62.6 tok/s）、prompt 1.54 / 1.91 倍；16 GB 时 decode 1.16 / 1.25 / 1.25 倍、
prompt 1.63 / 1.68 倍（闸门在那里保持关闭）；32 GB 里 `--resident-budget-gib 18` 时 decode
1.16 / 1.10 / 1.30 倍、prompt 1.55 / 1.91 倍；故事和代码的贪心 token id 相等（doc8k 在两种情况
下都因自适应层跨运行变化）。在真实 RTX 3060 12 GB（32 GB cgroup，3 对）上，仅零拷贝就给 decode
1.14 / 1.44 / 1.47 倍、prompt 1.06 / 1.02 倍。有了它们，容器的 RAM 检查把 cgroup 的干净页缓存
（active 与 inactive）算作可用，如同裸机上 MemAvailable 所做：此前 32 GiB cgroup 里的 18 GiB RAM
档在 io 预读的启动读取把它们的缓存留在组里之后会重检成 "1.2 GiB available"，它的文件层切换到
无缓冲读取（那也让零拷贝保持关闭）。关闭开关：`STRATA_COLD_OVERLAP=0`、`STRATA_IO_PREFETCH=0`、
`STRATA_IO_PF_GATE=0`；启动日志点名开着什么。Windows 和 macOS 不变。

<a id="linux-file-tier-020-defaults"></a>

**Linux 文件层 I/O 路径（`STRATA_IO_PREFETCH=1`；0.1.40.2，自 0.2.0 起在 Linux 上默认开——见上）：**
映射的文件层经页错误读 expert。开启后，本层未缓存的 expert 和 router 预测的下一层 expert
（`STRATA_LOOKAHEAD_K`、`STRATA_IO_PREFETCH_DEPTH`，开启时默认 2）由 I/O 线程
（`STRATA_IO_PF_THREADS`，默认 8）以整 blob 的 `pread` 读进页缓存，CPU 内核像之前一样读映射。
字节相同，所以输出相同（贪心 IQ3_XXS 和 Q2_0 哈希一致，预读关对开）。`STRATA_IO_PF_STAGE=1`
改为把每个未缓存 expert 暂存进缓冲区并让该层等它们；`STRATA_IO_PF_AHEAD=0` 只保留按需读取。
每请求日志行 "file tier I/O" 显示操作系统从盘读了什么（/proc/self/io，major faults）对照引擎的
expert 读取；`STRATA_IO_STATS=1`（或开预读）加上页缓存命中 / 未命中拆分（mincore）和预读的已用 /
未用计数。实测（交错 A/B，10 对，200-token 贪心中位数，每次运行前清页缓存，内存限制用 cgroup）：

<a id="linux-file-tier-io-path"></a>

| box, lane | default (fill) | `STRATA_IO_PF_STAGE=1` |
|---|---|---|
| RTX 3060, 32 GB, IQ3_XXS | -3.4% | **+25.7%** |
| RTX 3060, 16 GB, IQ3_XXS | -1.8% | -32.6% |
| Radeon 780M iGPU, 16 GB, Q2_0 (adaptive tier off) | -0.7% (story +3.8%, code +4.1%) | -19% |

expert 在页缓存里暖了之后（3060 上 46 GB、核显机器上 60 GB）文件层已经以常驻速度运行，所以那里
没有可赢的；差距在冷启动和低 RAM 各档。所以它默认关闭，而暂存变体只值得在 32 GB 级的机器上试试。
**核显警告：** 在 Radeon 780M 上，预读配自适应层（`--adapt-every`，默认开）在内存压力下约 9 次
运行里 7 次重置 GPU（引擎打印警告并按要求运行）；带 `--adapt-every 100000` 约 40 次运行没有重置。
原因未查明。

<a id="with-the-experts-warm-in-the-page-cache"></a>

**一个 RAM 预算（引擎 0.1.31，`--resident-budget-gib N`）：** 给 expert 不全装得下的模型的常驻
变体：expert profile 排名最热、GPU 缓存没装的 N GiB expert 在启动时拷进 RAM（锁定；驱动允许整份
预算时锁页），其余从文件读。它隐含 `--mmap-experts` 并留下 4 GiB 余量。Windows 上，可用提交容量
也限制预算；更大的 N 会被钳到更小的限制减 4 GiB 再加 256 MiB 余量，并带一条消息。被钳的预算不再
让随后的安全检查失败（#403）。完全无法保住的预算是一条警告，所有 expert 从文件读。Setup 用
`--resident-budget-gib N` 设 N。当排名后的 expert 之后预算还有空间时，prompt 路径借的槽也像没有
预算时那样把它们的 expert 保留在 RAM（引擎打印 `the prompt path's lend region: N of its M slots
keep their experts in RAM too`）；此前预算总是把那些排除，于是每个 prompt 都从模型文件重读被借槽
的 expert（RX 6900 XT，UD-Q4_K_XL，预算 71 GiB 剩 7.5 GiB 未用：每轮 agent 对话 0.5-2.6 GB 的文件
读取、每个全新 8K-16K prompt 从 SSD 的 1.9-2.8 GB 变成零；全新 prompt 读快 3-8%，prompt 路径的
host 暂存从每请求 0.8-2.1 秒降到 0.06-0.5 秒）。GGUF 就地读取时它还暖下一层可能的 expert：CPU 在
处理一层时，一个线程把下一层的 router 用到这一层的输入上，并向操作系统请求 GPU 和 RAM 预算都没装
的预测 expert 的页（只是页——计算的 expert 是同一批；`STRATA_LOOKAHEAD=0` 关闭它）。这就是让
[Unsloth 的 UD-Q4_K_XL](UNSLOTH_Q4.md)（72 GiB 的 expert）在 64 GB 的 PC 上运行的东西：RTX 5070
上 N = 40 时 7-8.5 tokens/s，这些改动之前约 3 tokens/s。

<a id="a-ram-budget-engine-0131"></a>

**在 Windows 下混合 Intel CPU 上实测的需显式开启开关（Core i7-13700K，8 P + 8 E 核，RTX 5090，
64 GB RAM，Windows 11，Swift 1.5 IQ3_XXS @128K 配置下）**：

<a id="opt-in-switches-measured-on-a-hybrid"></a>

- `--pool-affinity measure`（混合 CPU，原生包）：`all` 和 `p-cores` 两种布局在启动时按包自己的
  expert 计时（约一秒）并取更快的那个。源码构建的引擎 0.1.41，同一 prompt：`all`（15 个 worker）
  每个 verify 窗口在 CPU 池里花 6.7 毫秒、解 148 tok/s，`p-cores`（7 个 worker）2.2 毫秒、解
  188-191 tok/s；同一天另一次运行里 `all` 只花 2.4 毫秒，所以这个差异时来时去。测量只在启动时
  取一次：一天内四次服务器启动里它三次读 `all` 为 14.8、21.9 和 25.3 毫秒对 `p-cores` 的 6.1 到
  6.2 毫秒并选了 `p-cores`，一次读 `all` 6.6 毫秒对 8.6 毫秒选了 `all`。它在会话期间不再重看，
  所以 E 核速度在引擎运行中会变的地方，`--pool-affinity p-cores` 是稳妥的选择。未测：其他 CPU、
  Linux、AMD 卡。

<a id="--pool-affinity-measure"></a>

**精确的单 token expert 行（0.1.45，需显式开启，`STRATA_CPU_IQ_ALL_EXACT1=1`，#1415 #1544）：**
在 #152 规则之下（expert 分组里只有一个 token），没有 AVX-512 的 AVX-2 CPU 对 i-quant gate/up 行
跑 ggml-cpu 的 `vec_dot`。带该开关时，IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS 和 IQ3_S 改走 Strata 的
AVX-2 decode，用 ggml 的 float 顺序（同样的每 lane 整数和、每块一次 fmadd、格式常数在
`hsum_float_8` 之后），所以输出不变、只有时间变（`iq_avx2_parity`：每格式 5,120 个输出中 0 个与
ggml 的不同；带开关的同一性门 10/10 加上 RTX 3060、2x A4000 和 R9700 机器上 exact_pp 4K/20K 一致）。
它只在 ggml-cpu 自己是 AVX2 构建（不是 ISA-floor 构建、不是 AVX-512）处使用，且不与
`STRATA_IQ_MT_MIN=1` 同用。每 expert、gate/up、单线程：IQ3_S 在 Threadripper PRO 3975WX（Zen 2）
上 -32%、Ryzen 5 5600X（Zen 3）上 -11%，IQ2_XXS -5..-6%，IQ3_XXS -5%（Zen 2）/ +1%（Zen 3），
IQ2_S 0%（Zen 2）/ +7%（Zen 3，更慢）；在 Core Ultra 7 265（VNNI，gather）上，P 核上 IQ3_S -60%、
IQ3_XXS -39%、IQ2_XXS -34%、IQ2_XS -24%、IQ2_S -3%。端到端，带引擎的 decode 计时：窗口的 CPU 部分
在 Xeon E5-2643 v3 + RTX A4000 上降 5%（46.9 毫秒的窗口中 22.2 -> 21.1 毫秒，10 个交错对上 decode
+0.5%，故事和代码 prompt 上 +2.5%）和 Core Ultra 7 265 + RTX 3060 上（39.5 毫秒中 8.2 -> 7.8
毫秒，decode +0.4%），在 Threadripper PRO 3975WX + R9700 上不动（14.5 毫秒，+0.4%，5 对）；没有
情形端到端达到 +3%，所以开关保持关闭。

<a id="exact-one-token-expert-rows-0145"></a>

**0.1.40 新增的开关（除非注明都为关；没有一个改变默认输出）：**

<a id="switches-added-in-0140"></a>

- `--kv-grow`（或 `STRATA_KV_GROW=1`；`--no-kv-grow` 关闭）：只为使用中的上下文分配物理 GPU K/V
  内存。Expert 缓存使用剩余内存并在上下文增长时释放槽。逻辑的 `--max-context` 容量不变。带层级
  切分时，每个 GPU 分页管理它自己的 K/V 池和 expert 缓存。此选项要求每块 GPU 有 VMM 支持、一个
  expert profile、仅 GPU 的 K/V（不带 KV 流式），以及所有 expert 在 RAM（不是常驻低 RAM 模式）。
  它不支持远程 expert、`--batch`、`--vram-elastic` 或 `--peer-device`。这些条件让选项失效时引擎
  会报告。在活跃的请求序列期间，把最近的 K/V 目标保持在最多 65,536 个 cell。30 秒没有完成的请求
  之后，把多余的 K/V 内存交还 expert 缓存。当前请求仍可使用完整逻辑上下文。视觉和预留压力路径
  可以立即回收内存。这条规则避免短暂聊天切换之间反复的收缩、expert 回补与增长操作。已有的收缩
  迟滞仍然适用。
- `--host-core last`（或 `STRATA_HOST_CORE=last`，Windows）：host 线程跑在最后一个物理核、worker
  拿第一个。Windows 把 GPU 的中断送到第一个核，在 GPU 标志上自旋的 host 会在那里等它们
  （`--host-core first` 是默认；启动日志点名那些核）。它不动混合 CPU。`--host-core sibling`（或
  `STRATA_HOST_CORE=sibling`）把 host 保持在第一个核、但它的另一个硬件线程（SMT）上，所以中断
  占用第一个逻辑处理器、worker 保住每个核；它在混合 CPU 上也有效，在没有 SMT 的核上等于 `first`。
  在 4060 Ti + 5080 层级切分、i9-14900KF 上，它把逐次 ±6% 摆动的 decode 变成稳定的，平均快 8%。
- `STRATA_ADAPT_LAG=2`（#764）：窗口在自适应层的交换旧到两个窗口时才开始采用它们（默认 1，同
  0.1.39）。`STRATA_PREFILL_EQUAL=1`（#693）：prompt 段以等大的分块读取，而不是满分块加短的最后一
  块（改变舍入）。`STRATA_OWNED_PRICE=exact`（#796）：缓存定大小按 prompt 路径自己缓冲区的真实
  分配定价，而不是 0.1.39 的规则。
- `STRATA_Q2_BITPLANE=1`：给有 AVX2、无 AVX-512 的 CPU 的 Q2_0 用位平面行内核（改变最后的比特）。
  `STRATA_NO_AVXVNNI=1` 在有 AVX-VNNI 的 CPU（Alder Lake、Sapphire Rapids 及之后）上关掉 i-quant、
  IQ4_NL 和 Q2_0 行的 AVX-VNNI 形式；它们与 AVX2 形式给出相同的比特，行快 4-25%。
- 草稿层：`--lookup-chain-min M`（`--lookup-chain` 所扩展的最短上下文匹配，默认 3）、
  `--mtp-hnorm stream`（每流一个 norm，同 llama.cpp 的 MTP 图）和 `--mtp-draft-vocab FILE`（你自己
  的 token 子集，代替 `<mtp>/draft_vocab.bin`）。
- 其他：[AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration) 里的 HIP
  `STRATA_DENSE_MMQ=1` 和 `STRATA_HIP_ADAPT_KERNEL_COPY=1`；[TROUBLESHOOTING.md](TROUBLESHOOTING.md)
  里的 `STRATA_KEEP_EMPTY_TURNS=1` 和 `STRATA_TOPK_STREAM=0`。

<a id="--kv-grow"></a>

**启动时的提前读取（Linux）：** 权重、原生 dense 矩阵、按 profile 填充 GPU 缓存、常驻 RAM 副本
和 MTP 草稿文件都在读取之前被提前请求（madvise / posix_fadvise WILLNEED，128 KiB 步长），于是盘
看到的是深队列而不是一次一个页错误。在 Gen3 NVMe（RTX 5090、32 GB、Q2_0 常驻 @262K）上实测：
70 秒就绪，而不是约 920 秒；填充从 39 MB/s 变 3.2 GB/s。`STRATA_READ_AHEAD=0` 关闭它；
`STRATA_FILL_AHEAD=N` 设提前请求多少个填充对（默认 256）。

<a id="read-ahead-at-start-linux"></a>

**各部分贡献来自哪里：** 带 `--stats` 时引擎打印 decode 的各层（`expert tiers`：来自 RAM 副本的
blob、来自文件的 blob 与 MB、读它们花的时间；`routing prefetch`：多少次文件读取被提前暖过）。
服务器日志按请求有同样的信息（`expert tiers: GPU ... hits ...; RAM ... blobs, files ... blobs
... MB read`），`GET /metrics` 为每个近期请求列出 `ram_blobs`、`file_blobs` 和 `file_mb`（引擎
0.1.31 或更新）。它还列出每个请求的推测草稿 `drafts_offered` 和 `drafts_accepted`（引擎没报告时为
`null`），以及 `totals` 里服务器启动以来的总和（#457）。它的 `hit_rate` 是回答期间被查找的 expert
中 VRAM 的份额：GPU 经 PCIe 读的 expert（`--pcie-frac`）不在其中，所以更高的 `--pcie-frac` 会让它
升高即使 decode 变慢。`pcie_share`（引擎 0.1.39 或更新，#588）是它们在所有被路由 expert 中的份额，
服务器日志和 Monitor 标签页把它显示在命中率旁边。

<a id="how-much-came-from-where"></a>

**decode 窗口的时间去向（profiling，#610）：** 在环境里带 `STRATA_DECODE_TIMING=1`（GPU 那边还要
`STRATA_VERIFY_PROFILE=1`）启动服务器。之后每个请求的引擎日志会有一行 `strata decode timing:`——
窗口数、每窗口 token 数，以及每窗口的 verify 时间拆成等 GPU、CPU expert 池（plan、激活量化、jobs）
和暂存，外加 commit 与草稿——再加一行 `strata decode GPU stages (ms/window):`，给每个阶段的 GPU
时间（GDN 和 QSA 层、VRAM expert 命中、router、head……）。GPU profile 用事件为每个阶段计时，所以
会让 decode 慢一点：用它做对比，不要用来测速度。它对任何包都有效；`--gpu-stages`（逐层图的一次 token
重放）拒绝原生（IQ）包，它没有那样的图。

<a id="where-a-decode-windows-time-goes"></a>

**下一层 expert 预测有多准（仅测量）：** `STRATA_LOOKAHEAD_STATS=1` 在任何 expert 层上跑 router
前瞻，并对照下一层的路由打分：每 100 个窗口引擎日志有一行 `strata lookahead stats:`，给出 GPU 缓存
之外每层的 expert 数、top-k / k+4 / k+10 猜测以及按投票排名的前 4 / 8 / 12 的召回率与精确率、它们
能覆盖多少 PCIe 份额，以及预测线程每窗口的时间。它消耗 CPU（预测线程在 expert 池之外），所以用它做
测量、不要用来计时。`tools/replay_cache_policy.py` 离线把一份 `--dump-routing` 跟踪重放通过自适应层
的交换策略。

<a id="how-good-the-next-layer-expert-prediction-is"></a>

首 token 时间就是 prompt 长度除以 prompt 速度：Q2_0 约 4K 时 4 秒、32K 时 25 秒、128K 不到 2 分钟、
262K 时 4.5 分钟（引擎 0.1.13 让长 prompt 快了约一倍，见下）。

<a id="time-to-first-token-is-prompt-length"></a>

**更快的 prompt（引擎 0.1.13）：** prompt 以最多 8,192 token 的分块读取而不是 2,048（`--prefill
auto`：缓冲区装得进其所借 expert 缓存槽的最大分块，且一个请求只借它的 prompt 需要的量）；expert
用 llama.cpp 的量化 MMQ 内核相乘，而不是先展开成 FP16；当前层的 attention 运行时下一层的 expert
经 PCIe 流式进来；PLE 块一次为整个分块运行；未锁页的 expert 由辅助线程拷贝。在 RTX 5070 12 GB、
64 GB RAM、32K-token prompt 上实测：Q2_0 572 -> 1,290 tokens/s，IQ3_S 383 -> 1,208。经服务器
（Q2_0，128K 上下文）：999 个 token 353 -> 438 tokens/s、6,927 个 token 529 -> 1,077、28,584 个
token 584 -> 1,249。输出速度不变。大海捞针 5/5（1K-262K）。细节与质量检查：
[`bench/results/2026-09-28-prefill-speed`](../../bench/results/2026-09-28-prefill-speed/README.md)。
已有安装在下次 START-HERE / setup.sh 启动时切换到 `--prefill auto`。原始数字：
[`bench/results/`](../../bench/results/)。[论文](../paper/Strata-Paper.pdf)解释每个数字。

<a id="faster-prompts-engine-0113"></a>

## 其他 GPU（估计）

<a id="other-gpus-estimated"></a>

未测量——由上面的运行估计（同一 CPU 和 64 GB RAM）：GPU 部分按显存带宽缩放，CPU 部分按卡的 VRAM
多装多少 expert 缩放。请当作 **±20%** 看待。数字是 *prompt / 输出* tokens/s。Prompt 数字早于引擎
0.1.13，后者让实测卡上的 prompt 速度约翻倍；一张卡能拿到多少取决于它的 PCIe 链路（expert 经由它
流式），所以它们仍是更早的估计。

<a id="not-measured---estimated-from-the-runs"></a>

| GPU | Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti 16GB | Q2_0 | ~341 / ~80 | ~472 / ~87 | ~501 / ~81 | ~492 / ~72 | ~476 / ~62 | ~435 / ~53 |
|  | IQ2_XS | ~291 / ~80 | ~406 / ~77 | ~434 / ~63 | ~426 / ~62 | ~413 / ~51 | ~383 / ~47 |
|  | IQ3_XXS | ~249 / ~66 | ~359 / ~65 | ~381 / ~56 | ~374 / ~54 | ~363 / ~45 | - |
| RTX 3090 24GB | Q2_0 | ~355 / ~128 | ~491 / ~140 | ~521 / ~130 | ~512 / ~115 | ~495 / ~100 | ~453 / ~85 |
|  | IQ2_XS | ~303 / ~131 | ~422 / ~128 | ~451 / ~103 | ~444 / ~102 | ~430 / ~85 | ~398 / ~78 |
|  | IQ3_XXS | ~260 / ~106 | ~374 / ~103 | ~396 / ~89 | ~390 / ~85 | ~378 / ~71 | - |

更多 VRAM 比更快的 GPU 更重要：每多 1 GB 约多装 700 个 expert，GPU 上每个 expert 就是 CPU 不必算的
一个。3090 的 24 GB 拿走了大部分 CPU 工作。（自 0.1.14 起 expert profile 给全部 24,576 个 expert
排名；此前缓存停在 8,000 个，约 10-14 GB。`tools/make_profile.py` 用你自己的 prompt 建 profile：
带 `--dump-routing trace.bin` 跑引擎一次，看该工具的说明。随带的 profile 已给每一对排名，所以跟踪
只在使用 `--reorder` 时改变顺序（0.1.39，#587）：你的跟踪里的对排前面，然后基线的，再其余的。）

<a id="more-vram-matters-more-than"></a>

## 选哪个模型？

<a id="which-model"></a>

三个都是 [ISTA-DASLab 的 GSQ-RCO 量化](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)，
对象是 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。

<a id="all-three-are-ista-daslabs-gsq-rco-quantizations"></a>

| Model | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | 约 34 GB expert + 约 6 GB | 最快 | 好 |
| **IQ2_XS** | 68 GB | 约 36 GB expert + 约 6 GB | 接近 Q2_0 | 略好 |
| **IQ3_XXS** | 76 GB | 约 43 GB expert + 约 6 GB | 较慢（CPU 工作更多） | 最好 |

64 GB RAM 时三个都装得下（IQ3_XXS 请关掉浏览器，并把它的上下文保持在 128K 及以下）。48 GB 时只有
Q2_0 / IQ2_XS 可能装得下。32 GB：Coder（见下）。

<a id="with-64-gb-of-ram-all-three-fit"></a>

### 或者：Coder（expert 减半，为代码）

<a id="or-the-coder-half-the-experts-for-code"></a>

<a id="or-the-coder-half-the-experts"></a>

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**
是 ISTA-DASLab 的 expert 剪枝发行版：每层 512 个 expert 保留 256 个（仍每 token 活跃 10 个），用 RCO
在代码、agentic 和视觉校准数据上挑选；其作者报告 SWE-bench Verified 达到完整模型的 91.3%、
LiveCodeBench v6 的 98.7%。一个尺寸，因其 *原始* 参数每参数 1.89 比特而叫 IQ1_M；保留的 expert 按
IQ3_S 的方式存储（gate/up 是 IQ2_S-IQ4_XS、down 是 IQ4_NL/Q2_0）。Shard 1 是 29.6 GB（expert：23 GB
RAM），所以它能在 **32 GB RAM** 上运行，262K 时需要 64 GB。它的 shard 2 和视觉编码器就是原版的文件：
装了原版时 setup 只下载 shard 1。Strata 随带它的 expert profile
（`data/expert-profile-coder.bin`，随带的排名通过发行版的 `rco-allocation.txt` 映射到保留的 expert：
12 GB 的卡上 72% 的 expert 读取命中 GPU）。图像可用；实验性速度投影能在它上面加载和运行（那是为完整
模型做的）。

<a id="qwen38-flash-next-gsq-rco-coder"></a>

```
START-HERE.bat --setup --family coder
```

### 或者：Swift 1.5（想得更短的微调模型）

<a id="or-swift-15-a-fine-tune-that-thinks-shorter"></a>

Setup 的第一问也提供 **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**，
UkisAI 对 Qwen3.8-Flash-Next 的微调，训练目标是少得多的思考就到达答案（其作者：思考 token 少 63%、
答案早 1.8 倍、准确率损失不到 1%）。同样的架构、除 Q2_0 外原版的尺寸、它自己的视觉编码器；Strata 以
相同速度运行它（4K，IQ2_XS：465 prompt / 78.7 输出 tokens/s，原版是 467 / 78.3）。其作者推荐
**IQ2_XS**（他们的 Q2_0 标记为实验性）。许可证是 Swift Open License 1.0——请在模型页面读它。

<a id="the-setups-first-question-also-offers"></a>

我们的小检查（8 道推理题，默认 thinking，IQ2_XS）：两个模型都得 **8/8**；Swift 用 **1,234** 个输出
token 花 28 秒，原版 **2,682** 个花 46 秒——差别大部分来自一道原版想了 1,524 个 token 的题。不是基准，
但与那个说法一致。

<a id="our-small-check-8-reasoning-questions"></a>

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

### 实验性：Unsloth 的 UD-Q4_K_XL

<a id="experimental-unsloths-ud-q4_k_xl"></a>

同一模型的 4 位量化（111 GB，72 GiB 的 expert）。引擎 0.1.32 起 setup 提供它（`--family unsloth
--model UD-Q4_K_XL`：一个 RAM 预算为你的 RAM 减 24 GB，其余从 SSD 读）；它做什么以及手动流程在
**[docs/UNSLOTH_Q4.md](UNSLOTH_Q4.md)**。64 GB 的 PC 配 12 GB RTX 5070 上它写 7-8.5 tokens/s，多数
expert 从 SSD 读；在短贪心回答 97.5-99% 的位置上它选与 llama.cpp 读同一文件相同的 token，16K prompt
之后 90-91%，差异主要在接近平局处
（[实测](UNSLOTH_Q4.md#quality-against-llamacpp-on-the-same-file)）。

<a id="a-4-bit-quantization-of-the-same-model"></a>

## 开始之前

<a id="before-you-start"></a>

你**只需要一个显卡驱动**：NVIDIA 580 或更新（用 NVIDIA App 或从
[nvidia.com/drivers](https://www.nvidia.com/drivers) 更新），AMD 用
[INSTALL.md](INSTALL.md#what-you-need) 里那个。其他的一切首次都会替你装好。

<a id="you-need-only-a-graphics-driver"></a>

| | |
| --- | --- |
| GPU | NVIDIA 算力 **7.0 或更新**，**12 GB VRAM 或以上**（8 GB 能跑，慢）。RTX 30/40/50 用现成的引擎。Volta 卡如 16 GB Tesla V100 用 CUDA 12.x 本地编译；仅 Ampere 的可选内核被禁用，prompt GEMM 用可移植 FP32 回退。或 AMD **Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700、RX 6800 / 6900 系列**：[AMD_HIP.md](AMD_HIP.md)。 |
| RAM | 推荐 **64 GB**（见上表）。 |
| CPU | 带 AVX2 的 x86-64（最近约 8 年任意 Intel/AMD 桌面 CPU）。AVX-512（Ryzen 7000/9000）稍快。没有 AVX2 的更旧 CPU 是实验性且慢：[Older CPUs](INSTALL.md#older-cpus-experimental)。 |
| 磁盘 | 模型约需 70-80 GB 空闲，MTP 层约 6 GB（带图像 +1 GB）。**AVX-512 CPU 上的 Q2_0** 还会一次性写一个约 40 GB 的包给快速 CPU 内核（其布局的 34 GB expert，加 dense 权重）。强烈建议 NVMe SSD。 |
| OS | Windows 10/11，或 Linux（Ubuntu 22.04/24.04 和 Debian 12/13 全部自动安装）。 |

首次启动会安装：本文件夹内的 `.venv/`、`engine/` 和 `third_party/`；模型文件（`models/`、`packs/`、
`mtp/`，70-120 GB）在**本文件夹旁边的 `Strata-data`**，所以 Strata 的新副本（解压到别处的更新）能找到
它们并以同样的方式把自己配置好。这个位置按用户记忆（`%APPDATA%\Strata\settings.json`、
`~/.config/strata/settings.json`）；`--data-dir` 换一个。0.1.16 之前的安装在下次启动时被搬到那里
（同一磁盘上是重命名；其他磁盘上的文件就地使用）。没有 Python 时装 Python 3.12（给你的用户账户，不需要
管理员）、一个私有 Python 环境、NVIDIA 的 CUDA 库（来自 pip，约 0.4 GB）、给 RTX 20/30/40/50 的现成
Strata 引擎、模型和 MTP 草稿层。如果没有现成引擎适合你的 PC，它会提出安装构建工具（Windows 上 Visual
Studio Build Tools + CUDA Toolkit，Ubuntu 上 `build-essential` + CUDA）并为你的 GPU 编译引擎（先询问；
一次性 20-40 分钟）。针对 Volta 时必须 CUDA 12.x，因为 CUDA 13 不再发出 sm_70 代码。

<a id="what-the-first-start-installs"></a>

---

## Windows

<a id="windows"></a>

### 双击 `START-HERE.bat`

<a id="double-click-start-herebat"></a>

<a id="double-click-start-heremat"></a>

**第一次**它会问四个问题并做其余一切：

<a id="the-first-time-it-asks-four-questions"></a>

1. **哪个模型？** Qwen3.8-Flash-Next（原版）或 Swift 1.5（想得更短的微调模型）。
2. **哪个尺寸？** Q2_0、IQ2_XS 或 IQ3_XXS（它按你的 RAM 推荐一个）。
3. **多少上下文？** 8K 到 256K token（它按你的 VRAM 推荐一个）。
4. **图像？** 是 / 否（见 [图像](#images-vision)）。

然后它下载并准备一切（模型 66-76 GB，所以首次启动要一会儿；中断的下载从停下的地方继续）并**启动
模型**：你的浏览器打开 `http://127.0.0.1:8080`，即 Strata 应用。它有三个标签页：

<a id="then-it-downloads-and-prepares"></a>

- **Chat：** 流式回答、模型的 thinking（开始回答后折叠收起）、带复制按钮的代码、开启图像时的图片，
  以及采样与 thinking 级别设置。聊天保存在你的浏览器里。
- **Monitor：** 模型在做什么（读 prompt，带进度，或在写，以多少 tokens/s）；GPU 负载、VRAM、温度、
  功耗与 PCIe 流量；CPU、RAM 与磁盘；使用中的上下文；最近的请求。

Mac（Metal 后端）上 Monitor 的 GPU 负载和 VRAM 来自 IOKit 的 IOAccelerator 节点——驱动的 Device
Utilization % 与其在用系统内存，对照设备的 Metal 工作集——卡说明其余部分：不用 root 没有温度或功耗
读数（`powermetrics`），而 Apple Silicon 的 GPU 坐在片上，所以没有 PCIe 链路。引擎自己的 VRAM 数字
在它的日志里。

<a id="on-a-mac-metal-backend"></a>

- **About：** 模型与引擎设置，以及其他应用要连接的地址。

<a id="-about"></a>

`http://127.0.0.1:8080/?q=your question` 打开它并已经带着一个新聊天在提问。API 在
`http://127.0.0.1:8080/v1` 供你的应用使用。

<a id="http1270018080qyour-question"></a>

**之后每一次**，`START-HERE.bat` 只是启动模型（把 34-43 GB 装进 RAM 要 30-90 秒）。什么都不再下载。
关掉窗口就停止模型。

<a id="every-time-after-that"></a>

```
START-HERE.bat --setup                          install another model, or change context / images
SETUP.bat                                       the same (double-click it)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --gguf-dir D:\models\IQ2_XS       use GGUF files you already have
START-HERE.bat --data-dir E:\Strata-data         keep the model files somewhere else
START-HERE.bat --port 8081                      another port
START-HERE.bat --gpu 1                          another GPU (numbered as nvidia-smi; setup picks the one with the most VRAM)
START-HERE.bat --calibrate                      tune the engine for this PC (about 15-30 minutes, longer on a slow card), then start
```

装了多个模型时它会问启动哪个。`run-<model>.bat` 直接启动某个模型。

<a id="with-more-than-one-model-installed"></a>

**为你的 PC 调优（`--calibrate`，引擎 0.1.19）。** 这些引擎设置取决于 PC 多过取决于模型：

<a id="tuning-for-your-pc-calibrate"></a>

- VRAM 缺失的 expert 中拷到 GPU、而不是由 CPU 计算的份额（`--pcie-frac`：快的 PCIe 链路和较慢的 CPU
  想要更多，笔记本更窄的链路更少）；
- 草稿层必须有把握才往一次校验里多加一个猜测（`--spec-min-p`）；
- 多少个 CPU 线程计算 expert（`--pool-workers`：在有能效核的 CPU 上，更少可能更快）；
- VRAM expert 缓存多勤地跟随对话（`--adapt-every`、`--adapt-swaps`、`--adapt-decay`）：CPU 读缺失
  expert 很慢的 PC（DDR3 或少通道的 DDR4）从多交换中获益。在 Xeon E5-2673 v3 配 DDR3、RTX 4060 Ti 插
  PCIe 3.0 x8 上，每窗口 160 次交换实测比默认 +7.9%（Q2_0，交换晚一个窗口生效，#764）；见 #906；
- 两块或更多卡上（层级切分），让各卡同时读取的 prompt 分块（`--prefill-pipe`，全新 prompt 的 prompt
  读取速度而不是输出速度；见 docs/MULTI_GPU.md）。

<a id="-the-share-of-the-experts-missing-from-vram"></a>

默认值是在 Ryzen 5 7600 配 RTX 5070 上测的。Setup 在安装后提出在你的 PC 上测量它们；
`START-HERE.bat --calibrate`（Linux：`./setup.sh --calibrate`）随时可做。它用每个设置测量输出速度，
只在快 3% 以上时保留一个。结果按 PC 和模型记忆（在数据文件夹记录旁边的设置文件里），所以更新会保留它。

<a id="the-defaults-were-measured-on"></a>

测量 worker 数量需要一个新引擎，所以模型会被加载多次：PC 很忙，且每次重启会有一两分钟可能停止响应。
然后它像普通的 `START-HERE.bat` 那样启动模型。

<a id="measuring-the-worker-count-needs"></a>

**手动的 CPU 任务粒度。** `--pool-tasks N` 设置每个批处理 CPU expert Gate/Up 与 Down 阶段里行任务的
目标总数，不是线程数或每 expert 的任务数。默认 `0` 保持每个参与线程三个任务（启用时含 host）。取值
`1..4096` 会被各阶段的行数限制；大于池容量的原生批次把该目标应用到每个子批次。例如
`--pool-workers 13 --pool-tasks 192` 在使用默认 host worker 时用 14 个参与线程。更多任务可以减少核之间
的不均衡，但也增加调度开销：请在相同 worker 数和工作负载下与 `0` 对比。这不改变内核、阶段屏障或 PCIe
放置，也不影响传统的单 token/oracle 回退。Setup 的 `--calibrate` 还没有调它。对服务器，把
`"--pool-tasks", "192"` 加到它配置里已有的 `args` 列表，然后重启它。

<a id="manual-cpu-task-granularity"></a>

**引擎启动多少 CPU 线程（`--pool-workers`）。** 没有该设置时引擎按 CPU 而不是逻辑线程数选 CPU expert
worker 的数量：每个*物理*核一个 worker（绝不选 SMT 兄弟），主线程那个核除外；17 核及以上的 CPU 少一个
核；多插槽机器上只用主线程所在 CPU 插槽的核；最多 32；不超过容器的 CPU 限制（docker `--cpus`）允许的
量；在能效核多于性能核的混合 CPU 上，除一个之外的性能核加一半的能效核。当操作系统不说哪些逻辑处理器共享
一个核（某些容器）时它按一半计。启动行说明是什么限制了该数字（`CPU pool: 31 workers by default
(32 physical cores, 64 logical processors; bounded by ...)`）。Worker 靠自旋等待，所以没有自己专属核的
worker 会占用别人的时间：20 核的 CPU 上 24 个 worker 比默认慢 44%，32 核的 EPYC 上 63 个 worker 慢 35%。
配置里的数量（`--pool-workers N`）总是原样使用；当它高于 CPU 能良好使用的量时，引擎在启动时打印一条平实
的警告并给出推荐数量（`strata cpu pool: warning: ...`）。Setup 除了混合 CPU 的那个之外不写 worker 数量，
同样封顶，并给 CPU 视觉编码器最多 16 个线程（配置 `vision` 条目里的 `--threads` 可改）。

<a id="how-many-cpu-threads-the-engine-starts"></a>

### 开机自动运行（任务计划程序）

<a id="running-it-at-startup-task-scheduler"></a>

为了让模型在登录时就绪，人们从**任务计划程序**（或服务）启动 serve。注意：Windows 会给这类上下文限流，
模型约 40 GB 的 expert 加载会以 **约 0.05 GiB/s（13-14 分钟）**爬行，而不是 **约 1.4-1.5 GiB/s（约 35
秒）**——启动慢 24 倍。在 RTX 5070 Ti + Ryzen 7 9800X3D + NVMe 上实测，同一二进制、相同参数、相同缓存
状态：

<a id="to-have-the-model-up-at-logon"></a>

| How the serve starts | Expert load |
| --- | ---: |
| 双击 / 终端 / SSH | 1.42-1.52 GiB/s（约 35 秒） |
| 用默认值的任务计划程序 | 0.05 GiB/s（821-841 秒） |
| 带下面两项设置的任务计划程序 | 1.42 GiB/s（35 秒） |

在任务的属性里同时设这两项（默认值正好相反）：

<a id="-how-the-serve-starts"></a>

- **优先级级别：Normal**（选项标签页；默认是 Below normal），以及
- **使用最高权限运行**（常规标签页；不带它任务以受限的用户令牌运行——那也会剥掉
  `SeLockMemoryPrivilege`，即 Windows 大页所需的特权）。

<a id="-priority-level-normal"></a>

（两项是一起改的，所以各自单独的效应没有测。）如果模型启动仍然慢，引擎会在它的
`loaded ... GiB at ...` 行下方打印一条点名这个原因的提示。

<a id="both-were-changed-at-once"></a>

**大页需要一次新的登录（#1412）。** 给运行引擎的账户授予"锁定内存页"（`secpol.msc`，用户权限
指派）要在下一次登录才生效：Windows 在会话开始时把该特权放进访问令牌，所以授予后要注销再登录
（或重启）。在那之前启动日志仍会说大页被拒绝。`tools\windows\enable_large_pages.ps1`（以管理员
运行）用 `secedit` 把它授予当前账户。

<a id="large-pages-need-a-new-logon-1412"></a>

### 在终端里聊天（可选）

```
.venv\Scripts\python chat.py
```

---

## Linux

<a id="linux"></a>

```bash
./setup.sh
```

同样的问题、同样的自动安装（用 `sudo apt` 装 Python，只有在必须编译时才装构建工具），同样的
启动：`http://127.0.0.1:8080`。之后运行 `./setup.sh`（或 `./run-<model>.sh`）直接启动模型。
选项与 Windows 相同（`./setup.sh --setup`、`--model Q2_0 --yes`、`--gguf-dir /data/Q2_0`）。
终端聊天：`.venv/bin/python chat.py`。

<a id="the-same-questions-the-same-automatic-install"></a>

- **更新：** `git pull`，然后 `./setup.sh`：引擎源码变了就重新编译（改动的文件要一两分钟）。
  编译失败会说明并继续用你原有的引擎。
- **其他发行版**（Arch、Fedora...）：先用你的包管理器装 C++ 编译器和 CUDA Toolkit 13（Arch：
  `sudo pacman -S base-devel cuda`）；setup 会在 PATH、`/usr/local/cuda*` 和 `/opt/cuda*` 里
  找到 `nvcc`，其余照常。
- **WSL** 可用（Ubuntu 24.04 测过），有一个限制：NVIDIA 驱动在那里只能锁约 1 GB RAM，所以 KV
  流式（`--kv-resident`）关闭、KV 缓存留在 VRAM，而且 expert 从未锁页 RAM 拷到 GPU（prompt 比
  原生 Linux 慢）。

<a id="-updating-git-pull-then"></a>

### 把 V100 IQ3_S 服务器作为用户服务启动

<a id="starting-the-v100-iq3_s-server-as-a-user-service"></a>

仓库的 `strata-v100.service` 模板把服务器作为用户服务跑在两块 Tesla V100 上。它与上面默认的
Q2_0 安装分开；只在 IQ3_S 部署上用。

<a id="the-repositorys-strata-v100service-template"></a>

首次启动之前：

1. 把检出放到 `~/strata-v100`。指向检出的符号链接即可。检查检出里有
   `strata-iq3_s-arena.json`，使用原生 IQ3_S expert 包，且其 `env` 对象里有
   `STRATA_SPLIT_RING=192`。这份配置和必需的 `.strata-service.env` 文件保持私密。不要提交
   认证密钥。
2. 检查模型存储已挂载。该 unit 在服务器之前运行 `/usr/bin/mountpoint --quiet
   /mnt/strata-models`。挂载缺失时启动失败，unit 在 10 秒后重试，而不是从空磁盘路径加载模型。
3. 安装模板并重载用户管理器：

   ```bash
   mkdir -p ~/.config/systemd/user
   cp strata-v100.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   ```

4. 先停掉旧的、手动启动的服务器，新服务才不会撞手动服务器占用的端口。
5. 启动并启用服务：`systemctl --user enable --now strata-v100.service`。
6. 要在登录前的开机阶段启动服务，启用 lingering：`loginctl enable-linger "$USER"`。该命令可能
   需要管理员权限；由机器的特权策略决定。

<a id="before-the-first-start"></a>

日常命令：

<a id="daily-commands"></a>

```bash
systemctl --user restart strata-v100.service
systemctl --user status strata-v100.service
curl -fsS http://127.0.0.1:8088/health
journalctl --user -u strata-v100.service -f
```

该 unit 在任何退出之后 10 秒重启服务器，并把模型挂载保持为前置条件。磁盘缺失会阻止启动；它不会
加载一个不完整的模型。

<a id="the-unit-restarts-the-server-after-any-exit"></a>

---

## 与其他程序共享 GPU（可选）

默认模型一直保持加载直到你关闭 Strata。在同时玩游戏、渲染或跑另一个模型服务器的 PC 上，三个
服务器选项（默认都关；也可作 `strata-<model>.json` 的键）把 VRAM 交还：

<a id="sharing-the-gpu-with-other-programs-optional"></a>

| 选项 | 配置键 | 作用 |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | 600 秒无请求后卸载模型；下一个请求再加载它 |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | 只有那么多 VRAM 空闲时才加载未加载的模型（为正在交还的内存最多等 15 秒），否则回答 **503** "the GPU is in use by another program"，而不是在游戏留下的残余上开始（多块 GPU 时检查第一块） |
| `--before-load "cmd"` | `"before_load": "cmd"` 或 `["cmd", "arg"]` | 模型重新加载之前运行的命令，比如卸载另一个服务器的模型的那条 |
| | `"before_load_timeout_s": 120` | 该命令最多能运行多久（默认 120 秒）。它以非零退出或超时（对端没在跑会让 `curl` 以 7 退出）时，空闲 VRAM 说了算：设了 `min_free_vram_mib` 而空闲不足时**不**加载模型，请求得到带 `Retry-After` 的 **503**；有余量（或没设限制）则加载，日志里带一条警告。`"before_load_strict": true` 在失败时一律拒绝；`"before_load_ignore_failure": true` 一律加载 |

`POST /unload` 立刻卸载（有请求在跑时返回 `409`；`POST /unload?wait=1` 先等正在进行的轮次结束，
最多 300 秒，或 `?wait=<seconds>`，好让第二台服务器真正让出显卡），`POST /load` 在请求之前预先
加载（两者都要 `Content-Type: application/json`，例如 `curl -X POST -H "Content-Type:
application/json" localhost:8080/unload`）；`/health` 报 `"loaded"`，`/v1/models` 把它列为
`unloaded`（像 llama.cpp 的路由器），`/props` 设置 `is_sleeping`，Monitor 显示该状态。卸载结束
引擎进程——图像开着时还有图像编码器——它们会在最先被再次启动，如同一次启动——所以它们的 VRAM
和 RAM 立刻交还。模型文件留在操作系统文件缓存里，所以只要那块 RAM 没被别处需要，重新加载只需几
秒。在 RTX 5060 Ti 16 GB 上以低 RAM 模式的 Q2_0 实测：卸载约 0.3 秒；对未加载模型的一次请求在
4.6 秒（文本）或 14.7 秒（图片，图像编码器在 CPU 上）后得到回答。

<a id="post-unload-unloads-it-now"></a>

**边服务边交还一部分 VRAM（#533，需显式开启，单块 NVIDIA GPU；每个请求与回答见
[VRAM_ELASTIC.md](VRAM_ELASTIC.md)）。** 配置里设 `"vram_elastic": true`（引擎参数
`--vram-elastic`）后，expert 缓存以 512 MiB 段分配（`"vram_segment_mib"`），
`POST /v1/vram` 配 `{"reserve_mib": 8000}` 在请求之间把它缩小到给另一个程序空出那么多 VRAM；
`{"reserve_mib": null}` 把它长回接近满容量、保留引擎启动时的预留（`--vram-reserve-mib`），
`{"reserve_mib": 0}` 全部收回。正在运行的请求先完成。交还的那些段里的 expert 在 CPU 上计算，
与任何缓存之外的 expert 一样，所以回答继续、变慢；长回把同样的 expert 放回同样的槽。没有任何东西
自动改大小。不与层级切分、辅助缓存、`--peer-device` 或常驻低 RAM 模式同用。在 RTX 5070 12 GB
配 Q2_0（4.8 GiB 缓存）上实测：`{"reserve_mib": 6000}` 花 78 毫秒、释放 4.3 GiB（缓存为 prompt
路径留 0.5 GiB），decode 44 -> 33 tok/s；长回花 92 毫秒，回答与缩小前逐 token 相同。分段缓存的
行为需要 `--vram-elastic`。

<a id="giving-part-of-the-vram-back"></a>

仅 GPU 的 KV 增长也支持 `POST /v1/vram`，不需要 `--vram-elastic`。该模式从每块 GPU 的 expert
缓存释放整块物理内存段。它保持 KV 映射、缓存地址和 prompt 借出区间固定。外部分配释放后，
`{"reserve_mib": null}` 把可用的段映射回来并从 expert profile 填新槽。它保留每块 GPU 的启动
VRAM 预留。KV 仍在使用的内存在 KV 池收缩之前不能回到 expert 缓存。剩余的缓存段供不起足够内存时，
请求的预留可能失败。

<a id="gpu-only-kv-growth-also-supports"></a>

**跨重启保住 expert 缓存学到的东西（需显式开启，引擎 0.1.36，#477）：** 一次启动按随带的 profile
填 GPU 的 expert 缓存，随后自适应层（`--adapt-every`）移入你的请求用到的 expert。在
`strata-<model>.json` 里设 `"expert_profile_save": "expert-profile-learned.bin"`，引擎就会把它
存成一份 profile——VRAM 里的 expert 排最前，然后自启动以来统计到的路由，然后是随带的顺序——在干净
退出时以及每 10 分钟（请求间隙）写一次（`"expert_profile_save_every": 5` 换间隔，`0` 只在退出
时），写入临时文件再重命名，所以崩溃绝不会留下半个文件。下一次启动在它属于同一模型的 profile 时
从它开始，而不是配置里的 `--expert-profile`（否则照旧从配置的开始）。相对路径在 Strata 文件夹内；
每个模型一个文件，每个项目一份 profile 同理（把键指向另一个文件）。这个文件是你用模型做什么的
指纹：它留在你的 PC 上。不设该键时什么都不统计、什么都不写。再次运行 setup 会重写配置：那时把键
加回来。在设了 `STRATA_VERIFY_NO_HOST` 的 Intel Arc 卡上（Battlemage 上 SYCL 的默认，没有自适应
层），profile 按设备自己的路由计数排名、在下次启动生效；从一种文本学到的 profile 会加速那种文本
（代码，Arc Pro B70 上 decode +13-19%，token 相同）并拖慢其他种类（散文，-18%），所以每种工作
保持一个文件。

<a id="keep-what-the-expert-cache-learned-across-restarts"></a>

**expert 缓存大小是一份预算，而在 Windows 上超额的预算走换页而不是失败。** `--expert-cache N`
（`"expert_cache": N`）是 N 乘最大 expert blob，引擎只拿它与写槽**之前**读到的空闲 VRAM 比较。
`--expert-cache auto`（默认）按同一个数字定大小，但**在槽真正写好之后再次检查**，装不下就缩小；
第二次检查只在 auto 有（循环在 `!auto_cache` 处中断）。WDDM 下一块分配在被触碰前并不常驻，之前
读到的空闲数字可能高出约 1 GB，而驱动默认的 sysmem 回退会把超额分配放进系统内存而不是失败——于是
启动看起来正常、横幅照报槽数、只有速度崩掉。启动行是线索：`... MiB of VRAM free with everything
loaded`，低于约 256 MiB 时它加上 `LOW: requests may stall; add --vram-reserve-mib N` 并给出能
留出余量的值。

<a id="the-expert-cache-size-is-a-budget"></a>

在 RTX 4080 SUPER 32 GB、IQ3_S、引擎 0.1.39、`--max-context 1048576`（yarn 因子 4）上实测，
全新 43,969 / 45,670 / 47,956-token prompt 各生成 200 个，每行一次引擎启动（其余配置同
`bench/results/2026-10-04-community-rtx-4080s-iq3s`）：

<a id="measured-on-an-rtx-4080-super-32-gb"></a>

| expert cache | slots | VRAM free at start | decode tok/s |
| --- | ---: | --- | --- |
| `--expert-cache 8900` (11,631 slots, 22.07 GiB) | 11,631 | 0 MiB (`LOW`, suggests `--vram-reserve-mib 1212`) | 13.7 / 14.7 |
| `auto` | 11,178 | 217 MiB (`LOW`) | 102.1 / 122.3 |
| `auto` with `--vram-reserve-mib 1500` | 10,766 | 1,075 MiB | 88.9 / 96.5 / 101.4 |
| `--expert-cache 7000` | 9,148 | 4,130 MiB | 94.1 / 97.3 |
| the same 11,631-slot cache at `--max-context 524288` | 11,631 | 299 MiB | 113.2 / 107.9 |

453 个槽（0.86 GiB）把 13.7 与 102 tok/s 分开，而慢的那次运行缓存命中率**更高**（91.4 / 94.7%
对 90.4 / 93.9%），所以问题不是 miss：是 0 MiB 空闲这个状态付出了 7 倍代价。避开它有两种办法：
缓存保持 `auto`，或保留显式大小并提高 `--vram-reserve-mib`——引擎在给缓存定大小**之前**先扣掉它
（700 -> 1000 -> 1500 让空闲数字从 217 到 575 到 1,075 MiB，代价约 400 个槽）。

<a id="slots-086-gib-separate-from"></a>

**在 Windows 上核查它。** `\GPU Process Memory(<pid>)\Shared Usage` 和 `Dedicated Usage`
（性能监视器，或 `Get-Counter`）看起来像合适的仪器，但在这台机器上它们**没有**区分开 13.7 tok/s
和 102 tok/s 的运行：`Shared` 在 262/524K 时读约 55.5 GB、1M 时 62.4 GB，每次运行都如此，因为它
把刻意锁页的内存算在内（expert 区，这里 50.3 GB，加锁页 K/V，1M 时 6.19 -> 12.38 GiB）。日志里的
空闲行才是可靠的线索，而在 NVIDIA 硬件上给引擎可执行文件设 `CUDA - Sysmem Fallback Policy` 为
`Prefer No Sysmem Fallback`，是让超额缓存失败而不是换页的另一条路（这里未测）。随测量报告在
[issue #781](https://github.com/Niko1221/Strata/issues/781)。

<a id="checking-it-on-windows"></a>

同样的形态出现在第二块更小的卡上。16 GB RTX 5080 配 786,432 上下文、UD-IQ4_XS、INT8 KV，设了
`Prefer No Sysmem Fallback` 且显式 `--expert-cache 4000`：把 `--vram-reserve-mib` 从 770 降到
0 让缓存从 2,782 涨到 3,115 个槽（768 MiB），而卡的已用内存只动了 129 MiB，命中率保持平（59.9%
到 63.0%），decode 从 40.9 跌到 16.6 tok/s。每一步都在 0 MiB 空闲下到达 READY，所以即使关了回退
它仍然启动并退化而不是失败——留给整卡其余部分的余量比多出来的槽值钱，这与上面实测的排序一致
（enkynakamura，#781；每步一个 256-token 样本，所以上面几行有噪声）。

<a id="the-same-shape-appears-on-a-second"></a>

**空闲数字在启动之后还会变。** `... MiB of VRAM free with everything loaded` 行是在 verify
窗口捕获之前读的（每个捕获的窗口、每个 `--batch` 槽布局都要几十 MiB：#2065 里那行的 538 MiB
在最后一次捕获后变 395 MiB，一台 RTX 4090 D、`--batch 2`、262,144 上下文）。Windows 下低于约
1 GB 时驱动可以把 GPU 内存换页到系统内存而不是失败：70K token 时 decode 从 91 跌到 3 tok/s，
然后一个窗口永不完成，看门狗在 60 秒后停止引擎。自 0.1.43.1 起每个请求重读空闲 VRAM，一旦低于
1 GiB（其他地方 512 MiB）就一次性说明 `WARNING: only N MiB of VRAM free at the start of this
request` 并给出能留出余量的 `--vram-reserve-mib`，停滞报告也打印该数字。引擎只负责说；修法是更大
的 `--vram-reserve-mib`、更小的 `--max-context` 或 `--batch`，或给引擎设 `Prefer No Sysmem
Fallback`。

<a id="the-free-figure-moves-after-startup"></a>

**名为 "request" 的停滞可能是操作系统在给引擎自己的内存换页。** 在一次切换对话的请求之前，引擎
把离场的对话暂存进 RAM：其运行中检查点和 K/V 的副本，之后躺在普通内存里直到对话回来。在一台
64 GB 的 PC 上配 `--resident-budget-gib 40`、8 GiB 对话缓存和 256K 上下文，#2049 里工作集是 56 GB
中已提交的 8-30 GB，引擎每秒约 190 次页错误（每次若是页面读取则不足 1 MB/s）：0.1.41 和 0.1.43
上都出现过五次超过 60 秒的无声一步（线程转储：主线程在检查点拷贝里，每次同一指令）。那份报告里日志
的 `verify window (last window, not the current stage) ... host at layer step 48; the GPU rang
48` 行是上一个请求的最后一个窗口，不是停滞的那个。自 0.1.43.1 起，暂存在阶段名里点名
（`request: parking the previous conversation`），它的副本向看门狗报告进度，Windows 在判断文件层
是否繁忙时按每单位 8 次而不是 64 次页错误计。慢本身的解药是内存：更小的 `--resident-budget-gib`、
更小的 `--conversation-cache-mib`，或把页面文件放 SSD。

<a id="a-stall-named-request-can-be"></a>

---

## 使用它

<a id="using-it"></a>

服务器监听 `http://127.0.0.1:8080`（setup 里用 `--port` 改，或编辑启动脚本）。

<a id="the-server-listens-on-httpprometheus"></a>

| API | 端点 |
| --- | --- |
| OpenAI Chat Completions（流式与非流式、工具） | `POST /v1/chat/completions` |
| Anthropic Messages（流式与非流式、工具） | `POST /v1/messages` |
| OpenAI Responses（流式与非流式、工具；无状态，[下文](#the-responses-api-and-codex-cli)） | `POST /v1/responses` |
| 模型列表 / 健康 | `GET /v1/models`、`GET /models`、`GET /health` |
| 模型属性 | `GET /props`（也接受 `?model=<loaded-model-id>`） |
| 模型此刻在做什么 | `GET /status`、`GET /slots`（忙或闲，带 `n_prompt_tokens`；带 `--batch` 时每槽一项） |
| 把对话存/恢复到文件（会话文件，下文） | `POST /slots/0?action=save\|restore` |
| Monitor 标签页显示的一切（引擎、实时状态、最近的请求、硬件） | `GET /metrics` |
| 同样的数据给 Prometheus，用 vLLM 的指标名（以 `Accept: text/plain` 或 `?format=prometheus` 请求） | `GET /metrics` |
| Prometheus 文本格式的请求总量与在跑的东西，用 llama-server 的指标名 | `GET /metrics/prometheus` |
| MCP 服务器、它们的状态和工具（[下文](#tools-from-mcp-servers)） | `GET /mcp` |

`GET /metrics` 用带 vLLM 名字的文本格式回答 Prometheus 抓取（`Accept: text/plain` 或
`application/openmetrics-text`）——`vllm:num_requests_running` / `_waiting`、
`vllm:kv_cache_usage_perc`、token 与请求计数器、`vllm:prefix_cache_queries_total` /
`_hits_total`（读取/复用的 prompt token）、`vllm:spec_decode_num_draft_tokens_total` /
`_accepted_tokens_total`（MTP 草稿），以及直方图 `vllm:time_to_first_token_seconds`、
`vllm:inter_token_latency_seconds` 和 `vllm:e2e_request_latency_seconds`——所以为 vLLM 服务器写的
仪表盘和告警能读这一台。带 API 密钥时抓取器把它作为 bearer token 发送。其他请求保持 JSON。

<a id="get-metrics-answers-a-prometheus-scrape"></a>

JSON 自己那些 vLLM 没有名字的事实，以 `strata:` 打头、按其 JSON 键命名，进同一次抓取，所以 Monitor
标签页的面板和 Grafana 面板读同一个值：`strata:live_state{state="..."}`、`strata:live_tok_s`、
`strata:live_prefill_tok_s_mean`、`strata:live_prompt_read`、`strata:engine_max_context`、
`strata:totals_prompt_seconds_total` / `strata:totals_decode_seconds_total`、`strata:last_hit_rate`
和 `strata:last_decode_tok_s`（上一个请求的），以及硬件——`strata:gpu_util`、
`strata:gpu_mem_used_bytes`、`strata:gpu_mem_total_bytes`、`strata:gpu_temp_celsius`、
`strata:gpu_power_watts` 和 `strata:gpu_energy_wh`（驱动加载以来该卡用的能量；每卡一个样本，标签
`gpu`）、`strata:cpu`、`strata:ram_used_bytes`、`strata:ram_total_bytes`。服务器没有的值（没有 GPU
遥测、更旧的引擎）没有样本而不是零。

<a id="the-jsons-own-facts-that-vllm-has-no-name-for"></a>

开箱即用，在 `docs/monitoring/`：`prometheus.yml`（抓取配置，API 密钥作 bearer token）、
`servicemonitor.yaml`（给 Prometheus Operator 的同样配置）和 `grafana-strata.json`，一个待导入的
Grafana 仪表盘（它会问 Prometheus 数据源）：运行与等待的请求、每秒 token、首 token 时间、token
间隔与请求时长（p50 / p95）、prompt 缓存与 MTP 草稿率，以及引擎自己的面板（状态、批处理槽、速率、
expert 缓存命中率、GPU 和 RAM）。它的第一行只用 vLLM 的名字，所以也能读一台 vLLM 服务器。

<a id="ready-to-use-in-docsmonitoring"></a>

`/models` 和 `/v1/models` 只列出已加载的模型，带它的上下文上限与输入模态。`/props` 在可得时暴露
原始 chat 模板、上下文上限、配置的生成默认值（共享设置优先）、模型路径和引擎版本。上下文指完整引擎
上下文，不是常驻 KV 窗口。`n_predict: -1` 表示没有固定的输出上限。未配置的采样字段被省略。
`/slots` 用 llama.cpp 的名字：`n_ctx` 和 `n_prompt_tokens`（运行中请求的 prompt 大小，结束后保留：
前端上下文仪表除以 `n_ctx` 的那个数）。`/props` 里的 `total_slots` 是 `--batch` 槽的数量（不带为
1）。`autoload` 没有效果；未知的 `model` 返回 404。这些元数据端点和 `/slots` 在配置了 API 密钥时
需要它。它们不加载、卸载或重启模型。

<a id="models-and-v1models-list-only-the-loaded-model"></a>

`chat_template_caps` 用 llama.cpp 的字段名。Strata 加载活动模板时通过渲染小请求检查它们；被拒绝或
省略的功能为 `false`，任何以某种方式未通过探测的模板只关掉那个功能（服务器照常启动）。
`supports_tools` 检查工具定义与 XML 调用指令；`supports_tool_calls` 检查历史中的调用使用 Strata 的
XML 格式并保留工具结果。`supports_parallel_tool_calls` 检查一个 assistant 轮里的多个调用，
`supports_system_role` 检查开头的 system 消息。`supports_preserve_reasoning` 表示较早的 assistant
轮在渲染后的 prompt 里保留它们的 `reasoning_content`。它不描述新回复是否返回 reasoning。这些是给
客户端的兼容性提示，不是模型会照做某个请求的保证。Zed 的 llama.cpp 提供方读这个对象来决定提供工具。

<a id="chat_template_caps-uses-llamacpps-field-names"></a>

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

- **Thinking 级别：none、low、medium、high。** 模型在回答前先思考（以
  `reasoning_content` 流式发出，Anthropic：`thinking` 块）。按请求选择思考多少——聊天页面里
  （"Thinking" 菜单）、`chat.py` 里（`/think low`），或经 API：

  | API | 方式 |
  | --- | --- |
  | OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"`（也支持 `"reasoning": {"effort": ...}` 与 `"chat_template_kwargs": {"enable_thinking": false}` 两种写法同样接受 |
  | Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`、`"thinking": {"type": "disabled"}`，或 `"thinking": {"type": "enabled", "budget_tokens": N}`（低于 2K = low，低于 8K = medium，更多 = high） |

  不带设置时模型用它自己的默认值，**high**。`none` 立刻回答（最快）；`low` 把思考保持很短。这些
  级别是模型训练时用的指令，不是硬性 token 上限：简单问题上三种都短暂思考，难题上 `high` 思考最久、
  最准确。要让每个不指定级别的请求以另一个级别开始，把 `"reasoning_effort": "low"`（或 `none`、
  `medium`、`high`）放进 `strata-<model>.json`、在 `"args"` 之外（#1641）：引擎没有这个选项，
  `"args"` 里的 `--reasoning-effort` 会在启动时带一条说明移动到该键。网页应用的共享 Chat 设置
  （"Use for other apps too"）和请求自己的级别优先于它。

<a id="thinking-levels-none-low-medium-high"></a>

- **工作时让 Windows 保持清醒（需显式开启，#1727）。** `strata-<model>.json` 里的
  `"prevent_sleep": true` 在至少一个请求运行或等待期间阻止 PC 睡眠（Windows 的
  `SetThreadExecutionState` 配 `ES_SYSTEM_REQUIRED`；屏幕仍可能熄灭），并在服务器空闲以及退出时
  把正常睡眠规则交还。默认关。Linux 和 macOS 上它什么都不做（启动时有一行说明）。

<a id="keep-windows-awake-while-it-works-opt-in-1727"></a>

- **硬性 thinking 预算（需显式开启）。** 请求（OpenAI 或 Anthropic）里的
  `"reasoning_budget_tokens": N` 把思考限制在 N 个 token：到达时服务器以一条简短收尾行和
  `</think>` 结束它，模型从那里开始回答（引擎从它已持有的内容继续，所以什么都不重读）。收尾行是
  客户端所见 thinking 的一部分、计为输出 token。`strata-<model>.json` 里的
  `"reasoning_budget_tokens": N` 为每个请求设置它；请求自己的值优先，`0` 表示无预算。默认关；
  Anthropic 的 `"thinking": {"budget_tokens": N}` 选择级别（如上），且自 0.1.45 起同时也是预算
  （同一请求里的 `reasoning_budget_tokens` 更优先）。

<a id="a-hard-thinking-budget-opt-in"></a>

- **max_tokens 计入思考，所以回答保住自己的空间（0.1.45）。** 给整个回复设上限的客户端（桌面
  agent 的 harness 每个请求发 `max_completion_tokens: 8192`）加上模型要长期思考的任务，过去会
  在思考中途被切断、没有回答。现在一个开着思考、没有自己的预算、`max_tokens` 大于等于 1,024 的
  请求会得到一条规则：当上限只剩四分之一（至少 512 个 token）时收尾思考，模型在那块空间里回答
  （8,192：思考最多 6,144，回答 2,048）。不带 `max_tokens` 的请求（上下文的其余部分）得到其四分之
  三的预算。`strata-<model>.json` 里的 `"answer_room": false` 关闭它。Qwen 推荐最多 262,144 个
  token 的 reasoning 和 131,072 个的回答：客户端不应把回复压到远低于上下文允许的量。

<a id="max-tokens-counts-the-thinking"></a>

- **默认情况下过去的 thinking 不是下一个 prompt 的一部分。** 服务器移除历史
  `reasoning_content`，包括工具调用历史里的。它仍为当前回复生成并流式 thinking。要包含所提供的
  历史 reasoning，设 `"chat_template_kwargs": {"preserve_thinking": true}`。这是一个显式的请求
  选项，不是默认。

<a id="past-thinking-is-not-part-of-the-next-prompt"></a>

- **在其思考内部结束的回复（#1053，需显式开启）。** 某些轮写一句 reasoning 然后直接写轮末 token
  而没有 `</think>`：内容为空，agent 停住。`strata-<model>.json` 里的
  `"reasoning_close_retry": true` 关闭一次思考（如同 thinking 预算所做的）并继续，每请求一次，只
  针对以那种方式结束、没有回答也没有工具调用的回复。默认关。

<a id="a-reply-that-ends-inside-its-thinking-1053"></a>

- **卡在一个 token 上的回复被结束（0.1.39，#606）。** 当回复连续重复同一个 token 256 次时，服务器
  在那里以 `finish_reason` `"length"` 结束它并在窗口里说明：陷入循环的模型，或者一个永远答同一个
  token 的坏状态（#606 见过 36,689 个 `!` token）。`strata-<model>.json` 里的
  `"repeat_stop_tokens": N` 设置该连续长度；`0` 关闭它（给真的想要同一个 token 很多次的请求）。

<a id="a-reply-stuck-on-one-token-is-ended"></a>

- **非有限 logits 结束请求并丢弃缓存状态（#879，默认开）。** 罕见地，一个窗口的 logits 变 NaN、
  回复退化为重复的 token 0（`!`）（#606）；毒素在对话的 K/V 和递归状态里，所以带着 prompt 缓存，
  该对话之后的每一轮都被持续毒化。0.1.39 及之后的 q8_1 钳位移除了迄今发现的溢出；这个防护兜住剩下
  的任何原因。`--serve` 的每个 verify 窗口（decode 窗口、层级切分的 `--pipeline-windows` decode
  窗口、prompt 的短尾巴、批处理槽）在 GPU 上于贪心选择的旁边检查其 head logits（一个写进映射标志的
  小内核，在窗口本来就要做的同步之后读取）。被标记的窗口不发它的任何 token，引擎打印一行
  `non-finite logits (#879)` 带位置，以 `DONE ... nonfinite` 结束请求（一个槽：`BDONE <slot> <n>
  nonfinite`），下一个请求先丢弃活动会话、每个检查点、每个暂存的对话和每个空闲槽（仍在运行的槽先
  完成它的回复，之后不保留），并从 token 0 读取它的 prompt。服务器对尚未发出任何东西的请求透明地
  重试一次；已经开始的回复以 `"length"` 结束并在服务器窗口里留一行；重试上第二次触发以错误结束请求
  （回答前 503，流中一个 error 事件）。带副本时，服务器忘记那个副本持有的东西。`repeat_stop_tokens`
  作为后备保留。内核只写它的标志、host 只读它们，所以没有 NaN 的回答就是引擎在 `STRATA_NAN_GUARD=0`
  （不记录任何内核）下给出的回答：在 RTX 3080 上用 RVN IQ3_S 配置、固定 expert 集（一个 16K-token
  系统 prompt 上 8 轮的聊天重放），防护开与关在 3 对里给出逐字节相同的回答（其中一对
  `STRATA_ROUTE_TAIL_SKIP=0`、一对 `STRATA_SH_STREAM=0`）；又有一次关防护的运行从第一个回答起就与
  其他所有不同，而另一次关防护的运行与开防护的运行一致，所以那是运行之间的差异，不是防护。它的代价：
  平均每窗口 4.8 微秒 GPU 时间（最多 7.8 微秒），decode 时间的 0.009%（Nsight Systems，424 个
  窗口）。一次性命令行（不带 `--serve` 的 `strata`）和 `sycl/` 里的 Intel（SYCL）移植版不检查。
  调试开关：`STRATA_NAN_INJECT=<n>[,<n>...]` 把引擎进程里防护检查的第 n 个窗口（从 1 起对
  prompt-tail、decode 和批处理窗口计数）报为非有限，以便没有真实触发也演练整条路径。在那台 3080 上
  配 `=1,60`：新聊天的第一个窗口被标记，服务器从 token 0 重发该请求（客户端看到一个正常回答）；第
  60 个在下一轮回复中途，在 58 个 token 后把该回复结束为 `"length"`；再后一轮复用 0 个缓存 token
  并正常回答。

<a id="non-finite-logits-end-the-request"></a>

- **重复的 reasoning（需显式开启，#728）。** 上面的单 token 防护看不见重复整段文字的模型。
  `strata-<model>.json` 里的 `"reasoning_loop_recovery"` 取 `false`（默认）、`"stop"` 或
  `"recover"`（`true` 即 `"recover"`）。每 512 个输出 token，在一个完整的字符与解析器边界上，
  reasoning 在其最后 2,000 个词和标点（按最后 30,000 个词统计段落）上被测量，且其最后 2,048 个
  字符被检查至多 64 字符的周期（#1753：写成一个长词的循环，比如一个 24k 位数字串循环一个 11 位
  模式，在计数中是单个词、从不重复一个 token）；周期性的尾巴按完全重复计。若至少 25% 属于见过三次
  的 12 词段落，`"stop"` 在那里以 `"length"` 结束回复并在服务器窗口说明。`"recover"` 停止并排空
  那次生成，然后带模板的低努力句替换第一个 system 消息里的 xhigh 句，从它全部已生成的 token id
  继续一次（token id 的拼接：prompt 其余部分不解码也不重编码）。任务保持不变；不插入回答或
  `</think>`，两遍共享原始输出上限。仅在恢复时，温度被提到至少 1.0、presence 惩罚至少 1.5；客户端
  的 top-p、top-k 和 seed 从不改变。`/metrics` 记录 `reasoning_recoveries` 与覆盖率。这是一个策略，
  不是数值引擎的修复，也不是回答的保证。`"recover"` 需要第一个 system 消息里精确的 xhigh 句并跳过
  带图像的请求；重读改过的 prefix 要花 prompt 时间。两种模式都可能把重复的有用代码或检查误认为循环，
  所以测试时把记录的回答质量与完成率一起看。

<a id="repeated-reasoning-opt-in-728"></a>

- **不重读 prompt 就换 effort（需显式开启，0.1.39，#458）。** effort 的指令是 prompt 的第一样东西，
  所以只改 effort 的请求（agent 的 "think harder" 开关，快速工具步用 `none`）会把整个对话重读一遍。
  `strata-<model>.json` 里的 `"effort_position": "end"` 把每个请求的 prompt 开头渲染成默认 effort
  的，并把 `low` / `medium` effort 放进紧挨回答前的一条短 system 轮（没有思考：空的 thinking 块，
  照旧）；引擎（0.1.39+，服务器会检查）把它的检查点保持在那轮之前，所以下一个请求无论什么 effort
  都复用对话。默认（`"start"`）的 prompt 不变。模型在一个与训练时不同的位置看到非默认的级别；它在那
  里跟随得如何还没有测。

<a id="changing-the-effort-without-re-reading-the-prompt"></a>

- **没有要求思考的 Anthropic 请求（需显式开启，0.1.32，#278）。** 默认地，一个不带 `"thinking"`、
  effort 或预算的 `/v1/messages` 请求按模型模板的方式思考。`strata-<model>.json` 里的
  `"anthropic_thinking": "on_request"` 把这样的请求渲染成不带思考——Anthropic 自己的规则，也是
  Claude Code 的短辅助调用（几十个 token 的会话标题）所需要的；它真正的轮次在那边开着思考时会要求
  思考。

<a id="anthropic-requests-that-dont-ask-for-thinking"></a>

- **流式。** 带 `"stream": true` 一切都随产生随到达：思考、回答，以及工具调用（先工具名，然后参数
  一段一段，像 OpenAI 和 Anthropic 那样）。模型读长 prompt 期间，流发送 keep-alive，所以 agent 不会
  超时；服务器窗口每 15 秒打印进度，`GET /status` 说明它在做什么（`reading the prompt`、
  `answering`、目前的 token 数）。关闭连接或在你的应用里按停止会真正停止模型，所以下一个请求立刻
  开始。

<a id="streaming"></a>

- **其他形式的工具调用（需显式开启）。** `strata-<model>.json` 里的 `"tool_call_recovery": true`
  在模型把请求声明的工具写成 `<parameter=NAME>` 而不是 `<function=NAME>`、写成 `<tool_call>` 内的
  JSON（`{"name": ..., "arguments": ...}`）、写成行首无 `<tool_call>` 的 `<function=NAME>` 块
  （代码之外），或写成同一个 `<tool_call>` 里的第二个调用（不带该开关时第二个调用的参数并入第一个）
  时也能读出调用。那些形式里的其他东西保持原样是文本。在 1,462 个 agent 轮上实测（Claude Code 下的
  Qwen3.8，来自 signalnine/q27 的漂移语料）：开着时 98.3% 按预期读出，不带为 92.9%。默认关，所以
  除非你打开，客户端拿到的东西不变。

<a id="tool-calls-in-other-forms-opt-in"></a>

- **opencode 写在用户轮里的环境块（需显式开启）。** opencode 在其系统消息开头附近写一个 `<env>` 块
  （工作目录、今天的日期），在约 40K 字符的、每个会话都相同的指令与工具定义之前。prefix 缓存只复用
  prompt 的开头，所以每个新文件夹、每一天都把约 14K token 重读一遍。`"env_to_user": true` 把那个块
  移到第一条用户消息的前面：system 轮于是每个会话都一样，磁盘 prefix 缓存能恢复它。实测（RTX 5070 Ti
  Laptop，Swift 1.5 IQ3_XXS，第二个文件夹里的新 opencode 会话）：第一个请求 14,073 个 token、其中
  13,693 来自缓存，1.1 秒；不带它是读 14,753、11.9 秒；模型仍正确回答了文件夹和日期（3/3）。默认关。

<a id="opencodes-environment-block-in-the-user-turn"></a>

- **流中的 prefill 进度（需显式开启）。** `"return_progress": true` 把那个进度放到流上而不是只发
  keep-alive，作为一个空 delta 的 chunk 上一个额外字段：带 `total`、`cache`、`processed` 和
  `time_ms` 的 `prompt_progress`。那是 llama.cpp 的四个字段、含义也相同（`time_ms` 是从 prompt 开始
  读取以来的时间，还剩的工作是 `(total-cache) - (processed-cache)`），所以为 llama.cpp 画 prefill 条
  的客户端这里也能画。除非请求要求，它是关的，与 llama.cpp 一致。引擎每个 `--prefill` 分块说一行，所以
  那个分块就是步长：在两块 RTX 3090 上用 `--prefill auto`（8192 token）实测，一个 42,131 token 的
  prompt 在 15 秒内读完、发了六条。短于一个分块的 prompt 什么都不发，这是故意的：它唯一的一行在 prompt
  读完后才到，因为最后的 token 走 verify 窗口而不是批处理路径，所以它会停在离结尾至多 `--short-read`
  个 token 处。服务器丢弃离结尾 `--short-read` 之内的 PP，并按引擎的方式读 `--short-read`（args 里最后
  一个生效，负数为 0），所以被丢弃的行跟着配置的阈值。

<a id="prefill-progress-in-the-stream-opt-in"></a>

- **聊天应用。** 任何有 "OpenAI 兼容" 提供方设置的应用都能用：base URL
  `http://127.0.0.1:8080/v1`，任意 API 密钥。Zed 的 llama.cpp 提供方读 `/props` 并从
  `chat_template_caps` 提供工具。它的通用 OpenAI 兼容提供方不这样：在 Zed 设置里给那个模型加
  `"capabilities": {"tools": true, "chat_completions": true}`。
- **OpenCode**（#543）。`opencode.jsonc`（在你的项目里，或 `~/.config/opencode/`）的起点；字段名是
  OpenCode 的，版本不同请查它的配置文档：

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "strata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Strata (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "strata": {
            "name": "Qwen3.8-Flash-Next (Strata)",
            // context: what you chose in setup; output: what one reply may use (prompt + output must fit)
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },                  // sent as reasoning_effort
            "variants": {                                              // switch between them in OpenCode
              "low": { "reasoningEffort": "low" },
              "medium": { "reasoningEffort": "medium" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "strata/strata"
  }
  ```

  `limit.context` 设成你在 setup 里选的上下文：OpenCode 在到达它之前先压缩对话。`limit.output` 保持在
  它之下留有余量：prompt 加 `max_tokens` 超出上下文的请求，其 `max_tokens` 被缩短到剩余空间（见下面
  **上下文**）。要给思考设硬上限，把 `"reasoning_budget_tokens": N` 加进 `strata-<model>.json`（见上）。

<a id="chat-apps-any-app-with-an-openai-compatible-provider"></a>

- **Claude Code**（Strata 0.1.17 或更新）：设 `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`，并把
  `ANTHROPIC_MODEL` 设成它认识的某个 Claude 模型名（它拒绝不认识的名字；Strata 忽略这个名字），加任意
  `ANTHROPIC_AUTH_TOKEN`（或你的 `api_key`，如果你设了一个）。
- **Codex CLI**（0.1.39）：见下面 [Responses API](#the-responses-api-and-codex-cli)。
- **上下文。** setup 里选（8K-262K）。比它长的请求被拒绝，绝不悄悄截断。`max_tokens` 会超出上下文的
  请求以缩短到剩余空间的 `max_tokens` 得到回答（Claude Code 这样的 agent 每个请求都要 32,000）；思考
  预算跟着缩短后的上限。要严格的 llama.cpp 行为（400）在 `strata-<model>.json` 里设
  `"fit_max_tokens": false`，或给 `serve/server.py` 传 `--no-fit-max-tokens`。一点空间都不留的 prompt
  总是被拒绝。
- **模型别名**（0.1.32）。`strata-<model>.json` 里的 `"aliases": ["qwen", "local-model"]` 让模型在
  `/v1/models` 里也以那些名字列出（每个都有自己的 `id`，并在模型的 `aliases` 里），像 llama-server 的
  `--alias`；点名其中一个的请求以那个名字回答。其他名字照旧照样服务。
- **网页里的模型设置（0.1.39，#564）。** About 标签页的 Model settings 卡片显示并修改上面那些键中的
  一部分（`strata-<model>.json` 里）：`sampling` 默认值（temperature、top_p、top_k、min_p）、
  `reasoning_budget_tokens`、`fit_max_tokens`、`anthropic_thinking`、`effort_position`、`aliases`、
  `idle_unload_s`、`lazy_load`、`engine_silence_s`、`api_monitor`、`open_browser` 和
  `--vram-reserve-mib`。空的字段移除该键（其默认值）。文件的其他每个键原样保留，旧文件保存为
  `strata-<model>.json.bak`，模型从其下次启动起使用这个改动。只有 Strata 自己的页面能保存（JSON、设了
  密钥时带 API 密钥，与 Chat 设置相同）；网络、密钥、MCP 和程序类的键在那里不可编辑。

<a id="claude-code-strata-0117-or-newer"></a>

- **来自你网络上其他设备。** 服务器只监听你的 PC（`127.0.0.1`），除非你另有说明：用
  `START-HERE.bat --setup --host 0.0.0.0 --api-key some-long-secret` 跑 setup（或给
  `strata-<model>.json` 加 `"host": "0.0.0.0"` 和 `"api_key": "..."`）。服务器窗口于是打印这台 PC 的
  地址（`from other devices: http://192.168.x.x:8080/`）；在另一台设备上打开它，或把 `.../v1` 用作 API
  base URL。Windows 上防火墙会挡住直到你放行：对它弹出的 Python 提示放行（专用网络），或在管理员
  PowerShell 里运行 `New-NetFirewallRule -DisplayName "Strata 8080" -Direction Inbound -Protocol TCP
  -LocalPort 8080 -Action Allow -Profile Private`，并确认网络被设为专用。
- **来自互联网。** 在它前面放一个隧道，例如
  [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)：
  `cloudflared tunnel --url http://127.0.0.1:8080`。**先设一个密钥**，否则任何拿到链接的人都能用你的
  PC：给 `strata-<model>.json` 加 `"api_key": "some-long-secret"`（或设 `STRATA_API_KEY` 环境变量）；
  客户端之后把自己的 API 密钥发成它。多个密钥（每客户端一个，好让其中一个能单独撤销）：用逗号分隔，
  `--api-key key1,key2` 或 `"api_key": "key1,key2"`（像 llama.cpp 那样），或给一个列表，
  `"api_key": ["key1", "key2"]`（使用含逗号的密钥的方式）；请求带其中任一个即通过。流式回答带
  `X-Accel-Buffering: no`，所以 nginx 风格的代理把每个 token 立刻传下去。网页应用的设置和 MCP 工具只
  回答 Strata 自己的页面：当你通过地址不同的代理或隧道打开它时，把那个地址加进去，例如
  `"trusted_origins": ["https://strata.example.com"]`。设了密钥时，任何 `Host` 名字都能到达服务器
  （见下面"主机名"）。
- **来自浏览器里的网页（CORS）。** 默认关。`"cors_origins": ["https://chat.example.com"]` 让那些源
  的页面能从浏览器调用 `/v1/*`（Open WebUI 的直连、浏览器扩展）；`["*"]` 允许任何页面——只有配 API
  密钥才合理。它绝不打开 `/settings`、`/unload` 或 MCP 工具。
- **主机名（DNS 重绑定）。** 另一个网站的页面可以把它自己的名字指向 `127.0.0.1`，然后把这个服务器当作
  它自己的来访问，所以没有 API 密钥时服务器只回答 `Host` 是它认识的名字的请求（有密钥时该检查关闭：那样
  的页面发不出密钥，而传自己名字的隧道和代理照常可用）：`localhost`（及 `*.localhost`）、任何 IP 地址
  （`127.0.0.1`、`[::1]`、`192.168.x.x`...）、它监听的那个地址，以及当它监听范围超出本 PC
  （`0.0.0.0` 或局域网地址）时，本 PC 的名字（`mypc`、`mypc.local`）和 `host.docker.internal`；任意
  端口。其他的得到 **403** 并点名该设置，服务器窗口为每个打印一行。要用另一个名字访问它（保留名字的
  反向代理、隧道、你网络上的 DNS 名、另一个容器为它取的名字）？加上那个名字：
  `strata-<model>.json` 里的 `"allowed_hosts": ["strata.example.com"]` 或
  `STRATA_ALLOWED_HOSTS=strata.example.com`（逗号分隔）；`".example.com"` 允许那个名字及其下每个名字，
  `["*"]` 关闭该检查（设 `api_key` 同样关闭）。`trusted_origins` 里的主机算作允许。没有 `Host` 头的
  请求（HTTP/1.0 客户端）放行。
- **没有 API 密钥的网页。** 没有 `api_key` 时，带 `Origin` 头（浏览器页面发的）发到 `/v1/*` 的
  `POST` 只对以下情况回答：Strata 自己的页面、`localhost` 或允许的主机名上的页面（任意端口）、
  `trusted_origins` 或 `cors_origins` 里的源、浏览器扩展与桌面应用（`chrome-extension://`、
  `moz-extension://`、`app://`：没有网站能发这些），且只带 JSON 体；任何其他页面，以及 `Origin:
  null`，得到 **403**。不发 `Origin` 的客户端（curl、OpenAI 和 Anthropic SDK、其他服务器）不受影响。
  有 API 密钥时由密钥说了算。`POST /unload` 和 `POST /load` 像 `/settings` 一样接受 Strata 自己页面的
  `Content-Type: application/json`（或没有 `Origin`）。`POST /slots/0?action=save|restore` 保留 Host
  与 API 密钥检查，并且也只接受没有 `Origin`、Strata 自己的页面或受信任源的 JSON——设了 API 密钥时同样
  如此。

<a id="from-other-devices-on-your-network"></a>

**对话缓存。** 一次续用活动会话或一个兼容的已保存检查点。活动检查点上限默认为 6。检查点包含递归状态、
PLE 历史和索引器状态。它们的大小取决于模型，不取决于 token 数。引擎为普通的归档历史以及活动检查点列表
设界。它保留必需的前缀锚点和显式的钉。一个检查点必须匹配 prompt 的 token 与图片。较后的递归状态不能替代
较早的检查点。引擎在分支点之前恢复一个保留的锚点并处理剩余的 token。活动列表也在最后一个输入 token 之前
接纳一个锚点。第一个 decode 窗口处理那个 token。一次回复改写可以用这个常驻输入锚点，而不必从过时的已生成
尾巴加载一个较新的状态。有界的检查点归档不会移除一个完整的暂存源。之后的归档压缩因此能丢弃临时状态而不
移除原始续用的续读状态。已保存输入内部的一个分支也会冻结它的源归档。新的归档可以借用它的 K/V，但不会轮换
源的最初续读状态。

L2 分支共享它们未改动的 K/V 分配。一次追加或改写只为新的或改动的区间分配。缓存把每个共享的负载分配只计费
一次。一个共享长对话前缀的短请求不会移除那个长续用。一个系统 prompt 检查点也可以为分开的聊天提供一个公共
根。它的阈值由 `--prompt-cache-root` 设置。Claude Code 给它的系统 prompt 盖一个每次请求都变（`cch=...`）
且每个会话都变（`cc_version=` 的第 4 段）的计费头；服务器把两种戳都钉住（改成 `f`，像 llama.cpp 那样），
所以系统 prompt 和它前面的工具列表在每一轮都是同一个 prompt。引擎选项：`--prompt-cache N`（0 = 关）、
`--prompt-cache-every N`、`--prompt-cache-root N`（0 = 不做系统 prompt 检查点）、`--turn-token ID`。
一个之后没有请求续用的一次性请求（一次分类调用、一个探测）可以在其请求体里发 `"strata_checkpoint":
false`：它不保存其最后一个轮的检查点、也不每 16K token 保存一次，所以复用 prefix（或根）之后的东西一次
读完，它的会话也不为下一个请求保留或暂存。它仍从匹配上的检查点开始，且系统 prompt 根到达
`--prompt-cache-root` 时仍会保存它。不带该字段（或带 `true`）时什么都不变。

<a id="conversation-cache"></a>

- **回答-only 历史留在 GPU 内存里。** 客户端可以省略上一轮回复的 thinking。此时已生成的 GPU
  状态与下一个 prompt 不同。那个状态不能被当作 thinking 不存在一样复用。

  一次完整回复之后，串行服务器用一次免-decode 的空闲 prefill 准备好精确的回答-only 历史。它包含已完
  成的工具调用和原始图像 embedding。它不在下一个 prompt 里存隐藏的 thinking，也不生成额外的 token。
  随后的请求只有在其 token、图像与控制向量模式都匹配时才复用准备好的 prefix。服务器用与请求相同的规则
  编码 prefix，所以消息内部的一个控制 token 保持是文本。

  一个请求在等待引擎之前取消空闲 prefill。引擎在一个安全的 prompt 读取边界停下并在接受请求前排空命令。
  有一个例外。当 prefill 正在读那个请求自己 prompt 的开头时，请求改为等那次读取完成，然后复用已读的
  部分并处理更少的 token。为另一个对话做的 prefill 总是让路。

  一个文本请求在加载一个已经加载的引擎、或准备它的 prompt 时不取消空闲 prefill。这些步骤不需要引擎队列。
  图像编码、引擎加载，以及与请求不匹配的 prefill，在使用引擎队列之前仍会取消空闲 prefill。

  重建仍使用 GPU 时间。如果下一个请求在它完成之前到达，或改变了历史或 prompt 设置，一部分 prompt 处理
  留在请求路径上。最后一个会话留在 GPU 内存里；L2 与 L3 暂存是给冲突的会话用的，不是给每个完成的轮次。

  引擎支持这条路径时报告 `INFO prefill_only=1`。`GEN 0 <ids>` 和 `GENI 0 <embeddings> <ids>` 处理除
  最后一个 prompt token 以外的全部、不发出任何生成 token、以生成数 0 返回 `DONE`。被推迟的那个 token
  由下一个生成请求处理。

<a id="answer-only-history-in-gpu-memory"></a>

**检查点从结尾向前铺开（需显式开启，`--prompt-cache-spread`）。** 按最久未使用保留最新的点：每个轮在
它的开头加一个检查点，所以几轮短对话之后那 6 个点落在最后 1-2K token 之内。一个在几千 token 之前重写
它历史的 agent 客户端（修剪一个旧工具输出、丢掉一个步骤）就一个都匹配不上，只能从系统 prompt 续读——整个
对话重读一遍。带该参数时，离开的那个检查点是"移除它之后、按其与新端的距离留下的缺口最小"的那个，所以保留
的点按几何间隔稀疏化：200 轮每轮 1,000 token 之后，它们落在结尾前 0、1K、3K、8K、43K 和 200K token 处，
往回 d token 的一次编辑大约再读 d（往回 6K：读 8K 而不是 200K；往回 42K：读 43K）。检查点数量相同、不多占
RAM；系统 prompt 根、钉住的 prefix 和最新的点从不丢弃，续读仍是通常的精确 prefix 匹配。在 2x CMP 50HX
（层级切分，IQ3_XXS）上用 pi 编码 agent 实测：之前，一个与会话有 204,349 个 token 相同的 210,528-token
prompt 在 9,525 处续读、又把 201,003 个读了一遍（100 秒）。带该参数，一段逐轮增长的 82,593-token 对话、
往回约 6 步的一个轮被改写，在 73,610 个 token 处续读（10.2 秒，对比整读 40.0 秒），且固定 expert 放置
（`--adapt-swaps 0`、固定 `--pcie-frac`）后贪心回答与整读的文本相同。不带该参数时什么都不变。

<a id="checkpoints-spread-back-from-the-end"></a>

**一个长文档、许多问题：钉住的共享 prefix（需显式开启，0.1.40.2，
[RESEARCH_RUNS.md](RESEARCH_RUNS.md)）。** 一个请求可以说明文档到哪里结束：
`"strata_prefix": {"messages": 1}`（第一条消息是文档）、`{"message": 0, "chars": 210000}`（消息 0 的
前 210,000 个字符，给把文档和问题放在一条消息里发的客户端）或 `{"tokens": 12345}`。引擎在那里分两部分
读 prompt、把边界上的检查点钉住（保留策略永不逐出它、持有它的暂存对话保持暂存、`SAVE` 保留它），并且当
后面某个问题从它续读时，不暂存它离场的那个问题。于是下一个问题从文档结尾开始，20 个问题只付一次文档的
读取。不带该字段时服务器的检查点还在老地方：最后一个轮的开头、系统 prompt 的结尾、以及每
`--prompt-cache-every` 个 token；所以同一聊天里跟在长文档后面的问题会重读其中至多 16K token（Tesla
P100 在 262K 上是 55 秒）。该字段对照 prompt 自己的 id 校验（prefix 永远是最长的公共开头），无法标记的
prefix 在服务器窗口里说明并忽略，畸形的字段是 400。每个引擎一次只钉一个 prefix（新的取代旧的）；需要
`--prompt-cache 3` 或以上。`tools/research_run.py` 对着服务器带与不带它跑一个文档加问题清单。引擎自己的
键是 `GEN` 行上的 `pin=N`。

<a id="one-long-document-many-questions"></a>

**多对话（需显式开启）。** 在引擎参数里加 `--conversation-cache-mib 8192
--conversation-cache-slots 4`，把至多四个对话暂存进有界的 8 GiB host-RAM 缓存。这在其请求交替时保住
controller/worker 的历史；它不同时执行请求。不需要客户端会话 ID：只有精确的 token/图片 prefix 且 steering
模式匹配才复用。默认预算是 0（禁用）；`--prompt-cache 0` 也禁用暂存。带 `--layer-split` 时，一个暂存的
对话每分页持一张图片（每个后续分页的会话及其那份对话检查点），任何覆写之前每张图都校验，各自在自己的 GPU
上恢复。在 4 块 GPU 的切分上验证过：一个 9,276-token 对话 51 毫秒恢复完，它的追问解出的 token 与对话从未
离场时完全一致（`tools/parking_test.py`）。支持 FP16、INT8、Q4_0 和 identity-layout K8V4 快照；K8V4
草稿环保持 INT8，与 upstream 相同。Windows/HIP 与多 GPU 的运行时覆盖情况必须与 Linux/CUDA 的证据分开报告。

<a id="multiple-conversations-opt-in"></a>

网页的 Monitor 标签页有一张 **RAM 层（L2）** 卡片。`/api-monitor` 页面也显示 L2 RAM 与 L3 磁盘占用。L2
显示包括暂存的对话、槽上限、RAM 用量、字节预算、暂存、恢复、逐出，以及 prompt token 复用。根暂存计入这些
计数。`/metrics` 在 `"conversation_cache"` 和 `"l3"` 里提供同样的数据。

<a id="the-web-pages-monitor-tab-has"></a>

快照包含运行状态、检查点、已用的 K/V 页和草稿层 K/V。它们增加 host RAM，不是另一个模型或另一块 VRAM
分配。字节预算在一次切换期间也计入进来的快照。恢复之后，未改动的 K/V 页可以为下一次暂存操作保留；增长追加
存储而不拷贝已有的页，为至多每缓冲区八分之一（且至多 16 MiB）的后续轮次留空间，预算把它像快照其余部分一样
计入。回卷刷新受影响的页，运行状态和检查点重新捕获。保留的活动 K/V 计入同一字节预算，并在内存压力下逐出暂存
条目之前先丢弃它。最旧的暂存条目最先逐出。一个以暂存对话内部的检查点开头的新对话（共享系统 prompt 和工具
列表的 subagent、压缩过的历史）恢复那个 prefix 并把暂存的对话原地留下，所以它的下一轮仍能完整续读；引擎日志
说 "borrowed"。对上暂存对话的活动状态或它最新检查点的匹配，就是那个对话继续进行、按旧例把它移进会话；到达钉住
的共享 prefix（`pin=N`）的匹配也一样，并保持一个对话持有那个钉。离场对话只装得下暂存对话那份空间时，它保住
位置、暂存的那个按旧例整体移入。超大的快照或 host 分配失败退回普通 prompt 处理。
`--conversation-cache-min-tokens N`（默认 0 = 暂存每个对话）只暂存至少 N 个 token 的对话。一个暂存的对话
无论长短都占一个槽，而暂存是切换时刻的决定，所以把短侧请求与一个长对话交错发的客户端会把缓存花在侧请求上，
且一旦满了就按最旧优先逐出长的那个——恰恰是它想要的复用。把最小值设在侧请求长度之上就让长对话保持暂存。检查在
`make_room()` 之前跑，所以被跳过的暂存不逐出任何东西也不存任何东西；日志行是 `conversation cache: skip
parking (N tokens, below the M minimum; parked=P)`。0 是该参数之前的行为，该参数也不动回答——它只决定哪些
对话保留槽。`--conversation-cache-park-gain N`（需显式
开启，默认 0 = 每轮都暂存）只在一个对话比它要取代的那份副本多前进 N 个 token 之后才再次暂存它；只增长了几百
个 token 的一轮保留较旧的副本，而不是写一份下一轮就丢弃的完整快照。代价是从较旧副本续读时最多重读 N 个 token
（#2265）。`--conversation-cache-min-free-mib N`（默认 2560）额外要求留有那么多 RAM 余量：引擎在分配之前
检查（对照快照的估计大小，减去它已保留的 K/V）并在捕获之后再次检查。未知的遥测或 RAM 不足会跳过暂存，同一数字
与下限也决定会话文件能否保存或恢复（下文）。8 GiB 预算是一个上限，不是对每台机器的推荐。槽上限与字节预算同时
生效。例如 `--conversation-cache-mib 15360 --conversation-cache-slots 32` 允许在 15 GiB RAM 预算内至多
32 个暂存条目。引擎不在启动时分配整个预算。根图片也用槽和 RAM。字节上限未到就先发生槽逐出时，提高槽上限。保持
物理 RAM 余量检查开着。

<a id="snapshots-contain-running-state"></a>

**RAM 数字（`available_host_bytes()`）。** 余量是三个数字中最小的一个，每次被问时采样，且不是一项预留
（其他写入者之后可以把空间拿走）：

1. `MemAvailable`（Windows：`GlobalMemoryStatusEx` 的可用物理内存）。
2. 引擎自己 cgroup 的每个内存限制之下、以及 `/sys/fs/cgroup` 下可见的每个祖先 cgroup 之下的空间（命名于
   `/proc/self/cgroup` 的组，或该路径在本挂载能看到的最深一段）：cgroup v2 的 `memory.max` 和
   `memory.high`，各自减 `memory.current`；cgroup v1 的 `memory.limit_in_bytes` 减
   `memory.usage_in_bytes`（两者都再减去干净的非活跃文件缓存，见下）。`max` 和 v1 的 "unlimited" 算作无
   限制；文件缺失（该层没有 memory controller）跳过；文件存在但读不出或解析不了会让 cgroup 各项不可用：没有
   `--memory-limit-mib` 时防护改用单独的 `MemAvailable`，如同 0.1.41，并每进程打印一条警告；带该参数时样本
   为未知，未知则跳过暂存。
3. `--memory-limit-mib N`（或 `STRATA_MEMORY_LIMIT_MIB=N`；参数优先，`0` = 无）：N MiB 是本引擎容器或
   cgroup 总共可以用的量，减去那个容器现在的用量。容器看不见自己的限制时用它。用量取本挂载显示的最顶层 cgroup
   的 `memory.current`（在自己有 cgroup namespace 的容器里就是容器自己的）；那个文件读不出时取
   `MemTotal - MemAvailable`，那是较弱的数字（只和容器的 `/proc/meminfo` 一样真），`MemTotal` 也缺时样本
   为未知。这个上限在第 1、2 项之外额外生效、从不取代它们：最小的那个数说了算。它是引擎参数，所以服务器配置把它
   带在 `args` 里（`"--memory-limit-mib", "102400"`）；该变量从配置的 `"env"` 设置也有效。

用量按内核报告的 `memory.current`，**减去一类可回收的：干净的非活跃文件缓存**。那是组自己的 `memory.stat`
里的 `inactive_file`（cgroup v1：`total_inactive_file`），当 `memory.stat` 列出时再减去它的脏页与回写页
（`file_dirty`、`file_writeback`；v1 的 `total_dirty`、`total_writeback`），且从不超过 `memory.current`。
内核在 `memory.high` 或硬限制处先把这类页交还、然后才杀东西，而加载模型的文件读取会留下很多。其他一概不计：
`active_file`（它持有 mlocked 的页）、`shmem`（钉住的 expert 补集、KV 池）和匿名内存全额计费，因为硬限制上
的一次击杀比一次跳过的暂存更糟。对每个限制，空间是 `limit - (memory.current - credit)`，所以它从不超过限制。
缺失或解析不了的 `memory.stat` 不给信用；它也不让样本变未知。操作者的上限使用它所读 `memory.current` 的同一
顶层组的 `memory.stat`。捕获之后一个暂存的对话就是引擎自己用量的一部分；捕获之前的检查就是阻止它长过上限的那道。

<a id="the-ram-figure-available_host_bytes"></a>

启动日志为它有一行：`strata generate: memory guard: limit X GiB (source: flag|cgroup|meminfo),
current Y GiB, reclaimable cache credited Z GiB; available ...`，其中 `source` 点名那一刻最小的那个数的
来源（`meminfo`：MemTotal 与 `MemTotal - MemAvailable`），一次跳过的暂存或被拒绝的保存/恢复会说出它当时
有多少、来自哪个来源。这行在模型加载之前打印，所以它的信用是那一刻的缓存，不是某次暂存时刻的。没有 cgroup
限制且没有 `--memory-limit-mib` 时，数字就是单纯的 `MemAvailable`，与之前相同。

它不碰的东西：expert 区的启动定大小、文件层的常驻预算、Windows 提交检查以及 `STRATA_PIN_GUARD`（#1250：是否
按步骤给 expert 副本做页锁定跟随 `MemFree`，每一步对照带 cgroup 感知的 `host_available_memory` 检查）保留它们
自己的探测（`--resident-budget-gib`、#633、#1250），`--memory-limit-mib` 管不到它们。这个数字服务于启动之后的
事情：暂存一个对话、以及保存或恢复会话文件，它们过去只读 `MemAvailable`。`host_available_memory`
（expert_source.cpp）和这个数字都读 cgroup 限制；这个数字还读 `memory.high` 和操作者的上限；读不了的 cgroup
文件让它退回 `MemAvailable`（一条警告），设了上限时变未知。

<a id="the-startup-log-has-one-line"></a>

在一台引擎容器看不见其限制的 Proxmox LXC 上的早先测量（父 cgroup 上 100 GiB 的上限，在容器的 namespace 之外；
容器自己的 `memory.max` 和 `memory.high` 读 `max`）：`/proc/meminfo` 显示约 25 GiB `MemAvailable` 而实际只剩
约 1 GiB；就在引擎加载之后容器的 `memory.current` 是 98.29 GiB，`anon` 1.29 GiB、`shmem` 65.51 GiB、
`inactive_file` 3.81 GiB（读模型文件留下的干净页缓存）、`active_file` 0.01 GiB，其中 3.8 GiB 干净缓存正是信用
交还的部分。那里 `--memory-limit-mib 102400` 是上限，而底线（`--conversation-cache-min-free-mib`）按其下的剩余
来量：引擎已逼近上限时，几个 GiB 的底线拒绝每一次暂存；几百 MiB 给分配器和内核留一点边距，同时让真正装得下的快照
通过。

<a id="earlier-measurement-on-a-proxmox-lxc"></a>

在测试机上实测（2x RTX 3080 20 GB，UD-Q4_K_XL，90 GiB 容器，常驻 RAM 模式，2 个槽），每臂一次重启，
`--conversation-cache-min-free-mib 2048`。带 `--memory-limit-mib 101376`（我们的部署所用的上限）引擎每次尝试
都暂存：一轮 308 个请求里 135 次暂存、0 次跳过。带 `--memory-limit-mib 78000` 时，启动行读取的是
`limit 76.2 GiB (source: flag) ... available 73.0 GiB (MemAvailable 88.6 GiB)`，而 8 次暂存尝试全部跳过
（`skip parking
(physical RAM admission; need 226 MiB plus 2048 MiB floor, 0 MiB available, source flag)`）、没有暂存。容器
的 `memory.current` 在 89.7-89.9 GiB、高于 76.2 GiB 的上限，所以上限之下的余量是 0，而同一批样本里
`MemAvailable` 读 15.7-15.9 GiB；只查 `MemAvailable` 会让那些暂存放行（旧检查没跑）。请求照常服务（6 次
decode 的中位数 79.6 tok/s）；只有暂存被拒绝，日志这么写。单元测试 `memory_guard_test`（仅 CPU、无模型）通过
130 项检查。未测：78000 与 101376 之间的上限、不带该参数的运行、对吞吐的任何影响。

<a id="measured-on-the-test-machine-x"></a>

共享快照核心在应用任何状态之前校验所有层和检查点。无效条目被丢弃；传输/同步失败是致命的，而不是带着部分状态继续
的许可。索引器的备用键与移动的备用行被保住，包括检查点回卷。引擎日志报告暂存、恢复、字节、逐出、单个快照大小以及
捕获期间复用的 K/V 字节。`STRATA_SNAPSHOT_FULL_CAPTURE=1` 关闭保留以便诊断对比。RAM 层快照不跨重启持久。

<a id="the-shared-snapshot-core-validates"></a>

**磁盘上的对话缓存（需显式开启，L3）。** 对话状态有三层。第 1 层（L1）是 GPU 内存里的活动会话；它总是存在。
第 2 层（L2）是上面可选的 host-RAM 缓存（`--conversation-cache-mib`）。第 3 层（L3）是下面可选的磁盘存储。
L2 和 L3 是可选的。L1 总是存在。

<a id="conversation-cache-on-disk-opt-in-l3"></a>

在引擎参数里加 `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25`。引擎
随后把暂存的对话作为文件保存在 `strata-conversations` 目录里，重启后的服务器能恢复它们。路径与正的 GiB 预算
必须一起给。存储默认关、需要 `--serve`，并且能与 `--conversation-cache-mib 0` 一起工作。磁盘和 RAM 缓存都
支持 `--layer-split`。目录缺失时引擎创建它。创建不了的存储以错误停止启动。

<a id="add-conversation-cache-disk-strata-conversations"></a>

引擎先把离场的对话暂存进 L2。成功的 L2 接纳不往 L3 写副本。若 L2 字节预算、槽上限或物理 RAM 底线要求逐出，
引擎把选中的快照移到 L3。外溢使用已有的 host 缓冲区。它不拷贝负载、也不再从 GPU 读快照。每次外溢创建一条完整
的磁盘记录。L2 被禁用或无法接纳一个对话时，引擎直接把它暂存进 L3。同样的规则适用于共享的根 prefix。仅 L2 的
条目不能跨服务重启存活。重启后只有已经在 L3 的条目能恢复。

<a id="the-engine-first-parks-an-outgoing"></a>

两个可选参数微调该存储。`--conversation-cache-disk-slots N` 给记录数设上限（默认 0 = 无上限）。
`--conversation-cache-disk-min-free-mib N` 在该存储的文件系统上保留那么多 MiB 空闲（默认 0 = 无底线）。两者
都需要路径与 GiB 预算。

<a id="two-optional-flags-tune-the-store"></a>

还有两个参数钉住共享的系统 prompt。`--conversation-cache-keep-root`（默认开）把系统 prompt prefix 图片钉在
RAM 缓存里。`--conversation-cache-disk-keep-root`（默认开）给该 prefix 一条自己的磁盘 'p' 级记录。它们的
`--no-` 变体分别关掉各自的钉。

<a id="two-more-flags-pin-the-shared"></a>

一个请求只在它的 prompt 以记录精确的 token 与图片开头、且控制向量模式匹配时复用记录。存储保留最长匹配的 prefix；
等长平局偏向较小的记录：系统 prompt prefix 优先于包含它的对话记录。跨活动会话、RAM 缓存、磁盘层和空闲批处理槽，
最长的有效 prefix 胜出。磁盘记录只在它的 prefix 严格长于每个可用的常驻 prefix 时才用。等长的 RAM 匹配用 L2
而不是 L3。等长的活动或空闲槽匹配保持常驻。L3 仍可以供应一个更长的 prefix，或在 L2 逐出之后恢复一个对话。不需要
客户端会话 ID。

<a id="a-request-reuses-a-record-only"></a>

一个经过校验的磁盘根被恢复时，引擎可以把它现有的 host 镜像保留在 L2。RAM 槽上限、字节预算、物理 RAM 底线和根设置
依然适用。引擎不再从 GPU 重新捕获根。之后的 RAM 恢复让钉住的根对另一条聊天保持可用。只有 token 与图片的身份元数据
被拷贝进活动会话。缓存的 K/V 缓冲区与活动复用缓冲区保持分开。

<a id="when-a-validated-disk-root-is-restored"></a>

直接的 L3 回退在 GPU 会话改变之前捕获离场状态。它的磁盘写可以在后台跑。一次查找、一次存储变更或
另一次暂存在访问存储之前等那个写入者。L2 逐出的外溢在受害缓冲区被释放之前完成。一个已带完整有效常驻
prefix 的续用，不会等一次不可能改进复用的磁盘查找。

<a id="a-direct-l3-fallback-captures"></a>

**共享的系统 prompt（捕获一次）。** 一条新聊天的 prompt 从 token 0 读取时，若系统 prompt 的结尾
（第一个轮边界）有 `--prompt-cache-root` 个 token 或以上，引擎就在那里做检查点。引擎把那个 prefix
在 L2 捕获一次，L2 不可用时在 L3。它把 prefix 存成系统 prompt（根）图片。根比共享它的对话活得更久：
RAM 缓存和磁盘 LRU 只在所有未钉的记录之后才逐出它，等长匹配偏向根，因为恢复它只读共享 prefix，而不是
包含它的对话记录。一份系统 prompt 的字节只写一次，不是每条聊天一次。选项：
`--conversation-cache-keep-root` 和 `--conversation-cache-disk-keep-root`（默认都开）。

<a id="the-shared-system-prompt-captured-once"></a>

RAM 缓存先移除普通对话、后移除根图片。只剩根时，它移除最旧的根来在字节预算或槽上限内腾出空间。旧根不能
占满所有槽而阻止一个新对话进入 L2。显式的共享 prefix 检查点钉在这类逐出中受保护。物理 RAM 接纳检查和
空闲内存底线保持有效。

<a id="the-ram-cache-removes-ordinary-conversations"></a>

一条记录持有一个对话、以及每个 `--layer-split` 分页一张图片。两块卡的运行作为一个整体暂存与恢复，且每个
分页必须持有同样的 token 与图片链。记录带一个兼容性键，覆盖引擎版本、模型与 MTP 来源、几何、KV 格式与
旋转、上下文、rope 配置、每分页的层切分与设备、以及控制向量模式。存储只被同键的运行复用。

<a id="one-record-holds-one-conversation"></a>

文件带校验和与版本。每个文件有固定头部，含格式版本、元数据校验和、负载校验和。引擎把记录写进临时文件并重命名
覆盖目标文件（原子替换）。启动时引擎扫描目录、重建索引，并移除未通过校验的文件，包括来自另一个版本或另一个兼容性
键的文件。它也移除残留的临时文件。启动行报告恢复的记录数与字节总数。

<a id="the-files-are-checksummed-and-versioned"></a>

存储是有界的。字节预算或记录上限超出时、以及文件系统空闲空间会掉到空闲底线之下时，它逐出最久未使用的记录。比
整个预算还大的记录被拒绝。

<a id="the-store-is-bounded"></a>

引擎在恢复之前校验整条记录。它先检查所有分页在 token 与图片链上一致，然后对照当前几何校验每个分页。这个校验
通过之前它不做任何 CUDA 写入。损坏、不兼容或无效的记录被移除，请求退回普通 prompt 处理。校验之后，传输或
同步失败是致命的；引擎绝不从部分恢复的会话解码。

<a id="the-engine-validates-the-whole-record"></a>

GPU 捕获与恢复是同步的。引擎一次捕获或恢复一个分页，并同步每个分页的设备。捕获之后，文件写在下一个请求使用
GPU 的同时异步进行。磁盘查找或之后的暂存在访问存储索引之前等那个写入完成。保存或加载期间记录在 host RAM
里暂存，所以操作需要约记录大小的临时 host 内存。文件操作之后引擎释放那块内存。恢复保持在服务线程上。

<a id="gpu-capture-and-restore-are-synchronous"></a>

磁盘层让暂存的对话跨交替、跨重启存活。它本身不增加并发执行。设 `"parallel": N` 启用批处理槽；其内存代价与
限制见 [BATCHING.md](BATCHING.md)。

<a id="the-disk-tier-preserves-parked-conversations"></a>

**磁盘寿命与定大小。** 一个对话用一条 `.conversation` 链存 token 元数据和 K/V 面板，用一个 `.states`
侧车存保留的运行状态。一次兼容的追加只写 token 与 K/V 的增量。它可能重复一个边界页和移动的索引器行。新面板不
重复完整的检查点 token 列表、也不内联运行状态。若最长 token prefix 的 K/V 字节不同，写入者检查更早的输入与
检查点边界。回复单元的差值不得迫使未改动的输入 prefix 再存一份。重放也可能改变最后的输入单元。 saved 输入边界
之前的一个替换必须证明其余原始输入 token 与图片键未变。一次输入编辑保留原始分支。

<a id="disk-endurance-and-sizing"></a>

侧车为普通锚点有固定的状态槽。新状态取代保留策略要移除的锚点。显式的钉可能需要额外槽。普通轮次不是每轮加一个永久的
递归状态负载。状态发布失败时缓存写入失败；它不使用无限制的内联状态回退。当后续输入消费了一个已存的生成结尾，侧车可以把
那个原始续读状态保留在一个普通槽里。密集输入检查点不会把它轮换掉。标记和状态熬过重启。这不会每轮增加一个状态槽。

<a id="the-sidecar-has-fixed-state-slots"></a>

一次种子用临时链文件加重命名。追加写一个新面板并更新链头。读取者检查面板校验和，并把撕裂的尾巴截断到最后一个完整面板。
一个状态槽更新先写它的负载、再写它的槽头。无效的槽不能提供恢复状态。撕裂的侧车头会让它的全部锚点不可用。引擎于是需要
另一个有效锚点或一次 prompt 读取。

<a id="a-seed-uses-a-temporary-chain-file"></a>

带内联状态的版本 2 文件保持可读。加载器在分配它们的负载之前先选保留的状态。它校验被丢弃的负载字节而不保留它们。遗留的
内联状态留在磁盘上，直到其链被逐出或替换。

<a id="version-2-files-with-inline-states"></a>

引擎在选择一个已存对话之前检查活动的 GPU prefix。一次磁盘 restore 必须提供比所选常驻状态更长的、兼容的状态锚点。
一个活动续用不需要对话缓存传输。在一次回卷覆写另一个续用之前，引擎保住它合格的检查点。这对钉住 prefix 的请求同样
适用。一个钉不授权引擎丢弃长对话的其余部分。L3 开着时 L2 可以保持禁用。

<a id="the-engine-checks-the-live-gpu-prefix"></a>

Decode 期间，引擎在每块所属 GPU 上保留最近一个 decode 窗口之前状态的一个检查点。响应结束时，它把这个状态拷进一个
host 检查点。这个检查点包含递归状态、PLE 历史和索引器状态；它不包含 K/V 单元的拷贝。如果下一个 prompt 在回复接近
结尾处不同，引擎可以在那个最后窗口之前、而不是整个回复之前续读。历史更早处的不一致仍需要更早的兼容检查点。引擎在
end-of-sequence token 或请求的输出上限处限制投机提交，所以保存的活动历史不含超过该上限的 token。

<a id="during-decode-the-engine-keeps"></a>

兼容的活动续用不需要对话缓存传输。这不意味着盘是空闲的：模型的 PLE 表有自己的磁盘读路径。在 prompt 日志里，"read"
计的是处理的 token 数，不是从对话缓存读的字节。对话磁盘的读取与恢复消息标识 L3 加载。一次对话切换仍可能要几秒加载
一条大记录。

<a id="no-conversation-cache-transfer-is-required"></a>

新的 K/V 存储随改动的后缀、KV 格式和切分分页数量增长。保留的运行状态存储单独设界。字节预算同时包含链和它的侧车。缓存
大小与写入代价取决于模型和工作负载；一个字节预算不意味着固定的对话数。Flash 寿命仍取决于写入的字节。清空缓存目录之前
先停止服务器。引擎在其下次启动时建一个空索引。

<a id="new-k-v-storage-grows-with"></a>

存储把共享 K/V 支撑与递归状态恢复锚点分开选择。一条链即使在那个精确位置没有运行状态，也能借出一个较短的 prefix。
引擎在借用之前校验保留的 K/V 字节和已完成的索引器行。相同的 token id 本身不是相同 K/V 负载的证明。重放的边界单元
可能要求更短的共享范围；改动的后缀于是从那个经过校验的边界开始。

<a id="the-store-selects-shared-k-v-backing"></a>

一个分支引用已有链未改动的 prefix 并存它自己的后缀。它不截断源链的续用。一个引用可以指向一个面板内部的 prefix 或
另一个分支。存储检查内容、图片键、分页布局和基础依赖。它拒绝会使一个依赖分支悬空的回卷或替换。有限的缓存预算仍可能逐出
一条链及其依赖者。

<a id="a-branch-references-the-unchanged"></a>

引擎把输入 prompt 边界与生成的回复分开记录。它可以在那个边界之后替换过时的回复尾巴。这个操作不新增一个对话、也不是每轮
一个永久状态文件。那个边界之前的分支保留源续用。钉和依赖的 prefix 引用仍阻止破坏性的替换。已存的输入边界未知时，引擎
不从附近的检查点推断它。

<a id="the-engine-records-the-input-prompt-boundary"></a>

一次 prefix 加载跳过对请求范围不可能有贡献的面板。它完整读取包含切点的那个面板以校验其校验和。因此，从一个大型遗留
面板做的一次短加载仍可能需要一次大的磁盘读。一个 token 位置匹配但没有有效运行状态，不是恢复点。每条链和它的侧车保持
在一起。

<a id="a-prefix-load-skips-panels"></a>

旧的重复记录不会被后台去重操作移除。当一次保存需要在字节预算、记录限制或空闲限制内腾空间时，存储按最久未使用逐出对话
记录。系统 prompt 记录有自己的逐出优先级。所以旧的未用副本可能在之后的保存中被移除，但空闲的缓存不会缩小。这个修复不
压缩已有链内部的过时负载。不需要手动重置缓存。要立刻收回全部缓存空间，清空其缓存目录之前先停止服务；这会丢弃保存的对话，
让之后的请求重建它们的状态。

<a id="older-duplicate-records-are-not-removed"></a>

预算按你想跨重启保住的对话数量大小，而不是按一个对话反复的完整快照。25 GiB 预算是个例子：它在长上下文下装几个大对话，
或许多小对话。设 `--conversation-cache-disk-min-free-mib` 保护文件系统的空闲空间，想要硬性链数时设
`--conversation-cache-disk-slots`。引擎日志报告每次暂存与恢复的字节大小和时间，以及命中、未命中、逐出和损坏计数。
读它来为你的工作负载定预算。为诊断对比关闭保留。暂存的快照不跨重启持久；下面的会话文件持久。

<a id="size-the-budget-for-the-number"></a>

**会话文件（磁盘）。** 引擎持有的对话可以存进文件、之后恢复，也在同一引擎版本重启之后恢复，这样长 prompt 不必
重读。服务器带 `--slot-save-path DIR` 启动时（配置里也有 `"slot_save_path"`），为它的单槽 0 暴露 llama-server
槽 API 的保存与恢复请求；NAME 必须是 DIR 内的普通文件名。没有擦除动作，文件格式是 Strata 自己的、不是 llama.cpp 的：

<a id="session-files-disk"></a>

```bash
curl -X POST "http://127.0.0.1:8080/slots/0?action=save"    -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_saved": 63025, "n_written": 1198691396, "timings": {"save_ms": 709.5}}
curl -X POST "http://127.0.0.1:8080/slots/0?action=restore" -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_restored": 63025, "n_read": 1198691396, "timings": {"restore_ms": 898.2}}
```

DIR 在启动时成为一个绝对路径（`--slot-save-path` 相对服务器工作目录，配置的 `slot_save_path` 相对配置的
`"cwd"`），缺失时以 0700 模式创建。它应当对运行 Strata 的用户私密：文件持有对话的 token id 和状态，在 Linux 上以
0600 模式创建（Windows 上继承文件夹的权限），且没有任何东西删除它们——约每 63K-token 对话 1.2 GB。写之前，保存检查
磁盘装得下整个新文件加 `--session-min-free-mib`（一个引擎参数，在配置的 `args` 里；默认 4096，0 = 不检查）——替换
一个文件时也一样，那个文件的空间只在重命名之后才回来。这是一次预检，不是配额或预留。NAME 不得包含路径、驱动器、流
（`:`）、Windows 设备名（`NUL`、`CON.bin`、`COM1`...）、控制字符、开头的点，或结尾的点或空格。

<a id="dir-becomes-one-absolute-path"></a>

`--session-save-reclaim`（配置 `args` 里的引擎参数，默认关）让一次 RAM 预检失败的 SAVE 先释放可重建的 host
缓存：保留的 K/V 缓冲区、最旧的未钉暂存对话、然后除为该文件选中的那个之外的未钉检查点。测得的可用 RAM 一达到原始分配
估计加 `--conversation-cache-min-free-mib` 就停止；它不降低那个底线，也不假设已释放的分配到达了操作系统。钉住的
prefix 和被选中的检查点保持完好，包括它们的顺序和同分偏好。未知的 RAM 遥测或未知的分配估计不逐出任何东西。 usual 的
接纳检查仍在拷贝状态之前跑。

<a id="session-save-reclaim-an-engine-argument"></a>

带这个选项时，HTTP 服务器还在缓存回收之后给物理 RAM 计账最多 30 秒追平。只重试 SAVE RAM 预检发出的未发布 `memory`
拒绝，至多三次，条件是测得可用 RAM 达到报告的估计加原始底线和 2 MiB 舍入。全程保持同一个活动引擎进程与服务 FIFO；进程
改变、死亡、未知遥测或缺失估计都结束等待。这不卸载模型、也不裁剪另一个进程的工作集。持续的压力仍返回最后一次拒绝。等待在
引擎报告的 `save_ms` 之外。

<a id="with-this-option-the-http-server"></a>

这可以拿之后的 prefix 缓存命中换保存的空间。即便之后的接纳或文件 I/O 失败，已释放的缓存保持释放；当前 token、图片、
steering 和活动设备状态不变。它不能保证任意内存压力下 SAVE 成功。选项关着、或首次探测 RAM 足够时，缓存保留与会话文件格式
不变。RESTORE 不变。

<a id="this-can-trade-later-prefix-cache-hits"></a>

请求必须 `Content-Type: application/json`（否则 `415`），并且来自没有浏览器页面、Strata 自己或受信任的源
（另一个站的 `Origin` 得 `403`，设了 API 密钥也一样）；Host 与 API 密钥检查与各处相同。错误：不带
`--slot-save-path` 或有并行请求时 `501`；槽不是 0、未知的动作、被拒绝的文件名、或引擎以无效拒绝的文件（不是会话文件、
损坏、另一个模型或配置、超过本会话的限制；会话原样未动）时 `400`；恢复缺失的文件时 `404`；模型未加载、读文件所需 RAM
不在、或分配失败时 `503`；磁盘没有空间时（空闲保留，或操作系统报告无空间/配额）`507`；任何其他 I/O 失败（权限、读、写、
flush、重命名）以及引擎结束时（一次恢复的传输失败、下面那条、`engine_silence_s` 内什么都没说的引擎、或一个不按协议应答的
引擎）`500`——下一个请求再把它启动。状态码跟着引擎的类别（`error.kind`：`invalid`、`memory`、`storage`、`io`），
从不跟消息的文字。新文件替换旧文件之后才失败的保存用 `error.published: true` 说明（下文）。保存或恢复等待运行中的请求
（同一个队列），显示在 `/status` 里并计为空闲卸载意义上的活动。之后一个其消息续上已恢复对话的请求复用恢复的状态或它的
检查点；可能重读一小段尾巴（下面的测量里是 35 个 token）。只有最深的检查点被保存，所以更早的编辑会重读更多。客户端仍然
发送它们的消息（和图片）：文件持有引擎状态和 token/图片身份，不是聊天导出。底层，`strata --serve` 在请求之间于 stdin
接受 `SAVE <path>` 和 `RESTORE <path>`，回答 `SAVED <tokens> <bytes> <ms>`、`RESTORED <tokens> <bytes> <ms>`、
`SERR <kind> <published 0|1> <reason>`（失败，引擎和会话原样未动）或 `FATAL <reason>`（随后退出）。途中，
`SESSION <done> <total>` 跟着文件移动的每一个块（至多 16 MiB，最后的残块和小文件也算），一个在其调用内阻塞的步骤
（`fingerprint`、`capture`、`flush`、`publish`、`validate`、`transfer`）之前先来 `SWAIT <phase> <seconds>`：
引擎的看门狗和服务器都给那个步骤允许那些秒数——60 加它所涉每 4 MiB 一秒，至多 3600——之后它算卡住。下一行清掉那份允许。
服务器检查每一行：畸形或未知的行、为负或倒退的计数、超出 1..3600 的允许、或非有限数的时间，都按失步结束引擎。

<a id="the-request-must-be-content-type"></a>

一个文件持有运行状态、最深的检查点、每层 QSA 到对话长度为止的 K/V 以及草稿层的 K/V。格式 v1、小端、定宽整数、
IEEE-754 浮点（大端构建编译不过）：64 字节头（magic `STRSESS\x01`、版本 u32、头大小 u32、模型指纹 u64、配置指纹
u64、负载长度 u64、两个必须为 0 的保留 u64、前 56 字节的哈希）、负载（几何、层区间、cvec 标志、活动状态、检查点、
K/V 层；每个数组前有其 u64 计数）、负载哈希和结尾标记 `STRSEND\x01`。哈希是一个带 xxHash64 风格轮次的 64 位函数，
不是标准 XXH64 流；它检测意外损坏、不认证文件：只恢复本引擎写出的文件。未知版本被拒绝；新格式拿一个新版本号。

<a id="one-file-holds-the-running-state"></a>

文件绑定到模型输入、以及改变已存字节含义的设置。模型指纹按角色对引擎加载的每个文件取样（大小、头 1 MiB 和尾
1 MiB）：GGUF shard（还有 `--native-dense-gguf`、head shard 和 `--embd-gguf`）、PLE shard、包的
index/dense/embedding 文件和 `native_experts.txt`、expert 来源解析出的每个文件——包的 `experts.bin`，
或就地读取的包按 `native_experts.txt` 命名的每层 gate/up/down GGUF，包括 CLI shard 之外的——以及 MTP 的文件；
不包括那些文件夹里的其他文件，也不包括路径，所以搬过家的模型文件夹仍然匹配。文件中部一个保持大小的改动看不见：会话还留着
时不要改模型文件。配置指纹覆盖引擎版本字符串（另一个版本被拒绝；同一版本的两个构建不作区分——没有构建哈希）、CUDA 或
HIP、`--kv`、`STRATA_KV_ROT`、`--kv-resident`、`--max-context`、`--mtp-window`、解析后的 rope 配置（类型、
base、factor、freq scale、原始上下文、YaRN 各旋钮——缓存的 K 是 post-RoPE 的）、加载的控制向量（上传的表的摘要：
每个文件的内容乘它精确的 scale、模式、层区间和方向）以及算术开关（`--native-*`、`--no-ple`、A/B 臂）。采样、种子、
草稿调优和 expert 层不在其中：它们改变接下来的东西，不改变已存单元装的东西。

<a id="a-file-is-bound-to-the-model-inputs"></a>

一次保存写一个临时文件，在 `path` 旁边用一个全新的隐藏名字（独占创建，从不是已有的文件或链接）、把它 flush 到盘、
重命名覆盖 `path`（Windows 上 `MoveFileExW` 配 write-through，它在每种文件系统上都不承诺事务），POSIX 上还
flush 文件夹（Windows 没有文件夹 flush；文件 flush 和 write-through 重命名就是它能做的一切）。重命名之前失败的
保存把旧文件留在 `path`、只删它自己的临时文件。重命名一旦完成旧文件就没了：若文件夹 flush 随后失败，保存带着
`published` 失败——新文件的字节完整且已 flush，但它的名字可能在断电后不保。flush 不了文件夹的文件系统（`EINVAL`）
不算失败；引擎记录它。一次恢复打开 `path` 时跟随符号链接（或 Windows reparse point）会被拒绝，除单名单一普通文件
外都拒绝；它检查大小、头部、两个指纹（在负载解析之前；头含在内、第一个 16 MiB 块已经读了）、解析峰值（图片、读缓冲区、
每段开销）装得进底线（`--conversation-cache-min-free-mib`）之上的 RAM、文件大小对照本会话能恢复的最大值、任何状态数组
之前的几何与层区间、分配每个之前对照剩余字节和本会话精确限制的每个计数（上下文与 cell、检查点、层、每个运行状态数组、每个
K/V 部分）、负载哈希、然后是 usual 的快照校验——全在任何设备写入之前，一次拒绝让当前会话原样未动。设备写入开始之后的传输
失败结束引擎（`FATAL`）而不是从部分状态解码；服务器报 `500` 并再次启动它。恢复不暂存离场的会话。不支持与
`--layer-split`、`--peer-device`、`--batch`（配置的 `"parallel"`，#465；服务器答 `501`）或 `--prompt-cache
0`（RAM 对话缓存不必开着）。Linux 上文件在文件系统接受时以 `O_DIRECT` 按 16 MiB 块移动（否则缓冲 I/O，或
`STRATA_SESSION_BUFFERED=1`）；Windows 上用缓冲 I/O。引擎只在 Linux/CUDA 上跑过。更早一次修订的 CPU 文件 I/O
测试在 Wine 下以 32 位 Windows 可执行文件通过；当前代码没有为 Windows 构建过，Windows 引擎、HIP 和 AMD 卡没有跑过。

<a id="a-save-writes-a-temporary-file"></a>

会话文件保存的是对话状态，不是进程的全部执行历史。不保证跨重启逐 token 精确重放：expert
常驻和 CPU/GPU 舍入可能改变之后的输出。一次 63K 测试里，restore 之后头一次 32-token 续用与
那个一直运行的进程一致；下一次续用时，三个已恢复的进程彼此一致，但从第 28 个 token 起与那个
进程不同。中间一次被拒绝的、损坏的 restore 没有改变已恢复进程的续写。这个分歧的原因尚未被
隔离出来。

<a id="a-session-file-saves-conversation-state"></a>

在 RTX 4070 Ti（12 GB）、Ryzen 9 5900X、64 GB RAM、NVMe ext4、IQ3_XXS、一个 63,025-token
对话上实测（引擎 0.1.38 带此改动，二进制 sha256 `3bbe4fc3...`）：文件 1.20 GB，保存 0.65 秒
含 flush 到盘（再一轮之后对 1.20 GB 状态另做 10 次保存：替换一个文件 0.59-0.61 秒，写到新名字
0.80-1.18 秒），新引擎进程里恢复 1.05 秒，然后下一个 32-token 轮 1.14 秒、与不重启时同一个轮
产出同样的 32 个输出 token（1.00 秒）；冷启动的第一个轮要 25.5 秒。保存报告了 73 行
`SESSION`，彼此从不超 16 MiB。一个符号链接、一个第二硬链接和一个负载翻转了一字节的文件都被
拒绝（最后一个还是在一条已恢复对话活着时做的，它之后照常回答）；一个换了另一个
`--rope-freq-base` 的新引擎拒绝该文件然后正常回答；一次把 RAM 底线设在机器 RAM 之上的保存在
任何拷贝之前被以 `memory` 拒绝，它本要替换的文件原样未动。时间为单次运行。

<a id="measured-on-an-rtx-ti-gb-ryzen-ram"></a>

**当前限制（v1）：** 一次一个请求，除非设了 `"parallel": N`（需显式开启的批处理槽，至多 N
个请求一起 decode：[BATCHING.md](BATCHING.md)）；一次缓存一个对话（在两个聊天之间切换会重读
另一个，除非启用上面的可选缓存，或每个对话保住自己的批处理槽）；图片只在随它安装时才有（下
文）；无视频。**temperature / top_p / top_k / min_p / seed** 按请求生效（OpenAI 与 Anthropic
字段），停止串也一样（OpenAI `stop`，一个字符串或至多 4 个；Anthropic `stop_sequences`）：回
答在第一个命中之前结束、它不发送，引擎在那里停下（`finish_reason` "stop"；Anthropic 为
`stop_reason` "stop_sequence" 且 `stop_sequence` 设为找到的那个）；带默认自适应 expert 层时，
一次采样结果逐次运行不可复现——要 seed 可复现的输出，在引擎参数里加 `--adapt-every 100000`
（静态常驻）。运行配置的可选 `sampling` 块为不写这些字段的请求设默认值（`"sampling":
{"temperature": 1.0, "top_p": 0.95, "top_k": 20}`）；请求自己的字段永远优先。**完全不带该块时
（0.1.45），一个不带采样键的请求得到模型卡推荐的采样**，按请求的 thinking 状态挑选：
Qwen3.8-Flash-Next（和 Gyro）thinking 时 temperature 1.0、top_p 0.95、top_k 20；不 thinking 时
temperature 0.7、top_p 0.8、top_k 20、presence_penalty 1.5。Qwen3.6-35B-A3B thinking 时
1.0 / 0.95 / 20 配 presence_penalty 1.5，不 thinking 0.7 / 0.8 / 20 配 1.5（min_p 0 和
repetition_penalty 1 是"关"）。优先顺序：请求、网页的共享 Chat 设置、配置的 `sampling` 块、模
型卡。贪心只属于开口要它的人：请求里 `temperature: 0`，或配置里 `"sampling":
{"temperature": 0}`（0.1.45 之前，不带块的配置跑贪心，而贪心 decode 正是小本地模型打转的原因；
#2685）。网页在你改动它的 Sampling 设置之前不发自己的采样（此前它显示并发送 0.6）；一个旧的、
存下来的 0.6 配上默认 top_p 和 top_k 算未设置。惩罚项（`presence_penalty`、
`frequency_penalty`、`repetition_penalty`，`penalty_last_n` 限制它们往回数多少个 token，设了任
一惩罚时默认 64；llama.cpp 的名字 `repeat_penalty` 和 `repeat_last_n` 也可用，在配置的
`sampling` 块和请求里都行，#1819）走同一条路径；它们数的是请求已消费的 token，所以重复惩罚压制
的是模型自己刚说的话，不只是 prompt。自引擎 0.1.19 起，它们一次性作用于投机 decode 检查的每个
token，与逐 token decode 完全一致（此前每批只有第一个有）。这让带惩罚的请求比 0.1.18 慢
1-11%：草稿层不带惩罚地猜，所以它更多的猜现在被拒。不带惩罚的请求不变。`top_k` 至多保留 64 个
候选：`0`（"off"）或任何大于 64 的数用满 64 个。

<a id="current-limits-v-one-request-at"></a>

**层级切分上的批处理 MTP（`--batch-mtp`）。** 我们在 2x R9700、全部 expert 在 VRAM 上的测量：
对比流水线默认值 80.2 -> 62.7 tok/s（-21.9%，5 对里 0 对更快），所以它只在 expert 装不进 VRAM
时才划算（docs/BATCHING.md）。

<a id="batch-mtp-on-a-layer-split-batch-mtp"></a>

**Token 偏置（需显式开启）。** chat API 在不带连续批处理的受支持 CUDA/HIP 引擎上支持请求级
`logit_bias` 对象和 token/bias 对列表。值 -100（或对列表里的 `false`）排除一个 token。不支持的
引擎返回错误。格式、限制与校验见 [LOGIT_BIAS.md](LOGIT_BIAS.md)。

<a id="token-biases-opt-in"></a>

**采样式草稿（需显式开启，0.1.40.2）。** temperature 大于 0 时，引擎只在草稿等于模型为该位置
采样出的 token 时保留它，草稿层提出它的最佳猜测。两个可选开关改变草稿层如何为采样请求打草稿；
贪心请求（temperature 0）从不被触碰、保持逐字节相同。`STRATA_SPEC_COUPLED=1`（或
`--coupled-draft`）用请求自己的设置和检查行将要用的随机数采样来打草稿；同一个 seed 的文本不
变。`STRATA_SPEC_PROB=1` 是投机拒绝采样：草稿层从它自己的分布 q 采样它的猜测，检查以概率
min(1, p/q) 接受它（p 是模型在惩罚、top_k、top_p、min_p 和 temperature 之后的分布），否则从
剩余分布采样替代者，所以每个 token 的分布与没有草稿时完全一样（此时一个 seed 的文本取决于草
稿，因而只对同一引擎、同样的开关和同一个 prompt 可复现）。没有分布的猜测（prompt 查找、后缀草
稿）保持精确匹配规则，那本来就是精确的。没有一个在站得住的意义上更快：RTX 3060（IQ3_XXS，200
-token 故事与代码回答，10 个交错对）上，默认在 temperature 0.3 / 0.7 / 1.0 的输出速度中位数为
42.4 / 43.2 / 43.3 tok/s，耦合草稿 40.6 / 42.4 / 43.3，拒绝采样 41.0 / 42.6 / 42.6，重复之间逐
次差异最高 5%；每轮保留的草稿在 0.3 下降、在 1.0 涨约 5 个点，而更长采样窗口的开销吃掉了它。
`STRATA_SPEC_DEPTH=1` 为每个请求打印按草稿深度的接受率；`STRATA_SPEC_MIN_TEMP=0.9` 把采样式草
稿限制在更热的请求。拒绝规则保持分布的证据：`spec_prob_test`（卡方，每例 200 万次试验）和
`spec_verify_parity`（GPU 对照 host 参考实现）。

<a id="sampled-drafting-opt-in"></a>

---

### Responses API 与 Codex CLI

<a id="the-responses-api-and-codex-cli"></a>

`POST /v1/responses`（0.1.39，#451）说 OpenAI 较新的 Responses API，Codex CLI 用它（它不再说
Chat Completions）。它跑在与 `/v1/chat/completions` 相同的路径上，所以 thinking 级别、thinking
预算、对话缓存，以及同样的 API 密钥、Host 与 Origin 检查都适用。

<a id="post-v1-responses-openai-responses-codex"></a>

它是**无状态的**：什么都不存，所以客户端每次都在 `input` 里发送整个对话（Codex 就是这样，带
`store: false`）。`previous_response_id`、`conversation`、`background` 和
retrieve/delete/cancel 端点会被拒绝，错误信息说明原因。

<a id="it-is-stateless-nothing-is-stored"></a>

| 请求 | Strata 做什么 |
| --- | --- |
| `input` 为字符串或条目列表 | `message` 条目（`user`、`assistant`、`system`、`developer`；文本和图像）、`reasoning`、`function_call`、`function_call_output`、`custom_tool_call(_output)` |
| `instructions` | system 消息（连同开头的 `developer` 消息；较后的变成 user 消息，与 chat 路径相同） |
| `tools` | `function` 工具、`namespace` 工具（模型看到 `namespace.name`；调用带着 `namespace` 和 `name` 返回）、`custom` 工具（一个自由形式的 `input` 字符串）。托管工具（`web_search`、`file_search`...）被略去：模型跑不了它们 |
| `tool_choice` | `"none"` 隐藏工具；其他任何值让模型自选（不能强制） |
| `reasoning.effort` | `none`/`minimal`、`low`、`medium`、`high`/`xhigh`；没有它时用配置的 `reasoning_effort`（0.1.44.1；0.1.44 在这里忽略了它），再否则用模型默认（high） |
| `max_output_tokens` | 输出上限（含 thinking）。用尽时响应以 `incomplete`（`max_output_tokens`）结束 |
| `text.format` | `json_schema` 与 `json_object` 使用 [JSON 响应格式](#json-response-formats)（校验，不约束） |
| `client_metadata["x-codex-turn-metadata"]` 里的 `thread_source` `thread_title` | Codex 的会话标题请求（需显式开启：`strata-<model>.json` 里的 `"codex_thread_titles": true`，默认关），来自另一个会话、带 `tools: []`、在每个用户轮旁边发出：不经引擎回答 `{"title": "<用户的原话，截到 36 个字符>"}`，所以对话保住 prompt 缓存。标题是用户的原话，不是模型写的 |
| `temperature`、`top_p`、`reasoning_budget_tokens`，... | 与 chat 路径相同 |

模型的 thinking 以带 `reasoning_text` 内容的 `reasoning` 输出条目返回（以
`response.reasoning_text.delta` 流式发送）。这个模型不写单独的摘要，所以 `summary` 为空。带
`"include": ["reasoning.encrypted_content"]` 时条目还带 `encrypted_content`：一个不透明字符串
（base64，未加密；客户端本来就持有文本）。像 Codex 那样把 reasoning 条目连同对话其余部分发回：
它们的 thinking 回到 prompt 里，所以与模型写的一致，对话缓存得以复用。

<a id="the-model-s-thinking-comes-back"></a>

带 `"stream": true` 时事件是官方的那些、按序：`response.created`、`response.in_progress`，然后
每个输出条目有 `response.output_item.added`、它的增量（`response.reasoning_text.delta`、
`response.output_text.delta`、`response.function_call_arguments.delta`）、它的 `...done` 事件和
`response.output_item.done`，最后 `response.completed`、`response.incomplete` 或
`response.failed`。长 prompt 读取期间，每 15 秒发一个 `response.in_progress` 事件，Codex 把它
计为活动。回答开始前的错误用 Responses 的形式（`{"error": {"message", "type", "param",
"code"}}`）；开始之后以 `response.failed` 到达。

<a id="with-stream-true-the-events"></a>

**Codex CLI。** 在 `~/.codex/config.toml`（Windows：`%USERPROFILE%\.codex\config.toml`）里：

```toml
model = "strata"                        # 任意名字；Strata 用它自己的模型回答
model_provider = "strata"
model_context_window = 32768            # 你在 setup 里选的上下文：Codex 在到达它之前先压缩
show_raw_agent_reasoning = true         # 显示模型的思考（它不写摘要）
# model_reasoning_effort = "medium"     # none、low、medium 或 high；默认：模型的（high）

[model_providers.strata]
name = "Strata (local)"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # 第一个长 prompt 读完可能要几分钟
# env_key = "STRATA_API_KEY"            # 仅当服务器有 api_key 时：持有它的变量
```

然后照常运行 `codex`（或 `codex exec "..."`）。Codex 对本地模型名会警告 `Model metadata for
... not found`；那是预期内的。用 Codex CLI 0.160.0 和 Q2_0 在 RTX 5070 上实测：它的第一个
prompt（指令与工具描述）9,443 个 token，10 秒读完；在一个工具循环里，之后的每个轮从缓存复用约
96% 的 prompt，只读新增的部分，1-2 秒。在 Windows 上，Codex 的沙箱在该测试里拒绝了每条 shell
命令，直到它用 `-c 'windows.sandbox="unelevated"'` 启动（这是 Codex 的设置，不是 Strata 的）。

<a id="codex-cli-in-codex-config-toml"></a>

**Codex 的压缩（需显式开启：`strata-<model>.json` 里的 `"codex_compaction_cache": true`）。**
上下文装满时（或 `/compact` 时），Codex 把对话再发一遍并附带一个总结它的请求，且 `tools: []`。
模板把工具写在 prompt 顶部，所以那个 prompt 与引擎持有的对话只共享头几个 token，只能从头重
读，而且是它最长的时候。开着该选项时，Strata 记住是哪个 Codex 对话发过最后一个非压缩 prompt
（`client_metadata["x-codex-turn-metadata"]` 里的 `session_id` 和 `thread_id`）以及那个 prompt
渲染时用的工具；一个被 Codex 标记 `"request_kind": "compaction"`、来自同一会话且自己不带工具的
请求，就用那些工具渲染。只改 prompt：对输出解析器和响应而言，请求的工具仍是 Codex 发的那样，
即没有，所以模型写出的工具调用会作为不带命名空间或 custom-tool 形式的普通 function call 返
回。它只有一项，每个这样的 prompt 都会替换它；另一个会话的压缩、一个不带该元数据的请求（0.140
之前的 Codex）、或一次重启都按原样渲染请求，一个带保留工具仍装不进上下文的请求也一样。默认
关：压缩的 prompt 与客户端发的那个不同。作者用 Codex CLI 0.160.0 和一个只数共享 prompt 开头的
引擎实测：一段 85,000-token 对话之后的压缩复用了它 84,997 个 token 中的 84,895 个、只读
102 个；不带它时，80,683 个中复用 41 个。

<a id="codex-s-compaction-opt-in-codex-compaction-cache"></a>

## 来自 MCP 服务器的工具

<a id="tools-from-mcp-servers"></a>

聊天页面可以给模型提供来自 [MCP](https://modelcontextprotocol.io) 服务器的工具，像 LM Studio
和 Claude Desktop 那样：读你的文件、抓网页、搜索，任何 MCP 服务器提供的东西。把服务器列在
`strata-<model>.json` 的 `"mcp_servers"` 下——与 Claude Desktop 的 `mcpServers` 块同样的形状，
你也可以原样粘贴（键 `"mcpServers"`）：

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me\\Documents\\notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

或把它们放在自己的文件里，并用 `--mcp-config path\to\claude_desktop_config.json` 启动服务器（一
个带 `mcpServers` 块的文件；加到你启动脚本的 `serve/server.py` 行上）。改动后重启 Strata。

<a id="the-chat-page-can-give-the-model"></a>

- **一个程序**（`command`、`args`、可选 `env` 和 `cwd`）由 Strata 启动并经它的 stdin/stdout
  对话；`npx`、`uvx`、`python` 等照常从 `PATH` 找到（`npx` 服务器需要 Node.js）。**一个地址**
  （`url`、可选 `headers`）用 MCP 的 Streamable HTTP 传输（不支持较旧的仅 SSE 传输）。
  `"disabled": true` 把一项排除在外。
- 服务器随 Strata 在后台启动；服务器窗口说明每个提供了什么（`MCP server 'files': 14 tools
  (...)`），或它为什么没启动——那时它的工具被略去，聊天照常进行。Monitor 标签页列出它们，
  Sampling 抽屉里有 **Use tools from MCP servers**（默认开）。之后停掉的服务器在它下次被调用时
  重新启动。
- 聊天里每次调用显示为一个小区块（工具、参数、结果）；模型读结果并继续，每次回答最多连续
  `max_rounds` 次调用。失败或超过 `timeout_s`（默认 60 秒）的工具给模型一个 `error: ...` 结果而
  不是结束聊天。超过 `max_result_chars`（默认 20,000 字符）的结果在模型读到之前被截断并附说明。
  Stop 也会停止运行中的工具。
- 只有聊天页面用它们。API 客户端（omp、Claude Code、OpenAI 和 Anthropic SDK）看到的 API 与之前
  完全相同并保留自己的工具；一个对 `/v1/chat/completions` 的请求用 `"strata_mcp": true` 选择加
  入（之后它在流里得到 `strata_mcp` 工具事件）。

**安全。** MCP 工具在你的 PC 上以你的用户权限运行，而**模型自己决定何时调用它们**——也会因为
它读到的东西（一个网页或文件里可以藏指令）。给文件系统服务器只给必需的文件夹，优先只读工具，
不要添加你不信任的服务器。这些工具只能从聊天页面本身使用（带别的网站 Origin 或不带 JSON 内容
类型的请求被拒绝）；如果 Strata 能从其他设备访问，请设一个 API 密钥。

<a id="security-mcp-tools-run-on-your-pc"></a>

**262K 之后的上下文扩展（rope 缩放，实验性，除非你选它否则关）。** 模型在 262,144 个位置上训练
（rotary base 1e7）。Rope 缩放重新标定旋转角度，让更长的上下文保持可用，用 llama.cpp 的类型和
开关名。`linear` 是 Position Interpolation：每个角度按因子缩小。`yarn` 保留高频角度、插值低频角
度，并加上把注意力温度保持在训练位置的幅度校正。**不带这些开关时什么都不变：** 一次未缩放的运
行逐字节算的和该功能出现之前完全一样。缩放后的上下文需要成比例更多的 VRAM/RAM 给 KV 缓存和
rope 表（每 token 约 13 KB 和约 0.26 KB）。

<a id="context-extension-past-262k-rope-scaling-experimental"></a>

实测内容（贡献者的运行，RTX 5080 + IQ3_S，原生路径，PR #84）：同一批 token 上的按位置困惑度，
只改缩放开关。293K token（训练长度的 1.12 倍）时，`yarn` 因子 2 对比 `none` 和 `linear` 2 把
NLL 降低 0.18 nats（2.04 对 2.22 / 2.22）。那是同一长度测得路径噪声的 3-4 倍。`linear` 2 与
`none` 无法区分。2.7K 和 32K 上没有一个臂与噪声分开。大海捞针测试分不出各臂：未缩放的模型在
413K 也捞得到针。真实长文档问答在 `yarn` 2 @ 421K 和 `yarn` 4 @ 714K 上都成功（各 8/8），一次
1M-token 的 `yarn` 4 运行从头读到尾。综合起来，用 **yarn**。它仍是实验性的：数字来自一台机器和
一个量化。

<a id="what-was-measured-contributors-runs-rtx"></a>

- setup：`START-HERE.bat --setup --context 393216` 不多问什么——它替你选好方法（yarn；交互式运
  行时问一个问题）并从最终上下文推出因子（最终上下文 / 262,144，至少 1：393K 是 1.5，512K 是
  2，训练范围内是 1；`--rope-scaling`/`--rope-scale` 覆盖；一个显式的 `--rope-scale` 按给定值保
  留，即便它对实际服务的上下文太小，自己设了就要检查）。对超过 262,144 的上下文显式指定
  `--rope-scaling none` 会被拒绝：setup 不会在训练范围之外按原始角度配置一次运行。如果 RAM 检
  查把选定的 384K/512K 压回训练范围内，省略的方法不加缩放，显式选的保持因子 1——训练角度，不
  扩展（这不是 rope 总开关：显式提供的 rope 设置保持自己的行为）。
- 引擎：`--rope-scaling none|linear|yarn`、`--rope-scale F`，以及原始 ggml 旋钮
  `--rope-freq-base`、`--rope-freq-scale`、`--yarn-orig-ctx`（默认 262,144）、
  `--yarn-ext-factor`、`--yarn-attn-factor`、`--yarn-beta-fast`（32）、`--yarn-beta-slow`（1）。
  模型文件的 `rope.scaling.*` 键（一次微调版随带时）是被开关覆盖的默认值。

缩放对整次运行固定——引擎在旋转之后把 key 存进它的缓存，所以一个缓存绝不能混两种缩放，也没有
按请求的形式。在训练的 262,144 之内，一次缩放的运行是一个略不同的模型：重标的角度和 `yarn` 的
幅度校正在每个位置都生效，不只在训练终点之后。这就是 setup 只对超过 262,144 的上下文开启缩放的
原因。图片读同一张缩放后的表（它们的 (t, h, w) 位置喂给它）。这应该可行，但没有测过：上面所有
运行都是文本。

<a id="the-scaling-is-fixed-for-the-whole-run"></a>

---

## 从你的 AI 助手管理 Strata（MCP 服务器）

<a id="manage-strata-from-your-ai-assistant-mcp-server"></a>

`tools/strata_mcp.py` 是一个给 Claude Code、Claude Desktop、Cursor、VS Code、Codex 和其他助手
的 MCP 服务器。加上它之后，你可以问助手"给这台 PC 安装 Strata""启动 Strata"或"Strata 在
跑吗？"。在 Claude Code 里这样添加：

```bash
claude mcp add strata -- python C:\Users\you\Strata\tools\strata_mcp.py
```

<a id="tools-strata-mcp-py-is-an-mcp"></a>

它有八个工具：status（运行中的模型、装了些什么、硬件、一个推荐尺寸）、模型列表、install、
start、stop、logs、一次速度测试，以及给其他应用的连接设置。

Install 在后台用 `--yes` 运行 `setup.py`。下载任何东西之前，它展示计划并等你确认。Start 和
stop 的行为同启动脚本和服务器自己的 unload。该 MCP 服务器只结束它自己启动的进程。它只用 Python
标准库，所以在 `.venv` 存在之前就能工作。

<a id="install-runs-setup-py-with-yes"></a>

每个客户端的配置片段、工具参数和安全规则在 [MCP_SERVER.md](MCP_SERVER.md)。这与上面
[来自 MCP 服务器的工具](#tools-from-mcp-servers) 方向相反——那边是 Strata 模型调用*你的* MCP
工具。

<a id="the-config-snippets-for-every-client"></a>

---

## 图像（视觉）

<a id="images-vision"></a>

（Qwen3.6-35B-A3B 以同样的方式读图片，用它自己的 `mmproj-F16.gguf`：[QWEN36.md](QWEN36.md)。）

<a id="qwen-b-a-b-reads-pictures-the-same"></a>

模型有一个视觉编码器：[`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
（0.9 GB，一个 27 层 ViT 加进语言模型的投影器）。凡模型能读图片的地方，图片**默认开**。应用、
`strata launch` 和 setup 对读取者用一条规则：**24 GB 以下图形内存用 CPU 读取者**（不占显卡内存，
expert 保住它们的空间），**24 GB 起用显卡读取者**，NVIDIA 和 AMD 卡都如此。你可以关掉图片
（`--vision no`，或应用和 `strata launch` 里的 "No"），或按模型选读取者（`--vision gpu` 或
`--vision cpu`）。setup 下载编码器并在引擎旁边放一个小助手（`strata-vision`，来自 llama.cpp 的
`mtmd` 库）；它把它加进你的启动脚本。其他一切不变。读取者缺失的地方图片保持关闭并在界面上说
明：Mac 应用和 Mac 终端还没有图片读取者，Intel Arc 引擎也没有。AMD 上显卡读取者是
`strata-vision-hip`，引擎包里的第二个程序；它启动不了时，服务器自己启动 CPU 读取者并打印一行
（[AMD_HIP.md](AMD_HIP.md)）。

<a id="the-model-has-a-vision-encoder-mmproj"></a>

| 编码器开 | 每张图的时间 | 代价 |
| --- | --- | --- |
| **GPU**（24 GB 起的默认） | **0.1-0.5 秒**（至多 1,024 个图像 token） | 约 1.4 GB VRAM 为它留空，expert 缓存因此更小：文本输出慢几个 %（下表） |
| CPU（24 GB 以下的默认） | 8 核、300 个图像 token 约 3 秒（4 线程 6-13 秒）；token 更多花更长时间，成正比（#767、#625） | GPU 上什么都不占 |

一张图片成为至多 1,024 个上下文 token（一张 640x480 照片：300 个）。同一张图片再发一次——聊天
应用每一轮都这样——只编码一次。

<a id="a-picture-becomes-up-to-tokens"></a>

**惰性 vision（编码器的 VRAM 交给 expert 缓存，直到一张图片到来）：** 在 `strata-<model>.json`
的 `"vision"` 一节加 `"lazy": true`。配 `"vram_elastic": false` 用仅 GPU 的 KV 增长，或配
`"vram_elastic": true` 且不用仅 GPU 的 KV 增长。不要给惰性 vision 开批处理推理。编码器于是不在
启动时启动：它本会占的约 1.4 GB VRAM 留在 expert 缓存里，文本运行没有那点视觉开销。第一个图片
请求交还那块 VRAM——引擎的 `VRAM` 命令为每块 GPU 拿一份预留——并启动编码器；`"vram_mib"` 设定
每块 GPU 的量（层级切分用 `[1700, 600]`，单块 GPU 用 `[1800]`；默认值与本模型的编码器匹配）。
`"idle_s"` 秒没有图片请求之后编码器停止、VRAM 回到缓存（它编码过的图片仍缓存在磁盘上）。空闲空
隙后的第一张图片要几秒后才回答，因为编码器要重新启动。

<a id="lazy-vision-the-encoder-s-vram"></a>

**惰性 vision 配常驻低 RAM 模式或 `--peer-device`：** 那两条路都不存在。引擎在常驻低 RAM 模式
（它的缓存装着 RAM 装不下的 expert）和 `--peer-device` 下拒绝 `VRAM` 命令，而仅 GPU 的 KV 增长
要求所有 expert 在 RAM 且不带 `--vram-elastic`，所以常驻配置也得不到它。硬启动的话，编码器找不到
可拿的 VRAM：第一个图片请求回答 400 并附引擎的拒绝，其后每个请求都如此，而 `/health` 一直是
ok。因此启动器在这种配置里关掉 `"lazy"`，随模型一起启动编码器，让 expert 缓存在启动时按它定大小
（与 `"lazy": false` 同形）。

<a id="lazy-vision-with-the-resident-low-ram"></a>

**惰性 GPU vision 配仅 GPU 的 KV 增长：** 主配置里保持 `"vram_elastic": false`。在 `"vision"`
一节设 `"gpu": true`、`"lazy": true` 和 `"idle_s": 900`。编码器在一个图片请求时加载，900 个空闲
秒后卸载。运行时在编码器启动之前释放 expert 缓存的段。编码器停止时，运行时恢复可用的缓存段并从
expert profile 填满它们的槽。图像 embedding 仍缓存在磁盘上。

这条路径用与 KV 增长相同的每 GPU 控制器。它不缩小活动 KV 状态，也不移动 prompt 缓冲区。不要为
这条路径启用 `"vram_elastic": true`：分段式 expert 缓存分配会禁用仅 GPU 的 KV 增长。

双 V100 IQ3_S 服务用五秒的视觉空闲超时检查过。GPU 进程观察确认编码器为一个图片请求启动、在空
闲超时后退出、又为另一个图片再次启动。两次图片请求都返回了正确的回答。一张 27,978 个 prompt
token 的对话截图返回了正确的摘要，同时两块 GPU 的 KV 池都涨到 32,768 个 cell。随后一个短文本
请求把两个池都修剪回 8,192 个 cell。prompt 分块在整个 VRAM 转换过程中保持 8,192 个 token。服务
之后被调回 900 秒的空闲超时。这些检查没有跑完整的 524,288-token 上下文。

<a id="lazy-gpu-vision-with-gpu-only-kv"></a>

**更多图像 token（0.1.39，#625）：** setup 里的 `--vision-tokens N`（`START-HERE.bat --setup
--vision cpu --vision-tokens 768`）设定一张图片变成的最多 token 数——`strata-<model>.json` 的
`"vision"` 一节里的 `"max_tokens"`，也可以手工编辑。token 更多保留更多细节（小字、图表、截图）
且编码更久，CPU 上最明显；重跑 setup 保留该值。

<a id="more-image-tokens-39"></a>

**图像 token 的下限（0.1.40，#767）：** `strata-<model>.json` 的 `"vision"` 一节里的
`"min_tokens": N`（手工编辑；重跑 setup 保留它）以 `--min-tokens N` 传给编码器（mtmd 的
`image_min_tokens`）：一张小图被放大到至少 N 个 token。llama.cpp 的 mtmd 说 Qwen-VL 模型在
grounding 任务（指认、数小物件）上至少要 1,024；CPU 默认保持最多 300 且无下限，因为更大的下限
会改变图像回答并多花编码时间（8 核、300 token 约 3 秒，与 token 数成正比）。

<a id="a-minimum-number-of-image-tokens"></a>

**一个 Q8_0 编码器（#625）：** `"vision"` 一节里的 `"mmproj"` 可以指向本模型的另一个 mmproj
文件，例如一个 Q8_0 的（llama.cpp 的 `convert_hf_to_gguf.py --mmproj --outtype q8_0` 能做一
个）：编码器的库读量化权重，文件更小（590 MiB 对比 BF16 的 865 MiB，出自 `llama-quantize
mmproj-Qwen3.8-Flash-Next-BF16.gguf mmproj-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0`），而且在 CPU 上
用更少 RAM（编码器少约 280 MiB），默认 300 token 时编码可能比 BF16 更快。Setup 下载 BF16 文件，
重跑 setup 保留一个仍然存在的你自己的文件。**推荐配 `--vision cpu`。** 对 BF16 的精度，按图像
token：embedding 的余弦平均 0.997-0.999（#625 的报告，用一个社区 Q8_0 文件，300、768 和 1,024
token），用上面命令做的文件在两张图片上是 0.9988 和 0.9987（300 token；最差单 token
0.94-0.96）。768 token 以上的编码时间与 BF16 相当（约 3% 以内）。更多图片（图表、小字）的数字
欢迎发到 #625。

<a id="a-q8-encoder-625"></a>

**CPU 上的 flash attention（#660）：** ggml 的快速 CPU 内核要求编码器的 head size（72）是向量
宽度的整数倍：AVX2 是 8 个 float——发行构建用的——但 AVX-512 是 16，在有的 CPU（Zen 4 及更新、
部分 Intel）上从源码构建会拿到。那里 ggml 退回一个慢几倍、以 FP16 累加的内核，所以 `strata-vision`
在其 ggml 有 AVX-512 时于 CPU 上关掉 flash attention，否则保持 `auto`（开）；`--flash-attn
on|off|auto` 覆盖这个判断。Ryzen 7 7700X、8 线程、F16 mmproj、每个新进程一次编码，一张
1024x1024 图片（1,024 个图像 token）：

| 构建 | Flash attention | 编码 | 峰值 RSS | 输出对比 FP32 attention |
| --- | --- | --- | --- | --- |
| AVX2（如发行） | on（默认） | 8.3 / 9.9 s | 1,152 MiB | 1.2% |
| AVX2 | off | 15.2 / 14.0 s | 2,194 MiB | 1.1% |
| AVX-512（源码） | on | 43.5 s | 1,152 MiB | 26% |
| AVX-512（源码） | off（默认） | 13.3 s | 2,195 MiB | - |

最后一列是 embedding 与 AVX-512 不开 flash attention 那次运行的相对 L2 距离；两个构建本来就有
约 1% 的差异。

<a id="flash-attention-on-the-cpu-660"></a>

**给编码器一块备用 GPU（0.1.33，#408）：** 有一块引擎不用的显卡时，在 `strata-<model>.json` 的
`"vision"` 一节加 `"cuda_device": 2`（编号方式同 `nvidia-smi`）：编码器就只在那块卡上跑。同时
把 `"args"` 里的 `--vram-reserve-mib` 降到 700，让引擎的卡把那部分 VRAM 留给 expert 缓存。编码
器的卡需要在现成编码器里有对应代码（RTX 20/30/40/50）。

<a id="a-spare-gpu-for-the-encoder-33"></a>

**只在有图片时开编码器（需显式开启，`"vision"` 一节里的 `"elastic": true`，配配置的
`"vram_elastic": true`；[VRAM_ELASTIC.md](VRAM_ELASTIC.md#the-image-encoder-on-demand)）：** GPU
编码器不随服务器启动，expert 缓存保住全部 VRAM。第一张图片的那个轮让缓存留出 `"elastic_mib"`
（默认 2048）并在那里启动编码器（几秒）；之后的图片直接用现成的，`"elastic_idle_s"`（默认 120）
没有图片之后它停止、缓存把 VRAM 拿回。R9700 32 GB（gfx1201、Linux、UD-IQ4_XS、BF16 编码器至多
1,024 token，HIP 的虚拟内存见 #2189）上实测：编码器拿 1,339 MiB、4.7 秒启动（第二次 2.7 秒）；
一个带一张图的请求含启动花 8.6 秒，后来一次冷启动 5.0 秒；编码器在跑时 2 / 5 / 10 张图花 3.8 /
5.6 / 9.0 秒（每张编码约 0.45 秒，与在第二块卡上相同）。不设该键时什么都不变。

<a id="the-encoder-only-while-there-are-pictures"></a>

### 发送一张图片

**终端聊天：** 输入 `/image <图片路径>`，回车，然后输入你的问题。

```
you> /image C:\Users\me\Pictures\receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API**（一个 `image_url` 部分：一个 `data:` URL、一个 `http(s)://` URL 或一个本地文件
路径）：

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API：** 一个带 `base64`（或 `url`）源的 `image` 块，照惯例。

JPEG、PNG、BMP、GIF、WebP、TIFF 和 AVIF 都可用（后面几种先转成 PNG；omp 这样的 agent 发
WebP）。带图片上传的聊天应用以同样的方式使用。

<a id="sending-a-picture"></a>

### 开着图片时的速度（4K 上下文，实测）

| 模型 | Prompt tok/s，图片关 | Prompt tok/s，图片开 | 输出 tok/s，图片关 | 输出 tok/s，图片开 |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 541 | 531 | 90.7 | 86.9 |
| **IQ2_XS** | 468 | 458 | 77.1 | 74.0 |
| **IQ3_XXS** | 411 | 401 | 64.7 | 59.5 |

这些数字出自 2026-09-24（0.1.13 之前的引擎）。"关"是发行的 setup；"开"是编码器加载在 GPU 上并
为其留空 VRAM 的运行，于是约 1,000 个更少的 expert 装进 VRAM，CPU 每 token 多算几个：慢 2-8%。
对文本，打开图片不改变其他任何东西（VRAM 相同时输出逐比特相同）。

<a id="speed-with-images-on-k-context-measured"></a>

**一块 16 GB 卡上（0.1.45，#2406）。** 编码器的空间约是 16 GB 卡约 4,500 个 expert 槽中的 850
个，而大多数 expert 在 CPU 上计算的地方，它会移动命中率。在 Linux 上用 setup 写的配置实测（IQ3_S
配 Tesla P100，IQ3_XXS 配 RTX A4000，65K 上下文，KV 8-bit，5 轮全新引擎启动，中位数）：GPU 编码
器对比 CPU 编码器，聊天 decode 31.1 对 32.6 tok/s（P100，+4.8%）和 46.0 对 48.4（A4000，+5.2%），
4K prompt 315.5 对 316.0 和 1,022 对 1,053 tok/s，16K prompt 不变 / +0.8%；expert 槽 3,819 对
4,522（P100）、4,686 对 5,435（A4000）。"elastic" 编码器（`"vision": {"elastic": true}` 配
`"vram_elastic": true`；它 120 秒空闲后释放编码器，但会关掉 `--kv-grow`）在 65K 上下文给出了 CPU
编码器的文本速度（32.5、47.7 tok/s），加载后每张图 2.7 秒（第一张 4.5-5.8 秒）；196K 上下文配
q4_0 KV（那里它保不住 `--kv-grow`）时则没有（A4000：GPU 44.7、CPU 47.4、elastic 44.9 tok/s；CPU
编码器 +6%）。一份 Windows 下 RTX 4080 SUPER 的报告测得 GPU 编码器时 decode 剩 35-44%、prompt 剩
36-62%（空闲 VRAM 61 MiB）；那个规模在 Linux 的卡上没有复现，所以它可能来自显卡装满时 Windows 的
VRAM 管理（我们没有测）。一张图在 CPU 编码器上花 4-6 秒（300-token 的图），对比 P100 的 GPU 编码器
2.7 秒（A4000 上 1.7 秒）。自 0.1.45 起，setup 给 24 GB 以下 NVIDIA 卡上的 "images: yes" 配 CPU
编码器（图片按至多 300 token 读；要更多细节用 `--vision-tokens 1024`）；`--vision gpu` 仍可用，但
附一条说明。

<a id="on-a-gb-card-45-2406"></a>

一个关于图片的问题（一张 640x480 报纸版面 = 300 个图像 token，328-token 的 prompt；整个请求，
编码器在 GPU 上，经 API 实测）：

| 模型 | 回答（"MEN WALK ON MOON"） | 图片编码 | Prompt | 输出 |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 1.9 s | ~0.1 s | 207 tok/s | 65-73 tok/s |
| **IQ2_XS** | 2.3 s | ~0.1 s | 173 tok/s | 50-60 tok/s |
| **IQ3_XXS** | 2.7 s | ~0.1 s | 144 tok/s | 41-50 tok/s |

（短 prompt 跑得低于 4K prompt 速度：2,048-token 的分块才是 prompt 路径高效的区间。短回答跑得低
于长输出速度：头几轮还没有草稿。）

**内部原理：** 编码器把图片变成与模型词 embedding 同宽的若干行；Strata 把它们放到 prompt 里
`<|image_pad|>` token 的位置，并给每一行它的二维位置（图片里的行和列；模型用交错 M-RoPE）。在
我们的测试图片上，回答与 llama.cpp 的多模态实现逐 token 相同。

<a id="how-it-works-inside-the-encoder-turns"></a>

---

**Stager 等待（0.1.40.2，`STRATA_STAGER_SLEEP`）：** prompt 装载线程在 Linux 上等待时睡眠、在
Windows 上自旋，因为自旋在那里略快（一块 5070 读一个 8K Q2_0 prompt 快 1.2%）。
`STRATA_STAGER_SLEEP=1` 让 Windows 也睡眠，这是共享机器比速度更重要时的选择：Ryzen 9 7940HS 笔
记本配 RTX 4070（IQ3_S、64K 上下文）上，睡眠式等待读 5,914 和 22,305 个 token 的 prompt 慢 5-6%，
而整机 CPU 占用从 77-90% 降到 21-25%（issue #1101，感谢 midhatn）。`STRATA_STAGER_SLEEP=0` 强制
Linux 自旋。两种方式输出相同。

**就地读 GGUF 的 stager 线程（0.1.41，#1353）：** 一个原生包就地从它的 GGUF shard 读取（UD-Q4_K_XL、
一个 Q8_0 包）时，在字节是 SSD 上的缺页的情况下，用 32 线程和一个 128 深环形缓冲拷贝每个 expert。
若 shard 的页已在页缓存里（大 RAM 机器、热启动），那些拷贝是内存拷贝，同样的画像比 4 线程默认慢
4-7 倍（一个 124K-token Q8 prompt：默认下 57 -> 350-410 tok/s）。引擎现在在第一个 prompt 时（Linux）
用 `mincore` 抽查一部分 expert，其页 90% 以上常驻时采用 RAM 画像并在日志里说明；冷的或部分缓存的
GGUF 保持 SSD 画像。`STRATA_STAGER_SSD=1` / `0` 强制任一画像；`STRATA_STAGER_THREADS` 仍然更优先。
拷贝的字节相同。

<a id="stager-waits-40-strata-stager-sleep"></a>

## 短 prompt：CPU 分担 expert（单块 NVIDIA GPU 上默认开，`STRATA_PREFILL_CPU_SHARE`）

一个几千 token 的 prompt 分块（agent 的工具结果、一次测试的输出、一条短的追问）把每个不在 VRAM
里的被路由 expert 经 PCIe 流式搬运，而 decode 时闲着的 CPU 池和已经持有那些 expert 的 RAM 就在经
常状态。`STRATA_PREFILL_CPU_SHARE=auto` 把该分块只有少数 token 路由到的 expert 交给 CPU 池，按层测
量每边花多久，并把两边同时结束的那个 CPU 份额给它（`STRATA_PREFILL_CPU_SHARE=0.4` 固定一个份额）。
`auto` 还检查分担是否划算：它分别在有份额和无份额时给层计时（头几层交替直到凑满三次比较，之后每
29 次跑一次反面）并只在分担的层是更快的那些时才分担。分担更慢的地方（RX 7900 GRE 配 Ryzen 7
5700X3D、ROCm 下每种份额都慢，#1282），除了那些测量层之外 expert 留在 GPU 上。不设置时它是关的，
开启后输出与没有该功能的构建逐字节相同。

**0.1.41：默认开**——单 GPU、不带 `--batch` 槽的 CUDA 构建上，对 1,024 token 以下的 prompt 分块：
不设置时表现为 `auto` 且 `STRATA_PREFILL_CPU_SHARE_MAX=1024`，引擎在启动时打印一行说明。实测（交错
的整引擎运行、8 轮、中位数）：512 个 token 省 28..34%、1,000 个 token 省 20..26% 的 prompt 时间（RTX
3060、Tesla P100、RTX 5070，各 8/8 对）；对比关闭的平均 KL 0.004（最大 0.025）。回答可能与 0.1.40.3
略有不同。**`STRATA_PREFILL_CPU_SHARE=0` 关掉它并给出 0.1.40.3 的精确字节。** 层级切分、批处理槽和
AMD（HIP）构建保持关闭，除非你设该变量；`STRATA_PREFILL_CPU_SHARE_MAX=3072`（向更大分块的扩展）也保
持需显式开启：2,048 个 token 在 5070 上输了 2%。

<a id="short-prompts-the-cpu-shares-the-experts"></a>

开着它时，3,072 token 以下的分块在路由之后装载（只有那些能把一份份额交给 CPU），而不是从 1,024
个 token 起就流式搬运每个 expert；`STRATA_PREFILL_CPU_SHARE_MAX=1024` 保持旧上限。CPU 直接拿它从
RAM 读的 expert：arena、常驻低 RAM 模式的页锁定副本，或文件缓存里映射的 `experts.bin` 的页
（`--mmap-experts`），不只页锁定的那些。serve 不带 `--batch`；层级切分时每个 stage 都有自己的池，
且一次只有一个 stage 为一个分块取用它。

**0.1.42：内存紧张时不再默认启用。** 一份 0.1.41 的报告（RTX 5090，RAM 本已紧张）里聊天持续失败、
引擎反复重启，直到设了 `STRATA_PREFILL_CPU_SHARE=0`。现在，当本进程能拿到的 RAM（Linux 上 cgroup
限制计入）低于份额自己的缓冲区加 3 GiB 时、Windows 上提交上限（RAM + 页面文件）在它们之外不足
4 GiB 时、或它要读的 expert 是文件页（GPU 缓存与 RAM 副本之外的映射 `experts.bin`）而空闲 RAM 加
3 GiB 保不住时，默认值被跳过并在日志里说明原因。显式的 `STRATA_PREFILL_CPU_SHARE=auto`（或一个数）
被保留并只给一条警告。份额的两个页锁定缓冲区（约 50 MB）现在按其最大值一次性分配，分配失败时结束
本次运行的份额而不是让请求失败。

开着时，CPU 的行以 CPU 自己的激活格式计算，所以输出在最后一几个比特上变化（对比关闭的首 token
KL：22 个 prompt 上平均 0.006、最大 0.026 nats；500 和 1,000-token prompt 上约一半的 32-token 贪心
回答相同，其余在约 23 个 token 后近乎平手处分叉）。3,072 个 token 及以上的分块两种写法读法相同。
`auto` 按墙钟计时挑份额，所以同一个请求逐次运行可能被不同地切分——并给出不同的回答（一次续用对话的
重读在 1,536 个 teacher-forced argmax token 中有 31 个不同，#1684）；固定的份额可重复，`0` 给出没有
它时运行的精确字节。

<a id="not-armed-by-default-when-memory-is-short"></a>

**紧凑的 CPU 激活下载。** 一个分担的层只下载它的 CPU expert 读的那些行。这一层过去在计算流上把
整个激活张量（T x N 个 float）拷到 host，在该层的 host 同步之前，不管有没有 expert 交给 CPU。
2,048 个 token 时那次拷贝每层约 21 MB，其中 48 层排在同步前面。现在该层收集它的 CPU expert 读的
token 行，在设备上 gather，并在自己的流上拷贝紧凑的结果，与 expert GEMM 并行。一个不向 CPU 送
expert 的层什么都不拷。份额的每个缓冲区现在按该层实际拥有的份额分配，而不是按分块的最坏情况。CPU
量化的是同样的 FP32 行字节。自动分担仍依赖计时。不保证模型回答完全相同。

**短行 Volta expert GEMM。** `STRATA_PF_WMMA=1` 时，IQ3_XXS、IQ3_S、IQ2_S 的 gate/up 和 IQ4_NL
的 down 在组内最大 expert 不超过 128 行时使用一个紧凑 tile。该 tile 用 128 值的权重块和两个
64 行的输出 tile。IQ3_XXS、IQ3_S 和 IQ4_NL 用 64 值的激活块和 44,032 字节动态共享内存。静态共享
存储分别增加 1,024、2,048 和 16 字节。IQ2_S 用 32 值的激活块和 39,936 字节动态共享内存，其码本
增加 8,192 字节。每个总量都容得下一个 V100 SM 上两个块的共享内存预算。短内核定为每线程 128 个寄
存器上限。更大的组保留原来的 256 行输出 tile 以避免额外的权重解码。Q2_0 和其他量化格式保留它们现
有的 tile 选择。

用 `pf_wmma_parity` 检查短组和宽组、稀疏输入行索引、间接权重和输出守卫。在空闲 GPU 上用
`pf_wmma_parity --bench` 测量经过验证的 IQ3_XXS、IQ3_S、IQ2_S 和 IQ4_NL 形状。该模式在一个 CUDA
graph 里测八次启动并报告七个 CUDA-event 窗口的中位数。它不含请求准备、权重传输和 token 生成。

**切分 V100 上的提前借还。** 带 Volta WMMA 格式配对的原生包现在默认可以使用既有的提前补还路径。
所有 stage 必须都用 Volta GPU。一层可以用 IQ3_XXS、IQ3_S、IQ2_S 或 IQ4_XS 的 gate/up 配 IQ4_NL
或 Q2_0 的 down。Q2_0 的 gate/up 与 down 配对也受支持。这些配对可以逐层不同。不受支持的配对保留
普通的运行后补还路径。只有非最后的 stage 可以提前发出它的补还。它的计算流必须已完成一个批量段，
且下一段会归还那份借用：prompt 完成或一个 short-read 窗口皆可。短头窗口之前的检查点切分也符合条
件。后面还跟着另一个批量段的段保留它的临时借用。显式 `STRATA_PREFILL_REFILL_OVERLAP=0`、串行补还
和阻塞式补都会禁用提前补还。最后的 stage 保留普通补还路径。补还拷贝同样的 expert 权重字节。该
stage 在调用补还钩子之前还等 KV-prefix 预取的 DMA。这防止补还在一次 KV 拷贝仍占用中转池时写入该
池。

Prompt 时间，10 个交错对的中位数（关 / auto，ms，`--expert-cache 1500`）：

| 机器 | 512 tokens | 1,000 tokens | 2K / 4K / 16K |
|---|---|---|---|
| RTX 5070, Ryzen 5 7600, Q2_0 | 1,376 / 1,019 (-26%) | 1,788 / 1,392 (-22%) | unchanged |
| RTX 3060, Core Ultra 7 265, IQ3_XXS | 1,620 / 1,159 (-28%) | 2,196 / 1,770 (-19%) | unchanged |
| Tesla P100, Xeon E5-2690 v4, IQ3_XXS | 4,805 / 3,126 (-35%) | 6,821 / 5,085 (-25%) | unchanged |

3,072 个 token 以内、带映射的 expert、8K 预热后读全新 prompt，5 轮交错的中位数（关 / auto，ms；
Ryzen 9 5900XT、96 GB DDR4-3200、Windows 11、`--mmap-experts`、q4_0 KV；此前该份额在这里不拿任何
expert：没有一个被页锁定）：

| 机器 | 512 tokens | 1,000 tokens | 2,000 tokens | 3,000 tokens |
|---|---|---|---|---|
| RTX 5070 Ti (PCIe 3.0 x8), Swift 1.5 IQ3_XXS (huihui-ai's build) | 3,322 / 1,706 (-49%) | 3,990 / 2,357 (-41%) | 5,413 / 3,354 (-38%) | 5,525 / 4,309 (-22%) |
| RTX 3060 + RTX 5070 Ti (layers 0-11 / 12-47, both PCIe 3.0 x8), IQ3_S | 3,452 / 2,037 (-41%) | 4,506 / 2,680 (-41%) | 6,376 / 3,869 (-39%) | 6,733 / 5,138 (-24%) |

两块卡上录制的编码 agent 对话（一个 100K-token 的开头，然后 8 轮 1-5K 个 token 的代码，每轮写
128 个 token，两边读同样的 token）：关 137.4 秒，auto 130.6 秒（读取 121.1 -> 114.3 秒）。

<a id="compact-cpu-activation-download"></a>

## verify 窗口里错过的 expert：尾巴被跳过（CUDA 上默认开，`STRATA_ROUTE_TAIL_SKIP`）

一块卡装不下每个被路由的 expert 时，一个 decode token 的等待时间大多花在不在 VRAM 的那些
expert 上：它们经 PCIe 的拷贝或在 CPU 池上的行。在一个若干 token 的 verify 窗口里，一个**每个**
token 都把它路由到 top 10 最底部（排名 7 或更低，权重最小）的 expert 对哪个 token 都贡献不大。
**0.1.42：CUDA 构建上默认开**：这样的落选 expert 既不拷贝也不计算，它的贡献为零，其他 expert 保持
路由器给它们的权重（不重新归一化）。任何 token 把它排在 7 以上的 expert，以及已在 VRAM 里的每个
expert，照旧服务。

**`STRATA_ROUTE_TAIL_SKIP=0` 关掉它**并逐字节恢复没有它时的构建的回答（已验证：`mg_norepeat` 10
之 10、`exact_pp` 4096 和 20000 个 token 与没有该改动的 0.1.42 相同，在 RTX 3060、单独的 RTX
A4000 和作为 2-GPU 切分、以及 Tesla P100 上）。另一个排名（`STRATA_ROUTE_TAIL_SKIP=8`、`=9`；接
受 1-9）按给定值采用。启动行 `STRATA_ROUTE_TAIL_SKIP=7 is ON (default)` 点名这个退出开关。自
0.1.43 起，同样的默认在 HIP 上对 gfx1200 / gfx1201（RDNA4）成立，规则相同、`STRATA_ROUTE_TAIL_SKIP=0`
的退出开关相同；其他 AMD 部件不变。

适用的地方：单 GPU 或层级切分（每个 stage 自己跳过）的 CUDA 构建上 serve。不适用的地方，也不打印
行：所有 expert 都在 VRAM 时（没有落选），带 `--batch` / `parallel` 槽时（那时一个窗口混合多个请求，
一个请求的跳过会取决于另一个的 token：测量之前保持关闭），带 peer-expert 层或辅助 GPU 时，HIP 和
SYCL 构建上，以及（0.1.42.1，#1884）90% 或更多的 expert 已在 GPU 缓存里时：那里少数落选很便宜，
跳过损失的草稿接受率超过它省下的（一块 Tesla V100 32 GB 配 Coder IQ1_M，expert 94.8% 被缓存，
decode 命中 99.9%：草稿接受率从 70.1% 掉到 68.2%，decode 比 0.1.41 慢 1-2%）。启动行会说明；
`STRATA_ROUTE_TAIL_SKIP=7` 仍然打开它，`STRATA_TAIL_SKIP_MAX_CACHED=<percent>` 移动那条线（101 =
永不）。AMD 层级切分上那条线是 60%（2x R9700 按 16 GB 等效：Q2_0 缓存 73% 时跳过让故事 decode 慢
6.1%，IQ3_S 缓存 45% 时快 7-10%）。激活期间请求日志得到 `route tail skip: N missed experts
skipped`（累计）。

**回答与 0.1.41 略有不同**（`STRATA_ROUTE_TAIL_SKIP=0` 把它们带回来）。decode 路径的 teacher-forced
KL（默认值自己的贪心文本与默认值在 T=0.8 的采样文本，28 个涵盖代码、推理、聊天、六种语言和长文档
摘要的聊天 prompt，各 220 个 token，每组约 6,000 个计分 token，走真实的带 MTP 草稿的 verify 窗口）：

| | RTX 3060 12 GB, IQ3_XXS | RTX A4000 16 GB, Q2_0 |
|---|---|---|
| KL(0.1.41 行为 \|\| tail skip)，均值，贪心 / 采样文本 | 0.0058 / 0.0056 | 0.0032 / 0.0030 |
| p99，最大（贪心） | 0.061, 0.29 | 0.035, 0.32 |
| top-1 token 一致率 | 97.7% / 97.5% | 98.2% / 98.2% |
| 强制文本的困惑度，skip / no skip | x1.011 / x1.007 | x1.005 / x1.003 |
| prompt 之后第一个 token 的 KL | 0.024 / 0.033 | 0.015 / 0.016 |
| 同一引擎跑两次（底线；自适应层和 CPU/GPU 切分依赖计时） | 0.0006-0.0007 | 0.0007 |
| `STRATA_PREFILL_CPU_SHARE` 开 / 关（本来就是默认） | 0.0006-0.0007（首 token 0.002-0.0035） | 0.0005-0.0007（首 token 0.002-0.004） |
| Q2_0 对比 IQ3_XXS，同一文本 | - | 0.1006（p99 1.08，top-1 90.6%） |

<a id="missed-experts-in-a-verify-window"></a>

影响并不均匀：代码和推理 prompt 的 KL 为 0.002-0.004，聊天和非英语 prompt 为 0.004-0.009。
任务检查，贪心，thinking 关（HumanEval 每隔一题取一题，82 道；30 个生成的推理任务），默认 / tail
skip：RTX 3060 76 和 78 / 82 对 78 / 82，推理 27 和 27 / 29；RTX A4000 76 和 76 / 76，推理 27 和
30 / 27。这些运行分不出各设置（各跑一次；3 个推理任务的变化是噪声）。

decode 速度，`STRATA_ROUTE_TAIL_SKIP` 默认值对比 `=0`，全新引擎上 5 轮交错，一个故事、一个代码回
答和一个 6K-token 文档，各 200 个 token，轮内中位数（tok/s 之比，默认除以 `=0`；默认更快的那些
轮）：

| 机器 | 故事 | 代码 | 6K 文档 | 三者合计 |
|---|---|---|---|---|
| RTX 3060 12 GB, IQ3_XXS, resident experts | 1.104 (5/5) | 1.197 (5/5) | 1.241 (5/5) | **1.156 (5/5)** |
| RTX A4000 16 GB alone, Q2_0 | 1.151 (5/5) | 1.135 (5/5) | 1.101 (5/5) | **1.129 (5/5)** |
| Tesla P100 16 GB, IQ3_XXS, `--pcie-frac 0.29` | 1.149 (5/5) | 1.182 (5/5) | 1.118 (5/5) | **1.146 (5/5)** |
| two RTX A4000, layer split (hit rate 99.7%) | 0.987 (0/5) | 1.004 (5/5) | 1.002 (3/5) | 0.998 (1/5) |

增益跟随一个 token 有多少时间花在等待落选 expert 上（该画像下单卡的命中率 75-90%）；一个几乎装着
所有 expert 的 2-GPU 切分（命中率 99.7%）什么也得不到。它不碰 prompt 读取。保持需显式开启、不属于
此改动的开关：`STRATA_ROUTE_PRIOR`（增加速度，KL 翻倍）、`STRATA_ROUTE_RESIDENT` 和
`STRATA_ROUTE_RESIDENT_MTP`（后者不比前者快）。

<a id="the-effect-is-not-even-code-and"></a>

## 落选 expert 的 PCIe 份额按 CPU 的速度设定（CUDA，单 GPU，不带 `--pcie-frac`）

不在 VRAM 里的 expert，一部分经 PCIe 上 GPU，其余由 CPU 池计算；`--pcie-frac` 设定这个份额。0.1.41
之前默认值只来自 PCIe 探测（20 GB/s 起给 0.55），忽略了 CPU 有多快：开着 route tail skip 时，最好的
份额在 RTX 3060 上是 0.15-0.20（默认 0.55），Tesla P100 上 0.20（默认 0.29），RTX A4000 在 0.15 到
0.40 之间是平的。

自 0.1.42 起，单 GPU 且不带 `--pcie-frac` 的 CUDA 上，份额从旧值开始，并在头几个 decode 窗口里用
引擎已经在计时的东西细化：CPU 池每个落选 expert 的墙钟时间。设 `R` = expert 字节 / 该时间（GB/s）、
`L` = 探测到的链路速度，份额为 `L / (L + 1.85 R)`，取整到 0.05 并保持在 0.10 与 0.60 之间；它最多改
三次（48、96 和再 96 个窗口之后，约 10 秒 decode），且一次只动一整档。每次改动打印一行 `PCIe
share:`。一个自己设了 `pcie_frac` 的请求不被打扰。**`--pcie-frac N` 永远更优先；
`STRATA_PCIE_FRAC_DEFAULT=old` 保持 0.1.41 的规则。** 层级切分、`--batch`、peer GPU、所有 expert 都
在 VRAM 时不用，SYCL 上不用，除 gfx1200 / gfx1201（RDNA4，自 0.1.43 起测得的份额就是默认；
`STRATA_PCIE_FRAC_DEFAULT=old` 退出；其他 HIP 部件上 `STRATA_PCIE_FRAC_DEFAULT=auto` 打开它：RX 7900
XTX 未见回退，也未宣称有增益，#2224）之外的 HIP 上不用。规范 Q2_0 包自 0.1.43 起在 CUDA 上用它，
Linux 上默认、Windows 上带 `STRATA_PCIE_AUTO_CANON=1`（那里的 RTX 5070：代码 +2.5%，一个短回答
-3.3%）。常数 1.85 是对三台机器的拟合，所以它落在每台机器最优值的一档之内，而不是正中最优。HIP 上，
`STRATA_PCIE_AUTO_SPLIT=1`（需显式开启，0.1.43）也细化两 stage 的层级切分：池速度读一次，每个 stage
拿它自己探测链路给出的份额（2x R9700 按 16 GB 等效，tail skip 开：Q2_0 故事 +4.5%、代码 +9.9%、4K
+8.8%；IQ3_S 故事 -2.1%、代码 +8.0%、4K +23.3%；三个以上 stage 时流水线循环不跑它）。

实测（整引擎运行的交错对，新默认对比 `STRATA_PCIE_FRAC_DEFAULT=old`，中位数，route tail skip 开）：

| 卡 | 池占用 | 份额（旧 -> 新） | decode，新 / 旧 | 更快的对 |
|---|---|---|---|---|
| RTX 3060 12 GB, IQ3_XXS | 32-37 us per expert | 0.55 -> 0.20 | **1.300** | 5/5 |
| Tesla P100 16 GB, IQ3_XXS | 84-97 us | 0.29 -> 0.20-0.25 | 1.011 | 4/5 |
| RTX A4000 16 GB, Q2_0 | 97-115 us | 0.30 -> 0.30 (unchanged) | 0.999 (same setting) | - |

显式份额的运行与 0.1.41 逐字节相同（`mg_norepeat` 10 之 10，`exact_pp` 4096 和 20000）。带新默认，
回答与旧默认略有不同——份额的任何变化都会这样（CPU 和 GPU 舍入不同）：对旧默认的 teacher-forced KL
在 3060 上是 0.0015（逐次底线 0.0013），A4000 上 0.0011（底线 0.0012；那时份额是 0.25）。

`STRATA_PCIE_BALANCE=1`（需显式开启，CUDA，单 GPU，不带 `--batch`、不带 `--pcie-frac`；#1548，来自
sergiywith / stuchapin909）：与其给每层一个份额，每个 verify 层经 PCIe 读入让两边平衡的落选 expert
数。自旋内核给 GPU 时钟盖戳（计划的到达、PCIe 部分的开始、等 CPU 行），host 给它的池计时，带遗忘的
最小二乘拟合池的 `a + c*n_cpu + d*n_pcie` 和 GPU 的 `g0 + g*n_pcie`；引擎在 `generate` 结束时打印拟
合结果（`pcie model`）。每第 64 个有落选的层把一个 expert 挪到另一边，让两个拟合始终有测量支撑。不
设置时什么都不变（logits 与之前的构建相等）。我们实测（5 个交错对，机器默认开关，IQ3_XXS）：RTX
A4000 arena decode 48.45 -> 51.15 tok/s（+5.6%，5/5 对），RTX 3060 resident 59.12 -> 58.15（-1.6%），
Tesla P100 37.95 -> 37.48（-1.3%）：需显式开启，值得在 A4000 一档的卡上试试。

<a id="the-pcie-share-of-the-missed-experts"></a>

## prompt 分块跟随 host 的内存（Linux，文件层；默认开，`STRATA_PREFILL_HOST_PLAN=0` 关闭）

装不进 VRAM 的 expert 经页缓存从模型文件读取时，prompt 分块的大小决定一次长 prompt 在那批文件上
走几遍，以及每次的工作集有多大（分块的激活缓冲和借出的槽从缓存手里拿走 RAM 和 VRAM）。引擎 0.1.45.3
之前分块一直是今天的选择（大多数卡上 8192）。规划器用它在启动时能量到的东西给 host 定价：RAM 副本及
其借出的槽、剩下的页缓存、盘的类别（从 GGUF 路径读出）和 PCIe 链路，并挑出预测每个 prompt 读字节最
少的分块（以及环形缓冲）。改动选择时打印一行 `chunk 8192 -> 3840, ring 96 -> 96: ...`，不改时打印
`chunk 8192 kept: ...`。装得进今天分块的 prompt 保持一个分块；更长的按不超过规划大小的等大分块读
（结尾不留小尾巴）。没有权重或 expert 改变；只有批处理 prompt 路径的舍入在动（对固定分块的首窗口
logits KL 0.002-0.012，top-1 相同；30 个请求里 30 个文本相同；一次固定分块的运行与自身逐比特相
同）。

适用的地方：页缓存小于今天分块下那批文件的 Linux 文件层。以下情况什么都不做并说 `kept`：GPU 装着每
个 expert 时，那批文件装得进缓存时（owner-class 的 RTX 5070 机器：14.8 GiB 的文件对 41.2 GiB 的缓
存，20K 下 2024 对 2035 tok/s），那批文件永远装不下时（16 GiB cgroup、23.1 GiB 的文件：20K 下 99.9
对 98.9 tok/s，遍数最少的赢），以及在一块更快的 NVMe 上更大的分块即使抖动也获胜时（P2003，保持
8192）。Windows 不带缓冲地读文件层（没有页缓存），所以保持今天的选择。规划按 `--max-context` 个
token 的 prompt 制定（至多 32768，至少 4096）。`STRATA_SSD_GBPS` 覆盖盘速，`STRATA_PREFILL_PLAN_TOKS`
覆盖它假设的 prompt 速度（tokens/s）；`--prefill N` 和 `STRATA_PREFILL_RING` 永远更优先。

实测（RTX 3060 12 GB、IQ2_XS、20 GiB 的 expert 在 RAM、cgroup 34 GB、SATA 速度 emulate 成 450
MB/s、每次运行前丢页缓存并起新引擎、5 个交错对，规划关对比规划器的 `8192 -> 3840`，中位数）：

| prompt (tokens) | plan off, tok/s | planner, tok/s | ratio | pairs faster | drive read, GB (off -> planner) | decode after, tok/s |
|---|---|---|---|---|---|---|
| 4,101 | 138.1 | 138.2 | 1.001 (one chunk either way) | 3/5 (tie) | 12.8 -> 12.7 | 27.0 -> 28.2 |
| 20,400 | 334.0 | 822.3 | **2.46x** | 5/5 | 26.1 -> 2.7 | 25.4 -> 42.6 |
| 28,887 | 418.8 | 910.4 | **2.17x** | 5/5 | 29.5 -> 0.6 | 28.6 -> 45.7 |

同一台机器用它的真实 NVMe 和 32 GB 上限再次选 8192（更大分块在 3328 时 480 对 327 tok/s，即使它抖
动），16 GiB 上限不变。恒等门（固定开关、`--expert-cache 1500`）：规划行说 `kept`。host 模型在
`include/strata/program/prefill_plan.hpp`，配 `tests/program/prefill_plan_test.cpp`。

<a id="the-prompt-chunk-follows-the-host"></a>

## 为 0.1.41 测量的更多可选开关（全部默认关，除非你设置）

这些都不改变默认的回答；每一个都对照默认值用整引擎运行的关/开交错对测量（每边一个新引擎、预热、三
个固定 prompt 上 200-token 的贪心回答，中位数；"更快的对"数的是开启臂获胜的那些对）。把它们放在这
里，是为了让你能在自己的卡上试。

- **`STRATA_GDN_CHUNKED=1`：prompt 的 DeltaNet 递归，需显式开启**。CUDA 上它以 32 个 token 为一块
  运行（sergiywith，#1372；WY 形式，FP32，与默认有其他比特：对默认的首 token KL 平均 0.002-0.009
  nats，每个试过的 prompt 上 top token 相同）。内核需要一个 wave 里 128 个 SM，所以它自己只在 128
  个 SM 以上的卡上跑（RTX 5090：递归 1.8 倍）。更小的卡上 `STRATA_GDN_CHUNKED=2` 为测量强制启用，在
  那里更慢：RTX 3060（28 SM）0.73 倍、RTX 5070（48 SM）0.87 倍，2K-32K 个 token（`gdn_rec_parity
  --bench`），且 prompt 时间没有改善：RTX 5070 2K 1.01 倍、8K 1.00 倍于默认时间；RTX 3060 2K、8K、
  16K 1.01 倍，没有更快的对。HIP 上，gfx1100-gfx1102、gfx1150 和 gfx1151 用 16-token 的 WMMA 块配
  FP16 操作数和一个融合的出口 norm；其他卡和短块保留现有路径。它也改变舍入（对默认的首 token KL 平
  均 0.006-0.007，12 个 prompt 里 12 个 top token 相同）；Radeon 8060S（gfx1151）上该层的递归快 1.6
  倍，预热 prompt 之后一个 4K-128K 的 prompt 快 1.4-2.1%；见
  [HIP 的条件与数值检查](AMD_HIP.md#opt-in-chunked-prompt-gdn-on-gfx11)。
- **`STRATA_FS_SLOTS=N`：Foresight 交换空间**（q8atnight，#1348）：每层 N 个 VRAM 槽，用一个拷贝流
  以刚刚落选的 expert 和下一层路由器预测的 expert 补满（只要落选的用 `STRATA_FS_AHEAD=0`），让缓存缺
  的 expert 能在 GPU 上计算而不是交给 CPU。模型自己的路由器决定一切；槽只保存同样字节的副本
  （`STRATA_FS_VERIFY=1` 把每个落地的槽读回并比对：RTX 5070 上 2,300+ 次读取 0 次不符）。槽从 expert
  缓存本会拿的 VRAM 里出（把 `--vram-reserve-mib` 提高约 N x 80 MiB），而每台测过的机器上更小的缓存
  损失都超过槽的回报：decode 4 槽，tok/s 中位数 B/A（更快的对）：RTX 3060 只要落选时 0.95（0/6）、带
  前瞻 0.91（0/6）；Tesla P100 0.97（2/6）和 1.00（4/6）；P100 上 16 槽 0.96（0/6）；RTX 5070 0.98
  （1/6）。报告者在 2x RTX 3090 上测得相同（那里每层落选数低于一个，CPU 往返仍在）。不予采用。
- **`STRATA_STAGE_PIN`：页锁定的中转缓冲**（Zhong Uncle，#1237；**需显式开启，`STRATA_STAGE_PIN=1`**。
  它在开发期默认开，但 0.1.41 的发布门槛发现带着它、4096-token prompt 之后 IQ3_S 的 decode 损坏，疑似
  一个中转缓冲在它异步拷贝完成之前被回收；修复之前保持关闭）。expert 由 GGUF 就地服务的地方
  （`--mmap-experts`，或 RAM 不够放 arena），缓存填充的拷贝源缓冲被页锁定，于是拷贝不再走驱动的bounce
  缓冲（约 0.4-0.6 GiB 页锁定 RAM，驱动拒绝时一个缓冲退回可分页）。decode，开 / 关的中位 tok/s，整轮
  运行 6 对：Tesla P100 RAM 限到 24 GB（cgroup）13.9 -> 15.0（+8%，6/6 对更快）；RTX 5070
  `--mmap-experts` +1.2%（5/6）；RTX 3060 `--mmap-experts` -0.1%（5/6）。把所有 expert 放在 RAM 的机器
  从不用中转缓冲。
- **`STRATA_ADAPT_LAG=2`：自适应层的拷贝晚一个窗口再等**（#764）。decode，6 对：Tesla P100（PCIe 3.0
  x16）+3.5%（6/6 对更快），RTX 5070 +0.2%（4/6），RTX 3060 -1.3%（0/6），所以保持需显式开启。层不等
  它的拷贝的地方（`STRATA_ADAPT_NOWAIT=1`、`STRATA_ADAPT_LAG` 大于 1），设备的常驻表现在每个窗口之前
  跟随每次逐出（每次 verifier 启动前一个小补丁内核；#1517，来自 sergiywith）：此前，池可能用一个过时
  的激活在 CPU 上计算一个已被逐出的 expert（`STRATA_ADAPT_CHECKRES=1` 为这样的层计数；
  `STRATA_ADAPT_EVICT_SYNC=0` 关掉修复）。默认（层等待拷贝）不动。
- **`STRATA_ADAPT_OVERLAP`（`=1` 打开，所有后端；启动日志说明状态，跑不了时说明原因）：自适应层的
  拷贝在下一个窗口运行时落地，token 相同。**窗口在与 #463 的等待相同的时刻取用交换（所以每个窗口在
  GPU 和 CPU 上跑的是同样的 expert），但拷贝按层序进行，verify 窗口把一层的计划扣到该层的拷贝落地，而
  不是整个窗口等一整轮（A4000、PCIe 3.0 x16 上约每四个窗口 10 ms）。单 GPU、整个 arena 在 RAM、不带
  `--adapt-async`、`--pipeline-windows`、`STRATA_ADAPT_LAG`、`STRATA_ADAPT_NOWAIT` 或
  `STRATA_DMA_BATCH`。层级切分（两块以上 GPU）任何版本都不跑它：启动日志说 `off: not on a layer
  split`（0.1.43 只在设了变量时打印那行，而在切分上本来就没跑 overlap，所以 0.1.43.1 的需显式开启默认
  在那里什么都不改；2x A4000 实测，#2281）。自 0.1.43 起 Linux 的 CUDA 上默认开，Windows 和 HIP 上需
  显式开启；自 #2010 起所有后端都需显式开启，此前一块 RTX 5090 Laptop 带着它丢了 21% 的 decode。每台机
  器上它的价值：RTX A4000（各 6 个交错对，decode 中位 tok/s：Q2_0 16 GB 62.5 -> 66.2（+5.9%）、当 12 GB
  卡用的 Q2_0 51.3 -> 54.2（+5.7%）、IQ3_XXS 16 GB 48.6 -> 52.6（+8.3%）、当 12 GB 用的 IQ3_XXS 39.3 ->
  41.5（+5.5%），每次 6/6 对更快）和 Tesla P100（+4.4%）有增益，但 Windows 上的 RTX 5070 在 Q2_0 上丢
  2.6%（IQ3_XXS 没变快），一块 RTX 5090 Laptop（WSL2、Core Ultra 9 275HX、原生 IQ3_XXS 包、
  `--expert-cache-per-layer`）丢 21%：同一个构建的 5 次 256-token 生成，中位数 84.6 对 106.6 tok/s，快
  的那组最慢的一次在慢的那组最快的一次之上（#2010）。
- **`STRATA_ADAPT_CALLBACK_WARMUP`（Linux，自 0.1.43.1 起默认开；=0 关闭）：在 CPU 池锁定 host 线程之
  前启动 GPU host-callback 线程。**较晚创建的 callback 线程可能继承 host 的单 CPU 亲和性，与自旋等信号
  的 verifier 抢核心（#1975）。预热在构造池之前运行并同步一个空操作的 host callback；交换选择、按层
  callback 和常驻表更新保持不变。RTX 5090 / Ryzen 9 9950X 上，配对的 4K 三 prompt 测试平均 142.82 ->
  207.75 tok/s。其他 GPU 和驱动未测；见[复现方法、设置与限制](../../bench/results/2026-10-11-rtx-5090-callback-affinity/README.md)。
- **`STRATA_ADAPT_SIGNAL=host`：overlap 的按层"已落地"信号用 host 函数**（0.1.43 的做法）。自 0.1.43.1
  起默认是一次 4 字节、流序的拷贝进页锁定内存，不需要驱动 callback 线程：一台 20 核台式机（RTX 3060）
  上，即使线程早启动，host 函数仍要每个窗口约 7 ms，因为池的工作者占满了其余 CPU。**Windows 上 host
  函数保持默认**（0.1.44，#2188）：WDDM 下拷贝及其信号卡在驱动的 command buffer 里等，窗口在页锁定的
  int 上空转，第一个请求停在 "the adaptive tier's copies for layer 45 never landed"（RTX 4070 SUPER，驱
  动 617.42）。那里 `STRATA_ADAPT_SIGNAL=copy` 选用拷贝；引擎现在在排好交换队列后 flush 拷贝流，这本应
  修复它，但还没有在 Windows 上跑过。
- **`--prefill-head-alias`（`--serve`，单 GPU）：读 prompt 期间，输出头的 VRAM 装着 prompt 缓冲**，第
  一个生成 token 之前把头读回来。它没有按上面的配对方法测过。一块 8 GB RTX PRO 2000 Laptop（Windows、
  IQ3_XXS、590 MiB 预留）上，它让 prompt 路径用 2,048-token 分块而不是 512：一个 29,854-token 的
  prompt 上 prefill 292.5 -> 667.1 tok/s（12 次和 6 次新引擎试验的中位数）。读取至多 512 个新 token 每
  个请求约多花 0.6-0.75 秒，而 decode 时那张卡 0 MiB 空闲。细节、限制与测量：
  [PREFILL_HEAD_ALIAS.md](PREFILL_HEAD_ALIAS.md)。
  `STRATA_ADAPT_NOWAIT` 或 `STRATA_DMA_BATCH`；否则日志说明它为什么是关的。RTX A4000，各 6 个交错对，
  decode 中位 tok/s：Q2_0 16 GB 62.5 -> 66.2（+5.9%）、当 12 GB 卡用的 Q2_0 51.3 -> 54.2（+5.7%）、
  IQ3_XXS 16 GB 48.6 -> 52.6（+8.3%）、当 12 GB 用的 IQ3_XXS 39.3 -> 41.5（+5.5%），每次 6/6 对更快。
- **`STRATA_MMQ_DIRECT=1`：prompt 的 MMQ 就地读缓存槽里的常驻 expert**（CUDA，原生包；需要以
  `-DSTRATA_MMQ_TAB=ON` 构建的引擎，默认不编）。prompt 路径用 llama.cpp 的 MMQ 以 16 个一组乘它的
  expert，而 MMQ 要求一组的权重在一个固定步长的缓冲区里，所以一个分块路由到的每个 expert 都先被拷到那
  里（每个 2.3-3.1 MB，每分块每层），常驻的也一样。开着它时，Strata 侧的 MMQ 外层调度器副本
  （`src/prefill/moe_mmq_tab.cuh`）改为每个 expert 拿一个指针，常驻的 expert 就地读；流式的、peer
  GPU 的和其他 expert 仍被拷贝。启动（tile 大小、stream-k 切分、fixup）与 MMQ 为拷贝好的一组做的相同，
  tile 代码也是 llama.cpp 自己的，所以答案是相同的比特（`prefill_mmq_tab_test`：874 个乘积与拷贝路径逐
  比特相等，九种类型全——Q2_0、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q8_0——以及每
  种 tile 大小；可选的 Q5_0 和 K-quant expert 拿不到指针表、保留拷贝）。每个定尺寸的缓存槽在其 expert 后
  面留 256 字节，因为 MMQ 会多读一个 640 宽 down 矩阵之外至多 136 字节；它们在每组之前清零。约 9,300 个
  槽共多 2.3 MiB（一块 RTX 3090 上 6,998 个槽少一个）。`STRATA_MMQ_DIRECT=2` 保留那个布局但照旧拷贝（一
  个 A/B 参考）；0 或不设是默认路径，逐字节。`STRATA_MMQ_DIRECT_LOG=1` 在每个 prompt 之后把每个 stage
  就地读的 expert 数加进日志。HIP 下、不带 `STRATA_MMQ_TAB` 的构建上、或 CMake 关掉了该选项时（llama.cpp
  树的 `mmq.cuh` 与副本镜像的那份不同时会关并给出消息），该变量被忽略。
  **为什么构建选项默认关：** 引擎在启动时加载每个内核（`CUDA_MODULE_LOADING=EAGER`），九个指针表实例（
  +15 MB 二进制）在每块 GPU（sm_86）上拿 36-38 MiB VRAM，变量没设也一样——那是每个模型都要失去的 expert
  槽，如同 `STRATA_MMQ_KQUANTS`。它们还占上游 MMQ 实例每个 CUDA arch 编译时间约 70%（sm_86 上 175 秒
  CPU）。不带该选项时，设备代码与没有该功能的构建逐字节相同。
  4x RTX 3090、UD-IQ4_XS、切分 `12,24,36` 配 `--trim-stage-weights` 和 `--batch 16`（每个 expert 都常
  驻）、直接驱动引擎、其 "read in" 时间、8 次中位数实测：170-token 的 prompt 331 -> 251 ms（-24%）和第
  二组 369 -> 290 ms（-21%），1,217-token 的 prompt 773 -> 634 ms（-18%）和 838 -> 675 ms（-19%）。一个
  干脆跳过拷贝的 hack 跑到 225 和 577 ms，但它不是精确实现的上界：它改变了路由（一个 prompt 上 12,003
  个 expert-层条目掉到 777）所以做的工作更少。一块 RTX 3090、多数 expert 从 RAM 流式读取时，流式 expert
  的拷贝仍在，prompt 时间没有动（1.5% 以内）。精确性，带 `STRATA_PREFILL_CPU_SHARE=0` 且无自适应交换：
  prompt 之后第一个 decode 窗口的 logits（`STRATA_DUMP_FIRST_LOGITS`，fp32）和 32 个贪心 token 在 2 与
  1（和 0）之间逐字节相同——180-2,619 个 token 的 prompt，在 4-GPU 切分带 `--prefill 2048`（1,024 个
  token 及以上的分块：流式走查）、每 GPU 少 9 GB expert 缓存的同样切分（20-31% 的 expert 读取流式化，所
  以组里混合就地、拷贝和组中刷新的 expert）、以及单 GPU 带 `--expert-cache-per-layer` 上都如此；
  `tools/batch_test.py --n 16` 每个 prompt 的 150 个贪心 token（单独和批量）也相同，且与没有该选项的构
  建相同。统一缓存（无 profile）在原生包的 prompt 读取时不持有常驻 expert（serve 需要 profile，而一次性
  运行只在 decode 时填缓存），所以 `expert_cache_slot_guard_test` 覆盖它。

<a id="more-opt-in-switches-measured-for"></a>

## 线程摆放：`--aux-cpus`（Linux，默认关）

CPU 池锁定它的工作者（每个物理核一个逻辑 CPU），host 线程把自己锁到池保留的那核。引擎在那之后启动的
线程继承 host 的那一个 CPU：自适应层的工作线程、prefill 助手、路由器前瞻。锁定之前创建的 CUDA 驱动线
程可以在任何允许的 CPU 上跑，包括工作者的；而一个惰性创建的 host-callback 线程可能反而继承已锁定的
host 的那一个 CPU（见上面的预热）。`--aux-cpus` 把每个既非池工作者也非 host 线程的线程放到备用 CPU。
它只移动线程，从不改变结果。

- **哪些 CPU。** `--aux-cpus auto`（或 `STRATA_AUX_CPUS=auto`）：先拿 host 所在核的 SMT 兄弟，再拿没有
  池工作者的物理核的 CPU。与工作者共享一个核的 CPU 永不取用；没有备用 CPU 时该功能什么都不做，启动行
  会说明。一个列表（`--aux-cpus 24,26` 或 `24-27`）按给定值使用，减去 host 的 CPU 和工作者们的 CPU，且
  只用进程可以运行的 CPU。`--aux-cpus off` 是默认。
- **怎么做。** 引擎启动的线程第一件事就是自锁：自适应层的工作线程及其每轮线程、prefill 拷贝线程和助
  手、路由器前瞻、PLE 读取者、文件读取者、stdin 读取者和看门狗。别人启动的线程（CUDA 驱动的）在池存在
  之后、经一次 `/proc/self/task` 扫描被移动，并在每个请求开始时再扫一次。池的工作者和 host 线程永不移
  动，被任何人锁到一个 CPU 的线程也不动。显式的 `STRATA_ADAPT_JOB_CPU` 对自适应层的工作线程仍然更有优
  先权。`STRATA_AUX_STAGER=0` 让 prefill 拷贝线程留在它们被启动的地方：只剩一个备用 CPU 时，至多 32 个
  （`STRATA_STAGER_THREADS`）会共享那个 CPU，可能拖慢长 prompt 的读取。
- **按请求。** `"strata_tune": {"aux_cpus": 1}`（`serve/server.py` 转发 `1`、`0`、`true` 和 `false`）把
  线程移进那个集合；`0` 把每个被移动的线程放回它原来的摆放。不带该键的请求回到启动设置。带 `--batch`
  时，最近被接纳的那个请求的值生效，与其他 `strata_tune` 键相同。不带 `--aux-cpus` 启动的引擎照样规划那
  个集合，所以 `1` 取 `auto`。
- **它打印什么。** 启动时：`strata aux cpus: threads that are neither pool workers nor the host go to
  CPUs 24 (on; N threads placed so far)`，或 `(off at start; a request's aux_cpus=1 turns it on)`。一个改
  变状态的请求打印 `strata serve: aux cpus ON/off for this request`。
- **不覆盖。** 其他进程（Python 启动器、容器运行时）以及引擎的 host 线程。
- **实测。** 一台机器：2x RTX 3080 20 GB @ 220 W、Xeon E5-2696 v4（容器内 24 个逻辑 CPU，16 个池工作者
  各占一个 CPU），UD-Q4_K_XL，层级切分 23，`--batch-mtp`，两个槽，自适应层开。`auto` 把 host 线程放到
  CPU 2、助手线程放到 CPU 24。每条臂同一个二进制，只差这个开关。两对整引擎重启，顺序 `auto`、off、
  `auto`、off（每次重启 12 次单独 decode 和 10 轮双流，中位数）：单独 decode 带 `auto` 为 79.7 和 80.4
  tok/s，关时 72.1 和 72.4（off / `auto` = 0.904 和 0.900），双流合计 90.8 和 90.7 对 80.8 和 82.8
  （0.890 和 0.912）。同一配置重启两次的自身漂移：单独 1.009 和 1.005，双流 0.999 和 1.024。单引擎上活
  切换、按 `strata_tune` 逐请求并轮换顺序（每臂 20 个单独请求、16 轮双流）：off / on 单独为 0.888、双流
  为 0.917，同一设置的 A/A 对照为 0.985 和 0.990。约十分钟基准里 host 线程的非自愿上下文切换：开关关时
  43.8k 和 42.8k，`auto` 时 3.9k 和 2.8k。prompt 读取（24K、25K、51K、104K）不依赖这个开关：关对比
  `auto` 为 +0.7% 到 +1.5%，在重启噪声之内。门铃到标志时间的均值与 p99 不带开关并不更差；出现的是罕见
  的百微秒以上停滞，关时 12 个请求里有 7 和 8 个，`auto` 时 0 或 1 个。host 被抢占是损失的原因是推断；
  计数器没有证明它。每次运行的预热请求已被丢弃。

<a id="thread-placement-aux-cpus-linux-off"></a>

---

## 实验性速度投影（EXPERIMENTAL，默认关）

**这是一个实验，不是完成的功能。** 它随 Strata 发行，但默认关，除非你打开它。

一个给 Qwen3.8-Flash-Next 的 480 KB 控制向量（`data/experimental-speed-projection/`，见它的 README）。
在第 4-44 层每层之后，引擎从残差的每条 hyper-connection 流里去掉一个方向：`h -= (h . v) v`，每层一个单
位向量 `v`，与 llama.cpp 对包的 `--cvec-mode project` 补丁的做法完全相同。

**它改变什么。** 该向量自己的包把它描述为一个**拒绝方向投影**：带着它，模型拒绝的请求大大减少（它的
测试集上报告 50 之 1 对 50 之 50），而移除拒绝就是移除一种安全行为——开着它时模型写出什么由你负责。它
也让普通回答略有偏移（下面测量）。它不是引擎里的优化：同一文本上每个 token 多花 0.2-0.4%。一个聊天的
tok/s 带着它表现如何取决于模型写的文本（长度、重复、草稿命中好坏），所以在自己的 prompt 上测；Monitor
把每个请求标为 ESP 或 stock。

**打开它（在 setup 时）。** `START-HERE.bat --setup` 会问 "Turn on the experimental speed projection?"
（默认：否），或传 `--experimental-speed-projection on`（`off`，或另一个向量 GGUF 的路径）。只适用于原版
Qwen3.8-Flash-Next，不适用于 Swift 1.5。它把这些引擎开关（llama.cpp 的）写进 `strata-<model>.json`：

```
--control-vector-scaled <Strata>\data\experimental-speed-projection\Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

引擎日志随后说 `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`，网页应用的
About 标签页列出它。（`--cvec-mode add` 是 llama.cpp 的标配加法模式，给加法向量用。）

<a id="experimental-speed-projection-experimental-off-by-default"></a>

**按请求。** 一个已加载的向量对每个请求都开着，除非它另有说明：网页应用的 Sampling 抽屉里有一个
开关，API 在请求体里接受 `"experimental_speed_projection": false`（OpenAI 与 Anthropic；配置默认值
放进 `"sampling": {"experimental_speed_projection": false}`）。切换会一次性丢弃对话缓存，因为模型状态
是按另一种方式算出来的。关掉时，输出与 stock 模型逐 token 相同。

**此处实测**（Q2_0、固定 expert、2,557 个 teacher-forced token 的代码、一篇文档和一段聊天）：top-1
token 在 10% 的位置上变化，对 stock 模型的平均 KL 0.063 nats（最大 4.1），困惑度代码 +15%、文档
+2.3%、聊天 +0.4%。细节：`bench/results/2026-09-27-esp/`。

---

## 疑难解答

<a id="troubleshooting"></a>

| 症状 | 怎么办 |
| --- | --- |
| `the NVIDIA driver is too old` | 更新驱动（NVIDIA App 或 nvidia.com/drivers），重启，再运行 `START-HERE.bat`。 |
| Python 或构建工具装不上 | 装上它点名的东西（会打印链接），然后再运行。已经做完的都保留。 |
| `port 8080 is already in use` | Strata 已经在跑（找它的窗口），或别的程序占着这个端口：`START-HERE.bat --port 8081`。 |
| 日志里 `cudaHostRegister ... out of memory` | Windows 上正常：引擎改为按层分片锁定 expert。只有随后加载失败才算问题。 |
| `ExpertCache: cudaMalloc(...) failed: out of memory` 但 VRAM 明明空闲 | Windows 的页面文件被关掉或太小：显卡上的每次分配同时记入 Windows 的提交量（RAM + 页面文件）。设一个固定大小的页面文件，初始与最大相同（例如 65536 MB：系统 > 关于 > 高级系统设置 > 性能 > 高级 > 虚拟内存 > 自定义大小），并重启：一个由 Windows 按需增长的页面文件（"系统管理的大小"，或初始值低于最大值）在给显卡内存记账时可能来不及增长（#60："系统管理的大小"和 4096-32768 MB 仍然失败，固定 64 GB 成功）。自 0.1.19 起引擎改用更小的缓存重试而不是停止，setup 会对小于 4 GB 的页面文件发出警告（issue #60）。 |
| 第一次启动要几分钟 | 它正把 34-55 GB 读进 RAM；文件还在操作系统缓存里时第二次启动更快。 |
| PC 在启动时卡死几分钟 | 正常，第一次最明显（服务器窗口会在发生时说明）：引擎把 expert 装进 RAM、把一部分锁给 GPU 并给 expert 缓存定大小。等着；别关窗口。10 分钟后还卡着：重启 PC、关掉其他程序再试，或选一个小点的尺寸。 |
| `the engine stopped unexpectedly (exit code ...)` | 引擎进程在回答中途结束了——通常是 RAM 耗尽（Linux 会结束最大的程序：`sudo dmesg \| grep -i -E 'killed process\|out of memory'`）。下一个请求自己把它再启动。反复出现：关掉其他程序或选小一点的尺寸。模型 expert 之外只剩约 6 GB RAM 给其他一切时，服务器也会在启动时警告。 |
| 输出慢、硬盘灯狂闪 | 空闲 RAM 不够：关掉其他程序，或选 Q2_0 / IQ2_XS。 |
| `prompt ... exceeds the context` | 请求比你选的上下文长：用更大的 `--context` 重跑 setup。 |
| `the setup refuses --rope-scaling none for a past-trained context` | 超过训练的 262,144 的上下文需要重标旋转角度（实验性 rope 缩放），setup 不会在那个范围外按原始角度配置。让它自己选（`START-HERE.bat --setup --context 393216` 会加 yarn 和一个覆盖性的因子），或自己传 `--rope-scaling linear` 或 `yarn`。 |
| 比表格慢 | 显示器插在 GPU 上、以及其他 GPU 程序都从 expert 缓存手里拿走 VRAM；RAM 跑在低于标称的频率（在 BIOS 里启用 EXPO/XMP）会让 CPU 那一半变慢。 |
| `this server was started without the vision encoder` | 模型是按纯文本配置的：用 `--vision gpu` 重跑 setup。 |
| 移动或重装 Strata 之后 setup 把模型放在旧盘上（#1070） | setup 在其设置文件里记住你上次用的数据文件夹（Windows 上 `%APPDATA%\Strata\settings.json`，Linux 上 `~/.config/strata/settings.json`；`data_dir` 条目）并再次使用它。用 `--data-dir DIR`（模型文件、包、MTP 层）或 `--models-dir DIR`（仅 GGUF 文件）换一个位置，或删掉 `data_dir` 行；`--gguf-dir DIR` 直接使用你已有的 GGUF 文件而不拷贝。要搬一个模型，拷贝它的文件夹并用新的 `--data-dir` 跑 setup。 |
| 图片被拒或 `cannot read the image` | 该文件不是 Pillow 打得开的图片（JPEG、PNG、WebP、GIF、BMP、TIFF、AVIF 可用）。 |
| 图片很慢（3-30 秒） | 编码器跑在 CPU 上，这是 24 GB 以下图形内存的默认（显卡读取者是 24 GB 起的默认）。要用显卡读取者：用 `--vision gpu` 重跑 setup（需要约 1.4 GB VRAM；它从 expert 缓存拿空间）。 |
| 一个请求永不结束："reading the prompt"，GPU "100%" 但功耗很低 | GPU 的 VRAM 用尽（0.1.9 之前的引擎在大上下文下可能在剩约 30 MiB 时终止）。运行一次 `START-HERE.bat` 得到 0.1.9 或更新的引擎；日志随后会说 `... MiB of VRAM free with everything loaded`（几百），偏低时点出该加的 `--vram-reserve-mib`。 |
| 输出远慢于上面的表格，且只在大 `--context` 时，启动时说一个很小的 `... MiB of VRAM free with everything loaded` | 该上下文下 expert 缓存装不下这张卡。`--expert-cache auto` 时引擎会缩小它到装得下；固定数字时槽写好后不再检查，而 Windows 上超出的部分被放进系统内存而不是失败，所以只有速度能看出来。用 `auto`、更小的数字，或提高 `--vram-reserve-mib`（它在缓存定大小**之前**扣除）。见"与其他程序共享 GPU"下的"expert 缓存大小是一份预算"。 |
| 生成在回答中途停下，GPU "100%"，一个 CPU 核满载 | 引擎 0.1.12 修复（issue #29，大 VRAM 卡上 CPU expert 池的一个竞态）。自那以后一个停止不动的请求以错误结束而不是挂起（2 分钟后；0.1.13 起 1 分钟）：日志说 `no progress for ... s ... (issue #29)` 并带停下的位置，下一个请求把引擎再启动。看到这行请带着它开一个 issue。引擎 0.1.13 在其下增加一份停滞报告（每个 expert 池线程和 GPU 握手在做什么、内存和缺页），Windows 上还有带每个线程调用栈的 `strata-stall-<pid>.dmp` 文件：两个都附上。（`STRATA_WATCHDOG_S` 以秒设置该时间；0 关闭。）引擎 0.1.14 修复了这些报告找到的停滞（issue #31：IQ 系列包下，host 可能在 NVIDIA 驱动里永远等待，在 verify 窗口拷贝 expert 时；expert 现在由一个 GPU 内核拷贝，`--pcie-mode dma` 恢复旧方式）。 |
| Linux 上 `no progress for 60 s ... reading the prompt`，停滞报告说 `threads waiting on the disk (state D): 16 ...` | 引擎在等盘，不是死锁：n-gram 表是随机读（`--ple-io direct`），机械盘跟不上（#605）。表在机械盘上时引擎会在启动时警告；`--ple-io ram`（Linux，需要 RAM 放表）或把模型放 SSD 能修好。机械盘上 RAM 装得下表时 setup 自己加 `--ple-io ram`（0.1.39）。 |
| `the engine said nothing for ... s during the request` 或 `... did not finish the request after it was stopped (STOP)` | Issue #481：引擎和服务器失步（引擎等它的下一条命令，服务器等请求结束；GPU 0%，日志里什么都没有）。服务器在请求期间 300 秒没有引擎的一行输出时结束引擎（读 prompt 时：每个分块最多可花上一个分块三倍的时间，第一个最多按其 token 数以 50 tok/s 计），请求以错误结束，下一个请求把引擎再启动。`strata-<model>.json` 里的 `"engine_silence_s": 600` 设置该时间（0 = 永远等，像之前）。如果你遇到它，请把引擎日志的结尾加到 #481。 |
| `the engine said nothing for ... s and used no CPU or disk in that time (frozen ...)` | Issue #1317：给一个不是慢而是彻底停住的引擎（CUDA 调用内部的死锁、驱动 stall）准备的更快的上述检查版本：90 秒没有一行输出，且引擎进程期间也没用 CPU 时间、没移动磁盘字节时，服务器结束它，下一个请求把它再启动。一个安静但仍在干活的引擎永远不会被这个检查结束，服务器打印一条说明（`... but is still working; it is not ended`）并交给 `engine_silence_s`。`STRATA_ENGINE_STALL_S=180` 设置时间，`0` 关闭该检查（它需要 `psutil`，setup 会装）。 |
| 日志里 `out of memory: cudaFuncSetAttribute`（IQ3_XXS，长 prompt） | 引擎 0.1.15 修复：CUDA 在第一次需要一个内核的代码时才加载它，而 prompt 中途已没有 VRAM 给它。运行一次 `START-HERE.bat`（Windows）或 `./setup.sh`（Linux）更新。 |
| 桌面应用或 Terminal 说找不到引擎，或说某个模型需要更新的引擎；Mac 说应用"已损坏" | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-desktop-app-and-the-terminal)。 |
| 其他任何事 | 引擎日志是本文件夹里的 `strata-<model>.log`。应用里：Settings、Engine、Diagnostics、View logs、Save logs。Terminal 里：`strata doctor`。 |

---

## 工作原理

<a id="how-it-works"></a>

<p align="center"><img src="../paper/tiers.svg" width="760" alt="memory tiers"></p>

- **GPU（VRAM）：** attention 与 DeltaNet mixer、gated-residual 权重、路由器、共享 expert、输出头、MTP
  草稿层、KV 缓存（从 64K 起：只放其最常读的部分，其余从 RAM 流式），以及一个用最多用到的 expert 填满
  VRAM 剩余空间的 **expert 缓存**（它随你聊天的过程适应当前对话）。
- **RAM：** 全部 24,576 个 expert，页锁定。CPU **就地**计算不在 GPU 上的 expert，与 GPU 算缓存里的那些
  同时进行（AVX-512 / AVX2 内核，i-quant 用 ggml 的）。
- **SSD：** 28.8 GB 的 n-gram 表，每 token 读几行，不带缓冲、绕过操作系统缓存读取（`--ple-io direct`，
  默认，为 SSD 做的；机械盘上 `--ple-io ram` 把表留在 RAM，#605）。直接读测得慢（低于 15K 行/秒）时启动
  探测会警告，但从不自己切换（自 0.1.43.1；`STRATA_PLE_AUTO_MMAP=1` 恢复 0.1.42 切换到 `--ple-io mmap` 的
  行为）：那条线是为 NVMe 标定的，SATA SSD 在 direct 的 16 个在途读取下最多 10-14K 行/秒，而一块这样的盘
  上（Linux，77.8K-token prompt）mmap 读 2.1K 行/秒，direct 是 14.4K（#2040）；另一台 Windows 机器上映射
  文件很慢，mmap 让 prompt 停滞（#1984）。引擎按 GGUF 里表的格式读它：IQ4_NL（默认表）、Q4_0、Q5_0、
  Q5_1、Q8_0、FP8（带 scale 的 E4M3）或 BF16。对 4,000 个随机行、对照 checkpoint 自己的 BF16 表实测，每
  行平均误差 Q8_0 0.53%、FP8 2.64%、Q5_1 3.78%、Q5_0 4.25%、IQ4_NL 7.60%、Q4_1 7.80%、Q4_0 8.55%（Q8_0
  的表 54 GB，IQ4_NL 的 28.8 GB）。Q2_0 模型、700 个 teacher-forced token，对 BF16 表的平均 KL 为 0.0117
  （Q5_0）、0.0121（Q5_1）、0.0125（Q8_0）、0.0128（FP8）、0.0140（Q4_0）、0.0146（IQ4_NL），困惑度两种
  情况都在 BF16 的 1% 以内：对表的任何改动都让 2-bit 模型移动约 0.012，所以这些格式很难分出高下。Setup
  不提供别的表；它只决定给引擎哪个 GGUF。
- **投机：** 模型自己的 MTP 层至多草拟 3 个 token；一遍走完全部 48 层检查它们。平均每次走查 2.4-3.2
  个 token。回答重复上下文时（代码编辑、引用文本），**prompt 查找**（引擎 0.1.7）从更早的副本至多草拟
  5 个 token，但只在实测的接受率与成本说划算的地方：代码编辑快 6-11%，其他文本不变。草稿的检查方式与
  MTP 的相同，所以输出相同。
- **Prompt** 以至多 8,192 个 token 的分块处理（`--prefill auto`；32,768 需显式开启），expert 经 PCIe
  流式到 GPU。自引擎 0.1.39b（#583）起，流式环形缓冲按包定字节大小（blob 更大的原生包拿更少的环槽、保住
  更多缓存槽），`--prefill auto` 挑能让那个环保持满的最大分块。它只在 0.1.39 的规则把分块压在 4,096 个
  token 及以下的地方让出环槽；从 6,144 起它保持 0.1.39 的环、只在旁边寻找更大的分块（在那里缩小环形测得
  更慢：Coder -12%），而装得进 0.1.39 分块的 prompt 保持 0.1.39 的环（一个分块：更小的环只会拖慢它）。
  RTX 5070，32K prompt 对比 `STRATA_RING_BYTES=0`，2-3 轮交错：1,500 槽缓存下 IQ3_XXS +18.5%，另一个内存
  状态下 +9%，Coder +3%，IQ3_S 不变；4K prompt 不变（-0.7% 到 +1.1%，逐比特相同）。Q2_0（固定
  `--prefill 2048`）不受影响。它让长 prompt 的比特与 0.1.39 不同（expert 走的是缓存组与流式组不同的混
  合）。实测质量，一篇长文档后续 2,001 个 token 上对照 FP16 prompt 路径的 teacher-forced（IQ3_XXS、
  `--prefill auto`）：8K prompt KL 0.042（0.1.39：0.054），top-1 一致率 93.5%（92.2%）；32K prompt KL
  0.020（0.019）、top-1 95.3%（95.0%）——与之前同一区间。`STRATA_RING_BYTES=0` 恢复 0.1.39 的环、借用与
  分块选择。

- **启动 VRAM 计划（#765）：** 在 expert 缓存拿走 VRAM 剩余之前，引擎给所有还必须与它并存的东西定价——
  草稿头、prompt 路径借不到时的自有缓冲区（`Prefill::bytes_needed`，精确值，取代旧的 `160 + chunk * 680 /
  1024` 估算，24,576-token 分块时它高了约 7 倍）以及一个给 `--spec` 常驻表的槽——并打印该计划（`strata
  generate: VRAM plan:`）。一个所求 `--prefill` 而计划中的缓存借不出时，在缓存提交之前把它的缓冲区记账在先
  （缓存先缩小）；哪个地方都装不下的分块被降到装得下的最大 256-token 尺寸并明说；什么都满足不了的配置在启
  动时被拒绝并附上预算和腾空间的旋钮，而不是死在模型加载与 READY 之间。那笔价里有一部分此前是隐藏的：流式
  环约 384 个各自 2.7 MiB 的分配，每个都向上取整到一个 2 MiB 页，约 0.5 GiB 是计数从来没看见的——环现在是
  一次分配。RTX 4070 Ti SUPER（16 GB）实测：32K 上下文带 `--prefill auto`，计划与第一个 prompt 实际借的正
  好相同（8,192 个 token、2,487 个缓存槽、1,950 tok/s prefill——规划之前引擎达到的数字）；524K int8-KV 带
  `--prefill 24576` 的启动，规划之前死在第一个 prompt，现在从头跑到尾（17,664-token 的分块靠自己的缓冲
  区、1,950 之上还有 2,139 tok/s prefill，缓存保住 `--spec` 要的槽）。

在这台机架的生产切分上（两块 V100 16 GB、524,288-token int8 KV、`--prefill auto`、layer-split 20），
计划打印同样的交易：一个 8,192-token 的分块从计划中的缓存借出，缓存精确按计划打开（CUDA0 上 8,255 个
槽、CUDA1 上 5,750 个）。槽装不下时，下面的批处理槽回退在同一启动上仍会触发。

完整的故事，含测量、瓶颈与下一步：**[paper/Strata-Paper.pdf](../paper/Strata-Paper.pdf)**。

---

## 鸣谢与许可

<a id="credits-and-licenses"></a>

Strata 本身：[MIT](../../LICENSE)。模型文件不是它的一部分；它们的许可适用于它们（如下）。

- 模型：Qwen 团队的 [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；量化：
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
  Coder：
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  （按其卡片为 Apache-2.0）；它在 Strata 里的支持来自 @pjgmobile 的 PR #54。Swift 1.5：UkisAI 的
  [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
  它们的许可适用于权重。
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）：i-quant 格式、转录进
  `src/kernels/cuda/iq_kernels.cu` 的 GPU 点积与反量化器、为 i-quant expert 链接的 CPU 后端、图像编码器背后
  的 `mtmd` 库（`tools/vision/`），以及工具用的 `gguf-py`。见 `third_party/ggml/LICENSE`。
- 来自 [Splash](https://github.com/incoai/splash)、[ninfer](https://github.com/Neroued/ninfer) 和
  [HyperQwen](https://github.com/syv-ai/HyperQwen) 的想法；论文里有引用。
- 网页应用的字体：[Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1，见
  `serve/web/fonts/OFL.txt`）。它的 Monitor 标签页起步于 @code-martin 的仪表盘想法（PR #22）。
- 实验性速度投影的向量（`data/experimental-speed-projection/`）：Qwen Community License 1.0，由模型的
  激活制成（见其 README）。

### 不加载模型就启动 API

`serve/server.py --engine strata --config strata-<model>.json --lazy`（或该配置里的 `"lazy_load":
true`）启动轻量 HTTP API 而不启动原生引擎或视觉编码器。第一个生成请求、或 `POST /load`，在配置了视觉时
先启动视觉编码器，然后通过既有的重载路径加载引擎，包括 `before_load` 与 `min_free_vram_mib`。这对图像
请求同样有效：加载在 Strata 编码图片之前完成。`POST /unload` 停止两个进程。急切启动仍是默认。

`POST /v1/load` 与 `/v1/unload` 是给集成的 JSON 控制别名，接受 `{}` 或 `{"model":"<configured
model>"}` 并返回模型状态。它们要求配置的 API 密钥、`application/json`、且不带外站浏览器 Origin。一个请
求在活动或排队中时返回 **409**，未知模型返回 **404**。既有 `/load` 与 `/unload` 的行为保留。
`/api/health` 是 `/health` 的别名；`/v1/status` 暴露 `loaded` 与 `auto_load`。未加载的模型保持可发现。

卸载与关闭在发送 `QUIT` 之后关闭原生引擎的 stdin，让 Windows 的分离式 stdin 读取者看到 EOF。清理在释放
句柄之前等待进程退出；强制关闭仍然超时时，服务器保留所有权并报告错误，而不是声称模型已卸载。

<a id="start-the-api-without-loading-the-model"></a>

### JSON 响应格式

`POST /v1/chat/completions` 接受 `response_format: {"type":"json_object"}` 或
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}`。
Schema 的根必须只接受 JSON 对象：`"type":"object"`，或所有分支都是对象 schema 的 `anyOf`/`oneOf`（一个
含对象成员的 `allOf`，或指向对象的本地 `$ref`，也算），像为 llama.cpp 的 `json_schema` 写的应用发的那
样。也允许数组、字符串、数字、布尔或 null 的根被拒绝。本地 `#` 引用可用；远程引用被拒绝。装了 Python 包
`jsonschema` 时用它校验 `json_schema`（`python -m pip install "jsonschema>=4.23,<5"`；setup 不装它）；没
有它时只检查回答是一个 JSON 对象，服务器说明一次。

这是 **schema 提示加服务器校验**，不是语法约束的 decode。每个请求生成一次，没有隐藏的重试。成功的响应
包含一个经过校验的 JSON 对象。畸形 JSON、重复键、非有限数、schema 违规和不完整的生成返回 **502** 并带
`error.code: structured_output_failed`；无效的请求 schema 返回 **400**。JSON 格式与工具/MCP 组合会被明
确拒绝。不带 `response_format` 时，普通文本与工具行为不变。

结构化 SSE 在发送 keep-alive 注释的同时缓冲回答。它只在校验通过后发出 content，然后是 usage/timings 和
`[DONE]`；失败发出一个 SSE error 和 `[DONE]`，不带无效内容的增量。`/v1/status.structured_output` 声明格
式、校验方法与缓冲式流式行为。

<a id="json-response-formats"></a>

### API 请求监视器

默认关，因为它把 prompt 和回答留在内存里：用 `strata-<model>.json` 里的 `"api_monitor": true` 打开（或
`serve/server.py --api-monitor`）；否则什么都不记录，下面两个端点回答 404。
打开 `/api-monitor` 可以在不开聊天的情况下检查 API 流量。它显示模型状态、安全的 load/unload 控制、活动
/排队的请求、原始请求体、输出、分开的 reasoning 与非流式响应体。总墙钟时间包含 FIFO 等待和自动加载；加
载、排队、首 token、prompt/输出 token 和引擎 decode 计时分开显示。

`GET /api/requests` 返回紧凑摘要；`GET /api/requests?id=<id>` 返回一个被保留的请求。两者用现有的 API
密钥检查。监视器在内存里保留最新 **100 个请求**直到重启，**每个输入/输出/reasoning/响应字段 262,144 个
字符**并带可见的截断标志。实际 API 响应不受影响。请求头不被记录，监视器密钥保存在该标签页的会话存储
里。把 Strata 暴露到网络上时，请把请求历史当作敏感输入/输出：按上文设一个 API 密钥。页面用相对 URL，经
现有的 host 绑定或反向代理都能工作。

<a id="api-request-monitor"></a>

### Prompt 缓冲：`bo` 共用 `emb`（#1454）

Prompt 路径的半输出缓冲 `bo` 复用 embedding 缓冲 `emb`——它在第一次 hyper-connection 广播之后就死了：
每分块少 T x 2560 个 float 的 VRAM（32768 行时 320 MiB）。规划器默认仍把那些字节计入，所以 auto 分块和
prompt 路径借用的缓存槽与 0.1.40.3 完全相同，输出比特不变。`STRATA_EMB_REUSE_ACCOUNT=1` 让规划器使用节
省的字节：VRAM 限制分块的地方它会变大（RTX 3060、IQ3_XXS：6400 到 6656 个 token，借用槽 1652 到 1670，
prompt 约 +3.9%）。不同的分块改变 prompt 路径的舍入，所以 prompt 残差与默认值不逐字节相同；贪心输出在
我们的运行里相同。在拿到 KL 测量之前保持需显式开启。

**gfx12（RDNA4，R9700）上的融合 int8 prompt expert：IQ3_S 与 IQ3_XXS 包默认开。**
`STRATA_PF_FUSED=0` 关掉它们（回到 MMQ 的 prompt 内核、之前的比特）。R9700、1 卡：prompt 4K +15.7%、16K
+13..+17%、32K +19..+20%、一个 96K prompt +16%（回答文本相同）；96 个 prompt 上对 FP16 prompt 路径的首
token KL 与 MMQ 在噪声内相等（IQ3_XXS 0.00613 对 0.00683，IQ3_S 0.00416 对 0.00372）。其他量化和其他 GPU
不变；`STRATA_PF_FUSED=1` 仍为每个受覆盖的格式选择加入。

**gfx12（Linux）：裸的 `--prefill auto` 可能拿 16384-token 的分块。** 显式的 `--prefill N`、`auto:N` 和
`STRATA_PREFILL_AUTO_MAX` 不受影响；`STRATA_GFX12_CHUNK=0` 或 `STRATA_GFX12_DEFAULTS=0` 保持 8192。

<a id="prompt-buffers-bo-shares-emb"></a>

### CUDA 诊断：给 verify 窗口启动用一个 host 线程

`STRATA_VERIFY_ASYNC_LAUNCH=1` 把 `Verifier::run` 的 CUDA graph 启动放到一个单独的 host 线程上。这让
它的调用方能在带插桩的 `cudaGraphLaunch` 阻塞等 graph 时服务 CPU expert 的门铃。调用方加入该启动并在返
回一个 frontier 之前检查它真实的错误；提前的服务失败在加入之前释放映射的 GPU 等待。该开关默认关，只适用
于这个单请求 verifier 入口。直接的批处理与流水线 graph 启动以及 HIP/SYCL 路径保留它们现有的调度。

给 Compute Sanitizer，在 `compute-sanitizer --tool memcheck --error-exitcode 86` 下运行 usual 的模型命令
之前设该变量。用 `STRATA_BUILD_TESTS=ON` 构建并运行 `host_graph_launch_test` 做 CUDA graph/CPU 信号、提
前清理和 `--errors` 错误传播检查。它的 `--synchronous` 模式是 memcheck 下的阴性对照：一个八秒看门狗释放
原来的启动/服务死锁，测试失败。

<a id="cuda-diagnostics-a-host-thread"></a>

## RTX 30 系列（sm_86）与 Tesla P100（sm_60）：逐比特的 decode 开关默认开

在每个架构上逐开关测量于 12 GB RTX 3060、16 GB RTX A4000（单卡）和 16 GB Tesla P100，IQ3_XXS、默认开关
（自适应层开）、每次运行一个新引擎、8 个交错对、expert 缓存槽数相同、带一条噪声臂：整套 16 个开关给出
decode +9.6%（3060）、+14.6%（A4000）、+17.8%（P100），各 8/8 对，恒等门（10 个 prompt + exact_pp 4K/20K、
固定开关）带着它 IDENTICAL。一次移除一个开关（各 4 轮）保留了在各自架构上划算的那些。它们现在按架构默认
开（`include/strata/core/nv_defaults.hpp`，在引擎读它的开关之前设置；你自己设的开关，无论设成什么，都被
保留）：

- sm_86（RTX 3050-3090、RTX A2000-A6000）：`STRATA_ADAPT_NOWAIT=1`、`STRATA_ADAPT_LAG=2`、
  `STRATA_CPU_IQ_ALL_EXACT1=1`、`STRATA_SIDE_MIXER=2`、`STRATA_ATTN_DAILY=1`、`STRATA_GR_UP75=1`、
  `STRATA_GU_SWIGLU=1`、`STRATA_QSA_ROWS=1`、`STRATA_B6_MMVQ_ROWS=1`。`STRATA_SM86_PERF=0` 关掉整组。
- sm_60（Tesla P100）：`STRATA_ADAPT_NOWAIT=1`、`STRATA_ADAPT_LAG=2`、`STRATA_SIDE_MIXER=2`、
  `STRATA_FUSE_WAIT=2`、`STRATA_B6_MMVQ_ROWS=1`、`STRATA_FUSE_DECODE=1`、`STRATA_GR_UP75=1`。
  `STRATA_SM60_PERF=0` 关掉整组。

这张表只在每块可见显卡都是那一个 compute capability 时使用。`STRATA_ADAPT_OVERLAP=1`（需显式开启）不与
与 `STRATA_ADAPT_LAG` / `STRATA_ADAPT_NOWAIT` 并跑：和它一起用时设
`STRATA_ADAPT_LAG=1 STRATA_ADAPT_NOWAIT=0`。因为不划算或结果好坏参半而未纳入：
`STRATA_EXPDEC`、`STRATA_GR_NORM_FAST`、
`STRATA_RAM_ADAPT`、`STRATA_KV_HOST_REGISTER`、`STRATA_HOST_CORE`、`STRATA_GR_DOWN_TG`、
`STRATA_DEQUANT_WIDE`；`STRATA_EXCHANGE_ROTATE` 在 3060 上到不了 READY，`STRATA_PCIE_BALANCE` 在单卡上更
慢。

各架构设置本身（不带环境变量），对照没有它们的引擎，IQ3_XXS、默认开关、8 个全新引擎的交错对、每条臂同
样数量的 expert 缓存槽（引擎 0.1.45.3 及其测试构建）：

| 卡 | Decode 之前 -> 之后 | 增益 | 更快的对 | 同一引擎自比 | 带 `STRATA_SM86_PERF=0` / `STRATA_SM60_PERF=0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 12 GB (sm_86) | 60.36 -> 66.52 tok/s | +10.2% | 8/8 | 0.999 | 1.002（与之前相等） |
| RTX A4000 16 GB (sm_86) | 51.16 -> 57.82 tok/s | +13.0% | 8/8 | 0.979 | 0.994 |
| Tesla P100 16 GB (sm_60) | 36.56 -> 41.76 tok/s | +14.2% | 8/8 | 0.975 | 0.991 |

看过且**未改**的两件事。8192 以上的 prompt 分块（`--prefill auto:16384` 或 `auto:32768`）在 12-16 GB 卡、
20K 个 token 上让 prompt 速度变化 -3% 到 +8%（在 16384 / 32768 时：RTX 3060 +4.5% / +4.5%、Tesla P100
+3.1% / +7.7%、RTX A4000 -3.4% / +6.0%），但它在 CUDA 上不逐比特（残差转储在 P100 和 A4000 上与 8192 不
  同；32768 还改变了 P100 的输出 token），所以 NVIDIA 上默认保持 8192。它在 gfx12 上是精确的，那里 16384
是默认。而 `STRATA_ROUTE_PRIOR=0.5` 保持需显式开启：RTX 3060（IQ3_XXS）上它的 teacher-forced KL 是
0.0126，逐次底线 0.0012（top-1 一致率 96.6% 对 99.1%）；RTX A4000 上 HumanEval 得 164 之 154，两次默认的
157 和 159 对；推理 30 之 25 对 24 和 26（agent 6/6、大海捞针 3/3、CJK 8/8、工具调用 5/5 全通过），在逐次
散布之内。

<a id="rtx-30-series-sm_86-and-tesla-p100-sm_60-bitwise-decode-switches-on-by-default"></a>

## Compute capability 12.0 的卡：更快的 decode（逐比特）与默认开的融合 prompt expert

RTX PRO 5000 Blackwell 72 GB（110 SM、sm_120、Linux、CUDA 13.2）、IQ3_S、所有 expert 在 VRAM、setup 的开
关（`--spec 4 --spec-min-p 0.5`）、6 个全新引擎的交错对、300-token 贪心回答、中位数实测：

| | 故事 | 代码 | 8K 文档 | 4K / 20K / 98K prompt (tok/s) |
|---|---|---|---|---|
| 之前 (tok/s) | 168.8 | 224.6 | 186.6 | 3829 / 4613 / 4767 |
| 现在 | 192.0 (+13.8%, 6/6 对) | 260.0 (+15.8%, 6/6) | 211.9 (+13.6%, 6/6) | 4212 / 4870 / 4929 (+9.9% / +5.4% / +3.4%, 各 6/6) |

Decode 逐比特：`STRATA_PF_FUSED=0` 时回答与之前 token 相同且恒等门通过（10 个 prompt 加 4K 和 20K 的
prompt 转储，固定小缓存下和所有 expert 常驻下都跑了）。Prompt 在舍入层面变化是因为融合 prompt expert（见
下）。改动内容：

- 12.0 的卡上这些逐比特开关除非设置否则开着：`STRATA_SIDE_MIXER=2`（mixer 只依赖输入的活在 graph 分支
  上）、`STRATA_ATTN_DAILY=1`（Daily 的 decode attention）、`STRATA_GR_UP75=1` 和 `STRATA_GR_DOWN_TG=1`
  （hyper-connection 读取的上投影带着提前取出的装载，下投影每 token 一个块行；token 成对用 `=2`，80 SM 以
  下）、`STRATA_GU_SWIGLU=1`（新：路由 expert 的 gate/up、SwiGLU 和 q8_1 一次启动）。带着它们，一个固定 4
  token 的 verify 轮从 13.70 -> 11.80 ms。`STRATA_SM120_PERF=0` 关掉整组；每个开关自己的变量在两个方向上
  仍然更优先。
- 交错式小批 matvec（`native_mmvq_il`，docs/MMVQ_IL_TABLE.md）也跑 5 到 8 个 token（它原来停在 4，而 5
  token 的窗口比 4 token 的贵 2 ms），带一张 12.0 的 5-8 token 表，且在 80 SM 以上的 12.0 卡上有自己
  2-4 token 的表。逐比特（`mmvq_il_parity`，1-8 个 token）。`STRATA_MMVQ_IL_WIDE=0` / `STRATA_MMVQ_IL_120=0`
  关掉它们。
- 80 SM 以上的 12.0 卡上，原生包的 prompt expert 在融合 int8 内核（`STRATA_PF_FUSED`，gfx12 已有的规则）
  上跑 IQ2_S、IQ3_XXS、IQ3_S 和 IQ4_XS 的 gate/up；IQ2_XXS / IQ2_XS 保持 MMQ 直到测量。一块 RTX 5070（48
  SM、IQ2_XS 包、expert 部分在 RAM）带着它在 20K prompt 上慢 4.7%（0/6 对），所以更小的 12.0 卡保持
  MMQ。IQ3_S 的质量：对 MMQ 的 teacher-forced KL 0.0002（5998 个位置；MMQ 自比 0.0000），约 6K-token 文档
  之后 0.0010（1760 个位置）；对 FP16 prompt 路径的首 token KL 0.01181 对 MMQ 的 0.01236（48 个 1.5K-24K
  token 的 prompt）；HumanEval 164 之 156、推理 30 之 26、agent 6/6、大海捞针 6/6、CJK 8/8、工具调用 5 之
  4，与 MMQ 相同。`STRATA_PF_FUSED=0` 保持 MMQ。
- 行变便宜之后，更深的投机被重新测量（3 对）：`--spec 3` 为之前的 1.145 / 1.020 / 1.123，`--spec 4`
  1.122 / 1.141 / 1.132，`--spec 5` 1.079 / 1.101 / 1.074，`--spec 6` 1.060 / 1.177 / 1.097，`--spec 8`
  1.027 / 1.105 / 1.069（故事 / 代码 / 文档）。`--spec 4 --spec-min-p 0.5` 保持默认。
- Setup（Linux，非 WSL）：RAM 覆盖模型所需 + n-gram 表 + 16 GB 时（Q2_0 93 GB、IQ3_S 107 GB），表保持在
  RAM（`--ple-io ram`），新 prompt 不再在 SSD 上等它的行：这张卡上，每个 prompt 一个新引擎，4K -18%、20K
  -9%、98K -2%，比特相同。R9700 上读取本来就被藏住了；RAM 不足时（48 GB 上限）ram 模式要多花 20% 的
  prompt 和 14 秒启动，这就是更小的 PC 保留 SSD 读取的原因。

未测：其他 12.0 卡（RTX 5070）、Windows、部分常驻的缓存、IQ3_XXS 和 IQ2_XS 包。

<a id="compute-capability-cards-faster"></a>

## 为 0.1.45 测量的 CUDA 可选开关（RTX 3060 与 RTX A4000，sm_86；全部默认关，除非你设置）

从公开的 pull request 移植，并作为整引擎运行对照默认值做 A/B（每边一个新引擎、IQ3_XXS、5 对、中位数；逐
比特相同指开着该开关的 10-prompt + 4K/20K 恒等门等于关着它的门）。除注明外，为 30K-token prompt 之后的
decode；增益在这两张卡上不大，在作者们测量的 Turing 卡上更大。

| 开关 | 内容 | 回答 | RTX 3060 / A4000 |
|---|---|---|---|
| `STRATA_ATTN_DAILY=1` (#2075) | decode attention：每步两个 cell，一个跳过空块的归并 | 逐比特相同 | +3.2% (5/5) / +0.7% |
| `STRATA_QSA_ROWS=1` (#2059) | verify 窗口的块打分，query 放寄存器 ([QSAROWS.md](QSAROWS.md)) | 逐比特相同 | -0.3% / +0.3% |
| `STRATA_EXPDEC=1` (#2058) | 窗口常驻 expert 的 gate/up，q8_1 行放共享内存 ([EXPERTDEC.md](EXPERTDEC.md)) | 逐比特相同 | +0.5% / +1 到 +5%（短上下文） |
| `STRATA_SIDE_MIXER=1/2` (#1889) | mixer 只依赖输入的活在 graph 分支上 | 逐比特相同 | +0.5% / -0.6%（短上下文） |
| `STRATA_B6_MMVQ_ROWS=1` (#1418) | 跨行复用激活碎片的精确多列 MMVQ（K-quant、IQ4_XS） | 逐比特相同 | +0.2..1% / +0.5..2% |
| `STRATA_QSA_SPLIT_ATTN=1` (#1888) | int8 mma 路径上跨块拆分的 decode attention（sm_80+，int8 KV） | 舍入层面（teacher-forced KL 0.0016，底线 0.0012） | +2.8% / +3.5% |
| `STRATA_HC_UPMIX=1` (#1370) | hyper-connection 上投影与 `gr_mix_r` 合进一个内核（sm_80+） | 这里逐比特相同（门相同） | prompt +0.6% / +0.4% |
| `STRATA_PROMPT_ATTN_IMMA=1` (#1367) | int8-KV prompt attention 跑 INT8 tensor core | 舍入层面（KL 在底线上） | prompt 读取 +2.5% |
| `STRATA_HC_REQ8=1` (#1262) | hyper-connection 投影改为 int8 加每 32 一个 scale（更多 expert 缓存槽） | 舍入层面（KL 0.0018） | decode -0.2% / +2.2% |
| `STRATA_QUANT_GATHER=1` (#1368) | 把每 token 一次的 MoE 输入量化改回按行 gather | 不适用 | 未测出差异 |
| `STRATA_PREFILL_GROUP_GATHER=0` (#1107) | 关掉短 prompt 的分组 gather | 不适用 | 开着时快 0.6-1.3% |

`STRATA_DEQUANT_WIDE`（prompt 稠密权重的 Q4_K / Q5_K / Q6_K / IQ4_XS 宽反量化）现在在 sm_86 上与 sm_61 一
样默认开：逐比特相同，且 150 / 300 / 450-token prompt 到第一个 token 的时间在 RTX 3060 上降 11-15%、RTX
A4000 上降 6-10%（各 5 之 5 对）。`STRATA_DEQUANT_WIDE=0` 关闭。

### RTX 20（Turing，sm_75）内核系列（需显式开启，0.1.45）

rafatxf（#2069-#2072、#2093、#2220-#2222）、imanu86（#2081）和 xrip（#1821）为 compute capability 7.5 写
的内核。它们在一块 RTX 2080 Ti 上测过（按各 PR：decode 在 4K-32K 上 +5..6%，深度至多的 prompt 至多
+30%），与它们替换的内核逐比特相等，但团队里没有人有 sm_75 的卡来确认，所以在有人要求之前保持关闭：

`STRATA_SM75_PERF=1` 在 7.5 的卡上打开下面每个逐比特的 Turing 内核（其他卡永远不受影响）。每个都有自己的
变量，在两个方向上更优先（`=0` 关，`=1` 开）：`STRATA_SM75_IL`（交错式小批 matvec 表）、`STRATA_SM75_EXP`
（expert 布局模式 4）、`STRATA_GR_UP75` / `STRATA_GR_DOWN_TG`（hyper-connection 读取）、
`STRATA_SCORES_HALVE` 和 `STRATA_TOPK_SPLIT`（decode QSA 选择）、`STRATA_TOPK_SAMPLE`（prompt top-k，分数
只读一遍）、`STRATA_ATTN_V3`（decode attention）、`STRATA_Q2_WARP`（Q2_0 matvec）、`STRATA_FUSE_WAIT=2` 和
`STRATA_QIL`（每个 decode 层更少的启动）、`STRATA_TC_SM75`（BF16 反量化、16 字节存储）、
`STRATA_PA_V1_PREFETCH`（prompt attention v1 预取）、`STRATA_SCORES_THREAD`（#1825：prompt 路径的每头一线
程块打分器，逐比特等于 warp 内核的分数；`=1` 在任何 CUDA 卡上取用它，那里 Tesla P100 在 30-60K 个 token 上
测得 prompt 速度 +1..3%，所以它在 P100 上不是默认）。

改变答案的选项（不同的求和顺序）是分开的、显式的：`STRATA_SELECT_SIMT=1|2`（FP32 平铺块打分器，按 #2071
对 warp 打分器 KL 0.01）、`STRATA_PA_V2T=1`（v2 prompt attention，#2092/#2146）、`STRATA_PF_FUSED=1
STRATA_PF_FUSED_SM75=1`（Q2_0 包的融合 int8 prompt expert，#2145；只在 7.5 的卡上，sm_80+ 不变）、
`STRATA_BF16_TC=1` 配 `STRATA_XN_F16`（#2091）。

<a id="rtx-turing-sm-kernel-series"></a>
