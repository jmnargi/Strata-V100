# A770 decode 优化规格说明

<a id="a770-decode-optimization-specification"></a>

## 目标与完成标准

<a id="objective-and-completion-criteria"></a>

在本地 Intel Arc A770 上运行 Strata，**decode 达到至少 20 生成 token/秒**，并附可复现的基准
证据和正确输出。本规格说明描述下一轮优化战役；它并不声称目标已经达成。

<a id="achieve-at-least-20-generated-tokenssecond-in-decode"></a>

完成需要满足：

1. 在 A770 单卡上，跑与现有测量相同的 Qwen3.8-Flash-Next Coder IQ1_M 包、分词器和 MTP 产物。
2. 主对比保持当前的 131,072-token 上下文容量与 INT8 KV 配置。更短上下文的实验单独标注。
3. 在代表性编码工作负载至少三次预热重复中，展示引擎 decode 吞吐中位数 ≥20 tok/s，每次重复
   至少生成 256 个 token，除非模型先自然停止。较短的自然停止样本单独报告，不得用来顶替
   持续吞吐的验收门。
4. 与此前的引擎测量一致地统计 reasoning token。除速度外还要报告最终回答内容与结束原因。
   反复的乱码流、人为的重复 token 工作负载或缓存响应都不是有效的成功测量。
5. 通过下文的正确性门，包括原始算术回归 prompt 最终输出 `41`，以及生成的编码回答被实际
   执行测试。
6. 所选实现与运行配置必须能从记录的源码修订、构建选项和配置复现。内核微基准、预期增益或
   单条有利请求不能宣告 decode 目标完成。

<a id="completion-requires"></a>

## 已验证基线与硬件

<a id="verified-baseline-and-hardware"></a>

- 工作树：本地 Strata v0.1.40.3 检出。
- 当前源码基线：`501f675727ae8a54abed3ec467de1cbfbecd03ea`，分支 `a770-speed-upstream`，
  最后检查时基于 upstream `main` 的 `fb58e0d`。
- 已有的 upstream 工作：正确性 PR #1602 与草稿性能 PR #1710。
- 引擎二进制：`build-sycl/strata`；报告引擎版本 0.1.41。
- GPU：Intel Arc A770，PCI `0000:06:00.0`，设备 id `56a0`，`xe` 驱动，
  `ONEAPI_DEVICE_SELECTOR=level_zero:1`。`level_zero:0` 上的 B580 驱动显示器，不在这个
  单 A770 基准之内。
- CPU：Ryzen 5 5600，六核十二线程，AVX2；约 31 GiB RAM。
- A770 挂在芯片组的 PCIe 3.0 x4 连接后面。引擎传输探测测得 1.7-1.8 GB/s。不要只凭 GPU
  端点空闲时的 sysfs 速率推断实际装载的链路；要记录上游桥和实测传输量。
- 构建/运行时：`strata-sycl-dev` 镜像内的一 API 2026.1。

<a id="worktree-a-local-strata-v01403-checkout"></a>

实测进度，不是估计：

<a id="measured-progress-not-estimates"></a>

| 配置 | 预热 decode | 备注 |
| --- | --- | --- |
| 锁页镜像，4.46 GiB VRAM expert | 约 2.5 tok/s | 多数缺失 expert 由 GPU 经 PCIe 读取 |
| CPU 承接缺失、更大缓存与调好的 prompt 预留 | 约 9-11 tok/s | 相同的上下文容量 |
| 快速路由加借入与配对映射传输 | 13.5、11.9、12.3 tok/s | 546 输入 token，128 生成，无前缀复用 |
| 同一最终配置，短预热 prompt | 11.4 tok/s | 73 输入 token，128 生成 |

最终配置有 4,717 个 expert 槽位，VRAM 内约 8.98 GiB，常驻 RAM expert 15.08 GiB。现有运行
没有测过完整 128K prompt。快速路由隔离的 256-expert/top-10 计时从每次调用 395.47 微秒降到
118.62 微秒，结果逐位一致。

<a id="the-final-configuration-had-4717-expert-slots"></a>

早先战役的产物放在一个临时目录：`a770-speed-results.json`、`a770-tuning-round2-results.json`、
`a770-tuning-round3-results.json`、`a770-tuning-round4-results.json`、
`a770-tuning-paired-results.json`、它们的配置文件和引擎日志。它们是有用的历史证据，但临时
路径不是已完成优化工作的持久交付物。

<a id="artifacts-from-the-earlier-campaign-live"></a>

## 范围与约束

<a id="scope-and-constraints"></a>

- 优化现有 SYCL 引擎，而不是改模型权重或换成无关的推理引擎。保留 >4 GiB 内核寻址的构建修复。
- 保留 BF16 权重解释、FP32 累加约定、路由语义、MTP 校验和负载校验和。更快的 FP16 算术是
  独立的数值实验，不是悄悄替换 BF16。
- 每个新调度都要保留可选择的老/参考路径。改默认值需要奇偶校验和实际输出验证。
- 按内核、按架构分别调优。当按 CUDA 形状写的 shuffle 和归约代码仍假设 32 条 lane 时，不要
  全局把 subgroup 32 改成 8 或 16。
- 研究别的仓库时避免执行它的构建脚本；移植选中的技术时连同其许可证声明和测试一起。
- 本规格说明不暗示自动提交、推送、更新 PR 或改动用户常规运行配置。实验配置保持分离。
- 每次全模型测试前记录 CPU/RAM 争用、可用 RAM、交换活动与 GPU 时钟。要关闭新打开的用户
  应用之前先询问。

<a id="-optimize-the-current-sycl-engine"></a>

## 工作包 1：可靠的事件计时测量

<a id="work-package-1-reliable-event-based-measurements"></a>

### 问题

<a id="problem"></a>

最初的 base 阶段分析器写入的是零时间戳。实验性移植版让一个 host 线程持续更新一个 host USM
时钟字、GPU 以非缓存方式读取它。审查指出这种原子-host/非原子-device 的时钟访问缺少有正确性
保证的同步约定。启用的运行还报告过 `layer ... rang but its payload never arrived whole`；
这些事实之间的因果关系尚未证实。

<a id="the-original-base-stage-profiler-wrote-zero-timestamps"></a>

### 初步实现

<a id="initial-implementation"></a>

为 BF16 投影系列增加一个隔离的 SYCL 事件基准。它的队列必须与所选设备/上下文共享，并同时具备
`in_order` 和 `enable_profiling`。设备内存与启动必须使用同一个队列。

用设备事件时间戳测量一批预热后的批次；若公共启动器不返回其内核事件，就用原生队列标记内核
界定这批的边界。只有在全部受测提交都入队之后才等待。要把它报告为**设备队列跨度**，包含命令
之间的调度间隙；不要标成单内核的纯执行时长。同时报告 host 墙钟时间和空标记跨度，让标记开销
和提交饥饿可见。

<a id="add-an-isolated-sycl-event-benchmark"></a>

初步基准不得启用 `STRATA_VERIFY_PROFILE`、启动 host 时钟写入器、录制命令图，或在事件仍挂起
时查询它。要求起止时间戳非零且单调有序；否则带着有用的诊断信息失败，而不是打印零值或回绕的
时长。

<a id="the-initial-benchmark-must-not-enable"></a>

### 后续集成

<a id="subsequent-integration"></a>

对 eager 诊断启动，把每个操作返回的 `sycl::event` 收进一个有界的 host 侧集合，只在完成后
查询 profiling 起止。eager 计时要与捕获图的产出速率分开标注。不要假设录制期的节点事件包含
命令图每次重放的时间戳。图节点分析器需要一个受支持的重放 profiling API；否则单独为 constituent
内核做基准，并用正常的无插桩图重放校验整个窗口的时间。

<a id="for-eager-diagnostic-launches"></a>

不安全的 host 时钟分析器在每个速度验收运行中必须保持禁用。替换它必须移除忙写入器，并在
profiling 不可用时保留清晰的行为。Profiling 插桩必须需显式开启，且不得改变常规 expert 池
路径的同步。

<a id="the-unsafe-host-clock-profiler-must-remain-disabled"></a>

### 输出模式

<a id="output-schema"></a>

每个基准行必须包含：源码修订、设备名与 PCI 标识（可得时）、队列模式、编译器/后端选项、系列、
实现、N/K/T 维度、输入/输出主维度、重复次数、预热次数、设备跨度中位数/最小值、host 墙钟
时间、空标记代价、有效权重带宽、正确性结果与输出哈希。不要把缺少数据当成通过或更快的结果。

<a id="every-benchmark-row-must-include"></a>

## 工作包 2：现有的少 token BF16 投影路径

<a id="work-package-2-existing-few-token-bf16-projection-paths"></a>

### 首次对比

<a id="first-comparison"></a>

对比默认 `bf16_gemv_fp32_mmvf_multi` 路径与 `STRATA_MMVF_ROWS=1`。后者已在
`sycl/src/kernels/cuda/native_bf16.dp.cpp` 中实现；其选择器在首次使用时缓存。每种模式在
独立进程中运行，在任何启动之前设好环境变量。不要在同一进程里两次测量之间改 `setenv` 并
指望缓存的选择器会变。

<a id="compare-the-default-bf16_gemv_fp32_mmvf_multi-path"></a>

对每种模式，与每个输出列一次独立的单 token MMVF 调用对比。在实现约定如此的地方检查输出
逐位一致。用 FP64 CPU oracle 对一个有界子集做第二重检查；oracle 不取代逐位新老对比。

<a id="for-each-mode-compare-against-one-independent"></a>

### 形状矩阵

<a id="shape-matrix"></a>

形状按 `[N 输出行, K 输入列]` 理解，作用于 T 个激活向量。T 取 `{1, 2, 3, 4, 6, 8}`，至少
包括：

<a id="interpret-shapes-as-n-output-rows-k-input-columns"></a>

| N | K | 用途 |
| ---: | ---: | --- |
| 4 | 10240 | 超连接注入；输出行极少 |
| 320 | 10240 | 超连接下投影 |
| 10240 | 320 | 超连接上投影 |
| 256 | 2560 | Coder 路由 |
| 2560 | 2560 | dense 投影 |
| 10240 | 2560 | 宽 dense 投影 |

包含补齐后的偶数 `ldx` 和 `ldy`、一个非满输出 tile（例如 N=65）、按公共 API 的零尺寸/空操作
或被拒绝的形状、保护区域、有限随机数据、全零输入和尺度极端的有限数据。记录哪些形状真的
选中了行内核；N<64 目前会回退。

<a id="include-padded-even-ldx-and-ldy"></a>

只有当匹配配置的 A/B 运行显示出有益改进时，才把胜出者移进引擎。如果行内核对某一窄类别有
帮助却让另一类回退，就用按形状的特化调度，而不是全局新默认。

<a id="only-move-a-winner-into-the-engine"></a>

## 工作包 3：定向的寄存器与 subgroup 调优

<a id="work-package-3-targeted-register-and-subgroup-tuning"></a>

调查寄存器密集 A770 内核上 128 与 256-GRF 编译的区别，保留现有后端选项，包括
`-ze-opt-greater-than-4GB-buffer-required` 和正确舍入的 divide/sqrt。确认实际发出的编译
选项，拒绝不受支持的选项。

<a id="investigate-128-versus-256-grf-compilation"></a>

候选 subgroup 大小只有 8/16/32，且只在归约和工作组几何可以适配之处。初始目标：BF16 投影
行、GDN 递归、超连接下/上投影和量化 dense MMVQ。工具链可转储 ISA 时记录寄存器溢出或 ISA
证据。更多寄存器可能降低占用率；一个被接受的编译选项本身不是速度改进的证据。

<a id="candidate-subgroup-sizes-are-81632-only-where"></a>

先一次只评估一个变量，然后评估组合胜出者。每个矩阵都必须包含实际 verifier 使用的
两到四个 token 的区间，不只是大 prompt GEMM。Arcint 的 A770 大 GRF 与 subgroup-8 结果是
动机，不是可转移的性能论断。

<a id="evaluate-one-variable-at-a-time-first"></a>

## 工作包 6：从 Infernix 折入的条目（2026-10 轮）

<a id="work-package-6-items-folded-in-from-infernix-round-2026-10"></a>

Infernix（`github.com/Wallawalla47/Infernix`，审查的是 2026-10 的 README/RESEARCH_NOTES
版 `main`）是一个 CUDA/RTX-5090 引擎；它的内核（NVFP4/FP8 MMA、`mma.sync`、`cp.async`、
TMA、PDL、L2 预取）无法移植到 Intel Arc。三个想法可转移，纳入本轮。它自己的 CPU/PCIe
缺失拆分是为 PCIe Gen5 x8（约 32 GB/s）调的；A770 是 PCIe 3.0 x4（约 1.8 GB/s，少约 18
倍），所以任何拆分都必须在这里重新测量，不能照抄。

<a id="infernix-githubcomwallawalla47infernix-reviewed"></a>

1. **CPU + PCIe 混合的 expert 缺失拆分（`--pcie-frac`）。** Infernix "把每层缓存缺失的一部
   分放在 CPU 上计算，其余跨过 PCIe"。我们跑 `--pcie-frac 0`。在 resident-experts 配置里
   测试小的显式 `--pcie-frac`（例如 0.02、0.05），让 CPU 池与 GPU 侧 PCIe 取数重叠。
   警告：在 `xe` 上不带 `--stream-experts` 时，GPU 会对可分页 host 内存产生页错误，且显式
   非零值会警告它"hung an Arc Pro B70 (device lost)"。运行时要有防护：盯着设备丢失并立即
   回退。如果常驻补集实际上无法锁页，就把它记录为 `xe` 上不安全并停止。
2. **受控的 ms/轮测量。** Infernix 按受控 token 流（相同轮数/接受数）上的**每轮毫秒**给
   改动排序，两臂**在同一时间窗内交错**，标准差 <0.02 ms，并警告单侧探测和单内核排名会误导
   （我们自己的路由微基准快 3.3 倍却没有任何端到端增益）。这里所有 A/B 都采用它：固定种子、
   贪心，对比毫秒/窗口和轮数，不只是 tok/s。
3. **自适应草稿长度。** Infernix "对自己预测不佳的文本把草稿降到零"，所以糟糕的草稿从不
   拖慢一轮。我们有 `--spec-min-p`；用受控协议扫描它（0.0、0.3、0.5、0.7），确认当前的
   0.5 是否正确、更高阈值是否有利于偏散文的输出。

<a id="hybrid-cpu--pcie-expert-miss-split--pcie-frac"></a>

不可从 Infernix 转移（记录下来以免重试）：全部量化 MMA 内核和 KV 格式（fp8/nvfp4/vq2/k4v2）、
PDL/TMA/`cp.async` 重叠、它的 VRAM 台账与 Windows 预算处理、以及它的 6-CPU-worker 结论
（我们是 7；两边都同意"更多更慢，因为 CPU 和 PCIe 共享 host 内存带宽"）。

<a id="non-transferable-from-infernix"></a>

### WP6 目前结果

<a id="wp6-results-so-far"></a>

- **条目 1，`--pcie-frac`——在 `xe` 的 resident-experts 模式下惰性（已定论）。**
  `--pcie-frac 0.05` 被接受（带可分页内存警告），没有出错也没有丢设备，但 decode 计时报告
  **`PCIe 0.00`** 个 expert：计数器从未动过，所以没有任何 GPU 侧 PCIe expert 取数发生。
  Decode 为 12.0 / 11.9 / 12.2 tok/s，与 12.4 / 12.6 基线持平（而且那次运行 RAM 饥饿：
  常驻补集跌到 11.12 GiB，常驻 expert 更少）。`--pcie-frac` 在此模式下没有接通任何 PCIe
  读取路径；唯一的 PCIe expert 路径是锁页镜像（`--stream-experts`），因为 A770 链路是
  PCIe 3.0 x4（约 1.8 GB/s），它在这里慢约 5 倍。因此 Infernix 式的 CPU+PCIe 混合拆分
  **在这块卡上不可达 / 无用**。保持 `--pcie-frac 0`。
- **条目 3，自适应草稿长度**——已定论：保持默认 0.5。同一窗口簇内的稳定 decode
  （546-token prompt，256 生成）：`--spec-min-p 0.0` = 10.4 / 10.5 / 10.4 tok/s（草稿接受
  率约 52%）；`0.5` = 12.2 / 12.0 / 11.7（接受率约 68%）；`0.7` = 11.8 / 11.4 / 11.6
  （接受率约 93%）。更高阈值提高接受率但给出的草稿更少；0.5 最优或并列，0.0 明显更差。
  不改。

<a id="-item-1--pcie-frac--inert-on-xe-resident-experts-mode-resolved"></a>

本轮另一个被否决的旋钮：**`--adapt-swaps 0`**（不做自适应 expert 交换）在同一窗口内测得
11.2 / 10.9 / 10.5 tok/s，对比默认 8 的 12.0——默认的交换日程更好。保持 8。

<a id="additional-rejected-lever-this-round"></a>

重叠旋钮的同窗口 A/B：**`--spec-split` 有害**——9.9 / 10.2 / 10.3 tok/s，对比
`--no-spec-split` 的 11.7 / 12.1 / 12.6。所以让草稿器与 verify 窗口重叠在这块 A770 上是
负收益（与 fork 在 B70 上的结果相反）。引擎默认值与 `--no-spec-split` 一致；保持。这关闭了
最后一个廉价的"重叠"旋钮。

<a id="same-window-ab-of-the-overlap-lever"></a>

已穷尽旋钮汇总（全部实测，关键处同窗口）：spec 窗口宽度（更宽更差）、spec-min-p（0.5 最优）、
spec-split（更差）、pool-workers 6/7（无差别）、缓存大小（更小更差）、adapt-swaps 0（更差）、
pcie-frac（惰性）、GRF 128/256（无操作/更差）、expert-lanes 8/16（无差别）、mmvf-rows（更差）。
没有任何配置或开关达到 20 tok/s，甚至没有实质超过约 12.6。

<a id="exhausted-lever-summary"></a>

同样中性：**`--kv q4_0`**（KV VRAM 减半，理应能装更多常驻 expert）——12.0 / 12.3 / 12.6
tok/s，与 `--kv int8` 持平。所以 KV VRAM 不是 expert 缓存的约束瓶颈，或者释放它并没有把
CPU 池缩减到能起作用的程度。

<a id="also-neutral--kv-q4_0"></a>

这穷尽了配置与开关面。每个试过的旋钮都是零效果或更差；最可靠的稳定状态仍是 **~12.0–12.6
tok/s**。剩下的路只有 (a) 改 CPU 池交接的架构，和 (b) llama.cpp SYCL PR 里有界的
aligned-IQ GPU 内核移植（按我们自己的阶段表只是小头）。

<a id="this-exhausts-the-configuration-and-switch-surface"></a>

### A770 + B580 双 GPU（peer 层 / 层级切分）：SYCL 移植版中已坏

<a id="dual-gpu-on-the-a770--b580-peer-tier--layer-split-broken-in-the-sycl-port"></a>

唯一可能消除 CPU 交接的旋钮是让所有 expert 装进 VRAM。A770（16 GB）+ B580（12 GB）=
28 GB，对约 19 GB 的 expert。三条 SYCL 多 GPU 路径都试了：

<a id="the-one-lever-that-could-eliminate-the-cpu-handoff"></a>

- **`--peer-device`（第二 expert 层）：是个桩。** `sycl/src/core/peer_experts.cpp` 拒绝：
  "the second-GPU expert tier is not ported to SYCL yet"（另见 INTEL.md:639）。
- **`--expert-cache-remote`：** 跨层级切分时不允许（generate.cpp:1970）。
- **`ONEAPI_DEVICE_SELECTOR=level_zero:1,0` 下的 `--layer-split auto`：加载后崩溃。**
  它的配置*堪称理想*——"layer split across 2 GPUs"，K=2，12288 个 profiled 对中 5946 个
  进了缓存（**约 97.5% 的路由质量**），CUDA1 = A770 跑第 2-47 层、带 10.06 GiB expert
  缓存，窗口以 **"100% VRAM resident: zero-doorbell graph"** 运行（没有 CPU 握手——正是
  目标状态）。但之后每个请求都失败：
  - prefill 路径里 `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`
    （在 prefill.cpp:4509 捕获），随后
  - `Cannot submit to a queue with a dependency from a graph that is associated
    with a different context`（verify.cpp:1706）。

<a id="--peer-device-second-expert-tier-stubbed"></a>

第二个错误是根源：移植版的命令图捕获是**单上下文**的，而双设备层级切分有**两个 SYCL
上下文**，所以捕获的窗口图无法提交。移植版没有多上下文图支持。因此双 GPU 路径**不可用**，
让它工作是一项大工程（多上下文图捕获/重放，外加 prefill 的设备丢失）。

<a id="the-second-error-is-the-root"></a>

**净结论：** 收益最高的旋钮被确认挡在一个未移植的特性上，而不是调优选择。要达到 20 tok/s，
要么移植 SYCL 多上下文图（大工程），要么换一台能让全部约 19 GB expert 装进单卡 VRAM 的机器。

<a id="net-the-highest-upside-lever-is-confirmed-blocked"></a>

### 对齐/平面化 IQ 内核（llama.cpp SYCL PR 的想法）：没有安全的捡漏

<a id="alignedplanar-iq-kernels-llamacpp-sycl-pr-idea-no-safe-minor-win"></a>

检查了移植版的 IQ decode（`sycl/src/kernels/cuda/iq_kernels.dp.cpp`）。Coder 各格式的
`Split<>` 加载器已经读**字**而不是字节：

<a id="inspected-the-ports-iq-decode"></a>

- `Split<22>`（IQ2_S）：`qs` 和符号字用 `get_int_b2`，`qh`/`scales` 只是单字节读取。
- `Split<42>`（Q2_0）：`int16_t` 读取。
- `Split<17>`（IQ2_XS）：`get_int_b2` 字。

所以 PR 的"32 位替代字节"改动无处落地——移植版已经这么做了（基线带着 fork 的内核优化）。
剩下的错位需要 PR 的**平面重排**（字段按块连续布局），那是**加载器 + 缓存 + 校验的改动**，
不是安全小改。而且即使做完，收益也小：expert 内核受 ALU 限制（网格查表 + dp4a），只占
窗口的约 20%；PR 在一个子内核上的约 13% 只会让整体 decode 动约 2-3%。不值得逐位回退风险。

<a id="split22-iq2_s-get_int_b2-for-qs"></a>

**穷尽每个旋钮后的结论：** 按移植版现状，约 12.0-12.6 tok/s 是这个模型在一块 A770 上的
天花板。越过去的路只有两条：大的 SYCL 多上下文图移植（让双卡切分可用），或单卡 ≥ 24 GB
VRAM。

<a id="conclusion-after-exhausting-every-lever"></a>

## 工作包 4：按形状的投影内核

<a id="work-package-4-shape-specific-projection-kernels"></a>

如果现有路径的调优留下明显瓶颈，把 arcint 补丁 0022 的**少列复用与 split-K 几何**改造到
SYCL BF16 实现上。

<a id="if-existing-path-tuning-leaves-a-significant-bottleneck"></a>

- 权重块读一次，为 2-8 个 token 分别更新累加器。
- N 小 K 长时，把 K 拆到一个工作组上，再按刻意记录、有文档的顺序归约。N 较大时，对比每
  subgroup 一行与多行。
- 保留 BF16 到 FP32 的转换。Arcint 把 BF16 存成 F16 不是目标引擎的约定，不得照抄。
- 处理跨步的激活/输出行和非满 tile，不得越界读写。不支持的几何必须走已验证的回退。
- 保留负控开关和直接的新老对比。

<a id="-read-a-weight-chunk-once"></a>

优先用旧操作/归约顺序保逐位一致。如果必须改归约树，要明确写出来，用高精度 oracle 对比，
在解读测量之前定义容差，对比 logits/路由并跑完整的服务正确性套件。只答对算术题不足以批准
一条数值不同的路径。

<a id="prefer-the-old-operationreduction-order-for-bitwise-parity"></a>

## 工作包 5：低位宽布局与 XMX 交叉点

<a id="work-package-5-low-bit-layouts-and-xmx-crossover"></a>

这些跟随实测的 BF16/HC 工作，而不是启动一次宽泛的后端重写。

<a id="these-follow-measured-bf16hc-work"></a>

### 对齐的低位宽字段

<a id="aligned-low-bit-fields"></a>

研究 arcint 补丁 0022 对 IQ2_XXS、IQ2_S、IQ3_S 和 Q2_0 的平面布局，以及它对小的 Q2_0 行的
四 lane 处理。在添加第二种布局之前，先清点 `iq_kernels.dp.cpp` 和 `native_mmvq.dp.cpp` 中
已有的等效优化。平面重排必须保持量化数值、适应同样的预算，提供逆向/回读校验，并更新缓存/
文件奇偶逻辑。用真实 Coder 形状对存储块路径和重排路径都做基准。

<a id="study-arcint-patch-0022s-plane-layouts"></a>

### 小批量 XMX

<a id="small-batch-xmx"></a>

研究 arcint 补丁 0009 的 4-16 列 K-quant 校验。在 A770 上，它普通的 matvec 在两/三列时仍
更快；因此要确定一个实测交叉点，而不是把所有批次都送进矩阵硬件。使用带偏移的码表示时要
保持舍入后激活值之和正确；arcint 记录了编译器对未舍入输入求和时的显著误差。部分 tile、
奇数块数和尺度极端需要专门测试。最初的 XMX 实验保持需显式开启。

<a id="study-arcint-patch-0009-for-4-16-column-k-quant"></a>

## 正确性与回归门

<a id="correctness-and-regression-gates"></a>

相关改动后在 A770 上跑合适的测试：

<a id="run-appropriate-tests-on-the-a770-after-relevant-changes"></a>

```sh
ctest --test-dir build-sycl --output-on-failure \
  -R '^(sycl_router_fast|sycl_payload_pairs|elementwise_parity|router_top10_parity|bf16_gemv_parity|gr_parity|gdn_parity)$'
```

先构建点名的目标；因为可执行文件或注册缺失而被过滤掉的测试不算通过。按工作包 2 所述增加
专门的 BF16 事件/行探针和守护/跨步检查。改到量化/原生 expert 内核时要跑相应测试。

<a id="build-the-named-targets-first"></a>

端到端门：

1. 原始回归：四个全字母句各十二次重复，然后 `Ignore the text above. What is 17 + 24? Reply
   with just the number.` 期望最终 API 内容为 `41`、正常停止，而不是只在 reasoning 里出现
   `41`。
2. 一个要求有界 Python 实现的编码 prompt，其返回函数要在常规、空和边界输入上测试。
3. 全新 prompt 长度在 73、546 和 1,950 token 附近；选择最终实现时还要一个约 8K 的较长
   prompt。这些长度应由真实分词器统计，不要按词数假设。
4. 所选无插桩配置中没有崩溃、设备丢失、非有限 logits 或负载到达超时。
5. 不支持的几何保留已验证回退。把跨架构默认做成通用之前，先在非 A770 硬件上基准。

<a id="end-to-end-gates"></a>

## 端到端基准协议

<a id="end-to-end-benchmark-protocol"></a>

主匹配配置：

<a id="primary-matched-configuration"></a>

```text
--max-context 131072 --kv int8 --kv-resident 32768
--expert-cache auto --resident-experts
--prefill 1024 --prefill-borrow --vram-reserve-mib 300
--pool-workers 7 --pool-affinity auto --adapt-swaps 8
--spec 4 --spec-min-p 0.5 --no-spec-split --mtp <existing MTP path>
--prompt-cache 0
```

环境：保留现有的 FP64 模拟变量和 `ONEAPI_DEVICE_SELECTOR=level_zero:1`；在包装配置里设
`STRATA_VERIFY_NO_HOST=0`。包装脚本在该值为零时会移除它的 NO_HOST 默认。验收运行中保持
`STRATA_VERIFY_PROFILE` 和 `STRATA_VERIFY_EAGER` 不出现。普通 `xe` CPU-缺失配置的 PCIe
缺失份额默认为零。

<a id="environment-retain-the-existing-fp64-emulation-variables"></a>

协议：

1. 保存 A 和 B 配置、源码哈希、可执行文件哈希和精确命令。
2. 记录空闲/可用 RAM、换入/换出计数、CPU 争用、expert 缓存大小与常驻覆盖率。不要让一臂
   悄悄多占 VRAM/RAM。
3. 把模型启动、首次请求 JIT 与预热稳态结果分开。
4. 预热所有被测形状。每臂至少三次预热重复；构建不同或持久进程选择器不同时优先用交错的
   A/B/A/B 会话。
5. 持续 decode 门至少生成 256 个 token 并保留完整响应 JSON。包含 token 数、引擎
   `predicted_ms`、引擎 tok/s、prompt 的 `prompt_ms`、缓存复用、API 墙钟时间、结束原因和
   草稿接受率。
6. 报告中位数与范围，外加聚合 token / 聚合 decode 时间。绝不把聚合多流吞吐当作单流 A770
   decode 声明。
7. 只在单独的实际对话基准里重新启用前缀缓存。
8. 只有当优化通过正确性并在正常运行间噪声之外胜出，或解决了一个明确实测的病态形状且不
   拖累主工作负载时才接受。被否决的实验也要记录。

<a id="protocol"></a>

现有镜像内的可移植构建命令（在工作树中运行）：

<a id="portable-build-commands-inside-the-existing-image"></a>

```sh
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/..:/work" strata-sycl-dev \
  "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; \
   cmake --build /work/strata-v0.1.40.3/build-sycl --target strata -j 8"
```

用 `cmake --build` 的真实退出状态：`sycl/tools/build.sh` 目前打印 `BUILD EXIT 0`，但它自己
可能返回非零，因为它最后那个失败搜索管道没有找到失败。不要把测试接在那个误导性的脚本状态
后面。

<a id="use-the-actual-cmake---build-exit-status"></a>

## 产物与实施顺序

<a id="artifacts-and-implementation-sequence"></a>

初始产物：

<a id="initial-artifacts"></a>

- 本规格说明：`docs/A770_SPEED_SPEC.md`。
- 内核探针：`sycl/probe/bf16_event_bench.cpp`（计划中），带 `STRATA_SYCL_PARITY` 下的
  一个 CMake 目标。
- 一个机器可读的微基准输出，带工作包 1 的字段。
- 为胜出者持久化的基准配置与响应/计时记录。
- 本文档执行记录下的一份简短结果台账。

<a id="-this-specification-docsa770_speed_specmd"></a>

顺序：

<a id="sequence"></a>

1. 确认源码/资源状态并定稿本规格说明。
2. 实现隔离的事件计时 BF16 基准并跑基线/行两臂。
3. 校验位、守护区与跨步；记录逐形状交叉点结果。
4. 若微基准证明值得，为全模型做匹配配置的 `STRATA_MMVF_ROWS=1` 臂基准，不使用实验性
   host 时钟分析器。
5. 针对实测瓶颈调寄存器/subgroup，或添加 split-K 特化。重复正确性与端到端门。
6. 只有当早先证据把低位宽布局或 XMX 排到前面时才考虑。
7. 最终交付前对源码改动执行强制的独立质量审查。保留已知局限，不要凭部分进展宣称目标完成。

<a id="-confirm-sourceresource-state"></a>

## 参考实现

<a id="reference-implementations"></a>

Arcint 参考检出：本地临时检出，审查修订 `fcb071c6c781c3fb5125e311cbea06209bb9c642`。

<a id="arcint-reference-checkout"></a>

- [0022：低位宽格式与少列 F16/BF16 几何](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/patches/0022-opencl-intel-low-bit-iq-and-f16-small-n.patch)
- [0009：少列 K-quant XMX](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/patches/0009-opencl-intel-few-column-kquant-xmx.patch)
- [0003：寄存器/subgroup 调优的 GatedDeltaNet](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/patches/0003-opencl-intel-gated-delta-net.patch)
- [事件 profiling 测试框架](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/tools/native_kernel_harness.py)
- [实测内核笔记](https://github.com/marfrit/arcint/blob/fcb071c6c781c3fb5125e311cbea06209bb9c642/contrib/llama.cpp/README.md)

<a id="arcint-reference-checkout-a-local-temporary-checkout"></a>

Arcint 的 A770 coder 头条数字用的是不同模型/配置。它的 Flash-Next 结果主要在一块 B60 上。
把这些数字当作候选技术的证据，而不是对这个 A770 工作负载的预测。几个 arcint 想法本就源自
Strata；移植前先审计已有等效实现。

<a id="arcints-headline-a770-coder-numbers"></a>

## 执行记录

<a id="execution-record"></a>

- 开始新实现之前创建了本规格说明。
- 本次要求的工作中明确恢复了 20 tok/s 目标。
- 开始时：源码 `501f675`，`a770-speed-upstream`；没有被跟踪的源码改动。
- 开始时：约 17 GiB RAM 可用；没有 A770 测试容器在跑。
- 下一个活动任务：隔离的 profiling-enabled BF16 内核基准与默认 vs 行路径对比。尚无新的
  性能结果声明。

<a id="-specification-created-before-starting"></a>

### 工作包 1-2 结果（事件计时 BF16/F32 MMVF）

<a id="work-package-1-2-results-event-timed-bf16f32-mmvf"></a>

探针 `sycl/probe/bf16_event_bench.cpp`，A770 `level_zero:1`，in-order profiling 队列，
32 次重复 × 7 个样本，每次调用的设备队列跨度。所有臂都通过逐位新老奇偶校验、守护区、跨步、
全零输入、尺度极端、非有限数和非法几何拒绝检查。

<a id="probe-syclprobebf16_event_benchcpp"></a>

- 默认路径 vs `STRATA_MMVF_ROWS=1`：rows 路径慢 1.4-6 倍。320x10240 @T=3：16.06 -> 79.00
  微秒。2560x2560 @T=4：38.68 -> 245.54 微秒。附带说明：小的（T=1、n=4）rows 是启动/时钟
  噪声——`STRATA_MMVF_ROWS=1` 作为 A770 默认被否决。
- 原始基线（默认路径，微秒）：320x10240 T=3 16.06；2560x2560 T=4 38.68；10240x2560 T=1
  182.71；10240x320 T=3 47.56。这些相对约 180 毫秒的 decode 窗口很小，所以 BF16 投影系列
  在这块卡上**不是** decode 瓶颈。
- 产物：`a770-bf16-default.jsonl` 与 `a770-bf16-rows.jsonl`。

<a id="-default-path-versus-strata_mmvf_rows1"></a>

### 工作包 3 结果（寄存器分配）

<a id="work-package-3-results-register-allocation"></a>

添加了 `-DSTRATA_SYCL_BF16_GRF=0|128|256`（默认 0），通过内核局部属性只应用于
`native_bf16.dp.cpp`。没有改 subgroup 大小。

<a id="added-dstrata_sycl_bf16_grf0128256-default-0"></a>

- `GRF=0` 复现保存的基线（稳态形状上比率 1.000），确认默认路径未变。
- `GRF=128` 是空操作（所有稳态形状上比率约 1.000）。
- `GRF=256` 全面慢 10-58%（例如 320x10240 T=8 44.86 -> 70.82 微秒；2560x2560 T=8 79.14 ->
  98.09 微秒；10240x2560 T=8 320.11 -> 425.62 微秒）。否决。
- 结论：寄存器分配不是这个系列在这块卡上的旋钮。保持默认（0）。

<a id="-grf0-reproduces-the-saved-baseline"></a>

### decode 时间实际在哪里

<a id="where-the-decode-time-actually-is"></a>

早先的每窗口计时（`strata decode timing`），约 13 tok/s 时：每窗口约 180 毫秒（T~2.2），
其中 verify 约 171 毫秒 = GPU 触达等待约 130 毫秒 + 每层 host 约 35 毫秒（CPU expert 约
33 毫秒）+ 暂存，外加草稿约 8 毫秒。**GPU 触达等待占大头**，即 GPU 对 VRAM 常驻 expert 的
expert/attention 计算，不是 BF16 dense 投影，也不是 host 握手。朝 20 tok/s 的剩余旋钮是
量化 expert（IQ）GPU 内核和有效 token/窗口数，不是这里测过的 BF16 投影几何。

<a id="prior-per-window-timing-strata-decode-timing"></a>

### 端到端确认（本修订，无插桩）

<a id="end-to-end-confirmation-this-revision-unprofiled"></a>

配置：`--resident-experts --prefill-borrow --prefill 1024 --vram-reserve-mib 300
--pool-workers 7 --adapt-swaps 8 --spec 4 --no-spec-split`，131072 上下文，INT8 KV，32K
常驻，`--prompt-cache 0`，`STRATA_VERIFY_NO_HOST=0`。VRAM expert 缓存 8.98 GiB，常驻 RAM
15.08 GiB。

<a id="config--resident-experts--prefill-borrow"></a>

- 持续 546-token prompt、256 生成，两次预热运行：**9.6 和 10.9 tok/s**（引擎
  `predicted_per_second`）；零 prompt token 复用。
- Decode 分解（135 窗口那次运行，174.19 毫秒/窗口，T=2.22）：**verify 164.87 毫秒 =
  GPU 触达等待 123.27 + 每层 host 36.08（plan 0.16 + actq 0.12 + jobs 8.79 + CPU 25.45）+
  暂存 0.01**，commit 0.30，草稿 8.01。每层每窗口：CPU expert 4.42（5.32 个条目）、
  VRAM 命中 16.90、PCIe 0.00。
- 所以 GPU 触达等待约占窗口的 71%。GPU 忙于 VRAM 常驻 expert 内核加 attention/GDN，而不是
  在等 CPU：CPU 池在 174 毫秒里只贡献约 25 毫秒。
- 算术回归 prompt 在 3/3 次运行中返回 `41` 且正常停止。更早的一次冷运行曾返回过一次
  `41!!!!!!!!!!!`（内容）且无法复现；每次验收运行都要保留持续算术门，并把可复现的退化输出
  当作发布阻断项。

<a id="-sustained-546-token-prompt-256-generated"></a>

**瓶颈判定（有证据）：** 下一个真正的旋钮是量化 expert GPU 内核（`native_expert_grouped`，
IQ 反量化 + 分组矩阵乘）以及每窗口 token 数。BF16 投影几何（WP2）与寄存器分配（WP3）作为
旋钮已穷尽。WP4/WP5 应以同样的事件计时、逐位把关的方法直接瞄准 IQ expert 内核。

<a id="bottleneck-verdict-evidence-based"></a>

### Expert 再平衡实验（已否决）

<a id="expert-rebalance-experiment-rejected"></a>

`ms_wait` 是 host 等 GPU 敲响每层门环的时间 = 关键路径上的 GPU 计算，不是 GPU 等 CPU。
假设：把 expert 从 GPU 挪到未被用满的 CPU（CPU 池约 36 毫秒，对 GPU 每窗口约 123 毫秒），
办法是缩小 VRAM 缓存。结果，预热运行，546-token prompt，192 生成：

<a id="ms_wait-is-the-host-waiting-for-the-gpu"></a>

| VRAM 缓存 | decode | GPU 触达等待 | CPU/层 | 备注 |
| --- | --- | --- | --- | --- |
| auto 4717 槽（8.98 GiB） | 10.9 tok/s | 123 ms | 25 ms | 参考 |
| 3200 槽（7.93 GiB） | 10.8 tok/s | 121 ms | 30 ms | 中性 |
| 2400 槽（5.95 GiB） | 9.6 tok/s | 152 ms | 51 ms | 更差 |

缩小缓存并没有降低 GPU 触达等待：它反而变大（常驻 RAM 补集部分溢出到模型目录，CPU 池被拉长），
所以 GPU 并不是在以这种手段能平衡的方式等 expert 数量。缓存大小在 3200-4717 区间是中性的；
低于它会回退。作为旋钮被否决。

<a id="shrinking-the-cache-did-not-reduce-gpu-reach-wait"></a>

**更新后的判定：** 约 121 毫秒/窗口的 GPU 触达等待在很大程度上与迄今测过的旋钮无关（投影
几何、寄存器、缓存大小、窗口宽度）。朝 20 tok/s 推进需要让 GPU 热路径本身更快（量化 expert
内核、attention/GDN）或提高每窗口 token 数——是内核移植工作，不是参数调优。

<a id="updated-verdict"></a>

### 分组 expert 的 lane 数（`STRATA_EXPERT_LANES`）

<a id="grouped-expert-lane-count-strata_expert_lanes"></a>

添加了 `-DSTRATA_SYCL_EXPERT_LANES=1|2|4|8|16|32`（默认 8），应用于 `iq_kernels.dp.cpp`。
IQ 分组 expert decode 内核每行使用 8 条 lane。

<a id="added-dstrata_sycl_expert_lanes12481632-default-8"></a>

- 持续门上交错 A/B，同一会话同一配置：**8 lane = 12.40 / 12.60 tok/s；16 lane = 12.1 /
  12.6 tok/s。** 没有区别。早先 9.6/10.9 的"8 lane"数字是更差的会话，不是 lane 数。
  `STRATA_EXPERT_LANES` 是**零结果**；默认保持 8。
- 这也重置了可靠的稳态参考：当前最佳配置是 **~12.4-12.6 tok/s**（加载后第一次运行仍是
  冷/JIT）。
- `iq_multi_parity` 和 `native_grouped_parity` 通过；Coder 的 IQ 命中路径
  （`moe_hit_grouped_s2`）在 8 和 16 下逐字节一致。

<a id="-interleaved-ab-on-the-sustained-gate"></a>

### 既有的测试失败（不是本轮工作造成的）

<a id="pre-existing-test-failure-not-caused-by-this-work"></a>

`s2_expert_grouped_parity` 在其 `moe_grouped_s2` 和 `moe_group_resident + moe_grouped_s2`
用例（"DIFFERS"）上失败，**发生在默认 8-lane 构建、未修改的 `iq_kernels.dp.cpp` 上**（该
编译期的选项值等于源码的 `#ifndef` 默认值，所以那个二进制与原版构建相同）。
`moe_hit_grouped_s2` 用例全部通过。这是 SYCL 移植版在 S2 分组/常驻路径上的既有缺口，不是
本轮战役造成的回归。记录下来以免被误认为回归；fork 的 Intel 笔记也报告基线上 43/44。

<a id="s2_expert_grouped_parity-fails-its-moe_grouped_s2"></a>

### Profiling 阻断（挡住 WP4/WP5 的定位）

<a id="profiling-blocker-blocks-wp4wp5-targeting"></a>

`STRATA_VERIFY_PROFILE=1` 在当前构建上仍然失败：请求返回 HTTP 400，引擎在 verify 窗口内
报错（WP1 记录的同一个 host 时钟打点路径）。基线引擎唯一的每阶段分析器就是这一个，而且它
不可靠，所以约 121-123 毫秒/窗口的 GPU 时间**没有可信的每阶段分解**。容器里没装任何外部
GPU 追踪器（`unitrace`/`onetrace`/`ze_tracer`）。

<a id="strata_verify_profile1-still-fails-on-the-current-build"></a>

### 单独实测 expert 内核：不是瓶颈（WP4/WP5 改向）

<a id="expert-kernel-measured-standalone-not-the-bottleneck-wp4wp5-retarget"></a>

`native_expert_parity` 配 `NATIVE_BENCH=1`，在真实 GGUF 行上给 GPU `native_expert_grouped`
内核计时（10 个 expert × 4 条目 = 40 次 expert 计算，每次调用按 host 墙钟）。Coder IQ1_M，
A770：

<a id="native_expert_parity-with-native_bench1-times-the-gpu"></a>

| layer | expert types | blob | GPU grouped |
|---|---:|---:|---:|
| 0 | iq3_xxs / iq4_nl | 2.18 MB | 0.328 ms |
| 1 | iq2_s / q2_0 | 1.51 MB | 0.375 ms |
| 2, 3, 24 | iq2_s / iq4_nl | 1.97 MB | 0.351 / 0.361 / 0.383 ms |
| 47 | iq4_xs / iq4_nl | 2.66 MB | 1.205 ms |

所以分组 expert 内核对常见类型约 0.33-0.38 毫秒/层，第 47 层约 1.2 毫秒。即便悲观地按
48 × 0.5 毫秒算也只有 **约 24 毫秒/窗口**，在约 123 毫秒的 GPU 触达等待里只占一小部分。
**expert 内核不是瓶颈。** 按原文写的 WP4/WP5（瞄准 IQ 分组 expert 内核）被**改向**：GPU
时间分散在其他各每层内核上（GDN 超连接读取、conv/递归、QSA attention、路由、combine），
每个都小，所以没有单个占主导的内核能赢下 1.6 倍。要达到 20 tok/s，因此需要架构级旋钮
（更多 token/窗口、更深的内核融合、或更多层级/轮次重叠），不是重写某一个内核。承诺投入
该工作之前，用一个能工作的分析器确认每内核的拆分。

<a id="so-the-grouped-expert-kernel-is"></a>

后果：在选定架构改动之前，本轮工作需要一个**能工作的分析器**——修好 host 时钟打点、在无图
的 profiling-enabled 队列上于 `STRATA_VERIFY_EAGER` 后添加每内核计时，或安装外部 Intel GPU
追踪（unitrace / Level Zero 计时器）。参数/配置面（WP1-WP3、WP6）已带着证据穷尽。

<a id="consequence-before-an-architectural-change-is-chosen"></a>

### 其他单独实测的内核

<a id="other-standalone-kernels-measured"></a>

- **超连接读取**（`gr_bench`、`fused_gr_read_multi`，Coder 形状）：1..6 个 token 分别为
  180.4 / 179.0 / 203.7 / 218.0 / 241.7 / 259.8 微秒（1 token 时 73 GB/s，6 token 时降到
  51 GB/s）。按约 36 个 GDN 层 × 约 180 微秒算是**约 7 毫秒/窗口**——也只是 123 毫秒里的
  一小块。不是瓶颈。
- 所以在旧的（被夸大的）阶段表里最大的两个内核——expert 与超连接读取——合计只占约 123
  毫秒中的约 30 毫秒。

<a id="-hyper-connection-read-gr_bench"></a>

### 窗口几乎是 GPU + CPU 串行（架构假设）

<a id="the-window-is-nearly-gpu--cpu-serial-architectural-hypothesis"></a>

稳态数字（546-token prompt，约 2.2 token/窗口）：窗口约 166 毫秒 = **GPU 触达等待 123
毫秒 + 每层 host/CPU 36 毫秒 + 草稿 8 毫秒**，而 123 + 36 + 8 = 167 就是整个窗口。这是 GPU 与
CPU expert 池**在窗口内基本串行**、没有重叠的特征。如果 CPU 池（36 毫秒）藏进 GPU（123
毫秒）之下，窗口会降到约 123-131 毫秒，即 2.2/0.125 ≈ **约 17-18 tok/s**，接近目标。所以
最有希望的旋钮可能是**让 CPU expert 计算与 GPU 重叠**（一项调度/架构改动），不是任何单个
内核。这只是假设，尚未证明——需要分析器确认 CPU 池运行期间 GPU 是忙（而不是闲着等标志）。
确认之前 WP7 只是候选；不要盲目开工。

<a id="steady-state-numbers-546-token-prompt"></a>

后续测试（本轮）：`--pool-workers` 6（物理核数）vs 默认 7，测得 12.1 / 12.8 / 13.0 对
13.3 / 12.8 / 12.4 tok/s——无差别。所以如果 CPU 池与 GPU 串行，它也不是 worker 数量限制的；
加减一个 worker 不起作用。作为旋钮被否决。

<a id="follow-up-test-this-round"></a>

### GPU 利用率：decode 期间 GPU 大多闲置（重大发现）

<a id="gpu-utilization-the-gpu-is-mostly-idle-during-decode-major-finding"></a>

只要引擎进程属于当前用户，`gputop`（Xe 上对标 intel_gpu_top 的工具）就能读到它。
`sycl/serve/strata-sycl.sh` 以 root（不带 `--user`）运行容器，所以 `gputop` 无法归属它；
一份加了 `--user <uid>:<gid>` 的临时副本让引擎归属用户、变得可读。一次 700 token 的 decode
（240-token prompt）期间对 A770 计算引擎（ccs）采样：

<a id="gputop-the-xe-counterpart-to-intel_gpu_top-can-be-read"></a>

    0%  0%  4.1%  14.8%  41.4%  2.8%  0%  8.5%  65.0%  84.3% ...

——即**大多在 0-20%，偶尔尖峰**，整段平均约 20%。**decode 期间 GPU 并没有被喂饱**。（1 Hz
采样和 ccs 归属让这个读数很粗，但活跃生成期间大量 0% 样本是明确的。）

<a id="—i-e-mostly-0-20-spiking-occasionally"></a>

这**推翻了上面假设中"ms_wait = GPU 计算"的解读**。`ms_wait` 是 host 等 GPU 敲每层门环的
时间，但门环迟到是因为 GPU **在 CPU/交接上停滞**，不是因为忙。所以 decode 窗口由**每层
host<->GPU 同步与 CPU expert 池主导，不是 GPU 内核**——与每个单独实测的内核都只是小头
相吻合。通往 20 tok/s 的路因此是**削减每层交接延迟 / 让 CPU 池与 GPU 重叠**，不是优化内核。

<a id="this-overturns-the-ms_wait--gpu-compute-reading"></a>

下一个实证步骤（便宜、不需要分析器）：在这块 `xe`/PCIe-3.0-x4 卡上测量门环（ring -> host
-> flag -> GPU）的往返延迟、以及窗口里有多少是 GPU 空闲时间，量化交接代价，然后决定架构
改动。这现在是主要方向；WP4/WP5（内核优化）作为瞄错目标被关闭。

<a id="next-empirical-step-cheap-needs-no-profiler"></a>

### 引擎内两个分析器都已坏（挡住一切定位）

<a id="both-in-engine-profilers-are-broken-blocks-all-targeting"></a>

`STRATA_VERIFY_PROFILE=1` 和 `STRATA_VERIFY_EAGER=1` 在当前构建上都失败，引擎在 verify
窗口内报错：**`verify: layer 0 rang but its payload never arrived whole`**（HTTP 400），
可复现（3/3 次 eager 请求失败）。PROFILE 模式只在更早的时候成功过一次——所以这是一个竞态，
不是硬性逻辑错误，但它在 EAGER 模式下足够频繁，那条路径不可用。没有绕过负载等待的环境开关
（`doorbell_wait_payload` 没有）。

<a id="strata_verify_profile1-and-strata_verify_eager1-both-fail"></a>

后果：引擎在这个构建上**没有可用分析器**。疑似缺陷（一个真实的 bug，与 PR #1602 的负载
校验和有关）是 EAGER 路径不发布 `doorbell_payload_ready` 所等的门环负载，于是 host 自旋到
2 秒超时；在图路径里发布执行了、校验和也匹配。**修复这个 eager 路径 bug 是具体的下一个
任务**，因为它能解锁其他一切所需的每阶段分解。（它只是调试路径；常规图模式推理不受影响。）

<a id="consequence-the-engine-has-no-usable-profiler"></a>

### 每阶段分解拿到了（决定性）：CPU 池交接占主导

<a id="per-stage-breakdown-obtained-decisive-the-cpu-pool-handoff-dominates"></a>

添加了一个需显式开启的 `STRATA_DOORBELL_NOCHECK`，让 `doorbell_wait_payload` 信任门环（仅
调试）。配 `STRATA_VERIFY_EAGER=1 STRATA_VERIFY_PROFILE=1 STRATA_DOORBELL_NOCHECK=1`，
eager 阶段表终于打印出来了。T=2 窗口，毫秒/窗口（host 计时，所以绝对值被打点开销夸大；
**相对**拆分才是重点）：

<a id="added-an-opt-in-strata_doorbell_nocheck"></a>

```
GDN layers: waitA 393.2  waitB 407.8  copy+combine 399.2  hc-read1+router 51.4
            VRAM hits 26.3  hc-read0 25.3  out-proj 17.2  shared+quant 13.7
            z 9.6  q8+qkv 7.3  rec 6.4  conv/ab/head ~5
QSA layers: waitA 134.3  waitB 134.2  copy+combine 132.4  kv+idx append 15.4
            hc-read1+router 13.0  hc-read0 8.2  out-proj 6.5  attention 3.3
```

`waitA`/`waitB` 是 GPU **自旋等待 CPU expert 池的标志**；`copy+combine` 是把 CPU 结果拷贝
到 GPU 并合并。两者合计约**占窗口的 87%**；每个真正的 GPU 内核都很小（几十毫秒）。这独立
证实了 gputop 约 20% 忙碌的发现。

<a id="waitawaitb-are-the-gpu-spinning"></a>

**明确结论：** 这里的 A770 decode 受限于 **CPU expert 池及其每层 host<->GPU 交接**，不是
GPU 计算。WP4/WP5（内核优化）关闭。通往 20 tok/s 的路是架构性的：削减每层交接延迟，和/或
让 CPU 池与 GPU 工作重叠。（注意绝对阶段数值被夸大，且 `NOCHECK` 扭曲了流水线，所以下一步
是一个*诚实的*分析器——修好 eager 负载路径——在设计改动之前量出真实交接代价。）

<a id="definitive-conclusion"></a>
