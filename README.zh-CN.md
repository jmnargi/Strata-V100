# Strata-V100

[English](README.md) · **简体中文**

**[Strata](https://github.com/Niko1221/Strata) 的社区维护分支（fork），专注于 V100 支持。** 本分支让 NVIDIA Volta（`sm_70`）支持保持可用，在 Tesla V100 硬件上开发并实测性能改动，并定期合并上游改进，以获得 V100 之外的新特性和修复。

Strata 是 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的开源本地运行时。它把 GPU、系统内存和存储组合起来，让这个大模型能在个人电脑上运行。本分支保留了 Strata 项目的全部内容：其他受支持的 NVIDIA 和 AMD 硬件、Web 界面、OpenAI 兼容 API、工具、图片输入以及上游特性同样重要。

> **向 Niko（[@Niko1221](https://github.com/Niko1221)）和原版 Strata 的贡献者致以诚挚谢意。** 本项目建立在他们的工作之上：Strata 的模型运行时、架构、特性，以及让这个模型得以本地运行的所有工作，都来自上游项目。没有那个基础，就不会有 V100 分支。感谢你们让它成为可能，也感谢你们持续开发 Strata。

[上游 Strata](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussions](https://github.com/jmnargi/Strata-V100/discussions)

## 本分支的关注点

- **Tesla V100 / Volta 支持（`sm_70`）**：保住 V100 的正常构建和运行时支持。V100 构建使用 CUDA 12.x 工具包；CUDA 13 不编译 `sm_70` 代码。
- **测试机：** 两张 Tesla V100-PCIE-16GB；GPU 0 走 PCIe Gen3 x4，GPU 1 走 Gen3 x16。下文的双卡流水线测量使用两张卡。历史单卡基准只用了 x16 那张。两条完整 Gen3 x16 链路的系统尚未在本机测量。
- **上游整合**：这不是 Strata 的独立替代品。上游改动会定期合并进来，并检查、修正 V100 兼容性。上游新特性可能有高于 `sm_70` 的硬件要求；它们不会自动加速 V100。
- **更广泛的 Strata 社区**：欢迎针对其他受支持平台的改进、错误报告、测试、文档和贡献。

对 V100 的支持不意味着每个上游特性或每种其他 GPU 配置都在 V100 上测过。在依赖某个硬件相关特性之前，请先查看相关代码、发布说明和测量报告。

## 上游选择性同步 — 2026 年 10 月 19 日

本分支从上游提交 `3d1fb480` 引入了共享的 CUDA 引擎和服务器改动，保留了本分支的 Volta 内核、流水线 K/V 增长和会话缓存改动，排除了无关的桌面安装包以及独立的 Metal、AMD、Intel 后端更新。见[同步范围](improvements/RELEASE-2026-10-19-V100.md)。`sm_70` 请使用 CUDA 12.8。引入某个上游选项并不代表它在 V100 上有性能收益。

## 上游 v0.1.39 整合

本次更新整合了 [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39)（含其间的各版本），保留本分支的 V100 tensor-core 专家、注意力改动、非对称 GPU 切分和磁盘会话缓存。

- **并行请求：** 在模型配置中设置 `"parallel": 2`，或使用 `./setup.sh --parallel 2`。Setup 会为新的 Volta 配置写入该默认值，并保留已有设置。引擎可以同时解码多个独立会话。多余的请求等待空闲槽位。槽位会占用额外显存，并可能降低单请求速度；如果两个槽位会话放不下，引擎会明确说明，并退回一次只服务一个请求，配置保持不变。精确性设置、显存开销和限制见[批处理槽位](docs/zh_CN/BATCHING.md)。
- **API 与界面：** 本次更新新增 OpenAI Responses API、模型设置、会话缓存监控和按槽位的指标。见 [API 详情](docs/zh_CN/DETAILS.md#using-it)。
- **性能与内存：** 本次更新包含 verify-pass 启动次数削减、字节粒度提示环、Linux 大页、RAM 预算修正和多 GPU 选项。显式 `--layer-split` 现在会自动只加载各阶段自己那些层的 dense 权重（上游的 `--trim-stage-weights` 标志和 `STRATA_STAGE_TRIM` 已移除）；腾出的显存归专家缓存。对新显卡的优化不意味着 V100 提速。会改变舍入的可选路径使用前必须实测。
- **其他平台：** 上游的 Intel Arc、较老 CPU 和更多 AMD 路径仍然可用。本机不为这些平台提供硬件验证。

V100 需要 CUDA 12.x 和包含 `sm_70` 的构建。不要在 V100 上安装 CUDA 13 引擎。已有安装建议显式配置并行；更新不会覆盖模型配置。

**在两张 V100 上实测（2026 年 10 月 4 日，最终构建）。** 整合后的构建与之前安装的运行时做受控的前后对比，一次一个请求（无批处理槽位），冷启动，Qwen3.8-Flash-Next Q2_0 配 int8 KV、524,288 token 上下文，20/28 层切分和 MTP（`--spec 8`）保持不变；每个长度两个全新请求，取中位数：

| 目标提示长度 | 之前 (tok/s) | 整合后 (tok/s) | 变化 |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

每个数值是两个全新请求（`reused=0`）的中位数。增益很小；每个长度两次重复不构成统计结论。六次配对请求的首 token 全部与基线一致。不声称完整回复完全相同：自适应专家层在整合后的构建上把不同的专家组合留在 GPU 和 CPU 上，两条路径的舍入不同。不声称 decode 或质量有任何变化。这些是最终构建的实测值（引擎 SHA-256 `e9ad337a…`），完整数据见[完整报告](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md)。

524,288 token 上下文下的并行槽位取决于切分方式和最后一级剩余的显存；实测的回退行为（请求 2 个槽位、实际一次服务一个请求、配置不动）见 [docs/BATCHING.md](docs/zh_CN/BATCHING.md)。

## 双 V100 流水线性能 — 2026 年 10 月 15 日

实测服务配置启用 `--pipeline-windows 2`，GPU 顺序 `0,1`，层切分 `16`。使用两张 16 GiB 的 V100（175 W）、Ryzen 5 3600、原生 IQ3_S 混合格式模型、int8 KV，以及 `--spec 8` 的 MTP。配置的上下文容量仍为 524,288 token。实测的最大提示为 32,768 token。

**长回复安全性：** 把本配置与 `STRATA_KV_GROW=1` 组合使用时，请使用包含[流水线 KV 增长修复](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)的构建。更早的流水线循环在长回复期间可能访问未映射的 KV 单元。修复会在扩展映射之前排空投机工作，然后再恢复流水线。下面短输出的速度测量没有触及该边界。

### 全新提示的 prefill（至 32K）

每个数值是三个精确 token 数请求的中位数，无提示复用，输出上限 16 token。prefill 时间不含回复生成。吞吐量为提示 token 数除以 prefill 时间中位数。

| 提示 token 数 | 之前 prefill 时间 | 流水线 prefill 时间 | 之前 tokens/s | 流水线 tokens/s | 吞吐变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

全新提示的 prefill 基本不变。这些微小差异不足以说明存在有意义的回退。

### 多轮对话 decode

独立的 HTTP 对比使用 8,828 token 的初始提示和六轮续写，上下文达到约 11.9K token。每个数值是四个请求的中位数，交替轮次反转两臂顺序，输出上限 256 token。新增提示 token 指该轮实际处理的 token 数，不是上下文总大小。

| 轮次 | 新增提示 token | 之前 decode tokens/s | 流水线 decode tokens/s | decode 变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 121 | 70.20 | 79.45 | +13.18% |
| 2 | 321 | 70.60 | 82.05 | +16.22% |
| 3 | 1,086 | 76.60 | 79.00 | +3.13% |
| 4 | 128 | 73.00 | 79.90 | +9.45% |
| 5 | 313 | 74.35 | 86.55 | +16.41% |
| 6 | 1,089 | 74.15 | 80.75 | +8.90% |

六个 decode 吞吐比的中位数为 **+11.31%**。这不是“到 32K 的每个上下文长度都实测到 decode 增益”。自动 CPU 共享和常驻路由下，生成长度和 MTP 工作量各不相同。整请求延迟互有胜负；一个短轮次的 HTTP 中位数出现回退。不要把该 decode 增益解读为整请求的必然加速，也不要把它当作输出完全一致的对比。

另一组固定设置的对照在 2K、8K、32K 提示长度上，三个完整的 16 token 输出序列全部一致。这不证明每个生产请求的输出都相同。

精确计时、构建来源、被否决的试验和复现命令见[评估报告](benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)。在别的安装上启用该选项之前，先看[服务配置指南](docs/zh_CN/NVIDIA_V100.md#dual-pcie-pipeline-service-profile)。

## V100 基准

下表记录的是单张 Tesla V100-PCIE-16GB 走 PCIe Gen3 x16 的单卡运行。测试机上还有第二张走 PCIe Gen3 x4 的 V100，本次未使用。双卡系统或两条完整宽度链路的机器结果可能不同，可能更好。这是一次可复现的测量，不是对其他系统速度的承诺。

| 目标提示长度 | 提示 token 中位数 | Prefill (tokens/s) | Decode (tokens/s) | GPU 最高温度 |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**测试设置：** Qwen3.8-Flash-Next Q2_0；仅一张 V100（GPU1，PCIe Gen3 x16）；Ryzen 5 3600；48 GB DDR4-3200；CUDA 12.8；262,144 token 上下文；int8 KV 缓存；自动 prefill；MTP `--spec 8`、draft floor 0.70；配对专家变体；700 MiB 视觉预留。每个数值是三个全新、无缓存请求（`reused=0`）恰好输出 256 token 的中位数。计时取自引擎的 `/metrics` 端点。
- **较老的显卡**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[较老的 GPU](docs/zh_CN/OLDER_GPUS.md)。
- **Intel Arc**，在 Linux 上从源码构建：[Intel Arc](docs/zh_CN/INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**，在 Linux 上从源码构建：[Strix Halo](docs/zh_CN/STRIX_HALO.md)。
- **不支持 AVX2 的老处理器**：能用，但很慢。[较老的 CPU](docs/zh_CN/INSTALL.md#older-cpus-experimental)。

每次请求前，被动散热的显卡空闲至少 120 秒，并在没有软件热降频的情况下 15 秒内降到 55 °C 以下。起始温度为 51–55 °C。所有请求都计入。128K 和 256K 提示在提示处理期间达到 84 °C 并出现软件热降频，尽管起始温度已冷却；长时间运行可能降频。† 表示提示期间出现降频。

测量于 2026-10-02。引擎：本分支版本 0.1.31，构建提交 `78417ea`（SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`）。该基准记录于后来的上游 v0.1.36 整合和 V100 注意力 PR 之前，不是对当前 `main` 构建的测量。历史表格按其原始出处保留，不要把它当作与更早表格或当前构建的受控对比。

原始数据：[`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json)、[`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json)、[`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json)、[`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json) 和 [GPU 遥测](bench/results/2026-10-02-v100-single-cooled/gpu.csv)。完整方法与历史见 [docs/DETAILS.md#tesla-v100-fork-benchmark](docs/zh_CN/DETAILS.md#tesla-v100-fork-benchmark)，基准说明见 [benchmarks/README.md](benchmarks/README.md)，V100 性能报告见 [benchmarks/](benchmarks/)。

近期的 V100 pull request 包含实测的内核级改动，并在端到端增益未确立时明确说明。例如，从上游 PR #600 移植的提示注意力在其测量形状上报告 int8 注意力内核时间减少 27.9%，而模型 prefill 变化在所报各次运行中为 +1.3% 到 +3.0%；它没有确立可靠的 decode 增益。对比结果前先读[完整报告](benchmarks/v100-q2_0-pr600-2026-10-03.md)。

## V100 性能工作

V100 的工作覆盖提示路径、decode 路径、内存使用、存储读取和多 GPU 执行。以下改动都已合入本分支历史；每个链接都包含自己的测试方法、硬件细节和限制。内核计时的改善不一定在全模型吞吐上产生相同百分比的增益。

### 提示处理

- **Volta 上的 BF16 转 FP16 tensor-core GEMM：** Volta 没有原生 BF16 GEMM 支持。本分支把 BF16 权重精确转换为 FP16，让提示 GEMM 使用 V100 tensor core，而不是标量回退路径。相关的 FP16 提示激活图使用正确的位表示。
- **更快的 PLE 表读取：** 把 POSIX PLE 读取切换为排队的 `io_uring` O_DIRECT 操作，带同步回退。一次实测的全新 8K PLE gather 从约 6.9 秒降到 7 毫秒。
- **Q2_0 MMQ prefill：** 为 Strata 的内部专家类型启用 llama.cpp 已有的 Q2_0 矩阵-矩阵量化路径。一次实测的 14,486 token 提示 prefill 吞吐提升 6.5%（926.1 → 986.2–986.9 tokens/s）。
- **只对活跃 QSA 块打分：** 避免对超出当前提示活跃范围的块打分。一次实测的 15,423 token 提示上，QSA 选择时间下降 91.1%；在所报测试中端到端 prefill 吞吐从 961.0 升到 1,111.3–1,112.9 tokens/s。
- **Volta 的 tensor-core 提示注意力：** 在 V100 上使用 `mma.m8n8k4` 而不是 FP32 回退。较早的一个 PR 测得注意力内核约 1.6–1.9× 提速，其测试中提示吞吐更高。后来的上游 PR #600 移植在一个测量形状上把 int8 注意力内核时间减少 27.9%；所报的端到端 prefill 增量更小（其 4K–32K 测试中 +1.3% 到 +3.0%）。

### Decode 与多 GPU 执行

- **批量验证窗口：** 把多个投机解码位置合在一起跑，减少每 token 的启动开销。配对的历史测试报告输出生成更快，但结果随提示和运行条件变化。
- **分块 QSA 块打分：** 在查询块之间复用已选的 key 数据，而不是反复读取。实测 256 查询 / 65K 块的内核从 3.5 ms 降到 1.3 ms；每次打分的算术保持位级一致。
- **融合 Volta GR 与批量 KV 追加：** 跨验证位置合并工作，用更少的启动追加 KV 数据。在测试的双 V100 配置上，4K–32K 提示的 decode 提升 5.6–17.3%，prefill 基本不变。
- **非对称双 V100 调优：** 针对双卡配置改进 host-row 处理、device-plan 调度和内存所有权。一个较早的安装配置在 4K–32K 上显示 prefill 提升 13.3–18.7%，但 decode 结果互有胜负，且那次运行中两张 GPU 都出现热降频。这些数值没有分离出每个代码改动的贡献。
- **削减权重与调度开销：** 使用显式切分时每张卡只加载自己拥有的层，避免不必要的 CPU 激活量化和空调度屏障，并可选启用配对行专家内核。GPU0 实测原生 dense 分配从 1,376.20 MiB 降到 541.27 MiB。配对行模式是可选启用的；所报的全模型结果没有确立 decode 改进。

### 注意力内核与内存流量

- **Decode 注意力（上游 PR #540 移植）：** 减少 score 洗牌和共享内存流量，同时保持累加顺序。实测 M=1 时 GPU0 内核时间从 52.90 微秒降到 31.13 微秒，M=8 时从 180.42 降到 111.85 微秒。所报的全模型 decode 变化很小；prefill 基本不变。
- **提示注意力（上游 PR #600 移植）：** 使用带四个独立 MMA 计算的 Volta tensor-core 实现。该 PR 报告了正确性检查和内核级 / 模型级测量；27.9% 的内核削减不应读作全模型速度提升 27.9%。
- **KV gather 与 GDN 递归（上游 PR #627 移植）：** 使用更宽的对齐 int8 KV 加载并带安全的原始宽度回退，并为 V100 提示递归使用多个累加器。实测内核时间：32K KV gather 约改善 10%，4,096 token 递归约改善 7%。配对模型 prefill 差异小于 0.2%；没有确立可靠的端到端 decode 增益。

### 构建支持与证据

本分支还修复了 `sm_70` 的 CUDA 配置与链接，在上游合并过程中维持 V100 设备准入，并在上游改动的架构要求排除 Volta 时进行适配。这些兼容性修复让优化路径保持可构建；它们本身不是性能声明。

完整证据和实现范围见 [PR #1](https://github.com/jmnargi/Strata-V100/pull/1)、[#2](https://github.com/jmnargi/Strata-V100/pull/2)、[#5](https://github.com/jmnargi/Strata-V100/pull/5)、[#6](https://github.com/jmnargi/Strata-V100/pull/6)、[#11](https://github.com/jmnargi/Strata-V100/pull/11)、[#12](https://github.com/jmnargi/Strata-V100/pull/12)、[#15](https://github.com/jmnargi/Strata-V100/pull/15)、[#16](https://github.com/jmnargi/Strata-V100/pull/16) 和 [#17](https://github.com/jmnargi/Strata-V100/pull/17)。这些结果使用不同的日期、构建、提示、GPU 位置和协议。它们是历史测量，不是单一的受控前后对比，也不保证在每个 V100 系统上都有增益。

## 开始使用

本分支跟随上游的安装体验。完整且最新的平台要求、模型选择和选项见[安装指南](docs/zh_CN/INSTALL.md)、[模型指南](docs/zh_CN/MODELS.md)和[上游 AI 安装指南](docs/zh_CN/AI_SETUP.md)。

- **Windows：** 下载或克隆本仓库，运行 `START-HERE.bat`。
- **Linux：** 克隆本仓库，运行 `./setup.sh`。
- 按提示选择模型和上下文大小。Setup 会下载模型数据并启动本地服务。
- 打开 setup 打印的地址（通常是 `http://127.0.0.1:8080`）。

Tesla V100 请使用带 `sm_70` 支持的 CUDA 12 引擎。Setup 会为 Volta 选择较老 GPU 引擎；Linux 上会本地构建它。上游的 Windows CUDA 12 压缩包包含 Volta 代码，但没有在这台 Linux 机器上测试过。工具包和驱动要求见[较老 GPU 指南](docs/zh_CN/OLDER_GPUS.md)。Docker 请按照 [Docker 说明](docs/zh_CN/INSTALL.md)操作，并为目标 GPU 架构构建。

## 使用 Strata

- **Web 应用：** 使用 setup 打印的本地 URL 进行聊天和实时 Monitor。
- **OpenAI 兼容 API：** 把应用的 base URL 设为 `http://127.0.0.1:8080/v1`。
- **Anthropic 兼容 API：** 使用 `http://127.0.0.1:8080/v1/messages`。
- **OpenAI Responses API：** 使用 `http://127.0.0.1:8080/v1/responses`。这个无状态端点支持 Codex 风格的工具调用；它不存储 `previous_response_id`。
- **图片、MCP、多 GPU 与配置：** 见[详细说明](docs/zh_CN/DETAILS.md)、[MCP 服务器](docs/zh_CN/MCP_SERVER.md)和[多 GPU 指南](docs/zh_CN/MULTI_GPU.md)。

服务器默认只监听 localhost。如果你把它暴露给其他机器，请配置 API key 并使用可信网络。不要在 issue、基准结果或 pull request 中发布密钥或私有配置。

### 会话缓存磁盘（可选启用）

会话状态分三层。L1 是活跃会话及其常驻提示检查点。L2 是可选的主机内存缓存（`--conversation-cache-mib`）。L3 是可选的磁盘存储，在重启之间保留已保存的会话和检查点组。引擎捕获状态时，流式 KV 层使用其权威的主机内存池。

要使用磁盘存储，在引擎参数中加入 `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25`。路径和正的 GiB 预算必须一起提供。存储默认关闭，需要 `--serve`，可与 `--conversation-cache-mib 0` 一起用，并支持多 GPU 的 `--layer-split`。可选：`--conversation-cache-disk-slots N` 限制记录数，`--conversation-cache-disk-min-free-mib N` 在文件系统上保留空闲空间。

引擎会在保留策略移除某个活跃检查点、或某个请求覆盖其分支之前保住它。每个保存的检查点都包含 GDN 状态、PLE 历史、QSA 索引器状态和所需的 KV 后备。仅有 token KV 数据不足以恢复本模型。

相关检查点共享一个归档组内的一份 KV 镜像。引擎移动并扩展该镜像，而不是为每个检查点复制完整前缀。当后备仍然有效时，只复制新增或变化的 KV 页。组在到达磁盘元数据上限之前停止增长。后续的组有自己的后备。引擎不会把已保存的检查点与来自前缀另一次计算的 KV 字节拼在一起。

归档组与已保存会话使用相同的 L2 字节和槽位上限。启用磁盘层时，L2 容量受害者会移入 L3。成功的 L2 准入不会同时把镜像写到磁盘。如果镜像放不下 L2，引擎会尝试存入 L3。物理内存和磁盘空闲空间限制仍然生效。最小 token 设置同样适用于检查点归档。

保住检查点是优化，不是硬性要求。如果主机内存下限或磁盘预算拒绝该镜像，引擎会丢弃该检查点、记录日志并继续服务。它不会停止服务器。会话保持正确；只是失去该检查点的复用。

查找会在活跃会话、L2、L3 和空闲批处理槽位之间选择最长的兼容前缀。token、图片身份和控制向量模式必须匹配。对较早消息的编辑，复用在第一个变化 token 之前的某个已保存检查点处结束。引擎必须重新处理后缀；没有那个位置的递归状态，它不能从任意匹配的 token 处恢复。增大 `--prompt-cache` 不能替代检查点保留。

L2 归档在恢复后仍然可用。当字节预算和物理内存下限允许时，有效的 L3 镜像会提升进 L2。只有其中有界的一部分检查点会常驻到活跃会话。恢复一个分支不会从已保存归档中移除其他检查点。

磁盘存储支持只追加的会话链。继续的链可以追加新的 token ID、检查点、运行状态和 KV 页。L2 逐出会写出完整的种子镜像；不需要 GPU 捕获。回退（rewind）会开启一个独立分支，而不是替换源链。文件格式对检查点归档没有变化。

磁盘面板带有校验和与版本字段。种子使用临时文件加重命名。不完整的追加会回收到最后一个完整面板。无效或不兼容的记录会被移除。磁盘存储按最近最少使用逐出链，以执行其字节预算、记录上限和空闲空间下限。因此 L2 和 L3 都不提供无限历史。如果两层都已逐出所需检查点，请求必须重新处理缺失的前缀。

启用根保留时，引擎会把共享的系统提示根一次性捕获进 L2。L3 在 L2 逐出或 L2 准入失败时接收根。禁用 L2 时，启用的 L3 可以直接接收根。`--conversation-cache-keep-root` 和 `--conversation-cache-disk-keep-root` 分别控制各层的根保留，其 `--no-` 变体禁用之。根的逐出优先级低于普通镜像；显式的请求检查点钉选仍受保护。

磁盘层不增加并发执行。并发请求的显存开销见[批处理槽位](docs/zh_CN/BATCHING.md)。捕获必须在请求覆盖所需会话字节之前完成。恢复是同步的。L2 逐出写出使用主机快照，可能增加磁盘 I/O 延迟。按你需要的会话和检查点历史来设定两层缓存预算。检查点保留的验证见 [V100 指南](docs/zh_CN/NVIDIA_V100.md)，较早的测量见[磁盘基准报告](benchmarks/v100-l3-conversation-cache-2026-10-03.md)。

#### V100 实测结果

在本分支的双 V100 层切分配置上，从 25 GiB NVMe 缓存恢复一个 33,725 token 的会话，读取耗时 1,614.6 ms，恢复耗时 315.8 ms。完整的恢复提示阶段（含 22 个新 token）耗时 3,015.1 ms。冷读同样 33,725 token 前缀耗时 20,332.7 ms。对恢复的请求这是 **6.74 倍加速**和 **85.2% 的提示延迟削减**。943.6 MiB 的记录在两张 GPU 上都恢复了位级一致的主模型状态和相同输出。重启恢复通过位级一致性验证，损坏记录回退产生匹配的输出。见[基准报告](benchmarks/v100-l3-conversation-cache-2026-10-03.md)。

在本分支的双 V100 层切分机器上（Qwen3.8-Flash-Next Q2_0，int8 KV）：两个会话通过一个临时缓存交替，8 次请求正确服务，`corruptions=0 evictions=0`；每个会话一个链文件，在三次各 +12 新 token 的停放（park）中从 238 MiB 增长到 594 MiB（全快照存储在 100k+ token 时每次停放会写一个全新的约 2.4 GiB 文件）。全新短会话的第一次停放为链播种（约 118 MiB：每层一个 K/V 页加模型固定的运行状态）；之后每次停放都是增量（+12 token 约 118 MiB），读回/恢复路径不变（读取约 0.4 秒，恢复约 0.04–0.08 秒）。

### 懒加载视觉（可选启用）

图片编码器（`strata-vision`）只要运行就持有约 1.74 GiB 显存（第一张 GPU 1.43 GiB，第二张 0.31 GiB）。启用懒加载后，它在图片请求之间不运行：那部分显存留给专家缓存，纯文本工作保持完整的常驻专家数。

要启用它，在模型配置中加入：

- `"vram_elastic": true`（引擎标志 `--vram-elastic`）：专家缓存分段，引擎可以在请求之间归还并重新取回显存。
- `"vision": {"lazy": true, "idle_s": 900}`：编码器在第一个图片请求时启动，在 `idle_s` 秒（900 = 15 分钟）没有图片请求后停止。编码器需要引擎的一次一个请求模式，所以不要与 `"parallel"` 组合（或用其他方式给编码器显存，例如 `POST /v1/vram`）。

行为方式：

- 文本请求永远不会启动编码器。图片请求先让引擎归还各 GPU 的显存——每张 GPU 一个预留，层切分时为 `[1700, 600]`，单卡为 `[1800]`，可用 `"vram_mib"` 覆盖——然后在引擎空闲时启动编码器并编码图片。
- `idle_s` 秒没有图片请求后编码器停止，专家缓存长回完整大小。已编码的图片缓存在磁盘上，再发同一张图不会重新编码。
- 在本分支的双 V100 层切分上实测：编码器空闲时缓存常驻 **14,005** 个专家（编码器常驻时为 12,688），冷图片请求 **5.9 秒**完成，编码器卸载后两个缓存都回到 14,005。见[详细说明](docs/zh_CN/DETAILS.md)。

## 贡献

欢迎贡献。帮助不需要你有 V100：文档、测试、安装、服务器行为、API 兼容性和其他受支持设备的改进都有价值。V100 专属代码和性能结果最好在真实 Volta 硬件上验证。

1. 先查看 [open issues](https://github.com/jmnargi/Strata-V100/issues) 和 [pull requests](https://github.com/jmnargi/Strata-V100/pulls)，避免重复工作。
2. 对广泛的上游特性，先判断改动是否应属于 [Niko 的上游仓库](https://github.com/Niko1221/Strata)。本分支定期整合上游改动；聚焦的 V100 修复和测量可以在这里提出。
3. 做一个聚焦的改动，描述测试所用的硬件和确切步骤，并附上相关测试。
4. 对性能声明，报告基线和候选、完整运行时设置、工作负载、重复次数和限制。尽可能附上原始数据或可复现命令。把内核计时与全模型吞吐分开，不要声明测量没有显示的增益。
5. 向本仓库的 `main` 分支开 pull request。说明改动是 V100 专属、上游整合，还是通用 Strata 改进。不要把凭据、个人配置和无关的本地文件带进改动。

当前测量方法见 [`benchmarks/README.md`](benchmarks/README.md) 和 [`docs/DETAILS.md`](docs/zh_CN/DETAILS.md)。贡献仍受项目许可证以及所含组件和模型文件的许可证约束。

## 项目历史

本分支在保留 Strata 上游发展的同时开发 V100 支持和 V100 专属工作。近期的整合与性能 pull request 体现了这一过程：

- [#14 — 合并上游 v0.1.36 并保留 V100 支持](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — 从上游 PR #627 移植实测的 V100 KV gather 与 GDN 递归改动](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — 减少 V100 decode 注意力的洗牌与共享内存流量](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — 移植上游 PR #600 的 Volta 提示注意力内核](https://github.com/jmnargi/Strata-V100/pull/17)

这些报告描述的是具体测过的改动；它们不是跨模型或跨 GPU 的全面提速声明。更多细节见[完整 pull request 历史](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged)和[提交历史](https://github.com/jmnargi/Strata-V100/commits/main)。

## 致谢与许可证

本仓库是 [Niko1221 的 Strata](https://github.com/Niko1221/Strata) 的 fork。**我们深深感谢 Niko 和所有上游贡献者。** 他们的原始工作正是本分支、其 V100 支持和本地模型运行时得以存在的原因。我们在定期引入上游改动的同时，力求为上游工作署名并遵循其走向。

Strata 使用了 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 和其他开源组件。项目以 [MIT 许可证](LICENSE)授权；组件和模型文件可能有各自条款。见[致谢与许可证说明](docs/zh_CN/HOW_IT_WORKS.md#credits)。
