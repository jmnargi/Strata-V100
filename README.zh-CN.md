# Strata-V100

**[Strata](https://github.com/Niko1221/Strata) 的社区维护分支，专注于 V100。** 本分支让 NVIDIA Volta（`sm_70`）支持保持可用，在 Tesla V100 硬件上开发并实测性能改动，并定期合并上游改进，以获得 V100 之外的新特性和修复。

Strata 是 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的开源本地运行时。它把 GPU、系统内存和存储组合起来，让这个大模型能在个人电脑上运行。本分支保留了 Strata 项目的全部内容：其他受支持的 NVIDIA 和 AMD 硬件、Web 界面、OpenAI 兼容 API、工具、图片输入以及上游特性同样重要。

> **向 Niko（[@Niko1221](https://github.com/Niko1221)）和原版 Strata 的贡献者致以诚挚谢意。** 本项目建立在他们的工作之上：Strata 的模型运行时、架构、特性，以及让这个模型得以本地运行的所有工作，都来自上游项目。没有那个基础，就不会有 V100 分支。感谢你们让它成为可能，也感谢你们持续开发 Strata。

[上游 Strata](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussions](https://github.com/jmnargi/Strata-V100/discussions)

**语言：** [English](README.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Português (Brasil)](README.pt-BR.md) · [简体中文](README.zh-CN.md)

## 目录

- [可在此运行的模型与量化](#可在此运行的模型与量化)
- [本分支的关注点](#本分支的关注点)
- [文档](#文档)
- [发布说明](#发布说明)
- [V100 基准](#v100-基准)
- [V100 性能工作](#v100-性能工作)
- [其他硬件](#其他硬件)
- [安装与启动](#安装与启动)
- [使用 Strata](#使用-strata)
- [遇到问题时](#遇到问题时)
- [贡献](#贡献)
- [项目历史](#项目历史)
- [致谢与许可证](#致谢与许可证)

## 可在此运行的模型与量化

Strata 运行 **Qwen3.8-Flash-Next**，来源为七个量化发布系列，另有一个较小的 **Qwen3.6-35B-A3B**。下表的每一行都由本分支的引擎和 setup 支持；最后一列说明哪些已在这台双 V100 机器上端到端运行过。规模、质量与各规模的速度见 [docs/zh_CN/MODELS.md](docs/zh_CN/MODELS.md)。

| 模型 / 量化 | 路由专家 | 权重来源 | 下载 | 在此双 V100 上运行 |
| --- | --- | --- | ---: | --- |
| **NVFP4 — `local-inference-lab`（QAD）** | NVFP4，group 16；MXFP8 注意力与共享专家；NVFP4 n-gram 表 | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)，在你的电脑上转换（[NVFP4.md](docs/NVFP4.md#the-qad-distillation-local-inference-labqwen38-flash-next-nvfp4)） | 106 GB | 是 — 62.3 tokens/s decode |
| **NVFP4 — NVIDIA，为 FP4 训练（`FP4-Local`，QAD）** | NVFP4，group 16（缩放值采用 NVIDIA 的 CSF 形式） | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)，在你的电脑上转换（[NVFP4.md](docs/zh_CN/NVFP4.md#nvidias-fp4-local-checkpoint)） | 102 GB（准备期间为 +110 GB） | 支持；未在此运行 |
| **NVFP4 — NVIDIA 的版本** | NVFP4，group 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata)（现成模型包） | 128 GB | 支持；未在此运行 |
| **NVFP4 — Strata 的版本** | NVFP4，group 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)（现成模型包，GPTQ） | 128 GB | 支持；未在此运行 |
| **Unsloth Dynamic v3.1（1-4 bit）** | UD-IQ1_M、UD-Q2_0、UD-IQ2_XS、UD-Q2_K_XL、UD-IQ3_XXS、UD-Q3_K_XL、UD-IQ4_XS-v31 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)，2026-11-04 的修订版本 | 68-94 GB | 是 — UD-IQ4_XS-v31，81 tokens/s |
| **Unsloth Dynamic（较早的修订版本）** | UD-IQ4_XS、UD-Q4_K_XL、UD-Q5_K_XL、UD-Q6_K_XL、Q8_0 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 94-188 GB | 支持；未在此运行 |
| **ISTA-DASLab 的 i-quants 与 Q2_0（2-3.5 bit）** | Q2_0、IQ2_XS、IQ3_XXS、IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66-84 GB | 是 — IQ3_S，较早的日常配置 |
| **Gyro（agentionai），trellis** | Gyro-S TQ1_0、Gyro-M TQ2_0（pair trellis） | [`agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)（[GYRO.md](docs/zh_CN/GYRO.md)） | 58.5 / 92 GB | 是 — Gyro-M，约 70 tokens/s |
| **Coder（ISTA-DASLab），专家数量减半** | IQ1_M、IQ2_S-IQ4_XS，与 IQ3_S 类似 | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 58 GB | 支持；未在此运行 |
| **OrcaRouter Uncensored** | IQ3_XXS | `OrcaRouter` 的 Flash-Next Uncensored（[ORCA.md](docs/zh_CN/ORCA.md)；手动打包步骤，不是安装程序选项） | 84 GB | 支持；未在此运行 |
| **Qwen3.6-35B-A3B**（较小的模型） | Unsloth 的 UD-Q4_K_XL、UD-Q6_K_XL 和 Q8_0，带 MTP 草稿层 | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)（[MODELS.md](docs/zh_CN/MODELS.md#qwen36-35b-a3b)） | 23-38 GB | 支持；未在此运行 |

NVFP4 各行需要 NVIDIA 显卡（RTX 50 显卡以 FP4 宽度相乘；较老的显卡，包括 V100，以 8-bit 精度相乘）。两个 QAD 行在你的电脑上由作者的 safetensors 准备，这里不再分发他们的量化权重。Gyro 和 trellis 各行需要 0.1.45.1 或更新的引擎，Unsloth 的 Q5_0 down 各行需要 0.1.45.7。

## 本分支的关注点

- **Tesla V100 / Volta 支持（`sm_70`）**：保住 V100 的正常构建和运行时支持。V100 构建使用 CUDA 12.x 工具包；CUDA 13 不编译 `sm_70` 代码。
- **测试机：** 两张 Tesla V100-PCIE-16GB；GPU 0 走 PCIe Gen3 x4，GPU 1 走 Gen3 x16。下文的双卡流水线测量使用两张卡。历史单卡基准只用了 x16 那张。两条完整 Gen3 x16 链路的系统尚未在本机测量。
- **上游整合**：这不是 Strata 的独立替代品。上游改动会定期合并进来，并检查、修正 V100 兼容性。上游新特性可能有高于 `sm_70` 的硬件要求；它们不会自动加速 V100。
- **更广泛的 Strata 社区**：欢迎针对其他受支持平台的改进、错误报告、测试、文档和贡献。

对 V100 的支持不意味着每个上游特性或每种其他 GPU 配置都在 V100 上测过。在依赖某个硬件相关特性之前，请先查看相关代码、发布说明和测量报告。

## 文档

[文档索引](docs/README.md)列出了每个页面及其用途。以下页面是参考：

- [引擎标志](docs/ENGINE_FLAGS.md)：每个命令行选项、它的取值和默认值。
- [环境变量](docs/ENVIRONMENT.md)：引擎、服务器和终端读取的每个变量。
- [服务器](docs/SERVER.md)：服务器选项、运行配置键和全部 API 端点。
- [终端](docs/TERMINAL.md)：每个 `strata` 命令和子命令。
- [调优](docs/TUNING.md)：按目标列出的设置、各项优化和实测结果。
- [工作原理](docs/zh_CN/HOW_IT_WORKS.md)和[术语表](docs/GLOSSARY.md)：引擎的各个部分，以及本文档使用的术语。
- [详细说明](docs/zh_CN/DETAILS.md)：测量记录、请求格式和设置历史。

这些页面遵循[文档风格](docs/STYLE.md)，即 ASD-STE100（简化技术英语）。

## 发布说明

### 引擎 0.1.45.8 —— 2026 年 11 月 16 日

[发布版本 v0.1.45.8](https://github.com/jmnargi/Strata-V100/releases/tag/v0.1.45.8)
为当前 V100 引擎提供独立版本号。它包含 FP16 专家 prefill 行偏移修复、
NVFP4 解码改进、需手动启用的多卡 HC_Q8 内存节省、需手动启用的提示评分与
流式 KV 改进，以及每个槽位的提示处理进度。本次版本号更新本身不改变推理代码
或默认调优设置。测量结果和限制见 [NVIDIA_V100.md](docs/NVIDIA_V100.md)。

### Unsloth Dynamic v3.1 量化集（1-4 bit）—— 2026 年 11 月 8 日

解决 [#95](https://github.com/jmnargi/Strata-V100/issues/95)：setup 的 `--family unsloth` 菜单以前停在 UD-IQ4_XS（约 4 bit）。现在它也提供 Unsloth 的 Dynamic v3.1 文件，范围约 1 到 4 bit，采用 Unsloth 于 2026-11-04 发布的修订版本（提交 `d9620119`）：

| 规模 | 下载 | 路由专家 | 全部专家都在内存时的 RAM | 备注 |
| --- | ---: | ---: | ---: | --- |
| UD-IQ1_M | 68.0 GB | 32.4 GiB | ~48 GB | 最小；在需要仔细或长时间处理的任务上最弱 |
| UD-Q2_0 | 70.5 GB | 34.8 GiB | ~51 GB | Q2_0 专家，与最快的 2-bit 打包相同 |
| UD-IQ2_XS | 75.8 GB | 39.4 GiB | ~55 GB | 1 层带 Q5_0 down |
| UD-Q2_K_XL | 79.6 GB | 42.8 GiB | ~59 GB | 1 层带 Q5_0 down |
| UD-IQ3_XXS | 83.6 GB | 46.8 GiB | ~63 GB | 没有 Q5_0 |
| UD-Q3_K_XL | 89.5 GB | 52.3 GiB | ~68 GB | 20 层带 Q5_0 down |
| UD-IQ4_XS-v31 | 93.7 GB | 56.2 GiB | ~72 GB | 33 层带 Q5_0 down；有独立的目录名称，因为它在 Hub 上的文件名是 UD-IQ4_XS |

此前已提供的五个文件（UD-IQ4_XS、UD-Q4_K_XL、UD-Q5_K_XL、UD-Q6_K_XL、Q8_0）仍固定在上一个提交。UD-Q4_K_XL 在新提交下的第二种布局（四个分片，n-gram 表在最后）可从磁盘上已有的文件接受。UD-IQ1_S 仍被拒绝：它在一层里混用了两种专家格式。

为此需要的引擎工作：Q5_0 专家点积。它按权重计的 -16 偏移曾作用在原始激活之和上（llama.cpp 的 CUDA 形式），因此 down 投影为 Q5_0 的层，每个专家的精度比其他任何 down 格式低 2-3%。现在它改为乘上量化后的激活之和，与 Q5_1、Q4_0 路径以及 ggml-cpu 一致（CUDA 与 SYCL）。在本机用合成专家一致性测试实测（`native_expert_parity --synthetic`，IQ2_XXS / IQ2_S / IQ2_XS / IQ3_XXS / IQ4_XS 的 gate-up 配 Q5_0 down，相对浮点参考的 L1 误差）：**修改前 2.40e-2 .. 3.18e-2，修改后 1.25e-2 .. 1.30e-2**，与相同种子下 IQ4_NL 和 Q2_0 down（1.29e-2 .. 1.31e-2）相同。分组专家启动与窗口前的启动保持逐位相同（`native_grouped_parity`：每种 gate/up 格式都加入 Q5_0，以及这些文件使用的 Flash-Next 形状，0 次失败）。引擎版本：0.1.45.7；对这七个量化集，setup 会在下载任何内容之前拒绝更老的引擎。

五个固定文件、它们的字节数和其他所有格式均未改变：只有 Q5_0 *专家* 权重读取该内核（dense Q5_0 路径是 `native_mmvq`，未改动），它们相对浮点参考的整专家误差从 2.4-3.2e-2 变为 1.25-1.30e-2。

### 选定的上游 v0.2.11 性能移植 —— 2026 年 11 月 5 日

本次更新有选择地移植上游优化；它不是完整的 v0.2.11 合并。它新增更快的 trellis 专家内核、更小的提示词暂存缓冲区、更少的 PLE 磁盘读取器分配，以及一项 Linux CPU/CUDA 零自旋修复。引擎版本仍为 0.1.45.6。

**在本分支的双 V100 机器上实测：** 两张 V100-PCIE-16GB（175 W，GPU0 为 Gen3 x4，GPU1 为 x16）、Ryzen 5 3600，以及 Gyro-M abliterated Huihui TQ2_0。该配置使用切分 16、两个流水线窗口、MTP 4、int8 KV，以及 YaRN 2 的 524,288 token 上下文。

| 提示词 token 数 | 对照组提示词时间 | 分组暂存 + 关闭 WMMA |
| ---: | ---: | ---: |
| 2,048 | 2,360.8 ms | 2,214.3 ms |
| 8,192 | 6,932.7 ms | 6,541.0 ms |
| 32,768 | 18,008.6 ms | 17,636.7 ms |

这些是冷启动、交错 A-B-A-B 运行的平均值，每个长度和每个实验组四个请求。两组都已包含 trellis、HC 和 PLE 移植。这不是每个上下文长度都实测到的增益，也不是整请求速度的保证。

实测的可选组合是 `STRATA_TQ_SUB8=1 STRATA_EXPERT_PAIR=0 STRATA_PF_HC_REGION=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_WMMA=0`。分组暂存可能改变浮点舍入。PLE 分配改动不需要标志。零自旋修复在 `STRATA_POOL_SPIN_US=0` 时生效；它不能证明在日常设置 100 下也有节省。

各项测量、选项保护、数值检查、构建限制和未保留的测试候选见 [V100 移植报告](docs/NVIDIA_V100.md#selected-upstream-0211-ports-gyro-m-2026-11-05)。服务默认值没有改变。

### 选定的上游同步 —— 2026 年 10 月 19 日

本分支从上游提交 `3d1fb480` 引入了共享的 CUDA 引擎和服务器改动。它保留了本分支的 Volta 内核、流水线 K/V 增长和会话缓存改动。它排除了无关的桌面安装包以及独立的 Metal、AMD、Intel 后端更新。见[同步范围](improvements/RELEASE-2026-10-19-V100.md)。`sm_70` 请使用 CUDA 12.8。引入某个上游选项并不代表它在 V100 上有性能收益。

### 上游 v0.1.39 整合

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

524,288 token 上下文下的并行槽位取决于切分方式和最后一级剩余的显存；实测的回退行为（请求 2 个槽位、实际一次服务一个请求、配置不动）见 [docs/zh_CN/BATCHING.md](docs/zh_CN/BATCHING.md)。

### 双 V100 流水线性能 —— 2026 年 10 月 15 日

实测服务配置启用 `--pipeline-windows 2`，GPU 顺序 `0,1`，层切分 `16`。使用两张 16 GiB 的 V100（175 W）、Ryzen 5 3600、原生 IQ3_S 混合格式模型、int8 KV，以及 `--spec 8` 的 MTP。配置的上下文容量仍为 524,288 token。实测的最大提示为 32,768 token。

**长回复安全性：** 把本配置与 `STRATA_KV_GROW=1` 组合使用时，请使用包含[流水线 KV 增长修复](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)的构建。更早的流水线循环在长回复期间可能访问未映射的 KV 单元。修复会在扩展映射之前排空投机工作，然后再恢复流水线。下面短输出的速度测量没有触及该边界。

#### 全新提示词的 prefill（至 32K）

每个数值是三个精确 token 数请求的中位数，无提示复用，输出上限 16 token。prefill 时间不含回复生成。吞吐量为提示 token 数除以 prefill 时间中位数。

| 提示 token 数 | 之前 prefill 时间 | 流水线 prefill 时间 | 之前 tokens/s | 流水线 tokens/s | 吞吐变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

全新提示词的 prefill 基本不变。这些微小差异不足以说明存在有意义的回退。

#### 多轮对话 decode

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

### TurboQuant KV 缓存 —— 2026 年 10 月 25 日

TurboQuant 是一种可选启用的 KV 缓存格式。它用 Hadamard 变换旋转每个 256 值 head，然后按 Lloyd-Max 码本量化旋转后的值。每个 head 保留一个 FP16 范数。宽度为每个值 3、4、6 和 8 bit。拟合只采用最小平方误差；此版本没有 QJL 残差。

使用 `--kv tq6`（或 `tq3`、`tq4`、`tq8`）。该缓存只常驻：它留在显存中，没有主机副本，`--kv-resident` 会被拒绝。MTP 草稿层保留其 INT8 缓存。Setup 不提供这些取值。没有任何一项是默认值，日常服务仍使用 int8。

在配置的 524,288 token 容量下，模型的 12 个注意力层对应的 KV 缓存为：

| `--kv` | 每单元字节数，单层 | 满容量 KV，12 层 | 相对 int8 的变化 |
| --- | ---: | ---: | ---: |
| int8 | 1,056 | 6.1875 GiB | — |
| tq8 | 1,032 | 6.046875 GiB | −0.14 GiB |
| tq6 | 776 | 4.546875 GiB | −1.64 GiB |
| tq4 | 520 | 3.046875 GiB | −3.14 GiB |
| tq3 | 392 | 2.296875 GiB | −3.89 GiB |

tq6 的主要节省在满容量时为 1.64 GiB。这些是计算出的容量，不是实测的 `nvidia-smi` 下降值：专家缓存会占用腾出的显存，因此 GPU 总内存仍接近 15,600 MiB。

V100 提示词读取器是第二个可选启用项：在 sm_70 上设置 `STRATA_PROMPT_ATTN_TQ=1`。它为提示词路径解码压缩 tile；不设置或设为 `0` 时保留 decode 注意力回退路径。它不改变 KV 字节数。

在两张 175 W 的 Tesla V100 16 GB 上实测，模型为 IQ3_S，使用 `--spec 4`、`--layer-split 16`、`--pipeline-windows 2`，冷启动，输出上限 2,048 token，每个长度每个实验组运行一次。tq6 组开启了提示词读取器。在 32,768 提示词 token 时，tq6 的 prefill 为 1,687 tokens/s，int8 为 2,117 tokens/s；decode 为 120.2 tokens/s，对照为 125.5 tokens/s。开启读取器后，tq6 的 51,746 token 请求实际耗时从 40.3–40.4 s 降到 31.7–31.9 s，仍慢于 int8 的 25.8 s。这些测量没有显示速度增益。针对性查找（needle test）在两组中都于 122,289 提示词 token 范围内找到 15 个中的 15 个单元；这不是普遍的质量结论。

见[完整报告](benchmarks/v100-turboquant-2026-10-25.md) 和[详细说明](docs/zh_CN/DETAILS.md#tesla-v100-fork-benchmark)。

### 双 V100 上的 Gyro-L pair trellis —— 2026 年 10 月 31 日

本分支读取 agentionai 的 `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf`（108,323,559,808 字节，Hugging Face 修订版本 `05892758a67efff0895d23c6df10a6fdb6be8cf4`）。它的路由专家以 pair trellis 类型 TQP5、TQP6 和 TQP7（ggml 类型 147 到 149）存储在 Hadamard 旋转基中。打包步骤（`tools/iq_pack.py --compat-bf16`）和内核见 [docs/zh_CN/GYRO.md](docs/zh_CN/GYRO.md)。支持 CPU 和 CUDA；Metal 和 SYCL 没有 pair-trellis 内核。这需要 0.1.45.6 或更新的引擎。

必须先修复两个引擎缺陷。专家内核本身已经与作者的 ggml 逐位一致：

- **decode 窗口丢失了旋转。** 对于折叠打包（folded pack），窗口旋转了专家输入，随后在专家读取之前用未旋转的副本覆盖了该旋转副本，因此每个 decode token 上 gate/up 投影都在错误的基中运行。单次输出看起来合理，随后开始循环；服务器从第一个 token 起就是错的。窗口现在为每层保留一个“镜像已写入”标记。
- **层切分的提示词辅助器**（可选启用）只旋转了自己的 SwiGLU 输出，并把一个组的行写入按该阶段 WMMA tile 布局的缓冲区。它现在旋转传给它的输入，并拥有自己的 H。

在本分支的双卡机器上实测：两张 Tesla V100-PCIE-16GB（GPU 0 走 PCIe Gen3 x4，GPU 1 走 x16），175 W，Ryzen 5 3600，层切分 14，`--spec 4 --spec-min-p 0.70`，int8 KV 并启用 KV 增长，`--pipeline-windows 2`，262,144 token 上下文，使用服务自身的环境，交错重复运行：

| 提示词 token 数 | Prefill (tokens/s) |
| ---: | ---: |
| 2,328 | 632 |
| 8,199 | 1,036-1,084 |
| 16,371 | 1,344-1,411 |
| 32,679 | 1,597-1,642 |

在 2K 到 32K 上下文下，回复的 decode 为 68 到 75 tokens/s（重复 520 token 回复的中位数）；切分点不会把它改变到超出运行间噪声的程度。一次读取受限于不在显存中的专家：对 2.3K token 的读取（层切分 16 的配置），x16 卡通过 PCIe 流式读取 15.01 GiB，而两个阶段依次运行，因此 x4 卡在每次读取中约空闲 700 ms。

该模型包通过 OpenAI 兼容 API 提供服务：对首都问题回答“Paris”，一个复用了 86 个提示词 token 的续问轮次，一个代码请求，一次 `get_weather` 工具调用以及来自工具结果的正确回答，还有一个根据段落回答的 442 token 段落问题。

**层切分 14 而不是 16。** x4 卡上十四层、x16 卡上 34 层时，2.3K token 的读取快 7.9%（632 对 586 tokens/s），8.2K 时快 6%，16.3K 时相同，32.7K 时慢 1.5%。15 和 13 介于两者之间；20 在 2.3K 时慢 23%，8.2K 时慢 13%。智能体轮次的时间主要花在一次读取上，因此本配置使用 14。

**在这台双卡机器上实测过但没有收益的设置：** `--spec 6` 和 `--spec 8`、`--spec-min-p 0.50` 和 `0.85`、固定的 `STRATA_PREFILL_CPU_SHARE`、高于 6,144 的 `STRATA_PREFILL_CPU_SHARE_MAX`、`--prefill-pipe`、`--max-context 131072`、`STRATA_HC_Q8=1`，以及更大的 `--vram-reserve-mib`。提示词辅助器本身在这里更慢（2.3K 时 484 对 583 tokens/s）：它只能接管第一阶段的专家，而在这台机器上该阶段是 x4 卡，一次读取移动 16.8 GiB，其中该卡流式读取 1.77 GiB。`--mtp` 和默认的 `STRATA_ROUTE_TAIL_SKIP=7` 值得保留：没有它们，decode 降到 35.5 和 48 tokens/s。

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

每次请求前，被动散热的显卡空闲至少 120 秒，并在没有软件热降频的情况下 15 秒内降到 55 °C 以下。起始温度为 51–55 °C。所有请求都计入。128K 和 256K 提示在提示处理期间达到 84 °C 并出现软件热降频，尽管起始温度已冷却；长时间运行可能降频。† 表示提示期间出现降频。

测量于 2026-10-02。引擎：本分支版本 0.1.31，构建提交 `78417ea`（SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`）。该基准记录于后来的上游 v0.1.36 整合和 V100 注意力 PR 之前，不是对当前 `main` 构建的测量。历史表格按其原始出处保留，不要把它当作与更早表格或当前构建的受控对比。

原始数据：[`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json)、[`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json)、[`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json)、[`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json) 和 [GPU 遥测](bench/results/2026-10-02-v100-single-cooled/gpu.csv)。完整方法与历史见 [docs/zh_CN/DETAILS.md](docs/zh_CN/DETAILS.md#tesla-v100-fork-benchmark)，基准说明见 [benchmarks/README.md](benchmarks/README.md)，V100 性能报告见 [benchmarks/](benchmarks/)。

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

## 其他硬件

- **较老的显卡**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[较老的 GPU](docs/zh_CN/OLDER_GPUS.md)。
- **Intel Arc**，在 Linux 上从源码构建：[Intel Arc](docs/zh_CN/INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**，在 Linux 上从源码构建：[Strix Halo](docs/zh_CN/STRIX_HALO.md)。
- **不支持 AVX2 的老处理器**：能用，但很慢。[较老的 CPU](docs/zh_CN/INSTALL.md#older-cpus-experimental)。

## 安装与启动

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

在本分支的双 V100 层切分配置上，从 25 GiB NVMe 缓存恢复一个 33,725 token 的会话，读取耗时 1,614.6 ms，恢复耗时 315.8 ms。完整的恢复提示阶段（含 22 个新 token）耗时 3,015.1 ms。冷读同样 33,725 token 前缀耗时 20,332.7 ms。对恢复的请求这是 **6.74x 加速**和 **85.2% 的提示延迟削减**。943.6 MiB 的记录在两张 GPU 上都恢复了位级一致的主模型状态和相同输出。重启恢复通过位级一致性验证，损坏记录回退产生匹配的输出。见[基准报告](benchmarks/v100-l3-conversation-cache-2026-10-03.md)。

在本分支的双 V100 层切分机器上（Qwen3.8-Flash-Next Q2_0，int8 KV）：两个会话通过一个临时缓存交替，8 次请求正确服务，`corruptions=0 evictions=0`。每个会话一个链文件，在三次各 +12 新 token 的停放（park）中从 238 MiB -> 594 MiB（全快照存储在 100k+ token 时每次停放会写一个全新的 ~2.4 GiB 文件）。全新短会话的第一次停放为链播种（~118 MiB：每层一个 K/V 页加模型固定的运行状态）；之后每次停放都是增量（+12 token 约 ~118 MiB），读回/恢复路径不变（读取约 0.4 s，恢复约 0.04-0.08 s）。

### 懒加载视觉（可选启用）

图片编码器（`strata-vision`）只要运行就持有约 1.74 GiB 显存（第一张 GPU 1.43 GiB，第二张 0.31 GiB）。启用懒加载后，它在图片请求之间不运行：那部分显存留给专家缓存，纯文本工作保持完整的常驻专家数。

要启用它，在模型配置中加入：

- `"vram_elastic": true`（引擎标志 `--vram-elastic`）：专家缓存分段，引擎可以在请求之间归还并重新取回显存。
- `"vision": {"lazy": true, "idle_s": 900}`：编码器在第一个图片请求时启动，在 `idle_s` 秒（900 = 15 分钟）没有图片请求后停止。编码器需要引擎的一次一个请求模式，所以不要与 `"parallel"` 组合（或用其他方式给编码器显存，例如 `POST /v1/vram`）。

行为方式：

- 文本请求永远不会启动编码器。图片请求先让引擎归还各 GPU 的显存——每张 GPU 一个预留，层切分时为 `[1700, 600]`，单卡为 `[1800]`，可用 `"vram_mib"` 覆盖——然后在引擎空闲时启动编码器并编码图片。
- `idle_s` 秒没有图片请求后编码器停止，专家缓存长回完整大小。已编码的图片缓存在磁盘上，再发同一张图不会重新编码。
- 在本分支的双 V100 层切分上实测：编码器空闲时缓存常驻 **14,005** 个专家（编码器常驻时为 12,688），冷图片请求 **5.9 s** 完成，编码器卸载后两个缓存都回到 14,005。见[详细说明](docs/zh_CN/DETAILS.md)。

## 遇到问题时

常见问题及其修复方法见 [TROUBLESHOOTING.md](docs/zh_CN/TROUBLESHOOTING.md)。该页涵盖桌面应用、`strata` 命令和安装程序。完整的错误消息表见 [DETAILS.md](docs/zh_CN/DETAILS.md#troubleshooting)。

如果服务器启动但请求失败，运行 `strata doctor`。它读取已安装的组件，并打印缺失或过期的内容；见 [TERMINAL.md](docs/TERMINAL.md#doctor)。

如果回复只是慢而没有坏掉，请阅读 [TUNING.md](docs/TUNING.md)。它给出每个目标下真正重要的设置，以及每项设置的实测代价。

## 贡献

欢迎贡献。帮助不需要你有 V100：文档、测试、安装、服务器行为、API 兼容性和其他受支持设备的改进都有价值。V100 专属代码和性能结果最好在真实 Volta 硬件上验证。

1. 先查看 [open issues](https://github.com/jmnargi/Strata-V100/issues) 和 [pull requests](https://github.com/jmnargi/Strata-V100/pulls)，避免重复工作。
2. 对广泛的上游特性，先判断改动是否应属于 [Niko 的上游仓库](https://github.com/Niko1221/Strata)。本分支定期整合上游改动；聚焦的 V100 修复和测量可以在这里提出。
3. 做一个聚焦的改动，描述测试所用的硬件和确切步骤，并附上相关测试。
4. 对性能声明，报告基线和候选、完整运行时设置、工作负载、重复次数和限制。尽可能附上原始数据或可复现命令。把内核计时与全模型吞吐分开，不要声明测量没有显示的增益。
5. 向本仓库的 `main` 分支开 pull request。说明改动是 V100 专属、上游整合，还是通用 Strata 改进。不要把凭据、个人配置和无关的本地文件带进改动。

当前测量方法见 [`benchmarks/README.md`](benchmarks/README.md) 和 [`docs/zh_CN/DETAILS.md`](docs/zh_CN/DETAILS.md)。贡献仍受项目许可证以及所含组件和模型文件的许可证约束。

## 项目历史

本分支在保留 Strata 上游发展的同时开发 V100 支持和 V100 专属工作。近期的整合与性能 pull request 体现了这一过程：

- [#14 — 合并上游 v0.1.36 并保留 V100 支持](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — 从上游 PR #627 移植实测的 V100 KV gather 与 GDN 递归改动](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — 减少 V100 decode 注意力的洗牌与共享内存流量](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — 移植上游 PR #600 的 Volta 提示注意力内核](https://github.com/jmnargi/Strata-V100/pull/17)
- [#88 — 读取 Gyro-L 的 pair-trellis 专家（TQP5/6/7），修复折叠 decode 窗口和层切分的提示词辅助器，并把 V100 配置改为 `layer_split 14`](https://github.com/jmnargi/Strata-V100/pull/88)

这些报告描述的是具体测过的改动；它们不是跨模型或跨 GPU 的全面提速声明。更多细节见[完整 pull request 历史](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged)和[提交历史](https://github.com/jmnargi/Strata-V100/commits/main)。

## 致谢与许可证

本仓库是 [Niko1221 的 Strata](https://github.com/Niko1221/Strata) 的 fork。**我们深深感谢 Niko 和所有上游贡献者。** 他们的原始工作正是本分支、其 V100 支持和本地模型运行时得以存在的原因。我们在定期引入上游改动的同时，力求为上游工作署名并遵循其走向。

Strata 使用了 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 和其他开源组件。项目以 [MIT 许可证](LICENSE)授权；组件和模型文件可能有各自条款。见[致谢与许可证说明](docs/zh_CN/HOW_IT_WORKS.md#credits)。
