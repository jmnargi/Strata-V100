# Strata 文档（简体中文）

这里是 [`docs/`](../) 下全部 47 篇英文文档的简体中文译文，文件名与英文页一一对应。

> **英文原文是权威版本。** 译文以提交时的英文页为准；两者不一致时以
> [`docs/`](../) 下的英文文档为准。术语与翻译规范见 [GLOSSARY.md](GLOSSARY.md)。
> 仓库根目录的中文介绍见 [../../README.zh-CN.md](../../README.zh-CN.md)。

## 从这里开始

| 文档 | 内容 |
| --- | --- |
| [INSTALL.md](INSTALL.md) | 安装 Strata |
| [MODELS.md](MODELS.md) | 选哪个模型？规模、版本，以及什么跑得动 |
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | Strata 是如何工作的？ |
| [SAFETY.md](SAFETY.md) | Strata 安全吗、会保护隐私吗？ |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 出了什么问题？ |
| [DETAILS.md](DETAILS.md) | Strata - 细节（API、引擎开关与测量的总集） |
| [AI_SETUP.md](AI_SETUP.md) | 为用户安装 Strata（面向 AI 助手的说明） |
| [MCP_SERVER.md](MCP_SERVER.md) | Strata 的 MCP 服务器：让你的 AI 助手安装并运行 Strata |

## 硬件与后端

| 文档 | 内容 |
| --- | --- |
| [MULTI_GPU.md](MULTI_GPU.md) | Strata 跑在两块或更多 GPU 上（层级切分） |
| [SECOND_GPU.md](SECOND_GPU.md) | CUDA1–3 专家层实验 |
| [NVIDIA_V100.md](NVIDIA_V100.md) | NVIDIA Tesla V100 / Titan V（Volta，sm_70）：社区构建 |
| [OLDER_GPUS.md](OLDER_GPUS.md) | 较老的 GPU（实验性） |
| [AMD_HIP.md](AMD_HIP.md) | AMD Radeon：HIP 后端（gfx1100、gfx1101、gfx1200、gfx1201、gfx1030） |
| [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) | RX 7900 XTX 支持与性能证据 |
| [STRIX_HALO.md](STRIX_HALO.md) | Strix Halo（Ryzen AI Max，gfx1151）：在 Linux 上从源码构建 |
| [INTEL.md](INTEL.md) | Strata 跑在 Intel Arc 上 |
| [INTEL_ARC.md](INTEL_ARC.md) | Intel Arc（实验性） |
| [A770_SPEED_SPEC.md](A770_SPEED_SPEC.md) | A770 decode 优化规格说明 |

## 模型与量化

| 文档 | 内容 |
| --- | --- |
| [QWEN36.md](QWEN36.md) | Qwen3.6-35B-A3B：图片、同时多个请求与速度 |
| [IMAGES.md](IMAGES.md) | 图像生成（Qwen-Image-2.1-Turbo） |
| [GYRO.md](GYRO.md) | Gyro（agentionai 的 rotor 量化）在 Strata 上 |
| [QUANTS.md](QUANTS.md) | Unsloth 的 K-quants 与 Q8_0：格式、内核与质量 |
| [UNSLOTH_Q4.md](UNSLOTH_Q4.md) | Unsloth UD-Q4_K_XL |
| [UNSLOTH_Q6.md](UNSLOTH_Q6.md) | Unsloth UD-Q6_K_XL |
| [THIRD_PARTY_QUANTS.md](THIRD_PARTY_QUANTS.md) | 第三方动态量化（不是 IQ4_NL 的 n-gram 表） |
| [NVFP4.md](NVFP4.md) | NVFP4 路由专家层 |
| [ORCA.md](ORCA.md) | OrcaRouter IQ3_XXS 兼容性 |
| [ORCA_Q4_K_S.md](ORCA_Q4_K_S.md) | OrcaRouter Uncensored Q4_K_S（手动 setup） |

## 引擎机制与优化

| 文档 | 内容 |
| --- | --- |
| [BATCHING.md](BATCHING.md) | 同时处理多个请求（批量槽位） |
| [BATCHED_DMA.md](BATCHED_DMA.md) | 批量 DMA 提交（实验性） |
| [KV_PREFETCH.md](KV_PREFETCH.md) | 用 prefill 计算重叠 KV 的流式上传 |
| [PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md) | 提示词尾部附近的可选检查点 |
| [MESSAGE_BOUNDARY_CACHE.md](MESSAGE_BOUNDARY_CACHE.md) | 在最后一条用户消息被编辑时复用长聊天历史 |
| [DISJOINT_EXPERT_CACHE.md](DISJOINT_EXPERT_CACHE.md) | 避免在 helper 与主 GPU 之间出现重复的专家槽位 |
| [EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md) | 常驻专家层交换缓冲区轮转 |
| [PREFILL_HEAD_ALIAS.md](PREFILL_HEAD_ALIAS.md) | 把输出头的显存借给 prompt 通路（`--prefill-head-alias`，需手动开启） |
| [EXPERTDEC.md](EXPERTDEC.md) | EXPDEC：把 verify window 的路由专家层与该组的激活值放进共享内存 |
| [QSAROWS.md](QSAROWS.md) | QSA_ROWS：把 decode 窗口的 QSA block 分数放进寄存器 |
| [MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md) | 交错式小批量 matvec（STRATA_MMVQ_IL）：如何为自己的卡测量一张表 |
| [VRAM_ELASTIC.md](VRAM_ELASTIC.md) | POST /v1/vram：它做什么、答复什么 |
| [CLOSING.md](CLOSING.md) | 收尾：什么会结束引擎，以及何时 |

## 集成与测试

| 文档 | 内容 |
| --- | --- |
| [LLAMA_SWAP.md](LLAMA_SWAP.md) | 在 llama-swap 后面运行 Strata |
| [LOGIT_BIAS.md](LOGIT_BIAS.md) | 按请求设置 token 偏置 |
| [MANAGEMENT_PROTOCOL.md](MANAGEMENT_PROTOCOL.md) | 属主本地的管理桥，版本 1 |
| [RESEARCH_RUNS.md](RESEARCH_RUNS.md) | 一份长文档，许多问题（0.1.40.2，可选开启） |
| [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) | 社区基准测试结果 |
| [TEST_REQUESTS.md](TEST_REQUESTS.md) | 请求的测试与想要的硬件 |
