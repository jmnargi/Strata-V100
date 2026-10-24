# 选哪个模型？规模、版本，以及什么跑得动
<a id="which-model-sizes-versions-and-what-fits"></a>

Strata 的主模型是 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，有若干**规模**（同一个模型，压缩得多
一点或少一点）和若干**版本**（原版、一个代码版、一个微调版、其它厂商的压缩版）。它还跑一个更小的模型
[Qwen3.6-35B-A3B](#qwen36-35b-a3b)。桌面应用、终端（`strata launch`）和安装程序会为你的 PC 推荐一个；本页解释
这个选择。回到 [README](../../README.md)。

> **本页内容：** [按内存选](#pick-by-memory) · [速度](#how-fast-is-each-size) · [各规模](#the-sizes) ·
> [装得下吗？](#will-it-fit) · [各版本](#the-versions) · [Qwen3.6](#qwen36-35b-a3b) ·
> [添加或切换模型](#adding-or-switching-models)

## 按内存选
<a id="pick-by-memory"></a>

这些档位把 RAM 和 VRAM（显卡的内存）合在一起算，和 [README](../../README.md#the-other-sizes) 一样。`strata launch`、setup
和 MCP 服务器的安装工具按它们预选；桌面应用从 0.2.1 起也这样。其它每个规模都仍是一个可选项。一张 12 GB 显卡加 32 GB
RAM 的 PC 属于 40 GB 档（32 + 12 = 44）。

| 你的内存（RAM + VRAM） | 预选 | 说明 |
| --- | --- | --- |
| **32 GB** | **Gyro-S**，写代码则用 **Coder** | Coder 需要 32 GB RAM，且只保留一半专家层（expert），所以在代码之外更弱（[原因](#coder)）。Gyro-S 的质量低于 2-3-bit 各规模，而 2-bit 的 Q2_0 decode（解码）更快 |
| **40 GB** | **Swift 1.5** IQ2_XS（39.2 GB） | Gyro-M（35 GB）是另一个选择：质量接近 IQ3_XXS，但写答案比 Q2_0 慢 17-21%。Gyro 能在 NVIDIA 和 AMD 显卡上跑，不能在 Intel Arc 或 Mac 上跑 |
| **48 GB** | **IQ3_XXS**（47.0 GB） | 更慢，更好 |
| **64 GB** | **IQ3_S**（54.8 GB），原版。Swift 1.5 的 IQ3_S 是同等选择 | 最慢，最好：在已公布的测试上与原版全模型持平 |
| **96 GB 或更多** | [Unsloth 的 UD-IQ4_XS 到 Q8_0](#unsloths-files-ud-iq4_xs-to-q8_0)，4 到 8 bit | UD-IQ4_XS 从约 80 GB RAM 起就把每个专家层放进 RAM，Q8_0 从约 136 GB 起。直到 192 GB，5-8-bit 的文件都能装进 RAM。setup 在这里仍预选 IQ3_S，Strix Halo 除外 |

内存比某一行所说更少时，Strata 的[低内存模式](#a-big-graphics-card-and-little-ram)会自行启动，模型跑得更慢，显卡小的
话慢得多。上面的数字是 RAM + VRAM；[装得下吗？](#will-it-fit)只算 RAM，针对那些把专家层留在 RAM 里的规模。

拿不准？就用安装程序预选的那个。Coder 能装进 32 GB 的 PC，但它只保留 512 个专家层中的 256 个，而且是按代码数据挑
的，所以在代码之外、以及英语以外的语言里更弱。日常使用，只要内存允许就选保留全部专家层的规模（Q2_0、IQ2_XS 或
IQ3_S）。

## 各规模有多快
<a id="how-fast-is-each-size"></a>

这些是较旧的数字，保留下来是因为它们能在各规模之间互相比较。较新的引擎在有些显卡上更快（每项改动都带着自己的实测
数字列在[详细说明](DETAILS.md)里）；0.2.0 引擎（0.1.45.3）还没有在整张表上重测过。实测环境：RTX 5070（12 GB）、
Ryzen 5 7600、64 GB RAM，引擎 0.1.26（Q2_0 那一行：引擎 0.1.36，它让 Q2_0 的提示词更快）：

| 规模 | 写答案（短对话） | 写答案（128K 上下文） | 读你的提示词 |
| --- | ---: | ---: | ---: |
| **Q2_0** | 94 tokens/s | 76 tokens/s | 2,650 tokens/s |
| **IQ2_XS**（2-bit 规模） | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS**（3-bit 规模） | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

实测环境：AMD RX 9070 XT（16 GB）、Ryzen 9 3900X、47 GB RAM（Linux，setup 自己的安装，2026-10-02，一个早于 0.1.40 的
引擎；Q2_0 高于 setup 对 47 GB 的 RAM 估计，是用 `--model Q2_0 --yes` 装的）：

| 规模 | 写答案（短对话） | 写答案（128K 上下文） | 读你的提示词 |
| --- | ---: | ---: | ---: |
| **Q2_0** | 60 tokens/s | 48 tokens/s | 1,160 tokens/s |
| **IQ2_XS**（2-bit 规模） | 52 tokens/s | 36 tokens/s | 1,110 tokens/s |
| **Coder** (IQ1_M) | 44 tokens/s | 33 tokens/s | 1,420 tokens/s |

- **写答案** = 回复出现的速度（tokens per second，一个 token 约为 ¾ 个单词）。
- **读你的提示词** = 吃进你发的内容（长文档、代码、聊天记录）的速度，在 32K token 的提示词上测得；4K 提示词则为
  910-1,580 tokens/s。32K 提示词用 Q2_0 约 15 秒。

显存更多的显卡更快，因为模型有更多部分能装进 GPU。RTX 3090（24 GB）的[详细说明](DETAILS.md#other-gpus-estimated)给的
是估计值，不是实测。其它显卡的全部实测、长上下文数字和估计都在[详细说明](DETAILS.md#speed-measured)里。各 AMD 显卡的
实测（RX 9070 XT、Radeon AI PRO R9700、RX 7800 XT、RX 9060 XT、RX 6900 XT）在 [AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201)
里。

每台 PC 都不一样：`START-HERE.bat --calibrate` 会在你的机器上测几项引擎设置并保留最快的（约 15-30 分钟，慢显卡更久；
在上面的 PC 上它让 Coder 快了 7%；目前限 NVIDIA 显卡）。`strata launch` 和应用有自己的调优步骤（很快：约 20 分钟）。
自己实测过 Strata？[社区基准结果](COMMUNITY_BENCHMARKS.md)里有报告模板，以及如何通过 pull request 分享你的结果。

## 各规模
<a id="the-sizes"></a>

| 模型 | RAM+VRAM 要求 | 速度 | 质量 |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | 最快 | 好 |
| **IQ2_XS** | 39.2 GB | 快 | 更好（40 GB 档**推荐**，即 Swift 1.5） |
| **IQ3_XXS** | 47.0 GB | 更慢 | 很好 |
| **IQ3_S** | 54.8 GB | 最慢 | 最好：在已公布的测试上与原版全模型持平 |

三个更小的规模下载量为 66-76 GB（[详细说明](DETAILS.md#which-model)）；第一次启动还会取 MTP 草稿层（约 6 GB，带图片
时 +1 GB）。

## 装得下吗？
<a id="will-it-fit"></a>

分片 1 是模型启动时被加载的那部分：它的专家层进你的 **RAM**，其余部分进你的显卡（第二个分片，一张 29 GB 的查找表，
留在 SSD 上）。所以只要你的 **RAM 至少有专家层 + 约 10 GB** 留给 Windows 和其它程序，就装得下——专家层是分片 1 的
大部分：Q2_0 为 34 GB，IQ2_XS 35.5，IQ3_XXS 43，IQ3_S 50，Coder 23（分片 1 里的 dense 权重进显卡）。有 64 GB RAM 时每个
规模都装得下（IQ3_S 要少开别的程序）；有 48 GB 时是 Q2_0 和 IQ2_XS。显卡更大让它更快，但不会降低所需的 RAM
 - 下面的低内存模式除外。

### 大显卡配小内存
<a id="a-big-graphics-card-and-little-ram"></a>

在 RAM 装不下专家层加系统的 PC 上，setup 改为从模型文件里映射它们，只在 RAM 里保留显卡装不下的部分（[低内存模
式](DETAILS.md#speed-measured)，由 setup 选择）。例如，32 GB 的 PC 配 24 GB GPU 用这种方式跑 Q2_0、IQ2_XS 和 Coder，
而 32 GB 的 PC 配 12-16 GB GPU 跑 Coder。显卡小时，大多数专家层就来自 SSD，慢得多（setup 会说明）。
`START-HERE.bat --setup --low-ram on|off` 可以覆盖这个选择。

## 各版本
<a id="the-versions"></a>

**图片。** setup 菜单里的各版本都能读图片，只要模型能读图片，图片就默认开启。显卡小于 24 GB 时图片读取器跑在 CPU
上（每张图几秒，不占显卡），从 24 GB 起跑在显卡上（约 1-1.4 GB 显存；每张图 0.1-0.5 秒）。NVIDIA 和 AMD 显卡上的规则相
同。`--vision off`（或应用和 `strata launch` 里的 "No"）关掉图片；`--vision cpu` 或 `--vision gpu` 选读取器。在 AMD 上，
`strata-vision-hip` 是显卡用的读取器；它起不来时，Strata 会自行改用 CPU 读取器（[AMD_HIP.md](AMD_HIP.md)）。Mac 应用和
Mac 终端还没有图片读取器（[MAC.md](../MAC.md)），Intel Arc 引擎也没有（[INTEL.md](INTEL.md#not-done)）。更多：
[图片](DETAILS.md#images-vision)。

### Qwen3.8-Flash-Next
<a id="qwen38-flash-next"></a>

原版，四种规模都有。带图片，并可选用[实验性速度投影](DETAILS.md#experimental-speed-projection-experimental-off-by-default)。

### Coder
<a id="coder"></a>

**[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** —— ISTA-DASLab 的代码版：它保留每层
512 个专家层中的 256 个，按代码（连同 agentic 和图片）数据挑选，丢掉其余的（据其作者，为全模型 SWE-bench Verified 分数
的 91%、LiveCodeBench 的 99%）。只有一种规模（IQ1_M：其专家层的存储方式与 IQ3_S 相同）：分片 1 为 **29.6 GB**，所
以能装进 **32 GB RAM** 的 PC，在 64 GB 上跑 262K 上下文，并且读长提示词是所有版本里最快的。丢掉一半专家层后，它在
代码之外、以及英语和中文等 CJK 文本以外的语言里更弱（#438：英语正常的地方，中文答案出错或循环）。日常使用，或者只
要你的 RAM 允许，就选保留全部专家层的规模：Q2_0、IQ2_XS 或 IQ3_S。更多：
[详细说明](DETAILS.md#or-the-coder-half-the-experts-for-code)。

```
START-HERE.bat --setup --family coder
```

### Swift 1.5
<a id="swift-15"></a>

**[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** —— UkisAI 的一个微调版，回答前思
考得短得多，所以你更早拿到答案，质量大致相同。每个 token 的速度相同，RAM 也与同规模的原始版大致相同。适用它自己的
许可（见其页面）。更多：[详细说明](DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter)。

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

<a id="unsloth-ud-iq4_xs"></a><a id="unsloth-ud-q4_k_xl-experimental"></a>

### Unsloth 的文件：UD-IQ4_XS 到 Q8_0
<a id="unsloths-files-ud-iq4_xs-to-q8_0"></a>

setup 菜单里的第四个版本（`--family unsloth`）是同一模型的 **Unsloth 量化版**，约 4 到 8 bit。它们用比上面 2-3-bit 各规
模更多的 bit 保留原版的专家层，所以需要更多 RAM。专家层加另外 16 GB 装得下时，setup 把每个专家层都放进 RAM（最快的
方式）；RAM 更少时它放进尽可能多的专家层，其余的在回答期间从 SSD 读（慢得多），而对更大的文件会先问一声。图片默认
开启（显卡内存小于 24 GB 用 CPU 读取器，从 24 GB 起用显卡的读取器）。

| 规模 | 下载 | 专家层 | 全部放进 RAM 所需 RAM | 与 Q8_0 相同的 top token | AMD R9700（32 GB）：故事 / 代码 / 32K 提示词 |
| --- | ---: | ---: | ---: | ---: | --- |
| **UD-IQ4_XS**（setup 的第一个） | 94 GB | 55 GiB | ~72 GB | 88.9% | - |
| **UD-Q4_K_XL** | 111 GB | 72 GiB | ~88 GB | 93.1% | 63 / 87 tok/s / 1,110-1,320 tok/s |
| **UD-Q5_K_XL** | 158 GB | 92 GiB | ~108 GB | 94.6% | 57-60 / 82 / 810-1,260 |
| **UD-Q6_K_XL** | 169 GB | 102 GiB | ~118 GB | 94.6% | 54 / 76 / 1,020 |
| **Q8_0** | 188 GB | 120 GiB | ~136 GB | 100%（参考） | 53-57 / 66-74 / 1,230-1,300 |

作为对比，同一张卡上的 IQ3_S：75 / 95 tok/s，为与 Q8_0 相同 top token 的 87.7%。这些速度数字来自比 0.1.45.3 更早的引擎
（条件见 [QUANTS.md](QUANTS.md)）。速度上限以及 NVIDIA/AMD/Intel 的细节：[QUANTS.md](QUANTS.md)；
文件：[UD-Q4_K_XL](UNSLOTH_Q4.md)、[UD-Q6_K_XL](UNSLOTH_Q6.md)。

```
START-HERE.bat --setup --family unsloth --model UD-Q5_K_XL
```

<a id="gyro"></a>

### Gyro-S 和 Gyro-M（agentionai）
<a id="gyro-s-and-gyro-m-agentionai"></a>

setup 菜单里的第五个版本（`--family gyro`）是同一模型的 **agentionai Gyro 量化版**：在旋转基下做 trellis 编码的专家
层。它们是最小的选项，从来不是 setup 的默认。下载里含 28 GB 的 n-gram 表；内存数字是 VRAM 加 RAM。

| 规模 | 下载 | 所需内存（VRAM + RAM） | 通俗说法 |
| --- | ---: | ---: | --- |
| **Gyro-S** | 58.5 GB | 27.6 GB | 最小的选项，给小内存的 PC；质量较低 |
| **Gyro-M** | 92.0 GB | 35 GB | 质量约等于 IQ3_XXS，GPU 内存约少 20%；在 12 GB 显卡上 decode 比 Q2_0 慢约 20% |

NVIDIA，以及 AMD（实验性）；不支持 Intel Arc 或 Mac。Gyro 能读图片（规则同上）。它需要 0.1.45.1 或更新的引擎；引擎更
旧时，应用会说 "This model needs a newer engine" 并提供 Update engine。在带引擎 0.1.45.3 的 RTX 50 显卡上，Gyro 输出
正确。细节、实测与致谢：[GYRO.md](GYRO.md)。

```
START-HERE.bat --setup --family gyro --model Gyro-S
```

<a id="nvfp4"></a>

### NVFP4：三个版本（NVIDIA 显卡）
<a id="nvfp4-three-versions-nvidia-cards"></a>

setup 菜单里的第六个版本（`--family nvfp4`）把专家层保存在 NVIDIA 的 4-bit NVFP4 格式里。它有三个版本。应用按权重由
谁制作来显示它们：

| 应用中的名称 | setup 中的名称（`--model`） | 是什么 | 下载 |
| --- | --- | --- | ---: |
| **NVFP4 (Strata's version)** | `NVFP4` | Strata 自己的转换（GPTQ；现成的模型包是 sergqwer 的 strata-nvfp4）。三者中质量最好 | 128 GB |
| **NVFP4 (NVIDIA's version)** | `NVFP4-NVIDIA` | NVIDIA 自己的 NVFP4 checkpoint（普通舍入），为 Strata 转换而来。大小和速度与第一个相同 | 128 GB |
| **NVFP4 (NVIDIA, trained for FP4)** | `NVFP4-QAD` | NVIDIA 重新训练的 4-bit 模型，短提示词上略快（[见下](#nvfp4-qad)） | 102 GB（另需 110 GB 磁盘来准备） |

三者都需要约 80 GB RAM 才能把每个专家层放进 RAM，只在 NVIDIA 显卡上跑，并且能读图片（规则同上）。RTX 50 显卡有 FP4
tensor core；其它 NVIDIA 显卡以 8-bit 精度跑 NVFP4。前两个需要 0.1.44 或更新的引擎，第三个需要 0.1.45 或更新。细节与实
测：[NVFP4.md](NVFP4.md)。

```
START-HERE.bat --setup --family nvfp4 --model NVFP4
```

<a id="nvfp4-qad"></a>

#### NVIDIA 重新训练的 4-bit 模型（NVFP4-QAD）
<a id="nvidias-retrained-4-bit-model-nvfp4-qad"></a>

NVIDIA 为 4-bit 专家层重新训练了 Qwen3.8-Flash-Next（quantization-aware distillation），并以
[Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) 发布。在 setup 的 NVFP4 系列
里它就是 **NVFP4-QAD** 这个规模。它不是现成的模型包：setup 按原样下载 NVIDIA 的文件（102 GB），并**在你的 PC 上把它
们转成 Strata 模型包**（一次，20-40 分钟；运行期间另需约 110 GB 磁盘，之后那 102 GB 可以删掉）。Strata 不重新分发
NVIDIA 的任何权重。

| 规模 | 下载 | 所需内存 | 通俗说法 |
| --- | ---: | ---: | --- |
| **NVFP4-QAD**（应用里：NVFP4 (NVIDIA, trained for FP4)） | 102 GB（另需 110 GB 磁盘来准备） | 同 NVFP4：约 80 GB RAM 就能把每个专家层放进 RAM | NVIDIA 的 FP4，为 FP4 而训练——短提示词上比另外两个 NVFP4 版本略快（Strata 的版本质量最好） |

只支持 NVIDIA 显卡。它能读图片（通过原版的图片读取器文件，setup 会连同它一起下载）。其许可是 NVIDIA Open Model
Agreement 和 Qwen 的 Community License。技术细节与实测：[NVFP4.md](NVFP4.md#nvidias-fp4-local-checkpoint)。

```
START-HERE.bat --setup --family nvfp4 --model NVFP4-QAD
```

### OrcaRouter Uncensored IQ3_XXS
<a id="orcarouter-uncensored-iq3_xxs"></a>

**OrcaRouter 的 Flash-Next Uncensored IQ3_XXS** 见[手动兼容安装](ORCA.md)。它需要显式做一次模型包转换，不是安装程序
菜单里的选项。

## Qwen3.6-35B-A3B
<a id="qwen36-35b-a3b"></a>

来自同一厂商的更小模型：[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)（350 亿参数，每个 token 激活 30
亿，256 个专家层，Apache 2.0 许可）。Strata 在自己的引擎路径上跑[带 MTP 草稿层的 Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)，
由文件决定用哪条。它在应用的模型列表和 setup 里（`--family qwen36`）。它能在 NVIDIA 和 AMD 显卡上跑；Intel Arc 引擎和
Mac 引擎跑不了它。它能读图片，并能同时服务多个请求。

| 规模 | 文件 | setup 规划的 RAM | 通俗说法 |
| --- | ---: | ---: | --- |
| **UD-Q4_K_XL** | 22.9 GB | 32 GB | 4-bit（Unsloth Dynamic），最小 |
| **UD-Q6_K_XL** | 32.6 GB | 44 GB | 6-bit，更接近全模型 |
| **Q8_0** | 37.8 GB | 52 GB | 8-bit，最接近全模型 |

图片读取器文件再加 0.9 GB。引擎把能放下的每个专家层都留在显卡上，其余在 CPU 上算。用引擎 0.1.45 和 UD-Q4_K_XL 实测：
在 Radeon AI PRO R9700（32 GB）上 decode 117.8 tokens/s，在 RTX 3060（12 GB，大多数专家层在 CPU 上）上 59.7 tokens/s。
图片、并发请求、完整实测与限制：[QWEN36.md](QWEN36.md)。

```
START-HERE.bat --setup --family qwen36 --model UD-Q4_K_XL
```

## 添加或切换模型
<a id="adding-or-switching-models"></a>

- **在应用里：** Models 页面列出每个模型及其大小、是否适合你的电脑；挑一个，按安装按钮。
- **在终端里：** `strata launch` 有一个 Model 步骤，列表相同（还有 "Use model files I already have"）。
  `strata models list` 显示已安装的内容；`strata models import <folder or .gguf> --plan` 显示一组你已有的文件会变成什
  么，而不改动任何东西。
- **用安装程序：** `SETUP.bat`（等同于 `START-HERE.bat --setup`；Linux 上是 `./setup.sh --setup`）随时再加一个模型。

另一个模型共用的文件不会重复下载（Coder 用原版的分片 2 和视觉编码器）。装了不止一个模型时，`START-HERE.bat` 会问启
动哪一个；`run-<model>.bat`（Linux：`run-<model>.sh`）直接启动一个。
