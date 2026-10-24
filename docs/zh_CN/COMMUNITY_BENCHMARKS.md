# 社区基准测试结果
<a id="community-benchmark-results"></a>

分享你自己硬件上的实测数据，让其他用户能判断 Strata 在类似机器上可能跑成什么样。本文给出一种报告格式和 pull request
流程；只测一个模型、一块 GPU 或一种上下文长度，也足以贡献数据。

已有的[基准测试报告](../../bench/results)提供了范例。[引擎 0.1.26 速度报告](../../bench/results/2026-09-29-speed-0126/README.md)
包含硬件、设置和每次运行的 JSON。[技术细节](DETAILS.md#speed-measured)解释了已发布的实测数据及其局限。
请报告你实际测到的内容，并把估算值单独标注。

## 社区报告
<a id="community-reports"></a>

- [2026-09-30：RTX 5090、Core Ultra 9 285K、64 GB RAM](../../bench/results/2026-09-30-community-rtx-5090/README.md)：Strata 0.1.29，原版 Flash-Next IQ2_XS，131,072 token 上下文；分别在 4,096、32,768 和 128,000 提示词 token 下各跑三次，另加六项召回检查。
- [2026-10-08：2x TITAN RTX 24 GB（sm_75，NVLink 存在但未使用）、Xeon E5-2696 v4（无 AVX-512）、125 GiB RAM](../../bench/results/2026-10-08-community-2x-titan-rtx-0.1.40.3/README.md)：Strata v0.1.40.3 源码构建，原版 Flash-Next IQ3_S，262,144 token 上下文（模型原生窗口），带 --vision 和 --reasoning-budget-tokens 12000；分别在 4,096、32,768 和 128,000 提示词 token 下各跑三次。在这三种长度下 decode 为 70.9/77.3/71.1 tok/s，提示词为 868.9/1548.9/1626.9 tok/s，8,649 个缓存专家，32k 和 128k 处 needle 6/6。并指出引擎不使用 NVLink。
- [2026-10-03：2x AMD Instinct MI50 16 GB（gfx906）、Xeon E5-2666 v3、32 GB RAM](../../bench/results/2026-10-03-community-2x-mi50/README.md)：gfx906 构建（#638）加上 #639 和 #640，Coder IQ1_M，131,072 token 上下文，层切分跨两块卡；分别在 4,096、32,768 和 128,000 提示词 token 下各跑三次，另加六项召回检查。
- [2026-10-04：2x Intel Arc Pro B60 24 GB、Ryzen 5 5600、64 GB RAM](../../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md)：`6f32ec0` 上的 SYCL 移植版加两个修复，Coder IQ1_M 和 Flash-Next IQ2_XS，8,192 token 上下文，层切分跨两块卡；短提示词和 2K token 提示词，另加一次单块 B60 的运行。用的是 Intel 的 SYCL 引擎，没有做 4,096/32,768/128,000 token 的扫描。
- [2026-10-07：2x NVIDIA Quadro RTX 4000 8 GB、Xeon E5-2620 v3、96 GB RAM](../../bench/results/2026-10-07-community-2x-rtx-4000/README.md)：Strata 0.1.38，Swift-Qwen3.8-Flash-Next IQ2_XS，131,072 token 上下文，层切分跨双 GPU 并使用按 RAM 分层的专家缓存；实测了深上下文提示词摄取（73k–74k token，100–250 tok/s，decode 18–23 tok/s），以及与 12B 模型的生产流水线对比。
- [2026-10-08：Tesla V100-SXM2-32GB（sm_70）、Threadripper 2990WX（无 AVX-512）、96 GB RAM](../../bench/results/2026-10-08-community-v100-sm70-table/README.md)：Strata `fb58e0d`（v0.1.41）针对 sm_70 的源码构建，Unsloth UD-IQ4_XS 配原生专家层（expert）模型包，262,144 token 上下文，`--kv int8`、`--prefill auto`；**确认了需手动开启（opt-in）的 Volta decode 内核**（`STRATA_SM70_TABLE=1`），这正是 [NVIDIA_V100.md](NVIDIA_V100.md) 请 V100 用户核对的内容：decode 在两组交换顺序的对照中分别 **+9.8% 和 +15.0%**（28.16/28.64 → 30.92/32.94 tok/s），提示词吞吐不变，四次运行的生成 token **逐位相同（bit-identical）**。同时还指出 `tools/ab_engine.py` 的三段短对话低于这台机器的噪声本底（长提示词那一组方向相同，+5%）。
- [2026-10-08：一块 AMD Radeon AI PRO R9700 32 GB、两颗 EPYC 9334、503 GiB RAM](../../bench/results/2026-10-08-community-r9700-linux/README.md)：Linux/ROCm 7.2.3，冻结的实验版 0.1.40.3 源码，原版 Flash-Next IQ3_S 在 1K/4K/32K/128K 上各重复三次；另有针对 #1107 的 IQ3_S 五对和 IQ2_XS 两对短增量对比。包含逐请求数据和回归；没有给出适用于全部负载的建议，也没有更新 main 之后的实测。
- [2026-10-09：Radeon 8060S / Ryzen AI Max+ 395 Strix Halo 笔记本（HP ZBook Ultra G1a，70 W）、128 GB 统一内存](../../bench/results/2026-10-09-community-strix-halo-laptop-70w/README.md)：Strata v0.1.41 源码构建，ROCm 7.14.1，Unsloth UD-IQ4_XS，`docs/STRIX_HALO.md` §5 的快速配置，带 `--spec 5 --mtp-window 8192`；在 8,192 / 32,768 / 131,072 提示词 token 下做 4–6 次新进程运行：提示词 969 / 1,067 / 1,031 tok/s，decode 59.2 / 57.8 / 42.0 tok/s（中位数）。提示词受功耗限制（整包功耗上限 70 W）；needle 15/15，agentic 5/5。
- [2026-10-09：RTX 3060 12 GB、Core i7-12700、64 GB RAM](../../bench/results/2026-10-09-community-rtx-3060-0141/README.md)：Strata v0.1.41 官方 Windows 发行版，原版 Flash-Next Q2_0、IQ2_XS、IQ3_XXS 和 IQ3_S；提示词从 128 到 128K token 的扫描（其中三种做到 260K），另加召回、HumanEval 0-39、缓存、MTP 和并发检查。
- [2026-10-10：AMD Radeon RX 9070 16 GB（gfx1201）、Ryzen 7 5800X3D、64 GB RAM](../../bench/results/2026-10-10-community-rx-9070/README.md)：Strata 0.1.41 预编译 Windows HIP 引擎，Flash-Next IQ3_XXS，131,072 token 上下文；在 4,195、33,467 和约 130,000 提示词 token 下各跑三次，输出上限为 256 token，另加六项召回检查。在这三种长度下 decode 为 58.7/57.9/57.9 tok/s，提示词为 586/804/833 tok/s，33.5k 和 130k 处 needle 6/6；只用一个引擎会话，没有采集时钟频率。
- [2026-10-10：AMD Radeon RX 9060 XT 16 GB（gfx1200）、Ryzen 7 9800X3D、64 GB RAM](../../bench/results/2026-10-10-community-rx-9060-xt/README.md)：Strata 0.1.40.3 预编译 Windows HIP 引擎，Flash-Next IQ2_XS 和 IQ3_XXS：一轮 prefill（预填充）规模扫描（8,192 最好），一组对齐的 32K 编码对比（各 9/9 任务，IQ2_XS 的 decode 约快 17%），一次 113K 召回请求。包含脱敏后的数据、fixture 和脚本。
- [2026-10-10：2x Tesla P40 24 GB、Xeon E5-2699 v3、125 GB RAM](../../bench/results/2026-10-10-community-2x-p40-iq3_s/README.md)：Strata 0.1.41 CUDA 12 源码构建（sm_61），Flash-Next IQ3_S，131,072 token 上下文，层切分；decode 35.0 tok/s，3.9K/21.9K 处提示词 350/558 tok/s，needle 6/6，对话检查点。
- [2026-10-09：RTX 5060 Ti 16 GB + Xeon E5-2680 v4（Windows 10）](../../bench/results/2026-10-09-community-rtx-5060-ti-e5-2680-v4/README.md)：Strata 0.1.41，Coder IQ1_M 和 abliterated Q2_0，262K 上下文；在 4K / 32K / 128K 下的提示词与 decode。
- [2026-10-09：WSL2 下的 RTX 5060 Ti 16 GB、Ryzen 7 3700X、128 GB RAM](../../bench/results/2026-10-09-community-rtx-5060ti-wsl2-iq3_s-0141/README.md)：Strata 0.1.41 源码构建，IQ3_S 配实验性速度推估；只有单次运行（decode 48.2/47.7 tok/s，3.8K/19.2K 处提示词 343/598 tok/s），没有 needle 测试，也没有 TTFT。
- [2026-10-09：Radeon Pro VII 16 GB（gfx906）、Xeon E5-2666 v3、125.7 GiB RAM](../../bench/results/2026-10-09-community-radeon-pro-vii-16gb/README.md)：在发行版自带的 ROCm 6.2 上源码构建 0.1.40.4 和 0.1.41（宿主补丁见 BUILD.json），IQ2_XS 在 131K 下使用 KV 流式加载；decode 24.2/24.0/23.7 tok/s，4K/32K/128K 处提示词 350/339/284 tok/s，两轮扫描并带遥测。构建日志已裁剪（TRIMMED.md）。
- [2026-10-08：RTX 4070 12 GB、Core i5-12600KF、32 GB DDR4](../../bench/results/2026-10-08-community-rtx4070-iq2-xs/README.md)：IQ2_XS 配自适应缓存 E4/S24：在本地修改并自行构建的引擎上（不是官方发行版；源码 commit 和哈希见 provenance.json，步骤见 REPRODUCE.md）得到 48.23 accepted-decode tok/s（5 次的中位数）。

## 需要记录什么
<a id="what-to-record"></a>

请提供足够的信息，让别人能重现你的运行：

- **硬件：** GPU 型号、显存、多 GPU 运行时所选的 GPU、CPU、已装内存和存储类型。已知或相关时，还请给出 PCIe 链路速率与宽度、GPU 功耗上限，以及其他并发负载。
- **软件：** 操作系统、Strata commit 与引擎版本、驱动与 CUDA 版本（HIP 则为 ROCm），以及用的是发行版二进制还是源码构建。请给出改动过的构建选项。
- **模型：** 能提供时给出确切的仓库与 revision、量化格式、GGUF 文件名，以及启用时的视觉编码器。说明自定义模型包、专家层 profile 或 draft 词表，并给出哈希或可重复的准备步骤。
- **设置：** 启动命令和相关运行配置、上下文上限、KV 类型与流式窗口、专家缓存、低内存模式、prefill 规模、CPU worker 数、MTP、reasoning effort、采样和视觉。说明是否启用了校准（calibration）或实验性速度推估；并附上由此产生的设置。
- **负载：** 基准测试命令或脚本、可分享的提示词或构造提示词的说明、实际的提示词与生成 token 数、输出上限、重复次数，以及每次运行的缓存状态。对逐轮 agentic 运行，请给出每一轮实测的新增与复用的 token 数。

GB 与 GiB 请统一使用并写明单位。无法获得的值要显式标注，不要猜。报告请用 ASD-STE100 简化技术英语撰写：
句子要短，每句只讲一条指令或一个事实，用主动语态，同一个事物始终用同一个术语。发布前，
从配置、提示词和日志中删除凭据和私人内容。

## 如何测量
<a id="how-to-measure"></a>

先从你机器上已经能跑的配置开始。不需要下载每一个模型，也不必测试最大的上下文。一份有用的初始速度报告
只需测一个短提示词和一个能放下的较长提示词，并固定输出上限（例如 256 token）。如果模型提前结束，
请记录实际输出长度。

每种配置至少做三次实测运行，并把全部结果连同中位数和范围一起发布。如果只有一次运行，请明确说明。
请说明是否有预热（warm-up），以及计时是否包含模型加载。

Strata 会复用对话前缀并自适应调整专家缓存。请把首次完整处理的提示词与复用 token 的后续请求区分开来，
并从引擎日志中报告复用和重新读取的 token 数。说明你在两次运行之间如何重置或保留状态。已预热的专家缓存
与复用的提示词前缀是两种不同的条件；尽可能都记录下来。

请把这几种测量分开记录：

- **提示词吞吐：** 每秒新处理的提示词 token 数，取自引擎的计时输出。同时保留对应的 token 数和耗时。
- **Decode 吞吐：** 每秒生成的 token 数，取自引擎的计时输出。推理 token 也要计入，若采用别的口径请说明。
- **首 token 时间：** 从发出请求到产生第一个生成 token 的秒数。对流式 API 的测量，忽略 keep-alive 和空 delta，
  并说明第一个 token 是推理内容还是回答文本。
- **总延迟：** 整个请求经过的秒数。说明是在引擎侧还是客户端侧测量，以及是否包含加载、视觉编码或排队。
- **内存：** 观察到的 RAM 与显存占用，并说明该值是启动时的快照还是推理期间的峰值。请报告换页（paging）
  或显存不足（out-of-memory）的失败。

请附上引擎的计时输出行，以及你使用的测量脚本（如果有）。不要用生成 token 数除以请求总时间来计算
decode 吞吐，因为后者还包含提示词处理。被取消或失败的请求应作为失败记入报告，不要混进成功的吞吐汇总。

比较版本或设置时，请在同一台机器、同一负载上把两者都跑一遍，并列出每一处改动的设置。请保留每次运行的
结果：即使输入完全相同，输出文本和 draft 接受率也会改变速度。

## Agentic 短轮次
<a id="agentic-short-turns"></a>

编码 agent 不会只发一个长提示词。它先发一次很长的共享前缀，然后发很多短轮次。每一轮会追加一条
assistant 消息和一条工具结果，长度在几十到几百 token 之间。在下面的负载中，提示词的大部分已经在
对话缓存里，所以单轮的新增 token 很少。此时若存在与这些 token 无关的每请求固定开销，就会表现为
很大的 `prompt_per_token_ms`。请在你的机器上实测，不要假设。

用 [`tools/agent_turns_bench.py`](../../tools/agent_turns_bench.py) 测量这种形态。下面的例子把已安装的引擎
与一个增加两个引擎选项的候选版本作对比。它只是示例，本文没有实测它，也不是推荐做法。

```bash
python tools/agent_turns_bench.py strata-<model>.json 'base:' 'candidate:xargs=--pipeline-windows;1;--short-read;128' --workload agentic --rounds 2
```

同一个工具也能在没有 GPU、没有模型的情况下运行，对接服务器的 mock 引擎。用下面的形式检查测试框架本身：

```bash
python tools/agent_turns_bench.py strata-<model>.json 'base:' 'candidate:' --engine mock --workload agentic --prefix-tokens 256 --turns 4 --max-new 8 --rounds 1
```

该工具为每个 arm（对照臂）启动一个服务器，向每个 arm 发送相同的轮次，并报告服务器自己在 `timings` 中给出的数字：

- **新增 token：** `prompt_n`，引擎在这一轮读取的 token 数。
- **复用 token：** `cache_n`，对话缓存中已经持有的 token 数。请以占提示词的比例报告。
- **Prefill 延迟：** `prompt_ms` 和 `prompt_per_token_ms`，即每个新增 token 的耗时。短轮次上出现很大的
  `prompt_per_token_ms`，说明存在不随新增 token 增长的开销。
- **Decode：** `predicted_per_second` 和 `predicted_ms`。
- **整个调用：** 工具自身的 `request_ms`，用请求前后的墙钟时间测得。对话的停放（parking）与恢复，
  以及引擎外围的 setup，都包含在内。引擎自身的时间（`prefill_ms` 加 `predicted_ms`）并不是一轮的全部开销。

工具会把冷轮次（一个会话的第一轮：整个提示词都是新的）单独列出，不计入短轮次的中位数。一次长时间的
冷读取并不代表 agent 工作时的每轮开销。

工具还会给出每一轮中其它 arm 相对第一个 arm 的倍数。倍数大于 1 更好。对 prefill 延迟，倍数是第一个 arm
的毫秒数除以该 arm 的毫秒数。对 decode 速率，倍数是该 arm 的每秒 token 数除以第一个 arm 的。

`--prefix-tokens` 和 `--additions` 是请求的规模，不是实测值。工具用常见英文单词构造文本，因此一个单词
大约就是一个 token。一轮还会加上 assistant 的那一行和工具调用信封：在本文具的工具集和 chat template 下
约 55 token。请报告实测的 `added`、`prompt_n` 和 `prompt_tokens` 值。绝不要把请求的规模当作实测值报告。

如果两个 arm 在某一轮读取的新增 token 数不同，说明它们做的工作不同。不要在该轮上比较它们。

给每个 arm 单独的对话缓存磁盘目录。磁盘缓存能跨重启保留，若共用目录，一个 arm 就可能恢复另一个 arm
停放的对话。工具自己就这样做。它还会把服务器绑定到 127.0.0.1，从生成的配置里去掉 API key，
并在每个 arm 结束后停掉服务器及其引擎。

实测运行前，请把你线上服务的设置导出到环境变量。每个 arm 都继承该工具的环境。少设一项，
测的就是另一种配置。

## 可选的正确性检查
<a id="optional-correctness-checks"></a>

只看速度测量无法证明回答质量。请加一个小检查，与你的负载相关，例如一个带测试结果、已完成的编码任务，
一次工具调用及其返回结果，或者一道有预期答案的图片问题。

长上下文召回方面，仓库提供了 [`tools/needle_bench.py`](../../tools/needle_bench.py)。在本地服务器运行中时，
从仓库根目录执行：

```bash
python tools/needle_bench.py --help
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

请选择你所配置的上下文放得下的长度。脚本支持 `--api-key`，用于需要认证的服务器；不要把该 key 写进
公开的命令。除了成功项，也要保存报告的实际提示词长度、未命中、报错和跳过的情况。needle 测试衡量的是
这些输入上的召回，而不是模型的整体质量。

## 提交报告
<a id="submit-a-report"></a>

1. Fork Strata，并为你的报告创建一个分支。
2. 添加一个类似 `bench/results/YYYY-MM-DD-community-rtx-5090/` 的目录。用测量日期加简短的硬件标签；若重名就加后缀。
3. 在该目录里放入一份基于下面模板的 `README.md`，以及每次运行的 JSON 或 CSV、复现所需的小脚本或提示词。你可以用自己的数据格式，但要说明各字段和单位。不要把大模型文件和生成的模型包放进 PR。
4. 向 `Niko1221/Strata:main` 发起 pull request。概述硬件、模型、测过的配置和局限。只有结果的提交请与引擎改动分开，方便评审直接判断这些测量。
5. 报告用 ASD-STE100 简化技术英语撰写。句子要短，每句只讲一条指令或一个事实。用主动语态。同一个事物始终用同一个术语。引擎的术语（`prompt_n`、`cache_n`、`prompt_ms`）要与引擎输出完全一致。

维护者会评审报告并决定把它收进哪里。添加报告不需要改动 README 里头条的性能声明。运行失败或疑似 bug 时，
带配置和相关日志的 GitHub issue 可能比一个基准测试 PR 更有用。

## 报告模板
<a id="report-template"></a>

把下面的内容复制到你报告的 `README.md`，替换占位符，删掉不适用的章节。缺失的测量写 `not measured`；
表格里不含任何示例性能数字。

````markdown
# GPU 型号上的社区基准测试

由 HANDLE 于 DATE 实测。说明测了什么，以及主要的局限。

## 硬件与软件

- GPU 与显存；CPU；已装内存；存储；已知的 PCIe 链路：
- 操作系统；驱动；CUDA 或 ROCm：
- Strata commit；引擎版本；发行版二进制或源码构建：
- 后台负载与任何功耗上限：

## 模型与配置

- 模型仓库与 revision；量化格式；GGUF 文件名：
- 视觉编码器；自定义模型包或 profile：
- 上下文；KV 类型与流式加载；缓存；prefill；低内存模式：
- MTP；reasoning；采样；校准；实验性速度推估：

```text
把确切的启动命令和相关配置写在这里，并去掉凭据。
```

## 方法

链接提示词和脚本。说明输出上限、重复次数、预热、加载、提示词复用、专家缓存状态、计时边界和内存测量方式。

## 结果

| 配置 | 实际提示词 token | 复用 token | 生成 token | 运行次数 | 提示词 tok/s 中位数与范围 | Decode tok/s 中位数与范围 | TTFT 秒 中位数与范围 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 填入实测值 | | | | | | | |

链接每次运行的数据和日志。实测了就给出总延迟和内存观察值，并写明单位与计时边界。列出失败和被跳过的情况。

## 正确性与局限

说明各项检查、预期答案、观察到的结果，以及未测试的行为。
````
