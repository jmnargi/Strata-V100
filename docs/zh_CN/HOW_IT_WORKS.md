# Strata 是如何工作的？

<a id="how-does-strata-work"></a>

像 Qwen3.8-Flash-Next 这样的模型，通常跑在拥有数百 GB 显存的服务器上。而你的显卡只有 12-24 GB。Strata 通过
**把工作分摊到你整台 PC 上**来让它装得下——思路和厨房一样：常用的东西留在台面上，其余的放进储物间。回到
[README](../../README.md#how-does-it-work)。

## 谁做什么

<a id="who-does-what"></a>

<p align="center"><img src="media/how-it-works.svg" width="860" alt="模型的 24,576 个专家：最忙的在显卡上，全部在内存里，SSD 上放一张查找表"></p>

- **模型是一支由 24,576 个小型专家（expert）组成的团队，** 它每写出一个词，
  只需要其中 10 个。所以它不必把所有专家同时放在显卡上。
- **你的显卡** 承担每个词都要做的那部分工作，并常驻被问得最多的几千个专家。它会在你使用它的过程中
  不断学习哪些专家是它们。
- **你的内存** 保存全部专家。当一个词需要用到显卡上没有的专家时，**你的处理器** 会来计算它——
  与显卡同时进行，所以谁都不用等谁。
- **你的 SSD** 存放一张很大的查找表；模型每个词只读取其中很小的几行。

用技术语言说，也是这样：

- **GPU（显存）：** attention 与 DeltaNet mixer、gated-residual 权重、router、共享专家、输出头、
  MTP 草稿层、KV 缓存（从 64K 起：只放最常读的那部分，其余从内存流式读取），以及一个**专家
  缓存**，用最常用的专家填满显存的剩余部分（它会随你的对话自适应）。
  显存比更快的 GPU 更重要：每多 1 GB 就能多放约 700 个专家，而 GPU 上每多一个专家，CPU 就少算一个。
- **内存：** 全部 24,576 个专家，页锁定。CPU **就地**计算不在 GPU 上的那些专家，与 GPU 处理缓存中的专家
  同时进行（AVX-512 / AVX2 内核，i-quant 用 ggml 的内核）。在内存小、显卡大的 PC 上，
  [低内存模式](MODELS.md#a-big-graphics-card-and-little-ram)只在内存中保留 GPU 放不下的那些专家。
- **SSD：** 28.8 GB 的 n-gram 表，每个 token 通过操作系统缓存读取几行。

这在 NVIDIA（CUDA）和 AMD（HIP）显卡上完全一样：同一个引擎，分别为每种卡编译。

## 先猜，再检查

<a id="guess-then-check"></a>

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="一个小助手猜接下来的几个词；大模型一次性检查全部猜测，并保留正确的那些"></p>

- **先猜，再检查。** 模型内置一个小而快的助手来猜接下来的几个词，大模型则一次性检查所有的猜测。
  它保留自己认同的那些，并自己写出下一个词——于是一步常常产出好几个词。助手只负责猜——每个词都由
  大模型决定——所以你得到的答案是同样质量的，只是快了 1.6-1.8 倍。
- 用技术语言说：模型自带的 MTP 层草拟最多 3 个 token；一次遍历全部 48 层来检查它们，平均每趟 2.4-3.2
  个 token。当回复重复了上下文时（代码编辑、引用的文本），**prompt lookup** 会从先前的副本草拟最多 5 个
  token，但只在其实测接受率与代价表明划算的地方这么做：代码编辑快 6-11%，其他文本不变。这些草稿与 MTP 的
  草稿一样要经过检查，所以输出相同。

## 读取长文本

<a id="reading-long-texts"></a>

**长文本会分成大块读取**（一次最多 8,192 个 token——token 即文本被切分后的片段），所以一份长文档或代码库的读取速度超过每秒 1,000 个 token。下一层的专家通过 PCIe 流式传到 GPU，与此同时当前层的 attention 在运行。第一条消息之后，Strata 会保留整个对话，只读取新增的部分，所以后续提问几秒就能开始。

## 更多

<a id="more"></a>

[详情](DETAILS.md#how-it-works)解释了每一部分及其数字，[论文](../paper/Strata-Paper.pdf)讲述了完整的故事，以及它背后的实测数据。

## 鸣谢

<a id="credits"></a>

- 模型：[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，来自 Qwen 团队；
  压缩版本来自 [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)，
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) 来自 UkisAI；实验性的
  [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) 来自 Unsloth（它的支持见
  [eddoursul/Strata](https://github.com/eddoursul/Strata)）。它们的许可证适用于模型文件。
- 构建时使用了 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）的部分内容。想法来自
  [Splash](https://github.com/incoai/splash)、[ninfer](https://github.com/Neroued/ninfer) 和
  [HyperQwen](https://github.com/syv-ai/HyperQwen)。更多见[详情](DETAILS.md#credits-and-licenses)。
- AMD（HIP）后端：[AMD_HIP.md](AMD_HIP.md) 列出了贡献者以及用于验证的机器。

## 许可证

<a id="license"></a>

Strata 是开源软件，采用 [MIT License](../../LICENSE)。有少数部分带有自己的许可证：`third_party/ggml`
（MIT，llama.cpp / ggml）、Web 应用的字体（SIL Open Font License 1.1），以及实验性速度预测中位于
`data/experimental-speed-projection` 的向量（Qwen Community License 1.0，来自该模型的激活值）。
模型不属于本仓库；模型文件适用各自的许可证。
