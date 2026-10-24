# 避免在 helper 与主 GPU 之间出现重复的专家槽位
<a id="avoid-duplicate-expert-slots-across-helper-and-primary-gpus"></a>

## 为什么有帮助
<a id="why-this-helps"></a>

在有两块小 GPU 的 PC 上，helper 专家一开始与主缓存相互独立。之后，自适应主缓存可能提升一个
helper 已经持有的专家。两块卡于是为同一个专家各花一份内存。

我们在持续请求期间观察到约 1,900-2,000 个重复槽位。
这一可选（opt-in）改动把 helper 持有的专家排除在主缓存的提升候选之外，
服务器和命令行生成路径都是如此。

## 使用
<a id="use"></a>

在启动引擎前设置 `STRATA_DISJOINT_ADAPT=1`。它默认关闭。
它不改变现有的 peer 层检查和专家相关算术。
该检查是实时的：每个提升候选都会去问 helper 自己的缓存（`RemoteExperts::holds`），所以
helper 的层之后换入或换出的专家会被跟上（加载时拷贝的一份掩码会过期）。`holds` 需要 helper-share
改动（#854）。

## 验证环境
<a id="validation-environment"></a>

- 两块改装过的 RTX 3080 20 GB 显卡；CUDA SM86、CUDA 13.1、Linux。
- Intel Xeon E5-2686 v4（18 核 / 36 线程），可用 RAM 约 94 GiB。
- PCIe 3.0，GPU 之间无 peer-to-peer 访问；17 个专家池 worker。
- Qwen3.8-Flash-Next IQ3_S，131,072 token 上下文，INT8 KV，
  32,768 个常驻 KV cell。
- Medium thinking：2,048 token 的推理上限，8,192 token 的输出上限。
- 物理 GPU 功耗上限保持在 250 W 和 280 W。没有提高功耗。
- 历史性能测试使用的是基于官方 v0.1.38 加经评审的 architectds/Strata 改动（最佳为 05c0f36）的隔离引擎。
  本 PR 只包含我们的改动，已移植到官方 main（99f3dbd）。
  历史计时并非对这个纯上游移植版的实测加速。

## 历史收益
<a id="historical-benefit"></a>

相同二进制、相同 helper 容量，每组六个完整的 JSON 任务：

| 32K 缓存请求 | 原始 | 归属守卫 |
| --- | ---: | ---: |
| 生成 | 81.5 tokens/s | 94.1 tokens/s |
| 完整请求 | 12.997 s | 11.417 s |

生成提升了 15.5%；请求时间下降了 12.2%。
这些任务返回了完整的 100 条记录数组和正确的源事实。
原始、加守卫和更大缓存三组共 18 项检查全部通过。
更大 helper 的那一组没有包含在上面的对比中。
这些重复性任务并不代表所有 agent 负载。

## 验证与剩余工作
<a id="validation-and-remaining-work"></a>

- 独立的归属测试通过，覆盖禁用、空、无归属、有归属和越界条目，外加候选过滤。
- 针对纯官方 main（99f3dbd）的 Linux 编译/链接通过。没有启动引擎，也没有改动任何生产设置。
- 纯上游的 GPU 性能以及会话/视觉/取消回归检查仍有待运行。在那之前，本 PR 保持为 draft。
- 历史合并构建通过了那些安全检查，但这不能替代对这次独立移植的验证。
