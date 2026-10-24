# 用 prefill 计算重叠 KV 的流式上传
<a id="overlap-streamed-kv-uploads-with-prefill-computation"></a>

## 为什么有帮助
<a id="why-this-helps"></a>

当长上下文的 KV 有一部分放在系统 RAM 中时，prefill（预填充）会上传每个注意力层缓存的前缀。
在计算 stream 上等待每一次上传会让 GPU 空转。

这一可选（opt-in）改动会在单独的 stream 上启动下一个注意力层的上传，同时当前层继续计算。
它复用现有的暂存池；不会再分配一块 KV 缓冲区。

## 使用与顺序
<a id="use-and-ordering"></a>

在启动引擎前设置 `STRATA_KV_PREFETCH=1`。它默认关闭。
一个 release 事件保护暂存池最后的读取者。一个 ready 事件为下一个注意力消费者排序。
正常完成会在 stage-complete 钩子能启动专家槽位补充之前排空待处理的拷贝。
提前返回同样会在调用方可以补充借出的槽位之前排空拷贝。重新布局与销毁会排空该 stream。

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

ABBA 对比，每组两次观测，全部 20 项完整 JSON 检查都通过：

| 冷输入 | 串行 prefill | 重叠的 prefill | 串行请求 | 重叠的请求 |
| --- | ---: | ---: | ---: | ---: |
| 8K | 5.013 s | 5.025 s | 6.727 s | 6.726 s |
| 32K | 16.989 s | 16.082 s | 18.468 s | 17.559 s |
| 64K | 32.604 s | 31.259 s | 34.354 s | 32.775 s |

prefill 时间在 32K 上下降了 5.3%，在 64K 上下降了 4.1%。
8K 上没有有用的改善，decode（解码）也没有加速。
剖析显示约 939 ms / 1,553 ms 的串行 KV 阶段等待离开了关键路径。拷贝仍然会执行；它们并没有被消除。

## 验证与剩余工作
<a id="validation-and-remaining-work"></a>

- 针对纯官方 main（99f3dbd）的 Linux 编译/链接通过。没有启动引擎，也没有改动任何生产设置。
- 尚未声称有纯上游的 GPU 加速或逐位数值一致性。
- 独立的历史原型没有运行完整的视觉/120K/取消安全套件。后来的合并构建跑了，但那不能验证这次移植。
- 在纯上游的长上下文、取消和图像历史测试通过之前保持 draft。HIP 和其他 GPU 未验证。
