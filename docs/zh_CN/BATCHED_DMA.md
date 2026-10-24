# 批量 DMA 提交（实验性）
<a id="batched-dma-submission-experimental"></a>

在 `--pcie-mode dma` 下，`STRATA_DMA_BATCH=1` 会在 CUDA 13.0 及更高版本上通过 `cudaMemcpyBatchAsync` 提交一个
verifier（校验器）组的独立专家上传。默认值为 `0`，即保留现有的逐专家拷贝。常规的内核拷贝模式不变。
这是一次降低宿主提交开销的实验，并不是已证实的模型每秒 token 数提升。

`STRATA_DMA_BATCH=2` 还会设置 CUDA 13.4 的 `cudaMemcpyFlagPreferOverlapWithCompute`。它只是一个调度提示，
并不保证重叠。在 CUDA 13.0–13.3 上，该模式的行为与 `1` 相同。较旧的 CUDA 构建和 HIP 仍保留逐个拷贝。
单次上传或大于 128 的一组也使用逐个拷贝。

源必须是不可变的页锁定宿主内存块（pinned host blob），目标必须互不重叠。`expert_source.cpp`
已经把 DMA 组限制为页锁定的专家。helper 会为已注册的宿主分配解析出设备可见的别名；
Windows 上已注册的宿主指针不必等于其 GPU 别名。不会引入额外的显存或权重转换。
所有拷贝仍留在原有的 stream 上，位于向 flag B 发信号的 callback 之前。
提交出错会让引擎失败，而不是为过期的权重发出就绪信号。

## 测试
<a id="tests"></a>

使用常规的 CUDA 构建并设置 `STRATA_BUILD_TESTS=ON`：

```sh
cmake --build build --target dma_batch_parity
ctest --test-dir build -R dma_batch --output-on-failure
```

该测试在三种模式下检查 0、1、4、17、64、128 和 129 次上传的字节与跨 stream 事件顺序，每次上传分别为
37 字节、4 KiB 和 2 MiB。Windows 还用 `VirtualAlloc` 加 `cudaHostRegister` 跑同样的用例。
它分别打印提交与总传输时间的交错诊断计时。这些时间互相重叠，不能相加。在 Windows 上它还会用
`QueryThreadCycleTime` 报告提交线程的周期数。周期数既不是整个进程的 CPU 利用率，也不是能耗测量。

63 个用例的矩阵在 CUDA 13.4 和 13.0 上用两种宿主分配方法都通过了，在 CUDA 12.6 上则用逐个拷贝的
回退路径通过，三者用的都是同一块 RTX 4070 Laptop 和驱动 616.56。这检验的是较旧工具链的回退路径；
它不是一次 HIP 运行时测试。

## 初次测量
<a id="initial-measurement"></a>

RTX 4070 Laptop 8 GB、Ryzen 9 7940HS、64 GB DDR5-5600、Windows、CUDA 13.4 上：
16 次独立的已注册宿主上传、每次 2 MiB，预热后测量五次重复，交替模式顺序：

| 路径 | 宿主提交时间中位数 | 总传输时间中位数 |
| --- | ---: | ---: |
| 逐个拷贝 | 0.0936 ms | 2.5589 ms |
| 批量 | 0.0162 ms | 2.5248 ms |
| 带重叠提示的批量 | 0.0155 ms | 2.5206 ms |

批量提交的延迟下降了 82.7%；总传输延迟只下降了 1.3%。提交延迟量的是 API 调用前后的墙钟时间，
不是整个进程的 CPU 利用率或能耗。在 Swift 1.5 IQ3_XXS、64K 已分配上下文上做的首次全模型筛查
没有确立生成速度上的提升：一个编码请求用逐个 DMA 测得 30.3 tokens/s，用批量测得 30.0。
两者的生成长度不同，所以总任务时间无法隔离出传输上的变化。这些证据不足以支持改动默认值。

另一次带提交线程周期数测量的 CUDA 13.4 重复给出：逐个拷贝 449,824 个周期，批量 67,939 个周期，约少 84.9%。
提交的墙钟时间为 0.1127 对 0.0170 ms，而总传输时间为 2.5584 对 2.5309 ms（约低 1.1%）。
这些是与初次表格不同的运行，不能合并成一个更大的加速比。

## 在隔离补丁上的模型对比
<a id="model-comparison-on-the-isolated-patch"></a>

一次 A-B-B-A 对比只在 v0.1.39 上使用该补丁，并带 `--pcie-mode dma`。
每个全新引擎都跑同样的九个请求，包括工具调用、提示词复用，以及一个经过独立校验的区间合并实现。
编码请求使用 high reasoning、seed 4242、temperature 1、top-p 0.95、top-k 20 和 3,072 token 的输出上限。
上下文分配为 65,536，INT8 KV，20,480 个常驻 cell，1,040 个缓存专家，7 个 worker 加调用方，
PCIe 比例 0.35，以及 MTP 5、英文起草和 min-p 0.5。

| 运行 | 批量模式 | 代码 decode tok/s | 含推理的输出 token | 请求墙钟时间 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 0 | 29.7 | 928 | 32.98 s |
| B1 | 1 | 31.4 | 1,478 | 48.86 s |
| B2 | 1 | 30.3 | 1,314 | 45.12 s |
| A2 | 0 | 30.2 | 1,078 | 37.45 s |

全部 36 个请求都通过。四个实现都通过了 155 个独立用例，以及生成器、非法输入和不可变性的检查。
即便在两个对照之间，输出也不同，草稿接受率也不同。这些运行并未隔离出模型速度的提升，
也没有显示获得已验证答案的时间更短。字节一致性测试确立的是传输正确性，
而不是确定性的生成文本或通用的 agent 可靠性。

第五次启用批量的运行通过了同样的九个请求，外加一个 57,750 token 的提示词、辅助调用恢复，
以及三次 1,024 见方的图像颜色检查（含图像状态恢复）。一个 OpenAI 兼容的 HTTP 冒烟请求也通过了。
长提示词的读取耗时 78.414 秒；在旁路调用之后恢复 57,745 个 token，把提示词处理降到 0.301 秒。
这验证的是该补丁下现有的状态缓存，而不是新的缓存加速。五次干净补丁的运行通过了 53 个记录在案的请求
加上那次 HTTP 检查。经 NVML 观察到的可用物理显存始终不低于 347 MiB，可用的系统 RAM 始终高于 4.30 GiB。
引擎自己那行独立的 CUDA 空闲显存告警报告为 0，所以 NVML 的数字不能当作已验证的 WDDM 预算来读。
Linux 与 HIP 运行时的验证仍未进行。

## 设计参考
<a id="design-reference"></a>

批量提交的想法沿用了 NVIDIA NCCL 的
[copy-engine 实现](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/src/ce_coll.cc)
以及 [CUDA `cudaMemcpyBatchAsync` 契约](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)。
这里的 helper 和测试是新的；
没有移植任何 NCCL 源码。NCCL 的多 GPU 性能声明没有被当作这条单 GPU 路径的预测。
