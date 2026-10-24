# RX 7900 XTX 支持与性能证据
<a id="rx-7900-xtx-support-and-performance-evidence"></a>

这个选择性启用（opt-in）的 Linux `gfx1100` 后端取代了
[PR #94](https://github.com/Niko1221/Strata/pull/94) 中的初始支持。它保留了 HIP 运行时与
wave32 整数点积兼容性、原生 mmap 布局校验以及 MTP，然后增加了 HIP MMQ、可选的经校准的稠密 hipBLASLt GEMM、
原生 prefill（预填充）批处理，以及实测配置所用的宿主内存 / SSD 路径。HIP 的阻塞式专家层（expert）
上传使用一个可复用的页锁定暂存缓冲区，prompt-cache 的回填也包括在内，以避免反复注册可分页的源内存。

## 复现该配置
<a id="reproduce-the-configuration"></a>

构建步骤在 [AMD_HIP.md](AMD_HIP.md) 里。使用仓库固定的 llama.cpp 依赖；不要悄悄替换成其它修订版。构建时打开
`STRATA_PREFILL_MMQ=ON`，并在实测的那一组（arm）里使用这些运行时变量：

```sh
export STRATA_PREFILL_MMQ=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1100-hipblaslt-100100.txt"
export STRATA_PREFILL_RING=96
export STRATA_IO_THREADS=32
```

随附的调优表是针对 gfx1100 和 hipBLASLt 版本 100100 校准的。
它不是通用的 ROCm 调优表。运行时会检查架构、
库版本以及实际的 shape/stride/workspace 需求，在表中条目缺失或不兼容时回退。HIP MMQ 在运行时选择性启用。默认的 CUDA 选择保持不变。

实测的引擎配置：Orca Flash Next IQ3_XXS，原生模型包加上配套的
GGUF/tokenizer/template，`--mmap-experts --resident-cpu-experts`，固定的排序
`--expert-profile`，`--expert-cache auto`，`--prefill 8192`，`--spec 4`，
`--spec-min-p 0.5`，配套的 MTP 运行时，`--max-context 262144`，`--kv int8`，
`--kv-resident 32768`，`--pool-workers 15`，`--adapt-every 0`，`--pcie-frac 0`，
以及 `--vram-reserve-mib 1024`。PLE 留在 SSD 上。

`--resident-cpu-experts` 把静态 GPU 缓存的补集复制到普通
RAM。它需要 mmap 和固定的 expert profile。prompt 路径可能借用的槽位
也在下面预算允许的范围内把其专家层留在 RAM 中
（其余槽位在被借用期间以及回填时使用映射回退），自适应交换
（`--adapt-every`）在显存与 RAM 副本之间交换专家层，而不读取
文件。该副本是可分页的；`--resident-experts`（CUDA，由 setup 选择）
会把它页锁定，并使用 4 GiB 的余量。分配需要足够的可用 RAM；
在 Linux 上该选项要求有可读的标准 cgroup-v2 挂载。对每个有限的 cgroup 祖先，
守卫只计入扣除 `file_dirty` 和 `file_writeback` 之后的 `inactive_file`，并以当前用量为上界；
它同时受祖先上限和宿主 `MemAvailable` 约束，保留 8 GiB 余量。这样
可计入可回收的干净文件缓存，但无法针对并发的系统或进程分配预留内存。
POSIX 的 PLE 路径通过一个可配置的 worker 池发起直接读取。

使用一台专用的空闲服务器，在每组测试之间重启它，并记录它的引擎日志：

```sh
python3 tools/hip/bench_prefill.py \
  --model MODEL_NAME --url http://127.0.0.1:8080 \
  --engine-log /path/to/engine.log --label candidate --output candidate.json
```

该脚本使用与被报告测量相同的合成源提示词和请求顺序：
一次小规模预热，然后依次是 140/280/140/280 个函数，每个全新提示词之后各跟一个后续请求。它
强制全新提示词的复用 token 为零，并记录 prefill、decode、wall time 和停止原因。它要求
无缓冲日志中每个请求恰好包含一条已完成的计时记录。128 token 的输出上限
是为吞吐测量刻意设置的，与任务是否成功无关。
绝不要把被取消请求的计时行当作吞吐证据。

## 最终修订版基准
<a id="final-revision-benchmark"></a>

测量于 2026-09-29，源码修订版 `9568e78da663e6b224ab96d57baf450717b36e73`。
HIP 可执行文件的 SHA-256：`705f5925a788bc227bad72abfb5466032e0879d752fea9afc3eb32051b35182b`。
下面的对照组（control）和候选组（candidate）都是本次会话重新测量的；表中
没有混入历史基准数值。对照组二进制的 SHA-256
是 `4522d4937ca4a41ca294d31b60df8c2bc4df5b7a6caf65eaad10c036e08491f7`；
它的源码是本地已有的运行时快照，而不是干净的上游提交。
两组都使用一块 RX 7900 XTX 24 GiB / gfx1100、Ryzen 9 7950X3D、
64 GiB 已安装内存、Fedora 系 Linux、ROCm 7.1 和 272 W 的 GPU 功耗上限。
测量期间停止其它模型服务，之后恢复。
采样：temperature 0、top-k 1、top-p 1、min-p 0、seed 42，关闭推理。

| 全新请求，执行顺序 | 对照组 prefill t/s | 候选组 prefill t/s | 候选组输出 t/s | 候选组请求 wall time |
| --- | ---: | ---: | ---: | ---: |
| 4,210 tokens，首次使用 | 240.0 | 447.5 | 55.5 | 11.729 s |
| 8,830 tokens，该尺寸下首次使用 | 484.8 | 873.7 | 55.7 | 12.420 s |
| 4,210 tokens，已预热 | 457.2 | 901.1 | 56.7 | 6.937 s |
| 8,830 tokens，已预热 | 478.9 | 966.5 | 59.2 | 11.314 s |

每个请求都恰好生成 128 个 token，并在刻意的长度上限处结束。
四个全新提示词的复用 KV token 均为零。全新提示词的 prefill 是本次新测的现有 AMD 运行时的
1.80–2.02x；全新请求的 wall time 低 40–46%。这是与现有 HIP
运行时的比较，而不是与未经修改的上游比较——上游并不提供这个后端。候选组还在
2.95–3.26 秒内完成了四个带缓存的后续请求。那些后续请求不是同类可比的
prefill 对比：缓存复用情况随生成的文本和 checkpoint
选择而变化。完整的脱敏测量数据在
[本次运行的 JSON](../benchmarks/2026-09-29-gfx1100.json) 中。

在这个修订版之前，一个缺少 HIP 上传暂存缓冲区的中间构建在 8,830 token 的请求上卡住并被判为失败。恢复暂存缓冲区
之后，上面整段序列才得以完成。那次失败的尝试没有
计入吞吐表。为得到这些结果，没有调高任何看门狗（watchdog）上限。

KV 零复用不等同于冷的文件系统缓存。首次使用的速度
和预热后的速度是分开报告的。单个观测值既不能
给出置信区间，也不能代表每种上下文长度下的普遍速率。

## 正确性与质量限制
<a id="correctness-and-quality-limits"></a>

从实测源码修订版构建出的 HIP 和 CUDA 可执行文件都成功。本次运行对 CUDA 只做了
编译检查，没有做性能测试。启用调优表后，选定的 HIP CTest 套件
通过 **29/29**，其中包括专家层上传回读、
异步交接、QSA、MMQ、Lt GEMM、KV 流式读取和 PLE 读取。被排除的
`ple_parity` 需要外部 fixture；`platform_memory_test` 需要
比测试账户允许的更大的锁定内存限额。另外，三次真实的 IQ3_XXS PLE 矩阵图重放
通过，POSIX 直接文件测试通过了
排队读取、短 EOF、唤醒、关闭/排空和重新打开等检查。

开发阶段的检查通过了八项 MMQ 数值对比、四项实际 Lt GEMM
对比，以及原生 QSA/indexer/embedding 一致性（包括尾部状态、
chunk 续接和图像覆盖）。以原始 FP32/反量化权重为参照，真实量化 MMQ 的相对 L2 误差
约为 0.0026，这反映了
Q8 激活值的算术；它不是按位数值等价。

端到端的编码质量不能由这些数值检查或有长度上限的
吞吐请求来证明。请用独立的运行时测试验证完成的任务；这项
贡献不对回答质量或 agent 可靠性作任何宽泛的声明。

这些改动不声称模型推理更好、全上下文行为已经验证、
视觉端到端验证、Windows 上的 HIP、其它 AMD 架构，也不声称
AMD/NVIDIA 混合执行。已有的 CUDA 多 GPU 代码仍然存在，但它并不是对
HIP 多 GPU 支持的验证。

## 归属与被否决的实验
<a id="attribution-and-rejected-experiments"></a>

上述测量是在这个后端 rebase 到引擎 0.1.24 之前进行的。PR 自带的批处理
embedding gather 和 QSA indexer append（改编自 [PR #108](https://github.com/Niko1221/Strata/pull/108)，commit
`acd487233c0bbe2217a6881c5bb43f8a283b0de5`）在 rebase 中被丢弃：0.1.24 已经在每个
后端上按位精确地完成这两件事（C-2、C-4）。#108 中可选的 GDN 并行/切分路径没有包含在内。现有的上游 MMQ 编排
被保留，并在具备 AMD 架构识别和正确后端编译的情况下为
HIP 启用。

在评估的工作负载中，静态 16K chunk、扩展的 FP16 expert 调优和 192 槽位的 ring 都没有带来
一致的端到端收益。表中只包含选中的 30 行稠密形状。仅靠
microkernel 加速不足以选定一种配置。没有提高 GPU 功耗上限。
