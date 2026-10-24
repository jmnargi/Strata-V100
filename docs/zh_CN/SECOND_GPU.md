# CUDA1–3 专家层实验
<a id="cuda13-expert-experiment"></a>

CUDA0 保留稠密层、KV/state、MTP 以及它原有的专家缓存（expert cache）。最多三个相互独立的专家缓存按 CUDA1、CUDA2、
CUDA3 的顺序填满。每个专家层（expert）只有一个属主：CUDA0、某一块次要 GPU，或 CPU 池。次要 GPU 在 CPU 池工作的同时
计算它那部分行；结果通过私有的页锁定（pinned）宿主缓冲区返回，之后 CUDA0 才继续这一层。GPU 的启动可以重叠，
但每一层都要等待全部三方。

随版本发布的配置档对 8,000 个 `(layer, expert)` 组合排序。CUDA0 取它惯常的 `--expert-cache auto` 配额。只有 CUDA1 时，
它取排序中靠前的剩余组合。启用 CUDA2 或 CUDA3 后，剩下的已排序组合按轮询（round robin）方式分配给各块次要 GPU。
随后排序列表会按“先专家、后层”的顺序补入所有其余组合，并以同样方式分配。这样能填满额外的显存（VRAM），并让所有
次要 GPU 都见到高频专家，但未参与排序的专家可能几乎从不会被路由到。启动日志会报告每块卡的槽位数与 GiB；每个请求会
报告各块次要 GPU 上实际计算出的专家条目数。远端输出在经由 PCIe/USB4 返回之前会被打包：请求日志会把传输的 MiB 与
此前整行传输作比较。输入与路由元数据使用页锁定宿主的暂存（staging）。这减少了传输流量与启动开销，但不会释放系统
内存中的专家区，也不会显著改变显存占用。

可选的 `--gpu-placement layer` 把每一层交给恰好一块次要 GPU（层号对启用卡数取模）。它会先填满该卡所负责层中路由
最频繁的专家，再填入未参与排序的那些。当默认的 `stripe` 模式下多块次要卡服务同一层时，这可以减少 USB4 的设备切换
与同步。三块卡仍然各自保留所请求的专家槽位；主 CUDA0 缓存与 CPU 回退照常工作。日志会统计每块卡上活跃的层启动次数，
因此这一权衡可以和每秒 token 数一并测量。这种放置方式是实验性的，如果某一层需要的算力超过一块卡的容量，可能反而更慢。

该实验需要相应的可见 CUDA 设备以及启用了 CUDA 的构建。每块卡必须至少保留 512 MiB 空闲。各层必须按顺序启用。
在 WDDM 下（Windows、WSL2），宿主专家区的 CUDA 注册上限为 8 GiB，以便为 CUDA0 上的上下文与 MTP 留出空间
（`STRATA_ARENA_PIN_GIB` 可覆盖它；在 Linux 上注册整个专家区）。其余部分仍可供 CPU 池使用；PCIe 专家路径只对已注册的
层可用。没有次要 GPU 时，采用原先不设上限的注册行为。

在已有的 Linux 安装上，把补丁应用到它的源码树，然后运行：

```sh
./setup.sh --setup --gpu1-experts 5000 --gpu2-experts 5000 --gpu3-experts 5000
```

在 Windows 上，于安装目录的 PowerShell 中：

```powershell
.\START-HERE.bat --setup --gpu1-experts 5000 --gpu2-experts 5000 --gpu3-experts 5000
```

若想改为测试层放置方式，在 setup 命令后加上 `--gpu-placement layer`。想退回，用同一命令加 `--gpu-placement stripe`。
新引擎构建完成之后，也可以无需重新构建就切换该模式：把 `strata-iq3_xxs.json` 中的 `--expert-cache-remote-placement`
从 `layer` 改为 `stripe`（或反之），然后重启服务器。如果该参数不存在，默认值是 `stripe`。

被问到选择时，选择与你当前安装相同的模型、规模、上下文和图片设置。Setup 会重新编译引擎、更新其服务器配置并启动它。
后续启动会保留这些设置。想在不重新编译的情况下调整槽位数，就编辑已保存的 `strata-iq3_xxs.json` 中
`--expert-cache-device1`、`--expert-cache-device2` 和 `--expert-cache-device3` 后面的数值，然后用
`.\run-iq3_xxs.bat` 重启。

在单卡、双卡和四卡上比较完全相同的请求，最好多做几次重复。只看显存占用并不能说明卸载是否有效：请比较每个请求在
CUDA1–3 上的计数以及每秒 token 数。更多的 GPU 上下文与同步可能拉低速度。提示词的 prefill（预填充）仍使用 CUDA0；
次要 GPU 服务 decode（解码），包括 MTP 校验。不需要点对点（peer-to-peer）访问。

更长的上下文会在 CUDA0 上预留更多 KV/state 内存，从而减少它的自动专家缓存。在 64 GB 的 PC 上，`setup.py` 会把
IQ3_XXS 限制为 128K，即使你选了 262K；次级缓存仍然需要宿主专家区。想在 64 GB 上试 262K，请改选 Q2_0 或 IQ2_XS，
并确认加载后 CUDA0 仍有足够空闲显存。长提示词要与短 decode 请求分开基准测试。

关于 CUDA0 专家缓存 GPU 命中路径的既有警告仍然适用：它的输出会与关闭缓存的运行产生分歧。在生成的 token 得到验证
之前，请把性能视为实验性。

## 可选的助手 decode 优化
<a id="optional-helper-decode-optimization"></a>

`--remote-expert-opt`（仅 `--serve`）优化上文那些 CUDA1-3 助手缓存。引擎默认关闭；自 0.1.39b 起，setup 会在两块或更多
GPU 时把它写入配置（`--gpus`，或启动时的"use both"）。它只在配置了助手缓存时生效；层切分（layer split）完全照旧运行。
想去掉它：用 setup 的 `--no-remote-expert-opt`，或在模型的 `strata-*.json` 里写 `"remote_expert_opt": false`
（再次运行 setup 时会保留）。由该 PR 作者实测：双 RTX 4090 相比普通助手路径混合场景 +63% / 代码场景 decode +132%；
RTX 5090 + 4090 为 +28% / +63%。主缓存会避免接纳助手已经持有的专家，助手则用它们已有的同层槽位把冷专家替换为被频繁
路由的 CPU 未命中项。每个助手在其 GPU 上把专家输出归约为一个加权部分和，然后每 token 返回一个向量。没有 CPU 专家
工作的 token 会跳过 CPU 激活量化。

对于以 CUDA0 为主、CUDA1 为助手的既有服务器配置，在模型与 profile 参数之外使用这些引擎参数：

```text
--expert-cache auto --expert-cache-device1 auto --remote-expert-opt
```

`--expert-cache-device1`、`--expert-cache-device2` 和 `--expert-cache-device3` 现在也接受 `auto`：按分配给该助手的排序，
用实际对齐的专家字节数与空闲显存填满它，并保留既有的 512 MiB 余量。显式的数字预算依然有效。这是启动时的容量规划，
不是吞吐均衡；有多个助手时，某块卡被截断的候选尾部不会重新分配给另一块卡。

该优化使用既有的宿主调度与页锁定宿主传输，不使用 P2P 或张量并行。每一层仍要等待它参与的各助手。它会改变浮点求和
顺序，因此不声称启用后的输出逐位相同。其作者只实测过双卡的 CUDA 运行；有一台社区机器在 HIP 上实测了双卡
（2x RX 6900 XT，每块都是 PCIe 4.0 x8，IQ3_S）：在那台机器上，专家计划的 PCIe 份额一直去取助手已经持有的专家，
这使得主卡每个 verify window 等待 43 ms 而不是 19（decode 为 40 而不是 66 tok/s）。#854 让助手的专家不进入该份额；
在它之前，`--pcie-frac 0` 可以避开这一开销
（[bench/results/2026-10-04-rdna2-helper-pcie-share](../../bench/results/2026-10-04-rdna2-helper-pcie-share/README.md)）。
三/四卡尚未验证。不加该开关时，仍使用既有的 decode 路径。这不会优化下文的独立 `--peer-device` 路径，
也不会改变它与助手缓存之间既有的不兼容性。

## 对等层（`--peer-device`）
<a id="peer-tier---peer-device"></a>

`--peer-device N` 在 CUDA 设备 N 上放置第二个自适应专家缓存。它接管 CUDA0 缓存未持有的那些已排序组合，能装多少装多少。
对等卡计算它自己那些专家的行，既为 decode 窗口也为提示词分块；激活值与结果经由 NVLink 或另一条 P2P 路径往返。
该层会像主缓存一样在服务器运行期间自适应。它是上文 CUDA1-3 缓存的替代方案，而不是与它们并列的第三层。

- `--peer-device N`（默认关闭）：在 CUDA 设备 N 上启用该层（N >= 1）。
- `--peer-reserve-mib M`（默认 600）：在对等卡上留出 M MiB 空闲；缓存取走其余部分。提示词路径的缓冲区需要这段余量。
  在此之外，两块卡上都要再留几百 MiB：提示词路径会在首次使用时加载它的内核和 cuBLAS（一次在对等卡只剩 5 MiB、
  主卡剩 485 MiB 的启动，在第一个提示词处停在 "cuBLAS status 14"）。
- 两块卡是否真能交换数据会在启动时测试一次（#2666）：接在主桥后的两块 RTX A4000 报告支持对等访问，也启用了它，
  但随后 `cudaMemcpyPeerAsync` 什么也没复制，于是对等卡在零值上计算，第一个提示词返回 NaN。现在各方向的测试拷贝
  会做出判定；如果到达的内容不对，日志会说明，且提示词路径使用宿主路由（不走 P2P 的那条：激活值与行经过映射的
  宿主内存）。`STRATA_PEER_NO_P2P=1` 可手工指定该路由。
- `--peer-slots N`（默认 0）：把该层上限设为 N 个专家；0 = 能装多少装多少。
- `--peer-adapt-swaps N`（默认 -1）：对等卡每轮自适应的换入换出次数；-1 使用主缓存的 `--adapt-swaps`。
- `--peer-prefill-rows N`（默认 -1）：每个提示词分块的行中对等卡计算的比例；-1 为分块 x top-k 的一半，
  0 把提示词行全部留在主卡。

经由服务器使用时，在配置的 `"gpu"` 里列出两块卡（例如 `[0, 1]`，编号按 nvidia-smi 的编号），并在其 `"args"` 中加上
`--peer-device 1`；该数字是该卡在列表中的位置（`"gpu": [2, 0]` 时，`--peer-device 1` 指 nvidia-smi 的 0 号卡）。
`"gpu"` 中的多块 GPU 在其他情况下是层切分：只有当 `--peer-device` 不在 args 里时，服务器才添加 `--layer-split`。

`--peer-device` 需要 `--expert-profile` 和一个已启用的专家缓存，且该设备必须可见；否则会拒绝。它也会拒绝
`--layer-split`（另一种双卡模式：二选一）和 `--expert-cache-device1..3`（对等层已经在那块卡上缓存专家）：

    strata generate: --peer-device cannot be combined with --layer-split (use one or the other)
    strata generate: --peer-device cannot be combined with --expert-cache-device1..3 (the peer tier already caches experts there)

不加 `--peer-device` 时二进制不变；其输出与发行版逐字节相同。加上 `--peer-device` 且把同一专家集拆到两块卡上时，
在精确性门（exactness gate）下生成的 token 与单 GPU 运行逐字节相同。

有对等卡时，提示词路径保留 MMQ 路径；融合的 int8 提示词路径尚未与对等卡的行结合。映射的宿主缓冲区仅在有对等卡时
才加上 `cudaHostAllocPortable`，因为只有这样才有第二个上下文写入它们。该层的大小目前需手工设置
（`--peer-reserve-mib`、`--peer-slots`）；小卡上的自动规划需要 #216 的缓冲区借用，属于后续工作。

## 助手缓存配合常驻式 RAM 模式
<a id="the-resident-ram-mode-with-the-helper-caches"></a>

`--resident-experts`（低内存模式）可以与助手缓存一同工作：助手持有的专家层会像拆分阶段的情形一样被排除在 RAM 副本
之外，而助手们的重新填充通过源读取——RAM 副本持有那些字节时读它，否则读文件。本机实测环境为 RX 6800（gfx1030，
16 GB）+ RX 6600（gfx1032，8 GB）、Windows 11、i5-11400F 和 32 GB 内存（单独 6800：约 37.5 decode tok/s，
相同的 worker 配置）：6600 上 3.4-3.8k 专家的缓存，在 3800 槽位下把 8K-token 请求的 decode 提升到约
40.6 / 44.2 tok/s，CPU 池承担的专家份额从约每层-窗口 2.9 个条目降到约 0.4 个。助手的行在池的关键路径上大约每层花费
0.3-0.6 ms——通常低于它们所替代的 CPU 行与页中断，这就是 6600（每行成本约为 6800 的 2-3 倍）作为缓存层仍然划算的原因。

**请按显卡*真实*的空闲显存来给助手定规模。** `--expert-cache-device1 auto` 读取 `cudaMemGetInfo`，而它在 Windows 上
报告的是*进程*的 WDDM 预算：它看不到桌面、视觉编码器（独立进程）、推流宿主或该卡上的其他应用。在一块同时驱动显示器
或编码器的显卡上用 auto 定规模会超额分配，WDDM 会用系统内存（"shared GPU memory"）支撑超出部分，而那部分内存会离开
机器的内存——在一台 32 GB 的 PC 上，它把常驻的补集挤进了页面文件，并使助手通过 PCIe 读取它自己的缓存。显式的数量
（此处为 `--expert-cache-device1 3800`，空卡上约能容纳 4100）能把缓存保持在真实显存里；释放同一块卡上另一个应用扣住的
分配，效果大致又相当于这么多。任何与他人共享显存的显卡，都优先使用显式数量。

## 在 Windows 上选择主设备（`STRATA_PRIMARY_DEVICE`）
<a id="choosing-the-primary-device-on-windows-strata_primary_device"></a>

引擎在运行时编号为 0 的设备上运行模型。Windows 的 HIP 运行时按固定顺序枚举显卡，`HIP_VISIBLE_DEVICES` 只能过滤它——
无法重排——所以在一台较快的卡排在第二位的 PC 上（例如 RX 6800 排在 RX 6600 之后），两张卡都可见时只能把模型放到慢的那块。
`STRATA_PRIMARY_DEVICE=N` 选择另一个可见序号作为主设备；不设置或 0 就是上游的行为。

- 助手缓存（`--expert-cache-device1..3`）按顺序取除主设备以外的可见设备；
- 对等层的跨设备检查使用同一个序号；
- 设置该项时层切分会被拒绝（它后续的阶段必须 > 0，而阶段 0 是主设备）；
- NVIDIA 显卡不需要它：`CUDA_VISIBLE_DEVICES` 配合 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 已经可以重排。

在 RX 6800 + RX 6600、Windows 11 上实测；报告随本改动一并附上。
