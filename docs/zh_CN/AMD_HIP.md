# AMD Radeon：HIP 后端（gfx1100、gfx1101、gfx1200、gfx1201、gfx1030）
<a id="amd-radeon-the-hip-backend-gfx1100-gfx1101-gfx1200-gfx1201-gfx1030"></a>

Strata 通过它的 HIP 后端在 AMD Radeon 显卡上运行，也就是在 NVIDIA 上使用的那个引擎为 AMD 编译的版本。本页讲在
Linux 上的构建（在 Windows 上是现成引擎，见 [Windows](#windows)），针对 RX 7900 XT / XTX（RDNA3，gfx1100）
以及 RX 9070 / 9070 XT / Radeon AI PRO R9700（RDNA4，gfx1201；见 [RDNA4](#rdna4-gfx1201)）。RX 7800 XT /
7700 XT（gfx1101）与 RX 9060 XT（gfx1200）由其持有者验证（见[社区验证过的显卡](#community-validated-cards)）；
RX 6800 / 6900 系列（RDNA2，gfx1030）也能构建并运行，由一台社区机器报告、尚未由维护者们验证（见
[RDNA2](#rdna2-gfx1030)）。在一台没有 Strata 可用 NVIDIA 显卡的 PC 上，setup 会自行选择该后端（两种卡都有时
用 `--backend hip`）；面向用户的安装步骤在 [INSTALL.md](INSTALL.md#amd-cards)。gfx906（Instinct MI50 / MI60、
Radeon VII；wave64）有一个独立的可选开启构建，见 [gfx906](#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source)。
Ryzen AI Max "Strix Halo" APU（gfx1151，RDNA3.5，统一内存）在 Linux 上从源码构建：见
[STRIX_HALO.md](STRIX_HALO.md)。其他 AMD 架构，以及在一次运行中混合执行 AMD/NVIDIA，都不受支持。

该后端把 CUDA 形状的运行时与 BLAS 调用映射到 HIP/hipBLAS，为量化内核使用 RDNA2/RDNA3/RDNA4 的带符号整数
dot 指令，并提供 wave32 的 shuffle / 打包字节运算。仅 CUDA 的 QSA 矩阵指令有一个有序的 FP32 回退。Prefill
同时支持"反量化 + hipBLAS GEMM"与可选开启的 HIP ggml MMQ。一条可选的、经过标定的 hipBLASLt 路径会加速稠密
投影（按架构的表在 `tools/hip`）。这并不宣称跨后端的模型回答逐位相同。见
[性能设置与证据](AMD_HIP_PERFORMANCE.md)。

## 用 setup 安装（推荐）
<a id="install-with-setup-recommended"></a>

在使用内核自带 amdgpu 驱动（不需要安装 ROCm）的 Linux 上，配 RX 7900 XT / XTX、RX 7800 XT / 7700 XT、
RX 9060 XT、RX 9070 / 9070 XT 或 Radeon AI PRO R9700：

```sh
./setup.sh --backend hip
```

- **检测：** setup 通过内核的 KFD 拓扑找到该显卡。集成 Radeon 显卡会被列为不受支持。在一台没有 Strata 可用
  NVIDIA 显卡的 PC 上，`--backend hip` 会被自动选定。
- **ROCm：** 若 `/opt/rocm`（或 `$ROCM_PATH`）中存在带 hipcc 与 hipBLAS 的系统 ROCm 7，就会使用它。否则
  （或者它早于 7.0 时），ROCm 会从 AMD 的 TheRock wheel 装进 `.venv`（约 10 GB，无需 sudo），取自该显卡
  系列的索引：gfx1100 / gfx1101 用 `gfx110X-dgpu`，gfx1200 / gfx1201 用 `gfx120X-all`，gfx1030 用
  `gfx103X-all`，Strix Halo 用 `gfx1151`。版本按系列选定：该索引上有 7.10.0a20251120 时用它，
  `gfx103X-all` 用 7.13.0a20260515（它的索引从 7.13 起，#1103），`gfx1151` 用 7.14.0a20260608（7.10 的 wheel
  在内核 7.2.8 上会段错误，#1267）。当索引不再提供该版本时，setup 会警告并取同一 7.x 系列中最新的那个，
  否则取同一主版本中最新的。`STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` 会覆盖这些选择；gfx1030 的 wheel
  未经检验（那里经过检验的路径是系统 ROCm 7）。
- **引擎：** 在你的 PC 上为该显卡的架构编译（10-20 分钟，一次性；在一次改变了它的 `git pull` 之后，
  或者当你改选另一种架构的显卡时会再来一次）。这需要一个 C++ 编译器和 git
  （`sudo apt install build-essential git`）。
- **hipBLASLt 调优表：** setup 只在 `tools/hip/<arch>-hipblaslt-<version>.txt` 同时匹配该显卡的架构与已安装的
  hipBLASLt 版本（从 `hipblaslt-version.h` 读取；1.2.0 是 `100200`）时才使用它。否则会说明这一点，而提示词的
  稠密矩阵乘法使用普通 hipBLAS（提示词更慢，答案相同）。表中的 solution id 只对该组合有效，而引擎会拒绝任何
  其他表。
- **多块卡：** setup 取一块卡（显存最多的那块，或 `--gpu N`），除非你指名多块：
  `./setup.sh --backend hip --gpus 1,0` 把模型的层切分到它们上，第一块是主卡（编号按 setup 列出的顺序；
  `--gpus all` = 每块受支持的卡，显存最多的在前）。每块被选中的卡都必须是上述架构之一；引擎会为其中每一种
  编译（来自两个系列的卡，例如 gfx1100 + gfx1201，需要系统 ROCm 7：AMD 的 wheel 只含一个系列）。只有当没有
  单块卡能装下模型的专家层时，切分才划得来（见下文 RDNA4）。
- **目前的限制：** 图片只能经 CPU 编码器（`--vision cpu`，0.1.32）。setup 尚未在 AMD 上提供调优
  （校准）：它的各项控制正在 HIP 上逐个核对（#566）。自 0.1.39 起，手工运行的一次调优
  （`./setup.sh --calibrate`）会为它运行时所在的那块 AMD 卡保存下来，并在 setup 再次运行时复用。Monitor
  从 Linux sysfs 显示该卡的负载、显存、温度与功耗（0.1.32）。

setup 的其余部分与在 NVIDIA 上相同：模型下载、启动脚本、服务端。

## Windows
<a id="windows"></a>

自 0.1.34 起，Windows 上的一块 AMD 卡像 NVIDIA 卡那样安装：下载 Strata，双击 `START-HERE.bat`。在一台没有
Strata 可用 NVIDIA 显卡的 PC 上，AMD 卡会被自行选中；两种卡都有时 setup 会询问
（`START-HERE.bat --backend hip` 直接选 AMD）。

- **你需要：** Windows 10 或 11（64 位）、上文所列的一块卡，以及当前的 AMD 驱动（[AMD Software:
  Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html)）。此外什么都不需要：不装 ROCm 或
  HIP SDK，不需要编译器，不需要管理员权限。
- **检测：** setup 读取 Windows 列出的显示适配器（它们的 PCI id；显存大小取自显示驱动在注册表中的条目）。
  集成 Radeon 会被列为不受支持，除了 Ryzen AI Max "Strix Halo"（Radeon 8060S / 8050S / 8040S，gfx1151：
  [STRIX_HALO.md](STRIX_HALO.md)）；Strix Point（890M / 880M，gfx1150）、Krackan（860M / 840M，gfx1152）或
  Phoenix / Hawk Point（780M / 760M，gfx1103）的 Radeon 会被指名说出它是什么，且不受支持。
- **引擎：** 发行版里的现成包 `strata-windows-x64-hip.zip`（由 `tools\hip\build_windows.bat` 为 gfx1100、
  gfx1101、gfx1102、gfx1200、gfx1201、gfx1030 以及自 0.1.40 起的 gfx1151（Strix Halo，在 Windows 上未验证）
  构建）放入 `engine\`。它带有引擎要加载的 ROCm 库（`engine\rocm\bin`：HIP 运行时、hipBLAS / rocBLAS /
  hipBLASLt 及其针对这些显卡的内核、amd_comgr 与 Microsoft C++ 运行时；来自 AMD 的 TheRock 构建的
  ROCm 10.2.0a20260930，许可在 `engine\rocm\licenses`）。HIP 运行时通过 AMD 驱动自身的组件工作，所以驱动是
  它唯一需要这台 PC 提供的东西。
- **`strata.exe` 旁边的 HIP 运行时（0.1.35，#468 #461）：** `amdhip64_7.dll` 与 `amd_comgr.dll` 也会被放进
  `engine\`（setup 在每次启动时把它们复制到那里）。Windows 会先在程序所在文件夹中查找，之后才找 System32，
  而某些 AMD 驱动会把它们自己的 `amdhip64_7.dll` 装到 System32；用那一个时，随包附带的库会在第一个提示词上
  崩溃（一次访问违例，或 `hipErrorInvalidDeviceFunction`）。引擎的日志会指出它加载的运行时
  （`strata generate: HIP runtime ...`）。
- **在约 60 GB 的模型下载之前**，setup 会运行 `engine\strata-device.exe --list-devices`（把 `engine\rocm\bin`
  加入 PATH）：如果 HIP 运行时看不到该显卡，setup 会停在那里并指向驱动。它也会给出该显卡的 HIP 编号：
  有集成 Radeon 时那是设备 1，不是 0（#325）。此后 setup 都按 HIP 的编号列出 AMD 显卡，所以 `--gpu N` 与配置
  里的 `"gpu"` 都是 HIP 编号。
- **与 Windows-on-NVIDIA 和 Linux-on-AMD 的不同：** 尚不支持图片（CPU 图片编码器目前仅限 Linux）、
  每个模型一块卡（`--gpus` 目前仅限 Linux）、没有校准。
- 两处仅 Windows 的引擎细节（#247、#325）：hipBLAS 在某些 BF16/FP16 GEMM 之后可能返回成功却把
  `hipErrorInvalidValue` 留在置位状态（在 gfx1201 上见到）；引擎只在一个成功的 GEMM 之后清掉这一处陈旧的错误，
  且仅限 Windows。`hipHostGetDevicePointer` 在 Windows 上返回宿主指针本身：内核经由它读取映射的内存是正确的，
  但一次设备到设备的拷入不会落到那里，这就是 `tests/hip/handoff` 在那里超时的原因（引擎不使用那次拷贝；
  `tests/hip/mapped_alias` 会报告它）。
- 还有两处（#380、#377，由 BlueKingMuch 在一块同时驱动桌面的 RX 6800 上实测）：Windows 上的
  `hipMemGetInfo` 不会减去桌面与其他程序在该卡上占住的部分，于是 `--expert-cache auto` 把该卡填满到超过 Windows
  留在显存中的部分，decode 从 41 掉到 30 tok/s。引擎现在会把那个空闲数字按 Windows 为该进程保留的视频内存预算
  调低（只记录一次：`strata: Windows budgets N of this card's M MiB ...`；`STRATA_WDDM_BUDGET=0` 关闭它）。
  以及 PCIe 探测在那里用宿主时钟为它的拷贝计时，因为 HIP 的事件读出不可能的速度（3,300-26,000 GB/s），
  所以一条慢链路现在会像在 NVIDIA 上那样拿到更小的 `pcie_frac`。

**已验证的内容（0.1.34）：** #325 的作者在一块 RX 9070 XT（Windows 11、`.venv` 中的 ROCm
10.2.0a20260930，在该 PC 上编译）上运行了这个移植版的引擎：Coder IQ1_M，32K，decode 29.2 tok/s，
prefill 约 181 tok/s，回答正确；ctest 46 个中 42 个。维护者们没有 Windows 上的 AMD 显卡：发行版的 zip 是在
一台 NVIDIA PC 上构建的，并在那里在 Ryzen CPU 的集成 Radeon（gfx1036，同一代码树的一个测试构建）上做了检查：
PATH 上只有该 zip 的库时，`strata-device` 能列出该显卡，一次 hipBLAS BF16 GEMM 与 CPU 的结果一致；HIP 的
ctest 通过 56 个中的 52 个，4 个失败项与 RX 9070 XT 上相同（`hip_handoff`，以及三个需要模型包 fixture 的测试）。
现成的 zip 本身还没有在一块独立显卡上跑过模型——欢迎报告。

**报告一次 Windows 上的 AMD 运行**（开一个 issue，或在 #325 下）：你的显卡与驱动版本（AMD Software > System），
然后

```bat
engine\strata-device.exe --list-devices
engine\strata-device.exe --selftest
```

（在 Strata 文件夹中，先 `set PATH=%CD%\engine\rocm\bin;%PATH%`），`strata-<model>.log` 的结尾，以及服务端
窗口在第一个回答时打印的速度行。

**自己构建：** `tools\hip\build_windows.bat`（带 C++ 工作负载的 Visual Studio 2022 Build Tools、Python、
git；不需要管理员权限，不需要 AMD GPU）会把 ROCm 从 AMD 的 TheRock wheel 装进 `.rocm-win`，构建并打包出
`dist\strata-windows-x64-hip.zip`；`START-HERE.bat --backend hip --prebuilt dist\` 会安装那一个。
`tools\hip\build_windows.bat tests` 还会构建 HIP 测试（`build-hip-win` 中的 `ctest`，
PATH 上要有 `.rocm-win\Lib\site-packages\_rocm_sdk_devel\bin`）。`STRATA_HIP_ARCHS` 可以选择其他架构。

## 构建
<a id="build"></a>

要求：一个可用的 ROCm 驱动/运行时、HIP 开发头文件与编译器、hipBLAS，以及（为调优过的稠密 prefill）
hipBLASLt 开发文件、CMake 3.24+、一个 C++20 宿主编译器和 Git。模型仍需要足够的系统内存以及快速的 SSD 存储
来放它的 PLE 表。仅凭显存占用并不是吞吐测量。

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` 是 `gfx1100`、`gfx1101`、`gfx1200`、`gfx1201`，或像 `"gfx1100;gfx1201"` 这样的列表
（一个二进制同时覆盖两者）。gfx1102（同样的 wave32、64 KiB LDS 与 dot4 指令）构建时会有警告：它通过了 ctest
（#192），且一块 RX 7600 XT（16 GB，i7-13700K，64 GB）在最多 51K 上下文下的 146 次请求中运行 Swift IQ3_XXS，
decode 约 43 tok/s（28-50）、提示词 250-300 tok/s（#942）；setup 接受它（未验证，#938）；gfx1030 也一样
（RDNA2：较旧的 `v_dot4_i32_i8`，#311 中的一次社区运行）。在启动时，引擎与 `strata-device` 会把它们使用的每块
GPU（`gcnArchName` 取到 `:` 特性后缀为止）与该二进制编译时指定的架构作比较，并要求 wave32。被拿到另一块卡上的
二进制会带着该卡的名字、它的架构以及构建的列表停止，而不是稍后以 "invalid device function" 失败。

如果 CMake 找不到 HIP 编译器，请加上 `-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`。在 Fedora 系的测试
主机上这是 `/usr/lib64/rocm/llvm/bin/clang++`。请使用同一次 ROCm 安装中的编译器与库。不要在一个构建中同时启用
两个 GPU 后端。

原生 IQ 专家层默认启用。CMake 会取得本仓库所固定的 llama.cpp 修订。离线构建时，把 `STRATA_GGML_DIR` 指向
该确切修订的一份检出；使用任意的较新检出会改变被测试的那个依赖项。

## 模型与服务端配置
<a id="model-and-serving-configuration"></a>

用现有的工具准备一个受支持的模型包及其 MTP 运行时。对于 OrcaRouter IQ3_XXS，请按
[显式的兼容转换](ORCA.md)来做；本后端不改变量化、模型许可或分词器。在 `tools/iq_pack.py` 的命令里加上
`--experts-bin`：mmap 需要模型包的 `experts.bin`，而默认的打包命令并不产出它。这会额外占用磁盘空间。
原版 GSQ-RCO IQ3_XXS 用分片 2 放 PLE；Orca 用分片 1。请把每个模型自己的模型包与分词器放在一起。

在服务端 JSON 中把 `build-hip/strata` 作为可执行文件。必要时用 `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`
选择 AMD 设备。先用适中的上下文与 prefill 分块大小启动，再去测量更大的负载。请把 PLE 文件留在 SSD 上。
不要假定配置好的上下文长度就能证明整窗推理成功。

即使普通内存还有余量，ROCm 上大的页锁定宿主分配仍可能失败。现有的 `--mmap-experts` 路径避免了分配整块页锁定的
专家 arena；它仍依赖操作系统的文件缓存驻留，并可能在存储读取上停顿。它并不会让 SSD 访问等同于内存。

原版 GSQ-RCO IQ3_XXS 模型的启动参数（请替换其中的路径）：

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` 是引擎的内部 token 协议。要给浏览器或 OpenAI API 使用，请把这些参数放进
[服务端 JSON 示例](ORCA.md#local-server)的 `args` 数组，把 `exe` 设为 `build-hip/strata`，并使用对应模型包的
`tokenizer` 目录。用 `.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080`
启动。上面的 worker 数量是在一颗 16 核 CPU 上使用的；请为你的 CPU 实测。4K 上下文是一个冒烟测试的起点，
不是模型的上限。专家缓存会自行确定大小，并留出 1 GiB 的显存余量。

**该显卡同时驱动一个 Linux 桌面（#560 #516）：** 请留出比默认 700 MiB 更多的显存，例如
`./setup.sh --vram-reserve-mib 3072`。当缓存填满该卡时，桌面下一次需要显存会让 amdgpu 把 GPU 内存搬到系统内存
（GTT），随后 OOM killer 会结束 KWin/plasmashell，或 `systemd-oomd` 结束应用，或者合成器以
"Failed to pin framebuffer with error -12" 失败。

安装程序支持本后端（见上文"用 setup 安装"）。图片目前通过 CPU 编码器运行（`--vision cpu`）。
Setup 安装一块 AMD 卡，或用 `--gpus` 装多块（引擎的层切分；见下文 RDNA4）。

**针对 RDNA2 提示词与挂起的可选开启开关（0.1.40，默认关闭）：** `STRATA_DENSE_MMQ=1`（#820，HIP）让提示词的
稠密 GGUF 投影走 int8 MMQ 内核而不是反量化 + hipBLAS：在 RX 6900 XT 上一个 4K 提示词读取从 436 变为
757 tok/s（贪心文本在几十个 token 之后就会与默认值分岔：q8_1 激活值的舍入）。
`STRATA_HIP_ADAPT_KERNEL_COPY=1`（#884）用一个内核而不是 SDMA 引擎来复制自适应层的换入换出，这是对 gfx1030
上配合 MMQ 提示词路径与自适应换入换出时所见挂起的一个规避（在报告者的机器上未测试）。Windows HIP：门铃内核会为
它的存储加栅栏（#697），而共享专家的流分叉在 HIP 上是关闭的（#816）。

**共享专家的流分叉（`STRATA_SH_STREAM`）：** 它在 HIP 上默认关闭，gfx1151 的默认值表除外，而那个默认值保持不变。
各处实测因平台而异、彼此矛盾。Linux 上的 2x RX 6900 XT（#1272）：关掉它会让 decode 损失 5-13%
（单块卡 47.5 / 46.9 / 48.3 -> 开启时 50.5 / 48.5 / 51.3 tok/s）。Windows 上的 RX 6800（#816，
`bench/results/2026-10-04-community-rx-6800-windows`）：开启时慢 12-22%。4x R9700：关闭分叉快 13-45%。
Linux 上的 RX 6000 用户可以试试 `STRATA_SH_STREAM=1`；两种情况下输出完全相同。

**大显存卡上的长提示词：检查借出覆盖那一行（#1389，由报告者在 RX 7900 XTX、gfx1100、v0.1.40.2、50 GB 内存上
实测）。** 使用 `--resident-experts` 时，提示词路径会借用缓存槽位，而那些槽位的专家层也必须同时位于内存中。
在每次启动后的第一个请求之后，引擎会打印
`FileExpertSource: N of M of the prompt path's lendable slots keep their experts in RAM too`。当 N 不等于 M 时，
未被覆盖的那些专家层会在提示词期间从模型包读取（现在也会额外打印一行 `WARNING`）。一个 74K 的提示词会路由到
几乎全部专家，所以这些未命中会不断重复：报告者在 6201 of 6201 时看到 `wait copy` 占 0.2%、2,583 tok/s，
而在 5093 of 6202 时是 36.8% 与 1,810 tok/s（同一个二进制，只是启动不同）。短提示词（25K）两种情况下读取速度
相同。常驻集的大小在启动时由 `MemAvailable` 减去 `STRATA_RESIDENT_HEADROOM_GIB`（默认 4，单位 GiB；
`--resident-experts` 开关与 `--resident-budget-gib` 都会读取它）决定。如果这一行显示的覆盖不足，请把余量调低
一点并重启；如果之后机器开始换页或卡顿，再把它调高。在一次 ROCm 更新之后请重新检查这一行：报告者较新的运行时
多留下了约 1 GiB 宿主内存，这把最佳取值从 8 移到了 7。同一份报告中的其他可调项（他的数字，此处未重新实测）：
`STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096` 使 925 -> 1,092 tok/s，
`--resident-experts` 使 1,092 -> 2,503 tok/s，`--prefill auto:32768` 在 74K 上 +23%。引擎只在 hipBLASLt 调优表的
版本与已安装库匹配时使用它；否则它用普通 hipBLAS 做 prefill（一个 131K 提示词约慢 45%），所以请让表与库保持配对。
不值得尝试（实测持平）：改动 `--kv-resident`、超过 32768 的 prefill 分块，以及超过 4 的 `--spec`。

## Linux：内核回收宿主内存期间的超时验证（实验性规避方法）
<a id="linux-verify-timeouts-while-the-kernel-reclaims-host-memory-experimental-workarounds"></a>

一条 `verify: timed out at layer N` 或 "no progress for 60 s" 消息本身并不意味着内核或握手有 bug。两份社区
报告发现了同一个机制：当内核回收 GPU 通过 KFD userptr 所锁定的宿主页时，GPU 的各队列会被挂起，而一次反复返回
`-EAGAIN` 的恢复会让它们挂起几十秒。每份报告都是一台机器、一种规避方法，且原因在两者上都未被确认；它们都不是
默认值，也不是普遍的速度主张，且都不知道在 Windows 上或其他 ROCm 版本上是否重要。

- **可分页的宿主分配（#750，两块 Radeon AI PRO R9700，ROCm 7.2）：** 追踪把一次超时与两块卡上各约 31 s 的
  USERPTR 队列挂起关联了起来。除非设置 `HSA_USERPTR_FOR_PAGED_MEM=0`，ROCr 会对可分页的宿主分配使用 USERPTR。
  把它放进服务端 JSON 的 `env` 后，五次配对运行给出相同的输出与相同的中位数（7 个请求：16.65 s 对 16.63 s），
  且没有那次 29.8 s 与 42.2 s 的离群值；单独请求略慢（3.55 s 到 3.69 s）。
- **`--mmap-experts`（#920，RX 6800 gfx1030、ROCm 7.2.4、31 GiB 内存、IQ3_XXS）：** 每次运行都会停顿，直到
  `env` 中有了 `GPU_PINNED_MIN_XFER_SIZE=1048576`，此后再也没有过。HIP 会锁定大拷贝的可分页源页，这里就是那个
  映射的专家文件。（引擎为 `STRATA_ARENA_MMAP=1` 本来就设置这个变量。）

```json
"env": { "GPU_PINNED_MIN_XFER_SIZE": "1048576" }
```

请保持配置其余部分不变、一次只改一个设置，重启引擎以便运行时读取它，并把它移除以回到默认值。在大上下文测试前
检查空闲内存与驱动的 GTT 上限。如果引擎运行在一个 systemd 单元里，请用 `MemoryMax` 限制它，绝不要用
`MemoryHigh`：`MemoryHigh` 会把 `--mmap-experts` 读取所依赖的页缓存计入其内，而一个 16K 提示词在它之下停在
`pread` 里 8 分钟。在报告中请给出运行时与内核版本、内存回收是否与 KFD 队列驱逐同时发生，以及配对的时间数据。

## RDNA4（gfx1201）
<a id="rdna4-gfx1201"></a>

RX 9070 / 9070 XT 与 Radeon AI PRO R9700 运行与 gfx1100 相同的内核：wave32、每个工作群组 64 KiB 的 LDS、
带符号的 dot4 指令（经 `__builtin_amdgcn_sudot4` 的 `v_dot4_i32_iu8`），以及 100 MHz 的墙钟。第一份报告与
补丁来自 doplxyz（#178）。2026-09-30 在 doplxyz 的测试机上验证：一块 RX 9070 XT 16 GB 与一块 Radeon AI PRO
R9700 32 GB（都是 gfx1201）、Ryzen 9 3900X（16 线程，无 AVX-512）、47 GB 内存、KVM/VFIO 客户机中的
Ubuntu 24.04。

- **构建：** 带测试的完整 HIP 构建（`-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON
  -DSTRATA_PREFILL_MMQ=ON`），分别在系统 ROCm 7.14（hipBLASLt 1.4.1）下，以及在 setup 自己的路径下：
  来自 `gfx120X-all` 的 TheRock 7.10.0a20251120 wheel（hipBLASLt 1.2.0）加 `build_engine_hip`。
- **ctest**（全部 45 个已注册的测试，R9700）：42 个通过，`hip_prefill_hipblaslt_gemm` 跳过（没有表），
  另有两个因 GPU 之外的原因失败：`ple_parity` 需要 Q2_0 模型 fixture，`expert_multi_test` 需要 AVX-512 CPU。
  `hip_handoff` 需要 volatile 的环形缓冲写入：在 gfx1201 上，一次对映射页锁定内存的普通写入会留在 GPU 的 L2
  中直到该流被同步。
- **架构检查：** 仅 gfx1100 的构建在启动时会在 gfx1201 显卡上以上文那条消息停止（引擎与 `strata-device`）。
- **端到端**（Coder IQ1_M，setup 的参数：`--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`、
  MTP、`--kv int8`、`--max-context 32768`；128 个贪心 token，系统 ROCm 7.14，没有 hipBLASLt 表）。回答连贯，
  且在两块卡上相同。在线服务端冒烟测试（`serve/server.py`：网页、models、props、聊天、前缀复用、流式、
  Anthropic 端点、一个带采样的代码回答）在每块卡上都通过 9/9。

  | 显卡 | 专家缓存 | 显存峰值 | 4K 提示词 | 4K decode | 16K 提示词 | 16K decode |
  |---|---|---|---|---|---|---|
  | Radeon AI PRO R9700 32 GB | 12,288 槽位，23.4 GiB | 29.7 GiB | 982 tok/s | 45.5 tok/s | 1,402 tok/s | 48.3 tok/s |
  | RX 9070 XT 16 GB | 4,931 槽位，9.4 GiB | 15.6 GiB | 782 tok/s | 30.8 tok/s | 1,235 tok/s | 35.4 tok/s |
  | R9700，TheRock 7.10 wheel | 12,288 槽位 | | 957 tok/s | 44.2 tok/s | 1,284 tok/s | 43.9 tok/s |

  每次运行中引擎的常驻内存约为 26 GB。自 0.1.31 起，`__byte_perm` 是一条 `v_perm_b32`，打包字节的减法/比较
  一次处理四个通道（#262，ttio2tech）：R9700 上 decode +15%（4K 提示词上 46.0 -> 53.0 tok/s，预热后
  52.0 -> 60.5），9070 XT 上 +5-7%，提示词不变，token 相同。
- **hipBLASLt：** 上述验证使用的是 hipBLASLt 1.4.1。用该版本在 R9700 上按引擎的形状标定的表（每个 GEMM 相对
  hipBLAS 提升 0.98-1.76x）把端到端的提示词速度改变了 0-3%，在噪声之内，所以没有为它随包发布表：那里普通的
  hipBLAS 路径已经很接近（我在自己的 R9700 上用 1.4.1 标定的表测得相同的结果：每个 GEMM 1.15x，提示词 +1%
  ——见下文"调优表"）。对于 hipBLASLt 1.5.0（ROCm 10.2.0a nightly），随包发布
  `tools/hip/gfx1201-hipblaslt-100500.txt`（见下文"调优表"）；在一块 R9700 上，它在 4,210 个 token 的提示词上
  测得提示词速度 +3.9%（1,590 对 1,531 tok/s），是一个不大的收益。在系统 ROCm 7.2.4 的 hipBLASLt 1.2.2 下
  普通路径差得较多，因此为它随包发布了 `tools/hip/gfx1201-hipblaslt-100202.txt`（gfx1100 表 26 个形状中的
  24 个；setup 只在该确切版本下使用它）：在一块配完整 IQ3_XXS、`--kv int8 --kv-resident 32768` 的 R9700 上，
  一个全新的 32K 提示词读取从 638 -> 1,177 tok/s，一个 7K 的从 656 -> 1,164 tok/s，decode 不变。
  `hip_prefill_hipblaslt_gemm` 配合它通过。
- **两块卡在一次运行中（层切分，引擎 0.1.30）：** 配置的 `"backend": "hip", "gpu": [1, 0]`（R9700 在前）经由
  `serve/server.py` 运行（自 0.1.31 起 setup 用 `--gpus 1,0` 写出它）。自动切分把第 0-27 层放在 R9700、
  第 28-47 层放在 9070 XT 上。当全部专家层都在 GPU 上时，切分产出的 token 与单独用 R9700 完全相同（4K 与
  16K 提示词）；切分上的检查点（第二轮、回退、共享一个前缀的提示词、一个被取消后重试的提示词）给出的 token
  与一次全新读取完全相同。对话缓存在启动时会拒绝切分（退出码 2）。在这一对卡上切分并不划算：单独的 R9700
  已经装得下全部 12,288 对专家。

  | 运行（同一会话，已预热） | 4K 提示词 | 16K 提示词 | decode |
  |---|---|---|---|
  | 单独 R9700 | 1,794 tok/s | 1,804 tok/s | 51-52 tok/s |
  | R9700 + 9070 XT，自动切分 | 1,384 tok/s | 1,906 tok/s | 42-43 tok/s |
  | 单独 RX 9070 XT | 1,016 tok/s | 1,519 tok/s | 38-40 tok/s |

  只有当没有单块卡能装下模型的专家层时切分才值得。自 0.1.31 起，切分会在 Linux 上页锁定整个专家 arena
  （8 GiB 的上限只针对 Windows/WSL2，#253）：把 R9700 的专家缓存设为 1,500（多数专家是流式读取的）时，
  切分以 1,803 tok/s 读取一个 16K 提示词而不是 1,287，且在 5 + 5 次启动中 token 相同。配置中的
  `"split_skip_if_fits": true`（0.1.31，可选开启）会在第一块卡装得下每一个被配置档收录的专家时，让这样一对卡
  只用第一块运行：R9700 在前时，4K 提示词 1,776 tok/s（切分：1,244），decode 约 60 tok/s（切分：约 51），
  token 与单独用 R9700 相同（docs/MULTI_GPU.md）。
- **速度开关（引擎 0.1.32，在 R9700 / 9070 XT 上用 Coder IQ1_M 模型包实测）：**
  - MoE 路由器（`router_top10`）默认运行一个不带串行 FP64 求和与块屏障的 HIP 内核：id 与权重逐位相同
    （`hip_router_fast` 检查 65,536 行），每次调用 39 -> 9-12 us，贪心 token 相同（5 + 5 次启动）。
    decode：R9700 上 62.4 -> 70.0 tok/s，9070 XT 上在一次 A/B 中 +4%（ROCm 10.2 nightly）；一位用户用 setup 的
    TheRock 7.10 wheel 与 i5-12600K 做的重复 A/B/A/B 测得 +1-4%，落在 12-20% 的逐次波动之内（#432）——
    它到底能赚多少，取决于在那套配置中路由器占 decode 的比重。`STRATA_HIP_ROUTER_OLD=1` 运行可移植的内核。
  - `STRATA_HIP_WMMA=1`（可选开启，仅 gfx12，int8 KV）：提示词路径的 QSA 注意力跑在 RDNA4 矩阵核心上
    （在 gfx12 配 `--kv int8` 时引擎会在启动时打印一行关于它的提示；它从不自行启用，因为输出位会变；把该变量
    设成任何值，包括 0，都会让提示不再出现）（`v_wmma_f32_16x16x16_f16`，与 CUDA 张量核心内核一样用 FP16 高、
    低两半；`hip_prompt_attn_wmma` 用 FP32 内核与 FP64 为它划定误差范围）。是可移植内核的 7.2-7.5x；R9700
    提示词 4K 1,784 -> 2,427 tok/s，16K 1,797 -> 2,700（一位用 TheRock 7.10 的用户：+29-34%，#432）。
    并非逐位相同：贪心文本从大约第 50 个 token 起不同，与 CUDA 张量核心注意力一样。启用它时提示词路径的专家
    环形缓冲是 96 槽位（相当于 STRATA_PREFILL_RING=96）：用默认的 384 时 9070 XT 的 4K 提示词掉到 718 tok/s；
    用 96 时有收益（1,017 -> 1,211；16K 1,518 -> 2,032）。PR #329（bsorensen110）贡献了一个速度相当
    （1% 之内）的等价 gfx12 WMMA 内核；本实现还会像 decode 内核那样遮蔽 KV 流式化尚未常驻的 KV 页。
    在我的 R9700 上（引擎 0.1.40，IQ3_S，431072 上下文，`--kv int8`，下文 gfx1201 的 hipBLASLt 1.4.1 表，
    机器具名，GPU 空闲）：4,210 与 8,830 个 token 的全新提示词——我在那里测过的最长提示词是 8,830 个 token，
    431072 上下文是模型运行的配置，不是这些提示词达到的长度——各三次配对试验，每次启动后的第一个全新提示词
    不计入：在 8,830 个 token 上是 1,571 -> 2,064 与 1,568 -> 2,065 tok/s（+31.4 与 +31.7%），在 4,210 上是
    1,699 -> 1,990（+17.1%），中位数 1,571 -> 2,064，三次试验的几何平均 1.27x。decode 89.8 -> 86.9 tok/s，
    不变。在此基础上再加 `STRATA_SELECT_WMMA=1` 在那些长度上没有任何变化（2,026 tok/s）；它的 +1.5% 是
    16K/262K 上下文下的数字。`hip_prompt_attn_wmma` 在该卡上通过。
  - 提示词路径的 QSA top-k 按一个查询实际拥有的块数来选择内核，而不是缓存的容量（#337，bsorensen110）：
    id 相同，在 AMD 上默认开启（NVIDIA 保留它的容量规则：那里 64K 提示词用它反而慢 1-3%）。
    `STRATA_SELECT_WMMA=1`（可选开启，gfx12）加上 #337 的矩阵核心块打分器；它的选择略有不同（256 个查询中
    254 个相同），在 R9700 上于 262K 上下文下为 16K 提示词带来 +1.5%。
  - `STRATA_PF_FUSED=1`（可选开启，gfx1200 / gfx1201，原生 IQ 模型包，#1277）：提示词的专家跑在 int8
    矩阵核心内核上（`v_wmma_i32_16x16x16_iu8`：gfx11 的内核配 gfx12 的通道布局）而不是 MMQ。更早的引擎在
    RDNA4 上会忽略该标志（MMQ 仍在，也没有横幅）；现在引擎会在第一个提示词时打印一次
    `strata: prompt experts on the fused int8 kernels (STRATA_PF_FUSED=1, #136)`。它不是某个架构的默认值：
    它的舍入与 MMQ 不同，而那些默认值只收录逐位相同的开关。`STRATA_PF_FUSED_NATIVE=0` 会在标志开启时保留 MMQ
    （用于 A/B）。R9700（gfx1201、ROCm 7.14、Coder IQ1_M 模型包），提示词时间取服务端报告的数值，各分支交替
    启动：4K 1,612 -> 1,353 ms（+19.1%），16K 5,579 -> 5,147 ms（+8.4%），32K 11,890 -> 10,962 ms（+8.5%）；
    每分支 3 次启动的第二轮在 4K 上给出 +12% 到 +16%（取决于哪一次启动），16K 上 +8.3%；内核本身在
    4K / 16K / 32K 上分别是 MMQ 的 1.48x / 1.20x / 1.14x。质量：4K 上（48 个提示词）相对 FP16 提示词路径的
    首 token KL 是 MMQ 的 1.10x，而该样本量无法把它与 MMQ 区分开；top-1 在 48 个提示词中 48 个一致（每个提示词
    的 top token 都是同一个，所以这项检查说明不了太多）；4 of 4 长上下文检索检查通过。0.1.42 在 R9700
    （Linux、ROCm 7.14）上的复核：12 个 1.5K-24K token 的提示词，相对 FP16 提示词路径的首 token KL
    （MMQ / fused）：IQ3_S 0.00332 / 0.00284，top-1 两者都是 12 of 12；IQ3_XXS 在一组样本上是 0.00335 /
    0.00452，第二组上是 0.00344 / 0.00323（24 个提示词合并：0.00340 / 0.00388，top-1 两者都是 24 of 24）；
    Q2_0 模型包在 gfx12 上不采用这些内核（输出与 MMQ 逐位相同）。因为 IQ3_XXS 的合并值没有达到或低于 MMQ 的，
    该标志保持可选开启。仅在 Linux 上实测。gfx1200 会构建同样的内核但未运行；Windows 与 K-quant 模型包
    （`STRATA_PF_FUSED_KQ=1`）未测量。
- **已知：** 偶尔（约十分之一的一次启动），一次 HIP 运行的贪心输出会在某个 token 上与另一次启动不同，
  单块卡或两块卡都会，在引擎 0.1.29 上同样如此；尚未解释。
- **未验证：** 图片、16K 以外的长上下文、回答质量的基准测试。

## 社区验证过的显卡
<a id="community-validated-cards"></a>

由其持有者运行，不在维护者的机器上；setup 像对待 gfx1100 / gfx1201 一样接受它们。setup 随包发布 gfx1200 的
hipBLASLt 表（数字见下文）；gfx1101 没有（按[调优表](#tuning-table)自制一份，并比较用与不用它的提示词速度）。

- **gfx1101，RX 7800 XT 16 GB**（jhohertz，#254；引擎 0.1.29、Ryzen 9 5950X、121 GiB 内存、系统 ROCm 配
  hipBLASLt 1.4.1）：`./setup.sh --backend hip` 检测到该卡并编译了引擎；`strata-device --selftest` 通过；
  ctest 32 个中 30 个（`ple_parity` 需要 Q2_0 fixture，`platform_memory_test` 需要 memlock 上限）。Coder IQ1_M，
  64K 上下文，MTP，配一张由持有者标定的表：4K-9K token 的全新提示词 898-953 tok/s，decode 38-44 tok/s
  （128 个 token）。
- **gfx1200，RX 9060 XT 16 GB**（Efeisot，#256，接在 #176 之后；引擎 0.1.29、Ryzen 9 7950X、64 GiB 内存、
  ROCm 7.2 配 hipBLASLt 1.2.2）：ctest 32/32（不算 `ple_parity` 与 `platform_memory_test`）。Coder IQ1_M，
  贪心，MTP `--spec 4 --spec-min-p 0.5`，配一张由持有者标定的表：

  | 提示词 | 提示词速度 | decode |
  |---|---|---|
  | 2,374 个 token | 540 tok/s | 27.1 tok/s（草稿接受率 0.69；接受率为 1.00 时 31.0） |
  | 65K（`--kv int8 --kv-resident 65536 --prefill 16384`） | 748-753 tok/s | 26-40 tok/s，取决于接受率 |
  | 130K（同样的标志） | 725 tok/s | - |

  贪心输出在各次运行之间相同。作为对比，llama.cpp 的 HIP 构建在那块卡上测得 decode 20 tok/s、提示词 450 tok/s。

  自 0.1.38 起 setup 随包发布那张表（`tools/hip/gfx1200-hipblaslt-100202.txt`）。在引擎 0.1.31 上，一个二进制
  （ctest 37/37，不算 `ple_parity` 与 `platform_memory_test`），提示词为重复的代码块，短提示词用 prefill 2560，
  其余用 16384 并带 `--kv int8 --kv-resident 65536`，各两次运行——用表与用普通 hipBLAS（无表）的提示词速度：

  | 提示词 | 用表 | 普通 hipBLAS | 收益 |
  |---|---|---|---|
  | 2,374 个 token | 597-599 tok/s | 396-398 tok/s | 1.50x |
  | 65,045 个 token | 754-757 tok/s | 380 tok/s | 1.99x |
  | 130,091 个 token（同样的标志） | 719 tok/s | 363-370 tok/s | 1.98x |

## RDNA2（gfx1030）
<a id="rdna2-gfx1030"></a>

RX 6800 / 6800 XT / 6900 XT / 6950 XT 运行相同的内核：wave32，每个工作群组 64 KiB 的 LDS。唯一的区别是 dot
指令：RDNA2 没有 `v_dot4_i32_iu8`（gfx11 及更新才有），所以 `dp4a` 经 `__builtin_amdgcn_sdot4` 使用普通的带符号
`v_dot4_i32_i8`，它编译成一条 `v_dot4c_i32_i8`（同样的带符号乘带符号的字节乘积，模 2^32）。没有 WMMA；QSA
打分器采用与 gfx1100 相同的有序 FP32 回退。在一位维护者运行过之前，CMake 把 gfx1030 列为未验证（构建时会警告）；
下面的报告来自一台社区机器：一块 RX 6900 XT 16 GB（gfx1030）、i7-13700KF（8 个 P 核与 8 个 E 核，AVX2，
无 AVX-512）、63 GB 内存、NixOS、来自 nixpkgs 的 ROCm 7.2.3（clang 22、hipBLAS 3.2）。

- **构建与测试**（引擎 0.1.26）：为 gfx1030 做的完整 HIP 构建，用 cmake 与 ROCm 自带的 `clang++` 手工完成
  （nixpkgs 的 ROCm 不是一棵 `/opt/rocm` 目录树，所以 setup 的 `build_engine_hip` 未被走到；二进制被放进
  `engine/` 供 setup 使用）。全部 28 个已注册的 ctest 测试都在该卡上通过，包括 `hip_device_selftest`。
  本分支引擎 0.1.30 的构建在同一块卡上配置、构建、运行都干净；下面的速度就是在其上测得的。
- **端到端**（Swift 1.5 IQ3_XXS，`--context 131072` 配 `--kv-resident 32768 --adapt-every 1
  --vram-reserve-mib 1024`，200 个贪心 token）：默认 15 个 CPU 池 worker 下 decode 38-42 tok/s（除宿主线程外
  每个物理核一个；8 个 worker：36；24 = 每个逻辑核：27），即使在 131,072 token 上下文装满时仍然稳定。
- **Prefill**（同样的 15-worker 配置）：2,000 个 token 的提示词 246 tok/s，自动的 8,192 token 分块下
  330-339 tok/s（7,997 与 15,967 个 token；首个 token 时间 8.9 与 48.3 s），522 个 token 的提示词 133 tok/s
  ——短提示词是固定开销（34 个 token 时 19-25 tok/s）。16K prefill 之后的 decode 保持在 45.5 tok/s。
- **16 GB 的卡：** 专家缓存在 16K 上下文下持有 6,310 个槽位（10.2 GiB），在 131,072 下是 5,247 个槽位，
  那里 setup 把 KV 缓存留在显存中，因为 IQ3_XXS 需要约 60 GB 内存再加缓存才能流式化它。
  [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) 中那份 7900 XTX 配置里的 `--pcie-frac 0` 与 `--adapt-every 0`
  在这里分别损失 8.6 与 13.6 tok/s（用默认值时是 30）：在 16 GB 的卡上请保留默认值。
- **hipBLASLt：** ROCm 的 hipBLASLt 不随包发布 gfx1030 的内核，所以没有表，运行普通的 hipBLAS 路径。
- **FP16 的提示词 GEMM（#835）：** gfx1030 上的 rocBLAS 只有为 FP16 入 / FP16 出的 GEMM 调优过的内核；
  提示词路径的 FP16 入 / FP32 出以及 BF16 乘积跑的是通用内核，慢约 6.6 倍（在 N = 10240、T = 7313、K = 2560 下
  是 5.6 与约 5.3 TFLOPS，而前者是 37.7）。`STRATA_HIP_PROMPT_F16=1`（可选开启，默认关闭；引擎会在 gfx103x 上
  打印一条提示）让 16 位的提示词 GEMM 以 FP16 入、FP16 出；不设置就是原来的路径。`STRATA_DBG_NAN=1` 还会统计
  非有限的 FP16 输出（求和超过 65504）。输出与旧路径并非逐位相同（每个 GEMM 输出的 FP16 舍入取代了 BF16 GEMM
  激活值的 BF16 舍入）；分布检查显示出与未显示的内容在
  [bench/results/2026-10-04-rdna2-fp16-prompt](../../bench/results/2026-10-04-rdna2-fp16-prompt/README.md) 中。
  在第二台社区机器上实测——2x RX 6900 XT 16 GB（这些数字用其中一块）、Ryzen 5 5600X（6 核，AVX2）、128 GB、
  Ubuntu 26.04、ROCm 10.0.0、引擎 0.1.39 像 setup 那样手工构建并带 `-DSTRATA_PREFILL_MMQ=ON`、GSQ-RCO IQ3_S：
  9.4K / 34.7K / 105.8K token 的提示词读取为 744 / 915 / 926 tok/s，而原先是 439 / 466 / 461；decode 不变。
  在那台机器上，HIP 的 ctest 带与不带这一改动都通过 65 个中的 60 个：2 个跳过（`hip_prompt_attn_wmma`，
  没有矩阵核心；`hip_prefill_hipblaslt_gemm`，没有 hipBLASLt 表），3 个因引擎之外的原因失败（`ple_parity`
  需要那台机器上没有的 Q2_0 PLE 文件，`expert_multi_test` 拒绝没有 AVX-512 的 CPU，`platform_memory_test`
  在 shell 默认的 `ulimit -l` 下无法 `mlock`）。
- **gfx1031**（RX 6700 XT，#524）：setup 认识它（`gfx103X-all` 的 wheel，未验证）；它的报告者在单块卡上日常
  运行它。更多的报告：一块按 gfx1030 运行在 ROCm 7.2.4 上的 RX 6700 XT 12 GB（#1027：IQ2_XS，decode
  30-32 tok/s，提示词约 300 tok/s，needle 6 of 6），以及一块在 Windows 上用自行构建引擎的 RX 6800M 12 GB
  （#915、#1078：Q2_0，decode 9-27 tok/s，提示词 50-114 tok/s；在 31 GB 内存下专家从磁盘流式读取）。
  现成的 Windows zip 除非构建时包含，否则没有 gfx1031 的代码：在 `tools\hip\build_windows.bat` 里设
  `STRATA_HIP_ARCHS=gfx1031`（默认列表自 0.1.40.2 起就包含它）。
- **gfx1150**（Radeon 890M、Ryzen AI 9 HX PRO 370、Strix Point，#1217）：自 0.1.40.2 起作为一个未验证的目标
  可以构建并运行（CMake 会警告；设备代码本来就已覆盖它：WMMA 的门控与 `gfx_arch_is_gfx11_wmma()` 都包含它）。
  一台社区机器：MINISFORUM N5 PRO、96 GB DDR5-5600、GTT 提升到 64 GiB
  （`ttm.pages_limit=16777216 ttm.page_pool_size=16777216`）、为 gfx1150 的 TheRock ROCm 7.14.1、一个非特权的
  LXC 容器、IQ3_XXS：聊天与工具调用可用，decode 15-19 tok/s，HIP 的 ctest 除了那些需要公共检出所缺少文件的
  测试之外全部通过。提示词在普通 hipBLAS 下以约 125-130 tok/s 读取，用
  `tools/hip/gfx1150-hipblaslt-100401.txt`（ROCm 7.14.1 的 hipBLASLt；3.6K 个 token 124 -> 216 tok/s，
  7K 130 -> 226 tok/s，decode 不变）快 1.5-1.7 倍：
  [bench/results/2026-10-07-community-gfx1150](../../bench/results/2026-10-07-community-gfx1150/README.md)。
  Setup 目前不为它安装（除 Strix Halo 之外的集成 Radeon 会被指名并列为不受支持）：请用
  `-DCMAKE_HIP_ARCHITECTURES=gfx1150` 手工构建，并自己把 `STRATA_HIPBLASLT_TUNING` 指向那张表。
- **gfx1152**（Radeon 860M / 840M、Ryzen AI 300 "Krackan"，#1625）：和 gfx1150 一样在 CMake 的未验证列表里，
  所以 `-DCMAKE_HIP_ARCHITECTURES=gfx1152` 会带着警告构建。它运行可移植的内核（矩阵核心的门控只指名 gfx1150 与
  gfx1151），没有显卡报告过它，setup 也不为它安装（除 Strix Halo 之外的集成 Radeon 会被指名并列为不受支持）。
  欢迎来自真实 860M 的报告。
- **未验证：** gfx1032（同样的 `dp4a` 路径，无硬件报告）、setup 自己的构建路径与 gfx1030 上的
  `gfx103X-all` wheel、图片、回答质量的基准测试。RDNA1（gfx1012，RX 5500 XT）可手工构建：
  [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012)。

## gfx906（Instinct MI50 / MI60、Radeon VII）：wave64，从源码构建
<a id="gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source"></a>

gfx906 是 wave64，既没有 WMMA 也没有打包字节运算，所以上文那个 wave32 后端会拒绝它。一个独立的可选开启构建
通过一个小型兼容层（`include/strata/platform/hip_compat/`）把 CUDA 源码编译成 HIP，把一个 CUDA warp 映射到
64 通道 wavefront 的一个逻辑半区（32 宽的 shuffle，属于自己半区的 ballot）。热点内核各自有 wave64 的布局
（见下文）。Setup 目前还不构建它：请手工构建，并像在别的显卡上那样用配置运行 `serve/server.py`。

**社区实测、可选开启（0.1.42）。** gfx906 的构建与其用户共同维护：我们没有 gfx906 显卡，所以此处的内容都是
社区成员在他们的 MI50 上测得的（注意力的两个开关 `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` 与
`STRATA_GFX906_ATTN_REDUCE12=1`，即 0FL01 的 PR #1661 与 #1718，默认关闭且只编进这一个构建），而这里的引擎只
被检查过能为 gfx906 编译，仅此而已。其中没有任何内容改变其他显卡的构建。这一构建看到一块 wave32 显卡时会在
启动时带消息拒绝（#1728）。欢迎报告它在你的显卡上的运行情况。

**ROCm。** AMD 当前的 ROCm 发行版不再随包发布 gfx906 的库。下文的构建与测量使用了社区镜像
[`mixa3607/rocm-gfx906:7.14-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906) 的 HIP 7.14
（带 gfx906 内核的 rocBLAS/hipBLAS），运行在内核自带的 amdgpu 驱动上（Ubuntu 24.04，内核 6.8）。
不使用 hipBLASLt。

```sh
# inside the image, with this checkout at /src and llama.cpp at the pinned commit in third_party/llama.cpp
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

`STRATA_HIP_GFX906` 打开 CUDA 目标并拒绝 `STRATA_ENABLE_HIP`（一次只能有一个 HIP 构建）。请用
`--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1` 运行容器；在某些主板上
需要 `HSA_OVERRIDE_GFX_VERSION=9.0.6` 运行时才接受该显卡。

**与 CUDA 构建的不同之处**（都在 `STRATA_HIP_GFX906` 之下；每个做 A/B 的开关都会恢复 CUDA 的布局）：

| | gfx906 | 开关 |
|---|---|---|
| 分组的原生专家层 | 模式 7：符号用 `dp4a(g ^ m, u) - dp4a(m, u)`（没有字节 SIMD）、网格与激活值放 LDS，每组只加载一次权重；仅限 IQ 格式（Unsloth 的 K-quant / Q5_1 / Q8_0 专家用 CUDA 布局） | `STRATA_EXP_MODE=2`（先前的 AMD 布局） |
| MMVQ | 每行一个 wavefront，64 通道蝶形，无 LDS | `STRATA_MMVQ_WAVE=0` |
| 路由器 top-10 | 一个 wavefront，寄存器 argmax | - |
| verify window 的路由（256 专家路由器） | 该 window 的各 token 在一次多列 BF16 投影 + 一次 top-k 中完成，约占一个约 55 ms window 的 3 ms | `STRATA_ROUTE_PER_TOKEN=1` |
| hyper-connection 读取 | norm 的载入保持在途，`up` 每行 8 通道 | `STRATA_GR_FAST=0` |
| `gr_down_multi`，可选开启 | 沿 K 切分（8 行 x 5 段 2048），每个 window 约 3%；求和顺序与单 token 读取不同，所以默认关闭 | `STRATA_GR_SPLIT=1` |
| IQ4 的表查找 | llama.cpp 的 `v_perm_b32` 序列（HIP 的 `__byte_perm` 是一块暂存数组） | - |
| QSA 提示词注意力 | FP32 内核（没有张量核心 / WMMA） | - |

上文的每个开关都与它所替换的布局做过逐位检查：合成数据上的一致性测试、真实 GGUF 行上的
`native_expert_bench`（一种模式对另一种），以及一个固定请求的贪心文本。

**实测**于 2x Instinct MI50 16 GB（每块 85 W 功耗上限，两块都是 PCIe 3.0 x16，彼此间启用点对点访问）、
Xeon E5-2666 v3（10 核，AVX2）、32 GB DDR4、SATA SSD；Coder IQ1_M（`--native` 模型包），131,072 token 上下文，
`--kv int8 --kv-resident 32768`，`--layer-split 27`，`--prefill 4096`，MTP `--spec 4 --spec-min-p 0.5`，
`--pcie-frac 0`，`STRATA_ARENA_MMAP=1`。测量是在本构建所截取自我的那个移植版上做的（基于 0.1.30 与 0.1.33
的构建，相同的内核，外加层切分权重裁剪与映射 arena，它们是各自独立的拉取请求）：

- 两块卡合计在显存中持有全部 12,288 个专家；
- 一个 17,043 个 token 的 agent 提示词（Claude Code 的首个请求，含工具）在 43.5 s 内读完（392 tok/s），
  其上 decode 57.7-59.3 tok/s；
- 服务期间使用 32 GB 内存中的 25 GB；
- 该请求的六次重复（输出上限 6,000 个 token）以及一次 900 秒、两客户端的负载测试跑六轮
  （407 个请求，202,940 个 token）：没有停顿、没有错误、没有 GPU 故障。

同一台机器上的对比：llama.cpp（3cf0325，ROCm）在同一个 Coder IQ1_M 上测得 `tg128` 26.9 tok/s。

**尚未完成：** setup.py 的检测与自动构建、Windows、图片（`--vision`）、MI60 与 Radeon VII（同样的 gfx906
指令集；未运行）、单卡运行，以及张量切分实验（把每一层的一半放在两块卡上；它能工作但不属于这个构建）。

## 调优表
<a id="tuning-table"></a>

一张 hipBLASLt 表保存的 solution id 只对一种 GPU 架构与一种 hipBLASLt 版本有效，所以它必须在该引擎运行所用的
ROCm 下、在该显卡上标定。随包发布的 gfx1100 表的各行就是引擎的稠密 GEMM 形状；要为另一种卡或版本重新标定它们
（几分钟）：

```sh
cmake --build build-hip --target tune_hipblaslt
CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100200.txt)
./build-hip/tune_hipblaslt $CASES --tuning-out table.txt
```

文件的第二行会指明架构与版本（`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100401`）；把它存成
`tools/hip/<arch>-hipblaslt-<version>.txt` 供 setup 使用，或把 `STRATA_HIPBLASLT_TUNING` 指向它。在保留它之前，
请先比较用与不用它时的提示词速度。

随包发布的表：

- `gfx1100-hipblaslt-100100.txt`、`gfx1100-hipblaslt-100200.txt`：RX 7900 XTX。
- `gfx1100-hipblaslt-100401.txt`：RX 7900 XTX，用随包发布的 ROCm 10.0.0
  （`rocm/dev-ubuntu-24.04:10.0.0-full`，hipBLASLt 1.4.1）在一颗 Ryzen 7 9800X3D 上标定，覆盖随包发布的
  gfx1100 表的 26 个稠密 GEMM 几何。没有它时，那套栈通过普通 hipBLAS 读取提示词：在 Swift 1.5 IQ3_XXS、
  262144 上下文下，全新提示词的 prefill 在 1.7K 个 token 上测得 682 -> 1,038 tok/s，在 6.5K 个 token 上是
  857 -> 1,524 tok/s（3 次的中位数，decode 不变），且 `hip_prefill_hipblaslt_gemm` 报告 `fallbacks=0`。
  setup 只在已安装的 hipBLASLt 报告 1.4.1 时使用它。43 行的修订版补上 26 个稠密项覆盖不到的两个几何，用同一
  代码树的 `tune_hipblaslt` 与引擎的 32 MiB 工作区标定：`f16 1280 2560 1280`（小 T 的专家 GEMM，T=8..999）
  与 `bf16 10240 2560 10240`。一个 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS、192000 上下文的提示词，否则几乎每次启动
  都会回退出专家 GEMM：`STRATA_HIPBLASLT_VERBOSE=1` 用 26 行的表报告了 642 次回退，分布在 642 个不同的 T 值上
  （1..1873）。加上这些行在 3151 个 token 的提示词上测得 711 -> 938 tok/s（+31.9 %，3 组交错配对的平均值：
  +36.0 / +32.4 / +27.6 %），decode 不变（58.6 -> 60.6 tok/s，在这台机器 2.3 % 的逐次漂移之内），
  且 `fallbacks=0`。查表对匹配的几何会取最接近的 T 桶，所以各桶覆盖整个小 T 区间，而不是每个 T 一行。
- `gfx1100-hipblaslt-100500.txt`：RX 7900 XTX（gfx1100，24 GB），用一个 ROCm 10.2.0a20261003 nightly SDK
  （`libamdhip64.so.7.17.26392`，hipBLASLt 1.5.0，版本号 100500）标定。与其他 gfx1100 表相同的 16 个稠密 GEMM
  几何，在 T=4096 与 T=8192（32 行），用针对那个库运行的 `tune_hipblaslt` 标定。在这个 ROCm 上引擎拒绝
  100100/100200 表（版本不符）并用普通 hipBLAS 做 prefill；用这张表时，一个 0.1.38 血统的 nightly 引擎以
  1687 tok/s 为一个 131071 个 token 的提示词做 prefill，不用它时是 926（各 3 次干净运行的中位数，贪心；
  decode 不变）。逐次运行的数据与标定命令：PR #745 的 `2026-10-04-community-rx7900xtx-hipblaslt-100500` 文件夹。
- `gfx1201-hipblaslt-100500.txt`：Radeon AI PRO R9700（gfx1201，32 GB），用 ROCm 10.2.0a20260914
  （AMD 的 `gfx120X-all` nightly，hipBLASLt 1.5.0，库构建 `d3164197`）标定。16 个稠密 GEMM 几何，在 T=4096
  与 T=8192，32 行。setup 只在已安装的 hipBLASLt 报告 1.5.0 时使用它（当 `/opt/rocm` 是一套系统 ROCm 7 或更新时
  就能在那里找到它）。版本号是引擎唯一能检查的东西，所以另一个 1.5.0 构建可能给它的 solution 编不同的号。
  `hip_prefill_hipblaslt_gemm`（设置了 `STRATA_HIPBLASLT_TUNING`）是一个冒烟测试：它会拒绝为其他架构或版本的表，
  并检查存在一行 BF16 与一行 F16 且它们与 hipBLASEx 一致，这覆盖 32 行中的 2 行。它不能证明某个 solution id
  是有效的：对于库拒绝的 id，引擎会回退到 hipBLASEx，而该测试仍然通过。请设置 `STRATA_HIPBLASLT_VERBOSE=1`
  运行它并在它的汇总行中寻找 `fallbacks=0`，并在把这张表用于另一个 1.5.0 构建之前用 `tune_hipblaslt` 重新标定。
- `gfx1151-hipblaslt-100400.txt`：Radeon 8060S（Ryzen AI Max+ 395，gfx1151，128 GB；Fedora 44，内核 7.2.8），
  用 setup 自 0.1.40.2 起为 gfx1151 安装的那份 ROCm 7.14.0a20260608 wheel 标定（#1267）：它们的 hipBLASLt 是
  1.4.0（版本号 100400），所以 `gfx1151-hipblaslt-100401.txt` 与 `-100500.txt` 都不适用，setup 的引擎只能通过
  普通 hipBLAS 读取提示词。与 100401 表相同的 90 行（它的几何，含 `--prefill 16384` 下的提示词形状），用
  `tune_hipblaslt --workspace-mib 32` 与该表的 cases 标定：每个形状比普通 hipBLAS 快 3.1x 到 14.1x
  （中位数 6.2x）。设置了 `STRATA_HIPBLASLT_TUNING` 时，`hip_prefill_hipblaslt_gemm` 通过
  （`launches=4 fallbacks=0`），`hip_prefill_hcd_exact_parity` 通过（9 种分块大小下的 solution 1176 / 1177，
  0 个输出有差异，`fallbacks=0`），HIP 的 ctest 通过 62 个中的 61 个（`ple_parity` 需要 Q2_0 模型 fixture）。
  setup 只在已安装的 hipBLASLt 报告 1.4.0 时使用它。用这张表跑整个模型的提示词速度尚未测量。

- `gfx1150-hipblaslt-100401.txt`：Radeon 890M（gfx1150，Strix Point APU，16 CU），用 AMD 为 gfx1150 的
  TheRock ROCm 7.14.1（hipBLASLt 1.4.1 `cd957402`，版本号 100401）与引擎的 32 MiB 工作区标定。一块
  Qwen3.8-Flash-Next 提示词在该卡上记录的 16 个稠密 GEMM 几何（与 gfx1201 的 100500 表相同的 16 个），
  在 T = 64、128、...、16384：144 行。连续两轮 `tune_hipblaslt`；每一行保留全部六次重复中中位数最低的那个
  solution，因为偶尔会有一次重复停顿下来（一个 100 ms 的调用会读到 1.6 s），而两轮之间该工具的三次均值取法在
  26 行上给出不同的结果。每一行都胜过普通 hipBLAS，每个 GEMM 提升 1.55-22.6x（几何平均 4.5x），且
  `STRATA_HIPBLASLT_VERBOSE=1` 在 28-7,880 个 token 的提示词上没有报告任何回退。Setup 不会使用它（它不为
  gfx1150 安装）：请把 `STRATA_HIPBLASLT_TUNING` 指向它。实测数据与逐行的时间数据：
  [bench/results/2026-10-07-community-gfx1150](../../bench/results/2026-10-07-community-gfx1150/README.md)。

在我的 PC 上标定并保存在这份检出中、不随包发布：`gfx1201-hipblaslt-100401.txt`，一块 Radeon AI PRO R9700
（gfx1201，32 GB，1002:7551），位于一颗 Ryzen 9 9950X3D 配 62 GiB 内存的机器上，CachyOS（内核
7.3.0-rc6-1-cachyos-rc，`linux-firmware-amdgpu 20260916`，内核自带的 amdgpu 驱动），系统 ROCm
`rocm-gfx120x-bin 10.1.0`（hipBLASLt 1.4.1，HIP 7.16.26385，clang 24.0.0），setup 自己的 venv 中的
Python 3.14.7，引擎 0.1.40 在其上为 gfx1201 编译，模型文件放在一块 4 TB NVMe 上（btrfs），IQ3_S 在
131072 上下文配 `--kv int8`，GPU 空闲。对 `gfx1201-hipblaslt-100500.txt` 的 32 个几何运行 `tune_hipblaslt`：
相对 hipBLAS 每个 GEMM 几何平均 1.15x —— 在 N=512/640 K=2560 上是 1.55-1.68x，在 N=10240/12288 K=2560 上是
0.99-1.13x。用 `tools/hip/bench_prefill.py` 跨两次服务端启动的端到端结果，4,210 与 8,830 个 token 的全新提示词，
各三次配对试验——每次启动之后的第一个全新提示词不计入，那里是冷的（949 与 1,423 tok/s）：1,606 -> 1,613、
1,701 -> 1,726 与 1,605 -> 1,610 tok/s（+0.5、+1.5、+0.4%），中位数 1,606 -> 1,613，三次的几何平均 1.008x，
decode 87.1 -> 85.1 tok/s——在噪声之内，与上文 R9700 那段的发现一致。配合它 `hip_prefill_hipblaslt_gemm` 通过：
`tuning enabled (30 rows, gfx1201, version 100401)`，`launches=4 fallbacks=0`。

那张表中有两行被去掉：bf16 N=10240 K=320 ldy=10240，在 T=4096 与 T=8192。在 T=8192 时，hipBLAS 在五次独立的
`tune_hipblaslt` 运行中都更快——0.609-0.619 ms，而 16 个 hipBLASLt 候选中最优的是 0.648-0.677 ms
（0.90-0.95x），出现四个不同的胜出 solution id，全都比 hipBLAS 慢。只去掉 T=8192 那一行并不会让那次调用回到
hipBLAS：`closest()` 匹配 dtype/N/K/ldy 并取最接近的 T 桶，所以 T=8192 会去运行 T=4096 那一行；两行要么都去掉，
要么都保留。

来自我这台机器的两条标定说明。一个全新 `tune_hipblaslt` 进程的第一个用例会在时钟仍在爬升时为它的 hipBLAS
基线计时：那个 T=4096 的形状在五次运行中读出 0.31-0.63 ms，而紧随其后计时的用例每次都稳定在 0.606-0.619 ms，
所以请先给一个较重的用例计时，再读后面的那些。另外，模型加载之后的第一个全新提示词不是一次测量：在那次 A/B 的
两个分支中它给出 949 与 1,423 tok/s，而它之后的三次试验彼此只差 0.4-1.5%。

## 原始后端的验证（PR #94）
<a id="original-backend-validation-pr-94"></a>

以下是对原始后端的历史验证，不是对这一替代版本的新一轮测试计数。当前的构建/测试证据与性能限制记录在
[AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) 中。

针对上游 `c1e903310f211e6630780c3bd2038778c071c68d`（0.1.20）测试，固定的 llama.cpp 是
`3cf03257f219afbe7334045ff7c6a06ac68c627d`。该主机有一块 RX 7900 XTX（24 GiB）、Ryzen 9 7950X3D、
64 GiB 系统内存，以及配 ROCm 7.1.52802 / Clang 20 的 Fedora 系 Linux。用于冒烟检查的原版模型放在机械
RAID0 上；冷缺页很慢。请把对延迟敏感的数据放在 SSD 上，并单独测量预热后的驻留。

- 完整的 HIP Release 构建，包括可执行文件与一致性目标。
- 下列命令所选的全部 25 个测试于 2026-09-29 通过。
- 内在函数一致性包括带符号的打包字节点积与溢出。
- CPU/GPU 映射内存的交接覆盖延迟发布、变化的负载，以及两个方向上的重复计算图重放。
- GR 的覆盖包括最多八个 token 的融合批量，以及输入改变之后的计算图重放。原生 QSA 检查标量的 HIP 回退。
- Q8_K 量化与既有的参考逐字节一致。HIP 的 `__fmul_rn` 可能变成普通乘法；在 `quantize_act.cu` 中禁用收缩
  能在加上舍入魔数偏置之前保住那一次单独舍入的乘积。CUDA 的编译标志保持不变。
- mmap 回归覆盖规范与不同的原生层大小、截断/超大的文件、查找边界，以及出错后的重新打开。
- CUDA 13 / GCC 15 / sm_89 的可执行文件构建通过；在同一代码树上 GR、采样器、量化与 mmap 测试通过。
  这是一次回归检查，不是完整 CUDA 推理验证的主张。
- 仅 CPU 的 `strata-plan` 构建在两个 GPU 后端都关闭时通过。
- 带 mmap、随包发布的专家配置档、自动 GPU 缓存与 MTP 的真实 GSQ-RCO IQ3_XXS 服务端冒烟通过了算术、
  可执行文件内的 Python 加法、系统标记召回，以及 1,170 个 token 的批量 prefill 召回。四个请求全部以正确答案
  正常结束；一个引擎进程服务了它们全部。缓存持有 11,142 个专家（18.08 GiB），启动后还剩 874 MiB 显存空闲。
  这些短检查并不构成吞吐或质量的基准。

在运行已注册的专项检查之前，请先构建全部目标：

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` 需要一个外部的模型 fixture。`platform_memory_test` 申请 256 MiB 的锁定内存；测试 shell 的
8 MiB memlock 上限使该测试无法通过。这些是显式的排除项，不是被当作通过的跳过测试。另有未发布的上游
fixture 套件不在覆盖范围内。

已发布的性能表明确指出了它所测量的是那个开发中的快照；不可把它读作对之后每一次 rebase 的基准测试。视觉、
长上下文压力测试、广泛的回答质量等价性、其他 AMD 显卡，以及混厂推理，都不在此验证范围之内。
