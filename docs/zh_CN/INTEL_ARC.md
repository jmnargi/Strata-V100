# Intel Arc（实验性）
<a id="intel-arc-experimental"></a>

Strata 0.1.39 包含一个**实验性的 Intel Arc 引擎**：Strata 自己的引擎移植到 SYCL（Intel oneAPI）的版本。
maxfridbe 在 [#423](https://github.com/Niko1221/Strata/pull/423) 里写了它，参与测试的人补上了修复。
代码在 `sycl/` 里，这次移植自己的笔记、实测数字和维护流程在
[docs/INTEL.md](INTEL.md)。本页讲你需要什么、怎么构建，以及已经测过什么。

**「实验性」意味着：** Strata 维护者手上没有 Intel 显卡。我们会编译这次移植并跑它的内核测试，但
我们没在 Arc 上运行过。下面每一条 Arc 硬件上的结果都来自社区。NVIDIA 和 AMD
引擎没有变化：Intel 构建是一个独立的 CMake 目标，默认关闭。

发行版 zip 里**没有现成的 Intel 引擎**。你需要在 Linux 上从源码构建。

## 已经跑过什么，是谁跑的
<a id="what-has-been-run-and-by-whom"></a>

| | 硬件 | 结果 | 报告于 |
|---|---|---|---|
| 移植作者 | Arc Pro B70 32 GB，Ubuntu 24.04 | Coder IQ1_M：70-78 tok/s decode（解码），约 790 tok/s prompt（提示词）；IQ2_XS：51-64 tok/s；实测 256K 上下文 | #423、[INTEL.md](INTEL.md) |
| 社区 | 2x Arc Pro B70，`--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode，394 tok/s prompt（带现在已在 0.1.39 里的 `stage_room` 修复） | #423 |
| 社区 | Arc Pro B50 16 GB | Coder IQ1_M 约 23 tok/s，IQ2_XS 约 25-27 tok/s，最高 128K | #423 |
| 社区 | Arc B580 12 GB，WSL2 | IQ2_XS 约 21 tok/s，Coder 约 15 tok/s；**也见到了 device loss（设备丢失）** | #423 |
| 社区 | 2x Arc Pro B60 24 GB，`--layer-split` | Coder IQ1_M：56.6 tok/s decode，428 tok/s prompt（2,129 tokens）；Flash-Next IQ2_XS：58.6-61.3 tok/s decode，436 tok/s prompt；所有专家层都在显存里；8K 上下文 | [bench/results/2026-10-04-community-2x-arc-pro-b60](../../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |
| 社区 | 单张 Arc Pro B60 24 GB | Coder IQ1_M，把 12,288 个专家层中的 4,042 个镜像到 RAM：11.9 tok/s decode（需要同一份报告里的 ring-wait 修复） | 同上 |
| Strata 维护者（0.1.40.2 及以后） | Arc Pro B70 32 GB（xe）和 Arc A750 8 GB（i915），测试机 | B70 上跑 Flash-Next IQ3_S，24.6k 个专家层中 11.5k 常驻：10 个提示词的门禁通过，4K tokens 上 prompt 980-1,000 tok/s，decode 从 30（散文）到 41（代码）tok/s；xe 的别名页修复和 `NO_HOST` 要求见 [INTEL.md](INTEL.md) | INTEL.md, "Arc Pro B70 with a model that does not fit" |

0.1.39 的移植把 0.1.38 的移植重新搬到 0.1.39 的引擎源码上（#606 的 NaN 修复、#649 的 verify trace、
新的提示词路径）。0.1.39 的 `sycl/` 在 0.1.39 自己的引擎源码上编译不过（#784：#626 的
`ThreadAffinity` 类型，以及 #559 带来的 `NativeDense::load` 的层范围），而且它的 ring wait 从不认为某个慢层
还在跑（#866、#867）。这两处都在 0.1.40 里修好，下面两份 B60 报告正是在 0.1.39 之上叠加这两个修复测出来的。
**还没有人在 Arc 上跑过未打补丁的 0.1.39 移植**，其它各行是在更早的版本上测的。

B60 那两行跑的是 `6f32ec0` 加上两个小的 `sycl/` 修复（编译修复和 ring-wait 修复），AOT `bmg-g21`，环境是
Ubuntu 24.04，带 `xe`、Level Zero V2、NEO 26.09.37435.12 和 oneAPI 2026.1.1，没有用 Docker。主机：Ryzen 5 5600、
64 GB RAM、每张卡 PCIe 3.0 x8。
完整参数、每次请求的耗时和引擎日志都在表里链接的那份报告中。

## 这里测过什么（0.1.39）
<a id="what-was-tested-here-0139"></a>

- **能编译：** Ubuntu 24.04（WSL2）、Intel oneAPI DPC++ 2026.1.1 + oneMKL 2026.1。整个 `sycl/` 工程
  以 0 错误构建通过：`strata` 引擎外加全部 157 个目标（内核对拍测试和基准）。构建产物是
  SPIR-V（JIT）。AOT 构建（`STRATA_SYCL_AOT`）这里没有做，因为它需要 `ocloc`。
- **在 CPU 上跑内核对拍测试**（`ONEAPI_DEVICE_SELECTOR=opencl:cpu`，Intel 的 OpenCL CPU 运行时，AMD Ryzen 5 7600）：
  25 个里过 14 个。这和 PR #423 自己的 0.1.38 移植在同一设备上通过的集合相同：失败的是 CPU 数学上的严格
  浮点容差（相对误差 3e-6，上限 1e-6）、没有提供的模型 fixture，以及两个在 CPU 上超时的
  测试。这是检查内核能否编译和计算，不是对 Arc 的测试。
- **Setup：** `--backend sycl` 会先给出警告，然后在 Linux 上交给 `sycl/setup_intel.py`（单测：
  `tools/test_setup_sycl.py`）。
- **没有变化：** CUDA 引擎的 greedy 输出，与通过门禁的 0.1.39 构建逐字节比对一致（Q2_0 和 Coder）。
  HIP 构建不受影响（该选项默认关闭）。

**还没有人测过：** Windows（没有原生构建路径，见下文）、0.1.39 移植在 WSL2 上的 Arc、Alchemist
A 系列、集成 Arc 显卡（#515 里的 B390 / Panther Lake；共享内存的规划还不存在），以及
图片（Intel 上还没接）。

## 你需要什么
<a id="what-you-need"></a>

- **Linux**，例如 Ubuntu 24.04，装好 Intel 的显卡驱动（`xe` 或 `i915` 内核驱动，加上 compute runtime /
  Level Zero；Ubuntu 上是 `intel-opencl-icd libze1 libze-intel-gpu1`，或者看 Intel 的
  [客户端 GPU 指南](https://dgpu-docs.intel.com/driver/client/overview.html)）。
- **Intel oneAPI**：DPC++ 编译器（`icpx`，2025.3 或更新；这里构建用的是 2026.1）和 **oneMKL**。约 5 GB。
- `cmake` 3.24+、`ninja`、`git`（构建会拉取 ggml，除非你把 `STRATA_GGML_DIR` 指向一个 llama.cpp 检出）、
  Python 3。
- 就目前而言，`./setup.sh --backend sycl` 需要：**Docker**，上面那份清单里除了显卡驱动别的都不要。
  `sycl/setup_intel.py` 在由 `sycl/tools/Dockerfile` 构建的 `strata-sycl-dev` 镜像里运行引擎，引擎也在
  该镜像里构建（见下文「用 Docker 构建」）。Ubuntu 上：`sudo apt install docker.io git`，然后
  `sudo usermod -aG docker,render $USER` 并重新登录。注意 Dockerfile 基于社区版
  llama.cpp SYCL 镜像（`ghcr.io/snailium/...`），不是 Intel 或 Strata 的镜像。
- 显存：这次移植把专家层留在显卡上（`--stream-experts`）。32 GB 的卡装得下 Coder IQ1_M 或 IQ2_XS。
  更小的卡把一部分专家层镜像到 RAM，因此更慢。

## 用 Docker 构建（`setup --backend sycl` 走的就是这条路）
<a id="build-with-docker-what-setup---backend-sycl-uses"></a>

两条命令，和 [INTEL.md](INTEL.md) 的「How to build it」一样（第一次约 10 分钟）。在检出目录里执行，
这里把它叫作 `Strata`：

```sh
cd Strata
docker build -t strata-sycl-dev sycl/tools            # once: ~6 minutes, a 13 GB image
H=$(basename "$PWD")
# Arc B-series: bmg-g31 (Pro B70), bmg-g21 (B580, B570, Pro B60). Arc A-series (Alchemist): leave out AOT (JIT) and use build-sycl
docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
    -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"
```

然后执行 `./setup.sh --backend sycl`（见下文）。不要手工运行 `python3 sycl/setup_intel.py`：在 Ubuntu 上它会以
`externally-managed-environment` 失败；`setup.sh` 会在自己的 `.venv` 里运行它。

## 不用 Docker 构建（Linux）
<a id="build-without-docker-linux"></a>

这样构建出的引擎用于手工运行（[INTEL.md](INTEL.md) 里的「How to run it by hand」）；`setup` 本身用的是上面的
Docker 构建。从 Intel 的 apt 仓库安装 oneAPI（这里用的就是这个办法）：

```sh
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git
```

然后构建引擎。两条命令都可以：第一条走顶层的 CMake 选项，第二条直接配置 `sycl/` 工程。

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

选项：

- `-DSTRATA_SYCL_AOT=bmg-g31`（Arc Pro B70）或 `bmg-g21`（B580 / B570 / Pro B60）把 GPU 代码提前
  编译。这需要 `ocloc`（Intel 的 `intel-ocloc` 包）。没有它时，首次启动会 JIT 编译每个内核，
  大约要 47 秒。
- `-DSTRATA_SYCL_PARITY=OFF` 跳过内核测试（默认开启）。在显卡上运行
  `ctest --test-dir build-sycl` 即可跑这些测试。

`setup_intel.py` 会在检出目录里找 `build-sycl-aot/strata` 或 `build-sycl/strata`。用顶层
选项时引擎在 `build-sycl/sycl/strata`，所以要么改用 `-S sycl`，要么设置 `STRATA_SYCL_BIN`。

## Setup 与运行
<a id="setup-and-running"></a>

```sh
./setup.sh --backend sycl [setup's usual options, e.g. --model IQ2_XS --context 32768]
```

这会打印实验性警告，然后继续用 `sycl/setup_intel.py`（在 setup 自己的 Python 环境里）。
该脚本在 sysfs 里找到 Arc，使用你构建的 SYCL 引擎，并写出配置和 `run-<model>.sh`。
显卡小于 12 GB 或者是 `i915` 的卡（A 系列）时，它写出的是小卡配置（[INTEL.md](INTEL.md) 的「Arc A750
and the other Alchemist cards」）。模型的下载和打包仍按原来的流程走。要手工运行引擎（不用 Docker），见
[INTEL.md](INTEL.md) 里的「How to run it by hand」。

在 Arc 上要留意的几件事（细节见 INTEL.md）：

- **不要申请超过显卡显存的空间。** 在 `xe` 驱动上，超出显存的分配会把 buffer 挤进
  RAM，直到机器内存耗尽而卡死。留出约 1.5 GB 空闲。
- `SYCL_CACHE_PERSISTENT=0`：持久化的 JIT 缓存在 Xe2 上首次编译时崩了。
- 两张卡：`ONEAPI_DEVICE_SELECTOR=level_zero:*`（镜像里钉的是 `level_zero:0`；`strata-sycl.sh` 现在会把这个
  变量透传过去）加 `--layer-split`。
- **Arc Pro B60：** PCI id 是 `8086:e211`（`lspci -nn`，内核的 `xe` id 表把它归在 BMG-G21 的卡里），`sycl-ls` 打印
  `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`，`ocloc ids bmg-g21` 打印 20.1.0，所以 `-DSTRATA_SYCL_AOT=bmg-g21` 就是正确的目标。
  `setup_intel.py` 认识 B60 的两个 id，`e211` 和 `e221`（B60 卡上都见过）。`--layer-split` 时，启动行「N experts are neither in VRAM nor mirrored」
  把另一张卡的层也算进去了；要读的是随后那几行 `100% of the experts resident`。那种情况下设 `STRATA_MIRROR_MIB=0`，
  否则页锁定的镜像会被分配却用不上。
- **`setvars.sh` 与 `set -u`：** 在开了 `set -u` 的 shell 里执行 `source /opt/intel/oneapi/setvars.sh` 会停在 `OCL_ICD_FILENAMES: unbound variable`
  （oneAPI 2026.1.1）。在 `set -u` 之前 source，或者在它外面先 `set +u`。
- **不用 Docker：** `sycl/serve/strata-sycl.sh` 需要 Docker。原生方式是 `source setvars.sh`，导出 `SYCL_CACHE_PERSISTENT=0`、
  `ZES_ENABLE_SYSMAN=1` 和 `STRATA_VERIFY_DEVICE_PLAN=1`，然后运行 `build-sycl-aot/strata`，或者用 `sycl/serve/server_intel.py` 配一个做了这些事的 `exe`。
  在 Ubuntu 上普通用户不在 `render` 组里，此时 `sycl-ls` 只显示 CPU 设备。
- `STRATA_VERIFY_NO_HOST=1`（由 `strata-sycl.sh` 设置，除非配置里的 `"env"` 另有说明）在 `xe`
  卡上一部分专家层不在显存里时需要（那里看不到 host-to-GPU 的 flag 存储，否则 logits 会是
  NaN），在 `i915` 的 Arc A 系列上则保持关闭：见 [INTEL.md](INTEL.md)。配置里的 `"env"` 块和服务器环境里任何
  `STRATA_*`、`ONEAPI_*`、`UR_*`、`IGC_*`、`SYCL_*` 或 `ZES_*` 变量都会通过 `strata-sycl.sh` 到达引擎
  （例如 `STRATA_MIRROR_MIB`、`IGC_EnableDPEmulation`）。

## Windows
<a id="windows"></a>

还没有 Windows 这条路。Windows 上的 `setup --backend sycl` 会停下并指到这里。oneAPI 有 Windows 版，
但 `sycl/CMakeLists.txt` 用的是 GCC 风格的参数（`-mavx512f`、`-fp-model=precise`、`-qmkl`），而运行器是
bash/Docker 脚本，所以要做原生 Windows 构建还得干活。没有人试过。有一位测试者用过带 Arc 的 WSL2
（就是上面 B580 那一行），但 setup 在那里检测不到显卡，因为它读的是 `/sys/class/drm`，
而 WSL2 没有这个。

## 报告问题
<a id="reporting-a-problem"></a>

开一个 issue，附上：显卡、驱动版本、`sycl-ls` 输出、oneAPI 版本、模型和参数，以及
引擎的 stderr。来自真实显卡的结果才能把它从实验性推进到正式支持。
