# 安装 Strata
<a id="installing-strata"></a>

关于在 Windows、Linux 和 Mac 上、使用 NVIDIA 或 AMD 显卡或 Apple Silicon Mac 安装、启动、更新和配置 Strata 的一切。
简要版本在 [README](../../README.md#install)。AI 编程助手可以按 [AI_SETUP.md](AI_SETUP.md) 替你完成全部这些操作。

安装方式有三种。除非有理由不这么做，否则请选第一种：

1. **[桌面应用](#the-app)：** 一个内置引擎的桌面程序。安装它，选一个模型，聊天。
2. **[终端版](#the-terminal)：** 在终端里敲一行命令，然后 `strata launch`。适用于脚本、服务器和 AI 代理。
3. **[从源码安装](#from-source-windows)：** Windows 上运行 `START-HERE.bat`，Linux 和 Mac 上运行 `./setup.sh`。
   它会自行构建或获取引擎。

> **本页内容：** [你需要什么](#what-you-need) · [桌面应用](#the-app) · [终端版](#the-terminal) ·
> [从源码安装：Windows](#from-source-windows) · [Linux](#from-source-linux) · [Mac](#from-source-mac) ·
> [AMD 显卡](#amd-cards) · [两到三块卡](#two-or-three-cards) · [Docker](#docker-linux) ·
> [较老的 CPU](#older-cpus-experimental) · [更新](#updating) · [文件存放位置](#where-things-are-stored) ·
> [安装程序的提问](#setups-questions) · [为你的 PC 调优](#tuning-for-your-pc) · [全部选项](#options-without-questions)

## 你需要什么
<a id="what-you-need"></a>

| | |
| --- | --- |
| **GPU** | **NVIDIA** RTX 20、30、40 或 50 系列，**显存 12 GB 或以上**（8 GB 也能运行，但很慢）。在 RTX 5070 和 RTX 3090 上实测；RTX 20（Turing，自 0.1.27 起支持）由一位贡献者在 RTX 2070 上测试过。**AMD** Radeon RX 7900 XT / XTX、RX 9070 / 9070 XT 和 Radeon AI PRO R9700（已验证），RX 7800 XT / 7700 XT 和 RX 9060 XT（由其所有者验证），RX 6800 / 6900 系列（社区报告），显存 12 GB 或以上。见 [AMD 显卡](#amd-cards)。 |
| **内存** | 够你选择的规模即可（[选哪个模型](MODELS.md#pick-by-memory)）；**64 GB** 能跑所有规模。大显卡可以弥补内存不足——[低内存模式](MODELS.md#a-big-graphics-card-and-little-ram)。 |
| **CPU** | 支持 AVX2 的 x86-64（近 ~8 年内的任何 Intel/AMD 桌面 CPU）。AVX-512（Ryzen 7000/9000）稍快一些。不支持 AVX2 的较老 CPU 属实验性支持且很慢：[较老的 CPU](#older-cpus-experimental)。 |
| **磁盘** | 应用及其引擎约占 1-3 GB。模型需要约 70-80 GB 空闲空间，MTP 层约 6 GB（带图片则 +1 GB）。在 AVX-512 CPU 上使用 Q2_0 时，还会一次性写入约 40 GB 的专家副本供快速 CPU 内核使用。在 Linux 上使用 AMD 显卡时，若安装程序安装 ROCm，会额外占用约 10 GB。强烈建议使用 NVMe SSD：它能让首次启动快得多。 |
| **操作系统** | Windows 10/11、Linux（Ubuntu 22.04/24.04 和 Debian 12/13 会自动安装全部依赖；应用也能在 Fedora 44 上运行），或使用 Apple Silicon（M1 或更新）且 macOS 15 或更新的 Mac（[Mac](../MAC.md)）。 |
| **驱动** | **NVIDIA：** 当前驱动，版本 580 或更新（[nvidia.com/drivers](https://www.nvidia.com/drivers) 或 NVIDIA App）。**AMD：** Linux 上用内核自带的 amdgpu 驱动（无需安装 ROCm）；Windows 上用当前的 AMD Software: Adrenalin Edition 驱动（[amd.com/support](https://www.amd.com/en/support)）。 |

驱动是唯一需要你自己安装的东西。使用桌面应用时，引擎就在安装程序里。使用终端版时，`strata launch` 会从签名的更新源
为你获取适配你显卡的引擎，不编译任何东西。从源码安装则会在首次运行时把其余部分也准备好：如果你没有 Python 3.12 就
安装一个（仅针对你的用户账户，无需管理员）、`.venv/` 中的私有 Python 环境、Strata 引擎、模型和 MTP 草稿层。
在 NVIDIA 上它使用为 RTX 20/30/40/50 预备好的引擎以及来自 pip 的 NVIDIA CUDA 库（约 0.4 GB）；如果没有现成的引擎
适配你的电脑，它会提出安装构建工具（Windows 上是 Visual Studio Build Tools + CUDA Toolkit，Ubuntu 上是
`build-essential` + CUDA）并为你的 GPU 编译引擎（会先征求同意；一次性耗时 20-40 分钟）。更多：[细节](DETAILS.md#before-you-start)。

常规的 Python 安装使用 `requirements.txt`，不会安装 CMake 或 Ninja。如果安装程序必须编译引擎或图像编码器，它会优先
使用已有的 CMake/Ninja 工具，并把 `requirements-build.txt` 中缺失的固定版本辅助工具安装到 `.venv`。旧的 `.venv` 会
保留其已有的构建辅助工具；更新期间安装程序不会移除任何软件包。

## 桌面应用
<a id="the-app"></a>

桌面应用是最简单的方式。引擎内置其中，因此无需构建，也没有安装窗口。

1. 从 [releases 页面](https://github.com/Niko1221/Strata/releases/latest)下载适合你系统的安装程序。
   页面上会注明哪个安装程序对应哪种显卡（NVIDIA 或 AMD）。选择与你显卡匹配的那个。
   - **Windows 10 / 11：** 运行 `.exe`。它为你的用户安装，不需要管理员权限。Windows 可能会提示
     "Windows protected your PC"，因为安装程序尚未进行代码签名。选择**更多信息**，然后点击
     **仍要运行**。
   - **Linux：** `.AppImage`。使其可执行（右键、属性、权限，"Allow executing file as program"，或 `chmod +x`），
     然后双击运行。它需要 FUSE 2：Ubuntu 22.04 上 `sudo apt install libfuse2`，Ubuntu 24.04 及更新版本上
     `sudo apt install libfuse2t64`，Fedora 上 `sudo dnf install fuse fuse-libs`。没有 FUSE 时，仍可在文件名后加
     `--appimage-extract-and-run` 来启动。NVIDIA 构建需要驱动 580 或更新。
   - **Mac：** `.dmg`。打开它，把 Strata 拖到 Applications，然后从那里打开 Strata。它需要 Apple Silicon 和
     macOS 15 或更新。如果 macOS 提示应用"已损坏"，文件其实没有问题：见 [MAC.md](../MAC.md)。
2. 启动 Strata。在 Linux 和 Mac 上首次启动会稍慢一些，因为应用内的引擎会被复制到你的主目录。
3. 第一个界面会读取你的 PC 配置，并推荐一个适合你的模型。按下安装按钮。下载体积很大（40-70 GB）。
   如果中断，它会从断点继续。
4. 聊天。**关闭窗口会同时停止 Strata 和引擎。**

**更新：** Settings，然后 Updates，然后 **Check for updates**。按钮 **Update Strata** 和 **Update engine**
只在新版本出现时才会显示。你的模型和已保存的设置都会保留。请先卸载模型：正在运行的引擎无法被替换。
同一处的 **Roll back** 可以恢复上一个引擎。

**文件位置：** 应用的设置和聊天记录在你的用户文件夹下的 `.strata` 文件夹中，引擎在
`%LOCALAPPDATA%\Strata\components`（Linux 和 Mac：`~/.local/share/strata/components`），模型在你的文档目录下的
`Strata` 文件夹中（你可以在应用里更改模型文件夹）。移除应用不会删除它们。**提交 bug 报告时：** Settings、Engine、
Diagnostics、View logs，然后 **Save logs**。

Mac 上的应用暂不支持图片读取。其余与图片相关的内容见[安装程序的提问](#setups-questions)。

## 终端版
<a id="the-terminal"></a>

终端版就是 `strata` 命令。它与桌面应用使用相同的引擎、模型和已保存设置，是脚本、服务器和 AI 代理的使用方式。
支持 Linux（x86-64）、Mac（Apple Silicon）和 Windows（x64）。

Linux 和 Mac：

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows（PowerShell）：

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

安装脚本会识别这台计算机的型号，下载终端版压缩包，校验其 SHA-256，将其解压到你的主目录
（`~/.local/share/strata/terminal`，Windows 上为 `%LOCALAPPDATA%\Strata\terminal`；无需管理员权限），
并用刚解压出的终端版校验该发布版的签名更新源（如果该校验失败，它会删除已解压的内容），然后把 `strata` 加入你的
PATH。在 Windows 上，之后请打开一个新的终端窗口。然后：

```sh
strata launch                                 # guided screens: model, context, pictures, who may connect, then serve
strata launch --model q2_0 --yes --no-tui    # the same without questions (scripts, agents)
```

`strata launch` 会下载适配你显卡的引擎（经过签名，并在投入使用前先试启动验证）和模型，在 `http://127.0.0.1:8080`
上开始提供服务并显示监控界面。除非你加上 `--open`，它绝不会自动打开浏览器。它的标志和界面：
[TERMINAL_LAUNCH.md](../TERMINAL_LAUNCH.md)。

**更新：** `strata update --check` 说明将要发生什么变化，`strata update` 安装新引擎（它会先试启动，只有成功才切换；
上一个引擎会保留），`strata components rollback` 回退。正在服务模型时它会拒绝执行。
**检查你的 PC 和已安装的引擎：** `strata doctor`。

想在运行安装脚本前先读它，就先下载（`curl -fsSL <address> -o install.sh`），读完再运行 `sh install.sh`。

## 从源码安装：Windows
<a id="from-source-windows"></a>

1. [下载本项目](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)并解压（或 `git clone`）。
2. 双击 **`START-HERE.bat`**。
3. 回答几个问题——或者每次都直接按 Enter 采用推荐选项（[这些问题](#setups-questions)）。

然后它会下载全部内容（模型约 70 GB，所以首次要花一阵——你可以中断，它会从断点继续）并**启动模型**。
你的浏览器会打开 `http://127.0.0.1:8080` 上的 Strata 应用。

> **模型启动期间，你的 PC 可能变慢或停止响应 1-3 分钟**（首次最久）：Strata 会把 35-55 GB 载入你的内存并为其
> 锁定一部分供显卡使用。这是正常的——请等待，不要关闭窗口。窗口会告诉你它在做什么。

**下次**只需再次双击 `START-HERE.bat`：它会立即启动（加载模型 30-90 s），不会重复下载任何东西。关闭它的窗口即停止
模型。`SETUP.bat`（等同于 `START-HERE.bat --setup`）用于安装另一个模型或更改设置。用任务计划程序在登录时启动 Strata
需要两项任务设置，否则启动会慢 24 倍：[在开机时运行它](DETAILS.md#running-it-at-startup-task-scheduler)。

## 从源码安装：Linux
<a id="from-source-linux"></a>

```bash
./setup.sh
```

同样的问题、同样的自动安装（它用 `sudo apt` 安装 Python，并且只有在必须编译时才安装构建工具），同样的启动：
`http://127.0.0.1:8080`。之后再次运行 `./setup.sh`（或 `./run-<model>.sh`）会直接启动模型；`./setup.sh --setup`
用于安装另一个模型或更改设置。其他发行版、WSL 和编译：[细节](DETAILS.md#linux)。

在 Nix/NixOS 上，你可以使用 flake：

```bash
nix develop -c ./setup.sh
```

## 从源码安装：Mac
<a id="from-source-mac"></a>

在使用 Apple Silicon、macOS 15 或更新的 Mac 上，先安装一次 Apple 的 Command Line Tools（`xcode-select --install`），
然后在 Strata 文件夹里运行 `./setup.sh`。它会在你的 Mac 上构建 Mac（Metal）引擎并提出同样的问题。桌面应用和终端版
都不构建任何东西。全部内容，以及哪种内存规模装得下什么，见 [MAC.md](../MAC.md)。

## AMD 显卡
<a id="amd-cards"></a>

桌面应用和终端版会自行是为 Radeon 显卡挑选 AMD 引擎。从源码安装的步骤与 NVIDIA 相同：Windows 上 `START-HERE.bat`，
Linux 上 `./setup.sh`。在一台没有 Strata 可用 NVIDIA 显卡的 PC 上，setup 会发现 Radeon 显卡并自行选择 AMD（HIP）引擎；
两种卡都有时用 `--backend hip` 选择它。受支持的显卡：RX 7900 XT / XTX（gfx1100）、RX 7800 XT / 7700 XT（gfx1101）、
RX 9060 XT（gfx1200）、RX 9070 / 9070 XT 和 Radeon AI PRO R9700（gfx1201），以及 RX 6800 / 6900 系列（gfx1030）。
集成 Radeon 显卡会被列为不受支持。

在 Linux 上，setup 会使用系统的 ROCm 7（如果存在），否则从 AMD 的 wheel 把 ROCm 装进 `.venv`（约 10 GB，无需 sudo），
并在你的 PC 上为该显卡编译引擎（10-20 分钟，一次性；需要一个 C++ 编译器和 git：`sudo apt install build-essential git`）。
多块 AMD 显卡用 `--gpus` 共享模型，与 NVIDIA 相同。

目前与 NVIDIA 的不同之处：`--calibrate` 仅限 NVIDIA，且 Unsloth 的 4-bit 模型需要 NVIDIA 显卡。各显卡的实测数据、
手工构建和调优表：[AMD_HIP.md](AMD_HIP.md)。

## 两到三块卡
<a id="two-or-three-cards"></a>

**两到三块 NVIDIA 卡？** 直接运行 `START-HERE.bat`：它会列出你的显卡、说明哪些是 Strata 能用的，并询问是否把模型
跨它们共享（两块可用时推荐这么做）。在单块卡上完成的安装会在下次启动时询问一次。或者自己选择：
`START-HERE.bat --gpus 0,2`（两块，记住选择）、`--gpus all`，或 `--gpu 0`（单块卡，仅本次启动）。每块卡保留其所在层
的专家层，提示词以流水线方式流过各块卡：在 RTX 5080 + RTX 3090 上，提示词读取比单独用 5080 快 18-20%，解码持平。
每块卡都必须是 RTX 20 系列或更新且 8 GB 或以上。见 [MULTI_GPU.md](MULTI_GPU.md)。

**多块 AMD 卡：** setup 默认取一块卡（显存最多的那块，或 `--gpu N`），除非你指名多块：`--gpus 1,0` 把模型的层切分到
它们上，第一块是主卡（编号按 setup 列出的顺序）。只有当没有单块卡能装下模型的专家层时，切分才划算
（[AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201)）。

## Docker（Linux）
<a id="docker-linux"></a>

为 NVIDIA（CUDA 13 和 CUDA 12）、AMD（RX 9000 / R9700 和 RX 7000）和 Intel Arc 显卡预备好的镜像，内含终端版：
`docker run` 一条命令，模型在首次启动时下载进一个 volume，你的计算机上不编译任何东西。哪种卡用哪个镜像、命令、
Docker Compose、API key、更新与故障排查：[DOCKER.md](../DOCKER.md)。从源码构建：`docker build -t strata:src .`
（CUDA 13；[DOCKER.md](../DOCKER.md#build-it-yourself)）。

## 较老的 CPU（实验性）
<a id="older-cpus-experimental"></a>

Strata 的现成引擎需要 AVX2（Intel Haswell 2013、AMD Zen 2017 或更新）。自 0.1.39 起，较老的 CPU——只有 AVX
（Sandy Bridge / Ivy Bridge、Xeon E5 v1/v2、AMD Bulldozer）或只有 SSE4.2（Nehalem / Westmere、Xeon X5600）——
可以以 setup 在该 PC 上编译的**实验性**构建运行。照常运行 setup：它会说明该 CPU 没有 AVX2，警告这是实验性且很慢的，
并编译引擎（10-20 分钟，一次性；Windows 需要 Visual Studio Build Tools 和 CUDA Toolkit，setup 会提出安装）。
它不会中止。现成引擎仍然会拒绝这样的 CPU，并说明原因。

该构建的不同之处（`STRATA_ISA_FLOOR=avx` 或 `none`，依 CPU 选定）：

- ggml-cpu 为 AVX 或 SSE4.2 而非 AVX2 编译，且专家层的 CPU 部分跑在其内核上。Strata 自己的 AVX2 / AVX-512 内核
  仍保留在程序里，但只在具备这些指令集的 CPU 上使用。
- 只有 i-quant 模型的模型包能运行（setup 在这样的 CPU 上安装的每个模型都是）；AVX-512 的 Q2_0 模型包不行。
- 速度：GPU 部分不变而 CPU 部分更慢，所以显存更多的 GPU（在其上缓存更多专家层）帮助最大。在 Ryzen 5 7600 +
  RTX 5070 上强制启用较老路径、并把专家缓存固定在 1500 槽位时，贪心 decode 从 26 变为 AVX 构建的 11-17 tok/s、
  SSE4.2 构建的 3.5-3.8 tok/s（Coder 和 IQ3_XXS，各 10 个提示词，各跑一次）。一颗配 DDR3 的老 Xeon 会比这些更慢。
- 我们未在真实的老 CPU 上测试过。这里是在 Ryzen 5 7600 上强制启用较老路径来检查的（`STRATA_ISA=avx` 或 `sse`）：
  答案合理并通过了引擎的检查，但与常规构建并非逐 token 相同（ggml-cpu 对 CPU 专家层的舍入不同，正如 AVX2 内核与
  AVX-512 内核之间那样）。贡献者们把同一思路的更早版本在 Xeon E5-2680（配 GTX 1080 Ti 为 2.87 tok/s）以及
  Xeon E5-2687W / X5690 机器上运行过。
- AMD 显卡：仅限 Linux（setup 在那里本来就要编译 AMD 引擎）；NVIDIA 在 Windows 和 Linux 上都可以。
- 双路 Xeon：另一插槽上的内存很慢；如果可能，把内存保持在一颗 CPU 上。

手工方式：`cmake ... -DSTRATA_ISA_FLOOR=avx`（或 `none`）即可构建；为 setup 设置 `STRATA_ISA_FLOOR=avx` 就能在任何
PC 上构建它（用来试用）。真实硬件上的问题和结果欢迎作为 GitHub issue 提交。

## 更新
<a id="updating"></a>

**桌面应用：** Settings、Updates、**Check for updates**，然后 **Update Strata** 或 **Update engine**
（见[桌面应用](#the-app)）。

**终端版：** `strata update --check`，然后 `strata update`（见[终端版](#the-terminal)）。加上 `--app` 会同时下载
应用的安装程序。更新从不下载、替换或删除模型和已保存的调优预设。签名的更新源如何工作：
[packaging/UPDATES.md](../../packaging/UPDATES.md)。Strata 下载什么以及如何校验：[SAFETY.md](SAFETY.md)。

**从源码安装：`UPDATE.bat`**（Linux：`./update.sh`）更新 Strata 而不启动模型——适用于 GPU 正忙于别的事，或你只想把
新版本准备好。在 `git clone` 中它运行 `git pull`，然后做 `START-HERE.bat` 在启动前做的事：引擎（新版本需要时取新的
现成引擎；在 Linux 上，已编译引擎的源码变化时会重新编译）、Python 包，以及每个已安装模型的设置和草稿子集。
模型文件不会被改动（最多下载一个新引擎），也不会提问。请先关闭模型的窗口（正在运行的引擎无法被替换）；稍后照常
用 `START-HERE.bat` 启动模型。在作为 zip 下载的副本里，它会提示下载新的 zip（见下）：它无法自行获取新文件。

或者手工：下载新版本并解压到任意位置（或 `git pull`），然后在其中运行 `START-HERE.bat`（Linux：`./setup.sh`）。
模型文件保存在你的 Strata 文件夹旁边的 `Strata-data` 文件夹中，因此新副本会找到它们并以同样的方式完成配置——
不会重新下载任何大文件。在 Linux 上 `git pull` 之后，若引擎源码有变，setup 会重新编译它（变化的文件需要一两分钟）；
如果那次编译失败，它会说明并启动你原有的引擎。

## 文件存放位置
<a id="where-things-are-stored"></a>

- **你的聊天记录：只在你的浏览器里。** Chat 标签页把对话、其设置和你输入的 API key 保存在浏览器的本地存储中
  （`strata.*` 键）——不在服务器上，也不在 Strata 文件夹里。图片不会被保留，只保留其名称。另一个浏览器或隐私窗口
  打开时是空的；清除站点数据会删除它们。
- **模型如何启动：** Strata 文件夹中的 `strata-<model>.json`（上下文、GPU、host、API key……），由 setup 写入；
  旁边是 `run-<model>.bat` / `.sh`、日志 `strata-<model>.log`，以及当你使用 "Use for other apps too" 时的
  `strata-<model>.shared-settings.json`。对同一模型再次运行 setup（例如换上下文）会重写 setup 写入的键
  （`exe`、`args`、`port`、`gpu`、`host`、`api_key`、`vision`……）并保留你自己添加的键（`sampling`、`mcp_servers`、
  `mcp`、`cors_origins`……）；较早的文件会保留为 `strata-<model>.json.bak`（0.1.39）。你手工加到 `"args"` 里的引擎
  选项不会被带过去：setup 会列出它们，需要你再添加一次。
- **模型文件**（`models/`、`packs/`、`mtp/`，70-120 GB）：在 **Strata 文件夹旁边的 `Strata-data`** 中，
  或 `--data-dir` 指定的位置。
- **那个数据文件夹在哪里：** Windows 上是 `%APPDATA%\Strata\settings.json`，Linux 上是
  `~/.config/strata/settings.json`（[细节](DETAILS.md#before-you-start)）。
- **程序本身：** Strata 文件夹中的 `.venv/`、`engine/` 和 `third_party/`。
- **无法访问 github.com 时：** 从源码构建会按 `setup.py` 中写明的提交 `LLAMA_CPP_COMMIT` 从 GitHub 下载 llama.cpp 的
  源码（ggml、gguf-py）。你可以自己把该提交的树放到 `third_party/llama.cpp`（来自镜像或另一台电脑），setup 会改用它
  （#585）。

## 安装程序的提问
<a id="setups-questions"></a>

以下是从源码安装程序（`START-HERE.bat`、`./setup.sh`）的问题。应用的首屏和 `strata launch` 会用它们自己的样式问
同样的事情。

- **哪个模型和规模？** 原版、Swift 1.5、Coder 或 Unsloth 的 4-bit，以及 Q2_0、IQ2_XS、IQ3_XXS 或 IQ3_S ——
  见[选哪个模型](MODELS.md)。
- **多少上下文？** 它一次能记住多少文本（它会为你的显卡建议一个）。384K 和 512K（实验性）通过 rope scaling 把模型
  扩展到其训练的 262K 之外——setup 会自行开启它（yarn 和一个覆盖因子；`--rope-scaling`/`--rope-scale` 可覆盖）
  （[细节](DETAILS.md)："Context extension past 262K"）。
- **图片？** 是否也应该读取图片。对能读取图片的模型，图片默认开启。图像编码器（0.9 GB）在你的显卡少于 24 GB 时跑在
  CPU 上（每张图片约 3-13 s，不从 GPU 占用任何东西），从 24 GB 起跑在 GPU 上（每张图片 0.1-0.5 s，为其保留约
  1.4 GB 显存）。NVIDIA 和 AMD 显卡规则相同；`--vision gpu` 和 `--vision cpu` 由你自己选择读取方式，`--vision no`
  关闭图片。在 Radeon AI PRO R9700 上，GPU 读取一张 260-token 的图片耗时 57 ms，CPU 读取耗时 3.1 s。
  Mac 暂无图片读取器。[细节](DETAILS.md#images-vision)。
- **实验性速度投影？** 除非你同意否则关闭——请先[阅读它做什么](DETAILS.md#experimental-speed-projection-experimental-off-by-default)。
  它会改变模型的回答方式，且只有原版模型提供它。

加上 `--yes` 时，setup 对每个问题都采用推荐答案。

## 为你的 PC 调优
<a id="tuning-for-your-pc"></a>

每台 PC 都不同：`START-HERE.bat --calibrate`（Linux：`./setup.sh --calibrate`）会在你的机器上实测若干引擎设置并保留
最快的（约 15-30 分钟，慢卡上更久；在配 Ryzen 5 7600 的 RTX 5070 上它让 Coder 快了 7%）。只有当某设置快超过 3% 时
才保留，结果按 PC 和模型记住，因此更新会保留它。测量 CPU worker 数量需要一个全新的引擎，所以模型会被加载多次：
期间 PC 很忙，并可能在每次重启时停止响应一两分钟。完成后它会**启动模型**，因此返回时服务器已在运行——不要再启动
第二次。目前仅限 NVIDIA 显卡。[它测量什么](DETAILS.md#double-click-start-herebat)。

## 不提问的选项
<a id="options-without-questions"></a>

对于终端版，`strata launch --help` 列出每个标志；最常用的在 [TERMINAL_LAUNCH.md](../TERMINAL_LAUNCH.md)。
以下是从源码安装程序的选项：

```
START-HERE.bat --setup                          install another model, or change context / images
SETUP.bat                                       the same (double-click it)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --gguf-dir D:\models\IQ2_XS       use GGUF files you already have
START-HERE.bat --inspect D:\models\some.gguf     what a GGUF really holds and whether Strata runs it (no download)
START-HERE.bat --data-dir E:\Strata-data         keep the model files somewhere else
START-HERE.bat --port 8081                      another port
START-HERE.bat --gpu 1                          another GPU (setup picks the one with the most VRAM)
START-HERE.bat --gpus 0,2                       several GPUs sharing the model
START-HERE.bat --vram-reserve-mib 2048          leave 2 GB of VRAM free for other programs (remembered)
START-HERE.bat --no-browser                     do not open the chat page when the model is ready (remembered;
                                                --browser undoes it)
START-HERE.bat --setup --backend hip            the AMD engine on a PC that also has an NVIDIA card
START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>     reachable from other devices, with a key
START-HERE.bat --calibrate                      tune the engine for this PC (about 15-30 minutes, longer on a slow card), then start
START-HERE.bat --check                          only check this PC
```

**为其他程序留出显存（#493）：** Strata 用专家层（专家缓存）填满显卡的空闲显存，并留出 `--vram-reserve-mib` MiB 空闲：
默认 700。为了在它旁边跑游戏、3D 程序或另一个模型，请留更多：`START-HERE.bat --vram-reserve-mib 2048`
（Linux：`./setup.sh --vram-reserve-mib 2048`）会写入模型的 `strata-<model>.json` 并启动它；在 setup 时
（`--setup --vram-reserve-mib 2048`）它会进入新配置。手工方式：把 `"--vram-reserve-mib", "2048"` 加到配置的
`"args"` 列表并重启。专家缓存就会小那么多，因此回答可能稍慢。引擎按其启动时的空闲显存确定缓存大小：那时其他程序
已经占用的部分本来就不会去动；保留量是为它之后要占用的部分留的空间。

**同时驱动 Linux 桌面的 AMD 显卡（#560 #516）：** 用默认保留量时专家缓存会填满显卡，当桌面（合成器、浏览器、
新应用）需要更多显存时，amdgpu 驱动会把 GPU 内存搬到系统内存——而模型的专家层已经把它填满：随后 OOM killer 会结束
桌面会话（KWin、plasmashell），或合成器失败（"Failed to pin framebuffer"）。两份报告都用
`./setup.sh --vram-reserve-mib 3072` 解决了（显存中少约 2.3 GB 专家层，百分之几的速度）。在这样的 PC 上，setup 和
服务器窗口会说明这一点。

**8 GB 以下的显卡（#496）：** 当默认保留量给专家缓存留的空间太少时，引擎会降低保留量（直至 300 MiB）直到装得下，
并在显卡因此接近装满时警告。如果启动停在 "no VRAM is left for the expert cache"，那行日志会说明还差多少；
8K 上下文和 setup 中的 `--draft-vocab en` 能腾出最多空间。

**手工下载的模型文件，或来自镜像（#495）：** setup 的第 5 步会打印它期望文件所在的文件夹
（`Strata-data\models\<SIZE>\`，例如 `Strata-data\models\IQ3_XXS\`）：按原始名称放到那里，或用 `--gguf-dir` 让 setup
指向它们。想在下载或使用前看清一个文件到底装了什么，`--inspect` 只读取它的头部（几 MB，也可跨网络：一个文件、
一个文件夹、一个 URL，ModelScope 用 `ms:owner/repo`，或 `hf:owner/repo`）：每个权重组的真实每权重 bit 数与存储类型，
以及它是 setup 安装的文件之一（无论叫什么名字）、与其中某个存储方式相同，还是 Strata 跑不了的文件（另一种架构，
或专家层用的是没有任何 Strata 内核能读取的格式）。名称说明不了什么：Unsloth 的 `UD-IQ3_XXS` 其路由专家层是每权重
3.22 bits，`UD-Q4_K_XL` 是 5.10，原版 `Q2_0` 是 2.25，`IQ3_S` 是 3.33。想让 setup 自己从 Hugging Face 镜像下载，
先设置 `HF_ENDPOINT`（Windows：`set HF_ENDPOINT=https://hf-mirror.com`，Linux：`export HF_ENDPOINT=https://hf-mirror.com`）：
同样的固定版本修订与校验依然适用，MTP 草稿层也来自那里。

**来自 ModelScope（中国大陆）：** setup 下载的每个仓库（GSQ-RCO 模型、Swift 1.5、Coder、Unsloth 的文件，以及取自
其原始 checkpoint 的 MTP 草稿层）都在 [ModelScope](https://www.modelscope.cn) 上以同名存在，文件相同。setup 从不
自行切换到它（失败的下载会说明怎么做）：使用 `--source modelscope`（`STRATA_SOURCE=modelscope`）。ModelScope 提供
仓库的当前文件（没有固定版本修订），因此每个文件都对照 ModelScope 为其公布的 SHA-256 校验（自证性质：一个被改动的
上游仓库会带上它自己的哈希），MTP 张量则对照固定版本 checkpoint 自身的哈希校验，与来自 Hugging Face 时相同；
选择它时 setup 会说明。2026-10-05 在中国大陆的一台 PC 上实测：11-14 MB/s；Q2_0 和 IQ3_S 的文件与其公布的 SHA-256
匹配，且 Q2_0 的分片 1 与来自 hf-mirror.com 的副本一致。

在 Linux 上同样的选项传给 `./setup.sh`。`START-HERE.bat --help` 列出全部选项。服务器自身的设置（与游戏共享 GPU、
MCP 工具、CORS、API key、API 本身）在[详细说明](DETAILS.md#using-it)里。
