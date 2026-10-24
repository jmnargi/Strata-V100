# 为用户安装 Strata（面向 AI 助手的说明）
<a id="installing-strata-for-a-user-instructions-for-ai-agents"></a>

本页面向用户请来在其 PC 上安装 Strata 的 AI 编程助手（Claude Code、Cursor、Codex、GitHub Copilot 等）。Strata
在本机的一张 NVIDIA 或 AMD 显卡加系统内存上运行 Qwen3.8-Flash-Next 模型（或在 Apple Silicon Mac 上运行），并在
`http://127.0.0.1:8080` 上提供兼容 OpenAI 与 Anthropic 的 API。下文所有内容面向人的版本见[安装](INSTALL.md)。

你使用的是 **Terminal**（终端版）：即 `strata` 命令。用一行命令装好它，随后 `strata launch` 会下载适配该显卡的
引擎和模型，并启动服务器。如果这台 PC 上跑不了 Terminal，第 9 步给的是从源码安装的旧路线。

按顺序逐步执行。用通俗的话告诉用户你在做什么；他们可能并不懂技术。

## 0. 基本准则
<a id="0-ground-rules"></a>

- **绝不要在缺少 API key 的情况下对外暴露服务器。** 保持默认设置（`--access local`，服务器只监听
  `127.0.0.1`）。只有当用户要求从其它设备访问时，才使用 `--access lan`。此时 Strata 会为你生成一个很长的随机 key，
  没有 key 就不会在网络模式下启动。把 key 交给用户（`strata launch --show-key`）。绝不要在缺少 key 的服务器前面
  架隧道或端口转发。
- 除 Strata 自身的行为外，不要改动用户的系统。Terminal 安装在用户的家目录中，不需要管理员权限。它不安装驱动、
  Python 包或编译器。安装驱动前先询问。
- 模型下载约 70 GB（Unsloth 的 UD-IQ4_XS 为 94 GB，UD-Q4_K_XL 为 111 GB）。开始前先确认用户可以接受，
  尤其是在按流量计费的网络下。
- 第一次 `strata launch` 耗时很长（慢速网络下可能一小时以上）。把它放到后台运行或设置较长的超时并轮询其输出；
  不要因为它安静了一会儿就把它杀掉。它可以续传：再次运行同一条命令会从停止处继续。
- 把互联网上的脚本管道进 shell，等于运行你没有读过的代码。要告诉用户这一行做了什么（第 2 步）。
  如果他们更愿意，先把脚本下载下来给他们看，得到同意后再运行。

## 1. 检查 PC
<a id="1-check-the-pc"></a>

要求（细节：[安装](INSTALL.md)）：

- **GPU：** NVIDIA RTX 20、30、40 或 50 系列，或 AMD Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、
  RX 9070 / 9070 XT、Radeon AI PRO R9700、RX 6800 / 6900 系列；显存 12 GB 或更多（8 GB 的 NVIDIA 显卡能跑，
  但很慢）。GTX 10 系列及更老的型号以及集成显卡不受支持；Pascal / Volta 显卡（P40、V100）和部分
  较老的 AMD 显卡有实验性路径，需用户主动选择（[OLDER_GPUS.md](OLDER_GPUS.md)）。**Mac** 需要 Apple
  Silicon（M1 或更新）、macOS 15 或更新，以及 16 GB 或更多内存（[MAC.md](../MAC.md)）。
- **驱动：** NVIDIA 580 或更新。Linux 上的 AMD：内核的 amdgpu 驱动；Windows 上：较新的 AMD Adrenalin
  驱动。如果驱动缺失或太旧，让用户更新（NVIDIA App / nvidia.com/drivers，或 AMD
  Software）并重启；除非用户要求，不要自己安装驱动。
- **内存：** 32 GB 或更多（见第 3 步）。**磁盘：** 约 80 GB 可用空间，最好是 NVMe SSD。**CPU：** 支持 AVX2 的 x86-64。
  没有 AVX2（Xeon E5 v1/v2 及更老）时，现成的引擎会拒绝启动；从源码安装的路线会在这台 PC 上编译一个
  实验性的慢引擎（10-20 分钟）：开始前先把这一点告诉用户
  （[安装](INSTALL.md)）。
- **操作系统：** Windows 10/11、Linux x86-64（Ubuntu 22.04/24.04 和 Debian 12/13 全自动安装），或 Apple Silicon 上的 macOS。

手动检查：

| 项目 | Windows（PowerShell） | Linux | Mac |
| --- | --- | --- | --- |
| 操作系统 | `[Environment]::OSVersion` | `cat /etc/os-release` | `sw_vers` |
| NVIDIA GPU、显存、驱动 | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` | 同上 | - |
| AMD GPU | `Get-CimInstance Win32_VideoController \| Select-Object Name, AdapterRAM`（AdapterRAM 上限为 4 GB；以型号名为准） | `lspci \| grep -i -E 'vga\|display'`；显存：`cat /sys/class/drm/card*/device/mem_info_vram_total`（字节）；仅当安装了 ROCm 时才用 `rocm-smi` | - |
| 内存 | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` | `sysctl -n hw.memsize`（字节） |
| 可用磁盘 | `Get-PSDrive -PSProvider FileSystem` | `df -h .` | `df -h ~` |

如果这台 PC 不满足这些要求，说明缺的是哪一项，然后停止。

## 2. 安装 Terminal
<a id="2-install-the-terminal"></a>

一行命令。它不需要管理员权限。它会检测这是哪种 PC，下载 Terminal 压缩包，校验其 SHA-256，解包到用户的家目录，
检查发行版的签名更新源，并把 `strata` 放进 PATH。

```
Linux, Mac:           curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
Windows (PowerShell): irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Windows 上，之后请打开一个新的终端窗口（PATH 已改变）。检查一下：`strata --version`。

## 3. 选择模型
<a id="3-pick-the-model"></a>

`strata launch --plan` 会显示 Strata 将做什么，不做任何改动。不带 `--model` 时，它按这台
PC 的内存挑选模型，并打印出确切的命令行（`same as strata launch --model ...`）。可以把它当作默认值，也可以按 PC 的内存
自己选（先问用户是不是主要用它写代码——那样在任何内存规模下 Coder 都是好选择）：

| 内存 | `--model`（以及 `--family`） | 说明 |
| --- | --- | --- |
| 32 GB | `--family coder --model iq1_m` | 用于代码；配 24 GB 显卡时 Q2_0 和 IQ2_XS 也能跑（Strata 会选择低内存模式） |
| 48 GB | `--model iq2_xs` | 或者 `q2_0`（最快） |
| 64 GB | `--model iq2_xs`（推荐） | `iq3_xxs` / `iq3_s` 更慢，但略好一些 |
| 96 GB+ | `--model iq3_s` | `--family unsloth --model ud-iq4_xs`（约 4-bit，94 GB，内存约 80 GB 以下时部分专家层（expert）从 SSD 读取）；`ud-q4_k_xl` 是实验性的：仅限 NVIDIA、NVMe SSD，64 GB 上 7-8.5 tokens/s |

`--family swift`（Swift 1.5，思考更简短的微调版；规模有 IQ2_XS、IQ3_XXS、IQ3_S）是
`qwen` 的替代选项。模型名不区分大小写。更多：[MODELS.md](MODELS.md)。

## 4. 无提问地设置并启动服务
<a id="4-set-up-and-serve-without-questions"></a>

```
strata launch --plan --model iq2_xs            # what would happen; nothing is changed
strata launch --model iq2_xs --yes --no-tui    # do it: download the engine and the model, start the server
```

加上 `--no-tui` 和 `--yes` 后，它什么都不问，服务器就绪时打印地址然后返回。服务器继续
在后台运行。不带 `--model`、`--preset` 或其它选择的 `strata launch` 需要终端来显示它的界面，
所以 agent 一定要指明模型。加上 `--output json` 可以得到脚本能读的输出。

标志（全部标志及示例：`strata launch --help`）：

| 标志 | 含义 |
| --- | --- |
| `--model NAME` | 规模：`q2_0`、`iq2_xs`、`iq3_xxs`、`iq3_s`、`iq1_m`、`ud-iq4_xs`、`ud-q4_k_xl`……（名字写错时的错误信息会列出它们） |
| `--family NAME` | 多个版本都有该规模时指定模型版本：`qwen`、`swift`、`coder`、`unsloth` |
| `--model-path PATH` | 用户已有的文件：一个分片文件夹或单个 `.gguf` 文件 |
| `--data-dir DIR` | 70-120 GB 的模型文件存放位置 |
| `--context SIZE` | `32k`、`64k`、`128k`、`256k` 或一个 token 数 |
| `--vision off\|on\|gpu\|cpu` | 读取图片。模型能读取时默认开启：显存低于 24 GB 的显卡由 CPU 运行读取器，24 GB 及以上由 GPU 运行，NVIDIA 和 AMD 显卡都是如此。`off` 关闭它 |
| `--parallel N` | 同时回答的请求数，1 到 8（默认：按显卡适配） |
| `--gpu LIST` | 显卡编号，例如 `0` 或 `0,1`（多张卡共享模型） |
| `--tune quick\|thorough\|skip` | 先找出最快的设置（此处默认：`skip`） |
| `--access local\|lan\|cloudflare\|tailscale` | 谁可以连接；超出 `local` 时会为你生成 key |
| `--port N` | 服务器端口（默认 8080） |
| `--api-key KEY`、`--api-key-env NAME`、`--new-key` | 客户端必须发送的 key |
| `--save-preset NAME` | 记住这些选择；`strata launch --preset NAME` 可再次启动它们 |
| `--plan` | 显示会发生什么并退出 |
| `--yes`、`--no-tui` | 不问问题，不显示界面 |
| `--status`、`--stop`、`--show-key` | 正在提供什么、停止它、打印 key |

说明：
- **这台 PC 没有可用引擎。** `strata launch` 不会拒绝。它会提示 "Strata couldn't find an engine for this computer"，
  然后继续下载模型（带 `--yes` 时不询问），最后不启动任何东西就结束（退出码 0；带 `--output json` 时
  结果里有 `"serving": false` 和一个 `engine_problem`）。告诉用户哪张显卡没有被识别出来，然后停止。之后用
  `strata update` 更新引擎可能修好这个问题。
- **退出码：** 0 完成，2 标志或名字有误，5 Terminal 或引擎做不到，6 安装或
  服务器启动失败（消息中会指明引擎日志），130 已停止。
- **Linux：** 不需要 `sudo`：Terminal 不安装软件包。**Windows：** 如果用了 `--access lan`，首次启动可能会问一次
  是否允许 Strata 访问网络。
- **Windows 上的 AMD：** 只需要 AMD 驱动。AMD 引擎是现成的下载包。我们还没有在 Windows 的
  真实 AMD 显卡上带模型运行过它：请用户反馈运行情况（[AMD_HIP.md](AMD_HIP.md)）。
- **Mac：** Metal 引擎和别的引擎一样获取。Mac 上目前还没有图片读取器。

## 5. 下载期间告诉用户
<a id="5-while-it-downloads-tell-the-user"></a>

- 它从 Hugging Face 下载约 70 GB；耗时取决于他们的网络（100 Mbit/s 下大约 1.5-2
  小时）。可以停止，之后会从断点继续。
- 然后它会为他们的 PC 准备模型（几分钟）。
- **模型启动时，PC 可能会变慢或 1-3 分钟没有响应**（第一次最久）：Strata
  会把 35-55 GB 载入内存，并把其中一部分锁给显卡使用。这是正常的；他们不应该关闭窗口。
- 什么都不会离开他们的 PC：模型在本地运行。

## 6. 确认它正在提供服务
<a id="6-check-that-it-is-serving"></a>

`strata launch` 完成时会打印 `Strata is serving.` 以及地址。之后：

```
strata launch --status          # lists what is serving, with its address
strata launch --stop            # stops it and frees the graphics memory
strata launch --preset NAME     # starts a saved preset again (after --save-preset NAME)
```

浏览器不会自己打开。如果用户想要网页界面，给 `strata launch` 加上 `--open`。

## 7. 验证
<a id="7-verify"></a>

模型加载完成后 HTTP 服务器才会响应（之后启动 30-90 秒，第一次要几分钟）。轮询：

```
curl http://127.0.0.1:8080/health
```

它返回 `{"status": "ok", "model": ..., "max_context": ..., "images": ..., "api_key": ..., "loaded": true, ...}`。
然后：

```
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

有 API key 时，加上 `-H "Authorization: Bearer <key>"`。

## 8. 连接用户的应用程序
<a id="8-connect-the-users-apps"></a>

- **浏览器：** `http://127.0.0.1:8080` —— Chat、实时 Monitor，以及 About（设置与地址）。
- **任何兼容 OpenAI 的应用或 agent：** base URL `http://127.0.0.1:8080/v1`，任意 API key（或已配置的那个），
  任意模型名。
- **兼容 Anthropic 的应用：** base URL `http://127.0.0.1:8080`，不带 `/v1`。这类应用会自己加上 `/v1/messages`。
- **Codex CLI 和其它使用 Responses API 的应用：** `http://127.0.0.1:8080/v1/responses`（无状态；Codex 的
  `config.toml`：[DETAILS.md](DETAILS.md#the-responses-api-and-codex-cli)）。
- **Claude Code：** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`，把 `ANTHROPIC_MODEL` 设为它认识的 Claude 模型名
  （Strata 忽略这个名字），以及任意 `ANTHROPIC_AUTH_TOKEN`（或已配置的 key）。
- **给编程应用用的省事办法：** `strata connect` 会列出它们（OpenCode、Codex CLI、Claude Code、oh-my-pi、pi、
  Hermes Agent、OpenClaw）。`strata connect <app>` 会打印正在运行的服务器对应的文件和环境变量，
  `strata connect <app> --write` 询问后写入（并保留备份），`strata connect <app> --check` 按该应用的方式
  发送一个提示词和一次工具调用，并说明各自是否成功。
- **思考等级：** `"reasoning_effort": "none" | "low" | "medium" | "high"`（默认 high）。
- Strata 一次回答一个请求；`strata launch` 中的 `--parallel N` 可以同时 decode（解码）最多 N 个
  （[BATCHING.md](BATCHING.md)）。API 细节：[DETAILS.md](DETAILS.md#using-it)。

## 9. 如果 Terminal 无法运行：从源码安装
<a id="9-if-the-terminal-cannot-run-install-from-source"></a>

仅当第 2 步在这台 PC 上不奏效时才用它（例如 CPU 不支持 AVX2，或无法访问 GitHub
Releases）。它需要 git 或那个 zip，在 Linux 上还需要 `sudo` 来安装 Python 和构建工具。

```
git clone https://github.com/Niko1221/Strata.git
```

没有 git 的话：下载 https://github.com/Niko1221/Strata/archive/refs/heads/main.zip 并解压。选一个有
足够可用空间的磁盘：模型文件会放到 Strata 文件夹旁边的 `Strata-data` 文件夹（或用 `--data-dir <path>`
放到另一个磁盘）。setup 自带的检查会打印显卡、驱动、内存、CPU、可用磁盘，以及这台 PC 能否运行
Strata，并且除 Python 和运行所需的 `.venv` 之外不安装任何东西：

```
Windows:  START-HERE.bat --check
Linux:    ./setup.sh --check
```

在 Strata 文件夹中无提问地运行 setup（Windows：在 `cmd` 中，或在 PowerShell 里执行 `cmd /c START-HERE.bat ...`）。
模型的选法与第 3 步相同，只是改用 setup 自己的标志名（`--family qwen|swift|coder|unsloth --model Q2_0|IQ2_XS|...`）：

```
Windows:  START-HERE.bat --yes --family qwen --model IQ2_XS --vision yes --no-start
Linux:    ./setup.sh --yes --family qwen --model IQ2_XS --vision yes --no-start
```

标志（全部：`START-HERE.bat --help`）：

| 标志 | 含义 |
| --- | --- |
| `--yes` | 对每个问题都采用推荐答案（不提问） |
| `--family qwen\|swift\|coder\|unsloth` | 模型版本 |
| `--model Q2_0\|IQ2_XS\|IQ3_XXS\|IQ3_S\|IQ1_M\|UD-IQ4_XS\|UD-Q4_K_XL` | 规模（Coder 是 IQ1_M，Unsloth 是 UD-IQ4_XS 或 UD-Q4_K_XL） |
| `--context N` | 上下文长度（token 数）；按显存决定默认值：14 GB 以下 32768，20 GB 以下 65536，否则 131072 |
| `--vision yes\|no\|gpu\|cpu` | 读取图片。传 `yes` 会为显卡选择最合适的读取器（24 GB 以下用 CPU，24 GB 及以上用 GPU，任何品牌都一样）；`no` 关闭 |
| `--gpu N` / `--gpus 0,1` / `--gpus all` | 一张显卡，或共享该模型的多张（默认：显存最大的那张） |
| `--backend cuda\|hip` | NVIDIA 或 AMD 引擎；只有一种显卡的 PC 上会自动选择 |
| `--data-dir PATH` | 70-120 GB 的模型文件存放位置 |
| `--port N` | 服务器端口（默认 8080） |
| `--host H --api-key K` | 监听本机之外；只能与 key 一起使用 |
| `--no-start` | 只安装，不启动服务器 |
| `--setup` | 安装另一个模型，或更改已安装模型的设置 |
| `--check` | 只检查 PC |

建议 agent 使用 `--no-start`：服务器在前台运行，直到它的窗口被关闭，那会
阻塞你的 shell。请单独启动它：

```
Windows (PowerShell):  Start-Process -FilePath ".\run-iq2_xs.bat"            (opens its own window)
Linux:                 nohup ./run-iq2_xs.sh > strata-server.out 2>&1 &
```

setup 会打印它写好的启动脚本（`start script: run-<model>.bat`）。名字是 family 标签加规模，
小写：`run-iq2_xs`、`run-swift-iq2_xs`、`run-coder-iq1_m`、`run-unsloth-ud-iq4_xs`。引擎日志是
Strata 文件夹中的 `strata-<model>.log`。然后继续第 7 步。

说明：
- **Linux：** 缺少 Python 时，setup 用 `sudo apt`（或 dnf/pacman）安装带 venv 的 Python；在 AMD 上可能还需要
  `build-essential` 和 `git`。你无法输入用户的密码：如果某个 `sudo` 步骤必需，请用户来执行
  （例如 `sudo apt install python3-venv build-essential git`），然后重新运行 setup。
- **Linux 上的 AMD：** 除非系统中已有 ROCm 7，setup 会把 ROCm 安装到 `.venv`（约 10 GB，不需要 sudo），并为该显卡
  编译引擎（10-20 分钟，仅一次）。
- **Windows 上的 AMD：** setup 下载现成的 AMD 引擎（约 550 MB，内含 ROCm）——不编译任何东西，只需要
  AMD 驱动。在下载模型之前它会运行 `engine\strata-device.exe --list-devices`；如果那里没有列出显卡，
  问题就在驱动（让用户更新 AMD Software）。每个模型一张显卡。图片同样适用 24 GB 的规则。
- **NVIDIA 且没有适配该显卡的现成引擎：** setup 会提议安装构建工具并编译（20-40 分钟）；
  `--yes` 表示接受。
- **Mac：** `./setup.sh` 会在 Mac 上构建 Metal 引擎（需要 Apple 的 Command Line Tools，`xcode-select --install`）。

## 10. 出问题时
<a id="10-when-something-fails"></a>

| 现象 | 处理办法 |
| --- | --- |
| `the NVIDIA driver is too old` | 用户更新驱动（NVIDIA App 或 nvidia.com/drivers）并重启；然后重新运行该命令。 |
| 下载或安装中断 | 再次运行同一条命令；它会继续。 |
| `Strata couldn't find an engine for this computer` | 这张显卡不在引擎包的覆盖范围内，或无法访问更新源。对照第 1 步的列表检查显卡，检查网络，运行 `strata doctor`，然后运行 `strata update`。 |
| `port 8080 is already in use` | Strata 已经在运行（检查 `/health` 和 `strata launch --status`），或改用 `--port 8081`。 |
| 非常慢、磁盘繁忙，或提示 "the engine stopped unexpectedly" | 可用内存不足：关闭一些程序，或选择更小的规模（`q2_0` 或 `iq2_xs`）。 |
| 显存空闲（Windows）却出现 `ExpertCache: cudaMalloc(...) failed: out of memory` | 页面文件被关闭或太小：设置固定大小（自定义大小，初始值与最大值相同，例如 65536 MB）并重启；Windows 按需增长的页面文件（"System managed"）可能来不及增长（#60）。 |
| `prompt ... exceeds the context` | 用更大的 `--context` 重新启动。 |
| 未找到 AMD GPU（Linux） | amdgpu 驱动没有列出 GPU；检查显卡和驱动。集成显卡不受支持。 |
| Mac 提示应用"已损坏"（是应用，不是 Terminal） | 文件没有问题：[MAC.md](../MAC.md) 里有一条命令。 |

更多：[TROUBLESHOOTING.md](TROUBLESHOOTING.md) 和[完整表格](DETAILS.md#troubleshooting)。如果仍然失败，
运行 `strata doctor`，收集它的输出和错误消息中提到的引擎日志，并建议在
https://github.com/Niko1221/Strata/issues 提交 issue。

## 备选方案：MCP 服务器
<a id="alternative-the-mcp-server"></a>

Strata 还有 MCP 服务器，AI 工具可以通过工具调用
（`strata_status`、`strata_models`、`strata_install`、`strata_start`、`strata_stop`、`strata_logs`）来检查、安装、启动和停止 Strata，
而不必使用上面的 shell 命令。见 [MCP_SERVER.md](MCP_SERVER.md)。
