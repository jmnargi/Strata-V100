# Strata 安全吗、会保护隐私吗？
<a id="is-strata-safe-and-private"></a>

本页说明 Strata 会下载什么、从哪里下载、什么时候使用互联网、哪些内容留在你的 PC 上，以及这些下载是如何校验的。这里的所有内容都来自阅读这个版本的代码。每一节都列出了对应的文件，因此你自己或你信任的人都可以核对。凡是 Strata 做得不如你期望的地方，本页都会写明。

## 一览
<a id="at-a-glance"></a>

- **你的对话、提示词、文件、图片和模型都留在你的 PC 上。** 只有在你自己选择时它们才会离开：添加一个外部模型服务，把服务器开放给其它设备或互联网，或者让代理搜索网络、打开网页（见[编码代理能做什么](#what-the-coding-agent-can-do-on-your-pc)）。
- **没有遥测。** Strata 不发送使用数据、不上报崩溃信息、不做统计分析，也没有需要登录的账号。
- **启动和运行模型永远不会下载任何东西。** 下载和更新是彼此独立的步骤，由你主动发起。
- **服务器只在你自己的 PC 上监听。** 从其它设备访问它需要 API key。
- **模型下载完成之后，Strata 无需联网即可运行。** 启动、加载、服务和对话都不需要连接。只有你主动要求的事情才会用到网络：代理的网页搜索和网页、外部模型服务、隧道，以及安装和更新（首次安装模型还会获取 llama.cpp 的准备脚本）。
- **下载会被校验。** 更新来自经过签名的 feed。每个模型文件都有固定的 SHA-256 和大小；如果某个文件的固定版本（pinned）已经不存在，安装就会停止（细节见[下文](#how-downloads-are-checked)）。
- **在默认模式下，编码代理在改动任何东西之前都会先询问。** 它搜索网络之前不会询问。
- **代码是开放的。** 任何人都可以阅读，安全问题请提交到 [SECURITY.md](../../SECURITY.md)。

## Strata 会向外回传数据吗？
<a id="does-strata-phone-home"></a>

不会。对 app、Terminal、引擎服务器和安装程序做检索后，没有发现任何分析库、崩溃上报程序，也没有任何向别处上报使用情况的代码。app 的隐私说明也这么说（`apps/host/src/components/PrivacyNotes.tsx`："Strata sends no usage telemetry"）。

有些东西听起来相似，但它们都留在你的 PC 上：

- Monitor、`/metrics` 和 `/v1/telemetry` 向你展示硬件读数（温度、速度、内存）。它们由你自己的 PC 提供，如果你设置了 API key，它们也需要该 key。
- 在 Linux 上，app 窗口崩溃时会把崩溃日志写到你磁盘上的一个文件夹里。没有任何程序会发送它。**Save logs**（Settings、Engine、Diagnostics）会把一个文本文件放进你的 Downloads 文件夹，是否把它附到 bug 报告里由你决定。
- 编码代理是一个独立组件（DeepSeek Harness，MIT）。它有自己的遥测开关，而 Strata 启动它时会关闭该开关（`apps/host/server/agent/harness-runtime.mjs`）。

和任何下载一样，下面每一次连接都会把对方的 IP 地址和你请求的文件暴露给对端。

## Strata 什么时候使用互联网
<a id="when-strata-uses-the-internet"></a>

| 什么 | 去向 | 时机 |
| --- | --- | --- |
| 模型文件、图片读取器、加速层（MTP） | huggingface.co，或者你设置了镜像时 `HF_ENDPOINT` 中指定的镜像 | 你安装模型时 |
| 同样的文件从 ModelScope 获取，供无法访问 Hugging Face 的 PC 使用 | modelscope.cn | 只有你要求时才用（`--source modelscope`）。Strata 绝不会自行切换到它 |
| 引擎、Terminal 和 Control 的更新 | Niko1221/Strata 的 GitHub Releases 上的签名 feed | 你安装引擎、点击 **Update engine** 或运行 `strata update` 时。在 app 里打开 **Settings, Engine** 也会读取该 feed，以查看是否存在更新 |
| app 更新 | 同一个 GitHub Releases 上的 `latest.json`，经由 Tauri 的更新器 | 你点击 **Check for updates** 或 **Update Strata** 时，以及你打开 **Settings, Engine** 时 |
| llama.cpp 源码（固定的 commit） | github.com/ggml-org/llama.cpp | 安装模型期间，如果你的 PC 上还没有它。Strata 只使用它的模型准备脚本 |
| 网络搜索和网页 | DuckDuckGo，Bing 作为第二来源；如果你为 Exa、Perplexity 或 DeepSeek 保存了 key，也可以用它们。代理被要求打开的任何页面都会访问该页面所在的站点 | 代理使用它的搜索或 "open web pages" 工具时 |
| 外部模型服务 | 你添加的服务（OpenAI、Anthropic、OpenRouter、你网络上的服务器等） | 只在与该模型对话时，而且只有在你添加了它的 key 之后 |
| Cloudflare 隧道 | Cloudflare，经由 `cloudflared` 程序 | 只在你开启隧道期间，而且没有 API key 时它不会启动 |
| Tailscale、本地网络 | 不是 Strata 的流量。Strata 只读取 Tailscale 的状态并显示地址 | 只有当你在 `strata launch` 中选择这种访问方式时 |
| API 请求中的图片链接 | 链接里的地址 | 只有当某个程序向你的服务器发送带 `http(s)` 图片链接的请求时。`--offline` 会拒绝它 |
| 远程工具服务器（MCP） | 你填在运行配置里的地址 | 只有在你添加了它时 |
| 你在 app 的预览窗格中打开的网站 | 该网站 | 只有当你在 **Open a website** 中输入地址时 |
| 分享到 X 或 Reddit（Monitor 截图） | 你的浏览器会打开 x.com 或 reddit.com，并带上说明文字和指向 Strata 页面的链接 | 只有当你点击时。截图保存在你的磁盘上，不会被上传 |

这就是在代码中能找到的完整列表。一些细节：

- **没有任何东西在后台检查更新。** app 启动时不会检查。表中的检查只在你打开 Settings, Engine、点击按钮或运行命令时执行。Terminal 只有在没有安装任何引擎时才会自行获取引擎，而且 `strata launch` 会先询问。
- **启动和服务绝不会下载任何东西。** `strata serve` 和 app 的加载按钮不会获取任何东西。app 会为它的各个部分设置 `STRATA_PACKAGED=1` 和 `STRATA_NO_ACQUIRE=1`（`apps/desktop/src-tauri/src/main.rs`），下载器在这些变量下会拒绝运行（`strata_control/acquisition.py`、`apps/host/server/acquisition.mjs`）。这就是 app 在运行期间从不替你下载 `cloudflared` 的原因。
- **引擎本身没有网络代码。** Strata 中运行模型的那部分（`src/`、`include/`、`sycl/`、`metal/`）不包含任何网络功能。app 自己的页面不会从互联网加载字体、脚本或跟踪器。
- **从源码构建**会访问更多地方。从 python.org 或 winget 获取 Python（`START-HERE.bat`），在 Mac 上从 GitHub 获取独立版 Python（`setup.sh`），从 PyPI 获取 Python 包（`requirements.txt`，固定版本），从 PyPI 获取 NVIDIA 的 CUDA 库，在 Ubuntu 和 Debian 上还涉及 NVIDIA 的 apt key 和软件包、AMD 的 ROCm wheels、LunarG 的 Vulkan SDK，以及从 GitHub 获取 llama.cpp 和 stable-diffusion.cpp 的源码。这些都只在你运行 setup 时执行。

## 哪些东西留在你的 PC 上
<a id="what-stays-on-your-pc"></a>

- **对话、提示词、附件和图片**由你 PC 上的引擎处理。app 把对话保存在你磁盘上它自己的存储里。模型位于你选择的文件夹中。
- **API key**：你为外部服务保存的 key 存放在一个只有你的用户账户才能打开的文件里。它们没有加密。（`apps/host/server/model-providers.mjs`）
- **服务器在 `127.0.0.1` 上监听**，仅限本机。app 固定了这个地址（`STRATA_BIND=127.0.0.1`），引擎服务器的默认值也是它（`serve/server.py`，`--host` 选项）。
- **共享需要 key。** app 的 "Share with other devices" 和 Cloudflare 隧道在没有 API key 时不会启动，而且共享端口只响应 `/v1/models`、`/v1/chat/completions` 和 `/v1/messages`（`apps/host/server/local-share.mjs`、`cloudflare.mjs`）。当你选择的访问范围比本机更宽时，`strata launch` 会为你生成一个 key。裸的引擎服务器（`serve/server.py`）在你没有 key 就把它开放到网络时只会打印一条警告，所以这种情况请自行设置一个 key。
- **当你使用外部模型时**，对话内容（以及你开启的 skill 的指令）会发送到该服务。app 在隐私说明中已写明这一点。之后适用该服务自己的规则。
- **你连接的程序**（Claude Code、Codex 等）是独立的程序，有它们自己的网络行为。Strata 无法控制它们发送什么。

服务器还有面向浏览器的防护（host 检查、origin 检查，默认不启用 CORS）。它们写在 [SECURITY.md](../../SECURITY.md#what-the-server-exposes) 中。

## 下载如何校验
<a id="how-downloads-are-checked"></a>

**引擎、Terminal 和 Control 的更新。**

1. 发布时会公布 `strata-feed.json` 和 `strata-feed.json.sig`，它们用 Ed25519 密钥签名。匹配的公钥就在每一个已安装的 Strata 里（`strata_control/update_keys.json`）。私钥由所有者离线保管。
2. 在使用该 feed 之前，你的 PC 会校验签名、确认 feed 未过期，并且它不早于上一次被接受的 feed（这样旧的 feed 无法被重放）。
3. 每个引擎包都会对照它的 SHA-256 和签名过的 manifest 校验，然后包内每一个文件再对照各自的哈希校验，最后检查它是否能适配你的 PC。
4. 在 Strata 切换到新引擎之前，会先试启动它。如果有任何一步失败，你会保留原来的引擎。

更新永远不会替换你的模型或已保存的调优。完整说明见 [packaging/UPDATES.md](../../packaging/UPDATES.md)。

**app。** 它的更新用另一把密钥签名（Tauri 的 minisign），公钥的那一半在 app 里（`apps/desktop/src-tauri/tauri.conf.json`）。更新器会拒绝签名不匹配的安装程序。当你运行 `strata update --app` 时，Terminal 会校验同样的签名。

**模型文件。** `setup.py` 下载的每个模型文件都有固定的 SHA-256 和大小，并且写在代码里，所以校验不需要联网：主要的 Qwen3.8-Flash-Next 文件（Q2_0、IQ2_XS、IQ3_XXS、IQ3_S）、Swift 1.5、Coder、Unsloth 的各尺寸、Gyro 文件、Qwen3.6、NVFP4 模型包、图片读取器、图像模型文件，以及加速层（MTP，逐个 tensor 校验）。不匹配的文件会被删除，安装随之停止。如果文件在服务器上的大小不是固定的那个大小，安装会在下载开始前就停止。

- 如果某个文件的固定版本已从 Hugging Face 上消失，安装会停止，并提示你更新 Strata。它不会改用仓库中当前的文件。测试者可以设置 `STRATA_ALLOW_UNPINNED=1`，在没有校验的情况下直接取当前文件。该变量默认关闭，被设置时 setup 会说明这一点。
- 启动模型不会校验它的文件；对你已经有的模型再次运行 setup，也不会重新下载或重新计算哈希。在这些哈希存在之前安装的文件会原样保留。
- 来自 ModelScope 的文件会对照相同的固定 SHA-256 校验。ModelScope 提供的是仓库当前的文件，所以被它改动过的文件会校验失败。没有固定版本的文件会对照 ModelScope 公布的哈希校验，而当它不公布哈希时 setup 会给出警告。
- app 通过运行同一个 setup 来安装这些模型，所以它有相同的校验。app 的 host 还维护着自己的一份简短入门列表（`apps/host/server/real-model-catalog.mjs`）。它的下载路径从仓库当前的 `main` 分支取文件，只检查它们的大小是否大致符合预期。0.2.0 版的 app 界面没有使用它。（`setup.py`：`REPO_PINS`、`download`、`verify_sha256`；`tools/mtp_fetch.py`；`apps/host/server/image-models.mjs`。）

**源码路线中的引擎。** `setup.py` 会对照 GitHub 的 API 为该发布文件公布的 SHA-256 来校验引擎 zip。如果 GitHub 无法说明（离线、触发速率限制），setup 会给出警告并照常安装。`STRATA_SKIP_SHA256=1` 会关闭这项校验，而它默认从不开启。

**一行安装命令**（`curl ... | sh`、`irm ... | iex`）会用刚解包的 Terminal 校验 Terminal 归档的 SHA-256 和签名过的 feed。校验和与密钥来自与归档相同的发布，所以首次安装信任的是 HTTPS 和 `Niko1221/Strata` 这个发布本身。这意味着什么，以及如何先读一遍脚本，见 [SECURITY.md](../../SECURITY.md#how-far-to-trust-curl--sh-and-irm--iex)。之后的更新会对照你 PC 上已有的密钥校验。

**代码签名。** Windows 安装程序还没有代码签名，所以 Windows 会显示 "Windows protected your PC"（Windows 已保护你的 PC）。Mac 构建的签名和 Apple 公证（notarisation）在 `packaging/mac/` 中配置，并在维护者的 Mac 上运行。未经过公证的 Mac 构建会显示 macOS 的 "damaged"（已损坏）警告（[MAC.md](../MAC.md)）。

**校验覆盖不到的部分。** 它们能发现损坏或被调包的下载，以及没人用发布密钥签过名的 feed。但如果能发布版本的人（或者持有签名密钥的人）发布了有害内容，它们无法保护你。

## 编码代理能在你的 PC 上做什么
<a id="what-the-coding-agent-can-do-on-your-pc"></a>

app 的对话通过一个代理运行，它可以读取文件、运行命令、编辑文件、搜索网络，并使用你连接的工具。它可以在不询问的情况下做什么，取决于 **Mode** 菜单（`apps/host/server/agent/README.md`）。

| 模式 | 会发生什么 |
| --- | --- |
| **Manual**（默认） | 任何会改动东西的操作都会先询问，运行之前会出现一张 **Approve** 或 **Deny** 卡片。 |
| **Plan** | 什么都不能改。改动请求会被拒绝，也不会出现卡片。你连接的工具同样会被拒绝。 |
| **Bypass** | 完全访问，不做任何询问。 |

在 **Manual** 模式下：

- **以下操作会询问：** 写入或编辑文件、任何不在少量只读命令名单上的命令，以及你连接的每一个工具。在浏览器工具中，打开页面、点击和输入都会询问。
- **以下操作不会询问：** 读取文件、列出和搜索文件夹，以及 `ls`、`cat`、`git status` 之类的只读命令。带重定向、带链式操作（`;`、`&&`）、带变量，或者带任何检查无法识别的内容的命令都会询问。
- **网络搜索和打开网页不会询问。** 它们默认开启，可以在 Settings, Tools 中关闭。搜索词和地址由模型选择，所以你对话中的词语可能出现在搜索里或网址里（`strata-gate.mjs` 把两者都列为只读）。
- **对话文件夹只是代理的起点。** Manual 模式不会把它限制在那里。你批准的操作可以触及你的用户账户能触及的任何东西，所以按下 Approve 之前请先看清卡片。
- **你的密钥不会进入它的命令。** 代理的 shell 拿不到名字里包含 KEY、TOKEN、SECRET、PASSWORD、CREDENTIAL 或 AUTH 的环境变量。
- **计划任务**默认以 Plan 模式运行。屏幕前没有人时，审批请求会被回答为 "no"。
- **Stop** 会结束一次对话的运行。关闭标签页不会。

代理运行在回环地址上，并使用随机 token，所以网络上的其它程序无法访问它。

## 开源与报告问题
<a id="open-source-and-reporting-a-problem"></a>

Strata 采用 MIT 许可，全部代码在 [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)。你可以阅读它、构建它并自行比对。Strata 还没有经过外部安全审计。

要报告安全问题，请使用 GitHub 的私密报告（仓库的 **Security** 标签页），具体做法见 [SECURITY.md](../../SECURITY.md)。普通问题和想法可以发到 [issue](https://github.com/Niko1221/Strata/issues)。

## 在代码中哪里可以核对这些
<a id="where-to-check-this-in-the-code"></a>

| 主题 | 文件 |
| --- | --- |
| 模型下载与固定版本 | `setup.py`, `tools/mtp_fetch.py`, `tools/image/qimage.py`, `apps/host/server/models.mjs`, `image-models.mjs` |
| 更新 feed、签名 | `strata_control/feed.py`, `signing.py`, `transfer.py`, `update_keys.json`, `strata_cli/update_cmd.py` |
| app 更新器 | `apps/desktop/src-tauri/src/updater.rs`, `apps/desktop/src-tauri/tauri.conf.json` |
| 运行期间被拒绝的下载 | `strata_control/acquisition.py`, `apps/host/server/acquisition.mjs` |
| 一行安装脚本 | `packaging/install.sh`, `packaging/install.ps1` |
| 网络工具与代理的审批 | `apps/host/server/agent/plugins/strata-gate.mjs`, `agent/profile.mjs`, `web-search-ddg.mjs`, `web-search-bing.mjs`, `web-search-keyed.mjs` |
| 外部模型服务 | `apps/host/server/model-providers.mjs` |
| 共享、隧道 | `apps/host/server/local-share.mjs`, `cloudflare.mjs`, `strata_cli/launch/access.py` |
