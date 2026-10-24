# Strata 的 MCP 服务器：让你的 AI 助手安装并运行 Strata
<a id="stratas-mcp-server-let-your-ai-assistant-install-and-run-strata"></a>

`tools/strata_mcp.py` 是一个 [MCP](https://modelcontextprotocol.io) 服务器，给你已经在用的 AI 助手
（Claude Code、Claude Desktop、Cursor、VS Code、Codex……）使用。添加一次，之后用平常的话提要求就行：

- “给这台 PC 安装 Strata。”
- “启动 Strata。” / “停止 Strata。”
- “Strata 在跑吗？占多少显存？”
- “哪个 Strata 模型适合我的 PC？”
- “Strata 没启动起来——日志里怎么说？”
- “Strata 在这台 PC 上有多快？”（做一次简短的速度测试）
- “怎么把 Open WebUI / OpenAI SDK / Claude Code 接到 Strata 上？”

助手会调用这个服务器的工具。这些工具运行 Strata 自己的 setup，并按启动脚本的方式启动模型。它就是一个
只依赖标准库的 Python 文件，所以在 setup 创建 `.venv` 之前就能用。它需要 Python 3.10 或更新版本，Windows 上要 64 位。

它和[来自 MCP 服务器的工具](DETAILS.md#tools-from-mcp-servers)不是一回事。那个功能
（`serve/mcp.py`）让 *Strata 模型*在聊天页面里调用你的 MCP 服务器上的工具。本页是另一个
方向：*你的助手*来管理 Strata。

## 添加它
<a id="add-it"></a>

你需要一份 Strata：GitHub 上的 zip，或者 `git clone https://github.com/Niko1221/Strata`。下文中，
`C:\Users\you\Strata`（Windows）和 `/home/you/Strata`（Linux）指的就是那个文件夹。Linux 上用 `python3`。
Windows 上如果 `python` 不在 PATH 里，就用 `py`。

### Claude Code
<a id="claude-code"></a>

```bash
claude mcp add strata -- python C:\Users\you\Strata\tools\strata_mcp.py         # Windows
claude mcp add strata -- python3 /home/you/Strata/tools/strata_mcp.py           # Linux
```

加上 `--scope user` 让它在每个项目里都可用。Claude Code 里的 `/mcp` 会显示它已连接。

### Claude Desktop
<a id="claude-desktop"></a>

打开 Settings > Developer > Edit Config（`claude_desktop_config.json`）并加上：

```json
{
  "mcpServers": {
    "strata": {
      "command": "python",
      "args": ["C:\\Users\\you\\Strata\\tools\\strata_mcp.py"]
    }
  }
}
```

Linux 上用 `"command": "python3"` 和 `"args": ["/home/you/Strata/tools/strata_mcp.py"]`。改完重启 Claude
Desktop。

### Cursor
<a id="cursor"></a>

所有项目共用就用 `~/.cursor/mcp.json`，单个项目就用项目里的 `.cursor/mcp.json`。格式和
Claude Desktop 的一样：

```json
{
  "mcpServers": {
    "strata": {
      "command": "python",
      "args": ["C:\\Users\\you\\Strata\\tools\\strata_mcp.py"]
    }
  }
}
```

### VS Code（GitHub Copilot agent 模式）
<a id="vs-code-github-copilot-agent-mode"></a>

在工作区里用 `.vscode/mcp.json`，或者从命令面板运行 “MCP: Open User Configuration”：

```json
{
  "servers": {
    "strata": {
      "type": "stdio",
      "command": "python",
      "args": ["C:\\Users\\you\\Strata\\tools\\strata_mcp.py"]
    }
  }
}
```

Linux 上用 `"command": "python3"` 和 `"args": ["/home/you/Strata/tools/strata_mcp.py"]`。

### Codex CLI
<a id="codex-cli"></a>

在 `~/.codex/config.toml` 里：

```toml
[mcp_servers.strata]
command = "python"
args = ['C:\Users\you\Strata\tools\strata_mcp.py']    # Linux: command = "python3", args = ["/home/you/Strata/tools/strata_mcp.py"]
tool_timeout_sec = 180                                # strata_stop / strata_benchmark can take longer than 60 s
```

### 其它任何 MCP 客户端
<a id="any-other-mcp-client"></a>

命令是 `python <Strata folder>/tools/strata_mcp.py`。传输方式是 stdio。它支持 MCP 版本
2025-06-18，也支持 2025-03-26 和 2024-11-05。

`--root <folder>` 管理另一个 Strata 文件夹，而不是这个文件所在的那个。例如：
`python D:\tools\strata_mcp.py --root D:\Strata`。

## 工具
<a id="the-tools"></a>

| 工具 | 它做什么 | 参数（都可选） |
| --- | --- | --- |
| `strata_status` | 是否有模型在跑：模型、引擎版本、上下文、忙还是闲、占用的显存和 RAM。装了什么：模型、启动脚本、引擎版本、数据文件夹、下载。进行中的一次安装。这台 PC 的硬件：OS、CPU、RAM、显卡、页面文件、空闲磁盘。setup 为这台 PC 推荐的模型和上下文。 | `port`, `hardware` (true/false) |
| `strata_models` | setup 提供的每个系列和规模，含下载大小、需要的 RAM、是否适合这台 PC、是否已安装。还有各上下文长度和推荐项。 | - |
| `strata_install` | 在后台不问问题地跑 setup（见下文）。没有 `confirm: true` 时只返回计划。再调一次会显示进度。`cancel: true` 停止安装。 | `family` (qwen, swift, coder, unsloth), `model` (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M, UD-IQ4_XS, UD-Q4_K_XL), `context` (8192 ... 524288), `vision` (no, yes, gpu, cpu), `kv` (int8, q4_0, k8v4), `gpu` (一个数字) or `gpus` ("0,2", "all"), `backend` (auto, cuda, hip), `low_ram`, `port`, `data_dir`, `confirm`, `cancel` |
| `strata_start` | 在后台启动一个已安装的模型，并等它应答。如果 `wait_seconds`（默认 50）之后还没就绪，就返回 “loading”。 | `model` (`strata_status` 里的一个 id，例如 `q2_0`、`coder-iq1_m`；默认：最近用过的那个), `port`, `gpu`, `wait_seconds` (0-600) |
| `strata_stop` | 通过服务器自己的 API 卸载模型，然后结束它启动的那个服务器。 | `port`, `force` |
| `strata_logs` | 服务器日志、引擎日志或 setup 日志的最后几行。 | `source` (server, engine, setup), `model`, `lines` (1-500) |
| `strata_benchmark` | 发一个固定的短请求（greedy、关闭 thinking），报告输出和提示词的 tokens/s。 | `port`, `max_tokens` (16-512) |
| `strata_connect_info` | OpenAI 和 Anthropic 的 base URL、模型 id、是否需要 key，以及 Claude Code、Codex、OpenAI SDK、curl、Cursor、Continue 和 Open WebUI 的设置。 | `port` |

每个结果都是带一行 `summary` 的 JSON。它同时以文本和 MCP 结构化内容两种形式发送。

## 它是怎么工作的
<a id="how-it-works"></a>

**安装。** `strata_install` 用 setup 自己的逻辑和表（它导入 `setup.py` 的 `MODELS`、`FAMILIES`、
`CONTEXTS` 和 RAM 检查）来补上你没指定的部分：

- setup 自己会选的规模：60 GB RAM 起选 IQ3_XXS，约 44 GB 起选 Q2_0，约 28 GB 起选 Coder，
  再往下如果显卡能补上 RAM 的不足，就选低内存模式的 Coder；
- setup 按显卡显存推荐的上下文；
- 图片在哪里读（显存不到 24 GB 时读图跑在 CPU 上，24 GB 起跑在 GPU 上，适用于 NVIDIA 和 AMD 卡；Mac 上还没有）。

第一次调用返回计划：下载大小（58-111 GB）、空闲磁盘空间、RAM 检查，以及确切的
setup 命令。助手应该把计划给你看，你同意后带上 `confirm: true` 再调一次。

然后它执行 `START-HERE.bat` / `setup.sh` 所执行的同样步骤：

1. 如果 `.venv` 不存在，就用它自己运行的那个 Python 创建 `.venv`。
2. 以后台进程运行 `setup.py --yes --no-start --family ... --model ... --context ... --vision ...`。
   输出写到 Strata 文件夹里的 `.strata-mcp/setup.log`。

这个工具会立刻返回。再调一次就能看到步骤（1-7）、下载百分比，或 setup 的报错信息。
setup 已经做完的部分在下次运行时会跳过，下载会续传。AI 应用关掉后，安装仍在继续跑。当这个服务器
启动的模型正在运行时，安装会被拒绝，因为 setup 可能替换那个模型正在用的引擎文件。

**启动。** `strata_start` 运行和 `run-<model>` 脚本相同的命令，即 `serve/server.py --engine strata
--config strata-<model>.json --port N`，用 venv 的 Python，不带 `--open`（不开浏览器），也不开控制台
窗口。服务器的输出写到 `.strata-mcp/server.log`。然后工具轮询 `GET /health`，直到模型
应答。加载要 1-3 分钟。如果服务器在加载中结束，工具会返回它日志的末尾。AI 应用关掉后模型
会继续运行，直到 `strata_stop`。如果 AI 应用把它的工具放在一个不允许进程脱离的 Windows job 里运行，
模型会随应用一起结束。

有两种情况会被拒绝：

- 已经有模型在跑：工具把它返回，什么也不启动；
- 别的程序占用了端口：工具会说明。

**停止。** `strata_stop` 先发 `POST /unload`。服务器给引擎发 `QUIT`，引擎自己释放
显存和页锁定的 RAM，和你关窗口时一样。然后服务器结束：

- Linux：SIGTERM，走服务器自己的 Ctrl+C 路径。
- Windows：进程结束。它的 job object 会结束视觉编码器和它启动的 MCP 服务器。

有请求正在运行时，工具会拒绝，除非你传 `force: true`。

**状态。** 工具在 127.0.0.1 上读 `GET /health`、`/v1/status`、`/status`，在旧版服务器上还读 `/metrics`。
如果这个文件夹里某个配置设了 `api_key`，工具会在自己的请求里带上那个 key。它从不显示这个 key。

## 安全
<a id="safety"></a>

- **不走 shell，不给自由路径。** 每个参数都按固定 schema 和 setup 自己的选项检查。未知参数、
  setup 不提供的模型或系列、或者不是由数字和逗号组成的 `gpus` 值，都会在真正开始之前被拒绝。
  命令以参数列表形式传递，没有任何东西经过 shell。
- **路径。** 唯一的路径参数是 `data_dir`。它必须是以下之一：
  - 当前的 Strata 数据文件夹；
  - Strata 文件夹里面的一个文件夹；
  - 一个新的绝对路径文件夹，名字以 `Strata` 开头（例如 `D:\Strata-data`），其父目录存在，且不在
    系统文件夹里。

  `..`、相对路径和网络（UNC）路径都会被拒绝。日志只从 `.strata-mcp/` 读取，以及从配置指定的引擎日志
  读取，前提是该日志在 Strata 文件夹或数据文件夹里面。
- **只管自己的进程。** 服务器在 `.strata-mcp/*.json` 里记录它启动的每个进程的进程 id *和启动时间*。
  只有两者都匹配时它才结束进程，所以操作系统分配给别的程序的进程 id 绝不会被误杀。用其它方式启动的
  Strata（它自己的窗口、运行脚本、服务）只是通过它的 API 请求卸载模型。要结束那个服务器，请关掉它的窗口。
- **大的下载需要你点头。** 不带 `confirm: true` 调用时，`strata_install` 什么也不做。计划会先显示
  下载大小和空闲空间；磁盘太小时安装会被拒绝。
- **只在本机。** 服务器只检查 `127.0.0.1`，从不改动网络设置：它不会用 `--host 0.0.0.0` 或 API key 去安装。
  要在局域网里共享 Strata，请自己运行 setup（见[使用方式](DETAILS.md#using-it)）。实验性选项也一律
  不开：速度预测、超过 262K 的 rope scaling 和 `--build`。这些请手工运行 setup。

## 故障排查
<a id="troubleshooting"></a>

- **助手看不到这些工具。** 手工运行命令：
  `python C:\Users\you\Strata\tools\strata_mcp.py`。它应该向 stderr 打印 `[strata-mcp] ready (Strata folder ...)`
  并等待输入（Ctrl+C 结束）。如果它报 `not a Strata folder`，说明路径不对，或者需要 `--root`。
  低于 3.10 的 Python 会被拒绝。
- **安装失败。** 让助手给你 setup 日志（`strata_logs` 带 `source: setup`），或者打开
  `.strata-mcp/setup.log`。setup 的 `[X]` 行会说明要修什么。然后重新安装：已完成的步骤都会跳过。
- **模型起不来。** 用 `strata_logs` 取服务器日志，用 `source: engine` 取引擎日志。
  [故障排查](DETAILS.md#troubleshooting)里列了常见原因。
- **测试。** `python -m unittest tools/test_strata_mcp.py` 用假的 setup 和服务器跑这些测试。它们
  不下载任何东西，也不用 GPU。
