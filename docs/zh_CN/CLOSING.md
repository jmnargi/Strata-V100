# 收尾：什么会结束引擎，以及何时

<a id="closing-what-ends-the-engine-and-when"></a>

规则是：**启动它们的那个东西结束时，Strata 服务器以及它启动的引擎也随之结束，无论它以什么方式结束。** 有一条
干净路径（引擎完成或取消它的请求、释放内存、退出），在这条路径背后还有一层保证，它不需要任何 Strata 代码运行
（崩溃、`kill -9`、控制台被关闭、注销）。先尝试干净路径；如果引擎没有及时退出，就杀掉它。无论走哪条路，之后
端口和显存都是空闲的。

## 运行引擎的两种方式

<a id="the-two-ways-to-run-an-engine"></a>

| | 由谁启动 | 随什么结束 | 用什么停止 |
|---|---|---|---|
| **前台** | `strata serve --profile NAME`（或 `python serve/server.py`） | 终端 / 该命令 | Ctrl+C、关闭窗口、`kill` |
| **后台服务**（有意为之） | `strata runtime start --profile NAME` | 不随任何东西：终端关闭后它继续提供服务 | `strata runtime stop --profile NAME` |
| **桌面应用的引擎** | 应用（profile 为 `app-1`、`app-2`、...） | 应用：它的窗口、它的进程、它的 host | 关闭应用，或应用执行卸载 |

后台服务是有意的选择，而且将一直如此：`runtime start` 启动一个不依附于运行它的终端的 owner host。它记录在
`strata runtime --help` 中。如果你希望它跟随某个进程，`--die-with-pid PID` 会把它绑定到那个进程（应用对它的
每个引擎都这样做）。Terminal 启动的引擎永远不会被应用停止，而应用的 `app-N` 引擎也不会被 Terminal 停止——除
非你运行 `runtime stop --profile app-1`。

## 每种结束方式会做什么

<a id="what-each-way-of-ending-does"></a>

先走干净路径：服务器收到停止请求，排空或取消当前请求（`--shutdown-timeout`，Terminal 中为 30 s，应用的引擎
为 15 s），向引擎写入 `STOP` 再写入 `QUIT`（仍在运行的请求会在它的下一步被取消），等待 20 s，发送 terminate，
再等 5 s，最后 kill。引擎和图像编码器运行在各自的进程组中（Windows 上是新的进程组），这样终端的 Ctrl+C / 挂断
只会到达服务器，服务器按上述顺序停止引擎，而不是让引擎在请求进行中被杀掉。

| 事件 | Windows | Linux | macOS |
|---|---|---|---|
| 终端中的 Ctrl+C | 服务器处理它，干净停止 | 相同（SIGINT） | 相同 |
| `kill`（SIGTERM）、`docker stop`、systemd stop | 不适用 | 干净停止 | 干净停止 |
| 终端窗口关闭 / `ssh` 断开 | 控制台关闭事件：干净停止（系统大约给 5 s）；job object 结束剩下的一切 | SIGHUP：干净停止。用 `nohup` 启动时不会（那等于要求活得比终端久） | 相同 |
| Ctrl+Break | 干净停止 | 不适用 | 不适用 |
| 注销、关机 | 控制台 logoff/shutdown 事件，然后 job object | 来自会话管理器的 SIGHUP/SIGTERM | 相同 |
| 服务器被杀（`taskkill /F`、`kill -9`、崩溃） | 引擎位于服务器拥有的 kill-on-close job object 中：立刻结束 | `PR_SET_PDEATHSIG` = SIGKILL，在引擎启动前设置，来自一个长期存在的线程（信号绑定到执行 fork 的那个线程，而不是进程） | 一个 watcher 进程持有指向服务器的管道；服务器死亡后，watcher 核对 PID 和启动时间后结束引擎 |
| owner host（`runtime start`）被杀 | 持有服务器进程树的 job object 会结束它 | 服务器的 owner 管道关闭：它收到 SIGTERM 并干净停止，宽限期过后是 SIGKILL | 相同 |
| 应用被关闭（窗口 X、正常退出） | host 通过 Control 停止应用的引擎（排空、结束），然后退出；60 s 期限，然后 job object | 相同，但没有 job object：各 owner 跟随 host | 相同 |
| 应用被杀、崩溃、注销 | 包住应用的 job object 结束 host、Control、owner host、服务器和引擎 | host 在一秒内发现它的父进程消失，干净地停止引擎；每个 owner 跟随 host 的 PID（`--die-with-pid`） | 相同 |
| 应用的 host 崩溃 | 每个 owner 发现 host 的 PID 消失（约一秒），排空后停止它的引擎 | 相同 | 相同 |

`strata_control/procs.py` 是每一处“这还是我启动的那个进程吗”判断背后的身份规则：PID **和** 启动时间
**和** 命令行，并在每次发送信号时再核对一遍。绝不只凭 PID 发送信号。

## 遗留进程，以及如何找到它们

<a id="leftovers-and-how-they-are-found"></a>

Strata 启动的每个引擎、图像编码器和服务器都会写一条小的私有记录（`<state>/procs/<pid>.json`：PID、启动时间、
命令行，以及其父进程的同样信息）。进程结束时记录会被删除。如果一切都死了却仍留下一个进程，下面这些会在完成上述
核对之后结束它：

* 下一次启动引擎（`strata serve`、`runtime start`、应用加载模型）时，针对机器上父进程已消失的任何遗留进程；
* owner host 丢失时的 `strata runtime stop --profile NAME`（它会结束记录中的服务器和引擎并清除私有状态，而不是
  以“runtime control is unavailable”失败）；
* 按需运行 `strata runtime cleanup [--profile NAME]`（脚本用 `--json`）；正在运行的实例永远不会被触碰；
* 应用 host，在启动时执行一次。

更早版本的 Strata 留下的引擎不写记录。在启动引擎时，**与即将启动的可执行文件完全相同的孤儿进程**（命令行上带
`--serve`、没有存活父进程：它的服务器已经不在了）会以同样方式被结束；另一个安装的引擎，或任何父进程仍存活的
进程，都不会。设置 `STRATA_KEEP_LEFTOVERS=1` 可以关掉这两项检查。

## 一个应用，一个或几个引擎

<a id="one-app-one-or-several-engines"></a>

桌面应用是单实例的（Tauri 的单实例插件）：再次启动它只会聚焦已经打开的窗口，不启动任何东西，所以两个应用永远
不会争抢 `app-N` 槽位。一个应用可以运行多个引擎（例如每块 GPU 一个）：应用加载的每个模型都会在 Control 下获得
自己的 profile `app-1`、`app-2`、...，各自有自己的 owner、服务器和引擎，并且都随应用一起停止。模型用哪块 GPU
由该模型的运行配置（`gpu`）决定；应用为每个已加载的模型启动一个引擎。按模型选择显卡需要在 Models/Serving 界面
里有一个控件；目前还没有（需要什么见 `f-split-life` 流的说明）。

## 如何证明

<a id="proving-it"></a>

`tests/terminal/lifecycle_matrix.py` 用一个假的引擎包或真实引擎（`--config RUN.json --components DIR --vram`）
运行上面每个用例，并在每个用例后断言：进程树没有残留、端口空闲、运行时租约可以取得、显存已归还。
`tests/terminal/test_lifecycle_matrix.py` 是 pytest 包装（需要控制台窗口的用例在 `STRATA_MATRIX_CONSOLE=1` 下
运行）。`apps/desktop/scripts/app-close-matrix.py` 驱动已安装的 Windows 应用（正常关闭、窗口 X、强杀应用、
host 崩溃、第二次启动、两个引擎）。`tests/terminal/ssh_disconnect.py` 在 Linux 机器上真实断开一个 ssh 会话。
