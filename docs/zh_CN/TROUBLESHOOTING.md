# 出了什么问题？

<a id="something-went-wrong"></a>

常见问题及处理办法，适用于桌面应用、终端版（Terminal，`strata launch`）和安装程序。返回 [README](../../README.md#something-went-wrong)。完整的错误信息表（含旧引擎的修复方法）见[细节](DETAILS.md#troubleshooting)。

## 桌面应用与终端版

<a id="the-desktop-app-and-the-terminal"></a>

**警告提示 Strata "couldn't find an engine for this computer"（找不到适配这台电脑的引擎）。**
引擎是与你的显卡通信的那部分组件。Strata 只给出警告，仍会下载模型，模型文件也会留在你的电脑上，但在装好合适的引擎之前模型可能无法运行。常见原因是尚未安装任何引擎、而更新源里也没有适配你显卡的引擎，或者找到的引擎是给另一种显卡的，或者显卡驱动太旧。在应用中打开 Settings，然后打开 Engine。在终端版中，这条信息会指明下一步：用你拿到的更新源运行 `strata update --feed <address>`（也可以向发给你 Strata 的人索要适配你显卡的引擎），然后再次运行 `strata launch`。

**"This model needs a newer engine."（此模型需要更新的引擎，应用内提示），或一个名为 Update engine 的按钮。**
有些模型需要比已安装版本更新的引擎：Gyro 需要 0.1.45.1 或更新，NVFP4 需要 0.1.44，NVFP4（NVIDIA，为 FP4 训练）需要 0.1.45。在应用中打开 Settings，然后打开 Engine，按下 Update engine。它会先征求确认，保留你的模型文件和设置，并要求先卸载正在运行的模型。在终端版中：`strata update --check` 说明将会发生哪些变化，`strata update` 负责安装（只有在试启动成功之后才切换到新引擎）。如果你安装的引擎已是最新、提示却仍在，说明更新源里暂时还没有适配你显卡的更新引擎。

**在 Mac 上："Strata is damaged and can't be opened"（Strata 已损坏，无法打开）、"cannot be opened"（无法打开）或 "unidentified developer"（身份不明的开发者）。**
文件本身没有问题。macOS 会给浏览器下载的每个文件打上标记，只有经 Apple 检查过的被标记程序才允许运行。Strata 的发布构建经过了 Apple 检查（"notarised"，公证）：把 Strata 拖到 Applications 并打开；第一次时 macOS 会问 "Are you sure you want to open it?"（确定要打开吗？），你按下 Open 即可。未经公证的测试构建会提示"已损坏"。不要把它移到废纸篓。打开终端应用并移除标记，用与你手上文件相符的那一行（把文件从 Finder 拖进终端窗口即可输入其路径）：

    xattr -dr com.apple.quarantine ~/Downloads/<the .dmg file>        # the .dmg, before opening it
    xattr -dr com.apple.quarantine /Applications/Strata.app           # the app, after dragging it over
    xattr -dr com.apple.quarantine <the folder you unzipped>          # an unzipped Terminal folder

然后再次打开它。如果 macOS 只提示 "the developer cannot be verified"（无法验证开发者），你也可以先尝试打开一次，然后打开
系统设置 > 隐私与安全性，在关于 Strata 的提示旁按下 "Open Anyway"（仍要打开）。用
`curl ... | sh` 完成的安装不带标记，完全不需要这些操作。想确认自己属于哪种情况：`spctl -a -vv /Applications/Strata.app`
对公证构建会回答 "accepted" 和 "Notarized Developer ID"。（构建如何签名：`packaging/mac/README.md`。）

**检查已安装内容：`strata doctor`。**
它是对已安装引擎组件的只读检查（不改动任何东西）：一个已保存的模型需要哪个引擎和哪些辅助文件，以及它们是否存在。`strata doctor --live` 还会询问计算机它看到了哪些显卡，并与模型设定要使用的显卡做比较；`--output json` 以数据形式打印同样的内容。把它的输出放进 bug 报告。

**为 bug 报告发送日志。**
在应用中：Settings、Engine、Diagnostics、View logs，然后 Save logs。它会把一个文本文件
`strata-diagnostics-<date>.txt` 写入你的下载文件夹并显示其位置。该文件里包含引擎的消息、失败的步骤，以及每个模型服务器日志的末尾。在终端版中，请发送 `strata doctor` 的输出。如果 Linux 应用窗口崩溃或打不开，请在终端中给 AppImage 加上 `--save-logs` 运行；它会把崩溃日志和其他日志写到下载文件夹里的同一个文件中。发送之前请先浏览一遍日志，绝不要发送 API key。

## 安装或启动期间

<a id="while-installing-or-starting"></a>

**Strata 第一次启动时我的电脑卡死了，或变得非常慢。**
启动期间这是正常的，首次尤其明显。Strata 会把 35-55 GB 载入你的内存，为显卡锁定其中一部分，并算出模型有多少能放进你的 GPU。鼠标可能冻结几分钟。
**请耐心等待，不要关闭窗口。** 之后的启动会快得多。10 分钟后仍然卡死？重启电脑，关闭其他程序（浏览器占用大量内存），再试一次。如果反复出现，请选择更小的规模（Q2_0
或 IQ2_XS）。

**下载或安装中途停止了。**
再次运行 `START-HERE.bat`（Linux：`./setup.sh`）。它会从停下的地方继续。

**提示 NVIDIA 驱动太旧。**
更新驱动（NVIDIA App 或 [nvidia.com/drivers](https://www.nvidia.com/drivers)），重启电脑，然后再次运行
`START-HERE.bat`。

**提示端口 8080 已被占用。**
Strata 已经在运行了。找找它的窗口。或者是其他程序占用了该端口：`START-HERE.bat --port 8081`。
只有真正被占用的地址才会报 "already in use"。任何其他原因（Windows 保留了该端口，或
配置中的 `"host"` 不是这台电脑的地址）都会按操作系统给出的原始信息打印出来，并说明该改什么（#769）。

**"Windows blocked the Strata engine (or the image encoder) ... Smart App Control"（Windows 拦截了 Strata 引擎（或图像编码器）…… Smart App Control）。**
在干净的 Windows 11 上，Smart App Control 会拒绝未签名的程序，而 Strata 的 `strata.exe` 和
`strata-vision.exe` 并未签名（#735）。关掉它（Windows 安全中心 > 应用和浏览器控制 > Smart App Control
设置；不重置 Windows 就无法再打开），或者，如果只有图像编码器被拦截，运行
`START-HERE.bat --setup --vision no`。

**更新后新引擎行为异常。**
更新会把被替换掉的引擎保留在 `engine\.previous` 中（保留一代，约 210 MiB）。
`python setup.py --rollback-engine` 会把它放回去（并把较新的那个留在那里：再运行一次即可前进）（#670）。

**UPDATE.bat / update.sh 提示 "git pull did not succeed"，或 "has no commit in common"（没有共同提交）（#1276）。**
仓库历史在 2026-10-06 被整理过，因此在那之前做的 clone 无法用 `git pull` 更新。
0.1.40.2 的脚本能处理这种情况：当没有任何被跟踪的文件被编辑时，它们把你的旧提交保留在
`pre-cleanup-backup` 分支中，并把 clone 迁移到新历史；否则它们会打印出两条命令交给你执行。你的模型、
设置和引擎都是未被跟踪的文件，绝不会被触碰。更早的 UPDATE.bat / update.sh（0.1.40.2 之前）无法自己完成这件事：
在 Strata 文件夹中运行一次下面这几条命令，之后照常使用 UPDATE.bat（如果 `git status`
列出了被编辑的文件，先执行 `git stash`）：

    git fetch origin
    git branch pre-cleanup-backup
    git checkout -B main origin/main

**Python 或构建工具装不上。**
按它列出的内容安装（会打印链接），然后再次运行。已经完成的部分都会保留。

**Linux：引擎编译不过（`unsupported GNU version`，或 `cospi`/`sinpi`/`rsqrt` 报 `exception specification is incompatible`）。** CUDA 工具包与较新 Linux 之间的两种已知不匹配（#601）：
- gcc 高于 14（Ubuntu 26.04 默认是 15）：CUDA 12.x 和 13.0 拒绝它。安装 g++-14 并运行
  `CXX=g++-14 ./setup.sh`（CC、CXX 和 CUDAHOSTCXX 指定构建所用的编译器；安装程序会把它们交给 CMake，
  所以已配置好的构建文件夹也会采用它们，#1645）。
- glibc 2.43 配 CUDA 12.9：工具包的数学头文件与 glibc 的冲突，在 CMake 的第一次测试编译中就会失败。
  改用 CUDA 12.8 或 13.x。安装程序会取它能找到的最新工具包；`STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc
  ./setup.sh` 可强制它使用那一个（只用那一个）。

**第一次启动要花好几分钟。**
它正在把 34-55 GB 读入内存；只要文件还在操作系统缓存里，第二次启动就快得多。从任务计划程序启动时可能慢 24 倍：见[开机自启](DETAILS.md#running-it-at-startup-task-scheduler)。
在 Linux 上，引擎现在会请内核提前读取模型文件（在一台电脑上，冷启动从约 920 s 降到 70 s）；`STRATA_READ_AHEAD=0` 可关闭这一行为。在 Linux 上、透明大页设为 `always` 时，arena 不再在其之上额外请求
`MADV_HUGEPAGE`（一台内存碎片的机器曾花几分钟做内存规整，#771）；`STRATA_NO_ARENA_THP=1` 在任何设置下都跳过
该请求。

**安装程序提示引擎与 GitHub 的校验和不匹配。**
安装程序会把下载的现成引擎与 GitHub 公布的 SHA-256 比对。不匹配说明文件已损坏或在传输途中被改动，所以安装程序会删掉它并停止；重新运行安装程序即可重新下载。如果 GitHub 没有给出校验和（离线、被限流、使用你自己的镜像），安装程序会警告并仍然安装。`STRATA_SKIP_SHA256=1` 会跳过校验，仅在你坚持时使用。

## 回答期间

<a id="while-it-answers"></a>

**非常慢，硬盘灯一直闪。**
你的电脑空闲内存不够用了。关闭其他程序，或选择更小的规模（Q2_0 或 IQ2_XS）。

**回答中途停下，提示 "the engine stopped unexpectedly"（引擎意外停止）。**
通常是因为内存不足（在 Linux 上系统会因此终止引擎）。重新发送你的消息即可：Strata 会自行启动
引擎。如果反复出现，请关闭其他程序或选择更小的规模。

**提示说提示词超出上下文。**
这段对话比你选择的上下文更长。新建一个聊天，或运行 `SETUP.bat` 选择更大的
上下文。

**比表格里的速度慢。**
插在显卡上的显示器和其他 GPU 程序会从专家缓存（expert cache）中占用显存；内存运行在低于标称频率（在 BIOS 中启用 EXPO/XMP）会让 CPU 那一半变慢。

**一条早前返回空的回复从聊天记录中消失了（0.1.40，#843）。**
服务器会省略没有文本的已完成助手轮次，这样下一条回复就不会照抄这些空轮次。
在服务器的环境中设置 `STRATA_KEEP_EMPTY_TURNS=1` 可像以前那样渲染它们。

**Turing 显卡（RTX 20）读取约 90K token 以上的提示词时结果不同（0.1.40，#743）。**
提示词的 top-k 选择在那里使用更宽的内核，id 相同。`STRATA_TOPK_STREAM=0` 可恢复旧内核。

**使用 `--expert-cache N` 时，decode（解码）比 `auto` 慢好几倍（#781、#831）。**
显式的 N 是一个字节预算，槽位写入之后就不会再与空闲显存核对。在被填满的卡上（日志显示 `0 MiB of VRAM free with everything loaded - LOW`），Windows 会对 GPU 内存做分页，仅多出几百个槽位就会让 decode 从 100 tok/s 掉到 14 tok/s。使用 `--expert-cache auto`，或用更小的 N。

**内存较小的电脑（约 32 GB）在 `--resident-budget-gib N` 下出现停顿（#649）。**
该预算被固定（锁定）在内存中，留给操作系统和 CPU 工作线程的空间就很少。`STRATA_RESIDENT_PIN=0` 可让它保持可分页；更小的预算也有效。

**在 32 GB 的电脑上，0.1.40 使用 `--resident-budget-gib N` 时 decode 变慢，且硬盘被高强度读取（#1194、#1116、#1085）。**
启动时那句 "the file tier reads unbuffered" 表示内存副本之外的专家是直接从硬盘读取的。
0.1.40 在空闲内存装不下全部专家时就会选择这种方式；但反复回来的总是同样那批专家，而一个只有其规模二十分之一的缓存就能服务大部分重复访问。0.1.40.2 会申请那么大的缓存（在 4 GiB 余量之外至少 1.5 GiB），否则通过缓存读取，在一台 20 到 32 GB 的机器上实测 decode 快 7% 到 118%，硬盘流量只有三分之一到六分之一。在 0.1.40 / 0.1.40.1 上：`STRATA_UNBUFFERED_LOAD=0` 效果相同，`=1` 强制使用无缓冲读取。

**Linux：引擎分配页锁定的专家副本时，启动过程被 OOM 杀掉（#1250）。**
页锁定页由驱动一次性给出，无法等待内核归还文件缓存，因此在模型文件的缓存挡在路上时（真正空闲的内存很少，"available" 很多），主机可能耗尽内存。0.1.40.2 会在页锁定这份副本之前，先归还它映射的模型文件的缓存页；当真正空闲的内存仍不够覆盖时，它以 1 GiB 为步长取页，只要可用内存（含 cgroup 限制）保持在余量之上，就对已取到的部分做页锁定，剩下的部分保持常驻但可分页，并给出警告（"only N of M GiB could be page-locked"）：回答完全相同，只是可分页部分的副本送往 GPU 时要经过 CPU。`STRATA_PIN_GUARD=0` 关闭这一机制，`STRATA_PIN_RESERVE_GIB=N` 设置必须保持空闲的内存量（默认：`STRATA_RESIDENT_HEADROOM_GIB`，4 GiB）。

**Windows 显示驱动重置，随后电脑蓝屏（回答期间先 0x141，再 `nvlddmkm` 中的 0x116）（#961）。**
这是 NVIDIA 驱动的问题，不是 Strata 的问题：报告显示在同一驱动（610.88，RTX 4080 SUPER）上其他 CUDA 程序也会出现同样的卡死。换一个驱动试试（Studio 版，或更旧的版本），并降低功耗上限；如果只在 `--spec 4` 时出现，在等待驱动修复期间可以用 `--spec 0` 避开 verify 窗口。

**RTX 50 显卡（使用 CUDA 13.2 的源码构建）输出乱码，或读错提示词（#892、#968）。**
CUDA 13.2.0 和 13.2.1 的编译器（nvcc 13.2.51）为 sm_120 错误地编译了引擎的部分内核：在我们的 RTX 5070 上，IQ2_S 和 IQ3_S
的结果是错的（测试中相对误差 0.5 到 1.0），用 13.2 如此，用 13.0 则正确。现成引擎是用 13.0 构建的。如果你自己编译，请用 CUDA 13.0、13.1 或 13.2.2（13.2.2，nvcc build 13.2.86，已修复该问题；旧版本可以与新版本并存：`STRATA_NVCC=<path to its nvcc>`）；
安装程序在为这类显卡发现 13.2.0 或 13.2.1 时会警告，并在已装有更旧的 13.x 时改用它。

**图片被拒绝，或者很慢。**
"this server was started without the vision encoder"：该模型是按纯文本配置的（有人选了 "No"，或
引擎没有图片读取器）——用 `--vision yes`（或 `--vision gpu` / `--vision cpu`）重新运行安装程序，或者在应用中、或在 `strata launch` 的 Pictures 步骤里打开图片。只要模型能读图，图片默认就是开的。显存低于 24 GB 时读取器运行在 CPU 上，24 GB 起运行在显卡上，NVIDIA 和 AMD 显卡皆然。需要好几秒的图片（8 核上 300 个图像 token 约 3 s，token 越多越久）由 CPU 上的编码器读取；
`--vision gpu`（NVIDIA 和 AMD，约 1.4 GB 显存）可把它降到 0.1-0.5 s。在 AMD 上，显卡读取器是 `strata-vision-hip`；如果它
无法启动，服务器会打印 "the GPU picture reader did not start (...); pictures are read on the CPU" 并继续用
CPU 读取器。Mac 应用和 Mac 终端版还没有图片读取器。在 Intel Arc 上，图片目前默认关闭（`--vision yes` 可以打开）。

## AMD 显卡

<a id="amd-cards"></a>

**Linux 上："No AMD GPU found (the amdgpu driver's KFD topology is empty)"（未找到 AMD GPU，amdgpu 驱动的 KFD 拓扑为空）。**
内核的 amdgpu 驱动没有为这张卡加载。集成 Radeon 显卡被列为不支持；安装程序
会列出它找到的每张卡以及 Strata 能否使用它。

**引擎在启动时退出，并打印出显卡名称、其架构和构建支持的列表。**
引擎是为另一张卡编译的（例如把 Strata 文件夹搬到了另一台电脑上）。运行
`./setup.sh --setup --backend hip`：它会为这张卡的架构编译引擎。

**Windows 上，引擎在打印任何内容之前就以 `amdhip64_7.dll` 中的 `0xC0000005` 退出（#654）。**
你环境中的两种情况都可能导致它，现在服务器会在启动引擎之前把两者都修好（它会在日志中说明）：`HIP_PATH`、`HIP_DEVICE_LIB_PATH` 或 `LLVM_PATH` 指向一个已不存在的 ROCm 文件夹（被删掉的构建），
以及引擎无法在其中创建文件的 `TEMP`/`TMP` 文件夹（AMD 的运行时通过临时文件编译它最初的那几个内核）。如果你手动启动 `strata.exe`，请自己修好它们：`set HIP_PATH=`，并把 `TEMP` 和 `TMP` 指向一个正常文件夹，例如 `C:\Temp`。

**Windows AMD：驱动重置（VIDEO_ENGINE_TIMEOUT_DETECTED / 屏幕闪烁 / 引擎在回答中途死掉）。**
Windows 给 GPU 大约 2 秒的响应时间；显卡做不到时它就重置驱动（TDR）：屏幕
闪烁，引擎死掉，事件查看器显示 `VIDEO_ENGINE_TIMEOUT_DETECTED`（#613，Linux 上另有 #579 和 #541）。
Strata 0.1.39 及更新版本会在日志里说 "the GPU stopped responding"。这一现象曾出现在一张 gfx1201 显卡上开启 KV 流式传输
（`--kv-resident 32768`，安装程序从 64K 上下文起会开启）读取长提示词时。请按顺序一次试一项，并告诉我们什么起了作用：
1. **把证据发给我们。** Strata 文件夹中 `strata-<model>.log` 的最后约 80 行（服务器窗口显示
   同样的内容）、你的 AMD 驱动版本（AMD Software > System）、你的显卡和 Windows 版本、Strata
   版本，以及确切的操作步骤（上下文大小、提示词多长、由哪个应用发送）。如果 Windows 写了
   dump，`C:\Windows\LiveKernelReports\WATCHDOG`（或 `C:\Windows\Minidump`）中最新的 `.dmp` 会有帮助；它不含
   聊天内容，只有驱动状态。
2. **找到卡住的那一步：** 在 `strata-<model>.json` 的 `"env"` 块中加入 `"STRATA_PF_STEP_SYNC": "1"`
   并重启。此后提示词路径会在每一步之后等待 GPU，并记录任何超过 250 ms 的步骤，日志因此能点名是哪一步。它会变慢；用于诊断，之后请移除。
3. **`"STRATA_KV_HOST_DMA": "1"`** 加在同一个 `"env"` 块中：提示词期间不再有 GPU 内核写 K/V 的内存副本，改为通过 DMA 复制。回答结果相同。
4. **不用 KV 流式传输：** 重新运行安装程序并选择 32K 上下文（或从 JSON 的 `"args"` 中移除 `--kv-resident` 及其数值，并改用能放进显存的上下文）。
5. **更小的提示词分块：** 改动 `"args"` 中的 `--prefill`（例如 `--prefill 512`）：这样每次 GPU 启动在 Windows 的两次检查之间做的事更少。提示词的读取会变慢。
6. **可选，由你决定：给 Windows 更多时间（`TdrDelay`）。** 默认是 2 秒。在注册表
   `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` 中，把 DWORD `TdrDelay` 设为 `10`（秒），
   可让一个长的 GPU 任务在 Windows 重置驱动之前完成；然后重启电脑。这只是掩盖症状，并不能修好真正挂死的 GPU，而且挂死的 GPU 会让屏幕卡住那 10 秒。Strata 自己不会改动任何 Windows
   设置；要撤销，删掉该值即可。
7. **驱动：** 为你的显卡使用最新的 AMD Software 发布版（Adrenalin，而不是 Windows Update 的驱动），安装后重启。
   如果问题是驱动更新之后才开始的，请把两个版本都告诉我们；回退一个版本也是合理的验证。

**RAM 明明空闲，ROCm 上的大块页锁定（pinned）宿主内存分配却失败。**
见 [AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration)：mapped expert 模式可以避开完整的页锁定 arena。

**AMD（Linux）上出现 "verify: timed out at layer N" 或 "no progress for 60 s"，多发生在使用 `--mmap-experts` 时。**
两份报告分别通过在服务器 JSON 的 `env` 块中加入 `"GPU_PINNED_MIN_XFER_SIZE": "1048576"`（RX 6800）或 `"HSA_USERPTR_FOR_PAGED_MEM": "0"`（两台
R9700）并重启解决了问题。属实验性做法，各只在一台机器上；在 systemd 单元中请用
`MemoryMax`，不要用 `MemoryHigh`。细节：[AMD_HIP.md](AMD_HIP.md#linux-verify-timeouts-while-the-kernel-reclaims-host-memory-experimental-workarounds)。

## 仍然卡住？

<a id="still-stuck"></a>

查阅[完整故障排查表](DETAILS.md#troubleshooting)，或提交一个
[issue](https://github.com/Niko1221/Strata/issues) 并附上日志：在应用中，附上 Save logs 写出的文件（见
[上文](#the-desktop-app-and-the-terminal)）；在终端版中，附上 `strata doctor` 的输出；使用安装程序时，附上 Strata 文件夹中的
`strata-<model>.log`。请说明你用的是哪一种（应用、终端版还是安装程序）、你的 Strata 和
引擎版本以及你的显卡。
