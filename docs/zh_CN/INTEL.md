# Strata 跑在 Intel Arc 上

<a id="strata-on-an-intel-arc"></a>

Strata 的引擎是 CUDA。在 Intel Arc 上它以 **Strata 自己的引擎、移植到 SYCL** 的形式运行
（`sycl/`，见下方"Intel 上的引擎本体"一节），藏在同一个 Strata 服务器后面：OpenAI 与
Anthropic API、流式、工具调用、MCP 和网页应用都不变。表格里 llama.cpp 的 SYCL 后端是对照点
（它跑同一个 GGUF，速度约三分之一）。

<a id="stratas-engine-is-cuda"></a>

所有 Intel 专属的东西都在 `sycl/` 里；upstream 的任何共享文件都没有被改动，所以合并 upstream
保持干净。`sycl/setup_intel.py` 和 `sycl/serve/server_intel.py` 从外部包装 upstream 的
`setup.py` 和 `serve/server.py`。

<a id="everything-intel-specific-lives-in"></a>

本文在 **Arc Pro B70（32 GB）** 上编写并测量，跑 Coder（IQ1_M），Ubuntu 24.04，插在
PCIe 3.0 x8 插槽（卡在那里训练到 Gen3 x8；它能到 Gen5 x16）。

<a id="written-for-and-measured-on-an"></a>

## 你得到什么（2026-10-01）

<a id="what-you-get-2026-10-01"></a>

| | NVIDIA（Strata 引擎） | Intel Arc，Strata SYCL 移植版 | Intel Arc，llama.cpp |
|---|---|---|---|
| models | 全部四个 | Coder IQ1_M；原版 IQ2_XS 也行（超出 VRAM 的 expert 放进 host 镜像：58-64 tok/s） | Coder（装得进 VRAM） |
| 模型文件 | 同样的 GGUF | 同样的 GGUF + 一个原生包 | 同样的 GGUF |
| 模型住在哪 | expert 在 RAM，热的在卡上 | 每个 expert 都在 VRAM（`--stream-experts`：无 host 副本），shard 2 的查表按行从 SSD 读 | shard 1 全部在卡上，shard 2 从磁盘换页 |
| 需要 RAM | 32-64 GB | 很少（23 GB 就够） | 很少（23 GB 就够） |
| decode，Coder IQ1_M | RTX 5070 上 44-51 tok/s | **78.2 tok/s**（19-token prompt，256 贪心 token），2,184-token prompt 后 **75.7**，40K 后 69，128K 后 66.6，256K 后 55.9；走 API 48-72 | 23-25 tok/s |
| prompt 读取 | ~1,870 tok/s | 2,184 token 时 **790 tok/s**，40K 时 1,117，128K 时 888，256K 时 757 | ~150 tok/s（105K 时 424） |
| 推测解码（MTP） | 有 | 有（基础检查点的草稿层；代码上草稿接受率 70-85%） | 无 |
| 图像 | 有 | 还没有 | 还没有 |
| 上下文 | 最高 262K | 实测 256K，那里 decode 56 tok/s（`--kv-resident`：KV 在锁页 host 内存，被注意的窗口在 VRAM） | 实测上限 131K |

推测解码下的 decode 速度取决于文本：代码草拟得好，散文差些（见"速度取决于文本"）。SYCL 的
数字是引擎的贪心运行（如下方"如何手动运行"，256 个新 token），引擎 0.1.31-sycl
（2026-10-01；0.1.32-0.1.38-sycl 完全复现它们）；下方"Decode 第二轮"列出每项改动买到了什么。

<a id="decode-speed-with-speculative-decoding-depends-on-the-text"></a>

## 安装

<a id="setup"></a>

在 Ubuntu 24.04 或更新版上，从零开始。前提是已有 Intel GPU 驱动（`xe` 或 `i915` 内核驱动加
计算运行时；[INTEL_ARC.md](INTEL_ARC.md)，"What you need"）：`ls /dev/dri` 能看到
`renderD128`。

<a id="on-ubuntu-2404-or-newer-from-nothing"></a>

1. **Docker，一次。** 引擎跑在容器里，所以 PC 需要 Docker、不需要 oneAPI：

        sudo apt install docker.io git
        sudo usermod -aG docker,render $USER

   注销再登录（或重启）让组生效。`docker ps` 必须不用 `sudo` 就能工作；`render` 组让容器
   能打开 GPU。
2. **获取 Strata 并构建引擎**（下方"如何构建"，第一次约 10 分钟）。
3. **运行 setup：**

        ./setup.sh --backend sycl [setup.py's options, e.g. --model IQ3_S --context 32768 --port 8085]

   `setup.sh` 在 `.venv` 里建自己的 Python 环境，不碰 Ubuntu 的 "externally managed" 系统
   Python。**不要自己跑 `python3 sycl/setup_intel.py`：** 它会把 setup 的包装进运行它的
   那个 Python，而 Ubuntu 的会拒绝（`externally-managed-environment`）。
   `./setup.sh --backend sycl` 就是那一条命令，首次安装和之后每次启动都是它。

<a id="-docker-once"></a>

Setup 就是 upstream 的 `setup.py`，把 Intel 的步骤换了进去（`sycl/setup_intel.py` 导入它并
替换那些步骤；setup.py 本身未改）。模型选择、下载、打包、分词器、MTP 草稿层以及上下文和 KV
的提问都是 setup 自己的。变化的是：

<a id="setup-is-upstreams-setuppy-with"></a>

- **GPU 检查：** 在 sysfs 里找到 Arc（vendor 8086，挂在 `xe` 或 `i915` 下），通过 setup 的
  AMD 路径提供。那条路径本地构建、没有图像。
- **引擎步骤：** 用 SYCL 构建（`build-sycl-aot/strata` 或 `build-sycl/strata`，由
  `sycl/serve/strata-sycl.sh` 在 `strata-sycl-dev` 镜像里运行），而不是编译 CUDA 或 HIP。
- **RAM 规则：** 在 `xe` 卡上不适用。CUDA 引擎把所有 expert 放在 RAM；移植版从 GGUF 流式
  进 VRAM（`--stream-experts`），所以 RAM 只决定 KV 流式。`--check` 按 VRAM 列出装得下的。
- **配置：** 用容器的路径和 `"backend": "sycl"`。VRAM 预留 32K 及以下为 1,024 MiB，超过它
  用 4,096-token prompt 分块时 2,048 MiB；小于 12 GB 的卡 300 MiB；你给的
  `--vram-reserve-mib N` 原样保留。RAM 装得下 KV 时，64K 起启用 KV 流式
  （`--kv-resident 32768`）。早先配置里的 `model_switcher` 或 `sampling` 块会保留。
  `run-<model>.sh` 启动 `sycl/serve/server_intel.py`。
- **Arc A 系列（`i915`，例如 8 GB 的 A750）：** 不同配置，见下方"Arc A750 与其他 Alchemist
  卡"。
- **引擎设置：** 配置的 `"env"` 块能到达引擎。`strata-sycl.sh` 把每个以 `STRATA_`、
  `ONEAPI_`、`UR_`、`IGC_`、`SYCL_` 或 `ZES_` 开头的变量转发进容器，并在环境没有另行说明
  时设置 `STRATA_VERIFY_DEVICE_PLAN=1`、`STRATA_VERIFY_NO_HOST=1` 和
  `STRATA_STAGER_THREADS=12`（例如 `"env": {"STRATA_VERIFY_NO_HOST": "0"}`）。

若未来的 `setup.py` 去掉了本流程依赖的某个步骤，它会带消息停止，而不是写一个错误配置。模型、
包和检出必须放在 `strata-sycl.sh` 挂到 `/work` 的文件夹之下（检出上一级，或
`STRATA_SYCL_ROOT`）。同一台 PC 上更早的 Strata 安装把数据文件夹记在
`~/.config/strata/settings.json`；传 `--data-dir <检出上一级>/Strata-data` 把这次的放到挂载
之下。指向挂载之外文件夹的符号链接在容器里不工作；硬链接可以。

<a id="-gpu-check"></a>

然后 `run-<model>.sh`（或再跑一次 `./setup.sh --backend sycl`）启动模型。30 GB 模型的首次
启动约两分钟。`--port N` 与 `--host 0.0.0.0` 与 upstream 的 setup 相同。

<a id="then-run-modelsh"></a>

## 如何构建

<a id="how-to-build-it"></a>

引擎在 Docker 里构建，在 `strata-sycl-dev` 镜像中（oneAPI 2026.1 编译器、oneMKL 和
Level Zero，基于社区 llama.cpp SYCL 镜像），所以 PC 不需要 oneAPI。从检出目录（其文件夹这里
叫 `Strata`；上一级是容器挂到 `/work` 的目录）：

<a id="the-engine-is-built-in-docker"></a>

    cd Strata
    docker build -t strata-sycl-dev sycl/tools          # once: about 6 minutes, a 13 GB image
    H=$(basename "$PWD")

**Arc B 系列**（Pro B70：`bmg-g31`；B580、B570 和 Pro B60：`bmg-g21`；镜像里
`ocloc ids bmg-g21` 列出设备 id）。提前编译代码，首次启动不编译：

<a id="arc-b-series"></a>

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

**Arc A 系列**（Alchemist：A750、A770、A580、A380）：没有 `AOT=`。代码在首次启动时从
SPIR-V 即时编译（JIT，约一分钟）。没有人给这些卡跑过 AOT 构建。

<a id="arc-a-series"></a>

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

构建约 3 分钟，以 `BUILD EXIT 0` 和 `errors: 0` 结束。二进制是 `build-sycl-aot/strata`
（B 系列）或 `build-sycl/strata`（A 系列）；setup 两个都找。拉取改动了
`sycl/tools/Dockerfile` 的 Strata 更新后要重跑 `docker build`，每次引擎更新后重跑
`docker run`（setup 会在引擎过旧时说明）。

<a id="the-build-takes-about-3-minutes"></a>

## 这块 GPU 上要紧的事

<a id="things-that-matter-on-this-gpu"></a>

- **`SYCL_CACHE_PERSISTENT` 必须是 0。** 持久 JIT 缓存在 Xe2 上首次编译时段错误。启动脚本
  会设置它；手动跑 llama-server 时也要设。
- **整个模型上卡**（`--n-gpu-layers 999`），除了 `per_layer_token_embd.weight`——shard 2
  那唯一的 28.8 GB 张量。`--override-tensor per_layer_token_embd=CPU` 把它留在 host 内存、
  mmap、按行从 SSD 换页，模型作者们就是这样服务的。Host RSS 保持在约 2 GB。
- **Thinking。** 模型先推理再回答。Strata 的网页应用有这个设置；API 客户端把
  `{"reasoning_effort": "none"}` 放进配置旁边的 `strata-<model>.shared-settings.json`，或
  每个请求发 `chat_template_kwargs: {"enable_thinking": false}`。两者都不带时，小的
  `max_tokens` 会全部花在 think 块里，回答看起来是空的。
- **加载期间 `/health` 返回 503**；服务器改为轮询 `/props`。

<a id="-sycl_cache_persistent-must-be-0"></a>

## 装得下多少上下文

<a id="how-much-context-fits"></a>

架构让这一项很小：48 层里只有每第四层是完整 attention（12 层）；其余 36 层是
gated-delta-net 层，固定大小的递归状态不随上下文增长。每 token，12 个 attention 层在 q8_0
下保存 2 个 KV 头 × 256 ×（K + V），外加稀疏注意索引器的键。**实测：每 token 约 20 KiB**
（其他不变时，VRAM 每 32k 上下文增长 0.6 GB）。

<a id="the-architecture-keeps-this-small"></a>

| 上下文 | 已用 VRAM，模型加载后 | 32 GB 上的余量 | 状态 |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | 实测，默认 |
| 65,536 | 29.0 GB | 2.9 GB | 实测，可加载 |
| 98,304 | 29.6 GB | 2.3 GB | 实测，可加载 |
| 131,072 | 30.3 GB | 1.5 GB | **实测，实用上限**：104,798-token prompt（窗口的 79%）以 424 tok/s 读了 247 秒，然后正常回答 |
| 163,840 | ~31.0 GB | ~0.9 GB | 未尝试：低于 1.2 GB 安全余量 |
| 262,144 | ~32.6 GB | 无 | **装不下——要它会拖垮主机** |

**不要要求超过装得下的量。** 在这个驱动上，超出 VRAM 的 GPU 分配不会失败：xe 驱动把缓冲区
逐出到 host RAM，内核内存耗尽，机器活锁直到硬件看门狗复位。2026-09-29 一个 262K 请求就发生
了这样的事，llama.cpp 自己的 "failed to fit params" 检查触发得太晚没能阻止。先算 KV 大小，
留出 1.5 GB。

<a id="-do-not-ask-for-more-than-fits"></a>

Prompt 读取随长度变快，因为工作批处理得更好：2.7k-token prompt 约 150 tok/s，105k-token 的
424 tok/s。所以读完整个 128k 窗口约五分钟，不是十五分钟。

<a id="prompt-reading-gets-faster-with-size"></a>

## Intel 上的引擎本体：SYCL 移植版（`sycl/`）

<a id="the-engine-itself-on-intel-the-sycl-port"></a>

上面的一切通过 llama.cpp 跑模型。这是另一半：Strata 自己的引擎——50 个 CUDA 内核和驱动它们的
host 代码（流、事件、图捕获、锁页内存）——用 oneAPI 为 Arc 构建。它是对整棵源码树的迁移，不是
新后端：引擎没有可以插入的后端接缝。

<a id="everything-above-runs-the-model-through"></a>

**布局。** `sycl/src` 和 `sycl/include` 镜像源码树、只放移植改动过的文件；SYCL 构建
（`sycl/CMakeLists.txt`）从原位置取其余所有源文件。没有编辑任何 upstream 文件。
`sycl/include/dpct/` 是随带的 SYCLomatic 辅助库，所以移植版不需要迁移工具就能构建。

<a id="layout"></a>

**它是怎么做出来的，以便重做。**

<a id="how-it-was-made-so-it-can-be-redone"></a>

1. `sycl/tools/Dockerfile` - 开发镜像：llama.cpp SYCL 镜像加 SYCLomatic（`dpct` 2025.3）、
   ninja，以及 `sycl/tools/get-cuda-headers.sh` 从 NVIDIA 的 pip wheel 里取出的 CUDA 12.8
   头文件（dpct 解析 CUDA；它需要的是头文件，不是工具包）。
2. `sycl/tools/migrate.sh` - 为 86 个接触 CUDA 的翻译单元写编译数据库，并对它们跑 dpct。
   85 个迁移成功；dpct 没有报告无法迁移的行，约 1,400 条建议性提示。
3. `sycl/tools/fixups.py` - dpct 做错或做不到的，写成一个幂等脚本、每项一条理由。要紧的有：
   - CUDA 的 null stream 指默认流；dpct 把它变成了空 `sycl::queue*`。现在每个流转换都经过
     `strata::q_of()`（`sycl/include/strata/sycl_queue.hpp`）。
   - `__ldg((const float*) p)` 被写成 `*p`，把 float scale 只读了一个字节（两处，s_gemv）。
   - `__fadd_rn(a, b ? c : d)` 丢了括号（两处）。
   - ggml 查找表经由内核参数传递时模板用错了表；它们是设备代码直接读的普通 `static const`
     数组，与 ggml-sycl 相同。
   - dpct 2025.3 之后改名的辅助头文件（`entangle`、`chunked_partition`）、图自省与
     `cudaGraphUpload`（无 SYCL 对应物）、`%globaltimer`（阶段分析器读出零）。
4. `sycl/tools/build.sh` - 在镜像里用 icpx 配置 + 构建。两个编译选项是承重的：
   `-fp-model=precise`（icpx 默认快速浮点模型）和设备编译器的
   `-cl-fp32-correctly-rounded-divide-sqrt`——Arc 的 fp32 除法默认不是正确舍入（OpenCL 允许
   2.5 ulp），而 Strata 的量化器经 `amax / 127` 与 ggml 逐字节一致。实测：不带它
   `quantize_act_parity` 有 30.3 万处不一致，带上则没有。

<a id="sycltoolsdockerfile---the-dev-image"></a>

**现状（2026-09-29）。** 整棵树能构建能链接：`strata` 二进制和 22 个内核对等性测试。在 B70
上，22 个里 19 个逐字节或通过容差；其余三个需要模型 fixture（`iq_parity`、`ple_parity`）或
下面的 tensor-core 内核（`qsa_prompt_attn`）。

<a id="where-it-stands-2026-09-29"></a>

| 等期性测试 | 结果 |
|---|---|
| bf16_gemv、cvec、dequant_s2、elementwise、gdn、gr、kv_q4、kv_q8、kv_stream、qsa、quantize_act、rope、router_top10、s2_gemv、s2_gemv_q8、s_gemv、s_gemv_q8k、sampler、shared_expert | 通过 |
| iq_parity、ple_parity | `ctest` 会把 `iq_parity` 的 fixture 生成到构建树（`iq_parity_fixtures`，需要带 numpy 的 python3；没有则跳过）。`ple_parity` 需要 Q2_0 的 shard 2 GGUF 和 dense 包：配置时给 `-DSTRATA_PLE_GGUF=<shard 2 .gguf>`（以及 `-DSTRATA_PLE_FIXTURE_DIR=<放 ple_in.bin 和 ple_out.bin 的目录>`）；缺少输入是硬错误，与 CUDA 构建相同（#1795） |
| qsa_prompt_attn_parity | 需要 tensor-core 内核（见下） |
| kv_hybrid_parity（0.1.27，混合 K8V4 KV） | 追加、收集和 attention 通过；最后一步就是那个 tensor-core 内核 |

**引擎端到端（2026-09-29，同日晚些）。** 它能生成了。Prompt "Write a Python function that
returns the n-th Fibonacci number"，贪心，64 token：

<a id="the-engine-end-to-end-2026-09-29-later-the-same-day"></a>

> `<think>` The user wants a Python function that returns the n-th Fibonacci number. Let me consider the
> options: 1. **Recursive approach** - Simple but O(2^n) time complexity, very slow for large n. 2.
> **Iterative approach** - O(n) time, O(1) ...

同一块卡、同一个 2,185-token prompt、64 个贪心 token（AOT 构建，`--spec 2`，
`--no-prefill-borrow`）实测，对照跑同一 GGUF 的 llama.cpp SYCL 构建：

<a id="measured-on-the-same-card-same-2185-token-prompt"></a>

| | llama.cpp SYCL | Strata SYCL 移植版 |
|---|---|---|
| prompt 读取 | 138-319 tok/s | **560 tok/s**（2,000 token 时 693，300 时 180） |
| decode，仅后缀草稿器（`--spec 2`） | 24.2-26.0 tok/s | 20.2-20.8 tok/s |
| 带 MTP 草稿层的 decode（`--spec 4 --mtp`） | - | 2,185-token 上下文时 **42.7 tok/s**（接受率 81%，2.9 token/轮）；短 prompt 45.3 |

Prompt 读取正是 Strata 设计兑现的地方（oneMKL GEMM 对反量化的 expert，整个模型在卡上）。
Decode 约为 80%：一个推测轮无论多大都要约 53 毫秒，后缀草稿器的接受率只有 9%，所以几乎每轮
只产出一个 token；`--spec 4`（16.9 tok/s）因此比 `--spec 2`（20.8）更差。内核本身在能流式
600 GB/s 的卡上以 130-160 GB/s 读权重（`sycl/probe/bw.cpp`，不可压缩数据）——与 llama.cpp
达到的同一档——热 Q6_K matvec 的五个变体（每行 lane 数、每 warp 行数、展开、16 字节加载、
软件流水；`mmvq_bench`）都落在该区间内。草稿来源才是旋钮：基础 Qwen3.8-Flash-Next 检查点的
MTP 层（`tools/mtp_fetch.py fetch`、`mtp_pack.py --experts q2_0`、`mtp_rt.py`；下载 4.9 GB，
809 MiB VRAM）为 Coder 微调稿草拟的接受率 87%：每 53 毫秒轮 3.4 token、45 tok/s、同样的贪心
token。`mtp.cpp` 路径在相同参数下第一次就通了。

<a id="prompt-reading-is-where-stratas-design-pays"></a>

没有 AOT 时，运行时在首次使用每个内核时 JIT 编译，一个进程第一次跑要约 47 秒；
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` 能让它跨运行保留（15 MB）。

<a id="without-aot-the-runtime-jit-compiles"></a>

如何手动运行（一次贪心测试运行，本节每个数字都是这样测的；在 `strata-sycl-dev` 镜像里，AOT
构建在 `build-sycl-aot/`）：

<a id="how-to-run-it-by-hand"></a>

```
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

- `--stream-experts`（本移植版）：expert 没有常驻 host 副本；12,288 个中的每一个都从 GGUF
  经一个小暂存环进入 VRAM 缓存（`GgufExpertSource`）。Upstream 这个模型需要 32 GB RAM；带此
  参数引擎在 23 GiB 上运行。
- `STRATA_VERIFY_DEVICE_PLAN=1`：GPU 自己规划每层（upstream 的 E-6；那边默认关）。
- `STRATA_VERIFY_NO_HOST=1`（本移植版）：host 等整个窗口图，而不是逐层门环。只有在所有
  expert 都常驻时有效——32 GB 卡上正是如此。
- `--no-prefill-borrow`：prompt 路径不得出借 expert 槽位（借出的 expert 由 host 服务，靠一个
  在这里 GPU 无法可靠看到的标志；长 prompt 不带它会挂起）。

<a id="--stream-experts-this-port"></a>

移植除了能编译还必须做对的事（每条都是 `sycl/tools/fixups.py` 里的条目或一个参数）：

<a id="what-the-port-had-to-get-right-beyond-compiling"></a>

- **Host 映射标志。** Strata 的 decode 是经过 host 映射内存的 GPU/CPU 握手。在 Intel 上
  `volatile` 设备读取不会绕过缓存；系统作用域的原子量会，而且是唯一实测有效的（
  `sycl/probe/doorbell.cpp`，六种变体）。内核运行**期间**的 host 到设备可见性在这个平台上
  始终不可靠，这就是全常驻路径改为等整个窗口的原因。
- **有界自旋。** 永远等不到标志的设备自旋不是一个进程挂起那么简单：xe 驱动会把队列超时并
  逐 GT 节点复位——一个窗口图有 2,400 个——卡会一直楔住直到重启（发生过两次）。每个自旋
  都有上限（`kSpinMax`）。
- **32-lane 子组。** 内核是按 warp 写的；dpct 钉死了 289 个启动中的 134 个，其余会以 Xe2
  默认的 16 运行（`-fsycl-default-sub-group-size=32`）。
- **同步拷贝。** `cudaMemcpy` 会阻塞；dpct 默认的队列 `memcpy` 不等待。配上流式环，expert
  缓存被覆写后的缓冲区填满：非确定的残差，第 5 层出现 NaN。用逐层残差阶梯
  （`STRATA_VERIFY_DEBUG=1` 每层后打印 R）找到的。
- `native_expert_parity` 手工移植：GPU 原生 expert 内核在真实 IQ1_M 行上与 ggml 的 float
  参考一致（相对 1.1e-2，与 CPU 路径同一档）。

<a id="-host-mapped-flags"></a>

Profiling：`sycl/tools/Dockerfile.unitrace` 构建带 Intel unitrace 的开发镜像；引擎外加
`sycl/rank_kernels.py` 跑 `unitrace -d` 给出每内核设备时间。`mmvq_bench` 和
`native_expert_parity NATIVE_BENCH=1` 单独给两个热内核系列计时（先预热时钟：5 毫秒的运行
测的是爬频，不是内核）。

<a id="profiling-sycltoolsdockerfileunitrace-builds"></a>

**0.1.25-0.1.27 合并之后（2026-09-30，同卡同参数，AOT）。** Fibonacci prompt（19 token），
64 个贪心 token：decode 46.5 tok/s（草稿接受率 85%）。2,184-token prompt：prompt 566 tok/s、
decode 38.3 tok/s（该文本接受率 77%）；草稿层的 prompt 遍历现在是 upstream 的批量版
（2,184 token 花 38.6 毫秒，此前 19 token 要 105 毫秒）。等期性：22 个中 19 个，
`quantize_act` 恢复逐字节一致。

<a id="after-the-0125-0127-merge"></a>

**提速工作，2026-09-30（0.1.27 合并之后）。** 在 B70 上用与上文相同的 prompt 实测，是什么
推动了数字（每项改动前后输出相同）：

<a id="speed-work-2026-09-30"></a>

| 改动 | 效果 |
|---|---|
| 每个窗口图和草稿器的图在加载时捕获（`STRATA_WARM_GRAPHS=0`：回到从前） | 第一个请求不再付 250-290 毫秒的捕获 |
| commit 图继续运行、草稿器的轮在自家队列上跑（`Verifier::commit(wait=false)` + `commit_finish`） | 2,184-token 上下文时 decode 38.3 -> 43.0 tok/s，短时 46.5 -> 49.8 |
| 合并 upstream 0.1.29（GDN 递归流水化、块分数读取、采样器） | 2,184-token prompt 765 -> 792 tok/s，该上下文 decode 43.1 -> 45.2 |
| 草稿层的批量 prompt 遍历接手 64 行以下的 prompt（此前退回每 6 token 的图） | 19 token 上它的 prompt 代价 106 -> 16 毫秒 |
| 每内核一个设备模块（`-fsycl-device-code-split=per_kernel`） | prompt 路径里第一次启动（embedding gather）245 -> 1 毫秒 |
| expert 反量化把每个线程一段 FP16 值合成一次向量写（此前一次一个 2 字节写） | 每 expert 反量化 0.085 -> 0.030 毫秒；2,184 token 时 prompt 496 -> 575 tok/s，8,000 时 720 -> 841 |
| expert 点积里的 SWAR 符号比较/减法、一个本地内存常驻规划内核、一个 split-K 融合下投影内核（与单 token 版逐位一致） | 等期性干净，decode 无可测变化；保留 |
| PLE 行读取器默认队列深度 16 -> 64 个阻塞 O_DIRECT 线程（`STRATA_IO_THREADS`；128 和 256 不变：盘在 2,184-token prompt 需要的 2.7 万次随机 4 KB 读上顶到约 85k IOPS） | 该 prompt 的行 466 -> 330 毫秒 |
| 一个短的首个 prompt 分块（`STRATA_PREFILL_FIRST`，默认 256 token）让 GPU 在其余行还在读时就开始 | 2,184 token：711 -> 765 tok/s，8,000：857 -> 986；首 token 时间少约 150 毫秒。分块边界改变 expert GEMM 的舍入，所以长的贪心续写可能晚段分叉（测试 prompt 第 45/64 个 token） |

草稿策略扫描（2,184-token prompt，128 token）：6 的窗口（`--spec 4`）41.6-43.5 tok/s；
`--spec 2` 33.6，`--spec 6` 35-37；`--spec-min-p` 0.3-0.7 在噪声内。Profiles（unitrace，
`strata-sycl-dev:metrics` 配 Intel 的指标库和 `dev.xe.observation_paranoid=0`）：expert 点积
内核受 ALU 限制（77% XVE 活动），dense 投影受内存延迟限制（92-128 GB/s，占用率 84-93%），
一个 decode 轮 80-85% 是内核时间、2,400-2,600 个节点每节点约 5 微秒的启动间隙。

<a id="draft-policy-sweep-2184-token-prompt"></a>

**XMX。** oneMKL 的 FP16 GEMM 已经跑在 XMX 单元上（`xmx_gemm_bench` 里 30-60 TFLOP/s）；
prompt 路径受喂给它们的反量化限制，不是乘积。写了两个 joint_matrix 内核，正确但在这块卡上
输给现有路径，所以两者都保持需显式开启：`xmx_gemm_iq`（从量化行直接融合反量化 + GEMM，比
反量化 + oneMKL 慢 4-5 倍）和 `qsa_prompt_attn_xmx`（mma.sync prompt attention 的移植；
`qsa_prompt_attn_parity` 在 1e-6 of scale 通过，每分块比 FP32 回退慢 3 倍；
`STRATA_PROMPT_ATTN_XMX=1`）。prompt 时间现在花在哪（2,184 / 8,000 token）：反量化 28% /
27%，GEMM 22% / 20%，每层 host 分组 9.5% / 9%，attention 7% / 14%，embeddings + PLE 行
13% / 5%。

<a id="xmx-onemkls-fp16-gemms"></a>

一个值得知道的陷阱：带原生包时 `--prefill-until N` 不会把剩下的 prompt 走 token 循环（原生包
跳过它），所以 N 之后的 token 被丢弃、模型自由发挥。对比路径之间要对比输出 token，绝不能只
对比计时。

<a id="a-trap-worth-knowing"></a>

**一个 80,000-token prompt（2026-09-30）。** 在这个规模上时间的去向完全改变：

<a id="an-80000-token-prompt-2026-09-30"></a>

- **VRAM 规划。** `--expert-cache auto` 在 KV 状态（81,920 个 cell 的 INT8 约 1 GB）和
  prompt 分块缓冲区（`--prefill 8192` 约 2 GB）存在**之前**就把卡填到 `--vram-reserve-mib`
  为止。按 1,536 MiB 默认值，B70 的 32.6 GB 中 30.95 GB 在用，驱动开始迁移缓冲区，运行永远
  没结束。`--max-context 81920 --kv int8 --vram-reserve-mib 3072 --prefill 4096` 峰值
  28.8 GB。（80k 个 id 也超过 Linux 单参数 128 KB 限制：用 `--tokens-file`。）
- **流式 expert。** 预留花掉缓存槽（3 GB 时 12,288 个里 10,348 个），prompt 路径为每个分块
  从 GGUF 流式缺失的 expert。这条路上躺着两个移植 bug，已修复：流式规划跨约 1,900 次读取
  保留了 `GgufExpertSource::blob()` 的指针（blob 在拷贝前就被覆写：最初测得的 75 秒 /
  1,065 tok/s 部分是用错误的 expert 算的），以及 stream-all 遍历在这块卡的首个大分块上挂起
  （拷贝引擎在 barrier 上停住）。现在 stager 线程自己读 blob，移植用逐层仅路由遍历
  （upstream `STRATA_PREFILL_RING=8`，移植的默认；`STRATA_PREFILL_STREAM_ALL=1` 为调试恢复
  另一种）。
- **正确结果：80,000 token 花 101 秒 = 790 tok/s。** GPU 时间：expert 下投影 GEMM 26.0 秒，
  attention 15.4 秒，反量化 12.7 秒，QSA 块选择 9.3 秒（8k 时 0.2 秒：每个查询扫描上下文的
  每个块），gate/up GEMM 8.9 秒，gather 6.1 秒，每层分组同步 5.2 秒，GDN 递归 4.8 秒。PLE
  行在这个规模免费（行缓存命中 97.6%）。
- **这种 prompt 之后的 decode，以及修复。** 不给 prompt 路径借缓存槽（`--no-prefill-borrow`，
  移植从一开始就用它）时，预留永久逐出约 1,900 个 expert，prompt 之后 decode 只有 2 tok/s。
  借出给 prompt 路径约 940 个槽并在之后约一秒回补：`--vram-reserve-mib 2048 --prefill 4096`
  下 80,000 token 花 75 秒 = **1,062 tok/s，其后 decode 35-40 tok/s**。短 prompt 代价约 1 秒
  （2,184 token：610 对 792 tok/s），所以移植默认只在 32K 上下文以上借
  （`--prefill-borrow` / `--no-prefill-borrow` 明确决定）。

<a id="-the-vram-plan"></a>

**长上下文（2026-10-01，引擎 0.1.32-sycl，setup 的参数：`--kv-resident 32768
--vram-reserve-mib 2048 --prefill 4096`，prompt 路径借缓存槽——32K 以上的默认）。** 同一
文本重复到目标长度，然后 256 个贪心 token：

<a id="long-contexts-2026-10-01"></a>

| 上下文 | KV | prompt | 其后 decode | VRAM 峰值 | host 镜像里的 expert |
|---|---|---|---|---|---|
| 128K | int8 | 888 tok/s（144 秒） | **66.6 tok/s** | 29.9 GB | 无 |
| 128K | q4_0 | 848 tok/s（151 秒） | 62.8 tok/s | 29.7 GB | 无 |
| 128K | k8v4，KV 在 VRAM | **1,006 tok/s**（127 秒） | **66.8 tok/s** | 30.7 GB | 103（0.2 GB） |
| 256K | int8 | 757 tok/s（338 秒） | **55.9 tok/s** | 30.2 GB | 无 |
| 256K | q4_0 | 737 tok/s（347 秒） | 51.7 tok/s | 30.0 GB | 无 |
| 256K | k8v4，KV 在 VRAM | 823 tok/s（311 秒） | 52.2 tok/s | 30.9 GB | 865（1.7 GB） |

借让每个 expert 留在 VRAM（prompt 缓冲区住在借来的缓存槽里，prompt 后回补）。不借时
（`--no-prefill-borrow`）缓冲区占自己的 VRAM，约 1,000-2,300 个 expert 移进锁页 host 镜像：
prompt 快 2-5%，其后 decode 慢 10-23%（0.1.31 不借：128K int8 934 / 59.5，q4_0 887 / 53.4，
k8v4 1,026 / 54.4；256K int8 784 / 48.7，q4_0 763 / 43.2，k8v4 838 / 42.3 tok/s）。

<a id="borrowing-keeps-every-expert-in-vram"></a>

KV 流式（`--kv-resident`）把整个 KV 放在锁页 host 内存、只有被注意的窗口在 VRAM，所以 KV 不
挤出任何 expert；setup.py 从 64K 起打开它并保持 INT8（这里每个尺寸上是更快的一个）。不带它
时 KV 挤出 expert，128K-256K 下 prompt 之后 decode 跌到 4-9 tok/s（2026-09-30）。
`--kv k8v4` 还不支持流式：setup 把它的 KV 留在 VRAM，这会挤出更多 expert；那些经 PCIe 从锁页
host 镜像来（计划条目 2），所以 256K 时 decode 守住了 42 tok/s（镜像之前是 3.8）。k8v4 读
prompt 最快（没有流式拷贝），并走 FP32 attention 回退（它专用的 prompt 内核是下面那个 XMX）。

<a id="kv-streaming--kv-resident-keeps"></a>

**XMX prompt attention v2。** 64-cell 分块、向量打包的 K^T 和 V、hi+lo 的 Q 在每个 scale 组
一个累加器、每分块一次累加器更新：比 v1 快 1.4-1.5 倍且所有模式正确
（`qsa_prompt_attn_parity`，且 `kv_hybrid_parity` 配上它完整通过），但仍比 FP32 回退慢约 2
倍（每分块 13.4 对 5.5 毫秒，INT8，32K 上下文）。这个 attention 受 gather 限制：每个查询位置
选自己的约 2,000 个 cell，所以 K/V 取数占主导、16 行矩阵里只有 12 行是真头。需显式开启：
`STRATA_PROMPT_ATTN_XMX=1`（64-cell 分块，120 KB 本地内存）或 `=32`。

<a id="xmx-prompt-attention-v2"></a>

**XMX v2 进入完整矩阵，以及分组内核为何排不到前面（2026-09-30）。** 带
`STRATA_PROMPT_ATTN_XMX=1`（阶段计时开着，约 8-10% 开销），长 prompt 比 FP32 attention 慢
23-33%：128K int8 666 对 960 tok/s，k8v4 596 对 983；256K k8v4 506 对 752，int8 536 对 718
（q4_0 不用 XMX 内核）。显然的修法——把相邻位置的 cell 并集一次 gather——取决于它们选择的
重叠程度。在 80K prompt 的最后一个分块上实测（第一个 QSA 层，`STRATA_DUMP_SEL=<file>`）：
连续 8 个位置的并集是单个位置 2,051 个 cell 的 3.3 倍（16 个：5.1 倍），且一次选择里只有 12%
被 8 者共享。分组会把 K/V gather 砍到约 40%，但算术乘以 3-5 倍：最好也只是 80K prompt 里的
5-8 秒。没有做。

<a id="xmx-v2-in-the-full-matrix"></a>

**服务这个移植版（2026-09-30）。** `serve/server.py --engine strata` 通过
`sycl/serve/strata-sycl.sh` 原样运行 SYCL 引擎，后者是一个 `exe`，在 oneAPI 运行时镜像里接上
serve 管道启动二进制（配置 `args` 里的路径是容器的，数据根挂在 `/work`）。一份配置：

<a id="serving-the-port-2026-09-30"></a>

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}, ...}
```

<a id="serveserverpy--engine-strata-runs"></a>

带这些采样默认值经 OpenAI API 实测：首轮 decode 36 tok/s、追问 33（复用对话缓存的 prompt），
对照同卡 llama.cpp 的 24-26。decode 工作之后（2026-10-01 构建）：约 190-token 回答含 prompt
48-72 tok/s（启动后第一个请求是慢的那个）；2026-09-30 构建上 5,000-18,000-token 回答
55-59 tok/s。预留很重要：带 `--stream-experts` 时 expert 没有 host 副本，所以任何没进 VRAM
的 expert 在被路由时都要从 SSD 读并用 CPU 计算。1,536 MiB 时缓存上线差 128 个 expert，decode
跌到 5-10 tok/s；1,024 MiB 装下全部 12,288 个还剩 2 GB VRAM。

<a id="measured-through-the-openai-api"></a>

**Logprobs（2026-10-02）。** `/v1/chat/completions` 接受 OpenAI 的 `"logprobs": true` 和
`"top_logprobs": 0..20`，流式与否均可，经由 `sycl/serve/server_intel.py`；回复带
`choices[0].logprobs.content[]`（`token`、`logprob`、`bytes`、`top_logprobs`），开着 thinking
时 thinking token 在 `reasoning_content` 下。引擎（仅移植版）的 GEN 行加上 `logprobs=K`，在
每个 `T` 行后写一行 `LP logprob id:logprob ...`，取自 verify 窗口的 head logits——token 取自
哪个分布，在温度、惩罚和采样之前。Upstream 的服务器忽略这些行，没被要求的引擎一行都不写。被
要求时每个 token 花一次 1 MB 的 logits 行拷贝。实测（IQ2_XS，"capital of France"）："Paris"
于 -0.0019（p 0.998），同样的回答，不开销可测时间。还不在 `/v1/completions` 或 Anthropic
端点上。

<a id="logprobs-2026-10-02"></a>

**Intel 上的 Monitor 标签页。** `sycl/serve/server_intel.py` 是 `serve/server.py` 加两处运行
时添加：

<a id="the-monitor-tab-on-intel"></a>

- **GPU 读数。** NVML 没有卡时，它把 `sycl/serve/xe_telemetry.py` 插进 `serve/telemetry.py`
  的 `gpu_reader`。它经 sysfs 读 Arc（温度、功耗与其上限、卡训练到的 PCIe 链路）。负载与
  VRAM 来自 `/run/gpustat.json`：xe 按客户端在 `/proc/*/fdinfo` 里报告它们，只有 root 能读，
  所以一个 root 小采样器来写（`sycl/tools/gpustat.py`；按它头部说明连同 `gpustat.service`
  安装）。
- **一个 Model 菜单。** 运行配置可以指定一个 `model_switcher`（接受 `{"mode": m}` 的 RPC），
  用于在一块卡上换模型的主机。网页应用头部于是得到一个 Model 菜单
  （`sycl/serve/web/switcher.js`，注入页面）。

<a id="-gpu-readings"></a>

**速度取决于文本。** 推测解码的 decode 跟随草稿层猜对频率。服务中的引擎在 32K 上下文：测试
prompt（续写 Fibonacci 函数，草稿接受率 77-85%）45-51 tok/s，散文聊天回答 30-39 tok/s
（55-72%）；采样与重复惩罚没有可测开销。引擎过去在 serve 模式退出时中止（运行时开始拆除之后
析构函数里的一次队列等待）；现在请求完成就直接退出。

<a id="speed-depends-on-the-text"></a>

**计划（2026-09-30），按顺序。** 按这块卡上的收益排名；前三项先做了。

<a id="planned-2026-09-30-in-order"></a>

1. XMX 整数模式的 expert 点积：一个 decode 窗口（最多 6 token）装得下一个 INT8 DPAS（1-8
   行），i-quant 网格解码成小整数，激活本来就是 INT8。现状：标量 dp4a，受 ALU 限制（77%）。
   **搁置**：三个 `joint_matrix` 版本（需显式开启 `STRATA_EXPERT_XMX=1`，不要启用）比 dp4a 慢
   1.4 倍和 2-3 倍，第三个（用 `joint_matrix_apply` 就地填 B）挂死了 GPU。1-6 行时网格解码和
   打包 B 布局的开销超过 DPAS 省下的。
2. VRAM 缺失的 expert 经 PCIe 从锁页 host 内存读，而不是 SSD（256K decode 崩塌）。**完成**：
   启动时每个没有 VRAM 槽的 expert 读进一个锁页 host 镜像（`STRATA_MIRROR_MIB`，默认可用 RAM
   减 4 GiB），设备构建的 verify 规划直接指向它。之前那个规划丢弃非常驻 expert，所以有缺失时
   输出是错的。强制测试（`--expert-cache 8000`，1,879 个 expert / 3.6 GiB 出 VRAM）：decode
   2.6 -> 40.9 tok/s（满缓存 43.3），prompt 451 -> 640 tok/s（776），输出 token 与满缓存运行
   一致。约 3.6 GiB 的 VRAM 可以换给上下文，代价是约 6% 的 decode。
3. 64K 起的 KV 流式（`--kv-resident`）：只有 KV 被注意的窗口在 VRAM。**完成**：256K decode
   31 tok/s（从 4-5）。
4. XMX 上的 QSA 块选择：每个查询对所有池化块，一个随上下文增长的稠密乘积。
5. 为 Xe2 原生 16 宽子组重新调优热 decode 内核（每线程双倍的寄存器）。**试过，无增益**：
   多列 mmvq 和宽 Q6_K 内核的 SIMD16 构建（`STRATA_MMVQ_SG`：16 全开，1 仅 IQ4_XS，2 仅短
   输出；默认 32）。单独测（`mmvq_sg_bench`）SIMD16 在 IQ4_XS 上快至 1.45 倍、640 行
   shared-expert 投影上 1.3 倍，大 K-quant 上慢 10-25%；引擎内没有任何设置能赢过 SIMD32
   超出运行间噪声（SIMD32 本身落在 39 或 45 tok/s）。输出 token 一致。同一基准显示真正的
   空间：dense decode 内核在 608 GB/s 的卡上流 100-280 GB/s（Q6_K，最常见的类型，2-4 列时
   约 146 GB/s）。
   **发现并修复：未对齐加载。** 一个 Q6_K 块是 210 字节，所以每块的 `ql`/`qh` 行只有 2 字节
   对齐，B70 会把未对齐的 16 字节加载拆碎。同样权重按 224 字节跨步重排后快 2.3-4.7 倍
   （`q6k_align_bench`）。不改布局，宽 Q6_K 内核现在做两次对齐 16 字节加载加一次移位
   （`load16_a2`；两次都留在块内），并接管所有 Q6_K 形状和窗口宽度（此前限于 n_out >= 4096、
   1-4 列）：2560->10240 在 1/2/4/6 列 160/145/141/101 -> 493/432/331/273 GB/s，248K 行的
   head 在 1 列时 150 -> 407 GB/s。输出 token 一致。**Coder decode 44.9 -> 54 tok/s**
   （2k prompt，贪心）。`STRATA_MMVQ_A2=0` 恢复旧路径。仍未对齐：IQ4_XS（136 B，8 对齐，
   约 100 GB/s）、IQ4_NL（18 B）、Q8_0（34 B）；Q4_K/Q5_K 对齐但多内核里用 4 字节加载
   （230-280 GB/s）。

<a id="-expert-dot-products-on-xmx"></a>

**Decode 第二轮（2026-09-30），源自 256 个 decode token 的 unitrace。** 每步单独测量，
19-token prompt，256 token，各两次运行，输出 token 全程一致：

<a id="decode-round-2-2026-09-30"></a>

| | decode |
|---|---|
| Q6_K 对齐修复之后 | 53.9 tok/s |
| + `resident_plan` 分组并行（此前只有 thread 0 做，每层 73 微秒，一轮约 3.5 毫秒） | 57.1 |
| + GR 下投影内核直接读激活（每组 60 KB 的 SLM tile 压制了占用率） | 58.7 |
| + Q4_K、Q5_K、IQ4_XS 的宽 16 字节加载内核（码本在寄存器） | 62.6 |
| + Q8_0（MTP dense、第 0 层 shexp 下投影）和 IQ4_NL（shexp 下投影）的宽内核；Q6_K 的激活加载对齐 | 65.6 |
| + GR 下投影按列切片（每组只暂存 xn 的一个 128 列切片，每行 4 lane）+ 固定顺序归约 | 70.6 |
| + IQ4_NL expert 点积（48 层中 39 层的下投影）：码本在寄存器，权重按对齐 4 字节加载，整数逐位相同（2026-10-01） | 76.1 |
| + GR norm 按 (token, stream)、逐位相同；IQ3_XXS/IQ3_S/IQ2_S gate/up 点积里对齐的权重加载 | **77.9** |

<a id="-decode"></a>

最后一步改变输出：146 个 token 一致，之后逗号后的一个接近平局走向另一边（新内核求和顺序
不同）。Q8_0 内核 4-6 倍（2560 x 10240、2 列时 189 -> 31 微秒），IQ4_NL 640 -> 2560 从
28 -> 6.6 微秒。对齐加载助手只在地址未对齐时才加载第二块，所以从不在没有所需字节时读 16
字节块、也不会在分配结尾跨页。`STRATA_MMVQ_WIDE_32=0` 恢复旧的 Q8_0/IQ4_NL 内核。切片 GR
下投影内核（6 token 时 GR 读取 108.6 -> 76.5 微秒，和与 ~1e-7 相等）把那个接近平局翻回来：
它的输出就是原来的。`STRATA_GR_DOWN_SLICED=0` 恢复直接内核。

<a id="the-last-step-changes-the-output"></a>

内核层面：IQ4_XS 在 2 列时 107 -> 323 GB/s（6 列时 61 -> 212），Q4_K/Q5_K 1.3-1.6 倍；GR 读取
6 token 时 135 -> 110 微秒（逐位相等）。回老路径的开关：`STRATA_PLAN_PARALLEL=0`、
`STRATA_GR_DOWN_DIRECT=0`、`STRATA_MMVQ_WIDE_K=0`。（`iq_parity` 报告 10 个 "missing fixture"
失败，改动前后都有：oracle fixture 不在移植树里。）

<a id="kernel-level-iq4_xs-107--323-gbs"></a>

**两种速度的运行，解释（2026-09-30）。** 完全相同的贪心运行 decode 要么约 45 要么约
39 tok/s。PLE 读取器的逐 gather 跟踪（`STRATA_PLE_TRACE=1`）显示慢的运行在第一个 decode 轮、
窗口图捕获之后，付一次 226 毫秒的 PLE 读停顿；其余每次读取和轮都与快的运行一致。prompt 时间
加 decode 时间两种模式下相同（4.95-5.17 秒）：停顿要么落在 prompt 的 PLE 等待、要么落在第一
decode 轮，所以它是每进程一次的成本，不是损失的吞吐。已排除：NVMe APST、I/O 调度器（`none`）、
CPU 饥饿（decode 期间 93% 空闲）、I/O 线程数。比较运行要用首 token 时间 + decode。

<a id="two-speed-runs-explained-2026-09-30"></a>

6. INT8 prompt GEMM：expert 反量化成 INT8，oneMKL/oneDNN INT8 上 XMX（反量化字节减半、速率
   翻倍）。
7. 每个 decode 轮更少的图节点（约 2,500 个、各约 5 微秒）：norm+rope、scores+top-k、
   gate+quantize 融合。
8. 更宽的推测（每个 verify 窗口两条草稿分支）：内核受延迟限制，所以几乎免费。
9. 一个比 VRAM 大的模型：原版 Qwen3.8-Flash-Next IQ2_XS（35.5 GB expert；同样的路适合 Q2_0
   和 Swift 1.5）在 23 GB RAM 的 B70 上。约 24 GB expert 在 VRAM，其余（约 10-12 GB）在设备
   规划经 PCIe 读取的锁页 host 镜像（条目 2）。移植有 IQ2_XS/IQ3_XXS/Q2_0 的 expert 内核。
   待解：10+ GB 的锁页镜像能否与其他东西并存于 23 GB，以及该份额 expert 下 Gen3 x8 链路上的
   decode（镜像 3.6 GB 实测 40.9 tok/s；预期更低）。需要下载 68 GB。
   **完成（2026-10-01）**：shard 1 下载（39.2 GB；shard 2 与 Coder 的逐字节相同，所以用硬
   链接），原生包（`tools/iq_pack.py`，6 秒），原模型的 expert profile，同一个 MTP 草稿层。
   24,576 个 expert 中 18,329 个在 VRAM（24.6 GiB），6,247 个在锁页 host 镜像（8.4 GiB，填充
   9 秒），host RAM 始终至少有 12 GB 空闲。**Decode 50.8 tok/s**（19-token prompt）、2,184
   token 之后 **60.6**，prompt 549 tok/s；两者都有连贯正确的回答。Q2_0 和 Swift 1.5（大小
   相近）应表现相同；IQ3_XXS/IQ3_S（43-50 GB expert）需要镜像 19-26 GB，超过 23 GB RAM 的
   允许。

<a id="int8-prompt-gemms"></a>

**跟上 upstream。** 把 upstream `main` 合进 `b70` 后，凡是 upstream 动过 `sycl/` 所镜像的
文件，`sycl/` 里的副本就落后。刷新靠重新迁移，不是手工（为 0.1.25-0.1.27 做过，2026-09-30）：

<a id="keeping-up-with-upstream"></a>

1. 把 upstream 合进 `b70`。任何共享文件都不该冲突：Intel 的代码全在 `sycl/`。然后检查
   `sycl/setup_intel.py --check` 和 `sycl/serve/server_intel.py --engine mock` 对着新的
   setup.py 和 server.py 仍能运行。
2. 用 `migrate.sh` 迁移*旧*的 upstream 树（合并前提交的 `git archive`）到 `sycl-base`，迁移
   合并后的树到 `sycl-new`；开发镜像里各约 3 分钟。
3. 对两边跑 `sycl/tools/normalize.sh <dir> <commit>`：dpct 的文件名、未变的文件、消息序号，
   然后 `fixups.py`。现在对同一份源码跑两次 dpct，差别只在它生成的内核名哈希。
4. 对两边都有的每个文件：`git merge-file sycl/F sycl-base/F sycl-new/F`。upstream 没改的副本
   不动。dpct 从未生成的副本（`verify.cpp`、`mtp.cpp`、`remote_experts.cpp` 不直接 include
   任何 CUDA 头）手工带上 upstream 的 diff；新的等期性测试拷进来并登记在
   `sycl/CMakeLists.txt`。
5. 某个 fixup 的模式被 upstream 改了会以编译错误现身（PTX 门在 upstream 的 `__HIPCC__` 保护
   下变成了 `#elif`）；扩展该 fixup、重跑、重建。

<a id="-merge-upstream-into-b70"></a>

**0.1.31 合并（2026-10-01）。** 按上面步骤合并 upstream 0.1.29 -> 0.1.31（132 个提交）：45 个
迁移文件真实改变（27 个只是 dpct 内核名哈希），解决 38 个冲突，verify.cpp/mtp.cpp/
conversation_state.cpp 手工移植。符号审计（移植版特性标识符合并前后计数）现在是流程的一部分：
它抓到一个丢掉的镜像钩子和四个丢了自旋上限的门环等待（upstream 给循环改了名；`fixups.py`
两种拼写都覆盖）。除冲突外这次合并还需要：

<a id="the-0131-merge-2026-10-01"></a>

- **Expert 内核。** Upstream 把 i-quant expert 交给新的多内核（每 warp 一行、每组 8 行）；
  移植的网格按自己的内核定大小（每行 8 lane、每组 32 行），所以每个 expert 只有四分之一行被
  写、输出是 end-of-text token。移植的内核保持默认（它们也服务 upstream 新的
  Q4_K/Q5_K/Q5_1/Q8_0 expert）；`STRATA_EXPERT_SPLIT=1` 跑 upstream 的、用它们的网格。
- **Prompt 槽借出挂起**，在任何 prompt 的首个分块（40K、128K；GPU 忙）——0.1.32 合并后修复，
  见下方"Prompt 槽借出：那次挂起"。
- **Upstream 的新测试发现了首次迁移里的三处 dpct 误译。** dpct 把 `__fadd_rn(a, b)` 写成不带
  括号的 `a + b`，于是 `__fadd_rn(sum, c ? x : y)` 变成 `sum + c ? x : y`（被替换而非相加）：
  在原生 QSA 索引器的每 token 池化键里（decode 路径：生成期间完成的块拿到的池化键是错的；
  prompt 块来自批量内核、是对的），在原生 QSA 分数（一条需显式开启的路径）里，以及在 s2 激活
  舍入里（所有值为 0；Q2_0 的 s2 包，这里没用）。已修复，`fixups.py` 现在会加括号。
  `qsa_parity` 的批量对串行测试通过；长 prompt 之后的 decode 输出变了。
- **Upstream 新内核 vs 移植的，在 B70 上**（decode tok/s，Coder 19 / 2,184-token prompt，
  IQ2_XS 19）：移植默认 78.1 / 75.7 / 58.6；`STRATA_EXPERT_SPLIT=1` 65.6-69.5 / 67.2 / 49.1
  （-11 到 -16%）；`STRATA_GR_V3=1` 73.2 / 72.4（-4 到 -6%）。两者保持需显式开启。
- **移植 expert 内核的每行 lane 数**（`STRATA_GU_LANES` / `STRATA_DOWN_LANES`，运行时
  4/8/16/32）：gate/up 8 或 16 × down 4 或 8，每个 verify 轮都解在 37.7-38.6 毫秒（各两次
  运行，Coder，两种 prompt）；它们之间的 tok/s 差异是草稿接受（每个切分舍入不同）。默认
  （8 / 8）不变。
- **合并后的速度：** Coder 78.2 / 75.7 tok/s（与之前相同），IQ2_XS 58.6 / 64.2（从
  50.8 / 60.6）。
- **仍失败：**（`iq_multi_parity` 的 IQ2_XS：后来修复，见"两个模型特定的 bug"），
  `s2_expert_grouped_parity`（s2 路径），`kv_hybrid_parity` 的最后一步（需要未移植的
  tensor-core prompt 内核）。`ple_parity` 和 `native_expert_parity` 需要模型文件。

<a id="-the-expert-kernels"></a>

**0.1.32 合并（2026-10-01）。** 按同样步骤合并 upstream 0.1.31 -> 0.1.32（89 个提交）；仅哈希
差异现在在 `git merge-file` 之前被规范化（保留移植的内核名哈希），留下 10 个文件有真实冲突。
它需要：

<a id="the-0132-merge-2026-10-01"></a>

- **Upstream 的异步 commit**（`set_commit_async` / `wait_commit`）对应到移植自己的延迟 commit
  （`commit(n, err, false)` + `commit_finish`）；`wait_commit` 即 `commit_finish`。
- **Upstream 新的超连接读取变体**（split / staged，启动时按逐位自检选卡）移植不用，移植保持
  自己的切片下投影 / 拆分 norm 读取；自检在 B70 上段错误，所以在 SYCL 上只在
  `STRATA_HC_CHECK=1`（待办）时才跑。移植的 split-norm 内核被改名
  （`gr_norm_split_port_kernel`）：upstream 现在有一个同名的。
- 哈希规范化后两个内核名撞车（已改名），而 fused_gr 的每 block 共享内存查询成了一个 fixup。
- **结果：** 所有输出与 0.1.31 相同（Coder 19 / 2,184-token prompt 和 IQ2_XS，256 个贪心
  token），速度相同（Coder 77.7-78.1 / 75.7 tok/s，IQ2_XS 58.5），40K prompt 1,201 tok/s、
  其后 decode 65.1 tok/s。`kv_hybrid_parity` 和 `qsa_prompt_attn_parity` 通过（测试现在打开
  它们所检查的 XMX prompt attention）；`iq_multi_parity`（IQ2_XS）和
  `s2_expert_grouped_parity` 与之前相同。

<a id="-upstreams-async-commit"></a>

**0.1.33 合并（2026-10-01）。** Upstream 0.1.32 -> 0.1.33（17 个提交）：没有共享文件冲突
（Intel 代码全在 `sycl/`）；六个引擎文件 upstream 改了、按同样的重新步骤合并（只那六个：
其余 dpct 产出不同的文件不动）。两个冲突：`--resident-cpu-experts`（upstream 新的
`resident_cpu_explicit` 挨着移植的 `--stream-experts`）和 prompt attention 的算力检查（移植
保持自己的 XMX 调度）。`sycl/setup_intel.py` 和 `sycl/serve/server_intel.py` 对着新的
setup.py 和 server.py 未改就跑了。所有输出一致（Coder、IQ2_XS、带借出的 40K prompt）；
`gr_parity` 和 `qsa_prompt_attn_parity` 通过。

<a id="the-0133-merge-2026-10-01"></a>

**Prompt 槽借出：那次挂起（2026-10-01 修复）。** 从 0.1.31 起，借缓存槽的 prompt 在首个满
分块停住，GPU 100%、host 在计算队列上等待。从来不是借出的 bug：借出只是 Coder 在 prompt
期间流式 expert 的唯一方式（否则每个 expert 都常驻），而流式路径的 stager 死锁了。它的线程
把 expert 读进 16 个锁页缓冲区的环，为复用一个要等拷贝队列上最后一次读取它的 DMA 的事件
（CUDA 形式，`cudaEventSynchronize`）。在 Level Zero v2 适配器下，那个事件一旦被某个 host
线程等过，就不再释放同样列出它的另一个队列的 barrier，GPU 就永远等——只在某层流式超过 16 个
expert 时（环回绕）。v1 适配器（`SYCL_UR_USE_LEVEL_ZERO_V2=0`）和 256 缓冲区的环都能跑；每段
之后同步也能跑（`STRATA_PREFILL_SYNC=1`，作为调试开关保留）。修复：拷贝队列每次 DMA 后把
序列号写进锁页 host 内存，stager 轮询它——没有 host 线程等队列事件。输出与不借的运行逐位
一致；借出重新成为 32K 以上的默认。

<a id="prompt-slot-borrowing-the-hang"></a>

**用 Swift 1.5 发现的两个模型特定 bug（2026-10-01 修复）。** UkisAI 的 Swift 1.5 IQ2_XS
（setup 的 `swift` 家族，同样的 GSQ-RCO 布局）永远解 token 0：第 13 层起 logits 为 NaN。

<a id="two-model-specific-bugs-found-with-swift-15"></a>

- **`--stream-experts` 读了错误的 shard。** 自 upstream 0.1.31 起，`ExpertLayout::gguf_file`
  按层**和**角色（`3 * layer + role`；shard 边界可以落在层内部）。移植的 `GgufExpertSource`
  仍按层索引它，于是 shard 2 里的层从 shard 1 按 shard 2 的偏移读取：乱码的 IQ1_M scale，
  fp16 反量化之后是无穷大。原版模型把所有 expert 放在 shard 1、从未察觉；Swift 的第 13-47 层
  在 shard 2。`STRATA_DBG_NAN=1` 现在还报告 expert 的 fp16 GEMM 输入（激活、反量化的
  gate/up）。
- **IQ2_XS 的乘积是乱码**（长期失败的 `iq_multi_parity` IQ2_XS 用例）：ggml 的
  `vec_dot_iq2_xs` 经由指向 `int2` 的 `uint16_t*` 读它的四个 16 位码，SYCL 设备编译器不认这种
  类型双关；直接读码的反量化器是精确的。现在用移位提取码（点积和 expert 内核的 `Split<17>`）；
  测试还对照 CPU 上由 ggml 表解码的结果检查并通过。这里的模型都不用 IQ2_XS 张量，Swift 也不
  例外——这不是 Swift 的 bug，只是路上发现的。

<a id="--stream-experts-read-the-wrong-shard"></a>

两个都修后的 Swift 1.5 IQ2_XS：回答正确，短问题 51-118 词的 thinking，原始 decode 48 tok/s。

<a id="swift-15-iq2_xs-after-both"></a>

**加载时间（2026-10-02）。** 启动模型主要花在 expert 缓存从 GGUF 填充（`--stream-experts`）：
Coder 的 23.4 GiB expert 要 64-76 秒，IQ2_XS 的 24.9 GiB 要 91 秒，因为填充逐 expert 读三个
切片、拷贝 blob、等拷贝完成才读下一个。现在是一条流水线：槽位先按 profile 顺序准入（同样的
放置），读取按文件顺序由最多 8 个线程进行、进入 64 个一组的锁页批次，一批的拷贝与下一批的
读取重叠。冷页缓存：76.2 秒 -> 18.8 秒（0.33 -> 1.34 GB/s）；一次完整 Coder 启动从引擎启动
到首 token 82 秒 -> 26 秒，IQ2_XS 120 秒 -> 41 秒（它 8.2 GB 的 host 镜像约占其余 9 秒）。
输出一致。`STRATA_FILL_SERIAL=1` 是旧的填充。

<a id="load-time-2026-10-02"></a>

**stream-all prompt 遍历回来了（2026-10-03）。** 它逐层把 VRAM 缓存没装的每个 expert 都拷到
计算前面，而不只是路由到的那些。它在 B70 上挂起，直到 2026-10-02 的 stager 和 PLE 修复，之后
一直能跑。现在当 VRAM 装得下超过 90% 的（层，expert）对时默认运行；超过那个比例它会拷贝路由量
的几倍（IQ2_XS 把四分之一留在 RAM 镜像：带它 2,184-token prompt 从约 560 掉到 254 tok/s）。
`STRATA_PREFILL_STREAM_ALL=1` / `=0` 强制开关。Coder，两种输出一致：

<a id="the-stream-all-prompt-walk-is-back-2026-10-03"></a>

| prompt | 仅路由 | stream-all |
|---|---|---|
| 40K int8 | 1,106 tok/s | 1,160 |
| 128K int8，KV 流式 | 897 | 920 |
| 128K k8v4 | 1,006 | 1,041 |
| 256K int8，KV 流式 | 760 | 780 |

同时测得：2,184-token prompt 在 0.1.35+PR 和 0.1.38 构建上都读 784-799 tok/s（各三次运行），
所以 0.1.35 上见过一次的 825 是会话散布，不是回退。40K prompt 路径（Coder，GPU 时间线）：
expert 反量化 15.7%、expert GEMM 23.4%、host 分组 6.5%、gather 5.7%、combine 2.4%、prompt
attention 19.2%、QSA select 6.8%、DeltaNet 递归 5.8%、超连接读取 4.6%。Upstream 的 sm_90
cluster 内核（贪心采样器、QSA top-k）在这里几乎省不了什么：采样器每轮读约 6 MB，对一个
13 毫秒的 token。Q4_0 tensor-core prompt attention（#452）不值得移植：移植的 XMX prompt
attention 对它给每种 KV 格式用的普通内核慢 2-3 倍。

<a id="measured-alongside"></a>

**0.1.38 合并（2026-10-03）。** Upstream 0.1.35 -> 0.1.38（83 个提交）的合并包含了移植提前
拿下的七个 PR，所以它们在 `sycl/include` 里的头文件分叉没了；三方合并的基线是移植自己的
"main + PR" 迁移。随它进入移植的新东西：不带死 FP32 存储的 DeltaNet 输出 norm、两个 prompt
路径修复（fused 布局的缓冲区、流式遍历的常驻查找）、`STRATA_GR_DOWN_MAX4=1`（需显式开启；
B70 上中性：2,184-token prompt 793 对 788 tok/s，decode 76.2 对 76.3）。移植中打桩的：
`--peer-device`（第二块 GPU 作 expert 缓存层：upstream 的代码是 CUDA 调用；`open` 带消息
拒绝）和融合 int8 prompt 内核（#136，属于移植不构建的 MMQ 库）。SYCL 上关着的：sm_90
线程块 cluster 的贪心采样器和 QSA top-k（无 SYCL 对应物；调用方用普通内核）。
`STRATA_GDN_KEYHEAD=1` 保持需显式开启（这里仍慢 8%：727 对 788 tok/s）。
`qsa_prompt_attn_parity` 现在还测 upstream 的 Q4_0 tensor-core 模式（#452），移植没有：那些
用例报告 "refused"（int8 和 fp16 的 XMX 用例通过）；`--kv q4_0` 的 prompt 用更旧的内核，
与之前相同。输出一致（Coder、IQ2_XS、40K）；128K int8 KV 流式 892 / 66.8 tok/s。

<a id="the-0138-merge-2026-10-03"></a>

**0.1.35 与七个开放的 upstream PR（2026-10-02）。** Upstream main 0.1.33 -> 0.1.35 照旧合并，
然后七个开放 PR 先于 upstream 移植进 `sycl/`（main + 全部的一次 dpct 运行，三方合并只进它们
碰到的文件；`sycl/tools/merge_upstream.py`）。它们的头文件改动住在 `sycl/include` 直到
upstream 合并。

<a id="0135-and-seven-open-upstream-prs"></a>

| PR | 内容 | 在 B70 上 |
|---|---|---|
| #385 (sergqwer) | stager：一次生成中缓冲区的第一个任务等此前的 DMA | 竞态修复；输出相同 |
| #463 (constantindjonkam) | decode 在读常驻表前等自适应交换 | 确定性修复 |
| #453 (architectds) | 草稿器的批量 K/V 走 KV 流式环 | 128K int8：prompt 888 -> 895 tok/s，decode 66.6 -> 67.0 |
| #374 (sergqwer) | 首个分块的 PLE 行与第 0 层并行读 | 2,184-token prompt 784 -> 825 tok/s 的一部分 |
| #363 (BlueKingMuch) | PCIe expert 调用：分组跨步、融合 SwiGLU + q8_1 | IQ2_XS decode +1.7%；在移植的 lane 内核上重做 |
| #407 (sergqwer) | `--adapt-tuned`（需显式开启） | 这里中性；保持需显式开启 |
| #413 (BlueKingMuch) | DeltaNet 递归按 key head | 逐位一致但这里 prompt 慢 8%：关（`STRATA_GDN_KEYHEAD=1`） |

<a id="pr-what-on-the-b70"></a>

#374 需要两处移植修复：它的下一分块读取假设每个分块都是满分块长度（移植的首分块是 256 token：
第二个分块的行从错误的位置读），而它在层循环内对 PLE 上传事件的 host 等待，在 Level Zero v2
适配器下让约 4K token 以上的 prompt 死锁（又是 stager 的 bug）：上传现在用一个被轮询的序列号
标记。#413 的门是 NVIDIA SM 数量规则、它的等期性测试是一个扣住 SM 的 NVIDIA 基准（未构建）。
输出与 0.1.33 一致（Coder 19 / 2,184 token，IQ2_XS，40K）。

<a id="374-needed-two-port-fixes"></a>

**尚未移植（2026-10-01）。**

<a id="not-ported-yet-2026-10-01"></a>

- 三个内核带内联 PTX（`mma.sync` tensor-core 矩阵运算、`ldmatrix`、`cp.async`）：
  `qsa_prompt_attn`、`qsa_select` 的块分数、`native_qsa_score`。SYCL 构建采用 CUDA 构建在
  sm_80 以下用的 "older card" 回退。`qsa_prompt_attn` 还有一个 XMX 版本（`joint_matrix`，需
  显式开启 `STRATA_PROMPT_ATTN_XMX=1`）：正确，但比回退慢（见"XMX prompt attention v2"）。
- ggml MMQ prefill 路径（`moe_mmq.cu`）需要 llama.cpp 的 ggml-cuda 源；未构建。
- 跑的是 AOT 设备代码：`AOT=bmg-g31 BUILD_DIR=.../build-sycl-aot`（JIT 构建在第一个窗口要付
  约 47 秒编译）。

<a id="-three-kernels-carry-inline-ptx"></a>

## Arc A750 与其他 Alchemist 卡（`i915`，2026-10-07）

<a id="arc-a750-and-the-other-alchemist-cards"></a>

在 Arc A750（8 GB，`i915`，PCIe 4.0）、64 GB RAM、Ryzen 5 5600X（AVX2，无 AVX-512）的 PC 上
用 Flash-Next IQ3_XXS 实测。`./setup.sh --backend sycl` 为 `i915` 卡写不同的配置（对任何
小于 12 GB 的卡还有 300 MiB VRAM 预留和 `--draft-vocab en`）：

<a id="measured-on-an-arc-a750"></a>

| | Arc Pro B70 (`xe`, 32 GB) | Arc A750 (`i915`, 8 GB) |
|---|---|---|
| experts | `--stream-experts`：从 GGUF 进 VRAM，其余在锁页 RAM 镜像 | 无 `--stream-experts`：全部在 RAM 区（IQ3_XXS 是 39.97 GiB），24,576 个里 406 个在 VRAM，其余由 CPU 计算 |
| `STRATA_VERIFY_NO_HOST` | 1（包装脚本的默认） | 0，写进配置的 `"env"` |
| `--vram-reserve-mib` | 1024 / 2048 | 300 |
| `--ple-io` | `ram`（setup 的机械盘规则） | `ram` |
| FP64 | 原生 | 模拟：`"env"` 里 `IGC_EnableDPEmulation=1`、`OverrideDefaultFP64Settings=1`、`NEOReadDebugKeys=1` |
| 构建 | AOT `bmg-g31` | JIT（没有 `AOT=`） |

- **为何没有 `--stream-experts`。** 镜像是一整块锁页 host 分配，装卡装不下的一切（这里
  40 GB）。A750 上单次超过几 GB 的 `sycl::malloc_host` 会失败（`STRATA_MIRROR_MIB` 3,000 能
  过，8,192 和 24,000 不能，`memlock` 不限和放宽分配限制的变量也一样），引擎随后以
  `24170 experts are neither in VRAM nor mirrored` 拒绝启动。没有该参数时引擎把 GGUF 的
  expert 装进普通 RAM 区，CPU 计算不在 0.67 GiB 缓存里的 expert。这需要 RAM：IQ3_XXS 约
  43 GB，加上 `--ple-io ram` 的 29 GB n-gram 表（表在容器里不锁页，所以可回收；不要给容器
  `--ulimit memlock=-1`，锁页的表和区就装不进 64 GB、机器开始交换）。RAM 少于模型需要时
  setup 会警告。
- **为何 `STRATA_VERIFY_NO_HOST=0`。** 设了它 GPU 自己规划每层、根本不会要 CPU 的 expert：
  窗口挂起（i915 日志说 `Fence expiration time out`），或者池被拒绝时引擎报
  `REFUSED: ... neither in VRAM nor mirrored`。它是 B70 情形的开关——那里卡缺的每个 expert
  都在镜像里。引擎把 `0`（或空）当作"关"，`strata-sycl.sh` 也不会往下传这种值。
- **FP64。** Alchemist 没有 FP64 硬件。采样路径上有一个内核声明 `double`，SYCL 运行时在首次
  启动时拒绝（`'double' is not supported in 'Intel(R) Arc(TM) A750 Graphics' device`，引擎在
  第一个带温度的请求上死掉）。带上上面三个变量驱动会模拟它；代价是采样器的尾部（见 B70 节的
  "FP64."）。
- **启动后的第一个请求过去返回 `!!!!!`（token 0）或让引擎崩溃。** GPU 在每层以一个有界自旋
  等 CPU 的 expert（`strata::kSpinMax`，`sycl/include/strata/sycl_doorbell.hpp`）：永不结束的
  自旋会挂死驱动。上限原本是 20,000 次读取、几十毫秒，在没有东西要等时够用（B70），但对首个
  请求里的 CPU 层不够（冷页，26-token prompt 要 13 秒）。GPU 放弃、带着缺失的 expert 输出继续，
  后面各层的路由是乱码（一个 token 十次选中同一个 expert：`nt=10`），CPU 池越过它的 token
  数组写入、以段错误死掉（退出码 139）；没死时 logits 是 NaN、采样器答案是 token 0。之后的
  请求页是热的、是对的。现在上限是构建选项 `STRATA_SYCL_SPIN_MAX`（CMake；
  `sycl/tools/build.sh` 的 `SPIN_MAX=`）：2,000,000 次读取（几秒），除非是为 `bmg` 卡的 AOT
  构建，那保持 20,000。改动前 16 次引擎运行（各三个请求、26-token prompt）里 11 次死掉或第一
  请求给 token 0；用 2,000,000 次读取后 4/4 全部三个请求都对。
- **内核读分配内超过 4 GiB 的内存会读错（已在构建中修复）。** 在 Arc A770（16 GB，`xe`，
  Coder IQ1_M，`--stream-experts`，4.46 GiB expert 缓存）上每个 prompt 都答 `!!!!`（token 0）：
  缓存最后 0.46 GiB 里的 expert 在 prompt 路径反量化成 NaN。那些槽的拷贝回读与 GGUF 相等
  （`STRATA_VERIFY_ALL_SLOTS` 通过），同样字节的新鲜设备拷贝反量化干净；只有就地读那个槽的内核
  拿到错误字节。JIT 构建里 IGC 把内核指针参数编成相对分配基址的 32 位偏移。JIT 构建现在传
  `-ze-opt-greater-than-4GB-buffer-required`（`sycl/CMakeLists.txt`），让内核保持 64 位地址。
  改动后，73 到 542 token 的 prompt 有 0 个非有限值（`STRATA_DBG_NAN=1`），542-token prompt
  回答正确。速度代价未测。AOT（`bmg-*`）构建不变。oneMKL 自己预编译的内核拿不到该选项；GEMM
  为此会把一个操作数暂存到 4 GiB 之外。
- **速度。** decode 约 10 到 15 tok/s（草稿接受率 76% 到 92%），26-token 首 prompt 13 秒、
  之后的短 prompt 0.2 到 1.3 秒；A750 的 PCIe 链路探测为 10.6 GB/s。
- **A750 与 xe 错误计数器。** 引擎退出时在 CPU 池的一个工作线程里段错误（只在 dmesg，服务器
  已经打印 "stopped"）；不是 GPU 事件。

<a id="-why-no--stream-experts"></a>

## 没做的

<a id="not-done"></a>

- 图像，两个 Intel 引擎都是。Strata 的视觉路径用 `strata-vision` 编码成 CUDA 引擎读取的
  embedding；SYCL 移植和 llama.cpp 配置都还没接上它。
- llama.cpp 路径的推测解码（GGUF 里没有 llama.cpp 能用的草稿层）。SYCL 移植有（MTP 草稿层，
  `--mtp`）。

<a id="-images-on-both-intel-engines"></a>

## Arc Pro B70 配装不下的模型：Flash-Next IQ3_S（2026-10-07，分支 f-141-intel）

<a id="arc-pro-b70-with-a-model-that-does-not-fit"></a>

早先的 B70 行都是 expert 全装得进 VRAM 的模型。IQ3_S 有 24,576 个 1.97 MB 的 expert
（46 GB）：32 GB 的卡装 11.5k 个（22 GiB 缓存），其余放在 GPU 经 PCIe 读取的锁页 host 镜像。
在 B70 VM（xe 驱动，内核 7.0，oneAPI 2026.1，PCIe 实测 host 到设备 13.3 GB/s）上测，引擎
0.1.40-sycl。

<a id="the-earlier-b70-rows-are-for-models"></a>

**两件事让它输出乱码，都已修复或在此记录。**

<a id="two-things-made-it-produce-garbage"></a>

- **大分配里的别名页（已在引擎修复）。** xe 驱动返回 22 GiB expert 缓存区时，把其中两个 2
  MiB 页映射到同一块内存：在 +2 MiB 写会出现在 +1022 MiB。那里的缓存槽装着别的 expert 的
  字节，prompt 在第一个路由到它的层（第 4 层，expert 249）出来是 NaN；同样的运行换一个
  `--max-context`、或把 MTP 窗口设成 4 而不是 6，就是干净的。`STRATA_VERIFY_ALL_SLOTS=1`
  （`STRATA_VERIFY_FIND=1`、`STRATA_VERIFY_DUMP=dir`）把每个填满的槽回读并与 GGUF 比较；
  别名就是这样发现的（错误的字节是槽 0 的 blob）。现在每个 32 MiB 及以上的分配都经过
  `strata::malloc_device_guarded`：它每 64 KiB 打标记、第二个内核回读标记，并在增长的间隔垫
  之后重新分配直到区间干净（每次重试有一条警告命名；启动时检查花不到 0.5 秒；
  `STRATA_ARENA_ALIAS_CHECK=0` 跳过）。`sycl/probe/arena_rw.cpp` 和 `alias_scan.cpp` 单独
  复现不了：需要引擎的分配历史。
- **只要有一部分 expert 不在 VRAM，就必须 `STRATA_VERIFY_NO_HOST=1`。**
  `sycl/serve/strata-sycl.sh` 会设；手动运行也要。不带它时，每层的 host/GPU 握手在 xe 上跨
  总线不可见，回答开始几个 token 后 logits 变 NaN、每次运行都不一样。带上它（且镜像覆盖每次
  缺失）运行是确定的：两次运行给出相同的 token。

<a id="-aliased-pages-in-big-allocations"></a>

**数字**（贪心，`--spec 4 --spec-min-p 0.5 --mtp`，INT8 KV，32K 上下文，`--stream-experts
--vram-reserve-mib 1024 --ple-io ram`，11.5k 个 expert 常驻；10 个 prompt 的 `mg_norepeat`
门 10/10 通过，xe 错误计数器没有动）：

<a id="numbers"></a>

| | 之前 | 之后 |
|---|---|---|
| 4,095-token prompt，`--prefill auto`（2,048-token 分块） | 618 tok/s | 730 |
| 4,095-token prompt，`--prefill 4096` | 784 | **980-1,002** |
| 8,169-token prompt：auto / 4096 / 8192 借缓存槽 / 8192 自带缓冲区 | 676 | 953 / 1,037 / 1,117（9.4k 槽） |
| decode，200-token 故事（草稿接受率 66%，每轮 1.98 token） | 31.4 tok/s | 4096-token 分块时 30.5（少 700 个缓存槽） |
| decode，200-token 代码回答（接受率 90%，每轮 3.65 token） | 未测 | 4096-token 分块时 40.8 |

（交错 A/B 对，3-5 次的中位数。"之前"是同一构建用旧的短首分块；decode 两个数字是旧默认与
`--prefill 4096`。）

<a id="-4095-token-prompt--prefill-auto"></a>

**为何 prompt 在这里慢，以及两项改动。** 每个 prompt 分块都经 PCIe 流式缓存没装的 expert，
而一个 4K prompt 几乎触及每层全部 512 个 expert：13.5k 个缺失 expert × 1.97 MB = 26 GB，每个
分块两秒的总线。所以 prompt 用更少更大的分块读得更快，而短首分块（`STRATA_PREFILL_FIRST`，
256 token，为在读 PLE 行时让 GPU 启动而设）要多付一整遍那个流。现在它只在缓存装得下每个
expert 时是 256，否则 0。`setup_intel.py` 给 24 GB 及以上的卡写 `--prefill 4096`（该分块的
缓冲区花约 700 个缓存槽；`--prefill 8192 --prefill-borrow` 改为借缓存槽、把全部槽留给 decode，
代价是每个 prompt 约 1 秒回补）。4K prompt 的 GPU 时间现在花在哪（GPU 时间线，计时开着）：
expert 反量化 20%，两个 expert GEMM 34%，QSA attention 14%，host 分组 6.5%，GDN 9%，超连接
读取 4.6%。

<a id="why-the-prompt-is-slow-here"></a>

**草稿长度**（200 token，5 个交错对的中位数，`--spec-min-p 0.5`）：`--spec 6` 对 `--spec 4`
代码 +6.5%（43.4 对 40.8 tok/s，接受率 84% 对 90%），散文 -1.5%（29.9 对 30.4）。其余的两轮
扫描（`--spec 2/3/5`，min-p 0.3/0.7）在约 1 tok/s 的运行间噪声内；setup 默认（4，0.5）不变。

<a id="draft-length"></a>

**FP64。** Arc Alchemist 模拟它（`IGC_EnableDPEmulation=1`）；A750 上采样器 top_p/温度尾部的
一行在 double 下 451 微秒、float 下 20 微秒（`sycl/probe/fp64_cost.cpp`），采样 decode 跑
7.6 tok/s、贪心的 12。B70 上同样的内核 3.4 对 2.1 微秒，所以 FP64 不是 B70 的问题。用 double
的内核中，原生包路径（setup 构建的包）只跑采样器（16-token 贪心运行创建了 173 个内核，double
的在其中：只有 `sample_tokens`；原生路由、combine、gate 和 norm 内核是 float）。采样器尾部
现在用 `strata::samp_acc_t` 累加、默认 float（`-DSTRATA_SYCL_SAMPLER_FP64=1` 保持 double）：
A750 采样 decode 7.62 -> 8.10 tok/s（5 个交错对的中位数，各次 7.1-9.2），`sampler_parity` 两
卡都 0 失败，B70 门的采样用例选了相同的 token。

<a id="fp64"></a>

采样器的启动器还留着 dpct 的 `aspect::fp64` 检查，它在一个没带模拟键的 A750 上于第一个采样
token 抛 "'double' is not supported"，尽管自累加器改动后内核一直是 float；0.1.41 去掉了该
检查（在 `-DSTRATA_SYCL_SAMPLER_FP64=1` 构建里保留）。在 A750 上实测（IQ3_XXS，`--spec 4
--mtp`，采样，100 token，4 个交错对，同一二进制）：不带 `IGC_EnableDPEmulation` /
`OverrideDefaultFP64Settings` 为 11.0 / 11.6 / 12.4 / 12.4 tok/s，带着为 9.3 / 9.9 / 9.8 /
9.8（中位数 12.0 对 9.8，4/4 对都更快）。setup 仍为 A 系列卡写这些键：除采样器外使用 double
的内核（oracle 路由、norm、PLE、QSA 索引器）没有在不带它们、长上下文或带视觉的情况下被检验，
所以键保持安全默认。

<a id="the-samplers-launchers-also-kept"></a>

**XMX。** oneMKL 的 FP16 GEMM 已经跑在矩阵引擎上；每个 expert 的反量化（一个 gate/up 矩阵
42 微秒）是 GEMM（20 微秒）的两倍，而融合的 dequant+XMX 内核（`xmx_gemm_bench`）在这块卡上
只有那一对的 0.22 倍。这块卡对这个模型剩下的速度在反量化内核（约 200 GB/s，对 600 GB/s 的
卡）、host 分组，以及 QSA prompt attention（14%）；llama.cpp PR 29864 的多列 int8 DPAS decode
内核针对 K-quant 和 Q8_0 权重，不是 IQ-quant expert，没有移植。

<a id="xmx-onemkls-fp16-gemms-already-run-on-the-matrix"></a>

## 实测，2026-10-01，Arc Pro B70，Coder IQ1_M，32K 上下文，INT8 KV：SYCL 移植版（引擎 0.1.31-sycl）

<a id="measured-2026-10-01-arc-pro-b70"></a>

| | |
|---|---|
| decode | 19-token prompt 上 78.2 tok/s，2,184-token prompt 后 75.7（256 个贪心 token，MTP + 后缀草稿） |
| prompt | 2,184 token 上 780 tok/s；128K 时 934 tok/s，256K 时 784 |
| 经 API | 约 190-token 聊天回答含 prompt 48-72 tok/s（启动后第一个请求是慢的那个） |
| VRAM | 全部 12,288 个 expert 常驻，全加载后约剩 1.9 GB（`--vram-reserve-mib 1024`） |
| RAM | expert 没有 host 副本（`--stream-experts`） |

## 实测，2026-09-29，Arc Pro B70，Coder IQ1_M，32K 上下文，q8_0 KV：llama.cpp

<a id="measured-2026-09-29-arc-pro-b70"></a>

| | |
|---|---|
| 加载 | 约 100 秒 |
| VRAM | 32 GB 中 28.4 GB |
| decode | 23-25 tok/s；GPU 在 165 W 下 92% 忙碌，CPU 空闲 |
| prefill | 2,701-token prompt 上 149 tok/s；131k 上下文 104,798-token prompt 上 424 tok/s |
| 质量 | 每个测试的代码都正确；与 NVIDIA 路径在精神上逐 token 相符，未测量 |
