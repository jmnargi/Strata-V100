# Strix Halo（Ryzen AI Max，gfx1151）：在 Linux 上从源码构建
<a id="strix-halo-ryzen-ai-max-gfx1151-build-from-source-on-linux"></a>

Strata 通过 Ryzen AI Max "Strix Halo" APU（Radeon 8060S / 8050S、RDNA3.5、`gfx1151`，CPU 与 GPU 共用一整池 LPDDR5X）运行，
走的是与 [AMD_HIP.md](AMD_HIP.md) 中 Radeon 显卡相同的 HIP 后端。**状态：实验性。**
本文的一切都在同一台机器上构建和实测：维护者的 Strix Halo 主机（Ryzen AI Max+ 395、128 GB、Ubuntu 26.04、kernel 7.0、ROCm 7.14.1）。
Setup（`setup.sh`、`START-HERE.bat`）从 0.1.40 起就能识别它：靠 PCI id（`1002:1586`，8060S、8050S 和 8040S 共用这一个 id），
并按 `gfx1151` 精确匹配：Strix Point（890M / 880M，gfx1150）、Krackan（860M / 840M，gfx1152）或 Phoenix / Hawk Point
（780M / 760M，gfx1103）上的 Radeon 会按其本来型号报出名字，且不受支持。在 Linux 上，setup 自己就会为 gfx1151 编译引擎（下面的步骤
就是手动把同样的事做一遍）。在 Windows 上，现成的 `strata-windows-x64-hip.zip` 从 0.1.40 起带有 gfx1151 代码（尚未在 Windows 的
Strix Halo 上实测过）；不带它的 zip 会被拒绝并给出提示信息，绝不使用。Setup 把 GPU 的内存算作 BIOS 划分区加上共享内存
（即 GTT 池；Windows 上是内存的一半），并在内存约 80 GB 起推荐 UD-IQ4_XS；它不改动任何 BIOS 或内核设置。

引擎在这颗芯片上的不同做法：

- **统一内存。** GPU 的“显存”就是同一份内存的 GTT 池。`device_free_bytes()`（`--expert-cache auto`
  和内存计划采用的数字）统计的是 Linux 能交还的部分（`MemAvailable` 扣除留给系统的 6 GiB，这个 6 可由
  `STRATA_UMA_HEADROOM_GIB` 修改），而不是只看 `MemFree`；在 Linux 上对任何集成 GPU 都逐设备如此（这对 DGX Spark 同样适用）。
- **gfx11 矩阵核心。** prompt 注意力、块打分器、prompt GEMM 和 prompt 专家都有针对
  gfx1100 / 1101 / 1102 / 1150 / 1151 的 WMMA 内核。不在这份名单里的 gfx11 产品（gfx1103、gfx1152）不会启用它们：走可移植路径。
- **hipBLASLt 表** `tools/hip/gfx1151-hipblaslt-100401.txt`（ROCm 7.14.1 的 hipBLASLt 1.x，即 `100401`），
  包含 `--prefill 16384`（T 16383 / 16384）下的 prompt 形状。
  Setup 为 gfx1151 自带的 ROCm（7.14.0a20260608 wheels，hipBLASLt 1.4.0）使用
  `tools/hip/gfx1151-hipblaslt-100400.txt`：同样的形状，针对该库重新校准（见 docs/AMD_HIP.md 的 Tuning table）。
- **专家缓存与 RAM 预算共享同一份 RAM。** 对带 RAM 预算的模型（UD-IQ4_XS、UD-Q4_K_XL），setup 会写入一个
  `--expert-cache` 数值（RAM 减 24 GB 再减预算），而不是 `auto`：`auto` 会在启动时算上操作系统能交还的全部内存，
  在 55 GiB 预算且开着桌面的情况下，它占掉了 OS 需要的空间（#1715，一台 128 GB 机器上的全局 OOM）。
  这个数值写在配置的 `args` 里；用其中的 `--expert-cache N` 即可设置。手动启动引擎时也应传入一个。

## 1. 工具链（无需 root）
<a id="1-the-toolchain-no-root-needed"></a>

来自 AMD 的 TheRock 构建、面向 gfx1151 的 ROCm 7.14.1，解压到你自己的一个文件夹里：

```sh
mkdir -p ~/rocm && cd ~/rocm
curl -LO https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz   # 1.7 GB
sha256sum therock-dist-linux-gfx1151-7.14.1.tar.gz   # c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00
mkdir install && tar -xzf therock-dist-linux-gfx1151-7.14.1.tar.gz -C install
export SDK=$HOME/rocm/install
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
```

还需要 CMake 3.24+、Ninja、git，以及带 libstdc++ 头文件的 C++ 宿主工具链（`sudo apt install build-essential cmake ninja-build git`）。
维护者的机器上没有系统编译器：他们通过 `--gcc-toolchain=<dir>/usr` 让 ROCm 的 clang 指向用户空间里的 gcc 15
（已加入下文的 `CMAKE_HIP_FLAGS`；系统自带编译器时不要加）。

内核自带的 amdgpu 驱动以及对 `/dev/kfd` 的访问权限（你的用户在 `render` 和 `video` 组里）就是 GPU 侧的全部所需。
为了让 GPU 能用上一台 128 GB 机器的大部分内存，维护者的机器以内核参数 `ttm.pages_limit=29360128 ttm.page_pool_size=29360128`
启动（112 GiB 的 GTT；默认约为内存的一半），并带 `amd_iommu=off`；两者都是宿主机设置，引擎运行并不需要它们，
只是要有这些空间。

## 2. 引擎
<a id="2-the-engine"></a>

llama.cpp 的 `ggml` 源码由本仓库钉住版本：CMake 会拉取它需要的那个 revision。离线构建时，把
`STRATA_GGML_DIR` 指向恰好是该 revision 的检出版。

```sh
git clone https://github.com/Niko1221/Strata.git && cd Strata
cmake -S . -B build-halo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build build-halo --target strata -j 8        # about 25 minutes the first time
```

用构建自己的说法，`gfx1151` 是 "unvalidated"（会发出警告）。`STRATA_PREFILL_MMQ=ON` 会构建 prompt 专家所需的
ggml MMQ 内核。测试：`cmake --build build-halo -j 8` 把它们全部构建出来，`ctest --test-dir build-halo` 运行它们；
`hip_prefill_hcd_exact_parity` 和 `hip_prefill_hipblaslt_gemm` 需要那张表（下一节），没有表时自动跳过。

## 3. prompt 表，以及运行
<a id="3-the-prompt-table-and-running"></a>

把那张表交给引擎（对受支持的显卡 `setup` 会代劳；gfx1151 要自己动手）：没有它，prompt 的稠密矩阵
乘法会用普通的 hipBLAS，速度更慢。

```sh
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1151-hipblaslt-100401.txt
```

引擎只接受与表的第二行所写明的架构和 hipBLASLt 版本匹配的表
（`STRATA_HIPBLASLT_TUNING_V1 gfx1151 100401`：即 ROCm 7.14.1），并在启动时说明这一点
（`prefill gemm: hipBLASLt tuning enabled (90 rows, gfx1151, version 100401)`）。
之后运行 `strata` 时沿用与其它显卡相同的参数（模型包、`--serve`、[DETAILS.md](DETAILS.md) 里的设置）；
下文数字用的都是 Unsloth UD-IQ4_XS 或 UD-Q4_K_XL GGUF 的模型包。

## 4. gfx1151 上默认开启的项
<a id="4-what-is-on-by-default-on-gfx1151"></a>

启动时引擎会查看 GPU 0；认出 gfx1151 后，除非你自己设过，否则它会设好下面这些开关
（`STRATA_GFX1151_DEFAULTS=0` 关闭整张表，启动日志会写明设置了什么）。**其中每一项在维护者的机器上都给出
字节级完全相同的输出**（相同的 token id，4K / 32K / 64K 与四个模型，且每个内核都有逐位校验工具）：
它们只改变速度，不改变答案。这里没有任何一条适用于 CUDA 或其它 AMD 架构。

| 开关 | 作用 |
|---|---|
| `STRATA_GDN_HEAD`、`STRATA_GDN_PP=2`、`STRATA_GDN_CONVL2`、`STRATA_GDN_NOY` | prompt 的 gated-delta-net 递推：每列四条 lane、每 lane 两列，卷积与 norm 融合 |
| `STRATA_HCD_EXACT` | 以 WMMA 内核实现超连接下投影（N 320、K 10240），复现 hipBLASLt solution 1176 / 1177 的求和顺序（`hip_prefill_hcd_exact_parity` 逐位校验）；只在表让 hipBLASLt 选中这两个解的地方运行，做不到时提示一次 |
| `STRATA_CVEC_FUSE`、`STRATA_PF_PAD` | 被引导层的写出、控制向量与下一个 norm 合为一趟；GEMM 行步长补齐 |
| `STRATA_Q8_PACKED`、`STRATA_Q6_PACKED`、`STRATA_MMVF_ROWS`、`STRATA_ATTN_LANECELL` | decode 布局与内核（打包的 Q8_0 / Q6_K 权重、4 行 BF16 GEMV、每线程一格 attention 分数） |
| `STRATA_EXPERT_V2`、`STRATA_TSUM`、`STRATA_LFUSE` | 分组 decode 专家层（IQ3_S / IQ4_NL）、单蝶形 warp 求和、共享专家周围与 KV 追加周围更少的启动 |
| `STRATA_GDN_SPLIT`、`STRATA_QFUSE`、`STRATA_PLE_BATCH` | 每头分四个 block 的 GDN 步进、由生产者写入的激活 image、一次完成 verify 窗口的 PLE 键 / 值投影 |
| `STRATA_SH_STREAM=1` | 让共享专家走旁侧的第二条流。它在其它 AMD 显卡上默认关闭（#816），但在 gfx1151 上划算：8K 下 decode +1.8%（UD-IQ4_XS）和 +6.7%（UD-Q4_K_XL） |

## 5. 默认关闭的项（会改变比特）
<a id="5-what-is-not-on-by-default-it-changes-bits"></a>

这些在 gfx1151 上更快，但**舍入方式不同**（求和顺序不同，或中间量为半精度），
所以默认关闭、需手动开启，维护者用对默认路径的 KL 来把关
（`STRATA_PF_SWITCH_MIN_T=4096` 让短 prompt 保持默认数值，因为它们只在长 prompt 上划算）。
维护者的高速配置会把它们全部打开：

```sh
export STRATA_PF_FUSED=1        # the prompt experts on the matrix cores (IQ2 / IQ3 / IQ4_XS gate/up, Q2_0 / IQ4_NL / Q8_0 down)
export STRATA_PF_GEMM=1         # the prompt's FP16 projections on the 128 x 256 WMMA GEMM
export STRATA_HC_UPMIX=1        # the hyper-connection up projection with its epilogue
export STRATA_PA_FAST=1         # the prompt attention with single FP16 q and p
export STRATA_HIP_WMMA=1        # the prompt attention on matrix cores
export STRATA_SELECT_WMMA=1     # the block scorer on matrix cores
export STRATA_HC_Q8=1           # the hyper-connection read from the GGUF's own Q8_0 projections
export STRATA_PF_SWITCH_MIN_T=4096
export STRATA_PREFILL_STREAM_MIN=128   # prompt reads of 128-1,023 tokens on the fused experts too (below)
```

`STRATA_PF_FUSED=1` 只覆盖不小于 `STRATA_PREFILL_STREAM_MIN` 个 token 的块（默认 1,024，这个下限是在独显上实测出来的，
独显上更小的块抵不过流式读取每个专家的开销）；更小的读取会让它的专家走 MMQ。
当所有专家都在统一内存里时，没有任何东西需要流式读取，MMQ 就成了慢的部分：agent 的一轮
——缓存对话上一个约 1,000 token 的工具结果——按 1,024 读取要 3.1-6.0 秒，按 128 只要 0.9-2.5 秒
（C2T8，`--no-prefill-borrow`；12 轮 replay 每轮 3.4 -> 2.1 秒）。它的输出与融合路径的输出相同，
而 1,024 token 及以上的读取本来就走融合路径；它不受 `STRATA_PF_SWITCH_MIN_T` 管辖，后者只管超连接和补齐这两类开关。
UD-Q4_K_XL 或其它 K-quant 的模型包必须再加 `STRATA_PF_FUSED_KQ=1` 才会采用融合专家。

两个引擎参数在长上下文上有用，且并非 Strix 专属：`--mtp-window 8192`（MTP 草稿层只关注最近 8,192 个
cell：64K 输出 +3.5%，32K -0.4%）以及 `--spec`、`--lookup-chain`、`--mtp-q4`（见 [DETAILS.md](DETAILS.md)）。

## 6. 实测
<a id="6-measured"></a>

Ryzen AI Max+ 395（128 GB，gfx1151），在第 4 节默认项之上叠加第 5 节维护者的高速配置，`--spec 4`、
`--mtp`、`--lookup-chain 3`、`--mtp-q4 all`、`--prefill 16384`，int8 KV，全部专家都在统一内存中。
prompt tok/s =（prompt token 数 - 1）/ prompt 时间，output tok/s = token 数 / decode 时间；3 次运行的中位数，
每次运行都起新进程，两个引擎交替执行。"Before" 是维护者更早的引擎分支（同样的开关手动设置），"now" 是本发行版。
每一对运行都打印出相同的 token id。

| 模型 | 上下文 | prompt before -> now | 输出 before -> now |
|---|---|---|---|
| UD-IQ4_XS | 8K | 1,212 -> 1,293 (+6.7%) | 51.7 -> 53.8 (+4.0%) |
| UD-IQ4_XS | 128K | 1,241 -> 1,320 (+6.4%) | 49.3 -> 51.4 (+4.2%) |
| UD-Q4_K_XL | 8K | 1,136 -> 1,169 (+2.9%) | 50.6 -> 52.6 (+3.9%) |
| UD-Q4_K_XL | 128K | 1,220 -> 1,248 (+2.4%) | 41.6 -> 43.3 (+3.9%) |
| IQ3_S | 8K | 1,202 -> 1,242 (+3.4%) | 56.2 -> 59.7 (+6.3%) |
| IQ3_S | 128K | 1,254 -> 1,312 (+4.6%) | 40.6 -> 42.8 (+5.5%) |
| IQ3_XXS | 8K | 1,237 -> 1,238 (+0.1%) | 48.8 -> 51.5 (+5.6%) |
| IQ3_XXS | 128K | 1,259 -> 1,252 (-0.6%) | 48.4 -> 50.9 (+5.1%) |

UD-IQ4_XS 在 4K / 32K / 64K 下（5-6 组交替对比）：prompt +9.2% / +6.0% / +5.3%，输出 +5.0% / +3.9% / +4.6%。
IQ3_XXS 在 8K 下用 6 组交替对比重新实测（此前一组 3 次运行的数据偏低 5%，当时 GPU 上还有其它任务在共享）：prompt 持平，输出 +5.6%。
关掉默认项后，同一构建的 prompt 低 8%（1,137），可见那张表在那里同样有增益。

### 在合并后的 0.1.40 代码上重新实测
<a id="re-measured-on-the-merged-0140-code"></a>

同一台机器、同一方法，完全不带环境开关（只有架构默认项），跑在最终合并的代码树上。每格 4 组交替对比取中位数；
"j" 是上表的构建（其它 0.1.40 工作合并之前的 Aurora 移植），"merged" 是发行代码。
每次运行都打印相同的 token id，且这些 id 在 UD-IQ4_XS 和 UD-Q4_K_XL 的 a8 / 4K / 64K 上也与 `strata-int29`（v4.2 配置）一致。

| 模型 | 上下文 | prompt j -> merged | 输出 j -> merged |
|---|---|---|---|
| UD-IQ4_XS | 8K | 1,260 -> 1,159 (-8.0%，spread 1,070-1,320) | 53.94 -> 53.14 (-1.5%) |
| UD-IQ4_XS | 128K | 1,317 -> 1,299 (-1.4%) | 51.38 -> 50.72 (-1.3%) |
| UD-Q4_K_XL | 8K | 1,163 -> 1,104 (-5.0%，spread 890-1,240) | 52.66 -> 50.47 (-4.2%) |
| UD-Q4_K_XL | 128K | 1,252 -> 1,211 (-3.3%) | 43.28 -> 42.64 (-1.5%) |
| IQ3_S | 8K | 1,283 -> 1,265 (-1.4%) | 59.81 -> 59.10 (-1.2%) |
| IQ3_S | 128K | 1,311 -> 1,310 (-0.1%) | 42.74 -> 42.09 (-1.5%) |
| IQ3_XXS | 8K | 1,127 -> 1,212 (+7.6%) | 51.64 -> 50.95 (-1.3%) |
| IQ3_XXS | 128K | 1,252 -> 1,250 (-0.1%) | 50.88 -> 50.22 (-1.3%) |

8K 下的 prompt 数字摆动幅度比这些差值还大（其它任务共享这台机器），所以应把它们读作持平。
输出列稳定移动了 -1.2% 到 -1.5%（UD-Q4_K_XL 在 8K 下为 -4%，该档的默认组也有偏慢的运行）：自 0.1.39 起，
共享专家的流分叉在 HIP 上默认关闭（#816）。它把共享专家与路由专家放到第二条流上重叠执行，在 gfx1151 上确实有帮助，
因此 0.1.40 在这颗芯片上把它默认打开（第 4 节）。它不改变任何比特（id 相同）。8K 下 4 组交替对比，
合并后的代码从分叉关闭 -> 开启：UD-IQ4_XS 输出 53.1 -> 54.05（+1.8%），UD-Q4_K_XL 49.5 -> 52.8（+6.7%，默认组区间 46.0-52.1），
prompt 持平。分叉开启时，合并代码的输出不低于 j 构建（IQ4_XS 54.05 对 53.94）。
