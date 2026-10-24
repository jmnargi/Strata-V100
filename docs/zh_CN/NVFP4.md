# NVFP4 路由专家层
<a id="nvfp4-routed-experts"></a>

引擎可以运行路由专家层为 NVFP4 的 Qwen3.8-Flash-Next，存储方式与 NVIDIA 的 ModelOpt 一致：以 64 个值为一个块存放 E2M1 码，每块配四个 UE4M3 子块缩放值，此外每个专家层和投影各有一个 FP32 的 `weight_scale_2`。其余部分（attention、GDN、共享专家层、路由器、输出头）都保持 native 路径本来就能读取的格式。实测环境：Windows 11、RTX 5090（32 GB）、Ryzen 9 9950X3D、128 GB DDR5-5600。

**setup 与 App 里的三个 NVFP4 版本。** `NVFP4` 是 "NVFP4 (Strata's version)"，`NVFP4-NVIDIA` 是 "NVFP4 (NVIDIA's version)"，`NVFP4-QAD` 是 "NVFP4 (NVIDIA, trained for FP4)"（[简短指南](MODELS.md#nvfp4)）。三个版本都能读图片，靠的是 setup 随它们一起下载的原模型图片读取文件（见下文「图片」一节和 FP4-Local 一节）。除非另有说明，本页其余各节对三个版本都适用。

## 获取模型
<a id="getting-a-model"></a>

```
python tools/nvfp4_convert.py --model <ModelOpt NVFP4 checkpoint dir> --outfile <model>.gguf
python tools/iq_pack.py --gguf <model>.gguf --out <pack dir>
```

- `nvfp4_convert.py` 用 llama.cpp 自己的转换器（`third_party/llama.cpp`，setup 钉住的那个 commit），按引擎需要的类型策略运行。专家层的重新打包是无损的：`tools/nvfp4_verify.py` 会逐 bit 与 checkpoint 比对。
  - BF16 内核读取的那些小投影（路由器、SSM 门控、indexer、PLE 和 hyper-connection 投影）写成 BF16。
  - 所有以 native 方式服务的部分都写成 Q8_0：attention、`ssm_out`、共享专家层、输出头，以及 token 嵌入。
  - 51.2e9 个值的 PLE 表不写入。
- `iq_pack.py` 会独立地为 NVFP4 GGUF 写出 `experts.bin`。每个 blob 是 `[gate | up | down]`，外加一个 16 字节的尾部 `{s_gate, s_up, s_down, 0}`。GGUF 没有地方存放这些缩放值，所以引擎只从 `experts.bin` 读取 NVFP4 专家层。

像运行 IQ 模型包一样运行它：GGUF 同时充当自己的稠密权重来源，PLE 表则取自某个 IQ GGUF 的 PLE 分片：

```
strata --pack <pack dir> --native <model>.gguf --native-dense-gguf <model>.gguf \
       --ple-gguf <an IQ model's shard with per_layer_token_embd> --mtp <mtp/rt> \
       --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 ...
```

NVFP4 checkpoint 只改变路由专家层。因此原模型的 PLE 表、MTP 层和专家 profile 可以原样使用。

已用 `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4` 完整跑通端到端测试。按其 `hf_quant_config.json`，NVIDIA 的 `nvidia/Qwen3.8-Flash-Next-NVFP4` 以同样的方式量化路由专家层（NVFP4，group 16；其余层 BF16，PLE 表 FP8，MTP 专家层 FP8），因此走同一条路径。这里没有按下载原样运行过它。setup 把它作为现成模型包提供，对应的大小是 `NVFP4-NVIDIA`。

## 实测数据
<a id="measured"></a>

RTX 5090，64K 上下文，int8 K/V，`--vram-reserve-mib 1500`：

| | NVFP4 模型包 | IQ2_XS，同一引擎 |
| --- | ---: | ---: |
| RAM 中的专家层 | 63.3 GiB | 33.0 GiB |
| 专家缓存槽位 | ~8,350 | ~17,300 |
| decode（解码），400 token 的回答 | 114-116 tokens/s | 132-139 tokens/s |
| 一条 32K 提示词 | 3,900 tokens/s | 5,600 tokens/s |

一个 NVFP4 专家 blob 是 2.76 MB，而 IQ2_XS 只有 1.51 MB：缓存能装的专家少一半，一次未命中经过 PCIe 的开销也是两倍的字节量。因此 NVFP4 模型包的未命中专家 PCIe 占比取 0.25，而不是通常的 0.55。在 0.55 时，拷贝内核占全部 GPU 内核时间的 40%（nsys，300 token decode）。在 262K 上下文下做的一次扫描得到以下 decode 速度：

| PCIe 占比 | 短提示词 | 32K 提示词 |
| --- | ---: | ---: |
| 0.55 | 103 tokens/s | 105 |
| 0.40 | 110 | 117 |
| 0.25 | 117 | 124 |
| 0.10 | 112 | 118 |

## 在 Strata 中的实测（2026-10-13）
<a id="measured-in-strata-2026-10-13"></a>

上面的数字是 fork 作者在他的 RTX 5090 和 128 GB 机器上测的。下面这些是在这里测的，用的是现成的 GPTQ 模型包（`--family nvfp4 --model NVFP4`）。

route tail skip 对回答的影响比量化本身还大，所以对 NVFP4 专家层它默认是关闭的。`STRATA_ROUTE_TAIL_SKIP=7` 打开它：RTX 3060 上 decode +7-25%，RTX 5070（64 GB）上 +47-55%。

在 decode 内核里，每个权重在一个 verify 窗口内对所有 token 只解码一次（在该端口的内核工作之前：3060 上 4 token 时 2.26 ms），结果逐 bit 相同。

RTX 5070 上的提示词专家层（一次启动 16 个专家、每个 625 行，gate/up + down）：w4a8 1.52 ms，w4a4x2（`STRATA_PREFILL_NVFP4=w4a4x2`）1.13 ms，w4a4x2 连 down 也算上（`STRATA_PREFILL_X2_MIN_K=640`）0.995 ms，UD-IQ4_XS 的专家层 1.20 ms。相对精确 FP16 路径的首 token KL（24 条提示词，1.5-8K token）：w4a8 0.0068，w4a4x2 0.0056，带 down 的 w4a4x2 0.0044。在 64 GB 的 PC 上，提示词要等待从 SSD 读取的专家层，所以各模式在那台机器上的提示词速度相同；在有一台 96 GB+ 的 RTX 50 PC 实测其他模式之前，w4a8 仍是默认值。在 RTX 20-40 上，NVFP4 的提示词乘积比 IQ3_S / IQ4_NL 的在 32K 下慢 1.35-1.6x：NVFP4 的 16 值缩放要在每个 int8 步骤之后以 FP32 施加，只有 Blackwell 的 FP4 tensor core 能避免这一点。

低于其 RAM 需求（68 GB 专家层再加上余量）时，setup 会以 RAM 预算模式运行该模型包（`--resident-budget-gib`，即 RAM 减 24 GB），与处理 Unsloth 的文件时相同。在 64 GB 的 Windows PC 上启用映射专家层（`--mmap-experts`）会把 RAM 占满。

图片：模型包保留 Qwen 未经量化的视觉塔，所以由原模型的图片编码器替它们读图（`--vision`；strata-vision 打开稠密 GGUF 是为了取词表）。在 RTX 5070 和 RTX 3060 上对照 IQ3_XXS 对同一张图片的回答做了检查（读出图中一个数字、两个形状及其颜色和边数）。编码器放在 GPU 上要占 ~1.1 GiB 显存：在 3060 上、8K 上下文下，带图片时缓存容纳 1,293 个专家，不带图片时是 1,734 个。图片默认开启；显存低于 24 GB 时 setup 和 App 选择 CPU 读取器，24 GB 及以上选择 GPU 读取器（`--vision cpu` / `--vision gpu` 可更改）。

## 全局缩放值放在哪里
<a id="where-the-global-scales-go"></a>

每个缩放值乘在自己那个投影的 FP32 输出上，就像 llama.cpp 施加 `.scale` 那样：`h = silu(s_gate * (G x)) * (s_up * (U x))`，`y = s_down * (D h)`。

这些缩放值各自约 1e-4。把 `s_down` 折叠进 up 在代数上等价，但会让隐藏值落到 1e-5 附近。隐藏值随后要按 FP16 缩放舍入成 Q8_0（CPU）或 q8_1（GPU）块，而 `amax / 127` 会深入 FP16 的非规格化区：某个专家的输出误差因此到了 2-12%，而不是 1.1%（`nvfp4_avx512_parity --expert`）。FP16 提示词路径在反量化后的 up 权重上（~1e-8）也有同样的问题。任何未来的折叠都必须对照 FP16 的数值范围校验。

## 提示词路径精度
<a id="prompt-path-precision"></a>

`STRATA_PREFILL_NVFP4` 决定批处理的提示词路径如何与专家层相乘：

| 模式 | 内核 | 激活值 | 单个乘积对比 FP64（`mmq_nvfp4_parity`） |
|---|---|---|---|
| `w4a8`（默认） | llama.cpp MMQ，int8 tensor core | q8_1，float 块缩放 | 0.53% |
| `w4a4` | llama.cpp MMQ，Blackwell FP4 MMA（仅 sm_120a） | NVFP4 | 8.6% |
| `fp16` | 反量化 + cuBLAS FP16 GEMM | FP16 | 基准 |

`src/prefill/mmq_nvfp4_w4a8.cu` 编译 mmq.cuh 的 int8 NVFP4 路径，并把 Blackwell 的 FP4 MMA 对设备代码和宿主侧 tile 配置都隐藏起来，所以它在每种架构上都一样；decode 本来就以这一精度做乘法。FP4 MMA 只存在于 arch 专有的 target 里，所以 `src/prefill/mmq_nvfp4_w4a4.cu`（FP4 x FP4 内核及其 FP4 激活量化器）是 CMake 唯一以 `12x` -> `12xa` 构建的单元（`strata_mmq_w4a4`）；引擎其余部分按给定架构构建，也能在 sm_121 上运行。没有该单元镜像的显卡（其他架构，或 sm_121 对着 120a 构建）会带一条提示回退到 `w4a8`。

相对 `fp16` 的首 token KL 散度，8 条提示词（代码和散文，1K-8K token）。噪声底是同一个 `fp16` 路径按 4096 token 分块而不是 8192：另一种求和顺序，算术完全相同。

| 模式 | KL 均值 | KL 中位数 | KL 最大值 | top-1 一致率 | prefill（预填充），4-8K 提示词 |
|---|---|---|---|---|---|
| 噪声底 | 0.00023 | 0.00003 | 0.0016 | 8/8 | - |
| `w4a8` | 0.0018 | 0.0010 | 0.0096 | 8/8 | 2,503 tok/s |
| `w4a4` | 0.0080 | 0.0027 | 0.040 | 7/8 | 2,904 tok/s |
| `fp16` | 基准 | | | | 1,905 tok/s |

`w4a8` 是默认值。它离基准比 `w4a4` 近 4.5x，速度是后者的 86%，而且每个生成的 token 本来就以这个精度运行。当提示词必须被精确读取时，选 `fp16`。

`STRATA_DUMP_MOE_INPUT` + `mmq_nvfp4_parity --real` 在真实提示词的行上测试单个乘积：第 20 层、该层最忙的 12 个专家，median 的 max|x|/rms 为 4，最大为 9。误差如下：

| 模式 | gate/up | down（SwiGLU 尾部更重） |
| --- | ---: | ---: |
| `w4a8` | 0.46% | 0.90% |
| `w4a4` | 7.2% | 8.4% |
| `fp16` | 0.017% | 0.021% |

## CPU 专家层
<a id="cpu-experts"></a>

`src/kernels/cpu/nvfp4_avx512.cpp` 在 512-bit 通道上计算 CPU 池的 NVFP4 行。每个 64 值的块在一个 verify 窗口的每个 token 上只解码一次：对 |E2M1| 码做 `vpdpbusd`，符号移到激活值上。算术与 ggml-cpu 的 `ggml_vec_dot_nvfp4_q8_0` 相同，只是浮点加法的顺序不同（相对误差最差 4.5e-7，`nvfp4_avx512_parity`）。`STRATA_NO_NVFP4_512=1` 回退到 ggml-cpu。

没有 AVX-512 的 CPU（Zen 1-3、Intel Core 4th-10th 和 12th-14th gen、Core Ultra、Xeon E5 v3/v4）在 256-bit 通道上跑同样的行（`src/kernels/cpu/nvfp4_avx2.cpp`）：照搬 ggml 的 AVX2 算术并逐步对应，因此每一行都与 ggml-cpu 逐 bit 相等（`nvfp4_cpu_parity`，在 Ryzen 5 5600X、Ryzen 7 255 和 Core Ultra 7 265 上，1280x9 / 2560x9 的行有 0 行不一致）。`STRATA_NO_NVFP4_256=1` 回退到 ggml-cpu。NVFP4 模型包是 native 模型包（其文件夹里有 `native_experts.txt`），所以引擎在这样的 CPU 上能启动；AVX-512 的要求只属于规范的 Q2_0 模型包（#2347：没有 `native_experts.txt` 的 `--pack` 文件夹会被读作那个包，且引擎现在会在拒绝之前先说明这一点）。

| token 数 | ggml-cpu | AVX-512 | 加速比 |
|---|---|---|---|
| 1 | 0.102 ms | 0.058 ms | 1.76x |
| 4 | 0.410 ms | 0.125 ms | 3.29x |
| 7 | 0.719 ms | 0.196 ms | 3.66x |

这些是单个专家的 gate+up（2560 行中的 1280 行，缓存常驻）。池本身受 DRAM 限制。

DRAM 视角下的池（`nvfp4_avx512_parity --bw THREADS TOKENS`：1536 个专家，4.2 GB，读出的权重 GB/s；tokens = 一个专家在一个 verify 窗口内服务的 token 数，decode 中为 1-2）：

| CPU（线程数） | 内核 | 1 线程，1 tok | 全部核心，1 tok | 2 tok | 4 tok |
|---|---|---|---|---|---|
| Ryzen 7 255，Zen 4（8） | AVX-512 | 15.4 | 58.6 | 57.6 | 47.7 |
| | AVX2 行 | 13.9 | 58.9 | 58.1 | 42.1 |
| | ggml-cpu | 12.6 | 58.7 | 44.8 | 22.4 |
| Ryzen 5 5600X，Zen 3，DDR4-3200（6） | AVX2 行 | 12.1 | 42.2 | 41.5 | 30.1 |
| | ggml-cpu | 10.4 | 42.9 | 30.9 | 15.5 |
| Core Ultra 7 265，DDR5-5600（20） | AVX2 行 | 16.0 | 74.2 | 73.0 | 60.6 |
| | ggml-cpu | 10.3 | 74.1 | 62.7 | 37.1 |

没有 AVX-512 时，单个核心读得慢 ~10%（13.9 vs 15.4 GB/s），但一个以读 DRAM 为主的池在一个窗口两个 token 时速度相同，四个 token 时慢约 12%。决定速度的是 CPU 的内存，不是它的向量宽度。

## 其他 GPU
<a id="other-gpus"></a>

只有可选的 `w4a4` 需要 Blackwell。默认的 `w4a8` 是 ggml 的 int8 MMQ（sm_75 及以上），decode 使用引擎自己的 q8_1 内核，以软件方式解码 UE4M3/E2M1。引擎以 PTX 为 sm_75/86/89 构建，并在 RTX 5090 上运行过，由宿主按该代身份应答。在这些构建上，`mmq_nvfp4_parity` 和 `nvfp4_expert_gpu_parity` 给出的结果与 sm_120 完全相同。

HIP：decode 内核不使用 CUDA 专有的 intrinsic。NVFP4 MMQ 实例和 W4A8 单元是 CUDA 专有的，且有保护（`GGML_USE_HIP`）。这里没有为 HIP 构建过其中任何一个。

## 测试
<a id="tests"></a>

- `nvfp4_expert_gpu_parity`：decode 内核对照 ggml 的参考实现。
- `nvfp4_avx512_parity`：CPU 行对照 ggml-cpu（加 `--expert`：整个专家）。
- `mmq_nvfp4_parity`：提示词路径的乘积对照 FP64（加 `--real`：`STRATA_DUMP_MOE_INPUT` 转储的行）。
- `tools/nvfp4_verify.py`：转换过程，逐 bit。

## NVIDIA 的 FP4-Local checkpoint
<a id="nvidias-fp4-local-checkpoint"></a>

`nvidia/Qwen3.8-Flash-Next-FP4-Local`（setup 大小 **NVFP4-QAD**）是 NVIDIA 自己重新训练的 4-bit 模型：ModelOpt 0.47 safetensors，专家层经量化感知蒸馏重训。Strata 通过在用户 PC 上转换它来读取这个 checkpoint；不重新分发任何 NVIDIA 权重（许可证：NVIDIA Open Model Agreement 和 Qwen Community License 1.0）。

checkpoint 里有什么，以及 `tools/nvfp4_local_pack.py` 如何处理它们：

| checkpoint 中 | 转换结果 |
| --- | --- |
| 路由专家层：NVFP4（E2M1，每 16 个值一个 E4M3 缩放，每个张量一个 F32 缩放）；缩放以 NVIDIA 的 "CSF" 形式存储（`.nvfp4_csf_fixed` U8 + `.nvfp4_csf_exceptions` U32） | 缩放被解码为普通 E4M3 矩阵（字节精确；该格式就是 vLLM 加载器文档所写的格式），然后与其他所有 NVFP4 模型包完全一样地重新打包：`experts.bin`，同样的 36 字节超级块加 16 字节缩放尾部。已校验：304 个完整专家和整个 `experts.bin`（SHA-256）与经由完整 GGUF 的路径结果相等 |
| attention、GDN 投影、共享专家层、indexer：MXFP8（E4M3 值，每 32 个值一个 E8M0 缩放） | 精确反量化后转 Q8_0（同样是 32 值的块；在 MXFP8 自带的 ~3% 之上再加 +0.6% RMS），或对被 BF16 内核读取的小投影用 BF16 |
| n-gram（PLE）表，26.8 GiB：每行 160 个值的 NVFP4 | 一个单张量 GGUF，每行 90 字节（80 B 码 + 10 个 E4M3 缩放，整张表一个 F32），由 `ngram.cpp` 中新的 PLE 格式 `NVFP4`（`strata.ple.format = nvfp4x16`）读取；行大小与 IQ4_NL 相同 |
| MTP 头：NVFP4 专家层，其余 BF16 | 用 `mtp_pack.py` / `mtp_rt.py` 构建的 Q2_0 草稿层（225 个测试请求中 88.7% 的草稿被接受） |
| 视觉塔 | 不转换。图片由原模型的图片读取器读取（setup 随附下载的共享 `mmproj` 文件，与其他 NVFP4 版本相同）；稠密 GGUF 提供词表 |

`python tools/nvfp4_local_pack.py prepare --model <checkpoint> --out <dir>` 运行每一步（setup 就是这么做的）：n-gram 表、稠密 GGUF（不含专家层）、模型包（专家层直接从 checkpoint 写出）和草稿层。除了 102 GB 的下载之外，还需要约 110 GB 的可用磁盘，下载可以在完成后删除。在一台 Threadripper PRO 上、文件位于 NVMe 时，大约花了 12 分钟。

在 RTX PRO 5000 72 GB（Blackwell）上实测，128K 上下文，int8 KV，全部专家在显存里：decode 159 tokens/s（聊天回答的中位数，137-216），4K 提示词 3,981 tokens/s，16K 4,813，32K 4,916（首 token 6.6 s），agent 会话 151 tokens/s；引擎在 72 GB 显存之外还占着 69 GB RAM。对照 Unsloth 的 Q8_0（set v1）：top-1 90.5%，KL 均值 0.085，困惑度 3.95（Q8_0：4.05）。这是一个重训过的模型，所以它与原模型 Q8_0 的差距包含真实的权重变化，而不只是舍入。
