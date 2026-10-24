# Unsloth UD-Q4_K_XL
<a id="unsloth-ud-q4_k_xl"></a>

自 MODEL-UPDATE Phase 2 起它是一种常规规模（原先是实验性的）：setup 从引擎 0.1.32 起提供它
（[下文](#setup)）；那一节之后的手工流程适用于引擎 0.1.31 或更新。首次验证于 Windows 10、
一块 RTX 5070（12 GB）、64 GB 内存与一颗 AVX-512 CPU（2026-10-01）；此后又在 R9700（AMD，
每个专家层都在内存中）与一块 RTX A4000（62 GB，内存预算模式）上运行过——质量、速度与图片见
[QUANTS.md](QUANTS.md)。

目标是 Unsloth 对其他模型包所用的同一个模型做的 4-bit 量化：
[unsloth/Qwen3.8-Flash-Next-GGUF，`UD-Q4_K_XL`，修订 `38bb39e`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL)。
"UD-Q4_K_XL" 是多种格式的混合，并非到处都用 Q4_K：路由专家层的 gate/up 是 Q4_K（第 2 层是 Q5_K），
down 是 Q5_1（第 2、4、30、46、47 层是 Q8_0）；token 嵌入、输出头、注意力与共享专家投影以及 PLE 的
key 是 Q8_0；28.8 GB 的 PLE 表是 IQ4_NL。路由专家层占 71.7 GiB，约为 3-bit 模型的两倍。

## Setup
<a id="setup"></a>

`START-HERE.bat --setup`（Linux：`./setup.sh --setup`）并选择 **Qwen3.8-Flash-Next (Unsloth)**；或者直接用：

```sh
START-HERE.bat --setup --family unsloth --model UD-Q4_K_XL
```

setup 针对这个模型的做法有什么不同：

- 当 72 GiB 的专家层再加 16 GiB 能放得进内存时，让每个专家层都在内存中（约 88 GB：不用
  `--resident-budget-gib`；这条规则取自 sergqwer 的 fork，MIT 许可），否则使用下文的内存预算。它需要
  48 GB 或更多的内存（更少时会询问，默认为否；`--model UD-Q4_K_XL --yes` 仍然会安装它）以及引擎
  0.1.32 或更新（在下载任何东西之前检查）。NVIDIA 或 AMD 都可以：它在 R9700 上也跑通了（#429 的那个
  问题已不存在）。预算模式默认使用一块 GPU（这是经过测试的设置）；多块卡见下一点。不使用实验性的速度
  投影（未配合它测试）。图片（#967）在 16 GB 或以上的 NVIDIA 显卡上、以及在每个专家层都能放进内存的
  AMD 上默认开启（CPU 编码器）；已在 CUDA 与 HIP 上端到端运行（QUANTS.md）。
- 多块 GPU，且引擎能在层切分时保留常驻的内存副本（0.1.40 及更新，#642）：内存预算在切分时保留。
  每块卡缓存它所负责那些层的专家层，而预算按整份专家配置档保留其余专家中最热的那些（即没有任何卡
  持有的那些专家），于是内存的作用与单卡时相同。Setup 会询问（单块 GPU 仍是默认；`--gpus 0,1`、
  setup 时选 "2"，或启动时提示两块卡时选 "y" 都会采用切分），而在多块 GPU 上启动会保留
  `--resident-budget-gib`。在 RTX 3090（x16）+ RTX 3080 Ti（x4）、91 GB 内存、预算 64 GiB、
  引擎 0.1.41 上实测：decode 45 tok/s，而单独用 3090 是 38 tok/s（自动切分给那块 x4 的卡两层）；
  内存中 57.6 GiB 的专家层，每个请求没有文件读取。
- 多块 GPU 配合较旧的引擎（#498）：当内存足以容纳这些 GGUF 文件再加 24 GB（约 135 GB 内存）且两块或
  更多卡能共享它时，setup 会询问（单块 GPU 仍是默认；`--gpus 0,1` 采用切分）。切分**不**带内存预算
  运行：启动时全部 77 GB 的专家层从 GGUF 载入内存，文件在加载期间经过操作系统的文件缓存，配置得到
  `"gpu": [0, 1]` 与 `"layer_split": "auto"`。在 2x RTX 3090 配 165 GiB 上实测（#498）：单块卡带预算
  时 decode 31 tok/s，切分时 64-78 tok/s（128K 提示词下 55 tok/s），`MemAvailable` 始终不低于 68 GiB。
  内存更少时，`--gpus` 会保留单块 GPU 并说明原因，而在已安装的 UD-Q4_K_XL 上运行
  `START-HERE.bat --gpus 0,1` 会带着原因停止（它过去会保留预算，而引擎以退出码 2 退出）。
- 它从固定的修订 `38bb39e` 下载下文四个分片（可断点续传，与其他模型一样），然后对照下表的表格检查每个
  文件的大小与 SHA-256；这项检查一次性花费几分钟，并记录在文件的完成标记里。哈希不对的文件会被删除，
  因此下一次运行会重新下载它。
- 它用 `--compat-bf16` 打包，且从不写 `experts.bin`（`--low-ram` 不适用于此）。
- 内存预算（约 88 GB 以下的内存）是 PC 的内存减去 24 GB：64 GB 上是 `--resident-budget-gib 40`，
  在一台需要指定预算的 PC 上最多为其全部 71 GiB 的专家层。从 64K 上下文起，KV 缓存会移到内存中，
  它的体积要从预算里出。Setup 的 `--resident-budget-gib N` 可设成别的值（较大的值会被采纳，并附注
  这样做的风险）。
- 对 14 GB 以下的显卡，它建议用 8K 上下文（KV 缓存每占 1 GB，被缓存的专家层就少 1 GB）。

## 文件
<a id="the-files"></a>

四个分片，合计 111,334,654,784 字节（111.3 GB）。分片 1 只含元数据；第 11 层的 down 投影在分片 2、
它的 gate/up 在分片 3，本引擎能处理这种情况（`native_experts.txt` v4）。

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |

请把四个文件按这些名称放在一起（引擎从分片 1 的名称找到分片 2-4；缺失的分片会由一个指名它的错误报出）。
Setup 自己会检查它们；手工方式请在打包前先检查（`sha256sum -c`，Windows 上用 `Get-FileHash`）。
如果你传入的是快照中的文件名而不是它所指向的以哈希命名的 blob，Hugging Face 快照的符号链接也能用。

## 它需要什么
<a id="what-it-needs"></a>

| | |
| --- | --- |
| 磁盘 | 四个文件 111.3 GB，模型包 1.4 GB，若你还没有 MTP 草稿层则约 6 GB。**没有 `experts.bin`**：引擎就地从 GGUF 文件读取专家层。NVMe SSD 很重要：既放不进显存也放不进内存预算的那些专家层，每个 token 都要从它上面读取。 |
| 内存 | 实测 64 GB。内存预算（见下文）保存使用最频繁的专家层；其余部分经操作系统文件缓存从 SSD 读取。 |
| GPU | 实测 12 GB（RTX 5070）：在权重、草稿层与缓冲区之后，4K 上下文下专家缓存持有 24,576 个专家中的 1,280 个（3.7 GiB）。 |

## 模型包（手工）
<a id="the-pack-by-hand"></a>

构建 Strata（或安装 0.1.31 及以上），然后在仓库根目录，使用来自固定版本 llama.cpp 的 gguf-py
（setup 会安装它；`STRATA_GGUF_PY` 可以指向它的 `gguf-py` 文件夹）：

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --out packs/ud-q4_k_xl --compat-bf16
```

约 30 秒。它会写出 `index.txt`、`dense.bin`（1.4 GB）、`native_experts.txt`、分词器、`conversions.json`
与 `compat-bf16.json`。**`--compat-bf16` 是必需的**：这个文件把 195 个小的投影（hyper-connection 的
up/down 矩阵、`output_hc_*` 以及 PLE 的 value）存成 Q8_0，而引擎按 BF16 读取它们。没有该标志时打包器会
拒绝并不写任何文件；有它时，这些张量会被反量化并舍入到 BF16（就近取偶）。不要加 `--experts-bin`：
那会写出 77 GB 的专家层副本，而本模式并不需要它。

## 具体数字：什么是原生的、什么是转换过的
<a id="the-numbers-what-is-native-and-what-is-converted"></a>

路由专家层、token 嵌入、输出头、注意力与共享专家投影、PLE 的 key 以及 PLE 表都按 GGUF 存储的样子使用
（引擎对 Q4_K、Q5_K、Q5_1 与 Q8_0 有 GPU 内核以及 CPU（ggml）路径）。引擎按浮点读取的小张量会被转换，
而 `conversions.json` 会列出每一个的来源/目标类型、方法、是否精确、最大绝对误差以及源字节的 SHA-256：
264 个 F32 张量（路由器、注入、SSM 门控）其数值本来就是 BF16 的，被精确存为 BF16；上文 195 个 Q8_0
投影被舍入到 BF16（最大绝对误差 0.0144，出现在 `hc_ffn_up`）；PLE 的卷积（F32）在引擎加载它时收窄为
F16（最大误差 3e-8）。这就是 Unsloth 的文件所需要的兼容处理；它不是原始的 BF16 checkpoint。
输出与 llama.cpp 在同一文件上的一致：[质量](#quality-against-llamacpp-on-the-same-file)。

## MTP 草稿层
<a id="the-mtp-draft-layer"></a>

草稿层属于基础模型，与其他模型包用的是同一个。如果 setup 安装了 Strata，你已经有了它
（`Strata-data` 中的 `mtp/rt`）。否则按 [docs/ORCA.md](ORCA.md#preparation) 所示构建
（`tools/mtp_fetch.py`、`tools/mtp_pack.py`、`tools/mtp_rt.py`，然后把 `data/draft_vocab.bin` 复制到
`mtp/rt/`）。

## 服务器
<a id="the-server"></a>

在仓库根目录（或 setup 其他 `strata-*.json` 旁边）保存为 `strata-ud-q4_k_xl.json`，
其中 `/path/to/` 指向**分片 1**。不需要 `--ple-gguf`：引擎会按名称找到持有 PLE 表的分片
（`per_layer_token_embd.weight`，分片 2）。`--resident-budget-gib` 用一个内存预算开启映射模式
（它隐含 `--mmap-experts`）。

```json
{
  "exe": "engine/strata.exe",
  "args": [
    "--pack", "packs/ud-q4_k_xl",
    "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--resident-budget-gib", "40",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "8192"
  ],
  "cwd": ".",
  "tokenizer": "packs/ud-q4_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q4_k_xl",
  "log": "strata-ud-q4_k_xl.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-ud-q4_k_xl.json --port 8080
```

（`engine/strata.exe` 是 setup 在 Windows 上的引擎；自行构建的是 `build/strata`。）专家配置档属于基础
模型（`data/expert-profile.bin`）；它对同样的 48 x 512 个专家排序，并决定 GPU 缓存与内存预算优先取哪些。

**如何选择内存预算 N：** 你的内存减去 20-24 GB（操作系统、引擎自身、126 MB 的路由器副本，以及为
服务其余部分的操作系统文件缓存留出空间）。64 GB 上是 40。引擎会检查可用内存，并在 Windows 上检查可用
提交容量。如果 N 超过两者中较小值减去 4 GiB 的余量，它会把 N 降到该数值减去 256 MiB 的裕度，并报告
生效的是哪个上限。在 #403 的修复之前，被钳制的预算随后可能通不过安全检查。超出 N 的一切每个 token
都要从 SSD 读取，所以 N 是最重要的设置；更大的预算在每次测量中都更快（24 / 32 / 40 GiB）。当驱动把
整份预算页锁定（这台 PC 上 24 GiB 会，32 与 40 不会）时，GPU 还会像低内存常驻模式那样通过 PCIe 计算
未命中中的一部分；`STRATA_RESIDENT_PIN=0` 使用普通的可分页内存（那样每个未命中都由 CPU 计算）。

在 Windows 上，分配日志会报告分配之前以及失败之后的可用物理内存与提交容量。仅有可用内存并不能保证
Windows 能提交所请求的内存。如果可分页的回退也失败，警告会同时保留 GPU 的分配或映射错误与 `malloc`
错误。

**上下文：** 在 4K 下实测。KV 缓存要从专家缓存那里占用显存，所以更长的上下文会让 decode 更慢；
12 GB 上 8K 是一个合理的起点。`--kv int8` 会把 KV 缓存的显存减半。

## 可以期待什么（RTX 5070 12 GB、64 GB 内存、Windows、一个短的贪心提示词）
<a id="what-to-expect-rtx-5070-12-gb-64-gb-ram-windows-one-short-greedy-prompt"></a>

| 内存预算 | 贪心输出 | 备注 |
| --- | --- | --- |
| 24 GiB | 5.1-5.2 tok/s（2 次运行，在下文的路线预取之前） | 每轮 verify（约 3.5 个 token）从 SSD 读取 2.2 GB |
| 40 GiB | **7-8.5 tok/s**（6 次运行：6.7-8.6，均值 7.9） | 每轮 verify 从 SSD 读取 1.1 GB，每个 token 约 0.33 GB |

- 第一个回答在引擎启动后约 60 s 出现（大多是加载：内存预算是在启动时从 GGUF 文件拷入的）。
  随后一个 30 token 的提示词花了 5-6 s。
- 长提示词（引擎 0.1.32）：一个 16K 提示词以 **160 tokens/s** 读取（15,873 个 token 用时 99 s；
  0.1.31：57 tokens/s，273-282 s）。读取提示词时，每个 4K 分块会把每个专家层各流读一次，其中不在
  内存预算内的那三分之一来自 SSD（每个分块约 35 GB），所以当专家层是就地从 GGUF 读取时，提示词路径
  现在用 32 个线程、最多 128 个在途 blob 来读取它们（`STRATA_STAGER_THREADS` / `STRATA_STAGER_RING`
  可覆盖；原先是 4 与 16，其他所有模型仍用该默认值）。在各分块之间，28.8 GB 的 PLE 表的行每个分块要花
  约 5 s。
- 有了 `STRATA_MMQ_KQ=1`，引擎会用 llama.cpp 的 MMQ 内核来计算提示词上 Q4_K / Q5_K / Q5_1 专家层的
  乘法（就像其他模型的格式一直做的那样），而不是把它们反量化为 FP16：这些乘积的 GPU 时间少 4 倍，
  但在这台 PC 上提示词无论如何都在等 SSD（每个 4K 分块都是 29 s，开关都一样）。可选开启：它的舍入与
  FP16 路径不同。在 CUDA 上这些内核是引擎旁边的一个库（`libstrata_kq.so` / `strata_kq.dll`），
  只在需要时才打开，因此不会从其他模型那里占用显存（[QUANTS.md](QUANTS.md)）；此前它们是构建选项
  `-DSTRATA_MMQ_KQUANTS=ON`，会从每个模型占用 1-3 个专家槽位。
- 速度逐次运行有波动（同一提示词在 40 GiB 下是 6.7-8.6 tok/s），取决于操作系统文件缓存里有什么。
- 作为对比，IQ3_S（它的全部专家层都在内存中）在同一台 PC 上写到约 53 tok/s
  （[速度表](DETAILS.md#speed-measured)）。
- 时间花在哪里（命令行加 `--stats`）：在 40 GiB 下，每轮 verify（3.5 个 token）约 550-750 ms 是在从
  SSD 读取专家层，约 75 ms 是 CPU 的专家内核，约 13 ms 是 GPU。

## 质量：在同一文件上与 llama.cpp 比较
<a id="quality-against-llamacpp-on-the-same-file"></a>

让 Strata 与 llama.cpp（固定的提交 `3cf0325`，一个以内存映射方式读取 GGUF 的 CPU 构建）接收相同的
token 序列，并在每个位置各自写出它认为最可能的 20 个后续 token 及其对数概率：Strata 通过它的 verify
window（`STRATA_LOGPOS` 配合 `STRATA_LOGPOS_TOPK=20`），llama.cpp 通过一个批次上的
`llama_get_logits_ith`。Strata 使用推荐的设置（内存预算 40 GiB、MTP 草稿层开启、贪心、
`--adapt-every 100000` 以便专家缓存在运行期间不动）。逐位置统计：最可能的 token 是否相同（argmax
一致率）、双方各占有 5 / 10 个最可能 token 中的多少个、KL 散度（llama.cpp 的分布相对 Strata 的，
在 llama.cpp 的 top 20 之上再加一个收纳其余的桶），以及每个模型下该序列的困惑度。

| Token 集 | 位置数 | Argmax 一致率 | Top-5 / top-10 重合 | KL | 困惑度 Strata / llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 代码：Strata 对一个编码提示词的贪心回答（Python 的 LRU 缓存，56 token 提示词） | 400 | **99.0%** | 95.6% / 95.3% | 0.0008 | 1.067 / 1.070 |
| 思考：Strata 对一个推理谜题的贪心回答（带思考，55 token 提示词） | 324 | **97.5%** | 96.2% / 96.4% | 0.0016 | 1.107 / 1.106 |
| 16K 提示词之后：Strata 对下文 16,384 个真实文本 token 的贪心续写（256 个 token） | 256 | 90.2% | 88.1% / 89.3% | 0.039 | 1.619 / 1.623 |
| 16K 提示词之后的真实文本：本仓库文档与代码的 16,384 个 token 的最后 510 个 | 510 | 91.2% | 88.3% / 89.9% | 0.035 | 4.27 / 4.29 |
| 同样的 511 个位置，但前面只有 512 个 token | 511 | 89.2% | 86.0% / 86.3% | 0.061 | 8.99 / 8.87 |

作为对比，[eddoursul/Strata](https://github.com/eddoursul/Strata) 在 RTX 3090 上对同一文件与
llama.cpp 比较，报告了 98.2-99.3%（代码）、95.6-95.8%（思考）与 97.7%（16K 提示词之后）。

**差异出现在哪里：** 出现在接近平局的地方。按 llama.cpp 自身最可能与次可能 token 之间的差值分组后，
在代码与思考回答中该差值为 0.2 或以上的全部 695 个位置（695/695）上，Strata 都选出同一个 token；在
16K 提示词之后差值为 0.5 或以上的全部位置上也是如此（续写中 144/144，真实文本中 254/254；在 0.2 到
0.5 之间是 50/54 与 117/121）。长文档之后的文本比短回答有更多接近平局的位置（续写 256 个位置中有 58
个、真实文本 510 个位置中有 135 个差值小于 0.2，而代码回答 400 个中只有 13 个），这就是它一致率更低的
原因。这并非长上下文本身：同样的真实文本位置在只有 512 个 token 前文时一致率略低（89.2%），而每组中
困惑度都在 3% 以内吻合。为了有个参照，Strata 自己的贪心运行与它对同一段续写的 teacher-forced 重跑，
在 6% 的位置上选出不同的 token（93.8% 相同：其他窗口大小、显存中其他专家层），属于同一类的近局翻转。
16K 的提示词由 Strata 的批量提示词路径读取（15,872 个 token），其余通过 verify window 读取。这些数字
是 FP16 提示词路径（随发行版的引擎）的结果；使用 MMQ 提示词路径（上文的 `-DSTRATA_MMQ_KQUANTS=ON`）
时，16K 提示词之后的续写一致率为 92.2%（KL 0.029，三次运行，完全一致）。

要重复这项测试：Strata 一侧是 `STRATA_LOGPOS=<file>` 配 `STRATA_LOGPOS_TOPK=20`（引擎 0.1.32），
运行一个带 `--short-read` 覆盖待比较位置的 serve 引擎；llama.cpp 一侧是一段基于 `llama_decode` /
`llama_get_logits_ith` 的短小程序，写出每个位置同样的 top 20。

## 可选开启项与开关（环境变量）
<a id="opt-ins-and-switches-environment-variables"></a>

| | |
| --- | --- |
| `STRATA_LOOKAHEAD=0` | 关闭路线感知的预取（本模式默认开启）：当 CPU 在处理某一层时，一个线程会把下一层的路由器应用于本层的输入，并请操作系统读取被预测出的专家层的页。只是预读页面，答案不变。约一半的 SSD 读取被预测命中；均值 +14%（6.9 -> 7.9 tok/s）。`STRATA_LOOKAHEAD_K` 设置每个 token 的专家数（默认 10）。 |
| `STRATA_KQ256=1` | 用于 Q4_K / Q5_1 / Q8_0 专家层的多 token AVX2 内核。与 ggml 的逐位一致，但实测并不更快，所以关闭。 |
| `STRATA_PARTIAL_PIN=1` | 把内存预算中最热的部分（最多 `STRATA_PARTIAL_PIN_GIB`，默认 24）向 GPU 驱动注册，于是 GPU 通过 PCIe 计算未命中中的一部分（`--pcie-frac`）。在这台 PC 上实测并不更快，而且会改变这些专家层的数值行为（改用 GPU 内核而不是 CPU 的），所以关闭。 |
| `STRATA_FETCH_THREADS=N` | 在回答期间从 GGUF 读取专家层的线程数（默认 8；16 并不更快）。提示词路径有自己的：`STRATA_STAGER_THREADS`（这里默认 32）与 `STRATA_STAGER_RING`（128）。 |

## UD-IQ4_XS（setup 自 0.1.39 起支持，#621）
<a id="ud-iq4_xs-setup-from-0139-621"></a>

同一修订下 Unsloth 的 `UD-IQ4_XS` 也进入了 setup，作为一个常规选项（不是实验性；它是 Unsloth 家族
的第一个规模，也是其默认值）：`START-HERE.bat --setup --family unsloth --model UD-IQ4_XS`
（引擎 0.1.38 或更新）。它位于 IQ3_S 与 UD-Q4_K_XL 之间：它的路由专家层是 IQ3_S 的 gate/up（有一层是
IQ4_XS）配 IQ4_NL 的 down（五层是 Q8_0），共 59.5 GB；稠密侧（Q8_0 投影、IQ4_NL 的 PLE 表、一个 Q6_K
输出头）与 UD-Q4_K_XL 相同。三个分片，93.7 GB：

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

setup 把它当作 UD-Q4_K_XL 对待（上面那份列表）：同样的内存预算（你的内存减 24 GB；最多为其全部
55 GiB 的专家层，所以一台约 80 GB 或以上内存的 PC 能全部容纳），用 `--compat-bf16` 打包，不用
`experts.bin`，默认一块 GPU（多块卡时保留预算，同上）。它在 AMD 上不提问：它专家层的格式在那里同样
有提示词内核。图片是一个选项（会询问，默认关闭；NVIDIA）：图片路径对这个模型包没有限制，它使用原始模型
的图片编码器；尚未配合图片运行过。

目前已知：它能在 Strix Halo（AMD gfx1151、128 GB 统一内存）上打包并运行，那里提示词在 8K 与 64K 下
以约 820 tokens/s 读取，且回答通过了 8K 与 64K 的检索检查。它**尚未**在 NVIDIA 上实测；欢迎报告你看到
的情况。有一条适用于每个 `--compat-bf16` 模型包的保真度说明（UD-Q4_K_XL 也一样）：模型包会把文件的
Q8_0 hyper-connection 投影舍入到 BF16，而在 Strix Halo 上，这处舍入使困惑度比 llama.cpp 在同一文件上
高出 6-9%（teacher-forced、短上下文）。改为读取 Q8_0 的数值能消去其中的大部分差距；这只在 AMD 上实测过，
且不在本次发行中。

## Q4_0 文件
<a id="q4_0-files"></a>

一个用 `llama-quantize` 生成的普通 Q4_0 文件也以同样的方式运行：按上文用 `--compat-bf16` 打包分片 1，
并把 `--native` 指向它。已测试：
[bartowski/Qwen3.8-Flash-Next-GGUF，`Qwen3.8-Flash-Next-Q4_0`，修订 `928589f`](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_0)
（3 个分片，100.6 GB）。它的路由专家层 gate/up 与 down 都是 Q4_0，只有第 0-5 层的 down 是 Q4_1；
PLE 表与 token 嵌入是 Q4_0。模型包按存储原样服务 302 个张量，并转换 362 个（168 个精确，194 个舍入到
BF16，最大绝对误差 0.0156；1.23 GiB）。

Q4_0 的专家内核会在整数点积之前把码字中心化（q - 8），与 ggml-cpu 的 Q4_0 x Q8_0 点积相同，而不是
llama.cpp 的 CUDA 形式（码字不中心化，改为在浮点上减去 8 x Q8_1 的块和）；Q4_1 沿用 Q5_1 内核但不含
第五个 bit。在该文件上的 `native_expert_parity`（第 0 与 5 层：Q4_0/Q4_1，第 20 与 47 层：Q4_0/Q4_0）：
相对浮点乘积的误差在 CPU 与 GPU 上都是 1.1-1.3e-2，CPU 对 GPU 在 Q4_0/Q4_0 上是 1.1e-7，
搭配 Q4_1 的 down 时是 5e-4 到 1.1e-3。

在一块 RTX 3060 12 GB（PCIe 3.0 x16）、Xeon E5-2696 v4、121 GiB DDR4-2400、Linux、CUDA 12.4 上实测；
每个专家层都在内存中，`--max-context 131072 --kv int8 --kv-resident 20480`，专家缓存 `auto`
（1,960 个槽位，5.06 GiB），提示词分块 `auto`（6,144 个 token）；贪心，`reasoning_effort` medium，
一次一个请求：

| 提示词 | 提示词读取 | 输出 |
| --- | --- | --- |
| 63 个 token（散文） | - | 25.4 tok/s |
| 111 个 token（代码） | - | 28.0 tok/s |
| 24,461 个 token | 787 tok/s | 24.0 tok/s |
| 67,934 个 token（中段藏着一个事实：找到了） | 737 tok/s | 26.4 tok/s |

一个编码提示词被正确回答。未测量：该文件与 llama.cpp 的一致率（对 UD-Q4_K_XL 见
[上文](#quality-against-llamacpp-on-the-same-file)）、AMD、Windows，以及 bartowski 的 Swift-1.5
Q4_0，其头部显示的格式相同。

## 范围与验证
<a id="scope-and-validation"></a>

- 针对的是修订 `38bb39e` 下的 UD-Q4_K_XL 与 UD-IQ4_XS，以及普通的 Q4_0 文件（[上文](#q4_0-files)）。
  其他量化使用本引擎可能没有内核的格式；引擎会在启动时检查每一层的格式，并指名拒绝不受支持的那种。
- 测试：打包器的合成四分支与转换测试（`.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`）；
  CTest 的 `gguf_split_test`、`expert_layout_test`、`native_expert_parity_*`
  （三对真实专家格式对照 ggml-cpu、Q5_1 的 min 项、Q8_0 的行、Q4_0/Q4_0 与 Q4_0/Q4_1）、
  `prefill_mmq_kquant_test`（配合 `-DSTRATA_MMQ_KQUANTS=ON`：提示词路径的 MMQ 乘积，
  Q4_K / Q5_K / Q5_1 / Q8_0 对照 ggml 反量化后的权重）；以及用 Coder 对比就地模式与 `experts.bin`
  （token 与 logits 完全相同）。
- 真实运行：在每个预算与上述设置下对编码提示词的贪心回答（正确），彼此完全相同；在短贪心回答的
  97.5-99% 的位置上与 llama.cpp 选出相同 token，在 16K 提示词之后是 90-91%，差异大多出现在近局处
  （见上文）。尚未测试：带采样的 decode、长对话。欢迎报告你看到的情况。

## 致谢
<a id="credits"></a>

分片产物的加载器、专家格式的 GPU 内核，以及对 Unsloth 文件的打包，都参照
[eddoursul/Strata](https://github.com/eddoursul/Strata)（PR #245），它最先运行了这些文件。Q8_0 内核、
按角色的分片列，以及 PLE 卷积的 F32 -> F16 修正来自 @gopinath87607 的 PR #255；按角色的分片名称也由
@jagsan-cyber 在 PR #247 中提出。量化出自 [Unsloth](https://huggingface.co/unsloth)；模型属于
Qwen（[许可](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)）。
