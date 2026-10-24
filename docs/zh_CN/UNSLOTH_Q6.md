# Unsloth UD-Q6_K_XL
<a id="unsloth-ud-q6_k_xl"></a>

**setup 自 MODEL-UPDATE Phase 2 起支持**（`START-HERE.bat --setup --family unsloth --model UD-Q6_K_XL`，引擎 0.1.44
或更新：约 118 GB 起每个专家层（expert）都在内存中，否则会询问并保留一份内存预算；质量、速度与图片见
[QUANTS.md](QUANTS.md)；UD-Q5_K_XL 与 Q8_0 同理）。下文的手工流程仍然可用。这就是 Unsloth 为其他模型包所用的同一个
模型制作的 6-bit 文件：[unsloth/Qwen3.8-Flash-Next-GGUF，`UD-Q6_K_XL`，修订 `38bb39e`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q6_K_XL)。
请为引擎 0.1.38 或更新构建它（更早的引擎只能读 `per_layer_token_embd.weight` 的 IQ4_NL、Q5_0 或 FP8，见下文）。
UD-Q4_K_XL 的手工流程不要求 setup 做任何事，这个也不要求。

> **状态（MODEL-UPDATE Phase 1）。** Q6_K 的 gate/up 内核现已是每个构建的一部分（它们原先是可选的
> `-DSTRATA_Q6K_EXPERTS=ON`）：实测表明其他模型的专家缓存（expert cache）槽位数不受影响（[QUANTS.md](QUANTS.md)）。
> Q6_K 的 MMQ 提示词内核随 K-quant 库与 `STRATA_MMQ_KQ=1` 一起提供；没有它时提示词走 FP16 路径。该文件能跑在标准引擎上
> （AMD R9700：实测数据在 [QUANTS.md](QUANTS.md)）。

## 文件里有什么
<a id="what-is-in-the-file"></a>

六个分片，合计 169,165,382,688 字节（169.2 GB）（[大小与 SHA-256 来自 HF tree API](https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q6_K_XL)；
分片 1、2 和 6 在下载后还对照其 Hugging Face LFS 哈希逐位校验过）。分片 1 只含元数据；分片 2 只含输出头
（`output.weight`、`output_hc_*`，除 F32 的 `output_hc_norm` 外全是 Q8_0）；分片 3 恰好就是 PLE 表，别无其他。

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf` | 10,946,624 | `8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf` | 682,434,912 | `494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf` | 54,400,261,312 | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf` | 49,814,108,512 | `0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf` | 49,419,317,344 | `9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf` | 14,838,313,984 | `e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f` |

格式，用 `tools/gguf_reader.py` 从分片头读取（六个分片全部校验：分片 4 = 块 0-20、520 个张量，分片 5 = 块 20-41、
530 个张量，每层的类型即上文所列）：

- **路由专家层：** `ffn_gate_exps` 与 `ffn_up_exps` 是 Q6_K（在分片 6 的每一层都见到：块 41-47）；`ffn_down_exps` 是
  Q8_0（每层在 640×2560×512 下为 891.3 MB，而 gate 或 up 在 2560×640×512 下 Q6_K 为 688.1 MB）。48 层中有 47 层的
  gate 和 up 是 Q6_K，第 2 层是 Q8_0；down 在每一层都是 Q8_0（模型包的 `native_experts.txt`：47 行 ggml 类型 14/8，
  第 2 层是 8/8）。Q6_K 的 gate/up 是原生的：在下文的每条路径检查中引擎都在 GPU 上直接与其相乘，而 CPU 使用 ggml 的
  块布局（每 256 个元素 210 字节）；`native_experts.txt` v4 的按角色布局能处理分片边界恰好切过某一层的情形，
  正如 UD-Q4_K_XL 的第 11 层那种情况。
- **其余全部内容：** 注意力与共享专家投影、SSM、hyper-connection、PLE 的 key/value、token 嵌入与输出头都是 Q8_0，
  或在引擎读取浮点处为 F32/BF16 —— 与 [UD-Q4_K_XL](UNSLOTH_Q4.md) 用 `tools/iq_pack.py --compat-bf16` 处理的模式相同。
- **PLE 表**（`per_layer_token_embd.weight`，[160, 320001536]）：**Q8_0**，自成分片 3。一行是 5 个 32 元素的块、
  每块 34 字节 = 170 字节，整张表 54.4 GB，而 UD-Q4_K_XL 的 IQ4_NL 表是 28.8 GB。引擎 0.1.38 原生读取 Q8_0 行：
  `PleTable` 接受该类型，行读取器本来就按每行字节数参数化（它的自测跑 90、110 和 170 字节的行），且无需改动任何 GPU
  内核——`gather` 输出浮点数。（UD-Q4_K_XL 的表是 IQ4_NL；[tools/ple_fp8_pack.py](../../tools/ple_fp8_pack.py) 的 FP8 表
  需要原始 checkpoint，本文件不能替代。把这张 Q8_0 表转成 FP8 会用表精度换取小 6% 的行；直接读 Q8_0 没有那种转换误差。）

Q6_K 的 gate/up 每 256 个元素花费 210 字节，而 Q4_K 是 144，因此路由专家所占的文件份额按每个读取元素约增长 45%；
在同样硬件上，预期的 decode（解码）速度相应低于 UD-Q4_K_XL。表的行 gather 相比 IQ4_NL 增长 170/90 = 1.89 倍。

## 模型包（手工）
<a id="the-pack-by-hand"></a>

与 [UD-Q4_K_XL](UNSLOTH_Q4.md#the-pack-by-hand) 相同的流程，在仓库根目录：

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
  --out packs/ud-q6_k_xl --compat-bf16
```

**`--compat-bf16` 是必需的**，理由与那里相同（195 个小的 Q8_0 投影，引擎按 BF16 读取）。不要加 `--experts-bin`；
专家层留在 GGUF 文件里。本机已用该文件端到端构建完成（`packs/ud-q6_k_xl`，打包器自身的测试通过
`.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py` 覆盖分片与转换逻辑）：索引 1079 个张量，
303 个按原生方式服务，460 个张量被转换（264 个精确，196 个舍入，最大绝对误差 0.0144 —— 与 UD-Q4_K_XL 相同的分布），
195 个 compat-BF16 张量（1.20 GiB），专家区 1.38 GiB，以及 `native_experts.txt` v4（第 20 和 41 层带按角色的分片列）。

## 服务器
<a id="the-server"></a>

与 UD-Q4_K_XL 的[配置](UNSLOTH_Q4.md#the-server)类似，换成 Q6 的路径与名称。不需要 `--ple-gguf`：引擎会按表名找到
分片 3。在一台内存足以装下约 110 GB 路由专家层以及其余内容的机器上（共 499 GB，约 325 GB 空闲，
在撰写本文的机器上实测），不需要内存预算，也不需要摆弄 `--expert-cache`：从 Q4 配置中去掉
`--resident-budget-gib` 与 `--expert-cache-device1`。54.4 GB 的表使用 mmap（默认）——除非内存确实多到可以锁定它，
否则不要改为 `--ple-io ram`。`mtp/rt` 一如既往是基础模型的草稿层。

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/ud-q6_k_xl",
    "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "8192"
  ],
  "cwd": ".",
  "tokenizer": "packs/ud-q6_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q6_k_xl",
  "log": "strata-ud-q6_k_xl.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

在 64 GB 的机器上，把下文 Q4 的数字当作起点：内存预算的重要性相同，而它放不下的部分都来自 SSD，
只是现在每个专家（Q6_K）和每行表数据（170 字节）读取更快。

## 引擎原先没有、现在有的东西
<a id="what-the-engine-did-not-have-and-has-now"></a>

UD-Q4_K_XL 的专家格式（gate/up Q4_K、down Q5_1）在所有路径里都有；Q6_K 的 gate/up 没有。本分支补上了它
（PLE 的 Q8_0 读取器是另一半）：

- **decode 窗口：** 分组的专家内核（`native_expert_grouped`）支持 gate/up Q6_K —— 通过 `Fmt<14>` 使用 llama.cpp
  `vec_dot_q6_K_q8_1` 的固定副本，以及反量化器（`dequantize_q6_K` 折进内核的 32 个线程）。按 token 的 MMVQ
  （`native_mmvq`）本来就有 Q6_K。
- **提示词路径：** llama.cpp 的 MMQ 提示词内核实例化了 Q6_K（K-quant 库，`STRATA_MMQ_KQ=1`；它会与
  q4_k/q5_k/q5_1 一起构建 `template-instances/mmq-instance-q6_k.cu`；`moe_mmq` 接受并分派它）。
- **准入判断：** `native_expert_supported` 接受 Q6_K/Q8_0 组合（像第 2 层那样的 gate/up Q8_0 原先就在列表里）——
  引擎此前会以 "no GPU kernels for" 拒绝整个模型包。

测试：`native_grouped_parity` 循环测试 Q6_K/Q8_0 组合；`prefill_mmq_kquant_test` 在 1280×2560 上把 Q6_K 的
gate/up 乘积与双精度参考对比；`native_expert_parity --synthetic q6_K/q8_0` 将 GPU 与 CPU 和 ggml 反量化后的权重
互相校验。三者都会构建（CUDA，`86;120`）。它们还没有在 GPU 上跑过：那块显卡当时正在服务下面这个文件。

## 状态：什么是已验证的，什么不是
<a id="status-what-is-verified-and-what-is-not"></a>

本机已验证（测试克隆，Linux，NVIDIA GPU，源码构建）：

- 六个文件的大小与 SHA-256（HF tree API；分片 1、2、6 另外对照下载的 blob 校验）。
- 分片 1、2 和 6 的张量清单，由头部读取（`tools/gguf_reader.py`）；分片 4 和 5 的张量数与每层专家类型，
  来自模型包的 `conversions.json` 与 `native_experts.txt` v4。
- Q8_0 的 PLE 行：`ple_reader_test --selftest` 跑 90、110 和 170 字节的行（全部 OK）；`ple_q8_parity` 在合成的
  Q8_0 表（`tools/ple_q8_fixture.py`）上，经 Mmap 与 Direct 两条路径与 ggml 的参考反量化器逐位一致
  （`max_abs 0.000e+00`）。在本分支加入 Q6_K 专家工作之后，两者仍然通过。
- 成品文件上的模型包（数字见上文），以及引擎二进制可构建（CUDA `86;120`，MMQ K-quant 开启）。
- Q6_K 专家来源支持：decode 侧的 MMVQ 本来就有；分组的 verify window 内核、MMQ 提示词内核与
  `native_expert_supported` 准入判断原先没有，已在本分支（上一节）。

服务器已运行过该文件。2026-10-04，Linux，一块 RTX PRO 4500 Blackwell（计算能力 12.0）加第二块 GPU、
一颗 Threadripper PRO 5975WX（AVX2，无 AVX-512）、30 个专家池 worker 和 499 GB 内存，引擎 0.1.39 加载了
101.75 GiB 专家（8.43 GiB/s）并完成了一次评审。冷读取 14,915 个 token，速度 3,176 tok/s。十四轮对话生成
5,799 个 token 用时 74.1 s（78.3 tok/s），结束时上下文为 44,160 个 token；草稿接受 3,485 / 4,406（79.1%）。
同一台机器在引擎 0.1.38（本分支尚未 rebase 到 0.1.39 之前）冷读 249,193 个 token，速度 2,802 tok/s，
随后以 69.0 tok/s 生成 1,079 个 token。该文件相对 llama.cpp 的质量尚未实测，因此本页保持实验性状态。
上述三个 Q6_K 一致性程序是剩余的 GPU 检查项。
