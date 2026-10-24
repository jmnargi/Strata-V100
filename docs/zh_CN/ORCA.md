# OrcaRouter IQ3_XXS 兼容性
<a id="orcarouter-iq3_xxs-compatibility"></a>

2026-09-27 在 Linux 上用 RTX 5090（32 GB）、Ryzen 9 9950X3D 和 128 GB 内存验证。

目标是 `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` 的 **IQ3_XXS**（两个分片，共 85.20 GB）。
它采用 Strata 的 Qwen4Exp 几何结构，但普通量化也压缩了 hyperconnection、
小型 attention 和 PLE 投影，而 Strata 的内核按 BF16 读取这些投影。

`tools/iq_pack.py --compat-bf16` 只展开这些投影，并按 nearest-even 舍入到 BF16。
相对于 GGUF 的反量化权重，这会引入额外的舍入；它不会恢复原始 BF16
checkpoint 或其质量。专家层权重、原生 attention 权重、embedding 和 28.8 GB 的 PLE
查找表保持不变。后者仍留在磁盘上。模型包把转换记录在
`compat-bf16.json` 中。

原生加载器保留了这个模型包转换后的 PLE key。它原有的原生 Q2_0 key 路径仍然
对原模型可用。支持 Hugging Face snapshot 符号链接；所有分片都
必须在打包前下载完成。要传 snapshot 的文件名，而不是它按哈希命名的 blob 目标。

## 准备工作
<a id="preparation"></a>

按常规方式用固定的 llama.cpp 依赖构建 Strata。Python 需要
该依赖中的 numpy、regex 和 gguf-py（`STRATA_GGUF_PY` 可以指向它的 `gguf-py` 目录）。

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
  --out packs/orca-iq3_xxs --compat-bf16
```

使用导出到这个模型包里的、该模型自己的 tokenizer。不要共用其它模型的 `dense.bin`，
也不要把 Orca 的文件改名来冒充 setup 的某个 GSQ-RCO 模型。

Strata 的常驻服务器也使用 MTP 运行时（可选：不加 `--mtp` 时只按查找做草稿），由
现有的工具准备：

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

这里使用原模型的草稿头，与 setup 对 Swift 的做法相同。目标模型会校验它
提出的候选；草稿的接受率和性能必须针对这个微调模型单独测量。

## 本地服务器
<a id="local-server"></a>

把下面内容保存为仓库根目录的 `strata-orca-iq3_xxs.json`，两处 `/path/to/`
都替换成**同一个第一个分片**。Orca 的 PLE 表在分片 1 里。先从 32K 上下文和
512 token 的 prefill（预填充）分块开始；专家缓存会按可用显存自行确定大小。仅 expert arena 就
需要约 49.8 GiB 可用系统内存，另外还要加上草稿/运行时缓冲区和其它应用程序。

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/orca-iq3_xxs",
    "--native", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "/path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "32768", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/orca-iq3_xxs/tokenizer",
  "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
  "log": "strata-orca-iq3_xxs.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

在仓库根目录运行：

```sh
.venv/bin/python -m serve.server --engine strata --config strata-orca-iq3_xxs.json --port 8080
```

Web 界面在 `http://127.0.0.1:8080`；API 客户端使用 `http://127.0.0.1:8080/v1`。
这个纯文本示例没有配置视觉。原 GSQ-RCO 模型的性能
不能证明这个微调模型的速度或准确度。

## 范围与验证
<a id="scope-and-validation"></a>

- IQ3_XXS 是这项改动的目标；Orca 的其它量化版本未经验证。
- IQ3_M 还使用 Q5_0 的专家层 down 矩阵，原生 GPU 专家层路径不支持它。
- 安装程序的模型菜单没有变化。这是一套显式的本地打包流程。
- 针对性的转换与分片文件测试：`.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`。
- 全部八项打包测试、17 项服务器测试以及 GPU gated-residual 一致性检查均通过。
- 完整模型打包成功：转换了 460 个张量、1.39 GiB 的转换后 BF16 权重；
  打包后的稠密权重共 1.43 GiB。48 个专家层仍然是 IQ3_XXS 的 gate/up 和 IQ4_NL 的 down。
- 真实服务器检查通过：算术、Python 代码、翻译、一段较长的解释、多轮上下文
  复用（复用了 25 个提示词 token）、流式中断以及随后一次正确的回答。引擎
  PID 保持不变。这些是冒烟测试，不是质量基准，也不是全上下文压力测试。
- 使用上面的配置，一段 111 token 的解释以 77.7 token/s 生成（引擎 decode 计时）；
  请求总时间为 1.77 秒。这一次简短测量不构成对吞吐的普遍性声明。
