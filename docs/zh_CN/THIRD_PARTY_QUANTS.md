# 第三方动态量化（不是 IQ4_NL 的 n-gram 表）
<a id="third-party-dynamic-quants-a-n-gram-table-that-is-not-iq4_nl"></a>

引擎的 PLE 读取器要求 `per_layer_token_embd.weight` 是 IQ4_NL
（`src/kernels/ngram.cpp`）；这种编码 90 B 的行跨度已经写进了行收集（row-gather）路径。发布者量化原始
checkpoint 时可能把这张表编码成别的格式——AtomicChat 的 AD 构建用的是 Q5_1（模型卡
注明测过 6 位和 8.5 位，KLD 差异可忽略）。这种构建里剩下的东西引擎都已接受：
动态 i-quant 专家层（expert）、Q8_0 的 dense 矩阵，以及打包时通过 `--compat-bf16` 处理的
Q8_0 小投影。只有这张表需要处理。

重新量化一次，然后照常打包：

    # keep the original shard under a name shard discovery cannot match
    mv <model>-00002-of-000NN.gguf <model>-00002-of-000NN.gguf.q5_1.bak
    python tools/requant_ple_iq4nl.py \
        --src <model>-00002-of-000NN.gguf.q5_1.bak \
        --out <model>-00002-of-000NN.gguf --workers 16

    python tools/iq_pack.py --gguf <model>-00001-of-000NN.gguf \
        --out <pack-dir> --compat-bf16

- 编码器是 ggml `quantize_row_iq4_nl_impl` 的忠实移植（`ntry`
  7，不用 quant_weights，走单 scale 分支），包括完全一致的 `best_index_int8`
  平局判定。输出经规范 IQ4_NL 解码器解码结果一致；运行时会打印相对
  Q5_1 源的 RMSE 和最大绝对误差，并用规范解码器重新校验写出的文件的
  首部/中部/尾部。
- 替换品保留源分片的 header，只改动 4 字节的张量
  类型字段，所以这张表仍然正好填满它的分片——引擎
  在加载时会断言这一点。
- 替换品必须保留原分片的文件名：引擎按文件名
  查找分片。
- `--rows N` 把运行限制在前 N 行（在整表之前做一次
  快速冒烟；在 AtomicChat AD-4.27bpw-Q4_K_M-M64 上全量跑
  320,001,536 行、16 个 worker 花了 79 分钟，RMSE 0.000578）。

在重新量化过的构建上，MTP 草稿接受率以及 prefill/decode 的表现
与表本身就是 IQ4_NL 的构建相同；接受率数字会随采样随机性波动，
单次请求不具代表性（见添加该工具的 PR 里的
基准说明）。
