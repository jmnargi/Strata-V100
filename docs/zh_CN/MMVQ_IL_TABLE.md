# 交错式小批量 matvec（STRATA_MMVQ_IL）：如何为自己的卡测量一张表
<a id="the-interleaved-small-batch-matvec-strata_mmvq_il-how-to-measure-a-table-for-your-card"></a>

decode（解码）和 verify 窗口把它们的稠密投影（一次 2 到 8 个 token）交给 `native_mmvq_il` 运行，后者读取的是
q8_1 激活的一份交错副本。它做的每一个选择都与普通内核给出相同的位
（`STRATA_MMVQ_IL=0` 会关闭整条路径）；只有速度不同。一个 warp 取多少行矩阵
（1、2 或 4，或 0 = 使用普通内核）是 `src/kernels/cuda/native_mmvq.cu` 里的一张表，由
量化类型（IQ4_XS、Q4_K、Q5_K、Q6_K、Q3_K）、token 数（2-4）和矩阵的行数（五个类别）索引。
5 到 8 个 token 的窗口有第二张表（`kIlRows58_<cc>`，一个 warp 取 1 或 2 行）；目前只有 sm_120 有（在
RTX PRO 5000 上测得：例如 Q5_K 头在 5 个 token 时 899 -> 491 us），其他每块卡在这些窗口下都用普通内核。
`STRATA_MMVQ_IL_WIDE=0` 会关闭 5-8 token 路径。`--emit-table` 会打印两张表。

一块卡会得到哪张表：

| 卡 | 表 |
|---|---|
| sm_86（Ampere 消费级）以及 SM 少于 80 的 sm_120 卡（RTX 5070 及更小） | `kIlRowsShared`，在 RTX 3060 和一块 RTX 5070 上一起测得（只有当两块卡中较差的那块仍能获得 3% 时，该格才非零） |
| SM 为 80 或更多的 sm_120 卡（RTX PRO 5000 / 6000 Blackwell、RTX 5080 / 5090） | `kIlRows_120`，在 RTX PRO 5000 Blackwell（110 SM）上测得；`STRATA_MMVQ_IL_120=0` 会把共享表给它们 |
| sm_89（Ada）以及其他所有 sm_80 以上的卡 | 也是 `kIlRowsShared`。还没有人测过 sm_89（我们没有这样的卡）。 |
| sm_75 及更旧 | 该路径关闭；运行普通内核 |
| `kIlArch` 中列出的卡 | 它自己的表 |

## 测量 sm_89（或任何其他卡）
<a id="measuring-sm_89-or-any-other-card"></a>

1. 带测试构建：`cmake -S . -B build -DSTRATA_BUILD_TESTS=ON` 和 `cmake --build build --target mmvq_il_parity`。
2. 关掉 GPU 上其他一切，然后运行 `build/mmvq_il_parity --bench --emit-table`（几分钟）。
   它先检查每种形状和每个行数选择的逐位一致性（最后必须出现 `mmvq_il_parity: OK`），
   为每种形状和 token 数打印一行 `BENCH`，最后给出一个可直接粘贴的块，每种类型一行。
3. 把完整输出贴到 issue 上。维护者会把这个块粘贴为 `kIlRows_89[]`，并把
   `{89, kIlRows_89, sizeof(kIlRows_89) / sizeof(IlRows)}` 加入 `kIlArch`。
4. 想在不必重新构建的情况下试一张表，就设置 `STRATA_MMVQ_IL_ROWS="type:ncols:r0,r1,r2,r3,r4;..."`，例如
   `STRATA_MMVQ_IL_ROWS="23:3:0,0,2,4,4;12:2:0,1,1,1,1"`（五个数字分别是低于 2048、4096、8192、
   12288 以及以上的行类别；每个取 0、1、2 或 4）。然后对 decode 做 A/B：相同的标志，`STRATA_MMVQ_IL_ROWS` 设置或不设置，
   10 组交错的配对，取中位数。

生成规则：对每个格，如果最快的行数比普通内核的时间至少省下 3% 就采用它，否则为 0；同一类中
意见不一致的形状取 0；没有测得形状的类别复制它下面相邻的类别。
