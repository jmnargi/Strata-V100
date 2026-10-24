# QSA_ROWS：把 decode 窗口的 QSA block 分数放进寄存器（`STRATA_QSA_ROWS=1`，可选开启）
<a id="qsa_rows-the-decode-windows-qsa-block-scores-with-the-queries-in-registers-strata_qsa_rows1-opt-in"></a>

可选开启：`STRATA_QSA_ROWS=1`（必须正好是 `1`）。不设置时，引擎运行它之前运行的内核。仅限 CUDA，作用于 verify window 的 decode（调用 `qsa_block_scores` 时不带 `active_blocks`、且查询数不超过 4）；提示词路径和 HIP 不受影响。分数和 top-k id 与 multi 内核逐位相同。

## 为什么
<a id="why"></a>

在 131k 的 decode 中，随窗口行数增长最多的 QSA 内核是 `block_scores_multi_kernel`：在 RTX 2080 Ti 和 RTX 3060 上，窗口每多一行，每次调用就多花约 20 us（nsys 测量；层切分时第二张卡有 8 个 QSA 层，第一张有 4 个）。key 已经为所有查询读过一次；每行的开销在共享内存和 shuffle 流水线上：对每个（key block，查询），内核从共享内存读四份查询切片（4 x `LDS.128`），并用各自的五次 shuffle 归约每个 head 的 lane 乘积（每个 block 和查询 20 次）：在 32 769 个 block 时，每个查询就是 15-20 us 的发射开销。

## 它改了什么
<a id="what-it-changes"></a>

`block_scores_rows_kernel<NQ>`（`src/kernels/cuda/qsa_select.cu`，NQ = 1..4）：

1. NQ 个查询的切片（每个 lane 4 个 head x 4 个 float）在整个内核运行期间留在寄存器里（44 / 63 / 78 / 94 个寄存器，没有栈，`cuobjdump -res-usage`）；
2. 4 个 head 的 lane 乘积用一种保持 `warp_sum` 配对方式的 scatter 归约：xor 16 和 8 之后每个 lane 只剩一个 head，然后 4、2、1——四个 head 只需六次 shuffle 而不是二十次——再按 head 顺序在 lane 0 上汇总（三次 shuffle）。

每一次加法加的都是同一个 `warp_sum` 层级的同样两个操作数（float 加法可交换），乘积表达式是同一文件里的同一段文本，relu 按 head 顺序求和：与 multi 内核和单查询内核逐位相同。尾块和 `+1e9` 与之前一样。超过 4 个查询、带 `active_blocks`、或不设该变量时，运行 multi 内核。

## 在引擎中
<a id="in-the-engine"></a>

层切分 RTX 3060（层 0-15）+ RTX 2080 Ti（层 16-47，head），Windows，`--pipeline-windows 2`，131k 上下文：每个窗口，第二张卡在 T=1 时省下 -0.09 ms、T=4 时省下 -0.37 ms（8 个 QSA 层），第一张卡省下 -0.04 / -0.25（4 层）。在确定性的 131k 基准上端到端（固定 token 与窗口大小，1024 个 token，B-A-B）：两个 B 分支的 decode tokens/s 分别是 **+3.2% 和 +0.1%**，文本完全一致。

## 没做的
<a id="not-done"></a>

多行 attention 内核（一个 cell 的 K/V 为选中它的所有行只读一次）只有在每一行都保留自己的 chunk 划分时才可能逐位一致；在 decode 窗口的规模下（上限 2051 个 cell，T <= 4），所有行的 block 一波就能装下，内核受其 block 的延迟链约束，所以在 8 层上每行的收益上限约为 0.1-0.2 ms：这里不值得做。`STRATA_QSA_ROWS` 只打开分数内核。
