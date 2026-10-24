# EXPDEC：把 verify window 的路由专家层与该组的激活值放进共享内存（`STRATA_EXPDEC=1`，可选开启）
<a id="expdec-the-verify-windows-routed-experts-with-the-groups-activations-in-shared-memory-strata_expdec1-opt-in"></a>

可选开启：`STRATA_EXPDEC=1`。不设置（或设为 `0`）时，引擎启动的内核与之前完全一致。仅限 CUDA；输出与默认内核逐位相同（见下文“Bit for bit”）。

## 它改了什么
<a id="what-it-changes"></a>

`native_expert_grouped`（`src/kernels/cuda/iq_kernels.cu`）负责计算 verify window（校验窗口）中常驻显存的路由专家层（expert）：group g 是一个专家层的数据块，它的条目就是窗口中选中它的那些行（1 到 T，T 为该窗口的 token 数）。默认的 gate/up 内核（`native_gu_multi_kernel`）已经让每个权重部分在每个 block 里解码一次，并作用到该组的各个条目上（`row_dot_80_sub16<NC>`），但这一步会从全局内存重新读取每个条目的 q8_1 block，用 9 次分散的 4 字节载入（36 字节的 block：一个 warp 的载入会触及 16-32 个 sector）。在 RTX 2080 Ti 上做 131k 的 decode 时，这让每多一个条目多花约一整个专家层的 60%（约 1.6 us，而每个专家层每层约 2.6 us），尽管它并没有从 DRAM 读取任何东西。

`expdec_gu_kernel`（gate/up：n_embd 2560 下的 IQ2_XXS、IQ2_XS、IQ3_XXS、IQ3_S、IQ2_S）和 `expdec_down_kernel`（down：n_ff 640 下的 Q2_0；带 `STRATA_EXPDEC_D20=1` 的 IQ4_NL）改为：

- 把该组的条目（与默认实现一样，一次 4 条）搬进共享内存，布局使一次调用的 8 个 word 变成两次 16 字节载入且没有 bank conflict（block 的两半分处两个不同的 plane），fp16 的 block scale 已经是 float（`__low2float(ds)`，即 `apply` 算出的那个值）；对 IQ4_NL 则这些 word 保持在一起，一次调用读两次 8 字节载入；
- 每个 lane 计算 R 行（`STRATA_EXPDEC_R`，gate/up 取 1 或 2，默认 1；down 的 `STRATA_EXPDEC_DR` 取 1/2/4，默认 4）：激活值的 word 每 R 行只载入一次；
- 每个 block 处理 `rb` 行而不是 16 行（`STRATA_EXPDEC_RB` / `_DRB`；0 表示在启动的 block 数不低于 `STRATA_EXPDEC_MINBLK`（默认 800）的前提下取最大值），于是 staging 的开销被 35-140 KB 的权重摊薄；
- IQ3_XXS 和 IQ3_S 的码本也搬进共享内存（IQ2 的本来就在）；
- gate/up 使用 `__launch_bounds__(256, STRATA_EXPDEC_MINB)`，默认 4（最多 64 个寄存器，与默认内核一样每 SM 4 个 block；设为 `2` 则允许 128 个）。

列表之外的格式在该角色上仍用默认内核；`STRATA_OLD_IQ_MMVQ=1` 会关掉 EXPDEC，正如它会关掉 multi 内核一样。down 由 `STRATA_EXPDEC_DOWN=1` 控制：在引擎里，带 EXPDEC 的 Q2_0 down 比默认内核慢约 10%，尽管单独的微基准测出的是 0.78x（默认的 down 已经把该组的 h 行放在共享内存里），所以 EXPDEC 默认只接管 gate/up。

第一次让 `STRATA_EXPDEC=1` 遇到某个（角色，格式）时，引擎会写一行日志，例如 `strata expdec: CUDA1 gate/up type 21: expdec_gu_kernel R 1 MINB 4, 16 rows per block, grid 80 x 10 (cap 10 groups), captured into a graph`（或 `... not covered: the default kernel runs`）。

API（`include/strata/kernels/iq_kernels.hpp`）：`native_expert_set_expdec`、`native_expert_set_expdec_params` / `native_expert_expdec_params`（`NativeExpertExpdecParams`）、`native_expert_expdec_takes`。这些开关在启动时从环境变量读取；已捕获的图保留它们捕获的内核。

## 逐位相同
<a id="bit-for-bit"></a>

对每个（行，条目），求和结果都是默认内核的那棵树——gate/up 用 `row_dot_80_sub16`（16 lane 子 warp 的第 t 个 lane：把 t、t+32、t+64 累进 `s`，t+16、t+48 累进 `s16`，`s = __fadd_rn(s, s16)`，蝶形 xor 8-4-2-1），IQ4_NL 用 `row_dot_40_sub8`，Q2_0 用 `row_dot_20_sub16`：同一个 lane 以同样的顺序做同样的调用，`apply_u` 就是 `apply`，对相同的 word 以同样的顺序做同样的整数与浮点运算（只是 q8_1 block 从共享内存读取而不是全局内存）。该组的条目彼此独立，所以一次 stage 多少条并无影响；每 lane R 行和每 block `rb` 行只改变哪个 lane 算什么。`s += d * sumi` 的 FMA 合并，是两个内核里同一个编译器对同一表达式做出的同一决定。

## 在引擎中
<a id="in-the-engine"></a>

用 nsys 测 131k 的 decode（层切分：RTX 3060 层 0-15，RTX 2080 Ti 层 16-47 加 head，`--pipeline-windows 2`，Windows，96 个 verify window），每个窗口的内核时间中位数，T = 1 / 2 / 3 / 4：

| | 默认 | EXPDEC |
|---|---|---|
| RTX 2080 Ti，gate/up 内核 | 1.60 / 2.58 / 3.51 / 4.16 ms | 1.42 / 2.07 / 2.75 / 3.20 ms（-11 / -20 / -22 / -23%） |
| RTX 2080 Ti，gate/up 每行斜率 | +0.86 ms | +0.60 ms |
| RTX 3060，gate/up 内核 | 0.77 / 1.43 / 1.97 / 2.23 ms | 0.80 / 1.21 / 1.83 / 1.84 ms |
| RTX 2080 Ti，带 `STRATA_EXPDEC_DOWN=1` 的 Q2_0 down | - | 比默认慢约 10%（因此默认关闭） |

端到端，同一台机器，确定性 131k 基准（固定 token 与窗口大小，1024 个 token，B-A-B，开关按 开 / 关 / 开）：decode tokens/s **+1.1%**（两个 B 分支分别为 +0.0 和 +2.1，没有 B 低于 A），文本完全一致。专家内核约占总关键阶段时间的 20%，而 EXPDEC 又只占其中的约 20%，所以端到端的数字按设计就很小；这个开关开销低、逐位一致，且默认关闭。

这里没做：权重侧（`Split<TG>::load` 每次调用、每一行的 8-9 次两字节载入；把权重 tile 用 16 字节载入搬进共享内存是下一步），以及 T=1 时的固定开销。
