# Strata 跑在两块或更多 GPU 上（层级切分）

<a id="strata-on-two-or-more-gpus-layer-split"></a>

一个模型可以跑在一台 PC 的多块 NVIDIA 卡上。层被切成连续区间，每块卡一段：
第一块卡跑第 0 到 K-1 层，下一块从 K 开始，以此类推；最后一块卡还跑输出头（head）
和草稿（MTP）层。每块卡只为**自己负责的层**维护 expert 缓存，所以两块卡
装的 expert 约是一块卡的两倍——对 16 GB + 24 GB 组合上的 Coder 模型来说，几乎全部
expert 都能装下，速度提升正来自这里（decode 于是几乎不碰 CPU 池）。

<a id="strata-on-two-or-more-gpus-layer-split-1"></a>

这是流水线（层级）并行，不是张量并行：一个 token 每个 verify 窗口只跨卡传递一次
（几百 KB 经由锁页内存），不是每层两次。不需要 NVLink 或点对点访问；插在 x4 或 x1
插槽的卡也能用，每块卡的 PCIe 份额在它自己的链路上单独探测。你给的 `--pcie-frac`
会被当作每块卡的份额并跳过这些探测；目前还没有按卡单独设置的选项。

<a id="two-models-on-two-cards-the-app"></a>

**两块卡跑两个模型（应用）。** 层级切分是把一个模型放到多块卡上。用两块卡的另一种方式是两个
模型各占一块卡、各自独立作答。在应用的 Models 页打开 **Run multiple models**：卡列表会标出
已加载模型占了哪块卡（"Already used by ..."）、为下一个模型预选一块空闲卡，并在启动撞到
已占用的卡时解释失败原因。在配备 Radeon AI PRO R9700 的 PC（四块用两块，Linux）上实测：
一块卡上的 Q2_0 与另一块上的 IQ3_XXS 都加载成功（文件已缓存时 33 秒和 59 秒，首次 114 秒），
都能回答聊天（各约 60 token/秒），Monitor 同时列出两者，卸载其一后另一个继续回答。此实验只用
AMD 构建跑过；同样的应用代码也驱动 NVIDIA 卡，但我们没有在两块 NVIDIA 卡上跑过两个模型。

<a id="one-make-of-card-per-engine"></a>

**每个引擎只驱动一种卡。** 一个引擎驱动一种卡（NVIDIA、AMD 或 Intel）。应用、`strata launch`
和 setup 只提供引擎能驱动的卡，绝不把模型切分到两种卡上，当另一种卡上有更大的卡时会用一句
大白话说明，并说明两种卡上同时跑两个模型不受支持。

## 使用方法

<a id="using-it"></a>

**什么都不用输。** `START-HERE.bat`（Linux：`./setup.sh`）列出你的 NVIDIA 卡并说明每块
Strata 能否使用：

```
  Your NVIDIA GPUs:
    GPU 0: NVIDIA GeForce RTX 5080, 16 GB VRAM - can be used
    GPU 1: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - not supported - older than the RTX 20 series (compute capability 6.1; Strata needs 7.5 or newer)
    GPU 2: NVIDIA GeForce RTX 3090, 24 GB VRAM - can be used
  ...
  1) RTX 3090 + RAM: fits: needs 44 GB of the 70 GB available   (recommended)
  2) RTX 3090 + RTX 5080 + RAM (layer split): fits: needs 44 GB of the 84 GB available
       prompts are read about 20% faster; the cards share the layers (docs/MULTI_GPU.md)
  3) RTX 5080 + RAM: fits: needs 44 GB of the 66 GB available
Which GPUs? [1]:
```

**模型用哪些卡（一条规则，贯穿 setup、`strata launch` 和应用）。** Strata 首先尝试把模型放到
你的**最大**卡加计算机内存上，即普通混合模式，装得下（且还剩一些 RAM）时就推荐它。只有装不进
一块卡、或者太挤时，才改推荐在最大的两块卡上做层级切分。每个选项都配有平实数字（模型需要多少、
卡和 RAM 能提供多少；装不下时要从 SSD 读多少），最终由你选择。参与计算的卡：不含核显
（它共享系统内存，在真实独显旁边不增加什么）、至少 4 GB 显存、共享一个模型则需 8 GB 以上。
算术逻辑即应用的 `fitOn`（strata_control/gpu_fit.py 保持其常量一致）；切分中每多一块卡估计
要消耗它自己 2 GB 的缓冲区。

<a id="which-cards-a-model-uses-one-rule-in-setup-strata-launch-and-the-app"></a>

**自己选**（setup 时或任意一次启动）：

```
--gpus 0,2                 these cards together, as nvidia-smi numbers them; the first is the main one. Remembered.
--gpus all                 every card that can share the model
--gpu 0                    one card (at a start: for that start only)
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

<a id="choosing-yourself"></a>

**不支持**（setup 会说明，并点名可替代的卡）：
- 比 RTX 20 系列更老的卡（算力低于 7.5：GTX 10 及更老），除非你点名它：Pascal 和 Volta 卡
  （Tesla P100 / P40、GTX 10、V100）在你用 `--gpus`（或 `--gpu`、`--cuda 12`）点名时会放行，
  然后跑实验性的 CUDA 12 引擎，见 [OLDER_GPUS.md](OLDER_GPUS.md)。两块 Tesla P40
  （配置里 `"gpu": [0, 1]`）在一个社区基准里跑过层级切分：IQ2_XS decode 单卡 19.6 tok/s、
  双卡 34.8（#1028，实验性，单份报告）；
- 与其他卡一起用时显存小于 8 GB 的卡（每块卡都保存一份 dense 权重副本和自己的 prompt
  缓冲区）——除非你用 `--gpus` 点名它：那时 setup 说明风险并询问（点名卡配 `--yes` 直接继续）；
- Intel GPU，以及 NVIDIA 和 AMD 卡混用。（AMD 卡之间可以共享一个模型：`./setup.sh --backend
  hip --gpus 1,0`，见 [AMD_HIP.md](AMD_HIP.md)。）

<a id="not-supported"></a>

或者编辑现有配置（`strata-*.json`）然后重启：

```json
"gpu": [0, 2],
"layer_split": "auto"
```

<a id="or-edit-an-existing-config-strata-json-then-restart"></a>

`"layer_split"` 取 `"auto"`（按每块卡的空闲 VRAM 放置）或**每块后续卡的第一层**：第一块之后
每块卡一个递增数字，不是每卡的层数。4 块卡、48 层模型时，`"24,36,42"`（或 `[24, 36, 42]`）
把第 0-23 层放第一块卡、24-35 第二块、36-41 第三块、42-47 最后一块。
服务器在启动前检查它并指出问题（0.1.39，#644）。

<a id="layer_split-is-auto-placed-by-each-cards-free-vram-or-the-first-layer-of-each-later-card"></a>

**`"auto"` 下的卡序**（NVIDIA，#1352）：更快的卡（多处理器数 × 最高频率）排在**最后**——最后的
一级要跑 head、草稿层和 verify，prompt 分块要等它（4070 Ti SUPER + 5060 Ti 读 6K prompt
一种顺序 50 秒、反过来 15 秒）。得分相差 5% 以内的卡（同型号两块）保持你写的顺序；手动切分也
保持。**as_given**：`"gpu_order": "as_given"` 在 `"auto"` 下也保持你写的顺序。引擎日志行
`layer split: card order ...` 会说明顺序是否被改过。

<a id="card-order-with-auto-nvidia-1352"></a>

**`"auto"` 下的不对等双卡（#2644）。** 搜索为 decode 窗口定价，它可能把几乎每一层都给评分更快
的卡。只有一两层的那块卡只装着几个 expert，还会卡住整条流水线的 prompt 分块（#1616），于是搜索
随后要求每块卡至少 3 层——但在 4060 Ti 16 GB + 3060 12 GB（48 层，IQ3_S）上结果是 K=3：3060 跑
3 层、12 GB 闲置，prefill 452 tok/s，而 3060 跑 28 层时是 802（报告者的 `"layer_split"`
12、18、24、28、32 扫描：582、693、762、802、803 tok/s；3060 的 360 GB/s 显存带宽比它的 SM
数量影响更大，对手 4060 Ti 是 288）。现在双卡时引擎首先要求一种放置：每块卡按自己的 expert
容纳空间至少拿到应得的层份额，日志行会说明是哪条规则选中了这个切分。在被人为制造成不对等的
2x RTX A4000 上实测（stage 0 定价为 0.57 速、少 3.7 GiB 空间，`STRATA_SPLIT_SPEED_SCALE=0.57`
加 `--vram-reserve-mib 4400`；14.5K token prompt，IQ3_XXS，5 对交错）：旧规则选 K=3（stage 0
跑 3 层），新规则 K=20；prompt 读取 945 -> 1,441 tok/s（+52%，5/5 对），decode 50.2 -> 65.9
tok/s（+31%，5/5）。对等卡和三块以上 GPU 不受影响。
在 x16 + x4 的机器上，链路缺失缩放（#2007，源自 4 倍差距）也会让 `auto` 偏离均衡放置
（2x RTX 3090，x16 + x4：miss 代价 ×4.20 时 K=31，decode 114-119 tok/s，K=27 时 130-136，
#2418）：`STRATA_SPLIT_UNIFORM=1`（需显式开启，配 `"gpu_order": "as_given"`）像 0.1.41 之前
那样对所有链路同等定价。prefill 对 prompt 长度不敏感（4K 到 16K 无增益）是有一块卡在闲置的
信号；**在不对等双卡上把 `"layer_split"` 扫一遍**——只测 4K 的基准看不见它。

<a id="an-unequal-pair-under-auto-2644"></a>

**第一块卡装得下就跳过切分**（需显式开启，0.1.31）：配置里 `"split_skip_if_fits": true`
（引擎参数 `--split-skip-if-fits`，配 `--layer-split auto`）时，若第一块卡装得下所有被 profile
的 expert 加上下文的 KV、草稿层和预留量，就只在它上面跑并在日志中说明；否则仍按切分跑。
在 R9700 32 GB + RX 9070 XT 上，R9700 装得下 Coder 的全部 expert：开此参数后 4K prompt 读
1,776 tok/s（切分时 1,244），decode 约 60 tok/s（切分时约 51；16K prompt 比切分慢约 5%）。

<a id="skip-the-split-when-the-first-card-holds-everything-opt-in-0131"></a>

**切分下的短 prompt（0.1.32，#340）。** 0.1.30 让每块卡的 prompt 路径从自己的 expert 缓存里
借款、每次请求后回补；在几乎装下全部 expert 的卡上，这让短 prompt 损失多达三分之一的速度。
0.1.32 一次性回补所有卡、在切分时用更小的流式环，并让有空闲 VRAM 的卡保留自己的 prompt
缓冲区——输出与 0.1.31 相同，在 R9700 + RX 9070 XT 上实测：2K prompt 993 -> 1,265 tok/s、
16K 1,852 -> 1,950，decode 不变。`STRATA_SPLIT_OWN=1`（需显式开启）让每块卡都用自己的缓冲区：
那里 2K 1,450、16K 2,227 tok/s，但装满的卡会常驻另一组 expert，输出与默认不同（稳定且连贯）；
`STRATA_SPLIT_OWN=auto` 只在缓冲区不超过每块卡 VRAM 12% 时才这样。

<a id="short-prompts-on-a-split-0132-340"></a>

**调切分下的流式 expert 环。** `STRATA_SPLIT_RING=N` 在引擎启动时把环设为 N 个 expert 槽位。
`STRATA_SPLIT_RING=0` 使用普通的环规则。它不是关掉流式。若设置了 `STRATA_PREFILL_RING`，它优先。
每个槽位装一个完整 expert blob。环在 prefill 期间从该级的 expert 缓存借用容量。更小的环让更
多 expert 常驻，但给拷贝流更少的前瞻。

<a id="tune-the-streamed-expert-ring-on-a-split"></a>

在两块 V100 16 GB（Gen3 x4 和 x16 链路）上用原生 IQ3_S 包、层级切分 16 测试：192 槽环在初测中
把 4K prefill 提升 4.68%，验证测试提升 5.39%（对比 384 槽）。32K prefill 不变。96 槽环让 32K
prefill 变差，DMA decode 也比自动机制慢。这些测试没有显示 decode 增益，也不能排除 decode
回退。环设置请保持需显式开启、只用于短 prompt 的 prefill。
测试时，私有运行时配置和引擎默认值未变；测试用的是单独的 IQ3_S 服务器。安装的 V100 部署现在在
其启动模板（`strata-v100.service`）里设 `STRATA_SPLIT_RING=192`；见
[V100 IQ3_S 服务器作为用户服务启动](DETAILS.md#starting-the-v100-iq3_s-server-as-a-user-service)。
它仍是需显式开启的短 prompt prefill 设置，不是通用默认值。
测量、正确性边界和复现命令见 [expert 传输报告](../../benchmarks/v100-iq3_s-expert-transfer-2026-10-06.md)。

<a id="on-two-v100-16-gb-cards-with-gen3-x4-and-x16-links"></a>

**短 prefill 分块的 CPU 辅助（需显式开启）。**

<a id="cpu-assistance-for-short-prefill-chunks-opt-in"></a>

设置 `STRATA_PREFILL_CPU_SHARE=auto` 让所有符合条件的层级切分页在 prefill 期间使用现有的 CPU
expert 池。填一个分数则选固定份额；不设或 `0` 保持 CPU 共享关闭。此选项需要原生 expert 格式、
锁页常驻 host blob、以及小于 1,024 token 的 prompt 分块。启用 batch 槽、`--no-pool` 或活动的
peer-expert 路径时不会启用。

<a id="set-strata_prefill_cpu_shareauto-to-let-all-eligible-layer-split-stages-use"></a>

重叠的 GPU 分页在一个池上串行执行完整的 CPU 批次。它们不会为每块 GPU 建工作线程池。
prefill 链排空后才恢复 decode。只读的锁页 expert 查找不改变源计数器和暂存状态。

<a id="overlapping-gpu-stages-serialize-complete-cpu-batches-on-one-pool"></a>

在两块 V100 16 GB 加 Ryzen 5 3600 上，三对交错测量：256 token 时 prefill 延迟中位数低 34.7%，
600 token 时低 37.4%。2K 对照组基本不变。请保持自动分块大小；强制 512 token 分块不是推荐的
长 prompt 优化。

<a id="on-two-v100-16-gb-cards-and-a-ryzen-5-3600-three-interleaved-pairs-measured"></a>

CPU 与 GPU 的激活格式和舍入不同。答案不必逐字节相同，自动共享也可能各次运行不同。默认保持
关闭。回归、重叠、取消和服务检查见 [测量与所有权报告](../../benchmarks/v100-multi-gpu-cpu-prefill-2026-10-07.md)。

<a id="cpu-and-gpu-activation-formats-and-rounding-differ"></a>

**保留的 CPU 池改动。** 池保留了模型格式、524,288 token 上下文下的 int8 KV 缓存、视觉、MTP、
L2/L3 对话分层以及 NVMe 支撑的 n-gram 表。尚未证明 20% 的端到端增益，也不附带任何性能保证。

<a id="retained-cpu-pool-changes"></a>

在有 AVX2、无 AVX-512、无 AVX-VNNI 的 CPU 上，AVX-2 expert 内核默认以 NT1 服务单 token
（NT1）的 IQ3_XXS、IQ3_S 和 IQ2_S 行。NT 指一组中的 token 数；NT1 是每行一个 token 的分组。
IQ3_S 和 IQ2_S 的 decode 用 32 位加载读取打包索引字节。这些行按 ggml 自己的 float 顺序归约，
归约之后才施加格式归一化因子，所以每一行与 ggml 的点积逐位相同，包括非规格化乘积。IQ4_NL 下行
不变。显式的 `STRATA_IQ_MT_MIN` 保持其规则。

<a id="on-a-cpu-with-avx2-no-avx-512-and-no-avx-vnni"></a>

在每个非 host 物理核都有一个 worker 之后，池会把额外的 worker 绑定到 worker 核所允许 SMT
兄弟超线程上。SMT（同时多线程）指物理核的第二个逻辑处理器。host 核的兄弟排最后，所以额外
worker 只有在所有 worker 兄弟都用上之后才共享 host 的物理核。默认放置不变。

<a id="after-each-non-host-physical-core-has-a-worker"></a>

**L3 后台写入器的亲和性。** 异步直接 L3 回退写入器在其线程开头恢复允许的 CPU。新线程会继承
服务线程绑定到保留 host 核（第一个物理核、逻辑处理器 0）的亲和。池在构造时（那次绑定之前）
捕获允许的 CPU；每个写入器稍后在已绑定的线程上创建，进入时捕获副本并在入口处恢复它。
GPU host 线程保持绑定。L2 暂存、快照格式、FNV 哈希、磁盘预算和后续写入的顺序不变。
两个 NVMe 对话文件上的 strace 冒烟测试：旧写入器继承掩码 [0] 且不恢复。每个新写入器在打开
文件前恢复 [0..11]。在实测机（六核 12 线程）上每次恢复都返回 0。RAM A/B/A 复用两臂都是
4,620 个 token 中复用 4,615。trace 不是计时证据。匹配的 HTTP 研究未达到 20% 目标。
见 [CPU 与 SSD 研究](../../benchmarks/v100-iq3_s-prefill-wmma.md#cpu-and-ssd-study-2026-10-10)。

<a id="l3-background-writer-affinity"></a>

**空闲卡可以帮忙处理单分块 prompt（需显式开启，`STRATA_PREFILL_HELP=1`）。** 一个分块装得下的
prompt 会逐卡依次跑：一块卡读它的层时另一块闲着。开启后，每个分页把自己一部分流式 expert
交给空闲卡：空闲卡经由自己的 PCIe 链路把它们流进自己的（借出的）prompt 缓冲区，在 MMQ 路径上
算出它们的行再发回——即 `--peer-device` 的 peer 流式，但不需要 P2P（激活值和行经由映射的 host
内存，用拷贝内核读取，所以不会在两块卡的拷贝引擎上排在 expert blob 后面）。份额随 prompt 变长而
下降（1.5K token 时占流式 expert 的 0.41，3K 时 0.32），约 3.3K token 起关闭——那里分页本来就
重叠，没有共享可付。在 2x RTX 3090（UD-Q4_K_XL，无 P2P）上实测 prompt tok/s（关 / 开）：
1.1K 496 / 723、1.5K 674 / 833、2K 878 / 1,128、2.5K 1,017 / 1,296、3K 1,260 / 1,440；
4K 和 8K 不变，decode 不变。它不耗 VRAM（用空闲卡自己的 prompt 缓冲区），每 token 最大分块
约占 ~110 KB 映射 host 内存（3.3K token 时约 360 MB）。它计算行的舍入像另一种 MMQ 分组，
所以输出与默认的不逐位相同（但可重复：同 prompt 同输出），这就是它**需显式开启**的原因：
`STRATA_PREFILL_HELP=1` 打开。在本 fork 的两块 V100 上此选项没有测到有意义的 prefill 增益，
那里保持关闭。仅限走 MMQ prompt 路径的原生包（不与融合 prompt 内核 `STRATA_PF_FUSED=1`
同用，也不与 `--peer-device` 同用）。开启后 `STRATA_PREFILL_HELP_FRAC=f` 固定份额。

<a id="the-idle-card-can-help-one-chunk-prompts-opt-in-strata_prefill_help1"></a>

背后的引擎参数：`--layer-split K1[,K2..]|auto` 和 `--split-device D1[,D2..]`（后续分页的设备；
默认为接下来的可见卡）。`--layer-split K --split-device 0` 把两个分页都跑在一块卡上共享一切——
这是对手递做逐位一致校验用的，不是速度模式。

<a id="the-engine-flags-behind-it"></a>

**切分下批量处理草稿层的 prompt K/V（需显式开启，`STRATA_SPLIT_MTP_BATCH=1`）。** 用 `--mtp`
时草稿层位于最后一块卡，每个 prompt 分块以填满该层此分块各行的 K/V 收尾。单卡时 prompt 路径
批量做这件事（`Prefill::draft_kv`，每分块几次大矩阵乘）；切分时草稿器自己的遍历总是逐组做——
每 8 行一次图启动和四次设备拷贝（8,192 token 分块就是 1,024 组）。设置该变量后，最后那块卡的
prompt 路径像单卡那样批量化。仅重启生效；默认关闭，默认启动不变。它需要 `--native` 包（GGUF
形式的 token 表）和分页或环形（`--kv-resident`）的草稿 K/V（`STRATA_MTP_BATCH_RING=0`
把环退回旧遍历）；不满足时说明一次原因并保持旧遍历，日志为
`strata serve: draft layer prompt K/V batched on CUDA<d> ...`，
或 `strata serve: STRATA_SPLIT_MTP_BATCH=1 declined, the drafter's own pass runs: <reason>`。草稿层的 K/V 出自 Q8_1 x Q8_0 MMQ 而不是它自己的 mmvq，所以草稿
以及被接受的条数可能变化；目标的 token 由 verify 窗口决定。
`tools/split_mtp_batch_parity.py` 比较两臂的贪心文本。在一台机器上实测（2x RTX 3080 20 GB
限 220 W，Xeon E5-2696 v4，UD-Q4_K_XL，`--layer-split 23`，`--batch-mtp`，两槽，
`--prefill auto:16384`），每臂两次完整重启，顺序 off、on、off、on，各臂同一二进制：25K token
时 prompt 读取的合并中位数为关 1,996、开 2,129 tok/s（+6.7%），51K 2,448 / 2,553（+4.3%），
104K 2,846 / 2,940（+3.3%）；8 个开/关中位数比全部大于 1（1.03 到 1.08），两次 off 重启在
25K 相差至多 2%、其他尺寸 0.3%。host 端"每分块之后"的每请求耗时从 0.9-1.3 秒降到 0.1-0.25 秒
（每臂一次重启）。贪心输出（`--adapt-every 0 --suffix-draft 0`，五个 256 token prompt 加一个
25.5K token prompt）在全部六次比较中都 6/6 一致，包括同进程内两次运行（A/A）；每次运行接受
草稿 1,210 中 826，未见接受率差异。decode 未证明不变：设了该变量的中位数约低 3%（合并的
solo 82.6 -> 80.2 tok/s，双流 93.2 -> 90.8），而同配置重启两次也会漂移至多 4.7%，每臂两次
重启既不能证实也不能排除；此改动只涉及 prompt 路径。

<a id="the-draft-layers-prompt-kv-in-batches-on-a-split-opt-in-strata_split_mtp_batch1"></a>

**每块卡只加载自己那些层的 dense 权重**（0.1.39，PR #639），需要显式切分点（`--layer-split
27`，不是 `auto`）：过去每块卡都保留完整副本（Coder 约 3.4 GB），尽管它的分页只读自己的层，
释放的 VRAM 用于该卡的 expert 缓存（2x MI50 16 GB，Coder：VRAM 内 8,819 -> 10,626 个
expert，decode 39.2 -> 41.7 tok/s）。在本 fork 中，显式 `--layer-split` 的修剪是自动的：
显式切分的切分点在权重加载前就已知。`--layer-split auto` 保留完整表——它的放置要等切分搜索
之后才确定——upstream 的 `--trim-stage-weights` / `STRATA_STAGE_TRIM` 开关已移除。
一块卡装更多 expert 可能改变哪些 expert 跑在 GPU 上，所以输出可能与不带此行为略有不同。

<a id="each-card-loads-only-its-own-layers-dense-weights-0139-pr-639"></a>

**常驻 RAM 模式在切分上工作**（`--resident-experts`，低 RAM 模式）。expert 的 RAM 副本
省略每块卡缓存都装的（不只是第一块卡的），当剩下的装不下整份时按所有层的 expert profile
保留最热的。自适应交换会把被逐出的 expert 从拥有其层的卡拷回 RAM。此前，`--resident-experts`
配切分会以 `--mmap-experts` 运行，且 setup 在低 RAM 模式下只推荐一块卡；对有此能力的引擎
（setup.py 里的 `RESIDENT_SPLIT_ENGINE`），setup 让卡保持在一起，并把没有任何卡装的 expert
放进 RAM。同样一份副本、按 RAM 预算定大小（`--resident-budget-gib`，Unsloth UD-Q4_K_XL 和
UD-IQ4_XS 配置），在切分上也同样保留（[UNSLOTH_Q4.md](UNSLOTH_Q4.md)）。Swift 1.5 IQ3_XXS
@160K（q4_0 KV，`--prefill 4096`，`--spec 4` 配原版草稿层），RTX 4060 Ti（层 0-19）+
RTX 5080（20-47），i9-14900KF，32 GB RAM，Windows 11，同时四个贪心 prompt，decode tok/s：

<a id="the-resident-ram-mode-works-on-a-split-resident-experts-the-low-ram-mode"></a>

| | 前四个 prompt | 再过三轮之后 |
|---|---|---|
| 5080 单卡，`--resident-experts` | 25 | 29 |
| 切分，`--mmap-experts`（`--resident-experts` 在切分上变成的样子） | 32 | 64 |
| 切分，`--resident-experts`（22 GiB expert 锁在 RAM） | 71 | 69 |

切分加 `--mmap-experts` 在操作系统文件缓存装下 expert 之后能追平，前提是一台不跑别的东西的
PC；常驻副本从第一个请求起就在那里，其他程序需要 RAM 时它保持锁定。配 `--pcie-frac 0
--adapt-every 0`，切分的贪心输出两种模式相同。

<a id="the-split-with-mmap-experts-catches-up-once-the-os-file-cache"></a>

**切分时 prompt 路径的借款与其流式环（`STRATA_SPLIT_RING`）。** 切分时 prompt 路径把每块卡
expert 缓存的尾部槽位借来做它的缓冲区（prompt 结束后归还），常驻 RAM 副本也保留这些 expert，
所以被借的 expert 在 prompt 期间不会从模型文件回读。借多少槽取决于 prompt 路径的流式环：当至少
75% 的（层，expert）对落在某块卡的缓存里时切分用 96 个环槽，`STRATA_SPLIT_RING=N` 设为 N 槽
（`0` 回到锁页份额规则）。过去环是在 RAM 副本的区域定完大小之后才选的，于是区域是按与 prompt
路径实际所用不同的环定的：`STRATA_SPLIT_RING=384` 时它借 5,396 槽而副本只保留 4,662，另外 734
个每个分块都要从模型文件读。现在先选环（`Prefill::set_ring_override` 说明它必须在缓冲区计数
之前设置）。启动行 `strata serve: lend sizing: the prompt path borrows N slots (ring R slots
at chunk C), K of them keep their experts in RAM too` 显示两个数字，K 低于 N 时跟着一条
`WARNING`。没有切分、或没有 `--resident-experts` 时一切不变。有切分时，两个步骤的先后对每种环
选择都变了，96 槽规则也包括在内：它的区域过去按更大的默认环定大小，所以那里副本保留的槽可能
比借出的多（未实测）。5,396 / 4,662 的数字在测试机上测得（2x RTX 3080 20 GB 限 220 W，Xeon
E5-2696 v4，UD-Q4_K_XL，`--layer-split 23`，常驻 RAM 模式，`--prefill auto:16384`，2 槽），
每臂一次重启，`STRATA_SPLIT_RING=384`、`STRATA_PREFILL_RING` 未设。改动前一个 104K token
prompt 每请求产生 413 次 blob 读取、从模型文件读 10.3 GB expert；改动后为 0，全部 5,396 个
被借槽都保留在 RAM 中，代价是多锁 2.15 GiB RAM（13.61 -> 15.76 GiB）。25K 和 51K prompt
两种情况下都不从文件读。不设覆盖变量、`STRATA_PREFILL_RING=384` 时，借贷行前后相同
（5,396 / 5,396，15.76 GiB）。没有显示速度增益：三次 104K 读取的中位数前 2,803、后 2,907
tok/s，但改动前的散布是 2,540-2,897（其第一次读取从 NVMe 缺页读入 4.2 GB）且没有 A/A 重启。
这里页缓存吸收了每请求 10 GB；页缓存余量更小时它就是磁盘流量。未实测：两个环变量都不设。

<a id="the-prompt-paths-loan-on-a-split-and-its-streamed-ring-strata_split_ring"></a>

**给后续卡单独设 VRAM 预留：** `--vram-reserve-later-mib N`（默认：`--vram-reserve-mib` 的值）。
驱动显示器的卡比不驱动的卡需要更多余量；显示接在最后一块卡上时，`--vram-reserve-mib 300
--vram-reserve-later-mib 1800` 就把那份 VRAM 留给第一块卡的缓存。

<a id="a-separate-vram-reserve-for-the-later-cards"></a>

**auto** 尝试每一种放置（两块、三块以及自 0.1.40 起的四块全枚举；更多卡时按卡速比例分层）并保留
预测 decode 窗口时间最低的那个：每块卡的每层时间（SM 更多、频率更高的卡更低）加上其缓存会漏掉的
路由质量，最热 profile 对权重最大。这是 decode 代价模型，不是 prompt 带宽模型。启动日志打印选择：

<a id="auto-tries-every-placement"></a>

```
strata generate: layer split auto: K=22 - predicted 19.7 ms per decode window; the caches hold 11298 of 12288 profiled pairs (~99.7% of the routed mass)
strata serve: layer split: layers 0-21 (CUDA0), 22-47 (CUDA1), one hand-off per window
```

**每块卡的 PCIe 份额：** 没有 `--pcie-frac` 时每块卡探测自己的 host-to-device 链路。较慢的链路
为漏掉的 expert 拿到较小份额。显式启动的 `--pcie-frac` 值适用于每块卡。请求的 `pcie_frac` 值
只有在与第一块卡的启动份额不同时才覆盖每个分页。否则后续卡保持自己的启动份额。

<a id="pcie-share-per-card"></a>

## 每块卡持有什么

<a id="what-each-card-holds"></a>

- **每块卡**：其**自己**那些层的规范权重和原生 dense 投影（显式切分只加载这些；auto 放置保留
  完整表）、自己的会话状态（整个上下文的 KV 缓存）、自己的 verify 窗口和 prompt 路径缓冲区，
  以及一个按 profile 填充的、只服务其层的 expert 缓存；
- **最后一块卡**：还持有输出头和草稿层（约 0.8 GB）；
- **host RAM**：一份 expert 区（arena），所有卡共享（CPU 池计算没有任何卡装的 expert）。

prompt 按分块读取、依次流过各卡；后面的卡读分块 c 时，第一块卡已在读分块 c+1。（0.1.44，AMD
gfx12：切分上的自动分块保持 8192，而不是单卡的 16384：16K prompt 于是变成两个重叠分块而非
一个，32K 变四个而非两个。token 相同；2x / 3x / 4x R9700 在 16K 的 prompt tok/s +36% / +45%
/ +53%，32K +18% / +21% / +29%，64K +4% / +12% / +17%，8K 不变。`STRATA_SPLIT_CHUNK16K=1`
保持 16384。）对话检查点保存并恢复每块卡的状态；自适应 expert 交换拷入拥有该层的那块卡。

<a id="prompts-are-read-in-chunks-that-flow-through-the-cards-in-turn"></a>

需显式开启的对话缓存磁盘存储（`--conversation-cache-disk`）把这些状态保存进文件：每条对话一条
记录，每个分页一张图片。磁盘和 host-RAM 缓存都支持 `--layer-split`。

<a id="the-opt-in-conversation-cache-disk-store"></a>

从磁盘恢复的、经过校验的共享根可以把所有分页的镜像保留在有界的 RAM 缓存中。每个分页在恢复前
分别校验。草稿 K/V 随最后一个分页。之后的 RAM 恢复会为另一条聊天保留锁定的根。RAM 预算、
槽位上限和物理 RAM 下限依然适用。

<a id="a-validated-shared-root-restored-from-disk"></a>

一条退出的对话先进入 L2。成功的 L2 暂存不写磁盘。L2 需要空间时，引擎把被逐出者的完整分页镜像
移到 L3，不再二次 GPU 回读、也不复制负载。普通对话之后才外溢共享根。显式的共享前缀检查点钉
保持受保护。L2 被禁用或无法接纳快照时，仍可直接暂存到 L3。仅 L2 的快照不能跨服务重启存活。
实测的 [双 V100 共享根研究](../../benchmarks/v100-iq3_s-prefill-wmma.md#agentic-short-turn-latency-2026-10-10)
给出了增益和不改善的情形。

<a id="an-outgoing-conversation-first-enters-l2"></a>

## 重叠式交接（`STRATA_SPLIT_OVERLAP=1`，需显式开启）

<a id="overlapped-hand-off-strata_split_overlap1-opt-in"></a>

默认每个 verify 窗口把各分页依次跑：host 同步第 1 分页，然后暂存并启动第 2 分页的图，如此下去。
`STRATA_SPLIT_OVERLAP=1`（仅当分页在不同卡上）给交接一个就绪标志：较早分页的 GPU 用 volatile
store 写交接并置标志（`handoff_publish`），下一个分页的图在较早分页仍在跑时由辅助线程暂存并
启动，并在其 GPU 上等标志（`wait_flag_ge`），host 则直接从服务一个分页的层跳到下一个，只在
整条链结束后同步较早分页一次。它把每个窗口一次流同步和一次图启动移出关键路径。批处理窗口
（多条对话）保持串行顺序。它与 `--pipeline-windows` 并存时关闭，后者的循环以自己的方式重叠分页。

<a id="by-default-each-verify-window-runs-the-stages-one-after-another"></a>

在 2x RX 7900 XTX（gfx1100，ROCm，Linux）、Ryzen 9 9900X、52 GB RAM、Swift 1.5 IQ3_XXS、
128K 上下文、8 位 KV、层 0-25 / 26-47 上实测；512 token 贪心回答，8 次运行的中位数：

<a id="measured-on-2x-rx-7900-xtx-gfx1100-rocm-linux"></a>

| | 串行（默认） | `STRATA_SPLIT_OVERLAP=1` | 单卡 |
|---|---|---|---|
| decode，代码 prompt | 53 tok/s | 70.1 tok/s | ~70 tok/s |
| decode，散文 prompt | 45 tok/s | 55.9 tok/s | ~68 tok/s |

标志取 0 和 1 时贪心输出逐字节相同；测验集得分相同（47/51）。NVIDIA 上尚未实测，那边一次同步
和一次图启动的开销更小。

<a id="greedy-output-is-byte-identical"></a>

读取分页在 GPU 上等标志是有上限的：`STRATA_SPLIT_WAIT_MS`（默认 30000，0 = 无上限）。
若标志超时未到，等待会标记一个错误字并返回，窗口的图完成，host 以点名该设置的错误终止该窗口
（服务器随后重启引擎），而不是使用过期的交接。通常 host 自己的每层等待（20 秒，#267）会先
结束停滞的窗口并释放所有 GPU 等待，包括交接的那个。测试钩子：`STRATA_TEST_HANDOFF_DROP=N`
在第 N 个窗口扣住第一个分页的标志，就像发布丢失一样。

<a id="the-reading-stages-wait-for-the-flag-is-bounded-on-the-gpu"></a>

## 为流水线定大小的 prompt 分块（`--prefill-pipe`，需显式开启；setup 的 `--calibrate` 会设置它）

<a id="prompt-chunks-sized-for-the-pipeline--prefill-pipe-opt-in"></a>

`--prefill auto` 取缓冲区允许的最大分块（最多 8192），这正是单卡想要的：一个分块的开销是对
其各层 expert 的一次固定遍历（b：不在 VRAM 时要反量化或经 PCIe 流式）加每 token 一份（a），
所以少而大的分块最便宜。层级切分把 prompt 当流水线读——分页 s 读分块 c 时分页 s+1 在读分块
c-1——于是 S 个分页、分块 C_i 大约需要 sum_i (b + a C_i) + (S - 1) max_i (b + a C_i)：大块或
不均的分块让后面的卡干等（10.7K token prompt 分成 7680 和 3011 两块、四个分页：第一分页扛了
36 秒中的 24 秒），小分块则一遍又一遍付 b。

<a id="prefill-auto-takes-the-largest-chunk-the-buffers-allow"></a>

`--prefill-pipe R` 按 b/a = R 时流水线时间最短的分块数读每个 prompt，前提是它比今天的切分好
3% 以上：从缓冲区允许的最少块数起逐个数尝试，每块在 256-token 网格上取整（至少 512，至多
缓冲区的分块；请保留 `--prefill auto`，让长 prompt 仍能用大块）。b/a 在不同机器之间差一个
数量级——四颗 GP100 核心（每个 expert 都在 VRAM、算术慢）约 700 token，两块 RTX 3090 Ti
（每块分块流式部分 expert、算术快）约 5000——选错要付代价：固定 1024 在两块 RTX 3090 Ti 上
慢 19-24%，在两块 RTX 4090 上慢 22-36%（PR 中由 MistyMoonR 报告）。所以它既不靠猜，也不从
引擎实际服务的 prompt 学习（那会让一个 prompt 的分块、从而其输出位，取决于 PC 当时多忙）：
和 PCIe 份额、草稿下限一样，它是机器的一个设置，由 setup 的 `--calibrate` 测量一次
（`tools/calibrate.py`：在层级切分上，用 2K 和 8K token prompt 的读取速度——每次同一文本，
加一段自己的头部以免复用——对比关闭时的分块、在 256 网格取整以及 b/a 768、1536、3072、6144；
只有当它在三轮配对中每轮都快于关闭、且中位数快 3% 以上时才保留最优值）并写进配置。实测结果：
四颗 GP100 核心保留 b/a 768（每轮 355-356 对 219-220 tok/s，+62%）；两块 RTX 3090 Ti 保持
关闭（没有任何选项快于关闭，b/a 768 慢 20%）。一个早期版本对每个取值读不同的文本段、用两轮
确认，在 RTX 3090 Ti 双卡上凭一次快速读取选了 1536——在流式 expert 的机器上，读取速度取决于
文本。`--prefill-pipe 1` 不需要 b/a：它就是 `STRATA_PREFILL_EQUAL=1` 的规则（#693）——今天的
分块数在 256-token 网格上取整，且只在今天的最后一块有 1024 token 以上时（设了
`STRATA_PREFILL_CPU_SHARE` 则是 3072）。这么大的分块会把其卡没有装下的每个 expert 都流一遍，
更小的只流它路由到的 expert，所以刚好超过分块大小的 prompt 会读成一个满块加一条几乎免费的
尾巴，把它取整均匀反而多付一次 expert 全遍历：第一版对所有切分都取整，在两块 RTX 4090 上
MistyMoonR 的校准读到 1,601 tok/s，而关闭时是 2,129。0（默认）就是今天的切分；
配置装不下该参数时 `STRATA_PREFILL_PIPE=<R>` 与参数等效。分块几何改变舍入，这就是它需显式
开启的原因。

<a id="prefill-pipe-r-reads-each-prompt-in-the-chunk-count"></a>

一个 prompt 段（两次切分之间的 prompt：系统 prompt 根、最后一轮、某个 `pin=`）只按其完整长度
和 R 规划一次，之后每次运行都按那次定的分块读取。所以它的分块、以及为其缓冲区布局的借款，无论
是一次读完、还是在 `--batch` 槽解码的窗口之间分段读、还是从其中捕获的 `BYIELD` 之后或周期性
检查点续读，都保持一致（槽和检查点保存规划；续读同一段就继续用它，否则重新规划并在日志中说明）。
输出位于是在各卡所装 expert 一致的范围内与串行循环相同（要逐字节一致的重放仍用 `--adapt-swaps 0`
同不带流水线时一样）。在这之前，紧贴解码槽的读取会为每一小段重新规划，`BYIELD` 后续读的会对
剩余部分重新规划。流水线规划的分块不接受 `STRATA_PREFILL_CPU_SHARE`：各分页同时读这种分块，
哪个分页先拿到 CPU 池将决定 CPU 计算哪些 expert。DONE 行以 `chunk=`（请求最长批处理段的第一个
分块）和 `experts_streamed=`（其 prompt 拷到各卡的 expert blob 数）结尾，INFO 行带有
`prefill_cap`、`prefill_stages`、`stream_all_min` 和 `prefill_pipe`。日志中有一行说明某个
规划的选择：

<a id="a-prompt-segment-the-prompt-between-two-of-its-cuts"></a>

    strata prefill: 10058 tokens in 1280-token chunks over 4 stages (--prefill-pipe 706: b/a)

四颗 GP100 核心（PH402 SKU 200，`--layer-split 12,24,36`，Flash-Next IQ3_XXS，64K 上下文），
prompt tok/s，全新 prompt，两轮交错（每轮内差 0.2% 以内），b/a 1024（与 700-890 的分块相同，
在该机器上测得）：

<a id="four-gp100-dies-ph402-sku-200"></a>

| Prompt tokens | `--prefill auto` | `--prefill 2048` | `--prefill auto` + `--prefill-pipe 1024`（chunk） |
| ---: | ---: | ---: | ---: |
| 2,013 | 180.9 | 181.4 | 228.0 (1024) |
| 3,865 | 207.6 | 276.5 | 312.4 (1024) |
| 10,041 | 261.1 | 433.0 | 432.6 (2048) |
| 28,998 | 475.7 | 554.1 | 559.7 (3328) |

两块 RTX 3090 Ti（为 sm_86 编译的 CUDA 12.9 构建，两个分页，Swift 1.5 IQ3_XXS，1M 上下文，
24,576 个 expert 中 13,371 个在 VRAM），prompt tok/s，全新 prompt，每臂两次启动，每次读四段
prompt 两遍（释放卡之后的第一次冷读排除在外）；b/a 5836 和 6432（在机器上计时），对比 1024：

<a id="two-rtx-3090-ti-cuda-129-build-for-sm_86"></a>

| Prompt tokens | `--prefill auto` | + b/a 5836 / 6432（chunk） | 对比：b/a 1024 |
| ---: | ---: | ---: | ---: |
| 2,035 | 694-815 | 698-816（今天的） | 560-562 |
| 3,887 | 1,127-1,207 | 1,115-1,227（今天的） | 880-885 |
| 10,063 | 1,437-1,553 | 1,654-1,681 (2 x 5120) | 1,109-1,113 |
| 29,019 | 2,532-2,603 | 2,566-2,620（今天的） | 2,064-2,089 |

## 限制（暂时）

<a id="limits-for-now"></a>

- **跨卡可用**（bench/results/2026-09-29-layer-split-limits）：
  - 图像（`--vision`）：每块卡保存自己的图像位置表；
  - 控制向量与实验性速度投影：每块卡持有向量的表，请求时在所有卡上同时开或关；
  - KV 流式（`--kv-resident`）：每块卡流式自己会话的 KV；
  - 对话缓存磁盘存储（`--conversation-cache-disk`）：每条对话一条记录、每个分页一张图片；
    只有层级切分、设备和其余兼容性输入都匹配时，一次运行才复用一条记录；
  - MTP 草稿层自己的 K/V：设了窗口且小于上下文时（`--mtp-window`；默认 32,768 个 cell；
    窗口为 0 或不小于上下文时草稿层 K/V 完全留在 VRAM），草稿层 K/V 是一个该窗口大小的环，
    建在它自己的锁页 host 副本之上，独立于 `--kv-resident`。host 副本在 RAM 里装**整个**
    上下文（它是恢复时回填环的基础；任何部分都不会缩减）。host 副本无法锁页时，草稿层 K/V
    完全留在 VRAM。带环时草稿只注意最近窗口的 cell——更大的窗口只多占 VRAM 就能装下更多
    上下文；无论窗口多大，锁页 host 副本总是装整个上下文；
  - 中途 prompt 检查点（`--prompt-cache-every`）：每块卡读完那个分块时保存它那份检查点；
  - 更早的辅助 GPU 缓存（`--expert-cache-remote`，docs/SECOND_GPU.md）：它们使用没有任何
    分页运行的可见 GPU，只装各分页缓存都不装的 expert。在测试机上，2080 Ti 助手让解码变慢，
    和不切分时一样：它每层往返的代价超过那些 expert 在 CPU 池上的开销。
- `--mmap-experts` 需要规范包（`experts.bin`），切分与否都一样；原生（IQ）包启动时会说明。
- 切分时 prompt 路径在每块卡都有自己的缓冲区（默认 2048-token 分块下每块 1.5 GB；
  `--prefill 1024` 减半），而不是像单卡那样借缓存槽位。第一块卡上显式的 `--expert-cache`
  会被封顶以留出空间。
- WDDM 下（Windows 以及 WSL2）expert 区只有 8 GiB 被锁页（更多、映射进两个 GPU 上下文会让
  WDDM 拒绝分配）；其余通过锁页暂存环流式传输。Linux 驱动没有这个限制，那里整个区域都锁页
  （自 0.1.31 起；这个上限曾让一块 4090 + 3060 的切分损失三分之二的 prompt 速度，#253）。
  `STRATA_ARENA_PIN_GIB=N` 在任何 OS 上最多锁 N GiB，`0` 锁整个区域。
- 每块卡都需要算力 7.5（RTX 20 或更新）。pre-sm_80 的 QSA 打分路径是 fp32 FMA，所以 Turing
  卡跑相同的内核，而不是 tensor-core 的 prompt attention。

<a id="-works-across-cards-benchresults2026-09-29-layer-split-limits"></a>

- 有 expert profile（默认）时，每个分页的 prompt 路径把自己的 expert 缓存尾部借来做它的分块
  缓冲区，prompt 结束后回补；prompt 之外整个缓存又都是 expert 缓存，所以缓冲区不占分页的
  永久 VRAM。没有 profile、或带 `--no-prefill-borrow` 时，每个分页整个会话都保留自己的
  分块大小缓冲区；那时显式的 `--expert-cache` 会被封顶以留出空间。
- 每个分页在给缓存定大小前为自己的 verify 窗口留出空间（约 96 MiB）；草稿层和 head（约 1 GiB）
  只放在最后一个分页。
- WDDM 下（Windows 以及 WSL2）expert 区只有 8 GiB 被锁页（更多、映射进两个 GPU 上下文会让
  WDDM 拒绝分配）；其余通过锁页暂存环流式传输。Linux 驱动没有这个限制，那里整个区域都锁页。
  `STRATA_ARENA_PIN_GIB=N` 在任何 OS 上最多锁 N GiB，`0` 锁整个区域。只有已注册的分层能用
  直接 PCIe 拷贝；未注册的分层走现有的 CPU 和 host 暂存路径。
- 每块卡都需要算力 7.0（Volta 或更新）。构建为每块所选卡的架构包含代码；仅 Ampere 的 prompt
  内核在 Volta 和 Turing 上自动使用其可移植回退。

<a id="with-an-expert-profile-the-default"></a>

**decode 把各分页依次跑。** 一个 verify 窗口先在一块卡上完成，下一块卡才开始它那份：每个窗口
一次交接（几百 KB 经由锁页内存）。prompt 从切分中获益——每块卡读某个分块中自己的层，同时前一块
卡读下一个分块——但 decode 是各分页之和，不是较长那个。两块对等卡在 decode 期间各忙约一半时间
（在一对 V100 16 GB 上实测：一块卡 100% 利用率时另一块闲置，逐窗口交替）。

<a id="decode-runs-the-stages-one-after-another"></a>

窗口内部，无论窗口多大，每层的超连接读取与其 K/V 追加都以一小组固定启动运行：

<a id="inside-a-window-each-layers-hyper-connection-read"></a>

- **超连接读取（hc-read 两半）。** 该读取在每个层半计算一次 norm、低秩下投影和上投影。在 Volta
  卡（sm_70）上，一组至多 8 个 token（一组即窗口，或 `--spec-split` 下的一半）用两个内核读取：
  down 启动把原来的标量 down 主体按每 token 跑一次。每个 block 处理一个 token 和 8 条投影行一组。
  block 用未变的 8-warp 映射重算其 token 的 norm。没有单独的 norm 启动，归一化后的输入也不经
  全局内存往返；up 内核随后。**每个 token**的输出与标量内核逐位相同。其他所有架构保持原来的
  三次启动读取。
- **K/V 追加。** 每个 verify 组对 Q8、Q4 或 FP16 KV 使用一次追加启动。混合 KV 对 K8 一次、V4
  一次。每个 token 读自己的位置记录并写独立的 cell。存在 host 副本时追加也会写它。量化算术与
  单 token 追加相同。页边界和被拒尾巴的覆写都有奇偶校验。

<a id="hyper-connection-read-the-hc-read-halves"></a>

驱动窗口的主机线程被钉在自己的核上，等 GPU 时自旋（`SessionLoopScratch`、`Verifier::run`）；
在那台机器上，每 33.6 毫秒的窗口有 11.3 毫秒是等待，对比每层 0.7 毫秒的 host 工作和约 1.3
毫秒的卡侧标志等待。钉核是有意设计，也不是 decode 瓶颈：GPU 才是关键路径，把循环的工作摊到
更多核上没有改变任何吞吐量（见实测小节）。尚未解决的极限是让一个窗口的各位置跨交接流水化——
下一块卡跑位置 t 时上一块卡仍在跑位置 t+1——那会让 decode 变成较长分页而不是各分页之和。

<a id="the-host-thread-that-drives-the-windows-is-pinned"></a>

## 实测

<a id="measured"></a>

Coder 在 RTX 5080 + RTX 3090（Ryzen 9 9950X3D）上，32K 上下文；细节在
`bench/results/2026-09-29-layer-split/`：

<a id="the-coder-on-an-rtx-5080--rtx-3090"></a>

| | Prompt 16K / 28K tok/s | Decode 故事 / 代码 tok/s |
|---|---|---|
| 5080 单卡 | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090，最佳切分（K=26） | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090，auto（K=22） | 2,037 / 2,073 | 80 / 109 |

- **prompt 获益最大**（+18-20%）：每块卡读某分块自己的层，另一块卡读下一个分块。
- **decode 与更快那块单卡持平**，代码上更好。一旦两边缓存装下几乎每个路由到的 expert，就由
  每层 GPU 时间决定。
- **正确性：** 单卡与 0.1.20 逐字节一致，交接本身逐位精确。
- **两块对等卡**（一对 V100 16 GB，Qwen3.8-Flash-Next Q2_0，4K prompt，`--layer-split 20`）：
  decode 55-58 tok/s，单卡只有 19-22 tok/s——切分在这里让 decode 几乎翻三倍，因为第二个缓存
  砍掉了要 CPU 池计算的漏掉 expert 质量（命中率 0.93-0.98 对 0.73）。各分页仍然依次跑，所以两块
  对等卡不会让 decode 翻倍：每窗口时间是两个分页之和，每块卡大约一半时间忙碌。窗口把 host 线程
  钉住的核驱动到满自旋、GPU 同时在跑，而当这项工作本可以摊到各核时速度没有变化——host 上的卡侧
  标志等待（waitA + waitB + waitCPU）合计每 33.6 毫秒窗口约 1.3 毫秒。
- **该机器上的 hc 读取**（捕获事件 fixture：种子 99，bf16 权重，原生 2560/4/320，窗口大小
  1 到 8）：每次调用的 GPU 微秒数，每块卡上原始 vs 融合两内核读取：

<a id="-prompts-gain-the-most-18-20"></a>

  | T | 原 GPU0 | 新 GPU0 | 原 GPU1 | 新 GPU1 |
  |---|---------:|---------:|---------:|---------:|
  | 1 | 61.9     | 42.2     | 60.5     | 40.6     |
  | 2 | 71.3     | 50.0     | 69.8     | 48.5     |
  | 3 | 79.3     | 56.3     | 77.6     | 54.2     |
  | 4 | 87.8     | 78.6     | 86.4     | 74.0     |
  | 5 | 97.7     | 85.6     | 95.8     | 83.9     |
  | 6 | 105.7    | 90.2     | 104.0    | 88.1     |
  | 7 | 115.2    | 94.7     | 113.6    | 93.2     |
  | 8 | 124.9    | 116.2    | 123.1    | 113.8    |

  八个窗口的 old 与 new 完整输出转储逐字节相同（合计 3,360,768 字节）；标量奇偶校验、捕获重放
  和 no-inject 检查全部通过，最终构建上六个选定内核 ctest（kv q8/q4/stream/hybrid parity、
  gr parity、qsa parity）通过。捕获事件 fixture 测的是内核延迟而非吞吐量——它不是运行时增益的
  证明。

### V100 运行时对比

<a id="v100-runtime-comparison"></a>

**历史（2026 年 10 月 1 日）。** 本对比早于 upstream v0.1.39 集成，测量的是本 fork 的融合
Volta GR / 批量 KV-append 工作，对象是它之前安装的 v0.1.36 系运行时。它连同原始出处保留；
v0.1.39 构建是 [DETAILS.md](DETAILS.md#upstream-v0139-in-the-v100-fork) 中 prefill A/B 的对象。

<a id="historical-1-october-2026"></a>

最终对比使用两块 V100 PCIe 16 GB。GPU0 是 Gen3 x2 链路。GPU1 是 Gen3 x16 链路。模型是
Qwen3.8-Flash-Next Q2_0。层级切分为 20/28。主 KV 为 int8，容量 262,144 token。视觉保持开启。
Prefill 分块大小与 expert 放置用自动设置。

<a id="the-final-comparison-uses-two-v100-pcie-16-gb-cards"></a>

每个 prompt 尺寸三次全新请求、每请求 256 个输出 token。没有请求复用 prompt token。种子 1001
控制的是 prompt 生成器，不是模型采样器。服务器使用贪心采样。表格显示对比这些改动之前安装
运行时的中位数（token/秒）。

<a id="each-prompt-size-has-three-fresh-requests"></a>

| Prompt 目标 | Prefill 之前 | Prefill 之后 | Decode 之前 | Decode 之后 | Decode 变化 |
|---|---:|---:|---:|---:|---:|
| 4K | 441.8 | 442.6 | 61.0 | 64.4 | +5.6% |
| 8K | 619.7 | 621.0 | 58.7 | 68.0 | +15.8% |
| 16K | 818.9 | 820.3 | 56.6 | 66.4 | +17.3% |
| 32K | 974.4 | 977.1 | 58.6 | 66.8 | +14.0% |

两条路径在全部四个尺寸上都提升 20% 的目标没有达成。Prefill 吞吐大致不变。14/34 切分配
12,288-token prefill 分块把短 prompt prefill 提到 650.4/778.4 token/秒，但 32K prefill 降到
839.9。安装的配置保持原来的切分和自动分块大小。没有任何被否决的调优参数被启用。

<a id="the-goal-of-a-20-gain-in-both-paths"></a>

最终构建通过六个内核对等性测试。GR 内存与同步检查报告零错误。一次实时请求以 53,508 个 prompt
token 和 256 个输出 token 越过草稿环边界。它的直接前缀扩展恢复了 49,152-token 检查点并产生
128 个输出 token。两次请求的主 KV 都留在 VRAM。

<a id="the-final-build-passed-six-kernel-parity-tests"></a>

内核逐位一致并不意味着所有放置下生成文本相同。固定无-PCIe-缺失的对照组三次完成哈希全部一致。
其他对照改变了 expert 缓存容量，可能改变 CPU/GPU 算术。原始版本对原始版本的重启也改变了一个
完成结果。

<a id="kernel-byte-parity-does-not-establish"></a>

原始矩阵、对比数据与验证细节在 `bench/results/2026-10-01-v100-autonomous/`。要重复四尺寸矩阵，
用 `bench/run_v100_bench.py --model qwen --targets 4096,8192,16384,32768 --seed 1001
--repeats 3 --max-tokens 256`，并按本地安装加上模型 GGUF 与输出目录选项。

<a id="raw-matrices-comparison-data-and-verification-details"></a>

**用哪些卡、什么顺序：**
- 把最快的卡放前面；auto 会给它缓存允许的层数。
- 两块卡已装下模型时，不要加入慢得多的卡。RTX 2080 Ti 作为第三块卡让 5080 + 3090 组合变慢
  （decode 68 / 90 tok/s）：每多一块卡每个窗口多一轮。
- 当模型的路由 expert 装不进较快的卡时，更多卡才有回报。

<a id="which-cards-and-in-what-order"></a>

## 一条对话让两块卡都忙碌（`--pipeline-windows`，两块或更多 AMD gfx12 分页时默认开启）

<a id="one-conversation-with-both-cards-busy--pipeline-windows"></a>

**AMD RDNA4（gfx1200 / gfx1201）上默认开启。** 带 `--serve`、草稿层、**两块或更多分页**的层级
切分时，除非另有说明 `--pipeline-windows 2` 是开的（日志会说 "is on by default for a layer
split of N stages"；两分页自 0.1.44、三分页及以上自 0.1.43 起）。这不是质量交换：token 就是
串行循环的（除 CPU 舍入，见下）。`--pipeline-windows 0` 或 `STRATA_PIPELINE_AUTO=0` 关闭它；
`--pipeline-windows 2` 在任何切分上要求开启。那些在你主动要求时会关掉它的相同选项（`--batch`、
`--peer-device`、辅助缓存、没有草稿层、`--adapt-async 1`），在它只是默认值时无声关闭。NVIDIA
构建保持需显式开启。

<a id="default-on-amd-rdna4-gfx1200--gfx1201"></a>

**草稿层的位置（0.1.44）。** 草稿层的链过去跑在最后一块卡上，挤在最后一个分页的窗口旁边。
在 2x / 4x R9700 上，两条流于是互相拖慢，多数引擎启动都会发生且贯穿该引擎的一生：链运行时
最后一个分页的窗口要 27 毫秒而不是 13 毫秒（幸运启动时 14.9 毫秒，那些启动 decode 也快 25%；
0.1.43 笔记里的 "128.9 does not reproduce" 就是这个）。开流水线后，引擎现在把草稿层加载到
**第一块**卡上——链开始时它的窗口已经完成：链的源行每条链只跨卡拷贝一次，草稿头的行只在加载时
一次。token 相同；`STRATA_MTP_STAGE0=0` 保持旧位置。精确参数 A/B，5 对，decode tok/s
（串行循环 -> 现在）：双卡代码 103 -> 125（+21%），4K prompt +8%，散文持平；三卡代码 +19%、
4K +12%、散文 +2%；四卡代码 +22%、散文 +4%、4K +15%。

<a id="where-the-draft-layer-sits-0144"></a>

切分时各卡轮流跑一个 verify 窗口：第一块卡跑完它的层并交接，然后等最后一块卡跑完剩下的、head
和草稿。`--pipeline-windows 2` 让第一块卡在最后一块卡仍在 verify 当前窗口时就开始**下一个**
窗口。下一个窗口是猜测：猜当前窗口被整体接受、且其奖励 token 正是草稿层预测的那个（草稿层在
在途窗口的各草稿上一路跑下去）。猜中时，下一个窗口已经完成一半；没猜中时，第一块卡恢复它的
状态（窗口运行时保存的递归状态副本），下一个窗口用真实 token 重建。只有当草稿层的估计表明
猜测很可能被保留时才开始一个猜测窗口。从早先上下文拷贝来的窗口（`--suffix-draft`）也会被
猜着越过，这正是拷贝文本的编辑获益最多的地方。`--pipeline-windows 1` 只重叠穿过 verify 窗口
的短 prompt 读取（`--short-read`）。

<a id="with-a-split-the-cards-take-turns-on-a-verify-window"></a>

两块或更多卡，`--serve`。三个或更多分页时，除最后一个外每个分页都是前置分页：猜测的窗口跟随
已验证的窗口逐个通过它们、每块慢一拍（每个前置分页保存自己的一对递归状态副本，猜错时恢复自己
的状态），只有最后一个分页等裁决。`--pipeline-windows 1` 和异步层保持两级。在配置中：

<a id="two-or-more-cards--serve"></a>

```
"args": [ ..., "--pipeline-windows", "2" ],
"layer_split": "20"
```

- **代价**：每块卡上第二个 verify 窗口、每块卡 expert 缓存多让出 160 MiB，加上 `2` 时第一块
  卡上第一块卡递归状态的两份副本（它每跑一个 GDN 层约 3 MiB）（三个或更多分页时每块前置卡都
  如此；只有第一块卡的份额从它缓存中让出，其他卡之后自己分配）。
- **同样的文本**：最后一块卡只跑已验证的窗口，且每个窗口的每一行计算的内容与在任何其他窗口
  相同，所以 token 就是串行循环的。配 `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`，
  贪心输出与相同 expert 缓存的串行循环逐位相同。流水线把它的 VRAM 从缓存中让出，所以对比不带
  该参数的运行，有几个 expert 从卡移到 CPU、舍入不同，接近平局可能翻转（通过
  `--vram-reserve-mib` 获得相同缓存的串行运行与它完全一致）。
- **关闭，日志里写一行原因**：带 `--batch` 槽、`--peer-device`、辅助缓存
  （`--expert-cache-device1..3`，`--remote-expert-opt` 建立在它之上）、切到一块 GPU（`--split-device 0`）
  或没有草稿层时；三个或更多分页时 `--pipeline-windows 1` 和 `--adapt-async 1` 也关闭它。带重复
  惩罚（`penalty_last_n`）或耦合草稿采样的请求串行解码。
- **与常驻 RAM 模式的异步交换同用**（`--adapt-async 1`，[DETAILS.md](DETAILS.md)）时，一轮的
  步骤在已验证窗口之间推进。每块卡的副本由 decode 循环自己排队，在该卡没有在途窗口时、每个
  可能仍会读取被覆写内容的窗口结束之后；移入 RAM 的动作同样等待。
- **实测**（Swift 1.5 IQ3_XXS，160K 上下文，q4_0 KV，原版草稿层，RTX 4060 Ti（层 0-19）+
  RTX 5080（20-47），i9-14900KF，32 GB RAM，常驻 RAM 模式开在切分上（#848）；贪心，500 token，
  两组交错各三轮，decode tok/s）：Python 代码 90.4 -> 103.1，C 代码 76.6 -> 81.6，英文散文
  63.0 -> 71.1，意大利语散文 41.6 -> 46.3，拷贝密集的编辑（5 KB 文件改名带回）90.5 -> 117.9；
  平均 72.4 -> 84.0（+16%）。窗口受 GPU 限制时它才划算：expert 经操作系统文件缓存读取时
  （那台 32 GB 机器上的 `--mmap-experts`）文件读取占主导，实测没有更快。
- **四个分页上的实测**（Flash-Next GSQ-RCO IQ3_S，262K 上下文，int8 KV，层切分 12,24,36 在
  4x RX 7900 XT 上，每个 expert 都在 VRAM；贪心，reasoning 关，256 token，仓库文本 prompt）：
  decode 4K 时 59.4 -> 67.6 tok/s（3 次中位数），32K 时 51.8 -> 65.6。猜中的猜测窗口约 15 毫秒，
  全新窗口约 43 毫秒；约三分之一的猜测命中。贪心文本在全部四段 prompt 上与串行循环一致
  （`STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`），以及强制每次猜测落空时
  （`STRATA_PIPELINE_FORCE_MISS=1`）。
- **HIP**：草稿层每行的 `gr_read` 过去用最后一个分页的会话 scratch（`ss.block.gr`），而该分页的
  verifier 也用它；在裁决时启动的链与下一个窗口的最后分页同时跑在卡上，窗口的 head 读到被踩坏的
  混合数据（猜中之后的错误 token）。草稿层现在有自己的 scratch。

<a id="-cost"></a>

`STRATA_PIPELINE_*` 调优与测试变量（THETA、FORCE_MISS、SWITCH、LOG、TRACE、SPEC_DEPTH 等）只在
`STRATA_PIPELINE_DEBUG=1` 时才读取。`--pipeline-windows` 与 `--adapt-async 1` 可以组合：只有设了
`STRATA_PIPELINE_ADAPT_ASYNC=0` 时，引擎才在 `--pipeline-windows 2` 旁边关掉异步层。任何让二者
关闭的原因会在启动时打印一次（"is off: ..."）。`--remote-expert-opt` 只有配合辅助缓存
（`--expert-cache-device1..3`）才有作用；在普通层级切分上它是惰性的，setup 也不再在那里写它
（#1447）。

<a id="the-strata_pipeline_-tuning-and-test-variables"></a>

在 2x RTX 3090（sm_86，限 250 W；GPU0 PCIe 4.0 x16，GPU1 x4，无 NVLink；Ryzen 7 9800X3D，
32 GB RAM）上实测，Qwen3.8-Flash-Next GSQ-RCO IQ3_S，`--resident-experts`，KV int8，
`--spec 4 --mtp`，`--pipeline-windows` 各行用 `"layer_split": "29"`（不带
`--remote-expert-opt`）；其余是 setup 的配置（`auto`）。Decode 是温度 0.6 下 1,500-token 编码
回复第 2-6 次运行的平均值，prefill 是一次冷读 19.9K-token prompt；除注明外每臂一次运行
（由 adambenhassen 报告，#1447；未在我们的机器上复现）：

<a id="measured-on-2x-rtx-3090-sm_86-250-w-limits"></a>

| Arm | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| setup defaults (two runs) | 137.0 / 136.9 | 1798 |
| `--adapt-async 1` (two runs) | 145.2 / 143.7 | 1800 / 1797 |
| `--adapt-async 1`, no `--remote-expert-opt` | 145.0 | 1807 |
| `STRATA_ADAPT_LAG=2` | 142.8 | 1800 |
| `STRATA_EXCHANGE_ROTATE=1` | 139.6 | 1800 |
| `STRATA_PF_FUSED=1` | 137.2 | 1913 |
| `--pipeline-windows 2` | 136.7 | 2097 |
| `STRATA_SPEC_COUPLED=1` | 137.6 | 1804 |
| `STRATA_SPEC_PROB=1` | 137.2 | 1788 |
| `STRATA_SPEC_COUPLED=1` + `STRATA_SPEC_GUMBEL=1` | 136.5 | 1805 |
| `async + lag 2 + rotate + pf_fused` | 145.0 | 1829 |
| `pw2 + lag 2 + rotate + pf_fused` | 135.9 | 2296 |
| `pw2 + async + lag 2 + rotate + pf_fused` | 142.0 | 2302 |

最后一行保住了异步层大部分 decode 增益和流水线的 +28% prefill；任何臂都没有停顿。
这些是在一台机器上需显式开启的设置，不是默认值。

<a id="the-last-row-keeps-most-of-the-asynchronous-tiers-decode-gain"></a>

## 同时多条对话

<a id="several-conversations-at-once"></a>

有了层级切分，`--batch N --batch-groups G` 可以一起解码多条对话并把它们流水线地穿过各卡：见
[BATCHING.md](BATCHING.md)。槽位会话是在一个分页自己的会话、head 和 MTP 草稿层之后切出来的，
所以切分时 fit 检查看到的是每块卡上真正剩下的内存——包括草稿所在的最后一个分页。装不下的数量在
启动时回退到更少的槽，或回退到一次一个请求，配置保持不变（在本机柜的 20/28 切分、524,288-token
上下文上观察到；最终的准入时刻数字在基准报告中）。

<a id="with-a-layer-split--batch-n--batch-groups-g"></a>
