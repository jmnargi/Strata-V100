# 常驻专家层交换缓冲区轮转
<a id="resident-expert-exchange-buffer-rotation"></a>

当常驻 RAM 补足部分与 GPU 缓存交换一个专家层时，被换出的专家层会落到一个临时的宿主缓冲区。原有路径随后把
这些字节拷贝到被提升的专家层原先占用的 RAM 槽位。轮转让这最后一次宿主拷贝消失：临时缓冲区直接变为常驻，
而原来的 RAM 槽位成为下一个临时缓冲区。

## 启用
<a id="enable"></a>

轮转**默认关闭**。请用你现有的参数启动引擎之前先设置 `STRATA_EXCHANGE_ROTATE=1`：

```bash
export STRATA_EXCHANGE_ROTATE=1
```

```powershell
$env:STRATA_EXCHANGE_ROTATE = '1'
```

只有精确的 `1` 才会开启。设为 `0` 或删除该变量即恢复拷贝路径。该设置不选择常驻 RAM 模式，也不改动任何
预设。只有在常驻补足部分已完全页锁定（pinned）并映射到 GPU、且交换缓冲区也能页锁定并映射时才生效。
其它布局继续走拷贝路径，并记录轮转无法开启的原因。没有常驻补足部分的配置不会用到它。各层的专家块大小
可能不同（GGUF 每层量化不同的情况）：见[块大小不同的层](#layers-of-different-block-sizes)。

启动诊断会确认已启用：

```text
FileExpertSource: exchange buffer rotation enabled: ...; no host commit memcpy
```

服务端的诊断会报告累计轮转的块数和避免掉的 `memcpy` 载荷字节数。这里只按一次计算拷贝的载荷，而不是
读加写的流量，也不包含仍然会发生的 GPU 传输。

## 所有权与同步
<a id="ownership-and-synchronization"></a>

原始分配在 `FileExpertSource::close()` 之前一直拥有自己的内存。轮转会更新槽位所有权，并让每个宿主指针
与其 GPU 别名保持配对。原子的槽位 ID 让后台的常驻性查询能看到这些不可变的配对。计算读取方和 GPU 传输
仍必须在现有的串行化提交之前完成；轮转不引入重叠计算，也不改变同步策略。请在轮转开始前预留最大的交换
容量：运行中的交换 arena 不允许扩容，因为它现在可能存放常驻专家层。

## 块大小不同的层
<a id="layers-of-different-block-sizes"></a>

一次交换在同一层的两个专家层之间进行，因此被提升的专家层腾出的 RAM 槽位，尺寸正好等于被换出者所需块
的大小。所以轮转按块大小分别保留备用缓冲区：某一层的交换取走该尺寸的一个备用块，再交还一个同尺寸的；
每种尺寸的备用块数量从此保持不变。最初的版本要求所有层使用同一种尺寸，遇到其它模型包就退回拷贝路径；
Swift 1.5 IQ3_S 有七种尺寸（1,510,400 到 2,662,400 字节），IQ3_XXS 也有九种，所以两者都无法轮转。

交换分配会为每一轮中的每次交换，为每种尺寸各留一个备用块：`--adapt-swaps`（最多 96）乘以各不同尺寸
之和。使用 `--adapt-async 1` 时，后半部分缓冲区只用于暂存拷贝，仅按最大尺寸保留。在 IQ3_S 上以
`--adapt-swaps 80` 运行时，这占 1.06 GiB 的页锁定 RAM（使用 `--adapt-async 1` 时为 1.26 GiB），而拷贝
路径的缓冲区是 0.20 和 0.40 GiB。启动行会说明有多少种尺寸、占用多少：

```text
FileExpertSource: exchange buffer rotation enabled: 80 buffers in 7 block sizes (largest 2662400 bytes, 1.06 GiB page-locked); no host commit memcpy
```

使用 `--adapt-async 1` 时，轮转后的一轮还会提前一个 verify 窗口结束：后台作业没有字节要搬运，所以
所有权翻转在确认拷贝到位的那个 tick 里就完成了。

`tests/core/exchange_storage_test.cpp` 在不用 CUDA 的情况下覆盖了它（三种尺寸、6,144 次交换：字节、
别名、拒绝其它尺寸的备用块或其它层的专家层、拒绝重叠的常驻块）。

### 实测：RTX 5090（64 GB，Windows 11）上的 Swift 1.5 IQ3_S
<a id="measured-swift-15-iq3_s-on-an-rtx-5090-with-64-gb-windows-11"></a>

Core i7-13700K，RTX 5090 32 GB，64 GB DDR5，Windows 11，驱动 617.42，CUDA 13.0.88，引擎 0.1.42 带本次
改动自行构建。Swift 1.5 IQ3_S，128K 上下文，`--resident-experts`（31.4 GiB 的专家层页锁定在 RAM，
24,576 个中约 10,800 个在显存），`--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`，`--spec 4`，
`--pool-affinity p-cores`。第一行和第三行用同一个二进制跑两个 arm，只有 `STRATA_EXCHANGE_ROTATE` 不同；
第二行里关闭的 arm 是本次改动之前的构建（第三行所用的二进制还带有额外的计时器）。

| 运行 | 关闭轮转 | 开启轮转 |
|---|---|---|
| 一个 8K 提示词，同样强制 1,500 token（`--spec-follow`），blocking 档，decode tok/s | 163.4, 165.1, 163.1, 161.6, 162.6 | 187.4, 188.7, 185.8, 184.9 |
| `--serve`，23 个编码 agent 轮次（`tools/agent_turns_bench.py` 的轮次，贪心，最多 400 个新 token），blocking 档，全部轮次的 tok/s，三轮交错 | 147.8, 159.2, 158.4 | 175.2, 184.9, 172.8 |
| 同样的轮次，使用 `--adapt-async 1`，同一个二进制，两轮交错 | 166.0, 170.1 | 185.6, 182.5 |

第一行中，每个 arm 各有两次运行的专家缓存大小相同（10,681 个槽位），它们为全部 434 个窗口写出了相同的
`--window-hashes` 文件：逐窗口的算术完全一致。（`--expert-cache auto` 在启动时按空闲显存决定缓存大小，
槽位数不同的运行会以不同方式放置专家层，所以无论是否开启轮转，它的哈希都会不同。）在这种运行中，分层
机制交换了约 14,000 个专家层，其中 10,100 到 10,600 次是交换。在 serve 那两行里，开启轮转的 arm 在 23 轮
中全部更快（blocking 档），`--adapt-async 1` 下是 23 轮中的 22 轮。在这台机器上，分层机制在整次运行中
每个 verify 窗口交换约 30 个专家层（缓存容纳 44% 的专家层），因此拷贝路径在 decode 线程上每个窗口搬运
约 50 MiB 的 RAM；轮转省掉的正是这段时间。使用 `--adapt-async 1` 时同样的拷贝跑在后台作业上，decode
线程在它运行期间每个窗口仍要损失约 1 ms（按请求不同为 0.4 到 2.5）；开启轮转后，一个窗口中分层机制的
那部分只有 0.06 ms。

未实测：Linux、AMD 显卡、层切分，以及在这台机器上只有一种块大小的模型包（它的代码路径就是上面那条，
只是只有一种尺寸）。

## 测试
<a id="tests"></a>

所有权测试可以不依赖 CUDA 运行：

```bash
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread -Iinclude tests/core/exchange_storage_test.cpp -o /tmp/exchange_storage_test
/tmp/exchange_storage_test
```

已有配好的 CUDA 构建时：

```bash
cmake --build build --target strata exchange_storage_test file_expert_source_test
ctest --test-dir build -R '^(exchange_storage_test|file_expert_source_test)$' --output-on-failure
./build/file_expert_source_test --rotation-gpu
compute-sanitizer --tool memcheck --error-exitcode 71 ./build/file_expert_source_test --rotation-gpu
```

CPU 测试在 12,304 次交换中检查字节保持、指针/别名所有权、guard、非法操作和并发的常驻性查询。CUDA
fixture 在三种模式下各做 64 次交换：页锁定 RAM 的拷贝、页锁定 RAM 的轮转，以及在可换页（pageable）RAM
上请求轮转（回退）。它检查精确字节、GPU 别名读取、文件回退、容量，以及关闭/重新打开。GPU fixture
需要显式调用；普通的 CTest 调用不会运行它。

2026-10-04，在上游 `6f32ec070f23ced9f50e704d854d775da52591ab` 上的干净补丁于 Linux 用 GCC 15.2 和
CUDA 13.3 为 RTX 4090 构建通过。两个 CTest 测试、ASan/UBSan 所有权测试和 CUDA fixture 均通过。
Compute Sanitizer 报告 **0 errors**。该补丁未测试 AMD 和 Windows 上的 GPU 执行。

## 一次原生 token 完全一致的 A/B
<a id="one-native-token-identical-ab"></a>

2026-10-04 实测，使用位于 `18a30ad775bce86e99a54e6dc552ccb72c66d16e` 的干净轮转构建，基于上游
`6f32ec070f23ced9f50e704d854d775da52591ab`。模型是标准 GSQ-RCO **Q2_0**，配与其匹配的 IQ4_NL PLE 分片。
两个分片都已对照 receipt 中固定 revision 上发布者的 SHA-256 哈希校验过。原生模型包使用现有的
`tools/iq_pack.py`；不需要兼容性转换，也不需要 Q8 PLE 读取扩展。

两个 arm 使用同一个 Release 二进制（GCC 13.3，CUDA 13.2），RTX PRO 6000 Blackwell 96GB，Ryzen 9 7950X，
已装 128GB RAM。每个 arm 用一个全新引擎，在相同的 1,024 token 计数提示词之后精确生成 1,024 个 token。
上下文分配为 40,960，12,000 个 GPU 专家槽位，16.19 GiB 的页锁定/已映射常驻补足部分，FP16 KV，仅目标的
贪心解码，生成期间不提供 draft、不复用提示词 token、不读文件层 blob。只有 `STRATA_EXCHANGE_ROTATE`
不同。准入之前会等待传输完成。

GPU 缓存被刻意限制在 12,000 个槽位（15.45 GiB），以便触发常驻 RAM 交换。Q2 完全可以放进这块 96GB 的
GPU；这里比较的是存储路径，不是全 GPU 下最快的配置。它不改变项目的预设，也不会为现有用户开启常驻模式。

| 轮转 | Decode tokens/s | 常驻交换次数 | 避免的宿主拷贝载荷 | 输出 ID |
|---|---:|---:|---:|---|
| `0` | 104.90 | 3,802 | 0 bytes | 1,024，完全相同 |
| `1` | 113.44 | 3,802 | 5,255,884,800 bytes | 1,024，完全相同 |

decode 速率不含启动和 prefill。这一对运行中观察到的吞吐变化为 +8.1%；没有确立可重复的计时结果或普遍
加速。词表外的 EOS 哨兵值 `2147483647` 强制了精确的输出长度，因此这不是模型质量评估。投机（spec）路径
的 token 一致性、AMD 和 Windows 上的 GPU 执行仍未测试。

同一个 Blackwell 二进制通过了两个 CTest 检查和三模式 CUDA fixture；Compute Sanitizer 报告 0 errors。
[receipt](../../bench/results/exchange-rotation-ab.json)包含源码、二进制、模型、模型包和 profile 的哈希、
参数、计时和诊断信息。[raw IDs](../../bench/results/exchange-rotation-token-ids.json)保留了输入和两份输出。
要在没有模型的情况下核对 token 比较：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('bench/results')
r = json.loads((p / 'exchange-rotation-ab.json').read_text())
raw = (p / r['token_ids_file']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == r['token_ids_file_sha256']
d = json.loads(raw)
assert len(d['input']) == len(d['off']) == len(d['on']) == 1024
assert d['off'] == d['on']
for name, arm in zip(('off', 'on'), r['arms']):
    sha = hashlib.sha256(json.dumps(d[name], separators=(',', ':')).encode()).hexdigest()
    assert sha == arm['output_token_ids_sha256']
print('1,024 identical output token IDs')
PY
```
