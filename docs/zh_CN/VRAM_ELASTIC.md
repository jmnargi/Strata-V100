# POST /v1/vram：它做什么、答复什么（可选开启，单块 NVIDIA GPU，#533，#1034）
<a id="post-v1vram-what-it-does-and-what-it-answers-opt-in-one-nvidia-gpu-533-1034"></a>

本页依据 `serve/server.py`、引擎的 `VRAM` 命令（`src/program/generate.cpp`、`src/core/expert_cache.cpp`），以及
`serve/test_vram.py` 中单元测试所固定的各种情形写成。标有 **(code)** 的一行是从代码读出来的、本身没有对应测试；
把它当作一种意图说明，而不是来自实测的承诺。实测数据见 [DETAILS.md](DETAILS.md)（RTX 5070，Q2_0）。

## 它是什么
<a id="what-it-is"></a>

专家缓存（expert cache）通常是一整块 `cudaMalloc`。当配置里写了 `"vram_elastic": true`（引擎标志 `--vram-elastic`，
分段大小 `"vram_segment_mib"`，默认 512，至少 64）时，它改为分段分配。此后 `POST /v1/vram` 会通过解除最后若干分段的
映射，或在其空出更多时重新映射它们，从而为另一个程序保留若干 MiB 显存（VRAM）。没有任何东西会自动改变大小。
只有专家缓存会移动：稠密权重、K/V、MTP 头、CUDA 上下文和提示词缓冲区都留在原处，所以 Strata 绝不会交回"全部"。

引擎会在启动时关闭该特性，并在日志里写一行（`--vram-elastic is off: ...`），情形包括：不是 `--serve` 引擎、带层切分
（layer split）、其他 GPU 上有助手缓存、分段小于 64 MiB，以及在 HIP 上（仅限 NVIDIA）。如果驱动不提供虚拟内存管理，
启动会失败并报出该原因的错误 **(code)**。没有这个标志时一切不变：答复与不带它时完全一样。

## 请求
<a id="the-request"></a>

```
POST /v1/vram
Content-Type: application/json          (anything else: 415)
{"reserve_mib": 8000}                    (a whole number of MiB, 0 or more; null = the reserve the engine started with)
```

- 服务器的 API key 与其他 `/v1` 路由一样适用。带有外来 `Origin` 的请求会被拒绝（403），除非该 origin 在
  `trusted_origins` 里。
- `reserve_mib` 是你希望在这次调用后显卡上留下的空闲显存（它计入任何程序所占的量，不只是 Strata）。
  小于 0、字符串、布尔值或小数：400。
- `null` 与 `0` 是"涨回"请求：`null` 保留引擎启动时的 `--vram-reserve-mib`，`0` 则有意什么都不留。

## 返回什么
<a id="what-comes-back"></a>

| 情形 | 状态 | 响应体 |
|---|---|---|
| 完成（或无事可做） | 200 | `{"status": "ok", "reserve_mib", "expert_slots", "expert_slots_full", "expert_cache_mib", "expert_cache_full_mib", "vram_free_mib", "prompt_chunk"}` |
| 模型未加载 | 200 | `{"status": "not loaded", "reserve_mib": N, "note": "applied when the model loads"}` |
| `reserve_mib` 非法 | 400 | `invalid_request_error` |
| 引擎拒绝该命令 | 400 | `invalid_request_error`，引擎的原因写在 `message` 里。引擎继续运行。 |
| 并非来自 Strata 自己的页面，或内容类型错误 | 403 / 415 | |
| 仍有请求在运行或排队（最多等待 300 s） | 409 | `model_busy` |
| 引擎已结束，或 120 s 内没有答复 | 503 | `server_error` |

**部分成功也是一个 200。** JSON 并不会说保留量没达到；请拿 `vram_free_mib` 和你的 `reserve_mib` 比较
（并拿 `expert_cache_mib` 与 `expert_cache_full_mib` 比较）：

- **收缩。** 引擎会腾出缺口，但绝不低于它的下限：128 个槽位加上最小的提示词借用（一个 256-token 的分块）。到达下限后，
  调用仍会返回 200，而 `vram_free_mib` 低于保留量。日志行会写 `(the cache keeps its smallest size: the prompt path's buffers)`。
  分段是整段归还的，且引擎会保留足以覆盖必须留下内容的分段，所以空闲显存可能比保留量少一整个分段（默认 512 MiB）**(code)**。
- **扩张。** 它会映射分段，直到超出保留量之外的那部分空闲显存被用完，或缓存已满。如果驱动没有显存放得下下一个分段，
  缓存会保持更小，而调用仍返回 200；原因只在服务器窗口里（`(the driver has no VRAM for another expert-cache segment)`）。
- 这些数字是在调用之后测得的（`cudaMemGetInfo`），所以稍后才分配的另一个程序不会算在其中。

## 引擎说不的时候（400，引擎继续运行）
<a id="when-the-engine-says-no-400-engine-stays-up"></a>

- `VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)`。
- `VRAM: not with the resident low-RAM mode` 和 `VRAM: not with --peer-device`。
- `VRAM: not while batch slots are decoding`（配合 `--parallel`/`--batch`；服务器还会先取得引擎的控制锁，
  因此一个正在被接纳的提示词会在改变大小之前完成）。
- `VRAM: a prompt's loan is still out`。
- 改变大小之前出现设备错误：`VRAM: <cuda error>`。

## 改变大小内部的失败 **(code)**
<a id="failure-inside-a-resize-code"></a>

有两种失败可能在缓存已经改变之后发生。这两种情况下引擎都答复 `ERR`（HTTP 400）并继续服务，
但我不会再相信它的缓存表：

- 驱动拒绝释放一个分段（`the driver refused to release an expert-cache segment`）。槽位计数已经更新；
  专家的就驻留表（residency table）在这条路径上没有被修复；
- 涨回途中重新填充一个槽位失败（`VRAM: refilling the cache failed: ...`）。分段已映射，但某些专家没有回到其中。

对这两者的安全应对是卸载再加载模型（`POST /v1/unload`，然后 `POST /v1/load`），这会启动一个全新的引擎。
更小的缓存不会给出错误答案（缺失的专家在 CPU 上计算，与任何缓存之外的专家一样），但我没有对这两条路径跑过故障注入测试。

## 交出显存再取回
<a id="giving-the-vram-back-and-taking-it-again"></a>

1. `{"reserve_mib": 8000}`：缓存收缩直到空出 8000 MiB（或收缩到它的下限）。被归还分段里的专家在 CPU 上计算，
   所以答案仍会持续产出，只是更慢（RTX 5070，Q2_0：交出 4.3 GiB 时 decode 从 44 降到 33 tok/s；该调用耗时 78 ms）。
2. 另一个程序运行并结束。
3. `{"reserve_mib": null}`（或一个更小的数字）：缓存涨回到空闲显存所允许的程度。涨回会把同样的专家放回同样的槽位
   （在 DETAILS 的实测里，答复与收缩前逐 token 相同）；如果自适应层已经带回了某个专家，则该层被路由最多的缺失专家
   占据那个槽位 **(code)**。拿 `expert_cache_mib` 对照 `expert_cache_full_mib` 就能知道它是否已完整。

## 卸载、重启与崩溃
<a id="unload-restart-and-crash"></a>

- 服务器会在**内存中**记住上一次的 `reserve_mib`，并在引擎再次启动后重新发送它（空闲卸载、`POST /load`，
  或崩溃后的重启）。引擎在那次启动时依据*当时*的空闲量确定其缓存大小，而保留量只在其起来之后才施加。
  如果失败，服务器窗口会写 `the VRAM reserve (N MiB) was not applied: ...`，引擎则用它拿到的那个缓存运行。
- 如果加载时另一个程序正持有显存，则由 `min_free_vram_mib`（配置）决定：加载最多等待 15 s 让出那么多空闲显存，
  然后答复 503 `the GPU is in use by another program`，模型保持未加载。
- 服务器重启会忘记保留量：引擎以其完整缓存启动。
- 在未加载的模型上调用它不会改变 GPU 上的任何东西；它只为下一次加载设置保留量。
- `GET /v1/status` 有一个 `vram` 对象，含引擎报告的最后一组数字（在第一次成功调用之前为空）和 `elastic`。

## 按需的图片编码器
<a id="the-image-encoder-on-demand"></a>

当配置的 `"vision"` 段里写了 `"elastic": true`（且 `"vram_elastic": true`；没有它时服务器会说明这一点，
并像以前一样随模型一起启动编码器）时，GPU 编码器不会随服务器或随一次加载而启动：

- **某个图片的回合**（与一次编码始终采用的同一个 FIFO 回合）：如果编码器没在运行，引擎的 `VRAM` 命令会先在保留量
  之上再多留 `"elastic_mib"`（默认 2048）—— 依据最后一次 `POST /v1/vram`，若从未调用则为 0 —— 然后启动编码器。
  如果缓存无法收缩（常驻式低内存模式、不带 `--vram-elastic` 的引擎），它仍会启动，就像一个非弹性编码器那样，
  日志会说明原因。
- 没有图片而**空闲**达到 `"elastic_idle_s"`（默认 120）：一个线程会在请求之间停止编码器，并带着那个保留量重新发送
  `VRAM`，于是缓存涨回到它原有大小（用 0，而不是启动时的保留量：那会让它小上一个或多个分段）。
- **编码器运行时调用 `POST /v1/vram`** 会在编码器份额之上保留所请求的保留量；随后编码器的停止会回到所请求的保留量。
- 卸载模型也会停止编码器。日志行以 `[strata] images:` 开头 **(code；`serve/test_vision_elastic.py` 中的单元测试固定了
  每一种情形)**。

`--vram-elastic` 会关掉 `--kv-grow`（引擎在启动时会说明这一点）：两者都要求时，整个 K/V 都在显存里。

## 可以在此基础上构建什么
<a id="what-to-build-on"></a>

用于自动化：只把带 `status: "ok"` 的 `200` 当作一次真正执行了的调用，然后自行检查 `vram_free_mib >= reserve_mib`；
`409` 稍后重试；遇到 `503` 或上述失败路径里的 `400`，就卸载并加载；自己记录你请求过什么，因为服务器重启会忘记它。
