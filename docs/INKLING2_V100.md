# Inkling 2 Mini on two Tesla V100s

`strata-ik2` is an opt-in CUDA engine for Inkling 2 Mini. It runs the looped model from Unsloth's `UD-Q4_K_XL` GGUF directly; no expert pack or requantization is needed. This is separate from the Qwen engine and does not change its defaults or service.

## Build and download

Run these commands from the Strata checkout. Use CUDA 12.8 for the V100 (`sm_70`). The GGUF needs 17.6 GB of disk space.

```sh
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_INKLING2=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_ORCA_Q4KS_MMQ=OFF
cmake --build build-ik2 --target strata-ik2

hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f --local-dir models/inkling2-mini
python tools/strata_tokenizer.py --gguf models/inkling2-mini/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --out models/inkling2-mini --check
cp tools/ik2/chat_template.jinja models/inkling2-mini/tokenizer/chat_template.jinja
```

Use the repository's Python environment for the tokenizer and server. If the weights live on a separate drive, point the profile's model and tokenizer paths there instead.

## Serve

The sample profile starts at 8,192 tokens with FP16 KV. Both cards need to be available; do not start Inkling alongside another model using the same cards.

```sh
python serve/server.py --engine strata --config docs/inkling2-mini-v100.json \
  --host 127.0.0.1 --port 18000
curl http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 9 + 8? Answer with just the number."}],"temperature":0,"max_tokens":192}'
```

The profile is for text chat, one generation at a time. It does not configure vision or speculative decoding. Avoid forced tool choices; the typed tool format is different from the Qwen forced-prefix format.

On two V100 PCIe 16 GB cards, the arithmetic request returned `17` at 54.3 generated tokens/s. A coding request generated 1,536 reasoning tokens at 51.3 tokens/s and hit the cap before a final answer. These are short-prompt measurements with FP16 KV and an 8K configured window, not quality or long-context benchmarks.

### Prefill and decode by context, both cards

Measured on the V100 pair with the 524K profile's settings: `IK2_EP=1`, `IK2_EP_SPLIT=0.85`, `IK2_EP_DEVICE=1`, `IK2_KV=fp8`, `IK2_DENSE=q8`, `IK2_BATCH_PREFILL=1`, `--chunk 3072`, `--max-context 524288`, UD-Q4_K_XL. The prompt is 11,252 tokens of source text cycled to the stated length, and each run generates 127 greedy tokens, so prefill and decode are read at the same depth. One run per row.

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 2,048 | 1,193 | 53.6 |
| 8,192 | 1,281 | 50.8 |
| 16,384 | 1,208 | 48.4 |
| 32,768 | 1,022 | 43.1 |
| 65,536 | 622 | 35.4 |

`IK2_ATTN_GQA=1` measures 42.2 tok/s of decode at 65,536 and 41.4 with `IK2_ATTN_SPLIT=512`; the rows above predate it.

All five runs kept every one of the 3,072 experts in VRAM across the pair: zero expert-cache misses and 0.0 GiB copied. Prefill falls from its 8K peak as attention reads the accumulated keys, and decode falls with depth from 53.6 to 35.4 tok/s for the same reason. The 64K read moved 65,536 tokens in 105.3 s; expert work was 2.6 s of that, so the rest is attention, dense projections and chunk scheduling at depth.

## 524,288 tokens and concurrent clients

Inkling has eight full-attention layer visits. At 524,288 tokens their KV alone uses 16 GiB in FP16, 8 GiB in FP8, or 8.5 GiB in Q8_0. The sliding-window states and workspace are extra. FP16 KV plus the 16.4 GiB Q4 GGUF already exceeds the pair's 32 GiB.

For one active session, use the same server command with `--config docs/inkling2-mini-v100-524k.json`. This profile allocated 524,288 tokens of FP8 KV and kept all 3,072 experts in VRAM on the V100 pair (431 on the first card, 2,641 on the second). It generated the short arithmetic answer at 49.6 tokens/s. A prompt with 11,252 tokens of Strata source read at 561.7 tokens/s with zero expert-cache misses. Put the x16 PCIe card second for this split. This verified allocation and generation, not a prompt that filled the 524K window.

The 524K profile enables grouped expert prefill (`IK2_BATCH_PREFILL=1`). On the same 11,252-token source prompt, an interleaved A/B/B/A run improved whole-prompt throughput from 564 to 592 tok/s (+5.0%). Expert-phase time fell from 3.02 to 2.20 s (27%). All four runs generated the same token IDs; decode stayed at about 42.7 tok/s.

Prefill attention now takes the logits-and-cuBLAS path instead of the fused kernel (`IK2_FLASH=1` selects the fused one). The fused path runs on this port's `m16n8k16` emulation, which reaches about 0.7 TFLOP/s on Volta and was 41% of an 11,252-token read. Paired A/B on that prompt: **818 to 1,069 tok/s (+31%)**, decode unchanged at 50.4, generated tokens identical. With the faster attention a 2,048-token chunk reads the same prompt at 1,169 tok/s (+9.4% more). It used to spill 99 experts and cost 2.6% of decode; leaving the expert cache 1 GiB of VRAM instead of 1.5 GiB (`IK2_EP_ROOM_MIB`) holds all 3,072 experts again, so decode stays at 50.4 and the misses go to zero. 3,072 is the largest chunk that still holds every expert on the pair: 1,282 tok/s on that prompt, decode 50.3, zero misses. A 3,328-token chunk spills 32 experts and costs 1.7% of decode; 3,840 reaches 1,340 tok/s for 139 misses and 3.7%. Both profiles use 3,072. The chunk size changes the prompt's expert batching, and with it the summation order inside the experts, so 1,536 and 2,048 generate different token ids from the same prompt; each is a valid decoding, but a bit-for-bit comparison against an older run needs the same chunk.

Increasing the grouped prefill chunk from 512 to 1,536 raised throughput on that source prompt from 587–593 to 808 tok/s (19.0–19.2 s to 13.9 s). All 3,072 experts still fit with the 524K FP8 cache, with zero misses and identical generated token IDs. A 2,048-token chunk reached 858 tok/s but evicted experts, so the profile uses 1,536. Decode remained about 43 tok/s at this prompt depth.

Dual-GPU single-token decode now routes experts on the device: each card picks its own half from a device-side table, so a visit needs no host round trip for routing. Paired A/B on a 288-token prompt with 383 greedy tokens and all experts resident: 46.2 to 48.3 tok/s (+4.4%), identical generated tokens. `IK2_EP_DEVICE=1` turns it on.

The loop-gate apply kernel (8.2% of decode, 32 us a call) ran with 256-thread blocks, which for a 2,304-channel hidden size is nine blocks on a 80-SM card. It now defaults to 128-thread blocks (`IK2_GATE_TILE`), 18 blocks of four warps. Paired A/B against the build below: 48.2 to 50.9 tok/s (+5.7%), identical generated tokens.

`nsys` on that configuration (300 decoded tokens, 22.2 ms/token) shows about 1,275 kernel launches a token. The largest shares: cuBLAS dense GEMVs 21.6% (217 launches), resident expert gate/up 16.1% (80), loop-gate apply 8.2% (47, 32 us each), split attention 7.8% (45, 31 us each), expert down 7.8%, tensor-core dense GEMMs 7.3%, routing 3.7%. The token moves about 4 GB of weights, so at V100 HBM speed the arithmetic floor is near 5 ms/token; the rest is that serial chain of 48 visits, whose small kernels are latency-bound rather than bandwidth-bound. The attention split launches 384 blocks at a 1,024-key window and is still four times off its bandwidth floor. A cheaper router kernel (`mv_f16` instead of the cuBLAS GEMV) was measured and rejected: it changed the generated tokens and ran 0.4 tok/s slower.

A per-device breakdown of that trace, over the 6.65 s of generation: GPU 0 is busy 3.99 s (60% of the wall) with 1,207 kernel launches a token; GPU 1 is busy 1.12 s (17%) with 96. So a token is limited by the host-side launch chain, not by either card's arithmetic: GPU 0 has 2.3 s of idle gaps inside the generation. `CUDA_DEVICE_MAX_CONNECTIONS=1` and `IK2_OVERLAP=0` were both neutral within noise (50.4–51.1 tok/s). Capturing the token in one CUDA graph, which is what the single-GPU resident path does, is not available here: each of the 48 visits interleaves the two cards, and a graph cannot contain work for the other device.

Dense projections can instead be read as the GGUF's Q8_0 blocks (`IK2_DENSE=q8`), which is what both profiles now use. It halves the dense bytes a token reads and drops cuBLAS from the projection path. Paired A/B: 50.4 to 55.2 tok/s (+9.5%). The generated token ids differ from the FP16-copy path because a row's dot product is then summed in FP32 over the int8 values rather than in FP16 over their dequantized copies; the block path is the more precise of the two. Answers checked: `9 + 8` returned `17`, and a one-sentence question about Python context managers returned a correct answer, 59 tok/s. Dual-GPU use of this path was previously refused outright; that refusal is gone.

With the dense weights read as blocks, the space freed also lets the prefill chunk grow: 1,536 keeps every expert resident (808 tok/s on the 11,252-token source prompt, no cache misses), while 2,048 reaches 870 tok/s but spills 95 experts and costs 3.7% of decode, and 2,560 reaches 897 tok/s for 237 misses. The profile keeps 1,536. Shifting expert ownership toward the idle card (`IK2_EP_SPLIT` above 0.85) is 4% slower on decode, because the second card's cache cannot then hold its share; 0.85 stays. Disabling the relative bias (`IK2_RELB=0`) changes decode by under 1%, so decode attention is bound by the key/value reads, not the bias table.

The DSpark drafter (`--draft`, Thinking Machines' `mtp.safetensors`) now runs over the expert-parallel pair behind `IK2_EP_DRAFT=1`: each card computes the verify window's slots it owns, from its own device-side table, with the token path's slot kernels. It does not pay yet, and stays opt-in. At a 131,072-token context it decodes 54.4 tok/s against 56.0 without it, with 3.1 tokens accepted a step. Two reasons: the window cannot be captured in a CUDA graph because its expert exchange is a peer copy, while the token path gets one; and its dense products use the block path's `q8_tile` rather than the token path's `mv_q8p`, so its ids differ from plain decoding's and its acceptance test is not the token path's. At 524,288 tokens the drafter's 0.85 GiB also pushes the expert cache down to 2,428 of 3,072 and the stream of copies costs an order of magnitude. Making it win needs per-device graphs for the window, which is a larger change than the pair's token path.

Splitting the dense projections across both cards by output row (`IK2_SPLIT_DENSE`) was implemented and measured: identical generated tokens, but 46.4 against 55.6 tok/s. Ten cross-device round trips a visit cost more than the parallel halves save, so it stays off. It would need one exchange a visit, not one a matrix.

That puts the remaining headroom in two places. Reducing GPU 0's own work: it runs every dense projection and all attention for all 48 visits while GPU 1 sits idle, so splitting those projections across both cards by output row (which leaves each row's dot product unchanged) would nearly halve the critical path. And cutting launches per visit: Maya reports 781 launches a token after fusion, against 1,404 before.

Three research passes over the other Volta Inkling runtimes agree on the kernel form: on `sm_70` a one-token matmul must be an FFMA or `dp4a` dot product reading the quantized blocks where they are. Native `mma.m8n8k4` pays only for prefill and multi-token verify, and `mma.m16n8k16` on Volta is emulated at roughly 6–9 TFLOP/s, which this port uses only in the prefill attention. NInfer-3090 measures 62–73 tok/s plain on a 32 GB V100 with that form.

Quantized KV on this pair was checked against the report that llama.cpp's Inkling path silently keeps FP16 copies when a quantized cache is asked for. Neither engine here has that defect. In Strata, FP8 KV is one byte a value: at 524,288 tokens the engine reports 8.31 GiB across the 48 layer visits and keeps all 3,072 experts resident, while the same run with `IK2_KV=f16` dies on the first KV allocation — FP16 does not fit on the pair at all. The only FP16 KV buffers are the temporary dequantisation pair, capped at 32,768 tokens. The pinned llama.cpp build behaves too: at a 131,072-token cache it used 10,346/10,444 MiB with `q8_0` against 11,142/11,212 with `f16`, decode unchanged at 26–27 tok/s, so the quantized cache already saves memory there. Unsloth's fix for the fallback is newer than the build pinned here (`b12588`, 2026-11-20; their master takes the Inkling layout change on 2026-11-21), so a future bump is worth re-measuring, not a defect to work around now.

The 21.3 GiB `UD-Q8_K_XL` is the highest-precision [Unsloth GGUF](https://unsloth.ai/docs/models/inkling-2-mini). The [original checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is already mixed NVFP4 and MXFP8; the GGUF name does not mean it promotes all the experts to 8-bit precision. This Strata engine does not load the BF16 experts in that file. The [Unsloth CUDA 12 llama.cpp build](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) supports the GGUF and splits the KV by layer instead of keeping it all on the first GPU.

RAM prompt caching can park idle sessions and reload them when they resume. It does not increase the GPU KV available for active requests. Several shorter requests can share a 524K token pool; two distinct full 524K sessions cannot both fit alongside Q8_K_XL. Keep active KV on the cards to avoid reading it over PCIe on each output token.

With llama.cpp b12588 on the same V100 pair, Q4_K_XL and Q8_0 KV initialized a shared 524,288-token pool with four slots. This used 15,752 and 15,770 MiB with a `14,11` layer split and `--ubatch-size 16`. The short arithmetic reply ran at 26.6 tokens/s. Two short requests ran at the same time at 23.7 tokens/s each. These tests did not fill the 524K pool. A 256-token microbatch failed to allocate its compute buffers.

For better weight fidelity, llama.cpp b12588 also ran `UD-Q8_K_XL` with both GPUs, a `17,8` layer split, and these settings:

| Context pool | K and V | Slots | Microbatch | Free VRAM (GPU 0 / 1) | Short decode |
| --- | --- | ---: | ---: | ---: | ---: |
| 196,608 | FP16 | 4 | 32 | 485 / 539 MiB | 27.4 tok/s solo; 19.5 each with two requests |
| 393,216 | Q8_0 | 4 | 8 | 167 / 13 MiB | 26.3 tok/s solo; too little headroom for daily use |
| 524,288 | Q5_0 | 2 | 4 | 399 / 65 MiB | 26.4 tok/s solo; 18.6 each with two requests |

Q8_K_XL with Q8_0 or FP8 KV failed to allocate the 524K compute buffers in the tested configurations. The Q5_0 configuration fits but has very little headroom; a 524K prompt was not tested. For Q8_K_XL with unquantized KV, start with the 196K settings:

```sh
./llama-server --model Inkling-2-Mini-UD-Q8_K_XL.gguf --jinja \
  --ctx-size 196608 --kv-unified --kv-unified-per-slot 196608 --parallel 4 \
  --n-gpu-layers 99 --split-mode layer --tensor-split 17,8 --fit off \
  --cache-type-k f16 --cache-type-v f16 --flash-attn on --batch-size 512 --ubatch-size 32 \
  --cache-ram 32768 --cache-idle-slots --host 127.0.0.1 --port 18000
```

The measured split uses PCI bus order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,1`). The 32 GiB RAM cache is for reuse of idle prompts, not part of the active token pool. Current 1Cat-vLLM does not list Inkling in its [model registry](https://github.com/1CatAI/1Cat-vLLM/blob/63b1eaf9b0cfd9d32d4d733b2b8aa96417838d5a/vllm/model_executor/models/registry.py).

## A second engine: `strata-inkling2`

The branch also carries the upstream Strata Inkling 2 engine path (`src/inkling2`, `strata-inkling2`) beside `strata-ik2`. It is the more deeply tuned of the two on Volta: measured on a 32 GB V100 with `UD-Q4_K_M`, `dspark`, int8 KV and 3,072 experts resident it reads 1,349 tok/s at a 512-token prompt and decodes 130 tok/s there, and holds 910 tok/s prefill / 83.5 tok/s decode at 524,288 tokens. It is one card's engine: it makes no per-device calls, so it does not split experts across the pair.

Its VRAM need is experts (13.76 GiB for this model) + dense + KV + workspace, so a 16 GB card is about 1 GiB short at a short context, and the engine refuses with `expert placement failed`. The two ways out are a 32 GB card, or its host expert pool, which this tree's CPU lane does not yet compute for SITU activations. The engine builds here as `strata-inkling2` (`-DSTRATA_INKLING2=ON`) and loads this GGUF; the port adds `include/strata/inkling2/`, `include/strata/quant/` and `src/inkling2/`, the `Inkling2` model family and its header check, and the four host-side quant entry points its loader uses (over the block decoder `strata-ik2` already carries). Two hooks the engine asks for are not present in this build: the grouped WMMA expert kernel answers "not available" so the engine takes its own fallback, and the BF16 tensor-core reservation is a no-op because this prefill has no such path.

## Which quant fits the pair at 524,288

The profile runs Unsloth's `UD-Q4_K_XL` (16.42 GiB, sha256 `e62d463c…`) with FP8 E4M3 KV at the full
524,288-token window and every one of the model's 3,072 experts in VRAM: 431 slots (2.1 GiB) on the first card
and 2,641 (12.6 GiB) on the second; the 14.7 GiB of experts are read from the GGUF through pinned host
memory and copied to the cards (about 10 s). The engine is ready 18 s after start; GPU 0 then holds 14.2 GiB of its 16.4 and GPU 1 15.1, leaving 2.2 and 1.3 GiB.

That file is not a uniform Q4: 20 of its 24 expert blocks are the checkpoint's NVFP4 experts (10.68 GiB), four
are Q8_0 (4.13 GiB), and the dense, attention and output tensors are Q8_0. `UD-Q8_K_XL` keeps the same NVFP4
blocks but stores the four high-precision blocks as BF16 (8.0 GiB against 4.13) and the dense tensors as BF16:
21.30 GiB. With 8.31 GiB of FP8 KV that is 29.6 GiB before any workspace, against 32 GiB of VRAM and the 3.5 GiB
this profile has free, so it cannot stay fully resident at 524,288; the engine also has no BF16 expert path.
llama.cpp b12588 ran it at 524,288 only with a Q5_0 cache and 65 MiB of headroom, and never with a prompt that
filled the window. `UD-Q4_K_XL` is therefore the highest-quality quant that fits fully on the pair at this window.

Unsloth's 2026-11-23 update moved the root GGUFs to mainline llama.cpp's `inkling2_loop` layout: the loop tensors
are renamed (`loop_q_ctrl_down` against `loop_q_down`), the metadata keys are `inkling2_loop.*` and the file has
744 tensors instead of 734. The tensor bytes are unchanged (the first 64 KiB of `output.weight` hashes equal in
both), and the Unsloth layout this engine reads is still published under `Old_Support/`. Load the
`Old_Support` file (rename the `.gguf_file`) or the earlier revision; the speed claims in that update are for
Unsloth's own prebuilt, not for a different quant.

## GQA decode attention

Eight of the 48 layer visits are full-attention layers, and Inkling 2 gives each KV head three query heads (24
heads on 8). `IK2_ATTN_GQA=1` (the expert-parallel token path) runs one block per KV head for those visits: a half-warp covers one key's 128
values eight apiece (four shuffle steps a dot instead of five) and the key's K and V are read once for all three
query heads, with the V load issued beside the K load. On a 64K prompt this cut the eight full visits from
about 10.1 to 6.1 ms a token (nsys: 768 us a visit) and decode from 35.5 to 42.2 tok/s (+19 %). Short-context
decode is unchanged (55.8 tok/s at 512 tokens) and the 128 greedy tokens of a 512-token prompt are identical
with the flag on and off. `IK2_ATTN_SPLIT` overrides the 128 splits a KV head; 512 measured 41.4 tok/s at 64K,
so the default stays.

## Deep prompts

A 262,144-token prompt with `IK2_FLASH=1` read at 178.7 tok/s (1,466.8 s) and then decoded 8 tokens at 28.5
tok/s at that depth; the same flag is slower at 64K (490 against 622 tok/s), so the profiles leave it off. A
524,000-token prompt with the default path did not finish its prefill inside 3,600 s, so a prompt that fills the
window is not practical yet. The flash path alone grows at about the 1.7 power of the prompt length (64K
133.7 s, 256K 1,466.8 s), which is why the full window costs close to an hour: the KV cache lives on the first
card, and the default path also unpacks each FP8 KV row to FP16 twice per chunk per full layer. A faster deep
prefill needs the upstream engine's FA2 prefill kernel in this path.

Quality at depth: a 35,467-token request with the code `4815162342` in a note at the middle of the document
returned `4815162342`.

## Volta path

Volta's native FP16 Tensor Core instructions replace the Ampere matrix instructions. Tile loads use Volta-compatible shared-memory loads and synchronous copies. The two GPUs split experts, and each cache is sized only for that GPU's share. Attention, dense weights and KV stay on the first visible GPU. The profile does not need NVLink.

Engine and template derived from [Maya Inkling 2 Mini](https://github.com/alesha-pro/maya-inkling-mini/tree/c520fd14de3b21d67ddc6b7e7348b761aad26d52) (MIT).
