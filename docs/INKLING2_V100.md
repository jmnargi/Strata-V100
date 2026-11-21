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

## 524,288 tokens and concurrent clients

Inkling has eight full-attention layer visits. At 524,288 tokens their KV alone uses 16 GiB in FP16, 8 GiB in FP8, or 8.5 GiB in Q8_0. The sliding-window states and workspace are extra. FP16 KV plus the 16.4 GiB Q4 GGUF already exceeds the pair's 32 GiB.

For one active session, use the same server command with `--config docs/inkling2-mini-v100-524k.json`. This profile allocated 524,288 tokens of FP8 KV and kept all 3,072 experts in VRAM on the V100 pair (431 on the first card, 2,641 on the second). It generated the short arithmetic answer at 49.6 tokens/s. A prompt with 11,252 tokens of Strata source read at 561.7 tokens/s with zero expert-cache misses. Put the x16 PCIe card second for this split. This verified allocation and generation, not a prompt that filled the 524K window.

The 524K profile enables grouped expert prefill (`IK2_BATCH_PREFILL=1`). On the same 11,252-token source prompt, an interleaved A/B/B/A run improved whole-prompt throughput from 564 to 592 tok/s (+5.0%). Expert-phase time fell from 3.02 to 2.20 s (27%). All four runs generated the same token IDs; decode stayed at about 42.7 tok/s.

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

## Volta path

Volta's native FP16 Tensor Core instructions replace the Ampere matrix instructions. Tile loads use Volta-compatible shared-memory loads and synchronous copies. The two GPUs split experts, and each cache is sized only for that GPU's share. Attention, dense weights and KV stay on the first visible GPU. The profile does not need NVLink.

Engine and template derived from [Maya Inkling 2 Mini](https://github.com/alesha-pro/maya-inkling-mini/tree/c520fd14de3b21d67ddc6b7e7348b761aad26d52) (MIT).
