# 图像生成（Qwen-Image-2.1-Turbo）

<a id="image-generation-qwen-image-21-turbo"></a>

Strata 的服务器也能生成图像：`POST /v1/images/generations`，也就是 OpenAI 的图像 API。模型是
[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)（8 steps，CFG 1），使用
[Unsloth 的 GGUF 量化版](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF)，由
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)（`sd-cli`）运行，setup 会为你的 GPU 编译它。
它是可选功能，除非你主动要求，否则处于关闭状态。

该模型采用 **Qwen Research License**，而不是 Apache 2.0：在把生成的图像用于商业用途之前，请先阅读
[它的许可证](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE)。

## 设置

<a id="set-it-up"></a>

    ./setup.sh --setup --images qwen-image          (Linux)
    START-HERE.bat --setup --images qwen-image      (Windows)

或者，对已经完成设置的安装，不必再跑一遍完整 setup：

    .venv/bin/python tools/image/install.py --config strata-<model>.json

Setup 会把 `sd-cli`（固定在某个 commit 上的 stable-diffusion.cpp）编译进 `engine-image/`，把三个文件下载到数据
文件夹的 `models/qwen-image-2.1-turbo/` 并逐个校验 SHA-256，还会往该模型的运行配置里写入一个 `"images"` 段。

| 部件 | 文件 | 大小 |
|---|---|---|
| 降噪器（按显存选择） | 8 GB 及以下用 `qwen-image-2.1-turbo-Q4_K_M.gguf`，12 GB 及以下用 `Q6_K`，再往上用 `Q8_0` | 4.2 / 6.3 / 7.6 GB |
| 文本编码器 | `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` | 5.1 GB |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB |

选项：`--image-quant Q4_K_M|Q6_K|Q8_0`（比推荐值更大的会保留，并给出警告）、
`--image-backend cuda|hip|vulkan|cpu`、`--image-gpu-policy auto|keep|shrink|unload`、`--images no`（把该段
从配置中去掉）。

后端：NVIDIA 上用 CUDA（需要 CUDA Toolkit）；AMD 独立显卡上用 HIP（Linux，与引擎相同的 ROCm）；AMD APU/iGPU
（Radeon 780M、Strix Point/Halo）、Intel Arc 以及 Windows 下的 AMD 用 Vulkan。缺少 Vulkan 开发文件的 Linux PC
会把 LunarG Vulkan SDK 解包到 `third_party/`（无需 sudo）；Windows 上请先安装 Vulkan SDK。Vulkan 下 setup 会
加上 `--vae-tiling --vae-tile-overlap 0.25`：在 Radeon 780M 上，不分块的 1024x1024 VAE decode（解码）是一段长到
足以触发内核驱动作业超时的 GPU 工作（出不了图），而四分之一重叠的解码耗时只有 sd.cpp 默认二分之一重叠的一半。

## 使用

<a id="use-it"></a>

    curl http://127.0.0.1:8080/v1/images/generations -H "Content-Type: application/json" \
      -d '{"prompt": "a cartoon sloth mascot waving, flat vector illustration", "size": "1024x1024", "seed": 42}'

| 字段 | |
|---|---|
| `prompt` | 描述（必填） |
| `size` | `WIDTHxHEIGHT`，每边取 256 到 2048 之间 32 的倍数（`max_side`）；默认值与 `auto`：`1024x1024` |
| `n` | 图像数量，1 到 4（`max_n`）；第 *i* 张图使用 seed + *i* |
| `response_format` | `b64_json`（默认）或 `url`（`/v1/images/files/<random name>.png`；服务器保留最新的 64 个） |
| `seed` | Strata 专有：用整数获得可复现的图像（默认随机；回答里会说明用的是哪个） |
| `steps` | Strata 专有：8（Turbo 的调度表）。其他数值会改用 stable-diffusion.cpp 自己的调度表，而 Turbo 并不是为它训练的 |

回答是 OpenAI 的结构（`created`、`data[].b64_json` 或 `data[].url`、`revised_prompt` = 提示词），另外还有
`data[].seed` 和一个给出秒数、步数与后端的 `strata` 对象。这些设置取自模型卡为 stable-diffusion.cpp 给出的
值：8 steps、CFG 1、euler、Turbo 自己的 sigmas、`--diffusion-fa`。通过 API 得到的图像，与用相同参数直接运行
`sd-cli` 写出的 PNG 逐字节相同。

## 与文本模型共享 GPU

<a id="sharing-the-gpu-with-the-text-model"></a>

一次只跑一个作业：图像作业与聊天请求轮流占用，所以聊天请求要等图像完成，反之亦然。文本模型在此期间的行为由
配置中的 `images.gpu_policy` 决定：

| 策略 | 图像作业期间的文本模型 |
|---|---|
| `auto`（默认） | 图像作业能与之并存时（APU 总是如此）用 `keep`；否则在引擎以 `--vram-elastic` 启动时用 `shrink`；再否则用 `unload` |
| `keep` | 保持加载并空闲 |
| `shrink` | 从它的专家缓存中让出显存（与 `POST /v1/vram` 相同），之后收回；对话保留 |
| `unload` | 对话保存到文件，引擎停止，下次启动时恢复对话（`"reload": "eager"` 会在图像之后立刻重启它；默认 `lazy`：等下一个聊天请求） |

每张图都运行自己的 `sd-cli` 进程，所以图像之间 GPU 上不会残留图像内存。从页缓存加载这三个文件，每张图需要
几秒钟。

## 速度

<a id="speed"></a>

通过 API 得到每张图的秒数，Q8_0，8 steps（每张图一个 `sd-cli` 进程，含加载时间）：

| GPU | 后端 | 1024x1024 | 512x512 |
|---|---|---|---|
| RTX A4000 16 GB | CUDA | 27 | 8.8 |
| Radeon 780M（iGPU） | HIP | 131 | 29 |
| Radeon 780M（iGPU） | Vulkan | 179 | 43 |

在 780M 上 HIP 更快，但走的是 ROCm 路径，在当前固件下它的队列驱逐会重置这块 GPU（这是驱动/固件缺陷，不是本
功能的问题；这些图像运行中没有出现重置）；Vulkan 在那里的 setup 默认后端，`--image-backend hip` 可选择 HIP。
