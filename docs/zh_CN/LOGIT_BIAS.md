# 按请求设置 token 偏置

<a id="per-request-token-biases"></a>

在 CUDA 或 HIP 引擎支持该功能且未开启连续批处理时，`POST /v1/chat/completions` 接受 `logit_bias`。同一偏置
适用于该请求中生成的每个 token，包括投机解码（speculative decoding）期间一起检查的 token。它不会强制把验证
窗口限制为单个 token。

## 示例：Hello 变成 Hi

<a id="example-hello-becomes-hi"></a>

我们用 `Hello`，因为它是个熟悉的词，在被测的 tokenizer 中恰好是一个 token，而且无偏置的模型在这个提示词下
自然会选它。tokenizer 把 `Hello` 映射到 **token ID 9419**。这个数字标识该 token，并不是它的 logit 或概率。
值 **-100** 是所请求的偏置，本实现把它当作硬性排除。

之前：不带偏置发送这个请求。实测回答是 **Hello**。

```json
{
  "model": "bias-validation",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none"
}
```

之后：只加上 `logit_bias`。实测回答是 **Hi**（token 12675）。

```json
{
  "model": "bias-validation",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none",
  "logit_bias": {"9419": -100}
}
```

等价的 llama.cpp 风格对偶列表是 `"logit_bias": [[9419, false]]`；它同样产出了 **Hi**。在下一次请求中去掉这个
字段后，**Hello** 又回来了。默认不禁用任何东西。这是一个按请求的示例，不是内置列表。上面 `model` 的值是测试
服务器的别名；请使用你服务器的模型名。数值型对偶值也受支持，例如 `[[42, 2.5]]`。token ID 属于所加载模型的
tokenizer；示例中的 ID 不能通用到任意模型之间。前导空格和字母大小写也会改变 token ID。这四个英文请求是在
ISTA IQ3_XXS 上测得的；并附有[复现脚本](../../tools/logit_bias_english_example.py)。

## 语义与限制

<a id="semantics-and-limits"></a>

偏置必须是 `[-100, 100]` 范围内的有限数值。负值抑制某个 token，正值鼓励它，而 **-100 会排除它**（对偶列表中
的 `false` 含义相同）。偏置在重复/频率/存在惩罚、温度以及候选过滤之前加入。原始模型 logits 不会被覆盖。禁用
某个 token 并不等于禁用某个词或某种语言：其他 token 序列可能产出同样的文本。

省略该字段、`null`、`{}` 或 `[]` 会清除上一次请求的偏置。偏置是请求的采样状态；它们不会存入对话缓存。它们
不改变模型权重或 prompt 的 KV。偏置发生变化时会上传一个固定的稠密词表向量，并在该请求内复用。

格式错误的条目、对偶列表中的重复 ID、超出词表的 ID、超出范围的偏置，或者禁用整个词表，都会返回 HTTP 400。
聊天路由上的校验发生在 SSE 流式响应头发送之前。使用较旧的引擎、当前的 SYCL 引擎或连续批处理（`--batch`）时，
非空偏置也会返回 HTTP 400，而不是被静默忽略。存在该实现时，引擎会在它的 `INFO` 响应中声明 `logit_bias=1`。

MTP 目标验证会应用偏置；草稿模型不带偏置，所以限制性的列表可能降低草稿接受率。测试过的是普通 MTP。本次工作
没有验证实验性的耦合/拒绝采样组合以及多 GPU 执行。这次改动没有启动时的偏置文件选项：请随每个请求发送该字段。
