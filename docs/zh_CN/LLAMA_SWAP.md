# 在 llama-swap 后面运行 Strata

<a id="running-strata-behind-llama-swap"></a>

[llama-swap](https://github.com/mostlygeek/llama-swap) 把多个模型服务器放在一个地址后面，并启动请求所指的那个，
同时停掉其他的以释放 GPU。Strata 可以作为它的一个模型来工作。本页记录了我们让它跑起来所需要的设置；测试用的是
llama-swap 的默认停止行为、Strata 0.1.39（Tesla V100 上的 Coder IQ1_M），以及通过主机名访问这台机器的客户端。

## 条目

<a id="the-entry"></a>

先照常把 Strata 装好（`./setup.sh --no-start ...`），然后让 llama-swap 直接指向服务器。setup 写出的启动脚本
（`run-<model>.sh`）把端口固定为 8080，所以请改用 llama-swap 的端口去调用 `serve/server.py`：

```yaml
healthCheckTimeout: 300            # Strata takes ~1.5-2 min to load; give it room

models:
  strata:
    cmd: >-
      sh -c 'cd /path/to/Strata && exec .venv/bin/python serve/server.py
      --engine strata --config strata-coder-iq1_m.json --port ${PORT}'
    env: ["STRATA_ALLOWED_HOSTS=my-server"]     # the host name your clients use -- see below
    checkEndpoint: /health
    ttl: 1800
```

- `exec` 让 Python 服务器成为 llama-swap 所停止的那个进程。收到 llama-swap 的停止信号（SIGTERM）时，服务器会
  停掉它的引擎：在我们的测试中，引擎在 2 s 内退出，V100 的显存也释放了。
- `checkEndpoint: /health`——Strata 的 `/health` 只在模型加载完成后才应答。我们机器上的实测加载时间（模型文件
  在 SATA SSD 上）为 80 s；当时另有一个下载在占用磁盘时为 105 s。
- 如果其他模型共用这块 GPU，把它们放进 llama-swap 的分组，或让交换把它们卸载；Strata 需要独占显卡的显存。

## 陷阱：“Host ... is not allowed”

<a id="the-trap-host--is-not-allowed"></a>

第一个经由 llama-swap 的请求立刻失败，报：

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection)
```

Strata 只应答它认识的主机名，这样网页就无法通过把某个 DNS 名字指向 127.0.0.1 来访问它。llama-swap 会把客户端
的 Host 头原样透传，所以通过名字访问代理的客户端（`http://my-server:8040`）会被拒绝。用 IP 地址和访问
`localhost` 的请求总是可以通过，这就是为什么快速测试能成、而从另一台机器发起就会失败。以下任一做法都能解决：

- 在该条目的 `env` 中设置 `STRATA_ALLOWED_HOSTS=my-server`（逗号分隔的名字；端口会被忽略），或
- 在 `strata-<model>.json` 中写 `"allowed_hosts": ["my-server"]`，或
- 设置一个 `"api_key"`，它会关掉这项检查——之后每个客户端都必须发送该密钥。

## 通过 llama-swap 访问 Strata 自己的页面

<a id="stratas-own-pages-through-llama-swap"></a>

llama-swap 会原样转发 `/upstream/<model>/...`，所以 Strata 的状态页与健康页仍然可达，而且调用其中之一也会加载
模型，无需发送聊天请求：

```bash
curl http://my-server:8040/upstream/strata/health
curl http://my-server:8040/upstream/strata/v1/status
```

## Claude Code 与其他 Anthropic API 客户端：思考档位

<a id="claude-code-and-other-anthropic-api-clients-the-thinking-level"></a>

除非另有设置，Claude Code（以及 Claude Agent SDK）会随每个请求发送 `"output_config": {"effort": "high"}`，所以
Strata 以它最高的档位思考（把请求记入日志即可看到）。在客户端的环境里设置 `CLAUDE_CODE_EFFORT_LEVEL=medium`
（或 `low`）可以改变它。一次测量，仅供估量量级：在我们的环境里，一个真实的编码任务，`medium` 用时 6.5 min，
而 `high` 为 9.7 min，两者都正确；`low` 用了 10.1 min（更多试错修正的步骤）。各只跑了一次——不是普遍结论。
