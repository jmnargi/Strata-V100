# V100 IQ3_S concurrency measurements

## Configuration

The fork contains batch-slot support. The tested server used engine version
0.1.40, two Tesla V100-PCIE-16GB cards, a Ryzen 5 3600, IQ3_S
Qwen3.8-Flash-Next, INT8 KV, a layer split at layer 16, vision, and MTP
with eight draft tokens. The driver was 580.178.04. The base commit was
`1cb28c3`. CPU-assisted prefill was set to `auto`.

The local service configuration now requests four slots and a 32,768-token
context. The server reports four serving slots. This replaces the previous
524,288-token, single-request configuration. The local configuration contains
credentials and is not part of this change. The service remains configured
for concurrency after the measurements.

## Method

Use `bench/run_v100_concurrency.py` against an idle server. Set `--model-gguf`
to the first model shard and `--out` to the results directory. The defaults
run client counts 1, 2, 3, and 4 twice at approximate prompt sizes 1K, 4K,
16K, and 32K. The script uses the existing benchmark authentication and
prompt helpers. It sends fresh prompt prefixes and starts each wave with a
thread barrier. Each request uses greedy sampling and a 128-token output
limit. The server includes token usage in its streamed response.

Raw results: [matrix.json](../bench/results/2026-10-07-v100-concurrency/matrix.json).
The file includes each response, token usage, first-token latency, completion
latency, and sampled slot state. All 80 requests completed with 128 output
tokens and the `length` finish reason. Samples showed two, three, and four
slots decoding in the corresponding client-count runs. One client uses the
solo path, not a decoding batch slot.

Throughput below is total output tokens divided by total wave duration across
the two repetitions. It includes prompt processing, admission, and completion.
The wave clock includes metric polling and up to approximately 0.5 seconds
of completion-detection delay. Latencies are means of individual request
measurements. The prompt counts include the server chat template.

| Actual prompt tokens | Clients | Output tok/s | Mean completion s | Mean first-token s |
| --- | ---: | ---: | ---: | ---: |
| 912–916 | 1 | 22.08 | 5.71 | 3.73 |
| 912–916 | 2 | 19.52 | 12.77 | 5.93 |
| 912–916 | 3 | 21.76 | 17.37 | 7.92 |
| 912–916 | 4 | 22.07 | 22.70 | 10.30 |
| 4,085–4,089 | 1 | 15.87 | 7.65 | 5.98 |
| 4,085–4,089 | 2 | 13.73 | 18.26 | 10.32 |
| 4,085–4,089 | 3 | 14.10 | 26.84 | 12.93 |
| 4,085–4,089 | 4 | 15.39 | 33.04 | 16.93 |
| 16,275–16,280 | 1 | 6.68 | 18.87 | 14.96 |
| 16,275–16,280 | 2 | 6.31 | 37.97 | 24.88 |
| 16,275–16,280 | 3 | 6.48 | 47.24 | 35.49 |
| 16,275–16,280 | 4 | 6.49 | 66.34 | 44.87 |
| 31,974–31,979 | 1 | 4.23 | 30.03 | 27.93 |
| 31,974–31,979 | 2 | 3.80 | 57.54 | 45.18 |
| 31,974–31,979 | 3 | 3.73 | 76.13 | 61.82 |
| 31,974–31,979 | 4 | 3.83 | 94.04 | 78.51 |

## Interpretation and limits

Concurrency works: multiple slots decode at the same time. It did not increase
end-to-end throughput in this workload. Four clients were approximately equal
to one at 1K, 3% lower at 4K, 3% lower at 16K, and 9% lower near 32K.
Individual requests took longer as the client count increased.

These measurements use four configured slots at every client count. They do
not compare a one-slot engine with a four-slot engine. They also do not isolate
decode-only speed or prove exact token parity between solo and batch paths.
Long fresh prompts and short outputs make prompt admission an important part
of the measured duration. Two repetitions do not establish statistical
significance. The 32K row leaves room for the chat template and generated
output; an exact 32,768-token input cannot fit with output in this context.

The targeted command `python -m unittest serve.test_slots tools.test_setup_parallel`
passed all 35 tests. It emitted resource warnings for existing telemetry file
reads. The real-model matrix is the runtime verification of the concurrent path.
