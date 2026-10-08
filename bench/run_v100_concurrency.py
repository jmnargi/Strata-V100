#!/usr/bin/env python3
"""Measure streamed HTTP throughput and latency at several client counts.

Run against an idle server. Client concurrency does not change engine slot count.
Fresh prompt prefixes prevent conversation-prefix reuse. Raw rows are saved after
 each wave so an interrupted run retains completed measurements.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import threading
import time
from pathlib import Path

import requests
from run_v100_bench import api_key, fresh_head, UNIT, positive_int


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='http://127.0.0.1:8088')
    ap.add_argument('--model', default='qwen')
    ap.add_argument('--targets', default='1024,4096,16384,32256', help='Approximate prompt sizes; actual usage is recorded')
    ap.add_argument('--clients', default='1,2,3,4')
    ap.add_argument('--repeats', type=positive_int, default=2)
    ap.add_argument('--max-tokens', type=positive_int, default=128)
    ap.add_argument('--seed', type=int, default=20261007)
    ap.add_argument('--model-gguf', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from run_v100_bench import Tokenizer
    tok = Tokenizer.from_gguf(args.model_gguf)
    key = api_key()
    def session():
        s = requests.Session()
        s.headers['Authorization'] = 'Bearer ' + key
        return s
    def get(path):
        with session() as s:
            r = s.get(args.base + path, timeout=30)
            r.raise_for_status()
            return r.json()
    status = get('/v1/status')
    if status['activity']['in_flight']:
        raise RuntimeError('Server is not idle')
    targets = [positive_int(v) for v in args.targets.split(',')]
    clients = [positive_int(v) for v in args.clients.split(',')]
    context = status['context']['max_positions']
    if max(targets) + args.max_tokens + 256 > context:
        raise ValueError('Prompt, template allowance and output exceed server context')
    unit_n = len(tok.encode(UNIT))
    rng = random.Random(args.seed)
    result = {'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('model_gguf', 'base', 'out')},
              'server': {'engine': status['engine'], 'concurrency': status['concurrency'], 'context': context}, 'waves': []}
    args.out.mkdir(parents=True, exist_ok=True)
    def complete(content, barrier):
        with session() as s:
            barrier.wait()
            start = time.perf_counter()
            first = None
            usage = None
            finish = None
            pieces = []
            with s.post(args.base + '/v1/chat/completions', json={
                'model': args.model, 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'max_tokens': args.max_tokens, 'stream': True,
                'stream_options': {'include_usage': True}}, stream=True, timeout=(30, 1800)) as r:
                r.raise_for_status()
                for line in r.iter_lines(chunk_size=1):
                    if not line.startswith(b'data: '):
                        continue
                    payload = line[6:]
                    if payload == b'[DONE]':
                        break
                    event = json.loads(payload)
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        text = delta.get('content') or delta.get('reasoning_content') or ''
                        if text:
                            if first is None:
                                first = time.perf_counter() - start
                            pieces.append(text)
                        finish = choice.get('finish_reason') or finish
            if usage is None or first is None or finish is None:
                raise RuntimeError('Incomplete stream or missing token usage')
            return {'wall_s': time.perf_counter() - start, 'ttft_s': first,
                    'usage': usage, 'finish_reason': finish, 'text': ''.join(pieces)}
    for target in targets:
        for count in clients:
            for repeat in range(args.repeats):
                prompts = [fresh_head(rng) + UNIT * max(1, (target - 64) // unit_n) +
                           '\nExplain the implementation tradeoffs in detail.' for _ in range(count)]
                barrier = threading.Barrier(count)
                samples = []
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=count) as pool:
                    jobs = [pool.submit(complete, p, barrier) for p in prompts]
                    while not all(j.done() for j in jobs):
                        metrics = get('/metrics')
                        samples.append({'elapsed_s': time.perf_counter() - start, 'live': metrics.get('live')})
                        time.sleep(0.5)
                    rows = [j.result() for j in jobs]
                wall = time.perf_counter() - start
                tokens = sum(r['usage']['completion_tokens'] for r in rows)
                wave = {'target': target, 'clients': count, 'repeat': repeat,
                        'wall_s': wall, 'output_tokens': tokens, 'output_tok_s': tokens / wall,
                        'requests': rows, 'samples': samples}
                result['waves'].append(wave)
                (args.out / 'matrix.json').write_text(json.dumps(result, indent=2) + '\n')
                print(f'prompt~{target} clients={count} repeat={repeat} output={tokens} wall={wall:.2f}s throughput={tokens / wall:.2f} tok/s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
