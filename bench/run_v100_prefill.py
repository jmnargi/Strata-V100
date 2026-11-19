#!/usr/bin/env python3
"""GENI prefill benchmark for the Strata V100 fork.

This module measures prefill latency and throughput with exact token ids. It speaks the
engine's own stdin/stdout serve protocol (src/program/generate.cpp, the --serve block). It
sends the token ids verbatim. Thus the requested prompt length equals the engine's prompt
length: no template, no probe, and no calibration.

The HTTP harness bench/run_v100_bench.py calibrates template overhead and token marginals
with extra probe requests. This module does not calibrate. Use this module for exact-length
arms.

PROTOCOL (generate.cpp --serve):
  in : GEN  <max_new> [key=val ...] <id,id,...>
       GENI <max_new> <embeddings_file> [key=val ...] <id,id,...>
       STOP / QUIT
  out: INFO key=value ...            (once, before READY)
       READY <max_context> stop
       RESUME <n>                    prompt tokens taken from a checkpoint (0 = none)
       PP <reached> <total> <ms> <fresh tok/s>   one per prompt chunk
       REUSED <n>                    the prompt is read
       T <token_id>                  one generated token
       DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered>
            <reused> [hits] [lookups] [ram blobs] [file blobs] [file MB] [prompt read]
       ERR <message>

Text-only requests use GENI with an empty embeddings file. The file holds 0 strata-vision
records. The GENI parser accepts it and keeps the identity M-RoPE table. Thus the --vision
reservation stays byte-identical to a production run without starting strata-vision. Use
--request-mode gen when the arm binary has no --vision.

REUSE. Every cold request is a distinct, seeded prompt. The first tokens carry a unique tag. The
engine starts with --prompt-cache 0, --prompt-cache-every 0, --prompt-cache-root 0 and
--conversation-cache-mib 0, and without any --conversation-cache-disk* flag. Thus resume is
structurally 0. Every row asserts reused == 0 and aborts on a violation.

CHAINED AGENTIC MODE (--chain).  An agent sends the whole conversation again with every turn.  This
mode measures that shape directly, with no HTTP layer:

  * one prefix turn reads a deterministic prompt of --chain-prefix tokens (default 8192);
  * then one turn per --chain-diffs value (default 0,1,32,64,128,256,512,1024,2048).  Each turn adds
    exactly that many tokens, so every prompt is a prefix of the one before it.

The prompts are prefixes of one deterministic token stream, so the requested sizes are exact and the
same for every arm.  The requested diff is NOT the read count.  The engine reports what it reused
(DONE <reused>) and what it read (DONE <prompt read>).  This module records both and never equates
them.  The first chain diff is 0: that turn sends the prefix turn's identical prompt again.  A resume
stops at most one token short of the prompt, so the engine re-reads the tokens above its newest
checkpoint and the row keeps the requested 0 next to the actual read.

Prompt caches are ON in this mode.  --prompt-cache and --prompt-cache-every are set for the arm (see
--chain-cache-every), --prompt-cache-root is 0 (the synthetic prompts hold no chat turn token), and
--conversation-cache-mib keeps the config's value.  A conversation disk store the config names is
pointed at a private directory under --out and removed before the run, so the arm never reads or
writes the production store; --chain-no-disk drops it.  --expert-profile-save is dropped for the same
reason.  --short-read N, --pipeline-windows N, --xargs and --env pass an arm's own knobs.
pin=n-2 is sent with every request by default (the engine ignores pin >= n-1), so the turn after it
re-reads only the last two tokens; --no-chain-pin-tail turns the pin off and shows the checkpoint gap.

Usage (the cold sweep, unchanged):
  python bench/run_v100_prefill.py --arm-label baseline \\
      --exe /tmp/strata-prefill-baseline --targets 2048,4096,8192,16384,32768 --repeats 1

Diagnostic arm (phase instrumentation; timings are NOT comparable to the clean arm):
  python bench/run_v100_prefill.py --arm-label baseline-timing \\
      --exe /tmp/strata-prefill-baseline --repeats 1 --env STRATA_PREFILL_TIMING=1

Chained arms (one engine each; the same plan and the same prompts in every arm):
  python bench/run_v100_prefill.py --chain --arm-label chain-baseline --exe /tmp/strata-prefill-baseline \\
      --out /tmp/strata-agentic-prefill/chain-baseline
  python bench/run_v100_prefill.py --chain --arm-label chain-own512 --exe /tmp/strata-prefill-baseline \\
      --env STRATA_SPLIT_SMALL_OWN=512 --env STRATA_SPLIT_SMALL_MAX=2048 \\
      --out /tmp/strata-agentic-prefill/chain-own512
  python bench/run_v100_prefill.py --chain --arm-label chain-pipeline1 --exe /tmp/strata-prefill-baseline \\
      --pipeline-windows 1 --short-read 128 --out /tmp/strata-agentic-prefill/chain-pipeline1
  python bench/run_v100_prefill.py --compare \\
      /tmp/strata-agentic-prefill/chain-baseline/result-chain-baseline.json \\
      /tmp/strata-agentic-prefill/chain-own512/result-chain-own512.json
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import queue
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

# The only lengths this project measures (assignment: 2048,4096,8192,16384,32768), plus the
# 64K row the V100 profile's context ceiling asks for.
ALLOWED_TARGETS = (2048, 4096, 8192, 16384, 32768, 49152, 65536)

IMAGE_PAD = 248056   # qwen4exp.ple.image_token_id (generate.cpp kImagePad)

# Chained-agentic mode (--chain): one prefix turn, then one turn per diff.  The first diff is 0: that
# turn sends the prefix turn's identical prompt again.
CHAIN_DIFFS = "0,1,32,64,128,256,512,1024,2048"
CHAIN_PREFIX = 8192
CHAIN_POOL_MARGIN = 4096
# Arguments that write a file another run would read: never used by an arm.
CHAIN_DROPPED_ARGS = ("--expert-profile-save", "--expert-profile-save-every")
# The DONE line's fields, in the engine's own order (generate.cpp).
DONE_FIELDS = ("generated", "prompt_tokens", "prompt_ms", "decode_ms", "finish", "draft_accepted",
               "draft_offered", "reused", "hits", "lookups", "ram_blobs", "file_blobs", "file_mb",
               "prompt_read", "offloaded", "chain_accepted", "chain_offered", "suffix_accepted",
               "suffix_offered", "windows")
# Recorded once in every chained result: the requested size is not a read count.
CHAIN_NOTE = ("The requested diff is the tokens a turn adds to its prompt.  It is not a read count: "
              "every row records the engine's own DONE <reused> and DONE <prompt read>, and this file "
              "never equates the requested diff with either.")

# The live runtime settings the arms must share (strata-q2_0.json + the user service env).
REQUIRED_ARG_PAIRS = (
    ("--kv", "int8"),
    ("--max-context", "524288"),
    ("--rope-scaling", "yarn"),
    ("--rope-scale", "2"),
    ("--spec", "8"),
    ("--vram-reserve-mib", "700"),
    ("--layer-split", "20"),
    ("--prefill", "auto"),
    ("--expert-cache", "auto"),
)
REQUIRED_FLAGS = ("--vision", "--mtp", "--expert-profile")
REQUIRED_ENV = {"STRATA_EXPERT_PAIR": "1"}
# The user service's Environment= lines (runtime consistency; overridable with --env).
SERVICE_ENV = {"STRATA_ARENA_LOCK": "0", "STRATA_WATCHDOG_S": "300"}
# A/B diagnostic knobs that must never leak into a clean timing arm.
DIAGNOSTIC_ENV = ("STRATA_PREFILL_TIMING", "STRATA_TRACE", "STRATA_DECODE_TIMING", "STRATA_SPLIT_TIMING",
                  "STRATA_VERIFY_PROFILE")

NVIDIA_FIELDS = (
    "index", "temperature.gpu", "clocks.current.sm", "power.draw", "enforced.power.limit",
    "clocks_throttle_reasons.active", "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown", "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
)
NVIDIA_MIN_FIELDS = ("index", "temperature.gpu", "clocks.current.sm", "power.draw")

WORDS = (
    "attention cache layer kernel tensor token sequence prefill decode batch stride block "
    "memory bandwidth latency throughput shard expert router projection residual norm rotary "
    "context window prompt sample greedy verify draft acceptance queue pipeline stream slot "
    "index table hash map reduce scan prefix suffix segment scratch barrier launch grid warp "
    "the model reads a prompt in chunks and writes the key value state into the cache while "
    "each layer attends over the positions it has already seen and then projects the result "
    "a deterministic transform maps an integer through a modulo and returns the remainder"
).split()


# --------------------------------------------------------------------------- PII
def make_scrubber() -> "callable":
    home = str(Path.home())
    user = os.environ.get("USER") or getpass.getuser()

    def scrub(value):
        if not isinstance(value, str):
            return value
        value = value.replace(home, "$HOME")
        if user:
            value = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(user) + r"(?![A-Za-z0-9_])", "$USER", value)
        return value

    return scrub


SCRUB = make_scrubber()


def sanitize(obj):
    """Recursively replace the home directory and user name in every recorded string."""
    if isinstance(obj, str):
        return SCRUB(obj)
    if isinstance(obj, dict):
        return {SCRUB(k) if isinstance(k, str) else k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- tokenizer
def load_tokenizer(directory: Path):
    from strata_tokenizer import Tokenizer
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    merges = (directory / "merges.txt").read_text().splitlines()
    types = json.loads((directory / "token_type.json").read_text())
    return Tokenizer(tokens, merges, types)


def corpus_text(rng: random.Random, lines: int) -> str:
    out = []
    for i in range(lines):
        kind = rng.randrange(5)
        if kind == 0:
            out.append(f"def task_{i:05d}(value: int) -> int: return (value * {rng.randrange(1, 97)} "
                       f"+ {rng.randrange(0, 100003)}) % 100003")
        elif kind == 1:
            out.append(" ".join(rng.choice(WORDS) for _ in range(rng.randrange(9, 22))) + ".")
        elif kind == 2:
            out.append(f"// buffer {i:05d}: stride={rng.randrange(2, 64)} offset={rng.randrange(0, 4096)} "
                       f"flags=0x{rng.randrange(0, 1 << 32):08x} count={rng.randrange(1, 1 << 16)}")
        elif kind == 3:
            out.append(f"SELECT id, name, score FROM records WHERE score > {rng.randrange(0, 1000)} "
                       f"ORDER BY score DESC LIMIT {rng.randrange(1, 50)};")
        else:
            out.append(f"Stage {i:05d}: the layer attends over {rng.randrange(1, 4096)} positions, "
                       f"reduces {rng.randrange(1, 64)} heads and writes {rng.randrange(1, 128)} rows "
                       f"into slot {rng.randrange(0, 1 << 20)} with weight {rng.random():.6f}.")
    return "\n".join(out) + "\n"


def build_pool(tokenizer, seed: int, need: int, cache_path: Path | None) -> list[int]:
    """A deterministic pool of realistic text token ids, at least `need` long.

    The pool is a prefix of one deterministic corpus stream, so the same (seed, need) always
    yields the same ids.  A cache written for a DIFFERENT need is rejected: accepting a longer
    pool would move every prompt slice and break the cross-arm prompt-hash parity."""
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if (cached.get("seed") == seed and cached.get("need") == need
                    and len(cached.get("ids", [])) >= need):
                return cached["ids"]
        except (OSError, ValueError):
            pass
    rng = random.Random(seed)
    ids: list[int] = []
    lines = 64
    while len(ids) < need:
        ids.extend(tokenizer.encode(corpus_text(rng, lines), parse_special=False))
        lines = max(lines, int(lines * 1.5))
    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps({"seed": seed, "need": need, "ids": ids}))
        except OSError:
            pass
    return ids


def build_prompt(tokenizer, pool: list[int], seed: int, target: int, repeat: int) -> list[int]:
    """Exactly `target` realistic token ids, identical for the same (seed, target, repeat) across arms."""
    tag = f"[bench {seed:08x} t{target} r{repeat}] "
    prefix = tokenizer.encode(tag, parse_special=False)
    if len(prefix) >= target:
        raise SystemExit(f"target {target} is too small for the request tag")
    span = len(pool) - target
    if span < 0:
        raise SystemExit(f"token pool is shorter than target {target}")
    h = int.from_bytes(hashlib.sha256(f"{seed}:{target}:{repeat}".encode()).digest()[:8], "big")
    start = h % span
    body = pool[start:start + (target - len(prefix))]
    ids = prefix + body
    if len(ids) != target:
        raise SystemExit(f"prompt build produced {len(ids)} ids, wanted {target}")
    if IMAGE_PAD in ids:   # GENI reserves this id for image cells; natural text must never produce it
        raise SystemExit(f"prompt for target {target} contains the image-pad token {IMAGE_PAD}")
    return ids


def parse_diffs(value: str) -> list[int]:
    """The --chain-diffs list: non-negative token counts, in order."""
    out: list[int] = []
    for part in value.replace(" ", "").split(","):
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            raise SystemExit(f"--chain-diffs wants integers, got {part!r}")
        if n < 0:
            raise SystemExit(f"--chain-diffs wants non-negative counts, got {n}")
        out.append(n)
    if not out:
        raise SystemExit("--chain-diffs is empty")
    return out


def chain_plan(prefix: int, diffs: list[int], rounds: int = 1) -> list[dict]:
    """The chained schedule: the prefix turn, then one turn per diff that extends the prompt by it.

    Every prompt is a prefix of the next one, so each turn's requested size is exact.  The first diff
    is normally 0: that turn sends the prefix turn's identical prompt again."""
    plan: list[dict] = []
    for rnd in range(rounds):
        plan.append({"round": rnd, "turn": 0, "phase": "prefix", "requested_diff": prefix,
                     "prompt_tokens": prefix})
        cumulative = 0
        for i, diff in enumerate(diffs):
            cumulative += diff
            plan.append({"round": rnd, "turn": i + 1, "phase": "chain", "requested_diff": diff,
                         "prompt_tokens": prefix + cumulative})
    return plan


def chain_pool_need(prefix: int, diffs: list[int], margin: int = CHAIN_POOL_MARGIN) -> int:
    """The pool length the chain needs: the prefix, every extension, and a margin."""
    return prefix + sum(diffs) + margin


def chain_pin(prompt_tokens: int, pin_tail: bool) -> int | None:
    """The `pin` value a chained request sends, or None.

    The engine reads the prompt in two parts at `pin` and pins the checkpoint there, so the next
    request that resumes from it re-reads only what lies above `pin`.  generate.cpp ignores
    `pin >= n - 1` ("the prefix must be shorter than the prompt"), so n - 2 is the largest honoured
    value; n - 2 is therefore the one that makes the turn after it re-read the fewest tokens."""
    return (prompt_tokens - 2) if (pin_tail and prompt_tokens >= 3) else None



def build_chain_ids(head: list[int], pool: list[int], prompt_tokens: int) -> list[int]:
    """Exactly `prompt_tokens` ids: the request tag, then the first tokens of the deterministic pool.

    The pool is one stream, so a longer prompt starts with every shorter one.  That nesting is what
    lets the engine reuse a checkpoint instead of reading the whole prompt again."""
    if len(head) >= prompt_tokens:
        raise SystemExit(f"chain prompt {prompt_tokens} is too small for the request tag")
    need = prompt_tokens - len(head)
    if need > len(pool):
        raise SystemExit(f"token pool is shorter than the chain needs ({need} > {len(pool)})")
    ids = head + pool[:need]
    if len(ids) != prompt_tokens:
        raise SystemExit(f"chain prompt build produced {len(ids)} ids, wanted {prompt_tokens}")
    if IMAGE_PAD in ids:
        raise SystemExit(f"chain prompt of {prompt_tokens} tokens contains the image-pad token {IMAGE_PAD}")
    return ids


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- telemetry
class GpuSampler:
    """1 Hz nvidia-smi sampler: temperature, SM clock, power, power limit, thermal slowdown."""

    def __init__(self, gpus: list[int], interval: float, csv_path: Path):
        self.gpus = [str(g) for g in gpus]
        self.interval = interval
        self.csv_path = csv_path
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fields = NVIDIA_FIELDS
        self._csv = None

    def start(self):
        self._csv = self.csv_path.open("w", encoding="utf-8")
        self._csv.write("epoch_s,mono_s,gpu,temp_c,sm_mhz,power_w,power_limit_w,"
                        "throttle_active,sw_thermal,hw_thermal,hw_slowdown,sw_power_cap\n")
        self._csv.flush()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._csv is not None:
            self._csv.close()

    def _query(self) -> list[str] | None:
        for fields in (self._fields, NVIDIA_MIN_FIELDS):
            cmd = ["nvidia-smi", "-i", ",".join(self.gpus),
                   "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired) as e:
                self.errors.append(f"nvidia-smi failed: {e}")
                return None
            if out.returncode == 0:
                self._fields = tuple(fields)
                return [ln for ln in out.stdout.splitlines() if ln.strip()]
            self.errors.append(f"nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
        return None

    @staticmethod
    def _num(value: str):
        value = value.strip()
        if value in ("", "N/A", "[N/A]", "Not Active", "Active"):
            return value if value in ("Not Active", "Active") else None
        try:
            return float(value)
        except ValueError:
            return None

    def _sample_once(self):
        rows = self._query()
        if rows is None:
            return
        now = time.monotonic()
        epoch = time.time()
        with self._lock:
            for row in rows:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) < 4:
                    continue
                gpu = parts[0]
                temp, sm, power = self._num(parts[1]), self._num(parts[2]), self._num(parts[3])
                limit = self._num(parts[4]) if len(parts) > 4 else None
                active = parts[5] if len(parts) > 5 else ""
                sw_thermal = parts[6] if len(parts) > 6 else ""
                hw_thermal = parts[7] if len(parts) > 7 else ""
                hw_slow = parts[8] if len(parts) > 8 else ""
                sw_cap = parts[9] if len(parts) > 9 else ""
                sample = {"t": now, "epoch": epoch, "gpu": gpu, "temp_c": temp, "sm_mhz": sm,
                          "power_w": power, "power_limit_w": limit, "throttle_active": active,
                          "sw_thermal": sw_thermal, "hw_thermal": hw_thermal,
                          "hw_slowdown": hw_slow, "sw_power_cap": sw_cap}
                self.samples.append(sample)
                self._csv.write(f"{epoch:.3f},{now:.3f},{gpu},{temp},{sm},{power},{limit},"
                                f"{active},{sw_thermal},{hw_thermal},{hw_slow},{sw_cap}\n")
            self._csv.flush()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as e:                       # a sampler must never kill a run
                self.errors.append(f"sample error: {e}")
            self._stop.wait(self.interval)

    def latest_temps(self) -> dict[str, float] | None:
        with self._lock:
            if not self.samples:
                return None
            last = {}
            for s in reversed(self.samples):
                if s["gpu"] not in last:
                    last[s["gpu"]] = s
                if len(last) == len(self.gpus):
                    break
            temps = {}
            for g, s in last.items():
                if s["temp_c"] is None:
                    return None
                temps[g] = float(s["temp_c"])
            return temps

    def stats(self, t0: float, t1: float) -> dict:
        with self._lock:
            picked = [s for s in self.samples if t0 <= s["t"] <= t1]
        out: dict[str, dict] = {}
        for gpu in self.gpus:
            rows = [s for s in picked if s["gpu"] == gpu]
            if not rows:
                out[gpu] = {"samples": 0}
                continue
            temps = [s["temp_c"] for s in rows if s["temp_c"] is not None]
            powers = [s["power_w"] for s in rows if s["power_w"] is not None]
            clocks = [s["sm_mhz"] for s in rows if s["sm_mhz"] is not None]
            thermal = [s for s in rows if s["sw_thermal"] == "Active" or s["hw_thermal"] == "Active"]
            out[gpu] = {
                "samples": len(rows),
                "min_temp_c": min(temps) if temps else None,
                "max_temp_c": max(temps) if temps else None,
                "mean_temp_c": round(statistics.fmean(temps), 2) if temps else None,
                "max_power_w": max(powers) if powers else None,
                "max_sm_mhz": max(clocks) if clocks else None,
                "thermal_slowdown_samples": len(thermal),
                "max_throttle_mask": max((s["throttle_active"] for s in rows), default=""),
            }
        return out


def wait_cooldown(sampler: GpuSampler, target_c: float, stable_s: float, max_s: float, poll_s: float) -> dict:
    """Wait until every card is <= target_c continuously for stable_s; bounded by max_s."""
    t0 = time.monotonic()
    stable_since = None
    last_temps = None
    while True:
        temps = sampler.latest_temps()
        now = time.monotonic()
        if temps is None:
            if now - t0 >= max_s:
                return {"met": False, "waited_s": round(now - t0, 1), "temps_c": None,
                        "reason": "no telemetry"}
            time.sleep(poll_s)
            continue
        last_temps = temps
        ok = all(t <= target_c for t in temps.values())
        if ok:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_s:
                return {"met": True, "waited_s": round(now - t0, 1), "temps_c": temps}
        else:
            stable_since = None
        if now - t0 >= max_s:
            return {"met": False, "waited_s": round(now - t0, 1), "temps_c": last_temps,
                    "reason": "bounded wait exhausted"}
        time.sleep(poll_s)


# --------------------------------------------------------------------------- engine io
class EngineIO:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            self.q.put(None)

    def next_line(self, timeout: float) -> str:
        try:
            item = self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            raise TimeoutError("no protocol line within the timeout")
        if item is None:
            raise EOFError("the engine closed its stdout")
        return item


def wait_ready(io: EngineIO, timeout: float, on_line) -> dict:
    info = {}
    t0 = time.monotonic()
    while True:
        line = io.next_line(timeout - (time.monotonic() - t0))
        on_line(line)
        if line.startswith("INFO "):
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                info[k] = int(v) if v.lstrip("-").isdigit() else v
        elif line.startswith("READY"):
            return info
        elif line.startswith("ERR "):
            raise RuntimeError(f"engine error during startup: {line[4:]}")


def run_request(io: EngineIO, request: str, timeout: float, on_line) -> dict:
    io.proc.stdin.write(request + "\n")
    io.proc.stdin.flush()
    ev = {"resume": None, "reused": None, "pp": [], "tokens": [], "done": None, "err": None}
    deadline = time.monotonic() + timeout
    while True:
        line = io.next_line(deadline - time.monotonic())
        on_line(line)
        if line.startswith("T "):
            ev["tokens"].append(int(line[2:]))
        elif line.startswith("DONE "):
            ev["done"] = line.split()[1:]
            return ev
        elif line.startswith("ERR "):
            ev["err"] = line[4:]
            return ev
        elif line.startswith("RESUME "):
            ev["resume"] = int(line.split()[1])
        elif line.startswith("REUSED "):
            ev["reused"] = int(line.split()[1])
        elif line.startswith("PP "):
            ev["pp"].append(line.split()[1:])


def request_line(mode: str, max_new: int, emb_path: Path, ids: list[int], pin: int | None = None) -> str:
    """The one request line this benchmark sends: GENI with an empty embeddings file, or GEN.

    Both request modes carry the same ids and the same settings, so a chained arm measures the same
    prompts whichever one its engine supports.  `pin` is the opt-in request key (generate.cpp:
    `pin=N`): the engine reads the prompt in two parts at N and pins the checkpoint there, so a later
    request that resumes from it re-reads only what lies above N.  The engine ignores N >= n - 1, so
    the largest honored value is n - 2.  The key=value tokens come before the ids."""
    pin_key = f" pin={pin}" if pin is not None else ""
    csv = ",".join(map(str, ids))
    if mode == "geni":
        return f"GENI {max_new} temperature=0{pin_key} {emb_path} {csv}"
    return f"GEN {max_new} temperature=0{pin_key} {csv}"


def launch_engine(argv: list[str], cfg: dict, env: dict, log_path: Path, startup_timeout: float,
                  on_line) -> tuple[subprocess.Popen, EngineIO, object, dict, float]:
    """Start one serve engine and wait for READY.  A failed start ends the process and re-raises."""
    logf = log_path.open("a", encoding="utf-8")
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cfg.get("cwd") or str(REPO), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=logf, text=True, encoding="utf-8", bufsize=1)
    io = EngineIO(proc)
    try:
        info = wait_ready(io, startup_timeout, on_line)
    except BaseException:
        try:
            proc.kill()
            proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            logf.close()
        except OSError:
            pass
        raise
    return proc, io, logf, info, round(time.monotonic() - t0, 1)


def quit_engine(proc: subprocess.Popen, io: EngineIO, logf) -> None:
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    finally:
        try:
            logf.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- config/argv
def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if not isinstance(cfg.get("args"), list):
        raise SystemExit(f"{path}: no args list")
    return cfg


def drop_flag(args: list[str], *flags: str) -> list[str]:
    """Remove `flags` (and the value of each, unless it was written as --flag=value) from an argv list."""
    out: list[str] = []
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a.split("=", 1)[0] in flags:
            if "=" not in a and i + 1 < len(args):
                skip = True
            continue
        out.append(a)
    return out


def flag_value(argv: list[str], flag: str) -> str | None:
    """The effective value of `flag`: its LAST occurrence, which is the one the engine keeps.

    The engine's option loop assigns every occurrence in order, so an argument list that sets a scalar
    twice ends on the last one.  The cold arm appends `--conversation-cache-mib 0` after the config's
    own value for exactly this reason, and reading the first occurrence there reports a mismatch that
    the engine does not have.  `--flag=value` counts as an occurrence too."""
    value = None
    for i, a in enumerate(argv):
        if a == flag:
            if i + 1 < len(argv):
                value = argv[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    return value


def flag_count(argv: list[str], flag: str) -> int:
    return sum(1 for a in argv if a == flag or a.startswith(flag + "="))


def engine_argv(cfg: dict, exe: str, layer_split: str | None = None) -> list[str]:
    """The server's own construction (serve/server.py engine_args) plus the no-reuse flags.

    `layer_split` overrides the config's split for a diagnostic arm; None keeps the config's
    fixed setting (20 on this machine).  Disk offload is always removed and conversation
    reuse is always disabled."""
    args = list(cfg["args"])
    gpus = cfg.get("gpu") or [0]
    if len(gpus) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    if layer_split is not None:
        args = drop_flag(args, "--layer-split")
        args += ["--layer-split", str(layer_split)]
    args = drop_flag(args, "--conversation-cache-disk", "--conversation-cache-disk-gib",
                     "--conversation-cache-disk-slots", "--conversation-cache-disk-min-free-mib")
    # no conversation reuse: no checkpoints, no RAM parking, no disk tier
    args += ["--prompt-cache", "0", "--prompt-cache-every", "0", "--prompt-cache-root", "0",
             "--conversation-cache-mib", "0"]
    return [exe, "--serve", *args]


def chain_engine_argv(cfg: dict, exe: str, *, prompt_cache: int, cache_every: int, cache_root: int = 0,
                      layer_split: str | None = None, disk_dir: str | None = None, no_disk: bool = False,
                      xargs: list[str] = (), short_read: int | None = None,
                      pipeline_windows: int | None = None) -> tuple[list[str], dict]:
    """The server's own construction with the prompt caches ON and every path private.

    The cold arm removes the caches and the disk tier; this mode keeps them, because an agent turn
    reuses the prefix the turn before it read.  The engine's RAM cache belongs to this process.  A
    disk store the config names - as `--conversation-cache-disk` in its args, or as the
    `conversation_cache_disk`/`conversation_cache_disk_gib` keys the server maps to those flags - is
    pointed at `disk_dir` (under --out, removed before the run), so the arm never reads or writes the
    production store; `no_disk` drops it instead.  --expert-profile-save is always dropped: it writes a
    file the next run would read.  Returns the argv and what it changed.
    """
    args = list(cfg["args"])
    gpus = cfg.get("gpu") or [0]
    isolated: dict[str, object] = {"disk_from": None, "disk_to": None, "disk_dropped": None,
                                   "dropped": []}
    if len(gpus) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    if layer_split is not None:
        args = drop_flag(args, "--layer-split")
        args += ["--layer-split", str(layer_split)]
    for flag in CHAIN_DROPPED_ARGS:
        if flag_count(args, flag):
            isolated["dropped"].append(flag)
    args = drop_flag(args, *CHAIN_DROPPED_ARGS)
    disk_from = flag_value(args, "--conversation-cache-disk")
    disk_gib = flag_value(args, "--conversation-cache-disk-gib")
    if disk_from is None and cfg.get("conversation_cache_disk"):
        disk_from = str(cfg["conversation_cache_disk"])
        if disk_gib is None and cfg.get("conversation_cache_disk_gib") is not None:
            disk_gib = str(cfg["conversation_cache_disk_gib"])
    isolated["disk_from"] = disk_from
    args = drop_flag(args, "--conversation-cache-disk", "--conversation-cache-disk-gib",
                     "--conversation-cache-disk-slots", "--conversation-cache-disk-min-free-mib")
    if disk_from is not None:
        if no_disk:
            isolated["disk_dropped"] = "--chain-no-disk"
        elif disk_dir is None:
            isolated["disk_dropped"] = "no private directory"
        elif not (disk_gib or "").isdigit() or int(disk_gib) < 1:
            isolated["disk_dropped"] = "no positive conversation_cache_disk_gib"
        else:
            args += ["--conversation-cache-disk", str(disk_dir), "--conversation-cache-disk-gib", str(disk_gib)]
            for key, flag in (("conversation_cache_disk_slots", "--conversation-cache-disk-slots"),
                              ("conversation_cache_disk_min_free_mib", "--conversation-cache-disk-min-free-mib")):
                value = cfg.get(key)
                if isinstance(value, int) and value > 0:
                    args += [flag, str(value)]
            isolated["disk_to"] = str(disk_dir)
    # The prompt caches an agent turn needs.  Root 0: the synthetic prompts hold no chat turn token,
    # so a root checkpoint would never be taken anyway.
    args = drop_flag(args, "--prompt-cache", "--prompt-cache-every", "--prompt-cache-root")
    args += ["--prompt-cache", str(prompt_cache), "--prompt-cache-every", str(cache_every),
             "--prompt-cache-root", str(cache_root)]
    if short_read is not None:
        args = drop_flag(args, "--short-read")
        args += ["--short-read", str(short_read)]
    if pipeline_windows is not None:
        args = drop_flag(args, "--pipeline-windows")
        args += ["--pipeline-windows", str(pipeline_windows)]
    args += list(xargs)
    args, forced = isolate_final_args(args, disk_dir=isolated["disk_to"])
    if forced:
        isolated["xargs_sanitized"] = forced
    return [exe, "--serve", *args], isolated


def isolate_final_args(args: list[str], *, disk_dir: str | None) -> tuple[list[str], list[str]]:
    """Enforce the private-path rule on the FINAL argv, after --xargs.

    --xargs is appended last, so it could otherwise put a production store or a profile-save path back
    into an arm.  A disk store must be `disk_dir` (the private one) or absent; a profile save is always
    removed.  Returns the argv and what was removed or forced, so the result records it."""
    changed: list[str] = []
    for flag in CHAIN_DROPPED_ARGS:
        if flag_count(args, flag):
            changed.append(f"dropped {flag}")
        args = drop_flag(args, flag)
    disk = flag_value(args, "--conversation-cache-disk")
    if disk is not None and disk != disk_dir:
        if disk_dir is None:
            args = drop_flag(args, "--conversation-cache-disk", "--conversation-cache-disk-gib",
                             "--conversation-cache-disk-slots", "--conversation-cache-disk-min-free-mib")
            changed.append(f"dropped the disk store {disk}")
        else:
            args = drop_flag(args, "--conversation-cache-disk")
            args += ["--conversation-cache-disk", disk_dir]
            changed.append(f"forced the disk store to {disk_dir}")
    return args, changed


def engine_env(cfg: dict, gpus: list[int], overrides: list[str], drop_diagnostics: bool) -> dict:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    for k, v in SERVICE_ENV.items():
        env.setdefault(k, v)
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    lib_dirs = [d for d in (cfg.get("lib_dirs") or []) if Path(d).is_dir()]
    if lib_dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(lib_dirs + ([env[var]] if env.get(var) else []))
    if drop_diagnostics:
        for k in DIAGNOSTIC_ENV:
            env.pop(k, None)
    applied = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--env wants KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        env[k] = v
        applied[k] = v
    env["_HARNESS_OVERRIDES"] = json.dumps(applied)   # recorded, harmless to the engine
    return env


def base_settings_checks(argv: list[str], env: dict, expected_split: str = "20") -> tuple[dict, dict]:
    """The settings every arm must share, and the effective value each one was read with.

    The values are recorded next to the checks so a reader sees a divergence the arm accepted.  The
    `--vram-reserve-mib 700` requirement is one: the live config carries 750, and the cold baseline's
    result must stay comparable, so the requirement is kept and the effective value is reported."""
    checks: dict[str, object] = {}
    values: dict[str, object] = {}
    for flag, want in REQUIRED_ARG_PAIRS:
        want = expected_split if flag == "--layer-split" else want
        got = flag_value(argv, flag)
        values[flag] = got
        checks[f"{flag} {want}"] = (got == want)
    for flag in REQUIRED_FLAGS:
        checks[flag] = flag in argv
    for k, v in REQUIRED_ENV.items():
        checks[f"env {k}={v}"] = env.get(k) == v
    return checks, values


def settings_check(argv: list[str], env: dict, expected_split: str = "20") -> dict:
    checks, values = base_settings_checks(argv, env, expected_split)
    checks["no conversation disk"] = not any(a.startswith("--conversation-cache-disk") for a in argv)
    checks["prompt-cache 0"] = flag_count(argv, "--prompt-cache") == 1 and flag_value(argv, "--prompt-cache") == "0"
    checks["conversation-cache-mib 0"] = flag_value(argv, "--conversation-cache-mib") == "0"
    values["--prompt-cache"] = flag_value(argv, "--prompt-cache")
    values["--conversation-cache-mib"] = flag_value(argv, "--conversation-cache-mib")
    return {"ok": all(checks.values()), "checks": checks, "values": values}


def chain_settings_check(argv: list[str], env: dict, chain: dict, expected_split: str = "20") -> dict:
    """The shared settings plus the chained mode's own: the prompt caches are on and the state is private."""
    checks, values = base_settings_checks(argv, env, expected_split)
    prompt_cache = flag_value(argv, "--prompt-cache")
    cache_every = flag_value(argv, "--prompt-cache-every")
    cache_root = flag_value(argv, "--prompt-cache-root")
    disk = flag_value(argv, "--conversation-cache-disk")
    checks["prompt cache on"] = (prompt_cache or "0") not in ("0", "None")
    checks[f"prompt-cache {chain['prompt_cache']}"] = prompt_cache == str(chain["prompt_cache"])
    checks[f"prompt-cache-every {chain['cache_every']}"] = cache_every == str(chain["cache_every"])
    checks["prompt-cache-root 0"] = cache_root == "0"
    checks["no production disk store"] = disk is None or disk == chain.get("disk_dir")
    checks["no expert-profile-save"] = not any(a.startswith("--expert-profile-save") for a in argv)
    if chain.get("short_read") is not None:
        checks[f"--short-read {chain['short_read']}"] = flag_value(argv, "--short-read") == str(chain["short_read"])
    if chain.get("pipeline_windows") is not None:
        checks[f"--pipeline-windows {chain['pipeline_windows']}"] = (
            flag_value(argv, "--pipeline-windows") == str(chain["pipeline_windows"]))
    values.update({"--prompt-cache": prompt_cache, "--prompt-cache-every": cache_every,
                   "--prompt-cache-root": cache_root, "--conversation-cache-mib":
                   flag_value(argv, "--conversation-cache-mib"), "--conversation-cache-disk": disk})
    return {"ok": all(checks.values()), "checks": checks, "values": values}


# --------------------------------------------------------------------------- compare
def parse_done(fields: list[str]) -> dict:
    """Parse the positional protocol fields and the engine's named extension tail."""
    out: dict[str, object] = {}
    positional = []
    named = {}
    for value in fields:
        key, sep, tail_value = value.partition("=")
        if sep:
            named[key] = tail_value
        elif named:
            raise ValueError("positional DONE field after named extension")
        else:
            positional.append(value)
    for name, value in zip(DONE_FIELDS, positional):
        if name == "finish":
            out[name] = value
        elif name in ("prompt_ms", "decode_ms", "file_mb"):
            out[name] = float(value)
        else:
            out[name] = int(value)
    for name, value in named.items():
        try:
            out[name] = int(value)
        except ValueError:
            try:
                out[name] = float(value)
            except ValueError:
                out[name] = value
    return out


def chain_row_metrics(done: dict, *, round_index: int, turn: int, phase: str, requested_diff: int,
                      requested_prompt_tokens: int, cache_every: int, pin_tail: bool = False) -> dict:
    """One chain turn's accounting.

    `requested_diff` is the tokens the turn adds to the prompt.  `prompt_read` and `reused` are the
    engine's own DONE fields.  The row keeps all three and never equates them: the engine resumes at
    most one token short of the prompt, so even the diff-0 turn (the identical prompt again) reads the
    tokens above its newest checkpoint.  With `pin_tail` every request pins n-2, so the turn after it
    reads no more than `requested_diff + 2`; a larger read means the pinned checkpoint was not used and
    the measurement is rejected."""
    n = done.get("prompt_tokens")
    reused = done.get("reused")
    read = done.get("prompt_read")
    fresh = (n - reused) if isinstance(n, int) and isinstance(reused, int) else None
    previous = requested_prompt_tokens - requested_diff
    row: dict[str, object] = {
        "round": round_index, "turn": turn, "phase": phase,
        "requested_diff": requested_diff, "requested_prompt_tokens": requested_prompt_tokens,
        "prompt_tokens": n, "reused": reused, "prompt_read": read, "fresh_nominal": fresh,
        "read_vs_requested": (read - requested_diff) if isinstance(read, int) else None,
        "reuse_shortfall": (previous - reused) if isinstance(reused, int) else None,
        "reuse_ratio": (reused / n) if (isinstance(n, int) and n and isinstance(reused, int)) else None,
        "generated": done.get("generated"), "finish": done.get("finish"),
        "prompt_ms": done.get("prompt_ms"), "decode_ms": done.get("decode_ms"),
        "prompt_tps": None, "prompt_tps_nominal": None, "decode_tps": None,
        "error": None, "note": None, "warnings": [],
    }
    prompt_ms, decode_ms, generated = done.get("prompt_ms"), done.get("decode_ms"), done.get("generated")
    if isinstance(read, int) and isinstance(prompt_ms, float) and prompt_ms > 0:
        row["prompt_tps"] = round(1000.0 * read / prompt_ms, 2)          # the tokens the engine read
    if isinstance(fresh, int) and isinstance(prompt_ms, float) and prompt_ms > 0:
        row["prompt_tps_nominal"] = round(1000.0 * fresh / prompt_ms, 2)  # the nominal fresh tokens
    if isinstance(generated, int) and isinstance(decode_ms, float) and decode_ms > 0:
        row["decode_tps"] = round(1000.0 * generated / decode_ms, 2)
    if not isinstance(n, int) or n != requested_prompt_tokens:
        row["error"] = f"engine prompt_tokens {n} != requested {requested_prompt_tokens}"
    elif phase == "prefix" and round_index == 0:
        if reused != 0:      # only the process's first request can be cold
            row["error"] = f"the first request was not cold (reused={reused})"
    elif not isinstance(reused, int) or reused <= 0:
        row["error"] = f"no prompt reuse after the prefix (reused={reused})"
    if isinstance(read, int) and isinstance(fresh, int) and read > fresh:
        # A contradiction in the engine's own counts.  The size or reuse error above is the primary
        # one (it says which field is wrong), so this becomes a warning when one is already set.
        message = f"the engine read {read} tokens above its {fresh} fresh prompt tokens"
        if row["error"] is None:
            row["error"] = message
        else:
            row["warnings"].append(message)
    if pin_tail and phase == "chain" and isinstance(read, int) and read > requested_diff + 2:
        # the pinned checkpoint at n-2 was not there: this turn cannot be compared with the pinned arms
        message = (f"the pinned checkpoint was not used: read {read} tokens for a requested diff of "
                   f"{requested_diff} (at most {requested_diff + 2} with pin=n-2)")
        if row["error"] is None:
            row["error"] = message
        else:
            row["warnings"].append(message)
    if phase == "chain" and requested_diff == 0:
        row["note"] = (f"the requested diff is 0 (the identical prompt again): the engine reused {reused} "
                       f"tokens and read {read}.  A resume stops at most one token short of the prompt, so "
                       "an identical repeat re-reads the tokens above its newest checkpoint; the last "
                       "token stays for the first window's logits")
    elif phase == "prefix" and round_index > 0:
        row["note"] = (f"a later round's prefix turn: the prefix is already cached, so this is the warm "
                       f"repeat of turn 0 (reused {reused}, read {read})")
    if phase == "chain" and isinstance(reused, int) and requested_diff > 0 and previous - reused > cache_every:
        row["warnings"].append(
            f"reused {reused} is more than one checkpoint interval ({cache_every}) below the previous "
            f"prompt's {previous} tokens")
    return row


def compare_chain_results(a: dict, b: dict) -> dict:
    """Two chained runs side by side: identical prompts (hashes) and what the engine actually did.

    Every row is compared, including a row either arm rejected: a failed turn is listed in
    `failed_turns` with each arm's error, never dropped from both arms."""
    def key(row):
        return (row.get("round", 0), row.get("turn"), row.get("phase"), row.get("requested_diff"))
    ra = {key(r): r for r in a.get("rows", [])}
    rb = {key(r): r for r in b.get("rows", [])}
    if set(ra) != set(rb):
        raise ValueError("the arms ran different chain plans; compare the same --chain-prefix/--chain-diffs")
    out = {"mode": "chain",
           "a": {"label": a.get("arm_label"), "exe": a.get("build", {}).get("engine_sha256"),
                 "status": a.get("status")},
           "b": {"label": b.get("arm_label"), "exe": b.get("build", {}).get("engine_sha256"),
                 "status": b.get("status")},
           "turns": {}, "prompt_hash_mismatches": [], "read_parity": True, "failed_turns": [],
           "status": "ok",
           "notes": ["prompt_read and reused are the engine's own counts; the requested diff is not a read count",
                     "a turn either arm rejected is listed in failed_turns with its status"]}
    for k in sorted(ra):
        x, y = ra[k], rb[k]
        label = f"r{x.get('round', 0)} t{x.get('turn')} diff{x.get('requested_diff')}"
        if x.get("prompt_sha256") != y.get("prompt_sha256"):
            out["prompt_hash_mismatches"].append(label)
        failed = bool(x.get("error")) or bool(y.get("error"))
        if failed:
            out["failed_turns"].append(label)
        elif (x.get("prompt_read") != y.get("prompt_read") or x.get("reused") != y.get("reused")):
            out["read_parity"] = False
        out["turns"][label] = {
            "requested_prompt_tokens": x.get("requested_prompt_tokens"),
            "a_prompt_read": x.get("prompt_read"), "b_prompt_read": y.get("prompt_read"),
            "a_reused": x.get("reused"), "b_reused": y.get("reused"),
            "a_prompt_ms": x.get("prompt_ms"), "b_prompt_ms": y.get("prompt_ms"),
            "a_prompt_tps": x.get("prompt_tps"), "b_prompt_tps": y.get("prompt_tps"),
            "a_decode_tps": x.get("decode_tps"), "b_decode_tps": y.get("decode_tps"),
            "a_error": x.get("error"), "b_error": y.get("error"),
            "output_parity": x.get("output_ids") == y.get("output_ids"),
        }
    if out["failed_turns"] or out["prompt_hash_mismatches"]:
        out["status"] = "failed"
    return out


def compare_results(a_path: Path, b_path: Path) -> dict:
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    if a.get("mode") == "chain" or b.get("mode") == "chain":
        return compare_chain_results(a, b)
    out = {"a": {"label": a.get("arm_label"), "exe": a.get("build", {}).get("engine_sha256")},
           "b": {"label": b.get("arm_label"), "exe": b.get("build", {}).get("engine_sha256")},
           "targets": {}}
    for target in sorted({r["target"] for r in a.get("rows", [])} & {r["target"] for r in b.get("rows", [])}):
        ra = sorted((r for r in a["rows"] if r["target"] == target and not r.get("error")), key=lambda r: r["repeat"])
        rb = sorted((r for r in b["rows"] if r["target"] == target and not r.get("error")), key=lambda r: r["repeat"])
        if not ra or not rb:
            continue
        pairs_a = [(r["repeat"], r["prompt_sha256"]) for r in ra]
        pairs_b = [(r["repeat"], r["prompt_sha256"]) for r in rb]
        if pairs_a != pairs_b:
            raise ValueError(f"unpaired prompts at {target} tokens: repeats and prompt hashes must match")
        ma = statistics.median(r["prompt_ms"] for r in ra)
        mb = statistics.median(r["prompt_ms"] for r in rb)
        ta = statistics.median(r["prompt_tps"] for r in ra)
        tb = statistics.median(r["prompt_tps"] for r in rb)
        parity = [r["output_ids"] for r in ra] == [r["output_ids"] for r in rb]
        out["targets"][str(target)] = {
            "a_prompt_ms_median": ma, "b_prompt_ms_median": mb,
            "a_prompt_tps_median": ta, "b_prompt_tps_median": tb,
            "prompt_ms_change_pct": round(100.0 * (mb - ma) / ma, 2) if ma else None,
            "prompt_tps_change_pct": round(100.0 * (tb - ta) / ta, 2) if ta else None,
            "output_parity": parity,
        }
    return out


# --------------------------------------------------------------------------- arm lifecycle
def engine_build_info(result: dict, exe: str) -> None:
    """BUILD.json, the repository commit and the card names, recorded with every arm."""
    try:
        version = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        result["build"]["engine_version"] = version
    except (OSError, ValueError):
        pass
    for key, cmd in (("git_commit", ["git", "rev-parse", "HEAD"]),
                     ("git_describe", ["git", "describe", "--tags", "--always"])):
        try:
            result["build"][key] = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                                  timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        names, drivers = [], set()
        for row in q.stdout.splitlines():
            if "," in row:
                n, d = row.split(",", 1)
                names.append(n.strip())
                drivers.add(d.strip())
        result["build"]["gpu_names"] = names
        result["build"]["driver_version"] = ",".join(sorted(drivers)) or None
    except (OSError, subprocess.TimeoutExpired):
        pass


def warn_other_engine() -> None:
    """A second serve engine on the same cards invalidates a timing arm."""
    try:
        ps = subprocess.run(["pgrep", "-af", "strata --serve"], capture_output=True, text=True, timeout=10)
        others = [ln for ln in ps.stdout.splitlines() if ln.strip() and str(os.getpid()) not in ln]
        if others:
            print(f"[harness] WARNING: another serve engine looks alive:\n  " +
                  "\n  ".join(others), file=sys.stderr)
    except (OSError, subprocess.TimeoutExpired):
        pass


def parse_xargs(values: list[str]) -> list[str]:
    """--xargs: each value is a ';'-separated list of engine arguments, each split on spaces."""
    out: list[str] = []
    for value in values or []:
        for piece in value.split(";"):
            out.extend(piece.split())
    return out


def chain_summary(rows: list[dict]) -> dict:
    """Per diff: the engine's own counts and times over the clean turns."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("error"):
            continue
        # only round 0's prefix turn is the cold read; a later round's is a warm repeat of it
        name = (f"diff:{r.get('requested_diff')}" if r.get("phase") != "prefix"
                else ("prefix" if r.get("round", 0) == 0 else f"prefix:r{r.get('round')}"))
        groups.setdefault(name, []).append(r)
    out: dict[str, dict] = {}
    for name, group in groups.items():
        def med(field: str):
            vals = [r[field] for r in group if isinstance(r.get(field), (int, float))]
            return statistics.median(vals) if vals else None
        out[name] = {
            "turns": len(group),
            "requested_prompt_tokens": group[0].get("requested_prompt_tokens"),
            "requested_diff": group[0].get("requested_diff"),
            "reused_median": med("reused"),
            "prompt_read_all": [r.get("prompt_read") for r in group],
            "prompt_read_median": med("prompt_read"),
            "read_vs_requested_median": med("read_vs_requested"),
            "reuse_ratio_median": med("reuse_ratio"),
            "prompt_ms_median": med("prompt_ms"),
            "prompt_tps_median": med("prompt_tps"),
            "decode_tps_median": med("decode_tps"),
            "output_first_ids": [r.get("output_ids", [])[:1] for r in group],
        }
    return out


# --------------------------------------------------------------------------- chained mode
def run_chain(args, cfg: dict, exe: str, gpus: list[int], tokenizer_dir: Path) -> int:
    """One prefix turn, then one turn per diff.  One engine keeps the state between the turns."""
    diffs = parse_diffs(args.chain_diffs)
    prefix = args.chain_prefix
    if prefix < 1:
        raise SystemExit("--chain-prefix must be >= 1")
    if args.max_new < 1:
        raise SystemExit("--max-new must be >= 1")
    cache_every = args.chain_cache_every or max(64, prefix // 4)
    if cache_every < 1:
        raise SystemExit("--chain-cache-every must be >= 1")
    if cache_every > prefix:
        raise SystemExit("--chain-cache-every must be <= --chain-prefix: a longer interval saves no "
                         "checkpoint below the prefix, so no turn after it could reuse one")
    rounds = max(1, args.chain_rounds)
    if diffs[0] != 0:
        print("[harness] NOTE: the first --chain-diffs value is not 0, so the first chain turn does not "
              "repeat the prefix prompt", file=sys.stderr)
    if args.targets != "2048,4096,8192,16384,32768" or args.repeats != 1:
        print("[harness] NOTE: --chain uses its own schedule; --targets and --repeats are ignored", file=sys.stderr)

    tokenizer = load_tokenizer(tokenizer_dir)
    head = tokenizer.encode(f"[chain {args.seed:08x} p{prefix}] ", parse_special=False)
    if len(head) >= prefix:
        raise SystemExit(f"--chain-prefix {prefix} is too small for the request tag")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = None if args.rebuild_pool else (args.out / f"token-pool-chain-seed{args.seed}.json")
    if not args.keep_pool:
        cache_path = None
    pool = build_pool(tokenizer, args.seed, chain_pool_need(prefix, diffs), cache_path)

    plan = chain_plan(prefix, diffs, rounds)
    plan_rows = []
    for p in plan:
        ids = build_chain_ids(head, pool, p["prompt_tokens"])
        plan_rows.append({**p, "len": len(ids), "sha256": ids_sha256(ids),
                          "first_ids": ids[:8], "last_ids": ids[-4:]})

    prompt_cache = args.chain_prompt_cache or int(flag_value(cfg["args"], "--prompt-cache") or 6)
    if prompt_cache < 1:
        raise SystemExit("chained mode needs the prompt cache on (--chain-prompt-cache >= 1)")
    if args.chain_pin_tail and prompt_cache < 3:
        raise SystemExit("--chain-pin-tail needs --prompt-cache 3 or more (the engine ignores pin below it)")
    disk_dir = str(args.out / f"conversations-{args.arm_label}")
    argv, isolated = chain_engine_argv(
        cfg, exe, prompt_cache=prompt_cache, cache_every=cache_every, cache_root=0,
        layer_split=args.layer_split, disk_dir=disk_dir, no_disk=args.chain_no_disk,
        xargs=parse_xargs(args.xargs), short_read=args.short_read, pipeline_windows=args.pipeline_windows)
    if isolated["disk_from"] is not None and isolated["disk_to"] is None:
        print(f"[harness] NOTE: the config's conversation disk store is dropped for this arm "
              f"({SCRUB(str(isolated['disk_from']))}): {isolated['disk_dropped']}", file=sys.stderr)
    elif isolated["disk_to"] is not None:
        print(f"[harness] NOTE: the conversation disk store is private: "
              f"{SCRUB(str(isolated['disk_to']))}", file=sys.stderr)
    if isolated["dropped"]:
        print(f"[harness] NOTE: dropped for isolation: {', '.join(isolated['dropped'])}", file=sys.stderr)
    if args.pipeline_windows and len(gpus) != 2:
        print(f"[harness] WARNING: --pipeline-windows needs a layer split on exactly two GPUs; the config "
              f"lists {len(gpus)}", file=sys.stderr)

    env = engine_env(cfg, gpus, parse_envs(args.env), drop_diagnostics=True)
    chain_cfg = {"prefix": prefix, "diffs": diffs, "cache_every": cache_every,
                 "prompt_cache": prompt_cache, "cache_root": 0, "rounds": rounds,
                 "pool_tokens": len(pool), "disk_dir": isolated["disk_to"],
                 "short_read": args.short_read, "pipeline_windows": args.pipeline_windows,
                 "pin_tail": args.chain_pin_tail}
    check = chain_settings_check(argv, env, chain_cfg,
                                 args.layer_split or str(cfg.get("layer_split") or "20"))

    exe_sha = file_sha256(Path(exe))
    print(f"[harness] chain arm={args.arm_label} exe={SCRUB(exe)} sha256={exe_sha}", file=sys.stderr)
    print(f"[harness] prefix={prefix} diffs={diffs} rounds={rounds} cache_every={cache_every} "
          f"prompt_cache={prompt_cache} pool={len(pool)} turns={len(plan_rows)} "
          f"settings_ok={check['ok']}", file=sys.stderr)
    if not check["ok"]:
        for name, ok in check["checks"].items():
            if not ok:
                print(f"[harness] SETTINGS MISMATCH: {name}", file=sys.stderr)
    if args.expect_sha256 and exe_sha != args.expect_sha256:
        raise SystemExit(f"engine sha256 {exe_sha} != --expect-sha256 {args.expect_sha256}")

    if args.dry_run:
        sampler = GpuSampler(gpus, args.sample_interval_s, args.out / f"gpu-{args.arm_label}-dry.csv")
        sampler.start()
        time.sleep(1.2)
        sampler.stop()
        dry = {
            "mode": "chain", "arm_label": args.arm_label, "argv": sanitize(argv),
            "env_overrides": parse_envs(args.env), "service_env": SERVICE_ENV,
            "settings_check": check, "chain": {**chain_cfg, "isolated": sanitize(isolated)},
            "chain_plan": plan_rows, "pool_tokens": len(pool),
            "tokenizer_dir": sanitize(str(tokenizer_dir)),
            "telemetry_sample": sanitize(sampler.samples[:2]), "telemetry_errors": sampler.errors,
        }
        print(json.dumps(dry, indent=2))
        return 0

    if not args.no_settings_strict and not check["ok"]:
        raise SystemExit("runtime settings do not match the chained mode's required settings; "
                         "fix or pass --no-settings-strict")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is required for the thermal gate and telemetry")

    label = args.arm_label
    log_path = args.out / f"engine-{label}.log"
    protocol_path = args.out / f"protocol-{label}.jsonl"
    gpu_csv = args.out / f"gpu-{label}.csv"
    emb_path = (args.out / "geni-empty.sve").resolve()
    emb_path.write_bytes(b"")
    if isolated["disk_to"] is not None:      # a fresh store: no state from an earlier arm or run
        shutil.rmtree(isolated["disk_to"], ignore_errors=True)

    protocol_f = protocol_path.open("a", encoding="utf-8")

    def on_line(line: str):
        protocol_f.write(json.dumps({"t": round(time.time(), 3), "line": line}) + "\n")
        protocol_f.flush()

    result: dict = {
        "protocol_version": 1,
        "harness": "bench/run_v100_prefill.py",
        "harness_sha256": file_sha256(Path(__file__)),
        "started_epoch_s": time.time(),
        "arm_label": label,
        "mode": "chain",
        "build": {
            "engine_path": SCRUB(exe), "engine_sha256": exe_sha,
            "engine_version": None, "git_commit": None, "git_describe": None,
            "gpu_names": [], "driver_version": None, "python": sys.version.split()[0],
        },
        "runtime": {
            "argv": sanitize(argv), "cwd": SCRUB(str(cfg.get("cwd") or REPO)),
            "env_overrides": parse_envs(args.env), "service_env": SERVICE_ENV,
            "config_env": sanitize(cfg.get("env") or {}),
            "request_mode": args.request_mode, "max_new": args.max_new,
            "layer_split": args.layer_split or str(cfg.get("layer_split") or "20"),
            "embeddings_file": SCRUB(str(emb_path)), "embeddings_sha256": file_sha256(emb_path),
            "xargs": parse_xargs(args.xargs), "short_read": args.short_read,
            "pipeline_windows": args.pipeline_windows, "pin_tail": args.chain_pin_tail,
            "prompt_cache": prompt_cache, "prompt_cache_every": cache_every,
            "conversation_cache_mib": flag_value(argv, "--conversation-cache-mib"),
            "disk_offload": isolated["disk_to"] is not None,
            "sample_interval_s": args.sample_interval_s,
            "cooldown": {"target_c": args.cooldown_c, "stable_s": args.cooldown_stable_s,
                         "max_s": args.cooldown_max_s, "enabled": not args.no_cooldown},
        },
        "settings_check": check,
        "chain": {**chain_cfg, "isolated": sanitize(isolated)},
        "chain_plan": plan_rows,
        "engine_info": {},
        "startup_wall_s": None,
        "rows": [],
        "summary": {},
        "warnings": [],
        "notes": [CHAIN_NOTE],
        "errors": [],
        "status": "ok",
    }

    engine_build_info(result, exe)
    if not args.allow_other_engine:
        warn_other_engine()

    sampler = GpuSampler(gpus, args.sample_interval_s, gpu_csv)
    sampler.start()
    proc = None
    logf = None
    io = None
    try:
        proc, io, logf, info, startup_wall = launch_engine(argv, cfg, env, log_path,
                                                          args.startup_timeout_s, on_line)
        result["startup_wall_s"] = startup_wall
        result["engine_info"] = info
        print(f"[harness] READY after {startup_wall}s; context={info.get('context')} kv={info.get('kv')} "
              f"disk={info.get('conversation_cache_disk')}", file=sys.stderr)
        if isolated["disk_to"] is None and str(info.get("conversation_cache_disk")) not in ("0", "None"):
            raise SystemExit("engine reports a conversation disk tier this arm did not ask for; "
                             "refusing to measure")

        seq = 0
        for p in plan_rows:
            seq += 1
            ids = build_chain_ids(head, pool, p["prompt_tokens"])
            row: dict = {"seq": seq, "prompt_sha256": ids_sha256(ids), "error": None}
            if not args.no_cooldown:
                gate = wait_cooldown(sampler, args.cooldown_c, args.cooldown_stable_s,
                                     args.cooldown_max_s, args.cooldown_poll_s)
            else:
                gate = {"met": None, "waited_s": 0.0, "temps_c": None, "reason": "disabled"}
            row["cooldown"] = sanitize(gate)
            if gate["met"] is False:
                print(f"[harness] WARNING: cooldown not met for {p['phase']} t{p['turn']}: {gate}", file=sys.stderr)
            # pin=n-2 (the engine ignores pin >= n-1): the turn after it re-reads only the last two tokens
            pin = chain_pin(len(ids), args.chain_pin_tail)
            row["pin"] = pin
            request = request_line(args.request_mode, args.max_new, emb_path, ids, pin=pin)
            t0 = time.monotonic()
            try:
                ev = run_request(io, request, args.request_timeout_s, on_line)
            except (TimeoutError, EOFError, OSError) as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["request_wall_s"] = round(time.monotonic() - t0, 3)
                row["gpu_stats"] = sanitize(sampler.stats(t0, time.monotonic()))
                result["rows"].append(row)
                result["errors"].append(f"row {seq} ({p['phase']} t{p['turn']}): {row['error']}")
                break
            t1 = time.monotonic()
            row["request_wall_s"] = round(t1 - t0, 3)
            row["output_ids"] = ev["tokens"]
            row["resume"] = ev["resume"]
            row["reused_line"] = ev["reused"]
            row["pp"] = ev["pp"]
            row["gpu_stats"] = sanitize(sampler.stats(t0, t1))
            if ev["err"]:
                row["error"] = ev["err"]
            elif ev["done"] is None:
                row["error"] = "no DONE line"
            else:
                row.update(chain_row_metrics(parse_done(ev["done"]), round_index=p["round"], turn=p["turn"],
                                             phase=p["phase"], requested_diff=p["requested_diff"],
                                             requested_prompt_tokens=p["prompt_tokens"],
                                             cache_every=cache_every, pin_tail=args.chain_pin_tail))
            if row["error"]:
                result["errors"].append(
                    f"row {seq} ({p['phase']} t{p['turn']} diff{p['requested_diff']}): {row['error']}")
            for w in row.get("warnings", []):
                result["warnings"].append(f"row {seq} ({p['phase']} t{p['turn']}): {w}")
            result["rows"].append(row)
            print(f"[harness] {p['phase']} t{p['turn']} diff{p['requested_diff']} n={row.get('prompt_tokens')} "
                  f"reused={row.get('reused')} read={row.get('prompt_read')} prompt_ms={row.get('prompt_ms')} "
                  f"tps={row.get('prompt_tps')} dec={row.get('decode_tps')} tok0={row['output_ids'][:1]} "
                  f"err={row['error']}", file=sys.stderr)

        result["summary"] = chain_summary(result["rows"])
    except (TimeoutError, EOFError, RuntimeError, OSError) as e:
        result["errors"].append(f"fatal: {type(e).__name__}: {e}")
        result["status"] = "error"
    except SystemExit as e:
        result["errors"].append(f"fatal: {e}")
        result["status"] = "error"
    finally:
        if io is not None and proc is not None and proc.poll() is None:
            quit_engine(proc, io, logf)
        elif logf is not None:
            try:
                logf.close()
            except OSError:
                pass
        sampler.stop()
        protocol_f.close()
        result["finished_epoch_s"] = time.time()
        if result["errors"] and result["status"] == "ok":
            result["status"] = "partial" if result["rows"] else "error"
        result["sampler_errors"] = sampler.errors
        result_path = args.out / f"result-{label}.json"
        result_path.write_text(json.dumps(sanitize(result), indent=2) + "\n")
        (args.out / f"summary-{label}.json").write_text(json.dumps(sanitize(result["summary"]), indent=2) + "\n")
        print(f"[harness] wrote {SCRUB(str(result_path))}", file=sys.stderr)

    return 0 if result["status"] == "ok" else 1


# --------------------------------------------------------------------------- main
def parse_envs(values: list[str]) -> list[str]:
    return list(values or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "strata-q2_0.json")
    ap.add_argument("--exe", default=None, help="arm engine binary (default: the config's exe)")
    ap.add_argument("--arm-label", default="arm")
    ap.add_argument("--targets", default="2048,4096,8192,16384,32768",
                    help="comma-separated subset of 2048,4096,8192,16384,32768")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-new", type=int, default=1, help="greedy output tokens per request")
    ap.add_argument("--request-mode", choices=("geni", "gen"), default="geni")
    ap.add_argument("--layer-split", default=None,
                    help="diagnostic override of the config's layer split (default: the config's 20)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="runtime env override for A/B (repeatable)")
    ap.add_argument("--expect-sha256", default=None, help="assert the arm binary's sha256")
    ap.add_argument("--out", type=Path, default=REPO / "bench/results/2026-10-03-int8-prefill-direct")
    ap.add_argument("--sample-interval-s", type=float, default=1.0)
    ap.add_argument("--cooldown-c", type=float, default=55.0)
    ap.add_argument("--cooldown-stable-s", type=float, default=10.0)
    ap.add_argument("--cooldown-max-s", type=float, default=900.0)
    ap.add_argument("--cooldown-poll-s", type=float, default=1.0)
    ap.add_argument("--no-cooldown", action="store_true")
    ap.add_argument("--startup-timeout-s", type=float, default=2400.0)
    ap.add_argument("--request-timeout-s", type=float, default=3600.0)
    ap.add_argument("--no-settings-strict", action="store_true")
    ap.add_argument("--allow-other-engine", action="store_true")
    ap.add_argument("--keep-pool", action="store_true", help="cache the token pool JSON in --out")
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and print the plan; start no engine")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A.json", "B.json"))
    ap.add_argument("--chain", action="store_true",
                    help="chained-agentic mode: one prefix turn, then one turn per --chain-diffs value")
    ap.add_argument("--chain-prefix", type=int, default=CHAIN_PREFIX,
                    help=f"chained: the shared prefix's exact token count (default {CHAIN_PREFIX})")
    ap.add_argument("--chain-diffs", default=CHAIN_DIFFS,
                    help=f"chained: the tokens each turn adds, in order (default {CHAIN_DIFFS})")
    ap.add_argument("--chain-cache-every", type=int, default=0,
                    help="chained: --prompt-cache-every for the arm (default: prefix/4)")
    ap.add_argument("--chain-prompt-cache", type=int, default=0,
                    help="chained: --prompt-cache for the arm (default: the config's value, else 6)")
    ap.add_argument("--chain-rounds", type=int, default=1, help="chained: how many times to run the chain")
    ap.add_argument("--chain-no-disk", action="store_true",
                    help="chained: drop the config's conversation disk store instead of isolating it")
    ap.add_argument("--chain-pin-tail", action=argparse.BooleanOptionalAction, default=True,
                    help="chained: send pin=n-2 with every request (default on), so the turn after it "
                         "re-reads only the last two tokens (the engine ignores pin >= n-1); "
                         "--no-chain-pin-tail turns it off and shows the checkpoint-gap re-read")
    ap.add_argument("--xargs", action="append", default=[], metavar="ARGS",
                    help="engine arguments to append, ';'-separated (repeatable); for an arm's own knobs")
    ap.add_argument("--short-read", type=int, default=None, help="append --short-read N to the engine args")
    ap.add_argument("--pipeline-windows", type=int, default=None,
                    help="append --pipeline-windows N to the engine args (needs a two-GPU layer split)")
    args = ap.parse_args()

    if args.compare:
        print(json.dumps(sanitize(compare_results(*args.compare)), indent=2))
        return 0

    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    bad = [t for t in targets if t not in ALLOWED_TARGETS]
    if bad or not targets:
        raise SystemExit(f"--targets must be a subset of {ALLOWED_TARGETS}; got {bad or 'nothing'}")
    if args.repeats < 1 or args.max_new < 1:
        raise SystemExit("--repeats and --max-new must be >= 1")

    cfg = load_config(args.config)
    exe = str(args.exe or cfg["exe"])
    gpus = [int(g) for g in (cfg.get("gpu") or [0])]
    tokenizer_dir = Path(cfg.get("tokenizer") or (Path(cfg["args"][cfg["args"].index("--pack") + 1]) / "tokenizer"))
    if args.chain:
        return run_chain(args, cfg, exe, gpus, tokenizer_dir)

    argv = engine_argv(cfg, exe, args.layer_split)
    env = engine_env(cfg, gpus, parse_envs(args.env), drop_diagnostics=True)
    check = settings_check(argv, env, args.layer_split or str(cfg.get("layer_split") or "20"))

    tokenizer = load_tokenizer(tokenizer_dir)

    args.out.mkdir(parents=True, exist_ok=True)   # the pool cache and the dry-run CSV live here
    need = max(targets) * args.repeats + 4096
    cache_path = None if args.rebuild_pool else (args.out / f"token-pool-seed{args.seed}.json")
    if not args.keep_pool:
        cache_path = None
    pool = build_pool(tokenizer, args.seed, need, cache_path)

    plan = []
    for target in targets:
        for repeat in range(1, args.repeats + 1):
            ids = build_prompt(tokenizer, pool, args.seed, target, repeat)
            plan.append({"target": target, "repeat": repeat, "len": len(ids),
                         "sha256": ids_sha256(ids), "first_ids": ids[:12], "last_ids": ids[-4:]})

    exe_sha = file_sha256(Path(exe))
    print(f"[harness] arm={args.arm_label} exe={SCRUB(exe)} sha256={exe_sha}", file=sys.stderr)
    print(f"[harness] pool={len(pool)} tokens; {len(plan)} prompts; "
          f"settings_ok={check['ok']}", file=sys.stderr)
    if not check["ok"]:
        for name, ok in check["checks"].items():
            if not ok:
                print(f"[harness] SETTINGS MISMATCH: {name}", file=sys.stderr)
    if args.expect_sha256 and exe_sha != args.expect_sha256:
        raise SystemExit(f"engine sha256 {exe_sha} != --expect-sha256 {args.expect_sha256}")

    if args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        sampler = GpuSampler(gpus, args.sample_interval_s, args.out / f"gpu-{args.arm_label}-dry.csv")
        sampler.start()
        time.sleep(1.2)
        sampler.stop()
        dry = {
            "arm_label": args.arm_label, "argv": sanitize(argv), "env_overrides": parse_envs(args.env),
            "service_env": SERVICE_ENV, "settings_check": check,
            "prompts": plan, "pool_tokens": len(pool), "tokenizer_dir": sanitize(str(tokenizer_dir)),
            "telemetry_sample": sanitize(sampler.samples[:2]), "telemetry_errors": sampler.errors,
        }
        print(json.dumps(dry, indent=2))
        return 0

    if not args.no_settings_strict and not check["ok"]:
        raise SystemExit("runtime settings do not match the required arm settings; fix or pass --no-settings-strict")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is required for the thermal gate and telemetry")

    args.out.mkdir(parents=True, exist_ok=True)
    label = args.arm_label
    log_path = args.out / f"engine-{label}.log"
    protocol_path = args.out / f"protocol-{label}.jsonl"
    gpu_csv = args.out / f"gpu-{label}.csv"
    emb_path = (args.out / "geni-empty.sve").resolve()
    emb_path.write_bytes(b"")

    protocol_f = protocol_path.open("a", encoding="utf-8")

    def on_line(line: str):
        protocol_f.write(json.dumps({"t": round(time.time(), 3), "line": line}) + "\n")
        protocol_f.flush()

    result: dict = {
        "protocol_version": 1,
        "harness": "bench/run_v100_prefill.py",
        "harness_sha256": file_sha256(Path(__file__)),
        "started_epoch_s": time.time(),
        "arm_label": label,
        "build": {
            "engine_path": SCRUB(exe), "engine_sha256": exe_sha,
            "engine_version": None, "git_commit": None, "git_describe": None,
            "gpu_names": [], "driver_version": None, "python": sys.version.split()[0],
        },
        "runtime": {
            "argv": sanitize(argv), "cwd": SCRUB(str(cfg.get("cwd") or REPO)),
            "env_overrides": parse_envs(args.env), "service_env": SERVICE_ENV,
            "config_env": sanitize(cfg.get("env") or {}),
            "request_mode": args.request_mode, "max_new": args.max_new,
            "layer_split": args.layer_split or str(cfg.get("layer_split") or "20"),
            "embeddings_file": SCRUB(str(emb_path)), "embeddings_sha256": file_sha256(emb_path),
            "prompt_cache": 0, "conversation_cache_mib": 0, "disk_offload": False,
            "sample_interval_s": args.sample_interval_s,
            "cooldown": {"target_c": args.cooldown_c, "stable_s": args.cooldown_stable_s,
                         "max_s": args.cooldown_max_s, "enabled": not args.no_cooldown},
        },
        "settings_check": check,
        "prompt_plan": {"seed": args.seed, "targets": targets, "repeats": args.repeats,
                        "pool_tokens": len(pool), "prompts": plan},
        "engine_info": {},
        "startup_wall_s": None,
        "rows": [],
        "summary": {},
        "errors": [],
        "status": "ok",
    }

    engine_build_info(result, exe)

    if not args.allow_other_engine:
        warn_other_engine()

    sampler = GpuSampler(gpus, args.sample_interval_s, gpu_csv)
    sampler.start()
    proc = None
    logf = None
    io = None
    try:
        proc, io, logf, info, startup_wall = launch_engine(argv, cfg, env, log_path,
                                                          args.startup_timeout_s, on_line)
        result["startup_wall_s"] = startup_wall
        result["engine_info"] = info
        print(f"[harness] READY after {startup_wall}s; context={info.get('context')} "
              f"kv={info.get('kv')} disk={info.get('conversation_cache_disk')}", file=sys.stderr)
        if str(info.get("conversation_cache_disk")) not in ("0", "None"):
            raise SystemExit("engine reports a live conversation disk tier; refusing to measure")

        seq = 0
        for target in targets:
            for repeat in range(1, args.repeats + 1):
                seq += 1
                ids = build_prompt(tokenizer, pool, args.seed, target, repeat)
                row = {"seq": seq, "target": target, "repeat": repeat, "prompt_sha256": ids_sha256(ids),
                       "error": None}
                if not args.no_cooldown:
                    gate = wait_cooldown(sampler, args.cooldown_c, args.cooldown_stable_s,
                                         args.cooldown_max_s, args.cooldown_poll_s)
                else:
                    gate = {"met": None, "waited_s": 0.0, "temps_c": None, "reason": "disabled"}
                row["cooldown"] = sanitize(gate)
                if gate["met"] is False:
                    print(f"[harness] WARNING: cooldown not met for {target}x{repeat}: {gate}", file=sys.stderr)
                # temperature=0 is the engine default; written out so the protocol log proves greediness
                request = request_line(args.request_mode, args.max_new, emb_path, ids)
                t0 = time.monotonic()
                try:
                    ev = run_request(io, request, args.request_timeout_s, on_line)
                except (TimeoutError, EOFError, OSError) as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    row["request_wall_s"] = round(time.monotonic() - t0, 3)
                    row["gpu_stats"] = sanitize(sampler.stats(t0, time.monotonic()))
                    result["rows"].append(row)
                    result["errors"].append(f"row {seq} ({target}x{repeat}): {row['error']}")
                    break
                t1 = time.monotonic()
                row["request_wall_s"] = round(t1 - t0, 3)
                row["output_ids"] = ev["tokens"]
                row["resume"] = ev["resume"]
                row["reused"] = ev["reused"]
                row["pp"] = ev["pp"]
                row["gpu_stats"] = sanitize(sampler.stats(t0, t1))
                if ev["err"]:
                    row["error"] = ev["err"]
                    result["errors"].append(f"row {seq} ({target}x{repeat}): {ev['err']}")
                elif ev["done"] is None:
                    row["error"] = "no DONE line"
                    result["errors"].append(f"row {seq} ({target}x{repeat}): no DONE line")
                else:
                    d = ev["done"]
                    row["generated"] = int(d[0])
                    row["prompt_tokens"] = int(d[1])
                    row["prompt_ms"] = float(d[2])
                    row["decode_ms"] = float(d[3])
                    row["finish"] = d[4]
                    row["draft_accepted"] = int(d[5])
                    row["draft_offered"] = int(d[6])
                    row["done_reused"] = int(d[7])
                    row["prompt_read"] = int(d[13]) if len(d) > 13 else None
                    fresh = row["prompt_tokens"] - row["done_reused"]
                    row["prompt_tps"] = round(1000.0 * fresh / row["prompt_ms"], 2) if row["prompt_ms"] else None
                    if row["prompt_tokens"] != target:
                        row["error"] = f"engine prompt_tokens {row['prompt_tokens']} != requested {target}"
                    if row["done_reused"] != 0 or (row["resume"] or 0) != 0 or (row["reused"] or 0) != 0:
                        row["error"] = f"conversation reuse detected (resume={row['resume']} reused={row['reused']} done_reused={row['done_reused']})"
                    if row["generated"] < 1:
                        row["error"] = "no generated token"
                if row["error"]:
                    result["errors"].append(f"row {seq} ({target}x{repeat}): {row['error']}")
                result["rows"].append(row)
                print(f"[harness] {target} r{repeat}: prompt={row.get('prompt_tokens')} "
                      f"reused={row.get('done_reused')} prompt_ms={row.get('prompt_ms')} "
                      f"tps={row.get('prompt_tps')} tok0={row['output_ids'][:1]} err={row['error']}", file=sys.stderr)

        # summary per target over the clean rows
        for target in targets:
            rows = [r for r in result["rows"] if r["target"] == target and not r["error"]]
            if not rows:
                result["summary"][str(target)] = {"repeats": 0}
                continue
            pms = [r["prompt_ms"] for r in rows]
            tps = [r["prompt_tps"] for r in rows]
            result["summary"][str(target)] = {
                "repeats": len(rows),
                "prompt_tokens": rows[0]["prompt_tokens"],
                "prompt_ms_all": pms,
                "prompt_ms_median": statistics.median(pms),
                "prompt_tps_all": tps,
                "prompt_tps_median": statistics.median(tps),
                "output_first_ids": [r["output_ids"][:1] for r in rows],
            }
    except (TimeoutError, EOFError, RuntimeError, OSError) as e:
        result["errors"].append(f"fatal: {type(e).__name__}: {e}")
        result["status"] = "error"
    except SystemExit as e:
        result["errors"].append(f"fatal: {e}")
        result["status"] = "error"
    finally:
        if io is not None and proc is not None and proc.poll() is None:
            quit_engine(proc, io, logf)
        elif logf is not None:
            try:
                logf.close()
            except OSError:
                pass
        sampler.stop()
        protocol_f.close()
        result["finished_epoch_s"] = time.time()
        if result["errors"] and result["status"] == "ok":
            result["status"] = "partial" if result["rows"] else "error"
        result["sampler_errors"] = sampler.errors
        result_path = args.out / f"result-{label}.json"
        result_path.write_text(json.dumps(sanitize(result), indent=2) + "\n")
        (args.out / f"summary-{label}.json").write_text(json.dumps(sanitize(result["summary"]), indent=2) + "\n")
        print(f"[harness] wrote {SCRUB(str(result_path))}", file=sys.stderr)

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
