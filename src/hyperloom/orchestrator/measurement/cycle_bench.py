# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A short serving measurement: the decode cycle time of the recipe's server at the batch sizes the replay ran.

A full agentic replay takes over an hour per arm and moves by several percent between runs, so a change worth one
percent cannot be read off it. This boots the same server without its host KV cache, holds ``bs`` long requests in
steady decode, and reads the cycle time off the scheduler's decode log lines (``bs * accept_len / gen_throughput``).
Arms of one batch share a fixed token stream (simulated acceptance pinned to ``fixed``) and a fixed prompt set, and
are compared against two base arms that bracket them.

The figure it reports is a weighted decode-cycle gain, not an end-to-end prediction: how much of an agentic metric a
decode saving buys is calibrated by the session against its own paired runs.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERDICT_PASS = "pass"
VERDICT_NEUTRAL = "neutral"
VERDICT_REGRESS = "regress"

#: Server flags that bring up the host KV cache; the bench serves from device memory only.
_HICACHE_FLAGS = ("--enable-hierarchical-cache",)
_HICACHE_VALUE_PREFIX = "--hicache-"

_SIMULATED_ACCEPTANCE = "SGLANG_SIMULATE_ACC_"
_TOKEN_MODE = "SGLANG_SIMULATE_ACC_TOKEN_MODE"

_DECODE = re.compile(
    r"TP0.*Decode batch, #running-req: (\d+),.*accept len: ([\d.]+).*"
    r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)


def bench_argv(argv: Sequence[str], *, port: int) -> list[str]:
    """The recipe's server argv without its host KV cache flags, listening on *port*."""
    out: list[str] = []
    items = list(argv)
    i = 0
    while i < len(items):
        arg = items[i]
        if arg in _HICACHE_FLAGS:
            i += 1
            continue
        if arg.startswith(_HICACHE_VALUE_PREFIX):
            i += 1 if "=" in arg else 2
            continue
        if arg == "--port":
            out += ["--port", str(port)]
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def bench_env(env: Mapping[str, str]) -> dict[str, str]:
    """The recipe's env with simulated acceptance pinned to a fixed token stream, when the recipe simulates it."""
    out = dict(env)
    if any(key.startswith(_SIMULATED_ACCEPTANCE) for key in out):
        out[_TOKEN_MODE] = "fixed"
    return out


def decode_cycles_ms(lines: Iterable[str], *, bs: int) -> list[float]:
    """Cycle times, one per TP0 decode line logged at exactly *bs* running requests and an empty queue."""
    cycles: list[float] = []
    for line in lines:
        match = _DECODE.search(line)
        if not match or int(match[1]) != bs or int(match[4]) != 0:
            continue
        accept, tput = float(match[2]), float(match[3])
        if tput > 0:
            cycles.append(bs * accept / tput * 1e3)
    return cycles


def batch_weights(lines: Iterable[str], *, grid: Sequence[int]) -> dict[int, float]:
    """Share of the replay's decode lines at each grid batch size, each running count rounded up to the grid and
    anything above it counted at its largest size."""
    counts: Counter[int] = Counter()
    ordered = sorted(grid)
    for line in lines:
        match = _DECODE.search(line)
        if match:
            running = int(match[1])
            counts[next((bs for bs in ordered if running <= bs), ordered[-1])] += 1
    total = sum(counts.values())
    return {bs: round(counts[bs] / total, 6) for bs in ordered} if total else {}


def weighted_gain_pct(base: Mapping[int, float], cand: Mapping[int, float], weights: Mapping[int, float]) -> float:
    """Weighted share of the base cycle the candidate saves, in percent; positive is faster."""
    shared = [bs for bs in weights if bs in base and bs in cand and base[bs] > 0]
    total = sum(weights[bs] for bs in shared)
    if not total:
        return 0.0
    return sum(weights[bs] * (base[bs] - cand[bs]) / base[bs] for bs in shared) / total * 100.0


def batch_valid(
    base: Mapping[int, float], base2: Mapping[int, float], weights: Mapping[int, float], *, band_pct: float
) -> bool:
    """Whether the two bases of a batch agree within *band_pct* on the weighted cycle."""
    return abs(weighted_gain_pct(base, base2, weights)) <= band_pct


def screen_verdict(gain_pct: float, *, band_pct: float) -> str:
    """``regress`` past the band, ``pass`` for any positive estimate, else ``neutral``."""
    if gain_pct < -band_pct:
        return VERDICT_REGRESS
    return VERDICT_PASS if gain_pct > 0 else VERDICT_NEUTRAL


@dataclass(frozen=True)
class ArmSpec:
    """One server boot: its argv and env (already bench-shaped), where it writes, and what it drives."""

    argv: list[str]
    env: dict[str, str]
    out_dir: Path
    port: int
    batch_sizes: tuple[int, ...]
    prompt_tokens: int
    vocab_size: int
    seed: int = 0
    max_new_tokens: int = 60000
    settle_s: float = 10.0
    window_s: float = 60.0
    boot_timeout_s: float = 1800.0


@dataclass
class ArmResult:
    """What one arm measured: cycle medians per batch size, the acceptance it ran at, and why it is unusable."""

    cycle_ms: dict[int, float] = field(default_factory=dict)
    accept_len: dict[int, float] = field(default_factory=dict)
    samples: dict[int, int] = field(default_factory=dict)
    boot_s: float | None = None
    error: str = ""


def run_arm(spec: ArmSpec) -> ArmResult:
    """Boot the server, hold each batch size in steady decode for one log window, then tear the server down."""
    from hyperloom.orchestrator.actions.executors.bypass_engine import wait_for_server_ready
    from hyperloom.orchestrator.actions.executors.bypass_runner import _launch_server, _terminate_server

    spec.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = spec.out_dir / "server.log"
    result = ArmResult()
    started = time.monotonic()
    proc = _launch_server(spec.argv, spec.env, log_path)
    try:
        url = f"http://127.0.0.1:{spec.port}"
        if not wait_for_server_ready(url, timeout_s=spec.boot_timeout_s, server_exited=lambda: proc.poll() is not None):
            result.error = "server did not become ready"
            return result
        result.boot_s = round(time.monotonic() - started, 1)
        rng = random.Random(spec.seed)
        for bs in spec.batch_sizes:
            prompts = [[rng.randrange(1000, spec.vocab_size) for _ in range(spec.prompt_tokens)] for _ in range(bs)]
            window = _hold_steady_decode(url, log_path, prompts, spec)
            cycles = decode_cycles_ms(window, bs=bs) if window is not None else []
            if not cycles:
                result.error = f"no steady decode at bs={bs}"
                return result
            result.cycle_ms[bs] = statistics.median(cycles)
            result.samples[bs] = len(cycles)
            result.accept_len[bs] = _mean_accept(window, bs)
        return result
    finally:
        _terminate_server(proc)
        (spec.out_dir / "arm.json").write_text(json.dumps(result.__dict__, default=str, indent=1), encoding="utf-8")


def _hold_steady_decode(url: str, log_path: Path, prompts: list[list[int]], spec: ArmSpec) -> list[str] | None:
    """Send *prompts*, wait until all of them decode with an empty queue, and return one log window's lines."""
    since = _line_count(log_path)
    for ids in prompts:
        threading.Thread(target=_generate, args=(url, ids, spec.max_new_tokens), daemon=True).start()
    try:
        if not _wait_steady(log_path, since, len(prompts), timeout_s=900.0):
            return None
        time.sleep(spec.settle_s)
        start = _line_count(log_path)
        time.sleep(spec.window_s)
        return _read_lines(log_path)[start:]
    finally:
        _post(url + "/abort_request", {"abort_all": True})
        time.sleep(5.0)
        _post(url + "/flush_cache", None)


def _generate(url: str, ids: list[int], max_new: int) -> None:
    sampling = {"max_new_tokens": max_new, "ignore_eos": True, "temperature": 0}
    _post(url + "/generate", {"input_ids": ids, "sampling_params": sampling}, timeout=7200.0)


def _post(url: str, body: Any, timeout: float = 60.0) -> None:
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # nosec B310 - local bench server
            pass
    except OSError:
        pass  # an aborted generate or a slow flush; the log window is the measurement


def _wait_steady(log_path: Path, since: int, bs: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hits = [m for m in map(_DECODE.search, _read_lines(log_path)[since:]) if m]
        if hits and int(hits[-1][1]) == bs and int(hits[-1][4]) == 0:
            return True
        time.sleep(2.0)
    return False


def _mean_accept(lines: list[str], bs: int) -> float:
    values = [float(m[2]) for m in map(_DECODE.search, lines) if m and int(m[1]) == bs and int(m[4]) == 0]
    return round(statistics.mean(values), 4) if values else 0.0


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []


def _line_count(path: Path) -> int:
    return len(_read_lines(path))


__all__ = [
    "ArmResult",
    "ArmSpec",
    "VERDICT_NEUTRAL",
    "VERDICT_PASS",
    "VERDICT_REGRESS",
    "batch_valid",
    "batch_weights",
    "bench_argv",
    "bench_env",
    "decode_cycles_ms",
    "run_arm",
    "screen_verdict",
    "weighted_gain_pct",
]
