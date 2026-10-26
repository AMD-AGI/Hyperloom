# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One cycle-bench arm: boot the bench server, hold each batch size in steady decode for one log window, tear down.

The arithmetic -- the bench-shaped argv and env, reading cycles off the log, weights and verdicts -- lives in
:mod:`hyperloom.orchestrator.measurement.cycle_bench`.
"""

from __future__ import annotations

import json
import random
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.measurement.cycle_bench import DECODE_LINE, decode_cycles_ms

from .bypass_engine import wait_for_server_ready
from .bypass_runner import _launch_server, _terminate_server


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
        hits = [m for m in map(DECODE_LINE.search, _read_lines(log_path)[since:]) if m]
        if hits and int(hits[-1][1]) == bs and int(hits[-1][4]) == 0:
            return True
        time.sleep(2.0)
    return False


def _mean_accept(lines: list[str], bs: int) -> float:
    values = [float(m[2]) for m in map(DECODE_LINE.search, lines) if m and int(m[1]) == bs and int(m[4]) == 0]
    return round(statistics.mean(values), 4) if values else 0.0


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []


def _line_count(path: Path) -> int:
    return len(_read_lines(path))


__all__ = ["ArmResult", "ArmSpec", "run_arm"]
