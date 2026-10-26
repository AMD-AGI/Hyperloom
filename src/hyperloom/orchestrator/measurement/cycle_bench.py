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

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

VERDICT_PASS = "pass"
VERDICT_NEUTRAL = "neutral"
VERDICT_REGRESS = "regress"
#: The arm produced no cycle to grade: its server did not come up, or never reached steady decode.
VERDICT_BROKEN = "broken"

#: Status of an explore result that ran a cycle-bench batch rather than a replay round.
STATUS_SCREENED = "cycle_screened"

#: Server flags that bring up the host KV cache; the bench serves from device memory only.
_HICACHE_FLAGS = ("--enable-hierarchical-cache",)
_HICACHE_VALUE_PREFIX = "--hicache-"

_SIMULATED_ACCEPTANCE = "SGLANG_SIMULATE_ACC_"
_TOKEN_MODE = "SGLANG_SIMULATE_ACC_TOKEN_MODE"

#: A TP0 decode log line: running requests, accept length, generation throughput, queued requests.
DECODE_LINE = re.compile(
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
        match = DECODE_LINE.search(line)
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
        match = DECODE_LINE.search(line)
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


@dataclass(frozen=True)
class Bands:
    """How far two bases may differ on the weighted cycle and at each size, and how slow a candidate regresses."""

    batch_pct: float
    per_bs_pct: float
    regress_pct: float


def batch_valid(
    base: Mapping[int, float], base2: Mapping[int, float], weights: Mapping[int, float], *, bands: Bands
) -> bool:
    """Whether the two bases of a batch agree on the weighted cycle and at every weighted batch size."""
    if abs(weighted_gain_pct(base, base2, weights)) > bands.batch_pct:
        return False
    sizes = [bs for bs, weight in weights.items() if weight > 0 and bs in base and bs in base2 and base[bs] > 0]
    return all(abs(base2[bs] - base[bs]) / base[bs] * 100.0 <= bands.per_bs_pct for bs in sizes)


def screen_verdict(gain_pct: float, *, regress_pct: float) -> str:
    """``regress`` past *regress_pct* slower, ``pass`` for any positive estimate, else ``neutral``."""
    if gain_pct < -regress_pct:
        return VERDICT_REGRESS
    return VERDICT_PASS if gain_pct > 0 else VERDICT_NEUTRAL


__all__ = [
    "Bands",
    "DECODE_LINE",
    "STATUS_SCREENED",
    "VERDICT_BROKEN",
    "VERDICT_NEUTRAL",
    "VERDICT_PASS",
    "VERDICT_REGRESS",
    "batch_valid",
    "batch_weights",
    "bench_argv",
    "bench_env",
    "decode_cycles_ms",
    "screen_verdict",
    "weighted_gain_pct",
]
