# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One cycle-bench arm: boot the bench server, hold each batch size in steady decode for one log window, tear down.

The arithmetic -- the bench-shaped argv and env, reading cycles off the log, weights and verdicts -- lives in
:mod:`hyperloom.orchestrator.measurement.cycle_bench`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import statistics
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.measurement import cycle_bench as cb
from hyperloom.orchestrator.measurement.cycle_bench import DECODE_LINE, cycle_bench_enabled, decode_cycles_ms

from ._integrate_vocabulary import ATTRIBUTION_CYCLE_PENDING
from .bypass_engine import wait_for_server_ready
from .bypass_runner import _launch_server, _terminate_server

log = logging.getLogger(__name__)

#: Explore task param carrying a cycle-bench batch: ``[{"name": ..., "envs": {...}}]`` per candidate.
CYCLE_ARMS_PARAM = "cycle_arms"

#: Batch sizes the bench drives; the baseline replay's running counts are weighted onto them.
BATCH_GRID: tuple[int, ...] = (1, 2, 4)

#: About 2x / 3x the two-boot stdev of fixed-mode boots on GLM-5.3 (ROCm 10): 0.5% weighted, up to 1.3% at one size.
CYCLE_BANDS = cb.Bands(batch_pct=1.0, per_bs_pct=3.0, regress_pct=1.5)

#: Admission price of one arm: a warm boot without the host KV cache plus one log window per batch size.
CYCLE_ARM_MINUTES = 8.0

#: Candidates per batch: with its two bases, about an hour of serving lanes.
CYCLE_BATCH_MAX_ARMS = 6

#: Decode steps one attribution capture records: an eighth of the opening profile's, so the orchestrator parsing it
#: into the module table holds a fraction of that trace in memory.
ATTRIBUTION_STEPS = 32

#: Explore ``reason`` of a Coordinator-enqueued cycle-bench batch.
CYCLE_BATCH_REASON = "cycle_bench_batch"

#: Arm key naming the dormant-lever bundle whose switches the arm turns on together, by the bundle's round key.
BUNDLE_ARM_KEY = "lever_bundle"

#: A bundle screen whose batch ended without grading the bundle's arm.
BUNDLE_INCONCLUSIVE = "inconclusive"

#: Bundle screens that keep the bundle from its replay round: its levers did not boot, or ran slower, together.
BUNDLE_REFUSED = frozenset({cb.VERDICT_BROKEN, cb.VERDICT_REGRESS})
_BASES = ("base", "base2")
_ENCODING = "utf-8"
_SERVER_LOG = "server.log"


@dataclass(frozen=True)
class BatchSpec:
    """One cycle-bench batch: the stack's launch, the weights, the candidates' switch envs, and where it runs.

    *reference* is the previous batch's base2 on the same stack, the tree before this batch's patches were admitted.
    A nonzero *attribution_bs* has the base also record a profiler capture at that batch size.
    """

    argv: list[str]
    env: dict[str, str]
    weights: dict[int, float]
    candidates: list[tuple[str, dict[str, str]]]
    out_dir: Path
    port: int
    prompt_tokens: int
    vocab_size: int
    bands: cb.Bands
    reference: dict[int, float] | None = None
    attribution_bs: int = 0


def cycle_arms(levers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One arm per patch among *levers* awaiting the screen, oldest first, up to the batch's arm limit."""
    arms: dict[str, dict[str, str]] = {}
    for row in levers:
        if row.get("attribution_source") == ATTRIBUTION_CYCLE_PENDING:
            patch = str(row.get("specialist_task_id") or row.get("switch") or "")
            arms.setdefault(patch, {})[str(row["switch"])] = str(row.get("value") or "1")
    return [{"name": name, "envs": envs} for name, envs in arms.items()][:CYCLE_BATCH_MAX_ARMS]


def bundle_arm(key: str, envs: dict[str, str]) -> dict[str, Any]:
    """The arm that switches every lever of the bundle round *key* on together."""
    return {"name": "lever-bundle", "envs": dict(envs), BUNDLE_ARM_KEY: key}


def _weighting_log(launch_dir: Path, baseline: str) -> Path:
    """The server log to weight batch sizes by: the launch's own, else the baseline's, as one corpus drives both."""
    own = launch_dir / _SERVER_LOG
    if own.is_file() or not baseline:
        return own
    return next(iter(sorted(Path(baseline).rglob(_SERVER_LOG))), own)


def session_batch(state: Any, candidates: list[tuple[str, dict[str, str]]], *, out_dir: Path, port: int) -> BatchSpec:
    """A batch on the current best's own launch, weighted by where its replay ran and fed its corpus length.

    Until a round moves the current best that is the baseline's; after, the kept stack is what a candidate joins.
    """
    best = str((getattr(state, "current_best", None) or {}).get("workspace") or "")
    root = best or str(getattr(state, "baseline_workspace", "") or "")
    if not root:
        raise FileNotFoundError("the session has no baseline workspace to take the bench launch from")
    launch_file = next(iter(sorted(Path(root).rglob("agentx_server_launch.json"))), None)
    if launch_file is None:
        where = "the current best" if best else "the baseline workspace"
        raise FileNotFoundError(f"no agentx_server_launch.json under {where} {root}")
    launch = json.loads(launch_file.read_text(encoding=_ENCODING))
    log_path = _weighting_log(launch_file.parent, str(getattr(state, "baseline_workspace", "") or ""))
    weights = cb.batch_weights(log_path.read_text(encoding=_ENCODING, errors="ignore").splitlines(), grid=BATCH_GRID)
    if not weights:
        raise ValueError(f"no decode line in {log_path} to weight batch sizes by")
    config = json.loads((Path(str(state.model_path)) / "config.json").read_text(encoding=_ENCODING))
    isl = (getattr(state, "agentx_corpus_shape", None) or {}).get("isl") or {}
    return BatchSpec(
        argv=list(launch["effective_argv"]),
        env={**(launch.get("runtime_environment") or {}), **(launch.get("effective_env") or {})},
        weights=weights,
        candidates=list(candidates),
        out_dir=out_dir,
        port=port,
        prompt_tokens=int(isl["p50"]),
        vocab_size=int(config["vocab_size"]),
        bands=CYCLE_BANDS,
        reference=_reference(state),
        attribution_bs=_attribution_bs(state, weights),
    )


def _attribution_bs(state: Any, weights: dict[int, float]) -> int:
    """The heaviest batch size when a kept round moved the stack and no capture describes it yet, else 0."""
    stack_len = len(getattr(state, "optimization_stack", None) or [])
    held = getattr(state, "last_attribution_trace", None) or {}
    if not stack_len or held.get("stack_len") == stack_len:
        return 0
    return max(weights, key=lambda bs: weights[bs])


#: Source paths a steady-decode batch without the host KV cache never exercises: tokenization, the API front end,
#: and the prefix / hierarchical KV caches. A patch touching one is measured by its replay round instead.
_UNSEEN_PATHS = (
    "/srt/entrypoints/",
    "/srt/mem_cache/",
    "/srt/managers/tokenizer_manager",
    "/srt/managers/detokenizer_manager",
    "/srt/managers/template_manager",
    "/srt/managers/multi_tokenizer",
)


def cycle_bench_sees(files: list[str]) -> bool:
    """Whether every file a patch touches runs inside the steady decode the cycle bench measures."""
    return not any(marker in f"/{path}" for path in files for marker in _UNSEEN_PATHS)


def screen_open(state: Any) -> bool:
    """Whether a switched patch may be admitted dormant now: the screen is on and open, and the stack has a reference."""
    return (
        cycle_bench_enabled()
        and state is not None
        and not getattr(state, "cycle_screen_refused", "")
        and _reference(state) is not None
    )


def _reference(state: Any) -> dict[int, float] | None:
    """The previous valid batch's base2 cycle, when it ran the stack the session holds now."""
    ref = getattr(state, "cycle_reference", None) or {}
    if not ref or int(ref.get("stack_len", -1)) != len(getattr(state, "optimization_stack", None) or []):
        return None
    return {int(bs): float(ms) for bs, ms in (ref.get("cycle_ms") or {}).items()} or None


def run_batch(spec: BatchSpec, stop: threading.Event | None = None) -> dict[str, Any]:
    """Run ``base``, each candidate, then ``base2``, one server boot each, and grade the candidates.

    A base off the reference is booted once more; off it again, the admitted patches moved the switched-off tree and
    the batch stops before its candidates. Once *stop* is set no further arm boots.
    """
    arms = {_BASES[0]: _boot(spec, _BASES[0], {}, stop, attribution_bs=spec.attribution_bs)}
    if not arms[_BASES[0]].error and not _matches_reference(spec, arms[_BASES[0]]):
        arms[_BASES[0]] = _boot(spec, "base-rerun", {}, stop, attribution_bs=spec.attribution_bs)
        if not arms[_BASES[0]].error and not _matches_reference(spec, arms[_BASES[0]]):
            gap = cb.weighted_gain_pct(spec.reference or {}, arms[_BASES[0]].cycle_ms, spec.weights)
            reason = f"the switched-off tree runs {gap:+.2f}% against the previous batch's base2"
            return _graded(spec, arms, {"valid": False, "parity_broken": True, "reason": reason, "verdicts": {}})
    if not arms[_BASES[0]].error:
        for name, switches in [*spec.candidates, (_BASES[1], {})]:
            arms[name] = _boot(spec, name, switches, stop)
    summary = summarize_batch(arms, [name for name, _ in spec.candidates], spec.weights, bands=spec.bands)
    return _graded(spec, arms, {**summary, "parity_broken": False})


def _matches_reference(spec: BatchSpec, base: ArmResult) -> bool:
    return not spec.reference or cb.batch_valid(spec.reference, base.cycle_ms, spec.weights, bands=spec.bands)


def _graded(spec: BatchSpec, arms: dict[str, ArmResult], summary: dict[str, Any]) -> dict[str, Any]:
    return {**summary, "arms": {name: arm.__dict__ for name, arm in arms.items()}, "weights": spec.weights}


def _boot(
    spec: BatchSpec, name: str, switches: dict[str, str], stop: threading.Event | None, *, attribution_bs: int = 0
) -> ArmResult:
    if stop is not None and stop.is_set():
        return ArmResult(error="the batch was stopped before this arm")
    return run_arm(
        ArmSpec(
            argv=cb.bench_argv(spec.argv, port=spec.port),
            env=cb.bench_env({**os.environ, **spec.env, **switches}),
            out_dir=spec.out_dir / name,
            port=spec.port,
            batch_sizes=tuple(bs for bs in BATCH_GRID if spec.weights.get(bs, 0) > 0) or BATCH_GRID,
            prompt_tokens=spec.prompt_tokens,
            vocab_size=spec.vocab_size,
            attribution_bs=attribution_bs,
        )
    )


def summarize_batch(
    arms: dict[str, ArmResult], candidates: list[str], weights: dict[int, float], *, bands: cb.Bands
) -> dict[str, Any]:
    """Grade each candidate against the mean of the two bases, once the bases agree within *bands*."""
    base, base2 = arms.get(_BASES[0]), arms.get(_BASES[1])
    if base is None or base2 is None or base.error or base2.error:
        return {"valid": False, "reason": "a base arm produced no cycle", "verdicts": {}}
    if not {bs for bs, w in weights.items() if w > 0} & set(base.cycle_ms) & set(base2.cycle_ms):
        return {"valid": False, "reason": "no weighted batch size was measured by both bases", "verdicts": {}}
    gap = cb.weighted_gain_pct(base.cycle_ms, base2.cycle_ms, weights)
    if not cb.batch_valid(base.cycle_ms, base2.cycle_ms, weights, bands=bands):
        return {"valid": False, "reason": f"bases disagree by {gap:+.2f}%", "verdicts": {}}
    mean = {bs: (base.cycle_ms[bs] + base2.cycle_ms[bs]) / 2 for bs in base.cycle_ms if bs in base2.cycle_ms}
    verdicts: dict[str, dict[str, Any]] = {}
    for name in candidates:
        arm = arms.get(name)
        if arm is None or arm.error:
            verdicts[name] = {"verdict": cb.VERDICT_BROKEN, "error": arm.error if arm else "arm did not run"}
            continue
        gain = cb.weighted_gain_pct(mean, arm.cycle_ms, weights)
        verdict = cb.screen_verdict(gain, regress_pct=bands.regress_pct)
        verdicts[name] = {"verdict": verdict, "gain_pct": round(gain, 3), "accept_len": arm.accept_len}
    return {"valid": True, "base_gap_pct": round(gap, 3), "verdicts": verdicts}


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
    attribution_bs: int = 0
    attribution_steps: int = ATTRIBUTION_STEPS


@dataclass
class ArmResult:
    """What one arm measured: cycle medians per batch size, the acceptance it ran at, and why it is unusable."""

    cycle_ms: dict[int, float] = field(default_factory=dict)
    accept_len: dict[int, float] = field(default_factory=dict)
    samples: dict[int, int] = field(default_factory=dict)
    boot_s: float | None = None
    error: str = ""
    attribution_trace: str = ""


def run_arm(spec: ArmSpec) -> ArmResult:
    """Boot the server, hold each batch size in steady decode for one log window, then tear the server down.

    With an *attribution_bs*, the measured server then records a profiler capture at that size; the capture runs
    after every window, so it never touches the cycles.
    """
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = spec.out_dir / _SERVER_LOG
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
            window = _hold_steady_decode(url, log_path, _prompts(rng, spec, bs), spec)
            cycles = decode_cycles_ms(window, bs=bs) if window is not None else []
            if not cycles:
                result.error = f"no steady decode at bs={bs}"
                return result
            result.cycle_ms[bs] = statistics.median(cycles)
            result.samples[bs] = len(cycles)
            result.accept_len[bs] = _mean_accept(window, bs)
        if spec.attribution_bs:
            result.attribution_trace = _capture_attribution(url, log_path, spec, rng)
        return result
    finally:
        _terminate_server(proc)
        (spec.out_dir / "arm.json").write_text(json.dumps(result.__dict__, default=str, indent=1), encoding=_ENCODING)


def _prompts(rng: random.Random, spec: ArmSpec, bs: int) -> list[list[int]]:
    return [[rng.randrange(1000, spec.vocab_size) for _ in range(spec.prompt_tokens)] for _ in range(bs)]


def _hold_steady_decode(url: str, log_path: Path, prompts: list[list[int]], spec: ArmSpec) -> list[str] | None:
    """Send *prompts*, wait until all of them decode with an empty queue, and return one log window's lines."""
    with _steady_decode(url, log_path, prompts, spec) as steady:
        if not steady:
            return None
        start = _line_count(log_path)
        time.sleep(spec.window_s)
        return _read_lines(log_path)[start:]


def _capture_attribution(url: str, log_path: Path, spec: ArmSpec, rng: random.Random) -> str:
    """Hold the attribution batch size in steady decode and record a profiler capture of it; ``""`` without one."""
    with _steady_decode(url, log_path, _prompts(rng, spec, spec.attribution_bs), spec) as steady:
        if not steady:
            log.warning("cycle bench: no steady decode at bs=%d to capture", spec.attribution_bs)
            return ""
        return record_profile(url, (spec.out_dir / "attribution").resolve(), spec.attribution_steps)


@contextlib.contextmanager
def _steady_decode(url: str, log_path: Path, prompts: list[list[int]], spec: ArmSpec) -> Iterator[bool]:
    """Send *prompts* and yield, once settled, whether all of them decode with an empty queue; abort them after."""
    since = _line_count(log_path)
    for ids in prompts:
        threading.Thread(target=_generate, args=(url, ids, spec.max_new_tokens), daemon=True).start()
    try:
        steady = _wait_steady(log_path, since, len(prompts), timeout_s=900.0)
        if steady:
            time.sleep(spec.settle_s)
        yield steady
    finally:
        _post(url + "/abort_request", {"abort_all": True})
        time.sleep(5.0)
        _post(url + "/flush_cache", None)


def record_profile(url: str, trace_dir: Path, steps: int, *, timeout_s: float = 600.0, poll_s: float = 5.0) -> str:
    """Record *steps* steps with sglang's torch profiler into *trace_dir*; the first rank's capture, or ``""``.

    Step annotations give the module table its windows; Python stacks and shapes would only multiply the trace size.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "output_dir": str(trace_dir),
        "num_steps": steps,
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": False,
        "detailed_annotations": True,
        "merge_profiles": False,
        "profile_by_stage": False,
    }
    if not _post(url + "/start_profile", body, timeout=timeout_s):
        log.warning("cycle bench: %s refused /start_profile; the module table keeps its previous source", url)
        return ""
    captures = _settled_captures(trace_dir, timeout_s=timeout_s, poll_s=poll_s)
    if not captures:
        log.warning("cycle bench: no complete profiler capture under %s", trace_dir)
        return ""
    return str(next((path for path in captures if "-TP-0" in path.name), captures[0]))


def _settled_captures(trace_dir: Path, *, timeout_s: float, poll_s: float) -> list[Path]:
    """The capture files once two polls in a row see the same names and sizes; empty when none settle in time."""
    deadline = time.monotonic() + timeout_s
    seen: list[tuple[str, int]] = []
    while time.monotonic() < deadline:
        files = sorted(trace_dir.glob("*.trace.json*"))
        sizes = [(path.name, path.stat().st_size) for path in files]
        if sizes and sizes == seen:
            return files
        seen = sizes
        time.sleep(poll_s)
    return []


def _generate(url: str, ids: list[int], max_new: int) -> None:
    sampling = {"max_new_tokens": max_new, "ignore_eos": True, "temperature": 0}
    _post(url + "/generate", {"input_ids": ids, "sampling_params": sampling}, timeout=7200.0)


def _post(url: str, body: Any, timeout: float = 60.0) -> bool:
    """Whether the server answered with a success status."""
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # nosec B310 - local bench server
            return True
    except OSError:
        return False  # an aborted generate or a slow flush; the log window is the measurement


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
        return path.read_text(encoding=_ENCODING, errors="ignore").splitlines()
    except OSError:
        return []


def _line_count(path: Path) -> int:
    return len(_read_lines(path))


__all__ = [
    "BATCH_GRID",
    "ArmResult",
    "ArmSpec",
    "BatchSpec",
    "record_profile",
    "run_arm",
    "run_batch",
    "session_batch",
    "summarize_batch",
]
