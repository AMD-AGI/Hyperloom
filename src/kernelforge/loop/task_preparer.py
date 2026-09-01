# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-loop task preparation ("self-healing") for forge-loop.

Callers of forge-loop do not always hand it a task that already meets the driver
contract: some pass a driver that prints the wrong lines, some pass none at all.
This module runs BEFORE the optimization loop and, when needed, invokes a single
LLM agent to author/repair the measurement scaffolding (a ``driver.py`` and any
helper files it needs) so the task becomes optimizable — WITHOUT ever touching
the kernel/source being optimized.

Design (mirrors ``source_map.py`` for the pre-loop LLM pattern):

  * ``preflight_task`` — deterministic gate. Reuses the exact tools forge-loop
    itself uses to read a driver (``test_correctness`` + ``bench_wallclock``), so
    "conforms" here means "conforms to what the loop will parse".
  * ``prepare_task`` — bounded repair loop. Protects ONLY the source under
    optimization, lets the agent freely author/modify the driver and any other
    non-source files, then re-runs the deterministic preflight as the
    authoritative verdict. On success it commits the scaffolding; on failure it
    rolls the workspace back (via git + a source byte-snapshot) and reports it.

Guarantees requested by the integration:
  1. Source protection ONLY — the kernel and every ``source_files`` entry are
     restored after each attempt and on failure; the agent is told they are
     off-limits. Every OTHER file (driver, helpers, configs) is fair game.
  2. CUDA/HIP graph timing is strongly recommended and handed to the agent as an
     embedded reference harness, but NOT forced: an operator that cannot be
     captured into a static-input graph may use equivalent GPU-only timing.
  3. Explicit return contract — ``PrepareResult`` reports success, or rolls back
     and reports failure (``rolled_back=True``).
  4. Hard wall-clock budget with no orphan processes — the agent CLI runs in its
     own session/process group and is killed with ``killpg`` on timeout.
  5. Git-safe — prep commits BEFORE the loop captures its pristine ``base_sha``,
     so scaffolding is pristine, not part of the solution diff, and never
     collides with the loop's keep/revert.
  6. Preflight judges the driver in the same filesystem state the loop's baseline
     will run it in. Authoring-only material (the reference example bundle) is
     retired BEFORE the verdict, and the one task input a driver may legitimately
     read at runtime — the invocation specification — is durable and committed
     with the driver. Validating a state that preparation then dismantles once
     certified a driver that crashed on the very first baseline bench.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import glob
import logging
import os
import pathlib
import re
import shutil
import signal
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import NamedTuple
from pathlib import Path

from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
    with_writable_sandbox,
)
from kernelforge.agent_backends.registry import create_registered_backend
from kernelforge.llm.git import git
from kernelforge.config import Config
from kernelforge.loop.external_artifacts import (
    ExternalArtifactError,
    ExternalArtifactTransaction,
)
from kernelforge.loop.profile_contract import PROFILE_RUN_FLAG
from kernelforge.mcp_server.tools.bench import bench_wallclock
from kernelforge.mcp_server.tools.test import test_correctness
from kernelforge.resources import resource_path
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

log = logging.getLogger(__name__)


# Bounded repair budget. PREPARE_MAX_WALL_SEC is a single deadline across ALL
# attempts; each attempt is additionally capped by PER_ATTEMPT_CAP_SEC.
#
# COLD-JIT SIZING: this wall must cover the driver-gen agent (~600-900s of LLM
# authoring) PLUS the deterministic preflight run, whose correctness/bench stages
# JIT-compile CK/aiter GEMM kernels on first run (~44s+/module, serial baton-lock
# on gfx950). At the old 1200s a slow author left <5min for a cold preflight, so
# the preflight timed out (clamped by the remaining wall, never reaching its own
# PREFLIGHT_*_TIMEOUT_S ceilings) -> task_preparation_failed even though the
# driver was fine. Raise the wall so agent + cold preflight both fit; it is
# additionally clamped to the per-kernel deadline_unix (forge_submit passes the
# ~3600s budget), so a larger value never overruns the outer budget.
PREPARE_MAX_ATTEMPTS = int(os.environ.get("FORGE_PREPARE_MAX_ATTEMPTS", "3") or "3")
PREPARE_MAX_WALL_SEC = int(os.environ.get("FORGE_PREPARE_MAX_WALL", "3000") or "3000")
# Derived from the wall so it scales with it; a fixed constant falls below the
# wall/attempts ratio as soon as either grows.
PER_ATTEMPT_CAP_SEC = int(
    os.environ.get("FORGE_PREPARE_ATTEMPT_CAP") or max(1, PREPARE_MAX_WALL_SEC // max(1, PREPARE_MAX_ATTEMPTS))
)
# Smallest budget worth spending on a RETRY. Measured over 25 recorded attempts:
# successful ones ran 350-896s, and every retry that started with less than that
# floor (150s, 298s, 300s, 325s) burned its whole budget without writing a byte.
# A first attempt always runs, however little time it has — a long shot is still
# better than not trying — but handing the scraps to a retry only converts the
# tail of the wall into tokens and a misleading "FAILED after 2 attempts".
PREPARE_MIN_RETRY_SEC = int(os.environ.get("FORGE_PREPARE_MIN_RETRY", "350") or "350")
# Wall seconds reserved per attempt for the salvage preflight after a timeout.
_SALVAGE_RESERVE_SEC: float = float(os.environ.get("FORGE_SALVAGE_RESERVE", "120") or "120")

# Preflight bench is a quick format check, not a real measurement — keep it cheap.
# These deliberately differ from bench_wallclock's measurement defaults (10/30,
# which the loop's baseline and every candidate use): preflight only decides
# whether the driver PRINTS what the loop parses, and the first run of a CK/aiter
# driver JIT-compiles its kernels (44s+ per module). Tripling the timed iterations
# to match the baseline would spend that budget re-measuring a number preflight
# throws away, and cold preflight timeouts have already failed otherwise-valid
# preparations (see PREPARE_MAX_WALL_SEC). The counts a driver is judged on are
# passed to it as --warmup/--iters, so nothing about the contract depends on them.
PREFLIGHT_WARMUP = 3
PREFLIGHT_ITERS = 10

# The prep agent reads REAL reference example files (not just prompt text). We copy
# the shipped examples into this workspace subdir so the agent can Read them within
# its cwd. It is AUTHORING-ONLY scaffolding: it is removed before the deterministic
# preflight that accepts the driver, so preflight judges the driver in the same
# filesystem state the loop's baseline will run it in, and it is never committed.
# Nothing a driver needs at runtime may live here; the invocation specification,
# which a driver legitimately reads, goes beside the driver instead (see
# _materialize_invocation_spec).
REFERENCE_SUBDIR = ".forge_task_reference"
INVOCATION_SPEC_FILENAME = "invocation_spec.json"
MAX_INVOCATION_SPEC_BYTES = 1024 * 1024
#: Above this the spec is referenced by path instead of inlined in the prompt.
#: An observed document runs a few kilobytes -- it is bounded by the operand
#: count -- so this only guards against a producer that grew without warning.
_SPEC_INLINE_MAX_BYTES = 64 * 1024
_INVOCATION_SPEC_NAME_RE = re.compile(r"^invocation_spec_[A-Za-z0-9._-]+\.json$")
_REFERENCE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", "forge_experiments", ".git")


# ---------------------------------------------------------------------------
# Preflight — deterministic driver-contract validation
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """Whether a driver conforms to the forge-loop stdout contract."""

    ok: bool
    correctness_ok: bool
    bench_ok: bool
    graph_ok: bool = True
    profile_ok: bool = True
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    # Raw stdout+stderr tail per failed stage ("correctness", "bench", ...).
    # ``reasons`` only carries the verdict ("DRIVER CRASHED (exit 1)"), which on
    # its own tells the repair agent nothing about WHY — it then burns its whole
    # attempt re-running the driver to rediscover a traceback we already had.
    diagnostics: dict = field(default_factory=dict)
    # Wall time the whole check took, with per-stage seconds in ``details``. The
    # audit recorded no timing at all, so "which stage ate the budget" could only
    # be guessed at from file mtimes — which lie (see _audit_driver).
    duration_sec: float = 0.0

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else ("ok" if self.ok else "failed")

    def detail_report(self) -> str:
        """``summary()`` plus the captured output of every stage that failed."""
        report = self.summary()
        if not self.diagnostics:
            return report
        blocks = [
            f"\n### {stage} stage output (tail)\n```\n{tail.strip()}\n```"
            for stage, tail in self.diagnostics.items()
            if tail and tail.strip()
        ]
        return report + "\n" + "\n".join(blocks) if blocks else report

    @property
    def all_failures_are_timeouts(self) -> bool:
        """True when every primary failure is a TIMEOUT, none a CRASH.

        Cascading reasons like "cannot verify graph timing because bench
        produced no timing" are not primary failures — they just report
        that a downstream check could not run *because an earlier stage
        failed*. However, a "could not verify" reason that itself
        contains a timeout token IS a primary timeout (the graph probe
        ran and timed out).
        """
        if self.ok or not self.reasons:
            return False
        _CASCADING = ("cannot verify", "could not verify")
        _TIMEOUT_TOKENS = ("TIMEOUT", "timed out")
        primary = [r for r in self.reasons if not r.startswith(_CASCADING) or any(t in r for t in _TIMEOUT_TOKENS)]
        if not primary:
            return False
        return all(any(t in r for t in _TIMEOUT_TOKENS) for r in primary) and not any("CRASHED" in r for r in primary)


# Counts ACTUAL torch.cuda.CUDAGraph.replay calls (HIP graphs go through the same
# API on ROCm), detecting real graph timing independently of whatever the driver
# prints: an eager driver replays zero times, a graph-timed one replays once per
# timed iteration.
#
# Installed as a sitecustomize module rather than a wrapper around the driver so
# that it also covers the ranks of a self-launching multi-GPU driver. Those
# re-exec themselves under torchrun, which puts every replay in a child process
# where a wrapper's patch does not exist -- the parent then counts zero and a
# perfectly graph-timed collective driver is rejected as eager. Python imports
# sitecustomize in each of those children too, so each rank counts its own
# replays into $GRAPH_PROBE_OUT.<pid>; the caller validates the rank set and
# uses the least replayed rank.
_GRAPH_PROBE_SITECUSTOMIZE = r'''
import atexit, json, os

_n = [0]


def _ancestor_pids():
    """Capture the process ancestry while launcher and worker parents exist."""
    ancestors = []
    pid = os.getppid()
    for _ in range(64):
        if pid <= 1 or pid in ancestors:
            break
        ancestors.append(pid)
        try:
            stat = open(f"/proc/{pid}/stat").read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return ancestors


_ancestors = _ancestor_pids()
_import_pid = os.getpid()


def _install():
    """Patch CUDAGraph.replay lazily: torch may not be imported yet."""
    try:
        import torch
    except Exception:
        return False
    orig = torch.cuda.CUDAGraph.replay

    def _replay(self, *a, **k):
        _n[0] += 1
        return orig(self, *a, **k)

    torch.cuda.CUDAGraph.replay = _replay
    return True


if not _install():
    # torch is imported by the driver, not by us. Hook the import so the patch
    # lands before any graph is created.
    import builtins

    _real_import = builtins.__import__

    def _hooked(name, *a, **k):
        mod = _real_import(name, *a, **k)
        if name == "torch" or name.startswith("torch."):
            if _install():
                builtins.__import__ = _real_import
        return mod

    builtins.__import__ = _hooked


def _dump():
    out = os.environ.get("GRAPH_PROBE_OUT")
    if not out:
        return
    try:
        # One file per process: ranks of a torchrun job would otherwise
        # overwrite each other and the count would be one rank's, or zero.
        with open(f"{out}.{os.getpid()}", "w") as fh:
            json.dump(
                {
                    "replays": _n[0],
                    "rank": os.environ.get("RANK"),
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "world_size": os.environ.get("WORLD_SIZE"),
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "ancestors": (
                        _ancestors
                        if os.getpid() == _import_pid
                        else _ancestor_pids()
                    ),
                },
                fh,
            )
    except Exception:
        pass


atexit.register(_dump)
'''


# First-run JIT compilation of CK/aiter GEMM kernels on gfx950/rocm (serial
# baton-lock builds, ~44s+ per module) routinely blows the old 120s correctness
# / 300s bench preflight budgets, so task_preparation fails ("could not produce
# a conforming driver within the budget") before the loop even starts — even
# though the driver is fine and just needs to compile once. Give first-run JIT
# real headroom; override via env if a build farm is unusually slow/fast. Same
# root-cause family as kernelforge.gemm_tune's FORGE_TUNE_TASK_TIMEOUT (7200s), a
# different knob on the same JIT-latency problem.
PREFLIGHT_CORRECTNESS_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_CORRECTNESS_TIMEOUT", "1800") or "1800")
PREFLIGHT_BENCH_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_BENCH_TIMEOUT", "1800") or "1800")
# graph-replay and profiling preflight run *after* bench, so the JIT cache is
# usually warm by then, but on a cold first run a fresh module can still compile
# here. Keep them generous and overridable rather than the old bare 300s.
PREFLIGHT_GRAPH_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_GRAPH_TIMEOUT", "900") or "900")
PREFLIGHT_PROFILE_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_PROFILE_TIMEOUT", "900") or "900")


# How much of a failed stage's stdout+stderr to carry into the audit record and
# the repair agent's next prompt. The producing tools already cap their capture
# at 2000 chars; a traceback plus the lines that led to it fits well inside this.
DIAG_TAIL_CHARS = int(os.environ.get("FORGE_PREFLIGHT_DIAG_CHARS", "1500") or "1500")


def _record_stage_output(diagnostics: dict, stage: str, result: dict) -> None:
    """Keep the tail of a failed stage's captured output for the repair agent."""
    tail = (result or {}).get("output") or ""
    if tail.strip():
        diagnostics[stage] = tail[-DIAG_TAIL_CHARS:]


def _deadline_timeout(deadline_unix: float, default: float) -> float:
    """Clamp one subprocess timeout to the shared absolute deadline."""
    if deadline_unix <= 0:
        return default
    return max(1.0, min(default, deadline_unix - time.time()))


def _cleanup_probe(out_path: str, probe_dir: str) -> None:
    """Remove the probe's output shards and its sitecustomize directory."""
    for path in (out_path, *glob.glob(f"{out_path}.*")):
        with contextlib.suppress(Exception):
            os.unlink(path)
    with contextlib.suppress(Exception):
        shutil.rmtree(probe_dir, ignore_errors=True)


def _read_graph_probe_shards(out_path: str) -> tuple[int, str]:
    """Validate graph-probe shards and return the effective replay count."""
    unranked_replays: list[int] = []
    ranked_processes: dict[
        int,
        list[tuple[int, int | None, int | None, set[int]]],
    ] = {}
    world_sizes: set[int] = set()

    for shard in glob.glob(f"{out_path}.*"):
        try:
            payload = json.loads(Path(shard).read_text().strip())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return -1, f"invalid graph probe shard {Path(shard).name}: {exc}"

        if isinstance(payload, int) and not isinstance(payload, bool):
            if payload < 0:
                return -1, f"invalid negative replay count in {Path(shard).name}"
            unranked_replays.append(payload)
            continue
        if not isinstance(payload, dict):
            return -1, f"invalid graph probe shard payload in {Path(shard).name}"

        try:
            replays = int(payload["replays"])
        except (KeyError, TypeError, ValueError):
            return -1, f"invalid replay count in {Path(shard).name}"
        if replays < 0:
            return -1, f"invalid negative replay count in {Path(shard).name}"

        rank_value = payload.get("rank")
        local_rank_value = payload.get("local_rank")
        world_size_value = payload.get("world_size")
        if rank_value is None:
            if local_rank_value is not None:
                return -1, f"incomplete rank identity in {Path(shard).name}"
            # Launcher/helper processes are not workers even if they inherited
            # a WORLD_SIZE value from their environment.
            unranked_replays.append(replays)
            continue
        if "local_rank" in payload and local_rank_value is None:
            # A self-launcher can inherit RANK/WORLD_SIZE from its caller. Only
            # torchrun workers receive LOCAL_RANK, so this shard is not rank 0.
            unranked_replays.append(replays)
            continue
        if world_size_value is None:
            return -1, f"incomplete rank identity in {Path(shard).name}"

        try:
            rank = int(rank_value)
            local_rank = int(local_rank_value) if local_rank_value is not None else None
            world_size = int(world_size_value)
        except (TypeError, ValueError):
            return -1, f"invalid rank identity in {Path(shard).name}"
        if (
            world_size <= 0
            or rank < 0
            or rank >= world_size
            or (local_rank is not None and (local_rank < 0 or local_rank >= world_size))
        ):
            return -1, f"invalid rank identity in {Path(shard).name}"
        pid_value = payload.get("pid")
        ppid_value = payload.get("ppid")
        try:
            pid = int(pid_value) if pid_value is not None else None
            ppid = int(ppid_value) if ppid_value is not None else None
        except (TypeError, ValueError):
            return -1, f"invalid process identity in {Path(shard).name}"
        if (pid is None) != (ppid is None) or (pid is not None and (pid <= 0 or ppid is None or ppid < 0)):
            return -1, f"invalid process identity in {Path(shard).name}"
        raw_ancestors = payload.get("ancestors") or []
        if not isinstance(raw_ancestors, list):
            return -1, f"invalid process ancestry in {Path(shard).name}"
        try:
            ancestors = {int(ancestor) for ancestor in raw_ancestors}
        except (TypeError, ValueError):
            return -1, f"invalid process ancestry in {Path(shard).name}"
        if any(ancestor <= 0 for ancestor in ancestors):
            return -1, f"invalid process ancestry in {Path(shard).name}"
        ranked_processes.setdefault(rank, []).append((replays, pid, ppid, ancestors))
        world_sizes.add(world_size)

    if not ranked_processes:
        # A normal single-process driver writes one shard. If helper processes
        # also imported sitecustomize, summing their partial counts could let
        # several eager/partial processes collectively satisfy one replay gate.
        return max(unranked_replays, default=0), ""
    if len(world_sizes) != 1:
        return -1, "graph probe rank shards disagree on world_size"

    world_size = next(iter(world_sizes))
    expected_ranks = set(range(world_size))
    actual_ranks = set(ranked_processes)
    if actual_ranks != expected_ranks:
        missing = sorted(expected_ranks - actual_ranks)
        unexpected = sorted(actual_ranks - expected_ranks)
        identity_error = f"incomplete graph probe rank set; missing ranks: {missing}"
        if unexpected:
            identity_error += f"; unexpected ranks: {unexpected}"
        return -1, identity_error

    worker_replays: list[int] = []
    for rank, processes in ranked_processes.items():
        if len(processes) == 1:
            worker_replays.append(processes[0][0])
            continue
        if any(pid is None for _, pid, _, _ in processes):
            return -1, f"ambiguous graph probe shards for rank {rank}"
        roots = [
            replays
            for replays, pid, _ppid, _ancestors in processes
            if all(
                other_pid == pid or pid in other_ancestors or (not other_ancestors and other_ppid == pid)
                for _other_replays, other_pid, other_ppid, other_ancestors in processes
            )
        ]
        if len(roots) != 1:
            return -1, f"ambiguous graph probe process tree for rank {rank}"
        worker_replays.append(roots[0])

    # Unranked launcher shards and ranked helper descendants must not count as
    # workers. Each real rank must independently satisfy the caller's iters gate.
    return min(worker_replays), ""


async def _count_graph_replays(
    driver: str,
    warmup: int,
    iters: int,
    *,
    timeout_sec: float = 300,
) -> tuple[int, str]:
    """Run the driver and return its effective CUDA graph replay count.

    Returns (replay_count, tail). replay_count == -1 signals the probe itself
    failed (timeout / spawn error), distinct from a genuine 0 (eager timing).
    """
    fd, out_path = tempfile.mkstemp(prefix="forge_graph_probe_")
    os.close(fd)
    probe_dir = tempfile.mkdtemp(prefix="forge_graph_probe_site_")
    pathlib.Path(probe_dir, "sitecustomize.py").write_text(_GRAPH_PROBE_SITECUSTOMIZE)
    # PYTHONPATH rather than a wrapper script: torchrun children of a
    # self-launching driver inherit the environment, so each rank imports the
    # counter and reports its own replays.
    env = dict(
        os.environ,
        GRAPH_PROBE_OUT=out_path,
        PYTHONPATH=os.pathsep.join([probe_dir, *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])]),
    )
    cmd = [
        sys.executable,
        driver,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--bench-mode",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        out, err = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=10)
        with contextlib.suppress(Exception):
            from kernelforge.loop.aiter_cache import cleanup_current_owned_aiter_locks

            cleanup_current_owned_aiter_locks()
        _cleanup_probe(out_path, probe_dir)
        return -1, "benchmark timed out"
    except asyncio.CancelledError:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=10)
        with contextlib.suppress(Exception):
            from kernelforge.loop.aiter_cache import cleanup_current_owned_aiter_locks

            cleanup_current_owned_aiter_locks()
        _cleanup_probe(out_path, probe_dir)
        raise
    except Exception as exc:  # noqa: BLE001
        _cleanup_probe(out_path, probe_dir)
        return -1, f"{type(exc).__name__}: {exc}"
    tail = ((out.decode(errors="replace") if out else "") + (err.decode(errors="replace") if err else ""))[-400:]
    if proc.returncode != 0:
        _cleanup_probe(out_path, probe_dir)
        detail = f"benchmark exited {proc.returncode}"
        if tail:
            detail += f": {tail}"
        return -1, detail

    replays, shard_error = _read_graph_probe_shards(out_path)
    _cleanup_probe(out_path, probe_dir)
    if shard_error:
        detail = shard_error
        if tail:
            detail += f": {tail}"
        return -1, detail
    return replays, tail


async def _check_profile_contract(
    driver: str,
    *,
    timeout_sec: float,
) -> tuple[bool, str]:
    """Verify the prepared driver owns a kernel-only profiling path."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        driver,
        PROFILE_RUN_FLAG,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=10)
        return False, "profile-run timed out"
    except asyncio.CancelledError:
        _kill_process_group(proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=10)
        raise
    output = (out.decode(errors="replace") if out else "") + (err.decode(errors="replace") if err else "")
    if proc.returncode != 0:
        return False, f"profile-run exited {proc.returncode}: {output[-200:]}"
    return True, "verified"


async def _preflight_async(
    driver: str,
    snr_threshold: float,
    warmup: int,
    iters: int,
    require_graph: bool = False,
    require_profile: bool = False,
    deadline_unix: float = 0.0,
    expected_case_ids: list[str] | None = None,
) -> PreflightResult:
    reasons: list[str] = []
    details: dict = {}
    diagnostics: dict = {}
    started = time.monotonic()

    def _stage_seconds(since: float) -> float:
        return round(time.monotonic() - since, 3)

    if not driver or not Path(driver).is_file():
        return PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            graph_ok=not require_graph,
            profile_ok=not require_profile,
            reasons=[f"driver file not found: {driver}"],
        )

    # Correctness: the driver must EMIT a parseable metric and not crash. Whether
    # the baseline passes the SNR threshold is a separate (kernel) concern; here
    # we only validate contract conformance.
    correctness_ok = False
    stage_started = time.monotonic()
    try:
        cres = await test_correctness(
            driver_script=driver,
            driver_args=[],
            snr_threshold=snr_threshold,
            timeout_sec=int(_deadline_timeout(deadline_unix, PREFLIGHT_CORRECTNESS_TIMEOUT_S)),
        )
        details["correctness"] = {
            k: cres.get(k)
            for k in (
                "passed",
                "outcome",
                "snr_db",
                "allclose",
                "max_diff",
                "message",
            )
        }
        details["correctness"]["seconds"] = _stage_seconds(stage_started)
        has_metric = (
            cres.get("snr_db") is not None or cres.get("allclose") is not None or cres.get("max_diff") is not None
        )
        if has_metric:
            correctness_ok = True
        else:
            reasons.append(f"correctness mode produced no SNR/allclose metric ({cres.get('message')})")
            _record_stage_output(diagnostics, "correctness", cres)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"correctness run raised {type(exc).__name__}: {exc}")
        diagnostics["correctness"] = "".join(traceback.format_exception(exc))[-DIAG_TAIL_CHARS:]

    # Benchmark: the driver must accept --warmup/--iters/--bench-mode and print
    # per-iteration wall_ms or a single median_ms/mean_ms aggregate.
    bench_ok = False
    stage_started = time.monotonic()
    try:
        bres = await bench_wallclock(
            driver_script=driver,
            driver_args=[],
            warmup_iters=warmup,
            bench_iters=iters,
            timeout_sec=int(_deadline_timeout(deadline_unix, PREFLIGHT_BENCH_TIMEOUT_S)),
        )
        reported_cases = bres.get("case_times") or {}
        details["bench"] = {k: bres.get(k) for k in ("success", "median_ms", "message")}
        details["bench"]["case_count"] = len(reported_cases)
        details["bench"]["seconds"] = _stage_seconds(stage_started)
        # The declared suite is the contract, in both directions. Accepting a
        # non-empty subset is what let a driver be certified against fewer cases
        # than the task declares; accepting extra ones lets it be scored on more,
        # because the baseline takes its case table from what the driver prints,
        # so an undeclared case joins the mean the KEEP/REVERT decision reads.
        declared = set(expected_case_ids or ())
        missing_cases = sorted(declared - set(reported_cases))
        undeclared_cases = sorted(set(reported_cases) - declared) if declared else []
        details["bench"]["expected_case_count"] = len(expected_case_ids or ())
        details["bench"]["missing_cases"] = missing_cases
        details["bench"]["undeclared_cases"] = undeclared_cases
        if not bres.get("success") or not reported_cases:
            reasons.append(
                f"bench mode must produce an aggregate and case_ms timing for every suite case ({bres.get('message')})"
            )
            _record_stage_output(diagnostics, "bench", bres)
        elif missing_cases:
            reasons.append(
                "bench mode reported "
                f"{len(reported_cases)} of the {len(expected_case_ids or ())} cases "
                "this task declares; no case_ms line for "
                f"{', '.join(missing_cases)} — print one line per declared case "
                "using its CASE_ID verbatim"
            )
            _record_stage_output(diagnostics, "bench", bres)
        elif undeclared_cases:
            reasons.append(
                "bench mode reported case_ms for "
                f"{', '.join(undeclared_cases)}, which this task does not declare; "
                "measure the declared suite exactly, because every case printed "
                "here is scored"
            )
            _record_stage_output(diagnostics, "bench", bres)
        else:
            bench_ok = True
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"bench run raised {type(exc).__name__}: {exc}")
        diagnostics["bench"] = "".join(traceback.format_exception(exc))[-DIAG_TAIL_CHARS:]

    # Graph timing: required only for prepass-produced drivers. Detected for real
    # by counting actual torch.cuda.CUDAGraph replays during the benchmark (not by
    # trusting a printed label), so a driver that times eagerly — or whose capture
    # silently fell back to eager — performs < iters replays and is rejected.
    graph_ok = True
    if require_graph:
        graph_ok = False
        if bench_ok:
            stage_started = time.monotonic()
            replays, tail = await _count_graph_replays(
                driver,
                warmup,
                iters,
                timeout_sec=_deadline_timeout(deadline_unix, PREFLIGHT_GRAPH_TIMEOUT_S),
            )
            details["graph"] = {
                "replays": replays,
                "required": iters,
                "seconds": _stage_seconds(stage_started),
            }
            if replays >= iters:
                graph_ok = True
            elif replays < 0:
                diagnostics["graph"] = tail
                reasons.append(f"could not verify graph timing (probe failed): {tail[-160:]}")
            else:
                diagnostics["graph"] = tail
                reasons.append(
                    f"benchmark did not run under a CUDA/HIP graph (observed {replays} "
                    f"graph replays over {iters} timed iterations): {tail[-160:]}"
                )
        else:
            reasons.append("cannot verify graph timing because bench produced no timing")

    profile_ok = True
    if require_profile:
        profile_ok = False
        if bench_ok:
            stage_started = time.monotonic()
            profile_ok, profile_detail = await _check_profile_contract(
                driver,
                timeout_sec=_deadline_timeout(deadline_unix, PREFLIGHT_PROFILE_TIMEOUT_S),
            )
            details["profile"] = {
                "ok": profile_ok,
                "contract": profile_detail if profile_ok else "",
                "seconds": _stage_seconds(stage_started),
            }
            if not profile_ok:
                diagnostics["profile"] = profile_detail
                reasons.append(f"profiling contract failed ({profile_detail})")
        else:
            reasons.append("cannot verify profiling contract because bench produced no timing")

    ok = correctness_ok and bench_ok and graph_ok and profile_ok
    return PreflightResult(
        ok=ok,
        correctness_ok=correctness_ok,
        bench_ok=bench_ok,
        graph_ok=graph_ok,
        profile_ok=profile_ok,
        reasons=reasons,
        details=details,
        diagnostics=diagnostics,
        duration_sec=_stage_seconds(started),
    )


def preflight_task(
    *,
    driver: str,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    warmup: int = PREFLIGHT_WARMUP,
    iters: int = PREFLIGHT_ITERS,
    require_graph: bool = False,
    require_profile: bool = False,
    deadline_unix: float = 0.0,
    expected_case_ids: list[str] | None = None,
) -> PreflightResult:
    """Synchronous deterministic check of a driver against the loop's contract.

    Set ``require_graph`` to also require the benchmark to run under a CUDA/HIP
    graph (used for prepass-produced drivers; the CLI's initial gate leaves it off
    so a conforming caller-provided driver is never rejected on this basis).

    ``expected_case_ids`` is the suite the task declares (see
    ``declared_case_ids``); the driver must report a ``case_ms`` line for each.
    """

    return asyncio.run(
        _preflight_async(
            driver,
            snr_threshold,
            warmup,
            iters,
            require_graph,
            require_profile,
            deadline_unix,
            expected_case_ids,
        )
    )


# ---------------------------------------------------------------------------
# Prepare — bounded LLM repair loop with snapshot/rollback
# ---------------------------------------------------------------------------


class ScaffoldRetirementError(RuntimeError):
    """The authoring-only reference bundle survived the retirement before a verdict.

    Preparation cannot continue: the state the driver would be judged in is no
    longer the state it will be committed and re-run in, which is the whole
    invariant the retirement exists to hold.
    """


@dataclass
class PrepareResult:
    """Outcome of the preparation step (explicit success/failure contract)."""

    ok: bool
    attempts: int = 0
    wrote_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    rolled_back: bool = False
    final_preflight: PreflightResult | None = None
    message: str = ""
    audit_dir: str = ""


def _snapshot(paths: list[Path]) -> dict[Path, bytes | None]:
    """Record current bytes (or None if absent) for each path, for rollback."""
    snap: dict[Path, bytes | None] = {}
    for p in paths:
        try:
            snap[p] = p.read_bytes() if p.is_file() else None
        except Exception:
            snap[p] = None
    return snap


def _restore(snapshot: dict[Path, bytes | None]) -> None:
    """Restore snapshotted paths: rewrite originals, delete ones that were absent."""
    for p, original in snapshot.items():
        try:
            if original is None:
                if p.is_file():
                    p.unlink()
            else:
                p.write_bytes(original)
        except Exception:
            continue


def _abs(workspace: Path, path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (workspace / p)


def _git(workspace: Path, *args: str) -> tuple[int, str]:
    """Run one git command, returning ``(exit code, stdout+stderr)``.

    A git that could not be launched at all reports 128, git's own code for a
    fatal error, so a caller reading the exit code cannot mistake it for one of
    git's per-path answers (``ls-files --error-unmatch`` exits 1 for "not in the
    index", which is a very different fact from "the query never ran").
    """
    try:
        r = git(*args, cwd=workspace, check=False)
    except OSError as exc:
        return 128, str(exc)
    return r.returncode, (r.stdout + r.stderr)


def _git_head(workspace: Path) -> str:
    code, out = _git(workspace, "rev-parse", "HEAD")
    return out.strip() if code == 0 else ""


def _git_diff_patch(workspace: Path, base_sha: str) -> str:
    """Capture the working tree's uncommitted tracked modifications vs HEAD.

    Returned as a git patch (binary-safe) so a failure rollback can restore the
    caller's pre-prep uncommitted changes instead of blanket-resetting to HEAD.
    """
    if not base_sha:
        return ""
    return git("diff", "--binary", "HEAD", cwd=workspace).stdout


def _git_apply_patch(workspace: Path, patch: str) -> None:
    """Re-apply a patch captured by ``_git_diff_patch`` (no-op for an empty patch).

    A rollback that cannot put the caller's own uncommitted work back has left
    the workspace in a state nobody declared, so it says so rather than
    reporting a clean rollback over a dirty tree.
    """
    if not patch.strip():
        return
    git("apply", "--whitespace=nowarn", cwd=workspace, input=patch)


def _git_untracked(workspace: Path) -> set[str]:
    """Set of untracked (and not-ignored) paths, relative to the workspace."""
    code, out = _git(workspace, "ls-files", "--others", "--exclude-standard")
    if code != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _git_indexed(workspace: Path, path: Path) -> bool | None:
    """Whether ``path`` is in the workspace's index, i.e. will be committed.

    ``None`` when the question could not be answered — the path does not sit under
    the workspace, or git itself failed. Collapsing that into ``False`` sent the
    caller's failure message on to blame the workspace's ignore rules, which is
    the wrong thing to look at when nothing ever checked them.
    """
    try:
        relative = path.resolve().relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    code, out = _git(workspace, "ls-files", "--cached", "--error-unmatch", "--", relative)
    if code == 0:
        return True
    # ``--error-unmatch`` exits 1 for a path the index does not hold; anything else
    # is git failing, not git answering.
    return False if code == 1 else None


def _git_changed_since(workspace: Path, base_sha: str) -> list[str]:
    if not base_sha:
        return []
    code, out = _git(workspace, "diff", "--name-only", base_sha, "HEAD")
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _remove_new_untracked(workspace: Path, pre_untracked: set[str]) -> None:
    """Delete untracked files that appeared during prep (rollback of new files)."""
    for rel in _git_untracked(workspace) - pre_untracked:
        with contextlib.suppress(Exception):
            (workspace / rel).unlink()


def _safe_rmtree(path: Path | None) -> None:
    """Remove a tree best-effort; the caller checks whether it actually went."""
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


def _safe_unlink(path: Path) -> None:
    with contextlib.suppress(Exception):
        if path.is_file():
            path.unlink()


def _find_reference_harness(ref_dir: Path | None) -> str | None:
    """Return the text of a capture-guarded graph harness from the reference tree.

    Used to pre-place a known-good ``graph_harness.py`` in the workspace so the
    agent imports a correct ``cuda_graph_bench`` (one that accepts ``dirty``/
    ``verify``) instead of writing its own — a self-written harness can silently
    mismatch its own driver calls and degrade graph timing to eager.
    """
    if ref_dir is None or not ref_dir.is_dir():
        return None
    for cand in sorted(ref_dir.rglob("graph_harness.py")):
        try:
            text = cand.read_text()
        except Exception:
            continue
        if "def cuda_graph_bench" in text and "dirty" in text:
            return text
    return None


def _materialize_reference(workspace: Path) -> Path | None:
    """Make the shipped reference examples available for the agent to Read.

    Copies the packaged/source ``examples`` tree into ``workspace/REFERENCE_SUBDIR``
    so the agent reads REAL, complete reference tasks (driver.py, graph_harness.py,
    README contract) within its cwd — not truncated prompt text. Falls back to
    writing a compact contract + driver template when the examples tree cannot be
    resolved (e.g. a misconfigured install). Returns the reference dir, or None.
    """
    ref_dir = workspace / REFERENCE_SUBDIR
    _safe_rmtree(ref_dir)

    examples = resource_path("examples", missing_ok=True)
    try:
        if examples and Path(examples).is_dir():
            shutil.copytree(examples, ref_dir, ignore=_REFERENCE_IGNORE)
            return ref_dir
    except Exception:
        _safe_rmtree(ref_dir)

    # Fallback: no examples tree resolved — materialize the compact contract and a
    # driver template so the agent still has real files to Read.
    try:
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "CONTRACT.md").write_text(DRIVER_CONTRACT_SPEC)
        (ref_dir / "driver_template.py").write_text(REFERENCE_DRIVER_TEMPLATE.lstrip("\n"))
        return ref_dir
    except Exception:
        _safe_rmtree(ref_dir)
        return None


def _reference_note(ref_dir: Path | None, workspace: Path) -> str:
    """Prompt block that points the agent at the on-disk reference files to Read."""
    if ref_dir is None or not ref_dir.is_dir():
        return "No reference files were available; follow the contract above."
    rel_root = os.path.relpath(ref_dir, workspace)
    lines = [
        "## Reference files to Read (real, complete — do NOT rely on memory)",
        f"A copy of KernelForge's shipped reference material is in `./{rel_root}/`.",
        # A driver authored against this directory passed validation and then
        # crashed on the loop's first baseline bench, because the directory is
        # deleted between the two. Say so where the agent reads the path.
        f"`./{rel_root}/` is TEMPORARY authoring scaffolding: it is DELETED before "
        "your driver is validated and committed, so read it now but never read it "
        "at runtime — do not open, import, or glob anything under it from the "
        "driver or its helpers.",
        "Read the contract and ONE reference driver before writing; the others "
        "are there if the first does not match your case:",
    ]
    readme = ref_dir / "README.md"
    if readme.is_file():
        lines.append(f"- `./{rel_root}/README.md` — the full driver contract (files + rules)")
    contract = ref_dir / "CONTRACT.md"
    if contract.is_file():
        lines.append(f"- `./{rel_root}/CONTRACT.md` — the driver contract")
    # List each example task's key files (driver + any harness).
    for sub in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        drv = sub / "driver.py"
        if drv.is_file():
            rel = os.path.relpath(sub, workspace)
            extras = [f.name for f in sub.iterdir() if f.name in ("graph_harness.py", "program.md")]
            extra = f" (+ {', '.join(sorted(extras))})" if extras else ""
            lines.append(f"- `./{rel}/driver.py`{extra} — a complete working reference driver")
    tmpl = ref_dir / "driver_template.py"
    if tmpl.is_file():
        lines.append(f"- `./{rel_root}/driver_template.py` — a minimal driver skeleton to adapt")
    return "\n".join(lines)


def _materialize_invocation_spec(
    source_file: str,
    durable_dir: Path | None,
) -> tuple[Path | None, str]:
    """Place the invocation spec where the prepared driver can keep reading it.

    ``durable_dir`` is the driver's own directory, not the temporary reference
    bundle: the spec carries the declared case table, so a driver that derives
    its cases from the task (as the contract demands) legitimately reads it at
    runtime, and it therefore has to survive preparation and be committed with
    the driver.

    Returns the destination and the authoritative text at that destination. An
    equivalent payload already sitting there is left byte-identical — an external
    driver bundle ships its own spec next to the driver, and the artifact
    transaction guards that file as a read-only caller input, so a canonical
    rewrite of the same data would abort the publish.

    Anything else already occupying that name belongs to the caller and is left
    alone: the directory is the caller's, and preparation's rollback restores the
    driver and the Git-tracked state, so an untracked file replaced here could
    not be recovered. A symlink is refused for the same reason one step further
    out — writing through it would edit a file outside the directory this
    function was handed. Both cases return no destination, which the caller
    already reports as preparation continuing without the spec.
    """
    if not source_file or durable_dir is None:
        return None, ""
    try:
        source = Path(source_file).expanduser().resolve()
        if not source.is_file() or source.stat().st_size > MAX_INVOCATION_SPEC_BYTES:
            return None, ""
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None, ""
        destination_name = source.name if _INVOCATION_SPEC_NAME_RE.fullmatch(source.name) else INVOCATION_SPEC_FILENAME
        destination = durable_dir / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            log.warning(
                "invocation specification destination %s is a symbolic link; "
                "leaving it alone rather than writing through it",
                destination,
            )
            return None, ""
        if destination.exists():
            with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
                existing = destination.read_text(encoding="utf-8")
                if json.loads(existing) == payload:
                    return destination, existing
            log.warning(
                "invocation specification destination %s already holds different "
                "content; leaving it alone rather than replacing it",
                destination,
            )
            return None, ""
        canonical = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        destination.write_text(canonical, encoding="utf-8")
        return destination, canonical
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None, ""


def declared_case_ids(spec_path: Path | str | None) -> list[str]:
    """Case ids the task declares its driver must benchmark, sorted.

    ``tests.driver_contract.case_selectors`` is the task's own statement of the
    suite, and the prep prompt hands those ids to the agent verbatim. Preflight
    checks the driver's ``case_ms`` lines against this list so "conforms" means
    "measures the declared task", not merely "printed at least one case".

    An empty list means "this task declares no suite", which disables the gate,
    and only a spec that says so may produce one. A spec that was supplied and
    cannot be read is an error instead: returning empty for it switches the gate
    off, so the run optimizes and scores a case set nobody verified and says so
    in one log line among thousands of others. The operator named the file.

    Raises:
        ValueError: If ``spec_path`` names a file that cannot be read or does not
            hold a JSON object.
    """
    if not spec_path:
        return []
    try:
        payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read the invocation specification {spec_path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invocation specification {spec_path} is not a JSON object")
    tests = payload.get("tests")
    contract = tests.get("driver_contract") if isinstance(tests, dict) else None
    selectors = contract.get("case_selectors") if isinstance(contract, dict) else None
    if not isinstance(selectors, list):
        return []
    return sorted(
        {str(selector["CASE_ID"]) for selector in selectors if isinstance(selector, dict) and selector.get("CASE_ID")}
    )


class _SpecInline(NamedTuple):
    """The spec's text, or a statement of why it is not below.

    Three different things stop a spec from being inlined -- it could not be
    read, it is empty, it is too big -- and they call for three different next
    moves. Collapsing them into one empty string means the note has to guess,
    and the guess is wrong twice out of three times. This is the same defect
    the module's own quick reference had, one level up: a renderer that cannot
    distinguish absent from empty will state one when it means the other.
    """

    text: str
    #: Completes "The specification at `./<path>` ...". Empty when ``text`` is.
    refusal: str
    #: What the agent should do instead. Empty when ``text`` is usable.
    recourse: str


_SPEC_RECOVER_FROM_SOURCE = (
    "Recover the public callable, the operand shapes and dtypes, and the "
    "deployment context from the kernel source and the tests it names; do not "
    "invent them and do not fall back to round numbers of your own choosing."
)


def _invocation_spec_text(spec_path: Path) -> _SpecInline:
    """Return the spec verbatim, or say precisely why it is not inlined.

    Handed to the agent whole rather than summarised. Every selective rendering
    has to decide what an absent field looks like, and both ways of deciding are
    wrong: a heading over nothing claims the field is known and empty, while
    dropping the heading leaves no trace that the field exists at all. In the
    raw JSON an absent key is unambiguously absent, and the agent is reading the
    same bytes the driver will read at runtime.

    Content is NOT validated as JSON, deliberately. A corrupt document is what
    the driver will hit at runtime, and showing the agent the corruption beats
    replacing it with an empty note that says nothing happened.
    """
    try:
        text = spec_path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError) as error:
        return _SpecInline(
            "",
            f"could not be read ({type(error).__name__})",
            "Try `Read` on it yourself; if that fails too, "
            + _SPEC_RECOVER_FROM_SOURCE[0].lower()
            + _SPEC_RECOVER_FROM_SOURCE[1:],
        )
    if not text:
        return _SpecInline(
            "",
            "is empty, so it declares no invocation evidence at all",
            _SPEC_RECOVER_FROM_SOURCE,
        )
    if len(text.encode("utf-8")) > _SPEC_INLINE_MAX_BYTES:
        # Nothing observed comes close -- the document is bounded by the operand
        # count -- but a prompt is the wrong place to find out that some
        # producer emitted a megabyte.
        return _SpecInline(
            "",
            f"is larger than the {_SPEC_INLINE_MAX_BYTES // 1024} KB inline "
            "limit, so it is referenced rather than quoted",
            "Use `Read` on it before you touch the driver.",
        )
    return _SpecInline(text, "", "")


def _invocation_spec_note(spec_path: Path | None, workspace: Path) -> str:
    """Build the highest-priority prompt instruction for invocation evidence."""
    if spec_path is None:
        return ""
    rel_path = os.path.relpath(spec_path, workspace)
    inline = _invocation_spec_text(spec_path)
    if inline.text:
        spec_block = f"\n### The specification, verbatim\n```json\n{inline.text}\n```\n"
    else:
        spec_block = f"\nThe specification at `./{rel_path}` {inline.refusal}. {inline.recourse}\n"
    return f"""\
## Invocation specification — BUILD THE DRIVER FROM THIS
The JSON below is the authoritative evidence for this task. Build the driver
from it: the public callable that executes the operator, the ordered input and
output shapes and dtypes, the deployment batch and sequence-length context, the
editable source or device symbol (which may differ from the public callable),
and the relevant tests, benchmarks and runtime paths.

Benchmark the shapes and dtypes this operator actually runs at in the deployment
the spec describes. A toy size is not a smaller version of the real measurement,
it is a different one: a kernel tuned at a sequence length the workload never
serves can report a large speedup that disappears end to end. Where the spec
states the operand dims, use exactly those. Where it does not, recover them from
the kernel source and the tests it names, and size them from the deployment
context it carries -- do not fall back to round numbers of your own choosing.

`./{rel_path}` is DURABLE: it sits beside the driver, is committed with it, and
is the ONLY task input the driver may read at runtime. If your driver loads its
case table from this file, resolve the path relative to the driver's own
directory and nowhere else. Every other path handed to you below is authoring
scaffolding that will not exist when the driver runs.

The specification is read-only evidence. Do NOT edit it. Unknown or omitted
fields must be resolved from the referenced source/tests; do not invent
signatures, tensor shapes, dtypes, or correctness rules.
{spec_block}"""


def _kill_process_group(proc) -> None:
    """Kill the child and ALL its descendants (no orphans).

    The child is spawned with ``start_new_session=True``, so its pid IS its
    process-group id at creation. Signal *that* pgid directly. We deliberately do
    NOT consult ``os.getpgid(pid)`` first: once the leader exits and its pid is
    recycled, getpgid can resolve the reused pid to an *unrelated* live process's
    group and we'd SIGKILL innocents. The original pgid (== pid) is the only id we
    can trust, and it still reaps ninja/clang compile children that keep the group
    alive after the python driver leader has died (that leak left a cold CK
    compile burning a core for >26 min after a preflight timeout).
    """
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    signalled = False
    if hasattr(os, "killpg"):
        # pid == pgid under start_new_session; survives the leader's death.
        try:
            os.killpg(pid, signal.SIGKILL)
            signalled = True
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:  # noqa: BLE001 - group may already be gone
            pass
    if not signalled:
        with contextlib.suppress(Exception):
            proc.kill()


def _ensure_agent_git_workspace(workspace: Path) -> None:
    """Create a private baseline commit when a backend requires a git cwd."""
    code, _ = _git(workspace, "rev-parse", "--show-toplevel")
    if code == 0:
        return
    code, output = _git(workspace, "init")
    if code != 0:
        raise RuntimeError(f"could not initialize preparation workspace: {output}")
    code, output = _git(workspace, "add", "-A")
    if code != 0:
        raise RuntimeError(f"could not stage preparation workspace: {output}")
    code, output = _git(
        workspace,
        "-c",
        "user.name=KernelForge",
        "-c",
        "user.email=kernel-forge@localhost",
        "commit",
        "--allow-empty",
        "-m",
        "forge task preparation baseline",
    )
    if code != 0:
        raise RuntimeError(f"could not commit preparation workspace: {output}")


async def _run_prepare_agent(
    *,
    config: Config,
    workspace: Path,
    system_prompt: str,
    prompt: str,
    timeout_sec: float,
    additional_dirs: list[str] | None = None,
    allow_shell: bool = True,
    target_files: list[str] | None = None,
    protected_files: list[str] | None = None,
    usage=None,
    progress_log: list[str] | None = None,
) -> str:
    """Run one sandboxed driver-authoring turn through the selected backend."""
    runtime = with_writable_sandbox(config.agent_runtime())
    backend = create_registered_backend(runtime)
    if backend.capabilities.requires_workspace_cwd:
        _ensure_agent_git_workspace(workspace)
    protected_globs = list(dict.fromkeys(Path(path).name for path in (protected_files or []) if path))
    spec = AgentRunSpec(
        system_prompt=system_prompt,
        user_prompt=prompt,
        cwd=str(workspace),
        writable=True,
        timeout_sec=max(1, int(timeout_sec)),
        additional_directories=[directory for directory in (additional_dirs or []) if directory],
        target_files=list(target_files or []),
        # Deliberately no driver_script. That field declares the measurement
        # driver whose content a turn must preserve, which is the opposite of
        # this turn's job: preparation exists to author that file. Declaring it
        # snapshots the driver as protected, so the agent's rewrite is reported
        # as a protected file changed and rolled back. The driver is a target
        # here, and target_files already carries it.
        protected_globs=protected_globs,
        allow_dirty_targets=True,
        allow_untracked=True,
        # Preparation authors its own scaffolding before the agent starts: the
        # reference bundle's harness/config files and the durable invocation
        # spec, all of which match protected_globs and none of which are
        # targets. A provider guard that judges the worktree against HEAD reads
        # them as protected files the turn created, rejects it, and rolls the
        # driver back -- so the agent's edit is undone and every retry fails the
        # same way. Judge deviations from the state the turn inherited instead.
        allow_dirty_baseline=True,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=allow_shell,
            max_turns=50,
            permission_mode=os.environ.get(
                "FORGE_PERMISSION_MODE",
                "acceptEdits",
            ),
            bare=False,
        ),
        progress_log=progress_log,
    )
    result = await asyncio.wait_for(
        backend.run(spec, usage=usage),
        timeout=watchdog_timeout_sec(timeout_sec),
    )
    return result.text.strip()


def _read_limited(path: Path, limit: int = 16000) -> str:
    try:
        return path.read_text(errors="replace")[:limit]
    except Exception:
        return ""


_COMPILE_ONLY_RE = re.compile(
    r"""print\s*\(\s*["']compile_only:\s*True["']\s*\)""",
)


def _is_compile_only_driver(text: str) -> bool:
    """True when the driver text is a compile-only autogen stub."""
    return _COMPILE_ONLY_RE.search(text) is not None


def _build_evidence(
    *,
    workspace: Path,
    kernel: str,
    driver: str,
    program_md: str,
    target_functions: list[str],
    source_files: list[str],
    preflight: PreflightResult | None,
) -> str:
    kernel_path = _abs(workspace, kernel)
    parts = [
        "## Task metadata",
        f"- workspace: `{workspace}`",
        f"- kernel (PROTECTED — DO NOT EDIT): `{kernel}`",
        f"- driver to create/fix (write here): `{driver}`",
        f"- target_functions: {', '.join(target_functions) or '(none given)'}",
    ]
    if source_files:
        parts.append("- other PROTECTED source files (DO NOT EDIT):")
        parts += [f"  - `{s}`" for s in source_files if s and _abs(workspace, s) != kernel_path]
    if program_md.strip():
        parts += ["", "## program.md (task guidance)", "```", program_md[:4000], "```"]
    parts += [
        "",
        "## Kernel source (the operator to measure — read its public entry point)",
        f"### `{kernel}`",
        "```python",
        _read_limited(kernel_path, 16000),
        "```",
    ]
    dpath = _abs(workspace, driver)
    if dpath.is_file():
        driver_text = _read_limited(dpath, 12000)
        if _is_compile_only_driver(driver_text):
            parts += [
                "",
                "## Current driver is a COMPILE-ONLY STUB — rewrite it completely",
                "The driver below only verifies that the kernel compiles with hipcc. "
                "It has NO runtime measurement, NO correctness check, and NO timing "
                "output. You MUST write a complete measurement driver from scratch — "
                "do not adapt the compile-only boilerplate.",
                f"### `{driver}`",
                "```python",
                driver_text,
                "```",
            ]
        else:
            parts += ["", "## Current (non-conforming) driver", f"### `{driver}`", "```python", driver_text, "```"]
    if preflight is not None and preflight.reasons:
        parts += ["", "## Why the current task fails the forge-loop contract", *[f"- {r}" for r in preflight.reasons]]
        for stage, tail in (preflight.diagnostics or {}).items():
            if tail and tail.strip():
                parts += ["", f"### What the {stage} stage actually printed (tail)", "```", tail.strip(), "```"]
    return "\n".join(parts)


def summarize_agent_progress(progress_log: list[str]) -> str:
    """Compact "what did it actually do" line from a streamed progress sink."""
    if not progress_log:
        return "no tool activity was recorded"
    non_streaming = any(entry.startswith("progress: not supported") for entry in progress_log)
    counts: dict[str, int] = {}
    for entry in progress_log:
        if entry.startswith("tool: "):
            name = entry[6:].split(" ", 1)[0]
            counts[name] = counts.get(name, 0) + 1
    parts = []
    if counts:
        parts.append(
            "tool calls: "
            + ", ".join(f"{name}x{count}" for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        )
    elif non_streaming:
        parts.append("backend does not support progress streaming")
    else:
        parts.append("no tool calls at all")
    tail = [entry for entry in progress_log[-6:]]
    if tail:
        parts.append("last steps:\n" + "\n".join(f"  {entry}" for entry in tail))
    return "; ".join(parts[:1]) + ("\n" + parts[1] if len(parts) > 1 else "")


RETRY_HEADING_DEFAULT = "Your previous attempt still did NOT pass the deterministic check"
RETRY_HEADING_NO_EDIT = "Your previous attempt did not change the driver at all"


def _distributed_contract_note(nproc: int) -> str:
    """What a driver must do when the task runs on more than one rank.

    The loop passes ``--nproc-per-node`` to its profiler, which then expects one
    artifact set per rank. Nothing else launches those ranks: a driver that runs
    single-process leaves the profiler looking for ranks that never existed, and
    any timing it does produce describes a collective that never happened.

    Only the driver knows how to build its kernel's context (an IPC handle, a
    registered buffer, a communicator), so that setup belongs here rather than
    in a generic template.
    """
    if nproc <= 1:
        return ""
    return f"""
## This task runs on {nproc} GPUs — the driver must launch them

The kernel is a collective: it only computes the right answer when {nproc} ranks
participate. Your driver owns the launch. One file, two roles:

* No `RANK` in the environment: re-exec this same file under
  `torch.distributed.run --standalone --nproc-per-node={nproc}` and forward the
  exit code. Do NOT use `start_new_session`; the caller kills the whole process
  group on timeout and a detached torchrun would survive holding its GPUs.
* `RANK` present: bind `LOCAL_RANK` with `torch.cuda.set_device`, call
  `dist.init_process_group`, and run the measurement as a worker.

Requirements specific to a collective:

* Build whatever context the kernel needs before calling it. A compiled
  collective usually takes an opaque handle (IPC buffer, registered workspace,
  communicator) that must be created and exchanged across ranks first. Read the
  kernel's own Python binding to see what it expects.
* Check correctness against the matching `torch.distributed` collective — it is
  the only reference that is itself distributed.
* Reduce every metric across ranks before printing: take the SLOWEST rank's
  time (a collective is as fast as its laggard) and the WORST rank's SNR (one
  wrong rank is a wrong collective). Print only from rank 0.
* Destroy the process group before exiting, or the next stage inherits a wedged
  communicator.
"""


def _build_prompt(
    evidence: str,
    driver_rel: str,
    reference_note: str,
    prior_failure: str = "",
    invocation_note: str = "",
    prior_failure_heading: str = RETRY_HEADING_DEFAULT,
    distributed_note: str = "",
) -> str:
    retry = ""
    if prior_failure:
        retry = (
            f"\n## {prior_failure_heading}\n"
            f"{prior_failure}\n"
            "Fix the driver so all preflight checks pass (correctness, benchmark, "
            "graph timing, and profiling contract). Do not stop until "
            f"`python {driver_rel}` prints a correctness metric AND timing.\n"
        )
    return f"""\
Prepare this KernelForge task so forge-loop can optimize it. Author (or repair)
the measurement driver at `{driver_rel}` so it satisfies the driver contract
below. A deterministic validator will execute the driver after each attempt and
return any failures for the next attempt, so get a complete draft on disk early
and let the validator tell you what is wrong — do not try to read your way to a
perfect first version.

{invocation_note}
{distributed_note}
Rules:
- The ONLY off-limits files are the kernel and the listed source files — NEVER
  edit them (they are what forge-loop optimizes). You MAY create or modify any
  OTHER file you need (the driver, small helper modules, etc.).
- Do not create symlinks or modify task metadata, invocation specifications, or
  files outside the driver staging directory.
- The driver must still run after this preparation ends. Only the driver, the
  helpers you write beside it, and the invocation specification beside it are
  durable; the reference bundle is deleted before your driver is validated. Read
  it now, never at runtime.
- Graph timing is REQUIRED: the benchmark MUST capture the op into a CUDA/HIP
  graph and REPLAY it once per timed iteration. The deterministic check counts
  actual `torch.cuda.CUDAGraph` replays, so a printed label does NOT count and
  eager timing is REJECTED. The simplest way to pass is to bench through the
  provided `graph_harness.py` (see the note below); make capture work — allocate
  inputs once, reuse one output buffer, launch on the current stream, and pass
  `dirty`/`verify` — rather than settling for eager timing.
- The `verify` callback confirms graph replay actually ran the kernel; it is NOT
  a full correctness check. Use `_snr_db(ref, out) > 30.0` (SNR-based), NOT
  `torch.allclose` — for FP8/quantized kernels, allclose can fail after graph
  replay even though the kernel computed correctly, causing a false fallback to
  eager timing.
- Keep the driver deterministic (fixed seed) and exit 0 on success.

{reference_note}

{evidence}

{DRIVER_CONTRACT_SPEC}
{retry}
When done, ensure `python {driver_rel}` runs the complete correctness suite and
prints an `SNR:`/`allclose:` line,
`python {driver_rel} --bench-mode --warmup 3 --iters 10` runs the complete
benchmark suite and prints `wall_ms:`/`median_ms:` plus `case_ms:` lines, and
`python {driver_rel} --profile-run` runs only the driver-selected profile case's
target kernel; all commands must exit 0.
"""


_SYSTEM_PROMPT = """\
You are a measurement-harness engineer for GPU kernel optimization. Your only job
is to make a task benchmarkable by forge-loop by writing a correct measurement
driver (plus any helper files it needs). You NEVER modify the kernel or source
files under optimization — only the measurement scaffolding. Time on the GPU
(graph timing strongly preferred); the caller runs deterministic verification
after every edit attempt. The finished driver MUST also expose the complete
profiling contract: benchmark mode prints `case_ms: <case_id> <ms>` for every
case the task declares, and `--profile-run` lets the driver select one
representative case and run only its target kernel without a reference
implementation or timing output.

The driver you leave behind is committed and then re-run unchanged for hours. It
may only depend at runtime on itself, the helpers you write beside it, and the
invocation specification beside it. Reference material you are given for
authoring is deleted before validation; a driver that reads it at runtime passes
the check and then crashes on the first real measurement.

Working order — you are on a wall clock, and the caller keeps whatever is on
disk when it expires:
1. Read the invocation spec and ONE reference driver. That is enough to start.
2. WRITE a complete first draft of the driver. Complete means both modes, every
   required output line, runnable end to end — not a sketch, not a TODO.
3. Only then run it, read more, and iterate on what actually failed.
An attempt that ends with the driver file unchanged is a total loss: nothing is
salvaged and the next attempt restarts from the same broken state, while an
attempt that merely ran out of time still gets validated and can be accepted.
Never spend a whole attempt reading. If time is running short, save the best
driver you have rather than nothing.
"""


async def prepare_task(
    *,
    config: Config,
    workspace_dir: str,
    kernel: str,
    driver: str,
    program_md: str,
    target_functions: list[str],
    source_files: list[str],
    kernel_backend: str = "",
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    preflight: PreflightResult | None = None,
    deadline_sec: float = PREPARE_MAX_WALL_SEC,
    deadline_unix: float = 0.0,
    invocation_spec_file: str = "",
    expected_case_ids: list[str] | None = None,
    read_only_files: list[str] | None = None,
    nproc_per_node: int = 1,
    usage=None,
) -> PrepareResult:
    """Author/repair the driver so the task conforms; roll back on failure.

    ``expected_case_ids`` is the suite the task declares, passed in rather than
    re-derived here (see :func:`declared_case_ids`). The caller already gates its
    own driver on that list, and deriving it a second time from the materialized
    copy made the two agree only while materialization kept succeeding.
    """

    if deadline_unix > 0:
        deadline_sec = min(
            deadline_sec,
            max(0.0, deadline_unix - time.time()),
        )
    workspace = Path(workspace_dir).resolve()
    driver_input_path = Path(os.path.abspath(os.path.expanduser(str(_abs(workspace, driver)))))
    driver_path = driver_input_path.resolve(strict=False)
    try:
        driver_input_path.relative_to(workspace)
        lexical_driver_internal = True
    except ValueError:
        lexical_driver_internal = False
    try:
        driver_path.relative_to(workspace)
        resolved_driver_internal = True
    except ValueError:
        resolved_driver_internal = False
    driver_external = not (lexical_driver_internal and resolved_driver_internal)

    experiments_dir: Path | None = None
    audit_dir: Path | None = None
    try:
        experiments_dir = Path(config.experiments_dir)
        audit_dir = experiments_dir / "task_preparation"
        audit_dir.mkdir(parents=True, exist_ok=True)
    except (AttributeError, OSError, TypeError, ValueError):
        audit_dir = None
    audit_dir_str = str(audit_dir) if audit_dir is not None else ""

    external_transaction: ExternalArtifactTransaction | None = None
    agent_workspace = workspace
    if driver_external:
        external_exclusions = [workspace]
        if experiments_dir is not None:
            try:
                experiments_rel = experiments_dir.resolve().relative_to(driver_path.parent)
                if experiments_rel.parts:
                    # Runtime logs/results can change while the agent is running.
                    # Exclude their complete top-level subtree from the staged
                    # driver/helper transaction.
                    external_exclusions.append(driver_path.parent / experiments_rel.parts[0])
                elif audit_dir is not None:
                    external_exclusions.append(audit_dir)
            except (OSError, ValueError):
                # Keep the conservative workspace exclusion when the external
                # experiments path cannot be relativized safely.
                pass
        protected_external_inputs = [Path(path) for path in [*(read_only_files or []), invocation_spec_file] if path]
        try:
            external_transaction = ExternalArtifactTransaction(
                driver_path=driver_input_path,
                excluded_paths=external_exclusions,
                passthrough_paths=[workspace],
                read_only_paths=protected_external_inputs,
            )
        except ExternalArtifactError as exc:
            return PrepareResult(
                ok=False,
                attempts=0,
                rolled_back=True,
                message=f"could not stage external driver artifacts: {exc}",
                audit_dir=audit_dir_str,
            )
        driver_path = external_transaction.staged_driver_path
        agent_workspace = external_transaction.stage_root

    driver_access_dir = driver_path.parent.resolve()
    harness_path = driver_access_dir / "graph_harness.py" if driver_external else workspace / "graph_harness.py"

    def _audit_text(relative: str, text: str) -> None:
        if audit_dir is None:
            return
        try:
            path = audit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            # Audit artifacts are best-effort and never affect driver validity.
            pass

    def _audit_json(relative: str, payload: dict) -> None:
        try:
            _audit_text(
                relative,
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            )
        except (TypeError, ValueError):
            # Non-serializable audit metadata must not block task preparation.
            pass

    def _driver_digest() -> str:
        try:
            return hashlib.sha256(driver_path.read_bytes()).hexdigest()
        except OSError:
            return ""

    def _detect_driver_edited(digest_before: str) -> bool:
        return _driver_digest() != digest_before

    def _audit_driver(relative: str) -> None:
        if audit_dir is None or not driver_path.is_file():
            return
        try:
            path = audit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(driver_path, path)
            # copy2 carries the SOURCE mtime across, so every snapshot in the
            # audit trail claimed the driver's own mtime rather than when it was
            # captured — reconstructing a timeline from this directory (the
            # obvious thing to do when a prep fails) then yields wildly wrong
            # durations. Stamp the capture time instead.
            os.utime(path, None)
        except OSError:
            # A missing audit copy is non-fatal; the staged driver remains the
            # authoritative preparation artifact.
            pass

    if preflight is not None:
        _audit_json("initial_preflight.json", asdict(preflight))

    # (1) Protect ONLY the source under optimization: kernel + declared source
    # files. Everything else (driver, helpers, harness) is the agent's to author.
    protected = {_abs(workspace, kernel)}
    for s in source_files:
        if s:
            protected.add(_abs(workspace, s))

    # Rollback anchors: a byte-snapshot of the protected source (guaranteed
    # restore even if untracked) plus the git state, so ANY other file the agent
    # creates/modifies can be undone on failure without knowing it in advance.
    src_snapshot = _snapshot(list(protected))
    prep_base_sha = _git_head(workspace)
    pre_untracked = _git_untracked(workspace)
    # The caller's pre-prep uncommitted tracked modifications, captured so a
    # failure rollback restores them instead of resetting the whole tree to HEAD.
    pre_diff = _git_diff_patch(workspace, prep_base_sha)

    def _restore_sources() -> None:
        _restore(src_snapshot)
        if prep_base_sha:
            _git(workspace, "checkout", "--", *[p.as_posix() for p in protected])

    def _restore_kernel_workspace() -> None:
        # Undo everything the agent did while preserving the caller's pre-prep
        # state: reset tracked files to HEAD, drop only prep-created untracked
        # files, re-apply the caller's original uncommitted tracked modifications,
        # then authoritatively restore the protected source bytes (covers untracked
        # source too). Never blanket-discards the caller's uncommitted work.
        if prep_base_sha:
            _git(workspace, "reset", "-q")
            _git(workspace, "checkout", "--", ".")
            _remove_new_untracked(workspace, pre_untracked)
            _git_apply_patch(workspace, pre_diff)
        _restore(src_snapshot)

    def _rollback() -> None:
        _restore_kernel_workspace()

    driver_rel = os.path.relpath(driver_path, agent_workspace)
    evidence = _build_evidence(
        workspace=workspace,
        kernel=kernel,
        driver=str(driver_path),
        program_md=program_md,
        target_functions=target_functions,
        source_files=source_files,
        preflight=preflight,
    )

    # Give the agent REAL reference files to Read (copied into the workspace so
    # they are inside its cwd). Authoring-only: retired before every preflight
    # verdict and on cleanup, never committed.
    ref_dir = _materialize_reference(workspace)
    # Kept apart from the bundle's own file list so an attempt whose
    # re-materialization failed can drop that list and keep these (see
    # ``_open_scaffold``).
    reference_prefix = ""
    if driver_external:
        reference_prefix = (
            "## Transactional external driver staging\n"
            f"The driver and every helper it imports MUST be written under the "
            f"isolated staging directory "
            f"`{driver_access_dir}`. The kernel workspace is source evidence only; "
            "changes are published to the caller's artifact directory only after "
            "deterministic validation succeeds. Existing task metadata and invocation "
            "specifications are read-only.\n\n"
        )
    # The spec lives beside the driver, NOT in the reference bundle: the driver
    # may read it at runtime, so it has to outlive preparation and be committed
    # alongside the driver it feeds.
    spec_path, canonical_spec = _materialize_invocation_spec(
        invocation_spec_file,
        driver_path.parent,
    )
    if invocation_spec_file and spec_path is None:
        # Not a default: an explicitly supplied spec that cannot be used costs the
        # prompt's case table, the driver's durable runtime input and the
        # committed-alongside check all at once, and the caller asked for it.
        log.warning(
            "could not materialize the invocation specification %s beside the "
            "driver; preparation continues without its case table, without a "
            "durable runtime input for the driver, and without the check that the "
            "spec is committed with it",
            invocation_spec_file,
        )
    invocation_note = _invocation_spec_note(spec_path, agent_workspace)
    expected_case_ids = list(expected_case_ids or [])
    backend_protected_files = [
        *(path.as_posix() for path in protected),
        *(read_only_files or []),
        *([spec_path.as_posix()] if spec_path is not None else []),
    ]

    # (2) Pre-place a correct, capture-guarded graph_harness.py so the agent can
    # import a known-good cuda_graph_bench (accepting dirty/verify) instead of
    # writing its own — a self-written harness can silently mismatch its driver
    # calls and degrade graph timing to eager. NOT forced: the agent may still do
    # custom timing in the driver for non-capturable ops. We only place it when
    # the task did not ship its own, and we keep it correct across attempts.
    canonical_harness = None if harness_path.is_file() else _find_reference_harness(ref_dir)
    provided_harness = canonical_harness is not None
    if provided_harness:
        harness_path.write_text(canonical_harness)
        harness_display = str(harness_path) if driver_external else "./graph_harness.py"
        reference_prefix = (
            "## Graph timing harness (already available beside the driver)\n"
            f"`{harness_display}` is a correct, capture-guarded harness. Import it —\n"
            "`from graph_harness import cuda_graph_bench` (it accepts optional\n"
            "`dirty`/`verify`). Do NOT rewrite it. Only implement custom timing in the\n"
            "driver if this operator genuinely cannot be captured into a static-input\n"
            "graph.\n\n" + reference_prefix
        )
    reference_note = reference_prefix + _reference_note(ref_dir, agent_workspace)

    def _open_scaffold() -> None:
        """Put the authoring-only reference bundle back for the next attempt."""
        nonlocal reference_note
        if ref_dir is None or ref_dir.is_dir():
            return
        materialized = _materialize_reference(workspace)
        if materialized is None:
            # The note enumerates the contract and the reference drivers by path,
            # in a prompt that also tells the agent not to rely on memory, so
            # keeping it would send this attempt to Read files that are gone.
            log.warning(
                "could not re-materialize the authoring reference bundle at %s; "
                "this attempt's prompt drops its file list",
                ref_dir,
            )
        reference_note = reference_prefix + _reference_note(materialized, agent_workspace)

    def _reset_scaffold() -> None:
        # Put the workspace into the exact state the driver will be judged in —
        # which is also the state the loop's baseline will run it in. Protected
        # source and the provided harness are restored, the durable invocation
        # spec is un-tampered, and the authoring-only reference bundle is RETIRED:
        # validating with scaffolding that the prep commit then deletes certifies
        # a filesystem that never exists again, which is exactly how a driver
        # reading its case table from the bundle passed preflight and crashed on
        # the very first baseline bench.
        if external_transaction is not None:
            _restore_kernel_workspace()
            external_transaction.restore_passthroughs()
        else:
            _restore_sources()
        _safe_rmtree(ref_dir)
        if ref_dir is not None and ref_dir.exists():
            # A partial removal leaves both halves of the invariant broken at once
            # and neither is visible later: preflight judges the driver against
            # scaffolding the prep commit then deletes, and `git add -A` carries
            # what survived into the pristine commit.
            raise ScaffoldRetirementError(
                f"could not retire the authoring reference bundle at {ref_dir}; the "
                "driver would be validated against scaffolding that preparation "
                "then deletes, and the surviving files would enter the pristine "
                "commit"
            )
        if provided_harness:
            harness_path.write_text(canonical_harness)
        if spec_path is not None and canonical_spec:
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(canonical_spec, encoding="utf-8")

    async def _finish_success(
        pf: PreflightResult,
        attempt_count: int,
    ) -> PrepareResult:
        # Drop an unused provided harness so it does not become persistent
        # scaffolding when the driver does not import it.
        if provided_harness and not driver_external:
            try:
                uses_harness = "graph_harness" in driver_path.read_text()
            except Exception:
                uses_harness = True
            if not uses_harness:
                _safe_unlink(harness_path)
        if driver_external:
            # Publish the complete validated driver/helper change set from the
            # isolated staging tree. The editable kernel repository is restored
            # first and is never part of this external artifact transaction.
            _restore_kernel_workspace()
            assert external_transaction is not None
            external_transaction.restore_passthroughs()
            try:
                changes = external_transaction.publish()
            except ExternalArtifactError as exc:
                rollback_error = ""
                try:
                    external_transaction.rollback()
                except ExternalArtifactError as rollback_exc:
                    rollback_error = f"; rollback failed: {rollback_exc}"
                return PrepareResult(
                    ok=False,
                    attempts=attempt_count,
                    wrote_files=[],
                    created_files=[],
                    rolled_back=not rollback_error,
                    final_preflight=pf,
                    message=f"could not publish external driver artifacts: {exc}{rollback_error}",
                    audit_dir=audit_dir_str,
                )
            return PrepareResult(
                ok=True,
                attempts=attempt_count,
                wrote_files=list(changes.wrote_files),
                created_files=list(changes.created_files),
                rolled_back=False,
                final_preflight=pf,
                message="external task prepared",
                audit_dir=audit_dir_str,
            )

        # In-repository task scaffolding must become part of pristine before
        # IterationLoop captures its base SHA. Stage every newly authored source
        # file with -A (unlike the loop's -u), but exclude forge_experiments:
        # it holds the campaign's own run state, candidates and workspace.lock,
        # which are not part of the task and must never enter the pristine commit.
        # The authoring-only reference bundle is already gone (_reset_scaffold).
        _git(workspace, "add", "-A", "--", ".", ":(exclude)forge_experiments")
        # The driver is about to become pristine; anything it reads at runtime has
        # to become pristine with it. An ignored spec would leave a committed
        # driver whose input is untracked and can be cleaned away at any point
        # after this — checked before the commit so there is nothing to undo.
        spec_indexed = None if spec_path is None else _git_indexed(workspace, spec_path)
        if spec_path is not None and spec_indexed is not True:
            _rollback()
            return PrepareResult(
                ok=False,
                attempts=attempt_count,
                wrote_files=[],
                created_files=[],
                rolled_back=True,
                final_preflight=pf,
                message=(
                    "prepared a conforming driver but its invocation "
                    f"specification ({spec_path}) cannot be committed alongside "
                    "it, so the driver's runtime input would not be durable — "
                    + (
                        "check the workspace's git ignore rules"
                        if spec_indexed is False
                        else "git could not be asked whether it is staged, so the "
                        "ignore rules are not necessarily the reason"
                    )
                ),
                audit_dir=audit_dir_str,
            )
        _, commit_out = _git(
            workspace,
            "commit",
            "-m",
            "forge prepass: prepared measurement driver",
        )
        if _git_head(workspace) == prep_base_sha:
            _rollback()
            return PrepareResult(
                ok=False,
                attempts=attempt_count,
                wrote_files=[],
                created_files=[],
                rolled_back=True,
                final_preflight=pf,
                message=(f"prepared a conforming driver but the git commit did not land: {commit_out.strip()[-200:]}"),
                audit_dir=audit_dir_str,
            )
        return PrepareResult(
            ok=True,
            attempts=attempt_count,
            wrote_files=_git_changed_since(workspace, prep_base_sha),
            created_files=[],
            rolled_back=False,
            final_preflight=pf,
            message="task prepared",
            audit_dir=audit_dir_str,
        )

    start = time.monotonic()
    # The in-loop preflight must respect the preparation wall (deadline_sec), not
    # only the outer per-kernel deadline_unix. Anchored here (matching the loop's
    # own `remaining` accounting), this absolute deadline caps every preflight so
    # a late one can't run past the prep wall; each stage still additionally
    # clamps to deadline_unix. _deadline_timeout recomputes the remaining budget
    # per call, so a fixed absolute anchor shrinks correctly as time passes.
    preflight_deadline_unix = time.time() + deadline_sec
    if deadline_unix > 0:
        preflight_deadline_unix = min(preflight_deadline_unix, deadline_unix)
    attempts = 0
    prior_failure = ""
    prior_failure_heading = RETRY_HEADING_DEFAULT
    last_pf = preflight
    external_rollback_error = ""
    # An attempt that leaves the driver byte-identical is a distinct failure from
    # one that edited it badly, and the two need different guidance. Observed in
    # a real run: both attempts hit the agent timeout having written nothing, yet
    # the retry prompt said "your previous attempt still did NOT pass", and the
    # operator-facing failure quoted preflight reasons that made a never-touched
    # driver look broken.
    edited_any_attempt = False
    starved_retry_sec = 0.0
    scaffold_error = ""
    try:
        while attempts < PREPARE_MAX_ATTEMPTS:
            remaining = deadline_sec - (time.monotonic() - start)
            if remaining <= 10:
                break
            if attempts and remaining < PREPARE_MIN_RETRY_SEC:
                starved_retry_sec = remaining
                break
            attempts += 1
            is_last_attempt = attempts >= PREPARE_MAX_ATTEMPTS
            spendable = max(0.0, remaining - _SALVAGE_RESERVE_SEC)
            attempt_timeout = spendable if is_last_attempt else min(spendable, float(PER_ATTEMPT_CAP_SEC))
            # Each attempt authors against the reference bundle; the preceding
            # attempt's verdict retired it (see _reset_scaffold).
            _open_scaffold()
            prompt = _build_prompt(
                evidence,
                driver_rel,
                reference_note,
                prior_failure,
                invocation_note,
                prior_failure_heading,
                _distributed_contract_note(nproc_per_node),
            )
            attempt_dir = f"attempt_{attempts:02d}"
            _audit_text(f"{attempt_dir}/prompt.md", prompt)
            _audit_text(f"{attempt_dir}/system_prompt.md", _SYSTEM_PROMPT)
            _audit_driver(f"{attempt_dir}/driver_before.py")
            digest_before = _driver_digest()
            progress_log: list[str] = []
            agent_started = time.monotonic()

            try:
                agent_output = await _run_prepare_agent(
                    config=config,
                    workspace=agent_workspace,
                    system_prompt=_SYSTEM_PROMPT,
                    prompt=prompt,
                    timeout_sec=attempt_timeout,
                    additional_dirs=([str(workspace)] if driver_external else None),
                    allow_shell=not driver_external,
                    target_files=[
                        driver_path.as_posix(),
                        harness_path.as_posix(),
                    ],
                    protected_files=backend_protected_files,
                    usage=usage,
                    progress_log=progress_log,
                )
            except asyncio.TimeoutError:
                _audit_driver(f"{attempt_dir}/driver_at_timeout.py")
                elapsed_s = round(time.monotonic() - agent_started, 3)
                driver_edited = _detect_driver_edited(digest_before)
                edited_any_attempt = edited_any_attempt or driver_edited
                _audit_json(
                    f"{attempt_dir}/agent_event.json",
                    {
                        "status": "timeout",
                        "elapsed_s": elapsed_s,
                        "budget_s": round(attempt_timeout, 3),
                        "driver_edited": driver_edited,
                    },
                )
                _audit_text(
                    f"{attempt_dir}/agent_progress.txt",
                    "\n".join(progress_log),
                )
                _reset_scaffold()
                # The Agent may have completed a valid driver before getting
                # stuck on self-verification. Salvage it deterministically.
                last_pf = await _preflight_async(
                    driver_path.as_posix(),
                    snr_threshold,
                    PREFLIGHT_WARMUP,
                    PREFLIGHT_ITERS,
                    require_graph=True,
                    require_profile=True,
                    deadline_unix=preflight_deadline_unix,
                    expected_case_ids=expected_case_ids,
                )
                _audit_driver(f"{attempt_dir}/driver_after_timeout_preflight.py")
                _audit_json(f"{attempt_dir}/preflight.json", asdict(last_pf))
                if last_pf.ok:
                    return await _finish_success(last_pf, attempts)
                prior_failure_heading = RETRY_HEADING_DEFAULT if driver_edited else RETRY_HEADING_NO_EDIT
                if driver_edited:
                    jit_hint = ""
                    if last_pf.all_failures_are_timeouts:
                        jit_hint = (
                            "\nNOTE: Every failure above is a TIMEOUT, not a crash. "
                            "This usually means the driver is structurally correct "
                            "but the first execution triggered slow JIT compilation "
                            "(CK/aiter kernels can take 44s+ per module). Do NOT "
                            "rewrite the driver from scratch — verify it is "
                            "structurally correct and resubmit; the next preflight "
                            "run benefits from a warm JIT cache.\n"
                        )
                    prior_failure = (
                        "Agent timed out, then deterministic preflight failed:\n" + last_pf.detail_report() + jit_hint
                    )
                else:
                    prior_failure = (
                        f"Your previous attempt ran for {elapsed_s:.0f}s and timed out "
                        f"having made NO edit at all to `{driver_rel}` — the file is "
                        "byte-identical to before. Reading and planning is not "
                        "progress here. Open the driver and WRITE the fix as your "
                        "first substantive action, then verify; if you cannot finish "
                        "the whole contract in the time you have, still leave the "
                        "best driver you can on disk rather than nothing.\n"
                        f"What you spent that time on — {summarize_agent_progress(progress_log)}\n"
                        "The deterministic check on that unchanged driver reported:\n" + last_pf.detail_report()
                    )
                remaining_after_timeout = deadline_sec - (time.monotonic() - start)
                if attempts < PREPARE_MAX_ATTEMPTS:
                    if remaining_after_timeout >= PREPARE_MIN_RETRY_SEC:
                        continue
                    if remaining_after_timeout > 10:
                        starved_retry_sec = remaining_after_timeout
                break
            except Exception as exc:  # noqa: BLE001
                _audit_driver(f"{attempt_dir}/driver_at_exception.py")
                driver_edited = _detect_driver_edited(digest_before)
                edited_any_attempt = edited_any_attempt or driver_edited
                _audit_json(
                    f"{attempt_dir}/agent_event.json",
                    {
                        "status": "error",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "elapsed_s": round(time.monotonic() - agent_started, 3),
                        "budget_s": round(attempt_timeout, 3),
                        "driver_edited": driver_edited,
                    },
                )
                _audit_text(
                    f"{attempt_dir}/agent_progress.txt",
                    "\n".join(progress_log),
                )
                prior_failure = f"Agent invocation error: {type(exc).__name__}: {exc}"
                prior_failure_heading = RETRY_HEADING_DEFAULT
                _reset_scaffold()
                continue
            else:
                _audit_text(f"{attempt_dir}/agent_output.txt", str(agent_output or ""))
                _audit_text(
                    f"{attempt_dir}/agent_progress.txt",
                    "\n".join(progress_log),
                )
                _audit_driver(f"{attempt_dir}/driver_after.py")
                driver_edited = _detect_driver_edited(digest_before)
                edited_any_attempt = edited_any_attempt or driver_edited
                _audit_json(
                    f"{attempt_dir}/agent_event.json",
                    {
                        "status": "completed",
                        "elapsed_s": round(time.monotonic() - agent_started, 3),
                        "budget_s": round(attempt_timeout, 3),
                        "driver_edited": driver_edited,
                    },
                )

            # (1) Source protection: whatever the agent did, restore source (and
            # the provided harness) to pristine before we judge the driver.
            _reset_scaffold()

            # require_graph=True: a produced driver must actually time under a
            # CUDA/HIP graph, not eagerly (nor silently fall back to eager).
            last_pf = await _preflight_async(
                driver_path.as_posix(),
                snr_threshold,
                PREFLIGHT_WARMUP,
                PREFLIGHT_ITERS,
                require_graph=True,
                require_profile=True,
                deadline_unix=preflight_deadline_unix,
                expected_case_ids=expected_case_ids,
            )
            _audit_driver(f"{attempt_dir}/driver_at_preflight.py")
            _audit_json(f"{attempt_dir}/preflight.json", asdict(last_pf))
            if last_pf.ok:
                return await _finish_success(last_pf, attempts)
            prior_failure_heading = RETRY_HEADING_DEFAULT if driver_edited else RETRY_HEADING_NO_EDIT
            jit_hint = ""
            if driver_edited and last_pf.all_failures_are_timeouts:
                jit_hint = (
                    "\nNOTE: Every failure above is a TIMEOUT, not a crash. "
                    "This usually means the driver is structurally correct "
                    "but the first execution triggered slow JIT compilation "
                    "(CK/aiter kernels can take 44s+ per module). Do NOT "
                    "rewrite the driver from scratch — verify it is "
                    "structurally correct and resubmit; the next preflight "
                    "run benefits from a warm JIT cache.\n"
                )
            prior_failure = (
                (
                    "Deterministic preflight after your edit:\n"
                    if driver_edited
                    else (
                        f"You finished without editing `{driver_rel}` at all. The "
                        "deterministic check therefore ran the SAME driver again:\n"
                    )
                )
                + last_pf.detail_report()
                + jit_hint
            )
    except ScaffoldRetirementError as exc:
        scaffold_error = str(exc)
    finally:
        # Never leave reference or external staging bundles behind.
        _safe_rmtree(ref_dir)
        if external_transaction is not None:
            if not external_transaction.published:
                try:
                    external_transaction.rollback()
                except ExternalArtifactError as exc:
                    external_rollback_error = str(exc)
            if not external_transaction.published:
                _rollback()
            try:
                external_transaction.close()
            except OSError as exc:
                if not external_rollback_error:
                    external_rollback_error = f"staging cleanup failed: {exc}"

    # (3) Failure: roll the workspace back to its exact pre-prep state (tracked
    # files reset to HEAD, caller's uncommitted mods re-applied, prep-created
    # untracked removed, protected source restored). See _rollback.
    _rollback()
    rolled_back = not external_rollback_error
    if scaffold_error:
        # Lead with it: the preflight reasons describe a driver judged in a state
        # that was never valid, so quoting them first would send the operator after
        # the driver.
        log.error("%s", scaffold_error)
        return PrepareResult(
            ok=False,
            attempts=attempts,
            wrote_files=[],
            created_files=[],
            rolled_back=rolled_back,
            final_preflight=last_pf,
            message=scaffold_error,
            audit_dir=audit_dir_str,
        )
    return PrepareResult(
        ok=False,
        attempts=attempts,
        wrote_files=[],
        created_files=[],
        rolled_back=rolled_back,
        final_preflight=last_pf,
        message=(
            (
                f"prep wall exhausted after {attempts} attempt(s); the remaining "
                f"{starved_retry_sec:.0f}s is below the {PREPARE_MIN_RETRY_SEC}s "
                "minimum retry budget, so no further attempt was started — raise "
                "the per-kernel deadline to give preparation more room"
            )
            if starved_retry_sec
            else "could not produce a conforming driver within the budget"
            if edited_any_attempt or not attempts
            else (
                f"prep agent never edited the driver in {attempts} attempt(s); "
                "the driver is unchanged, so the preflight reasons below describe "
                "the ORIGINAL driver, not a failed repair"
            )
        )
        + (f"; external artifact rollback failed: {external_rollback_error}" if external_rollback_error else ""),
        audit_dir=audit_dir_str,
    )


def prepare_task_sync(**kwargs) -> PrepareResult:
    """Synchronous wrapper for CLI code."""

    return asyncio.run(prepare_task(**kwargs))


# ---------------------------------------------------------------------------
# Embedded canonical assets (examples/ is not packaged in the wheel)
# ---------------------------------------------------------------------------

DRIVER_CONTRACT_SPEC = """\
## forge-loop driver contract (what the driver MUST satisfy)

forge-loop treats the driver as a black box run as `python driver.py <args>` and
reads it purely over stdout. The driver owns all case selection:

Correctness — `python driver.py`
must run the complete correctness suite and
prints (at least one):
    SNR: 62.13 dB          # preferred; forge pre-filters on this vs the SNR threshold
    allclose: True         # optional fallback
Benchmark — `python driver.py --warmup <n> --iters <n> --bench-mode`
must run the complete benchmark suite and
prints per-iteration:
    wall_ms: 0.081920      # one line per timed iteration (forge takes the median)
  or one aggregate line: `median_ms: <ms>` / `mean_ms: <ms>` (label it honestly).
It MUST additionally print `case_ms: <case_id> <ms>` for every case the task
declares. The case_id is a no-whitespace token; when the invocation
specification lists `tests.driver_contract.case_selectors`, use each entry's
`CASE_ID` verbatim. The deterministic check compares your `case_ms` ids against
that declared set, and a driver that reports only some of them is rejected.
Single-case sweep — `python driver.py --warmup <n> --iters <n> --bench-mode --bench-case <case_id>`
SHOULD benchmark that one case and print only its lines, exiting non-zero if the
id is not one it declares. This is what makes "hold the code, vary one dispatch
constant, time one shape" cost seconds instead of a full suite; a driver that
ignores the flag still measures correctly, it just makes every such question cost
the whole suite.
Profiling — `python driver.py --profile-run`
selects one representative case inside the driver and runs only its target
kernel (no reference/correctness path), performs
only enough warmup to settle JIT selection, launches 1-3 profiled iterations,
synchronizes, and exits 0 without printing timing data.

Rules:
- Case definitions come from the task's real harness/config; do not invent
  hard-coded dimensions in the driver.
- Runtime inputs must be DURABLE. The driver is committed and then re-run
  unchanged, many times, by the optimization loop. The only non-source files
  guaranteed to exist then are the driver itself, the helper modules you write
  beside it, and the invocation specification beside it. Everything else handed
  to you for authoring — the reference example bundle above all — is deleted
  before the driver is validated and committed. Reading any of it at runtime
  produces a driver that passes validation and then crashes on the loop's first
  measurement. Resolve durable paths relative to the driver's own directory
  (`Path(__file__).resolve().parent`), never relative to the process cwd.
- Deterministic inputs use a fixed seed.
- Exit 0 on success; a non-zero exit is treated as a crash.
- REQUIRED: the benchmark MUST actually run under a CUDA/HIP graph — it must
  capture the op into a graph and REPLAY it once per timed iteration. The prepass
  verifies this for real (it counts `torch.cuda.CUDAGraph` replays during the
  bench), so printing a label is NOT enough and eager timing is REJECTED. The
  simplest way to satisfy it is to bench through the provided graph_harness.

## Why the benchmark MUST run under a CUDA/HIP graph (required)
A small kernel's wall time is dominated by host-side launch/dispatch overhead, not
GPU work. Timing eagerly makes the optimizer chase host cost it cannot change,
makes iteration-to-iteration numbers noisy and incomparable (breaking keep/revert),
and does not match production (AMD serving runs these ops under a HIP graph). So
the produced driver MUST time under a graph: capture ONE invocation and time
replays, so CUDA events bracket only GPU execution. Allocate inputs once, launch
on the CURRENT stream (so capture records the kernel), and pass dirty/verify so an
empty/invalid graph is detected instead of reported as a fake speedup. The verify
callback checks that graph replay actually ran the kernel — use SNR-based
verification (`_snr_db(ref, out) > 30.0`), NOT `torch.allclose`, because FP8 and
quantized kernels can produce results that differ enough from the torch reference
to fail allclose yet are numerically correct. Verification is by actual replay
count (via torch.cuda.CUDAGraph), not a printed label — an eager run (or a silent
eager fallback) performs no replays and is rejected. Make the capture work
(allocate once, reuse the same output buffer, launch on the current stream) rather
than settling for eager timing.
"""


REFERENCE_DRIVER_TEMPLATE = r'''
"""Measurement driver — correctness (SNR) + graph-timed benchmark + profiling."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from graph_harness import cuda_graph_bench
# Import the kernel's STABLE public entry point (adapt this import + call):
from your_kernel_module import your_entry_point  # noqa: F401

_SEED = 0

# Load every scored case from the task's existing harness or configuration. Do
# not invent dimensions here. Keys are the case IDs used in case_ms lines (the
# invocation spec's CASE_ID values when the task declares them). Resolve any file
# you read here against _HERE, and only read files that outlive preparation.
_HERE = Path(__file__).resolve().parent
CASES = {}


def _make_inputs(dims, mode, device):
    torch.manual_seed(_SEED)
    x = torch.randn(dims["M"], dims["N"], device=device, dtype=torch.float16)
    if mode == "stability":
        x = x * 50.0
    return x


def _reference(x):
    # Replace with the operator's reference (e.g. torch.softmax(x, dim=-1)).
    raise NotImplementedError


def _snr_db(ref, test):
    ref = ref.float(); test = test.float()
    noise = test - ref
    sp = torch.mean(ref * ref).item(); npow = torch.mean(noise * noise).item()
    if npow <= 0:
        return 100.0
    if sp <= 0:
        return 0.0
    return 10.0 * math.log10(sp / npow)


def _run_correctness(device):
    snrs = []
    close = True
    for dims in CASES.values():
        x = _make_inputs(dims, "full", device)
        out = your_entry_point(x)
        ref = _reference(x)
        snrs.append(_snr_db(ref, out))
        close = close and torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {close}")
    return 0


def _run_bench(dims, case_id, warmup, iters, device):
    x = _make_inputs(dims, "full", device)
    ref = _reference(x)
    out = torch.empty_like(x)
    step = lambda: your_entry_point(x, out)  # noqa: E731
    res = cuda_graph_bench(
        step, warmup=warmup, iters=iters,
        dirty=lambda: out.zero_(),
        verify=lambda: _snr_db(ref, out) > 30.0,
    )
    for t in res["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(res["times_ms"])
    print(f"case_ms: {case_id} {times[len(times) // 2]:.6f}")
    return 0


def _run_profile(dims, device):
    x = _make_inputs(dims, "full", device)
    for _ in range(3):
        your_entry_point(x)
    torch.cuda.synchronize()
    for _ in range(3):
        your_entry_point(x)
    torch.cuda.synchronize()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--bench-case", default="")
    p.add_argument("--profile-run", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    args, _ = p.parse_known_args()

    if not torch.cuda.is_available():
        print("error: no GPU"); return 1
    device = "cuda"

    if args.profile_run:
        profile_dims = next(iter(CASES.values()))
        return _run_profile(profile_dims, device)

    if args.bench_mode:
        selected = CASES
        if args.bench_case:
            if args.bench_case not in CASES:
                print(f"error: unknown case {args.bench_case}"); return 1
            selected = {args.bench_case: CASES[args.bench_case]}
        for case_id, case_dims in selected.items():
            _run_bench(case_dims, case_id, args.warmup, args.iters, device)
        return 0

    return _run_correctness(device)


if __name__ == "__main__":
    sys.exit(main())
'''
