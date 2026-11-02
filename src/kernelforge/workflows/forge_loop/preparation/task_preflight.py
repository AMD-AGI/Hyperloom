# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic driver-contract preflight: correctness, benchmark, graph-replay and profile probes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import glob
import logging
import os
import pathlib
import shutil
import signal
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.evaluation.benchmark.runner import BenchRun, bench_wallclock
from kernelforge.evaluation.correctness.driver import test_correctness
from kernelforge.evaluation.profiling.contracts import PROFILE_RUN_FLAG
from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB
from kernelforge.utils.aiter_cache import cleanup_current_owned_aiter_locks

log = logging.getLogger(__name__)

# Preflight — deterministic driver-contract validation


@dataclass
class PreflightResult:
    """Whether a driver conforms to the forge-loop stdout contract."""

    ok: bool
    correctness_ok: bool
    bench_ok: bool
    graph_ok: bool = True
    profile_ok: bool = True
    ranks_ok: bool = True
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    # Raw stdout+stderr tail per failed stage ("correctness", "bench", ...).
    diagnostics: dict = field(default_factory=dict)
    # Wall time the whole check took, with per-stage seconds in ``details``.
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
        """True when every primary failure is a TIMEOUT, none a CRASH."""
        if self.ok or not self.reasons:
            return False
        _CASCADING = ("cannot verify", "could not verify")
        _TIMEOUT_TOKENS = ("TIMEOUT", "timed out")
        primary = [r for r in self.reasons if not r.startswith(_CASCADING) or any(t in r for t in _TIMEOUT_TOKENS)]
        if not primary:
            return False
        return all(any(t in r for t in _TIMEOUT_TOKENS) for r in primary) and not any("CRASHED" in r for r in primary)


# Counts ACTUAL torch.cuda.CUDAGraph.replay calls (HIP graphs go through the same API on ROCm), detecting real graph
# timing independently of whatever the driver prints: an eager driver replays zero times, a graph-timed one replays
# once per timed iteration.
_GRAPH_PROBE_SITECUSTOMIZE = r'''
import atexit, json, os, sys

_n = [0]
# Harness evidence is collected only when the caller declares a rank count.
_expect_ranks = 0
try:
    _expect_ranks = int(os.environ.get("GRAPH_PROBE_EXPECT_RANKS") or 0)
except ValueError:
    _expect_ranks = 0


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
        except (OSError, ValueError, IndexError):
            break
    return ancestors


_ancestors = _ancestor_pids()
_import_pid = os.getpid()


def _install(torch):
    """Patch CUDAGraph.replay lazily: torch may not be imported yet.

    A module named torch may be a stub or partially initialized. Use the
    existing module rather than importing through our own lazy hook again.
    """
    try:
        orig = torch.cuda.CUDAGraph.replay

        def _replay(self, *a, **k):
            _n[0] += 1
            return orig(self, *a, **k)

        torch.cuda.CUDAGraph.replay = _replay
        return True
    except (AttributeError, TypeError):
        return False


_graph_ready = _install(sys.modules.get("torch"))

if not _graph_ready:
    # torch is imported by the driver, not by us. Hook the import so the patch
    # lands before any graph is created.
    import builtins

    _real_import = builtins.__import__

    def _hooked(name, *a, **k):
        global _graph_ready
        mod = _real_import(name, *a, **k)
        if not _graph_ready and (name == "torch" or name.startswith("torch.")):
            _graph_ready = _install(sys.modules.get("torch"))
        if _graph_ready and builtins.__import__ is _hooked:
            builtins.__import__ = _real_import
        return mod

    builtins.__import__ = _hooked


def _harness_measured():
    """Whether this rank measured inside dist_harness rather than on its own.

    The harness owns every property a multi-rank measurement has to hold -- the
    launch, the device binding, per-rank seeding, barrier placement, the
    cross-rank reduction and the teardown -- so this one fact stands in for all
    of them. A driver that reimplemented the launch is refused for that, rather
    than for whichever property it happened to get wrong.
    """
    import sys as _sys

    module = _sys.modules.get("dist_harness")
    if module is None:
        return False
    try:
        return int(module.MEASURED[0]) > 0
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _dump():
    out = os.environ.get("GRAPH_PROBE_OUT")
    if not out:
        return
    try:
        payload = {
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
        }
        if _expect_ranks > 1:
            payload["harness"] = _harness_measured()
        # One file per process: ranks of a torchrun job would otherwise
        # overwrite each other and the count would be one rank's, or zero.
        with open(f"{out}.{os.getpid()}", "w") as fh:
            json.dump(payload, fh)
    except OSError as exc:
        sys.stderr.write(f"graph probe: could not write {out}.{os.getpid()}: {exc}\n")


atexit.register(_dump)
'''


# First-run JIT compilation of CK/aiter GEMM kernels on gfx950/rocm (serial baton-lock builds, ~44s+ per module)
# routinely blows the old 120s correctness / 300s bench preflight budgets, so task_preparation fails ("could not
# produce a conforming driver within the budget") before the loop even starts — even though the driver is fine and
# just needs to compile once.
PREFLIGHT_CORRECTNESS_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_CORRECTNESS_TIMEOUT", "1800") or "1800")
PREFLIGHT_BENCH_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_BENCH_TIMEOUT", "1800") or "1800")
# graph-replay and profiling preflight run *after* bench, so the JIT cache is usually warm by then, but on a cold
# first run a fresh module can still compile here.
PREFLIGHT_GRAPH_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_GRAPH_TIMEOUT", "900") or "900")
PREFLIGHT_PROFILE_TIMEOUT_S = int(os.environ.get("FORGE_PREFLIGHT_PROFILE_TIMEOUT", "900") or "900")


# How much of a failed stage's stdout+stderr to carry into the audit record and the repair agent's next prompt.
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
        with contextlib.suppress(OSError):
            os.unlink(path)
    shutil.rmtree(probe_dir, ignore_errors=True)


# Replay counts are non-negative, so negative values carry the reason a count
# could not be produced. They are distinguished because they belong to
# different stages: a probe that never ran says nothing about the driver, while
# a violated distributed contract is a verdict on it.
PROBE_FAILED = -1
PROBE_DISTRIBUTED_VIOLATION = -2


@dataclass(frozen=True)
class _ProbeShard:
    """One graph-probe shard, once its replay count and identity fields check out.

    ``rank`` is None for a launcher or helper process -- one that wrote a shard
    but is not a worker -- and the identity fields are then unset with it.
    """

    replays: int
    rank: int | None
    pid: int | None
    ppid: int | None
    ancestors: frozenset[int]
    world_size: int | None
    payload: dict


def _shard_process_identity(payload: dict, name: str) -> tuple[int | None, int | None, frozenset[int], str]:
    """A shard's process identity, or why it cannot tell a worker from a helper it launched."""
    pid_value = payload.get("pid")
    ppid_value = payload.get("ppid")
    try:
        pid = int(pid_value) if pid_value is not None else None
        ppid = int(ppid_value) if ppid_value is not None else None
    except (TypeError, ValueError):
        return None, None, frozenset(), f"invalid process identity in {name}"
    if (pid is None) != (ppid is None) or (pid is not None and (pid <= 0 or ppid is None or ppid < 0)):
        return None, None, frozenset(), f"invalid process identity in {name}"
    raw_ancestors = payload.get("ancestors") or []
    if not isinstance(raw_ancestors, list):
        return None, None, frozenset(), f"invalid process ancestry in {name}"
    try:
        ancestors = frozenset(int(ancestor) for ancestor in raw_ancestors)
    except (TypeError, ValueError):
        return None, None, frozenset(), f"invalid process ancestry in {name}"
    if any(ancestor <= 0 for ancestor in ancestors):
        return None, None, frozenset(), f"invalid process ancestry in {name}"
    return pid, ppid, ancestors, ""


def _parse_graph_probe_shard(shard: Path) -> tuple[_ProbeShard | None, str]:
    """One shard's contents, or why it cannot be read as a measurement."""
    name = shard.name
    try:
        payload = json.loads(shard.read_text().strip())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid graph probe shard {name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid graph probe shard payload in {name}"
    try:
        replays = int(payload["replays"])
    except (KeyError, TypeError, ValueError):
        return None, f"invalid replay count in {name}"
    if replays < 0:
        return None, f"invalid negative replay count in {name}"

    rank_value = payload.get("rank")
    local_rank_value = payload.get("local_rank")
    world_size_value = payload.get("world_size")
    unranked = _ProbeShard(replays, None, None, None, frozenset(), None, payload)
    if rank_value is None:
        if local_rank_value is not None:
            return None, f"incomplete rank identity in {name}"
        # Launcher/helper processes are not workers even if they inherited a WORLD_SIZE value from their
        # environment.
        return unranked, ""
    if "local_rank" in payload and local_rank_value is None:
        # A self-launcher can inherit RANK/WORLD_SIZE from its caller.
        return unranked, ""
    if world_size_value is None:
        return None, f"incomplete rank identity in {name}"
    try:
        rank = int(rank_value)
        local_rank = int(local_rank_value) if local_rank_value is not None else None
        world_size = int(world_size_value)
    except (TypeError, ValueError):
        return None, f"invalid rank identity in {name}"
    if (
        world_size <= 0
        or rank < 0
        or rank >= world_size
        or (local_rank is not None and (local_rank < 0 or local_rank >= world_size))
    ):
        return None, f"invalid rank identity in {name}"
    pid, ppid, ancestors, error = _shard_process_identity(payload, name)
    if error:
        return None, error
    return _ProbeShard(replays, rank, pid, ppid, ancestors, world_size, payload), ""


def _rank_root_shard(rank: int, shards: list[_ProbeShard]) -> tuple[_ProbeShard | None, str]:
    """The worker shard for one rank: the process every other shard of that rank descends from."""
    if len(shards) == 1:
        return shards[0], ""
    if any(shard.pid is None for shard in shards):
        return None, f"ambiguous graph probe shards for rank {rank}"
    roots = [
        shard
        for shard in shards
        if all(
            other.pid == shard.pid or shard.pid in other.ancestors or (not other.ancestors and other.ppid == shard.pid)
            for other in shards
        )
    ]
    if len(roots) != 1:
        return None, f"ambiguous graph probe process tree for rank {rank}"
    return roots[0], ""


def _rank_set_error(actual_ranks: set[int], world_size: int) -> str:
    """Why the shards do not cover exactly the ranks the run declared, or "" when they do."""
    expected_ranks = set(range(world_size))
    if actual_ranks == expected_ranks:
        return ""
    missing = sorted(expected_ranks - actual_ranks)
    unexpected = sorted(actual_ranks - expected_ranks)
    error = f"incomplete graph probe rank set; missing ranks: {missing}"
    if unexpected:
        error += f"; unexpected ranks: {unexpected}"
    return error


def _read_graph_probe_shards(out_path: str, *, expected_world_size: int | None = None) -> tuple[int, str]:
    """Validate graph-probe shards and return the effective replay count.

    ``expected_world_size`` additionally holds the shards to the distributed
    contract: the declared rank count must be the one that actually launched,
    and every rank must have measured inside ``dist_harness``.

    Those are two facts, not a reading of how the run behaved. The harness owns
    device binding, per-rank seeding, barrier placement, the cross-rank
    reduction and the teardown, so a rank that ran inside it holds all of them
    by construction and one that did not is refused for that -- see
    :func:`_distributed_contract_violation`.
    """
    unranked_replays: list[int] = []
    ranked_shards: dict[int, list[_ProbeShard]] = {}
    world_sizes: set[int] = set()

    for path in glob.glob(f"{out_path}.*"):
        parsed, error = _parse_graph_probe_shard(Path(path))
        if error:
            return PROBE_FAILED, error
        if parsed.rank is None:
            unranked_replays.append(parsed.replays)
            continue
        ranked_shards.setdefault(parsed.rank, []).append(parsed)
        world_sizes.add(parsed.world_size)

    if not ranked_shards:
        if expected_world_size is not None and expected_world_size > 1:
            return (
                PROBE_DISTRIBUTED_VIOLATION,
                f"the benchmark ran single-process; this task declares {expected_world_size} ranks, so the "
                "driver must re-exec itself under torch.distributed.run when RANK is absent",
            )
        # A normal single-process driver writes one shard.
        return max(unranked_replays, default=0), ""
    if len(world_sizes) != 1:
        return PROBE_FAILED, "graph probe rank shards disagree on world_size"

    world_size = next(iter(world_sizes))
    if expected_world_size is not None and world_size != expected_world_size:
        return (
            PROBE_DISTRIBUTED_VIOLATION,
            f"the benchmark launched {world_size} ranks; this task declares {expected_world_size}",
        )
    rank_error = _rank_set_error(set(ranked_shards), world_size)
    if rank_error:
        return PROBE_FAILED, rank_error

    worker_replays: list[int] = []
    worker_payloads: dict[int, dict] = {}
    for rank, shards in ranked_shards.items():
        root, error = _rank_root_shard(rank, shards)
        if error:
            return PROBE_FAILED, error
        worker_replays.append(root.replays)
        worker_payloads[rank] = root.payload

    if expected_world_size is not None and expected_world_size > 1:
        violation = _distributed_contract_violation(worker_payloads)
        if violation:
            return PROBE_DISTRIBUTED_VIOLATION, violation

    # Unranked launcher shards and ranked helper descendants must not count as workers.
    return min(worker_replays), ""


def _distributed_contract_violation(worker_payloads: dict[int, dict]) -> str:
    """Return why the observed multi-rank run is not a trustworthy measurement.

    One question is asked, because one answer settles the rest: did the ranks
    measure inside ``dist_harness``? The harness binds each rank to its own
    device, seeds them apart, keeps the timed region free of synchronization,
    reduces with the slowest rank and destroys the process group -- properties
    that, inferred one at a time from what the run happened to do, would each be
    weaker than the property it stood for. A driver that went its
    own way is refused for that, not for whichever of them it broke first.
    """
    outside = sorted(rank for rank, shard in worker_payloads.items() if shard.get("harness") is not True)
    if outside:
        return (
            f"rank(s) {outside} did not measure inside dist_harness. Every guarantee this task's number "
            "rests on -- one device per rank, inputs that differ per rank, no synchronization in the timed "
            "region, the slowest rank's time, a process group that is torn down -- belongs to the harness, "
            "so a driver that launches the ranks itself is not measured. Call dist_harness.run() from the "
            "driver's __main__ and supply build_inputs / call_candidate / reference"
        )
    return ""


async def _count_graph_replays(
    driver: str,
    warmup: int,
    iters: int,
    *,
    timeout_sec: float = 300,
    require_ranks: int = 1,
) -> tuple[int, str]:
    """Run the driver and return its effective CUDA graph replay count."""
    fd, out_path = tempfile.mkstemp(prefix="forge_graph_probe_")
    os.close(fd)
    probe_dir = tempfile.mkdtemp(prefix="forge_graph_probe_site_")
    pathlib.Path(probe_dir, "sitecustomize.py").write_text(_GRAPH_PROBE_SITECUSTOMIZE)
    # PYTHONPATH rather than a wrapper script: torchrun children of a self-launching driver inherit the environment,
    # so each rank imports the counter and reports its own replays.
    env = dict(
        os.environ,
        GRAPH_PROBE_OUT=out_path,
        PYTHONPATH=os.pathsep.join([probe_dir, *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])]),
    )
    if require_ranks > 1:
        # The driver reads the rank count it must launch; the probe reads it to
        # decide whether to observe the distributed contract at all.
        env["FORGE_NPROC_PER_NODE"] = str(require_ranks)
        env["GRAPH_PROBE_EXPECT_RANKS"] = str(require_ranks)
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
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=10)
        cleanup_current_owned_aiter_locks()
        _cleanup_probe(out_path, probe_dir)
        return PROBE_FAILED, "benchmark timed out"
    except asyncio.CancelledError:
        _kill_process_group(proc)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=10)
        cleanup_current_owned_aiter_locks()
        _cleanup_probe(out_path, probe_dir)
        raise
    except OSError as exc:
        _cleanup_probe(out_path, probe_dir)
        return PROBE_FAILED, f"{type(exc).__name__}: {exc}"
    tail = ((out.decode(errors="replace") if out else "") + (err.decode(errors="replace") if err else ""))[-400:]
    if proc.returncode != 0:
        _cleanup_probe(out_path, probe_dir)
        detail = f"benchmark exited {proc.returncode}"
        if tail:
            detail += f": {tail}"
        return PROBE_FAILED, detail

    replays, shard_error = _read_graph_probe_shards(
        out_path,
        expected_world_size=require_ranks if require_ranks > 1 else None,
    )
    _cleanup_probe(out_path, probe_dir)
    if shard_error:
        detail = shard_error
        if tail:
            detail += f": {tail}"
        # The reader's sentinel has to survive: a violated distributed contract
        # is a verdict on the driver, and collapsing it into PROBE_FAILED would
        # report it as "the probe did not work".
        return (replays if replays == PROBE_DISTRIBUTED_VIOLATION else PROBE_FAILED), detail
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
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=10)
        return False, "profile-run timed out"
    except asyncio.CancelledError:
        _kill_process_group(proc)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=10)
        raise
    output = (out.decode(errors="replace") if out else "") + (err.decode(errors="replace") if err else "")
    if proc.returncode != 0:
        return False, f"profile-run exited {proc.returncode}: {output[-200:]}"
    return True, "verified"


@dataclass(frozen=True)
class DriverContract:
    """What a driver must satisfy: the correctness threshold, graph timing, a profile run, its ranks and cases."""

    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    require_graph: bool = False
    require_profile: bool = False
    require_ranks: int = 1
    expected_case_ids: list[str] | None = None


def _stage_seconds(since: float) -> float:
    return round(time.monotonic() - since, 3)


@dataclass(frozen=True)
class _GraphVerdict:
    """What the one graph probe run concluded: graph timing, and the rank contract it also observes."""

    graph_ok: bool
    ranks_ok: bool


@dataclass
class _PreflightWork:
    """Mutable accumulators for one preflight run; stages append reasons, details and diagnostics."""

    reasons: list
    details: dict
    diagnostics: dict
    bench_ok: bool = False

    async def check_correctness(self, driver: str, snr_threshold: float, deadline_unix: float) -> bool:
        stage_started = time.monotonic()
        try:
            cres = await test_correctness(
                driver_script=driver,
                driver_args=[],
                snr_threshold=snr_threshold,
                timeout_sec=int(_deadline_timeout(deadline_unix, PREFLIGHT_CORRECTNESS_TIMEOUT_S)),
            )
            keys = ("passed", "outcome", "snr_db", "allclose", "max_diff", "message")
            self.details["correctness"] = {k: cres.get(k) for k in keys}
            self.details["correctness"]["seconds"] = _stage_seconds(stage_started)
            has_metric = (
                cres.get("snr_db") is not None or cres.get("allclose") is not None or cres.get("max_diff") is not None
            )
            if has_metric:
                return True
            self.reasons.append(f"correctness mode produced no SNR/allclose metric ({cres.get('message')})")
            _record_stage_output(self.diagnostics, "correctness", cres)
        except Exception as exc:  # isolation boundary: one correctness-stage run of the authored driver
            log.warning("preflight correctness run raised: %s", exc, exc_info=True)
            self.reasons.append(f"correctness run raised {type(exc).__name__}: {exc}")
            self.diagnostics["correctness"] = "".join(traceback.format_exception(exc))[-DIAG_TAIL_CHARS:]
        return False

    async def check_bench(self, driver: str, warmup: int, iters: int, expected_case_ids, deadline_unix: float) -> bool:
        stage_started = time.monotonic()
        try:
            bres = await bench_wallclock(
                BenchRun(driver, driver_args=[], warmup_iters=warmup, bench_iters=iters),
                timeout_sec=int(_deadline_timeout(deadline_unix, PREFLIGHT_BENCH_TIMEOUT_S)),
            )
            reported_cases = bres.get("case_times") or {}
            self.details["bench"] = {k: bres.get(k) for k in ("success", "median_ms", "message")}
            self.details["bench"]["case_count"] = len(reported_cases)
            self.details["bench"]["seconds"] = _stage_seconds(stage_started)
            # The declared suite is the contract, in both directions.
            declared = set(expected_case_ids or ())
            missing_cases = sorted(declared - set(reported_cases))
            undeclared_cases = sorted(set(reported_cases) - declared) if declared else []
            self.details["bench"]["expected_case_count"] = len(expected_case_ids or ())
            self.details["bench"]["missing_cases"] = missing_cases
            self.details["bench"]["undeclared_cases"] = undeclared_cases
            if not bres.get("success") or not reported_cases:
                reason = (
                    "bench mode must produce an aggregate and case_ms timing for every "
                    f"suite case ({bres.get('message')})"
                )
            elif missing_cases:
                reason = (
                    "bench mode reported "
                    f"{len(reported_cases)} of the {len(expected_case_ids or ())} cases "
                    "this task declares; no case_ms line for "
                    f"{', '.join(missing_cases)} — print one line per declared case "
                    "using its CASE_ID verbatim"
                )
            elif undeclared_cases:
                reason = (
                    "bench mode reported case_ms for "
                    f"{', '.join(undeclared_cases)}, which this task does not declare; "
                    "measure the declared suite exactly, because every case printed "
                    "here is scored"
                )
            else:
                return True
            self.reasons.append(reason)
            _record_stage_output(self.diagnostics, "bench", bres)
        except Exception as exc:  # isolation boundary: one bench-stage run of the authored driver
            log.warning("preflight bench run raised: %s", exc, exc_info=True)
            self.reasons.append(f"bench run raised {type(exc).__name__}: {exc}")
            self.diagnostics["bench"] = "".join(traceback.format_exception(exc))[-DIAG_TAIL_CHARS:]
        return False

    async def check_graph(
        self, driver: str, warmup: int, iters: int, contract: DriverContract, deadline_unix: float
    ) -> _GraphVerdict:
        failed = _GraphVerdict(graph_ok=False, ranks_ok=contract.require_ranks <= 1)
        if not self.bench_ok:
            self.reasons.append("cannot verify graph timing because bench produced no timing")
            return failed
        stage_started = time.monotonic()
        replays, tail = await _count_graph_replays(
            driver,
            warmup,
            iters,
            timeout_sec=_deadline_timeout(deadline_unix, PREFLIGHT_GRAPH_TIMEOUT_S),
            require_ranks=contract.require_ranks,
        )
        self.details["graph"] = {
            "replays": replays,
            "required": iters,
            "seconds": _stage_seconds(stage_started),
        }
        if replays >= iters:
            return _GraphVerdict(graph_ok=True, ranks_ok=True)
        self.diagnostics["graph"] = tail
        if replays == PROBE_DISTRIBUTED_VIOLATION:
            # A verdict on the driver, not a failure of the probe, so it is
            # reported against the rank contract rather than graph timing.
            self.reasons.append(f"distributed measurement contract violated: {tail}")
        elif replays < 0:
            self.reasons.append(f"could not verify graph timing (probe failed): {tail[-160:]}")
        else:
            self.reasons.append(
                f"benchmark did not run under a CUDA/HIP graph (observed {replays} "
                f"graph replays over {iters} timed iterations): {tail[-160:]}"
            )
        return failed

    async def check_profile(self, driver: str, deadline_unix: float) -> bool:
        if not self.bench_ok:
            self.reasons.append("cannot verify profiling contract because bench produced no timing")
            return False
        stage_started = time.monotonic()
        profile_ok, profile_detail = await _check_profile_contract(
            driver,
            timeout_sec=_deadline_timeout(deadline_unix, PREFLIGHT_PROFILE_TIMEOUT_S),
        )
        self.details["profile"] = {
            "ok": profile_ok,
            "contract": profile_detail if profile_ok else "",
            "seconds": _stage_seconds(stage_started),
        }
        if not profile_ok:
            self.diagnostics["profile"] = profile_detail
            self.reasons.append(f"profiling contract failed ({profile_detail})")
        return profile_ok


async def _preflight_async(
    driver: str,
    contract: DriverContract,
    warmup: int,
    iters: int,
    *,
    deadline_unix: float = 0.0,
) -> PreflightResult:
    started = time.monotonic()
    if not driver or not Path(driver).is_file():
        return PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            graph_ok=not contract.require_graph,
            profile_ok=not contract.require_profile,
            ranks_ok=contract.require_ranks <= 1,
            reasons=[f"driver file not found: {driver}"],
        )

    work = _PreflightWork(reasons=[], details={}, diagnostics={})
    # Correctness: the driver must EMIT a parseable metric and not crash.
    correctness_ok = await work.check_correctness(driver, contract.snr_threshold, deadline_unix)
    # Benchmark: the driver must accept --warmup/--iters/--bench-mode and print per-iteration wall_ms or a single
    # median_ms/mean_ms aggregate.
    work.bench_ok = await work.check_bench(driver, warmup, iters, contract.expected_case_ids, deadline_unix)
    # Graph timing: required only for prepass-produced drivers. One probe run answers both stages: it counts
    # CUDAGraph replays and, for a task that declares several ranks, reports what the ranks actually did.
    graph = _GraphVerdict(graph_ok=True, ranks_ok=contract.require_ranks <= 1)
    if contract.require_graph:
        graph = await work.check_graph(driver, warmup, iters, contract, deadline_unix)
    elif contract.require_ranks > 1:
        # The rank observations ride on the graph probe, so there is nothing to
        # conclude from without it.
        work.reasons.append(
            f"cannot verify the {contract.require_ranks}-rank measurement contract because graph timing was not required"
        )
    if contract.require_ranks > 1:
        work.details["distributed"] = {"ok": graph.ranks_ok, "required_ranks": contract.require_ranks}
    profile_ok = True
    if contract.require_profile:
        profile_ok = await work.check_profile(driver, deadline_unix)
    ok = correctness_ok and work.bench_ok and graph.graph_ok and profile_ok and graph.ranks_ok
    return PreflightResult(
        ok=ok,
        correctness_ok=correctness_ok,
        bench_ok=work.bench_ok,
        graph_ok=graph.graph_ok,
        profile_ok=profile_ok,
        ranks_ok=graph.ranks_ok,
        reasons=work.reasons,
        details=work.details,
        diagnostics=work.diagnostics,
        duration_sec=_stage_seconds(started),
    )


def _kill_process_group(proc) -> None:
    """Kill the child and ALL its descendants (no orphans)."""
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    signalled = False
    if hasattr(os, "killpg"):
        # pid == pgid under start_new_session; survives the leader's death.
        try:
            os.killpg(pid, signal.SIGKILL)
            signalled = True
        except OSError:  # group may already be gone
            pass
    if not signalled:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            proc.kill()
