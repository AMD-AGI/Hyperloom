# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local measurement backend: one complete test on this machine, through the existing ``run_grid`` (design 4.4).

* a single-point test is ``run_grid`` with one variant: warmup round, measured round, optional accuracy round, teardown;
* one arm of a concurrency sweep boots the inference server once and measures the levels from high to low on it, the
  way ``conc_sweep`` does, stepping down a level when the boot fails;
* accuracy is the ``RUN_EVAL`` switch in the point's own YAML, which must say what the caller asked for.

Nothing in the old flow changes: this file only calls what ``orchestrator.actions.executors`` exports.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import socket as host
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    GridVariant,
    VariantResult,
    resolve_lifecycle_params,
    run_grid,
    teardown_lifecycle_server,
)
from hyperloom.orchestrator.actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
)

log = logging.getLogger(__name__)

__all__ = ["LocalMeasurementBackend", "TestCallRefused"]

#: ``VariantResult.error_class`` of a round the per-round cap reaped.
_TIMEOUT_CLASSES = frozenset({"magpie_timeout", "warmup_magpie_timeout"})
#: Return codes of rounds the run itself stopped, keyed by the class the result carries.
_STOP_RETURNCODES = {
    SESSION_TIME_EXHAUSTED_CLASS: SESSION_TIME_EXHAUSTED_RETURNCODE,
    ORCHESTRATOR_CANCELLED_CLASS: ORCHESTRATOR_CANCELLED_RETURNCODE,
}
_BOOT_ONLY_NOTE = "server_lifecycle_boot_only"
_OUTPUT_POLL_SEC = 5.0
_LOG_TAIL_CHARS = 64 * 1024
_ACCURACY = "accuracy"
_UTF8 = "utf-8"
_SUPPORTED_KINDS = ("test", _ACCURACY)

#: The old flow reads the process environment, so two tests in one process would see each other's settings.
_PROCESS_LOCK = threading.Lock()


class TestCallRefused(ValueError):
    """The call asks for something the local backend does not run, or contradicts itself."""

    __test__ = False


@dataclass
class _Run:
    """What one test has produced so far."""

    call: TestCall
    unset_envs: list[str]
    results: list[tuple[int, VariantResult]] = field(default_factory=list)
    stopped_rc: int | None = None
    notes: list[str] = field(default_factory=list)

    def add(self, point: int, result: VariantResult) -> bool:
        """Record one result; say whether the test must go on."""
        self.results.append((point, result))
        stop = _STOP_RETURNCODES.get(result.error_class)
        if stop is not None:
            self.stopped_rc = stop
        return stop is None


class LocalMeasurementBackend:
    """Runs a test on the machine this process is on; the runner executes this same code."""

    name = "local"

    def run_test(self, call: TestCall) -> TestOutcome:
        """Run ``call`` to the end and return its outcome; the inference server is reclaimed when this returns."""
        _validate(call)
        with _PROCESS_LOCK, _test_environment(call) as unset_envs:
            run = _Run(call, unset_envs)
            with _watch_output(call):
                asyncio.run(_sweep(run) if call.prep == "conc_sweep" else _single(run))
        _raise_on_timeout(run)
        return _outcome(run)


# --- validation ---------------------------------------------------------------------------------------------------


def _point_envs(point: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}
    return dict((config.get("benchmark") or {}).get("envs") or {})


def _validate(call: TestCall) -> None:
    if call.kind not in _SUPPORTED_KINDS:
        raise TestCallRefused(f"the local backend runs kinds {_SUPPORTED_KINDS}, not {call.kind!r}")
    if not call.points:
        raise TestCallRefused("a test needs at least one measurement point")
    if call.prep == "grid" and len(call.points) != 1:
        raise TestCallRefused(f"a single-point test has exactly one point, got {len(call.points)}")
    if call.kind == _ACCURACY and not call.accuracy:
        raise TestCallRefused("an accuracy test must ask for accuracy")
    _, dropped = filter_untrusted_env_mapping(call.env_overrides, allow_predicate=is_allowed_variant_env_key)
    if dropped:
        raise TestCallRefused(f"env_overrides may not set {sorted(dropped)}")
    for point in call.points:
        _check_run_eval(point, call.accuracy)
    if call.prep == "conc_sweep":
        _check_sweep_levels(call.points)


def _check_run_eval(point: Path, accuracy: bool) -> None:
    """The point must say, explicitly, whether it runs accuracy; an unset ``RUN_EVAL`` means on in the old flow."""
    raw = _point_envs(point).get("RUN_EVAL")
    if raw is None:
        raise TestCallRefused(f"{point} does not set RUN_EVAL; an unset value runs accuracy")
    if is_truthy(raw, default=True) != accuracy:
        raise TestCallRefused(f"{point} sets RUN_EVAL={raw!r} but the call asks for accuracy={accuracy}")


def _levels(points: Sequence[Path]) -> list[int]:
    return [int(_point_envs(point).get("CONC") or 0) for point in points]


def _check_sweep_levels(points: Sequence[Path]) -> None:
    levels = _levels(points)
    if any(level <= 0 for level in levels) or len(set(levels)) != len(levels):
        raise TestCallRefused(f"every sweep point names its own CONC, got {levels}")


# --- the process environment --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _test_environment(call: TestCall) -> Iterator[list[str]]:
    """Set the process environment ``run_grid`` reads for this test, and put it back afterwards.

    Yields the variables the test's rounds unset. The session directory is a fresh empty one, so the reference launch
    the old flow keeps there reads as empty whatever else is on this machine.
    """
    facts = call.session_facts
    session = Path(tempfile.mkdtemp(prefix="hl-session-"))
    settings = {
        "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR": str(session),
        "INFERENCE_OPTIMIZER_LEAK_ROOTS": str(call.cwd),
        "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC": str(int(call.timeout_sec)),
    }
    if call.silence_timeout_sec is not None:
        settings["INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC"] = str(call.silence_timeout_sec)
    if facts is not None and facts.inferencex_path:
        settings["INFERENCEX_PATH"] = facts.inferencex_path
    settings.update({str(key): str(value) for key, value in call.env_overrides.items()})
    saved = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield list(dict.fromkeys([*(facts.reference_unset_envs if facts else ()), *call.unset_envs]))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(session, ignore_errors=True)


@contextlib.contextmanager
def _watch_output(call: TestCall) -> Iterator[None]:
    """Call ``call.on_output`` whenever something under the output directory has been written since the last look."""
    if call.on_output is None:
        yield
        return
    stop = threading.Event()

    def newest() -> float:
        try:
            return max((p.stat().st_mtime for p in Path(call.output_dir).rglob("*") if p.is_file()), default=0.0)
        except OSError:
            return 0.0

    def watch() -> None:
        seen = newest()
        while not stop.wait(_OUTPUT_POLL_SEC):
            now = newest()
            if now > seen:
                seen = now
                call.on_output()

    thread = threading.Thread(target=watch, name="hl-output-watch", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_OUTPUT_POLL_SEC + 1)


# --- running ------------------------------------------------------------------------------------------------------


async def _grid(run: _Run, point: int, name: str, **kwargs: Any) -> VariantResult:
    """One ``run_grid`` call for the point at index ``point``."""
    call = run.call
    results = await run_grid(
        base_yaml_path=call.points[point],
        base_extra_args="",
        grid=[GridVariant(name, unset_envs=run.unset_envs)],
        output_root=Path(call.output_dir),
        magpie_python=call.magpie_python or None,
        result_dir=call.result_dir,
        session_deadline_sec=call.session_deadline_sec,
        serving_lease=call.serving_lease,
        **kwargs,
    )
    return results[0]


async def _single(run: _Run) -> None:
    warmup = run.call.rounds.warmup_rounds
    result = await _grid(run, 0, "test", warmup_before_measure=None if warmup is None else bool(warmup))
    run.add(0, result)


def _is_failure(result: VariantResult) -> bool:
    return result.status in {"failed", "skipped"}


class _Lifecycle:
    """The persistent server one sweep arm shares: where its pid files are and which port it listens on."""

    def __init__(self, run: _Run) -> None:
        pid_dir = Path(run.call.output_dir) / "server_arm"
        pid_dir.mkdir(parents=True, exist_ok=True)
        info = resolve_lifecycle_params(run.call.points[0])
        self.pid_dir = pid_dir
        self.eligible = bool(info.get("eligible"))
        self.port = int(info.get("port") or 0)
        self.framework = str(info.get("framework") or "")
        self.reason = str(info.get("reason") or "")

    def settings(self, *, cleanup: bool) -> dict[str, Any]:
        return {"cleanup": cleanup, "pid_dir": str(self.pid_dir), "port": self.port}

    def teardown(self) -> None:
        teardown_lifecycle_server(pid_dir=self.pid_dir, framework=self.framework, port=self.port)


async def _sweep(run: _Run) -> None:
    """One arm: boot once on the highest level that boots, measure every level on that server, tear it down."""
    order = sorted(range(len(run.call.points)), key=lambda i: -_levels(run.call.points)[i])
    lifecycle = _Lifecycle(run)
    if not lifecycle.eligible:
        run.notes.append(f"server restarted per level: {lifecycle.reason}")
        await _sweep_by_restart(run, order)
        return
    try:
        booted = await _boot(run, lifecycle, order)
        if run.stopped_rc is not None:
            return
        if booted is None:
            run.notes.append("every boot attempt failed; server restarted per level")
            await _sweep_by_restart(run, order)
            return
        await _reuse(run, lifecycle, order[booted:])
    finally:
        lifecycle.teardown()


async def _boot(run: _Run, lifecycle: _Lifecycle, order: list[int]) -> int | None:
    """Boot the server on the first level in ``order`` that boots (stepping down on failure); its position, or None."""
    for position, point in enumerate(order):
        boot = await _grid(
            run,
            point,
            f"c{_levels(run.call.points)[point]}",
            server_lifecycle=lifecycle.settings(cleanup=False),
            base_extra_envs={"MAGPIE_RUN_PHASE": "server"},
            warmup_before_measure=False,
            lifecycle_boot_only=True,
        )
        if not _is_failure(boot):
            return position
        if boot.error_class in _STOP_RETURNCODES:
            run.add(point, boot)
            return None
        lifecycle.teardown()
        run.notes.append(f"boot at {_levels(run.call.points)[point]} failed ({boot.error_class}); stepping down")
    return None


async def _reuse(run: _Run, lifecycle: _Lifecycle, order: list[int]) -> None:
    """Measure each level of ``order`` on the running server; the last one tears it down."""
    for position, point in enumerate(order):
        result = await _grid(
            run,
            point,
            f"c{_levels(run.call.points)[point]}",
            server_lifecycle=lifecycle.settings(cleanup=position == len(order) - 1),
            server_already_ready=True,
            warmup_before_measure=False,
        )
        if not run.add(point, result):
            return


async def _sweep_by_restart(run: _Run, order: list[int]) -> None:
    """The fallback ``conc_sweep`` has for a framework that cannot hold a server: one server per level."""
    for point in order:
        result = await _grid(run, point, f"c{_levels(run.call.points)[point]}", warmup_before_measure=False)
        if not run.add(point, result):
            return


# --- the outcome --------------------------------------------------------------------------------------------------


def _raise_on_timeout(run: _Run) -> None:
    for _, result in run.results:
        if result.error_class in _TIMEOUT_CLASSES:
            raise subprocess.TimeoutExpired(cmd=run.call.task_id, timeout=run.call.timeout_sec, output=result.error)


def _result_returncode(result: VariantResult) -> int:
    if result.returncode is not None:
        return int(result.returncode)
    stop = _STOP_RETURNCODES.get(result.error_class)
    if stop is not None:
        return stop
    return 1 if _is_failure(result) else 0


def _eval_returncode(result: VariantResult) -> int:
    """The accuracy pass's own exit code when the round recorded one, else the round's."""
    marker = Path(result.workspace or "") / "eval_returncode"
    try:
        return int(marker.read_text(encoding=_UTF8).strip())
    except (OSError, ValueError):
        return _result_returncode(result)


def _rounds(run: _Run, point: int, result: VariantResult) -> list[RoundOutcome]:
    workspace = Path(result.workspace) if result.workspace else Path(run.call.output_dir)
    rounds = []
    if any(w == "run_grid_warmup_discarded_first" for w in result.nonfatal_warnings):
        rounds.append(RoundOutcome(point, "warmup", workspace.parent / "warmup_round", 0))
    if result.note != _BOOT_ONLY_NOTE:
        rounds.append(RoundOutcome(point, "measure", workspace, _result_returncode(result)))
    if run.call.accuracy and not _is_failure(result):
        rounds.append(RoundOutcome(point, _ACCURACY, workspace, _eval_returncode(result)))
    return rounds


def _read_tail(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _LOG_TAIL_CHARS))
            return handle.read().decode(_UTF8, errors="replace")
    except OSError:
        return ""


def _sandbox_facts(call: TestCall) -> SandboxFacts:
    """What this machine can state about itself without asking a device; the rest stays unset, not guessed."""
    node = os.environ.get("NODE_NAME") or host.gethostname()
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    gpus = tuple(int(part) for part in visible.split(",") if part.strip().isdigit())
    return SandboxFacts(
        sandbox_id=f"local-{node}-{call.task_id}",
        node_name=node,
        gpu_count=len(gpus),
        gpu_ids=gpus,
        same_node=True,
        image_digest=call.environment.image_digest,
        backend="local",
    )


def _outcome(run: _Run) -> TestOutcome:
    from hyperloom.measure.failures import Disposition, classify_test_failure

    call = run.call
    rounds = [r for point, result in run.results for r in _rounds(run, point, result)]
    failed = [r for r in rounds if r.returncode != 0]
    returncode = failed[0].returncode if failed else 0
    stderr = "\n".join([*(result.error for _, result in run.results if result.error), *run.notes])
    logs = [stderr, *(_read_tail(result.server_log_path) for _, result in run.results if _is_failure(result))]
    verdict = classify_test_failure(returncode=returncode, logs=logs)
    summary = [
        {"point": p, "name": r.name, "status": r.status, "returncode": r.returncode, "error_class": r.error_class}
        for p, r in run.results
    ]
    failure_class = verdict.failure_class.value if verdict.disposition is Disposition.FAILED else None
    return TestOutcome(
        returncode=returncode,
        stdout=json.dumps(summary),
        stderr=stderr,
        rounds=tuple(rounds),
        sandbox=_sandbox_facts(call),
        request_id=call.task_id,
        failure_class=failure_class,
        failure_reason=verdict.failure_reason,
    )
