# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local measurement backend: one complete test on this machine, through the existing ``run_grid`` (design 4.4).

* a single-point test is ``run_grid`` with one variant: warmup round, measured round, optional accuracy round, teardown;
* one arm of a concurrency sweep boots the inference server once and measures the levels from high to low on it, the
  way ``conc_sweep`` does, stepping down a level when the boot fails;
* accuracy is the ``RUN_EVAL`` switch in the point's own YAML, which must say what the caller asked for.

Nothing in the old flow changes: this file only calls what ``orchestrator.actions.executors`` exports, and the pick of a
free port that its server lifecycle makes (``_server_lifecycle``).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.env import is_truthy
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure.attachments import AttachmentError, place_attachments, substitute
from hyperloom.measure.accuracy import read_accuracy
from hyperloom.measure.compile import CompiledEnvironment, LocalBlobStore
from hyperloom.measure.decode import DecodePassError, plan_pass, run_pass, vocab_size_of, write_outputs
from hyperloom.measure.farm import FarmError, build_farm, place_farm, server_python, verify_farm
from hyperloom.measure.profile_traces import PROFILE_DIR, collect_traces
from hyperloom.measure.repro import ReproError, find_case, run_case
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC
from hyperloom.sandbox.node import node_name
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.spec import Attachment
from hyperloom.measure.spec import EnvRoot as SpecEnvRoot
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    GridVariant,
    VariantResult,
    resolve_lifecycle_params,
    run_grid,
    teardown_lifecycle_server,
)
from hyperloom.orchestrator.actions.executors._server_lifecycle import REUSE_PORT_DEFAULT, _assign_free_port
from hyperloom.orchestrator.actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
)

log = logging.getLogger(__name__)

__all__ = ["LocalMeasurementBackend", "SUPPORTED_KINDS", "TestCallRefused"]

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
_PROFILE = "profile"
#: The ``/start_profile`` body of the benchmark client. It reads it from its own process environment, not from the
#: point's ``envs``; without it the client sends its built-in body, which captures one step per stage.
_PROFILE_BODY_ENV = "PROFILE_EXTRA_BODY"
_REPRO = "repro"
_GRID = "grid"
#: How long a repro case may run, and the share of what is left of the test's time it may take: a case that hangs must
#: end as its own failure, with time left to sign the result, not as a timeout of the whole test.
_CASE_TIMEOUT_SEC = 600.0
_CASE_TIME_SHARE = 0.8
#: The decode pass runs on a server of its own, booted after the measured round; the share of the test's time left it.
_DECODE_SERVER = "decode_server"
_DECODE_TIMEOUT_SEC = 900.0
_DECODE_WARNING = "decode_pass_failed"
_TEST = "test"
_DECODE_KINDS = (_TEST, _ACCURACY)
_SUPPORTED_KINDS = (_TEST, _ACCURACY, _PROFILE, _REPRO)
#: The kinds of ``TestCall`` this backend runs; a toolbox refuses any other at acceptance.
SUPPORTED_KINDS = _SUPPORTED_KINDS

#: The old flow reads the process environment, so two tests in one process would see each other's settings.
_PROCESS_LOCK = threading.Lock()

#: What a boot round pins in its round's environment, on top of the point. A boot round only brings a server up and
#: leaves it for the rounds that measure: Magpie runs the server phase and then a full client round whatever
#: ``MAGPIE_RUN_PHASE`` says (Magpie issue 130), so the client round has to be made small instead. It must not evaluate
#: (``RUN_EVAL``: 1319 questions on a GSM8K point) and it needs one request, not the point's ``NUM_PROMPTS``. Both are
#: in Magpie's ``CLIENT_ONLY_REUSE_ENV_KEYS``, the variables a persistent server's reuse check leaves out, so the
#: rounds that measure on that server see the same server. ``NUM_WARMUPS`` stays the point's: Magpie's InferenceX
#: scripts take their warmups from the concurrency, not from it, and it is not one of those keys. An AgentX point has no
#: evaluation round and no prompt count; its boot round is left as it is.
_BOOT_PHASE_ENVS = {"MAGPIE_RUN_PHASE": "server"}
_BOOT_CLIENT_ENVS = {"RUN_EVAL": "false", "NUM_PROMPTS": "1"}


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
    traceless: bool = False
    case_failed: bool = False
    started: float = field(default_factory=time.monotonic)
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
        with _PROCESS_LOCK, _rebuilt_environment(call) as call, _test_environment(call) as unset_envs:
            run = _Run(call, unset_envs)
            with _watch_output(call):
                asyncio.run(_runner_of(call)(run))
        _raise_on_timeout(run)
        if call.kind == _PROFILE:
            _gather_traces(run)
        return _outcome(run)


# --- validation ---------------------------------------------------------------------------------------------------


def _point_config(point: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}


def _point_envs(point: Path) -> dict[str, Any]:
    return dict((_point_config(point).get("benchmark") or {}).get("envs") or {})


def _validate(call: TestCall) -> None:
    if call.kind not in _SUPPORTED_KINDS:
        raise TestCallRefused(f"the local backend runs kinds {_SUPPORTED_KINDS}, not {call.kind!r}")
    if not call.points:
        raise TestCallRefused("a test needs at least one measurement point")
    if call.decode_pass and (call.kind not in _DECODE_KINDS or call.prep != _GRID):
        raise TestCallRefused("the decode pass belongs to a single-point test or accuracy test")
    if call.prep == _GRID and len(call.points) != 1:
        raise TestCallRefused(f"a single-point test has exactly one point, got {len(call.points)}")
    if call.kind == _ACCURACY and not call.accuracy:
        raise TestCallRefused("an accuracy test must ask for accuracy")
    if call.kind == _PROFILE:
        _validate_profile(call)
    if call.kind == _REPRO:
        _validate_repro(call)
    _, dropped = filter_untrusted_env_mapping(call.env_overrides, allow_predicate=is_allowed_variant_env_key)
    if dropped:
        raise TestCallRefused(f"env_overrides may not set {sorted(dropped)}")
    for point in call.points:
        _check_run_eval(point, call.accuracy)
    if call.prep == "conc_sweep":
        _check_sweep_levels(call.points)


def _validate_profile(call: TestCall) -> None:
    """A profile test is one profiled round of one point: no accuracy, no sweep, and the point must turn the profiler on.

    A point that does not would run a plain round and leave no trace, which the result would then be refused for. An
    AgentX point turns it on in its YAML alone, as the single-lineage AgentX profile does; the InferenceX client also
    needs ``PROFILE=1``.
    """
    if call.prep != _GRID:
        raise TestCallRefused("a profile test profiles one point; it is not a sweep")
    if call.accuracy:
        raise TestCallRefused("a profile test does not run accuracy")
    benchmark = _point_config(call.points[0]).get("benchmark") or {}
    torch_profiler = (benchmark.get("profiler") or {}).get("torch_profiler") or {}
    client_profiling = str((benchmark.get("envs") or {}).get("PROFILE", "")).strip() == "1"
    if torch_profiler.get("enabled") is not True or not (client_profiling or is_magpie_agentx(benchmark)):
        raise TestCallRefused(
            f"{call.points[0]} does not turn the torch profiler on (torch_profiler, and PROFILE=1 unless AgentX)"
        )


def _validate_repro(call: TestCall) -> None:
    """A repro test runs a case from its attachments against one server: one point, no accuracy."""
    if call.prep != _GRID:
        raise TestCallRefused("a repro test runs one point; it is not a sweep")
    if call.accuracy:
        raise TestCallRefused("a repro test does not run accuracy")
    if not call.attachments:
        raise TestCallRefused("a repro test needs the reproduction case as an attachment")


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


# --- the rebuilt environment --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _rebuilt_environment(call: TestCall) -> Iterator[TestCall]:
    """The call to run: when its environment is a compiled one with changes, the points name a rebuilt framework.

    The farm and the attachments are made the way the runner makes them, from the same blob store the compile wrote
    to, so a patched test measures the rebuilt framework here as it does in a sandbox. A test on the image as it is
    runs unchanged.
    """
    environment = call.environment
    if not isinstance(environment, CompiledEnvironment) or environment.is_base:
        yield call
        return
    work = Path(tempfile.mkdtemp(prefix="hl-farm-"))
    try:
        yield dataclasses.replace(call, points=_rebuilt_points(call, environment, work))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _rebuilt_points(call: TestCall, environment: CompiledEnvironment, work: Path) -> tuple[Path, ...]:
    store = LocalBlobStore(environment.blob_root)

    def read_blob(address: str) -> bytes:
        return store.get(address.removeprefix("sha256:"))

    try:
        placed = place_attachments(
            [Attachment(a.id, a.kind, f"sha256:{a.sha256}", "0" * 64) for a in environment.attachments],
            work / "attach",
            read_blob,
        )
        python = server_python(_point_config(call.points[0]), default=sys.executable)
        roots = [SpecEnvRoot.parse(root.to_document(), "environment.roots") for root in environment.roots]
        farm = build_farm(roots, work / "farm", python=python, read_blob=read_blob)
        points = []
        for index, point in enumerate(call.points):
            config = yaml.safe_load(substitute(Path(point).read_text(encoding=_UTF8), placed)) or {}
            target = work / f"point_{index}.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
            if farm is not None:
                verify_farm(farm, place_farm(target, farm), python=python, cwd=work)
            points.append(target)
    except (AttachmentError, FarmError, OSError, ValueError) as exc:
        raise TestCallRefused(f"the compiled environment cannot be rebuilt here: {exc}") from exc
    return tuple(points)


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
    if call.kind == _PROFILE and (body := _point_envs(call.points[0]).get(_PROFILE_BODY_ENV)):
        settings[_PROFILE_BODY_ENV] = str(body)
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


@contextlib.contextmanager
def _free_port_for(point: Path, server_lifecycle: dict[str, Any] | None) -> Iterator[Path]:
    """The point a round runs from: with a free port of its own in ``benchmark.envs.PORT``, unless a lifecycle pins one.

    A round the server lifecycle does not hold (a profile round, a framework that cannot hold a server, a warmup that
    is not planned) has no port assigned, and its server listens on the default: two of them on one machine collide on
    it. The InferenceX script and Magpie's AgentX server both read the port from ``benchmark.envs.PORT``, so it goes
    there, into a copy: the point itself is shared by every test that runs it. The port is picked where the server is
    launched, by the function the lifecycle picks its own with.
    """
    if server_lifecycle is not None:
        yield point  # ``inject_lifecycle`` pins PORT in every round the lifecycle builds
        return
    config = _point_config(point)
    benchmark = config.get("benchmark") or {}
    benchmark["envs"] = {**(benchmark.get("envs") or {}), "PORT": _assign_free_port(REUSE_PORT_DEFAULT)}
    config["benchmark"] = benchmark
    with tempfile.TemporaryDirectory(prefix="hl-port-") as work:
        pinned = Path(work) / Path(point).name
        pinned.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
        yield pinned


async def _grid(run: _Run, point: int, name: str, **kwargs: Any) -> VariantResult:
    """One ``run_grid`` call for the point at index ``point``."""
    call = run.call
    with _free_port_for(call.points[point], kwargs.get("server_lifecycle")) as base_yaml:
        results = await run_grid(
            base_yaml_path=base_yaml,
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
    # A profiled round is not a statistic: it runs once, with no warmup before it.
    warmup = 0 if run.call.kind == _PROFILE else run.call.rounds.warmup_rounds
    result = await _grid(run, 0, "test", warmup_before_measure=None if warmup is None else bool(warmup))
    if run.add(0, result) and run.call.decode_pass and not _is_failure(result):
        await _decode_pass(run, result)


async def _decode_pass(run: _Run, measured: VariantResult) -> None:
    """Decode the fixed prompt set on a server booted for it; a pass that cannot run is a warning on the measurement.

    ``run_grid`` boots, warms up, measures, evaluates and reclaims its server in one call, so the pass cannot share
    that server without changing the old flow; it boots another one the way a repro test does (same point, same
    recipe), after the measured round, and tears it down.
    """
    lifecycle = _Lifecycle(run)
    try:
        await _decode_on_new_server(run, lifecycle)
    except DecodePassError as exc:
        measured.nonfatal_warnings.append(f"{_DECODE_WARNING}:{exc}")
    except OSError as exc:
        measured.nonfatal_warnings.append(f"{_DECODE_WARNING}:write_failed_{type(exc).__name__}")
    finally:
        lifecycle.teardown()


def _positive_env(envs: dict[str, Any], name: str) -> int:
    try:
        return max(0, int(str(envs.get(name) or 0)))
    except ValueError:
        return 0


async def _decode_on_new_server(run: _Run, lifecycle: _Lifecycle) -> None:
    call = run.call
    benchmark = _point_config(call.points[0]).get("benchmark") or {}
    envs = dict(benchmark.get("envs") or {})
    plan = plan_pass(
        _positive_env(envs, "ISL"), _positive_env(envs, "OSL"), vocab_size_of(str(benchmark.get("model") or ""))
    )
    concurrency = _positive_env(envs, "CONC")
    if not lifecycle.eligible:
        raise DecodePassError("server_cannot_be_held_up")
    if concurrency < 1:
        raise DecodePassError("point_has_no_conc")
    boot = await _boot_server(run, lifecycle, 0, _DECODE_SERVER)
    if boot.error_class in _STOP_RETURNCODES:
        run.add(0, boot)  # the run itself was stopped: that ends the test, it is not a failed pass
    if _is_failure(boot):
        raise DecodePassError("server_did_not_boot")
    left = call.timeout_sec - (time.monotonic() - run.started)
    outputs = await asyncio.to_thread(
        run_pass,
        plan,
        base_url=f"http://127.0.0.1:{lifecycle.port}",
        model=str(benchmark.get("model") or ""),
        concurrency=concurrency,
        timeout_sec=max(1.0, min(_DECODE_TIMEOUT_SEC, left * _CASE_TIME_SHARE)),
    )
    write_outputs(Path(call.output_dir), plan, outputs)


def _runner_of(call: TestCall) -> Callable[[_Run], Awaitable[None]]:
    if call.kind == _REPRO:
        return _repro
    return _sweep if call.prep == "conc_sweep" else _single


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


def _boot_envs(point: Path) -> dict[str, str]:
    """The environment a boot round of ``point`` is run with: the server phase, and no evaluation and one request."""
    if is_magpie_agentx(_point_config(point).get("benchmark") or {}):
        return dict(_BOOT_PHASE_ENVS)
    return {**_BOOT_PHASE_ENVS, **_BOOT_CLIENT_ENVS}


async def _boot_server(run: _Run, lifecycle: _Lifecycle, point: int, name: str) -> VariantResult:
    """Start the persistent server for the point at index ``point`` and leave it running (no round is measured).

    The point is the one the measured rounds run from; only this round's environment says it is a boot (see
    ``_BOOT_CLIENT_ENVS``), so the validation of the point (``RUN_EVAL`` as the call asked) is untouched.
    """
    return await _grid(
        run,
        point,
        name,
        server_lifecycle=lifecycle.settings(cleanup=False),
        base_extra_envs=_boot_envs(run.call.points[point]),
        warmup_before_measure=False,
        lifecycle_boot_only=True,
    )


async def _boot(run: _Run, lifecycle: _Lifecycle, order: list[int]) -> int | None:
    """Boot the server on the first level in ``order`` that boots (stepping down on failure); its position, or None."""
    for position, point in enumerate(order):
        boot = await _boot_server(run, lifecycle, point, f"c{_levels(run.call.points)[point]}")
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


async def _repro(run: _Run) -> None:
    """Boot the server the way a test does, run the case against it, tear the server down.

    No warmup and no measured round: the server only has to be up. A server that does not boot ends the test as a
    failed test does; a case that cannot run to a verdict ends it as a failure of the measurement, never a verdict.
    """
    lifecycle = _Lifecycle(run)
    if not lifecycle.eligible:
        _fail_case(run, f"the server cannot be held up for the case ({lifecycle.reason})")
        return
    try:
        boot = await _boot_server(run, lifecycle, 0, "repro")
        run.add(0, boot)
        if not _is_failure(boot) and run.stopped_rc is None:
            await asyncio.to_thread(_run_case, run, lifecycle.port)
    finally:
        lifecycle.teardown()


def _fail_case(run: _Run, reason: str) -> None:
    run.case_failed = True
    run.notes.append(f"repro_case_error: {reason}")


def _run_case(run: _Run, port: int) -> None:
    call = run.call
    left = call.timeout_sec - (time.monotonic() - run.started)
    timeout = max(1.0, min(_CASE_TIMEOUT_SEC, left * _CASE_TIME_SHARE))
    try:
        script = find_case(Path(call.cwd) / "attach", [attachment.id for attachment in call.attachments])
        run_case(
            script,
            base_url=f"http://127.0.0.1:{port}",
            model=str((_point_config(call.points[0]).get("benchmark") or {}).get("model") or ""),
            timeout_sec=timeout,
            output_dir=Path(call.output_dir),
        )
    except ReproError as exc:
        _fail_case(run, str(exc))


def _gather_traces(run: _Run) -> None:
    """Move the profiled round's traces into ``<output_dir>/profile``, where the bundle and the analysis find them."""
    destination = Path(run.call.output_dir) / PROFILE_DIR
    ok = [result for _, result in run.results if result.workspace and not _is_failure(result)]
    moved = sum(collect_traces(Path(result.workspace), destination) for result in ok)
    if ok and not moved:
        run.traceless = True
        run.notes.append("profile_no_trace: the profiled round left no trace file")


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
        rounds.append(RoundOutcome(point, _ACCURACY, workspace, _eval_returncode(result), read_accuracy(workspace)))
    if rounds:
        rounds[0] = dataclasses.replace(rounds[0], nonfatal_warnings=tuple(result.nonfatal_warnings))
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


def _image_digest(call: TestCall) -> str | None:
    environment = call.environment
    return getattr(environment, "image_digest", None) or getattr(environment, "base_image_digest", None)


def _sandbox_facts(call: TestCall) -> SandboxFacts:
    """What this machine can state about itself without asking a device; the rest stays unset, not guessed."""
    node = node_name()
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    gpus = tuple(int(part) for part in visible.split(",") if part.strip().isdigit())
    return SandboxFacts(
        sandbox_id=f"local-{node}-{call.task_id}",
        node_name=node,
        gpu_count=len(gpus),
        gpu_ids=gpus,
        same_node=True,
        image_digest=_image_digest(call),
        backend="local",
    )


def _clean_returncode(run: _Run) -> int:
    """The return code of a test whose rounds all passed: a case that could not run, or a trace-less profile, is not 0."""
    return HOSTED_INFRA_FAILURE_RC if run.case_failed else int(run.traceless)


def _outcome(run: _Run) -> TestOutcome:
    from hyperloom.measure.failures import Disposition, classify_test_failure

    call = run.call
    rounds = [r for point, result in run.results for r in _rounds(run, point, result)]
    failed = [r for r in rounds if r.returncode != 0]
    returncode = failed[0].returncode if failed else _clean_returncode(run)
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
