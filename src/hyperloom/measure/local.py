# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local measurement backend: one complete test on this machine, through the existing ``run_grid``.

* a single-point test runs on one server the lifecycle holds up, one ``run_grid`` call per round, in one order whatever
  the test: the warmup bench (none for a point measured once, an AgentX replay), the measured bench, then, for a graded
  test (accuracy asked for), the evaluation in a round of its own that runs no bench, which the accuracy is read from.
  No bench evaluates, in a sweep arm either;
* a single-point test whose server the lifecycle cannot hold (a profile, a framework it cannot hold) is ``run_grid``
  with one variant: measured round, the evaluation after its bench when the point asks for one, teardown;
* one arm of a concurrency sweep boots the inference server once and benches the levels from high to low on it, the
  way ``conc_sweep`` does; a graded arm then evaluates once on that server. An arm whose levels must not share a
  server (``BenchmarkPoint.sweeps_on_cold_servers``), or whose server the lifecycle cannot hold, boots one per level;
* accuracy is the ``RUN_EVAL`` switch in the point's own YAML, which must say what the caller asked for.

Nothing in the old flow changes: this file only calls what ``orchestrator.actions.executors`` exports.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from inference_suite.contracts import FailedStep, RunLayout

from hyperloom.common.benchmark_config import BenchmarkPoint, BenchmarkPointError, read_benchmark_point
from hyperloom_common.env import is_truthy
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.common.outside_facts import (
    ENV_CONC,
    ENV_EVAL_ONLY,
    ENV_NUM_PROMPTS,
    ENV_RUN_EVAL,
    ENV_TP,
    ROCR_VISIBLE_DEVICES_ENV,
)
from hyperloom.contracts.measurement import PhaseTiming, SandboxFacts
from hyperloom.measure.attachments import AttachmentError, place_attachments, substitute
from hyperloom.measure.accuracy import read_accuracy
from hyperloom.measure import log_causes
from hyperloom.measure.boot_watch import BOOT_CANCEL_REASON, BootGuard, boot_limit_sec
from hyperloom.measure.compile import CompiledEnvironment, LocalBlobStore
from hyperloom.measure.jit.native.config import store_from_environment
from hyperloom.measure.jit.native.laying import NamedBuild, NativeLayError, SheetBuilds, sheet_builds
from hyperloom.measure.failures import (
    Disposition,
    FailureClassification,
    FailureEvidence,
    StepFailure,
    classify_test_failure,
    decides_test,
)
from hyperloom.measure.farm import (
    FarmError,
    build_farm,
    built_roots,
    inherited_environment,
    lay_profile_patch,
    patches_aiter_sources,
    place_farm,
    place_host_copies,
    record_unpatched,
    server_python,
    verify_farm,
)
from hyperloom.measure.framework.copies import PrivateFramework, private_framework
from hyperloom.measure.framework.rows import RowsOnlyRefused
from hyperloom.measure.long_context.held import (
    HeldAnswers,
    HeldChecks,
    HeldServer,
    long_context_unheld,
    not_held,
    run_held_checks,
    served_by,
)
from hyperloom.measure.long_context import staging as long_context_staging
from hyperloom.measure.phases import (
    JIT_CACHE,
    KERNEL_TABLES,
    OTHER,
    REPRO_CASE,
    ROUND_COLLECT,
    TEARDOWN,
    PhaseRecorder,
)
from hyperloom.measure.points import AMBIENT_SHAPE_ENVS, launch_carriers
from hyperloom.common.ports import pick_server_port
from hyperloom.measure.profile_traces import PROFILE_DIR, collect_traces, round_has_traces
from hyperloom.measure.repro import ReproError, find_case, run_case
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC, SERVER_BOOT_TIMEOUT_RC
from hyperloom.measure import ruppell
from hyperloom.measure.round_marks import ROLE_BOOT, ROLE_EVAL, ROLE_MEASURE, ROLE_WARMUP, RoundShape
from hyperloom.measure.round_marks import read_runs, round_boundaries
from hyperloom.sandbox.node import node_name
from hyperloom.measure.short_prompts.rounds import Serving, write_answered
from hyperloom.measure.short_prompts.staging import load_prompt_set
from hyperloom.measure.short_prompts.prompts import PromptSetError
from hyperloom.measure.socket import MeasurementTimeout, RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.spec import Attachment
from hyperloom.measure.spec import EnvRoot as SpecEnvRoot
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    RoundFolder,
    GridLaunch,
    GridRecipe,
    GridStop,
    GridVariant,
    LaunchLayer,
    VariantResult,
    is_round_folder,
    resolve_lifecycle_params,
    run_grid,
    run_grid_warmup_enabled,
    teardown_variant_server,
)
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, current_cancel_scope, use_cancel_scope
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS
from hyperloom.orchestrator.state.base_launch import freeze_base

__all__ = ["LocalMeasurementBackend", "SUPPORTED_KINDS", "TestCallRefused"]

#: ``VariantResult.error_class`` of a round the per-round cap reaped.
_TIMEOUT_CLASSES = frozenset({FailedStep.ROUND_TIMEOUT.value})
#: Return codes of rounds the run itself stopped, keyed by the class the result carries.
_STOP_RETURNCODES = {
    SESSION_TIME_EXHAUSTED_CLASS: SESSION_TIME_EXHAUSTED_RETURNCODE,
    ORCHESTRATOR_CANCELLED_CLASS: ORCHESTRATOR_CANCELLED_RETURNCODE,
}
_BOOT_ONLY_NOTE = "server_lifecycle_boot_only"
_OUTPUT_POLL_SEC = 5.0
_LOG_TAIL_CHARS = 64 * 1024
#: How far back a log is searched for the refusal of an evaluation request, in blocks that overlap by a line's length.
_MIB = 1 << 20
_REFUSAL_SCAN_CHARS = 16 * _MIB
_REFUSAL_BLOCK_CHARS = _MIB
_REFUSAL_BLOCK_OVERLAP_CHARS = 4096
_BENCHMARK_STDERR = "benchmark_stderr.log"
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
_TEST = "test"
#: The kinds of ``TestCall`` this backend runs; a toolbox refuses any other at acceptance.
SUPPORTED_KINDS = (_TEST, _ACCURACY, _PROFILE, _REPRO)
#: The kinds of test whose compiled code another test may be given: a profile or a repro ran under other settings.
_CONTRIBUTING = frozenset({_TEST, _ACCURACY})

#: The old flow reads the process environment, so two tests in one process would see each other's settings.
_PROCESS_LOCK = threading.Lock()

#: What a boot round pins in its round's environment, on top of the point. A boot round only brings a server up for the
#: rounds that measure, and pins it small in case it runs a load: no evaluation (``RUN_EVAL``: 1319 questions on a
#: GSM8K point) and one request, not the point's ``NUM_PROMPTS``. Both are client-only knobs, so the rounds that measure
#: reuse the server the boot round started. ``NUM_WARMUPS`` stays the point's. An AgentX point has no evaluation round
#: and no prompt count; its boot round pins nothing.
_BOOT_CLIENT_ENVS = {ENV_RUN_EVAL: "false", ENV_NUM_PROMPTS: "1"}
_WARMUP, _EVAL = ROLE_WARMUP, ROLE_EVAL  # a held test's rounds other than the measured one are named for their roles
#: No bench of a graded test evaluates, although its point does. ``RUN_EVAL`` is a client-only knob, so the server the
#: first round booted is the one each later round reuses.
_BENCH_ENVS = {ENV_RUN_EVAL: "false"}
#: The evaluation round runs no bench: InferenceX's ``run_benchmark_serving`` returns before it on ``EVAL_ONLY``, a
#: client-only knob (an ``EVAL_*`` name).
_EVAL_ENVS = {ENV_RUN_EVAL: "true", ENV_EVAL_ONLY: "true"}
#: What ``run_grid`` writes on a point whose warmup round it discarded, with the throughput of that round.
_WARMUP_DISCARDED = "run_grid_warmup_discarded_first"
#: What a test that failed records in place of the short-prompt check it would have run.
_CHECK_SKIPPED = "short_prompt_check skipped: test failed"
_LONG_CONTEXT_SKIPPED = "long_context_check skipped: test failed"
_NOT_HELD = "the test's server was not held up for the check (its framework's lifecycle cannot hold one)"
_NOT_CHECKED = "the test's server was not held up for the check (an AgentX test runs no accuracy check)"
_WARMUP_TPUT = "warmup_round_tput"


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
    #: The warmup and evaluation rounds of a test on a held server, which evaluates apart from its benches.
    warmup: VariantResult | None = None
    evaluation: VariantResult | None = None
    evaluates_apart: bool = False
    traceless: bool = False
    case_failed: bool = False
    started: float = field(default_factory=time.monotonic)
    notes: list[str] = field(default_factory=list)
    #: Ends a round whose server is not ready in time; ``None`` when the caller owns the cancellation.
    boot: BootGuard | None = None
    #: Where the test's phases are stamped: the caller's recorder when the call carries one, else the backend's own.
    phases: PhaseRecorder = field(default_factory=PhaseRecorder)
    #: What the accuracy checks left on the test's own server before its teardown; ``None`` until they ran there.
    held: HeldAnswers | None = None
    #: The ruppell that traces the test's server (a profile's); ``None`` when it runs untraced.
    traced: ruppell.Tracer | None = None

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
        phases = PhaseRecorder() if call.phases is None else call.phases
        notes: list[str] = []
        tracer = None
        try:
            with _PROCESS_LOCK:
                call = _seeded(call, phases)
                phases.begin(KERNEL_TABLES)
                tracer = _tracer(call, notes)
                run = _measure(call, phases, tracer, notes)
                failed = _traced_attempt_failed(run)
                if failed:
                    notes.append(f"ruppell: attempt failed ({failed}), profiled again without it")
                    _set_aside(run)
                    run = _measure(call, phases, None, notes)
            _raise_on_timeout(run)
            if run.call.kind == _PROFILE:
                _gather_traces(run)
            if run.traced is not None:
                notes.append(run.traced.report(Path(run.call.output_dir), _ranks(run.call)))
                if run.traced.packet_capture_off:
                    notes.append(ruppell.KINETO_NOTE)
        finally:
            if tracer is not None:
                tracer.close()
        return _staged(dataclasses.replace(_outcome(run), notes=tuple(notes)), call, phases)


def _seeded(call: TestCall, phases: PhaseRecorder) -> TestCall:
    """The call with the shared compiled-code cache seeded for it: the test's own cache directories are filled from the
    verified snapshots and the server is pointed at them."""
    if call.jit is None:
        return call
    phases.begin(JIT_CACHE)
    call.jit.seed()
    return dataclasses.replace(call, env_overrides={**call.env_overrides, **call.jit.environment()})


def _staged(outcome: TestOutcome, call: TestCall, phases: PhaseRecorder) -> TestOutcome:
    """Stage what the compilers added when the test succeeded and its kind may contribute; the outcome is unchanged."""
    if (
        call.jit is not None
        and outcome.returncode == 0
        and outcome.failure_class is None
        and call.kind in _CONTRIBUTING
    ):
        phases.begin(JIT_CACHE)
        call.jit.collect()
    return outcome


def _measure(call: TestCall, phases: PhaseRecorder, tracer: ruppell.Tracer | None, notes: list[str]) -> _Run:
    """Lay out the test's environment, run its rounds and give what they produced; ``tracer`` traces the server of a
    profile, when it can be given the points (a note says why not)."""
    with (
        _rebuilt_environment(call, notes) as call,
        _framework_tables(call) as private,
        _patched_server(private.call) as call,
    ):
        with _test_environment(call) as unset_envs:
            call, tracer = _traced(call, tracer, notes)
            run = _Run(call, [name for name in unset_envs if name not in private.redirected], phases=phases)
            run.traced = tracer
            with _watch_output(call), _guard_boot(run):
                asyncio.run(_runner_of(call)(run))
    return run


@contextlib.contextmanager
def _framework_tables(call: TestCall) -> Iterator[PrivateFramework]:
    """``private_framework``, with a table it cannot lay out refused the way the backend refuses any call it cannot run."""
    try:
        with private_framework(call) as private:
            yield private
    except RowsOnlyRefused as exc:
        raise TestCallRefused(str(exc)) from exc


# --- validation ---------------------------------------------------------------------------------------------------


def _point_config(point: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}


def _point(point: Path) -> BenchmarkPoint:
    try:
        return read_benchmark_point(Path(point))
    except BenchmarkPointError as exc:
        raise TestCallRefused(str(exc)) from exc


def _validate(call: TestCall) -> None:
    if call.kind not in SUPPORTED_KINDS:
        raise TestCallRefused(f"the local backend runs kinds {SUPPORTED_KINDS}, not {call.kind!r}")
    if not call.points:
        raise TestCallRefused("a test needs at least one measurement point")
    if call.prep == _GRID and len(call.points) != 1:
        raise TestCallRefused(f"a single-point test has exactly one point, got {len(call.points)}")
    if call.kind == _ACCURACY and not call.accuracy:
        raise TestCallRefused("an accuracy test must ask for accuracy")
    if call.kind == _PROFILE:
        _validate_profile(call)
    if call.kind == _REPRO:
        _validate_repro(call)
    _validate_checks(call)
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
    point = _point(call.points[0])
    client_profiling = str(point.envs.get("PROFILE", "")).strip() == "1"
    if not point.torch_profiler or not (client_profiling or point.agentx):
        raise TestCallRefused(
            f"{call.points[0]} does not turn the torch profiler on (torch_profiler, and PROFILE=1 unless AgentX)"
        )


def _validate_checks(call: TestCall) -> None:
    """An accuracy check needs a test that measures and a staged prompt set that is the one the call names; a test that
    would only find out after its rounds is refused before any server starts."""
    checks = (
        (call.short_prompts, "short-prompt consistency check", "short-prompt set", load_prompt_set),
        (call.long_context, "long-context check", "long-context set", long_context_staging.load_prompt_set),
    )
    for plan, check, prompt_set, load in checks:
        if plan is None:
            continue
        if call.kind in (_PROFILE, _REPRO):
            raise TestCallRefused(f"a {call.kind} test does not run the {check}")
        try:
            load(plan, os.environ)  # type: ignore[arg-type]
        except PromptSetError as exc:
            raise TestCallRefused(f"the {prompt_set} {plan.prompt_set} cannot be used: {exc}") from exc


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
    raw = _point(point).envs.get(ENV_RUN_EVAL)
    if raw is None:
        raise TestCallRefused(f"{point} does not set RUN_EVAL; an unset value runs accuracy")
    if is_truthy(raw, default=True) != accuracy:
        raise TestCallRefused(f"{point} sets RUN_EVAL={raw!r} but the call asks for accuracy={accuracy}")


def _levels(points: Sequence[Path]) -> list[int]:
    return [int(_point(point).envs.get(ENV_CONC) or 0) for point in points]


def _check_sweep_levels(points: Sequence[Path]) -> None:
    levels = _levels(points)
    if any(level <= 0 for level in levels) or len(set(levels)) != len(levels):
        raise TestCallRefused(f"every sweep point names its own CONC, got {levels}")


@contextlib.contextmanager
def _patched_server(call: TestCall) -> Iterator[TestCall]:
    """The call whose profile server imports its framework with TraceLens's patch from this host's copy (a point may
    have been rendered on another host), refused when a point asks for what only the patch adds and none can be laid."""
    if call.kind != _PROFILE:
        yield call
        return
    work = Path(tempfile.mkdtemp(prefix="hl-profile-"))
    try:
        try:
            points = place_host_copies(call.points, work)
        except FarmError as exc:
            raise TestCallRefused(str(exc)) from exc
        yield dataclasses.replace(call, points=points)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- the rebuilt environment --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _rebuilt_environment(call: TestCall, notes: list[str]) -> Iterator[TestCall]:
    """The call to run: when its environment is a compiled one with changes, the points name a rebuilt framework.

    The farm and the attachments are made the way the runner makes them, from the same blob store the compile wrote
    to, so a patched test measures the rebuilt framework here as it does in a sandbox. A test on the image as it is
    runs unchanged. A profile whose patch cannot be laid in the rebuilt framework says so in ``notes`` and its points.
    """
    environment = call.environment
    if not isinstance(environment, CompiledEnvironment) or environment.is_base:
        yield call
        return
    work = Path(tempfile.mkdtemp(prefix="hl-farm-"))
    try:
        yield _rebuilt_call(call, environment, work, notes)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _rebuilt_call(call: TestCall, environment: CompiledEnvironment, work: Path, notes: list[str]) -> TestCall:
    if call.jit is None and patches_aiter_sources(environment.roots):
        raise TestCallRefused(
            "the patch changes aiter's csrc or hsa, which is measured only by building the modules it touches in the "
            "test's own JIT directory (a runner's JIT session), and this call has none: the image's prebuilt modules "
            "would run instead"
        )
    store = LocalBlobStore(environment.blob_root)

    def read_blob(address: str) -> bytes:
        return store.get(address.removeprefix("sha256:"))

    try:
        attachments = [Attachment(a.id, a.kind, f"sha256:{a.sha256}", "0" * 64) for a in environment.attachments]
        placed = place_attachments(attachments, work / "attach", read_blob)
        python = server_python(_point_config(call.points[0]), default=sys.executable)
        roots = [SpecEnvRoot.parse(root.to_document(), "environment.roots") for root in environment.roots]
        inherited = inherited_environment(os.environ, call.env_overrides, call.unset_envs)
        roots = built_roots(roots, _builds(environment))
        farm = build_farm(roots, work / "farm", python=python, read_blob=read_blob, inherited=inherited)
        unpatched = lay_profile_patch(farm, _point_config(call.points[0])) if call.kind == _PROFILE else None
        if unpatched and unpatched not in notes:
            notes.append(unpatched)
        points = []
        for index, point in enumerate(call.points):
            config = yaml.safe_load(substitute(Path(point).read_text(encoding=_UTF8), placed)) or {}
            target = work / f"point_{index}.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
            if unpatched:
                record_unpatched(target)
            if farm is not None:
                verify_farm(farm, place_farm(target, farm), python=python, cwd=work)
            points.append(target)
    except (AttachmentError, FarmError, NativeLayError, OSError, ValueError) as exc:
        raise TestCallRefused(f"the compiled environment cannot be rebuilt here: {exc}") from exc
    return dataclasses.replace(call, points=tuple(points), framework_changed=farm is not None)


def _builds(environment: CompiledEnvironment) -> SheetBuilds | None:
    """The ready builds of the environment's runtime roots, read from the store this process's environment names."""
    named = tuple(NamedBuild(r.root_id, r.key, r.manifest_sha256) for r in environment.build_refs)
    image = environment.base_image_digest.rsplit("@", 1)[-1]
    return sheet_builds(named, store_from_environment(os.environ) if named else None, image)


# --- the process environment --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _test_environment(call: TestCall) -> Iterator[list[str]]:
    """Set the process environment ``run_grid`` reads for this test, and put it back afterwards; yields the variables
    the test's rounds unset. The session directory is a fresh one holding only the base the call carries, frozen as a
    session freezes it, so the rounds never read the asking application's session."""
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
    if call.kind == _PROFILE and (body := _point(call.points[0]).envs.get(_PROFILE_BODY_ENV)):
        settings[_PROFILE_BODY_ENV] = str(body)
    settings.update({str(key): str(value) for key, value in call.env_overrides.items()})
    saved = {key: os.environ.get(key) for key in (*AMBIENT_SHAPE_ENVS, *settings)}
    for key in AMBIENT_SHAPE_ENVS:
        os.environ.pop(key, None)
    os.environ.update(settings)
    try:
        if facts is not None and facts.base_launch is not None:
            freeze_base(session, facts.base_launch)
        yield list(dict.fromkeys([*(facts.reference_unset_envs if facts else ()), *call.unset_envs]))
    finally:
        os.environ.update({key: value for key, value in saved.items() if value is not None})
        for key in [key for key, value in saved.items() if value is None]:
            os.environ.pop(key, None)
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


@contextlib.contextmanager
def _guard_boot(run: _Run) -> Iterator[None]:
    """Publish the cancel scope a round's boot limit cancels (the call's, when it carries one), unless the caller already
    published its own."""
    if current_cancel_scope() is not None:
        yield
        return
    scope = run.call.cancel or CancelScope()
    run.boot = BootGuard(Path(run.call.output_dir), scope)
    with use_cancel_scope(scope):
        yield


def _booting(run: _Run, point: int, round_args: dict[str, Any]) -> contextlib.AbstractContextManager[None]:
    """Watch the boot of the round about to run, unless it reuses a ready server or no ready stamp can be expected."""
    call = run.call
    if run.boot is None or round_args.get("server_already_ready") or call.serving_lease is not None:
        return contextlib.nullcontext()
    limit = boot_limit_sec(_point(call.points[point]), call.timeout_sec)
    return contextlib.nullcontext() if limit is None else run.boot.watching(limit)


# --- running ------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _free_port_for(point: Path, server_lifecycle: dict[str, Any] | None) -> Iterator[Path]:
    """The point a round runs from: with a free port of its own in ``benchmark.envs.PORT``, unless a lifecycle pins one.

    A round the server lifecycle does not hold (a profile round, a framework that cannot hold a server, a warmup that
    is not planned) has no port assigned, and its server listens on the default: two of them on one machine collide on
    it. The InferenceX script and the AgentX server both read the port from ``benchmark.envs.PORT``, so it goes
    there, into a copy: the point itself is shared by every test that runs it. The port is picked where the server is
    launched (:func:`~hyperloom.common.ports.pick_server_port`).
    """
    if server_lifecycle is not None:
        yield point  # ``inject_lifecycle`` pins PORT in every round the lifecycle builds
        return
    config = _point_config(point)
    benchmark = config.get("benchmark") or {}
    benchmark["envs"] = {**(benchmark.get("envs") or {}), "PORT": pick_server_port()}
    config["benchmark"] = benchmark
    with tempfile.TemporaryDirectory(prefix="hl-port-") as work:
        pinned = Path(work) / Path(point).name
        pinned.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
        yield pinned


async def _grid(run: _Run, point: int, name: str, **launch: Any) -> VariantResult:
    """One ``run_grid`` call for the point at index ``point``: ``launch`` its GridLaunch, and its base_extra_envs."""
    call, envs = run.call, launch.pop("base_extra_envs", None)
    launch.setdefault("warmup_round", name == _WARMUP)  # the held warmup draws its own prompts (_held_rounds)
    with _free_port_for(call.points[point], launch.get("server_lifecycle")) as base_yaml, _booting(run, point, launch):
        started = time.time()
        results = await run_grid(
            grid=[GridVariant(name, unset_envs=run.unset_envs)],
            output_root=Path(call.output_dir),
            recipe=GridRecipe(base_yaml, base=LaunchLayer(extra_envs=envs)),
            launch=GridLaunch(call.result_dir, call.serving_lease, **launch),
            stop=GridStop(call.session_deadline_sec, checkpoint=call.checkpoint and call.checkpoint.on(name == _TEST)),
        )
        ended = time.time()
    _stamp_round(run, point, _round_shape(run, point, name, launch, results[0]), (started, ended))
    return results[0]


def _round_shape(run: _Run, point: int, name: str, kwargs: dict[str, Any], result: VariantResult) -> RoundShape:
    """What a round's call says about it, for reading the phases off the marks it left."""
    lifecycle = kwargs.get("server_lifecycle")
    role = ROLE_BOOT if kwargs.get("lifecycle_boot_only") else name if name in (_WARMUP, _EVAL) else ROLE_MEASURE
    stays_up = lifecycle is not None and not lifecycle.get("cleanup", True)
    return RoundShape(
        role,
        _point(run.call.points[point]).agentx,
        bool(kwargs.get("server_already_ready")),
        _is_failure(result),
        ROUND_COLLECT if stays_up else TEARDOWN,
    )


def _stamp_round(run: _Run, point: int, shape: RoundShape, window: tuple[float, float]) -> None:
    """Stamp the phases of the round that ran in ``window``; what follows it is ``other`` until something stamps."""
    for boundary in round_boundaries(window, read_runs(Path(run.call.output_dir), window), shape):
        run.phases.begin(boundary.name, at=boundary.at, point=point)
    run.phases.begin(OTHER, at=window[1])


async def _single(run: _Run) -> None:
    # A profiled round is not a statistic: it runs once, with no warmup before it.
    once = run.call.kind == _PROFILE or _point(run.call.points[0]).measured_once
    warmup = 0 if once else run.call.rounds.warmup_rounds
    lifecycle = _Lifecycle(run)
    if lifecycle.eligible:
        await _held_rounds(run, lifecycle, _warms(warmup))
    else:
        run.add(0, await _grid(run, 0, _TEST, warmup_before_measure=None if warmup is None else bool(warmup)))


def _warms(warmup: int | None) -> bool:
    return run_grid_warmup_enabled() if warmup is None else bool(warmup)


def _checks_held(call: TestCall) -> bool:
    """Whether the test runs an accuracy check, which asks the test's own server before its teardown; a point that holds
    none (``BenchmarkPoint.holds_accuracy_checks``) has its planned checks recorded as not held."""
    planned = call.short_prompts is not None or call.long_context is not None
    return planned and _point(call.points[0]).holds_accuracy_checks


async def _held_rounds(run: _Run, lifecycle: _Lifecycle, warm: bool) -> None:
    """The warmup bench when ``warm``, the measured bench and, for a graded test, the evaluation (``_evaluate``), on one
    server; then the accuracy checks on it (``_checks_held``). A bench that followed the evaluation started on the KV pool it left full, and
    read lower. The warmup draws prompts of its own (``_grid``); when it failed, was stopped or its logs name a cause
    that settles the test (a crashed server, a refused request), nothing is measured."""
    run.evaluates_apart = True
    bench_envs = _BENCH_ENVS if run.call.accuracy else None  # an explore's point does not evaluate
    try:
        if warm and not await _warmup_round(run, lifecycle, bench_envs):
            return
        measured = await _grid(
            run,
            0,
            _TEST,
            server_lifecycle=lifecycle.settings(cleanup=not (run.call.accuracy or _checks_held(run.call))),
            server_already_ready=warm,
            base_extra_envs=bench_envs,
            warmup_before_measure=False,
        )
        run.add(0, measured)
        await _evaluate(run, lifecycle, 0)
        await _held_checks(run, lifecycle)
    finally:
        lifecycle.teardown()


async def _warmup_round(run: _Run, lifecycle: _Lifecycle, envs: dict[str, str] | None) -> bool:
    """The warmup bench on the server it leaves up; whether the measured round follows."""
    held = lifecycle.settings(cleanup=False)
    warmup = await _grid(run, 0, _WARMUP, server_lifecycle=held, base_extra_envs=envs, warmup_before_measure=False)
    run.warmup = warmup
    return run.add(0, warmup) and not _is_failure(warmup) and not decides_test(_ended(run).verdict)


async def _evaluate(run: _Run, lifecycle: _Lifecycle, point: int) -> None:
    """A graded test's evaluation from the point at index ``point``, after its last bench, on the server it left up;
    none after a failed bench or a stop. One that fails fails the test's accuracy, never a bench's reading."""
    if run.call.accuracy and run.stopped_rc is None and not _test_failed(run):
        held = lifecycle.settings(cleanup=not _checks_held(run.call))
        launch = {"server_already_ready": True, "warmup_before_measure": False, "evaluation_only": True}
        run.evaluation = await _grid(run, point, _EVAL, server_lifecycle=held, base_extra_envs=_EVAL_ENVS, **launch)
        run.add(point, run.evaluation)


async def _held_checks(run: _Run, lifecycle: _Lifecycle) -> None:
    """The checks the test plans, on its own server after its last round; none for a test that failed or is stopping."""
    call = run.call
    if not _checks_held(call) or run.stopped_rc is not None or _test_failed(run):
        return
    serving = Serving(f"http://127.0.0.1:{lifecycle.port}", _model_of(call))
    server = HeldServer(serving, lifecycle.framework, lambda: run.stopped_rc is not None)
    run.held = await run_held_checks(_held_checks_of(call), server, run.phases)
    run.notes.append(run.held.flush)


def _runner_of(call: TestCall) -> Callable[[_Run], Awaitable[None]]:
    if call.kind == _REPRO:
        return _repro
    rounds = _sweep if call.prep == "conc_sweep" else _single
    if call.short_prompts is not None:
        rounds = _then_check(rounds, _short_prompts, _CHECK_SKIPPED)
    if call.long_context is not None:
        rounds = _then_check(rounds, _long_context_unheld, _LONG_CONTEXT_SKIPPED)
    return rounds


def _then_check(
    rounds: Callable[[_Run], Awaitable[None]], check: Callable[[_Run], Awaitable[None]], skipped: str
) -> Callable[[_Run], Awaitable[None]]:
    """``rounds``, then ``check`` when the test ran clean; a test that failed spends no time on it and says so."""

    async def run_all(run: _Run) -> None:
        await rounds(run)
        if run.stopped_rc is None and run.results and not _test_failed(run):
            await check(run)
        else:
            run.notes.append(skipped)

    return run_all


def _test_failed(run: _Run) -> bool:
    """Whether a round of the test failed: by its status, or by the return code of a part of it (a benchmark that served
    its whole protocol keeps its status when its evaluation fails, so the evaluation's return code says it)."""
    return any(
        _is_failure(result) or any(r.returncode != 0 for r in _rounds(run, point, result))
        for point, result in run.results
    )


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
        self._phases = run.phases
        self._notes = run.notes

    def settings(self, *, cleanup: bool) -> dict[str, Any]:
        return {"cleanup": cleanup, "pid_dir": str(self.pid_dir), "port": self.port}

    def teardown(self) -> None:
        self._phases.begin(TEARDOWN)
        survivors = teardown_variant_server(self.pid_dir)
        if survivors:
            self._notes.append(f"server pids {survivors} outlived SIGKILL; {self.pid_dir} keeps their record")
        self._phases.begin(OTHER)


async def _sweep(run: _Run) -> None:
    """One arm: boot once on the highest level, bench every level on that server, evaluate a graded arm once after the
    last bench from the level it booted on (the quickest to evaluate), tear the server down."""
    order = sorted(range(len(run.call.points)), key=lambda i: -_levels(run.call.points)[i])
    if _point(run.call.points[0]).sweeps_on_cold_servers:
        run.notes.append("server restarted per level: an AgentX replay is cache-sensitive")
        await _sweep_by_restart(run, order)
        return
    lifecycle = _Lifecycle(run)
    if not lifecycle.eligible:
        run.notes.append(f"server restarted per level: {lifecycle.reason}")
        await _sweep_by_restart(run, order)
        return
    try:
        if not await _boot(run, lifecycle, order[0]):
            return
        run.evaluates_apart = True
        await _reuse(run, lifecycle, order)
        await _evaluate(run, lifecycle, order[0])
        await _held_checks(run, lifecycle)
    finally:
        lifecycle.teardown()


def _boot_envs(point: Path) -> dict[str, str]:
    """The environment a boot round of ``point`` is run with: no evaluation and one request, nothing for AgentX."""
    return {} if _point(point).agentx else dict(_BOOT_CLIENT_ENVS)


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


async def _boot(run: _Run, lifecycle: _Lifecycle, point: int) -> bool:
    """Boot the server from the point at index ``point``; whether it is up. A failed boot is the arm's failed row: no
    level boots another server, as ``CONC`` never reaches the server's launch."""
    boot = await _boot_server(run, lifecycle, point, f"c{_levels(run.call.points)[point]}")
    if _is_failure(boot):
        run.add(point, boot)
        return False
    return True


async def _reuse(run: _Run, lifecycle: _Lifecycle, order: list[int]) -> None:
    """Bench each level of ``order`` on the running server; the last one tears it down unless a round follows it."""
    graded, stays_up = run.call.accuracy, run.call.accuracy or _checks_held(run.call)
    for position, point in enumerate(order):
        result = await _grid(
            run,
            point,
            f"c{_levels(run.call.points)[point]}",
            server_lifecycle=lifecycle.settings(cleanup=position == len(order) - 1 and not stays_up),
            server_already_ready=True,
            base_extra_envs=_BENCH_ENVS if graded else None,  # an explore's point does not evaluate
            warmup_before_measure=False,
        )
        if not run.add(point, result):
            return


async def _sweep_by_restart(run: _Run, order: list[int]) -> None:
    """One server per level, booted and torn down by the level's own round."""
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
            run.phases.begin(REPRO_CASE)
            await asyncio.to_thread(_run_case, run, lifecycle.port)
            run.phases.begin(OTHER)
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
            model=_model_of(call),
            timeout_sec=timeout,
            output_dir=Path(call.output_dir),
        )
    except ReproError as exc:
        _fail_case(run, str(exc))


def _model_of(call: TestCall) -> str:
    return _point(call.points[0]).model


async def _short_prompts(run: _Run) -> None:
    """Write the short-prompt check: the test's own server's answers, or why it gave none; a failure of the check is the
    check's (``outputs.json`` says), never the test's."""
    first = run.held.short if run.held is not None else not_held(_held_checks_of(run.call), _unheld_reason(run.call))
    if first is not None:
        write_answered(first, Path(run.call.output_dir))


async def _long_context_unheld(run: _Run) -> None:
    """The long-context check of a test whose server was not held up for it (one that was ran it there)."""
    if run.held is None:
        await long_context_unheld(_held_checks_of(run.call), _unheld_reason(run.call))


def _unheld_reason(call: TestCall) -> str:
    return _NOT_HELD if _point(call.points[0]).holds_accuracy_checks else _NOT_CHECKED


def _held_checks_of(call: TestCall) -> HeldChecks:
    served = served_by(*launch_carriers(_point_config(call.points[0])))
    return HeldChecks(call.short_prompts, call.long_context, Path(call.output_dir), served)


def _tracer(call: TestCall, notes: list[str]) -> ruppell.Tracer | None:
    """ruppell built for a profile test, or ``None`` with a note saying why the profile runs without it."""
    if call.kind != _PROFILE:
        return None
    prepared = ruppell.prepare(call.points, os.environ)
    if isinstance(prepared, str):
        notes.append(prepared)
        return None
    return prepared


def _traced(call: TestCall, tracer: ruppell.Tracer | None, notes: list[str]) -> tuple[TestCall, ruppell.Tracer | None]:
    """The call whose points launch their servers under ``tracer``; unchanged, with a note, when it cannot be given
    them. Read in the test's own process environment, which the server inherits."""
    if tracer is None:
        return call, None
    try:
        return dataclasses.replace(call, points=tracer.traced_points(call.points, os.environ)), tracer
    except (ruppell.RuppellError, OSError, ValueError, yaml.YAMLError) as exc:
        notes.append(f"ruppell: skipped ({str(exc).strip()[:300]})")
        return call, None


def _traced_attempt_failed(run: _Run) -> str:
    """Why the traced attempt of a profile failed in a way a run without ruppell may not, or ``""``.

    A round that failed, or passed with no trace, is run again; a timeout, a cancellation or a run out of the session's
    time is not: the time is spent.
    """
    if run.traced is None or run.stopped_rc is not None:
        return ""
    if any(result.error_class in _TIMEOUT_CLASSES for _, result in run.results):
        return ""
    for _, result in run.results:
        if _is_failure(result):
            first = (result.error or result.error_class or "the round failed").strip().splitlines()
            return (first[0] if first else "the round failed")[:300]
    if not any(result.workspace and round_has_traces(Path(result.workspace)) for _, result in run.results):
        return "the profiled round left no trace file"
    return ""


def _set_aside(run: _Run) -> None:
    """Remove what the failed traced attempt left in the test's output, keeping the tail of its server log, so the
    attempt without ruppell starts from an empty one and the bundle carries one attempt's traces."""
    tracer = run.traced
    assert tracer is not None
    output = Path(run.call.output_dir)
    for _, result in run.results:
        tracer.keep_attempt(result.server_log_path, output)
        if result.workspace:
            shutil.rmtree(result.workspace, ignore_errors=True)
    shutil.rmtree(output / "capture_traces", ignore_errors=True)


def _ranks(call: TestCall) -> int:
    """The worker processes ruppell's check expects: the point's tensor-parallel size."""
    try:
        return max(1, int(_point(call.points[0]).envs.get(ENV_TP) or 1))
    except (TypeError, ValueError):
        return 1


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
            raise MeasurementTimeout(
                run.call.task_id, run.call.timeout_sec, output=result.error, phases=_stamped_phases(run)
            )


def _stamped_phases(run: _Run) -> tuple[PhaseTiming, ...]:
    """The phases stamped so far; a recorder the backend made itself is closed first, a runner's goes on."""
    return run.phases.finish() if run.call.phases is None else run.phases.phases


def _result_returncode(result: VariantResult) -> int:
    if result.returncode is not None:
        return int(result.returncode)
    stop = _STOP_RETURNCODES.get(result.error_class)
    if stop is not None:
        return stop
    return 1 if _is_failure(result) else 0


def _eval_returncode(result: VariantResult) -> int:
    """The accuracy pass's own exit code when its run_dir's report has one, else the round's."""
    workspace = Path(result.workspace or "")
    code = RoundFolder.read(workspace).report.codes.eval if is_round_folder(workspace) else None
    return _result_returncode(result) if code is None else code


def _warmup_round_outcome(point: int, result: VariantResult, workspace: Path) -> RoundOutcome:
    """The warmup round of a held test, which says what ``run_grid`` says of a warmup it discarded."""
    warnings = list(result.nonfatal_warnings)
    if not _is_failure(result):
        discarded = [_WARMUP_DISCARDED]
        if result.output_throughput is not None:
            discarded.append(f"{_WARMUP_TPUT}:{float(result.output_throughput):.1f}")
        warnings = [*discarded, *warnings]
    return RoundOutcome(point, _WARMUP, workspace, _result_returncode(result), nonfatal_warnings=tuple(warnings))


def _rounds(run: _Run, point: int, result: VariantResult) -> list[RoundOutcome]:
    workspace = Path(result.workspace) if result.workspace else Path(run.call.output_dir)
    if result is run.warmup:
        return [_warmup_round_outcome(point, result, workspace)]
    if result is run.evaluation:  # no scores from an evaluation that failed
        scores, warnings = () if _is_failure(result) else read_accuracy(workspace), tuple(result.nonfatal_warnings)
        return [RoundOutcome(point, _ACCURACY, workspace, _eval_returncode(result), scores, warnings)]
    rounds = []
    if any(w == _WARMUP_DISCARDED for w in result.nonfatal_warnings):
        rounds.append(RoundOutcome(point, _WARMUP, workspace.parent / "warmup_round", 0))
    if result.note != _BOOT_ONLY_NOTE:
        rounds.append(RoundOutcome(point, "measure", workspace, _result_returncode(result)))
    if run.call.accuracy and not run.evaluates_apart and not _is_failure(result):
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


def _read_refusal(path: Path) -> str:
    """The last refusal of an evaluation request the log at ``path`` holds wherever it is in the log, else ``""``.

    The evaluation client prints a refusal when it happens and, once its retries are spent, one traceback for each
    request that gave up, thousands of them: the tail of the log is those tracebacks and the refusal sits far above.
    """
    try:
        with open(path, "rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            floor = max(0, end - _REFUSAL_SCAN_CHARS)
            while end > floor:
                start = max(floor, end - _REFUSAL_BLOCK_CHARS)
                handle.seek(start)
                block = handle.read(end - start).decode(_UTF8, errors="replace")
                # The block starts mid-line unless it starts the file; the overlap of the next block holds that line whole.
                found = log_causes.bad_request(block if start == 0 else block.partition("\n")[2])
                if found:
                    return found
                end = start + _REFUSAL_BLOCK_OVERLAP_CHARS if start > floor else floor
    except OSError:
        return ""
    return ""


def _image_digest(call: TestCall) -> str | None:
    environment = call.environment
    return getattr(environment, "image_digest", None) or getattr(environment, "base_image_digest", None)


def _sandbox_facts(call: TestCall) -> SandboxFacts:
    """What this machine can state about itself without asking a device; the rest stays unset, not guessed."""
    node = node_name()
    visible = os.environ.get(ROCR_VISIBLE_DEVICES_ENV, "")
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


def _failure_logs(failing: list[VariantResult]) -> list[str]:
    """The tails of the server log and of the benchmark's own stderr of every result with a failed round.

    The launcher's pipe seldom holds the cause: the server prints a refused flag or a crash into ``server.log``, and the
    evaluation client prints a refused request into ``benchmark_stderr.log``.
    """
    logs: list[str] = []
    for result in failing:
        if not result.server_log_path:
            continue
        beside = Path(result.server_log_path).parent
        stderr_logs = [beside / _BENCHMARK_STDERR, *sorted(beside.glob(f"benchmark_*/{_BENCHMARK_STDERR}"))]
        logs.extend(_read_tail(str(path)) for path in [Path(result.server_log_path), *stderr_logs])
        logs.extend(_read_refusal(path) for path in stderr_logs)
    return logs


def _client_logs(failing: list[VariantResult]) -> list[str]:
    """The tails of the AgentX replay client's own log in the run_dir of every result with a failed round.

    The round's report names only the client's exit code; what the client was refused is in this log alone.
    """
    return [
        _read_tail(str(RunLayout(Path(result.workspace)).agentx_client_log)) for result in failing if result.workspace
    ]


def _boot_timeout_detail(run: _Run, failing: list[VariantResult]) -> str:
    """What the boot guard saw when it ended the test, or ``""`` when it did not: the limit, the time aiter's module builds
    were left out of it, and where the log stopped."""
    if run.boot is None or run.boot.expired_after is None or run.stopped_rc != ORCHESTRATOR_CANCELLED_RETURNCODE:
        return ""
    logs = [_read_tail(result.server_log_path) for result in failing]
    ends = log_causes.last_line(next((text for text in logs if text), ""))
    detail = f"the server was not ready within {run.boot.expired_after:.0f}s"
    if run.boot.building_sec > 0:
        detail += f" plus {run.boot.building_sec:.0f}s of aiter module builds ({run.boot.waited_sec:.0f}s in all)"
    detail += f" of the {run.call.timeout_sec:.0f}s a round may take"
    return f"{detail}; its log ends: {ends}" if ends else detail


@dataclass(frozen=True)
class _Ended:
    """How the rounds a test has run end it: the rounds, the return code and stderr they leave, and the verdict on them."""

    rounds: list[RoundOutcome]
    returncode: int
    stderr: str
    verdict: FailureClassification


def _ended(run: _Run) -> _Ended:
    """The end the test has if it stops after the results it holds, which is what ``_outcome`` reports when it does."""
    by_result = [(result, _rounds(run, point, result)) for point, result in run.results]
    rounds = [r for _, of_result in by_result for r in of_result]
    # A benchmark that served its whole protocol keeps its status when its evaluation fails, so a round's own return
    # code, not its status, says whose logs hold the cause.
    failing = [result for result, of_result in by_result if any(r.returncode != 0 for r in of_result)]
    detail = _boot_timeout_detail(run, failing)
    notes = list(run.notes)
    if detail:
        # The guard ended the round by cancelling it; the cancellation is the symptom, the slow boot the failure.
        rounds = [
            dataclasses.replace(r, returncode=SERVER_BOOT_TIMEOUT_RC) if r.returncode == run.stopped_rc else r
            for r in rounds
        ]
        notes.append(f"{BOOT_CANCEL_REASON}: {detail}")
    failed = [r for r in rounds if r.returncode != 0]
    returncode = failed[0].returncode if failed else _clean_returncode(run)
    stderr = "\n".join([*(result.error for _, result in run.results if result.error), *notes])
    evidence = FailureEvidence(
        logs=(stderr, *_failure_logs(failing)),
        client_logs=tuple(_client_logs(failing)),
        detail=detail,
        step_failure=_step_failure(failing),
        unrun=_unrun(failing),
        framework_changed=run.call.framework_changed,
    )
    verdict = classify_test_failure(returncode=returncode, evidence=evidence)
    return _Ended(rounds, returncode, stderr, verdict)


def _step_failure(failing: list[VariantResult]) -> StepFailure | None:
    """The failed step and detail of the first failed round whose report names one; a round with no run_dir names none."""
    folders = (Path(result.workspace) for result in failing if result.workspace)
    for report in (RoundFolder.read(folder).report for folder in folders if is_round_folder(folder)):
        if report.failed_step is not None:
            return StepFailure(report.failed_step, report.failure_detail or "")
    return None


def _unrun(failing: list[VariantResult]) -> str:
    """``<error_class>: <error>`` of the first failed round that left neither a workspace nor a server log (it was refused
    before it ran), else ``""``."""
    for result in failing:
        if result.error_class and not result.workspace and not result.server_log_path:
            return f"{result.error_class}: {result.error}" if result.error else result.error_class
    return ""


def _outcome(run: _Run) -> TestOutcome:
    call = run.call
    ended = _ended(run)
    rounds, returncode, stderr, verdict = ended.rounds, ended.returncode, ended.stderr, ended.verdict
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
        phases=_stamped_phases(run),
    )
