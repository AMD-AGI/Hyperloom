# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One round, from its config to its ``report.json``: :func:`run_round` runs exactly one, into an empty run_dir.

Before anything starts the round checks what it runs on: the staged tools, the source files, the frozen base, the
installed framework's version (D6) and its parser (D7) unless the server is external, and the eval's served context.
It then acquires the server, runs the load and the eval on it, and releases it. The GPU sampler follows the round's
phase marks, so its readings cover the measured window alone; the KV sampler scrapes the ready server and tags each
reading with its phase. An AgentX profile round opens the server's capture window on the replay's phase clock, a fixed
profile round around its measured pass. A scriptable round runs its base's script in place of a server and a load, and
a scriptable profile round captures the script's whole run. Every child is reaped and every artifact written before :func:`run_round`
returns, so the run_dir does not change after it; ``report.json`` is written last.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from hyperloom_common.gpu_telemetry import GpuPhase, GpuPowerRecorder
from hyperloom_common.io import atomic_write_json
from inference_suite import framework_version, parser_check, staging
from inference_suite.compose import Refusal, RoundFacts, compose
from inference_suite.contracts.base_launch import DIGEST_PREFIX, BaseLaunch, EnvValue
from inference_suite.contracts.config import ConfigError, LifecycleMode, SuiteConfig, WorkloadKind
from inference_suite.contracts.launch import ClientInvocation, Launch, VersionCheck
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import (
    EvalResult,
    ExitCodes,
    FailedStep,
    GpuMonitorBlock,
    ProbeOutcome,
    ProfileResult,
    Report,
    RunStatus,
    ServerFacts,
)
from inference_suite.eval.definition import EvalPlan, EvalRefusal, StagedEval, plan_eval
from inference_suite.framework_probe import FrameworkProbeError, probe_install
from inference_suite.framework_version import VersionRefusal
from inference_suite.frameworks import Framework, FrameworkKind, framework_facts
from inference_suite.load import agentx, inherited_env, mlperf
from inference_suite.load.agentx import AgentxRound, AgentxTool, run_agentx
from inference_suite.load.fixed import (
    ClientTool,
    FixedOutcome,
    FixedRound,
    MeasuredWindow,
    OutputActivity,
    measured_invocation,
    run_fixed,
)
from inference_suite.load.mlperf import MlperfRound, run_mlperf
from inference_suite.load.scriptable import run_scriptable
from inference_suite.profile.agentx_window import AgentxWindow
from inference_suite.profile.fixed_window import FixedWindow
from inference_suite.profile.inject import WindowCapture, make_trace_dirs, profiler_layer, settle_traces
from inference_suite.pins import LM_EVAL_DISTRIBUTION
from inference_suite.runtime.phases import Phase, PhaseClock, PhaseListener
from inference_suite.runtime.process import spawn
from inference_suite.runtime.server import Server, ServerLaunch, ServerLifecycle, server_url
from inference_suite.runtime.watch import POLL_SEC, Cancel, RoundGate, Stop, wait_child
from inference_suite.telemetry.kv_poll import KvMetricsPoller
from inference_suite.telemetry.kv_record import PHASES as KV_PHASES
from inference_suite.telemetry.kv_record import KvMetricsRecorder, KvRound

#: The eval process's request, beside the definition it was planned from.
EVAL_REQUEST_FILE = "request.json"
#: lm-eval names each results file ``results_<time>.json`` below the output path it is given.
_LM_EVAL_RESULTS = "results_*.json"
#: Recorded when the serving interpreter's framework carries no version and the base names no recipe version to hold
#: it to (D6 is not applicable).
UNVERSIONED = "unversioned"
_GPU_PHASES: Mapping[Phase, GpuPhase] = {
    Phase.WARMUP: GpuPhase.WARMUP,
    Phase.MEASURED: GpuPhase.MEASURED,
    Phase.FINISHED: GpuPhase.FINISHED,
    Phase.EVAL: GpuPhase.EVAL,
}
#: The ``config_envs`` key :class:`KvMetricsPoller` derives its ``/metrics`` URL from.
_KV_BASE_URL_KEY = "BENCHMARK_BASE_URL"
_UTF8 = "utf-8"


@dataclass(frozen=True)
class RoundResult:
    """What one round left: its report, and the run_dir it wrote."""

    report: Report
    layout: RunLayout


def _not_cancelled() -> str:
    return ""


def _canonical(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(document: object) -> str:
    return DIGEST_PREFIX + hashlib.sha256(_canonical(document).encode()).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _write_config(config: SuiteConfig, layout: RunLayout) -> str:
    """Write ``config.yaml`` and return its sha256, the ``config_sha256`` the launch records."""
    data = yaml.safe_dump(config.to_document(), sort_keys=False).encode(_UTF8)
    layout.config.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _gpu_listener(recorder: GpuPowerRecorder) -> PhaseListener:
    def hear(phase: Phase, mono: float) -> None:
        gpu_phase = _GPU_PHASES.get(phase)
        if gpu_phase is not None:
            recorder.note_phase(gpu_phase, mono)

    return hear


def _recorder(config: SuiteConfig, layout: RunLayout) -> GpuPowerRecorder | None:
    telemetry = config.telemetry.gpu
    if not telemetry.enabled:
        return None
    return GpuPowerRecorder(
        output_path=str(layout.gpu_power),
        samples_path=str(layout.gpu_samples),
        gpus=set(config.server.gpus),
        interval_sec=telemetry.interval_sec,
    )


def _gpu_monitor(recorder: GpuPowerRecorder | None, aborted: bool) -> tuple[GpuMonitorBlock, ...]:
    if recorder is None:
        return ()
    recorder.close(aborted=aborted)
    return tuple(GpuMonitorBlock.from_document(block.as_dict()) for block in recorder.gpu_monitor())


def _load_base(config: SuiteConfig) -> BaseLaunch | Stop:
    path = Path(config.server.base.path)
    try:
        base = BaseLaunch.from_document(json.loads(path.read_text(encoding=_UTF8)))
    except (OSError, ValueError) as exc:
        return Stop(FailedStep.CONFIG_INVALID, f"the base {path} cannot be read: {exc}")
    if base.digest() != config.server.base.digest:
        detail = f"the config names {config.server.base.digest}, {path} gives {base.digest()}"
        return Stop(FailedStep.BASE_DIGEST_MISMATCH, detail)
    try:
        config.check_base(base)
    except ConfigError as exc:
        return Stop(FailedStep.CONFIG_INVALID, str(exc))
    return base


def _changed_source_files(config: SuiteConfig) -> Stop | None:
    server = config.server
    changed = [path for path, sha256 in server.source_files.items() if _file_sha256(Path(path)) != sha256]
    changed += [path for path in server.absent_source_files if Path(path).exists()]
    if not changed:
        return None
    return Stop(FailedStep.SOURCE_FILES_CHANGED, f"changed since the config was written: {', '.join(changed)}")


def _runnable_env(env: Mapping[str, EnvValue]) -> dict[str, str] | Stop:
    """``env`` with each redacted value taken from the host, which must hold the value the base was resolved with."""
    runnable: dict[str, str] = {}
    for name, value in env.items():
        if isinstance(value, str):
            runnable[name] = value
            continue
        held = os.environ.get(name)
        if held is None:
            return Stop(FailedStep.CONFIG_INVALID, f"{name} is redacted in the base and the host holds no value of it")
        if hashlib.sha256(held.encode()).hexdigest() != value.sha256:
            detail = f"{name} is redacted in the base and the host's value is not the one the base was resolved with"
            return Stop(FailedStep.CONFIG_INVALID, detail)
        runnable[name] = held
    return runnable


def _measured_invocation(config: SuiteConfig, client: ClientTool | AgentxTool) -> ClientInvocation | None:
    """The load's measured run as ``launch.json`` records it; a round without a load records none."""
    if isinstance(client, AgentxTool):
        return agentx.measured_invocation(config, client)
    if config.workload is WorkloadKind.FIXED:
        return measured_invocation(config, client, server_url(config.server))
    return None


def _provisional_check(base: BaseLaunch) -> VersionCheck:
    """The version check a launch is composed with before the probe answers; :meth:`_Round._hold_framework` replaces it."""
    return VersionCheck.NOT_APPLICABLE if base.image.framework_version is None else VersionCheck.PASSED


@dataclass(frozen=True)
class _Prepared:
    """What the round runs once every check before the server has passed."""

    launch: Launch
    server_env: Mapping[str, str]
    client: ClientTool | AgentxTool
    eval_plan: EvalPlan | None


class _KvSampler:
    """Scrapes the server's ``/metrics`` into ``kv_metrics.json`` from the ready server until its load and eval end.

    The recorder is not thread-safe: the lock orders the sampling thread's scrapes with the phase marks of the round's.
    """

    def __init__(self, base_url: str, layout: RunLayout) -> None:
        poller = KvMetricsPoller(config_envs={_KV_BASE_URL_KEY: base_url})
        self._recorder = KvMetricsRecorder(poller=poller, kv_round=KvRound(output_path=str(layout.kv_metrics)))
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def hear(self, phase: Phase, mono: float) -> None:
        if self._thread is not None and phase.value in KV_PHASES:
            with self._lock:
                self._recorder.note_phase(phase.value, mono)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, name="kv-metrics-sampler", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._done.wait(POLL_SEC):
            with self._lock:
                self._recorder.tick(time.monotonic())

    def close(self, *, aborted: bool) -> None:
        if self._thread is None:
            return
        self._done.set()
        self._thread.join()
        with self._lock:
            self._recorder.close(aborted=aborted)


def _lm_eval_version() -> str | Stop:
    try:
        return importlib.metadata.version(LM_EVAL_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return Stop(FailedStep.STAGING_MISSING, f"{LM_EVAL_DISTRIBUTION} is not installed for {sys.executable}")


class _Round:
    """One round's progress: the first step it failed at, and the report sections it has filled so far."""

    def __init__(self, config: SuiteConfig, layout: RunLayout, gpu: GpuPowerRecorder | None, cancel: Cancel) -> None:
        self.config = config
        self.layout = layout
        self.kv = _KvSampler(server_url(config.server), layout)
        self.clock = PhaseClock((self.kv.hear,) if gpu is None else (self.kv.hear, _gpu_listener(gpu)))
        self.cancel = cancel
        self.started_unix = time.time()
        self.gate = RoundGate(config.deadlines, cancel, time.monotonic())
        self.stop: Stop | None = None
        self.load_measured = False
        self.base_digest: str | None = None
        self.launch_digest: str | None = None
        self.server: ServerFacts | None = None
        self.codes = ExitCodes()
        self.sections: dict[str, object] = {}
        self.capture: WindowCapture | None = None

    def fail(self, stop: Stop) -> None:
        if self.stop is None:
            self.stop = stop

    @property
    def tool_cache(self) -> Path:
        return Path(self.config.paths.tool_cache)

    def run(self, config_sha256: str) -> None:
        stop = self._run(config_sha256)
        if stop is not None:
            self.fail(stop)

    def _run(self, config_sha256: str) -> Stop | None:
        staged = staging.verify(self.tool_cache, self.config.paths.tool_manifest_sha256)
        if staged.missing:
            return Stop(FailedStep.STAGING_MISSING, staged.detail)
        stop = _changed_source_files(self.config)
        if stop is not None:
            return stop
        base = _load_base(self.config)
        if isinstance(base, Stop):
            return base
        self.base_digest = base.digest()
        if self.config.workload is WorkloadKind.SCRIPTABLE:
            self._run_scriptable(base)
            return None
        prepared = self._prepare(base, RoundFacts(config_sha256, UNVERSIONED, _provisional_check(base), staged.pins))
        if isinstance(prepared, Stop):
            return prepared
        self.launch_digest = prepared.launch.digest()
        atomic_write_json(self.layout.launch, prepared.launch.to_document())
        self._serve(prepared)
        return None

    def _run_scriptable(self, base: BaseLaunch) -> None:
        profile = self.config.profile.enabled
        if profile:
            make_trace_dirs(self.layout)
        opened = time.time()
        outcome = run_scriptable(self.config, base, self.gate, self.clock)
        if profile and outcome.invocation is not None:
            capture = WindowCapture(None, opened, time.time(), None, None)
            self.sections["profile"] = _profile_result(self.layout, capture)
        self.codes = replace(self.codes, client=outcome.returncode)
        if outcome.result is not None:
            self.sections[WorkloadKind.SCRIPTABLE.value] = outcome.result
            self.load_measured = True
        if outcome.stop is not None:
            self.fail(outcome.stop)

    def _prepare(self, base: BaseLaunch, facts: RoundFacts) -> _Prepared | Stop:
        config = self.config
        refused = _load_refusal(config)
        if refused is not None:
            return refused
        framework = Framework(base.framework)
        if framework_facts(framework).kind is not FrameworkKind.SERVING:
            return Stop(
                FailedStep.CONFIG_INVALID, f"{framework.value} serves nothing for a {config.workload.value} round"
            )
        client = self._client()
        facts = replace(facts, client=_measured_invocation(config, client), profiler=profiler_layer(config, framework))
        composed = compose(config, base, facts)
        if isinstance(composed, Refusal):
            return Stop(composed.step, composed.detail)
        env = _runnable_env(composed.env)
        if isinstance(env, Stop):
            return env
        server_env = {**inherited_env(config.server), **env}
        launch = self._hold_framework(composed, server_env)
        if isinstance(launch, Stop):
            return launch
        eval_plan = self._plan_eval(launch)
        if isinstance(eval_plan, Stop):
            return eval_plan
        if eval_plan is not None:
            launch = replace(launch, eval=eval_plan.invocation(sys.executable, self._eval_request))
        if facts.profiler is not None:
            make_trace_dirs(self.layout)
        return _Prepared(launch, server_env, client, eval_plan)

    def _client(self) -> ClientTool | AgentxTool:
        root = staging.inferencex_root(self.tool_cache)
        if self.config.workload is WorkloadKind.AGENTX:
            return AgentxTool.staged(root)
        return ClientTool(sys.executable, str(root))

    def _hold_framework(self, launch: Launch, env: Mapping[str, str]) -> Launch | Stop:
        """``launch`` once the serving interpreter's install reaches the recipe's version (D6) and its parser accepts
        every flag (D7).

        An external round holds nothing: it never starts ``launch``, and the server it reaches may run on another host.
        """
        if self.config.lifecycle.mode is LifecycleMode.EXTERNAL:
            return replace(launch, framework=replace(launch.framework, version_check=VersionCheck.NOT_HELD))
        python = self.config.server.framework_python
        try:
            install = probe_install(python, launch.framework.name, env, cwd=self.layout.framework_cwd)
        except FrameworkProbeError as exc:
            return Stop(FailedStep.FRAMEWORK_VERSION_UNKNOWN, str(exc))
        verdict = framework_version.check(install.version, launch.framework.recipe_min_version)
        if isinstance(verdict, VersionRefusal):
            return Stop(verdict.step, verdict.detail)
        held = replace(launch.framework, installed_version=install.version or UNVERSIONED, version_check=verdict)
        launch = replace(launch, framework=held)
        try:
            refusal = parser_check.check(launch, python, env, cwd=self.layout.framework_cwd)
        except parser_check.ParserCheckError as exc:
            return Stop(FailedStep.FLAG_UNKNOWN, str(exc))
        except FrameworkProbeError as exc:
            # The interpreter could not run its own parser, so no flag was judged; the install is what failed.
            return Stop(FailedStep.FRAMEWORK_VERSION_UNKNOWN, str(exc))
        if refusal is None:
            return launch
        source = "an unnamed" if refusal.source is None else f"the {refusal.source.value}"
        return Stop(refusal.step, f"{source} layer: {refusal.detail}")

    @property
    def _eval_request(self) -> Path:
        return self.layout.eval_dir / EVAL_REQUEST_FILE

    def _plan_eval(self, launch: Launch) -> EvalPlan | Stop | None:
        if not self.config.eval.enabled:
            return None
        version = _lm_eval_version()
        if isinstance(version, Stop):
            return version
        task_yaml = staging.inferencex_root(self.tool_cache) / staging.GSM8K_TASK
        plan = plan_eval(self.config, launch, StagedEval(task_yaml, version), self.layout)
        if isinstance(plan, EvalRefusal):
            return Stop(plan.step, plan.detail)
        plan.write(self.layout, self._eval_request)
        return plan

    def _serve(self, prepared: _Prepared) -> None:
        launch = prepared.launch
        # The server's identity leaves out the load and the eval, so a reuse round finds the server a boot round left.
        server_digest = replace(launch, client=None, eval=None).digest()
        server_launch = ServerLaunch(launch.argv, prepared.server_env, launch.framework.name, server_digest)
        lifecycle = ServerLifecycle(self.config, server_launch, self.clock, self.cancel)
        try:
            acquired = lifecycle.acquire()
            if isinstance(acquired, Stop):
                self.fail(acquired)
            else:
                self._sampled(acquired, prepared)
        finally:
            released = lifecycle.release()
            server = self.config.server
            reused = self.config.lifecycle.mode.finds_server
            self.server = ServerFacts(server.host, server.port, server.served_model_name, reused, released.log_range)
            if released.teardown is not None:
                self.fail(released.teardown)
            if self.capture is not None:
                self.sections["profile"] = _profile_result(self.layout, self.capture)

    def _sampled(self, server: Server, prepared: _Prepared) -> None:
        if self.config.telemetry.kv:
            self.kv.start()
        try:
            self._on_server(server, prepared)
        finally:
            self.kv.close(aborted=self.stop is not None)

    def _on_server(self, server: Server, prepared: _Prepared) -> None:
        if not self._run_load(server, prepared) or prepared.eval_plan is None:
            return
        result = self._run_eval(server, prepared.eval_plan)
        if isinstance(result, Stop):
            self.fail(result)
        else:
            self.sections["eval"] = result

    def _run_load(self, server: Server, prepared: _Prepared) -> bool:
        """Run the round's load on ``server``, if it has one; whether the round goes on."""
        workload = self.config.workload
        if isinstance(prepared.client, AgentxTool):
            return self._run_agentx(server, prepared.client, Framework(prepared.launch.framework.name))
        if workload is WorkloadKind.MLPERF:
            outcome = run_mlperf(MlperfRound(self.config, server, self.gate, self.clock))
            return self._loaded(outcome.returncode, outcome.result, outcome.stop)
        if workload is WorkloadKind.FIXED:
            fixed = self._run_fixed(server, prepared)
            return self._loaded(fixed.returncode, fixed.metrics, fixed.stop)
        return True

    def _loaded(self, returncode: int | None, section: object | None, stop: Stop | None) -> bool:
        """Record the load's exit code, report section and stop; whether the round goes on."""
        self.codes = replace(self.codes, client=returncode)
        if section is not None:
            self.sections[self.config.workload.value] = section
        if stop is not None:
            self.fail(stop)
            return False
        self.load_measured = True
        return True

    def _run_fixed(self, server: Server, prepared: _Prepared) -> FixedOutcome:
        """Run the fixed load, with a profile round's window around its measured pass."""
        config = self.config
        if not config.profile.enabled:
            return run_fixed(FixedRound(config, prepared.client, server, self.gate, self.clock))
        framework = Framework(prepared.launch.framework.name)
        window = FixedWindow(config.profile, framework, server.base_url, self.clock)
        measured = MeasuredWindow(window.open, window.close)
        outcome = run_fixed(FixedRound(config, prepared.client, server, self.gate, self.clock, measured))
        self.capture = window.finish(self.layout)
        return outcome

    def _run_agentx(self, server: Server, harness: AgentxTool, framework: Framework) -> bool:
        """Replay the traces, with the profile window following the replay's phases; whether the round goes on."""
        config = self.config
        window = (
            AgentxWindow(config.profile, framework, server.base_url, self.clock) if config.profile.enabled else None
        )
        listeners = () if window is None else (window.hear,)
        outcome = run_agentx(AgentxRound(config, harness, server, self.gate, self.clock, listeners))
        if window is not None:
            self.capture = window.finish(self.layout)
        self.codes = replace(self.codes, client=outcome.returncode)
        if outcome.result is not None:
            self.sections[WorkloadKind.AGENTX.value] = outcome.result
            self.load_measured = True
        if outcome.stop is not None:
            self.fail(outcome.stop)
            return False
        return True

    def _run_eval(self, server: Server, plan: EvalPlan) -> EvalResult | Stop:
        layout = self.layout
        invocation = plan.invocation(sys.executable, self._eval_request)
        self.clock.mark(Phase.EVAL)
        child = spawn(invocation.argv, inherited_env(self.config.server), layout.eval_stdout, layout.eval_stderr)
        budget = _EvalBudget(self.config.deadlines.eval_sec, time.monotonic())
        activity = OutputActivity(server, layout.eval_stderr)
        ended = wait_child(child, self.gate, lambda now: budget.check(now) or activity.check(now))
        self.clock.mark(Phase.EVAL_END)
        if isinstance(ended, Stop):
            return ended
        self.codes = replace(self.codes, eval=ended.returncode)
        if ended.returncode != 0:
            return Stop(
                FailedStep.EVAL_FAILED, f"the eval exited with code {ended.returncode}; see {layout.eval_stderr}"
            )
        return _eval_result(layout, plan, self.config.eval.task.value)

    def report(self, gpu_monitor: tuple[GpuMonitorBlock, ...]) -> Report:
        config = self.config
        workload = getattr(config, config.workload.value, None)
        stop = self.stop
        return Report(
            run_status=RunStatus.of(None if stop is None else stop.step, self.load_measured),
            workload_kind=config.workload,
            started_unix=round(self.started_unix, 3),
            ended_unix=round(time.time(), 3),
            codes=self.codes,
            failed_step=None if stop is None else stop.step,
            failure_detail=None if stop is None else stop.detail,
            base_digest=self.base_digest,
            launch_digest=self.launch_digest,
            workload_digest=None if workload is None else _digest(workload.to_document()),
            concurrency=getattr(workload, "concurrency", None),
            server=self.server,
            gpu_monitor=gpu_monitor,
            **self.sections,  # type: ignore[arg-type]
        )


def _load_refusal(config: SuiteConfig) -> Stop | None:
    """Why the round's load cannot run as its config writes it, asked before the server starts."""
    if config.agentx is not None:
        return agentx.refusal(config.agentx)
    if config.mlperf is not None:
        return mlperf.refusal(config.mlperf)
    return None


def _profile_result(layout: RunLayout, capture: WindowCapture) -> ProfileResult:
    """The report's profile section once the server or the script has stopped writing its traces."""
    trace_dirs = settle_traces(layout)
    return ProfileResult(
        trace_dirs=trace_dirs,
        has_step_traces=layout.profile_torch_trace.relative_to(layout.root).as_posix() in trace_dirs,
        window_open_unix=capture.window_open_unix,
        window_close_unix=capture.window_close_unix,
        running_requests_in_window=capture.running_requests_in_window,
    )


@dataclass(frozen=True)
class _EvalBudget:
    """The eval's own time budget, inside the round's."""

    eval_sec: float
    started: float

    def check(self, now: float) -> Stop | None:
        if now - self.started < self.eval_sec:
            return None
        return Stop(FailedStep.EVAL_FAILED, f"the eval ran past its {self.eval_sec}s budget")


def _finite_scores(scores: Mapping[str, object]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in scores.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }


def _eval_result(layout: RunLayout, plan: EvalPlan, task: str) -> EvalResult | Stop:
    """The eval's score as lm-eval wrote it, with the probe's counts, under the definition it was planned with."""
    results = sorted(layout.eval_dir.rglob(_LM_EVAL_RESULTS))
    if not results:
        return Stop(FailedStep.EVAL_FAILED, f"lm-eval wrote no {_LM_EVAL_RESULTS} below {layout.eval_dir}")
    path = results[-1]
    try:
        document = json.loads(path.read_text(encoding=_UTF8))
        probe = ProbeOutcome.from_document(json.loads(layout.eval_probe.read_text(encoding=_UTF8)))
        scores = document["results"][task]
        version = document.get("versions", {}).get(task)
        return EvalResult(
            task=task,
            scores=_finite_scores(scores),
            task_version=None if version is None else str(version),
            definition=plan.definition,
            definition_digest=plan.digest,
            results_file=path.relative_to(layout.root).as_posix(),
            probe=probe,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return Stop(FailedStep.EVAL_FAILED, f"the eval output in {layout.eval_dir} cannot be read: {exc!r}")


def run_round(config: SuiteConfig, *, cancel: Cancel = _not_cancelled) -> RoundResult:
    """Run the one round ``config`` describes into its run_dir, which must be absent or empty.

    ``cancel`` is asked at every poll; a non-empty answer stops the round, reaps what it started, and is the report's
    failure detail.

    Raises:
        RunDirError: The run_dir already holds something.
    """
    layout = RunLayout(Path(config.paths.run_dir))
    layout.claim()
    config_sha256 = _write_config(config, layout)
    recorder = _recorder(config, layout)
    round_ = _Round(config, layout, recorder, cancel)
    try:
        round_.run(config_sha256)
    finally:
        gpu_monitor = _gpu_monitor(recorder, aborted=round_.stop is not None)
        round_.clock.write(layout)
    report = round_.report(gpu_monitor)
    atomic_write_json(layout.report, report.to_document())
    return RoundResult(report, layout)


__all__ = ["EVAL_REQUEST_FILE", "UNVERSIONED", "RoundResult", "run_round"]
