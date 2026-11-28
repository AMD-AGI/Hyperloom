# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Play a written-down round instead of booting a server.

:class:`ScriptedRounds` stands in for ``inference_suite.run_round``: each call plays the next attempt into the
config's run_dir and answers with the report a real round would write.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import yaml

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.rehearsal.clock import VirtualClock
from hyperloom.orchestrator.rehearsal.scenario import (
    DIED_SILENTLY,
    HANG,
    READY,
    STAGE_FAILED,
    LaunchAttempt,
    LaunchScenario,
    ScenarioError,
)
from hyperloom_common.io import atomic_write_json
from inference_suite import RoundResult
from inference_suite.contracts import (
    EvalDefinition,
    EvalResult,
    ExitCodes,
    FailedStep,
    FixedMetrics,
    LifecycleMode,
    LogRange,
    ProbeOutcome,
    ReadyStamp,
    Report,
    RunLayout,
    RunStatus,
    ScriptableResult,
    ServerFacts,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.contracts.base_launch import DIGEST_PREFIX
from inference_suite.contracts.launch import EvalInvocation, Launch
from inference_suite.server_log import death_excerpt

__all__ = ["RecordedRound", "ScenarioExhausted", "ScriptedRounds"]

#: Returncode a stage failure reports when the scenario does not name one.
_FAILED_RETURNCODE = 1

#: How many liveness callbacks an attempt makes, spread across its duration.
_OUTPUT_TICKS = 4

#: lm-eval names its results file ``results_<time>.json``; a scripted eval's carries no time, so replays match.
_EVAL_RESULTS_FILE = "results_scripted.json"
#: What stands in, in a scripted eval's definition, for the task file and the model class a staged eval hashes.
_UNSTAGED_SHA256 = "0" * 64
_SCRIPTED_LM_EVAL = "scripted"
#: The context every scripted server serves its eval in.
_SERVED_CONTEXT = 32768
_EVAL_ARGV = ("scripted-eval",)

_UTF8 = "utf-8"


class ScenarioExhausted(ScenarioError):
    """The round asked for more launches than the scenario described."""


def _returncode(attempt: LaunchAttempt) -> int:
    """Return the attempt's declared returncode, or the one it implies."""
    if attempt.returncode is not None:
        return int(attempt.returncode)
    return 0 if attempt.outcome == READY else _FAILED_RETURNCODE


def _write_artifacts(attempt: LaunchAttempt, slot: Path) -> None:
    """Write the attempt's artifacts under ``slot``; a mapping or list is written as JSON."""
    for relative, content in attempt.artifacts.items():
        path = slot / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, (Mapping, list)):
            path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding=_UTF8)
        else:
            path.write_text(str(content), encoding=_UTF8)


#: The lifecycles whose run_dir ``server.log`` is the stretch copied from a live log in ``pid_dir``.
_COPIED_LOG = frozenset({LifecycleMode.BOOT, LifecycleMode.REUSE})
_MEASURED_LOADS = frozenset({WorkloadKind.FIXED, WorkloadKind.SCRIPTABLE})


@dataclass(frozen=True)
class RecordedRound:
    """One round as it was requested, and the report it was answered with."""

    config: SuiteConfig
    report: Report


@dataclass(frozen=True)
class _Stop:
    step: FailedStep
    detail: str


@dataclass(frozen=True)
class _Course:
    """What an attempt does to one round before the session deadline and cancel cut it short.

    ``ready_sec`` is when the server reported ready, ``None`` when it never did or the round watches none;
    ``prelaunch`` marks a round refused before it composed a launch. ``eval`` is what the eval that follows the load
    does, its scores or the stop it fails with, and ``None`` for a round that reaches none.
    """

    end_sec: float
    stop: _Stop | None = None
    ready_sec: float | None = None
    log: str = ""
    load_measured: bool = False
    codes: ExitCodes = field(default_factory=ExitCodes)
    prelaunch: bool = False
    eval: Mapping[str, float] | _Stop | None = None


def _serves(config: SuiteConfig) -> bool:
    return config.workload is not WorkloadKind.SCRIPTABLE


def _boots(config: SuiteConfig) -> bool:
    return _serves(config) and not config.lifecycle.mode.finds_server


def _watched_log(attempt: LaunchAttempt, config: SuiteConfig) -> str:
    """The server log the round's run_dir keeps: an external server's log is not the round's to keep."""
    if not _serves(config) or config.lifecycle.mode is LifecycleMode.EXTERNAL:
        return ""
    return attempt.rendered_log()


def _ready_at(attempt: LaunchAttempt, config: SuiteConfig) -> float | None:
    """When the round's server reported ready: after the attempt's boot, at once for a reused one."""
    if _boots(config):
        return attempt.duration_sec
    return 0.0 if _serves(config) and config.lifecycle.mode is LifecycleMode.REUSE else None


def _client_failed(returncode: int) -> _Stop:
    return _Stop(FailedStep.CLIENT_FAILED, f"the client exited with code {returncode}")


def _scripted_eval(attempt: LaunchAttempt) -> Mapping[str, float] | _Stop:
    """What the attempt's eval does: the scores it writes, or the stop a failed one ends its round with."""
    if attempt.eval_failed:
        return _Stop(FailedStep.EVAL_FAILED, f"the eval exited with code {_FAILED_RETURNCODE}")
    if attempt.eval_scores is None:
        raise ScenarioError("the round runs an eval, and its attempt scripts neither eval_scores nor eval_failed")
    return attempt.eval_scores


def _ready_course(attempt: LaunchAttempt, config: SuiteConfig) -> _Course:
    returncode = _returncode(attempt)
    loads = config.workload in _MEASURED_LOADS
    return _Course(
        end_sec=attempt.duration_sec,
        stop=None if returncode == 0 else _client_failed(returncode),
        ready_sec=_ready_at(attempt, config),
        log=_watched_log(attempt, config),
        load_measured=loads and returncode == 0,
        codes=ExitCodes(client=returncode) if loads else ExitCodes(),
        eval=_scripted_eval(attempt) if config.eval.enabled and returncode == 0 else None,
    )


def _hang_course(attempt: LaunchAttempt, config: SuiteConfig) -> _Course:
    """A hang never ends by itself: the first gate that trips stops it, the round's own budget first on a tie."""
    deadlines = config.deadlines
    gates = [(float(deadlines.round_sec), FailedStep.ROUND_TIMEOUT)]
    stuck_booting = _boots(config) and attempt.failed_stage is not None
    ready = None if stuck_booting else _ready_at(attempt, config)
    if stuck_booting:
        gates.append((float(deadlines.boot_sec), FailedStep.SERVER_BOOT_TIMEOUT))
        gates.append((float(deadlines.pre_ready_silence_sec), FailedStep.SERVER_SILENT_BEFORE_READY))
    elif ready is not None:
        gates.append((ready + float(deadlines.silence_sec), FailedStep.SERVER_STALLED))
    at, step = min(gates, key=lambda gate: gate[0])
    return _Course(
        end_sec=at,
        stop=_Stop(step, f"the scripted hang was stopped after {at:.1f}s"),
        ready_sec=None if ready is None or ready >= at else ready,
        log=_watched_log(attempt, config),
    )


def _exited_before_ready(returncode: int) -> _Stop:
    return _Stop(FailedStep.SERVER_EXITED_BEFORE_READY, f"the server exited with code {returncode} before it was ready")


def _failed_course(attempt: LaunchAttempt, config: SuiteConfig) -> _Course:
    """A stage wall: the parser refuses an unknown flag before launch, a boot wall ends the boot, a serving one the
    client."""
    stage = attempt.failed_stage
    returncode = _returncode(attempt)
    if not _serves(config) or (stage is not None and stage > LadderStage.GENERATES):
        raise ScenarioError(f"attempt {attempt.outcome!r} fails at a stage a {config.workload.value} round never plays")
    if stage is LadderStage.ARGV_PARSE:
        refusal = _Stop(FailedStep.FLAG_UNKNOWN, attempt.rendered_stderr())
        return _Course(end_sec=attempt.duration_sec, stop=refusal, prelaunch=True)
    if stage is LadderStage.GENERATES:
        return _Course(
            end_sec=attempt.duration_sec,
            stop=_client_failed(returncode),
            ready_sec=_ready_at(attempt, config),
            log=_watched_log(attempt, config),
            codes=ExitCodes(client=returncode),
        )
    if not _boots(config):
        raise ScenarioError(
            f"attempt {attempt.outcome!r} fails a boot, but a {config.lifecycle.mode.value} round boots none"
        )
    return _Course(end_sec=attempt.duration_sec, stop=_exited_before_ready(returncode), log=attempt.rendered_log())


def _died_course(attempt: LaunchAttempt, config: SuiteConfig) -> _Course:
    returncode = _returncode(attempt)
    if _boots(config):
        return _Course(end_sec=attempt.duration_sec, stop=_exited_before_ready(returncode))
    return _Course(end_sec=attempt.duration_sec, stop=_client_failed(returncode), codes=ExitCodes(client=returncode))


_COURSES: Mapping[str, Callable[[LaunchAttempt, SuiteConfig], _Course]] = {
    READY: _ready_course,
    HANG: _hang_course,
    STAGE_FAILED: _failed_course,
    DIED_SILENTLY: _died_course,
}


def _refuse_unscripted(config: SuiteConfig) -> None:
    """The fixed, scriptable and load-free rounds are scripted, with the eval one asks for; AgentX and profile rounds
    are not yet."""
    unscripted = [
        name
        for name, asked in (
            ("AgentX load", config.workload is WorkloadKind.AGENTX),
            ("profile", config.profile.enabled),
        )
        if asked
    ]
    if unscripted:
        raise ScenarioError(f"a scripted round plays no {' or '.join(unscripted)} yet")


def _load_section(attempt: LaunchAttempt, config: SuiteConfig, layout: RunLayout) -> dict[str, object]:
    """The report section the round's measured load fills, keyed by its report name."""
    if config.workload is WorkloadKind.FIXED:
        return {WorkloadKind.FIXED.value: FixedMetrics.of(attempt.metrics)}
    if config.workload is not WorkloadKind.SCRIPTABLE:
        return {}
    atomic_write_json(layout.inferencex_result, dict(attempt.metrics))
    result_file = layout.inferencex_result.relative_to(layout.root).as_posix()
    return {WorkloadKind.SCRIPTABLE.value: ScriptableResult(result_file, dict(attempt.metrics))}


def _eval_definition(config: SuiteConfig) -> EvalDefinition:
    """The definition a scripted eval scores under: the config's settings, and stand-ins for what staging fixes."""
    settings = config.eval
    return EvalDefinition(
        task_yaml_sha256=_UNSTAGED_SHA256,
        lm_eval_version=_SCRIPTED_LM_EVAL,
        model_class_sha256=_UNSTAGED_SHA256,
        chat_template_sha256=None,
        concurrency=settings.concurrency,
        max_length=_SERVED_CONTEXT,
        max_tokens_cap=settings.max_tokens_cap,
        limit=settings.limit,
        num_fewshot=settings.num_fewshot,
        golden_applied=False,
        probe=settings.probe,
    )


def _definition_digest(config: SuiteConfig) -> str:
    """The identity of the round's eval definition: two scripted scores compare exactly when their settings match."""
    document = json.dumps(_eval_definition(config).to_document(), sort_keys=True)
    return DIGEST_PREFIX + hashlib.sha256(document.encode()).hexdigest()


def _eval_result(scores: Mapping[str, float], config: SuiteConfig, layout: RunLayout) -> EvalResult:
    """Write what a scored eval leaves below the run_dir, lm-eval's results file and the probe's counts, and answer
    with its report section."""
    task = config.eval.task.value
    results = layout.eval_dir / _EVAL_RESULTS_FILE
    atomic_write_json(results, {"results": {task: dict(scores)}})
    probe = ProbeOutcome(tripped=False, samples_seen=0, length_ratio_observed=None)
    atomic_write_json(layout.eval_probe, probe.to_document())
    return EvalResult(
        task=task,
        scores=dict(scores),
        task_version=None,
        definition=_eval_definition(config),
        definition_digest=_definition_digest(config),
        results_file=results.relative_to(layout.root).as_posix(),
        probe=probe,
    )


@dataclass
class ScriptedRounds:
    """Stands in for ``inference_suite.run_round``: serves one scenario attempt per round, in order.

    Each round claims its run_dir as a real one does and leaves what a real one leaves: ``config.yaml``, the server
    log it keeps, the ready stamp, ``launch.json`` when ``launch`` is given, the attempt's artifacts and
    ``report.json`` last. An attempt's ``duration_sec`` is when its server reported ready, or when it stopped; a hang
    runs until the first of the round's gates trips. The session deadline (``deadlines.deadline_unix``) and
    ``cancel`` cut any round short, as they do a real one. A round asked to evaluate runs its attempt's eval once
    its load went on, at no cost to the clock: a scored one leaves lm-eval's results file and the probe's counts
    below the run_dir's eval directory, a failed one ends the round ``eval_failed`` with its load kept.

    Attributes:
        scenario: The rounds to play.
        clock: The clock rounds are charged to.
        launch: The launch every round that composes one records, or ``None`` to record none.
        calls: Every round served, in order.
    """

    scenario: LaunchScenario
    clock: VirtualClock = field(default_factory=VirtualClock)
    launch: Launch | None = None
    calls: list[RecordedRound] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        """bool: Whether every attempt in the scenario has been played."""
        return len(self.calls) >= len(self.scenario.attempts)

    def run_round(self, config: SuiteConfig, *, cancel: Callable[[], str] | None = None) -> RoundResult:
        """Play the next attempt as the round ``config`` describes, into its run_dir.

        ``cancel``, when given, is asked as the round runs; a non-empty answer stops it as it stops a real one.

        Raises:
            RunDirError: The run_dir already holds something.
            ScenarioExhausted: The scenario has no attempt left.
            ScenarioError: The round is one this double does not script, its attempt cannot happen in it or scripts
                no eval for a round that reaches one, or it names a session deadline while this double's clock is
                not the one ``time.time`` reads.
        """
        _refuse_unscripted(config)
        budget_left = self._budget_left(config.deadlines.deadline_unix)
        if self.exhausted:
            raise ScenarioExhausted(
                f"the scenario described {len(self.scenario.attempts)} attempts; "
                f"round {len(self.calls) + 1} was requested"
            )
        attempt = self.scenario.attempts[len(self.calls)]
        course = _COURSES[attempt.outcome](attempt, config)
        layout = RunLayout(Path(config.paths.run_dir))
        layout.claim()
        started = self.clock.wall()
        layout.config.write_text(yaml.safe_dump(config.to_document(), sort_keys=False), encoding=_UTF8)
        played = self._play(started, course.end_sec, budget_left, cancel)
        report = self._report(attempt, config, course, played)
        atomic_write_json(layout.report, report.to_document())
        self.calls.append(RecordedRound(config, report))
        return RoundResult(report, layout)

    def _budget_left(self, deadline_unix: int | None) -> float | None:
        if deadline_unix is None:
            return None
        if time.time != self.clock.wall:
            raise ScenarioError(
                "deadline_unix is an instant on time.time()'s timeline; play this round inside "
                "installed_clock(clock) so the deadline and the attempt durations are read off the same clock"
            )
        return float(deadline_unix) - self.clock.wall()

    def _play(
        self, started: float, seconds: float, budget_left: float | None, cancel: Callable[[], str] | None
    ) -> _Played:
        """Charge the round to the clock until it ends, the session deadline passes or ``cancel`` answers."""
        target, cut = seconds, None
        if budget_left is not None and budget_left <= seconds:
            target, cut = max(0.0, budget_left), _Stop(FailedStep.DEADLINE_EXHAUSTED, "the session deadline passed")
        step = target / _OUTPUT_TICKS
        for tick in range(_OUTPUT_TICKS):
            reason = cancel() if cancel is not None else ""
            if reason:
                return _Played(started, tick * step, _Stop(FailedStep.CANCELLED, reason))
            self.clock.advance(step)
        return _Played(started, target, cut)

    def _report(self, attempt: LaunchAttempt, config: SuiteConfig, course: _Course, played: _Played) -> Report:
        """Write what the round leaves beside its report, and build the report."""
        layout = RunLayout(Path(config.paths.run_dir))
        if course.log:
            layout.server_log.write_text(course.log, encoding=_UTF8)
        if course.ready_sec is not None and course.ready_sec <= played.spent:
            layout.write_ready_stamp(ReadyStamp(played.started + course.ready_sec, course.ready_sec))
        _write_artifacts(attempt, layout.root)
        stop = played.cut or _excerpted(course.stop, layout)
        load_measured = course.load_measured and played.cut is None
        sections = _load_section(attempt, config, layout) if load_measured else {}
        codes = course.codes
        scripted_eval = None if stop is not None else course.eval
        if isinstance(scripted_eval, _Stop):
            stop = scripted_eval
            codes = replace(codes, eval=_FAILED_RETURNCODE)
        elif scripted_eval is not None:
            sections["eval"] = _eval_result(scripted_eval, config, layout)
            codes = replace(codes, eval=0)
        launch = self._recorded_launch(config, course)
        if launch is not None:
            atomic_write_json(layout.launch, launch.to_document())
        workload = getattr(config, config.workload.value, None)
        return Report(
            run_status=RunStatus.of(None if stop is None else stop.step, load_measured),
            workload_kind=config.workload,
            started_unix=round(played.started, 3),
            ended_unix=round(self.clock.wall(), 3),
            codes=codes,
            failed_step=None if stop is None else stop.step,
            failure_detail=None if stop is None else stop.detail,
            base_digest=config.server.base.digest,
            launch_digest=None if launch is None else launch.digest(),
            concurrency=getattr(workload, "concurrency", None),
            server=None if course.prelaunch else _server_facts(config, layout),
            **sections,  # type: ignore[arg-type]
        )

    def _recorded_launch(self, config: SuiteConfig, course: _Course) -> Launch | None:
        """The launch the round composed, naming the eval it planned as a real round's ``launch.json`` does."""
        if self.launch is None or course.prelaunch or not _serves(config):
            return None
        if not config.eval.enabled:
            return self.launch
        return replace(self.launch, eval=EvalInvocation(_EVAL_ARGV, _definition_digest(config)))


@dataclass(frozen=True)
class _Played:
    """When a round started on the wall clock, how long it ran, and what cut it short."""

    started: float
    spent: float
    cut: _Stop | None


def _excerpted(stop: _Stop | None, layout: RunLayout) -> _Stop | None:
    """A boot that died names its log's death excerpt, as a real round does, when the log carries one."""
    if stop is None or stop.step is not FailedStep.SERVER_EXITED_BEFORE_READY or not layout.server_log.exists():
        return stop
    excerpt = death_excerpt(str(layout.server_log))
    return stop if excerpt is None else _Stop(stop.step, excerpt)


def _server_facts(config: SuiteConfig, layout: RunLayout) -> ServerFacts | None:
    if not _serves(config):
        return None
    server, mode = config.server, config.lifecycle.mode
    log_range = None
    if mode in _COPIED_LOG:
        size = layout.server_log.stat().st_size if layout.server_log.exists() else 0
        log_range = LogRange(0, size)
    return ServerFacts(server.host, server.port, server.served_model_name, mode.finds_server, log_range)
