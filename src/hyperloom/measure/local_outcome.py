# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a local test has run so far, and the rounds, return code and failure verdict read off it."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.contracts.measurement import PhaseTiming
from hyperloom.measure import log_causes
from hyperloom.measure.accuracy import read_accuracy
from hyperloom.measure.boot_watch import BOOT_CANCEL_REASON, BootGuard
from hyperloom.measure.failures import FailureClassification, FailureEvidence, StepFailure, classify_test_failure
from hyperloom.measure.long_context.held import HeldAnswers
from hyperloom.measure.phases import PhaseRecorder
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC, SERVER_BOOT_TIMEOUT_RC
from hyperloom.measure.round_marks import ROLE_WARMUP, profile_stall
from hyperloom.measure.second_tracer import SecondTracer as _Tracer
from hyperloom.measure.socket import RoundOutcome, TestCall
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    RoundFolder,
    VariantResult,
    is_round_folder,
)
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS

#: Return codes of rounds the run itself stopped, keyed by the class the result carries.
_STOP_RETURNCODES = {
    SESSION_TIME_EXHAUSTED_CLASS: SESSION_TIME_EXHAUSTED_RETURNCODE,
    ORCHESTRATOR_CANCELLED_CLASS: ORCHESTRATOR_CANCELLED_RETURNCODE,
}
_BOOT_ONLY_NOTE = "server_lifecycle_boot_only"
ACCURACY = "accuracy"
#: What ``run_grid`` writes on a point whose warmup round it discarded, with the throughput of that round.
_WARMUP_DISCARDED = "run_grid_warmup_discarded_first"
_WARMUP_TPUT = "warmup_round_tput"


@dataclass
class Run:
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
    #: Whether the long-context check wrote its outputs on a server (or why a check server gave none).
    long_context_ran: bool = False
    #: The second tracer that traces the test's server (a profile's); ``None`` when it runs untraced.
    traced: _Tracer | None = None

    def add(self, point: int, result: VariantResult) -> bool:
        """Record one result; say whether the test must go on."""
        self.results.append((point, result))
        stop = _STOP_RETURNCODES.get(result.error_class)
        if stop is not None:
            self.stopped_rc = stop
        return stop is None


def test_failed(run: Run) -> bool:
    """Whether a round of the test failed: by its status, or by the return code of a part of it (a benchmark that served
    its whole protocol keeps its status when its evaluation fails, so the evaluation's return code says it)."""
    return any(
        is_failure(result) or any(r.returncode != 0 for r in _rounds(run, point, result))
        for point, result in run.results
    )


def is_failure(result: VariantResult) -> bool:
    return result.status in {"failed", "skipped"}


def stamped_phases(run: Run) -> tuple[PhaseTiming, ...]:
    """The phases stamped so far; a recorder the backend made itself is closed first, a runner's goes on."""
    return run.phases.finish() if run.call.phases is None else run.phases.phases


def _result_returncode(result: VariantResult) -> int:
    if result.returncode is not None:
        return int(result.returncode)
    stop = _STOP_RETURNCODES.get(result.error_class)
    if stop is not None:
        return stop
    return 1 if is_failure(result) else 0


def _eval_returncode(result: VariantResult) -> int:
    """The accuracy pass's own exit code when its run_dir's report has one, else the round's."""
    workspace = Path(result.workspace or "")
    code = RoundFolder.read(workspace).report.codes.eval if is_round_folder(workspace) else None
    return _result_returncode(result) if code is None else code


def _warmup_round_outcome(point: int, result: VariantResult, workspace: Path) -> RoundOutcome:
    """The warmup round of a held test, which says what ``run_grid`` says of a warmup it discarded."""
    warnings = list(result.nonfatal_warnings)
    if not is_failure(result):
        discarded = [_WARMUP_DISCARDED]
        if result.output_throughput is not None:
            discarded.append(f"{_WARMUP_TPUT}:{float(result.output_throughput):.1f}")
        warnings = [*discarded, *warnings]
    return RoundOutcome(point, ROLE_WARMUP, workspace, _result_returncode(result), nonfatal_warnings=tuple(warnings))


def _rounds(run: Run, point: int, result: VariantResult) -> list[RoundOutcome]:
    workspace = Path(result.workspace) if result.workspace else Path(run.call.output_dir)
    if result is run.warmup:
        return [_warmup_round_outcome(point, result, workspace)]
    if result is run.evaluation:  # no scores from an evaluation that failed
        scores, warnings = () if is_failure(result) else read_accuracy(workspace), tuple(result.nonfatal_warnings)
        return [RoundOutcome(point, ACCURACY, workspace, _eval_returncode(result), scores, warnings)]
    rounds = []
    if any(w == _WARMUP_DISCARDED for w in result.nonfatal_warnings):
        rounds.append(RoundOutcome(point, ROLE_WARMUP, workspace.parent / "warmup_round", 0))
    if result.note != _BOOT_ONLY_NOTE:
        rounds.append(RoundOutcome(point, "measure", workspace, _result_returncode(result)))
    if run.call.accuracy and not run.evaluates_apart and not is_failure(result):
        rounds.append(RoundOutcome(point, ACCURACY, workspace, _eval_returncode(result), read_accuracy(workspace)))
    if rounds:
        rounds[0] = dataclasses.replace(rounds[0], nonfatal_warnings=tuple(result.nonfatal_warnings))
    return rounds


def _clean_returncode(run: Run) -> int:
    """The return code of a test whose rounds all passed: a case that could not run, or a trace-less profile, is not 0."""
    return HOSTED_INFRA_FAILURE_RC if run.case_failed else int(run.traceless)


def _boot_timeout_detail(run: Run, failing: list[VariantResult]) -> str:
    """What the boot guard saw when it ended the test, or ``""`` when it did not: the limit, the time aiter's module builds
    were left out of it, and where the log stopped."""
    if run.boot is None or run.boot.expired_after is None or run.stopped_rc != ORCHESTRATOR_CANCELLED_RETURNCODE:
        return ""
    logs = [log_causes.log_tail(result.server_log_path) for result in failing]
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


def ended(run: Run) -> _Ended:
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
        logs=(stderr, *log_causes.failure_logs(failing)),
        client_logs=tuple(log_causes.client_logs(failing)),
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
    for round_ in (RoundFolder.read(folder) for folder in folders if is_round_folder(folder)):
        if (step := round_.report.failed_step) is not None:
            return StepFailure(step, round_.report.failure_detail or "", profile_stall(round_.layout.root, step))
    return None


def _unrun(failing: list[VariantResult]) -> str:
    """``<error_class>: <error>`` of the first failed round that left neither a workspace nor a server log (it was refused
    before it ran), else ``""``."""
    for result in failing:
        if result.error_class and not result.workspace and not result.server_log_path:
            return f"{result.error_class}: {result.error}" if result.error else result.error_class
    return ""
