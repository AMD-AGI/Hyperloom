"""Multi-attempt orchestration: ``quantize_via_prompt`` public entry."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .assessment import (
    ASK_RETRYABLE,
    AUTO_FAIL,
    AUTO_RECOVER,
    Assessment,
    build_assessment,
    classify_attempt,
    classify_unfinished_attempt,
    derive_status,
)
from .outcomes import OutcomeId, SUCCESS_TAGS, UNCLASSIFIED_FAILURE
from .result_collector import ATTEMPT_MARKER_FILES, collect_artifacts
from .runner import AttemptContext, RunOneAttemptFn, attempt_ended, run_one_attempt


# Upstream git URL for the Quark repo; quoted in the quark_root_missing error so operators know where to clone from.
DEFAULT_QUARK_GIT_URL = "https://github.com/amd/Quark.git"


@dataclass(frozen=True)
class QuantSkillRunResult:
    """Public return shape of :func:`quantize_via_prompt`."""

    status: str  # "success" | "partial" | "failed"
    quantized_model_dir: Path | None
    assessment: Assessment


_ARCHIVE_INFIX = ".attempt_"


def _archive_earlier_end(workspace: Path) -> bool:
    """Rename the markers an earlier attempt or run left to the next free ``.attempt_<k>``.

    Archiving is what lets the next attempt be judged on its own files. Returns
    whether the workspace holds any earlier attempt's markers, archived now or
    by an earlier attempt, and so the outputs that attempt left.
    """
    present = [name for name in ATTEMPT_MARKER_FILES if (workspace / name).is_file()]
    archived = [
        int(suffix)
        for name in ATTEMPT_MARKER_FILES
        for path in workspace.glob(f"{name}{_ARCHIVE_INFIX}*")
        if (suffix := path.name.removeprefix(f"{name}{_ARCHIVE_INFIX}")).isdigit()
    ]
    index = max(archived, default=0) + 1
    for name in present:
        (workspace / name).replace(workspace / f"{name}{_ARCHIVE_INFIX}{index}")
    return bool(present or archived)


async def _run_and_classify(
    attempt: AttemptContext, run_attempt: RunOneAttemptFn, *, model: str | None, log: Callable[[str], None] | None
) -> OutcomeId | None:
    """Run one attempt and classify it on the files it wrote itself.

    An attempt on a workspace holding an earlier attempt's model and reports
    has to write an end of its own; otherwise those outputs would pass as its
    result.
    """
    workspace = attempt.workspace
    inherits_outputs = _archive_earlier_end(workspace)
    sdk_error = (await run_attempt(attempt, model=model, log=log)).sdk_error or None
    artifacts = collect_artifacts(workspace)
    if inherits_outputs and not attempt_ended(artifacts):
        return classify_unfinished_attempt(sdk_error, phase=artifacts.last_phase)
    return classify_attempt(
        workspace,
        sdk_error=sdk_error,
        last_phase=artifacts.last_phase,
        acceptable_eval_gap=attempt.acceptable_eval_gap,
        artifacts=artifacts,
    )


# interactive prompt


def _resolve_interactive(interactive: bool | None) -> bool:
    """Resolve the effective interactive mode."""
    if interactive is not None:
        return interactive
    # Auto: enable only if both stdin and stderr are ttys.
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, OSError):
        return False


def _ask_operator(message: str) -> bool:
    """Prompt the operator on stderr for a yes/no decision."""
    print(message, file=sys.stderr, flush=True)
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return False
    return line.strip().lower() in ("y", "yes")


# retry decision


def _fix_hypothesis_path(workspace: Path, attempt_number: int) -> Path:
    """Where SKILL.md writes its hypothesis for the attempt after ``attempt_number``."""

    return workspace / f"fix_hypothesis_attempt_{attempt_number + 1}.md"


@dataclass(frozen=True)
class _RetryDecision:
    """Outcome of one ``_decide_next_step`` call."""

    retry: bool
    note: str
    promote_to: OutcomeId | None = None


def _decide_next_step(
    outcome: OutcomeId | None,
    *,
    fix_hypothesis: Path,
    interactive: bool,
    max_requantize_attempts: int,
    counter: int,
) -> _RetryDecision:
    """Decide whether to retry, accept, or stop after one attempt."""

    if outcome is None or outcome in SUCCESS_TAGS:
        return _RetryDecision(retry=False, note="")
    if outcome in AUTO_FAIL:
        return _RetryDecision(retry=False, note=f"auto_fail:{outcome}")
    if outcome in AUTO_RECOVER:
        # Surfaced here means SKILL.md couldn't self-heal; looping won't help.
        return _RetryDecision(retry=False, note=f"auto_recover_unresolved:{outcome}")

    # Remaining: ASK + unclassified_failure.
    if outcome == OutcomeId.checkpoint_aborted:
        # Missing prompt info — retry won't synthesize it; caller amends prompt.
        return _RetryDecision(retry=False, note="checkpoint_aborted_needs_prompt_change")

    if outcome == OutcomeId.eval_gap_exceeded:
        # Decision point, not a re-run candidate.
        if interactive and _ask_operator(
            f"[quantization-agent] Eval gap exceeded ({outcome}). Accept partial result? [y/N]: "
        ):
            return _RetryDecision(
                retry=False,
                note="eval_gap_accepted_by_operator",
                promote_to=OutcomeId.eval_gap_accepted,
            )
        return _RetryDecision(retry=False, note="eval_gap_exceeded_rejected")

    # Only ASK_RETRYABLE + UNCLASSIFIED_FAILURE increment the counter.
    if outcome in ASK_RETRYABLE or outcome == UNCLASSIFIED_FAILURE:
        if counter >= max_requantize_attempts:
            return _RetryDecision(
                retry=False,
                note=f"max_attempts_exhausted:counter={counter}/{max_requantize_attempts}",
            )
        if not fix_hypothesis.is_file():
            return _RetryDecision(retry=False, note="no_fix_hypothesis")
        if interactive and not _ask_operator(
            f"[quantization-agent] Outcome `{outcome}` → fix hypothesis at {fix_hypothesis.name}. Retry? [y/N]: "
        ):
            return _RetryDecision(retry=False, note="operator_declined_retry")
        return _RetryDecision(retry=True, note="")

    # Other ASK rows (none currently — partition keeps them in the sets above).
    raise AssertionError(f"_decide_next_step: unhandled outcome {outcome!r} fell through the partition")


# main entry


async def quantize_via_prompt(
    prompt: str,
    *,
    workspace: str | os.PathLike,
    quark_root: str | os.PathLike | None = None,
    interactive: bool | None = None,
    acceptable_eval_gap: float | None = None,
    max_requantize_attempts: int = 1,
    model: str | None = None,
    runner_fn: RunOneAttemptFn | None = None,
    log: Callable[[str], None] | None = None,
) -> QuantSkillRunResult:
    """Run the quantization-agent against ``prompt`` and return a result."""
    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    if quark_root is None:
        quark_root = os.environ.get("QUARK_ROOT")
    if not quark_root:
        return _build_failed_bootstrap_result(
            workspace_path,
            OutcomeId.quark_root_missing,
            f"quark_root is not configured; set $QUARK_ROOT or pass quark_root= (clone from {DEFAULT_QUARK_GIT_URL})",
        )
    quark_root_path = Path(quark_root).expanduser()
    if not quark_root_path.is_dir():
        return _build_failed_bootstrap_result(
            workspace_path,
            OutcomeId.quark_root_missing,
            f"quark_root path does not exist or is not a directory: {quark_root_path} "
            f"(set $QUARK_ROOT or pass quark_root=; clone from {DEFAULT_QUARK_GIT_URL})",
        )

    interactive_resolved = _resolve_interactive(interactive)
    run_attempt: RunOneAttemptFn = runner_fn or run_one_attempt
    # Named by attempt number only: left in place, an earlier run's would gate and prompt this run's retries.
    for stale in workspace_path.glob("fix_hypothesis_attempt_*.md"):
        stale.unlink()

    attempts_list: list[OutcomeId | None] = []
    notes: list[str] = []
    last_outcome: OutcomeId | None = None

    attempt_n = 1
    while True:
        attempt = AttemptContext(
            user_prompt=prompt,
            workspace=workspace_path,
            quark_root=quark_root_path,
            attempt_number=attempt_n,
            acceptable_eval_gap=acceptable_eval_gap,
            interactive=interactive_resolved,
            previous_outcome=last_outcome.value if isinstance(last_outcome, OutcomeId) else None,
        )
        outcome = await _run_and_classify(attempt, run_attempt, model=model, log=log)
        attempts_list.append(outcome)
        last_outcome = outcome

        decision = _decide_next_step(
            outcome,
            fix_hypothesis=_fix_hypothesis_path(workspace_path, attempt_n),
            interactive=interactive_resolved,
            max_requantize_attempts=max_requantize_attempts,
            counter=attempt_n - 1,
        )
        if decision.note:
            notes.append(decision.note)
        if decision.promote_to is not None:
            # Rewrite the final attempt so the Assessment is self-consistent.
            attempts_list[-1] = decision.promote_to
            last_outcome = decision.promote_to
        if not decision.retry:
            break

        if log:
            log(f"quantization-agent: retrying after outcome={outcome} (retry {attempt_n}/{max_requantize_attempts})")
        attempt_n += 1

    artifacts = collect_artifacts(workspace_path)
    assessment = build_assessment(attempts_list, workspace=workspace_path, artifacts=artifacts, notes=tuple(notes))

    status = derive_status(assessment, artifacts)
    quantized_model_dir = (
        artifacts.quantized_model_dir
        if status != "failed" and artifacts.quantized_model_dir and artifacts.has_weights
        else None
    )
    return QuantSkillRunResult(
        status=status,
        quantized_model_dir=quantized_model_dir,
        assessment=assessment,
    )


def _build_failed_bootstrap_result(
    workspace: Path,
    outcome: OutcomeId,
    note: str,
) -> QuantSkillRunResult:
    """Fast-path failure that bypasses the SDK (used for bootstrap errors)."""

    return QuantSkillRunResult(
        status="failed",
        quantized_model_dir=None,
        assessment=Assessment(
            final=outcome,
            attempts=(outcome,),
            recovered=False,
            eval_gap=None,
            notes=(note,),
        ),
    )


__all__ = [
    "QuantSkillRunResult",
    "quantize_via_prompt",
]
