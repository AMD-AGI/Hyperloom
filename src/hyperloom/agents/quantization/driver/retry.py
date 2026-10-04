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
    derive_status,
)
from .outcomes import OutcomeId, SUCCESS_TAGS, UNCLASSIFIED_FAILURE
from .result_collector import CollectedArtifacts, collect_artifacts
from .runner import RunOneAttemptFn, run_one_attempt


_COUNTER_FILE = "requantize_attempts.txt"

# Upstream git URL for the Quark repo; quoted in the quark_root_missing error so operators know where to clone from.
DEFAULT_QUARK_GIT_URL = "https://github.com/amd/Quark.git"


@dataclass(frozen=True)
class QuantSkillRunResult:
    """Public return shape of :func:`quantize_via_prompt`."""

    status: str  # "success" | "partial" | "failed"
    quantized_model_dir: Path | None
    assessment: Assessment


# counter file


def _read_counter(workspace: Path) -> int:
    """Read the persisted requantize-attempt counter."""
    f = workspace / _COUNTER_FILE
    if not f.is_file():
        return 0
    try:
        return int(f.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def _bump_counter(workspace: Path) -> int:
    """Increment and persist the requantize-attempt counter."""
    n = _read_counter(workspace) + 1
    (workspace / _COUNTER_FILE).write_text(str(n), encoding="utf-8")
    return n


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


def _has_fix_hypothesis(workspace: Path, attempt_number: int) -> bool:
    """Look for the hypothesis written by SKILL.md for the NEXT attempt."""

    return (workspace / f"fix_hypothesis_attempt_{attempt_number + 1}.md").is_file()


@dataclass(frozen=True)
class _RetryDecision:
    """Outcome of one ``_decide_next_step`` call."""

    retry: bool
    note: str
    promote_to: OutcomeId | None = None


def _decide_next_step(
    outcome: OutcomeId | None,
    *,
    workspace: Path,
    attempt_number: int,
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
        if not _has_fix_hypothesis(workspace, attempt_number):
            return _RetryDecision(retry=False, note="no_fix_hypothesis")
        if interactive and not _ask_operator(
            f"[quantization-agent] Outcome `{outcome}` → fix hypothesis at "
            f"fix_hypothesis_attempt_{attempt_number + 1}.md. Retry? [y/N]: "
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

    attempts_list: list[OutcomeId | None] = []
    notes: list[str] = []
    last_outcome: OutcomeId | None = None
    artifacts: CollectedArtifacts | None = None

    attempt_n = 1
    while True:
        attempt_result = await run_attempt(
            user_prompt=prompt,
            workspace=workspace_path,
            quark_root=quark_root_path,
            attempt_number=attempt_n,
            acceptable_eval_gap=acceptable_eval_gap,
            interactive=interactive_resolved,
            previous_outcome=last_outcome.value if isinstance(last_outcome, OutcomeId) else None,
            model=model,
            log=log,
        )

        artifacts = collect_artifacts(workspace_path)
        outcome = classify_attempt(
            workspace_path,
            sdk_error=attempt_result.sdk_error or None,
            last_phase=artifacts.last_phase,
            acceptable_eval_gap=acceptable_eval_gap,
            artifacts=artifacts,
        )
        attempts_list.append(outcome)
        last_outcome = outcome

        counter = _read_counter(workspace_path)
        decision = _decide_next_step(
            outcome,
            workspace=workspace_path,
            attempt_number=attempt_n,
            interactive=interactive_resolved,
            max_requantize_attempts=max_requantize_attempts,
            counter=counter,
        )
        if decision.note:
            notes.append(decision.note)
        if decision.promote_to is not None:
            # Rewrite the final attempt so the Assessment is self-consistent.
            attempts_list[-1] = decision.promote_to
            last_outcome = decision.promote_to
        if not decision.retry:
            break

        new_counter = _bump_counter(workspace_path)
        if log:
            log(
                f"quantization-agent: retrying after outcome={outcome} "
                f"(counter={new_counter}/{max_requantize_attempts})"
            )
        attempt_n += 1

    assessment = build_assessment(attempts_list, workspace=workspace_path, artifacts=artifacts, notes=tuple(notes))

    status = derive_status(assessment, artifacts)  # type: ignore[arg-type]
    quantized_model_dir = (
        artifacts.quantized_model_dir
        if artifacts and status != "failed" and artifacts.quantized_model_dir and artifacts.has_weights
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
