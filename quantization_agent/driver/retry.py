"""Multi-attempt orchestration: ``quantize_via_prompt`` public entry.

Wraps :func:`_runner.run_one_attempt` with the diagnose-fix-retry protocol
(the per-attempt contract lives in ``SKILL.md``):

  * Each attempt → classifier → outcome.
  * ``None`` / ``eval_gap_accepted``                → done, build assessment.
  * ``AUTO_FAIL`` (including ``upstream_change_required``) → done, failed.
  * ``AUTO_RECOVER`` outcomes that surfaced to the final attempt → done,
    partial (per §5.4 — SKILL.md should have auto-recovered in-session;
    if we see one here it means the auto-recovery itself ran out of
    budget and Python should not loop on it).
  * ``ASK_RETRYABLE`` (#3 / #6 / #16 / #26) + ``unclassified_failure`` (#30)
    → require ``fix_hypothesis_attempt_N.md`` from SKILL.md as the precondition
    for incrementing ``requantize_attempts.txt`` and trying again. Hard cap
    by ``max_requantize_attempts`` (default 1).
  * ``checkpoint_aborted`` (#2) and ``eval_gap_exceeded`` (#21) — Ask-class
    decision points that retrying won't help. In ``interactive=True`` we
    relay to stdin (y/n on stderr). In CI we stop and let the assessment
    surface the call.

The counter file persists across interpreter restarts so a CI re-invocation
of ``quantize_via_prompt`` on the same workspace continues counting from
where the prior run left off (matters when the caller wraps us in its own
retry budget).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .assessment import (
    ASK,
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
from .runner import AttemptResult, RunOneAttemptFn, run_one_attempt


_COUNTER_FILE = "requantize_attempts.txt"

# Canonical Quark checkout used when neither the ``quark_root`` kwarg nor the
# ``$QUARK_ROOT`` env var is set. Until the Quark team ships a public package
# that bundles ``.claude/skills/quark-torch-*`` (and a version-matched
# ``amd-quark`` wheel), agents resolve the repo from this local checkout.
DEFAULT_QUARK_ROOT = "/wekafs/hyperloom/Quark"

# Upstream git URL for the Quark repo. Left empty on purpose: fill this in once
# the Quark repo is open-sourced so installers can clone it when the default
# checkout is absent.
DEFAULT_QUARK_GIT_URL = ""


@dataclass(frozen=True)
class QuantSkillRunResult:
    """Public return shape of :func:`quantize_via_prompt`.

    Exactly three fields by design;
    legacy ``intent_digest`` / ``artifact_paths`` / ``sdk_error`` are folded
    into ``assessment`` (`final` / `attempts` / `recovered` / `eval_gap` +
    `notes`) so caller code never has to negotiate a sprawling result dict.
    """

    status: str  # "success" | "partial" | "failed"
    quantized_model_dir: Path | None
    assessment: Assessment


# ─────────────────────────────────────────────────────────────────────────────
# counter file
# ─────────────────────────────────────────────────────────────────────────────

def _read_counter(workspace: Path) -> int:
    f = workspace / _COUNTER_FILE
    if not f.is_file():
        return 0
    try:
        return int(f.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def _bump_counter(workspace: Path) -> int:
    n = _read_counter(workspace) + 1
    (workspace / _COUNTER_FILE).write_text(str(n), encoding="utf-8")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# interactive prompt
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_interactive(interactive: bool | None) -> bool:
    if interactive is not None:
        return interactive
    # Auto: only enable if stdin is a tty AND stderr is a tty (we use stderr
    # for the question to avoid clobbering structured stdout). Matches the
    # convention in the existing CLI prelude.
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, OSError):
        return False


def _ask_operator(message: str) -> bool:
    print(message, file=sys.stderr, flush=True)
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return False
    return line.strip().lower() in ("y", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# retry decision
# ─────────────────────────────────────────────────────────────────────────────

def _has_fix_hypothesis(workspace: Path, attempt_number: int) -> bool:
    """Look for the hypothesis written by SKILL.md for the NEXT attempt.

    Per §A.10, before re-running quark-torch-ptq SKILL.md must drop a concrete fix
    plan at ``fix_hypothesis_attempt_<next>.md``. Absence is the gate that
    prevents blind retries.
    """

    return (workspace / f"fix_hypothesis_attempt_{attempt_number + 1}.md").is_file()


@dataclass(frozen=True)
class _RetryDecision:
    """Outcome of one ``_decide_next_step`` call.

    Exactly one of ``retry`` / ``promote_to`` is meaningful at a time:
    * ``retry=True`` → run another attempt; loop bumps counter.
    * ``retry=False`` and ``promote_to`` set → operator overrode the outcome
      (currently only ``eval_gap_exceeded → eval_gap_accepted``); loop stops
      and rewrites the last attempt's outcome.
    * ``retry=False`` and ``promote_to`` unset → terminal, assemble assessment.

    ``note`` is appended to ``Assessment.notes`` either way for caller debugging.
    """

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
        # Auto-recover that surfaced here means SKILL.md couldn't self-heal
        # inside the session; Python looping won't help. §5.4.
        return _RetryDecision(retry=False, note=f"auto_recover_unresolved:{outcome}")

    # Remaining: ASK + unclassified_failure.
    if outcome == OutcomeId.checkpoint_aborted:
        # #2: missing prompt info — retry won't synthesize what the operator
        # didn't say. Caller needs to amend prompt.
        return _RetryDecision(retry=False, note="checkpoint_aborted_needs_prompt_change")

    if outcome == OutcomeId.eval_gap_exceeded:
        # #21: decision point, not a re-run candidate.
        if interactive and _ask_operator(
            f"[quantization-agent] Eval gap exceeded ({outcome}). "
            f"Accept partial result? [y/N]: "
        ):
            return _RetryDecision(
                retry=False,
                note="eval_gap_accepted_by_operator",
                promote_to=OutcomeId.eval_gap_accepted,
            )
        return _RetryDecision(retry=False, note="eval_gap_exceeded_rejected")

    # ASK_RETRYABLE (#3 / #6 / #16 / #26) + UNCLASSIFIED_FAILURE (#30) — only
    # these increment ``requantize_attempts.txt``.
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
    return _RetryDecision(retry=False, note=f"non_retryable_ask:{outcome}")


# ─────────────────────────────────────────────────────────────────────────────
# main entry
# ─────────────────────────────────────────────────────────────────────────────

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
    """Run the quantization-agent against ``prompt`` and return a result.

    All inputs except ``prompt`` and ``workspace`` are optional. ``quark_root``
    falls back to ``$QUARK_ROOT`` then to a hard error (mapped to
    ``quark_root_missing`` at the assessment level). The threshold resolves
    per ``_eval.resolve_threshold``; the interactive flag per
    ``_resolve_interactive``.
    """

    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)

    if quark_root is None:
        # Resolution order: $QUARK_ROOT env -> DEFAULT_QUARK_ROOT canonical
        # checkout. So an agent that sets neither the kwarg nor the env var
        # still finds the repo at the well-known location.
        quark_root = os.environ.get("QUARK_ROOT") or DEFAULT_QUARK_ROOT
    quark_root_path = Path(quark_root).expanduser()
    if not quark_root_path.is_dir():
        return _build_failed_bootstrap_result(
            workspace_path, OutcomeId.quark_root_missing,
            f"quark_root path does not exist or is not a directory: {quark_root_path} "
            f"(set $QUARK_ROOT or pass quark_root=; default is {DEFAULT_QUARK_ROOT}"
            + (f", clone from {DEFAULT_QUARK_GIT_URL}" if DEFAULT_QUARK_GIT_URL else "")
            + ")",
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
            # Operator overrode the outcome — rewrite the final attempt so the
            # assembled Assessment is self-consistent (no post-hoc patching).
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

    assessment = build_assessment(
        attempts_list, workspace=workspace_path, artifacts=artifacts, notes=tuple(notes)
    )

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
    """Fast-path failure that bypasses the SDK (used for bootstrap errors).

    Produces a well-formed ``QuantSkillRunResult`` so callers can branch on
    ``status`` / ``assessment.final`` without special-casing pre-flight
    failures.
    """

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


# Convenience sync wrapper for the CLI smoke path. The library entry remains
# async to compose cleanly with the orchestrator's asyncio loop.
def quantize_via_prompt_sync(prompt: str, **kwargs: Any) -> QuantSkillRunResult:
    return asyncio.run(quantize_via_prompt(prompt, **kwargs))


__all__ = [
    "QuantSkillRunResult",
    "quantize_via_prompt",
    "quantize_via_prompt_sync",
]
