"""Assessment dataclass + ``classify_attempt`` — turn one attempt's
workspace state into a single ``OutcomeId``.

``classify_attempt`` is a pure function (workspace snapshot + sdk_error +
phase hint → OutcomeId | None). The retry loop accumulates per-attempt
outcomes into a multi-attempt ``Assessment``.

Classification precedence (first match wins):

  1. Hard SDK-level signatures in ``sdk_error`` mapping to bootstrap-class
     outcomes, decisive even if some artifacts exist.
  2. Explicit ``blocked.md`` outcome marker (``outcome_id: <id>``) when it is
     a known ``OutcomeId``.
  3. Phase-aware artifact gaps under the recorded ``last_phase``.
  4. MUST-have model files on the quantized directory.
  5. Validator step results (FAIL > SKIPPED > absent step heading).
  6. Eval phase — ``eval_skipped.txt`` first, then ``eval_report.json`` gap
     vs. threshold.
  7. sdk_error pattern match, which can fire even when artifacts look
     partially intact.
  8. Fallback: ``unclassified_failure`` if any failure signal is present,
     otherwise ``None`` for clean success (with the ``eval_gap_accepted`` tag
     when the gap is non-zero and within budget).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .eval import decide as decide_eval
from .outcomes import (
    ASK,
    ASK_RETRYABLE,
    AUTO_FAIL,
    AUTO_RECOVER,
    MUST_HAVE_RECOVERS_THAT_FAIL_WITHOUT_ARTIFACT,
    OutcomeId,
    SUCCESS_TAGS,
)
from .result_collector import CollectedArtifacts, collect_artifacts


_BLOCKED_OUTCOME_RE = re.compile(r"(?:^|\n)\s*outcome_id\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

_GAP_NARRATIVE_EPSILON = 1e-4  # gaps smaller than this are clean success


@dataclass(frozen=True)
class Assessment:
    """Public summary of a (possibly multi-attempt) quantize call.

    Fields:

    * ``final`` — primary verdict. ``None`` = clean success; otherwise an
      ``OutcomeId``.
    * ``attempts`` — per-attempt outcomes in chronological order; len == 1
      for single-shot runs.
    * ``recovered`` — ``True`` iff len(attempts) > 1 AND final is in
      ``SUCCESS_TAGS`` (i.e. an earlier attempt failed and a later one
      cleaned up).
    * ``eval_gap`` — the ``relative_gap`` from the final attempt's
      ``eval_report.json`` when present (regardless of accept/reject).
    """

    final: OutcomeId | None
    attempts: tuple[OutcomeId | None, ...]
    recovered: bool
    eval_gap: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Serialize the assessment to a JSON-friendly dictionary.

        Returns:
            A dict with the final outcome, per-attempt outcomes, recovery flag,
            evaluation gap, and notes, using enum *values* for outcome ids.
        """
        return {
            "final": self.final.value if self.final is not None else None,
            "attempts": [a.value if a is not None else None for a in self.attempts],
            "recovered": self.recovered,
            "eval_gap": self.eval_gap,
            "notes": list(self.notes),
        }


# pattern banks

_SDK_RUNTIME_PATTERNS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "authentication",
    "auth_error",
    "unauthorized",
    "api key",
    "context length",
    "context_length",
    "context window",
    "connectionerror",
    "timeouterror",
    "anyio.endofstream",
    "anthropic.",
    "claude_agent_sdk",
)

_OOM_PATTERNS = ("out of memory", "cuda out of memory", "oom", "torch.outofmemoryerror")

_MODEL_LOAD_PATTERNS = (
    "safetensorerror",
    "missing keys",
    "no such file",
    "from_pretrained",
    "transformers.utils.import_utils",
    "weight shape",
    "dtype mismatch",
)

_EXPORT_PATTERNS = (
    "save_pretrained",
    "safetensors.write",
    "ioerror",
    "no space left",
    "stale file handle",
    "nfs",
)

_QUANTIZED_LOAD_PATTERNS = (
    "quantized_load",
    "quantized load",
    "failed to load quantized",
    "vllm engine.start failed",
    "sglang engine.start failed",
    "engine.start",
    "engine startup",
)


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    """Return whether any needle substring appears in ``haystack``.

    Args:
        haystack: String to search within.
        needles: Candidate substrings to look for.

    Returns:
        ``True`` if at least one needle is found, otherwise ``False``.
    """
    for n in needles:
        if n in haystack:
            return True
    return False


def _parse_blocked_outcome(text: str | None) -> OutcomeId | None:
    """Extract an explicit ``BLOCKED`` outcome id from agent output text.

    Args:
        text: Free-form text that may contain a ``BLOCKED`` outcome marker.

    Returns:
        The matching :class:`OutcomeId`, or ``None`` when no valid marker is
        present.
    """
    if not text:
        return None
    m = _BLOCKED_OUTCOME_RE.search(text)
    if not m:
        return None
    raw = m.group(1).lower()
    try:
        return OutcomeId(raw)
    except ValueError:
        return None


def _classify_eval_outcome(
    art: CollectedArtifacts,
    *,
    acceptable_eval_gap: float | None,
) -> OutcomeId | None:
    """Map eval-phase artifacts to an outcome.

    Args:
        art: Collected artifacts for the attempt.
        acceptable_eval_gap: Maximum tolerated relative accuracy gap, if
            configured.

    Returns:
        The matching eval :class:`OutcomeId`, or ``None`` when eval was
        not exercised this attempt.
    """

    if art.eval_skipped_reason:
        reason = art.eval_skipped_reason.lower()
        if _contains_any(reason, _OOM_PATTERNS):
            return OutcomeId.eval_oom
        if _contains_any(reason, _QUANTIZED_LOAD_PATTERNS):
            return OutcomeId.quantized_load_failed
        return OutcomeId.eval_env_unavailable

    if not art.eval_report_present:
        return None  # eval was not run this attempt

    decision = decide_eval(
        art.eval_report_data,
        workspace=art.workspace,
        acceptable_eval_gap=acceptable_eval_gap,
    )
    if decision.status == "missing":
        # File present but unparsable — treat as env-unavailable, not a pass.
        return OutcomeId.eval_env_unavailable
    if decision.status == "exceeded":
        return OutcomeId.eval_gap_exceeded
    # "within" — narrative tag only when gap is large enough to mention
    if decision.relative_gap and decision.relative_gap > _GAP_NARRATIVE_EPSILON:
        return OutcomeId.eval_gap_accepted
    return None


def _classify_sdk_phase_error(
    sdk_error: str,
    phase: str | None,
) -> OutcomeId | None:
    """Map an SDK error under a known phase to a phase-specific outcome.

    Args:
        sdk_error: Raw SDK error text.
        phase: The phase that was executing when the error occurred.

    Returns:
        The matching :class:`OutcomeId`, or ``None`` if the message is not
        phase-specific (the caller then falls through to bootstrap-level
        patterns).
    """

    msg = sdk_error.lower()
    if phase == "exec":
        if _contains_any(msg, _OOM_PATTERNS):
            return OutcomeId.exec_oom
        if _contains_any(msg, _MODEL_LOAD_PATTERNS):
            return OutcomeId.exec_model_load_failed
        if "calibration" in msg or "dataloader" in msg:
            return OutcomeId.exec_calibration_data_missing
    if phase == "export":
        if _contains_any(msg, _EXPORT_PATTERNS):
            return OutcomeId.export_crashed
    if phase == "eval":
        if _contains_any(msg, _OOM_PATTERNS):
            return OutcomeId.eval_oom
        if _contains_any(msg, _QUANTIZED_LOAD_PATTERNS):
            return OutcomeId.quantized_load_failed
        return OutcomeId.eval_env_unavailable
    if phase == "intake":
        if "config.json" in msg or "no such file" in msg or "not a directory" in msg:
            return OutcomeId.model_path_unreachable
    return None


_MANIFEST_REQUIRED_PHASES = frozenset({"manifest", "exec", "export", "validate", "eval"})


def _classify_phase_artifact_gap(
    art: CollectedArtifacts,
    phase: str | None,
) -> OutcomeId | None:
    """Detect disk-evidence gaps relative to the last-written phase.

    Args:
        art: Collected artifacts for the attempt.
        phase: The phase that last wrote ``last_phase.txt``.

    Returns:
        The matching :class:`OutcomeId`, or ``None`` if nothing is amiss
        at this phase boundary (the caller then continues to MUST-have /
        validator / eval checks).
    """

    if phase == "intake" and not art.model_analysis_present:
        return OutcomeId.analysis_artifact_invalid_or_missing
    if phase == "plan" and not art.quant_plan_present:
        return OutcomeId.plan_artifact_invalid_or_missing
    if not art.manifest_present and phase in _MANIFEST_REQUIRED_PHASES:
        return OutcomeId.manifest_artifact_invalid_or_missing
    if art.manifest_present and art.manifest_parse_error:
        if art.manifest_parse_error == "pyyaml_missing":
            return OutcomeId.nice_to_have_skipped
        return OutcomeId.manifest_artifact_invalid_or_missing
    return None


def _classify_bootstrap_sdk_error(sdk_error: str) -> OutcomeId | None:
    """Classify a bootstrap-phase SDK error message into an outcome.

    Args:
        sdk_error: Raw error text raised before the skill chain started.

    Returns:
        The matching bootstrap :class:`OutcomeId` (e.g. missing Quark root,
        unwritable workspace, runtime error), or ``None`` if unrecognized.
    """
    msg = sdk_error.lower()
    if "quark_root" in msg or ("quark root" in msg and ("missing" in msg or "not found" in msg)):
        return OutcomeId.quark_root_missing
    if "permissionerror" in msg and ("workspace" in msg or "permission denied" in msg):
        return OutcomeId.workspace_unwritable
    if "skill.md" in msg and ("missing" in msg or "not found" in msg):
        return OutcomeId.quark_skill_unavailable
    if _contains_any(msg, _SDK_RUNTIME_PATTERNS):
        return OutcomeId.sdk_runtime_error
    return None


# main entry


def classify_attempt(
    workspace: Path,
    *,
    sdk_error: str | None = None,
    last_phase: str | None = None,
    acceptable_eval_gap: float | None = None,
    artifacts: CollectedArtifacts | None = None,
) -> OutcomeId | None:
    """Classify a single attempt's workspace state.

    The retry loop assembles per-attempt outcomes and derives the final
    ``Assessment``.

    Args:
        workspace: Attempt workspace directory to inspect.
        sdk_error: Raw SDK error text, if the attempt raised one.
        last_phase: Phase that last executed (overrides the on-disk marker).
        acceptable_eval_gap: Maximum tolerated relative accuracy gap.
        artifacts: Pre-scanned artifacts; supply to avoid a duplicate disk
            pass.

    Returns:
        ``None`` for a clean success, otherwise the classified
        :class:`OutcomeId`.
    """

    art = artifacts if artifacts is not None else collect_artifacts(Path(workspace))
    phase = last_phase or art.last_phase

    # (1) Bootstrap-class sdk_error wins over disk evidence.
    if sdk_error:
        boot = _classify_bootstrap_sdk_error(sdk_error)
        if boot is not None:
            return boot

    # (2) Explicit blocked.md marker.
    blocked_oid = _parse_blocked_outcome(art.blocked_reason)
    if blocked_oid is not None:
        return blocked_oid

    # (3) Phase-aware artifact gaps before the model is fully produced.
    phase_gap = _classify_phase_artifact_gap(art, phase)
    if phase_gap is not None:
        return phase_gap

    # (3b) Phase-tagged sdk_error before generic artifact checks — can fire
    # even when partial artifacts remain on disk from a prior run.
    if sdk_error:
        phase_outcome = _classify_sdk_phase_error(sdk_error, phase)
        if phase_outcome is not None:
            return phase_outcome

    # (4) MUST-have files on the quantized directory.
    if art.manifest_present and not art.manifest_parse_error:
        if not art.quantized_dir_exists or not art.has_weights:
            return OutcomeId.must_have_weights_missing
        if not art.has_config_json:
            return OutcomeId.must_have_config_missing_or_invalid
        if not art.has_tokenizer:
            return OutcomeId.must_have_tokenizer_missing

    # (5) Validator results.
    if art.manifest_present and not art.manifest_parse_error:
        if not art.validation_report_present:
            return OutcomeId.validation_report_absent
        steps = art.validation_steps
        if steps.md5 == "FAIL":
            return OutcomeId.must_validate_md5_mismatch
        if steps.config == "FAIL":
            return OutcomeId.must_validate_config_mismatch
        if steps.fuzzy == "FAIL":
            return OutcomeId.fuzzy_check_failed
        if steps.auxiliary == "FAIL":
            return OutcomeId.should_have_aux_missing
        if steps.md5 == "skipped" or steps.config == "skipped":
            # Tier mapping (strict/lenient) is the retry loop's job.
            return OutcomeId.must_validate_skipped

    # (6) Eval phase.
    eval_outcome = _classify_eval_outcome(art, acceptable_eval_gap=acceptable_eval_gap)
    if eval_outcome is not None:
        return eval_outcome

    # (7) Residual sdk_error with no specific bucket.
    if sdk_error:
        return OutcomeId.unclassified_failure

    # Clean success.
    return None


# Assessment assembly


def build_assessment(
    attempts: list[OutcomeId | None],
    *,
    workspace: Path,
    artifacts: CollectedArtifacts | None = None,
    notes: tuple[str, ...] = (),
) -> Assessment:
    """Assemble an ``Assessment`` from a chronological attempts list.

    * ``final`` = last attempt's outcome.
    * ``recovered`` = True iff len(attempts) > 1 AND final ∈ SUCCESS_TAGS.
    * ``eval_gap`` = ``relative_gap`` from the latest ``eval_report.json``.

    Args:
        attempts: Per-attempt outcomes in chronological order.
        workspace: Workspace directory used to collect artifacts.
        artifacts: Pre-scanned artifacts; supply to avoid a disk pass.
        notes: Extra human-readable notes to attach to the assessment.

    Returns:
        The assembled :class:`Assessment`.
    """

    if not attempts:
        raise ValueError("attempts must be non-empty")

    art = artifacts if artifacts is not None else collect_artifacts(Path(workspace))
    final = attempts[-1]
    recovered = len(attempts) > 1 and final in SUCCESS_TAGS

    eval_gap: float | None = None
    if isinstance(art.eval_report_data, dict):
        raw = art.eval_report_data.get("relative_gap")
        try:
            eval_gap = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            eval_gap = None

    return Assessment(
        final=final,
        attempts=tuple(attempts),
        recovered=recovered,
        eval_gap=eval_gap,
        notes=tuple(notes),
    )


def derive_status(assessment: Assessment, artifacts: CollectedArtifacts) -> str:
    """Map an ``Assessment`` to a public status string.

    The mapping keys off the ``AUTO_RECOVER`` / ``AUTO_FAIL`` / ``ASK``
    category sets in :mod:`.outcomes`; see the "Return shape" section of the
    agent's ``README.md``. Consumed by
    :class:`hyperloom.agents.quantization.driver.retry.QuantSkillRunResult`.

    Args:
        assessment: The assembled assessment to map.
        artifacts: Collected artifacts used for status demotion checks.

    Returns:
        One of ``"success"``, ``"partial"``, or ``"failed"``.
    """

    final = assessment.final
    # Clean / accepted success.
    if final is None or final == OutcomeId.eval_gap_accepted:
        return "success"

    # Auto-fail buckets always fail.
    if final in AUTO_FAIL:
        return "failed"

    # eval_gap_exceeded is partial: model usable, but quality bar not met.
    if final == OutcomeId.eval_gap_exceeded:
        return "partial"

    # must_validate_skipped — STRICT controls demotion.
    if final == OutcomeId.must_validate_skipped:
        return "failed" if artifacts.strict_validation else "partial"

    # Auto-recover outcomes reaching the final attempt are partial (model
    # usable, audit/eval chain incomplete) unless a MUST-have file is missing.
    if final in AUTO_RECOVER:
        if final in MUST_HAVE_RECOVERS_THAT_FAIL_WITHOUT_ARTIFACT and not (
            artifacts.has_config_json and artifacts.has_tokenizer
        ):
            return "failed"
        return "partial"

    if final in ASK or final == OutcomeId.unclassified_failure:
        return "failed"

    raise AssertionError(f"unhandled outcome: {final}")


__all__ = [
    "Assessment",
    "classify_attempt",
    "build_assessment",
    "derive_status",
    "AUTO_FAIL",
    "AUTO_RECOVER",
    "ASK",
    "ASK_RETRYABLE",
    "SUCCESS_TAGS",
]
