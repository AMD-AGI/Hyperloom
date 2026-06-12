"""Eval-gap policy: threshold resolution + acceptance check.

``quark-torch-llm-eval`` runs one model at a time and only emits Markdown — there is
no JSON sidecar and no built-in source-vs-quantized comparison. So SKILL.md is
instructed to invoke ``quark-torch-llm-eval`` twice (once on source, once on
quantized), parse the headline metric from both reports, and synthesize an
agent-owned ``eval_report.json`` wrapper:

    {
        "metric_name": "gsm8k",
        "dataset": "gsm8k",
        "backend": "vllm",
        "source_score": 0.512,
        "quantized_score": 0.498,
        "relative_gap": 0.0273,
    }

This module reads that file, resolves the acceptance threshold, and decides
whether the gap is within budget. It does **not** parse the raw Markdown — the
LLM is responsible for normalizing scores into the schema above. We trust the
schema and validate keys; if anything is missing the classifier maps the
attempt to ``eval_env_unavailable`` (the closest "eval didn't really finish"
bucket) rather than silently treating absent fields as zero.

Threshold resolution priority (§3.1):

    1. ``acceptable_eval_gap`` Python arg (caller-supplied; not None)
    2. ``<workspace>/eval_gap_threshold.txt`` (LLM writes this when the
       prompt mentions a tolerance)
    3. Default ``0.03`` (3%)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACCEPTABLE_GAP = 0.03

_REQUIRED_EVAL_KEYS = ("source_score", "quantized_score", "relative_gap")


@dataclass(frozen=True)
class EvalDecision:
    """Outcome of comparing ``relative_gap`` to the acceptance threshold.

    * ``status == "within"``: gap ≤ threshold — counts as success. Caller may
      attach ``eval_gap_accepted`` narrative tag when a prior attempt
      exceeded the threshold and the user accepted it.
    * ``status == "exceeded"``: gap > threshold — classifier maps to #21
      ``eval_gap_exceeded``.
    * ``status == "missing"``: eval_report.json absent or malformed — handled
      by the classifier upstream (it inspects ``eval_skipped.txt`` first).
    """

    status: str  # "within" | "exceeded" | "missing"
    relative_gap: float | None
    threshold: float
    threshold_source: str  # "arg" | "file" | "default"


def resolve_threshold(
    workspace: Path,
    *,
    acceptable_eval_gap: float | None,
) -> tuple[float, str]:
    """Return ``(threshold, source)`` per §3.1 priority chain."""

    if acceptable_eval_gap is not None:
        return float(acceptable_eval_gap), "arg"
    threshold_file = workspace / "eval_gap_threshold.txt"
    if threshold_file.is_file():
        raw = threshold_file.read_text(encoding="utf-8").strip()
        if raw:
            try:
                return float(raw), "file"
            except ValueError:
                # Malformed file falls through to default rather than raising —
                # SKILL.md may have written a stray comment; default is safe.
                pass
    return DEFAULT_ACCEPTABLE_GAP, "default"


def decide(
    eval_report: dict | None,
    *,
    workspace: Path,
    acceptable_eval_gap: float | None,
) -> EvalDecision:
    """Decide whether an evaluation report passes the quality gap threshold.

    Args:
        eval_report: Parsed evaluation report, or ``None`` when absent.
        workspace: Run workspace, used to resolve a per-run gap threshold file.
        acceptable_eval_gap: Explicit maximum relative gap; falls back to the
            workspace threshold or the default when ``None``.

    Returns:
        An :class:`EvalDecision` describing the status (``missing``,
        ``within``, or ``exceeded``), the relative gap, and the threshold used.
    """
    threshold, source = resolve_threshold(
        workspace, acceptable_eval_gap=acceptable_eval_gap
    )

    if not isinstance(eval_report, dict):
        return EvalDecision(
            status="missing",
            relative_gap=None,
            threshold=threshold,
            threshold_source=source,
        )

    missing = [k for k in _REQUIRED_EVAL_KEYS if k not in eval_report]
    if missing:
        return EvalDecision(
            status="missing",
            relative_gap=None,
            threshold=threshold,
            threshold_source=source,
        )

    try:
        gap = float(eval_report["relative_gap"])
    except (TypeError, ValueError):
        return EvalDecision(
            status="missing",
            relative_gap=None,
            threshold=threshold,
            threshold_source=source,
        )

    status = "within" if gap <= threshold else "exceeded"
    return EvalDecision(
        status=status,
        relative_gap=gap,
        threshold=threshold,
        threshold_source=source,
    )


__all__ = [
    "DEFAULT_ACCEPTABLE_GAP",
    "EvalDecision",
    "decide",
    "resolve_threshold",
]
