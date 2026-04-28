"""Shared parsing / env helpers used by multiple executors."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..intent_parser import Intent, IntentType


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON file helpers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> dict[str, Any] | None:
    """Tolerant JSON reader; returns ``None`` when the file is missing
    or unparseable. Caller decides whether that's fatal."""
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to parse %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Result-file discovery
# ---------------------------------------------------------------------------
def find_first(result_dir: Path, *globs: str) -> Path | None:
    """Return the first matching file across ``*globs`` (in order)."""
    for pattern in globs:
        for hit in sorted(Path(result_dir).rglob(pattern)):
            if hit.is_file():
                return hit
    return None


def parse_serving_metrics(path: Path) -> dict[str, float]:
    """Parse one ``benchmark_serving`` JSON output.

    Recognised keys (subset):

        output_throughput     tok/s aggregate
        total_token_throughput tok/s
        mean_tpot_ms          mean time-per-output-token
        mean_ttft_ms          time-to-first-token
        mean_itl_ms           inter-token-latency
        mean_e2el_ms          end-to-end latency
        completed             #completed prompts
        num_prompts           total prompts
    """
    data = read_json(path) or {}
    keys = (
        "output_throughput", "total_token_throughput",
        "mean_tpot_ms", "mean_ttft_ms", "mean_itl_ms", "mean_e2el_ms",
        "completed", "num_prompts",
    )
    out: dict[str, float] = {}
    for k in keys:
        v = data.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def parse_eval_summary(path: Path) -> dict[str, float]:
    """Parse a ``eval_summary_<task>.json`` file written by
    ``eval_accuracy.sh`` (lm-eval-harness export). Returns the highest
    available accuracy metric in [0, 1] under key ``accuracy`` plus the
    raw scores dict under ``raw``.
    """
    data = read_json(path) or {}
    scores = data.get("scores", {}) or {}
    if not isinstance(scores, dict) or not scores:
        # Allow flat ``{"score": 0.71}`` form (test fixture).
        flat = data.get("score")
        if isinstance(flat, (int, float)):
            return {"accuracy": float(flat), "raw": flat}
        return {}
    # Pick the first task's strict-match score, else first numeric value.
    candidates = ("exact_match,strict-match", "exact_match,flexible-extract",
                  "acc,none", "exact_match,none")
    for _task_name, row in scores.items():
        if not isinstance(row, dict):
            continue
        for k in candidates:
            v = row.get(k)
            if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
                return {"accuracy": float(v), "raw": row}
        # Fallback: first numeric field.
        for k, v in row.items():
            if k != "task" and isinstance(v, (int, float)):
                return {"accuracy": float(v), "raw": row}
    return {}


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------
def update_state_intent(changes: dict[str, Any], *, rationale: str = "") -> Intent:
    """Build an ``update_state`` intent ready to send to the bus."""
    payload: dict[str, Any] = {"changes": dict(changes)}
    if rationale:
        payload["rationale"] = rationale
    return Intent(type=IntentType.UPDATE_STATE, payload=payload)


def send_message_intent(
    *,
    topic: str,
    body_md: str,
    to: str = "*",
    priority: int = 1,
    extras: dict[str, Any] | None = None,
) -> Intent:
    payload: dict[str, Any] = {
        "to": to, "topic": topic, "body_md": body_md, "priority": priority,
    }
    if extras:
        payload.update(extras)
    return Intent(type=IntentType.SEND_MESSAGE, payload=payload)


# ---------------------------------------------------------------------------
# Env merging
# ---------------------------------------------------------------------------
def merged_env(
    parent: dict[str, str] | None,
    *overrides: dict[str, Any],
) -> dict[str, str]:
    """Return ``parent`` merged with each override dict, stringifying
    values. Used to build the env passed to subprocess.

    Parent ``None`` → start from os.environ (best-effort cross-platform).
    """
    import os
    out = dict(parent if parent is not None else os.environ)
    for o in overrides:
        if not o:
            continue
        for k, v in o.items():
            if v is None:
                continue
            out[str(k)] = str(v)
    return out


__all__ = [
    "find_first",
    "merged_env",
    "parse_eval_summary",
    "parse_serving_metrics",
    "read_json",
    "send_message_intent",
    "update_state_intent",
]
