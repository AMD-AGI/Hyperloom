# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure helpers for structured variant failure evidence."""

from __future__ import annotations

import re
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

FAILURE_STAGE_WARMUP: str = "warmup"
FAILURE_STAGE_DECISION: str = "decision"

FAILURE_ATTRIBUTION_CANDIDATE: str = "candidate_caused"
FAILURE_ATTRIBUTION_ENVIRONMENT: str = "environment"
FAILURE_ATTRIBUTION_HARNESS: str = "harness"
FAILURE_ATTRIBUTION_UNKNOWN: str = "unknown"
FAILURE_ATTRIBUTIONS: frozenset[str] = frozenset(
    {
        FAILURE_ATTRIBUTION_CANDIDATE,
        FAILURE_ATTRIBUTION_ENVIRONMENT,
        FAILURE_ATTRIBUTION_HARNESS,
        FAILURE_ATTRIBUTION_UNKNOWN,
    }
)

# Variant outcomes that produced no usable measurement, so they need evidence.
# ``RECORDED`` is deliberately absent: it was measured, just not promoted.
UNMEASURED_OUTCOMES: frozenset[str] = frozenset({"FAILED", "KILLED_OVERTIME"})

# Matches every character replaced by ``_`` in a variant-name slug.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")
_CANDIDATE_ERROR_CLASSES = frozenset(
    {
        "capability_unsupported",
        "invalid_argument",
        "unsupported_config",
    }
)
_ENVIRONMENT_ERROR_CLASSES = frozenset({"mn_server_restart_failed"})
_HARNESS_ERROR_CLASSES = frozenset(
    {
        "benchmark_report_invalid_metric",
        "benchmark_report_missing",
        "eval_probe_unpatchable",
        "magpie_nonzero_after_valid_measurement",
        "magpie_nonzero_invalid_measurement",
        "magpie_timeout",
        "no_benchmark_workspace",
        "warmup_magpie_timeout",
        "warmup_round_failed",
        "warmup_yaml_build_error",
        "yaml_build_error",
    }
)
_ENVIRONMENT_EVIDENCE_RE = re.compile(
    r"(?:concurrent|foreign|orphan(?:ed)?|leak(?:ed)?)\s+(?:gpu\s+)?(?:holder|process|engine)"
    r"|card\s+(?:is\s+)?(?:not\s+)?(?:clean|idle)"
    r"|(?:gpu|device)\s+(?:is\s+)?busy"
    r"|address\s+already\s+in\s+use"
    r"|temporary\s+failure\s+in\s+name\s+resolution",
    re.IGNORECASE,
)
_CANDIDATE_EVIDENCE_RE = re.compile(
    r"unrecogni[sz]ed\s+(?:argument|option)"
    r"|unknown\s+(?:argument|option)"
    r"|invalid\s+choice"
    r"|unsupported\s+(?:argument|backend|configuration|dtype|option)",
    re.IGNORECASE,
)


def tail_excerpt(value: Any, *, limit: int = 1200) -> str | None:
    """Return the trailing ``limit`` characters of ``value`` after redaction."""
    if value is None:
        return None
    text = redact_secret_values(str(value))
    if not text:
        return None
    return text[-limit:] if len(text) > limit else text


def make_failure_id(*, task_id: str, fingerprint: str, variant_name: str = "") -> str:
    """Compute a failure id; must stay recomputable from the same inputs."""
    fp = (fingerprint or "").strip()
    key = fp[:12] if fp else _SLUG_RE.sub("_", (variant_name or "unknown"))[:12]
    return f"fail.{task_id}.{key}"


def classify_failure_attribution(
    *,
    error_class: Any,
    error_excerpt: Any = "",
    reason: Any = "",
    explicit: Any = "",
) -> str:
    """Classify whether an unmeasured failure says anything about the candidate.

    Classification is deliberately conservative. Generic launch, timeout, and
    benchmark failures remain outside the Experience corpus unless the producer
    supplies candidate-specific evidence. The full failure still remains in
    SBD V6 regardless of this result.
    """
    authored = str(explicit or "").strip().lower()
    if authored in FAILURE_ATTRIBUTIONS:
        return authored

    kind = str(error_class or "").strip().lower()
    evidence = f"{reason or ''}\n{error_excerpt or ''}"
    if kind in _ENVIRONMENT_ERROR_CLASSES or _ENVIRONMENT_EVIDENCE_RE.search(evidence):
        return FAILURE_ATTRIBUTION_ENVIRONMENT
    if kind in _CANDIDATE_ERROR_CLASSES or _CANDIDATE_EVIDENCE_RE.search(evidence):
        return FAILURE_ATTRIBUTION_CANDIDATE
    if kind in _HARNESS_ERROR_CLASSES:
        return FAILURE_ATTRIBUTION_HARNESS
    return FAILURE_ATTRIBUTION_UNKNOWN


def failure_from_variant_outcome(
    *,
    task_id: str,
    round_id: str,
    vo: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure evidence packet from a per-variant-outcome row."""
    fp = str(vo.get("fingerprint") or "")
    variant_name = str(vo.get("variant_name") or "")
    variant = vo.get("variant") or {}
    return {
        "failure_id": make_failure_id(task_id=task_id, fingerprint=fp, variant_name=variant_name),
        "task_id": task_id,
        "round_id": round_id,
        "variant_name": variant_name,
        "fingerprint": fp,
        "stage": str(vo.get("stage") or FAILURE_STAGE_DECISION),
        "outcome": str(vo.get("outcome") or ""),
        "error_class": str(vo.get("error_class") or ""),
        "attribution": classify_failure_attribution(
            error_class=vo.get("error_class"),
            error_excerpt=vo.get("error_excerpt"),
            reason=vo.get("reason"),
            explicit=vo.get("failure_attribution"),
        ),
        "error_excerpt": vo.get("error_excerpt") or "",
        "reason": str(vo.get("reason") or ""),
        "server_log_path": vo.get("server_log_path"),
        "workspace": vo.get("workspace"),
        "raw_result_path": vo.get("raw_result_path"),
        "variant": {
            "extra_server_args": str(variant.get("extra_server_args") or ""),
            "extra_envs": dict(variant.get("extra_envs") or {}),
            "note": str(variant.get("note") or ""),
            "reasoning_origin": str(variant.get("reasoning_origin") or ""),
        },
    }


def render_failure_line(fe: dict[str, Any], *, excerpt_chars: int = 160) -> str:
    """Format one failure evidence packet as a compact single line."""
    error_class = str(fe.get("error_class") or "")
    body = str(fe.get("error_excerpt") or fe.get("reason") or "")
    parts = [
        f"fid={fe.get('failure_id') or ''}",
        f"variant={str(fe.get('variant_name') or '')!r}",
        f"stage={fe.get('stage') or ''}",
    ]
    if error_class:
        parts.append(f"err={error_class}")
    attribution = str(fe.get("attribution") or "")
    if attribution:
        parts.append(f"attribution={attribution}")
    if body:
        parts.append(f"msg={body[-excerpt_chars:]!r}")
    return " ".join(parts)
