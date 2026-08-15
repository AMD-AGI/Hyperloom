# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure helpers for structured variant failure evidence.

Imported by both the executor layer and the Coordinator; must not import
SharedState or any orchestrator component that touches the session.
"""

from __future__ import annotations

import re
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

FAILURE_STAGE_WARMUP: str = "warmup"
FAILURE_STAGE_DECISION: str = "decision"

# Variant outcomes that produced no usable measurement, so they need evidence.
UNMEASURED_OUTCOMES: frozenset[str] = frozenset({"FAILED", "KILLED_OVERTIME"})

# Matches every character replaced by ``_`` in a variant-name slug.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def tail_excerpt(value: Any, *, limit: int = 1200) -> str | None:
    """Return the trailing ``limit`` characters of ``value`` after redaction.

    Tail, not head: a boot crash puts its assertion at the end of the blob.

    Args:
        value: Raw text; falsy inputs return ``None``.
        limit: Maximum retained trailing character count.

    Returns:
        Redacted tail, or ``None`` when ``value`` is falsy.
    """
    if value is None:
        return None
    text = redact_secret_values(str(value))
    if not text:
        return None
    return text[-limit:] if len(text) > limit else text


def make_failure_id(*, task_id: str, fingerprint: str, variant_name: str = "") -> str:
    """Compute a failure id; must stay recomputable from the same inputs.

    Args:
        task_id: The owning task's id.
        fingerprint: The variant's canonical fingerprint.
        variant_name: Used as fallback when ``fingerprint`` is empty.

    Returns:
        ``fail.<task_id>.<key>`` where ``key`` is the first 12 characters of
        ``fingerprint`` or a slug of ``variant_name``.
    """
    fp = (fingerprint or "").strip()
    key = fp[:12] if fp else _SLUG_RE.sub("_", (variant_name or "unknown"))[:12]
    return f"fail.{task_id}.{key}"


def failure_from_variant_outcome(
    *,
    task_id: str,
    round_id: str,
    vo: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure evidence packet from a per-variant-outcome row.

    Args:
        task_id: The owning task's id.
        round_id: The explore round id.
        vo: One entry from ``per_variant_outcomes``.

    Returns:
        A packet dict with all evidence fields populated from ``vo``.
    """
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
        "error_excerpt": vo.get("error_excerpt") or "",
        "reason": str(vo.get("reason") or ""),
        "server_log_path": vo.get("server_log_path"),
        "workspace": vo.get("workspace"),
        "raw_result_path": vo.get("raw_result_path"),
        "variant": {
            "extra_server_args": str(variant.get("extra_server_args") or ""),
            "extra_envs": dict(variant.get("extra_envs") or {}),
            "note": str(variant.get("note") or ""),
        },
    }


def render_failure_line(fe: dict[str, Any], *, excerpt_chars: int = 160) -> str:
    """Format one failure evidence packet as a compact single line.

    Args:
        fe: A failure evidence dict as produced by :func:`failure_from_variant_outcome`.
        excerpt_chars: Maximum characters shown from the error body.

    Returns:
        A single-line summary string.
    """
    error_class = str(fe.get("error_class") or "")
    body = str(fe.get("error_excerpt") or fe.get("reason") or "")
    parts = [
        f"fid={fe.get('failure_id') or ''}",
        f"variant={str(fe.get('variant_name') or '')!r}",
        f"stage={fe.get('stage') or ''}",
    ]
    if error_class:
        parts.append(f"err={error_class}")
    if body:
        parts.append(f"msg={body[-excerpt_chars:]!r}")
    return " ".join(parts)
