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
FAILURE_STAGES: frozenset[str] = frozenset({FAILURE_STAGE_WARMUP, FAILURE_STAGE_DECISION})

# Characters kept when building a variant-name slug for the fallback id.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def tail_excerpt(value: Any, *, limit: int = 1200) -> str | None:
    """Return the trailing ``limit`` characters of ``value`` after redaction.

    Boot-time crashes put the assertion at the end of the log blob; taking the
    tail retains the root cause where a head truncation would drop it.

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
    """Compute a stable, recomputable failure id.

    Both the executor and the Coordinator call this with the same arguments;
    the result must be identical on both sides.

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
    failure_id = make_failure_id(task_id=task_id, fingerprint=fp, variant_name=variant_name)
    variant = vo.get("variant") or {}
    return {
        "failure_id": failure_id,
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
    fid = str(fe.get("failure_id") or "")
    variant = str(fe.get("variant_name") or "")
    stage = str(fe.get("stage") or "")
    ec = str(fe.get("error_class") or "")
    # Prefer tail-truncated error body; fall back to the reason tag.
    body = str(fe.get("error_excerpt") or fe.get("reason") or "")
    if len(body) > excerpt_chars:
        body = body[-excerpt_chars:]
    parts = [f"fid={fid}", f"variant={variant!r}", f"stage={stage}"]
    if ec:
        parts.append(f"err={ec}")
    if body:
        parts.append(f"msg={body!r}")
    return " ".join(parts)
