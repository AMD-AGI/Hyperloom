# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Worktree git helpers for patch-authoring specialists.

Migrated out of the retired ``dynamic_action_tools`` module. Exposes the
worktree-scoped ``git`` helpers (self-check apply, cumulative-diff capture,
hard reset) used by :class:`SpecialistRunner`.

The legacy ``run_bench`` in-loop micro-bench tool (and its capped/stubbed
``BENCH_REGISTRY`` whitelist) has been removed: GPU specialists now run real
serving + benchmark / autotune loops on their own leased cards (see
``specialist_rebench`` and the opened-up iron rules), so a capped, serving-
forbidden micro-bench box is no longer the validation path.
"""

from __future__ import annotations

from typing import Any


def _error(reason: str, **extra: Any) -> dict[str, Any]:
    """Build a failure result envelope.

    Args:
        reason: Human-readable failure reason.
        **extra: Additional fields to merge into the envelope.

    Returns:
        Dict with ``ok=False`` plus the reason and any extra fields.
    """
    return {"ok": False, "reason": reason, **extra}


def _ok(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a success result envelope.

    Args:
        payload: Optional fields to merge into the envelope.

    Returns:
        Dict with ``ok=True`` plus any payload fields.
    """
    out: dict[str, Any] = {"ok": True}
    if payload:
        out.update(payload)
    return out


__all__: list[str] = []
