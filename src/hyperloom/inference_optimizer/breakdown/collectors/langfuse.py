# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``langfuse`` section: whether the trace was pushed live, and how much.

Authored by the emitter, not by this export, which is why it is read rather
than derived: the receipt is a different process's record of what it sent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def collect_langfuse(
    session_dir: Path,
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the ``langfuse`` section: was the trace pushed live, and how much.

    Two-tier source (the breakdown is normally written *before* the
    session-end ``flush_session``, so the on-disk receipt may not exist yet):

    1. ``reports/trace/langfuse_receipt.json`` if present -- the post-flush
       receipt with final counts.
    2. Otherwise a live read of the per-session emitter singleton -- the gating
       and the in-process running counts.

    Never raises: any failure degrades to a disabled view carrying only the
    session id.

    Args:
        session_dir (Path): Absolute session root.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place on receipt
            / emitter read failures).

    Returns:
        dict[str, Any]: The ``langfuse`` section.
    """
    from hyperloom.inference_optimizer.trace import langfuse_emitter as lfe

    # Tier 1: the persisted post-flush receipt (final counts).
    try:
        receipt = lfe.read_receipt(session_dir)
    except ValueError as exc:
        warnings.append(f"langfuse: read_receipt failed: {type(exc).__name__}: {exc}")
        receipt = None
    if receipt is not None:
        return receipt

    # Tier 2: live read of the emitter singleton (pre-flush / in-process).
    try:
        return lfe.get_emitter(session_dir).receipt()
    except (OSError, ValueError) as exc:
        warnings.append(f"langfuse: live receipt failed: {type(exc).__name__}: {exc}")

    return {
        "enabled": False,
        "disabled_reason": "unknown",
        "session_id": str(manifest.get("claw_session_id") or manifest.get("session_id") or ""),
    }
