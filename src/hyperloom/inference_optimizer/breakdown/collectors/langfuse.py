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
       receipt with final counts (``receipt_source="receipt_file"``).
    2. Otherwise a live read of the per-session emitter singleton -- reports
       the gating + redacted config + in-process running counts
       (``receipt_source="live_emitter"``, ``counts_final=False``).

    Either way credentials are redacted to host + presence booleans. Never
    raises: any failure degrades to a minimal ``config_only`` view so the
    breakdown still records whether the feature was even on.

    Args:
        session_dir (Path): Absolute session root.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place on receipt
            / emitter read failures).

    Returns:
        dict[str, Any]: The ``langfuse`` section, tagged with a
        ``receipt_source`` of ``receipt_file`` / ``live_emitter`` /
        ``config_only`` depending on which tier resolved.
    """
    from hyperloom.orchestrator.trace import langfuse_emitter as lfe

    # Tier 1: the persisted post-flush receipt (final counts).
    try:
        receipt = lfe.read_receipt(session_dir)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"langfuse: read_receipt failed: {type(exc).__name__}: {exc}")
        receipt = None
    if receipt is not None:
        receipt["receipt_source"] = "receipt_file"
        return receipt

    # Tier 2: live read of the emitter singleton (pre-flush / in-process).
    try:
        section = lfe.get_emitter(session_dir).receipt()
        section["receipt_source"] = "live_emitter"
        return section
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"langfuse: live receipt failed: {type(exc).__name__}: {exc}")

    # Tier 3 fallback: config-only view straight from env + manifest, so the
    # breakdown still records whether the feature was configured at all.
    from hyperloom.orchestrator.trace import trace_env as tenv

    creds = tenv.langfuse_credentials()
    return {
        "enabled": False,
        "disabled_reason": "unknown",
        "config": {
            "enable_flag": tenv.langfuse_live_enabled(),
            "host": creds.get(tenv.ENV_LANGFUSE_HOST),
            "public_key_set": tenv.ENV_LANGFUSE_PUBLIC_KEY in creds,
            "secret_key_set": tenv.ENV_LANGFUSE_SECRET_KEY in creds,
            "sdk_available": None,
        },
        "trace_id": None,
        "session_id": str(manifest.get("claw_session_id") or manifest.get("session_id") or ""),
        "correlated_on": (
            "claw_session_id" if str(manifest.get("claw_session_id") or "").strip() else "internal_session_id"
        ),
        "counts": {},
        "counts_final": False,
        "receipt_source": "config_only",
    }
