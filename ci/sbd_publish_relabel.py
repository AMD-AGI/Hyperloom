#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ci/sbd_publish_relabel.py — prepare a session_breakdown for the sbd-ingester.

The optimize-submit workflow publishes every collected ``session_breakdown.json``
to the sbd-ingester so pulse never leaves a model stuck in ``awaiting_collector``.
Before sending, the payload is:

1. migrated to the v2 schema (via :mod:`import_session_breakdown`) when it is
   still in the legacy v1-flat or universal shape,
2. stamped with ``schema_version`` and the task/claw/image identifiers, and
3. withheld from the leaderboard when it is a known failure/zero-gain terminal
   state, by setting ``show_on_leaderboard = False`` so the ingester does not
   surface a structurally-valid failure breakdown as a leaderboard success.

This logic used to live inline in ``optimize-submit.yml``; it is extracted here
so the v2 field-path handling and the withholding rule can be unit-tested.

NOTE: withholding only takes effect if the sbd-ingester honors a client-provided
``show_on_leaderboard=False`` instead of recomputing it server-side. Confirm that
contract against the running ingester before relying on this for prod.
"""

from __future__ import annotations

import json
import sys
from typing import Any

SCHEMA_VERSION = "hyperloom.session_breakdown.v1.1"

# Terminal states that can emit a structurally valid breakdown with gain=0. They
# stay ingested for auditability but must not surface as leaderboard successes.
WITHHELD_STATES = {
    "failed",
    "baseline_failed",
    "signal",
    "robustness_escalated",
    "model_config_incompatible",
}


def _float_value(value: Any) -> float:
    """Coerce ``value`` to float, treating None/blank/garbage as 0.0."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _as_dict(value: Any) -> dict:
    """Return ``value`` if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def resolve_stop_reason(data: dict) -> str:
    """Resolve the terminal status across v1-flat and v2 breakdown shapes.

    v1-flat carried ``status`` / ``stop_reason`` at the top level; v2 nests the
    stop reason under ``session.stop_reason`` (e.g. ``robustness_escalated``).

    Args:
        data: The (possibly already migrated) breakdown dict.

    Returns:
        The lowercased, stripped stop reason, or ``""`` if none is present.
    """
    session = _as_dict(data.get("session"))
    return (
        str(
            data.get("status")
            or data.get("stop_reason")
            or session.get("stop_reason")
            or ""
        )
        .strip()
        .lower()
    )


def resolve_gain(data: dict) -> float:
    """Resolve the validated cumulative gain across breakdown shapes.

    v2 stores the validated gain at ``final.cumulative_gain_pct_validated``;
    older payloads used the top-level ``gain_pct_sum`` or ``gain``.

    Args:
        data: The (possibly already migrated) breakdown dict.

    Returns:
        The validated gain as a float (0.0 when absent/unparsable).
    """
    final = _as_dict(data.get("final"))
    validated = final.get("cumulative_gain_pct_validated")
    if validated is not None:
        return _float_value(validated)
    if data.get("gain_pct_sum") is not None:
        return _float_value(data.get("gain_pct_sum"))
    return _float_value(data.get("gain"))


def apply_leaderboard_withholding(data: dict) -> dict:
    """Withhold known failure/zero-gain breakdowns from the leaderboard.

    A breakdown is withheld only when its terminal state is in
    :data:`WITHHELD_STATES` AND its validated gain is ``<= 0``. This keeps
    ``robustness_escalated`` sessions that still produced a real positive gain on
    the leaderboard, consistent with historical data.

    Args:
        data: The breakdown dict, mutated in place.

    Returns:
        The same ``data`` dict.
    """
    status = resolve_stop_reason(data)
    gain = resolve_gain(data)
    if status in WITHHELD_STATES and gain <= 0:
        data["show_on_leaderboard"] = False
        data["leaderboard_withheld_reason"] = status or "runtime_invalid"
    return data


def relabel(
    data: dict,
    *,
    task_id: str = "",
    claw_session_id: str = "",
    image: str = "",
    isb: Any = None,
) -> dict:
    """Migrate, stamp identifiers, and apply leaderboard withholding.

    Args:
        data: The raw breakdown dict.
        task_id: Task id to backfill onto the breakdown / session metadata.
        claw_session_id: Claw session id to backfill (also as ``session_id``).
        image: Server image string to backfill.
        isb: The ``import_session_breakdown`` module (optional); when provided,
            legacy schemas are migrated to v2 before withholding is evaluated.

    Returns:
        The migrated, stamped, possibly-withheld breakdown dict.
    """
    if isb is not None:
        if isb.looks_like_v1_flat_schema(data):
            data = isb.migrate_v1_to_v2(data)
        elif isb.looks_like_universal_schema(data) and not (
            isinstance(data.get("workload"), dict)
            and isinstance(data.get("baseline"), dict)
            and isinstance(data.get("final"), dict)
        ):
            migrated = isb.migrate_universal_to_v2(data)
            if migrated:
                data = migrated

    data["schema_version"] = SCHEMA_VERSION

    session_meta = data.setdefault("session_meta", {})
    session = data.setdefault("session", {})
    for obj in (data, session_meta, session):
        if task_id:
            obj.setdefault("task_id", task_id)
        if claw_session_id:
            obj.setdefault("claw_session_id", claw_session_id)
        if image:
            obj.setdefault("image", image)
    if claw_session_id:
        session.setdefault("session_id", claw_session_id)
        session_meta.setdefault("session_id", claw_session_id)

    # Withholding must run AFTER migration so it reads the v2 field paths.
    apply_leaderboard_withholding(data)
    return data


def _load_isb() -> Any:
    """Import ``import_session_breakdown`` if available, else return None."""
    sys.path.insert(0, "ci")
    try:
        import import_session_breakdown as isb  # noqa: WPS433 (lazy import by design)

        return isb
    except Exception:  # pragma: no cover - import shape varies by checkout
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI: ``sbd_publish_relabel.py SRC DST SELECTION_JSON``.

    Reads the breakdown ``SRC`` and the publish ``SELECTION_JSON`` (for
    task/claw/image identifiers), relabels, and writes the result to ``DST``.

    Returns:
        Process exit code (0 on success).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    src, dst, selection_path = argv[:3]
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    with open(selection_path, encoding="utf-8") as f:
        selection = json.load(f)

    data = relabel(
        data,
        task_id=selection.get("task_id") or "",
        claw_session_id=selection.get("ci_claw_session_id") or "",
        image=selection.get("ci_image") or "",
        isb=_load_isb(),
    )

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
