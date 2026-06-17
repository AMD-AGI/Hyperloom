# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Standalone collectors for *legacy* (pre-``phase_history``) sessions.

v0.6-era hyperloom sessions never wrote ``state.phase_history`` nor the
``reports/optimization_journal.json`` ledger, so the main-line
:func:`collectors.collect_phase_segments` returns ``[]`` for them and
the dashboard has no phase view at all.

Every v0.6 session *does* carry the per-action audit lists
(``baseline_attempts`` / ``profile_attempts`` / ``params_attempts`` /
``backends_attempts`` / ``validate_stack_attempts`` / ``sweep_attempts``
/ ``explore_attempts``) plus the adopted ``optimization_stack`` and the
``kernel_opt_attempts`` / ``kernel_integrate_attempts`` maps. This module
reconstructs an equivalent ``phase_segments`` (and, when present,
kernel invocation) view purely from those fields, mapping the old action
vocabulary onto the canonical phase names used by the v2 pipeline:
``PRELUDE / FRAMEWORK_PR / EXPLORE / KERNEL / SWEEP / CLOSE``.

It is kept deliberately separate from ``collectors.py`` so the main
pipeline carries no legacy fallback branches. Callers detect a legacy
session with :func:`is_legacy_session` and, when true, prefer the output
of :func:`collect_phase_segments` here.
"""

from __future__ import annotations

from typing import Any

from .collectors import _iso_z, _parse_iso_unix, _to_float

# Old v0.6 action vocabulary -> canonical v2 phase (``validate_stack`` is phase-neutral; see ``_PHASE_NEUTRAL``).
_ACTION_PHASE: dict[str, str] = {
    "baseline":       "PRELUDE",
    "profile":        "PRELUDE",
    "explore":        "EXPLORE",
    "params":         "EXPLORE",
    "backends":       "EXPLORE",
    "sweep":          "SWEEP",
    "select_kernels": "KERNEL",
    "kernel_opt":     "KERNEL",
    "integrate":      "KERNEL",
}

# Actions that inherit the active phase rather than open a new segment.
_PHASE_NEUTRAL = frozenset({"validate_stack"})

_DEFAULT_PHASE = "EXPLORE"


def is_legacy_session(state: dict[str, Any]) -> bool:
    """A session is *legacy* when it never recorded ``phase_history``.

    Args:
        state: Session state mapping to inspect.

    Returns:
        ``True`` when the session has no ``phase_history``, else ``False``.
    """
    history = state.get("phase_history")
    return not (isinstance(history, list) and history)


def _phase_for_event(action: str, prev_phase: str) -> str:
    """Determine the phase associated with an action event.

    Args:
        action: The action name from the event.
        prev_phase: The phase carried over from the previous event.

    Returns:
        The mapped phase, the previous phase for phase-neutral actions,
        or the default phase as a fallback.
    """
    if action in _PHASE_NEUTRAL:
        return prev_phase or _DEFAULT_PHASE
    return _ACTION_PHASE.get(action, _DEFAULT_PHASE)


def _stack_adoptions(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract timestamped optimization-stack adoption entries.

    Args:
        state: Session state mapping holding ``optimization_stack``.

    Returns:
        The stack entries that are dicts carrying a ``ts`` timestamp.
    """
    stack = state.get("optimization_stack") or []
    out: list[dict[str, Any]] = []
    if isinstance(stack, list):
        for e in stack:
            if isinstance(e, dict) and e.get("ts"):
                out.append(e)
    return out


def collect_phase_segments(
    state: dict[str, Any],
    phase_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Reconstruct ``phase_segments`` for a legacy session.

    Maps ``phase_timeline`` events to canonical phases and collapses
    consecutive same-phase events into segments matching the v2 collector
    wire shape; ``evidence.reconstructed_from == "legacy_audit_lists"``
    flags each as a derived view.

    Args:
        state: Session state mapping holding the optimization stack.
        phase_timeline: Timestamped phase/action events to segment.
        warnings: Mutable list to which any reconstruction warnings are
            appended.

    Returns:
        The reconstructed ``phase_segments`` list in v2 collector shape.
    """
    events = [e for e in (phase_timeline or []) if isinstance(e, dict) and e.get("ts")]
    events.sort(key=lambda e: str(e.get("ts") or ""))
    if not events:
        return []

    groups: list[dict[str, Any]] = []
    prev_phase = ""
    for ev in events:
        phase = _phase_for_event(str(ev.get("action") or ev.get("change") or ""), prev_phase)
        prev_phase = phase
        if groups and groups[-1]["phase"] == phase:
            groups[-1]["actions"].append(ev)
        else:
            groups.append({"phase": phase, "actions": [ev]})

    adoptions = _stack_adoptions(state)

    segments: list[dict[str, Any]] = []
    for i, g in enumerate(groups):
        acts = g["actions"]
        entered_ts = _iso_z(acts[0].get("ts"))
        entered_unix = _parse_iso_unix(acts[0].get("ts"))
        exit_ts = ""
        exit_unix: float | None = None
        if i + 1 < len(groups):
            nxt0 = groups[i + 1]["actions"][0]
            exit_ts = _iso_z(nxt0.get("ts"))
            exit_unix = _parse_iso_unix(nxt0.get("ts"))
        elapsed: float | None = None
        if entered_unix is not None and exit_unix is not None:
            elapsed = max(0.0, float(exit_unix) - float(entered_unix))

        best_gain: float | None = None
        for ev in acts:
            ex = ev.get("extras") or {}
            # Only treat ``key_metric`` as a gain when its kind says so;
            # baseline/profile carry raw output_throughput, not a gain pct.
            g_pct = (
                _to_float(ev.get("key_metric"))
                if ev.get("key_metric_kind") in ("gain_pct", "validated_gain_pct")
                else None
            )
            if g_pct is None:
                g_pct = (
                    _to_float(ex.get("gain_pct"))
                    or _to_float(ex.get("best_gain_pct_vs_base"))
                )
            if g_pct is not None and (best_gain is None or g_pct > best_gain):
                best_gain = g_pct
        adopted_here = [
            {
                "action": a.get("action"),
                "variant_name": a.get("variant_name"),
                # Canonical field; v0.6 raw state only has the pre-rename
                # ``candidate_extra_sglang_args``, so fall back to it.
                "extra_server_args": (
                    a.get("candidate_extra_server_args")
                    or a.get("candidate_extra_sglang_args")
                ),
                "tput": _to_float(a.get("tput")),
                "ts": _iso_z(a.get("ts")),
            }
            for a in adoptions
            if (not exit_ts or _iso_z(a.get("ts")) < exit_ts)
            and _iso_z(a.get("ts")) >= entered_ts
        ]
        evidence: dict[str, Any] = {"reconstructed_from": "legacy_audit_lists"}
        if best_gain is not None:
            evidence["best_gain_pct"] = best_gain
        if adopted_here:
            evidence["adopted"] = adopted_here

        segments.append({
            "phase":           g["phase"],
            "from_phase":      segments[-1]["phase"] if segments else "",
            "entered_ts":      entered_ts,
            "entered_unix":    entered_unix,
            "exit_ts":         exit_ts,
            "exit_unix":       exit_unix,
            "exit_reason":     "",
            "evidence":        evidence,
            "events":          [],
            "actions":         acts,
            "elapsed_seconds": elapsed,
        })

    return segments
