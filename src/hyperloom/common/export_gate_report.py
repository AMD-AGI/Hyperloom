# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a run reports an ahead-of-time export that was declared but never landed.

Here rather than beside ``report.py``: the launch CLI prints the summary line too, and the
layer contract places ``inference_optimizer`` below ``orchestrator``. The stop reason is
also the one summary field whose value comes from the export gate rather than from the
run's own terminal, and ``report.py`` is already at the module-length limit.
"""

from __future__ import annotations

from typing import Any


def export_gate_stop_reason(state: Any) -> str:
    """Name an unlanded ahead-of-time export, whatever else ended the run.

    Overrides every other terminal, so a short run, budget exhaustion and wall-clock
    expiry all report it rather than reading as an ordinary closeout: without an
    export there is no shippable artifact, which is the fact that matters most.

    Keyed on whether the *accepted* module exports, not on whether one ever landed. The
    two part only when the harness breaks -- the integrate gate refuses a candidate that
    cannot be shown to export -- and in that case what ships is what matters.
    """
    from hyperloom.common.optimization_scope import aoti_config

    try:
        config = aoti_config()
    except ValueError:
        return ""
    if config is None or not config.enabled:
        return ""
    if bool(getattr(state, "current_best_exports", False)):
        return ""
    return "export_never_landed"


def export_gate_summary_line(state: Any) -> str:
    """One summary line saying whether the declared ahead-of-time export landed.

    A landed export can still show a negative headline gain, measured against an eager
    path this workload cannot ship, so the gain figure alone misreads. Empty unless
    ``aoti.enabled``, and empty on an unreadable declaration -- the gate itself already
    fails closed on that, and a summary line is not where to surface it.
    """
    from hyperloom.common.optimization_scope import aoti_config

    try:
        config = aoti_config()
    except ValueError:
        return ""
    if config is None or not config.enabled:
        return ""
    if not getattr(state, "current_best_exports", False):
        if getattr(state, "export_landed", False):
            # Only reachable through a broken harness: the integrate gate refuses a
            # candidate that cannot be shown to export. Points at the shim rather than
            # at an optimization outcome, because that is what went wrong.
            return (
                "  aoti_export          : NOT IN THE ARTIFACT -- an export landed earlier but the "
                "accepted module cannot be shown to export; the start-up counters are missing, so "
                "treat this as a broken harness and re-run"
            )
        attempts = int(getattr(state, "export_attempts", 0) or 0)
        return (
            f"  aoti_export          : NEVER LANDED after {attempts} integrate round(s) -- "
            f"no shippable artifact was produced, whatever else this run kept"
        )
    gain = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
    tail = (
        "the artifact is faster than the eager path it replaces"
        if gain > 0
        else "it is still behind the eager path, which this workload does not ship; "
        "optimizing the export is what further budget buys"
    )
    return f"  aoti_export          : landed -- {tail}"
