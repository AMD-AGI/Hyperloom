# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a declared ahead-of-time export has landed: who waits for it, and how it reads.

In ``common`` rather than beside either caller: the holders are spread across
``orchestrator`` and the launch CLI prints the summary line, and the layer contract places
``inference_optimizer`` below ``orchestrator``. ``report.py``, the other candidate, is
already at the module-length limit.
"""

from __future__ import annotations

from typing import Any


def _accepted_module_exports(state: Any) -> bool:
    """Whether the module ``current_best`` holds exported, as every reader here asks it."""
    return bool(getattr(state, "current_best_exports", False))


def export_gate_is_shut(state: Any) -> bool:
    """Whether search must wait because the accepted module does not export.

    One predicate for every holder, because they are not all on one path: the admission
    gate sees what the agents propose, the FRAMEWORK pump benches waiting proposals on its
    own without passing that gate, and the backlog cap that holds specialists back has to
    stand down while nothing can drain it.

    Fail-closed on an unreadable declaration, as the gate is: a false hold costs a tick, a
    false admission spends the budget the gate exists to protect. That is the one place
    this parts from ``export_gate_stop_reason``, which reports rather than holds and must
    not turn a transient read error into a terminal.
    """
    from hyperloom.common.optimization_scope import aoti_config

    if _accepted_module_exports(state):
        return False
    try:
        config = aoti_config()
    except ValueError:
        return True
    return config is not None and config.enabled


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
    if _accepted_module_exports(state):
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
    if not _accepted_module_exports(state):
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
