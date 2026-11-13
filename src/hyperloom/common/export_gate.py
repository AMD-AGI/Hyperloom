# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a declared ahead-of-time export has landed, and who waits for it.

In ``common`` rather than beside any one holder: they are spread across ``orchestrator``,
and the layer contract places ``inference_optimizer`` below it.
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

    Fail-closed on an unreadable declaration: a false hold costs a tick, a false admission
    spends the budget the gate exists to protect.
    """
    from hyperloom.common.optimization_scope import aoti_config

    if _accepted_module_exports(state):
        return False
    try:
        config = aoti_config()
    except ValueError:
        return True
    return config is not None and config.enabled

