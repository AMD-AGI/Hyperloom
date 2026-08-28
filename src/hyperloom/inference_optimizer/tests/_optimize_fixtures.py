# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the merged optimisation phase.

Two rules these builders exist to enforce:

* Never hand-roll a stand-in for a dataclass the repo owns. A hand-rolled
  ``VariantResult`` once carried ``ttft_ms`` -- a field the real class does not
  have, invented to match what buggy production code read -- so the suite could
  not see the bug, and broke when it was fixed.
* Never hand-roll a stand-in for ``SharedState``. A partial one supplies only
  the fields today's assertions touch, so a rule that starts reading a new
  field is tested against a stub rather than the state machine.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.state.shared_state import SharedState

_MISSING = object()


def variant_result(**overrides: Any) -> VariantResult:
    """A real ``VariantResult`` with plausible defaults."""
    fields: dict[str, Any] = {
        "name": "v1",
        "extra_server_args": "",
        "extra_envs": {},
        "status": "succeeded",
        "output_throughput": 1000.0,
        "ttft_mean_ms": 10.0,
        "tpot_mean_ms": 2.0,
        "error": "",
        "nonfatal_warnings": [],
    }
    fields.update(overrides)
    return VariantResult(**fields)


def optimize_state(
    *,
    source_no_keep: int = 0,
    source_exhausted: bool = False,
    config_keep_gain_pct: float = 5.0,
    config_empty_rounds: int = 0,
    **overrides: Any,
) -> SharedState:
    """A real ``SharedState`` positioned in the optimisation phase.

    The four keyword knobs drive the two arms ``exit_normal_optimize`` reads,
    so a test states the arm condition it means rather than the ledger rows
    that happen to encode it today.

    Args:
        source_no_keep: Trailing resolved candidates with no KEEP.
        source_exhausted: Whether candidate discovery reported itself done.
        config_keep_gain_pct: Gain on each recent explore winner.
        config_empty_rounds: Trailing specialist rounds that produced nothing.
        **overrides: Any other ``SharedState`` attribute to set.

    Returns:
        The positioned state.
    """
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    state = SharedState()
    state.phase = PHASE_FRAMEWORK_AGENT
    state.macro_cycle = 0
    state.baseline_tput = 1500.0
    state.framework_agent_phase_done = source_exhausted
    state.framework_agent_phase_progress = [
        {"status": "reverted", "kept": False, "cycle": 0} for _ in range(source_no_keep)
    ]
    state.explore_search = {
        "winners_history": [{"gain_pct": config_keep_gain_pct, "cycle": 0} for _ in range(6)],
    }
    state.specialist_rounds = [
        {"proposals_total": 0, "proposals_kept": 0, "cycle": 0}
        if i < config_empty_rounds
        else {"proposals_total": 2, "proposals_kept": 1, "cycle": 0}
        for i in range(max(config_empty_rounds, 1))
    ]
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


class FakeCoordinator:
    """Answers the Coordinator's state surface; resolves the rest for real.

    Anything not passed as state is looked up in ``Coordinator._DELEGATED``,
    then on ``Coordinator`` itself, then among the collaborators, and served by
    the real object. Moving a method between collaborators costs one table edit
    and no test edits, where a stub binding one method per line had to be
    re-derived every time the call chain grew.
    """

    def __init__(self, session_dir: Any, **state: Any) -> None:
        from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

        self.session_dir = session_dir
        # The real catalogue: a stubbed one can only ever agree with the test.
        self.action_registry = ACTION_CATALOGUE
        for key, value in state.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        from hyperloom.orchestrator.loop.coordinator import Coordinator

        owner = Coordinator._DELEGATED.get(name)
        if owner is None:
            # A class constant or a method still defined on Coordinator itself.
            attr = inspect.getattr_static(Coordinator, name, _MISSING)
            if attr is not _MISSING:
                get = getattr(attr, "__get__", None)
                return get(self, type(self)) if get is not None else attr
            # A collaborator-internal helper: reachable only from inside its
            # own class in production, so it has no delegation entry. The test
            # still gets it without having to know which class owns it.
            owner = self._sole_owner(name)
        key = f"_collab_{owner}"
        collaborator = self.__dict__.get(key)
        if collaborator is None:
            module_path, cls_name = Coordinator._COLLAB_MODULES[owner]
            module = importlib.import_module(f"hyperloom.orchestrator.{module_path}")
            collaborator = getattr(module, cls_name)(self)
            self.__dict__[key] = collaborator
        return getattr(collaborator, name)

    @staticmethod
    def _sole_owner(name: str) -> str:
        """The one collaborator defining ``name``, or an explanation of why not."""
        from hyperloom.orchestrator.loop.coordinator import Coordinator

        owners = []
        for prop, (module_path, cls_name) in Coordinator._COLLAB_MODULES.items():
            cls = getattr(importlib.import_module(f"hyperloom.orchestrator.{module_path}"), cls_name)
            if name in vars(cls):
                owners.append(prop)
        if len(owners) == 1:
            return owners[0]
        if owners:
            raise AttributeError(f"{name!r} is defined by more than one collaborator: {owners}")
        raise AttributeError(
            f"{name!r} is neither state this fake was given, nor an attribute of "
            f"Coordinator, nor defined by any collaborator -- the test is reaching "
            f"for something that no longer exists."
        )
