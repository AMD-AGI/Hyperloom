"""F1-3 — RooflineExecutor end-to-end registration smoke tests.

These tests verify the wiring contracts that operators rely on:

* The ``roofline`` action name is in the action registry (driven by the
  yaml landed in F1-2.5) and tagged ``family=analysis``.
* The ``roofline`` task kind is in :data:`_AUDIT_ACTIONS` so every
  composite roofline run shows up in the prompt's RECENT ACTION
  ATTEMPTS block.
* :func:`cli._register_executors` registers either the real
  :class:`RooflineExecutor` (when ``use_roofline_composite=True``) or
  the :class:`RooflineStubExecutor` fallback (default off) so a
  speculative ``propose_action{action='roofline'}`` never hits
  ``no_executor``.

Reference: ``plan_roofline_framework/F1_roofline_composite.MD`` §F1-3.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def test_roofline_in_action_registry():
    from inference_optimizer.orchestrator.action_registry import ActionRegistry

    registry = ActionRegistry().load()
    meta = registry.get("roofline")
    assert meta is not None
    assert meta.family == "analysis"
    assert "profile_lane" in meta.requires_lanes
    assert "baseline" in meta.prerequisites


def test_roofline_in_audit_actions():
    from inference_optimizer.orchestrator.shared_state import _AUDIT_ACTIONS
    assert "roofline" in _AUDIT_ACTIONS


def test_roofline_key_metric_map():
    from inference_optimizer.orchestrator.shared_state import _KEY_METRIC_MAP
    assert _KEY_METRIC_MAP["roofline"] == ("snapshot_id", "snapshot_id")


def test_register_executors_wires_real_when_toggle_on():
    """``use_roofline_composite=True`` => real RooflineExecutor."""
    from inference_optimizer.cli import _register_executors
    from inference_optimizer.orchestrator.action_executors.roofline import (
        RooflineExecutor,
    )

    registered: dict[str, Any] = {}

    class _StubSub:
        def register_executor(self, kind: str, fn: Any) -> None:
            registered[kind] = fn

    coordinator = SimpleNamespace(
        sub=_StubSub(),
        shared_state=SimpleNamespace(use_roofline_composite=True),
    )
    _register_executors(coordinator, no_kernel=False)
    assert "roofline" in registered
    assert isinstance(registered["roofline"], RooflineExecutor)


def test_register_executors_wires_stub_when_toggle_off():
    """Default ``use_roofline_composite=False`` => stub fallback."""
    from inference_optimizer.cli import _register_executors
    from inference_optimizer.orchestrator.action_executors.roofline import (
        RooflineStubExecutor,
    )

    registered: dict[str, Any] = {}

    class _StubSub:
        def register_executor(self, kind: str, fn: Any) -> None:
            registered[kind] = fn

    coordinator = SimpleNamespace(
        sub=_StubSub(),
        shared_state=SimpleNamespace(use_roofline_composite=False),
    )
    _register_executors(coordinator, no_kernel=False)
    assert "roofline" in registered
    assert isinstance(registered["roofline"], RooflineStubExecutor)


def test_register_executors_skips_roofline_when_no_kernel():
    """``no_kernel=True`` short-circuits the kernel-only block, which
    includes ``roofline``: profiling has no consumer in --no-kernel mode.
    """
    from inference_optimizer.cli import _register_executors

    registered: dict[str, Any] = {}

    class _StubSub:
        def register_executor(self, kind: str, fn: Any) -> None:
            registered[kind] = fn

    coordinator = SimpleNamespace(
        sub=_StubSub(),
        shared_state=SimpleNamespace(use_roofline_composite=True),
    )
    _register_executors(coordinator, no_kernel=True)
    assert "roofline" not in registered
