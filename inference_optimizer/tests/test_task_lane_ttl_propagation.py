"""Coordinator manually-created tasks inherit requires_lanes +
lease_ttl_sec from the ActionRegistry (filtered to dispatcher lanes)."""

from __future__ import annotations

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.coordinator import Coordinator


class _Stub:
    action_registry = ActionRegistry().load()


_resolve = Coordinator._registry_lanes_ttl


def test_dynamic_action_lanes_ttl_filters_capability_tags():
    lanes, ttl = _resolve(_Stub(), "dynamic_action")
    assert lanes == ["research_lane"]  # emit_intent tag filtered out
    assert ttl == 1800


def test_roofline_lanes_ttl():
    lanes, ttl = _resolve(_Stub(), "roofline")
    assert lanes == ["profile_lane"]
    assert ttl == 2700


def test_integrate_patch_lanes_ttl():
    lanes, ttl = _resolve(_Stub(), "integrate_patch")
    assert lanes == [
        "server_lifecycle", "workspace_mutation", "benchmark_lane",
    ]
    assert ttl == 3600


def test_unknown_action_falls_back_to_no_lanes():
    lanes, ttl = _resolve(_Stub(), "definitely_not_an_action")
    assert lanes == []
    assert ttl == 0


def test_missing_registry_falls_back():
    class _NoReg:
        action_registry = None
    lanes, ttl = _resolve(_NoReg(), "dynamic_action")
    assert lanes == []
    assert ttl == 0


def test_resolved_lanes_are_dispatcher_known():
    from inference_optimizer.orchestrator.resource_lock import KNOWN_LANES
    for kind in ("specialist", "dynamic_action", "roofline", "profile",
                 "integrate_patch", "explore"):
        lanes, _ = _resolve(_Stub(), kind)
        for lane in lanes:
            assert lane in KNOWN_LANES, (
                f"{kind} resolved unknown dispatcher lane {lane!r}"
            )
