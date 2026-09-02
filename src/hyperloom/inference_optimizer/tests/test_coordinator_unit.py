# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused unit tests for ``coordinator`` module-level helpers and static utilities."""

from __future__ import annotations


from hyperloom.orchestrator.loop.coordinator import (
    Coordinator,
    _first_present,
    _format_inbox_event,
    _lifecycle_paths,
)
from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.policy.gate import SPECIALIST_FROM_AGENT_PREFIX


def test_first_present_non_dict_and_branches():
    """``_first_present`` ignores non-dicts and skips None values."""
    assert _first_present("not-a-dict", ("a",)) is None  # type: ignore[arg-type]
    assert _first_present({}, ("a", "b")) is None
    assert _first_present({"a": None, "b": 2}, ("a", "b")) == 2
    assert _first_present({"x": 0}, ("x",)) == 0


def test_lifecycle_paths_filters_and_types():
    """``_lifecycle_paths`` only keeps non-empty string values for known keys."""
    assert _lifecycle_paths(None) == {}
    assert _lifecycle_paths({"workspace": "  "}) == {}
    assert _lifecycle_paths({"workspace": "/data/ws", "report_path": "/r.md"}) == {
        "workspace": "/data/ws",
        "report_path": "/r.md",
    }


def test_format_inbox_delegated_result_with_msg_id():
    """delegated_result lines include outcome keys when ``result`` is a dict."""
    m = Message(
        msg_id="mid",
        from_agent="kernel_agent",
        to_agent="orch",
        topic="delegated_result",
        payload={
            "kind": "integrate",
            "state": "done",
            "result": {
                "status": "ok",
                "kept": True,
                "gain_pct": 3.5,
                "tokens_per_s": 900,
            },
        },
        seq=7,
    )
    line = _format_inbox_event(m)
    assert "msg_id=mid" in line and "seq=7" in line
    assert "gain=3.5" in line and "tput=900" in line


def test_format_inbox_delegated_result_no_msg_id():
    """Header omits ``msg_id=`` when the field is unset."""
    m = Message(
        msg_id="",
        from_agent="k",
        to_agent="o",
        topic="delegated_result",
        payload={"kind": "k", "state": "s", "result": {"verdict": "x"}},
        seq=2,
    )
    line = _format_inbox_event(m)
    assert "seq=2" in line and "msg_id=" not in line


def test_format_inbox_delegated_result_non_dict_result():
    m = Message(
        msg_id="1",
        from_agent="a",
        to_agent="b",
        topic="delegated_result",
        payload={"kind": "k", "state": "s", "result": "raw"},
        seq=1,
    )
    line = _format_inbox_event(m)
    assert "kind='k'" in line and "state='s'" in line and "raw" not in line


def test_format_inbox_delegated_result_with_error():
    m = Message(
        msg_id="e",
        from_agent="a",
        to_agent="b",
        topic="delegated_result",
        payload={"kind": "k", "state": "failed", "error": "boom" * 50, "result": {}},
        seq=1,
    )
    line = _format_inbox_event(m)
    assert "error=" in line


def test_format_inbox_policy_denial_topics():
    m1 = Message(
        msg_id="p",
        from_agent="orch",
        to_agent="k",
        topic="policy_denial",
        payload={"action_name": "act", "rule": "r1", "hint": "h" * 200},
        seq=3,
    )
    assert "action='act'" in _format_inbox_event(m1)

    m2 = Message(
        msg_id="p2",
        from_agent="o",
        to_agent="k",
        topic="denial",
        payload={"action_name": "a2", "rule": "r2", "hint": ""},
        seq=4,
    )
    assert "rule='r2'" in _format_inbox_event(m2)

    m3 = Message(
        msg_id="p3",
        from_agent="o",
        to_agent="k",
        topic="observation",
        payload={"kind": "policy_denial", "action_name": "x", "rule": "rx", "hint": "y"},
        seq=5,
    )
    out = _format_inbox_event(m3)
    assert "topic=observation" in out and "action='x'" in out and "rule='rx'" in out


def test_format_inbox_review_verdict():
    m = Message(
        msg_id="r",
        from_agent="critic",
        to_agent="orch",
        topic="review_verdict",
        payload={
            "target_proposal_msg_id": "pm1",
            "verdict": "approve",
            "reasoning": "ok" * 80,
        },
        seq=9,
    )
    line = _format_inbox_event(m)
    assert "verdict='approve'" in line and "target='pm1'" in line


def test_format_inbox_observation_generic():
    m = Message(
        msg_id="o",
        from_agent="a",
        to_agent="b",
        topic="observation",
        payload={"kind": "metric", "v": 1},
        seq=1,
    )
    assert "kind='metric'" in _format_inbox_event(m)


def test_format_inbox_fallback():
    m = Message(
        msg_id="f",
        from_agent="a",
        to_agent="b",
        topic="other",
        payload={"z": 1},
        seq=0,
    )
    assert "topic=other" in _format_inbox_event(m) and "payload=" in _format_inbox_event(m)


def test_skip_gemm_tuning_env(monkeypatch):
    """Env gate used before FP8 GEMM pre-kernel_opt scheduling."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "true")
    assert Coordinator._skip_gemm_tuning() is True
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "")
    assert Coordinator._skip_gemm_tuning() is False


def test_gap_layer_for_action_mapping():
    assert Coordinator._gap_layer_for_action("kernel_opt") == (
        "kernel_agent",
        "kernel_switch_specialist",
    )
    assert Coordinator._gap_layer_for_action("PROFILE")[0] == "kernel_agent"
    assert Coordinator._gap_layer_for_action("sweep")[0] == "framework"
    assert Coordinator._gap_layer_for_action("baseline")[0] == "system"
    assert Coordinator._gap_layer_for_action("  ") == ("framework", "serving_specialist")


def test_gap_layer_for_action_follows_framework_kind():
    # A framework-layer gap on a scriptable workload must name the rewrite
    # specialist: seeding it with serving_specialist is what steered a custom
    # workload back onto the serving surface once EXPLORE picked the gap up.
    assert Coordinator._gap_layer_for_action("sweep", "custom") == (
        "framework",
        "framework_rewrite_specialist",
    )
    assert Coordinator._gap_layer_for_action("sweep", "xdit")[1] == "framework_rewrite_specialist"
    assert Coordinator._gap_layer_for_action("sweep", "sglang")[1] == "serving_specialist"
    # Kernel and system rows are framework-independent.
    assert Coordinator._gap_layer_for_action("kernel_opt", "custom")[1] == "kernel_switch_specialist"
    assert Coordinator._gap_layer_for_action("baseline", "custom")[1] == "system_specialist"


def test_task_id_from_specialist_source():
    assert Coordinator._task_id_from_specialist_source("") == ""
    assert Coordinator._task_id_from_specialist_source("orch") == ""
    tid = "abc-123"
    assert (
        Coordinator._task_id_from_specialist_source(
            f"{SPECIALIST_FROM_AGENT_PREFIX}{tid}",
        )
        == tid
    )


def test_lanes_fit_headroom():
    assert Coordinator._lanes_fit(["lane_a"], {"lane_a": 0}, {"lane_a": 2}) is True
    assert Coordinator._lanes_fit(["lane_a"], {"lane_a": 2}, {"lane_a": 2}) is False
    assert Coordinator._lanes_fit(["lane_a"], {"lane_a": 0}, {"lane_a": 0}) is False
