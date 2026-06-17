# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Focused unit tests for ``coordinator`` module-level helpers and static utilities.

Keeps imports light: no full ``Coordinator`` session construction unless necessary.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _first_present,
    _format_inbox_event,
    _lifecycle_paths,
)
from inference_optimizer.orchestrator.message_bus import Message
from inference_optimizer.orchestrator.policy import SPECIALIST_FROM_AGENT_PREFIX


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
        from_agent="kernel",
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
    # Non-dict ``result`` omits outcome keys; only kind/state are surfaced.
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


def test_pr_summary_to_dict_normalizes():
    pr = SimpleNamespace(
        repo="R",
        number="3",
        title="T",
        url="U",
        state="open",
        labels=("x", "y"),
        author="me",
    )
    d = Coordinator._pr_summary_to_dict(pr)
    assert d == {
        "repo": "R",
        "number": 3,
        "title": "T",
        "url": "U",
        "state": "open",
        "labels": ["x", "y"],
        "author": "me",
    }


def test_build_sweep_params_skill_defaults():
    """No recipe → SKILL defaults marker and list-shaped sweep params."""
    st = SimpleNamespace(warm_start_recipe=None)
    p = Coordinator._build_sweep_params_from_recipe(st)
    assert p["source"] == "skill_md_default"
    assert isinstance(p["conc_values"], list) and p["conc_values"]
    assert isinstance(p["isl_osl_configs"], list)


def test_build_sweep_params_from_recipe_grid():
    """Recipe sweep_grid overrides defaults when fields are well-formed."""
    st = SimpleNamespace(
        warm_start_recipe={
            "sweep_grid": {
                "conc_values": [4, 8],
                "isl_osl_configs": ["1:1", [32, 64]],
                "num_prompts_factor": 3,
            },
        },
    )
    p = Coordinator._build_sweep_params_from_recipe(st)
    assert p["source"] == "cortex_recipe"
    assert p["conc_values"] == [4, 8]
    assert p["isl_osl_configs"][0] == "1:1"
    assert p["isl_osl_configs"][1] == "32:64"
    assert p["num_prompts_factor"] == 3


def test_gap_layer_for_action_mapping():
    assert Coordinator._gap_layer_for_action("kernel_opt") == (
        "kernel",
        "kernel_switch_specialist",
    )
    assert Coordinator._gap_layer_for_action("PROFILE")[0] == "kernel"
    assert Coordinator._gap_layer_for_action("sweep")[0] == "framework"
    assert Coordinator._gap_layer_for_action("baseline")[0] == "system"
    assert Coordinator._gap_layer_for_action("  ") == ("framework", "serving_specialist")


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
