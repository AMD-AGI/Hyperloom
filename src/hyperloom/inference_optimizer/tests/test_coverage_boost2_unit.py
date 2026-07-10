# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Second batch of focused unit coverage for small pure-logic helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace



# --------------------------------------------------------------------------- #
# orchestrator.orchestration_memory.deterministic_memory_fallback             #
# --------------------------------------------------------------------------- #
def test_deterministic_memory_fallback_bad_gain() -> None:
    from hyperloom.orchestrator.state.orchestration_memory import deterministic_memory_fallback

    state = SimpleNamespace(
        current_best={"tput": 123.0},
        optimization_stack=[],
        cumulative_gain_validated="not-a-number",  # triggers except (lines 235-236)
        phase="EXPLORE",
        macro_cycle=2,
    )
    record = deterministic_memory_fallback(state)
    assert "current_plan" in record
    assert "phase=EXPLORE" in record["current_plan"]


# --------------------------------------------------------------------------- #
# orchestrator.specialist_profile                                             #
# --------------------------------------------------------------------------- #
def test_coerce_bool_and_infer_scope() -> None:
    from hyperloom.orchestrator.specialists import profile as sp

    assert sp._coerce_bool("off", default=True) is False  # line 119
    assert sp._coerce_bool("yes", default=False) is True
    assert sp._coerce_bool(None, default=True) is True
    assert sp._coerce_bool("???", default=True) is True

    # Bare dispatch with no anchors -> freeform scope (line 142 path).
    profile = sp.resolve_specialist_profile({})
    assert profile.scope == sp.SCOPE_FREEFORM


def test_uses_whole_machine_gpu_lane() -> None:
    from hyperloom.orchestrator.specialists import profile as sp

    # Framework-authoring specialists always take the whole-machine lane.
    assert sp.uses_whole_machine_gpu_lane({"framework_agent_authoring": True}) is True

    # Bench-capable (mode=patch & bench=true) specialists take it too.
    assert (
        sp.uses_whole_machine_gpu_lane(
            {"scope": "freeform", "mode": "patch", "bench": True}
        )
        is True
    )

    # Non-bench patch probes and research specialists keep the disjoint pool.
    assert (
        sp.uses_whole_machine_gpu_lane(
            {"scope": "freeform", "mode": "patch", "bench": False}
        )
        is False
    )
    assert sp.uses_whole_machine_gpu_lane({"scope": "freeform", "mode": "research"}) is False
    assert sp.uses_whole_machine_gpu_lane(None) is False


# --------------------------------------------------------------------------- #
# orchestrator.action_executors._accuracy_gate.parse_quality_gate            #
# --------------------------------------------------------------------------- #
def test_parse_quality_gate_paths(tmp_path) -> None:
    from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag

    # No report present.
    assert ag.parse_quality_gate(tmp_path)["quality_gate"] is None

    # Malformed JSON (lines 160-161).
    (tmp_path / "benchmark_report.json").write_text("{not valid json", encoding="utf-8")
    res = ag.parse_quality_gate(tmp_path)
    assert res["quality_gate"] is None
    assert "parse error" in res["error"]

    # Valid JSON but no quality_gate block.
    (tmp_path / "benchmark_report.json").write_text(json.dumps({"throughput": {}}), encoding="utf-8")
    res2 = ag.parse_quality_gate(tmp_path)
    assert res2["quality_gate"] is None

    # Valid quality_gate block.
    (tmp_path / "benchmark_report.json").write_text(
        json.dumps({"quality_gate": {"passed": True}}), encoding="utf-8"
    )
    res3 = ag.parse_quality_gate(tmp_path)
    assert res3["quality_gate"] == {"passed": True}


# --------------------------------------------------------------------------- #
# orchestrator.trace.trace_env.env_flag                                        #
# --------------------------------------------------------------------------- #
def test_env_flag_tokens(monkeypatch) -> None:
    from hyperloom.orchestrator.trace import trace_env

    monkeypatch.setenv("HL_TEST_FLAG", "on")
    assert trace_env.env_flag("HL_TEST_FLAG") is True
    monkeypatch.setenv("HL_TEST_FLAG", "off")  # lines 61-62
    assert trace_env.env_flag("HL_TEST_FLAG") is False
    monkeypatch.setenv("HL_TEST_FLAG", "maybe")  # unrecognized -> default (line 63)
    assert trace_env.env_flag("HL_TEST_FLAG", default=True) is True
    monkeypatch.delenv("HL_TEST_FLAG", raising=False)
    assert trace_env.env_flag("HL_TEST_FLAG", default=False) is False


# --------------------------------------------------------------------------- #
# orchestrator.gpu_pool._parse_gpu_list                                       #
# --------------------------------------------------------------------------- #
def test_parse_gpu_list() -> None:
    from hyperloom.orchestrator.bus.gpu_pool import _parse_gpu_list

    # Mixed valid / invalid / duplicate / negative entries (lines 52-53 except).
    assert _parse_gpu_list("0, 1 ; 2, x, 1, -3") == [0, 1, 2]
    assert _parse_gpu_list("") == []
    assert _parse_gpu_list(None) == []  # type: ignore[arg-type]
