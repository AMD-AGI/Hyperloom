# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplemental coverage for kernel_request_handlers pure helpers: precision / budget / timeout resolution, backend order, tool-stdout shaping, roofline name lookup, artifact-path and in-flight scanning."""

from __future__ import annotations

from pathlib import Path

from kernelforge.gemm_tune.router import RoutingEvidence, RoutingTarget

from hyperloom.orchestrator.kernel import lane_budget
from hyperloom.orchestrator.actions.executors import _kernel_agent_tool as kernel_agent_tool
from hyperloom.orchestrator.kernel import _forge_lane_budget as forge_lane_budget
from hyperloom.orchestrator.kernel import _integrate_payload as integrate_payload


# -- _normalize_precision / _normalize_kernel_id --------------------------
def test_normalize_precision() -> None:
    assert integrate_payload._normalize_precision("  FP8 ") == "fp8"
    assert integrate_payload._normalize_precision(None) == ""
    assert integrate_payload._normalize_precision(0) == ""


# -- _gemm_tuning_timeout_sec ---------------------------------------------
def test_gemm_tuning_timeout_payload_floor(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert forge_lane_budget._gemm_tuning_timeout_sec({"timeout_sec": 30}) == 60  # floored
    assert forge_lane_budget._gemm_tuning_timeout_sec({"timeout_sec": 900}) == 900


def test_gemm_tuning_timeout_env_and_invalid(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", "300")
    assert forge_lane_budget._gemm_tuning_timeout_sec({}) == 300
    monkeypatch.setenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", "bad")
    assert forge_lane_budget._gemm_tuning_timeout_sec({}) == max(60, forge_lane_budget._DEFAULT_GEMM_TUNING_TIMEOUT_SEC)


def test_gemm_tuning_timeout_takes_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert forge_lane_budget._gemm_tuning_timeout_sec({}, lane_budget_sec=7140) == 7140


def test_gemm_tuning_timeout_without_an_allocation_keeps_the_module_default(monkeypatch) -> None:
    """An unbounded phase must not collapse the lane to a zero-second session."""
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert forge_lane_budget._gemm_tuning_timeout_sec({}, lane_budget_sec=0) == 18000


def test_an_explicit_gemm_timeout_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert forge_lane_budget._gemm_tuning_timeout_sec({"timeout_sec": 900}, lane_budget_sec=7140) == 900


def test_the_gemm_timeout_env_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", "300")
    assert forge_lane_budget._gemm_tuning_timeout_sec({}, lane_budget_sec=7140) == 300


# -- _forge_fusion_timeout_sec --------------------------------------------
def test_fusion_timeout_takes_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_FUSION_TIMEOUT", raising=False)
    assert forge_lane_budget._forge_fusion_timeout_sec({}, lane_budget_sec=10710) == 10710


def test_fusion_timeout_without_an_allocation_keeps_the_module_default(monkeypatch) -> None:
    """An unbounded phase must not collapse the lane to a one-second session."""
    monkeypatch.delenv("FORGE_FUSION_TIMEOUT", raising=False)
    assert forge_lane_budget._forge_fusion_timeout_sec({}, lane_budget_sec=0) == 7200


def test_an_explicit_fusion_timeout_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_FUSION_TIMEOUT", raising=False)
    assert forge_lane_budget._forge_fusion_timeout_sec({"timeout": 900}, lane_budget_sec=10710) == 900


def test_the_fusion_timeout_env_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "300")
    assert forge_lane_budget._forge_fusion_timeout_sec({}, lane_budget_sec=10710) == 300


# -- _gemm_router_targets / _gemm_capped_tuner ----------------------------
_ROUTER_PROBE = {
    "model_path": "/nonexistent-model-dir",
    "target": RoutingTarget(framework="sglang", precision="bf16", quant_type="auto", gpu_type="auto"),
    "evidence": RoutingEvidence(
        kernel_signature_log=None,
        has_untuned_csv=False,
        has_shapes_json=False,
        has_tunableop_input=False,
    ),
}


def test_an_unconsultable_router_yields_no_estimates() -> None:
    assert forge_lane_budget._gemm_router_targets(**_ROUTER_PROBE) == ()


def test_without_estimates_the_gemm_ceiling_divides_by_its_default_cost() -> None:
    """7140s of lane share buys five 20-minute targets when nothing is estimated."""
    costs = tuple(cost for _, cost in forge_lane_budget._gemm_router_targets(**_ROUTER_PROBE))
    assert lane_budget.max_targets(lane_budget.LANE_GEMM, 7140, target_costs_sec=costs) == 5
    assert 7140 // lane_budget.GEMM_DEFAULT_TARGET_SEC == 5


_GEMM_CMD_BASE = {
    "model_path": "/m",
    "framework": "sglang",
    "precision": "fp8",
    "output_dir": "/o",
}


def test_the_gemm_ceiling_travels_as_a_tuner_count() -> None:
    """The producer owns routing, so the lane supplies a count, not a name."""
    from hyperloom.orchestrator.kernel import forge_gemm_tuning as fgt

    cmd = fgt.build_command({**_GEMM_CMD_BASE, "max_tuners": 2})
    assert cmd[cmd.index("--max-tuners") + 1] == "2"


def test_no_gemm_ceiling_leaves_the_routed_tuner_set_untouched() -> None:
    """A zero ceiling means none could be derived, not "run nothing"."""
    from hyperloom.orchestrator.kernel import forge_gemm_tuning as fgt

    cmd = fgt.build_command(dict(_GEMM_CMD_BASE))
    assert "--max-tuners" not in cmd


# -- _gemm_tuning_workspace -----------------------------------------------
def test_gemm_tuning_workspace_explicit(tmp_path: Path) -> None:
    out = forge_lane_budget._gemm_tuning_workspace({"workspace_path": str(tmp_path / "ws")}, session_dir=tmp_path)
    assert out == tmp_path / "ws"


def test_gemm_tuning_workspace_from_task_id(tmp_path: Path) -> None:
    out = forge_lane_budget._gemm_tuning_workspace({"task_id": "t-1"}, session_dir=tmp_path)
    assert out == tmp_path / "runs" / "gemm_tuning" / "t-1"


def test_gemm_tuning_workspace_timestamp_fallback(tmp_path: Path) -> None:
    out = forge_lane_budget._gemm_tuning_workspace({}, session_dir=tmp_path)
    assert out.parent == tmp_path / "runs" / "gemm_tuning"
    assert out.name.startswith("request_")


# -- _artifact_paths_from_payload -----------------------------------------
def test_artifact_paths_string_wrapped() -> None:
    assert kernel_agent_tool._artifact_paths_from_payload({"artifact_paths": "/a/b.so"}) == ["/a/b.so"]


def test_artifact_paths_list_filters_falsy() -> None:
    out = kernel_agent_tool._artifact_paths_from_payload(
        {"compiled_artifact_paths": ["/a", "", None, "/b"]},
    )
    assert out == ["/a", "/b"]


def test_artifact_paths_other_type() -> None:
    assert kernel_agent_tool._artifact_paths_from_payload({"artifact_paths": 42}) == []
    assert kernel_agent_tool._artifact_paths_from_payload({}) == []


# -- _kernel_result_rank --------------------------------------------------- -- _parse_tool_stdout /
# _shape_tool_result ------------------------------
def test_parse_tool_stdout_whole_json() -> None:
    assert kernel_agent_tool._parse_tool_stdout('{"status": "ok", "x": 1}') == {"status": "ok", "x": 1}


def test_parse_tool_stdout_empty() -> None:
    assert kernel_agent_tool._parse_tool_stdout("   ") == {}


def test_parse_tool_stdout_last_line_json() -> None:
    out = kernel_agent_tool._parse_tool_stdout('noise line\nmore noise\n{"status": "ok"}')
    assert out == {"status": "ok"}


def test_parse_tool_stdout_no_json_returns_tail() -> None:
    out = kernel_agent_tool._parse_tool_stdout("just plain text, no json here")
    assert "raw_stdout_tail" in out


_PRETTY_TOOL_STDOUT = """\
[claude-sdk] Now Step 1: generating the performance report.
[claude-sdk] Perf report completed successfully.
TraceLens SDK orchestrator produced 43 hot kernels
{
  "hot_kernels": [
    {
      "kernel_id": "k043",
      "name": "aten::_flash_attention_forward"
    }
  ],
  "status": "ok",
  "trace_report_path": "/s/tracelens/analysis.md"
}
[aiter] import [module_aiter_core] under /sgl-workspace/aiter/aiter/jit/x.so
"""


def test_parse_tool_stdout_recovers_a_pretty_printed_result() -> None:
    """The shape a tool with a lot to say actually emits."""
    out = kernel_agent_tool._parse_tool_stdout(_PRETTY_TOOL_STDOUT)

    assert out["status"] == "ok"
    assert out["trace_report_path"] == "/s/tracelens/analysis.md"
    assert out["hot_kernels"][0]["name"] == "aten::_flash_attention_forward"


def test_shape_tool_result_will_not_call_unreadable_output_a_success() -> None:
    """Inferring ``ok`` from rc==0 made a tool whose output could not be read indistinguishable from one that worked, so the caller recorded an empty analysis over a real one and reported the leg as succeeded."""
    out = kernel_agent_tool._shape_tool_result(0, "progress chatter, no json at all", "")

    assert out["status"] == "failed"
    assert out["error_class"] == "tool_output_unparseable"
    assert "raw_stdout_tail" in out


def test_shape_tool_result_uses_parsed_json() -> None:
    out = kernel_agent_tool._shape_tool_result(0, '{"status": "ok", "kernel_id": "k1"}', "")
    assert out["status"] == "ok" and out["kernel_id"] == "k1"


def test_shape_tool_result_infers_status_and_stderr_tail() -> None:
    out = kernel_agent_tool._shape_tool_result(1, '{"kernel_id": "k1"}', "boom error")
    assert out["status"] == "failed"
    assert out["returncode"] == 1
    assert out["stderr_tail"].endswith("boom error")


def test_shape_tool_result_synthesizes_on_empty_stdout() -> None:
    # empty stdout -> _parse_tool_stdout returns {} -> synthesize branch
    out = kernel_agent_tool._shape_tool_result(2, "", "the stderr")
    assert out == {"status": "failed", "returncode": 2, "error": "the stderr"}


_OP_FANOUT_ROWS = [
    {
        "kernel_id": "k001",
        "name": "gqa_prefill_kernel",
        "gpu_pct": 9.0,
        "reusable_native_kernel": True,
        "source_file": "/pkg/aiter/gqa.py",
    },
    {
        "kernel_id": "k002",
        "name": "gqa_decode_kernel",
        "gpu_pct": 7.0,
        "reusable_native_kernel": True,
        "source_file": "/pkg/aiter/gqa.py",
    },
]
