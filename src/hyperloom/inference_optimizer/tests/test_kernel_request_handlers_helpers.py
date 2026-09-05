# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplemental coverage for kernel_request_handlers pure helpers: precision /
budget / timeout resolution, backend order, tool-stdout shaping, roofline name
lookup, artifact-path and in-flight scanning."""

from __future__ import annotations

from pathlib import Path


from hyperloom.orchestrator.kernel import lane_budget
from hyperloom.orchestrator.kernel import request_handlers as krh


# -- _normalize_precision / _normalize_kernel_id --------------------------
def test_normalize_precision() -> None:
    assert krh._normalize_precision("  FP8 ") == "fp8"
    assert krh._normalize_precision(None) == ""
    assert krh._normalize_precision(0) == ""


# -- _gemm_tuning_timeout_sec ---------------------------------------------
def test_gemm_tuning_timeout_payload_floor(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert krh._gemm_tuning_timeout_sec({"timeout_sec": 30}) == 60  # floored
    assert krh._gemm_tuning_timeout_sec({"timeout_sec": 900}) == 900


def test_gemm_tuning_timeout_env_and_invalid(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", "300")
    assert krh._gemm_tuning_timeout_sec({}) == 300
    monkeypatch.setenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", "bad")
    assert krh._gemm_tuning_timeout_sec({}) == max(60, krh._DEFAULT_GEMM_TUNING_TIMEOUT_SEC)


def test_gemm_tuning_timeout_takes_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert krh._gemm_tuning_timeout_sec({}, lane_budget_sec=7140) == 7140


def test_gemm_tuning_timeout_without_an_allocation_keeps_the_module_default(monkeypatch) -> None:
    """An unbounded phase must not collapse the lane to a zero-second session."""
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert krh._gemm_tuning_timeout_sec({}, lane_budget_sec=0) == 18000


def test_an_explicit_gemm_timeout_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", raising=False)
    assert krh._gemm_tuning_timeout_sec({"timeout_sec": 900}, lane_budget_sec=7140) == 900


def test_the_gemm_timeout_env_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC", "300")
    assert krh._gemm_tuning_timeout_sec({}, lane_budget_sec=7140) == 300


# -- _forge_fusion_timeout_sec --------------------------------------------
def test_fusion_timeout_takes_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_FUSION_TIMEOUT", raising=False)
    assert krh._forge_fusion_timeout_sec({}, lane_budget_sec=10710) == 10710


def test_fusion_timeout_without_an_allocation_keeps_the_module_default(monkeypatch) -> None:
    """An unbounded phase must not collapse the lane to a one-second session."""
    monkeypatch.delenv("FORGE_FUSION_TIMEOUT", raising=False)
    assert krh._forge_fusion_timeout_sec({}, lane_budget_sec=0) == 7200


def test_an_explicit_fusion_timeout_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_FUSION_TIMEOUT", raising=False)
    assert krh._forge_fusion_timeout_sec({"timeout": 900}, lane_budget_sec=10710) == 900


def test_the_fusion_timeout_env_outranks_the_lane_share(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "300")
    assert krh._forge_fusion_timeout_sec({}, lane_budget_sec=10710) == 300


# -- _gemm_router_targets / _gemm_capped_tuner ----------------------------
_ROUTER_PROBE = {
    "model_path": "/nonexistent-model-dir",
    "framework": "sglang",
    "precision": "bf16",
    "quant_type": "auto",
    "gpu_type": "auto",
    "kernel_signature_log": "",
    "has_untuned_csv": False,
    "has_shapes_json": False,
    "has_tunableop_input": False,
}


def test_an_unconsultable_router_yields_no_estimates() -> None:
    assert krh._gemm_router_targets(**_ROUTER_PROBE) == ()


def test_without_estimates_the_gemm_ceiling_divides_by_its_default_cost() -> None:
    """7140s of lane share buys five 20-minute targets when nothing is estimated."""
    costs = tuple(cost for _, cost in krh._gemm_router_targets(**_ROUTER_PROBE))
    assert lane_budget.max_targets(lane_budget.LANE_GEMM, 7140, target_costs_sec=costs) == 5
    assert 7140 // lane_budget.GEMM_DEFAULT_TARGET_SEC == 5


_GEMM_CMD_BASE = {
    "model_path": "/m",
    "framework": "sglang",
    "precision": "fp8",
    "output_dir": "/o",
}


def test_the_gemm_ceiling_travels_as_a_tuner_count() -> None:
    """The producer owns routing, so the lane supplies a count, not a name.

    Naming one tuner was the only ceiling the wrapper could express before, so a
    share paying for two of three routed tuners capped nothing at all and the
    third ran anyway, bounded only by the wall clock.
    """
    from hyperloom.agents.kernel.tools import forge_gemm_tuning as fgt

    cmd = fgt._build_cmd({**_GEMM_CMD_BASE, "max_tuners": 2})
    assert cmd[cmd.index("--max-tuners") + 1] == "2"


def test_no_gemm_ceiling_leaves_the_routed_tuner_set_untouched() -> None:
    """A zero ceiling means none could be derived, not "run nothing"."""
    from hyperloom.agents.kernel.tools import forge_gemm_tuning as fgt

    cmd = fgt._build_cmd(dict(_GEMM_CMD_BASE))
    assert "--max-tuners" not in cmd


# -- _gemm_tuning_workspace -----------------------------------------------
def test_gemm_tuning_workspace_explicit(tmp_path: Path) -> None:
    out = krh._gemm_tuning_workspace({"workspace_path": str(tmp_path / "ws")}, session_dir=tmp_path)
    assert out == tmp_path / "ws"


def test_gemm_tuning_workspace_from_task_id(tmp_path: Path) -> None:
    out = krh._gemm_tuning_workspace({"task_id": "t-1"}, session_dir=tmp_path)
    assert out == tmp_path / "runs" / "gemm_tuning" / "t-1"


def test_gemm_tuning_workspace_timestamp_fallback(tmp_path: Path) -> None:
    out = krh._gemm_tuning_workspace({}, session_dir=tmp_path)
    assert out.parent == tmp_path / "runs" / "gemm_tuning"
    assert out.name.startswith("request_")


# -- _write_gemm_tuning_benchmark_script ----------------------------------
def test_gemm_tuning_script_disables_the_eval(tmp_path: Path) -> None:
    path = krh._write_gemm_tuning_benchmark_script(
        workspace=tmp_path,
        model_path="/models/Qwen-Qwen3-8B",
        framework="sglang",
        gpu_type="mi355x",
        tp=1,
        conc=8,
        isl=256,
        osl=256,
    )
    script = path.read_text(encoding="utf-8")
    assert 'export RUN_EVAL="false"' in script
    assert "RUN_EVAL:-" not in script


# -- _optimization_budget_minutes / wrapper timeout -----------------------
# -- backend selection -----------------------------------------------------
def test_backend_order_ignores_payload_forge_without_explicit_env(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    assert krh._raw_kernel_backend_order({"backend_order": "FORGE,foo,unknown"}) == ["geak"]


def test_backend_order_exact_env_forge_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert krh._raw_kernel_backend_order({}) == ["forge"]


def test_backend_order_defaults_to_geak_phase_owner(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    assert krh._raw_kernel_backend_order({}) == ["geak"]


def test_removed_per_kernel_ladder_stays_removed(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    assert not hasattr(krh, "_backend_order")
    assert not hasattr(krh, "_run_backend_ladder")


# -- geak_selected ---------------------------------------------------------
def test_geak_selected_from_env_order(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "geak")
    assert krh.geak_selected() is True


def test_geak_selected_owns_phase_when_mixed(monkeypatch) -> None:
    # Mixed values are not the exact forge opt-in, so GEAK remains the owner.
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge,GEAK")
    assert krh.geak_selected() is True


def test_geak_selected_true_by_default(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    assert krh.geak_selected() is True


def test_geak_selected_false_only_for_exact_env_forge(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert krh.geak_selected({"backend_order": "geak"}) is False


def test_geak_selected_payload_cannot_override_exact_env_forge(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert krh.geak_selected({"backend_order": "geak"}) is False


# -- _artifact_paths_from_payload -----------------------------------------
def test_artifact_paths_string_wrapped() -> None:
    assert krh._artifact_paths_from_payload({"artifact_paths": "/a/b.so"}) == ["/a/b.so"]


def test_artifact_paths_list_filters_falsy() -> None:
    out = krh._artifact_paths_from_payload(
        {"compiled_artifact_paths": ["/a", "", None, "/b"]},
    )
    assert out == ["/a", "/b"]


def test_artifact_paths_other_type() -> None:
    assert krh._artifact_paths_from_payload({"artifact_paths": 42}) == []
    assert krh._artifact_paths_from_payload({}) == []


# -- _kernel_result_rank ---------------------------------------------------
# -- _parse_tool_stdout / _shape_tool_result ------------------------------
def test_parse_tool_stdout_whole_json() -> None:
    assert krh._parse_tool_stdout('{"status": "ok", "x": 1}') == {"status": "ok", "x": 1}


def test_parse_tool_stdout_empty() -> None:
    assert krh._parse_tool_stdout("   ") == {}


def test_parse_tool_stdout_last_line_json() -> None:
    out = krh._parse_tool_stdout('noise line\nmore noise\n{"status": "ok"}')
    assert out == {"status": "ok"}


def test_parse_tool_stdout_no_json_returns_tail() -> None:
    out = krh._parse_tool_stdout("just plain text, no json here")
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
    """The shape a tool with a lot to say actually emits.

    A tool that indents its result spans many lines, so the whole-text parse
    fails on the surrounding progress chatter and the per-line scan never sees a
    complete object. ``tracelens_analysis`` returned a megabyte of hot-kernel
    analysis exactly like this — the first time it ever succeeded — and every
    field of it was dropped.
    """
    out = krh._parse_tool_stdout(_PRETTY_TOOL_STDOUT)

    assert out["status"] == "ok"
    assert out["trace_report_path"] == "/s/tracelens/analysis.md"
    assert out["hot_kernels"][0]["name"] == "aten::_flash_attention_forward"


def test_shape_tool_result_will_not_call_unreadable_output_a_success() -> None:
    """Inferring ``ok`` from rc==0 made a tool whose output could not be read
    indistinguishable from one that worked, so the caller recorded an empty
    analysis over a real one and reported the leg as succeeded."""
    out = krh._shape_tool_result(0, "progress chatter, no json at all", "")

    assert out["status"] == "failed"
    assert out["error_class"] == "tool_output_unparseable"
    assert "raw_stdout_tail" in out


def test_shape_tool_result_uses_parsed_json() -> None:
    out = krh._shape_tool_result(0, '{"status": "ok", "kernel_id": "k1"}', "")
    assert out["status"] == "ok" and out["kernel_id"] == "k1"


def test_shape_tool_result_infers_status_and_stderr_tail() -> None:
    out = krh._shape_tool_result(1, '{"kernel_id": "k1"}', "boom error")
    assert out["status"] == "failed"
    assert out["returncode"] == 1
    assert out["stderr_tail"].endswith("boom error")


def test_shape_tool_result_synthesizes_on_empty_stdout() -> None:
    # empty stdout -> _parse_tool_stdout returns {} -> synthesize branch
    out = krh._shape_tool_result(2, "", "the stderr")
    assert out == {"status": "failed", "returncode": 2, "error": "the stderr"}


# -- _in_flight_kernel_ids -------------------------------------------------
# -- unattempted_skip_reason / gate-rejected dispatch ----------------------
def _state_owing_one_attempt():
    """SharedState whose trace still owes ``k001`` a kernel_opt attempt."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="skip-accounting")
    state.last_trace_analyze = {
        "hot_kernels": [
            {
                "kernel_id": "k001",
                "name": "gqa_decode_kernel",
                "gpu_pct": 8.5,
                "reusable_native_kernel": True,
                "source_file": "/pkg/aiter/gqa.py",
            }
        ],
        "task_groups": [],
    }
    return state


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
