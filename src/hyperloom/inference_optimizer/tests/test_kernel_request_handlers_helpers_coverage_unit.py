# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplemental coverage for kernel_request_handlers pure helpers: precision /
budget / timeout resolution, backend order, tool-stdout shaping, roofline name
lookup, artifact-path and in-flight scanning."""

from __future__ import annotations

import json
from pathlib import Path


from hyperloom.orchestrator.kernel import request_handlers as krh


# -- _normalize_precision / _normalize_kernel_id --------------------------
def test_normalize_precision() -> None:
    assert krh._normalize_precision("  FP8 ") == "fp8"
    assert krh._normalize_precision(None) == ""
    assert krh._normalize_precision(0) == ""


def test_normalize_kernel_id_folds_prefixes() -> None:
    assert krh._normalize_kernel_id("kn12") == "k12"
    assert krh._normalize_kernel_id("RN7") == "k7"
    assert krh._normalize_kernel_id("k3") == "k3"
    assert krh._normalize_kernel_id("knabc") == "knabc"  # non-digit tail untouched
    assert krh._normalize_kernel_id(None) == ""


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


# -- _optimization_budget_minutes / wrapper timeout -----------------------
def test_optimization_budget_uses_payload_budget_minutes() -> None:
    assert krh._optimization_budget_minutes({"backend_order": "forge", "budget_minutes": 20}) == 20.0


def test_optimization_budget_defaults_when_unset() -> None:
    assert krh._optimization_budget_minutes({}) == krh._DEFAULT_BACKEND_BUDGET_MINUTES


def test_optimization_wrapper_timeout_adds_grace() -> None:
    secs = krh._optimization_wrapper_timeout_sec({"backend_order": "forge", "budget_minutes": 10})
    assert secs == 10 * 60 + 180


# -- _backend_order --------------------------------------------------------
def test_backend_order_explicit_payload(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    # Unknown backends are filtered out of the ladder.
    assert krh._backend_order({"backend_order": "FORGE,foo,unknown"}) == ["forge"]


def test_backend_order_env_alias(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    # When nothing survives filtering, fall back to the legacy native ladder.
    monkeypatch.setenv("KERNEL_OPT_BACKENDS", "codex,claude")
    assert krh._backend_order({}) == ["forge"]


def test_backend_order_unknown_backends_yield_empty(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    # Unknown backends leave nothing to run and no legacy fallback.
    monkeypatch.setenv("KERNEL_OPT_BACKENDS", "foo,bar")
    assert krh._backend_order({}) == []


def test_backend_order_default_is_empty_because_geak_owns_phase(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    out = krh._backend_order({})
    assert out == []


def test_backend_order_drops_geak_from_per_kernel_ladder(monkeypatch) -> None:
    # geak is a phase-level delegate, never a per-kernel backend.
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    assert krh._backend_order({"backend_order": "geak,forge"}) == ["forge"]
    assert krh._backend_order({"backend_order": "geak"}) == []


# -- geak_selected ---------------------------------------------------------
def test_geak_selected_from_env_order(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "geak")
    assert krh.geak_selected() is True


def test_geak_selected_owns_phase_when_mixed(monkeypatch) -> None:
    # geak owns the phase when it appears anywhere in the order.
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge,GEAK")
    assert krh.geak_selected() is True


def test_geak_selected_true_by_default(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    assert krh.geak_selected() is True


def test_geak_selected_false_for_explicit_native_order(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    monkeypatch.delenv("KERNEL_OPT_BACKENDS", raising=False)
    assert krh.geak_selected({"backend_order": "forge,claude"}) is False


def test_geak_selected_payload_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert krh.geak_selected({"backend_order": "geak"}) is True


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
def test_kernel_result_rank_non_dict() -> None:
    assert krh._kernel_result_rank(None) == (0, 0.0)


def test_kernel_result_rank_keep_beats_higher_micro_review() -> None:
    keep = {
        "status": "ok",
        "proposal": {"decision": "KEEP"},
        "verification": {"micro_speedup": 1.05},
    }
    review = {
        "status": "ok",
        "proposal": {"decision": "NEEDS_REVIEW"},
        "verification": {"micro_speedup": 2.0},
    }
    assert krh._kernel_result_rank(keep) > krh._kernel_result_rank(review)


def test_kernel_result_rank_equal_keep_higher_micro_wins() -> None:
    a = {"status": "ok", "proposal": {"decision": "KEEP"}, "verification": {"micro_speedup": 1.1}}
    b = {"status": "ok", "proposal": {"decision": "KEEP"}, "verification": {"micro_speedup": 1.5}}
    assert krh._kernel_result_rank(b) > krh._kernel_result_rank(a)


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


# -- _lookup_kernel_roofline_name -----------------------------------------
def test_lookup_roofline_name_missing_sidecar(tmp_path: Path) -> None:
    assert krh._lookup_kernel_roofline_name(tmp_path, "k1") == ""


def test_lookup_roofline_name_found(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "kernel_roofline.json").write_text(
        json.dumps(
            {
                "kernels": [
                    {"kernel_id": "k1", "name": "fused_moe_kernel"},
                    {"kernel_id": "k2", "matched_kernel_name": "attn"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert krh._lookup_kernel_roofline_name(tmp_path, "k1") == "fused_moe_kernel"
    assert krh._lookup_kernel_roofline_name(tmp_path, "k2") == "attn"
    assert krh._lookup_kernel_roofline_name(tmp_path, "k9") == ""


def test_lookup_roofline_name_bad_shape(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "kernel_roofline.json").write_text(
        json.dumps({"kernels": "not-a-list"}),
        encoding="utf-8",
    )
    assert krh._lookup_kernel_roofline_name(tmp_path, "k1") == ""


# -- _in_flight_kernel_ids -------------------------------------------------
def test_in_flight_kernel_ids_no_dir(tmp_path: Path) -> None:
    assert krh._in_flight_kernel_ids(tmp_path) == set()


def test_in_flight_kernel_ids_scans_running(tmp_path: Path) -> None:
    sid = tmp_path.name
    status_dir = tmp_path / "kernel-agent" / "runs" / sid / "status" / "kernel_optimization"
    status_dir.mkdir(parents=True)
    # running with kernel_id in last_lines
    (status_dir / "ko-1.json").write_text(
        json.dumps({"state": "running", "last_lines": ["foo", "kernel_id=k7"]}),
        encoding="utf-8",
    )
    # running with top-level kernel_id fallback
    (status_dir / "ko-2.json").write_text(
        json.dumps({"state": "RUNNING", "kernel_id": "k8"}),
        encoding="utf-8",
    )
    # finished -> ignored
    (status_dir / "ko-3.json").write_text(
        json.dumps({"state": "done", "kernel_id": "k9"}),
        encoding="utf-8",
    )
    # malformed -> skipped
    (status_dir / "ko-4.json").write_text("{bad", encoding="utf-8")
    assert krh._in_flight_kernel_ids(tmp_path) == {"k7", "k8"}
