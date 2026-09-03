# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supplemental coverage for kernel_request_handlers pure helpers: precision /
budget / timeout resolution, backend order, tool-stdout shaping, roofline name
lookup, artifact-path and in-flight scanning."""

from __future__ import annotations

import json
from pathlib import Path


from hyperloom.orchestrator.kernel import lane_budget
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


def test_a_ceiling_of_one_pins_the_highest_priority_tuner() -> None:
    assert krh._gemm_capped_tuner({}, max_targets=1, target_names=("fmoe_ck", "a8w8")) == "fmoe_ck"


def test_a_ceiling_that_fits_every_tuner_forces_none() -> None:
    assert krh._gemm_capped_tuner({}, max_targets=1, target_names=("fmoe_ck",)) == ""
    assert krh._gemm_capped_tuner({}, max_targets=5, target_names=("fmoe_ck", "a8w8")) == ""


def test_no_allocation_leaves_the_routed_tuner_set_untouched() -> None:
    """A zero ceiling means none could be derived, not "run nothing"."""
    assert krh._gemm_capped_tuner({}, max_targets=0, target_names=("fmoe_ck", "a8w8")) == ""


def test_an_explicit_tuner_outranks_the_lane_ceiling() -> None:
    assert krh._gemm_capped_tuner({"tuner": "a8w8"}, max_targets=1, target_names=("fmoe_ck",)) == "a8w8"


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
def test_optimization_budget_uses_payload_budget_minutes(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", raising=False)
    assert krh._optimization_budget_minutes({"backend_order": "forge", "budget_minutes": 20}) == 20.0


def test_optimization_budget_defaults_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", raising=False)
    assert krh._optimization_budget_minutes({}) == krh._DEFAULT_BACKEND_BUDGET_MINUTES


def test_rewrite_route_opt_in_lifts_the_budget_to_its_own_minimum(monkeypatch) -> None:
    """Opting into the rewrite route has to buy a budget the route accepts.

    A budget under the route's 75-minute minimum declines every eligible
    candidate for ``budget_insufficient`` and falls back to forge-loop, so the
    opt-in buys nothing. The shipped default clears the gate on its own now, so
    the case that matters is a budget tuned down for some other reason.
    """
    from hyperloom.agents.kernel.tools.backends._flydsl_rewrite import MIN_BUDGET_SEC

    monkeypatch.delenv("KERNEL_OPT_BACKEND_BUDGET_MIN", raising=False)
    monkeypatch.setenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", "1")

    assert krh._optimization_budget_minutes({}) * 60 >= MIN_BUDGET_SEC
    # A payload asking for less cannot drag it back under the minimum.
    assert krh._optimization_budget_minutes({"budget_minutes": 20}) * 60 >= MIN_BUDGET_SEC


def test_rewrite_route_floor_never_lowers_an_operator_budget(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", "1")
    monkeypatch.setenv("KERNEL_OPT_BACKEND_BUDGET_MIN", "180")
    assert krh._optimization_budget_minutes({}) == 180.0


def test_optimization_wrapper_timeout_adds_grace() -> None:
    secs = krh._optimization_wrapper_timeout_sec({"backend_order": "forge", "budget_minutes": 10})
    assert secs == 10 * 60 + 180


# -- _backend_order --------------------------------------------------------
def test_backend_order_ignores_payload_forge_without_explicit_env(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    assert krh._backend_order({"backend_order": "FORGE,foo,unknown"}) == []


def test_backend_order_exact_env_forge_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert krh._backend_order({}) == ["forge"]


def test_backend_order_default_is_empty_because_geak_owns_phase(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    out = krh._backend_order({})
    assert out == []


def test_backend_order_drops_geak_from_per_kernel_ladder(monkeypatch) -> None:
    # geak is a phase-level delegate, never a per-kernel backend.
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    assert krh._backend_order({"backend_order": "geak,forge"}) == []
    assert krh._backend_order({"backend_order": "geak"}) == []


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


# -- unattempted_skip_reason / gate-rejected dispatch ----------------------
def test_unattempted_skip_reason_covers_the_bookkeeping_reasons() -> None:
    """Only reasons meaning "no backend ran" count as unattempted."""
    assert krh.unattempted_skip_reason("below_min_gpu_pct=5.0")
    assert krh.unattempted_skip_reason("group_exhausted")
    assert krh.unattempted_skip_reason("group_in_flight")
    assert krh.unattempted_skip_reason("group_task_complete")
    assert krh.unattempted_skip_reason("opfanout_merged_into=k002")
    assert not krh.unattempted_skip_reason("")
    assert not krh.unattempted_skip_reason("non_reusable_kernel")
    # ``not_live`` covers in-flight, rejected, terminal and cap-exhausted
    # alike; only the first is unattempted, so it cannot be classified here.
    assert not krh.unattempted_skip_reason("not_live")


def test_the_skip_whitelist_covers_every_reason_the_filter_emits() -> None:
    """A reason the filter can emit but nothing classifies falls through the
    dispatcher's early return into its validation guards, which report a
    technical failure no backend produced.
    """
    emitted = {
        "below_min_gpu_pct=5.0",
        "group_exhausted",
        "group_in_flight",
        "group_task_complete",
        "not_live",
        "opfanout_merged_into=k002",
    }
    unclassified = {r for r in emitted if not krh.unattempted_skip_reason(r)}
    assert unclassified == {"not_live"}, (
        f"either whitelist the reason or decide it means a real attempt: {sorted(unclassified)}"
    )


def test_batch_candidates_reports_its_skip_reasons(tmp_path: Path) -> None:
    """The filter's reasons reach the caller, not just the log."""
    artifact = tmp_path / "kernel_candidates.json"
    artifact.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k001",
                        "name": "cold_kernel",
                        "gpu_pct": 1.0,
                        "reusable_native_kernel": True,
                        "source_file": "/pkg/k.py",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    skipped: dict[str, str] = {}
    selected = krh._batch_kernel_candidates(
        {"candidates_path": str(artifact)},
        skipped_out=skipped,
    )
    assert selected == []
    assert skipped["k001"].startswith("below_min_gpu_pct")


def test_gate_rejected_named_kernel_is_skipped_not_failed(tmp_path: Path) -> None:
    """A threshold is not an optimization failure.

    Recording one spends the source's retry quota on a decision no backend made,
    and the report then explains a technical failure that never happened.
    """
    import asyncio

    artifact = tmp_path / "kernel_candidates.json"
    artifact.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k001",
                        "name": "cold_kernel",
                        "gpu_pct": 1.0,
                        "reusable_native_kernel": True,
                        "source_file": "/pkg/k.py",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    session = tmp_path / "session"
    session.mkdir()
    out = asyncio.run(
        krh.run_optimization_handler(
            {"candidates_path": str(artifact), "kernel_id": "k001"},
            session_dir=session,
        )
    )
    assert out["status"] == "skipped"
    assert out["reason"].startswith("below_min_gpu_pct")
    assert out["kernel_id"] == "k001"


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


def test_an_unattempted_skip_leaves_the_attempt_ledger_alone() -> None:
    """The kernel still owes an attempt, so it must stay in the queue.

    A ledger row is what removes it: ``untried_hot_reusable_kernels`` drops a
    kernel whose recorded source is empty once ``attempts`` is above zero, and
    that queue is what both the KERNEL-entry dispatch and the phase-advance gate
    ask. The row also reads as a decision-less attempt, which the summary
    reports as a failure that never happened.
    """
    for reason in (
        "below_min_gpu_pct=5.0",
        "group_exhausted",
        "group_in_flight",
        "group_task_complete",
        "opfanout_merged_into=k002",
    ):
        state = _state_owing_one_attempt()
        krh.record_kernel_opt(
            state,
            {"status": "skipped", "reason": reason, "kernel_id": "k001"},
        )
        assert state.kernel_opt_task_attempts == {}, reason
        assert state.untried_hot_reusable_kernels() == ["k001"], reason


def test_the_breakdown_records_which_gate_declined(tmp_path: Path) -> None:
    """ "skipped" is the same word for every gate; the reason names which one.

    The ledger exemption above keeps an unattempted skip from reading as a
    failure, and the canonical breakdown is where that distinction has to
    survive. Recording ``status`` there spent the whole fix: a reader could tell
    a skip from an attempt and could not tell a sub-floor kernel from one merged
    into an op-fanout representative or a group already in flight.
    """
    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts

    for reason in ("below_min_gpu_pct=5.0", "opfanout_merged_into=k002", "group_in_flight"):
        session = tmp_path / reason.replace("=", "_").replace(".", "_")
        session.mkdir()
        state = _state_owing_one_attempt()
        state._session_dir = session
        krh.record_kernel_opt(state, {"status": "skipped", "reason": reason, "kernel_id": "k001"})

        kernels = assemble_parts(session)["kernel_journey"]["kernels"]
        entry = next(k for k in kernels if k["kernel_id"] == "k001")
        assert entry["dispatch"]["dispatched"] is False, reason
        assert entry["dispatch"]["skip_reason"] == reason, reason


def test_an_in_flight_hold_is_unattempted_but_a_spent_one_is_not() -> None:
    """One ``not_live`` covered three states that need different answers.

    A kernel held back because a sibling dispatch is in flight has had no
    backend look at it, so charging it an attempt retires it from the phase's
    own work list for a dispatch that never happened. A kernel held back for a
    rejection or an exhausted attempt cap really did spend its attempts, so
    exempting those would let it be retried forever.
    """
    assert krh.unattempted_skip_reason("not_live_in_flight")
    assert not krh.unattempted_skip_reason("not_live_rejected")
    assert not krh.unattempted_skip_reason("not_live_attempts_exhausted")
    # The old undifferentiated reason must not read as unattempted either.
    assert not krh.unattempted_skip_reason("not_live")

    state = _state_owing_one_attempt()
    krh.record_kernel_opt(
        state,
        {"status": "skipped", "reason": "not_live_in_flight", "kernel_id": "k001"},
    )
    assert state.kernel_opt_task_attempts == {}
    assert state.untried_hot_reusable_kernels() == ["k001"]


def test_a_real_failure_still_spends_its_attempt() -> None:
    """The exemption is scoped to skips; a backend that ran and failed counts."""
    state = _state_owing_one_attempt()
    krh.record_kernel_opt(
        state,
        {
            "status": "failed",
            "kernel_id": "k001",
            "error_class": "CompileError",
            "source_file": "/pkg/aiter/gqa.py",
        },
    )
    assert state.kernel_opt_task_attempts
    assert state.untried_hot_reusable_kernels() == []


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


def test_the_dispatcher_publishes_the_merge_it_performed(tmp_path: Path) -> None:
    """The representative carries the ids it absorbed, and the result says so.

    Nothing in production read ``opfanout_collapsed_ids`` before: the batch
    filter stamped it and no consumer existed, so the merge was invisible to
    everything downstream of the dispatch.
    """
    artifact = tmp_path / "kernel_candidates.json"
    artifact.write_text(json.dumps({"hot_kernels": _OP_FANOUT_ROWS}), encoding="utf-8")

    skipped: dict[str, str] = {}
    selected = krh._batch_kernel_candidates(
        {"candidates_path": str(artifact)},
        skipped_out=skipped,
    )
    assert [str(row.get("kernel_id") or "") for row in selected] == ["k001"]
    assert skipped["k002"] == "opfanout_merged_into=k001"
    assert set(selected[0]["opfanout_collapsed_ids"]) == {"k001", "k002"}

    stamped = krh._stamp_task_group_result({"status": "ok"}, selected[0])
    assert set(stamped["opfanout_collapsed_ids"]) == {"k001", "k002"}


def test_an_op_fanout_merge_does_not_leave_the_sibling_owing_an_attempt() -> None:
    """The merge is reported as unattempted, so it writes no row of its own.

    The representative's row is therefore the only record that the sibling was
    handled at all. Without it the sibling stays in the work queue for a dispatch
    that cannot happen: ``kernel_work_pending()`` never goes False, KERNEL
    redispatches the entry batch every tick, and the orchestration prompt forbids
    ``report`` while the queue is non-empty -- a spin to the wall clock.

    Retirement follows the recorded merge, not the shared file: two ops in one
    file the dispatcher never merged stay two units of work, which several
    real-session regressions in test_shared_state_kernel_opt.py depend on.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="op-fanout-ledger")
    state.last_trace_analyze = {"hot_kernels": _OP_FANOUT_ROWS, "task_groups": []}

    # The merged sibling is reported, and deliberately records nothing.
    krh.record_kernel_opt(
        state,
        {"status": "skipped", "reason": "opfanout_merged_into=k001", "kernel_id": "k002"},
    )
    assert state.kernel_opt_task_attempts == {}
    assert set(state.untried_hot_reusable_kernels()) == {"k001", "k002"}

    # The representative runs, and its row carries what it absorbed.
    krh.record_kernel_opt(
        state,
        {
            "status": "failed",
            "kernel_id": "k001",
            "error_class": "CompileError",
            "source_file": "/pkg/aiter/gqa.py",
            "opfanout_collapsed_ids": ["k001", "k002"],
        },
    )
    entry = next(iter(state.kernel_opt_task_attempts.values()))
    assert set(entry["opfanout_collapsed_ids"]) == {"k001", "k002"}
    assert state.untried_hot_reusable_kernels() == []
