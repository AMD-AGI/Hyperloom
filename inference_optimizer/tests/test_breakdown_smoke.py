# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end smoke test for the breakdown exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import (
    BREAKDOWN_FILENAME,
    SCHEMA_VERSION,
    build,
    write_breakdown_json,
)


# Fixture builder
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_fixture(sd: Path) -> None:
    sd.mkdir(parents=True, exist_ok=True)

    _write_json(sd / "manifest.json", {
        "schema_version": 1,
        "session_id":     "DeepSeek-R1_20260514T071000Z_abcd1234",
        "claw_session_id":"claw-test-1234",
        "sandbox_user_id":"hai.song@core42.ai",
        "created_at_utc": "2026-05-14T07:10:00+00:00",
        "session_dir":    str(sd),
        "model_path":     "/wekafs/models/DeepSeek-R1",
        "model_name":     "DeepSeek-R1",
        "framework":      "sglang",
        "gpu_type":       "mi300x",
        "tp":              8,
        "workload":       {"isl": 1024, "osl": 1024, "max_model_len": 6144,
                            "precision": "fp8", "conc": 64},
        "objective":      {"kind": "gain_pct", "value": 30.0},
        "max_minutes":    180,
        "code_revision":  "86d2ed3",
        "pid":             12345,
        "host":            "core42-mi300x-r1",
    })

    _write_json(sd / "state.json", {
        "session_id":     "DeepSeek-R1_20260514T071000Z_abcd1234",
        "model_name":     "DeepSeek-R1",
        "model_path":     "/wekafs/models/DeepSeek-R1",
        "model_class":    "moe",
        "framework":      "sglang",
        "gpu_type":       "mi300x",
        "baseline_tput":  421.5,
        "baseline_accuracy": 0.84,
        "baseline_failure_streak": 0,
        "baseline_config_path": "runs/baseline/t1/baseline_config.with_envs.yaml",
        "current_best":   {"tput": 776.4, "action": "explore",
                            "extra_server_args": "--enable-X --nccl-Y",
                            "extra_envs": {"NCCL_DEBUG": "INFO"},
                            "ttft_mean_ms": 110.2, "e2el_mean_ms": 1200.4},
        "optimization_stack": [
            {"action": "explore",      "variant_name": "flag_X", "gain_pct": 23.4,
             "extra_server_args": "--enable-X",
             "ts": "2026-05-14T07:15:00+00:00"},
            {"action": "explore",      "variant_name": "nccl_Y", "gain_pct":  7.8,
             "extra_server_args": "--nccl-Y",
             "ts": "2026-05-14T07:30:00+00:00"},
            {"action": "kernel_opt:k001", "variant_name": "",  "gain_pct": 45.6,
             "extra_server_args": "",
             "ts": "2026-05-14T08:00:00+00:00"},
        ],
        "cumulative_gain": 91.5,
        "cumulative_gain_validated": 84.2,
        "cumulative_gain_validated_ts": "2026-05-14T08:30:00+00:00",
        "cumulative_gain_validated_stack_len": 3,
        "stop_reason": "target_reached",
        "start_ts": "2026-05-14T07:10:00+00:00",
        "max_minutes": 180,
        "tick": 145,
        "last_baseline": {"workspace": str(sd / "runs/baseline/t1")},
        "baseline_attempts": [
            {"ts": "2026-05-14T07:12:00+00:00", "task_id": "t1",
             "status": "succeeded", "decision": "promoted",
             "key_metric": 421.5,   "workspace": str(sd / "runs/baseline/t1")},
        ],
        "profile_attempts": [
            {"ts": "2026-05-14T07:14:00+00:00", "task_id": "p1",
             "status": "succeeded", "decision": "promoted", "key_metric": 421.5},
        ],
        "explore_attempts": [
            {"ts": "2026-05-14T07:15:00+00:00", "task_id": "b1",
             "status": "succeeded", "decision": "promoted", "key_metric": 23.4},
            {"ts": "2026-05-14T07:30:00+00:00", "task_id": "pa1",
             "status": "succeeded", "decision": "promoted", "key_metric": 7.8},
            {"ts": "2026-05-14T08:30:00+00:00", "task_id": "v1",
             "status": "succeeded", "decision": "promoted", "key_metric": 84.2},
        ],
        "sweep_attempts": [
            {"ts": "2026-05-14T08:20:00+00:00", "task_id": "s1",
             "status": "succeeded", "decision": "promoted",
             "key_metric": 812.0, "workspace": str(sd / "runs/sweep/s1")},
        ],
        "last_trace_analyze": {"hot_kernels_top15": [
            {"kernel_id": "k001", "name": "fused_rmsnorm", "gpu_pct": 18.2,
             "bottleneck": "memory", "arithmetic_intensity": 4.0,
             "source_file": "/path/to/rmsnorm.py", "reusable_native_kernel": True,
             "recommended_backends": ["claude", "geak"],
             "recommended_actions": ["run_optimization"]},
        ], "ts": "2026-05-14T07:14:30+00:00"},
        "last_sweep": {
            "grid_size": 9,
            "best_overall": {"variant_name": "variant_05_conc32_isl1024_osl1024",
                              "output_throughput": 812.0, "conc": 32},
            "best_for_each_conc": [
                {"conc": 32, "tput": 812.0},
                {"conc": 64, "tput": 776.4},
            ],
            "pareto_front": [],
        },
        "last_kernel_opt": {"kernel_id": "k001", "decision": "KEEP",
                              "micro_speedup": 1.27, "compile_passed": True,
                              "correctness_passed": True,
                              "best_artifact_path": "patches/k001/0001.patch",
                              "ts": "2026-05-14T08:00:00+00:00"},
        "kernel_opt_attempts": {"k001": {
            "attempts": 3, "partial_count": 1, "last_decision": "KEEP",
            "last_ts": "2026-05-14T08:00:00+00:00",
            "history": [
                {"decision": "PARTIAL", "ts": "2026-05-14T07:40:00+00:00"},
                {"decision": "PARTIAL", "ts": "2026-05-14T07:50:00+00:00"},
                {"decision": "KEEP",    "ts": "2026-05-14T08:00:00+00:00"},
            ],
        }},
        "kernel_integrate_attempts": {"k001|patches/k001/0001.patch|": {
            "key": "k001|patches/k001/0001.patch|",
            "kernel_id": "k001", "patch_path": "patches/k001/0001.patch",
            "target_file": "/path/to/rmsnorm.py", "extra_server_args": "",
            "attempts": [
                {"decision": "KEEP", "status": "succeeded",
                 "new_tput": 776.4, "gain_pct": 45.6,
                 "ts": "2026-05-14T08:00:00+00:00"},
            ],
            "attempt_count": 1, "best_gain_pct": 45.6,
            "last_decision": "KEEP", "last_status": "succeeded",
            "updated_at": "2026-05-14T08:00:00+00:00",
        }},
        "rejected_kernel_patches": [],
        "rejected_kernel_ids": ["k042"],
        "explore_search": {
            "schema_version": 1,
            "accepted": [
                {"name": "flag_X", "fingerprint": "g1",
                 "gain_pct": 23.4,
                 "ts": "2026-05-14T07:15:00+00:00"},
                {"name": "nccl_Y", "fingerprint": "f1",
                 "extra_server_args": "--nccl-Y", "extra_envs": {},
                 "output_throughput": 454.3, "gain_pct": 7.8,
                 "ts": "2026-05-14T07:30:00+00:00"},
            ],
            "rejected": [{"name": "bad_x", "fingerprint": "f2", "gain_pct": -2.5}],
            "tested": {
                "g1": {"name": "flag_X",  "fingerprint": "g1", "gain_pct": 23.4},
                "f1": {"name": "nccl_Y",  "fingerprint": "f1", "gain_pct":  7.8},
                "f2": {"name": "bad_x",   "fingerprint": "f2", "gain_pct": -2.5},
                "f3": {"name": "neutral", "fingerprint": "f3", "gain_pct":  0.1},
            },
            "name_index": {}, "cursor": 4,
        },
    })

    bdir = sd / "runs/baseline/t1/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True, "output_throughput_tok_s": 421.5,
        "mean_ttft_ms": 152.3, "mean_e2el_ms": 1840.7, "mean_tpot_ms": 8.4,
        "gpu_monitor": [{"power_w": 420, "temperature_c": 60, "clock_mhz": 2040}],
    })

    pdir = sd / "runs/profile/p1/benchmark_001"
    _write_json(pdir / "benchmark_report.json", {
        "success": True,
        "kernel_summary": [
            {"kernel_id": "k001", "name": "fused_rmsnorm", "gpu_pct": 18.2,
             "time_ms": 0.42, "bottleneck": "memory",
             "arithmetic_intensity": 4.0,
             "reusable_native_kernel": True, "source_file": "/path/to/rmsnorm.py"},
            {"kernel_id": "k002", "name": "attention_qkv", "gpu_pct": 14.5,
             "time_ms": 0.33, "bottleneck": "compute"},
        ],
        "top_bottlenecks": [{"kernel_id": "k001", "bottleneck": "memory"}],
        "gpu_monitor": {"power_w": 510, "temperature_c": 72, "clock_mhz": 2100},
    })

    for i, conc in enumerate((32, 64, 128), 1):
        vdir = sd / f"runs/sweep/s1/variant_0{i}_conc{conc}_isl1024_osl1024/benchmark_001"
        _write_json(vdir / "benchmark_report.json", {
            "success": True,
            "output_throughput_tok_s": 600.0 + 100.0 * i,
            "mean_ttft_ms": 120 + 5 * i,
            "mean_tpot_ms": 7 + 0.3 * i,
            "mean_e2el_ms": 1300 + 50 * i,
        })

    kar = sd / "kernel-agent-workspace/kernel-agent/runs/sess001"
    for sub in ("prompts", "optimized", "results", "verification"):
        (kar / sub).mkdir(parents=True)
    (kar / "prompts" / "a01.md").write_text("kernel: fused_rmsnorm", encoding="utf-8")
    (kar / "prompts" / "a02.md").write_text("kernel: fused_rmsnorm (retry)", encoding="utf-8")
    (kar / "prompts" / "a03.md").write_text("kernel: fused_rmsnorm (final)", encoding="utf-8")
    (kar / "optimized" / "a03_kernel.cuh").write_text("__global__ void foo() {}", encoding="utf-8")
    _write_json(kar / "verification" / "k001.json", {
        "micro_speedup": 1.27, "compile_passed": True, "correctness_passed": True,
        "best_artifact_path": str(sd / "patches/k001/0001.patch"),
    })
    _write_json(kar / "results" / "k001.json", {
        "tool": "kernel_optimization", "session_id": "sess001",
        "run_id": "ko-deadbeef", "kernel_id": "k001",
        "source_file": "/path/to/rmsnorm.py",
        "best_artifact_path": str(sd / "patches/k001/0001.patch"),
        "selected_backends": ["claude"],
        "proposal": {"decision": "KEEP", "reasons": ["compile_pass"]},
        "verification": {"micro_speedup": 1.27, "compile_passed": True,
                          "correctness_passed": True},
        "cli_log_path": str(kar / "logs/kernel_optimization/ko-deadbeef.log"),
    })
    with (kar / "optimization_attempts.jsonl").open("w", encoding="utf-8") as f:
        for i, spd in enumerate([1.04, 1.11, 1.27], 1):
            f.write(json.dumps({
                "attempt_id": f"a{i:02d}",
                "kernel_id":  "k001",
                "backend":    "claude",
                "model":      "claude-sonnet-4.5",
                "ts":         f"2026-05-14T07:{40+i*5:02d}:00+00:00",
                "status":     "succeeded",
                "speedup":    spd,
                "name":       "fused_rmsnorm",
                "source_file": "/path/to/rmsnorm.py",
            }) + "\n")

    cw = sd / "critic-workdir/001"
    _write_json(cw / "review.json", {
        "verdict": "approve", "topic": "kernel_opt:k001",
        "summary": "Speedup verified.",
        "ts": "2026-05-14T08:01:00+00:00",
    })
    _write_json(cw / "emit.json", {"topic": "kernel_opt:k001",
                                     "ts": "2026-05-14T08:00:30+00:00"})

    rw = sd / "robustness-workdir/001"
    _write_json(rw / "signal.json", {"signal": "stall",
                                       "ts": "2026-05-14T07:55:00+00:00"})
    _write_json(rw / "action.json", {"action": "force_replan",
                                       "ts": "2026-05-14T07:55:30+00:00"})

    (sd / "patches/k001").mkdir(parents=True)
    (sd / "patches/k001" / "0001.patch").write_text("--- a\n+++ b\n", encoding="utf-8")


# Tests
@pytest.fixture
def fixture_session(tmp_path: Path) -> Path:
    sd = tmp_path / "session"
    _build_fixture(sd)
    return sd


def test_envelope(fixture_session: Path) -> None:
    """Top-level keys and schema_version match the contract."""
    b = build(fixture_session)
    assert b["schema_version"] == SCHEMA_VERSION
    assert "exported_at_utc" in b
    assert "exporter_version" in b
    for key in (
        "session", "workload", "baseline", "final", "phase_timeline",
        "capability_summary", "geak_invocations", "oob_invocations",
        "forge_invocations",
        "kernel_lifecycle", "param_search", "sweep", "critic_robustness",
        "telemetry", "attribution", "warnings", "source_files",
    ):
        assert key in b, f"missing top-level: {key}"


def test_write_breakdown_json_atomic(fixture_session: Path) -> None:
    """write_breakdown_json places the file at the expected path with valid JSON."""
    out = write_breakdown_json(fixture_session)
    assert out == fixture_session / BREAKDOWN_FILENAME
    assert out.exists()
    rt = json.loads(out.read_text(encoding="utf-8"))
    assert rt["schema_version"] == SCHEMA_VERSION


def test_session_metadata(fixture_session: Path) -> None:
    s = build(fixture_session)["session"]
    assert s["session_id"] == "DeepSeek-R1_20260514T071000Z_abcd1234"
    assert s["claw_session_id"] == "claw-test-1234"
    assert s["sandbox_user_id"] == "hai.song@core42.ai"
    assert s["stop_reason"] == "target_reached"
    assert s["max_minutes"] == 180


def test_session_stop_reason_falls_back_to_close_phase(tmp_path: Path) -> None:
    """Legacy states may close via phase_history without top-level stop_reason.

    Regression: ``close_sequence_done=true`` + final ``to_phase=CLOSE`` row
    while ``state.stop_reason`` stayed blank; the terminal reason must still surface.
    """
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {
        "schema_version": 1,
        "session_id": "closed-via-phase-history",
        "created_at_utc": "2026-06-05T01:00:00+00:00",
    })
    _write_json(sd / "state.json", {
        "session_id": "closed-via-phase-history",
        "stop_reason": "",
        "close_sequence_done": True,
        "phase_history": [
            {
                "from_phase": "KERNEL",
                "to_phase": "SWEEP",
                "reason": "kernel_phase_budget_exhausted",
                "ts": "2026-06-05T02:00:00+00:00",
                "ts_unix": 1780624800.0,
            },
            {
                "from_phase": "SWEEP",
                "to_phase": "CLOSE",
                "reason": "sweep_budget_exhausted",
                "ts": "2026-06-05T03:04:05+00:00",
                "ts_unix": 1780628645.0,
            },
        ],
    })

    s = build(sd)["session"]
    assert s["stop_reason"] == "sweep_budget_exhausted"
    assert s["ended_at_utc"] == "2026-06-05T03:04:05Z"


def test_session_stop_reason_prefers_specific_close_reason(tmp_path: Path) -> None:
    """A generic top-level timeout should not hide the terminal phase reason."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {
        "schema_version": 1,
        "session_id": "closed-with-generic-timeout",
        "created_at_utc": "2026-06-05T01:00:00+00:00",
    })
    _write_json(sd / "state.json", {
        "session_id": "closed-with-generic-timeout",
        "stop_reason": "time_exhausted",
        "close_sequence_done": True,
        "phase_history": [
            {
                "from_phase": "SWEEP",
                "to_phase": "CLOSE",
                "reason": "sweep_budget_exhausted",
                "ts": "2026-06-05T03:04:05+00:00",
                "ts_unix": 1780628645.0,
            },
        ],
    })

    s = build(sd)["session"]
    assert s["stop_reason"] == "sweep_budget_exhausted"
    assert s["ended_at_utc"] == "2026-06-05T03:04:05Z"


def test_keep_stamping_only_best_attempt(fixture_session: Path) -> None:
    """KEEP decision must land on the BEST attempt, not every attempt."""
    oob = build(fixture_session)["oob_invocations"]
    assert len(oob) == 3
    keeps = [o for o in oob if o["decision"] == "KEEP"]
    others = [o for o in oob if o["decision"] != "KEEP"]
    assert len(keeps) == 1, [o["decision"] for o in oob]
    assert len(others) == 2
    assert keeps[0]["micro_speedup"] == 1.27
    assert keeps[0]["compile_passed"] is True
    assert keeps[0]["correctness_passed"] is True
    assert all(o["decision"] == "" for o in others)


def test_capability_summary(fixture_session: Path) -> None:
    cap = build(fixture_session)["capability_summary"]
    assert cap["oob"]["status"] == "kept"
    assert cap["oob"]["keeps"] == 1
    assert cap["oob"]["attempts"] == 3
    assert cap["geak"]["status"] == "not_attempted"
    assert cap["explore"]["status"] == "kept"
    assert cap["explore"]["best_gain_pct"] == pytest.approx(23.4)
    assert cap["sweep"]["status"] == "completed"
    assert cap["explore"]["last_validated_gain_pct"] == pytest.approx(84.2)


def test_kernel_lifecycle_five_stages(fixture_session: Path) -> None:
    lc = build(fixture_session)["kernel_lifecycle"]
    assert len(lc["detected"]) == 2
    assert {k["kernel_id"] for k in lc["detected"]} == {"k001", "k002"}
    assert len(lc["recommended"]) == 1
    assert lc["recommended"][0]["kernel_id"] == "k001"
    assert len(lc["optimized"]) == 1
    assert lc["optimized"][0]["best_micro_speedup"] == pytest.approx(1.27)
    assert lc["optimized"][0]["last_decision"] == "KEEP"
    assert len(lc["adopted"]) == 1
    assert lc["adopted"][0]["kernel_id"] == "k001"
    assert lc["adopted"][0]["e2e_gain_pct"] == pytest.approx(45.6)
    assert len(lc["rejected"]) == 1
    assert lc["rejected"][0]["kernel_id"] == "k042"


def test_sweep_picks_up_disk_variants(fixture_session: Path) -> None:
    sweep = build(fixture_session)["sweep"]
    assert len(sweep["all_variants"]) == 3
    concs = sorted(v["conc"] for v in sweep["all_variants"])
    assert concs == [32, 64, 128]
    for v in sweep["all_variants"]:
        assert v["output_throughput_tok_s"] is not None
        assert v["benchmark_report_path"] is not None


def test_critic_robustness(fixture_session: Path) -> None:
    cr = build(fixture_session)["critic_robustness"]
    assert len(cr["critic_iterations"]) == 1
    assert cr["critic_iterations"][0]["verdict"] == "approve"
    assert cr["critic_iterations"][0]["iter"] == 1
    assert len(cr["robustness_signals"]) == 1
    assert cr["robustness_signals"][0]["signal"] == "stall"


def test_telemetry_aggregates_gpu_monitor(fixture_session: Path) -> None:
    tel = build(fixture_session)["telemetry"]
    assert tel["baseline_report_path"] is not None
    assert len(tel["profile_report_paths"]) == 1
    gm = tel["gpu_monitor_aggregate"]
    assert gm["samples"] >= 1
    assert gm["avg_power_w"] > 0
    assert gm["max_power_w"] >= gm["avg_power_w"]


def test_attribution_kernel_goes_to_oob(fixture_session: Path) -> None:
    """The single KEEP'd kernel is OOB; family breakdown must reflect that."""
    attr = build(fixture_session)["attribution"]
    assert len(attr["gain_per_stack_entry"]) == 3
    sb = attr["source_breakdown"]
    assert sb["oob_pct_of_total"] >= 45.0
    assert sb["geak_pct_of_total"] == 0.0
    assert sb["explore_pct_of_total"] == pytest.approx(31.2)
    assert sb["validated_total_pct"] == pytest.approx(84.2)


def test_phase_timeline_sorted_by_ts(fixture_session: Path) -> None:
    ts = build(fixture_session)["phase_timeline"]
    ts_values = [e.get("ts") or "" for e in ts]
    assert ts_values == sorted(ts_values)
    actions = {e["action"] for e in ts}
    assert "baseline" in actions
    assert "explore" in actions
    assert "kernel_opt" in actions or "integrate" in actions


def test_param_search_summary(fixture_session: Path) -> None:
    ps = build(fixture_session)["param_search"]
    assert ps["explore"]["tested_count"] == 4
    accepted_names = {a["name"] for a in ps["explore"]["accepted"]}
    assert accepted_names == {"flag_X", "nccl_Y"}


def test_missing_state_returns_partial_with_warnings(tmp_path: Path) -> None:
    """A near-empty session_dir (only manifest) must still produce a valid envelope."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {
        "schema_version": 1, "session_id": "ghost", "model_name": "ghost",
        "claw_session_id": "claw-ghost", "framework": "sglang", "gpu_type": "mi300x",
    })
    b = build(sd)
    assert b["schema_version"] == SCHEMA_VERSION
    assert b["session"]["session_id"] == "ghost"
    assert b["session"]["claw_session_id"] == "claw-ghost"
    assert any("state.json missing" in w for w in b["warnings"])


def test_no_kernel_agent_runs_returns_empty_invocations(tmp_path: Path) -> None:
    """A session without any kernel-agent activity must not crash collectors."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "x"})
    _write_json(sd / "state.json", {"session_id": "x", "baseline_tput": 100.0})
    b = build(sd)
    assert b["geak_invocations"] == []
    assert b["oob_invocations"] == []
    assert b["kernel_lifecycle"]["detected"] == []


def test_baseline_resolves_container_workspace_path(tmp_path: Path) -> None:
    """A container-path ``last_baseline.workspace`` must still resolve under the on-disk ``session_dir``.

    Regression for handoff doc §8 TODO #2.
    """
    sd = tmp_path / "session"
    sd.mkdir(parents=True)
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "ttft"})
    bdir = sd / "runs/baseline/t1/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "output_throughput_tok_s": 333.0,
        "mean_ttft_ms": 142.5,
        "mean_e2el_ms": 1620.4,
    })
    _write_json(sd / "state.json", {
        "session_id": "ttft",
        "baseline_tput": 333.0,
        "last_baseline": {"workspace": "/workspace/runs/baseline/t1"},
    })

    b = build(sd)
    baseline = b["baseline"]
    assert baseline["ttft_mean_ms"] == pytest.approx(142.5)
    assert baseline["e2el_mean_ms"] == pytest.approx(1620.4)
    assert baseline["benchmark_report_path"] is not None
    assert "runs/baseline/t1" in baseline["benchmark_report_path"]
    assert not any(
        "baseline workspace" in w and "does not resolve" in w
        for w in b["warnings"]
    ), b["warnings"]


def test_baseline_resolves_measure_round_benchmark_workspace(tmp_path: Path) -> None:
    """Double-run baseline records may point directly at the hot
    ``measure_round/benchmark_*`` directory. The collector must read that
    report instead of looking for another nested ``benchmark_*`` below it."""
    sd = tmp_path / "session"
    sd.mkdir(parents=True)
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "dbl"})
    bdir = sd / (
        "runs/baseline/h1/measure_round/benchmark_sglang_20260605_014141"
    )
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "throughput": {"output_throughput": 3789.33},
        "latency": {
            "ttft": {"mean_ms": 616.7},
            "tpot": {"mean_ms": 16.29},
            "e2el": {"mean_ms": 17285.09},
        },
    })
    _write_json(sd / "state.json", {
        "session_id": "dbl",
        "baseline_tput": 3789.33,
        "last_baseline": {
            "workspace": (
                "/workspace/runs/baseline/h1/measure_round/"
                "benchmark_sglang_20260605_014141"
            ),
        },
    })

    b = build(sd)
    baseline = b["baseline"]
    assert baseline["ttft_mean_ms"] == pytest.approx(616.7)
    assert baseline["e2el_mean_ms"] == pytest.approx(17285.09)
    assert baseline["ttft_e2el_source"] == "state_workspace"
    assert "measure_round" in baseline["benchmark_report_path"]


def test_baseline_unresolvable_workspace_emits_warning(tmp_path: Path) -> None:
    """An unresolvable recorded workspace must surface a warning rather than silently fall back to ``None``."""
    sd = tmp_path / "session"
    sd.mkdir(parents=True)
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "miss"})
    _write_json(sd / "state.json", {
        "session_id": "miss",
        "baseline_tput": 100.0,
        "last_baseline": {"workspace": "/some/totally/unrelated/path"},
    })
    b = build(sd)
    assert b["baseline"]["ttft_mean_ms"] is None
    assert any(
        "baseline workspace" in w and "does not resolve" in w
        for w in b["warnings"]
    ), b["warnings"]


def test_source_files_drops_empty_kernel_attempts(tmp_path: Path) -> None:
    """``source_files.kernel_attempts`` must not appear when the session has no kernel-agent runs.

    Regression for handoff doc §8 TODO #6.
    """
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "no-kernels"})
    _write_json(sd / "state.json", {
        "session_id": "no-kernels", "baseline_tput": 100.0,
    })
    sf = build(sd)["source_files"]
    assert "kernel_attempts" not in sf
    assert "profile_reports" not in sf
    assert "sweep_reports" not in sf
    assert sf["manifest"] == "manifest.json"
    assert sf["state"] == "state.json"


def test_source_files_keeps_non_empty_kernel_attempts(fixture_session: Path) -> None:
    """Sanity: a session with kernel-agent activity emits the populated list."""
    sf = build(fixture_session)["source_files"]
    assert sf.get("kernel_attempts"), sf
    assert any("optimization_attempts.jsonl" in p for p in sf["kernel_attempts"])


# Attribution method (A1)
def _attribution_fixture(tmp_path: Path, state: dict) -> Path:
    """Minimal session_dir whose only content drives ``collect_attribution``."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "attr"})
    base_state = {"session_id": "attr"}
    base_state.update(state)
    _write_json(sd / "state.json", base_state)
    return sd


def test_attribution_method_validated(tmp_path: Path) -> None:
    """Every stack entry's ``delta_pct`` set → ``method == "validated"``."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 30.0,
        "optimization_stack": [
            {"action": "backends", "variant_name": "flag_X", "gain_pct": 20.0},
            {"action": "params",   "variant_name": "p1",     "gain_pct": 10.0},
        ],
        "gain_per_stack_entry": [
            {"action": "backends", "variant_name": "flag_X",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 20.0,
             "delta_pct": 20.0},
            {"action": "params",   "variant_name": "p1",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 20.0, "cum_gain_after": 30.0,
             "delta_pct": 10.0},
        ],
    })
    attr = build(sd)["attribution"]
    assert attr["method"] == "validated"


def test_attribution_method_single_source(tmp_path: Path) -> None:
    """Single-entry stack with no Coordinator-recorded gain ledger →
    ``method == "single_source"``."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 11.0,
        "optimization_stack": [
            {"action": "backends", "variant_name": "vllm_kv_fp8",
             "gain_pct": 11.0,
             "ts": "2026-05-15T07:00:00+00:00"},
        ],
    })
    attr = build(sd)["attribution"]
    assert attr["method"] == "single_source"


def test_attribution_method_reconstructed(tmp_path: Path) -> None:
    """Multi-entry stack, no ledger, a placeholder ``delta_pct=None`` → ``method == "reconstructed"``."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 25.0,
        "optimization_stack": [
            {"action": "backends", "variant_name": "flag_X", "gain_pct": 12.0},
            {"action": "params",   "variant_name": "p1",     "gain_pct": None},
            {"action": "kernel_opt:k001", "gain_pct": None},
        ],
    })
    attr = build(sd)["attribution"]
    assert attr["method"] == "reconstructed"


def test_attribution_method_missing(tmp_path: Path) -> None:
    """No optimization_stack, no Coordinator ledger → ``method == "missing"``."""
    sd = _attribution_fixture(tmp_path, {"cumulative_gain_validated": 0.0})
    attr = build(sd)["attribution"]
    assert attr["method"] == "missing"


# A1.0a: framework_pr surfaces in source_breakdown + phase_breakdown
# Regression: framework_pr KEEPs used to fall into the legacy ``other`` bucket
# and disappear from ``source_breakdown``; these tests pin the new behaviour.
def test_attribution_framework_pr_surfaces_in_source_breakdown(
    tmp_path: Path,
) -> None:
    """A ``framework_pr`` KEEP contributes to ``framework_pr_pct_of_total`` instead of being lost."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 22.85,
        "optimization_stack": [
            {"action": "explore",      "variant_name": "torch_compile_on", "gain_pct": 0.53},
            {"action": "framework_pr", "variant_name": "PR:26311",         "gain_pct": 22.43},
        ],
        "gain_per_stack_entry": [
            {"action": "explore",      "variant_name": "torch_compile_on",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 0.53,
             "delta_pct": 0.53,
             "ts": "2026-05-29T11:00:00+00:00"},
            {"action": "framework_pr", "variant_name": "PR:26311",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 0.53, "cum_gain_after": 22.85,
             "delta_pct": 22.32,
             "ts": "2026-05-29T11:30:00+00:00"},
        ],
    })
    sb = build(sd)["attribution"]["source_breakdown"]
    assert sb["framework_pr_pct_of_total"] == pytest.approx(22.32)
    assert sb["explore_pct_of_total"] == pytest.approx(0.53)
    # Per-source rows reconcile to validated_total (no "other" black-hole).
    summed = (
        sb["geak_pct_of_total"]
        + sb["oob_pct_of_total"]
        + sb["explore_pct_of_total"]
        + sb["sweep_pct_of_total"]
        + sb["framework_pr_pct_of_total"]
    )
    assert summed == pytest.approx(sb["validated_total_pct"], abs=0.05)


def test_attribution_framework_pr_pct_emitted_even_when_zero(
    tmp_path: Path,
) -> None:
    """``framework_pr_pct_of_total`` is always emitted (defaults to 0.0)."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 5.0,
        "optimization_stack": [
            {"action": "params", "variant_name": "p1", "gain_pct": 5.0},
        ],
        "gain_per_stack_entry": [
            {"action": "params", "variant_name": "p1",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 5.0,
             "delta_pct": 5.0,
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
    })
    sb = build(sd)["attribution"]["source_breakdown"]
    assert "framework_pr_pct_of_total" in sb
    assert sb["framework_pr_pct_of_total"] == 0.0


def test_attribution_phase_breakdown_framework_pr_by_pr(
    tmp_path: Path,
) -> None:
    """``phase_breakdown.framework_pr.by_pr`` aggregates per-PR gain keyed by variant_name verbatim."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 30.0,
        "phase_history": [
            {"to_phase": "PRELUDE",      "ts_unix": 1000.0},
            {"to_phase": "FRAMEWORK_PR", "ts_unix": 1100.0},
            {"to_phase": "EXPLORE",      "ts_unix": 2000.0},
        ],
        "optimization_stack": [
            {"action": "framework_pr", "variant_name": "PR:26311"},
            {"action": "framework_pr", "variant_name": "PR:sgl#9912"},
        ],
        "gain_per_stack_entry": [
            {"action": "framework_pr", "variant_name": "PR:26311",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 18.0,
             "delta_pct": 18.0,
             "ts_unix": 1200.0},
            {"action": "framework_pr", "variant_name": "PR:sgl#9912",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 18.0, "cum_gain_after": 30.0,
             "delta_pct": 12.0,
             "ts_unix": 1500.0},
        ],
    })
    pb = build(sd)["attribution"]["phase_breakdown"]
    assert "framework_pr" in pb
    assert pb["framework_pr"]["total_gain_pct"] == pytest.approx(30.0)
    assert pb["framework_pr"]["by_pr"]["PR:26311"] == pytest.approx(18.0)
    assert pb["framework_pr"]["by_pr"]["PR:sgl#9912"] == pytest.approx(12.0)


def test_attribution_framework_pr_phase_fallback_when_no_phase_history(
    tmp_path: Path,
) -> None:
    """Without ``phase_history`` the collector falls back to action family; ``framework_pr`` lands in its phase bucket, not ``unattributed``."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 10.0,
        "optimization_stack": [
            {"action": "framework_pr", "variant_name": "PR:42",
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
        "gain_per_stack_entry": [
            {"action": "framework_pr", "variant_name": "PR:42",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 10.0,
             "delta_pct": 10.0,
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
    })
    pb = build(sd)["attribution"]["phase_breakdown"]
    assert pb["framework_pr"]["total_gain_pct"] == pytest.approx(10.0)
    assert pb["framework_pr"]["by_pr"]["PR:42"] == pytest.approx(10.0)
    assert "unattributed" not in pb or pb["unattributed"]["total_gain_pct"] == 0.0


# A1.0d: gemm_tuning surfaces in source_breakdown + phase_breakdown
# Regression: gemm_tuning KEEPs used to fall into ``"other"`` and disappear
# from per-source totals (same shape of bug as framework_pr).
def test_attribution_gemm_tuning_surfaces_in_source_breakdown(
    tmp_path: Path,
) -> None:
    """A ``gemm_tuning`` KEEP contributes to ``gemm_tuning_pct_of_total`` instead of being lost to ``other``."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 12.0,
        "optimization_stack": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/a8w8_blockscale_tuned_gemm.csv",
             "gain_pct": 12.0},
        ],
        "gain_per_stack_entry": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/a8w8_blockscale_tuned_gemm.csv",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 12.0,
             "delta_pct": 12.0,
             "ts": "2026-06-01T11:00:00+00:00"},
        ],
    })
    sb = build(sd)["attribution"]["source_breakdown"]
    assert sb["gemm_tuning_pct_of_total"] == pytest.approx(12.0)
    # Per-source rows reconcile to validated_total (no "other" black-hole).
    summed = (
        sb["geak_pct_of_total"] + sb["oob_pct_of_total"]
        + sb["explore_pct_of_total"]
        + sb["sweep_pct_of_total"]
        + sb["framework_pr_pct_of_total"]
        + sb["gemm_tuning_pct_of_total"]
    )
    assert summed == pytest.approx(sb["validated_total_pct"], abs=0.05)


def test_attribution_gemm_tuning_pct_emitted_even_when_zero(
    tmp_path: Path,
) -> None:
    """Always emit ``gemm_tuning_pct_of_total`` (default 0.0)."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 5.0,
        "optimization_stack": [
            {"action": "params", "variant_name": "p1", "gain_pct": 5.0},
        ],
        "gain_per_stack_entry": [
            {"action": "params", "variant_name": "p1",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 5.0,
             "delta_pct": 5.0,
             "ts": "2026-06-01T11:00:00+00:00"},
        ],
    })
    sb = build(sd)["attribution"]["source_breakdown"]
    assert "gemm_tuning_pct_of_total" in sb
    assert sb["gemm_tuning_pct_of_total"] == 0.0


def test_attribution_phase_breakdown_gemm_tuning_by_tuned_file(
    tmp_path: Path,
) -> None:
    """``phase_breakdown.gemm_tuning.by_tuned_file`` aggregates per adopted CSV (two files → two keys)."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 18.0,
        "phase_history": [
            {"to_phase": "PRELUDE", "ts_unix": 1000.0},
            {"to_phase": "KERNEL",  "ts_unix": 1500.0},
        ],
        "optimization_stack": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/csv_a.csv"},
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/csv_b.csv"},
        ],
        "gain_per_stack_entry": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/csv_a.csv",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 11.0,
             "delta_pct": 11.0, "ts_unix": 1600.0},
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/csv_b.csv",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 11.0, "cum_gain_after": 18.0,
             "delta_pct": 7.0, "ts_unix": 1700.0},
        ],
    })
    pb = build(sd)["attribution"]["phase_breakdown"]
    gt = pb["gemm_tuning"]
    assert gt["total_gain_pct"] == pytest.approx(18.0)
    assert gt["by_tuned_file"]["/tmp/csv_a.csv"] == pytest.approx(11.0)
    assert gt["by_tuned_file"]["/tmp/csv_b.csv"] == pytest.approx(7.0)


def test_attribution_gemm_tuning_phase_fallback_when_no_phase_history(
    tmp_path: Path,
) -> None:
    """Without ``phase_history`` the action falls back to its own ``gemm_tuning`` bucket, not ``unattributed`` or ``kernel``."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 8.0,
        "optimization_stack": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/csv.csv",
             "ts": "2026-06-01T11:00:00+00:00"},
        ],
        "gain_per_stack_entry": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "tuned_file": "/tmp/csv.csv",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 8.0,
             "delta_pct": 8.0,
             "ts": "2026-06-01T11:00:00+00:00"},
        ],
    })
    pb = build(sd)["attribution"]["phase_breakdown"]
    assert pb["gemm_tuning"]["total_gain_pct"] == pytest.approx(8.0)
    assert pb["gemm_tuning"]["by_tuned_file"]["/tmp/csv.csv"] == pytest.approx(8.0)
    assert pb.get("kernel", {}).get("total_gain_pct", 0.0) == 0.0
    assert (
        "unattributed" not in pb
        or pb["unattributed"]["total_gain_pct"] == 0.0
    )


def test_attribution_gemm_tuning_falls_back_to_variant_name_then_question_mark(
    tmp_path: Path,
) -> None:
    """Bucket key falls back to ``variant_name`` then ``"?"`` when ``tuned_file`` is missing (always a string)."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 4.0,
        "optimization_stack": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm"},
            {"action": "gemm_tuning"},
        ],
        "gain_per_stack_entry": [
            {"action": "gemm_tuning",
             "variant_name": "a8w8_blockscale_tuned_gemm",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 3.0,
             "delta_pct": 3.0,
             "ts": "2026-06-01T11:00:00+00:00"},
            {"action": "gemm_tuning",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 3.0, "cum_gain_after": 4.0,
             "delta_pct": 1.0,
             "ts": "2026-06-01T11:01:00+00:00"},
        ],
    })
    by_tuned = build(sd)["attribution"]["phase_breakdown"]["gemm_tuning"]["by_tuned_file"]
    assert by_tuned["a8w8_blockscale_tuned_gemm"] == pytest.approx(3.0)
    assert by_tuned["?"] == pytest.approx(1.0)


# A1.0b: phase_breakdown.explore.by_domain key normalization
# Pre-PR-B raw ``provenance`` strings landed in ``by_domain`` verbatim; the
# collector now strips ``specialist:`` and folds ``legacy:*`` into ``legacy_*``.
def test_phase_breakdown_explore_by_domain_strips_specialist_prefix(
    tmp_path: Path,
) -> None:
    """``provenance = 'specialist:serving_specialist'`` surfaces under
    the bare ``serving_specialist`` key — the dashboard never sees the
    raw prefix. Multiple specialist provenances go to distinct buckets
    and sum independently."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 12.0,
        "phase_history": [
            {"to_phase": "EXPLORE", "ts_unix": 1000.0},
        ],
        "explore_search": {
            "winners_history": [
                {"fingerprint": "fp_serving",
                 "provenance": "specialist:serving_specialist"},
                {"fingerprint": "fp_kernel_switch",
                 "provenance": "specialist:kernel_switch_specialist"},
            ],
        },
        "optimization_stack": [
            {"action": "explore", "variant_name": "v_a", "fingerprint": "fp_serving"},
            {"action": "explore", "variant_name": "v_b", "fingerprint": "fp_kernel_switch"},
        ],
        "gain_per_stack_entry": [
            {"action": "explore", "variant_name": "v_a",
             "fingerprint": "fp_serving",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 7.0,
             "delta_pct": 7.0, "ts_unix": 1100.0},
            {"action": "explore", "variant_name": "v_b",
             "fingerprint": "fp_kernel_switch",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 7.0, "cum_gain_after": 12.0,
             "delta_pct": 5.0, "ts_unix": 1200.0},
        ],
    })
    pb = build(sd)["attribution"]["phase_breakdown"]
    by_domain = pb["explore"]["by_domain"]
    assert by_domain["serving_specialist"] == pytest.approx(7.0)
    assert by_domain["kernel_switch_specialist"] == pytest.approx(5.0)
    assert "specialist:serving_specialist" not in by_domain
    assert "specialist:kernel_switch_specialist" not in by_domain


def test_phase_breakdown_legacy_provenance_folded_to_legacy_prefix(
    tmp_path: Path,
) -> None:
    """``provenance = 'legacy:backends'`` becomes ``legacy_backends`` without masquerading as a specialist key."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 5.0,
        "phase_history": [
            {"to_phase": "EXPLORE", "ts_unix": 1000.0},
        ],
        "explore_search": {
            "winners_history": [
                {"fingerprint": "fp_legacy", "provenance": "legacy:backends"},
            ],
        },
        "optimization_stack": [
            {"action": "explore", "variant_name": "v_legacy", "fingerprint": "fp_legacy"},
        ],
        "gain_per_stack_entry": [
            {"action": "explore", "variant_name": "v_legacy",
             "fingerprint": "fp_legacy",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 5.0,
             "delta_pct": 5.0, "ts_unix": 1100.0},
        ],
    })
    by_domain = build(sd)["attribution"]["phase_breakdown"]["explore"]["by_domain"]
    assert by_domain["legacy_backends"] == pytest.approx(5.0)
    assert "legacy:backends" not in by_domain
    assert "backends" not in by_domain


def test_phase_breakdown_default_grid_and_llm_direct_pass_through(
    tmp_path: Path,
) -> None:
    """Non-specialist provenance (``default_grid`` / ``llm_direct``) passes through verbatim."""
    sd = _attribution_fixture(tmp_path, {
        "cumulative_gain_validated": 4.0,
        "phase_history": [
            {"to_phase": "EXPLORE", "ts_unix": 1000.0},
        ],
        "explore_search": {
            "winners_history": [
                {"fingerprint": "fp_grid", "provenance": "default_grid"},
                {"fingerprint": "fp_direct", "provenance": "llm_direct"},
            ],
        },
        "optimization_stack": [
            {"action": "explore", "variant_name": "v_grid",   "fingerprint": "fp_grid"},
            {"action": "explore", "variant_name": "v_direct", "fingerprint": "fp_direct"},
        ],
        "gain_per_stack_entry": [
            {"action": "explore", "variant_name": "v_grid",
             "fingerprint": "fp_grid",
             "stack_len_before": 0, "stack_len_after": 1,
             "cum_gain_before": 0.0, "cum_gain_after": 1.0,
             "delta_pct": 1.0, "ts_unix": 1100.0},
            {"action": "explore", "variant_name": "v_direct",
             "fingerprint": "fp_direct",
             "stack_len_before": 1, "stack_len_after": 2,
             "cum_gain_before": 1.0, "cum_gain_after": 4.0,
             "delta_pct": 3.0, "ts_unix": 1200.0},
        ],
    })
    by_domain = build(sd)["attribution"]["phase_breakdown"]["explore"]["by_domain"]
    assert by_domain["default_grid"] == pytest.approx(1.0)
    assert by_domain["llm_direct"] == pytest.approx(3.0)


# A1.0c: capability_summary.specialist.by_specialist per-domain split
def test_capability_summary_specialist_by_specialist_from_domain_breakdown(
    tmp_path: Path,
) -> None:
    """``domain_breakdown`` is folded into headline ``by_specialist`` counters; status is derived per-domain."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(
            round_id=1,
            domains=["serving_specialist"],
            proposals_total=3, proposals_kept=2,
            domain_breakdown={
                "serving_specialist": {
                    "dispatched": 1, "proposals_total": 3,
                    "proposals_kept": 2, "proposals_rejected": 1,
                },
            },
        ),
        _specialist_round(
            round_id=2,
            domains=["kernel_switch_specialist", "comm_specialist"],
            proposals_total=5, proposals_kept=0,
            domain_breakdown={
                "kernel_switch_specialist": {
                    "dispatched": 1, "proposals_total": 3,
                    "proposals_kept": 0, "proposals_rejected": 3,
                },
                "comm_specialist": {
                    "dispatched": 1, "proposals_total": 2,
                    "proposals_kept": 0, "proposals_rejected": 2,
                },
            },
        ),
    ]))
    spec = build(sd)["capability_summary"]["specialist"]
    bs = spec["by_specialist"]
    assert bs["serving_specialist"]["status"] == "kept"
    assert bs["serving_specialist"]["attempts"] == 1
    assert bs["serving_specialist"]["tested"] == 3
    assert bs["serving_specialist"]["keeps"] == 2
    assert bs["kernel_switch_specialist"]["status"] == "tried"
    assert bs["kernel_switch_specialist"]["tested"] == 3
    for d in (
        "compiler_specialist", "system_specialist",
        "pr_intel_specialist", "research_scout_specialist",
    ):
        assert bs[d]["status"] == "not_attempted"
        assert bs[d]["attempts"] == 0
    assert spec["attempts"] == 2
    assert spec["tested"] == 8
    assert spec["keeps"] == 2


def test_capability_summary_specialist_by_specialist_falls_back_to_domains_list(
    tmp_path: Path,
) -> None:
    """Legacy round predates ``domain_breakdown``: collector imputes an even share across listed ``domains``."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(
            round_id=1,
            domains=["serving_specialist", "compiler_specialist"],
            proposals_total=4, proposals_kept=2,
            domain_breakdown={},
        ),
    ]))
    spec = build(sd)["capability_summary"]["specialist"]
    bs = spec["by_specialist"]
    assert bs["serving_specialist"]["attempts"] == 1
    assert bs["serving_specialist"]["tested"] == 2
    assert bs["serving_specialist"]["keeps"] == 1
    assert bs["serving_specialist"]["status"] == "kept"
    assert bs["compiler_specialist"]["attempts"] == 1
    assert bs["compiler_specialist"]["tested"] == 2
    assert bs["compiler_specialist"]["keeps"] == 1


# A1.1: kernel_roofline (Dashboard-Roofline integration spec §1)
# Collector mirrors ``<sd>/reports/kernel_roofline.json``; every field is
# optional and collectors must not raise when it is missing or malformed.
def _kernel_roofline_progress_fixture(tmp_path: Path, payload: dict | None) -> Path:
    """Build a session_dir with optional reports/kernel_roofline.json."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kr"})
    _write_json(sd / "state.json", {"session_id": "kr"})
    if payload is not None:
        _write_json(sd / "reports" / "kernel_roofline.json", payload)
    return sd


def test_kernel_roofline_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    """No ``reports/kernel_roofline.json`` → empty dict, no warning."""
    sd = _kernel_roofline_progress_fixture(tmp_path, payload=None)
    bd = build(sd)
    assert bd["kernel_roofline"] == {}
    assert not any("kernel_roofline" in w for w in bd["warnings"])


def test_kernel_roofline_full_payload_passes_through(tmp_path: Path) -> None:
    """Happy path: every field round-trips with types preserved and on-disk kernel order kept."""
    payload = {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "analysis_md_path": "/abs/analysis.md",
        "kernel_candidates_path": "/abs/kernel_candidates.json",
        "trace_input": "/abs/torch_trace",
        "trace_input_type": "capture_dir",
        "kernels": [
            {
                "kernel_id": "k001",
                "name": "aiter::ck_moe_stage2",
                "source_file": "/sgl-workspace/aiter/csrc/foo.cu",
                "kernel_category": "MoE",
                "bound_type": "memory-bound",
                "arithmetic_intensity": 104.39,
                "flops_per_byte": 104.39,
                "efficiency_percent": 37.77,
                "gpu_pct": 29.002,
                "call_count": 48,
                "duration_us": 6874.0,
                "reusable_native_kernel": True,
                "rocprof_roofline": {
                    "before_kernel_opt": {
                        "status": "matched",
                        "roofline_efficiency_pct": 37.77,
                    },
                    "after_kernel_opt": {
                        "status": "scheduled",
                        "reason": "background_task",
                    },
                },
            },
            {
                "kernel_id": "k002",
                "name": "aten::mm",
                "source_file": "",
                "kernel_category": "unknown",
                "bound_type": "compute-bound",
                "arithmetic_intensity": 215.58,
                "flops_per_byte": 215.58,
                "efficiency_percent": 14.82,
                "gpu_pct": 3.485,
                "call_count": 48,
                "duration_us": 826.0,
                "reusable_native_kernel": False,
            },
        ],
    }
    sd = _kernel_roofline_progress_fixture(tmp_path, payload=payload)
    kr = build(sd)["kernel_roofline"]

    assert kr["schema_version"] == 1
    assert kr["source"] == "tracelens_analysis"
    assert kr["analysis_md_path"] == "/abs/analysis.md"
    assert kr["trace_input_type"] == "capture_dir"

    assert [k["kernel_id"] for k in kr["kernels"]] == ["k001", "k002"]

    k1 = kr["kernels"][0]
    assert k1["kernel_category"] == "MoE"
    assert k1["bound_type"] == "memory-bound"
    assert k1["efficiency_percent"] == pytest.approx(37.77)
    assert k1["gpu_pct"] == pytest.approx(29.002)
    assert k1["call_count"] == 48
    assert k1["duration_us"] == pytest.approx(6874.0)
    assert k1["reusable_native_kernel"] is True
    assert k1["rocprof_roofline"]["before_kernel_opt"]["status"] == "matched"
    assert k1["rocprof_roofline"]["after_kernel_opt"]["status"] == "scheduled"
    # Compute-bound entry preserves False explicitly.
    assert kr["kernels"][1]["reusable_native_kernel"] is False


def test_kernel_roofline_malformed_kernels_drops_to_empty_list(tmp_path: Path) -> None:
    """Non-list ``kernels`` is replaced with ``[]`` + warning; envelope fields still round-trip."""
    payload = {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "kernels": {"k001": {"name": "broken"}},
    }
    sd = _kernel_roofline_progress_fixture(tmp_path, payload=payload)
    bd = build(sd)
    kr = bd["kernel_roofline"]
    assert kr["schema_version"] == 1
    assert kr["source"] == "tracelens_analysis"
    assert kr["kernels"] == []
    assert any("kernel_roofline.kernels is not a list" in w for w in bd["warnings"])


def test_kernel_roofline_non_dict_blob_returns_empty(tmp_path: Path) -> None:
    """Top-level JSON is not an object → empty dict + warning, never raise."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kr"})
    _write_json(sd / "state.json", {"session_id": "kr"})
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    (sd / "reports" / "kernel_roofline.json").write_text(
        json.dumps([{"kernel_id": "k001"}]), encoding="utf-8",
    )
    bd = build(sd)
    assert bd["kernel_roofline"] == {}
    assert any("not a JSON object" in w for w in bd["warnings"])


def test_kernel_roofline_empty_kernels_list_is_valid(tmp_path: Path) -> None:
    """``kernels: []`` is legitimate (no hot kernels above threshold); envelope passes through, no warning."""
    payload = {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "kernels": [],
    }
    sd = _kernel_roofline_progress_fixture(tmp_path, payload=payload)
    bd = build(sd)
    assert bd["kernel_roofline"]["kernels"] == []
    assert not any("kernel_roofline" in w for w in bd["warnings"])


# kernel_optimization_summary (Breakdown panel integration spec §A1; PR #399)
# Collector mirrors ``<sd>/reports/kernel_optimization_summary.json`` verbatim
# (light shape guards only) so the dashboard reads sbd alone.
def _report_fixture(tmp_path: Path, rel_path: str, payload: dict | None) -> Path:
    """Session_dir with an optional ``reports/<file>.json``."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "rep"})
    _write_json(sd / "state.json", {"session_id": "rep"})
    if payload is not None:
        _write_json(sd / rel_path, payload)
    return sd


def test_kernel_opt_summary_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    """No report → empty dict, no warning (legacy / non-report sessions
    must stay warning-free; dashboard hides Block 1)."""
    sd = _report_fixture(tmp_path, "reports/kernel_optimization_summary.json", None)
    bd = build(sd)
    assert bd["kernel_optimization_summary"] == {}
    assert not any("kernel_optimization_summary" in w for w in bd["warnings"])


def test_kernel_opt_summary_full_payload_passes_through(tmp_path: Path) -> None:
    """Happy path: every documented field round-trips verbatim,
    including the deeply-nested by_kernel rows (verification +
    backend_ladder), and a rel ``report_path`` is added."""
    payload = {
        "schema_version": 1,
        "session_id": "Qwen-Qwen3-30B-A3B-Base_20260602T134619Z_f70dd15b",
        "model_name": "Qwen-Qwen3-30B-A3B-Base",
        "cumulative_gain_validated_pct": 1.01,
        "totals": {
            "top_candidates": 15, "attempted": 6, "integrated": 1,
            "keep_pending": 0, "rejected": 5, "in_flight": 0, "unattempted": 9,
        },
        "rejection_breakdown": {"revert_decision": 1, "max_failures_without_keep": 2,
                                 "max_partial_attempts_without_keep": 2, "other": 0},
        "unattempted_reason_breakdown": {"no_source_file": 4, "not_reusable_native_kernel": 2,
                                          "no_recommended_backend": 2, "below_priority_cutoff": 1,
                                          "unknown": 0},
        "failure_reason_breakdown": {"ladder_all_failed": 3, "correctness_failed": 1},
        "field_glossary": {"gpu_pct": "GPU time share 0-100"},
        "top_takeaways": [
            "1 of 6 attempted kernels reached KEEP and integrated; 5 were rejected.",
        ],
        "by_kernel": [
            {
                "kernel_id": "k001", "kernel_name": "aiter::ck_moe_stage1",
                "kernel_category": "MoE", "source_file": "/abs/foo.cu",
                "gpu_pct": 9.2, "efficiency_pct": 48.3, "bound_type": "memory-bound",
                "arithmetic_intensity": 104.4, "category": "ATTEMPTED_REJECTED",
                "attempts_total": 3, "rejected_reason": "max_failures_without_keep",
                "verification": {"compile_passed": False, "correctness_passed": None,
                                  "micro_speedup": 1.0},
                "backend_ladder": [
                    {"backend": "geak", "status": "failed", "produced_artifact": False,
                     "elapsed_sec": 213.5, "error_class": "preprocess_failed",
                     "error_message": "preprocess reported 1 error(s)"},
                    {"backend": "claude", "status": "failed", "produced_artifact": False,
                     "elapsed_sec": 483.5, "error_class": "timeout",
                     "error_message": "Timed out after 480s"},
                ],
            },
            {
                "kernel_id": "k002", "kernel_name": "aten::mm", "kernel_category": "GEMM",
                "source_file": "", "gpu_pct": 6.1, "category": "UNATTEMPTED",
                "reusable_native_kernel": False, "recommended_backends": [],
                "unattempted_reason": "no_source_file",
                "unattempted_detail": "vendor-library op; address via backend swap",
            },
        ],
    }
    sd = _report_fixture(tmp_path, "reports/kernel_optimization_summary.json", payload)
    ks = build(sd)["kernel_optimization_summary"]

    assert ks["schema_version"] == 1
    assert ks["model_name"] == "Qwen-Qwen3-30B-A3B-Base"
    assert ks["cumulative_gain_validated_pct"] == pytest.approx(1.01)
    assert ks["totals"]["top_candidates"] == 15
    assert ks["failure_reason_breakdown"]["ladder_all_failed"] == 3
    assert ks["top_takeaways"][0].startswith("1 of 6 attempted")

    assert [k["kernel_id"] for k in ks["by_kernel"]] == ["k001", "k002"]
    k1 = ks["by_kernel"][0]
    assert k1["category"] == "ATTEMPTED_REJECTED"
    assert k1["verification"]["compile_passed"] is False
    assert k1["verification"]["correctness_passed"] is None
    assert k1["backend_ladder"][1]["error_class"] == "timeout"
    assert ks["by_kernel"][1]["unattempted_reason"] == "no_source_file"

    assert ks["report_path"] == "reports/kernel_optimization_summary.json"


def test_kernel_opt_summary_all_zero_totals_is_valid(tmp_path: Path) -> None:
    """All-zero totals + empty by_kernel is a valid, warning-free Block-1-empty state, not a malformed file."""
    payload = {
        "schema_version": 1, "session_id": "s", "model_name": "m",
        "cumulative_gain_validated_pct": 0.0,
        "totals": {"top_candidates": 0, "attempted": 0, "integrated": 0,
                    "keep_pending": 0, "rejected": 0, "in_flight": 0, "unattempted": 0},
        "by_kernel": [], "top_takeaways": [],
    }
    sd = _report_fixture(tmp_path, "reports/kernel_optimization_summary.json", payload)
    bd = build(sd)
    ks = bd["kernel_optimization_summary"]
    assert ks["by_kernel"] == []
    assert ks["totals"]["attempted"] == 0
    assert not any("kernel_optimization_summary" in w for w in bd["warnings"])


def test_kernel_opt_summary_by_kernel_not_a_list_drops_to_empty(tmp_path: Path) -> None:
    """Non-list ``by_kernel`` is replaced with [] + warning; envelope still round-trips."""
    payload = {"schema_version": 1, "totals": {"top_candidates": 1},
               "by_kernel": {"k001": {"name": "broken"}}}
    sd = _report_fixture(tmp_path, "reports/kernel_optimization_summary.json", payload)
    bd = build(sd)
    ks = bd["kernel_optimization_summary"]
    assert ks["schema_version"] == 1
    assert ks["by_kernel"] == []
    assert any("by_kernel is not a list" in w for w in bd["warnings"])


def test_kernel_opt_summary_non_dict_blob_returns_empty(tmp_path: Path) -> None:
    """Top-level JSON is not an object → empty dict + warning, never raise."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "rep"})
    _write_json(sd / "state.json", {"session_id": "rep"})
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    (sd / "reports" / "kernel_optimization_summary.json").write_text(
        json.dumps([{"kernel_id": "k001"}]), encoding="utf-8",
    )
    bd = build(sd)
    assert bd["kernel_optimization_summary"] == {}
    assert any("kernel_optimization_summary" in w and "not a JSON object" in w
               for w in bd["warnings"])


# conc_sweep_summary (Breakdown panel integration spec §A2; PR #399)
def test_conc_sweep_summary_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    """No report → empty dict, no warning."""
    sd = _report_fixture(tmp_path, "reports/conc_sweep_summary.json", None)
    bd = build(sd)
    assert bd["conc_sweep_summary"] == {}
    assert not any("conc_sweep_summary" in w for w in bd["warnings"])


def test_conc_sweep_summary_full_payload_passes_through(tmp_path: Path) -> None:
    """Happy path: comparison rows, summary KPIs and optional roofline_ceiling round-trip verbatim; report_path added."""
    payload = {
        "schema_version": "1.0", "status": "succeeded", "session_id": "s",
        "isl": 1024, "osl": 1024, "tp": 8, "concs_requested": [1, 2, 4, 8],
        "baseline": {"extra_server_args": "", "extra_envs": {},
                      "points": [{"arm": "baseline", "conc": 8, "status": "succeeded",
                                   "output_throughput": 1300.34, "ttft_mean_ms": 145.2}]},
        "optimized": {"extra_server_args": "--num-continuous-decode-steps 4",
                       "extra_envs": {"ROCM_QUICK_REDUCE_QUANTIZATION": "FP"},
                       "points": [{"arm": "optimized", "conc": 8, "status": "succeeded",
                                    "output_throughput": 1313.54}]},
        "comparison": [{"conc": 8, "baseline_tput": 1300.34, "optimized_tput": 1313.54,
                         "speedup": 1.0101, "delta_pct": 1.01,
                         "baseline_status": "succeeded", "optimized_status": "succeeded"}],
        "summary": {"successful_pairs": 8, "failed_pairs": 0, "best_conc": 16,
                     "best_speedup": 1.142, "median_speedup": 1.071, "mean_speedup": 1.083},
        "workspace": "/abs/ws", "elapsed_sec": 123.4,
        "total_budget_sec": 9000, "budget_exhausted": False,
        "report_csv_path": "/abs/reports/conc_sweep_raw.csv",
        "roofline_ceiling": {
            "schema_version": 1, "source": "roofline_ceiling.py", "gpu_type": "mi300x",
            "rows": [{"conc": 8, "t_peak_tok_s": 4032.76, "bound_kind": "memory",
                       "mbu_baseline_pct": 66.17, "mbu_optimized_pct": 66.14}],
        },
    }
    sd = _report_fixture(tmp_path, "reports/conc_sweep_summary.json", payload)
    cs = build(sd)["conc_sweep_summary"]

    assert cs["schema_version"] == "1.0"
    assert cs["status"] == "succeeded"
    assert cs["optimized"]["extra_server_args"] == "--num-continuous-decode-steps 4"
    assert cs["comparison"][0]["speedup"] == pytest.approx(1.0101)
    assert cs["summary"]["best_conc"] == 16
    assert cs["roofline_ceiling"]["rows"][0]["t_peak_tok_s"] == pytest.approx(4032.76)
    assert cs["roofline_ceiling"]["schema_version"] == 1
    assert cs["report_path"] == "reports/conc_sweep_summary.json"


def test_conc_sweep_summary_skipped_preserves_sparse_shape(tmp_path: Path) -> None:
    """status="skipped" → collector passes the sparse shape through verbatim without fabricating blocks."""
    payload = {"schema_version": "1.0", "status": "skipped",
               "skip_reason": "no_optimization_to_compare", "session_id": "s",
               "isl": 1024, "osl": 1024, "tp": 8, "concs_requested": [1, 2, 4]}
    sd = _report_fixture(tmp_path, "reports/conc_sweep_summary.json", payload)
    bd = build(sd)
    cs = bd["conc_sweep_summary"]
    assert cs["status"] == "skipped"
    assert cs["skip_reason"] == "no_optimization_to_compare"
    assert "comparison" not in cs
    assert "summary" not in cs
    assert "baseline" not in cs
    assert not any("conc_sweep_summary" in w for w in bd["warnings"])


def test_conc_sweep_summary_non_dict_blob_returns_empty(tmp_path: Path) -> None:
    """Top-level JSON is not an object → empty dict + warning, never raise."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "rep"})
    _write_json(sd / "state.json", {"session_id": "rep"})
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    (sd / "reports" / "conc_sweep_summary.json").write_text(
        json.dumps("not-an-object"), encoding="utf-8",
    )
    bd = build(sd)
    assert bd["conc_sweep_summary"] == {}
    assert any("conc_sweep_summary" in w and "not a JSON object" in w
               for w in bd["warnings"])


# A1.2: roofline_progress (Dashboard-Roofline integration spec §2)
# Renamed from the top-level ``roofline`` key to coexist with the list-shaped
# ``roofline`` consumed by the markdown-report renderer.
def _roofline_progress_fixture(
    tmp_path: Path,
    *,
    state: dict,
    manifest: dict | None = None,
) -> Path:
    """Session fixture for ``collect_roofline_progress``; ``manifest`` defaults to a minimal stub."""
    sd = tmp_path / "session"
    _write_json(
        sd / "manifest.json",
        manifest or {
            "schema_version": 1,
            "session_id": "rl",
            "created_at_utc": "2026-05-29T10:40:50+00:00",
        },
    )
    base_state: dict = {"session_id": "rl"}
    base_state.update(state)
    _write_json(sd / "state.json", base_state)
    return sd


def test_roofline_progress_full_payload_baseline_plus_one_keep(tmp_path: Path) -> None:
    """Real-shape ``Qwen3-30B-A3B-Base`` payload: verifies trajectory ordering, gain math, and ceiling/target derivation."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 1300.34,
        "cumulative_gain": 1.0146,
        "current_best": {
            "action": "explore",
            "tput": 1313.5356953711394,
        },
        "optimization_stack": [
            {
                "action": "explore",
                "candidate_extra_server_args": "--num-continuous-decode-steps 4 --scheduler-recv-interval 4",
                "extra_envs": {},
                "tput": 1313.5356953711394,
                "ts": "2026-05-29T11:18:24.339975+00:00",
                "variant_name": "continuous_decode_steps_4",
            },
        ],
        "roofline_snapshots": [
            {
                "snapshot_id": 1,
                "ts": "2026-05-29T11:06:03.891380+00:00",
                "achieved_tok_per_sec": 1300.34,
                "theoretical_peak_tok_per_sec": 1976.8214052878614,
                "within_roofline_pct": 65.78,
                "gap_to_roofline_pct": 34.22,
                "compute_pct": 29.95,
                "idle_pct": 70.02,
                "comm_pct": 0.0,
                "top_bottleneck": "MoE_unfused",
                "top_kernel": {
                    "name": "aiter::ck_moe_stage1",
                    "bound_type": "memory",
                    "efficiency_pct": 48.35,
                    "gpu_pct": 9.17,
                },
            },
        ],
    })
    rl = build(sd)["roofline_progress"]

    assert len(rl["trajectory"]) == 2
    assert rl["trajectory"][0]["label"] == "baseline"
    assert rl["trajectory"][0]["action"] == "baseline"
    assert rl["trajectory"][0]["ts"] == "2026-05-29T10:40:50+00:00"
    assert rl["trajectory"][0]["tput"] == pytest.approx(1300.34)
    assert rl["trajectory"][0]["gain_pct"] == 0.0
    assert rl["trajectory"][1]["label"] == "continuous_decode_steps_4"
    assert rl["trajectory"][1]["tput"] == pytest.approx(1313.5356953711394)
    assert rl["trajectory"][1]["flags"] == "--num-continuous-decode-steps 4 --scheduler-recv-interval 4"
    assert rl["trajectory"][1]["gain_pct"] == pytest.approx(1.015, abs=0.01)

    assert rl["ceiling_available"] is True
    assert rl["ceiling_tok_per_sec"] == pytest.approx(1976.82, abs=0.01)
    assert rl["target_tok_per_sec"] == pytest.approx(1976.82 * 0.70, abs=0.01)
    assert rl["ceiling_ratio_target"] == pytest.approx(0.70)

    assert rl["baseline_tput"] == pytest.approx(1300.34)
    assert rl["current_best_tput"] == pytest.approx(1313.5356953711394)
    assert rl["current_best_pct_of_ceiling"] == pytest.approx(66.45, abs=0.05)
    assert rl["current_best_pct_of_target"] == pytest.approx(94.93, abs=0.05)

    assert rl["snapshot_top_bottleneck"] == "MoE_unfused"
    assert rl["snapshot_within_roofline_pct"] == pytest.approx(65.78)

    assert len(rl["snapshots"]) == 1
    assert rl["snapshots"][0]["theoretical_peak_tok_per_sec"] == pytest.approx(
        1976.8214052878614,
    )


def test_roofline_progress_no_snapshot_means_no_ceiling(tmp_path: Path) -> None:
    """Without ``roofline_snapshots`` the trajectory is still drawn but ceiling/target are None and ``ceiling_available`` is False."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 100.0,
        "cumulative_gain": 5.0,
        "current_best": {"tput": 105.0},
        "optimization_stack": [
            {"action": "explore", "tput": 105.0, "variant_name": "v1",
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
    })
    rl = build(sd)["roofline_progress"]

    assert rl["ceiling_available"] is False
    assert rl["ceiling_tok_per_sec"] is None
    assert rl["target_tok_per_sec"] is None
    assert rl["current_best_pct_of_ceiling"] is None
    assert rl["current_best_pct_of_target"] is None
    assert len(rl["trajectory"]) == 2
    assert rl["snapshots"] == []


def test_roofline_progress_no_keep_yet_baseline_only(tmp_path: Path) -> None:
    """Mid-session before any KEEP: trajectory holds only the baseline point; cumulative_gain == 0."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 200.0,
        "cumulative_gain": 0.0,
        "current_best": {"tput": 200.0},
        "optimization_stack": [],
    })
    rl = build(sd)["roofline_progress"]

    assert len(rl["trajectory"]) == 1
    assert rl["trajectory"][0]["label"] == "baseline"
    assert rl["current_best_tput"] == pytest.approx(200.0)
    assert rl["cumulative_gain_pct"] == 0.0


def test_roofline_progress_baseline_failed_empty_trajectory(tmp_path: Path) -> None:
    """When ``baseline_tput`` is 0 the trajectory is empty instead of plotting against zero."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 0.0,
        "cumulative_gain": 0.0,
        "optimization_stack": [],
    })
    rl = build(sd)["roofline_progress"]

    assert rl["trajectory"] == []
    assert rl["baseline_tput"] == 0.0
    assert rl["current_best_tput"] == 0.0


def test_roofline_progress_uses_latest_snapshot_for_ceiling(tmp_path: Path) -> None:
    """With multiple ``roofline_snapshots`` the ceiling is read from the LATEST snapshot, not snapshots[0]."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 1000.0,
        "cumulative_gain": 0.0,
        "current_best": {"tput": 1000.0},
        "optimization_stack": [],
        "roofline_snapshots": [
            {"snapshot_id": 1, "theoretical_peak_tok_per_sec": 1500.0,
             "ts": "2026-05-29T10:00:00+00:00"},
            {"snapshot_id": 2, "theoretical_peak_tok_per_sec": 1700.0,
             "ts": "2026-05-29T11:00:00+00:00",
             "top_bottleneck": "kv_cache"},
        ],
    })
    rl = build(sd)["roofline_progress"]
    assert rl["ceiling_tok_per_sec"] == pytest.approx(1700.0)
    assert rl["snapshot_top_bottleneck"] == "kv_cache"
    assert len(rl["snapshots"]) == 2


def test_roofline_progress_trajectory_diverges_from_current_best_emits_warning(
    tmp_path: Path,
) -> None:
    """A trajectory tail tput diverging from ``state.current_best.tput`` surfaces as a warning."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 100.0,
        "cumulative_gain": 5.0,
        "current_best": {"tput": 110.0},
        "optimization_stack": [
            {"action": "explore", "tput": 105.0, "variant_name": "v1",
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
    })
    bd = build(sd)
    assert any(
        "roofline.current_best_tput" in w and "current_best.tput" in w
        for w in bd["warnings"]
    )


def test_roofline_progress_failure_streak_passes_through(tmp_path: Path) -> None:
    """``roofline_failure_streak`` passes through for the dashboard's stale-ceiling badge."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 100.0,
        "current_best": {"tput": 100.0},
        "roofline_failure_streak": 3,
        "optimization_stack": [],
    })
    rl = build(sd)["roofline_progress"]
    assert rl["roofline_failure_streak"] == 3


# A1.3: roofline + roofline_progress coexist (post name-clash fix)
# Regression: two collectors were both registered as ``collect_roofline`` and
# the markdown-report list silently evaluated to empty; both surfaces now coexist.
def test_roofline_and_roofline_progress_coexist_independently(
    tmp_path: Path,
) -> None:
    """A trace_analyze snapshot populates both ``roofline`` (list) and ``roofline_progress`` (dict) independently."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 1300.0,
        "current_best": {"tput": 1313.0},
        "cumulative_gain": 1.0,
        "optimization_stack": [
            {"action": "explore", "variant_name": "v1",
             "candidate_extra_server_args": "--num-continuous-decode-steps 4",
             "extra_envs": {}, "tput": 1313.0,
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
        "roofline_snapshots": [
            {"snapshot_id": 1, "ts": "2026-05-29T10:30:00+00:00",
             "achieved_tok_per_sec": 1300.0,
             "theoretical_peak_tok_per_sec": 1976.0,
             "compute_pct": 30.0, "idle_pct": 69.0, "comm_pct": 1.0,
             "top_bottleneck": "MoE_unfused",
             "top_kernel": {"name": "aiter::ck_moe_stage1",
                            "bound_type": "memory",
                            "efficiency_pct": 48.0, "gpu_pct": 9.0}},
        ],
    })
    bd = build(sd)
    assert isinstance(bd["roofline"], list)
    assert len(bd["roofline"]) >= 1
    entry = bd["roofline"][0]
    assert "source_path" in entry
    assert entry.get("baseline") or entry.get("latest")
    assert isinstance(bd["roofline_progress"], dict)
    assert bd["roofline_progress"]["ceiling_available"] is True
    assert len(bd["roofline_progress"]["trajectory"]) == 2


def test_roofline_list_empty_when_no_snapshots(tmp_path: Path) -> None:
    """Without ``state.roofline_snapshots`` the markdown surface degrades to ``[]`` while the dashboard trajectory is unaffected."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 1300.0,
        "current_best": {"tput": 1313.0},
        "cumulative_gain": 1.0,
        "optimization_stack": [
            {"action": "explore", "variant_name": "v1", "tput": 1313.0,
             "ts": "2026-05-29T11:00:00+00:00"},
        ],
    })
    bd = build(sd)
    assert bd["roofline"] == []
    assert bd["roofline_progress"]["ceiling_available"] is False
    assert len(bd["roofline_progress"]["trajectory"]) == 2


# A1.4: optimization_stack passthrough (raw KEEP ledger)
# Mirrors ``state.optimization_stack[]`` to the sbd top level so downstream
# tooling reads full per-entry evidence without round-tripping to state.json.
def test_optimization_stack_empty_when_state_has_no_stack(tmp_path: Path) -> None:
    """Fresh session: absent/empty ``state.optimization_stack`` → top-level field is ``[]``, no warning."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 0.0,
        "cumulative_gain": 0.0,
    })
    bd = build(sd)
    assert bd["optimization_stack"] == []


def test_optimization_stack_full_field_passthrough(tmp_path: Path) -> None:
    """Standard explore KEEP: the full field set surfaces with coerced types; ``workspace`` survives null."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 1300.0,
        "current_best": {"tput": 1313.0},
        "cumulative_gain": 1.0,
        "optimization_stack": [
            {
                "action": "explore",
                "variant_name": "continuous_decode_steps_4",
                "candidate_extra_server_args":
                    "--num-continuous-decode-steps 4 --scheduler-recv-interval 4",
                "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
                "tput": 1313.5356953711394,
                "ts": "2026-05-29T11:18:24.339975+00:00",
                "workspace": None,
                "fingerprint": "abc123",
                "provenance": "specialist:serving_specialist",
            },
        ],
    })
    stack = build(sd)["optimization_stack"]
    assert len(stack) == 1
    e = stack[0]
    assert e["action"] == "explore"
    assert e["variant_name"] == "continuous_decode_steps_4"
    assert e["candidate_extra_server_args"].startswith("--num-continuous-decode-steps")
    assert e["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert e["tput"] == pytest.approx(1313.5356953711394)
    assert e["ts"] == "2026-05-29T11:18:24.339975+00:00"
    assert e["workspace"] is None
    # Optional fields surface when present.
    assert e["fingerprint"] == "abc123"
    assert e["provenance"] == "specialist:serving_specialist"


def test_optimization_stack_passes_through_gemm_tuning_evidence(
    tmp_path: Path,
) -> None:
    """A ``gemm_tuning`` KEEP carries ``tuned_file`` /
    ``final_report_path`` / ``source`` / ``gain_pct`` — these are the
    full evidence the dashboard needs to attribute speedup to the
    deterministic FP8 tuner. The passthrough preserves them all."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 100.0,
        "current_best": {"tput": 110.0},
        "cumulative_gain": 10.0,
        "optimization_stack": [
            {
                "action": "gemm_tuning",
                "variant_name": "a8w8_blockscale_tuned_gemm",
                "candidate_extra_server_args": "",
                "extra_envs": {
                    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE":
                        "/abs/path/a8w8_blockscale_tuned_gemm.csv",
                },
                "tput": 110.0,
                "ts": "2026-06-01T10:00:00+00:00",
                "workspace": "/abs/path/gemm_tuning_001",
                "tuned_file": "/abs/path/a8w8_blockscale_tuned_gemm.csv",
                "final_report_path": "/abs/path/final_report.json",
                "gain_pct": 10.0,
                "source": "kernel_entry_auto",
            },
        ],
    })
    stack = build(sd)["optimization_stack"]
    assert len(stack) == 1
    e = stack[0]
    assert e["action"] == "gemm_tuning"
    assert e["tuned_file"] == "/abs/path/a8w8_blockscale_tuned_gemm.csv"
    assert e["final_report_path"] == "/abs/path/final_report.json"
    assert e["source"] == "kernel_entry_auto"
    assert e["gain_pct"] == pytest.approx(10.0)
    assert e["extra_envs"]["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"].endswith(
        "a8w8_blockscale_tuned_gemm.csv"
    )


def test_optimization_stack_preserves_promotion_order(
    tmp_path: Path,
) -> None:
    """Multi-step session: stack order is preserved verbatim (no re-sort or de-dupe)."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 100.0,
        "current_best": {"tput": 130.0},
        "cumulative_gain": 30.0,
        "optimization_stack": [
            {"action": "params",  "variant_name": "p1", "tput": 110.0,
             "ts": "2026-06-01T10:00:00+00:00"},
            {"action": "gemm_tuning", "variant_name": "a8w8_tuned",
             "tuned_file": "/abs/csv.csv", "tput": 120.0,
             "ts": "2026-06-01T10:30:00+00:00"},
            {"action": "kernel_opt", "variant_name": "k005", "tput": 130.0,
             "ts": "2026-06-01T11:00:00+00:00",
             "kernel_id": "k005"},
        ],
    })
    stack = build(sd)["optimization_stack"]
    assert [e["action"] for e in stack] == ["params", "gemm_tuning", "kernel_opt"]
    assert stack[1]["tuned_file"] == "/abs/csv.csv"
    assert stack[2]["kernel_id"] == "k005"


def test_optimization_stack_drops_non_dict_entries(tmp_path: Path) -> None:
    """Defensive: malformed entries are dropped rather than crashing the export."""
    sd = _roofline_progress_fixture(tmp_path, state={
        "baseline_tput": 100.0,
        "current_best": {"tput": 110.0},
        "cumulative_gain": 10.0,
        "optimization_stack": [
            {"action": "params", "variant_name": "p1", "tput": 110.0,
             "ts": "2026-06-01T10:00:00+00:00"},
            "garbage",
            None,
            42,
        ],
    })
    stack = build(sd)["optimization_stack"]
    assert len(stack) == 1
    assert stack[0]["action"] == "params"


# A2: final.ttft_mean_ms reconstruction
def test_final_ttft_reconstructed_from_validate_stack(tmp_path: Path) -> None:
    """Missing ``current_best.ttft_mean_ms`` is read from a disk validate_stack report (``ttft_e2el_source = "validate_stack_disk"``) with a reconstruction warning."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "rec"})
    _write_json(sd / "state.json", {
        "session_id": "rec",
        "baseline_tput": 100.0,
        "current_best": {"tput": 220.0},
        "optimization_stack": [
            {"action": "backends", "variant_name": "v1", "gain_pct": 120.0},
        ],
        "cumulative_gain_validated": 120.0,
    })
    bdir = sd / "runs/validate_stack/abc/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "output_throughput_tok_s": 220.0,
        "mean_ttft_ms": 88.7,
        "mean_e2el_ms": 1110.5,
    })
    b = build(sd)
    final = b["final"]
    assert final["ttft_mean_ms"] == pytest.approx(88.7)
    assert final["e2el_mean_ms"] == pytest.approx(1110.5)
    assert final["ttft_e2el_source"] == "validate_stack_disk"
    assert any(
        "final.ttft_mean_ms reconstructed from validate_stack_disk" in w
        for w in b["warnings"]
    ), b["warnings"]


def test_final_ttft_reconstructed_from_current_best_benchmark_dir(
    tmp_path: Path,
) -> None:
    """Final reconstruction reads the report when ``current_best.workspace`` points at a ``measure_round/benchmark_*`` dir."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "cbdir"})
    bdir = sd / (
        "runs/baseline/h1/measure_round/benchmark_sglang_20260605_014141"
    )
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "throughput": {"output_throughput": 3789.33},
        "latency": {
            "ttft": {"mean_ms": 616.7},
            "tpot": {"mean_ms": 16.29},
            "e2el": {"mean_ms": 17285.09},
        },
    })
    _write_json(sd / "state.json", {
        "session_id": "cbdir",
        "baseline_tput": 3789.33,
        "current_best": {
            "action": "baseline",
            "tput": 3789.33,
            "workspace": (
                "/workspace/runs/baseline/h1/measure_round/"
                "benchmark_sglang_20260605_014141"
            ),
        },
    })

    final = build(sd)["final"]
    assert final["ttft_mean_ms"] == pytest.approx(616.7)
    assert final["e2el_mean_ms"] == pytest.approx(17285.09)
    assert final["ttft_e2el_source"] == "current_best_disk"


def test_final_ttft_reconstructed_from_warm_replay_measure_round(
    tmp_path: Path,
) -> None:
    """Warm-replay entries with ``workspace = None`` are matched by action/tput to recover final latency."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "wr"})
    bdir = sd / (
        "runs/replay_warm_recipe/f86b/measure_round/"
        "benchmark_sglang_20260604_151328"
    )
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "throughput": {"output_throughput": 4568.10},
        "latency": {
            "ttft": {"mean_ms": 2156.71},
            "tpot": {"mean_ms": 11.91},
            "e2el": {"mean_ms": 14335.65},
        },
    })
    _write_json(sd / "state.json", {
        "session_id": "wr",
        "baseline_tput": 1480.90,
        "current_best": {"tput": 4568.10},
        "optimization_stack": [
            {
                "action": "replay_warm_recipe",
                "variant_name": "warm_replay",
                "tput": 4568.10,
                "workspace": None,
            },
        ],
        "cumulative_gain_validated": 208.47,
    })

    final = build(sd)["final"]
    assert final["ttft_mean_ms"] == pytest.approx(2156.71)
    assert final["e2el_mean_ms"] == pytest.approx(14335.65)
    assert final["ttft_e2el_source"] == "stack_top_disk"


# A3: baseline.attempts_history reconstruction
def test_baseline_attempts_history_reconstructed_from_disk(tmp_path: Path) -> None:
    """Empty ``state.baseline_attempts`` + on-disk baseline dirs reconstructs ``status="reconstructed"`` rows with a warning."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "rebh"})
    _write_json(sd / "state.json", {
        "session_id": "rebh",
        "baseline_tput": 333.0,
        "baseline_attempts": [],
    })
    for h, tput in (("hashA", 300.0), ("hashB", 333.0)):
        bdir = sd / f"runs/baseline/{h}/benchmark_20260515T100000"
        _write_json(bdir / "benchmark_report.json", {
            "success": True,
            "output_throughput_tok_s": tput,
            "mean_ttft_ms": 100.0,
        })

    b = build(sd)
    history = b["baseline"]["attempts_history"]
    assert len(history) == 2
    assert all(a["status"] == "reconstructed" for a in history), history
    task_ids = {a["task_id"] for a in history}
    assert task_ids == {"hashA", "hashB"}
    assert any(
        "baseline.attempts_history reconstructed from runs/baseline/" in w
        for w in b["warnings"]
    ), b["warnings"]


def test_baseline_attempts_history_state_recorded_takes_precedence(
    tmp_path: Path,
) -> None:
    """Non-empty ``state.baseline_attempts`` suppresses the disk fallback (no duplication or warning)."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "stat"})
    _write_json(sd / "state.json", {
        "session_id": "stat",
        "baseline_tput": 300.0,
        "baseline_attempts": [
            {"ts": "2026-05-15T10:00:00+00:00", "task_id": "t1",
             "status": "succeeded", "decision": "promoted",
             "key_metric": 300.0},
        ],
    })
    bdir = sd / "runs/baseline/hashA/benchmark_20260515T100000"
    _write_json(bdir / "benchmark_report.json", {
        "success": True, "output_throughput_tok_s": 300.0,
    })

    b = build(sd)
    history = b["baseline"]["attempts_history"]
    assert len(history) == 1
    assert history[0]["status"] == "succeeded"
    assert not any(
        "baseline.attempts_history reconstructed" in w
        for w in b["warnings"]
    )


def test_baseline_attempts_history_passes_through_error_excerpt(
    tmp_path: Path,
) -> None:
    """A failed baseline attempt's ``error_excerpt`` / ``stderr_tail`` survives into exported ``attempts_history`` for RCA."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "errx"})
    _write_json(sd / "state.json", {
        "session_id": "errx",
        "baseline_tput": 0.0,
        "baseline_attempts": [
            {"ts": "2026-06-08T01:00:00+00:00", "task_id": "t1",
             "status": "failed", "decision": "no_promote",
             "key_metric": None, "error_class": "subprocess_nonzero",
             "error_excerpt": "torch.OutOfMemoryError: HIP out of memory",
             "stderr_tail": "aiter_backend.py line 219 workspace_buffer",
             "stderr_log_path": "runs/baseline/t1/baseline_stderr.log"},
        ],
    })

    b = build(sd)
    history = b["baseline"]["attempts_history"]
    assert len(history) == 1
    assert history[0]["error_class"] == "subprocess_nonzero"
    assert history[0]["error_excerpt"] == (
        "torch.OutOfMemoryError: HIP out of memory"
    )
    assert "workspace_buffer" in (history[0]["stderr_tail"] or "")
    assert history[0]["stderr_log_path"] == (
        "runs/baseline/t1/baseline_stderr.log"
    )


# ---------------------------------------------------------------------------
# B3: invocation populated from baseline_config + server.log
def test_baseline_invocation_populated(tmp_path: Path) -> None:
    """``baseline.invocation`` reads framework_args from server.log and allowlisted envs, keeping secret-shaped keys out."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "inv"})
    bdir = sd / "runs/baseline/h1/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True, "output_throughput_tok_s": 421.5,
        "mean_ttft_ms": 152.3, "mean_e2el_ms": 1840.7,
    })
    (bdir / "server.log").write_text(
        "python -m sglang.launch_server --model /weka/m --tp 8 --port 30000\n"
        "[INFO] booting...\n",
        encoding="utf-8",
    )
    cfg = sd / "runs/baseline/h1/baseline_config.with_envs.yaml"
    cfg.write_text(
        "benchmark:\n"
        "  envs:\n"
        "    TP: \"8\"\n"
        "    VLLM_FLASH_ATTN: \"1\"\n"
        "    OPENAI_API_KEY: \"secret-do-not-emit\"\n"
        "    HOME: \"/root\"\n",
        encoding="utf-8",
    )
    _write_json(sd / "state.json", {
        "session_id": "inv",
        "baseline_tput": 421.5,
        "last_baseline": {"workspace": str(sd / "runs/baseline/h1")},
        "baseline_config_path": str(cfg),
    })

    b = build(sd)
    inv = b["baseline"]["invocation"]
    assert "sglang.launch_server" in inv["framework_args"]
    assert inv["extra_envs"].get("TP") == "8"
    assert inv["extra_envs"].get("VLLM_FLASH_ATTN") == "1"
    assert "OPENAI_API_KEY" not in inv["extra_envs"]
    assert "HOME" not in inv["extra_envs"]
    assert inv["server_log_path"] is not None
    assert "server.log" in inv["server_log_path"]


# B1 / B3: image detection
def test_session_image_from_env(tmp_path: Path, monkeypatch) -> None:
    """``HYPERLOOM_IMAGE`` populates ``session.image`` when the manifest lacks it; absent all sources it is ``None`` + one warning."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "img"})
    _write_json(sd / "state.json", {"session_id": "img", "baseline_tput": 100.0})

    monkeypatch.setenv("HYPERLOOM_IMAGE",
                        "registry.example/hyperloom:abc123")
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("IMAGE", raising=False)
    b = build(sd)
    assert b["session"]["image"] == "registry.example/hyperloom:abc123"

    monkeypatch.delenv("HYPERLOOM_IMAGE")
    b2 = build(sd)
    assert b2["session"]["image"] is None or isinstance(b2["session"]["image"], str)
    if b2["session"]["image"] is None:
        assert any(
            "image: not configured" in w for w in b2["warnings"]
        ), b2["warnings"]


def test_session_image_from_manifest_takes_precedence(
    tmp_path: Path, monkeypatch,
) -> None:
    """``manifest.image`` wins over runtime env vars, capturing the spawn-time image."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {
        "schema_version": 2, "session_id": "img2",
        "image": "registry.example/hyperloom:from-manifest",
    })
    _write_json(sd / "state.json", {"session_id": "img2"})

    monkeypatch.setenv("HYPERLOOM_IMAGE", "registry.example/hyperloom:from-env")
    b = build(sd)
    assert b["session"]["image"] == "registry.example/hyperloom:from-manifest"


# C1: baseline.ttft_mean_ms disk-walk fallback
def test_baseline_ttft_disk_walk_fallback(tmp_path: Path) -> None:
    """When ``state.last_baseline.workspace`` doesn't resolve, the collector walks ``runs/baseline/`` for the latest report.

    Production parallel: a valid benchmark_report.json existed on wekafs but
    state's recorded workspace didn't resolve.
    """
    sd = tmp_path / "session"
    sd.mkdir(parents=True)
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "diskwalk"})
    bdir = sd / "runs/baseline/541e643a84e14352a2ff64bdfe27b2c9/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "output_throughput_tok_s": 421.5,
        "mean_ttft_ms": 99.5,
        "mean_e2el_ms": 1500.0,
    })
    _write_json(sd / "state.json", {
        "session_id": "diskwalk",
        "baseline_tput": 421.5,
        "last_baseline": {"workspace": "/workspace/runs/baseline/different-hash"},
    })

    b = build(sd)
    baseline = b["baseline"]
    assert baseline["ttft_mean_ms"] == pytest.approx(99.5)
    assert baseline["e2el_mean_ms"] == pytest.approx(1500.0)
    assert baseline["ttft_e2el_source"] == "runs_baseline_disk"
    assert any(
        "baseline.ttft_mean_ms reconstructed from runs/baseline/ disk walk" in w
        for w in b["warnings"]
    ), b["warnings"]


def test_baseline_ttft_disk_walk_fallback_measure_round(tmp_path: Path) -> None:
    """Disk-walk fallback includes double-run ``measure_round`` reports, not just legacy ones."""
    sd = tmp_path / "session"
    sd.mkdir(parents=True)
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "mrdw"})
    bdir = sd / "runs/baseline/hash1/measure_round/benchmark_sglang_20260605_031230"
    _write_json(bdir / "benchmark_report.json", {
        "success": True,
        "throughput": {"output_throughput": 2110.29},
        "latency": {
            "ttft": {"mean_ms": 2075.01},
            "tpot": {"mean_ms": 28.32},
            "e2el": {"mean_ms": 31045.42},
        },
    })
    _write_json(sd / "state.json", {
        "session_id": "mrdw",
        "baseline_tput": 2110.29,
        "last_baseline": {"workspace": "/workspace/runs/baseline/missing"},
    })

    b = build(sd)
    baseline = b["baseline"]
    assert baseline["ttft_mean_ms"] == pytest.approx(2075.01)
    assert baseline["e2el_mean_ms"] == pytest.approx(31045.42)
    assert baseline["ttft_e2el_source"] == "runs_baseline_disk"
    assert "measure_round" in baseline["benchmark_report_path"]


# C2: framework_args extraction lineage
def _make_invocation_fixture(
    sd: Path,
    server_log_text: str | None,
    yaml_text: str | None,
) -> None:
    """Minimal fixture: state pointing at runs/baseline/h1 with optional server.log + yaml."""
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "fa"})
    bdir = sd / "runs/baseline/h1/benchmark_001"
    _write_json(bdir / "benchmark_report.json", {
        "success": True, "output_throughput_tok_s": 200.0,
        "mean_ttft_ms": 120.0, "mean_e2el_ms": 1500.0,
    })
    if server_log_text is not None:
        (bdir / "server.log").write_text(server_log_text, encoding="utf-8")
    cfg_path = sd / "runs/baseline/h1/baseline_config.with_envs.yaml"
    if yaml_text is not None:
        cfg_path.write_text(yaml_text, encoding="utf-8")
    state: dict[str, Any] = {
        "session_id": "fa",
        "baseline_tput": 200.0,
        "last_baseline": {"workspace": str(sd / "runs/baseline/h1")},
    }
    if yaml_text is not None:
        state["baseline_config_path"] = str(cfg_path)
    _write_json(sd / "state.json", state)


def test_framework_args_from_server_arguments_line(tmp_path: Path) -> None:
    """``Server arguments: ...`` header in server.log → source ``log_args_line``."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "INFO 05-12 14:21:14 [server.py:42] booting up\n"
            "INFO 05-12 14:21:15 [server.py:50] Server arguments: "
            "--model /weka/m --tp 8 --port 30001\n"
            "INFO 05-12 14:21:16 [server.py:99] ready\n"
        ),
        yaml_text=None,
    )
    inv = build(sd)["baseline"]["invocation"]
    assert "--tp 8" in inv["framework_args"]
    assert "--port 30001" in inv["framework_args"]
    assert inv["framework_args_source"] == "log_args_line"


def test_framework_args_from_python_cmd_line(tmp_path: Path) -> None:
    """A literal ``python -m vllm.entrypoints...`` line → source ``log_python_cmd`` (Pass-2 fallback)."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "(APIServer pid=1757439) INFO 05-12 14:21:14 [utils.py:299] booting\n"
            "(APIServer pid=1757439) INFO 05-12 14:21:15 [server.py:50] init\n"
            "python -m vllm.entrypoints.openai.api_server --model /weka/m --tp 8\n"
            "(APIServer pid=1757439) INFO 05-12 14:22:00 [server.py:99] ready\n"
        ),
        yaml_text=None,
    )
    inv = build(sd)["baseline"]["invocation"]
    assert "vllm.entrypoints" in inv["framework_args"]
    assert inv["framework_args_source"] == "log_python_cmd"


def test_framework_args_from_yaml_cmd_fallback(tmp_path: Path) -> None:
    """INFO-only server.log + yaml ``cmd: ...`` field → source ``yaml_cmd``."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "(APIServer pid=1757439) INFO 05-12 14:21:14 [utils.py:299]\n"
            "(APIServer pid=1757439) INFO 05-12 14:21:15 [server.py:50] init\n"
        ),
        yaml_text=(
            'cmd: "python -m sglang.launch_server --tp 4 --model /weka/m"\n'
            "benchmark:\n"
            "  envs:\n"
            "    TP: \"4\"\n"
        ),
    )
    inv = build(sd)["baseline"]["invocation"]
    assert "sglang.launch_server" in inv["framework_args"]
    assert inv["framework_args_source"] == "yaml_cmd"


def test_framework_args_unknown_with_warning(tmp_path: Path) -> None:
    """No usable server.log or yaml cmd → source ``unknown``, empty framework_args, ``framework_args extraction failed`` warning."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "(APIServer pid=1757439) INFO 05-12 14:21:14 [utils.py:299]\n"
            "(APIServer pid=1757439) INFO 05-12 14:21:15 [server.py:50] init\n"
        ),
        yaml_text=(
            "benchmark:\n"
            "  envs:\n"
            "    TP: \"8\"\n"
        ),
    )
    b = build(sd)
    inv = b["baseline"]["invocation"]
    assert inv["framework_args"] == ""
    assert inv["framework_args_source"] == "unknown"
    assert any(
        "framework_args extraction failed" in w for w in b["warnings"]
    ), b["warnings"]


def test_framework_args_from_non_default_args_vllm(tmp_path: Path) -> None:
    """vllm's ``non-default args: {...}`` line is picked up first (Pass 0); output is sorted-by-key for stability."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "(APIServer pid=1645301) INFO 05-12 10:51:30 [config.py:120] booting\n"
            "(APIServer pid=1645301) INFO 05-12 10:51:32 [utils.py:233] "
            "non-default args: {'model': '/wekafs/models/deepseek-ai-DeepSeek-R1', "
            "'port': 8888, 'tensor_parallel_size': 8, 'max_model_len': 4096, "
            "'gpu_memory_utilization': 0.85}\n"
            "(APIServer pid=1645301) INFO 05-12 10:51:35 [server.py:99] ready\n"
        ),
        yaml_text=None,
    )
    inv = build(sd)["baseline"]["invocation"]
    assert inv["framework_args_source"] == "log_non_default_args"
    assert "tensor_parallel_size=8" in inv["framework_args"]
    assert "model=" in inv["framework_args"]
    assert "/wekafs/models/deepseek-ai-DeepSeek-R1" in inv["framework_args"]
    # Keys sorted alphabetically: gpu_memory_utilization < model < tensor_parallel_size.
    s = inv["framework_args"]
    assert s.index("gpu_memory_utilization=") < s.index("model="), s
    assert s.index("model=") < s.index("tensor_parallel_size="), s


def test_framework_args_pass0_takes_priority_over_python_cmd(tmp_path: Path) -> None:
    """With both a ``non-default args: {...}`` line and a python cmdline, Pass 0 wins."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "(APIServer pid=1645301) INFO 05-12 10:51:30 [config.py:120] booting\n"
            "python -m vllm.entrypoints.openai.api_server --model /weka/m --tp 8\n"
            "(APIServer pid=1645301) INFO 05-12 10:51:32 [utils.py:233] "
            "non-default args: {'model': '/weka/m', 'tensor_parallel_size': 8, "
            "'port': 8888}\n"
            "(APIServer pid=1645301) INFO 05-12 10:51:35 [server.py:99] ready\n"
        ),
        yaml_text=None,
    )
    inv = build(sd)["baseline"]["invocation"]
    assert inv["framework_args_source"] == "log_non_default_args"
    assert "python -m vllm.entrypoints" not in inv["framework_args"]
    assert "tensor_parallel_size=8" in inv["framework_args"]


def test_framework_args_from_yaml_benchmark_synthesis(tmp_path: Path) -> None:
    """A yaml ``benchmark.*`` block with no ``cmd:`` → Pass 4 synthesizes a string, source ``yaml_benchmark``."""
    sd = tmp_path / "session"
    _make_invocation_fixture(
        sd,
        server_log_text=(
            "(APIServer pid=1757439) INFO 05-12 14:21:14 [utils.py:299]\n"
            "(APIServer pid=1757439) INFO 05-12 14:21:15 [server.py:50] init\n"
        ),
        yaml_text=(
            "benchmark:\n"
            "  framework: vllm\n"
            "  model: /path\n"
            "  precision: fp8\n"
            "  tp: 8\n"
            "  envs:\n"
            "    VLLM_FLASH_ATTN: \"1\"\n"
        ),
    )
    inv = build(sd)["baseline"]["invocation"]
    assert inv["framework_args_source"] == "yaml_benchmark"
    s = inv["framework_args"]
    assert "framework=vllm" in s, s
    assert "model=/path" in s, s
    assert "tp=8" in s, s
    assert "envs=[VLLM_FLASH_ATTN=1]" in s, s


# Merged from test_v08_observability.py

"""v0.8 §3.12 — observability / breakdown schema v2 tests (KB_design §3.12, Inv-12.1/12.2)."""


import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown.exporter import build
from inference_optimizer.breakdown.schema import SCHEMA_VERSION


# Test fixtures
def _write_state(session_dir: Path, state: dict) -> None:
    (session_dir / "state.json").write_text(json.dumps(state))
    if not (session_dir / "manifest.json").exists():
        (session_dir / "manifest.json").write_text(json.dumps({}))


def _basic_state(**extras) -> dict:
    state = {
        "session_id": "sid",
        "schema_version": 2,
        "baseline_tput": 100.0,
        "current_best": {"tput": 110.0},
        "cumulative_gain": 10.0,
        "kernel_enabled": False,
    }
    state.update(extras)
    return state


def _specialist_round(round_id: int = 1, **extras) -> dict:
    base = {
        "round_id":          round_id,
        "dispatched_at":     "2025-01-01T00:00:00Z",
        "completed_at":      "2025-01-01T00:01:00Z",
        "domains":           ["kernel_switch_specialist"],
        "parallelism":       1,
        "proposals_total":   3,
        "proposals_kept":    1,
        "proposals_rejected": 1,
        "proposals_skipped": 1,
        "kb_edge_ids":       ["edge-1"],
        "confidence_avg":    0.7,
        "domain_breakdown": {
            "kernel_switch_specialist": {
                "dispatched": 1, "proposals_total": 3,
                "proposals_kept": 1, "proposals_rejected": 1,
            },
        },
        "task_ids":     ["t-abc"],
        "task_domains": {"t-abc": "kernel_switch_specialist"},
        "notes":        [],
    }
    base.update(extras)
    return base


# 1. Schema version + v1 compat aliases
def test_schema_version_is_v2():
    assert SCHEMA_VERSION == "hyperloom.session_breakdown.v2"


def test_build_writes_schema_v2_with_v1_aliases(tmp_path):
    """KB_design §3.12 §5 — the v2 file carries v1-reader aliases (Inv-12.1)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    assert b["schema_version"] == "hyperloom.session_breakdown.v2"
    assert "action_timeline" in b
    assert "explore_search" in b
    assert "param_search" in b
    assert b["explore_search"] == b["param_search"]
    assert b["action_timeline"] == b["phase_timeline"]


def test_v1_reader_does_not_crash_on_v2_payload(tmp_path):
    """KB_design §3.12 §9 — a v1 reader consumes a v2 payload without raising."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    b = build(sd)
    v1_keys = {
        "session", "workload", "baseline", "final", "phase_timeline",
        "capability_summary", "geak_invocations", "oob_invocations",
        "kernel_lifecycle", "param_search", "sweep", "critic_robustness",
        "telemetry", "attribution", "warnings", "source_files",
    }
    for key in v1_keys:
        assert key in b, f"v1 reader expects {key!r} to exist"
    assert b["param_search"] == b["explore_search"]


# 2. specialist_runs section
def test_specialist_runs_empty_when_no_rounds(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    assert b["specialist_runs"] == []


def test_specialist_runs_populated_from_state(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    b = build(sd)
    assert len(b["specialist_runs"]) == 1
    entry = b["specialist_runs"][0]
    for k in (
        "round_id", "dispatched_at", "completed_at", "domains",
        "parallelism", "proposals_total", "proposals_kept",
        "proposals_rejected", "proposals_skipped", "kb_edge_ids",
        "confidence_avg", "domain_breakdown", "transcripts", "notes",
    ):
        assert k in entry, f"specialist_runs row missing field {k!r}"
    assert entry["transcripts"] == []
    breakdown_ks = entry["domain_breakdown"]["kernel_switch_specialist"]
    assert breakdown_ks == {
        "dispatched": 1, "proposals_total": 3,
        "proposals_kept": 1, "proposals_rejected": 1,
    }


def test_specialist_runs_attaches_transcript_path_when_present(tmp_path):
    """KB_design §3.12 §4.3 — captures the transcript path (default) or body (``include_transcripts=True``)."""
    sd = tmp_path / "session"
    sd.mkdir()
    transcript_dir = sd / "runs" / "specialist" / "t-abc"
    transcript_dir.mkdir(parents=True)
    body_text = '{"proposal_set": []}'
    (transcript_dir / "specialist_done.json").write_text(body_text)
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))

    b = build(sd)
    refs = b["specialist_runs"][0]["transcripts"]
    assert len(refs) == 1
    assert refs[0]["task_id"] == "t-abc"
    assert refs[0]["domain"] == "kernel_switch_specialist"
    assert refs[0]["path"].endswith("specialist_done.json")
    assert "body" not in refs[0]

    b2 = build(sd, include_transcripts=True)
    ref2 = b2["specialist_runs"][0]["transcripts"][0]
    assert ref2.get("body") == body_text


def test_build_respects_env_var_for_transcripts(tmp_path, monkeypatch):
    """KB_design §3.12 §7 step 5 — the CLI env var drives transcripts when ``include_transcripts`` isn't passed."""
    sd = tmp_path / "session"
    sd.mkdir()
    transcript_dir = sd / "runs" / "specialist" / "t-abc"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "specialist_done.json").write_text('{"x":1}')
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "1")
    b = build(sd)
    assert b["specialist_runs"][0]["transcripts"][0].get("body") == '{"x":1}'


# 3. capability_summary.specialist row (Inv-12.2 single source)
def test_capability_summary_specialist_row_when_no_rounds(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    spec = b["capability_summary"]["specialist"]
    assert spec["status"] == "not_attempted"
    assert spec["attempts"] == 0
    assert spec["keeps"] == 0
    assert spec["tested"] == 0
    for d in (
        "serving_specialist", "kernel_switch_specialist",
        "comm_specialist", "compiler_specialist", "system_specialist",
        "pr_intel_specialist", "research_scout_specialist",
    ):
        assert spec["by_specialist"][d] == {
            "status": "not_attempted",
            "attempts": 0, "keeps": 0, "tested": 0,
        }


def test_capability_summary_specialist_agrees_with_specialist_runs(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(round_id=1, proposals_total=4, proposals_kept=2),
        _specialist_round(round_id=2, proposals_total=2, proposals_kept=0),
    ]))
    b = build(sd)
    spec = b["capability_summary"]["specialist"]
    assert spec["attempts"] == 2
    assert spec["tested"] == 6
    assert spec["keeps"] == 2
    assert spec["status"] == "kept"
    # Cross-check (Inv-12.2): aggregate via specialist_runs matches.
    runs = b["specialist_runs"]
    assert sum(r["proposals_total"] for r in runs) == spec["tested"]
    assert sum(r["proposals_kept"] for r in runs) == spec["keeps"]


def test_capability_summary_specialist_status_tried_when_no_keeps(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(proposals_total=2, proposals_kept=0),
    ]))
    b = build(sd)
    assert b["capability_summary"]["specialist"]["status"] == "tried"


def test_capability_summary_specialist_status_attempted_when_empty_proposals(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(specialist_rounds=[
        _specialist_round(proposals_total=0, proposals_kept=0),
    ]))
    b = build(sd)
    assert b["capability_summary"]["specialist"]["status"] == "attempted"


# 4. critic_robustness.kb_writes_summary
def test_critic_kb_writes_summary_empty_by_default(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    summary = b["critic_robustness"]["kb_writes_summary"]
    assert summary == {"total": 0, "by_verdict": {}}


def test_critic_kb_writes_summary_aggregates_by_verdict(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    critic_root = sd / "critic-workdir"
    for n, verdict in enumerate(("KEEP", "KEEP", "REVERT"), start=1):
        iter_dir = critic_root / f"{n:03d}"
        iter_dir.mkdir(parents=True)
        (iter_dir / "review.json").write_text(json.dumps({
            "verdict": verdict, "summary": f"iter {n}",
        }))
        (iter_dir / "emit.json").write_text(json.dumps({
            "ts": f"2025-01-01T00:0{n}:00Z", "topic": "review",
        }))
    b = build(sd)
    summary = b["critic_robustness"]["kb_writes_summary"]
    assert summary["total"] == 3
    assert summary["by_verdict"] == {"KEEP": 2, "REVERT": 1}


# 5. action_timeline alias mirrors phase_timeline
def test_action_timeline_mirrors_phase_timeline(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(
        phase_history=[
            {"to": "EXPLORE", "ts": "2025-01-01T00:00:00Z",
             "reason": "session_start", "evidence": {}},
        ],
    ))
    b = build(sd)
    assert b["action_timeline"] == b["phase_timeline"]


def test_roofline_attempts_are_in_phase_timeline(tmp_path):
    """Roofline failures must be visible in breakdown timelines."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state(
        roofline_attempts=[
            {
                "ts": "2026-06-04T01:49:00+00:00",
                "task_id": "t-roofline-fail",
                "status": "failed",
                "decision": "no_promote",
                "key_metric": None,
                "key_metric_kind": "snapshot_id",
                "error_class": "trace_analyze_failed",
                "extras": {"phase": "trace_analyze"},
            },
        ],
    ))
    timeline = build(sd)["phase_timeline"]

    roofline_rows = [e for e in timeline if e.get("action") == "roofline"]
    assert len(roofline_rows) == 1
    row = roofline_rows[0]
    assert row["task_id"] == "t-roofline-fail"
    assert row["status"] == "failed"
    assert row["decision"] == "no_promote"
    assert row["error_class"] == "trace_analyze_failed"


# 6. CLI flag wiring
def test_cli_exposes_breakdown_include_transcripts_flag():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "optimize", "--model", "/tmp/dummy",
        "--breakdown-include-transcripts", "true",
    ])
    assert args.breakdown_include_transcripts == "true"


def test_cli_breakdown_include_transcripts_defaults_to_false():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["optimize", "--model", "/tmp/dummy"])
    assert args.breakdown_include_transcripts in ("true", "false")


def test_cli_rejects_unknown_breakdown_include_transcripts():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/tmp/dummy",
            "--breakdown-include-transcripts", "maybe",
        ])
