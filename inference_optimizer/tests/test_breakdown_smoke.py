"""End-to-end smoke test for the breakdown exporter.

Builds a synthetic but realistic session_dir tree (manifest + state +
runs/ + kernel-agent-workspace + critic/robustness workdirs) and asserts
that:

1. ``build()`` returns a dict matching the top-level envelope.
2. ``write_breakdown_json()`` writes valid JSON that round-trips.
3. Per-attempt KEEP stamping correctly assigns the kernel-level decision
   to only the BEST attempt for each kernel (not every attempt).
4. Each of the 14 sections has the expected key facts.
"""

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


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------
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
        "current_best":   {"tput": 776.4, "action": "validate_stack",
                            "extra_sglang_args": "--enable-X --nccl-Y",
                            "extra_envs": {"NCCL_DEBUG": "INFO"},
                            "ttft_mean_ms": 110.2, "e2el_mean_ms": 1200.4},
        "optimization_stack": [
            {"action": "backends",     "variant_name": "flag_X", "gain_pct": 23.4,
             "extra_sglang_args": "--enable-X",
             "ts": "2026-05-14T07:15:00+00:00"},
            {"action": "params",       "variant_name": "nccl_Y", "gain_pct":  7.8,
             "extra_sglang_args": "--nccl-Y",
             "ts": "2026-05-14T07:30:00+00:00"},
            {"action": "kernel_opt:k001", "variant_name": "",  "gain_pct": 45.6,
             "extra_sglang_args": "",
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
        "backends_attempts": [
            {"ts": "2026-05-14T07:15:00+00:00", "task_id": "b1",
             "status": "succeeded", "decision": "promoted", "key_metric": 23.4},
        ],
        "params_attempts": [
            {"ts": "2026-05-14T07:30:00+00:00", "task_id": "pa1",
             "status": "succeeded", "decision": "promoted", "key_metric": 7.8},
        ],
        "sweep_attempts": [
            {"ts": "2026-05-14T08:20:00+00:00", "task_id": "s1",
             "status": "succeeded", "decision": "promoted",
             "key_metric": 812.0, "workspace": str(sd / "runs/sweep/s1")},
        ],
        "validate_stack_attempts": [
            {"ts": "2026-05-14T08:30:00+00:00", "task_id": "v1",
             "status": "succeeded", "decision": "promoted", "key_metric": 84.2},
        ],
        "last_select_kernels": {"hot_kernels_top15": [
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
            "target_file": "/path/to/rmsnorm.py", "extra_sglang_args": "",
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
        "params_search": {
            "schema_version": 2,
            "accepted": [{"name": "nccl_Y", "fingerprint": "f1",
                           "extra_sglang_args": "--nccl-Y", "extra_envs": {},
                           "output_throughput": 454.3, "gain_pct": 7.8,
                           "ts": "2026-05-14T07:30:00+00:00"}],
            "rejected": [{"name": "bad_x", "fingerprint": "f2", "gain_pct": -2.5}],
            "tested": {
                "f1": {"name": "nccl_Y",  "fingerprint": "f1", "gain_pct":  7.8},
                "f2": {"name": "bad_x",   "fingerprint": "f2", "gain_pct": -2.5},
                "f3": {"name": "neutral", "fingerprint": "f3", "gain_pct":  0.1},
            },
            "name_index": {}, "cursor": 3,
        },
        "backends_search": {
            "schema_version": 1,
            "accepted": [{"name": "flag_X", "fingerprint": "g1",
                           "gain_pct": 23.4,
                           "ts": "2026-05-14T07:15:00+00:00"}],
            "rejected": [],
            "tested": {"g1": {"name": "flag_X", "fingerprint": "g1", "gain_pct": 23.4}},
            "name_index": {}, "cursor": 1,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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
    assert cap["backends"]["status"] == "kept"
    assert cap["backends"]["best_gain_pct"] == pytest.approx(23.4)
    assert cap["params"]["status"] == "kept"
    assert cap["sweep"]["status"] == "completed"
    assert cap["validate_stack"]["last_validated_gain_pct"] == pytest.approx(84.2)


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
    assert sb["backends_pct_of_total"] == pytest.approx(23.4)
    assert sb["params_pct_of_total"] == pytest.approx(7.8)
    assert sb["validated_total_pct"] == pytest.approx(84.2)


def test_phase_timeline_sorted_by_ts(fixture_session: Path) -> None:
    ts = build(fixture_session)["phase_timeline"]
    ts_values = [e.get("ts") or "" for e in ts]
    assert ts_values == sorted(ts_values)
    actions = {e["action"] for e in ts}
    assert "baseline" in actions
    assert "validate_stack" in actions
    assert "kernel_opt" in actions or "integrate" in actions


def test_param_search_summary(fixture_session: Path) -> None:
    ps = build(fixture_session)["param_search"]
    assert ps["params"]["tested_count"] == 3
    assert ps["backends"]["tested_count"] == 1
    assert ps["params"]["accepted"][0]["name"] == "nccl_Y"
    assert ps["backends"]["accepted"][0]["name"] == "flag_X"


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
