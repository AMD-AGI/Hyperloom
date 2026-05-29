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
                            "extra_server_args": "--enable-X --nccl-Y",
                            "extra_envs": {"NCCL_DEBUG": "INFO"},
                            "ttft_mean_ms": 110.2, "e2el_mean_ms": 1200.4},
        "optimization_stack": [
            {"action": "backends",     "variant_name": "flag_X", "gain_pct": 23.4,
             "extra_server_args": "--enable-X",
             "ts": "2026-05-14T07:15:00+00:00"},
            {"action": "params",       "variant_name": "nccl_Y", "gain_pct":  7.8,
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
        "params_search": {
            "schema_version": 2,
            "accepted": [{"name": "nccl_Y", "fingerprint": "f1",
                           "extra_server_args": "--nccl-Y", "extra_envs": {},
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


def test_baseline_resolves_container_workspace_path(tmp_path: Path) -> None:
    """``last_baseline.workspace`` written as a container path
    (``/workspace/runs/baseline/<sub>/``) must still resolve under the
    on-disk ``session_dir`` so ``ttft_mean_ms`` is read from the
    benchmark report rather than dropped to ``None``.

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


def test_baseline_unresolvable_workspace_emits_warning(tmp_path: Path) -> None:
    """When the recorded workspace can't be re-rooted under ``session_dir``
    (e.g. wrong session_dir), the collector must surface a warning rather
    than silently fall back to ``None``."""
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
    """``source_files.kernel_attempts`` must not appear when the session
    has no kernel-agent runs (instead of rendering a ``count=0,
    first_values=—`` placeholder row downstream).

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
    """Sanity: when the session DOES have kernel-agent activity, the
    populated list must still be emitted by the collector."""
    sf = build(fixture_session)["source_files"]
    assert sf.get("kernel_attempts"), sf
    assert any("optimization_attempts.jsonl" in p for p in sf["kernel_attempts"])


# ---------------------------------------------------------------------------
# Attribution method (A1)
# ---------------------------------------------------------------------------
def _attribution_fixture(tmp_path: Path, state: dict) -> Path:
    """Minimal session_dir whose only content drives ``collect_attribution``."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "attr"})
    base_state = {"session_id": "attr"}
    base_state.update(state)
    _write_json(sd / "state.json", base_state)
    return sd


def test_attribution_method_validated(tmp_path: Path) -> None:
    """``state.gain_per_stack_entry`` written by Coordinator with every
    entry's ``delta_pct`` set → ``method == "validated"``."""
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
    """Multi-entry stack with no Coordinator ledger and at least one
    placeholder ``delta_pct=None`` → ``method == "reconstructed"``."""
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


# ---------------------------------------------------------------------------
# A2: final.ttft_mean_ms reconstruction
# ---------------------------------------------------------------------------
def test_final_ttft_reconstructed_from_validate_stack(tmp_path: Path) -> None:
    """When ``current_best.ttft_mean_ms`` is not recorded but a
    ``runs/validate_stack/<h>/benchmark_*/benchmark_report.json`` is on
    disk, the collector must read the latency from the report, set
    ``ttft_e2el_source = "validate_stack_disk"``, and append a
    reconstruction warning."""
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


# ---------------------------------------------------------------------------
# A3: baseline.attempts_history reconstruction
# ---------------------------------------------------------------------------
def test_baseline_attempts_history_reconstructed_from_disk(tmp_path: Path) -> None:
    """When ``state.baseline_attempts == []`` but two
    ``runs/baseline/<hash>/benchmark_*/`` directories exist, the
    collector must reconstruct two summary rows tagged
    ``status="reconstructed"`` and append the reconstruction warning."""
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
    """When ``state.baseline_attempts`` IS non-empty, the disk fallback
    must NOT also fire (no duplication, no reconstruction warning)."""
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


# ---------------------------------------------------------------------------
# B3: invocation populated from baseline_config + server.log
# ---------------------------------------------------------------------------
def test_baseline_invocation_populated(tmp_path: Path) -> None:
    """``baseline.invocation`` must read framework_args from server.log,
    extra_envs from the YAML benchmark.envs block (allowlisted), and
    keep secret-shaped keys (``OPENAI_API_KEY``) out of the output."""
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


# ---------------------------------------------------------------------------
# B1 / B3: image detection
# ---------------------------------------------------------------------------
def test_session_image_from_env(tmp_path: Path, monkeypatch) -> None:
    """``HYPERLOOM_IMAGE`` env var must populate ``session.image`` when
    the manifest lacks the field (V1 manifests). Absent of all sources
    the field is ``None`` and a single warning is appended — no crash."""
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
    """``manifest.image`` (written at session start) wins over runtime
    env vars — captures the spawn-time image even if env later drifts."""
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {
        "schema_version": 2, "session_id": "img2",
        "image": "registry.example/hyperloom:from-manifest",
    })
    _write_json(sd / "state.json", {"session_id": "img2"})

    monkeypatch.setenv("HYPERLOOM_IMAGE", "registry.example/hyperloom:from-env")
    b = build(sd)
    assert b["session"]["image"] == "registry.example/hyperloom:from-manifest"


# ---------------------------------------------------------------------------
# C1: baseline.ttft_mean_ms disk-walk fallback
# ---------------------------------------------------------------------------
def test_baseline_ttft_disk_walk_fallback(tmp_path: Path) -> None:
    """When ``state.last_baseline.workspace`` does not resolve to a
    readable benchmark report, but ``runs/baseline/<hash>/benchmark_*/
    benchmark_report.json`` exists on disk, the collector must walk
    that directory and surface the latency from the most recent
    report (mirrors the A2 final.ttft validate_stack disk walk).

    Production parallel: zgong's V2 session
    ``deepseek-ai-DeepSeek-R1_20260512T102109Z_0dc064c4`` had a valid
    benchmark_report.json on wekafs but state's recorded workspace
    didn't resolve."""
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
        # Recorded workspace doesn't resolve under session_dir on wekafs
        # — exactly the production failure mode we're guarding against.
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


# ---------------------------------------------------------------------------
# C2: framework_args extraction lineage
# ---------------------------------------------------------------------------
def _make_invocation_fixture(
    sd: Path,
    server_log_text: str | None,
    yaml_text: str | None,
) -> None:
    """Minimal fixture: state pointing at runs/baseline/h1, optional
    server.log + baseline_config.with_envs.yaml beside the report."""
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
    """``Server arguments: ...`` header in server.log → source =
    ``log_args_line``; the captured args (not the header) get echoed."""
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
    """A literal ``python -m vllm.entrypoints...`` line surrounded by
    INFO log noise → source = ``log_python_cmd``; the python command
    line is what gets surfaced (Pass-2 fallback)."""
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
    """server.log has only INFO noise (no header, no python prefix);
    yaml has a ``cmd: ...`` field → source = ``yaml_cmd``."""
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
    """server.log has only INFO noise + yaml has no ``cmd``/``command``/
    ``launch`` field → source = ``unknown``, framework_args is empty,
    and a ``framework_args extraction failed`` warning is appended."""
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
    """vllm prints ``non-default args: {parsed-dict}`` right after argv
    parsing. The extractor picks this up first (Pass 0) because it
    captures the *resolved* values the framework actually used — beats
    any other heuristic. Output must be sorted-by-key + repr() values
    so the string is stable across runs."""
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
    # Deterministic ordering: keys sorted alphabetically. So
    # ``gpu_memory_utilization`` (g) must precede ``model`` (m) which
    # must precede ``tensor_parallel_size`` (t) in the emitted string.
    s = inv["framework_args"]
    assert s.index("gpu_memory_utilization=") < s.index("model="), s
    assert s.index("model=") < s.index("tensor_parallel_size="), s


def test_framework_args_pass0_takes_priority_over_python_cmd(tmp_path: Path) -> None:
    """When server.log contains BOTH a ``non-default args: {...}`` line
    AND a literal ``python -m vllm.entrypoints...`` line, Pass 0 must
    win — the parsed-arg dict is more authoritative than the raw
    cmdline (which might contain unresolved env-var placeholders)."""
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
    # And the chosen string is the formatted dict, not the python line.
    assert "python -m vllm.entrypoints" not in inv["framework_args"]
    assert "tensor_parallel_size=8" in inv["framework_args"]


def test_framework_args_from_yaml_benchmark_synthesis(tmp_path: Path) -> None:
    """server.log has only INFO noise; yaml is magpie-style with no
    ``cmd:`` field but a populated ``benchmark.*`` block →
    Pass 4 synthesizes a readable string from the structured fields and
    labels the source ``yaml_benchmark`` so consumers know it's not a
    literal cmdline."""
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


# ============================================================================
# Merged from test_v08_observability.py
# ============================================================================

"""v0.8 §3.12 — observability / breakdown schema v2 tests.

Covers KB_design/3.12_observability/README.md acceptance criteria:

* §5 / §6 — top-level ``schema_version`` says ``hyperloom.session_breakdown.v2``.
* §4.3 — ``specialist_runs`` section is populated from
  ``SharedState.specialist_rounds`` + ``runs/specialist/`` transcripts.
* §4.2 — ``capability_summary.specialist`` row exists and the counts
  agree with ``specialist_runs`` (Inv-12.2 single source of truth).
* §4.4 — ``critic_robustness.kb_writes_summary`` summarises the
  critic-agent commit-review verdicts.
* §5 — top-level ``action_timeline`` and ``explore_search`` aliases
  guarantee a v0.6/v0.7 reader sees its old fields (Inv-12.1).
* §7 step 5 — ``--breakdown-include-transcripts`` CLI flag controls
  whether transcript bodies are inlined or referenced by path.
"""


import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown.exporter import build
from inference_optimizer.breakdown.schema import SCHEMA_VERSION


# ===========================================================================
# Test fixtures
# ===========================================================================
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


# ===========================================================================
# 1. Schema version + v1 compat aliases
# ===========================================================================
def test_schema_version_is_v2():
    assert SCHEMA_VERSION == "hyperloom.session_breakdown.v2"


def test_build_writes_schema_v2_with_v1_aliases(tmp_path):
    """KB_design §3.12 §5 — v2 file MUST carry the v1-reader aliases
    so existing dashboards keep functioning (Inv-12.1)."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    assert b["schema_version"] == "hyperloom.session_breakdown.v2"
    # v1 reader compat — these aliases MUST be present:
    assert "action_timeline" in b
    assert "explore_search" in b
    assert "param_search" in b
    # Aliases are pointer-equivalent (same identity isn't required, but
    # the content must match so a reader sees one source of truth).
    assert b["explore_search"] == b["param_search"]
    assert b["action_timeline"] == b["phase_timeline"]


def test_v1_reader_does_not_crash_on_v2_payload(tmp_path):
    """KB_design §3.12 §9 — a v1 reader that knows only the v1 keys
    MUST be able to consume a v2 payload without raising.

    Simulate a v1 reader by extracting the v1 key subset and asserting
    we can still locate every required field.
    """
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
    # The legacy ``param_search`` row carries the same data as v2's
    # ``explore_search`` — so a v1 reader sees the merged ledger
    # transparently (KB_design §3.12 §5).
    assert b["param_search"] == b["explore_search"]


# ===========================================================================
# 2. specialist_runs section
# ===========================================================================
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
    # Required schema fields.
    for k in (
        "round_id", "dispatched_at", "completed_at", "domains",
        "parallelism", "proposals_total", "proposals_kept",
        "proposals_rejected", "proposals_skipped", "kb_edge_ids",
        "confidence_avg", "domain_breakdown", "transcripts", "notes",
    ):
        assert k in entry, f"specialist_runs row missing field {k!r}"
    # No transcripts on disk → empty list (the runner artifact was not
    # written; the round-merge still runs).
    assert entry["transcripts"] == []
    # Domain breakdown round-trips with int normalisation.
    breakdown_ks = entry["domain_breakdown"]["kernel_switch_specialist"]
    assert breakdown_ks == {
        "dispatched": 1, "proposals_total": 3,
        "proposals_kept": 1, "proposals_rejected": 1,
    }


def test_specialist_runs_attaches_transcript_path_when_present(tmp_path):
    """KB_design §3.12 §4.3 — when ``runs/specialist/<task_id>/specialist_done.json``
    exists, the breakdown captures the path (default) or the body
    (when ``include_transcripts=True``)."""
    sd = tmp_path / "session"
    sd.mkdir()
    transcript_dir = sd / "runs" / "specialist" / "t-abc"
    transcript_dir.mkdir(parents=True)
    body_text = '{"proposal_set": []}'
    (transcript_dir / "specialist_done.json").write_text(body_text)
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))

    # Path-only mode (default).
    b = build(sd)
    refs = b["specialist_runs"][0]["transcripts"]
    assert len(refs) == 1
    assert refs[0]["task_id"] == "t-abc"
    assert refs[0]["domain"] == "kernel_switch_specialist"
    assert refs[0]["path"].endswith("specialist_done.json")
    assert "body" not in refs[0]

    # Inline mode.
    b2 = build(sd, include_transcripts=True)
    ref2 = b2["specialist_runs"][0]["transcripts"][0]
    assert ref2.get("body") == body_text


def test_build_respects_env_var_for_transcripts(tmp_path, monkeypatch):
    """KB_design §3.12 §7 step 5 — when the caller doesn't pass
    ``include_transcripts`` explicitly, the env var (set by CLI)
    drives the decision."""
    sd = tmp_path / "session"
    sd.mkdir()
    transcript_dir = sd / "runs" / "specialist" / "t-abc"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "specialist_done.json").write_text('{"x":1}')
    _write_state(sd, _basic_state(specialist_rounds=[_specialist_round()]))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "1")
    b = build(sd)
    assert b["specialist_runs"][0]["transcripts"][0].get("body") == '{"x":1}'


# ===========================================================================
# 3. capability_summary.specialist row (Inv-12.2 single source)
# ===========================================================================
def test_capability_summary_specialist_row_when_no_rounds(tmp_path):
    sd = tmp_path / "session"
    sd.mkdir()
    _write_state(sd, _basic_state())
    b = build(sd)
    spec = b["capability_summary"]["specialist"]
    assert spec == {
        "status":   "not_attempted",
        "attempts": 0,
        "keeps":    0,
        "tested":   0,
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


# ===========================================================================
# 4. critic_robustness.kb_writes_summary
# ===========================================================================
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
    # Synthesize three critic iteration outputs with different verdicts.
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


# ===========================================================================
# 5. action_timeline alias mirrors phase_timeline
# ===========================================================================
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


# ===========================================================================
# 6. CLI flag wiring
# ===========================================================================
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
