# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for benchmark-result parsing: leak harvesting, rescue-path
salvage, raw-result merging, TPOT derivation, and OSL resolution."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.actions.executors import benchmark_result as br


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- _candidate_raw_jsons ordering ----------------------------------------
def test_candidate_raw_jsons_ordering(tmp_path):
    (tmp_path / "profile_x.json").write_text("{}", encoding="utf-8")
    (tmp_path / "baseline.json").write_text("{}", encoding="utf-8")
    (tmp_path / "benchmark_report.json").write_text("{}", encoding="utf-8")
    out = br._candidate_raw_jsons(tmp_path)
    names = [p.name for p in out]
    assert "benchmark_report.json" not in names
    assert names.index("baseline.json") < names.index("profile_x.json")


# ---- _resolve_osl ---------------------------------------------------------
def test_resolve_osl_variants():
    assert br._resolve_osl(None) is None
    assert br._resolve_osl({"osl": 128}) == 128
    assert br._resolve_osl({"config": {"output_len": 64}}) == 64
    assert br._resolve_osl({"request": {"max_tokens": 32}}) == 32
    assert br._resolve_osl({"osl": 0, "output_len": -1}) is None


# ---- _derive_tpot_if_missing ----------------------------------------------
def test_derive_tpot_already_set():
    m = {"tpot_mean_ms": 5.0}
    br._derive_tpot_if_missing(m, {})
    assert m["tpot_mean_ms"] == 5.0


def test_derive_tpot_missing_latencies():
    m = {"tpot_mean_ms": None, "e2el_mean_ms": None, "ttft_mean_ms": None}
    br._derive_tpot_if_missing(m, {})
    assert m["tpot_mean_ms"] is None


def test_derive_tpot_e2el_le_ttft():
    m = {"tpot_mean_ms": None, "e2el_mean_ms": 10.0, "ttft_mean_ms": 20.0}
    br._derive_tpot_if_missing(m, {"osl": 10})
    assert m["tpot_mean_ms"] is None


def test_derive_tpot_bad_osl():
    m = {"tpot_mean_ms": None, "e2el_mean_ms": 100.0, "ttft_mean_ms": 10.0}
    br._derive_tpot_if_missing(m, {"osl": 1})
    assert m["tpot_mean_ms"] is None


def test_derive_tpot_success():
    m = {"tpot_mean_ms": None, "e2el_mean_ms": 100.0, "ttft_mean_ms": 10.0}
    br._derive_tpot_if_missing(m, {"osl": 10})
    assert m["tpot_mean_ms"] == (100.0 - 10.0) / 9


# ---- is_valid_measurement -------------------------------------------------
def test_is_valid_measurement():
    assert br.is_valid_measurement(None) is False
    assert br.is_valid_measurement("nope") is False
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 5}) is True
    assert br.is_valid_measurement({"output_throughput": 0.0, "completed_requests": 5}) is False
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 0}) is False


# ---- _resolve_leak_roots --------------------------------------------------
def test_resolve_leak_roots_explicit(tmp_path):
    assert br._resolve_leak_roots(tmp_path) == (tmp_path,)


def test_resolve_leak_roots_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", "/a:/b")
    roots = br._resolve_leak_roots(None)
    assert roots == (Path("/a"), Path("/b"))


def test_resolve_leak_roots_default(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", raising=False)
    assert br._resolve_leak_roots(None) == (br._DEFAULT_LEAK_ARTIFACT_ROOT,)


# ---- _materialize_rescue_into_workspace -----------------------------------
def test_materialize_rescue_copy(tmp_path):
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    src = _write_json(leak_dir / "inferencex_result.json", {"x": 1})
    ws = tmp_path / "ws"
    dest = br._materialize_rescue_into_workspace(src, ws)
    assert dest is not None and dest.exists()


def test_materialize_rescue_inside_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    src = _write_json(ws / "inferencex_result.json", {"x": 1})
    assert br._materialize_rescue_into_workspace(src, ws) is None


def test_materialize_rescue_copy_error(tmp_path, monkeypatch):
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    src = _write_json(leak_dir / "inferencex_result.json", {"x": 1})
    monkeypatch.setattr(br.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert br._materialize_rescue_into_workspace(src, tmp_path / "ws") is None


# ---- _rescue_candidate_paths ----------------------------------------------
def test_rescue_candidate_paths_default(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", raising=False)
    assert br._rescue_candidate_paths(tmp_path) == []


def test_rescue_candidate_paths_env_dir_and_file(tmp_path, monkeypatch):
    leak = tmp_path / "leaks"
    leak.mkdir()
    _write_json(leak / "inferencex_result_1.json", {"x": 1})
    standalone = _write_json(tmp_path / "extra.json", {"y": 2})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", f"{leak}:{standalone}")
    ws = tmp_path / "ws"
    ws.mkdir()
    cands = br._rescue_candidate_paths(ws)
    names = {p.name for p in cands}
    assert "inferencex_result_1.json" in names
    assert "extra.json" in names


def test_rescue_candidate_paths_mtime_gate(tmp_path, monkeypatch):
    leak = _write_json(tmp_path / "inferencex_result.json", {"x": 1})
    import os

    old = leak.stat().st_mtime - 1000
    os.utime(leak, (old, old))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    ws = tmp_path / "ws"
    ws.mkdir()
    cands = br._rescue_candidate_paths(ws, subprocess_started_unix=leak.stat().st_mtime + 10000)
    assert cands == []


def test_rescue_candidate_paths_dedup_and_in_ws(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    inside = _write_json(ws / "inferencex_result.json", {"x": 1})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", f"{inside}:{inside}")
    assert br._rescue_candidate_paths(ws) == []


# ---- harvest_leaked_artifacts ---------------------------------------------
def test_harvest_leaked_artifacts(tmp_path):
    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    (leak_root / "server.log").write_text("log", encoding="utf-8")
    (leak_root / "gpu_metrics.csv").write_text("csv", encoding="utf-8")
    dest = tmp_path / "dest"
    out = br.harvest_leaked_artifacts(dest, leak_root=leak_root)
    copied = {c.name for _, c in out}
    assert "server.log" in copied
    assert "gpu_metrics.csv" in copied
    assert (dest / "server.log").exists()


def test_harvest_skips_non_file(tmp_path):
    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    (leak_root / "server.log").mkdir()
    dest = tmp_path / "dest"
    out = br.harvest_leaked_artifacts(dest, leak_root=leak_root)
    assert out == []


def test_harvest_mtime_gate(tmp_path):
    import os

    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    f = leak_root / "server.log"
    f.write_text("log", encoding="utf-8")
    old = f.stat().st_mtime - 5000
    os.utime(f, (old, old))
    dest = tmp_path / "dest"
    out = br.harvest_leaked_artifacts(dest, leak_root=leak_root, subprocess_started_unix=f.stat().st_mtime + 10000)
    assert out == []


def test_harvest_dest_mkdir_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(OSError("ro")))
    out = br.harvest_leaked_artifacts(tmp_path / "dest")
    assert out == []


def test_harvest_missing_root(tmp_path):
    out = br.harvest_leaked_artifacts(tmp_path / "dest", leak_root=tmp_path / "nonexistent")
    assert out == []


def test_harvest_skips_already_under_dest(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "server.log").write_text("log", encoding="utf-8")
    out = br.harvest_leaked_artifacts(dest, leak_root=dest)
    assert out == []


def test_harvest_copy_error(tmp_path, monkeypatch):
    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    (leak_root / "server.log").write_text("log", encoding="utf-8")
    monkeypatch.setattr(br.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    out = br.harvest_leaked_artifacts(tmp_path / "dest", leak_root=leak_root)
    assert out == []


# ---- extract_benchmark_measurement ----------------------------------------
def test_extract_from_report_only():
    report = {
        "success": True,
        "framework": "sglang",
        "throughput": {
            "output_throughput": 100.0,
            "completed_requests": 50,
            "request_throughput": 5.0,
            "duration_seconds": 10.0,
        },
        "latency": {"ttft": {"mean_ms": 20.0}, "e2el": {"mean_ms": 200.0}},
    }
    m = br.extract_benchmark_measurement(report)
    assert m["valid_measurement"] is True
    assert m["output_throughput"] == 100.0


def test_extract_with_workspace_raw(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_json(
        ws / "result.json",
        {
            "output_throughput": 80.0,
            "completed_requests": 40,
        },
    )
    report = {"success": False, "throughput": {}, "latency": {}}
    m = br.extract_benchmark_measurement(report, workspace=ws)
    assert m["valid_measurement"] is True
    assert "raw_inferencex_result_used" in m["nonfatal_warnings"]
    assert "benchmark_report_success_false" in m["nonfatal_warnings"]


def test_extract_skips_raw_without_throughput(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_json(ws / "result.json", {"completed_requests": 5})
    m = br.extract_benchmark_measurement({"throughput": {}}, workspace=ws)
    assert m["valid_measurement"] is False


def test_extract_rescue_path(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(
        tmp_path / "inferencex_result.json",
        {
            "output_throughput": 90.0,
            "completed_requests": 45,
        },
    )
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws, subprocess_started_unix=leak.stat().st_mtime - 100
    )
    assert m["valid_measurement"] is True
    assert any("rescued_from_leaked_path" in w for w in m["nonfatal_warnings"])


def test_extract_rescue_copy_failed_warning(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(
        tmp_path / "inferencex_result.json",
        {
            "output_throughput": 90.0,
            "completed_requests": 45,
        },
    )
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    monkeypatch.setattr(br, "_materialize_rescue_into_workspace", lambda *a, **k: None)
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws, subprocess_started_unix=leak.stat().st_mtime - 100
    )
    assert m["valid_measurement"] is True
    assert any("rescued_copy_into_workspace_failed" in w for w in m["nonfatal_warnings"])


def test_extract_rescue_skips_no_throughput(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(tmp_path / "inferencex_result.json", {"completed_requests": 5})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws, subprocess_started_unix=leak.stat().st_mtime - 100
    )
    assert m["valid_measurement"] is False


# ---- _row_to_gpu_sample ----------------------------------------------------
def test_row_to_gpu_sample_picks_metrics():
    header = [
        "timestamp",
        "Temperature (Junction) (C)",
        "Average Socket Power (W)",
        "sclk clock (MHz)",
        "GPU use (%)",
        "VRAM% ",
    ]
    row = ["1000", "55", "310.5", "1400", "88", "42"]
    s = br._row_to_gpu_sample(header, row)
    assert s["temperature_c"] == 55
    assert s["power_w"] == 310.5
    assert s["clock_mhz"] == 1400
    assert s["gpu_util_pct"] == 88
    assert s["vram_pct"] == 42


def test_row_to_gpu_sample_empty_when_no_numeric():
    assert br._row_to_gpu_sample(["timestamp", "note"], ["1000", "n/a"]) == {}


# ---- _aggregate_gpu_samples_by_role ---------------------------------------
def test_aggregate_gpu_samples_by_role():
    samples = [
        {"role": "prefill", "power_w": 100.0, "temperature_c": 50.0, "gpu_util_pct": 80.0, "vram_pct": 40.0},
        {"role": "prefill", "power_w": 200.0, "temperature_c": 60.0, "gpu_util_pct": 90.0, "vram_pct": 50.0},
        {"role": "decode", "power_w": 150.0},
    ]
    out = br._aggregate_gpu_samples_by_role(samples)
    assert out["prefill"]["samples"] == 2
    assert out["prefill"]["avg_power_w"] == 150.0
    assert out["prefill"]["max_power_w"] == 200.0
    assert out["prefill"]["max_temp_c"] == 60.0
    assert out["decode"]["samples"] == 1
    # No temperature samples for decode -> 0.0 fallback.
    assert out["decode"]["avg_temp_c"] == 0.0


def test_aggregate_gpu_samples_by_role_empty_without_role():
    assert br._aggregate_gpu_samples_by_role([{"power_w": 100.0}]) == {}


# ---- harvest_mn_gpu_metrics -----------------------------------------------
def test_harvest_mn_gpu_metrics_single_node_noop(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn
    monkeypatch.setattr(mn, "is_multi_node", lambda: False)
    assert br.harvest_mn_gpu_metrics(tmp_path) == {}


def test_harvest_mn_gpu_metrics_folds_pod_csvs(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(
        mn, "pd_topology_from_state",
        lambda: {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"]},
    )

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))

    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    (shared / "gpu_metrics_10.0.0.1.csv").write_text(
        f"{header}\n1000,55,300\n1001,56,310\n", encoding="utf-8"
    )
    (shared / "gpu_metrics_10.0.0.2.csv").write_text(
        f"{header}\n1000,50,280\n", encoding="utf-8"
    )

    dest = tmp_path / "ws"
    dest.mkdir()
    (dest / "benchmark_report.json").write_text("{}", encoding="utf-8")

    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)
    assert out["rows"] == 3
    assert (dest / "gpu_metrics.csv").is_file()
    report = json.loads((dest / "benchmark_report.json").read_text(encoding="utf-8"))
    assert "gpu_monitor" in report
    assert report["pd"]["prefill_pod_ips"] == ["10.0.0.1"]
    assert "prefill" in report["gpu_monitor_by_role"]
    assert "decode" in report["gpu_monitor_by_role"]


def test_harvest_mn_gpu_metrics_no_csvs(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    assert br.harvest_mn_gpu_metrics(tmp_path / "ws") == {}


# ---- _is_scriptable_measurement / is_valid_measurement --------------------
def test_is_scriptable_via_quality_gate():
    # A quality_gate marker classifies the measurement as scriptable.
    assert br._is_scriptable_measurement({"quality_gate": {"passed": True}}) is True


def test_is_valid_measurement_scriptable_gate(monkeypatch):
    # Scriptable run: valid on positive throughput when the quality gate passes.
    assert br.is_valid_measurement(
        {"quality_gate": {"passed": True}, "output_throughput": 5.0}
    ) is True
    # Scriptable run whose quality gate fails is not selectable.
    monkeypatch.setattr(br, "quality_gate_passed", lambda qg, require=False: False, raising=False)
    import hyperloom.orchestrator.actions.executors._accuracy_gate as ag
    monkeypatch.setattr(ag, "quality_gate_passed", lambda qg, require=False: False)
    assert br.is_valid_measurement(
        {"quality_gate": {"passed": False}, "output_throughput": 5.0}
    ) is False


def test_is_valid_measurement_serving_and_bad_input():
    # Serving measurement: needs positive throughput AND completed requests.
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 3}) is True
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 0}) is False
    assert br.is_valid_measurement({"output_throughput": 0.0}) is False
    assert br.is_valid_measurement(None) is False


def test_harvest_mn_gpu_metrics_non_absolute_dir(monkeypatch, tmp_path):
    # Unresolved / relative shared dir is treated as absent.
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "relative/path")
    assert br.harvest_mn_gpu_metrics(tmp_path / "ws") == {}


def test_harvest_mn_gpu_metrics_window_and_malformed_rows(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", lambda: {})

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))

    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    # Rows: one in-window, one out-of-window (dropped), one blank, one bad ts.
    (shared / "gpu_metrics_hostA.csv").write_text(
        f"{header}\n1000,55,300\n5,50,280\n\n,51,290\n", encoding="utf-8"
    )
    # A too-short file (header only) is skipped.
    (shared / "gpu_metrics_hostB.csv").write_text(f"{header}\n", encoding="utf-8")

    dest = tmp_path / "ws"
    dest.mkdir()
    # No benchmark_report.json → sample-inject branch is skipped, csv still written.
    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)
    assert out["rows"] == 1
    assert (dest / "gpu_metrics.csv").is_file()


def test_harvest_mn_gpu_metrics_report_not_dict(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", lambda: {})

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    (shared / "gpu_metrics_hostA.csv").write_text(
        f"{header}\n1000,55,300\n", encoding="utf-8"
    )

    dest = tmp_path / "ws"
    dest.mkdir()
    # Non-dict report JSON → inject branch bails without raising.
    (dest / "benchmark_report.json").write_text("[]", encoding="utf-8")
    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)
    assert out["rows"] == 1
    assert "gpu_monitor_samples" not in out
