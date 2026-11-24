# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for benchmark-result parsing: leak harvesting, rescue-path salvage, raw-result merging, TPOT derivation, and OSL resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import benchmark_result as br


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- OSL resolution, through TPOT derivation --------------------------------
@pytest.mark.parametrize(
    "osl_fields,expected_tpot",
    [
        ({"osl": 10}, 10.0),
        ({"config": {"output_len": 10}}, 10.0),
        ({"request": {"max_tokens": 10}}, 10.0),
        ({"osl": 0, "output_len": -1}, None),
    ],
    ids=["osl", "config-output-len", "request-max-tokens", "non-positive-ignored"],
)
def test_tpot_is_derived_from_the_osl_wherever_the_report_records_it(osl_fields, expected_tpot):
    report = {
        "success": True,
        "throughput": {"output_throughput": 100.0, "completed_requests": 50},
        "latency": {"ttft": {"mean_ms": 10.0}, "e2el": {"mean_ms": 100.0}},
        **osl_fields,
    }
    assert br.extract_benchmark_measurement(report)["tpot_mean_ms"] == expected_tpot


# ---- _derive_tpot_if_missing ----------------------------------------------
@pytest.mark.parametrize(
    "measurement,report,expected",
    [
        ({"tpot_mean_ms": None, "e2el_mean_ms": 100.0, "ttft_mean_ms": 10.0}, {"osl": 10}, (100.0 - 10.0) / 9),
        ({"tpot_mean_ms": 5.0}, {}, 5.0),
        ({"tpot_mean_ms": None, "e2el_mean_ms": None, "ttft_mean_ms": None}, {}, None),
        ({"tpot_mean_ms": None, "e2el_mean_ms": 10.0, "ttft_mean_ms": 20.0}, {"osl": 10}, None),
        ({"tpot_mean_ms": None, "e2el_mean_ms": 100.0, "ttft_mean_ms": 10.0}, {"osl": 1}, None),
    ],
    ids=["derived", "already-set", "missing-latencies", "e2el-not-above-ttft", "osl-too-short"],
)
def test_derive_tpot_if_missing(measurement, report, expected):
    br._derive_tpot_if_missing(measurement, report)
    assert measurement["tpot_mean_ms"] == expected


# ---- is_valid_measurement -------------------------------------------------
def test_is_valid_measurement():
    assert br.is_valid_measurement(None) is False
    assert br.is_valid_measurement("nope") is False
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 5}) is True
    assert br.is_valid_measurement({"output_throughput": 0.0, "completed_requests": 5}) is False
    assert br.is_valid_measurement({"output_throughput": 10.0, "completed_requests": 0}) is False


# ---- _resolve_leak_roots --------------------------------------------------
def test_resolve_leak_roots_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", "/a:/b")
    roots = br._resolve_leak_roots()
    assert roots == (Path("/a"), Path("/b"))


# ---- _materialize_rescue_into_workspace -----------------------------------
def test_materialize_rescue_copy_error(tmp_path, monkeypatch):
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    src = _write_json(leak_dir / "inferencex_result.json", {"x": 1})
    monkeypatch.setattr(br.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert br._materialize_rescue_into_workspace(src, tmp_path / "ws") is None


# ---- _rescue_candidate_paths ----------------------------------------------
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


# ---- harvest_leaked_artifacts ---------------------------------------------
def test_harvest_skips_non_file(tmp_path, monkeypatch):
    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    (leak_root / "server.log").mkdir()
    dest = tmp_path / "dest"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(leak_root))
    out = br.harvest_leaked_artifacts(dest)
    assert out == []


def test_harvest_dest_mkdir_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(OSError("ro")))
    out = br.harvest_leaked_artifacts(tmp_path / "dest")
    assert out == []


def test_harvest_skips_already_under_dest(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "server.log").write_text("log", encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(dest))
    out = br.harvest_leaked_artifacts(dest)
    assert out == []


def test_harvest_copy_error(tmp_path, monkeypatch):
    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    (leak_root / "server.log").write_text("log", encoding="utf-8")
    monkeypatch.setattr(br.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(leak_root))
    out = br.harvest_leaked_artifacts(tmp_path / "dest")
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


def test_extract_rescue_skips_no_throughput(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(tmp_path / "inferencex_result.json", {"completed_requests": 5})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws, subprocess_started_unix=leak.stat().st_mtime - 100
    )
    assert m["valid_measurement"] is False


# ---- harvest_mn_gpu_metrics -----------------------------------------------
def test_harvest_mn_gpu_metrics_single_node_noop(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: False)
    assert br.harvest_mn_gpu_metrics(tmp_path) == {}


def test_harvest_mn_gpu_metrics_folds_pod_csvs(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(
        mn,
        "pd_topology_from_state",
        lambda: {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"]},
    )

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))

    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    (shared / "gpu_metrics_10.0.0.1.csv").write_text(f"{header}\n1000,55,300\n1001,56,310\n", encoding="utf-8")
    (shared / "gpu_metrics_10.0.0.2.csv").write_text(f"{header}\n1000,50,280\n", encoding="utf-8")

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


def test_harvest_mn_gpu_metrics_samples_name_their_metrics_and_aggregate_per_role(monkeypatch, tmp_path):
    """Each column lands on the sample field it names, a row with no numeric metric is no sample, and only samples of
    a pod with a role count towards that role's aggregate."""
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(
        mn,
        "pd_topology_from_state",
        lambda: {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"]},
    )
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    full = "timestamp,Temperature (Junction) (C),Average Socket Power (W),sclk clock (MHz),GPU use (%),VRAM% "
    (shared / "gpu_metrics_10.0.0.1.csv").write_text(
        f"{full}\n1000,50,100,1400,80,40\n1001,60,200,1500,90,50\n", encoding="utf-8"
    )
    (shared / "gpu_metrics_10.0.0.2.csv").write_text(
        "timestamp,Average Socket Power (W)\n1000,150\n1001,n/a\n", encoding="utf-8"
    )
    (shared / "gpu_metrics_10.0.0.3.csv").write_text("timestamp,Average Socket Power (W)\n1000,300\n", encoding="utf-8")
    dest = tmp_path / "ws"
    dest.mkdir()
    (dest / "benchmark_report.json").write_text("{}", encoding="utf-8")

    br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)

    report = json.loads((dest / "benchmark_report.json").read_text(encoding="utf-8"))
    first, second, decode, unassigned = report["gpu_monitor"]
    assert first == {
        "temperature_c": 50,
        "power_w": 100,
        "clock_mhz": 1400,
        "gpu_util_pct": 80,
        "vram_pct": 40,
        "role": "prefill",
    }
    assert (second["power_w"], second["temperature_c"], second["role"]) == (200, 60, "prefill")
    assert decode == {"power_w": 150, "role": "decode"}
    assert unassigned == {"power_w": 300}
    by_role = report["gpu_monitor_by_role"]
    assert set(by_role) == {"prefill", "decode"}
    assert (by_role["prefill"]["samples"], by_role["prefill"]["avg_power_w"], by_role["prefill"]["max_power_w"]) == (
        2,
        150.0,
        200.0,
    )
    assert by_role["prefill"]["max_temp_c"] == 60.0
    # No temperature sample for decode: the aggregate reads 0.0.
    assert (by_role["decode"]["samples"], by_role["decode"]["avg_temp_c"]) == (1, 0.0)


def test_harvest_mn_gpu_metrics_no_csvs(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    assert br.harvest_mn_gpu_metrics(tmp_path / "ws") == {}


# ---- is_valid_measurement, scriptable runs --------------------------------
def test_is_valid_measurement_scriptable_gate(monkeypatch):
    # Scriptable run: valid on positive throughput when the quality gate passes.
    assert br.is_valid_measurement({"quality_gate": {"passed": True}, "output_throughput": 5.0}) is True
    # Scriptable run whose quality gate fails is not selectable.
    monkeypatch.setattr(br, "quality_gate_passed", lambda qg, require=False: False, raising=False)
    import hyperloom.orchestrator.actions.executors._accuracy_gate as ag

    monkeypatch.setattr(ag, "quality_gate_passed", lambda qg, require=False: False)
    assert br.is_valid_measurement({"quality_gate": {"passed": False}, "output_throughput": 5.0}) is False


def test_harvest_mn_gpu_metrics_non_absolute_dir(monkeypatch, tmp_path):
    # Unresolved / relative shared dir is treated as absent.
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", "relative/path")
    assert br.harvest_mn_gpu_metrics(tmp_path / "ws") == {}


def test_harvest_mn_gpu_metrics_window_and_malformed_rows(monkeypatch, tmp_path):
    import hyperloom.orchestrator.actions.executors._multi_node_env as mn

    monkeypatch.setattr(mn, "is_multi_node", lambda: True)
    monkeypatch.setattr(mn, "pd_topology_from_state", dict)

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))

    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    # Rows: one in-window, one out-of-window (dropped), one blank, one bad ts.
    (shared / "gpu_metrics_hostA.csv").write_text(f"{header}\n1000,55,300\n5,50,280\n\n,51,290\n", encoding="utf-8")
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
    monkeypatch.setattr(mn, "pd_topology_from_state", dict)

    shared = tmp_path / "server_logs"
    shared.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(shared))
    header = "timestamp,Temperature (Junction) (C),Average Socket Power (W)"
    (shared / "gpu_metrics_hostA.csv").write_text(f"{header}\n1000,55,300\n", encoding="utf-8")

    dest = tmp_path / "ws"
    dest.mkdir()
    # Non-dict report JSON → inject branch bails without raising.
    (dest / "benchmark_report.json").write_text("[]", encoding="utf-8")
    out = br.harvest_mn_gpu_metrics(dest, subprocess_started_unix=900.0)
    assert out["rows"] == 1
    assert "gpu_monitor_samples" not in out
