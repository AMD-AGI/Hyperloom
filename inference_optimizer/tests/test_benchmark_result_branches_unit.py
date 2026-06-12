# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for benchmark-result parsing: leak harvesting, rescue-path
salvage, raw-result merging, TPOT derivation, and OSL resolution."""
from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.action_executors import benchmark_result as br


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- _load_json -----------------------------------------------------------
def test_load_json(tmp_path):
    good = _write_json(tmp_path / "g.json", {"a": 1})
    assert br._load_json(good) == {"a": 1}
    bad = tmp_path / "b.json"
    bad.write_text("{not json", encoding="utf-8")
    assert br._load_json(bad) is None
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]", encoding="utf-8")
    assert br._load_json(arr) is None  # not a dict
    assert br._load_json(tmp_path / "missing.json") is None


# ---- coercion helpers -----------------------------------------------------
def test_to_float_int_first():
    assert br._to_float(True) is None
    assert br._to_float(None) is None
    assert br._to_float(object()) is None
    assert br._to_float("1.5") == 1.5
    assert br._to_int(True) is None
    assert br._to_int(object()) is None
    assert br._to_int("3") == 3
    assert br._first_float(None, "bad", object()) is None
    assert br._first_float("bad", 2.0) == 2.0
    assert br._first_int(None, "bad", 4) == 4


# ---- _candidate_raw_jsons ordering ----------------------------------------
def test_candidate_raw_jsons_ordering(tmp_path):
    (tmp_path / "profile_x.json").write_text("{}", encoding="utf-8")
    (tmp_path / "baseline.json").write_text("{}", encoding="utf-8")
    (tmp_path / "benchmark_report.json").write_text("{}", encoding="utf-8")
    out = br._candidate_raw_jsons(tmp_path)
    names = [p.name for p in out]
    assert "benchmark_report.json" not in names
    # baseline (non-profile) sorts before profile
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
    assert br.is_valid_measurement(
        {"output_throughput": 10.0, "completed_requests": 5}) is True
    assert br.is_valid_measurement(
        {"output_throughput": 0.0, "completed_requests": 5}) is False
    assert br.is_valid_measurement(
        {"output_throughput": 10.0, "completed_requests": 0}) is False


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
    # source already inside workspace -> None
    assert br._materialize_rescue_into_workspace(src, ws) is None


def test_materialize_rescue_copy_error(tmp_path, monkeypatch):
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    src = _write_json(leak_dir / "inferencex_result.json", {"x": 1})
    monkeypatch.setattr(br.shutil, "copy2",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert br._materialize_rescue_into_workspace(src, tmp_path / "ws") is None


# ---- _rescue_candidate_paths ----------------------------------------------
def test_rescue_candidate_paths_default(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", raising=False)
    # default /workspace path doesn't exist -> not a file -> dropped
    assert br._rescue_candidate_paths(tmp_path) == []


def test_rescue_candidate_paths_env_dir_and_file(tmp_path, monkeypatch):
    leak = tmp_path / "leaks"
    leak.mkdir()
    f = _write_json(leak / "inferencex_result_1.json", {"x": 1})
    standalone = _write_json(tmp_path / "extra.json", {"y": 2})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS",
                       f"{leak}:{standalone}")
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
    # subprocess started way after the leak's mtime -> stale -> dropped
    cands = br._rescue_candidate_paths(
        ws, subprocess_started_unix=leak.stat().st_mtime + 10000)
    assert cands == []


def test_rescue_candidate_paths_dedup_and_in_ws(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    inside = _write_json(ws / "inferencex_result.json", {"x": 1})
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS",
                       f"{inside}:{inside}")  # duplicate + inside ws
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
    # a directory matching the glob is not a file -> skipped
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
    out = br.harvest_leaked_artifacts(
        dest, leak_root=leak_root,
        subprocess_started_unix=f.stat().st_mtime + 10000)
    assert out == []


def test_harvest_dest_mkdir_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "mkdir",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("ro")))
    out = br.harvest_leaked_artifacts(tmp_path / "dest")
    assert out == []


def test_harvest_missing_root(tmp_path):
    out = br.harvest_leaked_artifacts(
        tmp_path / "dest", leak_root=tmp_path / "nonexistent")
    assert out == []


def test_harvest_skips_already_under_dest(tmp_path):
    # leak file lives directly under the destination -> nothing to harvest
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "server.log").write_text("log", encoding="utf-8")
    out = br.harvest_leaked_artifacts(dest, leak_root=dest)
    assert out == []


def test_harvest_copy_error(tmp_path, monkeypatch):
    leak_root = tmp_path / "workspace"
    leak_root.mkdir()
    (leak_root / "server.log").write_text("log", encoding="utf-8")
    monkeypatch.setattr(br.shutil, "copy2",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    out = br.harvest_leaked_artifacts(tmp_path / "dest", leak_root=leak_root)
    assert out == []


# ---- extract_benchmark_measurement ----------------------------------------
def test_extract_from_report_only():
    report = {
        "success": True,
        "framework": "sglang",
        "throughput": {"output_throughput": 100.0, "completed_requests": 50,
                       "request_throughput": 5.0, "duration_seconds": 10.0},
        "latency": {"ttft": {"mean_ms": 20.0}, "e2el": {"mean_ms": 200.0}},
    }
    m = br.extract_benchmark_measurement(report)
    assert m["valid_measurement"] is True
    assert m["output_throughput"] == 100.0


def test_extract_with_workspace_raw(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_json(ws / "result.json", {
        "output_throughput": 80.0, "completed_requests": 40,
    })
    report = {"success": False, "throughput": {}, "latency": {}}
    m = br.extract_benchmark_measurement(report, workspace=ws)
    assert m["valid_measurement"] is True
    assert "raw_inferencex_result_used" in m["nonfatal_warnings"]
    assert "benchmark_report_success_false" in m["nonfatal_warnings"]


def test_extract_skips_raw_without_throughput(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_json(ws / "result.json", {"completed_requests": 5})  # no tput
    m = br.extract_benchmark_measurement({"throughput": {}}, workspace=ws)
    assert m["valid_measurement"] is False


def test_extract_rescue_path(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(tmp_path / "inferencex_result.json", {
        "output_throughput": 90.0, "completed_requests": 45,
    })
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws,
        subprocess_started_unix=leak.stat().st_mtime - 100)
    assert m["valid_measurement"] is True
    assert any("rescued_from_leaked_path" in w for w in m["nonfatal_warnings"])


def test_extract_rescue_copy_failed_warning(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(tmp_path / "inferencex_result.json", {
        "output_throughput": 90.0, "completed_requests": 45,
    })
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    # copy into workspace fails -> recorded path falls back to the leak path
    monkeypatch.setattr(br, "_materialize_rescue_into_workspace",
                        lambda *a, **k: None)
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws,
        subprocess_started_unix=leak.stat().st_mtime - 100)
    assert m["valid_measurement"] is True
    assert any("rescued_copy_into_workspace_failed" in w
               for w in m["nonfatal_warnings"])


def test_extract_rescue_skips_no_throughput(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    leak = _write_json(tmp_path / "inferencex_result.json",
                       {"completed_requests": 5})  # no output_throughput
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
    m = br.extract_benchmark_measurement(
        {"throughput": {}}, workspace=ws,
        subprocess_started_unix=leak.stat().st_mtime - 100)
    assert m["valid_measurement"] is False
