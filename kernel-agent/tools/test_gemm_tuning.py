# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the GEAK FP8 GEMM tuning wrapper (gemm_tuning.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent


def _load_module():
    """Load gemm_tuning.py as an isolated module without running main()."""
    spec = importlib.util.spec_from_file_location(
        "gemm_tuning_under_test", _TOOLS_DIR / "gemm_tuning.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


gt = _load_module()


def test_json_line_emits_single_sorted_line(capsys):
    gt._json_line({"b": 1, "a": 2})
    out = capsys.readouterr().out.strip()
    assert out == '{"a": 2, "b": 1}'


def test_safe_cleanup_clause_mentions_safety():
    clause = gt._safe_cleanup_clause()
    assert "SAFETY" in clause
    assert "killall" in clause


def test_build_task_with_known_baseline():
    args = gt._parse_args([
        "--benchmark-script", "/b.sh", "--tp", "2", "--conc", "8",
        "--isl", "128", "--osl", "256", "--model-path", "/m",
        "--framework", "sglang", "--gpu-type", "MI355X",
        "--precision", "fp8", "--baseline-tput", "123.5",
    ])
    task = gt._build_task(args, Path("/ws"))
    assert "123.5 tok/s" in task
    assert "TP=2, CONC=8, ISL=128, OSL=256" in task
    assert "/ws" in task


def test_build_task_without_baseline():
    args = gt._parse_args(["--benchmark-script", "/b.sh"])
    task = gt._build_task(args, Path("/ws"))
    assert "Run baseline at most once" in task


def test_latest_gemm_workspace_none_when_missing(tmp_path):
    assert gt._latest_gemm_workspace(tmp_path) is None


def test_latest_gemm_workspace_none_when_no_candidates(tmp_path):
    (tmp_path / "optimization_logs").mkdir()
    assert gt._latest_gemm_workspace(tmp_path) is None


def test_latest_gemm_workspace_picks_newest(tmp_path):
    base = tmp_path / "optimization_logs"
    base.mkdir()
    old = base / "gemm_tuning_1"
    new = base / "gemm_tuning_2"
    old.mkdir()
    new.mkdir()
    import os
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert gt._latest_gemm_workspace(tmp_path) == new


def test_load_report_none_workspace():
    assert gt._load_report(None) == {}


def test_load_report_missing_file(tmp_path):
    assert gt._load_report(tmp_path) == {}


def test_load_report_invalid_json(tmp_path):
    (tmp_path / "final_report.json").write_text("{not json", encoding="utf-8")
    out = gt._load_report(tmp_path)
    assert out["error"] == "final_report.json is not valid JSON"


def test_load_report_not_object(tmp_path):
    (tmp_path / "final_report.json").write_text("[1, 2]", encoding="utf-8")
    out = gt._load_report(tmp_path)
    assert out["error"] == "final_report.json is not a JSON object"


def test_load_report_valid(tmp_path):
    (tmp_path / "final_report.json").write_text('{"status": "ok"}', encoding="utf-8")
    out = gt._load_report(tmp_path)
    assert out["status"] == "ok"
    assert out["final_report_path"].endswith("final_report.json")


def test_apply_input_json_noop_when_empty():
    args = gt._parse_args([])
    assert gt._apply_input_json(args) is args


def test_apply_input_json_overlays(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"model-path": "/over", "tp": 4}), encoding="utf-8")
    args = gt._parse_args(["--input-json", str(p)])
    gt._apply_input_json(args)
    assert args.model_path == "/over"
    assert args.tp == 4


def test_apply_input_json_rejects_non_object(tmp_path):
    p = tmp_path / "in.json"
    p.write_text("[1]", encoding="utf-8")
    args = gt._parse_args(["--input-json", str(p)])
    with pytest.raises(ValueError):
        gt._apply_input_json(args)


def test_main_requires_cwd(capsys):
    # Non-empty argv (without --cwd) avoids the ``argv or sys.argv`` fallback.
    rc = gt.main(["--framework", "sglang"])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == "cwd_missing"


def test_main_requires_model_path(tmp_path, capsys):
    rc = gt.main(["--cwd", str(tmp_path)])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == "model_path_missing"


def test_main_requires_benchmark_script(tmp_path, capsys):
    rc = gt.main(["--cwd", str(tmp_path), "--model-path", "/m"])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == "benchmark_script_missing"


def test_main_dry_run(tmp_path, capsys):
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh", "--dry-run",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert (tmp_path / "gemm_tuning_task.txt").is_file()


def test_main_requires_config_when_not_dry_run(tmp_path, capsys):
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh",
    ])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["error_class"] == "geak_config_missing"


def _install_fake_minisweagent(monkeypatch, run_impl):
    """Inject a fake minisweagent.run.gemm_tuning module with run_impl."""
    pkg = types.ModuleType("minisweagent")
    run_pkg = types.ModuleType("minisweagent.run")
    gemm_mod = types.ModuleType("minisweagent.run.gemm_tuning")
    gemm_mod.run = run_impl
    pkg.run = run_pkg
    run_pkg.gemm_tuning = gemm_mod
    monkeypatch.setitem(sys.modules, "minisweagent", pkg)
    monkeypatch.setitem(sys.modules, "minisweagent.run", run_pkg)
    monkeypatch.setitem(sys.modules, "minisweagent.run.gemm_tuning", gemm_mod)


def test_main_success_keep(tmp_path, capsys, monkeypatch):
    def fake_run(**kwargs):
        cwd = kwargs["cwd"]
        ws = Path(cwd) / "optimization_logs" / "gemm_tuning_x"
        ws.mkdir(parents=True)
        (ws / "final_report.json").write_text(
            json.dumps({"status": "complete", "best_speedup": 1.5}), encoding="utf-8",
        )

    _install_fake_minisweagent(monkeypatch, fake_run)
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh", "--config", "/c.yaml",
    ])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    # report's own status ("complete") is spread last and wins the merge.
    assert out["status"] == "complete"
    assert out["decision"] == "KEEP"


def test_main_success_revert_when_no_speedup(tmp_path, capsys, monkeypatch):
    def fake_run(**kwargs):
        ws = Path(kwargs["cwd"]) / "optimization_logs" / "gemm_tuning_x"
        ws.mkdir(parents=True)
        (ws / "final_report.json").write_text(
            json.dumps({"status": "ok", "best_speedup": 0.9}), encoding="utf-8",
        )

    _install_fake_minisweagent(monkeypatch, fake_run)
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh", "--config", "/c.yaml",
    ])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["decision"] == "REVERT"


def test_main_report_missing(tmp_path, capsys, monkeypatch):
    _install_fake_minisweagent(monkeypatch, lambda **k: None)
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh", "--config", "/c.yaml",
    ])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["error_class"] == "final_report_missing"


def test_main_handles_run_exception(tmp_path, capsys, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    _install_fake_minisweagent(monkeypatch, boom)
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh", "--config", "/c.yaml",
    ])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["error_class"] == "RuntimeError"


def test_main_handles_systemexit(tmp_path, capsys, monkeypatch):
    def sysexit(**kwargs):
        ws = Path(kwargs["cwd"]) / "optimization_logs" / "gemm_tuning_x"
        ws.mkdir(parents=True)
        (ws / "final_report.json").write_text(
            json.dumps({"status": "failed"}), encoding="utf-8",
        )
        raise SystemExit(3)

    _install_fake_minisweagent(monkeypatch, sysexit)
    rc = gt.main([
        "--cwd", str(tmp_path), "--model-path", "/m",
        "--benchmark-script", "/b.sh", "--config", "/c.yaml",
    ])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["returncode"] == 3
