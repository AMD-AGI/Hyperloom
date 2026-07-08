#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the forge_gemm_tuning kernel-agent wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "forge_gemm_tuning.py"
_SPEC = importlib.util.spec_from_file_location("forge_gemm_tuning_tool", _MODULE_PATH)
assert _SPEC and _SPEC.loader
forge_gemm_tuning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(forge_gemm_tuning)


def _payload() -> dict:
    return {
        "model_path": "/models/qwen",
        "framework": "sglang",
        "precision": "bf16",
        "quant_type": "auto",
        "gpu_type": "mi300x",
        "tp": 1,
        "conc": 256,
        "mp": 8,
        "output_dir": "/tmp/out",
        "iters": 10,
        "warmup": 2,
        "min_improvement_pct": 3.0,
        "timeout": 123,
        "global_timeout": 456,
        "tuner": "fmoe_ck",
        "untuned_csv": "/tmp/in.csv",
        "shapes_json": "/tmp/shapes.json",
        "tunableop_input": "/tmp/tunable.txt",
        "kernel_signature_log": "/tmp/server.log",
        "gpu_ids": "0,1",
        "tokens": "64,128",
        "skip_gpu_check": True,
        "verbose": True,
        "thorough": True,
    }


def test_build_cmd_maps_all_options():
    cmd = forge_gemm_tuning._build_cmd(_payload())

    assert cmd[:4] == [forge_gemm_tuning.sys.executable, "-m", "forge_gemm_tune.cli", "run"]
    assert cmd[cmd.index("--model-path") + 1] == "/models/qwen"
    assert cmd[cmd.index("--framework") + 1] == "sglang"
    assert cmd[cmd.index("--precision") + 1] == "bf16"
    assert cmd[cmd.index("--quant-type") + 1] == "auto"
    assert cmd[cmd.index("--mp") + 1] == "8"
    assert cmd[cmd.index("--tuner") + 1] == "fmoe_ck"
    assert cmd[cmd.index("--tokens") + 1] == "64,128"
    assert "--skip-gpu-check" in cmd
    assert "--verbose" in cmd
    assert "--thorough" in cmd


def test_build_cmd_requires_mandatory_fields():
    payload = _payload()
    payload.pop("model_path")

    try:
        forge_gemm_tuning._build_cmd(payload)
    except ValueError as exc:
        assert "model_path is required" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("expected ValueError")


def test_load_input_json_rejects_non_object(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[]", encoding="utf-8")

    try:
        forge_gemm_tuning._load_input_json(str(p))
    except ValueError as exc:
        assert "must contain a JSON object" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("expected ValueError")


def test_main_runs_subprocess_and_relays_output(tmp_path, monkeypatch, capsys):
    p = tmp_path / "input.json"
    p.write_text(json.dumps(_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    class Proc:
        returncode = 0
        stdout = "OUT\n"
        stderr = "ERR\n"

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        return Proc()

    monkeypatch.setattr(forge_gemm_tuning.subprocess, "run", fake_run)

    rc = forge_gemm_tuning.main(["--input-json", str(p)])

    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "OUT\n"
    assert out.err == "ERR\n"
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert "--model-path" in captured["cmd"]


def test_main_reports_structured_input_error(capsys):
    rc = forge_gemm_tuning.main(["--input-json", "/does/not/exist.json"])

    out = capsys.readouterr()
    assert rc == 2
    payload = json.loads(out.out)
    assert payload["status"] == "failed"
    assert payload["micro_decision"] == "failed"
