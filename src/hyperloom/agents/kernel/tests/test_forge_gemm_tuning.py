#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the forge_gemm_tuning kernel-agent wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hyperloom.orchestrator.kernel import request_handlers as krh


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
        "moe_untuned_csv": "/tmp/untuned_fmoe_from_runtime.csv",
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

    assert cmd[:5] == [forge_gemm_tuning.sys.executable, "-m", "kernelforge.cli", "gemm-tune", "run"]
    assert cmd[cmd.index("--model-path") + 1] == "/models/qwen"
    assert cmd[cmd.index("--framework") + 1] == "sglang"
    assert cmd[cmd.index("--precision") + 1] == "bf16"
    assert cmd[cmd.index("--quant-type") + 1] == "auto"
    assert cmd[cmd.index("--mp") + 1] == "8"
    assert cmd[cmd.index("--tuner") + 1] == "fmoe_ck"
    assert cmd[cmd.index("--untuned-csv") + 1] == "/tmp/in.csv"
    assert cmd[cmd.index("--kernel-signature-log") + 1] == "/tmp/server.log"
    assert cmd[cmd.index("--tp") + 1] == "1"
    assert cmd[cmd.index("--conc") + 1] == "256"
    assert cmd[cmd.index("--timeout") + 1] == "123"
    assert cmd[cmd.index("--global-timeout") + 1] == "456"
    assert cmd[cmd.index("--tokens") + 1] == "64,128"
    assert "--skip-gpu-check" in cmd
    assert "--verbose" in cmd
    assert "--thorough" in cmd


def test_preflight_and_inner_cli_use_the_same_interpreter():
    """The readiness probe must describe the command the wrapper will run."""
    probe = krh._forge_gemm_tune_probe_cmd()
    inner = forge_gemm_tuning._build_cmd(_payload())

    assert probe[:4] == inner[:4]
    assert probe[0] == forge_gemm_tuning.sys.executable


def test_build_cmd_forwards_the_moe_untuned_csv():
    """The runtime-derived MoE key reaches forge only through this option.

    The orchestrator derives the CSV from the dispatch tuple in the server log;
    without the option forge infers the key from the model config instead --
    the exact failure this lane exists to remove, and one that leaves no trace
    because the tuning still reports success.
    """
    cmd = forge_gemm_tuning._build_cmd(_payload())

    assert cmd[cmd.index("--moe-untuned-csv") + 1] == "/tmp/untuned_fmoe_from_runtime.csv"


def test_build_cmd_omits_the_moe_untuned_csv_when_absent():
    """No runtime key observed: forge must not receive an empty option."""
    payload = _payload()
    payload.pop("moe_untuned_csv")

    assert "--moe-untuned-csv" not in forge_gemm_tuning._build_cmd(payload)


def test_build_cmd_asserts_every_option_it_can_emit():
    """Meta-guard: an option added to _build_cmd must be asserted in this file.

    This file is the only guard on the agent-tool argv, and it had drifted to
    covering 10 of the options it emits -- which is how the MoE CSV option went
    unasserted while being the whole point of this lane. Comparing the emitted
    flags against a declared set makes the next omission fail here.
    """
    emitted = {tok for tok in forge_gemm_tuning._build_cmd(_payload()) if tok.startswith("--")}
    declared = {
        "--model-path",
        "--framework",
        "--precision",
        "--quant-type",
        "--gpu-type",
        "--tp",
        "--conc",
        "--mp",
        "--output-dir",
        "--iters",
        "--warmup",
        "--min-improvement-pct",
        "--timeout",
        "--global-timeout",
        "--tuner",
        "--untuned-csv",
        "--moe-untuned-csv",
        "--shapes-json",
        "--tunableop-input",
        "--kernel-signature-log",
        "--gpu-ids",
        "--skip-gpu-check",
        "--verbose",
        "--thorough",
        "--tokens",
        "--kb-current-lib",
    }
    assert emitted <= declared, f"option(s) not declared here: {sorted(emitted - declared)}"


def test_build_cmd_forwards_provenance_but_no_knowledge_base_options(monkeypatch):
    """Tuning has no knowledge base; asking it to consult one aborts the run."""
    payload = _payload()
    payload.update(
        {
            "kb_read": True,
            "kb_accept_candidate": True,
            "kb_strict_lib": True,
            "kb_current_lib": "aiter-1",
        }
    )
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_READ", "1")
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_ACCEPT_CANDIDATE", "1")
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_STRICT_LIB", "1")

    cmd = forge_gemm_tuning._build_cmd(payload)

    for retired in ("--kb-read", "--kb-accept-candidate", "--kb-strict-lib"):
        assert retired not in cmd
    # Which backend build produced the artifact is provenance, and still travels.
    assert cmd[cmd.index("--kb-current-lib") + 1] == "aiter-1"


def test_build_cmd_requires_mandatory_fields():
    payload = _payload()
    payload.pop("model_path")

    try:
        forge_gemm_tuning._build_cmd(payload)
    except ValueError as exc:
        assert "model_path is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_load_input_json_rejects_non_object(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[]", encoding="utf-8")

    try:
        forge_gemm_tuning._load_input_json(str(p))
    except ValueError as exc:
        assert "must contain a JSON object" in str(exc)
    else:  # pragma: no cover
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
