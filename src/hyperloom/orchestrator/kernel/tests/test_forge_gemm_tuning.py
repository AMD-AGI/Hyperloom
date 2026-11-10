#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the forge gemm-tune command builder."""

from __future__ import annotations


from hyperloom.orchestrator.kernel import forge_gemm_tuning


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
        "timeout": 123,
        "global_timeout": 456,
        "untuned_csv": "/tmp/in.csv",
        "moe_untuned_csv": "/tmp/untuned_fmoe_from_runtime.csv",
        "shapes_json": "/tmp/shapes.json",
        "tunableop_input": "/tmp/tunable.txt",
        "kernel_signature_log": "/tmp/server.log",
        "tokens": "64,128",
        "thorough": True,
    }


def test_build_cmd_maps_all_options():
    cmd = forge_gemm_tuning.build_command(_payload())

    assert cmd[:5] == [forge_gemm_tuning.sys.executable, "-m", "kernelforge.cli", "gemm-tune", "run"]
    assert cmd[cmd.index("--model-path") + 1] == "/models/qwen"
    assert cmd[cmd.index("--framework") + 1] == "sglang"
    assert cmd[cmd.index("--precision") + 1] == "bf16"
    assert cmd[cmd.index("--quant-type") + 1] == "auto"
    assert cmd[cmd.index("--mp") + 1] == "8"
    assert cmd[cmd.index("--untuned-csv") + 1] == "/tmp/in.csv"
    assert cmd[cmd.index("--kernel-signature-log") + 1] == "/tmp/server.log"
    assert cmd[cmd.index("--tp") + 1] == "1"
    assert cmd[cmd.index("--conc") + 1] == "256"
    assert cmd[cmd.index("--timeout") + 1] == "123"
    assert cmd[cmd.index("--global-timeout") + 1] == "456"
    assert cmd[cmd.index("--tokens") + 1] == "64,128"
    assert "--skip-gpu-check" in cmd
    assert "--thorough" in cmd


def test_build_cmd_forwards_the_moe_untuned_csv():
    """The runtime-derived MoE key reaches forge only through this option."""
    cmd = forge_gemm_tuning.build_command(_payload())

    assert cmd[cmd.index("--moe-untuned-csv") + 1] == "/tmp/untuned_fmoe_from_runtime.csv"


def test_build_cmd_omits_the_moe_untuned_csv_when_absent():
    """No runtime key observed: forge must not receive an empty option."""
    payload = _payload()
    payload.pop("moe_untuned_csv")

    assert "--moe-untuned-csv" not in forge_gemm_tuning.build_command(payload)


def test_build_cmd_asserts_every_option_it_can_emit():
    """Meta-guard: an option added to _build_cmd must be asserted in this file."""
    emitted = {tok for tok in forge_gemm_tuning.build_command(_payload()) if tok.startswith("--")}
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
        "--timeout",
        "--global-timeout",
        "--untuned-csv",
        "--moe-untuned-csv",
        "--shapes-json",
        "--tunableop-input",
        "--kernel-signature-log",
        "--skip-gpu-check",
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
        }
    )
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_CURRENT_LIB", "aiter-1")
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_READ", "1")
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_ACCEPT_CANDIDATE", "1")
    monkeypatch.setenv("FORGE_GEMM_TUNE_KB_STRICT_LIB", "1")

    cmd = forge_gemm_tuning.build_command(payload)

    for retired in ("--kb-read", "--kb-accept-candidate", "--kb-strict-lib"):
        assert retired not in cmd
    # Which backend build produced the artifact is provenance, and still travels.
    assert cmd[cmd.index("--kb-current-lib") + 1] == "aiter-1"


def test_build_cmd_requires_mandatory_fields():
    payload = _payload()
    payload.pop("model_path")

    try:
        forge_gemm_tuning.build_command(payload)
    except ValueError as exc:
        assert "model_path is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
