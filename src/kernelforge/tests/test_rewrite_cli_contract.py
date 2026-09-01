"""Tests for the forge-rewrite-by-flydsl public CLI surface.

The capability handshake and the logical-op-name option are what a consumer
binds to before it can run anything, so they are exercised without a GPU, an
LLM, or a workspace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from kernelforge.cli import main
from kernelforge.rewrite_by_flydsl import protocol

from kernelforge.conftest import SRC_ROOT


def _rewrite_command():
    return main.commands["forge-rewrite-by-flydsl"]


def test_capabilities_query_short_circuits_the_required_options():
    # A bare capability query carries none of the five required options; it must
    # answer instead of failing with a usage error.
    result = CliRunner().invoke(main, ["forge-rewrite-by-flydsl", "--capabilities-json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == protocol.capabilities()


def test_capabilities_answer_over_a_real_subprocess():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kernelforge.cli",
            "forge-rewrite-by-flydsl",
            "--capabilities-json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(__file__).resolve().parent,
        env={
            **os.environ,
            "PYTHONPATH": (str(SRC_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
        },
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == protocol.capabilities()


def test_capabilities_query_needs_no_gpu_llm_or_workspace(monkeypatch):
    monkeypatch.delenv("GPU_TARGET", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("the capability query must not start a rewrite")

    monkeypatch.setattr("kernelforge.rewrite_by_flydsl.run_rewrite", fail)
    result = CliRunner().invoke(main, ["forge-rewrite-by-flydsl", "--capabilities-json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["rewrite_protocol_version"] == 2


def test_applyback_contract_query_uses_the_producer_schema():
    result = CliRunner().invoke(
        main,
        ["forge-rewrite-by-flydsl", "--applyback-contract-json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == protocol.applyback_contract_example()


def test_the_logical_op_name_and_its_deprecated_alias_are_one_option():
    parameters = {parameter.name: parameter for parameter in _rewrite_command().params}
    logical = parameters["op_name"]

    assert "--logical-op-name" in logical.opts
    assert "--op-name" in logical.opts
    assert logical.required is True


def _invoke_rewrite(monkeypatch, tmp_path, name_flag, extra_args=()):
    captured: dict = {}

    def fake_run_rewrite(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(
        "kernelforge.rewrite_by_flydsl.run_rewrite",
        fake_run_rewrite,
    )
    source = tmp_path / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    result = CliRunner().invoke(
        main,
        [
            "forge-rewrite-by-flydsl",
            "--source-kernel",
            str(source),
            "--driver",
            str(driver),
            name_flag,
            "vllm::softmax",
            "--workspace",
            str(tmp_path),
            "--experiments-dir",
            str(tmp_path / "exp"),
            *extra_args,
        ],
    )
    return result, captured


def test_the_deprecated_alias_still_selects_the_same_workload(monkeypatch, tmp_path):
    modern, from_modern = _invoke_rewrite(monkeypatch, tmp_path, "--logical-op-name")
    legacy, from_legacy = _invoke_rewrite(monkeypatch, tmp_path, "--op-name")

    assert modern.exit_code == 0
    assert legacy.exit_code == 0
    assert from_modern["op_name"] == "vllm::softmax"
    assert from_legacy["op_name"] == from_modern["op_name"]
    assert from_modern["prepare_driver"] is True
    assert from_modern["invocation_spec_file"] == ""
    assert from_modern["applyback_import_modules"] == ()
    assert from_modern["max_applyback_attempts"] == 2


def test_gpu_type_cli_override_reaches_rewrite_config(monkeypatch, tmp_path):
    result, captured = _invoke_rewrite(
        monkeypatch,
        tmp_path,
        "--logical-op-name",
        ("--gpu-type", "MI300X", "--gpu-target", "gfx950"),
    )

    assert result.exit_code == 0
    assert captured["config"].gpu_type == "mi300x"
    assert captured["config"].gpu_target == "gfx950"


def test_rewrite_rejects_an_unknown_option(monkeypatch, tmp_path):
    """An undeclared option is fatal here too, for the same reason as forge-loop.

    These two were the only tolerant entry points, granted the exemption because
    a consumer in another repository drove them. Vendoring removed that repository.
    """
    result, _captured = _invoke_rewrite(
        monkeypatch,
        tmp_path,
        "--logical-op-name",
        ("--e2e-pct", "3.2"),
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_rewrite_gpu_help_distinguishes_sku_from_architecture():
    result = CliRunner().invoke(main, ["forge-rewrite-by-flydsl", "--help"])

    assert result.exit_code == 0
    assert "--gpu-type" in result.output
    assert "--rewrite-kb" in result.output
    assert "--no-rewrite-kb" in result.output
    assert "Hardware SKU" in result.output
    assert "mi355x" in result.output
    assert "ROCm compilation architecture" in result.output
    assert "gfx950" in result.output
    assert "mi355x" in result.output


def test_default_rewrite_gpu_type_ignores_environment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("GPU_TYPE", "mi300x")

    result, captured = _invoke_rewrite(
        monkeypatch,
        tmp_path,
        "--logical-op-name",
    )

    assert result.exit_code == 0
    assert captured["rewrite_kb_enabled"] is True
    assert captured["config"].gpu_type == "mi355x"


def test_no_rewrite_kb_bypasses_remote_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)

    result, captured = _invoke_rewrite(
        monkeypatch,
        tmp_path,
        "--logical-op-name",
        ("--no-rewrite-kb",),
    )

    assert result.exit_code == 0
    assert captured["rewrite_kb_enabled"] is False


def test_driver_preparation_options_are_forwarded(monkeypatch, tmp_path):
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(
        "kernelforge.rewrite_by_flydsl.run_rewrite",
        capture,
    )
    source = tmp_path / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    invocation = tmp_path / "invocation.json"
    invocation.write_text('{"schema_version": 1}\n')
    result = CliRunner().invoke(
        main,
        [
            "forge-rewrite-by-flydsl",
            "--source-kernel",
            str(source),
            "--driver",
            str(tmp_path / "driver.py"),
            "--no-prepare-driver",
            "--invocation-spec-file",
            str(invocation),
            "--applyback-import-module",
            "sample.ops.softmax",
            "--max-applyback-attempts",
            "3",
            "--logical-op-name",
            "softmax",
            "--workspace",
            str(tmp_path),
            "--experiments-dir",
            str(tmp_path / "exp"),
        ],
    )

    assert result.exit_code == 0
    assert captured["prepare_driver"] is False
    assert captured["invocation_spec_file"] == str(invocation)
    assert captured["applyback_import_modules"] == ("sample.ops.softmax",)
    assert captured["max_applyback_attempts"] == 3


def test_a_framework_outside_the_handshake_is_rejected(monkeypatch, tmp_path):
    source = tmp_path / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")

    def fail(**kwargs):
        raise AssertionError("an unsupported framework must not start a rewrite")

    monkeypatch.setattr("kernelforge.rewrite_by_flydsl.run_rewrite", fail)
    result = CliRunner().invoke(
        main,
        [
            "forge-rewrite-by-flydsl",
            "--source-kernel",
            str(source),
            "--driver",
            str(driver),
            "--logical-op-name",
            "softmax",
            "--workspace",
            str(tmp_path),
            "--experiments-dir",
            str(tmp_path / "exp"),
            "--framework",
            "cuda",
        ],
    )

    assert result.exit_code != 0
    assert "unsupported framework" in result.output
    for framework in protocol.SUPPORTED_FRAMEWORKS:
        assert framework in result.output


@pytest.mark.parametrize("framework", protocol.SUPPORTED_FRAMEWORKS)
def test_an_advertised_framework_is_accepted(monkeypatch, tmp_path, framework):
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr("kernelforge.rewrite_by_flydsl.run_rewrite", capture)
    source = tmp_path / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('drive')\n")
    result = CliRunner().invoke(
        main,
        [
            "forge-rewrite-by-flydsl",
            "--source-kernel",
            str(source),
            "--driver",
            str(driver),
            "--logical-op-name",
            "softmax",
            "--workspace",
            str(tmp_path),
            "--experiments-dir",
            str(tmp_path / "exp"),
            "--framework",
            framework.upper(),
        ],
    )

    assert result.exit_code == 0
    assert captured["framework"] == framework


def test_the_deprecated_alias_warns_without_touching_the_result(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("sys.argv", ["kernelforge", "--op-name", "vllm::softmax"])
    result, _captured = _invoke_rewrite(monkeypatch, tmp_path, "--op-name")

    assert result.exit_code == 0
    assert "--op-name is deprecated" in result.output
    assert protocol.RESULT_SENTINEL not in result.output
