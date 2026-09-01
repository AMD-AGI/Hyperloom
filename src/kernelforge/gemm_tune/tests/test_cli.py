# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI contract: GPU resolution, provenance, and unconditional live tuning."""

from __future__ import annotations

import json

from click.testing import CliRunner

from kernelforge.gemm_tune import cli as cli_mod
from kernelforge.gemm_tune.cli import gemm_tune
from kernelforge.gemm_tune.tuners.base import TuneResult


def _model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model


def _stub_preflight(monkeypatch):
    monkeypatch.setattr(
        "kernelforge.gemm_tune.aiter_preflight.collect",
        lambda: {"soft": [], "hard": [], "aligned": False},
    )


def test_help_carries_no_knowledge_base_options():
    """Tuning has no knowledge base, so no option may imply one exists."""
    result = CliRunner().invoke(gemm_tune, ["run", "--help"])

    assert result.exit_code == 0
    for option in ("--kb-read", "--kb-accept-candidate", "--kb-strict-lib"):
        assert option not in result.output


def test_every_runnable_tuner_is_executed(tmp_path, monkeypatch):
    """Nothing may stand between a runnable tuner and a real tuning run."""
    _stub_preflight(monkeypatch)
    model = _model_dir(tmp_path)
    output = tmp_path / "output"
    executed: list[str] = []

    class _Tuner:
        def __init__(self, name):
            self.name = name

        def execute(self):
            executed.append(self.name)
            return TuneResult(tuner_name=self.name, status="no_improvement")

    monkeypatch.setattr(cli_mod, "_create_tuner", lambda name, ctx: _Tuner(name))
    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(model),
            "--framework",
            "vllm",
            "--precision",
            "fp8",
            "--gpu-type",
            "mi300x",
            "--tuner",
            "a8w8",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert executed == ["a8w8"]
    report = json.loads((output / "result.json").read_text())
    assert [t["tuner"] for t in report["tuners_run"]] == ["a8w8"]
    assert all("kb_cache" not in t for t in report["tuners_run"])


def test_cli_resolves_auto_once_and_records_effective_gpu(tmp_path, monkeypatch):
    from kernelforge.gemm_tune import router

    _stub_preflight(monkeypatch)
    model = _model_dir(tmp_path)
    output = tmp_path / "output"
    calls = []

    def detect():
        calls.append(True)
        return "gfx942"

    monkeypatch.setattr(router, "_detect_local_gfx_arch", detect)
    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(model),
            "--framework",
            "vllm",
            "--precision",
            "bf16",
            "--gpu-type",
            "auto",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [True]
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "result.json").read_text())
    assert plan["gpu_type"] == "mi300x"
    assert report["gpu_type"] == "mi300x"
    assert '"gpu_type": "auto"' not in json.dumps([plan, report]).lower()


def test_cli_auto_detection_failure_aborts_before_model_or_tuning(tmp_path, monkeypatch):
    from kernelforge.gemm_tune import model_analyzer, router

    monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: "")
    monkeypatch.setattr(
        model_analyzer,
        "analyze_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model analysis must not start")),
    )
    output = tmp_path / "output"

    result = CliRunner().invoke(
        gemm_tune,
        [
            "run",
            "--model-path",
            str(tmp_path / "model"),
            "--framework",
            "vllm",
            "--precision",
            "bf16",
            "--gpu-type",
            "auto",
            "--skip-gpu-check",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "--gpu-type" in result.output
    assert not (output / "plan.json").exists()
