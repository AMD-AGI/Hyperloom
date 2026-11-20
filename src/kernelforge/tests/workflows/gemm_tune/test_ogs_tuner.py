# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A measured matmul_ogs result becomes one deployable, version-bound candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kernelforge.workflows.gemm_tune.candidates import per_tuner_candidates
from kernelforge.workflows.gemm_tune.model_analyzer import ModelProfile
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext
from kernelforge.workflows.gemm_tune.tuners import vllm_mxfp4_matmul_ogs as ogs


def _context(tmp_path: Path, *, architecture: str = "GptOssForCausalLM", gpu_type: str = "r9700") -> TuneContext:
    profile = ModelProfile(
        model_path="/models/gpt-oss-20b",
        architecture=architecture,
        is_moe=True,
        num_experts=32,
        num_experts_per_tok=4,
        hidden_size=2880,
        intermediate_size=2880,
        quant_method="mxfp4",
        model_dtype="bfloat16",
    )
    return TuneContext(
        profile=profile,
        framework="vllm",
        precision="mxfp4",
        quant_type="mxfp4",
        gpu_type=gpu_type,
        tp=1,
        conc=15,
        tokens=[1, 15, 512],
        mp=1,
        output_dir=tmp_path,
        iters=3,
        warmup=2,
        min_improvement_pct=1.0,
        timeout_s=1800,
    )


def test_tuner_writes_a_serving_hook_only_for_a_measured_winner(tmp_path, monkeypatch):
    tuner = ogs.VllmMxfp4MatmulOgsTuner(_context(tmp_path))

    def fake_sweep(command, *, timeout_s, log_file):
        config = json.loads(Path(command[-1]).read_text())
        row = {
            **config["cases"][0],
            "flags": config["search_space"][0],
            "default_ms": 2.0,
            "tuned_ms": 1.5,
            "speedup": 4 / 3,
        }
        (tuner.work_dir / "sweep_results.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "opt_flags_sha256": "a" * 64,
                    "gpu_name": "AMD Radeon AI PRO R9700",
                    "rows": [row],
                    "measurements": [{**row, "best": {"tuned_ms": 1.5}}],
                }
            )
        )
        return 0, "ok", ""

    monkeypatch.setattr(ogs, "run_subprocess", fake_sweep)
    result = tuner.execute()

    assert result.status == "partial_output"
    assert result.candidate is True
    assert result.env_var == "HYPERLOOM_MATMUL_OGS_FLAGS"
    assert Path(result.env_value).is_file()
    hook = tuner.work_dir / "runtime_sitecustomize"
    assert result.env_vars["PYTHONPATH"].split(os.pathsep)[0] == str(hook)
    assert (hook / "sitecustomize.py").is_file()
    assert (hook / "hyperloom_ogs_runtime.py").is_file()
    assert per_tuner_candidates([result])[0].env[result.env_var] == result.env_value


@pytest.mark.parametrize(
    ("architecture", "gpu_type"),
    [("OtherMoE", "r9700"), ("GptOssForCausalLM", "mi355x"), ("GptOssForCausalLM", "r9600d")],
)
def test_tuner_refuses_unqualified_target(tmp_path, architecture, gpu_type):
    tuner = ogs.VllmMxfp4MatmulOgsTuner(_context(tmp_path, architecture=architecture, gpu_type=gpu_type))
    result = tuner.execute()
    assert result.status == "failed"
    assert result.candidate is False
