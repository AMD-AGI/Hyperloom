# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tune the BF16 activation × MXFP4 weight matmul_ogs path used by GPT-OSS on RDNA4."""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hyperloom_common.gpu_identity import board_named_in

from . import ogs_runtime
from .base import BaseTuner, TuneResult
from .ogs_tuning import cases, flag_search_space
from .vllm_dense_tunableop import _candidate_pythonpath
from ..utils import run_subprocess


_SITECUSTOMIZE = """\
from hyperloom_ogs_runtime import chain_sitecustomize, install

try:
    install()
except Exception as exc:
    raise SystemExit(f"HL_MATMUL_OGS_HOOK_INVALID {type(exc).__name__}: {exc}") from exc
chain_sitecustomize(__file__)
"""
_TEXT_ENCODING = "utf-8"


def _write_hook(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(ogs_runtime.__file__), directory / "hyperloom_ogs_runtime.py")
    (directory / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding=_TEXT_ENCODING)


def _sweep_config(tuner: "VllmMxfp4MatmulOgsTuner") -> dict[str, Any]:
    ctx = tuner.ctx
    return {
        "cases": [asdict(case) for case in cases(ctx.profile, ctx.token_hint or ctx.tokens, conc=ctx.conc)],
        "search_space": flag_search_space(thorough=ctx.thorough),
        "experts": ctx.profile.num_experts,
        "top_k": ctx.profile.num_experts_per_tok,
        "gpu_id": ctx.gpu_ids.split(",")[0] if ctx.gpu_ids else "0",
        "warmup": max(2, min(8, int(ctx.warmup))),
        "iters": max(3, min(24, int(ctx.iters))),
        "min_improvement_pct": ctx.min_improvement_pct,
        "timeout_s": ctx.timeout_s,
    }


class VllmMxfp4MatmulOgsTuner(BaseTuner):
    """Run a measured, correctness-gated flag sweep and publish a startup hook."""

    name = "vllm_mxfp4_matmul_ogs"
    env_var = ogs_runtime.FLAGS_ENV

    def validate(self) -> str | None:
        profile = self.ctx.profile
        if self.ctx.framework != "vllm" or not profile.is_moe:
            return "matmul_ogs tuner requires vLLM MoE"
        if profile.architecture.lower() != "gptossforcausallm" or profile.quant_method.lower() != "mxfp4":
            return "matmul_ogs tuner is qualified only for GPT-OSS MXFP4"
        if profile.model_dtype.lower() not in ("bfloat16", "torch.bfloat16", "bf16"):
            return "matmul_ogs tuner requires BF16 activations"
        board = board_named_in(self.ctx.gpu_type)
        if board is None or board.product != "r9700":
            return "matmul_ogs tuner requires R9700"
        if self.ctx.tp != 1:
            return "matmul_ogs tuner requires tensor parallelism 1"
        if profile.num_experts <= 0 or not 0 < profile.num_experts_per_tok <= profile.num_experts:
            return "GPT-OSS MoE expert shape is unavailable"
        if not math.isfinite(self.ctx.min_improvement_pct) or self.ctx.min_improvement_pct < 0:
            return "matmul_ogs tuner requires a finite nonnegative improvement threshold"
        return None

    def run(self) -> TuneResult:
        config_path = self.work_dir / "sweep_config.json"
        config = _sweep_config(self)
        config_path.write_text(json.dumps(config, indent=2), encoding=_TEXT_ENCODING)
        rc, stdout, stderr = run_subprocess(
            [sys.executable, "-m", "kernelforge.workflows.gemm_tune.tuners.ogs_sweep", str(config_path)],
            timeout_s=self.ctx.timeout_s,
            log_file=self.work_dir / "tune.log",
        )
        result_path = self.work_dir / "sweep_results.json"
        if rc != 0 or not result_path.is_file():
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"matmul_ogs sweep failed (rc={rc}): {stderr[-500:]}",
                error_class="timeout" if rc == 124 else "subprocess_error",
            )
        measured = json.loads(result_path.read_text(encoding=_TEXT_ENCODING))
        if measured.get("status") not in ("ok", "no_improvement"):
            return TuneResult(tuner_name=self.name, status="failed", error="matmul_ogs sweep returned no verdict")
        return self._publish_measurement(config, measured)

    def _publish_measurement(self, config: dict[str, Any], measured: dict[str, Any]) -> TuneResult:
        """Publish only measured winners while retaining the full sweep record."""
        rows = measured.get("rows") or []
        measurements = measured.get("measurements") or []
        if not rows:
            return TuneResult(
                tuner_name=self.name,
                status="no_improvement" if len(measurements) == len(config["cases"]) else "empty_output",
                total_shapes=len(measurements),
                expected_shapes=len(config["cases"]),
                improved_shapes=0 if measurements else None,
                shape_results=measurements,
            )
        table = {
            "schema_version": 1,
            "gpu_target": "gfx1201",
            "opt_flags_sha256": measured["opt_flags_sha256"],
            "gpu_name": measured["gpu_name"],
            "experts": config["experts"],
            "top_k": config["top_k"],
            "rows": rows,
        }
        table_path = self.work_dir / "matmul_ogs_flags.json"
        table_path.write_text(json.dumps(table, indent=2), encoding=_TEXT_ENCODING)
        ogs_runtime._read_table(table_path)
        hook_dir = self.work_dir / "runtime_sitecustomize"
        _write_hook(hook_dir)
        speedups = [
            float(row["default_ms"]) / float(row["best"]["tuned_ms"]) for row in measurements if row.get("best")
        ]
        return TuneResult(
            tuner_name=self.name,
            status="ok" if len(measurements) == len(config["cases"]) else "partial_output",
            artifact_path=str(table_path),
            env_var=self.env_var,
            env_value=str(table_path),
            env_vars={"PYTHONPATH": _candidate_pythonpath(hook_dir)},
            candidate=True,
            expected_shapes=len(config["cases"]),
            total_shapes=len(measurements),
            improved_shapes=len(rows),
            best_micro_speedup=max(speedups),
            avg_micro_speedup=sum(speedups) / len(speedups),
            shape_results=measurements,
        )
