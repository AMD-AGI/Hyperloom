# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""vLLM MoE Triton fused_moe deterministic tile parameter sweep.

Codified from GEAK's successful approach: sweep BLOCK_SIZE_M/N/K, GROUP_SIZE_M,
num_warps, num_stages, waves_per_eu, SPLIT_K for each batch size, benchmark
each config, and pick the best.

Output: JSON config folder compatible with VLLM_TUNED_CONFIG_FOLDER.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any

from .base import BaseTuner, TuneResult
from ..utils import TUNER_ENV_VARS, run_subprocess
from ..shapes import compute_vllm_moe_batch_sizes

log = logging.getLogger(__name__)

# Known-good configs carried over from GEAK results. They stay first in every
# search space so a truncated run still measures the configs we already trust.
_SEED_CONFIGS: list[dict[str, int]] = [
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "SPLIT_K": 1,
    },
    # Measured winners on the BK=256 axis. Widening --thorough alone did not
    # deliver them: Hyperloom only asks for thorough at session_max_min >= 1440
    # and mp >= 4, so almost every session runs the default list and would still
    # never see this axis. The generated space is what found them; these three
    # are here so the default search can reach them too.
    #
    # DeepSeek-V4-Flash-bf16 (E=256, topk=6, K=4096, N=2048): best at M=32, 256
    # and 1024, worth 1.0975x-1.1235x. Independently on Mixtral-8x7B (E=8,
    # N=14336) the M=1 and M=32 winners were also BK=256, and the seed list
    # above kept only 1 of 4 shapes there (avg 0.949x, i.e. a regression) while
    # a space containing BK=256 kept 3 of 4.
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "SPLIT_K": 1,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "SPLIT_K": 1,
    },
]

# BLOCK_SIZE_K=256 is why this is a generated space rather than a fixed list.
# Measured on DeepSeek-V4-Flash-bf16 (E=256, topk=6, K=4096, N=2048, vLLM
# 0.27.1, timing invoke_fused_moe_triton_kernel directly): the best config at
# M=32, 256 and 1024 used BK=256 every single time, worth 1.0975x-1.1235x over
# the eight seeds above. Those seeds top out at BK=128, so that axis was never
# searched -- and a fixed list cannot be wrong about a value it never contains.
# The three winners now also sit in _SEED_CONFIGS so the default search reaches
# them; the grid stays because the axis matters beyond those three points.
_AXES: dict[str, tuple[int, ...]] = {
    "BLOCK_SIZE_M": (16, 32, 64, 128),
    "BLOCK_SIZE_N": (64, 128, 256),
    "BLOCK_SIZE_K": (64, 128, 256),
    "GROUP_SIZE_M": (1, 4, 8),
    "num_warps": (4, 8),
    "num_stages": (2,),
    "waves_per_eu": (0, 2),
}

# The measurement above sampled 160 points of this grid; keep that as the
# default budget so --thorough reproduces a search we have evidence for.
# Override for a wider sweep at the cost of machine time.
_THOROUGH_CAP_ENV = "FORGE_MOE_TRITON_MAX_CONFIGS"
_DEFAULT_THOROUGH_CAP = 160


def _grid_configs() -> list[dict[str, int]]:
    """Full cross product of the tile axes, in a stable order."""
    names = list(_AXES)
    return [
        dict(zip(names, values, strict=True), SPLIT_K=1) for values in itertools.product(*(_AXES[n] for n in names))
    ]


def _thorough_cap() -> int:
    raw = os.environ.get(_THOROUGH_CAP_ENV, "").strip()
    try:
        cap = int(raw)
    except ValueError:
        return _DEFAULT_THOROUGH_CAP
    return cap if cap > 0 else _DEFAULT_THOROUGH_CAP


def build_search_space(thorough: bool) -> list[dict[str, int]]:
    """Configs to sweep.

    ``--thorough`` used to be inert here: the caller passed the fixed list in
    both modes, so asking for a thorough search changed nothing. It now widens
    the space for real, seeds first so a capped run keeps the trusted configs.

    Invalid combinations are not filtered out — the sweep script already times
    each config in isolation and skips the ones that fail to compile or run, so
    guessing hardware limits here would only risk excluding a winner.
    """
    if not thorough:
        return [dict(c) for c in _SEED_CONFIGS]

    seen = {tuple(sorted(c.items())) for c in _SEED_CONFIGS}
    space = [dict(c) for c in _SEED_CONFIGS]
    for cfg in _grid_configs():
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            space.append(cfg)
    # A cap below the seed count would truncate inside the seed prefix, which
    # contradicts the reason the seeds are first: a capped run is supposed to
    # keep the configs already measured to work and give up only the generated
    # ones. Thorough therefore never searches less than fast does.
    return space[: max(_thorough_cap(), len(_SEED_CONFIGS))]


def _generate_sweep_script(
    work_dir: Path,
    profile: Any,
    batch_sizes: list[int],
    iters: int,
    warmup: int,
    gpu_id: str,
    configs: list[dict[str, int]],
) -> Path:
    """Generate a standalone Python script that performs the Triton MoE sweep.

    This script imports vllm's fused_moe internals, runs each config, and
    writes results to a JSON file.
    """
    script_path = work_dir / "vllm_moe_sweep.py"
    config_path = work_dir / "sweep_config.json"

    num_experts = profile.num_experts
    inter_size = profile.effective_moe_intermediate
    hidden_size = profile.hidden_size
    topk = profile.num_experts_per_tok

    # Write config to file (avoids f-string injection of paths/values)
    sweep_config = {
        "gpu_id": gpu_id,
        "num_experts": num_experts,
        "intermediate_size": inter_size,
        "hidden_size": hidden_size,
        "topk": topk,
        "batch_sizes": batch_sizes,
        "iters": iters,
        "warmup": warmup,
        "configs": configs,
        "output_path": str(work_dir / "sweep_results.json"),
    }
    config_path.write_text(json.dumps(sweep_config, indent=2), encoding="utf-8")

    script_content = f'''#!/usr/bin/env python3
"""Auto-generated vLLM MoE Triton sweep script."""

import json
import os
import time
import sys
import inspect
from pathlib import Path

# Load config from file (avoids path injection issues)
_config = json.loads(Path("{config_path}").read_text())

os.environ.setdefault("CUDA_VISIBLE_DEVICES", _config["gpu_id"])
os.environ.setdefault("HIP_VISIBLE_DEVICES", _config["gpu_id"])

import torch

VLLM_AVAILABLE = False
USE_CONTEXT_MANAGER = False
_override_config_fn = None

try:
    from vllm.model_executor.layers.fused_moe import fused_experts
    VLLM_AVAILABLE = True
except ImportError:
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        VLLM_AVAILABLE = True
    except ImportError:
        pass

if VLLM_AVAILABLE:
    sig = inspect.signature(fused_experts)
    params = list(sig.parameters.keys())
    if "override_config" in params:
        USE_CONTEXT_MANAGER = False
        print("API mode: override_config kwarg", file=sys.stderr)
    else:
        try:
            from vllm.model_executor.layers.fused_moe import override_config as _oc_fn
            _override_config_fn = _oc_fn
            USE_CONTEXT_MANAGER = True
            print("API mode: override_config context manager", file=sys.stderr)
        except ImportError:
            alt_names = ["config", "triton_config", "kernel_config"]
            for alt in alt_names:
                if alt in params:
                    print(f"API mode: alt kwarg '{{alt}}'", file=sys.stderr)
                    break
            else:
                print(f"fused_experts params: {{params}}", file=sys.stderr)
                VLLM_AVAILABLE = False

NUM_EXPERTS = _config["num_experts"]
INTERMEDIATE_SIZE = _config["intermediate_size"]
HIDDEN_SIZE = _config["hidden_size"]
TOPK = _config["topk"]
BATCH_SIZES = _config["batch_sizes"]
ITERS = _config["iters"]
WARMUP = _config["warmup"]

CONFIGS = _config["configs"]


def _call_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, config):
    """Call fused_experts with the appropriate API for this vLLM version."""
    if USE_CONTEXT_MANAGER:
        with _override_config_fn(config):
            return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    else:
        return fused_experts(
            hidden_states, w1, w2, topk_weights, topk_ids,
            override_config=config,
        )


def _call_fused_experts_default(hidden_states, w1, w2, topk_weights, topk_ids):
    """Call fused_experts with vLLM's default auto-tuned config (baseline)."""
    return fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)


def _create_tensors(M, dtype=torch.bfloat16):
    device = "cuda"
    hidden_states = torch.randn(M, HIDDEN_SIZE, dtype=dtype, device=device)
    w1 = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=dtype, device=device)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=dtype, device=device)
    gating_output = torch.randn(M, NUM_EXPERTS, dtype=torch.float32, device=device)
    topk_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, TOPK, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, w1, w2, topk_weights, topk_ids


def benchmark_baseline(M, dtype=torch.bfloat16):
    """Benchmark with vLLM default config (no override) as baseline."""
    if not VLLM_AVAILABLE:
        return float("inf")
    hidden_states, w1, w2, topk_weights, topk_ids = _create_tensors(M, dtype)
    for _ in range(WARMUP):
        try:
            _call_fused_experts_default(hidden_states, w1, w2, topk_weights, topk_ids)
        except Exception as e:
            print(f"  [baseline warmup error M={{M}}] {{type(e).__name__}}: {{e}}", file=sys.stderr)
            return float("inf")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(ITERS):
        _call_fused_experts_default(hidden_states, w1, w2, topk_weights, topk_ids)
    torch.cuda.synchronize()
    return (time.time() - start) / ITERS * 1e6


def benchmark_config(M, config, dtype=torch.bfloat16):
    """Benchmark a single Triton config for fused MoE at batch size M."""
    if not VLLM_AVAILABLE:
        return float("inf")
    hidden_states, w1, w2, topk_weights, topk_ids = _create_tensors(M, dtype)
    for _ in range(WARMUP):
        try:
            _call_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, config)
        except Exception as e:
            print(f"  [warmup error M={{M}}] {{type(e).__name__}}: {{e}}", file=sys.stderr)
            return float("inf")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(ITERS):
        _call_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, config)
    torch.cuda.synchronize()
    return (time.time() - start) / ITERS * 1e6


def main():
    if not VLLM_AVAILABLE:
        result = {{"error": "vllm fused_experts not available or no compatible config API found", "status": "unsupported_vllm_version"}}
        print(json.dumps(result))
        return 1

    print(f"vLLM fused_experts API detected, context_manager={{USE_CONTEXT_MANAGER}}", file=sys.stderr)

    results = {{}}
    shape_details = []
    errors = []

    for M in BATCH_SIZES:
        baseline_time = benchmark_baseline(M)
        print(f"M={{M}}: baseline={{baseline_time:.1f}}us", file=sys.stderr)

        best_time = float("inf")
        best_config = None

        for config in CONFIGS:
            try:
                elapsed = benchmark_config(M, config)
                if elapsed < best_time:
                    best_time = elapsed
                    best_config = dict(config)
            except Exception as e:
                if not errors:
                    errors.append(f"M={{M}}: {{type(e).__name__}}: {{e}}")
                continue

        if best_config is not None:
            speedup = baseline_time / best_time if best_time > 0 else 1.0
            shape_details.append({{
                "M": M,
                "baseline_us": round(baseline_time, 2),
                "tuned_us": round(best_time, 2),
                "speedup": round(speedup, 4),
            }})
            if speedup > 1.0:
                results[str(M)] = best_config
                print(f"M={{M}}: best={{best_time:.1f}}us speedup={{speedup:.3f}}x KEEP config={{best_config}}", file=sys.stderr)
            else:
                print(f"M={{M}}: best={{best_time:.1f}}us speedup={{speedup:.3f}}x SKIP (not faster than default)", file=sys.stderr)
        else:
            if not errors:
                errors.append(f"M={{M}}: all configs returned inf")

        torch.cuda.empty_cache()

    output_path = _config["output_path"]
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    details_path = _config["output_path"].replace("sweep_results.json", "shape_details.json")
    with open(details_path, "w") as f:
        json.dump(shape_details, f, indent=2)

    speedups = [d["speedup"] for d in shape_details if d["speedup"] > 0]
    best_speedup = max(speedups) if speedups else 1.0
    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0

    out = {{
        "status": "ok",
        "output": output_path,
        "batch_sizes": len(results),
        "shape_details": shape_details,
        "best_speedup": round(best_speedup, 4),
        "avg_speedup": round(avg_speedup, 4),
    }}
    if errors:
        out["errors"] = errors[:5]
    if len(results) == 0 and errors:
        out["status"] = "unsupported_vllm_version"
        out["error"] = errors[0] if errors else "all configs failed"
    print(json.dumps(out))
    return 0 if len(results) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
'''
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


class VllmMoeTritonTuner(BaseTuner):
    """Deterministic Triton tile parameter sweep for vLLM fused MoE kernels."""

    name = "vllm_moe_triton"
    env_var = TUNER_ENV_VARS["vllm_moe_triton"]

    def validate(self) -> str | None:
        if not self.ctx.profile.is_moe:
            return "Model is not MoE"
        if self.ctx.profile.num_experts < 1:
            return "num_experts < 1"
        return None

    def run(self) -> TuneResult:
        profile = self.ctx.profile
        batch_sizes = compute_vllm_moe_batch_sizes(
            conc=self.ctx.conc,
            explicit_tokens=self.ctx.tokens if self.ctx.tokens else None,
        )

        gpu_id = self.ctx.gpu_ids.split(",")[0] if self.ctx.gpu_ids else "0"

        configs = build_search_space(self.ctx.thorough)
        log.info(
            "MoE Triton sweep: %d configs x %d batch sizes (thorough=%s)",
            len(configs),
            len(batch_sizes),
            self.ctx.thorough,
        )

        # Generate and run the sweep script
        script = _generate_sweep_script(
            work_dir=self.work_dir,
            profile=profile,
            batch_sizes=batch_sizes,
            iters=self.ctx.iters,
            warmup=self.ctx.warmup,
            gpu_id=gpu_id,
            configs=configs,
        )

        rc, stdout, stderr = run_subprocess(
            ["python3", str(script)],
            timeout_s=self.ctx.timeout_s,
            log_file=self.work_dir / "tune.log",
        )

        if rc == 124:
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"Sweep timed out after {self.ctx.timeout_s}s",
                error_class="timeout",
            )

        # Parse JSON output from the script
        try:
            result_line = stdout.strip().splitlines()[-1] if stdout.strip() else "{}"
            script_result = json.loads(result_line)
        except (json.JSONDecodeError, IndexError):
            script_result = {}

        if script_result.get("status") == "unsupported_vllm_version":
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error="vLLM not available or incompatible version",
                error_class="unsupported_vllm_version",
            )

        if rc != 0 or script_result.get("status") != "ok":
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"Sweep failed (rc={rc}): {stderr[-500:]}",
                error_class="subprocess_error",
            )

        # Build tuned config folder (VLLM_TUNED_CONFIG_FOLDER format)
        sweep_results_path = self.work_dir / "sweep_results.json"
        if not sweep_results_path.is_file():
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error="sweep_results.json not produced",
                error_class="output_missing",
            )

        tuned_configs_dir = self.work_dir / "tuned_configs"
        tuned_configs_dir.mkdir(exist_ok=True)

        sweep_data = json.loads(sweep_results_path.read_text(encoding="utf-8"))

        # Write config file in vLLM expected format:
        # E=<experts>,N=<N>,device_name=<gpu>,dtype=<dtype>.json
        E = profile.num_experts
        N = profile.effective_moe_intermediate
        gpu_name = self.ctx.gpu_type.replace(" ", "_").upper()
        if "mi300" in gpu_name.lower():
            gpu_name = "AMD_Instinct_MI300X"
        elif "mi355" in gpu_name.lower():
            gpu_name = "AMD_Instinct_MI355X"

        # Determine dtype string
        dtype_str = "bfloat16"
        if self.ctx.precision == "fp8":
            dtype_str = "fp8_w8a8"
        elif "awq" in self.ctx.quant_type or "gptq" in self.ctx.quant_type:
            dtype_str = "int8_w8a16"

        config_filename = f"E={E},N={N},device_name={gpu_name},dtype={dtype_str}.json"
        config_path = tuned_configs_dir / config_filename
        config_path.write_text(json.dumps(sweep_data, indent=4), encoding="utf-8")

        # Extract speedup metrics from sweep script output
        shape_details = script_result.get("shape_details", [])
        best_speedup = script_result.get("best_speedup", 1.0)
        avg_speedup = script_result.get("avg_speedup", 1.0)
        min_pct = self.ctx.min_improvement_pct / 100.0 if self.ctx.min_improvement_pct else 0.0
        improved_count = sum(1 for d in shape_details if d.get("speedup", 1.0) > 1.0 + min_pct)

        n_tuned = len(sweep_data)
        return TuneResult(
            tuner_name=self.name,
            status="ok" if n_tuned > 0 else "no_improvement",
            artifact_path=str(tuned_configs_dir),
            env_var=self.env_var,
            env_value=str(tuned_configs_dir),
            total_shapes=len(batch_sizes),
            improved_shapes=improved_count,
            best_micro_speedup=best_speedup,
            avg_micro_speedup=avg_speedup,
            shape_results=shape_details,
        )
