# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy gate — GSM8K eval integration for inference_optimizer.

Protocol: baseline always runs GSM8K; high-risk variants too. Threshold is
``baseline_accuracy - new_accuracy <= 0.05`` (5% tolerance), REVERT otherwise.
High-risk = precision/compute-path changes (fp8/fp4/quant, aiter/triton rope,
enforce-eager/compilation-config); kernel patches handled by kernel-agent.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.05  # 5% allowed deviation

# Flags / env vars that indicate accuracy risk > 0; matching variants must
# pass the accuracy gate before promotion.
_HIGH_RISK_CLI_PATTERNS: tuple[str, ...] = (
    "--kv-cache-dtype",
    "--enforce-eager",
    "--compilation-config",
    "--attention-backend",
    "--decode-attention-backend",
)

_HIGH_RISK_ENV_KEYS: frozenset[str] = frozenset({
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_LINEAR",
    "VLLM_ROCM_USE_AITER_RMSNORM",
    "VLLM_ROCM_USE_AITER_FP8BMM",
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
    "VLLM_ROCM_USE_AITER_TRITON_ROPE",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT",
    "AMDGCN_USE_BUFFER_OPS",
    "SGLANG_USE_AITER",
})


def is_high_accuracy_risk(
    extra_args: str = "",
    extra_envs: dict[str, str] | None = None,
) -> bool:
    """Return True if the variant changes precision or compute paths.

    Args:
        extra_args (str): The variant's server args to scan for high-risk
            CLI flags.
        extra_envs (dict[str, str] | None): The variant's env overrides to scan
            for high-risk keys.

    Returns:
        bool: True when the variant matches any high-risk flag / env key.
    """
    args_lower = extra_args.lower()
    for pattern in _HIGH_RISK_CLI_PATTERNS:
        if pattern in args_lower:
            return True
    if extra_envs:
        if set(extra_envs.keys()) & _HIGH_RISK_ENV_KEYS:
            return True
    return False


def parse_eval_results(workspace: Path | str) -> dict[str, Any]:
    """Extract accuracy score from Magpie workspace's eval output.

    Searches ``results*.json`` recursively for the GSM8K-primary
    ``exact_match,strict-match`` metric.

    Args:
        workspace (Path | str): The benchmark workspace to search recursively
            for ``results*.json``.

    Returns:
        dict[str, Any]: ``{"accuracy": float, "task": str, "metric": str,
            "source_file": str}`` on success, or ``{"accuracy": None,
            "error": str}`` when no result / metric is found.
    """
    workspace = Path(workspace)
    search_paths = [
        workspace / "eval_*" / "**" / "results*.json",
        workspace / "**" / "results*.json",
    ]
    result_files: list[Path] = []
    for pattern in search_paths:
        result_files.extend(
            Path(f) for f in glob.glob(str(pattern), recursive=True)
        )
    if not result_files:
        return {"accuracy": None, "error": f"no results*.json in {workspace}"}

    latest = sorted(result_files)[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"accuracy": None, "error": f"parse error: {exc}"}

    results = data.get("results", {})
    for task_name, metrics in results.items():
        for key in ("exact_match,strict-match", "exact_match,flexible-extract",
                    "exact_match,none", "acc,none"):
            if key in metrics:
                score = metrics[key]
                if isinstance(score, (int, float)):
                    log.info("accuracy_gate: task=%s metric=%s score=%.4f "
                             "source=%s", task_name, key, score, latest)
                    return {
                        "accuracy": float(score),
                        "task": task_name,
                        "metric": key,
                        "source_file": str(latest),
                    }

    return {"accuracy": None, "error": f"no recognized metric in {latest}"}


def accuracy_passed(
    baseline_accuracy: float,
    new_accuracy: float,
    threshold: float = ACCURACY_THRESHOLD,
) -> bool:
    """Return True if accuracy drop is within tolerance.

    threshold=0.05 means: if baseline_accuracy=0.80, new must be >= 0.75.

    Args:
        baseline_accuracy (float): The baseline accuracy score. ``<= 0`` skips
            the gate (returns True).
        new_accuracy (float): The candidate variant's accuracy score.
        threshold (float): The maximum allowed accuracy drop.

    Returns:
        bool: True when the drop is within ``threshold`` (or no baseline).
    """
    if baseline_accuracy <= 0:
        # No baseline accuracy recorded; skip gate (can't compare).
        return True
    drop = baseline_accuracy - new_accuracy
    return drop <= threshold
