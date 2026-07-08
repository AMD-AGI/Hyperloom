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

_HIGH_RISK_ENV_KEYS: frozenset[str] = frozenset(
    {
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
    }
)


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


# Recognize more metric keys + support sparse multi-task groups
# (tinyBenchmarks / metabench). Priority-ordered metric keys; the first key
# present on a task is the one used for that task. ``exact_match`` (generation)
# outranks ``acc`` / ``acc_norm`` (multiple-choice) so GSM8K-style tasks keep
# their historical metric; ``acc_norm,none`` is recognized so sparse
# multiple-choice subsets (tinyBenchmarks anchors, metabench) report a score
# instead of being skipped.
_RECOGNIZED_METRIC_KEYS: tuple[str, ...] = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "acc,none",
    "acc_norm,none",
)


def _pick_task_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    """Return ``(metric_key, score)`` for the first recognized numeric metric."""
    for key in _RECOGNIZED_METRIC_KEYS:
        if key in metrics:
            score = metrics[key]
            if isinstance(score, (int, float)):
                return key, float(score)
    return None


def parse_eval_results(workspace: Path | str) -> dict[str, Any]:
    """Extract an accuracy score from a Magpie workspace's eval output.

    Searches ``results*.json`` recursively and reads the first recognized
    metric (see :data:`_RECOGNIZED_METRIC_KEYS`) from each leaf task. A single
    task (e.g. plain ``gsm8k``) returns that task's score unchanged. When the
    run emits several leaf tasks — a sparse *group* such as ``tinyBenchmarks``
    or ``metabench`` — their scores are **averaged** into one number so the gate
    has a single value to compare. Group-aggregate rows (``group_subtasks``
    keys) are excluded so leaf tasks are never double-counted.

    Args:
        workspace (Path | str): The benchmark workspace to search recursively
            for ``results*.json``.

    Returns:
        dict[str, Any]: ``{"accuracy": float, "task": str, "metric": str,
            "tasks_used": list[str], "source_file": str}`` on success, or
            ``{"accuracy": None, "error": str}`` when no result / metric is
            found. ``task`` is the task name for a single task or a
            comma-joined list for several; ``metric`` is the shared metric key
            or ``"mixed"`` when leaf tasks used different metrics.
    """
    workspace = Path(workspace)
    search_paths = [
        workspace / "eval_*" / "**" / "results*.json",
        workspace / "**" / "results*.json",
    ]
    result_files: list[Path] = []
    for pattern in search_paths:
        result_files.extend(Path(f) for f in glob.glob(str(pattern), recursive=True))
    if not result_files:
        return {"accuracy": None, "error": f"no results*.json in {workspace}"}

    latest = sorted(result_files)[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"accuracy": None, "error": f"parse error: {exc}"}

    results = data.get("results", {})
    # Average across all recognized leaf tasks (for sparse groups like
    # tinyBenchmarks / metabench) instead of returning only the first single
    # task's exact_match. Group-aggregate rows (lm-eval reports a group alongside
    # its leaf subtasks) would double-count if averaged in, so drop any task that
    # owns subtasks.
    #
    # NOTE: this is a naive equal-weight arithmetic mean over leaf tasks -- every
    # task counts the same regardless of item count or difficulty, and there is
    # no IRT / difficulty normalization. All metrics are on a 0-1 scale so the
    # mean is well-formed, but treat the composite as an observability signal,
    # not a calibrated score. A single task is unaffected (mean of one == itself).
    group_subtasks = data.get("group_subtasks", {}) or {}
    group_names = {name for name, subs in group_subtasks.items() if subs}

    tasks: list[str] = []
    metrics_used: list[str] = []
    scores: list[float] = []
    for task_name, metrics in results.items():
        if task_name in group_names:
            continue
        picked = _pick_task_metric(metrics)
        if picked is None:
            continue
        metric_key, score = picked
        tasks.append(task_name)
        metrics_used.append(metric_key)
        scores.append(score)

    if not scores:
        return {"accuracy": None, "error": f"no recognized metric in {latest}"}

    accuracy = sum(scores) / len(scores)
    metric = metrics_used[0] if len(set(metrics_used)) == 1 else "mixed"
    task = tasks[0] if len(tasks) == 1 else ",".join(tasks)
    log.info(
        "accuracy_gate: tasks=%s metric=%s score=%.4f (n=%d) source=%s",
        task,
        metric,
        accuracy,
        len(scores),
        latest,
    )
    return {
        "accuracy": accuracy,
        "task": task,
        "metric": metric,
        "tasks_used": tasks,
        "source_file": str(latest),
    }


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
