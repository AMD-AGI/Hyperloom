# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy gate — GSM8K eval integration for hyperloom.inference_optimizer.

Baseline always runs GSM8K; high-risk variants too. Threshold is
``baseline_accuracy - new_accuracy <= 0.05`` (5% tolerance), REVERT otherwise.
High-risk = precision/compute-path changes; kernel patches handled by
kernel-agent.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.05  # allowed deviation


def require_framework_accuracy_default() -> bool:
    """Default for the framework source-patch accuracy-KEEP gate.

    Source patches require the accuracy gate by default; opt out with
    ``INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY=0``.

    Returns:
        ``True`` unless the env var disables it.
    """
    v = os.environ.get("INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def accuracy_keep_block(
    accuracy_pass: bool | None,
    *,
    required: bool,
    baseline_accuracy: Any,
) -> tuple[bool, str, bool]:
    """Decide whether the accuracy gate blocks a KEEP.

    A measured regression always blocks. When the gate is ``required`` but
    produced no verdict (``None``): block iff a positive baseline accuracy was
    available (eval should have run but didn't); otherwise *degrade* (allow
    throughput-only KEEP) so eval-less runs are not universally blocked.

    Args:
        accuracy_pass: The gate verdict (``True`` pass / ``False`` regression /
            ``None`` not evaluated).
        required: Whether the accuracy gate is mandatory for this KEEP.
        baseline_accuracy: The baseline accuracy the gate compared against.

    Returns:
        ``(blocked, reason, degraded)``: whether to block the KEEP, an audit
        reason, and whether enforcement degraded to throughput-only.
    """
    if accuracy_pass is False:
        return True, "accuracy regression detected", False
    if accuracy_pass is True:
        return False, "", False
    # accuracy_pass is None: no verdict.
    if not required:
        return False, "", False
    try:
        base = float(baseline_accuracy)
    except (TypeError, ValueError):
        base = 0.0
    if base > 0:
        return (
            True,
            "accuracy gate required but produced no eval result (RUN_EVAL/baseline accuracy missing)",
            False,
        )
    return False, "", True

# Flags indicating accuracy risk; matching variants must pass the gate.
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


def parse_quality_gate(workspace: Path | str) -> dict[str, Any]:
    """Read a scriptable (server-less) quality gate from the bench report.

    Scriptable workloads (e.g. xDiT diffusion) cannot run a GSM8K eval; their
    bench script computes an image-quality gate (LPIPS/SSIM/MSE vs a fixed
    reference) embedded in ``benchmark_report.json`` as a ``quality_gate``
    block. This reads the most recent such block in ``workspace``.

    Args:
        workspace (Path | str): The benchmark workspace to search recursively
            for ``benchmark_report.json``.

    Returns:
        dict[str, Any]: ``{"quality_gate": dict, "source_file": str}`` on
            success, or ``{"quality_gate": None, "error": str}`` otherwise.
    """
    workspace = Path(workspace)
    reports = [
        Path(f)
        for f in glob.glob(str(workspace / "**" / "benchmark_report.json"), recursive=True)
    ]
    if not reports:
        return {"quality_gate": None, "error": f"no benchmark_report.json in {workspace}"}
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"quality_gate": None, "error": f"parse error: {exc}"}
    qg = data.get("quality_gate")
    if not isinstance(qg, dict):
        return {"quality_gate": None, "error": f"no quality_gate in {latest}"}
    return {"quality_gate": qg, "source_file": str(latest)}


def quality_gate_passed(
    quality_gate: dict[str, Any] | None,
    require: bool = False,
) -> bool:
    """Return whether a scriptable quality gate passed.

    Prefers the explicit ``passed`` flag the bench script emits; falls back to
    evaluating any present thresholds (``lpips <= lpips_max``,
    ``ssim >= ssim_min``, ``mse <= mse_max``).

    Args:
        quality_gate (dict[str, Any] | None): The quality-gate block.
        require (bool): When ``True`` (scriptable workloads, where the gate is
            the only correctness signal) a missing/empty gate fails the gate
            (fail-closed). When ``False`` (serving) a missing/empty gate does
            not block (parity with the no-baseline accuracy skip).

    Returns:
        bool: ``True`` when the gate passes (or is absent and not required).
    """
    if not isinstance(quality_gate, dict) or not quality_gate:
        return not require
    # A SKIPPED gate carries no correctness signal. For scriptable workloads
    # (require=True) the image-quality gate is the ONLY correctness signal, so a
    # skip must not silently pass.
    if require and quality_gate.get("skipped"):
        # Baseline establishing the reference on its first run has nothing to
        # compare against yet -> pass.
        if str(quality_gate.get("reason") or "") == "reference_established":
            return True
        # Any other skip means the only correctness signal never ran -> fail
        # closed rather than trusting an unchecked speedup.
        return False
    if "passed" in quality_gate:
        return bool(quality_gate["passed"])
    checks = (
        ("lpips", "lpips_max", lambda v, lim: v <= lim),
        ("ssim", "ssim_min", lambda v, lim: v >= lim),
        ("mse", "mse_max", lambda v, lim: v <= lim),
    )
    evaluated = 0
    for metric_key, limit_key, ok in checks:
        val = quality_gate.get(metric_key)
        lim = quality_gate.get(limit_key)
        if isinstance(val, (int, float)) and isinstance(lim, (int, float)):
            evaluated += 1
            if not ok(float(val), float(lim)):
                return False
    # A required gate with neither ``passed`` nor any usable threshold pair is
    # ambiguous; treat it as a failure (fail-closed).
    if require and evaluated == 0:
        return False
    return True


def parse_eval_results(
    workspace: Path | str,
    framework: str | None = None,
) -> dict[str, Any]:
    """Extract accuracy score from Magpie workspace's eval output.

    Scriptable (server-less) workloads take precedence: when a
    ``benchmark_report.json`` carries a ``quality_gate`` block, that gate is
    mapped onto the accuracy contract (``1.0`` pass / ``0.0`` fail). Otherwise
    this searches ``results*.json`` recursively for the GSM8K-primary
    ``exact_match,strict-match`` metric. For scriptable frameworks the
    image-quality gate is the only correctness signal, so a missing/invalid
    gate fails closed (``accuracy=0.0``).

    Args:
        workspace (Path | str): The benchmark workspace to search recursively
            for ``benchmark_report.json`` / ``results*.json``.
        framework (str | None): Framework name, used to decide whether the
            quality gate is required. Defaults to serving semantics.

    Returns:
        dict[str, Any]: ``{"accuracy": float, "task": str, "metric": str,
            "source_file": str}`` on success, or ``{"accuracy": None,
            "error": str}`` when no result / metric is found.
    """
    workspace = Path(workspace)

    from hyperloom.inference_optimizer import framework_registry

    scriptable = framework_registry.is_scriptable(framework)

    # Scriptable quality gate first: map passed->1.0 / fail->0.0.
    qg_out = parse_quality_gate(workspace)
    if qg_out.get("quality_gate") is not None:
        passed = quality_gate_passed(qg_out["quality_gate"], require=scriptable)
        log.info(
            "accuracy_gate: quality_gate passed=%s source=%s",
            passed,
            qg_out.get("source_file"),
        )
        return {
            "accuracy": 1.0 if passed else 0.0,
            "task": "quality_gate",
            "metric": "quality_gate_passed",
            "quality_gate": qg_out["quality_gate"],
            "source_file": qg_out.get("source_file"),
        }

    # Scriptable workloads require the gate: a missing/invalid one fails closed.
    if scriptable:
        log.warning(
            "accuracy_gate: scriptable framework=%s but no quality_gate found: %s",
            framework,
            qg_out.get("error", "unknown"),
        )
        return {
            "accuracy": 0.0,
            "task": "quality_gate",
            "metric": "quality_gate_passed",
            "quality_gate": None,
            "error": qg_out.get("error", "no quality_gate"),
        }

    search_paths = [
        workspace / "eval_*" / "**" / "results*.json",
        workspace / "**" / "results*.json",
    ]
    result_files: list[Path] = []
    for pattern in search_paths:
        result_files.extend(Path(f) for f in glob.glob(str(pattern), recursive=True))
    # Never grade a discarded warmup round: grid/baseline warmups nest throwaway
    # eval output under a named warmup slot. When the search root sits above such
    # a slot, the recursive glob would otherwise also match the discarded eval.
    # Drop nested warmup results using a workspace-relative check so a parse
    # rooted AT the warmup slot itself still finds its own output.
    discarded_warmup_dirs = {"warmup_round", "mn_warmup"}
    result_files = [
        p
        for p in result_files
        if discarded_warmup_dirs.isdisjoint(p.relative_to(workspace).parts)
    ]
    if not result_files:
        return {"accuracy": None, "error": f"no results*.json in {workspace}"}

    latest = sorted(result_files)[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"accuracy": None, "error": f"parse error: {exc}"}

    results = data.get("results", {})
    for task_name, metrics in results.items():
        for key in ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc,none"):
            if key in metrics:
                score = metrics[key]
                if isinstance(score, (int, float)):
                    log.info("accuracy_gate: task=%s metric=%s score=%.4f source=%s", task_name, key, score, latest)
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
        # No baseline recorded; skip gate.
        return True
    drop = baseline_accuracy - new_accuracy
    return drop <= threshold


__all__ = [
    "ACCURACY_THRESHOLD",
    "accuracy_keep_block",
    "accuracy_passed",
    "is_high_accuracy_risk",
    "parse_eval_results",
    "require_framework_accuracy_default",
]
