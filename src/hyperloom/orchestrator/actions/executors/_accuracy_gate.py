# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Accuracy gate — GSM8K eval integration for hyperloom.inference_optimizer.

Baseline always runs GSM8K; high-risk variants too. Threshold is
``baseline_accuracy - new_accuracy <= 0.05`` (5% tolerance), REVERT otherwise.
High-risk = precision/compute-path changes; kernel patches handled by
kernel-agent.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any



log = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.05  # allowed deviation

# Shared accuracy floor, used by BOTH the baseline eval-failure trigger and the
# enablement KEEP gate so the two never diverge.
#
# It is non-zero because the floor is the ONLY correctness authority on the
# enablement KEEP path. At 0.0 the gate degenerates to ``accuracy > 0``, which
# admits a model that is answering essentially nothing: a real run KEPT a
# candidate scoring gsm8k=0.00076 (0.08% of a 0.906 baseline) as "correct".
# 0.05 is a floor of last resort -- it rejects the collapsed-output regime
# without judging genuine quality.
DEFAULT_ENABLEMENT_ACCURACY_FLOOR = 0.05

# Enablement admission, selected by the ``--enablement`` CLI flag. ``launch``
# covers the boot-failure self-heal lane, ``eval`` the accuracy-failure lane.
# Default ``off`` means neither lane engages and a broken baseline fast-fails.
ENABLEMENT_MODE_OFF = "off"
ENABLEMENT_MODE_LAUNCH = "launch"
ENABLEMENT_MODE_EVAL = "eval"
ENABLEMENT_MODE_ALL = "all"
ENABLEMENT_MODES: tuple[str, ...] = (
    ENABLEMENT_MODE_OFF,
    ENABLEMENT_MODE_LAUNCH,
    ENABLEMENT_MODE_EVAL,
    ENABLEMENT_MODE_ALL,
)

# params.reason marking a baseline that re-anchors a stack the enablement
# specialist changed, rather than re-measuring the established one.
ENABLEMENT_REVALIDATION_REASON = "enablement_eval_revalidation"

# Result-dict keys stamped by the baseline executor on an eval-rooted failure and
# read by writeback promotion/persistence.
BASELINE_EVAL_FAILED_KEY = "baseline_eval_failed"
BASELINE_EVAL_FAILURE_KIND_KEY = "baseline_eval_failure_kind"
BASELINE_EVAL_OBSERVED_ACCURACY_KEY = "baseline_eval_observed_accuracy"
BASELINE_EVAL_ACCURACY_FLOOR_KEY = "baseline_eval_accuracy_floor"
BASELINE_EVAL_EVIDENCE_KEY = "baseline_eval_evidence"
BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY = "baseline_eval_contract_fingerprint"

# Distinct eval-failure kinds.
EVAL_KIND_RUNTIME_FAILURE = "eval_runtime_failure"
EVAL_KIND_ACCURACY_UNAVAILABLE = "accuracy_unavailable"
EVAL_KIND_ACCURACY_BELOW_FLOOR = "accuracy_below_floor"
# The model never emitted EOS, so the eval was cut short and scored ~0.
# Distinct from ``accuracy_below_floor``: a broken generation loop, not a model
# that answered and got them wrong.
EVAL_KIND_GENERATION_PATHOLOGY = "eval_generation_pathology"

# Sidecar the probe writes into ``$RESULT_DIR`` when it trips. Deliberately not
# ``results*.json``: :func:`parse_eval_results` globs that name for the score.
EVAL_PROBE_FILENAME = "hyperloom_eval_probe.json"

# stop_reason recorded when the baseline could not produce an accuracy result
# even though the accuracy test was expected to run. A broken baseline accuracy
# means the environment/config is fundamentally wrong, so the whole run halts
# rather than optimizing against an unvalidated baseline.
BASELINE_ACCURACY_STOP_REASON = "baseline_accuracy_failed"


def request_baseline_accuracy_stop(shared_state: Any, *, context: str) -> bool:
    """Halt the run when the baseline accuracy test produced no result.

    Only the baseline stops the run on a missing accuracy verdict: a baseline
    with no accuracy signal (eval failed or the scriptable quality gate is
    missing) indicates a fundamentally broken setup. Post-baseline variants that
    fail the accuracy gate are reverted instead (the offending change is
    dropped, the run continues).

    Args:
        shared_state: The live SharedState (``None`` in some unit contexts).
        context: Short audit tag identifying the call site.

    Returns:
        bool: ``True`` if a stop reason was recorded.
    """
    if shared_state is None:
        return False
    setter = getattr(shared_state, "set_stop_reason", None)
    if not callable(setter):
        return False
    log.warning(
        "baseline accuracy test produced no result (%s); stopping run (broken baseline setup)",
        context,
    )
    setter(BASELINE_ACCURACY_STOP_REASON)
    return True


def require_framework_accuracy_default() -> bool:
    """Default for the framework source-patch accuracy-KEEP gate.

    Source patches require the accuracy gate by default; opt out with
    ``INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY=0``.

    Returns:
        ``True`` unless the env var disables it.
    """
    v = os.environ.get("INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def resolve_enablement_mode(shared_state: Any) -> str:
    """Read the session's enablement mode.

    Falls back to ``off`` rather than to the SharedState default: a caller that
    cannot produce the field has not told us the operator opted in, and denying
    an authoring round is the recoverable direction.

    Args:
        shared_state: The live SharedState (``None`` in some executor contexts).

    Returns:
        One of :data:`ENABLEMENT_MODES`; missing or unknown values yield ``off``.
    """
    mode = str(getattr(shared_state, "enablement_mode", "") or "").strip().lower()
    return mode if mode in ENABLEMENT_MODES else ENABLEMENT_MODE_OFF


def launch_enablement_allowed(shared_state: Any) -> bool:
    """Whether a baseline that cannot launch may route into enablement."""
    return resolve_enablement_mode(shared_state) in (ENABLEMENT_MODE_LAUNCH, ENABLEMENT_MODE_ALL)


def eval_enablement_allowed(shared_state: Any) -> bool:
    """Whether a baseline accuracy-eval failure may route into enablement."""
    return resolve_enablement_mode(shared_state) in (ENABLEMENT_MODE_EVAL, ENABLEMENT_MODE_ALL)


def _finite_score(score: Any) -> float | None:
    """Return ``score`` as a finite float, or ``None`` when unusable."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    val = float(score)
    return val if math.isfinite(val) else None


def accuracy_meets_floor(score: Any, floor: float) -> bool:
    """True only when ``score`` is finite, strictly positive and ``>= floor``."""
    val = _finite_score(score)
    if val is None or val <= 0.0:
        return False
    return val >= floor


def classify_accuracy_failure(score: Any, floor: float) -> str | None:
    """Classify an accuracy verdict; ``None`` means it passes the floor.

    Missing / non-numeric / non-finite scores are ``accuracy_unavailable``;
    finite scores that are non-positive or below the floor are
    ``accuracy_below_floor``.
    """
    val = _finite_score(score)
    if val is None:
        return EVAL_KIND_ACCURACY_UNAVAILABLE
    if val <= 0.0 or val < floor:
        return EVAL_KIND_ACCURACY_BELOW_FLOOR
    return None


def _extract_eval_contract_fields(config_path: str | Path | None) -> dict[str, str]:
    """Extract stable eval-contract fields from a materialized Magpie YAML.

    Reads the fields that define what workload is evaluated and how eval is
    controlled.  Server args, runtime paths, lifecycle envs and any field
    that a server-arg tuning candidate is allowed to change are excluded so
    the fingerprint stays stable across valid enablement patches.

    Returns an empty dict when the config is absent or unreadable.
    """
    if not config_path:
        return {}
    try:
        import yaml as _yaml

        p = Path(config_path)
        if not p.is_file():
            return {}
        data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — best-effort
        return {}

    bench = data.get("benchmark") or {}
    envs: dict = bench.get("envs") or {}

    # Eval-contract keys in benchmark.envs; all others are excluded. The probe
    # knobs belong here: they change how early an eval is cut short.
    _EVAL_CONTRACT_ENV_KEYS = (
        "RUN_EVAL",
        "MAGPIE_EVAL_TASKS",
        "MAGPIE_EVAL_LIMIT",
        "HYPERLOOM_EVAL_PROBE",
        "HYPERLOOM_EVAL_PROBE_MIN_SAMPLES",
        "HYPERLOOM_EVAL_PROBE_LENGTH_RATIO",
    )
    # Workload-shape keys that define what is being measured.
    _WORKLOAD_SHAPE_ENV_KEYS = (
        "CONC",
        "ISL",
        "OSL",
        "MAX_MODEL_LEN",
        "TP",
        "RANDOM_RANGE_RATIO",
    )
    contract: dict[str, str] = {
        "framework": str(bench.get("framework") or ""),
        "model": str(bench.get("model") or ""),
        "benchmark_script": str(bench.get("benchmark_script") or ""),
        "precision": str(bench.get("precision") or ""),
    }
    for k in _EVAL_CONTRACT_ENV_KEYS + _WORKLOAD_SHAPE_ENV_KEYS:
        v = envs.get(k)
        if v is not None:
            contract[k] = str(v)
    return contract


def eval_contract_fingerprint(
    *,
    config_path: str | Path | None,
    framework: str | None = None,
    model: str | None = None,
    task: str | None = None,
    metric: str | None = None,
) -> str:
    """Short stable digest of the eval contract (workload + eval definition).

    Derives the digest from stable eval-contract inputs extracted from the
    materialized YAML (framework, model, script, precision, workload shape,
    eval controls).  Result-level outputs such as task/metric names are NOT
    included so an eval crash (where those are absent) produces the same
    fingerprint as a successful eval on the identical contract.

    ``task`` and ``metric`` parameters are accepted for call-site compatibility
    but are not included in the hash.

    Returns an empty string when the config cannot be read, signalling to
    callers that the contract is invalid and drift checking should fail closed.
    """
    contract = _extract_eval_contract_fields(config_path)
    if not contract:
        # Unreadable or missing config — return invalid sentinel.
        return ""
    # Supplement with caller-supplied framework/model when the YAML lacks them.
    if framework and not contract.get("framework"):
        contract["framework"] = str(framework)
    if model and not contract.get("model"):
        contract["model"] = str(model)
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:16]


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
    reports = [Path(f) for f in glob.glob(str(workspace / "**" / "benchmark_report.json"), recursive=True)]
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
    # Prefer a non-warmup round, but fall back to the warmup's eval rather than
    # reporting no accuracy at all.
    #
    # What a warmup discards is THROUGHPUT: the first benchmark window after a
    # cold boot pays one-time costs that would inflate later gains. Accuracy is
    # not timing-sensitive -- it is a property of the model and its config, so a
    # warmup-round eval measures exactly what a measured-round eval would. The
    # baseline double-run now deliberately evaluates only in the warmup round
    # (once, not twice), which makes that file the sole accuracy source; dropping
    # it unconditionally discarded a perfectly good score and stopped the run.
    #
    # The workspace-relative check keeps a parse rooted AT the warmup slot
    # finding its own output.
    discarded_warmup_dirs = {"warmup_round", "mn_warmup"}
    measured = [p for p in result_files if discarded_warmup_dirs.isdisjoint(p.relative_to(workspace).parts)]
    result_files = measured or result_files
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


def read_eval_probe(workspace: Path | str) -> dict[str, Any] | None:
    """Read the generation-pathology probe sidecar, when the probe tripped.

    The probe only writes :data:`EVAL_PROBE_FILENAME` when it cuts an eval
    short, so ``None`` means the ordinary thing happened: the model terminated
    its answers, or the InferenceX patch never applied. Searched recursively
    because the baseline double-run evaluates in the warmup round, whose
    ``$RESULT_DIR`` nests under the task workspace.

    Args:
        workspace (Path | str): Benchmark workspace to search recursively.

    Returns:
        dict[str, Any] | None: The probe record stamped with ``kind`` and
        ``source_file``, or ``None`` when no readable sidecar exists.
    """
    matches = sorted(Path(workspace).rglob(EVAL_PROBE_FILENAME))
    if not matches:
        return None
    latest = matches[-1]
    try:
        record = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    record["kind"] = EVAL_KIND_GENERATION_PATHOLOGY
    record["source_file"] = str(latest)
    return record


def eval_probe_summary(probe: dict[str, Any] | None) -> str:
    """Render a one-line summary of a tripped probe for a log / journal reason.

    Args:
        probe (dict[str, Any] | None): A :func:`read_eval_probe` record.

    Returns:
        str: The summary, or ``""`` when ``probe`` is empty.
    """
    if not probe:
        return ""
    return (
        f"{EVAL_KIND_GENERATION_PATHOLOGY}: {probe.get('finish_reason_length', 0)}/"
        f"{probe.get('observed_samples', 0)} sampled responses hit the max_tokens cap "
        f"(up to {probe.get('max_completion_tokens_seen', 0)} tokens); the model never "
        "emitted EOS, so the eval was cut short and scored ~0"
    )


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
    "BASELINE_ACCURACY_STOP_REASON",
    "BASELINE_EVAL_ACCURACY_FLOOR_KEY",
    "BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY",
    "BASELINE_EVAL_EVIDENCE_KEY",
    "BASELINE_EVAL_FAILED_KEY",
    "BASELINE_EVAL_FAILURE_KIND_KEY",
    "BASELINE_EVAL_OBSERVED_ACCURACY_KEY",
    "DEFAULT_ENABLEMENT_ACCURACY_FLOOR",
    "ENABLEMENT_MODES",
    "ENABLEMENT_MODE_ALL",
    "ENABLEMENT_REVALIDATION_REASON",
    "ENABLEMENT_MODE_EVAL",
    "ENABLEMENT_MODE_LAUNCH",
    "ENABLEMENT_MODE_OFF",
    "EVAL_KIND_ACCURACY_BELOW_FLOOR",
    "EVAL_KIND_ACCURACY_UNAVAILABLE",
    "EVAL_KIND_GENERATION_PATHOLOGY",
    "EVAL_KIND_RUNTIME_FAILURE",
    "EVAL_PROBE_FILENAME",
    "_extract_eval_contract_fields",
    "accuracy_keep_block",
    "accuracy_meets_floor",
    "accuracy_passed",
    "classify_accuracy_failure",
    "eval_contract_fingerprint",
    "eval_enablement_allowed",
    "eval_probe_summary",
    "is_high_accuracy_risk",
    "launch_enablement_allowed",
    "parse_eval_results",
    "read_eval_probe",
    "request_baseline_accuracy_stop",
    "resolve_enablement_mode",
    "require_framework_accuracy_default",
]
