# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Accuracy gate — GSM8K eval integration for hyperloom.inference_optimizer."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.io import safe_mtime
from hyperloom.common.perf_metric import is_agentx_mode

log = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.05  # allowed deviation

# Upstream rejects an AgentX submission whose error rate over completed requests exceeds 10% (InferenceX
# ``validate_agentic_result.py``). A percentage because that is aiperf's unit: ``RequestErrorRateMetric`` is declared
# PERCENT and derives ``100.0 * errors / total``.
AGENTX_ERROR_RATE_THRESHOLD_PCT = 10.0

# Shared accuracy floor, used by BOTH the baseline eval-failure trigger and the enablement KEEP gate so the two never
# diverge.
DEFAULT_ENABLEMENT_ACCURACY_FLOOR = 0.5

# Enablement admission, selected by the ``--enablement`` CLI flag.
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

# params.reason marking a baseline that re-anchors a stack the enablement specialist changed, rather than re-measuring
# the established one.
ENABLEMENT_REVALIDATION_REASON = "enablement_eval_revalidation"

# Result-dict keys stamped by the baseline executor on an eval-rooted failure and read by writeback
# promotion/persistence.
BASELINE_EVAL_FAILED_KEY = "baseline_eval_failed"
BASELINE_EVAL_FAILURE_KIND_KEY = "baseline_eval_failure_kind"
BASELINE_EVAL_OBSERVED_ACCURACY_KEY = "baseline_eval_observed_accuracy"
BASELINE_EVAL_ACCURACY_FLOOR_KEY = "baseline_eval_accuracy_floor"
BASELINE_EVAL_EVIDENCE_KEY = "baseline_eval_evidence"
BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY = "baseline_eval_contract_fingerprint"

# Distinct eval-failure kinds.
EVAL_KIND_RUNTIME_FAILURE = "eval_runtime_failure"

# The serving configuration cannot answer an eval request at all, so no verdict can ever be produced under it.
EVAL_KIND_CONTEXT_TOO_SMALL = "eval_context_too_small"

# Smallest prompt an eval task is assumed to send.
_MIN_EVAL_PROMPT_TOKENS = 256
EVAL_KIND_ACCURACY_UNAVAILABLE = "accuracy_unavailable"
EVAL_KIND_ACCURACY_BELOW_FLOOR = "accuracy_below_floor"
# The model never emitted EOS, so the eval was cut short and scored ~0.
EVAL_KIND_GENERATION_PATHOLOGY = "eval_generation_pathology"

# Sidecar the probe writes into ``$RESULT_DIR`` when it trips.
EVAL_PROBE_FILENAME = "hyperloom_eval_probe.json"

# stop_reason recorded when the baseline could not produce an accuracy result even though the accuracy test was
# expected to run.
BASELINE_ACCURACY_STOP_REASON = "baseline_accuracy_failed"

# Truthy-false spellings that disable the accuracy gate.
_RUN_EVAL_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def materialized_run_eval_disabled(config_path: Path | str) -> bool:
    """Report whether lm-eval is disabled in the materialized benchmark config."""
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    envs = ((cfg.get("benchmark") or {}).get("envs")) or {}
    val = envs.get("RUN_EVAL")
    return val is not None and str(val).strip().lower() in _RUN_EVAL_FALSE_VALUES


def request_baseline_accuracy_stop(shared_state: Any, *, context: str) -> bool:
    """Halt the run when the baseline accuracy test produced no result."""
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
    """Default for the framework source-patch accuracy-KEEP gate."""
    v = os.environ.get("INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def require_kernel_accuracy_default() -> bool:
    """Default for the kernel-patch accuracy-KEEP gate."""
    v = os.environ.get("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def resolve_enablement_mode(shared_state: Any) -> str:
    """Read the session's enablement mode."""
    mode = str(getattr(shared_state, "enablement_mode", "") or "").strip().lower()
    return mode if mode in ENABLEMENT_MODES else ENABLEMENT_MODE_OFF


def launch_enablement_allowed(shared_state: Any) -> bool:
    """Whether a baseline that cannot launch may route into enablement."""
    return resolve_enablement_mode(shared_state) in (ENABLEMENT_MODE_LAUNCH, ENABLEMENT_MODE_ALL)


def eval_enablement_allowed(shared_state: Any) -> bool:
    """Whether a baseline accuracy-eval failure may route into enablement."""
    if bool(getattr(shared_state, "eval_disabled", False)):
        return False
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
    """Classify an accuracy verdict; ``None`` means it passes the floor."""
    val = _finite_score(score)
    if val is None:
        return EVAL_KIND_ACCURACY_UNAVAILABLE
    if val <= 0.0 or val < floor:
        return EVAL_KIND_ACCURACY_BELOW_FLOOR
    return None


def _extract_eval_contract_fields(config_path: str | Path | None) -> dict[str, str]:
    """Extract stable eval-contract fields from a materialized Magpie YAML."""
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

    # Eval-contract keys in benchmark.envs; all others are excluded.
    _EVAL_CONTRACT_ENV_KEYS = (
        "RUN_EVAL",
        "MAGPIE_EVAL_TASKS",
        "MAGPIE_EVAL_LIMIT",
        "HYPERLOOM_EVAL_PROBE",
        "HYPERLOOM_EVAL_PROBE_MIN_SAMPLES",
        "HYPERLOOM_EVAL_PROBE_LENGTH_RATIO",
        "HYPERLOOM_EVAL_MAX_TOKENS",
        "HYPERLOOM_EVAL_DERIVE_STOP",
        "HYPERLOOM_EVAL_STOP_STRINGS",
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
    """Short stable digest of the eval contract (workload + eval definition)."""
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


def resolve_served_context(
    *,
    server_args: str | None,
    env_max_model_len: Any = 0,
) -> int:
    """Resolve the context length the server was actually started with."""
    raw = str(server_args or "")
    if raw:
        try:
            toks = shlex.split(raw)
        except ValueError:
            toks = raw.split()
        flag = "--max-model-len"
        prefix = flag + "="
        for i, tok in enumerate(toks):
            value = None
            if tok == flag and i + 1 < len(toks):
                value = toks[i + 1]
            elif tok.startswith(prefix):
                value = tok[len(prefix) :]
            if value is not None:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    break
                if parsed > 0:
                    return parsed
                break
    try:
        return max(0, int(env_max_model_len or 0))
    except (TypeError, ValueError):
        return 0


def served_context_hosts_eval(
    *,
    served_max_model_len: Any,
    eval_max_tokens: Any,
) -> tuple[bool, str]:
    """Whether the served context can hold an eval prompt plus its completion."""
    try:
        ctx = int(served_max_model_len or 0)
    except (TypeError, ValueError):
        ctx = 0
    try:
        gen = int(eval_max_tokens or 0)
    except (TypeError, ValueError):
        gen = 0
    if ctx <= 0 or gen <= 0:
        return True, ""
    room = ctx - gen
    if room >= _MIN_EVAL_PROMPT_TOKENS:
        return True, ""
    return False, (
        f"served --max-model-len {ctx} cannot host the eval: {gen} completion "
        f"tokens leave {room} for the prompt, below the {_MIN_EVAL_PROMPT_TOKENS} "
        "token minimum, so every request is rejected before it reaches the model"
    )


def accuracy_keep_block(
    accuracy_pass: bool | None,
    *,
    required: bool,
    baseline_accuracy: Any,
) -> tuple[bool, str, bool]:
    """Decide whether the accuracy gate blocks a KEEP."""
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


# There is deliberately no "high accuracy risk" predicate here any more.


def parse_quality_gate(workspace: Path | str) -> dict[str, Any]:
    """Read a scriptable (server-less) quality gate from the bench report."""
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


def parse_agentx_error_rate(workspace: Path | str) -> float | None:
    """Read ``request_error_rate`` from the newest ``inferencex_result.json``; None when no result reported one."""
    # None rather than 0.0 so an export without the field is incomparable instead of a perfect score.
    workspace = Path(workspace)
    results = [Path(f) for f in glob.glob(str(workspace / "**" / "inferencex_result.json"), recursive=True)]
    if not results:
        return None
    latest = max(results, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("accuracy_gate: unreadable %s: %s", latest, exc)
        return None
    rate = data.get("request_error_rate")
    return float(rate) if isinstance(rate, (int, float)) else None


def quality_gate_passed(
    quality_gate: dict[str, Any] | None,
    require: bool = False,
) -> bool:
    """Return whether a scriptable quality gate passed."""
    if not isinstance(quality_gate, dict) or not quality_gate:
        return not require
    # A SKIPPED gate carries no correctness signal.
    if require and quality_gate.get("skipped"):
        # Baseline establishing the reference on its first run has nothing to compare against yet -> pass.
        if str(quality_gate.get("reason") or "") == "reference_established":
            return True
        # Any other skip means the only correctness signal never ran -> fail closed rather than trusting an unchecked
        # speedup.
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
    # A required gate with neither ``passed`` nor any usable threshold pair is ambiguous; treat it as a failure
    # (fail-closed).
    if require and evaluated == 0:
        return False
    return True


def parse_eval_results(
    workspace: Path | str,
    framework: str | None = None,
    benchmark_mode: str = "",
) -> dict[str, Any]:
    """Extract accuracy score from Magpie workspace's eval output; AgentX grades on its error rate instead."""
    workspace = Path(workspace)

    from hyperloom.inference_optimizer import framework_registry

    scriptable = framework_registry.is_scriptable(framework)

    # AgentX has no lm-eval; its correctness signal is the error rate upstream
    # gates a submission on. Fails closed like the scriptable gate below: a run
    # that reported no rate is not comparable, and treating that as a pass is
    # how an incomparable measurement reaches the leaderboard set.
    if is_agentx_mode(benchmark_mode):
        rate = parse_agentx_error_rate(workspace)
        passed = rate is not None and rate <= AGENTX_ERROR_RATE_THRESHOLD_PCT
        log.info("accuracy_gate: agentx request_error_rate=%s passed=%s", rate, passed)
        return {
            "accuracy": 1.0 if passed else 0.0,
            "task": "agentx_error_rate",
            "metric": "request_error_rate",
            "error_rate": rate,
        }

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
    # Prefer a non-warmup round, but fall back to the warmup's eval rather than reporting no accuracy at all.
    discarded_warmup_dirs = {"warmup_round", "mn_warmup"}
    measured = [p for p in result_files if discarded_warmup_dirs.isdisjoint(p.relative_to(workspace).parts)]
    result_files = measured or result_files
    if not result_files:
        return {"accuracy": None, "error": f"no results*.json in {workspace}"}

    latest = max(result_files, key=safe_mtime)
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
    """Read the generation-pathology probe sidecar, when the probe tripped."""
    try:
        matches = list(Path(workspace).rglob(EVAL_PROBE_FILENAME))
    except OSError:
        # A sibling directory under the search root can be removed by a parallel task while the recursive walk is in
        # flight; an unscannable tree yields no probe verdict rather than an error.
        return None
    if not matches:
        return None
    try:
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        record = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    record["kind"] = EVAL_KIND_GENERATION_PATHOLOGY
    record["source_file"] = str(latest)
    return record


def eval_probe_summary(probe: dict[str, Any] | None) -> str:
    """Render a one-line summary of a tripped probe for a log / journal reason."""
    if not probe:
        return ""
    return (
        f"{EVAL_KIND_GENERATION_PATHOLOGY}: {probe.get('cap_hits', 0)}/"
        f"{probe.get('observed_samples', 0)} sampled responses stopped at the "
        f"{probe.get('max_completion_tokens_seen', 0)}-token cap; the model never "
        "emitted EOS, so the eval was cut short and scored ~0"
    )


def accuracy_passed(
    baseline_accuracy: float,
    new_accuracy: float,
    threshold: float = ACCURACY_THRESHOLD,
) -> bool:
    """Return True if accuracy drop is within tolerance."""
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
    "EVAL_KIND_CONTEXT_TOO_SMALL",
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
    "launch_enablement_allowed",
    "parse_eval_results",
    "read_eval_probe",
    "request_baseline_accuracy_stop",
    "resolve_enablement_mode",
    "resolve_served_context",
    "require_framework_accuracy_default",
    "require_kernel_accuracy_default",
    "served_context_hosts_eval",
]
