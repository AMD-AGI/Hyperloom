# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Accuracy gate — GSM8K eval integration for hyperloom.inference_optimizer.

Baseline always runs GSM8K, and so does every variant whose round has ``RUN_EVAL``
on -- the default. The gate reads that score for every variant rather than
guessing from flag names which ones deserve reading. Threshold is
``baseline_accuracy - new_accuracy <= 0.05`` (5% tolerance), REVERT otherwise. A
session that opts out of eval records no baseline accuracy, and that is what
leaves serving ungated there. Kernel patches are handled by kernel-agent.
"""

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

log = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.05  # allowed deviation

# Shared accuracy floor, used by BOTH the baseline eval-failure trigger and the
# enablement KEEP gate so the two never diverge.
#
# It is non-zero because the floor is the ONLY correctness authority on the
# enablement KEEP path. At 0.0 the gate degenerates to ``accuracy > 0``, which
# admits a model that is answering essentially nothing: a real run KEPT a
# candidate scoring gsm8k=0.00076 (0.08% of a 0.906 baseline) as "correct".
#
# 0.5 separates a working baseline from a broken one with a wide margin on both
# sides. Across historical runs every healthy baseline scored >= 0.63 while the
# two genuinely broken ones scored 0.196 (MiniMax-M3-MXFP4, a miscompiled MoE
# kernel) and 0.000 (GLM-5.2-MXFP4). The previous 0.05 rejected only the fully
# collapsed regime and admitted the 0.196 case as a usable baseline.
DEFAULT_ENABLEMENT_ACCURACY_FLOOR = 0.5

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

# The serving configuration cannot answer an eval request at all, so no verdict
# can ever be produced under it. Distinct from every other eval failure because
# it is a property of the configuration rather than of the run: retrying the
# same round reproduces it exactly.
EVAL_KIND_CONTEXT_TOO_SMALL = "eval_context_too_small"

# Smallest prompt an eval task is assumed to send. Deliberately conservative: a
# five-shot gsm8k prompt runs to roughly a thousand tokens, so a context that
# cannot hold even 256 on top of the generation budget cannot hold any real
# task. Being conservative keeps this a proof of infeasibility, never a guess.
_MIN_EVAL_PROMPT_TOKENS = 256
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

# Truthy-false spellings that disable the accuracy gate.
_RUN_EVAL_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def materialized_run_eval_disabled(config_path: Path | str) -> bool:
    """Report whether lm-eval is disabled in the materialized benchmark config.

    ``materialize_config_with_envs`` writes the effective ``RUN_EVAL`` (folded
    from the base YAML ``benchmark.envs``, ``reference_envs``, ``extra_envs`` and
    process ``$RUN_EVAL``, defaulting to "true") into ``benchmark.envs.RUN_EVAL``
    -- the value the benchmark subprocess actually consumes. Reading it back is
    the single source of truth for "did eval run this round", reusing the shared
    ``_RUN_EVAL_FALSE_VALUES`` present-and-falsey semantics.

    Lives in this module because every arm that asks the question needs it --
    the baseline, the grid and the env materializer -- and this module imports no
    executor sibling, so all three can reach it without an import cycle.

    Args:
        config_path (Path | str): The materialized benchmark YAML config path.

    Returns:
        bool: ``True`` when the config's ``RUN_EVAL`` is present and falsey.
            A missing key reads as enabled (matches the materialize default).
            An unreadable config also reads as enabled: the eval-side guards
            keyed off this must fail closed, not skip themselves.
    """
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    envs = ((cfg.get("benchmark") or {}).get("envs")) or {}
    val = envs.get("RUN_EVAL")
    return val is not None and str(val).strip().lower() in _RUN_EVAL_FALSE_VALUES


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


def require_kernel_accuracy_default() -> bool:
    """Default for the kernel-patch accuracy-KEEP gate.

    A kernel patch that clears the E2E throughput bar must also clear the
    accuracy gate by default; opt out with
    ``INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY=0``.

    Returns:
        ``True`` unless the env var disables it.
    """
    v = os.environ.get("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "").strip().lower()
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
    """Whether a baseline accuracy-eval failure may route into enablement.

    ``--no-eval`` closes the lane: with no eval running there is nothing to repair.
    """
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
    # knobs belong here: they change how early an eval is cut short. So do the
    # generation-bounds knobs, for the same reason one rung lower: they decide
    # where each individual answer is truncated, so two runs that disagree on
    # them are not scoring the same eval even when the task and limit match.
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


def resolve_served_context(
    *,
    server_args: str | None,
    env_max_model_len: Any = 0,
) -> int:
    """Resolve the context length the server was actually started with.

    ``--max-model-len`` inside the server-args string wins over the
    ``MAX_MODEL_LEN`` env: the env is a request, the CLI flag is what the
    process honours, and the parameter search rewrites the flag while leaving
    the env untouched.

    Args:
        server_args: The server-args string (e.g. ``EXTRA_VLLM_ARGS``).
        env_max_model_len: The ``MAX_MODEL_LEN`` env value, used only when the
            server args do not carry the flag.

    Returns:
        The served context in tokens, or ``0`` when neither source resolves.
    """
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
    """Whether the served context can hold an eval prompt plus its completion.

    Answers only the question it can answer from configuration alone: is the
    context provably too small for ANY prompt once the generation budget is
    reserved. An unknown context or an unbounded generation budget yields
    ``True`` — the point is to identify configurations that cannot work, never
    to guess at ones that might not.

    Args:
        served_max_model_len: Context the server was started with; ``0`` when
            unknown.
        eval_max_tokens: Completion tokens the harness requests per sample;
            ``0`` or negative means unbounded.

    Returns:
        ``(fits, reason)``. ``reason`` is empty when it fits.
    """
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


# There is deliberately no "high accuracy risk" predicate here any more. It used
# to decide whether EXPLORE bothered to parse a variant's eval result, matching a
# hardcoded list of vLLM/SGLang flag names and VLLM_*/SGLANG_* env keys as
# substrings. Two ways that silently under-reported: a framework spelling the
# same knob differently (atom's ``--kv_cache_dtype`` never matched
# ``--kv-cache-dtype``) and a framework-specific knob nobody enrolled (atom's
# ``--online_quant_config``, which changes numeric precision directly). Since the
# round runs the eval whenever ``RUN_EVAL`` is on, the result is already on disk
# and the only thing the predicate bought was discarding it.


def parse_quality_gate(workspace: Path | str) -> dict[str, Any]:
    """Read a scriptable (server-less) quality gate from the bench report.

    Scriptable workloads cannot run a GSM8K eval, so their bench script decides
    for itself what correctness means and embeds the verdict in
    ``benchmark_report.json`` as a ``quality_gate`` block. xDiT diffusion
    compares an image (LPIPS/SSIM/MSE vs a fixed reference); an operator-supplied
    ``custom`` workload may use any measure it likes and report only ``passed``.
    This reads the most recent such block in ``workspace`` without interpreting
    it — see :func:`quality_gate_passed` for how a verdict is derived.

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
    """Read the generation-pathology probe sidecar, when the probe tripped.

    The probe only writes :data:`EVAL_PROBE_FILENAME` when it cuts an eval
    short, so ``None`` means the ordinary thing happened: the model terminated
    its answers, or the InferenceX patch never applied. Searched recursively
    because the baseline double-run evaluates in the warmup round, whose
    ``$RESULT_DIR`` nests under the task workspace.

    Newest by mtime wins: ``integrate_patch`` searches a grid slot whose sibling
    variants each own a sidecar, and attempt dirs are hash-named, so path order
    says nothing about which eval ran last.

    Args:
        workspace (Path | str): Benchmark workspace to search recursively.

    Returns:
        dict[str, Any] | None: The probe record stamped with ``kind`` and
        ``source_file``, or ``None`` when no readable sidecar exists.
    """
    try:
        matches = list(Path(workspace).rglob(EVAL_PROBE_FILENAME))
    except OSError:
        # A sibling directory under the search root can be removed by a parallel
        # task while the recursive walk is in flight; an unscannable tree yields
        # no probe verdict rather than an error.
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
    """Render a one-line summary of a tripped probe for a log / journal reason.

    Args:
        probe (dict[str, Any] | None): A :func:`read_eval_probe` record.

    Returns:
        str: The summary, or ``""`` when ``probe`` is empty.
    """
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
