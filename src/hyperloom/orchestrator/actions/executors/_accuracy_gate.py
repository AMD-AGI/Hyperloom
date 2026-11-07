# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Accuracy gate — GSM8K eval integration for hyperloom.inference_optimizer."""

from __future__ import annotations

import glob
import json
import logging
import math
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from hyperloom.common.benchmark_config import BenchmarkPointError, read_benchmark_point
from hyperloom.common.gain_math import (
    ACCURACY_METRICS,
    AGENTX_ERROR_RATE_THRESHOLD_PCT,
    DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
    accuracy_meets_floor,
    accuracy_passed,
    finite_score,
)
from hyperloom_common.coerce import optional_positive_float, to_float
from hyperloom_common.env import env_flag, is_truthy
from hyperloom_common.io import safe_mtime
from hyperloom_common.jsonio import read_json_strict
from hyperloom.common.perf_metric import is_agentx_mode
from inference_suite.contracts import ProbeOutcome, ReportError, RunLayout
from inference_suite.contracts.launch import Launch
from inference_suite.contracts.layout import REPORT_FILE

from ._round_folder import MAGPIE_REPORT_FILE, RoundFolder, is_round_folder

log = logging.getLogger(__name__)

#: Where in its slot an AgentX accuracy round writes its run_dir, which the gate reads the slot's score from.
ACCURACY_DIR = "accuracy"
#: The eval-results key naming the definition digest the score was taken under, "" outside an inference_suite round.
EVAL_DEFINITION_DIGEST_KEY = "definition_digest"
#: The result key a round records the definition digest of its ``accuracy`` under.
ACCURACY_DEFINITION_DIGEST_KEY = "accuracy_definition_digest"

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

# Sidecar the probe writes into ``$RESULT_DIR`` when it trips; inference_suite writes its ProbeOutcome there on every
# eval, tripped or not.
EVAL_PROBE_FILENAME = "hyperloom_eval_probe.json"
_SUITE_SAMPLES_KEY = "samples_seen"

# stop_reason recorded when the baseline could not produce an accuracy result even though the accuracy test was
# expected to run.
BASELINE_ACCURACY_STOP_REASON = "baseline_accuracy_failed"


def materialized_run_eval_disabled(config_path: Path | str) -> bool:
    """Report whether lm-eval is disabled in the materialized benchmark config."""
    try:
        envs = read_benchmark_point(Path(config_path)).envs
    except (OSError, yaml.YAMLError, BenchmarkPointError):
        return False
    return not is_truthy(envs.get("RUN_EVAL"), default=True)


def request_baseline_accuracy_stop(shared_state: Any, *, context: str, cause: str = "") -> None:
    """Halt the run when the baseline accuracy test produced no result.

    ``cause`` names what actually went wrong, for the callers that can establish it.
    Without one the line keeps its long-standing reading -- a missing accuracy reference
    nobody can explain *is* a broken baseline setup -- but asserting that next to an
    identified cause would tell the operator the opposite of what the caller just found.
    """
    log.warning(
        "baseline accuracy test produced no result (%s); stopping run (%s)",
        context,
        cause or "broken baseline setup",
    )
    shared_state.set_stop_reason(BASELINE_ACCURACY_STOP_REASON)


def require_kernel_accuracy_default() -> bool:
    """Default for the kernel-patch accuracy-KEEP gate."""
    return env_flag("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", default=True)


def resolve_enablement_mode(shared_state: Any) -> str:
    """Read the session's enablement mode."""
    mode = str(getattr(shared_state, "enablement_mode", "") or "").strip().lower()
    return mode if mode in ENABLEMENT_MODES else ENABLEMENT_MODE_OFF


def launch_enablement_allowed(shared_state: Any) -> bool:
    """Whether a baseline that cannot launch may route into enablement."""
    return resolve_enablement_mode(shared_state) in (ENABLEMENT_MODE_LAUNCH, ENABLEMENT_MODE_ALL)


def eval_enablement_allowed(shared_state: Any) -> bool:
    """Whether a baseline accuracy-eval failure may route into enablement."""
    return resolve_enablement_mode(shared_state) in (ENABLEMENT_MODE_EVAL, ENABLEMENT_MODE_ALL)


def classify_accuracy_failure(score: Any, floor: float) -> str | None:
    """Classify an accuracy verdict; ``None`` means it passes the floor."""
    val = finite_score(score)
    if val is None:
        return EVAL_KIND_ACCURACY_UNAVAILABLE
    if val <= 0.0 or val < floor:
        return EVAL_KIND_ACCURACY_BELOW_FLOOR
    return None


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


def _round_quality_gate(report: Path) -> object:
    """The quality gate the report at ``report`` carries: an inference_suite round's, or Magpie's bench report's.

    Raises:
        OSError, ValueError: When the report cannot be read; a round report off its contract raises too.
    """
    if report.name == REPORT_FILE:
        return RoundFolder.read(report.parent).measurement()["quality_gate"]
    return json.loads(report.read_text(encoding="utf-8")).get("quality_gate")


def parse_quality_gate(workspace: Path | str) -> dict[str, Any]:
    """Read a scriptable (server-less) quality gate from the newest round report under ``workspace``."""
    workspace = Path(workspace)
    reports = [Path(f) for f in glob.glob(str(workspace / "**" / MAGPIE_REPORT_FILE), recursive=True)]
    reports += [path for path in workspace.glob(f"**/{REPORT_FILE}") if is_round_folder(path.parent)]
    if not reports:
        return {"quality_gate": None, "error": f"no round report in {workspace}"}
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    try:
        qg = _round_quality_gate(latest)
    except json.JSONDecodeError as exc:
        return {"quality_gate": None, "error": f"parse error: {exc}"}
    if not isinstance(qg, dict):
        return {"quality_gate": None, "error": f"no quality_gate in {latest}"}
    return {"quality_gate": qg, "source_file": str(latest)}


def _latest_agentx_result(workspace: Path | str) -> dict[str, Any] | None:
    """The newest ``inferencex_result.json`` under *workspace*, or None."""
    from hyperloom.common.benchmark_config import configured_benchmark, is_agentx
    from hyperloom.inference_optimizer.agentx.report import read_agentx_report

    native = read_agentx_report(Path(workspace))
    if native is not None:
        return native
    if is_agentx(configured_benchmark()):
        return None
    results = [Path(f) for f in glob.glob(str(Path(workspace) / "**" / "inferencex_result.json"), recursive=True)]
    if not results:
        return None
    latest = max(results, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("accuracy_gate: unreadable %s: %s", latest, exc)
        return None
    return data if isinstance(data, dict) else None


def parse_agentx_error_rate(workspace: Path | str) -> float | None:
    """Read ``request_error_rate`` from the newest ``inferencex_result.json``; None when no result reported one."""
    # None rather than 0.0 so an export without the field is incomparable instead of a perfect score.
    rate = (_latest_agentx_result(workspace) or {}).get("request_error_rate")
    return float(rate) if isinstance(rate, (int, float)) and not isinstance(rate, bool) else None


def _mlperf_inline_accuracy(result: dict[str, Any] | None) -> float | None:
    """The harness's inline score when every turn was scored; None otherwise."""
    result = result or {}
    score = result.get("accuracy_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return None
    if result.get("accuracy_missing_turns") != 0:
        return None
    return float(score)


#: The image-quality limits a scriptable gate is judged by when its script names none (D2: the script writes raw
#: scores). xDiT's own tolerance: optimizations that keep precision pass, gross divergence does not.
SCRIPTABLE_QUALITY_LIMITS: Mapping[str, float] = MappingProxyType(
    {"lpips_max": 0.15, "ssim_min": 0.85, "mse_max": 0.006}
)


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
        lim = quality_gate.get(limit_key, SCRIPTABLE_QUALITY_LIMITS[limit_key])
        if isinstance(val, (int, float)) and isinstance(lim, (int, float)):
            evaluated += 1
            if not ok(float(val), float(lim)):
                return False
    # A required gate with neither ``passed`` nor any usable threshold pair is ambiguous; treat it as a failure
    # (fail-closed).
    if require and evaluated == 0:
        return False
    return True


def _agentx_accuracy_result(workspace: Path) -> dict[str, Any]:
    """Grade an AgentX run on the GSM8K score its slot's accuracy round wrote, or on the MLPerf harness's inline score.

    Outside MLPerf the request error rate is only the measurement's validity, never a score: a slot whose accuracy
    round did not run or left no result has no accuracy, which every gate fails closed on.
    """
    from hyperloom.common.agentx_workload import is_mlperf_backend

    if not is_mlperf_backend():
        return _lm_eval_accuracy_result(RunLayout(workspace / ACCURACY_DIR).eval_dir)
    rate = parse_agentx_error_rate(workspace)
    passed = rate is not None and rate <= AGENTX_ERROR_RATE_THRESHOLD_PCT
    log.info("accuracy_gate: agentx request_error_rate=%s passed=%s", rate, passed)
    # The harness scores every smoke run inline, so each KEEP is gated on that score against the baseline's, the
    # same way lm-eval accuracy is. Over the error-rate ceiling, an unscored turn or no scores.json is a drop
    # outright: 0.0, not None, which a lane without a required gate reads as "no eval ran" and skips.
    score = _mlperf_inline_accuracy(_latest_agentx_result(workspace)) if passed else None
    return {
        "accuracy": 0.0 if score is None else score,
        "task": "mlperf_agentic_inline",
        "metric": "score",
        "error_rate": rate,
    }


def parse_eval_results(
    workspace: Path | str,
    framework: str | None = None,
    benchmark_mode: str = "",
) -> dict[str, Any]:
    """Extract the accuracy score from a workspace's eval output; AgentX reads its slot's accuracy round instead."""
    workspace = Path(workspace)

    from hyperloom.inference_optimizer import framework_registry

    scriptable = framework_registry.is_scriptable(framework)

    if is_agentx_mode(benchmark_mode):
        return _agentx_accuracy_result(workspace)

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

    return _lm_eval_accuracy_result(workspace)


def _lm_eval_accuracy_result(workspace: Path) -> dict[str, Any]:
    """Read the accuracy score from the latest lm-eval results*.json under ``workspace``."""
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
        for key in ACCURACY_METRICS:
            if key in metrics:
                score = metrics[key]
                if isinstance(score, (int, float)):
                    log.info("accuracy_gate: task=%s metric=%s score=%.4f source=%s", task_name, key, score, latest)
                    return {
                        "accuracy": float(score),
                        "task": task_name,
                        "metric": key,
                        "source_file": str(latest),
                        EVAL_DEFINITION_DIGEST_KEY: _eval_definition_digest(latest),
                        # lm-eval's own wall-clock for the run; it ran inside the round's post-ready interval.
                        "eval_runtime_sec": optional_positive_float(data.get("total_evaluation_time_seconds")),
                    }

    return {"accuracy": None, "error": f"no recognized metric in {latest}"}


def _eval_definition_digest(results_file: Path) -> str:
    """The definition digest of the inference_suite round whose eval wrote ``results_file``, "" outside one."""
    for run_dir in results_file.parents:
        layout = RunLayout(run_dir)
        if layout.eval_dir in results_file.parents and layout.launch.is_file():
            return run_eval_definition_digest(run_dir)
    return ""


def run_eval_definition_digest(run_dir: Path) -> str:
    """The definition digest of the eval the inference_suite round in ``run_dir`` planned, "" when it planned none."""
    layout = RunLayout(run_dir)
    if not layout.launch.is_file():
        return ""
    invocation = Launch.from_document(read_json_strict(layout.launch)).eval
    return "" if invocation is None else invocation.definition_digest


def read_eval_probe(workspace: Path | str) -> dict[str, Any] | None:
    """Read the newest generation-pathology probe sidecar below ``workspace``, when that probe tripped."""
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
    if _SUITE_SAMPLES_KEY in record:
        try:
            tripped = ProbeOutcome.from_document(record).tripped
        except ReportError:
            return None
        if not tripped:
            return None
    record["kind"] = EVAL_KIND_GENERATION_PATHOLOGY
    record["source_file"] = str(latest)
    return record


def _probe_evidence(probe: dict[str, Any]) -> str:
    if _SUITE_SAMPLES_KEY in probe:
        ratio = probe.get("length_ratio_observed") or 0.0
        return f"{ratio:.0%} of {probe[_SUITE_SAMPLES_KEY]} sampled responses stopped at the length cap"
    return (
        f"{probe.get('cap_hits', 0)}/{probe.get('observed_samples', 0)} sampled responses stopped at the "
        f"{probe.get('max_completion_tokens_seen', 0)}-token cap"
    )


def eval_probe_summary(probe: dict[str, Any] | None) -> str:
    """Render a one-line summary of a tripped probe for a log / journal reason."""
    if not probe:
        return ""
    return (
        f"{EVAL_KIND_GENERATION_PATHOLOGY}: {_probe_evidence(probe)}; the model never emitted EOS, so the eval was "
        "cut short and scored ~0"
    )


@dataclass(frozen=True)
class AccuracyReference:
    """The baseline score candidates are graded against, and the eval definition digest it was taken under.

    ``score`` is 0.0 when there is no baseline score; ``definition_digest`` is "" for a score no inference_suite round
    took, which compares only with scores taken outside one too.
    """

    score: float
    definition_digest: str

    @classmethod
    def of_baseline(cls, state: Any) -> AccuracyReference:
        """The reference the session ``state`` measured at its baseline; no score without a state."""
        return cls(
            to_float(getattr(state, "baseline_accuracy", 0.0), 0.0) or 0.0,
            str(getattr(state, "baseline_accuracy_definition_digest", "") or ""),
        )

    @classmethod
    def of_session(cls, state: Any, params: Mapping[str, Any]) -> AccuracyReference:
        """The baseline the session measured, else the one a proposal's ``params`` carry, which no suite round took."""
        measured = cls.of_baseline(state)
        if measured.score > 0:
            return measured
        proposed = to_float(params.get("accuracy_baseline"), 0.0) or to_float(params.get("baseline_accuracy"), 0.0)
        return cls(proposed or 0.0, "")

    def comparable(self, eval_results: Mapping[str, Any]) -> float | None:
        """The score ``eval_results`` holds when taken under this reference's definition; ``None`` otherwise.

        Scores under different eval definitions do not compare, so a score under another one is no score.
        """
        score = finite_score(eval_results.get("accuracy"))
        digest = str(eval_results.get(EVAL_DEFINITION_DIGEST_KEY) or "")
        if score is not None and digest != self.definition_digest:
            log.warning(
                "accuracy_gate: score %.4f was taken under eval definition %r, the baseline's under %r; not compared",
                score,
                digest,
                self.definition_digest,
            )
            return None
        return score


def grade_accuracy(
    result_dir: str,
    reference: AccuracyReference,
    framework: str | None = None,
    benchmark_mode: str = "",
) -> bool | None:
    """Grade a bench's accuracy against the baseline ``reference``.

    With a recorded baseline the measured drop is enforced; without one, or without a score taken under the
    baseline's eval definition, the check is skipped (``None``) and warned loudly.
    For scriptable frameworks (xDiT) ``parse_eval_results`` fails closed on
    a missing quality gate instead of falling back to GSM8K.
    """
    baseline_value = reference.score
    eval_results = parse_eval_results(result_dir, framework=framework, benchmark_mode=benchmark_mode)
    new_accuracy = reference.comparable(eval_results)
    if new_accuracy is not None and baseline_value > 0:
        return accuracy_passed(baseline_value, new_accuracy)
    if baseline_value <= 0:
        log.warning(
            "integrate_patch: no baseline accuracy; accuracy gate skipped "
            "(throughput-only KEEP). Accuracy regressions will not be caught.",
        )
    else:
        log.warning("integrate_patch: variant produced no accuracy result; gate skipped")
    return None


def is_eval_origin(params: dict[str, Any]) -> bool:
    """Whether the enablement candidate came from the eval gate, not the boot gate."""
    return str(params.get("enablement_origin") or "") == "eval"


def framework_run_eval_envs(params: dict[str, Any]) -> dict[str, Any] | None:
    """Force ``RUN_EVAL=true`` for framework-authored source patches.

    Two independent triggers:

    * **Eval-origin enablement**: force ``RUN_EVAL=true`` so ``_bench_patch``
      can obtain a raw accuracy for the runnable gate, which fails closed
      without one. A boot-origin candidate is only ever provisional on a
      missing accuracy, so it inherits the session's contract instead.
    * **Perf framework authoring**: force only when a comparable baseline
      accuracy exists (``accuracy_baseline > 0``); otherwise leave the
      candidate's ``RUN_EVAL`` to the materializer's default handling.

    A plain configuration integrate_patch is untouched (returns ``None``).

    Args:
        params: The integrate_patch task params.

    Returns:
        ``{"RUN_EVAL": "true"}`` for eval-origin enablement patches or for
        framework-authored perf patches with a positive baseline accuracy
        to compare against; else ``None``.
    """
    if bool(params.get("enablement")):
        return {"RUN_EVAL": "true"} if is_eval_origin(params) else None
    fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
    try:
        baseline = float(params.get("accuracy_baseline") or 0.0)
    except (TypeError, ValueError):
        baseline = 0.0
    return {"RUN_EVAL": "true"} if (fw_authored and baseline > 0) else None


def enablement_correctness(
    params: dict[str, Any],
    gate_evidence: dict[str, Any],
) -> tuple[bool | None, dict[str, Any]]:
    """Judge the candidate's accuracy against the floor its origin demands.

    Returns:
        ``(correctness_ok, eval_provenance)``. ``correctness_ok`` is ``None``
        only for a boot-origin round with no score at all, which claimed
        nothing about accuracy; every other absence fails closed.
    """
    enablement_accuracy = gate_evidence.get("enablement_accuracy")
    param_floor = params.get("enablement_accuracy_floor")
    floor = float(param_floor) if isinstance(param_floor, (int, float)) else DEFAULT_ENABLEMENT_ACCURACY_FLOOR
    eval_origin = is_eval_origin(params)
    accuracy_kind = classify_accuracy_failure(enablement_accuracy, floor)
    correctness_ok: bool | None
    if enablement_accuracy is None:
        # Truly absent: eval-origin fails closed; boot-origin stays provisional.
        correctness_ok = False if eval_origin else None
    else:
        # Present but below floor / non-positive / non-finite is a refusal.
        correctness_ok = accuracy_meets_floor(enablement_accuracy, floor)
    # eval-origin only: a score with no task/metric did not come from a real
    # eval, so it cannot clear the gate. Both keys are read from the
    # candidate's own run, stamped beside the accuracy being judged, and not
    # from a stored contract fingerprint: RUN_EVAL is itself a hashed contract
    # field, so an eval-less re-baseline would change that digest and veto
    # every later candidate without ever consulting its accuracy.
    if (
        eval_origin
        and correctness_ok
        and not (gate_evidence.get("enablement_accuracy_task") and gate_evidence.get("enablement_accuracy_metric"))
    ):
        correctness_ok = False
        log.warning(
            "integrate_patch: eval-origin accuracy %s carries no task/metric; reverting",
            enablement_accuracy,
        )
    return correctness_ok, {
        "enablement_origin": str(params.get("enablement_origin") or ""),
        "enablement_observed_accuracy": enablement_accuracy,
        "enablement_accuracy_floor": floor,
        "accuracy_task": gate_evidence.get("enablement_accuracy_task") or "",
        "accuracy_metric": gate_evidence.get("enablement_accuracy_metric") or "",
        "enablement_eval_failure_kind": accuracy_kind or "",
    }


__all__ = [
    "ACCURACY_DEFINITION_DIGEST_KEY",
    "ACCURACY_DIR",
    "BASELINE_ACCURACY_STOP_REASON",
    "BASELINE_EVAL_ACCURACY_FLOOR_KEY",
    "BASELINE_EVAL_EVIDENCE_KEY",
    "BASELINE_EVAL_FAILED_KEY",
    "BASELINE_EVAL_FAILURE_KIND_KEY",
    "BASELINE_EVAL_OBSERVED_ACCURACY_KEY",
    "AccuracyReference",
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
    "EVAL_DEFINITION_DIGEST_KEY",
    "EVAL_KIND_RUNTIME_FAILURE",
    "EVAL_PROBE_FILENAME",
    "accuracy_keep_block",
    "classify_accuracy_failure",
    "enablement_correctness",
    "eval_enablement_allowed",
    "eval_probe_summary",
    "framework_run_eval_envs",
    "grade_accuracy",
    "is_eval_origin",
    "launch_enablement_allowed",
    "parse_eval_results",
    "read_eval_probe",
    "request_baseline_accuracy_stop",
    "resolve_enablement_mode",
    "run_eval_definition_digest",
    "resolve_served_context",
    "require_kernel_accuracy_default",
    "served_context_hosts_eval",
]
