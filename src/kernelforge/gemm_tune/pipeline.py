# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One GEMM tuning run, and the preview of which tuners it would run, with no command line attached."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from . import __version__
from .dense_shapes import MGrid
from .router import RoutingEvidence, RoutingTarget

if TYPE_CHECKING:
    from .model_analyzer import ModelProfile
    from .report import TuneReport
    from .router import TunerSpec
    from .tuners.base import TuneContext, TuneResult

log = logging.getLogger("kernelforge.gemm_tune")


class TuneError(Exception):
    """A tuning run refused before it started; the message says why."""


class TuneFailed(Exception):
    """A tuning run that ended in failure; ``result`` is the failure it recorded in ``result.json``."""

    def __init__(self, result: dict) -> None:
        super().__init__(str(result.get("error", "")))
        self.result = result


def _setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging: file + stderr."""
    level = logging.DEBUG if verbose else logging.INFO
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (all messages)
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stderr handler (INFO+)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    root = logging.getLogger("kernelforge.gemm_tune")
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(sh)


def _safe_is_file(value: str) -> bool:
    """``Path.is_file()`` that never raises on an over-long pathname."""
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _demand_from_serving_log(server_log: str, output_dir: Path) -> str:
    """Parse a serving log into a demand file, or \"\" when it carries no demand."""
    from .evidence import moe_dispatch_keys, parse_log_file, write_demand

    # Hyperloom sets this for serving runs, making zero hits conclusive.
    # Operator logs without it remain inconclusive.
    hit_logging = os.environ.get("AITER_LOG_TUNED_CONFIG", "").strip() not in ("", "0")
    report = parse_log_file(server_log, hit_logging=hit_logging or None)

    demands = report.get("demands") or []
    # The dense misses are not the only demand the log carries.
    moe_keys = moe_dispatch_keys(report) or []
    if not demands and not moe_keys:
        av = (report.get("apply_verdict") or {}).get("verdict")
        log.info(
            "serving log %s carries no tuned-config misses and no MoE dispatch key (verdict=%s); "
            "keeping the configured shape source",
            server_log,
            av,
        )
        return ""

    try:
        path = write_demand(report, output_dir / "demand.json")
    except OSError as exc:
        log.warning("could not write demand.json: %s", exc)
        return ""

    described = [
        f"{d.get('table')} ({d.get('miss_count')} misses, {len(d.get('keys') or [])} distinct keys)" for d in demands
    ]
    if moe_keys:
        described.append(f"fused_moe ({len(moe_keys)} runtime dispatch key(s))")
    log.info("Derived demand from %s: %s", server_log, ", ".join(described))
    return str(path)


def _load_demand_report(demand_json: str) -> dict | None:
    """Parse the demand file once, for both selection and the coverage report.

    Selection needs the tables the runtime consulted, and the coverage report needs the same document to say what
    stayed uncovered.
    """
    if not demand_json:
        return None
    from .evidence import load_demand

    return load_demand(demand_json)


def _coverage_gaps(
    demand_report: dict | None,
    tuner_specs: list,
    output_dir: Path,
    results: list | None = None,
) -> list:
    """Write and return demanded tables not covered by selected tuners.

    With results, owners that produced nothing landable also count as gaps.
    Reporting is best-effort and never affects tuning.
    """
    if not demand_report:
        return []
    try:
        from .tier3 import coverage_gaps

        gaps = coverage_gaps(demand_report, tuner_specs, results)
        # The second pass writes even when it finds nothing, or the pre-run
        # file it supersedes would be read as this run's final answer.
        if gaps or results is not None:
            (output_dir / "coverage_gaps.json").write_text(
                json.dumps([g.to_dict() for g in gaps], indent=2),
                encoding="utf-8",
            )
        return gaps
    except (OSError, TypeError, ValueError):
        log.debug("could not record coverage gaps", exc_info=True)
        return []


def _attempt_tier3(gaps: list, ctx: "TuneContext") -> "TuneResult | None":
    """Try Tier3 for the strongest otherwise-uncovered gap.

    Never raises: returns a referee-verified improvement, a failed result when the
    attempt crashed, or None. Accepting the profile object keeps attribute access
    inside this failure boundary.
    """
    if not gaps:
        return None
    try:
        profile = ctx.profile
        output_dir = ctx.output_dir
        model_name = str(getattr(profile, "model_path", "") or getattr(profile, "architecture", "") or "unknown")
        from .evidence import demand_shapes, load_demand
        from .tier3 import attempt_generated_tuner
        from .tier3.dispatch import adapters_for
        from .tier3.gate import should_generate
        from .tier3.runner import Retiming, Tier3Target

        decision = should_generate(gaps)
        if not decision.allowed or decision.gap is None:
            log.info("tier3: not attempted -- %s", "; ".join(decision.reasons))
            return None

        adapter = adapters_for(decision.gap.table)
        demand = load_demand(ctx.demand_json)

        def shapes_for(gap):
            # ``load_demand`` returns a parsed dict, not an object with ``tables``.
            for entry in (demand or {}).get("demands") or []:
                if str(entry.get("table") or "") == gap.table:
                    return demand_shapes(entry)
            return []

        try:
            outcome = attempt_generated_tuner(
                decision,
                shapes_for,
                output_dir,
                target=Tier3Target(model_name=model_name, gpu=ctx.gpu_type, framework=ctx.framework),
                retiming=(
                    Retiming(adapter.make_baseline, adapter.make_dispatch, adapter.make_correctness, adapter.sync())
                    if adapter
                    else None
                ),
            )
        finally:
            # An adapter steers the library under test through process-wide state, so the
            # attempt has to be closed before anything else in this process runs -- the
            # report and the e2e validation that follow must not be served by whatever
            # candidate happened to be dispatched last.
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        log.info("tier3: %s -- %s", outcome.stage, outcome.reason)
        (output_dir / "tier3_outcome.json").write_text(
            json.dumps(outcome.to_dict(), indent=2),
            encoding="utf-8",
        )
        return _tier3_result(outcome, decision.gap)
    except Exception as exc:  # isolation boundary: the optional Tier3 generated-tuner attempt
        log.warning("tier3 attempt failed; tuning continues", exc_info=True)
        from .tuners.base import TuneResult

        return TuneResult(
            tuner_name="tier3_generated", status="failed", error=repr(exc), error_class=type(exc).__name__
        )


def _tier3_result(outcome: Any, gap: Any) -> "TuneResult | None":
    """Convert a referee-approved Tier3 outcome into a tuner result.

    Unverified artifacts return nothing. Marking the result as a candidate
    forces normal e2e validation; microbenchmark speedup is not an e2e claim.
    """
    from .tuners.base import TuneResult, micro_metrics

    if not outcome.ok or not outcome.output_csv:
        return None
    micro = micro_metrics(
        {
            "speedup": float(j.best_timing.speedup) if j.best_timing and j.best_timing.usable else None,
            "improved": j.improved,
        }
        for j in outcome.judgements
    )
    return TuneResult(
        tuner_name=f"tier3_generated_{Path(outcome.table).stem}",
        status="ok",
        artifact_path=outcome.output_csv,
        env_var=gap.env_var,
        env_value=outcome.output_csv,
        candidate=True,
        total_shapes=len(outcome.judgements),
        improved_shapes=micro.improved,
        best_micro_speedup=micro.best,
        avg_micro_speedup=micro.avg,
        key_source="runtime_observed",
    )


def _normalize_inline_shapes_json(value: str, output_dir: Path) -> str:
    """Return a usable shapes-JSON *file path*, materializing inline content."""
    text = (value or "").strip()
    if not text:
        return ""
    if _safe_is_file(text):
        return text
    if text[0] not in "[{":
        return ""
    parsed: object
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            import ast

            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return ""
    try:
        out = output_dir / "_inline_shapes.json"
        out.write_text(json.dumps(parsed), encoding="utf-8")
        log.warning("Received inline --shapes-json content; materialized to %s", out)
        return str(out)
    except (OSError, TypeError, ValueError):
        return ""


def _exit_failed(output_dir: Path, *, error: str, error_class: str) -> NoReturn:
    """Record a failed run in ``result.json``, then leave with it as :class:`TuneFailed`.

    The caller publishes the same failure on its own channel too, because a consumer reading one and not the other
    would otherwise see a run that never reported at all.
    """
    report_dict = {
        "status": "failed",
        "micro_decision": "failed",
        "error": error,
        "error_class": error_class,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    raise TuneFailed(report_dict)


# Named once because both commands declare them and the path check names them back.
OPT_UNTUNED_CSV = "--untuned-csv"
OPT_SHAPES_MANIFEST = "--shapes-manifest"
OPT_DEMAND = "--demand"


def _existing_file_or_blank(value: str, option: str) -> str:
    """A path option's value, or blank when it does not name an existing file."""
    if value and not _safe_is_file(value):
        log.warning("%s is not an existing file; ignoring it", option)
        return ""
    return value


def _usable_sources(sources: ShapeSources, output_dir: Path) -> ShapeSources:
    """The shape inputs with inline shapes-JSON content materialized and every option naming no file blanked.

    Callers sometimes pass inline JSON content instead of a file path.
    """
    return dataclasses.replace(
        sources,
        shapes_json=_normalize_inline_shapes_json(sources.shapes_json, output_dir),
        untuned_csv=_existing_file_or_blank(sources.untuned_csv, OPT_UNTUNED_CSV),
        moe_untuned_csv=_existing_file_or_blank(sources.moe_untuned_csv, "--moe-untuned-csv"),
        shapes_manifest=_existing_file_or_blank(sources.shapes_manifest, OPT_SHAPES_MANIFEST),
        demand_json=_existing_file_or_blank(sources.demand_json, OPT_DEMAND),
    )


def _with_serving_log_demand(sources: ShapeSources | RoutingInputs, output_dir: Path) -> ShapeSources | RoutingInputs:
    """The inputs, with demand read off the serving log when no demand file was given.

    The serving log is already handed to us for MoE stage detection, and it is the same log the demand parser reads.
    """
    server_log = sources.kernel_signature_log
    if sources.demand_json or not server_log or not _safe_is_file(server_log):
        return sources
    return dataclasses.replace(sources, demand_json=_demand_from_serving_log(server_log, output_dir))


def _route(
    profile: Any, target: RoutingTarget, sources: ShapeSources | RoutingInputs, demand_report: dict | None
) -> list:
    """The tuners routed for this model and these inputs; ``plan`` asks the router exactly what ``run`` does."""
    from .router import select_tuners

    evidence = RoutingEvidence(
        kernel_signature_log=sources.kernel_signature_log or None,
        has_untuned_csv=bool(sources.untuned_csv),
        # A demand file is a shape source like the others, and a stronger one: it lists the keys the runtime actually
        # asked for.
        has_shapes_json=bool(sources.shapes_json or sources.shapes_manifest or sources.demand_json),
        has_tunableop_input=bool(sources.tunableop_input),
        # ...and a stronger *selection* input for the same reason.
        demand_report=demand_report,
    )
    return select_tuners(profile, target, evidence)


def _write_gpu_check(output_dir: Path) -> None:
    """Record what the GPUs are doing before tuning starts, and warn when they are busy."""
    from .utils import check_gpu_status

    gpus = check_gpu_status()
    if not gpus:
        return
    (output_dir / "gpu_check.json").write_text(
        json.dumps([{"gpu_id": g.gpu_id, "utilization": g.utilization, "busy": g.busy} for g in gpus], indent=2),
        encoding="utf-8",
    )
    busy = [g for g in gpus if g.busy]
    if busy:
        log.warning("GPUs appear busy: %s. Tuning may conflict with running workloads.", [g.gpu_id for g in busy])


def _aiter_preflight_problems(output_dir: Path) -> list[str]:
    """Record the aiter tune/serve alignment check and return the problems that block tuning."""
    from .aiter_preflight import collect as _aiter_collect

    preflight = _aiter_collect()
    (output_dir / "aiter_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    for message in preflight["soft"]:
        log.warning("aiter preflight: %s", message)
    for message in preflight["hard"]:
        log.warning("aiter preflight PROBLEM: %s", message)
    if preflight["aligned"]:
        log.info("aiter preflight: serve aiter aligned with tuner root")
    return list(preflight["hard"])


def _explicit_token_list(tokens: str) -> list[int] | None:
    """The token counts ``--tokens`` named, or None when it named none."""
    cleaned = tokens.strip().strip("[](){}") if tokens else ""
    if not cleaned:
        return None
    return [int(part.strip().strip("'\"")) for part in cleaned.split(",") if part.strip().strip("'\"")]


def _cap_tuner_count(tuner_specs: list, max_tuners: int) -> list:
    """Cut the routed set to what the caller's share pays for, in priority order.

    A tuner the router skipped books no time, so it is not what the share buys:
    it keeps its place in the plan for its skip reason but never displaces a
    tuner that could have run.
    """
    kept: list = []
    dropped: list = []
    runnable_so_far = 0
    for spec in tuner_specs:
        if spec.should_run:
            runnable_so_far += 1
        over_ceiling = spec.should_run and runnable_so_far > max_tuners
        (dropped if over_ceiling else kept).append(spec)
    if dropped:
        log.info(
            "gemm-tune: lane ceiling of %d tuner(s); dropping %s",
            max_tuners,
            ", ".join(spec.name for spec in dropped),
        )
    return kept


def _apply_tuner_choice(tuner_specs: list, choice: TunerChoice, quant_type: str, output_dir: Path) -> list:
    """The routed set narrowed to a forced ``--tuner`` and cut to ``--max-tuners``; exits when the forced one cannot run."""
    if choice.tuner:
        from .router import TunerSpec

        tuner_specs = [t for t in tuner_specs if t.name == choice.tuner]
        if not tuner_specs and choice.tuner in _tuner_registry():
            log.warning(
                "Requested tuner %r not auto-selected (quant_type=%r); honoring explicit --tuner anyway.",
                choice.tuner,
                quant_type,
            )
            tuner_specs = [TunerSpec(choice.tuner, priority=20, estimated_minutes=20)]
        if not tuner_specs:
            log.error("Requested tuner %r not applicable for this model/framework", choice.tuner)
            _exit_failed(
                output_dir,
                error=f"Tuner {choice.tuner!r} not applicable",
                error_class="tuner_not_applicable",
            )
    # 0 means no ceiling was supplied.
    if choice.max_tuners > 0:
        tuner_specs = _cap_tuner_count(tuner_specs, choice.max_tuners)
    return tuner_specs


def _write_plan(ctx: TuneContext, tuner_specs: list, model_path: str, global_timeout: int) -> None:
    """Record what this run will try in ``plan.json``, and warn when its estimate exceeds the global timeout."""
    runnable = [t for t in tuner_specs if t.should_run]
    total_est_min = sum(t.estimated_minutes for t in runnable)
    plan = {
        "model_path": model_path,
        "framework": ctx.framework,
        "precision": ctx.precision,
        "quant_type": ctx.quant_type,
        "gpu_type": ctx.gpu_type,
        "tokens": ctx.tokens,
        "tuners": [
            {
                "name": t.name,
                "will_run": t.should_run,
                "skip_reason": t.skip_reason,
                "estimated_minutes": t.estimated_minutes,
            }
            for t in tuner_specs
        ],
        "total_estimated_minutes": total_est_min,
    }
    (ctx.output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log.info(
        "Plan: %d tuners (%d will run, %d skipped), estimated %.0f min total",
        len(tuner_specs),
        len(runnable),
        len(tuner_specs) - len(runnable),
        total_est_min,
    )
    if global_timeout > 0 and total_est_min * 60 > global_timeout:
        log.warning(
            "Estimated total time (%.0f min) exceeds global timeout (%d s / %.0f min). "
            "Lower-priority tuners may be skipped. Consider increasing --global-timeout "
            "or reducing --tokens coverage.",
            total_est_min,
            global_timeout,
            global_timeout / 60,
        )


def _tuner_context_for(ctx: Any, spec: Any, effective_timeout: int) -> Any:
    """A per-tuner context copy, so one tuner's bounds cannot leak into the next."""
    import dataclasses

    tuner_ctx = dataclasses.replace(ctx, timeout_s=effective_timeout)
    if not spec.token_hint:
        return tuner_ctx
    log.info(
        "%s: tuning the %d token count(s) the log shows it serving (%s), not the run's full coverage",
        spec.name,
        len(spec.token_hint),
        spec.token_hint[:8],
    )
    # ``tokens`` bounds the sweep; ``token_hint`` marks the observed allowed set.
    return dataclasses.replace(tuner_ctx, tokens=list(spec.token_hint), token_hint=list(spec.token_hint))


def _execute_tuners(tuner_specs: list, ctx: Any, global_deadline: float) -> tuple[list, list[tuple[str, str]]]:
    """Run each tuner that still earns its place, and say why the others did not."""
    from .candidates import is_candidate

    results: list = []
    skipped: list[tuple[str, str]] = []
    for spec in tuner_specs:
        if not spec.should_run:
            skipped.append((spec.name, spec.skip_reason or "unknown"))
            log.info("SKIP %s: %s", spec.name, spec.skip_reason)
            continue
        # A fallback tuner runs only when no earlier non-fallback tuner produced a deployable candidate.
        if spec.fallback and any(is_candidate(result) for result in results):
            skipped.append((spec.name, "fallback not needed: an earlier tuner produced a candidate"))
            log.info("SKIP %s: an earlier tuner already produced a candidate", spec.name)
            continue
        remaining = global_deadline - time.time()
        if remaining <= 0:
            skipped.append((spec.name, "global timeout exceeded"))
            log.warning("SKIP %s: global timeout exceeded", spec.name)
            continue
        # Cap per-tuner timeout to remaining global budget.
        effective_timeout = int(min(ctx.timeout_s, remaining))
        log.info("Running tuner: %s (timeout=%ds)", spec.name, effective_timeout)
        tuner_instance = _create_tuner(spec.name, _tuner_context_for(ctx, spec, effective_timeout))
        if tuner_instance is None:
            log.error("Unknown tuner: %s", spec.name)
            continue
        result = tuner_instance.execute()
        results.append(result)
        log.info(
            "Tuner %s finished: status=%s, improved=%s/%d, best_speedup=%s, elapsed=%.1fs",
            spec.name,
            result.status,
            "unmeasured" if result.improved_shapes is None else result.improved_shapes,
            result.total_shapes,
            "unmeasured" if result.best_micro_speedup is None else f"{result.best_micro_speedup:.3f}x",
            result.elapsed_s,
        )
    return results, skipped


def _write_artifact_manifest(report: Any, results: list, output_dir: Path, **provenance: Any) -> None:
    """Ship a TuningArtifactManifest beside the tuned CSV, so a consumer can decide reuse-vs-stale."""
    try:
        from .artifact_manifest import write_artifact_manifest

        manifest_path = write_artifact_manifest(report, results, output_dir, **provenance)
    except (OSError, TypeError, ValueError) as exc:
        log.warning("artifact manifest write failed (non-fatal): %s", exc)
        return
    report.artifacts["tuning_artifact_manifest"] = str(manifest_path)
    log.info("Tuning artifact manifest written to %s", manifest_path)


@dataclass(frozen=True)
class ServingWorkload:
    """The serving deployment the token coverage is computed for."""

    tp: int
    conc: int
    tokens: str
    m_grid: str


@dataclass(frozen=True)
class BenchSettings:
    """How each tuner benchmarks a config and what counts as an improvement."""

    iters: int
    warmup: int
    min_improvement_pct: float


@dataclass(frozen=True)
class TuneTimeouts:
    """The per-tuner and whole-session timeouts, in seconds; a zero global timeout is unlimited."""

    timeout: int
    global_timeout: int


@dataclass(frozen=True)
class TunerChoice:
    """A forced tuner, and the cap on how many routed tuners run; 0 means uncapped."""

    tuner: str
    max_tuners: int


@dataclass(frozen=True)
class ShapeSources:
    """The shape inputs a tuning run reads, and the serving log it detects MoE stages and demand from."""

    untuned_csv: str
    moe_untuned_csv: str
    shapes_json: str
    shapes_manifest: str
    demand_json: str
    tunableop_input: str
    kernel_signature_log: str


@dataclass(frozen=True)
class GpuSelection:
    """The GPUs tuning uses, and whether the busy-GPU preflight is skipped."""

    gpu_ids: str
    skip_gpu_check: bool


@dataclass(frozen=True)
class TuneRequest:
    """Every option of one ``gemm-tune run`` invocation, grouped by concern."""

    model_path: str
    target: RoutingTarget
    workload: ServingWorkload
    mp: int
    output_dir: str
    bench: BenchSettings
    timeouts: TuneTimeouts
    thorough: bool
    choice: TunerChoice
    sources: ShapeSources
    gpus: GpuSelection
    kb_current_lib: str
    verbose: bool

    def tune_context(
        self, profile: Any, target: RoutingTarget, sources: ShapeSources, tokens: list[int]
    ) -> TuneContext:
        """The context every tuner starts from: this request on the resolved target and the usable shape inputs."""
        from .tuners.base import TuneContext

        return TuneContext(
            profile=profile,
            **dataclasses.asdict(target),
            tp=self.workload.tp,
            conc=self.workload.conc,
            m_grid=MGrid(self.workload.m_grid),
            tokens=tokens,
            mp=self.mp,
            output_dir=Path(self.output_dir),
            **dataclasses.asdict(self.bench),
            timeout_s=self.timeouts.timeout,
            thorough=self.thorough,
            **{name: Path(value) for name, value in dataclasses.asdict(sources).items() if value},
            gpu_ids=self.gpus.gpu_ids,
        )


def _preflight(request: TuneRequest, output_path: Path) -> None:
    """Refuse the run before any tuning when aiter cannot serve it; record the GPU check unless told not to."""
    if not request.gpus.skip_gpu_check:
        _write_gpu_check(output_path)

    preflight_hard = _aiter_preflight_problems(output_path)
    if preflight_hard:
        raise TuneError("aiter preflight failed: " + "; ".join(preflight_hard))


def _model_profile(request: TuneRequest, output_path: Path) -> ModelProfile:
    """The model's GEMM-relevant shape; a model that cannot be analysed ends the run as :class:`TuneFailed`."""
    from .model_analyzer import analyze_model

    try:
        return analyze_model(request.model_path)
    except Exception as exc:  # isolation boundary: third-party model analysis, recorded as the run's failure
        log.exception("Model analysis failed: %s", exc)
        _exit_failed(output_path, error=str(exc), error_class=type(exc).__name__)


def _token_coverage(request: TuneRequest, output_path: Path) -> list[int]:
    """The token counts the tuned configs must cover: the explicit list, or the serving concurrency's coverage."""
    from .shapes import compute_token_coverage

    try:
        explicit_tokens = _explicit_token_list(request.workload.tokens)
    except ValueError as exc:
        _exit_failed(output_path, error=f"Invalid --tokens value: {exc}", error_class="invalid_tokens")
    token_list = compute_token_coverage(conc=request.workload.conc, explicit_tokens=explicit_tokens)
    log.info("Token coverage: %s", token_list)
    return token_list


def run_tuning(request: TuneRequest) -> TuneReport:
    """Tune GEMMs for the requested model and framework, and return the report ``result.json`` holds.

    Raises :class:`TuneError` for a run refused before it starts and :class:`TuneFailed` for one that failed after
    recording why.
    """
    from .router import resolve_gpu_type
    from .report import build_report, write_report

    output_path = Path(request.output_dir)
    _setup_logging(output_path, request.verbose)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time = time.time()

    try:
        target = dataclasses.replace(request.target, gpu_type=resolve_gpu_type(request.target.gpu_type))
    except ValueError as exc:
        raise TuneError(str(exc)) from exc
    sources = _with_serving_log_demand(_usable_sources(request.sources, output_path), output_path)

    log.info("kernelforge gemm-tune starting (artifact layout v%s)", __version__)
    log.info("Model: %s, Framework: %s, Precision: %s", request.model_path, target.framework, target.precision)

    _preflight(request, output_path)
    profile = _model_profile(request, output_path)
    token_list = _token_coverage(request, output_path)

    demand_report = _load_demand_report(sources.demand_json)
    tuner_specs = _route(profile, target, sources, demand_report)
    # What the runtime asked for that nothing selected can write.
    _coverage_gaps(demand_report, tuner_specs, output_path)
    tuner_specs = _apply_tuner_choice(tuner_specs, request.choice, target.quant_type, output_path)

    ctx = request.tune_context(profile, target, sources, token_list)
    global_timeout = request.timeouts.global_timeout
    _write_plan(ctx, tuner_specs, request.model_path, global_timeout)

    global_deadline = (start_time + global_timeout) if global_timeout > 0 else float("inf")
    results, skipped = _execute_tuners(tuner_specs, ctx, global_deadline)

    # Last, and only on what the selected tuners left behind: by now they have
    # all run, so a generated tuner cannot take time from one that would have
    # produced something, and the recomputed list says which of them did.
    coverage_gap_list = _coverage_gaps(demand_report, tuner_specs, output_path, results)
    if coverage_gap_list and time.time() < global_deadline:
        generated = _attempt_tier3(coverage_gap_list, ctx)
        if generated is not None:
            results.append(generated)

    report = build_report(results, skipped, ctx, started_at, time.time() - start_time)
    log.info("Report written to %s", write_report(report, output_path))

    # Only when a candidate was produced: the manifest carries provenance + trace linkage + weighted coverage + the
    # CSV hash, none of which mean anything without a tuned CSV to describe.
    if report.recommended_env:
        _write_artifact_manifest(
            report,
            results,
            output_path,
            shape_manifest_path=sources.shapes_manifest or None,
            tuner_lib_version=request.kb_current_lib,
        )
    return report


@dataclass(frozen=True)
class RoutingInputs:
    """The serving log and shape inputs a routing preview weighs."""

    kernel_signature_log: str
    untuned_csv: str
    shapes_json: str
    shapes_manifest: str
    demand_json: str
    tunableop_input: str


@dataclass(frozen=True)
class TunePlan:
    """Which tuners a run would start, for the model and the resolved target they were routed on."""

    profile: ModelProfile
    target: RoutingTarget
    tuner_specs: list[TunerSpec]


def plan_tuning(model_path: str, target: RoutingTarget, inputs: RoutingInputs) -> TunePlan:
    """Route the tuners a run would start, without executing them; raises :class:`TuneError` for an unknown GPU."""
    import tempfile

    from .model_analyzer import analyze_model
    from .router import resolve_gpu_type

    profile = analyze_model(model_path)
    try:
        target = dataclasses.replace(target, gpu_type=resolve_gpu_type(target.gpu_type))
    except ValueError as exc:
        raise TuneError(str(exc)) from exc

    # Same derivation as `run_tuning`, or the preview answers a different question than the thing it previews: a
    # serving log that unblocks TunableOp there would show it skipped here.
    with tempfile.TemporaryDirectory(prefix="forge-plan-") as scratch:
        inputs = _with_serving_log_demand(inputs, Path(scratch))
        tuner_specs = _route(profile, target, inputs, _load_demand_report(inputs.demand_json))
    return TunePlan(profile=profile, target=target, tuner_specs=tuner_specs)


def _tuner_registry() -> dict:
    """Return the name -> tuner-class registry (imported lazily)."""
    from .tuners.fmoe_ck import FmoeCKTuner
    from .tuners.a8w8 import A8W8Tuner
    from .tuners.a8w8_blockscale import A8W8BlockscaleTuner
    from .tuners.a8w8_bpreshuffle import A8W8BpreshuffleTuner
    from .tuners.a8w8_blockscale_bpreshuffle import A8W8BlockscaleBpreshuffleTuner
    from .tuners.a4w4_blockscale import A4W4BlockscaleTuner
    from .tuners.vllm_moe_triton import VllmMoeTritonTuner
    from .tuners.vllm_dense_tunableop import VllmDenseTunableopTuner
    from .tuners.sglang_dense_bf16 import SglangDenseBf16Tuner

    return {
        "fmoe_ck": FmoeCKTuner,
        "a8w8": A8W8Tuner,
        "a8w8_blockscale": A8W8BlockscaleTuner,
        "a8w8_bpreshuffle": A8W8BpreshuffleTuner,
        "a8w8_blockscale_bpreshuffle": A8W8BlockscaleBpreshuffleTuner,
        "a4w4_blockscale": A4W4BlockscaleTuner,
        "vllm_moe_triton": VllmMoeTritonTuner,
        "vllm_dense_tunableop": VllmDenseTunableopTuner,
        "sglang_dense_bf16": SglangDenseBf16Tuner,
    }


def _create_tuner(name: str, ctx):
    """Factory: create a tuner instance by name."""
    cls = _tuner_registry().get(name)
    if cls is None:
        return None
    return cls(ctx)
