# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``kernelforge gemm-tune`` command group."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from . import __version__

if TYPE_CHECKING:
    from .tuners.base import TuneResult

log = logging.getLogger("kernelforge.gemm_tune")


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
    try:
        from .evidence import moe_dispatch_keys, parse_log_file, write_demand

        # Hyperloom sets this for serving runs, making zero hits conclusive.
        # Operator logs without it remain inconclusive.
        hit_logging = os.environ.get("AITER_LOG_TUNED_CONFIG", "").strip() not in ("", "0")
        report = parse_log_file(server_log, hit_logging=hit_logging or None)
    except Exception:  # noqa: BLE001 - deriving demand must never fail tuning
        log.debug("could not parse %s for demand", server_log, exc_info=True)
        return ""

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
    """Parse the demand file once, for both selection and the coverage report."""
    if not demand_json:
        return None
    try:
        from .evidence import load_demand

        return load_demand(demand_json)
    except Exception:  # noqa: BLE001 - evidence must never fail the run
        log.debug("could not load demand report", exc_info=True)
        return None


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
    except Exception:  # noqa: BLE001 - a report must never fail the run
        log.debug("could not record coverage gaps", exc_info=True)
        return []


def _attempt_tier3(
    gaps: list,
    demand_json: str,
    output_dir: Path,
    *,
    profile: Any,
    gpu_type: str,
    framework: str,
) -> "TuneResult | None":
    """Try Tier3 for the strongest otherwise-uncovered gap.

    Never raises and returns only referee-verified improvements. Accepting the
    profile object keeps attribute access inside this failure boundary.
    """
    if not gaps:
        return None
    try:
        model_name = str(getattr(profile, "model_path", "") or getattr(profile, "architecture", "") or "unknown")
        from .evidence import demand_shapes, load_demand
        from .tier3 import attempt_generated_tuner
        from .tier3.dispatch import adapters_for
        from .tier3.gate import should_generate

        decision = should_generate(gaps)
        if not decision.allowed or decision.gap is None:
            log.info("tier3: not attempted -- %s", "; ".join(decision.reasons))
            return None

        adapter = adapters_for(decision.gap.table)
        demand = load_demand(demand_json)

        def shapes_for(gap):
            # ``load_demand`` returns a parsed dict, not an object with ``tables``.
            for entry in (demand or {}).get("demands") or []:
                if str(entry.get("table") or "") == gap.table:
                    return demand_shapes(entry)
            return []

        try:
            outcome = attempt_generated_tuner(
                gaps,
                shapes_for,
                output_dir,
                model_name=model_name,
                gpu=gpu_type,
                framework=framework,
                decision=decision,
                make_baseline=adapter.make_baseline if adapter else None,
                make_dispatch=adapter.make_dispatch if adapter else None,
                make_correctness=adapter.make_correctness if adapter else None,
                sync=adapter.sync() if adapter else None,
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
    except Exception:  # noqa: BLE001 - a bonus attempt must not fail the run
        log.warning("tier3 attempt failed; tuning continues", exc_info=True)
        return None


def _tier3_result(outcome: Any, gap: Any) -> "TuneResult | None":
    """Convert a referee-approved Tier3 outcome into a tuner result.

    Unverified artifacts return nothing. Marking the result as a candidate
    forces normal e2e validation; microbenchmark speedup is not an e2e claim.
    """
    from .tuners.base import TuneResult

    if not outcome.ok or not outcome.output_csv:
        return None
    best = max(
        (j.best_timing.speedup for j in outcome.judgements if j.best_timing and j.best_timing.usable),
        default=1.0,
    )
    return TuneResult(
        tuner_name=f"tier3_generated_{Path(outcome.table).stem}",
        status="ok",
        artifact_path=outcome.output_csv,
        env_var=gap.env_var,
        env_value=outcome.output_csv,
        candidate=True,
        total_shapes=len(outcome.judgements),
        improved_shapes=outcome.improved_shapes,
        best_micro_speedup=float(best or 1.0),
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


@click.group("gemm-tune")
def gemm_tune():
    """Deterministic GEMM tuning for AMD GPUs."""


@gemm_tune.command()
@click.option("--model-path", required=True, help="Path to model directory (must contain config.json)")
@click.option(
    "--framework",
    required=True,
    type=click.Choice(["sglang", "vllm", "vllm-aiter"]),
    help="Target framework (vllm-aiter = vllm with VLLM_ROCM_USE_AITER=1)",
)
@click.option("--precision", required=True, help="Precision: bf16, fp8, fp4, int8, awq")
@click.option(
    "--quant-type",
    default="auto",
    help="Quant type: auto, none, per_token, blockscale, bpreshuffle, awq, gptq, fp4, mxfp4",
)
@click.option("--gpu-type", default="auto", help="GPU type: auto (detect via rocminfo), mi300x, mi355x, gfx942, ...")
@click.option("--tp", default=1, type=int, help="Tensor parallel degree")
@click.option("--conc", default=64, type=int, help="Target serving concurrency (for token coverage)")
@click.option("--tokens", default="", help="Comma-separated explicit token list (overrides auto)")
@click.option("--mp", default=1, type=int, help="Number of GPUs for parallel tuning")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory for all artifacts")
@click.option("--iters", default=80, type=int, help="Benchmark iterations per config")
@click.option("--warmup", default=20, type=int, help="Warmup iterations")
@click.option("--min-improvement-pct", default=3.0, type=float, help="Min improvement threshold (%%)")
@click.option(
    "--timeout",
    default=10800,
    type=int,
    help="Per-tuner timeout in seconds (default 3h; first run includes JIT compilation)",
)
@click.option("--global-timeout", default=0, type=int, help="Global timeout for entire session (0=unlimited)")
@click.option(
    "--thorough",
    is_flag=True,
    help="Thorough mode: full search space (all libtypes, more shapes, no per-shape timeout). Slower but finds absolute best config.",
)
@click.option("--tuner", default="", help="Force a specific tuner (skip routing)")
@click.option(
    "--max-tuners",
    default=0,
    type=int,
    help="Run at most this many of the routed tuners, in priority order. 0 means uncapped.",
)
@click.option("--untuned-csv", default="", help="Input untuned CSV for dense aiter tuners")
@click.option(
    "--moe-untuned-csv",
    default="",
    help="Input untuned CSV for the MoE tuner, keyed on the tuple aiter dispatched at run time",
)
@click.option("--shapes-json", default="", help="Input shapes JSON from TraceLens/Hyperloom")
@click.option(
    "--shapes-manifest",
    default="",
    help="Weighted TraceShapeManifest JSON (Hyperloom WP-1); preferred dense-shape source when set",
)
@click.option(
    "--demand",
    "demand_json",
    default="",
    help="demand.json from a serving log (kernelforge gemm-tune evidence); the highest-priority shape source",
)
@click.option("--tunableop-input", default="", help="PyTorch TunableOp shape file")
@click.option("--kernel-signature-log", default="", help="Server log for 1-stage ASM detection")
@click.option("--gpu-ids", default="", help="Comma-separated GPU IDs to use")
@click.option("--skip-gpu-check", is_flag=True, help="Skip rocm-smi preflight check")
@click.option("--kb-current-lib", default="", help="Current backend lib_version recorded as artifact provenance")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    model_path: str,
    framework: str,
    precision: str,
    quant_type: str,
    gpu_type: str,
    tp: int,
    conc: int,
    tokens: str,
    mp: int,
    output_dir: str,
    iters: int,
    warmup: int,
    min_improvement_pct: float,
    timeout: int,
    global_timeout: int,
    thorough: bool,
    tuner: str,
    max_tuners: int,
    untuned_csv: str,
    moe_untuned_csv: str,
    shapes_json: str,
    shapes_manifest: str,
    demand_json: str,
    tunableop_input: str,
    kernel_signature_log: str,
    gpu_ids: str,
    skip_gpu_check: bool,
    kb_current_lib: str,
    verbose: bool,
):
    """Run GEMM tuning for the specified model and framework."""
    from .model_analyzer import analyze_model
    from .router import resolve_gpu_type, select_tuners
    from .shapes import compute_token_coverage
    from .utils import check_gpu_status, emit_result_json
    from .tuners.base import TuneContext
    from .report import build_report, write_report
    from .candidates import is_candidate

    output_path = Path(output_dir)
    _setup_logging(output_path, verbose)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time = time.time()

    try:
        gpu_type = resolve_gpu_type(gpu_type)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Defensive input normalization: callers sometimes pass inline JSON content instead of a file path.
    shapes_json = _normalize_inline_shapes_json(shapes_json, output_path)
    if untuned_csv and not _safe_is_file(untuned_csv):
        log.warning("--untuned-csv is not an existing file; ignoring it")
        untuned_csv = ""
    if moe_untuned_csv and not _safe_is_file(moe_untuned_csv):
        log.warning("--moe-untuned-csv is not an existing file; ignoring it")
        moe_untuned_csv = ""
    if shapes_manifest and not _safe_is_file(shapes_manifest):
        log.warning("--shapes-manifest is not an existing file; ignoring it")
        shapes_manifest = ""
    if demand_json and not _safe_is_file(demand_json):
        log.warning("--demand is not an existing file; ignoring it")
        demand_json = ""

    # The serving log is already handed to us for MoE stage detection, and it is the same log the demand parser reads.
    if not demand_json and kernel_signature_log and _safe_is_file(kernel_signature_log):
        demand_json = _demand_from_serving_log(kernel_signature_log, output_path)

    log.info("kernelforge gemm-tune starting (artifact layout v%s)", __version__)
    log.info("Model: %s, Framework: %s, Precision: %s", model_path, framework, precision)

    # GPU preflight
    if not skip_gpu_check:
        gpus = check_gpu_status()
        if gpus:
            gpu_check_path = output_path / "gpu_check.json"
            gpu_check_path.write_text(
                json.dumps(
                    [{"gpu_id": g.gpu_id, "utilization": g.utilization, "busy": g.busy} for g in gpus], indent=2
                ),
                encoding="utf-8",
            )
            busy = [g for g in gpus if g.busy]
            if busy:
                log.warning(
                    "GPUs appear busy: %s. Tuning may conflict with running workloads.",
                    [g.gpu_id for g in busy],
                )

    # aiter tune/serve alignment preflight (warn-only, best-effort).
    try:
        from .aiter_preflight import collect as _aiter_collect

        _pf = _aiter_collect()
        (output_path / "aiter_preflight.json").write_text(json.dumps(_pf, indent=2), encoding="utf-8")
        for _m in _pf["soft"]:
            log.warning("aiter preflight: %s", _m)
        for _m in _pf["hard"]:
            log.warning("aiter preflight PROBLEM: %s", _m)
        if _pf["aligned"]:
            log.info("aiter preflight: serve aiter aligned with tuner root")
    except Exception as _exc:  # noqa: BLE001 - preflight must never break tuning
        log.debug("aiter preflight skipped: %s", _exc)

    # Analyze model
    try:
        profile = analyze_model(model_path)
    except Exception as exc:
        log.error("Model analysis failed: %s", exc)
        report_dict = {
            "status": "failed",
            "micro_decision": "failed",
            "error": str(exc),
            "error_class": type(exc).__name__,
        }
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "result.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        emit_result_json(report_dict)
        raise SystemExit(2)

    # Compute token coverage.
    try:
        tokens_clean = tokens.strip().strip("[](){}") if tokens else ""
        explicit_tokens = (
            [int(t.strip().strip("'\"")) for t in tokens_clean.split(",") if t.strip().strip("'\"")]
            if tokens_clean
            else None
        )
    except ValueError as exc:
        report_dict = {
            "status": "failed",
            "micro_decision": "failed",
            "error": f"Invalid --tokens value: {exc}",
            "error_class": "invalid_tokens",
        }
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "result.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        emit_result_json(report_dict)
        raise SystemExit(2)
    token_list = compute_token_coverage(conc=conc, explicit_tokens=explicit_tokens)
    log.info("Token coverage: %s", token_list)

    # Parsed once: selection needs the tables the runtime consulted, and the coverage report needs the same document
    # to say what stayed uncovered.
    demand_report = _load_demand_report(demand_json)

    # Select tuners
    tuner_specs = select_tuners(
        profile,
        framework=framework,
        precision=precision,
        quant_type=quant_type,
        gpu_type=gpu_type,
        kernel_signature_log=kernel_signature_log or None,
        has_untuned_csv=bool(untuned_csv),
        # A demand file is a shape source like the others, and a stronger one: it lists the keys the runtime actually
        # asked for.
        has_shapes_json=bool(shapes_json or shapes_manifest or demand_json),
        has_tunableop_input=bool(tunableop_input),
        # ...and a stronger *selection* input for the same reason.
        demand_report=demand_report,
    )

    # What the runtime asked for that nothing selected can write.
    coverage_gap_list = _coverage_gaps(demand_report, tuner_specs, output_path)

    # If --tuner specified, filter to only that one.
    if tuner:
        from .router import TunerSpec

        selected = [t for t in tuner_specs if t.name == tuner]
        if not selected and tuner in _tuner_registry():
            log.warning(
                "Requested tuner %r not auto-selected (quant_type=%r); honoring explicit --tuner anyway.",
                tuner,
                quant_type,
            )
            selected = [TunerSpec(tuner, priority=20, estimated_minutes=20)]
        tuner_specs = selected
        if not tuner_specs:
            log.error("Requested tuner %r not applicable for this model/framework", tuner)
            report_dict = {
                "status": "failed",
                "micro_decision": "failed",
                "error": f"Tuner {tuner!r} not applicable",
                "error_class": "tuner_not_applicable",
            }
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "result.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            emit_result_json(report_dict)
            raise SystemExit(2)

    # Cut the routed set to what the caller's share pays for, in priority order so the dropped ones rank last. 0 means
    # no ceiling was supplied.
    if max_tuners > 0 and len(tuner_specs) > max_tuners:
        log.info(
            "gemm-tune: lane ceiling of %d tuner(s); dropping %s",
            max_tuners,
            ", ".join(spec.name for spec in tuner_specs[max_tuners:]),
        )
        tuner_specs = tuner_specs[:max_tuners]

    # Write plan
    plan = {
        "model_path": model_path,
        "framework": framework,
        "precision": precision,
        "quant_type": quant_type,
        "gpu_type": gpu_type,
        "tokens": token_list,
        "tuners": [
            {
                "name": t.name,
                "will_run": t.should_run,
                "skip_reason": t.skip_reason,
                "estimated_minutes": t.estimated_minutes,
            }
            for t in tuner_specs
        ],
        "total_estimated_minutes": sum(t.estimated_minutes for t in tuner_specs if t.should_run),
    }
    (output_path / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    runnable = [t for t in tuner_specs if t.should_run]
    total_est_min = sum(t.estimated_minutes for t in runnable)
    log.info(
        "Plan: %d tuners (%d will run, %d skipped), estimated %.0f min total",
        len(tuner_specs),
        len(runnable),
        sum(1 for t in tuner_specs if not t.should_run),
        total_est_min,
    )

    # Budget warning: if global_timeout is set but estimated time exceeds it
    if global_timeout > 0 and total_est_min * 60 > global_timeout:
        log.warning(
            "Estimated total time (%.0f min) exceeds global timeout (%d s / %.0f min). "
            "Lower-priority tuners may be skipped. Consider increasing --global-timeout "
            "or reducing --tokens coverage.",
            total_est_min,
            global_timeout,
            global_timeout / 60,
        )

    # Build context
    ctx = TuneContext(
        profile=profile,
        framework=framework,
        precision=precision,
        quant_type=quant_type,
        gpu_type=gpu_type,
        tp=tp,
        conc=conc,
        tokens=token_list,
        mp=mp,
        output_dir=output_path,
        iters=iters,
        warmup=warmup,
        min_improvement_pct=min_improvement_pct,
        timeout_s=timeout,
        thorough=thorough,
        untuned_csv=Path(untuned_csv) if untuned_csv else None,
        moe_untuned_csv=Path(moe_untuned_csv) if moe_untuned_csv else None,
        shapes_json=Path(shapes_json) if shapes_json else None,
        shapes_manifest=Path(shapes_manifest) if shapes_manifest else None,
        demand_json=Path(demand_json) if demand_json else None,
        tunableop_input=Path(tunableop_input) if tunableop_input else None,
        kernel_signature_log=Path(kernel_signature_log) if kernel_signature_log else None,
        gpu_ids=gpu_ids,
    )

    # Execute tuners
    results = []
    skipped = []
    global_deadline = (start_time + global_timeout) if global_timeout > 0 else float("inf")

    for spec in tuner_specs:
        if not spec.should_run:
            skipped.append((spec.name, spec.skip_reason or "unknown"))
            log.info("SKIP %s: %s", spec.name, spec.skip_reason)
            continue

        # A fallback tuner runs only when no earlier non-fallback tuner produced a deployable candidate.
        if spec.fallback and any(is_candidate(r) for r in results):
            skipped.append((spec.name, "fallback not needed: an earlier tuner produced a candidate"))
            log.info("SKIP %s: an earlier tuner already produced a candidate", spec.name)
            continue

        # Check global timeout
        remaining = global_deadline - time.time()
        if remaining <= 0:
            skipped.append((spec.name, "global timeout exceeded"))
            log.warning("SKIP %s: global timeout exceeded", spec.name)
            continue

        # Cap per-tuner timeout to remaining global budget
        effective_timeout = min(timeout, int(remaining)) if global_timeout > 0 else timeout

        # Create per-tuner context copy to avoid shared state mutation.
        import dataclasses

        tuner_ctx = dataclasses.replace(ctx, timeout_s=effective_timeout)
        if spec.token_hint:
            log.info(
                "%s: tuning the %d token count(s) the log shows it serving (%s), not the run's full coverage",
                spec.name,
                len(spec.token_hint),
                spec.token_hint[:8],
            )
            # ``tokens`` bounds the sweep; ``token_hint`` marks the observed allowed set.
            tuner_ctx = dataclasses.replace(
                tuner_ctx,
                tokens=list(spec.token_hint),
                token_hint=list(spec.token_hint),
            )

        log.info("Running tuner: %s (timeout=%ds)", spec.name, effective_timeout)
        tuner_instance = _create_tuner(spec.name, tuner_ctx)
        if tuner_instance is None:
            log.error("Unknown tuner: %s", spec.name)
            continue

        result = tuner_instance.execute()
        results.append(result)
        log.info(
            "Tuner %s finished: status=%s, improved=%d/%d, best_speedup=%.3fx, elapsed=%.1fs",
            spec.name,
            result.status,
            result.improved_shapes,
            result.total_shapes,
            result.best_micro_speedup,
            result.elapsed_s,
        )

    # Last, and only on what the selected tuners left behind: by now they have
    # all run, so a generated tuner cannot take time from one that would have
    # produced something, and the recomputed list says which of them did.
    coverage_gap_list = _coverage_gaps(demand_report, tuner_specs, output_path, results)
    if coverage_gap_list and time.time() < global_deadline:
        generated = _attempt_tier3(
            coverage_gap_list,
            demand_json,
            output_path,
            profile=profile,
            gpu_type=gpu_type,
            framework=framework,
        )
        if generated is not None:
            results.append(generated)

    # Build report
    total_elapsed = time.time() - start_time
    report = build_report(
        results,
        skipped,
        profile=profile,
        framework=framework,
        precision=precision,
        quant_type=quant_type,
        gpu_type=gpu_type,
        tp=tp,
        conc=conc,
        tokens=token_list,
        started_at=started_at,
        total_elapsed_s=total_elapsed,
    )

    # Write report to file
    report_path = write_report(report, output_path)
    log.info("Report written to %s", report_path)

    # Ship a TuningArtifactManifest alongside the tuned CSV when a candidate was produced (provenance + trace linkage
    # + weighted coverage + CSV hash so a consumer can decide reuse-vs-stale).
    if report.recommended_env:
        try:
            from .artifact_manifest import write_artifact_manifest

            am_path = write_artifact_manifest(
                report,
                results,
                output_path,
                shape_manifest_path=shapes_manifest or None,
                gpu_type=gpu_type,
                framework=framework,
                precision=precision,
                quant_type=quant_type,
                tp=tp,
                tuner_lib_version=kb_current_lib,
                generated_at=report.finished_at,
            )
            report.artifacts["tuning_artifact_manifest"] = str(am_path)
            log.info("Tuning artifact manifest written to %s", am_path)
        except Exception as exc:  # noqa: BLE001 — manifest must never break tuning
            log.warning("artifact manifest write failed (non-fatal): %s", exc)

    # Emit sentinel-wrapped JSON to stdout
    emit_result_json(report.to_dict())

    # Exit code
    if report.status == "failed":
        raise SystemExit(1)
    raise SystemExit(0)


@gemm_tune.command()
@click.argument("logs", nargs=-1, required=True)
@click.option("--out", default="", help="Write demand.json here (default: stdout summary only)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def evidence(logs: tuple[str, ...], out: str, verbose: bool):
    """Parse serving log(s) into a tuning demand list and an apply verdict."""
    import logging as _logging

    from .evidence import parse_log_file, write_demand

    _logging.basicConfig(level=_logging.DEBUG if verbose else _logging.INFO)

    merged: dict[str, Any] = {}
    for path in logs:
        report = parse_log_file(path)
        av = report["apply_verdict"]
        click.echo(f"=== {path}")
        click.echo(f"  apply: hit={av['hit']} miss={av['miss']} verdict={av['verdict']}")
        click.echo(f"  merged_tables={len(report['merged_tables'])}")
        for d in report["demands"]:
            ms = sorted({int(k["M"]) for k in d["keys"] if k.get("M") is not None})
            click.echo(
                f"  DEMAND {d['table']} tuner={d['tuner']} "
                f"miss={d['miss_count']} distinct_keys={d['distinct_keys']} "
                f"distinct_M={len(ms)}"
            )
        merged = report  # last log wins when --out is a single file
    if out:
        write_demand(merged, Path(out))
        click.echo(f"demand written to {out}")
    raise SystemExit(0)


@gemm_tune.command()
@click.option("--model-path", required=True, help="Path to model directory")
@click.option("--framework", required=True, type=click.Choice(["sglang", "vllm", "vllm-aiter"]))
@click.option("--precision", required=True, help="Precision: bf16, fp8, fp4, int8, awq")
@click.option("--quant-type", default="auto")
@click.option("--gpu-type", default="auto", help="GPU type: auto (detect via rocminfo), mi300x, mi355x, gfx942, ...")
@click.option("--kernel-signature-log", default="")
@click.option("--untuned-csv", default="")
@click.option("--shapes-json", default="")
@click.option("--shapes-manifest", default="", help="Weighted TraceShapeManifest JSON (Hyperloom WP-1)")
@click.option("--demand", "demand_json", default="", help="demand.json from `evidence`")
@click.option("--tunableop-input", default="")
def plan(
    model_path: str,
    framework: str,
    precision: str,
    quant_type: str,
    gpu_type: str,
    kernel_signature_log: str,
    untuned_csv: str,
    shapes_json: str,
    shapes_manifest: str,
    demand_json: str,
    tunableop_input: str,
):
    """Show which tuners would run without executing them."""
    import tempfile

    from .model_analyzer import analyze_model
    from .router import resolve_gpu_type, select_tuners

    profile = analyze_model(model_path)
    try:
        gpu_type = resolve_gpu_type(gpu_type)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Same derivation as `run`, or the preview answers a different question than the thing it previews: a serving log
    # that unblocks TunableOp there would show it skipped here.
    with tempfile.TemporaryDirectory(prefix="forge-plan-") as scratch:
        if not demand_json and kernel_signature_log and _safe_is_file(kernel_signature_log):
            demand_json = _demand_from_serving_log(kernel_signature_log, Path(scratch))

        tuner_specs = select_tuners(
            profile,
            framework=framework,
            precision=precision,
            quant_type=quant_type,
            gpu_type=gpu_type,
            kernel_signature_log=kernel_signature_log or None,
            has_untuned_csv=bool(untuned_csv),
            has_shapes_json=bool(shapes_json or shapes_manifest or demand_json),
            has_tunableop_input=bool(tunableop_input),
            demand_report=_load_demand_report(demand_json),
        )

    click.echo(f"Model: {model_path}")
    click.echo(f"  Architecture: {profile.architecture}")
    click.echo(f"  MoE: {profile.is_moe} (experts={profile.num_experts}, topk={profile.num_experts_per_tok})")
    click.echo(
        f"  Hidden: {profile.hidden_size}, Inter: {profile.intermediate_size}, MoE Inter: {profile.moe_intermediate_size}"
    )
    click.echo(f"  Quant: {profile.quant_method or 'none'} ({profile.quant_bits}-bit)")
    click.echo(f"\nFramework: {framework}, Precision: {precision}, Quant Type: {quant_type}, GPU Type: {gpu_type}")
    click.echo(f"\nTuners ({len(tuner_specs)}):")

    for spec in tuner_specs:
        if not spec.should_run:
            click.echo(f"  [SKIP] {spec.name}: {spec.skip_reason}")
        elif spec.fallback:
            click.echo(f"  [FALLBACK] {spec.name}: runs only if no earlier tuner produces a candidate")
        else:
            click.echo(f"  [RUN]  {spec.name}")


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
