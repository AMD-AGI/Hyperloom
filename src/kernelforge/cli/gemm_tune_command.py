# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``kernelforge gemm-tune`` command group: parse each subcommand, run it, and report the outcome."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from kernelforge.cli.option_groups import option_group
from kernelforge.gemm_tune.dense_shapes import MGrid
from kernelforge.gemm_tune.pipeline import (
    OPT_DEMAND,
    OPT_SHAPES_MANIFEST,
    OPT_UNTUNED_CSV,
    BenchSettings,
    GpuSelection,
    RoutingInputs,
    ServingWorkload,
    ShapeSources,
    TuneError,
    TuneFailed,
    TuneRequest,
    TunerChoice,
    TuneTimeouts,
    plan_tuning,
    run_tuning,
)
from kernelforge.gemm_tune.router import RoutingTarget
from kernelforge.gemm_tune.utils import emit_result_json


@click.group("gemm-tune")
def gemm_tune():
    """Deterministic GEMM tuning for AMD GPUs."""


@gemm_tune.command()
@option_group(
    "request",
    TuneRequest,
    click.option("--model-path", required=True, help="Path to model directory (must contain config.json)"),
    option_group(
        "target",
        RoutingTarget,
        click.option(
            "--framework",
            required=True,
            type=click.Choice(["sglang", "vllm", "vllm-aiter"]),
            help="Target framework (vllm-aiter = vllm with VLLM_ROCM_USE_AITER=1)",
        ),
        click.option("--precision", required=True, help="Precision: bf16, fp8, fp4, int8, awq"),
        click.option(
            "--quant-type",
            default="auto",
            help="Quant type: auto, none, per_token, blockscale, bpreshuffle, awq, gptq, fp4, mxfp4",
        ),
        click.option(
            "--gpu-type", default="auto", help="GPU type: auto (detect via rocminfo), mi300x, mi355x, gfx942, ..."
        ),
    ),
    option_group(
        "workload",
        ServingWorkload,
        click.option("--tp", default=1, type=int, help="Tensor parallel degree"),
        click.option("--conc", default=64, type=int, help="Target serving concurrency (for token coverage)"),
        click.option("--tokens", default="", help="Comma-separated explicit token list (overrides auto)"),
        click.option(
            "--m-grid",
            default=MGrid.BUCKET.value,
            type=click.Choice([g.value for g in MGrid]),
            help="Decode and demand M: aiter's multiple-of-16 bucket up to 256 (bucket), or exact batch sizes (exact)",
        ),
    ),
    click.option("--mp", default=1, type=int, help="Number of GPUs for parallel tuning"),
    click.option("--output-dir", required=True, type=click.Path(), help="Output directory for all artifacts"),
    option_group(
        "bench",
        BenchSettings,
        click.option("--iters", default=80, type=int, help="Benchmark iterations per config"),
        click.option("--warmup", default=20, type=int, help="Warmup iterations"),
        click.option("--min-improvement-pct", default=3.0, type=float, help="Min improvement threshold (%%)"),
    ),
    option_group(
        "timeouts",
        TuneTimeouts,
        click.option(
            "--timeout",
            default=10800,
            type=int,
            help="Per-tuner timeout in seconds (default 3h; first run includes JIT compilation)",
        ),
        click.option("--global-timeout", default=0, type=int, help="Global timeout for entire session (0=unlimited)"),
    ),
    click.option(
        "--thorough",
        is_flag=True,
        help="Thorough mode: full search space (all libtypes, more shapes, no per-shape timeout). Slower but finds absolute best config.",
    ),
    option_group(
        "choice",
        TunerChoice,
        click.option("--tuner", default="", help="Force a specific tuner (skip routing)"),
        click.option(
            "--max-tuners",
            default=0,
            type=int,
            help="Run at most this many of the routed tuners, in priority order. 0 means uncapped.",
        ),
    ),
    option_group(
        "sources",
        ShapeSources,
        click.option(OPT_UNTUNED_CSV, default="", help="Input untuned CSV for dense aiter tuners"),
        click.option(
            "--moe-untuned-csv",
            default="",
            help="Input untuned CSV for the MoE tuner, keyed on the tuple aiter dispatched at run time",
        ),
        click.option("--shapes-json", default="", help="Input shapes JSON from TraceLens/Hyperloom"),
        click.option(
            OPT_SHAPES_MANIFEST,
            default="",
            help="Weighted TraceShapeManifest JSON (Hyperloom WP-1); preferred dense-shape source when set",
        ),
        click.option(
            OPT_DEMAND,
            "demand_json",
            default="",
            help="demand.json from a serving log (kernelforge gemm-tune evidence); the highest-priority shape source",
        ),
        click.option("--tunableop-input", default="", help="PyTorch TunableOp shape file"),
        click.option("--kernel-signature-log", default="", help="Server log for 1-stage ASM detection"),
    ),
    option_group(
        "gpus",
        GpuSelection,
        click.option("--gpu-ids", default="", help="Comma-separated GPU IDs to use"),
        click.option("--skip-gpu-check", is_flag=True, help="Skip rocm-smi preflight check"),
    ),
    click.option("--kb-current-lib", default="", help="Current backend lib_version recorded as artifact provenance"),
    click.option("--verbose", "-v", is_flag=True, help="Verbose logging"),
)
def run(request: TuneRequest):
    """Run GEMM tuning for the specified model and framework."""
    try:
        report = run_tuning(request)
    except TuneError as exc:
        raise click.ClickException(str(exc)) from exc
    except TuneFailed as failure:
        # Published on stdout as well as in result.json: a caller reading one and not the other would otherwise see a
        # run that never reported at all.
        emit_result_json(failure.result)
        raise SystemExit(2) from failure
    emit_result_json(report.to_dict())
    raise SystemExit(1 if report.status == "failed" else 0)


@gemm_tune.command()
@click.argument("logs", nargs=-1, required=True)
@click.option("--out", default="", help="Write demand.json here (default: stdout summary only)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def evidence(logs: tuple[str, ...], out: str, verbose: bool):
    """Parse serving log(s) into a tuning demand list and an apply verdict."""
    import logging as _logging

    from kernelforge.gemm_tune.evidence import parse_log_file, write_demand

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
@option_group(
    "target",
    RoutingTarget,
    click.option("--framework", required=True, type=click.Choice(["sglang", "vllm", "vllm-aiter"])),
    click.option("--precision", required=True, help="Precision: bf16, fp8, fp4, int8, awq"),
    click.option("--quant-type", default="auto"),
    click.option(
        "--gpu-type", default="auto", help="GPU type: auto (detect via rocminfo), mi300x, mi355x, gfx942, ..."
    ),
)
@option_group(
    "inputs",
    RoutingInputs,
    click.option("--kernel-signature-log", default=""),
    click.option(OPT_UNTUNED_CSV, default=""),
    click.option("--shapes-json", default=""),
    click.option(OPT_SHAPES_MANIFEST, default="", help="Weighted TraceShapeManifest JSON (Hyperloom WP-1)"),
    click.option(OPT_DEMAND, "demand_json", default="", help="demand.json from `evidence`"),
    click.option("--tunableop-input", default=""),
)
def plan(model_path: str, target: RoutingTarget, inputs: RoutingInputs):
    """Show which tuners would run without executing them."""
    try:
        preview = plan_tuning(model_path, target, inputs)
    except TuneError as exc:
        raise click.ClickException(str(exc)) from exc
    profile, target = preview.profile, preview.target

    click.echo(f"Model: {model_path}")
    click.echo(f"  Architecture: {profile.architecture}")
    click.echo(f"  MoE: {profile.is_moe} (experts={profile.num_experts}, topk={profile.num_experts_per_tok})")
    click.echo(
        f"  Hidden: {profile.hidden_size}, Inter: {profile.intermediate_size}, MoE Inter: {profile.moe_intermediate_size}"
    )
    click.echo(f"  Quant: {profile.quant_method or 'none'} ({profile.quant_bits}-bit)")
    click.echo(
        f"\nFramework: {target.framework}, Precision: {target.precision}, "
        f"Quant Type: {target.quant_type}, GPU Type: {target.gpu_type}"
    )
    click.echo(f"\nTuners ({len(preview.tuner_specs)}):")

    for spec in preview.tuner_specs:
        if not spec.should_run:
            click.echo(f"  [SKIP] {spec.name}: {spec.skip_reason}")
        elif spec.fallback:
            click.echo(f"  [FALLBACK] {spec.name}: runs only if no earlier tuner produces a candidate")
        else:
            click.echo(f"  [RUN]  {spec.name}")
