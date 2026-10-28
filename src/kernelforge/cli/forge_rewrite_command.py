# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-rewrite-by-flydsl`` command."""

from __future__ import annotations

import json
from dataclasses import dataclass

import click

from kernelforge.cli.options import (
    CeilingSwitch,
    GpuOptions,
    _gpu_options,
    _roofline_ceiling_option,
    _validate_max_hours,
)
from kernelforge.option_groups import option_group
from kernelforge.config import Config, KnowledgeConfig
from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB
from kernelforge.rewrite_by_flydsl.optimize import OptimizeOptions
from kernelforge.rewrite_by_flydsl.request import ApplybackSwitch, RewriteBudget, RewriteWorkspace


def _validate_rewrite_framework(_ctx, _param, value):
    """Accept only a framework the capability handshake advertises."""
    from kernelforge.rewrite_by_flydsl.protocol import SUPPORTED_FRAMEWORKS

    cleaned = (value or "").strip().lower()
    if cleaned and cleaned not in SUPPORTED_FRAMEWORKS:
        raise click.BadParameter(f"unsupported framework {value!r}; expected one of " + ", ".join(SUPPORTED_FRAMEWORKS))
    return cleaned


def _emit_rewrite_capabilities(ctx, _param, value):
    """Answer the capability handshake before any required option is parsed."""
    if not value or ctx.resilient_parsing:
        return
    from kernelforge.rewrite_by_flydsl.protocol import capabilities

    click.echo(json.dumps(capabilities(), indent=2, sort_keys=True))
    ctx.exit()


def _emit_rewrite_applyback_contract(ctx, _param, value):
    """Publish producer-owned example documents for cross-repository checks."""
    if not value or ctx.resilient_parsing:
        return
    from kernelforge.rewrite_by_flydsl.protocol import applyback_contract_example

    click.echo(json.dumps(applyback_contract_example(), indent=2, sort_keys=True))
    ctx.exit()


@dataclass(frozen=True)
class RewriteSource:
    """The source kernel to rewrite, its measurement driver and the workload identity."""

    source_kernel: str
    driver: str
    prepare_driver: bool
    invocation_spec_file: str
    op_name: str


@dataclass(frozen=True)
class SourceDescription:
    """What the source kernel is, the shapes it is checked on, and the correctness threshold."""

    target_functions: str
    source_language: str
    source_entry: str
    shapes_json: str
    snr_threshold: float
    flydsl_kernel_name: str


@dataclass(frozen=True)
class PortOptions:
    """The model and permission mode of the port sessions, and how many may run."""

    model: str | None
    permission_mode: str | None
    max_port_attempts: int


@dataclass(frozen=True)
class ApplybackTarget:
    """The framework the apply-back patch targets and the imports checked around it."""

    framework: str
    applyback_import_modules: tuple[str, ...]


@dataclass(frozen=True)
class RewriteRequest:
    """Every option of one ``forge-rewrite-by-flydsl`` invocation, grouped by concern."""

    source: RewriteSource
    workspace: RewriteWorkspace
    description: SourceDescription
    gpu: GpuOptions
    rewrite_kb: bool
    port: PortOptions
    applyback: ApplybackSwitch
    budget: RewriteBudget
    applyback_target: ApplybackTarget
    optimize: OptimizeOptions
    roofline_ceiling: CeilingSwitch
    result_json: str | None


@click.command("forge-rewrite-by-flydsl")
@click.option(
    "--capabilities-json",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_emit_rewrite_capabilities,
    help="Print the machine-readable rewrite capability handshake and exit.",
)
@click.option(
    "--applyback-contract-json",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_emit_rewrite_applyback_contract,
    help="Print producer-authored apply-back manifest and outer-result examples.",
)
@option_group(
    "request",
    RewriteRequest,
    option_group(
        "source",
        RewriteSource,
        click.option(
            "--source-kernel", required=True, help="Path to the source kernel to rewrite (e.g. a Triton .py or a .hip)"
        ),
        click.option(
            "--driver",
            required=True,
            help="Path to the rewrite measurement driver. A conforming driver is "
            "used unchanged; otherwise --prepare-driver authors or repairs "
            "this self-contained file.",
        ),
        click.option(
            "--prepare-driver/--no-prepare-driver",
            default=True,
            show_default=True,
            help="Author or repair a non-conforming dual-path rewrite driver before PORT.",
        ),
        click.option(
            "--invocation-spec-file",
            default="",
            help="Optional invocation evidence JSON used only by rewrite driver preparation.",
        ),
        click.option(
            "--logical-op-name",
            "op_name",
            required=True,
            help="Stable logical identity of the workload (a namespace or "
            "punctuation is allowed). KernelForge derives the FlyDSL factory "
            "symbol from it and reports the symbol in the result; never "
            "re-derive it downstream.",
        ),
    ),
    option_group(
        "workspace",
        RewriteWorkspace,
        click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir"),
        click.option("--experiments-dir", required=True, help="Where to write forge_experiments"),
    ),
    option_group(
        "description",
        SourceDescription,
        click.option(
            "--target-functions",
            default="",
            help="Comma-separated source kernel entry names (the @triton.jit name, "
            "or the __global__ function name for HIP/CUDA)",
        ),
        click.option(
            "--source-language",
            default="",
            help="Language the source kernel is written in; one of the "
            "source_languages reported by --capabilities-json. Inferred "
            "from the file when omitted, but a caller whose profiler saw "
            "the kernel run should state it: a traced Triton kernel lives "
            "in a .py that names no language.",
        ),
        click.option(
            "--source-entry",
            default="",
            help="Host callable in the source that runs the kernel, used as the "
            "live correctness oracle + baseline: ref(x)->y. Auto-discovered "
            "if omitted.",
        ),
        click.option(
            "--shapes-json", default="[]", help="JSON list of {M,N,dtype} shapes driving correctness + benchmark"
        ),
        click.option("--snr-threshold", default=DEFAULT_SNR_THRESHOLD_DB, type=float),
        click.option(
            "--flydsl-kernel-name", default="kernel.py", help="Filename of the produced FlyDSL kernel in the workspace"
        ),
    ),
    _gpu_options("Hardware SKU for rewrite KB identities, e.g. mi355x"),
    click.option(
        "--rewrite-kb/--no-rewrite-kb",
        default=True,
        show_default=True,
        help="Read and publish rewrite recipes.",
    ),
    option_group(
        "port",
        PortOptions,
        click.option("--model", default=None, help="LLM model (overrides CLAUDE_MODEL / CODEX_MODEL / DSH_MODEL)"),
        click.option("--permission-mode", default=None, help="Claude permission mode (default: acceptEdits)"),
        click.option(
            "--max-port-attempts", default=3, type=int, help="Max correctness-only port sessions before giving up"
        ),
    ),
    option_group(
        "applyback",
        ApplybackSwitch,
        click.option(
            "--applyback/--no-applyback",
            default=True,
            show_default=True,
            help="Integrate the optimized kernel back into the framework repository and "
            "publish the patch. Disable it to deliver only the standalone kernel: the "
            "stage is skipped, its 20-minute reserve returns to the search, and the "
            "run's success no longer depends on a patch the caller did not ask for.",
        ),
        click.option(
            "--max-applyback-attempts",
            default=2,
            show_default=True,
            type=click.IntRange(min=1),
            help="Maximum clean-room framework integration sessions.",
        ),
    ),
    option_group(
        "budget",
        RewriteBudget,
        click.option(
            "--max-hours",
            default=1.0,
            type=float,
            callback=_validate_max_hours,
            help="Total rewrite runtime budget (hours, minimum 1.0)",
        ),
        click.option(
            "--deadline-unix",
            default=0.0,
            type=float,
            help="Absolute UNIX deadline for PORT, OPTIMIZE, and apply-back finalization.",
        ),
    ),
    option_group(
        "applyback_target",
        ApplybackTarget,
        click.option(
            "--framework",
            default="",
            callback=_validate_rewrite_framework,
            help="Target framework for the apply-back patch (aiter, vllm, or sglang). "
            "Inferred from the source path when omitted.",
        ),
        click.option(
            "--applyback-import-module",
            "applyback_import_modules",
            multiple=True,
            help="Import target required to load before and after apply-back. Repeat for "
            "multiple modules; defaults to the source module inferred from its package.",
        ),
    ),
    option_group(
        "optimize",
        OptimizeOptions,
        click.option(
            "--git-branch",
            default="forge-rewrite-optimize",
            help="Development branch used by the nested FlyDSL forge-loop.",
        ),
        click.option(
            "--supervisor-backend",
            default="codex",
            help="OPTIMIZE supervisor backend on stall: 'codex' (default) or 'claude'",
        ),
        click.option(
            "--profile-timeout-sec",
            default=3600,
            type=int,
            help="OPTIMIZE: ceiling for the complete Analysis Agent workflow",
        ),
    ),
    _roofline_ceiling_option(
        "OPTIMIZE: passed to the nested forge-loop unchanged. 'off' (default) "
        "skips the roofline ceiling. 'on' has the loop estimate the kernel's "
        "per-shape theoretical achievable latency once its baseline is measured, "
        "and steer its planner by the attainment against it; the estimate costs a "
        "profiler pass and an analyst session, paid out of the OPTIMIZE budget."
    ),
    click.option("--result-json", default=None, help="Write the result dict here (also printed)"),
)
def forge_rewrite(request: RewriteRequest):
    """Rewrite a source kernel into FlyDSL and optimize it via forge-loop."""
    import os
    import re as _re

    overrides = {}
    if request.gpu.gpu_target:
        overrides["gpu_target"] = request.gpu.gpu_target
        os.environ["GPU_TARGET"] = request.gpu.gpu_target
    overrides["gpu_type"] = request.gpu.gpu_type
    if request.port.model:
        # Config exposes the model as ``agent_model`` (from_env keys on it); a bare ``model`` override is silently
        # ignored.
        overrides["agent_model"] = request.port.model
    rewrite_kb_enabled = bool(request.rewrite_kb)
    # A disabled KB must not validate ambient remote credentials.
    try:
        rewrite_knowledge_config = KnowledgeConfig.from_env(mode="local" if not rewrite_kb_enabled else None)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    config = Config.from_env(
        workspace=request.workspace.workspace_dir,
        knowledge_config=rewrite_knowledge_config,
        **overrides,
    )

    description = request.description
    targets = [t.strip() for t in _re.split(r"[,\n]", description.target_functions) if t.strip()]
    try:
        shapes = json.loads(description.shapes_json) if description.shapes_json else []
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--shapes-json is not valid JSON: {e}")

    # Root containers: claude CLI bypassPermissions needs IS_SANDBOX=1.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")

    from kernelforge.rewrite_by_flydsl import run_rewrite
    from kernelforge.rewrite_by_flydsl.ingest import RewriteSubject
    from kernelforge.rewrite_by_flydsl.request import DriverSetup, RewriteRun

    source = request.source
    subject = RewriteSubject(
        op_name=source.op_name,
        source_kernel=source.source_kernel,
        target_functions=targets,
        source_entry=description.source_entry,
        source_language=description.source_language,
        framework=request.applyback_target.framework,
        shapes=shapes,
        snr_threshold=description.snr_threshold,
    )
    run = RewriteRun(
        subject=subject,
        driver=DriverSetup(source.driver, source.prepare_driver, source.invocation_spec_file),
        workspace=request.workspace,
        budget=request.budget,
        optimize=request.optimize,
        roofline_ceiling=request.roofline_ceiling.enabled,
        applyback=request.applyback,
        applyback_import_modules=request.applyback_target.applyback_import_modules,
        rewrite_kb_enabled=rewrite_kb_enabled,
        flydsl_kernel_name=description.flydsl_kernel_name,
        max_port_attempts=request.port.max_port_attempts,
        permission_mode=request.port.permission_mode,
        result_json=request.result_json,
    )
    result = run_rewrite(run, config)
    # The structured result and sentinel were already emitted for callers to parse.
    if not (result or {}).get("success"):
        raise SystemExit(1)
