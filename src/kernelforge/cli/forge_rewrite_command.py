# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-rewrite-by-flydsl`` command."""

from __future__ import annotations

import json

import click

from kernelforge.cli.options import (
    CEILING_ON,
    _normalize_gpu_type,
    _roofline_ceiling_option,
    _validate_max_hours,
)
from kernelforge.config import Config
from kernelforge.knowledge.kb_store.config import KnowledgeConfig
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB


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
@click.option(
    "--source-kernel", required=True, help="Path to the source kernel to rewrite (e.g. a Triton .py or a .hip)"
)
@click.option(
    "--driver",
    required=True,
    help="Path to the rewrite measurement driver. A conforming driver is "
    "used unchanged; otherwise --prepare-driver authors or repairs "
    "this self-contained file.",
)
@click.option(
    "--prepare-driver/--no-prepare-driver",
    default=True,
    show_default=True,
    help="Author or repair a non-conforming dual-path rewrite driver before PORT.",
)
@click.option(
    "--invocation-spec-file",
    default="",
    help="Optional invocation evidence JSON used only by rewrite driver preparation.",
)
@click.option(
    "--logical-op-name",
    "op_name",
    required=True,
    help="Stable logical identity of the workload (a namespace or "
    "punctuation is allowed). KernelForge derives the FlyDSL factory "
    "symbol from it and reports the symbol in the result; never "
    "re-derive it downstream.",
)
@click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir")
@click.option("--experiments-dir", required=True, help="Where to write forge_experiments")
@click.option(
    "--target-functions",
    default="",
    help="Comma-separated source kernel entry names (the @triton.jit name, "
    "or the __global__ function name for HIP/CUDA)",
)
@click.option(
    "--source-language",
    default="",
    help="Language the source kernel is written in; one of the "
    "source_languages reported by --capabilities-json. Inferred "
    "from the file when omitted, but a caller whose profiler saw "
    "the kernel run should state it: a traced Triton kernel lives "
    "in a .py that names no language.",
)
@click.option(
    "--source-entry",
    default="",
    help="Host callable in the source that runs the kernel, used as the "
    "live correctness oracle + baseline: ref(x)->y. Auto-discovered "
    "if omitted.",
)
@click.option("--shapes-json", default="[]", help="JSON list of {M,N,dtype} shapes driving correctness + benchmark")
@click.option("--snr-threshold", default=DEFAULT_SNR_THRESHOLD_DB, type=float)
@click.option(
    "--flydsl-kernel-name", default="kernel.py", help="Filename of the produced FlyDSL kernel in the workspace"
)
@click.option(
    "--gpu-target",
    default=None,
    help="ROCm compilation architecture, e.g. gfx950 (also exported to env)",
)
@click.option(
    "--gpu-type",
    default=None,
    callback=_normalize_gpu_type,
    help="Hardware SKU for rewrite KB identities, e.g. mi355x",
)
@click.option(
    "--rewrite-kb/--no-rewrite-kb",
    default=True,
    show_default=True,
    help="Read and publish rewrite recipes.",
)
@click.option("--model", default=None, help="LLM model (overrides CLAUDE_MODEL / CODEX_MODEL)")
@click.option("--permission-mode", default=None, help="Claude permission mode (default: acceptEdits)")
@click.option("--max-port-attempts", default=3, type=int, help="Max correctness-only port sessions before giving up")
@click.option(
    "--applyback/--no-applyback",
    default=True,
    show_default=True,
    help="Integrate the optimized kernel back into the framework repository and "
    "publish the patch. Disable it to deliver only the standalone kernel: the "
    "stage is skipped, its 20-minute reserve returns to the search, and the "
    "run's success no longer depends on a patch the caller did not ask for.",
)
@click.option(
    "--max-applyback-attempts",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum clean-room framework integration sessions.",
)
@click.option(
    "--max-hours",
    default=1.0,
    type=float,
    callback=_validate_max_hours,
    help="Total rewrite runtime budget (hours, minimum 1.0)",
)
@click.option(
    "--deadline-unix",
    default=0.0,
    type=float,
    help="Absolute UNIX deadline for PORT, OPTIMIZE, and apply-back finalization.",
)
@click.option(
    "--framework",
    default="",
    callback=_validate_rewrite_framework,
    help="Target framework for the apply-back patch (aiter, vllm, or sglang). "
    "Inferred from the source path when omitted.",
)
@click.option(
    "--applyback-import-module",
    "applyback_import_modules",
    multiple=True,
    help="Import target required to load before and after apply-back. Repeat for "
    "multiple modules; defaults to the source module inferred from its package.",
)
@click.option(
    "--git-branch",
    default="forge-rewrite-optimize",
    help="Development branch used by the nested FlyDSL forge-loop.",
)
@click.option(
    "--supervisor-backend", default="codex", help="OPTIMIZE supervisor backend on stall: 'codex' (default) or 'claude'"
)
@click.option(
    "--profile-timeout-sec", default=3600, type=int, help="OPTIMIZE: ceiling for the complete Analysis Agent workflow"
)
@_roofline_ceiling_option(
    "OPTIMIZE: passed to the nested forge-loop unchanged. 'off' (default) "
    "skips the roofline ceiling. 'on' has the loop estimate the kernel's "
    "per-shape theoretical achievable latency once its baseline is measured, "
    "and steer its planner by the attainment against it; the estimate costs a "
    "profiler pass and an analyst session, paid out of the OPTIMIZE budget."
)
@click.option("--result-json", default=None, help="Write the result dict here (also printed)")
def forge_rewrite(
    source_kernel,
    driver,
    prepare_driver,
    invocation_spec_file,
    op_name,
    workspace_dir,
    experiments_dir,
    target_functions,
    source_language,
    source_entry,
    shapes_json,
    snr_threshold,
    flydsl_kernel_name,
    gpu_target,
    gpu_type,
    rewrite_kb,
    model,
    permission_mode,
    max_port_attempts,
    applyback,
    max_applyback_attempts,
    max_hours,
    deadline_unix,
    framework,
    applyback_import_modules,
    git_branch,
    supervisor_backend,
    profile_timeout_sec,
    roofline_ceiling,
    result_json,
):
    """Rewrite a source kernel into FlyDSL and optimize it via forge-loop."""
    import os
    import re as _re

    overrides = {}
    if gpu_target:
        overrides["gpu_target"] = gpu_target
        os.environ["GPU_TARGET"] = gpu_target
    overrides["gpu_type"] = gpu_type
    if model:
        # Config exposes the model as ``agent_model`` (from_env keys on it); a bare ``model`` override is silently
        # ignored.
        overrides["agent_model"] = model
    rewrite_kb_enabled = bool(rewrite_kb)
    # A disabled KB must not validate ambient remote credentials.
    try:
        rewrite_knowledge_config = KnowledgeConfig.from_env(mode="local" if not rewrite_kb_enabled else None)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    config = Config.from_env(
        workspace=workspace_dir,
        knowledge_config=rewrite_knowledge_config,
        **overrides,
    )

    targets = [t.strip() for t in _re.split(r"[,\n]", target_functions) if t.strip()]
    try:
        shapes = json.loads(shapes_json) if shapes_json else []
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--shapes-json is not valid JSON: {e}")

    # Root containers: claude CLI bypassPermissions needs IS_SANDBOX=1.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")

    from kernelforge.rewrite_by_flydsl import run_rewrite

    result = run_rewrite(
        op_name=op_name,
        source_kernel=source_kernel,
        driver=driver,
        workspace=workspace_dir,
        experiments_dir=experiments_dir,
        target_functions=targets,
        config=config,
        source_entry=source_entry,
        source_language=source_language,
        shapes=shapes,
        snr_threshold=snr_threshold,
        flydsl_kernel_name=flydsl_kernel_name,
        max_port_attempts=max_port_attempts,
        optimize_max_hours=max_hours,
        permission_mode=permission_mode,
        supervisor_backend=supervisor_backend,
        profile_timeout_sec=profile_timeout_sec,
        roofline_ceiling=roofline_ceiling == CEILING_ON,
        result_json=result_json,
        deadline_unix=deadline_unix,
        framework=framework,
        optimize_git_branch=git_branch,
        prepare_driver=prepare_driver,
        invocation_spec_file=invocation_spec_file,
        applyback_import_modules=applyback_import_modules,
        max_applyback_attempts=max_applyback_attempts,
        applyback_enabled=bool(applyback),
        rewrite_kb_enabled=rewrite_kb_enabled,
    )
    # The structured result and sentinel were already emitted for callers to parse.
    if not (result or {}).get("success"):
        raise SystemExit(1)
