# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-fuse`` command: parse one invocation, run the fusion pipeline, and print its summary."""

from __future__ import annotations

import json

import click

from hyperloom.common.llm_config import AGENT_BACKEND_CHOICES
from kernelforge.cli.option_groups import option_group
from kernelforge.fusion import __version__
from kernelforge.fusion.errors import FuseError, FuseUsageError
from kernelforge.fusion.harness_noise import measure_harness_noise
from kernelforge.fusion.pipeline import fuse
from kernelforge.fusion.request import (
    DecodeSampling,
    DiscoveryChoice,
    FuseAgentOptions,
    FuseRequest,
    HarnessNoiseProbe,
    RunStages,
    ServingLayout,
    SmokeOptions,
)


@click.command("forge-fuse")
@click.version_option(version=__version__)
@option_group(
    "request",
    FuseRequest,
    click.option(
        "--trace",
        "trace_path",
        default="",
        type=click.Path(),
        help="Decode kineto trace (*.trace.json[.gz]), captured with CUDA graphs disabled.",
    ),
    click.option("--model-path", default="", help="Path to the model directory (must contain config.json)."),
    click.option(
        "--framework",
        default=None,
        type=click.Choice(["sglang", "vllm", "vllm-aiter"]),
        help="Target inference framework.",
    ),
    click.option(
        "--output-dir",
        default="",
        type=click.Path(),
        help="Output directory for the manifest + logs.",
    ),
    option_group(
        "noise",
        HarnessNoiseProbe,
        click.option(
            "--harness-noise",
            "harness_noise_path",
            default="",
            hidden=True,
            help="Diagnostic: repeat one kernel-validation harness and report its variance.",
        ),
        click.option(
            "--harness-noise-repeat",
            default=20,
            type=int,
            hidden=True,
            help="Diagnostic: how many times to repeat the harness.",
        ),
        click.option(
            "--harness-noise-env",
            "harness_noise_env",
            multiple=True,
            hidden=True,
            help="Diagnostic: env flag to set for each harness run (repeatable).",
        ),
    ),
    click.option(
        "--framework-root",
        default="",
        help="Explicit framework source root (else auto-detect the installed package).",
    ),
    click.option(
        "--repo-scope/--no-repo-scope",
        "repo_scope",
        default=False,
        help="Give discovery and authoring the whole framework repository instead of one "
        "resolved model file. Discovery embeds no source and explores the tree with its own "
        "read/search tools, may return a fusion whose call sites span several files, and "
        "authoring may edit all of them. Use when the chain is not in the arch-class model "
        "file and you do not want to name its location.",
    ),
    option_group(
        "decode",
        DecodeSampling,
        click.option("--decode-batch", default=16, type=int, help="Representative decode batch size (T) for shapes."),
        click.option(
            "--decode-steps",
            default=0,
            type=int,
            help="Decode steps captured in the trace (to normalize kernels/step).",
        ),
    ),
    option_group(
        "discovery",
        DiscoveryChoice,
        click.option(
            "--discover",
            "discover_mode",
            type=click.Choice(["patterns", "llm", "anchored"]),
            default="patterns",
            help="Recipe discovery: 'patterns' (template library), 'llm' (LLM reads trace+source, autonomous), "
            "or 'anchored' (fuse around the kernel named by --fuse-kernel).",
        ),
        click.option(
            "--fuse-kernel",
            "fuse_kernel",
            default="",
            help="Full GPU kernel name, exactly as the trace spells it. Fusion is then built around "
            "this kernel and its trace neighbours instead of a ranked guess; implies --discover anchored.",
        ),
    ),
    option_group(
        "stages",
        RunStages,
        click.option(
            "--dry-run",
            is_flag=True,
            help="Diagnose + locate only; emit manifest with recipe skeleton (no author/validate).",
        ),
        click.option("--author/--no-author", default=True, help="Author the fused kernel via the LLM (non-dry-run)."),
        click.option("--validate/--no-validate", default=True, help="Run the A/B decode validation (non-dry-run)."),
        click.option(
            "--fuse-all-confirmed",
            is_flag=True,
            help="Author ALL source-confirmed patterns together (not just the top), and "
            "A/B all their flags. A compile-pass candidate cannot be authored with "
            "them, so it is claimed alone and the rest wait for a later round.",
        ),
    ),
    click.option("--gpu", default="0", help="HIP device id for author + A/B."),
    option_group(
        "agent",
        FuseAgentOptions,
        click.option(
            "--agent-backend",
            type=click.Choice(["auto", *AGENT_BACKEND_CHOICES]),
            default="auto",
            show_default=True,
            help="Registered Agent provider for discovery and authoring.",
        ),
        click.option(
            "--agent-sandbox-mode",
            type=click.Choice(["workspace-write", "read-only", "bypass"]),
            envvar="FORGE_AGENT_SANDBOX_MODE",
            default="workspace-write",
            show_default=True,
            help="Agent runtime sandbox. Use bypass only when an external boundary already enforces isolation.",
        ),
        click.option(
            "--model",
            "llm_model",
            default=None,
            help="Agent model. Explicit value wins; otherwise uses provider-specific "
            "$CODEX_MODEL/$CLAUDE_MODEL/$DSH_MODEL, then the registered provider default.",
        ),
        click.option("--max-turns", default=100, type=int, help="Max authoring turns."),
    ),
    click.option(
        "--max-recipes",
        default=0,
        type=int,
        help="Cap how many ranked recipes to try. 0 means uncapped: every discovered recipe is considered.",
    ),
    option_group(
        "smoke",
        SmokeOptions,
        click.option("--ab-isl", default=512, type=int, help="A/B input length."),
        click.option("--ab-osl", default=128, type=int, help="A/B output length."),
        click.option(
            "--server-extra",
            default="",
            help=(
                "Extra serving args for the smoke launch (e.g. '--kv-cache-dtype fp8'). "
                "A model whose engine refuses to start without a flag can never reach the "
                "kernel the smoke exists to exercise."
            ),
        ),
    ),
    click.option(
        "--gpu-target",
        "gpu_arch",
        default="",
        help="Canonical GPU arch the author writes for (e.g. gfx950); auto-detected via rocminfo when omitted.",
    ),
    option_group(
        "layout",
        ServingLayout,
        click.option(
            "--tp",
            default=1,
            type=int,
            help="Tensor-parallel size for the serving smoke (must match the session).",
        ),
        click.option(
            "--attn-tp",
            "attn_tp_size",
            default=1,
            type=int,
            help=(
                "Attention TP shard used to stamp n_local_heads / n_local_groups "
                "(default 1; keep 1 under DP-attention even when --tp > 1)."
            ),
        ),
        click.option(
            "--block-size",
            "block_size",
            default=0,
            type=int,
            help="vLLM KV --block-size for the serving smoke (0=omit). Required for "
            "sparse-attention models that reject the default block size.",
        ),
        click.option(
            "--max-model-len",
            "max_model_len",
            default=0,
            type=int,
            help="Serving-smoke max model / context length (0 uses the smoke default 4096).",
        ),
    ),
    click.option("--verbose", "-v", is_flag=True, help="Verbose logging."),
)
def forge_fuse(request: FuseRequest) -> None:
    """Diagnose a decode trace and locate a fusion opportunity for the model."""
    noise = request.noise
    if noise.harness_noise_path:
        click.echo(
            json.dumps(
                measure_harness_noise(
                    harness=noise.harness_noise_path,
                    repeat=noise.harness_noise_repeat,
                    gpu=request.gpu,
                    env_flags=noise.harness_noise_env,
                ),
                indent=2,
            )
        )
        return

    try:
        result = fuse(request)
    except FuseUsageError as error:
        raise click.UsageError(str(error)) from error
    except FuseError as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result.summary))
    if result.exit_code:
        raise SystemExit(result.exit_code)


__all__ = ["forge_fuse"]
