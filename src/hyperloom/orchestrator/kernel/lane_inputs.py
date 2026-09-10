# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project a :class:`KernelContext` onto each lane's input contract.

Every function here is pure: it reads the context and the lane's own execution
knobs and returns the wrapper payload. No ``SharedState``, no environment, no
disk. That is the point -- once a fact is in the context, handing it to a lane
is one line here, and what a lane actually receives can be read in one file
instead of three.

The lane CLIs are unchanged. Anything a lane must build for itself -- a shape
capture, a generated CSV, a re-keyed shape table -- is materialization and
arrives already resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel_context import KernelContext


@dataclass(frozen=True)
class GemmShapeSources:
    """Shape inputs after the GEMM lane materialized whatever it needed.

    Discovery puts the candidates in the context; this is what survived the
    lane's own preference rules, a TunableOp capture, a block-FP8 profile reuse
    or an aiter re-keying pass.
    """

    tokens: str = ""
    untuned_csv: str = ""
    moe_untuned_csv: str = ""
    shapes_json: str = ""
    shapes_manifest: str = ""
    tunableop_input: str = ""
    kernel_signature_log: str = ""


@dataclass(frozen=True)
class GemmExecution:
    """How the GEMM lane intends to spend its share of the phase.

    ``framework``, ``tp``, ``conc`` and ``gpu_type`` are carried here rather
    than read off the context because the tuner requires concrete values and
    this lane substitutes its own when the session did not state them -- unlike
    fusion, which omits the flag and lets forge-fuse decide. ``framework`` is
    the routed one, which is not always the one being served: a vLLM run whose
    GEMMs go through aiter is handed to the aiter tuner family.
    """

    framework: str
    global_timeout: int
    per_tuner_timeout: int
    mp: int
    tp: int
    conc: int
    gpu_type: str
    max_tuners: int = 0
    thorough: bool = False
    tuner: str = ""


@dataclass(frozen=True)
class FusionAgent:
    """The authoring agent forge-fuse should run."""

    backend: str
    model: str
    sandbox_mode: str
    max_turns: int


@dataclass(frozen=True)
class FusionExecution:
    """How the fusion lane intends to spend its share of the phase.

    ``framework`` is carried here rather than read off the context because the
    wrapper requires one and this lane substitutes ``sglang`` when the session
    never stated it. The context keeps an unstated fact absent; applying a
    default is the lane's business.
    """

    framework: str
    timeout: int
    max_recipes: int = 0
    discover_mode: str = "llm"
    gpu: str = "0"
    #: KV block size, read from the model config for vLLM. Not a context fact:
    #: it exists only where the framework needs it.
    block_size: int = 0
    fuse_all_confirmed: bool = False
    verbose: bool = False


def _when_positive(key: str, value: int) -> dict[str, int]:
    """Emit a key only when it carries a real value.

    The wrapper reads an absent flag as "use your own default", and that is a
    different instruction from an explicit zero.
    """
    return {key: int(value)} if int(value or 0) > 0 else {}


def gemm_input(
    context: KernelContext,
    *,
    workspace: Path,
    shapes: GemmShapeSources,
    execution: GemmExecution,
) -> dict[str, Any]:
    """Build the ``forge_gemm_tuning.py`` payload for one tuning run."""
    workload = context.workload
    return {
        # The resolved directory, not the logical id: forge reads config.json
        # off it. Provenance and durable artifact names keep the logical one.
        "model_path": workload.resolved_model_path,
        "framework": execution.framework,
        "precision": workload.precision,
        "quant_type": workload.quant_type,
        "gpu_type": execution.gpu_type,
        "tp": execution.tp,
        "conc": execution.conc,
        "mp": execution.mp,
        "output_dir": str(workspace),
        # Strictly below ``global_timeout`` so the producer's own
        # min(per_tuner, remaining) bounds something: at parity the first tuner
        # could consume the session and every later one was skipped for time.
        "timeout": execution.per_tuner_timeout,
        "global_timeout": execution.global_timeout,
        "skip_gpu_check": True,
        "tokens": shapes.tokens,
        "untuned_csv": shapes.untuned_csv,
        "moe_untuned_csv": shapes.moe_untuned_csv,
        "shapes_json": shapes.shapes_json,
        "shapes_manifest": shapes.shapes_manifest,
        "tunableop_input": shapes.tunableop_input,
        "kernel_signature_log": shapes.kernel_signature_log,
        "tuner": execution.tuner,
        # How many routed tuners the lane's share pays for. Omitted when none
        # could be derived, which leaves the producer's own routing intact.
        **_when_positive("max_tuners", execution.max_tuners),
        "thorough": execution.thorough,
    }


def fusion_input(
    context: KernelContext,
    *,
    workspace: Path,
    agent: FusionAgent,
    execution: FusionExecution,
) -> dict[str, Any]:
    """Build the ``forge_fusion.py`` payload for one fusion run."""
    workload = context.workload
    serving = context.serving
    return {
        "trace_path": context.evidence.decode_trace.usable,
        "model_path": workload.model_path,
        "framework": execution.framework,
        "output_dir": str(workspace),
        "discover_mode": execution.discover_mode,
        "agent_backend": agent.backend,
        "llm_model": agent.model,
        "agent_sandbox_mode": agent.sandbox_mode,
        "max_turns": agent.max_turns,
        "gpu": execution.gpu,
        "timeout": execution.timeout,
        # Multi-patch (one independent sibling per recipe) is the default; the
        # combine escape hatch must be requested explicitly.
        "fuse_all_confirmed": execution.fuse_all_confirmed,
        # How many recipes the lane's share pays for. Omitted when none could be
        # derived, which leaves forge-fuse on every discovered recipe.
        **_when_positive("max_recipes", execution.max_recipes),
        "verbose": execution.verbose,
        # The serving smoke has to launch the server shape the session measured,
        # and the A/B has to drive it at the same operating point. forge-fuse has
        # always accepted the sequence lengths and the decode batch; nothing
        # forwarded them, so the comparison ran at the CLI's defaults instead.
        **_when_positive("tp", workload.tp),
        **_when_positive("block_size", execution.block_size),
        **_when_positive("max_model_len", workload.max_model_len),
        **_when_positive("ab_isl", workload.isl),
        **_when_positive("ab_osl", workload.osl),
        **_when_positive("decode_batch", workload.conc),
        # Only when the operator named one: forge-fuse auto-detects the
        # installed package otherwise, and an empty flag says to go and look.
        **({"framework_root": serving.framework_repo_root} if serving.framework_repo_root else {}),
    }


__all__ = [
    "FusionAgent",
    "FusionExecution",
    "GemmExecution",
    "GemmShapeSources",
    "fusion_input",
    "gemm_input",
]
