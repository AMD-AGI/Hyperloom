# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project a :class:`KernelContext` onto each lane's existing input contract.

Every function here is pure: it reads the context and the lane's own execution
knobs and returns the wrapper payload. No ``SharedState``, no environment, no
disk. That is the whole point -- once a fact is in the context, wiring it to a
lane is one line here, and what a lane is actually handed can be read in one
file instead of three.

The lane CLIs are unchanged. Anything a lane must build for itself (a shape
capture, a generated CSV, a re-keyed shape table) is materialization and is
passed in already resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel_context import KernelContext


@dataclass(frozen=True)
class GemmShapeSources:
    """Shape inputs after the GEMM lane materialized whatever it needed.

    Discovery puts candidates in the context; this is what survived the lane's
    own preference rules, a TunableOp capture or an aiter re-keying pass.
    """

    shapes_json: str = ""
    shapes_manifest: str = ""
    untuned_csv: str = ""
    moe_untuned_csv: str = ""
    tunableop_input: str = ""
    kernel_signature_log: str = ""
    tokens: str = ""


@dataclass(frozen=True)
class GemmExecution:
    """How the GEMM lane intends to spend its share of the phase.

    ``tp``, ``conc`` and ``gpu_type`` are carried here rather than read off the
    context because the tuner requires concrete values and this lane
    substitutes its own when the session did not state them -- unlike fusion,
    which omits the flag entirely and lets forge-fuse decide. The caller has
    already resolved them for its own routing, so it hands them over rather
    than having this function derive them a second time.
    """

    forge_framework: str
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
    """How the fusion lane intends to spend its share of the phase."""

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

    Both wrappers read an absent flag as "use your own default", and that is a
    different instruction from a zero.
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
        "model_path": workload.resolved_model_path,
        "framework": execution.forge_framework,
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
        **_when_positive("max_recipes", execution.max_recipes),
        "verbose": execution.verbose,
        # The serving smoke has to launch the same server shape the session
        # measured, and the A/B has to drive it at the same operating point.
        # The sequence lengths and decode batch were reachable all along --
        # forge-fuse has always accepted them -- but nothing forwarded them, so
        # the comparison ran at the CLI's defaults instead of the workload.
        **_when_positive("tp", workload.tp),
        **_when_positive("block_size", execution.block_size),
        **_when_positive("max_model_len", workload.max_model_len),
        **_when_positive("ab_isl", workload.isl),
        **_when_positive("ab_osl", workload.osl),
        **_when_positive("decode_batch", workload.conc),
        **({"framework_root": context.serving.framework_repo_root} if context.serving.framework_repo_root else {}),
    }


def apply_overrides(payload: dict[str, Any], overrides: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Let an explicit request win over the projected value, for ``keys`` only.

    Scoped to a named set rather than a blanket update: a request payload also
    carries routing fields such as ``task_id`` that are not wrapper arguments,
    and forwarding those made the wrapper reject the run.
    """
    for key in keys:
        value = overrides.get(key)
        if value not in (None, ""):
            payload[key] = value
    return payload


__all__ = [
    "FusionAgent",
    "FusionExecution",
    "GemmExecution",
    "GemmShapeSources",
    "apply_overrides",
    "fusion_input",
    "gemm_input",
]
