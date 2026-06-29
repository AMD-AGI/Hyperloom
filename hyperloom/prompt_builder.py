"""Prompt builder — constructs system prompts for specialist agents.

Assembles context-specific prompts from skills, KB content, session state,
and GPU/model information.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .capabilities import Capabilities
from .config import SessionConfig
from .kb import load_kb_content, select_kb
from .model_profile import ModelInfo

log = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(role: str) -> str:
    """Load the base skill prompt for an agent role."""
    skill_path = SKILLS_DIR / f"{role}.md"
    if skill_path.exists():
        return skill_path.read_text()
    log.warning("Skill file not found: %s", skill_path)
    return ""


def build_system_prompt(
    role: str,
    config: SessionConfig,
    model_info: ModelInfo,
    caps: Capabilities,
    context: str = "",
) -> str:
    """Build a complete system prompt for a specialist agent.

    Combines:
      1. Base skill prompt (from skills/<role>.md)
      2. Environment context (GPU, model, capabilities)
      3. KB content relevant to the task
      4. Session-specific instructions
    """
    parts: list[str] = []

    skill = load_skill(role)
    if skill:
        parts.append(skill)

    parts.append(_build_env_section(config, model_info, caps))

    arch_hints = _build_arch_opt_hints(model_info)
    if arch_hints:
        parts.append(arch_hints)

    if context:
        kb_files = select_kb(context)
        if kb_files:
            kb_content = load_kb_content(kb_files)
            parts.append(f"## Relevant Knowledge\n\n{kb_content}")

    parts.append(_build_constraints_section(config, caps))

    return "\n\n---\n\n".join(parts)


def _build_env_section(config: SessionConfig, model_info: ModelInfo, caps: Capabilities) -> str:
    """Build the environment context section."""
    lines = [
        "## Environment",
        "",
        f"- Model: {model_info.name} ({model_info.architecture})",
        f"- Hidden size: {model_info.hidden_size}, Layers: {model_info.num_layers}",
        f"- MoE: {'yes' if model_info.is_moe else 'no'}"
        + (f" ({model_info.num_experts} experts)" if model_info.is_moe else ""),
        f"- Shared expert: {'yes' if model_info.has_shared_expert else 'no'}"
        + (
            f" (n_shared={model_info.num_shared_experts})"
            if model_info.has_shared_expert and model_info.num_shared_experts
            else ""
        ),
        f"- MLA: {'yes' if model_info.is_mla else 'no'}",
        f"- GPU type: {config.gpu_type or 'auto-detected'}",
        f"- GPUs: {config.gpus or 'all available'}",
        f"- TP: {config.tp or 'auto'}",
        f"- Execution mode: {config.mode.value}",
        "",
        "### Available Tools",
        f"- GEAK: {'yes' if caps.geak else 'no'}",
        f"- OOB agents: {'yes' if caps.oob else 'no'}",
        f"- TraceLens: {'yes' if caps.tracelens else 'no'}",
        f"- Magpie: {'yes' if caps.magpie else 'no'}",
        f"- Ray cluster: {'yes' if caps.ray else 'no'}",
        f"- Torch profiler: {'yes' if caps.torch_profiler else 'no'}",
    ]
    return "\n".join(lines)


def _build_arch_opt_hints(model_info: ModelInfo) -> str:
    """Architecture-derived optimization hints (auto-detected, no env needed).

    Turns detected model structure into concrete, KB-backed directions so the
    orchestrator starts from a known lever instead of cold exploration. Returns
    "" when nothing applies.
    """
    hints: list[str] = []

    # Shared-expert fusion for MoE with an always-on shared expert.
    if model_info.is_moe and model_info.has_shared_expert:
        n = model_info.num_shared_experts or "≥1"
        hints.append(
            f"- **MoE + shared expert detected (n_shared={n}).** This model runs an "
            "always-on shared expert as a separate dense MLP every layer — a prime "
            "candidate for **shared-expert fusion** (fold it into the routed grouped "
            "GEMM to remove per-layer launches; biggest win in launch-bound decode at "
            "low/medium concurrency). This is fundamentally a **code change**, not just "
            "a flag: the env var `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` only "
            "activates fusion on backends that already wire it in. FIRST verify, on the "
            "actual MoE backend this checkpoint uses (e.g. MXFP8 native Triton vs AITER "
            "CK), whether the flag truly routes the shared expert through the fused "
            "grouped GEMM — measure tok/s with the flag off vs on. If the flag is a "
            "no-op for this backend (the shared expert still runs as a separate MLP), "
            "the win requires **implementing** the fusion in code: pass `n_shared_experts` "
            "into `FusedMoE` and fold the shared expert into the routed grouped-GEMM path "
            "for this backend (the PR-style change), then A/B benchmark + accuracy-gate it. "
            "Dispatch a specialist with \"shared expert fusion\" in the task (so the fusion "
            "KB is injected) and instruct it to produce a concrete code patch, not just "
            "env exports. See kb/fusion/empirical_kb.md for the mechanism, source "
            "touchpoints, caveats, and the validated A/B recipe (backend-neutral; needs "
            "gated activation, uniform expert precision, expert-parallelism OFF)."
        )

    if not hints:
        return ""
    return "## Architecture-derived optimization leads (auto-detected — evaluate first)\n\n" + "\n".join(hints)


def _build_constraints_section(config: SessionConfig, caps: Capabilities) -> str:
    """Build the constraints/rules section."""
    lines = [
        "## Constraints",
        "",
        f"- Target gain: {config.target_gain}%",
        f"- Max runtime: {config.max_hours} hours",
        "- Always run benchmark after each change to measure impact",
        "- Accuracy eval is MANDATORY after every optimization — reject any change that fails",
        "- Revert changes that cause regression",
    ]

    if os.environ.get("HYPERLOOM_FORBID_SPEC") == "1":
        lines.append(
            "- ⛔ OUT OF SCOPE — speculative / MTP / EAGLE / nextn draft-head decoding. "
            "This is a NON-MTP leaderboard campaign (goal: beat the B200 non-MTP point). "
            "Do NOT set SPEC=true, do NOT add any --speculative-* flag, and do NOT enable "
            "the MTP/nextn draft head under any circumstance. Pursue ONLY non-speculative "
            "throughput wins: kernel fusion, MoE/GEMM/MLA tuning, scheduler/batching, "
            "memory layout, communication, quantization, and CUDA-graph capture."
        )

    if not caps.geak and not caps.oob:
        lines.append("- No autonomous kernel-authoring tooling (GEAK/OOB absent)")
        lines.append("- You MAY still hand-edit kernel/framework source (sglang, aiter/CK, model code) "
                     "via bash/write_file, rebuild if needed, restart, and re-benchmark — "
                     "config tuning and framework-level changes are the fastest path, but source/kernel edits are allowed")

    if not caps.tracelens:
        lines.append("- Use torch profiler for kernel discovery (TraceLens unavailable)")

    # Optional operator-profiling-derived priority directions, injected via env so
    # the orchestrator starts from known hotspots instead of cold exploration.
    hints = os.environ.get("HYPERLOOM_OPT_HINTS", "").strip()
    if hints:
        lines.append("")
        lines.append("## Priority Directions (from operator-level profiling — start here)")
        lines.append("")
        lines.append(hints)

    return "\n".join(lines)


def build_agent_prompt(
    task: str,
    kb_content: str = "",
    session_dir: str = "",
) -> str:
    """Build a prompt for a specialist agent given a task description and KB context."""
    parts = [
        "# Specialist Agent Task\n",
        task,
    ]
    if kb_content:
        parts.append(f"\n## Relevant Knowledge\n\n{kb_content}")
    if session_dir:
        parts.append(f"\n## Session Directory\n\n{session_dir}")
    return "\n\n".join(parts)


def build_orchestrator_prompt(
    config: SessionConfig,
    model_info: ModelInfo,
    caps: Capabilities,
    baseline: Any,
    baseline_acc: Any,
) -> str:
    """Build the system prompt for the orchestrator agent.

    The orchestrator controls the optimization loop: dispatches specialists,
    integrates results, re-benchmarks, gates on accuracy, and iterates.
    """
    from .plugins.base import BenchResult, AccuracyResult

    parts = [
        "# Hyperloom Orchestrator",
        "",
        "You are the optimization orchestrator. Your job is to improve model serving ",
        "throughput while maintaining accuracy.",
        "",
        "## CRITICAL: Use Specialist Agents (dispatch_agents tool)",
        "",
        "You have these tools available:",
        "- **dispatch_agents**: Launch specialist sub-agents to explore optimizations",
        "- **check_agents**: Poll status of running agents (call every 30-60 seconds)",
        "- **collect_agent_results**: Get results, patches, and config changes from completed agents",
        "- **bash**: For benchmarks, server restarts, health checks, applying patches",
        "- **read_file / write_file**: For reading/writing configs and patches",
        "",
        "**DO NOT do optimization work yourself.** Your role is ONLY to:",
        "1. Call dispatch_agents with clear task descriptions for specialists",
        "2. Monitor their progress with check_agents",
        "3. Collect their results with collect_agent_results when done",
        "4. Apply their recommended patches/configs (via bash or write_file)",
        "5. Re-benchmark (via bash) and run accuracy eval",
        "6. Accept or revert based on results",
        "",
        "Example: call the dispatch_agents tool with tasks like:",
        "- 'Explore chunked prefill settings for MoE model with TP=2 on MI300X'",
        "- 'Profile decode latency and identify top-5 hotspot kernels'",
        "- 'Search for FMoE tuning CSV and optimal expert parallelism configs'",
        "- 'Test num-scheduler-steps values [1,4,8,16] for batched throughput'",
        "",
        "## Rules",
        "- Every optimization MUST be benchmarked before and after",
        "- Every optimization MUST pass accuracy eval (accuracy is never optional)",
        "- If a change degrades throughput or accuracy, REVERT it immediately",
        "- Dispatch specialists in parallel when possible",
        "- Stop when target is reached or time runs out",
        "- Write a STOP file to end the session gracefully",
        "- You can use bash directly for benchmarks, server restarts, and health checks",
        "- But NEVER use bash to do optimization research — dispatch agents for that",
        "",
    ]

    parts.append(_build_env_section(config, model_info, caps))
    parts.append("")
    arch_hints = _build_arch_opt_hints(model_info)
    if arch_hints:
        parts.append(arch_hints)
        parts.append("")
    parts.append(_build_constraints_section(config, caps))
    parts.append("")
    parts.append("## Baseline")
    parts.append(f"- Throughput: {baseline.throughput:.1f} {baseline.throughput_unit}")
    if baseline_acc:
        parts.append(f"- Accuracy: {baseline_acc.metric_name} = {baseline_acc.score:.4f} (threshold: {baseline_acc.threshold})")
    parts.append(f"- Target: {baseline.throughput * (1 + config.target_gain/100):.1f} {baseline.throughput_unit}")

    return "\n".join(parts)
