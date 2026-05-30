"""Prompt builder — constructs system prompts for specialist agents.

Assembles context-specific prompts from skills, KB content, session state,
and GPU/model information.
"""

from __future__ import annotations

import logging
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

    if not caps.geak and not caps.oob:
        lines.append("- Kernel-level optimization unavailable (no GEAK or OOB)")
        lines.append("- Focus on configuration tuning and framework-level changes")

    if not caps.tracelens:
        lines.append("- Use torch profiler for kernel discovery (TraceLens unavailable)")

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
    parts.append(_build_constraints_section(config, caps))
    parts.append("")
    parts.append("## Baseline")
    parts.append(f"- Throughput: {baseline.throughput:.1f} {baseline.throughput_unit}")
    if baseline_acc:
        parts.append(f"- Accuracy: {baseline_acc.metric_name} = {baseline_acc.score:.4f} (threshold: {baseline_acc.threshold})")
    parts.append(f"- Target: {baseline.throughput * (1 + config.target_gain/100):.1f} {baseline.throughput_unit}")

    return "\n".join(parts)
