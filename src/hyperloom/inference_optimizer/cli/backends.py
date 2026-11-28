# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-role backend construction for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.llm import llm_config
from hyperloom.inference_optimizer.session.session_paths import agent_dir
from hyperloom.orchestrator.roles import (
    ClaudeBackend,
    CodexBackend,
    DeepSeekBackend,
)

_ORCHESTRATION_ROLE = "orchestration"


def orchestration_runs_on_codex(*, codex_follows_claude: bool, claude_follows_codex: bool = False) -> bool:
    """Whether the orchestration role runs on the Codex CLI rather than the Claude CLI."""
    # Both flags mean the caller has already rewritten one model id into the other's, which drops the backend that
    # would be handed the foreign id out of the running; only a launch that rewrote neither has two candidates left to
    # rank.
    return (not codex_follows_claude) and (
        claude_follows_codex or llm_config.preferred_agent_backend() == llm_config.AGENT_BACKEND_CODEX
    )


def _resolve_orchestration_backend(*, codex_follows_claude: bool, claude_follows_codex: bool = False) -> str:
    """Pick the agent CLI that runs the orchestration role."""
    requested = llm_config.explicit_agent_backend()
    if requested:
        return requested
    if orchestration_runs_on_codex(
        codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
    ):
        return llm_config.AGENT_BACKEND_CODEX
    return llm_config.AGENT_BACKEND_CLAUDE


@dataclass(frozen=True)
class ModelRouting:
    """The model ids a launch named for each CLI, and which side's id was rewritten into the other's."""

    claude_model: str
    codex_model: str
    codex_follows_claude: bool = False
    claude_follows_codex: bool = False
    deepseek_model: str = llm_config.DEFAULT_DEEPSEEK_HARNESS_MODEL


def _build_orchestration_backend(models: ModelRouting, *, session_dir: Path) -> Any:
    """The backend of the orchestration role, on the agent CLI :func:`_resolve_orchestration_backend` picks."""
    name = _resolve_orchestration_backend(
        codex_follows_claude=models.codex_follows_claude, claude_follows_codex=models.claude_follows_codex
    )
    # Scratch only.
    scratch = agent_dir(session_dir, _ORCHESTRATION_ROLE)
    if name == llm_config.AGENT_BACKEND_CODEX:
        return CodexBackend(model=models.codex_model, cwd=scratch / "codex_workspace")
    if name == llm_config.AGENT_BACKEND_DEEPSEEK:
        # The role reads state and answers with intents; it writes no files,
        # so the harness gets no write permission to confine.
        return DeepSeekBackend(
            model=models.deepseek_model,
            cwd=scratch / "deepseek_workspace",
            permission_mode="read-only",
        )
    # Orchestration is stateless: every tick opens a fresh session and
    # carries the full state projection, so it needs the agentic turn
    # budget.
    return ClaudeBackend(model=models.claude_model)


def _build_backends(models: ModelRouting, *, session_dir: Path) -> dict[str, Any]:
    """Construct all per-role backends."""
    return {_ORCHESTRATION_ROLE: _build_orchestration_backend(models, session_dir=session_dir)}


def _orchestration_banner(backend: Any, *, claude_model: str, codex_model: str) -> str:
    """Label the orchestration backend with the model it runs, for the launch banner."""
    if backend is None:
        return "DISABLED"
    name = str(getattr(backend, "name", "") or "").strip().lower()
    if name == "claude":
        return f"Claude({claude_model})"
    if name == "codex":
        return f"Codex({codex_model})"
    return f"{backend.__class__.__name__}({getattr(backend, 'model', '') or codex_model})"
