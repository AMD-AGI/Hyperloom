# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-role backend construction for the CLI."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import llm_config
from common.llm_config import has_anthropic_credential
from hyperloom.inference_optimizer.session.session_paths import agent_dir
from hyperloom.orchestrator.roles import (
    ClaudeBackend,
    CodexBackend,
    CriticAgentBackend,
    DeepSeekBackend,
    MockCriticBackend,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.scoring.proposal_scorer import ProposalScorer

CRITIC_PROTOCOL_CHOICES: tuple[str, ...] = ("auto", "openai", "anthropic")

_ORCHESTRATION_ROLE = "orchestration"


def _any_env_set(names: tuple[str, ...]) -> bool:
    return any((os.environ.get(name) or "").strip() for name in names)


def orchestration_runs_on_codex(*, codex_follows_claude: bool, claude_follows_codex: bool = False) -> bool:
    """Whether the orchestration role runs on the Codex CLI rather than the Claude CLI."""
    # Both flags mean the caller has already rewritten one model id into the other's, which drops the backend that
    # would be handed the foreign id out of the running; only a launch that rewrote neither has two candidates left to
    # rank.
    return (not codex_follows_claude) and (
        claude_follows_codex or llm_config.preferred_agent_backend() == llm_config.AGENT_BACKEND_CODEX
    )


def _resolve_orchestration_backend(*, codex_follows_claude: bool, claude_follows_codex: bool = False) -> str:
    """Pick the agent CLI that runs the orchestration role, and whose side the critic reviews on."""
    requested = llm_config.explicit_agent_backend()
    if requested:
        return requested
    if orchestration_runs_on_codex(
        codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
    ):
        return llm_config.AGENT_BACKEND_CODEX
    return llm_config.AGENT_BACKEND_CLAUDE


def critic_on_codex_side(*, codex_follows_claude: bool, claude_follows_codex: bool = False) -> bool:
    """Whether the critic reviews on the OpenAI side, given the backend the orchestration role runs on.

    The DeepSeek Harness has no side of its own, so a launch that names it keeps the side the credentials rank first.
    """
    backend = _resolve_orchestration_backend(
        codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
    )
    if backend == llm_config.AGENT_BACKEND_DEEPSEEK:
        return orchestration_runs_on_codex(
            codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
        )
    return backend == llm_config.AGENT_BACKEND_CODEX


def _resolve_critic_protocol(requested: str, *, orchestration_on_codex: bool) -> str:
    """Pick the critic's review protocol and verify that side is configured."""
    if requested == "auto":
        # The critic reviews on the side the orchestration model was validated on, so a launch configures one model
        # and every role uses it.
        return "openai" if orchestration_on_codex else "anthropic"

    if requested == "anthropic":
        # Asked through the registry rather than a local list of names, so a newly recognized credential form is
        # accepted here the moment it is registered instead of being rejected by a copy nobody updated.
        if not has_anthropic_credential():
            raise ValueError(
                "--critic-protocol=anthropic requires one of " + " / ".join(llm_config.ANTHROPIC_CREDENTIAL_ENV_ORDER)
            )
        return requested

    # Ask the resolver the review client will actually use, rather than re-deriving its key chain here.
    if requested == "openai":
        try:
            resolved = llm_config.resolve_openai_client_config()
        except llm_config.LLMConfigError as exc:
            raise ValueError(f"--critic-protocol=openai requires an OpenAI-capable credential: {exc}") from exc
        # A gateway key is scoped to its gateway.
        if resolved.base_url is None and not _any_env_set(("OPENAI_API_KEY",)):
            raise ValueError(
                "--critic-protocol=openai with only a gateway key requires a base URL "
                "(OPENAI_BASE_URL, or ANTHROPIC_BASE_URL to derive it from); otherwise "
                "the gateway key is sent to the official OpenAI endpoint"
            )
    return requested


def critic_review_target(
    requested: str,
    *,
    orchestration_on_codex: bool,
    claude_model: str,
    codex_model: str,
) -> tuple[str, str]:
    """Return the ``(protocol, model)`` the critic reviews with."""
    protocol = _resolve_critic_protocol(requested, orchestration_on_codex=orchestration_on_codex)
    return protocol, (claude_model if protocol == "anthropic" else codex_model)


def _load_action_verdict_policy() -> dict[str, str]:
    """Return the registry-derived per-action verdict policy (or empty on error)."""
    from ..protocol.action_surfaces import ACTION_CATALOGUE

    return {a.name: a.verdict_class for a in ACTION_CATALOGUE.values()}


@dataclass(frozen=True)
class ModelRouting:
    """The model ids a launch named for each CLI, and which side's id was rewritten into the other's."""

    claude_model: str
    codex_model: str
    codex_follows_claude: bool = False
    claude_follows_codex: bool = False
    deepseek_model: str = llm_config.DEFAULT_DEEPSEEK_HARNESS_MODEL


@dataclass(frozen=True)
class CriticSetup:
    """Which critic the run uses (``mock`` or ``agent``), its runtime root, and the review protocol."""

    choice: str
    agent_root: Path | None = None
    protocol: str = "auto"


def _build_orchestration_backend(models: ModelRouting, *, session_dir: Path) -> Any:
    """The backend of the orchestration role, on the agent CLI :func:`_resolve_orchestration_backend` picks."""
    name = _resolve_orchestration_backend(
        codex_follows_claude=models.codex_follows_claude, claude_follows_codex=models.claude_follows_codex
    )
    allowed_intents = default_role_registry()[_ORCHESTRATION_ROLE].allowed_intents
    # Scratch only.
    scratch = agent_dir(session_dir, _ORCHESTRATION_ROLE)
    if name == llm_config.AGENT_BACKEND_CODEX:
        return CodexBackend(allowed_intents=allowed_intents, model=models.codex_model, cwd=scratch / "codex_workspace")
    if name == llm_config.AGENT_BACKEND_DEEPSEEK:
        # The role reads state and answers with intents; it writes no files,
        # so the harness gets no write permission to confine.
        return DeepSeekBackend(
            allowed_intents=allowed_intents,
            model=models.deepseek_model,
            cwd=scratch / "deepseek_workspace",
            permission_mode="read-only",
        )
    # Orchestration is stateless: every tick opens a fresh session and
    # carries the full state projection, so it needs the agentic turn
    # budget.
    return ClaudeBackend(
        model=models.claude_model,
        allowed_intents=allowed_intents,
    )


def _build_backends(models: ModelRouting, critic: CriticSetup, *, session_dir: Path) -> dict[str, Any]:
    """Construct all per-role backends."""
    critic_choice, critic_agent_root = critic.choice, critic.agent_root
    claude_model, codex_model = models.claude_model, models.codex_model
    # The critic reviews on the side the orchestration backend runs on.
    orchestration_on_codex = critic_on_codex_side(
        codex_follows_claude=models.codex_follows_claude, claude_follows_codex=models.claude_follows_codex
    )

    if critic_choice == "mock":
        critic_backend: Any = MockCriticBackend()
    else:  # "agent"
        protocol = _resolve_critic_protocol(
            critic.protocol,
            orchestration_on_codex=orchestration_on_codex,
        )
        _policy = _load_action_verdict_policy()
        if protocol == "anthropic":
            critic_backend = CriticAgentBackend(
                critic_agent_root=critic_agent_root,
                session_dir=session_dir,
                protocol="anthropic",
                claude_model=claude_model,
                codex_model=codex_model,
                action_verdict_policy=_policy,
            )
        else:
            critic_backend = CriticAgentBackend(
                critic_agent_root=critic_agent_root,
                session_dir=session_dir,
                protocol="openai",
                codex_model=codex_model,
                action_verdict_policy=_policy,
            )

    return {
        _ORCHESTRATION_ROLE: _build_orchestration_backend(models, session_dir=session_dir),
        "critic": critic_backend,
    }


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


def _build_proposal_scorer(
    args: argparse.Namespace,
    session_dir: Path | None = None,
) -> ProposalScorer | None:
    """Construct the advisory specialist-proposal scorer, or ``None``."""
    if not getattr(args, "proposal_scoring", False):
        return None
    if llm_config.is_anthropic_only():
        # ProposalScorer is OpenAI-compatible only.
        return None
    models = tuple(m for m in (s.strip() for s in str(args.proposal_scorer_models).split(",")) if m)
    if not models:
        return None
    return ProposalScorer(models=models, session_dir=session_dir)
