# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-role backend construction + robustness option wiring for the CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import framework_registry
from hyperloom.common import llm_config
from hyperloom.common.llm_config import has_anthropic_credential
from hyperloom.inference_optimizer.session.session_paths import agent_dir
from hyperloom.orchestrator.roles import (
    ClaudeBackend,
    CodexBackend,
    CriticAgentBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    RobustnessAgentBackend,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.scoring.proposal_scorer import DEFAULT_SCORER_MODELS, ProposalScorer

if TYPE_CHECKING:
    from hyperloom.orchestrator.state.shared_state import SharedState

CRITIC_PROTOCOL_CHOICES: tuple[str, ...] = ("auto", "openai", "anthropic")


def _any_env_set(names: tuple[str, ...]) -> bool:
    return any((os.environ.get(name) or "").strip() for name in names)


def _official_anthropic_only() -> bool:
    """True when only the Anthropic-side endpoint is available."""
    from hyperloom.common import llm_config  # local import: keep module import-light

    return llm_config.is_anthropic_only()


def _official_openai_only() -> bool:
    """True when only the OpenAI-side endpoint is available."""
    from hyperloom.common import llm_config  # local import: keep module import-light

    return llm_config.is_openai_only()


def _resolve_critic_protocol(requested: str, *, provider_anthropic_only: bool) -> str:
    """Pick the critic's review protocol and verify that side is configured."""
    if requested not in CRITIC_PROTOCOL_CHOICES:
        raise ValueError(f"_build_backends: critic_protocol={requested!r} not in {set(CRITIC_PROTOCOL_CHOICES)}")

    if requested == "auto":
        return "anthropic" if provider_anthropic_only else "openai"

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


def _load_action_verdict_policy() -> dict[str, str]:
    """Return the registry-derived per-action verdict policy (or empty on error)."""
    from ..protocol.action_surfaces import ACTION_CATALOGUE

    return {a.name: a.verdict_class for a in ACTION_CATALOGUE.values()}


def _build_backends(
    *,
    claude_model: str,
    codex_model: str,
    critic_choice: str,
    session_dir: Path,
    critic_agent_root: Path | None = None,
    critic_kb_mode: str = "inmemory",
    robustness_choice: str = "mock",
    robustness_agent_root: Path | None = None,
    robustness_options: dict[str, Any] | None = None,
    codex_follows_claude: bool = False,
    critic_protocol: str = "auto",
) -> dict[str, Any]:
    """Construct all per-role backends."""
    if critic_choice not in ("mock", "agent"):
        raise ValueError(f"_build_backends: critic_choice={critic_choice!r} not in {{'mock','agent'}}")

    # The two operands differ only in when they were evaluated, and that is the point: the caller samples this before
    # _preflight() derives OPENAI_BASE_URL from ANTHROPIC_BASE_URL.
    provider_anthropic_only = codex_follows_claude or _official_anthropic_only()
    provider_openai_only = (not codex_follows_claude) and (
        _official_openai_only() or os.environ.get("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX") == "1"
    )

    if critic_choice == "mock":
        critic_backend: Any = MockCriticBackend()
    else:  # "agent"
        # No degraded critic: dropping the runtime would silently discard KB priors, session memory and
        # reviewed_msg_ids dedupe.
        if critic_agent_root is None:
            raise ValueError("_build_backends: critic_choice='agent' requires critic_agent_root")
        protocol = _resolve_critic_protocol(
            critic_protocol,
            provider_anthropic_only=provider_anthropic_only,
        )
        _policy = _load_action_verdict_policy()
        if protocol == "anthropic":
            critic_backend = CriticAgentBackend(
                critic_agent_root=critic_agent_root,
                session_dir=session_dir,
                protocol="anthropic",
                claude_model=claude_model,
                codex_model=codex_model,
                kb_mode=critic_kb_mode,
                action_verdict_policy=_policy,
            )
        else:
            critic_backend = CriticAgentBackend(
                critic_agent_root=critic_agent_root,
                session_dir=session_dir,
                protocol="openai",
                codex_model=codex_model,
                kb_mode=critic_kb_mode,
                action_verdict_policy=_policy,
            )

    if robustness_choice not in ("mock", "agent"):
        raise ValueError(f"_build_backends: robustness_choice={robustness_choice!r} not in {{'mock','agent'}}")
    if robustness_choice == "mock":
        robustness_backend: Any = MockRobustnessBackend()
    else:  # "agent"
        if robustness_agent_root is None:
            raise ValueError("_build_backends: robustness_choice='agent' requires robustness_agent_root")
        robustness_backend = RobustnessAgentBackend(
            robustness_agent_root=robustness_agent_root,
            session_dir=session_dir,
            options=robustness_options,
        )

    if provider_openai_only:
        # Official OpenAI has no Claude endpoint; use the Codex backend for Orchestration so an OpenAI-only config can
        # drive the coordinator.
        orchestration_backend: Any = CodexBackend(
            allowed_intents=default_role_registry()["orchestration"].allowed_intents,
            model=codex_model,
            # Scratch only.
            cwd=agent_dir(session_dir, "orchestration") / "codex_workspace",
        )
    else:
        # Orchestration is stateless: every tick opens a fresh session and
        # carries the full state projection, so it needs the agentic turn
        # budget and the orchestration effort tier.
        orchestration_backend = ClaudeBackend(
            model=claude_model,
            effort_role="orchestration",
            capture_turn_diagnostics=True,
            allowed_intents=default_role_registry()["orchestration"].allowed_intents,
        )

    return {
        "orchestration": orchestration_backend,
        "critic": critic_backend,
        "robustness": robustness_backend,
    }


def _build_proposal_scorer(
    args: argparse.Namespace,
    session_dir: Path | None = None,
) -> ProposalScorer | None:
    """Construct the advisory specialist-proposal scorer, or ``None``."""
    if not getattr(args, "proposal_scoring", False):
        return None
    if _official_anthropic_only():
        # ProposalScorer is OpenAI-compatible only.
        return None
    raw = getattr(args, "proposal_scorer_models", None)
    if raw is None:
        models = tuple(DEFAULT_SCORER_MODELS)
    else:
        models = tuple(m for m in (s.strip() for s in str(raw).split(",")) if m)
    if not models:
        return None
    return ProposalScorer(models=models, session_dir=session_dir)


def _build_robustness_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-default ``request.options`` overrides from CLI flags."""
    options: dict[str, Any] = {}
    llm_rca = getattr(args, "robustness_llm_rca", None)
    if llm_rca is not None:
        options["llm_rca_enabled"] = bool(llm_rca)

    nodes = int(getattr(args, "nodes", 1) or 1)
    multi_node = nodes >= 2
    if nodes > 1:
        options["nodes"] = nodes

    disable_local = getattr(args, "robustness_disable_local_probe", None)
    if disable_local is None and multi_node:
        disable_local = True
    if disable_local is not None:
        options["disable_local_probe"] = bool(disable_local)

    # ``auto_probe_inference_server`` controls the 127.0.0.1:8888 /health probe in LocalProbe.
    fw = (getattr(args, "framework", None) or os.environ.get("FRAMEWORK", "")).strip()
    scriptable_fw = framework_registry.is_scriptable(fw) if fw else False
    disable_server_probe = getattr(args, "robustness_disable_server_probe", None)
    if disable_server_probe is None and (multi_node or scriptable_fw):
        disable_server_probe = True
    if disable_server_probe is not None:
        options["auto_probe_inference_server"] = not bool(disable_server_probe)

    if multi_node:
        # Lift the no_levers_found elapsed-time floor to 60 min for multi-node (single-node default 45.0 stays
        # untouched).
        options["progress_no_levers_min_minutes"] = 60.0

    return options


def resolve_robustness_options(args: argparse.Namespace, state: SharedState) -> dict[str, Any]:
    """Layer this launch's robustness flags over the mapping persisted at launch."""
    return {**state.robustness_options, **_build_robustness_options(args)}
