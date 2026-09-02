# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-role backend construction + robustness option wiring for the CLI.

Builds the orchestration / critic / robustness / kernel backends, the
advisory proposal scorer, and robustness ``request.options`` overrides from
parsed CLI args. Imports orchestrator packages only (must not import ``cli``).
"""

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
    HermesBackend,
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
    """Pick the critic's review protocol and verify that side is configured.

    ``auto`` reproduces the historical derivation, with one widening: a
    subscription token now counts as a configured Anthropic side, so an
    oauth-only host resolves to ``anthropic`` instead of falling through to an
    OpenAI side it does not have. An explicit choice is a hard contract: a
    missing credential fails at startup instead of silently routing the review
    somewhere the operator did not ask for.

    Args:
        requested: One of :data:`CRITIC_PROTOCOL_CHOICES`.
        provider_anthropic_only: Whether only the Anthropic side is configured.

    Returns:
        Either ``"openai"`` or ``"anthropic"``.

    Raises:
        ValueError: If ``requested`` is unknown, or the forced side has no
            usable credential.
    """
    if requested not in CRITIC_PROTOCOL_CHOICES:
        raise ValueError(f"_build_backends: critic_protocol={requested!r} not in {set(CRITIC_PROTOCOL_CHOICES)}")

    if requested == "auto":
        return "anthropic" if provider_anthropic_only else "openai"

    if requested == "anthropic":
        # Asked through the registry rather than a local list of names, so a
        # newly recognized credential form is accepted here the moment it is
        # registered instead of being rejected by a copy nobody updated.
        if not has_anthropic_credential():
            raise ValueError(
                "--critic-protocol=anthropic requires one of " + " / ".join(llm_config.ANTHROPIC_CREDENTIAL_ENV_ORDER)
            )
        return requested

    # Ask the resolver the review client will actually use, rather than
    # re-deriving its key chain here. The local copy listed only OPENAI_API_KEY
    # and LLM_GATEWAY_KEY, so it rejected an Anthropic-only host whose gateway
    # token and derived base URL do configure this client -- a config that runs.
    if requested == "openai":
        try:
            resolved = llm_config.resolve_openai_client_config()
        except llm_config.LLMConfigError as exc:
            raise ValueError(f"--critic-protocol=openai requires an OpenAI-capable credential: {exc}") from exc
        # A gateway key is scoped to its gateway. Judged on the resolved base
        # URL, not on OPENAI_BASE_URL alone, so a URL derived from the Anthropic
        # side counts; with none the client would fall back to official OpenAI
        # and send the gateway key there.
        if resolved.base_url is None and not _any_env_set(("OPENAI_API_KEY",)):
            raise ValueError(
                "--critic-protocol=openai with only a gateway key requires a base URL "
                "(OPENAI_BASE_URL, or ANTHROPIC_BASE_URL to derive it from); otherwise "
                "the gateway key is sent to the official OpenAI endpoint"
            )
    return requested


def _load_action_verdict_policy() -> dict[str, str]:
    """Return the registry-derived per-action verdict policy (or empty on error).

    Feeds ``review_constraints.action_verdict_policy[<action_name>]`` so the
    critic-agent approves exploration / archival actions without demanding the
    before/after evidence they themselves produce.
    """
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
    """Construct all per-role backends.

    ``critic_choice`` ∈ {``mock``, ``agent``}: mock is the always-approve
    adapter; ``agent`` is :class:`CriticAgentBackend` (requires
    ``critic_agent_root``). ``robustness_choice`` ∈ {``mock``, ``agent``}:
    mock is the heartbeat-only backend; ``agent`` is
    :class:`RobustnessAgentBackend` (requires ``robustness_agent_root``).

    Args:
        claude_model: Claude model id for the orchestration backend.
        codex_model: Codex model id for the critic backend.
        critic_choice: Critic backend selector (``mock`` or ``agent``).
        session_dir: Session directory passed to agent backends.
        critic_agent_root: Critic-agent root, required when
            ``critic_choice='agent'``.
        critic_kb_mode: Knowledge-base mode for the critic agent.
        robustness_choice: Robustness backend selector (``mock`` or ``agent``).
        robustness_agent_root: Robustness-agent root, required when
            ``robustness_choice='agent'``.
        robustness_options: Optional ``request.options`` overrides for the
            robustness agent.
        codex_follows_claude: Whether the Codex model id follows the Claude one.
        critic_protocol: Review transport selector (``auto``, ``openai`` or
            ``anthropic``).

    Returns:
        A mapping of role name to its constructed backend.

    Raises:
        ValueError: If ``critic_choice`` / ``robustness_choice`` /
            ``critic_protocol`` is invalid, an ``agent`` choice is missing its
            required agent root, or a forced protocol has no credential.
    """
    if critic_choice not in ("mock", "agent"):
        raise ValueError(f"_build_backends: critic_choice={critic_choice!r} not in {{'mock','agent'}}")

    # The two operands differ only in when they were evaluated, and that is the
    # point: the caller samples this before _preflight() derives OPENAI_BASE_URL
    # from ANTHROPIC_BASE_URL. Re-deriving it here alone would read an
    # Anthropic-only deploy as two-sided, purely because preflight filled in the
    # endpoint. Only `auto` consults this; an explicit --critic-protocol wins.
    provider_anthropic_only = codex_follows_claude or _official_anthropic_only()
    provider_openai_only = (not codex_follows_claude) and (
        _official_openai_only() or os.environ.get("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX") == "1"
    )

    if critic_choice == "mock":
        critic_backend: Any = MockCriticBackend()
    else:  # "agent"
        # No degraded critic: dropping the runtime would silently discard KB
        # priors, session memory and reviewed_msg_ids dedupe. Operators who
        # genuinely want no review pass --critic-mock.
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

    requested_orchestration = os.environ.get("HYPERLOOM_AGENT_BACKEND", "auto").strip().lower() or "auto"
    if requested_orchestration not in {"auto", "claude", "codex", "hermes"}:
        raise ValueError("HYPERLOOM_AGENT_BACKEND must be one of: auto, claude, codex, hermes")

    intents = default_role_registry()["orchestration"].allowed_intents
    if requested_orchestration == "hermes":
        orchestration_backend: Any = HermesBackend(
            allowed_intents=intents,
            model=os.environ.get("HYPERLOOM_HERMES_MODEL", "").strip() or codex_model,
            profile=os.environ.get("HYPERLOOM_HERMES_PROFILE", "hyperloomfaithful").strip(),
            inference_provider=os.environ.get("HYPERLOOM_HERMES_PROVIDER", "openai-codex").strip(),
            hermes_bin=os.environ.get("HYPERLOOM_HERMES_BIN", "").strip(),
            cwd=agent_dir(session_dir, "orchestration") / "hermes_workspace",
        )
    elif requested_orchestration == "codex" or (requested_orchestration == "auto" and provider_openai_only):
        # Official OpenAI has no Claude endpoint; use the Codex backend for
        # Orchestration so an OpenAI-only config can drive the coordinator.
        # The role record supplies the intent set, so the enforced Codex
        # output schema and the Claude emit_intent tool describe one contract.
        orchestration_backend: Any = CodexBackend(
            allowed_intents=intents,
            model=codex_model,
            # Scratch only. The session directory itself stays outside the
            # sandbox's writable roots so the agent cannot rewrite session state.
            cwd=agent_dir(session_dir, "orchestration") / "codex_workspace",
        )
    else:
        # Orchestration runs as a persistent ReAct conversation: the Claude
        # session is resumed across ticks so the plan persists.
        orchestration_backend = ClaudeBackend(
            model=claude_model,
            max_turns_default=4,
            conversational=True,
            capture_turn_diagnostics=True,
            allowed_intents=intents,
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
    """Construct the advisory specialist-proposal scorer, or ``None``.

    Disabled by default: returns ``None`` unless ``--proposal-scoring`` is
    passed. Even when enabled, still returns ``None`` in Anthropic-only
    deployments (the scorer is OpenAI-compatible only) or when the resolved
    model list is empty (defaults to :data:`DEFAULT_SCORER_MODELS`). The
    scorer is purely advisory and never gates anything. The flag is not
    persisted across ``--resume-from``.

    ``session_dir`` is forwarded so the scorer can append its per-model
    token usage to the full-trace ledger; when omitted trace writes are skipped.

    Args:
        args: Parsed CLI args (``proposal_scoring`` /
            ``proposal_scorer_models``).
        session_dir: Optional session directory for token-usage trace writes.

    Returns:
        A configured :class:`ProposalScorer`, or ``None`` when scoring is
        disabled or no models resolve.
    """
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
    """Collect non-default ``request.options`` overrides from CLI flags.

    Only emits keys the operator actually passed so the runtime CLI falls
    back to its own defaults / env-discovery for the rest.

    Multi-node (``--nodes >= 2``): defaults ``disable_local_probe`` to True,
    disables the 127.0.0.1:8888 auto-probe, and lifts the ``no_levers_found``
    floor to 60 min. The sandbox-local probes would only see one pod, so the
    agent runs on the node-agnostic budget / progress / inbox signals instead.

    Single-node opt-in: ``--robustness-disable-server-probe`` sets
    ``auto_probe_inference_server=False`` to silence the 127.0.0.1:8888 /health
    probe while the rest of LocalProbe keeps running. Scriptable (server-less)
    frameworks (e.g. xDiT) default the same probe OFF.

    Args:
        args: Parsed CLI args carrying the robustness-related flags.

    Returns:
        The non-default ``request.options`` overrides derived from the flags
        (and multi-node policy); keys the operator did not set are omitted.
    """
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

    # ``auto_probe_inference_server`` controls the 127.0.0.1:8888 /health probe
    # in LocalProbe. Default it OFF for multi-node (server lives in the head pod)
    # and scriptable server-less frameworks; single-node operators opt in via
    # ``--robustness-disable-server-probe``. --framework wins, then $FRAMEWORK.
    fw = (getattr(args, "framework", None) or os.environ.get("FRAMEWORK", "")).strip()
    scriptable_fw = framework_registry.is_scriptable(fw) if fw else False
    disable_server_probe = getattr(args, "robustness_disable_server_probe", None)
    if disable_server_probe is None and (multi_node or scriptable_fw):
        disable_server_probe = True
    if disable_server_probe is not None:
        options["auto_probe_inference_server"] = not bool(disable_server_probe)

    if multi_node:
        # Lift the no_levers_found elapsed-time floor to 60 min for multi-node
        # (single-node default 45.0 stays untouched).
        options["progress_no_levers_min_minutes"] = 60.0

    return options


def resolve_robustness_options(args: argparse.Namespace, state: SharedState) -> dict[str, Any]:
    """Layer this launch's robustness flags over the mapping persisted at launch.

    :func:`_build_robustness_options` reads argv only, so a resume re-passing one
    ``--robustness-*`` flag would otherwise drop the rest to the runtime
    defaults. Per-key, so an unrelated flag cannot reopen a probe an earlier
    ``--robustness-disable-server-probe`` closed.

    Args:
        args: Parsed CLI args carrying the robustness flags.
        state: The SharedState carrying ``robustness_options`` from launch.

    Returns:
        The resolved ``request.options`` overrides.
    """
    return {**state.robustness_options, **_build_robustness_options(args)}
