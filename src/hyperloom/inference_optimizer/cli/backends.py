# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-role backend construction + robustness option wiring for the CLI.

Builds the orchestration / critic / robustness / kernel backends, the
advisory proposal scorer, and robustness ``request.options`` overrides from
parsed CLI args. Imports orchestrator packages only (must not import ``cli``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .. import framework_registry
from hyperloom.common.env import env_bool
from hyperloom.orchestrator.roles import (
    ClaudeBackend,
    CodexBackend,
    CriticAgentBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    RobustnessAgentBackend,
)
from hyperloom.orchestrator.scoring.proposal_scorer import DEFAULT_SCORER_MODELS, ProposalScorer


_KERNEL_AGENT_DEFAULT_MAX_TURNS = 5


def _official_anthropic_only() -> bool:
    """True when only the Anthropic-side endpoint is available."""
    has_anthropic = bool(
        (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        or (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
        or (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        or (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    )
    has_openai = bool(
        (os.environ.get("OPENAI_BASE_URL") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    return has_anthropic and not has_openai


def _official_openai_only() -> bool:
    """True when only the OpenAI-side endpoint is available."""
    has_openai = bool(
        (os.environ.get("OPENAI_BASE_URL") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    has_anthropic = bool(
        (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        or (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
        or (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        or (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    )
    return has_openai and not has_anthropic


def _deepseek_only() -> bool:
    """True for a DeepSeek-keyed provider-only config with no Anthropic key.

    DeepSeek is natively OpenAI-compatible, so its critic review runs over the
    DeepSeek OpenAI endpoint. An explicit Anthropic key takes precedence.
    """
    has_deepseek = bool(
        (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        or (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    )
    has_anthropic_key = bool(
        (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        or (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
    )
    has_openai = bool(
        (os.environ.get("OPENAI_BASE_URL") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    return has_deepseek and not has_anthropic_key and not has_openai


def _deepseek_openai_client_factory() -> Any:
    """Build a factory that points the critic's OpenAI client at DeepSeek.

    DeepSeek exposes an OpenAI-compatible ``/chat/completions`` API, so the
    critic-agent's OpenAI review path works once the client base URL / key are
    set to DeepSeek, without mutating the process env. The OpenAI SDK appends
    the route to ``base_url`` verbatim, so the default carries the ``/v1``
    suffix; an explicit ``DEEPSEEK_BASE_URL`` is respected as-is.
    """
    base_url = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1").strip().rstrip("/")
    api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()

    def _factory() -> Any:
        from openai import AsyncOpenAI  # local import: keep module import-light

        return AsyncOpenAI(base_url=base_url, api_key=api_key)

    return _factory


def _load_action_verdict_policy() -> dict[str, str]:
    """Return the registry-derived per-action verdict policy (or empty on error).

    Feeds ``review_constraints.action_verdict_policy[<action_name>]`` so the
    critic-agent approves exploration / archival actions without demanding the
    before/after evidence they themselves produce.
    """
    try:
        from hyperloom.orchestrator.actions.registry import ActionRegistry

        _reg = ActionRegistry().load()
        return {a.name: a.verdict_class for a in _reg.all()}
    except Exception:  # noqa: BLE001 — degrade to empty policy
        return {}


def _resolve_kernel_agent_max_turns() -> int:
    """Resolve the kernel_agent Claude turn budget.

    Reads ``INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS`` (a positive int);
    falls back to ``_KERNEL_AGENT_DEFAULT_MAX_TURNS`` on unset/invalid/<=0.
    """
    raw = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS", "").strip()
    if not raw:
        return _KERNEL_AGENT_DEFAULT_MAX_TURNS
    try:
        val = int(raw)
    except ValueError:
        return _KERNEL_AGENT_DEFAULT_MAX_TURNS
    return val if val >= 1 else _KERNEL_AGENT_DEFAULT_MAX_TURNS



def _build_backends(
    *,
    claude_model: str,
    codex_model: str,
    kernel_codex: bool,
    critic_choice: str,
    session_dir: Path,
    critic_agent_root: Path | None = None,
    critic_kb_mode: str = "inmemory",
    cortex_kb_url: str | None = None,
    robustness_choice: str = "mock",
    robustness_agent_root: Path | None = None,
    robustness_options: dict[str, Any] | None = None,
    no_kernel: bool = False,
    codex_follows_claude: bool = False,
) -> dict[str, Any]:
    """Construct all per-role backends.

    ``critic_choice`` ∈ {``mock``, ``agent``}: mock is the always-approve
    adapter; ``agent`` is :class:`CriticAgentBackend` (requires
    ``critic_agent_root``). ``robustness_choice`` ∈ {``mock``, ``agent``}:
    mock is the heartbeat-only backend; ``agent`` is
    :class:`RobustnessAgentBackend` (requires ``robustness_agent_root``).

    Args:
        claude_model: Claude model id for the orchestration / kernel backends.
        codex_model: Codex model id for the kernel / critic backends.
        kernel_codex: Use a Codex backend for the kernel_agent role when ``True``.
        critic_choice: Critic backend selector (``mock`` or ``agent``).
        session_dir: Session directory passed to agent backends.
        critic_agent_root: Critic-agent root, required when
            ``critic_choice='agent'``.
        critic_kb_mode: Knowledge-base mode for the critic agent.
        cortex_kb_url: Optional Cortex KB URL for the critic agent.
        robustness_choice: Robustness backend selector (``mock`` or ``agent``).
        robustness_agent_root: Robustness-agent root, required when
            ``robustness_choice='agent'``.
        robustness_options: Optional ``request.options`` overrides for the
            robustness agent.
        no_kernel: Skip building the kernel backend when ``True``.

    Returns:
        A mapping of role name to its constructed backend.

    Raises:
        ValueError: If ``critic_choice`` / ``robustness_choice`` is invalid, or
            an ``agent`` choice is missing its required agent root.
    """
    if critic_choice not in ("mock", "agent"):
        raise ValueError(f"_build_backends: critic_choice={critic_choice!r} not in {{'mock','agent'}}")

    provider_anthropic_only = codex_follows_claude or _official_anthropic_only()
    provider_openai_only = (not codex_follows_claude) and _official_openai_only()

    if critic_choice == "mock":
        critic_backend: Any = MockCriticBackend()
    elif provider_anthropic_only and critic_agent_root is not None:
        # Provider-only: keep the full KB+tools critic-agent, driving review
        # inference over the native provider endpoint (DeepSeek OpenAI-compatible
        # or Anthropic /v1/messages).
        _policy = _load_action_verdict_policy()
        if _deepseek_only():
            critic_backend = CriticAgentBackend(
                critic_agent_root=critic_agent_root,
                session_dir=session_dir,
                protocol="openai",
                codex_model=claude_model,
                codex_client_factory=_deepseek_openai_client_factory(),
                kb_mode=critic_kb_mode,
                cortex_kb_url=cortex_kb_url,
                action_verdict_policy=_policy,
            )
        else:
            critic_backend = CriticAgentBackend(
                critic_agent_root=critic_agent_root,
                session_dir=session_dir,
                protocol="anthropic",
                claude_model=claude_model,
                codex_model=codex_model,
                kb_mode=critic_kb_mode,
                cortex_kb_url=cortex_kb_url,
                action_verdict_policy=_policy,
            )
    elif provider_anthropic_only:
        # Fallback: critic-agent runtime unresolved, degrade to Claude tool-use.
        critic_backend = ClaudeBackend(
            model=claude_model,
            max_turns_default=4,
        )
    else:  # "agent"
        if critic_agent_root is None:
            raise ValueError("_build_backends: critic_choice='agent' requires critic_agent_root")
        critic_backend = CriticAgentBackend(
            critic_agent_root=critic_agent_root,
            session_dir=session_dir,
            codex_model=codex_model,
            kb_mode=critic_kb_mode,
            cortex_kb_url=cortex_kb_url,
            action_verdict_policy=_load_action_verdict_policy(),
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
        # Official OpenAI has no Claude endpoint; use the Codex backend for
        # Orchestration so an OpenAI-only config can drive the coordinator.
        orchestration_backend: Any = CodexBackend(model=codex_model)
    else:
        # Orchestration runs as a persistent ReAct conversation: the Claude
        # session is resumed across ticks so the plan persists.
        orchestration_backend = ClaudeBackend(
            model=claude_model,
            max_turns_default=4,
            conversational=True,
        )

    backends: dict[str, Any] = {
        "orchestration": orchestration_backend,
        "critic": critic_backend,
        "robustness": robustness_backend,
    }
    if not no_kernel:
        if provider_anthropic_only:
            backends["kernel_agent"] = ClaudeBackend(
                model=claude_model,
                max_turns_default=_resolve_kernel_agent_max_turns(),
            )
        elif provider_openai_only or kernel_codex:
            backends["kernel_agent"] = CodexBackend(model=codex_model)
        else:
            # Opt-in (default off): resume the kernel Claude session across LLM
            # turns for prompt-cache continuity.
            backends["kernel_agent"] = ClaudeBackend(
                model=claude_model,
                max_turns_default=_resolve_kernel_agent_max_turns(),
                conversational=env_bool("INFERENCE_OPTIMIZER_KERNEL_CLAUDE_CONVERSATIONAL", False),
            )
    return backends


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
    persisted across ``--resume``.

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


def _robustness_server_configured(args: argparse.Namespace) -> bool:
    """Return True when a robustness-server endpoint is configured.

    The server is the only cluster-wide signal source on multi-node; when
    wired the agent runs with ``disable_local_probe`` /
    ``enable_cluster_pod_metrics`` and the sandbox-local LocalProbe false
    positives are silenced. Configured = ``--robustness-server-url`` or
    ``ROBUSTNESS_SERVER_URL`` is set.

    Args:
        args: Parsed CLI args carrying ``robustness_server_url``.

    Returns:
        ``True`` when a robustness-server endpoint is configured via flag or
        environment.
    """
    url = (getattr(args, "robustness_server_url", None) or "").strip()
    if url:
        return True
    return bool((os.environ.get("ROBUSTNESS_SERVER_URL") or "").strip())


_MULTI_NODE_WORKLOAD_UID_ENV_KEYS: tuple[str, ...] = (
    "ROBUSTNESS_WORKLOAD_UID",
    "CLAW_WORKLOAD_UID",
    "WORKLOAD_UID",
    "KUBE_WORKLOAD_UID",
    "RAY_JOB_ID",
)


def _build_robustness_options(args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-default ``request.options`` overrides from CLI flags.

    Only emits keys the operator actually passed so the runtime CLI falls
    back to its own defaults / env-discovery for the rest.

    Multi-node (``--nodes >= 2``): defaults ``disable_local_probe`` +
    ``enable_cluster_pod_metrics`` to True, forwards a workload_uid hint,
    disables the 127.0.0.1:8888 auto-probe, and lifts the ``no_levers_found``
    floor to 60 min so cluster-sourced signals replace the local sandbox.

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
    server_url = getattr(args, "robustness_server_url", None)
    if server_url is not None:
        options["robustness_server_url"] = server_url
    llm_rca = getattr(args, "robustness_llm_rca", None)
    if llm_rca is not None:
        options["llm_rca_enabled"] = bool(llm_rca)

    nodes = int(getattr(args, "nodes", 1) or 1)
    multi_node = nodes >= 2
    if nodes > 1:
        options["nodes"] = nodes

    workload_uid = (getattr(args, "robustness_workload_uid", None) or "").strip()
    if not workload_uid:
        for key in _MULTI_NODE_WORKLOAD_UID_ENV_KEYS:
            candidate = (os.environ.get(key) or "").strip()
            if candidate:
                workload_uid = candidate
                break
    if workload_uid:
        options["workload_uid"] = workload_uid

    disable_local = getattr(args, "robustness_disable_local_probe", None)
    if disable_local is None and multi_node:
        disable_local = True
    if disable_local is not None:
        options["disable_local_probe"] = bool(disable_local)

    enable_pod_metrics = getattr(args, "robustness_enable_cluster_pod_metrics", None)
    if enable_pod_metrics is None and multi_node:
        enable_pod_metrics = True
    if enable_pod_metrics is not None:
        options["enable_cluster_pod_metrics"] = bool(enable_pod_metrics)

    categories_raw = getattr(args, "robustness_pod_metrics_categories", None)
    if categories_raw:
        if isinstance(categories_raw, (list, tuple)):
            cat_iter = categories_raw
        else:
            cat_iter = str(categories_raw).split(",")
        cat_list = [c.strip() for c in cat_iter if str(c).strip()]
        if cat_list:
            options["pod_metrics_categories"] = cat_list

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
