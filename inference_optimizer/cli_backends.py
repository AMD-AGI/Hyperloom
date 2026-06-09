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

from .orchestrator.backends import (
    ClaudeBackend,
    CodexBackend,
    CriticAgentBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    RobustnessAgentBackend,
)
from .orchestrator.proposal_scorer import DEFAULT_SCORER_MODELS, ProposalScorer


def _build_backends(
    *,
    claude_model: str,
    codex_model: str,
    kernel_codex: bool,
    critic_choice: str,
    session_dir: Path,
    critic_agent_root: Path | None = None,
    critic_kb_mode: str = "inmemory",
    robustness_choice: str = "mock",
    robustness_agent_root: Path | None = None,
    robustness_options: dict[str, Any] | None = None,
    no_kernel: bool = False,
) -> dict[str, Any]:
    """Construct all per-role backends.

    ``critic_choice`` ∈ {``mock``, ``agent``}: mock is the always-approve
    adapter; ``agent`` is :class:`CriticAgentBackend` (requires
    ``critic_agent_root``). ``robustness_choice`` ∈ {``mock``, ``agent``}:
    mock is the heartbeat-only backend; ``agent`` is
    :class:`RobustnessAgentBackend` (requires ``robustness_agent_root``).
    """
    if critic_choice not in ("mock", "agent"):
        raise ValueError(
            f"_build_backends: critic_choice={critic_choice!r} not in "
            "{'mock','agent'}"
        )

    if critic_choice == "mock":
        critic_backend: Any = MockCriticBackend()
    else:  # "agent"
        if critic_agent_root is None:
            raise ValueError(
                "_build_backends: critic_choice='agent' requires critic_agent_root"
            )
        # Feed the registry-derived per-action verdict policy so the
        # critic-agent runtime sees
        # ``review_constraints.action_verdict_policy[<action_name>]`` and
        # approves exploration / archival actions without demanding the
        # before/after evidence they themselves produce.
        try:
            from inference_optimizer.orchestrator.action_registry import (
                ActionRegistry,
            )
            _reg = ActionRegistry().load()
            _policy = {a.name: a.verdict_class for a in _reg.all()}
        except Exception:  # noqa: BLE001 — degrade to empty policy
            _policy = {}
        critic_backend = CriticAgentBackend(
            critic_agent_root=critic_agent_root,
            session_dir=session_dir,
            codex_model=codex_model,
            kb_mode=critic_kb_mode,
            action_verdict_policy=_policy,
        )

    if robustness_choice not in ("mock", "agent"):
        raise ValueError(
            f"_build_backends: robustness_choice={robustness_choice!r} not in "
            "{'mock','agent'}"
        )
    if robustness_choice == "mock":
        robustness_backend: Any = MockRobustnessBackend()
    else:  # "agent"
        if robustness_agent_root is None:
            raise ValueError(
                "_build_backends: robustness_choice='agent' requires "
                "robustness_agent_root"
            )
        robustness_backend = RobustnessAgentBackend(
            robustness_agent_root=robustness_agent_root,
            session_dir=session_dir,
            options=robustness_options,
        )

    backends: dict[str, Any] = {
        # Orchestration runs as a persistent ReAct conversation (plan
        # Step 1): the same Claude session is resumed across ticks so the
        # model's plan / chain-of-thought persists instead of being
        # re-derived from a full state dump each turn. The conversational
        # floors (max_turns / call_timeout) are applied inside
        # ClaudeBackend.__post_init__. kernel / critic / robustness keep
        # the stateless per-tick reactor mode.
        "orchestration": ClaudeBackend(
            model=claude_model, max_turns_default=4, conversational=True,
        ),
        "critic":        critic_backend,
        "robustness":    robustness_backend,
    }
    if not no_kernel:
        if kernel_codex:
            backends["kernel"] = CodexBackend(model=codex_model)
        else:
            kernel_max_turns = int(
                os.environ.get("HYPERLOOM_KERNEL_MAX_TURNS", "40") or "40"
            )
            backends["kernel"] = ClaudeBackend(
                model=claude_model, max_turns_default=kernel_max_turns
            )
    return backends


def _build_proposal_scorer(
    args: argparse.Namespace,
) -> ProposalScorer | None:
    """Construct the advisory specialist-proposal scorer, or ``None``.

    Returns ``None`` when ``--no-proposal-scoring`` is set or the resolved
    model list is empty (defaults to :data:`DEFAULT_SCORER_MODELS`). The
    scorer is purely advisory and never gates anything.
    """
    if getattr(args, "no_proposal_scoring", False):
        return None
    raw = getattr(args, "proposal_scorer_models", None)
    if raw is None:
        models = tuple(DEFAULT_SCORER_MODELS)
    else:
        models = tuple(
            m for m in (s.strip() for s in str(raw).split(",")) if m
        )
    if not models:
        return None
    return ProposalScorer(models=models)


def _robustness_server_configured(args: argparse.Namespace) -> bool:
    """Return True when a robustness-server endpoint is configured.

    The server is the only cluster-wide signal source on multi-node; when
    wired the agent runs with ``disable_local_probe`` /
    ``enable_cluster_pod_metrics`` and the sandbox-local LocalProbe false
    positives are silenced. Configured = ``--robustness-server-url`` or
    ``ROBUSTNESS_SERVER_URL`` is set.
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

    Multi-node policy (``--nodes >= 2``): the agent must source signals
    from the cluster (robustness-server) rather than the local sandbox,
    else the per-pod LocalProbe trips false ``ray_head_dead`` /
    ``local_server_unreachable`` symptoms. We therefore default
    ``disable_local_probe`` + ``enable_cluster_pod_metrics`` to True,
    forward a workload_uid hint, disable the 127.0.0.1:8888 auto-probe,
    and lift the ``no_levers_found`` floor to 60 min. Single-node
    semantics stay untouched.
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

    if multi_node:
        # The inference server runs in the head pod, so the hardcoded
        # 127.0.0.1:8888 health probe can never succeed and would flood
        # the bus with false-positive ``local_server_unreachable``
        # symptoms each tick — disable the auto-probe in multi-node.
        options["auto_probe_inference_server"] = False
        # B3 no_levers_found floor — multi-node large-model spends
        # 35-50 min on sglang cold start + baseline + profile +
        # turnaround alone before the first explore family runs, so lift
        # the elapsed-time floor from 45 to 60 minutes (single-node
        # default 45.0 stays untouched).
        options["progress_no_levers_min_minutes"] = 60.0

    return options
