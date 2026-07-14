# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``cli_backends``: per-role backend construction (mock/agent
choices, kernel selection, validation errors), advisory proposal-scorer
wiring, robustness-server detection, and robustness option overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import backends as clib


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch):
    """Replace heavy backend classes with lightweight stubs (no SDK / network)."""
    monkeypatch.setattr(clib, "ClaudeBackend", lambda **kw: ("claude", kw))
    monkeypatch.setattr(clib, "CodexBackend", lambda **kw: ("codex", kw))
    monkeypatch.setattr(clib, "MockCriticBackend", lambda: ("mock_critic",))
    monkeypatch.setattr(clib, "MockRobustnessBackend", lambda: ("mock_rob",))
    monkeypatch.setattr(
        clib,
        "CriticAgentBackend",
        lambda **kw: ("critic_agent", kw),
    )
    monkeypatch.setattr(
        clib,
        "RobustnessAgentBackend",
        lambda **kw: ("rob_agent", kw),
    )


def _build(**over):
    kwargs = dict(
        claude_model="claude-x",
        codex_model="codex-y",
        kernel_codex=False,
        critic_choice="mock",
        session_dir=Path("/tmp/s"),
    )
    kwargs.update(over)
    return clib._build_backends(**kwargs)


def test_build_backends_mock_defaults_with_kernel_claude() -> None:
    b = _build()
    assert b["orchestration"][0] == "claude"
    assert b["critic"] == ("mock_critic",)
    assert b["robustness"] == ("mock_rob",)
    assert b["kernel_agent"][0] == "claude"


def test_build_backends_kernel_codex() -> None:
    b = _build(kernel_codex=True)
    assert b["kernel_agent"][0] == "codex"


def test_build_backends_no_kernel() -> None:
    b = _build(no_kernel=True)
    assert "kernel_agent" not in b


def test_build_backends_invalid_critic_choice() -> None:
    with pytest.raises(ValueError, match="critic_choice"):
        _build(critic_choice="bogus")


def test_build_backends_critic_agent_requires_root() -> None:
    with pytest.raises(ValueError, match="critic_agent_root"):
        _build(critic_choice="agent")


def test_build_backends_critic_agent_with_root() -> None:
    b = _build(critic_choice="agent", critic_agent_root=Path("/tmp/critic"))
    assert b["critic"][0] == "critic_agent"


def test_build_backends_anthropic_only_uses_claude_for_critic_and_kernel(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
        kernel_codex=True,
    )
    assert b["orchestration"][0] == "claude"
    assert b["critic"][0] == "claude"
    assert b["kernel_agent"][0] == "claude"


def test_build_backends_openai_only_uses_codex_for_orchestration_and_kernel(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
        kernel_codex=False,
    )
    assert b["orchestration"][0] == "codex"
    assert b["critic"][0] == "critic_agent"
    assert b["kernel_agent"][0] == "codex"


def test_build_backends_invalid_robustness_choice() -> None:
    with pytest.raises(ValueError, match="robustness_choice"):
        _build(robustness_choice="bogus")


def test_build_backends_robustness_agent_requires_root() -> None:
    with pytest.raises(ValueError, match="robustness_agent_root"):
        _build(robustness_choice="agent")


def test_build_backends_robustness_agent_with_root() -> None:
    b = _build(robustness_choice="agent", robustness_agent_root=Path("/tmp/rob"))
    assert b["robustness"][0] == "rob_agent"


# -- _build_proposal_scorer ------------------------------------------------
def test_proposal_scorer_disabled_by_default(monkeypatch) -> None:
    # Default is OFF: without --proposal-scoring the scorer is not built.
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    args = argparse.Namespace(proposal_scoring=False, proposal_scorer_models=None)
    assert clib._build_proposal_scorer(args) is None


def test_proposal_scorer_disabled_for_anthropic_only(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    args = argparse.Namespace(
        proposal_scoring=True,
        proposal_scorer_models=None,
    )
    assert clib._build_proposal_scorer(args) is None


def test_proposal_scorer_empty_models_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    args = argparse.Namespace(
        proposal_scoring=True,
        proposal_scorer_models="  ,  ",
    )
    assert clib._build_proposal_scorer(args) is None


def test_proposal_scorer_default_models(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(clib, "ProposalScorer", lambda **kw: ("scorer", kw))
    args = argparse.Namespace(proposal_scoring=True)
    out = clib._build_proposal_scorer(args)
    assert out[0] == "scorer"
    assert len(out[1]["models"]) >= 1


def test_proposal_scorer_explicit_models(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(clib, "ProposalScorer", lambda **kw: ("scorer", kw))
    args = argparse.Namespace(
        proposal_scoring=True,
        proposal_scorer_models="m1, m2",
    )
    out = clib._build_proposal_scorer(args)
    assert out[1]["models"] == ("m1", "m2")


def test_proposal_scoring_flag_parsing() -> None:
    # Parser-level contract: default OFF, --proposal-scoring on, --no- off.
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    default = parser.parse_args(["optimize", "--model", "x"])
    assert default.proposal_scoring is False
    enabled = parser.parse_args(["optimize", "--model", "x", "--proposal-scoring"])
    assert enabled.proposal_scoring is True
    disabled = parser.parse_args(["optimize", "--model", "x", "--no-proposal-scoring"])
    assert disabled.proposal_scoring is False


def test_proposal_scorer_models_without_enable_stays_off(monkeypatch) -> None:
    # Passing only --proposal-scorer-models (a non-empty list, which is also
    # the parser default) must NOT turn scoring on: default OFF requires an
    # explicit --proposal-scoring. Guards the silent-off regression.
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    parser = _build_parser()
    args = parser.parse_args(
        ["optimize", "--model", "x", "--proposal-scorer-models", "m1,m2"]
    )
    # Sanity: models parsed, but scoring stayed off.
    assert args.proposal_scorer_models == "m1,m2"
    assert args.proposal_scoring is False
    assert clib._build_proposal_scorer(args) is None


# -- _robustness_server_configured -----------------------------------------
def test_robustness_server_configured_via_arg() -> None:
    args = argparse.Namespace(robustness_server_url="http://rob:9000")
    assert clib._robustness_server_configured(args) is True


def test_robustness_server_configured_via_env(monkeypatch) -> None:
    monkeypatch.delenv("ROBUSTNESS_SERVER_URL", raising=False)
    args = argparse.Namespace(robustness_server_url=None)
    assert clib._robustness_server_configured(args) is False
    monkeypatch.setenv("ROBUSTNESS_SERVER_URL", "http://env:9000")
    assert clib._robustness_server_configured(args) is True


# -- _build_robustness_options ---------------------------------------------
def test_robustness_options_single_node_minimal(monkeypatch) -> None:
    for k in clib._MULTI_NODE_WORKLOAD_UID_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    args = argparse.Namespace(
        robustness_server_url=None,
        robustness_llm_rca=None,
        nodes=1,
        robustness_workload_uid=None,
        robustness_disable_local_probe=None,
        robustness_enable_cluster_pod_metrics=None,
        robustness_pod_metrics_categories=None,
    )
    opts = clib._build_robustness_options(args)
    # single-node: no multi-node defaults emitted
    assert "auto_probe_inference_server" not in opts
    assert "nodes" not in opts


def test_robustness_options_multi_node_defaults(monkeypatch) -> None:
    for k in clib._MULTI_NODE_WORKLOAD_UID_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    args = argparse.Namespace(
        robustness_server_url="http://rob",
        robustness_llm_rca=True,
        nodes=4,
        robustness_workload_uid="wl-1",
        robustness_disable_local_probe=None,
        robustness_enable_cluster_pod_metrics=None,
        robustness_pod_metrics_categories="gpu,net",
    )
    opts = clib._build_robustness_options(args)
    assert opts["nodes"] == 4
    assert opts["robustness_server_url"] == "http://rob"
    assert opts["llm_rca_enabled"] is True
    assert opts["workload_uid"] == "wl-1"
    assert opts["disable_local_probe"] is True
    assert opts["enable_cluster_pod_metrics"] is True
    assert opts["pod_metrics_categories"] == ["gpu", "net"]
    assert opts["auto_probe_inference_server"] is False
    assert opts["progress_no_levers_min_minutes"] == 60.0


def test_robustness_options_workload_uid_from_env(monkeypatch) -> None:
    for k in clib._MULTI_NODE_WORKLOAD_UID_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RAY_JOB_ID", "ray-42")
    args = argparse.Namespace(
        robustness_server_url=None,
        robustness_llm_rca=None,
        nodes=1,
        robustness_workload_uid=None,
        robustness_disable_local_probe=None,
        robustness_enable_cluster_pod_metrics=None,
        robustness_pod_metrics_categories=None,
    )
    opts = clib._build_robustness_options(args)
    assert opts["workload_uid"] == "ray-42"


# ---- _resolve_kernel_agent_max_turns ----
def test_kernel_agent_max_turns_default(monkeypatch):
    from hyperloom.inference_optimizer.cli import backends as cb
    monkeypatch.delenv("INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS", raising=False)
    assert cb._resolve_kernel_agent_max_turns() == 5


def test_kernel_agent_max_turns_env_override(monkeypatch):
    from hyperloom.inference_optimizer.cli import backends as cb
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS", "9")
    assert cb._resolve_kernel_agent_max_turns() == 9


def test_kernel_agent_max_turns_invalid_falls_back(monkeypatch):
    from hyperloom.inference_optimizer.cli import backends as cb
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS", "not-an-int")
    assert cb._resolve_kernel_agent_max_turns() == 5
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS", "0")
    assert cb._resolve_kernel_agent_max_turns() == 5
