# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``cli_backends``: per-role backend construction (mock/agent
choices, kernel selection, validation errors), advisory proposal-scorer
wiring and robustness option overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import backends as clib


def _clear_provider_env(monkeypatch) -> None:
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")),
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
        "LLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


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
        critic_choice="mock",
        session_dir=Path("/tmp/s"),
    )
    kwargs.update(over)
    return clib._build_backends(**kwargs)


@pytest.fixture(autouse=True)
def _isolated_provider_env(monkeypatch):
    """Every case in this module resolves backends from the environment, so the
    machine running the suite must not be able to change the answer. Applied to
    all of them, including the ones that assert a default."""
    _clear_provider_env(monkeypatch)


def test_build_backends_mock_defaults() -> None:
    b = _build()
    assert b["orchestration"][0] == "claude"
    assert b["critic"] == ("mock_critic",)
    assert b["robustness"] == ("mock_rob",)
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


def test_build_backends_anthropic_only_uses_native_critic_agent(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
    )
    assert b["orchestration"][0] == "claude"
    # Provider-only keeps the critic-agent on the native Anthropic protocol.
    assert b["critic"][0] == "critic_agent"
    assert b["critic"][1]["protocol"] == "anthropic"
    assert b["critic"][1]["claude_model"] == "claude-x"
    assert "kernel_agent" not in b


def test_build_backends_anthropic_only_refuses_to_degrade_without_root(monkeypatch) -> None:
    """Silently swapping the critic for bare tool-use would drop KB priors and
    session memory with no signal, so a missing root is now an error."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    with pytest.raises(ValueError, match="critic_agent_root"):
        _build(critic_choice="agent", critic_agent_root=None)


def test_build_backends_oauth_only_uses_anthropic_critic_protocol(monkeypatch) -> None:
    """A subscription token alone is enough to route the review over the CLI."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")), "sk-ant-oat01-fake")
    b = _build(critic_choice="agent", critic_agent_root=Path("/tmp/critic"))
    assert b["critic"][0] == "critic_agent"
    assert b["critic"][1]["protocol"] == "anthropic"


def test_build_backends_forced_anthropic_protocol_wins_over_dual_config(monkeypatch) -> None:
    """With both sides configured, auto picks openai; the flag must override it."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    auto = _build(critic_choice="agent", critic_agent_root=Path("/tmp/critic"))
    assert auto["critic"][1]["protocol"] == "openai"
    forced = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
        critic_protocol="anthropic",
    )
    assert forced["critic"][1]["protocol"] == "anthropic"


def test_build_backends_forced_protocol_without_credential_fails(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    with pytest.raises(ValueError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        _build(
            critic_choice="agent",
            critic_agent_root=Path("/tmp/critic"),
            critic_protocol="anthropic",
        )


def test_build_backends_forced_openai_protocol_accepts_gateway_key(monkeypatch) -> None:
    """The review client resolves LLM_GATEWAY_KEY, so the flag must accept a
    gateway-only host instead of rejecting a config that would have run."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "ak-gateway-key")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
        critic_protocol="openai",
    )
    assert b["critic"][1]["protocol"] == "openai"


def test_build_backends_forced_openai_protocol_accepts_an_anthropic_gateway(monkeypatch) -> None:
    """resolve_openai_client_config derives an OpenAI side from an Anthropic
    gateway -- one host, two protocols, one token. The gate must ask it rather
    than re-deriving a shorter key chain, which rejected this working config."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/anthropic")
    monkeypatch.setenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), "gateway-bearer")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
        critic_protocol="openai",
    )
    assert b["critic"][1]["protocol"] == "openai"


def test_build_backends_forced_openai_protocol_without_any_key_fails(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")), "sk-ant-oat01-fake")
    with pytest.raises(ValueError, match="LLM_GATEWAY_KEY"):
        _build(
            critic_choice="agent",
            critic_agent_root=Path("/tmp/critic"),
            critic_protocol="openai",
        )


def test_build_backends_forced_anthropic_protocol_accepts_a_subscription_token(monkeypatch) -> None:
    """The Claude CLI authenticates from the token alone, so the flag must
    accept it instead of rejecting a config that would have run."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-token")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
        critic_protocol="anthropic",
    )
    assert b["critic"][1]["protocol"] == "anthropic"


def test_build_backends_forced_openai_protocol_rejects_bare_gateway_key(monkeypatch) -> None:
    """A gateway key without OPENAI_BASE_URL would be sent to official OpenAI."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_GATEWAY_KEY", "ak-gateway-key")
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        _build(
            critic_choice="agent",
            critic_agent_root=Path("/tmp/critic"),
            critic_protocol="openai",
        )


def test_build_backends_rejects_unknown_critic_protocol(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    with pytest.raises(ValueError, match="critic_protocol"):
        _build(
            critic_choice="agent",
            critic_agent_root=Path("/tmp/critic"),
            critic_protocol="bogus",
        )


def test_build_backends_dual_protocol_gateway_uses_standard_critic_agent(monkeypatch) -> None:
    """A dual-protocol gateway (e.g. DeepSeek) is just "both sides configured".

    Once normalized it carries an Anthropic AND an OpenAI endpoint, so it takes
    the ordinary two-sided path rather than a provider-specific branch.
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-deepseek-key")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
    )
    assert b["critic"][0] == "critic_agent"
    # Both sides configured means auto lands on openai; the point of the test is
    # that the gateway takes that ordinary path, so the protocol is the assertion.
    assert b["critic"][1]["protocol"] == "openai"
    assert b["critic"][1]["codex_model"] == "codex-y"
    assert "codex_client_factory" not in b["critic"][1]


def test_backends_have_no_provider_specific_branch() -> None:
    """The retired DeepSeek branch and its hardcoded client factory are gone.

    Names are read off the module rather than asserted absent blindly: a typo
    in either string would make the old form of this test pass forever.
    """
    exported = set(vars(clib))
    assert "_deepseek_only" not in exported
    assert "_deepseek_openai_client_factory" not in exported
    # The guard against that typo: the symbol the branch was replaced by exists.
    assert "_resolve_critic_protocol" in exported


def test_build_backends_openai_only_uses_codex_for_orchestration(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    b = _build(
        critic_choice="agent",
        critic_agent_root=Path("/tmp/critic"),
    )
    assert b["orchestration"][0] == "codex"
    assert b["critic"][0] == "critic_agent"
    assert "kernel_agent" not in b


def test_build_backends_invalid_robustness_choice() -> None:
    with pytest.raises(ValueError, match="robustness_choice"):
        _build(robustness_choice="bogus")


def test_build_backends_robustness_agent_requires_root() -> None:
    with pytest.raises(ValueError, match="robustness_agent_root"):
        _build(robustness_choice="agent")


def test_build_backends_robustness_agent_with_root() -> None:
    b = _build(robustness_choice="agent", robustness_agent_root=Path("/tmp/rob"))
    assert b["robustness"][0] == "rob_agent"


def test_proposal_scorer_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    args = argparse.Namespace(proposal_scoring=False, proposal_scorer_models=None)
    assert clib._build_proposal_scorer(args) is None


def test_proposal_scorer_disabled_for_anthropic_only(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
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
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    default = parser.parse_args(["optimize", "--model", "x"])
    assert default.proposal_scoring is False
    enabled = parser.parse_args(["optimize", "--model", "x", "--proposal-scoring"])
    assert enabled.proposal_scoring is True
    disabled = parser.parse_args(["optimize", "--model", "x", "--no-proposal-scoring"])
    assert disabled.proposal_scoring is False


def test_retired_enable_proposal_scoring_flag_rejected(capsys) -> None:
    # The deprecation window is over: the flag is now an ordinary unknown argument.
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["optimize", "--model", "x", "--enable-proposal-scoring"])
    err = capsys.readouterr().err
    assert "unrecognized arguments: --enable-proposal-scoring" in err


def test_proposal_scorer_models_without_enable_stays_off(monkeypatch) -> None:
    # Passing only --proposal-scorer-models must not turn scoring on.
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    parser = _build_parser()
    args = parser.parse_args(["optimize", "--model", "x", "--proposal-scorer-models", "m1,m2"])
    assert args.proposal_scorer_models == "m1,m2"
    assert args.proposal_scoring is False
    assert clib._build_proposal_scorer(args) is None


def test_robustness_options_single_node_minimal() -> None:
    args = argparse.Namespace(
        robustness_llm_rca=None,
        nodes=1,
        robustness_disable_local_probe=None,
    )
    opts = clib._build_robustness_options(args)
    assert "auto_probe_inference_server" not in opts
    assert "nodes" not in opts


def test_robustness_options_multi_node_defaults() -> None:
    args = argparse.Namespace(
        robustness_llm_rca=True,
        nodes=4,
        robustness_disable_local_probe=None,
    )
    opts = clib._build_robustness_options(args)
    assert opts["nodes"] == 4
    assert opts["llm_rca_enabled"] is True
    assert opts["disable_local_probe"] is True
    assert opts["auto_probe_inference_server"] is False
    assert opts["progress_no_levers_min_minutes"] == 60.0


