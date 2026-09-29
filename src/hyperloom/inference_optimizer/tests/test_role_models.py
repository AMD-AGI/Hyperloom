# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""--role-models: parsing, resolution, what a route does to the environment, and where it is applied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.role_models import (
    PLACEHOLDER_KEY,
    RoleModel,
    RoleModels,
    RoleModelsError,
    parse_role_models,
)
from hyperloom.inference_optimizer.cli import _resolve_role_models
from hyperloom.inference_optimizer.cli import backends as cli_backends
from hyperloom.orchestrator.phases.machine_state import PHASE_NAMES
from hyperloom.orchestrator.roles.claude import ClaudeBackend

GLM = {"model": "glm-5.3-flash", "base_url": "http://127.0.0.1:4000"}
LAUNCH_ENV = {
    "ANTHROPIC_BASE_URL": "https://gateway.example/anthropic",
    "ANTHROPIC_API_KEY": "launch-secret",
    "ANTHROPIC_AUTH_TOKEN": "launch-secret",
    "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: launch-secret",
    "OPENAI_API_KEY": "launch-secret",
    "PATH": "/usr/bin",
}


def _parse(spec: dict[str, Any]) -> RoleModels:
    return parse_role_models(json.dumps(spec), phases=PHASE_NAMES)


# --- parsing and resolution ------------------------------------------------------
def test_no_spec_routes_nothing():
    assert not parse_role_models(None)
    assert not parse_role_models("  ")
    assert RoleModels().resolve("orchestration", "KERNEL_AGENT") is None


def test_a_phase_route_wins_over_the_role_route(tmp_path):
    spec = {"orchestration": {"model": "opus-x"}, "orchestration@kernel_agent": GLM}
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(spec))
    rm = parse_role_models(str(path), phases=PHASE_NAMES)
    assert rm.resolve("orchestration", "KERNEL_AGENT").model == "glm-5.3-flash"
    assert rm.resolve("orchestration", "FRAMEWORK_AGENT").model == "opus-x"
    assert rm.resolve("critic") is None
    assert rm.has_role("orchestration") and not rm.has_role("critic")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"forge": GLM}, "unknown role"),
        ({"critic@KERNEL_AGENT": GLM}, "per role only"),
        ({"orchestration@NAPTIME": GLM}, "unknown phase"),
        ({"scorer": {**GLM, "protocol": "anthropic"}}, "speaks"),
        ({"critic": {"base_url": "http://x"}}, "needs an object with a 'model'"),
        ({"critic": {**GLM, "api_key": "sk-inline"}}, "unknown field"),
    ],
)
def test_specs_that_cannot_be_honoured_are_refused(spec, message):
    with pytest.raises(RoleModelsError, match=message):
        _parse(spec)


def test_the_cli_reads_the_flag_then_the_env(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ROLE_MODELS", json.dumps({"critic": GLM}))
    assert _resolve_role_models(argparse.Namespace(role_models_spec=None)).has_role("critic")
    flag = json.dumps({"scorer": {**GLM, "protocol": "openai"}})
    rm = _resolve_role_models(argparse.Namespace(role_models_spec=flag))
    assert rm.has_role("scorer") and not rm.has_role("critic")


# --- what a route does to the environment ------------------------------------------
def test_a_model_only_route_keeps_the_launch_endpoint_and_key():
    assert RoleModel(model="opus-x").env(LAUNCH_ENV) == LAUNCH_ENV


def test_a_route_elsewhere_never_carries_the_launch_credential():
    env = RoleModel(**GLM).env(LAUNCH_ENV)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert env["ANTHROPIC_API_KEY"] == env["ANTHROPIC_AUTH_TOKEN"] == PLACEHOLDER_KEY
    assert "launch-secret" not in json.dumps(env)
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == env["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-5.3-flash"
    assert env["PATH"] == "/usr/bin"
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in env, "it hides the SDK MCP tools"


def test_a_named_key_variable_is_used_and_must_be_set():
    route = RoleModel(**GLM, protocol="openai", api_key_env="GLM_KEY")
    env = route.env({**LAUNCH_ENV, "GLM_KEY": "glm-secret"})
    assert (env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"]) == ("http://127.0.0.1:4000", "glm-secret")
    with pytest.raises(RoleModelsError, match="GLM_KEY is not set"):
        route.env(LAUNCH_ENV)


def test_the_manifest_records_the_endpoint_host_and_never_a_key():
    described = _parse({"critic": {**GLM, "protocol": "openai", "api_key_env": "GLM_KEY"}}).describe()
    assert described == {
        "critic": {"model": "glm-5.3-flash", "protocol": "openai", "endpoint": "127.0.0.1", "api_key_env": "GLM_KEY"}
    }


# --- where routes are applied ---------------------------------------------------------
class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


async def _no_query(*, prompt, options):
    return
    yield


def test_orchestration_turns_follow_the_phase_route(monkeypatch):
    for name, value in LAUNCH_ENV.items():
        monkeypatch.setenv(name, value)
    backend = ClaudeBackend(
        model="claude-opus-5",
        sdk_query_factory=_no_query,
        sdk_options_cls=_Options,
        role_models=_parse({"orchestration@KERNEL_AGENT": GLM}),
        route_role="orchestration",
    )
    backend.set_route_phase("KERNEL_AGENT")
    kw = backend._build_options(tools=["Read"], max_turns=4, system_prompt="sp").kwargs
    assert kw["model"] == backend.turn_model == "glm-5.3-flash"
    assert kw["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    backend.set_route_phase("FRAMEWORK_AGENT")
    kw = backend._build_options(tools=["Read"], max_turns=4, system_prompt="sp").kwargs
    assert kw["model"] == backend.turn_model == "claude-opus-5"
    assert kw["env"]["ANTHROPIC_BASE_URL"] == "https://gateway.example/anthropic"


def test_backends_wire_the_orchestration_route_and_refuse_codex(tmp_path, monkeypatch):
    rm = _parse({"orchestration@KERNEL_AGENT": GLM})
    monkeypatch.setattr(cli_backends, "orchestration_runs_on_codex", lambda **_: False)
    monkeypatch.setattr(cli_backends, "ClaudeBackend", lambda **kw: kw)
    built = cli_backends._build_backends(
        claude_model="claude-opus-5", codex_model="gpt", critic_choice="mock", session_dir=tmp_path, role_models=rm
    )
    assert built["orchestration"]["model"] == "claude-opus-5"
    assert built["orchestration"]["role_models"] is rm and built["orchestration"]["route_role"] == "orchestration"
    monkeypatch.setattr(cli_backends, "orchestration_runs_on_codex", lambda **_: True)
    with pytest.raises(ValueError, match="Codex CLI"):
        cli_backends._build_backends(
            claude_model="m", codex_model="gpt", critic_choice="mock", session_dir=tmp_path, role_models=rm
        )


def test_a_critic_route_sets_protocol_model_and_env(tmp_path, monkeypatch):
    for name, value in LAUNCH_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli_backends, "orchestration_runs_on_codex", lambda **_: False)
    monkeypatch.setattr(cli_backends, "ClaudeBackend", lambda **kw: kw)
    monkeypatch.setattr(cli_backends, "CriticAgentBackend", lambda **kw: kw)
    built = cli_backends._build_backends(
        claude_model="claude-opus-5",
        codex_model="gpt-5.6-sol",
        critic_choice="agent",
        session_dir=tmp_path,
        critic_agent_root=tmp_path,
        role_models=_parse({"critic": {**GLM, "base_url": "http://127.0.0.1:4000/v1", "protocol": "openai"}}),
    )
    critic = built["critic"]
    assert (critic["protocol"], critic["codex_model"]) == ("openai", "glm-5.3-flash")
    assert critic["llm_env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:4000/v1"
    assert built["orchestration"]["model"] == "claude-opus-5", "a critic route leaves orchestration alone"


def test_a_scorer_route_replaces_the_rater_list(tmp_path, monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli_backends, "get_async_openai_client", lambda **kw: seen.update(kw) or "client")
    args = argparse.Namespace(
        proposal_scoring=True, role_models=_parse({"scorer": {**GLM, "base_url": "http://127.0.0.1:4000/v1"}})
    )
    scorer = cli_backends._build_proposal_scorer(args, tmp_path)
    assert scorer.models == ("glm-5.3-flash",)
    assert scorer._ensure_client() == "client"
    assert seen["env"]["OPENAI_BASE_URL"] == "http://127.0.0.1:4000/v1"


def test_the_coordinator_tells_the_backend_its_phase(tmp_path, monkeypatch):
    import asyncio

    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.inference_optimizer.session.paths import make_session_dir
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    phases: list[str] = []

    class _Routed(MockBackend):
        def set_route_phase(self, phase: str) -> None:
            phases.append(phase)

    plan = ScriptedPlan(turns=[], default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"body_md": "ok"}))

    async def _one_tick() -> None:
        c = Coordinator(make_session_dir(), backends={"orchestration": _Routed(plan), "critic": MockBackend(plan)})
        try:
            await c.tick(1)
        finally:
            await c.stop()

    asyncio.run(_one_tick())
    assert phases and phases[0] == "PRELUDE"


def test_path_not_json_is_a_clear_error(tmp_path):
    with pytest.raises(RoleModelsError, match="neither JSON nor a readable file"):
        parse_role_models(str(Path(tmp_path) / "missing.json"))
