# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the operator-configured LLM attribution header.

The riskiest behaviour here is not the rendering but the merge: these variables
already carry gateway auth in production, so the tests pin that an existing
setting survives injection verbatim -- including a ``${VAR}`` reference, which
``codex_session`` must still be able to recognize afterwards.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.common import llm_attribution
from hyperloom.common.llm_config import parse_custom_headers

_SPEC = llm_attribution.HEADER_SPEC_ENV
_CLAW = llm_attribution.CLAW_SESSION_ID_ENV
_ANTHROPIC = llm_attribution.ANTHROPIC_CUSTOM_HEADERS_ENV
_OPENAI = llm_attribution.OPENAI_CUSTOM_HEADERS_ENV


@pytest.fixture(autouse=True)
def _reset_published_phase() -> None:
    """Keep the process-wide phase from leaking between tests."""
    llm_attribution.set_current_phase("")


def _env(**overrides: str) -> dict[str, str]:
    """Build an environment carrying a configured spec plus a session id."""
    return {
        _SPEC: "x-tags:session,component,phase",
        _CLAW: "claw-abc",
        **overrides,
    }


class TestUnconfigured:
    """An operator who sets nothing must observe no change at all."""

    def test_call_headers_empty_without_spec(self) -> None:
        assert llm_attribution.call_headers(component="geak", env={_CLAW: "claw-abc"}) == {}

    def test_inject_env_leaves_environment_untouched(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", source={_CLAW: "claw-abc"})
        assert env == {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}

    def test_malformed_spec_is_ignored(self) -> None:
        # No colon, so no field list can be read.
        assert llm_attribution.call_headers(component="geak", env=_env(**{_SPEC: "x-tags"})) == {}

    def test_header_name_codex_would_reject_is_ignored(self) -> None:
        # resolve_codex_provider_config raises on a non-bare-key header name.
        env = _env(**{_SPEC: "x.tags:session"})
        assert llm_attribution.call_headers(component="geak", env=env) == {}


class TestRendering:
    """Field selection, ordering, and value hygiene."""

    def test_renders_selected_fields_in_spec_order(self) -> None:
        headers = llm_attribution.call_headers(component="geak", phase="KERNEL_AGENT", env=_env())
        assert headers == {"x-tags": "session=claw-abc,component=geak,phase=KERNEL_AGENT"}

    def test_unselected_fields_are_not_emitted(self) -> None:
        env = _env(**{_SPEC: "x-tags:component"})
        headers = llm_attribution.call_headers(component="geak", phase="KERNEL_AGENT", env=env)
        assert headers == {"x-tags": "component=geak"}

    def test_empty_fields_are_dropped(self) -> None:
        headers = llm_attribution.call_headers(component="geak", phase="", env=_env())
        assert headers == {"x-tags": "session=claw-abc,component=geak"}

    def test_no_header_when_every_selected_field_is_empty(self) -> None:
        env = _env(**{_SPEC: "x-tags:phase", _CLAW: ""})
        assert llm_attribution.call_headers(component="geak", phase="", env=env) == {}

    def test_extra_fields_keep_the_vocabulary_open(self) -> None:
        env = _env(**{_SPEC: "x-tags:component,kernel_id"})
        headers = llm_attribution.call_headers(component="geak", kernel_id="k-7", env=env)
        assert headers == {"x-tags": "component=geak,kernel_id=k-7"}

    def test_newlines_cannot_split_the_header_record(self) -> None:
        headers = llm_attribution.call_headers(component="geak\nX-Injected: 1", env=_env())
        assert "\n" not in headers["x-tags"]

    def test_dollar_is_stripped_so_values_cannot_be_re_expanded(self) -> None:
        # parse_custom_headers expands ${VAR}; a value must not reach it as one.
        headers = llm_attribution.call_headers(component="${SECRET}", env=_env())
        assert "$" not in headers["x-tags"]


class TestMergePreservesExistingSetting:
    """Operators keep gateway auth in these variables; it must survive."""

    def test_existing_line_form_header_is_kept(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", source=_env())
        parsed = parse_custom_headers(env[_ANTHROPIC], env={})
        assert parsed["Ocp-Apim-Subscription-Key"] == "secret"
        assert parsed["x-tags"] == "session=claw-abc,component=geak"

    def test_env_reference_is_not_expanded(self) -> None:
        # codex_session matches ${VAR} against the raw text to forward the
        # variable name instead of materializing the secret.
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: ${GATEWAY_KEY}"}
        llm_attribution.inject_env(env, component="geak", source=_env())
        assert "${GATEWAY_KEY}" in env[_ANTHROPIC]

    def test_json_form_stays_json_and_keeps_its_reference(self) -> None:
        env = {_ANTHROPIC: json.dumps({"Ocp-Apim-Subscription-Key": "${GATEWAY_KEY}"})}
        llm_attribution.inject_env(env, component="geak", source=_env())
        decoded = json.loads(env[_ANTHROPIC])
        assert decoded["Ocp-Apim-Subscription-Key"] == "${GATEWAY_KEY}"
        assert decoded["x-tags"] == "session=claw-abc,component=geak"

    def test_reinjection_replaces_instead_of_stacking(self) -> None:
        # The env hooks run once per turn, so this happens on every retry.
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", phase="A", source=_env())
        llm_attribution.inject_env(env, component="geak", phase="B", source=_env())
        assert env[_ANTHROPIC].count("x-tags") == 1
        assert parse_custom_headers(env[_ANTHROPIC], env={})["x-tags"].endswith("phase=B")


class TestOpenAIFallbackIsNotBroken:
    """resolve_openai_client_config reads Anthropic headers only while the
    OpenAI variable parses empty; creating it would drop gateway auth."""

    def test_openai_variable_is_not_created_when_it_would_end_the_fallback(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", source=_env())
        assert _OPENAI not in env

    def test_existing_openai_variable_is_still_enriched(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret", _OPENAI: "X-Other: 1"}
        llm_attribution.inject_env(env, component="geak", source=_env())
        assert parse_custom_headers(env[_OPENAI], env={})["X-Other"] == "1"
        assert "x-tags" in parse_custom_headers(env[_OPENAI], env={})

    def test_openai_variable_is_created_when_no_fallback_could_apply(self) -> None:
        env: dict[str, str] = {}
        llm_attribution.inject_env(env, component="geak", source=_env())
        assert "x-tags" in parse_custom_headers(env[_OPENAI], env={})


class TestPublishedPhase:
    """Spawn sites far from SharedState pick the phase up from the module."""

    def test_published_phase_is_used_when_the_call_site_omits_it(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="geak", env=_env())
        assert headers == {"x-tags": "session=claw-abc,component=geak,phase=KERNEL_AGENT"}

    def test_explicit_phase_overrides_the_published_one(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="critic", phase="COMMIT", env=_env())
        assert headers["x-tags"].endswith("phase=COMMIT")

    def test_explicit_empty_phase_suppresses_the_published_one(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="critic", phase="", env=_env())
        assert "phase=" not in headers["x-tags"]

    def test_published_phase_reaches_a_spawned_child(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        child: dict[str, str] = {}
        llm_attribution.inject_env(child, component="geak", source=_env())
        assert parse_custom_headers(child[_ANTHROPIC], env={})["x-tags"].endswith("phase=KERNEL_AGENT")


class TestConfigurationIsReadFromTheParent:
    """A child env is often an allowlisted subset carrying neither setting."""

    def test_spec_and_session_come_from_source_not_from_the_child_env(self) -> None:
        child: dict[str, str] = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(child, component="specialist", source=_env())
        assert parse_custom_headers(child[_ANTHROPIC], env={})["x-tags"] == (
            "session=claw-abc,component=specialist"
        )
