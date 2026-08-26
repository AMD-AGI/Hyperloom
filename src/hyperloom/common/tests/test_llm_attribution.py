# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the operator-configured LLM attribution header.

The riskiest behaviour here is not the rendering but the merge: these variables
already carry gateway auth in production, so the tests pin that an existing
setting survives injection verbatim -- including a ``${VAR}`` reference, which
``codex_session`` must still be able to recognize afterwards.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

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


class TestSdkEnvOverlay:
    """claude_agent_sdk merges options.env over the inherited environment."""

    def test_overlay_is_empty_when_unconfigured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_SPEC, raising=False)
        assert llm_attribution.sdk_env_overlay(component="framework") == {}

    def test_overlay_joins_the_gateway_header_instead_of_replacing_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A bare tag here would shadow the inherited auth header in the child.
        monkeypatch.setenv(_SPEC, "x-tags:component")
        monkeypatch.setenv(_ANTHROPIC, "Ocp-Apim-Subscription-Key: secret")
        overlay = llm_attribution.sdk_env_overlay(component="framework")
        parsed = parse_custom_headers(overlay[_ANTHROPIC], env={})
        assert parsed["Ocp-Apim-Subscription-Key"] == "secret"
        assert parsed["x-tags"] == "component=framework"

    def test_overlay_reports_only_variables_it_changes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_SPEC, "x-tags:component")
        monkeypatch.setenv(_ANTHROPIC, "Ocp-Apim-Subscription-Key: secret")
        monkeypatch.delenv(_OPENAI, raising=False)
        assert set(llm_attribution.sdk_env_overlay(component="framework")) == {_ANTHROPIC}


#: Entry points that only tag a call when the caller names a component, so an
#: untagged call site is spend the gateway cannot attribute to anything.
_TAGGED_ENTRY_POINTS = frozenset(
    {
        "achat_completion",
        "aanthropic_completion",
        "aanthropic_messages",
        "anthropic_completion",
        "anthropic_messages",
        "astream_chat_completion_text",
        "chat_completion",
        "claude_sdk_env_options",
        "stream_chat_completion_text",
    }
)

_SRC_ROOT = Path(__file__).resolve().parents[2]


def _scan_call_sites() -> tuple[int, list[str]]:
    """Find production calls to a tagged entry point that omit ``component``.

    Returns:
        The number of call sites seen and one line per offender. The count is
        reported so the guard cannot quietly pass by finding nothing.
    """
    seen = 0
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT)
        if any(part in {"tests", "test"} for part in relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our files to fix
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in _TAGGED_ENTRY_POINTS:
                continue
            seen += 1
            if not any(keyword.arg == "component" for keyword in node.keywords):
                offenders.append(f"{relative}:{node.lineno}: {name}(...) has no component=")
    return seen, offenders


def test_every_llm_entry_point_call_names_its_component() -> None:
    """Every production LLM call must say who is spending.

    This is the coverage half of the feature: the module can render a header,
    but a call site that never names a component silently drops out of gateway
    attribution, which is the accounting gap this exists to close.
    """
    seen, offenders = _scan_call_sites()
    assert not offenders, "untagged LLM call sites:\n" + "\n".join(offenders)
    # Guard the guard: a rename that emptied _TAGGED_ENTRY_POINTS would
    # otherwise turn this into a test that always passes.
    assert seen >= 15, f"only {seen} LLM call sites found; the scan is no longer finding them"


class TestConfigurationIsReadFromTheParent:
    """A child env is often an allowlisted subset carrying neither setting."""

    def test_spec_and_session_come_from_source_not_from_the_child_env(self) -> None:
        child: dict[str, str] = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(child, component="specialist", source=_env())
        assert parse_custom_headers(child[_ANTHROPIC], env={})["x-tags"] == (
            "session=claw-abc,component=specialist"
        )
