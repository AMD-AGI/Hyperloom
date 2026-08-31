# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for gateway attribution headers.

The riskiest behaviour here is not the rendering but the merge: the variables
these headers travel in already carry gateway auth in production, so the tests
pin that an existing setting survives injection verbatim -- including a
``${VAR}`` reference, which ``codex_session`` must still be able to recognize
afterwards.

Shape tests register their own preset rather than leaning on ``litellm``, so they
describe the mechanism and do not have to change when a gateway's headers do.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hyperloom.common import llm_attribution
from hyperloom.common.llm_attribution import AttributionHeader
from hyperloom.common.llm_config import parse_custom_headers

_ATTR = llm_attribution.ATTRIBUTION_ENV
_CLAW = llm_attribution.CLAW_SESSION_ID_ENV
_ANTHROPIC = llm_attribution.ANTHROPIC_CUSTOM_HEADERS_ENV
_OPENAI = llm_attribution.OPENAI_CUSTOM_HEADERS_ENV


@pytest.fixture(autouse=True)
def _reset_published_phase() -> None:
    """Keep the process-wide phase from leaking between tests."""
    llm_attribution.set_current_phase("")


@pytest.fixture
def shapes(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Register a preset exercising every value shape; return its environment."""
    monkeypatch.setitem(
        llm_attribution.PRESETS,
        "shapes",
        (
            AttributionHeader("x-combined", "combined", ("session", "component", "phase")),
            AttributionHeader("x-raw", "raw", ("session", "component")),
            AttributionHeader("x-json", "json", ("session", "component")),
        ),
    )
    return {_ATTR: "shapes", _CLAW: "claw-abc"}


@pytest.fixture
def every_field(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Register a preset selecting all six fields; return its environment."""
    monkeypatch.setitem(
        llm_attribution.PRESETS,
        "every-field",
        (
            AttributionHeader(
                "x-all",
                "combined",
                ("application", "session", "component", "phase", "type", "operation"),
            ),
        ),
    )
    return {_ATTR: "every-field", _CLAW: "claw-abc"}


def _env(**overrides: str) -> dict[str, str]:
    """An environment selecting the LiteLLM preset with a session id."""
    return {_ATTR: "litellm", _CLAW: "claw-abc", **overrides}


class TestUnselected:
    """An operator who selects no gateway must observe no change at all."""

    def test_call_headers_empty_without_a_selection(self) -> None:
        assert llm_attribution.call_headers(component="geak", env={_CLAW: "claw-abc"}) == {}

    def test_inject_env_leaves_environment_untouched(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", source={_CLAW: "claw-abc"})
        assert env == {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}

    def test_unknown_gateway_emits_nothing(self) -> None:
        assert llm_attribution.call_headers(component="geak", env=_env(**{_ATTR: "nope"})) == {}


class TestValueShapes:
    """Gateways disagree about the shape of a value, not about its content."""

    def test_combined_renders_field_value_pairs(self, shapes: dict[str, str]) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="geak", env=shapes)
        assert headers["x-combined"] == "session=claw-abc,component=geak,phase=KERNEL_AGENT"

    def test_raw_omits_the_field_prefix(self, shapes: dict[str, str]) -> None:
        # An identifier slot wants the value itself, not "session=<id>".
        assert llm_attribution.call_headers(component="geak", env=shapes)["x-raw"] == "claw-abc"

    def test_raw_falls_through_to_the_first_field_with_a_value(self, shapes: dict[str, str]) -> None:
        shapes[_CLAW] = ""
        assert llm_attribution.call_headers(component="geak", env=shapes)["x-raw"] == "geak"

    def test_json_renders_a_parseable_object(self, shapes: dict[str, str]) -> None:
        headers = llm_attribution.call_headers(component="geak", env=shapes)
        assert json.loads(headers["x-json"]) == {"session": "claw-abc", "component": "geak"}

    def test_empty_fields_are_dropped(self, shapes: dict[str, str]) -> None:
        headers = llm_attribution.call_headers(component="geak", phase="", env=shapes)
        assert headers["x-combined"] == "session=claw-abc,component=geak"

    def test_a_header_with_no_values_is_not_emitted(self, shapes: dict[str, str]) -> None:
        shapes[_CLAW] = ""
        assert llm_attribution.call_headers(component="", phase="", env=shapes) == {}

    def test_extra_fields_keep_the_vocabulary_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            llm_attribution.PRESETS,
            "extras",
            (AttributionHeader("x-tags", "combined", ("component", "kernel_id")),),
        )
        headers = llm_attribution.call_headers(component="geak", kernel_id="k-7", env={_ATTR: "extras"})
        assert headers == {"x-tags": "component=geak,kernel_id=k-7"}


class TestValueHygiene:
    """A field value must not be able to break out of the header encoding."""

    def test_newlines_cannot_split_the_header_record(self) -> None:
        headers = llm_attribution.call_headers(component="geak\nX-Injected: 1", env=_env())
        assert "\n" not in headers["x-litellm-tags"]

    def test_dollar_is_stripped_so_values_cannot_be_re_expanded(self) -> None:
        # parse_custom_headers expands ${VAR}; a value must not reach it as one.
        headers = llm_attribution.call_headers(component="${SECRET}", env=_env())
        assert "$" not in headers["x-litellm-tags"]

    def test_a_value_cannot_forge_extra_tags(self) -> None:
        # The gateway splits this header on "," and "=": a value carrying either
        # would arrive as tags nobody wrote, under keys nobody chose.
        headers = llm_attribution.call_headers(component="geak,team=other", env=_env())
        assert "component=geak_team_other" in headers["x-litellm-tags"]

    def test_separators_are_replaced_so_distinct_values_stay_distinct(self) -> None:
        # Dropping them instead would collapse "a,b" and "ab" onto one tag, and
        # merge two producers' spend into a rollup belonging to neither.
        first = llm_attribution.call_headers(component="a,b", env=_env())
        second = llm_attribution.call_headers(component="ab", env=_env())
        assert first["x-litellm-tags"] != second["x-litellm-tags"]


class TestPresets:
    """Presets are the whole configuration surface, so they are checked here."""

    def test_litellm_emits_tags_and_a_trace_id(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        assert llm_attribution.call_headers(component="geak", env=_env()) == {
            "x-litellm-tags": ("application=hyperloom,session=claw-abc,component=geak,phase=KERNEL_AGENT"),
            "x-litellm-trace-id": "claw-abc",
        }

    def test_every_shipped_preset_emits_something(self) -> None:
        """Importing the module already validated these; this states the floor.

        The per-header rules are exercised below against deliberately bad
        entries. What is left to check here is that a preset exists at all, so
        an entry emptied by a refactor cannot pass validation by having nothing
        left to validate.
        """
        assert llm_attribution.PRESETS
        for gateway, headers in llm_attribution.PRESETS.items():
            assert headers, f"{gateway} preset emits nothing"

    @pytest.mark.parametrize(
        ("header", "reason"),
        [
            (AttributionHeader("x litellm tags", "combined", ("session",)), "not a TOML bare key"),
            (AttributionHeader("x-litellm-tags", "sentence", ("session",)), "unknown shape"),
            (AttributionHeader("x-litellm-tags", "combined", ()), "selects no fields"),
        ],
    )
    def test_an_unusable_preset_header_is_refused(self, header: AttributionHeader, reason: str) -> None:
        """Adding a gateway is the only way to reach these, so they fail loudly.

        Each of the three would otherwise surface far from its cause: a name
        Codex cannot write as a TOML key raises on the first Codex child, an
        unknown shape raises from inside rendering, and a header selecting
        nothing renders empty and is silently dropped.
        """
        with pytest.raises(ValueError, match=reason):
            llm_attribution._validate_presets({"acme": (header,)})


class TestMergePreservesExistingSetting:
    """Operators keep gateway auth in these variables; it must survive."""

    def test_existing_line_form_header_is_kept(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", source=_env())
        parsed = parse_custom_headers(env[_ANTHROPIC], env={})
        assert parsed["Ocp-Apim-Subscription-Key"] == "secret"
        assert parsed["x-litellm-tags"] == "application=hyperloom,session=claw-abc,component=geak"

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
        assert decoded["x-litellm-tags"] == "application=hyperloom,session=claw-abc,component=geak"

    def test_reinjection_replaces_instead_of_stacking(self) -> None:
        # The env hooks run once per turn, so this happens on every retry.
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="geak", phase="A", source=_env())
        llm_attribution.inject_env(env, component="geak", phase="B", source=_env())
        assert env[_ANTHROPIC].count("x-litellm-tags") == 1
        assert parse_custom_headers(env[_ANTHROPIC], env={})["x-litellm-tags"].endswith("phase=B")


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
        parsed = parse_custom_headers(env[_OPENAI], env={})
        assert parsed["X-Other"] == "1"
        assert "x-litellm-tags" in parsed

    def test_openai_variable_is_created_when_no_fallback_could_apply(self) -> None:
        env: dict[str, str] = {}
        llm_attribution.inject_env(env, component="geak", source=_env())
        assert "x-litellm-tags" in parse_custom_headers(env[_OPENAI], env={})


class TestPublishedPhase:
    """Spawn sites far from SharedState pick the phase up from the module."""

    def test_published_phase_is_used_when_the_call_site_omits_it(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="geak", env=_env())
        assert headers["x-litellm-tags"].endswith("phase=KERNEL_AGENT")

    def test_explicit_phase_overrides_the_published_one(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="critic", phase="COMMIT", env=_env())
        assert headers["x-litellm-tags"].endswith("phase=COMMIT")

    def test_explicit_empty_phase_suppresses_the_published_one(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        headers = llm_attribution.call_headers(component="critic", phase="", env=_env())
        assert "phase=" not in headers["x-litellm-tags"]

    def test_published_phase_reaches_a_spawned_child(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        child: dict[str, str] = {}
        llm_attribution.inject_env(child, component="geak", source=_env())
        tags = parse_custom_headers(child[_ANTHROPIC], env={})["x-litellm-tags"]
        assert tags.endswith("phase=KERNEL_AGENT")


class TestNestedInjectionRefines:
    """A child names itself and keeps the ambient fields it cannot restate.

    The spawn sites that need this run *inside* the child -- kernelforge drives
    the CLI from the forge loop, one process below whoever built its env -- and
    there the phase and the action are both empty, because one is a module
    global and the other a context variable. Without inheritance a child that
    injected would trade them for the component it gained.
    """

    #: A child environment as it reaches the second process: gateway auth the
    #: operator set, plus the tag its parent wrote on the way out.
    def _spawned(self, **parent: str) -> dict[str, str]:
        """Return a child env carrying a parent's tag, with this process reset."""
        child = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.set_current_phase("KERNEL_AGENT")
        llm_attribution.inject_env(
            child,
            component="forge",
            operation="forge_loop",
            source=_env(),
            **parent,
        )
        # The child is a new interpreter: no published phase, no session id, and
        # CLAW_SESSION_ID is on no env_safety allowlist so it does not travel.
        llm_attribution.set_current_phase("")
        return child

    def _tags(self, child: dict[str, str]) -> str:
        """The tag value the child would send."""
        return parse_custom_headers(child[_ANTHROPIC], env={})["x-litellm-tags"]

    def test_child_keeps_the_phase_it_could_not_restate(self) -> None:
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        assert "phase=KERNEL_AGENT" in self._tags(child)

    def test_child_keeps_the_session_its_environment_never_carried(self) -> None:
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        assert "session=claw-abc" in self._tags(child)

    def test_child_keeps_the_action_it_was_spawned_under(self) -> None:
        child = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        with llm_attribution.current_action_scope("kernel_opt"):
            llm_attribution.inject_env(child, component="forge", source=_env())
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        assert "type=kernel_opt" in self._tags(child)

    def test_the_child_names_itself(self) -> None:
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        tags = self._tags(child)
        assert "component=fusion" in tags
        assert "component=forge" not in tags

    def test_the_parents_purpose_does_not_leak_into_the_child(self) -> None:
        # Inheriting operation= would label the child's calls with work the
        # parent was doing, which is worse than saying nothing.
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        assert "operation=" not in self._tags(child)

    def test_the_child_states_its_own_purpose(self) -> None:
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", operation="discover", source={_ATTR: "litellm"})
        assert "operation=discover" in self._tags(child)

    def test_explicit_empty_phase_suppresses_an_inherited_one(self) -> None:
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", phase="", source={_ATTR: "litellm"})
        assert "phase=" not in self._tags(child)

    def test_gateway_auth_and_tag_count_survive_the_nesting(self) -> None:
        child = self._spawned()
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        assert parse_custom_headers(child[_ANTHROPIC], env={})["Ocp-Apim-Subscription-Key"] == "secret"
        assert child[_ANTHROPIC].count("x-litellm-tags") == 1

    def test_json_form_is_read_back_without_expanding_its_reference(self) -> None:
        child = {_ANTHROPIC: json.dumps({"Ocp-Apim-Subscription-Key": "${GATEWAY_KEY}"})}
        llm_attribution.set_current_phase("KERNEL_AGENT")
        llm_attribution.inject_env(child, component="forge", source=_env())
        llm_attribution.set_current_phase("")
        llm_attribution.inject_env(child, component="fusion", source={_ATTR: "litellm"})
        decoded = json.loads(child[_ANTHROPIC])
        assert decoded["Ocp-Apim-Subscription-Key"] == "${GATEWAY_KEY}"
        assert "phase=KERNEL_AGENT" in decoded["x-litellm-tags"]

    def test_nothing_is_inherited_when_no_parent_wrote_a_tag(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        assert self._tags(env) == "application=hyperloom,component=fusion"

    def test_an_operator_header_is_not_mistaken_for_a_tag(self) -> None:
        env = {_ANTHROPIC: "X-Other: application=theirs,phase=THEIRS"}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        assert "phase=" not in self._tags(env)

    def test_a_bare_trace_id_alone_is_not_taken_for_a_session(self) -> None:
        # Nothing is lost by declining it: a tag this module wrote always
        # carries the combined header, because application is never empty.
        env = {_ANTHROPIC: "x-litellm-trace-id: operator-trace"}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        assert "operator-trace" not in self._tags(env)

    def test_an_inherited_reference_is_never_left_to_be_expanded(self) -> None:
        # A recovered value is re-rendered into a tag this process sends, and
        # whoever reads that header next expands ${VAR} -- which would put this
        # process's gateway secret into a tag the gateway itself logs. Asserted
        # against the unexpanded setting, since reading it back expands it.
        env = {_ANTHROPIC: "x-litellm-tags: application=hyperloom,session=${GATEWAY_KEY}"}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        assert "${" not in env[_ANTHROPIC]

    def test_the_self_describing_tag_outranks_a_bare_trace_id(self) -> None:
        # x-litellm-trace-id carries a bare value, so an operator's own tracing
        # header is indistinguishable from ours; letting it win would make their
        # trace id the run's session and misjoin every reconciliation.
        tag = "x-litellm-tags: application=hyperloom,session=real-run"
        env = {_ANTHROPIC: f"{tag}\nx-litellm-trace-id: operator-trace"}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        assert "session=real-run" in self._tags(env)
        assert "operator-trace" not in self._tags(env)

    def test_an_explicitly_empty_field_suppresses_an_inherited_one(self) -> None:
        child = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        with llm_attribution.current_action_scope("kernel_opt"):
            llm_attribution.inject_env(child, component="forge", source=_env())
        assert "type=kernel_opt" in self._tags(child)

        # Overriding an inherited field is not the same as saying there is none.
        llm_attribution.inject_env(child, component="fusion", type="", source={_ATTR: "litellm"})
        assert "type=" not in self._tags(child)


class TestSelfInjectionDoesNotAccumulate:
    """Writing the tag into our own environment must not pin ambient state.

    ``forge_fusion`` injects into ``os.environ`` itself so the CLI it spawns
    later inherits the tag. That tag then outlives every scope in this process,
    so reading our own ``phase``/``type`` back out of it would label unrelated
    later calls with the first action the process happened to run.
    """

    def _tags(self, env: dict[str, str]) -> str:
        return parse_custom_headers(env[_ANTHROPIC], env={})["x-litellm-tags"]

    def test_a_stale_action_does_not_survive_its_scope(self) -> None:
        own = dict(_env())
        with llm_attribution.current_action_scope("kernel_opt"):
            llm_attribution.inject_env(own, component="forge", source=own)
        assert "type=kernel_opt" in self._tags(own)

        llm_attribution.inject_env(own, component="forge", source=own)
        assert "type=" not in self._tags(own)

    def test_a_stale_phase_does_not_survive_the_transition(self) -> None:
        own = dict(_env())
        llm_attribution.set_current_phase("KERNEL_AGENT")
        llm_attribution.inject_env(own, component="forge", source=own)
        assert "phase=KERNEL_AGENT" in self._tags(own)

        llm_attribution.set_current_phase("VALIDATE")
        llm_attribution.inject_env(own, component="forge", source=own)
        tags = self._tags(own)
        assert "phase=VALIDATE" in tags
        assert "KERNEL_AGENT" not in tags

    def test_the_run_identity_still_survives(self) -> None:
        # session identifies the run, not this process's state, so it is kept.
        own = dict(_env())
        llm_attribution.inject_env(own, component="forge", source=own)
        assert "session=claw-abc" in self._tags(own)


class TestInheritanceReadsBothVariables:
    """The two header variables are read together, not one instead of the other."""

    def _tags(self, env: dict[str, str]) -> str:
        return parse_custom_headers(env[_ANTHROPIC], env={})["x-litellm-tags"]

    def test_a_partly_written_variable_does_not_hide_the_other(self) -> None:
        # Reading only the first variable that parses would drop exactly the
        # ambient fields inheritance exists to carry.
        env = {
            _ANTHROPIC: "x-litellm-tags: application=hyperloom,session=sess-1",
            _OPENAI: "x-litellm-tags: application=hyperloom,session=sess-1,phase=KERNEL_AGENT,type=kernel_opt",
        }
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        tags = self._tags(env)
        assert "phase=KERNEL_AGENT" in tags
        assert "type=kernel_opt" in tags


class TestNoGatewaySelected:
    """The no-op contract: an unconfigured deployment is not read at all."""

    def test_a_malformed_selection_writes_nothing(self) -> None:
        env = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "not-a-preset"})
        assert env == {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}

    def test_the_json_encoding_survives_an_injection(self) -> None:
        # One arbiter decides the encoding for both reading and writing, so a
        # setting stored as JSON is still JSON afterwards.
        env = {_ANTHROPIC: json.dumps({"Ocp-Apim-Subscription-Key": "secret"})}
        llm_attribution.inject_env(env, component="fusion", source={_ATTR: "litellm"})
        decoded = json.loads(env[_ANTHROPIC])
        assert decoded["Ocp-Apim-Subscription-Key"] == "secret"
        assert decoded["x-litellm-tags"] == "application=hyperloom,component=fusion"


class TestSdkEnvOverlay:
    """claude_agent_sdk merges options.env over the inherited environment."""

    def test_overlay_is_empty_when_no_gateway_is_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ATTR, raising=False)
        assert llm_attribution.sdk_env_overlay(component="framework") == {}

    def test_overlay_joins_the_gateway_header_instead_of_replacing_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A bare tag here would shadow the inherited auth header in the child.
        monkeypatch.setenv(_ATTR, "litellm")
        monkeypatch.setenv(_CLAW, "claw-abc")
        monkeypatch.setenv(_ANTHROPIC, "Ocp-Apim-Subscription-Key: secret")
        overlay = llm_attribution.sdk_env_overlay(component="framework")
        parsed = parse_custom_headers(overlay[_ANTHROPIC], env={})
        assert parsed["Ocp-Apim-Subscription-Key"] == "secret"
        assert parsed["x-litellm-tags"] == "application=hyperloom,session=claw-abc,component=framework"

    def test_overlay_reports_only_variables_it_changes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ATTR, "litellm")
        monkeypatch.setenv(_CLAW, "claw-abc")
        monkeypatch.setenv(_ANTHROPIC, "Ocp-Apim-Subscription-Key: secret")
        monkeypatch.delenv(_OPENAI, raising=False)
        assert set(llm_attribution.sdk_env_overlay(component="framework")) == {_ANTHROPIC}


class TestConfigurationIsReadFromTheParent:
    """A child env is often an allowlisted subset carrying neither setting."""

    def test_selection_and_session_come_from_source_not_from_the_child_env(self) -> None:
        child: dict[str, str] = {_ANTHROPIC: "Ocp-Apim-Subscription-Key: secret"}
        llm_attribution.inject_env(child, component="specialist", source=_env())
        parsed = parse_custom_headers(child[_ANTHROPIC], env={})
        assert parsed["x-litellm-tags"] == "application=hyperloom,session=claw-abc,component=specialist"


class TestActionScope:
    """``type`` names the action executing, which is a concurrent scope."""

    def test_the_action_labels_calls_made_inside_it(self, every_field: dict[str, str]) -> None:
        with llm_attribution.current_action_scope("kernel_opt"):
            headers = llm_attribution.call_headers(component="geak", env=every_field)
        assert "type=kernel_opt" in headers["x-all"]

    def test_no_action_label_outside_an_action(self, every_field: dict[str, str]) -> None:
        # The orchestration call that *chooses* the action runs here, and
        # claiming it belonged to the previous action would misreport it.
        assert "type=" not in llm_attribution.call_headers(component="orchestration", env=every_field)["x-all"]

    def test_the_label_is_dropped_once_the_action_returns(self, every_field: dict[str, str]) -> None:
        with llm_attribution.current_action_scope("baseline"):
            pass
        assert "type=" not in llm_attribution.call_headers(component="geak", env=every_field)["x-all"]

    def test_a_nested_action_restores_the_enclosing_one(self, every_field: dict[str, str]) -> None:
        with llm_attribution.current_action_scope("kernel_opt"):
            with llm_attribution.current_action_scope("integrate"):
                pass
            headers = llm_attribution.call_headers(component="geak", env=every_field)
        assert "type=kernel_opt" in headers["x-all"]

    async def test_concurrent_actions_never_see_each_others_label(self, every_field: dict[str, str]) -> None:
        """The reason this is a context variable and not a module global.

        Both actions are inside their scope before either reads, so a
        process-wide value would hand both of them whichever ran last.
        """
        seen: dict[str, str] = {}
        entered = {"baseline": asyncio.Event(), "kernel_opt": asyncio.Event()}

        async def act(action: str, sibling: str) -> None:
            with llm_attribution.current_action_scope(action):
                entered[action].set()
                await entered[sibling].wait()
                seen[action] = llm_attribution.call_headers(component="specialist", env=every_field)["x-all"]

        await asyncio.gather(act("baseline", "kernel_opt"), act("kernel_opt", "baseline"))
        assert "type=baseline" in seen["baseline"]
        assert "type=kernel_opt" in seen["kernel_opt"]

    async def test_a_child_environment_carries_the_running_action(self, every_field: dict[str, str]) -> None:
        # GEAK and the specialists are spawned as children, so the label has to
        # survive the env hand-off rather than only the in-process path.
        child: dict[str, str] = {}
        with llm_attribution.current_action_scope("kernel_opt"):
            llm_attribution.inject_env(child, component="geak", source=every_field)
        assert "type=kernel_opt" in parse_custom_headers(child[_ANTHROPIC], env={})["x-all"]


class TestOperation:
    """``operation`` is the one field only the call site can supply."""

    def test_it_reaches_the_header(self, every_field: dict[str, str]) -> None:
        headers = llm_attribution.call_headers(component="critic", operation="review", env=every_field)
        assert "operation=review" in headers["x-all"]

    def test_it_is_dropped_when_the_call_site_names_none(self, every_field: dict[str, str]) -> None:
        assert "operation=" not in llm_attribution.call_headers(component="critic", env=every_field)["x-all"]

    def test_all_six_fields_narrow_from_the_product_to_the_call(self, every_field: dict[str, str]) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        with llm_attribution.current_action_scope("kernel_opt"):
            headers = llm_attribution.call_headers(
                component="geak",
                operation="optimize_kernel",
                env=every_field,
            )
        assert headers["x-all"] == (
            "application=hyperloom,session=claw-abc,component=geak,"
            "phase=KERNEL_AGENT,type=kernel_opt,operation=optimize_kernel"
        )

    def test_the_litellm_preset_carries_application(self) -> None:
        headers = llm_attribution.call_headers(component="geak", env=_env())
        assert headers["x-litellm-tags"].startswith("application=hyperloom,")

    def test_the_litellm_preset_carries_both_new_fields(self) -> None:
        llm_attribution.set_current_phase("SWEEP")
        with llm_attribution.current_action_scope("conc_sweep"):
            headers = llm_attribution.call_headers(component="orchestration", operation="rank_candidates", env=_env())
        assert headers["x-litellm-tags"].endswith("type=conc_sweep,operation=rank_candidates")
