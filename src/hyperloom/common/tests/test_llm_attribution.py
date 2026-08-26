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

import ast
import asyncio
import json
from pathlib import Path

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


class TestPresets:
    """Presets are the whole configuration surface, so they are checked here."""

    def test_litellm_emits_tags_and_a_trace_id(self) -> None:
        llm_attribution.set_current_phase("KERNEL_AGENT")
        assert llm_attribution.call_headers(component="geak", env=_env()) == {
            "x-litellm-tags": ("application=hyperloom,session=claw-abc,component=geak,phase=KERNEL_AGENT"),
            "x-litellm-trace-id": "claw-abc",
        }

    def test_every_preset_header_is_usable(self) -> None:
        for gateway, headers in llm_attribution.PRESETS.items():
            assert headers, f"{gateway} preset emits nothing"
            for header in headers:
                # Codex rejects a header name that is not a TOML bare key.
                assert llm_attribution._VALID_HEADER_NAME_RE.match(header.name), (
                    f"{gateway}: header name {header.name!r} is not a TOML bare key"
                )
                assert header.shape in llm_attribution._RENDERERS, f"{gateway}: unknown shape {header.shape!r}"
                assert header.fields, f"{gateway}: header {header.name!r} selects no fields"


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


def _scan_call_sites(field: str) -> tuple[int, list[str]]:
    """Find production calls to a tagged entry point that omit ``field``.

    Args:
        field: The keyword every call site is required to pass.

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
            if not any(keyword.arg == field for keyword in node.keywords):
                offenders.append(f"{relative}:{node.lineno}: {name}(...) has no {field}=")
    return seen, offenders


@pytest.mark.parametrize("field", ["component", "operation"])
def test_every_llm_entry_point_call_names_its_attribution(field: str) -> None:
    """Every production LLM call must say who is spending, and on what.

    This is the coverage half of the feature: the module can render a header,
    but a call site that names neither silently drops out of gateway
    attribution, which is the accounting gap this exists to close. ``component``
    and ``operation`` are the two the call site alone knows -- the rest are
    filled in from the run's own state.
    """
    seen, offenders = _scan_call_sites(field)
    assert not offenders, f"LLM call sites with no {field}:\n" + "\n".join(offenders)
    # Guard the guard: a rename that emptied _TAGGED_ENTRY_POINTS would
    # otherwise turn this into a test that always passes.
    assert seen >= 15, f"only {seen} LLM call sites found; the scan is no longer finding them"
