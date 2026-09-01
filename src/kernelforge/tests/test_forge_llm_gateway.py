"""Unit tests for the OpenAI-compatible gateway line."""

from __future__ import annotations

import pytest

from kernelforge.llm import (
    LlmGateway,
    expand_env_refs,
    format_custom_headers,
    normalize_anthropic_base_url,
    parse_custom_headers,
    resolve_anthropic_gateway,
    resolve_openai_gateway,
)

_GATEWAY_ENV = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    "SAFE_API_KEY",
    "FORGE_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in _GATEWAY_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_nothing_configured(clean_env):
    gateway = resolve_openai_gateway()
    assert gateway == LlmGateway()
    assert not gateway.is_complete()


def test_complete_pair(clean_env):
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    gateway = resolve_openai_gateway()
    assert gateway.is_complete()
    assert gateway == LlmGateway("https://gw.example/llm-proxy/v1", "OPENAI_API_KEY", {})


@pytest.mark.parametrize("missing", ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
def test_half_a_pair_is_not_configured(clean_env, missing):
    """Either half alone leaves the line unusable rather than half-usable."""
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.delenv(missing)
    assert not resolve_openai_gateway().is_complete()


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("var", ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
def test_blank_value_is_not_configured(clean_env, var, blank):
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv(var, blank)
    assert not resolve_openai_gateway().is_complete()


@pytest.mark.parametrize(
    "configured",
    [
        "https://gw.example/llm-proxy",
        "https://gw.example/llm-proxy/",
        "https://gw.example/llm-proxy/v1",
        "https://api.openai.com/v1",
    ],
)
def test_base_url_is_used_exactly_as_configured(clean_env, configured):
    """No route suffix is appended and no path is rewritten.

    The operator knows their gateway's layout; guessing at it would also hide
    their typos behind ours.
    """
    clean_env.setenv("OPENAI_BASE_URL", configured)
    clean_env.setenv("OPENAI_API_KEY", "openai")
    assert resolve_openai_gateway().base_url == configured


def test_the_anthropic_endpoint_and_credential_are_never_borrowed(clean_env):
    """A fully configured Anthropic line does not make this one usable.

    The two lines are different protocols on different routes and belong to
    different consumers; substituting one produces a failure nobody can trace
    back to a variable. Claude reads ANTHROPIC_* itself. Headers are the one
    exception, and only within a single host -- see
    :func:`_resolve_openai_gateway_headers`.
    """
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "bearer")
    clean_env.setenv("ANTHROPIC_API_KEY", "console")
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub")
    assert not resolve_openai_gateway().is_complete()

    # Its own pair is what turns the line on; Anthropic headers fill in when the
    # OpenAI header slot was left empty (single-gateway / tag-injection setups).
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    assert resolve_openai_gateway() == LlmGateway(
        "https://gw.example/llm-proxy/v1",
        "OPENAI_API_KEY",
        {"Ocp-Apim-Subscription-Key": "sub"},
    )


def test_retired_keys_are_not_credentials(clean_env):
    """SAFE_API_KEY and FORGE_API_KEY no longer authenticate anything."""
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
    clean_env.setenv("SAFE_API_KEY", "safe")
    clean_env.setenv("FORGE_API_KEY", "forge")
    assert not resolve_openai_gateway().is_complete()


# ── headers ──────────────────────────────────────────────────────────────────
def test_expand_env_refs(monkeypatch):
    """Shared by both lines: one parses headers here, the other hands them off."""
    monkeypatch.setenv("SUBKEY", "xyz")
    assert expand_env_refs("Key: ${SUBKEY}") == "Key: xyz"
    # An unset reference becomes empty rather than staying literal, so a blank
    # header value points at the typo instead of shipping "${TYPO}" upstream.
    monkeypatch.delenv("NOPE", raising=False)
    assert expand_env_refs("Key: ${NOPE}") == "Key: "
    assert expand_env_refs("no refs here") == "no refs here"


def test_parse_custom_headers_lines_json_and_envref(monkeypatch):
    # newline-delimited "Name: value"
    assert parse_custom_headers("Ocp-Apim-Subscription-Key: abc123") == {"Ocp-Apim-Subscription-Key": "abc123"}
    # JSON object form
    assert parse_custom_headers('{"Ocp-Apim-Subscription-Key": "abc123"}') == {"Ocp-Apim-Subscription-Key": "abc123"}
    # ${VAR} expansion from env
    monkeypatch.setenv("SUBKEY", "xyz")
    assert parse_custom_headers("Ocp-Apim-Subscription-Key: ${SUBKEY}") == {"Ocp-Apim-Subscription-Key": "xyz"}
    # malformed JSON (starts with { but invalid) falls back to line parsing,
    # matching Hyperloom's behavior.
    assert parse_custom_headers('{"broken: value') == {'{"broken': "value"}
    assert parse_custom_headers(None) == {} and parse_custom_headers("") == {}


def test_comma_separated_pairs_are_not_split(caplog):
    """A header value may contain commas, so one line stays one header.

    The retired NTID regex stopped at the first comma, so a Secret written in
    that style now yields a wrong value rather than two headers — warn loudly
    instead of guessing which commas were separators.
    """
    with caplog.at_level("WARNING", logger="kernelforge.llm"):
        parsed = parse_custom_headers("user: alice, x-foo: bar")
    assert parsed == {"user": "alice, x-foo: bar"}
    assert "packs more headers on one line" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="kernelforge.llm"):
        parse_custom_headers("Accept: text/html, application/json")
    assert "packs more headers" not in caplog.text


def test_header_line_without_a_colon_is_reported(caplog):
    with caplog.at_level("WARNING", logger="kernelforge.llm"):
        assert parse_custom_headers("user: alice\nnonsense") == {"user": "alice"}
    assert "without a 'Name: value' colon" in caplog.text


def test_format_custom_headers_round_trips():
    raw = "Ocp-Apim-Subscription-Key: sub123\nuser: alice"
    assert format_custom_headers(parse_custom_headers(raw)) == raw
    assert format_custom_headers({}) == ""


# ── the Anthropic line ───────────────────────────────────────────────────────
def test_anthropic_line_reports_what_is_configured(clean_env):
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv("ANTHROPIC_API_KEY", "console")
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "user: alice")
    assert resolve_anthropic_gateway() == LlmGateway(
        "https://gw.example/llm-proxy", "ANTHROPIC_API_KEY", {"user": "alice"}
    )

    # Anthropic protocol, so the native x-api-key form stays ahead of the
    # gateway bearer token, the order Hyperloom's Claude paths also use.
    clean_env.setenv("ANTHROPIC_AUTH_TOKEN", "bearer")
    assert resolve_anthropic_gateway().key_env == "ANTHROPIC_API_KEY"

    clean_env.delenv("ANTHROPIC_API_KEY")
    assert resolve_anthropic_gateway().key_env == "ANTHROPIC_AUTH_TOKEN"


def test_anthropic_line_allows_a_missing_endpoint(clean_env):
    """The CLI applies its own default, so this is the native-Anthropic setup."""
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    resolved = resolve_anthropic_gateway()
    assert resolved.has_key and not resolved.has_endpoint
    assert resolved.key_env == "ANTHROPIC_API_KEY"


def test_incomplete_is_not_the_same_as_unusable(clean_env):
    """Neither Anthropic half is mandatory, so completeness must be asked for.

    A Claude CLI on a Max login needs no endpoint and no credential, and an
    API-credit user needs only the key. A single truthiness rule shared with the
    OpenAI line would label both of those as unconfigured.
    """
    assert not resolve_anthropic_gateway().is_complete()

    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    assert not resolve_anthropic_gateway().is_complete()
    assert resolve_anthropic_gateway().has_key

    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    assert resolve_anthropic_gateway().is_complete()


def test_anthropic_line_ignores_the_openai_one(clean_env):
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("OPENAI_CUSTOM_HEADERS", "user: not-mine")
    assert resolve_anthropic_gateway() == LlmGateway()


def test_openai_headers_win_and_the_same_gateway_fills_when_empty(clean_env):
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub")
    assert resolve_openai_gateway().headers == {"Ocp-Apim-Subscription-Key": "sub"}

    clean_env.setenv("OPENAI_CUSTOM_HEADERS", "user: mine\nOcp-Apim-Subscription-Key: own")
    assert resolve_openai_gateway().headers == {
        "user": "mine",
        "Ocp-Apim-Subscription-Key": "own",
    }


def test_a_different_host_never_receives_the_anthropic_headers(clean_env):
    """The headers carry a gateway secret, so they stop at that gateway.

    Reusing them across hosts would hand the operator's subscription key to a
    machine they never pointed at, and an ``Authorization`` among them would
    displace this line's own bearer on every call.
    """
    clean_env.setenv("OPENAI_BASE_URL", "https://other.example/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: not-mine")
    assert resolve_openai_gateway().headers == {}


def test_an_unknown_anthropic_host_is_not_assumed_to_match(clean_env):
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.delenv("ANTHROPIC_BASE_URL", raising=False)
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: not-mine")
    assert resolve_openai_gateway().headers == {}


def test_a_plaintext_endpoint_is_not_the_same_origin_as_a_tls_one(clean_env):
    # Same name, no TLS: sending the subscription key here puts it on the wire.
    clean_env.setenv("OPENAI_BASE_URL", "http://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub")
    assert resolve_openai_gateway().headers == {}


def test_the_default_port_matches_its_explicit_form(clean_env):
    # Spelling the default port must not silently drop a fallback that applies.
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example:443/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub")
    assert resolve_openai_gateway().headers == {"Ocp-Apim-Subscription-Key": "sub"}


def test_this_line_keeps_its_own_credential(clean_env):
    """Borrowed headers never carry the other line's authentication.

    ``default_headers`` are applied over the SDK's own, so an inherited
    ``Authorization`` would replace the bearer built from OPENAI_API_KEY and
    401 every call -- while the subscription key and the spend tag still have
    to get through.
    """
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "Authorization: Bearer anthropic-only\nx-api-key: anthropic-native\n"
        "Ocp-Apim-Subscription-Key: sub\nx-litellm-tags: application=hyperloom",
    )
    assert resolve_openai_gateway().headers == {
        "Ocp-Apim-Subscription-Key": "sub",
        "x-litellm-tags": "application=hyperloom",
    }


def test_openai_line_inherits_litellm_tags_from_anthropic(clean_env):
    clean_env.setenv("OPENAI_BASE_URL", "https://gw.example/llm-proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "openai")
    clean_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example/llm-proxy")
    clean_env.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "Ocp-Apim-Subscription-Key: sub\n"
        "x-litellm-tags: application=hyperloom,session=sess-1,component=forge,operation=forge_loop\n"
        "x-litellm-trace-id: sess-1",
    )
    assert resolve_openai_gateway().headers == {
        "Ocp-Apim-Subscription-Key": "sub",
        "x-litellm-tags": "application=hyperloom,session=sess-1,component=forge,operation=forge_loop",
        "x-litellm-trace-id": "sess-1",
    }


@pytest.mark.parametrize(
    "configured,expected",
    [
        # A bare route is what both clients want; leave it alone.
        ("https://llm-api.amd.com/anthropic", "https://llm-api.amd.com/anthropic"),
        ("https://api.anthropic.com", "https://api.anthropic.com"),
        # A LiteLLM proxy publishes its base with the version already on it.
        ("https://gw.example/llm-proxy/v1", "https://gw.example/llm-proxy"),
        ("https://gw.example/llm-proxy/v1/", "https://gw.example/llm-proxy"),
        # Someone pasted the whole endpoint out of a curl command.
        ("https://gw.example/llm-proxy/v1/messages", "https://gw.example/llm-proxy"),
    ],
)
def test_anthropic_base_url_loses_only_a_duplicated_tail(configured, expected):
    """Both the SDK and the CLI append /v1/messages, so a base carrying it 404s.

    Measured: with the tail left on, the CLI reports the model as missing or
    unauthorized rather than the doubled path it actually requested.
    """
    assert normalize_anthropic_base_url(configured) == expected
