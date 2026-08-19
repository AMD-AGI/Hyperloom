###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The LLM tier of source resolution: selection only, with canonical routing.

This pipeline was broken by an LLM writing a placeholder into a field that was
consumed as a path, so the tier that reintroduces an LLM has to be provably
unable to repeat that: it may only echo back one of the paths it was given, the
answer is checked against the filesystem, and a provider must be selected by
the role override or the canonical credential shape.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _llm_source_fallback as lsf  # noqa: E402
from hyperloom.common import llm_config  # noqa: E402


@pytest.fixture()
def enabled():
    """Kept as a no-op so the tier's tests read the same after it became default."""
    return None


@pytest.fixture()
def provider(monkeypatch):
    """Select a provider, which the finalizer settles before it greps."""
    monkeypatch.setenv("HYPERLOOM_LLM_SOURCE_PROVIDER", "openai_compatible")


@pytest.fixture()
def files(tmp_path):
    real = tmp_path / "kernel.py"
    real.write_text("@triton.jit\ndef my_kernel():\n    pass\n", encoding="utf-8")
    test_file = tmp_path / "test_kernel.py"
    test_file.write_text("def test_my_kernel():\n    pass\n", encoding="utf-8")
    return str(real), str(test_file)


def _replies(payload) -> callable:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda _prompt, _model, _timeout: text


# --- Always on ------------------------------------------------------------------


def test_runs_without_any_opt_in(files):
    """The tier is unconditional; no environment setup precedes a call."""
    picked, _conf, _reason = lsf.select_source_via_llm(
        "my_kernel", [files[0]], complete=_replies({"source_file": files[0], "confidence": 1.0})
    )
    assert picked == files[0]


def test_empty_shortlist_still_short_circuits(files):
    """Nothing to choose from means no call, independent of the tier being on."""
    picked, _conf, reason = lsf.select_source_via_llm("my_kernel", [], complete=_replies({}))
    assert picked == ""
    assert "no candidates" in reason


# --- Selection, never generation ----------------------------------------------


def test_accepts_a_candidate_from_the_shortlist(enabled, files):
    real, test_file = files
    picked, confidence, _reason = lsf.select_source_via_llm(
        "my_kernel",
        [test_file, real],
        complete=_replies({"source_file": real, "confidence": 0.9, "reason": "defines the kernel"}),
    )
    assert picked == real
    assert confidence == 0.9


def test_rejects_a_path_outside_the_shortlist(enabled, files, tmp_path):
    """An invented path is the failure mode this tier must not have."""
    invented = str(tmp_path / "hallucinated.py")
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel", [files[0]], complete=_replies({"source_file": invented, "confidence": 1.0})
    )
    assert picked == ""
    assert "not one of the candidates" in reason


def test_rejects_a_shortlisted_path_that_vanished(enabled, tmp_path):
    missing = str(tmp_path / "gone.py")
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel", [missing], complete=_replies({"source_file": missing, "confidence": 1.0})
    )
    assert picked == ""
    assert "does not exist" in reason


def test_rejects_a_path_outside_the_framework_roots(enabled, files):
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel",
        [files[0]],
        framework_roots=("/sgl-workspace/sglang",),
        complete=_replies({"source_file": files[0], "confidence": 1.0}),
    )
    assert picked == ""
    assert "outside every framework root" in reason


def test_rejects_a_symlink_that_escapes_the_framework_root(enabled, tmp_path):
    """A lexical in-root path may not resolve to a file outside the root."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def kernel():\n    pass\n", encoding="utf-8")
    link = root / "kernel.py"
    link.symlink_to(outside)
    picked, _conf, reason = lsf.select_source_via_llm(
        "kernel",
        [str(link)],
        framework_roots=(str(root),),
        complete=_replies({"source_file": str(link), "confidence": 1.0}),
    )
    assert picked == ""
    assert "outside every framework root" in reason


def test_accepts_a_symlink_whose_target_remains_inside_the_root(enabled, tmp_path):
    """Symlinks stay valid, but what is stored is the target that was checked.

    Root containment is decided on the resolved target, so keeping the link
    would record a location whose authorization can be revoked afterwards by
    retargeting it. The review tier resolves for the same reason.
    """
    root = tmp_path / "root"
    root.mkdir()
    target = root / "implementation.py"
    target.write_text("def kernel():\n    pass\n", encoding="utf-8")
    link = root / "kernel.py"
    link.symlink_to(target)
    picked, _conf, _reason = lsf.select_source_via_llm(
        "kernel",
        [str(link)],
        framework_roots=(str(root),),
        complete=_replies({"source_file": str(link), "confidence": 1.0}),
    )
    assert picked == str(target)


def test_no_candidates_short_circuits_without_calling_the_model(enabled):
    def _boom(*_args):  # pragma: no cover - must not run
        raise AssertionError("model called with an empty shortlist")

    picked, _conf, reason = lsf.select_source_via_llm("my_kernel", [], complete=_boom)
    assert picked == ""
    assert "no candidates" in reason


# --- Confidence and malformed replies -----------------------------------------


def test_low_confidence_is_discarded(enabled, files):
    picked, confidence, reason = lsf.select_source_via_llm(
        "my_kernel", [files[0]], complete=_replies({"source_file": files[0], "confidence": 0.4})
    )
    assert picked == ""
    assert confidence == 0.4
    assert "below" in reason


@pytest.mark.parametrize(
    "confidence",
    ["NaN", "Infinity", "-Infinity", '"NaN"', '"Infinity"', '"-Infinity"', "-0.1", "1.1"],
)
def test_non_finite_and_out_of_range_confidence_is_rejected(files, confidence):
    """Only finite confidence values in the declared range are accepted."""
    source = files[0]
    reply = f'{{"source_file": {json.dumps(source)}, "confidence": {confidence}}}'
    picked, parsed_confidence, reason = lsf.select_source_via_llm(
        "my_kernel",
        [source],
        complete=_replies(reply),
    )
    assert picked == ""
    assert parsed_confidence == 0.0
    assert "finite and in [0, 1]" in reason


def test_empty_answer_means_none_of_the_candidates(enabled, files):
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel", [files[0]], complete=_replies({"source_file": "", "confidence": 0.0})
    )
    assert picked == ""
    assert "no candidate" in reason


def test_prose_wrapped_json_is_still_parsed(enabled, files):
    real = files[0]
    reply = f'Sure!\n```json\n{{"source_file": "{real}", "confidence": 0.95}}\n```\n'
    picked, _conf, _reason = lsf.select_source_via_llm("my_kernel", [real], complete=_replies(reply))
    assert picked == real


def test_non_json_reply_is_rejected(enabled, files):
    errors = []
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel",
        [files[0]],
        complete=_replies("I could not determine the file."),
        errors=errors,
    )
    assert picked == ""
    assert "no JSON object" in reason
    assert errors == [reason]


@pytest.mark.parametrize(
    ("reply", "expected_reason"),
    [
        (None, "reply is not text"),
        ("{invalid}", "unparseable JSON"),
        ('{"confidence": "high"}', "confidence is not numeric"),
    ],
)
def test_parse_answer_rejects_invalid_payload_shapes(reply, expected_reason):
    """Malformed payload variants must fail with their precise safe reason."""
    parsed, source_file, confidence, reason = lsf._parse_answer(reply)
    assert parsed is False
    assert source_file == ""
    assert confidence == 0.0
    assert reason == expected_reason


def test_parse_answer_rejects_non_object_json(monkeypatch):
    """The defensive parser guard must reject a decoded non-object payload."""

    class Match:
        """Return a fixed JSON array from the regex match surface."""

        @staticmethod
        def group(_index):
            """Return the non-object JSON payload."""
            return "[]"

    class Pattern:
        """Expose the minimal search interface used by the parser."""

        @staticmethod
        def search(_text):
            """Return the fixed match."""
            return Match()

    monkeypatch.setattr(lsf, "_JSON_BLOCK_RE", Pattern())
    parsed, source_file, confidence, reason = lsf._parse_answer("ignored")
    assert parsed is False
    assert source_file == ""
    assert confidence == 0.0
    assert reason == "JSON payload is not an object"


def test_model_error_is_swallowed(enabled, files):
    def _raise(*_args):
        raise RuntimeError("gateway 401")

    errors: list[str] = []
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel",
        [files[0]],
        complete=_raise,
        errors=errors,
    )
    assert picked == ""
    assert "llm call failed" in reason
    assert errors == [reason]


def test_model_error_is_redacted_from_reason_errors_and_log(files):
    """Transport diagnostics retain only a stable type and status code."""

    class TransportError(RuntimeError):
        """Represent a provider failure carrying sensitive response details."""

        status_code = 401

    def _raise(*_args):
        raise TransportError("https://gateway.example/v1?token=query-secret Authorization: Bearer header-secret")

    errors: list[str] = []
    logs: list[str] = []
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel",
        [files[0]],
        complete=_raise,
        errors=errors,
        log=logs.append,
    )
    recorded = "\n".join([reason, *errors, *logs])
    assert picked == ""
    assert "TransportError" in recorded
    assert "status_code=401" in recorded
    assert "gateway.example" not in recorded
    assert "query-secret" not in recorded
    assert "header-secret" not in recorded
    assert "Authorization" not in recorded


def test_exception_label_skips_hostile_and_boolean_codes():
    """Exception metadata inspection must ignore unsafe or ambiguous values."""

    class HostileMetadataError(RuntimeError):
        """Expose unusable metadata before one safe code."""

        @property
        def status_code(self):
            """Raise instead of exposing provider response details."""
            raise RuntimeError("secret response body")

        code = True
        errno = "E_GATEWAY"

    assert lsf._safe_exception_label(HostileMetadataError()) == ("HostileMetadataError (errno=E_GATEWAY)")


# --- Provider routing and audit -----------------------------------------------


def _stub_credential_shape(
    monkeypatch: pytest.MonkeyPatch,
    *,
    anthropic: bool,
    openai: bool,
) -> None:
    """Make the canonical llm_config predicates report one credential shape."""
    monkeypatch.setattr(llm_config, "is_anthropic_only", lambda: anthropic and not openai)
    monkeypatch.setattr(llm_config, "is_openai_only", lambda: openai and not anthropic)
    monkeypatch.setattr(llm_config, "has_anthropic_side", lambda: anthropic)
    monkeypatch.setattr(llm_config, "has_openai_side", lambda: openai)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("anthropic", lsf._PROVIDER_CLAUDE),
        ("openai", lsf._PROVIDER_OPENAI),
    ],
)
def test_explicit_provider_override_wins_credential_shape(monkeypatch, configured, expected):
    """The role-specific provider knob wins without consulting credential shape."""
    monkeypatch.setenv(lsf._PROVIDER_ENV, configured)

    def _unexpected_shape_probe():
        raise AssertionError("explicit provider must short-circuit credential inference")

    for predicate in (
        "is_anthropic_only",
        "is_openai_only",
        "has_anthropic_side",
        "has_openai_side",
    ):
        monkeypatch.setattr(llm_config, predicate, _unexpected_shape_probe)

    assert lsf._resolve_provider() == expected


@pytest.mark.parametrize(
    ("anthropic", "openai", "expected"),
    [
        pytest.param(False, False, None, id="neither"),
        pytest.param(False, True, lsf._PROVIDER_OPENAI, id="openai-only"),
        pytest.param(True, False, lsf._PROVIDER_CLAUDE, id="anthropic-only"),
        pytest.param(True, True, lsf._PROVIDER_OPENAI, id="both"),
    ],
)
def test_provider_is_inferred_from_canonical_credential_shape(
    monkeypatch,
    anthropic,
    openai,
    expected,
):
    """All four shapes route canonically; dual-configured single shots use OpenAI."""
    monkeypatch.delenv(lsf._PROVIDER_ENV, raising=False)
    _stub_credential_shape(monkeypatch, anthropic=anthropic, openai=openai)

    if expected is None:
        with pytest.raises(RuntimeError, match=lsf._PROVIDER_ENV):
            lsf._resolve_provider()
    else:
        assert lsf._resolve_provider() == expected


@pytest.mark.parametrize(
    ("anthropic", "openai", "expected_provider"),
    [
        pytest.param(False, True, lsf._PROVIDER_OPENAI, id="openai-only"),
        pytest.param(True, False, lsf._PROVIDER_CLAUDE, id="anthropic-only"),
        pytest.param(True, True, lsf._PROVIDER_OPENAI, id="both"),
    ],
)
@pytest.mark.parametrize(
    ("preview_value", "expected_preview"),
    [
        pytest.param("", False, id="preview-unset"),
        pytest.param("1", True, id="preview-opted-in"),
    ],
)
def test_provider_inference_does_not_authorize_source_preview(
    monkeypatch,
    files,
    anthropic,
    openai,
    expected_provider,
    preview_value,
    expected_preview,
):
    """Credential inference changes routing only; source-content egress stays opt-in."""
    monkeypatch.delenv(lsf._PROVIDER_ENV, raising=False)
    monkeypatch.delenv(lsf._PREVIEW_ENV, raising=False)
    if preview_value:
        monkeypatch.setenv(lsf._PREVIEW_ENV, preview_value)
    _stub_credential_shape(monkeypatch, anthropic=anthropic, openai=openai)

    audit = lsf.llm_source_audit()
    prompt = lsf._build_prompt(
        "my_kernel",
        [files[0]],
        framework_roots=(str(Path(files[0]).parent),),
    )

    assert audit["provider"] == expected_provider
    assert audit["source_preview_authorised"] is expected_preview
    assert ("@triton.jit" in prompt) is expected_preview


def test_network_calls_fail_closed_without_override_or_credentials(monkeypatch):
    """A model name alone must never imply a provider or endpoint."""
    monkeypatch.delenv(lsf._PROVIDER_ENV, raising=False)
    _stub_credential_shape(monkeypatch, anthropic=False, openai=False)
    with pytest.raises(RuntimeError, match=lsf._PROVIDER_ENV):
        lsf._resolve_provider()


def test_provider_helpers_fail_closed_without_valid_configuration(monkeypatch):
    """Unsupported providers must fail and Claude audit hosts must stay generic."""
    with pytest.raises(RuntimeError, match="unsupported"):
        lsf._resolve_provider("unknown-provider")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    assert lsf._endpoint_host(lsf._PROVIDER_CLAUDE) == "provider-default"


def test_default_claude_model_handles_role_import_failure(monkeypatch):
    """Standalone tools must tolerate an unavailable orchestrator role module."""
    monkeypatch.setitem(sys.modules, "hyperloom.orchestrator.roles.agent_role", None)
    assert lsf._default_claude_model() == ""


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("claude", lsf._PROVIDER_CLAUDE),
        ("claude_agent_sdk", lsf._PROVIDER_CLAUDE),
        ("openai", lsf._PROVIDER_OPENAI),
        ("openai-compatible", lsf._PROVIDER_OPENAI),
    ],
)
def test_provider_aliases_route_to_native_backends(monkeypatch, configured, expected):
    """Explicit aliases normalize to one auditable provider identifier."""
    monkeypatch.setenv(lsf._PROVIDER_ENV, configured)
    calls = []
    monkeypatch.setattr(
        lsf,
        "_complete_claude_sdk",
        lambda *_args: calls.append(lsf._PROVIDER_CLAUDE) or "claude",
    )
    monkeypatch.setattr(
        lsf,
        "_complete_openai",
        lambda *_args: calls.append(lsf._PROVIDER_OPENAI) or "openai",
    )
    assert lsf._complete("prompt", "model", 1.0) in {"claude", "openai"}
    assert calls == [expected]


def test_openai_provider_adapter_uses_the_sanctioned_client_contract(monkeypatch):
    """The OpenAI adapter must get its client from ``llm_config``, not build one.

    Credential resolution is asserted by ``llm_config``'s own tests; what matters
    here is that the adapter goes through the contract and sends the request the
    fallback expects.
    """
    captured = {}

    def _create(**kwargs):
        """Capture the completion request and return one SDK-shaped response."""
        captured["request"] = kwargs
        message = types.SimpleNamespace(content="provider reply")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class Client:
        """Minimal OpenAI client surface used by the adapter."""

        def __init__(self):
            """Expose the chat-completions surface the contract calls."""
            completions = types.SimpleNamespace(create=_create)
            self.chat = types.SimpleNamespace(completions=completions)

    def _get_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return Client()

    monkeypatch.setattr("hyperloom.common.llm_config.get_openai_client", _get_client)

    reply = lsf._complete_openai("prompt", "source-model", 7.5)

    assert reply == "provider reply"
    assert captured["client_kwargs"] == {}
    assert captured["request"] == {
        "model": "source-model",
        "messages": [
            {"role": "system", "content": lsf._SYSTEM_PROMPT},
            {"role": "user", "content": "prompt"},
        ],
        "temperature": 0.0,
        "timeout": 7.5,
    }


def test_message_text_accepts_sdk_text_shapes():
    """Claude SDK string, text, and content-block forms must all be readable."""
    from hyperloom.common.claude_oneshot import message_text

    assert message_text("direct") == ["direct"]
    assert message_text(types.SimpleNamespace(text="attribute")) == ["attribute"]
    message = types.SimpleNamespace(
        content=[
            {"text": "dict"},
            types.SimpleNamespace(text="object"),
            {"other": "ignored"},
        ]
    )
    assert message_text(message) == ["dict", "object"]


def test_claude_provider_rejects_incomplete_sdk(monkeypatch):
    """A partial SDK installation must fail before any network request."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.ModuleType("claude_agent_sdk"))
    with pytest.raises(RuntimeError, match="missing query / ClaudeAgentOptions"):
        lsf._complete_claude_sdk("prompt", "model", 1.0)


def test_claude_provider_uses_a_tool_free_native_sdk_call(monkeypatch):
    """Claude routing uses the SDK without granting repository or shell tools."""
    captured = {}

    class Options:
        """Capture ClaudeAgentOptions keyword arguments."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["options"] = self

    async def query(*, prompt, options):
        """Yield one SDK-shaped result message."""
        captured["prompt"] = prompt
        captured["query_options"] = options
        yield types.SimpleNamespace(result='{"source_file": "", "confidence": 0}')

    fake_sdk = types.SimpleNamespace(query=query, ClaudeAgentOptions=Options)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)
    monkeypatch.setattr(
        "hyperloom.common.llm_config.claude_sdk_env_options",
        lambda **_kwargs: {
            "model": "overridden-model",
            "tools": ["Read"],
            "setting_sources": ["user"],
            "skills": ["unsafe-skill"],
            "strict_mcp_config": False,
            "mcp_servers": {"unsafe": {"command": "sh"}},
            "plugins": [{"type": "local", "path": "/tmp/plugin"}],
            "max_turns": 99,
            "allowed_tools": ["Bash"],
            "disallowed_tools": [],
        },
    )
    reply = lsf._complete_claude_sdk("prompt", "claude-model", 1.0)
    assert json.loads(reply)["source_file"] == ""
    assert captured["prompt"] == "prompt"
    assert captured["query_options"] is captured["options"]
    options = captured["options"].kwargs
    assert options["model"] == "claude-model"
    assert options["tools"] == []
    assert options["setting_sources"] == []
    assert options["skills"] == []
    assert options["strict_mcp_config"] is True
    assert options["mcp_servers"] == {}
    assert options["plugins"] == []
    assert options["max_turns"] == 1
    assert options["allowed_tools"] == []
    assert "Read" in options["disallowed_tools"]
    assert "Bash" in options["disallowed_tools"]


def test_provider_audit_never_records_url_credentials(monkeypatch):
    """Audit metadata contains only the endpoint hostname, never URL secrets."""
    monkeypatch.setenv(lsf._PROVIDER_ENV, "openai-compatible")
    monkeypatch.setenv(lsf._MODEL_ENV, "gpt-source")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://user:secret@gateway.example/Unified/v1?token=must-not-leave",
    )
    audit = lsf.llm_source_audit()
    assert audit == {
        "provider": lsf._PROVIDER_OPENAI,
        "model": "gpt-source",
        "endpoint_host": "gateway.example",
        "source_preview_authorised": False,
    }
    assert "secret" not in json.dumps(audit)
    assert "token" not in json.dumps(audit)


def test_provider_model_settings_do_not_cross(monkeypatch):
    """Each native provider reads only its own model fallback."""
    monkeypatch.delenv(lsf._MODEL_ENV, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-source")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-source")
    assert lsf._resolve_model("", lsf._PROVIDER_OPENAI) == "gpt-source"
    assert lsf._resolve_model("", lsf._PROVIDER_CLAUDE) == "claude-source"


def test_prompt_context_and_unreadable_preview_are_explicit(files, tmp_path):
    """Prompt context must be retained while preview failures stay non-fatal."""
    prompt = lsf._build_prompt(
        "my_kernel",
        [files[0]],
        context_block="Trace context",
        with_preview=False,
    )
    assert prompt.startswith("Trace context\n")
    assert lsf._preview(str(tmp_path / "missing.py")) == "<unreadable>"


def test_selection_reports_unconfigured_provider(monkeypatch, files):
    """Missing override and credentials must be reported without a transport."""
    monkeypatch.delenv(lsf._PROVIDER_ENV, raising=False)
    _stub_credential_shape(monkeypatch, anthropic=False, openai=False)
    errors = []
    logs = []

    picked, confidence, reason = lsf.select_source_via_llm(
        "my_kernel",
        [files[0]],
        model="source-model",
        errors=errors,
        log=logs.append,
    )

    assert picked == ""
    assert confidence == 0.0
    assert reason.startswith("llm configuration failed: RuntimeError")
    assert errors == [reason]
    assert logs == ["llm_source_fallback: configuration failed: RuntimeError"]


def test_selection_reports_missing_provider_model(monkeypatch, files):
    """A configured provider without a model must fail before prompt creation."""
    monkeypatch.setenv(lsf._PROVIDER_ENV, "openai_compatible")
    for name in (lsf._MODEL_ENV, "OPENAI_MODEL", "CODEX_MODEL"):
        monkeypatch.delenv(name, raising=False)
    errors = []
    logs = []

    picked, confidence, reason = lsf.select_source_via_llm(
        "my_kernel",
        [files[0]],
        errors=errors,
        log=logs.append,
    )

    assert picked == ""
    assert confidence == 0.0
    assert reason == "no model configured"
    assert errors == [reason]
    assert logs == [f"llm_source_fallback: no model configured; set ${lsf._MODEL_ENV}"]


# --- Wiring into finalization --------------------------------------------------


def test_finalizer_leaves_the_candidate_alone_when_grep_finds_nothing(provider, monkeypatch):
    """No shortlist means no call, and the candidate stays unresolved."""
    import tracelens_analysis as tl

    monkeypatch.setattr(tl, "collect_source_candidates_via_grep", lambda *_a, **_k: [])
    item = {"name": "some_kernel", "gpu_pct": 40.0, "source_file": ""}
    tl._apply_llm_source_fallback(item)
    assert item["source_file"] == ""
    assert item["source_resolution_reason"] == "llm_fallback_no_shortlist"


def test_finalizer_settles_the_provider_before_paying_for_the_shortlist(monkeypatch):
    """An unconfigured tier must not run the grep it would only decline to use.

    The shortlist walks every framework root once per keyword, so doing it
    first would charge that to every hot kernel on a deployment with neither a
    role override nor canonical provider credentials.
    """
    import tracelens_analysis as tl

    monkeypatch.delenv("HYPERLOOM_LLM_SOURCE_PROVIDER", raising=False)
    _stub_credential_shape(monkeypatch, anthropic=False, openai=False)
    grepped: list[str] = []
    monkeypatch.setattr(
        tl,
        "collect_source_candidates_via_grep",
        lambda name, *_a, **_k: grepped.append(name) or [],
    )
    item = {"name": "hot_kernel", "gpu_pct": 40.0, "source_file": ""}
    tl._apply_llm_source_fallback(item)
    assert grepped == []
    assert item["source_resolution_reason"] == "llm_fallback_skipped: no provider configured"
    assert item["source_resolution_llm_audit"]["outcome"] == "configuration_error"


def test_finalizer_skips_cold_kernels(monkeypatch):
    """Below the GPU-share floor the round-trip is not worth its cost."""
    import tracelens_analysis as tl

    called = []
    monkeypatch.setattr(tl, "collect_source_candidates_via_grep", lambda *_a, **_k: called.append(1) or [])
    tl._apply_llm_source_fallback({"name": "k", "gpu_pct": 0.5, "source_file": ""})
    assert not called


def test_gateway_failure_is_not_recorded_as_a_model_decline(provider, monkeypatch):
    """Transport failure and a valid refusal require different operator action."""
    import tracelens_analysis as tl

    monkeypatch.setattr(
        tl,
        "collect_source_candidates_via_grep",
        lambda *_args, **_kwargs: ["/repo/kernel.py"],
    )
    monkeypatch.setattr(
        lsf,
        "llm_source_audit",
        lambda **_kwargs: {
            "provider": "openai_compatible",
            "model": "gpt-source",
            "endpoint_host": "gateway.example",
            "source_preview_authorised": False,
        },
    )

    def _fail(*_args, errors=None, **_kwargs):
        """Return the public failure shape while reporting a transport error."""
        errors.append("llm call failed: gateway 401")
        return "", 0.0, "llm call failed: gateway 401"

    monkeypatch.setattr(lsf, "select_source_via_llm", _fail)
    item = {
        "name": "kernel",
        "gpu_pct": 40.0,
        "source_file": "",
        "source_resolution_reason": "trace_resolver_error: truncated",
    }
    tl._apply_llm_source_fallback(item)
    assert "trace_resolver_error" in item["source_resolution_reason"]
    assert "llm_fallback_error" in item["source_resolution_reason"]
    assert "llm_fallback_declined" not in item["source_resolution_reason"]
    assert item["source_resolution_llm_audit"]["outcome"] == "error"


def test_valid_model_refusal_is_still_recorded_as_declined(provider, monkeypatch):
    """A parsed no-candidate verdict remains distinct from call failure."""
    import tracelens_analysis as tl

    monkeypatch.setattr(
        tl,
        "collect_source_candidates_via_grep",
        lambda *_args, **_kwargs: ["/repo/kernel.py"],
    )
    monkeypatch.setattr(lsf, "llm_source_audit", lambda **_kwargs: {"provider": "test"})

    def _decline(*_args, errors=None, **_kwargs):
        """Return a valid refusal without adding a call error."""
        assert errors == []
        return "", 0.0, "model reported no candidate defines the kernel"

    monkeypatch.setattr(lsf, "select_source_via_llm", _decline)
    item = {"name": "kernel", "gpu_pct": 40.0, "source_file": ""}
    tl._apply_llm_source_fallback(item)
    assert item["source_resolution_reason"].startswith("llm_fallback_declined")
    assert item["source_resolution_llm_audit"]["outcome"] == "declined"


def test_fallback_provider_audit_is_projected_into_the_artifact():
    """Provider identity remains attached to the decision it produced."""
    import tracelens_analysis as tl

    audit = {
        "provider": "claude_agent_sdk",
        "model": "claude-source",
        "endpoint_host": "provider-default",
        "source_preview_authorised": False,
        "outcome": "accepted",
    }
    entry = tl.build_source_resolution_entries(
        [
            {
                "kernel_id": "k1",
                "name": "kernel",
                "gpu_pct": 10.0,
                "source_file": "/repo/kernel.py",
                "source_resolution_method": "llm_fallback",
                "source_resolution_llm_audit": audit,
            }
        ]
    )[0]
    assert entry["llm_audit"] == audit
    assert entry["method"] == "llm_fallback"


def test_runtime_api_names_yield_no_shortlist():
    import tracelens_analysis as tl

    assert tl.collect_source_candidates_via_grep("hipGraphLaunch") == []


def test_relaxed_shortlist_is_reachable_after_strict_grep_gives_up(
    monkeypatch,
    tmp_path,
):
    """Short mangled identifiers feed only the bounded LLM shortlist."""
    import tracelens_analysis as tl

    source = tmp_path / "kernel.py"
    source.write_text("def gemm():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(tl, "KNOWN_SEARCH_ROOTS", [str(tmp_path)])
    tl._GREP_CACHE.clear()
    mangled = "_Z2ab4gemm"
    assert tl._candidate_keywords(mangled) == []
    assert tl.locate_source_via_grep(mangled) == ""
    assert tl.collect_source_candidates_via_grep(mangled) == [str(source)]


# --- source egress ------------------------------------------------------------


def test_file_contents_do_not_leave_without_authorisation(monkeypatch, files):
    """Shipping file heads to a provider is an operator decision, not a default."""
    real, _ = files
    monkeypatch.delenv(lsf._PREVIEW_ENV, raising=False)
    prompt = lsf._build_prompt("my_kernel", [real])
    assert "@triton.jit" not in prompt
    # The path itself still carries most of the selection signal.
    assert real in prompt


def test_authorisation_restores_the_preview(monkeypatch, files):
    """The tier is not crippled by the default; an operator can opt back in."""
    real, _ = files
    monkeypatch.setenv(lsf._PREVIEW_ENV, "1")
    assert lsf.source_preview_authorised() is True
    assert "@triton.jit" in lsf._build_prompt(
        "my_kernel",
        [real],
        framework_roots=(str(Path(real).parent),),
    )


def test_preview_does_not_read_a_symlink_target_outside_the_root(monkeypatch, tmp_path):
    """An escaping symlink may be named but its target must never be read."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside_secret_marker\n", encoding="utf-8")
    link = root / "kernel.py"
    link.symlink_to(outside)
    reads: list[str] = []
    monkeypatch.setenv(lsf._PREVIEW_ENV, "1")
    monkeypatch.setattr(lsf, "_preview", lambda path: reads.append(path) or "leaked")

    prompt = lsf._build_prompt(
        "kernel",
        [str(link)],
        framework_roots=(str(root),),
    )

    assert reads == []
    assert "outside_secret_marker" not in prompt
    assert "leaked" not in prompt


def test_preview_and_storage_reuse_the_same_canonical_target(monkeypatch, tmp_path):
    """Retargeting a symlink after preview cannot change the stored path."""
    root = tmp_path / "root"
    root.mkdir()
    original = root / "implementation.py"
    original.write_text("original_kernel_marker\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("outside_kernel_marker\n", encoding="utf-8")
    link = root / "kernel.py"
    link.symlink_to(original)
    monkeypatch.setenv(lsf._PREVIEW_ENV, "1")

    def _retarget_after_preview(prompt, _model, _timeout):
        """Retarget the candidate only after its validated preview is built."""
        assert "original_kernel_marker" in prompt
        assert "outside_kernel_marker" not in prompt
        link.unlink()
        link.symlink_to(outside)
        return json.dumps({"source_file": str(link), "confidence": 1.0})

    picked, _confidence, _reason = lsf.select_source_via_llm(
        "kernel",
        [str(link)],
        framework_roots=(str(root),),
        complete=_retarget_after_preview,
    )

    assert picked == str(original)
