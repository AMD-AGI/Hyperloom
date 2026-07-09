# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""#574 — TraceLens SDK model id normalization.

``_resolve_tracelens_model`` must map the runtime image's dot-form
``ANTHROPIC_MODEL`` (e.g. ``Claude-Opus-4.7``) to the dash form
(``claude-opus-4-7``) strict gateways (core42/SAFE) accept, instead of
forwarding it raw and 400-ing with ``Invalid model name``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
TL_PATH = TOOLS_DIR / "tracelens_analysis.py"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


@pytest.fixture(scope="module")
def tl_module():
    spec = importlib.util.spec_from_file_location(
        "tracelens_analysis_model_norm_under_test", TL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clear_gateway_env(monkeypatch):
    # Isolate from the runner's own env so each test sets its own gateway.
    for k in ("ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "SAFE_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def _use_safe_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1")


def test_dot_form_opus_normalized_on_safe(tl_module, monkeypatch):
    # The exact in-loop failure: image default dot form rejected by core42.
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_safe_detected_via_safe_api_key(tl_module, monkeypatch):
    monkeypatch.setenv("SAFE_API_KEY", "ak-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_already_dash_is_unchanged_on_safe(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_non_safe_gateway_leaves_dot_form_untouched(tl_module, monkeypatch):
    # Direct Anthropic / other backends must not have their model id mangled.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"


def test_empty_env_yields_empty(tl_module, monkeypatch):
    # Empty env must keep the prior behavior (runner/SDK default applies).
    _use_safe_gateway(monkeypatch)
    assert tl_module._resolve_tracelens_model() == ""


def test_whitespace_only_yields_empty(tl_module, monkeypatch):
    _use_safe_gateway(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MODEL", "   ")
    assert tl_module._resolve_tracelens_model() == ""


def test_plain_openai_base_url_not_normalized(tl_module, monkeypatch):
    # A non-gateway OpenAI-compatible base URL must NOT mangle the model id.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"


def test_no_gateway_env_leaves_dot_form_untouched(tl_module, monkeypatch):
    # No base URL / key at all: the SDK default applies, id stays raw.
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "Claude-Opus-4.7"


def test_self_hosted_litellm_base_url_is_treated_as_gateway(tl_module, monkeypatch):
    # Documented behavior: any 'litellm' host is treated as the strict gateway
    # and normalized. A non-SAFE self-hosted LiteLLM deployment is included by
    # the 'litellm' marker; change the marker set if that is undesired.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://litellm.internal.example.com/v1")
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"
