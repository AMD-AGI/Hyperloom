# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""#574 — TraceLens SDK model id normalization.

``_resolve_tracelens_model`` must map the runtime image's dot-form
``ANTHROPIC_MODEL`` (e.g. ``Claude-Opus-4.7``) to the dash form
(``claude-opus-4-7``) strict gateways (core42/SAFE) accept, instead of
forwarding it raw and 400-ing with ``Invalid model name``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
TL_PATH = TOOLS_DIR / "tracelens_analysis.py"


@pytest.fixture(scope="module")
def tl_module():
    spec = importlib.util.spec_from_file_location(
        "tracelens_analysis_model_norm_under_test", TL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dot_form_opus_normalized_to_dash(tl_module, monkeypatch):
    # The exact in-loop failure: image default dot form rejected by core42.
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Opus-4.7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_already_dash_is_unchanged(tl_module, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    assert tl_module._resolve_tracelens_model() == "claude-opus-4-7"


def test_sonnet_dot_form_normalized(tl_module, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "Claude-Sonnet-4.6")
    assert tl_module._resolve_tracelens_model() == "claude-sonnet-4-6"


def test_empty_env_yields_empty(tl_module, monkeypatch):
    # Empty env must keep the prior behavior (runner/SDK default applies).
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert tl_module._resolve_tracelens_model() == ""


def test_whitespace_only_yields_empty(tl_module, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "   ")
    assert tl_module._resolve_tracelens_model() == ""
