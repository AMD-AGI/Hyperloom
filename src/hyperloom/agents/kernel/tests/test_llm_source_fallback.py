###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The LLM tier of source resolution: selection only, and off by default.

This pipeline was broken by an LLM writing a placeholder into a field that was
consumed as a path, so the tier that reintroduces an LLM has to be provably
unable to repeat that: it may only echo back one of the paths it was given, the
answer is checked against the filesystem, and it stays off unless enabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _llm_source_fallback as lsf  # noqa: E402


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK", "1")


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


# --- Off by default -----------------------------------------------------------


def test_disabled_by_default(monkeypatch, files):
    monkeypatch.delenv("HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK", raising=False)
    assert lsf.llm_fallback_enabled() is False
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel", [files[0]], complete=_replies({"source_file": files[0], "confidence": 1.0})
    )
    assert picked == ""
    assert reason == "disabled"


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_enabled_by_truthy_flag(monkeypatch, value):
    monkeypatch.setenv("HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK", value)
    assert lsf.llm_fallback_enabled() is True


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
    picked, _conf, reason = lsf.select_source_via_llm(
        "my_kernel", [files[0]], complete=_replies("I could not determine the file.")
    )
    assert picked == ""
    assert "no JSON object" in reason


def test_model_error_is_swallowed(enabled, files):
    def _raise(*_args):
        raise RuntimeError("gateway 401")

    picked, _conf, reason = lsf.select_source_via_llm("my_kernel", [files[0]], complete=_raise)
    assert picked == ""
    assert "llm call failed" in reason


# --- Wiring into finalization --------------------------------------------------


def test_finalizer_skips_the_tier_when_disabled(monkeypatch):
    import tracelens_analysis as tl

    monkeypatch.delenv("HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK", raising=False)
    item = {"name": "some_kernel", "gpu_pct": 40.0, "source_file": ""}
    tl._apply_llm_source_fallback(item)
    assert item["source_file"] == ""


def test_finalizer_skips_cold_kernels(monkeypatch):
    """Below the GPU-share floor the round-trip is not worth its cost."""
    import tracelens_analysis as tl

    monkeypatch.setenv("HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK", "1")
    called = []
    monkeypatch.setattr(tl, "collect_source_candidates_via_grep", lambda *_a, **_k: called.append(1) or [])
    tl._apply_llm_source_fallback({"name": "k", "gpu_pct": 0.5, "source_file": ""})
    assert not called


def test_runtime_api_names_yield_no_shortlist():
    import tracelens_analysis as tl

    assert tl.collect_source_candidates_via_grep("hipGraphLaunch") == []
