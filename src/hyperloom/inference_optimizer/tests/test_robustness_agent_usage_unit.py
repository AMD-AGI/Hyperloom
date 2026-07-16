# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``RobustnessAgentBackend._merge_llm_usage`` — folds the
runtime's ``llm_usage`` block onto ``BackendTurnResult.metadata`` token
counters."""

from __future__ import annotations

from hyperloom.orchestrator.roles.robustness_agent import (
    RobustnessAgentBackend,
)


def test_merge_llm_usage_maps_counters_and_model():
    md: dict = {"session_id": "s", "turn_idx": 0}
    RobustnessAgentBackend._merge_llm_usage(md, {
        "input_tokens": 12, "output_tokens": 5, "calls": 1,
        "latency_ms": 30, "model": "claude-opus-4-7",
    })
    assert md["input_tokens"] == 12
    assert md["output_tokens"] == 5
    assert md["model"] == "claude-opus-4-7"


def test_merge_llm_usage_noop_when_absent():
    md: dict = {"session_id": "s"}
    RobustnessAgentBackend._merge_llm_usage(md, None)
    RobustnessAgentBackend._merge_llm_usage(md, {})
    RobustnessAgentBackend._merge_llm_usage(md, {"calls": 1})
    assert "input_tokens" not in md
    assert "output_tokens" not in md
    assert "model" not in md


def test_merge_llm_usage_keeps_existing_model_when_usage_has_none():
    md: dict = {}
    RobustnessAgentBackend._merge_llm_usage(md, {
        "input_tokens": 1, "output_tokens": 2,
    })
    assert md["input_tokens"] == 1 and md["output_tokens"] == 2
    assert "model" not in md
