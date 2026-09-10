# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``RobustnessAgentBackend``'s per-turn bookkeeping — the
``llm_usage`` fold onto ``BackendTurnResult.metadata`` token counters, and the
robustness account it records at each of its three exits."""

from __future__ import annotations

import json

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6_robustness import collect_v6_robustness
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts
from pathlib import Path

from hyperloom.inference_optimizer.cli.credentials import _resolve_agent_root
from hyperloom.orchestrator.roles.base import BackendError
from hyperloom.orchestrator.roles.robustness_agent import (
    NoIntentEmitted,
    RobustnessAgentBackend,
)


def test_merge_llm_usage_maps_counters_and_model():
    md: dict = {"session_id": "s", "turn_idx": 0}
    RobustnessAgentBackend._merge_llm_usage(
        md,
        {
            "input_tokens": 12,
            "output_tokens": 5,
            "calls": 1,
            "latency_ms": 30,
            "model": "claude-opus-4-7",
        },
    )
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
    RobustnessAgentBackend._merge_llm_usage(
        md,
        {
            "input_tokens": 1,
            "output_tokens": 2,
        },
    )
    assert md["input_tokens"] == 1 and md["output_tokens"] == 2
    assert "model" not in md


# ---- per-turn robustness recording ----
#
# The agent's own account of a turn is what the ``robustness`` breakdown key
# carries, so what matters here is that every exit records one: the two failure
# exits used to reach the log and stop there, leaving a mute agent
# indistinguishable from a session with nothing to report.


@pytest.fixture
def agent_root() -> Path:
    root = _resolve_agent_root("robustness")
    if root is None:
        pytest.skip("robustness-agent runtime not found")
    return root


def _backend(agent_root, session_dir, emit: dict | None):
    """A backend whose runtime writes ``emit`` and nothing else."""

    def caller(call):
        if emit is not None:
            call.out_path.write_text(json.dumps(emit), encoding="utf-8")

    return RobustnessAgentBackend(
        robustness_agent_root=agent_root,
        session_dir=session_dir,
        runtime_caller_factory=lambda: caller,
    )


def _recorded_turns(session_dir):
    return collect_v6_robustness(assemble_parts(session_dir, warnings=[]).get("robustness"))["turns"]


@pytest.mark.asyncio
async def test_a_validated_envelope_records_what_it_raised(agent_root, tmp_path):
    emit = {
        "tick_index": 7,
        "parse_warnings": ["clipped"],
        "intent_envelope": {
            "intents": [
                {
                    "intent_type": "alert",
                    "payload": {"severity": "high", "topic": "crash_rate", "summary": "crash rate spiked"},
                }
            ]
        },
    }
    await _backend(agent_root, tmp_path, emit).run("prompt")

    (turn,) = _recorded_turns(tmp_path)
    assert turn["outcome"] == "intents"
    assert turn["tick_index"] == 7
    assert [(i["type"], i["severity"]) for i in turn["intents"]] == [("alert", "high")]
    assert turn["parse_warnings"] == ["clipped"]


@pytest.mark.asyncio
async def test_an_emit_without_an_envelope_still_records_the_turn(agent_root, tmp_path):
    with pytest.raises(BackendError):
        await _backend(agent_root, tmp_path, {"tick_index": 3}).run("prompt")

    (turn,) = _recorded_turns(tmp_path)
    assert turn["outcome"] == "no_envelope"
    assert turn["intents"] == []
    assert turn["detail"]


@pytest.mark.asyncio
async def test_an_invalid_envelope_records_why_it_failed(agent_root, tmp_path):
    emit = {"intent_envelope": {"intents": [{"intent_type": "not_a_real_intent", "payload": {}}]}}
    with pytest.raises((NoIntentEmitted, BackendError)):
        await _backend(agent_root, tmp_path, emit).run("prompt")

    (turn,) = _recorded_turns(tmp_path)
    assert turn["outcome"] == "invalid_envelope"
    assert turn["detail"]
