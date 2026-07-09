# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Serving-launch fidelity forwarding for the PerfSkills(GEAK) handoff (#805).

Guards that Hyperloom forwards the SAME max-model-len / gpu-mem-util its Magpie
baseline served with, so GEAK/e2e launches the IDENTICAL vLLM engine (else it
re-baselines a slower default stack and kernel wins never reproduce e2e). The
knobs are robustly sourced from the baseline server-args string (the common
case, e.g. gpt-oss ``--max-model-len 2248 --gpu-memory-utilization 0.9``), not
only from a dedicated CLI field, and are OMITTED when unresolved so the GEAK
adapter keeps its own defaults.

Run: python3 -m pytest inference_optimizer/tests/test_perfskills_handoff_fidelity.py -v
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.loop.coordinator_helpers import (
    _parse_server_arg_value,
    _resolve_serving_fidelity,
)


# ── _parse_server_arg_value ──────────────────────────────────────────────────


def test_parse_space_and_equals_forms() -> None:
    args = "--no-enable-prefix-caching --block-size 64 --max-model-len 2248 --gpu-memory-utilization=0.9"
    assert _parse_server_arg_value(args, "--max-model-len") == "2248"
    assert _parse_server_arg_value(args, "--gpu-memory-utilization") == "0.9"
    assert _parse_server_arg_value(args, "--block-size") == "64"


def test_parse_absent_and_valueless_and_empty() -> None:
    assert _parse_server_arg_value("--a 1", "--missing") is None
    assert _parse_server_arg_value("--flag", "--flag") is None      # no value at end
    assert _parse_server_arg_value("", "--flag") is None
    assert _parse_server_arg_value("--x 1", "") is None


def test_parse_quoted_value_survives_shlex() -> None:
    args = "--compilation-config='{\"cudagraph_mode\":\"FULL\"}' --max-model-len 4096"
    assert _parse_server_arg_value(args, "--max-model-len") == "4096"
    assert _parse_server_arg_value(args, "--compilation-config") == '{"cudagraph_mode":"FULL"}'


# ── _resolve_serving_fidelity ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)
    monkeypatch.delenv("GPU_MEMORY_UTILIZATION", raising=False)


def test_resolve_from_baseline_server_args() -> None:
    """The gpt-oss case: both knobs ride inside the baseline server-args blob."""
    args = "--no-enable-prefix-caching --block-size 64 --max-model-len 2248 --trust-remote-code --gpu-memory-utilization 0.9"
    out = _resolve_serving_fidelity(baseline_server_args=args, state_max_model_len=0)
    assert out == {"max_model_len": 2248, "mem_fraction": 0.9}


def test_resolve_state_field_wins_for_max_model_len() -> None:
    args = "--max-model-len 2248 --gpu-memory-utilization 0.85"
    out = _resolve_serving_fidelity(baseline_server_args=args, state_max_model_len=8192)
    assert out["max_model_len"] == 8192      # dedicated state field beats the blob
    assert out["mem_fraction"] == pytest.approx(0.85)


def test_resolve_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MODEL_LEN", "1234")
    monkeypatch.setenv("GPU_MEMORY_UTILIZATION", "0.7")
    out = _resolve_serving_fidelity(baseline_server_args="", state_max_model_len=0)
    assert out == {"max_model_len": 1234, "mem_fraction": pytest.approx(0.7)}


def test_resolve_omits_unresolved_knobs() -> None:
    """No source => key omitted entirely (GEAK adapter applies its own default)."""
    assert _resolve_serving_fidelity(baseline_server_args="", state_max_model_len=0) == {}
    # Only one knob present => only that key emitted.
    out = _resolve_serving_fidelity(
        baseline_server_args="--gpu-memory-utilization 0.9", state_max_model_len=0
    )
    assert out == {"mem_fraction": 0.9}
    assert "max_model_len" not in out


def test_resolve_ignores_malformed_values() -> None:
    out = _resolve_serving_fidelity(
        baseline_server_args="--max-model-len notanint --gpu-memory-utilization abc",
        state_max_model_len=0,
    )
    assert out == {}
