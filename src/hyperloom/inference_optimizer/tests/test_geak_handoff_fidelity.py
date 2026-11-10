# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Serving-launch fidelity forwarding for the GEAK handoff."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._gpu_pin import _resolve_serving_fidelity


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
    assert out["max_model_len"] == 8192
    assert out["mem_fraction"] == pytest.approx(0.85)


def test_resolve_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MODEL_LEN", "1234")
    monkeypatch.setenv("GPU_MEMORY_UTILIZATION", "0.7")
    out = _resolve_serving_fidelity(baseline_server_args="", state_max_model_len=0)
    assert out == {"max_model_len": 1234, "mem_fraction": pytest.approx(0.7)}


def test_resolve_omits_unresolved_knobs() -> None:
    """No source => key omitted entirely (GEAK adapter applies its own default)."""
    assert _resolve_serving_fidelity(baseline_server_args="", state_max_model_len=0) == {}
    out = _resolve_serving_fidelity(baseline_server_args="--gpu-memory-utilization 0.9", state_max_model_len=0)
    assert out == {"mem_fraction": 0.9}
    assert "max_model_len" not in out


def test_resolve_repeated_flag_takes_last_value() -> None:
    """The server's argparse keeps the last occurrence, so the handoff must forward that one."""
    args = "--max-model-len 4096 --gpu-memory-utilization 0.8 --max-model-len 8192 --gpu-memory-utilization=0.9"
    out = _resolve_serving_fidelity(baseline_server_args=args, state_max_model_len=0)
    assert out == {"max_model_len": 8192, "mem_fraction": 0.9}


def test_resolve_survives_unbalanced_quote() -> None:
    out = _resolve_serving_fidelity(baseline_server_args='--max-model-len 4096 "unbalanced', state_max_model_len=0)
    assert out == {"max_model_len": 4096}


def test_resolve_ignores_malformed_values() -> None:
    out = _resolve_serving_fidelity(
        baseline_server_args="--max-model-len notanint --gpu-memory-utilization abc",
        state_max_model_len=0,
    )
    assert out == {}
