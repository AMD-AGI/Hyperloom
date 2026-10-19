# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX budget profile, search-scope collapse, and the session-level guards."""

from __future__ import annotations

import argparse

import pytest

from hyperloom.inference_optimizer.cli import (
    _apply_agentx_budget_profile,
    _preflight_agentx_backend,
)
from hyperloom.inference_optimizer.cli.bootstrap import (
    AGENTX_MEASUREMENT_EPOCH,
    agentx_state_is_stale,
)
from hyperloom.inference_optimizer.cli.parser import DEFAULT_MAX_HOURS


def _budget_args(**over) -> argparse.Namespace:
    base = dict(
        # What the parser produces when ``--max-hours`` is absent: the flag
        # carries no argparse default, so the profile sees ``None``, not 2.0.
        max_hours=None,
        conc_sweep_total_budget_sec=9000,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _off(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _on(monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)


# --- budget profile -----------------------------------------------------------


def test_budget_profile_is_noop_without_agentx(monkeypatch):
    _off(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert vars(args) == vars(_budget_args())


def test_budget_profile_does_not_expand_benchmark_caps(native_benchmark, monkeypatch):
    _on(monkeypatch, native_benchmark)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.conc_sweep_total_budget_sec == 9000
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts

    monkeypatch.delenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", raising=False)
    assert resolve_benchmark_timeouts()[1] == 7800


def test_budget_profile_never_touches_max_hours(native_benchmark, monkeypatch):
    """``--max-hours`` is the operator's contract with the scheduler."""
    _on(monkeypatch, native_benchmark)
    args = _budget_args(max_hours=2.0)
    _apply_agentx_budget_profile(args)
    assert args.max_hours == 2.0


def test_the_note_fires_when_the_operator_passed_no_budget(native_benchmark, monkeypatch, capsys):
    _on(monkeypatch, native_benchmark)
    _apply_agentx_budget_profile(_budget_args())
    assert "--max-hours" in capsys.readouterr().err


def test_the_note_stays_quiet_for_a_budget_the_operator_typed(native_benchmark, monkeypatch, capsys):
    """An explicit value is a deliberate choice, even at the default's number."""
    _on(monkeypatch, native_benchmark)
    _apply_agentx_budget_profile(_budget_args(max_hours=DEFAULT_MAX_HOURS))
    assert capsys.readouterr().err == ""


def test_budget_profile_preserves_operator_values(native_benchmark, monkeypatch):
    """A value the operator typed is left exactly as typed."""
    _on(monkeypatch, native_benchmark)
    args = _budget_args(
        conc_sweep_total_budget_sec=1200,
    )
    _apply_agentx_budget_profile(args)
    assert args.conc_sweep_total_budget_sec == 1200


# --- explicit-flag detection --------------------------------------------------
def test_bypass_guard_allows_magpie(native_benchmark, monkeypatch):
    _on(monkeypatch, native_benchmark)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_is_inert_without_agentx(monkeypatch):
    _off(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_rejects_the_silent_combination(native_benchmark, monkeypatch):
    """AgentX + bypass runs synthetic work and labels it AgentX."""
    _on(monkeypatch, native_benchmark)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace())
    assert ei.value.code == 2


def test_guard_rejects_agentx_with_a_scriptable_framework(native_benchmark, monkeypatch):
    """The other way for the switch to no-op while every gate still fires."""
    _on(monkeypatch, native_benchmark)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace(framework="xdit"))
    assert ei.value.code == 2


def test_guard_allows_agentx_with_a_serving_framework(native_benchmark, monkeypatch):
    _on(monkeypatch, native_benchmark)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    for fw in ("vllm", "sglang"):
        _preflight_agentx_backend(argparse.Namespace(framework=fw))  # must not raise


def test_scriptable_guard_is_inert_without_agentx(monkeypatch):
    """A scriptable run on its own is perfectly normal."""
    _off(monkeypatch)
    _preflight_agentx_backend(argparse.Namespace(framework="xdit"))  # must not raise


# --- resume staleness ---------------------------------------------------------


class _St:
    def __init__(self, mode="", epoch=0, backend="magpie"):
        self.benchmark_mode = mode
        self.agentx_epoch = epoch
        self.agentx_backend = backend


def test_resume_accepts_matching_agentx_state(native_benchmark, monkeypatch):
    _on(monkeypatch, native_benchmark)
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH)) == ""


def test_resume_accepts_matching_synthetic_state(monkeypatch):
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("synthetic", 0)) == ""


def test_resume_rejects_mode_switch(native_benchmark, monkeypatch):
    """The KEEP ledger keys on server args alone, so the rows would collide."""
    _on(monkeypatch, native_benchmark)
    assert "benchmark_mode" in agentx_state_is_stale(_St("synthetic", 0))
    _off(monkeypatch)
    assert "benchmark_mode" in agentx_state_is_stale(_St("agentx", 1))


def test_resume_rejects_a_backend_switch(native_benchmark, monkeypatch):
    """Magpie and MLPerf sessions must not anchor each other, epoch unchanged."""
    _on(monkeypatch, native_benchmark)
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH, "magpie")) == ""
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    reason = agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH, "magpie"))
    assert "backend" in reason
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH, "mlperf")) == ""


def test_resume_rejects_a_missing_backend(native_benchmark, monkeypatch):
    _on(monkeypatch, native_benchmark)
    assert "backend" in agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH, ""))


def test_resume_rejects_stale_agentx_epoch(native_benchmark, monkeypatch):
    """Same knobs, different workload: the old numbers cannot anchor."""
    _on(monkeypatch, native_benchmark)
    reason = agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH - 1))
    assert "epoch" in reason


def test_resume_tolerates_sessions_predating_the_field(monkeypatch):
    """An empty mode means "not asserted", not "mismatch"."""
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("", 0)) == ""
