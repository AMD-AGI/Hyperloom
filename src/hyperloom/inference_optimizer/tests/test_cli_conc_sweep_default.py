# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests pinning the ``--enable-conc-sweep`` default to True (SharedState + CLI flag + env opt-out)."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import _build_parser
from hyperloom.orchestrator.state.shared_state import SharedState


def _parse(*extra: str):
    parser = _build_parser()
    return parser.parse_args(["optimize", *extra])


def test_shared_state_default_enables_conc_sweep():
    state = SharedState()
    assert state.conc_sweep_enabled is True, (
        "SharedState default for conc_sweep_enabled must be True. "
        "If you intentionally flipped this off, update this test "
        "with the rationale in the commit message."
    )


def test_cli_default_enables_conc_sweep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", raising=False)
    ns = _parse()
    assert ns.enable_conc_sweep is True


def test_cli_no_enable_conc_sweep_disables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", raising=False)
    ns = _parse("--no-enable-conc-sweep")
    assert ns.enable_conc_sweep is False


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "FALSE", "Off"])
def test_cli_env_var_can_opt_out(monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", val)
    ns = _parse()
    assert ns.enable_conc_sweep is False, f"env={val!r} should opt out of conc-sweep, got True"


@pytest.mark.parametrize("val", ["", "1", "true", "yes", "on", "something"])
def test_cli_env_var_truthy_or_unset_keeps_default_on(
    monkeypatch: pytest.MonkeyPatch,
    val: str,
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", val)
    ns = _parse()
    assert ns.enable_conc_sweep is True, f"env={val!r} should leave default-on intact, got False"


def test_cli_no_flag_overrides_truthy_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", "1")
    ns = _parse("--no-enable-conc-sweep")
    assert ns.enable_conc_sweep is False


def test_cli_default_disables_baseline_double_run(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", raising=False)
    ns = _parse()
    assert ns.baseline_double_run is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
def test_cli_env_var_enables_baseline_double_run(
    monkeypatch: pytest.MonkeyPatch,
    val: str,
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", val)
    ns = _parse()
    assert ns.baseline_double_run is True


def test_cli_baseline_double_run_flag_overrides_false_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    ns = _parse("--baseline-double-run")
    assert ns.baseline_double_run is True


def test_cli_no_baseline_double_run_flag_overrides_true_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "1")
    ns = _parse("--no-baseline-double-run")
    assert ns.baseline_double_run is False
