# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests pinning the ``--enable-conc-sweep`` defaults."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser
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


def test_cli_default_enables_conc_sweep():
    ns = _parse()
    assert ns.enable_conc_sweep is True


def test_cli_no_enable_conc_sweep_disables():
    ns = _parse("--no-enable-conc-sweep")
    assert ns.enable_conc_sweep is False


def test_cli_exposes_no_baseline_double_run_field(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", raising=False)
    ns = _parse()
    assert not hasattr(ns, "baseline_double_run")


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
def test_cli_ignores_env_var_for_baseline_double_run(
    monkeypatch: pytest.MonkeyPatch,
    val: str,
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", val)
    ns = _parse()
    assert not hasattr(ns, "baseline_double_run")


def test_cli_rejects_baseline_double_run_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    with pytest.raises(SystemExit) as exc_info:
        _parse("--baseline-double-run")
    assert exc_info.value.code == 2


def test_cli_rejects_no_baseline_double_run_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "1")
    with pytest.raises(SystemExit) as exc_info:
        _parse("--no-baseline-double-run")
    assert exc_info.value.code == 2


_EXPECTED_CONCS = [256, 128, 64, 32, 16, 8, 4, 2]
_EXPECTED_CONCS_STR = "256,128,64,32,16,8,4,2"


def test_shared_state_default_concs():
    """SharedState default ladder must be high-to-low for single-server arm reuse."""
    state = SharedState()
    assert state.conc_sweep_concs == _EXPECTED_CONCS, (
        f"SharedState.conc_sweep_concs default mismatch. "
        f"Expected {_EXPECTED_CONCS}, got {state.conc_sweep_concs}. "
        "Update all three sources: DEFAULT_CONCS, parser.py, shared_state.py."
    )


def test_cli_default_concs():
    """CLI --conc-sweep-concs default must match DEFAULT_CONCS."""
    ns = _parse()
    assert ns.conc_sweep_concs == _EXPECTED_CONCS_STR


def test_conc_sweep_module_default_concs():
    """conc_sweep.DEFAULT_CONCS must be kept in sync with CLI/state defaults."""
    from hyperloom.orchestrator.kernel.conc_sweep import DEFAULT_CONCS

    assert DEFAULT_CONCS == _EXPECTED_CONCS, (
        f"DEFAULT_CONCS mismatch: expected {_EXPECTED_CONCS}, got {DEFAULT_CONCS}"
    )


def test_cli_custom_concs():
    """Custom --conc-sweep-concs is parsed as a raw string (parsing happens in run_optimize)."""
    ns = _parse("--conc-sweep-concs", "4,8,16")
    assert ns.conc_sweep_concs == "4,8,16"
