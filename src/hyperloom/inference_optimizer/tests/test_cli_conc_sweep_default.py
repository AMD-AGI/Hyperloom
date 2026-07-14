# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests pinning the ``--enable-conc-sweep`` default to True."""

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
