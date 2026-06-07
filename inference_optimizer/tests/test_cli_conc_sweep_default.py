# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the ``--enable-conc-sweep`` default.

Pins three things at once:

1. ``SharedState`` dataclass field ``conc_sweep_enabled`` defaults to
   ``True`` — so any consumer that builds a ``SharedState`` outside the
   CLI path (tests, scripted runners) gets the same behaviour.
2. The CLI ``--enable-conc-sweep`` flag resolves to ``True`` when the
   operator passes no flag and no env var.
3. ``INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP=0`` / ``false`` / ``no`` /
   ``off`` flips the CLI default to ``False`` (explicit opt-out).
4. ``--no-enable-conc-sweep`` always wins over env / default.

If somebody later flips the default back to off, all four of these
assertions will trip — leaving an audit trail for the choice.
"""

from __future__ import annotations

import pytest

from inference_optimizer.cli import _build_parser
from inference_optimizer.orchestrator.shared_state import SharedState


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
    assert ns.enable_conc_sweep is False, (
        f"env={val!r} should opt out of conc-sweep, got True"
    )


@pytest.mark.parametrize("val", ["", "1", "true", "yes", "on", "something"])
def test_cli_env_var_truthy_or_unset_keeps_default_on(
    monkeypatch: pytest.MonkeyPatch, val: str,
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", val)
    ns = _parse()
    assert ns.enable_conc_sweep is True, (
        f"env={val!r} should leave default-on intact, got False"
    )


def test_cli_no_flag_overrides_truthy_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", "1")
    ns = _parse("--no-enable-conc-sweep")
    assert ns.enable_conc_sweep is False
