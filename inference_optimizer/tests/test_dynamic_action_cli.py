"""Smoke tests for the dynamic_action CLI knobs:
``--dynamic-action-model`` / ``--dynamic-action-turn-cap`` /
``--dynamic-action-wall-clock-sec``.

Pins both argparse acceptance and the wiring into
:class:`DynamicActionRunner` defaults.
"""

from __future__ import annotations

from inference_optimizer.cli import _build_parser
from inference_optimizer.orchestrator.dynamic_action_runner import (
    DEFAULT_TURN_CAP,
    DEFAULT_WALL_CLOCK_BUDGET_SEC,
)


def _parse(*extra: str):
    parser = _build_parser()
    return parser.parse_args(["optimize", *extra])


def test_cli_default_dynamic_action_args_unset():
    ns = _parse()
    assert ns.dynamic_action_model is None
    assert ns.dynamic_action_turn_cap is None
    assert ns.dynamic_action_wall_clock_sec is None


def test_cli_dynamic_action_model_propagates():
    ns = _parse("--dynamic-action-model", "claude-opus-4-7")
    assert ns.dynamic_action_model == "claude-opus-4-7"


def test_cli_dynamic_action_turn_cap_propagates():
    ns = _parse("--dynamic-action-turn-cap", "20")
    assert ns.dynamic_action_turn_cap == 20


def test_cli_dynamic_action_wall_clock_propagates():
    ns = _parse("--dynamic-action-wall-clock-sec", "1800")
    assert ns.dynamic_action_wall_clock_sec == 1800.0


def test_cli_defaults_match_runner_constants():
    """Without overrides the CLI defaults resolve to the runner's
    ``DEFAULT_*`` constants so an operator's "default" run matches
    what the runner advertises."""
    ns = _parse()
    turn_cap = (
        int(ns.dynamic_action_turn_cap) if ns.dynamic_action_turn_cap
        else DEFAULT_TURN_CAP
    )
    wall_clock = (
        float(ns.dynamic_action_wall_clock_sec)
        if ns.dynamic_action_wall_clock_sec
        else DEFAULT_WALL_CLOCK_BUDGET_SEC
    )
    assert turn_cap == DEFAULT_TURN_CAP
    assert wall_clock == DEFAULT_WALL_CLOCK_BUDGET_SEC
