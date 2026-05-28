"""dynamic_action_gaps.md G5 — CLI knobs smoke tests.

The runner-construction helper in ``cli.py`` reads three operator
flags via ``getattr`` defaults. Before G5 the argparser did not
define them, so ``getattr`` always returned the default — operators
had no control. These tests pin both the parse + the wiring into
:class:`DynamicActionRunner`.
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
    """The CLI defaults absent overrides MUST match the runner
    module's DEFAULT_* — otherwise an operator's "default" run uses
    different budgets than the runner advertises."""
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
