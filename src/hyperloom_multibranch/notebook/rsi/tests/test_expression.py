# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Frontier scorers written as expressions: what the syntax lets through, what it refuses, and the values they give."""

from __future__ import annotations

import math
import re

import pytest

from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis
from hyperloom_multibranch.notebook.rsi.expression import (
    FEATURES,
    MAX_CHARS,
    MAX_NODES,
    Expression,
    ExpressionError,
    ExpressionScorer,
    features,
)
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, Readings, ScoringContext, score_with
from hyperloom_multibranch.notebook.rsi.tests.trees import TASK, branch, tree

POLICY = FrontierPolicy("throughput")
CLIMBING = branch("s1", [(600.0, 1.0), (1800.0, 3.0)], test_times=(0.0, 600.0, 1200.0, 1800.0))
SESSION = tree("a", CLIMBING, branch("s2", [], opened_s=1200.0))
NOW = 2400.0
CONTEXT = ScoringContext(TASK, (), POLICY, SESSION.as_of(NOW))


def _value(text: str, **values: float) -> float:
    return Expression.read(text).value(values)


def test_an_expression_reads_arithmetic_comparisons_choices_and_the_functions() -> None:
    assert _value("2 * trend_pct + 1", trend_pct=1.5) == 4.0
    assert _value("trend_pct ** 2 / 4 - 1", trend_pct=4.0) == 3.0
    assert _value("1 if trend_pct > 0 and not young else -1", trend_pct=0.5, young=0.0) == 1.0
    assert _value("0 < trend_pct <= 1 or commits", trend_pct=2.0, commits=0.0) == 0.0
    assert _value("max(trend_pct, prior_remaining_pct, 0)", trend_pct=-1.0, prior_remaining_pct=0.5) == 0.5
    assert _value("clip(abs(trend_pct), 0, 2) + sqrt(4) + log1p(0)", trend_pct=-3.0) == 4.0
    assert Expression.read(" min(tests, commits) ") == Expression.read("min(tests, commits)")
    assert Expression.read("min(tests, commits)").names == frozenset({"tests", "commits"})


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("__import__('os')", "a call of __import__"),
        ("trend_pct.real", "Attribute"),
        ("[tests][0]", "List"),
        ("lambda: 1", "Lambda"),
        ("'a'", "the constant 'a'"),
        ("True", "the constant True"),
        ("speed", "the name speed, which is no feature"),
        ("max(tests)", "max of 1 arguments"),
        ("min(tests, key=tests)", "a call that is not"),
        ("tests % 2", "the operator Mod"),
        ("tests in (1, 2)", "that comparison"),
        ("(x := 1)", "NamedExpr"),
        ("1 +", "not an expression"),
        ("", "1 to"),
        ("1+" * MAX_CHARS + "1", f"1 to {MAX_CHARS}"),
        ("+".join(["tests"] * 30), f"at most {MAX_NODES} syntax nodes"),
    ],
)
def test_the_syntax_refuses_anything_but_numbers_features_arithmetic_and_the_functions(text: str, why: str) -> None:
    with pytest.raises(ExpressionError, match=re.escape(why)):
        Expression.read(text)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("1 / tests", "division by zero"),
        ("sqrt(0 - tests - 1)", "math domain error"),
        ("log1p(-1 - tests)", "math domain error"),
        ("10 ** 400", "range"),
        ("1e308 * 10", "not a finite number"),
    ],
)
def test_an_expression_without_a_finite_value_fails(text: str, why: str) -> None:
    with pytest.raises(ExpressionError, match=why):
        Expression.read(text).value({"tests": 0.0})


def test_an_expression_scorer_estimates_its_value_with_the_features_it_read_as_components() -> None:
    scorer = ExpressionScorer.read("trend-plus-v1", "trend_pct + 0.1 * commits + headroom_pct")
    readings = Readings(headroom_pct=5.0)

    estimate = scorer.estimate(CLIMBING.as_of(NOW), NOW, CONTEXT, readings)

    assert estimate.basis is FrontierBasis.CUSTOM
    assert estimate.expected_gain_pct == pytest.approx(2.0 + 0.2 + 5.0)
    assert estimate.components == {"commits": 2.0, "trend_pct": 2.0}
    assert score_with(scorer, CLIMBING.as_of(NOW), NOW, CONTEXT, readings).scorer == "trend-plus-v1"
    with pytest.raises(ExpressionError, match="no scorer label"):
        ExpressionScorer.read("Trend Plus", "trend_pct")


def test_every_feature_has_a_finite_value_and_the_vocabulary_lists_them_all() -> None:
    young = CONTEXT.tree.branches[2] if CONTEXT.tree is not None else None
    assert young is not None and young.name == "s2"
    for item in (CLIMBING.as_of(NOW), young):
        values = features(item, NOW, CONTEXT, Readings())
        assert set(values) == set(FEATURES) and all(math.isfinite(value) for value in values.values())

    values = features(CLIMBING.as_of(NOW), NOW, CONTEXT, Readings(True, 1.0, 2.0, 0.5))

    assert (values["open_branches"], values["best_gain_pct"], values["since_last_commit_s"]) == (2.0, 3.0, 600.0)
    assert (values["has_headroom"], values["amdahl_ceiling_pct"], values["pending_pct"]) == (1.0, 2.0, 0.5)
    assert (values["observing"], values["young"]) == (1.0, 0.0)
    assert features(young, NOW, CONTEXT, Readings())["young"] == 1.0
    assert values["elapsed_share"] == pytest.approx(NOW / SESSION.budget_seconds)
