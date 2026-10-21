# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frontier scorers a session may choose from, and the pool that dates them."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.notebook.rsi import scorers
from hyperloom_multibranch.notebook.rsi.bandit import BANDIT_LABEL, BanditScorer
from hyperloom_multibranch.notebook.rsi.expression import ExpressionScorer
from hyperloom_multibranch.notebook.rsi.frontier import FRONTIER_V1, FRONTIER_V1_LABEL
from hyperloom_multibranch.notebook.rsi.scorers import (
    MAX_EXPRESSIONS,
    PoolConflict,
    PoolRecord,
    ScorerPool,
    ScorerSource,
    custom_scorers,
    registered_scorers,
    scorer_named,
)

TREND_TWICE = "trend_pct * 2"


def _expression(label: str, at: float, text: str = "trend_pct") -> PoolRecord:
    return PoolRecord(label, ScorerSource.EXPRESSION, at, text, "why", {"model": "claude"})


def test_no_scorer_is_added_yet_and_the_bandit_is_registered_the_first_time_a_session_sees_it(tmp_path: Path) -> None:
    pool = ScorerPool(tmp_path)

    (bandit,) = registered_scorers(pool, 100.0)
    (again,) = registered_scorers(pool, 200.0)

    assert custom_scorers() == ()
    assert (bandit.scorer, bandit.since, again.since) == (BanditScorer(), 100.0, 100.0)
    assert [(item.label, item.source) for item in pool.records()] == [(BANDIT_LABEL, ScorerSource.CODE)]
    assert registered_scorers(None, 300.0)[0].since == 300.0


def test_a_scorer_added_to_custom_scorers_is_chosen_from_like_the_built_in_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    added = ExpressionScorer.read("acme-trend-v1", "trend_pct")
    monkeypatch.setattr(scorers, "custom_scorers", lambda: (added, ExpressionScorer.read(BANDIT_LABEL, "tests")))

    found = registered_scorers(ScorerPool(tmp_path), 100.0)

    assert [(item.scorer.label, item.since) for item in found] == [(BANDIT_LABEL, 100.0), ("acme-trend-v1", 100.0)]
    assert scorer_named("acme-trend-v1", None) == added


def test_the_pool_keeps_each_scorer_once_and_offers_its_most_recent_expressions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pool = ScorerPool(tmp_path)
    for n in range(MAX_EXPRESSIONS + 1):
        pool.register(_expression(f"expr-{n}", float(n)))

    kept = pool.register(_expression("expr-0", 99.0))
    with pytest.raises(PoolConflict):
        pool.register(_expression("expr-0", 99.0, "commits"))
    with pytest.raises(ValueError, match="no scorer label"):
        pool.register(_expression("Not A Label", 1.0))
    (pool.root / "broken.json").write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.rsi.scorers"):
        offered = pool.expressions()

    assert kept.registered_at == 0.0
    assert [item.label for item in offered] == [f"expr-{n}" for n in range(1, MAX_EXPRESSIONS + 1)]
    assert "broken.json" in caplog.text


def test_a_session_chooses_among_the_pools_expressions_but_one_that_no_longer_reads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pool = ScorerPool(tmp_path)
    pool.register(_expression("trend-v2", 5.0, TREND_TWICE))
    pool.register(_expression("broken-v1", 6.0, "trend_pct +"))

    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.rsi.scorers"):
        found = registered_scorers(pool, 10.0)

    assert [(item.scorer.label, item.since) for item in found] == [(BANDIT_LABEL, 10.0), ("trend-v2", 5.0)]
    assert "broken-v1" in caplog.text


def test_a_label_names_frontier_v1_a_scorer_in_code_or_a_pooled_expression(tmp_path: Path) -> None:
    pool = ScorerPool(tmp_path)
    pool.register(_expression("trend-v2", 5.0, TREND_TWICE))

    assert scorer_named(FRONTIER_V1_LABEL, pool) is FRONTIER_V1
    assert scorer_named(BANDIT_LABEL, None) == BanditScorer()
    assert scorer_named("trend-v2", pool) == ExpressionScorer.read("trend-v2", TREND_TWICE)
    assert scorer_named("nowhere", pool) is None
