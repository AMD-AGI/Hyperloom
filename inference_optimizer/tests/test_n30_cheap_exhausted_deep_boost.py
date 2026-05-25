"""N30 — cheap-exhausted deep-action boost in effective_score.

Background (SOLAR-10.7B TP=1 session 062837Z, May 2026):
After 1 baseline + 1 roofline + 3 params rounds + 1 backends round,
``last_cheap_delta_gain`` hit -0.063% (well below the N19c EPS=0.3%
threshold) -- so N19c's kernel_opt unlock fired. BUT
``kernel_opt.base_score=6.0`` was still ranked rank-6 by eff_score
(under params 9.5, backends 8.4, roofline 7.5, etc.) so the LLM
kept proposing params/backends instead of advancing to kernel_opt.

N30 bridges the gap: when ``_is_cheap_exhausted(shared_state)``
returns True (same predicate as N19c's gate), deep-family actions
(kernel_opt, operator_tuning, vendor_kernel_config,
deep_kernel_analysis, integrate) get their base_score multiplied by
``INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_DEEP_BOOST`` (default 2.0)
before the UCB + aging bonuses get added. SOLAR case becomes:
kernel_opt eff = 6.0 * 2.0 + bonuses ~= 11.1 -> rank 3 (above
params 9.06 / backends 8.32), and the kernel-owned deep_kernel_analysis
hits 17+ but LLM can't propose it directly -- so kernel_opt rises
to top-1 in the LLM's actionable set.

This file pins the contract:

* ``_is_cheap_exhausted`` returns True only when ALL three N19c
  prerequisites are satisfied (snapshot_id>=1, at least one cheap
  attempt, ``last_cheap_delta_gain < EPS``).
* ``effective_score`` with ``shared_state=None`` (test fixtures that
  predate N30) behaves exactly like pre-N30 -- no surprise behaviour
  change for callers that don't opt in.
* When cheap is exhausted AND shared_state is supplied, deep family
  actions get boosted; shallow / prep actions do NOT (no dampen).
* Env override (``INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_DEEP_BOOST``)
  controls the multiplier; bad values fall back to 2.0.
* Locked rows still short-circuit to _LOCKED_SCORE; boost does not
  bypass the lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from inference_optimizer.orchestrator.scoring import (
    ActionScore,
    _CHEAP_EXHAUSTED_DEEP_BOOST_DEFAULT,
    _CHEAP_EXHAUSTED_DEEP_BOOST_ENV,
    _is_cheap_exhausted,
    _resolve_cheap_exhausted_deep_boost,
    effective_score,
)
from inference_optimizer.orchestrator.action_registry import ActionMetadata


def _meta(family: str = "shallow", accuracy_risk: float = 0.0) -> ActionMetadata:
    """Minimal ActionMetadata stub for effective_score's risk math."""
    return ActionMetadata(
        name="stub",
        family=family,
        description="stub",
        accuracy_risk=accuracy_risk,
        crash_risk=0.0,
        cost_minutes_p50=1.0,
        cost_minutes_p75=2.0,
        expected_gain_pct=(0.0, 0.0),
        pipeline_phase="explore",
        typical_runtime_min=1.0,
    )


def _score(name: str, base: float, runs: int = 1) -> ActionScore:
    """Build an ActionScore. ``name`` isn't stored on the dataclass
    (it lives as the dict key in ``SharedState.action_scores``); we
    pass it separately to ``effective_score`` via ``action_name=``."""
    a = ActionScore(base_score=base, score_mult=1.0)
    a.runs = runs
    a.last_run_tick = 1
    a._name = name  # noqa: SLF001 -- test-only stash for asserts
    return a


@dataclass
class _StubState:
    """Drop-in SharedState shim for the N30 predicate.

    Carries only the attributes ``_is_cheap_exhausted`` reads
    (``last_trace_analyze``, ``backends_attempts``, ``params_attempts``,
    ``last_cheap_delta_gain``) so tests don't drag in the full
    SharedState dataclass.
    """
    last_trace_analyze: dict = None
    backends_attempts: list = None
    params_attempts: list = None
    last_cheap_delta_gain: float | None = None

    def __post_init__(self):
        if self.last_trace_analyze is None:
            self.last_trace_analyze = {}
        if self.backends_attempts is None:
            self.backends_attempts = []
        if self.params_attempts is None:
            self.params_attempts = []


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, raising=False)
    monkeypatch.delenv(
        "INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", raising=False,
    )


# ---------------------------------------------------------------------------
# _is_cheap_exhausted predicate
# ---------------------------------------------------------------------------


def test_is_cheap_exhausted_none_state():
    assert _is_cheap_exhausted(None) is False


def test_is_cheap_exhausted_no_snapshot():
    s = _StubState(
        last_trace_analyze={},  # snapshot_id missing -> 0
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=0.1,
    )
    assert _is_cheap_exhausted(s) is False


def test_is_cheap_exhausted_no_cheap_attempt():
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        last_cheap_delta_gain=0.1,
    )
    assert _is_cheap_exhausted(s) is False


def test_is_cheap_exhausted_delta_above_eps():
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=0.5,  # > EPS 0.3
    )
    assert _is_cheap_exhausted(s) is False


def test_is_cheap_exhausted_delta_none():
    """Defensive: a missing last_cheap_delta_gain (None) is treated as
    'no signal yet', not 'cheap exhausted'."""
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=None,
    )
    assert _is_cheap_exhausted(s) is False


def test_is_cheap_exhausted_solar_case():
    """The empirical SOLAR-10.7B case: delta below EPS triggers."""
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        backends_attempts=[{"task_id": "b1"}],
        params_attempts=[{"task_id": "p1"}, {"task_id": "p2"}],
        last_cheap_delta_gain=-0.063,
    )
    assert _is_cheap_exhausted(s) is True


def test_is_cheap_exhausted_negative_delta():
    """Defensive: negative delta also counts as 'exhausted' (cheap
    actually regressing, not finding gain)."""
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-1.0,
    )
    assert _is_cheap_exhausted(s) is True


def test_is_cheap_exhausted_eps_env_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CHEAP_EXHAUSTED_EPSILON", "0.05")
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=0.1,  # > 0.05 raised EPS -> NOT exhausted
    )
    assert _is_cheap_exhausted(s) is False


# ---------------------------------------------------------------------------
# Boost env resolver
# ---------------------------------------------------------------------------


def test_boost_default_is_2(monkeypatch):
    monkeypatch.delenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, raising=False)
    assert _resolve_cheap_exhausted_deep_boost() == 2.0
    assert _CHEAP_EXHAUSTED_DEEP_BOOST_DEFAULT == 2.0


def test_boost_env_overrides(monkeypatch):
    monkeypatch.setenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, "3.5")
    assert _resolve_cheap_exhausted_deep_boost() == 3.5


def test_boost_bad_env_falls_back(monkeypatch):
    for bad in ("0", "-1", "abc", ""):
        monkeypatch.setenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, bad)
        assert _resolve_cheap_exhausted_deep_boost() == 2.0


def test_boost_one_disables_boost(monkeypatch):
    """env=1.0 is a clean no-op operator-side disable."""
    monkeypatch.setenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, "1.0")
    assert _resolve_cheap_exhausted_deep_boost() == 1.0


# ---------------------------------------------------------------------------
# effective_score behavioural contract
# ---------------------------------------------------------------------------


def test_effective_score_back_compat_no_shared_state():
    """Pre-N30 callers passing shared_state=None must see identical
    behaviour to pre-N30: no boost, no surprises."""
    a = _score("kernel_opt", base=6.0)
    m = _meta(family="deep_kernel")
    eff_default = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    eff_explicit_none = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=None, action_name=a._name,
        )
    assert eff_default == eff_explicit_none


def test_effective_score_no_boost_when_not_exhausted():
    """shared_state supplied but predicate False -> no boost."""
    a = _score("kernel_opt", base=6.0)
    m = _meta(family="deep_kernel")
    s = _StubState()  # snapshot empty, no cheap attempts
    eff_no_state = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    eff_with_state = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        )
    assert eff_no_state == eff_with_state


def test_effective_score_boosts_deep_when_exhausted():
    """SOLAR scenario: cheap exhausted, kernel_opt boosted x2."""
    a = _score("kernel_opt", base=6.0, runs=0)
    m = _meta(family="deep_kernel")
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-0.063,
    )
    eff_no_boost = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    eff_boosted = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        )
    # base 6.0 * 2.0 = 12.0 vs 6.0 -> diff should be ~6 (plus UCB/aging
    # which are identical between the two calls). Assert at least 5
    # absolute units of boost; the exact value depends on UCB/aging
    # constants but the delta should NOT depend on them.
    assert eff_boosted - eff_no_boost == pytest.approx(6.0, abs=1e-6)


def test_effective_score_does_not_boost_shallow():
    """params / backends / sweep stay at base * mult, no boost."""
    a = _score("params", base=9.5, runs=0)
    m = _meta(family="shallow")
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-0.063,
    )
    eff_no_boost = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    eff_with_state = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        )
    assert eff_with_state == eff_no_boost


def test_effective_score_boost_by_name_fallback():
    """When meta.family isn't 'deep_kernel' (e.g. archived state.json
    pre-dating the family taxonomy), the action_name fallback set
    still triggers the boost for kernel_opt / operator_tuning / etc."""
    a = _score("kernel_opt", base=6.0, runs=0)
    m = _meta(family="other")  # NOT deep_kernel
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-0.063,
    )
    eff_with_state = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        )
    eff_no_state = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    # Should still get the +6.0 boost via name fallback.
    assert eff_with_state - eff_no_state == pytest.approx(6.0, abs=1e-6)


def test_effective_score_boost_respects_lock():
    """Locked rows still return _LOCKED_SCORE; boost does not unlock."""
    from inference_optimizer.orchestrator.scoring import _LOCKED_SCORE
    a = _score("kernel_opt", base=6.0)
    a.locked_reason = "params/grid_exhausted"
    m = _meta(family="deep_kernel")
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-0.063,
    )
    assert effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        ) == _LOCKED_SCORE


def test_effective_score_boost_env_factor(monkeypatch):
    """Operator can tune the boost via env -- 3.0 -> kernel_opt base
    6.0 * 3.0 = 18.0, diff = +12 vs no-boost."""
    monkeypatch.setenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, "3.0")
    a = _score("kernel_opt", base=6.0, runs=0)
    m = _meta(family="deep_kernel")
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-0.063,
    )
    eff_with_state = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        )
    eff_no_state = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    assert eff_with_state - eff_no_state == pytest.approx(12.0, abs=1e-6)


def test_effective_score_disabled_via_env(monkeypatch):
    """env=1.0 effectively disables N30 -- no boost applied."""
    monkeypatch.setenv(_CHEAP_EXHAUSTED_DEEP_BOOST_ENV, "1.0")
    a = _score("kernel_opt", base=6.0, runs=0)
    m = _meta(family="deep_kernel")
    s = _StubState(
        last_trace_analyze={"roofline_snapshot_id": 1},
        params_attempts=[{"task_id": "p1"}],
        last_cheap_delta_gain=-0.063,
    )
    eff_with_state = effective_score(
        a, meta=m, tick=10, total_runs=5, shared_state=s, action_name=a._name,
        )
    eff_no_state = effective_score(a, meta=m, tick=10, total_runs=5, action_name=a._name)
    assert eff_with_state == eff_no_state
