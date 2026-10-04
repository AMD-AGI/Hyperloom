"""How the shared warm-start search bounds resolve."""

from __future__ import annotations

import pytest

from kernelforge.knowledge import warmstart_policy as policy


def test_the_defaults_are_what_ships():
    assert policy.top_k() == 10
    assert policy.min_claimed_speedup() == pytest.approx(0.3)
    assert policy.budget_sec() == pytest.approx(1800.0)


@pytest.mark.parametrize(
    ("variable", "resolve", "value", "expected"),
    [
        ("FORGE_KB_WARMSTART_TOP_K", policy.top_k, "4", 4),
        ("FORGE_KB_WARMSTART_MIN_SPEEDUP", policy.min_claimed_speedup, "0.75", 0.75),
        ("FORGE_KB_WARMSTART_BUDGET_SEC", policy.budget_sec, "600", 600.0),
    ],
)
def test_each_bound_can_be_overridden(monkeypatch, variable, resolve, value, expected):
    monkeypatch.setenv(variable, value)
    assert resolve() == pytest.approx(expected)


@pytest.mark.parametrize("junk", ["", "   ", "not-a-number", "0", "-3"])
@pytest.mark.parametrize(
    ("variable", "resolve", "default"),
    [
        ("FORGE_KB_WARMSTART_TOP_K", policy.top_k, 10),
        ("FORGE_KB_WARMSTART_MIN_SPEEDUP", policy.min_claimed_speedup, 0.3),
        ("FORGE_KB_WARMSTART_BUDGET_SEC", policy.budget_sec, 1800.0),
    ],
)
def test_an_unusable_override_falls_back_to_the_default(
    monkeypatch,
    junk,
    variable,
    resolve,
    default,
):
    """A zero or negative bound would disable the search, not widen it."""
    monkeypatch.setenv(variable, junk)
    assert resolve() == pytest.approx(default)


def test_a_claim_under_the_floor_is_refused():
    assert policy.below_floor(0.0047) is True
    assert policy.below_floor(0.29) is True


def test_a_claim_at_or_over_the_floor_stands():
    assert policy.below_floor(0.3) is False
    assert policy.below_floor(1.5) is False


def test_a_candidate_that_claims_nothing_is_not_under_the_floor():
    """Nothing was claimed, so nothing is contradicted.

    Such a record still has to earn its place by measurement; it just is not thrown out before being measured.
    """
    assert policy.below_floor(None) is False
