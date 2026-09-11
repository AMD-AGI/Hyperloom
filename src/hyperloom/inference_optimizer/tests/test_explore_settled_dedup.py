"""The cross-round dedup guard must never cost the search an optimization.

Skipping a variant the ledger has already settled saves a server boot and a full
benchmark. It is also the one kind of pruning that can silently make Hyperloom
worse, because a variant that lost on its own can win once something it composes
with has landed. So most of these tests are about what the guard refuses to skip:
anything whose stack moved, whose workload moved, that never reached a verdict,
or that landed close enough to the KEEP threshold that noise could have decided
it.
"""
from __future__ import annotations

from hyperloom.orchestrator.actions.executors.explore import (
    REPEAT_NOISE_PCT,
    SETTLED_MARGIN,
    _settled_against_same_stack,
)

WS = "workload-sig-abc"
BASE = 15235.83
THRESH = 1.0

# Distance from the threshold beyond which a repeat cannot plausibly flip.
BAND = REPEAT_NOISE_PCT * SETTLED_MARGIN


def _prior(**over):
    """A conclusive prior result, far below the KEEP threshold."""
    d = {
        "workload_signature": WS,
        "status": "succeeded",
        "base_tput": BASE,
        "gain_pct": THRESH - BAND - 0.5,
        "outcome": "REVERT",
        "round_id": "explore-001",
    }
    d.update(over)
    return d


def test_settled_repeat_is_skipped():
    """Same config, same stack, same workload, verdict nowhere near the line."""
    assert _settled_against_same_stack(_prior(), WS, BASE, THRESH) is True


def test_a_moved_stack_is_retested():
    """A KEEP advanced the running baseline, so the variant may compose
    differently now and has to be measured again."""
    assert _settled_against_same_stack(_prior(), WS, BASE * 1.04, THRESH) is False


def test_a_drifted_baseline_is_retested():
    """A baseline that merely drifted on re-measurement is indistinguishable
    from a moved stack, so it falls the safe way."""
    assert _settled_against_same_stack(_prior(), WS, BASE + 0.01, THRESH) is False


def test_a_different_workload_is_retested():
    assert _settled_against_same_stack(_prior(), "other-sig", BASE, THRESH) is False


def test_a_variant_that_never_got_a_verdict_is_retested():
    """Crashed, killed on overtime, or failed at warmup: the ledger holds no
    result to reuse, only the fact that it did not finish."""
    for status in ("failed", "killed_overtime", ""):
        assert _settled_against_same_stack(
            _prior(status=status, gain_pct=None), WS, BASE, THRESH
        ) is False


def test_a_result_near_the_threshold_is_retested():
    """Within a couple of noise envelopes of the KEEP line, the verdict may have
    been decided by the measurement rather than by the variant. That deserves a
    second sample, not a skip."""
    for gain in (THRESH - BAND * 0.5, THRESH + BAND * 0.5, THRESH, THRESH - BAND):
        assert _settled_against_same_stack(
            _prior(gain_pct=gain), WS, BASE, THRESH
        ) is False


def test_a_clear_winner_is_also_settled():
    """The guard is symmetric: a variant that won by a wide margin is as settled
    as one that lost by one, and re-running it re-derives a known number."""
    assert _settled_against_same_stack(
        _prior(gain_pct=THRESH + BAND + 5.0, outcome="KEEP"), WS, BASE, THRESH
    ) is True


def test_missing_fields_are_retested():
    for missing in ("base_tput", "gain_pct"):
        assert _settled_against_same_stack(
            _prior(**{missing: None}), WS, BASE, THRESH
        ) is False


def test_no_baseline_yet_is_retested():
    assert _settled_against_same_stack(_prior(), WS, 0.0, THRESH) is False


def test_only_the_decisive_real_session_variants_are_settled():
    """The audited session tested six variants against a stack that never moved,
    and their gains were clustered just under the 1% line. Only the two that lost
    by more than the noise band are settled; the three that landed inside it are
    re-measured, which is the guard declining to prune on a number it cannot
    distinguish from noise."""
    settled = {"chunked-prefill-8192": -0.812, "sched-conservativeness-03": -0.157}
    inside_band = {"attn-aiter": 0.045, "cuda-graph-max-bs-256": 0.241,
                   "mem-frac-092": 0.165}
    for name, g in settled.items():
        assert _settled_against_same_stack(
            _prior(gain_pct=g), WS, BASE, THRESH
        ) is True, name
    for name, g in inside_band.items():
        assert _settled_against_same_stack(
            _prior(gain_pct=g), WS, BASE, THRESH
        ) is False, name


def test_the_variant_that_failed_at_warmup_is_not_skipped():
    """One of the six never produced a number at all. It is not settled."""
    assert _settled_against_same_stack(
        _prior(status="failed", gain_pct=None, outcome="REVERT"), WS, BASE, THRESH
    ) is False
