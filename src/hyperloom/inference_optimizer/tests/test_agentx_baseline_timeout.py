# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX baseline cap, and why the aiter cold/warm probe cannot supply it.

The probe counts .so files across the whole aiter JIT dir and calls anything
above 20 warm. The signature it is really about is (model, dtype, TP,
max_model_len) -- and AgentX is precisely what moves max_model_len, from the
synthetic 6144 to the model's native window. So the first AgentX round on any
box that has run synthetic work is reported WARM, handed the 7800s cap, and then
pays the 30+ minute first-compile for a signature it has never built.

Measured rounds are 4774s (SGLang) and 6676s (vLLM) before that compile; with it
and a cold corpus mmap the worst case is ~9316s. Neither the warm cap (7800) nor
the cold cap (9000) covers that, and neither escape hatch reaches it -- the
cold-cap env var is only read when the probe says cold, and nothing writes
params["timeout_sec"] for a baseline. A baseline timeout kills the session
before the search starts, so this is not a risk but a certainty.

The synthetic path must keep the probe-driven behaviour exactly.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors.baseline import (
    AGENTX_BASELINE_OVERHEAD_SEC,
    AGENTX_CANON_WARMUP_CONC,
    AGENTX_CANON_WARMUP_GRACE_SEC,
    AGENTX_DEFAULT_DURATION_SEC,
    BASELINE_DEFAULT_TIMEOUT_SEC,
    BaselineExecutor,
    agentx_baseline_timeout_sec,
    agentx_warmup_grace_conc,
    agentx_warmup_grace_sec,
)

_MEASURED_VLLM_SEC = 6676  # the E4 round, warm corpus, no first-compile
_FIRST_COMPILE_SEC = 1800  # the cold-start comment's own "30+ minutes"
_COLD_CORPUS_SEC = 840  # the client's own "4-14 min" upper bound


def _clear(monkeypatch):
    for k in (
        "HYPERLOOM_AGENTX",
        "AGENTX_DURATION",
        "AGENTX_BASELINE_TIMEOUT_SEC",
        "AGENTX_BASELINE_OVERHEAD_SEC",
        "AGENTX_WARMUP_GRACE_PERIOD",
        # An inherited CONC would silently scale the warmup share and make every
        # cap assertion below concurrency-dependent.
        "CONC",
        "AGENTX_WARMUP_GRACE_CONC",
    ):
        monkeypatch.delenv(k, raising=False)


# --- the derived cap -----------------------------------------------------------


def test_default_cap_covers_the_measured_worst_case(monkeypatch):
    """The number has to clear a measurement, not a hunch."""
    _clear(monkeypatch)
    worst = _MEASURED_VLLM_SEC + _FIRST_COMPILE_SEC + _COLD_CORPUS_SEC
    cap = agentx_baseline_timeout_sec()
    assert cap == AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC
    assert cap > worst, f"cap {cap} does not clear the modelled worst case {worst}"


def test_neither_stock_cap_would_have_covered_it():
    """Why this exists at all: both existing caps lose to the same arithmetic."""
    from hyperloom.orchestrator.actions.executors._aiter_jit import (
        BASELINE_COLD_START_TIMEOUT_SEC,
    )

    worst = _MEASURED_VLLM_SEC + _FIRST_COMPILE_SEC + _COLD_CORPUS_SEC
    assert BASELINE_DEFAULT_TIMEOUT_SEC < worst
    assert BASELINE_COLD_START_TIMEOUT_SEC < worst


def test_cap_tracks_the_measurement_window(monkeypatch):
    """Derived, not pinned: a longer window must not need a second edit."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_DURATION", "7200")
    assert agentx_baseline_timeout_sec() == 7200 + AGENTX_BASELINE_OVERHEAD_SEC


def test_overhead_budget_is_tunable(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "3600")
    assert agentx_baseline_timeout_sec() == AGENTX_DEFAULT_DURATION_SEC + 3600


def test_default_overhead_warns_it_may_not_fit_every_model(monkeypatch, caplog):
    """A raw aiperf run against Kimi-K3 (conc=64) measured warmup alone taking
    ~12075s -- longer than this whole default cap. Nothing here can tell a
    long-context/slow-prefill model apart from GLM-5.2/Qwen3.8, the models this
    constant was measured on, so the gap must be surfaced instead of silently
    assumed to fit every model.
    """
    _clear(monkeypatch)
    with caplog.at_level("WARNING"):
        agentx_baseline_timeout_sec()
    assert any("AGENTX_BASELINE_OVERHEAD_SEC" in r.message for r in caplog.records)


def test_explicit_overhead_override_suppresses_the_warning(monkeypatch, caplog):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "20000")
    with caplog.at_level("WARNING"):
        agentx_baseline_timeout_sec()
    assert not any("AGENTX_BASELINE_OVERHEAD_SEC" in r.message for r in caplog.records)


def test_overhead_tracks_the_warmup_grace_the_operator_set(monkeypatch):
    """The knob that bounds the warmup must also size the cap that has to cover it.

    A model whose warmup runs long is a model whose operator has already had to
    raise AGENTX_WARMUP_GRACE_PERIOD for the round to finish -- the Kimi-K3
    case, where the flat overhead was smaller than the warmup itself. Deriving
    from that same knob is what stops the two numbers disagreeing.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    grown = 14400 - AGENTX_CANON_WARMUP_GRACE_SEC
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC + grown)


def test_canonical_grace_reproduces_the_measured_constant(monkeypatch):
    """Splitting the constant must not move it: same inputs, same number."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", str(AGENTX_CANON_WARMUP_GRACE_SEC))
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC)


def test_explicit_overhead_outranks_the_derivation(monkeypatch):
    """A pinned overhead is an answer, not an input: the grace must not add to it."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "3600")
    assert agentx_baseline_timeout_sec() == AGENTX_DEFAULT_DURATION_SEC + 3600


def test_a_tuned_grace_suppresses_the_uncalibrated_warning(monkeypatch, caplog):
    """The warning is about nobody having sized this model, not about the default."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    with caplog.at_level("WARNING"):
        agentx_baseline_timeout_sec()
    assert not any("AGENTX_BASELINE_OVERHEAD_SEC" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-1"])
def test_unparseable_grace_falls_back_to_canonical(monkeypatch, bad):
    """A typo in the grace must not shrink the cap below the measured default."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", bad)
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC)


# --- the CONC-scaled warmup floor ----------------------------------------------


@pytest.mark.parametrize("conc", ["1", "4", "8"])
def test_at_or_below_the_anchor_the_cap_is_unchanged(monkeypatch, conc):
    """Every round already validated at conc<=8 must keep its exact cap.

    The floor is a floor, not a re-derivation: anchoring at the lowest
    concurrency this repo has a measured agentic warmup for means the change is
    provably a no-op for the rounds that were measured with the old arithmetic.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("CONC", conc)
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC)


@pytest.mark.parametrize("conc", [16, 32, 64])
def test_the_warmup_share_scales_linearly_with_conc(monkeypatch, conc):
    """Warmup is per-lane requests x CONC lanes, so its budget must track CONC."""
    _clear(monkeypatch)
    monkeypatch.setenv("CONC", str(conc))
    grown = (AGENTX_CANON_WARMUP_GRACE_SEC * conc) // AGENTX_CANON_WARMUP_CONC - AGENTX_CANON_WARMUP_GRACE_SEC
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC + grown)


def test_the_floor_composes_with_an_operator_raised_grace(monkeypatch):
    """A grace the operator already raised is the thing that gets scaled.

    Scaling the canonical constant instead would throw away the only
    model-specific measurement in the derivation.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    scaled = (3600 * 32) // AGENTX_CANON_WARMUP_CONC
    grown = scaled - AGENTX_CANON_WARMUP_GRACE_SEC
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC + grown)


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-8", "8.5"])
def test_an_unusable_conc_leaves_the_derivation_alone(monkeypatch, bad):
    """A missing or malformed CONC must not move the cap in either direction."""
    _clear(monkeypatch)
    monkeypatch.setenv("CONC", bad)
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC)


def test_the_floor_never_shrinks_a_cap(monkeypatch):
    """Whatever CONC says, the cap may only grow -- an under-sized cap kills a
    round that would have finished, while an over-sized one costs a longer wait
    on a round that was hung anyway.
    """
    _clear(monkeypatch)
    base = agentx_baseline_timeout_sec()
    for conc in (1, 2, 4, 8, 9, 16, 24, 32, 64, 128):
        monkeypatch.setenv("CONC", str(conc))
        assert agentx_baseline_timeout_sec() >= base


def test_a_pinned_cap_outranks_the_conc_floor(monkeypatch):
    """The explicit escape hatch stays the last word, as it is for every other input."""
    _clear(monkeypatch)
    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "12345")
    assert agentx_baseline_timeout_sec() == 12345


def test_a_pinned_overhead_outranks_the_conc_floor(monkeypatch):
    """A pinned overhead is an answer, not an input -- same rule as the grace."""
    _clear(monkeypatch)
    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "3600")
    assert agentx_baseline_timeout_sec() == AGENTX_DEFAULT_DURATION_SEC + 3600


def test_the_scaling_is_announced(monkeypatch, caplog):
    """A cap that moved silently is a cap nobody can reconcile against a log."""
    _clear(monkeypatch)
    monkeypatch.setenv("CONC", "32")
    with caplog.at_level("INFO"):
        agentx_baseline_timeout_sec()
    assert any("CONC=32" in r.getMessage() for r in caplog.records)


def test_explicit_cap_wins_outright(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_DURATION", "7200")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "12345")
    assert agentx_baseline_timeout_sec() == 12345


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-1"])
def test_unparseable_or_nonpositive_env_falls_back(monkeypatch, bad):
    """A typo must not silently produce a cap of zero."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", bad)
    monkeypatch.setenv("AGENTX_DURATION", bad)
    assert agentx_baseline_timeout_sec() == AGENTX_DEFAULT_DURATION_SEC + AGENTX_BASELINE_OVERHEAD_SEC


# --- wiring into _resolve_timeout ----------------------------------------------


def _probe_must_not_run(*_a, **_k):
    raise AssertionError("the aiter probe was consulted on the AgentX path")


def test_agentx_bypasses_the_probe(monkeypatch):
    """Asking the probe first would return WARM and the cap that cannot work."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline._probe_aiter_jit_cache",
        _probe_must_not_run,
    )
    assert BaselineExecutor()._resolve_timeout({}) == agentx_baseline_timeout_sec()


def test_explicit_task_param_still_outranks_agentx(monkeypatch):
    """The pre-existing highest-priority override keeps its place."""
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert BaselineExecutor()._resolve_timeout({"timeout_sec": 4242}) == 4242


def test_synthetic_path_still_uses_the_probe(monkeypatch):
    """Zero regression: AgentX off must reach the probe and its warm default."""
    seen = {}

    def _fake_probe():
        seen["called"] = True
        return {"probe_status": "found", "is_cold": False, "path": "/x", "kernel_count": 99, "size_mb": 1}

    _clear(monkeypatch)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline._probe_aiter_jit_cache",
        _fake_probe,
    )
    assert BaselineExecutor()._resolve_timeout({}) == BASELINE_DEFAULT_TIMEOUT_SEC
    assert seen.get("called") is True


def test_synthetic_cold_start_bump_is_untouched(monkeypatch):
    from hyperloom.orchestrator.actions.executors._aiter_jit import (
        BASELINE_COLD_START_TIMEOUT_SEC,
    )

    _clear(monkeypatch)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline._probe_aiter_jit_cache",
        lambda: {"probe_status": "found", "is_cold": True, "path": "/x", "kernel_count": 1, "size_mb": 1},
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline.sweep_stale_aiter_locks_if_dead",
        lambda: {},
    )
    assert BaselineExecutor()._resolve_timeout({}) == BASELINE_COLD_START_TIMEOUT_SEC


# --- the warmup grace both layers must read -------------------------------------


def test_the_cap_and_the_clients_bound_come_from_one_function(monkeypatch):
    """The cap's warmup share IS the client's bound, or the round gets cut early.

    This is the invariant the whole helper exists for: ``aiperf_client.sh``
    passes this number to aiperf as ``--warmup-grace-period``, and that is what
    actually stops the warmup. If the cap budgets more than the client is
    allowed to spend, warmup ends mid-corpus and the round reports a
    prefix-reuse figure taken before the cache was populated.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    grace = agentx_warmup_grace_sec()
    _NON_WARMUP = AGENTX_BASELINE_OVERHEAD_SEC - AGENTX_CANON_WARMUP_GRACE_SEC
    assert grace == 3600 * 32 // AGENTX_CANON_WARMUP_CONC
    assert agentx_baseline_timeout_sec() == (AGENTX_DEFAULT_DURATION_SEC + _NON_WARMUP + grace)


@pytest.mark.parametrize("conc", ["1", "4", "8"])
def test_the_grace_is_untouched_at_or_below_the_anchor(monkeypatch, conc):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", conc)
    assert agentx_warmup_grace_sec() == 3600


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-1"])
def test_an_unusable_grace_falls_back_to_canonical(monkeypatch, bad):
    """A typo must not hand the client a warmup bound of zero."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", bad)
    assert agentx_warmup_grace_sec() == AGENTX_CANON_WARMUP_GRACE_SEC


@pytest.mark.parametrize("bad", ["", "abc", "0", "-8", "8.5"])
def test_an_unusable_conc_leaves_the_grace_alone(monkeypatch, bad):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", bad)
    assert agentx_warmup_grace_sec() == 3600


def test_the_grace_never_shrinks(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    for conc in (1, 2, 4, 8, 9, 16, 32, 64, 128):
        monkeypatch.setenv("CONC", str(conc))
        assert agentx_warmup_grace_sec() >= 3600


# --- the grace declares which concurrency it was measured at -------------------


def test_the_anchor_defaults_to_the_repo_measurement(monkeypatch):
    """Unset means "8", which is where this repo's measurements start."""
    _clear(monkeypatch)
    assert agentx_warmup_grace_conc() == AGENTX_CANON_WARMUP_CONC


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-8", "8.5"])
def test_an_unusable_anchor_falls_back_rather_than_dividing_by_it(monkeypatch, bad):
    """A zero or garbage anchor must not reach the division."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", bad)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    assert agentx_warmup_grace_conc() == AGENTX_CANON_WARMUP_CONC
    assert agentx_warmup_grace_sec() == 3600 * 32 // AGENTX_CANON_WARMUP_CONC


def test_a_grace_measured_at_a_higher_conc_is_not_double_counted(monkeypatch):
    """The defect a hardcoded anchor causes, stated as a test.

    An operator who measured 14400s of warmup at CONC=16 and says so must get
    14400s back at CONC=16 -- not 28800s, which is what an 8-anchored scaler
    silently produces and what forced the operator to hand-convert instead.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "16")
    monkeypatch.setenv("CONC", "16")
    assert agentx_warmup_grace_sec() == 14400


def test_the_declared_anchor_drives_the_ratio(monkeypatch):
    """Both numbers, not one: 14400s at CONC=16 doubles at CONC=32."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "16")
    monkeypatch.setenv("CONC", "32")
    assert agentx_warmup_grace_sec() == 28800


def test_below_the_declared_anchor_the_grace_is_untouched(monkeypatch):
    """Identity holds at the anchor the operator declared, not at a fixed 8."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "16")
    for conc in (2, 4, 8, 16):
        monkeypatch.setenv("CONC", str(conc))
        assert agentx_warmup_grace_sec() == 14400


def test_declaring_the_default_anchor_changes_nothing(monkeypatch):
    """Explicit 8 and unset must be the same derivation, not two code paths."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    implicit = agentx_warmup_grace_sec()
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", str(AGENTX_CANON_WARMUP_CONC))
    assert agentx_warmup_grace_sec() == implicit


def test_the_cap_follows_the_declared_anchor_too(monkeypatch):
    """The subprocess cap and the client bound stay one number under any anchor."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "16")
    monkeypatch.setenv("CONC", "32")
    non_warmup = AGENTX_BASELINE_OVERHEAD_SEC - AGENTX_CANON_WARMUP_GRACE_SEC
    assert agentx_baseline_timeout_sec() == AGENTX_DEFAULT_DURATION_SEC + non_warmup + 28800
