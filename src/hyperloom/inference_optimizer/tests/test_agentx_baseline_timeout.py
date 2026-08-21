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
    AGENTX_DEFAULT_DURATION_SEC,
    BASELINE_DEFAULT_TIMEOUT_SEC,
    BaselineExecutor,
    agentx_baseline_timeout_sec,
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
