###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The producer side of the AgentX timeout defect.

Raising the variant cap was necessary and not sufficient. Integrate passes an
explicit ``timeout_sec`` into the re-baseline task, and ``_resolve_timeout``
deliberately lets an explicit param outrank the AgentX derivation -- a contract
with its own test. So the raise belongs where the param is produced, next to
the existing cold-start raise, which exists for exactly the same reason: an
explicit param suppresses the executor's own sizing branch.

Measured on Qwen3.8: a round whose server answered all 685 chat/completions
with 200 was cut at exactly its 7200s param, mid-warmup, after which the client
could no longer connect. aiperf reports the cancelled warmup credit as
``warmup_failure``, so nothing in the abort reason names the timeout.
"""

from types import SimpleNamespace

from hyperloom.orchestrator.kernel.request_handlers import (
    _agentx_rebaseline_timeout,
)

# the two values observed killing real rounds
OBSERVED = (7200, 9000)


def test_default_path_is_untouched(monkeypatch):
    """AgentX off: the resolved value passes through, exactly as before."""
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    for value in (*OBSERVED, 60, 50000):
        assert _agentx_rebaseline_timeout(value) == value


def test_default_path_untouched_with_stale_agentx_vars(monkeypatch):
    """A leftover AGENTX_* var must not switch the raise on by itself."""
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "28800")
    assert _agentx_rebaseline_timeout(7200) == 7200


def test_raises_the_observed_killers(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "28800")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    for value in OBSERVED:
        assert _agentx_rebaseline_timeout(value) == 32400


def test_never_lowers_a_larger_value(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "7200")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    assert _agentx_rebaseline_timeout(50000) == 50000


def test_tracks_the_baseline_derivation(monkeypatch):
    """One number, not two: it follows baseline's own resolver."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "7200")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    assert _agentx_rebaseline_timeout(7200) == 10800

    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "44000")
    assert _agentx_rebaseline_timeout(7200) == 44000


def test_persisted_benchmark_mode_raises_without_the_env_var(monkeypatch):
    """A re-baseline driven from a subprocess that never inherited
    ``HYPERLOOM_AGENTX`` must still get the raise from the session's
    persisted ``benchmark_mode`` -- otherwise it reproduces the exact
    mid-warmup kill this function exists to prevent.
    """
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "28800")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    shared_state = SimpleNamespace(benchmark_mode="agentx")
    assert _agentx_rebaseline_timeout(7200, shared_state=shared_state) == 32400


def test_unrelated_benchmark_mode_does_not_raise(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    shared_state = SimpleNamespace(benchmark_mode="synthetic")
    assert _agentx_rebaseline_timeout(7200, shared_state=shared_state) == 7200
