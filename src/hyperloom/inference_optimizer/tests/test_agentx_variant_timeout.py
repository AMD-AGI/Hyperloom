###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The variant hard cap has to survive a canonical AgentX warmup.

Found by running the AgentX path end to end with nothing disabled: a GLM-5.2
variant launched 09:47:41 was killed at 11:47:41.575 -- the synthetic cap minus
its reserve -- with twenty-plus connections dropping in the same millisecond
while the server was still prefilling with 55 requests running. aiperf reports
that as ``warmup_failure`` because a cancelled root warmup credit is terminal,
so the subprocess kill never appears in the abort reason and the round looks
like a workload problem.

The caps involved are all sized for the synthetic 1024/1024 shape: 7800s for
integrate, 2400s for explore, 1800s for the conc sweep. ``baseline`` already
derives an AgentX-aware cap; only that path got it.
"""

from hyperloom.orchestrator.actions.executors._grid_runner import (
    agentx_variant_timeout_sec,
)

# the three synthetic caps that killed real rounds
SYNTHETIC_CAPS = (1800, 2400, 7800)


def test_default_path_is_untouched(monkeypatch):
    """AgentX off must behave exactly as before -- it is an opt-in branch.

    This is the property that matters most: AgentX is a new benchmark branch,
    not the default, so with it disabled every cap has to come back unchanged.
    """
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    for cap in (*SYNTHETIC_CAPS, 99, 36000):
        assert agentx_variant_timeout_sec(cap) == cap


def test_default_path_untouched_even_with_agentx_vars_present(monkeypatch):
    """Leftover AGENTX_* vars must not switch the branch on by themselves."""
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "28800")
    assert agentx_variant_timeout_sec(7800) == 7800


def test_agentx_raises_the_synthetic_defaults(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "7200")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    for cap in SYNTHETIC_CAPS:
        assert agentx_variant_timeout_sec(cap) == 10800


def test_never_lowers_an_operator_choice(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "7200")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    assert agentx_variant_timeout_sec(36000) == 36000


def test_tracks_the_baseline_derivation(monkeypatch):
    """One number, not two: the cap follows baseline's own resolver."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "28800")
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    assert agentx_variant_timeout_sec(7800) == 32400

    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "50000")
    assert agentx_variant_timeout_sec(7800) == 50000


# --- the cap must survive a lost env var ---------------------------------------


class _StateWithMode:
    def __init__(self, mode):
        self.benchmark_mode = mode


def test_persisted_state_raises_the_cap_when_the_env_var_is_gone(monkeypatch):
    """The original report: a resumed session whose shell lost HYPERLOOM_AGENTX.

    ``benchmark_mode`` is stamped at seed precisely so it survives a restart.
    Without consulting it the round reads as synthetic here and is killed by the
    synthetic cap mid-warmup -- the failure this helper exists to prevent,
    reached by the one route it did not cover.
    """
    from hyperloom.orchestrator.actions.executors._grid_runner import agentx_variant_timeout_sec

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    for k in ("AGENTX_BASELINE_TIMEOUT_SEC", "AGENTX_BASELINE_OVERHEAD_SEC", "AGENTX_WARMUP_GRACE_PERIOD"):
        monkeypatch.delenv(k, raising=False)

    assert agentx_variant_timeout_sec(1800) == 1800
    raised = agentx_variant_timeout_sec(1800, shared_state=_StateWithMode("agentx"))
    assert raised > 1800


def test_a_synthetic_session_state_does_not_raise_the_cap(monkeypatch):
    """Zero effect on the default path: a synthetic benchmark_mode changes nothing."""
    from hyperloom.orchestrator.actions.executors._grid_runner import agentx_variant_timeout_sec

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert agentx_variant_timeout_sec(1800, shared_state=_StateWithMode("synthetic")) == 1800
    assert agentx_variant_timeout_sec(1800, shared_state=None) == 1800
