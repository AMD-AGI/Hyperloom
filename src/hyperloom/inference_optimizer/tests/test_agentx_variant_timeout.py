###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The variant hard cap has to survive a canonical AgentX warmup."""

from hyperloom.orchestrator.actions.executors._grid_runner import (
    agentx_variant_timeout_sec,
)

# the three synthetic caps that killed real rounds
SYNTHETIC_CAPS = (1800, 2400, 7800)


def test_default_path_is_untouched(monkeypatch):
    """AgentX off must behave exactly as before -- it is an opt-in branch."""
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
    """The original report: a resumed session whose shell lost HYPERLOOM_AGENTX."""
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
