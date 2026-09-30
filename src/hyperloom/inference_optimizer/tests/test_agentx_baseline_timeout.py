# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed benchmark caps remain independent of AgentX client warmup settings."""

from __future__ import annotations

import sys

import pytest

from hyperloom.orchestrator.actions.executors import _agentx_timeouts as _timeouts
from hyperloom.orchestrator.actions.executors._agentx_timeouts import (
    AGENTX_CANON_WARMUP_CONC,
    AGENTX_CANON_WARMUP_GRACE_SEC,
    agentx_warmup_grace_conc,
    agentx_warmup_grace_sec,
)
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor
from hyperloom.orchestrator.actions.executors._workload_envs import apply_agentx_switch


def _clear(monkeypatch):
    _timeouts._AGENTX_SAID.clear()
    for key in (
        "HYPERLOOM_AGENTX",
        "AGENTX_WARMUP_GRACE_PERIOD",
        "AGENTX_WARMUP_GRACE_CONC",
        "CONC",
        "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("agentx", ["0", "1"])
def test_baseline_cap_does_not_expand_with_workload(monkeypatch, tmp_path, agentx):
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", agentx)
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "20000")
    executor = BaselineExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    assert executor._resolve_timeout({"timeout_sec": 99}) == 7800
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "123.5")
    assert executor._resolve_timeout({}) == 123.5


def test_profile_retains_nonbenchmark_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "99")
    executor = ProfileExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    assert executor.benchmark_watchdog is False
    assert executor._resolve_timeout({}) == 14400
    assert executor._resolve_timeout({"timeout_sec": 15000}) == 15000


def test_agentx_switch_keeps_profile_yaml_cap(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    bench = {"framework": "sglang", "timeout_seconds": 14400}
    apply_agentx_switch(bench, active=True)
    assert bench["timeout_seconds"] == 14400


@pytest.mark.parametrize("conc", ["1", "4", "8"])
def test_the_grace_is_untouched_at_or_below_the_anchor(monkeypatch, conc):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
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
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    monkeypatch.setenv("CONC", bad)
    assert agentx_warmup_grace_sec() == 3600


def test_the_grace_never_shrinks(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    for conc in (1, 2, 4, 8, 9, 16, 32, 64, 128):
        monkeypatch.setenv("CONC", str(conc))
        assert agentx_warmup_grace_sec() >= 3600


# --- the grace declares which concurrency it was measured at -------------------


def test_the_anchor_defaults_to_the_repo_measurement(monkeypatch):
    """Unset means "8", which is where this repo's measurements start."""
    _clear(monkeypatch)
    assert agentx_warmup_grace_conc() == AGENTX_CANON_WARMUP_CONC


@pytest.mark.parametrize("bad", ["", "  ", "abc", "0", "-8", "8.5"])
def test_an_unusable_anchor_disables_scaling_rather_than_dividing_by_it(monkeypatch, bad):
    """A zero or garbage anchor must not reach the division -- and must not be quietly replaced by the default either."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", bad)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    assert agentx_warmup_grace_conc() == AGENTX_CANON_WARMUP_CONC
    assert agentx_warmup_grace_sec() == 3600


def test_a_grace_measured_at_a_higher_conc_is_not_double_counted(monkeypatch):
    """The defect a hardcoded anchor causes, stated as a test."""
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


def test_declaring_the_default_anchor_is_what_enables_the_floor(monkeypatch):
    """Declaring 8 is not a no-op: it is the statement that turns scaling on."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    assert agentx_warmup_grace_sec() == 3600
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", str(AGENTX_CANON_WARMUP_CONC))
    assert agentx_warmup_grace_sec() == 3600 * 32 // AGENTX_CANON_WARMUP_CONC


def test_a_whole_number_written_with_a_decimal_point_is_honoured(monkeypatch):
    """\"16.0\" is an anchor of 16, not a missing anchor."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "14400.0")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "16.0")
    monkeypatch.setenv("CONC", "16")
    assert agentx_warmup_grace_conc() == 16
    assert agentx_warmup_grace_sec() == 14400


@pytest.mark.parametrize("bad", ["8.5", "abc", "", "-16.0", "0"])
def test_a_fractional_or_unparseable_anchor_is_still_rejected(monkeypatch, bad):
    """A non-integral concurrency is a typo, not an intent."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", bad)
    assert agentx_warmup_grace_conc() == AGENTX_CANON_WARMUP_CONC


def test_an_exponent_form_whole_number_is_accepted(monkeypatch):
    """``1e3`` is 1000, unambiguously. The bar is integrality, not notation."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "1e3")
    assert agentx_warmup_grace_conc() == 1000


def test_a_changed_derivation_still_speaks_up(monkeypatch, caplog):
    """Deduping on the payload, not on a bare flag: new numbers are new news."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    with caplog.at_level("INFO"):
        monkeypatch.setenv("CONC", "16")
        agentx_warmup_grace_sec()
        monkeypatch.setenv("CONC", "32")
        agentx_warmup_grace_sec()
    scaled = [r for r in caplog.records if "scaling the warmup share" in r.getMessage()]
    assert len(scaled) == 2, [r.getMessage() for r in scaled]


def test_declaring_the_anchor_is_what_turns_scaling_on(monkeypatch):
    """The floor is opt-in, and one line buys it."""
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    assert agentx_warmup_grace_sec() == 3600
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    assert agentx_warmup_grace_sec() == 14400
