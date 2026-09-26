# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests pinning the ``--enable-conc-sweep`` defaults."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.state.shared_state import SharedState


def _parse(*extra: str):
    parser = _build_parser()
    return parser.parse_args(["optimize", *extra])


def test_shared_state_default_enables_conc_sweep():
    state = SharedState()
    assert state.conc_sweep_enabled is True, (
        "SharedState default for conc_sweep_enabled must be True. "
        "If you intentionally flipped this off, update this test "
        "with the rationale in the commit message."
    )


def test_cli_omitted_conc_sweep_defers_to_the_benchmark_mode():
    assert _parse().enable_conc_sweep is None
    assert _parse("--enable-conc-sweep").enable_conc_sweep is True


def test_cli_no_enable_conc_sweep_disables():
    ns = _parse("--no-enable-conc-sweep")
    assert ns.enable_conc_sweep is False


def test_cli_exposes_no_baseline_double_run_field(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", raising=False)
    ns = _parse()
    assert not hasattr(ns, "baseline_double_run")


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
def test_cli_ignores_env_var_for_baseline_double_run(
    monkeypatch: pytest.MonkeyPatch,
    val: str,
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", val)
    ns = _parse()
    assert not hasattr(ns, "baseline_double_run")


def test_cli_rejects_baseline_double_run_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "0")
    with pytest.raises(SystemExit) as exc_info:
        _parse("--baseline-double-run")
    assert exc_info.value.code == 2


def test_cli_rejects_no_baseline_double_run_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "1")
    with pytest.raises(SystemExit) as exc_info:
        _parse("--no-baseline-double-run")
    assert exc_info.value.code == 2


_EXPECTED_CONCS = [256, 128, 64, 32, 16, 8, 4, 2]
_EXPECTED_AGENTX_CONCS = [1, 4, 8, 10, 14, 20, 28]


def test_the_ladder_has_one_source():
    """The default ladder used to be restated in four places and hand-synced."""
    from hyperloom.orchestrator.kernel.conc_sweep import (
        AGENTX_DEFAULT_CONCS,
        DEFAULT_CONCS,
        default_concs_for_mode,
    )

    assert DEFAULT_CONCS == _EXPECTED_CONCS
    assert AGENTX_DEFAULT_CONCS == _EXPECTED_AGENTX_CONCS
    assert default_concs_for_mode("") == _EXPECTED_CONCS
    assert default_concs_for_mode("synthetic") == _EXPECTED_CONCS
    assert default_concs_for_mode("agentx") == _EXPECTED_AGENTX_CONCS
    assert default_concs_for_mode("AgentX") == _EXPECTED_AGENTX_CONCS


def test_the_resolved_ladder_is_the_callers_to_keep():
    """A caller that mutates what it was handed must not move the module default."""
    from hyperloom.orchestrator.kernel.conc_sweep import DEFAULT_CONCS, default_concs_for_mode

    resolved = default_concs_for_mode("")
    resolved.append(1)
    assert DEFAULT_CONCS == _EXPECTED_CONCS


def test_an_omitted_flag_is_distinguishable_from_a_typed_ladder():
    """The flag defaults to None so the mode can pick; a typed value is a string."""
    assert _parse().conc_sweep_concs is None
    assert _parse("--conc-sweep-concs", "4,8,16").conc_sweep_concs == "4,8,16"


def test_shared_state_seeds_its_ladder_rather_than_restating_one():
    """A bare state carries no ladder; bootstrap seeds it from the workload."""
    assert SharedState().conc_sweep_concs == []


class TestTheLadderFallsBackToTheWorkload:
    """An unset flag resolves against the workload the session actually runs."""

    def _parse(self, raw, mode):
        from argparse import Namespace

        from hyperloom.inference_optimizer.cli import bootstrap as cb

        return cb._parse_conc_sweep_concs(Namespace(conc_sweep_concs=raw), mode)

    def test_synthetic(self):
        assert self._parse(None, "synthetic") == _EXPECTED_CONCS

    def test_agentx(self):
        assert self._parse(None, "agentx") == _EXPECTED_AGENTX_CONCS

    def test_a_typed_ladder_outranks_both(self):
        assert self._parse("4,8,16", "agentx") == [4, 8, 16]

    def test_an_all_garbage_ladder_falls_back_to_the_workload(self):
        assert self._parse("x,y", "agentx") == _EXPECTED_AGENTX_CONCS


class TestTheSweepSwitchFallsBackToTheWorkload:
    """An omitted ``--enable-conc-sweep`` is off under AgentX, where each rung is a 3600s window."""

    @pytest.fixture(autouse=True)
    def _no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hyperloom.inference_optimizer.cli import bootstrap as cb
        from hyperloom.orchestrator.policy import gate as policy

        monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
        monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
        monkeypatch.setattr(cb, "_resolve_reference_recipe", lambda _args: ("", {}, "", "", {}))
        monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
        monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)

    def _seeded(self, tmp_path, monkeypatch: pytest.MonkeyPatch, *, agentx: bool, extra: tuple[str, ...] = ()):
        from hyperloom.inference_optimizer.cli import bootstrap as cb

        if agentx:
            monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
        else:
            monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
        args = _parse("--model", "/models/m", *extra)
        return cb._seed_shared_state(tmp_path, args, session_id="s")

    def test_synthetic_sweeps_by_default(self, tmp_path, monkeypatch):
        assert self._seeded(tmp_path, monkeypatch, agentx=False).conc_sweep_enabled is True

    def test_agentx_does_not_sweep_by_default(self, tmp_path, monkeypatch):
        assert self._seeded(tmp_path, monkeypatch, agentx=True).conc_sweep_enabled is False

    def test_agentx_sweeps_when_asked(self, tmp_path, monkeypatch):
        state = self._seeded(tmp_path, monkeypatch, agentx=True, extra=("--enable-conc-sweep",))
        assert state.conc_sweep_enabled is True

    def test_synthetic_skips_when_told(self, tmp_path, monkeypatch):
        state = self._seeded(tmp_path, monkeypatch, agentx=False, extra=("--no-enable-conc-sweep",))
        assert state.conc_sweep_enabled is False
