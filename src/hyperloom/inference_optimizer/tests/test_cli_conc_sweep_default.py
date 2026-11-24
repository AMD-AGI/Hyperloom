# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests pinning the ``--enable-conc-sweep`` defaults."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import sweeps_after_close
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.state.shared_state import SharedState


def _parse(*extra: str):
    parser = _build_parser()
    return parser.parse_args(["optimize", *extra])


def test_the_post_session_conc_sweep_is_on_unless_turned_off():
    assert _parse().enable_conc_sweep is True
    assert _parse("--enable-conc-sweep").enable_conc_sweep is True
    assert _parse("--no-enable-conc-sweep").enable_conc_sweep is False


def test_the_operators_points_are_parsed_distinct_and_ascending():
    assert _parse().conc_sweep_concs == ()
    assert _parse("--conc-sweep-concs", "16,4,16").conc_sweep_concs == (4, 16)
    with pytest.raises(SystemExit):
        _parse("--conc-sweep-concs", "4,x")
    with pytest.raises(SystemExit):
        _parse("--conc-sweep-concs", "0,4")


@pytest.mark.parametrize(
    ("enabled", "closed", "stop_reason", "sweeps"),
    [
        (True, True, "time_exhausted", True),
        (True, True, "signal", False),
        (True, False, "time_exhausted", False),
        (False, True, "time_exhausted", False),
    ],
)
def test_a_session_sweeps_after_close_only_when_asked_closed_and_not_stopped_by_a_signal(
    enabled: bool, closed: bool, stop_reason: str, sweeps: bool
):
    state = SharedState(conc_sweep_enabled=enabled, close_sequence_done=closed)

    assert sweeps_after_close(state, stop_reason) is sweeps


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


class TestTheSeededSweep:
    """The session records the switch and the points as given, for a native AgentX workload too."""

    @pytest.fixture(autouse=True)
    def _no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hyperloom.inference_optimizer.cli import bootstrap as cb
        from hyperloom.orchestrator.policy import gate as policy

        monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
        monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
        monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
        monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)

    def _seeded(
        self, native_benchmark, tmp_path, monkeypatch: pytest.MonkeyPatch, *, agentx: bool, extra: tuple[str, ...] = ()
    ):
        from hyperloom.inference_optimizer.cli import bootstrap as cb

        if agentx:
            monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
        else:
            monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
        args = _parse("--model", "/models/m", *extra)
        return cb._seed_shared_state(
            tmp_path,
            args,
            session_id="s",
            framework="sglang",
            settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
        )

    @pytest.mark.parametrize("agentx", [False, True], ids=["synthetic", "agentx"])
    def test_an_asked_for_sweep_and_its_points_are_seeded(self, native_benchmark, tmp_path, monkeypatch, agentx):
        extra = ("--enable-conc-sweep", "--conc-sweep-concs", "8,2")
        state = self._seeded(native_benchmark, tmp_path, monkeypatch, agentx=agentx, extra=extra)
        assert (state.conc_sweep_enabled, state.conc_sweep_concs) == (True, [2, 8])

    def test_native_agentx_accepts_a_concurrency_sweep(self, native_benchmark, monkeypatch):
        from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark

        monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", "")
        args = _parse("--benchmark-config", str(native_benchmark), "--enable-conc-sweep")
        configure_benchmark(args)
        assert args.enable_conc_sweep is True
