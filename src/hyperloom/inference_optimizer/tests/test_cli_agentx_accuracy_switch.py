# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--no-agentx-accuracy``: on by default, recorded with the session, refused outside native AgentX."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import bootstrap as cb
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.state.shared_state import SharedState

_OPT_OUT = "--no-agentx-accuracy"


def _parse(*extra: str):
    return _build_parser().parse_args(["optimize", "--model", "/models/m", *extra])


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.policy import gate as policy

    # setenv, not delenv: configure_benchmark writes the config path into os.environ, and only a recorded key is
    # restored after the test.
    for key in ("HYPERLOOM_BENCHMARK_CONFIG", "HYPERLOOM_AGENTX", "HYPERLOOM_AGENTIC_BACKEND"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)


def _seed(tmp_path, monkeypatch: pytest.MonkeyPatch, native_benchmark, *extra: str) -> SharedState:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    return cb._seed_shared_state(
        tmp_path,
        _parse(*extra),
        session_id="s",
        framework="sglang",
        settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
    )


def test_the_round_is_on_by_default(tmp_path, monkeypatch, native_benchmark):
    assert _parse().no_agentx_accuracy is False
    assert _seed(tmp_path, monkeypatch, native_benchmark).agentx_accuracy is True


def test_the_opt_out_persists_across_a_reload(tmp_path, monkeypatch, native_benchmark):
    _seed(tmp_path, monkeypatch, native_benchmark, _OPT_OUT)
    assert SharedState.load_or_init(tmp_path).agentx_accuracy is False


def test_native_agentx_accepts_the_opt_out(native_benchmark):
    args = _build_parser().parse_args(["optimize", "--benchmark-config", str(native_benchmark), _OPT_OUT])
    configure_benchmark(args)
    assert args.no_agentx_accuracy is True


@pytest.mark.parametrize(
    "env",
    [{"HYPERLOOM_AGENTX": "0"}, {"HYPERLOOM_AGENTX": "1", "HYPERLOOM_AGENTIC_BACKEND": "mlperf"}],
    ids=["synthetic", "mlperf"],
)
def test_other_workloads_refuse_the_opt_out(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="requires native AgentX"):
        configure_benchmark(_parse(_OPT_OUT))


def test_a_resume_cannot_turn_the_round_off():
    conflict = cb.constraint_budget_resume_conflict(_parse(_OPT_OUT), SharedState(agentx_accuracy=True))
    assert "accuracy round" in conflict


@pytest.mark.parametrize("flags", [(), (_OPT_OUT,)])
def test_a_resume_keeps_the_recorded_opt_out(flags):
    assert cb.constraint_budget_resume_conflict(_parse(*flags), SharedState(agentx_accuracy=False)) == ""
