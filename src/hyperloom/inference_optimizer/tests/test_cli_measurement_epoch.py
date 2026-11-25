# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A seeded session records the build's measurement epoch, and only a session that carries it resumes."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import _reset_state_file, bootstrap as cb
from hyperloom.inference_optimizer.cli.launch_shape import _export_operator_launch_shape, _seed_env_from_resumed_state
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.tests._workload_envs_helpers import clear_env, materialize, write_config
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.policy import gate as policy

    for key in ("HYPERLOOM_BENCHMARK_CONFIG", "HYPERLOOM_AGENTX", "HYPERLOOM_AGENTIC_BACKEND", "DATASET_PATH"):
        monkeypatch.setenv(key, "")
    for key in ("NUM_PROMPTS", "INFERENCE_OPTIMIZER_EXTRA_ENV"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)


def _seed_and_reload(tmp_path) -> SharedState:
    cb._seed_shared_state(
        tmp_path,
        _build_parser().parse_args(["optimize", "--model", "/models/m"]),
        session_id="s",
        framework="sglang",
        settled=cb.SettledLaunch(compute_partition={}, gpu_power_settings={}, base=None),
    )
    return SharedState.load_or_init(tmp_path)


def test_a_synthetic_session_records_the_epoch_and_resumes(tmp_path):
    state = _seed_and_reload(tmp_path)
    assert state.measurement_epoch == cb.MEASUREMENT_EPOCH
    assert state.agentx_backend == ""
    assert cb.measurement_state_is_stale(state) == ""


def test_an_agentx_session_records_inferencex_and_resumes(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    state = _seed_and_reload(tmp_path)
    assert (state.measurement_epoch, state.agentx_backend) == (cb.MEASUREMENT_EPOCH, "inferencex")
    assert cb.measurement_state_is_stale(state) == ""


@pytest.mark.parametrize(("seeded", "resumed"), [("/data/prompts.jsonl", ""), ("", "/data/prompts.jsonl")])
def test_a_resume_that_sends_another_prompt_dataset_than_the_seed_recorded_is_refused(
    tmp_path, monkeypatch, seeded, resumed
):
    monkeypatch.setenv("DATASET_PATH", seeded)
    state = _seed_and_reload(tmp_path)
    assert (state.prompt_dataset, cb.measurement_state_is_stale(state)) == (seeded, "")

    monkeypatch.setenv("DATASET_PATH", resumed)
    assert "prompt dataset" in cb.measurement_state_is_stale(state)


def test_an_agentx_session_records_no_prompt_dataset_its_rounds_do_not_send(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.setenv("DATASET_PATH", "/data/prompts.jsonl")
    state = _seed_and_reload(tmp_path)
    assert (state.prompt_dataset, cb.measurement_state_is_stale(state)) == ("", "")


def test_a_state_written_before_the_switch_is_refused(tmp_path):
    (tmp_path / "state.json").write_text('{"benchmark_mode": "synthetic", "agentx_epoch": 1}', encoding="utf-8")
    assert "epoch 0" in cb.measurement_state_is_stale(SharedState.load_or_init(tmp_path))


def test_a_reset_session_still_resumes(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    _reset_state_file(tmp_path, _seed_and_reload(tmp_path))
    state = SharedState.load_or_init(tmp_path)
    assert (state.baseline_tput, state.agentx_backend) == (0.0, "inferencex")
    assert cb.measurement_state_is_stale(state) == ""


def test_a_reset_session_keeps_the_prompt_dataset_it_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_PATH", "/data/prompts.jsonl")
    _reset_state_file(tmp_path, _seed_and_reload(tmp_path))
    assert SharedState.load_or_init(tmp_path).prompt_dataset == "/data/prompts.jsonl"


def test_a_state_written_before_valid_measurement_judged_its_evidence_is_refused(tmp_path):
    # Epoch 2 recorded benchmark_backend and benchmark_valid as constraint fields; epoch 3 records valid_measurement.
    (tmp_path / "state.json").write_text('{"benchmark_mode": "agentx", "measurement_epoch": 2}', encoding="utf-8")
    assert "epoch 2" in cb.measurement_state_is_stale(SharedState.load_or_init(tmp_path))


def _rendered_count_after_resume(state: SharedState, tmp_path, monkeypatch, shell: str, pins=None) -> int:
    """The request count of a candidate rendered once *state* is resumed in a shell exporting ``NUM_PROMPTS=shell``."""
    clear_env(monkeypatch)
    monkeypatch.setenv("CONC", "4")
    monkeypatch.setenv("NUM_PROMPTS", shell)
    _export_operator_launch_shape(server_args="", extra_env=dict(pins or {}))
    assert cb.measurement_state_is_stale(state) == ""
    _seed_env_from_resumed_state(state, dict(pins or {}))
    return int(materialize(write_config(tmp_path / "cfg.yaml"), tmp_path / "out")["envs"]["NUM_PROMPTS"])


@pytest.mark.parametrize("shell", ["", "7"], ids=["no-export", "another-export"])
def test_a_resumed_session_sends_the_request_count_it_recorded_whatever_the_shell_exports(tmp_path, monkeypatch, shell):
    monkeypatch.setenv("NUM_PROMPTS", "39")
    state = _seed_and_reload(tmp_path)
    assert state.num_prompts == 39

    assert _rendered_count_after_resume(state, tmp_path, monkeypatch, shell) == 39


def test_a_session_that_recorded_no_request_count_resumes_on_the_default_with_one_exported(tmp_path, monkeypatch):
    state = _seed_and_reload(tmp_path)
    assert state.num_prompts == 0
    default = _rendered_count_after_resume(state, tmp_path, monkeypatch, "")

    assert _rendered_count_after_resume(state, tmp_path, monkeypatch, "39") == default != 39


@pytest.mark.parametrize(("seeded", "pinned"), [("39", "50"), ("", "50")])
def test_a_resume_that_pins_another_request_count_than_the_seed_recorded_is_refused(
    tmp_path, monkeypatch, seeded, pinned
):
    monkeypatch.setenv("NUM_PROMPTS", seeded)
    state = _seed_and_reload(tmp_path)

    _export_operator_launch_shape(server_args="", extra_env={"NUM_PROMPTS": pinned})

    assert "NUM_PROMPTS" in cb.measurement_state_is_stale(state)


def test_a_resume_that_pins_the_recorded_request_count_sends_it(tmp_path, monkeypatch):
    monkeypatch.setenv("NUM_PROMPTS", "39")
    state = _seed_and_reload(tmp_path)

    assert _rendered_count_after_resume(state, tmp_path, monkeypatch, "7", {"NUM_PROMPTS": "39"}) == 39


@pytest.mark.parametrize("pinned", ["039", "+39"])
def test_a_session_whose_pin_spells_the_count_another_way_resumes_on_it(tmp_path, monkeypatch, pinned):
    _export_operator_launch_shape(server_args="", extra_env={"NUM_PROMPTS": pinned})
    state = _seed_and_reload(tmp_path)
    assert state.num_prompts == 39

    assert _rendered_count_after_resume(state, tmp_path, monkeypatch, "7", {"NUM_PROMPTS": pinned}) == 39
    _export_operator_launch_shape(server_args="", extra_env={"NUM_PROMPTS": "040"})
    assert "NUM_PROMPTS=40" in cb.measurement_state_is_stale(state)


@pytest.mark.parametrize("pinned", ["all", "0", "-3"])
def test_a_resume_whose_pin_is_no_positive_integer_is_refused_naming_it(tmp_path, monkeypatch, pinned):
    monkeypatch.setenv("NUM_PROMPTS", "39")
    state = _seed_and_reload(tmp_path)

    _export_operator_launch_shape(server_args="", extra_env={"NUM_PROMPTS": pinned})

    assert f"NUM_PROMPTS={pinned!r} must be a positive integer" in cb.measurement_state_is_stale(state)


def test_a_session_pinned_before_the_count_was_recorded_resumes_on_its_pin(tmp_path, monkeypatch):
    state = _seed_and_reload(tmp_path)
    state.operator_extra_env = {"NUM_PROMPTS": "39"}

    assert _rendered_count_after_resume(state, tmp_path, monkeypatch, "", state.operator_extra_env) == 39
    # A resume that passes other pins drops the launch's, and the count stays the session's.
    assert _rendered_count_after_resume(state, tmp_path, monkeypatch, "7", {"FOO": "1"}) == 39


def test_a_reset_session_keeps_the_request_count_it_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("NUM_PROMPTS", "39")
    _reset_state_file(tmp_path, _seed_and_reload(tmp_path))
    assert SharedState.load_or_init(tmp_path).num_prompts == 39


def test_an_agentx_session_records_no_request_count_its_rounds_do_not_send(tmp_path, monkeypatch, native_benchmark):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.setenv("NUM_PROMPTS", "39")
    _export_operator_launch_shape(server_args="", extra_env={"NUM_PROMPTS": "39"})
    state = _seed_and_reload(tmp_path)
    assert (state.num_prompts, cb.measurement_state_is_stale(state)) == (0, "")


@pytest.mark.parametrize("value", ["all", "0", "-3"])
def test_a_session_is_not_seeded_with_a_request_count_that_is_no_positive_integer(tmp_path, monkeypatch, value, capsys):
    monkeypatch.setenv("NUM_PROMPTS", value)
    with pytest.raises(SystemExit) as refused:
        _seed_and_reload(tmp_path)
    assert refused.value.code == 2
    assert "NUM_PROMPTS" in capsys.readouterr().err


def test_a_state_written_before_the_request_count_was_recorded_reads_none(tmp_path):
    raw = SharedState(session_id="old").to_dict()
    del raw["num_prompts"]
    assert SharedState.from_dict(raw).num_prompts == 0
