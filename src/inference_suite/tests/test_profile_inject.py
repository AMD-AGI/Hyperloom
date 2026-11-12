# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the profiler layer: each framework's injection, its place last in the launch, and the trace dirs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from inference_suite.compose import compose
from inference_suite.contracts import FixedWorkload, ProfileConfig, RunLayout, RunPaths, WorkloadKind
from inference_suite.contracts.base_launch import Workload
from inference_suite.contracts.launch import Launch, LaunchError, LayerSource
from inference_suite.frameworks import Framework, framework_facts
from inference_suite.profile.inject import (
    SGLANG_PROFILER_DIR_ENV,
    make_trace_dirs,
    profiler_layer,
    settle_traces,
    start_body,
)
from inference_suite.tests.test_base_launch_contract import _base
from inference_suite.tests.test_compose import _FACTS
from inference_suite.tests.test_config_contract import _FIXED, _SHA, _config

_WINDOW = ProfileConfig(enabled=True, start_body={"record_shapes": True}, num_prompts=64, delay_iters=40, max_iters=8)


def _fixed_config(tmp_path: Path, profile: ProfileConfig = _WINDOW, workload: FixedWorkload = _FIXED):
    paths = RunPaths(run_dir=str(tmp_path / "run"), tool_cache="/s/tools", tool_manifest_sha256=_SHA)
    return _config(workload=WorkloadKind.FIXED, agentx=None, fixed=workload, profile=profile, paths=paths)


def _launch(tmp_path: Path, framework: Framework, base_args: tuple[str, ...] = ()) -> Launch:
    base = _base(
        query=replace(_base().query, framework=framework.value, workload=Workload.FIXED, isl=1024, osl=1024),
        framework=framework.value,
        entrypoint=framework_facts(framework).server_command.entrypoint,
        args=base_args,
        golden_al=None,
    )
    config = _fixed_config(tmp_path)
    config = replace(config, server=replace(config.server, base=replace(config.server.base, digest=base.digest())))
    launch = compose(config, base, replace(_FACTS, profiler=profiler_layer(config, framework)))
    assert isinstance(launch, Launch), launch
    return launch


def _trace_dir(tmp_path: Path) -> str:
    return str(tmp_path / "run" / "profile" / "torch_trace")


def test_sglang_gets_the_trace_dir_in_its_env(tmp_path):
    launch = _launch(tmp_path, Framework.SGLANG)

    assert launch.env[SGLANG_PROFILER_DIR_ENV] == _trace_dir(tmp_path)
    assert {entry.name: entry.source for entry in launch.env_sources}[SGLANG_PROFILER_DIR_ENV] is LayerSource.PROFILE
    assert not any(arg.source is LayerSource.PROFILE for arg in launch.args)


def test_vllm_gets_the_profiler_config_and_window_last_beating_an_earlier_dir(tmp_path):
    launch = _launch(tmp_path, Framework.VLLM, ("--profiler-config.torch_profiler_dir", "/placeholder"))

    assert launch.argv[-8:] == (
        "--profiler-config.profiler",
        "torch",
        "--profiler-config.torch_profiler_dir",
        _trace_dir(tmp_path),
        "--profiler-config.delay_iterations",
        "40",
        "--profiler-config.max_iterations",
        "8",
    )
    assert "/placeholder" not in launch.argv
    dir_arg = next(arg for arg in launch.args if arg.flag == "--profiler-config.torch_profiler_dir")
    assert (dir_arg.source, dir_arg.overrode) == (LayerSource.PROFILE, (LayerSource.BASE,))
    assert Launch.from_document(launch.to_document()) == launch


def test_vllm_agentx_profile_carries_no_step_window(tmp_path):
    config = replace(_config(), paths=RunPaths(str(tmp_path / "run"), "/s/tools", _SHA))

    layer = profiler_layer(config, Framework.VLLM)

    assert layer is not None
    assert not any("iterations" in token for token in layer.args_add)


def test_atom_gets_the_trace_dir_as_a_flag(tmp_path):
    launch = _launch(tmp_path, Framework.ATOM)

    assert launch.argv[-2:] == ("--torch-profiler-dir", _trace_dir(tmp_path))
    assert launch.args[-1].source is LayerSource.PROFILE


def test_no_layer_without_a_profile_and_none_for_a_serverless_framework(tmp_path):
    assert profiler_layer(_fixed_config(tmp_path, ProfileConfig(enabled=False)), Framework.SGLANG) is None
    with pytest.raises(LaunchError, match="xdit runs no server"):
        profiler_layer(_fixed_config(tmp_path), Framework.XDIT)


def test_sglang_start_body_carries_the_step_window_and_the_others_take_it_as_given():
    assert start_body(_WINDOW, Framework.SGLANG) == {"record_shapes": True, "start_step": 40, "num_steps": 8}
    assert start_body(_WINDOW, Framework.VLLM) == {"record_shapes": True}
    assert start_body(ProfileConfig(True, agentx_open_after_sec=0, agentx_duration_sec=30), Framework.SGLANG) == {}


def test_capture_sidecars_land_beside_torch_trace_not_flattened(tmp_path):
    layout = RunLayout(tmp_path)
    make_trace_dirs(layout)
    torch_trace = layout.profile_dir / "torch_trace"
    (torch_trace / "step-rank0.pt.trace.json.gz").write_text("{}")
    for name in ("capture_traces", "graph_capture_profile"):
        (torch_trace / name / "bs_8").mkdir(parents=True)
        (torch_trace / name / "bs_8" / "trace.json").write_text("{}")

    assert settle_traces(layout) == ("profile/torch_trace", "profile/capture_traces", "profile/graph_capture_profile")
    assert sorted(path.name for path in layout.profile_dir.iterdir()) == [
        "capture_traces",
        "graph_capture_profile",
        "torch_trace",
    ]
    assert [path.name for path in torch_trace.iterdir()] == ["step-rank0.pt.trace.json.gz"]
    assert (layout.profile_dir / "capture_traces" / "bs_8" / "trace.json").is_file()


def test_an_empty_capture_names_no_trace_dir(tmp_path):
    layout = RunLayout(tmp_path)
    make_trace_dirs(layout)

    assert settle_traces(layout) == ()
    assert all(directory.is_dir() for directory in layout.profile_trace_dirs())
