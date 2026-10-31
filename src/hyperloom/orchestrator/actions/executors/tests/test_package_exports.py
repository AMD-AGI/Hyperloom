# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``executors`` package exports: callers outside the package reach the executors through these names."""

from __future__ import annotations

from hyperloom.orchestrator.actions import executors
from hyperloom.orchestrator.actions.executors import _grid_runner, _profile_setup, _server_lifecycle, _server_patcher


def test_every_exported_name_resolves():
    missing = [name for name in executors.__all__ if not hasattr(executors, name)]
    assert missing == []
    assert len(executors.__all__) == len(set(executors.__all__))


def test_the_exports_are_the_owning_modules_objects_not_copies():
    assert executors.run_grid is _grid_runner.run_grid
    assert executors.num_gpus_for_config is _grid_runner._num_gpus_for_config
    assert executors.assign_free_port is _server_lifecycle._assign_free_port
    assert executors.ensure_vllm_patched_for_tracelens is _server_patcher.ensure_vllm_patched_for_tracelens
    assert executors.VLLM_PATCH_PROFILER_FIELDS is _profile_setup.VLLM_PATCH_PROFILER_FIELDS
