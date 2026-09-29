# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Candidate updates must survive the same final launch authority as a baseline."""

from __future__ import annotations

import sys

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import native
from hyperloom.inference_optimizer.agentx.identity import canonical_sha256, native_workload_fingerprint
from hyperloom.inference_optimizer.agentx.native import resolve_native_recipe
from hyperloom.inference_optimizer.agentx.runtime import maybe_prepare_agentx
from hyperloom.inference_optimizer.tests.test_agentx_runtime import _pinned_native_cfg
from hyperloom.orchestrator.actions.executors._native_candidate import update_native_candidate_file


@pytest.mark.parametrize("candidate_kind", ["runtime", "source"])
def test_post_materialization_candidate_refreshes_execution_and_passes_final_boundary(
    tmp_path, monkeypatch, candidate_kind
):
    config_path, pins = _pinned_native_cfg(tmp_path, monkeypatch)
    data = yaml.safe_load(config_path.read_text())
    bench = data["benchmark"]
    bench["agentx"] = {"enabled": True, "launch_overrides": {"version": 1}}
    bench["inferencex_path"] = str(tmp_path)
    bench["envs"]["ROCR_VISIBLE_DEVICES"] = "0,1,2,3"
    bench["gpu_selection"] = {"auto": False}

    # The external recipe/source resolver is represented by a deterministic
    # identity over its returned config; both real Hyperloom boundary checks run.
    def execution_identity(**kwargs):
        current = kwargs["resolved_benchmark"]
        return {
            "inferencex_commit": "d" * 40,
            "magpie_commit": "c" * 40,
            "static_execution_fingerprint": "b" * 64,
            "execution_fingerprint": canonical_sha256(current["agentx"]),
            "workload_fingerprint": native_workload_fingerprint(current, "b" * 64),
        }

    monkeypatch.setattr("hyperloom.inference_optimizer.agentx.native.native_execution_identity", execution_identity)
    original_preview = native.preview_native_recipe

    def with_recipe_fingerprint(current, **kwargs):
        result = original_preview(current, **kwargs)
        result["entry"]["recipe-fingerprint"] = "a" * 64
        return result

    monkeypatch.setattr(native, "preview_native_recipe", with_recipe_fingerprint)
    pins.pop("HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT")
    for name, value in pins.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HYPERLOOM_IMAGE", "example/sglang:agentx")
    monkeypatch.setenv("HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT", "")
    resolve_native_recipe(bench, inferencex_path=tmp_path, expected_gpu_count=4)
    initial_execution = dict(bench["workload_spec"]["execution"])
    pins["HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT"] = initial_execution["workload_fingerprint"]
    monkeypatch.setenv(
        "HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT", pins["HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT"]
    )
    config_path.write_text(yaml.safe_dump(data))

    candidate = {"runtime_override": {"runtime_python_exe": sys.executable}}
    if candidate_kind == "source":
        source_root = tmp_path / "source"
        package = source_root / "sglang"
        package.mkdir(parents=True)
        init = package / "__init__.py"
        init.write_text("")
        module = package / "kernel.py"
        module.write_text("OPTIMIZED = True\n")
        from hyperloom.orchestrator.actions.executors._native_source import source_file_hashes

        candidate["runtime_override"]["pythonpath_prefix"] = str(source_root)
        candidate["source_files"] = source_file_hashes([module])
    update_native_candidate_file(config_path, **candidate)
    actual = yaml.safe_load(config_path.read_text())["benchmark"]
    assert actual["workload_spec"]["execution"]["execution_fingerprint"] != initial_execution["execution_fingerprint"]
    assert actual["workload_spec"]["execution"]["workload_fingerprint"] == initial_execution["workload_fingerprint"]
    assert maybe_prepare_agentx(env=dict(pins), inferencex_path=str(tmp_path), config_path=config_path) is True

    # Updating the payload without refreshing its accepted receipt still fails.
    actual["agentx"]["launch_overrides"]["env"]["SGLANG_USE_AITER"] = "1"
    config_path.write_text(yaml.safe_dump({"benchmark": actual}))
    with pytest.raises(ValueError, match="identity changed after materialization"):
        maybe_prepare_agentx(env=dict(pins), inferencex_path=str(tmp_path), config_path=config_path)
