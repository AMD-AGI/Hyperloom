# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native source handback must benchmark the complete accepted candidate stack."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.inference_optimizer.tests.test_native_candidate import _benchmark
from hyperloom.inference_optimizer.agentx.identity import canonical_sha256
from hyperloom.orchestrator.actions.executors import _ray_serving, integrate_patch
from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult, _build_variant_yaml


@pytest.mark.asyncio
async def test_native_integration_applies_controls_overlay_and_retains_measured_identity(tmp_path, monkeypatch):
    config = tmp_path / "baseline.yaml"
    config.write_text(yaml.safe_dump({"benchmark": _benchmark()}))
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("# accepted kernel overlay\n")
    monkeypatch.setattr(integrate_patch, "materialize_config_with_envs", lambda *a, **kw: config)
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **kw: None)
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.resolve_native_recipe",
        lambda benchmark, **kw: benchmark["workload_spec"]["resolved_topology"],
    )
    actual = {}

    async def measured_grid(**kwargs):
        variant = kwargs["grid"][0]
        candidate_path = _build_variant_yaml(
            kwargs["base_yaml_path"],
            kwargs["base_extra_args"],
            variant,
            output_subdir=tmp_path / "candidate",
            base_extra_envs=kwargs["base_extra_envs"],
            base_args_mode=kwargs["base_args_mode"],
            base_remove_args=kwargs["base_remove_args"],
            base_unset_envs=kwargs["base_unset_envs"],
        )
        actual.update(yaml.safe_load(candidate_path.read_text())["benchmark"])
        overrides = actual["agentx"]["launch_overrides"]
        base_argv = actual["workload_spec"]["server_launch"]["base_argv"]
        evidence = {
            "version": 1,
            "framework": "sglang",
            "base_argv": base_argv,
            "effective_argv": base_argv[:-1] + overrides["append_args"],
            "effective_env": {**overrides["env"], **{key: None for key in overrides["unset_env"]}},
            "base_env": {key: None for key in set(overrides["env"]) | set(overrides["unset_env"])},
            "resolved_executable": base_argv[0],
            "runtime_environment": dict(overrides["env"]),
            "source_files": overrides["source_files"],
            "absent_source_files": overrides["absent_source_files"],
            "overrides_sha256": canonical_sha256(overrides),
        }
        evidence["evidence_sha256"] = canonical_sha256(evidence)
        return [
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=variant.extra_envs,
                status="succeeded",
                output_throughput=280.0,
                materialized_config=str(candidate_path),
                native_measurement={
                    "agentx_launch_contract": 1,
                    "agentx_workload_fingerprint": "a" * 64,
                    "agentx_server_launch": evidence,
                },
            )
        ]

    monkeypatch.setattr(integrate_patch, "run_grid", measured_grid)
    executor = integrate_patch.IntegratePatchExecutor(session_dir=tmp_path)
    bench, _ = await executor._bench_patch(
        params={
            "config_path": str(config),
            "base_extra_args": "--mem-fraction-static 0.7",
            "base_extra_envs": {"SGLANG_TEST_OLD": "1"},
            "remove_args": ["--disable-cuda-graph"],
            "unset_envs": ["SGLANG_TEST_OLD"],
            "args_mode": "replace",
            "overlay_pythonpath": str(overlay),
        },
        output_root=tmp_path / "out",
        extra_server_args_applied="--mem-fraction-static 0.8",
        extra_envs_applied={"SGLANG_TEST_NEW": "1"},
        specialist_task_id="source-handback",
    )
    override = actual["agentx"]["launch_overrides"]
    assert override["append_args"] == ["--mem-fraction-static", "0.8"]
    assert override["remove_args"] == ["--disable-cuda-graph"]
    assert override["replace_args"] is True
    assert override["unset_env"] == ["SGLANG_TEST_OLD"]
    assert override["env"]["PYTHONPATH"].split(":")[0] == str(overlay)
    assert override["env"]["SGLANG_TEST_NEW"] == "1"
    assert str(overlay / "sitecustomize.py") in override["source_files"]
    assert bench["materialized_config"] == str(tmp_path / "candidate/config.yaml")
    assert bench["agentx_workload_fingerprint"] == "a" * 64
    assert bench["effective_config"]["args_mode"] == "replace"
    assert bench["effective_config"]["final_overlay"] == str(overlay)
