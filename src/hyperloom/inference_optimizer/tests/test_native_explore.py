# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native explore must retain launch identity while stacking canonical wins."""

from __future__ import annotations

import shlex

import pytest
import yaml

from hyperloom.inference_optimizer.tests.test_native_candidate import _benchmark
from hyperloom.orchestrator.actions.executors import explore
from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult, _build_variant_yaml
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task


@pytest.mark.asyncio
async def test_public_native_explore_keeps_comparable_candidates_without_reapplying_snapshot(tmp_path, monkeypatch):
    config = tmp_path / "native.yaml"
    config.write_text(yaml.safe_dump({"benchmark": _benchmark()}))
    state = SharedState(framework="sglang", benchmark_mode="agentx", agentx_epoch=3, agentx_backend="native")
    state.baseline_tput = 200.0
    state.baseline_perf = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 300.0,
        "e2e_norm_intvty_p50": 300.0,
        "duration_seconds": 3600.0,
        "request_error_rate": 0.0,
        "agentx_launch_contract": 1,
        "agentx_workload_fingerprint": "a" * 64,
    }
    monkeypatch.setattr(explore, "materialize_config_with_envs", lambda *a, **kw: config)
    monkeypatch.setattr(explore, "resolve_lifecycle_params", lambda *a: {"eligible": False})
    monkeypatch.setattr(explore, "maybe_serving_lease", lambda **kw: None)
    monkeypatch.setattr(explore, "apply_compatibility_filter", lambda rows, **kw: (rows, []))
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.resolve_native_recipe",
        lambda benchmark, **kw: benchmark["workload_spec"]["resolved_topology"],
    )
    launched = []

    async def canonical_grid(**kwargs):
        variant = kwargs["grid"][0]
        path = _build_variant_yaml(
            kwargs["base_yaml_path"],
            kwargs["base_extra_args"],
            variant,
            output_subdir=kwargs["output_root"],
            base_extra_envs=kwargs["base_extra_envs"],
            base_remove_args=kwargs["base_remove_args"],
            base_unset_envs=kwargs["base_unset_envs"],
            base_args_mode=kwargs["base_args_mode"],
            base_native_launch_overrides=kwargs["base_native_launch_overrides"],
        )
        snapshot = yaml.safe_load(path.read_text())["benchmark"]["agentx"]["launch_overrides"]
        launched.append(snapshot)
        return [
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=variant.extra_envs,
                status="succeeded",
                output_throughput=200.0 + len(launched) * 10,
                total_token_throughput=22000.0,
                intvty_p90=300.0 + len(launched) * 30,
                intvty_p50=300.0 + len(launched) * 30,
                duration_seconds=3600.0,
                request_error_rate=0.0,
                materialized_config=str(path),
                native_measurement={
                    "agentx_launch_contract": 1,
                    "agentx_workload_fingerprint": "a" * 64,
                    "agentx_candidate_fingerprint": str(len(launched)) * 64,
                },
            )
        ]

    monkeypatch.setattr(explore, "run_grid", canonical_grid)
    task = Task(
        task_id="native-explore",
        kind="explore",
        state="running",
        idempotency_key="native-explore",
        params={
            "config_path": str(config),
            "base_tput": 200.0,
            "grid": [
                {"name": "json", "extra_args": '--compilation-config \'{"mode": "max-autotune"}\''},
                {"name": "next", "extra_args": "--mem-fraction-static 0.75"},
            ],
        },
    )
    result = await explore.ExploreExecutor(session_dir=tmp_path)(
        RunnerContext(task=task, lease=None, extra={"shared_state": state})
    )
    assert [winner["name"] for winner in result["winners"]] == ["json", "next"]
    assert launched[1]["append_args"] == [
        "--compilation-config",
        '{"mode": "max-autotune"}',
        "--mem-fraction-static",
        "0.75",
    ]
    assert launched[1]["replace_args"] is False
    winner = result["best_variant"]
    assert winner["agentx_workload_fingerprint"] == "a" * 64
    assert winner["e2e_norm_intvty_p50"] == 360.0
    assert winner["duration_seconds"] == 3600.0
    assert winner["materialized_config"].endswith("config.yaml")
    assert shlex.split(winner["extra_server_args"]) == launched[1]["append_args"]
    assert winner["native_launch_overrides"] == launched[1]
