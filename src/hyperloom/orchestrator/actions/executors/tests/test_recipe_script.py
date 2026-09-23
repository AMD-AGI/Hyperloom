# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the recipe's server script accepts, read off the script itself."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._recipe_script import (
    RecipeLeverUnavailableError,
    recipe_launch_contract,
    resolve_launch_server_script,
)

# The shape every agentic recipe in InferenceX has: a literal vllm argv with no
# extra-args sink, and unconditional exports that outrank any --extra-env.
_AGENTIC_RECIPE = """#!/bin/bash
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
export VLLM_ROCM_USE_AITER=1
export PORT="${PORT:-8000}"
VLLM_CMD=( vllm serve "$MODEL" --tensor-parallel-size 4 )
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
"""


def _checkout(tmp_path: Path, name: str, body: str) -> Path:
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / name).write_text(body, encoding="utf-8")
    # Same basename one level down: the client only ever runs the one beside
    # itself, so this must not be what resolution returns.
    (benchmarks / "single_node" / "agentic" / name).write_text("# a different recipe\n", encoding="utf-8")
    return benchmarks.parent


def _bench(root: Path, server_script: str) -> dict:
    return {
        "benchmark_script": "aiperf_client.sh",
        "framework": "vllm",
        "inferencex_path": str(root),
        "envs": {"AGENTX_SERVER_SCRIPT": server_script},
    }


def test_the_agentx_client_resolves_to_the_server_script_it_delegates_to(tmp_path):
    root = _checkout(tmp_path, "dsv41flash.sh", _AGENTIC_RECIPE)

    resolved = resolve_launch_server_script(_bench(root, "dsv41flash.sh"))

    # The one beside the client, not the same-named recipe a directory down.
    assert Path(resolved) == root / "benchmarks" / "dsv41flash.sh"


def test_a_recipe_without_an_extra_args_sink_reports_the_lever_as_unavailable(tmp_path):
    root = _checkout(tmp_path, "dsv41flash.sh", _AGENTIC_RECIPE)

    reads_extra_args, _ = recipe_launch_contract(_bench(root, "dsv41flash.sh"))

    assert reads_extra_args is False


def test_a_recipe_that_reads_the_extra_args_variable_keeps_the_lever(tmp_path):
    root = _checkout(tmp_path, "sinked.sh", _AGENTIC_RECIPE + '\nvllm serve "$M" $EXTRA_VLLM_ARGS\n')

    reads_extra_args, _ = recipe_launch_contract(_bench(root, "sinked.sh"))

    assert reads_extra_args is True


def test_a_recipe_that_forwards_its_positional_arguments_keeps_the_lever(tmp_path):
    root = _checkout(tmp_path, "sinked.sh", _AGENTIC_RECIPE + '\nvllm serve "$M" "$@"\n')

    reads_extra_args, _ = recipe_launch_contract(_bench(root, "sinked.sh"))

    assert reads_extra_args is True


def test_only_unguarded_exports_count_as_overwritten(tmp_path):
    root = _checkout(tmp_path, "dsv41flash.sh", _AGENTIC_RECIPE)

    _, overwritten = recipe_launch_contract(_bench(root, "dsv41flash.sh"))

    assert "VLLM_ROCM_USE_AITER" in overwritten
    assert "HIP_VISIBLE_DEVICES" in overwritten
    # ``export PORT="${PORT:-8000}"`` defers to a caller-supplied value.
    assert "PORT" not in overwritten


def test_an_unresolvable_recipe_constrains_nothing(tmp_path):
    root = _checkout(tmp_path, "dsv41flash.sh", _AGENTIC_RECIPE)

    reads_extra_args, overwritten = recipe_launch_contract(_bench(root, "absent.sh"))

    assert reads_extra_args is True
    assert overwritten == frozenset()


def _variant_yaml(tmp_path: Path, root: Path, variant) -> dict:
    """Materialize one grid variant against the sink-less recipe."""
    import yaml

    from hyperloom.orchestrator.actions.executors import _grid_runner

    bench = {
        "benchmark_script": "dsv41flash.sh",
        "framework": "vllm",
        "model": "/m",
        "inferencex_path": str(root),
        "envs": {},
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    out = _grid_runner._build_variant_yaml(
        base, "", variant, output_subdir=tmp_path / "out", model_path="/m", benchmark_script="dsv41flash.sh"
    )
    return yaml.safe_load(Path(out).read_text(encoding="utf-8"))["benchmark"]["envs"]


def test_a_variant_arg_the_recipe_cannot_carry_is_refused(tmp_path):
    """The grid path composes its own args, so the base-config guard never sees them."""
    root = _checkout(tmp_path, "dsv41flash.sh", _AGENTIC_RECIPE)
    variant = GridVariant(name="v1", extra_server_args="--enable-torch-compile")

    with pytest.raises(RecipeLeverUnavailableError):
        _variant_yaml(tmp_path, root, variant)


def test_a_variant_env_the_recipe_overwrites_is_dropped(tmp_path):
    root = _checkout(tmp_path, "dsv41flash.sh", _AGENTIC_RECIPE)
    variant = GridVariant(name="v1", extra_envs={"VLLM_ROCM_USE_AITER": "0", "VLLM_SOMETHING_ELSE": "1"})

    envs = _variant_yaml(tmp_path, root, variant)

    assert "VLLM_ROCM_USE_AITER" not in envs
    assert envs["VLLM_SOMETHING_ELSE"] == "1"


def test_the_refusal_is_graded_as_an_integration_fault():
    """A patch that was never benchmarked must not burn the gate's verdict quota."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    result = {"status": "reverted", "error_class": "recipe_lever_unavailable"}

    assert SharedState._is_integrate_fault(result) is True
