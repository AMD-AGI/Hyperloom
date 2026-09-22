# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the recipe's server script accepts, read off the script itself."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors._recipe_script import (
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
