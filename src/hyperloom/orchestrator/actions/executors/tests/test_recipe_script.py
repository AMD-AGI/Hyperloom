# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for recipe-script ownership detection and overwritten-env reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._recipe_script import (
    RecipeLeverUnavailableError,
    launcher_overwritten_envs,
    recipe_owns_argv,
    resolve_launch_server_script,
)

# An agentic recipe with hardcoded argv and unconditional exports (no env-args sink).
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
    agentic_script = benchmarks / "single_node" / "agentic" / name
    agentic_script.write_text(body, encoding="utf-8")
    return benchmarks.parent


def _bench_agentx(root: Path, server_script: str) -> dict:
    return {
        "benchmark_script": "aiperf_client.sh",
        "framework": "vllm",
        "inferencex_path": str(root),
        "envs": {"AGENTX_SERVER_SCRIPT": f"single_node/agentic/{server_script}"},
    }


def _bench_generic(root: Path, script: str) -> dict:
    """Bench whose benchmark_script is a direct launcher (not aiperf_client.sh)."""
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / script).write_text("#!/bin/bash\nexport SGLANG_USE_AITER=1\n$EXTRA_SGLANG_ARGS\n", encoding="utf-8")
    return {
        "benchmark_script": script,
        "framework": "sglang",
        "inferencex_path": str(root),
        "envs": {},
    }


def test_agentx_client_resolves_to_the_agentic_server_script(tmp_path):
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE)
    resolved = resolve_launch_server_script(_bench_agentx(root, "minimaxm3.sh"))
    assert Path(resolved) == root / "benchmarks" / "single_node" / "agentic" / "minimaxm3.sh"


def test_agentic_recipe_owns_argv(tmp_path):
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE)
    assert recipe_owns_argv(_bench_agentx(root, "minimaxm3.sh")) is True


def test_generic_launcher_does_not_own_argv(tmp_path):
    root = tmp_path / "ix"
    bench = _bench_generic(root, "sglang_mi355x.sh")
    assert recipe_owns_argv(bench) is False


def test_unresolvable_script_does_not_own_argv(tmp_path):
    bench = _bench_agentx(tmp_path / "nowhere", "absent.sh")
    assert recipe_owns_argv(bench) is False


def test_launcher_overwritten_envs_returns_unguarded_exports(tmp_path):
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE)
    # Agentic recipe owns argv: Hyperloom writes env levers via the script copy,
    # so the drop filter must return empty to avoid self-defeating its own writes.
    overwritten = launcher_overwritten_envs(_bench_agentx(root, "minimaxm3.sh"))
    assert overwritten == frozenset()


def test_generic_launcher_overwritten_envs_reports_unguarded_exports(tmp_path):
    root = tmp_path / "ix"
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    # A launcher that unconditionally exports two names and guards one.
    script = benchmarks / "sglang_mi355x.sh"
    script.write_text(
        "#!/bin/bash\nexport SGLANG_USE_AITER=1\nexport SGLANG_AITER_MLA_PERSIST=1\n"
        'export PORT="${PORT:-8000}"\n$EXTRA_SGLANG_ARGS\n',
        encoding="utf-8",
    )
    bench = {
        "benchmark_script": "sglang_mi355x.sh",
        "framework": "sglang",
        "inferencex_path": str(root),
        "envs": {},
    }
    overwritten = launcher_overwritten_envs(bench)
    assert "SGLANG_USE_AITER" in overwritten
    assert "SGLANG_AITER_MLA_PERSIST" in overwritten
    assert "PORT" not in overwritten  # guarded


def test_unresolvable_recipe_reports_empty_overwritten_envs(tmp_path):
    bench = _bench_agentx(tmp_path / "nowhere", "absent.sh")
    assert launcher_overwritten_envs(bench) == frozenset()


def test_integrate_fault_grading_applies_to_recipe_lever_unavailable():
    """A lever the recipe cannot carry must not burn the gate's verdict quota."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    result = {"status": "reverted", "error_class": "recipe_lever_unavailable"}
    assert SharedState._is_integrate_fault(result) is True


# ── Materialization tests via the workload envs path ─────────────────────────

def _make_sparse_model_config(model_dir: Path) -> None:
    import json

    cfg = {
        "model_type": "minimax_m3",
        "text_config": {
            "sparse_attention_config": {
                "sparse_block_size": 128,
            }
        },
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _build_base_yaml(tmp_path: Path, recipe_root: Path, framework: str, script: str) -> Path:
    import yaml

    bench: dict = {
        "framework": framework,
        "model": str(tmp_path / "model"),
        "inferencex_path": str(recipe_root),
        "envs": {
            "AGENTX_SERVER_SCRIPT": script,
            "HYPERLOOM_AGENTX": "1",
        },
        "benchmark_script": "aiperf_client.sh",
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return base


def test_no_sparse_block_size_injected_for_agentic_vllm(tmp_path):
    """On an agentic recipe the sparse --block-size default must not be injected."""
    import os

    import yaml

    from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

    model_dir = tmp_path / "model"
    _make_sparse_model_config(model_dir)
    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(
        _AGENTIC_RECIPE, encoding="utf-8"
    )

    base = _build_base_yaml(tmp_path, recipe_root, "vllm", "single_node/agentic/minimaxm3.sh")
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["HYPERLOOM_AGENTX"] = "1"
    env["AGENTX_SERVER_SCRIPT"] = "single_node/agentic/minimaxm3.sh"
    env["INFERENCEX_PATH"] = str(recipe_root)
    old_env = dict(os.environ)
    os.environ.update(env)
    try:
        materialized = materialize_config_with_envs(
            base,
            out_dir,
            model_path=str(model_dir),
            gpu_type="mi355x",
            agentx_mode=True,
        )
        envs = yaml.safe_load(Path(materialized).read_text())["benchmark"]["envs"]
        extra_args = envs.get("EXTRA_VLLM_ARGS", "")
        assert "--block-size" not in extra_args, f"block-size was injected on agentic recipe: {extra_args!r}"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_block_size_still_injected_for_generic_vllm(tmp_path):
    """On a generic launcher the sparse --block-size default IS injected."""
    import os

    import yaml

    from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

    model_dir = tmp_path / "model"
    _make_sparse_model_config(model_dir)
    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    launcher = benchmarks / "vllm_mi355x.sh"
    launcher.write_text(
        "#!/bin/bash\nvllm serve $MODEL $EXTRA_VLLM_ARGS\n", encoding="utf-8"
    )
    bench = {
        "framework": "vllm",
        "model": str(model_dir),
        "inferencex_path": str(recipe_root),
        "envs": {},
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    old_env = dict(os.environ)
    os.environ["INFERENCEX_PATH"] = str(recipe_root)
    try:
        materialized = materialize_config_with_envs(
            base,
            tmp_path / "out",
            model_path=str(model_dir),
            gpu_type="mi355x",
            agentx_mode=False,
        )
        envs = yaml.safe_load(Path(materialized).read_text())["benchmark"]["envs"]
        assert "--block-size 128" in envs.get("EXTRA_VLLM_ARGS", "")
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_profile_refused_on_agentic_recipe(tmp_path):
    """A profile run on an agentic recipe is structurally impossible."""
    import os

    import yaml

    from hyperloom.orchestrator.actions.executors._recipe_script import RecipeLeverUnavailableError
    from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(
        _AGENTIC_RECIPE, encoding="utf-8"
    )
    bench = {
        "framework": "vllm",
        "model": str(tmp_path / "model"),
        "inferencex_path": str(recipe_root),
        "envs": {
            "AGENTX_SERVER_SCRIPT": "single_node/agentic/minimaxm3.sh",
            "HYPERLOOM_AGENTX": "1",
            "PROFILE": "1",
        },
        "benchmark_script": "aiperf_client.sh",
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    old_env = dict(os.environ)
    os.environ["HYPERLOOM_AGENTX"] = "1"
    os.environ["INFERENCEX_PATH"] = str(recipe_root)
    try:
        with pytest.raises(RecipeLeverUnavailableError, match="profiling is not available"):
            materialize_config_with_envs(
                base,
                tmp_path / "out",
                model_path=str(tmp_path / "model"),
                gpu_type="mi355x",
                agentx_mode=True,
            )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_grid_build_variant_yaml_raises_recipe_lever_unavailable_on_profile(tmp_path):
    """_build_variant_yaml raises RecipeLeverUnavailableError (not a generic error)
    when materializing a profile round against an agentic recipe."""
    import os

    import yaml

    from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml
    from hyperloom.orchestrator.actions.executors._recipe_script import RecipeLeverUnavailableError

    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(
        _AGENTIC_RECIPE, encoding="utf-8"
    )
    bench = {
        "framework": "vllm",
        "model": str(tmp_path / "model"),
        "inferencex_path": str(recipe_root),
        "envs": {
            "AGENTX_SERVER_SCRIPT": "single_node/agentic/minimaxm3.sh",
            "HYPERLOOM_AGENTX": "1",
            "PROFILE": "1",
        },
        "benchmark_script": "aiperf_client.sh",
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    variant = GridVariant(name="v1", extra_server_args="", extra_envs={})
    old_env = dict(os.environ)
    os.environ["HYPERLOOM_AGENTX"] = "1"
    os.environ["INFERENCEX_PATH"] = str(recipe_root)
    try:
        with pytest.raises(RecipeLeverUnavailableError, match="profiling is not available"):
            _build_variant_yaml(
                base,
                "",
                variant,
                output_subdir=tmp_path / "slot",
                model_path=str(tmp_path / "model"),
                gpu_type="mi355x",
            )
    finally:
        os.environ.clear()
        os.environ.update(old_env)
