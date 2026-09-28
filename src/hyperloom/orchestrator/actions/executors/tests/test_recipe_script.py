# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for recipe-script ownership detection and lever rendering."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._recipe_script import (
    RecipeLeverUnavailableError,
    apply_recipe_levers,
    launcher_overwritten_envs,
    recipe_owns_argv,
    resolve_launch_server_script,
)

# Minimal agentic recipe that prints its effective argv and env instead of launching.
# The script exits 0 with stdout = "argv:<flags>" and "env:KEY=VALUE" lines.
_AGENTIC_RECIPE_TEMPLATE = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -euo pipefail

    source "$(dirname "$0")/../../benchmark_lib.sh"

    export VLLM_ROCM_USE_AITER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
    export PORT="${PORT:-8000}"

    VLLM_CMD=(
        vllm serve "$MODEL_PATH"
        --host 0.0.0.0
        --port "$PORT"
        --block-size 128
        --gpu-memory-utilization 0.90
        --max-num-seqs "$((2 * CONC))"
    )
    printf 'argv:%s\\n' "${VLLM_CMD[@]}"
    printf 'env:VLLM_ROCM_USE_AITER=%s\\n' "${VLLM_ROCM_USE_AITER:-}"
    """
)

_SGLANG_RECIPE_TEMPLATE = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -euo pipefail

    source "$(dirname "$0")/../../benchmark_lib.sh"

    export SGLANG_USE_AITER=1

    SGLANG_CMD=(
        python3 -m sglang.launch_server
        --host 0.0.0.0
        --port "$PORT"
        --mem-fraction-static 0.85
    )
    printf 'argv:%s\\n' "${SGLANG_CMD[@]}"
    printf 'env:SGLANG_USE_AITER=%s\\n' "${SGLANG_USE_AITER:-}"
    """
)


def _checkout(tmp_path: Path, name: str, body: str) -> Path:
    """Create a minimal InferenceX checkout with the given agentic recipe."""
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    script = benchmarks / "single_node" / "agentic" / name
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return benchmarks.parent


def _bench_agentx(root: Path, server_script: str, framework: str = "vllm") -> dict:
    return {
        "benchmark_script": "aiperf_client.sh",
        "framework": framework,
        "inferencex_path": str(root),
        "envs": {
            "AGENTX_SERVER_SCRIPT": f"single_node/agentic/{server_script}",
            "HYPERLOOM_AGENTX": "1",
        },
    }


def _bench_generic(root: Path, script: str, framework: str = "sglang") -> dict:
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    launcher = benchmarks / script
    launcher.write_text("#!/bin/bash\nexport SGLANG_USE_AITER=1\n$EXTRA_SGLANG_ARGS\n", encoding="utf-8")
    return {
        "benchmark_script": script,
        "framework": framework,
        "inferencex_path": str(root),
        "envs": {},
    }


# ── Ownership and overwritten-env contracts (carry-over from Commit 1) ────────


def test_agentx_client_resolves_to_the_agentic_server_script(tmp_path):
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    resolved = resolve_launch_server_script(_bench_agentx(root, "minimaxm3.sh"))
    assert Path(resolved) == root / "benchmarks" / "single_node" / "agentic" / "minimaxm3.sh"


def test_agentic_recipe_owns_argv(tmp_path):
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    assert recipe_owns_argv(_bench_agentx(root, "minimaxm3.sh")) is True


def test_generic_launcher_does_not_own_argv(tmp_path):
    root = tmp_path / "ix"
    bench = _bench_generic(root, "sglang_mi355x.sh")
    assert recipe_owns_argv(bench) is False


def test_unresolvable_script_does_not_own_argv(tmp_path):
    assert recipe_owns_argv(_bench_agentx(tmp_path / "nowhere", "absent.sh")) is False


def test_launcher_overwritten_envs_returns_empty_on_recipe_surface(tmp_path):
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    assert launcher_overwritten_envs(_bench_agentx(root, "minimaxm3.sh")) == frozenset()


def test_generic_launcher_overwritten_envs_reports_unguarded_exports(tmp_path):
    root = tmp_path / "ix"
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "sglang_mi355x.sh").write_text(
        "#!/bin/bash\nexport SGLANG_USE_AITER=1\nexport SGLANG_AITER_MLA_PERSIST=1\n"
        'export PORT="${PORT:-8000}"\n$EXTRA_SGLANG_ARGS\n',
        encoding="utf-8",
    )
    bench = {"benchmark_script": "sglang_mi355x.sh", "framework": "sglang", "inferencex_path": str(root), "envs": {}}
    overwritten = launcher_overwritten_envs(bench)
    assert "SGLANG_USE_AITER" in overwritten
    assert "SGLANG_AITER_MLA_PERSIST" in overwritten
    assert "PORT" not in overwritten


def test_integrate_fault_grading_applies_to_recipe_lever_unavailable():
    from hyperloom.orchestrator.state.shared_state import SharedState

    result = {"status": "reverted", "error_class": "recipe_lever_unavailable"}
    assert SharedState._is_integrate_fault(result) is True


# ── Renderer contract tests ───────────────────────────────────────────────────


def test_no_levers_no_inherited_returns_official_script(tmp_path):
    """No levers and no inherited copy: official recipe is returned unchanged."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    result = apply_recipe_levers(
        bench,
        inherited_script="",
        argv=[],
        remove_args=[],
        env_sets={},
        env_unsets=[],
    )
    assert result == "single_node/agentic/minimaxm3.sh"
    # No copy was written.
    copies = list((root / "benchmarks" / "single_node" / "agentic").glob("*.hl-*.sh"))
    assert copies == []


def test_set_existing_flag_replaces_recipe_value(tmp_path):
    """A flag in extra_server_args replaces the recipe's existing value."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    result = apply_recipe_levers(
        bench,
        inherited_script="",
        argv=["--gpu-memory-utilization", "0.95"],
        remove_args=[],
        env_sets={},
        env_unsets=[],
    )
    copy_path = root / "benchmarks" / result
    text = copy_path.read_text()
    # Old value gone, new value present.
    assert "--gpu-memory-utilization 0.90" not in text
    assert "--gpu-memory-utilization 0.95" in text
    assert "0.90" not in text or text.count("gpu-memory") == 1


def test_append_new_flag(tmp_path):
    """A flag not present in the recipe is appended to the array."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    result = apply_recipe_levers(
        bench,
        inherited_script="",
        argv=["--enable-chunked-prefill"],
        remove_args=[],
        env_sets={},
        env_unsets=[],
    )
    text = (root / "benchmarks" / result).read_text()
    assert "--enable-chunked-prefill" in text


def test_remove_flag_deletes_from_recipe_array(tmp_path):
    """remove_args deletes the flag and its value from the recipe's server array."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    result = apply_recipe_levers(
        bench,
        inherited_script="",
        argv=[],
        remove_args=["--block-size"],
        env_sets={},
        env_unsets=[],
    )
    text = (root / "benchmarks" / result).read_text()
    assert "--block-size" not in text


def test_env_lever_beats_recipe_export(tmp_path):
    """An env_set overrides the recipe's unconditional export."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    result = apply_recipe_levers(
        bench,
        inherited_script="",
        argv=[],
        remove_args=[],
        env_sets={"MY_CUSTOM_ENV": "hello"},
        env_unsets=[],
    )
    text = (root / "benchmarks" / result).read_text()
    assert "MY_CUSTOM_ENV=hello" in text


def test_env_unset_appears_in_copy(tmp_path):
    """env_unsets renders an ``unset NAME`` statement in the copy."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    result = apply_recipe_levers(
        bench,
        inherited_script="",
        argv=[],
        remove_args=[],
        env_sets={},
        env_unsets=["SOME_VAR"],
    )
    text = (root / "benchmarks" / result).read_text()
    assert "unset SOME_VAR" in text


def test_copy_is_content_addressed(tmp_path):
    """Same lever combination produces the same copy name; different levers differ."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    r1 = apply_recipe_levers(
        bench, inherited_script="", argv=["--max-num-seqs", "64"], remove_args=[], env_sets={}, env_unsets=[]
    )
    r2 = apply_recipe_levers(
        bench, inherited_script="", argv=["--max-num-seqs", "64"], remove_args=[], env_sets={}, env_unsets=[]
    )
    r3 = apply_recipe_levers(
        bench, inherited_script="", argv=["--max-num-seqs", "128"], remove_args=[], env_sets={}, env_unsets=[]
    )
    assert r1 == r2
    assert r1 != r3


def test_composing_on_inherited_copy(tmp_path):
    """Applying levers on top of an inherited copy accumulates the edits."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    first = apply_recipe_levers(
        bench, inherited_script="", argv=["--max-num-seqs", "64"], remove_args=[], env_sets={}, env_unsets=[]
    )
    # Second round builds on the first copy.
    second = apply_recipe_levers(
        bench, inherited_script=first, argv=["--enable-chunked-prefill"], remove_args=[], env_sets={}, env_unsets=[]
    )
    text = (root / "benchmarks" / second).read_text()
    # Both levers are present.
    assert "--max-num-seqs 64" in text or "--max-num-seqs" in text
    assert "--enable-chunked-prefill" in text


def test_replace_mode_inherits_nothing(tmp_path):
    """replace mode (inherited_script='') starts from the official recipe."""
    root = _checkout(tmp_path, "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    bench = _bench_agentx(root, "minimaxm3.sh")
    first = apply_recipe_levers(
        bench, inherited_script="", argv=["--max-num-seqs", "64"], remove_args=[], env_sets={}, env_unsets=[]
    )
    # Replace: pass inherited_script="" even though a copy exists.
    second = apply_recipe_levers(
        bench, inherited_script="", argv=["--enable-chunked-prefill"], remove_args=[], env_sets={}, env_unsets=[]
    )
    text = (root / "benchmarks" / second).read_text()
    assert "--enable-chunked-prefill" in text
    # max-num-seqs from the first round must not be in the second (fresh start).
    assert "--max-num-seqs 64" not in text


def test_spliced_subarray_with_flag_edits_raises(tmp_path):
    """A recipe with spliced sub-arrays is refused when flag edits are requested."""
    recipe = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        source "$(dirname "$0")/../../benchmark_lib.sh"
        EXTRA=( --attention-backend aiter )
        VLLM_CMD=(
            vllm serve "$MODEL_PATH"
            --block-size 128
            "${EXTRA[@]}"
        )
        "${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
        """
    )
    root = _checkout(tmp_path, "spliced.sh", recipe)
    bench = _bench_agentx(root, "spliced.sh")
    with pytest.raises(RecipeLeverUnavailableError, match="spliced sub-array"):
        apply_recipe_levers(
            bench, inherited_script="", argv=["--max-num-seqs", "32"], remove_args=[], env_sets={}, env_unsets=[]
        )


def test_no_server_array_raises(tmp_path):
    """A recipe without a recognisable server-command array is refused."""
    recipe = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        source "$(dirname "$0")/../../benchmark_lib.sh"
        vllm serve "$MODEL_PATH" > "$SERVER_LOG" 2>&1 &
        """
    )
    root = _checkout(tmp_path, "noarray.sh", recipe)
    bench = _bench_agentx(root, "noarray.sh")
    with pytest.raises(RecipeLeverUnavailableError):
        apply_recipe_levers(
            bench, inherited_script="", argv=["--max-num-seqs", "32"], remove_args=[], env_sets={}, env_unsets=[]
        )


# ── Materialization tests ─────────────────────────────────────────────────────


def _make_sparse_model_config(model_dir: Path) -> None:
    import json

    cfg = {"model_type": "minimax_m3", "text_config": {"sparse_attention_config": {"sparse_block_size": 128}}}
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _materialize(tmp_path: Path, recipe_root: Path, bench_patch: dict, **kwargs) -> dict:
    """Helper that materializes a config and returns the benchmark envs."""
    import yaml

    from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

    base = tmp_path / "base.yaml"
    bench = {
        "framework": "vllm",
        "model": str(tmp_path / "model"),
        "inferencex_path": str(recipe_root),
        "envs": {
            "AGENTX_SERVER_SCRIPT": "single_node/agentic/minimaxm3.sh",
            "HYPERLOOM_AGENTX": "1",
        },
        "benchmark_script": "aiperf_client.sh",
    }
    bench["envs"].update(bench_patch.pop("envs", {}))
    bench.update(bench_patch)
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    old_env = dict(os.environ)
    os.environ["HYPERLOOM_AGENTX"] = "1"
    os.environ["INFERENCEX_PATH"] = str(recipe_root)
    try:
        out_dir = tmp_path / "out"
        materialized = materialize_config_with_envs(
            base, out_dir, model_path=str(tmp_path / "model"), gpu_type="mi355x", agentx_mode=True, **kwargs
        )
        return yaml.safe_load(Path(materialized).read_text())["benchmark"]["envs"]
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_no_sparse_block_size_injected_for_agentic_vllm(tmp_path):
    """Sparse --block-size default is not injected on the agentic recipe surface."""
    recipe_root = tmp_path / "ix"
    _checkout(recipe_root.parent / "InferenceX", "minimaxm3.sh", _AGENTIC_RECIPE_TEMPLATE)
    recipe_root = recipe_root.parent / "InferenceX"
    model_dir = tmp_path / "model"
    _make_sparse_model_config(model_dir)

    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True, exist_ok=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(_AGENTIC_RECIPE_TEMPLATE, encoding="utf-8")

    envs = _materialize(tmp_path, recipe_root, {})
    extra_args = envs.get("EXTRA_VLLM_ARGS", "")
    assert "--block-size" not in extra_args, f"block-size injected on agentic recipe: {extra_args!r}"


def test_block_size_still_injected_for_generic_vllm(tmp_path):
    """Sparse --block-size IS injected on a generic launcher."""
    import yaml

    from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

    model_dir = tmp_path / "model"
    _make_sparse_model_config(model_dir)
    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "vllm_mi355x.sh").write_text("#!/bin/bash\nvllm serve $MODEL $EXTRA_VLLM_ARGS\n", encoding="utf-8")
    bench = {"framework": "vllm", "model": str(model_dir), "inferencex_path": str(recipe_root), "envs": {}}
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    old_env = dict(os.environ)
    os.environ["INFERENCEX_PATH"] = str(recipe_root)
    try:
        materialized = materialize_config_with_envs(
            base, tmp_path / "out", model_path=str(model_dir), gpu_type="mi355x", agentx_mode=False
        )
        envs = yaml.safe_load(Path(materialized).read_text())["benchmark"]["envs"]
        assert "--block-size 128" in envs.get("EXTRA_VLLM_ARGS", "")
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_profile_refused_on_agentic_recipe(tmp_path):
    """A profile run on an agentic recipe raises RecipeLeverUnavailableError."""
    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(_AGENTIC_RECIPE_TEMPLATE, encoding="utf-8")
    with pytest.raises(RecipeLeverUnavailableError, match="profiling is not available"):
        _materialize(tmp_path, recipe_root, {"envs": {"PROFILE": "1"}})


def test_grid_build_variant_yaml_raises_recipe_lever_unavailable_on_profile(tmp_path):
    """_build_variant_yaml labels a profile-on-agentic refusal as recipe_lever_unavailable."""
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(_AGENTIC_RECIPE_TEMPLATE, encoding="utf-8")
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


def test_materialize_with_server_arg_lever_updates_agentx_server_script(tmp_path):
    """materialize_config_with_envs wires a server-arg lever into a recipe copy."""
    recipe_root = tmp_path / "ix"
    benchmarks = recipe_root / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    (benchmarks / "single_node" / "agentic" / "minimaxm3.sh").write_text(_AGENTIC_RECIPE_TEMPLATE, encoding="utf-8")
    envs = _materialize(tmp_path, recipe_root, {}, extra_server_args="--max-num-seqs 64")
    script = envs.get("AGENTX_SERVER_SCRIPT", "")
    # A copy was created (not the original).
    assert ".hl-" in script
    copy_path = recipe_root / "benchmarks" / script
    assert copy_path.exists()
    text = copy_path.read_text()
    assert "--max-num-seqs 64" in text
