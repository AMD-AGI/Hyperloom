# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which launcher owns the server argv, and the recipe copy that carries levers into an agentic one."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml
from hyperloom.orchestrator.actions.executors._recipe_script import (
    RecipeLeverUnavailableError,
    apply_recipe_levers,
    launcher_overwritten_envs,
    recipe_owns_argv,
    resolve_launch_server_script,
)
from hyperloom.orchestrator.actions.executors._server_argv import reseal_config_argv
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

_RECIPE = "single_node/agentic/minimaxm3_mtp.sh"

# The shape of an InferenceX agentic recipe, printing its argv and one env
# instead of booting a server.
_AGENTIC = """#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
export VLLM_ROCM_USE_AITER=1
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
SPEC_CONFIG='{"method": "eagle3", "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.78}'
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --block-size 128
    --gpu-memory-utilization 0.90 --max-num-seqs "$((2 * CONC))"
    --speculative-config "$SPEC_CONFIG"
    "${PARALLEL_ARGS[@]}"
)
printf '%s\\n' "${VLLM_CMD[@]}" > "$ARGV_OUT"
printf '%s\\n' "${VLLM_ROCM_USE_AITER-unset}" > "$ENV_OUT"
"""


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    (benchmarks / "single_node" / "agentic").mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("", encoding="utf-8")
    (benchmarks / _RECIPE).write_text(_AGENTIC, encoding="utf-8")
    (benchmarks / "vllm_mi355x.sh").write_text(
        'export HSA_NO_SCRATCH_RECLAIM=1\nexport PORT="${PORT:-8000}"\nvllm serve $MODEL $EXTRA_VLLM_ARGS\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(benchmarks.parent))
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", _RECIPE)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    return benchmarks


def _bench(script: str = _RECIPE) -> dict:
    return {"benchmark_script": "aiperf_client.sh", "framework": "vllm", "envs": {"AGENTX_SERVER_SCRIPT": script}}


def _levers(**kwargs) -> str:
    args = {"inherited_script": "", "server_args": "", "remove_args": [], "env_levers": {}, **kwargs}
    return apply_recipe_levers(_bench(), **args)


def _run(benchmarks: Path, script: str, tmp_path: Path) -> tuple[list[str], str]:
    """Execute a recipe and return the argv it launches with and its VLLM_ROCM_USE_AITER."""
    argv_out, env_out = tmp_path / "argv", tmp_path / "env"
    env = {"PATH": "/usr/bin:/bin", "MODEL_PATH": "/m", "CONC": "4", "TP": "2"}
    env.update(ARGV_OUT=str(argv_out), ENV_OUT=str(env_out))
    subprocess.run(["bash", str(benchmarks / script)], env=env, check=True)
    return argv_out.read_text().split("\n")[:-1], env_out.read_text().strip()


def test_only_an_agentic_recipe_owns_the_argv(checkout):
    assert recipe_owns_argv(_bench()) is True
    assert recipe_owns_argv({"benchmark_script": "vllm_mi355x.sh", "framework": "vllm"}) is False
    assert recipe_owns_argv(_bench("absent.sh")) is False


def test_an_agentic_ancestor_of_the_checkout_does_not_make_a_launcher_own_the_argv(tmp_path, monkeypatch):
    benchmarks = tmp_path / "agentic" / "InferenceX" / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("", encoding="utf-8")
    (benchmarks / "vllm_mi355x.sh").write_text("vllm serve $MODEL $EXTRA_VLLM_ARGS\n", encoding="utf-8")
    monkeypatch.setenv("INFERENCEX_PATH", str(benchmarks.parent))

    assert recipe_owns_argv({"benchmark_script": "vllm_mi355x.sh", "framework": "vllm"}) is False
    assert recipe_owns_argv(_bench("vllm_mi355x.sh")) is False


def test_the_agentx_client_resolves_to_the_script_beside_it_not_a_same_named_one_below(checkout):
    (checkout / "single_node" / "agentic" / "vllm_mi355x.sh").write_text("# a different recipe\n", encoding="utf-8")

    assert resolve_launch_server_script(_bench("vllm_mi355x.sh")) == str(checkout / "vllm_mi355x.sh")


def test_only_unguarded_exports_of_a_generic_launcher_count_as_overwritten(checkout):
    assert launcher_overwritten_envs(_bench()) == frozenset()
    # ``export PORT="${PORT:-8000}"`` defers to a caller-supplied value.
    assert launcher_overwritten_envs({"benchmark_script": "vllm_mi355x.sh"}) == {"HSA_NO_SCRATCH_RECLAIM"}


def test_a_variant_env_the_generic_launcher_overwrites_is_dropped(checkout, tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX")
    base = _config(tmp_path)
    base_yaml = yaml.safe_load(base.read_text())
    base_yaml["benchmark"]["benchmark_script"] = "vllm_mi355x.sh"
    base.write_text(yaml.safe_dump(base_yaml), encoding="utf-8")
    variant = GridVariant(name="v", extra_envs={"HSA_NO_SCRATCH_RECLAIM": "0", "VLLM_SOMETHING_ELSE": "1"})

    out = _build_variant_yaml(base, "", variant, output_subdir=tmp_path / "slot")

    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "HSA_NO_SCRATCH_RECLAIM" not in envs
    assert envs["VLLM_SOMETHING_ELSE"] == "1"


def test_no_lever_boots_the_official_recipe(checkout):
    assert _levers() == _RECIPE
    assert not list(checkout.rglob("*.hl-*.sh"))


def test_a_flag_replaces_the_recipe_value_and_new_flags_append(checkout, tmp_path):
    copy = _levers(server_args='--gpu-memory-utilization 0.95 --compilation-config {"level":3}')

    argv, _ = _run(checkout, copy, tmp_path)

    assert argv.count("--gpu-memory-utilization") == 1
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.95"
    assert argv[argv.index("--compilation-config") + 1] == '{"level":3}'
    assert argv[argv.index("--max-num-seqs") + 1] == "8"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"


def test_a_removal_deletes_the_recipe_flag_and_its_value(checkout, tmp_path):
    argv, _ = _run(checkout, _levers(remove_args=["--block-size"]), tmp_path)

    assert "--block-size" not in argv and "128" not in argv


def test_env_levers_override_the_recipe_export(checkout, tmp_path):
    assert _run(checkout, _levers(env_levers={"VLLM_ROCM_USE_AITER": "0"}), tmp_path)[1] == "0"
    assert _run(checkout, _levers(env_levers={"VLLM_ROCM_USE_AITER": None}), tmp_path)[1] == "unset"


def test_a_flag_set_inside_a_spliced_array_is_refused(checkout):
    with pytest.raises(RecipeLeverUnavailableError, match="PARALLEL_ARGS"):
        _levers(server_args="--tensor-parallel-size 4")


@pytest.mark.parametrize(
    "levers",
    [
        {"server_args": '--speculative-config {"num_speculative_tokens":4}'},
        {"server_args": '--speculative-config {"synthetic_acceptance_length":3.5}'},
        {"server_args": "--speculative-config.attention_backend TRITON_MLA"},
        {"remove_args": ["--speculative-config"]},
        {"remove_args": ["--spec-decode-acceptance-length"]},
        {"server_args": "--num-speculative-tokens 7"},
        {"env_levers": {"SGLANG_SIMULATE_ACC_LEN": "6"}},
    ],
    ids=[
        "config-draft-length",
        "config-acceptance",
        "config-dotted",
        "config-removed",
        "atom-acceptance",
        "atom-draft-length",
        "sglang-simulated-acceptance",
    ],
)
def test_a_lever_on_the_draft_or_its_simulated_acceptance_is_refused(checkout, levers):
    with pytest.raises(RecipeLeverUnavailableError, match="simulated acceptance"):
        _levers(**levers)


def test_a_draft_length_lever_on_an_sglang_recipe_is_refused(checkout):
    sglang = "single_node/agentic/glm_sglang_mtp.sh"
    (checkout / sglang).write_text(
        "export SGLANG_SIMULATE_ACC_LEN=3.61\n"
        "SGLANG_CMD=(\n    python3 -m sglang.launch_server\n"
        "    --speculative-num-steps 5\n    --speculative-num-draft-tokens 6\n)\n"
        '"${SGLANG_CMD[@]}" &\n',
        encoding="utf-8",
    )
    bench = {"benchmark_script": "aiperf_client.sh", "framework": "sglang", "envs": {"AGENTX_SERVER_SCRIPT": sglang}}

    with pytest.raises(RecipeLeverUnavailableError, match="speculative-num-steps"):
        apply_recipe_levers(
            bench,
            inherited_script="",
            server_args="--speculative-num-steps 3 --speculative-num-draft-tokens 4",
            remove_args=[],
            env_levers={},
        )


def test_a_draft_compute_key_merges_into_the_recipe_speculative_config(checkout, tmp_path):
    first = _levers(server_args='--speculative-config {"attention_backend":"ROCM_AITER_MLA"}')
    both = _levers(inherited_script=first, server_args='--speculative-config {"kv_cache_dtype":"fp8"}')

    argv, _ = _run(checkout, both, tmp_path)

    assert argv.count("--speculative-config") == 1
    assert json.loads(argv[argv.index("--speculative-config") + 1]) == {
        "method": "eagle3",
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.78,
        "attention_backend": "ROCM_AITER_MLA",
        "kv_cache_dtype": "fp8",
    }


def test_the_copy_is_content_addressed_beside_the_recipe(checkout):
    first = _levers(server_args="--max-num-seqs 64")

    assert first == _levers(server_args="--max-num-seqs 64")
    assert first != _levers(server_args="--max-num-seqs 32")
    assert first.startswith("single_node/agentic/minimaxm3_mtp.hl-")


def test_levers_compose_on_an_inherited_copy_and_replace_starts_over(checkout, tmp_path):
    inherited = _levers(server_args="--max-num-seqs 64")

    kept, _ = _run(checkout, _levers(inherited_script=inherited, env_levers={"A": "1"}), tmp_path)
    removed, _ = _run(checkout, _levers(inherited_script=inherited, remove_args=["--max-num-seqs"]), tmp_path)
    replaced, _ = _run(checkout, _levers(env_levers={"A": "1"}), tmp_path)

    assert kept[kept.index("--max-num-seqs") + 1] == "64"
    assert "--max-num-seqs" not in removed
    assert replaced[replaced.index("--max-num-seqs") + 1] == "8"


def test_a_copy_of_another_recipe_is_not_inherited(checkout):
    assert _levers(inherited_script="single_node/agentic/other.hl-0123456789ab.sh") == _RECIPE


def _config(tmp_path: Path, **envs) -> Path:
    base = tmp_path / "base.yaml"
    bench = {"framework": "vllm", "model": str(tmp_path / "model"), "envs": envs}
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return base


def _sparse_model(tmp_path: Path) -> str:
    model = tmp_path / "model"
    model.mkdir()
    config = {"text_config": {"sparse_attention_config": {"sparse_block_size": 128}}}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model)


def _materialize(tmp_path: Path, *, agentx: bool, **kwargs) -> dict:
    out = materialize_config_with_envs(
        _config(tmp_path),
        tmp_path / "out",
        model_path=_sparse_model(tmp_path),
        gpu_type="mi355x",
        agentx_mode=agentx,
        **kwargs,
    )
    return yaml.safe_load(out.read_text())["benchmark"]["envs"]


def test_an_agentic_recipe_gets_no_launcher_default_and_carries_the_lever(checkout, tmp_path):
    envs = _materialize(tmp_path, agentx=True, extra_server_args="--max-num-seqs 64")

    assert envs["EXTRA_VLLM_ARGS"] == "--max-num-seqs 64"
    argv, _ = _run(checkout, envs["AGENTX_SERVER_SCRIPT"], tmp_path)
    assert argv[argv.index("--max-num-seqs") + 1] == "64"
    assert argv[argv.index("--block-size") + 1] == "128"


def test_a_generic_launcher_still_gets_launcher_defaults(checkout, tmp_path):
    assert "--block-size 128" in _materialize(tmp_path, agentx=False)["EXTRA_VLLM_ARGS"]


def test_profiling_an_agentic_recipe_is_refused(checkout, tmp_path):
    with pytest.raises(RecipeLeverUnavailableError, match="no server phase to profile"):
        materialize_config_with_envs(
            _config(tmp_path, PROFILE="1"), tmp_path / "out", gpu_type="mi355x", agentx_mode=True
        )


def test_a_replacing_grid_base_keeps_its_removals_on_the_recipe(checkout, tmp_path):
    base = _config(tmp_path, AGENTX_SERVER_SCRIPT=_RECIPE)
    base_yaml = yaml.safe_load(base.read_text())
    base_yaml["benchmark"].update(benchmark_script="aiperf_client.sh", runner_type="mi355x")
    base.write_text(yaml.safe_dump(base_yaml), encoding="utf-8")

    out = _build_variant_yaml(
        base,
        "--gpu-memory-utilization 0.8",
        GridVariant(name="v", extra_server_args="--max-num-seqs 16"),
        output_subdir=tmp_path / "slot",
        base_args_mode="replace",
        base_remove_args=["--block-size"],
    )

    argv, _ = _run(checkout, yaml.safe_load(out.read_text())["benchmark"]["envs"]["AGENTX_SERVER_SCRIPT"], tmp_path)
    assert "--block-size" not in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "16"


def test_a_resealed_argv_rerenders_the_copy_without_the_dropped_flag(checkout, tmp_path):
    envs = _materialize(tmp_path, agentx=True, extra_server_args="--max-num-seqs 64 --moe-backend triton")
    config = tmp_path / "out" / "baseline_config.with_envs.yaml"

    reseal_config_argv(config, "--max-num-seqs 64")

    resealed = yaml.safe_load(config.read_text())["benchmark"]["envs"]
    argv, _ = _run(checkout, resealed["AGENTX_SERVER_SCRIPT"], tmp_path)
    assert resealed["AGENTX_SERVER_SCRIPT"] != envs["AGENTX_SERVER_SCRIPT"]
    assert "--moe-backend" not in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "64"


def test_the_refusal_is_graded_as_an_integration_fault():
    """A patch that was never benchmarked must not burn the gate's verdict quota."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    assert SharedState._is_integrate_fault({"status": "reverted", "error_class": "recipe_lever_unavailable"}) is True
