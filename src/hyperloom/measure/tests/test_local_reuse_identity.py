# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sweep arm's rounds share one persistent server: what Magpie's reuse check compares must not differ between them.

Magpie records the environment of the round that booted the server (every ``benchmark.envs`` variable except its
client-only ones) and refuses each later round whose own environment differs ("Reuse metadata mismatch"). These tests
run a real sweep arm through ``LocalMeasurementBackend`` with the points the toolbox renders and ``run_grid``'s own
``_build_variant_yaml``, and compare the rounds the way Magpie does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure import local
from hyperloom.measure.compile import LocalBlobStore, PackageBase, compile_environment
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.points import default_canonical, generate_points
from hyperloom.measure.socket import TestCall
from hyperloom.measure.tests._recipes import IMAGE, recipe_with
from hyperloom.orchestrator.actions.executors import VariantResult
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config

LEVELS = (4, 16, 64)
PORT = 8123

#: Magpie's ``CLIENT_ONLY_REUSE_ENV_KEYS`` and the prefixes its ``_desired_reuse_server_meta`` leaves out of the
#: ``server_env`` it stores and compares (pinned Magpie ``modes/benchmark/benchmarker.py``). Replicated here because
#: Magpie is not a dependency of the test environment; ``test_the_copy_of_magpies_filter_is_magpies`` holds the copy to
#: Magpie's own function wherever Magpie is importable.
MAGPIE_CLIENT_ONLY = frozenset({"CONC", "ISL", "OSL", "NUM_PROMPTS", "RANDOM_RANGE_RATIO", "REQUEST_RATE", "RUN_EVAL"})
MAGPIE_GPU_KEYS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
MAGPIE_LEFT_OUT_PREFIXES = ("EVAL_", "MAGPIE_EVAL_", "RESULT_")


def magpie_server_env(envs: dict[str, Any]) -> dict[str, str]:
    """What Magpie stores for, and compares of, a round: its ``benchmark.envs`` without the keys the reuse check skips."""
    upper = {str(key).upper(): str(value) for key, value in envs.items()}
    return {
        key: value
        for key, value in upper.items()
        if key not in MAGPIE_CLIENT_ONLY and key not in MAGPIE_GPU_KEYS and not key.startswith(MAGPIE_LEFT_OUT_PREFIXES)
    }


@pytest.fixture(autouse=True)
def clean_process_env(monkeypatch, tmp_path):
    """The renderer reads the workload from the process environment; start from a known, empty one."""
    for name in (
        "CONC",
        "ISL",
        "OSL",
        "RANDOM_RANGE_RATIO",
        "RUN_EVAL",
        "TP",
        "EP",
        "PORT",
        "MAX_MODEL_LEN",
        "PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "MAGPIE_EVAL_TASKS",
        "MAGPIE_EVAL_LIMIT",
        "MAGPIE_EVAL_TOKENIZED_REQUESTS",
        "HYPERLOOM_BENCHMARK_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SERVER_ARGS", raising=False)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "userdata"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")


class _Rounds:
    """``run_grid`` stand-in that builds each round's configuration with ``run_grid``'s own ``_build_variant_yaml``."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.rounds: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        variant = kwargs["grid"][0]
        candidate = _build_variant_yaml(
            Path(kwargs["base_yaml_path"]),
            kwargs["base_extra_args"],
            variant,
            output_subdir=self.tmp_path / "candidates" / str(len(self.rounds)),
            server_lifecycle=kwargs.get("server_lifecycle"),
            base_extra_envs=kwargs.get("base_extra_envs"),
        )
        envs = yaml.safe_load(candidate.read_text())["benchmark"]["envs"]
        self.rounds.append({"name": variant.name, "boot": bool(kwargs.get("lifecycle_boot_only")), "envs": envs})
        workspace = self.tmp_path / "ws" / str(len(self.rounds))
        workspace.mkdir(parents=True)
        return [
            VariantResult(
                name=variant.name,
                extra_server_args="",
                extra_envs={},
                status="succeeded",
                returncode=0,
                workspace=str(workspace),
            )
        ]


@pytest.fixture
def rounds(monkeypatch, tmp_path) -> _Rounds:
    fake = _Rounds(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    monkeypatch.setattr(
        local,
        "resolve_lifecycle_params",
        lambda path: {"eligible": True, "port": PORT, "framework": "sglang", "reason": ""},
    )
    monkeypatch.setattr(local, "teardown_lifecycle_server", lambda **kw: None)
    return fake


def _points(tmp_path: Path, *, accuracy: bool) -> tuple[Path, ...]:
    """The points the toolbox renders for a sweep over ``LEVELS``: the real planner, the shipped baseline recipe."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    canonical = default_canonical(
        tmp_path / "canonical",
        primary_concurrency=LEVELS[0],
        config_path=default_baseline_config(),
        model_path=str(model),
    )
    environment = compile_environment(
        recipe_with(),
        RecipeDelta(),
        blobs=LocalBlobStore(tmp_path / "blobs"),
        active_image_digest=IMAGE,
        locator=lambda _layout: PackageBase("sglang", tmp_path, None, None),
    )
    return generate_points(environment, canonical, tmp_path / "points", concurrency=LEVELS, accuracy=accuracy)


def _sweep(tmp_path: Path, *, accuracy: bool = False) -> TestCall:
    return TestCall(
        points=_points(tmp_path, accuracy=accuracy),
        output_dir=tmp_path / "out",
        timeout_sec=900,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t-1",
        prep="conc_sweep",
        accuracy=accuracy,
    )


@pytest.mark.parametrize("accuracy", [False, True])
def test_on_magpie_the_boot_and_the_rounds_that_measure_have_one_server_identity(
    rounds, tmp_path, monkeypatch, accuracy
):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")

    LocalMeasurementBackend().run_test(_sweep(tmp_path, accuracy=accuracy))

    boot, *measured = rounds.rounds
    assert boot["boot"] and [r["name"] for r in measured] == ["c64", "c16", "c4"]
    assert not any(r["boot"] for r in measured)
    stored = magpie_server_env(boot["envs"])
    for round_ in measured:
        assert magpie_server_env(round_["envs"]) == stored, f"the round at {round_['name']}"
    assert "MAGPIE_RUN_PHASE" not in boot["envs"], "Magpie sets the phase itself"
    assert (boot["envs"]["RUN_EVAL"], boot["envs"]["NUM_PROMPTS"]) == ("false", "1"), "a boot is still small"


def test_on_bypass_the_boot_round_carries_the_server_phase(rounds, tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")

    LocalMeasurementBackend().run_test(_sweep(tmp_path))

    boot, *measured = rounds.rounds
    assert boot["envs"]["MAGPIE_RUN_PHASE"] == "server"
    assert not any("MAGPIE_RUN_PHASE" in r["envs"] for r in measured)


def test_the_copy_of_magpies_filter_is_magpies(tmp_path):
    benchmarker = pytest.importorskip("Magpie.modes.benchmark.benchmarker")
    envs = {
        **dict.fromkeys(sorted(MAGPIE_CLIENT_ONLY), "1"),
        "TP": "1",
        "NUM_WARMUPS": "4",
        "EVAL_X": "1",
        "RESULT_DIR": "x",
    }
    envs.update({"MAGPIE_EVAL_LIMIT": "5", "ROCR_VISIBLE_DEVICES": "0", "MAGPIE_RUN_PHASE": "server", "PORT": "8123"})
    config = SimpleNamespace(envs=envs, inferencex_path=str(tmp_path), framework="sglang", model="m", run_mode="local")

    desired = benchmarker.BenchmarkMode._desired_reuse_server_meta(SimpleNamespace(config=config), PORT)

    assert desired["server_env"] == magpie_server_env(envs)
    assert set(MAGPIE_CLIENT_ONLY) == set(benchmarker.CLIENT_ONLY_REUSE_ENV_KEYS)
