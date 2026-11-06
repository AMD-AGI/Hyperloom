# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sweep arm's rounds share one persistent server: what Magpie's reuse check compares must not differ between them.

Magpie records the environment of the round that booted the server (every ``benchmark.envs`` variable except its
client-only ones) and refuses each later round whose own environment differs ("Reuse metadata mismatch"). These tests
run a real sweep arm through ``LocalMeasurementBackend`` with the points the toolbox renders and ``run_grid``'s own
``_build_variant_yaml``, and compare the rounds the way Magpie does.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, RecipeDelta
from hyperloom.measure import local
from hyperloom.measure.compile import CompiledEnvironment, LocalBlobStore, PackageBase, compile_environment
from hyperloom.measure.framework.base import framework_envs
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.points import default_canonical, generate_points
from hyperloom.measure.socket import EnvironmentRef, RoundPlan, TestCall
from hyperloom.measure.tests._recipes import IMAGE, recipe_with, sha
from hyperloom.measure.tests.framework_fixtures import make_framework
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
        #: (Magpie's launcher directory, the InferenceX ``benchmarks`` directory it copies them into when a round starts)
        self.copies_launchers: tuple[Path, Path] | None = None

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        variant = kwargs["grid"][0]
        candidate = _build_variant_yaml(
            kwargs["recipe"],
            variant,
            output_subdir=self.tmp_path / "candidates" / str(len(self.rounds)),
            server_lifecycle=kwargs["launch"].server_lifecycle,
        )
        envs = yaml.safe_load(candidate.read_text())["benchmark"]["envs"]
        if self.copies_launchers is not None:
            for script in self.copies_launchers[0].glob("*.sh"):
                shutil.copyfile(script, self.copies_launchers[1] / script.name)
        tables = [
            part for key, value in envs.items() if key.startswith("AITER_CONFIG_") for part in value.split(os.pathsep)
        ]
        self.rounds.append(
            {
                "name": variant.name,
                "boot": bool(kwargs["launch"].lifecycle_boot_only),
                "ready": bool(kwargs["launch"].server_already_ready),
                "envs": envs,
                "tables_present": [Path(part).read_text() for part in tables if Path(part).is_file()],
            }
        )
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


def _environment(tmp_path: Path, delta: RecipeDelta | None = None) -> CompiledEnvironment:
    return compile_environment(
        recipe_with(),
        delta or RecipeDelta(),
        blobs=LocalBlobStore(tmp_path / "blobs"),
        active_image_digest=IMAGE,
        locator=lambda _layout: PackageBase("sglang", tmp_path, None, None),
    )


def _points(
    tmp_path: Path, *, accuracy: bool, environment: CompiledEnvironment | None = None, **options: Any
) -> tuple[Path, ...]:
    """The points the toolbox renders for a sweep over ``LEVELS``: the real planner, the shipped baseline recipe."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    canonical = default_canonical(
        tmp_path / "canonical",
        primary_concurrency=LEVELS[0],
        config_path=default_baseline_config(),
        model_path=str(model),
        **options,
    )
    environment = environment or _environment(tmp_path)
    return generate_points(environment, canonical, tmp_path / "points", concurrency=LEVELS, accuracy=accuracy)


def _sweep(
    tmp_path: Path, *, accuracy: bool = False, environment: CompiledEnvironment | None = None, **options: Any
) -> TestCall:
    return TestCall(
        points=_points(tmp_path, accuracy=accuracy, environment=environment, **options),
        environment=environment or EnvironmentRef.current_stack(),
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


def test_on_magpie_the_warmup_and_the_measured_round_of_a_graded_test_have_one_server_identity(
    rounds, tmp_path, monkeypatch
):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
    (point,) = _points(tmp_path, accuracy=True)[:1]
    call = TestCall(
        points=(point,),
        output_dir=tmp_path / "out",
        timeout_sec=900,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t-1",
        kind="accuracy",
        accuracy=True,
        rounds=RoundPlan(warmup_rounds=1),
    )

    LocalMeasurementBackend().run_test(call)

    warmup, measured = rounds.rounds
    assert (warmup["name"], measured["name"]) == ("warmup", "test")
    assert (warmup["ready"], measured["ready"]) == (False, True), "the measured round reuses the server it booted"
    assert (warmup["envs"]["RUN_EVAL"], measured["envs"]["RUN_EVAL"]) == ("true", "false"), "one evaluation, first"
    assert magpie_server_env(measured["envs"]) == magpie_server_env(warmup["envs"]), "Magpie's reuse check passes"
    assert warmup["envs"]["NUM_PROMPTS"] == measured["envs"]["NUM_PROMPTS"], "both are full rounds"


TABLE_ENV = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
TABLE = b"cu_num,M,N,K,libtype,solidx\n304,64,5120,5120,hipblaslt,7\n"


@pytest.fixture
def framework_session(tmp_path, monkeypatch):
    """The toolbox's start on a machine with aiter's tables: the snapshot is taken and the runner's variables are set."""
    for name in list(os.environ):
        if name.startswith(("AITER_CONFIG_", "HYPERLOOM_FRAMEWORK_")):
            monkeypatch.delenv(name)
    for name, value in framework_envs(tmp_path / "framework", str(make_framework(tmp_path))).items():
        monkeypatch.setenv(name, value)


def test_a_put_kernel_table_has_one_path_in_every_round_of_a_sweep_arm(
    rounds, tmp_path, monkeypatch, framework_session
):
    """The private copy of a table is named by what the table is, not by the point that names it first.

    The boot round runs on the highest level and the rounds that measure on the levels below it, each from a point of
    its own that names the same table; a path that counted the points ("2-k-..." at the boot, "1-k-..." next) made
    Magpie's reuse check refuse every level after the boot.
    """
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
    LocalBlobStore(tmp_path / "blobs").put(TABLE)
    table = KernelArtifact(
        KernelKind.GEMM, "fp8", "kernel/t.csv", sha(TABLE), {"load": {"via": "env_file", "env": TABLE_ENV}}
    )
    environment = _environment(tmp_path, RecipeDelta(add_kernels=(table,)))

    LocalMeasurementBackend().run_test(_sweep(tmp_path, environment=environment))

    boot, *measured = rounds.rounds
    assert boot["boot"] and [r["name"] for r in measured] == ["c64", "c16", "c4"]
    for round_ in measured:
        assert magpie_server_env(round_["envs"]) == magpie_server_env(boot["envs"]), f"the round at {round_['name']}"
    for round_ in rounds.rounds:
        assert round_["tables_present"] == [TABLE.decode()], f"the table {round_['name']} names is a private copy"
        assert round_["envs"][TABLE_ENV] == boot["envs"][TABLE_ENV]


def test_a_variable_the_launcher_exports_is_kept_or_dropped_alike_in_every_round_of_a_sweep_arm(
    rounds, tmp_path, monkeypatch
):
    """Magpie puts its launcher in the InferenceX checkout when the first round starts, and ``run_grid`` drops from a
    round's environment what that launcher exports unconditionally: a checkout the runner has just made clean had no
    launcher while the boot round's configuration was built, so the boot kept ``SGLANG_USE_AITER`` and every round after
    it dropped it, and Magpie's reuse check refused them.
    """
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
    checkout, launchers = tmp_path / "inferencex", tmp_path / "magpie" / "Magpie" / "scripts" / "benchmark"
    for path in (checkout / "benchmarks", launchers):
        path.mkdir(parents=True)
    (checkout / "benchmarks" / "benchmark_lib.sh").write_text("")
    (launchers / "sglang_mi355x.sh").write_text("export SGLANG_USE_AITER=1\n")
    delta = RecipeDelta(config=ConfigDelta(envs={"SGLANG_USE_AITER": "1", "AITER_LOG_TUNED_CONFIG": "1"}))
    call = _sweep(tmp_path, environment=_environment(tmp_path, delta), gpu_type="mi355x")
    monkeypatch.setenv("INFERENCEX_PATH", str(checkout))
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / "magpie"))
    rounds.copies_launchers = (launchers, checkout / "benchmarks")

    LocalMeasurementBackend().run_test(call)

    boot, *measured = rounds.rounds
    assert boot["boot"] and [r["name"] for r in measured] == ["c64", "c16", "c4"]
    assert "AITER_LOG_TUNED_CONFIG" in boot["envs"], "a variable the launcher does not export stays"
    for round_ in measured:
        assert magpie_server_env(round_["envs"]) == magpie_server_env(boot["envs"]), f"the round at {round_['name']}"


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
