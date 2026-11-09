# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rounds of a sweep arm and of a graded test, as the local backend asks ``run_grid`` for them.

These tests run a real sweep arm through ``LocalMeasurementBackend`` with the points the toolbox renders and
``run_grid``'s own ``_build_variant_yaml``: one small boot, the levels from the highest down on the server it booted, one
evaluation last, and a put kernel table that every round names by the same path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.contracts.recipe import KernelArtifact, KernelKind, RecipeDelta
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
        "HYPERLOOM_EVAL_TASKS",
        "HYPERLOOM_EVAL_LIMIT",
        "HYPERLOOM_EVAL_TOKENIZED_REQUESTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SERVER_ARGS", raising=False)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "userdata"))


class _Rounds:
    """``run_grid`` stand-in that builds each round's configuration with ``run_grid``'s own ``_build_variant_yaml``."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.rounds: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        variant = kwargs["grid"][0]
        candidate = _build_variant_yaml(
            kwargs["recipe"],
            variant,
            output_subdir=self.tmp_path / "candidates" / str(len(self.rounds)),
            server_lifecycle=kwargs["launch"].server_lifecycle,
        )
        envs = yaml.safe_load(candidate.read_text())["benchmark"]["envs"]
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
    monkeypatch.setattr(local, "teardown_variant_server", lambda slot: [])
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
        task_id="t-1",
        prep="conc_sweep",
        accuracy=accuracy,
    )


@pytest.mark.parametrize("accuracy", [False, True])
def test_a_sweep_boots_small_then_measures_every_level_on_that_server_and_evaluates_once_last(
    rounds, tmp_path, monkeypatch, accuracy
):

    LocalMeasurementBackend().run_test(_sweep(tmp_path, accuracy=accuracy))

    boot, *measured = rounds.rounds
    evaluation = ["eval"] if accuracy else []
    assert boot["boot"] and [r["name"] for r in measured] == ["c64", "c16", "c4", *evaluation]
    assert not any(r["boot"] for r in measured)
    assert not any("HYPERLOOM_RUN_PHASE" in r["envs"] for r in rounds.rounds), "no round names a server-only phase"
    assert (boot["envs"]["RUN_EVAL"], boot["envs"]["NUM_PROMPTS"]) == ("false", "1"), "a boot is still small"
    assert [r["envs"]["RUN_EVAL"] for r in measured] == ["false"] * 3 + ["true"] * accuracy, "one evaluation, last"
    assert [r["envs"].get("EVAL_ONLY") for r in measured] == [None] * 3 + ["true"] * accuracy, "and no bench in it"


def test_a_graded_test_warms_up_measures_and_evaluates_last_on_the_server_its_warmup_booted(
    rounds, tmp_path, monkeypatch
):
    (point,) = _points(tmp_path, accuracy=True)[:1]
    call = TestCall(
        points=(point,),
        output_dir=tmp_path / "out",
        timeout_sec=900,
        cwd=str(tmp_path),
        task_id="t-1",
        kind="accuracy",
        accuracy=True,
        rounds=RoundPlan(warmup_rounds=1),
    )

    LocalMeasurementBackend().run_test(call)

    warmup, measured, _ = rounds.rounds
    assert [r["name"] for r in rounds.rounds] == ["warmup", "test", "eval"]
    assert [r["ready"] for r in rounds.rounds] == [False, True, True], "the later rounds reuse the server it booted"
    assert [r["envs"]["RUN_EVAL"] for r in rounds.rounds] == ["false", "false", "true"], "one evaluation, last"
    assert [r["envs"].get("EVAL_ONLY") for r in rounds.rounds] == [None, None, "true"], "and no bench in it"
    assert warmup["envs"]["NUM_PROMPTS"] == measured["envs"]["NUM_PROMPTS"], "both benches are full rounds"


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
    its own that names the same table; a path that counted the points ("2-k-...", then "1-k-...") would give the server
    the later rounds reuse another table path than the one it booted with.
    """
    LocalBlobStore(tmp_path / "blobs").put(TABLE)
    table = KernelArtifact(
        KernelKind.GEMM, "fp8", "kernel/t.csv", sha(TABLE), {"load": {"via": "env_file", "env": TABLE_ENV}}
    )
    environment = _environment(tmp_path, RecipeDelta(add_kernels=(table,)))

    LocalMeasurementBackend().run_test(_sweep(tmp_path, environment=environment))

    boot, *measured = rounds.rounds
    assert boot["boot"] and [r["name"] for r in measured] == ["c64", "c16", "c4"]
    for round_ in rounds.rounds:
        assert round_["tables_present"] == [TABLE.decode()], f"the table {round_['name']} names is a private copy"
        assert round_["envs"][TABLE_ENV] == boot["envs"][TABLE_ENV]
