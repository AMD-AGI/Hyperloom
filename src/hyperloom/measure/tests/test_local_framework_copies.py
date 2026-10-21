# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test that rewrites the framework's tuned tables does not change what the next test reads.

The framework here is a small stand-in install in the test's own directory, never the machine's ``/sgl-workspace``.
The stand-in for ``run_grid`` plays the inference server: it reads the tables the environment it is given names, and
rewrites them the way aiter's duplicate-shape cleanup does, through the path the variable holds.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.measure import local
from hyperloom.measure.framework_base import (
    FRAMEWORK_BASE_FILE,
    SNAPSHOT_ENV,
    framework_envs,
    load_snapshot,
    tree_files,
)
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.socket import SessionFacts, TestCall
from hyperloom.measure.tests.framework_fixtures import DAMAGED, PRISTINE, SHARED_GEMM, make_framework
from hyperloom.orchestrator.actions.executors import VariantResult


@pytest.fixture(autouse=True)
def magpie_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
    for name in list(os.environ):
        if name.startswith(("AITER_CONFIG_", "HYPERLOOM_FRAMEWORK_")):
            monkeypatch.delenv(name)


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    return make_framework(tmp_path)


@pytest.fixture
def session(tmp_path: Path, configs: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The toolbox's start: the snapshot is taken and the runner template's variables are this process's."""
    directory = tmp_path / "framework"
    for name, value in framework_envs(directory, str(configs)).items():
        monkeypatch.setenv(name, value)
    return directory


class _Server:
    """``run_grid`` as the inference server's side of it: the environment it gets, what it does with the tables."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.act: Callable[[dict[str, str]], None] = lambda env: None
        self.envs: list[dict[str, str]] = []
        self.unset: list[list[str]] = []
        self.files: list[set[str]] = []  # the listed tables that existed while the server ran

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        config = yaml.safe_load(Path(kwargs["base_yaml_path"]).read_text())
        unset = list(kwargs["grid"][0].unset_envs)
        env = {**os.environ, **{str(k): str(v) for k, v in config["benchmark"]["envs"].items()}}
        env = {k: v for k, v in env.items() if k not in unset}
        self.envs.append(env)
        self.unset.append(unset)
        listed = {
            part for name, value in env.items() if name.startswith("AITER_CONFIG_") for part in value.split(os.pathsep)
        }
        self.files.append({part for part in listed if Path(part).is_file()})
        self.act(env)
        workspace = self.tmp_path / "ws" / str(len(self.envs))
        workspace.mkdir(parents=True, exist_ok=True)
        fields = {"extra_server_args": "", "extra_envs": {}, "status": "succeeded", "returncode": 0}
        return [VariantResult(name="test", workspace=str(workspace), **fields)]


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Server:
    fake = _Server(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    return fake


def _point(tmp_path: Path, **envs: str) -> Path:
    path = tmp_path / "point.yaml"
    benchmark = {"framework": "sglang", "envs": {"TP": 1, "CONC": 4, "RUN_EVAL": "false", **envs}}
    path.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    return path


def _run(tmp_path: Path, name: str, point: Path | None = None, **kwargs: Any) -> dict[str, Any] | None:
    """One test through the local backend's entry point; its ``framework_base.json``, if it left one."""
    out = tmp_path / "out" / name
    call = TestCall(
        points=(point or _point(tmp_path),),
        output_dir=out,
        timeout_sec=900,
        cwd=str(tmp_path),
        magpie_python="",
        task_id=name,
        **kwargs,
    )
    LocalMeasurementBackend().run_test(call)
    evidence = out / FRAMEWORK_BASE_FILE
    return json.loads(evidence.read_text()) if evidence.is_file() else None


def _paths(value: str) -> list[Path]:
    return [Path(part) for part in value.split(os.pathsep)]


def _rewrite_every_table(env: dict[str, str]) -> None:
    """What aiter's write-back does: rewrite each source of each variable in place."""
    for name, value in env.items():
        if name.startswith("AITER_CONFIG_"):
            for path in _paths(value):
                path.write_text(DAMAGED, encoding="utf-8")


def _tables_aiter_reads(env: dict[str, str], configs: Path) -> list[Path]:
    """The tables aiter reads for the GEMM variable: the ones it names, else the install's default and per-model ones."""
    if "AITER_CONFIG_GEMM" in env:
        return _paths(env["AITER_CONFIG_GEMM"])
    models = [
        entry.path
        for entry in os.scandir(configs / "model_configs")
        if fnmatch.fnmatchcase(entry.name, "*gemm_tuned*.csv") and "untuned" not in entry.name
    ]
    return [configs / SHARED_GEMM, *map(Path, models)]


def test_a_test_that_rewrites_its_tables_does_not_change_what_the_next_test_reads(
    tmp_path: Path, configs: Path, session: Path, server: _Server
) -> None:
    shared = tree_files(configs)
    reads: list[list[str]] = []

    def act(env: dict[str, str]) -> None:
        tables = _tables_aiter_reads(env, configs)
        reads.append([table.read_text() for table in tables])
        for table in tables:
            table.write_text(DAMAGED, encoding="utf-8")

    server.act = act
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")

    assert reads[0] == reads[1], "the second test read what the first did, not what the first left"
    assert reads[1][0] == PRISTINE
    assert tree_files(configs) == shared, "the shared tables are as they were"
    for evidence in (first, second):
        assert evidence is not None and evidence["isolation"] == "private_copies"
        assert evidence["matches_session"] is True and evidence["framework_base_changed"] is False
        assert evidence["shared_sha256_start"] == evidence["shared_sha256_end"] == evidence["session_sha256"]
        assert "configs/gemm_tuned.csv" in evidence["private_rewritten"], "the write-back landed in the test's copy"
        assert evidence["restored"] == {"start": [], "end": []}


def test_a_variable_the_test_does_not_set_names_private_copies_of_what_aiter_reads_when_it_is_unset(
    tmp_path: Path, configs: Path, session: Path, server: _Server
) -> None:
    _run(tmp_path, "t")

    env = server.envs[0]
    snapshot = load_snapshot(session)
    gemm = _paths(env["AITER_CONFIG_GEMM"])
    in_listed_order = [name for name in snapshot.model_order if name.startswith(("a_gemm", "b_gemm"))]
    assert [path.name for path in gemm] == [SHARED_GEMM, *in_listed_order], "the order aiter's glob lists, no untuned"
    assert sorted(in_listed_order) == ["a_gemm_tuned.csv", "b_gemm_tuned.csv"]
    assert [path.name for path in _paths(env["AITER_CONFIG_MOE"])] == ["moe_tuned.csv", "d_moe_tuned.csv"]
    assert [path.name for path in _paths(env["AITER_CONFIG_SPLIT"])] == ["split_name_tuned.csv"]
    for name in ("AITER_CONFIG_GEMM", "AITER_CONFIG_MOE", "AITER_CONFIG_SPLIT"):
        for path in _paths(env[name]):
            assert str(path) in server.files[0] and configs not in path.parents, f"{path} is not a private copy"
    assert "AITER_CONFIG_GEMM" not in os.environ, "the process environment is put back"


def test_the_tables_a_test_names_itself_are_copied_and_the_variable_names_the_copies(
    tmp_path: Path, configs: Path, session: Path, server: _Server
) -> None:
    own = tmp_path / "own.csv"
    own.write_text(PRISTINE + "7,7,7,1\n", encoding="utf-8")
    point = _point(tmp_path, AITER_CONFIG_GEMM=f"{configs / SHARED_GEMM}{os.pathsep}{own}")
    shared = tree_files(configs)
    server.act = _rewrite_every_table

    first = _run(tmp_path, "first", point)
    second = _run(tmp_path, "second", point)

    assert tree_files(configs) == shared
    assert own.read_text() == PRISTINE + "7,7,7,1\n", "the point's own table is not written either"
    for env in server.envs:
        assert all(configs not in path.parents and path != own for path in _paths(env["AITER_CONFIG_GEMM"]))
    assert first is not None and second is not None
    listed = first["point_tables"]["AITER_CONFIG_GEMM"]
    assert [t["source"] for t in listed] == [str(configs / SHARED_GEMM), str(own)]
    assert second["point_tables"] == first["point_tables"], "the second test copied the same pristine tables"


def test_a_table_the_test_lists_that_is_not_there_stays_listed_as_it_was(
    tmp_path: Path, configs: Path, session: Path, server: _Server
) -> None:
    missing = str(tmp_path / "missing.csv")
    _run(tmp_path, "t", _point(tmp_path, AITER_CONFIG_MOE=f"{configs / 'moe_tuned.csv'}{os.pathsep}{missing}"))

    listed = server.envs[0]["AITER_CONFIG_MOE"].split(os.pathsep)
    assert listed[1] == missing and listed[0] in server.files[0] and configs not in Path(listed[0]).parents


def test_a_variable_the_process_carries_is_copied_like_one_the_point_names(
    tmp_path: Path, configs: Path, session: Path, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = tmp_path / "image.csv"
    inherited.write_text(PRISTINE, encoding="utf-8")
    monkeypatch.setenv("AITER_CONFIG_GEMM", str(inherited))
    server.act = _rewrite_every_table

    _run(tmp_path, "t")

    assert inherited.read_text() == PRISTINE
    assert os.environ["AITER_CONFIG_GEMM"] == str(inherited)


def test_unsetting_a_table_variable_means_its_defaults_and_the_defaults_are_private(
    tmp_path: Path, configs: Path, session: Path, server: _Server
) -> None:
    facts = SessionFacts(reference_unset_envs=("AITER_CONFIG_MOE",))
    _run(tmp_path, "t", session_facts=facts, unset_envs=("AITER_CONFIG_GEMM", "STALE_PIN"))

    env = server.envs[0]
    assert server.unset[0] == ["STALE_PIN"]
    for name in ("AITER_CONFIG_GEMM", "AITER_CONFIG_MOE"):
        assert all(configs not in path.parents for path in _paths(env[name]))


def test_a_write_no_variable_redirects_is_caught_marked_and_put_back_before_the_next_test(
    tmp_path: Path, configs: Path, session: Path, server: _Server
) -> None:
    shared = tree_files(configs)
    server.act = lambda env: (configs / SHARED_GEMM).write_text(DAMAGED, encoding="utf-8")

    first = _run(tmp_path, "first")

    assert first is not None and first["framework_base_changed"] is True
    assert first["restored"]["end"] == [SHARED_GEMM] and first["restore_failed"] == []
    assert first["shared_sha256_end"] != first["session_sha256"]
    assert tree_files(configs) == shared, "put back from the snapshot"

    (configs / "moe_tuned.csv").write_text(DAMAGED, encoding="utf-8")  # something else, between two tests
    server.act = lambda env: None
    second = _run(tmp_path, "second")

    assert second is not None and second["restored"]["start"] == ["moe_tuned.csv"]
    assert second["framework_base_changed"] is False and second["matches_session"] is True
    assert tree_files(configs) == shared


def test_a_file_that_cannot_be_put_back_is_reported_and_the_test_still_ran(
    tmp_path: Path, configs: Path, session: Path, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(snapshot: Any, relatives: Any) -> tuple[list[str], list[str]]:
        return [], list(relatives)

    monkeypatch.setattr("hyperloom.measure.framework_copies.restore", refuse)
    server.act = lambda env: (configs / SHARED_GEMM).write_text(DAMAGED, encoding="utf-8")

    evidence = _run(tmp_path, "t")

    assert evidence is not None and evidence["restore_failed"] == [SHARED_GEMM]
    assert len(server.envs) == 1


def test_a_machine_whose_snapshot_is_unusable_still_runs_the_test_and_says_so(
    tmp_path: Path, configs: Path, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SNAPSHOT_ENV, str(tmp_path / "no-such-snapshot"))

    evidence = _run(tmp_path, "t")

    assert len(server.envs) == 1 and "AITER_CONFIG_GEMM" not in server.envs[0]
    assert evidence is not None and evidence["isolation"] == "none"
    assert evidence["reason"].startswith("no usable snapshot") and evidence["matches_session"] is None


def test_copies_that_cannot_be_made_do_not_stop_the_test_which_then_says_it_ran_on_the_shared_tables(
    tmp_path: Path, configs: Path, session: Path, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_disk(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copytree", full_disk)
    shared = tree_files(configs)

    evidence = _run(tmp_path, "t")

    assert len(server.envs) == 1 and "AITER_CONFIG_GEMM" not in server.envs[0]
    assert evidence is not None and evidence["isolation"] == "none" and "No space left" in evidence["reason"]
    assert evidence["matches_session"] is True, "the shared tables were the session's"
    assert tree_files(configs) == shared
    assert not list(Path(os.environ.get("TMPDIR", "/tmp")).glob("hl-framework-*")), "nothing is left behind"


def test_a_process_without_a_snapshot_runs_the_call_as_it_is(tmp_path: Path, server: _Server) -> None:
    assert _run(tmp_path, "t") is None
    assert not (tmp_path / "out" / "t" / FRAMEWORK_BASE_FILE).exists()
    assert not any(name.startswith("AITER_CONFIG_") for name in server.envs[0])


def test_a_test_whose_copies_failed_halfway_reports_no_private_files_it_never_had(
    tmp_path: Path, configs: Path, session: Path, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = shutil.copyfile

    def full_disk_for_the_points_tables(source: Any, target: Any, **kwargs: Any) -> Any:
        if "tables" in Path(target).parts:
            raise OSError(28, "No space left on device")
        return real(source, target, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", full_disk_for_the_points_tables)
    own = tmp_path / "own.csv"
    own.write_text(PRISTINE, encoding="utf-8")

    evidence = _run(tmp_path, "t", _point(tmp_path, AITER_CONFIG_GEMM=str(own)))

    assert len(server.envs) == 1 and evidence is not None and evidence["isolation"] == "none"
    assert evidence["private_rewritten"] == [] and evidence["redirected"] == []


def test_a_variable_the_call_sets_empty_means_its_defaults_whatever_the_process_carries(
    tmp_path: Path, configs: Path, session: Path, server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = tmp_path / "image.csv"
    inherited.write_text(PRISTINE, encoding="utf-8")
    monkeypatch.setenv("AITER_CONFIG_GEMM", str(inherited))

    _run(tmp_path, "t", env_overrides={"AITER_CONFIG_GEMM": ""})

    names = [path.name for path in _paths(server.envs[0]["AITER_CONFIG_GEMM"])]
    assert names[0] == SHARED_GEMM and "image.csv" not in names
