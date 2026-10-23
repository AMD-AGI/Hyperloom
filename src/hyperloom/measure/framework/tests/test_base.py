# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's snapshot of the framework's tuned tables: taking it, keeping it, comparing the shared tables to it."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom.measure.framework.base import (
    DEFAULT_CONFIGS,
    SESSION_SHA_ENV,
    SNAPSHOT_ENV,
    FrameworkBaseError,
    files_digest,
    framework_envs,
    load_snapshot,
    restore,
    shared_view,
    take_snapshot,
    tree_files,
)
from hyperloom.measure.framework.tables import TableVariable, read_variables
from hyperloom.measure.tests.framework_fixtures import CORE, MODELS, PRISTINE, make_framework


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    return make_framework(tmp_path)


def test_the_variables_are_read_from_aiters_source_in_each_form_it_declares_them(tmp_path: Path) -> None:
    core = tmp_path / "core.py"
    core.write_text(CORE, encoding="utf-8")
    assert read_variables(core) == {
        "AITER_CONFIG_GEMM": TableVariable("gemm_tuned.csv", "gemm_tuned"),
        "AITER_CONFIG_MOE": TableVariable("moe_tuned.csv", "moe_tuned"),
        "AITER_CONFIG_SPLIT": TableVariable("split_name_tuned.csv", "split_tuned"),
    }


def test_a_source_that_is_missing_or_has_another_shape_names_no_variables(tmp_path: Path) -> None:
    assert read_variables(tmp_path / "absent.py") == {}
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n", encoding="utf-8")
    assert read_variables(broken) == {}
    other = tmp_path / "other.py"
    other.write_text("X = 1\n", encoding="utf-8")
    assert read_variables(other) == {}


def test_the_snapshot_holds_every_table_its_hashes_and_the_order_the_models_were_listed_in(
    tmp_path: Path, configs: Path
) -> None:
    taken = take_snapshot(configs, tmp_path / "snap")

    assert tree_files(taken.copies) == tree_files(configs) == dict(taken.files)
    assert sorted(taken.model_order) == sorted(MODELS) and "AITER_CONFIG_GEMM" in taken.variables
    assert load_snapshot(tmp_path / "snap") == taken
    assert taken.sha256 == files_digest(tree_files(configs))


def test_a_restarted_session_keeps_its_first_snapshot_even_when_the_tables_changed_since(
    tmp_path: Path, configs: Path
) -> None:
    first = take_snapshot(configs, tmp_path / "snap")
    (configs / "gemm_tuned.csv").write_text("damaged\n", encoding="utf-8")

    assert take_snapshot(configs, tmp_path / "snap").sha256 == first.sha256


def test_a_snapshot_whose_copies_were_changed_is_taken_again(tmp_path: Path, configs: Path) -> None:
    first = take_snapshot(configs, tmp_path / "snap")
    (first.copies / "gemm_tuned.csv").write_text("tampered\n", encoding="utf-8")

    again = take_snapshot(configs, tmp_path / "snap")

    assert again.sha256 == first.sha256 and (again.copies / "gemm_tuned.csv").read_text() == PRISTINE


def test_a_directory_that_holds_no_snapshot_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(FrameworkBaseError, match="no usable snapshot"):
        load_snapshot(tmp_path / "nothing")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "snapshot.json").write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(FrameworkBaseError):
        load_snapshot(tmp_path / "bad")


def test_the_shared_tables_are_compared_with_the_snapshot_and_put_back_from_it(tmp_path: Path, configs: Path) -> None:
    snapshot = take_snapshot(configs, tmp_path / "snap")
    assert shared_view(snapshot) == (snapshot.sha256, [])

    (configs / "gemm_tuned.csv").write_text("damaged\n", encoding="utf-8")
    (configs / "model_configs" / "a_gemm_tuned.csv").unlink()
    digest, changed = shared_view(snapshot)
    assert digest != snapshot.sha256 and sorted(changed) == ["gemm_tuned.csv", "model_configs/a_gemm_tuned.csv"]

    assert restore(snapshot, changed) == (changed, [])
    assert tree_files(configs) == dict(snapshot.files)


def test_a_file_that_cannot_be_put_back_is_named(tmp_path: Path, configs: Path) -> None:
    snapshot = take_snapshot(configs, tmp_path / "snap")
    (configs / "gemm_tuned.csv").write_text("damaged\n", encoding="utf-8")
    (configs / "model_configs").chmod(0o500)
    try:
        if os.geteuid() == 0:
            pytest.skip("root writes anywhere")
        assert restore(snapshot, ["model_configs/a_gemm_tuned.csv"]) == ([], ["model_configs/a_gemm_tuned.csv"])
    finally:
        (configs / "model_configs").chmod(0o700)


def test_the_runner_template_variables_name_the_snapshot_and_its_digest(tmp_path: Path, configs: Path) -> None:
    envs = framework_envs(tmp_path / "snap", str(configs))

    assert envs == {SNAPSHOT_ENV: str(tmp_path / "snap"), SESSION_SHA_ENV: load_snapshot(tmp_path / "snap").sha256}


def test_a_machine_without_the_tables_gets_no_variables(tmp_path: Path) -> None:
    assert framework_envs(tmp_path / "snap", str(tmp_path / "absent")) == {}
    assert framework_envs(tmp_path / "snap", "") == {}
    if not Path(DEFAULT_CONFIGS).is_dir():
        assert framework_envs(tmp_path / "snap") == {}


def test_a_snapshot_that_cannot_be_taken_still_names_its_directory_so_each_test_says_it_found_none(
    tmp_path: Path, configs: Path
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory")

    assert framework_envs(blocked / "snap", str(configs)) == {SNAPSHOT_ENV: str(blocked / "snap")}
