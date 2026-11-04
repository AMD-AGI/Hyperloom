# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch to aiter's tuned tables reaches the server through the test's private copies of the tables.

The image is an interpreter with a stand-in aiter installed whose ``jit/core.py`` declares the table variables the way
aiter's does. The session snapshots the image's tables; the test's farm patches one of them (a row added, a per-model
table added, one deleted). Before the fix the private copies were the snapshot's alone, every ``AITER_CONFIG_*`` named
them, and the server built the module the patch makes stale from the image's table: the candidate was never measured.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.farm import Farm, build_farm, patched_configs
from hyperloom.measure.framework.base import FRAMEWORK_BASE_FILE, framework_envs
from hyperloom.measure.framework.copies import private_framework
from hyperloom.measure.framework.table_modules import stale_table_modules
from hyperloom.measure.framework.tests.test_table_modules import BLOCK, BPRESHUFFLE, CONFIG, CORE, OPUS, TABLES
from hyperloom.measure.socket import TestCall
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.measure.tests._recipes import write_tree

DEFAULT = "configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
ADDED_ROW = "16,7168,2048,bp_candidate\n"
PATCHED_TABLE = (TABLES[DEFAULT] + ADDED_ROW).encode()
NEW_MODEL = "configs/model_configs/new_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
NEW_MODEL_TABLE = b"M,N,K,kernelName\n32,7168,2048,bp_new_model\n"
OLD_MODEL = "configs/model_configs/dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith(("AITER_CONFIG_", "HYPERLOOM_FRAMEWORK_", "AITER_META_DIR")):
            monkeypatch.delenv(name)


@pytest.fixture
def image(tmp_path, monkeypatch) -> tuple[str, Path]:
    """An interpreter with the stand-in aiter installed, its package directory, and the session's snapshot of it."""
    venv = tmp_path / "image"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    write_tree(site, {"aiter_meta/__init__.py": "", "aiter_meta/csrc/gemm.cu": "// image source\n"})
    package = site / "aiter"
    write_tree(package, {"__init__.py": "", "install_mode": "install", "jit/__init__.py": "", "jit/core.py": CORE})
    write_tree(package, TABLES)
    for name, value in framework_envs(tmp_path / "framework", str(package / "configs")).items():
        monkeypatch.setenv(name, value)
    return str(venv / "bin" / "python"), package


def _upsert(path: str, data: bytes) -> EnvEntry:
    return EnvEntry(path, "upsert", "sha256:" + hashlib.sha256(data).hexdigest(), "0644")


def _farm(tmp_path: Path, python: str, *entries: EnvEntry, blobs: dict[str, bytes]) -> Farm:
    by_address = {"sha256:" + hashlib.sha256(data).hexdigest(): data for data in blobs.values()}
    root = EnvRoot("aiter", "package", "aiter", "amd-aiter", "0", entries)
    farm = build_farm([root], tmp_path / "farm", python=python, read_blob=by_address.__getitem__)
    assert farm is not None
    return farm


def _call(tmp_path: Path, farm: Farm) -> TestCall:
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"envs": {"TP": 1}}}), encoding="utf-8")
    return TestCall(
        points=(point,),
        output_dir=tmp_path / "out",
        timeout_sec=60,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t1",
        tables_patch=patched_configs(farm),
    )


def _read(value: str) -> list[bytes]:
    return [Path(part).read_bytes() for part in value.split(os.pathsep)]


def test_a_row_the_configs_patch_adds_is_in_the_table_the_server_is_told_to_read(tmp_path, image) -> None:
    python, package = image
    farm = _farm(tmp_path, python, _upsert(DEFAULT, PATCHED_TABLE), blobs={DEFAULT: PATCHED_TABLE})
    call = _call(tmp_path, farm)

    with private_framework(call) as private:
        environment = dict(private.call.env_overrides)
        tables = _read(environment[BPRESHUFFLE])
        # what the runner judged stale before the copies were made, from the farm's package with no variable set,
        # is what the server's own environment makes stale: the module is built from the table the patch made
        judged = stale_table_modules(CONFIG, package, farm.site / "aiter", [{}])
        served = stale_table_modules(CONFIG, package, farm.site / "aiter", [environment])

    assert tables[0] == PATCHED_TABLE and ADDED_ROW.encode() in tables[0]
    assert judged == served == {BLOCK, OPUS}
    assert (package / DEFAULT).read_text() == TABLES[DEFAULT], "the image's table is not written"
    document = yaml.safe_load((tmp_path / "out" / FRAMEWORK_BASE_FILE).read_text())
    assert document["isolation"] == "private_copies" and document["configs_patched"] == [DEFAULT[len("configs/") :]]
    assert document["matches_session"] is True
    assert document["private_rewritten"] == [], "the patched table is what the test began on, not a write-back"


def test_a_per_model_table_the_patch_adds_or_deletes_is_merged_or_left_out(tmp_path, image) -> None:
    python, package = image
    delete = EnvEntry(OLD_MODEL, "delete", None, None)
    farm = _farm(tmp_path, python, _upsert(NEW_MODEL, NEW_MODEL_TABLE), delete, blobs={NEW_MODEL: NEW_MODEL_TABLE})

    with private_framework(_call(tmp_path, farm)) as private:
        value = private.call.env_overrides[BPRESHUFFLE]
        tables = _read(value)
        private_configs = Path(value.split(os.pathsep)[0]).parent
        deleted_copy = (private_configs / OLD_MODEL.removeprefix("configs/")).exists()

    assert tables == [TABLES[DEFAULT].encode(), NEW_MODEL_TABLE]
    assert not deleted_copy, "the private copies hold no table the patch deletes"
    assert (package / OLD_MODEL).is_file(), "the image keeps its table"


def test_a_farm_that_patches_no_table_leaves_the_private_copies_as_the_snapshot(tmp_path, image) -> None:
    python, _ = image
    code = b"PATCH = 'candidate'\n"
    farm = _farm(tmp_path, python, _upsert("ops/gemm.py", code), blobs={"ops": code})

    assert patched_configs(farm) == {}
    with private_framework(_call(tmp_path, farm)) as private:
        tables = _read(private.call.env_overrides[BPRESHUFFLE])

    assert tables == [TABLES[DEFAULT].encode(), TABLES[OLD_MODEL].encode()]
