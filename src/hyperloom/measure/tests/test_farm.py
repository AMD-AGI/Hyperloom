# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The farm against a real installed package (PyYAML) and the interpreter running the tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.farm import FarmError, build_farm, place_farm, prepend_pythonpath, verify_farm
from hyperloom.measure.spec import EnvEntry, EnvRoot

INSTALLED = Path(yaml.__file__).parent
MARKER = b"MARKER = 'rebuilt'\nfrom yaml import *  # noqa\n"


def _address(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _root(*entries: EnvEntry, kind: str = "package") -> EnvRoot:
    return EnvRoot("yaml", kind, "yaml", "PyYAML", "0", tuple(entries))


def _blobs(*datas: bytes):
    store = {_address(d): d for d in datas}

    def read(address: str) -> bytes:
        return store[address]

    return read


def _build(tmp_path, *entries, read=None):
    return build_farm([_root(*entries)], tmp_path / "farm", python=sys.executable, read_blob=read or _blobs())


def test_untouched_files_are_links_to_the_image_and_the_image_is_not_written(tmp_path):
    before = sorted(p.name for p in INSTALLED.iterdir())

    farm = _build(tmp_path, EnvEntry("error.py", "delete"))

    assert (farm.site / "yaml" / "loader.py").is_symlink()
    assert (farm.site / "yaml" / "loader.py").resolve() == (INSTALLED / "loader.py").resolve()
    assert sorted(p.name for p in INSTALLED.iterdir()) == before


def test_an_upsert_is_a_real_copy_with_its_mode_and_a_delete_is_absent(tmp_path):
    entries = (
        EnvEntry("error.py", "delete"),
        EnvEntry("new_dir/added.py", "upsert", _address(b"x = 1\n"), "0640"),
    )

    farm = _build(tmp_path, *entries, read=_blobs(b"x = 1\n"))

    added = farm.site / "yaml" / "new_dir" / "added.py"
    assert added.read_bytes() == b"x = 1\n" and not added.is_symlink() and oct(added.stat().st_mode & 0o777) == "0o640"
    assert not (farm.site / "yaml" / "error.py").exists()


def test_a_replaced_file_is_the_rebuilt_one_not_the_images(tmp_path):
    entry = EnvEntry("__init__.py", "upsert", _address(MARKER), "0644")

    farm = _build(tmp_path, entry, read=_blobs(MARKER))

    assert (farm.site / "yaml" / "__init__.py").read_bytes() == MARKER
    assert (INSTALLED / "__init__.py").read_bytes() != MARKER


def test_a_blob_that_does_not_hash_to_its_address_is_refused(tmp_path):
    entry = EnvEntry("__init__.py", "upsert", _address(b"good"), "0644")

    with pytest.raises(FarmError, match="does not hash"):
        _build(tmp_path, entry, read=lambda address: b"bad")


def test_deleting_a_file_the_image_does_not_have_is_refused(tmp_path):
    with pytest.raises(FarmError, match="has no such file"):
        _build(tmp_path, EnvEntry("no_such.py", "delete"))


def test_a_runtime_root_is_refused_not_ignored(tmp_path):
    with pytest.raises(FarmError, match="runtime root"):
        build_farm([_root(kind="runtime")], tmp_path / "farm", python=sys.executable, read_blob=_blobs())


def test_a_sheet_that_changes_no_framework_builds_no_farm(tmp_path):
    assert build_farm([], tmp_path / "farm", python=sys.executable, read_blob=_blobs()) is None


def test_prepend_pythonpath_puts_the_farm_first_and_keeps_the_rest_once(tmp_path):
    envs = {"PYTHONPATH": "/rt:/other:/rt"}

    prepend_pythonpath(envs, tmp_path / "site")

    assert envs["PYTHONPATH"] == f"{tmp_path}/site:/rt:/other"


def test_the_python_that_gets_the_yaml_path_imports_the_rebuilt_package(tmp_path):
    entry = EnvEntry("__init__.py", "upsert", _address(MARKER), "0644")
    farm = _build(tmp_path, entry, read=_blobs(MARKER))
    envs: dict = {}
    prepend_pythonpath(envs, farm.site)

    verify_farm(farm, envs, python=sys.executable, cwd=tmp_path)

    done = subprocess.run(
        [sys.executable, "-c", "import yaml; print(yaml.MARKER)"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PYTHONPATH": envs["PYTHONPATH"], "PATH": ""},
    )
    assert done.stdout.strip() == "rebuilt"


def test_a_farm_that_is_not_first_on_the_path_is_refused(tmp_path):
    farm = _build(tmp_path, EnvEntry("error.py", "delete"))

    with pytest.raises(FarmError, match="would be imported from"):
        verify_farm(farm, {"PYTHONPATH": ""}, python=sys.executable, cwd=tmp_path)


def test_a_farm_that_rebuilds_no_aiter_carries_no_meta_directory_and_adds_no_variable(tmp_path):
    farm = _build(tmp_path, EnvEntry("error.py", "delete"))
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"envs": {"TP": 1}}}), encoding="utf-8")

    envs = place_farm(point, farm)

    assert farm.aiter_meta is None
    assert dict(envs) == {"TP": 1, "PYTHONPATH": str(farm.site)}
    assert yaml.safe_load(point.read_text())["benchmark"]["envs"] == dict(envs)
