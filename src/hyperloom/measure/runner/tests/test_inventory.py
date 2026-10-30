# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib

import pytest

from hyperloom.measure.runner.inventory import OutputRejected, drop_agentx_runtime, inventory


def test_every_file_is_listed_with_its_size_and_hash(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.log").write_bytes(b"hello")
    (tmp_path / "y.json").write_bytes(b"{}")

    files = inventory(tmp_path, max_total_bytes=100)

    assert [(f.path, f.size, f.sha256) for f in files] == [
        ("a/x.log", 5, hashlib.sha256(b"hello").hexdigest()),
        ("y.json", 2, hashlib.sha256(b"{}").hexdigest()),
    ]


def test_a_symlink_is_refused_as_unsafe_output(tmp_path):
    (tmp_path / "secret").write_text("s")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "link").symlink_to(tmp_path / "secret")

    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path / "out", max_total_bytes=100)

    assert caught.value.error_class == "unsafe_output"


def test_output_over_the_limit_is_refused_as_too_large(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 60)
    (tmp_path / "b").write_bytes(b"x" * 60)

    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path, max_total_bytes=100)

    assert caught.value.error_class == "output_too_large"


def test_a_name_the_bundle_owns_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")

    with pytest.raises(OutputRejected, match="the bundle owns"):
        inventory(tmp_path, max_total_bytes=100)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path / "nope", max_total_bytes=100)

    assert caught.value.error_class == "unsafe_output"


# --- the farm inventory -------------------------------------------------------------------------------------------


def _farm(tmp_path, entries):
    import sys

    from hyperloom.measure.farm import build_farm
    from hyperloom.measure.spec import EnvRoot

    blobs = {"sha256:" + hashlib.sha256(d).hexdigest(): d for d in (b"x = 1\n",)}
    root = EnvRoot("yaml", "package", "yaml", "PyYAML", "0", tuple(entries))
    return build_farm([root], tmp_path / "farm", python=sys.executable, read_blob=blobs.__getitem__)


def test_the_inventory_lists_each_upserted_files_hash_as_it_is_in_the_farm_and_the_deletes(tmp_path):
    from hyperloom.measure.runner.inventory import farm_inventory
    from hyperloom.measure.spec import EnvEntry

    address = "sha256:" + hashlib.sha256(b"x = 1\n").hexdigest()
    farm = _farm(tmp_path, [EnvEntry("pkg/new.py", "upsert", address, "0644"), EnvEntry("error.py", "delete")])

    document = farm_inventory(farm)

    assert document["roots"] == {
        "yaml": {
            "import_name": "yaml",
            "upserts": {"pkg/new.py": hashlib.sha256(b"x = 1\n").hexdigest()},
            "deletes": ["error.py"],
        }
    }


def test_a_file_changed_in_the_farm_after_the_build_shows_its_new_hash(tmp_path):
    from hyperloom.measure.runner.inventory import farm_inventory
    from hyperloom.measure.spec import EnvEntry

    address = "sha256:" + hashlib.sha256(b"x = 1\n").hexdigest()
    farm = _farm(tmp_path, [EnvEntry("new.py", "upsert", address, "0644")])
    (farm.site / "yaml" / "new.py").write_bytes(b"x = 2\n")

    assert farm_inventory(farm)["roots"]["yaml"]["upserts"]["new.py"] == hashlib.sha256(b"x = 2\n").hexdigest()


def test_without_a_farm_the_inventory_has_no_roots():
    from hyperloom.measure.runner.inventory import farm_inventory

    assert farm_inventory(None)["roots"] == {}


def test_the_file_is_written_into_the_output_and_listed_by_the_inventory(tmp_path):
    from hyperloom.measure.runner.inventory import FARM_INVENTORY_FILE, write_farm_inventory

    write_farm_inventory(tmp_path / "files", None)

    listed = inventory(tmp_path / "files", max_total_bytes=10_000)
    assert [f.path for f in listed] == [FARM_INVENTORY_FILE]


def test_the_installation_is_removed_below_the_output_and_nowhere_else(tmp_path):
    outside = tmp_path / "outside"
    (outside / ".agentx-runtime").mkdir(parents=True)
    (outside / ".agentx-runtime" / "x").write_text("x")
    (tmp_path / "out" / "w" / ".agentx-runtime").mkdir(parents=True)
    (tmp_path / "out" / "w" / ".agentx-runtime" / "x").write_text("x")
    (tmp_path / "out" / "w" / "report.json").write_text("{}")
    (tmp_path / "out" / "link").symlink_to(outside)

    drop_agentx_runtime(tmp_path / "out")

    assert not (tmp_path / "out" / "w" / ".agentx-runtime").exists()
    assert (tmp_path / "out" / "w" / "report.json").is_file()
    assert (outside / ".agentx-runtime" / "x").is_file()


def test_an_output_directory_that_is_a_link_is_not_walked(tmp_path):
    outside = tmp_path / "outside"
    (outside / ".agentx-runtime").mkdir(parents=True)
    (tmp_path / "out").symlink_to(outside)

    drop_agentx_runtime(tmp_path / "out")

    assert (outside / ".agentx-runtime").is_dir()
    with pytest.raises(OutputRejected) as caught:
        inventory(tmp_path / "out", max_total_bytes=100)
    assert caught.value.error_class == "unsafe_output"
