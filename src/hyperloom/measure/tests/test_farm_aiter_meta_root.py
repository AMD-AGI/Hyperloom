# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch to aiter's ``csrc`` or ``hsa`` is a root of its own in the farm: a mirror of the image's meta directory with the
patched files laid over it, which the server is told to read (``AITER_META_DIR``), beside a package root or without one.

The image is the real second interpreter of ``test_farm_aiter_meta``, laid as a wheel and as an editable checkout."""

from __future__ import annotations

import subprocess

import pytest
import yaml

from hyperloom.measure.farm import Farm, FarmError, build_farm, place_farm, verify_farm
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.measure.tests._recipes import write_tree
from hyperloom.measure.tests.test_farm_aiter_meta import (
    LAYOUTS,
    PATCHED,
    Image,
    _address,
    _aiter_root,
    _make_image,
    _point,
)

PATCHED_PA = b"PA = 'patched pa'\n"
PA_PATH = "csrc/cpp_itfs/pa/pa.py"


@pytest.fixture(autouse=True)
def _no_inherited_meta_directory(monkeypatch) -> None:
    monkeypatch.delenv("AITER_META_DIR", raising=False)


@pytest.fixture(params=LAYOUTS)
def image(request, tmp_path) -> Image:
    made = _make_image(tmp_path / "image", request.param)
    write_tree(made.meta, {"csrc/kernels/k.cu": "// image\n", "hsa/gfx942/pa/pa.csv": "a,b\n", "hsa/codegen.py": "#\n"})
    return made


def _meta_root(path: str = PA_PATH, data: bytes = PATCHED_PA) -> EnvRoot:
    return EnvRoot(
        "aiter.meta", "package", "aiter_meta", "amd-aiter", "0", (EnvEntry(path, "upsert", _address(data), "0644"),)
    )


def _blobs(address: str) -> bytes:
    return {_address(PATCHED_PA): PATCHED_PA, _address(PATCHED): PATCHED}[address]


def _build(tmp_path, image: Image, *roots: EnvRoot, **kwargs) -> Farm:
    farm = build_farm(list(roots), tmp_path / "farm", python=image.python, read_blob=_blobs, **kwargs)
    assert farm is not None
    return farm


def _server_reads(tmp_path, image: Image, envs: dict, statement: str) -> str:
    """What the server's interpreter prints after importing aiter and csrc with the variables the YAML gave it."""
    environment = {"PATH": "", **{name: str(envs[name]) for name in ("PYTHONPATH", "AITER_META_DIR") if name in envs}}
    done = subprocess.run(
        [image.python, "-c", f"import aiter.ops.attention as a\nimport csrc.cpp_itfs.pa.pa as p\n{statement}"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    assert done.returncode == 0, done.stderr[-500:]
    return done.stdout.strip()


def test_a_meta_root_alone_is_a_farm_whose_meta_directory_is_the_mirror_and_no_package_is_rebuilt(tmp_path, image):
    farm = _build(tmp_path, image, _meta_root())

    assert (farm.packages, farm.meta_patched, farm.aiter_meta) == ((), True, farm.site / "aiter_meta")
    assert [root.id for root in farm.roots] == ["aiter.meta"]


def test_the_mirror_holds_the_patched_file_and_links_what_the_patch_leaves_and_never_the_images_aiter(tmp_path, image):
    farm = _build(tmp_path, image, _meta_root())

    mirror = farm.aiter_meta
    assert (mirror / PA_PATH).read_bytes() == PATCHED_PA and not (mirror / PA_PATH).is_symlink()
    assert (mirror / "csrc/kernels").is_symlink() and (mirror / "hsa").is_symlink()
    assert (image.meta / PA_PATH).read_text() == "PA = 'pa'\n"
    assert not (mirror / "aiter").exists()


def test_the_server_reads_the_patched_csrc_from_the_mirror_and_the_image_is_left_as_it_was(tmp_path, image):
    farm = _build(tmp_path, image, _meta_root())
    envs = place_farm(_point(tmp_path, image), farm)

    assert envs["AITER_META_DIR"] == str(farm.aiter_meta)
    verify_farm(farm, envs, python=image.python, cwd=tmp_path)
    assert _server_reads(tmp_path, image, envs, "print(p.PA)") == "patched pa"
    assert (image.meta / PA_PATH).read_text() == "PA = 'pa'\n"


def test_with_a_package_root_too_the_farms_aiter_is_still_the_one_imported_and_csrc_is_the_patched_one(tmp_path, image):
    farm = _build(tmp_path, image, _aiter_root(), _meta_root())
    envs = place_farm(_point(tmp_path, image), farm)

    verify_farm(farm, envs, python=image.python, cwd=tmp_path)
    where = _server_reads(tmp_path, image, envs, "print(a.PATCH, p.PA, a.__file__)")
    assert where.startswith("patched patched pa ") and where.endswith(str(farm.site / "aiter/ops/attention.py"))
    assert farm.packages == ("aiter",) and [root.id for root in farm.roots] == ["aiter", "aiter.meta"]


def test_a_server_that_inherits_a_meta_directory_is_refused_because_it_would_read_other_sources(tmp_path, image):
    with pytest.raises(FarmError, match="inherits AITER_META_DIR='/elsewhere'"):
        _build(tmp_path, image, _meta_root(), inherited={"AITER_META_DIR": "/elsewhere"})


def test_a_yaml_that_names_another_meta_directory_fails_the_check_and_the_farms_own_passes_it(tmp_path, image):
    farm = _build(tmp_path, image, _meta_root())
    envs = place_farm(_point(tmp_path, image, AITER_META_DIR=str(image.meta)), farm)

    with pytest.raises(FarmError, match="would build the image's sources"):
        verify_farm(farm, envs, python=image.python, cwd=tmp_path)
    verify_farm(farm, {**envs, "AITER_META_DIR": str(farm.aiter_meta)}, python=image.python, cwd=tmp_path)


def test_an_entry_below_aiter_is_refused_because_it_would_shadow_the_farms_package(tmp_path, image):
    with pytest.raises(FarmError, match="a second aiter package ahead of the farm's"):
        _build(tmp_path, image, _meta_root("aiter/ops/attention.py"))


def test_an_image_whose_meta_directory_has_no_csrc_is_refused_naming_it(tmp_path, image):
    (image.meta / "csrc" / "cpp_itfs" / "pa" / "pa.py").unlink()
    for directory in sorted((image.meta / "csrc").rglob("*"), reverse=True):
        directory.unlink() if directory.is_file() else directory.rmdir()
    (image.meta / "csrc").rmdir()

    with pytest.raises(FarmError, match="keeps no csrc directory"):
        _build(tmp_path, image, _meta_root())


def test_the_yaml_the_farm_writes_is_unchanged_but_for_the_variable(tmp_path, image):
    farm = _build(tmp_path, image, _meta_root())
    point = _point(tmp_path, image)

    place_farm(point, farm)

    envs = yaml.safe_load(point.read_text())["benchmark"]["envs"]
    assert envs["AITER_META_DIR"] == str(farm.aiter_meta) and envs["PYTHONPATH"] == str(farm.site)
