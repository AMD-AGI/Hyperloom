# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A farm that lays a patched aiter is told where the image's aiter keeps its sources, so the server can import it.

The image is a real second interpreter with a stand-in aiter that finds its meta directory the way aiter's
``jit/core.py`` does: from the lexical location of the file, with ``AITER_META_DIR`` as the override, and it imports
``csrc.cpp_itfs.pa.pa`` from there (what ``aiter/ops/attention.py`` does). Both ways the image carries it are laid:
a wheel (``aiter_meta`` beside the package) and an editable checkout (``csrc`` beside the package, found through a
finder that maps only ``aiter``, as an editable install does).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.farm import (
    Farm,
    FarmError,
    build_farm,
    image_aiter_meta,
    inherited_environment,
    place_farm,
    verify_farm,
)
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.measure.tests._recipes import write_tree

CORE = """\
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
try:
    with open(f"{this_dir}/../install_mode") as handle:
        develop = handle.read().strip() == "develop"
except FileNotFoundError:
    develop = True
AITER_META_DIR = AITER_ROOT_DIR if develop else os.path.abspath(f"{AITER_ROOT_DIR}/aiter_meta/")
env_meta = os.environ.get("AITER_META_DIR")
if env_meta:
    AITER_META_DIR = os.path.abspath(env_meta)
if not os.path.exists(os.path.join(AITER_META_DIR, "csrc")):
    AITER_META_DIR = AITER_ROOT_DIR
sys.path.insert(0, AITER_META_DIR)
"""
ATTENTION = "from csrc.cpp_itfs.pa.pa import PA\nPATCH = {patch!r}\n"
PA = "PA = 'pa'\n"
EDITABLE_FINDER = """\
import sys
from importlib.machinery import PathFinder


class _Finder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        return PathFinder.find_spec(name, [{root!r}]) if name == "aiter" else None


sys.meta_path.append(_Finder)
"""
PACKAGE_FILES = {
    "aiter/__init__.py": "from aiter.jit import core  # noqa\nfrom aiter.ops import attention  # noqa\n",
    "aiter/jit/__init__.py": "",
    "aiter/jit/core.py": CORE,
    "aiter/ops/__init__.py": "",
    "aiter/ops/attention.py": ATTENTION.format(patch="image"),
}
PATCHED = ATTENTION.format(patch="patched").encode()
LAYOUTS = ("wheel", "checkout")


@dataclass(frozen=True)
class Image:
    """An interpreter with an aiter installed, and the directory its aiter reads ``csrc`` from."""

    python: str
    meta: Path


def _address(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _make_image(root: Path, layout: str) -> Image:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / "venv")], check=True, capture_output=True)
    site = next((root / "venv" / "lib").glob("python*/site-packages"))
    if layout == "wheel":
        write_tree(site, {**PACKAGE_FILES, "aiter/install_mode": "install", "aiter_meta/__init__.py": ""})
        meta = site / "aiter_meta"
    else:
        meta = root / "checkout"
        write_tree(meta, {**PACKAGE_FILES, "aiter/install_mode": "develop"})
        write_tree(
            site,
            {
                "__editable_aiter.py": EDITABLE_FINDER.format(root=str(meta)),
                "editable_aiter.pth": "import __editable_aiter\n",
            },
        )
    write_tree(meta, {"csrc/cpp_itfs/pa/pa.py": PA})
    return Image(str(root / "venv" / "bin" / "python"), meta)


@pytest.fixture(autouse=True)
def _no_inherited_meta_directory(monkeypatch) -> None:
    monkeypatch.delenv("AITER_META_DIR", raising=False)


@pytest.fixture(params=LAYOUTS)
def image(request, tmp_path) -> Image:
    return _make_image(tmp_path / "image", request.param)


def _aiter_root() -> EnvRoot:
    entry = EnvEntry("ops/attention.py", "upsert", _address(PATCHED), "0644")
    return EnvRoot("aiter", "package", "aiter", "amd-aiter", "0", (entry,))


def _build(tmp_path: Path, image: Image, **kwargs) -> Farm:
    farm = build_farm(
        [_aiter_root()], tmp_path / "farm", python=image.python, read_blob=lambda address: PATCHED, **kwargs
    )
    assert farm is not None
    return farm


def _point(tmp_path: Path, image: Image, **envs: str) -> Path:
    path = tmp_path / "point.yaml"
    document = {"benchmark": {"framework": "sglang", "envs": {"HYPERLOOM_FRAMEWORK_PYTHON": image.python, **envs}}}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _import_aiter(tmp_path: Path, image: Image, envs: dict) -> subprocess.CompletedProcess[str]:
    """What the server's interpreter does at start: import aiter with the variables the YAML gave it."""
    neutral = tmp_path / "cwd"
    neutral.mkdir(exist_ok=True)
    environment = {"PYTHONPATH": str(envs["PYTHONPATH"]), "PATH": ""}
    if "AITER_META_DIR" in envs:
        environment["AITER_META_DIR"] = str(envs["AITER_META_DIR"])
    return subprocess.run(
        [image.python, "-c", "import aiter.ops.attention as a; print(a.PATCH)"],
        capture_output=True,
        text=True,
        cwd=neutral,
        env=environment,
    )


def test_the_farm_finds_the_image_meta_directory_where_aiter_keeps_it(tmp_path, image):
    assert _build(tmp_path, image).aiter_meta == image.meta


def test_a_patched_aiter_in_the_farm_imports_because_it_is_told_the_image_meta_directory(tmp_path, image):
    farm = _build(tmp_path, image)
    point = _point(tmp_path, image)

    envs = place_farm(point, farm)

    assert envs["AITER_META_DIR"] == str(image.meta)
    assert yaml.safe_load(point.read_text())["benchmark"]["envs"]["AITER_META_DIR"] == str(image.meta)
    verify_farm(farm, envs, python=image.python, cwd=tmp_path)
    done = _import_aiter(tmp_path, image, envs)
    assert done.stdout.strip() == "patched", done.stderr[-400:]


def test_the_same_farm_without_the_variable_does_not_import_which_is_the_defect(tmp_path, image):
    farm = _build(tmp_path, image)
    envs = dict(place_farm(_point(tmp_path, image), farm))
    del envs["AITER_META_DIR"]

    done = _import_aiter(tmp_path, image, envs)

    assert done.returncode != 0 and "No module named 'csrc'" in done.stderr


def test_the_unpatched_image_imports_without_the_variable_so_the_stand_in_is_faithful(tmp_path, image):
    neutral = tmp_path / "cwd"
    neutral.mkdir()

    done = subprocess.run(
        [image.python, "-c", "import aiter.ops.attention as a; print(a.PATCH)"],
        capture_output=True,
        text=True,
        cwd=neutral,
        env={"PATH": ""},
    )

    assert done.stdout.strip() == "image", done.stderr[-400:]


def test_a_variable_the_yaml_already_names_is_not_overwritten(tmp_path, image):
    farm = _build(tmp_path, image)

    envs = place_farm(_point(tmp_path, image, AITER_META_DIR=str(tmp_path)), farm)

    assert envs["AITER_META_DIR"] == str(tmp_path)


def test_a_develop_install_keeps_its_meta_directory_around_the_package_and_a_wheel_in_aiter_meta(tmp_path):
    package = tmp_path / "site" / "aiter"
    package.mkdir(parents=True)

    (package / "install_mode").write_text("install\n")
    assert image_aiter_meta(package) == tmp_path / "site" / "aiter_meta"
    (package / "install_mode").write_text("develop\n")
    assert image_aiter_meta(package) == tmp_path / "site"
    (package / "install_mode").unlink()
    assert image_aiter_meta(package) == tmp_path / "site", (
        "no install_mode file is a develop install, as aiter reads it"
    )


def test_the_environment_a_server_inherits_is_the_runners_with_the_tests_process_variables_minus_what_it_unsets():
    inherited = inherited_environment({"A": "1", "B": "2", "C": "3"}, {"B": "20", "D": "4"}, ["C", "D"])

    assert inherited == {"A": "1", "B": "20"}


def test_an_image_whose_aiter_has_no_sources_directory_refuses_a_patched_aiter_rather_than_boot_it_broken(tmp_path):
    image = _make_image(tmp_path / "image", "wheel")
    for path in sorted((image.meta / "csrc").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (image.meta / "csrc").rmdir()

    with pytest.raises(FarmError, match=f"keeps no csrc directory in {image.meta}"):
        _build(tmp_path, image)


def test_a_wheels_aiter_meta_is_not_used_by_a_develop_install_which_aiter_does_not_either(tmp_path):
    image = _make_image(tmp_path / "image", "checkout")
    (image.meta / "csrc").rename(image.meta.parent / "csrc-moved")
    (image.meta / "aiter_meta" / "csrc").mkdir(parents=True)

    with pytest.raises(FarmError, match=f"keeps no csrc directory in {image.meta}"):
        _build(tmp_path, image)


def test_a_rebuilt_aiter_that_is_not_told_a_meta_directory_with_csrc_is_refused(tmp_path, image):
    farm = _build(tmp_path, image)
    envs = dict(place_farm(_point(tmp_path, image), farm))
    del envs["AITER_META_DIR"]

    with pytest.raises(FarmError, match="AITER_META_DIR=''.*no csrc"):
        verify_farm(farm, envs, python=image.python, cwd=tmp_path)
    envs["AITER_META_DIR"] = str(tmp_path)
    with pytest.raises(FarmError, match=f"AITER_META_DIR='{tmp_path}'.*no csrc"):
        verify_farm(farm, envs, python=image.python, cwd=tmp_path)


def test_a_meta_directory_the_server_inherits_is_kept_so_nothing_is_pinned_and_it_still_imports(tmp_path, image):
    inherited = {"AITER_META_DIR": str(image.meta)}
    farm = _build(tmp_path, image, inherited=inherited)
    point = _point(tmp_path, image)

    envs = place_farm(point, farm)
    verify_farm(farm, envs, python=image.python, cwd=tmp_path)

    assert farm.aiter_meta is None and "AITER_META_DIR" not in envs
    done = _import_aiter(tmp_path, image, {**envs, **inherited})
    assert done.stdout.strip() == "patched", done.stderr[-400:]


def test_an_empty_or_unset_inherited_variable_is_no_meta_directory_so_the_image_s_is_pinned(tmp_path, image):
    unset = inherited_environment({"AITER_META_DIR": "/x"}, {}, ["AITER_META_DIR"])
    for index, inherited in enumerate(({}, {"AITER_META_DIR": ""}, unset)):
        (tmp_path / str(index)).mkdir()
        assert _build(tmp_path / str(index), image, inherited=inherited).aiter_meta == image.meta


def test_without_an_inherited_environment_given_this_processs_own_is_the_one_asked(tmp_path, image, monkeypatch):
    for name in ("set", "unset"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("AITER_META_DIR", str(tmp_path))

    assert _build(tmp_path / "set", image).aiter_meta is None
    monkeypatch.delenv("AITER_META_DIR")
    assert _build(tmp_path / "unset", image).aiter_meta == image.meta
