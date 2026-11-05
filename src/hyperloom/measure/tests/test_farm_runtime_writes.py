# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the server writes into a rebuilt aiter stays in the test's farm: a module its JIT builds, a build directory, a
FlyDSL lock and a tuned table aiter rewrites. The image is a second interpreter with a stand-in aiter that writes each
the way the real one does (``atomic_copy`` renames a temporary file over ``<module>.so``; ``build`` and the table are
opened in place; FlyDSL opens ``<key>.lock`` read-write), from where it was imported, as ``jit/core.py`` does. Before the
fix the farm linked ``jit`` and ``configs`` whole into the image, so one test's build replaced the module every later
test on the node loaded.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.measure.farm import Farm, build_farm
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.measure.tests._recipes import write_tree

CORE = """\
import os
import sys
import tempfile

this_dir = os.path.dirname(os.path.abspath(__file__))


def build(name, data):
    os.makedirs(f"{this_dir}/build/{name}", exist_ok=True)
    with open(f"{this_dir}/build/{name}/{name}.o", "wb") as handle:
        handle.write(data)
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.so.", suffix=".tmp", dir=this_dir)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(temporary, f"{this_dir}/{name}.so")


def rewrite_table(name, text):
    with open(f"{this_dir}/../configs/{name}", "w") as handle:
        handle.write(text)


def lock(key):
    fd = os.open(f"{this_dir}/flydsl_cache/kernel/{key}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    os.write(fd, b"held")
    os.close(fd)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"build": lambda n, d: build(n, d.encode()), "table": rewrite_table, "lock": lock}[command](*args)
"""
IMAGE_MODULE = b"\x7fELF built for the image's tables"
IMAGE_TABLE = "kernelName\nimage_kernel\n"
IMAGE_LOCK = ""
IMAGE_FILES = {
    "aiter/__init__.py": "",
    "aiter/install_mode": "install",
    "aiter/jit/__init__.py": "",
    "aiter/jit/core.py": CORE,
    "aiter/jit/build/module_gemm/module_gemm.o": "image object",
    "aiter/jit/flydsl_cache/kernel/k1.pkl": "image kernel",
    "aiter/jit/flydsl_cache/kernel/k1.lock": IMAGE_LOCK,
    "aiter/configs/gemm_tuned.csv": IMAGE_TABLE,
    "aiter/configs/model_configs/model_tuned.csv": IMAGE_TABLE,
    "aiter/ops/__init__.py": "",
    "aiter/ops/gemm.py": "PATCH = 'image'\n",
    "aiter_meta/__init__.py": "",
    "aiter_meta/csrc/gemm.cu": "// image source\n",
}
PATCHED = b"PATCH = 'candidate'\n"


@pytest.fixture(autouse=True)
def _no_inherited_meta_directory(monkeypatch) -> None:
    monkeypatch.delenv("AITER_META_DIR", raising=False)


@pytest.fixture
def image(tmp_path) -> tuple[str, Path]:
    """An interpreter with the stand-in aiter installed, and the installed package's directory."""
    venv = tmp_path / "image"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    write_tree(site, IMAGE_FILES)
    (site / "aiter" / "jit" / "module_gemm.so").write_bytes(IMAGE_MODULE)
    return str(venv / "bin" / "python"), site / "aiter"


def _farm(tmp_path: Path, python: str, name: str) -> Farm:
    entry = EnvEntry("ops/gemm.py", "upsert", "sha256:" + hashlib.sha256(PATCHED).hexdigest(), "0644")
    root = EnvRoot("aiter", "package", "aiter", "amd-aiter", "0", (entry,))
    farm = build_farm([root], tmp_path / name, python=python, read_blob=lambda address: PATCHED)
    assert farm is not None
    return farm


def _server(python: str, farm: Farm, *args: str) -> None:
    """What the server's aiter does: run from the farm's copy of ``jit/core.py``, which is where it was imported from."""
    core = farm.site / "aiter" / "jit" / "core.py"
    done = subprocess.run([python, str(core), *args], capture_output=True, text=True, cwd=farm.site.parent)
    assert done.returncode == 0, done.stderr


def test_a_module_a_test_builds_replaces_its_farms_link_and_the_image_and_the_next_farm_keep_the_original(
    tmp_path, image
) -> None:
    python, installed = image
    first = _farm(tmp_path, python, "one")
    linked = first.site / "aiter" / "jit" / "module_gemm.so"
    assert linked.is_symlink() and os.path.realpath(linked) == str((installed / "jit" / "module_gemm.so").resolve())

    _server(python, first, "build", "module_gemm", "built for this test's table")

    assert linked.read_bytes() == b"built for this test's table" and not linked.is_symlink()
    assert (installed / "jit" / "module_gemm.so").read_bytes() == IMAGE_MODULE
    assert (installed / "jit" / "build" / "module_gemm" / "module_gemm.o").read_text() == "image object"
    assert not [path.name for path in (installed / "jit").iterdir() if path.name.endswith(".tmp")]
    second = _farm(tmp_path, python, "two")
    assert (second.site / "aiter" / "jit" / "module_gemm.so").read_bytes() == IMAGE_MODULE


def test_a_test_builds_in_a_build_directory_of_its_own_that_starts_empty(tmp_path, image) -> None:
    python, installed = image
    farm = _farm(tmp_path, python, "one")
    build = farm.site / "aiter" / "jit" / "build"

    assert build.is_dir() and not build.is_symlink() and not list(build.iterdir())
    _server(python, farm, "build", "module_new", "new")
    assert not (installed / "jit" / "build" / "module_new").exists()


def test_a_tuned_table_aiter_rewrites_in_place_is_rewritten_in_the_farm(tmp_path, image) -> None:
    python, installed = image
    farm = _farm(tmp_path, python, "one")
    configs = farm.site / "aiter" / "configs"

    _server(python, farm, "table", "gemm_tuned.csv", "kernelName\ndeduplicated\n")
    _server(python, farm, "table", "model_configs/model_tuned.csv", "kernelName\ndeduplicated\n")

    assert (configs / "gemm_tuned.csv").read_text() == "kernelName\ndeduplicated\n"
    assert (installed / "configs" / "gemm_tuned.csv").read_text() == IMAGE_TABLE
    assert (installed / "configs" / "model_configs" / "model_tuned.csv").read_text() == IMAGE_TABLE
    assert not any(path.is_symlink() for path in configs.rglob("*"))


def test_flydsl_finds_the_images_kernels_but_takes_its_locks_in_the_farm(tmp_path, image) -> None:
    python, installed = image
    farm = _farm(tmp_path, python, "one")
    kernel = farm.site / "aiter" / "jit" / "flydsl_cache" / "kernel"

    _server(python, farm, "lock", "k1")

    assert (kernel / "k1.pkl").is_symlink() and (kernel / "k1.pkl").read_text() == "image kernel"
    assert (kernel / "k1.lock").read_bytes() == b"held" and not (kernel / "k1.lock").is_symlink()
    assert (installed / "jit" / "flydsl_cache" / "kernel" / "k1.lock").read_text() == IMAGE_LOCK


def test_code_of_the_package_the_server_does_not_write_stays_one_link(tmp_path, image) -> None:
    python, installed = image
    farm = _farm(tmp_path, python, "one")

    assert (farm.site / "aiter" / "jit" / "core.py").is_symlink()
    assert (farm.site / "aiter" / "ops" / "gemm.py").read_bytes() == PATCHED
    assert (farm.site / "aiter" / "ops" / "__init__.py").is_symlink()


def test_a_table_the_candidate_patches_is_its_own_beside_copies_of_the_others(tmp_path, image) -> None:
    python, _ = image
    table = b"kernelName\ncandidate_kernel\n"
    entry = EnvEntry("configs/gemm_tuned.csv", "upsert", "sha256:" + hashlib.sha256(table).hexdigest(), "0644")
    root = EnvRoot("aiter", "package", "aiter", "amd-aiter", "0", (entry,))

    farm = build_farm([root], tmp_path / "farm", python=python, read_blob=lambda address: table)

    assert farm is not None
    assert (farm.site / "aiter" / "configs" / "gemm_tuned.csv").read_bytes() == table
    copied = farm.site / "aiter" / "configs" / "model_configs" / "model_tuned.csv"
    assert copied.read_text() == IMAGE_TABLE and not copied.is_symlink()
