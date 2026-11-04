# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import COMPONENT_LAYOUTS, ComponentLayout, EnvironmentCompiler, LocalBlobStore
from hyperloom.measure.framework_python import locator_for_python
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with, write_tree

VLLM = COMPONENT_LAYOUTS["vllm"]
ORIGINAL = "def serve():\n    return 1\n"
PATCHED = "def serve():\n    return 2\n"


@pytest.fixture(scope="module")
def venv_python(tmp_path_factory) -> tuple[str, Path]:
    """A real second interpreter with a fake ``vllm`` installed in its site-packages, as a framework venv has."""
    root = tmp_path_factory.mktemp("venv-vllm")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, capture_output=True)
    (site,) = (root / "lib").glob("python3*/site-packages")
    write_tree(site, {"vllm/__init__.py": "", "vllm/entry.py": ORIGINAL})
    write_tree(
        site, {"vllm-0.20.1+rocm721.dist-info/METADATA": "Metadata-Version: 2.1\nName: vllm\nVersion: 0.20.1+rocm721\n"}
    )
    (site / "vllm-0.20.1+rocm721.dist-info/top_level.txt").write_text("vllm\n")
    (site / "vllm-0.20.1+rocm721.dist-info/RECORD").write_text("")
    return str(root / "bin" / "python"), site


def test_finds_the_package_in_the_other_interpreters_site_packages(venv_python):
    python, site = venv_python

    base = locator_for_python(python)(VLLM)

    assert (base.import_name, base.path, base.dist_name, base.dist_version) == (
        "vllm",
        (site / "vllm").resolve(),
        "vllm",
        "0.20.1+rocm721",
    )


def test_a_package_the_other_interpreter_lacks_is_refused_even_if_this_one_has_it(venv_python):
    python, _ = venv_python
    here = ComponentLayout("pytest", "pytest")  # importable in this process, not in the venv

    with pytest.raises(Exception) as caught:
        locator_for_python(python)(here)

    assert type(caught.value).__name__ == "UnsupportedComponentError"
    assert "pytest is not installed for" in str(caught.value)


def test_a_tree_outside_site_packages_is_not_a_base(tmp_path, venv_python):
    python, site = venv_python
    source = write_tree(tmp_path / "checkout", {"srcpkg/__init__.py": ""})
    (site / "srcpkg.pth").write_text(f"{source}\n")
    try:
        with pytest.raises(Exception) as caught:
            locator_for_python(python)(ComponentLayout("srcpkg", "srcpkg"))
    finally:
        (site / "srcpkg.pth").unlink()

    assert type(caught.value).__name__ == "UnsupportedComponentError"
    assert "not installed in site-packages" in str(caught.value)


def test_a_missing_interpreter_is_an_unsupported_component(tmp_path):
    with pytest.raises(Exception) as caught:
        locator_for_python(str(tmp_path / "no-such-python"))(VLLM)

    assert type(caught.value).__name__ == "UnsupportedComponentError"
    assert "cannot ask" in str(caught.value)


def test_a_vllm_patch_compiles_against_the_venv_copy_and_leaves_it_untouched(tmp_path, venv_python):
    python, site = venv_python
    blobs = LocalBlobStore(tmp_path / "blobs")
    patch = git_patch(tmp_path / "repo", {"vllm/entry.py": ORIGINAL}, {"vllm/entry.py": PATCHED})
    recipe = recipe_with((put_patch(blobs, "v.patch", patch),), component="vllm")

    env = EnvironmentCompiler(blobs, IMAGE, locator_for_python(python)).compile(recipe, RecipeDelta())

    (root,) = env.roots
    assert (root.id, root.import_name, root.dist_version) == ("vllm", "vllm", "0.20.1+rocm721")
    (entry,) = root.entries
    assert (entry.path, entry.op) == ("entry.py", "upsert")
    assert blobs.get(entry.blob) == PATCHED.encode()
    assert (site / "vllm/entry.py").read_text() == ORIGINAL
