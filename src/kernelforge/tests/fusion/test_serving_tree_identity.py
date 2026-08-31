# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The smoke must exercise the tree the loop patched, or say that it did not.

The server imports the installed package. Point --framework-root anywhere else
and it boots stock code with the fusion flag set, comes up clean, and the loop
records SERVING SMOKE OK for a kernel that was never loaded -- a pass certifying
exactly the thing the smoke exists to check.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.fusion.validate import framework_tree_is_the_imported_one


def _tree(tmp_path: Path, name: str, pkg: str = "vllm") -> Path:
    root = tmp_path / name
    (root / pkg).mkdir(parents=True)
    return root


def test_a_copy_of_the_tree_is_not_the_installed_one(tmp_path: Path) -> None:
    patched = _tree(tmp_path, "fwroot")
    installed = _tree(tmp_path, "site-packages")

    ok, reason = framework_tree_is_the_imported_one(str(patched), "vllm", _finder=lambda pkg: str(installed / pkg))

    assert ok is False
    assert "without ever loading the kernel" in reason


def test_the_installed_tree_passes(tmp_path: Path) -> None:
    root = _tree(tmp_path, "site-packages")

    ok, reason = framework_tree_is_the_imported_one(str(root), "vllm", _finder=lambda pkg: str(root / pkg))

    assert (ok, reason) == (True, "")


def test_a_symlink_to_the_install_is_the_install(tmp_path: Path) -> None:
    real = _tree(tmp_path, "site-packages")
    link = tmp_path / "linked"
    link.mkdir()
    (link / "vllm").symlink_to(real / "vllm", target_is_directory=True)

    ok, _ = framework_tree_is_the_imported_one(str(link), "vllm", _finder=lambda pkg: str(real / pkg))

    assert ok is True


def test_sglang_is_checked_against_its_own_package(tmp_path: Path) -> None:
    patched = _tree(tmp_path, "fwroot", pkg="sglang")
    installed = _tree(tmp_path, "site-packages", pkg="sglang")

    ok, reason = framework_tree_is_the_imported_one(str(patched), "sglang", _finder=lambda pkg: str(installed / pkg))

    assert ok is False
    assert "sglang" in reason


def test_an_unknown_root_is_not_second_guessed() -> None:
    assert framework_tree_is_the_imported_one("", "vllm") == (True, "")


def test_an_unresolvable_package_is_not_second_guessed(tmp_path: Path) -> None:
    patched = _tree(tmp_path, "fwroot")

    assert framework_tree_is_the_imported_one(str(patched), "vllm", _finder=lambda pkg: "") == (True, "")
