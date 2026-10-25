# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The parity build against the image's binary, with symbols and imports faked (no ``nm`` or ROCm needed)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.measure.runner.native import parity
from hyperloom.measure.runner.native.parity import exported_symbols, import_problem, parity_build
from hyperloom.measure.runner.native.tests._native import OUTPUT, Bench, upstream_tree


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    return Bench(tmp_path)


def _same_symbols(path: Path) -> frozenset[str]:
    return frozenset({"kernel"})


def _imports(python: str, parent: Path, name: str) -> str | None:
    return None


def _parity(bench: Bench, *, symbols=_same_symbols, importer=_imports):
    return parity_build(bench.request(), bench.blobs, bench.tmp / "parity", symbols=symbols, importer=importer)


def test_a_build_that_reproduces_the_image_passes(bench: Bench) -> None:
    report = _parity(bench)

    assert (report.ok, report.differences) == (True, ())


def test_a_missing_exported_symbol_is_a_difference(bench: Bench) -> None:
    def symbols(path: Path) -> frozenset[str]:
        return frozenset({"kernel", "image_only"}) if "image" in path.parts else frozenset({"kernel"})

    report = _parity(bench, symbols=symbols)

    assert report.differences == (
        f"{OUTPUT}: 1 exported symbols missing (for example image_only), 0 extra (for example -)",
    )


def test_an_image_binary_the_build_does_not_make_is_a_difference(bench: Bench) -> None:
    (bench.image / "fakepkg/other.so").write_bytes(b"\x7fELF")

    assert _parity(bench).differences == ("fakepkg/other.so: the image has it, the build did not make it",)


def test_a_built_binary_the_image_lacks_is_a_difference(bench: Bench) -> None:
    (bench.image / OUTPUT).unlink()

    assert _parity(bench).differences == (f"{OUTPUT}: built, but the image has no such file",)


def test_a_package_that_does_not_import_is_a_difference(bench: Bench) -> None:
    report = _parity(bench, importer=lambda python, parent, name: "ImportError: undefined symbol: x")

    assert report.differences == ("the built package does not import: ImportError: undefined symbol: x",)


def test_an_unpatched_tree_that_does_not_build_fails_parity(tmp_path: Path) -> None:
    bench = Bench(tmp_path)
    bench.upstream = upstream_tree(tmp_path / "broken", source="#error\n")

    report = _parity(bench)

    assert report.differences == ("the unpatched tree does not build: compile_error: the compile exited 1",)


def test_the_import_probe_names_why_a_package_does_not_import(tmp_path: Path) -> None:
    (tmp_path / "good").mkdir()
    (tmp_path / "good/__init__.py").write_text("")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad/__init__.py").write_text("raise ImportError('no symbol')\n")

    assert import_problem(sys.executable, tmp_path, "good") is None
    assert import_problem(sys.executable, tmp_path, "bad") == "ImportError: no symbol"


def test_hip_compilation_unit_ids_are_not_compared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    listing = (
        "__hip_cuid_1391fe7dff99c08a B 00000000011f4a10 0000000000000001\n"
        "_Z10topk_fusedv T 0000000000102340 00000000000001a0\n"
        "my__hip_cuid_ff B 00000000011f4a11 0000000000000001\n"
    )

    def nm(argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")

    monkeypatch.setattr(parity.subprocess, "run", nm)

    assert exported_symbols(tmp_path / "common_ops.so") == frozenset({"_Z10topk_fusedv", "my__hip_cuid_ff"})
