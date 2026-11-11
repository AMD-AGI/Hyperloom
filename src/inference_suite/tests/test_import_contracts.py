# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``.importlinter`` contracts that bound ``inference_suite``, run against a synthetic source tree."""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
CONFIG = SRC.parent / ".importlinter"
ONLY_COMMON = "is-only-common"
PUBLIC_INTERFACE = "is-public-interface"
LAYERS = "is-layers"
SUITE_CONTRACTS = (ONLY_COMMON, PUBLIC_INTERFACE, LAYERS)
SUITE = "inference_suite"
PUBLIC = ("contracts", "argv", "golden", "server_log", "frameworks")
INTERNAL = (
    "__main__",
    "cli",
    "run",
    "load",
    "eval",
    "profile",
    "runtime",
    "telemetry",
    "compose",
    "recipe",
    "parser_check",
    "framework_version",
    "framework_probe",
    "staging",
    "pins",
    "aiperf_patch",
)
HOST_IMPORTERS = ("hyperloom.consumer", "hyperloom_multibranch.consumer")
#: The cross-module imports the package design relies on (dev plan 2.1 and 2.3).
PLANNED_IMPORTS: dict[str, tuple[str, ...]] = {
    SUITE: (f"{SUITE}.run", f"{SUITE}.recipe"),
    f"{SUITE}.__main__": (f"{SUITE}.cli",),
    f"{SUITE}.cli": (f"{SUITE}.run",),
    f"{SUITE}.run": (f"{SUITE}.load", f"{SUITE}.eval", f"{SUITE}.profile", "hyperloom_common"),
    f"{SUITE}.profile": (f"{SUITE}.telemetry",),
    f"{SUITE}.runtime": (f"{SUITE}.server_log", f"{SUITE}.contracts"),
    f"{SUITE}.compose": (f"{SUITE}.recipe", f"{SUITE}.parser_check", f"{SUITE}.framework_version"),
    f"{SUITE}.recipe": (f"{SUITE}.contracts", f"{SUITE}.golden"),
    f"{SUITE}.parser_check": (f"{SUITE}.framework_probe", f"{SUITE}.frameworks"),
    f"{SUITE}.contracts": (f"{SUITE}.argv", f"{SUITE}.golden", f"{SUITE}.frameworks"),
    f"{SUITE}.golden": (f"{SUITE}.argv",),
    "hyperloom.consumer": (SUITE, *(f"{SUITE}.{name}" for name in PUBLIC)),
    "hyperloom_multibranch.consumer": (SUITE, f"{SUITE}.contracts"),
}


def _real_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(CONFIG, encoding="utf-8")
    return parser


def _root_packages(parser: configparser.ConfigParser) -> list[str]:
    return parser.get("importlinter", "root_packages").split()


def _layer_modules(parser: configparser.ConfigParser) -> list[str]:
    layers = parser.get(f"importlinter:contract:{LAYERS}", "layers")
    return [name.strip().strip("()") for line in layers.split() for name in line.split("|") if name.strip("()")]


def _write_tree(root: Path, imports: Mapping[str, tuple[str, ...]]) -> Path:
    parser = _real_config()
    modules = {*_root_packages(parser), *_layer_modules(parser), *HOST_IMPORTERS, *imports}
    src = root / "src"
    for module in sorted(modules):
        package = src.joinpath(*module.split("."))
        package.mkdir(parents=True, exist_ok=True)
        lines = [f"import {imported}\n" for imported in imports.get(module, ())]
        (package / "__init__.py").write_text("".join(lines), encoding="utf-8")
    suite_only = configparser.ConfigParser(interpolation=None)
    for section in ("importlinter", *(f"importlinter:contract:{contract}" for contract in SUITE_CONTRACTS)):
        suite_only[section] = dict(parser[section])
    with (root / ".importlinter").open("w", encoding="utf-8") as handle:
        suite_only.write(handle)
    return src


def _lint(tmp_path: Path, imports: Mapping[str, tuple[str, ...]], contract: str | None = None) -> str:
    return _lint_tree(tmp_path, _write_tree(tmp_path, imports), contract)


def _lint_tree(tmp_path: Path, src: Path, contract: str | None = None) -> str:
    selected = ["--contract", contract] if contract else []
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlinter.cli import lint_imports_command; lint_imports_command()",
            "--config",
            str(tmp_path / ".importlinter"),
            "--no-cache",
            *selected,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(src), os.environ.get("PYTHONPATH")]))},
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout + completed.stderr


def test_planned_imports_keep_every_suite_contract(tmp_path: Path) -> None:
    output = _lint(tmp_path, PLANNED_IMPORTS)

    assert f"Contracts: {len(SUITE_CONTRACTS)} kept, 0 broken." in output, output


@pytest.mark.parametrize("project_package", ["hyperloom", "hyperloom_kb", "kernelforge", "hyperloom_multibranch"])
def test_suite_importing_another_project_package_breaks(tmp_path: Path, project_package: str) -> None:
    output = _lint(tmp_path, {f"{SUITE}.compose": (project_package,)}, ONLY_COMMON)

    assert "Contracts: 0 kept, 1 broken." in output, output


@pytest.mark.parametrize("importer", HOST_IMPORTERS)
@pytest.mark.parametrize("module", INTERNAL)
def test_host_importing_an_internal_module_breaks(tmp_path: Path, importer: str, module: str) -> None:
    output = _lint(tmp_path, {importer: (f"{SUITE}.{module}",)}, PUBLIC_INTERFACE)

    assert "Contracts: 0 kept, 1 broken." in output, output


def test_host_importing_inside_an_internal_package_breaks(tmp_path: Path) -> None:
    output = _lint(tmp_path, {"hyperloom.consumer": (f"{SUITE}.runtime.server",)}, PUBLIC_INTERFACE)

    assert "Contracts: 0 kept, 1 broken." in output, output


@pytest.mark.parametrize(
    ("importer", "imported"),
    [
        ("argv", "contracts"),
        ("contracts", "recipe"),
        ("telemetry", "runtime"),
        ("load", "eval"),
        ("recipe", "parser_check"),
        ("golden", "server_log"),
    ],
)
def test_upward_or_same_layer_import_breaks(tmp_path: Path, importer: str, imported: str) -> None:
    output = _lint(tmp_path, {f"{SUITE}.{importer}": (f"{SUITE}.{imported}",)}, LAYERS)

    assert "Contracts: 0 kept, 1 broken." in output, output


#: What a host resolving a base launch imports, all of it from the package root (dev plan 2.1 and 4.1).
ROOT_NAMES = ("LaunchQuery", "Miss", "SourceSnapshot", "fetch_sources", "resolve_base")


def test_host_importing_the_resolution_names_from_the_package_root_keeps_every_suite_contract(tmp_path: Path) -> None:
    src = _write_tree(tmp_path, {})
    for package in (SUITE, "hyperloom_common"):
        shutil.rmtree(src / package)
        shutil.copytree(SRC / package, src / package, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    (src / "hyperloom" / "consumer" / "__init__.py").write_text(
        f"from {SUITE} import {', '.join(ROOT_NAMES)}\n", encoding="utf-8"
    )

    output = _lint_tree(tmp_path, src)

    assert f"Contracts: {len(SUITE_CONTRACTS)} kept, 0 broken." in output, output


def test_the_package_root_exports_the_resolution_names() -> None:
    import inference_suite

    assert set(ROOT_NAMES) <= set(inference_suite.__all__)
    assert all(callable(getattr(inference_suite, name)) for name in ROOT_NAMES)
