# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression guard for the wheel build/ recursion incident.

The pyproject lives INSIDE the package dir. Discovering packages with
``where = [".."]`` makes a rebuild from a checkout carrying leftover ``build/``
output rediscover / recursively copy it into the new ``build/lib`` -> a
``build/lib/<pkg>/build/lib/<pkg>/...`` tree and a ``dist-info`` "File exists"
wheel-build failure (observed across 8 top models on the shared
``/shared_nfs/hyperloom/KernelForge`` checkout). Explicit package mapping breaks
that loop; these tests keep the config from regressing AND build the wheel to
lock the incident's actual trigger.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]  # the package dir (src/<pkg>)
_PKG_NAME = _PKG_ROOT.name
_PYPROJECT = _PKG_ROOT / "pyproject.toml"


def _setuptools_cfg() -> dict | None:
    """Parsed [tool.setuptools], or None when no TOML parser is available."""
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:  # py3.10
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return None
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8")).get("tool", {}).get("setuptools", {})


def _noncomment(raw: str) -> str:
    return "\n".join(ln for ln in raw.splitlines() if not ln.lstrip().startswith("#"))


def _listed_packages() -> set[str]:
    """Explicit packages list, via TOML parse or a raw-text fallback (never skips)."""
    cfg = _setuptools_cfg()
    if cfg is not None and isinstance(cfg.get("packages"), list):
        return set(cfg["packages"])
    m = re.search(r"packages\s*=\s*\[([^\]]*)\]", _noncomment(_PYPROJECT.read_text(encoding="utf-8")))
    assert m, "no explicit `packages = [...]` list found"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_explicit_packages_not_parent_find():
    code = _noncomment(_PYPROJECT.read_text(encoding="utf-8"))
    assert _PKG_NAME in _listed_packages()
    assert "package-dir" in code
    # The fragile parent-find discovery must be gone (check non-comment text so
    # the explanatory comment mentioning it does not trip the assertion).
    assert "packages.find" not in code, "packages.find reintroduces the build/ recursion"
    assert 'where = [".."]' not in code, "where=['..'] rediscovers leftover build/ output"


def test_all_runtime_subpackages_are_listed():
    listed = _listed_packages()
    found: set[str] = set()
    for init in _PKG_ROOT.rglob("__init__.py"):
        parts = init.parent.relative_to(_PKG_ROOT).parts
        if parts and (parts[0] == "tests" or "build" in parts or "__pycache__" in parts):
            continue
        found.add(".".join((_PKG_NAME, *parts)))
    missing = found - listed
    assert not missing, f"runtime subpackages missing from packages list: {sorted(missing)}"


def test_wheel_builds_twice_without_recursion(tmp_path):
    # Behavior-level guard: build the wheel TWICE (the 2nd with a stale build/
    # from the 1st) and assert the incident signature never appears.
    src = tmp_path / _PKG_NAME
    shutil.copytree(
        _PKG_ROOT,
        src,
        ignore=shutil.ignore_patterns("build", "__pycache__", "*.egg-info", ".pytest_cache"),
    )
    wheeldir = tmp_path / "wh"
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", ".", "-w", str(wheeldir)],
            cwd=src,
            capture_output=True,
            text=True,
        )
        blob = (r.stdout or "") + (r.stderr or "")
        # The incident's exact signatures -> hard fail (regression caught).
        assert "Errno 17" not in blob and "File exists" not in blob, (
            f"wheel-recursion incident reproduced:\n{blob[-500:]}"
        )
    # No nested build/ dir (recursion) in the on-disk tree.
    build_dir = src / "build"
    if build_dir.exists():
        nested = [str(p) for p in build_dir.rglob("build")]
        assert not nested, f"recursive nested build/ tree: {nested[:2]}"
    wheels = list(wheeldir.glob("*.whl"))
    if not wheels:
        pytest.skip("no wheel produced (build toolchain unavailable in this env)")
    names = zipfile.ZipFile(wheels[-1]).namelist()
    assert not any("/build/" in n for n in names), "build/ leaked into the wheel"
    assert not any("/tests/" in n for n in names), "tests/ leaked into the wheel"
    assert any(n.startswith(f"{_PKG_NAME}/") and n.endswith(".py") for n in names), "no package modules in wheel"


@pytest.mark.standalone_wheel_e2e
def test_standalone_wheel_imports_without_kernelforge(tmp_path):
    """Install the subpackage wheel alone and import its knowledge-store path."""
    src = tmp_path / _PKG_NAME
    shutil.copytree(
        _PKG_ROOT,
        src,
        ignore=shutil.ignore_patterns(
            "build",
            "__pycache__",
            "*.egg-info",
            ".pytest_cache",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            ".",
            "-w",
            str(wheel_dir),
        ],
        cwd=src,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, (build.stdout or "") + (build.stderr or "")
    [wheel] = wheel_dir.glob("*.whl")

    venv_dir = tmp_path / "venv"
    create = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
    )
    assert create.returncode == 0, (create.stdout or "") + (create.stderr or "")
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, (install.stdout or "") + (install.stderr or "")

    smoke = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.util;"
                "assert importlib.util.find_spec('kernelforge') is None;"
                "from forge_gemm_tune.router import resolve_gpu_type;"
                "from forge_gemm_tune.artifact_manifest import build_artifact_manifest;"
                "assert resolve_gpu_type('mi300x') == 'mi300x'"
            ),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert smoke.returncode == 0, (smoke.stdout or "") + (smoke.stderr or "")
