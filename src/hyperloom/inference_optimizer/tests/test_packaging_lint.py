# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packaging-lint guard for ``pyproject.toml``.

Checks run in both directions, because each catches a different defect:

* declaration -> tree: setuptools silently ignores a ``package-data`` glob that
  matches nothing, so deleting a file leaves invisible rot at build time.
* tree -> declaration: a newly added asset that no glob covers is silently left
  out of the wheel, which breaks code that reads it from the install.

Wheel *contents* are asserted separately by ``.github/workflows/packaging.yml``,
which builds a real wheel; this module only needs the source tree.
"""

from __future__ import annotations

import ast
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

# Directory names whose contents never ship. Kept in sync with the
# ``packages.find`` exclude patterns by test_test_packages_are_excluded_*.
_TEST_DIR_NAMES = frozenset({"tests", "test", "testing"})

# Files under src/ that intentionally stay out of the wheel. Anything else that
# no declaration covers is a packaging bug, so keep this list justified.
_UNPACKAGED_ASSETS = (
    # Developer-only tooling, meaningless in an installed package.
    "**/.gitignore",
    "**/.ci-deferred/*",
    # Container image build context: the Dockerfile clones the repo and the
    # scripts hardcode /opt/Hyperloom, so they are only used from a checkout.
    "hyperloom/inference_optimizer/assets/quick-start/*",
    # The gemm-tune subpackage's own docs describe the source tree (how to run
    # the tuner from a checkout), not the installed package.
    "kernelforge/gemm_tune/*.md",
)

try:  # tomllib is stdlib from 3.11; the ``ci`` extra pins tomli for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on py3.10
    import tomli as tomllib  # type: ignore[no-redef]


def _find_repo_root() -> Path | None:
    """Walk up for the pyproject.toml; returns None when installed from a wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()

pytestmark = pytest.mark.skipif(
    _REPO_ROOT is None,
    reason="packaging lint needs the source checkout (pyproject.toml + src/)",
)


def _pyproject() -> dict:
    assert _REPO_ROOT is not None
    return tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _src() -> Path:
    assert _REPO_ROOT is not None
    return _REPO_ROOT / "src"


def test_package_data_globs_all_match_at_least_one_file():
    cfg = _pyproject()["tool"]["setuptools"]["package-data"]
    dead: list[str] = []
    for package, patterns in cfg.items():
        package_dir = _src() / package.replace(".", "/")
        if not package_dir.is_dir():
            dead.append(f"{package}: package directory {package_dir} does not exist")
            continue
        for pattern in patterns:
            if not list(package_dir.glob(pattern)):
                dead.append(f"{package}: '{pattern}' matches no file")
    assert not dead, "dead package-data declarations (setuptools ignores these silently): " + "; ".join(dead)


def test_data_files_sources_exist():
    assert _REPO_ROOT is not None
    cfg = _pyproject()["tool"]["setuptools"]["data-files"]
    missing = [
        f"{dest} <- {source}"
        for dest, sources in cfg.items()
        for source in sources
        if not (_REPO_ROOT / source).exists()
    ]
    assert not missing, f"data-files entries point at missing sources: {missing}"


def _module_path(dotted: str) -> Path | None:
    base = _src() / dotted.replace(".", "/")
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _defines_top_level(path: Path, name: str) -> bool:
    """True when ``name`` is defined or re-exported at module top level."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".")[0]) == name for alias in node.names
        ):
            return True
    return False


def test_console_script_targets_resolve():
    broken: list[str] = []
    for script, target in _pyproject()["project"]["scripts"].items():
        module, _, attribute = target.partition(":")
        path = _module_path(module)
        if path is None:
            broken.append(f"{script}: module '{module}' not found under src/")
        elif not _defines_top_level(path, attribute):
            broken.append(f"{script}: '{module}' does not define '{attribute}'")
    assert not broken, f"[project.scripts] entry points do not resolve: {broken}"


def _discovered_packages() -> list[str]:
    """Dotted names setuptools would discover (packages.find defaults to namespaces=true)."""
    names: list[str] = []
    for path in sorted(_src().rglob("*")):
        if not path.is_dir() or "__pycache__" in path.parts:
            continue
        if any(child.suffix == ".py" for child in path.iterdir() if child.is_file()):
            names.append(str(path.relative_to(_src())).replace("/", "."))
    return names


def test_test_packages_are_excluded_from_distribution():
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    excludes = find.get("exclude", [])
    leaked = [
        name
        for name in _discovered_packages()
        if _TEST_DIR_NAMES.intersection(name.split(".")) and not any(fnmatchcase(name, pattern) for pattern in excludes)
    ]
    assert not leaked, (
        "test packages would ship in the wheel (their fixture data is not package-data, "
        f"so a shipped copy cannot run): {leaked}"
    )


def _declared_asset_paths() -> set[str]:
    """Every src-relative non-Python file some declaration ships."""
    assert _REPO_ROOT is not None
    cfg = _pyproject()["tool"]["setuptools"]
    declared: set[str] = set()
    for package, patterns in cfg["package-data"].items():
        package_dir = _src() / package.replace(".", "/")
        for pattern in patterns:
            declared.update(str(match.relative_to(_src())) for match in package_dir.glob(pattern))
    # data-files sources are repo-relative; only src/ ones matter here.
    for sources in cfg["data-files"].values():
        for source in sources:
            path = _REPO_ROOT / source
            if path.is_relative_to(_src()):
                declared.add(str(path.relative_to(_src())))
    return declared


def _tracked_src_assets() -> list[str]:
    """Git-tracked non-Python files under src/, excluding test trees."""
    assert _REPO_ROOT is not None
    try:
        result = subprocess.run(
            ["git", "ls-files", "src"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git is required to enumerate tracked assets: {exc}")
    listing = result.stdout.split()
    assets = []
    for entry in listing:
        relative = Path(entry).relative_to("src")
        if relative.suffix == ".py" or _TEST_DIR_NAMES.intersection(relative.parts):
            continue
        assets.append(str(relative))
    return assets


def test_no_undeclared_assets_under_src():
    """A new asset must be declared or justified, never silently unshipped."""
    declared = _declared_asset_paths()
    orphans = [
        asset
        for asset in _tracked_src_assets()
        if asset not in declared and not any(fnmatchcase(asset, pattern) for pattern in _UNPACKAGED_ASSETS)
    ]
    assert not orphans, (
        "assets under src/ that no package-data or data-files entry ships; declare them "
        f"or add a justified entry to _UNPACKAGED_ASSETS: {orphans}"
    )


def test_ruff_per_file_ignores_paths_exist():
    assert _REPO_ROOT is not None
    cfg = _pyproject()["tool"]["ruff"]["lint"]["per-file-ignores"]
    dead = [pattern for pattern in cfg if not list(_REPO_ROOT.glob(pattern))]
    assert not dead, f"ruff per-file-ignores patterns match nothing: {dead}"


def test_license_uses_pep639_expression():
    project = _pyproject()["project"]
    assert isinstance(project.get("license"), str), (
        "project.license must be a PEP 639 SPDX string; the TOML-table form is "
        "deprecated by setuptools and becomes a hard error on 2027-Feb-18"
    )
    assert project.get("license-files"), "project.license-files must list the license file(s)"
