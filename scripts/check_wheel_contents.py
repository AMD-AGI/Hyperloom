# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assert a built wheel's contents against ``pyproject.toml``.

Complements ``test_packaging_lint.py``: that module lints the declarations
against the source tree, this one opens the real artifact. Checks are derived
from ``pyproject.toml`` rather than hardcoded, so the list cannot rot.

Usage: python scripts/check_wheel_contents.py dist/*.whl
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _excluded_dir_names(cfg: dict) -> set[str]:
    """Literal directory names in the packages.find exclude patterns.

    Derived rather than hardcoded so this cannot narrow while the exclude list
    widens: ``*.testing.*`` contributes ``testing``, the ``*`` segments nothing.
    """
    patterns = cfg["tool"]["setuptools"]["packages"]["find"].get("exclude", [])
    return {segment for pattern in patterns for segment in pattern.split(".") if segment != "*"}


def _check_no_test_packages(cfg: dict, names: list[str]) -> list[str]:
    excluded = _excluded_dir_names(cfg)
    if not excluded:
        return ["packages.find declares no exclude, so nothing keeps test trees out of the wheel"]
    leaked = sorted(n for n in names if excluded & set(Path(n).parts))
    if not leaked:
        return []
    return [f"{len(leaked)} test entries shipped in the wheel, e.g. {leaked[:5]}"]


def _check_declared_package_data_is_present(cfg: dict, names: list[str]) -> list[str]:
    """Every file a package-data glob matches on disk must exist in the wheel."""
    present = set(names)
    errors: list[str] = []
    for package, patterns in cfg["tool"]["setuptools"]["package-data"].items():
        package_dir = _REPO_ROOT / "src" / package.replace(".", "/")
        for pattern in patterns:
            matches = sorted(package_dir.glob(pattern))
            if not matches:
                errors.append(f"{package}: '{pattern}' matches no file on disk (dead declaration)")
                continue
            for match in matches:
                arcname = str(match.relative_to(_REPO_ROOT / "src"))
                if arcname not in present:
                    errors.append(f"{package}: declared '{arcname}' is missing from the wheel")
    return errors


def _check_data_files_are_present(cfg: dict, names: list[str]) -> list[str]:
    """Skill assets install via the wheel's ``.data/data/`` tree."""
    data_entries = {n.split(".data/data/", 1)[1] for n in names if ".data/data/" in n}
    errors: list[str] = []
    for dest, sources in cfg["tool"]["setuptools"]["data-files"].items():
        for source in sources:
            expected = f"{dest}/{Path(source).name}"
            if expected not in data_entries:
                errors.append(f"data-files: '{expected}' is missing from the wheel .data/data/ tree")
    return errors


def _check_license_metadata(zf: zipfile.ZipFile, names: list[str]) -> list[str]:
    metadata_name = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    if metadata_name is None:
        return ["wheel has no dist-info/METADATA"]
    metadata = zf.read(metadata_name).decode("utf-8")
    if "License-Expression:" not in metadata:
        return ["METADATA lacks License-Expression (PEP 639 license form regressed)"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="path to the built .whl")
    args = parser.parse_args()

    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    with zipfile.ZipFile(args.wheel) as zf:
        names = zf.namelist()
        errors = [
            *_check_no_test_packages(cfg, names),
            *_check_declared_package_data_is_present(cfg, names),
            *_check_data_files_are_present(cfg, names),
            *_check_license_metadata(zf, names),
        ]

    if errors:
        print(f"FAIL {args.wheel.name}: {len(errors)} problem(s)", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"OK {args.wheel.name}: {len(names)} entries, no test packages, declared assets all present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
