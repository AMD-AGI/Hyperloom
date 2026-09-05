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
from fnmatch import fnmatchcase
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _excluded_package_patterns(cfg: dict) -> list[str]:
    """The packages.find exclude patterns that keep a tree out of the wheel.

    Patterns naming a shipped package-data *subtree* are dropped.
    ``kernelforge.data`` is excluded from *package* discovery -- its resource
    trees contain .py sample kernels that must not be handed out as importable
    modules -- but its files do ship, declared as ``kernelforge =
    ["data/**/*"]``, so matching wheel entries against it would flag the whole
    resource tree as a leak.

    The skip is keyed on the subtree the globs actually name (``kernelforge`` +
    ``data/**/*`` -> ``kernelforge.data``), not on the package-data key alone.
    A bare ``startswith("kernelforge.")`` would also swallow a future
    ``kernelforge.tests`` exclusion, narrowing this check while the exclude
    list widened.
    """
    patterns = cfg["tool"]["setuptools"]["packages"]["find"].get("exclude", [])
    shipped = tuple(
        f"{key}.{glob.split('/', 1)[0]}"
        for key, globs in cfg["tool"]["setuptools"].get("package-data", {}).items()
        for glob in globs
        if "/" in glob and "*" not in glob.split("/", 1)[0]
    )
    return [
        pattern for pattern in patterns if not any(pattern == key or pattern.startswith(f"{key}.") for key in shipped)
    ]


def _check_no_excluded_packages(cfg: dict, names: list[str]) -> list[str]:
    """No wheel entry may sit in a package packages.find was told to exclude.

    Entries are matched as the dotted package name of their directory, against
    the exclude patterns verbatim -- the same comparison ``test_packaging_lint``
    makes against the source tree, so the two cannot disagree about what a
    pattern covers. Reading literal directory *segments* out of the patterns
    instead would put "hyperloom" and "orchestrator" in the leak vocabulary the
    moment a single subpackage was excluded, and condemn the whole wheel.
    """
    patterns = _excluded_package_patterns(cfg)
    if not patterns:
        return ["packages.find declares no exclude, so nothing keeps test trees out of the wheel"]
    leaked = sorted(
        name for name in names if any(fnmatchcase(".".join(Path(name).parent.parts), pattern) for pattern in patterns)
    )
    if not leaked:
        return []
    return [f"{len(leaked)} entries from excluded packages shipped in the wheel, e.g. {leaked[:5]}"]


def _check_declared_package_data_is_present(cfg: dict, names: list[str]) -> list[str]:
    """Every file a package-data glob matches on disk must exist in the wheel."""
    present = set(names)
    errors: list[str] = []
    for package, patterns in cfg["tool"]["setuptools"]["package-data"].items():
        package_dir = _REPO_ROOT / "src" / package.replace(".", "/")
        for pattern in patterns:
            # ``data/**/*`` matches directories too; only files are wheel entries.
            matches = sorted(match for match in package_dir.glob(pattern) if match.is_file())
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


#: Resource trees that must never ship empty. A ``package-data`` glob that
#: silently matches nothing is caught above as a dead declaration, but a tree
#: that lost all but one file would still pass -- and forge-loop running against
#: an empty knowledge base produces no error, just worse kernels. Absorbed from
#: KernelForge's deleted ``test_wheel_content.py``, which built its own wheel.
_NON_EMPTY_TREES = {
    # kernelforge/data/knowledge_base/ used to be listed here with a floor of
    # 100. The tree was removed after an audit found no reader: nothing consumed
    # config.knowledge_dir, no prompt pointed at it, and it was never granted to
    # an agent sandbox.
    # Was 700, when local_knowledge still carried per-operator cards duplicated
    # across every language folder. That duplication was removed deliberately
    # (the same card existed 3-5 times over, and operator-level facts go stale
    # faster than they can be maintained), taking the tree from 720 .md files to
    # 213. Then languages/asm/ went too (117 files: AMD RAD's vendored IntelliKit
    # ASM skills plus the CDNA4 ISA extracts), when the intellikit kernel backend
    # it served was removed -- no other backend maps to that language folder.
    # The floor is a "did the tree get wiped" guard, not a size assertion --
    # 120 keeps that guard meaningful against the current 134 files.
    "kernelforge/data/local_knowledge/": 120,
    "kernelforge/data/examples/": 40,
    # 1, not 3. The tree holds exactly three files today, so a floor of 3 was
    # really "all of them", and the two non-patch files (a README and a
    # SUPPORTED_VERSIONS.txt) are documentation whose legitimate removal would
    # have turned this check red for no packaging reason. What must actually
    # ship is the patch itself, and _REQUIRED_FILES asserts that by name --
    # a floor cannot, since three READMEs would satisfy it.
    "kernelforge/data/serving_patches/": 1,
}

#: Individual resources whose absence is a packaging bug rather than a smaller
#: tree. A count floor cannot express "this specific file", and for a tree of
#: three files the distinction is the whole guard.
_REQUIRED_FILES = ("kernelforge/data/serving_patches/sglang/sglang_0_5_12/fp8_blockscale_ck_routing.patch",)


def _check_resource_trees_are_populated(names: list[str]) -> list[str]:
    errors = []
    for prefix, floor in sorted(_NON_EMPTY_TREES.items()):
        count = sum(1 for n in names if n.startswith(prefix) and not n.endswith("/"))
        if count < floor:
            errors.append(f"{prefix} ships {count} files, below the floor of {floor}")
    present = set(names)
    errors.extend(
        f"{required} is declared but missing from the wheel" for required in _REQUIRED_FILES if required not in present
    )
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
            *_check_no_excluded_packages(cfg, names),
            *_check_declared_package_data_is_present(cfg, names),
            *_check_data_files_are_present(cfg, names),
            *_check_resource_trees_are_populated(names),
            *_check_license_metadata(zf, names),
        ]

    if errors:
        print(f"FAIL {args.wheel.name}: {len(errors)} problem(s)", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"OK {args.wheel.name}: {len(names)} entries, no excluded packages, declared assets all present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
