# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuild the framework the candidate changed, next to the image's, and put it first on the import path (design 4.8.3).

The farm mirrors the installed package: a directory no entry touches is one symlink to the image's directory, a
directory an entry touches is real, its untouched files are symlinks, an ``upsert`` is a copied file and a ``delete`` is
absent. The image is never written. Whether the rebuilt copy is what the server imports is then checked with the same
interpreter and the same path the server gets: a farm that exists but is not the first import location would measure
the image's framework and call it the candidate's.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure.spec import EnvEntry, EnvRoot

__all__ = ["Farm", "FarmError", "build_farm", "prepend_pythonpath", "verify_farm"]

_FIND_SPEC = (
    "import importlib.util, sys\n"
    "spec = importlib.util.find_spec(sys.argv[1])\n"
    "print('\\n'.join(spec.submodule_search_locations or []) if spec else '')\n"
)
_PROBE_TIMEOUT_SEC = 120


class FarmError(Exception):
    """The environment cannot be rebuilt as the sheet describes, or the rebuilt one is not what is imported."""


@dataclass(frozen=True)
class Farm:
    """The directory to put first on ``PYTHONPATH`` and the packages rebuilt in it."""

    site: Path
    packages: tuple[str, ...]
    roots: tuple[EnvRoot, ...] = ()


def _locate(import_name: str, python: str, cwd: Path) -> Path:
    """Where ``python`` finds the package by default (``-I``: no ``PYTHONPATH``), looked up from a neutral directory."""
    try:
        done = subprocess.run(
            [python, "-I", "-c", _FIND_SPEC, import_name],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FarmError(f"cannot ask {python} where {import_name} is: {exc}") from exc
    locations = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0 or not locations:
        raise FarmError(f"{python} cannot import {import_name}: {done.stderr.strip()[-300:]}")
    return Path(locations[0])


def _touched_directories(entries: Iterable[EnvEntry]) -> set[str]:
    return {
        "/".join(parts[:end]) for entry in entries for parts in [entry.path.split("/")] for end in range(len(parts))
    }


def _mirror(base: Path, target: Path, relative: str, touched: set[str], skip: set[str]) -> None:
    """Symlink every file and untouched directory of ``base/relative`` into ``target/relative``."""
    source, destination = base / relative, target / relative
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        child_rel = f"{relative}/{child.name}" if relative else child.name
        if child_rel in skip:
            continue
        if child.is_dir() and not child.is_symlink() and child_rel in touched:
            _mirror(base, target, child_rel, touched, skip)
        else:
            (destination / child.name).symlink_to(child)


def _write_upsert(entry: EnvEntry, destination: Path, read_blob: Callable[[str], bytes]) -> None:
    data = read_blob(entry.blob or "")
    if f"sha256:{hashlib.sha256(data).hexdigest()}" != entry.blob:
        raise FarmError(f"blob for {entry.path} does not hash to {entry.blob}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    destination.chmod(int(entry.mode or "0644", 8))


def _build_root(root: EnvRoot, base: Path, target: Path, read_blob: Callable[[str], bytes]) -> None:
    deletes = {entry.path for entry in root.entries if entry.op == "delete"}
    for path in deletes:
        if not (base / path).is_file():
            raise FarmError(f"{root.id}: {path} is to be deleted but the installed {root.import_name} has no such file")
    touched = _touched_directories(root.entries)
    _mirror(base, target, "", touched, deletes | {e.path for e in root.entries if e.op == "upsert"})
    for entry in root.entries:
        if entry.op == "upsert":
            _write_upsert(entry, target / entry.path, read_blob)


def build_farm(
    roots: Iterable[EnvRoot], farm_dir: Path, *, python: str, read_blob: Callable[[str], bytes]
) -> Farm | None:
    """Build ``farm_dir/site/<import_name>`` for every root; ``None`` when the sheet changes no framework.

    ``read_blob`` returns the bytes of a ``sha256:<hex>`` address and raises when it is not there.
    """
    site = farm_dir / "site"
    built = []
    rebuilt = []
    for root in roots:
        if root.kind != "package":
            raise FarmError(f"{root.id}: a runtime root is not supported by this runner yet")
        base = _locate(root.import_name, python, farm_dir.parent)
        _build_root(root, base, site / root.import_name, read_blob)
        built.append(root.import_name)
        rebuilt.append(root)
    return Farm(site, tuple(built), tuple(rebuilt)) if built else None


def prepend_pythonpath(envs: MutableMapping[str, Any], site: Path) -> None:
    """Put ``site`` first on the ``PYTHONPATH`` the server gets from the YAML, ahead of whatever the YAML had."""
    parts = [str(site), *str(envs.get("PYTHONPATH") or "").split(os.pathsep)]
    envs["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


def verify_farm(farm: Farm, envs: dict[str, Any], *, python: str, cwd: Path) -> None:
    """Check that ``python``, given this YAML's ``PYTHONPATH``, finds every rebuilt package in the farm first."""
    environment = {**os.environ, "PYTHONPATH": str(envs.get("PYTHONPATH") or "")}
    for name in farm.packages:
        done = subprocess.run(
            [python, "-c", _FIND_SPEC, name],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
            cwd=cwd,
            env=environment,
        )
        found = [line for line in done.stdout.splitlines() if line.strip()]
        expected = str(farm.site / name)
        if done.returncode != 0 or not found or found[0] != expected:
            raise FarmError(f"{name} would be imported from {found[:1] or done.stderr.strip()[-200:]}, not {expected}")
