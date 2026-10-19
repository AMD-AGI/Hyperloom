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
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.orchestrator.actions.executors.magpie_agentx import apply_native_runtime

__all__ = [
    "Farm",
    "FarmError",
    "build_farm",
    "place_farm",
    "prepend_pythonpath",
    "server_environment",
    "server_python",
    "verify_farm",
]

_FIND_SPEC = (
    "import importlib.util, sys\n"
    "spec = importlib.util.find_spec(sys.argv[1])\n"
    "print('\\n'.join(spec.submodule_search_locations or []) if spec else '')\n"
)
_PROBE_TIMEOUT_SEC = 120
_AGENTX_DEFAULT_PYTHON = "python3"


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


def _agentx_launch(bench: Mapping[str, Any]) -> Mapping[str, Any]:
    agentx = bench.get("agentx")
    return (agentx.get("launch_overrides") if isinstance(agentx, Mapping) else None) or {}


def server_environment(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The variables the framework server of ``config`` is launched with.

    Magpie launches an AgentX server from ``agentx.launch_overrides.env`` and never reads ``benchmark.envs`` for it;
    any other server gets ``benchmark.envs``.
    """
    bench = config.get("benchmark") or {}
    return (_agentx_launch(bench).get("env") if is_magpie_agentx(bench) else bench.get("envs")) or {}


def named_python(config: Mapping[str, Any]) -> str:
    """The interpreter ``config`` names for its server through ``HYPERLOOM_FRAMEWORK_PYTHON``, or ``""``.

    An AgentX server's interpreter can also be the ``executable`` its launch already carries.
    """
    named = server_environment(config).get("HYPERLOOM_FRAMEWORK_PYTHON")
    if not named and is_magpie_agentx(config.get("benchmark") or {}):
        named = _agentx_launch(config["benchmark"]).get("executable")
    return str(named or "")


def server_python(config: Mapping[str, Any], *, default: str) -> str:
    """The interpreter the server of ``config`` runs with: the one it names, else its launcher's default.

    Magpie starts an AgentX server with ``python3`` unless the launch names an executable; any other server uses
    ``default``.
    """
    agentx = is_magpie_agentx(config.get("benchmark") or {})
    return named_python(config) or (_AGENTX_DEFAULT_PYTHON if agentx else default)


def place_farm(config_path: Path, farm: Farm) -> Mapping[str, Any]:
    """Put the farm first on the import path of the server ``config_path`` launches; return the variables it gets.

    An AgentX server's variables and interpreter are the launch's, so the farm goes where the single-lineage flow puts
    an overlay: ``apply_native_runtime`` writes it into ``launch_overrides.env`` and binds the rebuilt files as source
    evidence, and an interpreter the YAML names becomes the launch executable.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if is_magpie_agentx(config.get("benchmark") or {}):
        python = named_python(config)
        apply_native_runtime(config_path, {"runtime_python_exe": python} if python else {}, str(farm.site))
        return server_environment(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    envs = config.setdefault("benchmark", {}).setdefault("envs", {})
    prepend_pythonpath(envs, farm.site)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return envs


def verify_farm(farm: Farm, envs: Mapping[str, Any], *, python: str, cwd: Path) -> None:
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
