# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuild the framework the candidate changed, next to the image's, and put it first on the import path (design 4.8.3).

The farm mirrors the installed package: a directory no entry touches is one symlink to the image's directory, a
directory an entry touches is real, its untouched files are symlinks, an ``upsert`` is a copied file and a ``delete`` is
absent. The image is never written. Whether the rebuilt copy is what the server imports is then checked with the same
interpreter and the same path the server gets: a farm that exists but is not the first import location would measure
the image's framework and call it the candidate's.

A rebuilt aiter also has to be told where the image keeps the sources it imports ``csrc`` from and builds from, because
aiter looks for them around its own location, which is now the farm: unless the server already inherits one, the farm
sets ``AITER_META_DIR`` to the image's own meta directory, resolved the way aiter resolves it.
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
from hyperloom.common.outside_facts import (
    AITER_DEVELOP_MODE,
    AITER_INSTALL_MODE_FILE,
    AITER_META_DIR_ENV,
    AITER_META_MARKER_DIR,
    AITER_META_WHEEL_DIR,
)
from hyperloom.measure.jit.key import AITER_PACKAGE
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.orchestrator.actions.executors.magpie_agentx import apply_native_runtime

__all__ = [
    "Farm",
    "FarmError",
    "build_farm",
    "image_aiter_meta",
    "inherited_environment",
    "locate_package",
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
_UTF8 = "utf-8"


class FarmError(Exception):
    """The environment cannot be rebuilt as the sheet describes, or the rebuilt one is not what is imported."""


@dataclass(frozen=True)
class Farm:
    """The directory to put first on ``PYTHONPATH``, the packages rebuilt in it and, when aiter is one of them and the
    server inherits no meta directory of its own, the image's, which the server is told to read aiter's sources from."""

    site: Path
    packages: tuple[str, ...]
    roots: tuple[EnvRoot, ...] = ()
    aiter_meta: Path | None = None


def locate_package(import_name: str, python: str, cwd: Path) -> Path:
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


def image_aiter_meta(package: Path) -> Path:
    """The meta directory the image's own aiter reads ``csrc`` from, resolved as ``aiter/jit/core.py`` resolves it.

    ``package`` is the aiter package directory the image's interpreter finds. A develop install (what the
    ``install_mode`` file beside the package says, and what a missing file means) keeps ``csrc`` in the directory around
    the package; a wheel install keeps it in the ``aiter_meta`` package beside it.
    """
    try:
        develop = (package / AITER_INSTALL_MODE_FILE).read_text(encoding=_UTF8).strip() == AITER_DEVELOP_MODE
    except FileNotFoundError:
        develop = True
    return package.parent if develop else package.parent / AITER_META_WHEEL_DIR


def inherited_environment(
    environ: Mapping[str, str], process_env: Mapping[str, str], unset_envs: Iterable[str]
) -> dict[str, str]:
    """What a server's process inherits before its YAML's variables: the runner's environment with the test's own
    process variables over it, minus the ones the test unsets."""
    unset = set(unset_envs)
    return {name: value for name, value in {**environ, **process_env}.items() if name not in unset}


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
    roots: Iterable[EnvRoot],
    farm_dir: Path,
    *,
    python: str,
    read_blob: Callable[[str], bytes],
    inherited: Mapping[str, str] | None = None,
) -> Farm | None:
    """Build ``farm_dir/site/<import_name>`` for every root; ``None`` when the sheet changes no framework.

    ``read_blob`` returns the bytes of a ``sha256:<hex>`` address and raises when it is not there. ``inherited`` is what the
    server's process inherits (:func:`inherited_environment`), this process's environment when not given: a rebuilt aiter
    is told the image's meta directory only when it names none.
    """
    site = farm_dir / "site"
    server_has = os.environ if inherited is None else inherited
    built = []
    rebuilt = []
    meta = None
    for root in roots:
        if root.kind != "package":
            raise FarmError(f"{root.id}: a runtime root is not supported by this runner yet")
        base = locate_package(root.import_name, python, farm_dir.parent)
        _build_root(root, base, site / root.import_name, read_blob)
        if root.import_name == AITER_PACKAGE and not server_has.get(AITER_META_DIR_ENV):
            meta = _checked_aiter_meta(root, base)
        built.append(root.import_name)
        rebuilt.append(root)
    return Farm(site, tuple(built), tuple(rebuilt), meta) if built else None


def _checked_aiter_meta(root: EnvRoot, base: Path) -> Path:
    meta = image_aiter_meta(base)
    if not (meta / AITER_META_MARKER_DIR).is_dir():
        raise FarmError(
            f"{root.id}: the installed {root.import_name} at {base} keeps no {AITER_META_MARKER_DIR} directory in {meta}, "
            f"so a rebuilt copy of it could not be told where to read its sources from ({AITER_META_DIR_ENV}) and would "
            f"not import"
        )
    return meta


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


def _loaded(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding=_UTF8)) or {}


def _launch_environment(config: dict[str, Any]) -> dict[str, Any]:
    """The mapping the server of ``config`` takes its variables from, created when it has none: what :func:`server_environment`
    reads, for a caller that writes."""
    bench = config.setdefault("benchmark", {})
    if is_magpie_agentx(bench):
        return bench["agentx"].setdefault("launch_overrides", {}).setdefault("env", {})
    return bench.setdefault("envs", {})


def place_farm(config_path: Path, farm: Farm) -> Mapping[str, Any]:
    """Put the farm first on the import path of the server ``config_path`` launches; return the variables it gets.

    An AgentX server's variables and interpreter are the launch's, so the farm goes where the single-lineage flow puts
    an overlay: ``apply_native_runtime`` writes it into ``launch_overrides.env`` and binds the rebuilt files as source
    evidence, and an interpreter the YAML names becomes the launch executable.

    A farm that pins the image's aiter meta directory sets ``AITER_META_DIR`` to it, unless the YAML already names one:
    the rebuilt aiter sits in the farm, where it finds no ``csrc`` around it.
    """
    config = _loaded(config_path)
    agentx = is_magpie_agentx(config.get("benchmark") or {})
    if agentx:
        python = named_python(config)
        apply_native_runtime(config_path, {"runtime_python_exe": python} if python else {}, str(farm.site))
        config = _loaded(config_path)
    envs = _launch_environment(config)
    if not agentx:
        prepend_pythonpath(envs, farm.site)
    if farm.aiter_meta is not None:
        envs.setdefault(AITER_META_DIR_ENV, str(farm.aiter_meta))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
    return envs


def verify_farm(farm: Farm, envs: Mapping[str, Any], *, python: str, cwd: Path) -> None:
    """Check that ``python``, given this YAML's variables, finds every rebuilt package in the farm first, and that a
    rebuilt aiter is told a meta directory that holds ``csrc``."""
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
    if AITER_PACKAGE in farm.packages:
        _verify_aiter_meta(farm, envs)


def _verify_aiter_meta(farm: Farm, envs: Mapping[str, Any]) -> None:
    """A meta directory the farm pins or the YAML names must hold ``csrc``: aiter ignores one that does not and falls back
    to the directory around the farm, which has none. A value the server inherits is its own and is not looked at."""
    named = envs.get(AITER_META_DIR_ENV)
    if named is None and farm.aiter_meta is None:
        return
    value = str(named or "")
    if not value or not (Path(value) / AITER_META_MARKER_DIR).is_dir():
        raise FarmError(
            f"the rebuilt aiter would be told {AITER_META_DIR_ENV}={value!r}, which holds no {AITER_META_MARKER_DIR}: "
            f"it would not find its sources and would not import"
        )
