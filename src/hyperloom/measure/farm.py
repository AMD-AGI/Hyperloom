# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuild the framework the candidate changed, next to the image's, and put it first on the import path (design 4.8.3).

The farm mirrors the installed package: a directory no entry touches is one symlink to the image's directory, a
directory an entry touches is real, its untouched files are symlinks, an ``upsert`` is a copied file and a ``delete`` is
absent. A directory the package itself writes into while the server runs is real too, whatever the entries touch,
so nothing a test's server writes reaches the image, where every later test on the node would load it. The image is
never written. Whether the rebuilt copy is what the server imports is then checked with the same
interpreter and the same path the server gets: a farm that exists but is not the first import location would measure
the image's framework and call it the candidate's.

A rebuilt aiter also has to be told where the image keeps the sources it imports ``csrc`` from and builds from, because
aiter looks for them around its own location, which is now the farm. ``aiter/jit/core.py`` puts that meta directory first
on ``sys.path`` of the server's process, and every worker the server spawns starts from that ``sys.path``; the image's own
meta directory of a develop install holds the image's ``aiter`` package, which then shadows the farm's in every worker. So
the farm mirrors the image's meta directory without its ``aiter`` and sets ``AITER_META_DIR`` to the mirror; a server that
inherits a meta directory holding an ``aiter`` package is refused. A patch to ``csrc`` or ``hsa`` is a root of its own
(``aiter_meta``): its patched files are laid over the same mirror, which aiter then builds its modules from.

Which copy each process of the server really imported is checked by the runner
(:mod:`~hyperloom.measure.runner.import_outcome`), which watches the imports from the farm's ``site`` directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.outside_facts import AITER_META_DIR_ENV, AITER_META_MARKER_DIR, AITER_META_WHEEL_DIR
from hyperloom.measure.jit.key import AITER_PACKAGE
from hyperloom.measure.jit.modules import image_aiter_meta
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.orchestrator.actions.executors.magpie_agentx import apply_native_runtime

__all__ = [
    "Farm",
    "FarmError",
    "build_farm",
    "image_aiter_meta",
    "inherited_environment",
    "locate_package",
    "patched_configs",
    "patches_aiter_sources",
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
_CONFIGS_DIR = "configs"

#: How the farm lays a directory a package writes into at run time. ``linked``: real, its files links (a writer that
#: renames a new file over a name replaces the link, not the image's file). ``empty``: real and empty. ``linked_tree``:
#: real down to every level, its files links, and lock files left out (a lock is opened for writing). ``copied``: real
#: down to every level, its files copies (a writer that opens a file in place would write through a link).
_LINKED, _EMPTY, _LINKED_TREE, _COPIED = "linked", "empty", "linked_tree", "copied"
_WHOLE_TREE = frozenset({_LINKED_TREE, _COPIED})
_LOCK_SUFFIX = ".lock"
#: The directories each package writes into while the server runs, by import name. aiter installs a module it built by
#: writing a temporary file in its JIT directory and renaming it over ``<module>.so`` (``atomic_copy`` of
#: ``aiter/jit/utils/jit_cache.py``), builds below ``jit/build``, keeps FlyDSL's kernel cache in ``jit/flydsl_cache``
#: (added by rename, guarded by ``<key>.lock`` files opened read-write) and, when a tuned table has duplicate rows,
#: rewrites the table in place below ``configs``.
_RUNTIME_WRITTEN: Mapping[str, Mapping[str, str]] = {
    AITER_PACKAGE: {"jit": _LINKED, "jit/build": _EMPTY, "jit/flydsl_cache": _LINKED_TREE, _CONFIGS_DIR: _COPIED},
}


class FarmError(Exception):
    """The environment cannot be rebuilt as the sheet describes, or the rebuilt one is not what is imported."""


@dataclass(frozen=True)
class Farm:
    """The directory to put first on ``PYTHONPATH``, the packages rebuilt in it and, when aiter is one of them and the
    server inherits no meta directory of its own, the image's, which the server is told to read aiter's sources from.

    When the sheet patches aiter's ``csrc`` or ``hsa`` (``meta_patched``), ``aiter_meta`` is the farm's own copy of the
    meta directory instead, and the server must read that one."""

    site: Path
    packages: tuple[str, ...]
    roots: tuple[EnvRoot, ...] = ()
    aiter_meta: Path | None = None
    meta_patched: bool = False


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


def patches_aiter_sources(roots: Iterable[Any]) -> bool:
    """Whether ``roots`` change aiter's ``csrc`` or ``hsa`` (the root of its meta directory has an entry)."""
    return any(root.import_name == AITER_META_WHEEL_DIR and root.entries for root in roots)


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


@dataclass(frozen=True)
class _Mirror:
    """What a mirror of ``base`` into ``target`` lays: ``touched`` directories are real, ``skip`` paths are absent, and
    a directory in ``written`` (relative path to how it is laid, :data:`_RUNTIME_WRITTEN`) is real whether or not an
    entry touches it."""

    base: Path
    target: Path
    touched: set[str]
    skip: set[str]
    written: Mapping[str, str] = field(default_factory=dict)


def _lay(mirror: _Mirror, relative: str, mode: str | None) -> None:
    """Symlink every file and untouched directory of ``base/relative`` into ``target/relative``, the directory laid as
    ``mode`` says (``None``: its files are links)."""
    source, destination = mirror.base / relative, mirror.target / relative
    destination.mkdir(parents=True, exist_ok=True)
    if mode == _EMPTY:
        return
    for child in sorted(source.iterdir()):
        child_rel = f"{relative}/{child.name}" if relative else child.name
        if child_rel in mirror.skip:
            continue
        if child.is_dir() and not child.is_symlink():
            inner = mirror.written.get(child_rel) or (mode if mode in _WHOLE_TREE else None)
            if inner is not None or child_rel in mirror.touched:
                _lay(mirror, child_rel, inner)
                continue
        _lay_file(child, destination / child.name, mode)


def _lay_file(source: Path, destination: Path, mode: str | None) -> None:
    """Lay one entry of a mirrored directory: a link to the image's, a copy of it, or nothing for a lock file."""
    if mode == _COPIED and source.is_file():
        shutil.copy2(source, destination)
    elif not (mode == _LINKED_TREE and source.name.endswith(_LOCK_SUFFIX)):
        destination.symlink_to(source)


def _write_upsert(entry: EnvEntry, destination: Path, read_blob: Callable[[str], bytes]) -> None:
    data = read_blob(entry.blob or "")
    if f"sha256:{hashlib.sha256(data).hexdigest()}" != entry.blob:
        raise FarmError(f"blob for {entry.path} does not hash to {entry.blob}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    destination.chmod(int(entry.mode or "0644", 8))


def _build_root(
    root: EnvRoot,
    base: Path,
    target: Path,
    read_blob: Callable[[str], bytes],
    *,
    left_out: frozenset[str] = frozenset(),
) -> None:
    deletes = {entry.path for entry in root.entries if entry.op == "delete"}
    for path in deletes:
        if not (base / path).is_file():
            raise FarmError(f"{root.id}: {path} is to be deleted but the installed {root.import_name} has no such file")
    touched = _touched_directories(root.entries)
    skip = deletes | {e.path for e in root.entries if e.op == "upsert"} | left_out
    _lay(_Mirror(base, target, touched, skip, _RUNTIME_WRITTEN.get(root.import_name, {})), "", None)
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
    gets the farm's mirror of the image's meta directory only when it inherits none.
    """
    site = farm_dir / "site"
    roots = list(roots)
    meta_root = next((root for root in roots if root.import_name == AITER_META_WHEEL_DIR), None)
    packages = [root for root in roots if root is not meta_root]
    for root in roots:
        if root.kind != "package":
            raise FarmError(f"{root.id}: a runtime root is not supported by this runner yet")
    for root in packages:
        _build_root(root, locate_package(root.import_name, python, farm_dir.parent), site / root.import_name, read_blob)
    meta = None
    if meta_root is not None or any(root.import_name == AITER_PACKAGE for root in packages):
        aiter = locate_package(AITER_PACKAGE, python, farm_dir.parent)
        meta = _build_meta_root(meta_root, aiter, site, os.environ if inherited is None else inherited, read_blob)
    rebuilt = tuple(packages) + (() if meta_root is None else (meta_root,))
    if not rebuilt:
        return None
    return Farm(site, tuple(root.import_name for root in packages), rebuilt, meta, meta_patched=meta_root is not None)


def _build_meta_root(
    root: EnvRoot | None, aiter: Path, site: Path, server_has: Mapping[str, str], read_blob: Callable[[str], bytes]
) -> Path | None:
    """Mirror the image's meta directory into ``site`` with the patched ``csrc`` and ``hsa`` of ``root`` (when there is
    one) laid over it; ``None`` when the sheet patches only aiter's Python and the server inherits a meta directory.

    The image's own ``aiter`` directory is left out, since aiter puts the meta directory first on its import path, where a
    second ``aiter`` would shadow the package of the farm in every worker. For the same reason an inherited meta directory
    that holds an ``aiter`` package is refused; one that patched sources need is refused whatever it holds, since the server
    would build the image's sources and call them the candidate's.
    """
    inherited = server_has.get(AITER_META_DIR_ENV)
    if inherited:
        if root is not None:
            raise FarmError(
                f"{root.id}: the server inherits {AITER_META_DIR_ENV}={inherited!r}, which it would read instead of the "
                "patched csrc and hsa; the test cannot be measured with it set"
            )
        if (Path(inherited) / AITER_PACKAGE).is_dir():
            raise FarmError(
                f"the server inherits {AITER_META_DIR_ENV}={inherited!r}, which holds an {AITER_PACKAGE} package: aiter "
                f"puts that directory first on the import path its workers start from, so they would import that "
                f"{AITER_PACKAGE} and not the farm's"
            )
        return None
    target, left_out = site / AITER_META_WHEEL_DIR, frozenset({AITER_PACKAGE})
    image_meta = _checked_aiter_meta(aiter)
    if root is None:
        _lay(_Mirror(image_meta, target, set(), set(left_out)), "", None)
        return target
    shadowing = next((entry.path for entry in root.entries if entry.path.split("/")[0] == AITER_PACKAGE), None)
    if shadowing is not None:
        raise FarmError(f"{root.id}: {shadowing} would put a second {AITER_PACKAGE} package ahead of the farm's")
    _build_root(root, image_meta, target, read_blob, left_out=left_out)
    return target


def _checked_aiter_meta(aiter: Path) -> Path:
    meta = image_aiter_meta(aiter)
    if not (meta / AITER_META_MARKER_DIR).is_dir():
        raise FarmError(
            f"the installed {AITER_PACKAGE} at {aiter} keeps no {AITER_META_MARKER_DIR} directory in {meta}, so a rebuilt "
            f"copy of it could not be told where to read its sources from ({AITER_META_DIR_ENV}) and would not import"
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


def patched_configs(farm: Farm | None) -> dict[str, Path | None]:
    """The tuned tables the farm's aiter patches, by their path below ``configs``: the farm's copy of each, ``None`` for
    one the patch deletes. Empty when the farm rebuilds no aiter or leaves its ``configs`` as the image has them."""
    if farm is None:
        return {}
    package = farm.site / AITER_PACKAGE
    patched: dict[str, Path | None] = {}
    for root in farm.roots:
        if root.import_name != AITER_PACKAGE:
            continue
        for entry in root.entries:
            relative = entry.path.removeprefix(f"{_CONFIGS_DIR}/")
            if relative != entry.path:
                patched[relative] = None if entry.op == "delete" else package / entry.path
    return patched


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
    if AITER_PACKAGE in farm.packages or farm.meta_patched:
        _verify_aiter_meta(farm, envs)


def _verify_aiter_meta(farm: Farm, envs: Mapping[str, Any]) -> None:
    """A meta directory the farm pins or the YAML names must hold ``csrc``: aiter ignores one that does not and falls back
    to the directory around the farm, which has none. A value the server inherits is its own and is not looked at."""
    named = envs.get(AITER_META_DIR_ENV)
    if named is None and farm.aiter_meta is None:
        return
    value = str(named or "")
    if farm.meta_patched and value != str(farm.aiter_meta):
        raise FarmError(
            f"the patched csrc and hsa are in the farm's meta directory {farm.aiter_meta}, but the server would be told "
            f"{AITER_META_DIR_ENV}={value!r} and would build the image's sources"
        )
    if not value or not (Path(value) / AITER_META_MARKER_DIR).is_dir():
        raise FarmError(
            f"the rebuilt aiter would be told {AITER_META_DIR_ENV}={value!r}, which holds no {AITER_META_MARKER_DIR}: "
            f"it would not find its sources and would not import"
        )
