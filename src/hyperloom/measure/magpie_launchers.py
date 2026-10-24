# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Magpie's launcher is in the InferenceX checkout before the first round of a test builds its configuration.

Magpie copies the launchers it ships (``scripts/benchmark/*.sh``) into ``<InferenceX>/benchmarks/`` when a round starts,
over a script of the same name. ``run_grid`` reads the launcher from that directory while it builds a round's
configuration, and drops from ``benchmark.envs`` every variable the launcher exports unconditionally (the recipe
overwrites it). A checkout the runner has just made clean has no launcher while the first round of a test is built, so
that round (the boot of a sweep arm, the warmup of a graded test) keeps a variable every later round drops, and
Magpie's reuse check, which compares those environments, refuses the later rounds ("Reuse metadata mismatch ...
server_env"). Putting the launcher there first gives every round the same answer.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx

__all__ = ["install_launchers"]

log = logging.getLogger(__name__)

_MAGPIE_PATH_ENV = "MAGPIE_PATH"
_INFERENCEX_PATH_ENV = "INFERENCEX_PATH"
_LAUNCHERS = Path("scripts") / "benchmark"
_BENCHMARKS = "benchmarks"
_EXECUTABLE = 0o700
_UTF8 = "utf-8"


def install_launchers(points: Iterable[Path]) -> None:
    """Put Magpie's copy of each launcher ``points`` name into the InferenceX checkout they run from, as Magpie would.

    A point whose launcher Magpie does not ship, a checkout without a ``benchmarks`` directory and a copy that cannot
    be made are left as they are (the last one is logged): the test runs either way.
    """
    source = _magpie_launchers()
    if source is None:
        return
    for point in points:
        bench = (yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}).get("benchmark") or {}
        script = Path(str(bench.get("benchmark_script") or "")).name
        if not script or is_magpie_agentx(bench):
            continue
        for checkout in _checkouts(bench):
            _install(source / script, checkout / _BENCHMARKS / script)


def _magpie_launchers() -> Path | None:
    """The directory of the launchers the Magpie that runs the test ships, ``None`` when it cannot be found."""
    root = os.environ.get(_MAGPIE_PATH_ENV, "").strip()
    package = Path(root) / "Magpie" if root else _installed_package()
    directory = None if package is None else package / _LAUNCHERS
    return directory if directory is not None and directory.is_dir() else None


def _installed_package() -> Path | None:
    try:
        spec = importlib.util.find_spec("Magpie")
    except (ImportError, ValueError):
        return None
    places = list(spec.submodule_search_locations or ()) if spec is not None else []
    return Path(places[0]) if places else None


def _checkouts(bench: dict[str, Any]) -> list[Path]:
    """Where ``run_grid`` looks for the launcher: the point's own checkout, then the process environment's."""
    named = (str(bench.get("inferencex_path") or "").strip(), os.environ.get(_INFERENCEX_PATH_ENV, "").strip())
    return [Path(root) for root in dict.fromkeys(name for name in named if name)]


def _install(source: Path, target: Path) -> None:
    """Copy ``source`` over ``target`` unless it is already the same file, never exposing a half-written script."""
    if not source.is_file() or not target.parent.is_dir():
        return
    try:
        if target.is_file() and not target.is_symlink() and target.read_bytes() == source.read_bytes():
            return
        handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(source.read_bytes())
            os.chmod(name, _EXECUTABLE)
            os.replace(name, target)
        except OSError:
            Path(name).unlink(missing_ok=True)
            raise
    except OSError as exc:
        log.warning("the launcher %s cannot be put in the InferenceX checkout: %s", target.name, exc)
