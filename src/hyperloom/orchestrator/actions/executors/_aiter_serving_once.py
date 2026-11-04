# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch preflight of aiter's serving modules under this host's lock, and the build once of what a caller whose
servers never build into the image moves aside.

:func:`~._aiter_jit.prepare_serving_so_for_csvs` moves a serving module aside when the tables a boot loads name kernels
the module lacks for this card, so that the next server builds it. :func:`prepare_serving_so_once` runs it under a lock
in this host's ``/tmp``, never on a shared mount, so a second session's preflight, or the GEMM tuning lane
(:func:`under_host_lock`), waits for it with no bound (and says for whom); when that directory cannot be used the
preflight runs unlocked. The launch of a boot builds nothing here: its servers build what was moved aside, as before.

A caller whose servers each build into a directory of their own (the measure runner gives each test one, with links to
the image's modules) would have every test build a moved-aside module again, as no server's build reaches the image.
It calls :func:`build_image_own_once`, which checks the image's own tables for this card -- what a server loads and
builds for when nothing sets an ``AITER_CONFIG_*``, ``GPU_ARCHS`` or ``CU_NUM`` -- and never a test's: no server would
put back a module moved aside for a test's own tables or target, and such a test is not given a module they make stale
anyway. It builds what it moved aside once, under the same lock: ``build`` builds them on CPU from the image's tree and
tables for this card's gfx and CU count as ``rocminfo`` reports them, checks them with the preflight's coverage test and
installs them into the image's ``aiter/jit`` with the prewarm manifest. A later test links them unless its own tables,
build variables or patch touch them, and a later preflight keeps them. When the card's target is unknown it checks
nothing: every row would count, and a module moved aside for another GPU's rows would be built by no one. A record in
the lock's directory of each attempt keeps a build that failed from being tried again on this container: its servers
then build the modules themselves.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from . import _aiter_jit
from ._aiter_jit import AITER_ENV_TO_SERVING_MODULES, prepare_serving_so_for_csvs, tuned_tables_sha256

__all__ = [
    "HOST_DIR",
    "ModuleBuild",
    "Rebuild",
    "ServingPreflight",
    "build_image_own_once",
    "prepare_serving_so_once",
    "under_host_lock",
]

log = logging.getLogger(__name__)

#: This host's own disk: the lock and the record of attempts must never sit on a mount other hosts share.
HOST_DIR = Path("/tmp/hyperloom-aiter-serving")
_LOCK, _ATTEMPTS = "preflight.lock", "attempts.json"
_HOLDER_BYTES = 4096
_UTF8 = "utf-8"
_T = TypeVar("_T")

#: ``build(package, variables, gfx, cu_num) -> report``: builds into the aiter at ``package`` the modules of the tables
#: of the ``AITER_CONFIG_*`` ``variables`` for ``gfx`` and ``cu_num``, installs them with the prewarm manifest and
#: reports ``{"built", "present", "left", "failure", "seconds"}``; it never raises for a build that fails.
ModuleBuild = Callable[[Path, Sequence[str], str, int], Mapping[str, Any]]


@dataclass(frozen=True)
class Rebuild:
    """What became of building the moved-aside modules once: ``built`` were installed in ``seconds``, ``left`` are the
    modules moved aside that the servers build (aiter cannot build them by name, or nothing was built), and ``why`` says
    why nothing was built, when nothing was."""

    built: tuple[str, ...] = ()
    left: tuple[str, ...] = ()
    seconds: float = 0.0
    why: str = ""


@dataclass(frozen=True)
class ServingPreflight:
    """The preflight's outcome (:func:`~._aiter_jit.prepare_serving_so_for_csvs`), and the build of what it moved aside,
    ``None`` when it moved nothing."""

    outcome: dict[str, Any]
    rebuild: Rebuild | None = None

    def summary(self) -> str:
        """One line on what the preflight decided, for the log of the boot it ran for."""
        outcome = self.outcome
        targets = ", ".join(outcome.get("targets") or ()) or "every GPU (this card's target is not known)"
        if outcome.get("action") == "noop":
            return f"nothing checked: {outcome.get('reason')}"
        if outcome.get("action") == "skip":
            kept = outcome.get("kept_image_own")
            if kept:
                return (
                    f"nothing moved: the serving .so covers the CSVs this boot loads for {targets} but in "
                    f"{', '.join(kept)}, which the prewarm built from these same tables"
                )
            return f"the serving .so covers the CSVs this boot loads for {targets}; nothing moved"
        removed = _moved(outcome)
        line = (
            f"the CSVs this boot loads outran the serving .so for {targets}; moved {len(removed)} module(s) aside to "
            f"{outcome.get('backup')}: {', '.join(removed)}"
        )
        rebuild = self.rebuild or Rebuild(left=removed, why="not attempted")
        if rebuild.built:
            line += (
                f"; built {', '.join(rebuild.built)} once from the image's own tables in {rebuild.seconds:.0f} s and "
                f"installed them in {outcome.get('jit_dir')}, where every later test and preflight finds them"
            )
        else:
            line += f"; not built once ({rebuild.why})"
        if rebuild.left:
            line += f"; left for the servers to build: {', '.join(rebuild.left)} (each server builds them itself)"
        return line


def prepare_serving_so_once(envs: dict[str, str], environ: Mapping[str, str]) -> ServingPreflight:
    """Run the preflight for a boot whose ``AITER_CONFIG_*`` are ``envs`` and whose servers run in ``environ``, leaving
    what it moves aside to the boot's servers. Blocks while another preflight on this host holds the lock.

    Raises:
        OSError: The jit directory cannot be used.
    """
    with _host_lock("the launch preflight of a boot"):
        outcome = prepare_serving_so_for_csvs(envs, environ=environ)
    if outcome.get("action") != "invalidate":
        return ServingPreflight(outcome)
    return ServingPreflight(outcome, Rebuild(left=_moved(outcome), why="the launch leaves them to its servers"))


def build_image_own_once(build: ModuleBuild) -> ServingPreflight:
    """Run the preflight on the image's own tables for this card, and build what it moves aside once for this container
    with ``build`` (see the module); nothing is checked when the card's target is unknown. Blocks while another
    preflight on this host holds the lock.

    Raises:
        OSError: The jit directory cannot be used.
    """
    card = _aiter_jit.card_target()
    if card is None:
        return ServingPreflight({"action": "noop", "reason": "this card's target is unknown"})
    with _host_lock("the prewarm of the image's own serving modules") as lock:
        outcome = prepare_serving_so_for_csvs({}, environ={})
        if outcome.get("action") != "invalidate":
            return ServingPreflight(outcome)
        if lock is None:
            return ServingPreflight(outcome, Rebuild(left=_moved(outcome), why="this host's lock cannot be used"))
        return ServingPreflight(outcome, _rebuild_once(outcome, lock, build, card))


def _moved(outcome: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(Path(path).stem for path in outcome.get("removed") or ())


def under_host_lock(doing: str, call: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """``call(*args, **kwargs)`` under this host's preflight lock, saying it is ``doing`` to a preflight that waits; unlocked
    (with a warning) when the lock cannot be used. For a caller that moves serving modules aside outside the launch."""
    with _host_lock(doing):
        return call(*args, **kwargs)


class _HostLock:
    """This host's preflight lock, held through ``descriptor``; the file holds what the holder said it is doing."""

    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor

    def doing(self, what: str) -> None:
        """Say what the holder is doing, for a preflight that waits."""
        said = json.dumps({"pid": os.getpid(), "since": _now(), "doing": what}).encode(_UTF8)
        try:
            os.ftruncate(self.descriptor, 0)
            os.pwrite(self.descriptor, said, 0)
        except OSError:  # what the holder does is only told to a waiter; the lock itself is held
            return

    def release(self) -> None:
        """Take back what the holder said, and let the lock go with the descriptor."""
        with suppress(OSError):  # what the holder said outlives it at worst; the lock goes all the same
            os.ftruncate(self.descriptor, 0)
        os.close(self.descriptor)


@contextmanager
def _host_lock(doing: str) -> Iterator[_HostLock | None]:
    """Hold this host's preflight lock, waiting for it with a warning that names its holder; ``None`` (with a warning)
    when it cannot be used, and the caller then runs unlocked."""
    lock = _acquire(doing)
    try:
        yield lock
    finally:
        if lock is not None:
            lock.release()


def _acquire(doing: str) -> _HostLock | None:
    path = HOST_DIR / _LOCK
    try:
        HOST_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        log.warning(
            "aiter serving preflight: this host's lock %s cannot be used (%s); %s runs unlocked", path, exc, doing
        )
        return None
    try:
        _wait_for(descriptor, path, doing)
        lock = _HostLock(descriptor)
        lock.doing(doing)
    except OSError as exc:
        os.close(descriptor)
        log.warning(
            "aiter serving preflight: this host's lock %s cannot be held (%s); %s runs unlocked", path, exc, doing
        )
        return None
    except BaseException:
        os.close(descriptor)
        raise
    return lock


def _wait_for(descriptor: int, path: Path, doing: str) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass
    log.warning(
        "aiter serving preflight: %s waits for this host's lock %s, held by %s", doing, path, _holder(descriptor)
    )
    started = time.monotonic()
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    log.warning("aiter serving preflight: %s got this host's lock after %.0f s", doing, time.monotonic() - started)


def _holder(descriptor: int) -> str:
    """Who holds the lock and what it does, as it said so; a plain word when it has said nothing yet."""
    try:
        said = json.loads(os.pread(descriptor, _HOLDER_BYTES, 0).decode(_UTF8))
    except (OSError, ValueError):
        said = None
    if not isinstance(said, dict):
        return "a process that has not said what it does"
    return f"pid {said.get('pid')} since {said.get('since')}: {said.get('doing')}"


def _rebuild_once(outcome: Mapping[str, Any], lock: _HostLock, build: ModuleBuild, card: tuple[str, int]) -> Rebuild:
    """Build the moved-aside modules of the image's own tables for this card, unless this container tried them already."""
    moved, variables = _moved(outcome), list(outcome.get("variables") or ())
    if Path(outcome["jit_dir"]) != Path(outcome["package"]) / "jit":
        return Rebuild(left=moved, why=f"the serving modules are not the image's own aiter/jit: {outcome['jit_dir']}")
    package, target = Path(outcome["package"]), _target_name(card)
    attempt = {
        "jit_dir": outcome["jit_dir"],
        "target": target,
        "tables_sha256": tuned_tables_sha256(package / "configs"),
    }
    attempts = _read_attempts()
    tried = _tried(attempts, attempt)
    todo = [name for name in variables if name not in tried]
    if not todo:
        return Rebuild(left=moved, why=f"tried once on this container already, {tried[variables[0]]}")
    _announce(lock, [name for name in moved if _read_by(name, todo)], todo, target)
    # Recorded before the build: a build that raises or never returns is not tried again by every later test either.
    started = {**attempt, "variables": todo, "result": "started, and did not end", "at": _now()}
    _write_attempts([*attempts, started])
    rebuild = _rebuild_of(build(package, todo, *card), moved)
    result = f"nothing built: {rebuild.why}" if rebuild.why else f"built {', '.join(rebuild.built)}"
    _write_attempts([*attempts, {**started, "result": result, "at": _now()}])
    return rebuild


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tried(attempts: Sequence[Mapping[str, Any]], attempt: Mapping[str, str]) -> dict[str, str]:
    """The variables this container tried to build for the jit directory, target and tables of ``attempt``, each with
    when and with what result it was last tried."""
    same = [earlier for earlier in attempts if all(earlier.get(key) == value for key, value in attempt.items())]
    return {
        name: f"at {earlier.get('at')}: {earlier.get('result')}"
        for earlier in same
        for name in earlier.get("variables") or ()
    }


def _read_by(module: str, variables: Sequence[str]) -> bool:
    return any(module in AITER_ENV_TO_SERVING_MODULES.get(variable, ()) for variable in variables)


def _announce(lock: _HostLock, building: Sequence[str], variables: Sequence[str], target: str) -> None:
    """Say, in the log and to a preflight that waits for the lock, that the build of ``building`` starts."""
    lock.doing(f"building {', '.join(building)} for {target} (the prewarm)")
    log.warning(
        "aiter serving preflight: building %s once for %s before this boot (the prewarm of %s)",
        ", ".join(building),
        target,
        ", ".join(variables),
    )


def _target_name(card: tuple[str, int]) -> str:
    return f"{card[0]}/{card[1]}"


def _rebuild_of(report: Mapping[str, Any], moved: Sequence[str]) -> Rebuild:
    built, present = tuple(report.get("built") or ()), tuple(report.get("present") or ())
    why = str(report.get("failure") or "")
    if not why and not built:
        why = f"{', '.join(present)} in place already" if present else "aiter cannot build them by name"
    unbuilt = dict.fromkeys([*moved, *(report.get("left") or ())])
    left = tuple(name for name in unbuilt if name not in built and name not in present)
    return Rebuild(built, left, float(report.get("seconds") or 0.0), why)


def _read_attempts() -> list[dict[str, Any]]:
    try:
        document = json.loads((HOST_DIR / _ATTEMPTS).read_text(encoding=_UTF8))
    except (OSError, ValueError):
        return []
    return [entry for entry in document if isinstance(entry, dict)] if isinstance(document, list) else []


def _write_attempts(attempts: list[dict[str, Any]]) -> None:
    pending = HOST_DIR / f".{_ATTEMPTS}.{os.getpid()}"
    pending.write_text(json.dumps(attempts, indent=2, sort_keys=True), encoding=_UTF8)
    os.replace(pending, HOST_DIR / _ATTEMPTS)
