# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch preflight of aiter's serving modules, with what it moves aside built once per container.

:func:`~._aiter_jit.prepare_serving_so_for_csvs` moves a serving module aside when the tables a boot loads name kernels
the module lacks for this card, so that the next server builds it. Where every server builds into a directory of its own
(the measure runner gives each test one, with links to the image's modules), no server's build reaches the image, and
every test that calls the module builds it again.

:func:`prepare_serving_so_once` builds the modules it moved aside whose tables are the image's own (the boot leaves
their ``AITER_CONFIG_*`` unset), once, before the boot: the prewarm (``python -m hyperloom.measure.runner.native.prewarm``,
a process of its own because the measure package imports this one) builds them on CPU from the image's tree and tables
for this card's gfx and CU count as ``rocminfo`` reports them (what a test's server builds for), under the environment a
test's server has when nothing of the test sets a build variable, checks them with the preflight's coverage test and
installs them into the image's ``aiter/jit`` with its manifest. A later test links them unless its own tables, build
variables or patch touch them, and a later preflight keeps them. A boot whose own servers build for another target
(``GPU_ARCHS`` or ``CU_NUM`` in its environment) builds nothing once: a module for either target is wrong for the other.

The preflight and the build hold a lock in this host's ``/tmp``, never on a shared mount, so a second session's preflight
waits (and says for whom) and then finds the modules in place; when that directory cannot be used the preflight runs
unlocked and builds nothing. A record there of each attempt keeps a build that failed from being tried again on this
container: its servers then build the modules themselves, as before.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, TypeVar

from hyperloom.orchestrator.enablement.runtime.targeted_build import default_budget_sec

from . import _aiter_jit
from ._aiter_jit import AITER_ENV_TO_SERVING_MODULES, _tables_sha256, prepare_serving_so_for_csvs

__all__ = ["HOST_DIR", "PREWARM_MODULE", "Rebuild", "ServingPreflight", "prepare_serving_so_once", "under_host_lock"]

log = logging.getLogger(__name__)

#: This host's own disk: the lock and the record of attempts must never sit on a mount other hosts share.
HOST_DIR = Path("/tmp/hyperloom-aiter-serving")
PREWARM_MODULE = "hyperloom.measure.runner.native.prewarm"
_LOCK, _ATTEMPTS = "preflight.lock", "attempts.json"
#: The prewarm's own build runs under the aiter recipe's budget (plus its margin); this bounds the process around it.
_PREWARM_TIMEOUT_SEC = default_budget_sec("aiter") + 15 * 60
#: How long the readers of a prewarm that ended get to drain its pipes; a build process it left behind may hold them.
_DRAIN_SEC = 30
_STDERR_TAIL = 400
_STDERR_LINES = 40
#: The prewarm logs as ``<LEVEL> <logger>: <message>``; its warnings (the start of its build among them) are relayed.
_RELAYED = "WARNING "
_HOLDER_BYTES = 4096
_UTF8 = "utf-8"
_T = TypeVar("_T")


@dataclass(frozen=True)
class Rebuild:
    """What became of building the moved-aside modules once: ``built`` were installed in ``seconds``, ``left`` are the
    modules moved aside that the servers build (aiter cannot build them by name, the boot names their table, or nothing
    was built), and ``why`` says why nothing was built, when nothing was."""

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
        removed = [Path(path).stem for path in outcome.get("removed") or ()]
        line = (
            f"the CSVs this boot loads outran the serving .so for {targets}; moved {len(removed)} module(s) aside to "
            f"{outcome.get('backup')}: {', '.join(removed)}"
        )
        rebuild = self.rebuild or Rebuild(left=tuple(removed), why="not attempted")
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
    """Run the preflight for a boot whose ``AITER_CONFIG_*`` are ``envs`` and whose servers run in ``environ``, and build
    what it moves aside once (see the module). Blocks while another preflight on this host holds the lock.

    Raises:
        OSError: The jit directory cannot be used.
    """
    with _host_lock("the launch preflight of a boot") as lock:
        outcome = prepare_serving_so_for_csvs(envs, environ=environ)
        if outcome.get("action") != "invalidate":
            return ServingPreflight(outcome)
        moved = tuple(Path(path).stem for path in outcome.get("removed") or ())
        if lock is None:
            return ServingPreflight(outcome, Rebuild(left=moved, why="this host's lock cannot be used"))
        return ServingPreflight(outcome, _rebuild_once(outcome, envs, moved, lock))


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
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        said = json.dumps({"pid": os.getpid(), "since": stamp, "doing": what}).encode(_UTF8)
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


def _rebuild_once(
    outcome: Mapping[str, Any], envs: Mapping[str, str], moved: Sequence[str], lock: _HostLock
) -> Rebuild:
    """Build the moved-aside modules of the image's own tables for this card, unless this container tried them already."""
    variables = [name for name in outcome.get("variables") or () if not str(envs.get(name) or "").strip()]
    card = _aiter_jit.card_target()
    refusal = _refusal(outcome, variables, card)
    if refusal or card is None:
        return Rebuild(left=tuple(moved), why=refusal)
    package, target = Path(outcome["package"]), _target_name(card)
    attempt = {"jit_dir": outcome["jit_dir"], "target": target, "tables_sha256": _tables_sha256(package / "configs")}
    attempts = _read_attempts()
    tried = _tried(attempts, attempt)
    todo = [name for name in variables if name not in tried]
    if not todo:
        return Rebuild(left=tuple(moved), why=f"tried once on this container already, {tried[variables[0]]}")
    _announce(lock, [name for name in moved if _read_by(name, todo)], todo, target)
    rebuild = _rebuild_of(_run_prewarm(package, todo, *card), moved)
    result = f"nothing built: {rebuild.why}" if rebuild.why else f"built {', '.join(rebuild.built)}"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_attempts([*attempts, {**attempt, "variables": todo, "result": result, "at": stamp}])
    return rebuild


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
    lock.doing(f"building {', '.join(building)} for {target} (the prewarm, at most {_PREWARM_TIMEOUT_SEC} s)")
    log.warning(
        "aiter serving preflight: building %s once for %s before this boot (the prewarm of %s, at most %d s)",
        ", ".join(building),
        target,
        ", ".join(variables),
        _PREWARM_TIMEOUT_SEC,
    )


def _target_name(card: tuple[str, int]) -> str:
    return f"{card[0]}/{card[1]}"


def _refusal(outcome: Mapping[str, Any], variables: Sequence[str], card: tuple[str, int] | None) -> str:
    """Why the moved-aside modules of ``variables`` are not the image's own to build here for the card ``card``; empty
    when they are."""
    targets = outcome.get("targets") or ()
    if not variables:
        return "the boot names its own table for every module it moved aside"
    if len(targets) != 1:
        return f"the servers build for {len(targets)} targets" if targets else "this card's target is unknown"
    if card is None:
        return "this card's target is unknown"
    if targets[0] != _target_name(card):
        return (
            f"this boot's servers build for {targets[0]} (GPU_ARCHS or CU_NUM in its environment) and a test's for "
            f"the card's {_target_name(card)}"
        )
    if Path(outcome["jit_dir"]) != Path(outcome["package"]) / "jit":
        return f"the serving modules are not the image's own aiter/jit: {outcome['jit_dir']}"
    return ""


def _rebuild_of(report: Mapping[str, Any], moved: Sequence[str]) -> Rebuild:
    built, present = tuple(report.get("built") or ()), tuple(report.get("present") or ())
    why = str(report.get("failure") or "")
    if not why and not built:
        why = f"{', '.join(present)} in place already" if present else "aiter cannot build them by name"
    unbuilt = dict.fromkeys([*moved, *(report.get("left") or ())])
    left = tuple(name for name in unbuilt if name not in built and name not in present)
    return Rebuild(built, left, float(report.get("seconds") or 0.0), why)


def _run_prewarm(package: Path, variables: Sequence[str], gfx: str, cu_num: int) -> dict[str, Any]:
    """The prewarm's report on building ``variables``' modules into ``package`` for ``gfx`` and ``cu_num``; a report
    naming the failure when the process gave none. Its warnings are logged as they come."""
    tables = [part for name in variables for part in ("--table", name)]
    argv = [sys.executable, "-m", PREWARM_MODULE, "--gfx", gfx, "--cu-num", str(cu_num), "--package", str(package)]
    try:
        child = subprocess.Popen([*argv, *tables], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        return {"failure": f"the prewarm could not start: {exc}"}
    stdout: list[str] = []
    stderr: deque[str] = deque(maxlen=_STDERR_LINES)
    readers = [
        threading.Thread(target=lambda: stdout.append(child.stdout.read()), daemon=True),  # type: ignore[union-attr]
        threading.Thread(target=_relay, args=(child.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        child.wait(timeout=_PREWARM_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        return {"failure": f"the prewarm ran over {_PREWARM_TIMEOUT_SEC} s"}
    for reader in readers:
        reader.join(timeout=_DRAIN_SEC)
    return _report("".join(stdout), child.returncode, stderr)


def _relay(stream: IO[str], tail: deque[str]) -> None:
    """Keep the prewarm's last lines of stderr, and log each warning it gives (the start of its build, with what it
    builds and its estimate, among them) as it comes."""
    for line in stream:
        text = line.rstrip("\n")
        tail.append(text)
        if text.startswith(_RELAYED):
            log.warning("aiter serving preflight: the prewarm says %s", text.removeprefix(_RELAYED))


def _report(stdout: str, returncode: int, stderr: Sequence[str]) -> dict[str, Any]:
    try:
        report = json.loads(stdout)
    except ValueError:
        report = None
    if not isinstance(report, dict):
        tail = "\n".join(stderr).strip()[-_STDERR_TAIL:]
        return {"failure": f"the prewarm exited {returncode} with no report: {tail}"}
    return report


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
