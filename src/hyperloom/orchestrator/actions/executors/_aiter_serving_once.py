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
for this card's gfx and CU count, checks them with the preflight's coverage test and installs them into the image's
``aiter/jit`` with its manifest. A later test links them unless its own tables, build variables or patch touch them,
and a later preflight keeps them.

The preflight and the build hold a lock in this host's ``/tmp``, never on a shared mount, so a second session's preflight
waits and then finds the modules in place. A record there of each attempt keeps a build that failed from being tried
again on this container: its servers then build the modules themselves, as before.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.enablement.runtime.targeted_build import default_budget_sec

from ._aiter_jit import _tables_sha256, prepare_serving_so_for_csvs

__all__ = ["HOST_DIR", "PREWARM_MODULE", "Rebuild", "ServingPreflight", "prepare_serving_so_once"]

#: This host's own disk: the lock and the record of attempts must never sit on a mount other hosts share.
HOST_DIR = Path("/tmp/hyperloom-aiter-serving")
PREWARM_MODULE = "hyperloom.measure.runner.native.prewarm"
_LOCK, _ATTEMPTS = "preflight.lock", "attempts.json"
#: The prewarm's own build runs under the aiter recipe's budget (plus its margin); this bounds the process around it.
_PREWARM_TIMEOUT_SEC = default_budget_sec("aiter") + 15 * 60
_STDERR_TAIL = 400
_UTF8 = "utf-8"


@dataclass(frozen=True)
class Rebuild:
    """What became of building the moved-aside modules once: ``built`` were installed in ``seconds``, ``left`` are for
    the servers to build (aiter cannot build them by name), and ``why`` says why nothing was built, when nothing was."""

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
        rebuild = self.rebuild or Rebuild(why="not attempted")
        if rebuild.built:
            line += (
                f"; built {', '.join(rebuild.built)} once from the image's own tables in {rebuild.seconds:.0f} s and "
                f"installed them in {outcome.get('jit_dir')}, where every later test and preflight finds them"
            )
        else:
            line += f"; not built once ({rebuild.why}): each server builds them itself"
        if rebuild.left:
            line += f"; left to the servers: {', '.join(rebuild.left)}"
        return line


def prepare_serving_so_once(envs: dict[str, str], environ: Mapping[str, str]) -> ServingPreflight:
    """Run the preflight for a boot whose ``AITER_CONFIG_*`` are ``envs`` and whose servers run in ``environ``, and build
    what it moves aside once (see the module). Blocks while another preflight on this host holds the lock.

    Raises:
        OSError: The jit directory or this host's ``/tmp`` cannot be used.
    """
    with _host_lock():
        outcome = prepare_serving_so_for_csvs(envs, environ=environ)
        if outcome.get("action") != "invalidate":
            return ServingPreflight(outcome)
        return ServingPreflight(outcome, _rebuild_once(outcome, envs))


@contextmanager
def _host_lock() -> Iterator[None]:
    HOST_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(HOST_DIR / _LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _rebuild_once(outcome: Mapping[str, Any], envs: Mapping[str, str]) -> Rebuild:
    """Build the moved-aside modules of the image's own tables, unless this container tried them already."""
    variables = [name for name in outcome.get("variables") or () if not str(envs.get(name) or "").strip()]
    refusal = _refusal(outcome, variables)
    if refusal:
        return Rebuild(why=refusal)
    package, target = Path(outcome["package"]), outcome["targets"][0]
    attempt = {"jit_dir": outcome["jit_dir"], "target": target, "tables_sha256": _tables_sha256(package / "configs")}
    attempts = _read_attempts()
    tried = [earlier for earlier in attempts if all(earlier.get(key) == value for key, value in attempt.items())]
    done = {name for earlier in tried for name in earlier.get("variables") or ()}
    todo = [name for name in variables if name not in done]
    if not todo:
        return Rebuild(why=f"tried once on this container already, at {tried[-1].get('at')}: {tried[-1].get('result')}")
    gfx, cu_num = target.split("/")
    rebuild = _rebuild_of(_run_prewarm(package, todo, gfx, int(cu_num)))
    result = f"nothing built: {rebuild.why}" if rebuild.why else f"built {', '.join(rebuild.built)}"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_attempts([*attempts, {**attempt, "variables": todo, "result": result, "at": stamp}])
    return rebuild


def _refusal(outcome: Mapping[str, Any], variables: Sequence[str]) -> str:
    """Why the moved-aside modules of ``variables`` are not the image's own to build here; empty when they are."""
    targets = outcome.get("targets") or ()
    if not variables:
        return "the boot names its own table for every module it moved aside"
    if len(targets) != 1:
        return f"the servers build for {len(targets)} targets" if targets else "this card's target is unknown"
    if Path(outcome["jit_dir"]) != Path(outcome["package"]) / "jit":
        return f"the serving modules are not the image's own aiter/jit: {outcome['jit_dir']}"
    return ""


def _rebuild_of(report: Mapping[str, Any]) -> Rebuild:
    built, present = tuple(report.get("built") or ()), tuple(report.get("present") or ())
    why = str(report.get("failure") or "")
    if not why and not built:
        why = f"{', '.join(present)} in place already" if present else "aiter cannot build them by name"
    return Rebuild(built, tuple(report.get("left") or ()), float(report.get("seconds") or 0.0), why)


def _run_prewarm(package: Path, variables: Sequence[str], gfx: str, cu_num: int) -> dict[str, Any]:
    """The prewarm's report on building ``variables``' modules into ``package``; a report naming the failure when the
    process gave none."""
    tables = [part for name in variables for part in ("--table", name)]
    argv = [sys.executable, "-m", PREWARM_MODULE, "--gfx", gfx, "--cu-num", str(cu_num), "--package", str(package)]
    try:
        done = subprocess.run(
            [*argv, *tables], capture_output=True, text=True, timeout=_PREWARM_TIMEOUT_SEC, check=False
        )
    except subprocess.TimeoutExpired:
        return {"failure": f"the prewarm ran over {_PREWARM_TIMEOUT_SEC} s"}
    except OSError as exc:
        return {"failure": f"the prewarm could not start: {exc}"}
    try:
        report = json.loads(done.stdout)
    except ValueError:
        report = None
    if not isinstance(report, dict):
        tail = done.stderr.strip()[-_STDERR_TAIL:]
        return {"failure": f"the prewarm exited {done.returncode} with no report: {tail}"}
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
