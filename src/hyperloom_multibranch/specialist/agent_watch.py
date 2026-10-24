# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The watch over a session's agent sandboxes: a process of its own that ends every sandbox once the session's process
is gone, and kills a process of a sandbox that set a device mask reaching beyond the sandbox's cards.

A sandbox's processes carry :data:`MARKER`, set to the sandbox's root, whichever session or process group they moved
to, so the marker is what the watch and every kill select by. The session's process registers each sandbox's root and
cards in a directory of its own (:func:`watch`) and starts the watch with the first; the watch outlives a SIGKILL of
the session, which is how an agent kept running for an hour after its session died. The session's own kill terminates
the watch with its other children and SIGKILLs it seconds later, so a terminated watch ends the sandboxes at once.

The pod allows no device cgroup and no mount or user namespace, so the cards are confined by the environment:
``ROCR_VISIBLE_DEVICES`` names them and every HIP-level mask indexes into what ROCr exposes, so only a ROCr-level mask
a command sets itself can reach another card (:func:`beyond`). The watch reads each process's environment as it was
at exec, which a mask a process sets in its own memory, as Python's ``os.environ`` does, does not change, and a
command run with an emptied environment carries no marker either.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from hyperloom.common.proctree import TERM_GRACE_SEC
from hyperloom.common.visible_devices import ROCR_LEVEL_VARS, effective_mask_tokens

#: The environment variable every process of a sandbox carries, set to the sandbox's root.
MARKER = "HYPERLOOM_AGENT_SANDBOX"
#: SIGKILL skips sglang's cleanup and leaves its GPU memory held, so a kill waits this long after SIGTERM first.
GRACE_SECONDS = 30.0
#: The grace when the watch is terminated itself: the session's kill SIGKILLs the watch ``TERM_GRACE_SEC`` after its
#: SIGTERM, so the sandboxes must be gone well before that.
STOP_GRACE_SECONDS = TERM_GRACE_SEC / 3
POLL_SECONDS = 0.1
#: How often the watch looks: a command on a card it may not use has run at most this long.
WATCH_SECONDS = 1.0
#: The file of a sandbox's root the watch says what it killed in.
WATCH_LOG = "watch.log"
REGISTRY_PREFIX = "hyperloom-agent-watch-"

_watches: dict[int, tuple[subprocess.Popen[bytes], Path]] = {}


def watch(root: Path, cards: tuple[int, ...]) -> None:
    """Have this process's watch confine the processes of the sandbox at ``root`` to ``cards`` and end them once this
    process is gone, starting the watch when none runs for this process."""

    owner = os.getpid()
    running = _watches.get(owner)
    if running is None or running[0].poll() is not None:
        registry = Path(tempfile.gettempdir()) / f"{REGISTRY_PREFIX}{owner}"
        registry.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", __name__, str(owner), str(registry)]
        with (registry / "watch.err").open("ab") as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=err, stderr=err, start_new_session=True)
        running = _watches[owner] = (process, registry)
    entry = running[1] / f"{hashlib.sha256(str(root).encode()).hexdigest()[:16]}.json"
    written = entry.with_suffix(".tmp")
    written.write_text(json.dumps({"root": str(root), "cards": list(cards)}), encoding="utf-8")
    written.replace(entry)


def beyond(environ: Mapping[str, str], cards: Collection[int]) -> str:
    """The ROCr-level mask of ``environ`` that reaches a card outside ``cards``, as ``NAME=value``, or that it sets none
    and so reaches every card; empty when it reaches none. ROCr reads ``HSA_VISIBLE_DEVICES`` only when
    ``ROCR_VISIBLE_DEVICES`` is unset; an empty mask and a UUID name no card of ``cards``."""

    for name in ROCR_LEVEL_VARS:
        if name in environ:
            tokens = effective_mask_tokens(environ[name])
            within = tokens and all(token.isdigit() and int(token) in cards for token in tokens)
            return "" if within else f"{name}={environ[name]}"
    return f"no {ROCR_LEVEL_VARS[0]}"


def marked_pids(value: str) -> list[int]:
    """The live processes whose environment sets :data:`MARKER` to ``value``; a zombie, which has ended and only waits
    for its parent to reap it, shows no environment."""

    return [pid for pid, environ in _marked() if environ[MARKER] == value]


async def kill_marked(
    marker: str, grace_seconds: float, poll_seconds: float, reap: Callable[[], object] = lambda: None
) -> None:
    """SIGTERM every process carrying ``marker``, SIGKILL what is left after ``grace_seconds``; return once none is
    left. ``reap`` collects a child of this process that ended, which would otherwise linger as a zombie."""

    until = time.monotonic() + grace_seconds
    termed: set[int] = set()
    while True:
        reap()
        pids = marked_pids(marker)
        if not pids:
            return
        if time.monotonic() >= until:
            _signal(pids, signal.SIGKILL)
        else:
            _signal(set(pids) - termed, signal.SIGTERM)
            termed.update(pids)
        await asyncio.sleep(poll_seconds)


def _marked() -> Iterator[tuple[int, dict[str, str]]]:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "environ").read_bytes()
        except OSError:
            continue
        if f"{MARKER}=".encode() in raw:
            environ = dict(item.decode(errors="replace").partition("=")[::2] for item in raw.split(b"\0") if item)
            if MARKER in environ:
                yield int(entry.name), environ


def _registered(registry: Path) -> dict[str, tuple[int, ...]]:
    """The cards of every sandbox ``registry`` holds, by root."""

    sandboxes = {}
    for entry in registry.glob("*.json"):
        document = json.loads(entry.read_text(encoding="utf-8"))
        sandboxes[document["root"]] = tuple(document["cards"])
    return sandboxes


def _confine(sandboxes: Mapping[str, tuple[int, ...]]) -> None:
    """SIGKILL every process of ``sandboxes`` whose environment reaches beyond its sandbox's cards, at once: a server
    left to start would take memory on a card a measurement runs on."""

    for pid, environ in _marked():
        root = environ[MARKER]
        reaches = beyond(environ, sandboxes[root]) if root in sandboxes else ""
        if not reaches:
            continue
        said = f"killed by the sandbox: {reaches} reaches beyond its cards {list(sandboxes[root])}"
        _tell(pid, said)
        _signal([pid], signal.SIGKILL)
        with (Path(root) / WATCH_LOG).open("a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid {pid} {said}\n")


def _tell(pid: int, said: str) -> None:
    """Say ``said`` on the stderr of the process about to be killed, which is what its agent reads of the command; a
    killed process has no descriptors left to write to."""

    try:
        with open(f"/proc/{pid}/fd/2", "a", encoding="utf-8") as stderr:
            stderr.write(f"\n{said}\n")
    except OSError:
        return


async def _end(roots: Iterable[str], grace_seconds: float) -> None:
    await asyncio.gather(*(kill_marked(root, grace_seconds, POLL_SECONDS) for root in roots))


def main(argv: Sequence[str]) -> int:
    """Watch the sandboxes ``argv``'s registry holds while ``argv``'s owner, this process's parent, runs or until a
    SIGTERM, then end them all and remove the registry."""

    owner, registry = int(argv[0]), Path(argv[1])
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        grace = STOP_GRACE_SECONDS if _watch(owner, registry) else GRACE_SECONDS
        asyncio.run(_end(_registered(registry), grace))
    finally:
        shutil.rmtree(registry)
    return 0


def _watch(owner: int, registry: Path) -> bool:
    """Confine the sandboxes ``registry`` holds while ``owner`` runs; whether a SIGTERM ended the watch."""

    while os.getppid() == owner:
        _confine(_registered(registry))
        if signal.sigtimedwait([signal.SIGTERM], WATCH_SECONDS) is not None:
            return True
    return False


def _signal(pids: Iterable[int], number: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, number)
        except ProcessLookupError:
            continue


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


__all__ = [
    "GRACE_SECONDS",
    "MARKER",
    "POLL_SECONDS",
    "REGISTRY_PREFIX",
    "STOP_GRACE_SECONDS",
    "WATCH_LOG",
    "WATCH_SECONDS",
    "beyond",
    "kill_marked",
    "marked_pids",
    "watch",
]
