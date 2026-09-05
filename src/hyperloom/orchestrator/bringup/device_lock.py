# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A host-level claim on the GPUs a serving process is about to use.

One ``flock``-held lock file per device, in one directory per host, since every
other exclusion in the session is per-session. The descriptor is inherited by
the serving process rather than held by the coordinator, so the claim outlives a
coordinator that dies with an orphaned server still on the cards. Within one
process it is reference-counted.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections.abc import Mapping
from pathlib import Path

log = logging.getLogger(__name__)

try:  # POSIX runtime (Linux): the only place this mechanism exists.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX dev hosts
    fcntl = None  # type: ignore[assignment]

#: Where the per-device lock files live. One variable rather than a search
#: order: optimizers that disagree on the path exclude nothing.
DEVICE_LOCK_DIR_ENV = "HYPERLOOM_DEVICE_LOCK_DIR"

#: Set to ``0``/``false`` to launch without claiming anything.
DEVICE_LOCK_ENABLE_ENV = "HYPERLOOM_DEVICE_LOCK"

#: The default lock directory. Under ``/var/tmp`` so it outlives a tmpfs
#: cleanup, and outside any session directory so another session can see it.
DEFAULT_LOCK_DIR = "/var/tmp/hyperloom/device-locks"  # noqa: S108 — a fixed host path all optimizers must share

#: The variables that decide which cards a launch will see. ROCm honours the
#: first of these that is set, and the preflight drops ``HIP_VISIBLE_DEVICES``
#: when ``ROCR_VISIBLE_DEVICES`` is set, so the order matches
#: :func:`hyperloom.orchestrator.bus.gpu_pool._visible_device_mask`.
_VISIBLE_DEVICE_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

#: The device nodes a process can open, which is what a container restricted to
#: some of the host's cards actually sees.
_RENDER_NODE_DIR = Path("/dev/dri")
_RENDER_NODE_PREFIX = "renderD"

__all__ = [
    "DeviceClaim",
    "DevicesBusy",
    "claim_devices",
    "present_devices",
    "visible_devices",
]

#: Descriptors held by this process, by device token, with a use count. Guarded
#: by ``_LOCK`` because launches happen on worker threads.
_HELD: dict[str, list] = {}
_LOCK = threading.Lock()


class DevicesBusy(RuntimeError):
    """Raised when another process on this host already holds a device.

    ``devices`` are the tokens that could not be claimed and ``holder`` what
    their lock file says about who holds them.
    """

    def __init__(self, devices: tuple[str, ...], holder: dict):
        """Record the contended devices and the owner note read off their lock."""
        self.devices = tuple(devices)
        self.holder = dict(holder)
        who = f" (held by pid={self.holder.get('pid')} session={self.holder.get('session')})" if self.holder else ""
        super().__init__(f"GPUs {', '.join(self.devices)} are claimed by another process on this host{who}")


def present_devices() -> tuple[str, ...]:
    """Return the render nodes this process can open, in device order.

    Returns:
        tuple[str, ...]: One node name per card, empty when there is no device
        directory to read. A node name is the same on every process that can see
        that card, which an index into a mask is not, so it is what a claim is
        named by.
    """
    try:
        entries = list(_RENDER_NODE_DIR.iterdir())
    except OSError:
        return ()
    minors = sorted(
        int(entry.name[len(_RENDER_NODE_PREFIX) :])
        for entry in entries
        if entry.name.startswith(_RENDER_NODE_PREFIX) and entry.name[len(_RENDER_NODE_PREFIX) :].isdigit()
    )
    return tuple(f"{_RENDER_NODE_PREFIX}{minor}" for minor in minors)


def visible_devices(env: Mapping[str, str] | None) -> tuple[str, ...]:
    """Return the device tokens a launch with this environment will see.

    A mask that is set decides the answer even when it is empty, which means the
    launch sees no card at all. An unmasked launch sees every card this process
    can open, so two sessions each given disjoint cards by their container claim
    disjoint tokens instead of both claiming the host.

    Args:
        env: The launch environment; ``None`` reads the current process's.

    Returns:
        tuple[str, ...]: One token per card, empty when the launch will see none.
    """
    environ = os.environ if env is None else env
    present = present_devices()
    for name in _VISIBLE_DEVICE_VARS:
        raw = environ.get(name)
        if raw is None:
            continue
        return tuple(sorted({_device_token(part, present) for part in str(raw).split(",") if part.strip()}))
    return present


def _device_token(masked: str, present: tuple[str, ...]) -> str:
    """Name the card a mask entry selects.

    A mask indexes the cards the process can see, so an index resolves to that
    card's node name; anything else (a UUID, an index naming a card that is not
    here) stands for itself.
    """
    token = masked.strip()
    if token.isdigit() and int(token) < len(present):
        return present[int(token)]
    return token


class DeviceClaim:
    """The descriptors a launch must inherit to keep its claim alive.

    ``devices`` are the tokens claimed and ``fds`` the descriptors to pass to
    the child, empty when nothing was claimed.
    """

    def __init__(self, devices: tuple[str, ...], fds: tuple[int, ...]):
        """Record the claimed tokens and the inheritable descriptors holding them."""
        self.devices = devices
        self.fds = fds

    def release(self) -> None:
        """Drop this process's share of the claim.

        The lock outlives this call whenever a child inherited the descriptor.
        """
        with _LOCK:
            for token in self.devices:
                entry = _HELD.get(token)
                if entry is None:
                    continue
                entry[1] -= 1
                if entry[1] > 0:
                    continue
                _HELD.pop(token, None)
                os.close(entry[0])

    def __enter__(self) -> "DeviceClaim":
        """Return self, for use as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release this process's share."""
        self.release()


def claim_devices(
    env: Mapping[str, str] | None,
    *,
    session_dir: Path | str = "",
    lock_dir: Path | str = "",
) -> DeviceClaim:
    """Claim the cards a launch with ``env`` will use.

    Args:
        env: The launch environment, read for the visible-device variables.
        session_dir: Recorded in the lock body so a contended card names a
            session and not just a pid.
        lock_dir: Where the locks live; read from the environment when empty.

    Returns:
        DeviceClaim: The claim, whose ``fds`` the caller must let the serving
        process inherit. A host that is disabled, non-POSIX, cannot write the
        lock directory, or shows the launch no card at all yields an empty claim
        rather than refusing to launch.

    Raises:
        DevicesBusy: When another process on this host holds one of the cards.
    """
    environ = os.environ if env is None else env
    if str(environ.get(DEVICE_LOCK_ENABLE_ENV, "1")).strip().lower() in {"0", "false", "no", "off"}:
        return DeviceClaim((), ())
    if fcntl is None:
        return DeviceClaim((), ())
    directory = Path(str(lock_dir) or str(environ.get(DEVICE_LOCK_DIR_ENV, "") or DEFAULT_LOCK_DIR))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.warning("device lock: %s is unusable; launching without a host claim", directory, exc_info=True)
        return DeviceClaim((), ())
    devices = visible_devices(environ)
    taken: list[str] = []
    fds: list[int] = []
    try:
        for token in devices:
            fd = _claim_one(directory, token, session_dir=str(session_dir))
            taken.append(token)
            fds.append(fd)
    except DevicesBusy:
        DeviceClaim(tuple(taken), tuple(fds)).release()
        raise
    except OSError:
        log.warning("device lock: could not claim %s; launching without a host claim", devices, exc_info=True)
        DeviceClaim(tuple(taken), tuple(fds)).release()
        return DeviceClaim((), ())
    return DeviceClaim(tuple(taken), tuple(fds))


def _claim_one(directory: Path, token: str, *, session_dir: str) -> int:
    """Return the inheritable descriptor holding this process's claim on one
    device, taking it when this process does not already hold it.

    Raises:
        DevicesBusy: When another process holds it.
        OSError: When the lock file cannot be opened.
    """
    with _LOCK:
        entry = _HELD.get(token)
        if entry is not None:
            entry[1] += 1
            return int(entry[0])
        path = directory / f"gpu-{token}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        # Python opens every descriptor close-on-exec, which would drop the
        # claim the moment the serving process replaced its image.
        os.set_inheritable(fd, True)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = _read_holder(path)
            os.close(fd)
            raise DevicesBusy((token,), holder) from exc
        _write_holder(fd, token=token, session_dir=session_dir)
        _HELD[token] = [fd, 1]
        return fd


def _write_holder(fd: int, *, token: str, session_dir: str) -> None:
    """Write the owner note onto the locked descriptor, for whoever is refused by it."""
    body = json.dumps(
        {"pid": os.getpid(), "hostname": socket.gethostname(), "device": token, "session": session_dir},
        sort_keys=True,
    )
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, (body + "\n").encode("utf-8"))


def _read_holder(path: Path) -> dict:
    """Read a lock file's owner note, or ``{}`` when there is none to read.

    The note is diagnostics on the refusal message; the claim itself is the lock.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}
