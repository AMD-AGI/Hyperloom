# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Set the GPU power settings a session runs under, and put them back.

Only reached with ``--apply-gpu-power-settings``; :mod:`hyperloom.common.gpu_power_settings` stays read-only.
:class:`PowerSettingsLease` makes setting a card safe to leave unattended: the card's originals are written to a
per-card record before anything is set, and the record is held under an exclusive ``flock`` for the session's
lifetime. A second session cannot set the same card while it is held, and a record whose lock is free belongs to a
session that died without restoring, so the next launch restores it first.

Each record is stamped with the host's boot id. Records live on host-local storage, so another host never sees them;
the boot id retires a record written before a reboot, which reset the card.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import stat
import subprocess  # nosec B404 - fixed amd-smi invocations.
from pathlib import Path
from typing import Any, Callable, Mapping

from hyperloom.common.gpu_power_settings import (
    GpuPowerSettingsError,
    _gpu_rows,
    _watts,
    normalize_perf_level,
)

__all__ = [
    "GPU_POWER_LEDGER_DIR",
    "PowerSettingsLease",
    "apply_gpu_power_settings",
    "current_boot_id",
    "orphaned_power_records",
    "read_resident_vram_mb",
    "restore_gpu_power_settings",
]

#: Host-local, outside any workspace or runtime directory a job can redirect: every session on the host has to find
#: the same records, whatever its ``HYPERLOOM_RUNTIME_DIR`` or ``USER_DATA_PATH``.
GPU_POWER_LEDGER_DIR = Path("/var/tmp/hyperloom/gpu_power_settings")

_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_POWER_CAP_W = "power_cap_w"
_PERF_LEVEL = "perf_level"
_SET = "set"
_BOOT_ID = "boot_id"
_UTF8 = "utf-8"


def current_boot_id(path: Path = _BOOT_ID_PATH) -> str:
    """This boot's id, or ``""`` when the kernel does not expose one."""
    try:
        return path.read_text(encoding=_UTF8).strip()
    except OSError:
        return ""


def _run_amd_smi(run: Callable[..., subprocess.CompletedProcess], args: list[str], timeout: float) -> str:
    try:
        done = run(["amd-smi", *args], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
    if done.returncode != 0:
        # amd-smi prints a traceback on failure; its last line is the one that says why.
        lines = [line.strip() for line in (done.stderr or done.stdout or "").splitlines() if line.strip()]
        detail = (lines[-1] if lines else "")[:200]
        raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} exited {done.returncode}: {detail}")
    return done.stdout


def read_resident_vram_mb(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> dict[int, float]:
    """``{gpu: used VRAM in MB}``; how a launch tells a card someone else has a model resident on."""
    try:
        payload = json.loads(_run_amd_smi(run, ["metric", "--mem-usage", "--json"], timeout))
    except ValueError as exc:
        raise GpuPowerSettingsError("amd-smi metric --mem-usage printed no JSON") from exc
    out: dict[int, float] = {}
    for row in _gpu_rows(payload):
        mem = row.get("mem_usage") if isinstance(row.get("mem_usage"), dict) else {}
        used = _watts(mem.get("used_vram"))
        if used is not None:
            out[row["gpu"]] = used
    return out


def _set_commands(settings: Mapping[int, Mapping[str, Any]]) -> list[list[str]]:
    """``amd-smi set`` argument lists, one per distinct value, so cards sharing a value are set in one call."""
    by_cap: dict[float, list[int]] = {}
    by_level: dict[str, list[int]] = {}
    for gpu, row in sorted(settings.items()):
        cap = row.get(_POWER_CAP_W)
        if _is_number(cap):
            by_cap.setdefault(float(cap), []).append(gpu)
        level = normalize_perf_level(row.get(_PERF_LEVEL))
        if level:
            by_level.setdefault(level, []).append(gpu)
    commands = [[_SET, "-g", *map(str, gpus), "-o", "ppt0", f"{cap:g}"] for cap, gpus in by_cap.items()]
    commands += [[_SET, "-g", *map(str, gpus), "-l", level.upper()] for level, gpus in by_level.items()]
    return commands


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def restore_gpu_power_settings(
    originals: Mapping[int, Mapping[str, Any]],
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 60.0,
) -> list[str]:
    """Put each card back to its recorded cap and perf level; the problems, empty when every set succeeded.

    Every command is attempted even after one fails: a card left un-restored because an earlier card's set failed is
    the outcome this exists to prevent.
    """
    problems: list[str] = []
    for args in _set_commands(originals):
        try:
            _run_amd_smi(run, args, timeout)
        except GpuPowerSettingsError as exc:
            problems.append(str(exc))
    return problems


def apply_gpu_power_settings(
    settings: Mapping[int, Mapping[str, Any]],
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 60.0,
) -> None:
    """Set each card's cap and perf level: ``{gpu: {"power_cap_w": W, "perf_level": level}}``, either one optional.

    Raises:
        GpuPowerSettingsError: a set failed (typically not privileged). The caller restores from its record.
    """
    for args in _set_commands(settings):
        _run_amd_smi(run, args, timeout)


def _check_trusted(path: Path, st: os.stat_result, *, directory: bool) -> None:
    """Refuse a record, or a directory holding records, that another unprivileged account could have written.

    A restore runs ``amd-smi set`` with whatever a record says, usually as root, so a record is trusted only when
    nobody but this account or root can have put its content there.
    """
    if st.st_uid not in (0, os.geteuid()):
        raise GpuPowerSettingsError(
            f"{path} is owned by uid {st.st_uid}, not by this account or root, so its power-settings records "
            "cannot be trusted; remove it, or launch as its owner"
        )
    writable_by_others = st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_by_others and not (directory and st.st_mode & stat.S_ISVTX):
        raise GpuPowerSettingsError(
            f"{path} is writable by other accounts, so its power-settings records cannot be trusted; remove it"
        )


def _ensure_ledger_dir(directory: Path) -> None:
    """Create the record directory and its parent so every account on the host can add its own per-card records,
    and refuse either when another account could have planted records in it."""
    for level in (directory.parent, directory):
        with contextlib.suppress(FileExistsError):
            level.mkdir()
            # Sticky and world-writable, like /var/tmp itself: any account may create a card's record, and only
            # its owner may replace or remove it.
            os.chmod(level, 0o1777)  # nosec B103
        _check_trusted(level, os.lstat(level), directory=True)
        if not stat.S_ISDIR(os.lstat(level).st_mode):
            raise GpuPowerSettingsError(f"{level} is not a directory")


def _open_record(path: Path, flags: int) -> int:
    """Open a record without following a symlink, and refuse one another account could have written."""
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o644)
    try:
        st = os.fstat(fd)
        _check_trusted(path, st, directory=False)
        if flags & os.O_CREAT and st.st_uid == os.geteuid():
            # Readable by every account so a launch without the flag can warn about a card left changed;
            # writable only by its owner.
            os.fchmod(fd, 0o644)
    except (GpuPowerSettingsError, OSError):
        os.close(fd)
        raise
    return fd


class PowerSettingsLease:
    """Exclusive, crash-recoverable ownership of the power settings of a set of cards.

    One record file per card, ``gpu<N>.json`` under ``directory``, held under ``flock(LOCK_EX)`` from acquisition to
    :meth:`release`. The kernel drops the lock when the holder dies however it dies, so a record that can be locked
    and still has content is an orphan: its session set the card and never restored it.
    """

    def __init__(self, directory: Path | str, gpus: set[int]) -> None:
        """Lock every card's record, or none: a partial lease would set cards another session still owns.

        Raises:
            GpuPowerSettingsError: a card's record is held by a live session, or cannot be opened.
        """
        self._dir = Path(directory)
        self._fds: dict[int, int] = {}
        try:
            _ensure_ledger_dir(self._dir)
            for gpu in sorted(gpus):
                fd = _open_record(self._dir / f"gpu{gpu}.json", os.O_RDWR | os.O_CREAT)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise GpuPowerSettingsError(
                        f"GPU {gpu}'s power settings are held by another running Hyperloom session"
                    ) from None
                self._fds[gpu] = fd
        except OSError as exc:
            self.release()
            raise GpuPowerSettingsError(
                f"cannot open the power-settings record under {self._dir}: {exc}; a card's record belongs to the "
                "account that first applied settings to it, so apply as that account or as root"
            ) from exc
        except GpuPowerSettingsError:
            self.release()
            raise

    @property
    def directory(self) -> Path:
        """Where this lease's per-card records live."""
        return self._dir

    def orphaned(self) -> dict[int, dict[str, Any]]:
        """Records a dead session left behind on these cards: ``{gpu: record}``."""
        out: dict[int, dict[str, Any]] = {}
        for gpu, fd in self._fds.items():
            record = _read_record(fd)
            if record:
                out[gpu] = record
        return out

    def record(self, originals: Mapping[int, Mapping[str, Any]], *, applied: Mapping[str, Any], owner: str) -> None:
        """Write each card's originals before anything is set, so a crash mid-apply is still recoverable."""
        stamp = {_BOOT_ID: current_boot_id(), "host": socket.gethostname(), "owner": owner}
        try:
            for gpu, fd in self._fds.items():
                payload = {"gpu": gpu, "original": dict(originals.get(gpu) or {}), "applied": dict(applied), **stamp}
                data = json.dumps(payload, sort_keys=True).encode(_UTF8)
                os.ftruncate(fd, 0)
                os.pwrite(fd, data, 0)
                os.fsync(fd)
        except OSError as exc:
            raise GpuPowerSettingsError(f"cannot write the power-settings record under {self._dir}: {exc}") from exc

    def clear(self, gpus: set[int] | None = None) -> None:
        """Empty the records of ``gpus`` (every leased card by default): their originals are back."""
        try:
            for gpu, fd in self._fds.items():
                if gpus is None or gpu in gpus:
                    os.ftruncate(fd, 0)
                    os.fsync(fd)
        except OSError as exc:
            raise GpuPowerSettingsError(f"cannot clear the power-settings record under {self._dir}: {exc}") from exc

    def release(self) -> None:
        """Drop the locks. Idempotent."""
        for fd in self._fds.values():
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._fds.clear()


def _read_record(fd: int) -> dict[str, Any]:
    """The record behind ``fd``, or ``{}`` when it is empty, unreadable, or from an earlier boot of this host."""
    size = os.fstat(fd).st_size
    if not size:
        return {}
    try:
        payload = json.loads(os.pread(fd, size, 0).decode(_UTF8))
    except (ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get(_BOOT_ID) != current_boot_id():
        return {}
    return payload


def orphaned_power_records(directory: Path | str, gpus: set[int] | None = None) -> dict[int, dict[str, Any]]:
    """Records left by sessions that died without restoring, without taking ownership of the cards.

    A launch that is not permitted to set anything still has to say that a card was left at a value another session
    applied, since that value is what it is about to be measured under.
    """
    out: dict[int, dict[str, Any]] = {}
    root = Path(directory)
    if not root.is_dir():
        return out
    try:
        _check_trusted(root, os.lstat(root), directory=True)
    except GpuPowerSettingsError:
        return out
    for path in sorted(root.glob("gpu*.json")):
        try:
            gpu = int(path.stem[3:])
        except ValueError:
            continue
        if gpus is not None and gpu not in gpus:
            continue
        try:
            fd = _open_record(path, os.O_RDONLY)
        except (OSError, GpuPowerSettingsError):
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            record = _read_record(fd)
            if record:
                out[gpu] = record
        finally:
            os.close(fd)
    return out
