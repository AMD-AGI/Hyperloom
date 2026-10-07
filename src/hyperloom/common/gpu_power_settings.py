# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GPU power settings a session is measured under: read by default, applied only when the operator permits it.

The power cap and the DPM performance level decide how much of a card's throughput is available and at what power,
so they are part of the measurement contract. By default nothing here changes them: setting either is privileged and
card-wide, and belongs to the operator before launch (``amd-smi set --power-cap`` / ``--perf-level``), the same way
``--compute-partition-mode`` asserts a mode rather than setting one. This module reads what the cards are at, so a
session can record it and refuse to start when an operator's declared value did not take effect.

With ``--apply-gpu-power-settings`` the session sets the declared values itself on the cards it uses, for its whole
lifetime, and restores the originals when it exits. :class:`PowerSettingsLease` makes that safe to leave unattended:
the originals are written to a per-card record before anything is set, and the record is held under an exclusive
``flock`` for the session's lifetime. A second session cannot set the same card while it is held, and a record whose
lock is free belongs to a session that died without restoring, so the next launch restores it first.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess  # nosec B404 - fixed amd-smi invocations.
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "PowerSettingsLease",
    "GpuPowerSettingsError",
    "apply_gpu_power_settings",
    "declared_setting_problems",
    "normalize_perf_level",
    "orphaned_power_records",
    "read_gpu_power_settings",
    "read_resident_vram_mb",
    "restore_gpu_power_settings",
    "visible_gpu_indices",
]

_PERF_LEVEL_PREFIX = "AMDSMI_DEV_PERF_LEVEL_"


class GpuPowerSettingsError(RuntimeError):
    """``amd-smi`` could not be run or its output could not be read."""


def normalize_perf_level(value: Any) -> str:
    """``AMDSMI_DEV_PERF_LEVEL_AUTO`` and ``auto`` both read as ``"auto"``; ``""`` when absent."""
    text = str(value or "").strip()
    if text.upper().startswith(_PERF_LEVEL_PREFIX):
        text = text[len(_PERF_LEVEL_PREFIX) :]
    return text.lower()


def visible_gpu_indices(env: Mapping[str, str] | None = None) -> set[int] | None:
    """Card indices the session may use, from ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``; ``None`` for all."""
    source = os.environ if env is None else env
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw = str(source.get(name) or "").strip()
        if not raw:
            continue
        try:
            return {int(part) for part in raw.split(",") if part.strip()}
        except ValueError:
            return None
    return None


def _gpu_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("gpu"), int)]


def _watts(block: Any) -> float | None:
    if isinstance(block, dict):
        block = block.get("value")
    return float(block) if isinstance(block, (int, float)) and not isinstance(block, bool) else None


def _power_cap_w(limit: Any) -> float | None:
    """The socket power limit, whichever key this amd-smi release uses for it."""
    if not isinstance(limit, dict):
        return None
    for key in ("ppt0", "ppt"):
        section = limit.get(key)
        if isinstance(section, dict) and (cap := _watts(section.get("socket_power_limit"))) is not None:
            return cap
    for key in ("socket_power_limit", "power_cap"):
        if (cap := _watts(limit.get(key))) is not None:
            return cap
    return None


def read_gpu_power_settings(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> dict[int, dict[str, Any]]:
    """``{gpu: {"power_cap_w": float | None, "perf_level": str}}`` for every card ``amd-smi`` lists.

    Raises:
        GpuPowerSettingsError: ``amd-smi`` is absent, failed, or printed something that is not its JSON.
    """
    if shutil.which("amd-smi") is None and run is subprocess.run:
        raise GpuPowerSettingsError("amd-smi is not on PATH")

    def _query(*args: str) -> Any:
        try:
            done = run(["amd-smi", *args, "--json"], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
        if done.returncode != 0:
            raise GpuPowerSettingsError(
                f"amd-smi {' '.join(args)} exited {done.returncode}: {done.stderr.strip()[:200]}"
            )
        try:
            return json.loads(done.stdout)
        except ValueError as exc:
            raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} printed no JSON") from exc

    settings: dict[int, dict[str, Any]] = {}
    for row in _gpu_rows(_query("static", "--limit")):
        settings.setdefault(row["gpu"], {})["power_cap_w"] = _power_cap_w(row.get("limit"))
    for row in _gpu_rows(_query("metric", "--perf-level")):
        settings.setdefault(row["gpu"], {})["perf_level"] = normalize_perf_level(row.get("perf_level"))
    for entry in settings.values():
        entry.setdefault("power_cap_w", None)
        entry.setdefault("perf_level", "")
    return settings


def declared_setting_problems(
    settings: Mapping[int, Mapping[str, Any]],
    *,
    power_cap_w: float | None = None,
    perf_level: str | None = None,
    gpus: set[int] | None = None,
) -> list[str]:
    """Why the cards are not at the declared settings; empty when they are, or when nothing is declared.

    A declared value that cannot be read counts as a mismatch: an assertion nobody verified is not a satisfied one.
    """
    wanted_level = normalize_perf_level(perf_level) if perf_level else ""
    checked = {gpu: row for gpu, row in settings.items() if gpus is None or gpu in gpus}
    if (power_cap_w is not None or wanted_level) and not checked:
        return ["no GPU the session can use was reported by amd-smi"]
    problems: list[str] = []
    for gpu, row in sorted(checked.items()):
        if power_cap_w is not None:
            cap = row.get("power_cap_w")
            if not isinstance(cap, (int, float)) or abs(float(cap) - float(power_cap_w)) > 0.5:
                problems.append(
                    f"GPU {gpu} power cap is {cap if cap is not None else 'unreadable'} W, declared {power_cap_w:g} W"
                )
        if wanted_level and row.get("perf_level") != wanted_level:
            problems.append(
                f"GPU {gpu} perf level is {row.get('perf_level') or 'unreadable'!r}, declared {wanted_level!r}"
            )
    return problems


def _run_amd_smi(run: Callable[..., subprocess.CompletedProcess], args: list[str], timeout: float) -> str:
    try:
        done = run(["amd-smi", *args], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()[:200]
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
        cap = row.get("power_cap_w")
        if isinstance(cap, (int, float)) and not isinstance(cap, bool):
            by_cap.setdefault(float(cap), []).append(gpu)
        level = normalize_perf_level(row.get("perf_level"))
        if level:
            by_level.setdefault(level, []).append(gpu)
    commands = [["set", "-g", *map(str, gpus), "-o", "ppt0", f"{cap:g}"] for cap, gpus in by_cap.items()]
    commands += [["set", "-g", *map(str, gpus), "-l", level.upper()] for level, gpus in by_level.items()]
    return commands


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
    gpus: set[int],
    *,
    power_cap_w: float | None,
    perf_level: str | None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 60.0,
) -> None:
    """Set the declared cap and perf level on ``gpus``.

    Raises:
        GpuPowerSettingsError: a set failed (typically not privileged). The caller restores from its record.
    """
    wanted = {gpu: {"power_cap_w": power_cap_w, "perf_level": perf_level or ""} for gpu in gpus}
    for args in _set_commands(wanted):
        _run_amd_smi(run, args, timeout)


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
            self._dir.mkdir(parents=True, exist_ok=True)
            for gpu in sorted(gpus):
                fd = os.open(self._dir / f"gpu{gpu}.json", os.O_RDWR | os.O_CREAT, 0o600)
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
            raise GpuPowerSettingsError(f"cannot open the power-settings record under {self._dir}: {exc}") from exc
        except GpuPowerSettingsError:
            self.release()
            raise

    @property
    def gpus(self) -> set[int]:
        """Cards this lease holds."""
        return set(self._fds)

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
        for gpu, fd in self._fds.items():
            payload = {"gpu": gpu, "original": dict(originals.get(gpu) or {}), "applied": dict(applied), "owner": owner}
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            os.ftruncate(fd, 0)
            os.pwrite(fd, data, 0)
            os.fsync(fd)

    def clear(self) -> None:
        """Empty the records: the originals are back, and nothing is left to recover."""
        for fd in self._fds.values():
            os.ftruncate(fd, 0)
            os.fsync(fd)

    def release(self) -> None:
        """Drop the locks. Idempotent."""
        for fd in self._fds.values():
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._fds.clear()


def _read_record(fd: int) -> dict[str, Any]:
    size = os.fstat(fd).st_size
    if not size:
        return {}
    try:
        payload = json.loads(os.pread(fd, size, 0).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def orphaned_power_records(directory: Path | str, gpus: set[int] | None = None) -> dict[int, dict[str, Any]]:
    """Records left by sessions that died without restoring, without taking ownership of the cards.

    A launch that is not permitted to set anything still has to say that a card was left at a value another session
    applied, since that value is what it is about to be measured under.
    """
    out: dict[int, dict[str, Any]] = {}
    root = Path(directory)
    if not root.is_dir():
        return out
    for path in sorted(root.glob("gpu*.json")):
        try:
            gpu = int(path.stem[3:])
        except ValueError:
            continue
        if gpus is not None and gpu not in gpus:
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
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
