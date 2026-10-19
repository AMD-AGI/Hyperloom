# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--apply-gpu-power-settings``: set the declared cap / perf level on the session's cards, and put
them back.

The cards are set once, before anything runs, and restored when the session ends. The restore runs from the
``finally`` of :func:`gpu_power_restore_guard`, which wraps the whole session, with ``atexit`` as a backstop. From the
moment the cards are set until they are restored, SIGTERM and SIGHUP end the process through that ``finally`` instead
of killing it: the coordinator's own stop handling takes them over while its loop runs and hands them back when it
closes, so the startup and close windows are covered too. A SIGKILL cannot be caught; the record it leaves behind is
restored by the next launch that passes the flag.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterator, Mapping

from hyperloom.common.gpu_power_control import (
    GPU_POWER_LEDGER_DIR,
    PowerSettingsLease,
    apply_gpu_power_settings,
    orphaned_power_records,
    read_resident_vram_mb,
    restore_gpu_power_settings,
)
from hyperloom.common.gpu_power_settings import (
    PERF_LEVEL_MANUAL,
    DeclaredPowerSettings,
    GpuPowerSettingsError,
    normalize_perf_level,
    read_gpu_power_settings,
    visible_gpu_indices,
)

__all__ = [
    "AppliedPowerSettings",
    "apply_for_session",
    "gpu_power_restore_guard",
    "orphaned_power_settings_warning",
    "warn_about_orphaned_cards",
]

#: A card holding more than this much VRAM before the session has started anything has someone else's model resident,
#: and a cap is card-wide: setting it would change that tenant's run too.
_FOREIGN_RESIDENT_VRAM_MB = 2048.0
_POWER_CAP_W = "power_cap_w"
_PERF_LEVEL = "perf_level"
_DETERMINISM = "determinism"
_LABELS = {_POWER_CAP_W: "power cap", _PERF_LEVEL: "perf level"}
#: Starting perf levels a perf-level restore cannot bring back: determinism's clock soft-max is set with
#: ``amd-smi set -d``, and manual carries clock limits an operator tuned by hand, which setting a level resets.
_UNRESTORABLE_LEVELS = (_DETERMINISM, PERF_LEVEL_MANUAL)
_RESTORE_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@dataclass
class AppliedPowerSettings:
    """What :func:`apply_declared_gpu_power_settings` set: the record to archive, and how to put it back."""

    record: dict[str, Any]
    restore: Callable[[], list[str]] = field(repr=False)


@dataclass
class _Hooks:
    """The amd-smi reads and writes, and the lease, injectable for tests."""

    read: Callable[[], Mapping[int, Mapping[str, Any]]] = read_gpu_power_settings
    resident_vram: Callable[[], Mapping[int, float]] = read_resident_vram_mb
    apply: Callable[[Mapping[int, Mapping[str, Any]]], None] = apply_gpu_power_settings
    restore: Callable[[Mapping[int, Mapping[str, Any]]], list[str]] = restore_gpu_power_settings
    lease: Callable[[Path, set[int]], Any] = PowerSettingsLease
    mask: Callable[[], set[int] | None] = visible_gpu_indices
    arm: Callable[[Callable[[], list[str]]], None] = field(default=lambda restore: _GUARD.arm(restore))
    disarm: Callable[[], None] = field(default=lambda: _GUARD.disarm())


def _session_gpus(observed: Mapping[int, Any], mask: set[int] | None) -> set[int]:
    """The cards the session may set: exactly its integer visible-device mask, every one of them readable."""
    if mask is None:
        raise GpuPowerSettingsError(
            "--apply-gpu-power-settings needs an integer ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES mask naming the "
            "cards to set; without one it would set every card on the host, including ones the session does not use"
        )
    missing = sorted(mask - set(observed))
    if missing:
        raise GpuPowerSettingsError(
            f"amd-smi did not report the power settings of masked GPU(s) {', '.join(map(str, missing))}, so they "
            "could not be restored after the session"
        )
    return set(mask)


def _refuse_busy_cards(gpus: set[int], resident: Mapping[int, float]) -> None:
    """Refuse a card holding someone else's model, or one whose VRAM cannot be read, which cannot be ruled out."""
    unread = sorted(gpu for gpu in gpus if gpu not in resident)
    if unread:
        raise GpuPowerSettingsError(
            f"the VRAM use of GPU(s) {', '.join(map(str, unread))} could not be read, so a model resident on them "
            "cannot be ruled out; a power cap is card-wide and would change that workload too"
        )
    busy = sorted(gpu for gpu in gpus if resident[gpu] > _FOREIGN_RESIDENT_VRAM_MB)
    if busy:
        raise GpuPowerSettingsError(
            f"GPU(s) {', '.join(map(str, busy))} already hold a resident model; a power cap is card-wide and would "
            "change that workload too. Free the cards or pin the session away from them"
        )


def _originals(observed: Mapping[int, Mapping[str, Any]], touched: Mapping[int, list[str]]) -> dict:
    """Each card's current value of every setting this session changes; refuses when one cannot be read.

    A value that cannot be read cannot be put back: restore would skip it, report success and clear the record,
    leaving the card at the applied setting with nothing left to recover it.
    """
    originals = {
        gpu: {key: (observed.get(gpu) or {}).get(key) for key in keys} for gpu, keys in sorted(touched.items())
    }
    unreadable = sorted(
        gpu
        for gpu, row in originals.items()
        if any(value is None or isinstance(value, bool) or value == "" for value in row.values())
    )
    if unreadable:
        raise GpuPowerSettingsError(
            f"the current {' / '.join(dict.fromkeys(_label(key) for keys in touched.values() for key in keys))} of GPU(s) "
            f"{', '.join(map(str, unreadable))} could not be read, so it could not be restored after the session; "
            "nothing was applied"
        )
    return originals


def _refuse_unrestorable_levels(originals: Mapping[int, Mapping[str, Any]]) -> None:
    """Refuse cards whose starting perf level a restore could not bring back; checked before anything is set."""
    levels = {gpu: normalize_perf_level(row.get(_PERF_LEVEL)) for gpu, row in originals.items() if _PERF_LEVEL in row}
    for level in _UNRESTORABLE_LEVELS:
        stuck = sorted(gpu for gpu, at in levels.items() if at == level)
        if stuck:
            why = (
                "whose clock soft-max a restore could not put back"
                if level == _DETERMINISM
                else "whose hand-tuned clock limits setting another level would reset"
            )
            raise GpuPowerSettingsError(
                f"GPU(s) {', '.join(map(str, stuck))} are at the {level} perf level, {why}; nothing was applied. "
                "Leave --gpu-perf-level unset for these cards, or move them to auto first"
            )


def _label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " "))


def _reads_as(got: Any, want: Any, key: str) -> bool:
    """Whether a card's reading of one setting is the given value."""
    if key == _PERF_LEVEL:
        return normalize_perf_level(got) == normalize_perf_level(want)
    return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 0.5


def _restore_mismatches(
    observed: Mapping[int, Mapping[str, Any]], originals: Mapping[int, Mapping[str, Any]]
) -> dict[int, list[str]]:
    """``{gpu: problems}`` for each card not back at every recorded setting after a restore; empty when all are."""
    problems: dict[int, list[str]] = {}
    for gpu, row in sorted(originals.items()):
        now = observed.get(gpu) or {}
        for key, want in row.items():
            got = now.get(key)
            if not _reads_as(got, want, key):
                problems.setdefault(gpu, []).append(
                    f"GPU {gpu} {_label(key)} reads {got!r} after the restore, recorded {want!r}"
                )
    return problems


def _restore_and_verify(hooks: _Hooks, originals: Mapping[int, Mapping[str, Any]]) -> dict[int, list[str]]:
    """Put the originals back and judge each card by what it reads back, not by the commands' exit codes.

    Returns ``{gpu: problems}`` for the cards not back, empty when all are. An amd-smi set can exit 0 without the card
    taking the value, and fail on a card that never changed (an apply that was refused for lack of privilege); either
    way each card's state decides whether its record can go. A failed read counts against every card.
    """
    failed_sets = hooks.restore(originals)
    try:
        mismatches = _restore_mismatches(hooks.read(), originals)
    except GpuPowerSettingsError as exc:
        unread = f"the restored settings could not be read back: {exc}"
        return {gpu: [*failed_sets, unread] for gpu in originals}
    return {gpu: [*failed_sets, *problems] for gpu, problems in mismatches.items()}


def _messages(problems: Mapping[int, list[str]]) -> list[str]:
    """Every distinct problem, in card order."""
    return list(dict.fromkeys(message for gpu in sorted(problems) for message in problems[gpu]))


def _readable(value: Any) -> bool:
    return value not in (None, "") and not isinstance(value, bool)


def _still_applied(record: Mapping[str, Any], now: Mapping[str, Any] | None) -> dict[str, Any]:
    """The recorded originals of the settings a card still reads at the dead session's applied values.

    Judged setting by setting: a restore that put back the cap but not the level, or a SIGKILL between the two sets,
    leaves only some settings applied, and those are still the session's to put back. A setting moved since (an
    operator restored it, or set a value of their own) is not: restoring the stale original over it would undo them.
    """
    applied, original, now = record.get("applied") or {}, record.get("original") or {}, now or {}
    return {
        key: original[key] for key, want in applied.items() if key in original and _reads_as(now.get(key), want, key)
    }


def _unread(record: Mapping[str, Any], now: Mapping[str, Any] | None) -> list[str]:
    """The recorded settings a card's reading does not show, so whether they are still applied cannot be told."""
    original, now = record.get("original") or {}, now or {}
    return [key for key in (record.get("applied") or {}) if key in original and not _readable(now.get(key))]


def _unread_originals(record: Mapping[str, Any], now: Mapping[str, Any] | None) -> dict[str, Any]:
    original = record.get("original") or {}
    return {key: original[key] for key in _unread(record, now)}


def _recover_orphans(lease: Any, hooks: _Hooks) -> list[int]:
    """Restore what a dead session left on these cards; the cards it recovered.

    The cards are read under the lease, so no other session can change them between the reading and the restore. A
    record none of whose settings still reads at the applied values is stale and only cleared.
    """
    orphans = lease.orphaned()
    if not orphans:
        return []
    observed = hooks.read()
    unread = sorted(gpu for gpu, rec in orphans.items() if _unread(rec, observed.get(gpu)))
    if unread:
        # Neither cleared nor restored: the record may still be the only way back to these cards' originals.
        files = ", ".join(str(lease.directory / f"gpu{gpu}.json") for gpu in unread)
        raise GpuPowerSettingsError(
            f"GPU(s) {', '.join(map(str, unread))} carry a record a dead session left, and their settings could not "
            f"be read to tell whether it still applies; the record is kept. Check those cards by hand (amd-smi), then "
            f"remove {files}"
        )
    live = {gpu: left for gpu, rec in orphans.items() if (left := _still_applied(rec, observed.get(gpu)))}
    lease.clear(set(orphans) - set(live))
    if not live:
        return []
    problems = _restore_and_verify(hooks, live)
    lease.clear(set(live) - set(problems))
    if problems:
        raise GpuPowerSettingsError(
            "a previous session left these cards at settings it applied and they could not be restored: "
            + "; ".join(_messages(problems))
        )
    return sorted(live)


def apply_declared_gpu_power_settings(
    declared: DeclaredPowerSettings,
    *,
    nodes: int,
    owner: str,
    ledger_dir: Path = GPU_POWER_LEDGER_DIR,
    hooks: _Hooks | None = None,
) -> AppliedPowerSettings:
    """Set the declared cap / perf level on the session's masked cards, recording the originals first.

    The read-back check is still the caller's :func:`resolve_gpu_power_settings`, so an applied value is verified the
    same way an operator-set one is.

    Raises:
        GpuPowerSettingsError: the launch must stop. Nothing was left changed, unless the message says that putting
            the originals back failed too; the record is then kept for the next launch.
    """
    hooks = hooks or _Hooks()
    if not any(value for value in declared.as_row().values()):
        raise GpuPowerSettingsError("--apply-gpu-power-settings needs --gpu-power-cap-w and/or --gpu-perf-level")
    if nodes >= 2:
        raise GpuPowerSettingsError("--apply-gpu-power-settings cannot set the cards of a multi-node session")
    gpus = _session_gpus(hooks.read(), hooks.mask())
    lease = hooks.lease(ledger_dir, gpus)
    # Only what this session changes is recorded, so only that is put back: restoring a perf level nobody declared
    # would reset a card left in MANUAL clocks for reasons this session knows nothing about.
    wanted = declared.as_row()
    applied_keys = [key for key, value in wanted.items() if value]
    try:
        recovered = _recover_orphans(lease, hooks)
        _refuse_busy_cards(gpus, hooks.resident_vram())
        observed = hooks.read()
        # A setting a card is already at is left alone: neither set nor put back. So a manual or determinism card
        # asked to stay where it is needs no refusal, and no record holds an applied value equal to its original.
        touched = {
            gpu: [key for key in applied_keys if not _reads_as((observed.get(gpu) or {}).get(key), wanted[key], key)]
            for gpu in gpus
        }
        originals = _originals(observed, touched)
        _refuse_unrestorable_levels(originals)
        lease.record(originals, applied={key: wanted[key] for key in applied_keys}, owner=owner)
    except GpuPowerSettingsError:
        lease.release()
        raise

    done = False

    def _restore() -> list[str]:
        nonlocal done
        with _stop_signals_held():
            if done:
                return []
            done = True
            problems = _restore_and_verify(hooks, originals)
            try:
                lease.clear(set(originals) - set(problems))
            except GpuPowerSettingsError as exc:
                problems = {**problems, -1: [str(exc)]}
            lease.release()
        return _messages(problems)

    # Armed before the first set: a card is changed from the moment its set returns, and a stop signal or an
    # interrupt from then on has to unwind through the session's restore.
    hooks.arm(_restore)
    try:
        hooks.apply({gpu: {key: wanted[key] for key in keys} for gpu, keys in touched.items()})
    except GpuPowerSettingsError as exc:
        leftover = _restore()
        hooks.disarm()
        suffix = f" (restoring the originals also failed: {'; '.join(leftover)})" if leftover else ""
        raise GpuPowerSettingsError(f"{exc}{suffix}") from exc
    record: dict[str, Any] = {
        "by": "hyperloom",
        "gpus": sorted(gpus),
        "originals": {str(gpu): row for gpu, row in originals.items()},
    }
    if recovered:
        record["recovered_from_orphan"] = recovered
    return AppliedPowerSettings(record=record, restore=_restore)


def _describe(values: Mapping[str, Any]) -> str:
    cap, level = values.get(_POWER_CAP_W), values.get(_PERF_LEVEL)
    parts = [f"cap {cap:g} W"] if isinstance(cap, (int, float)) and not isinstance(cap, bool) else []
    return ", ".join(parts + ([f"perf level {level}"] if level else []))


def orphaned_power_settings_warning(
    ledger_dir: Path = GPU_POWER_LEDGER_DIR,
    read: Callable[[], Mapping[int, Mapping[str, Any]]] = read_gpu_power_settings,
) -> str:
    """A warning when a dead session left one of this session's cards at a value it applied; ``""`` otherwise.

    A setting a card no longer reads at was put back or reset since, and is not reported; one whose reading is
    missing is, since it may still be applied. When the cards cannot be read at all, every record is reported.
    """
    orphans = orphaned_power_records(ledger_dir, visible_gpu_indices())
    if not orphans:
        return ""
    try:
        observed = read()
    except GpuPowerSettingsError:
        observed = None
    if observed is not None:
        orphans = {
            gpu: {
                **rec,
                "applied": {key: rec["applied"][key] for key in left},
                "original": left,
                "unread": bool(_unread(rec, observed.get(gpu))),
            }
            for gpu, rec in orphans.items()
            if (left := {**_still_applied(rec, observed.get(gpu)), **_unread_originals(rec, observed.get(gpu))})
        }
    if not orphans:
        return ""
    parts = [
        f"GPU {gpu} {'may still be at' if record.get('unread') else 'at'} "
        f"{_describe(record.get('applied') or {}) or '?'} from session {record.get('owner') or '?'}, "
        f"originally {_describe(record.get('original') or {}) or '?'}"
        for gpu, record in sorted(orphans.items())
    ]
    return (
        "these cards were left at power settings a Hyperloom session applied and never restored: "
        + "; ".join(parts)
        + ". Pass --apply-gpu-power-settings to restore them at launch, or restore them with amd-smi set"
    )


@contextmanager
def _stop_signals_held() -> Iterator[None]:
    """Ignore the stop signals while a restore runs, so one cannot cut it short; give them back after."""
    held = {}
    for sig in (*_RESTORE_SIGNALS, signal.SIGINT):
        try:
            held[sig] = signal.signal(sig, signal.SIG_IGN)
        except ValueError:
            # Off the main thread signal handlers cannot change, and signals are delivered to the main thread anyway.
            pass
    try:
        yield
    finally:
        for sig, handler in held.items():
            signal.signal(sig, handler)


class _RestoreGuard:
    """Holds the session's restore from the moment the cards are set until they are put back."""

    def __init__(self) -> None:
        self._restore: Callable[[], list[str]] | None = None
        self._previous: dict[int, Any] = {}

    @staticmethod
    def _exit_on_signal(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    def arm(self, restore: Callable[[], list[str]]) -> None:
        """Take ownership of ``restore``: run it on the way out, however the process leaves short of SIGKILL."""
        self._restore = restore
        for sig in _RESTORE_SIGNALS:
            self._previous.setdefault(sig, signal.signal(sig, self._exit_on_signal))
        atexit.register(self.run)

    def disarm(self) -> None:
        """Drop a restore that already ran, and give the stop signals back."""
        self._restore = None
        for sig, handler in list(self._previous.items()):
            signal.signal(sig, handler)
        self._previous.clear()

    def run(self) -> None:
        """Put the cards back, once; a stop signal during the restore is ignored so it cannot cut it short."""
        restore, self._restore = self._restore, None
        if restore is None:
            return
        held = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (*_RESTORE_SIGNALS, signal.SIGINT)}
        try:
            problems = restore()
        finally:
            for sig, handler in held.items():
                signal.signal(sig, self._previous.pop(sig, handler))
        if problems:
            print(
                "WARNING: could not restore the GPU power settings this session applied: "
                + "; ".join(problems)
                + ". The record is kept, so the next launch with --apply-gpu-power-settings restores them.",
                file=sys.stderr,
            )
        else:
            print("GPU power settings restored to the values the session found.")


_GUARD: _RestoreGuard = _RestoreGuard()


@contextmanager
def gpu_power_restore_guard() -> Iterator[None]:
    """Wrap a session: whatever the session applied is restored when this block is left."""
    try:
        yield
    finally:
        _GUARD.run()


def warn_about_orphaned_cards() -> None:
    """Without permission to set anything, say which of this session's cards a dead session left changed."""
    if warning := orphaned_power_settings_warning():
        print(f"WARNING: {warning}.", file=sys.stderr)


def apply_for_session(declared: DeclaredPowerSettings, *, nodes: int, owner: str) -> dict[str, Any]:
    """Apply the declared settings, their restore held by the session's guard; the ``applied`` record to archive.

    Raises:
        GpuPowerSettingsError: the settings could not be applied. Nothing was left changed, unless the message says
            that putting the originals back failed too; the record is then kept for the next launch.
    """
    if os.geteuid() != 0:
        # amd-smi set needs root; refused before the records exist, so an unprivileged attempt leaves no file behind
        # that a later root launch would have to distrust.
        raise GpuPowerSettingsError(
            "--apply-gpu-power-settings needs root, as amd-smi set does: relaunch as root, or drop the flag and set "
            "the cards with amd-smi set before launch"
        )
    applied = apply_declared_gpu_power_settings(declared, nodes=nodes, owner=owner)
    print(f"GPU power settings applied to GPU(s) {', '.join(map(str, applied.record['gpus']))}; restored at exit.")
    return applied.record
