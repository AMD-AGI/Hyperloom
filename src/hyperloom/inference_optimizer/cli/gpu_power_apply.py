# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--apply-gpu-power-settings``: set the declared cap / perf level / sclk max on the session's cards, and put
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
    declared_combination_error,
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
_SCLK_MAX_MHZ = "sclk_max_mhz"
_DETERMINISM = "determinism"
_LEVELS_AFTER_SCLK = frozenset({"auto", PERF_LEVEL_MANUAL})
_LABELS = {_POWER_CAP_W: "power cap", _PERF_LEVEL: "perf level", _SCLK_MAX_MHZ: "sclk max"}
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
    apply: Callable[[set[int], DeclaredPowerSettings], None] = apply_gpu_power_settings
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


def _originals(
    observed: Mapping[int, Mapping[str, Any]], gpus: set[int], touched: list[str], *, sclk_coupled: bool = False
) -> dict:
    """Each card's current value of every setting this session changes; refuses when one cannot be read.

    A value that cannot be read cannot be put back: restore would skip it, report success and clear the record,
    leaving the card at the applied setting with nothing left to recover it.
    """
    originals = {gpu: {key: (observed.get(gpu) or {}).get(key) for key in touched} for gpu in sorted(gpus)}
    if sclk_coupled:
        # Recorded only to put back an operator's limit, which only a manual card can hold: restoring any other level
        # resets the limit itself, so for those cards it is neither needed nor required to be readable.
        for row in originals.values():
            if normalize_perf_level(row.get(_PERF_LEVEL)) != PERF_LEVEL_MANUAL:
                row.pop(_SCLK_MAX_MHZ, None)
    unreadable = sorted(
        gpu
        for gpu, row in originals.items()
        if any(value is None or isinstance(value, bool) or value == "" for value in row.values())
    )
    if unreadable:
        raise GpuPowerSettingsError(
            f"the current {' / '.join(_LABELS[key] for key in touched)} of GPU(s) "
            f"{', '.join(map(str, unreadable))} could not be read, so it could not be restored after the session; "
            "nothing was applied"
        )
    return originals


def _refuse_unrestorable_levels(originals: Mapping[int, Mapping[str, Any]], *, sclk_applied: bool) -> None:
    """Refuse cards whose starting perf level a restore could not bring back; checked before anything is set.

    A determinism card's clock soft-max is set with ``amd-smi set -d``, which a perf-level restore does not carry. And
    once a clock limit has put a card in manual, setting a level other than auto does not take it out of manual.
    """
    levels = {gpu: normalize_perf_level(row.get(_PERF_LEVEL)) for gpu, row in originals.items() if _PERF_LEVEL in row}
    determinism = sorted(gpu for gpu, level in levels.items() if level == _DETERMINISM)
    if determinism:
        raise GpuPowerSettingsError(
            f"GPU(s) {', '.join(map(str, determinism))} are at the determinism perf level, whose clock soft-max a "
            "restore could not put back; nothing was applied. Leave --gpu-perf-level / --gpu-sclk-max-mhz unset for "
            "these cards, or move them to auto first"
        )
    stuck = sorted(gpu for gpu, level in levels.items() if sclk_applied and level not in _LEVELS_AFTER_SCLK)
    if stuck:
        named = ", ".join(sorted({levels[gpu] for gpu in stuck}))
        raise GpuPowerSettingsError(
            f"GPU(s) {', '.join(map(str, stuck))} are at the {named} perf level; a clock limit puts a card in manual, "
            "and setting that level again would leave it there, so the sclk cap could not be undone. Nothing was "
            "applied; move the cards to auto or manual first"
        )


def _restore_mismatches(observed: Mapping[int, Mapping[str, Any]], originals: Mapping[int, Mapping[str, Any]]) -> list:
    """Each recorded setting a card is not back at after a restore; empty when every one is."""
    problems = []
    for gpu, row in sorted(originals.items()):
        now = observed.get(gpu) or {}
        for key, want in row.items():
            got = now.get(key)
            if key == _PERF_LEVEL:
                matches = normalize_perf_level(got) == normalize_perf_level(want)
            else:
                matches = isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 0.5
            if not matches:
                problems.append(f"GPU {gpu} {_LABELS[key]} reads {got!r} after the restore, recorded {want!r}")
    return problems


def _restore_and_verify(hooks: _Hooks, originals: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Put the originals back and judge the cards by what they read back, not by the commands' exit codes.

    An amd-smi set can exit 0 without the card taking the value, and fail on a card that never changed (an apply that
    was refused for lack of privilege); either way the cards' state decides whether the record can go.
    """
    problems = hooks.restore(originals)
    try:
        mismatches = _restore_mismatches(hooks.read(), originals)
    except GpuPowerSettingsError as exc:
        return [*problems, f"the restored settings could not be read back: {exc}"]
    return [*problems, *mismatches] if mismatches else []


def _recover_orphans(lease: Any, hooks: _Hooks) -> list[int]:
    """Restore what a dead session left on these cards; the cards it recovered."""
    orphans = lease.orphaned()
    if not orphans:
        return []
    problems = _restore_and_verify(hooks, {gpu: dict(rec.get("original") or {}) for gpu, rec in orphans.items()})
    if problems:
        raise GpuPowerSettingsError(
            "a previous session left these cards at settings it applied and they could not be restored: "
            + "; ".join(problems)
        )
    lease.clear()
    return sorted(orphans)


def apply_declared_gpu_power_settings(
    declared: DeclaredPowerSettings,
    *,
    nodes: int,
    owner: str,
    ledger_dir: Path = GPU_POWER_LEDGER_DIR,
    hooks: _Hooks | None = None,
) -> AppliedPowerSettings:
    """Set the declared cap / perf level / sclk max on the session's masked cards, recording the originals first.

    The read-back check is still the caller's :func:`resolve_gpu_power_settings`, so an applied value is verified the
    same way an operator-set one is.

    Raises:
        GpuPowerSettingsError: nothing was left changed and the launch must stop.
    """
    hooks = hooks or _Hooks()
    if not any(value for value in declared.as_row().values()):
        raise GpuPowerSettingsError(
            "--apply-gpu-power-settings needs at least one of --gpu-power-cap-w, --gpu-perf-level, --gpu-sclk-max-mhz"
        )
    if conflict := declared_combination_error(perf_level=declared.perf_level, sclk_max_mhz=declared.sclk_max_mhz):
        raise GpuPowerSettingsError(conflict)
    if nodes >= 2:
        raise GpuPowerSettingsError("--apply-gpu-power-settings cannot set the cards of a multi-node session")
    gpus = _session_gpus(hooks.read(), hooks.mask())
    lease = hooks.lease(ledger_dir, gpus)
    # Only what this session changes is recorded, so only that is put back: restoring a perf level that neither a
    # declared level nor a declared sclk cap touched would reset a card left in MANUAL clocks for reasons this session
    # knows nothing about.
    wanted = declared.as_row()
    applied_keys = [key for key, value in wanted.items() if value]
    # The perf level and the sclk limit change each other: a clock limit puts the card in manual, and setting a level
    # resets the limit. Applying either records and restores both.
    coupled = [key for key in (_PERF_LEVEL, _SCLK_MAX_MHZ) if key not in applied_keys]
    touched = applied_keys + (coupled if any(key in applied_keys for key in (_PERF_LEVEL, _SCLK_MAX_MHZ)) else [])
    try:
        recovered = _recover_orphans(lease, hooks)
        _refuse_busy_cards(gpus, hooks.resident_vram())
        originals = _originals(
            hooks.read(), gpus, touched, sclk_coupled=_SCLK_MAX_MHZ in touched and _SCLK_MAX_MHZ not in applied_keys
        )
        _refuse_unrestorable_levels(originals, sclk_applied=declared.sclk_max_mhz is not None)
        lease.record(originals, applied={key: wanted[key] for key in applied_keys}, owner=owner)
    except GpuPowerSettingsError:
        lease.release()
        raise

    done = False

    def _restore() -> list[str]:
        nonlocal done
        if done:
            return []
        done = True
        problems = _restore_and_verify(hooks, originals)
        if not problems:
            lease.clear()
        lease.release()
        return problems

    # Armed before the first set: a card is changed from the moment its set returns, and a stop signal or an
    # interrupt from then on has to unwind through the session's restore.
    hooks.arm(_restore)
    try:
        hooks.apply(gpus, declared)
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
    cap, level, sclk = values.get(_POWER_CAP_W), values.get(_PERF_LEVEL), values.get(_SCLK_MAX_MHZ)
    parts = [f"cap {cap:g} W"] if isinstance(cap, (int, float)) and not isinstance(cap, bool) else []
    parts += [f"perf level {level}"] if level else []
    parts += [f"sclk max {sclk:g} MHz"] if isinstance(sclk, (int, float)) and not isinstance(sclk, bool) else []
    return ", ".join(parts)


def orphaned_power_settings_warning(ledger_dir: Path = GPU_POWER_LEDGER_DIR) -> str:
    """A warning when a dead session left one of this session's cards at a value it applied; ``""`` otherwise."""
    orphans = orphaned_power_records(ledger_dir, visible_gpu_indices())
    if not orphans:
        return ""
    parts = [
        f"GPU {gpu} at {_describe(record.get('applied') or {}) or '?'} from session {record.get('owner') or '?'}, "
        f"originally {_describe(record.get('original') or {}) or '?'}"
        for gpu, record in sorted(orphans.items())
    ]
    return (
        "these cards were left at power settings a Hyperloom session applied and never restored: "
        + "; ".join(parts)
        + ". Pass --apply-gpu-power-settings to restore them at launch, or restore them with amd-smi set"
    )


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


_GUARD = _RestoreGuard()


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
        GpuPowerSettingsError: the settings could not be applied; nothing was left changed.
    """
    applied = apply_declared_gpu_power_settings(declared, nodes=nodes, owner=owner)
    print(f"GPU power settings applied to GPU(s) {', '.join(map(str, applied.record['gpus']))}; restored at exit.")
    return applied.record
