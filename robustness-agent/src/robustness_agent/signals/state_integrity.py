# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""State-integrity signals (I1-I5).

Five detectors guard the session files Coordinator relies on and the
process it runs as:

* **I1 ``state_json_corrupt``** — ``state.json`` failed to parse or
  shrank below known-good size (partial write); resume would lose
  baseline / current_best / explore_search ledgers.
* **I2 ``coordinator_wal_bloat``** — ``coordinator.db-wal`` past the
  size threshold (default 1 GiB); un-checkpointed WAL tanks SQLite I/O.
* **I3 ``stale_lease``** — a ``leases`` row held by a dead PID, freezing
  every downstream proposal on that lane.
* **I4 ``inbox_bloat``** — a role's ``inbox/outbox.jsonl`` past the
  threshold; per-tick JSONL parsing slows then agent backends time out.
* **I5 ``coordinator_zombie``** — recorded PID dead but ``state.json``
  has no ``stop_reason``; HIGH, operator must restart manually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.envelope import ROBUSTNESS_STATE_FIELDS  # noqa: F401  (audit ref)
from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity



@dataclass
class StateIntegrityConfig:
    """Tunables for :func:`evaluate_state_integrity_signals`."""

    # I2 — WAL size warn/crit; Robustness only nags, Coordinator owns checkpointing.
    wal_bytes_warn_threshold: int = 1 * 1024 * 1024 * 1024  # 1 GiB
    wal_bytes_critical_threshold: int = 4 * 1024 * 1024 * 1024  # 4 GiB
    # I3 — min stale-lease age before firing, to avoid racing the reaper.
    stale_lease_min_age_s: float = 60.0
    # I4 — agent-file thresholds.
    inbox_bloat_warn_bytes: int = 100 * 1024 * 1024     # 100 MiB
    inbox_bloat_critical_bytes: int = 500 * 1024 * 1024  # 500 MiB


def evaluate_state_integrity_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: StateIntegrityConfig | None = None,
) -> list[Symptom]:
    """Run the I1-I5 state-integrity rules and aggregate their symptoms.

    Args:
        ctx (ReactorContext): Reactor context for the current tick.
        data (SourceData): Collected source data including
            ``local_state_integrity``.
        config (StateIntegrityConfig | None): Tunables; defaults to
            :class:`StateIntegrityConfig` when ``None``.

    Returns:
        list[Symptom]: All state-integrity symptoms found this tick, possibly
            empty.
    """
    cfg = config or StateIntegrityConfig()
    si = data.local_state_integrity
    if not isinstance(si, dict) or not si:
        return []
    out: list[Symptom] = []
    out.extend(_state_json_symptoms(si))
    out.extend(_wal_bloat_symptoms(si, cfg))
    out.extend(_stale_lease_symptoms(ctx, si, cfg))
    out.extend(_inbox_bloat_symptoms(si, cfg))
    out.extend(_coordinator_zombie_symptoms(si))
    return out


# ---------------------------------------------------------------------------
# I1 — state.json corruption
# ---------------------------------------------------------------------------

def _state_json_symptoms(si: dict[str, Any]) -> list[Symptom]:
    """I1: fire ``state_json_corrupt`` when ``state.json`` is unreadable.

    Stays silent for a merely-absent file (normal on tick 0); I5 covers the
    "should exist but the run died" case.

    Args:
        si (dict[str, Any]): The state-integrity probe sample.

    Returns:
        list[Symptom]: A one-element list with the ``state_json_corrupt``
            symptom on corruption, otherwise an empty list.
    """
    state = si.get("state_json")
    if not isinstance(state, dict) or not state:
        return []
    if state.get("valid"):
        return []
    error = str(state.get("error") or "unknown")
    # "missing" is normal pre-first-persist; stay silent so a fresh
    # sandbox doesn't false-fire (I5 covers the should-exist-but-died case).
    if error == "missing":
        return []
    return [
        Symptom(
            name="state_json_corrupt",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"state.json is unreadable: {error}; resume from this "
                f"session would lose baseline / current_best / "
                f"explore_search progress"
            ),
            evidence={
                "path": state.get("path"),
                "error": error,
                "size_bytes": state.get("size_bytes"),
            },
            subject={},
            source="local",
            suggestion=(
                "back up the broken state.json and stop the run; "
                "Coordinator atomic-writes are supposed to prevent "
                "partial writes — investigate the file system"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# I2 — coordinator.db-wal bloat
# ---------------------------------------------------------------------------

def _wal_bloat_symptoms(
    si: dict[str, Any], cfg: StateIntegrityConfig,
) -> list[Symptom]:
    """I2: fire ``coordinator_wal_bloat`` when the SQLite WAL grows too large.

    Args:
        si (dict[str, Any]): The state-integrity probe sample.
        cfg (StateIntegrityConfig): Tunables (provides WAL warn/crit
            thresholds).

    Returns:
        list[Symptom]: A one-element list with the ``coordinator_wal_bloat``
            symptom when the WAL crosses a threshold, otherwise an empty list.
    """
    wal = si.get("wal")
    if not isinstance(wal, dict):
        return []
    wal_bytes = wal.get("wal_bytes")
    if not isinstance(wal_bytes, int) or wal_bytes <= 0:
        return []
    if wal_bytes >= cfg.wal_bytes_critical_threshold:
        severity = SymptomSeverity.HIGH
    elif wal_bytes >= cfg.wal_bytes_warn_threshold:
        severity = SymptomSeverity.MEDIUM
    else:
        return []
    return [
        Symptom(
            name="coordinator_wal_bloat",
            severity=severity,
            summary=(
                f"coordinator.db-wal at {wal_bytes / (1024 ** 3):.2f} GiB "
                f"(warn={cfg.wal_bytes_warn_threshold / (1024 ** 3):.1f} GiB / "
                f"crit={cfg.wal_bytes_critical_threshold / (1024 ** 3):.1f} GiB)"
            ),
            evidence={
                "wal_bytes": wal_bytes,
                "wal_path": wal.get("wal_path"),
                "db_bytes": wal.get("db_bytes"),
                "warn_threshold": cfg.wal_bytes_warn_threshold,
                "critical_threshold": cfg.wal_bytes_critical_threshold,
            },
            subject={},
            source="local",
            suggestion=(
                "Coordinator should run PRAGMA wal_checkpoint(TRUNCATE) "
                "more aggressively; for now SQLite read/write latency "
                "will degrade until the WAL is rolled forward"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# I3 — stale leases (holder PID dead but lease not released)
# ---------------------------------------------------------------------------

def _stale_lease_symptoms(
    ctx: ReactorContext,
    si: dict[str, Any],
    cfg: StateIntegrityConfig,
) -> list[Symptom]:
    """I3: fire ``stale_lease`` for leases held by dead PIDs past the min age.

    Args:
        ctx (ReactorContext): Reactor context (provides the current unix time).
        si (dict[str, Any]): The state-integrity probe sample.
        cfg (StateIntegrityConfig): Tunables (provides the minimum stale age).

    Returns:
        list[Symptom]: One ``stale_lease`` symptom per stale lease, possibly
            empty.
    """
    leases = si.get("leases")
    if not isinstance(leases, list) or not leases:
        return []
    now = float(ctx.now_unix or 0.0)
    out: list[Symptom] = []
    for entry in leases:
        if not isinstance(entry, dict):
            continue
        if entry.get("alive") is not False:
            continue
        # Skip recently-acquired leases (reaper hasn't run); coerce
        # unix-seconds or ISO ``acquired_at``.
        acquired_unix = _coerce_unix(entry.get("acquired_at"))
        age_s = (
            (now - acquired_unix)
            if acquired_unix is not None and now > 0
            else cfg.stale_lease_min_age_s + 1.0
        )
        if age_s < cfg.stale_lease_min_age_s:
            continue
        task_id = str(entry.get("task_id") or "unknown")
        out.append(
            Symptom(
                name="stale_lease",
                severity=SymptomSeverity.HIGH,
                summary=(
                    f"lease for task_id={task_id!r} on lane={entry.get('lane')!r} "
                    f"is held by dead pid={entry.get('holder_pid')!r}; "
                    f"downstream proposals on the same lane are blocked"
                ),
                evidence={
                    "task_id": task_id,
                    "holder_pid": entry.get("holder_pid"),
                    "lane": entry.get("lane"),
                    "acquired_at": entry.get("acquired_at"),
                    "age_s": round(age_s, 2),
                    "stale_lease_min_age_s": cfg.stale_lease_min_age_s,
                },
                subject={"task_id": task_id},
                source="local",
                suggestion=(
                    f"kill_task(scope='task', task_id={task_id!r}) to "
                    f"release the lease; the lane will free up for the "
                    f"next pending proposal"
                ),
            )
        )
    return out


def _coerce_unix(value: Any) -> float | None:
    """Coerce a timestamp value to unix seconds.

    Accepts numeric epoch seconds or an ISO-8601-ish string (``Z`` suffix
    tolerated). Booleans are rejected.

    Args:
        value (Any): The raw timestamp value.

    Returns:
        float | None: Unix seconds, or ``None`` when the value cannot be
            interpreted as a timestamp.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            from datetime import datetime
            try:
                return datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# I4 — inbox / outbox bloat
# ---------------------------------------------------------------------------

def _inbox_bloat_symptoms(
    si: dict[str, Any], cfg: StateIntegrityConfig,
) -> list[Symptom]:
    """I4: fire ``inbox_bloat`` for agent inbox/outbox files over threshold.

    Args:
        si (dict[str, Any]): The state-integrity probe sample.
        cfg (StateIntegrityConfig): Tunables (provides the bloat thresholds).

    Returns:
        list[Symptom]: One ``inbox_bloat`` symptom per oversized agent file,
            possibly empty.
    """
    agents = si.get("agents")
    if not isinstance(agents, dict) or not agents:
        return []
    out: list[Symptom] = []
    for role, slot in agents.items():
        if not isinstance(slot, dict):
            continue
        for slot_name in ("inbox_bytes", "outbox_bytes"):
            size = slot.get(slot_name)
            if not isinstance(size, int) or size <= 0:
                continue
            if size >= cfg.inbox_bloat_critical_bytes:
                severity = SymptomSeverity.MEDIUM
            elif size >= cfg.inbox_bloat_warn_bytes:
                severity = SymptomSeverity.LOW
            else:
                continue
            kind = "inbox" if slot_name == "inbox_bytes" else "outbox"
            path_key = f"{kind}_path"
            out.append(
                Symptom(
                    name="inbox_bloat",
                    severity=severity,
                    summary=(
                        f"agent {role!r} {kind}.jsonl at "
                        f"{size / (1024 ** 2):.0f} MiB (warn="
                        f"{cfg.inbox_bloat_warn_bytes / (1024 ** 2):.0f} MiB)"
                    ),
                    evidence={
                        "role": role,
                        "kind": kind,
                        "size_bytes": size,
                        "path": slot.get(path_key),
                        "warn_bytes": cfg.inbox_bloat_warn_bytes,
                        "critical_bytes": cfg.inbox_bloat_critical_bytes,
                    },
                    subject={"role": role, "kind": kind},
                    source="local",
                    suggestion=(
                        "Coordinator should roll the agent log; per-tick "
                        "JSONL parsing cost is now O(file_size)"
                    ),
                )
            )
    return out


# ---------------------------------------------------------------------------
# I5 — coordinator zombie (PID dead but state.json says running)
# ---------------------------------------------------------------------------

def _coordinator_zombie_symptoms(si: dict[str, Any]) -> list[Symptom]:
    """I5: fire ``coordinator_zombie`` when the PID is dead but no stop reason.

    Args:
        si (dict[str, Any]): The state-integrity probe sample.

    Returns:
        list[Symptom]: A one-element list with the ``coordinator_zombie``
            symptom when the Coordinator died ungracefully, otherwise an empty
            list.
    """
    coord = si.get("coordinator")
    state = si.get("state_json")
    if not isinstance(coord, dict) or not coord:
        return []
    pid = coord.get("recorded_pid")
    alive = coord.get("alive")
    # Skip cases we can't judge: no PID file or alive==None (probe didn't run).
    if pid is None or alive is None:
        return []
    if alive:
        return []
    stop_reason = ""
    if isinstance(state, dict) and state.get("valid"):
        stop_reason = str(state.get("stop_reason") or "")
    if stop_reason:
        # Clean wind-down — terminal stop_reason already written.
        return []
    return [
        Symptom(
            name="coordinator_zombie",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"Coordinator pid={pid} not alive but state.json carries "
                f"no stop_reason — the run died without graceful "
                f"shutdown. External monitors may think it's still active"
            ),
            evidence={
                "recorded_pid": pid,
                "pid_file": coord.get("pid_file"),
                "state_json_valid": (
                    bool(state.get("valid")) if isinstance(state, dict) else None
                ),
                "stop_reason": stop_reason or "(empty)",
            },
            subject={},
            source="local",
            suggestion=(
                "operator restart required: Robustness lives inside the "
                "Coordinator process tree, its own subprocess will exit "
                "alongside; check optimizer_runs/run_*.log for the crash"
            ),
        )
    ]


__all__ = [
    "StateIntegrityConfig",
    "evaluate_state_integrity_signals",
]
