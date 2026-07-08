# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


from ._common import (
    _find_benchmark_report,
    _load_json_safe,
    _rel,
    _resolve_under_session,
    _scan_profile_reports,
    _to_float,
)



# §12 Critic / Robustness
def collect_critic_robustness(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the §12 critic / robustness section.

    Walks ``critic-workdir/<iter>/`` for review + emit JSON (verdict, topic,
    truncated summary, artifact paths) and ``robustness-workdir/<iter>/`` for
    signal + action JSON, then summarizes critic verdicts.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: ``{"critic_iterations", "robustness_signals",
        "kb_writes_summary"}``.
    """
    critic_iters: list[dict[str, Any]] = []
    critic_root = session_dir / "critic-workdir"
    if critic_root.exists():
        for iter_dir in sorted(critic_root.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            try:
                iter_n = int(iter_dir.name)
            except ValueError:
                iter_n = -1
            review = _load_json_safe(iter_dir / "review.json", warnings) or {}
            emit = _load_json_safe(iter_dir / "emit.json", warnings) or {}
            critic_iters.append(
                {
                    "iter": iter_n,
                    "ts": str(emit.get("ts") or review.get("ts") or ""),
                    "topic": str(emit.get("topic") or review.get("topic") or ""),
                    "verdict": str(review.get("verdict") or emit.get("verdict") or ""),
                    "summary": str(review.get("summary") or emit.get("summary") or "")[:500],
                    "request_path": _rel(iter_dir / "request.json", session_dir),
                    "judge_bundle_path": _rel(iter_dir / "judge_bundle.json", session_dir),
                    "emit_path": _rel(iter_dir / "emit.json", session_dir),
                    "review_path": _rel(iter_dir / "review.json", session_dir),
                }
            )

    robustness_signals: list[dict[str, Any]] = []
    rob_root = session_dir / "robustness-workdir"
    if rob_root.exists():
        for iter_dir in sorted(rob_root.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            signal_data = _load_json_safe(iter_dir / "signal.json", warnings) or {}
            action_data = _load_json_safe(iter_dir / "action.json", warnings) or {}
            robustness_signals.append(
                {
                    "ts": str(signal_data.get("ts") or action_data.get("ts") or ""),
                    "signal": str(signal_data.get("signal") or signal_data.get("kind") or ""),
                    "action": str(action_data.get("action") or action_data.get("kind") or ""),
                    "workdir": _rel(iter_dir, session_dir) or str(iter_dir),
                }
            )

    # kb_writes_summary: commit-review counts by verdict, reusing the parsed iters.
    kb_writes_summary = _critic_kb_writes_summary(critic_iters)

    return {
        "critic_iterations": critic_iters,
        "robustness_signals": robustness_signals,
        "kb_writes_summary": kb_writes_summary,
    }


def _critic_kb_writes_summary(
    critic_iters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build ``critic_robustness.kb_writes_summary`` by counting each iteration's verdict into ``by_verdict``.

    Args:
        critic_iters (list[dict[str, Any]]): The parsed critic-iteration rows.

    Returns:
        dict[str, Any]: ``{"total", "by_verdict"}`` where ``by_verdict`` counts
        each non-empty, upper-cased verdict.
    """
    by_verdict: dict[str, int] = {}
    total = 0
    for entry in critic_iters:
        verdict = str((entry or {}).get("verdict") or "").strip().upper()
        if not verdict:
            continue
        total += 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    return {
        "total": total,
        "by_verdict": by_verdict,
    }


# §13 Telemetry
def _scan_all_benchmark_reports(session_dir: Path) -> Iterable[Path]:
    """Find every ``benchmark_*/benchmark_report.json`` under ``runs/``.

    Args:
        session_dir (Path): Absolute session root.

    Returns:
        Iterable[Path]: All benchmark reports, sorted. Empty when no ``runs/``
        tree exists.
    """
    runs = session_dir / "runs"
    if not runs.exists():
        return ()
    return sorted(runs.rglob("benchmark_*/benchmark_report.json"))


def _scan_run_dirs(session_dir: Path, pattern: str) -> list[Path]:
    """Find all directories matching ``pattern`` under ``runs/``, sorted.

    Args:
        session_dir (Path): Absolute session root.
        pattern (str): ``rglob`` pattern (e.g. ``torch_trace*`` /
            ``system_profile*``).

    Returns:
        list[Path]: Matching directories, sorted. Empty when none exist.
    """
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(p for p in runs.rglob(pattern) if p.is_dir())


def _scan_server_logs(session_dir: Path) -> list[Path]:
    """Find all ``server*.log`` files under ``runs/``.

    Args:
        session_dir (Path): Absolute session root.

    Returns:
        list[Path]: Matching server log files, sorted. Empty when none exist.
    """
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(runs.rglob("server*.log"))


def _aggregate_gpu_monitor(
    reports: list[Path],
    warnings: list[str],
) -> dict[str, Any]:
    """Aggregate GPU-monitor samples across benchmark reports.

    Collects every ``gpu_monitor`` sample from the given reports and computes
    average / max power, temperature, and average clock.

    Args:
        reports (list[Path]): Benchmark report paths to scan.
        warnings (list[str]): Shared warnings list (mutated in place when a
            report fails to parse).

    Returns:
        dict[str, Any]: Aggregate stats (sample count plus avg/max power,
        avg/max temp, avg clock). ``{}`` when no samples were found.
    """
    samples: list[dict[str, Any]] = []
    for r in reports:
        d = _load_json_safe(r, warnings)
        if not isinstance(d, dict):
            continue
        gm = d.get("gpu_monitor")
        if isinstance(gm, list):
            for s in gm:
                if isinstance(s, dict):
                    samples.append(s)
        elif isinstance(gm, dict):
            samples.append(gm)
    if not samples:
        return {}

    def _avg(key: str) -> float:
        """Mean of a numeric field across the collected samples.

        Args:
            key (str): Sample field name.

        Returns:
            float: The rounded mean of present values, or ``0.0`` when none.
        """
        vals = [_to_float(s.get(key)) for s in samples]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _max(key: str) -> float:
        """Maximum of a numeric field across the collected samples.

        Args:
            key (str): Sample field name.

        Returns:
            float: The rounded max of present values, or ``0.0`` when none.
        """
        vals = [_to_float(s.get(key)) for s in samples]
        vals = [v for v in vals if v is not None]
        return round(max(vals), 2) if vals else 0.0

    return {
        "samples": len(samples),
        "avg_power_w": _avg("power_w") or _avg("power"),
        "max_power_w": _max("power_w") or _max("power"),
        "avg_temp_c": _avg("temperature_c") or _avg("temperature"),
        "max_temp_c": _max("temperature_c") or _max("temperature"),
        "avg_clock_mhz": _avg("clock_mhz") or _avg("sclk_mhz"),
    }


def _collect_lane_timeline(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """v0.8 M6 (KB_design §3.12 §4.5) — per-lane capacity/occupancy summary from ``storage/coordinator.db``.

    One row per lane (capacity, live_holders, lease_expired_count). The
    per-tick holders timeline is deferred; the aggregates suffice for the
    ``benchmark_lane.peak ≤ 1`` invariant check.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place on DB
            errors).

    Returns:
        list[dict[str, Any]]: One per-lane occupancy row plus a ``__total__``
        aggregate row. Empty when the coordinator DB is absent / unreadable.
    """
    db_path = session_dir / "storage" / "coordinator.db"
    if not db_path.exists():
        return []
    import sqlite3 as _sqlite3

    try:
        conn = _sqlite3.connect(str(db_path), timeout=2.0)
        conn.row_factory = _sqlite3.Row
    except _sqlite3.Error as exc:
        warnings.append(f"lane_timeline: open {db_path} failed: {exc!r}")
        return []
    try:
        try:
            cur = conn.execute(
                "SELECT lane, capacity FROM lane_capacity ORDER BY lane",
            )
            capacities = {r["lane"]: int(r["capacity"]) for r in cur.fetchall()}
        except _sqlite3.OperationalError:
            # Older DB without lane_capacity — fall back to defaults.
            from hyperloom.orchestrator.bus.storage.schema import DEFAULT_LANE_CAPACITIES as _DEFAULT

            capacities = dict(_DEFAULT)
        try:
            cur = conn.execute(
                "SELECT lane, COUNT(*) AS n FROM leases WHERE expires_at > datetime('now') GROUP BY lane",
            )
            holders = {r["lane"]: int(r["n"]) for r in cur.fetchall()}
        except _sqlite3.OperationalError as exc:
            warnings.append(f"lane_timeline: leases query failed: {exc!r}")
            holders = {}
        try:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE topic = 'lease_expired'",
            )
            row = cur.fetchone()
            expired_total = int(row["n"]) if row else 0
        except _sqlite3.OperationalError:
            expired_total = 0
        # Per-lane expired count (lane is in the lease_expired payload).
        per_lane_expired: dict[str, int] = {}
        try:
            cur = conn.execute(
                "SELECT payload FROM events WHERE topic = 'lease_expired'",
            )
            for r in cur.fetchall():
                try:
                    p = json.loads(r["payload"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                lane = str(p.get("lane") or "")
                if lane:
                    per_lane_expired[lane] = per_lane_expired.get(lane, 0) + 1
        except _sqlite3.OperationalError:
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    rows: list[dict[str, Any]] = []
    for lane in sorted(set(capacities) | set(holders)):
        rows.append(
            {
                "lane": lane,
                "capacity": int(capacities.get(lane, 1)),
                "live_holders": int(holders.get(lane, 0)),
                "lease_expired_count": int(per_lane_expired.get(lane, 0)),
            }
        )
    # Append a totals row for consumers that aggregate across lanes.
    if rows:
        rows.append(
            {
                "lane": "__total__",
                "capacity": sum(r["capacity"] for r in rows),
                "live_holders": sum(r["live_holders"] for r in rows),
                "lease_expired_count": int(expired_total),
            }
        )
    return rows


def collect_telemetry(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the §13 telemetry section.

    Gathers references to the baseline / profile benchmark reports, torch
    traces, system profiles, and server logs on disk, aggregates GPU-monitor
    stats across all reports, and attaches the per-lane occupancy summary.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: Telemetry section with artifact path lists, the GPU
        monitor aggregate, and the lane timeline.
    """
    baseline_report: Path | None = None
    last_b = state.get("last_baseline") or {}
    if isinstance(last_b, dict) and last_b.get("workspace"):
        workspace = _resolve_under_session(session_dir, last_b.get("workspace"))
        if workspace is not None:
            baseline_report = _find_benchmark_report(workspace)

    profile_reports = [report for _task, report in _scan_profile_reports(session_dir)]
    all_reports = list(_scan_all_benchmark_reports(session_dir))

    return {
        "baseline_report_path": _rel(baseline_report, session_dir) if baseline_report else None,
        "profile_report_paths": [_rel(p, session_dir) or str(p) for p in profile_reports],
        "torch_trace_paths": [_rel(p, session_dir) or str(p) for p in _scan_run_dirs(session_dir, "torch_trace*")],
        "system_profile_paths": [_rel(p, session_dir) or str(p) for p in _scan_run_dirs(session_dir, "system_profile*")],
        "server_log_paths": [_rel(p, session_dir) or str(p) for p in _scan_server_logs(session_dir)],
        "gpu_monitor_aggregate": _aggregate_gpu_monitor(all_reports, warnings),
        # per-lane occupancy / capacity summary from the leases DB.
        "lane_timeline": _collect_lane_timeline(session_dir, warnings),
    }


# §15 KB Provenance — Cortex KB integration audit
def collect_kb_provenance(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the Cortex KB integration audit, merging SharedState warm-start fields, the NDJSON queue counts, and the audit-log status tail.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place; also used
            to surface PR-monitor / flusher status markers).

    Returns:
        dict[str, Any]: The KB-provenance section (warm-start fields, queue
        counts, audit-status tail, recipe-snapshot reads, and flusher status).
    """
    from ...session.session_paths import (
        cortex_audit_jsonl as _audit_path,
        cortex_dead_letter_ndjson as _dl_path,
        cortex_flushed_ndjson as _flushed_path,
        cortex_flusher_pid as _flusher_pid_path,
        cortex_flusher_status_json as _flusher_status_path,
        cortex_pending_ndjson as _pending_path,
        pr_monitor_status_json as _pr_status_path,
        recipe_snapshot_audit_jsonl as _recipe_audit_path,
    )

    # Surface the PR Monitor reachability snapshot via ``warnings`` so it's greppable.
    pr_status_path = _pr_status_path(session_dir)
    if pr_status_path.exists():
        try:
            with pr_status_path.open("r", encoding="utf-8") as f:
                pr_status = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"pr_monitor:status_marker_unreadable:{exc!r}"[:240])
        else:
            if not pr_status.get("enabled"):
                warnings.append("pr_monitor:disabled")
            elif not pr_status.get("reachable"):
                # Dashboard keys the PR Monitor ingress alert on this exact string.
                url = str(pr_status.get("url") or "")
                warnings.append(f"pr_monitor:unreachable:{url}"[:240] if url else "pr_monitor:unreachable")

    def _count_lines(p: Path) -> int:
        """Count non-blank lines in a file, recording read errors.

        Args:
            p (Path): File to scan.

        Returns:
            int: Number of non-blank lines, or ``0`` when missing / unreadable.
        """
        try:
            if not p.exists():
                return 0
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError as exc:
            warnings.append(f"kb_provenance: failed to count {p}: {exc!r}")
            return 0

    def _read_last_n_audit(p: Path, n: int = 50) -> list[dict[str, Any]]:
        """Read the last ``n`` JSON rows of an audit log, never raising.

        Args:
            p (Path): The ``.jsonl`` audit log.
            n (int): Maximum number of trailing rows to return. Defaults to 50.

        Returns:
            list[dict[str, Any]]: Up to the last ``n`` parsed rows, or ``[]``
            when the file is missing / unreadable.
        """
        try:
            if not p.exists():
                return []
            with p.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            return rows[-n:]
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"kb_provenance: failed to read audit {p}: {exc!r}")
            return []

    pending_path = _pending_path(session_dir)
    flushed_path = _flushed_path(session_dir)
    dl_path = _dl_path(session_dir)
    audit_path = _audit_path(session_dir)

    audit_tail = _read_last_n_audit(audit_path, n=50)
    status_counts: dict[str, int] = {}
    for row in audit_tail:
        st = str(row.get("status") or "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    # Recipe-snapshot / gbrain remote read audit (RecipeKB.audit_hook ->
    # recipe_snapshot/.audit.jsonl). Summarises whether the snapshot KB was
    # actually consulted, which backend served it, and how each read resolved.
    recipe_audit = _read_last_n_audit(_recipe_audit_path(session_dir), n=50)
    recipe_by_resolution: dict[str, int] = {}
    recipe_by_remote: dict[str, int] = {}
    # Per-path (e.g. gbrain vs cortex) attribution derived from the composite
    # remote's provenance, emitted by the dispatcher audit. ``by_source``
    # counts how often each path contributed a returned row; ``best_config_by
    # _source`` counts which path supplied the replayable champion config.
    recipe_by_source: dict[str, int] = {}
    recipe_best_config_by_source: dict[str, int] = {}
    recipe_hits = 0
    for row in recipe_audit:
        recipe_by_resolution[str(row.get("resolution") or "unknown")] = (
            recipe_by_resolution.get(str(row.get("resolution") or "unknown"), 0) + 1
        )
        recipe_by_remote[str(row.get("remote") or "unknown")] = (
            recipe_by_remote.get(str(row.get("remote") or "unknown"), 0) + 1
        )
        if row.get("hit"):
            recipe_hits += 1
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        for src in (result.get("sources") or []):
            recipe_by_source[str(src)] = recipe_by_source.get(str(src), 0) + 1
        best_config_src = result.get("best_config_source")
        for src in (
            best_config_src if isinstance(best_config_src, list)
            else [best_config_src] if best_config_src else []
        ):
            recipe_best_config_by_source[str(src)] = (
                recipe_best_config_by_source.get(str(src), 0) + 1
            )

    cortex_sid = (state.get("cortex_session_id") or "").strip()
    warm = state.get("warm_start_recipe") or {}
    # FINAL reference attribution: which path supplied the warm recipe that was
    # actually applied this session. Prefer the merged row's field provenance,
    # then the WarmStartContext source tag set at T0.
    warm_recipe_row = warm.get("recipe") if isinstance(warm, dict) else {}
    warm_start_recipe_source = ""
    # 1) Precise self-case: the merged identity row owns its replayable config,
    #    so its per-field provenance is the authoritative applied-config source.
    if isinstance(warm_recipe_row, dict):
        warm_field_sources = warm_recipe_row.get("_field_sources")
        if isinstance(warm_field_sources, dict):
            bc_src = warm_field_sources.get("best_config")
            if isinstance(bc_src, str) and bc_src:
                warm_start_recipe_source = bc_src
            elif isinstance(bc_src, list) and bc_src:
                warm_start_recipe_source = str(bc_src[0])
    # 2) Config-donor case: the identity row carries no replayable config (its
    #    best_config was borrowed from a sibling), so the T0 WarmStartContext
    #    source — resolved donor-aware in _warm_recipe_source — is authoritative
    #    over the identity row's generic _sources.
    if not warm_start_recipe_source:
        wsc = state.get("warm_start_context") or {}
        warm_start_recipe_source = str(
            ((wsc.get("match") or {}).get("source") or "")
        ) if isinstance(wsc, dict) else ""
    # 3) Last resort: the identity row's first contributing source.
    if not warm_start_recipe_source and isinstance(warm_recipe_row, dict):
        warm_sources = warm_recipe_row.get("_sources")
        if isinstance(warm_sources, list) and warm_sources:
            warm_start_recipe_source = str(warm_sources[0])
    pitfalls = state.get("warm_start_pitfalls") or []
    lessons = state.get("warm_start_lessons") or []
    # warm-recipe replay outcome; empty before completion / when --no-warm-replay.
    warm_replay_outcome = state.get("warm_replay_outcome") or {}

    out: dict[str, Any] = {
        "cortex_session_id": cortex_sid,
        "warm_start_ts": state.get("warm_start_ts") or "",
        "warm_start_recipe_seen": bool(warm and warm.get("raw")),
        "warm_start_recipe_tier": str(warm.get("tier") or "") if isinstance(warm, dict) else "",
        "warm_start_recipe_source": warm_start_recipe_source,
        "warm_start_pitfall_count": len(pitfalls) if isinstance(pitfalls, list) else 0,
        "warm_start_lesson_count": len(lessons) if isinstance(lessons, list) else 0,
        # operator-visible replay summary, passed through verbatim.
        "warm_replay": dict(warm_replay_outcome) if isinstance(warm_replay_outcome, dict) else {},
        "warm_replay_attempted": bool(state.get("warm_replay_attempted")),
        "warm_history_injected": bool(state.get("warm_history_injected")),
        "stack_fingerprint": manifest.get("stack_fingerprint") or {},
        "queue": {
            "pending_lines": _count_lines(pending_path),
            "flushed_bookmarks": _count_lines(flushed_path),
            "dead_letter_lines": _count_lines(dl_path),
        },
        "audit_tail_count": len(audit_tail),
        "audit_status_counts": status_counts,
        "recipe_snapshot_reads": {
            "count": len(recipe_audit),
            "hits": recipe_hits,
            "by_resolution": recipe_by_resolution,
            "by_remote": recipe_by_remote,
            "by_source": recipe_by_source,
            "best_config_by_source": recipe_best_config_by_source,
            "tail": recipe_audit[-10:],
        },
        "flusher_status": _collect_flusher_status(
            session_dir,
            status_path=_flusher_status_path(session_dir),
            pid_path=_flusher_pid_path(session_dir),
            warnings=warnings,
        ),
        "kb_degraded_reason": (manifest.get("kb_degraded_reason") or "") or None,
        "pr_degraded_reason": (manifest.get("pr_degraded_reason") or "") or None,
    }
    fs = out["flusher_status"]
    # Only warn when a boot marker exists; a missing one is a legacy session, not a misconfig.
    if fs.get("reason") != "no_marker":
        if not fs.get("enabled", True):
            warnings.append("kb_flusher:disabled")
        elif not fs.get("alive", False):
            warnings.append("kb_flusher:not_alive")
    return out


def _collect_flusher_status(
    session_dir: Path,
    *,
    status_path: Path,
    pid_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Merge the ``.kb_flusher_status.json`` boot marker with a live pid probe into one stable shape.

    Args:
        session_dir (Path): Absolute session root.
        status_path (Path): The ``.kb_flusher_status.json`` boot marker path.
        pid_path (Path): The flusher pid-file path used for the liveness probe.
        warnings (list[str]): Shared warnings list (mutated in place when the
            marker is unreadable).

    Returns:
        dict[str, Any]: The merged flusher-status dict (enabled / spawned /
        alive / pid / config / reason).
    """
    base: dict[str, Any] = {
        "enabled": False,
        "spawned": False,
        "alive": False,
        "pid": None,
        "cortex_kb_url": None,
        "interval_sec": 0.0,
        "batch_size": 0,
        "reason": "no_marker",
        "ts": "",
        "pid_path": str(pid_path),
    }
    if status_path.exists():
        try:
            with status_path.open("r", encoding="utf-8") as f:
                marker = json.load(f)
            if isinstance(marker, dict):
                for k in (
                    "enabled",
                    "spawned",
                    "pid",
                    "cortex_kb_url",
                    "interval_sec",
                    "batch_size",
                    "reason",
                    "ts",
                    "pid_path",
                ):
                    if k in marker:
                        base[k] = marker[k]
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"kb_flusher:status_marker_unreadable:{exc!r}"[:240])

    pid_alive = False
    pid_from_file: int | None = None
    if pid_path.exists():
        try:
            raw = pid_path.read_text(encoding="utf-8").strip().splitlines()
            pid_from_file = int(raw[0]) if raw else None
        except (OSError, ValueError):
            pid_from_file = None
        if pid_from_file:
            try:
                os.kill(pid_from_file, 0)
                pid_alive = True
            except (OSError, ProcessLookupError):
                pid_alive = False
    if pid_from_file and not base.get("pid"):
        base["pid"] = pid_from_file
    base["alive"] = pid_alive
    return base


# specialist_runs section
def _coerce_round_id(value: Any) -> int | str:
    """Normalise ``round_id`` to int when purely numeric, else keep the string (empty/None → 0). Never raises.

    Args:
        value (Any): The raw ``round_id`` value.

    Returns:
        int | str: The integer round id when ``value`` is purely numeric,
        ``0`` for empty / ``None``, else the original string.
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return text


def collect_specialist_runs(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
    *,
    include_transcripts: bool = False,
) -> list[dict[str, Any]]:
    """Build the ``specialist_runs`` section by merging ``state.specialist_rounds[]`` with on-disk transcripts; best-effort.

    ``include_transcripts`` inlines the transcript bytes under each ref's
    ``body`` (default False = path-only).

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place on scan /
            read failures).
        include_transcripts (bool): When ``True``, inline each transcript's
            bytes under its ``body``. Defaults to ``False`` (path-only).

    Returns:
        list[dict[str, Any]]: One shaped specialist-round row (with transcript
        refs). Empty when no rounds exist.
    """
    rounds = state.get("specialist_rounds") or []
    if not isinstance(rounds, list) or not rounds:
        return []

    # Pre-index runs/specialist/ for O(1) per-task lookup.
    runs_root = session_dir / "runs" / "specialist"
    by_task: dict[str, Path] = {}
    if runs_root.exists():
        try:
            for child in runs_root.iterdir():
                if not child.is_dir():
                    continue
                done_path = child / "specialist_done.json"
                if done_path.exists():
                    by_task[child.name] = done_path
        except OSError as exc:
            warnings.append(f"specialist_runs: failed to scan {runs_root}: {exc!r}")

    out: list[dict[str, Any]] = []
    for raw in rounds:
        if not isinstance(raw, dict):
            continue
        # Tolerate both singular (domain/task_id/confidence) and plural shapes.
        domains = list(raw.get("domains") or [])
        if not domains and raw.get("domain"):
            domains = [str(raw.get("domain"))]
        entry: dict[str, Any] = {
            "round_id": _coerce_round_id(raw.get("round_id")),
            "dispatched_at": str(raw.get("dispatched_at") or ""),
            "completed_at": str(raw.get("completed_at") or ""),
            "domains": domains,
            "tags": list(raw.get("tags") or []),
            "parallelism": int(raw.get("parallelism") or 0),
            "proposals_total": int(raw.get("proposals_total") or 0),
            "proposals_kept": int(raw.get("proposals_kept") or 0),
            "proposals_rejected": int(raw.get("proposals_rejected") or 0),
            "proposals_skipped": int(raw.get("proposals_skipped") or 0),
            "kb_edge_ids": list(raw.get("kb_edge_ids") or []),
            "confidence_avg": _to_float(
                raw.get("confidence_avg") if raw.get("confidence_avg") is not None else raw.get("confidence")
            ),
            "domain_breakdown": _normalize_specialist_domain_breakdown(
                raw.get("domain_breakdown"),
            ),
            "notes": list(raw.get("notes") or []),
        }
        # Attach transcript refs, tolerating a singular ``task_id`` anchor.
        task_ids = list(raw.get("task_ids") or [])
        if not task_ids and raw.get("task_id"):
            task_ids = [str(raw.get("task_id"))]
        transcripts: list[dict[str, Any]] = []
        for tid in task_ids:
            tid_str = str(tid)
            done_path = by_task.get(tid_str)
            if done_path is None:
                continue
            ref: dict[str, Any] = {
                "task_id": tid_str,
                "domain": _domain_for_task(raw, tid_str),
                "path": _rel(done_path, session_dir) or str(done_path),
            }
            if include_transcripts:
                try:
                    ref["body"] = done_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError as exc:
                    warnings.append(f"specialist_runs: cannot read transcript {done_path}: {exc!r}")
            transcripts.append(ref)
        entry["transcripts"] = transcripts
        out.append(entry)
    return out


def _normalize_specialist_domain_breakdown(
    raw: Any,
) -> dict[str, dict[str, int]]:
    """Coerce a round's per-domain breakdown to a stable int-counted shape.

    Args:
        raw (Any): The raw ``domain_breakdown`` value from a specialist round.

    Returns:
        dict[str, dict[str, int]]: Per-domain counts (dispatched /
        proposals_total / proposals_kept / proposals_rejected). ``{}`` when
        ``raw`` is not a dict.
    """
    if not isinstance(raw, dict):
        return {}
    norm: dict[str, dict[str, int]] = {}
    for domain, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        norm[str(domain)] = {
            "dispatched": int(payload.get("dispatched") or 0),
            "proposals_total": int(payload.get("proposals_total") or 0),
            "proposals_kept": int(payload.get("proposals_kept") or 0),
            "proposals_rejected": int(payload.get("proposals_rejected") or 0),
        }
    return norm


def _domain_for_task(round_entry: dict[str, Any], task_id: str) -> str:
    """Best-effort domain for ``task_id`` within a round; "" when unmapped (older M5 rounds).

    Args:
        round_entry (dict[str, Any]): One specialist-round record.
        task_id (str): The task id to resolve a domain for.

    Returns:
        str: The mapped domain, an unambiguous single tag/domain, or ``""``
        when it cannot be determined.
    """
    mapping = round_entry.get("task_domains")
    if isinstance(mapping, dict):
        v = mapping.get(task_id)
        if isinstance(v, str):
            return v
    # Fallback: a round with exactly one tag/domain attributes unambiguously.
    domains = round_entry.get("tags") or round_entry.get("domains") or []
    if isinstance(domains, list) and len(domains) == 1:
        return str(domains[0])
    if round_entry.get("domain"):
        return str(round_entry.get("domain"))
    return ""

