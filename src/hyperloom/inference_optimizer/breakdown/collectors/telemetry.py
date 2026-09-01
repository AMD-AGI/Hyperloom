# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import json
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


# Critic / Robustness
def collect_critic_robustness(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the critic / robustness section.

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


# Telemetry
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
    """Per-lane capacity/occupancy summary from ``storage/coordinator.db``.

    One row per lane (capacity, live_holders, lease_expired_count) plus a
    ``__total__`` aggregate. ``live_holders`` is a point-in-time count of
    unexpired leases, not a peak; no per-tick holders timeline is recorded.

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
            # Telemetry DB locked/absent; skip this expiry pass.
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            # Closing a read-only telemetry connection is best-effort.
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


def _collect_orchestration_context(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Summarize the orchestration conversation's compaction loop.

    Joins the SEED/DELTA census on ``state.json`` with the
    ``orchestration_checkpoint`` events in ``storage/coordinator.db``.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place on DB
            errors).

    Returns:
        dict[str, Any]: The ``orchestration_context`` section; counts are 0
        when the session predates the census or the DB is unreadable.
    """
    modes = state.get("orchestration_prompt_modes")
    modes = modes if isinstance(modes, dict) else {}
    seed = int(modes.get("seed") or 0)
    delta = int(modes.get("delta") or 0)
    tick_count = int(state.get("tick") or 0)

    levels: list[int] = []
    compactions = 0
    degenerate = 0
    db_path = session_dir / "storage" / "coordinator.db"
    if db_path.exists():
        import sqlite3 as _sqlite3

        try:
            conn = _sqlite3.connect(f"file:{db_path}?mode=ro", timeout=2.0, uri=True)
            try:
                cur = conn.execute(
                    "SELECT payload FROM events WHERE payload LIKE '%orchestration_checkpoint%'",
                )
                for row in cur.fetchall():
                    try:
                        payload = json.loads(row[0] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        continue
                    kind = str(payload.get("kind") or "")
                    if kind == "orchestration_checkpoint":
                        compactions += 1
                        level = payload.get("context_tokens")
                        if isinstance(level, int) and level > 0:
                            levels.append(level)
                    # The repeat-degeneracy advisory re-emits the same kind with
                    # a severity; count only the first, per-checkpoint one.
                    elif kind == "orchestration_checkpoint_degraded" and not payload.get("severity"):
                        degenerate += 1
            finally:
                conn.close()
        except _sqlite3.Error as exc:
            warnings.append(f"orchestration_context: read {db_path} failed: {exc!r}")

    levels.sort()
    at_compaction: dict[str, int] = {}
    if levels:
        at_compaction = {
            "min": levels[0],
            "median": levels[len(levels) // 2],
            "max": levels[-1],
        }
    pushes = seed + delta
    return {
        "seed_prompts": seed,
        "delta_prompts": delta,
        "compactions": compactions,
        "degenerate_compactions": degenerate,
        "tick_count": tick_count,
        "compactions_per_tick": round(compactions / tick_count, 4) if tick_count else 0.0,
        "delta_ratio": round(delta / pushes, 4) if pushes else 0.0,
        "context_tokens_at_compaction": at_compaction,
    }


def collect_telemetry(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the telemetry section.

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
        "system_profile_paths": [
            _rel(p, session_dir) or str(p) for p in _scan_run_dirs(session_dir, "system_profile*")
        ],
        "server_log_paths": [_rel(p, session_dir) or str(p) for p in _scan_server_logs(session_dir)],
        "gpu_monitor_aggregate": _aggregate_gpu_monitor(all_reports, warnings),
        # per-lane occupancy / capacity summary from the leases DB.
        "lane_timeline": _collect_lane_timeline(session_dir, warnings),
        # SEED/DELTA census + compaction rate for the orchestration loop.
        "orchestration_context": _collect_orchestration_context(session_dir, state, warnings),
    }


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

    ``include_transcripts`` inlines the transcript bytes under each ref's ``body``.

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
