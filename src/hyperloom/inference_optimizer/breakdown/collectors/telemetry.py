# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``."""

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
    """Collect the critic / robustness section."""
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
    """Build ``critic_robustness.kb_writes_summary`` by counting each iteration's verdict into ``by_verdict``."""
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
    """Find every ``benchmark_*/benchmark_report.json`` under ``runs/``."""
    runs = session_dir / "runs"
    if not runs.exists():
        return ()
    return sorted(runs.rglob("benchmark_*/benchmark_report.json"))


def _scan_run_dirs(session_dir: Path, pattern: str) -> list[Path]:
    """Find all directories matching ``pattern`` under ``runs/``, sorted."""
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(p for p in runs.rglob(pattern) if p.is_dir())


def _scan_server_logs(session_dir: Path) -> list[Path]:
    """Find all ``server*.log`` files under ``runs/``."""
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(runs.rglob("server*.log"))


# ``gpu_monitor`` metric aliases, current producer name first. Magpie -- the
# only writer in the tree today -- emits ``power_watts`` / ``temperature_c`` /
# ``gpu_clock_mhz``. The shorter spellings are older shapes, kept so archived
# reports still parse. Reading only the short names is what left every
# single-node session's GPU section reading 0.0.
_GPU_POWER_KEYS = ("power_watts", "power_w", "power")
_GPU_TEMP_KEYS = ("temperature_c", "temp_c", "temperature")
_GPU_CLOCK_KEYS = ("gpu_clock_mhz", "clock_mhz", "sclk_mhz")


def _gpu_metric(block: dict[str, Any], keys: tuple[str, ...], field: str) -> float | None:
    """Read one GPU metric out of a single ``gpu_monitor`` block.

    Two producer shapes exist and both are read here:

    * a flat scalar per sample (``{"power_w": 301.2}``) -- ``field`` does not
      apply, the scalar is both that sample's mean and its peak;
    * a pre-aggregated block (``{"power_watts": {"min", "max", "avg"}}``),
      which is what Magpie's ``GPUMonitor`` returns. ``field`` picks the
      statistic. The previous implementation coerced this dict through
      ``_to_float``, which returns ``None`` for a dict, so the value was
      dropped even when the key name matched.

    Args:
        block (dict[str, Any]): One ``gpu_monitor`` entry.
        keys (tuple[str, ...]): Metric aliases to try, in order.
        field (str): ``"avg"`` or ``"max"``; consulted only for nested blocks.

    Returns:
        float | None: The reading, or ``None`` when this block carries none.
        ``None`` rather than ``0.0`` on purpose -- a metric that was never
        sampled has to stay distinguishable from one that measured zero.
    """
    for key in keys:
        if key not in block:
            continue
        raw = block[key]
        value = _to_float(raw.get(field)) if isinstance(raw, dict) else _to_float(raw)
        if value is not None:
            return value
    return None


def _aggregate_gpu_monitor(
    reports: list[Path],
    warnings: list[str],
) -> dict[str, Any]:
    """Aggregate GPU-monitor blocks across benchmark reports.

    Every metric is tri-state and never coerced to ``0.0``: absent and "measured
    zero" have to stay apart, and the previous ``_avg(a) or _avg(b)`` form could
    not express that -- a real 0.0 fell through to the alias, and an all-absent
    metric shipped as a plausible-looking zero. Returns ``{}`` when no report
    carried a ``gpu_monitor`` block.
    """
    blocks: list[dict[str, Any]] = []
    for r in reports:
        d = _load_json_safe(r, warnings)
        if not isinstance(d, dict):
            continue
        gm = d.get("gpu_monitor")
        if isinstance(gm, list):
            for s in gm:
                if isinstance(s, dict):
                    blocks.append(s)
        elif isinstance(gm, dict):
            blocks.append(gm)
    if not blocks:
        return {}

    # A Magpie block already summarises ``sample_count`` underlying samples, so
    # weight its mean by that count; a flat per-sample block counts as one.
    # Unweighted, a 10-sample block would pull the session mean as hard as a
    # 10,000-sample one.
    weights = [max(1.0, _to_float(b.get("sample_count")) or 1.0) for b in blocks]

    def _avg(keys: tuple[str, ...]) -> float | None:
        """Sample-count-weighted mean of one metric, or ``None`` if unread."""
        total = 0.0
        weight_sum = 0.0
        for block, weight in zip(blocks, weights):
            value = _gpu_metric(block, keys, "avg")
            if value is None:
                continue
            total += value * weight
            weight_sum += weight
        return round(total / weight_sum, 2) if weight_sum else None

    def _max(keys: tuple[str, ...]) -> float | None:
        """Peak of one metric across all blocks, or ``None`` if unread."""
        values: list[float] = []
        for block in blocks:
            value = _gpu_metric(block, keys, "max")
            if value is not None:
                values.append(value)
        return round(max(values), 2) if values else None

    return {
        "samples": int(sum(weights)),
        "blocks": len(blocks),
        "avg_power_w": _avg(_GPU_POWER_KEYS),
        "max_power_w": _max(_GPU_POWER_KEYS),
        "avg_temp_c": _avg(_GPU_TEMP_KEYS),
        "max_temp_c": _max(_GPU_TEMP_KEYS),
        "avg_clock_mhz": _avg(_GPU_CLOCK_KEYS),
    }


def _collect_lane_timeline(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Per-lane capacity/occupancy summary from ``storage/coordinator.db``."""
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


def collect_telemetry(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the telemetry section."""
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
        "orchestration_context": {"tick_count": int(state.get("tick") or 0)},
    }


# specialist_runs section
def _coerce_round_id(value: Any) -> int | str:
    """Normalise ``round_id`` to int when purely numeric, else keep the string (empty/None → 0). Never raises."""
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
    """Build the ``specialist_runs`` section by merging ``state.specialist_rounds[]`` with on-disk transcripts; best-effort."""
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
    """Coerce a round's per-domain breakdown to a stable int-counted shape."""
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
    """Best-effort domain for ``task_id`` within a round; \"\" when unmapped (older M5 rounds)."""
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
