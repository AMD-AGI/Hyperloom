# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure functions that turn raw session records into the views the page shows.

The main job is to surface what the raw data hides: a specialist whose
subprocess crashed still lands as a ``succeeded`` task, and a stuck round
shows up only as files that stop growing.
"""

from __future__ import annotations

from typing import Any

# Keys of state.json the header and panels read. Everything else stays server-side.
STATE_KEYS = (
    "session_id",
    "model_name",
    "phase",
    "tick",
    "start_ts",
    "stop_ts",
    "stop_reason",
    "elapsed_charged_sec",
    "max_minutes",
    "baseline_tput",
    "crash_count",
    "degraded_mode",
    "gpu_type",
    "tp",
    "ep",
    "nodes",
    "conc",
    "isl",
    "osl",
    "precision",
    "framework",
    "framework_version",
    "agent_last_active",
    "cumulative_gain_validated",
)

# Task states and health values, and the baseline run kind, that the views spell several times.
_CRASHED = "crashed"
_FAILED = "failed"
_RUNNING = "running"
_BASELINE = "baseline"
# Severity levels of the top-of-page badges.
_LEVEL_ERROR = "error"
_LEVEL_WARN = "warn"
_LEVEL_INFO = "info"

# Runner statuses of a specialist whose subprocess ended badly. ``stale`` (nothing usable left behind) and ``partial``
# (it died after writing specialist_done.json, or its work was salvaged from a checkpoint) are the ones the specialist
# runner writes (orchestrator/specialists/runner.py, SpecialistRunner); failure_class.classify_specialist_failure
# counts both as failures. The rest are older spellings.
STALE_RUNNER = {"stale", "partial", _CRASHED, "crash", "timeout", "killed"}
# Leading tokens of the failure reasons the specialist runner records for a reaped subprocess
# (orchestrator/specialists/runner.py, _subprocess_backend_error). Mirrored, not imported: the Inspector may not
# import the orchestrator (the import-linter layers and acyclic contracts); the health contract tests call the
# producer and fail when a token changes.
_SUBPROCESS_FAILURES = ("subprocess_timeout", "subprocess_stale_heartbeat", "subprocess_error", "subprocess_exit_code")
_CLEAN_EXIT = "subprocess_exit_code:0"


def state_summary(state: dict | None) -> dict:
    if not state:
        return {}
    out = {k: state.get(k) for k in STATE_KEYS if k in state}
    best = state.get("current_best") or {}
    out["current_best"] = {k: best.get(k) for k in ("action", "hot_tput", "cold_tput", "e2el_mean_ms")}
    out["failures_count"] = len(state.get("failures") or [])
    out["model_warnings"] = [
        w.get("detail", w) if isinstance(w, dict) else w for w in (state.get("model_warnings") or [])
    ][:10]
    out["phase_history"] = (state.get("phase_history") or [])[-100:]
    out["specialist_rounds"] = [
        {
            k: r.get(k)
            for k in (
                "task_id",
                "domain",
                "reason",
                "summary",
                "proposals_total",
                "confidence",
                "completed_at",
                "source_phase",
            )
        }
        for r in (state.get("specialist_rounds") or [])[-100:]
    ]
    return out


def _subprocess_failed(reason: str) -> bool:
    """Whether *reason* says a specialist subprocess timed out, went stale, errored or exited non-zero."""
    return any(token in reason for token in _SUBPROCESS_FAILURES) and not reason.endswith(_CLEAN_EXIT)


def result_health(result: Any) -> tuple[str, str]:
    """Classify a delegated task result into (health, reason)."""
    if not isinstance(result, dict):
        return "ok", ""
    runner = str(result.get("runner_status") or "").lower()
    reason = str(result.get("reason") or result.get("error") or "")
    exit_code = result.get("subprocess_exit_code", result.get("exit_code"))
    if runner in STALE_RUNNER:
        return _CRASHED, f"runner_status={runner}" + (f": {reason}" if reason else "")
    if isinstance(exit_code, int) and exit_code != 0:
        return _CRASHED, f"exit code {exit_code}"
    if _subprocess_failed(reason):
        return _CRASHED, reason
    if str(result.get("status") or "").lower() in {_FAILED, "error"}:
        return _FAILED, reason
    return "ok", ""


def task_health(
    task: dict, result: Any, specialist_reason: str | None, log_age_sec: float | None, stall_after_sec: float
) -> dict:
    """Health of one task: ok / running / stalled / crashed / failed / cancelled / queued."""
    state = task.get("state")
    if state == _RUNNING:
        if log_age_sec is not None and log_age_sec > stall_after_sec:
            return {"health": "stalled", "reason": f"no output for {int(log_age_sec // 60)} min"}
        return {"health": _RUNNING, "reason": ""}
    if state in (_FAILED, "cancelled", "queued"):
        return {"health": state, "reason": ""}
    health, reason = result_health(result)
    # Every way a subprocess ends badly is ``crashed``, as the runner statuses above are: the task is recorded as
    # succeeded although its subprocess did not finish, and the reason says how it ended.
    if health == "ok" and specialist_reason and _subprocess_failed(specialist_reason):
        health, reason = _CRASHED, specialist_reason
    return {"health": health, "reason": reason}


def task_state_at(task: dict, clock: float, parse_ts) -> dict | None:
    """Replay helper: the task as it looked at unix time *clock*; ``None`` if not yet created."""
    created = parse_ts(task.get("created_at"))
    if created is not None and created > clock:
        return None
    state = "queued"
    updated = task.get("created_at")
    for tr in task.get("history") or []:
        ts = parse_ts(tr.get("ts"))
        if ts is not None and ts > clock:
            break
        state = tr.get("to") or state
        updated = tr.get("ts") or updated
    return {**task, "state": state, "updated_at": updated}


PROGRESS_UNITS = ("variant", "baseline_round")


def _progress_notes(history: list | None, clock: float | None, parse_ts) -> list[dict]:
    notes = []
    for h in history or []:
        if not isinstance(h, dict) or not isinstance(h.get("progress"), dict):
            continue
        ts = parse_ts(h.get("ts"))
        if clock is not None and ts is not None and ts > clock:
            break
        notes.append({**h["progress"], "ts": ts})
    return notes


def task_progress(history: list | None, clock: float | None, parse_ts, outcomes: Any = None) -> dict:
    """Progress notes a task recorded in its history, up to *clock* (None: all).

    ``current`` is the latest note; ``units`` has one row per finished variant or baseline round, with the
    failure ``reason`` and ``stage`` from the result's ``per_variant_outcomes`` when it has landed.
    """
    notes = _progress_notes(history, clock, parse_ts)
    if not notes:
        return {}
    why = {
        o.get("variant_name"): {"reason": o.get("reason"), "stage": o.get("stage")}
        for o in (outcomes if isinstance(outcomes, list) else [])
        if isinstance(o, dict) and o.get("outcome") == "FAILED"
    }
    units: list[dict] = []
    for n in notes:
        if n.get("unit") not in PROGRESS_UNITS or n.get("status") in ("started", _RUNNING):
            continue
        row = {k: n.get(k) for k in ("unit", "label", "status", "output_throughput", "ts")}
        row.update(why.get(n.get("label"), {}))
        # A variant reports once per round (warm-up, then measure): keep one row, with the earlier values.
        if units and (units[-1]["unit"], units[-1]["label"]) == (row["unit"], row["label"]):
            prev = units.pop()
            row["earlier"] = [*prev.get("earlier", []), prev.get("output_throughput")]
        units.append(row)
    current = {k: notes[-1].get(k) for k in ("unit", "label", "status", "index", "total", "ts")}
    return {"current": current, "units": units[-100:], "failed": sum(u["status"] == _FAILED for u in units)}


def llm_summary(calls: list[dict]) -> dict:
    """Per-component call counts, token totals, latency and error counts."""
    by_comp: dict[str, dict] = {}
    for c in calls:
        comp = str(c.get("component") or c.get("role") or "unknown")
        agg = by_comp.setdefault(
            comp,
            {
                "calls": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "latency_ms_sum": 0,
                "models": {},
            },
        )
        agg["calls"] += 1
        if (c.get("status") or "ok") != "ok":
            agg["errors"] += 1
        agg["input_tokens"] += c.get("input_tokens") or 0
        agg["output_tokens"] += c.get("output_tokens") or 0
        agg["cache_read_tokens"] += c.get("cache_read_input_tokens") or 0
        agg["cache_write_tokens"] += c.get("cache_creation_input_tokens") or 0
        agg["latency_ms_sum"] += c.get("latency_ms") or 0
        model = str(c.get("model") or "?")
        agg["models"][model] = agg["models"].get(model, 0) + 1
    for agg in by_comp.values():
        agg["latency_ms_avg"] = round(agg["latency_ms_sum"] / agg["calls"]) if agg["calls"] else 0
    return by_comp


def budget(summary: dict) -> dict:
    used = float(summary.get("elapsed_charged_sec") or 0.0)
    total = float(summary.get("max_minutes") or 0.0) * 60.0
    return {"used_sec": used, "total_sec": total, "pct": round(100.0 * used / total, 1) if total else None}


def agent_staleness(summary: dict, now: float, stale_after_sec: float) -> list[dict]:
    """Each agent's quiet time at *now* (the clock, or the run's end once it has ended), stale past *stale_after_sec*."""
    out = []
    for agent, ts in (summary.get("agent_last_active") or {}).items():
        if isinstance(ts, (int, float)):
            age = max(0.0, now - ts)
            out.append({"agent": agent, "age_sec": round(age), "stale": age > stale_after_sec})
    return out


def throughput_series(journal: dict | None) -> list[dict]:
    """The journal's entries, oldest first; ``throughput`` is what the entry measured (``throughput_after``)."""
    if not journal:
        return []
    out = []
    for e in journal.get("entries") or []:
        out.append(
            {
                "ts": e.get("ts"),
                "phase": e.get("phase"),
                "kind": e.get("kind"),
                "change": e.get("change"),
                "variant_name": e.get("variant_name"),
                "outcome": e.get("outcome"),
                "throughput": e.get("throughput_after"),
                "gain_pct": e.get("gain_pct"),
                "task_id": e.get("task_id"),
                "tick": e.get("tick"),
            }
        )
    return out


LATENCY_KINDS = ("ttft", "tpot", "e2el")
LATENCY_STATS = ("mean", "median", "p99")


def _numbers(*values: Any) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _number(value: Any) -> float | None:
    return float(value) if _numbers(value) else None


def _latencies(source: dict, key) -> dict[str, dict[str, float | None]]:
    """Each latency kind's stats, read from *source* under ``key(kind, stat)``; a stat it lacks is None."""
    return {kind: {stat: _number(source.get(key(kind, stat))) for stat in LATENCY_STATS} for kind in LATENCY_KINDS}


def _has_latency(latency: dict) -> bool:
    return any(v is not None for stats in latency.values() for v in stats.values())


# Directories a grid variant's slot runs a discarded round in: its warm-up round and its multi-node client warm-up
# (orchestrator/actions/executors/grid/_grid_base.py WARMUP_ROUND_DIR and MN_WARMUP_DIR). Mirrored, not imported:
# the Inspector may not import the orchestrator (the import-linter layers and acyclic contracts), and _grid_base would
# add about half a second to its start; test_discarded_rounds_are_not_points reads the producer's constants.
DISCARDED_ROUND_DIRS = frozenset({"warmup_round", "mn_warmup"})


def benchmark_point(parts: tuple[str, ...], raw: Any, mtime: float | None) -> dict | None:
    """One measured benchmark run as a (tok/s, latencies) point, from its result file.

    *parts* is the file's path below ``runs/``: ``<kind>/<task_id>/.../benchmark_*/inferencex_result.json``.
    Discarded rounds (:data:`DISCARDED_ROUND_DIRS`) and runs with no throughput or no latency are not points,
    and the baseline is :func:`baseline_point`'s.
    """
    if (
        not isinstance(raw, dict)
        or len(parts) < 3
        or not DISCARDED_ROUND_DIRS.isdisjoint(parts)
        or parts[0] == _BASELINE
    ):
        return None
    tput = raw.get("output_throughput")
    latency = _latencies(raw, lambda kind, stat: f"{stat}_{kind}_ms")
    if not _numbers(tput) or not _has_latency(latency):
        return None
    variant = next((p.split("_", 1)[1] for p in parts[2:] if p[:1] == "v" and p[1:3].isdigit() and "_" in p), None)
    return {
        "ts": raw.get("benchmark_end_time_unix") or mtime,
        "kind": parts[0],
        "task_id": parts[1],
        "variant": variant,
        "output_throughput": float(tput),
        "latency": latency,
        "concurrency": raw.get("max_concurrency"),
    }


def _measured_throughput(measurement: dict) -> Any:
    """The whole-server output rate a baseline timeline record measured.

    Hyperloom 1.2.0 renamed the key from ``throughput_tok_s_per_gpu`` (which never was per GPU) to
    ``output_throughput``; the old key is still read, holding the same figure, so a session recorded before the
    rename keeps its baseline point.
    """
    return measurement.get("output_throughput", measurement.get("throughput_tok_s_per_gpu"))


def baseline_point(timeline: Any, parse_ts, read_raw) -> dict | None:
    """The baseline's (tok/s, latencies) point from its timeline record, once it is measured.

    The record only has mean latencies; *read_raw* (task id -> the baseline's raw result, or None) adds the rest.
    """
    if not isinstance(timeline, dict) or timeline.get("kind") != _BASELINE:
        return None
    for action in (timeline.get("ext") or {}).get("actions") or []:
        m = action.get("measurement") or {}
        tput, task_id = _measured_throughput(m), action.get("task_id")
        if not task_id or not _numbers(tput):
            continue
        raw = read_raw(task_id)
        if isinstance(raw, dict):
            latency = _latencies(raw, lambda kind, stat: f"{stat}_{kind}_ms")
        else:
            latency = _latencies(m, lambda kind, stat: f"{kind}_{stat}_ms")
        if _has_latency(latency):
            return {
                "ts": parse_ts(action.get("end_time")),
                "kind": _BASELINE,
                "task_id": task_id,
                "variant": None,
                "output_throughput": float(tput),
                "latency": latency,
                "concurrency": None,
            }
    return None


def accuracy_results(state: dict, parse_ts) -> dict:
    """The accuracy a run's state.json records, as fractions: the baseline's (``baseline_accuracy``, 0 until one is
    measured) and each kept variant's with the time it was measured (the ``explore_search.accepted`` rows; a variant
    that was never gated has none). Reverted variants' accuracy is not kept there."""
    base = state.get("baseline_accuracy")
    rows = (state.get("explore_search") or {}).get("accepted") or []
    variants = [
        {"variant": row.get("name"), "value": float(row["accuracy"]), "t": parse_ts(row.get("ts"))}
        for row in rows
        if isinstance(row, dict) and _numbers(row.get("accuracy"))
    ]
    return {"baseline": float(base) if _numbers(base) and base > 0 else None, "variants": variants}


def result_throughput(kind: str, result: Any) -> tuple[float, str] | None:
    """(tok/s, label) of the configuration a succeeded baseline/explore result left as the best."""
    if not isinstance(result, dict) or str(result.get("status") or "").lower() != "succeeded":
        return None
    tput = result.get("output_throughput")
    if not isinstance(tput, (int, float)) or tput <= 0:
        return None
    variant = result.get("best_variant")
    name = variant.get("name") if isinstance(variant, dict) else None
    return float(tput), f"{kind}: {name}" if name else kind


def best_throughput(current: dict, results: list[tuple[str, Any]], series: list[dict]) -> dict:
    """state.json's ``current_best`` when it has a throughput, else the best measured one.

    ``current_best.hot_tput`` is ``None`` while a later round is running (e.g. after a resume), but the
    succeeded baseline/explore results and the KEEP journal entries still hold what was measured.
    """
    if current.get("hot_tput") is not None:
        return {**current, "source": "state"}
    found: list[tuple[float, str, str]] = []
    for kind, result in results:
        hit = result_throughput(kind, result)
        if hit:
            found.append((hit[0], hit[1], "result"))
    for e in series:
        tput = e.get("throughput")
        if isinstance(tput, (int, float)) and str(e.get("outcome") or "").upper() == "KEEP":
            found.append((float(tput), str(e.get("change") or e.get("kind") or "journal"), "journal"))
    if not found:
        return {**current, "hot_tput": None, "source": None}
    tput, action, source = max(found)
    return {**current, "hot_tput": tput, "action": action, "source": source}


def health_badges(summary: dict, tasks: list[dict], memory: dict, agents: list[dict]) -> list[dict]:
    """Top-of-page warnings, most severe first."""
    badges = []
    crashed = [t for t in tasks if t.get("health") == _CRASHED]
    if crashed:
        kinds = sorted({t.get("label") or t.get("kind") for t in crashed})
        badges.append(
            {
                "level": _LEVEL_ERROR,
                "text": f"{len(crashed)} task(s) crashed but recorded as succeeded: "
                + ", ".join(str(k) for k in kinds),
            }
        )
    stalled = [t for t in tasks if t.get("health") == "stalled"]
    if stalled:
        badges.append({"level": _LEVEL_WARN, "text": f"{len(stalled)} running task(s) with no recent output"})
    used, limit = memory.get("used_bytes"), memory.get("limit_bytes")
    if used and limit and used / limit > 0.85:
        badges.append(
            {
                "level": _LEVEL_ERROR if used / limit > 0.95 else _LEVEL_WARN,
                "text": f"container memory at {100 * used / limit:.0f}% of limit (OOM risk)",
            }
        )
    if summary.get("crash_count"):
        badges.append({"level": _LEVEL_WARN, "text": f"crash_count={summary['crash_count']}"})
    if summary.get("degraded_mode"):
        badges.append({"level": _LEVEL_INFO, "text": "degraded mode (model warnings)"})
    stale = [a["agent"] for a in agents if a.get("stale")]
    if stale and not summary.get("stop_reason"):
        badges.append({"level": _LEVEL_WARN, "text": "agents idle: " + ", ".join(stale)})
    return badges
