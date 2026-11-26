# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A plain-text digest of one snapshot, for the narrator's LLM.

The input is the watcher's snapshot (already redacted, streams included); the output is redacted again and capped,
so the model sees what the page shows at the clock and nothing past it.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from hyperloom.inspector.redact import redact_text

MAX_CHARS = 24_000
_LINE = 240
_REPLAY = "replay"
_EVENTS, _FINISHED, _CALLS, _JOURNAL, _REVIEWS = 25, 12, 10, 10, 5
_CALL_KEYS = ("phase", "status", "latency_ms", "output_tokens", "error_type", "error_message")
_JOURNAL_KEYS = ("phase", "change", "outcome", "throughput", "gain_pct")
_GIST_KEYS = ("kind", "state", "domain", "verdict", "decision", "status", "outcome", "reason", "change", "summary")


def hms(t: float | None) -> str:
    if t is None:
        return "?"
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S")


def ago(t: float | None, clock: float) -> str:
    if t is None:
        return "?"
    sec = max(0, int(clock - t))
    return f"{sec}s ago" if sec < 90 else f"{sec // 60}m ago" if sec < 5400 else f"{sec / 3600:.1f}h ago"


def _clip(text: str, n: int = _LINE) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def _compact(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(_round(value), default=str, separators=(",", ":"))


def _odd(task: dict) -> bool:
    """Health worth mentioning: anything but ok or a restatement of the state."""
    return task.get("health") not in (None, "ok", task.get("state"))


def _fields(row: dict, keys: tuple[str, ...]) -> str:
    return " ".join(f"{k}={_compact(row[k])}" for k in keys if row.get(k) not in (None, "", [], {}))


def _run(view: dict) -> list[str]:
    run, s, clock = view.get("run") or {}, view.get("summary") or {}, view.get("clock") or {}
    mode = _REPLAY if clock.get("mode") == _REPLAY else "live"
    manifest = run.get("manifest") or {}
    lines = [
        f"Session {run.get('session_id')} ({mode}); clock {hms(clock.get('now'))} UTC"
        + (f", replay started {hms(clock.get('start'))}, ends {hms(clock.get('end'))}" if mode == _REPLAY else ""),
        f"Model {run.get('model')}; framework {s.get('framework')} {s.get('framework_version') or ''}; "
        f"GPU {s.get('gpu_type')} TP={s.get('tp')} EP={s.get('ep')} nodes={s.get('nodes')}",
        f"Workload ISL={s.get('isl')} OSL={s.get('osl')} conc={s.get('conc')} precision={s.get('precision')}",
    ]
    if manifest.get("objective"):
        lines.append("Objective: " + _clip(_compact(manifest["objective"])))
    return lines


def _health(view: dict) -> list[str]:
    lines = [f"Badge ({b.get('level')}): {b.get('text')}" for b in view.get("badges") or []]
    optimizer = (view.get("run") or {}).get("optimizer") or {}
    if optimizer.get("alive") is not None:
        lines.append(f"Optimizer process alive: {optimizer.get('alive')}")
    stale = [a["agent"] for a in view.get("agents") or [] if a.get("stale")]
    if stale:
        lines.append("Agents quiet for too long: " + ", ".join(stale))
    return lines


def _state(view: dict) -> list[str]:
    s, clock = view.get("summary") or {}, (view.get("clock") or {}).get("now") or 0.0
    budget, best = view.get("budget") or {}, s.get("current_best") or {}
    lines = [
        f"Phase {s.get('phase')} (tick {s.get('tick')})",
        f"Budget used {budget.get('pct')}% of {round((budget.get('total_sec') or 0) / 60)} min; "
        f"baseline {_compact(s.get('baseline_tput'))} tok/s; best {_clip(_compact(best), 160)}",
    ]
    phases = [r for r in s.get("phase_history") or [] if isinstance(r, dict)][-6:]
    if phases:
        lines.append(
            "Phase history: "
            + "; ".join(f"{r.get('from_phase')}→{r.get('to_phase')} {ago(_ts(r), clock)}" for r in phases)
        )
    return lines + _health(view)


def _ts(row: dict) -> float | None:
    for key in ("ts_unix", "ts", "at"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


def _task_times(task: dict) -> tuple[float | None, float | None]:
    times = [_ts(h) for h in task.get("history") or [] if isinstance(h, dict)]
    times = [t for t in times if t is not None]
    return (times[0] if times else None), (times[-1] if times else None)


def _tasks(view: dict) -> list[str]:
    tasks, clock = view.get("tasks") or [], (view.get("clock") or {}).get("now") or 0.0
    by_state: dict[str, int] = {}
    for t in tasks:
        key = f"{t.get('state')}/{t.get('health')}" if _odd(t) else str(t.get("state"))
        by_state[key] = by_state.get(key, 0) + 1
    lines = ["Tasks: " + (", ".join(f"{k} {n}" for k, n in sorted(by_state.items())) or "none")]
    for t in tasks:
        if t.get("state") != "running":
            continue
        current = (t.get("progress") or {}).get("current") or {}
        start, _ = _task_times(t)
        lines.append(
            f"Running: {t.get('label')} ({t.get('kind')}, node {t.get('node')}) since {ago(start, clock)}"
            + (f", step {current.get('label')}" if current.get("label") else "")
            + (f", health {t.get('health')}: {t.get('reason')}" if _odd(t) else "")
        )
    done = [t for t in tasks if t.get("state") not in ("running", "queued", "pending")]
    done.sort(key=lambda t: _task_times(t)[1] or 0)
    for t in done[-_FINISHED:]:
        reason = t.get("reason")
        lines.append(
            f"Finished {ago(_task_times(t)[1], clock)}: {t.get('label')} ({t.get('kind')}) {t.get('state')}"
            + (f", health {t.get('health')}" if _odd(t) else "")
            + (f": {_clip(_compact(reason), 120)}" if reason else "")
        )
    return lines


def _gist(payload: Any) -> str:
    if not isinstance(payload, dict):
        return _clip(_compact(payload), 120) if payload else ""
    result = payload.get("result")
    status = _fields(result, ("runner_status",)) if isinstance(result, dict) else ""
    return _clip(" ".join(p for p in (_fields(payload, _GIST_KEYS), status) if p), 160)


def _events(view: dict, labels: dict[str, str]) -> list[str]:
    events, clock = (view.get("streams") or {}).get("event") or [], (view.get("clock") or {}).get("now") or 0.0
    lines = [f"Bus events: {len(events)} so far; the last {min(len(events), _EVENTS)}:"] if events else []
    for e in events[-_EVENTS:]:
        payload = e.get("payload")
        task = labels.get(payload.get("task_id")) if isinstance(payload, dict) else None
        lines.append(
            f"  {ago(e.get('_t'), clock)} {e.get('from_agent')}→{e.get('to_agent')} {e.get('topic')}"
            + (f" [{task}]" if task else "")
            + (f" {_gist(payload)}" if payload else "")
        )
    return lines


def _llm(view: dict, labels: dict[str, str]) -> list[str]:
    roles, clock = view.get("llm") or {}, (view.get("clock") or {}).get("now") or 0.0
    lines = [
        f"LLM {role}: {r.get('calls')} calls, {r.get('errors')} errors, avg {r.get('latency_ms_avg')} ms"
        for role, r in roles.items()
        if isinstance(r, dict)
    ]
    for c in ((view.get("streams") or {}).get("llm_call") or [])[-_CALLS:]:
        task = labels.get(c.get("task_id") or "")
        head = f"  call {ago(c.get('_t'), clock)} {c.get('role') or c.get('component')}" + (
            f" [{task}]" if task else ""
        )
        lines.append(_clip(f"{head} {_fields(c, _CALL_KEYS)}"))
    return _collapse(lines)


def _collapse(lines: list[str]) -> list[str]:
    """Runs of identical lines (a specialist's turns, say) as one line with a count."""
    out: list[tuple[str, int]] = []
    for line in lines:
        if out and out[-1][0] == line:
            out[-1] = (line, out[-1][1] + 1)
        else:
            out.append((line, 1))
    return [line if n == 1 else f"{line} (×{n})" for line, n in out]


def _journal(view: dict, labels: dict[str, str]) -> list[str]:
    rows, clock = (view.get("streams") or {}).get("journal") or [], (view.get("clock") or {}).get("now") or 0.0
    lines = []
    for r in rows[-_JOURNAL:]:
        task = labels.get(r.get("task_id") or "")
        head = f"Journal {ago(r.get('_t'), clock)}:" + (f" [{task}]" if task else "")
        lines.append(f"{head} {_fields(r, _JOURNAL_KEYS)}")
    for r in ((view.get("streams") or {}).get("review") or [])[-_REVIEWS:]:
        body = _compact({k: v for k, v in r.items() if k != "_t"})
        lines.append(_clip(f"Critic review {ago(r.get('_t'), clock)}: {body}"))
    metrics = (view.get("streams") or {}).get("metrics") or []
    if metrics:
        last = {k: v for k, v in metrics[-1].items() if k != "_t"}
        lines.append(f"Resources {ago(metrics[-1].get('_t'), clock)}: {_clip(_compact(last), 300)}")
    return lines


def _earlier(entries: list[dict], clock: float) -> list[str]:
    lines = []
    for e in entries:
        head = f"{'Narration' if e.get('kind') == 'narration' else 'Q&A'} at {hms(e.get('clock'))}"
        head += f" ({ago(e.get('clock'), clock)})"
        if e.get("question"):
            head += f": Q: {_clip(e['question'], 300)}"
        lines += [head, _clip(e.get("text") or "", 1500)]
    return lines


def build(view: dict, earlier: list[dict] | None = None, max_chars: int = MAX_CHARS) -> str:
    """The digest of *view*; *earlier* are prior narrations and answers, oldest first, all at or before the clock."""
    labels = {t["task_id"]: str(t.get("label") or t.get("kind")) for t in view.get("tasks") or [] if t.get("task_id")}
    clock = (view.get("clock") or {}).get("now") or 0.0
    sections = [
        ("Run", _run(view)),
        ("State", _state(view)),
        ("Tasks", _tasks(view)),
        ("Recent bus traffic", _events(view, labels)),
        ("LLM use", _llm(view, labels)),
        ("Results", _journal(view, labels)),
        ("Earlier narration and questions", _earlier(earlier or [], clock)),
    ]
    text = "\n\n".join(f"## {title}\n" + "\n".join(lines) for title, lines in sections if lines)
    text = redact_text(text)
    if len(text) > max_chars:
        text = text[: max_chars - 40] + "\n[digest truncated]"
    return text
