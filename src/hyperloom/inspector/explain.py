# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One item's own records as plain text, for "Explain this" in the narrator.

The page sends a small reference (``{"kind": "task", "id": ...}`` and so on, see :func:`parse_ref`); the server
resolves it against its own data and never trusts text from the page. The section uses only records released by the
clock, so in replay an item from the future is refused like an unknown one (``LookupError``, a 404). The text is
redacted and capped, since it goes into a prompt next to the digest.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.inspector import derive
from hyperloom.inspector.collectors import parse_ts, tail_file
from hyperloom.inspector.digest import _clip, _compact, _fields, _task_times, ago, hms
from hyperloom.inspector.redact import redact, redact_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from hyperloom.inspector.watcher import Watcher

MAX_CHARS = 6_000
MAX_FIELD = 200
_TRUNCATED = "\n[section truncated]"
_NODE_TASKS = 15
_LOG_LINES = 40
_LOG_LINE = 300
_LOG_BYTES = 32 * 1024
_LOG_FILES = 500
_TEXT = 1_500
_NEARBY = 5
_LONG_LINE = 600  # cap of a line that lists names or carries an excerpt

# Item kinds, and record fields that several of the field lists below name.
_CALL = "call"
_JOURNAL = "journal"
_STATUS = "status"
_REASON = "reason"
_OUTPUT_THROUGHPUT = "output_throughput"
_GAIN_PCT = "gain_pct"

# The fields of each kind's reference, with their type; ``call`` needs ``call_id`` or ``task_id``.
_FIELDS: dict[str, dict[str, type]] = {
    "node": {"id": str},
    "task": {"id": str},
    "event": {"seq": int},
    _CALL: {"call_id": str, "task_id": str, "turn": int},
    _JOURNAL: {"task_id": str, "change": str, "ts": str},
    "variant": {"task_id": str, "label": str},
}
KINDS = tuple(_FIELDS)
_OPTIONAL = {(_CALL, "call_id"), (_CALL, "task_id"), (_CALL, "turn")}
# A journal entry written outside a task (the baseline, say) has no task id; the page sends "" for it.
_MAY_BE_EMPTY = {(_JOURNAL, "task_id")}

_RESULT_KEYS = (
    _STATUS,
    "runner_status",
    _REASON,
    "error",
    "error_class",
    _OUTPUT_THROUGHPUT,
    "base_tput",
    _GAIN_PCT,
    "best_gain_pct",
)
_JOURNAL_KEYS = ("ts", "phase", "kind", "change", "outcome", "throughput", _GAIN_PCT, "tick")
_UNIT_KEYS = (_STATUS, _OUTPUT_THROUGHPUT, "earlier", "stage", _REASON, "error")
_OUTCOME_KEYS = ("outcome", "stage", _REASON, "error_class", "metrics")
_CALL_KEYS = (
    "phase",
    _STATUS,
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "error_type",
    "error_message",
)


# ----- references --------------------------------------------------------------------------------------------


def parse_ref(obj: Any) -> dict:
    """The item reference *obj*, checked and reduced to its kind's fields; ``ValueError`` says what is wrong.

    Refused: a non-object, an unknown kind, a missing or wrong-typed field, or a string over 200 characters.
    """
    if not isinstance(obj, dict):
        raise ValueError("item must be a JSON object")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise ValueError(f"unknown item kind {_clip(str(kind), 40)!r}; expected one of {', '.join(KINDS)}")
    out: dict[str, Any] = {"kind": kind}
    for name, typ in _FIELDS[kind].items():
        value = obj.get(name)
        if value is None or (value == "" and (kind, name) not in _MAY_BE_EMPTY):
            if (kind, name) in _OPTIONAL:
                continue
            raise ValueError(f"item.{name} is required for a {kind}")
        if typ is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"item.{name} must be an integer")
        if typ is str and not isinstance(value, str):
            raise ValueError(f"item.{name} must be a string")
        if isinstance(value, str) and len(value) > MAX_FIELD:
            raise ValueError(f"item.{name} is over {MAX_FIELD} characters")
        out[name] = value
    if kind == _CALL and not (out.get("call_id") or out.get("task_id")):
        raise ValueError("item.call_id or item.task_id is required for a call")
    return out


def item_section(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, str]:
    """(label, text) for the item *ref* in *view* (the watcher's snapshot at *clock*).

    ``label`` is the item's display name; ``text`` is its records, redacted and capped at :data:`MAX_CHARS`.
    Raises ``ValueError`` for a bad reference and ``LookupError`` for an unknown or not yet released item.
    """
    ref = parse_ref(ref)
    label, lines = _BUILDERS[ref["kind"]](watcher, view, ref, clock)
    text = redact_text("\n".join(lines))
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - len(_TRUNCATED)] + _TRUNCATED
    return redact_text(label), text


# ----- shared lookups ----------------------------------------------------------------------------------------


def _stream(view: dict, name: str) -> list[dict]:
    return (view.get("streams") or {}).get(name) or []


def _find_task(view: dict, task_id: str) -> dict:
    task = next((t for t in view.get("tasks") or [] if t.get("task_id") == task_id), None)
    if task is None:
        raise LookupError(f"no task {task_id!r} at the clock")
    return task


def _result(watcher: Watcher, task_id: str, clock: float) -> Any:
    """The task's ``delegated_result`` once it has landed by *clock* (as ``Watcher.tasks_at`` decides), redacted."""
    with watcher.lock:
        res = watcher.results.get(task_id)
    if not res or (res["t"] is not None and res["t"] > clock):
        return None
    return redact(res["result"])


def _outcomes(result: Any) -> list[dict]:
    rows = result.get("per_variant_outcomes") if isinstance(result, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _grid(task: dict) -> list[dict]:
    grid = (task.get("params") or {}).get("grid")
    return [g for g in grid if isinstance(g, dict)] if isinstance(grid, list) else []


def _best_before(watcher: Watcher, view: dict, t: float | None, task_id: str | None) -> dict:
    """Best tok/s from results and journal entries strictly before *t* (``Watcher._best_at`` without state).

    The task's own result is left out: it lands just before the journal rows it writes and already holds its best
    variant. Its earlier journal rows count (an explore round's first KEEP raises the bar for the next variant).
    """
    if t is None:
        return {}
    kinds = {x.get("task_id"): x.get("kind") or "task" for x in view.get("tasks") or []}
    with watcher.lock:
        landed = [(i, r) for i, r in watcher.results.items() if i != task_id and r["t"] is not None and r["t"] < t]
    results = [(kinds.get(i, "task"), redact(r["result"])) for i, r in landed]
    series = [e for e in _stream(view, _JOURNAL) if (e.get("_t") or 0) < t]
    return derive.best_throughput({}, results, series)


def _best_line(view: dict, best: dict | None = None, threshold: Any = None) -> str:
    summary = view.get("summary") or {}
    best = (summary.get("current_best") or {}) if best is None else best
    base, top = summary.get("baseline_tput"), best.get("hot_tput")
    line = f"Baseline {_tput(base)}; best so far {_tput(top)}" + (f" ({best['action']})" if best.get("action") else "")
    return line + (f"; keep threshold {_compact(threshold)}%" if threshold is not None else "")


def _tput(value: Any) -> str:
    return f"{_compact(value)} tok/s" if isinstance(value, (int, float)) else "none yet"


def _task_line(task: dict, clock: float) -> str:
    start, end = _task_times(task)
    line = f"{task.get('label')} ({task.get('kind')}, {task.get('task_id')}) {task.get('state')}"
    line += f", health {task.get('health')}" if task.get("health") not in (None, task.get("state")) else ""
    line += f", started {ago(start, clock)}" + (f", last change {ago(end, clock)}" if end != start else "")
    return line + (f": {_clip(_compact(task['reason']), 160)}" if task.get("reason") else "")


def _cut(text: Any, n: int = _TEXT) -> str:
    """*text* with its line breaks, keeping the head and the tail when it is longer than *n*."""
    text = text if isinstance(text, str) else _compact(text)
    if len(text) <= n:
        return text
    head = n // 3
    return f"{text[:head]}\n[… {len(text) - n} characters cut …]\n{text[len(text) - (n - head) :]}"


# ----- node --------------------------------------------------------------------------------------------------


def _node_traffic(edges: list, node_id: str) -> list[str]:
    lines = []
    for name, mine, other in (("in", "target", "source"), ("out", "source", "target")):
        rows = [f"{e.get(other)} {e.get('label')} ×{e.get('count')}" for e in edges if e.get(mine) == node_id]
        lines.append(f"Traffic {name}: " + ("; ".join(rows) or "none"))
    return lines


def _node(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, list[str]]:
    graph = view.get("graph") or {}
    node = next((n for n in graph.get("nodes") or [] if n.get("id") == ref["id"]), None)
    if node is None:
        raise LookupError(f"no node {ref['id']!r} at the clock")
    node_id, label = node["id"], str(node.get("label") or node["id"])
    trees = node.get("trees") or {}
    lines = [
        f"Agent node {node_id} ({label}, group {node.get('group')}); health {node.get('health')}",
        "Its tasks by health: " + (_compact(node.get("tasks")) if node.get("tasks") else "none"),
        f"LLM calls {node.get('llm_calls')}, tokens {node.get('llm_tokens')}; artifacts {node.get('artifacts')}"
        + (f", trees {_compact(trees)}" if trees else ""),
    ]
    lines += _node_traffic(graph.get("edges") or [], node_id)
    tasks = [t for t in view.get("tasks") or [] if t.get("node") == node_id]
    tasks.sort(key=lambda t: _task_times(t)[0] or 0)
    if tasks:
        lines.append(f"Its tasks ({len(tasks)}; newest {min(len(tasks), _NODE_TASKS)} first):")
        lines += ["  " + _task_line(t, clock) for t in reversed(tasks[-_NODE_TASKS:])]
    return label, lines


# ----- task --------------------------------------------------------------------------------------------------


def _task(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, list[str]]:
    task = _find_task(view, ref["id"])
    result = _result(watcher, task["task_id"], clock)
    label = str(task.get("label") or task.get("kind") or task["task_id"])
    lines = [f"Task {label} (kind {task.get('kind')}, node {task.get('node')}, id {task['task_id']})"]
    lines.append(
        f"State {task.get('state')}, health {task.get('health')}"
        + (f": {_clip(_compact(task['reason']), 300)}" if task.get("reason") else "")
    )
    lines += _history(task, clock)
    progress = task.get("progress") or {}
    current = progress.get("current") or {}
    if current.get("label"):
        lines.append(
            f"Current step: {current.get('unit')} {current.get('label')} {current.get('status')}"
            + (f" ({current.get('index')}/{current.get('total')})" if current.get("total") else "")
            + f", {ago(current.get('ts'), clock)}"
        )
    lines += _params(task.get("params"))
    lines += _result_lines(result)
    units = progress.get("units") or []
    if units:
        lines.append(f"Variant rows ({len(units)}, {progress.get('failed') or 0} failed):")
        lines += [f"  {u.get('unit')} {u.get('label')}: {_fields(u, _UNIT_KEYS)}" for u in units[-30:]]
    lines += _log_tail(watcher, task, clock)
    return label, lines


def _history(task: dict, clock: float) -> list[str]:
    rows = [h for h in task.get("history") or [] if isinstance(h, dict) and h.get("to")]
    rows = [h for h in rows if (parse_ts(h.get("ts")) or 0) <= clock]
    if not rows:
        return []
    steps = [f"{h.get('from') or '?'}→{h.get('to')} {hms(parse_ts(h.get('ts')))}" for h in rows[-20:]]
    return ["State history (UTC): " + "; ".join(steps)]


def _params(params: Any) -> list[str]:
    """The task's params without bulky fields: the grid is listed by name, long values are clipped."""
    if not isinstance(params, dict) or not params:
        return []
    small = {k: v if isinstance(v, (int, float, bool)) else _clip(_compact(v), 200) for k, v in params.items()}
    small.pop("grid", None)
    lines = ["Params: " + _clip(_compact(small), 1000)] if small else []
    grid = [g for g in params.get("grid") or [] if isinstance(g, dict)]
    if grid:
        names = ", ".join(str(g.get("name")) for g in grid)
        lines.append(_clip(f"Grid ({len(grid)} candidates): {names}", _LONG_LINE))
    return lines


def _result_lines(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    lines = ["Result: " + (_fields(result, _RESULT_KEYS) or "(no key fields)")]
    best = result.get("best_variant")
    if isinstance(best, dict):
        lines.append(f"Best variant: {best.get('name')} " + _fields(best, (_GAIN_PCT, "tput", _OUTPUT_THROUGHPUT)))
    for key in ("winners", "losers"):
        rows = [r for r in result.get(key) or [] if isinstance(r, dict)]
        if rows:
            body = "; ".join(f"{r.get('name')} {_fields(r, (_GAIN_PCT, 'tput', _REASON))}".strip() for r in rows)
            lines.append(_clip(f"{key.capitalize()}: {body}", _LONG_LINE))
    return lines


def _log_tail(watcher: Watcher, task: dict, clock: float) -> list[str]:
    path = _main_log(watcher, task, clock)
    if path is None:
        return []
    lines = tail_file(path, _LOG_BYTES).splitlines()[-_LOG_LINES:]
    rel = path.relative_to(watcher.session_dir.resolve()).as_posix()
    if not lines:
        return [f"Log {rel} is empty"]
    return [f"Last {len(lines)} lines of {rel}:"] + ["  " + _clip(line, _LOG_LINE) for line in lines]


def _main_log(watcher: Watcher, task: dict, clock: float) -> Path | None:
    """``process.log`` for a specialist, else the newest ``*.log`` under ``runs/<kind>/<task_id>/``.

    In replay a log counts once the clock passes its mtime (or the replay is at its end), as artifacts do.
    """
    root = watcher.session_dir.resolve()
    base = (root / "runs" / str(task.get("kind")) / str(task.get("task_id"))).resolve()
    if root not in base.parents or not base.is_dir():
        return None
    found = [base / "process.log"] if task.get("kind") == "specialist" else []
    if not found:
        for i, path in enumerate(base.rglob("*.log")):
            if i >= _LOG_FILES:
                break
            found.append(path)
    clock_info = watcher.clock
    at_end = clock_info.end is not None and clock >= clock_info.end
    best: tuple[float, Path] | None = None
    for path in found:
        try:
            real, mtime = path.resolve(), path.stat().st_mtime
        except OSError:
            continue
        if root not in real.parents or real.name.startswith(".env") or not real.is_file():
            continue
        if clock_info.replay and not at_end and mtime > clock:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, real)
    return best[1] if best else None


# ----- event and call ----------------------------------------------------------------------------------------


def _event(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, list[str]]:
    event = next((e for e in _stream(view, "event") if e.get("seq") == ref["seq"]), None)
    if event is None:
        raise LookupError(f"no bus event #{ref['seq']} at the clock")
    topic, src, dst = event.get("topic"), event.get("from_agent"), event.get("to_agent")
    lines = [f"Bus event #{event.get('seq')}: {topic} from {src} to {dst} at {hms(event.get('_t'))} UTC"]
    lines[0] += f" ({ago(event.get('_t'), clock)})"
    lines.append(_fields(event, ("msg_id", "in_reply_to")))
    payload = event.get("payload")
    task_id = payload.get("task_id") if isinstance(payload, dict) else None
    if task_id:
        task = next((t for t in view.get("tasks") or [] if t.get("task_id") == task_id), None)
        lines.append(f"About task {task_id}" + (f": {_task_line(task, clock)}" if task else ""))
    lines.append("Payload: " + _clip(_compact(payload), _TEXT))
    if topic == "delegated_result" and isinstance(payload, dict):
        lines += _result_lines(payload.get("result"))
    return f"{topic} from {src} to {dst}", [x for x in lines if x]


def _call_matches(call: dict, ref: dict) -> bool:
    if ref.get("call_id"):
        return call.get("call_id") == ref["call_id"]
    if call.get("task_id") != ref.get("task_id"):
        return False
    return "turn" not in ref or str(call.get("turn")) == str(ref["turn"])


def _conversation_lines(watcher: Watcher, call: dict, task_id: Any, turn: Any) -> list[str]:
    conv = watcher.conversation(str(call.get("call_id") or ""), str(task_id or ""), "" if turn is None else str(turn))
    if conv is None:
        return ["No prompt or response was recorded for this call."]
    lines = []
    if conv.get("source"):
        lines.append(f"Recorded in {conv['source']}")
    for key in ("prompt", "response", "tool_results"):
        if conv.get(key):
            lines += [f"{key.replace('_', ' ').capitalize()}:", _cut(conv[key])]
    return lines


def _call(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, list[str]]:
    call = next((c for c in _stream(view, "llm_call") if _call_matches(c, ref)), None)
    if call is None:
        raise LookupError("no such LLM call at the clock")
    component = str(call.get("component") or call.get("role") or "LLM")
    task_id, turn = call.get("task_id"), call.get("turn")
    task = next((t for t in view.get("tasks") or [] if t.get("task_id") == task_id), None) if task_id else None
    who = f"{component} call" + (f" {call['call_id']}" if call.get("call_id") else "")
    if task_id:
        who += f" for task {task.get('label') if task else task_id}" + (f" turn {turn}" if turn is not None else "")
    lines = [
        f"LLM {who} at {hms(call.get('_t'))} UTC ({ago(call.get('_t'), clock)}), model {call.get('model')}",
        _fields(call, _CALL_KEYS),
    ]
    lines += _conversation_lines(watcher, call, task_id, turn)
    return who, [x for x in lines if x]


# ----- journal entry and variant -----------------------------------------------------------------------------


def _same_ts(a: Any, b: str) -> bool:
    if a == b:
        return True
    ta, tb = parse_ts(a), parse_ts(b)
    return ta is not None and tb is not None and abs(ta - tb) < 1e-3


def _recipe_change(row: dict) -> str:
    """The journal's ``change`` text for a grid entry or outcome variant (as ``summarize_change`` writes it)."""
    args = row.get("extra_args", row.get("extra_server_args"))
    args = " ".join(str(a) for a in args) if isinstance(args, list) else str(args or "")
    envs = row.get("extra_envs")
    if isinstance(envs, dict) and envs:
        env = " ".join(f"{k}={v}" for k, v in envs.items())
        return f"{args.strip()} | env: {env}" if args.strip() else f"env: {env}"
    return args.strip() or str(row.get("name") or "")


def _grid_for_change(task: dict | None, result: Any, change: str) -> dict | None:
    grid = _grid(task or {})
    hit = next((g for g in grid if change in (_recipe_change(g), str(g.get("name") or ""))), None)
    if hit is not None:
        return hit
    named = {str(g.get("name")): g for g in grid}
    for o in _outcomes(result):
        if isinstance(o.get("variant"), dict) and _recipe_change(o["variant"]) == change:
            return named.get(str(o.get("variant_name")))
    return None


def _find_journal(rows: list[dict], ref: dict) -> dict:
    entry = next(
        (
            r
            for r in rows
            if (r.get("task_id") or "") == ref["task_id"]
            and r.get("change") == ref["change"]
            and _same_ts(r.get("ts"), ref["ts"])
        ),
        None,
    )
    if entry is None:
        raise LookupError("no such journal entry at the clock")
    return entry


def _nearby_journal(rows: list[dict], entry: dict, task_id: Any) -> list[str]:
    same = [r for r in rows if r is not entry and (r.get("task_id") or "") == (task_id or "")]
    lines: list[str] = []
    for name, part in (
        ("before", [r for r in same if r["_t"] <= entry["_t"]][-_NEARBY:]),
        ("after", [r for r in same if r["_t"] > entry["_t"]][:_NEARBY]),
    ):
        if part:
            lines.append(f"Entries of the same task {name} it:")
            lines += ["  " + _fields(r, _JOURNAL_KEYS) for r in part]
    return lines


def _journal(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, list[str]]:
    rows = _stream(view, _JOURNAL)
    entry = _find_journal(rows, ref)
    task_id = entry.get("task_id")
    task = next((t for t in view.get("tasks") or [] if t.get("task_id") == task_id), None) if task_id else None
    params = (task or {}).get("params") or {}
    lines = [f"Journal entry: {_fields(entry, _JOURNAL_KEYS)}"]
    if task is not None:
        lines.append("Task: " + _task_line(task, clock))
    best = _best_before(watcher, view, entry.get("_t"), task_id)
    lines.append("Before it: " + _best_line(view, best, params.get("keep_threshold_pct")))
    lines += _nearby_journal(rows, entry, task_id)
    result = _result(watcher, task_id, clock) if task_id else None
    grid = _grid_for_change(task, result, str(entry.get("change") or ""))
    if grid is not None:
        lines += _recipe_lines(grid)
        lines += _outcome_lines(result, str(grid.get("name") or ""))
    return _clip(str(entry.get("change") or entry.get("kind") or "journal entry"), 80), lines


def _recipe_lines(grid: dict) -> list[str]:
    lines = [f"Candidate {grid.get('name')}: recipe {_recipe_change(grid) or '(no args or envs)'}"]
    lines.append(_fields(grid, ("provenance", "scope", "args_mode", "remove_args", "unset_envs", "note", _REASON)))
    for key in sorted(k for k in grid if k.endswith("_evidence") and grid[k]):
        lines.append(_clip(f"{key}: {_compact(grid[key])}", 400))
    return [x for x in lines if x]


def _outcome_lines(result: Any, name: str) -> list[str]:
    outcome = next((o for o in _outcomes(result) if o.get("variant_name") == name), None)
    if outcome is None:
        return []
    lines = [f"Outcome of {name}: " + _fields(outcome, _OUTCOME_KEYS)]
    if outcome.get("error_excerpt"):
        lines.append("Error excerpt: " + _clip(_compact(outcome["error_excerpt"]), _LONG_LINE))
    return lines


def _variant_progress(current: dict, units: list[dict], label: str, clock: float) -> list[str]:
    lines: list[str] = []
    if current.get("label") == label:
        lines.append(f"Now: {current.get('unit')} {current.get('status')}, {ago(current.get('ts'), clock)}")
    lines += [f"Progress row ({u.get('unit')}): {_fields(u, _UNIT_KEYS)}" for u in units]
    return lines


def _variant(watcher: Watcher, view: dict, ref: dict, clock: float) -> tuple[str, list[str]]:
    task = _find_task(view, ref["task_id"])
    label, result = ref["label"], _result(watcher, task["task_id"], clock)
    grid = next((g for g in _grid(task) if str(g.get("name")) == label), None)
    progress = task.get("progress") or {}
    units = [u for u in progress.get("units") or [] if u.get("label") == label]
    outcome = _outcome_lines(result, label)
    if grid is None and not units and not outcome:
        raise LookupError(f"no candidate {label!r} in task {ref['task_id']!r} at the clock")
    lines = [f"Candidate {label} of task {_task_line(task, clock)}"]
    lines += _recipe_lines(grid) if grid is not None else ["Not in the task's grid (added by the executor)."]
    lines += _variant_progress(progress.get("current") or {}, units, label, clock)
    lines += outcome or ["No outcome recorded yet."]
    params = task.get("params") or {}
    threshold = params.get("keep_threshold_pct")
    lines.append("Task settings: " + (_fields(params, ("base_tput", "keep_threshold_pct")) or "none recorded"))
    t = next((u.get("ts") for u in units if u.get("ts") is not None), None)
    best = _best_before(watcher, view, t, task["task_id"]) if t is not None else None
    lines.append(("Before it ran: " if t is not None else "At the clock: ") + _best_line(view, best, threshold))
    return label, lines


_BUILDERS: dict[str, Callable[[Watcher, dict, dict, float], tuple[str, list[str]]]] = {
    "node": _node,
    "task": _task,
    "event": _event,
    _CALL: _call,
    _JOURNAL: _journal,
    "variant": _variant,
}
