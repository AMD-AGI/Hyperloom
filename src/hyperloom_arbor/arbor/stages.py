"""A kernel plan's items, stage by stage: the agent declares, the harness gates.

A rewrite's plan (the dossier skill's ``kernel-plan.md``) is a chain of stages
per design, each with its own gate, because the campaign's A/B gate can judge
nothing before the last one. Until now a plan's stage records lived in the
agent's prose: C5 wrote 21 addenda, and its plan's own stage table stopped at
S1. Here the plan carries its items in a block the harness reads:

    ## Items

    ITEMS: [{"id": "s1", "stage": "S1", "gate": {...}, "stop": {...}, "stops_plan": true},
            {"id": "c.s2", "stage": "S2", "after": ["s1"], "gate": {...}}, ...]

and the agent declares a stage's result on one line:

    STAGE: {"item": "<direction>.<item>", "result": "pass|fail|stop",
            "measured": {...}, "evidence": {"artifacts": [...], "ledger": [...]}}

Recording takes two steps. ``capture`` keeps the declaration; it runs with the
map's lines (``arbor.directions.apply_output``, after the agent). ``settle``
judges it at the next ``arbor.directions.advance``, once the iteration has its
verdict in session.json:

- the item is open, and the items it comes after are done (a stop may come
  earlier: an item can be killed unbuilt);
- the iteration's intent fits the stage (LEARN for S1-S3, FIX or stricter for
  S4, WIN for S5; a stop, which kills the item, may come from any);
- every artifact exists and was written during the iteration, and every
  ledger id is in the session's ledger;
- the numbers clear the plan's gate: a pass they do not clear is recorded as a
  fail, with why; S4 needs the iteration to KEEP and the harness's check of
  its switches, served on, to pass (arbor/measure/switches.py); S5 closes only from a booked
  KEEP whose DONE line cites the item, and ``attempts`` REVERTs kill it;
- a stop from a ``stops_plan`` item kills every later item.

Every attempt lands in ``<session>/stages.json``; the item's verdict lands in
the lane's map (``arbor.directions``), which closes the lane when every item
has one. The 1 % gate never judges S1-S4: they run as LEARN (S4 as FIX), and
while the next item is S1-S3 the LEARN cap is waived (``waives_learn_cap``).

    python3 -m arbor.stages check <plan.md>
        validate a plan's ITEMS block, and print its items in an order they allow
    python3 -m arbor.stages seed --config <cfg> --direction m-x --plan <plan.md> [--session DIR]
        make a campaign one lane: its session gets the lane's map and a pinned copy of the plan
    python3 -m arbor.stages show --session DIR
        the lane's items, their attempts, and what waits to be judged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from arbor import directions
from arbor.directions import CLOSED, OPEN

FILE = "stages.json"
PLAN_COPY = "lane_plan.md"
VERSION = 1
RESULTS = ("pass", "fail", "stop")
STOP_STATUSES = ("killed", "below_floor")
# An artifact older than its iteration's start is stale. The slack covers the
# start's rounding (0.1 s in session.json) and the clocks of a container's
# node and the NFS server that stamps the file.
MTIME_SLACK_S = 2.0

ITEM_RE = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")
# Upper case only: "Items:" and "Stage:" are ordinary words in a plan and a DONE line.
_ITEMS_HEAD = re.compile(r"(?m)^[ \t>*_`#-]*ITEMS[*_`]*[ \t]*:[*_`]*")
_STAGE_HEAD = re.compile(r"(?m)^[ \t>*_`#-]*STAGE[*_`]*[ \t]*:[*_`]*")

# Intents from the most lenient to the strictest (arbor/measure/intents.py): a
# stage is declared by its own intent or a stricter one.
_RANK = {"LEARN": 0, "FIX-CORRECTNESS": 1, "FIX": 2, "WIN": 3}


@dataclass(frozen=True)
class Kind:
    """One stage kind, the complete rule for it, as arbor/measure/intents.py keeps intents.

    ``intent``  the most lenient intent that may declare it.
    ``keep``    the declaring iteration's verdict must be KEEP.
    ``booked``  closed by the harness from a booked KEEP whose DONE line cites the item.
    ``flags``   flags every pass sets true, whatever the plan's gate says.
    ``points``  a pass reports a table (``measured.points``) of at least this many points.
    ``upstream``  a pass names the commit it ports (``measured.upstream_commit``).
    ``learn``   a measuring stage: the LEARN cap is waived while it is the next item.
    ``switched``  the harness served the KEEP's switches on and every check passed
                (arbor/measure/switches.py; the row's ``switches``).
    """

    name: str
    title: str
    lane: str
    intent: str
    done_when: str
    keep: bool = False
    booked: bool = False
    flags: tuple[str, ...] = ()
    points: int = 0
    upstream: bool = False
    learn: bool = False
    switched: bool = False


REGISTRY: tuple[Kind, ...] = (
    Kind("S1", "Reference", "kernel-plan", "LEARN",
         "the region's time now, at the served shapes, within the plan's tolerance of the in-situ profile",
         learn=True),
    Kind("S2", "Correct", "kernel-plan", "LEARN",
         "the new kernel matches the replaced path over the served shapes and edges, "
         "with a negative control that fires", flags=("negative_control_fired",), learn=True),
    Kind("S3", "Fast", "kernel-plan", "LEARN",
         "graph-captured time at the served shapes takes the plan's share of the gap", learn=True),
    Kind("S4", "In tree", "kernel-plan", "FIX",
         "behind a switch that defaults off, declared with SWITCH: the iteration KEEPs, and the "
         "harness, serving it on, finds dispatch on every rank, zero JIT in serving and the "
         "accuracy gate passed", keep=True, switched=True),
    Kind("S5", "A/B", "kernel-plan", "WIN",
         "the campaign's own gate: a booked KEEP whose DONE line cites the item", booked=True),
    Kind("ARM", "Arm", "architectural", "LEARN",
         "one layout measured at every point of the plan's table", points=2, learn=True),
    Kind("FRONTIER", "Frontier", "architectural", "FIX",
         "the layout that dominates the table lands, and the iteration KEEPs", keep=True),
    Kind("PORT", "Port", "upstream", "FIX",
         "the upstream change is in the tree with its commit named, and the iteration KEEPs",
         keep=True, upstream=True),
    Kind("CORRECT", "Correct", "upstream", "LEARN",
         "the port matches the path it replaces, with a negative control that fires",
         flags=("negative_control_fired",), learn=True),
    Kind("AB", "A/B", "upstream", "WIN",
         "the campaign's own gate: a booked KEEP whose DONE line cites the item", booked=True),
)
KINDS: dict[str, Kind] = {k.name: k for k in REGISTRY}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ── the plan's ITEMS block ────────────────────────────────────────────────────

def _blocks(text: str) -> list[tuple[object, str | None]]:
    """Every ``ITEMS: [...]`` block in ``text``: (its JSON value, None) or (None, why it does not parse)."""
    out = []
    dec = json.JSONDecoder()
    for m in _ITEMS_HEAD.finditer(text or ""):
        rest = text[m.end():]
        start = directions._JSON_START.match(rest).end()
        if start >= len(rest) or rest[start] != "[":
            continue   # "ITEMS:" in prose
        try:
            out.append((dec.raw_decode(rest[start:])[0], None))
        except ValueError as exc:
            out.append((None, f"ITEMS: not valid JSON ({exc})"))
    return out


def _rule(raw, what: str, keys: tuple[str, ...], iid: str) -> tuple[dict | None, str | None]:
    """A gate or a stop rule, checked for its keys and types."""
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, f"{iid}: {what} is a JSON object"
    extra = sorted(set(raw) - set(keys))
    if extra:
        return None, f"{iid}: {what} has {', '.join(extra)}; it takes {', '.join(keys)}"
    for k in ("reference", "tolerance_pct", "min", "max", "below", "above"):
        if k in raw and not _num(raw[k]):
            return None, f"{iid}: {what}.{k} is a number"
    numeric = [k for k in ("reference", "min", "max", "below", "above") if k in raw]
    if numeric and not str(raw.get("measure") or "").strip():
        return None, f"{iid}: {what} names the measure its {numeric[0]} applies to"
    if "tolerance_pct" in raw and "reference" not in raw:
        return None, f"{iid}: {what}.tolerance_pct needs a reference"
    if "require" in raw and (not isinstance(raw["require"], list)
                             or not all(isinstance(x, str) and x.strip() for x in raw["require"])):
        return None, f"{iid}: {what}.require is a list of flag names"
    return dict(raw), None


def _item(raw) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not a JSON object"
    iid = str(raw.get("id", "")).strip().lower()
    if not ITEM_RE.match(iid):
        return None, f"id {raw.get('id')!r} is not an item ID (s1, c.s2, …)"
    stage = str(raw.get("stage", "")).strip().upper()
    if stage not in KINDS:
        return None, f"{iid}: stage {raw.get('stage')!r} is not one of {', '.join(KINDS)}"
    kind = KINDS[stage]
    after = raw.get("after") or []
    after = [str(a).strip().lower() for a in (after if isinstance(after, list) else [after])]
    if iid in after:
        return None, f"{iid}: comes after itself"
    gate, err = _rule(raw.get("gate"), "gate", ("measure", "reference", "tolerance_pct", "min", "max", "require"), iid)
    if err:
        return None, err
    if gate and kind.booked:
        return None, f"{iid}: {stage} is judged by the campaign's own gate; it carries none of its own"
    stop, err = _rule(raw.get("stop"), "stop", ("measure", "below", "above", "status"), iid)
    if err:
        return None, err
    if stop is not None:
        if stop.get("measure") and ("below" in stop) == ("above" in stop):
            return None, f"{iid}: stop takes one of below or above"
        stop["status"] = str(stop.get("status") or "killed").strip().lower()
        if stop["status"] not in STOP_STATUSES:
            return None, f"{iid}: stop.status is one of {', '.join(STOP_STATUSES)}"
    attempts = raw.get("attempts")
    if attempts is not None and (isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1):
        return None, f"{iid}: attempts is a whole number of at least 1"
    if not isinstance(raw.get("stops_plan", False), bool):
        return None, f"{iid}: stops_plan is true or false"
    return {"id": iid, "stage": stage, "title": str(raw.get("title") or "").strip()[:160], "after": after,
            "gate": gate or {}, "stop": stop, "stops_plan": bool(raw.get("stops_plan")),
            "attempts": attempts}, None


def validate_items(raw) -> tuple[list[dict], list[str]]:
    """A plan's items, or every reason they are refused (a block is taken whole or not at all)."""
    if not isinstance(raw, list) or not raw:
        return [], ["ITEMS: expected a non-empty list of items"]
    items, errs, seen = [], [], set()
    for i, r in enumerate(raw):
        it, err = _item(r)
        if err:
            errs.append(f"ITEMS[{i}]: {err}")
        elif it["id"] in seen:
            errs.append(f"ITEMS[{i}]: {it['id']} appears twice")
        else:
            seen.add(it["id"])
            items.append(it)
    if not errs:
        errs = [f"{it['id']} comes after {a}, which is not an item" for it in items for a in it["after"]
                if a not in seen]
    if not errs:
        cyc = directions._cycle([{"id": it["id"], "depends_on": it["after"]} for it in items])
        if cyc:
            errs = [f"an after cycle: {' -> '.join(cyc)}"]
    return ([], errs[:8]) if errs else (items, [])


def parse_items(text: str) -> tuple[list[dict], list[str]]:
    """The plan's one ITEMS block, validated: (items, []) or ([], why not)."""
    blocks = _blocks(text)
    if not blocks:
        return [], ["no ITEMS block (the plan's ## Items section)"]
    if len(blocks) > 1:
        return [], [f"{len(blocks)} ITEMS blocks; a plan carries one"]
    raw, err = blocks[0]
    return ([], [err]) if err else validate_items(raw)


# ── the agent's STAGE lines ───────────────────────────────────────────────────

def _declaration(raw) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not a JSON object"
    iid = str(raw.get("item", "")).strip().strip("`").lower()
    if not directions.ID_RE.match(iid):
        return None, f"item {raw.get('item')!r} is not a lane item ID (<direction>.<item>)"
    result = str(raw.get("result", "")).strip().lower()
    if result not in RESULTS:
        return None, f"{iid}: result {raw.get('result')!r} is not pass, fail or stop"
    measured = raw.get("measured") or {}
    if not isinstance(measured, dict):
        return None, f"{iid}: measured is a JSON object of the numbers and flags the gate reads"
    if len(json.dumps(measured, default=str)) > 4000:
        return None, f"{iid}: measured is over 4000 characters; put the table in an artifact"
    ev = raw.get("evidence") or {}
    if not isinstance(ev, dict):
        return None, f'{iid}: evidence is {{"artifacts": [...], "ledger": [...]}}'
    evidence = {}
    for k in ("artifacts", "ledger"):
        v = ev.get(k)
        if v is not None:
            v = [str(x).strip()[:300] for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
            if v:
                evidence[k] = v[:20]
    return {"item": iid, "result": result, "measured": measured, "evidence": evidence}, None


def parse_stage_lines(output: str) -> tuple[list[dict], list[str]]:
    """``STAGE: {...}`` lines, as ``arbor.directions.parse_agent_lines`` reads DIRECTION."""
    out, errs = [], []
    dec = json.JSONDecoder()
    text = output or ""
    for m in _STAGE_HEAD.finditer(text):
        rest = text[m.end():]
        start = directions._JSON_START.match(rest).end()
        if start >= len(rest) or rest[start] != "{":
            errs.append("STAGE: no JSON object after the colon")
            continue
        try:
            raw, _ = dec.raw_decode(rest[start:])
        except ValueError as exc:
            errs.append(f"STAGE: not valid JSON ({exc})")
            continue
        e, err = _declaration(raw)
        (out.append(e) if e else errs.append(f"STAGE: {err}"))
    return out, errs


# ── the file ──────────────────────────────────────────────────────────────────

def load(root) -> dict | None:
    p = Path(root) / FILE
    if not p.is_file():
        return None
    try:
        st = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return st if isinstance(st, dict) and isinstance(st.get("items"), dict) else None


def save(root, st: dict) -> None:
    p = Path(root) / FILE
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def plan_items(root, doc: dict) -> dict[str, dict]:
    """The pinned plan's items by lane ID, each with its place in the plan. ValueError when it cannot be trusted."""
    lane = doc.get("lane") or {}
    name = lane.get("plan") or PLAN_COPY
    path = Path(name) if os.path.isabs(name) else Path(root) / name
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"the lane's plan cannot be read: {exc}") from exc
    sha = hashlib.sha256(data).hexdigest()
    pinned = lane.get("plan_sha256") or ""
    if sha != pinned:
        raise ValueError(f"the lane's plan {path.name} changed since it was pinned (sha256 {sha[:12]}, "
                         f"pinned {pinned[:12]}): nothing is judged until the lane is seeded again")
    items, errs = parse_items(data.decode("utf-8", errors="replace"))
    if errs:
        raise ValueError(f"the lane's plan has no valid ITEMS: {'; '.join(errs[:3])}")
    pre = f"{lane.get('direction')}."
    return {pre + it["id"]: {**it, "after": [pre + a for a in it["after"]], "index": k}
            for k, it in enumerate(items)}


# ── capture: after the agent ──────────────────────────────────────────────────

def capture(root, output: str, iteration: int) -> list[dict]:
    """Keep this iteration's STAGE lines for ``settle``; return what happened, as map events. Never raises."""
    try:
        st = load(root)
        decls, errs = parse_stage_lines(output)
        if st is None:
            return ([{"iteration": iteration, "ok": False,
                      "error": "STAGE: this lane has no stage items (no stages.json); close items with DIRECTION"}]
                    if decls or errs else [])
        events = [{"iteration": iteration, "ok": False, "error": e} for e in errs]
        last: dict[str, dict] = {}   # one declaration per item per iteration: the last
        for d in decls:
            if d["item"] in st["items"]:
                last[d["item"]] = d
            else:
                events.append({"iteration": iteration, "ok": False, "id": d["item"],
                               "error": f"{d['item']} is not an item of this lane"})
        if last:
            now = time.time()
            st["pending"] = (st.get("pending") or []) + [{**d, "iteration": iteration, "ts": now}
                                                         for d in last.values()]
            save(root, st)
        events += [{"iteration": iteration, "ok": True, "id": d["item"],
                    "note": f"STAGE {d['result']}, judged once iteration {iteration} has its verdict"}
                   for d in last.values()]
        return events
    except Exception as exc:  # noqa: BLE001 - a stage record must not cost the iteration
        return [{"iteration": iteration, "ok": False, "error": f"STAGE lines not recorded: {exc}"}]


# ── settle: at the next advance ───────────────────────────────────────────────

def _intent(row: dict | None) -> str:
    if not row:
        return ""
    from arbor.measure.intents import resolve
    return resolve(row.get("intent") or row.get("claim_class")).name


def _measured_text(measured: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in list(measured.items())[:6] if _num(v) or isinstance(v, bool))


def judge(spec: dict, kind: Kind, measured: dict, row: dict | None, served: bool = True) -> tuple[str, list[str]]:
    """The harness's own reading of a stage that is not a stop: ("pass" | "fail", what failed)."""
    why = []
    gate = spec.get("gate") or {}
    flags = list(gate.get("require") or []) + [f for f in kind.flags if f not in (gate.get("require") or [])]
    why += [f"{f} is not true" for f in flags if measured.get(f) is not True]
    m = gate.get("measure")
    if m and any(k in gate for k in ("reference", "min", "max")):
        v = measured.get(m)
        if not _num(v):
            why.append(f"measured has no number for {m}")
        else:
            if "reference" in gate:
                ref, tol = gate["reference"], gate.get("tolerance_pct", 0)
                dev = abs(v - ref) / abs(ref) * 100 if ref else (0.0 if v == ref else float("inf"))
                if dev > tol:
                    why.append(f"{m} {v} is {dev:.1f} % from the reference {ref} (tolerance {tol} %)")
            if "min" in gate and v < gate["min"]:
                why.append(f"{m} {v} < {gate['min']}")
            if "max" in gate and v > gate["max"]:
                why.append(f"{m} {v} > {gate['max']}")
    if kind.points:
        pts = measured.get("points")
        if not isinstance(pts, list) or len(pts) < kind.points:
            why.append(f"measured.points is a table of at least {kind.points} points")
    if kind.upstream and not str(measured.get("upstream_commit") or "").strip():
        why.append("measured.upstream_commit names the commit ported")
    if kind.keep and (row or {}).get("decision") != "KEEP":
        why.append(f"{kind.name} needs the iteration to KEEP; it was {(row or {}).get('decision')}")
    if kind.switched and served and (row or {}).get("decision") == "KEEP":
        sw = (row or {}).get("switches") or {}
        if sw.get("verdict") != "pass":
            why.append(f"{kind.name} needs the harness to serve the switch on and pass: "
                       + ("no SWITCH: line was declared" if not sw
                          else f"its check was {sw.get('verdict')} ({'; '.join(sw.get('reasons') or [])})"))
    return ("fail" if why else "pass"), why


def _stop_reading(stop: dict | None, measured: dict) -> tuple[bool | None, str]:
    """(does the stop rule fire, the reading): None when the rule has no measure or none was reported."""
    m = (stop or {}).get("measure")
    if not m:
        return None, ""
    v = measured.get(m)
    if not _num(v):
        return None, f"the stop rule reads {m}, and measured has none"
    if "below" in stop:
        return v < stop["below"], f"{m} {v} {'<' if v < stop['below'] else '≥'} {stop['below']}"
    return v > stop["above"], f"{m} {v} {'>' if v > stop['above'] else '≤'} {stop['above']}"


def _booked(row: dict | None, iid: str) -> bool:
    return (bool(row) and row.get("decision") == "KEEP" and row.get("booked") is True
            and _intent(row) == "WIN" and bool(directions.cited(directions._row_text(row), [iid])[0]))


def _refusal(root, doc: dict, spec: dict, kind: Kind, entry: dict, row: dict | None) -> list[str]:
    """Why a declaration cannot be judged at all (it is not an attempt); [] when it can."""
    it, result = entry["iteration"], entry["result"]
    if row is None:
        return [f"iteration {it} has no verdict in session.json"]
    why = []
    status = {d["id"]: d.get("status") for d in doc.get("directions") or []}
    waits = [a for a in spec["after"] if status.get(a) != "done"]
    if waits and result != "stop":
        why.append(f"it comes after {', '.join(waits)}, not done")
    have = _intent(row)
    if result != "stop" and _RANK.get(have, 0) < _RANK[kind.intent]:   # a stop is a kill: any intent
        why.append(f"{kind.name} is declared by {kind.intent} or stricter; iteration {it} declared {have}")
    ev = entry.get("evidence") or {}
    if result in ("pass", "stop") and not kind.booked and not (ev.get("artifacts") or ev.get("ledger")):
        why.append("a pass or a stop cites its evidence (artifacts, ledger ids)")
    started = row.get("started")
    for a in ev.get("artifacts") or []:
        p = directions.artifact_path(root, a)
        if not p.exists():
            why.append(f"artifact {a} does not exist")
        elif isinstance(started, (int, float)) and p.stat().st_mtime < started - MTIME_SLACK_S:
            why.append(f"artifact {a} was written before iteration {it} started")
    if ev.get("ledger"):
        from arbor import ledger
        known = {f.id for f in ledger.load(ledger.ledger_path(root))}
        missing = [x for x in ev["ledger"] if x not in known]
        if missing:
            why.append("not in the session's ledger: " + ", ".join(missing))
    return why


def _reading(spec: dict, kind: Kind, entry: dict, row: dict | None, served: bool = True) -> tuple[str, list[str]]:
    """The harness's result for a declaration it can judge: pass, fail, stop or refused, and why."""
    declared, measured, iid = entry["result"], entry.get("measured") or {}, entry["item"]
    fires, reading = _stop_reading(spec.get("stop"), measured)
    if fires:
        return "stop", [f"the stop rule fires: {reading}"] + ([f"declared {declared}"] if declared != "stop" else [])
    if declared == "stop":
        if (spec.get("stop") or {}).get("measure"):
            return "refused", [f"the stop rule does not fire: {reading}" if fires is False else reading]
        return "stop", []
    if kind.booked:
        if _booked(row, iid):
            return "pass", []
        return "fail", [f"no booked KEEP citing {iid}: iteration {entry['iteration']} was "
                        f"{_intent(row)} {(row or {}).get('decision')}"]
    result, why = judge(spec, kind, measured, row, served)
    if declared == "fail":
        return "fail", why or ["declared fail"]
    if result == "fail":
        why.append("declared pass; the harness records a fail")
    return result, why


def _close(doc: dict, st: dict, iid: str, status: str, why: str, evidence: dict, iteration: int) -> None:
    directions.close_item(doc, iid, status, why, evidence, iteration)
    st["items"].setdefault(iid, {"attempts": []})["status"] = status


def _record(doc: dict, st: dict, items: dict, iid: str, attempt: dict) -> list[dict]:
    """Book one attempt and what it decides for the item (and, for a stop of the plan, the rest)."""
    spec = items[iid]
    rec = st["items"].setdefault(iid, {"stage": spec["stage"], "status": OPEN, "attempts": []})
    rec["attempts"].append(attempt)
    it, h, why = attempt["iteration"], attempt["harness"], attempt["why"]
    evidence = {"iterations": [it], **(attempt.get("evidence") or {})}
    ev = {"iteration": it, "id": iid}
    if h == "pass":
        text = f"{spec['stage']} passed at iteration {it}"
        if _measured_text(attempt["measured"]):
            text += f": {_measured_text(attempt['measured'])}"
        _close(doc, st, iid, "done", text, evidence, it)
        return [{**ev, "ok": True, "status": "done"}]
    if h == "fail":
        fails = [a["iteration"] for a in rec["attempts"] if a["harness"] == "fail"]
        limit = spec.get("attempts")
        out = [{**ev, "ok": attempt["declared"] != "pass",
                ("error" if attempt["declared"] == "pass" else "note"):
                    f"{iid}: a fail ({len(fails)}{f' of {limit}' if limit else ''}): {'; '.join(why)}"}]
        if limit and len(fails) >= limit:
            _close(doc, st, iid, "killed", f"{spec['stage']} failed {len(fails)} of {limit} attempts "
                                           f"(iterations {', '.join(map(str, fails))}): {'; '.join(why)}",
                   {"iterations": fails}, it)
            out.append({**ev, "ok": True, "status": "killed"})
        return out
    if h == "stop":
        status = (spec.get("stop") or {}).get("status", "killed")
        _close(doc, st, iid, status, f"{spec['stage']} stopped at iteration {it}"
                                     + (f": {'; '.join(why)}" if why else ""), evidence, it)
        out = [{**ev, "ok": True, "status": status}]
        if spec.get("stops_plan"):
            open_ = {d["id"] for d in doc.get("directions") or [] if d.get("status") == OPEN}
            for later, s in sorted(items.items(), key=lambda kv: kv[1]["index"]):
                if s["index"] > spec["index"] and later in open_:
                    _close(doc, st, later, "killed", f"the plan stopped at {iid} (iteration {it})",
                           {"cascade": iid, "iteration": it}, it)
                    st["items"][later].setdefault("attempts", [])
                    out.append({"iteration": it, "id": later, "ok": True, "status": "killed"})
        return out
    return [{**ev, "ok": False, "error": f"{iid}: STAGE {attempt['declared']} not judged ({h}): {'; '.join(why)}"}]


def _attempt(entry: dict, row: dict | None, harness: str, why: list[str]) -> dict:
    return {"iteration": entry["iteration"], "declared": entry["result"], "harness": harness,
            "intent": _intent(row), "verdict": (row or {}).get("verdict") or (row or {}).get("decision"),
            "measured": entry.get("measured") or {}, "evidence": entry.get("evidence") or {},
            "ts": time.time(), "why": why}


def _settle_one(root, doc: dict, st: dict, items: dict, entry: dict, row: dict | None,
                served: bool = True) -> list[dict]:
    iid = entry["item"]
    cur = next((d for d in doc.get("directions") or [] if d["id"] == iid), None)
    if iid not in items or cur is None:
        return [{"iteration": entry["iteration"], "id": iid, "ok": False,
                 "error": f"{iid} is not an item of this lane"}]
    if cur.get("status") in CLOSED:
        return [{"iteration": entry["iteration"], "id": iid, "ok": False,
                 "error": f"{iid} is already {cur['status']}: nothing reopens inside a run"}]
    spec, kind = items[iid], KINDS[items[iid]["stage"]]
    why = _refusal(root, doc, spec, kind, entry, row)
    if why:
        return _record(doc, st, items, iid, _attempt(entry, row, "refused", why))
    if row.get("decision") == "INVALID" or row.get("verdict") == "NOT_MEASURED":
        return _record(doc, st, items, iid, _attempt(entry, row, "not_measured",
                                                    [f"iteration {entry['iteration']} was not measured"]))
    harness, why = _reading(spec, kind, entry, row, served)
    return _record(doc, st, items, iid, _attempt(entry, row, harness, why))


def _settle_booked(doc: dict, st: dict, items: dict, lessons) -> list[dict]:
    """S5 and AB close from the campaign's own verdicts: WIN rows whose DONE line cites the item."""
    events = []
    start = doc.get("round_started") or 0
    status = {d["id"]: d.get("status") for d in doc.get("directions") or []}
    for iid, spec in sorted(items.items(), key=lambda kv: kv[1]["index"]):
        if not KINDS[spec["stage"]].booked or status.get(iid) != OPEN:
            continue
        if any(status.get(a) != "done" for a in spec["after"]):
            continue
        seen = {a["iteration"] for a in (st["items"].get(iid) or {}).get("attempts") or []}
        rows = sorted({x["iteration"]: x for x in lessons or [] if isinstance(x.get("iteration"), int)
                       and x["iteration"] >= start and x["iteration"] not in seen
                       and str(x.get("decision") or "") not in directions.ANNOTATIONS}.values(),
                      key=lambda x: x["iteration"])
        for row in rows:
            if _intent(row) != "WIN" or row.get("decision") == "INVALID":
                continue
            if not directions.cited(directions._row_text(row), [iid])[0]:
                continue
            entry = {"item": iid, "result": "pass", "iteration": row["iteration"], "measured": {}, "evidence": {}}
            attempt = _attempt(entry, row, "pass" if _booked(row, iid) else "fail",
                               [] if _booked(row, iid) else [f"iteration {row['iteration']}: {row.get('decision')}"
                                                             + (f" ({row['reason']})" if row.get("reason") else "")])
            attempt["declared"] = None   # read off the verdict, not declared
            events += _record(doc, st, items, iid, attempt)
            status = {d["id"]: d.get("status") for d in doc.get("directions") or []}
            if status.get(iid) != OPEN:
                break
    return events


def settle(root, doc: dict, state, served: bool = True) -> list[dict]:
    """Judge what earlier iterations declared, now that they have verdicts; return map events. Never raises.

    ``served``: the campaign serves a model. A document workload's S4 has no switch to serve on."""
    root = Path(root)
    try:
        st = load(root)
        if st is None:
            return []
        try:
            items = plan_items(root, doc)
        except ValueError as exc:
            return [{"iteration": state.iteration, "ok": False, "error": str(exc)}]
        events, waiting = [], []
        for entry in sorted(st.get("pending") or [], key=lambda e: e["iteration"]):
            row = directions.verdict_row(state.lessons, entry["iteration"])
            if row is None and entry["iteration"] > state.iteration:
                waiting.append(entry)
                continue
            events += _settle_one(root, doc, st, items, entry, row, served)
        st["pending"] = waiting
        events += _settle_booked(doc, st, items, state.lessons)
        save(root, st)
        return events
    except Exception as exc:  # noqa: BLE001 - judging a stage must not stop an iteration
        return [{"iteration": getattr(state, "iteration", 0), "ok": False, "error": f"stages not settled: {exc}"}]


def next_item(doc: dict | None) -> dict | None:
    """The first open item, in plan order, whose ``after`` items are done; else the first open one."""
    dirs = (doc or {}).get("directions") or []
    status = {d["id"]: d.get("status") for d in dirs}
    open_ = [d for d in dirs if d.get("status") == OPEN]
    ready = [d for d in open_ if all(status.get(a) == "done" for a in d.get("after") or [])]
    return (ready or open_ or [None])[0]


def waives_learn_cap(doc: dict | None) -> bool:
    """True while the lane's next item is a measuring stage (S1-S3 and their kin)."""
    d = next_item(doc)
    kind = KINDS.get(str((d or {}).get("stage") or ""))
    return bool(kind and kind.learn)


# ── the prompt ────────────────────────────────────────────────────────────────

STAGE_SYNTAX = ('STAGE: {"item": "<item ID>", "result": "pass|fail|stop", '
                '"measured": {"<the gate\'s measure>": 0.0, "<flag>": true}, '
                '"evidence": {"artifacts": ["iter_N/<file>"], "ledger": ["<ledger id>"]}}')


def gate_text(spec: dict) -> str:
    """An item's gate in words, as the program shows it."""
    kind = KINDS[spec["stage"]]
    g, bits = spec.get("gate") or {}, []
    m = g.get("measure")
    if "reference" in g:
        bits.append(f"`{m}` within {g.get('tolerance_pct', 0)} % of {g['reference']}")
    if "min" in g:
        bits.append(f"`{m}` ≥ {g['min']}")
    if "max" in g:
        bits.append(f"`{m}` ≤ {g['max']}")
    flags = list(g.get("require") or []) + [f for f in kind.flags if f not in (g.get("require") or [])]
    if flags:
        bits.append(", ".join(f"`{f}`" for f in flags) + " true")
    if kind.points:
        bits.append(f"`points`, a table of at least {kind.points}")
    if kind.upstream:
        bits.append("`upstream_commit` named")
    if kind.keep:
        bits.append(f"declared {kind.intent} or stricter, and the iteration KEEPs")
    elif kind.booked:
        bits.append("a booked KEEP: `INTENT: WIN`, and a DONE line that cites the item")
    else:
        bits.append(f"declared {kind.intent} or stricter")
    s = "; ".join(bits)
    stop = spec.get("stop")
    if stop:
        when = (f"`{stop['measure']}` {'<' if 'below' in stop else '>'} {stop.get('below', stop.get('above'))}"
                if stop.get("measure") else "declared")
        s += f". Stop when {when}: {stop['status'].replace('_', ' ')}"
        if spec.get("stops_plan"):
            s += ", and the plan stops"
    if spec.get("attempts"):
        s += f". {spec['attempts']} failed attempt(s) kill it"
    return s


def render_host_plan(doc: dict) -> str:
    """A component-mode campaign's share of its host's directions (``component.plan``)."""
    lane = doc.get("lane") or {}
    dirs = doc.get("directions") or []
    n_closed = sum(1 for d in dirs if d.get("status") in CLOSED)
    lines = [
        f"## The host's plan — {n_closed} of {len(dirs)} directions closed\n",
        f"Your host (`{lane.get('run', 'host')}`) gave this campaign these directions, and other campaigns "
        "work the rest of its map. Work these and nothing else: no dossier map of your own (a DIRECTIONS "
        "line is refused); the dossier serves these directions. Lead your DONE line with the direction's "
        "ID. Close one with `DIRECTION: {\"id\": …, \"status\": \"done|killed|below_floor\", \"why\": …, "
        "\"evidence\": {\"iterations\": [N], \"ledger\": […]}}`: the harness checks the evidence once that "
        "iteration has its verdict, and a done increment needs a KEEP whose DONE line cites it. Nothing "
        "reopens. A direction worth working that is not here: propose it with DIRECTION (`kind` and "
        "`title`); it goes to the host, not onto this map. When every direction is closed, the campaign "
        "stops.\n",
        "**Directions:**",
    ]
    for d in sorted(dirs, key=lambda d: (d.get("rank") is None, d.get("rank") or 0)):
        line = (f"- [{'x' if d.get('status') in CLOSED else ' '}] `{d['id']}` ({d.get('kind', 'increment')}) "
                f"**{str(d.get('status', OPEN)).upper()}** — {d.get('title', '')}")
        if d.get("ceiling"):
            line += f" (ceiling {d['ceiling']})"
        if d.get("status") in CLOSED and d.get("why"):
            line += f". Why: {d['why']}"
        elif d.get("why"):
            line += f". {d['why']}"
        lines.append(line)
    if doc.get("proposals"):
        lines.append("\nProposed to the host: " + ", ".join(f"`{p['id']}`" for p in doc["proposals"]))
    rejected = [e for e in (doc.get("events") or [])
                if not e.get("ok") and e.get("iteration") == doc.get("last_events")]
    if rejected:
        lines.append("\n**Rejected last iteration** (fix and emit again):\n"
                     + "\n".join(f"- {e.get('error')}" for e in rejected[-6:]))
    return "\n".join(lines) + "\n"


def render(doc: dict, state=None, campaign=None) -> str:
    """The lane's section of the program: its items, the ready ones with their gates, and STAGE."""
    lane = doc.get("lane") or {}
    if lane.get("kind") == "host-plan":
        return render_host_plan(doc)
    dirs = doc.get("directions") or []
    root = getattr(campaign, "session_root", None)
    n_closed = sum(1 for d in dirs if d.get("status") in CLOSED)
    try:
        items = plan_items(root, doc) if root else {}
        plan_err = ""
    except ValueError as exc:
        items, plan_err = {}, str(exc)
    st = (load(root) if root else None) or {"items": {}, "pending": []}
    status = {d["id"]: d.get("status") for d in dirs}
    where = (f"lane `{lane.get('lane')}` of run `{lane['run']}`, round {lane.get('round')}"
             if lane.get("run") else "a single-node campaign seeded with one plan")
    lines = [
        f"## The lane — `{lane.get('direction', '?')}`: {n_closed} of {len(dirs)} items closed\n",
        f"This campaign is {where}. It works one {lane.get('kind') or 'kernel-plan'} plan and nothing else: "
        f"`{lane.get('plan') or PLAN_COPY}` in the session (pinned, sha256 {str(lane.get('plan_sha256'))[:12]}; "
        f"from `{lane.get('source', '?')}`). Each item has its own gate, which the harness applies; the "
        "campaign's A/B gate judges only the A/B items. Lead your DONE line with the item ID you worked: the "
        "harness credits the item from it.\n",
    ]
    if plan_err:
        lines.append(f"**The plan cannot be read:** {plan_err}. Stop and say so in your DONE line.\n")
    ready = [d for d in dirs if d.get("status") == OPEN
             and all(status.get(a) == "done" for a in d.get("after") or [])]
    if ready:
        lines.append("**Ready now:**")
        for d in ready:
            spec = items.get(d["id"])
            lines.append(f"- `{d['id']}` {d.get('stage')} {KINDS[d['stage']].title if d.get('stage') in KINDS else ''}"
                         f" — {d.get('title', '')}" + (f". Gate: {gate_text(spec)}." if spec else ""))
    blocked = [d for d in dirs if d.get("status") == OPEN
               and any(status.get(a) in CLOSED and status.get(a) != "done" for a in d.get("after") or [])]
    if blocked:
        lines.append("\n**Blocked for good** (an item each comes after closed without passing): "
                     + ", ".join(f"`{d['id']}`" for d in blocked) + ". Nothing kills them for you: close each "
                     "with a STAGE `stop` and its evidence once you know why its design cannot go on.")
    lines += [
        "\nDeclare a stage's result on its own line, with the numbers its gate reads and the evidence:\n",
        f"    {STAGE_SYNTAX}\n",
        "The harness judges it at the start of the next iteration, after this one's verdict: the item must be "
        "ready (a `stop` may kill it unbuilt), the iteration's INTENT must fit the stage, every artifact must "
        "be written during this iteration (paths relative to the session, or absolute), every ledger id must "
        "be in the ledger, and the numbers must clear the gate. A pass they do not clear is booked as a fail, "
        "with why. A/B items close from a booked KEEP whose DONE line cites them; no STAGE line is needed. "
        "An item without a stage closes with `DIRECTION: {\"id\": …, \"status\": …, \"why\": …, "
        "\"evidence\": {\"iterations\": [N], …}}`. Nothing reopens. A direction this plan lacks: propose it "
        "with DIRECTION (`kind` and `title`); it goes to the next round's map. Never write `stages.json` or "
        "`directions.json`. When every item has its verdict, the lane closes and the campaign stops.\n",
        "**Items:**",
    ]
    for d in dirs:
        rec = st["items"].get(d["id"]) or {}
        tries = rec.get("attempts") or []
        bits = [d.get("stage") or "item"]
        if d.get("after"):
            bits.append(f"after {', '.join(d['after'])}")
        line = f"- [{'x' if d.get('status') in CLOSED else ' '}] `{d['id']}` ({'; '.join(bits)}) " \
               f"**{str(d.get('status', OPEN)).upper()}** — {d.get('title', '')}"
        if tries:
            last = tries[-1]
            line += (f" (attempts {len(tries)}; last: iteration {last['iteration']}, "
                     f"{last.get('declared') or 'verdict'} → {last['harness']})")
        if d.get("status") in CLOSED and d.get("why"):
            line += f". Why: {d['why']}"
        lines.append(line)
    waiting = st.get("pending") or []
    if waiting:
        lines.append("\nWaiting to be judged: " + ", ".join(f"`{p['item']}` {p['result']} (iteration "
                                                             f"{p['iteration']})" for p in waiting))
    if doc.get("proposals"):
        lines.append("\nProposed for the next round: " + ", ".join(f"`{p['id']}`" for p in doc["proposals"]))
    rejected = [e for e in (doc.get("events") or [])
                if not e.get("ok") and e.get("iteration") == doc.get("last_events")]
    if rejected:
        lines.append("\n**Rejected last iteration** (fix and emit again):\n"
                     + "\n".join(f"- {e.get('error')}" for e in rejected[-6:]))
    return "\n".join(lines)


# ── seed: one plan as one campaign ────────────────────────────────────────────

def seed(config, direction: str, plan, session=None, kind: str = "kernel-plan", run: str | None = None,
         round_no: int = 1, lane_id: str | None = None, force: bool = False) -> Path:
    """Make a campaign's session one lane: the lane's map, ``stages.json`` and a pinned copy of the plan."""
    cfg = json.loads(Path(config).read_text())
    root = Path(session or cfg.get("session_root") or "")
    if not str(session or cfg.get("session_root") or ""):
        raise ValueError("the config names no session_root: give --session")
    did = direction.strip().lower()
    if not directions.ID_RE.match(did):
        raise ValueError(f"{direction!r} is not a map ID (m-<component>)")
    if kind not in directions.LANES:
        raise ValueError(f"kind {kind!r} is not one of {', '.join(directions.LANES)}")
    data = Path(plan).read_bytes()
    items, errs = parse_items(data.decode("utf-8", errors="replace"))
    if errs:
        raise ValueError(f"{plan}: " + "; ".join(errs))
    root.mkdir(parents=True, exist_ok=True)
    if (root / directions.FILE).exists() and not force:
        raise FileExistsError(f"{root / directions.FILE} exists: this session has a map (--force replaces it)")
    for stale in (directions.LANE_FILE, FILE):
        (root / stale).unlink(missing_ok=True)
    (root / PLAN_COPY).write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    start = 1
    if (root / "session.json").is_file():
        from arbor.state import CampaignState
        start = CampaignState.load(root / "session.json").iteration + 1
    now = time.time()
    item_kind = "increment" if kind == "increment" else "rewrite"
    doc = directions.new_doc()
    doc.update(scope="lane", phase="active", map_round=round_no, round_started=start,
               round_started_ts=now, phase_started_ts=now,
               lane={"run": run, "round": round_no, "lane": lane_id or did, "direction": did, "kind": kind,
                     "plan": PLAN_COPY, "plan_sha256": sha, "source": str(plan)},
               directions=[{"id": f"{did}.{it['id']}", "kind": item_kind,
                            "title": it["title"] or f"{it['stage']} {KINDS[it['stage']].title}",
                            "stage": it["stage"], "after": [f"{did}.{a}" for a in it["after"]],
                            "status": OPEN, "why": "", "added_at": start, "closed_at": None,
                            "worked": [], "keeps": [], "flagged": False} for it in items])
    directions.save(root, doc)
    save(root, {"version": VERSION, "plan": {"path": PLAN_COPY, "sha256": sha, "source": str(plan)},
                "items": {f"{did}.{it['id']}": {"stage": it["stage"], "status": OPEN, "attempts": []}
                          for it in items},
                "pending": []})
    return root


# ── the command line ──────────────────────────────────────────────────────────

def _order(items: list[dict]) -> list[str]:
    return directions.topo_order([{"id": it["id"], "depends_on": it["after"]} for it in items])


def show(root) -> str:
    doc, err = directions.load(root)
    if err or not directions.is_lane(doc):
        return err or f"{root}: no lane here (no directions.json with scope lane)"
    st = load(root) or {"items": {}, "pending": []}
    lane = doc.get("lane") or {}
    out = [f"lane {lane.get('lane')} ({lane.get('kind')}): {lane.get('direction')}, plan {lane.get('source')}, "
           f"phase {doc.get('phase')}"]
    for d in doc.get("directions") or []:
        out.append(f"  {d['id']:<40} {str(d.get('stage') or ''):<9} {d.get('status', OPEN):<12} "
                   f"{directions.dur(d.get('wall_s'))}")
        for a in (st["items"].get(d["id"]) or {}).get("attempts") or []:
            out.append(f"      it {a['iteration']}: {a.get('declared') or 'verdict'} -> {a['harness']}"
                       + (f" ({'; '.join(a.get('why') or [])})" if a.get("why") else ""))
    for p in st.get("pending") or []:
        out.append(f"  waiting: {p['item']} {p['result']} (iteration {p['iteration']})")
    if (Path(root) / directions.LANE_FILE).is_file():
        out.append(f"  closed: {Path(root) / directions.LANE_FILE}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m arbor.stages",
                                 description="A kernel plan's items and their stage gates.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("check", help="validate a plan's ITEMS block")
    p.add_argument("plan")
    p = sub.add_parser("seed", help="make a campaign's session one lane, from a plan")
    p.add_argument("--config", required=True)
    p.add_argument("--direction", required=True, help="the map ID the plan belongs to (m-<component>)")
    p.add_argument("--plan", required=True)
    p.add_argument("--session", help="the session directory (default: the config's session_root)")
    p.add_argument("--kind", default="kernel-plan", choices=directions.LANES)
    p.add_argument("--run", help="the parallel run this lane belongs to")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--lane", help="the lane's ID in its run (default: the direction)")
    p.add_argument("--force", action="store_true", help="replace the session's map")
    p = sub.add_parser("show", help="a lane's items and attempts")
    p.add_argument("--session", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        items, errs = parse_items(Path(a.plan).read_text(errors="replace"))
        for e in errs:
            print(f"REJECTED {e}")
        for it in items:
            print(f"  {it['id']:<10} {it['stage']:<9} after {', '.join(it['after']) or '-':<12} "
                  f"{gate_text(it)}")
        if items:
            print(f"OK: {len(items)} items; order: {' -> '.join(_order(items))}")
        return 1 if errs else 0
    if a.cmd == "seed":
        try:
            root = seed(a.config, a.direction, a.plan, a.session, a.kind, a.run, a.round, a.lane, a.force)
        except (OSError, ValueError) as exc:
            print(f"seed: {exc}", file=sys.stderr)
            return 1
        print(show(root))
        return 0
    if a.cmd == "show":
        print(show(a.session))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
