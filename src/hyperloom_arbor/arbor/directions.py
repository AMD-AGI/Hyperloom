"""The campaign's map: its dossier's directions, which are explored, and map rounds.

A campaign works from its dossier's part 11, a ranked list of directions, each
with a map ID (``m-<component>``). Until now nothing tracked which of them had
been explored: the ID was free text in DONE lines and ledger ids, and once the
map ran out the campaign explored aimlessly until the drought notices fired.
Operator, 2026-10-20: "if all the directions were explored already, the
campaign should not explore aimlessly, but should run dossier and profile
again, then continue 'like new'", with a checklist on the dashboard.

How it works:

- The map lives in ``<session>/directions.json``. Only the harness writes it
  (an agent hand-writing harness state once crashed every wake of a mesh); an
  older driver, which drops unknown ``session.json`` keys, leaves it alone.
- The agent states the map with one ``DIRECTIONS: [...]`` line, and settles a
  direction with ``DIRECTION: {"id": ..., "status": ..., "why": ...}``. A
  direction is explored when the agent closes it: done, killed, below_floor, or
  plan (a rewrite whose kernel plan is written; rewrites are the operator's).
- The harness marks a direction worked by every iteration whose DONE line or
  findings cite its ID, and flags one worked often but never closed.
- When every direction of a round is closed, the next iterations are the
  dossier skill's update round (re-profile, lessons, re-ranked directions) as
  LEARNs with the LEARN cap waived. Its new map starts round K+1 in the same
  session, with the streak counters reset: the campaign continues like new.
- With a round-end A/B hook (arbor/rounds.py; operator, 2026-10-21) a closed map
  ends the round the way a reached target does: the map waits ``verifying``
  while the driver runs a same-node A/B of the campaign's base against its
  current tree, and the update round begins once the A/B confirms the gain. A
  target reached with directions still open ends the round there too
  (``after_round_end``); the update round may carry them onto the next map.
- A new map from an update round with no open increment does not stop the campaign
  (operator, 2026-10-22: b0 stopped there, and "it should have started like a new
  campaign"). A fresh start follows (``fresh``): a new dossier from the code read
  again whole and a new profile, the old map history. If that map has no open
  increment either, the harness takes the top planned rewrite as the next round's
  work (``rewrite``). It stops for the operator (``STOP``, ``MAP_STOP.md``) when no
  rewrite is left, or when two rounds in a row end without a KEEP.

``advance`` runs at the start of every iteration (``arbor.cli.iterate.cmd_begin``),
``apply_output`` after the agent (``arbor.policy.claude_loop.run_iteration``),
``render`` inside the program (``arbor.pack.render``).

Version 2 (runner decision 0003, the parallel mode) adds what a coordinator
needs to run each direction as a lane, all optional, so a v1 map and v1 lines
read as before: ``lane`` (increment, kernel-plan, architectural or upstream;
by default increment for an increment and kernel-plan for a rewrite),
``plan`` (the kernel plan, relative to the dossier), ``depends_on`` (map IDs
on the same list, no cycles; a closed dependency counts as met), ``nodes``,
``upstream`` ({repo, pr, commit, image}, which an upstream lane needs),
``reopens`` ({round, why}: a direction an earlier round closed, back on
evidence), and ``evidence`` on a close.

A lane's map (``"scope": "lane"``) is one direction's plan, its items
``<direction>.<item>``, written by ``arbor.stages seed`` or the coordinator:
it is on whatever the pack's skills, has no bootstrap or update round, takes
no DIRECTIONS list, reopens nothing, sends a direction the agent adds to
``proposals``, and closes an item only on evidence the harness has checked
(stage items through ``arbor.stages``). When every item is closed it writes
``LANE_CLOSED.json`` and a STOP the driver obeys.

    python3 -m arbor.directions check [file] [--dossier DIR]
        validate a DIRECTIONS line or list (stdin by default); with --dossier,
        also each lane's plan (it exists, its ITEMS parse) and the order the
        dependencies allow
    python3 -m arbor.directions split MAP --shares N [--out DIR]
        a host's several Arbors from one map: its open directions in N disjoint
        shares, each a component.plan (DIR/share-K.json, or JSON on stdout);
        directions joined by depends_on stay in one share. MAP is a session's
        directions.json, a JSON list or {"directions": [...]}, or a DIRECTIONS
        line or list (a dossier's part 11)
    python3 -m arbor.directions generate --profile PROFILE --out MAP [--min-share PCT]
        a map without a dossier pass: the host's root profile's op families, ranked
        by share, framed by the v2 pack's roadmap (arbor/mapgen.py); exit 1 when
        the profile cannot be used
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

FILE = "directions.json"
HARNESS_COPY = ".directions.harness.json"   # save()'s own copy: a FILE that differs was edited by hand
STOP_FILE = "MAP_STOP.md"
STOP_PREFIX = "map-stop:"   # never "chunk-timer": scripts/cluster/chunk_lib.sh deletes those
LANE_FILE = "LANE_CLOSED.json"
LANE_STOP_PREFIX = "lane-closed:"
VERSION = 2

KINDS = ("increment", "rewrite")
LANES = ("increment", "kernel-plan", "architectural", "upstream")
# The v2 fields, kept wherever a direction is copied.
LANE_KEYS = ("lane", "plan", "depends_on", "nodes", "upstream", "reopens")
OPEN = "open"
CLOSED = ("done", "killed", "below_floor", "plan")
STATUSES = (OPEN,) + CLOSED
ANNOTATIONS = {"BASELINE", "ESCALATE", "NOTE", "REBASELINE", "MAP_ROUND", "ROUND_END", "ROUND_AB"}

ID_RE = re.compile(r"^m-[a-z0-9]+(?:[-_.][a-z0-9]+)*$")
CITE_RE = re.compile(r"(?<![A-Za-z0-9_.-])m-[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.I)
_HEAD = re.compile(r"(?im)^[ \t>*_`#-]*(DIRECTIONS?)[*_`]*[ \t]*:[*_`]*")
_JSON_START = re.compile(r"[\s`]*(?:json\b)?[\s`]*", re.I)

# The knobs, with their defaults (overridable in the pack's or the config's
# `policy`, and persisted in session.json's policy by arbor/state.py).
KNOBS = {
    "map_rounds": 1,             # 0 turns map rounds off
    "map_worked_flag": 4,        # an open direction worked this often is flagged
    "map_update_max_iters": 8,   # an update round that yields no map by then stops
    "map_fresh_max_iters": 10,   # likewise a fresh start, which reads the code again whole
    "map_bootstrap_waiver": 2,   # LEARN-cap waiver for writing the first map
    "map_round_additions": 3,    # directions the agent may add inside a round
}


# ── parsing what the agent wrote ──────────────────────────────────────────────

@dataclass
class Parsed:
    closes: list[dict] = field(default_factory=list)   # DIRECTION entries, in order
    new_map: list[dict] | None = None                  # the last valid DIRECTIONS list
    source: str = ""
    errors: list[str] = field(default_factory=list)


def _status(raw) -> str | None:
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in STATUSES else None


def _entry(raw, listed: bool) -> tuple[dict | None, str | None]:
    """One direction from the agent, validated. ``listed``: an entry of a DIRECTIONS list."""
    if not isinstance(raw, dict):
        return None, "not a JSON object"
    did = str(raw.get("id", "")).strip().strip("`").lower()
    if not ID_RE.match(did):
        return None, f"id {raw.get('id')!r} is not a map ID (m-<component>)"
    e: dict = {"id": did}
    if listed or raw.get("kind") is not None:
        kind = str(raw.get("kind") or "").strip().lower()
        if kind not in KINDS:
            return None, f"{did}: kind {raw.get('kind')!r} is not increment or rewrite"
        e["kind"] = kind
    if listed or raw.get("title") is not None:
        title = str(raw.get("title") or "").strip()
        if listed and not title:
            return None, f"{did}: no title"
        e["title"] = title[:160]
    if raw.get("status") is not None or listed:
        s = _status(raw.get("status", OPEN))
        if s is None:
            return None, f"{did}: status {raw.get('status')!r} is not one of {', '.join(STATUSES)}"
        e["status"] = s
    elif not ("kind" in e and e.get("title")):
        return None, f"{did}: a DIRECTION line carries a status (to close or reopen) or a kind and a title (to add)"
    why = str(raw.get("why") or "").strip()
    if e.get("status") in CLOSED and not why:
        return None, f"{did}: a closed direction says why (the evidence: iteration, number, ledger id)"
    if e.get("status") == "plan" and e.get("kind") == "increment":
        return None, f"{did}: plan closes a rewrite, not an increment"
    if why:
        e["why"] = why[:400]
    if raw.get("rank") is not None:
        try:
            e["rank"] = int(raw["rank"])
        except (TypeError, ValueError):
            pass
    if raw.get("ceiling") is not None:
        e["ceiling"] = str(raw["ceiling"]).strip()[:60]
    err = _lane_fields(raw, e, did)
    return (None, err) if err else (e, None)


def _lane_fields(raw: dict, e: dict, did: str) -> str | None:
    """The v2 fields, validated into ``e`` as given (no defaults added); an error or None."""
    if raw.get("lane") is not None:
        lane = str(raw["lane"]).strip().lower()
        if lane not in LANES:
            return f"{did}: lane {raw['lane']!r} is not one of {', '.join(LANES)}"
        e["lane"] = lane
    if raw.get("plan") is not None and str(raw["plan"]).strip():
        e["plan"] = str(raw["plan"]).strip()[:300]
    if e.get("lane") == "kernel-plan" and not e.get("plan"):
        return f"{did}: a kernel-plan lane names its plan (a file in the dossier)"
    if raw.get("depends_on") is not None:
        deps = raw["depends_on"] if isinstance(raw["depends_on"], list) else [raw["depends_on"]]
        out = []
        for x in deps:
            x = str(x).strip().strip("`").lower()
            if not ID_RE.match(x):
                return f"{did}: depends_on {x!r} is not a map ID"
            if x == did:
                return f"{did}: depends on itself"
            if x not in out:
                out.append(x)
        e["depends_on"] = out
    if raw.get("nodes") is not None:
        n = raw["nodes"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            return f"{did}: nodes {n!r} is not a whole number of at least 1"
        e["nodes"] = n
    if raw.get("upstream") is not None or e.get("lane") == "upstream":
        up = raw.get("upstream")
        if not isinstance(up, dict) or not all(str(up.get(k) or "").strip() for k in ("repo", "pr", "commit")):
            return (f'{did}: an upstream lane names its upstream: {{"repo": …, "pr": …, "commit": …}}, '
                    f'and "image" when it needs one')
        e["upstream"] = {k: str(up[k]).strip()[:200] for k in ("repo", "pr", "commit", "image")
                         if str(up.get(k) or "").strip()}
    if raw.get("reopens") is not None:
        ro = raw["reopens"]
        rnd = ro.get("round") if isinstance(ro, dict) else None
        if (isinstance(rnd, bool) or not isinstance(rnd, int) or rnd < 1
                or not str(ro.get("why") or "").strip()):
            return f'{did}: reopens is {{"round": <the round that closed it>, "why": "<what changed since>"}}'
        e["reopens"] = {"round": rnd, "why": str(ro["why"]).strip()[:400]}
    if raw.get("evidence") is not None:
        ev, err = _evidence(raw["evidence"])
        if err:
            return f"{did}: {err}"
        e["evidence"] = ev
    return None


def _evidence(raw) -> tuple[dict | None, str | None]:
    """A close's evidence: the iterations, ledger ids and artifacts it cites."""
    if not isinstance(raw, dict):
        return None, 'evidence is {"iterations": [N, …], "ledger": ["<id>", …], "artifacts": ["<path>", …]}'
    ev: dict = {}
    its = raw.get("iterations", raw.get("iteration"))
    if its is not None:
        its = its if isinstance(its, list) else [its]
        if not its or not all(isinstance(i, int) and not isinstance(i, bool) for i in its):
            return None, "evidence.iterations are iteration numbers"
        ev["iterations"] = sorted(set(its))
    for k in ("ledger", "artifacts"):
        v = raw.get(k)
        if v is not None:
            v = [str(x).strip()[:300] for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
            if v:
                ev[k] = v[:20]
    if not ev:
        return None, "evidence cites nothing (iterations, ledger ids or artifacts)"
    return ev, None


def lane_of(d: dict) -> str:
    """A direction's lane: as given, else increment for an increment and kernel-plan for a rewrite."""
    return d.get("lane") or ("kernel-plan" if d.get("kind") == "rewrite" else "increment")


def _cycle(entries) -> list[str] | None:
    """The first dependency cycle a depth-first walk finds, as a path back to its start."""
    deps = {e["id"]: e.get("depends_on") or [] for e in entries}
    state: dict[str, int] = {}   # 1 on the walk's stack, 2 finished
    stack: list[str] = []

    def walk(n):
        state[n] = 1
        stack.append(n)
        for m in deps.get(n, []):
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state:
                found = walk(m)
                if found:
                    return found
        state[n] = 2
        stack.pop()
        return None

    for n in deps:
        if n not in state:
            found = walk(n)
            if found:
                return found
    return None


def _graph_errors(entries) -> list[str]:
    ids = {e["id"] for e in entries}
    errs = [f"DIRECTIONS: {e['id']} depends on {x}, which is not on the list"
            for e in entries for x in e.get("depends_on") or [] if x not in ids]
    if errs:
        return errs
    cyc = _cycle(entries)
    return [f"DIRECTIONS: a dependency cycle: {' -> '.join(cyc)}"] if cyc else []


def topo_order(dirs) -> list[str]:
    """The IDs in an order the dependencies allow, ties by rank then list order; [] on a cycle."""
    pos = {d["id"]: k for k, d in enumerate(dirs)}
    key = {d["id"]: (d["rank"] if isinstance(d.get("rank"), int) else 999, pos[d["id"]]) for d in dirs}
    need = {d["id"]: {x for x in d.get("depends_on") or [] if x in pos} for d in dirs}
    out: list[str] = []
    while need:
        ready = [i for i, n in need.items() if not n]
        if not ready:
            return []
        nxt = min(ready, key=key.get)
        out.append(nxt)
        del need[nxt]
        for n in need.values():
            n.discard(nxt)
    return out


def deps_met(doc: dict, d: dict) -> bool:
    """Every direction ``d`` depends on is closed (one no longer on the map counts as met)."""
    status = {x["id"]: x.get("status") for x in doc.get("directions") or []}
    return all(status.get(x, "done") in CLOSED for x in d.get("depends_on") or [])


def parse_agent_lines(output: str) -> Parsed:
    """``DIRECTIONS: [...]`` and ``DIRECTION: {...}`` lines, as ``ledger.parse_agent_blocks`` reads LEDGER.

    The JSON may run over several lines or sit in a fenced block. A list is
    taken whole or not at all, and the last valid list wins; every rejection is
    kept, with its reason, for the next prompt.
    """
    p = Parsed()
    dec = json.JSONDecoder()
    text = output or ""
    for m in _HEAD.finditer(text):
        word = m.group(1).upper()
        rest = text[m.end():]
        start = _JSON_START.match(rest).end()
        if start >= len(rest) or rest[start] not in "[{":
            p.errors.append(f"{word}: no JSON after the colon")
            continue
        try:
            raw, _ = dec.raw_decode(rest[start:])
        except ValueError as exc:
            p.errors.append(f"{word}: not valid JSON ({exc})")
            continue
        if word == "DIRECTION":
            e, err = _entry(raw, listed=False)
            (p.closes.append(e) if e else p.errors.append(f"DIRECTION: {err}"))
            continue
        items, source = raw, ""
        if isinstance(raw, dict):
            items, source = raw.get("directions"), str(raw.get("source") or "")[:200]
        if not isinstance(items, list) or not items:
            p.errors.append("DIRECTIONS: expected a non-empty list of directions")
            continue
        entries, errs, seen = [], [], set()
        for i, item in enumerate(items):
            e, err = _entry(item, listed=True)
            if err:
                errs.append(f"DIRECTIONS[{i}]: {err}")
            elif e["id"] in seen:
                errs.append(f"DIRECTIONS[{i}]: {e['id']} appears twice")
            else:
                seen.add(e["id"])
                entries.append(e)
        errs = errs or _graph_errors(entries)
        if errs:
            p.errors.extend(errs[:8])
            continue
        p.new_map, p.source = entries, source
    return p


def cited(text: str, ids) -> tuple[list[str], list[str]]:
    """The map IDs ``text`` cites: (on the map, not on it).

    A token counts for the longest map ID it equals or extends at a separator,
    so ``m-moe-experts-sweep`` works ``m-moe-experts`` and does not credit ``m-moe``.
    """
    hits, stray = [], []
    for tok in CITE_RE.findall(text or ""):
        tok = tok.lower().rstrip("-_.")
        best = max((i for i in ids if tok == i or tok[len(i):len(i) + 1] in ("-", "_", ".")
                    and tok.startswith(i)), key=len, default=None)
        bucket, value = (hits, best) if best else (stray, tok)
        if value not in bucket:
            bucket.append(value)
    return hits, stray


# ── the file ──────────────────────────────────────────────────────────────────

def map_path(root) -> Path:
    return Path(root) / FILE


def new_doc() -> dict:
    return {"version": VERSION, "map_round": 0, "phase": "bootstrap", "phase_iters": 0,
            "round_started": None, "added_this_round": 0, "pending": None,
            "directions": [], "rounds": [], "events": [], "unmapped_ids": [], "stop": None,
            # Wall clocks: the round's and the phase's start (epoch seconds), and the
            # iterations' own wall time in this round (each gate row's wall_s).
            "round_started_ts": None, "phase_started_ts": time.time(), "round_wall_s": 0.0,
            # "lane": one direction's plan in a parallel run (the module's docstring), with
            # its run, round and plan in "lane"; what its agent adds goes to "proposals",
            # and its DIRECTION closes wait in "pending_closes" for their iteration's verdict.
            "scope": "campaign", "lane": None, "proposals": [], "pending_closes": []}


def is_lane(doc: dict | None) -> bool:
    return bool(doc) and doc.get("scope") == "lane"


def dur(seconds) -> str:
    """3h05m, 42m, 50s; an empty string when unknown."""
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def load(root) -> tuple[dict | None, str | None]:
    """(the map, None); (None, None) when there is none; (None, why) when it cannot be read.

    Only the harness writes the map (``save``, which keeps its own copy beside it). A file
    that differs from that copy was edited by hand: Model M day-0 b0's agent wrote its own map
    into it at iteration 61 (2026-10-22) after its DIRECTIONS list was rejected, then worked
    from it while the harness waited for the list, and the fresh start ran out. The copy is
    put back, and the map says so (``hand_edit``) until the next map is accepted.
    """
    p = map_path(root)
    if not p.is_file():
        return None, None
    try:
        text = p.read_text()
        doc = json.loads(text)
    except (OSError, ValueError) as exc:
        return None, f"{p} unreadable: {exc}"
    if not isinstance(doc, dict) or not isinstance(doc.get("directions"), list):
        return None, f"{p} is not a map"
    copy = p.with_name(HARNESS_COPY)
    try:
        mine = copy.read_text() if copy.is_file() else None
        kept = json.loads(mine) if mine is not None and mine != text else None
    except (OSError, ValueError):
        kept = None
    if isinstance(kept, dict) and isinstance(kept.get("directions"), list):
        print(f"[arbor] map: {p} was edited outside the harness -- its own copy is put back", file=sys.stderr)
        kept["hand_edit"] = {"ts": time.time(), "after_iteration": kept.get("last_events")}
        save(root, kept)
        doc = kept
    base = new_doc()
    base.update(doc)
    return base, None


def save(root, doc: dict) -> None:
    p = map_path(root)
    text = json.dumps(doc, indent=1) + "\n"
    for target in (p.with_name(HARNESS_COPY), p):
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, target)


def knob(name: str, campaign=None, state=None) -> int:
    """A map knob: the live config's policy, then the pack's, then the session's, then the default."""
    for src in ((getattr(campaign, "config", None) or {}).get("policy"),
                getattr(getattr(campaign, "pack", None), "policy", None),
                getattr(state, "policy", None)):
        v = src.get(name) if isinstance(src, dict) else getattr(src, name, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return KNOBS[name]


def enabled(campaign, state=None) -> bool:
    """Map rounds run for a campaign that maps, outside document workloads: its agent has the dossier skill, or its
    pack's lifecycle names skills in the map's chapters (arbor/lifecycle.py)."""
    try:
        from arbor import lifecycle
        from arbor.pack.loader import agent_skills
        pack = campaign.pack
        plan = lifecycle.declared(campaign)
        if plan is not None:
            if not any(plan.runs(c) or plan.uses(c) for c in lifecycle.MAP_CHAPTERS):
                return False
        elif "dossier" not in (agent_skills(pack) or []):
            return False
        if (getattr(campaign.pack, "workload", None) or {}).get("kind") == "document":
            return False
        return knob("map_rounds", campaign, state) > 0
    except Exception:  # noqa: BLE001 - a broken pack is not this module's to report
        return False


def current(campaign, state=None) -> dict | None:
    """The session's map when map rounds are on and it reads, or a lane's map; None otherwise."""
    root = getattr(campaign, "session_root", None)
    if not root:
        return None
    doc, _ = load(root)
    if is_lane(doc):
        return doc   # seeded on purpose: on whatever the pack's skills and workload
    return doc if enabled(campaign, state) else None


# ── the state of the map ──────────────────────────────────────────────────────

def exhausted(doc: dict) -> bool:
    dirs = doc.get("directions") or []
    return bool(dirs) and all(d.get("status") in CLOSED for d in dirs)


def reconcile(doc: dict, lessons: list[dict], flag_at: int = KNOBS["map_worked_flag"]) -> dict:
    """Which iterations of this round worked each direction, its KEEPs and its wall time. Pure.

    An iteration that worked two directions counts its wall time to both.
    """
    dirs = doc.get("directions") or []
    by_id = {d["id"]: d for d in dirs}
    for d in dirs:
        d["worked"], d["keeps"], d["wall_s"] = [], [], 0.0
    stray: dict[str, int] = {}
    start = doc.get("round_started")
    round_wall = 0.0
    if start is not None and dirs:
        for lesson in lessons or []:
            it, dec = lesson.get("iteration"), str(lesson.get("decision") or "")
            if not isinstance(it, int) or it < start or dec in ANNOTATIONS:
                continue
            wall = lesson.get("wall_s") if isinstance(lesson.get("wall_s"), (int, float)) else 0.0
            round_wall += wall
            if dec == "INVALID":
                continue
            text = " ".join([str(lesson.get("change") or "")]
                            + [str(f) for f in (lesson.get("findings") or [])])
            hits, off = cited(text, list(by_id))
            for did in hits:
                d = by_id[did]
                if it not in d["worked"]:
                    d["worked"].append(it)
                    d["wall_s"] = round(d["wall_s"] + wall, 1)
                if dec == "KEEP" and it not in d["keeps"]:
                    d["keeps"].append(it)
            for tok in off:
                stray[tok] = stray.get(tok, 0) + 1
    for d in dirs:
        d["flagged"] = d.get("status") == OPEN and len(d["worked"]) >= flag_at
    doc["unmapped_ids"] = sorted(stray, key=lambda t: -stray[t])[:10]
    doc["round_wall_s"] = round(round_wall, 1)
    return doc


def _round_keeps(lessons, start) -> int:
    return sum(1 for lesson in lessons or []
               if lesson.get("decision") == "KEEP" and isinstance(lesson.get("iteration"), int)
               and start is not None and lesson["iteration"] >= start)


def _counts(dirs) -> dict:
    out: dict[str, int] = {}
    for d in dirs:
        out[d.get("status", OPEN)] = out.get(d.get("status", OPEN), 0) + 1
    return out


def waives_learn_cap(doc: dict | None, campaign=None, state=None) -> bool:
    """The update round, and the first map's writing, are LEARNs past the usual cap.

    In a lane: while its next item is a measuring stage (S1-S3; ``arbor.stages``).
    """
    if not doc:
        return False
    if is_lane(doc):
        from arbor import stages
        return stages.waives_learn_cap(doc)
    if doc.get("phase") in ("updating", "fresh"):
        return True
    return (doc.get("phase") == "bootstrap"
            and doc.get("phase_iters", 0) <= knob("map_bootstrap_waiver", campaign, state))


# ── the harness's two calls ───────────────────────────────────────────────────

def apply_output(root, output: str, iteration: int, campaign=None, state=None,
                 scope: dict | None = None) -> list[dict]:
    """Apply the agent's DIRECTIONS / DIRECTION lines to the map; return what happened.

    ``scope`` is the iteration's image and operating point (arbor/measure/scope.py):
    a kill holds there, so its ``why`` carries it.
    """
    doc, err = load(root)
    if doc is None:
        return [{"iteration": iteration, "ok": False, "error": err}] if err else []
    p = parse_agent_lines(output)
    from arbor.measure.scope import describe
    where = describe(scope)
    for e in p.closes:
        if e.get("status") == "killed" and where:
            e["why"] = f"{e.get('why', '')} [killed at {where}]".strip()
    events = [{"iteration": iteration, "ok": False, "error": e} for e in p.errors]
    if is_lane(doc):
        return _apply_lane(Path(root), doc, p, output, iteration, events)
    phase = doc["phase"]
    if p.new_map is not None:
        if phase in ("bootstrap", "updating", "unmapped", "fresh"):
            doc["pending"] = {"directions": p.new_map, "source": p.source,
                              "iteration": iteration, "from": phase}
            events.append({"iteration": iteration, "ok": True,
                           "note": f"a map of {len(p.new_map)} directions; it starts next iteration"})
        else:
            events.append({"iteration": iteration, "ok": False,
                           "error": "DIRECTIONS is taken only while the map is being written or updated; "
                                    "inside a round, close or add directions one at a time with DIRECTION:"})
    by_id = {d["id"]: d for d in doc["directions"]}
    cap = knob("map_round_additions", campaign, state)
    for e in p.closes:
        if phase != "active":
            events.append({"iteration": iteration, "ok": False, "id": e["id"],
                           "error": "DIRECTION lines apply inside a map round; this iteration writes the map"})
            continue
        cur = by_id.get(e["id"])
        if cur is None:
            if "kind" in e and e.get("title"):
                if doc.get("added_this_round", 0) >= cap:
                    events.append({"iteration": iteration, "ok": False, "id": e["id"],
                                   "error": f"at most {cap} directions are added inside a round"})
                    continue
                unknown = [x for x in e.get("depends_on") or [] if x not in by_id]
                if unknown:
                    events.append({"iteration": iteration, "ok": False, "id": e["id"],
                                   "error": f"{e['id']} depends on {', '.join(unknown)}, not on the map"})
                    continue
                cur = {"id": e["id"], "kind": e["kind"], "title": e["title"], "rank": e.get("rank"),
                       "ceiling": e.get("ceiling", ""), "status": e.get("status", OPEN),
                       "why": e.get("why", ""), "added_at": iteration,
                       "closed_at": iteration if e.get("status") in CLOSED else None,
                       "worked": [], "keeps": [], "flagged": False}
                cur.update({k: e[k] for k in LANE_KEYS + ("evidence",) if k in e})
                doc["directions"].append(cur)
                by_id[cur["id"]] = cur
                doc["added_this_round"] = doc.get("added_this_round", 0) + 1
                events.append({"iteration": iteration, "ok": True, "id": cur["id"], "status": cur["status"],
                               "note": "added to the map"})
            else:
                events.append({"iteration": iteration, "ok": False, "id": e["id"],
                               "error": f"{e['id']} is not on round {doc['map_round']}'s map "
                                        "(to add it, give kind and title)"})
            continue
        if "status" not in e:
            continue
        if e["status"] == "plan" and cur.get("kind") != "rewrite":
            events.append({"iteration": iteration, "ok": False, "id": e["id"],
                           "error": "plan closes a rewrite, not an increment"})
            continue
        if e["status"] == "plan" and cur["id"] == doc.get("rewrite"):
            events.append({"iteration": iteration, "ok": False, "id": e["id"],
                           "error": "this round carries out that plan: close it done, killed or below_floor"})
            continue
        cur["status"] = e["status"]
        cur["why"] = e.get("why", "") if e["status"] in CLOSED else cur.get("why", "")
        cur["closed_at"] = iteration if e["status"] in CLOSED else None
        if e["status"] in CLOSED and e.get("evidence"):
            cur["evidence"] = e["evidence"]
        events.append({"iteration": iteration, "ok": True, "id": e["id"], "status": e["status"]})
    # Saved every iteration, so "rejected last iteration" never shows an older one's.
    doc["events"] = ((doc.get("events") or []) + events)[-30:]
    doc["last_events"] = iteration
    save(root, doc)
    return events


def advance(campaign, state, root) -> str | None:
    """Move the map on before an iteration. "stop" when the campaign stops for the operator.

    Never raises: a map that cannot be read or moved is reported and skipped,
    and the campaign goes on as it did before map rounds.
    """
    try:
        doc, err = load(root)
        if is_lane(doc):
            return _advance_lane(campaign, state, Path(root), doc)
        if not enabled(campaign, state):
            return None
        if err:
            print(f"[arbor] map: {err} -- left as it is, no map this iteration", file=sys.stderr)
            return None
        doc = doc or new_doc()
        flag_at = knob("map_worked_flag", campaign, state)
        nxt = state.iteration + 1

        if doc["phase"] == "stopped":
            # The driver exits on STOP, so reaching here means the operator removed it.
            # A map stopped for having nothing open resumes with a fresh start (before
            # 2026-10-22 that stop came straight after the update round: b0's).
            # One stopped for writing no map in time resumes that phase, its count from
            # zero, whatever the file holds (b0's fresh start, 2026-10-22, after a hand edit).
            stop = doc.get("stop") or {}
            reason = str(stop.get("reason") or "")
            empty = stop.get("kind") == "empty" or "no open increment" in reason
            no_map = stop.get("kind") == "no_map" or "produced no valid map" in reason
            doc["stop"] = None
            has_open = any(d.get("status") == OPEN for d in doc["directions"])
            if no_map:
                doc["phase"] = stop.get("phase") or ("fresh" if "fresh start" in reason else "updating")
            else:
                doc["phase"] = "active" if has_open else ("fresh" if empty else "updating")
            doc["phase_iters"] = 0
            doc["phase_started_ts"] = time.time()
            if doc["phase"] == "fresh" and not doc.get("history"):
                doc["history"] = [dict(x) for x in doc["directions"]]
            state.apply_map_round(doc["map_round"], "resumed",
                                  f"the operator removed the map STOP; {doc['phase']}")

        pend = doc.get("pending")
        if pend:
            doc["pending"] = None
            dirs = []
            for e in pend["directions"]:
                d = {"id": e["id"], "kind": e["kind"], "title": e.get("title", ""), "rank": e.get("rank"),
                     "ceiling": e.get("ceiling", ""), "status": e.get("status", OPEN), "why": e.get("why", ""),
                     "added_at": pend.get("iteration"), "worked": [], "keeps": [], "flagged": False}
                d.update({k: e[k] for k in LANE_KEYS + ("evidence",) if k in e})
                d["closed_at"] = pend.get("iteration") if d["status"] in CLOSED else None
                dirs.append(d)
            doc["map_round"] += 1
            now = time.time()
            doc.update(directions=dirs, round_started=nxt, phase="active", phase_iters=0, added_this_round=0,
                       round_started_ts=now, phase_started_ts=now, round_wall_s=0.0)
            n_inc = sum(1 for d in dirs if d["kind"] == "increment" and d["status"] == OPEN)
            n_rw = sum(1 for d in dirs if d["kind"] == "rewrite" and d["status"] == OPEN)
            doc["map_from"], doc["rewrite"], doc["hand_edit"] = pend.get("from"), None, None
            then = ""
            if pend.get("from") in ("updating", "fresh") and n_inc == 0:
                then = _no_increment(campaign, state, root, doc, pend["from"], nxt)
                if then == "stop":
                    return "stop"
            if then != "fresh":
                state.apply_map_round(
                    doc["map_round"], "start",
                    f"map round {doc['map_round']}: {len(dirs)} directions ({n_inc} open increments, "
                    f"{n_rw} open rewrites) from the {pend.get('from')} map"
                    + (f" ({pend['source']})" if pend.get("source") else "")
                    + (f"; the round carries out the rewrite {doc['rewrite']}" if then == "rewrite" else "")
                    + "; streak counters reset")

        reconcile(doc, state.lessons, flag_at)

        if doc["phase"] == "verifying":
            # A round end's A/B (arbor/rounds.py) runs before any iteration; the driver
            # runs it while one is pending. Without one (it was confirmed, or the
            # operator cleared it), the update round begins.
            from arbor import rounds
            if rounds.pending(root):
                save(root, doc)
                return "round_end"
            doc.update(phase="updating", phase_iters=0, phase_started_ts=time.time())
            state.apply_map_round(doc["map_round"], "verified",
                                  "no round-end A/B pending; the dossier update round begins")

        if doc["phase"] == "active" and exhausted(doc):
            keeps, ran, stop = _close_round(doc, state)
            if stop:
                return _stop(campaign, state, root, doc, stop)
            from arbor import rounds
            if rounds.hook(campaign):
                # Unified with the target (operator, 2026-10-21): an explored map ends the
                # round like a reached target does, with a same-node A/B first.
                doc.update(phase="verifying", phase_iters=0, phase_started_ts=time.time())
                save(root, doc)
                state.apply_map_round(
                    doc["map_round"], "exhausted",
                    f"map round {doc['map_round']} closed every direction ({keeps} KEEPs over "
                    f"{ran} iterations); the round-end A/B runs, then the dossier update round")
                rounds.request(root, state, "map", f"map round {doc['map_round']} closed every direction "
                               f"({keeps} KEEPs over {ran} iterations)", map_round=doc["map_round"])
                return "round_end"
            doc.update(phase="updating", phase_iters=0, phase_started_ts=time.time())
            state.apply_map_round(
                doc["map_round"], "exhausted",
                f"map round {doc['map_round']} closed every direction ({keeps} KEEPs over "
                f"{ran} iterations); the dossier update round begins")

        if doc["phase"] in ("updating", "fresh"):
            doc["phase_iters"] += 1
            limit = knob("map_update_max_iters" if doc["phase"] == "updating" else "map_fresh_max_iters",
                         campaign, state)
            if doc["phase_iters"] > limit:
                what = "update round" if doc["phase"] == "updating" else "fresh start"
                doc["stop_phase"] = doc["phase"]
                return _stop(campaign, state, root, doc,
                             f"the {what} produced no valid map in {limit} iterations", kind="no_map")
        elif doc["phase"] in ("bootstrap", "unmapped"):
            doc["phase_iters"] += 1
            if (doc["phase"] == "bootstrap"
                    and doc["phase_iters"] > 2 * knob("map_bootstrap_waiver", campaign, state)):
                doc["phase"] = "unmapped"
        save(root, doc)
        return None
    except Exception as exc:  # noqa: BLE001 - the map must never stop an iteration by accident
        print(f"[arbor] map: could not advance ({exc}) -- skipped this iteration", file=sys.stderr)
        return None


def _close_round(doc: dict, state, ended_by: str = "map") -> tuple[int, int, str | None]:
    """Record the round that just ended: (its KEEPs, its iterations, a stop reason or None)."""
    start = doc.get("round_started")
    keeps = _round_keeps(state.lessons, start)
    ran = max(0, state.iteration - start + 1) if start is not None else 0
    began = doc.get("round_started_ts")
    doc["rounds"].append({"round": doc["map_round"], "started": start, "ended": state.iteration,
                          "iterations": ran, "keeps": keeps, "closed": _counts(doc["directions"]),
                          "ended_by": ended_by,
                          "wall_s": round(time.time() - began, 1) if began else None,
                          "iteration_wall_s": doc.get("round_wall_s"),
                          "directions": [{"id": x["id"], "status": x.get("status"),
                                          "wall_s": x.get("wall_s")} for x in doc["directions"]]})
    counted = [r for r in doc["rounds"] if r.get("iterations", 0) > 0][-2:]
    if len(counted) == 2 and all(r["keeps"] == 0 for r in counted):
        return keeps, ran, (f"map rounds {counted[0]['round']} and {counted[1]['round']} both ended "
                            f"without a KEEP")
    return keeps, ran, None


def after_round_end(campaign, state, root, trigger: str) -> str | None:
    """After a confirmed round-end A/B (arbor/rounds.py): the dossier update round begins.

    An explored map was left ``verifying``; a reached target ends the active round here,
    open directions and all, recorded as ended by the target. "stop" when that round
    was the second in a row without a KEEP. Never raises.
    """
    try:
        doc, err = load(root)
        if doc is None or err or is_lane(doc) or not enabled(campaign, state):
            return None
        if doc["phase"] == "verifying":
            doc.update(phase="updating", phase_iters=0, phase_started_ts=time.time())
            state.apply_map_round(doc["map_round"], "verified",
                                  f"the round-end A/B confirmed map round {doc['map_round']}'s gain; "
                                  "the dossier update round begins")
        elif doc["phase"] == "active" and trigger == "target":
            n_open = sum(1 for d in doc["directions"] if d.get("status") == OPEN)
            keeps, ran, stop = _close_round(doc, state, ended_by="target")
            if stop:
                return _stop(campaign, state, root, doc, stop)
            doc.update(phase="updating", phase_iters=0, phase_started_ts=time.time())
            state.apply_map_round(
                doc["map_round"], "target",
                f"map round {doc['map_round']} ended at the target ({keeps} KEEPs over {ran} iterations, "
                f"{n_open} direction(s) still open) and the A/B confirmed it; the dossier update round "
                "begins and may carry open directions onto the next map")
        else:
            return None
        save(root, doc)
        return None
    except Exception as exc:  # noqa: BLE001 - the map must never stop the campaign by accident
        print(f"[arbor] map: could not move on after the round end ({exc})", file=sys.stderr)
        return None


def _next_rewrite(doc: dict) -> dict | None:
    """The top rewrite not yet taken: one with a kernel plan first, then by rank.

    From the map, else from the history a fresh start kept (a fresh map may leave a
    planned rewrite out); one from the history is put back on the map.
    """
    taken = set(doc.get("rewrites_taken") or [])

    def best(dirs):
        c = [x for x in dirs if x.get("kind") == "rewrite" and x.get("status") in (OPEN, "plan")
             and x.get("id") not in taken]
        return min(c, key=lambda x: (x.get("status") != "plan",
                                     x["rank"] if isinstance(x.get("rank"), int) else 999), default=None)
    rw = best(doc.get("directions") or [])
    if rw is None:
        old = best(doc.get("history") or [])
        if old is not None:
            rw = {**old, "worked": [], "keeps": [], "wall_s": 0.0, "flagged": False, "added_at": None}
            doc["directions"].append(rw)
    return rw


def _no_increment(campaign, state, root, doc: dict, source: str, nxt: int) -> str:
    """A new map with no open increment: "fresh", "rewrite" or "stop".

    Operator, 2026-10-22: b0 stopped on such a map, and "it should have started like a
    new campaign"; then, if that finds no increment either, the top planned rewrite.
    From an update round: a fresh start, the map just written kept as history. From a
    fresh start: the top planned rewrite opened as the round's work. Stop when none is left.
    """
    k, now = doc["map_round"], time.time()
    if source == "updating":
        doc["history"] = [dict(x) for x in doc["directions"]]
        doc.update(phase="fresh", phase_iters=0, phase_started_ts=now)
        state.apply_map_round(k, "fresh", f"round {k}'s new map has no open increment: a fresh start, "
                              "the code read again whole and a new profile, the old map kept as history")
        return "fresh"
    rw = _next_rewrite(doc)
    if rw is None:
        return _stop(campaign, state, root, doc, f"round {k}'s map has no open increment after a fresh "
                     "start, and no planned rewrite is left", kind="empty")
    rw["taken"] = {"iteration": nxt, "why_before": rw.get("why", "")}
    rw.update(status=OPEN, closed_at=None, why="")
    doc["rewrite"] = rw["id"]
    doc["rewrites_taken"] = (doc.get("rewrites_taken") or []) + [rw["id"]]
    doc.update(phase="active", phase_iters=0, round_started=nxt, round_started_ts=now, phase_started_ts=now)
    state.apply_map_round(k, "rewrite", f"round {k}'s map has no open increment after a fresh start: "
                          f"the round carries out the rewrite {rw['id']} ({rw.get('title', '')})")
    return "rewrite"


def _stop(campaign, state, root, doc: dict, reason: str, kind: str | None = None) -> str:
    root = Path(root)
    doc.update(phase="stopped", stop={"reason": reason, "iteration": state.iteration, "ts": time.time(),
                                      "kind": kind, "phase": doc.pop("stop_phase", None)})
    save(root, doc)
    rounds = "\n".join(f"| {r['round']} | {r.get('started')}–{r.get('ended')} | {r.get('iterations')} | "
                       f"{dur(r.get('wall_s')) or '–'} | {r.get('keeps')} | "
                       f"{', '.join(f'{v} {k}' for k, v in sorted(r.get('closed', {}).items()))} |"
                       for r in doc["rounds"]) or "| – | | | | | |"
    rewrites = "\n".join(f"- `{d['id']}` ({d.get('status')}): {d.get('title', '')}"
                         + (f" — {d['why']}" if d.get("why") else "")
                         for d in doc["directions"] if d.get("kind") == "rewrite") or "- none"
    (root / STOP_FILE).write_text(
        f"# The map stopped the campaign\n\n**Why:** {reason}.\n\n"
        f"Stopped before iteration {state.iteration + 1}, in map round {doc['map_round']}.\n\n"
        f"## Rounds\n\n| Round | Iterations | Count | Wall time | KEEPs | Directions closed |\n"
        f"|---|---|---|---|---|---|\n{rounds}\n\n"
        f"## Rewrites on the map\n\n{rewrites}\n\n"
        f"## To resume\n\nDecide what the campaign does next (a rewrite campaign, a new pack, or more of "
        f"this map), then remove `STOP` and, in a rolling chain, `chain/CHAIN_STOP`, and launch it again. "
        f"The first iteration after that resumes the map: an open direction continues the round, a map "
        f"stopped for having nothing open starts afresh, and otherwise a dossier update round writes a "
        f"new map.\n")
    (root / "STOP").write_text(f"{STOP_PREFIX} {reason}\n")
    state.apply_map_round(doc["map_round"], "stop", reason)
    print(f"[arbor] map: STOP -- {reason} (see {root / STOP_FILE})", file=sys.stderr)
    return "stop"


# ── a lane's map ──────────────────────────────────────────────────────────────

def verdict_row(lessons, iteration: int) -> dict | None:
    """The gate's row for ``iteration`` (the last one, annotations aside), or None."""
    rows = [x for x in lessons or [] if x.get("iteration") == iteration
            and str(x.get("decision") or "") not in ANNOTATIONS]
    return rows[-1] if rows else None


def _row_text(lesson: dict) -> str:
    return " ".join([str(lesson.get("change") or "")] + [str(f) for f in (lesson.get("findings") or [])])


def artifact_path(root, a: str) -> Path:
    """An artifact the agent cites: absolute, or relative to the session."""
    return Path(a) if os.path.isabs(a) else Path(root) / a


def check_evidence(root, lessons, item_id: str, ev: dict, upto: int, need_keep: bool = False) -> list[str]:
    """What is wrong with a close's evidence; [] when it holds.

    Cited iterations are at most ``upto`` and have a verdict; ledger ids are in
    the session's ledger; artifacts exist. ``need_keep``: one cited iteration is
    a KEEP whose DONE line or findings cite the item.
    """
    why = []
    rows = {it: verdict_row(lessons, it) for it in ev.get("iterations") or []}
    for it, row in rows.items():
        if it > upto:
            why.append(f"iteration {it} is after this one ({upto})")
        elif row is None:
            why.append(f"iteration {it} has no verdict in session.json")
    if ev.get("ledger"):
        from arbor import ledger
        known = {f.id for f in ledger.load(ledger.ledger_path(root))}
        missing = [x for x in ev["ledger"] if x not in known]
        if missing:
            why.append("not in the session's ledger: " + ", ".join(missing))
    for a in ev.get("artifacts") or []:
        if not artifact_path(root, a).exists():
            why.append(f"artifact {a} does not exist")
    if need_keep and not any(row and row.get("decision") == "KEEP" and cited(_row_text(row), [item_id])[0]
                             for row in rows.values()):
        why.append(f"done needs a KEEP among the cited iterations whose DONE line cites {item_id}")
    return why


def close_item(doc: dict, did: str, status: str, why: str, evidence: dict, iteration: int) -> None:
    for d in doc.get("directions") or []:
        if d["id"] == did:
            d.update(status=status, why=why[:400], evidence=evidence, closed_at=iteration)


def _apply_lane(root: Path, doc: dict, p: Parsed, output: str, iteration: int, events: list) -> list[dict]:
    """A lane's agent lines: closes wait for their iteration's verdict, additions are proposals."""
    if p.new_map is not None:
        events.append({"iteration": iteration, "ok": False,
                       "error": "DIRECTIONS: a lane's map is the coordinator's; close its items (STAGE:, "
                                "DIRECTION:) and propose new directions with DIRECTION (kind and title)"})
    by_id = {d["id"]: d for d in doc["directions"]}
    for e in p.closes:
        cur = by_id.get(e["id"])
        if cur is None:
            if "kind" in e and e.get("title"):
                props = [x for x in doc.get("proposals") or [] if x.get("id") != e["id"]]
                props.append({**{k: e[k] for k in ("id", "kind", "title", "rank", "ceiling", "why")
                                 + LANE_KEYS if k in e}, "iteration": iteration})
                doc["proposals"] = props[-20:]
                events.append({"iteration": iteration, "ok": True, "id": e["id"],
                               "note": "proposed for the next round's map"})
            else:
                events.append({"iteration": iteration, "ok": False, "id": e["id"],
                               "error": f"{e['id']} is not an item of this lane (to propose a direction "
                                        "for the next round, give kind and title)"})
            continue
        if "status" not in e:
            continue
        if e["status"] == OPEN:
            if cur.get("status") in CLOSED:
                events.append({"iteration": iteration, "ok": False, "id": e["id"],
                               "error": f"nothing reopens inside a run: {e['id']} is {cur['status']}; "
                                        "propose it for the next round instead"})
            continue
        err = ("already closed" if cur.get("status") in CLOSED
               else "a stage item closes with STAGE:, which the harness gates" if cur.get("stage")
               else 'a lane closes on evidence: add "evidence": {"iterations": [N], "ledger": [...], '
                    '"artifacts": [...]}' if not e.get("evidence") else None)
        if err:
            events.append({"iteration": iteration, "ok": False, "id": e["id"], "error": f"{e['id']}: {err}"})
            continue
        doc["pending_closes"] = (doc.get("pending_closes") or []) + [
            {"id": e["id"], "status": e["status"], "why": e.get("why", ""), "evidence": e["evidence"],
             "iteration": iteration}]
        events.append({"iteration": iteration, "ok": True, "id": e["id"],
                       "note": f"{e['status']}, checked once iteration {iteration} has its verdict"})
    from arbor import stages
    events += stages.capture(root, output, iteration)
    doc["events"] = ((doc.get("events") or []) + events)[-30:]
    doc["last_events"] = iteration
    save(root, doc)
    return events


def _settle_closes(root: Path, doc: dict, state) -> list[dict]:
    """The lane's waiting DIRECTION closes, checked against session.json and the ledger."""
    events, waiting = [], []
    by_id = {d["id"]: d for d in doc["directions"]}
    for c in doc.get("pending_closes") or []:
        it, cur = c["iteration"], by_id.get(c["id"])
        if verdict_row(state.lessons, it) is None and it > state.iteration:
            waiting.append(c)
            continue
        why = (["already closed"] if cur is None or cur.get("status") in CLOSED else
               check_evidence(root, state.lessons, c["id"], c["evidence"], it,
                              need_keep=c["status"] == "done" and cur.get("kind") == "increment"))
        if why:
            events.append({"iteration": it, "ok": False, "id": c["id"],
                           "error": f"{c['id']} not closed: " + "; ".join(why)})
            continue
        close_item(doc, c["id"], c["status"], c["why"], c["evidence"], it)
        events.append({"iteration": it, "ok": True, "id": c["id"], "status": c["status"]})
    doc["pending_closes"] = waiting
    return events


def _advance_lane(campaign, state, root: Path, doc: dict) -> str | None:
    """A lane before an iteration: settle what the last one declared; close the lane when all is settled."""
    if doc.get("phase") == "closed":
        # Nothing reopens inside a run: a closed lane stays closed, whoever removed STOP.
        if not (root / "STOP").exists():
            (root / "STOP").write_text(f"{LANE_STOP_PREFIX} {(doc.get('stop') or {}).get('reason', 'closed')}\n")
        return "stop"
    from arbor import stages
    served = (getattr(campaign.pack, "workload", None) or {}).get("kind") != "document"
    events = _settle_closes(root, doc, state) + stages.settle(root, doc, state, served)
    if events:
        doc["events"] = ((doc.get("events") or []) + events)[-30:]
        for ev in events:
            what = ev.get("note") or ev.get("status") or ev.get("error")
            print(f"[arbor] lane: {'OK' if ev.get('ok') else 'REJECTED'} {ev.get('id', '')} {what}",
                  file=sys.stderr)
    reconcile(doc, state.lessons, knob("map_worked_flag", campaign, state))
    if exhausted(doc):
        return _lane_close(state, root, doc)
    save(root, doc)
    return None


def _lane_close(state, root: Path, doc: dict) -> str:
    """Every item has its verdict: LANE_CLOSED.json, then a STOP the driver obeys."""
    items = doc["directions"]
    counts = _counts(items)
    lane = doc.get("lane") or {}
    began = doc.get("round_started_ts")
    now = time.time()
    record = {"lane": lane, "iteration": state.iteration, "ts": now,
              "started": doc.get("round_started"), "wall_s": round(now - began, 1) if began else None,
              "iteration_wall_s": doc.get("round_wall_s"), "counts": counts,
              "keeps": sorted({k for d in items for k in d.get("keeps") or []}),
              "items": [{k: d.get(k) for k in ("id", "stage", "status", "why", "evidence", "closed_at",
                                                "worked", "keeps", "wall_s")} for d in items],
              "proposals": doc.get("proposals") or []}
    (root / LANE_FILE).write_text(json.dumps(record, indent=1) + "\n")
    reason = (f"every item of {lane.get('direction') or 'the lane'} has its verdict ("
              + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(counts.items())) + ")")
    doc.update(phase="closed", stop={"reason": reason, "iteration": state.iteration, "ts": now})
    save(root, doc)
    (root / "STOP").write_text(f"{LANE_STOP_PREFIX} {reason}\n")
    state.apply_map_round(doc.get("map_round", 1), "lane-closed", reason)
    print(f"[arbor] lane: STOP -- {reason} (see {root / LANE_FILE})", file=sys.stderr)
    return "stop"


# ── the prompt ────────────────────────────────────────────────────────────────

_CLOSE_SYNTAX = ('DIRECTION: {"id": "m-…", "status": "done|killed|below_floor|plan", '
                 '"why": "<the evidence: iteration, number, ledger id>"}')
_LIST_SYNTAX = ('DIRECTIONS: [{"id": "m-…", "kind": "increment|rewrite", "title": "…", "rank": 1, '
                '"ceiling": "<largest gain, target unit>", "status": "open"}, …]')


def _line(d: dict) -> str:
    bits = [d.get("kind", "?")]
    if d.get("rank") is not None:
        bits.append(f"#{d['rank']}")
    if d.get("ceiling"):
        bits.append(f"ceiling {d['ceiling']}")
    if d.get("depends_on"):
        bits.append(f"after {', '.join(d['depends_on'])}")
    work = []
    if d.get("worked"):
        work.append("worked " + ", ".join(map(str, d["worked"][-8:])))
    if d.get("keeps"):
        work.append("KEEP " + ", ".join(map(str, d["keeps"])))
    if dur(d.get("wall_s")):
        work.append(dur(d["wall_s"]))
    s = f"- `{d['id']}` [{' '.join(bits)}] **{d.get('status', OPEN).upper()}** — {d.get('title', '')}"
    if work:
        s += f" ({' · '.join(work)})"
    if d.get("status") in CLOSED and d.get("why"):
        s += f". Why: {d['why']}"
    return s


def render(doc: dict, state=None, campaign=None) -> str:
    """The map's section of the program, for the phase it is in."""
    if is_lane(doc):
        from arbor import stages
        return stages.render(doc, state, campaign)
    phase = doc.get("phase")
    k = doc.get("map_round", 0)
    # While the map is being written, a rejected list stays shown until a map is accepted:
    # b0's agent saw its rejection once, then worked from a map the harness never took.
    last = doc.get("last_events")
    writing = phase in ("bootstrap", "unmapped", "updating", "fresh")
    rejected = [e for e in (doc.get("events") or []) if not e.get("ok") and (
        e.get("iteration") == last
        or (writing and "id" not in e and (e.get("iteration") or 0) >= (doc.get("round_started") or 0)))]
    rej = (("\n\n**Rejected last iteration** (fix and emit again):\n"
            if all(e.get("iteration") == last for e in rejected) else
            "\n\n**Rejected, and no map accepted since** (fix and emit the list again):\n")
           + "\n".join(f"- {e.get('error')}" for e in rejected[-6:])) if rejected else ""
    if doc.get("hand_edit"):
        rej = ("\n\n**`directions.json` was edited by hand, and the edit is undone.** Only the harness "
               "writes the map: it takes it from a `DIRECTIONS:` line in your output (and, inside a "
               "round, from `DIRECTION:` lines). Nothing written into the file counts.") + rej
    root = getattr(campaign, "session_root", None) or "<session>"

    if phase in ("bootstrap", "unmapped"):
        waiver = waives_learn_cap(doc, campaign, state)
        head = ("## DIRECTIONS MISSING — write the campaign's map" if phase == "bootstrap"
                else "## The campaign has no map yet (reminder)")
        return (
            f"{head}\n\n"
            "Map rounds track which of the dossier's directions this campaign has explored, and "
            "when every one is closed the harness starts a dossier update round instead of letting "
            "the campaign explore aimlessly. That needs the map in a form the harness reads: read "
            "the dossier's newest part 11 (`11*.md`; the dossier skill's find-dossier.sh finds it) "
            "and emit it as ONE line:\n\n"
            f"    {_LIST_SYNTAX}\n\n"
            "Give every direction a map ID (`m-<component>`; reuse the IDs the dossier and the "
            "ledger already use) and its kind (increment or rewrite, ranked apart as part 11 ranks "
            "them). Carry over what is settled: a direction already finished, killed or below the "
            "floor gets that status and a `why`; a rewrite whose kernel plan exists is `plan`. "
            "Check the list first: `python3 -m arbor.directions check list.json`. Never write "
            "`directions.json` yourself.\n\n"
            + ("Declare `INTENT: LEARN` with a LEDGER FACT naming the part-11 file, and do not change "
               "the served tree: the LEARN budget is waived for this." if waiver else
               "Emit it alongside this iteration's own work.")
            + rej)

    if phase == "updating":
        rounds = doc.get("rounds") or []
        last = rounds[-1] if rounds else {}
        closed = ", ".join(f"{v} {s.replace('_', ' ')}" for s, v in sorted((last.get("closed") or {}).items()))
        limit = knob("map_update_max_iters", campaign, state)
        return (
            f"## MAP EXHAUSTED — dossier update round (iteration {doc.get('phase_iters', 1)} of at most {limit})\n\n"
            f"Map round {k} closed every direction ({closed}; {last.get('keeps', 0)} KEEPs over "
            f"iterations {last.get('started')}–{last.get('ended')}). Do not explore off the map. Run the "
            "dossier skill's update round (\"Updating the dossier\", the in-session variant) on the "
            "tree as it now is:\n\n"
            "1. **07.uK, the profile again**: the measured phase, with the profiler skill.\n"
            f"2. **10.uK, what round {k} learned**: `python3 -m arbor.collect knowledge --session {root} "
            "--config <this campaign's config> --types outcomes,findings --no-push`, then every "
            "KEEP, kill and probe of the round, and the map against the work.\n"
            "3. **11.uK, the directions again**, from 07.uK and 10.uK, with map IDs. A killed direction "
            "stays killed unless the new tree removes the reason it died or 10.uK marks it reopenable. "
            "Rank rewrites apart; when the increments are running out, write a kernel plan for each "
            "top rewrite.\n\n"
            "03.uK is brief (what the round's KEEPs changed in the code); 06.uK cites the session's "
            "bar, with no new baseline. Each part is its own `INTENT: LEARN` iteration with a LEDGER "
            "FACT naming the file it wrote; the LEARN budget is waived for this round. The round ends "
            "when you emit the new map:\n\n"
            f"    {_LIST_SYNTAX}\n\n"
            "Its first iteration is a new map round, with the streak counters reset. A map with no "
            "open increment starts the campaign afresh: a new dossier from the code read again and a "
            "new profile. If that finds none either, the harness takes the top planned rewrite as "
            "the next round's work. A second round in a row without a KEEP stops the campaign."
            + rej)

    if phase == "fresh":
        hist = doc.get("history") or doc.get("directions") or []
        by: dict[str, list[str]] = {}
        for x in hist:
            by.setdefault(str(x.get("status")).replace("_", " "), []).append(f"`{x.get('id')}`")
        past = "; ".join(f"{s}: {', '.join(v)}" for s, v in sorted(by.items()))
        limit = knob("map_fresh_max_iters", campaign, state)
        return (
            f"## FRESH START — a new dossier, as a new campaign writes one (iteration "
            f"{doc.get('phase_iters', 1)} of at most {limit})\n\n"
            f"Map round {k}'s update round found no open increment. The campaign does not stop: it "
            "starts again like a new campaign, on the tree as it now is (operator, 2026-10-22). Run "
            "the dossier skill's full update (\"Updating the dossier\", the operator's version, steps "
            "1 and 3 to 5) in this session, each part its own `INTENT: LEARN` iteration with a LEDGER "
            "FACT naming its file:\n\n"
            "1. **03.uK, the code read again whole**, method by method, each in-path method with its "
            "opportunity label and its history. Not the diff: what moved around code that did not change.\n"
            "2. **07.uK, the profile again**, plan then results, before anything is ranked.\n"
            "3. **10.uK, what the campaign learned**, every gated iteration, and which kills are reopenable.\n"
            "4. **11.uK, the directions from 03.uK and 07.uK**: what this tree's code and profile offer, "
            "not the old map with new statuses. A direction the history killed stays killed unless "
            "the new tree removes the reason it died or 10.uK marks it reopenable; a planned rewrite "
            "keeps its plan.\n\n"
            "06.uK cites the session's bar. The old map is history, not a list to carry over"
            + (f" ({past})" if past else "") + ". The fresh start ends when you emit the new map:\n\n"
            f"    {_LIST_SYNTAX}\n\n"
            "If it has no open increment, the next round carries out the top planned rewrite instead."
            + rej)

    if phase != "active":
        return ""
    dirs = doc.get("directions") or []
    n_closed = sum(1 for d in dirs if d.get("status") in CLOSED)
    ordered = sorted(dirs, key=lambda d: (d.get("status") in CLOSED, d.get("kind") != "increment",
                                          d.get("rank") if isinstance(d.get("rank"), int) else 999))
    flagged = [d for d in dirs if d.get("flagged")]
    lines = [
        f"## The map — round {k}: {n_closed} of {len(dirs)} directions closed"
        + (f" ({dur(time.time() - doc['round_started_ts'])} into the round)"
           if doc.get("round_started_ts") and dur(time.time() - doc["round_started_ts"]) else "") + "\n",
        "Work the map. Lead your DONE line with the map ID you worked; the harness marks the "
        "direction worked from it. When a direction is settled, close it on its own line:\n",
        f"    {_CLOSE_SYNTAX}\n",
        "`done`: worked out, what it could give is in the tree. `killed`: it died on evidence. "
        "`below_floor`: its ceiling is under the gate. `plan`: a rewrite whose kernel plan is "
        "written (a later round may carry it out). Reopen with `\"status\": \"open\"`. A "
        "direction the map lacks: add it with `kind` and `title` (at most "
        f"{knob('map_round_additions', campaign, state)} per round). Never write `directions.json`. "
        "When every direction is closed, the harness starts a dossier update round and the "
        "campaign continues on a new map.\n",
    ]
    rw = next((x for x in dirs if x.get("id") == doc.get("rewrite") and x.get("status") == OPEN), None)
    if rw:
        taken = rw.get("taken") or {}
        lines[1:1] = [
            f"**This round carries out the rewrite `{rw['id']}`** ({rw.get('title', '')}). After a fresh "
            "start found no increment, the harness took the top planned rewrite (operator, 2026-10-22). "
            f"Its plan: {rw.get('plan') or taken.get('why_before') or 'part 11 names it'}. Build it as the "
            "plan stages it, each stage a gated iteration held to the bar like any change. Close it `done` "
            "when the plan is delivered, `killed` on evidence, or `below_floor` when its ceiling falls "
            "under the gate; not `plan`.\n"]
    lines.extend(_line(d) for d in ordered)
    for d in flagged:
        lines.append(f"\n**`{d['id']}` was worked {len(d['worked'])} times and is still open**: close "
                     "it, or say what the next iteration on it buys that the last ones did not.")
    if doc.get("unmapped_ids"):
        lines.append("\nCited but not on the map: " + ", ".join(f"`{t}`" for t in doc["unmapped_ids"])
                     + " (add it with DIRECTION, or cite the map ID it belongs to).")
    return "\n".join(lines) + rej


def suppresses_barren(doc: dict | None, campaign=None, state=None) -> bool:
    """While the map is written or updated, the drought blocks would push the wrong way."""
    return waives_learn_cap(doc, campaign, state)


def waiver_block(doc: dict | None) -> str:
    """The program's LEARN-budget block while the cap is waived."""
    if is_lane(doc):
        return ("## LEARN budget — waived for the plan's measuring stages\n\n"
                "The lane's next item is a measuring stage (S1–S3): declare `INTENT: LEARN`, "
                "record what it measured in a STAGE line and the ledger, and leave the served "
                "tree alone until a stage needs it in the tree (S4).")
    return ("## LEARN budget — waived for the map round\n\n"
            "Writing or updating the map is LEARN work: declare `INTENT: LEARN`, record "
            "a LEDGER FACT per part written, and leave the served tree alone until the "
            "new map starts.")


# ── the command line ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "split":
        return _split_main(argv[1:])
    if argv and argv[0] == "generate":
        from arbor import mapgen
        return mapgen.main(argv[1:])
    if not argv or argv[0] != "check":
        print(__doc__[__doc__.index("    python3 -m arbor.directions"):].rstrip())
        return 2
    dossier = None
    if "--dossier" in argv:
        i = argv.index("--dossier")
        if i + 1 >= len(argv):
            print("--dossier needs the dossier's directory")
            return 2
        dossier = Path(argv[i + 1])
        del argv[i:i + 2]
    text = Path(argv[1]).read_text() if len(argv) > 1 and argv[1] != "-" else sys.stdin.read()
    if not _HEAD.search(text):
        text = "DIRECTIONS: " + text
    p = parse_agent_lines(text)
    for e in p.errors:
        print(f"REJECTED {e}")
    bad = 0
    if p.new_map is not None:
        kinds = ", ".join(f"{sum(1 for d in p.new_map if d['kind'] == k)} {k}s" for k in KINDS)
        print(f"OK: a map of {len(p.new_map)} directions ({kinds}); "
              f"{sum(1 for d in p.new_map if d['status'] == OPEN)} open")
        if dossier is not None:
            bad = _check_lanes(p.new_map, dossier)
    for e in p.closes:
        print(f"OK: DIRECTION {e['id']} -> {e.get('status', 'added')}")
    return 1 if bad or p.errors or (p.new_map is None and not p.closes) else 0


def _check_lanes(dirs: list[dict], dossier: Path) -> int:
    """Each direction's lane, its plan checked against the dossier, and the order; the failures."""
    from arbor import stages
    bad = 0
    for d in dirs:
        line = f"  {d['id']}: {lane_of(d)} lane, {d.get('status', OPEN)}"
        if d.get("depends_on"):
            line += f", after {', '.join(d['depends_on'])}"
        if d.get("nodes"):
            line += f", {d['nodes']} node(s)"
        if d.get("plan"):
            path = Path(d["plan"]) if os.path.isabs(d["plan"]) else dossier / d["plan"]
            if not path.is_file():
                line, bad = line + f"; plan {d['plan']}: MISSING", bad + 1
            else:
                items, errs = stages.parse_items(path.read_text(errors="replace"))
                if errs:
                    line, bad = line + f"; plan {d['plan']}: " + "; ".join(errs[:3]), bad + 1
                else:
                    line += f"; plan {d['plan']}: {len(items)} items"
        elif lane_of(d) == "kernel-plan" and d.get("status") == OPEN:
            line += "; no plan yet"
        print(line)
    print("Order: " + " -> ".join(topo_order(dirs)))
    return bad


# ── a host's shares: one map, several Arbors ──────────────────────────────────

# What a share keeps of a direction: what component.plan reads; status and closing fields go.
SHARE_KEYS = ("id", "kind", "title", "why", "rank", "ceiling", "lane", "plan", "depends_on",
              "nodes", "upstream")


def split(dirs: list[dict], shares: int) -> list[list[dict]]:
    """A map's open directions in ``shares`` disjoint shares, one ``component.plan`` each.

    A host that runs several Arbors from one set of directions (the 2026-11-04 release) gives each a share. Directions joined by ``depends_on``
    stay in one share: a dependency missing from a map counts as met (``deps_met``), so
    split apart, one Arbor would start what another has not finished. A dependency on a
    closed direction is dropped, being met. The groups are dealt in rank order, each to
    the share with the fewest directions so far (ties to the first), so every share starts
    with high-ranked work; within a share the map's order is kept. ValueError when there
    are fewer groups than shares.
    """
    if shares < 1:
        raise ValueError(f"shares must be at least 1, not {shares}")
    live = [d for d in dirs if d.get("status", OPEN) == OPEN]
    ids = {d["id"] for d in live}
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for d in live:
        for x in d.get("depends_on") or []:
            if x in ids:
                parent[find(x)] = find(d["id"])
    groups: dict[str, list[dict]] = {}
    for d in live:
        groups.setdefault(find(d["id"]), []).append(d)
    pos = {d["id"]: k for k, d in enumerate(live)}

    def rank(d):
        return d["rank"] if isinstance(d.get("rank"), int) else 999

    order = sorted(groups.values(),
                   key=lambda g: (min(rank(d) for d in g), min(pos[d["id"]] for d in g)))
    if len(order) < shares:
        raise ValueError(f"{len(live)} open directions in {len(order)} independent group(s) "
                         f"cannot fill {shares} shares: each share needs a group of its own")
    out: list[list[dict]] = [[] for _ in range(shares)]
    for g in order:
        out[min(range(shares), key=lambda k: (len(out[k]), k))].extend(g)
    return [[_share_entry(d, ids) for d in sorted(s, key=lambda d: pos[d["id"]])] for s in out]


def _share_entry(d: dict, live: set) -> dict:
    e = {k: d[k] for k in SHARE_KEYS if d.get(k) not in (None, "", [])}
    deps = [x for x in d.get("depends_on") or [] if x in live]
    if deps:
        e["depends_on"] = deps
    else:
        e.pop("depends_on", None)
    return e


def load_map(path) -> tuple[list[dict], list[str]]:
    """The directions in a file, and what is wrong with them: a session's directions.json,
    a JSON list or ``{"directions": [...]}``, or a DIRECTIONS line or list in text (a
    dossier's part 11). Closed entries of a JSON map are kept as closed, unchecked.
    """
    text = sys.stdin.read() if str(path) == "-" else Path(path).read_text()
    try:
        raw = json.loads(text)
    except ValueError:
        raw = None
    if raw is None:
        if not _HEAD.search(text):
            text = "DIRECTIONS: " + text
        p = parse_agent_lines(text)
        if p.new_map is None:
            return [], p.errors or [f"{path}: no DIRECTIONS list"]
        return p.new_map, p.errors
    items = raw.get("directions") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], [f'{path}: neither a list of directions nor {{"directions": [...]}}']
    out, errors, seen = [], [], set()
    for i, item in enumerate(items):
        status = _status(item.get("status", OPEN)) if isinstance(item, dict) else None
        if status in CLOSED:
            out.append({"id": str(item.get("id", "")), "status": status})
            continue
        e, err = _entry(item, listed=True)
        if err:
            errors.append(f"[{i}]: {err}")
        elif e["id"] in seen:
            errors.append(f"[{i}]: {e['id']} appears twice")
        else:
            seen.add(e["id"])
            out.append(e)
    return out, errors


def _split_main(argv: list[str]) -> int:
    import argparse
    from arbor import component
    ap = argparse.ArgumentParser(prog="python3 -m arbor.directions split",
                                 description="One map's open directions in N disjoint shares, "
                                             "one component.plan each.")
    ap.add_argument("map", help="directions.json, a JSON list, or a DIRECTIONS line or list ('-': stdin)")
    ap.add_argument("--shares", type=int, required=True, help="how many Arbors")
    ap.add_argument("--out", help="write DIR/share-K.json; without it, JSON on stdout")
    args = ap.parse_args(argv)
    dirs, errors = load_map(args.map)
    for e in errors:
        print(f"REJECTED {e}", file=sys.stderr)
    if errors or not dirs:
        print(f"{args.map}: no map to split", file=sys.stderr)
        return 1
    try:
        shares = split(dirs, args.shares)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    for k, share in enumerate(shares, 1):     # each must be a plan Arbor accepts
        bad = component.plan_entries({"component": {"plan": share}})[1]
        if bad:
            print(f"share {k}: " + "; ".join(bad), file=sys.stderr)
            return 1
    closed = sum(1 for d in dirs if d.get("status", OPEN) != OPEN)
    if not args.out:
        print(json.dumps({"shares": [{"directions": s} for s in shares]}, indent=1))
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, share in enumerate(shares, 1):
        (out / f"share-{k}.json").write_text(json.dumps({"directions": share}, indent=1) + "\n")
        print(f"share-{k}.json: {len(share)} directions ({', '.join(d['id'] for d in share)})")
    if closed:
        print(f"({closed} closed directions left out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
