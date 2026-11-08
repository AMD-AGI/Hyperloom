"""The master and its workers: which side runs the next pass, and the two records that cross between them.

A session is a master and its workers (``lifecycle.ROLES``):

- **The master** sets the campaign up, writes the map, ends each round (its A/B, the dossier's update, the next map),
  and ends the campaign.
- **A worker** runs a round: it iterates on the map the master set, until the round ends, the map leaves ``active``,
  or the run ends, and then reports.

Two records cross between them, in the session, each with its schema:

- **The work order** (``work_order.json``, ``arbor.work_order/1``): what a worker's stretch starts from. The campaign
  and its session, the round, the base, the map and its open directions, the dossier, the skills loaded, the bar and the
  target, and the limits.
- **The worker report** (``worker_report.json``, ``arbor.worker_report/1``, and ``WORKER_REPORT.md``): what the
  stretch did. Why it ended, its iterations and decisions, its KEEPs, its best and baseline with their scale, the
  directions it closed and left open, those it added, and a summary.

Every stretch's pair is kept too, as ``roles/NNNN-work_order.json`` and ``roles/NNNN-worker_report.json`` (NNNN its
first iteration). The report names its order's ``id``.

**A worker of its own session** (``arbor worker --order W --config C``, C's session not the master's): the order carries
the directions it works (``map.directions``), and the worker's session takes them as a lane (``seed_map``), the
master's dossier when it has none (``take_dossier``), and the order itself, renumbered from its own next iteration
(``localize``). ``python3 -m arbor.roles split`` cuts an order into shares, one worker each, for a host.

**Dealing**: instead of a share each up front, a free worker gets the next few directions
of the master's map that no report settled and no running worker holds (``deal``, ``python3 -m arbor.roles deal``,
from ``pool``). A direction reported back open is dealt again after the fresh ones, and not after ``MAX_RETURNS``
returns. ``fold`` writes a report's closures and proposals into the master's own map.

**How they run** (``arbor/driver.py``):
- **``arbor run``:** one process, which switches sides between passes; ``--split processes`` runs each side as its own
  command instead, in turn.
- **``arbor master --config``:** the master's passes, until the next is a worker's (its work order written) or the
  campaign ends.
- **``arbor worker --order``:** one worker's stretch, from the order to its report.
- **Under a host:** Hyperloom, the master that owns the session, calls Arbor's master, then its workers, through these
  records.

How each command ended goes in ``role_exit.json`` (``arbor.role_exit/1``): the side, the outcome (``ordered``,
``reported`` or ``ended``), the record it wrote, and the run's own reason when the run ended.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from arbor import lifecycle

ORDER_FILE, REPORT_FILE, REPORT_DOC = "work_order.json", "worker_report.json", "WORKER_REPORT.md"
#: The worker's own account of its stretch, which the ``worker-summary`` skill writes on ``stretch_end``.
ACCOUNT_DOC = "WORKER_SUMMARY.md"
ORDER_SCHEMA, REPORT_SCHEMA = "arbor.work_order/1", "arbor.worker_report/1"
EXIT_FILE, EXIT_SCHEMA = "role_exit.json", "arbor.role_exit/1"
#: How a side's command ended: the master wrote a work order; a worker wrote its report; the campaign ended.
ORDERED, REPORTED, ENDED = "ordered", "reported", "ended"
#: A report's ``ended`` when a worker of its own session handed its share back spent (``worker.Stretch.spend``).
SPENT = "spent"
HISTORY = "roles"
#: The map's phases in which it is being written; a map set in one of them (``pending``) starts at the next pass.
WRITING = ("bootstrap", "unmapped", "updating", "fresh")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def role_of(campaign, state, session_dir) -> str:
    """The side that runs the next pass, as the pass's own start (``directions.advance``) will leave the map.

    - **The master:** a pending round end; no map yet in a campaign that maps (the next pass starts one); the map being
      written or updated; a map waiting on its round end's A/B (``verifying``); and an active map with every direction
      closed, whose next pass starts the update round.
    - **A worker:** an active map with a direction open, and a map that is set (``pending``) and starts at the next
      pass, the worker's first.
    - **Without a map:** a campaign without map rounds is a worker's after its setup; a document campaign is the
      master's throughout."""

    from arbor import directions, rounds

    if rounds.pending(session_dir):
        return lifecycle.MASTER
    doc = directions.current(campaign, state)
    if doc is None and directions.enabled(campaign, state):
        return lifecycle.MASTER                 # no map yet: the next pass's start makes one, in bootstrap
    if doc:
        phase = doc.get("phase")
        if doc.get("pending") and phase in WRITING:
            return lifecycle.WORKER
        if phase == "verifying" or (phase == "active" and directions.exhausted(doc)):
            return lifecycle.MASTER
    return lifecycle.role_of_chapter(lifecycle.chapter_of(campaign, state))


def _write(session_dir, name: str, doc: dict, first: int) -> Path:
    path = Path(session_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, default=str) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    kept = Path(session_dir) / HISTORY / f"{first:04d}-{name}"
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(text, encoding="utf-8")
    return path


def load(session_dir, name: str) -> dict | None:
    """The record ``name`` (``ORDER_FILE`` or ``REPORT_FILE``) of the session's latest stretch; None without one."""

    try:
        doc = json.loads((Path(session_dir) / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ── the work order ────────────────────────────────────────────────────────────

def _head(config: dict) -> str | None:
    """The workspace's HEAD, best effort: None when there is no workspace or git cannot answer."""

    container, workspace = config.get("docker_container", ""), config.get("workspace_root", "")
    if not (container and workspace):
        return None
    try:
        from arbor.execute.workspace import git
        return git(container, workspace, "rev-parse", "HEAD").strip() or None
    except Exception:  # noqa: BLE001 - a record field, never a reason to stop the run
        return None


def _entries(doc: dict) -> list[dict]:
    """The directions a stretch that starts now works, in a DIRECTIONS line's form: a set map's, which the stretch's
    first pass makes active, or else the active map's open ones."""

    from arbor import directions

    pend = doc.get("pending") or {}
    listed = pend.get("directions") if pend else doc.get("directions")
    live = {d.get("id") for d in listed or [] if d.get("status", directions.OPEN) == directions.OPEN}
    return [directions._share_entry(d, live) for d in listed or [] if d.get("id") in live]


def work_order(campaign, state, session_dir, arbor_root, config_path, limits: dict | None = None,
               target=None) -> dict:
    """The work order for a worker's stretch that starts at the next iteration."""

    from arbor import chapters, directions, rebaseline
    from arbor.pack.loader import agent_skills
    from arbor.version import VERSION

    doc, _ = directions.load(session_dir)
    doc = doc or {}
    entries = _entries(doc)
    plan = lifecycle.declared(campaign)
    places = chapters.places(campaign, session_dir, arbor_root)
    config = getattr(campaign, "config", None) or {}
    return {
        "schema": ORDER_SCHEMA,
        "id": uuid.uuid4().hex[:12],
        "arbor": VERSION,
        "written_at": _now(),
        "campaign": {"identity": getattr(campaign, "identity", ""), "config": str(config_path),
                     "pack": getattr(campaign.pack, "name", ""), "mode": getattr(campaign, "mode", "standalone")},
        "session": str(session_dir),
        "start_iteration": state.iteration + 1,
        "round": (doc.get("map_round") or 0) + 1 if doc.get("pending") else doc.get("map_round"),
        "map": {"path": str(directions.map_path(session_dir)), "phase": doc.get("phase"),
                "scope": doc.get("scope"), "pending": bool(doc.get("pending")),
                "open": [e["id"] for e in entries], "directions": entries},
        "base": {"container": config.get("docker_container", ""), "workspace": config.get("workspace_root", ""),
                 "commit": _head(config)},
        "dossier": places.get("dossier"),
        "benchmark": places.get("benchmark"),
        "skills": list(plan.load) if plan is not None else list(agent_skills(campaign.pack) or []),
        "bar": {"metric": state.metric_name, "direction": state.metric_direction, "baseline": state.baseline,
                "best": state.best_metric, "scale": rebaseline.current(state.lessons)[0]},
        "target": target,
        "limits": dict(limits or {}),
    }


def write_order(session_dir, order: dict) -> Path:
    return _write(session_dir, ORDER_FILE, order, int(order.get("start_iteration") or 0))


def read_order(path) -> dict:
    """A work order from its file; ValueError when it is not one this Arbor can run."""

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != ORDER_SCHEMA:
        raise ValueError(f"{path} is not a work order ({ORDER_SCHEMA}): its schema is "
                         f"{doc.get('schema') if isinstance(doc, dict) else type(doc).__name__}")
    for key in ("id", "campaign", "session", "start_iteration"):
        if not doc.get(key):
            raise ValueError(f"{path}: the work order has no {key}")
    if not (doc["campaign"].get("config") if isinstance(doc["campaign"], dict) else None):
        raise ValueError(f"{path}: the work order names no campaign config")
    return doc


def split_order(order: dict, shares: int) -> list[dict]:
    """``order`` in ``shares`` orders, each with a share of its directions (``directions.split``: those joined by
    ``depends_on`` stay together) and an id of its own; ``share`` names the order it came from. ValueError when the
    directions do not fill the shares."""

    from arbor import directions

    out = []
    for k, part in enumerate(directions.split(order["map"].get("directions") or [], shares), 1):
        doc = json.loads(json.dumps(order))
        doc["id"] = uuid.uuid4().hex[:12]
        doc["written_at"] = _now()
        doc["share"] = {"index": k, "of": shares, "order": order["id"]}
        doc["map"].update(directions=part, open=[d["id"] for d in part])
        out.append(doc)
    return out


# ── dealing: the master's map handed out a few directions at a time ──────────

#: A direction reported back open this often is not dealt again in the round: that many workers could not settle it.
MAX_RETURNS = 2


def pool(order: dict, reports=(), held=()) -> dict:
    """Where each of ``order``'s directions (the master's map) stands, from the ``reports`` its workers wrote and the
    directions running workers ``held``: ``settled`` (a report closed it, with its status), ``held``, ``returned``
    (reported back open, how often), ``exhausted`` (returned ``MAX_RETURNS`` times), and ``free`` (none of those;
    ``deal`` still holds one back while a direction it depends on is held, or exhausted and unsettled)."""

    entries = list(order["map"].get("directions") or [])
    ids = [d["id"] for d in entries]
    settled: dict[str, str] = {}
    returned: Counter = Counter()
    for report in reports:
        for d in report.get("directions", {}).get("closed") or []:
            if d.get("id") in ids:
                settled[d["id"]] = d.get("status") or "done"
        for did in report.get("directions", {}).get("open") or []:
            if did in ids:
                returned[did] += 1
    held = [d for d in held if d in ids]
    exhausted = [d for d in ids if returned[d] >= MAX_RETURNS and d not in settled and d not in held]
    free = [d for d in ids if d not in settled and d not in held and d not in exhausted]
    return {"settled": settled, "held": held, "returned": dict(returned), "exhausted": exhausted, "free": free}


def deal(order: dict, reports=(), held=(), size: int = 2) -> dict | None:
    """The next work order from the master's ``order``: up to ``size`` free directions (``pool``), never dealt ones
    before returned ones, then by rank and the map's order. A direction comes with the free ones it depends on, or
    waits while one it depends on is held; a group bigger than ``size`` is dealt whole. None when nothing is free."""

    if size < 1:
        raise ValueError(f"size must be at least 1, not {size}")
    state = pool(order, reports, held)
    entries = {d["id"]: d for d in order["map"].get("directions") or []}
    position = {did: k for k, did in enumerate(entries)}
    free = set(state["free"])

    def group(did: str, seen: set) -> set | None:
        """``did`` with the free directions it depends on; None while one is held, or exhausted and unsettled."""
        seen.add(did)
        for dep in entries[did].get("depends_on") or []:
            if dep in seen or dep not in entries or state["settled"].get(dep):
                continue
            if dep not in free:
                return None
            sub = group(dep, seen)
            if sub is None:
                return None
        return seen

    def rank(did: str) -> tuple:
        r = entries[did].get("rank")
        return state["returned"].get(did, 0), r if isinstance(r, int) else 999, position[did]

    for first in sorted(free, key=rank):
        dealt = group(first, set())
        if dealt is None:
            continue
        for did in sorted(free - dealt, key=rank):
            if len(dealt) >= size:
                break
            more = group(did, set(dealt))
            if more is not None and len(more) <= size:
                dealt = more
        part = [{k: v for k, v in entries[did].items()} for did in sorted(dealt, key=lambda d: position[d])]
        doc = json.loads(json.dumps(order))
        doc["id"] = uuid.uuid4().hex[:12]
        doc["written_at"] = _now()
        doc["deal"] = {"order": order["id"], "returned": {d: state["returned"][d] for d in dealt
                                                          if state["returned"].get(d)}}
        doc["map"].update(directions=part, open=[d["id"] for d in part])
        doc.pop("share", None)
        return doc
    return None


def fold(session_dir, report: dict, at: int | None = None) -> dict:
    """A worker's report into the master's map (``directions.json`` in ``session_dir``): each direction it closed is
    closed there too, with its status, why and the report's order, unless the map already settled it; its proposals
    join the map's, for the master's update round. ``at`` is the master's iteration to close them at (a worker of its
    own session counts its own); the report's last by default. What changed."""

    from arbor import directions

    doc, _ = directions.load(session_dir)
    if doc is None:
        return {"closed": [], "proposals": 0}
    by_id = {d.get("id"): d for d in doc.get("directions") or []}
    closed = []
    for d in report.get("directions", {}).get("closed") or []:
        mine = by_id.get(d.get("id"))
        if mine is not None and mine.get("status") == directions.OPEN:
            mine.update(status=d.get("status") or "done", why=d.get("why") or "", closed_by=report.get("order"),
                        closed_at=at if at is not None else (report.get("iterations") or [None, None])[1])
            if d.get("spent"):
                mine["spent"] = True
            closed.append(mine["id"])
    have = {json.dumps(p, sort_keys=True) for p in doc.get("proposals") or []}
    added = [p for p in report.get("directions", {}).get("proposals") or []
             if json.dumps(p, sort_keys=True) not in have]
    doc["proposals"] = list(doc.get("proposals") or []) + added
    directions.save(session_dir, doc)
    return {"closed": closed, "proposals": len(added)}


def localize(order: dict, session_dir, state) -> dict:
    """The order as a worker of its own session runs it: its stretch starts at that session's next iteration; ``from``
    keeps where the master wrote it. The id stays, so the report still answers the master's order."""

    doc = json.loads(json.dumps(order))
    doc["from"] = {"session": order["session"], "start_iteration": order["start_iteration"]}
    doc.update(session=str(session_dir), start_iteration=state.iteration + 1)
    return doc


def seed_map(session_dir, order: dict) -> str:
    """A worker's session's map from its order, once: a lane of the order's directions (``directions.seed_lane``). A
    session that has a map keeps it, as a resume does."""

    from arbor import directions

    doc, _ = directions.load(session_dir)
    if doc is not None:
        return f"the work order: kept the map this session already has ({len(doc.get('directions') or [])} directions)"
    entries = []
    for raw in order["map"].get("directions") or []:
        e, err = directions._entry(raw, listed=True)
        if err:
            raise ValueError(f"the work order's direction {raw.get('id')}: {err}")
        entries.append(e)
    share = order.get("share") or {}
    directions.seed_lane(session_dir, entries, {
        "run": order["id"], "lane": f"share {share['index']} of {share['of']}" if share else order["id"],
        "kind": "work-order", "source": str(order.get("session") or "")})
    return f"the work order: this worker works {len(entries)} directions ({', '.join(e['id'] for e in entries)})"


def take_dossier(order: dict, place) -> str | None:
    """The master's dossier, copied to ``place`` (this campaign's) when it has none; what was done, for the log."""

    import shutil

    theirs, mine = order.get("dossier"), Path(place)
    if mine.exists() or not theirs or Path(theirs).resolve() == mine.resolve():
        return None
    if not Path(theirs).is_dir():
        return f"the work order's dossier {theirs} is not there: this worker has none"
    shutil.copytree(theirs, mine)
    return f"the master's dossier copied to {mine}"


def reported(session_dir, order: dict) -> bool:
    """Whether the session's latest report answers ``order``."""

    report = load(session_dir, REPORT_FILE)
    return bool(report) and report.get("order") == order.get("id")


def open_order(session_dir) -> dict | None:
    """The session's latest work order while it waits for its worker's report; None otherwise."""

    order = load(session_dir, ORDER_FILE)
    return order if order and not reported(session_dir, order) else None


def write_exit(session_dir, side: str, outcome: str, path=None, reason: str | None = None,
               detail: str = "") -> dict:
    """``role_exit.json``: how a side's command ended, for the caller that runs the next."""

    doc = {"schema": EXIT_SCHEMA, "side": side, "outcome": outcome, "path": str(path) if path else None,
           "reason": reason, "detail": detail, "written_at": _now()}
    target = Path(session_dir) / EXIT_FILE
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return doc


# ── the worker report ─────────────────────────────────────────────────────────

def worker_report(campaign, state, session_dir, order: dict, ended: str, detail: str = "") -> dict:
    """What a worker's stretch did, from the session: its rows of ``session.json`` from the order's first iteration
    on, and the map as it stands."""

    from arbor import directions, rebaseline
    from arbor.version import VERSION

    first = int(order.get("start_iteration") or 1)
    rows = [row for row in state.lessons if isinstance(row.get("iteration"), int) and row["iteration"] >= first]
    doc, _ = directions.load(session_dir)
    doc = doc or {}
    dirs = doc.get("directions") or []
    listed = set((order.get("map") or {}).get("open") or [])

    def since(value) -> bool:
        return isinstance(value, int) and value >= first

    report = {
        "schema": REPORT_SCHEMA,
        "order": order.get("id"),
        "arbor": VERSION,
        "written_at": _now(),
        "session": str(session_dir),
        "round": order.get("round"),
        "iterations": [first, max(first - 1, state.iteration)],
        "ended": ended,
        "detail": detail,
        "decisions": dict(Counter(str(row.get("decision")) for row in rows if row.get("decision"))),
        "keeps": [{key: row.get(key) for key in ("iteration", "change", "metric", "secondary")}
                  for row in rows if row.get("decision") == "KEEP"],
        "bar": {"metric": state.metric_name, "direction": state.metric_direction, "baseline": state.baseline,
                "best": state.best_metric, "scale": rebaseline.current(state.lessons)[0]},
        "directions": {
            "closed": [{"id": d.get("id"), "status": d.get("status"), "why": d.get("why", ""),
                        **({"spent": True} if d.get("spent") else {})}
                       for d in dirs if d.get("status") != directions.OPEN and since(d.get("closed_at"))],
            "open": [d.get("id") for d in dirs if d.get("status") == directions.OPEN],
            # The agent's own: not the map's, which are older or the order's (a lane a host seeds at the stretch's first
            # iteration reads as added at it).
            "added": [d.get("id") for d in dirs if since(d.get("added_at")) and d.get("id") not in listed],
            "proposals": list(doc.get("proposals") or []),
        },
    }
    report["summary"] = summary(report)
    return report


def _number(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


def summary(report: dict) -> str:
    """The report in a few sentences: what a reader of the round needs first."""

    first, last = report["iterations"]
    counts = report["decisions"]
    done = ", ".join(f"{n} {name}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1])) or "no decisions"
    bar = report["bar"]
    gain = ""
    if isinstance(bar.get("best"), (int, float)) and isinstance(bar.get("baseline"), (int, float)) and bar["baseline"]:
        gain = f" ({(bar['best'] / bar['baseline'] - 1) * 100:+.1f}% on the baseline {_number(bar['baseline'])})"
    closed = report["directions"]["closed"]
    parts = [
        f"Round {report.get('round')}, iterations {first}-{last}: {done}.",
        f"Best {bar.get('metric') or 'metric'} {_number(bar.get('best'))}{gain}.",
        (f"Closed {len(closed)}: " + "; ".join(f"{d['id']} {d['status']}" for d in closed) + ".") if closed else
        "Closed no direction.",
        f"Open: {', '.join(report['directions']['open']) or 'none'}.",
        f"Ended: {report['ended']}" + (f" ({report['detail']})" if report.get("detail") else "") + ".",
    ]
    return " ".join(parts)


def _markdown(report: dict) -> str:
    lines = [f"# Worker report: round {report.get('round')}, iterations {report['iterations'][0]}-"
             f"{report['iterations'][1]}", "", report["summary"], "", "## KEEPs", ""]
    lines += [f"- iteration {k['iteration']}: {k.get('change')} ({_number(k.get('metric'))})" for k in report["keeps"]]
    lines += ["- none"] if not report["keeps"] else []
    lines += ["", "## Directions", ""]
    lines += [f"- **{d['id']}** {d['status']}: {d.get('why') or ''}".rstrip(": ") for d in report["directions"]["closed"]]
    lines += [f"- open: {', '.join(report['directions']['open']) or 'none'}"]
    if report["directions"]["added"]:
        lines.append(f"- added in this round: {', '.join(report['directions']['added'])}")
    if report["directions"]["proposals"]:
        lines.append(f"- proposed to the master: {len(report['directions']['proposals'])}")
    return "\n".join(lines) + "\n"


def account_mark(session_dir) -> tuple | None:
    """What ``ACCOUNT_DOC`` is now, to tell whether a stretch's end wrote it: None when there is none."""

    try:
        st = (Path(session_dir) / ACCOUNT_DOC).stat()
    except OSError:
        return None
    return st.st_ino, st.st_size, st.st_mtime_ns


def attach_account(session_dir, report: dict, before: tuple | None) -> bool:
    """Name the worker's account in its report (``account``) when the stretch's end wrote one: ``ACCOUNT_DOC``, changed
    since ``before`` (``account_mark`` before the end's skills ran). Its copy is kept with the stretch's records, as
    ``roles/NNNN-WORKER_SUMMARY.md``. Whether it did."""

    after = account_mark(session_dir)
    if after is None or after == before:
        return False
    first = int(report["iterations"][0])
    kept = Path(session_dir) / HISTORY / f"{first:04d}-{ACCOUNT_DOC}"
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes((Path(session_dir) / ACCOUNT_DOC).read_bytes())
    report["account"] = ACCOUNT_DOC
    _write(session_dir, REPORT_FILE, report, first)
    return True


def write_report(session_dir, report: dict) -> Path:
    path = _write(session_dir, REPORT_FILE, report, int(report["iterations"][0]))
    (Path(session_dir) / REPORT_DOC).write_text(_markdown(report), encoding="utf-8")
    return path


def order_of_map(path) -> dict:
    """A host's map (a session's ``directions.json``, a JSON list or ``{"directions": [...]}``, as ``arbor.directions
    split`` reads it) as an order to deal from: its open directions, in the share form. ValueError when it does not
    read as a map."""

    import hashlib

    from arbor import directions

    dirs, errors = directions.load_map(path)
    if errors:
        raise ValueError("; ".join(errors))
    live = {d["id"] for d in dirs if d.get("status", directions.OPEN) == directions.OPEN}
    entries = [directions._share_entry(d, live) for d in dirs if d["id"] in live]
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return {"schema": ORDER_SCHEMA, "id": f"map-{digest}", "map": {"directions": entries, "open": sorted(live)}}


def held_in(path) -> list[str]:
    """The directions an order a running worker holds names: a work order's, a share's (``{"directions": [...]}``), or
    a JSON list's."""

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and doc.get("schema") == ORDER_SCHEMA:
        return list(doc["map"].get("open") or [d["id"] for d in doc["map"].get("directions") or []])
    items = doc.get("directions") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        raise ValueError(f"{path}: neither a work order, a share nor a list of directions")
    return [d["id"] if isinstance(d, dict) else str(d) for d in items]


def _deal_main(order: dict, reports: list, held: list, size: int, out: Path, form: str = "order") -> int:
    """``deal``'s command: on stdout, one JSON line a host reads. ``form`` ``order``: the work order written to ``out``
    when there is one. ``share``: ``{"directions": [...]}`` written to ``out`` always, the list empty when nothing is
    left (what a host's ``component.plan`` takes)."""

    doc = deal(order, reports, held, size)
    state = pool(order, reports, held)
    line = {"order": None, "directions": [], "free": len(state["free"]), "held": state["held"],
            "settled": state["settled"], "exhausted": state["exhausted"]}
    if form == "share":
        out.parent.mkdir(parents=True, exist_ok=True)
        part = doc["map"]["directions"] if doc else []
        out.write_text(json.dumps({"directions": part}, indent=1) + "\n", encoding="utf-8")
        line.update(order=str(out), directions=[d["id"] for d in part])
    elif doc is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        line.update(order=str(out), id=doc["id"], directions=doc["map"]["open"])
    print(json.dumps(line))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python3 -m arbor.roles",
                                 description="The work order and the worker report, for a caller that runs workers.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="an order in N shares, one worker each (directions joined by depends_on stay "
                                      "together)")
    sp.add_argument("--order", required=True, help="the master's work_order.json")
    sp.add_argument("--shares", type=int, required=True)
    sp.add_argument("--out", required=True, help="writes DIR/work_order-K.json")
    dp = sub.add_parser("deal", help="the next order for a free worker: a few directions of the master's map that no "
                                     "report settled and no running worker holds")
    source = dp.add_mutually_exclusive_group(required=True)
    source.add_argument("--order", help="the master's work_order.json")
    source.add_argument("--map", help="a host's map: a directions.json, a JSON list or {\"directions\": [...]}")
    dp.add_argument("--report", action="append", default=[], help="a worker_report.json written so far (repeat)")
    dp.add_argument("--reports", nargs="+", default=[], help="worker_report.json files written so far")
    dp.add_argument("--held", action="append", default=[], help="a direction a running worker holds (repeat)")
    dp.add_argument("--held-order", action="append", default=[],
                    help="an order or share a running worker holds: all its directions are held (repeat)")
    dp.add_argument("--size", type=int, default=2, help="directions per order (default 2)")
    dp.add_argument("--format", choices=("order", "share"), default=None,
                    help="order: a work order, written when there is one (the default with --order); share: "
                         "{\"directions\": [...]}, always written, empty when nothing is left (the default with --map)")
    dp.add_argument("--out", required=True, help="where to write it")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "deal" and args.map:
            order = order_of_map(args.map)
        else:
            order = json.loads(Path(args.order).read_text(encoding="utf-8"))
            if order.get("schema") != ORDER_SCHEMA or not order.get("id"):
                raise ValueError(f"not a work order ({ORDER_SCHEMA}) with an id")
        if args.cmd == "deal":
            reports = [json.loads(Path(r).read_text(encoding="utf-8")) for r in [*args.report, *args.reports]]
            held = [*args.held, *(d for o in args.held_order for d in held_in(o))]
            form = args.format or ("share" if args.map else "order")
            return _deal_main(order, reports, held, args.size, Path(args.out), form)
        parts = split_order(order, args.shares)
    except (OSError, ValueError, KeyError) as exc:
        print(f"{getattr(args, 'map', None) or args.order}: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for doc in parts:
        k = doc["share"]["index"]
        (out / f"work_order-{k}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"work_order-{k}.json: {doc['id']}, {len(doc['map']['directions'])} directions "
              f"({', '.join(doc['map']['open'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
