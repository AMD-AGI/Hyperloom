"""Lanes: one direction each, an Arbor campaign on its own node, from queue to closure.

A lane's life (its `lane.json`):

    WAITING_DEPS -> (every dependency CLOSED) QUEUED -> (the scheduler) SUBMITTING
      -> PENDING <-> RUNNING -> (LANE_CLOSED.json, and no chain job live) CLOSED
    a chain that ends without closing: FAILED -> resubmitted once, without its node -> PENDING;
    failing again: BROKEN, and the run pauses for the operator.

A CLOSED lane is never submitted again: nothing reopens inside a run.

Before its first chunk a lane gets its start tree (the base, plus each
dependency's final tree that merges, exported as `start/base.{diff,blobs}`,
which the launcher deploys as `base_vmu`), its own worktree of the run's
pinned Arbor commit (lanes must not share a pack's eval or calibration files),
its config (the template campaign's, with the overrides of `kinds.py`), and
its map, seeded into its session (`arbor.stages seed`, or the direction as one
item). The launcher runs it as a rolling chain (scripts/cluster/chunk_lib.sh);
the chain's head is found through `<session>/chain/next_of_<job>`.

A lane's clocks, in `lane.json`: created, deps_met, queued, first_submitted,
started and closed; `wait_deps_s`, `wait_node_s` (queued, or pending with no
chunk running), `on_node_s` (time with a chunk running), `node_s` (that times
the nodes it holds) and `chunks`. At closure sacct's intervals replace what
the ticks observed, and each item's wall time comes from LANE_CLOSED.json.

An Unknown job state changes nothing; every submit records its job name first.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arbor import directions
from arbor.execute.workspace import HOST
from arbor.parallel import kinds, status
from arbor.parallel.cluster import Failed, Known, Unknown
from arbor.parallel.scheduler import Slot
from arbor.parallel.store import atomic_write_json, atomic_write_text, read_json
from arbor.parallel.trees import (Conflict, FlagConflict, flags_from_workspace, flags_to_workspace,
                                  merge_flag_sets, strip_artifacts, workspace_from_session)

ACTIVE = ("SUBMITTING", "PENDING", "RUNNING")
FINAL = ("CLOSED", "STOPPED")
INTENT_RANK = {"LEARN": 0, "FIX-CORRECTNESS": 1, "FIX": 2, "WIN": 3}
MAX_REFUSALS = 3


@dataclass
class Lane:
    lane: str
    round: int
    direction: str
    kind: str
    title: str = ""
    rank: int = 999
    level: int = 0
    nodes: int = 1
    depends_on: list[str] = field(default_factory=list)       # lane IDs of this round
    plan: dict | None = None                                  # plans.pin's record
    state: str = "WAITING_DEPS"
    campaign: str = ""
    config_path: str = ""
    session_root: str = ""
    worktree: str = ""
    jobs: list[str] = field(default_factory=list)             # the first job of each chain
    job_name: str | None = None
    head: str | None = None
    node: str | None = None
    exclude: list[str] = field(default_factory=list)
    resubmits: int = 0
    refusals: int = 0
    failures: list[dict] = field(default_factory=list)
    start: dict | None = None
    seeded: str = ""
    clocks: dict = field(default_factory=dict)
    closing: dict | None = None                               # the operator's close-lane
    outcome: dict | None = None
    final: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Lane":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def slot(self) -> Slot:
        return Slot(self.lane, self.state, self.level, self.rank, self.nodes, tuple(self.depends_on))


def load(store, n: int, lane: str) -> Lane | None:
    d = store.lane(n, lane)
    return Lane.from_dict(d) if d else None


def load_all(store, n: int) -> list[Lane]:
    return [Lane.from_dict(store.lane(n, x)) for x in store.lanes(n)]


def save(ctx, rec: Lane, kind: str | None = None, **fields: Any) -> None:
    ctx.store.set_lane(rec.round, rec.lane, rec.to_dict())
    if kind:
        ctx.store.event(kind, lane=rec.lane, **fields)


def lane_dir(ctx, rec: Lane) -> Path:
    return ctx.store.lane_dir(rec.round, rec.lane)


def campaign_name(run_id: str, lane: str) -> str:
    """What the launcher calls the lane (CAMPAIGN): its log, its chunks' job names, its session."""
    return f"par-{run_id}-{lane}"


# -- clocks ----------------------------------------------------------------------------


def _to(ctx, rec: Lane, state: str, **event: Any) -> None:
    """Move the lane to ``state``, closing the clock interval the old state was counting."""
    now, c = ctx.clock(), rec.clocks
    span = now - c["since"] if c.get("since") is not None else 0.0
    if c.get("counting") == "wait":
        c["wait_node_s"] = round(c.get("wait_node_s", 0.0) + span, 1)
    elif c.get("counting") == "run":
        c["on_node_s"] = round(c.get("on_node_s", 0.0) + span, 1)
    c["counting"] = ("run" if state == "RUNNING" else
                     "wait" if state in ("QUEUED", "SUBMITTING", "PENDING", "FAILED") else None)
    c["since"] = now if c["counting"] else None
    if state == "RUNNING":
        c.setdefault("started", now)
    if state == "PENDING":
        c.setdefault("first_submitted", now)
    if state == "QUEUED":
        c.setdefault("queued", now)
    rec.state = state
    kind = event.pop("kind", None)
    save(ctx, rec, kind, **event)


def live_clocks(rec: Lane, now: float) -> dict:
    """The lane's clocks as of ``now``, counting the interval in progress (status.py's reading)."""
    return status.live_clocks(rec.to_dict(), now)


# -- creation and the queue ------------------------------------------------------------


def create(ctx, n: int, d: dict, lane_id: str, *, kind: str, level: int, depends_on: list[str],
           plan: dict | None) -> Lane:
    now = ctx.clock()
    rec = Lane(lane=lane_id, round=n, direction=d["id"], kind=kind, title=str(d.get("title") or ""),
               rank=d["rank"] if isinstance(d.get("rank"), int) else 999, level=level,
               nodes=kinds.nodes(kind, d), depends_on=list(depends_on), plan=plan,
               campaign=campaign_name(ctx.cfg.run_id, lane_id), clocks={"created": now})
    lane_dir(ctx, rec).mkdir(parents=True, exist_ok=True)
    atomic_write_json(lane_dir(ctx, rec) / "direction.json", d)
    rec.state = "WAITING_DEPS"
    save(ctx, rec, "lane.created", direction=d["id"], lane_kind=kind, level=level, depends_on=depends_on)
    if not depends_on:
        promote(ctx, rec)
    return rec


def promote(ctx, rec: Lane) -> None:
    now = ctx.clock()
    rec.clocks["deps_met"] = now
    rec.clocks["wait_deps_s"] = round(now - rec.clocks.get("created", now), 1)
    _to(ctx, rec, "QUEUED", kind="lane.queued")


# -- the start tree --------------------------------------------------------------------


def start_tree(ctx, rec: Lane, by_id: dict[str, Lane], base: dict) -> dict:
    """The base plus each dependency's final tree, in rank order; what each dependency gave."""
    sha, flags, used = base["sha"], base.get("flags") or {}, []
    for dep in sorted((by_id[x] for x in rec.depends_on), key=lambda r: (r.rank, r.lane)):
        f = dep.final or {}
        row = {"lane": dep.lane, "direction": dep.direction, "used": False}
        if not f.get("kept"):
            row["why"] = f.get("why") or "closed with no KEEP"
            used.append(row)
            continue
        merged = ctx.trees.merge(sha, f["commit"], message=f"{rec.lane}: + {dep.lane}") if f.get("commit") else sha
        fl = merge_flag_sets(dep.start.get("flags") or {}, flags, f.get("flags") or {})
        if isinstance(merged, Conflict) or isinstance(fl, FlagConflict):
            row["why"] = "conflicts with the tree before it in " + ", ".join(
                list(merged.paths if isinstance(merged, Conflict) else ())
                + [f"flag:{k}" for k in (fl.keys if isinstance(fl, FlagConflict) else ())])
            used.append(row)
            continue
        sha, flags = merged, fl
        used.append({**row, "used": True, "keeps": f.get("keeps") or []})
    return {"sha": sha, "flags": flags, "base": base["index"], "deps": used}


def _dep_lines(rec: Lane) -> list[str]:
    out = []
    for d in (rec.start or {}).get("deps") or []:
        if d["used"]:
            out.append(f"plus {d['direction']}'s final tree (lane {d['lane']}, its KEEPs at iterations "
                       f"{', '.join(map(str, d.get('keeps') or [])) or '-'})")
        else:
            out.append(f"WITHOUT {d['direction']} (lane {d['lane']}): {d['why']}; say in your verdicts what "
                       "its outcome changed for this direction")
    return out


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def prepare(ctx, rec: Lane, by_id: dict[str, Lane], others: list[str]) -> None:
    """Everything a lane's first chunk needs, each step skipped once done (a crash re-runs the rest)."""
    base = ctx.store.current_base()
    d = lane_dir(ctx, rec)
    if rec.start is None:
        rec.start = start_tree(ctx, rec, by_id, base)
        save(ctx, rec, "lane.start_tree", sha=rec.start["sha"], deps=rec.start["deps"])
    if not (d / "start" / "base.blobs").exists():
        ctx.trees.export(rec.start["sha"], d / "start" / "base")
        atomic_write_json(d / "start" / "flags.json", rec.start["flags"])
        atomic_write_json(d / "start" / "workspace_config.json", flags_to_workspace(rec.start["flags"]))
    if not rec.worktree:
        rec.worktree = str(_worktree(ctx, d / "arbor"))
    tmpl = ctx.cfg.template()
    rec.session_root = f"{ctx.cfg.lane_sessions.rstrip('/')}/{rec.campaign}"
    rec.config_path = str(d / "config.json")
    atomic_write_json(rec.config_path, derive_config(ctx, rec, tmpl, others))
    session = Path(rec.session_root)
    aiter = ctx.store.base_dir(base["index"]) / "landed_aiter.py"
    if aiter.is_file() and not (session / "landed_aiter.py").exists():
        session.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aiter, session / "landed_aiter.py")
    if not rec.seeded:
        plan = Path(rec.plan["path"]) if rec.plan else None
        items = kinds.default_items(rec.kind)
        if plan is None and items:
            plan = kinds.items_plan(d, items, rec.direction)
        direction = read_json(d / "direction.json", {"id": rec.direction})
        rec.seeded = kinds.seed(kind=rec.kind, config_path=Path(rec.config_path), session=session,
                                direction=direction, plan=plan, run=ctx.cfg.run_id, round_no=rec.round,
                                lane_id=rec.lane)
    save(ctx, rec)


def _worktree(ctx, path: Path) -> Path:
    """The lane's own checkout of the run's pinned Arbor commit."""
    if (path / ".git").exists():
        return path
    sha = ctx.store.meta()["arbor_sha"]
    p = subprocess.run(["git", "-C", str(ctx.cfg.arbor_root), "worktree", "add", "--detach", "-f", str(path), sha],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git worktree add {path}: {p.stderr.strip()[:300]}")
    return path


def derive_config(ctx, rec: Lane, tmpl: dict, others: list[str]) -> dict:
    """The template campaign's config, as this lane runs it."""
    cfg, d = ctx.cfg, lane_dir(ctx, rec)
    direction = read_json(d / "direction.json", {"id": rec.direction})
    base = f"base_{rec.start['base']:03d}"
    # A lane has no round end of its own (arbor/rounds.py): the run's integration is its A/B.
    c = {k: v for k, v in tmpl.items() if k not in ("time_budget_minutes", "target", "pinned_intent", "round_ab")}
    c.update(flags_to_workspace(rec.start["flags"]))
    c.update(kinds.overrides(rec.kind, cfg.kind(rec.kind), tmpl))
    c["identity"] = f"parallel:{cfg.run_id}:r{rec.round:02d}:{rec.direction}"
    c["session_root"] = rec.session_root
    # A lane is the run's, which a person reviewed when they submitted it: its own review gate
    # (arbor/review.py) would hold the lane's GPU node until the job's wall (#30 c4555).
    c["review_gate"] = False
    if tmpl.get("docker_container") == HOST:          # a document lane: its own workspace on the host
        c["workspace_root"] = str(d / "ws")
    else:
        c["docker_container"] = f"par_{cfg.run_id}_{rec.lane}".replace("-", "_")
    c["base_vmu"] = str(d / "start" / "base")
    c["baseline_record"] = str(d / "baseline.json")
    c["directives"] = list(tmpl.get("directives") or []) + kinds.directives(
        rec.kind, lane=rec.lane, run=cfg.run_id, round_no=rec.round, d=direction,
        plan=(rec.plan or {}).get("path"), others=others, deps=_dep_lines(rec), base=base,
        image_tag=str(tmpl.get("image_tag") or ""))
    c["parallel"] = {"run": cfg.run_id, "round": rec.round, "lane": rec.lane, "direction": rec.direction,
                     "kind": rec.kind, "lane_dir": str(d), "store": str(ctx.store.root)}
    return c


# -- jobs ------------------------------------------------------------------------------


def submit(ctx, rec: Lane) -> Lane:
    """Submit the lane's next chain: the name first, then sbatch."""
    (Path(rec.session_root) / "chain" / "CHAIN_STOP").unlink(missing_ok=True)   # a chain that ended wrote it
    cl = ctx.cfg.cluster
    rec.job_name = f"{rec.campaign}-c{len(rec.jobs)}"
    if rec.state != "SUBMITTING":
        _to(ctx, rec, "SUBMITTING", kind="lane.submitting", job_name=rec.job_name)
    env = {"CAMPAIGN": rec.campaign, "CAMPAIGN_CONFIG": rec.config_path, "CAMPAIGN_ARBOR": rec.worktree,
           "ARBOR_ROOT": rec.worktree, "CALIBRATE": "0", "MAX_CHUNKS": "0", "ROUND_AB": "0",
           "PAR_LANE_DIR": str(lane_dir(ctx, rec)), **kinds.env(rec.kind)}
    r = ctx.cluster.submit(str(Path(rec.worktree) / cl.lane_script), name=rec.job_name, env=env,
                           exclude=rec.exclude, time_limit=cl.time_limit,
                           output=f"{cl.log_dir}/campaign_{rec.campaign}.out" if cl.log_dir else None)
    if isinstance(r, Failed):
        rec.refusals += 1
        rec.failures.append({"ts": ctx.clock(), "why": f"sbatch refused it: {r.reason}"})
        if rec.refusals >= MAX_REFUSALS:
            _to(ctx, rec, "BROKEN", kind="lane.broken", why=f"sbatch refused it {rec.refusals} times: {r.reason}")
        else:
            save(ctx, rec, "lane.submit_refused", reason=r.reason)
    elif isinstance(r, Unknown):
        save(ctx, rec, "lane.submit_unknown", job_name=rec.job_name)
    else:
        rec.jobs.append(r)
        rec.head = r
        _to(ctx, rec, "PENDING", kind="lane.submitted", job=r)
    return rec


def chain(session_root: str, first: str) -> list[str]:
    """A chain's jobs in order: its first, then each successor `<session>/chain/next_of_<job>` names."""
    out, job = [], str(first)
    while job and job not in out:
        out.append(job)
        nxt = Path(session_root) / "chain" / f"next_of_{job}"
        job = nxt.read_text().strip() if nxt.exists() else ""
    return out


def chain_head(session_root: str, first: str) -> str:
    return chain(session_root, first)[-1]


def chain_jobs(rec: Lane) -> list[str]:
    """Every job of every chain the lane ran."""
    return [j for first in rec.jobs for j in chain(rec.session_root, first)]


def reconcile(ctx, rec: Lane, *, mode: str = "run") -> Lane:
    """Move a lane to what its jobs and its session say; ``mode`` run, paused or stopping."""
    if rec.state == "SUBMITTING":
        found = ctx.cluster.find_by_name(rec.job_name) if rec.job_name else []
        if found is None:
            return rec
        if found:
            rec.jobs.append(found[0])
            rec.head = found[0]
            _to(ctx, rec, "PENDING", kind="lane.found", job=found[0])
        elif mode == "run":
            submit(ctx, rec)                      # sbatch never queued it, or the tick died first
        return rec
    if rec.state not in ("PENDING", "RUNNING") or not rec.jobs:
        return rec
    jobs = chain(rec.session_root, rec.jobs[-1])
    head = jobs[-1]
    if head != rec.head:
        rec.head = head
        save(ctx, rec, "lane.rolled", job=head)
    # The head and its predecessor: during a handoff the successor waits while the
    # predecessor still runs, and the lane is on a node all along.
    sts = ctx.cluster.status(jobs[-2:])
    if any(s is None or isinstance(s, Unknown) for s in (sts.get(j) for j in jobs[-2:])):
        return rec
    st = sts[head]
    running = next((s for s in sts.values() if isinstance(s, Known) and s.state == "RUNNING"), None)
    now = "RUNNING" if running else st.state if isinstance(st, Known) and st.state == "PENDING" else None
    if now:
        node = (running.node if running else None) or rec.node
        if now != rec.state or node != rec.node:
            rec.node = node
            _to(ctx, rec, now, kind=f"lane.{now.lower()}", job=head, node=node)
        return rec
    rec.node = getattr(st, "node", None) or rec.node
    return _ended(ctx, rec, st, mode)


def _ended(ctx, rec: Lane, st, mode: str) -> Lane:
    session = Path(rec.session_root)
    if (session / directions.LANE_FILE).exists() or rec.closing:
        return close(ctx, rec)
    if mode == "stopping":
        _to(ctx, rec, "STOPPED", kind="lane.stopped", job=rec.head)
        _finish_clocks(ctx, rec)
        save(ctx, rec)
        return rec
    why = f"job {rec.head} ended {getattr(st, 'state', '?')} and the lane had not closed"
    if (session / "ABORTED.md").exists():
        why += ": ABORTED.md (" + (session / "ABORTED.md").read_text(errors="replace").strip()[:200] + ")"
    rec.failures.append({"ts": ctx.clock(), "job": rec.head, "node": rec.node, "why": why})
    if rec.node and rec.node not in rec.exclude:
        rec.exclude.append(rec.node)
    _to(ctx, rec, "FAILED", kind="lane.failed", why=why, node=rec.node)
    if mode == "run":
        retry(ctx, rec)
    return rec


def retry(ctx, rec: Lane) -> Lane:
    """A FAILED lane: resubmitted once without its node; a second failure is BROKEN."""
    if rec.state != "FAILED":
        return rec
    if rec.resubmits >= 1:
        _to(ctx, rec, "BROKEN", kind="lane.broken", why=rec.failures[-1]["why"] if rec.failures else "")
        return rec
    return resubmit(ctx, rec)


def resubmit(ctx, rec: Lane, *, exclude_node: str | None = None, by: str = "controller") -> Lane:
    if rec.state in FINAL:
        raise ValueError(f"{rec.lane} is {rec.state}: a closed lane is never submitted again")
    if exclude_node and exclude_node not in rec.exclude:
        rec.exclude.append(exclude_node)
    rec.resubmits += 1
    aborted = Path(rec.session_root) / "ABORTED.md"
    if aborted.exists():
        aborted.rename(aborted.with_name(f"ABORTED.{rec.resubmits}.md"))
    save(ctx, rec, "lane.resubmit", exclude=rec.exclude, resubmits=rec.resubmits, by=by)
    rec.state = "FAILED" if rec.state == "BROKEN" else rec.state
    return submit(ctx, rec)


def stop(ctx, rec: Lane, why: str) -> None:
    """Ask the lane's driver to stop at its next iteration boundary, and its chain to end."""
    session = Path(rec.session_root)
    if not rec.session_root or not session.exists():
        return
    (session / "chain").mkdir(exist_ok=True)
    for f in (session / "STOP", session / "chain" / "CHAIN_STOP"):
        if not f.exists():
            atomic_write_text(f, f"parallel: {why}\n")
    if rec.state == "PENDING" and rec.head:
        ctx.cluster.cancel(rec.head)
    ctx.store.event("lane.stop_requested", lane=rec.lane, why=why)


def operator_close(ctx, rec: Lane, as_: str, why: str) -> Lane:
    """close-lane: the operator ends a lane as killed; nothing of it is integrated."""
    rec.closing = {"as": as_, "why": why, "ts": ctx.clock()}
    save(ctx, rec, "lane.operator_close", status=as_, why=why)
    if rec.state in ACTIVE:
        stop(ctx, rec, f"the operator closed it ({as_}): {why}")
        return rec
    return close(ctx, rec)


# -- closure ---------------------------------------------------------------------------


def strongest_intent(session: Path) -> str:
    """The strictest intent among the lane's KEEPs, booked ones first; WIN when there is none."""
    from arbor.measure.intents import resolve
    lessons = (read_json(session / "session.json", {}) or {}).get("lessons") or []
    keeps = [x for x in lessons if x.get("decision") == "KEEP"]
    rows = [x for x in keeps if x.get("booked") is True] or keeps
    names = [resolve(x.get("intent") or x.get("claim_class")).name for x in rows]
    return max(names, key=lambda n: INTENT_RANK.get(n, 3)) if names else "WIN"


def freeze(ctx, rec: Lane) -> dict:
    """final/: the lane's landed work as it closed, with sha256s, and the commit it makes on its start."""
    session, fin = Path(rec.session_root), lane_dir(ctx, rec) / "final"
    fin.mkdir(parents=True, exist_ok=True)
    closed = read_json(session / directions.LANE_FILE, {}) or {}
    src = session / "landed.patch"
    text = src.read_text(errors="replace") if src.is_file() else ""
    atomic_write_text(fin / "landed.patch", text)
    wcfg = (workspace_from_session(read_json(session / "workspace_config.json"))
            or flags_to_workspace(rec.start["flags"]))
    atomic_write_json(fin / "workspace_config.json", wcfg)
    if (session / "landed_aiter.py").is_file():
        shutil.copy2(session / "landed_aiter.py", fin / "landed_aiter.py")
    if closed:
        atomic_write_json(fin / directions.LANE_FILE, closed)
    patch, dropped = strip_artifacts(text)
    commit, why = None, ""
    if patch.strip():
        r = ctx.trees.apply_patch(rec.start["sha"], patch, directory="vllm", message=f"{rec.lane}: final")
        if isinstance(r, Conflict):
            why = f"landed.patch does not apply to the lane's own start tree ({', '.join(r.paths)})"
        else:
            commit = r
    flags = flags_from_workspace(wcfg)
    start_aiter = ctx.store.base_dir(rec.start["base"]) / "landed_aiter.py"
    aiter_changed = _sha(fin / "landed_aiter.py") not in (None, _sha(start_aiter))
    kept = not why and (commit is not None or flags != rec.start["flags"])
    if rec.closing:
        kept, why = False, f"closed by the operator as {rec.closing['as']}: {rec.closing['why']}"
    elif aiter_changed and kept:
        # TODO(phase 4): AITER edits in the tree store; until then an addition that needs them is not carried.
        kept, why = False, "its AITER edits (landed_aiter.py) are not carried by the tree store (phase 4)"
    elif not kept and not why:
        why = "closed with no KEEP" if not closed.get("keeps") else "its KEEPs left the tree and flags unchanged"
    manifest = {"lane": rec.lane, "direction": rec.direction, "kind": rec.kind, "start_sha": rec.start["sha"],
                "commit": commit, "flags": flags, "kept": kept, "why": why, "keeps": closed.get("keeps") or [],
                "counts": closed.get("counts") or {}, "intent": strongest_intent(session),
                "aiter_changed": aiter_changed, "bytecode_dropped": len(dropped), "ts": ctx.clock(),
                "files": {p.name: _sha(p) for p in sorted(fin.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    atomic_write_json(fin / "manifest.json", manifest)
    if rec.kind == "upstream":
        atomic_write_json(fin / "upstream.json", _upstream(ctx, rec, session, text))
    return manifest


def _upstream(ctx, rec: Lane, session: Path, patch: str) -> dict:
    """What an upstream lane ported: the PR, the commit PORT named, the files, the image's digest."""
    from arbor import stages
    d = read_json(lane_dir(ctx, rec) / "direction.json", {}) or {}
    st = stages.load(session) or {"items": {}}
    port = next((a for iid, it in st["items"].items() if it.get("stage") == "PORT"
                 for a in reversed(it.get("attempts") or []) if a.get("harness") == "pass"), {})
    m = port.get("measured") or {}
    return {**(d.get("upstream") or {}), "ported_commit": m.get("upstream_commit"),
            "image_digest": m.get("image_digest"), "image_tag": ctx.cfg.template().get("image_tag"),
            "files": [ln.split(" b/", 1)[-1] for ln in patch.splitlines() if ln.startswith("diff --git ")]}


def _finish_clocks(ctx, rec: Lane) -> None:
    """sacct's intervals of every chain job, when it answers: the lane's exact time on nodes."""
    jobs = chain_jobs(rec)
    ivs = ctx.cluster.intervals(jobs) if hasattr(ctx.cluster, "intervals") and jobs else None
    c = rec.clocks
    c["chunks"] = len(jobs)
    spans = sorted((s, e) for v in (ivs or {}).values() for s, e, _ in v)
    if spans:
        union, cur = 0.0, None
        for s, e in spans:
            if cur is None or s > cur[1]:
                union += (cur[1] - cur[0]) if cur else 0.0
                cur = [s, e]
            else:
                cur[1] = max(cur[1], e)
        union += cur[1] - cur[0]
        c["on_node_s"] = round(union, 1)
        c["node_s"] = round(sum(e - s for s, e in spans) * rec.nodes, 1)
        c["source"] = "sacct"
    else:
        c["node_s"] = round(c.get("on_node_s", 0.0) * rec.nodes, 1)
        c["source"] = "ticks"


def close(ctx, rec: Lane) -> Lane:
    rec.final = freeze(ctx, rec) if rec.start else {"kept": False, "why": "never started", "keeps": []}
    closed = read_json(Path(rec.session_root) / directions.LANE_FILE, {}) if rec.session_root else {}
    items = [{k: i.get(k) for k in ("id", "stage", "status", "why", "closed_at", "wall_s")}
             for i in (closed or {}).get("items") or []]
    statuses = {i["status"] for i in items}
    status = (rec.closing["as"] if rec.closing else "done" if rec.final["kept"] else
              "killed" if statuses and statuses <= {"killed", "below_floor"} else "closed")
    rec.outcome = {"status": status, "counts": (closed or {}).get("counts") or {}, "items": items,
                   "keeps": rec.final.get("keeps") or [], "kept": rec.final["kept"], "why": rec.final.get("why", ""),
                   "proposals": (closed or {}).get("proposals") or [],
                   "iteration_wall_s": (closed or {}).get("iteration_wall_s")}
    rec.clocks["closed"] = ctx.clock()
    _to(ctx, rec, "CLOSED")
    _finish_clocks(ctx, rec)
    save(ctx, rec, "lane.closed", status=status, kept=rec.final["kept"], keeps=rec.outcome["keeps"])
    return rec
