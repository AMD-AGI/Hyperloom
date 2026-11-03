"""What each kind of lane is: its items, its nodes, its config overrides and its directives.

Four kinds (arbor/directions.py `LANES`), each one Arbor campaign on one node:

- increment: one direction worked to a verdict. Its map is the direction
  itself, closed done by a KEEP that cites it, or killed or below_floor on
  evidence; a plan with an ITEMS block makes those its items instead.
- kernel-plan: a rewrite's plan, stage by stage (S1-S5, arbor/stages.py),
  with the kernel level only and ruppell built (PROFILES=1).
- architectural: a layout change judged at several operating points. ARM
  measures one layout at every point of the plan's table on this node (a
  same-node multi-point A/B); FRONTIER lands the layout that dominates it
  (no point worse past noise, one or more better past the threshold) as a
  FIX KEEP. A trade across bands is recorded for the operator, and its tree
  is reverted. Multi-node points, and integrating at several points, are
  phase 4: see the TODOs below.
- upstream: a port of work done elsewhere (a PR at a commit): PORT, CORRECT,
  AB. It runs on the run's image; PORT names the commit it ported and the
  image's digest, which the lane's closure records in upstream.json.

Nothing here touches a pack: a lane's config is the template campaign's with
overrides, and its pack is its own worktree's.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from arbor import directions, stages

UPSTREAM_ITEMS = [
    {"id": "port", "stage": "PORT", "title": "the upstream change in the tree, its commit named"},
    {"id": "correct", "stage": "CORRECT", "after": ["port"], "title": "matches the path it replaces"},
    {"id": "ab", "stage": "AB", "after": ["correct"], "title": "the campaign's own A/B"},
]
PLAN_KINDS = ("kernel-plan", "architectural")


def nodes(kind: str, d: dict) -> int:
    """The nodes a lane holds against the cap. Only an architectural plan asks for more than one."""
    return max(1, int(d.get("nodes") or 1)) if kind == "architectural" else 1


def needs_plan(kind: str) -> bool:
    return kind in PLAN_KINDS


def default_items(kind: str) -> list[dict] | None:
    """Items a lane runs when its direction names no plan with an ITEMS block."""
    return [dict(i) for i in UPSTREAM_ITEMS] if kind == "upstream" else None


def env(kind: str) -> dict:
    """What the launcher is told besides the common variables."""
    return {"PROFILES": "1"} if kind == "kernel-plan" else {}


def overrides(kind: str, run_kind_cfg: dict, template: dict) -> dict:
    """Config keys this kind sets on top of the common ones."""
    out: dict = {}
    if kind == "kernel-plan":
        out["space"] = {"levels": ["kernel"], **(run_kind_cfg.get("space") or {})}
    elif run_kind_cfg.get("space"):
        out["space"] = dict(run_kind_cfg["space"])
    if run_kind_cfg.get("policy"):
        out["policy"] = {**(template.get("policy") or {}), **run_kind_cfg["policy"]}
    if kind == "upstream" and template.get("image_tag"):
        out["image_tag"] = template["image_tag"]           # the run's image, not the PR's
    return out


# -- the lane's map ------------------------------------------------------------------


def items_plan(lane_dir: Path, items: list[dict], direction: str) -> Path:
    """Default items written as a plan file, so arbor/stages.py pins and judges them as any plan's."""
    p = Path(lane_dir) / "items_plan.md"
    if not p.exists():
        p.write_text(f"# {direction}: the lane's items\n\nWritten by arbor.parallel for a lane whose "
                     f"direction names no plan.\n\n## Items\n\nITEMS: {json.dumps(items)}\n")
    return p


def seed(*, kind: str, config_path: Path, session: Path, direction: dict, plan: Path | None, run: str,
         round_no: int, lane_id: str) -> str:
    """The lane's map in its session, before its first chunk; idempotent. Returns how it was seeded."""
    session = Path(session)
    if (session / directions.FILE).exists():
        return "already"
    did = direction["id"]
    if plan is not None and stages._blocks(plan.read_text(errors="replace")):
        stages.seed(config_path, did, plan, session, kind, run, round_no, lane_id)
        return "plan"
    _seed_single(session, did, direction.get("title") or did, run, round_no, lane_id, plan)
    return "single"


def _seed_single(session: Path, did: str, title: str, run: str, round_no: int, lane_id: str,
                 plan: Path | None) -> None:
    """An increment's map: the direction as its one item, closed by DIRECTION with evidence."""
    session.mkdir(parents=True, exist_ok=True)
    now = time.time()
    doc = directions.new_doc()
    doc.update(scope="lane", phase="active", map_round=round_no, round_started=1, round_started_ts=now,
               phase_started_ts=now,
               lane={"run": run, "round": round_no, "lane": lane_id, "direction": did, "kind": "increment",
                     "plan": None, "plan_sha256": None, "source": str(plan) if plan else None},
               directions=[{"id": did, "kind": "increment", "title": title[:160], "status": "open", "why": "",
                            "added_at": 1, "closed_at": None, "worked": [], "keeps": [], "flagged": False}])
    directions.save(session, doc)


# -- directives ----------------------------------------------------------------------


def directives(kind: str, *, lane: str, run: str, round_no: int, d: dict, plan: str | None,
               others: list[str], deps: list[str], base: str, image_tag: str = "") -> list[str]:
    """The lane's standing instructions: who it is, what it may touch, and what its tree holds."""
    did = d["id"]
    out = [
        f"PARALLEL LANE {lane} (run {run}, round {round_no}): your direction is {did}, "
        f"\"{(d.get('title') or '')[:120]}\", a {kind} lane. Your map is that direction's items "
        "(the program lists them); the lane ends when every item has a verdict with evidence, "
        "and nothing reopens inside the run.",
    ]
    if plan:
        out.append(f"THE PLAN is pinned at {plan}. Read it first, all of it; it is the lane's work.")
    out.append("CITE THE ITEM in every DONE line, first (`DONE: <item id> <what changed>`). A close cites its "
               "evidence: the iterations, ledger ids and artifacts the harness checks.")
    if others:
        out.append(f"OTHER DIRECTIONS ARE OTHER LANES': {', '.join(others)} run beside you this round. Do not "
                   "work them. A new direction you find is a proposal for the next round: a DIRECTION line "
                   "with its kind and title.")
    out.append(f"YOUR TREE is the round's base ({base})" + ("; " + "; ".join(deps) if deps else "") + ". "
               "Your bar is this node's, rebased every chunk: your gain is a screen, and only the round's "
               "integration A/B, on one node, backs a claim.")
    out += _kind_directives(kind, d, image_tag)
    return out


def _kind_directives(kind: str, d: dict, image_tag: str) -> list[str]:
    did = d["id"]
    if kind == "increment":
        return [f"INCREMENT: close {did} done with a KEEP whose DONE line cites it, or killed or below_floor "
                "with its evidence (`DIRECTION: {\"id\": ..., \"status\": ..., \"why\": ..., \"evidence\": "
                "{\"iterations\": [N]}}`)."]
    if kind == "kernel-plan":
        return ["KERNEL PLAN: work its items stage by stage and declare each with a STAGE line. S1-S3 are "
                "LEARN, S4 FIX (behind a switch that defaults off), S5 the WIN that turns it on. A stop rule "
                "that fires ends the plan, and that is a result, not a failure."]
    if kind == "architectural":
        out = ["ARCHITECTURAL: each ARM iteration measures one layout at every point of the plan's table on "
               "this node, B and the layout interleaved (a same-node multi-point A/B), and reports the "
               "per-point table in measured.points. A layout no worse past noise at any point and better "
               "past the threshold at one or more lands as a FIX KEEP (FRONTIER). A trade across bands is a "
               "LEDGER FACT for the operator, and its tree is reverted."]
        if int(d.get("nodes") or 1) > 1:
            # TODO(phase 4): a multi-node lane holds d["nodes"] nodes and measures across them.
            out.append(f"This plan asks for {d['nodes']} nodes; the lane runs on one (multi-node lanes are "
                       "phase 4). Measure the points one node can hold; record the rest as UNKNOWN.")
        return out
    up = d.get("upstream") or {}
    return [f"UPSTREAM PORT of {up.get('repo')} PR {up.get('pr')} at {up.get('commit')}: PORT lands it with "
            "measured.upstream_commit (the commit ported) and measured.image_digest (`docker inspect` of the "
            "image you run on); then CORRECT, then AB. The image is the run's"
            + (f" ({image_tag})" if image_tag else "")
            + (f", not the PR's {up['image']}" if up.get("image") else "") + "."]


# -- architectural tables ------------------------------------------------------------


def classify_points(points: list[dict], noise_pct: float, threshold_pct: float) -> str:
    """dominates | trade | worse | neutral, from a per-point table of `delta_pct` (better is positive).

    TODO(phase 4): integration measures only the single-node default point; a
    dominating layout's other points are not re-measured there.
    """
    ds = [float(p["delta_pct"]) for p in points if isinstance(p.get("delta_pct"), (int, float))]
    if not ds:
        return "neutral"
    worse = any(x < -noise_pct for x in ds)
    better = any(x > threshold_pct for x in ds)
    return "trade" if worse and better else "worse" if worse else "dominates" if better else "neutral"
