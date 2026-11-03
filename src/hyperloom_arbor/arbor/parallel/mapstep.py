"""A round's map: the profile and the dossier's first half in parallel, then its second half, then the checks.

- **The profile** (`profile_round.sbatch`, a node): the round's base served
  once, its bar replays and a profile capture, into map/profile/attempt-K/
  ({bar.json, profile/, DONE}).
- **Dossier A** (`dossier_round.sh A`, a user unit on this host): a document
  campaign that reads the base tree, the last round's ROUND.md and every lane's
  session (`arbor.collect knowledge`), and writes 03.uK (the code) and 10.uK
  (the lessons). It needs no measurement, so it runs beside the profile.
- **Dossier B**, once both are DONE, in the same workspace and session with
  new directives: 06.uK (the bar), 07.uK (the profile) and 11.uK, ending in
  its `## Map` line.
- **The checks**: `check-dossier.py --round uK` on the round's files, the
  `## Map` line through `arbor.directions.parse_agent_lines` (which refuses
  cycles), every plan's ITEMS block, and the reopen rule: a direction that
  died earlier in this run (killed, below its floor, or dropped at an
  integration) comes back only with `reopens`.

Each job gets two attempts. A job that ends without its DONE marker, or a
dossier half whose files do not check, is tried once more (dossier B with the
checks' complaints in its directives); a second failure ends the map step, and
the run stops for the operator.

Round 1 can be seeded instead: `seed.map` names an existing 11.uK, `seed.overlay`
adds the lane fields ({id: {lane, plan, depends_on, nodes}}) and
`seed.bar_record` the bar, and no job runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from arbor import directions
from arbor.execute.workspace import HOST
from arbor.parallel import plans
from arbor.parallel.cluster import track_poll, track_submit
from arbor.parallel.store import atomic_write_json, read_json

MAX_ATTEMPTS = 2
PARTS = {"A": ("03", "10"), "B": ("03", "06", "07", "10", "11")}
CHECK_DOSSIER = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "dossier" / "check-dossier.py"
_UK = re.compile(r"(?<![a-z0-9])u(\d+)(?![0-9])")


def uk(cfg, n: int) -> int:
    """Round n's dossier update round: the seed map's K plus n - 1, else first_round plus n - 1."""
    if cfg.seed:
        m = _UK.search(Path(cfg.seed["map"]).name)
        if m:
            return int(m.group(1)) + n - 1
    return cfg.first_round + n - 1


def map_dir(store, n: int) -> Path:
    return store.round_dir(n) / "map"


def state(store, n: int) -> dict:
    return read_json(map_dir(store, n) / "map.json") or {"jobs": {}, "errors": []}


def _save(store, n: int, st: dict) -> None:
    atomic_write_json(map_dir(store, n) / "map.json", st)


# -- the step --------------------------------------------------------------------------


def step(ctx, n: int) -> dict | None:
    """Drive the map one tick; the result once there is one ({"ok", "entries", "where", "errors"})."""
    st = state(ctx.store, n)
    if st.get("result"):
        return st["result"]
    st["uk"] = uk(ctx.cfg, n)
    if n == 1 and ctx.cfg.seed:
        st["result"] = seeded(ctx, n)
    else:
        st["result"] = _jobs(ctx, n, st)
    _save(ctx.store, n, st)
    if st["result"]:
        ctx.store.event("map.result", round=n, ok=st["result"]["ok"], errors=st["result"].get("errors", [])[:5])
    return st["result"]


def _jobs(ctx, n: int, st: dict) -> dict | None:
    for job in ("profile", "A"):
        _drive(ctx, n, st, job)
    jobs = st["jobs"]
    if all(jobs.get(j, {}).get("state") == "DONE" for j in ("profile", "A")):
        _drive(ctx, n, st, "B")
    gave_up = [j for j, r in jobs.items() if r.get("state") == "GAVE_UP"]
    if gave_up:
        return {"ok": False, "errors": [f"no valid {j} after {MAX_ATTEMPTS} attempts: {jobs[j].get('why', '')}"
                                        for j in gave_up]}
    if jobs.get("B", {}).get("state") != "DONE":
        return None
    entries, errors, where = check_round(ctx, n, st["uk"])
    if not errors:
        errors = validate(ctx, n, entries, where)
    if not errors:
        return {"ok": True, "entries": entries, "where": str(where), "errors": [],
                "source": str(_map_file(where, st["uk"]))}
    st["errors"] = errors
    if jobs["B"]["attempt"] >= MAX_ATTEMPTS:
        return {"ok": False, "errors": [f"no valid map after {MAX_ATTEMPTS} attempts of dossier B"] + errors}
    ctx.store.event("map.invalid", round=n, attempt=jobs["B"]["attempt"], errors=errors[:5])
    _start(ctx, n, st, "B", jobs["B"]["attempt"] + 1)
    return None


def _done_marker(rec: dict) -> bool:
    return (Path(rec["dir"]) / "DONE").exists()


def _drive(ctx, n: int, st: dict, job: str) -> None:
    rec = st["jobs"].get(job)
    if rec is None:
        _start(ctx, n, st, job, 1)
        return
    if rec["state"] in ("DONE", "GAVE_UP"):
        return
    backend = ctx.cluster if job == "profile" else ctx.host
    if not _done_marker(rec):
        track_poll(backend, rec)
        if rec["state"] == "LOST":                 # sbatch never queued it: the same attempt again
            _start(ctx, n, st, job, rec["attempt"])
            return
        if rec["state"] in ("SUBMITTING", "PENDING", "RUNNING"):
            return
    why = "" if _done_marker(rec) else f"the job ended {rec['state']} without its DONE marker"
    if not why and job != "profile":
        why = _deliverables(ctx, n, st["uk"], job)
    if not why:
        rec["state"] = "DONE"
        ctx.store.event("map.job_done", round=n, job=job, attempt=rec["attempt"], node=rec.get("node"))
        return
    rec["why"] = why
    if rec["attempt"] >= MAX_ATTEMPTS:
        rec["state"] = "GAVE_UP"
        ctx.store.event("map.job_gave_up", round=n, job=job, why=why)
        return
    ctx.store.event("map.job_failed", round=n, job=job, attempt=rec["attempt"], why=why)
    _start(ctx, n, st, job, rec["attempt"] + 1, exclude=[rec["node"]] if rec.get("node") else [])


def _start(ctx, n: int, st: dict, job: str, attempt: int, exclude: list[str] = ()) -> None:
    cfg, md = ctx.cfg, map_dir(ctx.store, n)
    name = f"par-{cfg.run_id}-r{n:02d}-{'prof' if job == 'profile' else 'dos' + job}-a{attempt}"
    if job == "profile":
        d = md / "profile" / f"attempt-{attempt}"
    else:
        d = md / "dossier" / "jobs" / job / f"attempt-{attempt}"
    d.mkdir(parents=True, exist_ok=True)
    prev = st["jobs"].get(job) or {}
    rec = {"name": name, "state": "SUBMITTING", "attempt": attempt, "dir": str(d),
           "exclude": sorted(set(prev.get("exclude", [])) | set(exclude))}
    st["jobs"][job] = rec
    log = f"{cfg.cluster.log_dir}/{name}.out" if cfg.cluster.log_dir else str(d / "job.log")
    if job == "profile":
        env, script, backend = _profile_env(ctx, n, d), cfg.path(cfg.cluster.profile_script), ctx.cluster
    else:
        env, script, backend = _dossier_env(ctx, n, st, job, d), cfg.path(cfg.cluster.dossier_script), ctx.host
    _save(ctx.store, n, st)                         # the name, before the job exists
    track_submit(backend, rec, str(script), env=env, exclude=rec["exclude"], time_limit=cfg.cluster.time_limit,
                 output=log)
    if rec["state"] == "FAILED":
        rec["state"] = "GAVE_UP" if attempt >= MAX_ATTEMPTS else rec["state"]
    _save(ctx.store, n, st)
    ctx.store.event("map.job_submitted", round=n, job=job, attempt=attempt, name=name, id=rec.get("id"),
                    state=rec["state"])


def _base_files(ctx) -> dict:
    base = ctx.store.current_base()
    d = ctx.store.base_dir(base["index"])
    aiter = d / "landed_aiter.py"
    return {"PAR_VMU": str(d / "base"), "PAR_FLAGS": str(d / "workspace_config.json"),
            "PAR_AITER": str(aiter) if aiter.is_file() else "", "PAR_BASE": f"base_{base['index']:03d}"}


def _profile_env(ctx, n: int, d: Path) -> dict:
    cfg = ctx.cfg
    return {"PAR_OUT": str(d), "PAR_TEMPLATE": str(cfg.template_config), "PAR_TAG": f"{cfg.run_id}-r{n:02d}",
            "CAMPAIGN_ARBOR": str(cfg.arbor_root), **_base_files(ctx)}


# -- the dossier campaign --------------------------------------------------------------


def dossier_paths(store, n: int) -> dict:
    d = map_dir(store, n) / "dossier"
    return {"dir": d, "ws": d / "ws", "session": d / "session", "config": d / "config.json",
            "reports": d / "ws" / "reports"}


def _killed_text(ctx, n: int) -> str:
    dead = closed_before(ctx.store, n)
    return ", ".join(f"{k} ({v})" for k, v in sorted(dead.items())) or "none"


def dossier_directives(ctx, n: int, part: str, k: int, errors: list[str]) -> list[str]:
    run = ctx.cfg.run_id
    out = [f"PARALLEL DOSSIER, run {run}, round {n}: the dossier skill's update round u{k} for the round's "
           f"base tree (inputs/tree/). Write into reports/, as NN.u{k}-<part>.md; inputs/ is read-only."]
    if part == "A":
        out += [f"PART A (a profile of this tree runs on a node meanwhile): write reports/03.u{k}-code.md, the "
                f"code read again on inputs/tree/, and reports/10.u{k}-lessons.md, from every lane's "
                "iterations (inputs/lanes/<lane>/: outcomes, findings, artifacts) and the last round's "
                "inputs/ROUND.md, marking each kill reopenable or not, with what it rests on. Nothing else "
                "yet: part B writes the rest once the profile is in."]
    else:
        out += [f"PART B: the profile is in inputs/profile/ and the bar in inputs/bar.json. Write "
                f"reports/06.u{k}-baseline.md (the bar), reports/07.u{k}-profile.md and "
                f"reports/11.u{k}-directions.md, ranked from 03, 06, 07 and 10, ending in its ## Map line "
                "with the lane fields (lane, plan, depends_on, nodes, upstream). Each rewrite's plan is "
                f"reports/11.u{k}-plan-<region>.md with its ## Items block.",
                f"THE REOPEN RULE: a direction that died earlier in this run comes back only with "
                f"`reopens` {{round, why}}, on 10.u{k}'s evidence. Died so far: {_killed_text(ctx, n)}.",
                f"CHECK BEFORE DONE: python3 .claude/skills/dossier/check-dossier.py reports --round u{k} "
                "--no-index, and python3 -m arbor.directions check reports/11.u"
                f"{k}-directions.md --dossier reports."]
        if errors:
            out.append("THE LAST ATTEMPT'S MAP WAS REFUSED: " + "; ".join(errors[:8]))
    return out


def dossier_config(ctx, n: int, part: str, k: int, errors: list[str]) -> dict:
    cfg, p = ctx.cfg, dossier_paths(ctx.store, n)
    pack = str(cfg.dossier.get("pack") or "parallel-dossier")
    v = cfg.arbor_root / "packs" / pack / "validator"
    return {"pack": pack, "identity": f"parallel-dossier:{cfg.run_id}:r{n:02d}:u{k}",
            "goal": f"The dossier's update round u{k} for round {n} of parallel run {cfg.run_id}: the map "
                    "the round's lanes run from.",
            "docker_container": HOST, "model_name": "none", "workspace_root": str(p["ws"]),
            "session_root": str(p["session"]), "bench_script": str(v / "bench_dossier.sh"),
            "correctness_script": str(v / "correctness_dossier.sh"), "bench_timeout": 300,
            "baseline_record": str(p["dir"] / "baseline.json"),
            "directives": dossier_directives(ctx, n, part, k, errors)}


def _lane_sessions(ctx, n: int) -> list[str]:
    from arbor.parallel import lanes
    return [r.session_root for m in range(1, n) for r in lanes.load_all(ctx.store, m) if r.session_root]


def _dossier_env(ctx, n: int, st: dict, part: str, d: Path) -> dict:
    cfg, p = ctx.cfg, dossier_paths(ctx.store, n)
    k = st["uk"]
    atomic_write_json(p["config"], dossier_config(ctx, n, part, k, st.get("errors") or []))
    prof = st["jobs"].get("profile") or {}
    prev = ctx.store.round_dir(n - 1) / "ROUND.md"
    request = {"part": part, "round": n, "uk": f"u{k}", "run": cfg.run_id, "attempt_dir": str(d),
               "ws": str(p["ws"]), "session": str(p["session"]), "config": str(p["config"]),
               "dossier_dir": str(cfg.dossier_dir), "trees": str(ctx.store.trees),
               "base_sha": ctx.store.current_base()["sha"], "prev_round": str(prev) if prev.is_file() else None,
               "lane_sessions": _lane_sessions(ctx, n), "profile": prof.get("dir") if part == "B" else None,
               "deliverables": [f"{x}.u{k}" for x in PARTS[part]], "iterations": cfg.dossier_iters(part),
               "errors": st.get("errors") or []}
    atomic_write_json(d / "request.json", request)
    return {"PAR_REQUEST": str(d / "request.json"), "ARBOR_ROOT": str(cfg.arbor_root),
            "PYTHONPATH": str(cfg.arbor_root)}


def check_dossier(reports: Path, k: int) -> list[dict]:
    """check-dossier.py's rows for round uK (the controller's own copy of the skill)."""
    p = subprocess.run([sys.executable, str(CHECK_DOSSIER), str(reports), "--round", f"u{k}", "--no-index",
                        "--json"], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(p.stdout)["files"]
    except (ValueError, KeyError):
        return [{"part": "check-dossier", "ok": False, "why": (p.stderr or p.stdout).strip()[:300]}]


def _deliverables(ctx, n: int, k: int, part: str) -> str:
    """Why a dossier half's files do not check ("" when they do)."""
    rows = check_dossier(dossier_paths(ctx.store, n)["reports"], k)
    want = PARTS[part]
    bad = [f"{r['part']}: {r.get('why') or 'too thin'}" for r in rows
           if not r.get("ok") and (part == "B" or str(r.get("part", "")).split(".")[0] in want)]
    return "; ".join(bad[:6])


def _map_file(where: Path, k: int) -> Path:
    named = where / f"11.u{k}-directions.md"
    if named.is_file():
        return named
    hits = [h for h in sorted(where.glob(f"11.u{k}-*.md")) if "-plan-" not in h.name]
    return hits[0] if hits else named


def check_round(ctx, n: int, k: int) -> tuple[list[dict], list[str], Path]:
    where = dossier_paths(ctx.store, n)["reports"]
    errors = [f"{r['part']}: {r.get('why') or 'too thin'}" for r in check_dossier(where, k) if not r.get("ok")]
    entries, more = parse_map(_map_file(where, k))
    return entries, errors + more, where


def parse_map(path: Path, overlay: dict | None = None) -> tuple[list[dict], list[str]]:
    """The 11.uK's `## Map` line, with an operator overlay of lane fields; (entries, errors).
    An overlay key that starts with `_` is a note."""
    if not path.is_file():
        return [], [f"no map: {path.name} does not exist"]
    p = directions.parse_agent_lines(path.read_text(errors="replace"))
    if p.new_map is None:
        return [], [f"{path.name}: " + ("; ".join(p.errors[:3]) or "no DIRECTIONS line (## Map)")]
    overlay = {k: v for k, v in (overlay or {}).items() if not k.startswith("_")}
    if not overlay:
        return p.new_map, []
    unknown = sorted(set(overlay) - {e["id"] for e in p.new_map})
    if unknown:
        return [], [f"the overlay names directions not on the map: {', '.join(unknown)}"]
    merged = [{**e, **{k: v for k, v in (overlay.get(e["id"]) or {}).items()
                       if k in directions.LANE_KEYS + ("rank", "status", "why")}} for e in p.new_map]
    q = directions.parse_agent_lines("DIRECTIONS: " + json.dumps(merged))
    if q.new_map is None:
        return [], ["the map with the overlay: " + "; ".join(q.errors[:3])]
    return q.new_map, []


def seeded_preview(ctx, n: int = 1) -> dict:
    """Round 1's seeded map, read and checked; writes nothing (`tick --dry-run` shows it)."""
    cfg = ctx.cfg
    src = cfg.path(cfg.seed["map"])
    overlay = read_json(cfg.path(cfg.seed["overlay"])) if cfg.seed.get("overlay") else None
    if cfg.seed.get("overlay") and not isinstance(overlay, dict):
        return {"ok": False, "entries": [], "errors": [f"seed.overlay {cfg.seed['overlay']} is not a JSON object"]}
    entries, errors = parse_map(src, overlay)
    errors = errors or validate(ctx, n, entries, src.parent)
    return {"ok": not errors, "entries": entries, "where": str(src.parent), "errors": errors,
            "source": str(src), "seeded": True}


def seeded(ctx, n: int) -> dict:
    r = seeded_preview(ctx, n)
    if ctx.cfg.seed.get("bar_record") and r["ok"]:
        md = map_dir(ctx.store, n)
        md.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ctx.cfg.path(ctx.cfg.seed["bar_record"]), md / "bar.json")
    return r


# -- the checks ------------------------------------------------------------------------


def closed_before(store, n: int) -> dict[str, str]:
    """Directions that died in rounds before n: killed or below the floor on a map or in a lane,
    or dropped or conflicting at an integration. {id: why}."""
    dead: dict[str, str] = {}
    for m in range(1, n):
        doc = read_json(store.round_dir(m) / "directions.json", {}) or {}
        for d in doc.get("directions") or []:
            st = (d.get("outcome") or {}).get("status") or d.get("status")
            if st in ("killed", "below_floor"):
                dead[d["id"]] = f"r{m:02d} {st}"
        integ = read_json(store.round_dir(m) / "integrate" / "integration.json", {}) or {}
        for a in integ.get("additions") or []:
            if a.get("state") in ("DROPPED", "CONFLICT"):
                dead[a["direction"]] = f"r{m:02d} {a['state'].lower()} at integration"
    return dead


def validate(ctx, n: int, entries: list[dict], where: Path) -> list[str]:
    """Every plan's ITEMS, and the reopen rule. (The map's own syntax and cycles: parse_map.)"""
    errors = []
    for e in entries:
        if e.get("plan"):
            try:
                plans.check(plans.resolve(e["plan"], where))
            except plans.PlanError as exc:
                errors.append(f"{e['id']}: {exc}")
    dead = closed_before(ctx.store, n)
    for e in entries:
        if e["id"] in dead and e.get("status") == "open" and not e.get("reopens"):
            errors.append(f"{e['id']} died earlier in this run ({dead[e['id']]}): it comes back only with "
                          "reopens {round, why}")
    return errors
