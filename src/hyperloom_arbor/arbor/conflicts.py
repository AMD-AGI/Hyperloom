"""Two loaded skills that write the same thing, or overlap: a conflict, and what a person, or an agent, decides.

**What a conflict is** (``detect``), between two runtime skills a campaign loads:

- **the same file:** their ``produces`` share a file, or a folder one declares (a pattern ending in ``/``) holds a file
  of the other's;
- **declared:** one names the other in ``metadata.arbor.overlaps``, for overlap that files cannot show.

**What happens,** at the start of every session, before ``init`` (``gate``, from the driver):

- **Standalone, by default, a pause.** Arbor records each conflict in ``<session>/conflicts.json`` (``pending``) and
  ``CONFLICTS.md``, and waits, as the review gate does. It records them before the review gate, so a reviewer can
  decide them in the same pause, and waits after it, on the reviewed pack.
- **The decisions,** one per conflict (``python3 -m arbor conflicts --session D show|pick|merge``):
  - ``pick ID SKILL``: SKILL stays and the other is dropped, for this session. To merge by hand, edit one skill in the
    tree, then pick it: the skills are read again after the decision, and the record names the skills that changed.
  - ``merge ID [--keep SKILL]``: an agent merges the two into one skill under SKILL's name (by default the one loaded
    first), written into the session. The merge must check clean, keep that skill's name, modes and chapter, and
    produce every file both did; one that does not is pending again.
- **Auto-merge** (``run --skill-conflicts merge``, ``ARBOR_SKILL_CONFLICTS=merge``, the config's
  ``"skill_conflicts": "merge"``): every conflict is merged by an agent, with no pause.
- **Under a host,** a session never waits for a person: it merges when auto-merge is set, and refuses to start
  otherwise, naming the conflicts and the two ways out.
- **The journal** gets each decision (kind ``conflict``).

A decision lasts for its session (``applied`` is how the lifecycle reads it). To make one last, drop a skill from the
pack's ``lifecycle.load``, or move a merged skill into the tree.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from arbor import lifecycle, skillcheck, skills

CONFLICTS_FILE = "conflicts.json"
CONFLICTS_DOC = "CONFLICTS.md"
ENV = "ARBOR_SKILL_CONFLICTS"
PAUSE, MERGE = "pause", "merge"
POLICIES = (PAUSE, MERGE)
#: A conflict's states: waiting for a decision; one side picked; a merge asked for; a merge done.
PENDING, PICKED, TO_MERGE, MERGED = "pending", "picked", "merge", "merged"
#: Where a session keeps its merged skills, and the plugin that carries a merged flat skill.
MERGED_DIR = Path("skills") / "merged"
MERGED_FAMILY = "merged"
POLL_S = 15.0
REMIND_S = 600.0


def log(msg: str) -> None:
    print(f"[arbor] conflicts: {msg}", flush=True)


# ── what a conflict is ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Conflict:
    """Two loaded skills (in load order) that conflict: the files both write, and whether one declares the overlap."""

    skills: tuple[str, str]
    files: tuple[str, ...]
    declared: bool

    @property
    def key(self) -> str:
        return "|".join(sorted(self.skills))

    def why(self) -> str:
        parts = [f"both write {', '.join(self.files)}"] if self.files else []
        if self.declared:
            parts.append("one declares that it overlaps the other (metadata.arbor.overlaps)")
        return "; ".join(parts)


def overlap(first: str, second: str) -> bool:
    """Whether two ``produces`` patterns name the same file, or a folder (ending in ``/``) that holds the other."""

    a, b = first.rstrip("/"), second.rstrip("/")
    return a == b or (first.endswith("/") and b.startswith(a + "/")) or (second.endswith("/") and a.startswith(b + "/"))


def detect(load, found: dict) -> list[Conflict]:
    """The conflicts among the runtime skills ``load`` names (``found`` is ``skillcheck.find``'s), in load order."""

    names = [name for name in dict.fromkeys(load) if name in found and not found[name].development]
    out = []
    for at, first in enumerate(names):
        for second in names[at + 1:]:
            mine, theirs = found[first].arbor, found[second].arbor
            files = [a if a.rstrip("/") == b.rstrip("/") else f"{a} and {b}"
                     for a in mine.get("produces") or [] for b in theirs.get("produces") or [] if overlap(a, b)]
            declared = second in (mine.get("overlaps") or []) or first in (theirs.get("overlaps") or [])
            if files or declared:
                out.append(Conflict((first, second), tuple(dict.fromkeys(files)), declared))
    return out


# ── the switch ─────────────────────────────────────────────────────────────────

def policy(config: dict, cli: str | None = None, environ=None) -> tuple[str, str]:
    """``merge`` or ``pause``, and what said so: the run's flag, ``ARBOR_SKILL_CONFLICTS``, the config's
    ``skill_conflicts``, in that order; ``pause`` (``default``) when none says. A value that is not one of them is
    ``pause``, and says so."""

    env = (os.environ if environ is None else environ).get(ENV, "").strip().lower()
    for value, source in ((cli, "--skill-conflicts"), (env, ENV), ((config or {}).get("skill_conflicts"), "config")):
        if value:
            value = str(value).strip().lower()
            return (value, source) if value in POLICIES else (PAUSE, f"{source}={value} (unknown; pausing)")
    return PAUSE, "default"


# ── the record ─────────────────────────────────────────────────────────────────

def load(session_dir) -> dict | None:
    try:
        doc = json.loads((Path(session_dir) / CONFLICTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _save(session_dir, doc: dict) -> None:
    path = Path(session_dir) / CONFLICTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def records(session_dir) -> list[dict]:
    return list((load(session_dir) or {}).get("conflicts") or [])


def pending(session_dir) -> list[dict]:
    return [rec for rec in records(session_dir) if rec.get("status") == PENDING]


def _digests(found: dict, names) -> dict[str, str]:
    return {name: skills.digest(found[name].path) for name in names if name in found}


def request(session_dir, conflicts: list[Conflict], found: dict) -> list[dict]:
    """Record ``conflicts`` in ``conflicts.json``, keeping the decision a pair already has, and write
    ``CONFLICTS.md``; the records still pending. A pair that no longer conflicts drops out of the record."""

    old = {rec.get("key"): rec for rec in records(session_dir)}
    recs = []
    for number, conflict in enumerate(conflicts, 1):
        rec = dict(old.get(conflict.key) or {"status": PENDING, "asked_at": _now()})
        rec.update(id=f"c{number}", key=conflict.key, skills=list(conflict.skills), files=list(conflict.files),
                   declared=conflict.declared, why=conflict.why())
        if rec["status"] == PENDING:
            rec["digests"] = _digests(found, conflict.skills)
        recs.append(rec)
    if recs or old:
        _save(session_dir, {"conflicts": recs, "updated_at": _now()})
        _document(session_dir, recs, found)
    return [rec for rec in recs if rec["status"] == PENDING]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def commands(session_dir) -> str:
    s = str(session_dir)
    return (f"    python3 -m arbor conflicts --session {s} show\n"
            f"    python3 -m arbor conflicts --session {s} pick <id> <skill>      # one side, for this session\n"
            f"    python3 -m arbor conflicts --session {s} merge <id> [--keep <skill>]   # an agent merges them")


def _document(session_dir, recs: list[dict], found: dict) -> Path:
    lines = ["# Skill conflicts", "",
             "Two skills this campaign loads write the same thing, or one says it overlaps the other. Decide each, "
             "then the session goes on:", "",
             "- **pick** one side: the other is dropped for this session. To merge by hand, edit one skill first, "
             "then pick it.",
             "- **merge:** an agent merges them into one skill, under the kept one's name, for this session.", "",
             "To make a decision last, drop a skill from the pack's `lifecycle.load`, or move a merged skill into "
             "the tree.", ""]
    for rec in recs:
        lines += [f"## {rec['id']}: `{rec['skills'][0]}` and `{rec['skills'][1]}` ({rec['status']})", "",
                  f"- **Why:** {rec['why']}"]
        lines += [f"- **`{name}`:** `{found[name].path / skillcheck.SKILL_FILE}`" for name in rec["skills"]
                  if name in found]
        if rec["status"] != PENDING:
            lines.append(f"- **Decided:** {rec['status']}, keep `{rec.get('keep')}`, drop `{rec.get('drop')}`, by "
                         f"{rec.get('by')}" + (f" ({rec['note']})" if rec.get("note") else ""))
        if rec.get("failed"):
            lines.append(f"- **The last merge failed:** {'; '.join(rec['failed'])}")
        lines.append("")
    lines += ["## Commands", "", "```", commands(session_dir), "```", ""]
    path = Path(session_dir) / CONFLICTS_DOC
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _who(by: str | None) -> str:
    if by:
        return by
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - a missing login name is not a reason to refuse a decision
        return "unknown"


def decide(session_dir, ident: str, action: str, keep: str | None = None, note: str = "", by: str | None = None,
           root=None) -> list[str]:
    """Record a person's decision on conflict ``ident`` (its id or its key): ``pick`` (``keep`` stays) or ``merge``
    (an agent merges them under ``keep``'s name, the first loaded by default); what is wrong, ``[]`` when recorded."""

    doc = load(session_dir) or {}
    recs = doc.get("conflicts") or []
    rec = next((r for r in recs if ident in (r.get("id"), r.get("key"))), None)
    if rec is None:
        return [f"no conflict {ident!r} in {Path(session_dir) / CONFLICTS_FILE}"]
    if rec.get("status") == MERGED:
        return [f"{rec['id']} is merged already"]
    keep = keep or (rec["skills"][0] if action == MERGE else None)
    if keep not in rec["skills"]:
        return [f"keep one of {', '.join(rec['skills'])}, not {keep!r}"]
    found = skillcheck.find(Path(root)) if root is not None else {}
    changed = [name for name, value in _digests(found, rec["skills"]).items()
               if (rec.get("digests") or {}).get(name) not in (None, value)]
    rec.update(status=PICKED if action == "pick" else TO_MERGE, keep=keep,
               drop=next(name for name in rec["skills"] if name != keep), by=_who(by), at=_now(), note=note,
               changed=changed)
    rec.pop("failed", None)
    _save(session_dir, {**doc, "conflicts": recs, "updated_at": _now()})
    return []


# ── what a session runs with ───────────────────────────────────────────────────

def applied(session_dir, names: list[str], found: dict) -> tuple[list[str], dict]:
    """The load list and the skills found, as the session's decisions leave them: each dropped skill out of the list,
    each merged skill in place of the one it keeps."""

    dropped, merged = set(), {}
    for rec in records(session_dir):
        if rec.get("status") in (PICKED, MERGED):
            dropped.add(rec.get("drop"))
        if rec.get("status") == MERGED and rec.get("path") and Path(rec["path"], skillcheck.SKILL_FILE).is_file():
            keep = rec["keep"]
            family = keep.split(":", 1)[0] if ":" in keep else None
            merged[keep] = skillcheck._skill(Path(rec["path"]), keep, family)  # noqa: SLF001 - one module's Skill
    if not dropped and not merged:
        return names, found
    return [name for name in names if name not in dropped], {**found, **merged}


def merged_paths(session_dir) -> dict[str, Path]:
    """Each merged skill's folder, by the name it keeps."""

    return {rec["keep"]: Path(rec["path"]) for rec in records(session_dir)
            if rec.get("status") == MERGED and rec.get("path")}


def dispatch(session_dir, name: str) -> str:
    """The name a turn invokes ``name`` by: a merged flat skill is carried by the session's ``merged`` plugin
    (``merged:<name>``); every other skill by its own name."""

    if session_dir is not None and ":" not in name and name in merged_paths(session_dir):
        return f"{MERGED_FAMILY}:{name}"
    return name


def plugins(root, session_dir, families: list[str]) -> list[str]:
    """The session's plugins: each family in ``families`` with the merged skills it keeps in place of its own, and,
    when a flat skill was merged, the ``merged`` plugin that carries it."""

    by_family: dict[str, dict[str, Path]] = {}
    flat: dict[str, Path] = {}
    for keep, path in merged_paths(session_dir).items():
        family, _, leaf = keep.partition(":")
        if leaf:
            by_family.setdefault(family, {})[leaf] = path
        else:
            flat[keep] = path
    out = skills.session_plugins(root, session_dir, families, by_family)
    if flat:
        holder = Path(session_dir) / MERGED_DIR / f".{MERGED_FAMILY}-family"
        shutil.rmtree(holder, ignore_errors=True)
        holder.mkdir(parents=True)
        (holder / skillcheck.FAMILY_FILE).write_text(json.dumps(
            {"name": MERGED_FAMILY, "description": "skills merged for this session"}) + "\n", encoding="utf-8")
        for name, path in flat.items():
            shutil.copytree(path, holder / name)
        out.append(str(skills.build(holder, Path(session_dir) / skills.SESSION_PLUGINS / MERGED_FAMILY)))
    return out


# ── the merge ──────────────────────────────────────────────────────────────────

def merge_prompt(rec: dict, keep, other, dest: Path) -> str:
    produces = list(dict.fromkeys([*(keep.arbor.get("produces") or []), *(other.arbor.get("produces") or [])]))
    leaf = keep.name.split(":", 1)[-1]
    return "\n".join([
        f"Merge two Arbor skills into one, for one campaign session (conflict {rec['id']}: {rec['why']}).",
        "",
        f"- **The kept skill, `{keep.name}`:** `{keep.path}` (its SKILL.md and the files beside it)",
        f"- **The other, `{other.name}`:** `{other.path}`",
        f"- **Write the merged skill into** `{dest}/`: its `SKILL.md`, and every file it links.",
        "",
        "The merged skill does what both did, as one procedure: where they overlap, write it once; where they "
        "disagree, keep the kept skill's way and say what the other did instead. Read both whole first.",
        "",
        "Its front matter:",
        f"- `name: {leaf}`, and a `description` that covers both;",
        f"- `metadata.arbor`: the kept skill's `modes` ({list(keep.modes)}) and its "
        f"`chapter` ({keep.arbor.get('chapter')!r}) or `chapters` and `triggers` as it has them; `produces` "
        f"with at least {produces}; the other keys of either, joined.",
        "",
        "It takes the kept skill's place in its folder, so keep the kept skill's relative links as they are. Copy a "
        "file the other skill links into the merged folder, and link it there. Do not edit either skill in the tree.",
        "",
        f"End with one line: `MERGED: {dest / skillcheck.SKILL_FILE}`.",
    ])


def check_merged(dest: Path, keep, other) -> list[str]:
    """What is wrong with a merged skill at ``dest``: its own check, and that it keeps ``keep``'s name, modes and
    chapter and produces every file both did."""

    if not (dest / skillcheck.SKILL_FILE).is_file():
        return [f"no {dest / skillcheck.SKILL_FILE}"]
    merged = skillcheck._skill(dest, keep.name, keep.family)  # noqa: SLF001 - one module's Skill
    errors = list(skillcheck.skill_problems(merged)[0])
    if merged.modes != keep.modes:
        errors.append(f"its modes are {list(merged.modes)}, not {list(keep.modes)}")
    if merged.arbor.get("chapter") != keep.arbor.get("chapter"):
        errors.append(f"its chapter is {merged.arbor.get('chapter')!r}, not {keep.arbor.get('chapter')!r}")
    wanted = {*(keep.arbor.get("produces") or []), *(other.arbor.get("produces") or [])}
    missing = sorted(wanted - set(merged.arbor.get("produces") or []))
    if missing:
        errors.append(f"it does not produce {', '.join(missing)}")
    return errors


def run_merges(session_dir, root, run_agent: Callable[[str, str], str]) -> list[str]:
    """Merge each conflict a person, or auto-merge, asked to, through ``run_agent(prompt, label)``; what failed,
    ``[]`` when every merge checked clean. A failed merge is pending again, with why."""

    failures = []
    for rec in records(session_dir):
        if rec.get("status") != TO_MERGE:
            continue
        base = skillcheck.find(Path(root))
        _names, found = applied(session_dir, list(rec["skills"]), base)   # an earlier merge of the kept one counts
        keep, other = found.get(rec["keep"]), found.get(rec["drop"])
        dest = Path(session_dir) / MERGED_DIR / rec["id"] / rec["keep"].split(":", 1)[-1]
        if keep is None or other is None:
            errors = [f"{rec['keep'] if keep is None else rec['drop']} is not there to merge"]
        else:
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            run_agent(merge_prompt(rec, keep, other, dest), f"merge:{rec['id']}")
            errors = check_merged(dest, keep, other)
        _update(session_dir, rec["id"], errors, dest)
        lifecycle.record(session_dir, "conflict", rec["id"], skills=rec["skills"], decision="merge",
                         keep=rec["keep"], drop=rec["drop"], by=rec.get("by"),
                         status="merged" if not errors else "failed", errors=errors)
        if errors:
            failures.append(f"{rec['id']} ({' and '.join(rec['skills'])}): {'; '.join(errors)}")
    return failures


def _update(session_dir, ident: str, errors: list[str], dest: Path) -> None:
    doc = load(session_dir) or {}
    for rec in doc.get("conflicts") or []:
        if rec.get("id") == ident:
            if errors:
                rec.update(status=PENDING, failed=errors)
            else:   # run_merges journals it, so the gate does not journal it again
                rec.update(status=MERGED, path=str(dest), merged_at=_now(), journaled=True)
    _save(session_dir, {**doc, "updated_at": _now()})


# ── the wait, and the gate the driver runs ─────────────────────────────────────

def wait(session_dir, deadline: float | None = None, margin: float = 0.0, clock=None, sleep=None,
         poll_s: float = POLL_S, remind_s: float = REMIND_S) -> str:
    """Poll ``conflicts.json`` until no conflict is pending: ``decided``; ``stop`` when ``<session>/STOP`` appears;
    ``deadline`` when the deadline less ``margin`` passes."""

    clock = clock or time.time
    sleep = sleep or time.sleep
    started = last = clock()
    log(f"waiting for a person: read {Path(session_dir) / CONFLICTS_DOC}, then\n{commands(session_dir)}")
    while True:
        if not pending(session_dir):
            return "decided"
        if (Path(session_dir) / "STOP").exists():
            return "stop"
        now = clock()
        if deadline is not None and now >= deadline - margin:
            return "deadline"
        if now - last >= remind_s:
            log(f"still waiting ({(now - started) / 60:.0f} min): {Path(session_dir) / CONFLICTS_DOC}")
            last = now
        sleep(poll_s)


def check(campaign, root, session_dir) -> list[dict]:
    """Detect the campaign's conflicts and record them (``request``); the ones pending. ``[]`` for a pack without a
    ``lifecycle:``."""

    names = lifecycle.loaded(campaign)
    if not names:
        return []
    found = skillcheck.find(Path(root))
    return request(session_dir, detect(names, found), found)


def gate(campaign, root, session_dir, cli: str | None = None, hosted: bool = False,
         run_agent: Callable[[str, str], str] | None = None, deadline: float | None = None, margin: float = 0.0,
         wait_for=None) -> tuple[str, list[str]]:
    """Settle the campaign's conflicts before it runs: ``("go", [])``, ``("refuse", reasons)`` under a host, or
    ``("stop" | "deadline", [])`` when the wait for a person ends that way."""

    from arbor import component

    wait_for = wait_for or wait
    mode, source = policy(campaign.config, cli)
    while True:
        open_ = check(campaign, root, session_dir)
        if open_:
            component.phase("conflicts")             # on the host's heartbeat, when one runs
        if open_ and mode == MERGE:
            for rec in open_:
                decide(session_dir, rec["id"], MERGE, by=f"auto-merge ({source})", root=root)
        elif open_ and hosted:
            listed = "; ".join(f"{r['id']} {' and '.join(r['skills'])}: {r['why']}" for r in open_)
            return "refuse", [f"skill conflicts, and no person under a host to decide them: {listed}. Drop one skill "
                              f"of each from lifecycle.load, or set \"skill_conflicts\": \"merge\" to let an agent "
                              f"merge them ({Path(session_dir) / CONFLICTS_DOC})"]
        elif open_:
            log(f"{len(open_)} pending: " + "; ".join(f"{r['id']} {' and '.join(r['skills'])}" for r in open_))
            outcome = wait_for(session_dir, deadline=deadline, margin=margin)
            if outcome != "decided":
                return outcome, []
        if run_agent is not None and any(rec.get("status") == TO_MERGE for rec in records(session_dir)):
            failed = run_merges(session_dir, root, run_agent)
            if failed and hosted:
                return "refuse", [f"a skill merge failed: {f}" for f in failed]
            if failed:
                log("a merge failed, and is pending again: " + "; ".join(failed))
                mode = PAUSE                     # a person decides what the agent could not
                continue
        for rec in records(session_dir):
            if rec.get("status") in (PICKED, MERGED) and not rec.get("journaled"):
                lifecycle.record(session_dir, "conflict", rec["id"], skills=rec["skills"], decision=rec["status"],
                                 keep=rec.get("keep"), drop=rec.get("drop"), by=rec.get("by"))
                _mark_journaled(session_dir, rec["id"])
        if not any(rec.get("status") in (PENDING, TO_MERGE) for rec in records(session_dir)):
            return "go", []
        if run_agent is None:
            return "go", []


def _mark_journaled(session_dir, ident: str) -> None:
    doc = load(session_dir) or {}
    for rec in doc.get("conflicts") or []:
        if rec.get("id") == ident:
            rec["journaled"] = True
    _save(session_dir, doc)


def summary_lines(session_dir) -> list[str]:
    """What a review document says about the session's conflicts; ``[]`` when there are none."""

    recs = records(session_dir)
    if not recs:
        return []
    waiting = [rec for rec in recs if rec.get("status") == PENDING]
    return [f"**Skill conflicts:** {len(recs)}, {len(waiting)} undecided: read `{Path(session_dir) / CONFLICTS_DOC}`. "
            "Decide them now, and the session will not pause for them after the review:", "", "```",
            commands(session_dir), "```"]


# ── the command ────────────────────────────────────────────────────────────────

def parser() -> argparse.ArgumentParser:
    """The command's parser, built without parsing (arbor/cli/payload.py makes its argv and schema)."""
    parser = argparse.ArgumentParser(prog="python3 -m arbor conflicts",
                                     description="Decide a session's skill conflicts (arbor/conflicts.py).")
    parser.add_argument("--session", required=True, type=Path, help="the campaign's session directory")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("show", help="list the conflicts and their decisions")
    pick = sub.add_parser("pick", help="keep one side, drop the other, for this session")
    pick.add_argument("id", help="the conflict's id, as show lists it")
    pick.add_argument("skill", help="the skill to keep")
    merge = sub.add_parser("merge", help="let an agent merge both into one skill, for this session")
    merge.add_argument("id", help="the conflict's id, as show lists it")
    merge.add_argument("--keep", help="the skill whose name, modes and chapter the merge keeps (the first loaded)")
    for each in (pick, merge):
        each.add_argument("--note", default="", help="why, for the record")
        each.add_argument("--by", help="who decides (default: the login name)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="the Arbor checkout (default: this one)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "show":
        recs = records(args.session)
        if not recs:
            print(f"no skill conflicts recorded in {args.session}")
        for rec in recs:
            print(f"{rec['id']}  {rec['status']:8} {rec['skills'][0]}  ~  {rec['skills'][1]}\n      {rec['why']}"
                  + (f"\n      keep {rec.get('keep')}, by {rec.get('by')}" if rec.get("keep") else "")
                  + (f"\n      last merge failed: {'; '.join(rec['failed'])}" if rec.get("failed") else ""))
        return 0
    errors = decide(args.session, args.id, args.action, getattr(args, "skill", None) or args.keep, args.note,
                    args.by, args.root)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print(f"{args.id}: {args.action} recorded; the session goes on once every conflict is decided")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
