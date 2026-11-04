"""Arbor runs a chapter's ``run`` skills, each in an agent turn of its own (arbor/lifecycle.py, mode ``run``).

A turn in a chapter whose pack lists ``run`` skills is not the main agent's: Arbor runs the skills in order. For
``map``, that is the benchmark's, then ``map.discover``'s, ``map.measure``'s and ``map.directions``'s. For
``round_end``, it is that chapter's own.

**Each skill gets:**
- **its own agent turn,** dispatched as ``/<skill> <brief>``: a skill marked ``disable-model-invocation`` is not one
  an agent picks, but a dispatch runs it;
- **a brief,** written under ``<session>/runs/``: the campaign, the chapter, the files it must write, and the
  program the main agent would have read;
- **its chapter's ``use`` skills** and the pack's families.

**After its turn,** Arbor checks that the files the skill declares (``metadata.arbor.produces``) exist. A skill whose
files are there, for the same ``reuse_key`` values as its last complete run, is skipped (``fresh``). A skill that
leaves one missing stops the chapter, and the next turn takes it up again.

**What the iteration sees:** the skills' outputs, joined, then ``INTENT: LEARN`` and a ``DONE:`` line naming what
ran, so it is booked as the map's writing is today. The last skill's ``DIRECTIONS:`` line sets the map, so a dossier
pass exits ``mapped`` after ``map.directions``. Each run, skip or stop is written to the journal (``lifecycle.record``,
kind ``run``).

**Placeholders in ``produces``:**
- ``{dossier}``: ``docs/dossier/<name>/`` in the Arbor tree, ``<name>`` the config's ``dossier``, else the campaign's
  identity as a slug;
- ``{benchmark}``: ``benchmarks/<name>/``, ``<name>`` the config's ``benchmark``, else the pack's name;
- ``{session}``: the session directory;
- ``{workspace}``: the workspace root.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from arbor import lifecycle

RUNS_DIR = "runs"
DOSSIER_ROOT = Path("docs") / "dossier"
BENCHMARK_ROOT = Path("benchmarks")
#: The chapters a turn in each chapter runs the ``run`` skills of, in order.
TURN_CHAPTERS = {lifecycle.MAP: ("benchmark", *lifecycle.MAP_CHAPTERS), "round_end": ("round_end",)}
DONE, INCOMPLETE, FRESH = "done", "incomplete", "fresh"
SLUG = re.compile(r"[^a-z0-9.]+")


@dataclass(frozen=True)
class Run:
    """A ``run`` skill as a turn runs it: its chapter, its name, and its declared files and reuse key."""

    chapter: str
    skill: str
    produces: tuple[str, ...]
    reuse_key: tuple[str, ...]


def slug(text: str) -> str:
    """``text`` as a directory name: lower case, runs of other characters as one ``-``."""

    return SLUG.sub("-", text.lower()).strip("-.") or "campaign"


def places(campaign, session_dir, arbor_root) -> dict[str, str]:
    """The values of ``produces``'s placeholders for this campaign."""

    config = getattr(campaign, "config", None) or {}
    dossier = str(config.get("dossier") or slug(str(getattr(campaign, "identity", "") or "campaign")))
    benchmark = str(config.get("benchmark") or getattr(campaign.pack, "name", "") or "benchmark")
    return {
        "dossier": str(Path(dossier) if Path(dossier).is_absolute() else Path(arbor_root) / DOSSIER_ROOT / dossier),
        "benchmark": str(Path(arbor_root) / BENCHMARK_ROOT / benchmark),
        "session": str(session_dir),
        "workspace": str(config.get("workspace_root") or ""),
    }


def resolve(pattern: str, values: dict[str, str]) -> Path:
    """A ``produces`` pattern with its placeholders filled; an unknown one is left as it is."""

    return Path(re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), pattern))


def turn_runs(campaign, chapter: str, arbor_root) -> list[Run]:
    """The ``run`` skills a turn in ``chapter`` runs, in order, with what each declares; ``[]`` when there are none."""

    from arbor import skillcheck

    plan = lifecycle.declared(campaign)
    if plan is None:
        return []
    found = skillcheck.find(Path(arbor_root))
    runs = []
    for each in TURN_CHAPTERS.get(chapter, ()):
        for name in plan.runs(each):
            meta = found[name].arbor if name in found else {}
            runs.append(Run(each, name, tuple(meta.get("produces") or ()), tuple(meta.get("reuse_key") or ())))
    return runs


def _record_path(session_dir, skill: str) -> Path:
    return Path(session_dir) / RUNS_DIR / f"{skill.replace(':', '__')}.json"


def _reuse(campaign, run: Run) -> dict[str, str]:
    config = getattr(campaign, "config", None) or {}
    return {key: str(config.get(key, "")) for key in run.reuse_key}


def fresh(session_dir, run: Run, outputs: list[Path], reuse: dict[str, str]) -> bool:
    """Whether ``run``'s last complete run stands: every file it declares is there, for the same reuse key."""

    try:
        record = json.loads(_record_path(session_dir, run.skill).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (record.get("status") == DONE and record.get("reuse") == reuse and bool(outputs)
            and all(path.exists() for path in outputs))


def brief(session_dir, run: Run, outputs: list[Path], values: dict[str, str], program: str, iteration: int) -> Path:
    """Write the brief ``run``'s turn reads, and return its path."""

    lines = [
        f"# Brief: {run.skill}, chapter {run.chapter}, iteration {iteration}",
        "",
        "Arbor runs this skill in a turn of its own. Do the skill's work for this campaign, then end your reply with",
        "the lines its instructions ask for. Write each file below; Arbor checks they exist when the turn ends.",
        "",
        "## Files to write",
        *([f"- `{path}`" for path in outputs] or ["- (the skill declares none)"]),
        "",
        "## Where things are",
        *[f"- **{key}:** `{value}`" for key, value in values.items() if value],
        "",
        "## The campaign (what the main agent reads)",
        "",
        program,
    ]
    path = Path(session_dir) / RUNS_DIR / f"{run.skill.replace(':', '__')}.brief.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_turns(campaign, chapter: str, session_dir, arbor_root, iteration: int, program: str,
              run_agent: Callable[[str, list[str], str], str]) -> str:
    """Run the ``run`` skills of a turn in ``chapter`` (``turn_runs``), each through ``run_agent(prompt, skills,
    label)``, and return what the iteration reads as the agent's output."""

    values = places(campaign, session_dir, arbor_root)
    declared = lifecycle.declared(campaign) or lifecycle.Lifecycle()
    parts: list[str] = []
    summary: list[str] = []
    for run in turn_runs(campaign, chapter, arbor_root):
        outputs = [resolve(pattern, values) for pattern in run.produces]
        reuse = _reuse(campaign, run)
        if fresh(session_dir, run, outputs, reuse):
            lifecycle.record(session_dir, "run", run.skill, chapter=run.chapter, iteration=iteration, status=FRESH)
            summary.append(f"{run.skill} fresh")
            continue
        path = brief(session_dir, run, outputs, values, program, iteration)
        began = time.time()
        text = run_agent(f"/{run.skill} {path}", list(declared.uses(run.chapter)), f"run:{run.skill}")
        missing = [str(p) for p in outputs if not p.exists()]
        status = INCOMPLETE if missing else DONE
        record = {"skill": run.skill, "chapter": run.chapter, "iteration": iteration, "status": status,
                  "reuse": reuse, "outputs": [str(p) for p in outputs], "missing": missing,
                  "seconds": round(time.time() - began, 1)}
        _record_path(session_dir, run.skill).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        lifecycle.record(session_dir, "run", run.skill, chapter=run.chapter, iteration=iteration, status=status,
                         missing=missing)
        parts.append(f"## {run.skill}\n\n{text.strip()}")
        summary.append(f"{run.skill} {status}" + (f" (missing {', '.join(missing)})" if missing else ""))
        if missing:
            break
    tail = f"INTENT: LEARN\nDONE: chapter {chapter}: {'; '.join(summary) or 'nothing to run'}"
    return "\n\n".join([*parts, tail])
