"""One scale for a rolling campaign's numbers, across the nodes its chunks land on.

A rolling chunk measures the current tree on its own node and makes that the bar
(scripts/model-m-campaign/rebase_bar.py), so the gate and the agent compare against the
node they run on. Read in sequence, though, the bar then jumps with the node: Model M
day-0's b0 went 175.76 -> 155.90 on a slower node after six KEEPs, which reads as a
regression. Operator, 2026-10-20: interpolate instead of changing 175 to 155.

Each REBASELINE lesson carries its step: the bar before it (the previous node's)
over this node's reading of the same tree. The steps chain into a factor that puts
a number measured on any node on the campaign's first node's scale, the scale its
baseline and target were set on. `best_metric` itself stays this node's: the gate
and the agent keep comparing like with like.

The chained value is the product of the gains the gate booked. A KEEP books one
replay, so the chain carries that replay's noise forward; a same-node A/B is what
backs a claim. A round end's confirmed A/B (arbor/rounds.py, a ROUND_AB lesson) makes
its node's A/B mean the bar, so the chain starts again there at 1.0: from then on
"the first node's scale" is that A/B node's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def applies(campaign, session=None) -> tuple[bool, str]:
    """Whether a rolling chunk rebases the bar on its node, and why.

    A rebased bar, and the chain that puts every number on one scale, only mean
    something for a campaign working through a dossier's directions: the bar is what
    the next direction is graded against, and a round end re-baselines it before the
    dossier's update round. A campaign with no dossier (its agent lacks the dossier
    skill, map rounds are off, or the workload is a document), or whose map has no
    directions yet, keeps the bar it was given (operator, 2026-10-28). A parallel
    lane's seeded map counts: its directions came from the run's dossier.

    Reads <session>/directions.json without loading it through arbor.directions,
    which puts back a hand-edited map: a check must not write. A map that cannot be
    read is a "no", with the reason, so nothing rebases on a guess.
    """
    root = Path(session) if session else getattr(campaign, "session_root", None)
    if not root:
        return False, "no session"
    from arbor import directions
    path = Path(root) / directions.FILE
    doc = None
    if path.is_file():
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return False, f"the map cannot be read ({exc})"
    dirs = doc.get("directions") if isinstance(doc, dict) else None
    if directions.is_lane(doc):
        return (True, f"a lane's map, {len(dirs)} direction(s)") if dirs else (False, "a lane's map with no directions")
    if not directions.enabled(campaign):
        return False, "no dossier: the agent has no dossier skill, map rounds are off, or the workload is a document"
    if not dirs:
        return False, "no directions yet: the dossier's map has not been accepted"
    return True, f"{len(dirs)} direction(s) from the dossier"


# rebase_bar.py's lesson text, for lessons written before they carried the fields:
# "bar rebased on <node> (job <id>): <old> -> <new> (<rel>), <secondary> <old> -> <new>; ..."
_PRIMARY = re.compile(r"bar rebased on \S+ \(job [^)]*\): (\S+) -> ([-+.\deE]+)")
_SECONDARY = re.compile(r"-> [-+.\deE]+(?: \([^)]*\))?, \S+ (\S+) -> ([-+.\deE]+|None);")


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _ratio(before, after) -> float:
    b, a = _num(before), _num(after)
    return b / a if b and a else 1.0


def steps(lesson: dict) -> tuple[float, float]:
    """A REBASELINE lesson's (primary, secondary) factor from its node to the one before."""
    if "bar_before" in lesson:
        return (_ratio(lesson.get("bar_before"), lesson.get("metric")),
                _ratio(lesson.get("secondary_before"), lesson.get("secondary")))
    text = lesson.get("change") or ""
    m, m2 = _PRIMARY.search(text), _SECONDARY.search(text)
    return (_ratio(m.group(1), m.group(2)) if m else 1.0,
            _ratio(m2.group(1), m2.group(2)) if m2 else 1.0)


def scales(lessons) -> list[tuple[float, float]]:
    """Per lesson, the (primary, secondary) factor to the first node's scale in force after it."""
    s = s2 = 1.0
    out = []
    for lesson in lessons or []:
        if isinstance(lesson, dict) and lesson.get("decision") == "REBASELINE":
            f, f2 = steps(lesson)
            s, s2 = s * f, s2 * f2
        elif isinstance(lesson, dict) and lesson.get("decision") == "ROUND_AB":
            s = s2 = 1.0
        out.append((s, s2))
    return out


def current(lessons) -> tuple[float, float]:
    """The factors in force now: this node's numbers times these are on the first node's scale."""
    sc = scales(lessons)
    return sc[-1] if sc else (1.0, 1.0)


def on_first_scale(value, factor: float):
    """`value` on the first node's scale, or None when it is not a number."""
    v = _num(value)
    return v * factor if v is not None else None
