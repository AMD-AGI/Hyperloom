"""Arbor's lifecycle: its chapters, its events, and the three ways a skill takes part.

A session runs through ordered **chapters** (``CHAPTERS``). A skill takes part in up to three ways (``MODES``):

- ``run``: Arbor runs it in a chapter, in an agent turn of its own, in the order the pack lists it;
- ``trigger``: Arbor runs it when an event fires (``EVENTS``), without holding the session up;
- ``use``: the chapter's agent may use it whenever it chooses, as every skill is used today.

A fourth mode, ``develop`` (``DEVELOP``), marks a skill for people working on Arbor itself, and their coding agents. It
takes no part in a campaign: it lives in a family of its own, a pack that loads that family is refused, and the agent's
SDK never gets it (``plugin_paths``).

"Chapter", because "stage" (kernel-plan items, ``arbor/stages.py``) and "phase" (map, controller and component phases)
are taken. "Trigger", because YAML reads a bare ``on`` as the boolean true.

A pack declares where skills take part in ``pack.yaml``: ``plugins:`` (skill families, Claude Code plugins under
``plugins/<family>/``), ``lifecycle:`` (a chapter's ``run`` and ``use`` lists) and ``triggers:`` (an event's skills).
A skill says where it can take part under ``metadata.arbor`` in its ``SKILL.md`` (``arbor/skillcheck.py``).

Nothing dispatches by chapter yet: until it does, a pack's ``skills:`` still decides what the agent gets, and a pack
without ``lifecycle:`` reads as today, its skills ``use`` in every chapter (``from_pack``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The chapters in order, each with what happens in it. A dot marks a sub-chapter.
CHAPTERS: tuple[tuple[str, str], ...] = (
    ("review", "a person reviews the campaign before anything runs (arbor/review.py); a fresh session only"),
    ("init", "the validators are pinned, the workspace seeded and a recorded baseline taken"),
    ("benchmark", "the benchmark is studied: its rules, its reference and what to measure (benchmarks/<name>/)"),
    ("accuracy", "the root's accuracy is measured once (component mode's full GSM8K)"),
    ("map.discover", "the dossier's reading parts, no GPU: model, system, code, workload, earlier optimizations"),
    ("map.measure", "the dossier's measured parts, on a node: baseline, profile, kernels, fast bench"),
    ("map.directions", "the directions and kernel plans, ending in the map"),
    ("iterate", "each iteration: the agent's change, then server, bench, correctness, gate and record"),
    ("round_end", "a round ends: its A/B, a rebaseline, and the dossier's update round"),
    ("end", "the session ends: its exit, report and archive"),
)
CHAPTER_NAMES: tuple[str, ...] = tuple(name for name, _ in CHAPTERS)

#: The events a ``trigger`` skill can be registered for, each with where Arbor records it today.
EVENTS: tuple[tuple[str, str], ...] = (
    ("session_start", "the init's trace mark"),
    ("review_decided", "review.json approved or rejected"),
    ("baseline_set", "the BASELINE annotation"),
    ("map_set", "the map turns active (a dossier pass exits mapped)"),
    ("map_exhausted", "the MAP_ROUND exhausted annotation"),
    ("iteration_start", "cmd_begin"),
    ("keep", "a KEEP decision"),
    ("revert", "a REVERT decision"),
    ("probe", "a PROBE decision"),
    ("invalid", "an INVALID decision"),
    ("escalate", "an ESCALATE annotation"),
    ("agent_failed", "an agent turn that crashed or timed out"),
    ("round_end", "the ROUND_END annotation"),
    ("rebaseline", "the REBASELINE annotation"),
    ("deadline_near", "the last iteration that fits before the deadline"),
    ("end", "the session's exit"),
)
EVENT_NAMES: tuple[str, ...] = tuple(name for name, _ in EVENTS)

#: The three ways a skill takes part, and the per-chapter keys of a pack's ``lifecycle:``.
MODES: tuple[str, ...] = ("run", "trigger", "use")
CHAPTER_KEYS: tuple[str, ...] = ("run", "use")
#: The mode of a skill for developing Arbor, never a campaign's; and every mode a skill may declare.
DEVELOP = "develop"
SKILL_MODES: tuple[str, ...] = (*MODES, DEVELOP)


@dataclass(frozen=True)
class Lifecycle:
    """Which skills take part where: each chapter's ``run`` and ``use`` lists, and each event's skills."""

    run: dict[str, tuple[str, ...]] = field(default_factory=dict)
    use: dict[str, tuple[str, ...]] = field(default_factory=dict)
    triggers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def runs(self, chapter: str) -> tuple[str, ...]:
        return self.run.get(chapter, ())

    def uses(self, chapter: str) -> tuple[str, ...]:
        return self.use.get(chapter, ())

    def triggered(self, event: str) -> tuple[str, ...]:
        return self.triggers.get(event, ())

    def named(self) -> tuple[str, ...]:
        """Every skill named anywhere, once each, in order."""

        found: list[str] = []
        for lists in (self.run, self.use, self.triggers):
            for names in lists.values():
                found += [name for name in names if name not in found]
        return tuple(found)


def _names(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(s, str) and s and s.strip() == s for s in value)


def errors(lifecycle: object, triggers: object) -> list[str]:
    """What is wrong with a pack's ``lifecycle:`` and ``triggers:`` sections, by their shape alone."""

    found: list[str] = []
    if lifecycle is not None:
        if not isinstance(lifecycle, dict):
            return [f"lifecycle must map chapters to their run and use lists: {lifecycle!r}"]
        for chapter, lists in lifecycle.items():
            if chapter not in CHAPTER_NAMES:
                found.append(f"lifecycle: unknown chapter {chapter!r} (chapters: {', '.join(CHAPTER_NAMES)})")
                continue
            if not isinstance(lists, dict) or not set(lists) <= set(CHAPTER_KEYS):
                found.append(f"lifecycle: {chapter} takes only {' and '.join(CHAPTER_KEYS)} lists: {lists!r}")
                continue
            found += [
                f"lifecycle: {chapter}.{key} must be a list of skill names: {names!r}"
                for key, names in lists.items()
                if not _names(names)
            ]
    if triggers is not None:
        if not isinstance(triggers, dict):
            return [*found, f"triggers must map events to skill names: {triggers!r}"]
        for event, names in triggers.items():
            if event not in EVENT_NAMES:
                found.append(f"triggers: unknown event {event!r} (events: {', '.join(EVENT_NAMES)})")
            elif not _names(names):
                found.append(f"triggers: {event} must be a list of skill names: {names!r}")
    return found


def from_pack(lifecycle: dict | None, triggers: dict | None, skills: list[str] | None) -> Lifecycle:
    """The lifecycle a pack declares. A pack without ``lifecycle:`` gets its agent ``skills`` (``agent_skills``) to
    ``use`` in every chapter, which is what its agent gets today. Call it on a pack ``errors`` found nothing in."""

    if lifecycle is None:
        everywhere = tuple(skills or ())
        use = {chapter: everywhere for chapter in CHAPTER_NAMES} if everywhere else {}
        return Lifecycle(use=use, triggers=_tuples(triggers))
    run = {chapter: tuple(lists.get("run") or ()) for chapter, lists in lifecycle.items() if lists.get("run")}
    use = {chapter: tuple(lists.get("use") or ()) for chapter, lists in lifecycle.items() if lists.get("use")}
    return Lifecycle(run=run, use=use, triggers=_tuples(triggers))


def _tuples(mapping: dict | None) -> dict[str, tuple[str, ...]]:
    return {key: tuple(names) for key, names in (mapping or {}).items()}


#: The keys of a pack's ``component:`` section that Arbor reads; the others proposed
#: (``knowledge``, ``void``, ``checks``, ``objective``) are refused until they are built.
COMPONENT_KEYS = ("lifecycle",)


def declared(campaign) -> Lifecycle | None:
    """The lifecycle the campaign's pack declares, ``None`` when it declares none. Under a host (component mode), the
    pack's ``component.lifecycle`` replaces the chapters it names, so a pack can say what changes when a host owns the
    node, its client and its gate: a ``benchmark`` chapter the host's rules make empty, say."""

    pack = campaign.pack
    if getattr(pack, "lifecycle", None) is None:
        return None
    chapters = dict(pack.lifecycle)
    if getattr(campaign, "mode", "standalone") == "component":
        chapters.update(((getattr(pack, "component", None) or {}).get("lifecycle")) or {})
    return from_pack(chapters, pack.triggers, None)


# ── the chapter of an agent's turn ────────────────────────────────────────────

#: The map's three sub-chapters. Until Arbor runs each as turns of its own, a dossier round is one agent turn in all
#: three, the chapter ``map``.
MAP = "map"
MAP_CHAPTERS: tuple[str, ...] = ("map.discover", "map.measure", "map.directions")
#: The map's phases (arbor/directions.py) in which the agent writes the dossier and its map, by chapter: a first map
#: (``bootstrap``, and ``unmapped`` once it is late) is ``map``; an update round or a fresh start is ``round_end``.
#: Every other phase, and a campaign without map rounds, is ``iterate``.
PHASE_CHAPTERS = {"bootstrap": MAP, "unmapped": MAP, "updating": "round_end", "fresh": "round_end"}
ITERATE = "iterate"


def chapter_of(campaign, state=None) -> str:
    """The chapter of the main agent's turn: ``map`` in a dossier round or a document campaign (whose work is the
    dossier's), ``round_end`` in the dossier's update round, else ``iterate``."""

    if (getattr(campaign.pack, "workload", None) or {}).get("kind") == "document":
        return MAP
    from arbor import directions

    doc = directions.current(campaign, state)
    return PHASE_CHAPTERS.get(str((doc or {}).get("phase")), ITERATE)


def sub_chapters(chapter: str) -> tuple[str, ...]:
    """The chapters a turn in ``chapter`` is in: the map's three for ``map``."""

    return MAP_CHAPTERS if chapter == MAP else (chapter,)


def turn_skills(campaign, state=None) -> tuple[str, list[str] | None]:
    """The chapter of the main agent's turn, and the skills it gets.

    A pack with ``lifecycle:`` gives the chapter's ``use`` list (for ``map``, its three sub-chapters' in order). A pack
    without one gives ``agent_skills``, the same in every chapter, as before chapters; ``None`` when it declares no
    skills at all. ``run`` skills are not the agent's: Arbor runs them in turns of their own."""

    from arbor.pack.loader import agent_skills

    chapter = chapter_of(campaign, state)
    plan = declared(campaign)
    if plan is None:
        return chapter, agent_skills(campaign.pack)
    names: list[str] = []
    for each in sub_chapters(chapter):
        names += [name for name in plan.uses(each) if name not in names]
    return chapter, names


def plugin_paths(campaign, arbor_root) -> list[str]:
    """The directories of the skill families the pack loads, for the SDK's ``plugins``. A family for developing Arbor
    is never one of them, whatever the pack says (validate_pack refuses such a pack first)."""

    from pathlib import Path

    from arbor import skillcheck

    develop = skillcheck.develop_families(Path(arbor_root))
    return [str(Path(arbor_root) / "plugins" / name) for name in getattr(campaign.pack, "plugins", None) or []
            if name not in develop]


# ── the journal a host reads ──────────────────────────────────────────────────

#: Where a session records its lifecycle, one JSON object a line: each chapter it enters (``kind`` ``chapter``), and,
#: as they are wired, the events it fires. It is the feed a host follows (docs/hosts.md); callbacks build on it.
JOURNAL = "lifecycle.jsonl"


def record(session_dir, kind: str, name: str, **detail) -> dict:
    """Append one entry to the session's journal; the entry written."""

    import json
    import time
    from pathlib import Path

    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "name": name, **detail}
    with (Path(session_dir) / JOURNAL).open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(entry) + "\n")
    return entry


def last(session_dir, kind: str) -> dict | None:
    """The journal's last entry of ``kind``; ``None`` when it has none or cannot be read."""

    import json
    from pathlib import Path

    try:
        lines = (Path(session_dir) / JOURNAL).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("kind") == kind:
            return entry
    return None


def enter(session_dir, chapter: str, iteration: int, skills: list[str] | None) -> bool:
    """Record that the session is in ``chapter`` at ``iteration``, with the skills its agent gets, unless it was
    already; whether it entered it now."""

    previous = last(session_dir, "chapter")
    if previous is not None and previous.get("name") == chapter:
        return False
    record(session_dir, "chapter", chapter, iteration=iteration, skills=list(skills or []))
    return True
