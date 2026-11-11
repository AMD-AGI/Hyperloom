"""Arbor's lifecycle: its chapters, its events, and where a campaign's skills take part.

A session runs through ordered **chapters** (``CHAPTERS``) and fires **events** (``EVENTS``).

A skill is one of two **types** (``TYPES``), declared as ``metadata.arbor.type`` in its ``SKILL.md``:

- **runtime** (the default): a campaign loads it;
- **development**: for people working on Arbor itself, and their coding agents. It takes no part in a campaign: it
  lives in a family of its own, a pack that loads it is refused, and the agent's SDK never gets it (``plugin_paths``).

A runtime skill's own metadata says how it takes part (``MODES``) and where:

- ``run``: Arbor runs it in its ``chapter``, in an agent turn of its own, after the skills it names in ``after`` and
  before those in ``before``, else in the order the pack loads them;
- ``use``: the agent of each of its ``chapters`` (every chapter when it names none) may use it whenever it chooses;
- ``trigger``: Arbor runs it when one of its ``triggers`` (events) fires, without holding the session up.

A pack only says which skills it loads, in ``pack.yaml``: ``lifecycle: {load: [...]}`` (``place`` turns that list into
each chapter's and each event's skills). A pack without ``lifecycle:`` reads as before chapters, its ``skills:`` used
in every chapter (``from_pack``).

"Chapter", because "stage" (kernel-plan items, ``arbor/stages.py``) and "phase" (map, controller and component phases)
are taken. "Trigger", because YAML reads a bare ``on`` as the boolean true. Design: docs/skills.md and
docs/skill-integration.md.
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
    ("stretch_end", "a worker's stretch ends: its report is written (arbor/worker.py)"),
    ("end", "the session's exit"),
)
EVENT_NAMES: tuple[str, ...] = tuple(name for name, _ in EVENTS)

#: A skill's two types: a campaign loads runtime skills; development skills are for people working on Arbor.
RUNTIME, DEVELOPMENT = "runtime", "development"
TYPES: tuple[str, ...] = (RUNTIME, DEVELOPMENT)
#: The three ways a runtime skill takes part.
MODES: tuple[str, ...] = ("run", "trigger", "use")
#: The one key of a pack's ``lifecycle:`` (and of its ``component.lifecycle``): the skills it loads.
LOAD = "load"


@dataclass(frozen=True)
class Lifecycle:
    """Which skills take part where: the skills loaded, each chapter's ``run`` and ``use`` lists, each event's skills."""

    load: tuple[str, ...] = ()
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
        """Every skill loaded, or named anywhere, once each, in order."""

        found: list[str] = list(self.load)
        for lists in (self.run, self.use, self.triggers):
            for names in lists.values():
                found += [name for name in names if name not in found]
        return tuple(found)


def _names(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(s, str) and s and s.strip() == s for s in value)


def errors(lifecycle: object) -> list[str]:
    """What is wrong with a pack's ``lifecycle:`` section (or its ``component.lifecycle``), by its shape alone."""

    if lifecycle is None:
        return []
    if not isinstance(lifecycle, dict):
        return [f"lifecycle must be {{{LOAD}: [skill, ...]}}: {lifecycle!r}"]
    if set(lifecycle) & set(CHAPTER_NAMES):
        return [f"lifecycle: chapters with run and use lists are gone; list the skills to load, "
                f"lifecycle: {{{LOAD}: [skill, ...]}}, and each skill's metadata places it"]
    found = [f"lifecycle: unknown key {key!r}; it takes only {LOAD}" for key in lifecycle if key != LOAD]
    names = lifecycle.get(LOAD)
    if LOAD not in lifecycle:
        found.append(f"lifecycle: no {LOAD} list")
    elif not _names(names):
        found.append(f"lifecycle: {LOAD} must be a list of skill names: {names!r}")
    else:
        found += [f"lifecycle: {LOAD} lists {name} twice" for name in dict.fromkeys(names) if names.count(name) > 1]
    return found


def place(load: list[str] | tuple[str, ...], found: dict) -> tuple[Lifecycle, list[str]]:
    """Where the skills ``load`` names take part, by their metadata (``found`` is ``skillcheck.find``'s), and what is
    wrong with that: a skill that is not there or is a development skill, an order that ``after`` and ``before`` make
    a cycle. A run chapter's skills follow ``after`` and ``before``; ties, and everything else, follow ``load``. A run
    skill without a chapter it can run in is left out of every chapter; its own check reports it (skillcheck)."""

    run: dict[str, list[str]] = {}
    use: dict[str, list[str]] = {}
    triggers: dict[str, list[str]] = {}
    problems: list[str] = []
    for name in load:
        skill = found.get(name)
        if skill is None:
            problems.append(f"skill not found: {name}")
            continue
        if skill.development:
            problems.append(f"skill {name} is for developing Arbor, not for a campaign (its type is {DEVELOPMENT})")
            continue
        meta = skill.arbor
        if "run" in skill.modes and meta.get("chapter") in CHAPTER_NAMES:
            run.setdefault(meta["chapter"], []).append(name)
        if "use" in skill.modes:
            for chapter in meta.get("chapters") or CHAPTER_NAMES:
                use.setdefault(chapter, []).append(name)
        if "trigger" in skill.modes:
            for event in meta.get("triggers") or ():
                triggers.setdefault(event, []).append(name)
    ordered = {}
    for chapter, names in run.items():
        ordered[chapter], cycle = _ordered(names, found)
        if cycle:
            problems.append(f"lifecycle: {chapter}'s run skills {', '.join(cycle)} must each come before another of "
                            f"them (after and before make a cycle)")
    plan = Lifecycle(load=tuple(load), run=ordered, use=_tuples(use), triggers=_tuples(triggers))
    return plan, problems


def _ordered(names: list[str], found: dict) -> tuple[tuple[str, ...], list[str]]:
    """``names`` in the order their ``after`` and ``before`` ask, ties in the order given; and the names left in a
    cycle, in the order given (``[]`` when there is none)."""

    first = {name: set() for name in names}          # name -> the names that must come before it
    for name in names:
        meta = found[name].arbor
        first[name] |= {other for other in meta.get("after") or [] if other in first and other != name}
        for other in meta.get("before") or []:
            if other in first and other != name:
                first[other].add(name)
    done: list[str] = []
    while len(done) < len(names):
        ready = [name for name in names if name not in done and first[name] <= set(done)]
        if not ready:
            return (*done, *(n for n in names if n not in done)), [n for n in names if n not in done]
        done.append(ready[0])
    return tuple(done), []


def from_pack(lifecycle: dict | None, skills: list[str] | None, found: dict | None = None) -> Lifecycle:
    """The lifecycle a pack declares. A pack without ``lifecycle:`` gets its agent ``skills`` (``agent_skills``) to
    ``use`` in every chapter, as before chapters. Call it on a pack ``errors`` and ``place`` found nothing in."""

    if lifecycle is None:
        everywhere = tuple(skills or ())
        use = {chapter: everywhere for chapter in CHAPTER_NAMES} if everywhere else {}
        return Lifecycle(load=everywhere, use=use)
    return place(lifecycle.get(LOAD) or [], found or {})[0]


def _tuples(mapping: dict) -> dict[str, tuple[str, ...]]:
    return {key: tuple(names) for key, names in mapping.items()}


#: The keys of a pack's ``component:`` section that Arbor reads; the others once proposed
#: (``knowledge``, ``void``, ``checks``, ``objective``) are refused until they are built.
COMPONENT_KEYS = ("lifecycle",)


def arbor_root(campaign):
    """The Arbor checkout the campaign's pack lives in (``<root>/packs/<name>``)."""

    from pathlib import Path

    return Path(campaign.pack.pack_dir).resolve().parent.parent


def loaded(campaign) -> list[str] | None:
    """The skills the campaign's pack loads, ``None`` when it declares no ``lifecycle:``. Under a host (component mode),
    the pack's ``component.lifecycle`` load list replaces it, so a pack can say what changes when a host owns the node,
    its client and its gate: no ``benchmark:study`` when the host's rules apply, say."""

    pack = campaign.pack
    if getattr(pack, "lifecycle", None) is None:
        return None
    names = list(pack.lifecycle.get(LOAD) or [])
    if getattr(campaign, "mode", "standalone") == "component":
        hosted = ((getattr(pack, "component", None) or {}).get("lifecycle")) or {}
        if LOAD in hosted:
            names = list(hosted[LOAD] or [])
    return names


def declared(campaign, root=None) -> Lifecycle | None:
    """Where the campaign's skills take part (``place`` on ``loaded``), ``None`` when its pack declares no
    ``lifecycle:``. ``root`` is the Arbor checkout whose skills it reads, the pack's by default. The session's
    conflict decisions apply (``arbor/conflicts.py``): a dropped skill is out, a merged one in place of the one it
    keeps."""

    from pathlib import Path

    from arbor import conflicts, skillcheck

    names = loaded(campaign)
    if names is None:
        return None
    found = skillcheck.find(Path(root) if root is not None else arbor_root(campaign))
    session = getattr(campaign, "session_root", None)
    if session and (Path(session) / conflicts.CONFLICTS_FILE).is_file():
        names, found = conflicts.applied(session, names, found)
    return place(names, found)[0]


def session_of(campaign):
    """The campaign's session directory, ``None`` when it has none (``session_root``)."""

    session = getattr(campaign, "session_root", None)
    return session if session else None


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


# ── the master and its workers ────────────────────────────────────────────────

#: Which side of a session runs each chapter (``arbor/roles.py``). The master
#: sets the campaign up, writes the map, ends each round and ends the campaign; a worker runs the rounds. Under a host,
#: the host is the master that owns the session, and calls Arbor's master, then its workers.
MASTER, WORKER = "master", "worker"
ROLES: dict[str, tuple[str, ...]] = {
    MASTER: ("review", "init", "benchmark", "accuracy", *MAP_CHAPTERS, "round_end", "end"),
    WORKER: (ITERATE,),
}


def role_of_chapter(chapter: str) -> str:
    """The side that runs ``chapter`` (``map``, the map's three together, is the master's)."""

    return WORKER if all(each in ROLES[WORKER] for each in sub_chapters(chapter)) else MASTER


def turn_skills(campaign, state=None) -> tuple[str, list[str] | None]:
    """The chapter of the main agent's turn, and the skills it gets.

    A pack with ``lifecycle:`` gives the chapter's ``use`` list (for ``map``, its three sub-chapters' in order). A pack
    without one gives ``agent_skills``, the same in every chapter, as before chapters; ``None`` when it declares no
    skills at all. ``run`` skills are not the agent's: Arbor runs them in turns of their own."""

    from arbor import conflicts
    from arbor.pack.loader import agent_skills

    chapter = chapter_of(campaign, state)
    plan = declared(campaign)
    if plan is None:
        return chapter, agent_skills(campaign.pack)
    names: list[str] = []
    for each in sub_chapters(chapter):
        names += [name for name in plan.uses(each) if name not in names]
    return chapter, [conflicts.dispatch(session_of(campaign), name) for name in names]


def plugin_paths(campaign, arbor_root, session_dir) -> list[str]:
    """The families of the skills the pack loads, each built as a Claude Code plugin for the session
    (``arbor/skills.py``), for the SDK's ``plugins``, with the session's merged skills in them
    (``arbor/conflicts.py``). A family for developing Arbor is never one of them, whatever the pack says (validate_pack
    refuses such a pack first); a pack without ``lifecycle:`` loads flat skills only."""

    from pathlib import Path

    from arbor import conflicts, skillcheck

    development = skillcheck.development_families(Path(arbor_root))
    plan = declared(campaign, arbor_root)            # the load list as the session's decisions leave it
    families = [name.split(":", 1)[0] for name in (plan.load if plan else ()) if ":" in name]
    return conflicts.plugins(arbor_root, session_dir, [f for f in families if f not in development])


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


def last(session_dir, kind: str, name: str | None = None) -> dict | None:
    """The journal's last entry of ``kind`` (and ``name``, when given); ``None`` when it has none or cannot be read."""

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
        if isinstance(entry, dict) and entry.get("kind") == kind and name in (None, entry.get("name")):
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
