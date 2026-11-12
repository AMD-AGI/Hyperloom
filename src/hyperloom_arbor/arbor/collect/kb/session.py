"""The adapter: the only module (with ``layout.py``) that reads Arbor's files, so the rest of the export sees one shape.

An Arbor session directory read once: which gated iteration owns each row, record and trace item, and the bars. The
session's format is the one Arbor declares (``arbor.checkpoints.session_format``, interface ``session``); a format
this adapter does not read is refused by name, never guessed from which fields are present. A new session format is
a new branch here, and ``records.py``, ``export.py``, the declaration and the mapping stay as they are.

One record per iteration the gate ruled on: an iteration with a KEEP, REVERT, PROBE or INVALID row among
``session.json`` lessons. Every lesson row, ledger record, hook-trace session and Claude session entry has one owner
among those iterations, and an owner's share never changes once a campaign ``end`` follows it, so collecting a
campaign again after it resumes writes each earlier record unchanged.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arbor.collect.kb.layout import (
    BENCH_ROOT,
    GATE,
    HOST_BENCH_ROOT,
    REBASING_INTENTS,
    Link,
    SETS_BAR,
    as_dict,
    epoch,
    measured,
    number,
    read_json,
    read_jsonl,
    sha256_file,
)

#: The session formats this adapter reads (Arbor's interface ``session``; 0 is a session without a checkpoint).
READS_SESSION_FORMATS = (0, 1, 2)
EVAL_DIR = "eval"

#: How Overseer tells an agent session's role from its first prompt (its role prefixes).
ROLE_PREFIX = (
    ("critic", "# Arbor Critic"),
    ("server", "The harness has committed your change"),
    ("bench", "Run the bench and report the metric"),
)
PHASES = ("worker", "server", "bench", "critic")
_WORKER_HEAD = re.compile(r"# [\w.-]+ —")
_PROMPT_BEST = re.compile(r"^- Best: ([0-9.eE+-]+)\s*$", re.M)
_LOG_ITERATION = re.compile(r"^\[arbor\] === Iteration (\d+) ===")
_LOG_BEST = re.compile(r"^\[arbor\] tier=\S* best=(\S+)")
_LOG_GATE = re.compile(r"^\[arbor\] Gate: (\w+)(?: \(([^)]*)\))? — (.*)$")
_LOG_CRITIC = re.compile(r"^\[arbor\] Critic: (.*)$")
_LOG_COMMITTED = re.compile(r"^\[arbor\] Committed: ([0-9a-f]{7,40})")


class Owners:
    """Which gated iteration owns each item, and which gated iterations are final.

    A row or record belongs to the gated iteration of its own number, else to the next gated one. A trace session
    belongs to the gated iteration whose window holds its first event. A window runs from the previous gated
    iteration's last row to this one's, as Overseer's do, and on to a campaign ``end`` that comes before the next
    gated iteration's last row: a resumed campaign starts its next iteration after the end. An iteration is final
    once an ``end`` comes after every row of it; from then on what it owns cannot change.
    """

    def __init__(self, lessons: list[dict], end_times: list[float]):
        self.gated = sorted({r["iteration"] for r in lessons
                             if r.get("decision") in GATE and isinstance(r.get("iteration"), int)})
        last: dict[int, float] = {}
        for row in lessons:
            ts, it = epoch(row.get("ts")), row.get("iteration")
            if ts is not None and it in self.gated:
                last[it] = max(last.get(it, ts), ts)
        self.t1 = [last.get(it, float("-inf")) for it in self.gated]
        self.end = max(end_times) if end_times else None
        nexts = self.t1[1:] + [float("inf")]
        self.bounds = [max([t1] + [e for e in end_times if t1 <= e < t2]) for t1, t2 in zip(self.t1, nexts)]

    def by_iteration(self, iteration: Any, ts: float | None) -> int | None:
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            return self.by_time(ts)
        at = bisect.bisect_left(self.gated, iteration)
        return self.gated[at] if at < len(self.gated) else None

    def by_time(self, ts: float | None) -> int | None:
        if not self.gated:
            return None
        if ts is None:
            return self.gated[0]
        at = bisect.bisect_left(self.bounds, ts)
        return self.gated[at] if at < len(self.gated) else None

    def final(self, iteration: int) -> bool:
        return self.end is not None and self.t1[self.gated.index(iteration)] <= self.end


@dataclass
class Bars:
    """The bars an iteration began under, as CampaignState holds them (arbor/state.py)."""

    best: float | None = None
    best_secondary: float | None = None
    set_by: int | None = None


def _apply_keep(state: Bars, row: dict, directions: tuple[str, str]) -> None:
    metric, secondary = measured(row.get("metric")), measured(row.get("secondary"))
    if row.get("frontier") or metric is None:
        return
    if row.get("intent") in REBASING_INTENTS:
        state.best, state.set_by = metric, row.get("iteration")
        state.best_secondary = secondary if secondary is not None else state.best_secondary
        return
    better = state.best is None or (metric < state.best if directions[0] == "min" else metric > state.best)
    if better:
        state.best, state.set_by = metric, row.get("iteration")
    if secondary is not None:
        sec_better = state.best_secondary is None or (
            secondary < state.best_secondary if directions[1] == "min" else secondary > state.best_secondary)
        state.best_secondary = secondary if sec_better else state.best_secondary


def replay_bars(lessons: list[dict], session: dict) -> tuple[dict[int, Bars], Bars]:
    """The bars each iteration began under, replayed by CampaignState's rules, and the last ones.

    A BASELINE, REBASELINE or ROUND_AB row sets both bars to its readings. A KEEP with a reading raises the bar to
    it (lowers, for a minimized metric) unless it is a frontier trade, which moves neither bar; a non-frontier
    FIX-CORRECTNESS or FIX-STABILITY KEEP rebases both bars to its readings, either way. A retraction, a KEEP that
    names what it retracts, puts the bars back where they stood before the retracted KEEP, which its row does not
    record: the bars are unknown from it until a row sets them.
    """

    directions = (session.get("metric_direction") or "max", session.get("secondary_direction") or "max")
    state, unknown = Bars(), False
    before: dict[int, Bars] = {}
    for row in lessons:
        it, decision = row.get("iteration"), row.get("decision")
        if isinstance(it, int) and it not in before and decision != "BASELINE":
            before[it] = Bars() if unknown else Bars(state.best, state.best_secondary, state.set_by)
        if decision in SETS_BAR:
            secondary = measured(row.get("secondary"))
            if decision == "BASELINE" and secondary is None:
                secondary = number(session.get("baseline_secondary"))
            if measured(row.get("metric")) is not None:
                state.best, state.set_by, unknown = measured(row.get("metric")), it, False
            state.best_secondary = secondary if secondary is not None else state.best_secondary
        elif decision == "KEEP" and row.get("retracts"):
            unknown = True
        elif decision == "KEEP" and not unknown:
            _apply_keep(state, row, directions)
    return before, (Bars() if unknown else state)


@dataclass
class LogSegment:
    """What one ``=== Iteration N ===`` segment of the campaign log says, when the export was given the log."""

    text: str
    best: float | None = None
    gate_reason: str | None = None
    critic: str | None = None
    commit: str | None = None


def read_campaign_logs(paths: list[Path]) -> dict[int, LogSegment]:
    """Each iteration's last segment across the campaign logs, in the order given (a resumed campaign appends)."""

    segments: dict[int, LogSegment] = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        current: int | None = None
        for line in lines:
            head = _LOG_ITERATION.match(line)
            if head:
                current = int(head.group(1))
                segments[current] = LogSegment(text="")
            if current is not None:
                _read_log_line(segments[current], line)
    return segments


def _read_log_line(segment: LogSegment, line: str) -> None:
    segment.text += line + "\n"
    best, gate = _LOG_BEST.match(line), _LOG_GATE.match(line)
    critic, commit = _LOG_CRITIC.match(line), _LOG_COMMITTED.match(line)
    if best:
        try:
            segment.best = float(best.group(1))
        except ValueError:
            segment.best = None
    if gate:
        segment.gate_reason = gate.group(3).strip() or None
    if critic:
        segment.critic = critic.group(1).strip() or None
    if commit:
        segment.commit = commit.group(1)


@dataclass
class TraceSession:
    """One agent session of the hook trace: its role as Overseer reads it, its span, and its first prompt."""

    sid: str
    role: str
    t0: float
    t1: float
    first_prompt: str


def _trace_sessions(events: list[tuple[int, str, dict]]) -> dict[str, TraceSession]:
    spans: dict[str, list[float]] = {}
    prompts: dict[str, tuple[float, str]] = {}
    for _, _, event in events:
        sid, ts = event.get("claude_session_id"), epoch(event.get("ts"))
        if not sid or ts is None:
            continue
        span = spans.setdefault(sid, [ts, ts])
        span[0], span[1] = min(span[0], ts), max(span[1], ts)
        if event.get("event") == "user_prompt" and (sid not in prompts or ts < prompts[sid][0]):
            prompts[sid] = (ts, str(event.get("prompt") or ""))
    sessions = {}
    for sid, (t0, t1) in spans.items():
        first = prompts.get(sid, (t0, ""))[1]
        role = next((r for r, prefix in ROLE_PREFIX if first.startswith(prefix)), None)
        role = role or ("worker" if _WORKER_HEAD.match(first) else "other")
        sessions[sid] = TraceSession(sid, role, t0, t1, first)
    return sessions


@dataclass
class Session:
    """An Arbor session directory, read once, with every item given to its owning gated iteration."""

    dir: Path
    session: dict
    meta: dict
    lessons: list[dict]
    owners: Owners
    run_id: str
    format: int = 0
    own_lessons: dict[int, list[dict]] = field(default_factory=lambda: defaultdict(list))
    own_ledger: dict[int, list[dict]] = field(default_factory=lambda: defaultdict(list))
    own_events: dict[int, list[tuple[int, str]]] = field(default_factory=lambda: defaultdict(list))
    own_traces: dict[int, list[TraceSession]] = field(default_factory=lambda: defaultdict(list))
    own_claude: dict[int, list[dict]] = field(default_factory=lambda: defaultdict(list))
    events: list[tuple[int, str, dict]] = field(default_factory=list)
    directions: set[str] = field(default_factory=set)
    log: dict[int, LogSegment] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def config(self) -> dict:
        return as_dict(self.meta.get("config"))

    def main(self, iteration: int) -> dict:
        """The iteration's last gate row, the one its record describes."""

        rows = [r for r in self.lessons if r.get("iteration") == iteration and r.get("decision") in GATE]
        return rows[-1] if rows else {}

    def agg(self, iteration: int) -> Path | None:
        d = self.dir / f"iter_{iteration}"
        aggs = sorted(p for p in d.glob("agg_*.json") if p.is_file()) if d.is_dir() else []
        return aggs[0] if aggs else None

    def path(self, iteration: int, rel: str) -> Path:
        return self.dir / f"iter_{iteration}" / rel

    def iteration_json(self, iteration: int, rel: str) -> dict:
        """An iteration's JSON file (bench_output.json, correctness_output.json, agentic_run/…), ``{}`` if absent."""

        return as_dict(read_json(self.path(iteration, rel)))

    def agg_doc(self, iteration: int) -> dict:
        path = self.agg(iteration)
        return as_dict(read_json(path)) if path else {}

    def patch(self, iteration: int) -> Path | None:
        path = self.path(iteration, "patch.diff")
        return path if path.is_file() else None

    def iteration_links(self, iteration: int, rels: tuple[str, ...]) -> list[Link]:
        """Links to those of ``rels`` the iteration holds, placed where a rebuilt session directory puts them."""

        return [Link(self.path(iteration, rel), f"iter_{iteration}/{rel}") for rel in rels
                if self.path(iteration, rel).is_file()]

    def effective_flags(self, iteration: int) -> dict | None:
        """The server flags the iteration ran (effective_config.json), else the nearest earlier iteration's."""

        for it in range(iteration, -1, -1):
            flags = self.iteration_json(it, "effective_config.json").get("flags")
            if isinstance(flags, dict) and flags:
                return flags
        return None

    def eval_files(self, names_in: str, window: tuple[float, float]) -> list[Link]:
        """Files under eval/ whose name ``names_in`` mentions and that were modified inside ``window``."""

        eval_dir = self.dir / EVAL_DIR
        if not eval_dir.is_dir():
            return []
        return [Link(p, p.relative_to(self.dir).as_posix(), "eval") for p in sorted(eval_dir.rglob("*"))
                if p.is_file() and p.name in names_in and window[0] < p.stat().st_mtime <= window[1]]

    def transcripts(self, iteration: int) -> list[Link]:
        """The Claude transcripts of the sessions the iteration owns, and their subagents', when this host can read
        them (``<sid>.jsonl`` and ``<sid>/subagents/*.jsonl`` beside it, as Claude Code keeps them)."""

        links = []
        for item in self.own_claude.get(iteration, []):
            path = Path(str(item.get("transcript_path") or ""))
            if not (path.is_file() and os.access(path, os.R_OK)):
                continue
            links.append(Link(path, f"claude/{path.name}", "transcript"))
            subagents = path.with_suffix("") / "subagents"
            for sub in sorted(subagents.glob("*.jsonl")) if subagents.is_dir() else []:
                if os.access(sub, os.R_OK):
                    links.append(Link(sub, f"claude/{path.stem}/subagents/{sub.name}", "transcript"))
        return links

    def bench_roots(self, iteration: int) -> list[str]:
        """Where the iteration's bench wrote its run: ``agentic_run``, and a host's ``host_bench`` or its ``runK``."""

        roots = [BENCH_ROOT] if self.path(iteration, BENCH_ROOT).is_dir() else []
        host = self.path(iteration, HOST_BENCH_ROOT)
        if host.is_dir():
            runs = sorted(p for p in host.iterdir() if p.is_dir() and p.name.startswith("run"))
            roots += [f"{HOST_BENCH_ROOT}/{p.name}" for p in runs] or [HOST_BENCH_ROOT]
        return roots

    def keeps(self, iteration: int) -> list[Link]:
        """A host's ``keeps/NNNN/keep.json`` naming this iteration: how Overseer maps the host's gate tests to it."""

        keeps = self.dir / "keeps"
        found = []
        for path in sorted(keeps.glob("*/keep.json")) if keeps.is_dir() else []:
            if as_dict(read_json(path)).get("iteration") == iteration:
                found.append(Link(path, path.relative_to(self.dir).as_posix(), "keep"))
        return found

    def prompt_best(self, iteration: int) -> float | None:
        """The ``- Best:`` line of the first worker prompt the iteration owns: the bar the worker was shown."""

        header = f"# {self.session.get('pack_name')} "
        for trace in sorted(self.own_traces.get(iteration, []), key=lambda t: t.t0):
            if trace.role == "worker" and trace.first_prompt.startswith(header):
                found = _PROMPT_BEST.search(trace.first_prompt)
                return float(found.group(1)) if found else None
        return None


def load(session_dir: Path, campaign_logs: list[Path]) -> Session | str:
    """The session read and every item owned, or why it could not be read."""

    fmt = session_format(session_dir)
    if fmt not in READS_SESSION_FORMATS:
        return (f"session format {fmt} is not one this exporter reads "
                f"({', '.join(map(str, READS_SESSION_FORMATS))}); update arbor/collect/kb/session.py")
    session = read_json(session_dir / "session.json")
    if not isinstance(session, dict) or not isinstance(session.get("lessons"), list):
        return "session.json is missing or holds no lessons"
    lessons = [row for row in session["lessons"] if isinstance(row, dict)]
    meta = as_dict(read_json(session_dir / "reports" / "trace" / "meta.json"))
    lifecycle = [row for _, _, row in read_jsonl(session_dir / "lifecycle.jsonl")]
    ends = [t for t in (epoch(r.get("at")) for r in lifecycle if r.get("name") == "end") if t is not None]
    start = number(session.get("start_epoch")) or 0.0
    s = Session(dir=session_dir, session=session, meta=meta, lessons=lessons, owners=Owners(lessons, ends),
                run_id=f"{session.get('identity') or session_dir.name}@{int(start)}", format=fmt)
    _own_rows(s)
    _own_trace(s)
    directions = as_dict(read_json(session_dir / "directions.json")).get("directions") or []
    s.directions = {d.get("id") for d in directions if isinstance(d, dict) and d.get("id")}
    s.log = read_campaign_logs(campaign_logs)
    return s


def _own_rows(s: Session) -> None:
    for index, row in enumerate(s.session["lessons"]):
        if isinstance(row, dict):
            owner = s.owners.by_iteration(row.get("iteration"), epoch(row.get("ts")))
            if owner is not None:
                s.own_lessons[owner].append({"index": index, "row": row})
    for line, _, row in read_jsonl(s.dir / "ledger.jsonl"):
        owner = s.owners.by_iteration(row.get("iteration"), epoch(row.get("ts")))
        if owner is not None:
            s.own_ledger[owner].append({"line": line, "row": row})


def _own_trace(s: Session) -> None:
    s.events = read_jsonl(s.dir / "reports" / "trace" / "events.jsonl")
    traces = _trace_sessions(s.events)
    owner_of = {sid: s.owners.by_time(trace.t0) for sid, trace in traces.items()}
    for sid, trace in traces.items():
        if owner_of[sid] is not None:
            s.own_traces[owner_of[sid]].append(trace)
    for line, raw, event in s.events:
        sid = event.get("claude_session_id")
        owner = owner_of.get(sid) if sid else s.owners.by_iteration(event.get("iteration"), epoch(event.get("ts")))
        if owner is not None:
            s.own_events[owner].append((line, raw))
    for item in s.meta.get("claude_sessions") or []:
        if isinstance(item, dict):
            sid = item.get("session_id")
            seen = traces[sid].t0 if sid in traces else epoch(item.get("first_seen"))
            owner = s.owners.by_time(seen)
            if owner is not None:
                s.own_claude[owner].append(item)


def session_format(session_dir: Path) -> int:
    """The session's format as Arbor declares it: its latest checkpoint's ``session_format``, 0 without one."""

    from arbor import checkpoints

    return checkpoints.session_format(session_dir)


def arbor_version() -> dict:
    """This Arbor's release and its session interface, which the export records beside what it read."""

    from arbor import version

    return {"arbor_version": version.VERSION, "arbor_session_interface": version.INTERFACES.get("session")}


# ---------------------------------------------------------------- the pack copy
def find_pack(name: str, session_dir: Path, pack_roots: list[Path]) -> Path | None:
    """The pack copy ``name``: under a ``--pack-root``, else in an Arbor checkout beside the session directory."""

    roots = [root.resolve() for root in pack_roots]
    here = session_dir.parent
    for _ in range(3):
        roots += [here / "Arbor" / "packs", here / "packs"]
        here = here.parent
    return next((root / name for root in roots if name and (root / name / "pack.yaml").is_file()), None)


def pack_links(pack: Path, digests: Any) -> tuple[list[Link], str]:
    """The pack copy's files when every validator file the session pinned matches it; else why not."""

    if not isinstance(digests, dict) or not digests:
        return [], "the session pinned no validator digests to match a pack copy against"
    marker = f"/{pack.name}/"
    for pinned, sha in digests.items():
        rel = pinned.split(marker, 1)[1] if marker in pinned else None
        if rel is None or not (pack / rel).is_file() or sha256_file(pack / rel) != sha:
            return [], f"the pack copy {pack} does not hold {pinned} as the session pinned it"
    links = [Link(p, f"{pack.name}/{p.relative_to(pack).as_posix()}", "pack") for p in sorted(pack.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts]
    return links, ""


def pack_measurement(pack: Path) -> dict:
    """The pack copy's ``objective.measurement`` block, ``{}`` when it cannot be read."""

    try:
        import yaml  # Arbor's pack loader needs it; without it the gate settings are left out.

        doc = yaml.safe_load((pack / "pack.yaml").read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return {}
    return as_dict(as_dict(as_dict(doc).get("objective")).get("measurement"))
