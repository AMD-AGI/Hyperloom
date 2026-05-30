"""Introspection — lessons learned, pitfalls, and research findings.

Persists knowledge from each optimization session so future sessions
(and future specialist agents) can avoid repeating mistakes and build
on proven techniques.

Storage: JSONL files in session_dir:
  - lessons.jsonl:           all lessons (kept and reverted)
  - research_findings.jsonl: research discoveries from scouts/agents
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Lesson:
    """A lesson learned from an optimization attempt."""

    title: str
    description: str
    outcome: str  # "kept" | "reverted" | "failed" | "neutral"
    evidence: str
    timestamp: str
    session_id: str = ""
    tags: list[str] | None = None

    @property
    def is_pitfall(self) -> bool:
        return self.outcome in ("reverted", "failed")


@dataclass
class ResearchFinding:
    """A research discovery from a scout or agent."""

    source: str
    topic: str
    summary: str
    relevance: str  # "high" | "medium" | "low"
    actionable: bool = True
    pr_url: str = ""
    timestamp: str = ""


def _append_jsonl(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


def _load_jsonl(path: str, limit: int = 0) -> list[dict]:
    if not os.path.exists(path):
        return []
    lines: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if limit > 0:
        return lines[-limit:]
    return lines


def save_lesson(session_dir: str, lesson: Lesson) -> None:
    """Persist a lesson to the session."""
    path = os.path.join(session_dir, "lessons.jsonl")
    _append_jsonl(path, asdict(lesson))


def load_lessons(session_dir: str, limit: int = 50) -> list[Lesson]:
    """Load all lessons from the session."""
    path = os.path.join(session_dir, "lessons.jsonl")
    data = _load_jsonl(path, limit)
    return [Lesson(**{k: v for k, v in d.items() if k in Lesson.__dataclass_fields__})
            for d in data]


def load_pitfalls(session_dir: str, limit: int = 20) -> list[Lesson]:
    """Load only pitfalls (reverted/failed lessons)."""
    all_lessons = _load_jsonl(os.path.join(session_dir, "lessons.jsonl"))
    pitfalls = [l for l in all_lessons if l.get("outcome") in ("reverted", "failed")]
    results = pitfalls[-limit:]
    return [Lesson(**{k: v for k, v in d.items() if k in Lesson.__dataclass_fields__})
            for d in results]


def save_research(session_dir: str, finding: ResearchFinding) -> None:
    """Persist a research finding."""
    path = os.path.join(session_dir, "research_findings.jsonl")
    _append_jsonl(path, asdict(finding))


def load_research(session_dir: str, limit: int = 30) -> list[ResearchFinding]:
    """Load research findings."""
    path = os.path.join(session_dir, "research_findings.jsonl")
    data = _load_jsonl(path, limit)
    return [ResearchFinding(**{k: v for k, v in d.items()
                               if k in ResearchFinding.__dataclass_fields__})
            for d in data]


def get_introspection_context(session_dir: str) -> dict[str, Any]:
    """Get full introspection context for prompt building."""
    return {
        "pitfalls": [asdict(p) for p in load_pitfalls(session_dir)],
        "research": [asdict(r) for r in load_research(session_dir)],
        "lessons_count": len(load_lessons(session_dir, limit=0)),
    }


def format_pitfalls_for_prompt(session_dir: str, max_items: int = 10) -> str:
    """Format pitfalls into a readable string for agent prompts."""
    pitfalls = load_pitfalls(session_dir, limit=max_items)
    if not pitfalls:
        return "No known pitfalls from this session yet."

    lines = ["## Known Pitfalls (do NOT repeat these):\n"]
    for p in pitfalls:
        lines.append(f"- **{p.title}** [{p.outcome}]: {p.description}")
        if p.evidence:
            lines.append(f"  Evidence: {p.evidence[:200]}")
    return "\n".join(lines)
