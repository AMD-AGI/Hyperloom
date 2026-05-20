"""Knowledge-base priors reader for framework_optimize.

Read-only counterpart of :mod:`kb_write` (PR-I). Reads the
``framework_optimization`` partition lessons + boundary rules + pitfall
markers + perf priors so the LLM patch proposer can ground its
proposals on past KEEP / REVERT outcomes (design §6.4 / §Appendix A).

PR-G P3 scope:

* Read the 8 seed entries shipped with PR-I (when present).
* Read any session-appended lessons (KEEP outcomes from prior runs in
  the same KB root).
* Return a flat list of :class:`KbEntry` records sorted by relevance:
  pitfall markers first (LLM must avoid these), boundary rules
  second, perf priors third. Within each category, entries with a
  ``target_framework`` matching the active framework rank above
  cross-framework entries.

When the KB directory is missing / empty / unreadable, an empty list
is returned -- the LLM proposer treats absence of priors as a soft
signal (proposals still work, just without seed wisdom).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


log = logging.getLogger(__name__)


_PARTITION_NAME = "framework_optimization"
_SEEDS_SUBDIR = "seeds"
_LESSONS_FILE = "empirical_kb.md"


@dataclass(frozen=True)
class KbEntry:
    """One framework_optimization KB record.

    ``category`` distinguishes seed lineage:

    * ``perf``     -- baseline known-good optimisation (vllm chunked
      prefill, sglang radix tree eviction, PagedAttention block size).
    * ``boundary`` -- "this lives in kernel-agent, not framework" rule
      (sampler kernel vs Python config split, etc.).
    * ``pitfall``  -- "this change tends to crash / regress" marker.
    * ``lesson``   -- session-appended KEEP outcome (PR-I write path).
    """

    entry_id: str
    title: str
    body: str
    category: Literal["perf", "boundary", "pitfall", "lesson"]
    target_framework: str = ""  # "" = cross-framework / both vllm + sglang
    tags: tuple[str, ...] = ()
    source_path: str = ""


_SEED_HEADER_RE = re.compile(
    r"^#\s+(fw-(?P<cat>perf|boundary|pitfall)-(?P<num>\d+))\s+(?P<title>.+)$",
)


def resolve_kb_root() -> Path | None:
    """Locate the framework-agent KB root.

    Priority: ``FRAMEWORK_AGENT_KB_DIR`` env > ``FRAMEWORK_AGENT_ROOT/kb``.
    Returns None when neither resolves to an existing directory.
    """
    env_kb = os.environ.get("FRAMEWORK_AGENT_KB_DIR", "").strip()
    if env_kb:
        p = Path(env_kb).expanduser()
        if p.is_dir():
            return p.resolve()
    fa_root = os.environ.get("FRAMEWORK_AGENT_ROOT", "").strip()
    if fa_root:
        candidate = Path(fa_root).expanduser() / "kb"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _parse_seed_file(path: Path) -> KbEntry | None:
    """Parse a single seed file. Markdown header carries the entry_id +
    title + category; body is the rest. Tags are first-line ``Tags:``
    + comma list; ``Framework:`` line carries target framework.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    m = _SEED_HEADER_RE.match(lines[0])
    if m is None:
        return None
    entry_id = m.group(1)
    title = m.group("title").strip()
    cat: Literal["perf", "boundary", "pitfall"] = m.group("cat")  # type: ignore[assignment]
    target_fw = ""
    tags: list[str] = []
    body_lines: list[str] = []
    for raw in lines[1:]:
        line = raw.rstrip()
        if line.startswith("Framework:"):
            target_fw = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Tags:"):
            tags = [
                t.strip() for t in line.split(":", 1)[1].split(",")
                if t.strip()
            ]
        else:
            body_lines.append(line)
    return KbEntry(
        entry_id=entry_id,
        title=title,
        body="\n".join(body_lines).strip(),
        category=cat,
        target_framework=target_fw,
        tags=tuple(tags),
        source_path=str(path),
    )


def _parse_lessons_file(path: Path) -> list[KbEntry]:
    """Parse the session-appended empirical_kb.md.

    Format (PR-I writes this):

        # fw-keep-<ts>-<hash>  KEEP: <framework> <one-line-summary>
        Framework: <fw>
        Source: <session>
        ...body...

    Returns a list of KbEntry objects (category='lesson').
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    entries: list[KbEntry] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("# fw-keep-"):
            if current:
                entry = _build_lesson(current)
                if entry is not None:
                    entries.append(entry)
            current = [line]
        elif current:
            current.append(line)
    if current:
        entry = _build_lesson(current)
        if entry is not None:
            entries.append(entry)
    # Inject source_path on the parsed entries.
    return [
        KbEntry(
            entry_id=e.entry_id, title=e.title, body=e.body,
            category=e.category, target_framework=e.target_framework,
            tags=e.tags, source_path=str(path),
        )
        for e in entries
    ]


_LESSON_HEADER_RE = re.compile(
    r"^#\s+(fw-keep-[a-z0-9-]+)\s+KEEP:\s*(?P<fw>\S+)?\s*(?P<title>.*)$",
)


def _build_lesson(block: list[str]) -> KbEntry | None:
    if not block:
        return None
    m = _LESSON_HEADER_RE.match(block[0])
    if m is None:
        return None
    target_fw = ""
    for line in block[1:]:
        if line.startswith("Framework:"):
            target_fw = line.split(":", 1)[1].strip().lower()
            break
    if not target_fw and m.group("fw"):
        target_fw = m.group("fw").lower()
    return KbEntry(
        entry_id=m.group(1),
        title=(m.group("title") or "").strip(),
        body="\n".join(block[1:]).strip(),
        category="lesson",
        target_framework=target_fw,
        tags=(),
    )


def read_priors(
    target_framework: str = "",
    *,
    kb_root: Path | None = None,
) -> list[KbEntry]:
    """Load all available priors for ``target_framework``.

    Returns ``[]`` when the KB partition is unavailable or empty --
    the LLM proposer treats this as "no priors" and falls back to
    generic patterns. Sorted: pitfall > boundary > perf > lesson;
    within category, framework-matching entries rank above
    cross-framework.
    """
    root = kb_root or resolve_kb_root()
    if root is None:
        log.debug("read_priors: KB root unresolved; returning []")
        return []
    partition = root / _PARTITION_NAME
    if not partition.is_dir():
        log.debug("read_priors: partition %s not found", partition)
        return []

    entries: list[KbEntry] = []
    seeds_dir = partition / _SEEDS_SUBDIR
    if seeds_dir.is_dir():
        for seed in sorted(seeds_dir.glob("*.md")):
            entry = _parse_seed_file(seed)
            if entry is not None:
                entries.append(entry)
    lessons_path = partition / _LESSONS_FILE
    if lessons_path.is_file():
        entries.extend(_parse_lessons_file(lessons_path))
    return _rank_priors(entries, target_framework)


_CATEGORY_RANK: dict[str, int] = {
    "pitfall":  0,  # avoid these first
    "boundary": 1,  # respect role boundaries
    "perf":     2,  # known-good patterns
    "lesson":   3,  # session-accumulated wisdom
}


def _rank_priors(
    entries: Iterable[KbEntry], target_framework: str,
) -> list[KbEntry]:
    """Sort priors: category rank > framework match > entry_id.

    Framework match tiers (lower wins):
      0 = exact match (target_framework == tfw)
      1 = cross-framework entry (target_framework == "")
      2 = mismatch (target_framework != tfw)
    """
    tfw = (target_framework or "").strip().lower()

    def _fw_rank(e: KbEntry) -> int:
        if e.target_framework == tfw:
            return 0
        if not e.target_framework:
            return 1
        return 2

    return sorted(entries, key=lambda e: (
        _CATEGORY_RANK.get(e.category, 99),
        _fw_rank(e),
        e.entry_id,
    ))


__all__ = ["KbEntry", "read_priors", "resolve_kb_root"]
