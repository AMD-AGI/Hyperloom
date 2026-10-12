# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prompt manifests, prefix-break detection, and context-compaction rows for the trajectory ledger.

A manifest describes a composed prompt by its ``=== <title> ===`` sections (size, token estimate, content hash), never
by its text. Diffing it against the same agent's previous prompt shows how much of the prompt a provider prefix cache
could still reuse (the longest common prefix) and which section broke it first. The system prompt and tool list sit
ahead of the prompt in the cached prefix, so a change to either breaks the whole prefix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .trajectory_trace import EVENT_CONTEXT_COMPACTION, EVENT_PROMPT_SNAPSHOT, record_event

log = logging.getLogger(__name__)

PREAMBLE = "(preamble)"
# The coarse chars-per-token ratio; the manifest's token counts are estimates, the provider's usage is the truth.
CHARS_PER_TOKEN = 4
COMPACT_BOUNDARY = "compact_boundary"

_SECTION_RE = re.compile(r"^=== (.+?) ===$", re.MULTILINE)
# A trailing ``(...)`` carries per-render detail (counts, "newest last"); the section is the same without it.
_SECTION_DETAIL_RE = re.compile(r"\s*\([^()]*\)$")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tokens(chars: int) -> int:
    return -(-chars // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class PromptSection:
    """One ``=== <title> ===`` block of a prompt (the heading line included)."""

    key: str
    title: str
    chars: int
    sha: str


def prompt_sections(prompt: str) -> list[PromptSection]:
    """Split ``prompt`` at its section headings; text before the first heading is the preamble."""
    starts = [(m.start(), m.group(1)) for m in _SECTION_RE.finditer(prompt)]
    bounds = [(0, PREAMBLE), *starts] if not starts or starts[0][0] > 0 else starts
    sections: list[PromptSection] = []
    for index, (start, title) in enumerate(bounds):
        end = bounds[index + 1][0] if index + 1 < len(bounds) else len(prompt)
        text = prompt[start:end]
        key = _SECTION_DETAIL_RE.sub("", title).strip() or title
        sections.append(PromptSection(key=key, title=title, chars=len(text), sha=_sha(text)))
    return sections


def _common_prefix_chars(a: str, b: str) -> int:
    return len(os.path.commonprefix([a, b]))


def _section_at(sections: list[PromptSection], offset: int) -> str | None:
    position = 0
    for section in sections:
        position += section.chars
        if offset < position:
            return section.key
    return None


def _tools_sha(tools: Iterable[str] | None) -> str | None:
    return None if tools is None else _sha(json.dumps(sorted(str(t) for t in tools)))


@dataclass
class _Snapshot:
    prompt: str
    sections: list[PromptSection]
    system_sha: str | None
    tools_sha: str | None


class PromptSnapshotTracker:
    """Per-agent memory of the previous prompt, so each snapshot can say what changed since."""

    def __init__(self) -> None:
        self._previous: dict[str, _Snapshot] = {}

    def observe(
        self,
        agent: str,
        *,
        prompt: str,
        system_prompt: str | None,
        tools: Iterable[str] | None,
    ) -> dict[str, Any]:
        """Return the ``prompt.snapshot`` attributes for ``agent``'s next prompt and remember it."""
        sections = prompt_sections(prompt)
        system_sha = None if system_prompt is None else _sha(system_prompt)
        tool_list = None if tools is None else list(tools)
        tools_sha = _tools_sha(tool_list)
        attributes: dict[str, Any] = {
            "name": agent,
            "prompt_chars": len(prompt),
            "prompt_tokens_est": _tokens(len(prompt)),
            "prompt_sha": _sha(prompt),
            "system_prompt_chars": None if system_prompt is None else len(system_prompt),
            "system_prompt_sha": system_sha,
            "tools_count": None if tool_list is None else len(tool_list),
            "tools_sha": tools_sha,
            "sections": [
                {"key": s.key, "chars": s.chars, "tokens_est": _tokens(s.chars), "sha": s.sha} for s in sections
            ],
        }
        previous = self._previous.get(agent)
        self._previous[agent] = _Snapshot(prompt, sections, system_sha, tools_sha)
        if previous is None:
            attributes["first_prompt"] = True
            return attributes
        attributes.update(_diff(previous, prompt, sections, system_sha=system_sha, tools_sha=tools_sha))
        return attributes


def _diff(
    previous: _Snapshot,
    prompt: str,
    sections: list[PromptSection],
    *,
    system_sha: str | None,
    tools_sha: str | None,
) -> dict[str, Any]:
    before = {s.key: s.sha for s in previous.sections}
    after = {s.key: s.sha for s in sections}
    lcp = _common_prefix_chars(previous.prompt, prompt)
    identical = previous.prompt == prompt
    system_changed = system_sha != previous.system_sha
    tools_changed = tools_sha != previous.tools_sha
    return {
        "first_prompt": False,
        "lcp_chars": lcp,
        "lcp_tokens_est": _tokens(lcp),
        "lcp_ratio": round(lcp / len(prompt), 4) if prompt else 1.0,
        "first_changed_section": None if identical else _section_at(sections, lcp),
        "changed_sections": [key for key, sha in after.items() if key in before and before[key] != sha],
        "added_sections": [key for key in after if key not in before],
        "removed_sections": [key for key in before if key not in after],
        "system_prompt_changed": system_changed,
        "tools_changed": tools_changed,
        "prefix_break": system_changed or tools_changed,
    }


def record_prompt_snapshot(attributes: dict[str, Any]) -> str | None:
    """Append one ``prompt.snapshot`` point row (a no-op outside a trajectory scope)."""
    return record_event(EVENT_PROMPT_SNAPSHOT, attributes=attributes)


def compaction_attributes(data: Any) -> dict[str, Any]:
    """Attributes of a ``compact_boundary`` system message, whether its metadata is nested or flat."""
    data = data if isinstance(data, dict) else {}
    metadata = data.get("compact_metadata") if isinstance(data.get("compact_metadata"), dict) else data
    return {
        "name": str(metadata.get("trigger") or "compaction"),
        "trigger": metadata.get("trigger"),
        "pre_tokens": metadata.get("pre_tokens"),
    }


def stream_json_compactions(log_path: str | Path) -> list[dict[str, Any]]:
    """The ``compact_boundary`` rows of a Claude CLI stream-json log, in order."""
    found: list[dict[str, Any]] = []
    try:
        with Path(log_path).open("r", encoding="utf-8") as f:
            for line in f:
                if COMPACT_BOUNDARY not in line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict) and obj.get("type") == "system" and obj.get("subtype") == COMPACT_BOUNDARY:
                    found.append(obj)
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("context_events: failed reading stream-json log %s: %r", log_path, exc)
        return []
    return found


def record_stream_json_compactions(log_path: str | Path) -> int:
    """Put every compaction of a finished stream-json log on the trajectory (untimed); returns rows written."""
    written = 0
    for obj in stream_json_compactions(log_path):
        span_id = record_event(
            EVENT_CONTEXT_COMPACTION,
            attributes={**compaction_attributes(obj), "timing_source": "none"},
        )
        written += span_id is not None
    return written


__all__ = [
    "CHARS_PER_TOKEN",
    "COMPACT_BOUNDARY",
    "PREAMBLE",
    "PromptSection",
    "PromptSnapshotTracker",
    "compaction_attributes",
    "prompt_sections",
    "record_prompt_snapshot",
    "record_stream_json_compactions",
    "stream_json_compactions",
]
