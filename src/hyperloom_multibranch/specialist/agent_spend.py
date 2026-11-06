# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch's agent spends on its own LLM calls, read from the transcripts its Claude CLI keeps.

The agent calls its LLM itself, through the session's gateway variables, so no call of it passes through MB. A Claude
CLI, and an SDK that runs one, writes every assistant message with its model and usage to a transcript in
``<config dir>/projects/<working directory>/``, the directory named by the working directory with every character
other than a letter or a digit written as ``-``. The agent works under its branch's directory and its sandbox's, which
no other run of the session shares, so the transcripts of those directories are its own; the onboarding CLI keeps
none. Each read takes what the transcripts gained since the last one, counts each message id once and records it as
the branch's spend, one record per CLI session; what was read is kept in a file of the branch's, so a restarted run
counts nothing twice. The file moves past a session's messages only once their record is written: a failed record is
tried again at the next read, and a ledger write that fails after it counts that record twice rather than never. An
agent on another backend keeps no such transcript, and its spend stays unrecorded.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.blackboard.interfaces import UsageSink
from hyperloom_multibranch.contracts.blackboard.records import LlmUsage
from hyperloom_multibranch.contracts.blackboard.vocab import UsageSource
from hyperloom_multibranch.specialist.agent_onboarding import ENCODING
from hyperloom_multibranch.specialist.llm_usage import launch_cost

PROJECTS_DIR = "projects"
TRANSCRIPT_SUFFIX = ".jsonl"


def transcript_dir(config_dir: Path, cwd: Path) -> Path:
    """Where a Claude CLI run in ``cwd`` keeps its transcripts."""

    return config_dir / PROJECTS_DIR / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


@dataclass
class _Ledger:
    """What was read: each transcript's offset past its last whole line, and every message id counted."""

    offsets: dict[str, int] = field(default_factory=dict)
    counted: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> _Ledger:
        """The ledger ``path`` holds, empty when there is none; raise ``ValueError`` when it holds no ledger, which
        starting afresh would count every call again."""

        if not path.exists():
            return cls()
        try:
            document = json.loads(path.read_text(encoding=ENCODING))
            return cls(dict(document["offsets"]), set(document["counted"]))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} holds no spend ledger: {error!r}") from error

    def save(self, path: Path) -> None:
        document = {"offsets": self.offsets, "counted": sorted(self.counted)}
        written = path.with_name(f"{path.name}.tmp")
        written.write_text(json.dumps(document), encoding=ENCODING)
        os.replace(written, path)


@dataclass
class _Session:
    """What one CLI session's messages new to a read used, by model, and their ids."""

    by_model: dict[str, TokenCounts] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Read:
    """One read: the ledger as saved, the offsets past what the transcripts hold now, and each session's new spend."""

    ledger: _Ledger
    offsets: dict[str, int]
    sessions: dict[str, _Session]


@dataclass(frozen=True)
class AgentSpend:
    """The spend of ``branch``'s agent, whose CLI keeps its transcripts in ``config_dir`` and works under ``roots``;
    a call whose transcript names no model is priced as ``model``'s. ``ledger`` is the file of what was read."""

    config_dir: Path
    roots: tuple[Path, ...]
    ledger: Path
    branch: str
    model: str
    usage: UsageSink

    async def record(self) -> None:
        """Record what the agent's calls since the last read used."""

        read = await asyncio.to_thread(self._read)
        ledger = read.ledger
        for session, spent in read.sessions.items():
            total = sum(spent.by_model.values(), TokenCounts())
            cost = launch_cost(spent.by_model, self.model)
            await self.usage.record_llm_usage(LlmUsage(self.branch, UsageSource.SPECIALIST, cost, total, session))
            ledger.counted |= spent.ids
            await asyncio.to_thread(ledger.save, self.ledger)
        ledger.offsets = read.offsets
        await asyncio.to_thread(ledger.save, self.ledger)

    def _read(self) -> _Read:
        ledger = _Ledger.load(self.ledger)
        offsets = dict(ledger.offsets)
        sessions: dict[str, _Session] = {}
        seen = set(ledger.counted)
        for transcript in self._transcripts():
            for session, message_id, model, tokens in _messages(transcript, offsets):
                if message_id in seen:
                    continue
                seen.add(message_id)
                spent = sessions.setdefault(session, _Session())
                spent.ids.add(message_id)
                spent.by_model[model] = spent.by_model.get(model, TokenCounts()) + tokens
        return _Read(ledger, offsets, sessions)

    def written_at(self) -> float | None:
        """When the agent's CLI last wrote a transcript; ``None`` when it keeps none."""
        return newest_write(self._transcripts())

    def _transcripts(self) -> Iterable[Path]:
        projects = self.config_dir / PROJECTS_DIR
        if not projects.is_dir():
            return
        names = [transcript_dir(self.config_dir, root).name for root in self.roots]
        for directory in sorted(projects.iterdir()):
            if any(directory.name == name or directory.name.startswith(f"{name}-") for name in names):
                yield from sorted(directory.rglob(f"*{TRANSCRIPT_SUFFIX}"))


def newest_write(paths: Iterable[Path]) -> float | None:
    """The latest modification time among ``paths``; one removed while it is read counts as not there."""
    times = []
    for path in paths:
        try:
            times.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return max(times, default=None)


def _messages(transcript: Path, offsets: dict[str, int]) -> Iterable[tuple[str, str, str, TokenCounts]]:
    """The assistant messages of ``transcript``'s whole lines past its offset in ``offsets``, which moves past them."""

    key = str(transcript)
    with transcript.open("rb") as handle:
        handle.seek(offsets.get(key, 0))
        chunk = handle.read()
    whole = chunk[: chunk.rfind(b"\n") + 1]
    offsets[key] = offsets.get(key, 0) + len(whole)
    for line in whole.decode(ENCODING, errors="replace").splitlines():
        row = _row(line)
        message = row.get("message")
        if row.get("type") != "assistant" or not isinstance(message, Mapping):
            continue
        usage = message.get("usage")
        tokens = TokenCounts.from_anthropic_usage(usage) if isinstance(usage, Mapping) else TokenCounts()
        # A row the CLI writes itself, such as a gateway error (model "<synthetic>"), used nothing and has no price.
        if message.get("id") and tokens != TokenCounts():
            model = str(message.get("model") or "")
            yield (
                str(row.get("sessionId") or transcript.stem),
                str(message["id"]),
                model,
                TokenCounts.from_anthropic_usage(usage),
            )


def _row(line: str) -> Mapping[str, object]:
    if '"usage"' not in line:
        return {}
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return {}
    return row if isinstance(row, dict) else {}


__all__ = ["AgentSpend", "newest_write", "transcript_dir"]
