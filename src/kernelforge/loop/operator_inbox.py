# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Operator messages to a running forge-loop.

A message is one file in ``forge_experiments/inbox/``. The loop consumes the inbox at the start of each
iteration, moving each message to ``inbox/archive/``. Every message archived since the last ``/clear`` is
standing guidance, so state is derived from disk and survives a resume unchanged.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

INBOX_DIRNAME = "inbox"
ARCHIVE_DIRNAME = "archive"
CLEAR = "/clear"
# Guidance rides in planning payloads and prompts; past this many characters the oldest messages drop out.
GUIDANCE_CHAR_LIMIT = 4000


@dataclass(frozen=True)
class OperatorMessage:
    """One message: the file name it was posted under and its text."""

    name: str
    text: str


def inbox_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "forge_experiments" / INBOX_DIRNAME


def post_message(workspace: str | Path, text: str) -> Path:
    """Write one message in a single rename, so the loop never reads half of it; return its path."""
    folder = inbox_dir(workspace)
    folder.mkdir(parents=True, exist_ok=True)
    # Nanoseconds first, so names sort in posting order even for messages posted within one second.
    name = f"{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex[:6]}.md"
    staging = folder / f".{name}.tmp"
    staging.write_text(text.strip() + "\n", encoding="utf-8")
    os.replace(staging, folder / name)
    return folder / name


def _posted_name(archived: Path) -> str:
    """The name a message was posted under: archived files carry an ``iter_NNN-`` prefix."""
    return archived.name.split("-", 1)[1] if archived.name.startswith("iter_") else archived.name


class OperatorInbox:
    """The inbox of one workspace."""

    def __init__(self, workspace: str | Path) -> None:
        self.folder = inbox_dir(workspace)
        self.archive = self.folder / ARCHIVE_DIRNAME

    def pending(self) -> list[Path]:
        """Messages posted and not yet consumed, oldest first; files being written (dotfiles) are skipped."""
        if not self.folder.is_dir():
            return []
        return sorted(
            path for path in self.folder.iterdir() if path.is_file() and path.suffix == ".md" and path.name[0] != "."
        )

    def consume(self, iteration: int) -> list[OperatorMessage]:
        """Move every pending message to the archive under this iteration; return them."""
        messages = []
        for path in self.pending():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            self.archive.mkdir(parents=True, exist_ok=True)
            os.replace(path, self.archive / f"iter_{iteration:03d}-{path.name}")
            messages.append(OperatorMessage(name=path.name, text=text))
        return messages

    def active_guidance(self) -> tuple[str, ...]:
        """Messages archived since the last ``/clear``, newest first, within ``GUIDANCE_CHAR_LIMIT``."""
        if not self.archive.is_dir():
            return ()
        texts = [
            path.read_text(encoding="utf-8", errors="replace").strip()
            for path in sorted(self.archive.glob("iter_*.md"), key=_posted_name)
        ]
        cleared = max((index for index, text in enumerate(texts) if text == CLEAR), default=-1)
        guidance, used = [], 0
        for text in reversed(texts[cleared + 1 :]):
            if used + len(text) > GUIDANCE_CHAR_LIMIT:
                break
            guidance.append(text)
            used += len(text)
        return tuple(guidance)
