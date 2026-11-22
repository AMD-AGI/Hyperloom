# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign's insight list: what its sessions have established about the kernel, carried from one to the next."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path

from hyperloom_common.io import DURABLE, atomic_write_text
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout

log = logging.getLogger(__name__)

# The marker each entry's line begins with, in the record turn's reply and in the stored list.
INSIGHT_PREFIX = "INSIGHT:"

# How many entries the list holds. It is read in full by every session, so it stays a list of what bounds the search
# rather than a log of everything observed.
MAX_INSIGHTS = 12

# The longest one entry may be; a longer one is cut at this length.
MAX_INSIGHT_CHARS = 320

_INSIGHT_REQUEST = """\
After the record, maintain the campaign's insight list. It carries from session
to session what has been established about this kernel and about measuring it.
An entry is a fact with the measurement behind it. It is not a direction and not
advice, so the rules above still hold for it. The kinds worth an entry:

  - how to measure so that a number agrees with the driver: what a quick bench
    has to reproduce, and what made one disagree;
  - a bound you measured — a memory bandwidth, a launch cost, a register
    budget — stated as the number and how it was taken;
  - how the toolchain compiles this kernel where that constrains a change:
    what a construct lowers to, what it cannot express;
  - a correctness condition a change has to keep.

The list as it stands:
{current}

Write the list as it should stand after this session, one line per entry, each
beginning `{prefix}`. Keep an entry that still holds word for word. Rewrite one
that your measurements contradicted, and say what you measured. Add what this
session established. At most {limit} entries of one sentence each, the ones
that bound the most of the remaining search first; when the list is full, an
entry gives way only to one that bounds more. If this session established
nothing new, repeat the list unchanged. The list does not count toward the word
budget above.
"""


def insight_request(current: Sequence[str]) -> str:
    """What the record turn is asked to do with the list as it stands."""
    listed = "\n".join(f"  {INSIGHT_PREFIX} {entry}" for entry in current) if current else "  (empty)"
    return _INSIGHT_REQUEST.format(current=listed, prefix=INSIGHT_PREFIX, limit=MAX_INSIGHTS)


def _entry_of(line: str) -> str | None:
    """The entry a line carries, or None when it is not an insight line."""
    stripped = line.strip().lstrip("-*#` ").strip()
    if not stripped.upper().startswith(INSIGHT_PREFIX):
        return None
    # A reply may set the marker in bold or code, which leaves the closing marks at the head of the entry.
    return " ".join(stripped[len(INSIGHT_PREFIX) :].split()).strip("*` ")[:MAX_INSIGHT_CHARS].rstrip()


def split_insights(reply: str) -> tuple[str, list[str]]:
    """A record turn's reply as the record without its insight lines, and the entries those lines carried."""
    kept: list[str] = []
    entries: list[str] = []
    for line in (reply or "").splitlines():
        entry = _entry_of(line)
        if entry is None:
            kept.append(line)
        elif entry and entry not in entries:
            entries.append(entry)
    return "\n".join(kept).strip(), entries[:MAX_INSIGHTS]


class InsightStore:
    """The list on disk: one file, replaced whole by each record turn that returns one."""

    def __init__(self, workspace_dir: str) -> None:
        self.path: Path = ExperimentsLayout.of(workspace_dir).insights

    def read(self) -> list[str]:
        """The entries as they stand, oldest decision first; empty when no session has written any."""
        with contextlib.suppress(OSError):
            return [
                entry
                for entry in (_entry_of(line) for line in self.path.read_text(errors="replace").splitlines())
                if entry
            ]
        return []

    def replace(self, iteration: int, entries: Sequence[str]) -> bool:
        """Make ``entries`` the list. An empty reply leaves the list as it was: nothing a session omits is erased."""
        if not entries:
            return False
        body = [f"<!-- as of iteration {iteration} -->", *(f"{INSIGHT_PREFIX} {entry}" for entry in entries)]
        try:
            atomic_write_text(self.path, "\n".join(body) + "\n", options=DURABLE)
        except OSError as error:
            log.debug("insights: could not write %s: %s", self.path, error)
            return False
        return True

    def render_for_prompt(self) -> str:
        """The list as the next session reads it, or "" while it is empty."""
        entries = self.read()
        if not entries:
            return ""
        lines = [
            "## Campaign insights (cumulative)",
            "What earlier sessions established about this kernel and about measuring it, each with the measurement "
            "behind it. They describe what was measured, not what to do: build on them, and re-measure one before "
            "relying on it where your own result disagrees.",
            "",
            *(f"- {entry}" for entry in entries),
        ]
        return "\n".join(lines)
