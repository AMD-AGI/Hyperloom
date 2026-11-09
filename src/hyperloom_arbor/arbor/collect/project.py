"""Project names: one name per model family, whatever a campaign calls the model.

The share lays campaigns out as ``projects/<project>/<user>/<campaign>/``, so
every campaign on one model has to land in one ``<project>``. Campaigns name the
model many ways (pack ``dsr1-fp4``, session ``dsr1-c32-run2``, model
``deepseek-ai/DeepSeek-R1``); ``projects.json`` maps them to one name. A name it does not know becomes a slug
of itself, and the collector says so, so a new family shows up as its own
project until someone adds it to the table.

The table here holds public model families. A site's own (a model not yet public, a customer's) go in a file of the
same form that ``ARBOR_PROJECTS`` names, in the environment or in ``~/.arbor/settings.env``; it is read first.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECTS_FILE = Path(__file__).with_name("projects.json")

# Every path component on the share: no spaces, quotes or slashes, so a
# component can sit unquoted in a remote command or an sftp batch.
SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9.]+", "-", text.strip().lower())
    return re.sub(r"-{2,}", "-", s).strip("-.")


def load_table(path: str | Path | None = None) -> list[tuple[str, re.Pattern]]:
    """The table: a site's own projects first (the file ``ARBOR_PROJECTS`` names, in ``projects.json``'s form), then
    the public model families of ``projects.json``. ``path`` reads that one file alone."""
    if path:
        files = [Path(path)]
    else:
        site = Path(os.environ.get("ARBOR_PROJECTS", "").strip()).expanduser()
        files = ([site] if str(site) not in ("", ".") and site.is_file() else []) + [PROJECTS_FILE]
    table = []
    for f in files:
        table += [(p["project"], re.compile(p["match"])) for p in json.loads(f.read_text())["projects"]]
    return table


def match_project(text: str, table: list[tuple[str, re.Pattern]]) -> str | None:
    squashed = squash(text)
    for name, pattern in table:
        if pattern.search(squashed):
            return name
    return None


def resolve_project(explicit: str | None, candidates: list[tuple[str, str | None]],
                    table: list[tuple[str, re.Pattern]] | None = None) -> tuple[str, str]:
    """The project, and how it was chosen.

    ``explicit`` (a flag, ``ARBOR_PROJECT``, or the config's ``project``) wins,
    normalized through the table so "DeepSeek R1" and "dsr1" agree.
    Otherwise the first candidate, in order, that the table knows; otherwise a
    slug of the first non-empty candidate.
    """
    table = load_table() if table is None else table
    if explicit:
        return match_project(explicit, table) or slug(explicit), f"given ({explicit})"
    for label, text in candidates:
        if text:
            name = match_project(text, table)
            if name:
                return name, f"{label} ({text})"
    for label, text in candidates:
        if text and slug(text):
            return slug(text), f"{label} ({text}); no known project matched, see projects.json"
    return "unassigned", "nothing to name it from"


def component(text: str) -> str:
    """A safe share path component from any name, or ValueError."""
    s = slug(text)
    if not SAFE_COMPONENT.match(s):
        raise ValueError(f"cannot make a share path component from {text!r}")
    return s
