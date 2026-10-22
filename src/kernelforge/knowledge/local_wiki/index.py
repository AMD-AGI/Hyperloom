# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local knowledge loader for the forge-loop."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from kernelforge.resources import resource_path

if TYPE_CHECKING:
    from kernelforge.knowledge.local_wiki.skills import ExternalSkill

# The stable logical name resolves to this package's resources in built wheels
# and to an operator override when one is configured.
_DEFAULT_ROOT = resource_path("local_knowledge")

# Only these extensions are indexed (docs + runnable skeletons/scripts).
_INDEX_EXT = {".md", ".py"}

_TLDR_RE = re.compile(r"^#{1,6}\s*TL;?DR\b", re.IGNORECASE)
_H1_RE = re.compile(r"^#\s+(.+)$")
_H2_RE = re.compile(r"^##\s+(.+)$")
_DOCSTRING_RE = re.compile(r'("""|\'\'\')(.*?)\1', re.DOTALL)
# .py comment lines to ignore when falling back (license/shebang boilerplate).
_PY_SKIP_COMMENT = re.compile(r"^#\s*(spdx-|copyright|!|-\*-|type:|noqa)", re.IGNORECASE)


def _clip(s: str, limit: int = 220) -> str:
    """Collapse whitespace to one line; end on a full sentence when possible."""
    s = re.sub(r"\s+", " ", s).strip().strip("*`").strip()
    # A complete first sentence, if it fits, reads best.
    dot = s.find(". ")
    if 0 <= dot <= limit:
        return s[: dot + 1]
    if len(s) <= limit:
        return s
    cut = s[:limit]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp >= 80 else cut).rstrip() + "…"


def _py_docstring(text: str) -> str:
    """First non-empty line of the module docstring, else first useful comment."""
    m = _DOCSTRING_RE.search(text)
    if m:
        for ln in m.group(2).splitlines():
            s = ln.strip()
            if s:
                return s
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#") and not _PY_SKIP_COMMENT.match(s):
            return s.lstrip("#").strip()
        if s and not s.startswith("#"):
            break  # reached code before any useful comment
    return ""


def _frontmatter_field(fm: list[str], key: str) -> str:
    """Value of a front-matter ``key:`` — supports inline and folded (``>``) form."""
    for i, ln in enumerate(fm):
        m = re.match(rf"^{key}:\s*(.*)$", ln)
        if not m:
            continue
        val = m.group(1).strip()
        if val and val not in (">", "|", ">-", "|-", ">+", "|+"):
            return val.strip("\"'")
        # folded scalar: gather the indented continuation lines.
        buf: list[str] = []
        for nxt in fm[i + 1 :]:
            if re.match(r"^\s+\S", nxt):
                buf.append(nxt.strip())
            elif nxt.strip() == "":
                continue
            else:
                break
        return " ".join(buf)
    return ""


def _tldr_line(lines: list[str]) -> str:
    """The first non-empty line under a TL;DR heading, blockquote marker stripped."""
    for index, line in enumerate(lines):
        if not _TLDR_RE.match(line.strip()):
            continue
        for following in lines[index + 1 : index + 6]:
            text = following.strip().lstrip(">").strip()
            if text:
                return text
        return ""
    return ""


def _frontmatter_summary(lines: list[str]) -> str:
    """The front matter's ``description``, else its ``title``."""
    if not lines or lines[0].strip() != "---":
        return ""
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        block.append(line)
    for key in ("description", "title"):
        value = _frontmatter_field(block, key)
        if value:
            return value
    return ""


def _intro_blockquote(lines: list[str]) -> str:
    """The guide pattern: a blockquote before the first ``## `` section."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            return ""
        if not stripped.startswith(">"):
            continue
        quoted = stripped.lstrip(">").strip()
        if quoted and not quoted.lower().startswith("**important"):
            return quoted
    return ""


def _first_heading(lines: list[str]) -> str:
    """The first H1, else the first H2."""
    for pattern in (_H1_RE, _H2_RE):
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                return match.group(1).strip()
    return ""


def _first_prose_line(lines: list[str]) -> str:
    """The first line that is prose: not front matter, a heading, a table, code or a quote."""
    in_frontmatter = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped and not stripped.startswith(("#", ">", "|", "`", "---")):
            return stripped
    return ""


# The fallback chain, in the order the module doc describes it.
_DESCRIPTOR_STRATEGIES = (
    _tldr_line,
    _frontmatter_summary,
    _intro_blockquote,
    _first_heading,
    _first_prose_line,
)


def _descriptor(path: Path) -> str:
    """One-line descriptor for a file via the fallback chain (see module doc)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.stem
    if path.suffix.lower() == ".py":
        # A module docstring is the best summary a Python file has, and the markdown chain says nothing about one.
        docstring = _py_docstring(text)
        return _clip(docstring) if docstring else path.stem
    lines = text.splitlines()
    for strategy in _DESCRIPTOR_STRATEGIES:
        found = strategy(lines)
        if found:
            return _clip(found)
    return path.stem


# Pillars every operator-optimization task must load, in reading order.
_MANDATORY_PILLARS = ("hardware", "common_methodology")


def _flat_listing(folder: Path) -> str:
    """Flat ``<relative path> — <descriptor>`` listing for a folder (INDEX-less fallback)."""
    files = sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _INDEX_EXT),
        key=lambda p: p.relative_to(folder).as_posix(),
    )
    out: list[str] = []
    for f in files:
        rel = f.relative_to(folder).as_posix()
        desc = _descriptor(f)
        out.append(f"- {rel} — {desc}" if desc else f"- {rel}")
    return "\n".join(out)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---`` YAML block from a knowledge map.

    The block carries title/kind/scope/updated -- metadata describing the file
    to whoever maintains the KB. It is not navigation: nothing downstream
    reads it, and an agent handed it learns nothing it cannot see from the H1
    on the next line. It is inlined into every implementer, specialist,
    orchestration and analysis prompt, so it is paid for once per map per
    session and then again on every turn that re-reads the prefix.
    """
    if not text.startswith("---"):
        return text
    lines = text.split("\n")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :]).lstrip("\n")
    # An opening fence with no close is not front matter; leave it alone
    # rather than swallow the whole map.
    return text


def _render_level(root: Path, rel: str) -> str:
    """Render one knowledge level as a titled section."""
    folder = root / rel
    if not folder.is_dir():
        return ""
    header = f"## {rel}/  —  base: {folder}"
    index = folder / "INDEX.md"
    if index.is_file():
        try:
            body = _strip_frontmatter(index.read_text(encoding="utf-8", errors="replace").strip()).strip()
        except OSError:
            body = ""
        if body:
            return f"{header}\n\n{body}"
    listing = _flat_listing(folder)
    if not listing:
        return ""
    return f"{header}\n\n{listing}"


def _render_pointer(root: Path, rel: str, *, carried: bool = True) -> str:
    """Render one knowledge level as a one-line pointer instead of its whole map.

    Used for the SECOND language a backend carries, and for every pillar when
    ``defer_all`` is set. Triton and Gluon carry each other so a campaign knows
    that switching is an available move rather than a different project -- but
    knowing the move exists needs the map's location, not its 2.7k-token body
    inlined ahead of every turn of every session. The pointer keeps the
    affordance and defers the map to a ``Read`` the agent makes only if it
    actually needs it.

    ``carried`` picks the wording: a level the backend merely carries is an
    available move ("if this task crosses into ..."), while a deferred pillar is
    the map itself and reads as one.
    """
    folder = root / rel
    if not folder.is_dir():
        return ""
    index = folder / "INDEX.md"
    if not index.is_file():
        # No map to defer; a flat listing is already short, so inline it.
        return _render_level(root, rel)
    title = ""
    try:
        body = _strip_frontmatter(index.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        body = ""
    for line in body.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break
    header = f"## {rel}/  —  base: {folder}"
    what = f" — {title}" if title else ""
    call = (
        f"`Read` `{index}` if this task crosses into `{rel}`."
        if carried
        else f"`Read` `{index}` for this pillar's map before opening any card under it."
    )
    return f"{header}\n\nMap not inlined{what}. {call}"


def _render_skill(skill: ExternalSkill) -> str:
    """Point at one external skill: where it is, what it covers, and how the paths it gives resolve."""
    lines = [f"## skill {skill.name}/  —  base: {skill.root}", ""]
    if skill.description:
        lines += [skill.description, ""]
    lines.append(
        f"`Read` `{skill.skill_file}` before planning or editing a kernel it covers; it routes to the cards "
        f"beneath it. A path it lists relative to its own folder resolves against the `base:` above, and a "
        f"sibling skill it links as `../<name>/SKILL.md` is `{skill.root.parent}/<name>/SKILL.md`."
    )
    if skill.repository_root is not None:
        lines.append(
            f"A path it gives relative to its repository resolves against `{skill.repository_root}`. A section "
            "it scopes to that repository applies only when this task lives in that repository."
        )
    return "\n".join(lines)


def build_forge_knowledge(
    root: str | Path | None = None,
    *,
    language: str | Sequence[str] | None = None,
    include_aiter: bool = False,
    include_mori: bool = False,
    defer_all: bool = False,
    skills: Sequence[ExternalSkill] = (),
) -> str:
    """Assemble the layered knowledge block for one forge-loop kernel task.

    Layers, in reading order: ``hardware/`` + ``common_methodology/`` always;
    ``framework/aiter/`` when ``include_aiter``; ``framework/mori/`` when
    ``include_mori`` (ablation-only, off by default); ``languages/<language>/``
    when given and present. ``language`` accepts a sequence (triton/gluon are one
    toolchain and carry each other); duplicates collapse. Each level follows the
    INDEX.md convention. Returns "" if the root or all levels are missing.

    ``defer_all`` renders EVERY level as a pointer rather than only the carried
    language -- what ``Config.defer_knowledge_maps`` (on by default) selects.
    Called directly, the maps stay inlined unless it is set.

    ``skills`` are folders outside the KB, each with a ``SKILL.md``; every one
    is rendered after the KB levels as a pointer to that file.
    """
    root_path = Path(root) if root else _DEFAULT_ROOT
    if not root_path.exists() and not skills:
        return ""

    rels: list[str] = list(_MANDATORY_PILLARS)
    if include_aiter:
        rels.append("framework/aiter")
    if include_mori:
        rels.append("framework/mori")
    languages = [language] if isinstance(language, str) else list(language or ())
    deferred: set[str] = set()
    for position, name in enumerate(dict.fromkeys(item for item in languages if item)):
        rel = f"languages/{name}"
        rels.append(rel)
        # The primary language is inlined whole; every language after it is the
        # one the backend merely carries, and is deferred to a pointer.
        if position:
            deferred.add(rel)
    if defer_all:
        deferred.update(rels)

    def render(rel: str) -> str:
        if rel not in deferred:
            return _render_level(root_path, rel)
        # A level deferred only because the backend merely carries it reads as an
        # available move; one deferred by the ablation is the pillar itself.
        return _render_pointer(root_path, rel, carried=not defer_all)

    sections = [s for s in (render(rel) for rel in rels) if s]
    sections += [_render_skill(skill) for skill in skills]
    if not sections:
        return ""

    preamble = "\n".join(
        [
            "# Knowledge base (maps for this task; full cards on disk — Read on demand)",
            "",
            f"Knowledge root (KB): {root_path}",
            "",
            "The curated knowledge maps for this kernel task are below. Open a card with the",
            "`Read` tool using an ABSOLUTE path — a bare relative path resolves against the",
            "kernel's working directory (NOT the KB) and will miss. Build the absolute path:",
            "- a path a map lists relative to its own folder (e.g. `overall/…`,",
            "  `skills/optimize/…`) → prepend that section's `base:` shown below;",
            "- a cross-reference written as `<pillar>/…` or `local_knowledge/<pillar>/…`",
            f"  (e.g. `hardware/…`, `framework/aiter/…`) → it lives at `{root_path}/<pillar>/…`.",
            "Read a card only when it is relevant — decide for yourself what to read.",
            "",
        ]
    )
    return preamble + "\n" + "\n\n".join(sections)
