# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local knowledge loader for the forge-loop.

Assembles the layered knowledge block injected into the agent system prompt for
one kernel-optimization task. The block is built from the curated
``local_knowledge/`` tree in reading order:

  1. ``hardware/`` and ``common_methodology/`` — always (mandatory background).
  2. ``framework/aiter/`` — only when the target is an AITER-framework operator.
  3. ``languages/<language>/`` — the kernel's implementation language.

Each level is loaded per the KernelForge INDEX convention: a folder that has an
``INDEX.md`` is navigated through it and that map is loaded WHOLE; a folder
without one falls back to a flat ``<relative path> — <one-line descriptor>``
listing. Full card content stays on disk and is fetched with the ``Read`` tool
on demand (progressive disclosure).

Design goals:
  * The block is generated LIVE from the directory tree at prompt-build time, so
    adding, removing, or retitling a file needs NO code change.
  * The per-file descriptor (flat-listing fallback) is auto-extracted from the
    file itself (a fallback chain), never a hand-maintained table — so
    descriptions stay in sync.

Descriptor fallback chain (first hit wins) — every source is mined from the file
itself, so descriptions stay in sync with no hand-maintained table:
  * .py : first non-empty line of the module docstring
  1. first sentence of a ``## TL;DR`` section
  2. YAML front-matter ``description:`` (folded ``>`` scalars supported)
  3. YAML front-matter ``title:``
  4. the intro blockquote (``> ...`` right under the H1 — the guide pattern)
  5. first ``# H1`` heading
  6. first ``## H2`` heading
  7. first prose line
  8. the file stem
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from kernelforge.resources import resource_path

# local_knowledge/ lives at the repo root in source checkouts and under
# kernelforge/data in built wheels.
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
    """Collapse whitespace to one line; end on a full sentence when possible.

    Prefers a complete first sentence; only appends '…' when a single sentence
    genuinely exceeds ``limit`` (so descriptions are not cut mid-thought).
    """
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


def _descriptor(path: Path) -> str:
    """One-line descriptor for a file via the fallback chain (see module doc)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.stem
    lines = text.splitlines()

    # .py — the module docstring is the best summary.
    if path.suffix.lower() == ".py":
        d = _py_docstring(text)
        return _clip(d) if d else path.stem

    # 1. TL;DR — first non-empty line under the heading (strip blockquote '>').
    for i, ln in enumerate(lines):
        if _TLDR_RE.match(ln.strip()):
            for nxt in lines[i + 1 : i + 6]:
                s = nxt.strip().lstrip(">").strip()
                if s:
                    return _clip(s)
            break

    # 2 / 3. front-matter description (preferred), else title.
    if lines and lines[0].strip() == "---":
        fm: list[str] = []
        for ln in lines[1:]:
            if ln.strip() == "---":
                break
            fm.append(ln)
        for key in ("description", "title"):
            val = _frontmatter_field(fm, key)
            if val:
                return _clip(val)

    # 4. intro blockquote ('> ...' before the first '## ' section) — guide pattern.
    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            break
        if s.startswith(">"):
            q = s.lstrip(">").strip()
            if q and not q.lower().startswith("**important"):
                return _clip(q)

    # 5 / 6. first H1, else first H2.
    for pat in (_H1_RE, _H2_RE):
        for ln in lines:
            m = pat.match(ln.strip())
            if m:
                return _clip(m.group(1).strip())

    # 7. first prose line (skip front-matter, headings, tables, code, quotes).
    in_fm = False
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if idx == 0 and s == "---":
            in_fm = True
            continue
        if in_fm:
            if s == "---":
                in_fm = False
            continue
        if s and not s.startswith(("#", ">", "|", "`", "---")):
            return _clip(s)

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


def _render_level(root: Path, rel: str) -> str:
    """Render one knowledge level as a titled section.

    Per the KernelForge convention: if the folder has an ``INDEX.md`` it is the
    navigation map and is loaded WHOLE; otherwise fall back to a flat
    ``<path> — <descriptor>`` listing of the folder's files. Returns "" when the
    folder is missing or empty.
    """
    folder = root / rel
    if not folder.is_dir():
        return ""
    header = f"## {rel}/  —  base: {folder}"
    index = folder / "INDEX.md"
    if index.is_file():
        try:
            body = index.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            body = ""
        if body:
            return f"{header}\n\n{body}"
    listing = _flat_listing(folder)
    if not listing:
        return ""
    return f"{header}\n\n{listing}"


def build_forge_knowledge(
    root: str | Path | None = None,
    *,
    language: str | Sequence[str] | None = None,
    include_aiter: bool = False,
    include_mori: bool = False,
) -> str:
    """Assemble the layered knowledge block for one forge-loop kernel task.

    Layers, in reading order (see module docstring):
      1. ``hardware/`` + ``common_methodology/`` — always.
      2. ``framework/aiter/`` — only when ``include_aiter`` (an AITER operator).
      3. ``framework/mori/`` — only when ``include_mori`` (experimental,
         ablation-only knob; off by default — see ``config.include_mori_kb``).
      4. ``languages/<language>/`` — when ``language`` is given and its folder
         exists.

    ``language`` accepts a sequence, rendered in the order given, for a backend
    served by more than one language folder (triton/gluon are one toolchain and
    carry each other; see ``kernel_backends.constants.resolve_language_dirs``).
    Duplicates collapse so the same folder is never rendered twice.

    Each level is loaded per the INDEX.md convention (whole INDEX.md if present,
    else a flat file listing). Returns "" if the root or all levels are missing.
    """
    root_path = Path(root) if root else _DEFAULT_ROOT
    if not root_path.exists():
        return ""

    rels: list[str] = list(_MANDATORY_PILLARS)
    if include_aiter:
        rels.append("framework/aiter")
    if include_mori:
        rels.append("framework/mori")
    languages = [language] if isinstance(language, str) else list(language or ())
    for name in dict.fromkeys(item for item in languages if item):
        rels.append(f"languages/{name}")

    sections = [s for s in (_render_level(root_path, rel) for rel in rels) if s]
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
