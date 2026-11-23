# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Section-scoped staging of one Recipe KB knowledge document, shared by agents through a directory."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hyperloom_common.kb_store import FILES_MEMBER_ROOT, KBStoreError

#: Where a sectioned document keeps its per-section maps. The service treats
#: ``knowledge`` as opaque, so this is a producer-side convention rather than
#: part of the record schema; it is fixed because documents already in the
#: store are written under this key.
SECTION_ROOT = "value"

_DRAFT_DIR_ENV = "KB_DRAFT_DIR"
_WARM_START_DIR_ENV = "KB_WARM_START_DIR"
_SECTIONS_MEMBER = "sections"
_RECIPE_MEMBER = "recipe.json"
_ENCODING = "utf-8"
_MERGE = "merge"


def _checked_section(name: str) -> str:
    """Reject a section name that would escape its subtree or collide oddly."""
    section = str(name or "").strip()
    if not section:
        raise KBStoreError("section name is required")
    if section != section.strip("."):
        raise KBStoreError(f"section {name!r} may not start or end with a dot")
    bad = set(section) & set("/\\\0")
    if bad or section in (".", ".."):
        raise KBStoreError(f"section {name!r} may not contain a path separator")
    return section


class SectionContent:
    """One section's knowledge map plus the local files that belong to it."""

    __slots__ = ("files", "knowledge", "section")

    def __init__(
        self,
        section: str,
        knowledge: dict[str, Any],
        files: list[Path] | None = None,
    ) -> None:
        self.section = section
        self.knowledge = knowledge
        self.files = list(files or [])

    def __repr__(self) -> str:
        return f"SectionContent(section={self.section!r}, keys={sorted(self.knowledge)!r}, files={len(self.files)})"


class KnowledgeSections:
    """Section-scoped staging for one knowledge document, backed by a directory.

    A producer is usually several processes: agents that each own one section
    and a publisher that uploads once at the end. They cannot share a client
    object, so the draft is a directory that both sides open by path.

    Layout under ``root``::

        sections/<section>.json     one section's staged knowledge map
        files/<section>/<kind>/...  that section's artifacts, ready for put_dir

    ``files`` is laid out exactly as ``put_dir`` expects, so publishing is
    ``put_dir(cid, sid, sections.files_dir)`` with no repacking.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        warm_start_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.warm_start_dir = Path(warm_start_dir) if warm_start_dir else None

    @classmethod
    def from_env(cls) -> KnowledgeSections | None:
        """Open the draft an orchestrator prepared, or ``None`` when absent.

        Lets an agent stay agnostic about whether this run publishes at all:
        no draft directory means nobody is collecting, so skip the write.
        """
        draft = (os.environ.get(_DRAFT_DIR_ENV, "") or "").strip()
        if not draft:
            return None
        warm = (os.environ.get(_WARM_START_DIR_ENV, "") or "").strip()
        return cls(draft, warm_start_dir=warm or None)

    @property
    def files_dir(self) -> Path:
        """The subtree to hand to :meth:`KBStoreClient.put_dir`."""
        return self.root / FILES_MEMBER_ROOT

    # -- write ---------------------------------------------------------------

    def write(
        self,
        section: str,
        knowledge: dict[str, Any],
        *,
        files: Iterable[str | Path] = (),
        kind: str = "artifacts",
        mode: str = _MERGE,
    ) -> SectionContent:
        """Stage one section's knowledge map and copy its files into the draft.

        ``mode="merge"`` (the default) shallow-merges over what this section
        already staged and appends to its file list, so an agent that reports
        incrementally does not silently drop its earlier calls. ``"replace"``
        discards the staged map first; staged files always survive because
        they may already be referenced by the map being written.
        """
        name = _checked_section(section)
        if mode not in (_MERGE, "replace"):
            raise KBStoreError(f"mode must be 'merge' or 'replace', got {mode!r}")
        if not isinstance(knowledge, dict):
            raise KBStoreError(f"knowledge for section {name!r} must be a dict")
        try:
            json.dumps(knowledge, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise KBStoreError(f"section {name!r} is not strict JSON: {exc}") from exc

        staged = self.staged(name)
        merged = dict(knowledge)
        refs: list[str] = []
        if staged is not None:
            refs = [path.relative_to(self.files_dir).as_posix() for path in staged.files]
            if mode == _MERGE:
                merged = {**staged.knowledge, **knowledge}

        added = [self._copy_in(name, source, kind) for source in files]
        for ref in added:
            if ref and ref not in refs:
                refs.append(ref)

        target = self.root / _SECTIONS_MEMBER / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"knowledge": merged, "files": refs}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding=_ENCODING,
        )
        return SectionContent(name, merged, [self.files_dir / ref for ref in refs])

    def _copy_in(self, section: str, source: str | Path, kind: str) -> str:
        raw = str(source or "").strip()
        if not raw:
            return ""
        src = Path(raw)
        if src.is_symlink():
            raise KBStoreError(f"artifact must not be a symlink: {src}")
        if not src.is_file():
            raise KBStoreError(f"artifact is not a readable file: {src}")
        safe_kind = _checked_section(kind)
        rel = f"{section}/{safe_kind}/{src.name}"
        destination = self.files_dir / rel
        if destination.exists() and not _same_bytes(src, destination):
            digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:10]
            rel = f"{section}/{safe_kind}/{src.stem}-{digest}{src.suffix}"
            destination = self.files_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(src.read_bytes())
        return rel

    # -- read ----------------------------------------------------------------

    def staged(self, section: str) -> SectionContent | None:
        """Read back what this draft already holds for ``section``."""
        name = _checked_section(section)
        target = self.root / _SECTIONS_MEMBER / f"{name}.json"
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding=_ENCODING))
        except (OSError, ValueError) as exc:
            raise KBStoreError(f"staged section {name!r} is unreadable: {exc}") from exc
        knowledge = payload.get("knowledge")
        refs = payload.get("files") or []
        return SectionContent(
            name,
            dict(knowledge) if isinstance(knowledge, dict) else {},
            [self.files_dir / str(ref) for ref in refs if str(ref).strip()],
        )

    def read(self, section: str) -> SectionContent | None:
        """Return ``section`` from the warm-start record, or ``None``.

        ``None`` means this run has no prior knowledge for the section: either
        nothing was downloaded, or the record predates the section. Callers
        should treat it as a cold start rather than an error.
        """
        name = _checked_section(section)
        if self.warm_start_dir is None:
            return None
        recipe = self.warm_start_dir / _RECIPE_MEMBER
        if not recipe.is_file():
            return None
        try:
            document = json.loads(recipe.read_text(encoding=_ENCODING))
        except (OSError, ValueError) as exc:
            raise KBStoreError(f"warm start record is unreadable: {exc}") from exc
        value = document.get(SECTION_ROOT)
        knowledge = (value or {}).get(name) if isinstance(value, dict) else None
        if not isinstance(knowledge, dict):
            return None
        root = self.warm_start_dir / FILES_MEMBER_ROOT / name
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
        return SectionContent(name, dict(knowledge), files)

    def sections(self) -> list[str]:
        """Every section staged in this draft, in a stable order."""
        root = self.root / _SECTIONS_MEMBER
        if not root.is_dir():
            return []
        return sorted(path.stem for path in root.glob("*.json"))

    def document(self) -> dict[str, Any]:
        """The staged ``{section: knowledge}`` map to publish under ``value``."""
        return {name: (self.staged(name) or SectionContent(name, {})).knowledge for name in self.sections()}


def _same_bytes(left: Path, right: Path) -> bool:
    try:
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False


__all__ = ["SECTION_ROOT", "KnowledgeSections", "SectionContent"]
