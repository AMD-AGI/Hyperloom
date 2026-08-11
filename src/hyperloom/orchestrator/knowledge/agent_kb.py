# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel agent's own columns of this run's knowledge document.

This module defines the handoff surface only. GEMM, fusion and rewrite
production call sites are intentionally owned by their agent implementations
and are not wired by the Remote Recipe migration itself.

The document keeps one column per producer, and the kernel column keeps one
sub-column per kernel backend::

    "kernel": {"gemm": {...}, "fusion": {...}, "rewrite": {...}}

Each backend hands over the complete picture of its sub-column plus the files
that belong to it, and gets back the refs to record for those files::

    kb = KernelAgentKB.open()
    prior = kb.read_rewrite()
    refs = kb.write_rewrite({"items": [...]}, files=[patch_path])

``refs`` comes back in the order the files were passed, so a caller that needs
a ref inside its own metadata writes twice: once to stage the files, once with
the refs folded into the payload. A caller that does not can ignore the return
value, because the same refs are recorded under the column's ``files`` key.

The write replaces the sub-column: what an agent hands over is the whole
picture of that sub-column, not a patch against the last call. Files already
staged for that sub-column remain referenced when a later write supplies no new
files. Sibling sub-columns and every other column in the document are left
untouched, so a section-aware backend and a not-yet-migrated one publish into
one record.

Wiring is unconditional. A run with no draft directory -- local knowledge
mode, or a run that never publishes -- leaves the facade inactive and turns
every call into a no-op, so a caller never has to branch on whether the KB is
on. Nothing here raises into the agent: knowledge is advisory, and a failure
to record must not fail an optimization.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .remote_recipe._vendor.kb_store_client import (
    FILES_MEMBER_ROOT,
    KBStoreError,
    KnowledgeSections,
)

log = logging.getLogger(__name__)

KERNEL_SECTION = "kernel"
KERNEL_COLUMNS = ("gemm", "fusion", "rewrite")


class KernelAgentKB:
    """Read and write the ``gemm``/``fusion``/``rewrite`` sub-columns."""

    def __init__(self, sections: KnowledgeSections | None) -> None:
        self._sections = sections

    @classmethod
    def open(cls) -> "KernelAgentKB":
        """Open this run's draft, staying inactive when there is none."""
        try:
            sections = KnowledgeSections.from_env()
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("kernel kb: draft unavailable: %s", exc)
            sections = None
        return cls(sections)

    @property
    def active(self) -> bool:
        """True when this run collects sections and the calls below do something."""
        return self._sections is not None

    # -- read ----------------------------------------------------------------

    def read_gemm(self) -> dict[str, Any]:
        """Return the prior ``gemm`` sub-column, ``{}`` on a cold start."""
        return self._read("gemm")

    def read_fusion(self) -> dict[str, Any]:
        """Return the prior ``fusion`` sub-column, ``{}`` on a cold start."""
        return self._read("fusion")

    def read_rewrite(self) -> dict[str, Any]:
        """Return the prior ``rewrite`` sub-column, ``{}`` on a cold start."""
        return self._read("rewrite")

    def prior_file(self, ref: str) -> Path | None:
        """Resolve a ref recorded in prior knowledge to its downloaded file."""
        if self._sections is None:
            return None
        warm_start_dir = self._sections.warm_start_dir
        if warm_start_dir is None:
            return None
        rel = str(ref or "").strip().lstrip("/")
        parts = Path(rel).parts if rel else ()
        if not parts or ".." in parts or Path(rel).is_absolute():
            return None
        candidate = warm_start_dir / FILES_MEMBER_ROOT / rel
        return candidate if candidate.is_file() else None

    # -- write ---------------------------------------------------------------

    def write_gemm(
        self,
        knowledge: Mapping[str, Any],
        files: Iterable[str | Path] = (),
    ) -> list[str]:
        """Record the complete ``gemm`` sub-column and its files."""
        return self._write("gemm", knowledge, files)

    def write_fusion(
        self,
        knowledge: Mapping[str, Any],
        files: Iterable[str | Path] = (),
    ) -> list[str]:
        """Record the complete ``fusion`` sub-column and its files."""
        return self._write("fusion", knowledge, files)

    def write_rewrite(
        self,
        knowledge: Mapping[str, Any],
        files: Iterable[str | Path] = (),
    ) -> list[str]:
        """Record the complete ``rewrite`` sub-column and its files."""
        return self._write("rewrite", knowledge, files)

    # -- internals -----------------------------------------------------------

    def _read(self, column: str) -> dict[str, Any]:
        if self._sections is None:
            return {}
        try:
            content = self._sections.read(KERNEL_SECTION)
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("kernel kb: cannot read %s: %s", column, exc)
            return {}
        if content is None:
            return {}
        node = content.knowledge.get(column)
        return dict(node) if isinstance(node, Mapping) else {}

    def _write(
        self,
        column: str,
        knowledge: Mapping[str, Any],
        files: Iterable[str | Path],
    ) -> list[str]:
        if self._sections is None:
            return []
        if not isinstance(knowledge, Mapping):
            log.warning("kernel kb: %s knowledge must be a mapping", column)
            return []
        refs = [ref for source in files if (ref := self._stage(column, source))]
        payload = dict(knowledge)
        try:
            staged = self._sections.staged(KERNEL_SECTION)
            prefix = f"{KERNEL_SECTION}/{column}/"
            column_refs = (
                sorted(
                    path.relative_to(self._sections.files_dir).as_posix()
                    for path in staged.files
                    if path.relative_to(self._sections.files_dir)
                    .as_posix()
                    .startswith(prefix)
                )
                if staged is not None
                else []
            )
            if column_refs:
                payload["files"] = column_refs
            document = dict(staged.knowledge) if staged is not None else {}
            document[column] = payload
            self._sections.write(KERNEL_SECTION, document, mode="replace")
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("kernel kb: cannot record %s: %s", column, exc)
            return []
        return refs

    def _stage(self, column: str, source: str | Path) -> str:
        """Copy one artifact into the draft and return the ref that names it."""
        try:
            staged = self._sections.staged(KERNEL_SECTION)
            before = len(staged.files) if staged is not None else 0
            content = self._sections.write(
                KERNEL_SECTION,
                staged.knowledge if staged is not None else {},
                files=[source],
                kind=column,
                mode="merge",
            )
            staged_refs = [
                path.relative_to(self._sections.files_dir).as_posix()
                for path in content.files
            ]
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("kernel kb: cannot stage %s for %s: %s", source, column, exc)
            return ""
        if len(staged_refs) > before:
            return staged_refs[-1]
        # Re-staging a byte-identical artifact adds no ref; reuse the existing one.
        expected = f"{KERNEL_SECTION}/{column}/{Path(str(source)).name}"
        return expected if expected in staged_refs else ""


class KernelRecordReader:
    """Read a downloaded independent kernel-agent KB record (``kernel:`` scheme).

    The recipe warm-start nests kernel columns under ``value.kernel.*`` and is
    read through :class:`KernelAgentKB`. The independent kernel-agent record
    instead stores them flat at ``value.gemm``/``value.fusion``/``value.rewrite``
    with a sibling ``files/`` tree. This reader exposes the same read surface
    (``active`` + ``read_gemm``/``read_fusion``/``read_rewrite`` + ``prior_file``)
    so the PRELUDE warm-apply planner can consume either source unchanged.

    ``record_dir`` is a directory populated by ``read_remote_recipe`` — a
    ``recipe.json`` plus a ``files/`` tree. An absent/empty record leaves the
    reader inactive and every call a no-op.
    """

    def __init__(self, record_dir: str | Path | None) -> None:
        self._dir = Path(record_dir) if record_dir else None
        self._value: dict[str, Any] | None = None
        if self._dir is not None:
            recipe = self._dir / "recipe.json"
            if recipe.is_file():
                try:
                    document = json.loads(recipe.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    document = None
                if isinstance(document, Mapping):
                    value = document.get("value")
                    self._value = dict(value) if isinstance(value, Mapping) else {}

    @property
    def active(self) -> bool:
        """True when a record was loaded and its columns can be read."""
        return isinstance(self._value, Mapping)

    def _column(self, name: str) -> dict[str, Any]:
        node = (self._value or {}).get(name)
        return dict(node) if isinstance(node, Mapping) else {}

    def read_gemm(self) -> dict[str, Any]:
        return self._column("gemm")

    def read_fusion(self) -> dict[str, Any]:
        return self._column("fusion")

    def read_rewrite(self) -> dict[str, Any]:
        return self._column("rewrite")

    def prior_file(self, ref: str) -> Path | None:
        """Resolve a recorded ref to its downloaded file under ``files/``."""
        if self._dir is None:
            return None
        rel = str(ref or "").strip().lstrip("/")
        parts = Path(rel).parts if rel else ()
        if not parts or ".." in parts or Path(rel).is_absolute():
            return None
        candidate = self._dir / FILES_MEMBER_ROOT / rel
        return candidate if candidate.is_file() else None


__all__ = [
    "KERNEL_COLUMNS",
    "KERNEL_SECTION",
    "KernelAgentKB",
    "KernelRecordReader",
]
