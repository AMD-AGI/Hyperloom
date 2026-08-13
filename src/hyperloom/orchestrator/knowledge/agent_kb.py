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

``refs`` comes back positionally — one slot per file passed, empty when that
artifact could not be staged — so a caller that needs a ref inside its own
metadata writes twice: once to stage the files, once with the refs folded into
the payload. A caller that does not can ignore the return
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

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .remote_recipe._vendor.kb_store_client import (
    FILES_MEMBER_ROOT,
    KBStoreError,
    KnowledgeSections,
    _same_bytes,
)

log = logging.getLogger(__name__)

KERNEL_SECTION = "kernel"
KERNEL_COLUMNS = ("gemm", "fusion", "rewrite")
EXPLORE_SECTION = "explore"
FRAMEWORK_SECTION = "framework"

_PATCH_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class _ConfigPatchAgentKB:
    """Section facade for one config owner and its ordered source overlays."""

    SECTION = ""

    def __init__(self, sections: KnowledgeSections | None) -> None:
        self._sections = sections

    @classmethod
    def open(cls):
        """Open this run's draft, staying inactive when there is none."""
        try:
            sections = KnowledgeSections.from_env()
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("%s kb: draft unavailable: %s", cls.SECTION, exc)
            sections = None
        return cls(sections)

    @property
    def active(self) -> bool:
        return self._sections is not None

    def read(self) -> dict[str, Any]:
        """Return the prior complete section, or ``{}`` on a cold start."""
        if self._sections is None:
            return {}
        try:
            content = self._sections.read(self.SECTION)
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("%s kb: cannot read section: %s", self.SECTION, exc)
            return {}
        return dict(content.knowledge) if content is not None else {}

    def read_config(self) -> dict[str, Any]:
        """Return the prior replay config using the stable public field names."""
        prior = self.read()
        return {
            "extra_server_args": str(prior.get("extra_server_args") or ""),
            "extra_envs": (
                dict(prior.get("extra_envs") or {})
                if isinstance(prior.get("extra_envs"), Mapping)
                else {}
            ),
        }

    def write_config(
        self,
        extra_server_args: str | Mapping[str, Any] = "",
        extra_envs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Replace this owner's final config/env snapshot, preserving patches.

        A current-best-like mapping is accepted as a convenience for writeback;
        callers may also pass the two fields directly.
        """
        if self._sections is None:
            return False
        if isinstance(extra_server_args, Mapping):
            snapshot = extra_server_args
            args = str(
                snapshot.get("effective_extra_server_args")
                or snapshot.get("extra_server_args")
                or ""
            )
            raw_envs = snapshot.get("extra_envs")
            envs = dict(raw_envs) if isinstance(raw_envs, Mapping) else {}
        else:
            args = str(extra_server_args or "")
            envs = dict(extra_envs or {})
        try:
            staged = self._sections.staged(self.SECTION)
            document = dict(staged.knowledge) if staged is not None else {}
            document["extra_server_args"] = args
            document["extra_envs"] = {
                str(key): str(value) for key, value in envs.items()
            }
            self._sections.write(self.SECTION, document, mode="replace")
        except (KBStoreError, OSError, TypeError, ValueError) as exc:
            log.warning("%s kb: cannot record config: %s", self.SECTION, exc)
            return False
        return True

    write_snapshot = write_config

    def stage_patches(
        self,
        patches: Iterable[str | Path],
        *,
        stack_index: int,
    ) -> list[str]:
        """Stage one KEEP's patch members in caller order.

        Repeating the same call returns the same refs and leaves one physical
        copy. The complete set is rejected when any member cannot be staged.
        """
        if self._sections is None:
            return []
        try:
            index = int(stack_index)
        except (TypeError, ValueError):
            log.warning("%s kb: invalid stack index %r", self.SECTION, stack_index)
            return []
        if index < 0:
            log.warning("%s kb: negative stack index %r", self.SECTION, stack_index)
            return []

        requested = list(patches)
        if len(requested) > 100:
            log.warning(
                "%s kb: refusing %d patch members; maximum is 100",
                self.SECTION,
                len(requested),
            )
            return []
        prepared: list[tuple[Path, str, bytes]] = []
        for member_index, source in enumerate(requested):
            src = Path(str(source or ""))
            stem = src.name
            for suffix in (".patch", ".diff"):
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            safe_name = _PATCH_NAME_RE.sub("-", stem).strip("._-") or "patch"
            ref = (
                f"{self.SECTION}/overlays/{index:06d}/"
                f"{member_index:02d}-{safe_name}.patch"
            )
            try:
                if src.is_symlink() or not src.is_file():
                    raise KBStoreError(
                        f"artifact is not a readable regular file: {src}"
                    )
                content = src.read_bytes()
                destination = self._sections.files_dir / ref
                if destination.exists() and destination.read_bytes() != content:
                    raise KBStoreError(
                        f"artifact ref already has different bytes: {ref}"
                    )
                prepared.append((src, ref, content))
            except (KBStoreError, OSError, ValueError) as exc:
                log.warning(
                    "%s kb: atomic patch staging rejected %s: %s",
                    self.SECTION,
                    source,
                    exc,
                )
                return []
        refs = [ref for _src, ref, _content in prepared]
        if not refs:
            return []
        created: list[Path] = []
        try:
            staged = self._sections.staged(self.SECTION)
            document = dict(staged.knowledge) if staged is not None else {}
            recorded = [
                str(ref)
                for ref in (document.get("patches") or [])
                if str(ref).strip()
            ]
            for ref in refs:
                if ref not in recorded:
                    recorded.append(ref)
            document["patches"] = sorted(
                recorded,
                key=lambda ref: (
                    0 if "/overlays/" in ref else 1,
                    ref,
                ),
            )
            existing_files = (
                [
                    path.relative_to(self._sections.files_dir).as_posix()
                    for path in staged.files
                ]
                if staged is not None
                else []
            )
            all_files = list(dict.fromkeys([*existing_files, *refs]))
            for _src, ref, content in prepared:
                destination = self._sections.files_dir / ref
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_name(f".{destination.name}.tmp")
                temp.write_bytes(content)
                os.replace(temp, destination)
                created.append(destination)
            target = (
                self._sections.root
                / "sections"
                / f"{self.SECTION}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_section = target.with_name(f".{target.name}.tmp")
            temp_section.write_text(
                json.dumps(
                    {"knowledge": document, "files": all_files},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temp_section, target)
        except (KBStoreError, OSError, ValueError) as exc:
            for destination in created:
                destination.unlink(missing_ok=True)
            log.warning(
                "%s kb: cannot atomically record patch set: %s",
                self.SECTION,
                exc,
            )
            return []
        return refs

    def prior_file(self, ref: str) -> Path | None:
        """Resolve a prior section ref to its downloaded artifact."""
        if self._sections is None or self._sections.warm_start_dir is None:
            return None
        rel = str(ref or "").strip().lstrip("/")
        parts = Path(rel).parts if rel else ()
        if (
            not parts
            or parts[0] != self.SECTION
            or ".." in parts
            or Path(rel).is_absolute()
        ):
            return None
        candidate = self._sections.warm_start_dir / FILES_MEMBER_ROOT / rel
        return candidate if candidate.is_file() else None


class ExploreAgentKB(_ConfigPatchAgentKB):
    """Read/write EXPLORE's final config and accepted source overlays."""

    SECTION = EXPLORE_SECTION


class FrameworkAgentKB(_ConfigPatchAgentKB):
    """Read/write FRAMEWORK_AGENT's final config and accepted overlays."""

    SECTION = FRAMEWORK_SECTION


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
        # Positional: a caller folds these back into its own metadata by index,
        # so a failed stage keeps an empty placeholder rather than shifting every
        # later artifact one slot up.
        refs = [self._stage(column, source) for source in files]
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
            self._sections.write(
                KERNEL_SECTION,
                staged.knowledge if staged is not None else {},
                files=[source],
                kind=column,
                mode="merge",
            )
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("kernel kb: cannot stage %s for %s: %s", source, column, exc)
            return ""
        return self._staged_ref(column, source)

    def _staged_ref(self, column: str, source: str | Path) -> str:
        """Name the copy this artifact just landed as, by the draft's own rule.

        The draft keeps ``{section}/{kind}/{name}`` and only falls back to a
        digest-suffixed name when that path is already taken by different bytes.
        Deriving the ref from the files on disk — rather than from whether the
        staged list happened to grow — is what keeps a re-staged artifact from
        being handed the ref of a same-named neighbour.
        """
        src = Path(str(source))
        files_dir = self._sections.files_dir
        plain = f"{KERNEL_SECTION}/{column}/{src.name}"
        landed = files_dir / plain
        if landed.is_file() and _same_bytes(src, landed):
            return plain
        digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:10]
        suffixed = f"{KERNEL_SECTION}/{column}/{src.stem}-{digest}{src.suffix}"
        return suffixed if (files_dir / suffixed).is_file() else ""


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
    "EXPLORE_SECTION",
    "ExploreAgentKB",
    "FRAMEWORK_SECTION",
    "FrameworkAgentKB",
    "KERNEL_COLUMNS",
    "KERNEL_SECTION",
    "KernelAgentKB",
    "KernelRecordReader",
]
