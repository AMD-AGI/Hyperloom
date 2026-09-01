# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-column facades over this run's KB draft and its warm-start record.

A published Recipe carries three columns, and each one has exactly one facade:

``config``
    The cross-session server-args / env layer.
``patch``
    Source overlays in replay order (``patches``), plus one provenance row per
    overlay saying how it was captured (``provenance``). Nothing else: a report
    or a changed-file listing is not replay material.
``kernel``
    The ``gemm`` / ``fusion`` / ``rewrite`` sub-columns.

A facade both reads the prior record's column and stages this run's, so a
column's on-the-wire shape has exactly one owner. Staging is atomic per call:
either every member lands and the section document names it, or nothing is
written and the draft is left as it was.

Overlay refs are ``patch/overlays/<stack_index>/<member>-<name>.patch`` with
both indices zero-padded, so the lexicographic order of ``patch.patches`` *is*
the replay order and there is no separate timeline to keep in step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .remote_recipe._vendor.kb_store_client import (
    FILES_MEMBER_ROOT,
    KBStoreError,
    KnowledgeSections,
)
from .remote_recipe.models import (
    CONFIG_SECTION,
    KERNEL_SECTION,
    PATCH_SECTION,
    RECIPE_SECTIONS,
)
from .remote_recipe.sanitize import HOST_ORIGIN_KEY

log = logging.getLogger(__name__)

_SECTIONS_MEMBER = "sections"
_MEMBER_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
#: Ceiling on one KEEP's overlay set. A larger set is a caller bug.
_MAX_OVERLAY_MEMBERS = 100


def _safe_member_name(name: str) -> str:
    """Reduce an artifact basename to the charset a ref may carry."""
    stem = str(name or "")
    for suffix in (".patch", ".diff"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return _MEMBER_NAME_RE.sub("-", stem).strip("._-") or "patch"


def _prior_member(
    sections: KnowledgeSections | None,
    ref: str,
    *,
    owner: str,
) -> Path | None:
    """Resolve one downloaded member without following symlinks."""
    if sections is None or sections.warm_start_dir is None:
        return None
    rel = str(ref or "").strip().lstrip("/")
    parts = Path(rel).parts if rel else ()
    if not parts or parts[0] != owner or ".." in parts or Path(rel).is_absolute():
        return None
    root = sections.warm_start_dir / FILES_MEMBER_ROOT
    if root.is_symlink():
        return None
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return None
    try:
        cursor.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return cursor if cursor.is_file() else None


class DraftArtifactSink:
    """Collect local files into one column's staged member set.

    Mirrors the ref layout the CLOSE-time upload sink produces
    (``<category>/<kind>/<basename>``) so a value builder does not care whether
    it is writing into a draft or straight into the upload tree. Bytes are held
    in memory until the owning facade commits them, which is what keeps a
    staging call atomic.
    """

    def __init__(self) -> None:
        self._members: dict[str, bytes] = {}
        self._sources: dict[tuple[Path, str, str], str] = {}

    @property
    def members(self) -> list[tuple[str, bytes]]:
        """Staged ``(ref, content)`` pairs in insertion order."""
        return list(self._members.items())

    def add(self, source: Any, *, category: str, kind: str, name: str = "") -> str:
        """Return the ref ``source`` will occupy, or ``""`` when unusable.

        A missing file yields ``""`` so an optional artifact does not fail the
        whole column; a symlink or an oversized file raises, because those are
        contract violations rather than absences. Raises the same error type the
        CLOSE-time upload sink does, so a value builder has one failure mode.
        """
        from .remote_recipe.models import MAX_FILE_BYTES, RemoteRecipeValidationError

        raw = str(source or "").strip()
        if not raw:
            return ""
        src = Path(raw)
        if src.is_symlink():
            raise RemoteRecipeValidationError(f"artifact source must not be a symlink: {src}")
        if not src.is_file():
            return ""
        if src.stat().st_size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(f"artifact {src} exceeds the {MAX_FILE_BYTES}-byte KB Store limit")
        resolved = src.resolve()
        source_key = (resolved, str(category), str(kind))
        if source_key in self._sources:
            return self._sources[source_key]
        basename = Path(name or src.name).name or "artifact"
        ref = f"{category}/{kind}/{basename}"
        content = src.read_bytes()
        if self._members.get(ref, content) != content:
            digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:10]
            ref = f"{category}/{kind}/{src.stem}-{digest}{src.suffix}"
        self._members[ref] = content
        self._sources[source_key] = ref
        return ref

    def write(self, text: str, *, rel: str, kind: str) -> str:
        """Stage a document the builder synthesised rather than found on disk."""
        del kind  # the ref already carries the column and kind
        self._members[rel] = text.encode("utf-8")
        return rel

    def discard(self, ref: str) -> None:
        """Undo a just-added ref so a skipped item leaves no staged orphan.

        A kernel KEEP whose checkout cannot be named is dropped from the Recipe
        rather than aborting the whole publish, but its patch was already staged
        to reserve the ref. Without removing it the column's members would carry
        a file the published value no longer references, which the section
        mismatch guard rejects.
        """
        self._members.pop(ref, None)
        self._sources = {key: value for key, value in self._sources.items() if value != ref}


class _ColumnKB:
    """One published column: read the prior record, stage this run's."""

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
        """True when this run has a draft to read priors through and stage into."""
        return self._sections is not None

    # -- read ----------------------------------------------------------------

    def prior(self) -> dict[str, Any]:
        """This column as the warm-start record carries it, ``{}`` on a cold start."""
        if self._sections is None:
            return {}
        try:
            content = self._sections.read(self.SECTION)
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("%s kb: cannot read section: %s", self.SECTION, exc)
            return {}
        return dict(content.knowledge) if content is not None else {}

    def prior_file(self, ref: str) -> Path | None:
        """Resolve a ref this column recorded to its downloaded artifact."""
        return _prior_member(self._sections, ref, owner=self.SECTION)

    # -- write ---------------------------------------------------------------

    def _staged(self) -> tuple[dict[str, Any], list[str]]:
        """The document and member refs this run already staged for the column."""
        if self._sections is None:
            return {}, []
        staged = self._sections.staged(self.SECTION)
        if staged is None:
            return {}, []
        refs = [path.relative_to(self._sections.files_dir).as_posix() for path in staged.files]
        return dict(staged.knowledge), refs

    def _publish(
        self,
        document: Mapping[str, Any],
        *,
        members: Sequence[tuple[str, bytes]] = (),
    ) -> bool:
        """Commit ``document`` and any new members as one section update.

        A ref that is already staged with different bytes fails the whole call,
        so a ref never silently changes meaning. Files written before the
        failure are removed, leaving the draft as it was.
        """
        if self._sections is None:
            return False
        created: list[Path] = []
        try:
            _, staged_refs = self._staged()
            all_refs = list(dict.fromkeys([*staged_refs, *(ref for ref, _ in members)]))
            for ref, content in members:
                destination = self._sections.files_dir / ref
                if destination.exists():
                    if destination.read_bytes() != content:
                        raise KBStoreError(f"artifact ref already has different bytes: {ref}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_name(f".{destination.name}.tmp")
                temp.write_bytes(content)
                os.replace(temp, destination)
                created.append(destination)
            target = self._sections.root / _SECTIONS_MEMBER / f"{self.SECTION}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_section = target.with_name(f".{target.name}.tmp")
            temp_section.write_text(
                json.dumps(
                    {"knowledge": dict(document), "files": all_refs},
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
            log.warning("%s kb: cannot atomically stage the column: %s", self.SECTION, exc)
            return False
        return True


class ConfigKB(_ColumnKB):
    """The cumulative server-args / env layer, published as one final value."""

    SECTION = CONFIG_SECTION

    def read(self) -> dict[str, Any]:
        """Return the prior replay config under the stable public field names."""
        prior = self.prior()
        envs = prior.get("extra_envs")
        return {
            "extra_server_args": str(prior.get("extra_server_args") or ""),
            "extra_envs": dict(envs) if isinstance(envs, Mapping) else {},
        }

    def stage(self, config: Mapping[str, Any]) -> bool:
        """Stage the final config layer, replacing whatever was staged before.

        The authority is the session's accepted stack, so this is published once
        from the settled value rather than accumulated per KEEP.
        """
        envs = config.get("extra_envs")
        return self._publish(
            {
                "extra_server_args": str(config.get("extra_server_args") or ""),
                "extra_envs": dict(envs) if isinstance(envs, Mapping) else {},
            }
        )


class PatchKB(_ColumnKB):
    """Source overlays in replay order plus their source-layer snapshots."""

    SECTION = PATCH_SECTION

    # -- read ----------------------------------------------------------------

    def read_patches(self) -> list[str]:
        """Return the prior overlay refs in replay order."""
        patches = self.prior().get("patches")
        if not isinstance(patches, list):
            return []
        return [str(ref) for ref in patches if isinstance(ref, str) and str(ref).strip()]

    def read_provenance(self) -> list[dict[str, Any]]:
        """Return each overlay's capture provenance, ordered by stack index.

        Metadata only: it says how trustworthy the overlay beside it is, not
        what the overlay contains. ``complete`` is the one a consumer must
        honour -- a capture that could not account for every path its patch
        claimed to touch may have produced an incomplete overlay.
        """
        rows = self.prior().get("provenance")
        if not isinstance(rows, list):
            return []
        provenance = [dict(row) for row in rows if isinstance(row, Mapping)]
        provenance.sort(key=lambda row: _stack_index(row.get("stack_index")))
        return provenance

    # -- write ---------------------------------------------------------------

    def stage_patches(
        self,
        patches: Iterable[str | Path],
        *,
        stack_index: int,
    ) -> list[str]:
        """Stage one KEEP's overlay members in caller order.

        Repeating the same call returns the same refs and leaves one physical
        copy. The complete set is rejected when any member cannot be staged.
        """
        if self._sections is None:
            return []
        index = _stack_index(stack_index)
        if index < 0:
            log.warning("%s kb: unusable stack index %r", self.SECTION, stack_index)
            return []

        requested = list(patches)
        if len(requested) > _MAX_OVERLAY_MEMBERS:
            log.warning(
                "%s kb: refusing %d overlay members; maximum is %d",
                self.SECTION,
                len(requested),
                _MAX_OVERLAY_MEMBERS,
            )
            return []

        members: list[tuple[str, bytes]] = []
        for member_index, source in enumerate(requested):
            src = Path(str(source or ""))
            ref = f"{self.SECTION}/overlays/{index:06d}/{member_index:02d}-{_safe_member_name(src.name)}.patch"
            try:
                if src.is_symlink() or not src.is_file():
                    raise KBStoreError(f"artifact is not a readable regular file: {src}")
                members.append((ref, src.read_bytes()))
            except (KBStoreError, OSError, ValueError) as exc:
                log.warning("%s kb: overlay staging rejected %s: %s", self.SECTION, source, exc)
                return []
        if not members:
            return []

        document, _ = self._staged()
        recorded = [str(ref) for ref in (document.get("patches") or []) if str(ref).strip()]
        for ref, _content in members:
            if ref not in recorded:
                recorded.append(ref)
        # Zero-padded indices make the lexicographic order the replay order.
        document["patches"] = sorted(recorded)
        if not self._publish(document, members=members):
            return []
        return [ref for ref, _content in members]

    def stage_provenance(
        self,
        *,
        stack_index: int,
        base_sha: str = "",
        complete: bool = True,
        artifacts_outside_root: int = 0,
        realized: bool = True,
        host_origin: Mapping[str, Any] | None = None,
    ) -> bool:
        """Record how the overlay at ``stack_index`` was captured.

        Metadata only, no files. Restaging the same index replaces its row, so a
        retried handoff cannot accumulate duplicates.

        Args:
            stack_index: The KEEP whose overlay this describes.
            base_sha: The framework sha the capture ran against. Provenance for
                a reader diagnosing a failed apply; it is not a gate, because a
                session commits every KEEP and so each capture's base is the
                previous KEEP's session-local commit.
            complete: False when the capture could not account for every path
                its patch claimed to touch, which means the overlay may be
                incomplete too.
            artifacts_outside_root: How many applied artifacts landed outside the
                framework root and are therefore in no overlay at all. A nonzero
                count means this KEEP's gain is not fully reproducible from the
                record.
            realized: True when the overlay is the diff the KEEP actually landed;
                False when it fell back to the patch as delivered.
            host_origin: Absolute paths on the producing host. ``apply_roots``
                maps each overlay ref to the checkout it was applied into, which
                is what a later session replays against; the rest records where
                the patch, snapshot and manifest were written, for reading a
                record back that would not replay.
        """
        if self._sections is None:
            return False
        index = _stack_index(stack_index)
        if index < 0:
            log.warning("%s kb: unusable stack index %r", self.SECTION, stack_index)
            return False
        row = {
            "stack_index": index,
            "base_sha": str(base_sha or ""),
            "complete": bool(complete),
            "artifacts_outside_root": max(0, int(artifacts_outside_root or 0)),
            "realized": bool(realized),
        }
        if origin := _host_origin(host_origin):
            row[HOST_ORIGIN_KEY] = origin
        document, _ = self._staged()
        rows = [
            dict(existing)
            for existing in (document.get("provenance") or [])
            if isinstance(existing, Mapping) and _stack_index(existing.get("stack_index")) != index
        ]
        rows.append(row)
        rows.sort(key=lambda existing: _stack_index(existing.get("stack_index")))
        document["provenance"] = rows
        return self._publish(document)


class KernelAgentKB(_ColumnKB):
    """The ``gemm`` / ``fusion`` / ``rewrite`` sub-columns."""

    SECTION = KERNEL_SECTION

    #: Fields naming a local source tree. They describe where a result came
    #: from on one machine, so they never travel to a shared record.
    _BLOCKED_READ_KEYS = frozenset(
        {
            "source_file",
            "source_files",
            "target_file",
            "target_files",
            "target_path",
        }
    )

    # -- read ----------------------------------------------------------------

    def read_gemm(self) -> dict[str, Any]:
        """Return the prior ``gemm`` sub-column, ``{}`` on a cold start."""
        return self._read_column("gemm")

    def read_fusion(self) -> dict[str, Any]:
        """Return the prior ``fusion`` sub-column, ``{}`` on a cold start."""
        return self._read_column("fusion")

    def read_rewrite(self) -> dict[str, Any]:
        """Return the prior ``rewrite`` sub-column, ``{}`` on a cold start."""
        return self._read_column("rewrite")

    def _read_column(self, column: str) -> dict[str, Any]:
        node = self.prior().get(column)
        if not isinstance(node, Mapping):
            return {}

        def without_source_metadata(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key): without_source_metadata(nested)
                    for key, nested in value.items()
                    if str(key) not in self._BLOCKED_READ_KEYS
                }
            if isinstance(value, list):
                return [without_source_metadata(item) for item in value]
            return value

        return without_source_metadata(node)

    # -- write ---------------------------------------------------------------

    def stage_from_state(self, state: Any, *, kernel_optimizer: str) -> bool:
        """Stage all three sub-columns from the session's accepted stack.

        A ``geak`` run publishes the columns empty: its kernels live in the
        GEAK-owned record, and duplicating them here would let two records
        disagree about the same kernel.
        """
        from .remote_recipe.values import (
            build_kernel_fusion_value,
            build_kernel_gemm_value,
            build_kernel_rewrite_value,
        )

        if self._sections is None:
            return False
        if str(kernel_optimizer or "").strip().lower() == "geak":
            return self._publish(
                {
                    "gemm": {"optimizations": []},
                    "fusion": {"items": []},
                    "rewrite": {"items": []},
                }
            )
        # A builder that cannot materialize an accepted kernel still raises, and
        # that must reach CLOSE. A KEEP whose checkout cannot be named is the one
        # exception: the builder drops just that item and publishes the rest, so
        # one unrooted kernel no longer takes config, patch, and the other kernels
        # down with it -- publishing it rootless would instead poison the whole
        # combined replay.
        sink = DraftArtifactSink()
        document = {
            "gemm": build_kernel_gemm_value(state, sink),
            "fusion": build_kernel_fusion_value(state, sink),
            "rewrite": build_kernel_rewrite_value(state, sink),
        }
        return self._publish(document, members=sink.members)


def _host_origin(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only the absolute paths of a host-origin record.

    ``apply_roots`` maps each overlay ref to the checkout it was applied into, so
    a Recipe whose overlays came from more than one tree stays replayable: the
    ref carries its own answer and nothing has to reconcile them.

    A relative value is dropped. Anywhere but ``apply_roots`` that would be a
    session-local artifact ref, which the column's own ``patches`` list already
    carries, and recording it twice would invite a reader to treat this subtree
    as replay material rather than as provenance.
    """
    if not isinstance(raw, Mapping):
        return {}
    origin: dict[str, Any] = {}
    for key in ("snapshot", "manifest"):
        value = str(raw.get(key) or "").strip()
        if value.startswith("/"):
            origin[key] = value
    sources = [str(path).strip() for path in (raw.get("sources") or []) if str(path or "").strip().startswith("/")]
    if sources:
        origin["sources"] = sources
    apply_roots = {
        str(ref).strip(): str(root).strip()
        for ref, root in (raw.get("apply_roots") or {}).items()
        if str(ref or "").strip() and str(root or "").strip().startswith("/")
    }
    if apply_roots:
        origin["apply_roots"] = apply_roots
    return origin


def _stack_index(value: Any) -> int:
    """Coerce a recorded stack index, or ``-1`` when it is unusable."""
    try:
        index = int(value)
    except (TypeError, ValueError):
        return -1
    return index if index >= 0 else -1


__all__ = [
    "CONFIG_SECTION",
    "KERNEL_SECTION",
    "PATCH_SECTION",
    "RECIPE_SECTIONS",
    "ConfigKB",
    "DraftArtifactSink",
    "KernelAgentKB",
    "PatchKB",
]
