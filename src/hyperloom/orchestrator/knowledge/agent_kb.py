# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent facades for Recipe config, patch overlays, and prior Kernel columns."""

from __future__ import annotations

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
)

log = logging.getLogger(__name__)

KERNEL_SECTION = "kernel"
EXPLORE_SECTION = "explore"
FRAMEWORK_SECTION = "framework"

_PATCH_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _prior_member(
    sections: KnowledgeSections | None,
    ref: str,
    *,
    owner: str = "",
) -> Path | None:
    """Resolve one downloaded member without following symlinks."""
    if sections is None or sections.warm_start_dir is None:
        return None
    rel = str(ref or "").strip().lstrip("/")
    parts = Path(rel).parts if rel else ()
    if not parts or (owner and parts[0] != owner) or ".." in parts or Path(rel).is_absolute():
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
        """Return the prior section, or ``{}`` on a cold start or logged read error."""
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
            "extra_envs": (dict(prior.get("extra_envs") or {}) if isinstance(prior.get("extra_envs"), Mapping) else {}),
        }

    def read_patches(self) -> list[str]:
        """Return this owner's ordered patch refs from the prior Recipe."""
        prior = self.read()
        patches = prior.get("patches")
        if not isinstance(patches, list):
            return []
        return [str(ref) for ref in patches if isinstance(ref, str) and str(ref).strip()]

    def read_patch_roots(self) -> dict[str, str]:
        """Return ``{patch_ref: apply_root}``; empty for records predating the field."""
        prior = self.read()
        raw = prior.get("patch_roots")
        if not isinstance(raw, Mapping):
            return {}
        return {str(k): str(v) for k, v in raw.items() if str(k).strip() and str(v).strip()}

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
            ref = f"{self.SECTION}/overlays/{index:06d}/{member_index:02d}-{safe_name}.patch"
            try:
                if src.is_symlink() or not src.is_file():
                    raise KBStoreError(f"artifact is not a readable regular file: {src}")
                content = src.read_bytes()
                destination = self._sections.files_dir / ref
                if destination.exists() and destination.read_bytes() != content:
                    raise KBStoreError(f"artifact ref already has different bytes: {ref}")
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
            recorded = [str(ref) for ref in (document.get("patches") or []) if str(ref).strip()]
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
                [path.relative_to(self._sections.files_dir).as_posix() for path in staged.files]
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
            target = self._sections.root / "sections" / f"{self.SECTION}.json"
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
        return _prior_member(self._sections, ref, owner=self.SECTION)


class ExploreAgentKB(_ConfigPatchAgentKB):
    """Read the configuration section and stage accepted source overlays."""

    SECTION = EXPLORE_SECTION


class FrameworkAgentKB(_ConfigPatchAgentKB):
    """Read legacy FRAMEWORK config and stage accepted source overlays."""

    SECTION = FRAMEWORK_SECTION


class RecipeReplayKB:
    """Read global replay ordering from the downloaded current Recipe."""

    def __init__(self, sections: KnowledgeSections | None) -> None:
        self._sections = sections

    @classmethod
    def open(cls) -> "RecipeReplayKB":
        try:
            sections = KnowledgeSections.from_env()
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("recipe replay kb: unavailable: %s", exc)
            sections = None
        return cls(sections)

    @property
    def active(self) -> bool:
        return self._sections is not None and self._sections.warm_start_dir is not None

    def _read_value(self) -> dict[str, Any]:
        """Return the validated current Recipe value object."""
        if not self.active:
            return {}
        from .remote_recipe.models import (
            RemoteRecipeValidationError,
        )
        from .remote_recipe.values import (
            CURRENT_KNOWLEDGE_SCHEMA_VERSION,
            RECORD_KIND_HYPERLOOM_RECIPE,
        )

        recipe = self._sections.warm_start_dir / "recipe.json"
        try:
            document = json.loads(recipe.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError(f"recipe.json is not a JSON object, got {type(document).__name__}")
        except (OSError, ValueError) as exc:
            raise RemoteRecipeValidationError(f"current Recipe is unreadable: {exc}") from exc
        knowledge = document.get("knowledge") if isinstance(document.get("knowledge"), Mapping) else document
        if (
            knowledge.get("knowledge_schema_version") != CURRENT_KNOWLEDGE_SCHEMA_VERSION
            or knowledge.get("record_kind") != RECORD_KIND_HYPERLOOM_RECIPE
        ):
            raise RemoteRecipeValidationError("downloaded Recipe does not match the current knowledge contract")
        value = knowledge.get("value")
        if not isinstance(value, Mapping):
            raise RemoteRecipeValidationError("downloaded Recipe value must be an object")
        return dict(value)

    def read_config(self) -> dict[str, Any]:
        """Return the single final replay config, or ``{}`` for legacy records."""
        config = self._read_value().get("config")
        if not isinstance(config, Mapping):
            return {}
        return {
            "extra_server_args": str(config.get("extra_server_args") or ""),
            "extra_envs": (
                dict(config.get("extra_envs") or {}) if isinstance(config.get("extra_envs"), Mapping) else {}
            ),
        }

    def read_patch_timeline(self) -> list[str]:
        """Return the exact flat global timeline, rejecting malformed records."""
        from .remote_recipe.models import (
            RemoteRecipeValidationError,
            validate_relative_path,
        )

        value = self._read_value()
        timeline = value.get("patch_timeline") if isinstance(value, Mapping) else None
        if not isinstance(timeline, list) or not all(isinstance(ref, str) for ref in timeline):
            raise RemoteRecipeValidationError("current Recipe value.patch_timeline must be a flat string list")
        return [validate_relative_path(ref) for ref in timeline]


class KernelAgentKB:
    """Read prior ``gemm``/``fusion``/``rewrite`` sub-columns."""

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
        """True when section-backed prior reads are available."""
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
        return _prior_member(self._sections, ref, owner=KERNEL_SECTION)

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
        if not isinstance(node, Mapping):
            return {}

        blocked = {
            "source_file",
            "source_files",
            "target_file",
            "target_files",
            "target_path",
        }

        def without_source_metadata(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key): without_source_metadata(nested)
                    for key, nested in value.items()
                    if str(key) not in blocked
                }
            if isinstance(value, list):
                return [without_source_metadata(item) for item in value]
            return value

        return without_source_metadata(node)


__all__ = [
    "EXPLORE_SECTION",
    "ExploreAgentKB",
    "FRAMEWORK_SECTION",
    "FrameworkAgentKB",
    "KERNEL_SECTION",
    "KernelAgentKB",
    "RecipeReplayKB",
]
