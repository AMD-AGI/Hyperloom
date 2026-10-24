# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One KEEP's patch files, and the section handoff that publishes them."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from hyperloom.orchestrator.lever import (
    LEVER_CONFIG,
    LEVER_ENABLEMENT,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    patch_lever_kind,
)

from ..knowledge.agent_kb import OverlayProvenance, PatchKB
from ..knowledge.remote_recipe.sanitize import HOST_ORIGIN_KEY
from ..state.task_registry import Task

log = logging.getLogger(__name__)


#: The one owner label a patch KEEP stages under. The three-column layout has a
#: single ``patch`` column, so explore- and framework-agent lifts both stage
#: under this marker. Attribution keeps its own explore/framework split on the
#: lever kind.
_PATCH_KEEP_OWNER = "PATCH"

#: Levers whose overlays feed the one patch column. ``kernel`` publishes through
#: its own column and is absent; a KEEP that names none of these falls back to
#: the authoring phase to decide.
_PATCH_COLUMN_LEVERS = frozenset({LEVER_CONFIG, LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR, LEVER_ENABLEMENT})


def _lever_for_keep(task_params: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Name the lever a settled KEEP moved, reading the delivery first.

    What came back outranks what was asked for: a mandate that goes out naming
    one lever routinely returns another, and the result carries the applier's
    own markers.
    """
    return patch_lever_kind(result) or patch_lever_kind(task_params)


def _is_patch_column_keep(task_params: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Whether a settled KEEP's overlay belongs in the patch column.

    A patch-column lever answers on its own; a KEEP that names no such lever
    (kernel, or none recorded) falls back to the authoring phase, exactly as the
    old owner label did before explore and framework shared one column. This is
    now one boolean rather than a per-agent owner, because there is one column.
    """
    lever = _lever_for_keep(task_params, result)
    if lever in _PATCH_COLUMN_LEVERS:
        return True
    phase = str(task_params.get("source_phase") or result.get("source_phase") or "").strip().upper()
    return phase in {"EXPLORE", "FRAMEWORK_AGENT"}


def _patch_member_list(raw: Any) -> list[Any]:
    if isinstance(raw, (str, Path)):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _declared_patch_members(result: Mapping[str, Any], params: Mapping[str, Any]) -> list[Any] | None:
    if "patches_applied" in result:
        return _patch_member_list(result.get("patches_applied"))
    declared: list[Any] = []
    named = "patches" in params
    for key in ("patches", "prior_patches", "patch_path", "patch"):
        if key in result:
            named = True
            declared.extend(_patch_member_list(result.get(key)))
    declared.extend(_patch_member_list(params.get("patches")))
    return declared if named else None


def _workspace_patch_files(workspace: str) -> list[Path]:
    root = Path(workspace) if workspace else None
    if root is None or not root.is_dir():
        return []
    found: list[Path] = []
    for base in (root, root / "patches", root / "worktree" / "patches"):
        if base.is_dir():
            found.extend(sorted(base.glob("*.patch")))
            found.extend(sorted(base.glob("*.diff")))
    return found


def _existing_patch_paths(candidates: list[Any]) -> tuple[list[Path], list[str]]:
    resolved: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for raw in candidates:
        if isinstance(raw, Mapping):
            raw = raw.get("patch_path") or raw.get("patch_ref") or raw.get("patch_file") or ""
        raw_text = str(raw or "").strip()
        if not raw_text:
            missing.append("<empty-patch-member>")
            continue
        path = Path(raw_text)
        if not path.is_file():
            missing.append(raw_text)
            continue
        try:
            canonical = path.resolve()
        except OSError:
            canonical = path
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    return resolved, missing


def _unstageable_patch_sources(row: Mapping[str, Any]) -> list[str]:
    missing = [str(path) for path in (row.get("missing_patch_sources") or []) if str(path).strip()]
    if row.get("include_patches"):
        missing.extend(
            str(path)
            for path in (row.get("patch_sources") or [])
            if str(path).strip() and not Path(str(path)).is_file()
        )
    return list(dict.fromkeys(missing))


def _keep_origin_provenance(result: Mapping[str, Any], sources: list[Path], *, realized: bool) -> dict[str, Any]:
    return {
        "base_sha": str(result.get("base_sha") or ""),
        "complete": bool(result.get("source_snapshot_complete", True)),
        "artifacts_outside_root": int(result.get("source_artifacts_outside_root") or 0),
        "realized": bool(realized),
        HOST_ORIGIN_KEY: {
            "apply_root": str(result.get("framework_root") or ""),
            "snapshot": str(result.get("source_snapshot") or ""),
            "manifest": str(result.get("source_manifest") or ""),
            "sources": [str(path) for path in sources],
        },
    }


def _source_snapshot_layers(stack: Any) -> list[dict[str, Any]]:
    from ..source_snapshot import source_layer_overlay_dir, source_layer_reproducible

    return [
        {
            "id": str(entry.get("variant_name") or entry.get("name") or ""),
            # GEAK PYTHONPATHs this value; it is the import root inside the
            # snapshot, not the snapshot's own top level.
            "snapshot_dir": source_layer_overlay_dir(entry) if snapshot else "",
            "framework_root": str(entry.get("framework_root") or ""),
            "base_sha": str(entry.get("base_sha") or ""),
            "reproducible": source_layer_reproducible(entry) if snapshot else False,
        }
        for entry in (stack or [])
        if isinstance(entry, dict) and entry.get("scope") == "source_patch"
        for snapshot in (str(entry.get("source_snapshot") or "").strip(),)
    ]


def _source_layer_handles(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source-layer fields a ``source_patch`` lift must carry.

    Every field here has to survive the hop from the executor result into the
    optimization_stack entry, where ``build_env_spec`` reads it. Assembling them
    in one place is what keeps a second lift path from forwarding a subset and
    degrading the overlay it feeds.

    Args:
        result: An integrate_patch KEEP result.

    Returns:
        dict[str, Any]: Handles to merge into the lift. ``source_snapshot_complete``
            is present only when the result recorded it, so entries predating the
            field still fall back to reading their manifest.
    """
    handles: dict[str, Any] = {
        "source_snapshot": result.get("source_snapshot") or "",
        "source_manifest": result.get("source_manifest") or "",
        "target_files": [str(path) for path in (result.get("target_files") or []) if str(path).strip()],
        "framework_root": result.get("framework_root") or "",
        "base_sha": result.get("base_sha") or "",
        "source_import_root": result.get("source_import_root") or "",
    }
    if "source_snapshot_complete" in result:
        handles["source_snapshot_complete"] = bool(result["source_snapshot_complete"])
    return handles


class AgentKeepStagingMixin:
    """Staging a settled KEEP into the patch column."""

    @staticmethod
    def _keep_patch_sources(
        result: Mapping[str, Any],
        task: "Task | None",
    ) -> tuple[list[Path], list[str]]:
        """Locate authoritative patch files without reconstructing their diff."""
        params = (getattr(task, "params", None) or {}) if task is not None else {}
        declared = _declared_patch_members(result, params)
        workspace = str(result.get("workspace") or "").strip()
        candidates = declared if declared is not None else _workspace_patch_files(workspace)
        return _existing_patch_paths(candidates)

    @staticmethod
    def _provenance_with_apply_roots(
        provenance: Mapping[str, Any],
        refs: list[str],
    ) -> dict[str, Any]:
        """Turn this KEEP's single apply root into a per-ref answer.

        One KEEP lands in one checkout, so every ref it staged shares that root;
        recording it per ref is what lets a Recipe whose KEEPs came from
        different trees stay replayable without anything reconciling them.
        """
        row = {key: value for key, value in provenance.items() if key != HOST_ORIGIN_KEY}
        origin = dict(provenance.get(HOST_ORIGIN_KEY) or {})
        apply_root = str(origin.pop("apply_root", "") or "").strip()
        if apply_root and refs:
            origin["apply_roots"] = {str(ref): apply_root for ref in refs if str(ref).strip()}
        if origin:
            row[HOST_ORIGIN_KEY] = origin
        return row

    def _stage_agent_keep(
        self,
        *,
        owner: str,
        stack_index: int,
        result: Mapping[str, Any],
        task: "Task | None",
        include_patches: bool,
        provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Stage one KEEP into the patch column.

        Returns ``False`` when staging must be retried. The overlay bytes are
        captured now rather than at CLOSE because the worktree they were
        harvested from does not outlive the round. Config and kernel are
        published once from the settled stack at CLOSE.
        """
        if not str(owner or "").strip():
            return False
        patch_kb = PatchKB.open()
        if not patch_kb.active:
            return False
        sources, missing = self._keep_patch_sources(result, task)
        if include_patches and missing:
            log.warning(
                "patch kb: KEEP at stack index %d references missing patch members: %s",
                stack_index,
                missing,
            )
            return False
        if not include_patches:
            return True
        refs: list[str] = []
        if sources:
            refs = patch_kb.stage_patches(sources, stack_index=stack_index)
            if len(refs) != len(sources) or any(not ref for ref in refs):
                return False
        # How the overlay was captured travels with it, so a later session can
        # tell a complete capture from one that could not account for every path,
        # and can apply each overlay to the checkout it was taken from.
        if provenance and not patch_kb.stage_provenance(
            stack_index=stack_index,
            provenance=OverlayProvenance(**self._provenance_with_apply_roots(provenance, refs)),
        ):
            return False
        return True

    def _enqueue_agent_keep_outbox(
        self,
        *,
        stack_index: int,
        result: Mapping[str, Any],
        task: "Task | None",
        include_patches: bool,
    ) -> None:
        """Persist an idempotent section handoff to run after state durability.

        The caller has already decided this KEEP feeds the patch column, so
        every stored field (row id, outbox owner, the stack's
        ``kb_required_owner``) uses the single patch-owner marker: the
        three-column layout has one patch column, not a per-agent one.
        """
        from ..knowledge.remote_recipe import KnowledgeSections

        if (
            str(os.environ.get("KNOWLEDGE_STORE_MODE") or "local").strip() != "remote"
            or KnowledgeSections.from_env() is None
        ):
            return
        sources, missing = self._keep_patch_sources(result, task) if include_patches else ([], [])
        # The realized diff is what the tree ended up holding, so it replaces the
        # delivered patch rather than joining it -- staging both would apply the
        # same change twice. It is chosen here, not at drain, because drain only
        # ever sees the row.
        realized = Path(str(result.get("source_realized_patch") or "").strip() or ".")
        realized_supersedes = include_patches and realized.is_file()
        if realized_supersedes:
            sources, missing = [realized], []
        row = {
            "id": f"{_PATCH_KEEP_OWNER}:{int(stack_index)}",
            "owner": _PATCH_KEEP_OWNER,
            "stack_index": int(stack_index),
            "include_patches": bool(include_patches),
            "patch_sources": [str(path) for path in sources],
            "missing_patch_sources": missing,
            "provenance": _keep_origin_provenance(result, sources, realized=realized_supersedes),
        }
        outbox = list(self.shared_state.kb_stage_outbox or [])
        if not any(isinstance(existing, dict) and existing.get("id") == row["id"] for existing in outbox):
            outbox.append(row)
        if include_patches and (sources or missing):
            self._claim_patch_section(int(stack_index))
        self.shared_state.kb_stage_outbox = outbox

    def _claim_patch_section(self, stack_index: int) -> None:
        stack = list(self.shared_state.optimization_stack or [])
        if 0 <= stack_index < len(stack) and isinstance(stack[stack_index], dict):
            stack[stack_index]["kb_required_owner"] = _PATCH_KEEP_OWNER
            self.shared_state.optimization_stack = stack

    def _dead_letter_keep_section(
        self,
        dead_letter: list[dict[str, Any]],
        row: Mapping[str, Any],
        missing: list[str],
    ) -> list[dict[str, Any]]:
        owner = str(row.get("owner") or "").upper()
        index = int(row.get("stack_index") or 0)
        stack = self.shared_state.optimization_stack or []
        claimed = 0 <= index < len(stack) and isinstance(stack[index], dict)
        if claimed and str(stack[index].get("kb_required_owner") or "").upper() == owner:
            stack[index].pop("kb_required_owner", None)
        log.warning(
            "%s kb: dropping owner section %s because patch sources are unavailable: %s",
            owner,
            row.get("id"),
            missing,
        )
        failed = {**dict(row), "missing_patch_sources": missing, "reason": "patch_source_missing"}
        return [item for item in dead_letter if item.get("id") != failed.get("id")] + [failed]

    def _drain_agent_keep_outbox(self) -> None:
        """Run section writes only after the authoritative state save."""
        pending = list(self.shared_state.kb_stage_outbox or [])
        if not pending:
            return
        retained: list[dict[str, Any]] = []
        dead_letter = [dict(row) for row in (self.shared_state.kb_stage_dead_letter or []) if isinstance(row, dict)]
        stack = list(self.shared_state.optimization_stack or [])
        for row in pending:
            if not isinstance(row, dict):
                continue
            missing = _unstageable_patch_sources(row)
            if missing:
                dead_letter = self._dead_letter_keep_section(dead_letter, row, missing)
            elif not self._stage_agent_keep(
                owner=str(row.get("owner") or ""),
                stack_index=int(row.get("stack_index") or 0),
                result={"patches": list(row.get("patch_sources") or [])},
                task=SimpleNamespace(params={}),
                include_patches=bool(row.get("include_patches")),
                provenance=row.get("provenance"),
            ):
                retained.append(row)
        self.shared_state.kb_stage_outbox = retained
        self.shared_state.kb_stage_dead_letter = dead_letter[-200:]
        self.shared_state.optimization_stack = stack
        self.shared_state.save(self.session_dir)
