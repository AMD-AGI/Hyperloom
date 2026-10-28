# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe-snapshot KB + Phase 1 KnowledgePlane bootstrap for the CLI."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import JsonLineFormat, append_jsonl
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import T0AnchorRequest, run_t0_anchor
from hyperloom.orchestrator.state.shared_state import SharedState

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
    from hyperloom.orchestrator.knowledge.recipe_kb import RecipeKB


log = logging.getLogger(__name__)


def _publish_section_dirs(session_dir: Path, warm_start_dir: Path) -> None:
    """Point this run's agents at the shared draft and warm-start directories."""
    draft_dir = session_dir / "runtime" / "kb_draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KB_DRAFT_DIR"] = str(draft_dir)
    os.environ["KB_WARM_START_DIR"] = str(warm_start_dir)


def _attach_recipe_audit_hook(kb: "RecipeKB", session_dir: Path) -> None:
    """Wire ``RecipeKB.audit_hook`` to append local Recipe trace events."""
    from datetime import datetime, timezone

    from ..breakdown.recorder import warm_start_event
    from ..session.session_paths import recipe_snapshot_audit_jsonl

    audit_path = recipe_snapshot_audit_jsonl(Path(session_dir))

    def _hook(event: dict[str, Any]) -> None:
        """Append a timestamped recipe-snapshot read event to the audit log."""
        try:
            row = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **event,
            }
            append_jsonl(audit_path, row, make_parents=True, fmt=JsonLineFormat(sort_keys=True))
        except (OSError, TypeError, ValueError):
            log.debug("recipe_snapshot audit append failed", exc_info=True)
        # The same hook feeds the warm_start event, which claims only the reads
        # served while T0's own lookup is open. Kept separate from the append
        # above so a failed audit write does not cost the event its row.
        warm_start_event.record_read(session_dir, event)

    kb.audit_hook = _hook


def _build_recipe_kb_dispatcher() -> Any:
    """Build the local RecipeKB dispatcher, or validate remote mode."""
    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB

    config = KnowledgeConfig.from_env()

    if config.mode is KnowledgeStoreMode.REMOTE:
        return None

    # No remote client is constructed in local mode, even when ambient KB Store or GBrain credentials are present.
    kb = RecipeKB(local=LocalRecipeStore(root=Path(config.local_root)))
    kb.knowledge_config = config
    return kb


def _mirror_stack_fingerprint_meta(
    state: SharedState,
    stack_fp: Mapping[str, Any],
    image_digest: str,
) -> None:
    """Copy usable stack fingerprint fields onto SharedState for the CLOSE-time recipe write."""
    if not stack_fp:
        return
    merged_meta = dict(state.stack_fingerprint_meta)
    for key, value in stack_fp.items():
        if value not in (None, "", "unknown"):
            merged_meta[str(key)] = value
    if image_digest and image_digest != "unknown":
        merged_meta["image_digest"] = image_digest
    if merged_meta:
        state.stack_fingerprint_meta = merged_meta


def _bootstrap_recipe_kb(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
):
    """Boot the recipe-snapshot KB integration, run the T0 anchor, and return the dispatcher."""
    if bool(getattr(args, "degraded_kb", False)):
        print("Recipe KB       : DISABLED (--degraded-kb)")
        return None

    kb = _build_recipe_kb_dispatcher()
    state = SharedState.load_or_init(session_dir)
    workload = (
        state.model_name
        or manifest.get("model_name", "")
        or Path(manifest.get("model_path", "") or "").name
        or "unknown_model"
    )
    hw = state.gpu_type or manifest.get("gpu_type", "") or "unknown_gpu"
    stack_fp = manifest.get("stack_fingerprint") or {}
    image_digest = manifest.get("image") or ""
    _mirror_stack_fingerprint_meta(state, stack_fp, image_digest)
    extra_attrs = {
        "framework_name": state.framework or manifest.get("framework", ""),
        "model_class": state.model_class or "",
        # Operator traceability.
        "claw_session_id": manifest.get("claw_session_id") or "",
        "sandbox_user_id": manifest.get("sandbox_user_id") or "",
    }
    try:
        if kb is None:
            from hyperloom.orchestrator.knowledge.remote_recipe import (
                RecipeScope,
                RemoteRecipeClient,
                RemoteRecipeValidationError,
                RemoteWarmRecipeAdapter,
            )

            warm_start_dir = session_dir / "runtime" / "remote_recipe"
            try:
                recipe_scope = RecipeScope.from_state(state)
            except RemoteRecipeValidationError as exc:
                print(
                    f"WARNING: Remote Recipe KB scope is invalid: {exc}\nContinuing without warm-start.",
                    file=sys.stderr,
                )
                args.kb_degraded_reason = getattr(args, "kb_degraded_reason", None) or "recipe_scope_invalid"
                return kb
            t0_kb = RemoteWarmRecipeAdapter(RemoteRecipeClient.from_env(), warm_start_dir, recipe_scope)
            _publish_section_dirs(session_dir, warm_start_dir)
        else:
            _attach_recipe_audit_hook(kb, session_dir)
            t0_kb = kb
        run_t0_anchor(
            t0_kb,
            state,
            T0AnchorRequest(
                workload=workload,
                hw=hw,
                image_digest=image_digest,
                stack_fingerprint=stack_fp,
                extra_attrs=extra_attrs,
                resume=resume,
            ),
            on_status=print,
            session_dir=session_dir,
        )
        if kb is None:
            print("Recipe KB       : REMOTE (KB Store current Recipe warm replay)")
    except Exception as exc:  # isolation boundary: the T0 recipe warm-start anchor
        log.warning("T0 recipe-snapshot anchor failed mid-flight: %s", exc, exc_info=True)
        print(
            f"WARNING: T0 recipe-snapshot anchor failed mid-flight: {exc}\nContinuing without warm-start.",
            file=sys.stderr,
        )
        args.kb_degraded_reason = getattr(args, "kb_degraded_reason", None) or "t0_runtime_fail"
    return kb


def _bootstrap_knowledge_plane(
    args: argparse.Namespace,
    *,
    recipe_kb_client: Any = None,
) -> "KnowledgePlane":
    """Construct the :class:`KnowledgePlane` facade."""
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
    from hyperloom.common.pr_monitor_urls import pr_monitor_mcp_url

    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    pr_mcp_url = pr_monitor_mcp_url() if pr_enabled else ""

    if not pr_enabled:
        reason = getattr(args, "pr_degraded_reason", None) or "explicit_flag"
        print(f"PR Monitor       : DISABLED ({reason})")
    elif not pr_mcp_url:
        print("PR Monitor       : DISABLED (no MCP URL configured)")
    else:
        print(f"PR Monitor       : {pr_mcp_url}")

    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig

    kb_disabled = bool(getattr(args, "degraded_kb", False))
    if kb_disabled:
        # A complete KB opt-out must not validate or activate an ambient remote configuration.
        degraded_env = dict(os.environ)
        degraded_env["KNOWLEDGE_STORE_MODE"] = "local"
        config = KnowledgeConfig.from_env(degraded_env)
    else:
        config = getattr(recipe_kb_client, "knowledge_config", None) or KnowledgeConfig.from_env()
    return KnowledgePlane.from_clients(
        pr_monitor_enabled=bool(pr_mcp_url),
        pr_monitor_mcp_url=pr_mcp_url,
        recipe_kb=recipe_kb_client,
        config=config,
        kb_disabled=kb_disabled,
    )
