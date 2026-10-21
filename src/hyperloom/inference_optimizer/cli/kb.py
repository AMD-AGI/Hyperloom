# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe-snapshot KB + Phase 1 KnowledgePlane bootstrap for the CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import JsonLineFormat, append_jsonl
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import run_t0_anchor
from hyperloom.orchestrator.state.shared_state import SharedState

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane


log = logging.getLogger(__name__)


def _publish_section_dirs(session_dir: Path, warm_start_dir: Path) -> None:
    """Point this run's agents at the shared draft and warm-start directories."""
    draft_dir = session_dir / "runtime" / "kb_draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KB_DRAFT_DIR"] = str(draft_dir)
    os.environ["KB_WARM_START_DIR"] = str(warm_start_dir)


def _attach_recipe_audit_hook(kb: Any, session_dir: Path | None) -> None:
    """Wire ``RecipeKB.audit_hook`` to append local Recipe trace events."""
    if session_dir is None:
        return
    target = getattr(kb, "_inner", kb)
    if not hasattr(target, "audit_hook"):
        return

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

    target.audit_hook = _hook


def _build_recipe_kb_dispatcher(
    args: argparse.Namespace,
) -> Any:
    """Build the local RecipeKB dispatcher, or validate remote mode."""
    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB

    if bool(getattr(args, "degraded_kb", False)):
        return None
    config = KnowledgeConfig.from_env()

    if config.mode is KnowledgeStoreMode.REMOTE:
        return None

    # No remote client is constructed in local mode, even when ambient KB Store or GBrain credentials are present.
    store: Any = LocalRecipeStore(root=Path(config.local_root))
    kb = RecipeKB(
        local=store,
        mode=config.mode.value,
        backend_name=config.backend,
    )
    kb.knowledge_config = config
    return kb


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

    kb = _build_recipe_kb_dispatcher(args)

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
    # Mirror version + image fingerprint onto SharedState for the CLOSE-time recipe write.
    if isinstance(stack_fp, dict) and stack_fp:
        merged_meta = dict(getattr(state, "stack_fingerprint_meta", {}) or {})
        for key, value in stack_fp.items():
            if value not in (None, "", "unknown"):
                merged_meta[str(key)] = value
        if image_digest and image_digest != "unknown":
            merged_meta["image_digest"] = image_digest
        if merged_meta:
            state.stack_fingerprint_meta = merged_meta
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
                HyperloomRemoteKB,
                RecipeScope,
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
            t0_kb = RemoteWarmRecipeAdapter(
                HyperloomRemoteKB.from_env(),
                warm_start_dir,
                recipe_scope,
            )
            _publish_section_dirs(session_dir, warm_start_dir)
        else:
            _attach_recipe_audit_hook(kb, session_dir)
            t0_kb = kb
        run_t0_anchor(
            t0_kb,
            state,
            workload=workload,
            hw=hw,
            image_digest=image_digest,
            stack_fingerprint=stack_fp,
            extra_attrs=extra_attrs,
            resume=resume,
            on_status=print,
            session_dir=session_dir,
            save_state=True,
        )
        if kb is None:
            print("Recipe KB       : REMOTE (KB Store current Recipe warm replay)")
    except Exception as exc:
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
    session_dir: Path | None = None,
) -> "KnowledgePlane":
    """Construct the :class:`KnowledgePlane` facade."""
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
    from hyperloom.common.pr_monitor_urls import pr_monitor_mcp_url
    from hyperloom.orchestrator.knowledge.pr_monitor import PRMonitorClient

    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    pr_mcp_url = pr_monitor_mcp_url() if pr_enabled else ""

    pr_client = PRMonitorClient.from_args(enabled=pr_enabled)
    if not pr_enabled:
        reason = getattr(args, "pr_degraded_reason", None) or "explicit_flag"
        status_text = f"disabled ({reason})"
        print(f"PR Monitor       : DISABLED ({reason})")
        pr_reachable = False
    elif not pr_mcp_url:
        status_text = "disabled (no_mcp_url)"
        print("PR Monitor       : DISABLED (no MCP URL configured)")
        pr_reachable = False
        pr_client.enabled = False
    else:
        status_text = f"MCP {pr_mcp_url}"
        print(f"PR Monitor       : {pr_mcp_url}")
        pr_reachable = True

    # One-shot status marker so breakdown.warnings can surface pr_monitor:* without scraping logs.
    if session_dir is not None:
        try:
            from ..session.session_paths import pr_monitor_status_json

            marker = pr_monitor_status_json(session_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "enabled": bool(pr_enabled),
                        "reachable": bool(pr_reachable),
                        "mcp_url": pr_mcp_url if pr_enabled else "",
                        "status_text": status_text,
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
        except OSError as exc:
            log.warning(
                "pr_monitor_status marker write failed: %r (breakdown.warnings will miss pr_monitor row)",
                exc,
            )

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
        pr_monitor=pr_client,
        pr_monitor_mcp_url=pr_mcp_url,
        recipe_kb=recipe_kb_client,
        config=config,
        kb_disabled=kb_disabled,
    )
