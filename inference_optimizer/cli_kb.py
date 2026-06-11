# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Recipe-snapshot KB + KnowledgePlane bootstrap for the CLI.

Resolves the local KB root, builds the RecipeKB local-write/remote-read
dispatcher, runs the T0 warm-start anchor, and wires the KnowledgePlane
facade (PR Monitor + KB). Extracted from ``cli.py``; imports orchestrator
packages only and must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .orchestrator.cortex_t0 import run_t0_anchor
from .orchestrator.shared_state import SharedState
from .paths import workspace_root as _workspace_root_resolve

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from .orchestrator.knowledge_plane import KnowledgePlane


log = logging.getLogger(__name__)


def _resolve_local_kb_root(args: argparse.Namespace) -> Path:
    """Resolve the local recipe-snapshot KB root: ``--local-kb-root`` ->
    ``$HYPERLOOM_LOCAL_KB_ROOT`` -> ``workspace_root()/kb``. Not created here
    (LocalRecipeStore creates it lazily on first write).
    """
    explicit = (
        getattr(args, "local_kb_root", None)
        or os.environ.get("HYPERLOOM_LOCAL_KB_ROOT", "")
    )
    if explicit:
        return Path(str(explicit).strip())
    return _workspace_root_resolve() / "kb"


def _build_recipe_kb_dispatcher(
    args: argparse.Namespace,
) -> Any:
    """Build the local-write / remote-read RecipeKB dispatcher. Local store
    always wired; remote half enabled only when not --degraded-kb and a URL
    resolves (foreground 2s + 1-retry; no hard-coded default endpoint).
    """
    from .recipe_kb import LocalRecipeStore, RecipeKB, RemoteRecipeClient

    local_root = _resolve_local_kb_root(args)
    local_store = LocalRecipeStore(root=local_root)

    if bool(getattr(args, "degraded_kb", False)):
        return RecipeKB(local=local_store, remote=None)  # opt-out: no network

    # Aggregated read remote (opt-in: RECIPE_KB_REMOTE=both). Fans reads across
    # gbrain (GBRAIN_*) and the cortex kb-service (--cortex-kb-url /
    # $CORTEX_KB_URL), then dedups/field-merges same-cid rows. Writes remain
    # local-only; mirroring policy is handled only by the gbrain-only path.
    if os.environ.get("RECIPE_KB_REMOTE", "").strip().lower() == "both":
        from .recipe_kb.composite_remote import CompositeRemoteRecipeClient
        from .recipe_kb.gbrain_remote_client import build_gbrain_remote_from_env

        sources: list[Any] = []
        names: list[str] = []
        gbrain_remote = build_gbrain_remote_from_env()
        if gbrain_remote is not None and gbrain_remote.enabled:
            sources.append(gbrain_remote)
            names.append("gbrain")
        cortex_url = (getattr(args, "cortex_kb_url", None) or "").strip()
        if not cortex_url:
            cortex_url = (os.environ.get("CORTEX_KB_URL", "") or "").strip()
        if cortex_url:
            sources.append(
                RemoteRecipeClient(kb_url=cortex_url, foreground=True, enabled=True)
            )
            names.append("cortex")
        if sources:
            return RecipeKB(
                local=local_store,
                remote=CompositeRemoteRecipeClient(sources, names=names),
            )
        return RecipeKB(local=local_store, remote=None)

    # gbrain read-side remote (opt-in: RECIPE_KB_REMOTE=gbrain + GBRAIN_*).
    # Writes stay local-only; gbrain serves the read side only.
    if os.environ.get("RECIPE_KB_REMOTE", "").strip().lower() == "gbrain":
        from .recipe_kb.gbrain_remote_client import build_gbrain_remote_from_env

        gbrain_remote = build_gbrain_remote_from_env()
        if gbrain_remote is not None and gbrain_remote.enabled:
            kb = RecipeKB(local=local_store, remote=gbrain_remote)
            # RECIPE_KB_MIRROR_MODE (default ``external``): external => an
            # out-of-band CronJob ingests the local store into gbrain (gbrain
            # off the write path); ``inline`` => best-effort mirror each local
            # write into gbrain in-process (local write stays authoritative).
            mirror_mode = (
                os.environ.get("RECIPE_KB_MIRROR_MODE", "external").strip().lower()
            )
            if mirror_mode == "inline":
                from .recipe_kb.gbrain_ingest import (
                    GbrainMirroringRecipeKB,
                    build_mirror_mcp_from_env,
                )
                mirror_mcp = build_mirror_mcp_from_env()
                return (
                    GbrainMirroringRecipeKB(kb, mirror_mcp)
                    if mirror_mcp is not None
                    else kb
                )
            return kb  # external (default): no in-process mirror
        # Selected but not configured: stay local-only.
        return RecipeKB(local=local_store, remote=None)

    cortex_url = (getattr(args, "cortex_kb_url", None) or "").strip()
    if not cortex_url:
        cortex_url = (os.environ.get("CORTEX_KB_URL", "") or "").strip()
    if not cortex_url:
        return RecipeKB(local=local_store, remote=None)  # no URL: local-only

    remote = RemoteRecipeClient(
        kb_url=cortex_url,
        foreground=True,
        enabled=True,
    )
    return RecipeKB(local=local_store, remote=remote)


def _bootstrap_cortex_kb(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
):
    """Boot the recipe-snapshot KB integration, run the T0 anchor, and return
    the dispatcher. KB unavailability never aborts the launch
    (fail_fast=False); a hard T0 failure warns and continues warm-start-empty.
    """
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
    # Mirror version + image fingerprint onto SharedState so the CLOSE-time
    # recipe write can stamp recipe.extras without re-reading manifest.
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
        "framework":            state.framework or manifest.get("framework", ""),
        "model_class":          state.model_class or "",
        # Operator traceability: which Claw job / sandbox produced best_config.
        "claw_session_id":      manifest.get("claw_session_id") or "",
        "sandbox_user_id":      manifest.get("sandbox_user_id") or "",
    }
    try:
        run_t0_anchor(
            kb,
            state,
            workload=workload,
            hw=hw,
            image_digest=image_digest,
            stack_fingerprint=stack_fp,
            extra_attrs=extra_attrs,
            resume=resume,
            # KB unavailability must not abort the launch (remote absorbs read
            # failures; local store is always writable).
            fail_fast=False,
            on_status=print,
            session_dir=session_dir,
            save_state=True,
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        print(
            f"WARNING: T0 recipe-snapshot anchor failed mid-flight: {exc}\n"
            f"Continuing without warm-start (recipes for this 5-tuple "
            f"will be created on first KEEP/REVERT).",
            file=sys.stderr,
        )
        args.kb_degraded_reason = (
            getattr(args, "kb_degraded_reason", None) or "t0_runtime_fail"
        )
    return kb


def _bootstrap_knowledge_plane(
    args: argparse.Namespace,
    *,
    cortex_client: Any = None,
    session_dir: Path | None = None,
) -> "KnowledgePlane":
    """Construct the :class:`KnowledgePlane` facade. Wires the optional PR
    Monitor REST client (KB reads go through RecipeKB, so cortex_kb=None here).
    Both backends fail-soft; --degraded-pr yields a disabled PRMonitorClient.
    """
    from .orchestrator.knowledge_plane import (
        KnowledgePlane,
        load_domain_repos,
    )
    from .orchestrator.pr_monitor import (
        DEFAULT_PR_FEED_WINDOW_DAYS,
        DEFAULT_PR_MONITOR_MCP_URL,
        PRMonitorClient,
    )

    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    pr_url = (getattr(args, "pr_monitor_url", None) or "").strip() or None
    pr_mcp_url = (
        (getattr(args, "pr_monitor_mcp_url", None) or "").strip()
        or DEFAULT_PR_MONITOR_MCP_URL
    )
    window_days = int(
        getattr(args, "pr_feed_window_days", DEFAULT_PR_FEED_WINDOW_DAYS)
        or DEFAULT_PR_FEED_WINDOW_DAYS
    )

    pr_client = PRMonitorClient.from_args(url=pr_url, enabled=pr_enabled)
    if not pr_enabled:
        reason = getattr(args, "pr_degraded_reason", None) or "explicit_flag"
        status_text = f"disabled ({reason})"
        print(f"PR Monitor       : DISABLED ({reason})")
        pr_reachable = False
    else:
        status_text = f"REST {pr_client.base_url} (window={window_days}d)"
        print(
            f"PR Monitor       : REST {pr_client.base_url} (window="
            f"{window_days}d, mcp={pr_mcp_url})"
        )
        pr_reachable = True

    # One-shot status marker so breakdown.warnings can surface pr_monitor:*
    # without scraping logs (best-effort).
    if session_dir is not None:
        try:
            from .session_paths import pr_monitor_status_json
            from .paths import asset_actions_dir  # noqa: F401 (unused import warning suppress)
            marker = pr_monitor_status_json(session_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "enabled":      bool(pr_enabled),
                "url":          (pr_client.base_url if pr_enabled else ""),
                "reachable":    bool(pr_reachable),
                "mcp_url":      pr_mcp_url if pr_enabled else "",
                "window_days":  int(window_days),
                "status_text":  status_text,
            }, sort_keys=True, indent=2))
        except OSError as exc:  # noqa: BLE001 — defensive
            log.warning(
                "pr_monitor_status marker write failed: %r "
                "(breakdown.warnings will miss pr_monitor row)", exc,
            )

    # cortex_kb=None per the local-kb design (KB reads go via RecipeKB).
    return KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=pr_client,
        domain_repos=load_domain_repos(),
        pr_feed_window_days=window_days,
        pr_monitor_mcp_url=pr_mcp_url,
    )
