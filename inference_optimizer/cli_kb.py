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
    """Resolve the local recipe-snapshot KB root.

    Resolution ladder (highest priority first): ``--local-kb-root`` →
    ``$HYPERLOOM_LOCAL_KB_ROOT`` → ``paths.workspace_root() / "kb"``.
    ``workspace_root()`` returns ``$USER_DATA_PATH`` when set (so a single
    override moves the whole KB tail) and otherwise falls back to
    ``/workspace/hyperloom`` with a one-shot loud warning. The directory
    is NOT created here (:class:`LocalRecipeStore` lazily creates it on
    first write).
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
    """Build the local-write / remote-read dispatcher for the recipe KB.

    Returns a :class:`recipe_kb.RecipeKB`. The local store is always
    wired; the remote half is enabled only when not ``--degraded-kb`` and
    a URL is resolved (``--cortex-kb-url`` or ``$CORTEX_KB_URL``), using
    the foreground-friendly 2s + 1-retry profile so a slow/unreachable
    kb-service never blocks the main loop. No hard-coded default endpoint.
    """
    from .recipe_kb import LocalRecipeStore, RecipeKB, RemoteRecipeClient

    local_root = _resolve_local_kb_root(args)
    local_store = LocalRecipeStore(root=local_root)

    if bool(getattr(args, "degraded_kb", False)):
        # Operator explicitly opted out — no network calls regardless
        # of CORTEX_KB_URL value.
        return RecipeKB(local=local_store, remote=None)

    # gbrain read-side remote (opt-in). Selected with
    # ``RECIPE_KB_REMOTE=gbrain`` + ``GBRAIN_BASE_URL`` / ``GBRAIN_TOKEN``.
    # Writes still go local-only; gbrain only serves the read side, so
    # this slots into the same dispatcher contract as the cortex remote.
    if os.environ.get("RECIPE_KB_REMOTE", "").strip().lower() == "gbrain":
        from .recipe_kb.gbrain_remote_client import build_gbrain_remote_from_env

        gbrain_remote = build_gbrain_remote_from_env()
        if gbrain_remote is not None and gbrain_remote.enabled:
            kb = RecipeKB(local=local_store, remote=gbrain_remote)
            # Mirror policy (``RECIPE_KB_MIRROR_MODE``, default ``external``):
            #   * ``external`` (default) — the optimizer does NOT mirror; an
            #     out-of-band service (hyperloom-recipe-mirror CronJob) ingests
            #     the local store into gbrain, keeping gbrain off the write
            #     path. REQUIRES that CronJob to be deployed: until it runs,
            #     new champions persist locally only (durable iff
            #     USER_DATA_PATH is the injected persistent path) and won't
            #     appear in gbrain.
            #   * ``inline`` — close the loop in-process: each local recipe
            #     write is best-effort mirrored into gbrain (the read cache)
            #     so a future session's remote read returns the champion
            #     config. Local write stays authoritative.
            mirror_mode = (
                os.environ.get("RECIPE_KB_MIRROR_MODE", "external").strip().lower()
            )
            if mirror_mode == "inline":
                # Opt-in best-effort in-process mirror. Falls back to the bare
                # dispatcher if the write-side MCP can't be built.
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
            # ``external`` (default): no in-process mirror.
            return kb
        # Selected but not configured: stay local-only rather than
        # silently falling back to the cortex kb-service.
        return RecipeKB(local=local_store, remote=None)

    cortex_url = (getattr(args, "cortex_kb_url", None) or "").strip()
    if not cortex_url:
        cortex_url = (os.environ.get("CORTEX_KB_URL", "") or "").strip()
    if not cortex_url:
        # No URL configured anywhere — local-only.
        return RecipeKB(local=local_store, remote=None)

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
    """Boot the recipe-snapshot KB integration and run the T0 anchor.

    Builds a RecipeKB dispatcher (local writes + optional remote-read
    fall-through) and runs ``run_t0_anchor`` against it; returns the
    dispatcher so the caller can thread it into the Coordinator. KB
    unavailability never aborts the launch (``fail_fast=False``); a hard
    T0 failure logs a warning and continues with an empty warm-start.
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
    # Mirror version + image fingerprint onto SharedState so the
    # CLOSE-time recipe write (coordinator._collect_workload_tags)
    # can stamp them onto the recipe.extras WITHOUT re-reading
    # manifest at every write. Resume reads ``stack_fingerprint_meta``
    # back from state.json verbatim.
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
        "marathon_dispatch_id": manifest.get("session_id", ""),
        "framework":            state.framework or manifest.get("framework", ""),
        "model_class":          state.model_class or "",
        # Operator traceability — the recipe.extras carry the most-
        # recent tracing tuple so a future debugger can answer
        # "which Claw job / sandbox produced this best_config".
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
            # KB unavailability MUST NOT abort the launch (operator
            # requirement). The dispatcher's remote half absorbs read
            # failures internally and the local store is always writable,
            # so a hard failure here is a programming bug, not an outage.
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
    """Construct the :class:`KnowledgePlane` facade for one session.

    Wires the (optional) PR Monitor REST client into a single read/write
    surface (KB reads go through the RecipeKB dispatcher, so
    ``cortex_kb=None`` here per the local-kb design). Both backends
    fail-soft. ``--degraded-pr`` yields a disabled PRMonitorClient. Trusts
    the IR-3 preflight probe result for ``pr_monitor_enabled``.
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

    # Record a one-shot status marker so ``breakdown.warnings`` can surface
    # ``pr_monitor:disabled`` / ``:unreachable`` without scraping logs.
    # Best-effort: a write failure only loses the breakdown row.
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

    # ``cortex_kb=None`` per the local-kb-recipe-snapshot design: the
    # central kb-service is consulted only as a recipe-read source via the
    # RecipeKB dispatcher. PR Monitor is still routed through here.
    return KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=pr_client,
        domain_repos=load_domain_repos(),
        pr_feed_window_days=window_days,
        pr_monitor_mcp_url=pr_mcp_url,
    )
