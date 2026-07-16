# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe-snapshot KB + KnowledgePlane bootstrap for the CLI.

Resolves the local KB root, builds the RecipeKB local-write/remote-read
dispatcher, runs the T0 warm-start anchor, and wires the KnowledgePlane
facade (PR Monitor + KB). Must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import append_jsonl
from hyperloom.orchestrator.knowledge.cortex_t0 import run_t0_anchor
from hyperloom.orchestrator.state.shared_state import SharedState
from ..session.paths import workspace_root as _workspace_root_resolve

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane


log = logging.getLogger(__name__)


def _resolve_local_kb_root(args: argparse.Namespace) -> Path:
    """Resolve the local recipe-snapshot KB root: ``--local-kb-root`` ->
    ``$HYPERLOOM_LOCAL_KB_ROOT`` -> ``workspace_root()/kb``. Not created here
    (LocalRecipeStore creates it lazily on first write).

    Args:
        args: Parsed CLI arguments; ``local_kb_root`` is consulted first.

    Returns:
        Path: The resolved local KB root directory.
    """
    explicit = getattr(args, "local_kb_root", None) or os.environ.get("HYPERLOOM_LOCAL_KB_ROOT", "")
    if explicit:
        return Path(str(explicit).strip())
    return _workspace_root_resolve() / "kb"


def _attach_recipe_audit_hook(kb: Any, session_dir: Path | None) -> None:
    """Wire ``RecipeKB.audit_hook`` to append remote-read trace events.

    Each recipe-snapshot remote read (``get_recipe`` / ``search``) is appended
    to ``recipe_snapshot/.audit.jsonl`` so the trace records whether the
    gbrain snapshot KB was consulted, the request, and how it resolved.
    Best-effort and never raises into the KB op. No-op without a session dir
    or when the dispatcher predates ``audit_hook``.

    Args:
        kb (Any): The RecipeKB dispatcher (or a mirroring wrapper around it).
        session_dir (Path | None): Session dir hosting the audit log.
    """
    if session_dir is None:
        return
    # Unwrap the inline gbrain-mirroring wrapper so the hook lands on the
    # RecipeKB whose reads emit the audit events.
    target = getattr(kb, "_inner", kb)
    if not hasattr(target, "audit_hook"):
        return

    from datetime import datetime, timezone

    from ..session.session_paths import recipe_snapshot_audit_jsonl

    audit_path = recipe_snapshot_audit_jsonl(Path(session_dir))

    def _hook(event: dict[str, Any]) -> None:
        """Append a timestamped recipe-snapshot read event to the audit log.

        Args:
            event (dict[str, Any]): The remote-read trace event to record.
        """
        try:
            row = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **event,
            }
            append_jsonl(audit_path, row, make_parents=True, sort_keys=True)
        except Exception:  # noqa: BLE001 — audit must never break a KB op
            log.debug("recipe_snapshot audit append failed", exc_info=True)

    target.audit_hook = _hook


def _build_recipe_kb_dispatcher(
    args: argparse.Namespace,
) -> Any:
    """Build the local-write / gbrain-read RecipeKB dispatcher. Local store is
    always wired; the read-side remote is the gbrain page store (``GBRAIN_*``),
    enabled unless ``--degraded-kb`` is set or gbrain is unconfigured.

    Writes stay local-only; gbrain serves the read side (and an optional
    in-process mirror of each local write via ``RECIPE_KB_MIRROR_MODE=inline``).

    Args:
        args: Parsed CLI arguments (``degraded_kb`` etc.).

    Returns:
        Any: A configured ``RecipeKB`` dispatcher (optionally gbrain-mirroring).
    """
    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB

    local_root = _resolve_local_kb_root(args)
    local_store = LocalRecipeStore(root=local_root)

    if bool(getattr(args, "degraded_kb", False)):
        return RecipeKB(local=local_store, remote=None)  # opt-out: no network

    # Read-side remote is gbrain only. Writes stay local-only; gbrain is
    # consulted for READS and (optionally) mirrored to on local write.
    from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import build_gbrain_remote_from_env

    gbrain_remote = build_gbrain_remote_from_env()
    if gbrain_remote is None or not gbrain_remote.enabled:
        return RecipeKB(local=local_store, remote=None)  # gbrain unconfigured: local-only

    kb = RecipeKB(local=local_store, remote=gbrain_remote)
    # RECIPE_KB_MIRROR_MODE (default ``external``): ``external`` keeps gbrain off
    # the write path; ``inline`` best-effort mirrors each local write into gbrain
    # in-process (local write stays authoritative).
    mirror_mode = os.environ.get("RECIPE_KB_MIRROR_MODE", "external").strip().lower()
    if mirror_mode == "inline":
        from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_ingest import (
            GbrainMirroringRecipeKB,
            build_mirror_mcp_from_env,
        )

        mirror_mcp = build_mirror_mcp_from_env()
        return GbrainMirroringRecipeKB(kb, mirror_mcp) if mirror_mcp is not None else kb
    return kb  # external (default): no in-process mirror


def _bootstrap_cortex_kb(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
):
    """Boot the recipe-snapshot KB integration, run the T0 anchor, and return
    the dispatcher. KB unavailability never aborts the launch; a hard T0
    failure warns and continues warm-start-empty.

    Args:
        args: Parsed CLI arguments.
        session_dir: The current session directory.
        manifest: The session manifest dict (model, framework, fingerprint).
        resume: Whether this launch is resuming an existing session.

    Returns:
        Any: The configured ``RecipeKB`` dispatcher.
    """
    kb = _build_recipe_kb_dispatcher(args)
    _attach_recipe_audit_hook(kb, session_dir)

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
    # Mirror version + image fingerprint onto SharedState for the CLOSE-time
    # recipe write.
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
        run_t0_anchor(
            kb,
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
    except Exception as exc:  # noqa: BLE001 — defensive
        print(
            f"WARNING: T0 recipe-snapshot anchor failed mid-flight: {exc}\n"
            f"Continuing without warm-start (recipes for this 5-tuple "
            f"will be created on first KEEP/REVERT).",
            file=sys.stderr,
        )
        args.kb_degraded_reason = getattr(args, "kb_degraded_reason", None) or "t0_runtime_fail"
    return kb


def _bootstrap_knowledge_plane(
    args: argparse.Namespace,
    *,
    cortex_client: Any = None,
    session_dir: Path | None = None,
) -> "KnowledgePlane":
    """Construct the :class:`KnowledgePlane` facade. Wires the optional PR
    Monitor REST client (KB reads go through RecipeKB, no Cortex KB client).
    Both backends fail-soft; --degraded-pr yields a disabled PRMonitorClient.

    Args:
        args: Parsed CLI arguments (PR Monitor enablement, URLs, window).
        cortex_client: Optional cortex client; unused (KB reads go via RecipeKB).
        session_dir: Optional session directory; when set a status marker is
            written for breakdown warnings.

    Returns:
        KnowledgePlane: The wired KnowledgePlane facade.
    """
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
    from hyperloom.orchestrator.knowledge.pr_monitor import (
        DEFAULT_PR_MONITOR_MCP_URL,
        PRMonitorClient,
    )

    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    pr_url = (getattr(args, "pr_monitor_url", None) or "").strip() or None
    pr_mcp_url = (getattr(args, "pr_monitor_mcp_url", None) or "").strip() or DEFAULT_PR_MONITOR_MCP_URL

    pr_client = PRMonitorClient.from_args(url=pr_url, enabled=pr_enabled)
    if not pr_enabled:
        reason = getattr(args, "pr_degraded_reason", None) or "explicit_flag"
        status_text = f"disabled ({reason})"
        print(f"PR Monitor       : DISABLED ({reason})")
        pr_reachable = False
    else:
        status_text = f"MCP {pr_mcp_url}"
        print(f"PR Monitor       : {pr_mcp_url}")
        pr_reachable = True

    # One-shot status marker so breakdown.warnings can surface pr_monitor:*
    # without scraping logs.
    if session_dir is not None:
        try:
            from ..session.session_paths import pr_monitor_status_json
            from ..session.paths import asset_actions_dir  # noqa: F401 (unused import warning suppress)

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
        except OSError as exc:  # noqa: BLE001 — defensive
            log.warning(
                "pr_monitor_status marker write failed: %r (breakdown.warnings will miss pr_monitor row)",
                exc,
            )

    # Read-only KB-graph MCP advertised to specialists as ``cortex_kb``. Default
    # is the gbrain MCP; HYPERLOOM_SPECIALIST_KB_MCP_URL / _TOKEN override it.
    # Empty => disabled (mcp__cortex_kb__* tools stripped from the whitelist).
    kb_mcp_url, kb_mcp_headers = _resolve_specialist_kb_mcp(args)
    if kb_mcp_url:
        print(f"Specialist KB MCP: {kb_mcp_url} (cortex_kb, read-only)")
    else:
        print("Specialist KB MCP: DISABLED (no GBRAIN_* / HYPERLOOM_SPECIALIST_KB_MCP_URL)")

    return KnowledgePlane.from_clients(
        pr_monitor=pr_client,
        pr_monitor_mcp_url=pr_mcp_url,
        cortex_kb_mcp_url=kb_mcp_url,
        cortex_kb_mcp_headers=kb_mcp_headers,
    )


def _resolve_specialist_kb_mcp(args: Any) -> tuple[str, dict[str, str]]:
    """Resolve the specialist read-only KB-graph (``cortex_kb``) MCP endpoint.

    Precedence: explicit ``--specialist-kb-mcp-url`` /
    ``$HYPERLOOM_SPECIALIST_KB_MCP_URL`` (token from
    ``$HYPERLOOM_SPECIALIST_KB_MCP_TOKEN``), else the gbrain MCP derived from
    ``$GBRAIN_BASE_URL`` (+ ``/mcp``) with a bearer ``$GBRAIN_TOKEN``.

    Args:
        args: Parsed CLI namespace.

    Returns:
        A ``(url, headers)`` pair; ``("", {})`` when nothing is configured.
    """
    override = (
        (getattr(args, "specialist_kb_mcp_url", None) or "").strip()
        or (os.environ.get("HYPERLOOM_SPECIALIST_KB_MCP_URL", "") or "").strip()
    )
    if override:
        token = (os.environ.get("HYPERLOOM_SPECIALIST_KB_MCP_TOKEN", "") or "").strip()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return override, headers

    gbrain_base = (os.environ.get("GBRAIN_BASE_URL", "") or "").strip().rstrip("/")
    gbrain_token = (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    if gbrain_base and gbrain_token:
        return f"{gbrain_base}/mcp", {"Authorization": f"Bearer {gbrain_token}"}
    return "", {}
