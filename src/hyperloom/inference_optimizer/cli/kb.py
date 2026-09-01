# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe-snapshot KB + Phase 1 KnowledgePlane bootstrap for the CLI."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from hyperloom.common.io import append_jsonl, atomic_write_json
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import run_t0_anchor
from hyperloom.orchestrator.state.shared_state import SharedState

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane


log = logging.getLogger(__name__)

_LEGACY_WORKSPACE_KB_ROOT = Path("/workspace/hyperloom/kb")
_RECIPE_MIGRATION_MARKER = ".recipe-kb-migration-v1.json"
_RECIPE_MIGRATION_LOCK = ".recipe-kb-migration.lock"
_RECIPE_DATA_FILES = frozenset({"recipe.json", "attempts.ndjson"})


def _recipe_live_paths(root: Path) -> list[Path]:
    """Return valid seven-component Recipe live rows below *root*."""

    if not root.exists():
        return []
    if not root.is_dir():
        raise RuntimeError(f"legacy Recipe KB source is not a directory: {root}")
    rows: list[Path] = []
    try:
        for path in root.rglob("recipe.json"):
            if path.is_symlink() or not path.is_file():
                continue
            if len(path.parent.relative_to(root).parts) == 7:
                rows.append(path)
    except OSError as exc:
        raise RuntimeError(f"could not inspect legacy Recipe KB source {root}: {exc}") from exc
    return sorted(rows)


def _is_migratable_recipe_file(path: Path, recipe_dir: Path) -> bool:
    """Select durable Recipe data while excluding live lock/temporary files."""

    if path.is_symlink() or not path.is_file():
        return False
    relative = path.relative_to(recipe_dir)
    if any(part == ".lock" or part.endswith(".tmp") or part.startswith(".tmp") for part in relative.parts):
        return False
    return relative.name in _RECIPE_DATA_FILES or relative.parts[0] == "history" or not relative.name.startswith(".")


@contextmanager
def _recipe_migration_lock(destination: Path) -> Iterator[None]:
    """Serialize one-time migration among concurrent local-mode startups."""

    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / _RECIPE_MIGRATION_LOCK
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _copy_recipe_corpus(
    source: Path,
    destination: Path,
    live_paths: list[Path],
) -> tuple[int, list[Path], list[Path]]:
    """Copy complete Recipe directories without clobbering destination files."""

    files: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for live_path in live_paths:
        recipe_dir = live_path.parent
        for source_path in recipe_dir.rglob("*"):
            if not _is_migratable_recipe_file(source_path, recipe_dir):
                continue
            target = destination / source_path.relative_to(source)
            if target in seen:
                continue
            seen.add(target)
            if target.exists():
                raise RuntimeError(f"legacy Recipe migration would clobber existing file: {target}")
            files.append((source_path, target))

    created_files: list[Path] = []
    created_dirs: list[Path] = []
    completed = False
    try:
        for source_path, target in sorted(files, key=lambda pair: pair[1].as_posix()):
            missing_parents: list[Path] = []
            parent = target.parent
            while parent != destination and not parent.exists():
                missing_parents.append(parent)
                parent = parent.parent
            for directory in reversed(missing_parents):
                directory.mkdir()
                created_dirs.append(directory)
            with source_path.open("rb") as source_stream, target.open("xb") as target_stream:
                created_files.append(target)
                shutil.copyfileobj(source_stream, target_stream)
                target_stream.flush()
                os.fsync(target_stream.fileno())
            shutil.copystat(source_path, target, follow_symlinks=False)
        completed = True
    finally:
        if not completed:
            for path in reversed(created_files):
                with suppress(OSError):
                    path.unlink()
            for path in reversed(created_dirs):
                with suppress(OSError):
                    path.rmdir()
    return len(live_paths), created_files, created_dirs


def _legacy_recipe_root(env: Mapping[str, str]) -> Path:
    """Resolve the legacy implicit source without changing the new default."""

    user_data_path = str(env.get("USER_DATA_PATH") or "").strip()
    return Path(user_data_path).expanduser() / "kb" if user_data_path else _LEGACY_WORKSPACE_KB_ROOT


def _migrate_legacy_recipe_kb_once(*, destination: Path, source: Path) -> bool:
    """Migrate legacy Recipe data once, failing startup on a real copy error."""

    destination = destination.expanduser()
    source = source.expanduser()
    marker = destination / _RECIPE_MIGRATION_MARKER
    if marker.exists() or _recipe_live_paths(destination):
        return False
    live_paths = _recipe_live_paths(source)
    if not live_paths:
        return False
    with _recipe_migration_lock(destination):
        if marker.exists() or _recipe_live_paths(destination):
            return False
        live_paths = _recipe_live_paths(source)
        if not live_paths:
            return False
        created_files: list[Path] = []
        created_dirs: list[Path] = []
        completed = False
        try:
            recipe_count, created_files, created_dirs = _copy_recipe_corpus(source, destination, live_paths)
            atomic_write_json(
                marker,
                {"version": 1, "source": str(source), "recipes": recipe_count},
                fsync=True,
                mode=0o600,
            )
            directory_fd = os.open(destination, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            completed = True
        except Exception as exc:
            raise RuntimeError(
                f"legacy Recipe KB data exists at {source}, but migration into {destination} failed: {exc}"
            ) from exc
        finally:
            if not completed:
                with suppress(OSError):
                    marker.unlink()
                for path in reversed(created_files):
                    with suppress(OSError):
                        path.unlink()
                for path in reversed(created_dirs):
                    with suppress(OSError):
                        path.rmdir()
    log.info("migrated %d legacy Recipe row(s) from %s into %s", recipe_count, source, destination)
    return True


def _resolve_local_kb_root(args: argparse.Namespace) -> Path:
    """Resolve the shared local knowledge root without creating it.

    Args:
        args: Parsed CLI arguments; ``local_kb_root`` is consulted first.

    Returns:
        Path: The resolved local KB root directory.
    """
    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig

    explicit = getattr(args, "local_kb_root", None) or os.environ.get("HYPERLOOM_LOCAL_KB_ROOT", "")
    if explicit and "KNOWLEDGE_LOCAL_ROOT" not in os.environ:
        warnings.warn(
            "--local-kb-root/HYPERLOOM_LOCAL_KB_ROOT is deprecated; use KNOWLEDGE_LOCAL_ROOT",
            DeprecationWarning,
            stacklevel=2,
        )
        return Path(str(explicit).strip())
    return Path(KnowledgeConfig.from_env().local_root)


def _publish_section_dirs(session_dir: Path, warm_start_dir: Path) -> None:
    """Point this run's agents at the shared draft and warm-start directories.

    Agents run out of process, so the handoff is two paths in the environment
    rather than an object. Both are exported before T0 because a child may
    start before the warm-start download lands; a reader treats a directory
    that is not there yet as a cold start.
    """
    draft_dir = session_dir / "runtime" / "kb_draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KB_DRAFT_DIR"] = str(draft_dir)
    os.environ["KB_WARM_START_DIR"] = str(warm_start_dir)


def _attach_recipe_audit_hook(kb: Any, session_dir: Path | None) -> None:
    """Wire ``RecipeKB.audit_hook`` to append local Recipe trace events.

    Each recipe-snapshot read/write is appended to
    ``recipe_snapshot/.audit.jsonl`` so the trace records the request and how
    the local store resolved it.
    Best-effort and never raises into the KB op. No-op without a session dir
    or when the dispatcher predates ``audit_hook``.

    Args:
        kb (Any): The RecipeKB dispatcher (or a mirroring wrapper around it).
        session_dir (Path | None): Session dir hosting the audit log.
    """
    if session_dir is None:
        return
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
    """Build the local RecipeKB dispatcher, or validate remote mode.

    Args:
        args: Parsed CLI arguments (``degraded_kb`` etc.).

    Returns:
        Any: A configured local ``RecipeKB``. Remote and degraded modes return
        ``None`` because remote Recipe writes use KB Store at CLOSE only.
    """
    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB

    if bool(getattr(args, "degraded_kb", False)):
        return None
    config = KnowledgeConfig.from_env()

    if config.mode is KnowledgeStoreMode.REMOTE:
        return None

    # No remote client is constructed in local mode, even when ambient
    # KB Store or GBrain credentials are present.
    explicit_compatibility_root = getattr(args, "local_kb_root", None) or os.environ.get("HYPERLOOM_LOCAL_KB_ROOT")
    explicit_knowledge_root = str(os.environ.get("KNOWLEDGE_LOCAL_ROOT") or "").strip()
    if not explicit_knowledge_root and explicit_compatibility_root:
        config = replace(config, local_root=str(_resolve_local_kb_root(args)))
    if not explicit_knowledge_root and not explicit_compatibility_root:
        _migrate_legacy_recipe_kb_once(
            destination=Path(config.local_root),
            source=_legacy_recipe_root(os.environ),
        )
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
    """Boot the recipe-snapshot KB integration, run the T0 anchor, and return
    the dispatcher. KB unavailability never aborts the launch; a hard T0
    failure warns and continues warm-start-empty.

    Returns ``None`` when ``--degraded-kb`` is set (T0/T2/T3/T4 become no-ops).

    Args:
        args: Parsed CLI arguments.
        session_dir: The current session directory.
        manifest: The session manifest dict (model, framework, fingerprint).
        resume: Whether this launch is resuming an existing session.

    Returns:
        Any | None: The configured ``RecipeKB`` dispatcher, or ``None`` when
        KB hooks are disabled.
    """
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
    except Exception as exc:  # noqa: BLE001 — defensive
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
    """Construct the :class:`KnowledgePlane` facade. Wires the PR Monitor MCP
    URL and the PRMonitorClient enablement stub (KB reads go through RecipeKB).
    Fail-soft; --degraded-pr yields a disabled PRMonitorClient.

    Args:
        args: Parsed CLI arguments carrying ``pr_monitor_enabled`` and
            ``pr_degraded_reason``. The endpoint is derived from
            ``KB_STORE_URL``.
        recipe_kb_client: Optional recipe KB client; unused (KB reads go via RecipeKB).
        session_dir: Optional session directory; when set a status marker is
            written for breakdown warnings.

    Returns:
        KnowledgePlane: The wired KnowledgePlane facade.
    """
    from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
    from hyperloom.orchestrator.knowledge.pr_monitor import (
        PRMonitorClient,
        pr_monitor_mcp_url,
    )

    pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
    pr_mcp_url = pr_monitor_mcp_url()

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

    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig

    kb_disabled = bool(getattr(args, "degraded_kb", False))
    if kb_disabled:
        # A complete KB opt-out must not validate or activate an ambient remote
        # configuration. Keep a local-shaped config only for status plumbing.
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
