# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy enablement round deliverables into ``reports/enablement/<task_id>/``.

The archive collector drops ``runs/`` wholesale and retains ``reports/``, so a
patch or launch config left in the specialist workspace never reaches the
archive and the fix cannot be replayed by a later session.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import atomic_write_json, atomic_write_text
from hyperloom.inference_optimizer.session.session_paths import (
    enablement_dir,
    enablement_round_dir,
    runs_dir,
)

if TYPE_CHECKING:
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

# A patch is a few KB; anything this large is a stray build output and would
# eat into the archive's per-session budget.
_FILE_SIZE_LIMIT = 2 * 1024 * 1024

_LAUNCH_LOG_EXCERPT_CHARS = 1200


def _copy(src: Path, dest: Path) -> bool:
    """Copy ``src`` to ``dest`` when it exists and is under the size limit.

    Returns:
        ``True`` when the file landed at ``dest``.
    """
    if not src.is_file() or src.stat().st_size > _FILE_SIZE_LIMIT:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return True


def snapshot_round(session_dir: str | Path, res: dict[str, Any]) -> None:
    """Archive one enablement round's patches, specialist result and launch config.

    Rounds the phase synthesises carry no task id and no deliverables, and are
    skipped rather than colliding on a shared directory.

    Args:
        session_dir: The session root directory.
        res: The ``integrate_patch`` result for an enablement round.
    """
    task_id = str(res.get("specialist_task_id") or "").strip()
    if not task_id:
        return
    round_dir = enablement_round_dir(Path(session_dir), task_id)
    round_dir.mkdir(parents=True, exist_ok=True)

    launch_log = str(res.get("enablement_launch_log") or "")
    atomic_write_json(
        round_dir / "round.json",
        {
            "status": res.get("status"),
            "specialist_task_id": task_id,
            "patches_applied": res.get("patches_applied") or [],
            "config_changes_applied": res.get("config_changes_applied") or {},
            "extra_envs_applied": res.get("extra_envs_applied") or {},
            "extra_server_args_applied": res.get("extra_server_args_applied") or "",
            "setup_commands_applied": res.get("setup_commands_applied") or [],
            "framework_switch_problems": res.get("framework_switch_problems") or [],
            "after_signature": res.get("after_signature") or {},
            "enablement_accepted_config_path": res.get("enablement_accepted_config_path") or "",
            "enablement_effective_config": res.get("enablement_effective_config") or {},
            "launch_log_excerpt": launch_log[:_LAUNCH_LOG_EXCERPT_CHARS],
        },
    )

    patches_dir = round_dir / "patches"
    copied: set[str] = set()
    for applied in res.get("patches_applied") or []:
        src = Path(str(applied))
        _copy(src, patches_dir / src.name)
        copied.add(src.name)

    workspace = runs_dir(Path(session_dir), "specialist", task_id)
    _copy(workspace / "specialist_done.json", round_dir / "specialist_done.json")
    _copy(workspace / "prompt.md", round_dir / "prompt.md")

    # Patches the round did not apply still explain what was attempted.
    for base in (workspace, workspace / "worktree"):
        for pattern in ("*.patch", "*.diff"):
            for src in sorted((base / "patches").glob(pattern)):
                if src.name not in copied:
                    _copy(src, patches_dir / src.name)
                    copied.add(src.name)

    accepted_config = str(res.get("enablement_accepted_config_path") or "").strip()
    if accepted_config:
        _copy(Path(accepted_config), round_dir / "launch_config.yaml")


def write_setting_script(
    session_dir: str | Path,
    enablement: "EnablementRound",
    framework: str,
    *,
    model: str | None = None,
    tp: int | None = None,
    max_model_len: int | None = None,
    gpu_type: str | None = None,
) -> str:
    """Write ``reports/enablement/enablement_setting.sh`` from accumulated enablement state.

    Idempotently rewritten on every ``kept`` or ``advanced`` verdict.  Patches are
    copied to ``reports/enablement/patches/`` under a stack-ordered name, since
    specialists across rounds pick colliding file names, and are referenced only
    once the copy lands.  Patches are dropped entirely without a framework root,
    because ``git apply`` would have no target to run against.

    Whole-file artifacts are copied to ``reports/enablement/artifacts/`` and
    become ``install -D`` lines. Each one's pre-image is copied alongside as
    ``.orig``, which is what an upstream PR has to be written against.

    Args:
        session_dir: The session root directory.
        enablement: The current ``EnablementRound`` state object.
        framework: Framework identifier for the server entrypoint.
        model: Model path emitted as ``export MODEL=``.
        tp: Tensor-parallel degree.
        max_model_len: Context length cap.
        gpu_type: GPU type string.

    Returns:
        Session-relative path of the written script.
    """
    from hyperloom.inference_optimizer.reference_script import render_reference_script

    framework_root = str(enablement.framework_root or "").strip()

    script_patches: list[str] = []
    if framework_root:
        patches_dest = enablement_dir(Path(session_dir)) / "patches"
        for idx, patch_str in enumerate(enablement.kept_patches or [], start=1):
            src = Path(str(patch_str))
            name = f"{idx:03d}_{src.name}"
            if _copy(src, patches_dest / name):
                script_patches.append(f"patches/{name}")

    artifacts_dest = enablement_dir(Path(session_dir)) / "artifacts"
    script_artifacts: list[dict[str, str]] = []
    for idx, art in enumerate(enablement.kept_artifacts or [], start=1):
        target = str(art.get("target") or "")
        name = f"{idx:03d}_{Path(target).name}"
        if not _copy(Path(str(art.get("source") or "")), artifacts_dest / name):
            continue
        # ``artifact_backups/`` is outside PACKAGE_GLOBS, so this is the only
        # copy of the pre-image that reaches the archive.
        _copy(Path(str(art.get("backup") or "")), artifacts_dest / f"{name}.orig")
        script_artifacts.append({"archive_path": f"artifacts/{name}", "target": target})

    accepted_cfg = dict(enablement.accepted_config or {})
    extra_envs = {str(k): str(v) for k, v in (accepted_cfg.get("extra_envs") or {}).items()}
    extra_server_args = str(accepted_cfg.get("extra_server_args") or "").strip()

    active = enablement.active_runtime or {}
    runtime_path = str(active.get("venv_root") or "").strip() if isinstance(active, dict) else ""

    text = render_reference_script(
        framework=framework,
        server_args=extra_server_args,
        envs=extra_envs,
        model=model,
        tp=tp,
        max_model_len=max_model_len,
        gpu_type=gpu_type,
        setup_commands=list(enablement.setup_commands or []) or None,
        patches=script_patches or None,
        framework_root=framework_root if script_patches else None,
        runtime=runtime_path or None,
        artifacts=script_artifacts or None,
    )

    out = enablement_dir(Path(session_dir)) / "enablement_setting.sh"
    atomic_write_text(out, text, make_parents=True, mode=0o700)
    return str(out.relative_to(session_dir))


__all__ = ["snapshot_round", "write_setting_script"]
