# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy enablement round deliverables into ``reports/enablement/<task_id>/``.

The archive collector drops ``runs/`` wholesale and retains ``reports/``, so a
patch, launch config or server log left in the specialist workspace never
reaches the archive and the fix cannot be replayed by a later session.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import atomic_write_json, atomic_write_text
from hyperloom.orchestrator.actions.executors.integrate_patch import _sanitize_setup_command
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

# Server logs routinely exceed _FILE_SIZE_LIMIT, so they are truncated rather
# than skipped. Half the patch ceiling still holds tens of thousands of lines of
# a crash while bounding what a many-round session adds to the archive.
_SERVER_LOG_TAIL_LIMIT = 1024 * 1024

_LOG_TRUNCATION_NOTE = "[hyperloom] truncated: the first {dropped} bytes are missing; the tail follows.\n"

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


def _copy_log_tail(src: Path, dest: Path, limit: int = _SERVER_LOG_TAIL_LIMIT) -> bool:
    """Copy at most the last ``limit`` bytes of ``src`` to ``dest``.

    The tail and not the head: a launch failure writes its traceback at the end
    of the log. What lands never exceeds ``limit`` plus the truncation note.

    Returns:
        ``True`` when the file landed at ``dest``.
    """
    if not src.is_file():
        return False
    dropped = max(0, src.stat().st_size - limit)
    with src.open("rb") as fh:
        if dropped:
            fh.seek(dropped)
        # Bounded here and not by EOF: the stat above can under-report a log
        # that is still being appended to.
        raw = fh.read(limit)
    # The seek lands mid-codepoint, so the decode has to be lenient. It ignores
    # rather than replaces: U+FFFD is three bytes, so a tail of binary noise
    # would otherwise write three times ``limit``.
    text = raw.decode("utf-8", errors="ignore")
    if dropped:
        text = _LOG_TRUNCATION_NOTE.format(dropped=dropped) + text
    atomic_write_text(dest, text, make_parents=True)
    return True


def role_path(files: list[dict[str, str]], role: str) -> str:
    """The session-relative path recorded for ``role``, or ``""`` when absent.

    For the single-valued roles: ``patch`` repeats and needs the list itself.
    """
    return next((entry["path"] for entry in files if entry["role"] == role), "")


def snapshot_round(session_dir: str | Path, res: dict[str, Any]) -> list[dict[str, str]]:
    """Archive one enablement round's patches, specialist result and launch config.

    Rounds the phase synthesises carry no task id and no deliverables, and are
    skipped rather than colliding on a shared directory.

    Args:
        session_dir: The session root directory.
        res: The ``integrate_patch`` result for an enablement round.

    Returns:
        One ``{"path", "role"}`` entry per deliverable that landed, ``path``
        session-relative POSIX. Roles: ``patch`` (any number),
        ``specialist_result``, ``prompt``, ``launch_config``, ``server_log``.
        A copy the size ceiling refused is absent rather than listed.
    """
    task_id = str(res.get("specialist_task_id") or "").strip()
    if not task_id:
        return []
    root = Path(session_dir)
    round_dir = enablement_round_dir(root, task_id)
    round_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, str]] = []

    def _record(role: str, dest: Path) -> None:
        written.append({"path": dest.relative_to(root).as_posix(), "role": role})

    patches_dir = round_dir / "patches"
    copied: set[str] = set()
    for applied in res.get("patches_applied") or []:
        src = Path(str(applied))
        dest = patches_dir / src.name
        if _copy(src, dest):
            _record("patch", dest)
        # Marked seen even when refused, so the sweep below does not retry it.
        copied.add(src.name)

    workspace = runs_dir(root, "specialist", task_id)
    for name, role in (("specialist_done.json", "specialist_result"), ("prompt.md", "prompt")):
        dest = round_dir / name
        if _copy(workspace / name, dest):
            _record(role, dest)

    # Patches the round did not apply still explain what was attempted.
    for base in (workspace, workspace / "worktree"):
        for pattern in ("*.patch", "*.diff"):
            for src in sorted((base / "patches").glob(pattern)):
                if src.name in copied:
                    continue
                dest = patches_dir / src.name
                if _copy(src, dest):
                    _record("patch", dest)
                copied.add(src.name)

    accepted_config = str(res.get("enablement_accepted_config_path") or "").strip()
    if accepted_config:
        dest = round_dir / "launch_config.yaml"
        if _copy(Path(accepted_config), dest):
            _record("launch_config", dest)

    # Only a round that reached a bench has one: a rejected patch or a broken
    # build never started a server, so an absent log is normal.
    bench = res.get("bench_result")
    server_log = str(bench.get("server_log_path") or "").strip() if isinstance(bench, dict) else ""
    if server_log:
        dest = round_dir / "server.log"
        if _copy_log_tail(Path(server_log), dest):
            _record("server_log", dest)

    launch_log = str(res.get("enablement_launch_log") or "")
    # Written last so the config path it names is the copy that just landed
    # under ``reports/``, not the ``runs/`` original the collector drops.
    atomic_write_json(
        round_dir / "round.json",
        {
            "status": res.get("status"),
            "specialist_task_id": task_id,
            "patches_applied": res.get("patches_applied") or [],
            "config_changes_applied": res.get("config_changes_applied") or {},
            "extra_envs_applied": res.get("extra_envs_applied") or {},
            "dropped_env_overrides": res.get("dropped_env_overrides") or [],
            "extra_server_args_applied": res.get("extra_server_args_applied") or "",
            # Redacted HERE and not where the list is built: the same field is
            # the replay channel -- ``lane.py`` stacks it into
            # ``state.enablement.setup_commands`` and the next round EXECUTES
            # what it finds there. The allowlist admits
            # ``pip install --index-url https://user:token@host/simple foo``,
            # so the command that has to stay runnable is also the one that must
            # not be written down verbatim. Sanitising at the source would send
            # a redacted string to pip; sanitising here separates the two.
            "setup_commands_applied": [_sanitize_setup_command(c) for c in (res.get("setup_commands_applied") or [])],
            "framework_switch_problems": res.get("framework_switch_problems") or [],
            "after_signature": res.get("after_signature") or {},
            "enablement_accepted_config_path": role_path(written, "launch_config"),
            "enablement_effective_config": res.get("enablement_effective_config") or {},
            "launch_log_excerpt": launch_log[:_LAUNCH_LOG_EXCERPT_CHARS],
        },
    )
    return written


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

    Idempotently rewritten on every ``kept`` or ``advanced`` verdict. Deliverables
    are emitted round by round from ``kept_rounds`` so the replay order matches the
    order integrate_patch applied them in; a round's patches precede its artifacts,
    which is what lets a round both patch and whole-file-replace the same file.

    Patches are copied to ``reports/enablement/patches/`` under a stack-ordered
    name, since specialists across rounds pick colliding file names, and are
    referenced only once the copy lands. Patches are dropped entirely without a
    framework root, because ``git apply`` would have no target to run against.

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
    patches_dest = enablement_dir(Path(session_dir)) / "patches"
    artifacts_dest = enablement_dir(Path(session_dir)) / "artifacts"

    patch_counter = 0
    artifact_counter = 0
    script_rounds: list[dict] = []

    for rnd in enablement.kept_rounds or []:
        rnd_script_patches: list[str] = []
        rnd_script_artifacts: list[dict[str, str]] = []

        if framework_root:
            for patch_str in rnd.get("patches") or []:
                patch_counter += 1
                src = Path(str(patch_str))
                name = f"{patch_counter:03d}_{src.name}"
                if _copy(src, patches_dest / name):
                    rnd_script_patches.append(f"patches/{name}")

        for art in rnd.get("artifacts") or []:
            artifact_counter += 1
            target = str(art.get("target") or "")
            name = f"{artifact_counter:03d}_{Path(target).name}"
            if not _copy(Path(str(art.get("source") or "")), artifacts_dest / name):
                continue
            _copy(Path(str(art.get("backup") or "")), artifacts_dest / f"{name}.orig")
            rnd_script_artifacts.append({"archive_path": f"artifacts/{name}", "target": target})

        if rnd_script_patches or rnd_script_artifacts:
            script_rounds.append({"patches": rnd_script_patches, "artifacts": rnd_script_artifacts})

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
        framework_root=framework_root if any(r.get("patches") for r in script_rounds) else None,
        runtime=runtime_path or None,
        rounds=script_rounds or None,
    )

    out = enablement_dir(Path(session_dir)) / "enablement_setting.sh"
    atomic_write_text(out, text, make_parents=True, mode=0o700)
    return str(out.relative_to(session_dir))


__all__ = ["role_path", "snapshot_round", "write_setting_script"]
