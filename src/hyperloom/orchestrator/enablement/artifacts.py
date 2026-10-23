# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy enablement round deliverables into ``reports/enablement/<task_id>/``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import WriteOptions, atomic_write_text
from hyperloom.common.nofollow import read_bytes_nofollow, regular_files_nofollow
from hyperloom.inference_optimizer.grid_server_args import strip_benchmark_harness_flags
from hyperloom.inference_optimizer.reference_script import ServingTarget
from hyperloom.inference_optimizer.session.session_paths import (
    enablement_dir,
    enablement_round_dir,
    runs_dir,
)
from hyperloom.inference_optimizer.breakdown.round_archive import (
    ROLE_ARTIFACT_PREIMAGE,
    ROLE_ARTIFACT_SOURCE,
    ROLE_LAUNCH_CONFIG,
    ROLE_PATCH,
    ROLE_PATCH_EVIDENCE,
    ROLE_PROMPT,
    ROLE_SERVER_LOG,
    ROLE_SPECIALIST_RESULT,
    RoundArchive,
)

if TYPE_CHECKING:
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

# A patch is a few KB; anything this large is a stray build output and would eat into the archive's per-session
# budget.
_FILE_SIZE_LIMIT = 2 * 1024 * 1024
_PATCHES_DIR = "patches"

# Server logs routinely exceed _FILE_SIZE_LIMIT, so they are truncated rather than skipped.
_SERVER_LOG_TAIL_LIMIT = 1024 * 1024

_LOG_TRUNCATION_NOTE = "[hyperloom] truncated: the first {dropped} bytes are missing; the tail follows.\n"


def _archive_copy(archive: RoundArchive, role: str, src: Path, dest: Path, *, child_written: bool = False) -> None:
    """Copy ``src`` to ``dest`` and record it as ``role`` when it exists and is under the size limit.

    ``child_written``: the file lives where a demoted specialist can replace it, so a symlink is refused, not followed.
    """
    if not src.is_file() or src.stat().st_size > _FILE_SIZE_LIMIT:
        return
    try:
        data = read_bytes_nofollow(src) if child_written else src.read_bytes()
    except OSError:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    archive.record(role, dest)


def _artifact_archive_name(idx: int, target: str) -> str:
    """Archive filename for the ``idx``-th artifact a round installed.

    Derived from the round-local index so a reader addresses a copy by name
    rather than by position in the archive directory.
    """
    return f"{idx:03d}_{Path(target).name}"


def _copy_log_tail(src: Path, dest: Path, limit: int = _SERVER_LOG_TAIL_LIMIT) -> bool:
    """Copy at most the last ``limit`` bytes of ``src`` to ``dest``."""
    if not src.is_file():
        return False
    dropped = max(0, src.stat().st_size - limit)
    with src.open("rb") as fh:
        if dropped:
            fh.seek(dropped)
        # Bounded here and not by EOF: the stat above can under-report a log that is still being appended to.
        raw = fh.read(limit)
    # The seek lands mid-codepoint, so the decode has to be lenient.
    text = raw.decode("utf-8", errors="ignore")
    if dropped:
        text = _LOG_TRUNCATION_NOTE.format(dropped=dropped) + text
    atomic_write_text(dest, text, options=WriteOptions(make_parents=True))
    return True


def snapshot_round(session_dir: str | Path, res: dict[str, Any]) -> RoundArchive:
    """Archive one enablement round's deliverables and report what landed.

    Rounds the phase synthesises carry no task id and no deliverables, and are
    skipped rather than colliding on a shared directory.

    Args:
        session_dir: The session root directory.
        res: The ``integrate_patch`` result for an enablement round.

    Returns:
        RoundArchive: One record per deliverable that landed. A copy the size
        ceiling refused leaves no record, so no consumer is handed a path that
        resolves to nothing. Patches the round never applied are recorded under
        :data:`ROLE_PATCH_EVIDENCE`, never :data:`ROLE_PATCH`.
    """
    task_id = str(res.get("specialist_task_id") or "").strip()
    root = Path(session_dir)
    archive = RoundArchive(root)
    if not task_id:
        return archive
    round_dir = enablement_round_dir(root, task_id)
    round_dir.mkdir(parents=True, exist_ok=True)

    copied: set[str] = set()
    for applied in res.get("patches_applied") or []:
        src = Path(str(applied))
        _archive_copy(archive, ROLE_PATCH, src, round_dir / _PATCHES_DIR / src.name)
        # Marked seen even when refused, so the sweep below does not retry it.
        copied.add(src.name)

    workspace = runs_dir(root, "specialist", task_id)
    for name, role in (("specialist_done.json", ROLE_SPECIALIST_RESULT), ("prompt.md", ROLE_PROMPT)):
        _archive_copy(archive, role, workspace / name, round_dir / name, child_written=True)

    # Disk scans include rejected output: preserve evidence, not accepted patches.
    for base in (workspace, workspace / "worktree"):
        for src in (
            *regular_files_nofollow(base, _PATCHES_DIR, (".patch",)),
            *regular_files_nofollow(base, _PATCHES_DIR, (".diff",)),
        ):
            if src.name not in copied:
                _archive_copy(
                    archive, ROLE_PATCH_EVIDENCE, src, round_dir / "attempted_patches" / src.name, child_written=True
                )
                copied.add(src.name)

    artifacts_dir = round_dir / "artifacts"
    for idx, art in enumerate(res.get("artifacts_applied") or []):
        name = _artifact_archive_name(idx, str(art.get("target") or ""))
        source = str(art.get("source") or "").strip()
        backup = str(art.get("backup") or "").strip()
        if source:
            _archive_copy(archive, ROLE_ARTIFACT_SOURCE, Path(source), artifacts_dir / name)
        if backup:
            _archive_copy(archive, ROLE_ARTIFACT_PREIMAGE, Path(backup), artifacts_dir / f"{name}.orig")

    accepted_config = str(res.get("enablement_accepted_config_path") or "").strip()
    if accepted_config:
        _archive_copy(archive, ROLE_LAUNCH_CONFIG, Path(accepted_config), round_dir / "launch_config.yaml")

    # Only a round that reached a bench has one: a rejected patch or a broken build never started a server, so an
    # absent log is normal.
    bench = res.get("bench_result")
    server_log = str(bench.get("server_log_path") or "").strip() if isinstance(bench, dict) else ""
    if server_log and _copy_log_tail(Path(server_log), round_dir / "server.log"):
        archive.record(ROLE_SERVER_LOG, round_dir / "server.log")

    return archive


def write_setting_script(
    session_dir: str | Path,
    enablement: "EnablementRound",
    framework: str,
    *,
    target: ServingTarget = ServingTarget(),
) -> str:
    """Write ``reports/enablement/enablement_setting.sh`` and the setup script it runs.

    The setup lives in ``enablement_setup.sh``, which starts no server, so the
    environment can be reproduced on its own; this script runs it and then
    launches with the accepted envs and server args.
    """
    from hyperloom.inference_optimizer.reference_script import LaunchSettings, render_reference_script

    from .setup_script import SETUP_SCRIPT_NAME, write_setup_script

    write_setup_script(session_dir, enablement)
    accepted_cfg = dict(enablement.accepted_config or {})
    text = render_reference_script(
        framework=framework,
        launch=LaunchSettings(
            # The accepted config can carry a harness flag the measurement ran with; an exported recipe must not.
            server_args=strip_benchmark_harness_flags(str(accepted_cfg.get("extra_server_args") or "").strip()),
            envs={str(k): str(v) for k, v in (accepted_cfg.get("extra_envs") or {}).items()},
        ),
        target=target,
        setup_script=SETUP_SCRIPT_NAME,
    )
    out = enablement_dir(Path(session_dir)) / "enablement_setting.sh"
    atomic_write_text(out, text, options=WriteOptions(make_parents=True, mode=0o700))
    return str(out.relative_to(session_dir))


__all__ = ["snapshot_round", "write_setting_script"]
