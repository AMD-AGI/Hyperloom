# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy enablement round deliverables into ``reports/enablement/<task_id>/``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import atomic_write_text
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

# Server logs routinely exceed _FILE_SIZE_LIMIT, so they are truncated rather than skipped.
_SERVER_LOG_TAIL_LIMIT = 1024 * 1024

_LOG_TRUNCATION_NOTE = "[hyperloom] truncated: the first {dropped} bytes are missing; the tail follows.\n"


def _copy(src: Path, dest: Path) -> bool:
    """Copy ``src`` to ``dest`` when it exists and is under the size limit."""
    if not src.is_file() or src.stat().st_size > _FILE_SIZE_LIMIT:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return True


def _artifact_archive_name(idx: int, target: str) -> str:
    """Archive filename for the ``idx``-th artifact a round installed.

    Derived from the round-local index so ``write_setting_script`` addresses a
    copy by name rather than by position in the archive directory.
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
    atomic_write_text(dest, text, make_parents=True)
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

    patches_dir = round_dir / "patches"
    copied: set[str] = set()
    for applied in res.get("patches_applied") or []:
        src = Path(str(applied))
        dest = patches_dir / src.name
        if _copy(src, dest):
            archive.record(ROLE_PATCH, dest)
        # Marked seen even when refused, so the sweep below does not retry it.
        copied.add(src.name)

    workspace = runs_dir(root, "specialist", task_id)
    for name, role in (("specialist_done.json", ROLE_SPECIALIST_RESULT), ("prompt.md", ROLE_PROMPT)):
        dest = round_dir / name
        if _copy(workspace / name, dest):
            archive.record(role, dest)

    # Disk scans include rejected output: preserve evidence, not accepted patches.
    for base in (workspace, workspace / "worktree"):
        for pattern in ("*.patch", "*.diff"):
            for src in sorted((base / "patches").glob(pattern)):
                if src.name in copied:
                    continue
                dest = round_dir / "attempted_patches" / src.name
                if _copy(src, dest):
                    archive.record(ROLE_PATCH_EVIDENCE, dest)
                copied.add(src.name)

    artifacts_dir = round_dir / "artifacts"
    for idx, art in enumerate(res.get("artifacts_applied") or []):
        name = _artifact_archive_name(idx, str(art.get("target") or ""))
        source = str(art.get("source") or "").strip()
        backup = str(art.get("backup") or "").strip()
        if source and _copy(Path(source), artifacts_dir / name):
            archive.record(ROLE_ARTIFACT_SOURCE, artifacts_dir / name)
        if backup and _copy(Path(backup), artifacts_dir / f"{name}.orig"):
            archive.record(ROLE_ARTIFACT_PREIMAGE, artifacts_dir / f"{name}.orig")

    accepted_config = str(res.get("enablement_accepted_config_path") or "").strip()
    if accepted_config:
        dest = round_dir / "launch_config.yaml"
        if _copy(Path(accepted_config), dest):
            archive.record(ROLE_LAUNCH_CONFIG, dest)

    # Only a round that reached a bench has one: a rejected patch or a broken build never started a server, so an
    # absent log is normal.
    bench = res.get("bench_result")
    server_log = str(bench.get("server_log_path") or "").strip() if isinstance(bench, dict) else ""
    if server_log:
        dest = round_dir / "server.log"
        if _copy_log_tail(Path(server_log), dest):
            archive.record(ROLE_SERVER_LOG, dest)

    return archive


def _hunk_extent(header: str) -> tuple[int, int]:
    """Return the old-side and new-side line counts a hunk header announces.

    Kept as a pair rather than a sum: a context line belongs to both sides, so
    adding them over-counts the body by the number of context lines, and the
    hunk then swallows the next file section's headers.
    """
    import re as _re

    match = _re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", header)
    if match is None:
        return 0, 0
    return int(match.group(1) or 1), int(match.group(2) or 1)


def _is_null_path(path: str) -> bool:
    """Whether a diff header names no file, as a creation or deletion does."""
    return path.strip().lstrip("ab/").strip("/") in ("dev/null", "") or path.strip().endswith("/dev/null")


def _top_level_of(path: str) -> str:
    """Return the first real directory a diff header names, or ``""``."""
    parts = [p for p in path.split("/") if p not in ("", ".")]
    # git's ``a/``, ``b/``, and the ``a2/``/``b3/`` forms it writes when one diff
    # spans several trees, name no directory.
    while (
        parts and len(parts[0]) <= 2 and parts[0][0] in "ab" and parts[0][1:].isdigit() or parts[:1] in (["a"], ["b"])
    ):
        parts.pop(0)
    return parts[0] if len(parts) > 1 else ""


def _root_the_patches_name(patches_dest: Path, script_rounds: list[dict], framework_root: str) -> str:
    """Return the root the archived patches actually target.

    ``enablement.framework_root`` is the root of whichever round last set it, so
    a run that patched a second tree at any point leaves it naming that tree.
    The replay script is written from the accumulated rounds, and naming a root
    the patches do not belong to makes every one of them fail to apply -- the
    script is then unusable, and says nothing about why.

    The patches name their own tree in their headers. When every one of them
    agrees on a top-level directory and the recorded root's own name differs,
    the siblings of the recorded root are searched for that name; the recorded
    root is kept when they agree, when the patches disagree among themselves, or
    when no sibling matches -- a guessed path is worse than a wrong one that at
    least came from the run.
    """
    recorded = Path(framework_root) if framework_root else None
    if recorded is None:
        return framework_root
    names: set[str] = set()
    for rnd in script_rounds:
        for rel in rnd.get("patches") or []:
            source = patches_dest.parent / rel
            try:
                # Every file section, not only the first: one patch can touch
                # two trees, and stopping at the head would redirect the root to
                # whichever happened to come first.
                pending_old = ""
                old_left = new_left = 0
                named_here = False
                with source.open(encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        if old_left > 0 or new_left > 0:
                            # Inside a hunk. A removed line reading ``-- x`` and
                            # an added one reading ``++ x`` render with exactly
                            # the header prefixes, so payload is counted out by
                            # side rather than pattern-matched.
                            head = line[:1]
                            if head == "\\":  # "\ No newline at end of file"
                                continue
                            if head == "-":
                                old_left -= 1
                                continue
                            if head == "+":
                                new_left -= 1
                                continue
                            if head in (" ", "\n", "\r"):
                                old_left -= 1
                                new_left -= 1
                                continue
                            old_left = new_left = 0
                        if line.startswith("@@"):
                            old_left, new_left = _hunk_extent(line)
                            continue
                        if line.startswith("diff --git "):
                            # Rename-only, mode-only and some binary patches
                            # carry no ``---``/``+++`` pair at all; this header
                            # is the only place they name their tree.
                            for token in line[len("diff --git ") :].split():
                                top = _top_level_of(token)
                                if top:
                                    names.add(top)
                                    named_here = True
                            continue
                        if line.startswith("--- "):
                            pending_old = line[4:].strip().split("\t")[0]
                            continue
                        if not line.startswith("+++ "):
                            continue
                        new_path = line[4:].strip().split("\t")[0]
                        # A creation's old side is /dev/null and a deletion's new
                        # side is; the other side is the one that names the tree.
                        chosen = new_path if _is_null_path(pending_old) else pending_old
                        if _is_null_path(chosen):
                            chosen = pending_old if _is_null_path(new_path) else new_path
                        pending_old = ""
                        top = _top_level_of(chosen)
                        if top:
                            names.add(top)
                            named_here = True
            except OSError:
                return framework_root
            if not named_here:
                # A patch this could not classify may target another tree, and
                # redirecting on the strength of the ones it could read would
                # leave that one unreplayable.
                return framework_root
    if len(names) != 1:
        return framework_root
    wanted = names.pop()
    if recorded.name == wanted:
        return framework_root
    sibling = recorded.parent / wanted
    return str(sibling) if sibling.is_dir() else framework_root


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
    """Write ``reports/enablement/enablement_setting.sh`` from accumulated enablement state."""
    from hyperloom.inference_optimizer.reference_script import render_reference_script
    from hyperloom.orchestrator.bringup.trees import tree_kind

    root = Path(session_dir)
    framework_root = str(enablement.framework_root or "").strip()
    patches_dest = enablement_dir(root) / "patches"
    artifacts_dest = enablement_dir(root) / "artifacts"

    patch_counter = 0
    artifact_counter = 0
    script_rounds: list[dict] = []
    effective_framework_root = ""

    for rnd in enablement.kept_rounds or []:
        rnd_script_patches: list[str] = []
        rnd_script_artifacts: list[dict[str, str]] = []

        task_id = str(rnd.get("task_id") or "").strip()
        round_dir = enablement_round_dir(root, task_id) if task_id else None

        if framework_root and round_dir is not None:
            patches_archive = round_dir / "patches"
            archived_patches = sorted(patches_archive.glob("*.patch")) if patches_archive.is_dir() else []
            for src in archived_patches:
                patch_counter += 1
                name = f"{patch_counter:03d}_{src.name}"
                if _copy(src, patches_dest / name):
                    rnd_script_patches.append(f"patches/{name}")

        if round_dir is not None:
            art_archive = round_dir / "artifacts"
            for idx, art in enumerate(rnd.get("artifacts") or []):
                artifact_counter += 1
                target = str(art.get("target") or "")
                archived = art_archive / _artifact_archive_name(idx, target)
                name = f"{artifact_counter:03d}_{Path(target).name}"
                if not _copy(archived, artifacts_dest / name):
                    continue
                _copy(archived.parent / f"{archived.name}.orig", artifacts_dest / f"{name}.orig")
                rnd_script_artifacts.append({"archive_path": f"artifacts/{name}", "target": target})

        if rnd_script_patches or rnd_script_artifacts:
            script_rounds.append({"patches": rnd_script_patches, "artifacts": rnd_script_artifacts})

    accepted_cfg = dict(enablement.accepted_config or {})
    extra_envs = {str(k): str(v) for k, v in (accepted_cfg.get("extra_envs") or {}).items()}
    extra_server_args = str(accepted_cfg.get("extra_server_args") or "").strip()

    active = enablement.active_runtime or {}
    runtime_path = str(active.get("venv_root") or "").strip() if isinstance(active, dict) else ""

    if any(r.get("patches") for r in script_rounds):
        effective_framework_root = _root_the_patches_name(patches_dest, script_rounds, framework_root)

    text = render_reference_script(
        framework=framework,
        server_args=extra_server_args,
        envs=extra_envs,
        model=model,
        tp=tp,
        max_model_len=max_model_len,
        gpu_type=gpu_type,
        setup_commands=list(enablement.setup_commands or []) or None,
        framework_root=effective_framework_root or None,
        # Derived from the root the script will actually name: correcting a git
        # checkout to an installed package while still announcing "git" emits
        # ``git -C "$FRAMEWORK_ROOT" apply`` against a tree with no repository.
        framework_root_vcs=tree_kind(effective_framework_root) if effective_framework_root else "",
        runtime=runtime_path or None,
        rounds=script_rounds or None,
    )

    out = enablement_dir(Path(session_dir)) / "enablement_setting.sh"
    atomic_write_text(out, text, make_parents=True, mode=0o700)
    return str(out.relative_to(session_dir))


__all__ = ["snapshot_round", "write_setting_script"]
