# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge-loop integration helpers for remote experience warm-start/write-back."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.workflows.forge_loop.campaign.workspace_files import tracked_editable_paths
from kernelforge.utils.git import git, git_apply, git_bytes, git_head
from kernelforge.knowledge.kb_store.identity.run import LoopKernel
from kernelforge.knowledge.kb_store.identity.implementation import (
    canonical_owner_framework,
)
from kernelforge.workflows.forge_loop.campaign.loop_log import kb_write_line
from hyperloom_common.io import DURABLE, atomic_write_text, fsync_directory

_KB_REFERENCES_REL = Path("forge_experiments") / "kb_references"


class WarmStartRollbackError(RuntimeError):
    """A rejected warm-start could not restore the original workspace."""


# The final warm-start implementation uses the more specific restore name while the CLI recovery boundary keeps the
# established rollback exception contract.
WarmStartRestoreError = WarmStartRollbackError


def _git_cumulative_diff(workspace_dir: str, base_sha: str) -> str:
    """Full diff from ``base_sha`` to HEAD (the run's net winning change)."""
    if not base_sha:
        return ""
    try:
        r = git_bytes("diff", base_sha, "HEAD", cwd=workspace_dir, check=False)
    except OSError:
        return ""
    return "" if r.returncode != 0 else r.stdout.decode("utf-8", errors="replace")


# Strip depths tried when applying a KB diff, in order.
_GIT_APPLY_STRIP_DEPTHS = (1, 2, 3, 4, 5, 6)


def _patch_paths(patch: str) -> list[str]:
    """Extract every source/destination path from a git-format patch."""
    paths: list[str] = []
    for match in re.finditer(
        r"^diff --git a/(\S+) b/(\S+)$",
        patch or "",
        re.MULTILINE,
    ):
        for path in match.groups():
            if path not in paths:
                paths.append(path)
    return paths


def _editable_workspace_paths(
    workspace_dir: str,
    kernel: str,
    source_files: list[str] | None,
    driver: str = "",
) -> set[str]:
    """Return tracked non-protected paths plus explicitly declared new files."""

    workspace = Path(workspace_dir).resolve()
    protected = [driver] if driver else []
    allowed = tracked_editable_paths(
        workspace_dir,
        exact_protected_paths=protected,
    )
    # Declarations are not an upper bound, but they may explicitly authorize a new implementation file that does not
    # exist in the pristine tree yet.
    for raw in [kernel, *(source_files or [])]:
        path = Path(raw)
        absolute = path.resolve() if path.is_absolute() else (workspace / path).resolve()
        try:
            relative = absolute.relative_to(workspace).as_posix()
        except ValueError:
            continue
        if not FORGE_PROTECTED_PATHS.is_protected(
            relative,
            workspace=workspace,
            exact_paths=protected,
        ):
            allowed.add(relative)
    return allowed


def _tracked_workspace_clean(workspace_dir: str) -> bool:
    """Return whether warm-start can exclusively own tracked workspace edits."""
    result = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        cwd=workspace_dir,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _safe_apply_depth(patch: str, allowed_paths: set[str]) -> int | None:
    """Find one strip depth mapping every patched path into the editable set."""
    changed = _patch_paths(patch)
    if not changed or not allowed_paths:
        return None
    for depth in _GIT_APPLY_STRIP_DEPTHS:
        mapped: list[str] = []
        for raw in changed:
            parts = Path(raw).parts
            strip_count = depth - 1
            if strip_count >= len(parts):
                break
            mapped.append(Path(*parts[strip_count:]).as_posix())
        if len(mapped) == len(changed) and set(mapped).issubset(allowed_paths):
            return depth
    return None


def _match_canonical_patch_path(
    raw_path: str,
    canonical_paths: set[str],
) -> str | None:
    """Resolve one producer patch path to exactly one canonical editable path."""
    raw = Path(raw_path).as_posix().lstrip("./")
    raw_parts = tuple(part for part in Path(raw).parts if part != "src")
    matches: list[str] = []
    for candidate in sorted(canonical_paths):
        candidate_parts = Path(candidate).parts
        direct = raw == candidate or raw.endswith(f"/{candidate}") or candidate.endswith(f"/{raw}")
        normalized_raw = tuple(canonical_owner_framework(part) for part in raw_parts)
        owner_match = False
        if candidate_parts:
            owner = candidate_parts[0]
            for index, part in enumerate(normalized_raw):
                if part == owner and normalized_raw[index:] == candidate_parts:
                    owner_match = True
                    break
        if direct or owner_match:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def _rewrite_patch_to_consumer_paths(
    patch: str,
    *,
    canonical_source_paths: set[str],
    consumer_source_map: dict[str, str],
    allowed_paths: set[str],
) -> str | None:
    """Rewrite producer paths using the canonical implementation identity."""
    raw_paths = _patch_paths(patch)
    if not raw_paths:
        return None
    replacements: dict[str, str] = {}
    for raw in raw_paths:
        canonical = _match_canonical_patch_path(raw, canonical_source_paths)
        target = consumer_source_map.get(canonical or "")
        if canonical is None or not target or target not in allowed_paths:
            return None
        replacements[raw] = target

    lines: list[str] = []
    for line in patch.splitlines(keepends=True):
        diff_match = re.match(r"^diff --git a/(\S+) b/(\S+)(\r?\n)?$", line)
        if diff_match:
            left, right = diff_match.group(1), diff_match.group(2)
            ending = diff_match.group(3) or ""
            lines.append(f"diff --git a/{replacements[left]} b/{replacements[right]}{ending}")
            continue
        header_match = re.match(r"^(--- a/|\+\+\+ b/)(\S+)(\r?\n)?$", line)
        if header_match:
            prefix, raw = header_match.group(1), header_match.group(2)
            lines.append(f"{prefix}{replacements[raw]}{header_match.group(3) or ''}")
            continue
        rename_match = re.match(r"^(rename from |rename to )(\S+)(\r?\n)?$", line)
        if rename_match:
            prefix, raw = rename_match.group(1), rename_match.group(2)
            lines.append(f"{prefix}{replacements[raw]}{rename_match.group(3) or ''}")
            continue
        lines.append(line)
    return "".join(lines)


@dataclass(frozen=True)
class _PatchTarget:
    """Where a KB patch may land: the editable paths, and how its canonical paths map onto this workspace."""

    allowed_paths: set[str]
    canonical_source_paths: set[str] | None = None
    consumer_source_map: dict[str, str] | None = None


def _git_apply(
    workspace_dir: str,
    patch: str,
    check_only: bool = False,
    *,
    target: _PatchTarget | None = None,
) -> bool:
    """Apply (or --check) a unified diff from stdin, normalizing the strip depth."""

    def _run(extra: list[str]) -> bool:
        # A depth that does not apply is the question being asked, not a failure.
        return git_apply(patch, *extra, "-", cwd=workspace_dir, check=False).returncode == 0

    depths = _GIT_APPLY_STRIP_DEPTHS
    if target is not None:
        allowed_paths = target.allowed_paths
        safe_depth = None
        if target.canonical_source_paths and target.consumer_source_map:
            rewritten = _rewrite_patch_to_consumer_paths(
                patch,
                canonical_source_paths=target.canonical_source_paths,
                consumer_source_map=target.consumer_source_map,
                allowed_paths=allowed_paths,
            )
            if rewritten is not None:
                rewritten_depth = _safe_apply_depth(rewritten, allowed_paths)
                if rewritten_depth is not None:
                    patch = rewritten
                    safe_depth = rewritten_depth
        if safe_depth is None:
            safe_depth = _safe_apply_depth(patch, allowed_paths)
        if safe_depth is None:
            return False
        depths = (safe_depth,)
    for depth in depths:
        pflag = f"-p{depth}"
        if not _run(["--check", pflag]):
            continue
        if check_only:
            return True
        return _run([pflag])
    return False


def _git_commit_all(
    workspace_dir: str,
    message: str,
    *,
    allowed_paths: set[str],
) -> str:
    """Commit exactly the approved non-protected paths, raising on failure."""
    before = git_head(workspace_dir)
    if not before:
        raise RuntimeError("could not resolve HEAD before warm-start commit")
    git("add", "-A", "-f", "--", *sorted(allowed_paths), cwd=workspace_dir)
    staged = git("diff", "--cached", "--name-only", cwd=workspace_dir)
    staged_paths = {line.strip() for line in staged.stdout.splitlines() if line.strip()}
    if not staged_paths or not staged_paths.issubset(allowed_paths):
        raise RuntimeError("warm-start staged files escape the approved path set")
    commit = git("commit", "-m", message, cwd=workspace_dir, check=False)
    if commit.returncode != 0:
        after_failed_commit = git_head(workspace_dir)
        if after_failed_commit and after_failed_commit != before:
            git("reset", "--mixed", before, cwd=workspace_dir, check=False)
        raise RuntimeError(f"git commit failed: {(commit.stderr or commit.stdout).strip()}")
    after = git_head(workspace_dir)
    if not after or after == before:
        raise RuntimeError("warm-start commit did not advance HEAD")
    committed = git("diff", "--name-only", before, after, cwd=workspace_dir, check=False)
    committed_paths = {line.strip() for line in committed.stdout.splitlines() if line.strip()}
    dirty = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        cwd=workspace_dir,
        check=False,
    )
    if (
        committed.returncode != 0
        or not committed_paths
        or not committed_paths.issubset(allowed_paths)
        or dirty.returncode != 0
        or bool(dirty.stdout.strip())
    ):
        git("reset", "--mixed", before, cwd=workspace_dir, check=False)
        _git_discard_worktree(workspace_dir)
        raise RuntimeError("warm-start commit verification failed or left tracked changes")
    return after


def _untracked_files(workspace_dir: str) -> set[str]:
    """Snapshot ignored and non-ignored untracked files without reading them."""
    paths: set[str] = set()
    commands = (
        ("ls-files", "--others", "--exclude-standard", "-z"),
        ("ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
    )
    for command in commands:
        result = git_bytes(*command, cwd=workspace_dir, check=False)
        if result.returncode != 0:
            raise WarmStartRestoreError("failed to snapshot pre-existing untracked files")
        paths.update(item.decode(errors="surrogateescape") for item in result.stdout.split(b"\0") if item)
    return paths


def _remove_new_untracked(
    workspace_dir: str,
    before: set[str],
) -> None:
    """Remove only untracked paths created after ``before`` was captured."""
    workspace = Path(workspace_dir).resolve()
    additions = _untracked_files(workspace_dir) - before
    for relative in sorted(additions, key=lambda value: len(Path(value).parts), reverse=True):
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise WarmStartRestoreError(f"unsafe untracked path reported by git: {relative}")
        target = workspace / rel_path
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        parent = target.parent
        while parent != workspace:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _git_discard_worktree(
    workspace_dir: str,
    pre_untracked: set[str] | None = None,
) -> bool:
    """Restore staged and unstaged tracked changes after a rejected candidate."""
    try:
        restored = git(
            "restore",
            "--source=HEAD",
            "--staged",
            "--worktree",
            "--",
            ".",
            cwd=workspace_dir,
            check=False,
        )
        status = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            cwd=workspace_dir,
            check=False,
        )
        if pre_untracked is not None:
            _remove_new_untracked(workspace_dir, pre_untracked)
    except (OSError, subprocess.SubprocessError, WarmStartRestoreError) as error:
        raise WarmStartRestoreError(f"failed to restore rejected warm-start: {error}") from error
    if restored.returncode != 0 or status.returncode != 0 or bool(status.stdout.strip()):
        detail = (restored.stderr or restored.stdout or status.stderr or status.stdout).strip()
        raise WarmStartRestoreError(f"failed to restore rejected warm-start: {detail or 'workspace remains dirty'}")
    return True


def _reference_markdown(sol: dict, rank: int) -> str:
    """Render one complete historical solution reference."""
    speedup = sol.get("speedup")
    speedup_text = f"{float(speedup):.6g}x" if isinstance(speedup, (int, float)) else "unknown"
    candidate_signature = str(sol.get("implementation_signature") or "")
    consumer_signature = str(sol.get("consumer_implementation_signature") or "")
    identity = sol.get("implementation_identity") if isinstance(sol.get("implementation_identity"), dict) else {}
    consumer_identity = (
        sol.get("consumer_implementation_identity")
        if isinstance(sol.get("consumer_implementation_identity"), dict)
        else {}
    )
    patch = str(sol.get("patch_content") or "")
    return (
        f"# Historical KB reference {rank:02d}\n\n"
        f"- Solution: `{sol.get('solution_slug', '')}`\n"
        f"- Speedup: {speedup_text}\n"
        f"- Implementation match: `{bool(sol.get('implementation_match'))}`\n"
        f"- Implementation signature: `{candidate_signature}`\n"
        f"- Consumer implementation signature: `{consumer_signature}`\n\n"
        "## Implementation identity\n\n"
        f"```json\n{json.dumps(identity, indent=2, sort_keys=True)}\n```\n\n"
        "## Consumer implementation identity\n\n"
        f"```json\n{json.dumps(consumer_identity, indent=2, sort_keys=True)}\n```\n\n"
        "## Strategy\n\n"
        f"{str(sol.get('strategy') or '(not recorded)')}\n\n"
        "## Recipe\n\n"
        f"{str(sol.get('recipe') or '(not recorded)')}\n\n"
        "## Lessons\n\n"
        f"{str(sol.get('lessons') or '(not recorded)')}\n\n"
        "## Complete patch diff\n\n"
        f"````diff\n{patch}\n````\n"
    )


def _reference_index_markdown(
    sols: list[dict],
    statuses: list[str],
    generation: str,
) -> str:
    """Render the ranked reference index with per-candidate apply outcomes."""
    lines = [
        "# KernelForge KB references",
        "",
        "Historical code solutions are design references. Validate any adapted "
        + "idea against the current implementation and full driver suite.",
        "",
    ]
    for index, sol in enumerate(sols):
        speedup = sol.get("speedup")
        speedup_text = f"{float(speedup):.6g}x" if isinstance(speedup, (int, float)) else "unknown"
        status = statuses[index] if index < len(statuses) else "not_attempted"
        lines.append(
            f"- Rank {index + 1}: "
            f"`sets/{generation}/reference_{index + 1:02d}.md` | "
            f"solution `{sol.get('solution_slug', '')}` | speedup {speedup_text} | "
            f"status `{status}`"
        )
    return "\n".join(lines) + "\n"


def _cleanup_old_reference_generations(root: Path, current: str) -> None:
    """Remove superseded generations only after the root index is published."""
    sets_root = root / "sets"
    if sets_root.is_dir():
        for path in sets_root.iterdir():
            if path.name == current:
                continue
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        fsync_directory(sets_root)


def _persist_kb_references(
    workspace_dir: str,
    sols: list[dict],
    statuses: list[str],
) -> Path:
    """Publish one complete immutable reference generation atomically."""
    root = Path(workspace_dir).resolve() / _KB_REFERENCES_REL
    sets_root = root / "sets"
    sets_root.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    temporary_generation = sets_root / f".{generation}.tmp"
    final_generation = sets_root / generation
    temporary_generation.mkdir()
    for rank, sol in enumerate(sols, start=1):
        atomic_write_text(
            temporary_generation / f"reference_{rank:02d}.md",
            _reference_markdown(sol, rank),
            options=DURABLE,
        )
    fsync_directory(temporary_generation)
    os.replace(temporary_generation, final_generation)
    fsync_directory(sets_root)
    atomic_write_text(
        root / "index.md",
        _reference_index_markdown(sols, statuses, generation),
        options=DURABLE,
    )
    with contextlib.suppress(OSError):
        _cleanup_old_reference_generations(root, generation)
    return root / "index.md"


def _clear_kb_references(workspace_dir: str) -> None:
    """Atomically retire stale fresh-lookup references, then remove them."""
    workspace = Path(workspace_dir).resolve()
    parent = workspace / _KB_REFERENCES_REL.parent
    root = parent / _KB_REFERENCES_REL.name
    if not root.exists() and not root.is_symlink():
        return
    retired = parent / f".{root.name}.cleared-{uuid.uuid4().hex}"
    os.replace(root, retired)
    fsync_directory(parent)
    if retired.is_symlink() or retired.is_file():
        retired.unlink()
    elif retired.is_dir():
        shutil.rmtree(retired)
    fsync_directory(parent)


def kb_reference_program_md(
    workspace_dir: str,
    *,
    applied_rank: int | None = None,
    solution_slug: str = "",
    detect_applied: bool = True,
) -> str:
    """Return the compact prompt pointer for persisted KB references."""
    index_path = Path(workspace_dir).resolve() / _KB_REFERENCES_REL / "index.md"
    if not index_path.is_file():
        return ""
    if applied_rank is None and detect_applied:
        with contextlib.suppress(OSError):
            for line in index_path.read_text(errors="replace").splitlines():
                match = re.match(
                    r"- Rank (\d+): .* solution `([^`]*)` .* status `applied`$",
                    line,
                )
                if match:
                    applied_rank = int(match.group(1))
                    solution_slug = match.group(2)
                    break
    parts = [
        "## Historical KB design references",
        "Read `forge_experiments/kb_references/index.md` and the referenced files "
        + "on demand. These historical code solutions are design references for "
        + "this search; their full metadata and diffs are stored there.",
    ]
    if applied_rank is not None:
        parts.append(f"Rank {applied_rank} solution `{solution_slug}` is already applied and is the search start.")
    return "\n".join(parts)


def mark_kb_reference_rejected(
    workspace_dir: str,
    rank: int,
    reason: str,
) -> None:
    """Update an applied index entry after external publication rollback."""
    index_path = Path(workspace_dir).resolve() / _KB_REFERENCES_REL / "index.md"
    if rank < 1 or not index_path.is_file():
        return
    text = index_path.read_text(errors="replace")
    pattern = re.compile(
        rf"(^- Rank {rank}: .* status `)applied(`$)",
        re.MULTILINE,
    )
    updated, count = pattern.subn(
        rf"\1rejected:{reason}\2",
        text,
        count=1,
    )
    if count:
        atomic_write_text(index_path, updated, options=DURABLE)


def _cheap_summary(archive: Any) -> dict:
    """Build a non-LLM experience summary from the on-disk candidate archive."""
    strategy = ""
    if archive is not None:
        index = archive.load_index()
        keeps = [
            entry for entry in index if entry.get("decision") == "KEEP" and entry.get("mean_case_speedup") is not None
        ]
        if keeps:
            best = max(keeps, key=lambda entry: entry["mean_case_speedup"])
            strategy = (best.get("plan") or "").strip()
    return {"category": "", "strategy": strategy, "recipe": "", "lessons": ""}


@dataclass(frozen=True)
class ExperiencePublication:
    """What the caller adds to the loop's own record when publishing: diff base, target, ids, anchors and summary."""

    base_sha: str
    gpu_target: str
    experience_id: str = ""
    pristine_baseline_ms: float | None = None
    reused_speedup: float | None = None
    snr_db_override: float | None = None
    llm_summary: bool = True
    incremental_summary: dict | None = None


def write_experience_to_kb(
    *,
    config,
    loop_runner: Any,
    kernel: LoopKernel,
    publication: ExperiencePublication,
    usage=None,
) -> dict:
    """Gather the run's outcome and mirror the best solution into the KB Store."""
    from kernelforge.knowledge.kb_store.writer import RunOutcome, write_run_experience

    checkpoint_experiment_id = getattr(loop_runner.experiment, "experiment_id", "") or ""
    kb_experience_id = publication.experience_id or checkpoint_experiment_id
    baseline_ms = (
        publication.pristine_baseline_ms
        or getattr(loop_runner.ic, "pristine_baseline_wall_ms", None)
        or getattr(loop_runner.ic, "baseline_wall_ms", None)
    )
    best_ms = getattr(loop_runner, "best_wall_ms", None)
    mean_case_speedup = getattr(loop_runner, "best_mean_case_speedup", None)
    cumulative_diff = _git_cumulative_diff(kernel.workspace, publication.base_sha)

    snr_db = None
    digest = ""
    archive = getattr(loop_runner, "archive", None)
    if archive is not None:
        with contextlib.suppress(OSError, ValueError, KeyError):
            keeps = [entry for entry in archive.load_index() if entry.get("decision") == "KEEP"]
            scored_keeps = [entry for entry in keeps if entry.get("mean_case_speedup") is not None]
            if scored_keeps:
                best_entry = max(
                    scored_keeps,
                    key=lambda entry: entry["mean_case_speedup"],
                )
                snr_db = best_entry.get("snr_db")
            digest = archive.render_digest()
    if publication.snr_db_override is not None:
        snr_db = publication.snr_db_override

    kernel_source = ""
    with contextlib.suppress(OSError):
        kernel_source = Path(kernel.kernel_path).read_text(errors="replace")

    summary_override = None if publication.llm_summary else publication.incremental_summary or _cheap_summary(archive)
    pristine_signature = getattr(loop_runner.ic, "implementation_signature", "")
    pristine_identity = getattr(loop_runner.ic, "implementation_identity", None)

    status = write_run_experience(
        config=config,
        kernel=replace(kernel, kernel_source=kernel_source),
        outcome=RunOutcome(
            experiment_id=kb_experience_id,
            gpu_target=publication.gpu_target,
            baseline_wall_ms=baseline_ms,
            best_wall_ms=best_ms,
            cumulative_diff=cumulative_diff,
            digest=digest,
            mean_case_speedup=mean_case_speedup,
            snr_db=snr_db,
            reused_speedup=publication.reused_speedup,
            implementation_signature=pristine_signature,
            implementation_identity=pristine_identity,
        ),
        summary_override=summary_override,
        usage=usage,
    )
    print(kb_write_line(status), flush=True)
    return status


def kb_read_status(warm: dict) -> dict:
    """Compact warm-start status safe to persist in result/experiment JSON."""
    writebacks = warm.get("measured_writebacks") or []
    return {
        "measured_writebacks": len(writebacks),
        "measured_writeback_failures": [
            str(item.get("reason") or "write_failed") for item in writebacks if not item.get("recorded")
        ],
        "candidate": bool(warm.get("candidate")),
        "read_reason": warm.get("read_reason", ""),
        "read_error": warm.get("read_error", ""),
        "applied": bool(warm.get("applied")),
        "match_mode": warm.get("match_mode", ""),
        "reference_reason": warm.get("reference_reason", ""),
        "solution_slug": warm.get("solution_slug", ""),
        "speedup": warm.get("speedup", 0.0),
        "pristine_ms": warm.get("pristine_ms"),
        "keep_baseline_ms": warm.get("keep_baseline_ms"),
        "applied_commit": warm.get("applied_commit", ""),
    }
