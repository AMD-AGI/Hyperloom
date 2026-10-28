# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge-loop integration helpers for remote experience warm-start/write-back."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kernelforge.workspace.edit_policy import (
    FORGE_PROTECTED_PATHS,
    tracked_editable_paths,
)
from kernelforge.tools.git import git, git_bytes, git_head
from kernelforge.knowledge.kb_store import warmstart_policy
from kernelforge.knowledge.kb_store.identity.run import LoopKernel
from kernelforge.knowledge.kb_store.identity.implementation import (
    canonical_owner_framework,
)
from kernelforge.loop.loop_log import kb_write_line
from kernelforge.tools.durable_io import atomic_write_text, fsync_directory
from kernelforge.evaluation.scoring import (
    DEFAULT_SNR_THRESHOLD_DB,
    KEEP_MEASUREMENT_COUNT,
    aggregate_regression_detail,
    keep_score,
    passes_keep_threshold,
)
from kernelforge.evaluation.benchmark.runner import (
    CaseCoverageError,
    aggregate_benchmark_measurements,
    calculate_measurement_case_speedups,
)
from kernelforge.evaluation.jit_rebuild import JitRebuildUnavailable

log = logging.getLogger(__name__)

# How much of the speedup a candidate was ranked on its own measurement has to reproduce for that ranking to count as
# honest. The regression this answers measured 32% below the claim that had won the ranking.
_WARMSTART_CLAIM_CONFIRMED_RATIO = 0.9

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
        return (
            git(
                "apply",
                *extra,
                "-",
                cwd=workspace_dir,
                input=patch,
                check=False,
            ).returncode
            == 0
        )

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
    except Exception as error:
        raise WarmStartRestoreError(f"failed to restore rejected warm-start: {error}") from error
    if restored.returncode != 0 or status.returncode != 0 or bool(status.stdout.strip()):
        detail = (restored.stderr or restored.stdout or status.stderr or status.stdout).strip()
        raise WarmStartRestoreError(f"failed to restore rejected warm-start: {detail or 'workspace remains dirty'}")
    return True


def _bench_once(driver: str, bench_repeat: int = 1) -> dict | None:
    """Run the driver's full benchmark suite once."""
    from kernelforge.evaluation.benchmark.runner import BenchRun, bench_wallclock

    try:
        res = asyncio.run(bench_wallclock(BenchRun(driver, driver_args=[], repeat=bench_repeat)))
        if not isinstance(res, dict) or not res.get("success") or not res.get("case_times"):
            return None
        return res
    except (OSError, ValueError) as e:  # a failed probe just disables warm-start
        print(f"  [kb] bench probe failed: {e}", flush=True)
        return None


def _correctness_once(driver: str, snr_threshold: float) -> bool:
    """Run the driver's complete SNR parity probe once."""
    from kernelforge.evaluation.correctness.driver import test_correctness

    try:
        res = asyncio.run(
            test_correctness(
                driver_script=driver,
                driver_args=[],
                snr_threshold=snr_threshold,
            )
        )
        return bool(res.get("passed")) if isinstance(res, dict) else False
    except (OSError, ValueError) as e:  # a failed probe just rejects warm-start
        print(f"  [kb] correctness probe failed: {e}", flush=True)
        return False


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
        )
    fsync_directory(temporary_generation)
    os.replace(temporary_generation, final_generation)
    fsync_directory(sets_root)
    atomic_write_text(
        root / "index.md",
        _reference_index_markdown(sols, statuses, generation),
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
        atomic_write_text(index_path, updated)


def _apply_candidate_patch(
    sol: dict,
    *,
    workspace_dir,
    allowed_paths,
    pre_untracked,
) -> str:
    """Put one candidate's diff in the working tree, or say why it did not land."""
    patch = sol.get("patch_content") or ""
    if not patch.strip():
        return "empty_patch"
    implementation_identity = (
        sol.get("implementation_identity") if isinstance(sol.get("implementation_identity"), dict) else {}
    )
    canonical_source_paths = {str(path) for path in implementation_identity.get("source_paths", []) if str(path)}
    consumer_source_map = sol.get("consumer_source_map") if isinstance(sol.get("consumer_source_map"), dict) else {}
    target = _PatchTarget(allowed_paths, canonical_source_paths, consumer_source_map)
    if not _git_apply(workspace_dir, patch, check_only=True, target=target):
        return "patch_touches_protected_path_or_not_applicable"
    if not _git_apply(workspace_dir, patch, target=target):
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        return "apply_failed"
    return ""


def _force_jit_rebuild(workspace_dir, kernel, source_files) -> None:
    """Invalidate the artifacts of the sources the applied patch just changed."""
    from kernelforge.evaluation.jit_rebuild import force_jit_rebuild_for_changes

    force_jit_rebuild_for_changes(
        workspace_dir,
        [path for path in [kernel, *(source_files or [])] if path],
    )


def _adopt_measured_candidate(
    sol: dict,
    *,
    kernel,
    workspace_dir,
    source_files,
    allowed_paths,
) -> tuple[str, str]:
    """Re-apply one already-measured candidate and commit it as the start."""
    pre_untracked = _untracked_files(workspace_dir)
    reject_reason = _apply_candidate_patch(
        sol,
        workspace_dir=workspace_dir,
        allowed_paths=allowed_paths,
        pre_untracked=pre_untracked,
    )
    if reject_reason:
        print(
            f"  [kb] warm-start candidate rejected: re-apply failed ({reject_reason})",
            flush=True,
        )
        return "", reject_reason
    try:
        _force_jit_rebuild(workspace_dir, kernel, source_files)
    except (JitRebuildUnavailable, subprocess.TimeoutExpired) as error:
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: rebuild failed ({error})",
            flush=True,
        )
        return "", "rebuild_failed"
    # The driver already rejected an incorrect or slower candidate before this
    # one was chosen, and re-applying the same patch to the same base it was
    # measured on reproduces what it measured. Judging it again here asked the
    # driver the same question, through a task configuration the engine had to
    # assume was written in one particular shape.
    try:
        commit = _git_commit_all(
            workspace_dir,
            f"kb warm-start: apply {sol.get('solution_slug', '')}",
            allowed_paths=allowed_paths,
        )
    except WarmStartRestoreError:
        raise
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: commit failed ({error})",
            flush=True,
        )
        return "", "commit_failed"
    return commit, ""


def _ranked_speedup(sol: dict) -> float | None:
    """The speedup a candidate was ranked on: its measurement, else its claim."""
    for value in (sol.get("measured_speedup"), sol.get("speedup")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if float(value) > 0.0:
            return float(value)
    return None


def _measurement_confirms_rank(sol: dict, measured_mean_case_speedup: float) -> bool:
    """Whether a measurement backs the speedup this candidate was ranked on."""
    ranked = _ranked_speedup(sol)
    if ranked is None:
        return False
    return measured_mean_case_speedup >= ranked * _WARMSTART_CLAIM_CONFIRMED_RATIO


def _outranks_remaining(
    measured_mean_case_speedup: float,
    remaining: list[dict],
) -> bool:
    """Whether no candidate left in the field is ranked above this measurement."""
    for sol in remaining:
        ranked = _ranked_speedup(sol)
        if ranked is not None and ranked > measured_mean_case_speedup:
            return False
    return True


def _record_measured_speedup(
    config,
    sol: dict,
    measured_mean_case_speedup: float,
    *,
    rank: int,
) -> dict:
    """Write one measured speedup back onto the KB record it was read from."""
    from kernelforge.knowledge.kb_store.errors import (
        kb_store_secrets,
        sanitize_read_error,
    )
    from kernelforge.knowledge.kb_store.recipe import KernelRecipeKB
    from kernelforge.knowledge.kb_store.record_store import KBStoreError, RewriteRecordError

    solution_slug = str(sol.get("solution_slug") or "")
    canonical_id = str(sol.get("kernel_slug") or "")
    session_id = str(sol.get("session_id") or "")
    if not canonical_id or not session_id:
        reason = "missing_record_address"
    else:
        try:
            outcome = KernelRecipeKB.open_canonical_id(
                canonical_id,
                config,
            ).record_measured_speedup(session_id, measured_mean_case_speedup)
        except (KBStoreError, RewriteRecordError, OSError, ValueError) as error:
            outcome = {
                "recorded": False,
                "reason": sanitize_read_error(
                    error,
                    secrets=kb_store_secrets(config),
                ),
            }
        reason = "" if outcome.get("recorded") else str(outcome.get("reason") or "write_failed")
    if reason:
        print(
            f"  [kb] warm-start measured write-back failed for {solution_slug}: {reason}",
            flush=True,
        )
    return {
        "rank": rank,
        "solution_slug": solution_slug,
        "measured_mean_case_speedup": measured_mean_case_speedup,
        "recorded": not reason,
        "reason": reason,
    }


def _rejected_reference_status(reason: str, writeback: dict | None) -> str:
    """The reference index status for a candidate this run did not adopt."""
    if writeback is None:
        return f"rejected:{reason}"
    measured = float(writeback["measured_mean_case_speedup"])
    outcome = "recorded" if writeback["recorded"] else "write-back refused"
    return f"rejected:{reason} (measured {measured:.6f}x {outcome})"


@dataclass(frozen=True)
class _CandidateTrial:
    """What trying one warm-start candidate established about it."""

    adoptable_ms: float | None
    adoptable_mean_case_speedup: float | None
    adoptable_bench: dict | None
    reject_reason: str
    measured_mean_case_speedup: float | None

    @classmethod
    def rejected(
        cls,
        reason: str,
        *,
        measured_mean_case_speedup: float | None,
    ) -> "_CandidateTrial":
        """A candidate that will not be adopted, and what it measured first."""
        return cls(None, None, None, reason, measured_mean_case_speedup)

    @classmethod
    def adoptable(
        cls,
        *,
        applied_ms: float,
        mean_case_speedup: float,
        bench: dict,
    ) -> "_CandidateTrial":
        """A candidate that cleared every measured gate at ``mean_case_speedup``."""
        return cls(applied_ms, mean_case_speedup, bench, "", mean_case_speedup)


@dataclass(frozen=True)
class WarmStartHarness:
    """How a warm-start candidate is judged on this machine: its driver, correctness threshold and bench repeats."""

    driver: str
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    bench_repeat: int = 1


@dataclass(frozen=True)
class _CandidateField:
    """What every candidate trial is judged against: the kernel, the harness, the pristine bench and editable paths."""

    kernel: LoopKernel
    harness: WarmStartHarness
    pristine_bench: dict
    allowed_paths: set[str]


def _try_apply_candidate(sol: dict, trial: _CandidateField, *, pre_untracked) -> _CandidateTrial:
    """Measure one candidate solution as a possible starting point."""
    workspace_dir, harness, pristine_bench = trial.kernel.workspace, trial.harness, trial.pristine_bench
    reject_reason = _apply_candidate_patch(
        sol,
        workspace_dir=workspace_dir,
        allowed_paths=trial.allowed_paths,
        pre_untracked=pre_untracked,
    )
    if reject_reason:
        return _CandidateTrial.rejected(
            reject_reason,
            measured_mean_case_speedup=None,
        )
    try:
        _force_jit_rebuild(workspace_dir, trial.kernel.kernel_path, trial.kernel.source_files)
        passed = _correctness_once(harness.driver, harness.snr_threshold)
        if not passed:
            _git_discard_worktree(
                workspace_dir,
                pre_untracked=pre_untracked,
            )
            return _CandidateTrial.rejected(
                "correctness_failed",
                measured_mean_case_speedup=None,
            )
        applied_runs = [_bench_once(harness.driver, harness.bench_repeat) for _ in range(KEEP_MEASUREMENT_COUNT)]
        applied_bench = aggregate_benchmark_measurements(applied_runs)
        try:
            measurement_scores = calculate_measurement_case_speedups(
                applied_bench,
                pristine_bench.get("case_times"),
                expected_measurements=KEEP_MEASUREMENT_COUNT,
            )
        except (AttributeError, CaseCoverageError):
            _git_discard_worktree(
                workspace_dir,
                pre_untracked=pre_untracked,
            )
            return _CandidateTrial.rejected(
                "case_coverage_failed",
                measured_mean_case_speedup=None,
            )
        mean_case_speedup = keep_score(measurement_scores)
        applied_bench["measurement_mean_case_speedups"] = measurement_scores
        applied_bench["mean_case_speedup"] = mean_case_speedup
        applied_ms = applied_bench.get("median_ms") if isinstance(applied_bench, dict) else None
        pristine_ms = pristine_bench.get("median_ms")
    except WarmStartRestoreError:
        raise
    except Exception as exc:  # isolation boundary: one KB warm-start candidate; worktree is discarded
        log.warning("warm-start candidate probe failed: %s", exc, exc_info=True)
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        return _CandidateTrial.rejected(
            "probe_failed",
            measured_mean_case_speedup=None,
        )

    # The suite ran, so this candidate measured something the KB record it came from can be amended with.
    measured_mean_case_speedup = float(mean_case_speedup)

    if (
        not isinstance(applied_ms, (int, float))
        or float(applied_ms) <= 0
        or not passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=1.0,
        )
    ):
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        return _CandidateTrial.rejected(
            "performance_failed",
            measured_mean_case_speedup=measured_mean_case_speedup,
        )

    # The gate below is only as closed as the baseline it is given: aggregate_regression_detail reports no
    # contradiction when either wall time is unknown -- correct for a run holding no best yet, wrong as an adoption
    # verdict -- so a pristine aggregate that is absent, non-numeric or not positive would pass a candidate on a
    # silent "" instead of on a comparison.
    if not isinstance(pristine_ms, (int, float)) or float(pristine_ms) <= 0:
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        print(
            "  [kb] warm-start candidate rejected: the pristine bench reported "
            "no usable aggregate wall time to compare against",
            flush=True,
        )
        return _CandidateTrial.rejected(
            "pristine_aggregate_missing",
            measured_mean_case_speedup=measured_mean_case_speedup,
        )

    # The keep gate above votes on the equal-weight mean of per-case speedups, which can clear the threshold while the
    # candidate is slower in aggregate wall time: a few cheap cases improving outvote one expensive case collapsing,
    # because that mean is unbounded above and bounded at 0 below.
    aggregate_regression = aggregate_regression_detail(
        baseline_ms=pristine_ms,
        best_ms=applied_ms,
        mean_case_speedup=mean_case_speedup,
    )
    if aggregate_regression:
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        print(
            f"  [kb] warm-start candidate rejected: {aggregate_regression}",
            flush=True,
        )
        return _CandidateTrial.rejected(
            "aggregate_regression",
            measured_mean_case_speedup=measured_mean_case_speedup,
        )

    return _CandidateTrial.adoptable(
        applied_ms=float(applied_ms),
        mean_case_speedup=float(mean_case_speedup),
        bench=applied_bench,
    )


def _resume_warmstart_result(workspace_dir) -> dict:
    """The warm-start verdict for a resumed run, which re-reads nothing.

    The field was measured and adopted in the run being resumed; all that is
    owed is the pointer to the references it left behind.
    """
    result = {
        "candidate": False,
        "skipped": "resume",
        "read_reason": "resume",
        "read_error": "",
    }
    pointer = kb_reference_program_md(workspace_dir)
    if pointer:
        result["program_md_addition"] = pointer
        result["reference_program_md_addition"] = pointer
    return result


@dataclass(frozen=True)
class _MeasuredField:
    """What measuring the warm-start field established, in rank order."""

    measurements: list[dict]
    writebacks: list[dict]
    reject_reason: str


def _measure_warmstart_field(
    sols: list[dict],
    statuses: list[str],
    *,
    config,
    trial: _CandidateField,
) -> _MeasuredField:
    """Measure candidates in rank order until one confirms its rank or the budget is spent.

    ``statuses`` is the per-candidate ledger and is filled in place. Every trial
    starts from the pristine tree, so a measured candidate is put back before
    the next one is tried; the winner is re-applied from its own patch later.

    ``top_k()`` caps the field, but a count does not bound wall time: one
    candidate is a compile plus a correctness suite plus a benchmark, minutes on
    the heaviest kernels. On expiry the field closes and the best already
    measured is adopted by the caller.
    """
    measurements: list[dict] = []
    writebacks: list[dict] = []
    reject_reason = ""
    search_deadline = time.monotonic() + warmstart_policy.budget_sec()
    for idx, sol in enumerate(sols):
        if time.monotonic() >= search_deadline:
            for later_index in range(idx, len(statuses)):
                statuses[later_index] = "not_attempted_search_budget"
            print(
                f"  [kb] warm-start search budget spent after {len(measurements)} measured candidate(s); "
                "adopting the best of them",
                flush=True,
            )
            break
        pre_untracked = _untracked_files(trial.kernel.workspace)
        outcome = _try_apply_candidate(sol, trial, pre_untracked=pre_untracked)
        # One write-back per measured candidate, adopted or not.
        writeback = None
        if outcome.measured_mean_case_speedup is not None:
            writeback = _record_measured_speedup(config, sol, outcome.measured_mean_case_speedup, rank=idx + 1)
            writebacks.append(writeback)
        if outcome.reject_reason:
            statuses[idx] = _rejected_reference_status(outcome.reject_reason, writeback)
            reject_reason = outcome.reject_reason
            continue
        _git_discard_worktree(trial.kernel.workspace, pre_untracked=pre_untracked)
        measurements.append(
            {
                "index": idx,
                "ms": float(outcome.adoptable_ms),
                "mean_case_speedup": float(outcome.adoptable_mean_case_speedup),
                "bench": dict(outcome.adoptable_bench or {}),
            }
        )
        ranked = _ranked_speedup(sol)
        ranked_txt = f"{ranked:.6f}x" if ranked is not None else "unrecorded"
        print(
            f"  [kb] warm-start rank {idx + 1} {sol.get('solution_slug')} measured "
            f"{float(outcome.adoptable_mean_case_speedup):.6f}x against a ranked {ranked_txt}",
            flush=True,
        )
        measured_now = float(outcome.adoptable_mean_case_speedup)
        if _measurement_confirms_rank(sol, measured_now) and _outranks_remaining(measured_now, sols[idx + 1 :]):
            for later_index in range(idx + 1, len(statuses)):
                statuses[later_index] = "not_attempted_after_apply"
            break
    return _MeasuredField(measurements, writebacks, reject_reason)


@dataclass
class _WarmStartAttempt:
    """One warm-start search: the candidate field, statuses, and the measured winner if any."""

    config: Any
    kernel: LoopKernel
    harness: WarmStartHarness
    read_status: dict[str, str] = field(
        default_factory=lambda: {"read_reason": "solution_pages_missing", "read_error": ""}
    )
    sols: list[dict] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    pristine_bench: dict = field(default_factory=dict)
    pristine_ms: float | None = None
    keep_baseline_ms: float | None = None
    mean_case_speedup: float | None = None
    applied_bench: dict = field(default_factory=dict)
    reference_reason: str = ""
    applied_commit: str = ""
    applied_idx: int | None = None
    measured_writebacks: list[dict] = field(default_factory=list)

    def load_solutions(self) -> None:
        """Read and filter the ranked field; clear references when the reader raises."""
        from kernelforge.knowledge.kb_store.reader import read_top_solutions

        kernel_source = ""
        with contextlib.suppress(OSError):
            kernel_source = Path(self.kernel.kernel_path).read_text(errors="replace")
        try:
            sols = read_top_solutions(
                config=self.config,
                kernel=replace(self.kernel, kernel_source=kernel_source),
                top_k=warmstart_policy.top_k(),
                read_status=self.read_status,
            )
        except Exception:
            _clear_kb_references(self.kernel.workspace)
            raise
        # A trial's cost scales with how slow the candidate is -- the correctness suite and the benchmark both run the
        # kernel -- so a port claiming to be orders of magnitude off the pace can spend the whole search budget on
        # itself. The claim only has to be right about the magnitude for that to be the wrong trade.
        admissible = [sol for sol in sols if not warmstart_policy.below_floor(_ranked_speedup(sol))]
        if len(admissible) != len(sols):
            print(
                f"  [kb] warm-start ignoring {len(sols) - len(admissible)} candidate(s) "
                f"claiming under {warmstart_policy.min_claimed_speedup():.2f}x",
                flush=True,
            )
        self.sols = admissible

    def measure_and_adopt(self) -> None:
        """Bench the pristine tree, measure the field, and adopt the best surviving candidate."""
        pristine_runs = [
            _bench_once(self.harness.driver, self.harness.bench_repeat) for _ in range(KEEP_MEASUREMENT_COUNT)
        ]
        self.pristine_bench = aggregate_benchmark_measurements(pristine_runs)
        self.pristine_ms = self.pristine_bench.get("median_ms")
        self.keep_baseline_ms = self.pristine_ms
        if self.pristine_ms is None:
            self.reference_reason = "baseline_unavailable"
            self.statuses = [f"rejected:{self.reference_reason}" for _ in self.sols]
            _persist_kb_references(self.kernel.workspace, self.sols, self.statuses)
            print(
                "  [kb] warm-start reference-only: baseline_unavailable; injecting top solutions as reference",
                flush=True,
            )
            return
        self.mean_case_speedup = 1.0
        allowed_paths = _editable_workspace_paths(
            self.kernel.workspace,
            self.kernel.kernel_path,
            self.kernel.source_files,
            self.harness.driver,
        )
        field = _measure_warmstart_field(
            self.sols,
            self.statuses,
            config=self.config,
            trial=_CandidateField(self.kernel, self.harness, self.pristine_bench, allowed_paths),
        )
        self.measured_writebacks = field.writebacks
        self.reference_reason = field.reject_reason
        # One entry per candidate that reached a measurement, in rank order.
        for measurement in sorted(
            field.measurements,
            key=lambda item: (-item["mean_case_speedup"], item["index"]),
        ):
            idx = measurement["index"]
            sol = self.sols[idx]
            self.applied_commit, reject_reason = _adopt_measured_candidate(
                sol,
                kernel=self.kernel.kernel_path,
                workspace_dir=self.kernel.workspace,
                source_files=self.kernel.source_files,
                allowed_paths=allowed_paths,
            )
            if reject_reason:
                self.statuses[idx] = f"rejected:{reject_reason}"
                self.reference_reason = reject_reason
                continue
            self.applied_idx = idx
            self.statuses[idx] = "applied"
            self.keep_baseline_ms = measurement["ms"]
            self.mean_case_speedup = measurement["mean_case_speedup"]
            self.applied_bench = dict(measurement["bench"])
            base_txt = f"{self.pristine_ms:.4f} ms" if self.pristine_ms is not None else "unmeasured"
            print(
                f"  [kb] warm-start applied: {sol.get('solution_slug')} "
                f"(rank {idx + 1}, prior speedup {sol.get('speedup')}, "
                f"measured mean case speedup "
                f"{measurement['mean_case_speedup']:.6f}x, "
                f"raw mean {measurement['ms']:.4f} ms vs baseline {base_txt})",
                flush=True,
            )
            break
        if self.applied_idx is not None:
            for measurement in field.measurements:
                other = measurement["index"]
                if other != self.applied_idx and self.statuses[other] == "not_attempted":
                    self.statuses[other] = f"rejected:outperformed_by_rank_{self.applied_idx + 1}"
        else:
            self.reference_reason = self.reference_reason or "no_candidate_applied"
            print(
                "  [kb] warm-start reference-only: no candidate applied "
                "cleanly + faster; injecting top solutions as reference",
                flush=True,
            )

    def final_result(self) -> dict:
        _persist_kb_references(self.kernel.workspace, self.sols, self.statuses)
        chosen = self.sols[self.applied_idx] if (self.applied_idx is not None) else self.sols[0]
        prompt_pointer = kb_reference_program_md(
            self.kernel.workspace,
            applied_rank=(None if self.applied_idx is None else self.applied_idx + 1),
            solution_slug=str(chosen.get("solution_slug") or "") if (self.applied_idx is not None) else "",
        )
        return {
            "candidate": True,
            **self.read_status,
            "applied": (self.applied_idx is not None),
            "match_mode": str(chosen.get("match_mode") or "reference"),
            "reference_reason": "" if (self.applied_idx is not None) else self.reference_reason,
            "pristine_ms": self.pristine_ms,
            "baseline_case_times": dict(self.pristine_bench.get("case_times") or {}),
            "baseline_unscored_cases": list(self.pristine_bench.get("unscored_cases") or []),
            "keep_baseline_ms": self.keep_baseline_ms,
            "mean_case_speedup": self.mean_case_speedup,
            "case_times": dict(self.applied_bench.get("case_times") or {}),
            "unscored_cases": list(self.applied_bench.get("unscored_cases") or []),
            "applied_commit": self.applied_commit,
            "program_md_addition": prompt_pointer,
            "reference_program_md_addition": kb_reference_program_md(
                self.kernel.workspace,
                detect_applied=False,
            ),
            "solution_slug": str(chosen.get("solution_slug") or ""),
            "speedup": chosen.get("speedup", 0.0),
            "num_references": len(self.sols),
            "applied_rank": (None if self.applied_idx is None else self.applied_idx + 1),
            "measured_writebacks": self.measured_writebacks,
        }


def kb_warmstart(
    *,
    config,
    kernel: LoopKernel,
    harness: WarmStartHarness,
    resume=False,
) -> dict:
    """Look up + apply the best prior solution as the loop's starting point.

    Candidates are measured on this machine and the fastest is adopted, because the speedup a record claims is not
    evidence this consumer can reproduce it. ``warmstart_policy`` bounds that search and is read at call time.
    """
    if resume:
        return _resume_warmstart_result(kernel.workspace)
    try:
        attempt = _WarmStartAttempt(config, kernel, harness)
        attempt.load_solutions()
        if not attempt.sols:
            _clear_kb_references(kernel.workspace)
            return {
                "candidate": False,
                "read_reason": attempt.read_status["read_reason"],
                "read_error": attempt.read_status["read_error"],
            }
        # Keep identity-match metadata populated by the reader (for example a
        # fuzzy donor's requested/selected canonical ids) while normalizing the
        # final read verdict.
        attempt.read_status.update(read_reason="hit", read_error="")
        attempt.statuses = ["not_attempted" for _ in attempt.sols]
        _persist_kb_references(kernel.workspace, attempt.sols, attempt.statuses)
        if not _tracked_workspace_clean(kernel.workspace):
            reason = "workspace_dirty"
            attempt.statuses = [f"rejected:{reason}" for _ in attempt.sols]
            _persist_kb_references(kernel.workspace, attempt.sols, attempt.statuses)
            reference = kb_reference_program_md(kernel.workspace)
            best = attempt.sols[0]
            return {
                "candidate": True,
                **attempt.read_status,
                "applied": False,
                "match_mode": str(best.get("match_mode") or "reference"),
                "reference_reason": reason,
                "pristine_ms": None,
                "keep_baseline_ms": None,
                "applied_commit": "",
                "program_md_addition": reference,
                "reference_program_md_addition": reference,
                "solution_slug": str(best.get("solution_slug") or ""),
                "speedup": best.get("speedup", 0.0),
                "num_references": len(attempt.sols),
                "applied_rank": None,
            }
        attempt.measure_and_adopt()
        return attempt.final_result()
    except WarmStartRestoreError:
        raise
    except Exception as e:  # noqa: BLE001 - isolation boundary: the run's KB warm-start; it cold-starts instead
        from kernelforge.knowledge.kb_store.errors import (
            kb_store_secrets,
            sanitize_read_error,
        )

        error = sanitize_read_error(e, secrets=kb_store_secrets(config))
        # Sanitized reason and stack frames only: the raw message may carry the KB store credentials.
        log.warning(
            "KB warm-start failed: %s (%s)\n%s", error, type(e).__name__, "".join(traceback.format_tb(e.__traceback__))
        )
        print(f"  [kb] warm-start skipped ({error})", flush=True)
        return {
            "candidate": False,
            "read_reason": "warm_start_error",
            "read_error": error,
        }


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
