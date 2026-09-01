# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge-loop integration helpers for remote experience warm-start/write-back."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.llm.workspace_policy import (
    is_protected_path,
    tracked_editable_paths,
)
from kernelforge.llm.git import git
from kernelforge.knowledge.implementation_identity import (
    canonical_owner_framework,
)
from kernelforge.durable_io import atomic_write_text, fsync_directory
from kernelforge.loop.canonical_correctness import accept_candidate
from kernelforge.loop.scoring import (
    DEFAULT_SNR_THRESHOLD_DB,
    KEEP_MEASUREMENT_COUNT,
    aggregate_regression_detail,
    keep_score,
    passes_keep_threshold,
)
from kernelforge.mcp_server.tools.bench import (
    CaseCoverageError,
    aggregate_benchmark_measurements,
    calculate_measurement_case_speedups,
)

# How many best-ranked prior solutions to read for warm-start. More than one so
# a champion that fails to apply -- a signature mismatch, a patch that no longer
# lands -- still leaves something to fall back to, and so a record whose claim
# does not survive measurement can lose to one that does.
_WARMSTART_TOP_K = 3

# How many candidates one warm start may fully evaluate. Each evaluation costs a
# correctness run plus KEEP_MEASUREMENT_COUNT benchmark runs on the real driver,
# so the search for the best measured start is bounded, not exhaustive.
_WARMSTART_MAX_MEASURED_CANDIDATES = 3

# How much of the speedup a candidate was ranked on its own measurement has to
# reproduce for that ranking to count as honest. A confirmed top candidate is
# adopted without paying for the rest; the regression this answers measured 32%
# below the claim that had won the ranking.
_WARMSTART_CLAIM_CONFIRMED_RATIO = 0.9

# Ceiling on the task's declared correctness suite when a warm start runs it,
# used when no caller passes the loop's own ``validate_stage_timeout_sec``. A
# candidate must not be judged under a looser clock for having arrived from the
# KB, and clamping can only turn a pass into a failure.
_WARMSTART_CANONICAL_TIMEOUT_CAP_SEC = 1800

_KB_REFERENCES_REL = Path("forge_experiments") / "kb_references"


class WarmStartRollbackError(RuntimeError):
    """A rejected warm-start could not restore the original workspace."""


# The final warm-start implementation uses the more specific restore name while
# the CLI recovery boundary keeps the established rollback exception contract.
WarmStartRestoreError = WarmStartRollbackError


def git_head(workspace_dir: str) -> str:
    """Return the current HEAD sha of ``workspace_dir`` (empty on failure).

    Every caller reads this as "the commit to anchor to, if there is one" and
    supplies its own anchor otherwise, so an unborn HEAD or a directory that is
    not a repository is an answer here rather than an error.
    """
    try:
        return git("rev-parse", "HEAD", cwd=workspace_dir, check=False).stdout.strip()
    except OSError:
        return ""


def git_checkout_branch(workspace_dir: str, branch: str) -> str:
    """Create/switch to the loop branch before any warm-start edits.

    Branch existence is probed with ``git rev-parse`` (locale-independent)
    rather than matching git's localizable "already exists" message.
    """
    if not branch:
        return ""
    try:
        exists = (
            git(
                "rev-parse",
                "--verify",
                "--quiet",
                f"refs/heads/{branch}",
                cwd=workspace_dir,
                check=False,
            ).returncode
            == 0
        )
        r = git(
            "checkout",
            *(() if exists else ("-b",)),
            branch,
            cwd=workspace_dir,
            check=False,
        )
        return (r.stdout + "\n" + r.stderr).strip()
    except OSError as e:
        return f"checkout failed: {e}"


def _git_cumulative_diff(workspace_dir: str, base_sha: str) -> str:
    """Full diff from ``base_sha`` to HEAD (the run's net winning change).

    Captured as bytes and decoded without newline translation. ``text=True``
    would fold every ``\\r\\n`` in the diff to ``\\n``, and a patch is applied by
    matching its context lines byte for byte: against a CRLF source the folded
    patch no longer describes any file, so ``git apply`` rejects it and the
    solution is unreusable while still looking perfectly well-formed.
    """
    if not base_sha:
        return ""
    try:
        r = git("diff", base_sha, "HEAD", cwd=workspace_dir, check=False, text=False)
    except OSError:
        return ""
    return "" if r.returncode != 0 else r.stdout.decode("utf-8", errors="replace")


# Strip depths tried when applying a KB diff, in order. A KB patch is produced
# by ``git diff`` in the PRODUCER's workspace, so its ``a/`` ``b/`` paths are
# relative to that git root. The consumer's workspace root may sit at a different
# depth (e.g. a nested package copy), so ``-p1`` alone can miss. Trying a few
# strip depths absorbs a "consumer is deeper" layout difference without risky
# path rewriting; each real apply is preceded by ``--check`` so a wrong depth
# never half-applies. (A "consumer is shallower" layout can't be fixed by
# stripping — the workspace-root constraint in the Hyperloom launcher handles
# that; see the design doc §2.3/§3.3.)
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
    # Declarations are not an upper bound, but they may explicitly authorize a
    # new implementation file that does not exist in the pristine tree yet.
    for raw in [kernel, *(source_files or [])]:
        path = Path(raw)
        absolute = path.resolve() if path.is_absolute() else (workspace / path).resolve()
        try:
            relative = absolute.relative_to(workspace).as_posix()
        except ValueError:
            continue
        if not is_protected_path(
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


def _git_apply(
    workspace_dir: str,
    patch: str,
    check_only: bool = False,
    *,
    allowed_paths: set[str] | None = None,
    canonical_source_paths: set[str] | None = None,
    consumer_source_map: dict[str, str] | None = None,
) -> bool:
    """Apply (or --check) a unified diff from stdin, normalizing the strip depth.

    Tries ``-p1`` first (the normal ``git diff`` layout), then deeper strips.
    Returns True on the first depth that applies. When ``check_only`` is set,
    only the dry-run ``--check`` is attempted (no mutation). The real apply path
    always ``--check``s a depth before applying it, so the working tree is never
    left half-patched by a wrong depth.
    """

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
    if allowed_paths is not None:
        safe_depth = None
        if canonical_source_paths and consumer_source_map:
            rewritten = _rewrite_patch_to_consumer_paths(
                patch,
                canonical_source_paths=canonical_source_paths,
                consumer_source_map=consumer_source_map,
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
    allowed_paths: set[str] | None = None,
) -> str:
    """Commit exactly the approved non-protected paths, raising on failure."""
    before = git_head(workspace_dir)
    if not before:
        raise RuntimeError("could not resolve HEAD before warm-start commit")
    if allowed_paths is None:
        changed = git("diff", "--name-only", "HEAD", cwd=workspace_dir)
        allowed_paths = {line.strip() for line in changed.stdout.splitlines() if line.strip()}
    git("add", "-A", "--", *sorted(allowed_paths), cwd=workspace_dir)
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
        result = git(*command, cwd=workspace_dir, check=False, text=False)
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
    """Run the driver's full benchmark suite once.

    ``bench_repeat`` must match what the loop itself uses. This value can become
    the loop's keep threshold, and comparing a single-shot probe against
    repeat-and-median candidates injects a systematic offset (measured at 3.7% on
    the TP4 all-reduce suite) that the KEEP gate reads as a free improvement.

    The driver owns the source-to-artifact contract for its backend. A successful
    result therefore means the currently patched source was built or JIT-compiled
    as required before measurement.
    """
    from kernelforge.mcp_server.tools.bench import bench_wallclock

    try:
        repeat_kwargs = {"repeat": bench_repeat} if bench_repeat > 1 else {}
        res = asyncio.run(bench_wallclock(driver_script=driver, driver_args=[], **repeat_kwargs))
        if not isinstance(res, dict) or not res.get("success") or not res.get("case_times"):
            return None
        return res
    except Exception as e:  # noqa: BLE001 - a failed probe just disables warm-start
        print(f"  [kb] bench probe failed: {e}", flush=True)
        return None


def _correctness_once(driver: str, snr_threshold: float) -> bool:
    """Run the driver's complete SNR parity probe once.

    Mirrors the loop's pre-filter, so an obviously broken candidate is dropped
    before it is benchmarked. It decides nothing: adoption is decided by the
    task's own correctness suite in ``_adopt_measured_candidate``.
    Returns True only when the driver reports a passing metric; any
    failure/crash/timeout returns False so warm-start treats it as a reject.
    """
    from kernelforge.mcp_server.tools.test import test_correctness

    try:
        res = asyncio.run(
            test_correctness(
                driver_script=driver,
                driver_args=[],
                snr_threshold=snr_threshold,
            )
        )
        return bool(res.get("passed")) if isinstance(res, dict) else False
    except Exception as e:  # noqa: BLE001 - a failed probe just rejects warm-start
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
    for legacy in root.glob("reference_*.md"):
        legacy.unlink()
    fsync_directory(root)


def _persist_kb_references(
    workspace_dir: str,
    sols: list[dict],
    statuses: list[str],
) -> Path:
    """Publish one complete immutable reference generation atomically.

    The stable root index is the commit point. Before its replacement, the old
    index continues to reference an intact old generation. After replacement,
    the new index references a fully written and durably renamed new generation.
    Superseded generations are removed only after that commit point.
    """
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
    """Put one candidate's diff in the working tree, or say why it did not land.

    Returns an empty string once the patch is applied. A patch that is refused
    leaves the tree exactly as it was found, so the caller can move on to the
    next candidate without a restore of its own.
    """
    patch = sol.get("patch_content") or ""
    if not patch.strip():
        return "empty_patch"
    implementation_identity = (
        sol.get("implementation_identity") if isinstance(sol.get("implementation_identity"), dict) else {}
    )
    canonical_source_paths = {str(path) for path in implementation_identity.get("source_paths", []) if str(path)}
    consumer_source_map = sol.get("consumer_source_map") if isinstance(sol.get("consumer_source_map"), dict) else {}
    if not _git_apply(
        workspace_dir,
        patch,
        check_only=True,
        allowed_paths=allowed_paths,
        canonical_source_paths=canonical_source_paths,
        consumer_source_map=consumer_source_map,
    ):
        return "patch_touches_protected_path_or_not_applicable"
    if not _git_apply(
        workspace_dir,
        patch,
        allowed_paths=allowed_paths,
        canonical_source_paths=canonical_source_paths,
        consumer_source_map=consumer_source_map,
    ):
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        return "apply_failed"
    return ""


def _force_jit_rebuild(workspace_dir, kernel, source_files) -> None:
    """Invalidate the artifacts of the sources the applied patch just changed."""
    from kernelforge.loop.jit_rebuild import force_jit_rebuild_for_changes

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
    canonical_timeout_cap_sec: int,
) -> tuple[str, str]:
    """Re-apply one already-measured candidate and commit it as the start.

    This is the moment a historical kernel becomes this run's incumbent, so it
    is where the shared acceptance step runs: the candidate is judged by the
    task's own correctness suite before the adopting commit exists, and a
    failure returns the same rejection the caller already handles.

    Returns the commit and an empty reason, or an empty commit and the reason
    the candidate could not be adopted. Both outcomes leave the working tree
    free of a partially adopted patch, so the caller can try the next best
    measured candidate on a clean tree.
    """
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
    except WarmStartRestoreError:
        raise
    except Exception as error:  # noqa: BLE001 - a stale artifact must not be kept
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: rebuild failed ({error})",
            flush=True,
        )
        return "", "rebuild_failed"
    try:
        canonical = asyncio.run(
            accept_candidate(
                workspace_dir,
                timeout_cap_sec=canonical_timeout_cap_sec,
                candidate_label=(f"KB warm-start {sol.get('solution_slug', '')}".strip()),
            )
        )
    except Exception as error:  # noqa: BLE001 - a suite forge cannot run rejects
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: the canonical correctness suite could not be run ({error})",
            flush=True,
        )
        return "", "canonical_correctness_failed"
    if not canonical.passed:
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: the task's own correctness suite failed ({canonical.detail})",
            flush=True,
        )
        return "", "canonical_correctness_failed"
    try:
        commit = _git_commit_all(
            workspace_dir,
            f"kb warm-start: apply {sol.get('solution_slug', '')}",
            allowed_paths=allowed_paths,
        )
    except WarmStartRestoreError:
        raise
    except Exception as error:  # noqa: BLE001 - reported as a rejected candidate
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
    """Whether a measurement backs the speedup this candidate was ranked on.

    An honest record is worth no more trials on its own account, but that alone
    does not end the search: see :func:`_outranks_remaining`.
    """
    ranked = _ranked_speedup(sol)
    if ranked is None:
        return False
    return measured_mean_case_speedup >= ranked * _WARMSTART_CLAIM_CONFIRMED_RATIO


def _outranks_remaining(
    measured_mean_case_speedup: float,
    remaining: list[dict],
) -> bool:
    """Whether no candidate left in the field is ranked above this measurement.

    Ranking puts every measured candidate ahead of every merely claimed one
    however large the claim, and a record only earns a measurement by being
    adopted, so a later rank routinely claims more than the leader. Stopping on
    a confirmed leader alone would therefore pin warm start to the first record
    that was ever measured and leave every better solution published since
    unevaluated forever. A ranked value is the most a candidate can deliver if
    it is honest, so matching the best of them is what ends the search.
    """
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
    """Write one measured speedup back onto the KB record it was read from.

    Without this the KB keeps ranking an unverified claim forever, since nothing
    else ever compares it against a measurement. A store that refuses the
    amendment cannot fail the run, so the outcome is returned for the warm-start
    result and printed; it is never dropped.

    The amendment sanitizes what it raises itself, but opening the record's
    address does not: ``create_rewrite_record_store`` builds the store client
    from the KB Store URL and bearer token and lets anything that is not a
    ``KBStoreError`` out. This reason is persisted, so the exception is redacted
    and bounded here as well.
    """
    from kernelforge.knowledge.experience_reader import sanitize_read_error
    from kernelforge.rewrite_by_flydsl.agent_kb import (
        KernelRecipeKB,
        kb_store_secrets,
    )

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
        except Exception as error:  # noqa: BLE001 - reported below, never fatal
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
    """The reference index status for a candidate this run did not adopt.

    ``writeback`` is the outcome of amending the candidate's KB record, or None
    when the candidate left no measurement to amend it with. An operator reading
    the index has to be able to tell those apart from the entry itself: whether a
    rejected candidate corrected the record it came from decides whether the same
    claim is going to lead the ranking again tomorrow. A refusal names itself
    here and carries its reason in ``measured_writeback_failures``.
    """
    if writeback is None:
        return f"rejected:{reason}"
    measured = float(writeback["measured_mean_case_speedup"])
    outcome = "recorded" if writeback["recorded"] else "write-back refused"
    return f"rejected:{reason} (measured {measured:.6f}x {outcome})"


@dataclass(frozen=True)
class _CandidateTrial:
    """What trying one warm-start candidate established about it.

    ``reject_reason`` is empty exactly when the candidate is adoptable, and the
    three ``adoptable_`` values are set only then, so a rejected candidate cannot
    be read as an adopted one. Together those four are the adoption verdict.

    ``measured_mean_case_speedup`` is deliberately not one of them: it is the
    value the KB record this candidate came from has to be amended with, and it
    survives rejection. A candidate whose driver suite was benchmarked measured
    something whether or not it then cleared the gate, and the records carrying
    the most inflated claims are precisely the ones that lose. It is ``None``
    when no benchmark completed, which is not evidence a later run can rank on.
    """

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
        """A candidate that cleared every measured gate at ``mean_case_speedup``.

        The task's own correctness suite has not judged it yet: that runs once,
        on the candidate this field of measured candidates wins with, as it is
        adopted.
        """
        return cls(applied_ms, mean_case_speedup, bench, "", mean_case_speedup)


def _try_apply_candidate(
    sol: dict,
    *,
    kernel,
    driver,
    workspace_dir,
    snr_threshold,
    source_files,
    pristine_bench,
    allowed_paths,
    pre_untracked,
    bench_repeat=1,
) -> _CandidateTrial:
    """Measure one candidate solution as a possible starting point.

    Applies the candidate's diff to the working tree, rebuilds JIT sources, and
    validates it end to end on the consumer's complete driver suite. A KB lookup
    already establishes the logical operator; implementation identity remains
    diagnostic metadata and never suppresses a safe trial. The patch may touch
    any tracked non-protected file, must pass the SNR pre-filter, and must beat
    the pristine baseline on both measures the loop reports: the per-case mean
    has to clear the KEEP threshold and the aggregate wall time has to be faster
    than the pristine aggregate. ``pristine_bench`` must therefore carry both
    halves of that measurement -- ``case_times`` and ``median_ms`` -- and a
    candidate is refused rather than adopted unmeasured when either is missing.
    On success, returns an adoptable
    :class:`_CandidateTrial` carrying the raw mean, the mean case speedup and the
    complete benchmark result. On rejection it cleanly restores the tree and
    returns a rejected trial naming the reason, which tells an aggregate
    regression apart from a threshold miss, plus the measurement the suite
    produced before losing -- see :class:`_CandidateTrial`.

    A historical solution is adopted only if it works and clears the same
    full-suite performance gate used by the optimization loop: an adopted
    candidate becomes this run's incumbent, so it is held to the bar every
    later candidate is. The measured candidate is left in the working tree for
    the caller to keep or discard.
    """
    reject_reason = _apply_candidate_patch(
        sol,
        workspace_dir=workspace_dir,
        allowed_paths=allowed_paths,
        pre_untracked=pre_untracked,
    )
    if reject_reason:
        return _CandidateTrial.rejected(
            reject_reason,
            measured_mean_case_speedup=None,
        )
    try:
        _force_jit_rebuild(workspace_dir, kernel, source_files)
        passed = _correctness_once(driver, snr_threshold)
        if not passed:
            _git_discard_worktree(
                workspace_dir,
                pre_untracked=pre_untracked,
            )
            return _CandidateTrial.rejected(
                "correctness_failed",
                measured_mean_case_speedup=None,
            )
        applied_runs = [_bench_once(driver, bench_repeat) for _ in range(KEEP_MEASUREMENT_COUNT)]
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
    except Exception:
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        return _CandidateTrial.rejected(
            "probe_failed",
            measured_mean_case_speedup=None,
        )

    # The suite ran, so this candidate measured something the KB record it came
    # from can be amended with.
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

    # The gate below is only as closed as the baseline it is given:
    # aggregate_regression_detail reports no contradiction when either wall time
    # is unknown -- correct for a run holding no best yet, wrong as an adoption
    # verdict -- so a pristine aggregate that is absent, non-numeric or not
    # positive would pass a candidate on a silent "" instead of on a comparison.
    # The per-case half of the same measurement is already mandatory a few lines
    # above, so the aggregate is required here rather than left to the caller's
    # discipline. The reason is named apart from aggregate_regression: this
    # candidate was never compared to a baseline at all, which is a broken
    # baseline rather than a slow candidate.
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

    # The keep gate above votes on the equal-weight mean of per-case speedups,
    # which can clear the threshold while the candidate is slower in aggregate
    # wall time: a few cheap cases improving outvote one expensive case
    # collapsing, because that mean is unbounded above and bounded at 0 below.
    # Adopting such a candidate would start the run from a baseline worse than
    # pristine. This is the invariant the published manifest already refuses to
    # badge, so the warm-start gate reuses its derivation instead of open-coding
    # a comparison the two could drift apart on. The reason is named apart from
    # performance_failed because this candidate did clear the threshold.
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


def kb_warmstart(
    *,
    config,
    kernel,
    driver,
    workspace_dir,
    kernel_backend,
    target_functions=None,
    framework="",
    snr_threshold=DEFAULT_SNR_THRESHOLD_DB,
    source_files=None,
    operator_name="",
    resume=False,
    bench_repeat=1,
    canonical_timeout_cap_sec=_WARMSTART_CANONICAL_TIMEOUT_CAP_SEC,
) -> dict:
    """Look up + apply the best prior solution as the loop's starting point.

    ``target_functions`` and framework identity are forwarded so the read
    resolves the same kernel slug the write side uses when the anchor is a
    wrapper. Must stay in sync with ``write_experience_to_kb``.

    Candidates arrive ranked on measured evidence ahead of bare claims. Up to
    ``_WARMSTART_MAX_MEASURED_CANDIDATES`` of them are measured on this machine
    and the best measured one is adopted, because the number a record claims is
    not evidence that this consumer can reproduce it: adopting the first
    candidate that merely applied is what let an inflated claim displace a
    verified better start. The search stops early once a candidate reproduces
    the value it was ranked on.

    Every measurement is written back to its own KB record so the next run ranks
    that record on evidence, including the measurement of a candidate this run
    then rejected: a record only loses the gate by promising more than this
    machine delivers, so those are the claims most in need of correcting. A
    rejected candidate is never adoptable, whatever it measured.

    Every solution returned for the logical operator may be attempted regardless
    of implementation-signature or declared-source drift: the protected
    measurement boundary constrains the patch, and the canonical driver owns
    backend-specific build/JIT behavior and must prove correctness plus a strict
    pristine-performance improvement before the patch is accepted. Every
    candidate is persisted as reference material. ``snr_threshold`` is the cheap
    parity pre-filter, not the gate: the candidate this run adopts is accepted by
    the task's own correctness suite through the shared acceptance step, under
    ``canonical_timeout_cap_sec``. ``source_files`` is forwarded to
    ``force_jit_rebuild`` for frameworks that require explicit cache
    invalidation.
    """
    if resume:
        pointer = kb_reference_program_md(workspace_dir)
        result = {
            "candidate": False,
            "skipped": "resume",
            "read_reason": "resume",
            "read_error": "",
        }
        if pointer:
            result["program_md_addition"] = pointer
            result["reference_program_md_addition"] = pointer
        return result
    try:
        from kernelforge.knowledge.experience_reader import read_top_solutions

        read_status = {
            "read_reason": "solution_pages_missing",
            "read_error": "",
        }
        kernel_source = ""
        with contextlib.suppress(Exception):
            kernel_source = Path(kernel).read_text(errors="replace")

        try:
            read_kwargs = {
                "config": config,
                "kernel_path": kernel,
                "kernel_source": kernel_source,
                "kernel_backend": kernel_backend,
                "target_functions": target_functions,
                "framework": framework,
                "top_k": _WARMSTART_TOP_K,
                "source_files": source_files,
                "workspace": workspace_dir,
                "operator_name": operator_name,
            }
            reader_parameters = inspect.signature(read_top_solutions).parameters
            if "read_status" in reader_parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in reader_parameters.values()
            ):
                read_kwargs["read_status"] = read_status
            sols = read_top_solutions(**read_kwargs)
        except Exception:
            _clear_kb_references(workspace_dir)
            raise
        if not sols:
            _clear_kb_references(workspace_dir)
            return {
                "candidate": False,
                "read_reason": read_status["read_reason"],
                "read_error": read_status["read_error"],
            }

        read_status = {"read_reason": "hit", "read_error": ""}

        statuses = ["not_attempted" for _ in sols]
        _persist_kb_references(workspace_dir, sols, statuses)
        best = sols[0]
        if not _tracked_workspace_clean(workspace_dir):
            reason = "workspace_dirty"
            statuses = [f"rejected:{reason}" for _ in sols]
            _persist_kb_references(workspace_dir, sols, statuses)
            reference = kb_reference_program_md(workspace_dir)
            return {
                "candidate": True,
                **read_status,
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
                "num_references": len(sols),
                "applied_rank": None,
            }

        applied = False
        applied_idx: int | None = None
        pristine_ms: float | None = None
        keep_baseline_ms: float | None = None
        mean_case_speedup: float | None = None
        applied_bench: dict = {}
        reference_reason = ""
        applied_commit = ""
        # One entry per candidate that reached a measurement, in rank order.
        measurements: list[dict] = []
        measured_writebacks: list[dict] = []
        pristine_runs = [_bench_once(driver, bench_repeat) for _ in range(KEEP_MEASUREMENT_COUNT)]
        pristine_bench = aggregate_benchmark_measurements(pristine_runs)
        pristine_ms = (
            pristine_bench.get("median_ms")
            if isinstance(pristine_bench, dict)
            else pristine_bench
            if isinstance(pristine_bench, (int, float))
            else None
        )
        keep_baseline_ms = pristine_ms
        mean_case_speedup = 1.0 if pristine_ms is not None else None
        if pristine_ms is None:
            reference_reason = "baseline_unavailable"
            statuses = [f"rejected:{reference_reason}" for _ in sols]
            _persist_kb_references(workspace_dir, sols, statuses)
            print(
                "  [kb] warm-start reference-only: baseline_unavailable; injecting top solutions as reference",
                flush=True,
            )
        else:
            allowed_paths = _editable_workspace_paths(
                workspace_dir,
                kernel,
                source_files,
                driver,
            )
            for idx, sol in enumerate(sols):
                if len(measurements) >= _WARMSTART_MAX_MEASURED_CANDIDATES:
                    statuses[idx] = "not_attempted_after_apply"
                    continue
                pre_untracked = _untracked_files(workspace_dir)
                trial = _try_apply_candidate(
                    sol,
                    kernel=kernel,
                    driver=driver,
                    workspace_dir=workspace_dir,
                    snr_threshold=snr_threshold,
                    source_files=source_files,
                    pristine_bench=pristine_bench,
                    allowed_paths=allowed_paths,
                    pre_untracked=pre_untracked,
                    bench_repeat=bench_repeat,
                )
                # One write-back per measured candidate, adopted or not. A
                # rejected candidate is the one whose record most likely carries
                # an inflated claim -- an overstated number is what loses the
                # gate -- so leaving it unamended is what lets the same claim win
                # rank 1, be applied and benchmarked, and lose again on every
                # later run. Both outcomes report through measured_writebacks, so
                # a store that refuses either is equally visible.
                writeback = None
                if trial.measured_mean_case_speedup is not None:
                    writeback = _record_measured_speedup(
                        config,
                        sol,
                        trial.measured_mean_case_speedup,
                        rank=idx + 1,
                    )
                    measured_writebacks.append(writeback)
                if trial.reject_reason:
                    statuses[idx] = _rejected_reference_status(
                        trial.reject_reason,
                        writeback,
                    )
                    reference_reason = trial.reject_reason
                    continue
                # Every trial starts from the pristine tree, so a measured
                # candidate is put back before the next one is tried; the
                # candidate that wins the field is re-applied from its own patch.
                _git_discard_worktree(
                    workspace_dir,
                    pre_untracked=pre_untracked,
                )
                measurements.append(
                    {
                        "index": idx,
                        "ms": float(trial.adoptable_ms),
                        "mean_case_speedup": float(trial.adoptable_mean_case_speedup),
                        "bench": dict(trial.adoptable_bench or {}),
                    }
                )
                ranked = _ranked_speedup(sol)
                ranked_txt = f"{ranked:.6f}x" if ranked is not None else "unrecorded"
                print(
                    f"  [kb] warm-start rank {idx + 1} "
                    f"{sol.get('solution_slug')} measured "
                    f"{float(trial.adoptable_mean_case_speedup):.6f}x "
                    f"against a ranked {ranked_txt}",
                    flush=True,
                )
                measured_now = float(trial.adoptable_mean_case_speedup)
                if _measurement_confirms_rank(sol, measured_now) and (
                    _outranks_remaining(measured_now, sols[idx + 1 :])
                ):
                    for later_index in range(idx + 1, len(statuses)):
                        statuses[later_index] = "not_attempted_after_apply"
                    break

            for measurement in sorted(
                measurements,
                key=lambda item: (-item["mean_case_speedup"], item["index"]),
            ):
                idx = measurement["index"]
                sol = sols[idx]
                applied_commit, reject_reason = _adopt_measured_candidate(
                    sol,
                    kernel=kernel,
                    workspace_dir=workspace_dir,
                    source_files=source_files,
                    allowed_paths=allowed_paths,
                    canonical_timeout_cap_sec=canonical_timeout_cap_sec,
                )
                if reject_reason:
                    statuses[idx] = f"rejected:{reject_reason}"
                    reference_reason = reject_reason
                    continue
                applied = True
                applied_idx = idx
                statuses[idx] = "applied"
                keep_baseline_ms = measurement["ms"]
                mean_case_speedup = measurement["mean_case_speedup"]
                applied_bench = dict(measurement["bench"])
                base_txt = f"{pristine_ms:.4f} ms" if pristine_ms is not None else "unmeasured"
                print(
                    f"  [kb] warm-start applied: {sol.get('solution_slug')} "
                    f"(rank {idx + 1}, prior speedup {sol.get('speedup')}, "
                    f"measured mean case speedup "
                    f"{measurement['mean_case_speedup']:.6f}x, "
                    f"raw mean {measurement['ms']:.4f} ms vs baseline {base_txt})",
                    flush=True,
                )
                break

            if applied:
                for measurement in measurements:
                    other = measurement["index"]
                    if other != applied_idx and statuses[other] == "not_attempted":
                        statuses[other] = f"rejected:outperformed_by_rank_{applied_idx + 1}"
            else:
                reference_reason = reference_reason or "no_candidate_applied"
                print(
                    "  [kb] warm-start reference-only: no candidate applied "
                    "cleanly + faster; injecting top solutions as reference",
                    flush=True,
                )

        _persist_kb_references(workspace_dir, sols, statuses)
        chosen = sols[applied_idx] if applied else best
        prompt_pointer = kb_reference_program_md(
            workspace_dir,
            applied_rank=(applied_idx + 1) if applied_idx is not None else None,
            solution_slug=str(chosen.get("solution_slug") or "") if applied else "",
        )
        return {
            "candidate": True,
            **read_status,
            "applied": applied,
            "match_mode": str(chosen.get("match_mode") or "reference"),
            "reference_reason": "" if applied else reference_reason,
            "pristine_ms": pristine_ms,
            "baseline_case_times": (
                dict(pristine_bench.get("case_times") or {}) if isinstance(pristine_bench, dict) else {}
            ),
            "baseline_unscored_cases": (
                list(pristine_bench.get("unscored_cases") or []) if isinstance(pristine_bench, dict) else []
            ),
            "keep_baseline_ms": keep_baseline_ms,
            "mean_case_speedup": mean_case_speedup,
            "case_times": dict(applied_bench.get("case_times") or {}),
            "unscored_cases": list(applied_bench.get("unscored_cases") or []),
            "applied_commit": applied_commit,
            "program_md_addition": prompt_pointer,
            "reference_program_md_addition": kb_reference_program_md(
                workspace_dir,
                detect_applied=False,
            ),
            "solution_slug": str(chosen.get("solution_slug") or ""),
            "speedup": chosen.get("speedup", 0.0),
            "num_references": len(sols),
            "applied_rank": (applied_idx + 1) if applied_idx is not None else None,
            "measured_writebacks": measured_writebacks,
        }
    except WarmStartRestoreError:
        raise
    except Exception as e:  # noqa: BLE001 - warm-start must never break the run
        from kernelforge.knowledge.experience_reader import sanitize_read_error

        error = sanitize_read_error(
            e,
            secrets=(
                str(getattr(config, "gbrain_token", "") or ""),
                os.environ.get("GBRAIN_TOKEN", ""),
            ),
        )
        print(f"  [kb] warm-start skipped ({error})", flush=True)
        return {
            "candidate": False,
            "read_reason": "warm_start_error",
            "read_error": error,
        }


def _cheap_summary(archive: Any) -> dict:
    """Build a non-LLM experience summary from the on-disk candidate archive.

    Used by the incremental publish (invoked inside the running loop on every new
    best): it must not spend ~150s on an LLM call or nest an event loop. Strategy
    is taken from the best kept iteration's ``plan``. Free-form per-iteration
    records are not compressed into a synthetic lesson field. The final graceful
    write later overwrites the same page with the precise LLM summary.
    """
    strategy = ""
    if archive is not None:
        try:
            index = archive.load_index()
            keeps = [
                entry
                for entry in index
                if entry.get("decision") == "KEEP" and entry.get("mean_case_speedup") is not None
            ]
            if keeps:
                best = max(keeps, key=lambda entry: entry["mean_case_speedup"])
                strategy = (best.get("plan") or "").strip()
        except Exception:  # noqa: BLE001 - best-effort; empty summary is acceptable
            pass
    return {"category": "", "strategy": strategy, "recipe": "", "lessons": ""}


def write_experience_to_kb(
    *,
    config,
    loop_runner: Any,
    workspace_dir,
    kernel,
    kernel_backend,
    gpu_target,
    base_sha,
    pristine_baseline_ms=None,
    source_files=None,
    target_functions=None,
    framework="",
    experience_id="",
    operator_name="",
    implementation_signature_value="",
    implementation_identity_value=None,
    llm_summary=True,
    incremental_summary=None,
    snr_db_override=None,
    reused_speedup=None,
    usage=None,
) -> dict:
    """Gather the run's outcome and mirror the best solution into the KB Store.

    ``source_files`` and ``target_functions`` make the identity correct for
    repository tasks (the operation is the real entry and dtypes are parsed from
    the file that defines it). Must stay in sync with ``kb_warmstart`` so
    read/write slugs match.

    ``llm_summary`` controls the experience prose: True (final graceful write)
    pays for the LLM summary; False (incremental publish on each new best) uses a
    cheap archive-derived summary so it neither stalls the loop nor nests an
    event loop. Both write to the same per-run solution page, so the final write
    upgrades the interim one in place.
    """
    try:
        from kernelforge.knowledge.experience_sink import write_run_experience

        checkpoint_experiment_id = getattr(loop_runner.experiment, "experiment_id", "") or ""
        kb_experience_id = experience_id or checkpoint_experiment_id
        baseline_ms = (
            pristine_baseline_ms
            or getattr(loop_runner.ic, "pristine_baseline_wall_ms", None)
            or getattr(loop_runner.ic, "baseline_wall_ms", None)
        )
        best_ms = getattr(loop_runner, "best_wall_ms", None)
        mean_case_speedup = getattr(loop_runner, "best_mean_case_speedup", None)
        cumulative_diff = _git_cumulative_diff(workspace_dir, base_sha)

        snr_db = None
        digest = ""
        archive = getattr(loop_runner, "archive", None)
        if archive is not None:
            with contextlib.suppress(Exception):
                keeps = [entry for entry in archive.load_index() if entry.get("decision") == "KEEP"]
                scored_keeps = [entry for entry in keeps if entry.get("mean_case_speedup") is not None]
                if scored_keeps:
                    best_entry = max(
                        scored_keeps,
                        key=lambda entry: entry["mean_case_speedup"],
                    )
                    snr_db = best_entry.get("snr_db")
                digest = archive.render_digest()
        if snr_db_override is not None:
            snr_db = snr_db_override

        kernel_source = ""
        with contextlib.suppress(Exception):
            kernel_source = Path(kernel).read_text(errors="replace")

        summary_override = None if llm_summary else incremental_summary or _cheap_summary(archive)
        pristine_signature = implementation_signature_value or getattr(loop_runner.ic, "implementation_signature", "")
        pristine_identity = implementation_identity_value or getattr(loop_runner.ic, "implementation_identity", None)

        status = write_run_experience(
            config=config,
            workspace=workspace_dir,
            kernel_path=kernel,
            kernel_source=kernel_source,
            kernel_backend=kernel_backend,
            gpu_target=gpu_target,
            experiment_id=kb_experience_id,
            baseline_wall_ms=baseline_ms,
            best_wall_ms=best_ms,
            mean_case_speedup=mean_case_speedup,
            cumulative_diff=cumulative_diff,
            digest=digest,
            snr_db=snr_db,
            source_files=source_files,
            target_functions=target_functions,
            operator_name=operator_name,
            implementation_signature_override=pristine_signature,
            implementation_identity_override=pristine_identity,
            framework=framework,
            summary_override=summary_override,
            reused_speedup=reused_speedup,
            usage=usage,
        )
        if status.get("written"):
            print(
                f"  [kb] experience written: {status.get('solution')} (speedup {status.get('speedup'):.3f})", flush=True
            )
        else:
            print(f"  [kb] experience not written: {status.get('reason')}", flush=True)
        return status
    except Exception as e:  # noqa: BLE001 - never let KB write affect the run
        print(f"  [kb] experience write skipped ({e})", flush=True)
        return {"written": False, "reason": f"error:{e!r}"}


def kb_read_status(warm: dict) -> dict:
    """Compact warm-start status safe to persist in result/experiment JSON.

    A refused amendment leaves the KB ranking a claim no consumer reproduced,
    which is the condition this run was supposed to correct, so it is summarized
    here rather than living only in the console log. Bounded by the number of
    candidates a warm start may measure.
    """
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
