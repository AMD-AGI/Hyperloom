# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""IntegratePatchExecutor — PR-A4 (Arbor-into-Hyperloom).

Serving-lane-locked patch integration: consumes a specialist's worktree
patches, applies them to the live framework source roots, runs a
throughput + optional accuracy gate, then KEEPs (advances the stack) or
REVERTs (rolls back the tree).

Deterministic Python executor (no LLM). Per Inv-5.1, this is the single
allowed ``git apply`` channel against framework_source_roots (specialists
author patches into their isolated worktree only).

Inputs (``ctx.task.params``)::

    specialist_task_id (str, required) — completed specialist task
        whose worktree under ``runs/specialist/<task_id>/`` carries
        the patches.
    patches (list[str], optional) — explicit patch paths. Defaults to
        ``specialist_done.patches_written``.
    config_changes (dict[str, str], optional) — env vars layered on
        the variant's launch env. Reverted with the patches on REVERT.
    keep_threshold_pct (float, optional) — first-pass KEEP threshold;
        defaults to DEFAULT_KEEP_THRESHOLD_PCT (1.0), the grid noise floor.
        A KEEP is then re-confirmed by a full-stack rebench unless
        ``enable_stack_rebench`` is False.
    accuracy_baseline (float, optional) — baseline accuracy for the gate.
        With a positive baseline the measured drop is enforced; without one,
        the gate skips with a warning.
    enable_stack_rebench (bool, optional) — when True (default) a KEEP
        is confirmed by a second full-stack rebench (stability floor +
        accuracy) before it is committed.
    rebench_stable_threshold_pct (float, optional) — stability floor for
        the confirmation rebench, as a percentage above ``base_tput``
        (default 0.0).
    base_tput (float, optional) — baseline throughput to compare
        against. Falls back to ``SharedState.baseline_tput`` if zero.
    benchmark_script / result_dir / variant_timeout_sec — same
        semantics as the explore executor's params.
    framework_source_root (str, optional) — explicit override for the
        ``git apply`` target. Defaults to the first existing entry of
        ``resolve_source_file_allowlist()``.
    apply_only (bool, optional) — when True, skip the benchmark step
        entirely (used by tests + a future smoke-only mode). The
        executor still applies the patches but returns
        ``status='applied_no_bench'`` so downstream bookkeeping can
        differentiate from a genuine KEEP/REVERT.

Outputs (dict, returned to the bus as ``delegated_result.result``)::

    status: "kept" | "reverted" | "apply_failed" | "no_patches" |
            "applied_no_bench" | "failed"
    output_throughput: float | None
    delta_pct: float | None
    accuracy_pass: bool | None
    patches_applied: list[str]
    patches_reverted: list[str]
    config_changes_applied: dict[str, str]
    reason: str
    specialist_task_id: str
    workspace: str
    bench_result: dict | None
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from .._time import now_iso
from ..framework_paths import resolve_source_file_allowlist
from ..specialist_patch_safety import patch_file_targets, patch_targets_missing
from ._accuracy_gate import accuracy_passed, parse_eval_results
from ._git import _run_git_cp
from ._grid_runner import (
    GridVariant,
    VariantResult,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._stack_rebench import measure_stack_rebench
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


DEFAULT_KEEP_THRESHOLD_PCT = 1.0  # grid noise floor; KEEP is re-confirmed by a stack rebench
DEFAULT_VARIANT_TIMEOUT_SEC = 7800  # 130 min; aligns with BASELINE_DEFAULT_TIMEOUT_SEC for Qwen3-32B TP=1 long workload
_HYPERLOOM_AUTO_STASH_MSG = "hyperloom-auto-stash: preserving user changes before candidate run"


# Bare ``isoformat()`` (auto timespec): microseconds only when non-zero.
_now_iso = functools.partial(now_iso, "auto")


def _root_contains_patch_targets(root: Path, patch_paths: list[Path]) -> bool:
    """True when *every* supplied patch has all its modify/delete targets
    present under ``root`` (at some ``-p`` strip level).

    A patch only applies in the package tree that actually contains the files
    it edits; ``vllm/...`` patches can never apply under the ``aiter`` root.
    Returns False if any patch is unreadable or has a missing target here.
    """
    if not patch_paths:
        return False
    for patch in patch_paths:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if patch_targets_missing(text, root):
            return False
    return True


def _resolve_framework_root(
    explicit: str | None, patch_paths: list[Path] | None = None,
) -> Path | None:
    """Pick the framework source root for patches.

    Precedence: explicit param → first allowlist root whose tree actually
    contains the patch targets (target-aware: a ``vllm/...`` patch must apply
    under the vllm root, not the first allowlist entry which is ``aiter``) →
    first existing git root → first existing dir. None when nothing resolves.

    Args:
        explicit: Explicit framework-root override, or ``None`` to use the
            allowlist.
        patch_paths: Patch target paths used to pick the allowlist root whose
            tree actually contains them.

    Returns:
        The resolved framework source root, or ``None`` when nothing resolves.
    """
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p
        log.warning(
            "integrate_patch: framework_source_root override %r does not exist; falling back to allowlist",
            explicit,
        )
    roots = [Path(r) for r in resolve_source_file_allowlist()]
    # Target-aware: prefer the root that actually holds the patch's targets.
    if patch_paths:
        for p in roots:
            if p.is_dir() and _root_contains_patch_targets(p, patch_paths):
                return p
    for p in roots:
        if p.is_dir() and (p / ".git").exists():
            return p
    # Last resort: a non-git dir (prefer surfacing as clean apply_failed).
    for p in roots:
        if p.is_dir():
            return p
    return None


# Candidate ``-p`` strip levels, tried in priority order. ``-p1`` is the
# git-native default and stays first for backward-compat; specialists author
# patches with heterogeneous path prefixes (``a/vllm/...`` -> -p1,
# ``b/_aiter_ops.py`` -> -p0/-p2, full absolute
# ``b/usr/local/lib/python3.12/dist-packages/vllm/...`` -> -p7), so we must
# auto-detect rather than assume a single level.
_P_LEVELS: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8)


def _run_git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    p_level: int,
    three_way: bool,
    check_only: bool,
) -> tuple[bool, str]:
    """Single ``git apply`` invocation at an explicit strip level.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        p_level: The ``-p<N>`` strip level.
        three_way: Whether to pass ``-3`` for a three-way merge.
        check_only: Whether to pass ``--check`` (dry run, no mutation).

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True on a zero return code.
    """
    args = ["-C", str(framework_root), "apply", f"-p{p_level}"]
    if three_way:
        args.append("-3")
    if check_only:
        args.append("--check")
    args.append(str(patch_path))
    cp = _run_git_cp(args, timeout=120.0)
    if cp is None:
        return False, "git apply spawn failed"
    return cp.returncode == 0, cp.stderr.strip()


def _preflight_missing_targets(
    framework_root: Path,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """Return per-patch records for patches whose modify/delete targets are
    absent from ``framework_root`` at every ``-p`` strip level.

    A hallucinated-layout patch (e.g. modifying a CUDA-only file on a ROCm
    build) can never apply; flagging it here yields an actionable advisory
    instead of an opaque ``git_apply_failed`` after a wasted apply attempt.
    Defense-in-depth: ``specialist_patch_safety`` already drops these at
    authoring time, but patches supplied directly via ``params.patches``
    bypass that gate.

    Args:
        framework_root: The git checkout the patches target.
        patch_paths: The patch files to preflight.

    Returns:
        A list of per-patch records (``patch`` + ``missing_targets``) for
        patches whose targets are absent at every strip level.
    """
    records: list[dict[str, Any]] = []
    for patch in patch_paths:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        missing = patch_targets_missing(text, framework_root)
        if missing:
            records.append({"patch": str(patch), "missing_targets": missing})
    return records


def _detect_p_level(
    framework_root: Path,
    patch_path: Path,
    *,
    three_way: bool,
) -> int | None:
    """Return the first ``-p`` level whose ``--check`` applies cleanly.

    Args:
        framework_root: The git checkout to test against.
        patch_path: The patch file to probe.
        three_way: Whether to probe with ``-3``.

    Returns:
        The first ``-p<N>`` level that applies cleanly, or ``None`` when none
        do.
    """
    for lvl in _P_LEVELS:
        ok, _ = _run_git_apply(
            framework_root,
            patch_path,
            p_level=lvl,
            three_way=three_way,
            check_only=True,
        )
        if ok:
            return lvl
    return None


def _git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    three_way: bool = False,
    check_only: bool = False,
) -> tuple[bool, str]:
    """Run ``git apply [-3] -p<auto> [--check] <patch>`` inside
    ``framework_root``, auto-detecting the strip level. Returns
    ``(ok, stderr)``.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        three_way: Whether to pass ``-3`` for a three-way merge.
        check_only: Whether to only check (dry run) rather than mutate.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the apply (or check)
        succeeds.
    """
    lvl = _detect_p_level(framework_root, patch_path, three_way=three_way)
    if lvl is None:
        # Surface a representative error at the git-native default level.
        return _run_git_apply(
            framework_root,
            patch_path,
            p_level=1,
            three_way=three_way,
            check_only=check_only,
        )
    if check_only:
        return True, ""
    return _run_git_apply(
        framework_root,
        patch_path,
        p_level=lvl,
        three_way=three_way,
        check_only=False,
    )


def _git_apply_reverse(
    framework_root: Path,
    patch_path: Path,
) -> tuple[bool, str]:
    """Reverse-apply ``patch_path`` (``git apply -R -p<auto>``) as the REVERT
    path; caller falls back to ``git checkout`` on failure. Auto-detects the
    same strip level the forward apply used via ``-R --check``.

    Args:
        framework_root: The git checkout to reverse-apply into.
        patch_path: The patch file to reverse-apply.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the reverse apply
        succeeds.
    """
    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is None:
            return False, "git apply -R spawn failed"
        if cp.returncode != 0:
            continue
        cp2 = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", str(patch_path)],
            timeout=120.0,
        )
        if cp2 is None:
            return False, "git apply -R spawn failed"
        if cp2.returncode == 0:
            return True, ""
        return False, cp2.stderr.strip()
    return False, f"git apply -R: no matching -p level for {patch_path}"


def _find_hyperloom_auto_stash(framework_root: Path) -> str:
    """Return the newest Hyperloom auto-stash ref, or ``""`` if absent."""
    cp = _run_git_cp(
        ["-C", str(framework_root), "stash", "list", "--format=%gd:%gs"],
        timeout=30.0,
    )
    if cp is None:
        return ""
    if cp.returncode != 0:
        return ""
    for line in cp.stdout.splitlines():
        ref, _sep, msg = line.partition(":")
        if ref and _HYPERLOOM_AUTO_STASH_MSG in msg:
            return ref
    return ""


def _git_stash_if_dirty(framework_root: Path) -> tuple[str, str]:
    """Stash uncommitted user changes so destructive resets don't lose them.

    Only stashes when the working tree is dirty (``git status --porcelain``
    is non-empty). The stash message is tagged for easy retrieval via
    ``git stash list | grep hyperloom-auto-stash``.

    Returns:
        ``(state, note)`` where ``state`` is one of:

        - ``"clean"`` — working tree was already clean, safe to proceed.
        - ``"stashed"`` — dirty tree was successfully stashed; ``note`` is the
          stash ref to restore when the candidate finishes.
        - ``"failed"`` — tree is dirty but stash command failed; callers
          MUST NOT proceed with destructive operations.
    """
    cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
    if cp is None:
        return "failed", "git status check failed"
    if cp.returncode != 0:
        # Non-git directory (rc=128 "not a git repository") or other git
        # status errors: treat as clean — no git-managed changes to protect.
        log.debug(
            "integrate_patch: git status rc=%d in %s (not a git repo?), "
            "treating as clean",
            cp.returncode,
            framework_root,
        )
        return "clean", ""
    if not cp.stdout.strip():
        return "clean", ""
    cp2 = _run_git_cp(
        ["-C", str(framework_root), "stash", "push", "-u", "-m", _HYPERLOOM_AUTO_STASH_MSG],
        timeout=60.0,
    )
    if cp2 is None:
        return "failed", "git stash push failed"
    if cp2.returncode == 0:
        stash_ref = _find_hyperloom_auto_stash(framework_root) or "stash@{0}"
        log.info(
            "integrate_patch: stashed user changes in %s as %s",
            framework_root,
            stash_ref,
        )
        return "stashed", stash_ref
    return "failed", f"git stash push rc={cp2.returncode}: {cp2.stderr.strip()}"


def _git_restore_stash_if_needed(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
) -> str:
    """Restore the user-change stash created before candidate mutation."""
    if stash_state != "stashed":
        return ""
    ref = stash_ref or _find_hyperloom_auto_stash(framework_root)
    if not ref:
        return "auto-stash ref not found; user changes remain in git stash"
    cp = _run_git_cp(["-C", str(framework_root), "stash", "pop", "--index", ref], timeout=120.0)
    if cp is None:
        return f"git stash pop failed; user changes remain in {ref}"
    if cp.returncode == 0:
        log.info("integrate_patch: restored user changes from %s", ref)
        return ""
    return (
        f"git stash pop {ref} rc={cp.returncode}: {(cp.stderr or '').strip()}; "
        "user changes remain in git stash"
    )


def _with_stash_restore(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Restore a pre-candidate stash before returning an executor result."""
    note = _git_restore_stash_if_needed(framework_root, stash_state, stash_ref)
    if not note:
        return result
    log.warning("integrate_patch: user-change stash restore failed: %s", note)
    out = dict(result)
    out["stash_restore_error"] = note
    return out


def _git_checkout_clean(framework_root: Path) -> tuple[bool, str]:
    """``git checkout -- .`` + ``git clean -fd`` to discard candidate changes.

    Last-resort REVERT path when individual reverse-apply fails. User changes
    must already have been stashed before candidate apply; this helper must not
    create a new stash because the remaining dirty state is candidate-owned.

    Args:
        framework_root (Path): Directory to run ``git checkout`` in.

    Returns:
        tuple[bool, str]: ``(ok, stderr)`` where ``ok`` is ``True`` on
        return code 0.
    """
    cp = _run_git_cp(["-C", str(framework_root), "checkout", "--", "."], timeout=60.0)
    if cp is None:
        return False, "git checkout spawn failed"
    if cp.returncode != 0:
        return False, cp.stderr.strip()
    cp2 = _run_git_cp(["-C", str(framework_root), "clean", "-fd"], timeout=60.0)
    if cp2 is None:
        return False, "git clean spawn failed"
    return cp2.returncode == 0, cp2.stderr.strip()


_PATCH_DEV_NULL = "/dev/null"


def _strip_path_prefix(path: str, level: int) -> str:
    """Drop ``level`` leading path components (mimics ``git apply -p<level>``).

    Args:
        path: The diff-header path to strip.
        level: The number of leading components to drop (``<= 0`` is a no-op).

    Returns:
        The path with ``level`` leading components removed (or the basename
        when there are not enough components).
    """
    if level <= 0:
        return path
    parts = path.split("/")
    return "/".join(parts[level:]) if len(parts) > level else parts[-1]


def _commit_strip_level(
    framework_root: Path,
    pairs: list[tuple[str, str]],
) -> int:
    """Pick the ``-p`` strip level resolving the most targets to existing files.

    The patch has already been applied, so modify/create targets exist in the
    tree; the level that maximises those hits is the one the forward apply used.

    Args:
        framework_root: The git checkout the patch was applied into.
        pairs: ``(old_path, new_path)`` header pairs from the patch.

    Returns:
        The ``-p`` strip level resolving the most targets to existing files.
    """
    best_lvl, best_hits = 1, -1
    for lvl in _P_LEVELS:
        hits = 0
        for old, new in pairs:
            for raw in (new, old):
                if not raw or raw == _PATCH_DEV_NULL:
                    continue
                try:
                    if (framework_root / _strip_path_prefix(raw, lvl)).exists():
                        hits += 1
                except OSError:
                    continue
        if hits > best_hits:
            best_hits, best_lvl = hits, lvl
    return best_lvl


def _patch_touched_paths(
    framework_root: Path,
    patches: list[Path],
) -> list[str]:
    """Repo-relative paths the applied ``patches`` created / modified / deleted.

    Used to scope the commit-on-KEEP to *only* the files this patch touched, so
    an unrelated dirty working tree under ``framework_root`` (generated files,
    manual edits, stray artifacts) is never swept into the ``hyperloom KEEP``
    commit.

    Per header pair (``old`` ``---``, ``new`` ``+++``):
      * created / modified → the ``new`` target exists post-apply → emit it.
      * deleted (Issue 6) → ``new`` is ``/dev/null`` (or its target is gone)
        and ``old`` existed pre-apply → emit the ``old`` path so the subsequent
        ``git add -A -- <path>`` stages the *removal* of a tracked file.
        Without this a pure-deletion KEEP committed nothing, and a later cycle's
        ``git checkout -- .`` REVERT resurrected the deleted file.
    A header that resolves to neither (matches nothing pre or post) is dropped
    so ``git add`` cannot error on a bogus pathspec.

    Args:
        framework_root: The git checkout the patches were applied into.
        patches: The applied patch files to inspect.

    Returns:
        The repo-relative paths the patches created/modified (existing
        post-apply) plus the old paths of pure deletions, so the subsequent
        ``git add`` stages removals too.
    """
    out: list[str] = []
    for patch in patches:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pairs = patch_file_targets(text)
        if not pairs:
            continue
        lvl = _commit_strip_level(framework_root, pairs)
        for old, new in pairs:
            rel_new = (
                _strip_path_prefix(new, lvl)
                if new and new != _PATCH_DEV_NULL else None
            )
            rel_old = (
                _strip_path_prefix(old, lvl)
                if old and old != _PATCH_DEV_NULL else None
            )
            try:
                new_exists = bool(rel_new) and (framework_root / rel_new).exists()
            except OSError:
                new_exists = False
            if rel_new and new_exists:
                # Created or modified — the post-apply file is on disk.
                if rel_new not in out:
                    out.append(rel_new)
            elif rel_old:
                # Deletion — new target is gone; stage the removal of old.
                if rel_old not in out:
                    out.append(rel_old)
    return out


def _git_commit_kept(
    framework_root: Path,
    message: str,
    paths: list[str],
) -> tuple[bool, str]:
    """Commit only the patch-touched ``paths`` to git (R1 cross-cycle durability).

    In the cyclic phase machine, KEEP patches accumulate across macro-cycles as
    *uncommitted* working-tree edits. A later cycle's REVERT may fall back to
    ``git checkout -- .`` (discards ALL uncommitted changes), which would wipe
    every prior cycle's win. Committing each KEEP makes those wins survive the
    checkout fallback (it only clears uncommitted state). The commit is scoped
    to the exact paths the patch touched (never ``git add -A``) so an unrelated
    dirty framework tree is not folded into the win commit. Best-effort: a
    commit failure (e.g. nothing staged) is non-fatal — the KEEP still stands in
    the working tree exactly as before.

    Args:
        framework_root: The git checkout to commit in.
        message: The commit message for the KEEP.
        paths: The repo-relative patch-touched paths to stage and commit.

    Returns:
        A ``(ok, note)`` tuple; ``ok`` is ``True`` on a successful commit or a
        benign no-op (nothing to commit), and ``note`` carries any detail.
    """
    if not paths:
        return True, "no patch-touched paths to commit"
    cp_add = _run_git_cp(
        ["-C", str(framework_root), "add", "-A", "--", *paths],
        timeout=60.0,
    )
    if cp_add is None:
        return False, "git commit spawn failed"
    if cp_add.returncode != 0:
        return False, f"git add failed: {cp_add.stderr.strip()}"
    cp = _run_git_cp(
        [
            "-C",
            str(framework_root),
            "-c",
            "user.email=hyperloom@local",
            "-c",
            "user.name=Hyperloom",
            "commit",
            "-q",
            "-m",
            message,
        ],
        timeout=60.0,
    )
    if cp is None:
        return False, "git commit spawn failed"
    if cp.returncode == 0:
        return True, ""
    # "nothing to commit" is a benign no-op, not an error.
    out = (cp.stdout + cp.stderr).lower()
    if "nothing to commit" in out:
        return True, "nothing to commit"
    return False, cp.stderr.strip()


def _is_within(child: Path, root: Path) -> bool:
    """True iff ``child`` is ``root`` or nested under it (both pre-resolved)."""
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_patch_paths(
    *,
    specialist_workspace: Path,
    explicit_patches: list[str] | None,
    done_payload: dict[str, Any] | None,
) -> list[Path]:
    """Resolve the list of patch files to apply.

    Order: ``params.patches`` → ``specialist_done.patches_written`` →
    filesystem scan of ``specialist_workspace/{worktree/,}patches/``.
    Entries normalised to absolute Paths; missing ones logged + dropped.

    Security (Issue 5a): a resolved patch path must live inside the specialist
    workspace (or its worktree). ``params.patches`` is LLM-/specialist-
    controllable, so an absolute path pointing outside the sandbox (another
    session's patch, ``/etc/...``) is dropped — otherwise it would be read and
    ``git apply``-ed against the framework tree. Both sides are ``resolve()``-d
    first so a legitimately symlinked workspace still matches.

    Args:
        specialist_workspace: The specialist task workspace to resolve
            relative paths / scan for patches.
        explicit_patches: Explicit patch paths from params, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        The resolved, existing patch files as absolute Paths.
    """
    candidates: list[str] = []
    if explicit_patches:
        candidates.extend(str(p) for p in explicit_patches)
    elif done_payload and isinstance(done_payload.get("patches_written"), list):
        candidates.extend(str(p) for p in done_payload["patches_written"] if p)
    else:
        for base in (
            specialist_workspace / "worktree" / "patches",
            specialist_workspace / "patches",
        ):
            if base.is_dir():
                for p in sorted(base.glob("*.patch")):
                    candidates.append(str(p))
                for p in sorted(base.glob("*.diff")):
                    candidates.append(str(p))

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]

    out: list[Path] = []
    for c in candidates:
        p = Path(c)
        # Resolve relative paths against the specialist workspace + worktree.
        if not p.is_absolute():
            for base in (
                specialist_workspace / "worktree",
                specialist_workspace,
            ):
                cand = base / c
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            log.warning(
                "integrate_patch: patch %r not found (specialist_workspace=%s)",
                c,
                specialist_workspace,
            )
            continue
        resolved = p.resolve()
        if not any(_is_within(resolved, root) for root in allowed_roots):
            log.warning(
                "integrate_patch: patch %r resolves outside the specialist "
                "workspace (%s); dropping for safety",
                c, specialist_workspace,
            )
            continue
        out.append(resolved)
    return out


@dataclass
class _ArtifactSpec:
    """One resolved non-diff tuned artifact to install at integration.

    Attributes:
        source: Absolute path to the artifact file inside the specialist
            workspace / worktree (sandbox-validated).
        target: Absolute install path inside an allowlisted framework root
            (sandbox-validated; no escape).
        rel_target: The framework-relative target as authored by the
            specialist (for reporting).
        kind: Free-form artifact kind label (e.g. ``config_json``).
        description: Free-form human description.
    """

    source: Path
    target: Path
    rel_target: str
    kind: str = ""
    description: str = ""


def _resolve_artifact_target(rel_target: str) -> Path | None:
    """Resolve a framework-relative artifact target to an absolute path.

    Picks the allowlisted framework root whose tree already contains the
    target's parent directory (so a ``vllm/...`` config lands under the vllm
    root); else the first existing root. The resolved path must stay within
    the chosen root (no ``..`` escape).

    Args:
        rel_target: The framework-relative install path authored by the
            specialist.

    Returns:
        The absolute target path, or ``None`` when nothing resolves safely.
    """
    rel = (rel_target or "").strip()
    if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    roots = [Path(r).resolve() for r in resolve_source_file_allowlist()]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return None
    # Prefer a root whose tree already holds the target's parent dir.
    for root in roots:
        cand = (root / rel).resolve()
        if not _is_within(cand, root):
            continue
        if cand.parent.is_dir():
            return cand
    # Fall back to the first root that keeps the path contained.
    for root in roots:
        cand = (root / rel).resolve()
        if _is_within(cand, root):
            return cand
    return None


def _resolve_artifact_specs(
    *,
    specialist_workspace: Path,
    explicit_artifacts: list[dict[str, Any]] | None,
    done_payload: dict[str, Any] | None,
) -> tuple[list[_ArtifactSpec], list[dict[str, str]]]:
    """Resolve non-diff tuned artifacts to install (B6 / §3.5 contract).

    Order: ``params.artifacts`` → ``specialist_done.artifacts_written``. Each
    entry is ``{source, target, kind, description}``: ``source`` is resolved
    inside the specialist workspace/worktree (sandbox) and ``target`` is
    resolved inside an allowlisted framework root. Malformed / out-of-sandbox
    entries are dropped and reported.

    Args:
        specialist_workspace: The specialist task workspace.
        explicit_artifacts: ``params.artifacts`` override list, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        A ``(specs, errors)`` tuple: resolved specs, plus per-entry error
        records (``{artifact, error}``) for entries that could not be resolved.
    """
    raw: list[Any] = []
    if explicit_artifacts:
        raw = list(explicit_artifacts)
    elif done_payload and isinstance(done_payload.get("artifacts_written"), list):
        raw = list(done_payload["artifacts_written"])

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]
    specs: list[_ArtifactSpec] = []
    errors: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            errors.append({"artifact": str(entry), "error": "not_a_mapping"})
            continue
        src_rel = str(entry.get("source") or "").strip()
        tgt_rel = str(entry.get("target") or "").strip()
        if not src_rel or not tgt_rel:
            errors.append({"artifact": json.dumps(entry), "error": "missing_source_or_target"})
            continue
        # Resolve source inside the workspace / worktree sandbox.
        src = Path(src_rel)
        if not src.is_absolute():
            for base in (specialist_workspace / "worktree", specialist_workspace):
                cand = base / src_rel
                if cand.exists():
                    src = cand
                    break
        if not src.exists() or not src.resolve().is_file():
            errors.append({"artifact": src_rel, "error": "source_not_found"})
            continue
        src_resolved = src.resolve()
        if not any(_is_within(src_resolved, root) for root in allowed_roots):
            errors.append({"artifact": src_rel, "error": "source_outside_workspace"})
            continue
        target = _resolve_artifact_target(tgt_rel)
        if target is None:
            errors.append({"artifact": tgt_rel, "error": "target_unresolved_or_escapes_root"})
            continue
        specs.append(
            _ArtifactSpec(
                source=src_resolved,
                target=target,
                rel_target=tgt_rel,
                kind=str(entry.get("kind") or "").strip(),
                description=str(entry.get("description") or "").strip(),
            )
        )
    return specs, errors


def _read_done_payload(workspace: Path) -> dict[str, Any] | None:
    """Read and parse ``specialist_done.json`` from a workspace.

    Args:
        workspace (Path): The specialist task workspace directory.

    Returns:
        dict[str, Any] | None: The parsed payload, or ``None`` when the
        file is absent or cannot be parsed.
    """
    done = workspace / "specialist_done.json"
    if not done.exists():
        return None
    try:
        return json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "integrate_patch: failed to parse %s: %r",
            done,
            exc,
        )
        return None


class IntegratePatchExecutor:
    """ActionRunner for the ``integrate_patch`` action (PR-A4)."""

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the integrate-patch executor.

        Args:
            session_dir (Path | str | None): Session output directory;
                auto-resolved when ``None``.
            default_config_path (Path | str | None): Fallback benchmark
                config path, if any.
            variant_timeout_sec (int): Per-variant benchmark hard timeout.
                Defaults to :data:`DEFAULT_VARIANT_TIMEOUT_SEC`.
            keep_threshold_pct (float): Minimum gain to KEEP a patch.
                Defaults to :data:`DEFAULT_KEEP_THRESHOLD_PCT`.
        """
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Apply a specialist's patches/config changes and benchmark them.

        Resolves the completed specialist's patches and config changes,
        applies them against the framework source root, benchmarks the
        result with KEEP/REVERT gating, and reverts on regression.

        Args:
            ctx: The action runner context carrying the task and params
                (notably ``specialist_task_id``).

        Returns:
            dict[str, Any]: The integration result payload (status plus
            applied/reverted patches and config changes), or a failure
            dict on error.
        """
        params = dict(ctx.task.params or {})

        # Multi-node guard. This executor git-applies the specialist patch
        # ONLY to the sandbox framework_source_roots; in multi-node mode the
        # live sglang/vllm runs on RayJob pods, not the sandbox, so a
        # sandbox-only apply would silently NOT affect pod-side serving — the
        # bench would measure the unpatched pod and the KEEP/REVERT verdict
        # would be meaningless. Until a git-diff pod fan-out exists, return a
        # NEUTRAL "skipped" result (no patch touched, no error). This is NOT a
        # failure: the Coordinator only records integrate_patch results whose
        # ``status == "kept"`` (coordinator.py: "any other status → NOT
        # recorded"), so a skip rolls no failure tally and the session keeps
        # running every other action (baseline/profile/explore/sweep/
        # roofline). ``is_multi_node()`` is False single-node, so the normal
        # path below is reached bit-for-bit unchanged.
        from ._multi_node_env import is_multi_node

        if is_multi_node():
            return {
                "status": "skipped",
                "skipped_reason": "multi_node_unsupported",
                "specialist_task_id": str(params.get("specialist_task_id") or "").strip(),
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "reason": (
                    "specialist integrate_patch is not supported in "
                    "multi-node mode (no git-diff pod fan-out); skipped "
                    "without applying any patch. Other actions "
                    "(baseline/profile/explore/sweep/roofline) continue "
                    "normally. Use the kernel-agent integrate path (which "
                    "fans out via `multi_node apply-patch`) or run single-node."
                ),
            }

        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "integrate_patch requires params.specialist_task_id "
                    "(the completed specialist whose worktree carries "
                    "the patches to integrate)"
                ),
            }
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or extra.get("state")
        # Specialist workspace conventionally at runs/specialist/<id>/.
        specialist_workspace = runs_dir(self.session_dir, "specialist", specialist_task_id)
        if not specialist_workspace.is_dir():
            return {
                "status": "failed",
                "error_class": "missing_specialist",
                "error": (f"specialist workspace not found at {specialist_workspace}"),
                "specialist_task_id": specialist_task_id,
            }

        # Read done payload for patches_written + config_changes_default.
        done_payload = _read_done_payload(specialist_workspace)

        # Patch resolution.
        explicit_patches = params.get("patches") or None
        patch_paths = _resolve_patch_paths(
            specialist_workspace=specialist_workspace,
            explicit_patches=(list(explicit_patches) if isinstance(explicit_patches, list) else None),
            done_payload=done_payload,
        )
        config_changes = dict(params.get("config_changes") or {})
        # Seed config_changes from specialist_done when params didn't.
        if not config_changes and done_payload:
            cc = done_payload.get("config_changes")
            if isinstance(cc, dict):
                config_changes = {str(k): str(v) for k, v in cc.items()}

        # §3.5: non-diff tuned artifacts (e.g. an autotuned config JSON) are a
        # first-class integrable output alongside unified diffs + config_changes.
        explicit_artifacts = params.get("artifacts")
        artifact_specs, artifact_resolve_errors = _resolve_artifact_specs(
            specialist_workspace=specialist_workspace,
            explicit_artifacts=(list(explicit_artifacts) if isinstance(explicit_artifacts, list) else None),
            done_payload=done_payload,
        )

        if not patch_paths and not config_changes and not artifact_specs:
            return {
                "status": "no_patches",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "artifacts_applied": [],
                "artifact_errors": artifact_resolve_errors,
                "reason": (
                    "neither patches, config_changes, nor installable artifacts "
                    "were supplied / discoverable for this specialist task"
                ),
            }

        framework_root = _resolve_framework_root(
            params.get("framework_source_root") or None,
            patch_paths=patch_paths,
        )
        # Pure config_changes path works without a framework root.
        if patch_paths and framework_root is None:
            return {
                "status": "apply_failed",
                "error_class": "no_framework_root",
                "error": (
                    "no framework_source_root resolved; cannot apply "
                    "patches. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                ),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
            }

        # Preflight: reject patches whose modify/delete targets do not exist in
        # the framework tree before spending a benchmark on a doomed apply.
        if patch_paths and framework_root is not None:
            missing_records = _preflight_missing_targets(framework_root, patch_paths)
            if missing_records:
                await self._maybe_write_framework_pr_kb_record(
                    done_payload=done_payload,
                    outcome="rejected_apply_fail",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                return {
                    "status": "apply_failed",
                    "error_class": "patch_target_missing",
                    "error": missing_records,
                    "advisory": (
                        "patch target file(s) absent from framework_source_root "
                        f"{framework_root}; author patches only against files that "
                        "exist in the installed framework tree (inspect it with "
                        "Glob/Grep before writing the diff)."
                    ),
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "config_changes_applied": {},
                }

        # Per-action workspace under runs/integrate_patch/<task_id>/.
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "integrate_patch", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Long-run #4: mark the non-transactional integrate window before any
        # framework tree mutation. The Coordinator clears this after promoting
        # the final KEEP/REVERT/APPLY_FAILED result into SharedState.
        if shared_state is not None:
            try:
                shared_state.pending_integrate = {
                    "specialist_task_id": specialist_task_id,
                    "task_id": str(getattr(ctx.task, "task_id", "") or ""),
                    "patches": [str(p) for p in patch_paths],
                    "artifacts": [
                        {"target": str(s.target), "rel_target": s.rel_target}
                        for s in artifact_specs
                    ],
                    "config_changes": dict(config_changes),
                    "framework_source_root": str(framework_root or ""),
                    "workspace": str(output_root),
                    "ts": _now_iso(),
                }
                shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — sentinel is best-effort
                log.exception("integrate_patch: failed to persist pending_integrate sentinel")

        # Preserve user's uncommitted changes BEFORE applying patches.
        # Stashing here ensures only user state enters the stash (not
        # Hyperloom candidate artifacts), so `git stash pop` after the
        # run cleanly restores only the user's original modifications.
        stash_state, stash_note = _git_stash_if_dirty(framework_root)
        if stash_state == "failed":
            log.error(
                "integrate_patch: cannot stash user changes in %s: %s; "
                "aborting to avoid data loss",
                framework_root,
                stash_note,
            )
            return {
                "status": "apply_failed",
                "error_class": "stash_failed",
                "error": f"refusing to proceed: user changes could not be stashed ({stash_note})",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
            }

        # Stage 1: apply patches (best-effort with -3 fallback).
        applied: list[Path] = []
        applied_artifacts: list[dict[str, Any]] = []
        apply_errors: list[dict[str, str]] = []
        for patch in patch_paths:
            ok, err = _git_apply(framework_root, patch, three_way=False)
            if not ok:
                ok2, err2 = _git_apply(framework_root, patch, three_way=True)
                if not ok2:
                    apply_errors.append(
                        {
                            "patch": str(patch),
                            "stderr": err + " | -3 retry: " + err2,
                        }
                    )
                    break
                err = err2
            applied.append(patch)
        if apply_errors:
            # Mid-apply failure — reverse the partial set back to clean.
            reverted = self._revert_patches(framework_root, applied)
            await self._maybe_write_framework_pr_kb_record(
                done_payload=done_payload,
                outcome="rejected_apply_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            return _with_stash_restore(framework_root, stash_state, stash_note, {
                "status": "apply_failed",
                "error_class": "git_apply_failed",
                "error": apply_errors,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "workspace": str(output_root),
            })

        # Stage 1b: install non-diff tuned artifacts (after patches, before
        # config_changes). On any artifact error, roll back artifacts + patches
        # and surface a clean apply_failed (not an opaque git_apply_failed).
        if artifact_specs:
            applied_artifacts, artifact_apply_errors = self._apply_artifacts(
                artifact_specs,
                backup_root=output_root / "artifact_backups",
            )
            if artifact_apply_errors:
                self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                await self._maybe_write_framework_pr_kb_record(
                    done_payload=done_payload,
                    outcome="rejected_apply_fail",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                return {
                    "status": "apply_failed",
                    "error_class": "artifact_install_failed",
                    "error": artifact_resolve_errors + artifact_apply_errors,
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_applied": [],
                    "config_changes_applied": {},
                    "workspace": str(output_root),
                }

        # Stage 2: layer config_changes onto the launch env (via the
        # variant's ``extra_envs`` knob).
        config_changes_applied = dict(config_changes)

        # Defensive double-check on the Critic verdict. PolicyGate's
        # ``integrate_patch_requires_critic_verdict`` already gates the
        # delegate; this is belt-and-braces for paths that bypass PolicyGate
        # (legacy resume / test injection). No-ops when SharedState is absent.
        if shared_state is not None and not params.get("bypass_critic"):
            try:
                recorded = shared_state.get_specialist_patch_verdict(
                    specialist_task_id,
                )
            except AttributeError:
                recorded = ""
            if recorded and recorded.lower() == "reject":
                artifacts_reverted = self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                return _with_stash_restore(framework_root, stash_state, stash_note, {
                    "status": "rejected_by_critic",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "reason": (
                        f"Critic verdict 'reject' recorded for specialist "
                        f"task {specialist_task_id!r}; integrate_patch "
                        f"refuses to bench. Pass bypass_critic=True to "
                        f"force."
                    ),
                    "workspace": str(output_root),
                })

        # Stage 3: optionally skip the bench (test / smoke).
        if params.get("apply_only"):
            return _with_stash_restore(framework_root, stash_state, stash_note, {
                "status": "applied_no_bench",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "artifacts_applied": applied_artifacts,
                "config_changes_applied": config_changes_applied,
                "reason": "apply_only=True; benchmark skipped",
                "workspace": str(output_root),
            })

        # Stage 4: bench the patched config via run_grid (1 variant).
        try:
            bench_result, gate_evidence = await self._bench_patch(
                params=params,
                output_root=output_root,
                config_changes_applied=config_changes_applied,
                specialist_task_id=specialist_task_id,
            )
        except FrameworkScriptMismatchError as exc:
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(framework_root, stash_state, stash_note, {
                "status": "reverted",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "artifacts_reverted": artifacts_reverted,
                "config_changes_applied": {},
                "reason": str(exc),
                "workspace": str(output_root),
            })
        except Exception as exc:  # noqa: BLE001
            self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(framework_root, stash_state, stash_note, {
                "status": "reverted",
                "error_class": "bench_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "reason": f"bench raised: {exc!r}",
                "workspace": str(output_root),
            })

        # Stage 5: KEEP / REVERT decision.
        # B4: the Coordinator seeds ``base_tput`` per-dispatch, but a direct
        # invocation (resume path / test / external caller) may bypass that and
        # leave it 0.0, which would make ``delta_pct`` None and auto-REVERT a
        # genuinely valid patch. Fall back to the live ``SharedState`` anchor
        # (current_best.tput else baseline_tput) so the gate still measures.
        base_tput = float(params.get("base_tput") or 0.0)
        if base_tput <= 0 and shared_state is not None:
            cb = getattr(shared_state, "current_best", None)
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                base_tput = float(cb_tput)
            else:
                ss_base = getattr(shared_state, "baseline_tput", 0.0)
                if isinstance(ss_base, (int, float)) and ss_base > 0:
                    base_tput = float(ss_base)
        keep_threshold_pct = float(
            params.get("keep_threshold_pct", self.keep_threshold_pct),
        )
        new_tput = bench_result.get("output_throughput")
        delta_pct = None
        if isinstance(new_tput, (int, float)) and new_tput > 0 and base_tput > 0:
            delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0

        accuracy_pass: bool | None = gate_evidence.get("accuracy_pass")
        # KEEP requires delta_pct ≥ keep_threshold AND accuracy_pass != False.
        gate_pass = (
            delta_pct is not None and delta_pct >= keep_threshold_pct and (accuracy_pass is None or accuracy_pass)
        )

        if not gate_pass:
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            reasons: list[str] = []
            if delta_pct is None:
                reasons.append("no measurable throughput")
            elif delta_pct < keep_threshold_pct:
                reasons.append(f"throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%")
            if accuracy_pass is False:
                reasons.append("accuracy regression detected")
            await self._maybe_write_framework_pr_kb_record(
                done_payload=done_payload,
                outcome="reverted_smoke_fail",
                tps_delta_pct=float(delta_pct or 0.0),
                extra=extra,
            )
            return _with_stash_restore(framework_root, stash_state, stash_note, {
                "status": "reverted",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "artifacts_reverted": artifacts_reverted,
                "config_changes_applied": {},
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "reason": "; ".join(reasons) or "gate failed",
                "bench_result": bench_result,
                "workspace": str(output_root),
            })

        # Confirmation rebench: a patch only KEEPs if a second full-stack run
        # still clears the stability floor and the accuracy gate.
        if params.get("enable_stack_rebench", True) and base_tput > 0:
            confirm = await self._confirm_stack_rebench(
                params=params,
                output_root=output_root,
                config_changes_applied=config_changes_applied,
                specialist_task_id=specialist_task_id,
                base_tput=base_tput,
            )
            if not confirm["stable"] or confirm["accuracy_pass"] is False:
                artifacts_reverted = self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                reasons = []
                if not confirm["stable"]:
                    reasons.append(
                        f"stack rebench {confirm['tput']} below stability floor {confirm['stable_floor']:.2f}"
                    )
                if confirm["accuracy_pass"] is False:
                    reasons.append("accuracy regression on rebench")
                await self._maybe_write_framework_pr_kb_record(
                    done_payload=done_payload,
                    outcome="reverted_smoke_fail",
                    tps_delta_pct=float(delta_pct or 0.0),
                    extra=extra,
                )
                return _with_stash_restore(framework_root, stash_state, stash_note, {
                    "status": "reverted",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "output_throughput": new_tput,
                    "delta_pct": delta_pct,
                    "accuracy_pass": confirm["accuracy_pass"],
                    "base_tput": base_tput,
                    "keep_threshold_pct": keep_threshold_pct,
                    "reason": "; ".join(reasons) or "stack rebench failed",
                    "bench_result": bench_result,
                    "stack_rebench": confirm,
                    "workspace": str(output_root),
                })
            # Confirmed: the rebench tput is the headline.
            if isinstance(confirm["tput"], (int, float)) and confirm["tput"] > 0:
                new_tput = confirm["tput"]
                delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0
            if confirm["accuracy_pass"] is not None:
                accuracy_pass = confirm["accuracy_pass"]

        await self._maybe_write_framework_pr_kb_record(
            done_payload=done_payload,
            outcome="integrated",
            tps_delta_pct=float(delta_pct or 0.0),
            extra=extra,
        )
        # R1: in cyclic mode, commit the KEEP so a later macro-cycle's REVERT
        # checkout fallback can't wipe this win (best-effort, non-fatal).
        try:
            from ..phase_state import is_cyclic_phases_enabled

            if is_cyclic_phases_enabled():
                touched = _patch_touched_paths(framework_root, applied)
                ok, note = _git_commit_kept(
                    framework_root,
                    f"hyperloom KEEP {specialist_task_id} ({delta_pct:+.2f}%)",
                    touched,
                )
                if not ok:
                    log.warning(
                        "integrate_patch: commit-on-KEEP failed (%s); win remains uncommitted in the working tree",
                        note,
                    )
        except Exception:  # noqa: BLE001 — commit durability is best-effort
            log.exception("integrate_patch: commit-on-KEEP raised")
        return _with_stash_restore(framework_root, stash_state, stash_note, {
            "status": "kept",
            "specialist_task_id": specialist_task_id,
            "patches_applied": [str(p) for p in applied],
            "patches_reverted": [],
            "artifacts_applied": applied_artifacts,
            "config_changes_applied": config_changes_applied,
            "output_throughput": new_tput,
            "delta_pct": delta_pct,
            "accuracy_pass": accuracy_pass,
            "base_tput": base_tput,
            "keep_threshold_pct": keep_threshold_pct,
            "reason": (f"throughput delta {delta_pct:+.2f}% >= {keep_threshold_pct:.2f}%"),
            "bench_result": bench_result,
            "workspace": str(output_root),
        })

    # Helpers
    @staticmethod
    def _find_framework_pr_proposal(
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the first proposal whose provenance starts with
        ``specialist:serving:framework_pr`` (F2-5); ``None`` otherwise so
        the KB writeback hook no-ops for legacy / kernel outputs.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.

        Returns:
            The matching framework_pr proposal dict, or ``None`` when absent.
        """
        if not isinstance(done_payload, dict):
            return None
        proposal_set = done_payload.get("proposal_set") or []
        if not isinstance(proposal_set, list):
            return None
        for proposal in proposal_set:
            if not isinstance(proposal, dict):
                continue
            provenance = str(proposal.get("provenance") or "")
            if provenance.startswith("specialist:serving:framework_pr"):
                return proposal
        return None

    async def _maybe_write_framework_pr_kb_record(
        self,
        *,
        done_payload: dict[str, Any] | None,
        outcome: str,
        tps_delta_pct: float,
        extra: dict[str, Any],
    ) -> None:
        """F2-5: append a JSONL record to ``lessons.jsonl`` when the patch
        came from the FRAMEWORK_PR phase.

        No-op for other provenance or when both dedup keys (``fa_pr_url`` /
        ``fa_pr_sha``) are missing. Write errors are logged + swallowed.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.
            outcome: The outcome label to record (e.g. integrated / reverted).
            tps_delta_pct: The measured throughput delta percentage.
            extra: The runner ``extra`` mapping (provides shared state /
                session id).
        """
        proposal = self._find_framework_pr_proposal(done_payload)
        if proposal is None:
            return
        pr_url = str(proposal.get("fa_pr_url") or "").strip()
        pr_sha = str(proposal.get("fa_pr_sha") or "").strip()
        if not pr_url and not pr_sha:
            log.warning(
                "integrate_patch: framework_pr proposal lacks both fa_pr_url and fa_pr_sha; KB writeback skipped",
            )
            return
        patches_written = proposal.get("patches_written") or []
        patch_path = ""
        if isinstance(patches_written, list) and patches_written:
            patch_path = str(patches_written[0])
        session_id = ""
        shared_state = extra.get("shared_state") or extra.get("state")
        if shared_state is not None:
            session_id = str(getattr(shared_state, "cortex_session_id", "") or "")
        try:
            from ..kb_writeback import write_framework_pr_record

            written = await write_framework_pr_record(
                pr_url=pr_url,
                pr_sha=pr_sha,
                patch_path=patch_path,
                outcome=outcome,
                tps_delta_pct=float(tps_delta_pct),
                session_id=session_id,
            )
            log.info(
                "integrate_patch: wrote framework_pr KB record to %s (outcome=%s pr_url=%s tps_delta=%+.2f%%)",
                written,
                outcome,
                pr_url,
                float(tps_delta_pct),
            )
        except Exception as exc:  # noqa: BLE001 — KB write is best-effort
            log.warning(
                "integrate_patch: framework_pr KB writeback failed: %r",
                exc,
            )

    def _revert_patches(
        self,
        framework_root: Path | None,
        applied: list[Path],
    ) -> list[Path]:
        """Reverse-apply the applied patches (best-effort); returns those
        actually reverted.

        Args:
            framework_root: The git checkout to revert in, or ``None`` (no-op).
            applied: The patches that were applied this run.

        Returns:
            The patches actually reverted (may be the full ``applied`` list
            when the checkout fallback fires).
        """
        reverted: list[Path] = []
        if framework_root is None or not applied:
            return reverted
        # Reverse order so dependent patches unstick correctly.
        for patch in reversed(applied):
            ok, err = _git_apply_reverse(framework_root, patch)
            if ok:
                reverted.append(patch)
            else:
                log.warning(
                    "integrate_patch: git apply -R failed for %s: %s; falling back to git checkout",
                    patch,
                    err,
                )
                # Reverse-apply failed → checkout clears all uncommitted at once.
                ok2, err2 = _git_checkout_clean(framework_root)
                if ok2:
                    reverted = list(applied)
                    break
                log.error(
                    "integrate_patch: git checkout fallback failed: %s",
                    err2,
                )
                break
        return reverted

    def _apply_artifacts(
        self,
        specs: list[_ArtifactSpec],
        *,
        backup_root: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Install non-diff tuned artifacts, backing up any clobbered targets.

        Each artifact's existing target is backed up under ``backup_root`` (or
        recorded as newly-created) so :meth:`_revert_artifacts` can restore the
        framework tree exactly. Applied artifacts are returned in order so a
        revert can undo them in reverse.

        Args:
            specs: The resolved artifact specs to install.
            backup_root: Directory under which clobbered targets are saved.

        Returns:
            A ``(applied, errors)`` tuple: per-artifact apply records (with the
            backup bookkeeping) and per-artifact error records.
        """
        applied: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        backup_root.mkdir(parents=True, exist_ok=True)
        for idx, spec in enumerate(specs):
            try:
                spec.target.parent.mkdir(parents=True, exist_ok=True)
                existed = spec.target.exists()
                backup_path: str | None = None
                if existed:
                    backup_path = str(backup_root / f"{idx:03d}_{spec.target.name}.bak")
                    shutil.copy2(spec.target, backup_path)
                shutil.copy2(spec.source, spec.target)
                applied.append(
                    {
                        "target": str(spec.target),
                        "rel_target": spec.rel_target,
                        "kind": spec.kind,
                        "existed": existed,
                        "backup": backup_path,
                    }
                )
            except OSError as exc:
                errors.append({"artifact": spec.rel_target, "error": repr(exc)})
        return applied, errors

    @staticmethod
    def _revert_artifacts(applied: list[dict[str, Any]]) -> list[str]:
        """Undo installed artifacts (restore backups / delete created files).

        Args:
            applied: The apply records returned by :meth:`_apply_artifacts`.

        Returns:
            The framework-relative targets actually reverted.
        """
        reverted: list[str] = []
        for rec in reversed(applied):
            target = Path(str(rec.get("target") or ""))
            if not target.name:
                continue
            try:
                if rec.get("existed") and rec.get("backup"):
                    shutil.copy2(str(rec["backup"]), target)
                elif not rec.get("existed"):
                    if target.exists():
                        target.unlink()
                reverted.append(str(rec.get("rel_target") or target))
            except OSError as exc:  # noqa: BLE001 — best-effort restore
                log.warning("integrate_patch: failed to revert artifact %s: %r", target, exc)
        return reverted

    async def _bench_patch(
        self,
        *,
        params: dict[str, Any],
        output_root: Path,
        config_changes_applied: dict[str, str],
        specialist_task_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server + accuracy gate.

        Returns ``(bench_result_dict, gate_evidence)`` where gate_evidence
        carries ``accuracy_pass`` (True / False / None).

        Args:
            params: The task params (config / model / bench knobs).
            output_root: The per-task workspace root for the bench.
            config_changes_applied: Env overrides layered onto the variant.
            specialist_task_id: The originating specialist task id (names the
                variant).

        Returns:
            A ``(bench_result_dict, gate_evidence)`` tuple where
            ``gate_evidence`` carries ``accuracy_pass`` (True / False / None).
        """
        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not config_path.exists():
            raise RuntimeError(f"integrate_patch bench: config not found at {config_path}")
        resolved_model = str(params.get("model_path") or "").strip() or os.environ.get("MODEL_PATH", "").strip()
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        override_script = sanitize_script_name(params.get("benchmark_script"))
        override_result_dir = sanitize_result_dir(params.get("result_dir"))
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            out_name="integrate_patch.with_envs.yaml",
        )

        # Single-variant grid with config_changes_applied as extra_envs.
        variant = GridVariant(
            name=f"integrate-patch-{specialist_task_id[:8]}",
            extra_server_args=str(params.get("base_extra_args") or "").strip(),
            extra_envs=dict(config_changes_applied),
            note=f"integrate_patch:{specialist_task_id}",
        )

        results: list[VariantResult] = await run_grid(
            base_yaml_path=config_path,
            base_extra_args=str(params.get("base_extra_args") or "").strip(),
            grid=[variant],
            output_root=output_root,
            magpie_python=params.get("magpie_python") or None,
            variant_timeout_sec=int(
                params.get("variant_timeout_sec", self.variant_timeout_sec),
            ),
            keep_going_on_failure=False,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            result_dir=override_result_dir,
        )

        bench: dict[str, Any] = {}
        if results:
            r = results[0]
            bench = {
                "name": r.name,
                "status": r.status,
                "output_throughput": getattr(r, "output_throughput", None),
                "ttft_ms": getattr(r, "ttft_ms", None),
                "itl_ms": getattr(r, "itl_ms", None),
                # ``VariantResult`` exposes the benchmark dir as ``workspace``
                # (there is no ``result_dir`` attribute); using the wrong name
                # left ``_grade_accuracy`` with an empty path so the accuracy
                # gate silently skipped on every patch.
                "workspace": str(getattr(r, "workspace", "") or ""),
                "error": getattr(r, "error", "") or "",
                "nonfatal_warnings": list(getattr(r, "nonfatal_warnings", []) or []),
            }

        accuracy_pass: bool | None = None
        if bench.get("status") == "succeeded":
            accuracy_pass = self._grade_accuracy(bench["workspace"], params.get("accuracy_baseline"))

        return bench, {"accuracy_pass": accuracy_pass}

    @staticmethod
    def _grade_accuracy(result_dir: str, baseline_accuracy: Any) -> bool | None:
        """Grade a bench's accuracy against the baseline.

        With a recorded baseline the measured drop is enforced; without one
        (or no eval result) the check is skipped (``None``) and warned loudly.
        """
        # Accept numeric strings (e.g. ``"0.85"``) in addition to int/float so a
        # baseline carried as text is not silently coerced to 0.0 (which would
        # skip the gate). Non-numeric / missing values fall back to 0.0 (skip).
        try:
            baseline_value = float(baseline_accuracy)
        except (TypeError, ValueError):
            baseline_value = 0.0
        try:
            eval_results = parse_eval_results(result_dir)
            new_accuracy = eval_results.get("accuracy")
            if new_accuracy is not None and baseline_value > 0:
                return accuracy_passed(baseline_value, float(new_accuracy))
            if baseline_value <= 0:
                log.warning(
                    "integrate_patch: no baseline accuracy; accuracy gate skipped "
                    "(throughput-only KEEP). Accuracy regressions will not be caught.",
                )
            else:
                log.warning("integrate_patch: variant produced no accuracy result; gate skipped")
        except Exception:  # noqa: BLE001
            log.exception("integrate_patch: accuracy gate parse failed; treating as None (gate skipped)")
        return None

    async def _confirm_stack_rebench(
        self,
        *,
        params: dict[str, Any],
        output_root: Path,
        config_changes_applied: dict[str, str],
        specialist_task_id: str,
        base_tput: float,
    ) -> dict[str, Any]:
        """Re-bench the patched stack once more and re-grade throughput + accuracy.

        Mirrors the explore ledger's post-KEEP confirmation: a patch only KEEPs
        if a second full-stack run still clears the stability floor and the
        accuracy gate. Returns ``stable`` / ``tput`` / ``accuracy_pass`` / etc.
        """
        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        resolved_model = str(params.get("model_path") or "").strip() or os.environ.get("MODEL_PATH", "").strip()
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        override_script = sanitize_script_name(params.get("benchmark_script"))
        override_result_dir = sanitize_result_dir(params.get("result_dir"))
        base_extra_args = str(params.get("base_extra_args") or "").strip()
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            out_name="integrate_patch.rebench.yaml",
        )
        variant = GridVariant(
            name=f"integrate-patch-rebench-{specialist_task_id[:8]}",
            extra_server_args=base_extra_args,
            extra_envs=dict(config_changes_applied),
            note=f"integrate_patch_rebench:{specialist_task_id}",
        )
        rebench = await measure_stack_rebench(
            config_path=config_path,
            base_extra_args=base_extra_args,
            variant=variant,
            base_tput=base_tput,
            stable_threshold_pct=float(params.get("rebench_stable_threshold_pct", 0.0)),
            output_slot=output_root / "stack_rebench",
            variant_timeout_sec=int(params.get("variant_timeout_sec", self.variant_timeout_sec)),
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            result_dir=override_result_dir,
            magpie_python=params.get("magpie_python") or None,
        )
        accuracy_pass = (
            self._grade_accuracy(rebench.workspace, params.get("accuracy_baseline")) if rebench.workspace else None
        )
        return {
            "stable": rebench.stable,
            "tput": rebench.tput,
            "workspace": rebench.workspace,
            "warnings": rebench.warnings,
            "stable_floor": rebench.stable_floor,
            "accuracy_pass": accuracy_pass,
        }


__all__ = [
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_VARIANT_TIMEOUT_SEC",
    "IntegratePatchExecutor",
    "_detect_p_level",
    "_git_apply",
    "_git_apply_reverse",
    "_git_checkout_clean",
    "_run_git_apply",
    "_resolve_framework_root",
    "_resolve_patch_paths",
    "_read_done_payload",
]
