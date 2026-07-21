# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply specialist patches to live framework roots and KEEP or REVERT by benchmark."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...framework.paths import resolve_source_file_allowlist
from ...specialists.patch_safety import patch_file_targets, patch_targets_missing
from ._accuracy_gate import accuracy_keep_block, accuracy_passed, parse_eval_results
from ._apply_feedback import ApplyFeedback, build_apply_feedback
from ._git import _run_git_cp
from ._nogit_patch import (
    _P_LEVELS,
    _PATCH_DEV_NULL,
    _apply_patch_no_git,
    _is_git_tree,
    _is_within,
    _revert_patches_no_git,
    _strip_path_prefix,
)
from ._grid_runner import (
    GridVariant,
    VariantResult,
    _num_gpus_for_config,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._grid_server_args import merge_server_args, split_config_changes
from ._stack_rebench import DEFAULT_STACK_STABLE_PCT, measure_stack_rebench
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


DEFAULT_KEEP_THRESHOLD_PCT = 1.0  # grid noise floor; KEEP is re-confirmed by a stack rebench
DEFAULT_VARIANT_TIMEOUT_SEC = 7800
# Minimal-correctness floor for the enablement runnable gate: accuracy strictly
# above this counts as "not garbage".
ENABLEMENT_ACCURACY_FLOOR = 0.0
_HYPERLOOM_AUTO_STASH_MSG = "hyperloom-auto-stash: preserving user changes before candidate run"


# Enablement environment-setup replay: allowlist of install-only command shapes.
# A specialist may run arbitrary Bash in its own sandboxed session, but the
# durable *replay* performed here (before applying patches + booting) is limited
# to package/tool installation so a recorded ``setup_commands`` list can never be
# a vector for arbitrary side effects (rm, curl|bash, service restarts, etc.).
# Matched against the command with leading `sudo `/env-assignments stripped.
_SETUP_CMD_ALLOWLIST: tuple[str, ...] = (
    r"pip3?\s+install\b",
    r"(?:python3?|uv)\s+-m\s+pip\s+install\b",
    r"uv\s+pip\s+install\b",
    r"pip3?\s+uninstall\s+-y\b",
    r"apt(?:-get)?\s+(?:install|update)\b",
    r"npm\s+(?:install|i|ci)\b",
    r"npm\s+install\s+-g\b",
    r"pnpm\s+(?:install|add)\b",
    r"yarn\s+(?:add|install)\b",
    r"conda\s+install\b",
    r"mamba\s+install\b",
)
_SETUP_CMD_MAX = 12  # cap on distinct setup commands per integrate
_SETUP_CMD_TIMEOUT_SEC = 1800  # 30 min per install command


def _is_allowlisted_setup_command(cmd: str) -> bool:
    """True when ``cmd`` is an install-only command safe to replay.

    Strips a leading ``sudo`` and any ``KEY=VALUE`` env-assignment prefixes, then
    requires the remainder to start with a known package/tool installer. Rejects
    anything with shell control operators that could chain an arbitrary payload.

    Args:
        cmd: The raw command string.

    Returns:
        bool: ``True`` when the command is a single allowlisted installer.
    """
    text = (cmd or "").strip()
    if not text:
        return False
    # Reject command substitution / backticks / newlines outright — these can
    # smuggle an arbitrary payload regardless of tokenization.
    if re.search(r"[`\n]|\$\(", text):
        return False
    # Guard against genuine shell chaining/redirection while allowing pip/pkg
    # *version specifiers* that legitimately contain ``>``/``<`` (e.g.
    # ``transformers>=4.58``, ``torch<2.11``). The previous guard rejected any
    # ``>``/``<``/``|``/``&``, which false-positived on essentially every
    # ``pip install 'pkg>=x.y'`` and silently skipped the enablement
    # env-upgrade (the durable replay never ran, so the model was never
    # actually enabled). We first neutralise the safe, non-shell uses, then
    # reject on any leftover metacharacter. Because the replay runs under
    # ``shell=True``, we must ensure no redirection/chaining survives.
    scrubbed = text
    # Drop quoted segments (their contents cannot act as shell operators).
    scrubbed = re.sub(r"'[^']*'", " ", scrubbed)
    scrubbed = re.sub(r'"[^"]*"', " ", scrubbed)
    # Drop version-specifier / comparison operators used by pip & friends:
    #   ``>=`` ``<=`` ``==`` ``!=`` ``~=`` and a bare ``>``/``<`` that is
    #   immediately followed by a version-like token (digit or letter), which
    #   is a requirement spec, never a redirection (redirection needs a path
    #   target, typically separated by whitespace).
    scrubbed = re.sub(r"(>=|<=|==|!=|~=)", " ", scrubbed)
    scrubbed = re.sub(r"[<>](?=[0-9A-Za-z])", " ", scrubbed)
    # Any remaining shell chaining/redirection metacharacter => unsafe.
    if re.search(r"[;&|<>]", scrubbed):
        return False
    # Strip a leading sudo and leading KEY=VALUE env assignments.
    text = re.sub(r"^\s*sudo\s+", "", text)
    text = re.sub(r"^(?:\s*[A-Za-z_][A-Za-z0-9_]*=[^\s]*\s+)+", "", text)
    return any(re.match(pat, text) for pat in _SETUP_CMD_ALLOWLIST)


_now_iso = functools.partial(now_iso, "auto")


def _resolve_setup_commands(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
) -> list[str]:
    """Resolve the ordered, deduped enablement setup commands to replay.

    Sources (in order; deduped preserving first occurrence): base commands
    stacked from prior rounds (``params['enablement_setup_commands']``) then the
    current specialist's ``specialist_done.setup_commands``. Non-string / blank
    entries are dropped; the list is capped at :data:`_SETUP_CMD_MAX`.

    Args:
        params: The integrate_patch task params.
        done_payload: The specialist ``specialist_done`` payload (may be None).

    Returns:
        list[str]: Ordered unique candidate setup commands (pre-allowlist).
    """
    out: list[str] = []
    seen: set[str] = set()
    sources: list[Any] = []
    base = params.get("enablement_setup_commands")
    if isinstance(base, list):
        sources.extend(base)
    if isinstance(done_payload, dict):
        dp = done_payload.get("setup_commands")
        if isinstance(dp, list):
            sources.extend(dp)
    for c in sources:
        s = str(c or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= _SETUP_CMD_MAX:
            break
    return out


def _run_setup_commands(commands: list[str], *, cwd: Path, log_dir: Path) -> dict[str, Any]:
    """Replay allowlisted enablement setup commands (installs) before boot.

    Runs each allowlisted command non-interactively with a per-command timeout,
    appending combined output to ``<log_dir>/enablement_setup.log``. Commands
    that fail the allowlist are skipped (never executed). A non-zero install is
    recorded but does NOT hard-fail the integration — the subsequent boot/gate
    is the source of truth for runnability.

    Args:
        commands: Candidate setup commands (already deduped / capped).
        cwd: Working directory for the commands.
        log_dir: Directory to write ``enablement_setup.log`` into.

    Returns:
        dict[str, Any]: ``{"applied": [...], "skipped": [...], "failed": [...]}``
        where ``applied`` are the allowlisted commands that ran (rc==0).
    """
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    if not commands:
        return {"applied": applied, "skipped": skipped, "failed": failed}
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Logging is best-effort.
        pass
    log_path = log_dir / "enablement_setup.log"
    env = dict(os.environ)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    for cmd in commands:
        if not _is_allowlisted_setup_command(cmd):
            skipped.append(cmd)
            log.warning("integrate_patch: skipping non-allowlisted enablement setup command: %s", cmd)
            continue
        log.info("integrate_patch: enablement setup replay: %s", cmd)
        try:
            proc = subprocess.run(  # noqa: S602  # nosec B602 - allowlisted install-only shell command.
                cmd,
                shell=True,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=_SETUP_CMD_TIMEOUT_SEC,
            )
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"$ {cmd}\n{proc.stdout}\n{proc.stderr}\n(rc={proc.returncode})\n\n")
            except OSError:
                # Logging is best-effort.
                pass
            if proc.returncode == 0:
                applied.append(cmd)
            else:
                failed.append(cmd)
                log.warning("integrate_patch: enablement setup rc=%d for: %s", proc.returncode, cmd)
        except (subprocess.TimeoutExpired, OSError) as exc:
            failed.append(cmd)
            log.warning("integrate_patch: enablement setup errored (%s) for: %s", type(exc).__name__, cmd)
    return {"applied": applied, "skipped": skipped, "failed": failed}


def _root_contains_patch_targets(root: Path, patch_paths: list[Path]) -> bool:
    """True when *every* supplied patch has all its modify/delete targets
    present under ``root`` (at some ``-p`` strip level).

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
    explicit: str | None,
    patch_paths: list[Path] | None = None,
) -> Path | None:
    """Pick the framework source root for patches.

    Precedence: explicit param (must lie under
    :func:`resolve_source_file_allowlist`) → first allowlist root whose tree
    actually contains the patch targets (target-aware: a ``vllm/...`` patch must
    apply under the vllm root, not the first allowlist entry which is ``aiter``)
    → first existing git root → first existing dir. None when nothing resolves.

    Args:
        explicit: Explicit framework-root override, or ``None`` to use the
            allowlist. Overrides outside the allowlist are rejected.
        patch_paths: Patch target paths used to pick the allowlist root whose
            tree actually contains them.

    Returns:
        The resolved framework source root, or ``None`` when nothing resolves.
    """
    if explicit:
        try:
            p = Path(explicit).resolve()
        except (OSError, RuntimeError):
            log.warning(
                "integrate_patch: framework_source_root override %r could not be resolved",
                explicit,
            )
            return None
        if not p.is_dir():
            log.warning(
                "integrate_patch: framework_source_root override %r does not exist",
                explicit,
            )
            return None
        for r in resolve_source_file_allowlist():
            try:
                root = Path(r).resolve()
            except (OSError, RuntimeError):
                continue
            if _is_within(p, root):
                return p
        log.warning(
            "integrate_patch: framework_source_root override %r rejected (not under allowlist)",
            explicit,
        )
        return None
    roots = [Path(r) for r in resolve_source_file_allowlist()]
    # Target-aware: prefer the root that actually holds the patch's targets.
    if patch_paths:
        for p in roots:
            if p.is_dir() and _root_contains_patch_targets(p, patch_paths):
                return p
    for p in roots:
        if p.is_dir() and (p / ".git").exists():
            return p
    # Last resort: a non-git dir.
    for p in roots:
        if p.is_dir():
            return p
    return None


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


def _derive_lane(params: dict[str, Any]) -> str:
    """Derive the retry lane name from integrate_patch params.

    Returns:
        ``"enablement"``, ``"perf_framework"``, or ``"perf_explore"``.
    """
    if params.get("enablement"):
        return "enablement"
    if params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"):
        return "perf_framework"
    return "perf_explore"


def _preflight_missing_targets(
    framework_root: Path,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """Return per-patch records for patches whose modify/delete targets are
    absent from ``framework_root`` at every ``-p`` strip level.

    A hallucinated-layout patch (e.g. modifying a CUDA-only file on a ROCm
    build) can never apply; flagging it here yields an actionable advisory
    instead of an opaque ``git_apply_failed`` after a wasted apply attempt.
    Patches supplied directly via ``params.patches`` bypass the
    authoring-time ``specialist_patch_safety`` gate, so they are checked here.

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


def _localization_paths_outside_allowlist(
    touched_paths: list[str],
    framework_root: Path | None,
    allow_roots: list[str],
) -> list[str]:
    """Return the touched paths that resolve outside the allowed source roots.

    A localization diff (M2) may only write under the source-file allowlist or
    the attempt-local root. Paths are resolved against ``framework_root`` when
    relative. Returns the offending paths (empty when all are in-bounds).

    Args:
        touched_paths: Repo-relative paths the localization diff touches.
        framework_root: The tree the diff applies to (resolves relative paths).
        allow_roots: The allowed root prefixes (allowlist + attempt-local root).

    Returns:
        list[str]: Paths that fall outside every allowed root.
    """
    roots = [Path(r).resolve() for r in allow_roots if str(r).strip()]
    if framework_root is not None:
        roots.append(Path(framework_root).resolve())
    if not roots:
        return []
    outside: list[str] = []
    for rel in touched_paths:
        rel_s = str(rel or "").strip()
        if not rel_s:
            continue
        base = framework_root if framework_root is not None else Path("/")
        cand = (base / rel_s).resolve() if not Path(rel_s).is_absolute() else Path(rel_s).resolve()
        if not any(_is_within(cand, root) for root in roots):
            outside.append(rel_s)
    return outside


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


def _git_apply_collect_feedback(
    framework_root: Path,
    patch_path: Path,
    *,
    three_way: bool = False,
) -> "tuple[bool, str, ApplyFeedback | None]":
    """Like :func:`_git_apply` but also returns an :class:`ApplyFeedback` on failure.

    On success returns ``(True, "", None)``.  On failure returns
    ``(False, stderr, ApplyFeedback)`` where *ApplyFeedback* carries the
    combined stderr from both the initial and ``-3`` attempt, the list of
    tried ``-p`` levels, and a source-context snippet.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        three_way: Whether to fall back to ``-3`` on first failure.

    Returns:
        ``(ok, err, feedback)`` — feedback is ``None`` on success.
    """
    from ._nogit_patch import _P_LEVELS

    # Collect per-level check stderr for the feedback record.
    tried_levels: list[int] = []
    level_stderrs: list[str] = []
    for lvl in _P_LEVELS:
        ok_check, stderr_check = _run_git_apply(
            framework_root, patch_path, p_level=lvl, three_way=three_way, check_only=True
        )
        tried_levels.append(lvl)
        if stderr_check:
            level_stderrs.append(f"-p{lvl}: {stderr_check}")
        if ok_check:
            # Level works; now apply for real.
            ok_apply, stderr_apply = _run_git_apply(
                framework_root, patch_path, p_level=lvl, three_way=three_way, check_only=False
            )
            if ok_apply:
                return True, "", None
            feedback = build_apply_feedback(
                patch_path,
                channel="git",
                tried_levels=tried_levels,
                stderr=stderr_apply,
                framework_root=framework_root,
            )
            return False, stderr_apply, feedback

    # All levels failed; retry with -3.
    if not three_way:
        ok3, err3, fb3 = _git_apply_collect_feedback(framework_root, patch_path, three_way=True)
        if ok3:
            return True, "", None
        # Merge both sets of stderrs.
        all_stderrs = "\n".join(level_stderrs)
        if err3:
            all_stderrs = all_stderrs + "\n-3 retry: " + err3 if all_stderrs else "-3 retry: " + err3
        feedback = build_apply_feedback(
            patch_path,
            channel="git",
            tried_levels=tried_levels,
            stderr=all_stderrs,
            framework_root=framework_root,
        )
        return False, all_stderrs, feedback

    all_stderrs = "\n".join(level_stderrs)
    feedback = build_apply_feedback(
        patch_path,
        channel="git",
        tried_levels=tried_levels,
        stderr=all_stderrs,
        framework_root=framework_root,
    )
    return False, all_stderrs, feedback


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
        # Non-git directory or other git status errors: treat as clean.
        log.debug(
            "integrate_patch: git status rc=%d in %s (not a git repo?), treating as clean",
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
    return f"git stash pop {ref} rc={cp.returncode}: {(cp.stderr or '').strip()}; user changes remain in git stash"


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
    must already have been stashed before candidate apply.

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

    Used to scope the commit-on-KEEP to only the files this patch touched.

    Per header pair (``old`` ``---``, ``new`` ``+++``):
      * created / modified → the ``new`` target exists post-apply → emit it.
      * deleted → ``new`` is ``/dev/null`` (or its target is gone)
        and ``old`` existed pre-apply → emit the ``old`` path so the subsequent
        ``git add -A -- <path>`` stages the removal of a tracked file.
    A header that resolves to neither is dropped so ``git add`` cannot error.

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
            rel_new = _strip_path_prefix(new, lvl) if new and new != _PATCH_DEV_NULL else None
            rel_old = _strip_path_prefix(old, lvl) if old and old != _PATCH_DEV_NULL else None
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
    """Commit only the patch-touched ``paths`` to git for cross-cycle durability.

    Committing each KEEP makes wins survive a later cycle's ``git checkout -- .``
    revert fallback. The commit is scoped to the exact paths the patch touched
    (never ``git add -A``). Best-effort: a commit failure is non-fatal.

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
    # "nothing to commit" is a benign no-op.
    out = (cp.stdout + cp.stderr).lower()
    if "nothing to commit" in out:
        return True, "nothing to commit"
    return False, cp.stderr.strip()


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

    Security: a resolved patch path must live inside the specialist workspace
    (or its worktree); an absolute path pointing outside the sandbox is dropped.
    Both sides are ``resolve()``-d first so a symlinked workspace still matches.

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
                "integrate_patch: patch %r resolves outside the specialist workspace (%s); dropping for safety",
                c,
                specialist_workspace,
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
        rel_target: The framework-relative target, normalized to the matched
            allowlisted root via ``_resolve_artifact_target`` (an author's
            absolute target is converted to this relative form). Used for
            reporting AND as the framework-relative key for the durable KEEP
            source snapshot.
        kind: Free-form artifact kind label (e.g. ``config_json``).
        description: Free-form human description.
    """

    source: Path
    target: Path
    rel_target: str
    kind: str = ""
    description: str = ""


def _resolve_artifact_target(rel_target: str) -> tuple[Path, str] | None:
    """Resolve an artifact target (framework-relative, or absolute) to a path.

    A relative target picks the allowlisted framework root whose tree already
    contains the target's parent directory (so a ``vllm/...`` config lands under
    the vllm root); else the first existing root. An absolute target is accepted
    ONLY when it resolves strictly inside an allowlisted root. Either way the
    resolved path must stay within the chosen root (no ``..`` escape).

    Args:
        rel_target: The install path authored by the specialist (framework-
            relative, or an absolute path inside an allowlisted root).

    Returns:
        A ``(absolute_target, framework_relative_target)`` tuple, or ``None``
        when nothing resolves safely. ``framework_relative_target`` is the path
        relative to the matched root (POSIX). Callers MUST persist THIS as the
        artifact ``rel_target`` so the durable KEEP source snapshot captures the
        installed file even when the author used an absolute path.
    """
    rel = (rel_target or "").strip()
    if not rel or ".." in Path(rel).parts:
        return None
    roots = [Path(r).resolve() for r in resolve_source_file_allowlist()]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return None
    # An absolute target is accepted only when it resolves strictly inside an
    # allowlisted framework root.
    if Path(rel).is_absolute():
        cand = Path(rel).resolve()
        for root in roots:
            if _is_within(cand, root):
                return cand, cand.relative_to(root).as_posix()
        return None
    # Prefer a root whose tree already holds the target's parent dir.
    for root in roots:
        cand = (root / rel).resolve()
        if not _is_within(cand, root):
            continue
        if cand.parent.is_dir():
            return cand, cand.relative_to(root).as_posix()
    # Fall back to the first root that keeps the path contained.
    for root in roots:
        cand = (root / rel).resolve()
        if _is_within(cand, root):
            return cand, cand.relative_to(root).as_posix()
    return None


def _resolve_artifact_specs(
    *,
    specialist_workspace: Path,
    explicit_artifacts: list[dict[str, Any]] | None,
    done_payload: dict[str, Any] | None,
) -> tuple[list[_ArtifactSpec], list[dict[str, str]]]:
    """Resolve non-diff tuned artifacts to install.

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
        # Resolve source inside the workspace sandbox.
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
        resolved = _resolve_artifact_target(tgt_rel)
        if resolved is None:
            errors.append({"artifact": tgt_rel, "error": "target_unresolved_or_escapes_root"})
            continue
        target, rel_norm = resolved
        specs.append(
            _ArtifactSpec(
                source=src_resolved,
                target=target,
                rel_target=rel_norm,
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


_FRAMEWORK_KB_PROVENANCE_PREFIX = "specialist:serving:framework"


def _stamp_framework_kb_provenance(
    done_payload: dict[str, Any] | None,
    *,
    params: dict[str, Any],
    shared_state: Any,
) -> None:
    """Ensure a FRAMEWORK-dispatched deliverable carries KB-writeback provenance.

    Stamps the ``specialist:serving:framework...`` provenance prefix (that
    :meth:`IntegratePatchExecutor._find_frameworkoposal` requires) from the
    dispatch context, so same-framework deliverables reach ``lessons.jsonl``.

    Mutates ``done_payload["proposal_set"][0]`` in place; no-ops when this
    action was not dispatched from FRAMEWORK authoring, or when a proposal
    already carries a matching provenance.

    Args:
        done_payload: The specialist's parsed ``specialist_done.json`` (may
            be ``None``/malformed; no-ops in that case).
        params: The ``integrate_patch`` action's dispatch params (carries
            ``framework_agent_authoring`` / ``framework_agent_candidate_id``
            when this run came from FRAMEWORK_AGENT).
        shared_state: The run's ``SharedState`` (best-effort ``framework``
            read for the provenance suffix).
    """
    if not isinstance(done_payload, dict):
        return
    if not params.get("framework_agent_authoring"):
        return
    pr_url = str(params.get("framework_agent_candidate_id") or "").strip()
    if not pr_url:
        return
    proposals = done_payload.get("proposal_set")
    if not isinstance(proposals, list) or not proposals or not isinstance(proposals[0], dict):
        # No proposal_set entry to stamp; synthesize a minimal anchor.
        done_payload["proposal_set"] = [{}]
        proposals = done_payload["proposal_set"]
    target = proposals[0]
    existing = str(target.get("provenance") or "")
    if existing.startswith(_FRAMEWORK_KB_PROVENANCE_PREFIX):
        return  # cross-framework (or already-stamped) path already complies
    framework = str(getattr(shared_state, "framework", "") or "").strip().lower()
    target["provenance"] = (
        f"{_FRAMEWORK_KB_PROVENANCE_PREFIX}:{framework}" if framework else _FRAMEWORK_KB_PROVENANCE_PREFIX
    )
    target.setdefault("fa_pr_url", pr_url)
    target.setdefault("framework", framework)


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
        """Apply a specialist's patches/config changes and benchmark them."""
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}

        early = await self._stage_resolve(ctx, params, extra)
        if early is not None:
            return early

        # _stage_resolve populates these onto ctx for stage communication.
        specialist_task_id: str = ctx._ip_specialist_task_id  # type: ignore[attr-defined]
        shared_state = ctx._ip_shared_state  # type: ignore[attr-defined]
        specialist_workspace: Path = ctx._ip_specialist_workspace  # type: ignore[attr-defined]
        done_payload: dict[str, Any] = ctx._ip_done_payload  # type: ignore[attr-defined]

        # Rung 3 (M1): provision an attempt-scoped runtime AFTER the Critic gate
        # (in _stage_resolve) and BEFORE any patch apply / setup replay (§10.2).
        provision_early = await self._stage_provision_attempt_runtime(ctx, params, specialist_task_id)
        if provision_early is not None:
            return provision_early

        # Rung 4 (M2): localize a merged-PR / vendored closure into the source
        # tree. Fetch happens post-Critic; a compiled/build closure defers to
        # Rung 5 (clean revert). Localized patches are prepended in _stage_apply.
        localize_early = await self._stage_localize_source(ctx, params, specialist_task_id)
        if localize_early is not None:
            return localize_early

        apply_result = await self._stage_apply(ctx, params, extra, specialist_task_id, shared_state, done_payload)
        if apply_result is not None:
            return apply_result

        # _stage_apply populates these.
        output_root: Path = ctx._ip_output_root  # type: ignore[attr-defined]
        framework_root: Path | None = ctx._ip_framework_root  # type: ignore[attr-defined]
        stash_state: str = ctx._ip_stash_state  # type: ignore[attr-defined]
        stash_note: str = ctx._ip_stash_note  # type: ignore[attr-defined]
        applied: list[Path] = ctx._ip_applied  # type: ignore[attr-defined]
        applied_artifacts: list[dict[str, Any]] = ctx._ip_applied_artifacts  # type: ignore[attr-defined]
        config_changes_applied: dict[str, str] = ctx._ip_config_changes_applied  # type: ignore[attr-defined]
        setup_result: dict[str, Any] = ctx._ip_setup_result  # type: ignore[attr-defined]

        return await self._stage_gate(
            ctx,
            params,
            extra,
            specialist_task_id=specialist_task_id,
            shared_state=shared_state,
            done_payload=done_payload,
            output_root=output_root,
            framework_root=framework_root,
            stash_state=stash_state,
            stash_note=stash_note,
            applied=applied,
            applied_artifacts=applied_artifacts,
            config_changes_applied=config_changes_applied,
            setup_result=setup_result,
        )

    # ---------------------------------------------------------------------------
    # Stage helpers (called sequentially by __call__)
    # ---------------------------------------------------------------------------

    async def _stage_resolve(
        self,
        ctx: Any,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Guards (multi-node, task-id, workspace, Critic) + param normalisation.

        Returns an early-exit result dict on failure, or None to continue.
        Stores resolved values as ``ctx._ip_*`` attributes for the next stages.
        """
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

        shared_state = extra.get("shared_state") or extra.get("state")
        if shared_state is not None and not params.get("accuracy_baseline"):
            _base_acc = getattr(shared_state, "baseline_accuracy", 0.0)
            if isinstance(_base_acc, (int, float)) and _base_acc > 0:
                params["accuracy_baseline"] = float(_base_acc)

        # Launch-only mode: pure bench of a pre-built runtime (no specialist, no Critic).
        if params.get("enablement_launch_only"):
            task_id = str(getattr(ctx.task, "task_id", "") or "build_launch_probe")
            scratch = runs_dir(self.session_dir, "integrate_patch", task_id)
            scratch.mkdir(parents=True, exist_ok=True)
            ctx._ip_specialist_task_id = task_id  # type: ignore[attr-defined]
            ctx._ip_shared_state = shared_state  # type: ignore[attr-defined]
            ctx._ip_specialist_workspace = scratch  # type: ignore[attr-defined]
            ctx._ip_done_payload = {}  # type: ignore[attr-defined]
            return None

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

        specialist_workspace = runs_dir(self.session_dir, "specialist", specialist_task_id)
        if not specialist_workspace.is_dir():
            return {
                "status": "failed",
                "error_class": "missing_specialist",
                "error": (f"specialist workspace not found at {specialist_workspace}"),
                "specialist_task_id": specialist_task_id,
            }

        # Critic-verdict gate — enforced BEFORE any side effect. No-op when
        # SharedState is absent. Override is out-of-band only (HYPERLOOM_BYPASS_CRITIC=1).
        if shared_state is not None and os.environ.get("HYPERLOOM_BYPASS_CRITIC") != "1":
            if params.get("bypass_critic"):
                log.warning(
                    "integrate_patch executor: in-band bypass_critic ignored; "
                    "enforcing Critic verdict for specialist_task_id=%r (operator "
                    "override is HYPERLOOM_BYPASS_CRITIC=1, out-of-band only).",
                    specialist_task_id,
                )
            try:
                from ...policy.gate import INTEGRATE_PATCH_PERMISSIVE_VERDICTS as _PERMISSIVE
            except Exception:  # noqa: BLE001
                _PERMISSIVE = frozenset({"approve", "advise"})
            try:
                recorded = shared_state.get_specialist_patch_verdict(specialist_task_id)
            except AttributeError:
                recorded = ""
            if (recorded or "").lower() not in _PERMISSIVE:
                _detail = f"verdict {recorded!r}" if recorded else "no Critic verdict on record"
                return {
                    "status": "rejected_by_critic",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "config_changes_applied": {},
                    "reason": (
                        f"integrate_patch requires a permissive Critic verdict "
                        f"(approve/advise) for specialist task "
                        f"{specialist_task_id!r}; {_detail}. Refusing to run. "
                        f"Set HYPERLOOM_BYPASS_CRITIC=1 out-of-band to force."
                    ),
                }

        done_payload = _read_done_payload(specialist_workspace)
        _stamp_framework_kb_provenance(done_payload, params=params, shared_state=shared_state)

        ctx._ip_specialist_task_id = specialist_task_id  # type: ignore[attr-defined]
        ctx._ip_shared_state = shared_state  # type: ignore[attr-defined]
        ctx._ip_specialist_workspace = specialist_workspace  # type: ignore[attr-defined]
        ctx._ip_done_payload = done_payload  # type: ignore[attr-defined]
        return None

    async def _stage_provision_attempt_runtime(
        self,
        ctx: Any,
        params: dict[str, Any],
        specialist_task_id: str,
    ) -> dict[str, Any] | None:
        """Provision the attempt-scoped runtime from ``params['runtime_candidate']``.

        Rung 3 (M1). No-op when no candidate is present or in multi-node mode.
        Runs a disk preflight, delegates provision+probe to the framework
        adapter, and on success stores the resolved runtime on
        ``ctx._ip_provision_result`` / ``ctx._ip_stack_action`` for the gate to
        activate via the YAML-layer ``runtime_override``. Returns an early-exit
        ``reverted`` dict on any provision failure (no patch side effects yet),
        or ``None`` to continue.
        """
        ctx._ip_provision_result = None  # type: ignore[attr-defined]
        ctx._ip_stack_action = None  # type: ignore[attr-defined]
        raw = params.get("runtime_candidate")
        if not isinstance(raw, dict) or not raw:
            return None

        from ._multi_node_env import is_multi_node

        if is_multi_node():
            log.info("integrate_patch: skipping runtime provision in multi-node mode")
            return None

        from ...framework.adapters import get_adapter
        from ...framework.stack_actions import EnablementStackAction

        action = EnablementStackAction.from_state(raw)
        attempt_dir = (
            self.session_dir / "enablement" / "stacks" / (action.framework or "unknown") / (specialist_task_id or "attempt")
        )

        from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

        try:
            disk_preflight(attempt_dir.parent, n_candidates=1)
        except DiskPreflightError as exc:
            return {
                "status": "reverted",
                "error_class": "disk_preflight_failed",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": f"attempt-runtime provision aborted: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 — preflight is best-effort advisory
            log.warning("integrate_patch: disk preflight raised (%r); continuing", exc)

        adapter = get_adapter(action.framework)
        try:
            result = adapter.provision(action, attempt_dir)
            if result.ok and not adapter.probe(result, action):
                from ...framework.stack_actions import ProvisionResult as _PR

                result = _PR(ok=False, log_path=result.log_path, error="adapter probe failed after provision")
        except Exception as exc:  # noqa: BLE001 — provision failure is a clean revert, not a crash
            log.exception("integrate_patch: attempt-runtime provision raised")
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": f"attempt-runtime provision raised: {exc!r}",
            }

        if not result.ok:
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_failed",
                "error": result.error,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": f"attempt-runtime provision failed: {result.error}",
                "provision_result": result.to_state(),
            }

        # Record the attempt venv root on the action so KEEP can persist it and
        # resume/GC can find it.
        action = EnablementStackAction.from_state(
            {**action.to_state(), "attempt_venv_root": result.runtime.venv_root}
        )
        ctx._ip_provision_result = result  # type: ignore[attr-defined]
        ctx._ip_stack_action = action  # type: ignore[attr-defined]
        ctx._ip_attempt_venv_root = result.runtime.venv_root  # type: ignore[attr-defined]
        log.info(
            "integrate_patch: attempt runtime provisioned for %s (venv=%s, versions=%s)",
            action.framework,
            result.runtime.venv_root,
            result.installed_versions,
        )
        return None

    @staticmethod
    def _gc_attempt_dir(attempt_dir: Path) -> None:
        """Remove a half/failed attempt-runtime dir (best-effort)."""
        try:
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001 — GC is best-effort
            log.debug("integrate_patch: attempt-dir GC failed for %s", attempt_dir, exc_info=True)

    async def _stage_localize_source(
        self,
        ctx: Any,
        params: dict[str, Any],
        specialist_task_id: str,
    ) -> dict[str, Any] | None:
        """Fetch/synthesize a localization diff and stage it for _stage_apply.

        Rung 4 (M2). No-op when no ``localization_candidate`` is present or in
        multi-node mode. Fetches the merged-PR / vendored diff (post-Critic),
        rejects a compiled / build-backend closure to Rung 5 (clean revert),
        enforces the source-file allowlist (+ the attempt-local root only), and
        writes the diff to a patch file recorded on ``ctx._ip_localization_patches``
        which ``_stage_apply`` prepends to the patch set. Returns an early-exit
        ``reverted`` dict on any gate/fetch failure (no tree mutation yet), or
        ``None`` to continue.
        """
        ctx._ip_localization_patches = []  # type: ignore[attr-defined]
        ctx._ip_localization_manifest = {}  # type: ignore[attr-defined]
        raw = params.get("localization_candidate")
        if not isinstance(raw, dict) or not raw:
            return None

        from ._multi_node_env import is_multi_node

        if is_multi_node():
            log.info("integrate_patch: skipping localization in multi-node mode")
            return None

        from ...framework.localization import build_localization_diff
        from ...framework.stack_actions import EnablementStackAction

        action = EnablementStackAction.from_state(raw)

        from hyperloom.agents.framework.sources import github as _gh

        def _base_reverted(error_class: str, reason: str) -> dict[str, Any]:
            return {
                "status": "reverted",
                "error_class": error_class,
                "error": reason,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": reason,
            }

        try:
            diff_text, touched_paths, verdict = build_localization_diff(
                action,
                fetch_pr_patches=lambda slug, num: _gh.pr_patches(slug, num),
                fetch_raw_file=lambda slug, ref, path: _gh.fetch_raw_file(slug, ref, path),
            )
        except Exception as exc:  # noqa: BLE001 — fetch failure is a clean revert
            log.exception("integrate_patch: localization fetch raised")
            return _base_reverted("localization_fetch_failed", f"localization fetch raised: {exc!r}")

        if not verdict.is_localizable:
            error_class = (
                "localization_rung5_deferred" if verdict.kind == "needs_rung5" else "localization_fetch_failed"
            )
            return _base_reverted(error_class, f"localization not applicable: {verdict.reason}")
        if not diff_text.strip():
            return _base_reverted("localization_fetch_failed", "localization produced an empty diff")

        # Allowlist gate: touched paths must resolve under the source-file
        # allowlist or the attempt-local root only (no global env mutation).
        framework_root: Path | None = _resolve_framework_root(
            params.get("framework_source_root") or None, patch_paths=[]
        )
        allow_roots = list(resolve_source_file_allowlist())
        attempt_root = str(getattr(ctx, "_ip_attempt_venv_root", "") or "")
        if attempt_root:
            allow_roots.append(str(Path(attempt_root).parent))
        outside = _localization_paths_outside_allowlist(touched_paths, framework_root, allow_roots)
        if outside:
            return _base_reverted(
                "localization_outside_allowlist",
                f"localization touches path(s) outside the allowlist: {outside[:8]}",
            )

        loc_dir = runs_dir(self.session_dir, "integrate_patch", str(getattr(ctx.task, "task_id", "") or "localize"))
        loc_dir = loc_dir / "localization"
        loc_dir.mkdir(parents=True, exist_ok=True)
        gap_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.gap_id or "localization")
        patch_path = loc_dir / f"{gap_slug}.patch"
        patch_path.write_text(diff_text, encoding="utf-8")

        ctx._ip_localization_patches = [patch_path]  # type: ignore[attr-defined]
        ctx._ip_stack_action = action  # type: ignore[attr-defined]
        ctx._ip_localization_touched = list(touched_paths)  # type: ignore[attr-defined]
        log.info(
            "integrate_patch: localization staged %s (%d file(s), kind=%s)",
            patch_path,
            len(touched_paths),
            action.kind,
        )
        return None

    async def _stage_apply(
        self,
        ctx: Any,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        shared_state: Any,
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Setup replay, patch/artifact apply, pending_integrate sentinel.

        Returns an early-exit result dict on failure/no-patches/apply_only,
        or None to continue to bench+gate. Stores output values as ``ctx._ip_*``.
        """
        setup_result: dict[str, Any] = {"applied": [], "skipped": [], "failed": []}
        if bool(params.get("enablement")):
            setup_cmds = _resolve_setup_commands(params=params, done_payload=done_payload)
            if setup_cmds:
                setup_result = _run_setup_commands(
                    setup_cmds,
                    cwd=self.session_dir,
                    log_dir=runs_dir(
                        self.session_dir, "integrate_patch", str(getattr(ctx.task, "task_id", "") or "setup")
                    ),
                )

        specialist_workspace: Path = ctx._ip_specialist_workspace  # type: ignore[attr-defined]
        explicit_patches = params.get("patches") or None
        patch_paths = _resolve_patch_paths(
            specialist_workspace=specialist_workspace,
            explicit_patches=(list(explicit_patches) if isinstance(explicit_patches, list) else None),
            done_payload=done_payload,
        )
        # Rung 4 (M2): prepend localized closure patches (applied first, before
        # this round's patch) so the enablement composes on top of the localization.
        localization_patches = list(getattr(ctx, "_ip_localization_patches", None) or [])
        if localization_patches:
            seen_loc = {str(p) for p in patch_paths}
            prefix_loc = [p for p in localization_patches if p.is_file() and str(p) not in seen_loc]
            if prefix_loc:
                log.info("integrate_patch: prepending %d localization patch(es)", len(prefix_loc))
                patch_paths = prefix_loc + list(patch_paths)
        base_patches = params.get("enablement_base_patches")
        if bool(params.get("enablement")) and isinstance(base_patches, list) and base_patches:
            seen = {str(p) for p in patch_paths}
            prefix: list[Path] = []
            for bp in base_patches:
                bp_path = Path(str(bp))
                if bp_path.is_file() and str(bp_path) not in seen:
                    prefix.append(bp_path)
                    seen.add(str(bp_path))
            if prefix:
                log.info(
                    "integrate_patch: enablement stacking %d base patch(es) before this round's patch",
                    len(prefix),
                )
                patch_paths = prefix + list(patch_paths)

        config_changes = dict(params.get("config_changes") or {})
        if not config_changes and done_payload:
            cc = done_payload.get("config_changes")
            if isinstance(cc, dict):
                config_changes = {str(k): str(v) for k, v in cc.items()}

        explicit_artifacts = params.get("artifacts")
        artifact_specs, artifact_resolve_errors = _resolve_artifact_specs(
            specialist_workspace=specialist_workspace,
            explicit_artifacts=(list(explicit_artifacts) if isinstance(explicit_artifacts, list) else None),
            done_payload=done_payload,
        )

        _setup_ran = bool(setup_result.get("applied"))
        if not patch_paths and not config_changes and not artifact_specs and not _setup_ran:
            # Launch-only mode: skip the no-patches early-return and fall through to bench.
            if params.get("enablement_launch_only"):
                output_root = runs_dir(self.session_dir, "integrate_patch", specialist_task_id)
                output_root.mkdir(parents=True, exist_ok=True)
                ctx._ip_output_root = output_root  # type: ignore[attr-defined]
                ctx._ip_framework_root = None  # type: ignore[attr-defined]
                ctx._ip_stash_state = "clean"  # type: ignore[attr-defined]
                ctx._ip_stash_note = ""  # type: ignore[attr-defined]
                ctx._ip_applied = []  # type: ignore[attr-defined]
                ctx._ip_applied_artifacts = []  # type: ignore[attr-defined]
                ctx._ip_config_changes_applied = {}  # type: ignore[attr-defined]
                ctx._ip_setup_result = setup_result  # type: ignore[attr-defined]
                return None
            _no_patches: dict[str, Any] = {
                "status": "no_patches",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "artifacts_applied": [],
                "artifact_errors": artifact_resolve_errors,
                "setup_commands_applied": list(setup_result.get("applied") or []),
                "reason": (
                    "neither patches, config_changes, installable artifacts, nor "
                    "allowlisted setup commands were supplied / discoverable for "
                    "this specialist task"
                ),
            }
            if params.get("enablement"):
                _no_patches["enablement"] = True
            return _no_patches

        explicit_framework_root = str(params.get("framework_source_root") or "").strip() or None
        framework_root = _resolve_framework_root(
            explicit_framework_root,
            patch_paths=patch_paths,
        )
        if patch_paths and framework_root is None:
            _lane_early = _derive_lane(params)
            if explicit_framework_root:
                _error_class = "framework_source_root_rejected"
                _error = (
                    f"framework_source_root {explicit_framework_root!r} is not "
                    "under the configured source allowlist"
                )
            else:
                _error_class = "no_framework_agent_root"
                _error = (
                    "no framework_source_root resolved; cannot apply "
                    "patches. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                )
            _early: dict[str, Any] = {
                "status": "apply_failed",
                "error_class": _error_class,
                "error": _error,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "lane": _lane_early,
                "retry_feedback": [],
                "prior_patches": [str(p) for p in patch_paths],
            }
            if params.get("enablement"):
                _early["enablement"] = True
            return _early

        if patch_paths and framework_root is not None:
            missing_records = _preflight_missing_targets(framework_root, patch_paths)
            if missing_records:
                await self._maybe_write_framework_kb_record(
                    done_payload=done_payload,
                    outcome="rejected_apply_fail",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                _lane_missing = _derive_lane(params)
                _missing_result: dict[str, Any] = {
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
                    "lane": _lane_missing,
                    "retry_feedback": [],
                    "prior_patches": [str(p) for p in patch_paths],
                }
                if params.get("enablement"):
                    _missing_result["enablement"] = True
                return _missing_result

        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "integrate_patch", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Write pending_integrate sentinel before any framework tree mutation.
        # The Coordinator clears this after promoting the final result.
        # shared_state.save() is called here (executor owns the sentinel write;
        # the Coordinator owns state promotion on completion).
        if shared_state is not None:
            try:
                shared_state.pending_integrate = {
                    "specialist_task_id": specialist_task_id,
                    "task_id": str(getattr(ctx.task, "task_id", "") or ""),
                    "patches": [str(p) for p in patch_paths],
                    "artifacts": [{"target": str(s.target), "rel_target": s.rel_target} for s in artifact_specs],
                    "config_changes": dict(config_changes),
                    "framework_source_root": str(framework_root or ""),
                    "workspace": str(output_root),
                    # Rung 3: attempt venv root for crash-resume GC (§10.3).
                    "attempt_venv_root": str(getattr(ctx, "_ip_attempt_venv_root", "") or ""),
                    "ts": _now_iso(),
                }
                shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — sentinel is best-effort
                log.exception("integrate_patch: failed to persist pending_integrate sentinel")

        stash_state, stash_note = _git_stash_if_dirty(framework_root)
        if stash_state == "failed":
            log.error(
                "integrate_patch: cannot stash user changes in %s: %s; aborting to avoid data loss",
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

        git_tree = _is_git_tree(framework_root) if framework_root is not None else False
        self._nogit_patch_backups: list[dict[str, Any]] = []

        applied: list[Path] = []
        applied_artifacts: list[dict[str, Any]] = []
        apply_errors: list[dict[str, str]] = []
        apply_feedbacks: list[ApplyFeedback] = []
        for patch in patch_paths:
            if git_tree:
                ok, err, fb = _git_apply_collect_feedback(framework_root, patch, three_way=False)
                if not ok:
                    apply_errors.append({"patch": str(patch), "stderr": err})
                    if fb is not None:
                        apply_feedbacks.append(fb)
                    break
            else:
                nogit_backup_root = output_root / "patch_backups"
                ok, err, backups, fb = _apply_patch_no_git(
                    framework_root,
                    patch,
                    nogit_backup_root,
                    seq_offset=len(self._nogit_patch_backups),
                )
                self._nogit_patch_backups.extend(backups)
                if not ok:
                    apply_errors.append({"patch": str(patch), "stderr": err})
                    if fb is not None:
                        apply_feedbacks.append(fb)
                    break
            applied.append(patch)

        if apply_errors:
            reverted = self._revert_patches(framework_root, applied)
            await self._maybe_write_framework_kb_record(
                done_payload=done_payload,
                outcome="rejected_apply_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            lane = _derive_lane(params)
            base_result: dict[str, Any] = {
                "status": "apply_failed",
                "error_class": "git_apply_failed",
                "error": apply_errors,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "workspace": str(output_root),
                "lane": lane,
                "retry_feedback": [fb.to_dict() for fb in apply_feedbacks],
                "prior_patches": [str(p) for p in patch_paths],
            }
            if bool(params.get("enablement")):
                base_result["enablement"] = True
            return _with_stash_restore(framework_root, stash_state, stash_note, base_result)

        if artifact_specs:
            applied_artifacts, artifact_apply_errors = self._apply_artifacts(
                artifact_specs,
                backup_root=output_root / "artifact_backups",
            )
            if artifact_apply_errors:
                self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                await self._maybe_write_framework_kb_record(
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

        config_changes_applied = dict(config_changes)

        if params.get("apply_only"):
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "applied_no_bench",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [str(p) for p in applied],
                    "patches_reverted": [],
                    "artifacts_applied": applied_artifacts,
                    "config_changes_applied": config_changes_applied,
                    "reason": "apply_only=True; benchmark skipped",
                    "workspace": str(output_root),
                },
            )

        ctx._ip_output_root = output_root  # type: ignore[attr-defined]
        ctx._ip_framework_root = framework_root  # type: ignore[attr-defined]
        ctx._ip_stash_state = stash_state  # type: ignore[attr-defined]
        ctx._ip_stash_note = stash_note  # type: ignore[attr-defined]
        ctx._ip_applied = applied  # type: ignore[attr-defined]
        ctx._ip_applied_artifacts = applied_artifacts  # type: ignore[attr-defined]
        ctx._ip_config_changes_applied = config_changes_applied  # type: ignore[attr-defined]
        ctx._ip_setup_result = setup_result  # type: ignore[attr-defined]
        return None

    async def _stage_gate(
        self,
        ctx: Any,
        params: dict[str, Any],
        extra: dict[str, Any],
        *,
        specialist_task_id: str,
        shared_state: Any,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        config_changes_applied: dict[str, str],
        setup_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Bench + enablement/perf KEEP/REVERT gate.

        Runs _bench_patch, applies the appropriate gate, and returns the
        final integration result. Never returns None.
        """
        # Rung 3: activate the provisioned attempt runtime by threading its
        # YAML-layer override into params so both bench wirings pick it up.
        provision_result = getattr(ctx, "_ip_provision_result", None)
        if provision_result is not None and getattr(provision_result, "ok", False):
            params = dict(params)
            params["runtime_override"] = provision_result.runtime.to_runtime_override()

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
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
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
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "error_class": "bench_exception",
                    "error": repr(exc),
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "config_changes_applied": {},
                    "reason": f"bench raised: {exc!r}",
                    "workspace": str(output_root),
                },
            )

        if params.get("enablement"):
            return await self._gate_enablement(
                params=params,
                extra=extra,
                specialist_task_id=specialist_task_id,
                done_payload=done_payload,
                output_root=output_root,
                framework_root=framework_root,
                stash_state=stash_state,
                stash_note=stash_note,
                applied=applied,
                applied_artifacts=applied_artifacts,
                config_changes_applied=config_changes_applied,
                setup_result=setup_result,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
                ctx=ctx,
            )

        return await self._gate_perf(
            params=params,
            extra=extra,
            specialist_task_id=specialist_task_id,
            shared_state=shared_state,
            done_payload=done_payload,
            output_root=output_root,
            framework_root=framework_root,
            stash_state=stash_state,
            stash_note=stash_note,
            applied=applied,
            applied_artifacts=applied_artifacts,
            config_changes_applied=config_changes_applied,
            bench_result=bench_result,
            gate_evidence=gate_evidence,
            ctx=ctx,
        )

    async def _gate_enablement(
        self,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        config_changes_applied: dict[str, str],
        setup_result: dict[str, Any],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
        ctx: Any = None,
    ) -> dict[str, Any]:
        """Enablement gate: runnability + minimal-correctness.

        Three states:
          accuracy > floor      -> correctness_ok=True  (KEEP, verified)
          accuracy <= floor/NaN -> correctness_ok=False (REVERT, garbage)
          accuracy is None      -> correctness_ok=None  (KEEP but provisional)

        On KEEP, when a Rung 3 attempt runtime was provisioned, the stack action
        is recorded in the result (``enablement_kept_stack_action``) so it
        survives rearm (§8.5). On REVERT / non-KEEP, the attempt runtime dir is
        GC'd (§8.6).
        """
        import math as _math

        stack_action = getattr(ctx, "_ip_stack_action", None) if ctx is not None else None
        provision_result = getattr(ctx, "_ip_provision_result", None) if ctx is not None else None

        def _gc_on_revert() -> None:
            """GC the attempt runtime dir on a non-KEEP enablement outcome."""
            root = str(getattr(ctx, "_ip_attempt_venv_root", "") or "") if ctx is not None else ""
            if root:
                # venv_root is ``<attempt_dir>/venv``; GC the whole attempt dir.
                self._gc_attempt_dir(Path(root).parent)

        from hyperloom.agents.framework.enablement import (
            FailureSignature,
            classify_failure,
            enablement_made_progress,
            runnable_decision,
        )

        new_tput = bench_result.get("output_throughput")
        booted = isinstance(new_tput, (int, float)) and new_tput > 0
        probe_timed_out = bool(gate_evidence.get("timed_out"))

        enablement_accuracy = gate_evidence.get("enablement_accuracy")
        correctness_ok: bool | None
        if isinstance(enablement_accuracy, (int, float)) and not _math.isnan(float(enablement_accuracy)):
            correctness_ok = float(enablement_accuracy) > ENABLEMENT_ACCURACY_FLOOR
        elif isinstance(enablement_accuracy, float) and _math.isnan(enablement_accuracy):
            correctness_ok = False
        else:
            correctness_ok = None

        after_signature = classify_failure(str(bench_result.get("error") or ""))
        before_signature: FailureSignature | None = None
        raw_before = params.get("enablement_before_signature")
        if isinstance(raw_before, dict):
            try:
                before_signature = FailureSignature(**raw_before)
            except (TypeError, ValueError):
                before_signature = None

        runs, run_reason = runnable_decision(
            probe_returncode=0 if booted else 1,
            correctness_ok=correctness_ok,
            probe_timed_out=probe_timed_out,
            before_signature=before_signature,
            after_signature=after_signature,
        )
        if not runs:
            advanced = (not booted) and enablement_made_progress(before_signature, after_signature)
            if advanced:
                stacked_patches = [str(p) for p in applied]
                new_log = str(bench_result.get("error") or "")
                artifacts_reverted = self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                await self._maybe_write_framework_kb_record(
                    done_payload=done_payload,
                    outcome="integrated",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": "advanced",
                        "specialist_task_id": specialist_task_id,
                        "patches_applied": stacked_patches,
                        "patches_reverted": [str(p) for p in reverted],
                        "artifacts_reverted": artifacts_reverted,
                        "config_changes_applied": {},
                        "output_throughput": new_tput,
                        "enablement": True,
                        "advanced": True,
                        "runnable": False,
                        "correctness_verified": False,
                        "reason": (
                            f"enablement progressed: {run_reason}; boot advanced "
                            f"to a new gap ({after_signature.kind}) — patch recorded "
                            f"as a base for the next round"
                        ),
                        "after_signature": after_signature.to_dict(),
                        "enablement_launch_log": new_log,
                        "setup_commands_applied": list(setup_result.get("applied") or []),
                        "bench_result": bench_result,
                        "workspace": str(output_root),
                    },
                )
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            _gc_on_revert()
            await self._maybe_write_framework_kb_record(
                done_payload=done_payload,
                outcome="reverted_smoke_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "output_throughput": new_tput,
                    "enablement": True,
                    "runnable": False,
                    "correctness_verified": correctness_ok is True,
                    "reason": f"enablement not runnable: {run_reason}",
                    "bench_result": bench_result,
                    "workspace": str(output_root),
                },
            )

        provisional = correctness_ok is None
        reason = f"enablement runnable: {run_reason}"
        if provisional:
            reason += " (provisional: booted but eval produced no accuracy; correctness not verified)"
        await self._maybe_write_framework_kb_record(
            done_payload=done_payload,
            outcome="integrated",
            tps_delta_pct=0.0,
            extra=extra,
        )
        kept_result: dict[str, Any] = {
            "status": "kept",
            "specialist_task_id": specialist_task_id,
            "patches_applied": [str(p) for p in applied],
            "patches_reverted": [],
            "artifacts_applied": applied_artifacts,
            "config_changes_applied": config_changes_applied,
            "output_throughput": new_tput,
            "enablement": True,
            "runnable": True,
            "correctness_verified": correctness_ok is True,
            "provisional": provisional,
            "reason": reason,
            "setup_commands_applied": list(setup_result.get("applied") or []),
            "bench_result": bench_result,
            "workspace": str(output_root),
        }
        # Rung 3: record the KEEP'd attempt runtime so it survives rearm (§8.5)
        # and every later bench in this session re-activates it (§10.4).
        if stack_action is not None and provision_result is not None and getattr(provision_result, "ok", False):
            kept_result["enablement_kept_stack_action"] = stack_action.to_state()
            kept_result["enablement_active_runtime"] = provision_result.runtime.to_state()
            kept_result["installed_versions"] = dict(getattr(provision_result, "installed_versions", {}) or {})
        # Rung 4 (M2): editable-refresh the localized closure + snapshot a manifest
        # that survives rearm so the closure is recorded and not re-fetched.
        manifest = self._finalize_localization_keep(
            ctx,
            framework_root=framework_root,
            specialist_task_id=specialist_task_id,
            provision_result=provision_result,
        )
        if manifest:
            kept_result["enablement_localization_manifest"] = manifest
        return _with_stash_restore(framework_root, stash_state, stash_note, kept_result)

    def _finalize_localization_keep(
        self,
        ctx: Any,
        *,
        framework_root: Path | None,
        specialist_task_id: str,
        provision_result: Any,
    ) -> dict[str, Any]:
        """Editable-refresh a localized closure and snapshot its manifest (M2).

        Runs the framework adapter's editable-refresh argv against the attempt
        interpreter (best-effort; skipped when there is no attempt runtime or no
        refresh argv), then records a localization manifest via
        :func:`snapshot_source_layer`. Returns the manifest dict (empty when no
        localization ran).
        """
        touched = list(getattr(ctx, "_ip_localization_touched", None) or [])
        if not touched or framework_root is None:
            return {}
        action = getattr(ctx, "_ip_stack_action", None)
        # Editable-refresh so localized Python changes take effect in the attempt
        # runtime (no-op for plain wheel trees like atom).
        try:
            from ...framework.adapters import get_adapter

            venv_py = ""
            if provision_result is not None and getattr(provision_result, "ok", False):
                venv_py = str(getattr(provision_result.runtime, "python_path", "") or "")
            fw = str(getattr(action, "framework", "") or "")
            argv = get_adapter(fw).editable_refresh_argv(venv_py, str(framework_root)) if venv_py else None
            if argv:
                subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)  # noqa: S603
        except Exception:  # noqa: BLE001 — refresh is best-effort
            log.debug("integrate_patch: localization editable-refresh failed", exc_info=True)
        # Manifest via the existing snapshot mechanism.
        try:
            from ...source_snapshot import snapshot_source_layer

            base_sha = ""
            _cp = _run_git_cp(["-C", str(framework_root), "rev-parse", "HEAD"], timeout=30.0)
            if _cp is not None and getattr(_cp, "returncode", 1) == 0:
                base_sha = (_cp.stdout or "").strip()
            dest = self.session_dir / "optimization_stack" / "localization" / (specialist_task_id or "keep")
            snap = snapshot_source_layer(
                framework_root=framework_root,
                base_sha=base_sha,
                rel_paths=touched,
                dest_dir=dest,
                provenance="localization",
                extra={
                    "specialist_task_id": specialist_task_id,
                    "kind": str(getattr(action, "kind", "") or ""),
                    "repo_url": str(getattr(action, "repo_url", "") or ""),
                    "pr_number": int(getattr(action, "pr_number", 0) or 0),
                },
            )
            return dict(snap) if snap else {}
        except Exception:  # noqa: BLE001 — manifest is best-effort durability
            log.exception("integrate_patch: localization snapshot failed")
            return {}

    async def _gate_perf(
        self,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        shared_state: Any,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        config_changes_applied: dict[str, str],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        """Throughput KEEP / REVERT decision with optional stack rebench."""
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

        keep_threshold_pct = float(params.get("keep_threshold_pct", self.keep_threshold_pct))
        new_tput = bench_result.get("output_throughput")
        delta_pct = None
        if isinstance(new_tput, (int, float)) and new_tput > 0 and base_tput > 0:
            delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0

        accuracy_pass: bool | None = gate_evidence.get("accuracy_pass")
        fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
        acc_required = bool(params.get("require_accuracy_for_keep", fw_authored))
        acc_baseline = params.get("accuracy_baseline")
        if acc_required and not acc_baseline:
            _ss = extra.get("shared_state") or extra.get("state")
            if _ss is not None:
                acc_baseline = getattr(_ss, "baseline_accuracy", None)
        acc_block, acc_reason, acc_degraded = accuracy_keep_block(
            accuracy_pass,
            required=acc_required,
            baseline_accuracy=acc_baseline,
        )
        if acc_degraded:
            log.warning(
                "integrate_patch: accuracy gate required but no baseline accuracy; "
                "KEEP allowed on throughput only (task=%s)",
                specialist_task_id,
            )
        gate_pass = delta_pct is not None and delta_pct >= keep_threshold_pct and not acc_block

        if not gate_pass:
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            reasons: list[str] = []
            if delta_pct is None:
                reasons.append("no measurable throughput")
            elif delta_pct < keep_threshold_pct:
                reasons.append(f"throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%")
            if acc_block and acc_reason:
                reasons.append(acc_reason)
            _tput_ok = delta_pct is not None and delta_pct >= keep_threshold_pct
            revert_status = (
                "accuracy_unavailable_reject" if (acc_block and accuracy_pass is None and _tput_ok) else "reverted"
            )
            await self._maybe_write_framework_kb_record(
                done_payload=done_payload,
                outcome="reverted_smoke_fail",
                tps_delta_pct=float(delta_pct or 0.0),
                extra=extra,
            )
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": revert_status,
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
                },
            )

        if params.get("enable_stack_rebench", True) and base_tput > 0:
            confirm = await self._confirm_stack_rebench(
                params=params,
                output_root=output_root,
                config_changes_applied=config_changes_applied,
                specialist_task_id=specialist_task_id,
                base_tput=base_tput,
            )
            rb_acc_block, rb_acc_reason, _rb_degraded = accuracy_keep_block(
                confirm["accuracy_pass"],
                required=acc_required,
                baseline_accuracy=acc_baseline,
            )
            if not confirm["stable"] or rb_acc_block:
                artifacts_reverted = self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                reasons = []
                if not confirm["stable"]:
                    reasons.append(
                        f"stack rebench {confirm['tput']} below stability floor {confirm['stable_floor']:.2f}"
                    )
                if confirm["accuracy_pass"] is False:
                    reasons.append("accuracy regression on rebench")
                elif rb_acc_block and rb_acc_reason:
                    reasons.append(rb_acc_reason)
                rb_revert_status = (
                    "accuracy_unavailable_reject"
                    if (rb_acc_block and confirm["accuracy_pass"] is None and confirm["stable"])
                    else "reverted"
                )
                await self._maybe_write_framework_kb_record(
                    done_payload=done_payload,
                    outcome="reverted_smoke_fail",
                    tps_delta_pct=float(delta_pct or 0.0),
                    extra=extra,
                )
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": rb_revert_status,
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
                    },
                )
            if isinstance(confirm["tput"], (int, float)) and confirm["tput"] > 0:
                new_tput = confirm["tput"]
                delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0
            if confirm["accuracy_pass"] is not None:
                accuracy_pass = confirm["accuracy_pass"]

        await self._maybe_write_framework_kb_record(
            done_payload=done_payload,
            outcome="integrated",
            tps_delta_pct=float(delta_pct or 0.0),
            extra=extra,
        )
        # Commit the KEEP so a later REVERT checkout fallback can't wipe this
        # win (best-effort, non-fatal).
        try:
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

        source_snapshot_dir = ""
        source_base_sha = ""
        try:
            from ...source_snapshot import snapshot_source_layer

            if framework_root is not None:
                _cp = _run_git_cp(["-C", str(framework_root), "rev-parse", "HEAD"], timeout=30.0)
                if _cp is not None and getattr(_cp, "returncode", 1) == 0:
                    source_base_sha = (_cp.stdout or "").strip()
                rel_paths = list(_patch_touched_paths(framework_root, applied))
                rel_paths += [str(a.get("rel_target") or "") for a in (applied_artifacts or []) if isinstance(a, dict)]
                dest = (
                    self.session_dir
                    / "optimization_stack"
                    / "src"
                    / (specialist_task_id or str(getattr(ctx.task, "task_id", "") or "keep"))
                )
                snap = snapshot_source_layer(
                    framework_root=framework_root,
                    base_sha=source_base_sha,
                    rel_paths=rel_paths,
                    dest_dir=dest,
                    provenance="integrate_patch",
                    extra={"specialist_task_id": specialist_task_id},
                )
                if snap:
                    source_snapshot_dir = str(snap.get("snapshot_dir") or "")
        except Exception:  # noqa: BLE001 — snapshot is best-effort durability
            log.exception("integrate_patch: source-layer snapshot failed")

        return _with_stash_restore(
            framework_root,
            stash_state,
            stash_note,
            {
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
                "source_snapshot": source_snapshot_dir,
                "framework_root": str(framework_root or ""),
                "base_sha": source_base_sha,
            },
        )

    # Helpers
    @staticmethod
    def _find_frameworkoposal(
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the first proposal whose provenance starts with
        ``specialist:serving:framework`` (F2-5); ``None`` otherwise so
        the KB writeback hook no-ops for legacy / kernel outputs.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.

        Returns:
            The matching framework proposal dict, or ``None`` when absent.
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
            if provenance.startswith("specialist:serving:framework"):
                return proposal
        return None

    async def _maybe_write_framework_kb_record(
        self,
        *,
        done_payload: dict[str, Any] | None,
        outcome: str,
        tps_delta_pct: float,
        extra: dict[str, Any],
    ) -> None:
        """Append a JSONL record to ``lessons.jsonl`` when the patch
        came from the FRAMEWORK_AGENT phase.

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
        proposal = self._find_frameworkoposal(done_payload)
        if proposal is None:
            return
        pr_url = str(proposal.get("fa_pr_url") or "").strip()
        pr_sha = str(proposal.get("fa_pr_sha") or "").strip()
        if not pr_url and not pr_sha:
            log.warning(
                "integrate_patch: framework proposal lacks both fa_pr_url and fa_pr_sha; KB writeback skipped",
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
            from ...knowledge.kb_writeback import write_framework_record

            gap_keywords = proposal.get("gap_keywords") or (done_payload or {}).get("gap_keywords") or []
            if isinstance(gap_keywords, str):
                gap_keywords = [gap_keywords]
            changed_files = proposal.get("changed_files") or (done_payload or {}).get("changed_files") or []
            if isinstance(changed_files, str):
                changed_files = [changed_files]
            try:
                accuracy_delta_pct = float(
                    proposal.get("accuracy_delta_pct") or (done_payload or {}).get("accuracy_delta_pct") or 0.0
                )
            except (TypeError, ValueError):
                accuracy_delta_pct = 0.0
            written = await write_framework_record(
                pr_url=pr_url,
                pr_sha=pr_sha,
                patch_path=patch_path,
                outcome=outcome,
                tps_delta_pct=float(tps_delta_pct),
                session_id=session_id,
                framework=str(proposal.get("framework") or (done_payload or {}).get("framework") or "").strip().lower(),
                gap_canonical_id=str(
                    proposal.get("gap_canonical_id") or (done_payload or {}).get("gap_canonical_id") or ""
                ).strip(),
                gap_keywords=[str(k).strip().lower() for k in gap_keywords if str(k).strip()],
                model_class=str(getattr(shared_state, "model_class", "") if shared_state is not None else "").strip(),
                gpu_type=str(getattr(shared_state, "gpu_type", "") if shared_state is not None else "").strip(),
                precision=str(getattr(shared_state, "precision", "") if shared_state is not None else "").strip(),
                applicability=str(
                    proposal.get("applicability") or (done_payload or {}).get("applicability") or ""
                ).strip(),
                provenance=str(proposal.get("provenance") or (done_payload or {}).get("provenance") or "").strip(),
                accuracy_delta_pct=accuracy_delta_pct,
                changed_files=[str(f).strip() for f in changed_files if str(f).strip()],
                source_framework=str(
                    proposal.get("source_framework") or (done_payload or {}).get("source_framework") or ""
                )
                .strip()
                .lower(),
                target_framework=str(
                    proposal.get("target_framework") or (done_payload or {}).get("target_framework") or ""
                )
                .strip()
                .lower(),
            )
            log.info(
                "integrate_patch: wrote framework KB record to %s (outcome=%s pr_url=%s tps_delta=%+.2f%%)",
                written,
                outcome,
                pr_url,
                float(tps_delta_pct),
            )
        except Exception as exc:  # noqa: BLE001 — KB write is best-effort
            log.warning(
                "integrate_patch: framework KB writeback failed: %r",
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
            framework_root: The source root to revert in, or ``None`` (no-op).
            applied: The patches that were applied this run.

        Returns:
            The patches actually reverted (may be the full ``applied`` list
            when the checkout fallback fires).
        """
        reverted: list[Path] = []
        if framework_root is None or not applied:
            return reverted
        nogit_backups = getattr(self, "_nogit_patch_backups", None)
        if nogit_backups is not None and not _is_git_tree(framework_root):
            _revert_patches_no_git(nogit_backups)
            return list(applied)
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
                # Reverse-apply failed: checkout clears all uncommitted at once.
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
            extra_envs=self._framework_run_eval_envs(params),
            remove_args=params.get("base_remove_args"),
            unset_envs=params.get("base_unset_envs"),
            args_mode=str(params.get("base_args_mode") or "append"),
            out_name="integrate_patch.with_envs.yaml",
        )

        # Split config_changes into server args (--flags) and env vars so that
        # --prefixed flags reach EXTRA_{FW}_ARGS instead of being silently dropped.
        _cc_args, _cc_envs = split_config_changes(config_changes_applied)
        variant = GridVariant(
            name=f"integrate-patch-{specialist_task_id[:8]}",
            extra_server_args=merge_server_args(str(params.get("base_extra_args") or "").strip(), _cc_args),
            extra_envs=_cc_envs,
            remove_args=to_str_list(params.get("base_remove_args")),
            unset_envs=to_str_list(params.get("base_unset_envs")),
            args_mode=str(params.get("base_args_mode") or "append"),
            note=f"integrate_patch:{specialist_task_id}",
        )
        _rt = params.get("runtime_override")
        if isinstance(_rt, dict) and _rt:
            variant.runtime_override = {str(k): str(v) for k, v in _rt.items()}

        # Ray-managed GPU execution (phase-3 §3.1 / invariant §6.2): hold a
        # serving lease (num_gpus=TP + serving_slot) for the whole run_grid so
        # the patch benchmark serializes against other serving on the
        # whole-machine mutex instead of colliding with a concurrently-running
        # GPU specialist server on the same card (the observed
        # ``reverted_smoke_fail`` root cause). ``None`` keeps the local path
        # (multi-node / RAY_EXEC off / pytest default).
        from ._ray_serving import maybe_serving_lease

        serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        try:
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
                base_args_mode=str(params.get("base_args_mode") or "append"),
                serving_lease=serving_lease,
            )
        finally:
            if serving_lease is not None:
                serving_lease.close()

        bench: dict[str, Any] = {}
        if results:
            r = results[0]
            bench = {
                "name": r.name,
                "status": r.status,
                "output_throughput": getattr(r, "output_throughput", None),
                "ttft_ms": getattr(r, "ttft_ms", None),
                "itl_ms": getattr(r, "itl_ms", None),
                # Benchmark dir; ``_grade_accuracy`` locates accuracy artifacts here.
                "workspace": str(getattr(r, "workspace", "") or ""),
                "error": getattr(r, "error", "") or "",
                "nonfatal_warnings": list(getattr(r, "nonfatal_warnings", []) or []),
            }

        accuracy_pass: bool | None = None
        # lm-eval writes to ``$EVAL_RESULT_DIR`` under the grid slot, not inside
        # the ``benchmark_*`` workspace. Grade from the slot so the recursive
        # search finds eval output while honoring an explicit ``result_dir``
        # override the same way the grid subprocess does.
        eval_search_root = override_result_dir or (
            str(Path(bench["workspace"]).parent) if bench.get("workspace") else ""
        )
        if bench.get("status") == "succeeded":
            accuracy_pass = self._grade_accuracy(
                eval_search_root,
                params.get("accuracy_baseline"),
                framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
            )

        # Enablement path: surface the raw accuracy so the branch can apply a floor.
        enablement_accuracy: float | None = None
        if bool(params.get("enablement")) and bench.get("status") == "succeeded":
            try:
                eval_results = parse_eval_results(
                    eval_search_root,
                    framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
                )
                acc = eval_results.get("accuracy")
                if isinstance(acc, (int, float)):
                    enablement_accuracy = float(acc)
            except Exception:  # noqa: BLE001 — eval may not produce a result
                log.debug("integrate_patch: enablement eval parse failed", exc_info=True)

        return bench, {"accuracy_pass": accuracy_pass, "enablement_accuracy": enablement_accuracy}

    @staticmethod
    def _framework_run_eval_envs(params: dict[str, Any]) -> dict[str, Any] | None:
        """Force ``RUN_EVAL=true`` for framework-authored source patches.

        Two independent triggers:

        * **Enablement** (``params["enablement"]``): force ``RUN_EVAL=true``
          unconditionally so ``_bench_patch`` can obtain a raw accuracy for the
          runnable gate.
        * **Perf framework authoring**: force only when a comparable baseline
          accuracy exists (``accuracy_baseline > 0``); otherwise leave the
          candidate's ``RUN_EVAL`` to the materializer's default handling.

        Generic EXPLORE integrate_patch is untouched (returns ``None``).

        Args:
            params: The integrate_patch task params.

        Returns:
            ``{"RUN_EVAL": "true"}`` for enablement patches, or for
            framework-authored perf patches that have a positive baseline
            accuracy to compare against; else ``None``.
        """
        if bool(params.get("enablement")):
            return {"RUN_EVAL": "true"}
        fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
        try:
            baseline = float(params.get("accuracy_baseline") or 0.0)
        except (TypeError, ValueError):
            baseline = 0.0
        return {"RUN_EVAL": "true"} if (fw_authored and baseline > 0) else None

    @staticmethod
    def _grade_accuracy(
        result_dir: str,
        baseline_accuracy: Any,
        framework: str | None = None,
    ) -> bool | None:
        """Grade a bench's accuracy against the baseline.

        With a recorded baseline the measured drop is enforced; without one
        (or no eval result) the check is skipped (``None``) and warned loudly.
        For scriptable frameworks (xDiT) ``parse_eval_results`` fails closed on
        a missing quality gate instead of falling back to GSM8K.
        """
        # Accept numeric strings in addition to int/float; non-numeric / missing
        # values fall back to 0.0 (skip).
        try:
            baseline_value = float(baseline_accuracy)
        except (TypeError, ValueError):
            baseline_value = 0.0
        try:
            eval_results = parse_eval_results(result_dir, framework=framework)
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
            extra_envs=self._framework_run_eval_envs(params),
            remove_args=params.get("base_remove_args"),
            unset_envs=params.get("base_unset_envs"),
            args_mode=str(params.get("base_args_mode") or "append"),
            out_name="integrate_patch.rebench.yaml",
        )
        _cc_args_rb, _cc_envs_rb = split_config_changes(config_changes_applied)
        variant = GridVariant(
            name=f"integrate-patch-rebench-{specialist_task_id[:8]}",
            extra_server_args=merge_server_args(base_extra_args, _cc_args_rb),
            extra_envs=_cc_envs_rb,
            remove_args=to_str_list(params.get("base_remove_args")),
            unset_envs=to_str_list(params.get("base_unset_envs")),
            args_mode=str(params.get("base_args_mode") or "append"),
            note=f"integrate_patch_rebench:{specialist_task_id}",
        )
        _rt_rb = params.get("runtime_override")
        if isinstance(_rt_rb, dict) and _rt_rb:
            variant.runtime_override = {str(k): str(v) for k, v in _rt_rb.items()}
        rebench = await measure_stack_rebench(
            config_path=config_path,
            base_extra_args=base_extra_args,
            variant=variant,
            base_tput=base_tput,
            stable_threshold_pct=float(params.get("rebench_stable_threshold_pct", DEFAULT_STACK_STABLE_PCT)),
            output_slot=output_root / "stack_rebench",
            variant_timeout_sec=int(params.get("variant_timeout_sec", self.variant_timeout_sec)),
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            result_dir=override_result_dir,
            magpie_python=params.get("magpie_python") or None,
            base_args_mode=str(params.get("base_args_mode") or "append"),
        )
        # See ``_bench_patch``: lm-eval writes to the grid slot (the parent of
        # ``rebench.workspace``), so grade from there, honoring ``result_dir``.
        rebench_eval_root = override_result_dir or (str(Path(rebench.workspace).parent) if rebench.workspace else "")
        accuracy_pass = (
            self._grade_accuracy(
                rebench_eval_root,
                params.get("accuracy_baseline"),
                framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
            )
            if rebench.workspace
            else None
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
