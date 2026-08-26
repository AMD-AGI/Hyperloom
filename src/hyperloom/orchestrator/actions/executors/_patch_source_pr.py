# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialise an upstream PR candidate into local patch files.

This is the half of upstream-PR work that is genuinely its own: turning a
candidate row into a diff on disk. Everything after that -- apply, structural
vetting, bench, KEEP/REVERT, revert, KB record -- is what every other patch
source already does, and lives in :mod:`integrate_patch`.

Three sources, in priority order: explicit ``params.patches``, a net diff taken
from a worktree checked out at the PR head, and finally the raw ``diff_url``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hyperloom.common.url_safety import require_http_url

from ._git import _run_git, _run_git_cp

log = logging.getLogger(__name__)


def _git_head_sha(framework_root: Path) -> tuple[str | None, str]:
    """``git rev-parse HEAD`` in ``framework_root``; ``(sha, stderr)``,
    sha None on failure.

    Args:
        framework_root: The git checkout to read HEAD from.

    Returns:
        A ``(sha, stderr)`` tuple; ``sha`` is ``None`` on failure with the
        error text in ``stderr``.
    """
    cp = _run_git_cp(["-C", str(framework_root), "rev-parse", "HEAD"], timeout=30.0)
    if cp is None:
        return None, "git rev-parse spawn failed"
    if cp.returncode != 0:
        return None, cp.stderr.strip()
    return cp.stdout.strip() or None, ""


def _candidate_slug(candidate: dict[str, Any]) -> str:
    """Filesystem-safe candidate id (variant names + paths). Prefer
    ``repo/pr_number``.

    Args:
        candidate: The PR metadata row.

    Returns:
        A filesystem-safe slug derived from the candidate's repo / pr_number
        / ref.
    """
    repo = str(candidate.get("repo") or "").replace("/", "-")
    pr = candidate.get("pr_number")
    if repo and pr not in (None, "", 0):
        return f"{repo}-pr-{pr}"
    ref = str(candidate.get("ref") or "").replace(":", "-")
    if repo and ref:
        return f"{repo}-{ref}"
    return repo or ref or "candidate"


def _fetch_diff_to_path(
    diff_url: str,
    dest: Path,
    *,
    timeout_sec: float,
) -> tuple[bool, str]:
    """Curl ``diff_url`` into ``dest`` (.patch path); returns ``(ok, stderr)``.
    Uses curl for consistent HTTPS_PROXY behaviour in restricted-network
    sessions. The scheme is restricted to http/https; the host is logged but
    not restricted.

    Args:
        diff_url: The unified-diff URL to download.
        dest: Destination ``.patch`` path to write.
        timeout_sec: Per-request curl timeout in seconds.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is False on failure with the error
        text in ``stderr``.
    """
    try:
        require_http_url(diff_url, context="diff_url")
    except ValueError as exc:
        return False, str(exc)
    log.info("framework: fetching diff from host=%s", urlparse(diff_url).hostname or "unknown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fsSL",
        "--retry",
        "2",
        "--max-time",
        str(int(timeout_sec)),
        "-o",
        str(dest),
        diff_url,
    ]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 5.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"curl spawn / timeout: {exc!r}"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    if not dest.exists() or dest.stat().st_size == 0:
        return False, "curl wrote empty / missing file"
    return True, ""


def _normalize_repo_id(url_or_slug: str) -> str:
    """Reduce a repo URL / slug to a canonical lowercase ``owner/name`` token.
    Tolerates https, ssh, and bare ``Owner/Name`` forms.

    Args:
        url_or_slug: A repo URL or slug in any supported form.

    Returns:
        The canonical lowercase ``owner/name`` token, or ``""`` when empty.
    """
    s = (url_or_slug or "").strip().lower()
    if not s:
        return ""
    s = s.rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    for sep in ("github.com/", "github.com:"):
        if sep in s:
            s = s.split(sep, 1)[1]
            break
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return s


def _candidate_is_same_repo(
    candidate: dict[str, Any],
    framework_root: Path,
) -> bool:
    """True unless we can positively prove the candidate lives in a different
    repo than the framework_root's origin (where checkout-head's fetch would
    resolve the wrong ref).

    Fails OPEN when inconclusive (no candidate repo, unreadable / non-GitHub
    origin); only fires when both sides yield differing ``owner/name`` tokens.

    Args:
        candidate: The PR metadata row (carries the candidate repo).
        framework_root: The live framework checkout whose origin is compared.

    Returns:
        True unless the candidate is positively proven to live in a different
        repo than ``framework_root``'s origin.
    """
    cand_repo = _normalize_repo_id(str(candidate.get("repo") or candidate.get("discovered_repo_url") or ""))
    if not cand_repo or "/" not in cand_repo:
        return True
    ok, out, _err = _run_git(
        ["-C", str(framework_root), "remote", "get-url", "origin"],
        timeout=30.0,
    )
    if not ok or not out.strip():
        return True
    origin_raw = out.strip()
    # Only a GitHub origin yields a comparable owner/name token; otherwise fail open.
    if "github.com" not in origin_raw.lower():
        return True
    return _normalize_repo_id(origin_raw) == cand_repo


def _materialize_pr_diff_via_worktree(
    framework_root: Path,
    candidate: dict[str, Any],
    dest: Path,
    *,
    timeout_sec: float,
) -> tuple[bool, str]:
    """checkout-head (diff source) mode.

    Fetches the PR head into ``framework_root``, checks it out into an
    isolated worktree (the live KEPT stack is undisturbed), computes the
    PR's net diff against its merge-base, and writes it to ``dest``.
    Worktree always removed in ``finally``. Returns ``(ok, err)``.

    Head ref order: ``candidate.head_sha`` → ``candidate.ref`` →
    ``refs/pull/<pr_number>/head``.

    Args:
        framework_root: The live framework checkout to fetch the PR head into.
        candidate: The PR metadata row (head_sha / ref / pr_number).
        dest: Destination ``.patch`` path the net diff is written to.
        timeout_sec: Per-git-operation timeout in seconds.

    Returns:
        A ``(ok, err)`` tuple; ``ok`` is False on failure with the error text
        in ``err``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    root = str(framework_root)

    head_sha = str(candidate.get("head_sha") or "").strip()
    ref = str(candidate.get("ref") or "").strip()
    pr_number = candidate.get("pr_number")

    # Decide the fetch refspec.
    fetch_ref = ""
    if ref:
        fetch_ref = ref
    elif pr_number not in (None, "", 0):
        fetch_ref = f"refs/pull/{int(pr_number)}/head"

    # Fetch the head ref.
    if fetch_ref:
        ok, _out, err = _run_git(
            ["-C", root, "fetch", "--no-tags", "origin", fetch_ref],
            timeout=timeout_sec,
        )
        if not ok:
            return False, f"git fetch {fetch_ref!r} failed: {err}"
        if not head_sha:
            # FETCH_HEAD now points at the fetched head.
            ok2, out2, err2 = _run_git(
                ["-C", root, "rev-parse", "FETCH_HEAD"],
                timeout=30.0,
            )
            if not ok2 or not out2.strip():
                return False, f"could not resolve FETCH_HEAD: {err2}"
            head_sha = out2.strip()
    if not head_sha:
        return False, ("checkout-head: no head_sha / ref / pr_number on candidate; cannot resolve PR head")

    # Isolated worktree at the fetched head (clean any stale prior-run dir).
    wt_dir = dest.parent / f"wt-{_candidate_slug(candidate)}"
    _run_git(["-C", root, "worktree", "remove", "--force", str(wt_dir)], timeout=60.0)
    ok, _out, err = _run_git(
        ["-C", root, "worktree", "add", "--detach", str(wt_dir), head_sha],
        timeout=timeout_sec,
    )
    if not ok:
        return False, f"git worktree add failed: {err}"
    try:
        # Diff against the merge-base so applying introduces only the PR's
        # own commits, not the full divergence from the live HEAD.
        ok_hb, base_out, _e = _run_git(
            ["-C", root, "rev-parse", "HEAD"],
            timeout=30.0,
        )
        live_head = base_out.strip() if ok_hb else ""
        merge_base = ""
        if live_head:
            ok_mb, mb_out, _mb_e = _run_git(
                ["-C", root, "merge-base", live_head, head_sha],
                timeout=60.0,
            )
            if ok_mb and mb_out.strip():
                merge_base = mb_out.strip()
        diff_range = f"{merge_base}..{head_sha}" if merge_base else head_sha
        ok_d, diff_out, err_d = _run_git(
            ["-C", root, "diff", "--binary", diff_range],
            timeout=timeout_sec,
        )
        if not ok_d:
            return False, f"git diff {diff_range!r} failed: {err_d}"
        if not diff_out.strip():
            return False, (
                f"checkout-head produced an empty diff for range "
                f"{diff_range!r} (PR head already merged into live tree?)"
            )
        try:
            dest.write_text(diff_out, encoding="utf-8")
        except OSError as exc:
            return False, f"could not write diff to {dest}: {exc!r}"
        return True, ""
    finally:
        _run_git(
            ["-C", root, "worktree", "remove", "--force", str(wt_dir)],
            timeout=60.0,
        )
