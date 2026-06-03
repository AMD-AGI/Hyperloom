"""FrameworkPrExecutor — FRAMEWORK_PR phase per-candidate executor.

Counterpart to :class:`IntegratePatchExecutor` for the FRAMEWORK_PR
phase. The Coordinator-side pump enumerates PR candidates via
``fa phase-discover``, gates each one through the Critic
(:meth:`Coordinator._critic_review_framework_pr_candidate` —
``approve`` proceeds, ``reject`` short-circuits with a
``critic_denied`` progress row), and dispatches **this** executor per
approved candidate to:

  1. Fetch the unified diff: when ``params.patches`` is provided, use
     those paths directly; otherwise ``curl candidate.diff_url`` into
     the per-task workspace. (We do NOT shell out to ``fa phase-fetch``
     — apply happens in the live framework_source_roots, not in an
     fa-managed worktree.)
  2. Snapshot the live tree's pre-apply HEAD SHA via ``git rev-parse``
     so the per-candidate REVERT path can ``git reset --hard`` back to
     it without disturbing prior KEEP commits (PR-327 P1.c fix).
  3. Apply the diff via ``git apply`` (single integration channel,
     mirrors ``integrate_patch``).
  4. Bench the patched server with ``run_grid([GridVariant])`` (size=1,
     same throughput + accuracy gate plumbing as ``integrate_patch``).
  5. KEEP / REVERT decision: KEEP commits the change to the live tree
     so the next candidate stacks on top; REVERT runs
     ``git reset --hard <pre_apply_sha>`` followed by ``git clean -fd``
     to restore exactly the pre-apply state.

This is a Coordinator-internal action (``framework_pr_action_not_llm_proposable``
denies any LLM-side delegate / propose_action / request). It is
registered for the FRAMEWORK_PR phase only.

Inputs (``ctx.task.params``):
    candidate (dict, required) — PR metadata row:
        ``{repo, pr_number, ref, title, diff_url, pr_url?, framework?}``
    framework (str, optional) — ``"sglang"`` / ``"vllm"``. Falls back to
        ``candidate["framework"]`` then ``$INFERENCE_OPTIMIZER_FRAMEWORK``.
    batch_id (str, optional) — passed back in the result so the phase
        loop can group ``framework_pr_phase_progress`` entries.
    patches (list[str], optional) — explicit patch paths. When omitted,
        the executor curls ``candidate.diff_url`` into the per-task
        workspace and applies that.
    keep_threshold_pct (float, optional) — default 0.2.
    base_tput (float, optional) — baseline throughput; falls back to
        ``SharedState.baseline_tput``.
    accuracy_baseline (float, optional) — forwarded to the accuracy gate.
    benchmark_script / result_dir / variant_timeout_sec / base_extra_args
        — same semantics as ``integrate_patch``.
    framework_source_root (str, optional) — explicit ``git apply`` target;
        defaults to first existing entry of ``resolve_source_file_allowlist()``.
    apply_only (bool, optional) — skip the bench step (test / smoke).

Outputs (dict returned to the bus as ``delegated_result.result``):
    status: "kept" | "reverted" | "apply_failed" | "no_patch" |
            "fetch_failed" | "applied_no_bench" | "failed"
    output_throughput: float | None
    delta_pct: float | None
    accuracy_pass: bool | None
    candidate: dict (echoes the input row)
    batch_id: str
    patches_applied: list[str]
    patches_reverted: list[str]
    reason: str
    workspace: str
    bench_result: dict | None
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from ._accuracy_gate import accuracy_passed, parse_eval_results
from ._grid_runner import (
    GridVariant,
    VariantResult,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._workload_envs import default_baseline_config, materialize_config_with_envs
from .integrate_patch import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    DEFAULT_VARIANT_TIMEOUT_SEC,
    _git_apply,
    _resolve_framework_root,
)


log = logging.getLogger(__name__)


DEFAULT_DIFF_FETCH_TIMEOUT_SEC: float = 30.0


# ---------------------------------------------------------------------------
# Git checkpoint helpers — FRAMEWORK_PR phase processes candidates serially
# in the same framework_root. To prevent a failed REJECT from clobbering
# previously KEPT patches that share the worktree, every KEEP is committed
# (so it becomes the new HEAD) and every REJECT/failure resets HEAD to the
# sha captured immediately before apply.
# ---------------------------------------------------------------------------
def _git_head_sha(framework_root: Path) -> tuple[str | None, str]:
    """``git rev-parse HEAD`` in ``framework_root``. Returns
    ``(sha, stderr)``; sha is None when the call fails."""
    cmd = ["git", "-C", str(framework_root), "rev-parse", "HEAD"]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git rev-parse spawn failed: {exc!r}"
    if cp.returncode != 0:
        return None, cp.stderr.strip()
    return cp.stdout.strip() or None, ""


def _git_reset_hard(framework_root: Path, sha: str) -> tuple[bool, str]:
    """Revert ``framework_root`` to ``sha``: ``git reset --hard <sha>``
    followed by ``git clean -fd`` to also discard untracked files added
    by the candidate (e.g. patches that create new files). Used as the
    REVERT path so a failed candidate cannot leak partial state into
    the next candidate's baseline."""
    cmd = ["git", "-C", str(framework_root), "reset", "--hard", sha]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"git reset --hard spawn failed: {exc!r}"
    if cp.returncode != 0:
        return False, cp.stderr.strip()
    clean_cmd = ["git", "-C", str(framework_root), "clean", "-fd"]
    try:
        cp2 = subprocess.run(
            clean_cmd, capture_output=True, text=True, timeout=60.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"git clean -fd spawn failed: {exc!r}"
    if cp2.returncode != 0:
        return False, (cp2.stderr or "").strip()
    return True, ""


def _git_commit_keep(
    framework_root: Path, message: str,
) -> tuple[str | None, str]:
    """``git add -A && git commit -m <message>`` with hyperloom identity,
    then return the new HEAD sha. ``add -A`` (not ``commit -am``) is used
    so a PR that adds *new* files is captured in the KEEP commit — ``-am``
    only stages already-tracked modifications, which would either leave a
    new file untracked in the worktree (polluting the next candidate's
    baseline) or fail outright for an add-only PR. Identity is forced via
    ``-c`` so callers don't need to depend on whatever user.email is
    configured in the framework_root git repo (Magpie clones may not have
    one)."""
    add = ["git", "-C", str(framework_root), "add", "-A"]
    try:
        cp_add = subprocess.run(
            add, capture_output=True, text=True, timeout=60.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git add -A spawn failed: {exc!r}"
    if cp_add.returncode != 0:
        return None, cp_add.stderr.strip()
    cmd = [
        "git",
        "-c", "user.email=framework-pr@hyperloom.local",
        "-c", "user.name=hyperloom framework_pr",
        "-C", str(framework_root),
        "commit", "-m", message,
    ]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git commit spawn failed: {exc!r}"
    if cp.returncode != 0:
        return None, cp.stderr.strip()
    new_sha, err = _git_head_sha(framework_root)
    if new_sha is None:
        return None, err or "commit succeeded but HEAD unreadable"
    return new_sha, ""


def _candidate_slug(candidate: dict[str, Any]) -> str:
    """Short, filesystem-safe identifier for the candidate (for variant
    names + workspace paths). Prefer ``repo/pr_number`` when present."""
    repo = str(candidate.get("repo") or "").replace("/", "-")
    pr = candidate.get("pr_number")
    if repo and pr not in (None, "", 0):
        return f"{repo}-pr-{pr}"
    ref = str(candidate.get("ref") or "").replace(":", "-")
    if repo and ref:
        return f"{repo}-{ref}"
    return repo or ref or "candidate"


def _fetch_diff_to_path(
    diff_url: str, dest: Path, *, timeout_sec: float,
) -> tuple[bool, str]:
    """Curl ``diff_url`` into ``dest`` (a .patch file path). Returns
    ``(ok, stderr)``. Uses curl rather than aiohttp because the
    integrate_patch path is also subprocess-based and we want consistent
    behaviour for restricted-network sessions (curl honours the same
    HTTPS_PROXY plumbing as the rest of the runtime)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fsSL", "--retry", "2", "--max-time",
        str(int(timeout_sec)), "-o", str(dest), diff_url,
    ]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec + 5.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"curl spawn / timeout: {exc!r}"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    if not dest.exists() or dest.stat().st_size == 0:
        return False, "curl wrote empty / missing file"
    return True, ""


def _run_git(
    args: list[str], *, timeout: float = 120.0,
) -> tuple[bool, str, str]:
    """Run ``git <args>`` capturing output. Returns ``(ok, stdout, stderr)``.
    Never raises (spawn / timeout failures map to ``(False, "", reason)``)."""
    try:
        cp = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, "", f"git spawn/timeout failed: {exc!r}"
    if cp.returncode != 0:
        return False, cp.stdout or "", (cp.stderr or "").strip()
    return True, cp.stdout or "", cp.stderr or ""


def _normalize_repo_id(url_or_slug: str) -> str:
    """Reduce a repo URL or ``owner/name`` slug to a canonical
    ``owner/name`` lowercase token for same-repo comparison. Tolerates
    ``https://github.com/Owner/Name.git``, ``git@github.com:Owner/Name``,
    and a bare ``Owner/Name``."""
    s = (url_or_slug or "").strip().lower()
    if not s:
        return ""
    s = s.rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    # Strip scheme / host so only the trailing owner/name remains.
    for sep in ("github.com/", "github.com:"):
        if sep in s:
            s = s.split(sep, 1)[1]
            break
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return s


def _candidate_is_same_repo(
    candidate: dict[str, Any], framework_root: Path,
) -> bool:
    """True unless we can POSITIVELY prove the candidate lives in a
    different repo than the live framework_root's origin (in which case
    checkout-head's ``git fetch origin`` would resolve the wrong ref).

    Fails OPEN (returns True) whenever the comparison is inconclusive:
    no candidate repo, unreadable origin, or an origin that is not a
    GitHub-style ``owner/name`` URL (e.g. a local-path clone). The guard
    only fires when BOTH sides yield an ``owner/name`` token and they
    differ — so a normal single-repo session is never downgraded."""
    cand_repo = _normalize_repo_id(
        str(candidate.get("repo") or candidate.get("discovered_repo_url") or "")
    )
    # A candidate repo token without an owner/name shape can't be compared.
    if not cand_repo or "/" not in cand_repo:
        return True
    ok, out, _err = _run_git(
        ["-C", str(framework_root), "remote", "get-url", "origin"],
        timeout=30.0,
    )
    if not ok or not out.strip():
        return True
    origin_raw = out.strip()
    # Only a GitHub-style origin gives an owner/name token directly
    # comparable to the candidate's ``repo`` slug. A local-path / mirror
    # / non-GitHub origin is inconclusive → fail open (don't downgrade a
    # legitimate same-repo session whose clone uses a local path).
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

    Fetches the candidate PR's head into ``framework_root`` (which must be
    a git repo with a fetchable origin), checks it out into an isolated
    ``git worktree`` so the live tree's already-KEPT stack is never
    disturbed, computes the PR's *net* diff against its merge-base, and
    writes that unified diff to ``dest``. The caller then ``git apply``s
    ``dest`` onto the live tree and benches it exactly like the
    ``diff_url`` path — so this mode only changes *where the patch text
    comes from*, not how it is applied or measured.

    The worktree is always removed in ``finally``. Returns ``(ok, err)``.

    Resolution of the head ref, in order:
      1. ``candidate.head_sha`` (explicit sha from discovery).
      2. ``candidate.ref`` (e.g. a branch / tag the origin already has).
      3. ``refs/pull/<pr_number>/head`` (GitHub PR head ref).
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

    # Fetch the head (skip when we already have an explicit sha that the
    # local repo can resolve, but try a fetch anyway for completeness).
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
                ["-C", root, "rev-parse", "FETCH_HEAD"], timeout=30.0,
            )
            if not ok2 or not out2.strip():
                return False, f"could not resolve FETCH_HEAD: {err2}"
            head_sha = out2.strip()
    if not head_sha:
        return False, (
            "checkout-head: no head_sha / ref / pr_number on candidate; "
            "cannot resolve PR head"
        )

    # Isolated worktree at the fetched head.
    wt_dir = dest.parent / f"wt-{_candidate_slug(candidate)}"
    # Clean any stale worktree dir from a prior crashed run.
    _run_git(["-C", root, "worktree", "remove", "--force", str(wt_dir)],
             timeout=60.0)
    ok, _out, err = _run_git(
        ["-C", root, "worktree", "add", "--detach", str(wt_dir), head_sha],
        timeout=timeout_sec,
    )
    if not ok:
        return False, f"git worktree add failed: {err}"
    try:
        # Merge-base against the live tree's current HEAD gives the PR's
        # net change relative to the branch point, so applying it onto the
        # live tree introduces only the PR's own commits (not the entire
        # divergence between head and the live HEAD).
        ok_hb, base_out, _e = _run_git(
            ["-C", root, "rev-parse", "HEAD"], timeout=30.0,
        )
        live_head = base_out.strip() if ok_hb else ""
        merge_base = ""
        if live_head:
            ok_mb, mb_out, _mb_e = _run_git(
                ["-C", root, "merge-base", live_head, head_sha], timeout=60.0,
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


class FrameworkPrExecutor:
    """ActionRunner for the ``framework_pr`` action (FRAMEWORK_PR phase)."""

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
        diff_fetch_timeout_sec: float = DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
    ):
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)
        self.diff_fetch_timeout_sec = float(diff_fetch_timeout_sec)

    async def __call__(self, ctx) -> dict[str, Any]:
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}
        candidate = params.get("candidate") or {}
        if not isinstance(candidate, dict) or not candidate:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "framework_pr requires params.candidate (the PR metadata "
                    "row produced by `fa phase-discover`)"
                ),
            }
        batch_id = str(params.get("batch_id") or "")
        slug = _candidate_slug(candidate)

        # Per-task workspace under runs/framework_pr/<task_id>/.
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "framework_pr", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        framework_root = _resolve_framework_root(
            params.get("framework_source_root") or None,
        )
        if framework_root is None:
            return {
                "status": "apply_failed",
                "error_class": "no_framework_root",
                "error": (
                    "no framework_source_root resolved; cannot apply "
                    "candidate PR. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                ),
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "workspace": str(output_root),
            }

        # Resolve patch sources. Three modes, in priority order:
        #   1. Explicit ``params.patches`` paths (test / smoke).
        #   2. checkout-head (diff source): extract the PR's net diff from
        #      an isolated worktree at the PR head. Selected when the
        #      candidate / params request it, or as a fallback when no
        #      diff_url is present.
        #   3. diff_url (default): curl the unified diff GitHub serves.
        # Modes 2 & 3 both produce a .patch that is applied + benched on
        # the live tree identically — see Stage 1 below.
        explicit_patches = params.get("patches") or None
        patch_paths: list[Path] = []
        patch_source_mode = ""
        if isinstance(explicit_patches, list) and explicit_patches:
            patch_source_mode = "explicit"
            for p in explicit_patches:
                pp = Path(str(p))
                if pp.exists():
                    patch_paths.append(pp.resolve())
                else:
                    log.warning(
                        "framework_pr: explicit patch %r not found", p,
                    )
            # Refuse to run the benchmark on an unpatched tree: if every
            # explicit patch path was missing, downstream measurements
            # would silently reflect the un-modified framework_root.
            if not patch_paths:
                return {
                    "status": "no_patch",
                    "error_class": "explicit_patches_missing",
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "reason": (
                        "all explicit patches were missing from disk; "
                        "refusing to benchmark unpatched tree"
                    ),
                    "missing_patches": [str(p) for p in explicit_patches],
                    "workspace": str(output_root),
                }
        else:
            diff_url = str(candidate.get("diff_url") or "").strip()
            apply_mode = str(
                params.get("apply_mode")
                or candidate.get("apply_mode")
                or "",
            ).strip().lower()
            prefer_checkout = bool(
                params.get("prefer_checkout")
                or candidate.get("prefer_checkout")
            )
            # A candidate is checkout-headable only if it carries a head
            # ref the worktree helper can resolve (head_sha / ref /
            # pr_number).
            has_checkout_ref = bool(
                str(candidate.get("head_sha") or "").strip()
                or str(candidate.get("ref") or "").strip()
                or candidate.get("pr_number") not in (None, "", 0)
            )
            explicit_checkout = (
                apply_mode in {"checkout_head", "checkout-head", "checkout"}
                or prefer_checkout
            )
            use_checkout_head = explicit_checkout or (
                not diff_url and has_checkout_ref
            )
            # Same-repo guard: checkout-head fetches the candidate's ref
            # from the LIVE framework_root's origin, so it only works when
            # the PR lives in that same repo. A cross-repo candidate (e.g.
            # a ROCm/vllm PR while the live tree is sglang) would fetch the
            # wrong ref, so disable checkout-head and rely on diff_url.
            if use_checkout_head and not _candidate_is_same_repo(
                candidate, framework_root,
            ):
                log.info(
                    "framework_pr: candidate repo %r differs from live "
                    "framework_root origin; disabling checkout-head, "
                    "using diff_url",
                    candidate.get("repo") or candidate.get("discovered_repo_url"),
                )
                use_checkout_head = False
            if not diff_url and not use_checkout_head:
                # No served diff and nothing to check out → genuine
                # no-patch (preserves the pre-checkout-head contract).
                return {
                    "status": "no_patch",
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "reason": (
                        "candidate carries no diff_url, no explicit "
                        "patches, and no head ref to check out"
                    ),
                    "workspace": str(output_root),
                }
            dest = output_root / f"{slug}.patch"
            if use_checkout_head:
                patch_source_mode = "checkout_head"
                ok, err = _materialize_pr_diff_via_worktree(
                    framework_root, candidate, dest,
                    timeout_sec=self.diff_fetch_timeout_sec * 4.0,
                )
                if not ok and diff_url:
                    # Fall back to the served diff_url so a worktree /
                    # fetch hiccup does not strand an otherwise-applyable
                    # candidate.
                    log.warning(
                        "framework_pr: checkout-head failed (%s); "
                        "falling back to diff_url", err,
                    )
                    patch_source_mode = "diff_url_fallback"
                    ok, err = _fetch_diff_to_path(
                        diff_url, dest,
                        timeout_sec=self.diff_fetch_timeout_sec,
                    )
                if not ok:
                    return {
                        "status": "fetch_failed",
                        "error_class": "checkout_head_failed",
                        "error": err,
                        "candidate": candidate,
                        "batch_id": batch_id,
                        "patches_applied": [],
                        "patches_reverted": [],
                        "patch_source_mode": patch_source_mode,
                        "reason": f"checkout-head diff extraction failed: {err}",
                        "workspace": str(output_root),
                    }
                patch_paths.append(dest.resolve())
            else:
                patch_source_mode = "diff_url"
                ok, err = _fetch_diff_to_path(
                    diff_url, dest, timeout_sec=self.diff_fetch_timeout_sec,
                )
                if not ok:
                    return {
                        "status": "fetch_failed",
                        "error_class": "diff_fetch_failed",
                        "error": err,
                        "candidate": candidate,
                        "batch_id": batch_id,
                        "patches_applied": [],
                        "patches_reverted": [],
                        "patch_source_mode": patch_source_mode,
                        "reason": f"failed to fetch {diff_url!r}: {err}",
                        "workspace": str(output_root),
                    }
                patch_paths.append(dest.resolve())

        # Capture HEAD before any apply so REVERT/REJECT can reset back
        # cleanly. Previously-KEPT candidates in this phase are committed
        # (see below), so they live in history past this sha and survive
        # a reset.
        pre_apply_sha, sha_err = _git_head_sha(framework_root)
        if pre_apply_sha is None:
            return {
                "status": "apply_failed",
                "error_class": "no_pre_apply_sha",
                "error": (
                    f"could not capture HEAD sha in {framework_root}: "
                    f"{sha_err or 'unknown'}"
                ),
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "workspace": str(output_root),
            }

        # Stage 1: apply patches (with -3 fallback like integrate_patch).
        applied: list[Path] = []
        apply_errors: list[dict[str, str]] = []
        for patch in patch_paths:
            ok, err = _git_apply(framework_root, patch, three_way=False)
            if not ok:
                ok2, err2 = _git_apply(framework_root, patch, three_way=True)
                if not ok2:
                    apply_errors.append({
                        "patch": str(patch),
                        "stderr": err + " | -3 retry: " + err2,
                    })
                    break
            applied.append(patch)
        if apply_errors:
            reverted = self._revert_patches(
                framework_root, applied, pre_apply_sha=pre_apply_sha,
            )
            return {
                "status": "apply_failed",
                "error_class": "git_apply_failed",
                "error": apply_errors,
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "patch_source_mode": patch_source_mode,
                "reason": "git apply failed (see error)",
                "workspace": str(output_root),
            }

        if params.get("apply_only"):
            return {
                "status": "applied_no_bench",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "patch_source_mode": patch_source_mode,
                "reason": "apply_only=True; benchmark skipped",
                "workspace": str(output_root),
            }

        # Stage 2: bench via run_grid (size=1).
        try:
            bench_result, gate_evidence = await self._bench_candidate(
                params=params,
                output_root=output_root,
                slug=slug,
            )
        except Exception as exc:  # noqa: BLE001
            reverted = self._revert_patches(
                framework_root, applied, pre_apply_sha=pre_apply_sha,
            )
            return {
                "status": "reverted",
                "error_class": "bench_exception",
                "error": repr(exc),
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "reason": f"bench raised: {exc!r}",
                "workspace": str(output_root),
            }

        # Stage 3: KEEP / REVERT.
        base_tput = float(params.get("base_tput") or 0.0)
        if base_tput <= 0:
            ss = extra.get("shared_state") or extra.get("state")
            if ss is not None:
                base_tput = float(getattr(ss, "baseline_tput", 0.0) or 0.0)
        keep_threshold_pct = float(
            params.get("keep_threshold_pct", self.keep_threshold_pct),
        )
        new_tput = bench_result.get("output_throughput")
        delta_pct: float | None = None
        if (
            isinstance(new_tput, (int, float)) and new_tput > 0
            and base_tput > 0
        ):
            delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0

        accuracy_pass = gate_evidence.get("accuracy_pass")
        gate_pass = (
            delta_pct is not None
            and delta_pct >= keep_threshold_pct
            and (accuracy_pass is None or accuracy_pass)
        )

        if not gate_pass:
            reverted = self._revert_patches(
                framework_root, applied, pre_apply_sha=pre_apply_sha,
            )
            reasons: list[str] = []
            if delta_pct is None:
                reasons.append("no measurable throughput")
            elif delta_pct < keep_threshold_pct:
                reasons.append(
                    f"throughput delta {delta_pct:+.2f}% < keep_threshold "
                    f"{keep_threshold_pct:.2f}%"
                )
            if accuracy_pass is False:
                reasons.append("accuracy regression detected")
            return {
                "status": "reverted",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "patch_source_mode": patch_source_mode,
                "reason": "; ".join(reasons) or "gate failed",
                "bench_result": bench_result,
                "workspace": str(output_root),
            }

        # KEEP: commit the applied patches in framework_root so they
        # survive a subsequent candidate's REJECT (which resets to its
        # own pre_apply_sha — that sha already includes this commit).
        keep_message = f"framework_pr KEEP {slug}"
        keep_sha, commit_err = _git_commit_keep(framework_root, keep_message)
        if keep_sha is None:
            # Commit failed — surface as an apply_failed result and reset
            # to pre_apply_sha so we don't leave uncommitted changes that
            # the next candidate would see as "dirty baseline".
            reverted = self._revert_patches(
                framework_root, applied, pre_apply_sha=pre_apply_sha,
            )
            return {
                "status": "apply_failed",
                "error_class": "keep_commit_failed",
                "error": commit_err or "git commit returned no sha",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "reason": f"KEEP commit failed: {commit_err}",
                "bench_result": bench_result,
                "workspace": str(output_root),
            }

        return {
            "status": "kept",
            "candidate": candidate,
            "batch_id": batch_id,
            "patches_applied": [str(p) for p in applied],
            "patches_reverted": [],
            "output_throughput": new_tput,
            "delta_pct": delta_pct,
            "accuracy_pass": accuracy_pass,
            "base_tput": base_tput,
            "keep_threshold_pct": keep_threshold_pct,
            "keep_commit_sha": keep_sha,
            "patch_source_mode": patch_source_mode,
            "reason": (
                f"throughput delta {delta_pct:+.2f}% >= "
                f"{keep_threshold_pct:.2f}%"
            ),
            "bench_result": bench_result,
            "workspace": str(output_root),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _revert_patches(
        self,
        framework_root: Path | None,
        applied: list[Path],
        *,
        pre_apply_sha: str,
    ) -> list[Path]:
        """Roll back the current candidate's changes.

        FRAMEWORK_PR processes candidates serially in the same
        framework_root, and previously-KEPT candidates have been
        committed (see ``_git_commit_keep``). So REVERT only needs to
        discard what *this* candidate added on top of ``pre_apply_sha``
        — ``git reset --hard <pre_apply_sha>`` does exactly that without
        touching the kept history. This replaces the older reverse-apply
        + ``git checkout -- .`` fallback, which could clobber uncommitted
        changes from prior candidates if a future caller ever omitted
        the per-KEEP commit.

        Returns the list of patches reverted (currently the full
        ``applied`` list when the reset succeeds, empty otherwise) for
        downstream telemetry / result schema compat.
        """
        if framework_root is None or not applied:
            return []
        ok, err = _git_reset_hard(framework_root, pre_apply_sha)
        if not ok:
            log.error(
                "framework_pr: git reset --hard %s failed in %s: %s",
                pre_apply_sha, framework_root, err,
            )
            return []
        return list(applied)

    async def _bench_candidate(
        self, *,
        params: dict[str, Any],
        output_root: Path,
        slug: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server +
        evaluate the accuracy gate. Mirrors
        :meth:`IntegratePatchExecutor._bench_patch` so the gain
        bookkeeping is identical across the two integration channels."""
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or default_baseline_config()
        )
        if not config_path.exists():
            raise RuntimeError(
                f"framework_pr bench: config not found at {config_path}"
            )
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        override_script = sanitize_script_name(params.get("benchmark_script"))
        override_result_dir = sanitize_result_dir(params.get("result_dir"))
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            out_name="framework_pr.with_envs.yaml",
        )

        variant = GridVariant(
            name=f"framework-pr-{slug}"[:96],
            extra_server_args=str(params.get("base_extra_args") or "").strip(),
            extra_envs={},
            note=f"framework_pr:{slug}",
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
                "result_dir": str(getattr(r, "result_dir", "")),
                "error": getattr(r, "error", "") or "",
                "nonfatal_warnings": list(
                    getattr(r, "nonfatal_warnings", []) or []
                ),
            }

        accuracy_pass: bool | None = None
        baseline_accuracy = params.get("accuracy_baseline")
        if (
            bench.get("status") == "succeeded"
            and isinstance(baseline_accuracy, (int, float))
            and float(baseline_accuracy) > 0
        ):
            try:
                eval_results = parse_eval_results(bench["result_dir"])
                if eval_results.get("score") is not None:
                    accuracy_pass = accuracy_passed(
                        eval_results["score"], float(baseline_accuracy),
                    )
            except Exception:  # noqa: BLE001
                log.exception(
                    "framework_pr: accuracy gate parse failed; "
                    "treating as None (gate skipped)"
                )

        return bench, {"accuracy_pass": accuracy_pass}


framework_pr_executor = FrameworkPrExecutor


__all__ = [
    "DEFAULT_DIFF_FETCH_TIMEOUT_SEC",
    "FrameworkPrExecutor",
    "framework_pr_executor",
]
