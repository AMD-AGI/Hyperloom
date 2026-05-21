"""Framework-Agent integration glue for inference_optimizer.

Originally a *one-shot precursor* invoked from ``cli.py`` before
``_run_optimize``; the architecture was migrated (plan
``fa-as-io-arm-design``) so that PR discovery + apply runs as a
regular bandit arm (:mod:`framework_pr` executor) instead of a
session-killing pre-stage.

This module retained two roles after the migration:

1. **Helpers** consumed by the ``framework_pr`` executor:

   * :func:`discover_pr` — run ``fa candidates`` (no side effects),
     return the top-1 candidate hand-off dict.
   * :func:`apply_to_sglang` — resolve head_sha + checkout + (opt) pip
     install.
   * :func:`rollback_to` — checkout the sglang worktree back to a
     previously stashed git ref (used on arm DISCARD path).
   * :func:`current_head_sha` — read ``HEAD`` so the executor can stash
     the pre-arm ref for rollback.
   * :func:`_resolve_head_sha` — ``git ls-remote`` fallback for PRs
     whose head_sha did not come back populated from fa.

2. **Deprecated** legacy CLI hooks consumed by ``cli.py:_run_optimize``:

   * :func:`run` / :func:`explicit_pr_apply` — invoked when the user
     still passes ``--framework-pr`` / ``--framework-pr-discover``.
     Phase 3 of the migration repointed those flags to inject
     ``framework_pr`` task params instead; the helpers stay here only
     so unit tests + any external caller still importing them keep
     working until the next release.

Error policy
------------
Helper functions raise :class:`FrameworkPRError` on any operational
failure. The new arm executor catches this and turns it into a soft
``status='failed'`` result with ``error_class`` set; **the IO session
keeps running**. Legacy CLI :func:`run` still surfaces the exception
to the operator (cli.py maps it to exit code 2) for backward compat.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_DEFAULT_SGLANG_PATH = Path("/sgl-workspace/sglang")
_DEFAULT_PRIMUS_URL = (
    "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local"
)
_DEFAULT_BASELINE_PLACEHOLDER = 1.0  # only used to satisfy fa's winner-gate input contract
_FA_EXECUTE_TIMEOUT_SEC = 600
_GIT_FETCH_TIMEOUT_SEC = 600
_PIP_INSTALL_TIMEOUT_SEC = 1800


class FrameworkPRError(RuntimeError):
    """Raised when the framework-pr pre-stage cannot complete.

    Inherits from :class:`RuntimeError` so callers that wrap in a
    blanket ``except Exception`` (e.g. the CLI top-level) translate
    it into exit code 2 with the original message preserved.
    """


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_fa_binary() -> str:
    """Return the absolute path of the ``fa`` CLI, raising on missing.

    Looks up ``fa`` on PATH first; falls back to the sglang plugin
    venv location ``/opt/venv/bin/fa`` which the framework-agent
    install.sh prefers when the host image ships that venv.
    """
    bin_path = shutil.which("fa")
    if bin_path:
        return bin_path
    fallback = "/opt/venv/bin/fa"
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback
    raise FrameworkPRError(
        "fa CLI not found on PATH or /opt/venv/bin/fa. Run "
        "framework-agent/scripts/install.sh before invoking "
        "--framework-pr-discover."
    )


def _parse_pr_number(ref: str) -> int:
    """Convert a ``PR:N`` ref into the integer PR number."""
    if not ref.startswith("PR:"):
        raise FrameworkPRError(
            f"unsupported ref shape {ref!r}; framework-pr only handles PR:N refs"
        )
    try:
        return int(ref.split(":", 1)[1])
    except (ValueError, IndexError) as exc:
        raise FrameworkPRError(f"could not parse PR number from ref {ref!r}: {exc}") from exc


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout_sec: int,
    env: dict[str, str] | None = None,
    label: str,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with strict timeout + log it on entry/exit."""
    log.info("framework_pr_discover: running %s", label)
    log.debug("framework_pr_discover: argv=%r cwd=%s", cmd, cwd)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameworkPRError(
            f"{label} timed out after {timeout_sec}s; consider raising "
            f"the timeout or simplifying the request"
        ) from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-1000:]
        raise FrameworkPRError(
            f"{label} exited with rc={result.returncode}; tail:\n{tail}"
        )
    log.info("framework_pr_discover: %s ok (rc=0)", label)
    return result


# ---------------------------------------------------------------------------
# fa explore driver
# ---------------------------------------------------------------------------


def _build_explore_request(
    *,
    gap_description: str,
    repo_url: str,
    primus_cortex_url: str,
    work_dir: Path,
    framework: str = "sglang",
    baseline_throughput: float = _DEFAULT_BASELINE_PLACEHOLDER,
    max_candidates: int = 1,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Materialise the ExploreRequest JSON fa expects on stdin/file.

    Notes
    -----
    * ``baseline.throughput`` is a placeholder; the IO main flow
      computes the real baseline downstream. fa only needs it to
      satisfy its own winner-gate arithmetic so the discovered PR
      is *picked* (not validated) here.
    * ``prepare_candidate_env=true`` so fa creates a worktree at the
      PR head_sha and writes pr.patches against that commit; this is
      what makes ``git apply`` / ``git checkout`` deterministic on the
      IO side.
    * No ``commands`` block: discover-mode only needs the winner ref,
      not real GPU benchmarks. The bench/accuracy gates are bypassed
      when commands are empty; fa marks the candidate ``planned`` and
      promotes it to ``winner`` only if explicit gates pass. We work
      around that by accepting *any* primus_cortex hit as winner via
      the ``max_search_candidates=1`` + ``search_perf_prs=true`` combo
      and short-circuiting in :func:`discover_pr` below.
    """
    req: dict[str, Any] = {
        "framework": framework,
        "repo_url": repo_url,
        "work_dir": str(work_dir),
        "baseline": {
            "throughput": baseline_throughput,
            "accuracy": None,
            "completed": "1/1",
        },
        "thresholds": {"min_throughput_ratio": 1.0, "max_accuracy_drop": 1.0},
        "search_perf_prs": True,
        "max_search_candidates": max_candidates,
        "primus_cortex": {
            "base_url": primus_cortex_url,
            "timeout_sec": 10.0,
        },
        "search_modes": ["primus_cortex"],
        "gap_description": gap_description,
        "prepare_candidate_env": False,
        "commands": {},
    }
    if keywords:
        # C: explicit override; fa side `_resolve_keywords` bypasses
        # extract_keywords() when this field is non-empty.
        req["keywords"] = list(keywords)
    return req


def discover_pr(
    gap_description: str,
    repo_url: str,
    primus_cortex_url: str = _DEFAULT_PRIMUS_URL,
    *,
    framework: str = "sglang",
    work_dir: Path | None = None,
    max_candidates: int = 1,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Drive ``fa explore`` (plan mode) and return the first candidate.

    Discover-mode does NOT run benchmarks - we just need a candidate
    PR to feed to IO as the new baseline. The IO main flow then
    runs its own baseline / params / backends against the patched
    sglang source, which is the real performance comparison.

    Returns
    -------
    dict with keys ``winner_ref``, ``head_sha``, ``winner_dir``,
    ``patch_path``, ``files_json_path``. ``winner_ref`` is always
    populated; head_sha may be empty when the primus payload omits it
    (caller can fall back to ``git ls-remote`` via :func:`_resolve_head_sha`).
    """
    fa_bin = _resolve_fa_binary()
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="io-framework-pr-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    req = _build_explore_request(
        gap_description=gap_description,
        repo_url=repo_url,
        primus_cortex_url=primus_cortex_url,
        work_dir=work_dir,
        framework=framework,
        max_candidates=max_candidates,
        keywords=keywords,
    )
    req_path = work_dir / "req.json"
    req_path.write_text(json.dumps(req, indent=2), encoding="utf-8")

    out_path = work_dir / "plan_summary.json"
    _run(
        [fa_bin, "explore", "--request", str(req_path), "--out", str(out_path)],
        timeout_sec=_FA_EXECUTE_TIMEOUT_SEC,
        label="fa explore (discover)",
    )

    if not out_path.is_file():
        raise FrameworkPRError(
            f"fa explore did not produce {out_path}; check logs"
        )
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    cands = summary.get("candidates") or []
    if not cands:
        raise FrameworkPRError(
            f"fa explore returned no candidates for gap={gap_description!r}; "
            "either widen the gap_description or supply --framework-pr explicitly"
        )
    first = cands[0]
    cand = first.get("candidate") or {}
    winner_ref = cand.get("ref") or ""
    if not winner_ref.startswith("PR:"):
        raise FrameworkPRError(
            f"first candidate is not a PR ref: {winner_ref!r}; framework-pr "
            "only handles PR:N refs at this time"
        )
    return {
        "winner_ref": winner_ref,
        "head_sha": cand.get("head_sha") or "",
        "winner_dir": first.get("candidate_dir") or "",
        "patch_path": first.get("patches_path") or "",
        "files_json_path": first.get("files_json_path") or "",
        "candidate": cand,
    }


def enumerate_candidates_via_fa(
    *,
    gap_description: str,
    repo_url: str,
    primus_cortex_url: str = _DEFAULT_PRIMUS_URL,
    framework: str = "sglang",
    work_dir: Path | None = None,
    max_candidates: int = 5,
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run ``fa candidates`` (read-only) and return the full ranked list.

    Side-effect free: no git fetch, no worktree, no pip install. Used by the
    :mod:`framework_pr` arm executor so a single tick can cheaply probe the
    candidate set, filter out refs already tried this session, and apply the
    survivor on the IO side via :func:`apply_to_sglang`. Returns the raw
    ``candidates`` list straight out of fa's JSON envelope; each entry is the
    asdict(Candidate) form so ``score`` / ``ref`` / ``head_sha`` / ``title``
    are all available to the executor's selection logic.

    Raises :class:`FrameworkPRError` on fa subprocess failure / empty result
    so the arm executor can treat that as a soft DISCARD with a real
    ``error_class`` and the bandit cooldown can kick in.
    """
    fa_bin = _resolve_fa_binary()
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="io-fa-cands-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    req = _build_explore_request(
        gap_description=gap_description,
        repo_url=repo_url,
        primus_cortex_url=primus_cortex_url,
        work_dir=work_dir,
        framework=framework,
        max_candidates=max_candidates,
        keywords=keywords,
    )
    req_path = work_dir / "req.json"
    req_path.write_text(json.dumps(req, indent=2), encoding="utf-8")

    out_path = work_dir / "candidates.json"
    _run(
        [fa_bin, "candidates", "--request", str(req_path), "--out", str(out_path)],
        timeout_sec=_FA_EXECUTE_TIMEOUT_SEC,
        label="fa candidates",
    )
    if not out_path.is_file():
        raise FrameworkPRError(
            f"fa candidates did not produce {out_path}; check logs"
        )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    cands = payload.get("candidates") or []
    if not isinstance(cands, list):
        raise FrameworkPRError(
            f"fa candidates returned malformed payload (candidates not a list): "
            f"{type(cands).__name__}"
        )
    return cands


# ---------------------------------------------------------------------------
# git checkout + pip install
# ---------------------------------------------------------------------------


def _resolve_head_sha(sglang_path: Path, pr_number: int) -> str:
    """Resolve a PR's head SHA via ``git ls-remote`` (fallback path).

    Used when fa's enrichment did not populate head_sha (e.g. primus
    response missing the SHA field). The remote query is cheap.
    """
    result = _run(
        ["git", "ls-remote", "origin", f"refs/pull/{pr_number}/head"],
        cwd=sglang_path,
        timeout_sec=60,
        label=f"git ls-remote refs/pull/{pr_number}/head",
    )
    out = (result.stdout or "").strip()
    if not out:
        raise FrameworkPRError(
            f"no remote ref for refs/pull/{pr_number}/head; PR may have been "
            "closed or the remote is offline"
        )
    head_sha = out.split()[0]
    if len(head_sha) < 7:
        raise FrameworkPRError(
            f"git ls-remote produced unexpected output: {out!r}"
        )
    return head_sha


def current_head_sha(sglang_path: Path | None = None) -> str:
    """Return ``git rev-parse HEAD`` for the sglang worktree.

    Used by the framework_pr arm BEFORE attempting a PR checkout so the
    DISCARD path can roll back to this exact commit. Returns "" on any
    git error (caller should treat empty as "rollback disabled / unsafe").
    """
    sglang_path = sglang_path or _DEFAULT_SGLANG_PATH
    if not (sglang_path / ".git").is_dir():
        return ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(sglang_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("current_head_sha: git rev-parse failed: %s", exc)
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def rollback_to(
    target_sha: str,
    *,
    sglang_path: Path | None = None,
) -> None:
    """``git checkout --detach <target_sha>`` to undo a prior PR apply.

    Used by the framework_pr arm DISCARD path. ``target_sha`` should
    be the value returned by :func:`current_head_sha` BEFORE the
    arm's :func:`apply_to_sglang` call. Raises :class:`FrameworkPRError`
    on git failure so the executor can record ``rollback_done=False``
    and the operator can intervene before the next arm runs against a
    dirty worktree.
    """
    sglang_path = sglang_path or _DEFAULT_SGLANG_PATH
    target_sha = (target_sha or "").strip()
    if not target_sha:
        raise FrameworkPRError("rollback_to: empty target_sha")
    if not (sglang_path / ".git").is_dir():
        raise FrameworkPRError(
            f"rollback_to: {sglang_path} is not a git checkout"
        )
    _run(
        ["git", "checkout", "--detach", target_sha],
        cwd=sglang_path,
        timeout_sec=120,
        label=f"git checkout (rollback) {target_sha[:12]}",
    )


def _worktree_is_dirty(sglang_path: Path) -> bool:
    """Return True iff ``git status --porcelain`` reports tracked-file changes.

    Untracked files alone do not block ``git checkout`` (they remain
    on disk), so we only treat **tracked** modifications/deletions as
    dirty. This matches the behaviour of plain ``git checkout`` and
    avoids needless stashes for the quant-config JSONs that the
    sandbox image typically leaves untracked.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(sglang_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return bool((result.stdout or "").strip())


def _stash_dirty(sglang_path: Path, label: str) -> bool:
    """``git stash push -u`` the worktree. Returns True iff a stash was created.

    The ``-u`` flag also stashes untracked files so subsequent
    operations see a fully clean tree; this is safe because the
    sandbox is ephemeral and we do not pop the stash back. The
    timestamp label keeps the stash log readable when debugging.
    """
    _run(
        ["git", "stash", "push", "-u", "-m", label],
        cwd=sglang_path,
        timeout_sec=120,
        label=f"git stash push -u (pre-checkout, label={label!r})",
    )
    return True


def apply_to_sglang(
    head_sha: str,
    *,
    pr_number: int | None = None,
    sglang_path: Path | None = None,
    pip_reinstall: bool = False,
    auto_stash: bool = True,
) -> Path:
    """Fetch + checkout a PR head into the sglang source dir.

    Parameters
    ----------
    head_sha:
        40-char git SHA to check out. If empty AND ``pr_number`` is
        set, we resolve it via ``git ls-remote``.
    pr_number:
        Integer PR number; used to populate ``refs/pull/N/head`` on
        ``git fetch`` so the remote serves the head ref (otherwise a
        bare SHA fetch may fail when the SHA is not on a regular
        branch).
    sglang_path:
        Defaults to ``/sgl-workspace/sglang``.
    pip_reinstall:
        Whether to ``pip install -e python/`` after checkout.
        Default **False**: in the standard sglang sandbox image the
        package is already installed in editable mode, so ``git
        checkout`` alone is enough for the new code to be picked up
        on next ``import sglang``. Set to True only when the host
        image lacks the editable install or when the PR head changes
        ``python/pyproject.toml`` in a way that requires a refresh
        (note: sglang's full build pulls in sgl-kernel which needs a
        Rust toolchain not present in standard images).
    auto_stash:
        If True (default), ``git stash push -u`` any dirty tracked
        files before checkout. The sandbox sglang worktree commonly
        ships dirty (modified pyproject.toml, additional kernel
        configs), so without this the checkout aborts with
        ``Your local changes would be overwritten``. The stash is
        intentionally *not* popped: the sandbox is ephemeral and
        restoring the dirty state on top of the PR head would
        reintroduce conflicts. Set to False to fail fast instead.

    Returns
    -------
    The resolved sglang_path, after checkout and (optionally)
    reinstall.
    """
    sglang_path = sglang_path or _DEFAULT_SGLANG_PATH
    if not (sglang_path / ".git").is_dir():
        raise FrameworkPRError(
            f"sglang_path={sglang_path} is not a git checkout; cannot apply PR"
        )

    if not head_sha and pr_number is not None:
        head_sha = _resolve_head_sha(sglang_path, pr_number)
    if not head_sha:
        raise FrameworkPRError("head_sha is empty and pr_number was not provided")

    if auto_stash and _worktree_is_dirty(sglang_path):
        import time as _time
        _stash_dirty(sglang_path, f"io-pre-stage-{int(_time.time())}")

    # Pre-fetch the PR head ref so a bare-SHA checkout succeeds even
    # when the SHA is not reachable from the default branch tip.
    if pr_number is not None:
        _run(
            [
                "git",
                "fetch",
                "origin",
                f"refs/pull/{pr_number}/head:refs/pull/{pr_number}/head",
            ],
            cwd=sglang_path,
            timeout_sec=_GIT_FETCH_TIMEOUT_SEC,
            label=f"git fetch refs/pull/{pr_number}/head",
        )

    _run(
        ["git", "checkout", "--detach", head_sha],
        cwd=sglang_path,
        timeout_sec=120,
        label=f"git checkout {head_sha[:12]}",
    )

    if pip_reinstall:
        python_pkg = sglang_path / "python"
        if not python_pkg.is_dir():
            raise FrameworkPRError(
                f"sglang_path={sglang_path}/python does not exist; cannot "
                "reinstall after checkout"
            )
        _run(
            [
                "python3",
                "-m",
                "pip",
                "install",
                "-e",
                str(python_pkg),
                "--no-deps",
                "--no-build-isolation",
                "--quiet",
            ],
            timeout_sec=_PIP_INSTALL_TIMEOUT_SEC,
            label="pip install -e python/ (sglang)",
        )

    return sglang_path


# ---------------------------------------------------------------------------
# top-level entry consumed by cli.py
# ---------------------------------------------------------------------------


def explicit_pr_apply(
    pr_ref: str,
    *,
    sglang_path: Path | None = None,
    pip_reinstall: bool = False,
    auto_stash: bool = True,
) -> dict[str, Any]:
    """Apply a user-supplied ``PR:N`` ref without running fa first.

    Symmetric return shape with :func:`discover_pr` so the CLI can
    log the same fields regardless of which path was taken.
    """
    sglang_path = sglang_path or _DEFAULT_SGLANG_PATH
    pr_number = _parse_pr_number(pr_ref)
    head_sha = _resolve_head_sha(sglang_path, pr_number)
    apply_to_sglang(
        head_sha,
        pr_number=pr_number,
        sglang_path=sglang_path,
        pip_reinstall=pip_reinstall,
        auto_stash=auto_stash,
    )
    return {
        "winner_ref": pr_ref,
        "head_sha": head_sha,
        "winner_dir": "",
        "patch_path": "",
        "files_json_path": "",
        "candidate": {"ref": pr_ref, "head_sha": head_sha, "source": "explicit"},
    }


def run(args, *, sglang_path: Path | None = None) -> dict[str, Any]:
    """Orchestrate the framework-pr pre-stage based on argparse args.

    Recognised attributes on ``args``:

    * ``framework_pr`` (str): explicit ``PR:N`` ref; bypass discover.
    * ``framework_pr_discover`` (bool): run fa discover.
    * ``framework_gap`` (str): gap_description passed to fa.
    * ``framework_repo_url`` (str): repo URL; default sglang upstream.
    * ``framework_primus_url`` (str): override primus cortex URL.

    Returns the hand-off dict (same shape as :func:`discover_pr`).
    Raises :class:`FrameworkPRError` on any failure.
    """
    explicit_ref = (getattr(args, "framework_pr", "") or "").strip()
    do_discover = bool(getattr(args, "framework_pr_discover", False))
    sglang_path = sglang_path or _DEFAULT_SGLANG_PATH

    if explicit_ref and do_discover:
        raise FrameworkPRError(
            "--framework-pr and --framework-pr-discover are mutually exclusive"
        )

    pip_reinstall = bool(getattr(args, "framework_pip_reinstall", False))
    auto_stash = not bool(getattr(args, "framework_no_auto_stash", False))

    # C: parse explicit keyword override (comma- or space-separated).
    # When non-empty, fa skips extract_keywords() and uses these verbatim.
    keywords: list[str] = []
    kw_raw = (getattr(args, "framework_keywords", "") or "").strip()
    if kw_raw:
        for token in kw_raw.replace(",", " ").split():
            t = token.strip()
            if t:
                keywords.append(t)

    if explicit_ref:
        log.info("framework_pr_discover: applying explicit ref %s", explicit_ref)
        return explicit_pr_apply(
            explicit_ref,
            sglang_path=sglang_path,
            pip_reinstall=pip_reinstall,
            auto_stash=auto_stash,
        )

    if not do_discover:
        # Caller should not have invoked us; surface a clear error
        # rather than silently no-op.
        raise FrameworkPRError(
            "neither --framework-pr nor --framework-pr-discover is set; "
            "framework_pr_discover.run() should not be called"
        )

    gap = (getattr(args, "framework_gap", "") or "").strip()
    # C: gap is optional when explicit keywords are given (keywords drive
    # the search). At least one of them must be set so we don't run an
    # empty-handed primus query.
    if not gap and not keywords:
        raise FrameworkPRError(
            "--framework-pr-discover requires --framework-gap or "
            "--framework-keywords "
            "(e.g. --framework-gap 'improve sglang fp8 MoE on MI300X' or "
            "--framework-keywords 'fp8,moe')"
        )
    repo_url = (
        getattr(args, "framework_repo_url", "")
        or "https://github.com/sgl-project/sglang.git"
    )
    primus_url = (
        getattr(args, "framework_primus_url", "") or _DEFAULT_PRIMUS_URL
    )

    handoff = discover_pr(
        gap_description=gap,
        repo_url=repo_url,
        primus_cortex_url=primus_url,
        keywords=keywords or None,
    )
    log.info(
        "framework_pr_discover: winner=%s head_sha=%s",
        handoff["winner_ref"],
        handoff["head_sha"][:12] if handoff.get("head_sha") else "(empty)",
    )
    pr_number = _parse_pr_number(handoff["winner_ref"])
    apply_to_sglang(
        handoff.get("head_sha") or "",
        pr_number=pr_number,
        sglang_path=sglang_path,
        pip_reinstall=pip_reinstall,
        auto_stash=auto_stash,
    )
    # Refresh head_sha if we filled it from ls-remote above.
    if not handoff.get("head_sha"):
        handoff["head_sha"] = _resolve_head_sha(sglang_path, pr_number)
    return handoff


__all__ = [
    "FrameworkPRError",
    "apply_to_sglang",
    "current_head_sha",
    "discover_pr",
    "enumerate_candidates_via_fa",
    "explicit_pr_apply",
    "rollback_to",
    "run",
]
