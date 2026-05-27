"""Coordinator-side thin client for ``fa candidates`` (framework-agent
PR enumeration).

Used by :meth:`Coordinator._warm_specialist_params` to pre-fetch PR
candidates **before** dispatching a ``serving_specialist`` with
``sub_kind='framework_pr_scout'``. The specialist subprocess then
fetches the actual diff bodies itself (via ``curl`` against the
``diff_url``), keeping the main-process call to ~hundreds of ms with
no network I/O happening inside the LLM loop.

Failure modes (binary missing / non-zero exit / timeout / JSON
parse error) all degrade gracefully to ``[]`` so the specialist
dispatch never blocks on the framework_agent CLI's health.

Design contract:

* **No diff bodies inline** — only PR metadata (repo / number / ref /
  title / summary / score / labels / html_url). The specialist fetches
  diff bodies on demand via ``curl``.
* **Best-effort** — log + return [] on any error; never raise out of
  ``fetch_pr_candidates``.
* **Pure async wrapper** around the sync ``subprocess.run`` call;
  Coordinator's reactor loop must not block on the CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


DEFAULT_FA_CANDIDATES_TIMEOUT_SEC: float = 15.0
DEFAULT_MAX_CANDIDATES: int = 20

# Repo URL table — keyed by framework name. Mirrors
# ``specialist_domains.SpecialistDomain.pr_repos`` for serving_specialist
# but flattens it to a single repo URL the ``fa candidates`` CLI needs
# (the CLI accepts only one ``repo_url`` per ExploreRequest).
_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm":   "https://github.com/ROCm/vllm.git",
}


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``.

    Returns an empty string for unknown frameworks; the caller is
    expected to bail out / log when this happens.
    """
    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


def _resolve_fa_binary() -> str | None:
    """Return the absolute path to the ``fa`` binary, or ``None`` if
    it cannot be located.

    Resolution order:

    1. ``$FA_BIN`` env var (operator pin / unit-test injection).
    2. ``shutil.which('fa')`` — the production path after
       ``framework-agent/scripts/install.sh`` ran.
    3. ``$FRAMEWORK_AGENT_ROOT/scripts/fa`` — repo-local fallback for
       sessions that source the repo's runtime env but don't have the
       global ``fa`` symlink on $PATH yet.
    """
    explicit = (os.environ.get("FA_BIN") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    via_path = shutil.which("fa")
    if via_path:
        return via_path
    fa_root = (os.environ.get("FRAMEWORK_AGENT_ROOT") or "").strip()
    if fa_root:
        candidate = Path(fa_root) / "scripts" / "fa"
        if candidate.exists():
            return str(candidate)
    return None


def _build_explore_request(
    *,
    framework: str,
    repo_url: str,
    gap_description: str,
    max_candidates: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Build the minimal ExploreRequest payload ``fa candidates``
    accepts.

    The candidates subcommand only reads ``framework / repo_url /
    gap_description / search_modes / max_search_candidates / baseline``
    (the baseline block is a hard requirement at parse time but its
    values don't matter for the candidates path). Everything else
    keeps its dataclass default.
    """
    return {
        "framework": (framework or "sglang").strip().lower(),
        "repo_url": repo_url,
        "gap_description": gap_description,
        "search_modes": ["primus_cortex", "github"],
        "max_search_candidates": int(max_candidates),
        # ``Baseline.from_dict`` requires ``throughput > 0``. Stub a
        # placeholder so the request parses; the candidates subcommand
        # never reads this value.
        "baseline": {"throughput": 1.0},
        "work_dir": str(work_dir),
    }


def _candidate_to_dict(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalise one ``Candidate`` JSON row into the
    specialist-prompt-friendly shape.

    Inputs come from ``asdict(Candidate)`` in
    ``framework_agent.runtime.cli._cmd_candidates`` so the keys mirror
    :class:`framework_agent.models.Candidate` (``ref / repo / source /
    head_sha / title / labels / author / changed_files / updated_at /
    html_url / score``).

    Output shape (extra-conservative; only fields the prompt section
    actually renders):

        {
          "repo":       str,
          "pr_number":  int | "",
          "ref":        str,
          "title":      str,
          "summary":    str,    # short labels-derived blurb (Candidate
                                #   carries labels but no summary; we
                                #   join labels for now).
          "score":      float,
          "diff_url":   str,    # constructed from html_url / repo+ref.
          "source_url": str,    # html_url verbatim.
          "labels":     list[str],
          "author":     str,
        }
    """
    repo = str(entry.get("repo") or "")
    ref = str(entry.get("ref") or "")
    # Extract pr_number from ``PR:<n>`` style refs (mirrors
    # ``Candidate.pr_number``).
    pr_number: int | str = ""
    if ref.startswith("PR:"):
        try:
            pr_number = int(ref.split(":", 1)[1])
        except (ValueError, IndexError):
            pr_number = ""
    labels_raw = entry.get("labels") or []
    if isinstance(labels_raw, (list, tuple)):
        labels = [str(l) for l in labels_raw if l]
    else:
        labels = []
    html_url = str(entry.get("html_url") or "")
    # Construct diff_url when we have a github-style html_url for a PR.
    diff_url = ""
    if html_url and isinstance(pr_number, int):
        # html_url is typically https://github.com/<repo>/pull/<n>
        # → .diff is just append ``.diff``.
        diff_url = f"{html_url}.diff"
    elif repo and isinstance(pr_number, int):
        diff_url = f"https://github.com/{repo}/pull/{pr_number}.diff"
    score_raw = entry.get("score")
    try:
        score = float(score_raw) if score_raw is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    # ``Candidate`` carries no free-form summary; approximate by joining
    # labels (so the prompt's table column isn't perpetually empty).
    summary = ", ".join(labels) if labels else ""
    return {
        "repo": repo,
        "pr_number": pr_number,
        "ref": ref,
        "title": str(entry.get("title") or ""),
        "summary": summary,
        "score": score,
        "diff_url": diff_url,
        "source_url": html_url,
        "labels": labels,
        "author": str(entry.get("author") or ""),
    }


def _run_fa_candidates_sync(
    fa_bin: str,
    request_path: Path,
    timeout_sec: float,
) -> "tuple[int, str, str]":
    """Sync helper: run ``fa candidates`` and return (rc, stdout, stderr).

    Pulled into its own function so ``asyncio.to_thread`` has a simple
    callable to hand off. Never raises — converts FileNotFoundError /
    TimeoutExpired into a non-zero rc + descriptive stderr.
    """
    cmd = [fa_bin, "candidates", "--request", str(request_path), "--out", "-"]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, "", f"fa binary not found: {exc!r}"
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"fa candidates timed out after {timeout_sec}s: {exc!r}"
    return cp.returncode, cp.stdout, cp.stderr


async def fetch_pr_candidates(
    *,
    gap_description: str,
    framework: str,
    repo_url: str = "",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    timeout_sec: float = DEFAULT_FA_CANDIDATES_TIMEOUT_SEC,
    session_dir: Path,
) -> list[dict[str, Any]]:
    """Run ``fa candidates`` in the main process and return a list of
    PR-candidate metadata dicts (no diff bodies).

    Args:
      gap_description: free-form bottleneck description used by the
        framework_agent's source ranking. Pass the specialist gap's
        ``gap_symptom / gap_layer / gap_canonical_id`` joined; the
        framework_agent extracts keywords from it.
      framework: ``sglang`` or ``vllm``. Drives the repo URL lookup
        when ``repo_url`` is empty.
      repo_url: explicit GitHub repo URL. When empty, falls back to
        :func:`repo_url_for_framework`.
      max_candidates: cap on the result list (default 20).
      timeout_sec: hard wall-clock cap on the subprocess call.
      session_dir: used for the temporary request JSON location
        (``<session_dir>/.fa-tmp/<uuid>.json``).

    Returns:
      A list of metadata dicts (shape documented on
      :func:`_candidate_to_dict`), capped at ``max_candidates``.
      Returns ``[]`` on any failure path so the caller can dispatch the
      specialist with an empty PR feed instead of crashing.
    """
    fa_bin = _resolve_fa_binary()
    if not fa_bin:
        log.warning(
            "framework_pr_scout: `fa` binary not found "
            "(checked $FA_BIN, $PATH, $FRAMEWORK_AGENT_ROOT/scripts/fa); "
            "returning empty candidate list. Run "
            "framework-agent/scripts/install.sh to provision it."
        )
        return []

    resolved_repo_url = (repo_url or repo_url_for_framework(framework)).strip()
    if not resolved_repo_url:
        log.warning(
            "framework_pr_scout: no repo_url for framework=%r; "
            "returning empty candidate list", framework,
        )
        return []

    tmp_dir = session_dir / ".fa-tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("framework_pr_scout: cannot create %s: %r", tmp_dir, exc)
        return []

    request = _build_explore_request(
        framework=framework or "sglang",
        repo_url=resolved_repo_url,
        gap_description=gap_description or "",
        max_candidates=max_candidates,
        work_dir=tmp_dir,
    )
    request_path = tmp_dir / f"request-{uuid.uuid4().hex[:12]}.json"
    try:
        request_path.write_text(
            json.dumps(request, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "framework_pr_scout: cannot write %s: %r", request_path, exc,
        )
        return []

    try:
        rc, stdout, stderr = await asyncio.to_thread(
            _run_fa_candidates_sync, fa_bin, request_path, timeout_sec,
        )
    finally:
        with contextlib.suppress(OSError):
            request_path.unlink()

    if rc != 0:
        log.warning(
            "framework_pr_scout: `fa candidates` exited rc=%d; stderr=%r",
            rc, (stderr or "")[-512:],
        )
        return []

    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning(
            "framework_pr_scout: cannot parse `fa candidates` output: %r "
            "(first 200 chars: %r)",
            exc, (stdout or "")[:200],
        )
        return []

    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(raw_candidates, list):
        log.warning(
            "framework_pr_scout: `fa candidates` output has no "
            "candidates list (got keys=%r)",
            sorted(payload.keys()) if isinstance(payload, dict) else "?",
        )
        return []

    out: list[dict[str, Any]] = []
    for entry in raw_candidates:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(_candidate_to_dict(entry))
        except Exception:  # noqa: BLE001 — never raise out of warm
            log.exception(
                "framework_pr_scout: failed to normalise candidate %r",
                entry,
            )
    if len(out) > max_candidates:
        out = out[:max_candidates]
    log.info(
        "framework_pr_scout: pre-fetched %d PR candidates "
        "(framework=%s repo=%s)",
        len(out), framework, resolved_repo_url,
    )
    return out


DEFAULT_FA_PHASE_TIMEOUT_SEC: float = 60.0


def _run_fa_subcommand_sync(
    fa_bin: str,
    subcommand: str,
    request_path: Path,
    timeout_sec: float,
) -> "tuple[int, str, str]":
    """Sync helper: run ``fa <subcommand> --request <path> --out -``.

    Mirrors :func:`_run_fa_candidates_sync` for the FRAMEWORK_PR phase
    subcommands (``phase-discover`` / ``phase-fetch`` /
    ``phase-emit-proposal``). Never raises.
    """
    cmd = [fa_bin, subcommand, "--request", str(request_path), "--out", "-"]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, "", f"fa binary not found: {exc!r}"
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"fa {subcommand} timed out after {timeout_sec}s: {exc!r}"
    return cp.returncode, cp.stdout, cp.stderr


async def _invoke_fa_phase(
    *,
    subcommand: str,
    request: dict[str, Any],
    session_dir: Path,
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Generic async runner for ``fa phase-*`` subcommands.

    Writes ``request`` as a temp JSON, runs the subcommand, returns the
    parsed JSON output. Raises :class:`RuntimeError` on binary missing
    / non-zero exit / JSON parse failure so callers can degrade.
    """
    fa_bin = _resolve_fa_binary()
    if not fa_bin:
        raise RuntimeError(
            f"fa binary not found (subcommand={subcommand!r}); "
            "checked $FA_BIN, $PATH, $FRAMEWORK_AGENT_ROOT/scripts/fa"
        )
    tmp_dir = session_dir / ".fa-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    request_path = tmp_dir / f"phase-{subcommand}-{uuid.uuid4().hex[:12]}.json"
    request_path.write_text(
        json.dumps(request, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    try:
        rc, stdout, stderr = await asyncio.to_thread(
            _run_fa_subcommand_sync,
            fa_bin, subcommand, request_path, timeout_sec,
        )
    finally:
        with contextlib.suppress(OSError):
            request_path.unlink()
    if rc != 0:
        raise RuntimeError(
            f"fa {subcommand} exited rc={rc}; stderr={(stderr or '')[-512:]!r}"
        )
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"fa {subcommand} produced invalid JSON: {exc!r}; "
            f"first 200 chars={ (stdout or '')[:200]!r}"
        )


async def phase_discover(
    *,
    model: str,
    framework: str,
    gpu_type: str,
    gaps: list[dict[str, str]],
    session_dir: Path,
    repo_url: str = "",
    max_candidates: int = 5,
    batch_id: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK_PR-phase batch discovery shim.

    Returns the parsed payload from ``fa phase-discover``:
    ``{batch_id, framework, repo_url, candidates: [...]}``.
    """
    resolved_repo_url = (repo_url or repo_url_for_framework(framework)).strip()
    request = {
        "model":     model,
        "framework": (framework or "sglang").strip().lower(),
        "gpu_type":  gpu_type,
        "gaps":      gaps,
        "repo_url":  resolved_repo_url,
        "work_dir":  str(session_dir / ".fa-tmp" / "phase-discover"),
        "max_search_candidates": int(max_candidates),
        "batch_id":  batch_id,
    }
    return await _invoke_fa_phase(
        subcommand="phase-discover",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


async def phase_fetch(
    *,
    pr_url: str,
    repo: str,
    ref: str,
    framework: str,
    worktree_dir: Path,
    session_dir: Path,
    repo_url: str = "",
    title: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK_PR-phase candidate-fetch shim.

    Returns ``{status, worktree_path, applied_files, message}``.
    """
    request = {
        "pr_url":       pr_url,
        "repo":         repo,
        "ref":          ref,
        "framework":    (framework or "sglang").strip().lower(),
        "repo_url":     repo_url or repo_url_for_framework(framework),
        "worktree_dir": str(worktree_dir),
        "title":        title,
    }
    return await _invoke_fa_phase(
        subcommand="phase-fetch",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


async def phase_emit_proposal(
    *,
    task_id: str,
    pr_url: str,
    worktree_path: str,
    gap_canonical_id: str,
    framework: str,
    session_dir: Path,
    patches_written: list[str] | None = None,
    rationale: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK_PR-phase ``specialist_done`` envelope emitter.

    Returns the envelope dict produced by ``fa phase-emit-proposal``.
    """
    request = {
        "task_id":          task_id,
        "pr_url":           pr_url,
        "worktree_path":    worktree_path,
        "gap_canonical_id": gap_canonical_id,
        "framework":        (framework or "sglang").strip().lower(),
        "patches_written":  list(patches_written or []),
        "rationale":        rationale,
    }
    return await _invoke_fa_phase(
        subcommand="phase-emit-proposal",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


__all__ = [
    "DEFAULT_FA_CANDIDATES_TIMEOUT_SEC",
    "DEFAULT_FA_PHASE_TIMEOUT_SEC",
    "DEFAULT_MAX_CANDIDATES",
    "fetch_pr_candidates",
    "phase_discover",
    "phase_fetch",
    "phase_emit_proposal",
    "repo_url_for_framework",
]
