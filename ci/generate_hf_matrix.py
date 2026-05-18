#!/usr/bin/env python3
"""Generate GitHub Actions matrix JSON from HuggingFace top-N, explicit list,
or a pre-built candidates file (preferred for batch-driven dispatch).

Inputs (env vars, set by the workflow), in priority order:
  INPUT_MODELS           Space-separated HF repo IDs (overrides everything)
  INPUT_CANDIDATES_FILE  Path to JSON built by ci/build_candidates.py
                         + INPUT_BATCH_INDEX (0-based) + INPUT_BATCH_SIZE
                         takes the matching slice from candidates[].
  INPUT_HF_TOP           Live HF top-N (legacy fallback, default 5)
  INPUT_MIN_PARAMS       Filter to >= B billion params (default 7)
  HF_TOKEN               Optional, for gated repo metadata access

Output:
  GITHUB_OUTPUT receives `matrix={"include": [...]}` where each entry is
  {"model": "<repo_id>", "key": "<safe-slug>"}. The slug is used as the matrix
  display name and the per-task artifact suffix (avoids '/' in artifact names).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# Reuse HuggingFaceClient from the existing script — single source of truth
# for the pool-then-filter logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_submit import HuggingFaceClient   # noqa: E402

DEFAULT_CRON_CANDIDATES_FILE = "ci/candidates/top2000_2026-05-15.json"
LEADERBOARD_URL = (
    "https://core42.example-internal-host.invalid/model-leaderboard/api/v1/"
    "leaderboard?sort_by=gain&order=desc&limit=5000&offset=0"
)


def slugify(repo_id: str) -> str:
    """Turn 'Qwen/Qwen3-8B' into 'qwen-qwen3-8b' for use as artifact key."""
    s = repo_id.lower().replace("/", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _leaderboard_models() -> set[str]:
    try:
        with urllib.request.urlopen(LEADERBOARD_URL, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        raise RuntimeError(f"failed to query leaderboard for exclusion: {e}") from e
    rows = data.get("results") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise RuntimeError("leaderboard response did not contain a results list")
    return {
        str(item.get("model") or "").strip().lower()
        for item in rows
        if isinstance(item, dict) and item.get("model")
    }


def _active_workflow_slugs() -> set[str]:
    """Return optimize matrix slugs from queued/in-progress workflow runs.

    This prevents cron from dispatching a model that is already queued/running
    in another optimize-submit invocation but not yet visible in the
    leaderboard.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not token or not repo:
        return set()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "hyperloom-generate-hf-matrix",
    }
    slugs: set[str] = set()

    def _get(url: str) -> dict:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    for status in ("queued", "in_progress"):
        try:
            runs = _get(
                f"https://api.github.com/repos/{repo}/actions/runs"
                f"?status={status}&per_page=30"
            ).get("workflow_runs", [])
        except Exception as e:
            print(f"WARN: failed to list {status} workflow runs: {e}",
                  file=sys.stderr)
            continue
        for run in runs:
            if str(run.get("id")) == str(run_id):
                continue
            if run.get("name") != "SaFE Optimize Submit":
                continue
            jobs_url = run.get("jobs_url")
            if not jobs_url:
                continue
            try:
                jobs = _get(jobs_url + "?per_page=100").get("jobs", [])
            except Exception as e:
                print(f"WARN: failed to list jobs for run {run.get('id')}: {e}",
                      file=sys.stderr)
                continue
            for job in jobs:
                name = str(job.get("name") or "")
                if name.startswith("optimize-"):
                    slugs.add(name.removeprefix("optimize-"))
    return slugs


def _apply_exclusions(repos: list[str]) -> list[str]:
    excluded_models: set[str] = set()
    excluded_slugs: set[str] = set()
    if _truthy(os.environ.get("INPUT_EXCLUDE_LEADERBOARD")):
        excluded_models = _leaderboard_models()
        print(f"leaderboard exclusion: {len(excluded_models)} models",
              file=sys.stderr)
    if _truthy(os.environ.get("INPUT_EXCLUDE_ACTIVE_WORKFLOWS")):
        excluded_slugs = _active_workflow_slugs()
        print(f"active workflow exclusion: {len(excluded_slugs)} slugs",
              file=sys.stderr)
    if not excluded_models and not excluded_slugs:
        return repos
    out = []
    skipped = 0
    for repo in repos:
        if repo.lower() in excluded_models or slugify(repo) in excluded_slugs:
            skipped += 1
            continue
        out.append(repo)
    print(f"exclusions skipped {skipped}; returning {len(out)} repos",
          file=sys.stderr)
    return out


def _slice_from_candidates(cands_path: Path) -> list[str]:
    """Load candidates.json (built by build_candidates.py) and take a batch
    slice based on INPUT_BATCH_INDEX (0-based) + INPUT_BATCH_SIZE.

    If INPUT_BATCH_SIZE is unset or 0, returns all candidates (whole-batch
    run). The slice is ordered as the candidates list appears in the JSON
    (which is HF download rank). Use this for deterministic, reproducible
    batch dispatch.
    """
    try:
        data = json.loads(cands_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"failed to read candidates file {cands_path}: {e}",
              file=sys.stderr)
        return []
    cands = data.get("candidates") or []
    all_repos = [c["repo_id"] for c in cands if c.get("repo_id")]

    batch_size_raw = (os.environ.get("INPUT_BATCH_SIZE") or "").strip()
    try:
        batch_size = int(batch_size_raw)
    except ValueError:
        batch_size = 0
    if batch_size <= 0:
        print(f"candidates: returning all {len(all_repos)} repos "
              f"(BATCH_SIZE unset)", file=sys.stderr)
        return all_repos

    batch_index_raw = (os.environ.get("INPUT_BATCH_INDEX") or "0").strip()
    try:
        batch_index = int(batch_index_raw)
    except ValueError:
        batch_index = 0
    start = batch_index * batch_size
    end = start + batch_size
    sliced = all_repos[start:end]
    print(f"candidates: batch_index={batch_index} size={batch_size} "
          f"→ {len(sliced)} of {len(all_repos)} ({start}:{end})",
          file=sys.stderr)
    return sliced


def _all_from_candidates(cands_path: Path) -> list[str]:
    try:
        data = json.loads(cands_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"failed to read candidates file {cands_path}: {e}",
              file=sys.stderr)
        return []
    cands = data.get("candidates") or []
    all_repos = [c["repo_id"] for c in cands if c.get("repo_id")]
    print(f"candidates: loaded all {len(all_repos)} repos for exclusion-first selection",
          file=sys.stderr)
    return all_repos


def collect_repos() -> list[str]:
    explicit = (os.environ.get("INPUT_MODELS") or "").strip()
    if explicit:
        # Whitespace OR comma separated, both supported (workflow_dispatch UI is
        # space-friendly; downstream callers may comma-list).
        repos = re.split(r"[\s,]+", explicit)
        return [r for r in repos if r]

    cands_file = (os.environ.get("INPUT_CANDIDATES_FILE") or "").strip()
    if not cands_file and os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        cands_file = DEFAULT_CRON_CANDIDATES_FILE
    if cands_file:
        cands_path = Path(cands_file)
        if not cands_path.is_absolute():
            # Relative to repo root (workflow `working-directory: ci` means
            # CWD = ci/, so resolve from there's parent).
            cwd = Path.cwd()
            for base in [cwd, cwd.parent]:
                p = base / cands_file
                if p.exists():
                    cands_path = p
                    break
        if not cands_path.exists():
            print(f"ERROR: candidates file not found: {cands_file} "
                  f"(tried as relative + absolute)", file=sys.stderr)
            return []
        exclusion_first = (
            _truthy(os.environ.get("INPUT_EXCLUDE_LEADERBOARD"))
            or _truthy(os.environ.get("INPUT_EXCLUDE_ACTIVE_WORKFLOWS"))
        )
        repos = _all_from_candidates(cands_path) if exclusion_first else _slice_from_candidates(cands_path)
        repos = _apply_exclusions(repos)
        batch_size_raw = (os.environ.get("INPUT_BATCH_SIZE") or "").strip()
        try:
            batch_size = int(batch_size_raw)
        except ValueError:
            batch_size = 0
        # When exclusions are active, slice *after* exclusion so cron gets the
        # next N runnable models rather than a shrunken fixed slice.
        if batch_size > 0 and exclusion_first:
            repos = repos[:batch_size]
            print(f"post-exclusion batch cap: {len(repos)} repos",
                  file=sys.stderr)
        return repos

    hf_top = int(os.environ.get("INPUT_HF_TOP") or "5")
    min_params = float(os.environ.get("INPUT_MIN_PARAMS") or "7")
    hf = HuggingFaceClient(os.environ.get("HF_TOKEN", ""))
    print(f"fetching HF top-{hf_top} (>={min_params}B)...", file=sys.stderr)
    return _apply_exclusions(hf.top_models(hf_top, min_params_b=min_params))


def main() -> int:
    repos = collect_repos()
    if not repos:
        print("no models selected — empty matrix", file=sys.stderr)
        # GitHub Actions errors on empty matrix; emit a sentinel that the
        # downstream job can detect and skip.
        matrix = {"include": []}
    else:
        matrix = {"include": [{"model": r, "key": slugify(r)} for r in repos]}

    print(json.dumps(matrix, indent=2), file=sys.stderr)

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"matrix={json.dumps(matrix)}\n")
            f.write(f"count={len(matrix['include'])}\n")
    else:
        print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
