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

import concurrent.futures
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Reuse HuggingFaceClient from the existing script — single source of truth
# for the pool-then-filter logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_submit import HuggingFaceClient   # noqa: E402

DEFAULT_CRON_CANDIDATES_FILE = (
    "ci/candidates/production_1000_from_hf_2026-05-25.json"
)


def slugify(repo_id: str) -> str:
    """Turn 'Qwen/Qwen3-8B' into 'qwen-qwen3-8b' for use as artifact key."""
    s = repo_id.lower().replace("/", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


_LB_BASE = "https://core42.primus-safe.amd.com/model-leaderboard"


def _paginate_models(api_path: str) -> tuple[set[str], set[str]]:
    """Walk a paginated list endpoint, returning ``(models, task_ids)``.

    The service caps each response at 500 rows even when a larger ``limit`` is
    supplied, so we follow ``pagination.has_more`` / ``next_offset`` until the
    cursor is exhausted, with a paranoid 50-page / offset 10000 safety stop.
    """
    models: set[str] = set()
    task_ids: set[str] = set()
    offset = 0
    pages = 0
    sep = "&" if "?" in api_path else "?"
    while True:
        url = f"{_LB_BASE}{api_path}{sep}limit=500&offset={offset}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            raise RuntimeError(
                f"failed to query {api_path} page offset={offset}: {e}"
            ) from e
        rows = data.get("results") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise RuntimeError(f"{api_path} response did not contain a results list")
        if not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            if item.get("model"):
                models.add(str(item["model"]).strip().lower())
            if item.get("task_id"):
                task_ids.add(str(item["task_id"]))
            for sub in item.get("tasks") or []:
                if isinstance(sub, dict) and sub.get("task_id"):
                    task_ids.add(str(sub["task_id"]))
        pages += 1
        pg = data.get("pagination") if isinstance(data, dict) else None
        if not isinstance(pg, dict) or not pg.get("has_more"):
            break
        next_off = pg.get("next_offset")
        offset = int(next_off) if isinstance(next_off, int) else offset + len(rows)
        if offset >= 10000 or pages >= 50:
            break
    return models, task_ids


def _resolve_task_models(task_ids: list[str], max_workers: int = 16) -> set[str]:
    """Fetch ``/api/v1/tasks/{tid}`` for each id and collect the ``model`` field.

    Single-GET still returns the row even when the list filter hides it, so we
    use this to recover entries (typically gain >200% rows) that the public
    list endpoints suppress.
    """
    if not task_ids:
        return set()

    def _one(tid: str) -> str | None:
        try:
            with urllib.request.urlopen(
                f"{_LB_BASE}/api/v1/tasks/{tid}", timeout=15,
            ) as r:
                d = json.load(r)
        except Exception as e:
            print(
                f"WARN: single-GET task {tid} failed: {e}", file=sys.stderr,
            )
            return None
        if isinstance(d, dict) and isinstance(d.get("model"), str):
            return d["model"].strip().lower()
        return None

    out: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for m in ex.map(_one, task_ids):
            if m:
                out.add(m)
    return out


def _dashboard_task_ids() -> set[str]:
    """Scrape every ``api/v1/tasks/<tid>`` href from the public dashboard HTML.

    The server-side dashboard exposes rows that the list APIs hide (gain
    >200% etc). We use it purely as a discovery source for hidden task ids.
    """
    tids: set[str] = set()
    pages_seen = 0
    offset = 0
    while True:
        url = f"{_LB_BASE}/dashboard?limit=500&offset={offset}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(
                f"WARN: dashboard scrape page offset={offset} failed: {e}",
                file=sys.stderr,
            )
            break
        page_tids = set(re.findall(r"api/v1/tasks/([A-Za-z0-9_-]+)", html))
        if not page_tids:
            break
        new_count = len(page_tids - tids)
        tids.update(page_tids)
        pages_seen += 1
        # 500-row pages; stop when we've drained
        if new_count == 0 or pages_seen >= 20:
            break
        offset += 500
    return tids


def _leaderboard_models() -> set[str]:
    """Return the full set of model ids already known to the leaderboard.

    Combines three sources to bypass the public list filters:

    * Paginated ``/api/v1/leaderboard`` — the parent-row aggregation.
    * Paginated ``/api/v1/tasks`` — covers tasks the leaderboard collapses.
    * ``/dashboard`` HTML scrape + single-GET — recovers ~24 rows the list
      APIs hide (typically gain >200%); single-GET still returns them.
    """
    models, lb_tids = _paginate_models("/api/v1/leaderboard?sort_by=gain&order=desc")
    task_models, task_tids = _paginate_models("/api/v1/tasks")
    models |= task_models
    visible_tids = lb_tids | task_tids
    hidden_tids = sorted(_dashboard_task_ids() - visible_tids)
    if hidden_tids:
        recovered = _resolve_task_models(hidden_tids)
        models |= recovered
        print(
            f"leaderboard exclusion: recovered {len(recovered)} hidden models "
            f"from {len(hidden_tids)} dashboard-only task ids",
            file=sys.stderr,
        )
    return models


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


def _entry_repo(entry: dict | str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("repo_id") or entry.get("model") or "")
    return str(entry or "")


def _entry_slug(entry: dict | str) -> str:
    return slugify(_entry_repo(entry))


def _apply_exclusions_to_entries(entries: list[dict | str]) -> list[dict | str]:
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
        return entries
    out: list[dict | str] = []
    skipped = 0
    for entry in entries:
        repo = _entry_repo(entry)
        if repo.lower() in excluded_models or slugify(repo) in excluded_slugs:
            skipped += 1
            continue
        out.append(entry)
    print(f"exclusions skipped {skipped}; returning {len(out)} repos",
          file=sys.stderr)
    return out


def _apply_exclusions(repos: list[str]) -> list[str]:
    return [_entry_repo(e) for e in _apply_exclusions_to_entries(repos)]


def _load_candidate_entries(cands_path: Path) -> list[dict]:
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
    out: list[dict] = []
    pool_id = data.get("pool_id") or data.get("id") or cands_path.stem
    for idx, cand in enumerate(cands):
        if not isinstance(cand, dict) or not cand.get("repo_id"):
            continue
        entry = dict(cand)
        entry.setdefault("pool_id", pool_id)
        entry.setdefault("pool_index", idx)
        out.append(entry)
    return out


def _resolve_batch_index(pool_size: int, batch_size: int) -> int:
    raw = (os.environ.get("INPUT_BATCH_INDEX") or "").strip()
    if raw:
        try:
            return max(int(raw), 0)
        except ValueError:
            return 0
    if batch_size <= 0 or pool_size <= 0:
        return 0
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        return _cron_batch_index(pool_size, batch_size)
    run_number = (os.environ.get("GITHUB_RUN_NUMBER") or "").strip()
    try:
        rn = int(run_number)
    except ValueError:
        return 0
    batches = max((pool_size + batch_size - 1) // batch_size, 1)
    # GITHUB_RUN_NUMBER is 1-based; subtract one so the first run maps to
    # slice 0 rather than slice 1.
    return (rn - 1) % batches


def _cron_batch_index(pool_size: int, batch_size: int) -> int:
    """Deterministic production-pool rotation for Beijing 00:00/12:00 cron.

    Manual workflow dispatches increment GitHub's run number, so using
    ``GITHUB_RUN_NUMBER`` for schedule rotation makes the next cron batch depend
    on ad-hoc smoke runs. Instead, schedule uses the Beijing half-day slot:
    00:00-11:59 => slot 0, 12:00-23:59 => slot 1. The epoch is pinned to the
    production pool date so every operator can predict the slice before cron
    fires.
    """
    batches = max((pool_size + batch_size - 1) // batch_size, 1)
    epoch = datetime(2026, 5, 25, tzinfo=timezone(timedelta(hours=8)))
    now_raw = (os.environ.get("INPUT_CRON_NOW") or "").strip()
    if now_raw:
        try:
            now_utc = datetime.fromisoformat(now_raw.replace("Z", "+00:00"))
        except ValueError:
            now_utc = datetime.now(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(epoch.tzinfo)
    days = (now_bj.date() - epoch.date()).days
    half_day_slot = 0 if now_bj.hour < 12 else 1
    slot = max(days, 0) * 2 + half_day_slot
    batch_index = slot % batches
    print(
        "cron rotation: "
        f"beijing_time={now_bj.isoformat()} epoch={epoch.date().isoformat()} "
        f"slot={slot} batches={batches} batch_index={batch_index}",
        file=sys.stderr,
    )
    return batch_index


def _slice_entries(entries: list[dict | str]) -> list[dict | str]:
    all_count = len(entries)

    batch_size_raw = (os.environ.get("INPUT_BATCH_SIZE") or "").strip()
    try:
        batch_size = int(batch_size_raw)
    except ValueError:
        batch_size = 0
    if batch_size <= 0:
        print(f"candidates: returning all {all_count} repos "
              f"(BATCH_SIZE unset)", file=sys.stderr)
        return entries

    batch_index = _resolve_batch_index(all_count, batch_size)
    start = batch_index * batch_size
    end = start + batch_size
    sliced = entries[start:end]
    if len(sliced) < batch_size and start and entries:
        sliced = sliced + entries[:batch_size - len(sliced)]
    for item in sliced:
        if isinstance(item, dict):
            item["_selected_batch_index"] = batch_index
            item["_selected_batch_size"] = batch_size
    print(f"candidates: batch_index={batch_index} size={batch_size} "
          f"→ {len(sliced)} of {all_count} ({start}:{end})",
          file=sys.stderr)
    return sliced


def _slice_entries_with_active_refill(
    entries: list[dict | str],
    excluded_slugs: set[str],
) -> list[dict | str]:
    """Rotate over the fixed pool, skipping active jobs and refilling forward.

    This preserves the production pool ordering. Unlike filtering the whole
    pool first, a model that is active in a future slice does not shift today's
    slice boundaries.
    """
    batch_size_raw = (os.environ.get("INPUT_BATCH_SIZE") or "").strip()
    try:
        batch_size = int(batch_size_raw)
    except ValueError:
        batch_size = 0
    if batch_size <= 0:
        out = [e for e in entries if _entry_slug(e) not in excluded_slugs]
        for item in out:
            if isinstance(item, dict):
                item["_selected_batch_index"] = 0
                item["_selected_batch_size"] = len(out)
        return out

    pool_size = len(entries)
    if pool_size == 0:
        return []
    batch_index = _resolve_batch_index(pool_size, batch_size)
    start = (batch_index * batch_size) % pool_size
    selected: list[dict | str] = []
    visited = 0
    pos = start
    skipped = 0
    while visited < pool_size and len(selected) < batch_size:
        entry = entries[pos]
        if _entry_slug(entry) in excluded_slugs:
            skipped += 1
        else:
            selected.append(entry)
        visited += 1
        pos = (pos + 1) % pool_size
    for item in selected:
        if isinstance(item, dict):
            item["_selected_batch_index"] = batch_index
            item["_selected_batch_size"] = batch_size
    print(
        f"active-refill slice: index={batch_index} start={start} "
        f"selected={len(selected)} skipped_active={skipped}",
        file=sys.stderr,
    )
    return selected


def _slice_from_candidates(cands_path: Path) -> list[str]:
    return [_entry_repo(e) for e in _slice_entries(_load_candidate_entries(cands_path))]


def _all_from_candidates(cands_path: Path) -> list[str]:
    all_repos = [_entry_repo(e) for e in _load_candidate_entries(cands_path)]
    print(f"candidates: loaded all {len(all_repos)} repos for exclusion-first selection",
          file=sys.stderr)
    return all_repos


def collect_entries() -> list[dict | str]:
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
        entries = _load_candidate_entries(cands_path)
        exclude_leaderboard = _truthy(os.environ.get("INPUT_EXCLUDE_LEADERBOARD"))
        exclude_active = _truthy(os.environ.get("INPUT_EXCLUDE_ACTIVE_WORKFLOWS"))
        if exclude_leaderboard:
            # Leaderboard exclusion is only for discovery mode; production
            # reruns set this false. Apply globally when requested.
            excluded_models = _leaderboard_models()
            print(f"leaderboard exclusion: {len(excluded_models)} models",
                  file=sys.stderr)
            entries = [
                e for e in entries
                if _entry_repo(e).lower() not in excluded_models
            ]
        if exclude_active:
            excluded_slugs = _active_workflow_slugs()
            print(f"active workflow exclusion: {len(excluded_slugs)} slugs",
                  file=sys.stderr)
            entries = _slice_entries_with_active_refill(entries, excluded_slugs)
            return entries
        else:
            entries = _slice_entries(entries)
        return entries

    hf_top = int(os.environ.get("INPUT_HF_TOP") or "5")
    min_params = float(os.environ.get("INPUT_MIN_PARAMS") or "7")
    hf = HuggingFaceClient(os.environ.get("HF_TOKEN", ""))
    print(f"fetching HF top-{hf_top} (>={min_params}B)...", file=sys.stderr)
    return _apply_exclusions(hf.top_models(hf_top, min_params_b=min_params))


def collect_repos() -> list[str]:
    return [_entry_repo(e) for e in collect_entries()]


def _matrix_entry(entry: dict | str) -> dict:
    repo = _entry_repo(entry)
    out = {"model": repo, "key": slugify(repo)}
    if isinstance(entry, dict):
        for key in (
            "pool_id", "pool_index", "task_count", "positive_task_count",
            "last_success_at", "framework", "precision", "gpu", "tp",
            "conc", "gain", "task_id", "created_at",
        ):
            if entry.get(key) is not None:
                out[key] = entry[key]
        batch_size_raw = (os.environ.get("INPUT_BATCH_SIZE") or "").strip()
        selected_batch_size = entry.get("_selected_batch_size")
        selected_batch_index = entry.get("_selected_batch_index")
        try:
            batch_size = int(selected_batch_size or batch_size_raw or 0)
        except ValueError:
            batch_size = 0
        if batch_size > 0:
            out["batch_size"] = batch_size
        if selected_batch_index is not None:
            out["batch_index"] = selected_batch_index
    return out


def main() -> int:
    entries = collect_entries()
    if not entries:
        print("no models selected — empty matrix", file=sys.stderr)
        # GitHub Actions errors on empty matrix; emit a sentinel that the
        # downstream job can detect and skip.
        matrix = {"include": []}
    else:
        matrix = {"include": [_matrix_entry(e) for e in entries]}

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
