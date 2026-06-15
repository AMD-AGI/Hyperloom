#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Generate GitHub Actions matrix JSON from HuggingFace top-N, explicit list,
or a pre-built candidates file (preferred for batch-driven dispatch).

Inputs (env vars, set by the workflow), in priority order:
  INPUT_MODELS           Space-separated HF repo IDs. When INPUT_CANDIDATES_FILE
                         is also set, this filters candidates[] while preserving
                         fixed per-model config; otherwise it overrides discovery.
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

import bisect
import concurrent.futures
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Reuse HuggingFaceClient for the pool-then-filter logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_submit import HuggingFaceClient   # noqa: E402

# The schedule always sets INPUT_CANDIDATES_FILE explicitly (built from the
# WEKAFS_CHENYI_DIR secret), so this is only the empty-env fallback. Keep the
# personal /wekafs root out of source; allow a CRON_CANDIDATES_FILE env override.
DEFAULT_CRON_CANDIDATES_FILE = os.environ.get(
    "CRON_CANDIDATES_FILE",
    "ci/candidates/hf_downloads_gt100_rotate_2026-06-11.json",
)


def slugify(repo_id: str) -> str:
    """Turn 'Qwen/Qwen3-8B' into 'qwen-qwen3-8b' for use as artifact key.

    Args:
        repo_id (str): HuggingFace repo id (``owner/name``).

    Returns:
        str: A lowercase, hyphenated slug safe for artifact names; ``"model"``
            if the input reduces to empty.
    """
    s = repo_id.lower().replace("/", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def _truthy(value: str | None) -> bool:
    """Interpret a string env value as a boolean flag.

    Args:
        value (str | None): Raw value (e.g. from an env var).

    Returns:
        bool: ``True`` for ``1/true/yes/y/on`` (case-insensitive), else
            ``False``.
    """
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


_LB_BASE = "https://core42.primus-safe.amd.com/model-leaderboard"


def _paginate_models(api_path: str) -> tuple[set[str], set[str]]:
    """Walk a paginated list endpoint, returning ``(models, task_ids)``.

    The service caps each response at 500 rows even when a larger ``limit`` is
    supplied, so we follow ``pagination.has_more`` / ``next_offset`` until the
    cursor is exhausted, with a paranoid 50-page / offset 10000 safety stop.

    Args:
        api_path (str): Leaderboard API path (may include a query string).

    Returns:
        tuple[set[str], set[str]]: ``(models, task_ids)`` collected across all
            pages, with model ids lowercased.

    Raises:
        RuntimeError: If a page request fails or returns an unexpected shape.
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

    Args:
        task_ids (list[str]): Task ids to resolve.
        max_workers (int): Thread-pool size for concurrent single-GETs.

    Returns:
        set[str]: Lowercased model ids recovered from the tasks.
    """
    if not task_ids:
        return set()

    def _one(tid: str) -> str | None:
        """Fetch one task and return its model id.

        Args:
            tid (str): Task id to fetch.

        Returns:
            str | None: The lowercased model id, or ``None`` on failure or if
                absent.
        """
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

    Returns:
        set[str]: Task ids scraped from the dashboard HTML pages.
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

    Returns:
        set[str]: The union of all known leaderboard model ids (lowercased).
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

    Returns:
        set[str]: Matrix slugs (job-name suffixes) of optimize jobs currently
            queued or in progress; empty when GitHub auth env is missing.
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
        """GET a GitHub API URL with auth headers and parse JSON.

        Args:
            url (str): Fully-qualified GitHub API URL.

        Returns:
            dict: The parsed JSON response.
        """
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
    """Return the repo id for a candidate entry (dict or bare string).

    Args:
        entry (dict | str): Candidate dict (``repo_id``/``model``) or a plain
            repo id string.

    Returns:
        str: The repo id, or an empty string if none is present.
    """
    if isinstance(entry, dict):
        return str(entry.get("repo_id") or entry.get("model") or "")
    return str(entry or "")


def _entry_slug(entry: dict | str) -> str:
    """Return the artifact slug for a candidate entry.

    Args:
        entry (dict | str): Candidate dict or bare repo id string.

    Returns:
        str: The slugified repo id.
    """
    return slugify(_entry_repo(entry))


def _parse_explicit_models(value: str) -> list[str]:
    """Split an explicit model list into individual repo ids.

    Accepts whitespace- or comma-separated input so both the
    ``workflow_dispatch`` UI (space-friendly) and programmatic callers
    (comma-lists) work.

    Args:
        value: Raw model-list string.

    Returns:
        The non-empty repo ids in order.
    """
    return [r for r in re.split(r"[\s,]+", value.strip()) if r]


def _filter_entries_by_explicit_models(
    entries: list[dict | str],
    explicit_repos: list[str],
) -> list[dict | str]:
    """Filter candidate entries by repo id while preserving fixed config.

    Historically INPUT_MODELS returned bare repo strings and therefore lost
    framework / precision / tp / conc from candidates JSON. For manual reruns
    of a small subset from a fixed pool, keep the candidate dicts intact.
    """
    if not explicit_repos:
        return entries
    by_repo: dict[str, dict | str] = {
        _entry_repo(entry).lower(): entry
        for entry in entries
        if _entry_repo(entry)
    }
    out: list[dict | str] = []
    missing: list[str] = []
    for repo in explicit_repos:
        entry = by_repo.get(repo.lower())
        if entry is None:
            missing.append(repo)
            continue
        out.append(entry)
    if missing:
        print(
            "WARNING: INPUT_MODELS requested repo(s) not present in "
            f"candidates file: {', '.join(missing)}",
            file=sys.stderr,
        )
    return out


def _apply_exclusions_to_entries(entries: list[dict | str]) -> list[dict | str]:
    """Drop entries excluded by leaderboard and/or active-workflow filters.

    Exclusions are controlled by the ``INPUT_EXCLUDE_LEADERBOARD`` and
    ``INPUT_EXCLUDE_ACTIVE_WORKFLOWS`` env vars.

    Args:
        entries (list[dict | str]): Candidate entries to filter.

    Returns:
        list[dict | str]: The surviving entries (input order preserved).
    """
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
    """Apply the exclusion filters to a list of bare repo ids.

    Args:
        repos (list[str]): Repo ids to filter.

    Returns:
        list[str]: The surviving repo ids.
    """
    return [_entry_repo(e) for e in _apply_exclusions_to_entries(repos)]


def _load_candidate_entries(cands_path: Path) -> list[dict]:
    """Load candidates.json and take an INPUT_BATCH_INDEX/INPUT_BATCH_SIZE slice.

    BATCH_SIZE unset/0 returns all candidates. Slice order follows the JSON
    (HF download rank) for deterministic batch dispatch.
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
    """Determine which batch slice to dispatch for this run.

    Honors an explicit ``INPUT_BATCH_INDEX``; otherwise derives the index from
    the sequential scheduled-fire counter (schedule events, see
    ``_cron_batch_index``) or the GitHub run number (manual dispatch).

    Args:
        pool_size (int): Total number of candidate entries.
        batch_size (int): Entries per batch.

    Returns:
        int: The 0-based batch index.
    """
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
    # GITHUB_RUN_NUMBER is 1-based; subtract one so run 1 maps to slice 0.
    return (rn - 1) % batches


# UTC hours at which the schedule cron fires (keep in sync with the
# ``schedule: cron`` expression at the top of optimize-submit.yml).
# 2026-06-15: dropped to twice a day (UTC 04:00 / 16:00 = Beijing 12:00 / 00:00).
_CRON_FIRE_HOURS_UTC = (4, 16)

# Anchor fire: the first scheduled run at/after this instant maps to batch 0.
# The rotation pool is ordered not-run-first, so batch 0 hits the not-yet-run
# head. Merge the candidate-pool change before this fire so the very next cron
# starts at batch 0; bump this if the merge slips to a later fire.
# 2026-06-15: re-anchored to the next 16:00 UTC fire so the freshly front-loaded
# 449 >12B multimodal NOT-run models are swept starting at batch 0.
_CRON_ANCHOR_UTC = datetime(2026, 6, 15, 16, 0, tzinfo=timezone.utc)


def _cron_fire_counter(now_utc: datetime) -> int:
    """Map a UTC instant to a strictly increasing scheduled-fire counter.

    Each scheduled cron fire (UTC 4/16) advances the counter by exactly one,
    so consecutive cron runs step through batch 0, 1, 2, ... in order (unlike a
    half-day slot, which would collapse multiple fires onto one index).
    """
    idx = bisect.bisect_right(_CRON_FIRE_HOURS_UTC, now_utc.hour) - 1
    if idx < 0:
        # Before the day's first fire — count as the last fire of the prior day.
        base_date = now_utc.date() - timedelta(days=1)
        idx = len(_CRON_FIRE_HOURS_UTC) - 1
    else:
        base_date = now_utc.date()
    return base_date.toordinal() * len(_CRON_FIRE_HOURS_UTC) + idx


def _cron_batch_index(pool_size: int, batch_size: int) -> int:
    """Sequential production-pool rotation for the twice-daily cron.

    Each scheduled fire advances the batch index by one (0, 1, 2, ... wrapping
    at the batch count), so the cron marches the whole pool in order and then
    repeats — independent of ad-hoc manual dispatches (which would otherwise
    perturb a ``GITHUB_RUN_NUMBER``-based scheme). The index is anchored to
    ``_CRON_ANCHOR_UTC`` so the first fire at/after the anchor is batch 0.

    Fires strictly before the anchor are clamped to batch 0 (the not-run head)
    instead of wrapping to the pool tail: a negative ``steps % batches`` would
    otherwise land on the last batch, silently skipping the not-run backlog the
    anchor is meant to drain first.
    """
    batches = max((pool_size + batch_size - 1) // batch_size, 1)
    now_raw = (os.environ.get("INPUT_CRON_NOW") or "").strip()
    if now_raw:
        try:
            now_utc = datetime.fromisoformat(now_raw.replace("Z", "+00:00"))
        except ValueError:
            now_utc = datetime.now(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)
    now_utc = now_utc.astimezone(timezone.utc)
    steps = max(_cron_fire_counter(now_utc) - _cron_fire_counter(_CRON_ANCHOR_UTC), 0)
    batch_index = steps % batches
    print(
        "cron rotation (sequential): "
        f"utc_time={now_utc.isoformat()} anchor={_CRON_ANCHOR_UTC.isoformat()} "
        f"steps={steps} batches={batches} batch_index={batch_index}",
        file=sys.stderr,
    )
    return batch_index


def _slice_entries(entries: list[dict | str]) -> list[dict | str]:
    """Select this run's batch slice from the candidate entries.

    Reads ``INPUT_BATCH_SIZE`` (0/unset returns all). For a manual dispatch a
    partial tail slice wraps to the front of the pool to keep the batch full.
    For the ``schedule`` rotation the tail is NOT wrapped: the sequential cron
    index advances to batch 0 on the very next fire, so a head-refill here would
    re-submit those head models in the same cycle (the wrap slice and the next
    fire's batch 0 overlap), double-dispatching them while the slow tasks from
    this fire are still in flight. Letting the tail batch be short keeps every
    batch a disjoint slice.

    Args:
        entries (list[dict | str]): All candidate entries in pool order.

    Returns:
        list[dict | str]: The selected slice (or all entries when no batch
            size is set).
    """
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
    is_schedule = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    if len(sliced) < batch_size and start and entries and not is_schedule:
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

    Args:
        entries (list[dict | str]): All candidate entries in pool order.
        excluded_slugs (set[str]): Slugs of currently-active jobs to skip.

    Returns:
        list[dict | str]: The selected slice with active entries skipped and
            forward-refilled, annotated with batch index/size.
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
    """Load a candidates file and return the batch slice as repo ids.

    Args:
        cands_path (Path): Path to the candidates JSON file.

    Returns:
        list[str]: Repo ids in the selected batch slice.
    """
    return [_entry_repo(e) for e in _slice_entries(_load_candidate_entries(cands_path))]


def _all_from_candidates(cands_path: Path) -> list[str]:
    """Load a candidates file and return all repo ids (no slicing).

    Args:
        cands_path (Path): Path to the candidates JSON file.

    Returns:
        list[str]: Every candidate repo id, for exclusion-first selection.
    """
    all_repos = [_entry_repo(e) for e in _load_candidate_entries(cands_path)]
    print(f"candidates: loaded all {len(all_repos)} repos for exclusion-first selection",
          file=sys.stderr)
    return all_repos


def collect_entries() -> list[dict | str]:
    """Resolve the model entries to run from env-var inputs.

    Selection priority: explicit ``INPUT_MODELS`` > candidates file
    (``INPUT_CANDIDATES_FILE``, or the cron default on schedule events) with
    batch slicing and optional leaderboard/active-workflow exclusions > live
    HuggingFace top-N fallback.

    Returns:
        list[dict | str]: Candidate entries (dicts and/or repo id strings);
            empty if nothing is selected.
    """
    explicit = (os.environ.get("INPUT_MODELS") or "").strip()
    explicit_repos = _parse_explicit_models(explicit) if explicit else []
    cands_file = (os.environ.get("INPUT_CANDIDATES_FILE") or "").strip()
    if explicit_repos and not cands_file:
        return explicit_repos

    if not cands_file and os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        cands_file = DEFAULT_CRON_CANDIDATES_FILE
    if cands_file:
        cands_path = Path(cands_file)
        if not cands_path.is_absolute():
            # Resolve relative to CWD and its parent (workflow CWD is ci/).
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
        if explicit_repos:
            entries = _filter_entries_by_explicit_models(entries, explicit_repos)
            return _apply_exclusions_to_entries(entries)
        exclude_leaderboard = _truthy(os.environ.get("INPUT_EXCLUDE_LEADERBOARD"))
        exclude_active = _truthy(os.environ.get("INPUT_EXCLUDE_ACTIVE_WORKFLOWS"))
        if exclude_leaderboard:
            # Discovery mode only; production reruns set this false.
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
    """Resolve the selected entries and return them as bare repo ids.

    Returns:
        list[str]: Repo ids for the selected models.
    """
    return [_entry_repo(e) for e in collect_entries()]


def _matrix_entry(entry: dict | str) -> dict:
    """Build one GitHub Actions matrix include entry from a candidate.

    Args:
        entry (dict | str): Candidate dict or bare repo id.

    Returns:
        dict: A matrix entry with at least ``model`` and ``key``, plus any
            available pool/benchmark metadata and batch index/size.
    """
    repo = _entry_repo(entry)
    out = {"model": repo, "key": slugify(repo)}
    if isinstance(entry, dict):
        for key in (
            "pool_id", "pool_index", "task_count", "positive_task_count",
            "last_success_at", "framework", "precision", "gpu", "tp",
            "conc", "gain", "task_id", "created_at",
            "nodes", "rayjob_image",
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
    """Build the matrix JSON and emit it to ``GITHUB_OUTPUT`` or stdout.

    Returns:
        int: Process exit code (always ``0``).
    """
    entries = collect_entries()
    if not entries:
        print("no models selected — empty matrix", file=sys.stderr)
        # Empty include sentinel; GitHub Actions errors on a truly empty matrix.
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
