#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Build the production rerun corpus directly from the HuggingFace catalog.

Pool selection: pipeline_tag=text-generation, inference_provider=all,
num_parameters >= --min-params (default 12), minus the exclusion keywords in
``ci/inferenceX_models.yaml::production_pool_exclusion_keywords``.

The HF list endpoint caps a single sort at 500 rows, so we crawl every
``sort`` × ``direction`` plus a per-author slice list, dedup by repo id, then
sort by downloads descending. Output keeps the ``hyperloom.production_corpus.v1``
schema so downstream consumers (generate_hf_matrix.py, optimize-submit) are
unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HF_API_BASE = "https://huggingface.co/api/models"
HF_TIMEOUT = 30
HF_LIST_WORKERS = 10
HF_LIMIT = 500  # HF list endpoint hard-caps a single sort at 500 rows.

# Sort × direction combinations. Each one returns up to 500 rows in a
# distinct ranking; their union covers ~3× a single sort's row count.
SORT_VARIANTS = [
    ("downloads", -1),
    ("downloads", 1),
    ("likes", -1),
    ("likes", 1),
    ("trendingScore", -1),
    ("trendingScore", 1),
    ("lastModified", -1),
    ("lastModified", 1),
    ("createdAt", -1),
    ("createdAt", 1),
]

# Author slices pick up models that don't make any of the global
# top-500 leaderboards (long-tail Qwen variants, unsloth fine-tunes, etc.).
DEFAULT_AUTHOR_SLICES = [
    "Qwen", "meta-llama", "deepseek-ai", "mistralai", "microsoft", "nvidia",
    "RedHatAI", "unsloth", "NousResearch", "MiniMaxAI", "zai-org",
    "moonshotai", "Salesforce", "google", "ibm-granite", "01-ai",
    "tiiuae", "AIDC-AI", "allenai", "stabilityai", "Snowflake",
    "stepfun-ai", "QuantTrio", "huihui-ai", "BAAI", "LGAI-EXAONE",
    "rinna", "OpenLLM-France", "TheBloke", "PrimeIntellect",
    "fblgit", "lightonai", "EleutherAI", "EpistemeAI", "MoxoffSrL",
]


def _load_yaml_exclusions(path: Path) -> tuple[set[str], list[str]]:
    """Read ``ci/inferenceX_models.yaml`` and return ``(exact_ids, keywords)``.

    Consumes ``models[].hf_model`` (exact lower-case match) and
    ``production_pool_exclusion_keywords`` (substring match).
    """
    if not path.exists():
        return set(), []
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        models = data.get("models") or []
        exact = {
            str(item.get("hf_model")).strip().lower()
            for item in models
            if isinstance(item, dict) and item.get("hf_model")
        }
        keywords_raw = data.get("production_pool_exclusion_keywords") or []
        keywords = [
            str(k).strip().lower() for k in keywords_raw
            if isinstance(k, str) and str(k).strip()
        ]
        return exact, keywords
    except Exception:
        exact: set[str] = set()
        for line in text.splitlines():
            ls = line.strip()
            if ls.startswith("#"):
                continue
            if ls.startswith("hf_model:"):
                exact.add(ls.split(":", 1)[1].strip().strip("'\"").lower())
        return exact, []


def _fetch_one(url: str) -> list[dict[str, Any]]:
    """Fetch a single HF list URL, returning an empty list on any error.

    Args:
        url (str): Fully-formed HuggingFace models list endpoint URL.

    Returns:
        list[dict[str, Any]]: The decoded JSON rows, or an empty list if the
        request fails or the response is not a list.
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=HF_TIMEOUT) as r:
            data = json.load(r)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _crawl(
    *, min_params: float, base_extra: str = "",
    workers: int = HF_LIST_WORKERS,
) -> dict[str, dict[str, Any]]:
    """Pull every (sort × direction) × (author slice) shard concurrently.

    Each shard is a separate HF list request; results are de-duplicated by
    repo id across all shards.

    Args:
        min_params (float): Minimum parameter count (in billions) for the
            server-side ``num_parameters`` filter.
        base_extra (str): Optional extra query string appended to the base
            filter.
        workers (int): Number of concurrent fetch workers.

    Returns:
        dict[str, dict[str, Any]]: Mapping of repo id to its HF model row.
    """
    base = (
        "pipeline_tag=text-generation"
        f"&num_parameters=min:{int(min_params)}B"
        "&inference_provider=all"
        "&expand=safetensors&expand=downloads&expand=likes"
        "&expand=trendingScore&expand=lastModified&expand=pipeline_tag"
    )
    if base_extra:
        base = f"{base}&{base_extra}"

    urls: list[str] = []
    for sort_field, direction in SORT_VARIANTS:
        urls.append(
            f"{HF_API_BASE}?{base}"
            f"&sort={sort_field}&direction={direction}&limit={HF_LIMIT}"
        )
    for author in DEFAULT_AUTHOR_SLICES:
        urls.append(
            f"{HF_API_BASE}?{base}"
            f"&sort=downloads&direction=-1&limit={HF_LIMIT}"
            f"&author={author}"
        )

    seen: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def _ingest(rows: list[dict[str, Any]]) -> int:
        """Merge fetched rows into the shared ``seen`` map under a lock.

        Args:
            rows (list[dict[str, Any]]): Rows from one fetched shard.

        Returns:
            int: The number of newly added (previously unseen) rows.
        """
        added = 0
        with lock:
            for m in rows:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id")
                if mid and mid not in seen:
                    seen[mid] = m
                    added += 1
        return added

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for added in ex.map(_fetch_one, urls):
            _ingest(added)
    elapsed = time.time() - started
    print(
        f"  HF crawl: fetched {len(urls)} shards in {elapsed:.1f}s, "
        f"got {len(seen)} unique rows", file=sys.stderr,
    )
    return seen


def _params_total(m: dict[str, Any]) -> int:
    """Return a model's total parameter count from its safetensors metadata.

    Args:
        m (dict[str, Any]): An HF model row.

    Returns:
        int: The total parameter count, or ``0`` if unavailable or non-positive.
    """
    sf = m.get("safetensors") or {}
    total = sf.get("total")
    return int(total) if isinstance(total, (int, float)) and total > 0 else 0


def _is_excluded(mid: str, exact_ids: set[str], keywords: list[str]) -> bool:
    """Check whether a repo id is excluded by exact id or keyword match.

    Args:
        mid (str): The model repo id.
        exact_ids (set[str]): Lower-cased repo ids to exclude exactly.
        keywords (list[str]): Lower-cased substrings; any match excludes.

    Returns:
        bool: True if the model should be excluded from the pool.
    """
    ml = mid.lower()
    if ml in exact_ids:
        return True
    return any(kw in ml for kw in keywords)


def build_candidates(
    raw: dict[str, dict[str, Any]],
    *,
    min_params_b: float,
    max_models: int,
    sort_mode: str,
    pool_id: str,
    exact_ids: set[str],
    exclusion_keywords: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply size/exclusion filters and produce the candidates list.

    Filters raw HF rows by minimum parameter count and exclusion rules, maps
    survivors into the ``hyperloom.production_corpus.v1`` candidate schema,
    sorts them, and applies an optional cap.

    Args:
        raw (dict[str, dict[str, Any]]): Mapping of repo id to HF model row.
        min_params_b (float): Minimum parameter count in billions.
        max_models (int): Cap on the number of candidates (0 = no cap).
        sort_mode (str): ``"downloads"`` (desc) or ``"model"`` (alphabetical).
        pool_id (str): Pool identifier stamped on each candidate.
        exact_ids (set[str]): Repo ids to exclude exactly.
        exclusion_keywords (list[str]): Substrings used to exclude repo ids.

    Returns:
        tuple[list[dict[str, Any]], dict[str, int]]: The candidate list and a
        stats dict counting rows seen and exclusions applied.
    """
    stats = {
        "hf_rows_seen": len(raw),
        "models_excluded_by_keyword": 0,
        "models_excluded_lt_min_params": 0,
        "models_without_safetensors": 0,
    }
    threshold = int(min_params_b * 1e9) if min_params_b > 0 else 0

    candidates: list[dict[str, Any]] = []
    for m in raw.values():
        mid = m.get("id")
        if not mid:
            continue
        params = _params_total(m)
        if threshold and params == 0:
            # Defensive: HF min: filter should have removed these already.
            stats["models_without_safetensors"] += 1
            continue
        if threshold and params < threshold:
            stats["models_excluded_lt_min_params"] += 1
            continue
        if _is_excluded(mid, exact_ids, exclusion_keywords):
            stats["models_excluded_by_keyword"] += 1
            continue
        candidates.append({
            "pool_id": pool_id,
            "pool_index": None,
            "repo_id": mid,
            "params_b": round(params / 1e9, 3) if params else None,
            "downloads": m.get("downloads"),
            "likes": m.get("likes"),
            "trending_score": m.get("trendingScore"),
            "last_modified": m.get("lastModified"),
            "pipeline_tag": m.get("pipeline_tag") or "text-generation",
            # Stub fields the legacy schema expects; filled by the results
            # service after the first optimization run.
            "data_quality": "hf_inference_listed",
            "framework": None,
            "precision": None,
            "gpu": None,
            "tp": None,
            "conc": None,
            "gain": None,
            "task_id": None,
            "created_at": None,
            "task_count": 0,
            "positive_task_count": 0,
            "partial_task_count": 0,
            "last_success_at": None,
            "best_task": None,
            "latest_task": None,
        })

    if sort_mode == "model":
        candidates.sort(key=lambda x: str(x["repo_id"]).lower())
    else:  # downloads (default)
        candidates.sort(
            key=lambda x: (-(x.get("downloads") or 0), str(x["repo_id"]).lower()),
        )

    if max_models > 0:
        candidates = candidates[:max_models]
    for idx, item in enumerate(candidates):
        item["pool_index"] = idx
    return candidates, stats


def main() -> int:
    """Crawl the HF catalog, build the candidate pool, and write the JSON.

    Parses CLI arguments, crawls HuggingFace, applies exclusion filters, and
    writes the ``hyperloom.production_corpus.v1`` payload (unless ``--dry-run``).

    Returns:
        int: Process exit code (``0`` on success or dry-run).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-params", type=float, default=12.0,
        help="Server-side num_parameters>=NB filter (default 12 — matches the "
             "operator search URL).",
    )
    ap.add_argument(
        "--max-models", type=int, default=0,
        help="Truncate the pool to this size after sorting (0 = no cap). "
             "We always keep ALL candidates that pass the filter so cron "
             "can rotate through every model on the operator-defined cycle.",
    )
    ap.add_argument(
        "--sort", choices=["downloads", "model"], default="downloads",
        help="In-pool ordering. downloads sorts by downloads desc (production "
             "default); model sorts alphabetically (deterministic for diffs).",
    )
    ap.add_argument("--pool-id", default="",
                    help="Override pool_id label (default: hf-inference-<date>)")
    ap.add_argument("--output", default="",
                    help="Override output JSON path "
                         "(default: ci/candidates/production_1000_from_hf_<date>.json)")
    ap.add_argument(
        "--exclude-config",
        default="ci/inferenceX_models.yaml",
        help="YAML containing models[].hf_model (exact match) plus "
             "production_pool_exclusion_keywords (substring match).",
    )
    ap.add_argument(
        "--workers", type=int, default=HF_LIST_WORKERS,
        help="Parallelism for the HF API crawl.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print policy/stats and exit without writing JSON.")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    pool_id = args.pool_id or f"hf-inference-{now.strftime('%Y-%m-%d')}"
    output = Path(
        args.output
        or f"ci/candidates/production_1000_from_hf_{now.strftime('%Y-%m-%d')}.json"
    )

    print(f"Building HF inference pool: min_params>={args.min_params}B, "
          f"sort={args.sort}, pool_id={pool_id}", file=sys.stderr)
    raw = _crawl(min_params=args.min_params, workers=args.workers)

    exact_ids, keywords = _load_yaml_exclusions(Path(args.exclude_config))
    print(f"  exclusion config: {len(exact_ids)} exact ids + {len(keywords)} keywords",
          file=sys.stderr)

    candidates, stats = build_candidates(
        raw,
        min_params_b=args.min_params,
        max_models=args.max_models,
        sort_mode=args.sort,
        pool_id=pool_id,
        exact_ids=exact_ids,
        exclusion_keywords=keywords,
    )

    payload = {
        "schema_version": "hyperloom.production_corpus.v1",
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "huggingface-api",
        "source_filter": {
            "pipeline_tag": "text-generation",
            "inference_provider": "all",
            "num_parameters_min": f"{int(args.min_params)}B",
        },
        "pool_id": pool_id,
        "policy": {
            "min_params_b": args.min_params,
            "max_models": args.max_models,
            "sort": args.sort,
            "rotation": "cron rotates over the whole pool; missing models are "
                        "submitted on the next cycle (no leaderboard-history "
                        "intersection — every model in the pool is rerunnable "
                        "regardless of prior task data).",
            "excluded_exact_ids": sorted(exact_ids),
            "exclusion_keywords": keywords,
            "refresh_mode": "manual",
        },
        "stats": {
            **stats,
            "candidates_written": len(candidates),
        },
        "candidates": candidates,
    }

    print("=== stats ===", file=sys.stderr)
    print(json.dumps(payload["stats"], indent=2), file=sys.stderr)

    if args.dry_run:
        print(f"[DRY RUN] would write to: {output}", file=sys.stderr)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} ({len(candidates)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
