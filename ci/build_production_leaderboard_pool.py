#!/usr/bin/env python3
"""Build the fixed production rerun corpus from live leaderboard data.

The production CI no longer discovers brand-new models on every cron tick.
Instead, operators periodically run this script to snapshot up to N models
that already have real leaderboard data. The schedule workflow then rotates
over the generated candidates file.

Selection rule:
  * source of truth is the online model-leaderboard API
  * a model is eligible when at least one task has positive baseline and
    optimized throughput
  * best_task = highest gain among positive-throughput tasks
  * latest_task = newest positive-throughput task
  * output order defaults to latest_success_at desc, then model id asc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEADERBOARD_URL = "https://core42.primus-safe.amd.com/model-leaderboard"


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_task(task: dict[str, Any]) -> bool:
    baseline = _to_float(task.get("baseline_throughput"))
    optimized = _to_float(task.get("optimized_throughput"))
    return bool(baseline and baseline > 0 and optimized and optimized > 0)


def _partial_task(task: dict[str, Any]) -> bool:
    baseline = _to_float(task.get("baseline_throughput"))
    optimized = _to_float(task.get("optimized_throughput"))
    return bool((baseline and baseline > 0) or (optimized and optimized > 0))


def _get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            try:
                import requests  # type: ignore
                requests.packages.urllib3.disable_warnings()
                resp = requests.get(url, timeout=timeout, verify=False)
                resp.raise_for_status()
                payload = resp.json()
            except ImportError:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    payload = json.load(r)
            break
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(2 + attempt * 3)
                continue
            raise RuntimeError(f"failed to GET {url}: {last_err}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object response from {url}")
    return payload


def iter_leaderboard_rows(base_url: str, *, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    pages = 0
    base = base_url.rstrip("/")
    while True:
        qs = urllib.parse.urlencode({"limit": limit, "offset": offset})
        data = _get_json(f"{base}/api/v1/leaderboard?{qs}")
        batch = data.get("results") or []
        if not isinstance(batch, list):
            raise RuntimeError("leaderboard response missing results[]")
        rows.extend(item for item in batch if isinstance(item, dict))
        pages += 1
        pg = data.get("pagination") or {}
        if not isinstance(pg, dict) or not pg.get("has_more"):
            break
        next_offset = pg.get("next_offset")
        offset = int(next_offset) if isinstance(next_offset, int) else offset + len(batch)
        if pages > 100 or offset > 100_000:
            raise RuntimeError(
                f"leaderboard pagination did not converge: pages={pages} offset={offset}"
            )
    return rows


def _task_created_at(task: dict[str, Any]) -> str:
    return str(task.get("created_at") or task.get("updated_at") or "")


def _task_gain(task: dict[str, Any]) -> float:
    gain = _to_float(task.get("gain_pct"))
    return gain if gain is not None else -1e9


def _task_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "final_status": task.get("final_status"),
        "baseline_throughput": task.get("baseline_throughput"),
        "optimized_throughput": task.get("optimized_throughput"),
        "gain_pct": task.get("gain_pct"),
        "framework": task.get("framework"),
        "precision": task.get("precision"),
        "gpu_type": task.get("gpu_type"),
        "tp": task.get("tp"),
        "conc": task.get("conc"),
        "isl": task.get("isl"),
        "osl": task.get("osl"),
        "claw_session_id": task.get("claw_session_id"),
        "claw_session_url": task.get("claw_session_url"),
    }


def _load_inferencex_exclusions(path: Path | None) -> set[str]:
    """Read ci/inferenceX_models.yaml and return HF model ids to exclude.

    PyYAML is present in the workflow, but keep a tiny regex fallback so the
    script stays usable in minimal local shells.
    """
    if not path or not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        models = data.get("models") or []
        return {
            str(item.get("hf_model")).strip().lower()
            for item in models
            if isinstance(item, dict) and item.get("hf_model")
        }
    except Exception:
        out: set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if line.startswith("hf_model:"):
                out.add(line.split(":", 1)[1].strip().strip("'\"").lower())
        return out


def _quality_rank(task: dict[str, Any]) -> int:
    if _positive_task(task):
        return 2
    if _partial_task(task):
        return 1
    return 0


def _quality_name(rank: int) -> str:
    if rank >= 2:
        return "positive_metrics"
    if rank == 1:
        return "partial_metrics"
    return "leaderboard_record"


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    max_models: int,
    min_models: int,
    sort_mode: str,
    pool_id: str,
    excluded_models: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    stats = {
        "rows_seen": len(rows),
        "models_excluded_inferencex": 0,
        "models_with_positive_task": 0,
        "models_with_partial_task": 0,
        "models_with_leaderboard_record_only": 0,
    }

    for row in rows:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        if model.lower() in excluded_models:
            stats["models_excluded_inferencex"] += 1
            continue
        tasks = [t for t in (row.get("tasks") or []) if isinstance(t, dict)]
        if not tasks:
            continue

        ranked = sorted(
            tasks,
            key=lambda t: (_quality_rank(t), _task_gain(t), _task_created_at(t)),
            reverse=True,
        )
        usable = ranked[0]
        quality_rank = _quality_rank(usable)
        if quality_rank >= 2:
            stats["models_with_positive_task"] += 1
        elif quality_rank == 1:
            stats["models_with_partial_task"] += 1
        else:
            stats["models_with_leaderboard_record_only"] += 1

        # Best/latest are computed within the highest available quality tier,
        # so positive rows never get displaced by empty newer tasks.
        same_quality = [t for t in tasks if _quality_rank(t) == quality_rank]
        best_task = max(same_quality, key=lambda t: (_task_gain(t), _task_created_at(t)))
        latest_task = max(same_quality, key=_task_created_at)
        item = {
            "pool_id": pool_id,
            # pool_index is assigned after sorting.
            "pool_index": None,
            "repo_id": model,
            "data_quality": _quality_name(quality_rank),
            "task_count": len(tasks),
            "positive_task_count": sum(1 for t in tasks if _positive_task(t)),
            "partial_task_count": sum(1 for t in tasks if _partial_task(t)),
            "last_success_at": latest_task.get("created_at"),
            "framework": latest_task.get("framework") or best_task.get("framework"),
            "precision": latest_task.get("precision") or best_task.get("precision"),
            "gpu": latest_task.get("gpu_type") or best_task.get("gpu_type"),
            "tp": latest_task.get("tp") or best_task.get("tp"),
            "conc": latest_task.get("conc") or best_task.get("conc"),
            "gain": best_task.get("gain_pct"),
            "task_id": latest_task.get("task_id"),
            "created_at": latest_task.get("created_at"),
            "best_task": _task_summary(best_task),
            "latest_task": _task_summary(latest_task),
        }
        candidates.append(item)

    # Always prefer higher data quality first; sort_mode only orders models
    # within the same quality tier. This satisfies ">=1000" by filling from
    # partial/empty leaderboard records only after all positive rows.
    def _quality_key(item: dict[str, Any]) -> int:
        return {
            "positive_metrics": 2,
            "partial_metrics": 1,
            "leaderboard_record": 0,
        }.get(str(item.get("data_quality")), 0)

    if sort_mode == "model":
        candidates.sort(key=lambda x: (-_quality_key(x), str(x["repo_id"]).lower()))
    elif sort_mode == "gain":
        candidates.sort(
            key=lambda x: (
                _quality_key(x),
                _to_float((x.get("best_task") or {}).get("gain_pct")) or -1e9,
                str(x.get("last_success_at") or ""),
                str(x["repo_id"]).lower(),
            ),
            reverse=True,
        )
    else:  # latest
        candidates.sort(
            key=lambda x: (
                _quality_key(x),
                str(x.get("last_success_at") or ""),
                str(x["repo_id"]).lower(),
            ),
            reverse=True,
        )

    target_count = max(min_models, max_models)
    candidates = candidates[:target_count]
    for idx, item in enumerate(candidates):
        item["pool_index"] = idx
    return candidates, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaderboard-url", default=DEFAULT_LEADERBOARD_URL)
    ap.add_argument("--max-models", type=int, default=1000)
    ap.add_argument("--min-models", type=int, default=1000,
                    help="Minimum corpus size; fills with partial/empty leaderboard rows when needed")
    ap.add_argument("--sort", choices=["latest", "gain", "model"], default="latest")
    ap.add_argument("--pool-id", default="")
    ap.add_argument("--output", default="")
    ap.add_argument("--exclude-inferencex-models",
                    default="ci/inferenceX_models.yaml",
                    help="YAML mapping whose hf_model entries are excluded from this production pool")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    pool_id = args.pool_id or f"leaderboard-valid-{now.strftime('%Y-%m-%d')}"
    output = Path(
        args.output
        or f"ci/candidates/production_1000_from_leaderboard_{now.strftime('%Y-%m-%d')}.json"
    )

    rows = iter_leaderboard_rows(args.leaderboard_url)
    excluded_models = _load_inferencex_exclusions(Path(args.exclude_inferencex_models))
    candidates, stats = build_candidates(
        rows,
        max_models=args.max_models,
        min_models=args.min_models,
        sort_mode=args.sort,
        pool_id=pool_id,
        excluded_models=excluded_models,
    )
    if len(candidates) < args.min_models:
        print(
            f"WARN: only {len(candidates)} models found; requested at least {args.min_models}",
            file=sys.stderr,
        )

    payload = {
        "schema_version": "hyperloom.production_corpus.v1",
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "model-leaderboard",
        "leaderboard_url": args.leaderboard_url.rstrip("/"),
        "pool_id": pool_id,
        "policy": {
            "max_models": args.max_models,
            "min_models": args.min_models,
            "sort": args.sort,
            "primary_tier": "baseline_throughput > 0 and optimized_throughput > 0",
            "fallback_tiers": [
                "baseline_throughput > 0 or optimized_throughput > 0",
                "leaderboard row exists but numeric throughput is empty/zero",
            ],
            "excluded_inferencex_models": sorted(excluded_models),
            "refresh_mode": "manual",
        },
        "stats": {
            **stats,
            "candidates_written": len(candidates),
        },
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(candidates)} candidates)")
    print(json.dumps(payload["stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
