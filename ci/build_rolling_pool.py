# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Merge production_1000 + new_models_to_run into one de-duplicated pool.

Outputs:
* production_1000_from_hf_2026-05-25.json: now the full merged 1714-model pool.
* manual_100.json: top-100 by downloads, reserved for manual workflow_dispatch.
* unrun_excluding_manual_100.json: current not-on-leaderboard subset, with the
  manual top-100 removed. "Ran" is determined from Pulse session-breakdowns,
  matching either full repo slug or basename slug because SaFE often writes
  model_name as the repo basename.

Run:  python3 ci/build_rolling_pool.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CI = Path(__file__).resolve().parent
CAND = CI / "candidates"

PROD = CAND / "production_1000_from_hf_2026-05-25.json"
NEWM = CI / "new_models_to_run.json"
MANUAL_OUT = CAND / "manual_100.json"
UNRUN_OUT = CAND / "unrun_excluding_manual_100.json"
MANUAL_N = 100
PULSE_BREAKDOWNS_URL = (
    "https://core42.primus-safe.amd.com/pulse/api/v1/session-breakdowns"
    "?show_on_leaderboard=false&order=desc&sort_by=updated_at"
)


sys.path.insert(0, str(CI))
from generate_hf_matrix import slugify  # noqa: E402


def _norm(repo: str | None) -> str:
    """Normalize a repo id for case-insensitive comparison.

    Args:
        repo: Repo id, possibly ``None``.

    Returns:
        The trimmed, lower-cased repo id (empty string when ``repo`` is falsy).
    """
    return (repo or "").strip().lower()


def _candidate_keys(repo_id: str) -> set[str]:
    """Build the set of slug keys a candidate may be matched by.

    Args:
        repo_id: Full Hugging Face repo id (``org/name``).

    Returns:
        Slugs for both the full repo id and its basename, since Pulse may store
        either form as ``model_name``.
    """
    repo_id = (repo_id or "").strip()
    return {slugify(repo_id), slugify(repo_id.split("/")[-1])}


def _pulse_model_keys() -> set[str]:
    """Fetch slug keys for every model already seen in Pulse breakdowns.

    Pages through the Pulse session-breakdowns API (with retries) and collects
    a slugified key per ``model_name``.

    Returns:
        The set of slugified model keys that have already been run.
    """
    keys: set[str] = set()
    offset = 0
    limit = 200
    total: int | None = None
    while True:
        url = f"{PULSE_BREAKDOWNS_URL}&limit={limit}&offset={offset}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=45) as r:
                    data = json.load(r)
                break
            except Exception:
                if attempt >= 3:
                    raise
                time.sleep(2 * (attempt + 1))
        rows = data.get("results") if isinstance(data, dict) else []
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            if isinstance(row, dict) and row.get("model_name"):
                keys.add(slugify(str(row["model_name"])))
        pagination = data.get("pagination") if isinstance(data, dict) else {}
        if isinstance(pagination, dict) and isinstance(pagination.get("total"), int):
            total = pagination["total"]
        offset += len(rows)
        if total is not None and offset >= total:
            break
    return keys


def main() -> int:
    """Merge the production and new-model pools and write the rolling outputs.

    Reads the curated production corpus and the new-models list, applies the
    production exclusion policy, de-duplicates by repo id, then writes the
    merged pool, the reserved manual top-100, and the unrun subset.

    Returns:
        Process exit code (``0`` on success).
    """
    prod = json.loads(PROD.read_text(encoding="utf-8"))
    newm = json.loads(NEWM.read_text(encoding="utf-8"))

    # Reuse production's curated exclusion policy so known-bad / unsupported
    # families (gpt-oss-120b, kimi-k2.5, deepseek-r1-0528, ...) never leak in
    # through the new_models side either.
    policy = prod.get("policy", {})
    excl_ids = {_norm(x) for x in policy.get("excluded_exact_ids", [])}
    excl_kw = [k.lower() for k in policy.get("exclusion_keywords", [])]

    def excluded(repo: str) -> bool:
        """Return whether a repo is filtered out by the exclusion policy.

        Args:
            repo: Candidate repo id.

        Returns:
            ``True`` when the repo is empty, exactly excluded, or matches an
            exclusion keyword.
        """
        k = _norm(repo)
        if not k:
            return True
        if k in excl_ids:
            return True
        return any(kw in k for kw in excl_kw)

    merged: dict[str, dict] = {}

    # production first — the curated, already-validated corpus wins on dedup.
    for c in prod.get("candidates", []):
        rid = c.get("repo_id")
        if not rid or excluded(rid):
            continue
        merged.setdefault(_norm(rid), {
            "repo_id": rid,
            "params_b": c.get("params_b"),
            "downloads": c.get("downloads") or 0,
            "pipeline_tag": c.get("pipeline_tag") or "text-generation",
            "source": "production_1000",
        })

    # new_models_to_run — convert num_parameters (raw count) -> params_b.
    for m in newm.get("models", []):
        rid = m.get("repo_id")
        if not rid or excluded(rid):
            continue
        k = _norm(rid)
        if k in merged:
            continue
        np_ = m.get("num_parameters")
        pb = round(np_ / 1e9, 3) if isinstance(np_, (int, float)) else None
        merged[k] = {
            "repo_id": rid,
            "params_b": pb,
            "downloads": m.get("downloads") or 0,
            "pipeline_tag": m.get("pipeline_tag") or "text-generation",
            "source": "new_models_to_run",
        }

    allc = sorted(merged.values(),
                  key=lambda x: x.get("downloads") or 0, reverse=True)
    for i, c in enumerate(allc):
        c["pool_index"] = i

    manual = allc[:MANUAL_N]
    manual_keys = {_norm(c.get("repo_id")) for c in manual}
    pulse_keys = _pulse_model_keys()
    unrun = [
        c for c in allc
        if _norm(c.get("repo_id")) not in manual_keys
        and not (_candidate_keys(c.get("repo_id") or "") & pulse_keys)
    ]
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def dump(path: Path, cands: list[dict], note: str) -> None:
        """Write a candidate pool to ``path`` as the standard corpus JSON.

        Args:
            path: Destination file; its stem is used as the ``pool_id``.
            cands: Candidate records to serialize.
            note: Human-readable description stored under the ``note`` key.
        """
        path.write_text(json.dumps({
            "schema_version": "hyperloom.production_corpus.v1",
            "pool_id": path.stem,
            "note": note,
            "generated_at": generated_at,
            "sort": "downloads desc",
            "sources": [PROD.name, NEWM.name],
            "count": len(cands),
            "candidates": cands,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dump(PROD, allc,
         "Merged full pool: production_1000 ∪ new_models_to_run, de-duped by "
         "repo_id and sorted by downloads desc. This replaces the previous "
         "1000-ish production pool as the main corpus.")
    dump(MANUAL_OUT, manual,
         f"Top-{MANUAL_N} by downloads, RESERVED for manual workflow_dispatch "
         "(operator triggers these by hand; NOT in the cron priority pool).")
    dump(UNRUN_OUT, unrun,
         "Current not-on-Pulse-session-breakdowns subset from the merged full "
         "pool, with manual_100 removed. Matching uses either full repo slug or "
         "repo basename slug because SaFE breakdown model_name often stores the "
         "basename. optimize-submit cron prioritizes this file and also passes "
         "exclude_leaderboard=true to avoid duplicate submits after models finish.")

    prod_n = len(prod.get("candidates", []))
    new_n = len(newm.get("models", []))
    print(f"production_1000     = {prod_n}")
    print(f"new_models_to_run   = {new_n}")
    print(f"pulse model keys    = {len(pulse_keys)}")
    print(f"merged_unique (post-exclusion) = {len(allc)}")
    print(f"  -> manual_100.json   = {len(manual)}")
    print(f"  -> {PROD.name} = {len(allc)}")
    print(f"  -> {UNRUN_OUT.name} = {len(unrun)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
