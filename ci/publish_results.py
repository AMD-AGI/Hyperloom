#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Publish normalized Hyperloom results to the results service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Public ingress URL for the hyperloom-results-service (deployed in the
# primus-claw-dev namespace on the core42 data-plane cluster). Higress rewrites
# /hyperloom-results/* -> backend /*, so /api/import becomes
# /hyperloom-results/api/import here. We default to the ingress URL (not the
# cluster-internal service DNS the helm chart exposes) so callers running on
# any GHA runner — public ubuntu-latest or the hyperloom-mh826 self-hosted
# pod — can reach the service without extra config. Override via
# HYPERLOOM_RESULTS_SERVICE_URL when targeting a different deployment.
DEFAULT_SERVICE_URL = "http://core42.primus-safe.amd.com/hyperloom-results"


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    if path.suffix == ".ndjson":
        results: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    results.append(item)
        return results

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [r for r in payload["results"] if isinstance(r, dict)]
    if isinstance(payload, dict) and payload.get("schema_version"):
        return [payload]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _normalize_submitted_at(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move ``run.submitted_at`` ISO strings into ``run.submitted_at_iso``
    and set the original field to ``None``.

    Required because the hyperloom-results-service ingest path declares
    ``submitted_at`` as a ``timestamptz`` column and binds the value via
    a helper that returns ``str``; asyncpg rejects this and the POST
    returns HTTP 500 ``invalid input for query argument $9 (expected a
    datetime.date or datetime.datetime instance, got 'str')``.

    Keeping the raw ISO value under a sibling key means the information
    still lands in the row's ``raw_result`` JSONB blob, so consumers can
    reconstruct the timestamp without a backend round-trip. Drop this
    function once the ingest path accepts ISO strings directly.
    """
    cleaned: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            cleaned.append(r)
            continue
        rr = dict(r)
        run = dict(rr.get("run") or {})
        sa = run.get("submitted_at")
        if isinstance(sa, str) and sa.strip():
            run["submitted_at_iso"] = sa
            run["submitted_at"] = None
        rr["run"] = run
        cleaned.append(rr)
    return cleaned


def publish(
    results: list[dict[str, Any]],
    url: str,
    token: str = "",
    timeout: int = 60,
    max_retries: int = 5,
    initial_backoff_s: float = 5.0,
) -> dict:
    """POST results to /api/import with exponential-backoff retry.

    Retries cover the two known intermittent failure modes on the
    hyperloom-results-service side:
      * PG pod crashloop drops the service's connection pool — first
        request after the restart returns ConnectionDoesNotExistError
        (HTTP 500 with empty body), the next one usually succeeds once
        asyncpg re-establishes a connection.
      * Postgres liveness-probe restarts can briefly make the whole
        endpoint unreachable (HTTP 5xx / connection refused).
    """
    import time
    import requests

    endpoint = url.rstrip("/") + "/api/import"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"results": _normalize_submitted_at(results)}

    backoff = initial_backoff_s
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                endpoint, headers=headers, json=body, timeout=timeout,
            )
            # 5xx is the only thing worth retrying — 4xx means our payload
            # is wrong and retrying won't help.
            if 500 <= resp.status_code < 600:
                snippet = (resp.text or "")[:200].replace("\n", " ")
                err = RuntimeError(
                    f"HTTP {resp.status_code} from {endpoint}: {snippet}"
                )
                last_err = err
                print(
                    f"publish attempt {attempt}/{max_retries} failed: {err} "
                    f"(retrying after {backoff:.0f}s)",
                    flush=True,
                )
            else:
                resp.raise_for_status()
                return resp.json()
        except requests.RequestException as e:
            last_err = e
            print(
                f"publish attempt {attempt}/{max_retries} network error: "
                f"{e!r} (retrying after {backoff:.0f}s)",
                flush=True,
            )

        if attempt < max_retries:
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    raise RuntimeError(
        f"publish failed after {max_retries} retries: {last_err!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="normalized_results.json or .ndjson")
    parser.add_argument(
        "--url",
        default=os.environ.get("HYPERLOOM_RESULTS_SERVICE_URL", DEFAULT_SERVICE_URL),
        help="Results service base URL (defaults to the core42 ingress)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HYPERLOOM_RESULTS_SERVICE_TOKEN", ""),
        help="Optional bearer token for the results service",
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not args.url:
        print("HYPERLOOM_RESULTS_SERVICE_URL is not set; skipping publish")
        return 0

    results = load_results(Path(args.input))
    if not results:
        print(f"no results found in {args.input}; skipping publish")
        return 0

    response = publish(results, args.url, args.token, args.timeout)
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
