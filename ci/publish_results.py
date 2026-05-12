#!/usr/bin/env python3
"""Publish normalized Hyperloom results to the results service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


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


def publish(results: list[dict[str, Any]], url: str, token: str = "", timeout: int = 60) -> dict:
    import requests

    endpoint = url.rstrip("/") + "/api/import"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(
        endpoint,
        headers=headers,
        json={"results": results},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="normalized_results.json or .ndjson")
    parser.add_argument(
        "--url",
        default=os.environ.get("HYPERLOOM_RESULTS_SERVICE_URL", ""),
        help="Results service base URL, e.g. http://hyperloom-results-service",
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
