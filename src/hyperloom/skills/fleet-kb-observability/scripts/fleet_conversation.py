#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Invoke Fleet Catalog and run-scoped verification tools for Slack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperloom_kb import FleetBotClient, FleetBotConfig


def _client() -> FleetBotClient:
    config = FleetBotConfig.from_env()
    if config is None:
        raise RuntimeError("HYPERLOOM_FLEET_KB_URL is not configured")
    return FleetBotClient(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--scope-id", required=True)
    discover.add_argument("--actor-id", required=True)
    discover.add_argument("--thread-id", required=True)
    discover.add_argument("--message", required=True)
    discover.add_argument("--context-json", type=Path, required=True)

    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--scope-id", required=True)
    catalog.add_argument("--after", type=int, default=0)
    catalog.add_argument("--limit", type=int, default=100)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--scope-id", required=True)
    verify.add_argument("--actor-id", required=True)
    verify.add_argument("--thread-id", required=True)
    verify.add_argument("--experience-id", action="append", required=True)
    verify.add_argument("--reason", required=True)

    args = parser.parse_args()
    client = _client()
    if args.command == "discover":
        context = json.loads(args.context_json.read_text(encoding="utf-8"))
        if not isinstance(context, dict):
            raise RuntimeError("context-json must contain one JSON object")
        result = client.discover(
            args.message,
            context,
            scope_id=args.scope_id,
            actor_id=args.actor_id,
            thread_id=args.thread_id,
        )
        value = {
            "discovery_id": result.discovery_id,
            "scope_id": result.scope_id,
            "status": result.status,
            "candidates": list(result.candidates),
            "warnings": list(result.warnings),
        }
    elif args.command == "catalog":
        result = client.list_catalog(
            scope_id=args.scope_id,
            after=args.after,
            limit=args.limit,
        )
        value = {
            "scope_id": result.scope_id,
            "trust_state": result.trust_state,
            "items": list(result.items),
            "next_cursor": result.next_cursor,
            "has_more": result.has_more,
        }
    else:
        result = client.verify_for_run(
            tuple(args.experience_id),
            scope_id=args.scope_id,
            actor_id=args.actor_id,
            verification_reason=args.reason,
            thread_id=args.thread_id,
        )
        value = {
            "scope_id": result.scope_id,
            "status": result.status,
            "verifications": list(result.verifications),
            "created_count": result.created_count,
            "unchanged_experience_ids": list(result.unchanged_experience_ids),
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
