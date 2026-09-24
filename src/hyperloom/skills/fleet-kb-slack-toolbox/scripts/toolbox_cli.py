#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI fallback for the importable Fleet KB Slack tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperloom_kb import SlackFleetTools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--scope-id", required=True)
    catalog.add_argument("--after", type=int, default=0)
    catalog.add_argument("--limit", type=int, default=100)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--scope-id", required=True)
    discover.add_argument("--actor-id", required=True)
    discover.add_argument("--thread-id", required=True)
    discover.add_argument("--message", required=True)
    discover.add_argument("--context-json", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--scope-id", required=True)
    verify.add_argument("--actor-id", required=True)
    verify.add_argument("--thread-id", required=True)
    verify.add_argument("--experience-id", action="append", required=True)
    verify.add_argument("--reason", required=True)

    verifications = subparsers.add_parser("verifications")
    verifications.add_argument("--scope-id", required=True)

    events = subparsers.add_parser("events")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()
    tools = SlackFleetTools.from_env()
    if args.command == "catalog":
        value = tools.list_kb_experiences(
            scope_id=args.scope_id,
            after=args.after,
            limit=args.limit,
        )
    elif args.command == "discover":
        context = json.loads(args.context_json.read_text(encoding="utf-8"))
        if not isinstance(context, dict):
            raise RuntimeError("context-json must contain one JSON object")
        value = tools.discover_kb_experiences(
            message=args.message,
            context=context,
            scope_id=args.scope_id,
            actor_id=args.actor_id,
            thread_id=args.thread_id,
        )
    elif args.command == "verify":
        value = tools.verify_kb_experiences_for_scope(
            experience_ids=tuple(args.experience_id),
            scope_id=args.scope_id,
            actor_id=args.actor_id,
            verification_reason=args.reason,
            thread_id=args.thread_id,
        )
    elif args.command == "verifications":
        value = tools.list_scope_verified_experiences(
            scope_id=args.scope_id,
        )
    else:
        value = tools.list_kb_events(
            after=args.after,
            limit=args.limit,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
