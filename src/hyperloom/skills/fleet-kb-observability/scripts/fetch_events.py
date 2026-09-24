#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fetch Fleet KB events without advancing a consumer cursor."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    base_url = os.environ["HYPERLOOM_FLEET_KB_URL"].rstrip("/")
    token = os.environ["HYPERLOOM_FLEET_KB_BOT_TOKEN"]
    fleet_id = os.environ.get(
        "HYPERLOOM_FLEET_KB_ID",
        "customer-demo",
    )
    request = urllib.request.Request(
        f"{base_url}/v1/events?after={args.after}&limit={args.limit}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Hyperloom-Fleet-ID": fleet_id,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
