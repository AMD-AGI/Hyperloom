#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: aiperf ``profile_export_aiperf.json`` -> ``inferencex_result.json``.

Runs from the benchmarks dir, where hyperloom is usually not installed;
``deploy_agentx_assets`` publishes ``agentx/mapping.py`` beside it as ``agentx_mapping.py``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_mapping import map_aiperf


def _noncanonical_reasons():
    """Workload deviations the client detected; see aiperf_client.sh."""
    raw = (os.environ.get("AGENTX_NONCANONICAL_REASONS") or "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else []


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    res = map_aiperf(data, noncanonical_reasons=_noncanonical_reasons())
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
