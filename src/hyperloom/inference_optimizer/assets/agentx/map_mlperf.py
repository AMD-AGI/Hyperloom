#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: MLPerf harness ``result_summary.json`` -> ``inferencex_result.json``."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_mapping import map_mlperf


def _noncanonical_reasons():
    raw = (os.environ.get("AGENTX_NONCANONICAL_REASONS") or "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else []


def _load_optional(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(src, dst, accuracy_src=None):
    with open(src, encoding="utf-8") as handle:
        summary = json.load(handle)
    accuracy = _load_optional(accuracy_src)
    result = map_mlperf(summary, accuracy=accuracy, noncanonical_reasons=_noncanonical_reasons())
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
