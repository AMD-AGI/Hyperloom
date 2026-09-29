#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: MLPerf harness ``result_summary.json`` (+ ``scores.json``) -> ``inferencex_result.json``."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_mapping import map_mlperf


def _required_env(name):
    value = (os.environ.get(name) or "").strip()
    if not value:
        sys.exit(f"map_mlperf: {name} is not set; the AgentX switch exports it")
    return value


def main(src, dst, scores_src=None):
    with open(src, encoding="utf-8") as handle:
        report = json.load(handle)
    scores = None
    if scores_src:
        with open(scores_src, encoding="utf-8") as handle:
            scores = json.load(handle)
    result = map_mlperf(
        report,
        corpus=os.path.splitext(os.path.basename(_required_env("AGENTIC_DATASET_PATH")))[0],
        scores=scores,
    )
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
