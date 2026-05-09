#!/usr/bin/env python3
"""Generate GitHub Actions matrix JSON from HuggingFace top-N or explicit list.

Inputs (env vars, set by the workflow):
  INPUT_MODELS    Space-separated HF repo IDs (overrides INPUT_HF_TOP if set)
  INPUT_HF_TOP    Top-N text-gen models to fetch (default: 5)
  INPUT_MIN_PARAMS Filter to >= B billion params (default: 7)
  HF_TOKEN        Optional, for gated repo metadata access

Output:
  GITHUB_OUTPUT receives `matrix={"include": [...]}` where each entry is
  {"model": "<repo_id>", "key": "<safe-slug>"}. The slug is used as the matrix
  display name and the per-task artifact suffix (avoids '/' in artifact names).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Reuse HuggingFaceClient from the existing script — single source of truth
# for the pool-then-filter logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_submit import HuggingFaceClient   # noqa: E402


def slugify(repo_id: str) -> str:
    """Turn 'Qwen/Qwen3-8B' into 'qwen-qwen3-8b' for use as artifact key."""
    s = repo_id.lower().replace("/", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def collect_repos() -> list[str]:
    explicit = (os.environ.get("INPUT_MODELS") or "").strip()
    if explicit:
        # Whitespace OR comma separated, both supported (workflow_dispatch UI is
        # space-friendly; downstream callers may comma-list).
        repos = re.split(r"[\s,]+", explicit)
        return [r for r in repos if r]

    hf_top = int(os.environ.get("INPUT_HF_TOP") or "5")
    min_params = float(os.environ.get("INPUT_MIN_PARAMS") or "7")
    hf = HuggingFaceClient(os.environ.get("HF_TOKEN", ""))
    print(f"fetching HF top-{hf_top} (>={min_params}B)...", file=sys.stderr)
    return hf.top_models(hf_top, min_params_b=min_params)


def main() -> int:
    repos = collect_repos()
    if not repos:
        print("no models selected — empty matrix", file=sys.stderr)
        # GitHub Actions errors on empty matrix; emit a sentinel that the
        # downstream job can detect and skip.
        matrix = {"include": []}
    else:
        matrix = {"include": [{"model": r, "key": slugify(r)} for r in repos]}

    print(json.dumps(matrix, indent=2), file=sys.stderr)

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"matrix={json.dumps(matrix)}\n")
            f.write(f"count={len(matrix['include'])}\n")
    else:
        print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
