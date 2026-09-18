#!/usr/bin/env python3
"""Review SBD V6 Framework attempts for canonical Experience fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.experience_v1 import (
    build_framework_experience_review,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Session directory or session_breakdown.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.session
    breakdown_path = source if source.is_file() else source / "session_breakdown.json"
    session_dir = breakdown_path.parent
    review = build_framework_experience_review(session_dir, _load(breakdown_path))
    encoded = json.dumps(review, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
