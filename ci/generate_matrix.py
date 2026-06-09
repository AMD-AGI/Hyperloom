#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Generate GitHub Actions matrix JSON from ci-config.yaml."""

import json
import os
import sys

import yaml


def _entry_key(m: dict) -> str:
    """Effective matrix/filter key: explicit `key` field overrides `inferenceX_key`."""
    return m.get("key") or m["inferenceX_key"]


def generate_matrix(config_path: str = "ci-config.yaml", selected_models: str = "") -> dict:
    # Force UTF-8: ci-config.yaml uses box-drawing chars (──); Windows cp1252 would raise UnicodeDecodeError.
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])

    selected = selected_models.strip()
    if selected:
        keys = set(selected.split(","))
        models = [m for m in models if _entry_key(m) in keys]

    matrix = [{"key": _entry_key(m)} for m in models]
    return {"include": matrix}


def main():
    selected = os.environ.get("INPUT_MODELS", "")
    matrix = generate_matrix(selected_models=selected)

    print(f"Models: {[m['key'] for m in matrix['include']]}", file=sys.stderr)

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"matrix={json.dumps(matrix)}\n")
    else:
        print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
