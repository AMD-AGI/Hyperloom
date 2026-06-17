#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Generate GitHub Actions matrix JSON from ci-config.yaml."""

import json
import os
import sys

import yaml


def _entry_key(m: dict) -> str:
    """Return the effective matrix/filter key for a model entry.

    Args:
        m: A model config entry.

    Returns:
        The explicit ``key`` field when set, otherwise ``inferenceX_key``.
    """
    return m.get("key") or m["inferenceX_key"]


def generate_matrix(config_path: str = "ci-config.yaml", selected_models: str = "") -> dict:
    """Build a GitHub Actions matrix from the CI config.

    Args:
        config_path: Path to ``ci-config.yaml``.
        selected_models: Optional comma-separated list of model keys to
            include; when empty, all configured models are used.

    Returns:
        A dict of the form ``{"include": [{"key": ...}, ...]}`` suitable for a
        GitHub Actions matrix.
    """
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
    """Generate the matrix and write it to the GitHub Actions output.

    Reads the model selection from the ``INPUT_MODELS`` environment variable,
    builds the matrix, and appends it to the file named by ``GITHUB_OUTPUT``.
    When that variable is unset, the matrix is pretty-printed to stdout instead.
    """
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
