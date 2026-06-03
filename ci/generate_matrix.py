#!/usr/bin/env python3
"""Generate GitHub Actions matrix JSON from ci-config.yaml."""

import json
import os
import sys

import yaml


def _entry_key(m: dict) -> str:
    """Effective matrix/filter key: explicit `key` field overrides `inferenceX_key`.

    Two reasons an entry sets its own `key` instead of inheriting from
    `inferenceX_key`:

    1. Self-contained entries with no upstream amd-master baseline. The GLM-5
       multi-node entry (`key: glm5-multinode-fp8-mi300x-sglang`) is the
       canonical case — `inferenceX_parser.synthesize_entry_from_ci_config`
       builds the lookup row from the ci-config fields themselves.
    2. Two ci-config entries sharing the same `inferenceX_key` (e.g. variants
       of the same amd-master row) need unique matrix display names to avoid
       dedupe.

    Args:
        m (dict): A single ci-config ``models`` entry.

    Returns:
        str: The explicit ``key`` field if present, otherwise ``inferenceX_key``.
    """
    return m.get("key") or m["inferenceX_key"]


def generate_matrix(config_path: str = "ci-config.yaml", selected_models: str = "") -> dict:
    """Build the GitHub Actions matrix include list from a ci-config file.

    Loads the YAML config, optionally filters the models down to a
    comma-separated selection, and emits one matrix entry per remaining model.

    Args:
        config_path (str): Path to the ci-config YAML file to read.
        selected_models (str): Optional comma-separated list of model keys to
            include. When empty, all configured models are used.

    Returns:
        dict: A matrix mapping of the form ``{"include": [{"key": ...}, ...]}``.
    """
    # Force UTF-8 — ci-config.yaml uses box-drawing chars (──) in section
    # comments. Linux runners default to UTF-8, but Windows defaults to
    # cp1252 which raises UnicodeDecodeError on byte 0x9d.
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
