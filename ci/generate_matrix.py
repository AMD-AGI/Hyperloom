#!/usr/bin/env python3
"""Generate GitHub Actions matrix JSON from ci-config.yaml."""

import json
import os
import sys

import yaml


def generate_matrix(config_path: str = "ci-config.yaml", selected_models: str = "") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])

    selected = selected_models.strip()
    if selected:
        keys = set(selected.split(","))
        models = [m for m in models if m["inferenceX_key"] in keys]

    matrix = [{"key": m["inferenceX_key"]} for m in models]
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
