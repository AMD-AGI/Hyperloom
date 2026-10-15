# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A minimal two-workload SIKL bundle, shaped like an exported schema collection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DEFINITION_NAME = "gemm_demo_nt_n32_k64"
CASES = (("case-a", 1), ("case-b", 8))

# Stand-in for `sikl.benchmarks.api` that passes a candidate only when its source contains `passing`.
SOURCE_JUDGING_SIKL = """import json, pathlib, sys
bundle = pathlib.Path(sys.argv[1])
name = sys.argv[sys.argv.index("--solution") + 1]
(published,) = bundle.glob(f"solutions/**/{{name}}.json")
source = json.loads(published.read_text())["sources"][0]["content"]
status = "PASSED" if {passing!r} in source else "INCORRECT_NUMERICAL"
path = sys.argv[sys.argv.index("--json-report") + 1]
records = [{{"workload": {{"uuid": uuid, "axes": {{"m": rows}}}}, "evaluation": {{"status": status}}}}
           for uuid, rows in {cases!r}]
open(path, "w").write(json.dumps(records))
"""


def install_harness(root: Path, source: str) -> Path:
    """Plant `source` as the `sikl.benchmarks.api` a process with `root` on its PYTHONPATH runs."""
    package = root / "sikl" / "benchmarks"
    package.mkdir(parents=True, exist_ok=True)
    (root / "sikl" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api.py").write_text(source, encoding="utf-8")
    return root


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    _write(
        root / "definitions" / "gemm" / f"{DEFINITION_NAME}.json",
        {
            "schema_version": 2,
            "name": DEFINITION_NAME,
            "description": "16-bit gemm.",
            "op_type": "gemm",
            "axes": {
                "m": {"type": "var", "description": "rows"},
                "n": {"type": "const", "value": 32},
                "k": {"type": "const", "value": 64},
            },
            "inputs": {
                "a": {"shape": ["m", "k"], "dtype": "bfloat16"},
                "b": {"shape": ["n", "k"], "dtype": "bfloat16"},
            },
            "outputs": {"out": {"shape": ["m", "n"], "dtype": "bfloat16"}},
            "reference": "from __future__ import annotations\n\n\ndef run(a, b):\n    return a * b\n",
        },
    )
    _write(
        root / "solutions" / "baseline" / "gemm" / DEFINITION_NAME / f"{DEFINITION_NAME}__baseline.json",
        {
            "name": f"{DEFINITION_NAME}__baseline",
            "definition": DEFINITION_NAME,
            "author": "baseline",
            "spec": {
                "language": "python",
                "target": [{"arch": "gfx950", "hardware_id": "MI355X"}],
                "entry_point": "main.py::run",
                "destination_passing_style": False,
            },
            "sources": [{"path": "main.py", "content": "def run(a, b):\n    return a @ b.T\n"}],
        },
    )
    rows = [
        {"definition": DEFINITION_NAME, "workload": {"uuid": uuid, "axes": {"m": m, "n": 32, "k": 64}, "inputs": {}}}
        for uuid, m in CASES
    ]
    workloads = root / "workloads" / "gemm" / f"{DEFINITION_NAME}.jsonl"
    workloads.parent.mkdir(parents=True, exist_ok=True)
    workloads.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return root
