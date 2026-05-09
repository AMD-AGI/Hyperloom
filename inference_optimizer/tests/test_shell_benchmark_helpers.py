"""Regression tests for shell benchmark post-processing helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


COMMON_SH = (
    Path(__file__).resolve().parents[2]
    / ".cursor"
    / "skills"
    / "inference-optimization"
    / "scripts"
    / "common.sh"
)


def test_shell_helpers_tolerate_missing_optional_summary_fields(tmp_path):
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps({
        "output_throughput": 1872.0,
        "completed": 320,
        "num_prompts": 320,
    }))
    tsv = tmp_path / "results.tsv"

    script = (
        f"source {COMMON_SH} && "
        f"benchmark_json_valid {result_json} && "
        f"print_benchmark_summary {result_json} sglang && "
        f"append_sweep_row {result_json} {tsv} sglang 64 1024 1024 320"
    )
    proc = subprocess.run(
        ["bash", "-lc", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Output throughput: 1872.00 tok/s" in proc.stdout
    assert "swept" in tsv.read_text()
