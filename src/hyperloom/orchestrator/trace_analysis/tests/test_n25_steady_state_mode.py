# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""N25 — TraceLens splitter steady-state chunk selection contract."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest


_TL_MODULE = "hyperloom.orchestrator.trace_analysis.tracelens_analysis"


@pytest.fixture(scope="module")
def tl_module():
    """Import tracelens_analysis as a module without executing main()."""
    from hyperloom.orchestrator.trace_analysis import tracelens_analysis

    return tracelens_analysis


def _write_exec_details(
    split_dir: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Write a minimal execution_details.csv matching TraceLens splitter output."""
    path = split_dir / "execution_details.csv"
    cols = [
        "idx",
        "output_path",
        "event_count",
        "num_gpu_events",
        "gpu_duration",
        "gpu_busy_duration",
        "phase_num_prefill",
        "phase_num_prefilldecode",
        "phase_num_decode",
        "phase_avg_bs",
        "phase_avg_conc",
        "num_steps",
    ]
    with path.open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            full = dict.fromkeys(cols, "")
            full.update({k: str(v) for k, v in row.items()})
            w.writerow(full)
    return path


def _make_chunk_files(split_dir: Path) -> dict[str, Path]:
    """Create empty placeholder chunk files (the gate inspects the CSV, not trace contents)."""
    chunks = {}
    for label in (
        "mixed_steady_state",
        "decode_only_steady_state",
        "prefilldecode_steady_state",
    ):
        p = split_dir / f"{label}_chunk.trace.json.gz"
        p.write_bytes(b"")
        chunks[label] = p
    return chunks


@pytest.fixture
def split_dir(tmp_path):
    return tmp_path / "trace_split"


def _run_help():
    """Invoke `python -m hyperloom.orchestrator.trace_analysis.tracelens_analysis --help` and return stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", _TL_MODULE, "--help"],
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr


def test_cli_flag_appears_in_help():
    """--steady-state-mode is wired into argparse and documented."""
    out = _run_help()
    assert "--steady-state-mode" in out
    # All three choices visible in usage/help.
    assert "mixed" in out
    assert "decode_only" in out
    assert "prefilldecode" in out


def test_cli_rejects_unknown_mode():
    """argparse choices=() must reject random strings."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            _TL_MODULE,
            "--trace-input",
            "/tmp/does-not-exist",
            "--workspace-path",
            "/tmp",
            "--steady-state-mode",
            "garbage_mode",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    err = proc.stderr or ""
    assert "garbage_mode" in err
    assert "choose from" in err or "invalid choice" in err
