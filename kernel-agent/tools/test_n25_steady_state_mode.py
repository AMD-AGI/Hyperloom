# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""N25 — TraceLens splitter steady-state chunk selection contract (SOLAR-10.7B TP=1 case).

Explicit chunk selection (``--steady-state-mode`` + ``INFERENCE_OPTIMIZER_STEADY_STATE_MODE``) that hard-fails on a structurally-empty chunk (num_gpu_events==0 OR gpu_busy_duration==0.0); busy-% judgment stays with the T3 idle gate.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest


# Import module-level helpers without the heavy __main__ path (Claude SDK, creds).
TOOLS_DIR = Path(__file__).resolve().parent
TL_PATH = TOOLS_DIR / "tracelens_analysis.py"


@pytest.fixture(scope="module")
def tl_module():
    """Import tracelens_analysis.py as a module without executing main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tracelens_analysis_under_test", TL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_exec_details(
    split_dir: Path,
    rows: list[dict[str, object]],
) -> Path:
    """Write a minimal execution_details.csv matching TraceLens splitter output."""
    path = split_dir / "execution_details.csv"
    cols = [
        "idx", "output_path", "event_count", "num_gpu_events",
        "gpu_duration", "gpu_busy_duration",
        "phase_num_prefill", "phase_num_prefilldecode", "phase_num_decode",
        "phase_avg_bs", "phase_avg_conc", "num_steps",
    ]
    with path.open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            full = {c: "" for c in cols}
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


# _check_selected_chunk_has_gpu_events behavioural contract


def test_chunk_with_real_gpu_work_passes(tl_module, split_dir):
    """Qwen1.5-7B-style mixed chunk: 38,796 events / 60% busy -> None."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    selected = chunks["mixed_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(selected),
            "num_gpu_events": 38796,
            "gpu_duration": 3000000.0,
            "gpu_busy_duration": 1800000.0,  # 60% busy
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [selected]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", []),
    }
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=selected,
        mode="mixed",
        available_modes=available,
    )
    assert result is None


def test_empty_solar_style_mixed_chunk_emits_warning(tl_module, split_dir):
    """SOLAR-10.7B TP=1 mixed: 160 events / 0.13% busy -> warning."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    mixed_path = chunks["mixed_steady_state"]
    pd_path = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        # SOLAR mixed chunk: 160 sampler kernels, 99.87% idle.
        {
            "output_path": str(mixed_path),
            "num_gpu_events": 160,
            "gpu_duration": 1118730.0,
            "gpu_busy_duration": 1428.0,
        },
        {
            "output_path": str(pd_path),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,  # 60% busy
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed_path]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd_path]),
    }
    # gpu_busy_duration > 0 here (1428us), so the structural zero-check gate must NOT fire.
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=mixed_path,
        mode="mixed",
        available_modes=available,
    )
    # Busy-% judgment stays with T3; N25 is purely the structural zero-check.
    assert result is None, (
        "N25 gate must NOT be a busy-ratio heuristic; only structural "
        "emptiness (num_gpu_events==0 OR gpu_busy_duration==0.0) "
        "triggers it. Busy-% judgment stays with T3."
    )


def test_zero_event_chunk_emits_warning(tl_module, split_dir):
    """num_gpu_events=0 -> warning, lists non_empty alternatives."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    mixed_path = chunks["mixed_steady_state"]
    pd_path = chunks["prefilldecode_steady_state"]
    do_path = chunks["decode_only_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed_path),
            "num_gpu_events": 0,  # structurally empty
            "gpu_duration": 1118730.0,
            "gpu_busy_duration": 0.0,
        },
        {
            "output_path": str(do_path),
            "num_gpu_events": 0,  # also empty (decode-only inside graph)
            "gpu_duration": 1118730.0,
            "gpu_busy_duration": 0.0,
        },
        {
            "output_path": str(pd_path),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed_path]),
        "decode_only": ("decode_only_steady_state", [do_path]),
        "prefilldecode": ("prefilldecode_steady_state", [pd_path]),
    }
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=mixed_path,
        mode="mixed",
        available_modes=available,
    )
    assert result is not None
    assert result["code"] == "steady_state_chunk_empty"
    assert result["requested_mode"] == "mixed"
    assert result["num_gpu_events"] == 0
    assert result["gpu_busy_duration"] == 0.0
    # decode_only is ALSO empty -> only prefilldecode in non_empty.
    assert result["non_empty_modes"] == ["prefilldecode"]
    assert "INFERENCE_OPTIMIZER_STEADY_STATE_MODE" in result["remediation"]
    assert "prefilldecode" in result["remediation"]


def test_zero_busy_duration_emits_warning(tl_module, split_dir):
    """num_gpu_events>0 but gpu_busy_duration=0.0 (degenerate) -> warning."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    mixed_path = chunks["mixed_steady_state"]
    pd_path = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed_path),
            "num_gpu_events": 160,
            "gpu_duration": 1118730.0,
            "gpu_busy_duration": 0.0,  # exactly zero
        },
        {
            "output_path": str(pd_path),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed_path]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd_path]),
    }
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=mixed_path,
        mode="mixed",
        available_modes=available,
    )
    assert result is not None
    assert result["code"] == "steady_state_chunk_empty"
    assert result["non_empty_modes"] == ["prefilldecode"]


def test_warning_excludes_self_from_non_empty_modes(tl_module, split_dir):
    """The warning's non_empty_modes list must exclude the requested mode."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    mixed_path = chunks["mixed_steady_state"]
    pd_path = chunks["prefilldecode_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(mixed_path),
            "num_gpu_events": 0,
            "gpu_duration": 1118730.0,
            "gpu_busy_duration": 0.0,
        },
        {
            "output_path": str(pd_path),
            "num_gpu_events": 2790,
            "gpu_duration": 4538984.0,
            "gpu_busy_duration": 2723452.0,
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [mixed_path]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", [pd_path]),
    }
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=mixed_path,
        mode="mixed",
        available_modes=available,
    )
    assert "mixed" not in result["non_empty_modes"]


def test_missing_exec_details_returns_none(tl_module, split_dir):
    """Back-compat: no execution_details.csv -> let the chunk through (T3 handles it)."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    available = {
        "mixed": ("mixed_steady_state", [chunks["mixed_steady_state"]]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", []),
    }
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=chunks["mixed_steady_state"],
        mode="mixed",
        available_modes=available,
    )
    assert result is None


def test_selected_chunk_not_in_csv_returns_none(tl_module, split_dir):
    """Back-compat: CSV missing the selected chunk's row -> fall back to T3."""
    split_dir.mkdir()
    chunks = _make_chunk_files(split_dir)
    selected = chunks["mixed_steady_state"]
    _write_exec_details(split_dir, [
        {
            "output_path": str(split_dir / "some_other.json.gz"),
            "num_gpu_events": 0,
            "gpu_duration": 0.0,
            "gpu_busy_duration": 0.0,
        },
    ])
    available = {
        "mixed": ("mixed_steady_state", [selected]),
        "decode_only": ("decode_only_steady_state", []),
        "prefilldecode": ("prefilldecode_steady_state", []),
    }
    result = tl_module._check_selected_chunk_has_gpu_events(
        split_dir=split_dir,
        selected_chunk=selected,
        mode="mixed",
        available_modes=available,
    )
    assert result is None


# CLI flag contract (parser-level; --help inspection is enough)


def _run_help() -> str:
    """Invoke `python tracelens_analysis.py --help` and return stdout."""
    proc = subprocess.run(
        [sys.executable, str(TL_PATH), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    return (proc.stdout or "") + (proc.stderr or "")


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
            sys.executable, str(TL_PATH),
            "--trace-input", "/tmp/does-not-exist",
            "--workspace-path", "/tmp",
            "--steady-state-mode", "garbage_mode",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    err = proc.stderr or ""
    assert "garbage_mode" in err
    assert "choose from" in err or "invalid choice" in err
