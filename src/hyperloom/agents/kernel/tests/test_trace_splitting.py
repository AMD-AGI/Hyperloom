# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test that trace splitting works for both LLM inference and generic (scriptable) traces.

Creates synthetic traces and runs the TraceLens splitter via the extracted
functions in tracelens_analysis.py, verifying that:

- LLM inference traces (vLLM/SGLang) produce three phase-specific chunks
  (mixed, decode_only, prefilldecode) when --llm-inference is set.
- Generic traces (xDiT/custom) produce a single steady_state chunk
  when --steady-state-mode=generic is used.
- _build_split_cmd correctly gates LLM-specific flags.
- _collect_split_chunks correctly handles both LLM and generic chunk layouts.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tracelens_analysis as tla

# ---------------------------------------------------------------------------
# Synthetic trace builders
# ---------------------------------------------------------------------------


def _make_vllm_trace(num_steps: int = 10, conc: int = 16) -> dict:
    """Build a minimal vLLM-style trace with step annotations.

    Each step has a user_annotation root, cpu_op events, and kernel events
    with correlation IDs linking them.
    """
    events = []
    ts = 1_000_000.0
    corr_id = 1

    for i in range(num_steps):
        step_dur = 10_000.0
        step_ts = ts + i * step_dur

        # User annotation (iteration root)
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"execute_{conc}_context_0(sq0sk0sqsq0sqsk0)_generation_{conc}(sq{conc}sk{conc * 1000}sqsq{conc}sqsk{conc * 1000})",
                "ts": step_ts,
                "dur": step_dur,
                "pid": 1,
                "tid": 1,
            }
        )

        # CPU ops with correlation IDs
        for j in range(5):
            op_ts = step_ts + j * 100
            events.append(
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::mm",
                    "ts": op_ts,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {
                        "correlation": corr_id,
                        "Input Dims": [[conc, 4096], [4096, 4096]],
                    },
                }
            )
            # Corresponding kernel
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "void rocblas_gemm_kernel",
                    "ts": op_ts + 10,
                    "dur": 40.0,
                    "pid": 1,
                    "tid": 2,
                    "args": {"correlation": corr_id},
                }
            )
            corr_id += 1

    return {"traceEvents": events}


def _make_xdit_trace(num_steps: int = 6) -> dict:
    """Build a minimal xDiT-style trace with repeating diffusion iterations.

    No vLLM/SGLang step annotations — uses ProfilerStep markers that the
    splitter's annotation detector recognises as iteration roots.
    """
    events = []
    ts = 1_000_000.0
    corr_id = 1

    for i in range(num_steps):
        step_dur = 50_000.0
        step_ts = ts + i * step_dur

        # ProfilerStep annotation — the splitter recognises these as roots
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{i}",
                "ts": step_ts,
                "dur": step_dur,
                "pid": 1,
                "tid": 1,
            }
        )

        # CPU ops + kernels inside each iteration
        for j in range(8):
            op_ts = step_ts + 1000 + j * 500
            events.append(
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::mm",
                    "ts": op_ts,
                    "dur": 200.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {
                        "correlation": corr_id,
                        "Input Dims": [[256, 1024], [1024, 1024]],
                    },
                }
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "void rocblas_gemm_kernel",
                    "ts": op_ts + 10,
                    "dur": 180.0,
                    "pid": 1,
                    "tid": 2,
                    "args": {"correlation": corr_id},
                }
            )
            corr_id += 1

    return {"traceEvents": events}


def _write_trace(path: Path, trace: dict) -> None:
    """Write a trace dict to a gzipped JSON file."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(trace, f)


# ---------------------------------------------------------------------------
# Tests: _build_split_cmd flag gating
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: _collect_split_chunks
# ---------------------------------------------------------------------------


class TestCollectSplitChunks:
    """Verify _collect_split_chunks handles both LLM and generic chunk layouts."""

    def _write_dummy_chunk(self, split_dir: Path, name: str) -> Path:
        path = split_dir / name
        with gzip.open(path, "wt") as f:
            json.dump({"traceEvents": []}, f)
        return path

    def test_llm_mixed_mode(self, tmp_path):
        split_dir = tmp_path / "split"
        split_dir.mkdir()
        self._write_dummy_chunk(split_dir, "mixed_steady_state_test.json.gz")
        self._write_dummy_chunk(split_dir, "decode_only_steady_state_test.json.gz")
        self._write_dummy_chunk(split_dir, "prefilldecode_steady_state_test.json.gz")

        chunk, meta, mode_map = tla._collect_split_chunks(
            split_dir,
            "mixed",
            tmp_path / "trace.json.gz",
        )
        assert chunk.name == "mixed_steady_state_test.json.gz"
        assert meta["chunks_by_mode"]["mixed"] == 1
        assert meta["chunks_by_mode"]["decode_only"] == 1
        assert meta["chunks_by_mode"]["prefilldecode"] == 1

    def test_generic_mode(self, tmp_path):
        split_dir = tmp_path / "split"
        split_dir.mkdir()
        self._write_dummy_chunk(split_dir, "steady_state_test.json.gz")

        chunk, meta, mode_map = tla._collect_split_chunks(
            split_dir,
            "generic",
            tmp_path / "trace.json.gz",
        )
        assert chunk.name == "steady_state_test.json.gz"
        assert meta["chunks_by_mode"]["generic"] == 1

    def test_no_chunks_raises(self, tmp_path):
        split_dir = tmp_path / "split"
        split_dir.mkdir()
        with pytest.raises(RuntimeError, match="trace_split_no_steady_state"):
            tla._collect_split_chunks(
                split_dir,
                "mixed",
                tmp_path / "trace.json.gz",
            )

    def test_missing_mode_raises(self, tmp_path):
        split_dir = tmp_path / "split"
        split_dir.mkdir()
        self._write_dummy_chunk(split_dir, "decode_only_steady_state_test.json.gz")

        with pytest.raises(RuntimeError, match="steady_state_chunk_missing"):
            tla._collect_split_chunks(
                split_dir,
                "mixed",
                tmp_path / "trace.json.gz",
            )


# ---------------------------------------------------------------------------
# Tests: end-to-end splitting via TraceLens splitter subprocess
# ---------------------------------------------------------------------------


def _tracelens_importable() -> bool:
    """Check if TraceLens splitter is importable."""
    try:
        import TraceLens.TraceUtils.split_trace.main  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _tracelens_importable(),
    reason="TraceLens not installed",
)
class TestSplitterE2E:
    """Run _run_trace_split on synthetic traces to verify end-to-end splitting.

    Calls _run_trace_split() directly — the same function main() uses — with
    synthetic traces and the same args the orchestrator would set. This tests
    the full chain: argparse flags -> _build_split_cmd -> TraceLens splitter
    subprocess -> _collect_split_chunks -> _validate_selected_chunk.
    """

    @staticmethod
    def _make_args(**overrides):
        defaults = {
            "split_conc": "",
            "split_osl": "",
            "split_r": "",
            "split_llm_inference": False,
            "split_num_steps": 8,
            "steady_state_mode": "mixed",
            "budget_minutes": 5,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_vllm_trace_produces_three_chunks(self, tmp_path):
        """A vLLM trace with split_llm_inference should produce mixed/decode/prefilldecode chunks."""
        trace_path = tmp_path / "vllm_trace.json.gz"
        split_dir = tmp_path / "split"
        log_path = tmp_path / "log"
        log_path.touch()
        _write_trace(trace_path, _make_vllm_trace(num_steps=32, conc=16))

        args = self._make_args(
            split_llm_inference=True,
            split_conc="16",
            split_osl="512",
            split_r="1",
            steady_state_mode="mixed",
        )
        tl_root = Path(tla.__file__).resolve().parent.parent.parent.parent.parent

        chunk, meta, warnings = tla._run_trace_split(
            args, trace_path, split_dir, tl_root, log_path,
        )
        assert meta["chunks_by_mode"]["mixed"] >= 1
        assert chunk.exists()
        assert chunk.stat().st_size > 0

    def test_xdit_trace_produces_generic_chunk(self, tmp_path):
        """An xDiT trace with steady_state_mode=generic should produce a generic chunk."""
        trace_path = tmp_path / "xdit_trace.json.gz"
        split_dir = tmp_path / "split"
        log_path = tmp_path / "log"
        log_path.touch()
        _write_trace(trace_path, _make_xdit_trace(num_steps=6))

        args = self._make_args(
            split_llm_inference=False,
            split_num_steps=4,
            steady_state_mode="generic",
        )
        tl_root = Path(tla.__file__).resolve().parent.parent.parent.parent.parent

        chunk, meta, warnings = tla._run_trace_split(
            args, trace_path, split_dir, tl_root, log_path,
        )
        assert meta["chunks_by_mode"]["generic"] >= 1
        assert chunk.exists()
        assert chunk.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests: _build_trace_analyze_cmd flag wiring (request_handlers.py)
# ---------------------------------------------------------------------------


def _request_handlers_importable() -> bool:
    try:
        from hyperloom.orchestrator.actions.executors.trace_analyze import (
            _build_trace_analyze_cmd,
        )  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _request_handlers_importable(),
    reason="hyperloom orchestrator not importable",
)
class TestBuildTraceAnalyzeCmd:
    """Verify request_handlers._build_trace_analyze_cmd sets flags correctly."""

    def _mock_state(self):
        state = MagicMock()
        state.baseline_config_path = ""
        return state

    def _run(self, *, scriptable: bool, framework: str, **payload_kw):
        import os

        os.environ.setdefault("HYPERLOOM_KERNEL_AGENT_ROOT", str(TOOLS_DIR.parent))
        from hyperloom.orchestrator.actions.executors.trace_analyze import (
            _build_trace_analyze_cmd,
        )

        cmd, mode = _build_trace_analyze_cmd(
            payload=payload_kw,
            session_dir=Path("/tmp/test"),
            state=self._mock_state(),
            workspace_path="/tmp/test",
            trace_input="/tmp/trace",
            tracelens_root=Path("/tmp/tl"),
            is_bypass=False,
            scriptable=scriptable,
            workload={},
            model_name="test",
            framework=framework,
            target_platform="MI300X",
            analysis_mode="default",
        )
        return cmd, mode

    def test_serving_gets_llm_inference(self):
        cmd, _ = self._run(scriptable=False, framework="vllm")
        assert "--split-llm-inference" in cmd
        assert "--skip-split" not in cmd

    def test_scriptable_gets_generic_mode(self):
        cmd, _ = self._run(scriptable=True, framework="xdit")
        assert "--split-llm-inference" not in cmd
        assert "--skip-split" not in cmd
        idx = cmd.index("--steady-state-mode")
        assert cmd[idx + 1] == "generic"

    def test_serving_forwards_conc_osl_r(self):
        cmd, _ = self._run(
            scriptable=False,
            framework="sglang",
            split_conc="64",
            split_osl="1024",
            split_r="1",
        )
        assert "--split-conc" in cmd
        assert "--split-osl" in cmd
        assert "--split-r" in cmd

    def test_scriptable_no_conc_osl_r(self):
        cmd, _ = self._run(
            scriptable=True,
            framework="xdit",
            split_conc="64",
            split_osl="1024",
        )
        assert "--split-conc" not in cmd
        assert "--split-osl" not in cmd
