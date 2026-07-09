###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replay test: run the bypass CLI against a real profiler trace.

CI-safe: the large trace is not shipped in-repo, so this test is skipped
unless a trace path is provided via ``HYPERLOOM_BYPASS_REPLAY_TRACE`` (or the
known local dev path exists). When it runs it asserts the downstream artifact
contract and the golden ranking (attention/SDPA is the top GPU-time kernel for
the reference vLLM Llama session).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bypass_trace_analysis as bta  # noqa: E402

_DEFAULT_DEV_TRACE = "/tmp/bp_trace/profile_inferencex_result.trace.json.gz"


def _resolve_trace() -> str | None:
    env = os.environ.get("HYPERLOOM_BYPASS_REPLAY_TRACE", "").strip()
    if env and Path(env).is_file():
        return env
    if Path(_DEFAULT_DEV_TRACE).is_file():
        return _DEFAULT_DEV_TRACE
    return None


def test_replay_real_trace_contract_and_ranking(tmp_path, capsys, monkeypatch):
    trace = _resolve_trace()
    if not trace:
        pytest.skip("no replay trace (set HYPERLOOM_BYPASS_REPLAY_TRACE)")
    monkeypatch.delenv("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", raising=False)

    rc = bta.main([
        "--trace-input", trace,
        "--session-id", "replay",
        "--workspace-path", str(tmp_path),
        "--framework", "vllm",
        "--target-platform", "MI300X",
        "--model-name", "NousResearch-Llama-2-7b-hf",
        "--top-k", "12",
    ])
    assert rc == 0
    out = capsys.readouterr()
    lines = [ln for ln in out.out.splitlines() if ln.strip()]
    result = json.loads(lines[-1])

    assert result["status"] == "ok"
    assert result["aggregation_scope"] == "full_trace"
    hot = result["hot_kernels"]
    assert hot, "expected non-empty hot kernels from a real trace"

    # Artifact contract present + on disk.
    for key in ("kernel_candidates", "kernel_roofline", "tracelens_summary", "trace_input_manifest", "trace_report_path"):
        p = result["artifact_paths"][key]
        assert p and Path(p).is_file(), f"missing artifact {key}: {p}"

    # Golden ranking: attention (SDPA) dominates GPU time -> top candidate.
    assert hot[0]["kernel_category"] == "SDPA", hot[0]["kernel_category"]
    assert hot[0]["gpu_pct"] > 10.0

    # Vendor GEMM (Cijk / scaled_mm) present but routed as non-reusable.
    gemm = [k for k in hot if k["kernel_category"] == "GEMM"]
    assert gemm, "expected GEMM kernels"
    assert all(not k["reusable_native_kernel"] for k in gemm)
