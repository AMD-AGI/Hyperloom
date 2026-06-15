# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""cuda-graph capture failure classification + one-shot --enforce-eager fallback.

A cuda-graph capture failure (``operation not permitted when stream is
capturing`` / ``Capture cuda graph failed``) is often recoverable by retrying
with ``--enforce-eager``. These tests pin the classifier, the idempotent flag
injection, and the one-shot consume contract.
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.orchestrator.action_executors.baseline import (
    BaselineExecutor,
    _disable_cuda_graph_flag,
    _is_cuda_graph_capture_failure,
    _with_cuda_graph_disabled,
)
from inference_optimizer.orchestrator.shared_state import SharedState


def test_cuda_graph_capture_markers_detected():
    assert _is_cuda_graph_capture_failure(
        "torch.AcceleratorError: HIP error: operation not permitted "
        "when stream is capturing"
    )
    assert _is_cuda_graph_capture_failure("Capture cuda graph failed: HIP error")
    assert _is_cuda_graph_capture_failure("hipErrorStreamCaptureUnsupported")


def test_non_cuda_graph_failures_not_flagged():
    assert not _is_cuda_graph_capture_failure("HIP out of memory")
    assert not _is_cuda_graph_capture_failure("Floating point exception")
    assert not _is_cuda_graph_capture_failure("")


def test_disable_cuda_graph_flag_per_framework():
    # sglang rejects vLLM's --enforce-eager; it must use --disable-cuda-graph.
    assert _disable_cuda_graph_flag("sglang") == "--disable-cuda-graph"
    assert _disable_cuda_graph_flag("vllm") == "--enforce-eager"
    # unknown / empty framework defaults to the sglang-safe flag.
    assert _disable_cuda_graph_flag("") == "--disable-cuda-graph"
    assert _disable_cuda_graph_flag("atom") == "--disable-cuda-graph"


def test_with_cuda_graph_disabled_is_idempotent():
    assert _with_cuda_graph_disabled("", "sglang") == "--disable-cuda-graph"
    assert _with_cuda_graph_disabled("--mem-fraction-static=0.8", "sglang") == (
        "--mem-fraction-static=0.8 --disable-cuda-graph"
    )
    assert _with_cuda_graph_disabled("--disable-cuda-graph", "sglang") == (
        "--disable-cuda-graph"
    )
    assert _with_cuda_graph_disabled("--enforce-eager", "vllm") == (
        "--enforce-eager"
    )
    assert _with_cuda_graph_disabled(
        "--a --disable-cuda-graph --b", "sglang"
    ).count("--disable-cuda-graph") == 1


def test_eager_fallback_is_consumed_once(tmp_path: Path):
    state = SharedState.load_or_init(tmp_path)
    state.baseline_eager_fallback = True
    state.save(tmp_path)

    executor = BaselineExecutor(session_dir=tmp_path)
    assert executor._consume_eager_fallback() is True
    assert executor._consume_eager_fallback() is False

    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.baseline_eager_fallback is False


def test_eager_fallback_absent_returns_false(tmp_path: Path):
    SharedState.load_or_init(tmp_path).save(tmp_path)
    executor = BaselineExecutor(session_dir=tmp_path)
    assert executor._consume_eager_fallback() is False
