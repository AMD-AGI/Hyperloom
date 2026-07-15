# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""cuda-graph capture failure classification + one-shot --enforce-eager fallback.

A cuda-graph capture failure is often recoverable by retrying with
``--enforce-eager``. These tests pin the classifier, the idempotent flag
injection, and the one-shot consume contract.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors import (
    baseline as baseline_module,
)
from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
    _disable_cuda_graph_flag,
    _is_cuda_graph_capture_failure,
    _with_cuda_graph_disabled,
)
from hyperloom.orchestrator.state.shared_state import SharedState


def test_cuda_graph_capture_markers_detected():
    assert _is_cuda_graph_capture_failure(
        "torch.AcceleratorError: HIP error: operation not permitted when stream is capturing"
    )
    # Non-OOM capture failure is recoverable.
    assert _is_cuda_graph_capture_failure("Capture cuda graph failed: HIP error: operation not permitted")
    assert _is_cuda_graph_capture_failure("hipErrorStreamCaptureUnsupported")


def test_oom_capture_failure_not_flagged():
    # An OOM-rooted capture failure must not arm the one-shot fallback.
    assert not _is_cuda_graph_capture_failure(
        "Exception: Capture cuda graph failed: HIP out of memory. Tried to allocate 4.78 GiB."
    )
    assert not _is_cuda_graph_capture_failure("Capture cuda graph failed: torch.OutOfMemoryError: HIP out of memory")


def test_unrelated_earlier_oom_does_not_mask_capture_failure():
    # An unrelated OOM warning must not mask a genuine stream-capture failure;
    # OOM exclusion is scoped to the marker's local context, not the whole blob.
    blob = "\n".join(
        ["[warn] some cache out of memory, retrying"]
        + ["  ... unrelated startup line ..."] * 20
        + ["operation not permitted when stream is capturing"]
    )
    assert _is_cuda_graph_capture_failure(blob)


def test_compile_error_capture_failure_not_flagged():
    # A bare marker rooted in a compile/lowering error stays a weak signal only.
    assert not _is_cuda_graph_capture_failure("Capture cuda graph failed: LoweringException: AssertionError")
    assert not _is_cuda_graph_capture_failure("Capture cuda graph failed\n  CompilationError: invalid kernel")


def test_strong_marker_ignores_compile_error_in_context():
    # A strong stream-capture marker arms the fallback even with a nearby
    # compile error; compile-error exclusion only gates the bare/weak marker.
    assert _is_cuda_graph_capture_failure(
        "AssertionError: shape mismatch\noperation not permitted when stream is capturing"
    )


def test_strong_and_weak_same_line_with_compile_error_is_recoverable():
    # A line matching both markers with a compile error is recoverable: strong
    # wins and the weak-only compile gate must not demote it.
    assert _is_cuda_graph_capture_failure(
        "Capture cuda graph failed: LoweringException: operation not permitted when stream is capturing"
    )


def test_strong_marker_with_adjacent_unrelated_oom_warning_is_recoverable():
    # An unrelated OOM warning a few lines from a strong marker must not demote
    # it (strong uses a tight ±1-line window).
    blob = "\n".join(
        ["[warn] kv cache out of memory, shrinking"]
        + ["  ... startup line ..."] * 3
        + ["operation not permitted when stream is capturing"]
    )
    assert _is_cuda_graph_capture_failure(blob)


def test_strong_marker_with_oom_on_same_line_not_flagged():
    # OOM on the strong marker line is a real OOM-rooted capture failure.
    assert not _is_cuda_graph_capture_failure("HIP out of memory; operation not permitted when stream is capturing")


def test_weak_marker_with_distant_oom_not_flagged():
    # Weak marker keeps whole-blob OOM exclusion: an OOM anywhere demotes it.
    blob = "\n".join(
        ["Capture cuda graph failed"]
        + ["  at frame %d" % i for i in range(20)]
        + ["torch.OutOfMemoryError: HIP out of memory"]
    )
    assert not _is_cuda_graph_capture_failure(blob)


def test_non_cuda_graph_failures_not_flagged():
    assert not _is_cuda_graph_capture_failure("HIP out of memory")
    assert not _is_cuda_graph_capture_failure("Floating point exception")
    assert not _is_cuda_graph_capture_failure("")


def test_disable_cuda_graph_flag_per_framework():
    # sglang uses --disable-cuda-graph; vLLM uses --enforce-eager.
    assert _disable_cuda_graph_flag("sglang") == "--disable-cuda-graph"
    assert _disable_cuda_graph_flag("vllm") == "--enforce-eager"
    # Unknown / empty framework defaults to the sglang-safe flag.
    assert _disable_cuda_graph_flag("") == "--disable-cuda-graph"
    assert _disable_cuda_graph_flag("atom") == "--disable-cuda-graph"


def test_with_cuda_graph_disabled_is_idempotent():
    assert _with_cuda_graph_disabled("", "sglang") == "--disable-cuda-graph"
    assert _with_cuda_graph_disabled("--mem-fraction-static=0.8", "sglang") == (
        "--mem-fraction-static=0.8 --disable-cuda-graph"
    )
    assert _with_cuda_graph_disabled("--disable-cuda-graph", "sglang") == ("--disable-cuda-graph")
    assert _with_cuda_graph_disabled("--enforce-eager", "vllm") == ("--enforce-eager")
    assert _with_cuda_graph_disabled("--a --disable-cuda-graph --b", "sglang").count("--disable-cuda-graph") == 1
    # Token-level dedup: a longer flag must not block the real one.
    assert (

        _with_cuda_graph_disabled("--disable-cuda-graph-extra=1", "sglang")
        == "--disable-cuda-graph-extra=1 --disable-cuda-graph"
    )


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


def test_capture_failure_wins_over_server_init_dead_marker_helper():
    # A capture marker co-occurring with a server-death marker still classifies
    # as a capture failure.
    blob = (
        "server engine/worker init failed (reaped by liveness watchdog)\n"
        "operation not permitted when stream is capturing"
    )
    assert _is_cuda_graph_capture_failure(blob) is True


# ── executor-level __call__ tests (subprocess mocked) ────────────────────────
def _baseline_yaml(path: Path) -> None:
    path.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  benchmark_script: sglang_mi300x.sh\n"
        "  envs:\n"
        "    MODEL: /wekafs/models/Qwen-Qwen3-8B\n",
        encoding="utf-8",
    )


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-cg-exec", params=params)
    return SimpleNamespace(task=task, extra={})


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _run_executor_with_server_log(
    tmp_path: Path,
    server_log_text: str,
    *,
    framework: str | None = "sglang",
    eager_armed: bool = False,
    shared_state: SharedState | None = None,
    make_workspace: bool = False,
) -> tuple[dict, dict]:
    """Run BaselineExecutor.__call__ with a mocked Magpie that writes a server.log.

    By default produces no benchmark_* workspace (no_workspace path); with
    ``make_workspace`` it creates one with an invalid report so the
    invalid_measurement branch is exercised instead.

    Returns (result, captured) where captured["extra_server_args"] is the
    effective server-args string that reached materialization.
    """
    base = tmp_path / "base.yaml"
    _baseline_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    if eager_armed:
        state = shared_state or SharedState.load_or_init(tmp_path)
        state.baseline_eager_fallback = True
        state.save(tmp_path)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "server.log").write_text(server_log_text, encoding="utf-8")
        if make_workspace:
            # A workspace whose report yields no valid measurement, so __call__
            # reaches the invalid_measurement branch.
            ws = slot / "benchmark_sglang_20260101_000000"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "benchmark_report.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    real_materialize = baseline_module.materialize_config_with_envs

    def spy_materialize(config_path, output_dir, *args, **kwargs):
        captured["extra_server_args"] = kwargs.get("extra_server_args", "")
        return real_materialize(config_path, output_dir, *args, **kwargs)

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=shared_state,
    )
    task_params = {
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "extra_server_args": "--mem-fraction-static=0.8",
    }
    if framework is not None:
        task_params["framework"] = framework
    ctx = _make_ctx(task_params)
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
        patch.object(
            baseline_module,
            "materialize_config_with_envs",
            side_effect=spy_materialize,
        ),
    ):
        result = asyncio.run(executor(ctx))
    return result, captured


def test_executor_capture_failure_wins_over_server_init_dead(tmp_path: Path):
    # server-death + capture markers co-occur; the no_workspace branch must
    # classify cuda_graph_capture_failed so the one-shot fallback is armed.
    log_text = (
        "server engine/worker init failed (reaped by liveness watchdog)\n"
        "operation not permitted when stream is capturing\n"
    )
    result, _ = _run_executor_with_server_log(tmp_path, log_text)
    assert result["status"] == "failed"
    assert result["error_class"] == "cuda_graph_capture_failed"


def test_executor_consumes_flag_and_injects_disable_flag(tmp_path: Path):
    # With the eager flag armed, the next baseline consumes it and the
    # disable-cuda-graph flag reaches materialization.
    result, captured = _run_executor_with_server_log(
        tmp_path,
        "boot failed\n",
        eager_armed=True,
    )
    assert result["status"] == "failed"
    assert "--disable-cuda-graph" in captured["extra_server_args"]
    # One-shot: the flag is consumed after this run.
    assert SharedState.load_or_init(tmp_path).baseline_eager_fallback is False


def test_executor_keeps_flag_armed_when_framework_unknown(
    tmp_path: Path,
    monkeypatch,
):
    # Unknown framework cannot pick a safe flag, so the one-shot stays armed for
    # a later baseline with a known framework.
    monkeypatch.delenv("FRAMEWORK", raising=False)
    result, captured = _run_executor_with_server_log(
        tmp_path,
        "boot failed\n",
        framework=None,
        eager_armed=True,
    )
    assert result["status"] == "failed"
    assert "--disable-cuda-graph" not in captured["extra_server_args"]
    assert "--enforce-eager" not in captured["extra_server_args"]
    # Flag preserved for a later known-framework retry.
    assert SharedState.load_or_init(tmp_path).baseline_eager_fallback is True


def test_executor_no_inject_when_flag_not_armed(tmp_path: Path):
    result, captured = _run_executor_with_server_log(
        tmp_path,
        "boot failed\n",
        eager_armed=False,
    )
    assert result["status"] == "failed"
    assert "--disable-cuda-graph" not in captured["extra_server_args"]


def test_executor_capture_wins_over_server_init_dead_invalid_measurement(
    tmp_path: Path,
):
    # A workspace exists but yields no valid measurement, and server.log carries
    # both a server-death and a capture marker; the invalid_measurement branch
    # must still classify cuda_graph_capture_failed.
    log_text = (
        "server engine/worker init failed (reaped by liveness watchdog)\n"
        "operation not permitted when stream is capturing\n"
    )
    result, _ = _run_executor_with_server_log(
        tmp_path,
        log_text,
        make_workspace=True,
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "cuda_graph_capture_failed"
