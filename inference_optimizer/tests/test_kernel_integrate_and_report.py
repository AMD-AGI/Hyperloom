# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P2-4 tests: integrate kernel-request handler + report runner + e2e."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.action_executors import (
    ReportExecutor,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import (
    SubAgentRunner,
)
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.paths import make_session_dir
from inference_optimizer.storage import SqliteConnection


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # Point HYPERLOOM_KERNEL_AGENT_ROOT at the repo's kernel-agent tree so
    # ``integrate_handler`` can resolve ``apply_kernel_patch.py``.
    kernel_agent_root = Path(__file__).resolve().parents[2] / "kernel-agent"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
    # Stub the interpreter resolver so the unit test never spawns a real probe.
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    from inference_optimizer.orchestrator.action_executors import _grid_runner
    monkeypatch.setattr(
        _grid_runner, "_resolve_magpie_python", lambda: "/usr/bin/python3",
    )
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n)
            for n in ("orchestration", "kernel", "critic", "robustness")}


def _write_baseline_yaml(path: Path) -> None:
    cfg = {"benchmark": {
        "framework": "sglang", "model": "/x", "precision": "bf16",
        "run_mode": "local", "envs": {"TP": 1},
        "benchmark_script": "sglang_mi300x.sh", "timeout_seconds": 600,
        "profiler": {"torch_profiler": {"enabled": False},
                      "system_profiler": {"enabled": False},
                      "tracelens": {"enabled": False}},
        "gpu_selection": {"auto": False},
    }}
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float = 800.0) -> Path:
    workspace = slot / "benchmark_sglang_smoke"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True, "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": tput / 256, "output_throughput": tput,
            "total_token_throughput": tput * 2, "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {
            "ttft": {"mean_ms": 140, "p99_ms": 158},
            "e2el": {"mean_ms": 2500, "p99_ms": 2580},
        },
    }))
    return workspace


def _write_patch_pair(
    tmp_path: Path,
    *,
    suffix: str = ".py",
    original: str = "def kernel():\n    return 'original'\n",
    optimized: str = "def kernel():\n    return 'optimized'\n",
) -> tuple[Path, Path]:
    target = tmp_path / f"kernel{suffix}"
    patch_file = tmp_path / f"optimized_kernel{suffix}"
    target.write_text(original, encoding="utf-8")
    patch_file.write_text(optimized, encoding="utf-8")
    return target, patch_file


# integrate_handler
def test_resolve_integrate_payload_fills_source_when_patch_path_present(
    session_dir, tmp_path,
):
    """Queued KEEPs may pass patch_path while relying on kernel_opt_attempts for source_file."""
    state = SharedState.load_or_init(session_dir)
    patch_path = tmp_path / "k001_opt.cu"
    source_path = tmp_path / "gemm_moe_ck2stages.cu"
    patch_path.write_text("// optimized\n", encoding="utf-8")
    source_path.write_text("// original\n", encoding="utf-8")
    state.last_kernel_opt = {
        "kernel_id": "k004",
        "best_artifact_path": "/tmp/k004_opt.cu",
        "source_file": "/tmp/rmsnorm.cu",
    }
    state.kernel_opt_attempts = {
        "k001": {
            "last_decision": "KEEP",
            "last_artifact_path": str(patch_path),
            "last_source_file": str(source_path),
        },
    }
    state.save(session_dir)

    resolved, err = krh._resolve_integrate_payload(
        {"kernel_id": "k001", "patch_path": str(patch_path)},
        session_dir=session_dir,
    )

    assert err is None
    assert resolved["patch_path"] == str(patch_path)
    assert resolved["source_file"] == str(source_path)


@pytest.mark.asyncio
async def test_integrate_handler_keep_decision(session_dir, tmp_path):
    """re-baseline returns 900 vs base 800 → KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput":   800.0,
        "config_path": str(base_yaml),
        "kernel_id":   "k_abc",
        "patch_path":  str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["kernel_id"] == "k_abc"
    assert res["patch_path"] == str(patch_file)
    assert res["base_tput"] == 800.0
    assert res["new_tput"] == 900.0
    assert res["gain_pct"] == pytest.approx((900 - 800) / 800 * 100)
    assert "report_path" in res
    assert "workspace" in res


@pytest.mark.asyncio
async def test_integrate_handler_keeps_positive_stack_increment(
    session_dir, tmp_path,
):
    """When a kernel stack already exists, a positive incremental gain should KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = 90.0
    state.current_best = {
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }
    state.optimization_stack = [{
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }]
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=100.75)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k001",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["gain_pct"] == pytest.approx(0.75)
    assert res["decision_reason"] == "stack_positive_increment"
    assert res["stack_incremental_gain_pct"] == pytest.approx(0.75)
    assert res["stack_incremental_keep_threshold_pct"] == pytest.approx(0.5)
    assert res["revert_result"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_integrate_handler_rejects_stack_increment_under_noise_floor(
    session_dir, tmp_path,
):
    """A sub-0.5% stack increment should remain NEEDS_REVIEW, not KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = 90.0
    state.current_best = {
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }
    state.optimization_stack = [{
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }]
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=100.49)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k001",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "NEEDS_REVIEW"
    assert res["gain_pct"] == pytest.approx(0.49)
    assert "decision_reason" not in res
    assert res["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_integrate_handler_keeps_exact_stack_increment_noise_floor(
    session_dir, tmp_path,
):
    """A +0.5% stack increment should KEEP at the configured noise floor."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = 90.0
    state.current_best = {
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }
    state.optimization_stack = [{
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }]
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=100.5)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k001",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["stack_incremental_gain_pct"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_integrate_handler_accepts_valid_rebaseline_with_wrapper_warning(session_dir, tmp_path):
    """Valid throughput should drive KEEP even if Magpie reports success=false."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        workspace = _fake_workspace(slot, tput=900.0)
        report_path = workspace / "benchmark_report.json"
        data = json.loads(report_path.read_text())
        data["success"] = False
        report_path.write_text(json.dumps(data))
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="cleanup failed",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_warn",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["new_tput"] == 900.0


@pytest.mark.asyncio
async def test_integrate_handler_revert_decision(session_dir, tmp_path):
    """re-baseline returns 700 vs base 800 → REVERT."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=700.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_bad",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)
    assert res["decision"] == "REVERT"
    assert res["gain_pct"] < -1


@pytest.mark.asyncio
async def test_integrate_handler_reverts_applied_source_on_non_keep(
    session_dir, tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target = tmp_path / "kernel.py"
    patch_file = tmp_path / "optimized_kernel.py"
    target.write_text("def kernel():\n    return 'original'\n", encoding="utf-8")
    patch_file.write_text("def kernel():\n    return 'optimized'\n", encoding="utf-8")

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=700.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_bad",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "REVERT"
    assert res["apply_result"]["status"] == "ok"
    assert res["revert_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
async def test_integrate_handler_resolves_patch_and_target_from_state(
    session_dir, tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    state = SharedState(
        session_id=session_dir.name,
        last_kernel_opt={
            "kernel_id": "k006",
            "best_artifact_path": str(patch_file),
        },
        last_trace_analyze={
            "hot_kernels_top15": [{
                "kernel_id": "k006",
                "source_file": str(target),
                "reusable_native_kernel": True,
            }],
        },
    )
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k006",
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["patch_path"] == str(patch_file)
    assert res["target_file"] == str(target)
    assert res["apply_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'optimized'\n"


@pytest.mark.asyncio
async def test_integrate_handler_fails_when_patch_inputs_missing(
    session_dir, tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)

    res = await krh.integrate_handler(
        {
            "base_tput": 800.0,
            "config_path": str(base_yaml),
            "kernel_id": "k_missing",
        },
        session_dir=session_dir,
    )

    assert res["status"] == "failed"
    assert res["decision"] == "REVERT"
    assert res["error_class"] == "missing_integration_inputs"
    assert "patch_path" in res["missing"]
    assert "target_file/source_file" in res["missing"]


@pytest.mark.asyncio
async def test_integrate_handler_rejects_text_patch_artifact(session_dir, tmp_path):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target = tmp_path / "kernel.py"
    patch_file = tmp_path / "optimized.txt"
    target.write_text("def kernel():\n    return 'original'\n", encoding="utf-8")
    patch_file.write_text("```python\ndef kernel():\n    return 'optimized'\n```\n", encoding="utf-8")

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_text",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "failed"
    assert res["decision"] == "REVERT"
    assert "complete source file" in res["apply_result"]["error"]
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
async def test_integrate_handler_rejects_incompatible_standalone_cpp(
    session_dir, tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target = tmp_path / "kernel.cu"
    patch_file = tmp_path / "optimized.cu"
    target.write_text(
        "namespace aiter {\nvoid add_rmsnorm() {}\nvoid rmsnorm() {}\n}\n",
        encoding="utf-8",
    )
    patch_file.write_text(
        "#include <torch/extension.h>\n"
        "__global__ void optimized_kernel() {}\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n",
        encoding="utf-8",
    )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_standalone",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "failed"
    assert "PYBIND11" in res["apply_result"]["error"]
    assert "add_rmsnorm" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_integrate_handler_injects_extra_server_args(
    session_dir, tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    seen: dict[str, object] = {}

    def _fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        seen["envs"] = cfg["benchmark"]["envs"]
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_good",
        "extra_server_args": "--cuda-graph-max-bs 8",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "KEEP"
    # extra_server_args is preserved verbatim and the watchdog timeout is
    # auto-appended; assert both rather than exact equality.
    sglang_args = seen["envs"]["EXTRA_SGLANG_ARGS"]
    assert "--cuda-graph-max-bs 8" in sglang_args
    assert "--watchdog-timeout" in sglang_args


@pytest.mark.asyncio
async def test_integrate_handler_needs_review_when_within_threshold(
    session_dir, tmp_path,
):
    """re-baseline returns 805 (+0.625%) vs base 800 → NEEDS_REVIEW."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=805.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_review",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)
    assert res["decision"] == "NEEDS_REVIEW"


@pytest.mark.asyncio
async def test_integrate_handler_rejects_zero_base_tput(session_dir):
    res = await krh.integrate_handler({"base_tput": 0}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "base_tput" in res["error"]


def test_integrate_registered_under_two_aliases():
    assert krh.has_handler("integrate")
    assert krh.has_handler("apply_patch")
    assert krh.get_handler("integrate") is krh.get_handler("apply_patch")


# Coordinator wiring of integrate request
@pytest.mark.asyncio
async def test_coordinator_integrate_request_emits_keep_response(session_dir, tmp_path):
    """REQUEST{kind=integrate} → handler runs → RESPONSE carries KEEP/REVERT."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "reusable_native_kernel_ids": ["k1"],
        }
        c.shared_state.save(session_dir)
        with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "integrate",
                    "params": {
                        "base_tput": 800.0,
                        "config_path": str(base_yaml),
                        "kernel_id": "k1",
                        "patch_path": str(patch_file),
                        "target_file": str(target),
                        "allow_unknown_target": True,
                        "skip_rebuild": True,
                    },
                },
            ))
        responses = sorted(
            await c.bus.tail(topic="response", to_agent="orchestration"),
            key=lambda msg: msg.seq,
        )
        assert responses
        r = responses[0]
        assert r.payload["kind"] == "integrate_done"
        assert r.payload["status"] == "ok"
        result = r.payload["result"]
        assert result["decision"] == "KEEP"
        assert result["new_tput"] == 900.0
        assert c.shared_state.current_best["action"] == "integrate"
        assert c.shared_state.current_best["kernel_id"] == "k1"
        assert any(
            item.get("action") == "integrate"
            and item.get("kernel_id") == "k1"
            for item in c.shared_state.optimization_stack
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_stops_repeating_same_kernel_integrate_after_cap(
    session_dir, tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    run_calls = 0

    def _fake_run(cmd, *args, **kwargs):
        nonlocal run_calls
        run_calls += 1
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=805.0)  # +0.625%, below KEEP threshold
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "reusable_native_kernel_ids": ["k_repeat"],
        }
        c.shared_state.save(session_dir)
        payload = {
            "target_agent": "kernel",
            "kind": "integrate",
            "params": {
                "base_tput": 800.0,
                "config_path": str(base_yaml),
                "kernel_id": "k_repeat",
                "patch_path": str(patch_file),
                "target_file": str(target),
                "allow_unknown_target": True,
                "skip_rebuild": True,
            },
        }
        with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
            for _ in range(4):
                await c._handle_intent(
                    "orchestration",
                    Intent(type=IntentType.REQUEST, payload=payload),
                )

        responses = sorted(
            await c.bus.tail(topic="response", to_agent="orchestration"),
            key=lambda msg: msg.seq,
        )
        integrate_results = [
            r.payload["result"]
            for r in responses
            if r.payload.get("kind") == "integrate_done"
        ]
        assert len(integrate_results) == 4
        # Cap: first 3 attempts run the integrate path, the 4th is short-circuited.
        # Assert proportionally since per-attempt subprocess count is an impl detail.
        assert run_calls > 0, "first 3 attempts must spawn subprocess"
        assert run_calls % 3 == 0, (
            f"first 3 attempts should contribute equal subprocess counts; "
            f"got {run_calls} (4th attempt should contribute 0)"
        )
        assert [r["decision"] for r in integrate_results[:3]] == [
            "NEEDS_REVIEW",
            "NEEDS_REVIEW",
            "NEEDS_REVIEW",
        ]
        assert integrate_results[-1]["status"] == "skipped"
        assert integrate_results[-1]["decision"] == "REVERT"
        assert integrate_results[-1]["error_class"] == "kernel_patch_rejected"

        saved = SharedState.load_or_init(session_dir)
        assert saved.rejected_kernel_patches
        assert saved.rejected_kernel_patches[0]["kernel_id"] == "k_repeat"
    finally:
        await c.stop()


# ReportExecutor
@pytest.mark.asyncio
async def test_report_executor_writes_md_and_json(session_dir):
    """Run the report runner against seeded state + bus events; both files parse."""
    state = SharedState(
        session_id=session_dir.name,
        model_name="Qwen-Qwen3-8B",
        model_path="/wekafs/models/Qwen-Qwen3-8B",
        baseline_tput=800.0,
        cumulative_gain=12.5,
        current_best={"action": "backends", "tput": 900.0,
                       "ttft_mean_ms": 130.0, "e2el_mean_ms": 2400.0,
                       "workspace": "/x/y/z"},
        max_minutes=120,
        stop_reason="target_reached",
    )
    state.save(session_dir)

    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        await c._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
        ))
        await c._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "explore", "predicted_gain_pct": 5.0},
        ))
        await c._handle_intent("robustness", Intent(
            type=IntentType.ALERT,
            payload={"severity": "low", "summary": "noise"},
        ))
    finally:
        await c.stop()

    db = SqliteConnection(tmp_path_helper(session_dir))
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("report", ReportExecutor())

    task = await tr.create(
        kind="report",
        params={"session_dir": str(session_dir)},
        idempotency_key="rep-1",
    )
    res = await sub.run_task(task)
    db.close()
    assert res.state == "succeeded"

    md = Path(res.result["md_path"])
    js = Path(res.result["json_path"])
    assert md.exists()
    assert js.exists()
    summary = json.loads(js.read_text())
    assert summary["session_id"] == session_dir.name
    assert summary["baseline_tput"] == 800.0
    assert summary["cumulative_gain"] == 12.5
    assert summary["stop_reason"] == "target_reached"
    assert summary["event_counts_by_topic"].get("proposal", 0) >= 2
    assert summary["event_counts_by_topic"].get("alert", 0) >= 1
    md_text = md.read_text()
    assert session_dir.name in md_text
    assert "## Throughput" in md_text
    assert "12.50%" in md_text
    assert "target_reached" in md_text


def tmp_path_helper(session_dir: Path) -> Path:
    """Point ReportExecutor's SqliteConnection at the session's DB."""
    return session_dir / "storage" / "coordinator.db"


@pytest.mark.asyncio
async def test_report_executor_failed_when_session_dir_unresolvable(tmp_path,
                                                                      monkeypatch):
    """An unresolvable session_dir yields a structured failure, not a crash."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "noses"))
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("report", ReportExecutor())
    task = await tr.create(
        kind="report",
        params={},
        idempotency_key="rep-fail-1",
    )
    res = await sub.run_task(task)
    db.close()
    assert res.state == "succeeded"
    assert res.result["status"] == "failed"
    assert "session_dir" in res.result.get("error", "")


# ---------------------------------------------------------------------------
# after_kernel_opt rocprof: KEEP triggers it, REVERT/NEEDS_REVIEW skips it.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_integrate_keep_schedules_after_kernel_opt_rocprof(
    session_dir, tmp_path, monkeypatch,
):
    """On KEEP, after-opt rocprof is scheduled without blocking integrate."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    calls: list[dict] = []

    def _fake_rocprof(*, kernel_id, session_dir, log):
        calls.append({"kernel_id": kernel_id})
        return {"status": "scheduled", "reason": "stub"}

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=900.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(krh, "_schedule_after_kernel_opt_rocprof", _fake_rocprof)
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_rocprof_test",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "KEEP"
    assert len(calls) == 1
    assert calls[0]["kernel_id"] == "k_rocprof_test"
    assert res["rocprof_after_kernel_opt"]["status"] == "scheduled"


@pytest.mark.asyncio
async def test_integrate_revert_skips_after_kernel_opt_rocprof(
    session_dir, tmp_path, monkeypatch,
):
    """On REVERT, _run_after_kernel_opt_rocprof is NOT called."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    calls: list[dict] = []

    def _fake_rocprof(*, kernel_id, session_dir, log):  # pragma: no cover
        calls.append({"kernel_id": kernel_id})
        return {"status": "stub"}

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        # return tput lower than base → REVERT
        _fake_workspace(Path(cmd[out_idx + 1]), tput=500.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(krh, "_schedule_after_kernel_opt_rocprof", _fake_rocprof)
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_revert_test",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "REVERT"
    assert len(calls) == 0
    assert "rocprof_after_kernel_opt" not in res


@pytest.mark.asyncio
async def test_after_kernel_opt_rocprof_uses_profile_mode_and_safe_timeout(
    session_dir, tmp_path, monkeypatch,
):
    """After-opt rocprof must be best-effort and profile the same workload as before."""
    state = SharedState.load_or_init(session_dir)
    source_file = tmp_path / "kernel.cu"
    source_file.write_text("// kernel\n", encoding="utf-8")
    state.kernel_opt_attempts = {
        "k_after": {
            "test_command": "python /tmp/run/unittest/harness_moe.py --correctness",
            "last_source_file": str(source_file),
        }
    }
    state.save(session_dir)
    sidecar = session_dir / "reports" / "kernel_roofline.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "kernels": [{
            "kernel_id": "k_after",
            "name": "aiter::moe_kernel",
            "rocprof_roofline": {
                "before_kernel_opt": {"status": "matched"},
                "after_kernel_opt": None,
            },
        }]
    }), encoding="utf-8")
    captured: dict[str, list[str]] = {}

    async def _fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = cmd
        captured["timeout_sec"] = timeout_sec
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        out_json.write_text(json.dumps({
            "status": "ok",
            "target_kernel": "aiter::moe_kernel",
            "results": [{
                "name": "aiter::moe_kernel",
                "matched_kernel_name": "aiter::moe_kernel",
                "status": "matched",
                "rocprof_roofline": {"bound_type": "memory"},
            }],
        }), encoding="utf-8")
        return 0, "ok", ""

    monkeypatch.setenv("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "abc")
    monkeypatch.setattr(krh, "_run_subprocess", _fake_run_subprocess)

    res = await krh._run_after_kernel_opt_rocprof(
        kernel_id="k_after",
        session_dir=session_dir,
        log=krh.log,
    )

    cmd = captured["cmd"]
    assert res["status"] == "ok"
    assert cmd[cmd.index("--cmd") + 1].endswith("harness_moe.py --profile")
    assert cmd[cmd.index("--timeout-sec") + 1] == "1800"
    assert captured["timeout_sec"] == 1830
    assert cmd[cmd.index("--target-kernel") + 1] == "aiter::moe_kernel"


def test_schedule_after_kernel_opt_rocprof_marks_scheduled(
    session_dir, monkeypatch,
):
    sidecar = session_dir / "reports" / "kernel_roofline.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "kernels": [{
            "kernel_id": "k_sched",
            "name": "aiter::sched_kernel",
            "rocprof_roofline": {
                "before_kernel_opt": {"status": "matched"},
                "after_kernel_opt": None,
            },
        }]
    }), encoding="utf-8")
    created: list[object] = []

    async def _fake_run_after(**_kwargs):
        return {"status": "ok"}

    class _DummyTask:
        def result(self):
            return {"status": "ok"}

        def add_done_callback(self, callback):
            callback(self)

    def _fake_create_task(coro):
        coro.close()
        task = _DummyTask()
        created.append(task)
        return task

    monkeypatch.setattr(krh, "_run_after_kernel_opt_rocprof", _fake_run_after)
    monkeypatch.setattr(krh.asyncio, "create_task", _fake_create_task)

    res = krh._schedule_after_kernel_opt_rocprof(
        kernel_id="k_sched",
        session_dir=session_dir,
        log=krh.log,
    )

    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    after = updated["kernels"][0]["rocprof_roofline"]["after_kernel_opt"]
    assert res["status"] == "scheduled"
    assert after == {"status": "scheduled", "reason": "background_task"}
    assert len(created) == 1
