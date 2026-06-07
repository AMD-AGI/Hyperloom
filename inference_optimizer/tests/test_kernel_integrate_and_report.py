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


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # Point HYPERLOOM_KERNEL_AGENT_ROOT at the repo's kernel-agent tree so
    # ``integrate_handler`` -> ``_maybe_apply_kernel_patch`` ->
    # ``_load_apply_tool`` can resolve ``apply_kernel_patch.py`` without
    # requiring the operator to source ``$KERNEL_AGENT_ENV`` before
    # running pytest. ``krh`` is already imported at module top — no
    # inline reimport. Same convention as test_p2_2_profile_and_handlers.py.
    kernel_agent_root = Path(__file__).resolve().parents[2] / "kernel-agent"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
    # Keep the unit test hermetic: stub the interpreter resolver so it never
    # spawns a real ``python3 -c "import Magpie"`` probe. (Setting
    # MAGPIE_PYTHON alone no longer short-circuits the probe — the resolver
    # validates the env value via run_with_session_kill since the
    # stale-interpreter self-heal landed.) The mocked Magpie launch only ever
    # sees this fixed interpreter as ``cmd[0]``.
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


# ===========================================================================
# integrate_handler
# ===========================================================================
@pytest.mark.asyncio
async def test_integrate_handler_keep_decision(session_dir, tmp_path):
    """re-baseline returns 900 vs base 800 → KEEP."""
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
    target, patch_file = _write_patch_pair(tmp_path)

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
    # The operator-supplied extra_server_args must be preserved verbatim, and
    # the shared materialization choke point auto-appends the sglang
    # scheduler ``--watchdog-timeout`` (MI300X cold-compile guard) on top of
    # it. Assert both rather than exact equality so the watchdog default
    # (1800s, or $SGLANG_WATCHDOG_TIMEOUT) can evolve without breaking this
    # test. See ``inject_sglang_watchdog_timeout`` / test_watchdog_injection.
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
    # Both must point to the same callable.
    assert krh.get_handler("integrate") is krh.get_handler("apply_patch")


# ===========================================================================
# Coordinator wiring of integrate request
# ===========================================================================
@pytest.mark.asyncio
async def test_coordinator_integrate_request_emits_keep_response(session_dir, tmp_path):
    """End-to-end coordinator flow: REQUEST{kind=integrate} → handler runs →
    RESPONSE on bus contains KEEP/REVERT decision."""
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
        _fake_workspace(slot, tput=805.0)  # +0.625%, below KEEP threshold.
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
        # Cap verification: the first 3 attempts run the integrate path
        # (each spawning the rebaseline benchmark subprocess and any
        # apply-side helper subprocesses), and the 4th attempt is
        # short-circuited before any subprocess runs. The exact subprocess
        # count per attempt is an implementation detail (today it's 2 —
        # apply pre-flight + rebaseline — totalling 6 over the first 3
        # attempts), so assert proportionally: ``run_calls`` must be a
        # positive multiple of the 3 non-capped attempts, with no
        # additional growth past the 3rd attempt (i.e. the 4th attempt
        # contributes zero, which is what ``status == "skipped"`` below
        # also pins).
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


# ===========================================================================
# ReportExecutor
# ===========================================================================
@pytest.mark.asyncio
async def test_report_executor_writes_md_and_json(session_dir):
    """Pre-load the session with realistic state + a few bus events,
    then run the report runner and assert both files exist + parse."""
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

    # Make the coordinator seed the bus with a few realistic events.
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

    # Build a Task ourselves and invoke the runner.
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
    """ReportExecutor needs its own SqliteConnection — point it at the
    session's DB."""
    return session_dir / "storage" / "coordinator.db"


@pytest.mark.asyncio
async def test_report_executor_failed_when_session_dir_unresolvable(tmp_path,
                                                                      monkeypatch):
    """Without an explicit session_dir param + nothing under env override,
    the runner reports a structured failure (not a crash). Note the
    SubAgentRunner state stays "succeeded" because the runner returned
    a dict (didn't raise) — the failure signal is inside result['status']."""
    # Pin USER_DATA_PATH at a path with no state.json so
    # ReportExecutor's resolution returns None (canonical default would
    # also work, but we use tmp_path to keep the test hermetic).
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
