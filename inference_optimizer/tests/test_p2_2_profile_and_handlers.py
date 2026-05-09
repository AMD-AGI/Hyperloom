"""P2-2 tests: ProfileExecutor + kernel REQUEST programmatic handlers."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.action_executors.baseline import (
    BaselineExecutor,
    _default_baseline_config,
    _materialize_config_with_envs,
)
from inference_optimizer.orchestrator.action_executors.profile import (
    PROFILE_DEFAULT_CONFIG,
    ProfileExecutor,
    _default_profile_config,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
    MockTurn,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import (
    RunnerContext, SubAgentRunner,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.storage import SqliteConnection


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_ROOT", str(tmp_path))
    return make_session_dir("p2-2-test")


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n)
            for n in ("orchestration", "kernel", "critic", "robustness")}


# ===========================================================================
# ProfileExecutor
# ===========================================================================
def test_profile_default_config_path_is_in_assets():
    assert "profile_sglang.yaml" in str(PROFILE_DEFAULT_CONFIG)
    assert PROFILE_DEFAULT_CONFIG.exists(), \
        "profile YAML must ship as a package asset"


def test_profile_yaml_has_torch_profiler_enabled():
    """The whole point of the profile config is profiler ON."""
    import yaml
    with PROFILE_DEFAULT_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True


# ===========================================================================
# Regression: model_path injection beats the YAML's hardcoded fallback.
#
# Bug: the shipped baseline_sglang.yaml / profile_sglang.yaml pin
# `benchmark.model: /wekafs/models/Qwen-Qwen3-8B` as a fallback for offline
# Magpie use. The CLI's --model arg only flowed into SharedState.model_path;
# if the executor did not propagate it into the materialized YAML, Magpie
# silently benchmarked Qwen3-8B no matter what the user asked for.
# _materialize_config_with_envs(model_path=...) is the single seam that
# prevents this — locking it down here.
# ===========================================================================
def test_materialize_config_injects_model_path(tmp_path):
    """Default YAML's hardcoded Qwen3-8B must be overridden when caller
    passes ``model_path`` — otherwise the silent fallback bug returns."""
    import yaml
    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        model_path="/wekafs/models/DeepSeek-R1-0528",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["model"] == "/wekafs/models/DeepSeek-R1-0528"


def test_materialize_config_leaves_model_alone_without_override(tmp_path, monkeypatch):
    """When no model_path is passed, the materialized YAML still has the
    original model field from the source YAML (not overwritten)."""
    import yaml
    # Clear ISL/OSL/MAX_MODEL_LEN env so they don't inject
    for k in ("ISL", "OSL", "MAX_MODEL_LEN", "PRECISION"):
        monkeypatch.delenv(k, raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert "Qwen" in rendered["benchmark"]["model"]


def test_materialize_config_injects_model_with_other_overrides(tmp_path):
    """Co-existence: model_path + extra_envs should both land in the
    materialized YAML."""
    import yaml
    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        extra_envs={"FOO": "bar"},
        model_path="/some/model",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["model"] == "/some/model"
    assert rendered["benchmark"]["envs"]["FOO"] == "bar"


# ===========================================================================
# Regression: gpu_type injection sets runner_type AND removes the legacy
# `benchmark_script` field so Magpie's runner_type -> script logic wins.
# ===========================================================================
def test_materialize_config_injects_runner_type(tmp_path):
    """gpu_type kwarg must land in benchmark.runner_type as-is."""
    import yaml
    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        gpu_type="mi355x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["runner_type"] == "mi355x"


def test_materialize_config_pops_legacy_benchmark_script(tmp_path):
    """If the source YAML still hardcodes a benchmark_script (priority 1
    in Magpie's resolver), gpu_type must remove it; otherwise runner_type
    is silently ignored and the run uses the wrong GPU's script."""
    import yaml
    src_yaml = tmp_path / "src.yaml"
    src_yaml.write_text(yaml.safe_dump({
        "benchmark": {
            "framework": "sglang",
            "model": "/m",
            "benchmark_script": "sglang_mi300x.sh",  # legacy field
        },
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = _materialize_config_with_envs(
        src_yaml, out_dir, gpu_type="mi355x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["runner_type"] == "mi355x"
    assert "benchmark_script" not in rendered["benchmark"], \
        "legacy benchmark_script must be popped so runner_type wins"


# ===========================================================================
# Regression: TP / CONC env override yaml hardcode (DSR1-0528 verification
# was deadlooping because TP=8 env was silently ignored, vllm ran with
# yaml-hardcoded TP=1 and OOM-ed retry forever).
# ===========================================================================
def test_materialize_config_tp_env_overrides_yaml_hardcode(tmp_path, monkeypatch):
    """TP env var must override yaml hardcode (was 1, becomes 8)."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["TP"] == 8, f"TP not overridden: {envs.get('TP')}"


def test_materialize_config_conc_env_overrides_yaml_hardcode(tmp_path, monkeypatch):
    """CONC env var must override yaml hardcode."""
    import yaml
    monkeypatch.setenv("CONC", "64")
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["CONC"] == 64, f"CONC not overridden: {envs.get('CONC')}"


def test_materialize_config_rocr_visible_devices_auto_expands_when_tp_overridden(
    tmp_path, monkeypatch,
):
    """When TP=8 is set via env but ROCR_VISIBLE_DEVICES isn't explicit,
    expand the GPU list to 0..TP-1 so vllm/sglang sees enough devices."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7", (
        f"ROCR_VISIBLE_DEVICES not auto-expanded: {envs.get('ROCR_VISIBLE_DEVICES')}"
    )


def test_materialize_config_rocr_visible_devices_explicit_env_wins_when_enough(
    tmp_path, monkeypatch,
):
    """Explicit ROCR_VISIBLE_DEVICES wins when it has at least TP devices."""
    import yaml
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "4,5,6,7"


def test_materialize_config_rocr_visible_devices_expands_when_under_tp(
    tmp_path, monkeypatch,
):
    """If explicit ROCR_VISIBLE_DEVICES has fewer devices than TP requires,
    `_workload_envs` auto-expands to 0..TP-1 and logs a warning, so SGLang
    actually sees enough GPUs to start."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_materialize_config_rocr_unchanged_when_tp1(tmp_path, monkeypatch):
    """When TP=1 (default), don't auto-touch ROCR_VISIBLE_DEVICES."""
    import yaml
    src_yaml = tmp_path / "src.yaml"
    src_yaml.write_text(yaml.safe_dump({
        "benchmark": {
            "framework": "sglang",
            "model": "/m",
            "envs": {
                "TP": 1,
                "CONC": 8,
                "ISL": 256,
                "OSL": 256,
                "ROCR_VISIBLE_DEVICES": "1",
            },
        },
    }))
    for k in ("TP", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    out = _materialize_config_with_envs(src_yaml, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    # yaml default is "1" — should be preserved as-is when TP not overridden upward
    assert envs.get("ROCR_VISIBLE_DEVICES") == "1"


# ===========================================================================
# Regression: $FRAMEWORK env switches the default yaml between sglang/vllm
# without anyone passing config_path explicitly. Locks down the entry-layer
# fix for vLLM support — the optimizer used to be sglang-only because all 5
# executors hardcoded baseline_sglang.yaml.
# ===========================================================================
def test_default_baseline_config_resolves_sglang_by_default(monkeypatch):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    assert _default_baseline_config().name == "baseline_sglang.yaml"


def test_default_baseline_config_resolves_vllm_when_env_set(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert _default_baseline_config().name == "baseline_vllm.yaml"


def test_default_baseline_config_falls_back_on_unknown_value(monkeypatch):
    """Unknown $FRAMEWORK is treated as sglang (matches CLI default).
    The CLI fail-fasts on unknown values, but if a user shell has a stale
    or weird FRAMEWORK env, we should not blow up — sglang is the safe
    default."""
    monkeypatch.setenv("FRAMEWORK", "tensorrt")
    assert _default_baseline_config().name == "baseline_sglang.yaml"


def test_default_profile_config_tracks_framework(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert _default_profile_config().name == "profile_vllm.yaml"
    monkeypatch.setenv("FRAMEWORK", "sglang")
    assert _default_profile_config().name == "profile_sglang.yaml"


def test_baseline_executor_picks_framework_yaml_at_call_time(tmp_path, monkeypatch):
    """No config_path override + FRAMEWORK=vllm => baseline_vllm.yaml is
    the resolved default, NOT baseline_sglang.yaml. This is the very
    regression that was blocking vllm users."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    pe = BaselineExecutor()
    # Default constructor leaves default_config_path=None so the resolver
    # is consulted at call time.
    assert pe.default_config_path is None
    assert pe._resolve_default_config().name == "baseline_vllm.yaml"


def test_profile_executor_picks_framework_yaml_at_call_time(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    pe = ProfileExecutor()
    assert pe.default_config_path is None
    assert pe._resolve_default_config().name == "profile_vllm.yaml"


@pytest.mark.asyncio
async def test_profile_executor_extracts_trace_dir(tmp_path):
    """When the workspace contains torch_trace/*.trace.json.gz, the
    runner surfaces them in the result so downstream consumers can
    feed them into tracelens_analysis.py."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    # Build a fake workspace dir matching what Magpie would create.
    output_dir = tmp_path / "out"
    workspace = output_dir / "benchmark_sglang_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": 3.2, "output_throughput": 800.0,
            "total_token_throughput": 1600.0, "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158},
                    "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
    }))
    trace_dir = workspace / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "TP-0_main.trace.json.gz").write_bytes(b"fake-trace")
    (trace_dir / "TP-0_aux.trace.json.gz").write_bytes(b"fake-trace")

    # Stub subprocess.run so we don't actually launch sglang.
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr="",
    )
    def _fake_run(*args, **kwargs):
        return fake_completed

    pe = ProfileExecutor(default_output_root=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-1",
    )
    sub.register_executor("profile", pe)
    with patch("subprocess.run", side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result["framework"] == "sglang"
    assert res.result["trace_dir"] == str(trace_dir)
    assert len(res.result["trace_files"]) == 2
    assert "main_trace_path" in res.result
    db.close()


# ===========================================================================
# kernel_request_handlers — direct unit
# ===========================================================================
@pytest.mark.asyncio
async def test_select_kernels_handler_dry_run_returns_structured_result(session_dir):
    """Tracelens tool always emits structured JSON (even on validation
    failure). Our handler must surface it verbatim — including ``status``
    + run_id + session_id — so callers can debug without parsing logs."""
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    payload = {
        "trace_input": str(fake_trace),
        "session_id": session_dir.name,
        "model_name": "Qwen3-8B",
        "framework": "sglang",
        "top_k": 5,
        "dry_run": True,
        "budget_minutes": 1,
    }
    res = await krh.select_kernels_handler(payload, session_dir=session_dir)
    # The tool will return failed because the dir has no trace files,
    # but the response must be structured (not generic returncode-only).
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "tool" in res or "run_id" in res or "error" in res
    assert res.get("session_id") == session_dir.name or "run_id" in res


@pytest.mark.asyncio
async def test_select_kernels_handler_surfaces_candidates_path(session_dir, monkeypatch):
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "artifact_paths": {
                "kernel_candidates": "/tmp/kernel_candidates.json",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {
            "trace_input": str(session_dir),
            "dry_run": True,
            "capture_folder": "/tmp/capture_traces",
        },
        session_dir=session_dir,
    )
    assert res["candidates_path"] == "/tmp/kernel_candidates.json"
    assert "--capture-folder" in captured["cmd"]
    assert "/tmp/capture_traces" in captured["cmd"]


@pytest.mark.asyncio
async def test_select_kernels_handler_backfills_workload_context_from_state(
    session_dir, monkeypatch,
):
    """When the payload omits framework/gpu_type/model, the handler must
    fall back to SharedState so tracelens_analysis.py receives the real
    workload context (vllm/MI300X/Qwen3-30B-A3B/inference) instead of
    the script defaults (""/MI355X/default)."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.framework = "vllm"
    state.gpu_type = "mi300x"
    state.model_path = "/wekafs/models/Qwen3-30B-A3B"
    state.model_name = "Qwen3-30B-A3B"
    state.save(session_dir)

    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok"}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--framework" in cmd and "vllm" in cmd
    assert "--target-platform" in cmd and "mi300x" in cmd
    assert "--model-name" in cmd and "Qwen3-30B-A3B" in cmd
    assert "--analysis-mode" in cmd and "inference" in cmd


@pytest.mark.asyncio
async def test_select_kernels_handler_missing_trace_input(session_dir):
    res = await krh.select_kernels_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "trace_input" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_missing_kernel_id(session_dir):
    res = await krh.run_optimization_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "kernel_id" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_dry_run(session_dir):
    payload = {
        "kernel_id": "fake_kernel_1",
        "session_id": session_dir.name,
        "dry_run": True,
        "budget_minutes": 1,
    }
    res = await krh.run_optimization_handler(payload, session_dir=session_dir)
    assert res.get("status") in ("ok", "succeeded", "failed")  # dry-run may still fail validation


def test_handlers_dispatch_table():
    """P2-2 only registered select_kernels + run_optimization. P2-4
    added apply_patch + integrate (covered in test_p2_4_integrate_report)."""
    assert krh.has_handler("select_kernels")
    assert krh.has_handler("run_optimization")
    assert not krh.has_handler("totally_unknown_kind")


# ===========================================================================
# Coordinator — REQUEST programmatic handler integration
# ===========================================================================
@pytest.mark.asyncio
async def test_coordinator_request_select_kernels_uses_handler(session_dir):
    """When Orchestration emits REQUEST{kind=select_kernels}, the Coordinator
    should run the registered handler programmatically and emit RESPONSE
    on the bus *without* waiting for the Kernel LLM."""
    c = Coordinator(session_dir, backends=_backends_silent())

    captured: dict = {}

    async def fake_handler(payload, *, session_dir):
        captured["payload"] = payload
        captured["session_dir"] = session_dir
        return {"status": "ok", "hot_kernels": ["kernel_a", "kernel_b"]}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"select_kernels": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "select_kernels",
                    "params": {"trace_input": "/tmp/fake-trace.json.gz"},
                },
            ))
            req_msgs = await c.bus.tail(topic="request", to_agent="kernel")
            assert req_msgs, "request must be mirrored to kernel inbox"
            req_id = req_msgs[0].msg_id

            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs, "handler must emit RESPONSE without LLM"
            r = resp_msgs[0]
            assert r.from_agent == "kernel"
            assert r.payload["kind"] == "select_kernels_done"
            assert r.payload["status"] == "ok"
            assert r.payload["result"]["hot_kernels"] == ["kernel_a", "kernel_b"]
            assert r.payload["in_reply_to"] == req_id
            assert r.payload["source"] == "programmatic_handler"

            # And the handler did receive merged payload (params flattened in).
            assert captured["payload"].get("trace_input") == "/tmp/fake-trace.json.gz"
            assert captured["session_dir"] == session_dir
        finally:
            await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_unknown_kind_routes_to_llm(session_dir):
    """REQUEST whose kind has no handler is mirrored to kernel inbox
    (LLM responder path) — no auto-RESPONSE."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        await c._handle_intent("orchestration", Intent(
            type=IntentType.REQUEST,
            payload={
                "target_agent": "kernel",
                "kind": "invent_brand_new_kind",  # NOT in registry
            },
        ))
        req_msgs = await c.bus.tail(topic="request", to_agent="kernel")
        assert req_msgs, "request must be mirrored even when no handler"
        # No auto-response should have been emitted.
        resp_msgs = await c.bus.tail(topic="response")
        assert not resp_msgs
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_handler_exception_recorded(session_dir):
    """Handler crashes → RESPONSE.status='failed' + error_class set."""
    c = Coordinator(session_dir, backends=_backends_silent())

    async def bad_handler(payload, *, session_dir):
        raise RuntimeError("boom")

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"select_kernels": bad_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel", "kind": "select_kernels"},
            ))
            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs
            r = resp_msgs[0]
            assert r.payload["status"] == "failed"
            assert r.payload["result"]["error_class"] == "handler_exception"
            assert "boom" in r.payload["result"]["error"]
        finally:
            await c.stop()
