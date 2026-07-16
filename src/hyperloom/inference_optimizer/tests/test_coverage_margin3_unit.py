# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import TimeoutError as FuturesTimeoutError
from types import SimpleNamespace
import urllib.error

import pytest


def test_common_io_bytes_and_safe_mtime_edges(monkeypatch, tmp_path) -> None:
    from hyperloom.common import io

    path = tmp_path / "nested" / "payload.bin"
    io.atomic_write_bytes(path, b"abc", make_parents=True, fsync=True, mode=0o777)
    assert path.read_bytes() == b"abc"
    assert path.stat().st_mode & 0o777 == 0o700

    assert io.safe_mtime(tmp_path / "missing") == 0.0

    def _boom_replace(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", _boom_replace)
    with pytest.raises(OSError, match="replace failed"):
        io.atomic_write_bytes(tmp_path / "will_fail.bin", b"x")
    assert not list(tmp_path.glob(".will_fail.bin.*.tmp"))


def test_llm_config_parse_and_derive_edges() -> None:
    from hyperloom.common.llm_config import (
        claude_sdk_env_options,
        derive_openai_base_url,
        parse_custom_headers,
        resolve_openai_client_config,
    )

    assert parse_custom_headers(None) == {}
    assert parse_custom_headers("   ") == {}
    assert parse_custom_headers('{"X-Team": " hyperloom ", "": "drop"}') == {"X-Team": "hyperloom"}
    assert parse_custom_headers("{not json}\nX-Fallback: yes") == {"X-Fallback": "yes"}

    assert derive_openai_base_url(None) is None
    assert derive_openai_base_url("   ") is None
    assert derive_openai_base_url("https://gw.example/Unified") == "https://gw.example/Unified/v1"
    assert derive_openai_base_url("https://gw.example/custom") == "https://gw.example/custom"

    cfg = resolve_openai_client_config(
        api_key_env="CUSTOM_KEY",
        base_url_env="CUSTOM_BASE",
        env={
            "CUSTOM_KEY": " key ",
            "CUSTOM_BASE": " https://base.example/v1 ",
            "OPENAI_CUSTOM_HEADERS": '{"X-Trace": " 1 "}',
        },
    )
    assert cfg.as_kwargs() == {
        "api_key": "key",
        "base_url": "https://base.example/v1",
        "default_headers": {"X-Trace": "1"},
    }
    assert claude_sdk_env_options(env={}) == {}


def test_retry_policy_env_and_on_retry_error(monkeypatch) -> None:
    from hyperloom.orchestrator.roles import base
    from hyperloom.orchestrator.roles.base import RetryPolicy

    monkeypatch.setenv("HL_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("HL_RETRY_BASE_S", "bad")
    monkeypatch.setenv("HL_RETRY_MAX_S", "inf")
    monkeypatch.setenv("HL_RETRY_MULT", "-1")
    monkeypatch.setenv("HL_RETRY_JITTER_S", "2")
    policy = RetryPolicy.from_env("HL_RETRY")
    assert policy.max_attempts == 1
    assert policy.base_delay_s == 1.0
    assert policy.max_delay_s == 30.0
    assert policy.multiplier == 2.0
    assert policy.jitter_s == 2.0

    monkeypatch.setattr(base.random, "uniform", lambda _lo, _hi: 0.25)
    assert RetryPolicy(base_delay_s=2.0, max_delay_s=3.0, multiplier=10.0, jitter_s=0.5).delay_for(2) == 3.25


@pytest.mark.asyncio
async def test_retry_with_backoff_swallows_on_retry_callback_error() -> None:
    from hyperloom.orchestrator.roles.base import RetryPolicy, retry_with_backoff

    calls = {"n": 0}
    slept: list[float] = []

    async def _flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return "ok"

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    def _bad_on_retry(*_args):
        raise RuntimeError("telemetry failed")

    out = await retry_with_backoff(
        _flaky,
        policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, jitter_s=0.0),
        retry_on=(ConnectionError,),
        sleep=_sleep,
        on_retry=_bad_on_retry,
    )
    assert out == "ok"
    assert slept == [0.0]


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> str:
        return self._raw


def test_model_compat_whitelist_and_local_file_edges(tmp_path) -> None:
    import model_compat

    model_compat.load_whitelist.cache_clear()
    whitelist = tmp_path / "whitelist.json"
    whitelist.write_text(
        json.dumps({"candidates": [{"repo_id": "org/a"}, {"repo_id": ""}, {"no_repo": "ignored"}]}),
        encoding="utf-8",
    )
    assert model_compat.load_whitelist(str(whitelist)) == frozenset({"org/a"})
    assert model_compat.load_whitelist(str(tmp_path / "missing.json")) == frozenset()

    missing_dir = tmp_path / "missing-model"
    assert model_compat.has_weights(missing_dir) is False
    assert model_compat.has_tokenizer(missing_dir) is True


def test_hf_gated_rotates_auth_failures_to_gated(monkeypatch) -> None:
    import model_compat

    model_compat._tok_idx[0] = 0
    calls = {"n": 0}

    def _urlopen(_req, timeout):
        assert timeout == 20
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _urlopen)
    assert model_compat.hf_gated("org/model", ["tok-a", "tok-b"]) == "gated"
    assert calls["n"] == 3


def test_hf_gated_retries_rate_limit_and_generic_errors(monkeypatch) -> None:
    import model_compat

    model_compat._tok_idx[0] = 0
    sleeps: list[int] = []
    monkeypatch.setattr(model_compat.time, "sleep", lambda delay: sleeps.append(delay))

    calls = {"n": 0}

    def _rate_limit_then_open(_req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return _JsonResponse({"gated": False})

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _rate_limit_then_open)
    assert model_compat.hf_gated("org/model", ["tok"]) is None
    assert sleeps == [5]

    def _always_boom(_req, timeout):
        raise RuntimeError("network")

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _always_boom)
    assert model_compat.hf_gated("org/model", ["tok"]) is None


def test_hf_missing_tokenizer_rate_limit_and_http_fallbacks(monkeypatch) -> None:
    import model_compat

    model_compat._tok_idx[0] = 0
    monkeypatch.setattr(model_compat.time, "sleep", lambda *_args: None)
    calls = {"n": 0}

    def _rate_limit_then_missing(_req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return _JsonResponse({"siblings": [{"rfilename": "model.safetensors"}]})

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _rate_limit_then_missing)
    assert model_compat.hf_missing_tokenizer("org/model", ["tok"]) == "missing_tokenizer"

    def _server_error(_req, timeout):
        raise urllib.error.HTTPError("u", 500, "server", {}, None)

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", _server_error)
    assert model_compat.hf_missing_tokenizer("org/model", ["tok"]) is None


def test_gpu_type_resolution_env_probe_and_runner(monkeypatch) -> None:
    from hyperloom.inference_optimizer import gpu_types

    assert gpu_types._gpu_runner_type("MI325X") == "mi300x"
    assert gpu_types._gpu_runner_type("mi355x") == "mi355x"

    resolved, warnings = gpu_types._resolve_gpu_type("mi300x", "mi355x")
    assert resolved == "mi355x"
    assert "disagrees" in warnings[0]
    assert gpu_types._resolve_gpu_type("mi300x", "") == ("mi300x", [])

    monkeypatch.delenv("GPU_TYPE", raising=False)
    assert gpu_types._resolve_amd_gpu_type("mi308x") == "mi308x"
    assert gpu_types._resolve_amd_gpu_type("nvidia") is None

    monkeypatch.setenv("GPU_TYPE", " MI325X ")
    assert gpu_types._resolve_amd_gpu_type() == "mi325x"
    monkeypatch.setenv("GPU_TYPE", "unknown")
    assert gpu_types._resolve_amd_gpu_type() is None

    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda: "mi355x")
    assert gpu_types._resolve_amd_gpu_type() == "mi355x"
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda: "gfx000")
    assert gpu_types._resolve_amd_gpu_type() is None


def test_gpu_type_autodetect_rocm_and_torch_fallback(monkeypatch) -> None:
    from hyperloom.inference_optimizer import gpu_types

    class _Completed:
        stdout = "GPU[0] : Card series: AMD Instinct MI325X"

    def _rocm_ok(cmd, capture_output, text, timeout):
        assert cmd == ["rocm-smi", "--showproductname"]
        assert capture_output is True
        assert text is True
        assert timeout == 5
        return _Completed()

    monkeypatch.setattr(subprocess, "run", _rocm_ok)
    assert gpu_types._autodetect_gpu_type() == "mi325x"

    def _rocm_missing(*_args, **_kwargs):
        raise FileNotFoundError("rocm-smi")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _idx: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        )
    )
    monkeypatch.setattr(subprocess, "run", _rocm_missing)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert gpu_types._autodetect_gpu_type() == "mi355x"

    fake_torch.cuda.get_device_properties = lambda _idx: (_ for _ in ()).throw(RuntimeError("no gpu"))
    assert gpu_types._autodetect_gpu_type() is None


def test_action_registry_names_all_and_lazy_load(tmp_path) -> None:
    from hyperloom.orchestrator.actions.registry import ActionRegistry

    meta_dir = tmp_path / "_meta"
    meta_dir.mkdir()
    (meta_dir / "_ignored.yaml").write_text("name: ignored\n", encoding="utf-8")
    (meta_dir / "target_analysis.yaml").write_text(
        "\n".join(
            [
                "name: target_analysis",
                "family: prep",
                "cost_minutes_p50: 0.1",
                "cost_minutes_p75: 0.2",
                "expected_gain_pct: [0, 0]",
                "accuracy_risk: 0",
                "crash_risk: 0",
                "requires_lanes: []",
                "allowed_tools: [Read]",
                "side_effects: [writes_state]",
                "pipeline_phase: prep",
                "verdict_class: archival",
            ]
        ),
        encoding="utf-8",
    )

    reg = ActionRegistry(tmp_path)
    assert reg.names() == ["target_analysis"]
    assert [meta.name for meta in reg.all()] == ["target_analysis"]
    meta = reg.get("target_analysis")
    assert meta is not None
    assert meta.description == "target_analysis"
    assert reg.get("missing") is None


def test_kernel_decision_retry_budget_env(monkeypatch) -> None:
    from hyperloom.orchestrator.state import kernel_decision_settings as settings

    monkeypatch.delenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", raising=False)
    assert settings.resolve_kernel_opt_max_failures() == 2

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "0")
    assert settings.resolve_kernel_opt_max_failures() == 1

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "bad")
    assert settings.resolve_kernel_opt_max_failures() == 2


def test_dispatcher_inline_whitelist_filters_and_registry_errors(monkeypatch) -> None:
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    reg = SimpleNamespace(names=lambda: ["report", "missing", "lane_action", "ok_action"])
    coord = SimpleNamespace(
        action_registry=reg,
        sub=SimpleNamespace(executor_registry={"lane_action": object(), "ok_action": object()}),
        _INLINE_ACTION_DENY=frozenset({"report"}),
    )
    disp = DispatcherCollaborator(coord)
    monkeypatch.setattr(disp, "_registry_lanes_ttl", lambda name: (["gpu"] if name == "lane_action" else [], 60))
    assert disp._inline_action_whitelist() == frozenset({"ok_action"})

    coord.action_registry = SimpleNamespace(names=lambda: (_ for _ in ()).throw(RuntimeError("bad registry")))
    assert disp._inline_action_whitelist() == frozenset()

    coord.action_registry = None
    assert disp._inline_action_whitelist() == frozenset()


def test_dispatcher_inline_whitelist_all_fallback(monkeypatch) -> None:
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    reg = SimpleNamespace(all=lambda: [SimpleNamespace(name="from_all")])
    coord = SimpleNamespace(
        action_registry=reg,
        sub=SimpleNamespace(executor_registry={"from_all": object()}),
        _INLINE_ACTION_DENY=frozenset(),
    )
    disp = DispatcherCollaborator(coord)
    monkeypatch.setattr(disp, "_registry_lanes_ttl", lambda _name: ([], 60))
    assert disp._inline_action_whitelist() == frozenset({"from_all"})


def test_dispatcher_run_action_now_sync_edge_returns(monkeypatch) -> None:
    from hyperloom.orchestrator.loop import dispatcher as dispatcher_mod
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator

    coord = SimpleNamespace(
        _inline_fast_actions_enabled=True,
        _coordinator_loop=None,
        _INLINE_ACTION_DENY=frozenset(),
        action_registry=None,
        sub=SimpleNamespace(executor_registry={}),
    )
    disp = DispatcherCollaborator(coord)
    assert "action_name required" in disp._run_action_now_sync("  ", {})

    monkeypatch.setattr(disp, "_inline_action_whitelist", lambda: frozenset({"probe"}))
    assert "coordinator loop not running" in disp._run_action_now_sync("probe", {})

    coord._coordinator_loop = SimpleNamespace(is_closed=lambda: False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "not-a-float")
    monkeypatch.setattr(disp, "_run_action_now", lambda _name, _params: object())

    class _TimeoutFuture:
        def result(self, timeout):
            assert timeout == 120.0
            raise FuturesTimeoutError()

    monkeypatch.setattr(dispatcher_mod.asyncio, "run_coroutine_threadsafe", lambda _coro, _loop: _TimeoutFuture())
    assert "still running after 120s" in disp._run_action_now_sync("probe", {})

    class _ErrorFuture:
        def result(self, timeout):
            raise RuntimeError("boom")

    monkeypatch.setattr(dispatcher_mod.asyncio, "run_coroutine_threadsafe", lambda _coro, _loop: _ErrorFuture())
    assert "errored" in disp._run_action_now_sync("probe", {})

    monkeypatch.setattr(
        dispatcher_mod.asyncio,
        "run_coroutine_threadsafe",
        lambda _coro, _loop: (_ for _ in ()).throw(RuntimeError("closed")),
    )
    assert "could not schedule" in disp._run_action_now_sync("probe", {})


def test_multi_node_state_paths_resolution_and_migration(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    with pytest.raises(RuntimeError, match="cannot resolve"):
        state_paths.resolve_state_file()

    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(explicit))
    assert state_paths.resolve_state_file() == explicit

    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    session = tmp_path / "session"
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session))
    assert state_paths.resolve_state_file() == session / "runtime" / "multi_node_state.json"

    missing = tmp_path / "missing.json"
    assert state_paths.state_file_safe_to_read(missing) is False
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text("{}", encoding="utf-8")
    unsafe.chmod(0o666)
    assert state_paths.state_file_safe_to_read(unsafe) is False
    unsafe.chmod(0o600)
    assert state_paths.state_file_safe_to_read(unsafe) is True

    src = tmp_path / "source_state.json"
    src.write_text('{"nodes": []}', encoding="utf-8")
    src.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(src))
    bound = state_paths.bind_state_file_to_session(session)
    assert bound == session / "runtime" / "multi_node_state.json"
    assert bound.read_text(encoding="utf-8") == '{"nodes": []}'
    assert state_paths.resolve_state_file() == bound
    assert bound.stat().st_mode & 0o777 == 0o600
    assert bound.parent.stat().st_mode & 0o777 == 0o700


def test_multi_node_state_paths_warn_on_permission_failures(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths

    messages: list[str] = []
    monkeypatch.setattr(state_paths, "warn", messages.append)

    class _BadPath:
        def chmod(self, _mode):
            raise OSError("chmod denied")

    state_paths._chmod_state_file(_BadPath())
    assert "could not chmod state file" in messages[-1]

    runtime_dir = tmp_path / "runtime"
    original_chmod = type(runtime_dir).chmod

    def _bad_chmod(self, mode):
        if self == runtime_dir:
            raise OSError("runtime chmod denied")
        return original_chmod(self, mode)

    monkeypatch.setattr(type(runtime_dir), "chmod", _bad_chmod)
    state_paths._ensure_runtime_dir(runtime_dir)
    assert runtime_dir.is_dir()
    assert "could not chmod runtime dir" in messages[-1]


def test_llm_prompt_parse_response_edges() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters.llm_prompt import parse_llm_response

    fenced = """```json
{"executive_summary": "  ok  ", "section_narratives": {"a": "  first  ", "2": "two"}}
```"""
    parsed = parse_llm_response(fenced)
    assert parsed["executive_summary"] == "ok"
    assert parsed["section_narratives"] == {"a": "first", "2": "two"}

    assert parse_llm_response("not json") == {"executive_summary": "", "section_narratives": {}}
    assert parse_llm_response("[]") == {"executive_summary": "", "section_narratives": {}}
