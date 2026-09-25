# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the AgentX execution-boundary helper maybe_prepare_agentx."""

from __future__ import annotations

import subprocess

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import runtime
from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_agentx_runtime

_DEPLOY = "hyperloom.inference_optimizer.agentx.deploy.deploy_agentx_assets"
_RESOLVE = "hyperloom.inference_optimizer.agentx.preflight.resolve_aiperf_bin"
_CHECK = "hyperloom.inference_optimizer.agentx.preflight.check_aiperf_capability"


@pytest.fixture(autouse=True)
def _clear_memo():
    runtime._PREFLIGHTED_BINS.clear()
    yield
    runtime._PREFLIGHTED_BINS.clear()


def _cfg(tmp_path, script):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "benchmark_script": script}}),
        encoding="utf-8",
    )
    return p


def _native_cfg(tmp_path, *, model_path: str = ""):
    launcher = tmp_path / "benchmarks" / "single_node" / "agentic" / "model.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/usr/bin/env bash\nSGLANG_CMD=(\n  python3 -m sglang.launch_server\n)\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "native.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "agentx": "enable",
                    "benchmark_script": "single_node/agentic/model.sh",
                    "envs": {"MODEL_PATH": model_path} if model_path else {},
                }
            }
        ),
        encoding="utf-8",
    )
    return cfg, launcher


def _pinned_native_cfg(tmp_path, monkeypatch):
    cfg, _launcher = _native_cfg(tmp_path)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    benchmark = data["benchmark"]
    benchmark.update(
        {
            "model": "amd/model",
            "precision": "fp4",
            "docker_image": "example/sglang:agentx",
            "envs": {"CONC": 8},
        }
    )
    topology = {
        "tp": 4,
        "pp": 1,
        "pcp_size": 1,
        "ep": 4,
        "gpu_count": 4,
        "conc": 8,
        "duration_seconds": 3600,
        "recipe_fingerprint": "a" * 64,
    }
    recipe = {
        "name": "recipe-a",
        "config_file": "configs/amd-master.yaml",
        "recipe_fingerprint": "a" * 64,
        "image": "example/sglang:agentx",
        "runner": "cluster:mi355x-amds",
        "model": "amd/model",
        "model_prefix": "model",
        "framework": "sglang",
        "precision": "fp4",
        "concurrency": 8,
        "duration_seconds": 3600,
        "launcher": "single_node/agentic/model.sh",
    }
    execution = {
        "inferencex_commit": "d" * 40,
        "magpie_commit": "c" * 40,
        "static_execution_fingerprint": "b" * 64,
        "execution_fingerprint": "b" * 64,
    }
    benchmark["workload_spec"] = {
        "resolved_topology": dict(topology),
        "recipe": dict(recipe),
        "execution": dict(execution),
    }
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    def _preview(current, *, inferencex_path):
        current_conc = int(current.get("envs", {}).get("CONC", 8))
        current_image = str(current.get("docker_image") or recipe["image"])
        current_topology = {**topology, "conc": current_conc}
        return {
            "benchmark": dict(current),
            "entry": {
                "image": current_image,
                "runner": recipe["runner"],
                "model": recipe["model"],
                "model-prefix": recipe["model_prefix"],
                "framework": recipe["framework"],
                "precision": recipe["precision"],
            },
            "recipe": recipe["name"],
            "config_file": recipe["config_file"],
            "topology": current_topology,
            "magpie_execution": {"fingerprint": "e" * 64, "source_commit": "c" * 40},
        }

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.preview_native_recipe",
        _preview,
    )
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.native_execution_identity",
        lambda **_kwargs: dict(execution),
    )
    env = {
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT": "a" * 64,
        "HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT": "b" * 64,
        "HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT": "b" * 64,
        "HYPERLOOM_AGENTX_GPU_COUNT": "4",
        "MAGPIE_REF": "c" * 40,
        "INFERENCEX_REF": "d" * 40,
    }
    return cfg, env


def _pinned_profile_compat_cfg(tmp_path):
    output_dir = tmp_path / "profile-output"
    output_dir.mkdir()
    checkout = output_dir / ".agentx-profile-inferencex"
    benchmark_lib = checkout / "benchmarks" / "benchmark_lib.sh"
    benchmark_lib.parent.mkdir(parents=True)
    benchmark_lib.write_text("#!/bin/sh\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Hyperloom Test",
            "-c",
            "user.email=hyperloom@example.invalid",
            "commit",
            "-qm",
            "profile fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = output_dir / "baseline_config.with_envs.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "benchmark_script": "aiperf_client.sh",
                    "inferencex_path": str(checkout),
                    "envs": {"PROFILE": "1"},
                    "profiler": {"torch_profiler": {"enabled": True}},
                    "workload_spec": {
                        "kind": "agentx_trace_replay",
                        "harness": "hyperloom-profiler-compat",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    env = {
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT": "a" * 64,
        "HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT": "b" * 64,
        "HYPERLOOM_AGENTX_GPU_COUNT": "4",
        "MAGPIE_REF": "c" * 40,
        "INFERENCEX_REF": head,
    }
    return config, checkout, env


def test_noop_when_not_aiperf_script(tmp_path, monkeypatch):
    calls = {"deploy": 0, "preflight": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: calls.__setitem__("deploy", calls["deploy"] + 1))
    monkeypatch.setattr(_CHECK, lambda b, **k: calls.__setitem__("preflight", calls["preflight"] + 1))
    cfg = _cfg(tmp_path, "vllm_mi300x.sh")
    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg) is False
    assert calls == {"deploy": 0, "preflight": 0}


def test_deploy_before_preflight_with_resolved_bin(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(_DEPLOY, lambda d: order.append("deploy"))
    monkeypatch.setattr(_RESOLVE, lambda env: "/venv/bin/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: order.append(("preflight", b)))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg) is True
    assert order == ["deploy", ("preflight", "/venv/bin/aiperf")]


def test_native_agentx_validates_launcher_without_modifying_checkout(tmp_path, monkeypatch):
    calls = {"deploy": 0, "preflight": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: calls.__setitem__("deploy", calls["deploy"] + 1))
    monkeypatch.setattr(_CHECK, lambda b, **k: calls.__setitem__("preflight", calls["preflight"] + 1))
    cfg, launcher = _native_cfg(tmp_path)
    original = launcher.read_bytes()

    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg) is True

    assert calls == {"deploy": 0, "preflight": 0}
    assert launcher.read_bytes() == original


def test_native_agentx_uses_checkout_pinned_in_materialized_yaml(tmp_path):
    cfg, launcher = _native_cfg(tmp_path)
    original = launcher.read_bytes()
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["benchmark"]["inferencex_path"] = str(tmp_path)
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert runtime.maybe_prepare_agentx(env={}, inferencex_path="", config_path=cfg) is True

    assert launcher.read_bytes() == original


def test_native_agentx_scrubs_remote_model_id_from_child_environment(tmp_path):
    cfg, _launcher = _native_cfg(tmp_path)
    child = {"MODEL_PATH": "amd/GLM-5.2-MXFP4", "PATH": "/opt/venv/bin"}

    assert (
        runtime.maybe_prepare_agentx(
            env=child,
            inferencex_path=str(tmp_path),
            config_path=cfg,
        )
        is True
    )

    assert "MODEL_PATH" not in child


def test_native_agentx_keeps_materialized_local_model_path(tmp_path):
    cfg, _launcher = _native_cfg(tmp_path, model_path="/models/glm")
    child = {"MODEL_PATH": "amd/GLM-5.2-MXFP4"}

    assert (
        runtime.maybe_prepare_agentx(
            env=child,
            inferencex_path=str(tmp_path),
            config_path=cfg,
        )
        is True
    )

    assert child["MODEL_PATH"] == "/models/glm"


def test_pinned_native_agentx_revalidates_unchanged_materialization(tmp_path, monkeypatch):
    cfg, env = _pinned_native_cfg(tmp_path, monkeypatch)

    assert (
        runtime.maybe_prepare_agentx(
            env=env,
            inferencex_path=str(tmp_path),
            config_path=cfg,
        )
        is True
    )


def test_pinned_native_agentx_rejects_forwarded_env_changed_after_acceptance(
    tmp_path,
    monkeypatch,
):
    cfg, env = _pinned_native_cfg(tmp_path, monkeypatch)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    benchmark = data["benchmark"]
    benchmark["envs"]["PYTHONPATH"] = "/candidate/a"
    saved_execution = benchmark["workload_spec"]["execution"]
    saved_execution["launch_config_sha256"] = "f" * 64
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    def _identity(**kwargs):
        current = dict(saved_execution)
        if kwargs["resolved_benchmark"]["envs"].get("PYTHONPATH") != "/candidate/a":
            current["launch_config_sha256"] = "0" * 64
            current["execution_fingerprint"] = "1" * 64
        return current

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.native_execution_identity",
        _identity,
    )
    assert (
        runtime.maybe_prepare_agentx(
            env=env,
            inferencex_path=str(tmp_path),
            config_path=cfg,
        )
        is True
    )

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["benchmark"]["envs"]["PYTHONPATH"] = "/candidate/b"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved BenchmarkConfig changed"):
        runtime.maybe_prepare_agentx(
            env=env,
            inferencex_path=str(tmp_path),
            config_path=cfg,
        )


def test_pinned_native_agentx_cannot_be_disabled_in_materialized_yaml(tmp_path, monkeypatch):
    cfg, env = _pinned_native_cfg(tmp_path, monkeypatch)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["benchmark"]["agentx"] = False
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="requires benchmark.agentx"):
        runtime.maybe_prepare_agentx(env=env, inferencex_path=str(tmp_path), config_path=cfg)


def test_pinned_agentx_profile_compatibility_path_is_explicitly_allowed(
    tmp_path,
    monkeypatch,
):
    config, checkout, env = _pinned_profile_compat_cfg(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(_DEPLOY, lambda path: calls.append(("deploy", path)))
    monkeypatch.setattr(_RESOLVE, lambda _env: "/venv/bin/aiperf")
    monkeypatch.setattr(
        _CHECK,
        lambda binary, **kwargs: calls.append(("preflight", binary, kwargs)),
    )

    assert runtime.maybe_prepare_agentx(
        env=env,
        inferencex_path=str(checkout),
        config_path=config,
        allow_profile_compat=True,
    )

    assert calls[0] == ("deploy", checkout / "benchmarks")
    assert calls[1][0:2] == ("preflight", "/venv/bin/aiperf")
    assert calls[1][2]["require_progress_api"] is True


def test_pinned_agentx_profile_compatibility_rejects_shell_unsafe_checkout_path(
    tmp_path,
):
    unsafe_root = tmp_path / "profile root;unsafe"
    unsafe_root.mkdir()
    config, checkout, env = _pinned_profile_compat_cfg(unsafe_root)

    with pytest.raises(ValueError, match="shell-unsafe"):
        runtime.maybe_prepare_agentx(
            env=env,
            inferencex_path=str(checkout),
            config_path=config,
            allow_profile_compat=True,
        )


def test_pinned_agentx_generic_client_still_fails_closed(tmp_path):
    config, checkout, env = _pinned_profile_compat_cfg(tmp_path)
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["benchmark"]["workload_spec"].pop("harness")
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic harness identity"):
        runtime.maybe_prepare_agentx(
            env=env,
            inferencex_path=str(checkout),
            config_path=config,
            allow_profile_compat=True,
        )


def test_pinned_native_agentx_requires_session_owned_execution_identity(tmp_path, monkeypatch):
    cfg, env = _pinned_native_cfg(tmp_path, monkeypatch)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["benchmark"]["workload_spec"].pop("execution")
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing workload_spec.execution"):
        runtime.maybe_prepare_agentx(env=env, inferencex_path=str(tmp_path), config_path=cfg)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("CONC", 16, "resolved topology changed"),
        ("docker_image", "attacker/image:latest", "recipe identity changed"),
    ],
)
def test_pinned_native_agentx_rejects_recipe_inputs_changed_after_materialization(
    tmp_path,
    monkeypatch,
    field,
    value,
    match,
):
    cfg, env = _pinned_native_cfg(tmp_path, monkeypatch)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    if field == "CONC":
        data["benchmark"]["envs"][field] = value
    else:
        data["benchmark"][field] = value
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        runtime.maybe_prepare_agentx(env=env, inferencex_path=str(tmp_path), config_path=cfg)


def test_preflight_memoized_per_bin(tmp_path, monkeypatch):
    n = {"p": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: n.__setitem__("p", n["p"] + 1))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    assert n["p"] == 1  # second call reuses the memoized capability result


def test_profile_config_requires_progress_api(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: seen.append(k))
    cfg = tmp_path / "profile.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "benchmark_script": "aiperf_client.sh",
                    "envs": {"PROFILE": "1"},
                }
            }
        ),
        encoding="utf-8",
    )

    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)

    assert seen == [{"env": {}, "require_progress_api": True}]


def test_stronger_progress_api_preflight_satisfies_later_basic_check(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: seen.append(k))
    profile_cfg = tmp_path / "profile.yaml"
    profile_cfg.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "benchmark_script": "aiperf_client.sh",
                    "envs": {"PROFILE": "1"},
                }
            }
        ),
        encoding="utf-8",
    )
    baseline_cfg = _cfg(tmp_path, "aiperf_client.sh")

    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=profile_cfg)
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=baseline_cfg)

    assert seen == [{"env": {}, "require_progress_api": True}]


def test_incapable_bin_not_memoized(tmp_path, monkeypatch):
    n = {"p": 0}

    def _raise(b, **k):
        n["p"] += 1
        raise AgentXPreflightError("no weka-trace")

    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, _raise)
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    with pytest.raises(AgentXPreflightError):
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    with pytest.raises(AgentXPreflightError):
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    assert n["p"] == 2  # a failed preflight is re-checked, not memoized


# ── prepare_agentx_runtime: shared gate used by grid AND baseline/profile ─────
def test_prepare_runtime_noop_under_pytest(tmp_path, monkeypatch):
    """The pytest self-disable short-circuits even when AgentX is ON."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")  # PYTEST_CURRENT_TEST is set by pytest
    calls = {"d": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: calls.__setitem__("d", calls["d"] + 1))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    assert prepare_agentx_runtime(env={}, inferencex_path=str(tmp_path), config_path=cfg) is None
    assert calls["d"] == 0


def test_prepare_runtime_off_noop(tmp_path, monkeypatch):
    """OFF returns None and never deploys (A2: agentx package not imported)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    calls = {"d": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: calls.__setitem__("d", calls["d"] + 1))
    cfg = _cfg(tmp_path, "vllm_mi300x.sh")
    assert prepare_agentx_runtime(env={}, inferencex_path=str(tmp_path), config_path=cfg) is None
    assert calls["d"] == 0


def test_prepare_runtime_uses_explicit_persisted_agentx_decision(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    order = []
    monkeypatch.setattr(_DEPLOY, lambda d: order.append("deploy"))
    monkeypatch.setattr(_RESOLVE, lambda env: "/venv/bin/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: order.append("preflight"))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    assert (
        prepare_agentx_runtime(
            env={},
            inferencex_path=str(tmp_path),
            config_path=cfg,
            active=True,
        )
        is None
    )
    assert order == ["deploy", "preflight"]


def test_prepare_runtime_uses_materialized_agentx_script_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    order = []
    monkeypatch.setattr(_DEPLOY, lambda d: order.append("deploy"))
    monkeypatch.setattr(_RESOLVE, lambda env: "/venv/bin/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: order.append("preflight"))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    assert prepare_agentx_runtime(env={}, inferencex_path=str(tmp_path), config_path=cfg) is None
    assert order == ["deploy", "preflight"]


def test_prepare_runtime_detects_native_agentx_without_ambient_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    cfg, launcher = _native_cfg(tmp_path)
    original = launcher.read_bytes()
    assert prepare_agentx_runtime(env={}, inferencex_path=str(tmp_path), config_path=cfg) is None
    assert launcher.read_bytes() == original


def test_prepare_runtime_on_deploys_returns_none(tmp_path, monkeypatch):
    """ON deploys the client + preflights, returning None on success."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    order = []
    monkeypatch.setattr(_DEPLOY, lambda d: order.append("deploy"))
    monkeypatch.setattr(_RESOLVE, lambda env: "/venv/bin/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: order.append("preflight"))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    assert prepare_agentx_runtime(env={}, inferencex_path=str(tmp_path), config_path=cfg) is None
    assert order == ["deploy", "preflight"]


def test_prepare_runtime_preflight_error_returns_string(tmp_path, monkeypatch):
    """A failed preflight is returned as an error string (caller surfaces it)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")

    def _raise(b, **k):
        raise AgentXPreflightError("no weka-trace capability")

    monkeypatch.setattr(_CHECK, _raise)
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    msg = prepare_agentx_runtime(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    assert msg is not None and "AgentX preflight failed" in msg and "weka-trace" in msg
