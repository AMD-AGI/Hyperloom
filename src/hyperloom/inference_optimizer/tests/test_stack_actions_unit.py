# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Rung 3 (M1) stack-action data model (§8.2)."""

from __future__ import annotations

from hyperloom.orchestrator.framework.stack_actions import (
    EnablementStackAction,
    FrameworkRuntime,
    ProvisionResult,
)


# ---------------------------------------------------------------------------
# EnablementStackAction round-trip
# ---------------------------------------------------------------------------

def test_stack_action_round_trip():
    a = EnablementStackAction(
        kind="runtime_candidate",
        framework="vllm",
        gap_id="gap.enablement.missing_model_arch",
        capability="deepseek_v4",
        acquisition_method="wheel",
        index_url="https://rocm.index/whl",
        packages=("vllm",),
        expected_symbols=("deepseek_v4",),
        server_args="--tokenizer-mode deepseek_v4",
        envs={"VLLM_ROCM_USE_AITER": "1"},
        attempt_venv_root="/s/venv",
    )
    d = a.to_state()
    b = EnablementStackAction.from_state(d)
    assert b == a


def test_stack_action_from_state_defaults():
    b = EnablementStackAction.from_state({"framework": "SGLang"})
    assert b.framework == "sglang"  # lowercased
    assert b.kind == "runtime_candidate"
    assert b.acquisition_method == "none"
    assert b.packages == ()
    assert b.envs == {}


def test_stack_action_from_state_invalid_method_coerced():
    b = EnablementStackAction.from_state({"framework": "vllm", "acquisition_method": "compile_kernel"})
    assert b.acquisition_method == "none"


def test_stack_action_from_state_none():
    b = EnablementStackAction.from_state(None)
    assert b.framework == ""
    assert b.packages == ()


# ---------------------------------------------------------------------------
# FrameworkRuntime.to_runtime_override
# ---------------------------------------------------------------------------

def test_runtime_to_override_keys_match_apply_runtime_override():
    rt = FrameworkRuntime(
        bin_path="/s/venv/bin",
        python_path="/s/venv/bin/python",
        venv_root="/s/venv",
        pythonpath_prefix="/s/src/python",
    )
    ov = rt.to_runtime_override()
    # These are the exact keys apply_runtime_override recognizes.
    assert ov["path_prefix"] == "/s/venv/bin"
    assert ov["framework_bin"] == "/s/venv/bin"
    assert ov["framework_python"] == "/s/venv/bin/python"
    assert ov["framework_venv_root"] == "/s/venv"
    assert ov["pythonpath_prefix"] == "/s/src/python"


def test_runtime_to_override_omits_empty():
    rt = FrameworkRuntime(bin_path="/s/venv/bin")
    ov = rt.to_runtime_override()
    assert "pythonpath_prefix" not in ov
    assert "framework_python" not in ov
    assert "framework_venv_root" not in ov


def test_runtime_override_lands_in_yaml():
    """to_runtime_override output must flow through apply_runtime_override."""
    from hyperloom.orchestrator.actions.executors._grid_runner import apply_runtime_override

    rt = FrameworkRuntime(bin_path="/attempt/bin", python_path="/attempt/bin/python", venv_root="/attempt")
    envs: dict[str, str] = {}
    apply_runtime_override(envs, rt.to_runtime_override())
    assert "/attempt/bin" in envs["PATH"]
    assert envs["HYPERLOOM_FRAMEWORK_BIN"] == "/attempt/bin"
    assert envs["HYPERLOOM_FRAMEWORK_PYTHON"] == "/attempt/bin/python"
    assert envs["HYPERLOOM_FRAMEWORK_VENV_ROOT"] == "/attempt"


def test_runtime_round_trip():
    rt = FrameworkRuntime(bin_path="/b", python_path="/b/py", venv_root="/v", server_args="-x", envs={"A": "1"})
    assert FrameworkRuntime.from_state(rt.to_state()) == rt


# ---------------------------------------------------------------------------
# ProvisionResult
# ---------------------------------------------------------------------------

def test_provision_result_ok_false_propagation():
    r = ProvisionResult(ok=False, error="boom")
    assert r.ok is False
    d = r.to_state()
    assert d["ok"] is False
    assert d["error"] == "boom"
    r2 = ProvisionResult.from_state(d)
    assert r2.ok is False
    assert r2.error == "boom"


def test_provision_result_ok_round_trip():
    rt = FrameworkRuntime(bin_path="/b", venv_root="/v")
    r = ProvisionResult(ok=True, runtime=rt, installed_versions={"vllm": "0.21"}, log_path="/l")
    r2 = ProvisionResult.from_state(r.to_state())
    assert r2.ok is True
    assert r2.runtime.venv_root == "/v"
    assert r2.installed_versions == {"vllm": "0.21"}
