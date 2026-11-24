# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A change this session saw stop a server before it was ready is not launched again inside a bigger one."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors.explore_prefilter.dead_changes import carried_dead_change, dead_changes

_WS = "ws-1"
_AITER_ON = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MOE": "1"}
_MOE_DEATH = (
    "(EngineCore_DP0 pid=7) ERROR [core.py:700] ValueError: Unquantized MoE backend 'ROCm AITER' does not support "
    "the deployment configuration.\n"
    "(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above."
)
_EMPTY_STACK: dict[str, Any] = {"extra_server_args": "", "extra_envs": {}}


def _failed(name: str, envs: dict[str, str], **row: Any) -> dict[str, Any]:
    return {
        "name": name,
        "round_id": "explore-001",
        "outcome": "FAILED",
        "workload_signature": _WS,
        "extra_server_args": "",
        "extra_envs": dict(envs),
        "error_class": "server_init_dead",
        "error_excerpt": _MOE_DEATH,
        **row,
    }


def _variant(envs: dict[str, str], args: str = "") -> dict[str, Any]:
    return {"extra_server_args": args, "extra_envs": dict(envs)}


def test_a_variant_carrying_a_dead_change_whole_is_named_with_the_cause():
    dead = dead_changes([_failed("v00_enable_aiter_moe", _AITER_ON)], workload_signature=_WS)
    found = carried_dead_change(_variant(_AITER_ON, "--cuda-graph-max-bs 64"), dead, stack=_EMPTY_STACK)
    assert found is not None
    assert "v00_enable_aiter_moe" in found.detail()
    assert "ValueError: Unquantized MoE backend 'ROCm AITER'" in found.detail()


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(_variant({"VLLM_ROCM_USE_AITER": "1"}), id="part-of-the-change"),
        pytest.param(_variant({**_AITER_ON, "VLLM_ROCM_USE_AITER_MOE": "0"}), id="another-value"),
        pytest.param(_variant({}, "--no-enforce-eager"), id="unrelated-change"),
    ],
)
def test_a_variant_that_does_not_carry_the_whole_change_launches(variant):
    dead = dead_changes([_failed("v00", _AITER_ON)], workload_signature=_WS)
    assert carried_dead_change(variant, dead, stack=_EMPTY_STACK) is None


def _change(**fields: Any) -> dict[str, Any]:
    return {"extra_server_args": "", "extra_envs": {}, **fields}


@pytest.mark.parametrize(
    ("change", "carrier", "other"),
    [
        pytest.param(
            _change(extra_server_args="--kv-cache-dtype fp8"),
            _change(extra_server_args="--max-num-seqs 64 --kv-cache-dtype fp8"),
            _change(extra_server_args="--kv-cache-dtype auto"),
            id="server-arg",
        ),
        pytest.param(
            _change(remove_args=["--enforce-eager"]),
            _change(remove_args=["--enforce-eager"], extra_server_args="--max-num-seqs 64"),
            _change(remove_args=["--trust-remote-code"]),
            id="remove-arg",
        ),
        pytest.param(
            _change(unset_envs=["VLLM_USE_TRITON_FLASH_ATTN"]),
            _change(unset_envs=["VLLM_USE_TRITON_FLASH_ATTN"], extra_envs={"HL_KNOB": "1"}),
            _change(unset_envs=["VLLM_ROCM_USE_AITER"]),
            id="unset-env",
        ),
        pytest.param(
            _change(extra_server_args="--max-model-len 4096", args_mode="replace"),
            _change(extra_server_args="--max-model-len 4096 --max-num-seqs 64", args_mode="replace"),
            _change(extra_server_args="--max-model-len 4096"),
            id="replace-mode",
        ),
    ],
)
def test_every_kind_of_change_is_carried_whole_or_not_at_all(change, carrier, other):
    dead = dead_changes([_failed("v00", {}, **change)], workload_signature=_WS)
    assert len(dead) == 1
    assert carried_dead_change(carrier, dead, stack=_EMPTY_STACK) is dead[0]
    assert carried_dead_change(other, dead, stack=_EMPTY_STACK) is None


def test_a_stack_that_already_serves_the_change_lifts_it():
    dead = dead_changes([_failed("v00", _AITER_ON)], workload_signature=_WS)
    stack = {"extra_server_args": "", "extra_envs": dict(_AITER_ON)}
    assert carried_dead_change(_variant(_AITER_ON, "--max-num-seqs 128"), dead, stack=stack) is None


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"error_excerpt": "torch.OutOfMemoryError: HIP out of memory"}, id="memory"),
        pytest.param({"error_excerpt": "ValueError: port 30000 is unavailable"}, id="host-evidence-in-a-refusal"),
        pytest.param({"error_excerpt": "OSError: [Errno 98] address already in use"}, id="port-in-use"),
        pytest.param({"error_excerpt": "MemoryError: Unable to allocate 12.0 GiB for an array"}, id="host-memory"),
        pytest.param({"error_excerpt": "OSError: [Errno 28] No space left on device"}, id="disk-full"),
        pytest.param({"error_excerpt": "RuntimeError: No HIP GPUs are available"}, id="no-device"),
        pytest.param({"error_excerpt": "huggingface_hub.errors.HfHubHTTPError: 429 Client Error"}, id="hub-429"),
        pytest.param(
            {
                "error_excerpt": "ValueError: Free memory on device (10.2/191.98 GiB) on startup is less than desired "
                "GPU memory utilization (0.9, 172.78 GiB)."
            },
            id="device-memory-capacity",
        ),
        pytest.param({"error_class": "engine_capacity"}, id="capacity"),
        pytest.param({"error_class": "round_timeout"}, id="not-a-start-failure"),
        pytest.param({"framework_changed": True}, id="code-change"),
        pytest.param({"workload_signature": "ws-2"}, id="other-workload"),
        pytest.param({"outcome": "REVERT"}, id="measured"),
        pytest.param({"extra_envs": {}}, id="no-change"),
        pytest.param({"error_excerpt": "the server exited with code 1 before it was ready"}, id="no-fault-named"),
    ],
)
def test_only_a_start_failure_of_the_variants_own_change_counts(row):
    assert dead_changes([_failed("v00", _AITER_ON, **row)], workload_signature=_WS) == []


def test_a_configuration_the_framework_rejects_by_validation_counts():
    row = _failed("v00", _AITER_ON, error_excerpt="pydantic_core._pydantic_core.ValidationError: 1 validation error")
    (dead,) = dead_changes([row], workload_signature=_WS)
    assert dead.cause.startswith("pydantic_core._pydantic_core.ValidationError")


def test_a_decision_round_failure_carries_its_log_text_as_the_reason():
    row = _failed("v00", _AITER_ON, error_excerpt=None, reason=_MOE_DEATH)
    (dead,) = dead_changes([row], workload_signature=_WS)
    assert dead.cause.startswith("ValueError: Unquantized MoE backend")
