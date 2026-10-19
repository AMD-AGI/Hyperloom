# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK inherits explicit launch settings beyond Magpie's runtime snapshot allowlist."""

from copy import deepcopy

import pytest

from hyperloom.inference_optimizer.agentx.geak_proxy import seed_native_geak_proxy


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_native_geak_preserves_explicit_env_and_unsets(framework):
    module = "sglang.launch_server" if framework == "sglang" else "vllm.entrypoints.openai.api_server"
    receipt = {
        "effective_argv": ["/opt/framework/bin/python", "-m", module],
        "runtime_environment": {"PATH": "/usr/bin", "SGLANG_USE_AITER": "1", "ROCR_VISIBLE_DEVICES": "4,5"},
        "effective_env": {"AMD_SERIALIZE_KERNEL": "3", "MALLOC_CONF": "background_thread:true", "AITER_REMOVED": None},
    }
    original = deepcopy(receipt)
    handoff = {
        "framework": framework,
        "baseline_env_spec": {"config": {"unset_envs": ["KEEP_UNSET", "AMD_SERIALIZE_KERNEL"]}},
    }
    seed_native_geak_proxy(
        handoff,
        benchmark={"agentx": {"enabled": True}},
        measurement={"launch_evidence": {"magpie_server_launch": receipt}},
    )
    config = handoff["baseline_env_spec"]["config"]
    assert config["extra_envs"] == {
        "PATH": "/opt/framework/bin:/usr/bin",
        "SGLANG_USE_AITER": "1",
        "AMD_SERIALIZE_KERNEL": "3",
        "MALLOC_CONF": "background_thread:true",
    }
    assert config["unset_envs"] == ["AITER_REMOVED", "KEEP_UNSET"]
    assert "AMD_SERIALIZE_KERNEL=3" in handoff["accepted_env"]
    assert "AITER_REMOVED=" not in handoff["accepted_env"]
    assert handoff["native_server_launch"] == original == receipt
