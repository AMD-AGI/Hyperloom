# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK receives native tunables but never inherits canonical performance credit."""

import shlex

import pytest

from hyperloom.inference_optimizer.agentx.geak_proxy import seed_native_geak_proxy
from hyperloom.inference_optimizer.agentx.identity import canonical_sha256


@pytest.mark.parametrize(
    ("framework", "argv"),
    [
        ("sglang", ["/runtime/bin/python", "-m", "sglang.launch_server", "--model-path", "/models/model"]),
        ("vllm", ["/runtime/bin/vllm", "serve", "/models/model"]),
    ],
)
def test_geak_effective_config_receives_observed_tunables_but_owns_proxy_endpoint(framework, argv):
    flags = ["--kv-cache-dtype", "fp8", "--json-option", '{"key": "two words"}']
    launch = {
        "effective_argv": [*argv, "--port", "8123", "--tensor-parallel-size", "4", *flags],
        "runtime_environment": {
            "PYTHONPATH": "/patched/runtime",
            "SGLANG_USE_AITER": "1",
            "HIP_VISIBLE_DEVICES": "0,1",
        },
    }
    launch["evidence_sha256"] = canonical_sha256(launch)
    handoff = {
        "framework": framework,
        "raw_baseline_tput": 200.0,
        "baseline_env_spec": {"config": {"extra_server_args": "--stale-delta"}, "source_snapshots": [{"id": "source"}]},
    }
    seed_native_geak_proxy(handoff, benchmark={}, measurement={"agentx_server_launch": launch})
    config = handoff["baseline_env_spec"]["config"]
    assert shlex.split(config["server_launch_flags"]) == flags
    assert config["extra_server_args"] == ""
    assert handoff["accepted_flags"] == ""
    assert config["extra_envs"] == {"PYTHONPATH": "/patched/runtime", "SGLANG_USE_AITER": "1", "PATH": "/runtime/bin"}
    assert handoff["baseline_env_spec"]["source_snapshots"] == [{"id": "source"}]
    assert handoff["native_server_launch"] == launch
    assert handoff["raw_baseline_tput"] == handoff["orchestrator_best_tput_same_config"] == 0.0
    assert handoff["same_config_reference_verification_status"] == "unverified_workload"


@pytest.mark.parametrize("evidence", [None, {}, {"effective_argv": ["python"], "evidence_sha256": "0" * 64}])
def test_geak_rejects_missing_or_changed_native_launch_evidence(evidence):
    with pytest.raises(ValueError, match="launch"):
        seed_native_geak_proxy({}, benchmark={}, measurement={"agentx_server_launch": evidence})
