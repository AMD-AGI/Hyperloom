# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for multi-node SSH env key validation.

``env_safety`` is the last-mile gate on keys that already entered a forward
dict: it blocks shell/loader injection vectors (``_DENY_KEYS``) and invalid
key shapes. Credential exclusion (``*_API_KEY``, ``SAFE_API_KEY``, ``*_BASE_URL``)
happens upstream in ``infera._collect_forward_env`` (prefix whitelist) and
``create-infera`` (operator ``--extra-env`` only); those keys are never placed
into the forward dict, so they are not SSH-forwarded to inference pods.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.multi_node._internal import env_safety
from hyperloom.common import env_safety as common_env_safety


def test_valid_mori_and_sglang_keys_allowed():
    assert env_safety.is_forward_env_key_allowed("MORI_DISPATCH_FOO")
    assert env_safety.is_forward_env_key_allowed("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK")
    assert env_safety.is_forward_env_key_allowed("SGLANG_USE_AITER")
    assert env_safety.is_forward_env_key_allowed("SGLANG_TORCH_PROFILER_DIR")


def test_invalid_key_shape_rejected():
    assert not env_safety.is_forward_env_key_allowed("X Y")
    assert not env_safety.is_forward_env_key_allowed("bad-key")


def test_denylist_keys_rejected():
    """Only exact _DENY_KEYS entries are blocked (loader/python/PATH/shell vectors)."""
    assert not env_safety.is_forward_env_key_allowed("LD_PRELOAD")
    assert not env_safety.is_forward_env_key_allowed("PYTHONPATH")
    assert not env_safety.is_forward_env_key_allowed("PATH")
    assert not env_safety.is_forward_env_key_allowed("IFS")


def test_non_denylist_tuning_keys_pass_shape_gate():
    """Keys outside _DENY_KEYS pass the low-level forward gate when present."""
    assert env_safety.is_forward_env_key_allowed("NCCL_IB_HCA")
    assert env_safety.is_forward_env_key_allowed("SGLANG_USE_AITER")


def test_filter_forward_env_drops_bad_keys():
    out = env_safety.filter_forward_env(
        {
            "MORI_FOO": "1",
            "LD_PRELOAD": "/evil.so",
            "X Y": "nope",
        },
        warn_on_drop=False,
    )
    assert out == {"MORI_FOO": "1"}


def test_assert_forward_env_keys_raises():
    with pytest.raises(ValueError, match="disallowed SSH forward env keys"):
        env_safety.assert_forward_env_keys({"LD_PRELOAD": "/tmp/x.so"})


def test_common_env_safety_filters_workload_dotenv_and_kernel_agent_keys():
    assert common_env_safety.is_allowed_workload_env_key("SGLANG_USE_AITER_FP8_PER_TOKEN")
    assert common_env_safety.is_allowed_workload_env_key("EXTRA_SGLANG_ARGS")
    assert not common_env_safety.is_allowed_workload_env_key("OPENAI_API_KEY")
    assert not common_env_safety.is_allowed_workload_env_key("LD_PRELOAD")

    assert common_env_safety.is_allowed_dotenv_key("OPENAI_API_KEY")
    assert common_env_safety.is_allowed_dotenv_key("HYPERLOOM_RUNTIME_DIR")
    assert not common_env_safety.is_allowed_dotenv_key("PYTHONPATH")
    assert not common_env_safety.is_allowed_dotenv_key("BAD-NAME")

    assert common_env_safety.is_allowed_kernel_agent_env_key("TRACELENS_ROOT")
    assert not common_env_safety.is_allowed_kernel_agent_env_key("TRACELENS_TOKEN")

    allowed, dropped = common_env_safety.filter_untrusted_env_mapping(
        {
            "bench_foo": 1,
            "OPENAI_API_KEY": "secret",
            "bad key": "nope",
            "": "empty",
        },
        allow_predicate=common_env_safety.is_allowed_workload_env_key,
    )
    assert allowed == {"BENCH_FOO": "1"}
    assert dropped == {
        "OPENAI_API_KEY": "not_allowed",
        "bad key": "invalid_env_key",
        "<empty>": "invalid_env_key",
    }

    env = {"LD_PRELOAD": "evil.so", "PATH": "/bin", "SAFE": "1"}
    assert common_env_safety.scrub_child_process_env(env) is env
    assert env == {"PATH": "/bin", "SAFE": "1"}
