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
