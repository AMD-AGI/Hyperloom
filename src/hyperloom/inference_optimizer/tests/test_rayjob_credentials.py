# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for RayJob credential fan-out."""

from __future__ import annotations

from hyperloom.inference_optimizer.multi_node._internal.rayjob_credentials import rayjob_credential_fanout


def test_rayjob_credential_fanout_is_always_empty():
    env = {
        "SAFE_API_KEY": "secret-safe",
        "OPENAI_API_KEY": "secret-openai",
        "ANTHROPIC_API_KEY": "secret-anthropic",
        "LLM_API_KEY": "secret-llm",
        "LLM_API_BASE": "https://llm.example",
    }
    assert rayjob_credential_fanout(env) == {}
    assert rayjob_credential_fanout() == {}
