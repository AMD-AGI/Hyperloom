# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for :class:`runtime.in_memory_kb_client.InMemoryKBClient` contract promises."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.errors import KBValidationError
from hyperloom.agents.critic.runtime.in_memory_kb_client import InMemoryKBClient
from hyperloom.agents.critic.runtime.kb_client import KBListQuery


def _scope():
    return {
        "org": "hyperloom",
        "framework": "sglang",
        "model": "deepseek-r1",
        "model_family": "deepseek",
        "workload": "decode",
        "precision": "fp8",
    }


def test_upsert_creates_then_returns_existing():
    kb = InMemoryKBClient()
    out1 = kb.upsert(
        {
            "scope": _scope(),
            "kind": "pitfall",
            "slug": "mla-fp8-torch-compile-incompat",
            "importance": 0.5,
            "summary": "first",
            "metadata": {"topic": "t"},
        }
    )
    assert out1["created"] is True
    out2 = kb.upsert(
        {
            "scope": _scope(),
            "kind": "pitfall",
            "slug": "mla-fp8-torch-compile-incompat",
            "importance": 0.4,
            "summary": "second",
            "metadata": {"topic": "t2"},
        }
    )
    assert out2["created"] is False
    # Importance protection keeps the existing (higher) value.
    assert out2["row"]["importance"] == 0.5
    assert "importance_protected" in out2["warnings"]
    assert out2["row"]["summary"] == "second"
    assert out2["row"]["metadata"]["topic"] == "t2"


def test_upsert_missing_field_raises():
    kb = InMemoryKBClient()
    with pytest.raises(KBValidationError):
        kb.upsert({"kind": "pitfall", "slug": "x", "importance": 0.5})


def test_list_returns_only_matching_scope_after_normalisation():
    kb = InMemoryKBClient()
    kb.upsert(
        {
            "scope": _scope(),
            "kind": "pitfall",
            "slug": "a-pitfall1",
            "importance": 0.5,
        }
    )
    other = dict(_scope())
    other["framework"] = "vllm"
    kb.upsert(
        {
            "scope": other,
            "kind": "pitfall",
            "slug": "b-pitfall2",
            "importance": 0.5,
        }
    )
    out = kb.list(KBListQuery(scope_filter={"framework": "  SGLang "}))
    assert all(r["scope"]["framework"] == "sglang" for r in out["entries"])
    assert any(r["slug"] == "a-pitfall1" for r in out["entries"])


def test_list_scope_value_normalisation_handles_uppercase_in_filter_too():
    kb = InMemoryKBClient()
    kb.upsert({"scope": _scope(), "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5})
    out = kb.list(KBListQuery(scope_filter={"model": "DEEPSEEK-R1"}))
    assert out["entries"]
