# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the explore aiter-MoE pin filter."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import (
    apply_aiter_moe_pin_filter,
)


def _v(name: str, *, args: str = "", envs: dict | None = None) -> GridVariant:
    return GridVariant(name=name, extra_server_args=args, extra_envs=envs or {})


@pytest.fixture(autouse=True)
def _clear_pin(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_EXTRA_ENV", raising=False)


def test_noop_when_aiter_not_pinned(monkeypatch):
    # No operator pin -> filter is a strict no-op even for aiter-MoE variants.
    grid = [
        _v("aiter-moe", args="--moe-runner-backend aiter"),
        _v("master-switch", envs={"SGLANG_USE_AITER": "1"}),
    ]
    kept, dropped = apply_aiter_moe_pin_filter(grid)
    assert [v.name for v in kept] == ["aiter-moe", "master-switch"]
    assert dropped == []


def test_drops_aiter_moe_variants_when_pinned_off(monkeypatch):
    # Operator pinned SGLANG_USE_AITER=0 -> variants re-enabling the aiter MoE
    # runner (master switch or explicit --moe-runner-backend aiter) are dropped,
    # while aiter *attention* (MoE stays triton) is kept.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"SGLANG_USE_AITER": "0"}')
    grid = [
        _v("attn-aiter", args="--moe-runner-backend triton --attention-backend aiter"),
        _v("moe-aiter", args="--attention-backend aiter --moe-runner-backend aiter"),
        _v("master-switch", args="--moe-runner-backend triton", envs={"SGLANG_USE_AITER": "1"}),
        _v("allreduce", args="--enable-aiter-allreduce-fusion"),
    ]
    kept, dropped = apply_aiter_moe_pin_filter(grid)

    assert [v.name for v in kept] == ["attn-aiter", "allreduce"]
    dropped_names = sorted(d["name"] for d in dropped)
    assert dropped_names == ["master-switch", "moe-aiter"]
    assert all(d["source"] == "aiter_moe_pinned_off" for d in dropped)


def test_pin_truthy_value_does_not_trigger(monkeypatch):
    # SGLANG_USE_AITER=1 (aiter ON) is not a pin-off -> no filtering.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"SGLANG_USE_AITER": "1"}')
    grid = [_v("moe-aiter", args="--moe-runner-backend aiter")]
    kept, dropped = apply_aiter_moe_pin_filter(grid)
    assert [v.name for v in kept] == ["moe-aiter"]
    assert dropped == []


def test_malformed_pin_env_is_ignored(monkeypatch):
    # Invalid JSON handoff -> treated as no pin (no crash, no filtering).
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "{not json")
    grid = [_v("moe-aiter", args="--moe-runner-backend aiter")]
    kept, dropped = apply_aiter_moe_pin_filter(grid)
    assert [v.name for v in kept] == ["moe-aiter"]
    assert dropped == []
