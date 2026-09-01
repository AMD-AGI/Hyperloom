# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for _optimization_budget_minutes prioritisation and floor enforcement."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.kernel.request_handlers import (
    _optimization_budget_minutes,
    _rewrite_route_budget_floor_minutes,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_BACKEND_BUDGET_MIN", raising=False)
    monkeypatch.delenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", raising=False)


def test_payload_budget_is_used_when_no_env(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", "0")
    result = _optimization_budget_minutes({"budget_minutes": 120})
    assert result == 120.0


def test_env_budget_beats_payload_when_above_floor(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", "0")
    monkeypatch.setenv("KERNEL_OPT_BACKEND_BUDGET_MIN", "180")
    result = _optimization_budget_minutes({"budget_minutes": 120})
    assert result == 180.0


def test_env_budget_below_floor_raises_to_floor_when_rewrite_enabled(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", "1")
    monkeypatch.setenv("KERNEL_OPT_BACKEND_BUDGET_MIN", "60")
    floor = _rewrite_route_budget_floor_minutes()
    assert floor > 0.0
    result = _optimization_budget_minutes({"budget_minutes": 120})
    assert result == floor


def test_env_budget_above_floor_wins_when_rewrite_enabled(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_FORGE_REWRITE_BY_FLYDSL", "1")
    floor = _rewrite_route_budget_floor_minutes()
    assert floor > 0.0
    high = floor + 100
    monkeypatch.setenv("KERNEL_OPT_BACKEND_BUDGET_MIN", str(high))
    result = _optimization_budget_minutes({})
    assert result == high
