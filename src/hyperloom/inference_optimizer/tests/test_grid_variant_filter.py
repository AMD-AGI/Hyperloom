# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for apply_multi_node_invalid_variants."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_env as mn
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import (
    apply_multi_node_invalid_variants,
)


def _v(name: str, *, args: str = "") -> GridVariant:
    return GridVariant(name=name, extra_server_args=args)


@pytest.fixture()
def _multi_node(monkeypatch):
    monkeypatch.setattr(mn, "is_multi_node", lambda: True)


def test_single_node_is_a_strict_noop(monkeypatch):
    monkeypatch.setattr(mn, "is_multi_node", lambda: False)
    monkeypatch.setenv("CONC", "64")
    grid = [
        _v("low-graph", args="--cuda-graph-max-bs 8"),
        _v("keep", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    # Identity is the production contract: single-node returns the input
    # list (`return grid, []`), not a copy. Sibling filters copy on their
    # no-op path; this one does not, so `is` is the pin.
    assert kept is grid
    assert dropped == []


def test_multi_node_drops_cuda_graph_max_bs_below_conc(_multi_node, monkeypatch):
    monkeypatch.setenv("CONC", "64")
    grid = [
        _v("space-form", args="--cuda-graph-max-bs 32"),
        _v("equals-form", args="--cuda_graph_max_bs=8"),
        _v("at-threshold", args="--cuda-graph-max-bs 64"),
        _v("above", args="--cuda-graph-max-bs 128"),
        _v("no-flag", args="--chunked-prefill-size 8192"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-threshold", "above", "no-flag"]
    assert [row["name"] for row in dropped] == ["space-form", "equals-form"]
    assert all(row["source"] == "multi_node_invalid" for row in dropped)
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_conc_zero_does_not_drop(_multi_node, monkeypatch):
    # Documents observable behaviour: CONC=0 never drops anything.
    # The `conc > 0 and` guard was removed from production because the regex
    # (\d+) guarantees conc is always >= 0, making the guard unreachable; the
    # invariant is preserved by `n < conc` being False for all n >= 1 when conc=0.
    monkeypatch.setenv("CONC", "0")
    grid = [_v("low-graph", args="--cuda-graph-max-bs 1")]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["low-graph"]
    assert dropped == []


def test_multi_node_flag_in_non_leading_position_is_detected(_multi_node, monkeypatch):
    """The filter uses re.search(), not re.match() — flag anywhere in the string must fire."""
    monkeypatch.setenv("CONC", "64")
    grid = [
        _v("multi-flag-drop", args="--tp 8 --cuda-graph-max-bs 32"),
        _v("multi-flag-keep", args="--tp 8 --cuda-graph-max-bs 128"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["multi-flag-keep"]
    assert [row["name"] for row in dropped] == ["multi-flag-drop"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]
    assert "CONC=64" in dropped[0]["reason"]


def test_conc_unset_defaults_to_64(_multi_node, monkeypatch):
    """CONC env var absent → the os.environ.get default of '64' applies."""
    monkeypatch.delenv("CONC", raising=False)
    grid = [
        _v("below-default", args="--cuda-graph-max-bs 32"),
        _v("at-default", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-default"]
    assert [row["name"] for row in dropped] == ["below-default"]
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_conc_empty_string_defaults_to_64(_multi_node, monkeypatch):
    """CONC='' → the `or 64` branch applies (empty string is falsy)."""
    monkeypatch.setenv("CONC", "")
    grid = [
        _v("below-default", args="--cuda-graph-max-bs 32"),
        _v("at-default", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-default"]
    assert [row["name"] for row in dropped] == ["below-default"]
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_unparseable_conc_falls_back_to_64(_multi_node, monkeypatch):
    monkeypatch.setenv("CONC", "not-an-int")
    grid = [
        _v("below-default", args="--cuda-graph-max-bs 32"),
        _v("at-default", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-default"]
    assert [row["name"] for row in dropped] == ["below-default"]
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_none_extra_server_args_is_treated_as_empty(_multi_node, monkeypatch):
    """`v.extra_server_args or ""` must not raise or match when args is None."""
    monkeypatch.setenv("CONC", "64")
    v = GridVariant(name="none-args", extra_server_args=None)
    kept, dropped = apply_multi_node_invalid_variants([v])
    assert kept == [v]
    assert dropped == []
