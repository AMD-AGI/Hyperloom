"""Tests for :mod:`runtime.category_mapping`."""

from __future__ import annotations

import pytest

from runtime.category_mapping import (
    CATEGORY_TO_KIND,
    KB_KINDS,
    filter_supported_categories,
    map_category_to_kind,
)
from runtime.errors import RuntimeAdapterError


def test_kb_kinds_set_matches_contract():
    assert KB_KINDS == frozenset({"pitfall", "technique", "params_catalog", "model_profile"})


def test_kernel_opt_maps_to_technique():
    assert map_category_to_kind("kernel_optimization") == "technique"


def test_pitfall_categories_collapse_correctly():
    assert map_category_to_kind("crash_recovery") == "pitfall"
    assert map_category_to_kind("benchmark_methodology") == "pitfall"
    assert map_category_to_kind("architecture_constraint") == "pitfall"


def test_unknown_category_raises():
    with pytest.raises(RuntimeAdapterError):
        map_category_to_kind("not_a_thing")


def test_filter_supported_partitions_lists():
    supported, rejected = filter_supported_categories(
        ["kernel_optimization", "definitely_not_a_thing", "server_params"]
    )
    assert supported == ["kernel_optimization", "server_params"]
    assert rejected == ["definitely_not_a_thing"]


def test_full_table_targets_known_kinds():
    for category, kind in CATEGORY_TO_KIND.items():
        assert kind in KB_KINDS, (category, kind)
