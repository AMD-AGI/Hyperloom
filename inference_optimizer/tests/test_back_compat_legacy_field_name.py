# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Back-compat coverage for the ``extra_sglang_args`` -> ``extra_server_args`` rename."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from inference_optimizer.compat.payload_aliases import (
    CANONICAL_KEY,
    LEGACY_KEY,
    read_extra_server_args,
)
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
)
from inference_optimizer.orchestrator.shared_state import (
    SharedState,
    _migrate_legacy_extra_sglang_args_keys,
)


# Compat helper — one representative legacy-key read
@pytest.mark.expects_legacy_alias_warning
def test_compat_helper_reads_legacy_payload_with_deprecation_warning():
    """A legacy ``extra_sglang_args`` payload surfaces via the canonical reader with a DeprecationWarning."""
    payload = {LEGACY_KEY: "--legacy-flag-from-N-1"}
    with pytest.warns(DeprecationWarning) as caught:
        out = read_extra_server_args(payload)
    assert out == "--legacy-flag-from-N-1"
    assert any(LEGACY_KEY in str(w.message) for w in caught)
    assert any(CANONICAL_KEY in str(w.message) for w in caught)


# GridVariant — legacy kwarg back-compat alias
@pytest.mark.expects_legacy_alias_warning
def test_grid_variant_legacy_kwarg_routes_to_canonical_attribute():
    """``GridVariant(extra_sglang_args=...)`` still works; value lands on the canonical attribute."""
    with pytest.warns(DeprecationWarning):
        v = GridVariant(name="legacy-row", extra_sglang_args="--foo")
    assert v.extra_server_args == "--foo"
    assert v.name == "legacy-row"


@pytest.mark.expects_legacy_alias_warning
def test_grid_variant_canonical_wins_over_legacy_when_both_passed():
    """If both kwargs are passed the canonical wins; the DeprecationWarning still fires."""
    with pytest.warns(DeprecationWarning):
        v = GridVariant(
            name="dual",
            extra_server_args="--new",
            extra_sglang_args="--old",
        )
    assert v.extra_server_args == "--new"


def test_grid_variant_canonical_only_emits_no_warning():
    """Smoke: the canonical construction path must NOT emit the legacy DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = GridVariant(name="canonical", extra_server_args="--ok")
    assert v.extra_server_args == "--ok"
    assert not [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "extra_sglang_args" in str(w.message)
    ]


# SharedState — legacy state.json load
@pytest.mark.expects_legacy_alias_warning
def test_migrate_legacy_keys_walks_nested_structures():
    """The SharedState walker rewrites every legacy key occurrence in deeply-nested lists/dicts."""
    raw = {
        "baseline_artifacts": {LEGACY_KEY: "--a"},
        "explore_search": {
            "tested": [
                {"name": "v1", LEGACY_KEY: "--b"},
                {"name": "v2", CANONICAL_KEY: "--c"},
            ],
        },
        "winners": [
            {"name": "w1", "candidate_extra_sglang_args": "--d"},
        ],
        "unrelated_field": 42,
    }
    n = _migrate_legacy_extra_sglang_args_keys(raw)
    assert n == 3
    assert raw["baseline_artifacts"] == {CANONICAL_KEY: "--a"}
    assert raw["explore_search"]["tested"][0] == {"name": "v1", CANONICAL_KEY: "--b"}
    assert raw["explore_search"]["tested"][1] == {"name": "v2", CANONICAL_KEY: "--c"}
    assert raw["winners"][0] == {"name": "w1", "candidate_extra_server_args": "--d"}
    assert raw["unrelated_field"] == 42


@pytest.mark.expects_legacy_alias_warning
def test_shared_state_from_dict_silently_migrates_legacy_payload():
    """``--resume`` smoke path: loading a legacy-key state.json ends up canonical everywhere."""
    raw = {
        "schema_version": 2,
        "last_baseline": {LEGACY_KEY: "--legacy-baseline"},
    }
    state = SharedState.from_dict(raw)
    assert state.last_baseline.get(CANONICAL_KEY) == "--legacy-baseline"
    assert LEGACY_KEY not in state.last_baseline


@pytest.mark.expects_legacy_alias_warning
def test_shared_state_round_trip_writes_canonical_after_legacy_load(tmp_path: Path):
    """After loading and saving back a legacy state.json, only the canonical name remains."""
    state_dir = tmp_path
    state_file = state_dir / "state.json"
    state_file.write_text(json.dumps({
        "schema_version": 2,
        "last_baseline": {LEGACY_KEY: "--legacy"},
    }))
    state = SharedState.load_or_init(state_dir)
    assert state.last_baseline.get(CANONICAL_KEY) == "--legacy"
    assert LEGACY_KEY not in state.last_baseline


def test_migrate_helper_returns_zero_on_canonical_only_payload():
    """A canonical-only state.json triggers zero rewrites — the helper is idempotent."""
    raw = {
        "baseline_artifacts": {CANONICAL_KEY: "--ok"},
        "explore_search": {"tested": [{"name": "v", CANONICAL_KEY: "--x"}]},
    }
    n = _migrate_legacy_extra_sglang_args_keys(raw)
    assert n == 0
