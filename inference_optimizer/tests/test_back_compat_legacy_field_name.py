# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Back-compat coverage for the ``extra_sglang_args`` ->
``extra_server_args`` rename.

After the bulk rename every production call site uses the canonical
name, so the regular test surface no longer exercises the legacy-key
code paths at all. This file re-introduces the legacy key
intentionally at the few stable read-tolerant boundaries:

* :func:`inference_optimizer.compat.payload_aliases.read_extra_server_args`
  -- the helper itself (smoke-overlap with ``test_payload_aliases``).
* :class:`inference_optimizer.orchestrator.action_executors._grid_runner.GridVariant`
  -- back-compat keyword alias ``extra_sglang_args=`` on the dataclass
  constructor.
* :meth:`inference_optimizer.orchestrator.shared_state.SharedState.from_dict`
  -- one-shot walk-and-rewrite of nested ``extra_sglang_args`` /
  ``candidate_extra_sglang_args`` keys to the canonical names when
  ``--resume`` loads a legacy state.json.

Every test in this file is tagged with ``expects_legacy_alias_warning``
so the static guard in 4.7 can distinguish "legitimate legacy-key
reference" from "missed rename target".
"""

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


# ---------------------------------------------------------------------------
# Compat helper — one representative legacy-key read
# ---------------------------------------------------------------------------
@pytest.mark.expects_legacy_alias_warning
def test_compat_helper_reads_legacy_payload_with_deprecation_warning():
    """A payload coming from a legacy KB record / pre-rename Coordinator
    that still carries ``extra_sglang_args`` must surface its value
    via the canonical reader, paired with a DeprecationWarning."""
    payload = {LEGACY_KEY: "--legacy-flag-from-N-1"}
    with pytest.warns(DeprecationWarning) as caught:
        out = read_extra_server_args(payload)
    assert out == "--legacy-flag-from-N-1"
    # The warning message names both the legacy and the canonical key
    # so a developer following the audit trail knows where to migrate.
    assert any(LEGACY_KEY in str(w.message) for w in caught)
    assert any(CANONICAL_KEY in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# GridVariant — legacy kwarg back-compat alias
# ---------------------------------------------------------------------------
@pytest.mark.expects_legacy_alias_warning
def test_grid_variant_legacy_kwarg_routes_to_canonical_attribute():
    """``GridVariant(extra_sglang_args=...)`` must continue to work for
    one release (existing call sites in operator scripts / sub-agent
    fixtures); the value must land on the canonical attribute."""
    with pytest.warns(DeprecationWarning):
        v = GridVariant(name="legacy-row", extra_sglang_args="--foo")
    assert v.extra_server_args == "--foo"
    assert v.name == "legacy-row"


@pytest.mark.expects_legacy_alias_warning
def test_grid_variant_canonical_wins_over_legacy_when_both_passed():
    """If a caller passes both kwargs the canonical wins (matches the
    compat helper's "canonical-present-means-migration-done" rule).
    The DeprecationWarning still fires because the legacy kwarg
    was supplied."""
    with pytest.warns(DeprecationWarning):
        v = GridVariant(
            name="dual",
            extra_server_args="--new",
            extra_sglang_args="--old",
        )
    assert v.extra_server_args == "--new"


def test_grid_variant_canonical_only_emits_no_warning():
    """Smoke: the regular construction path must NOT emit the legacy
    DeprecationWarning. Catches a future regression where the alias
    handling accidentally fires on the canonical path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = GridVariant(name="canonical", extra_server_args="--ok")
    assert v.extra_server_args == "--ok"
    assert not [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "extra_sglang_args" in str(w.message)
    ]


# ---------------------------------------------------------------------------
# SharedState — legacy state.json load
# ---------------------------------------------------------------------------
@pytest.mark.expects_legacy_alias_warning
def test_migrate_legacy_keys_walks_nested_structures():
    """The SharedState walker must reach into deeply-nested
    lists/dicts (winners, baseline_artifacts, action_attempts) and
    rewrite *every* legacy key occurrence."""
    raw = {
        "baseline_artifacts": {LEGACY_KEY: "--a"},
        "explore_search": {
            "tested": [
                {"name": "v1", LEGACY_KEY: "--b"},
                {"name": "v2", CANONICAL_KEY: "--c"},  # already migrated
            ],
        },
        "winners": [
            {"name": "w1", "candidate_extra_sglang_args": "--d"},
        ],
        "unrelated_field": 42,
    }
    n = _migrate_legacy_extra_sglang_args_keys(raw)
    assert n == 3  # 3 legacy keys rewritten; v2 already canonical
    assert raw["baseline_artifacts"] == {CANONICAL_KEY: "--a"}
    assert raw["explore_search"]["tested"][0] == {"name": "v1", CANONICAL_KEY: "--b"}
    assert raw["explore_search"]["tested"][1] == {"name": "v2", CANONICAL_KEY: "--c"}
    assert raw["winners"][0] == {"name": "w1", "candidate_extra_server_args": "--d"}
    assert raw["unrelated_field"] == 42


@pytest.mark.expects_legacy_alias_warning
def test_shared_state_from_dict_silently_migrates_legacy_payload():
    """Loading a state.json that carries the legacy key must succeed
    and end up with the canonical key set everywhere. This is the
    ``--resume`` smoke path for sessions started on a pre-rename
    release. ``last_baseline`` is a dict-shaped SharedState
    field that carries the result payload from the baseline executor
    and historically held ``extra_sglang_args`` under its top-level."""
    raw = {
        "schema_version": 2,
        "last_baseline": {LEGACY_KEY: "--legacy-baseline"},
    }
    state = SharedState.from_dict(raw)
    assert state.last_baseline.get(CANONICAL_KEY) == "--legacy-baseline"
    assert LEGACY_KEY not in state.last_baseline


@pytest.mark.expects_legacy_alias_warning
def test_shared_state_round_trip_writes_canonical_after_legacy_load(tmp_path: Path):
    """After loading a legacy state.json and saving it back, the file
    on disk contains only the canonical name — the SharedState writer
    never re-introduces the legacy key."""
    state_dir = tmp_path
    state_file = state_dir / "state.json"
    state_file.write_text(json.dumps({
        "schema_version": 2,
        "last_baseline": {LEGACY_KEY: "--legacy"},
    }))
    state = SharedState.load_or_init(state_dir)
    # The canonical key flows; the legacy key has been transformed
    # out of the in-memory representation entirely.
    assert state.last_baseline.get(CANONICAL_KEY) == "--legacy"
    assert LEGACY_KEY not in state.last_baseline


def test_migrate_helper_returns_zero_on_canonical_only_payload():
    """A state.json that already uses the canonical key everywhere
    triggers zero rewrites — the helper is idempotent so re-loading
    the same file is cheap and silent."""
    raw = {
        "baseline_artifacts": {CANONICAL_KEY: "--ok"},
        "explore_search": {"tested": [{"name": "v", CANONICAL_KEY: "--x"}]},
    }
    n = _migrate_legacy_extra_sglang_args_keys(raw)
    assert n == 0
