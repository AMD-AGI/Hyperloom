# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Landing N sibling patches from one nomination without cross-contamination."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.kernel import patch_landing as pl


# --- record_source_path: one spelling read on both sides ---------------------


def test_source_path_prefers_target_file() -> None:
    """target_file wins so the write side and read side cannot disagree."""
    assert pl.record_source_path({"target_file": "a.py", "source_file": "b.py"}) == "a.py"


def test_source_path_falls_back_to_source_file() -> None:
    assert pl.record_source_path({"source_file": "b.py"}) == "b.py"


def test_source_path_is_empty_when_neither_is_set() -> None:
    assert pl.record_source_path({}) == ""


def test_source_path_strips_whitespace() -> None:
    assert pl.record_source_path({"target_file": "  a.py \n"}) == "a.py"


@pytest.mark.parametrize("value", [None, "not-a-mapping", 42, ["a.py"]])
def test_source_path_of_a_non_mapping_is_empty(value: Any) -> None:
    assert pl.record_source_path(value) == ""


def test_source_path_treats_blank_target_as_unset() -> None:
    """An empty target_file must not shadow a real source_file."""
    assert pl.record_source_path({"target_file": "", "source_file": "b.py"}) == "b.py"


# --- patch_budget: a positive ceiling, always ---------------------------------


def test_budget_defaults_when_nothing_is_configured() -> None:
    assert pl.patch_budget() == pl.DEFAULT_PATCH_BUDGET


def test_budget_honours_a_configured_override() -> None:
    assert pl.patch_budget(5) == 5


def test_budget_accepts_a_numeric_string() -> None:
    assert pl.patch_budget("7") == 7


@pytest.mark.parametrize("value", [0, -1, "0", "-4"])
def test_a_non_positive_override_falls_back_to_default(value: Any) -> None:
    assert pl.patch_budget(value) == pl.DEFAULT_PATCH_BUDGET


@pytest.mark.parametrize("value", [None, "abc", object(), True, False])
def test_an_unparsable_override_falls_back_to_default(value: Any) -> None:
    assert pl.patch_budget(value) == pl.DEFAULT_PATCH_BUDGET


def test_a_broken_default_still_yields_the_module_floor() -> None:
    """Even a nonsense default cannot produce a non-positive ceiling."""
    assert pl.patch_budget(None, default=0) == pl.DEFAULT_PATCH_BUDGET


# --- clamp_by_budget: fit vs deferred, nothing dropped ------------------------


def _rows(n: int) -> list[dict[str, Any]]:
    return [{"integration_id": str(i), "micro_speedup": float(i)} for i in range(n)]


def test_clamp_splits_at_the_budget() -> None:
    fit, deferred = pl.clamp_by_budget(_rows(5), 3)
    assert [r["integration_id"] for r in fit] == ["0", "1", "2"]
    assert [r["integration_id"] for r in deferred] == ["3", "4"]


def test_clamp_defers_rather_than_drops() -> None:
    """Every input reappears in exactly one of the two lists."""
    fit, deferred = pl.clamp_by_budget(_rows(5), 2)
    assert len(fit) + len(deferred) == 5


def test_clamp_under_budget_defers_nothing() -> None:
    fit, deferred = pl.clamp_by_budget(_rows(2), 3)
    assert len(fit) == 2 and deferred == []


def test_clamp_at_zero_budget_defers_everything() -> None:
    fit, deferred = pl.clamp_by_budget(_rows(3), 0)
    assert fit == [] and len(deferred) == 3


def test_clamp_treats_a_negative_budget_as_zero() -> None:
    fit, _deferred = pl.clamp_by_budget(_rows(3), -2)
    assert fit == []


def test_clamp_copies_rows_so_callers_cannot_mutate_state() -> None:
    original = {"integration_id": "0", "micro_speedup": 1.0}
    fit, _ = pl.clamp_by_budget([original], 1)
    fit[0]["micro_speedup"] = 999.0
    assert original["micro_speedup"] == 1.0


def test_clamp_skips_non_mapping_rows() -> None:
    fit, deferred = pl.clamp_by_budget([{"integration_id": "0"}, "junk", None], 5)
    assert len(fit) == 1 and deferred == []


# --- evict_terminal: the queue's only deletion point --------------------------


def _pending(i: str, micro: float = 0.0) -> dict[str, Any]:
    return {"integration_id": i, "status": "pending", "micro_speedup": micro}


def _terminal(i: str, status: str, micro: float = 0.0) -> dict[str, Any]:
    return {"integration_id": i, "status": status, "micro_speedup": micro}


def test_pending_records_are_always_kept() -> None:
    queue = {"a": _pending("a"), "b": _pending("b")}
    assert set(pl.evict_terminal(queue, budget=1, retention_multiple=0)) == {"a", "b"}


def test_terminal_records_within_the_cap_are_kept() -> None:
    queue = {"a": _terminal("a", "integrated"), "b": _terminal("b", "rejected")}
    kept = pl.evict_terminal(queue, budget=1, retention_multiple=2)
    assert set(kept) == {"a", "b"}


def test_the_weakest_terminal_records_are_dropped_first() -> None:
    queue = {
        "keep": _terminal("keep", "integrated", micro=9.0),
        "drop": _terminal("drop", "rejected", micro=0.1),
    }
    kept = pl.evict_terminal(queue, budget=1, retention_multiple=1)
    assert set(kept) == {"keep"}


def test_a_zero_retention_reaps_all_terminal_records() -> None:
    queue = {"a": _terminal("a", "dispatch_failed"), "b": _pending("b")}
    kept = pl.evict_terminal(queue, budget=3, retention_multiple=0)
    assert set(kept) == {"b"}


@pytest.mark.parametrize("status", sorted(pl.TERMINAL_STATUSES))
def test_every_terminal_status_is_eligible_for_eviction(status: str) -> None:
    queue = {"x": _terminal("x", status)}
    assert pl.evict_terminal(queue, budget=0, retention_multiple=0) == {}


def test_an_unknown_status_is_treated_as_live() -> None:
    queue = {"x": {"integration_id": "x", "status": "benching"}}
    assert set(pl.evict_terminal(queue, budget=0, retention_multiple=0)) == {"x"}


def test_a_missing_status_defaults_to_pending() -> None:
    queue = {"x": {"integration_id": "x"}}
    assert set(pl.evict_terminal(queue, budget=0, retention_multiple=0)) == {"x"}


def test_non_dict_entries_are_preserved_verbatim() -> None:
    queue = {"x": "foreign", "y": _terminal("y", "integrated")}
    kept = pl.evict_terminal(queue, budget=0, retention_multiple=0)
    assert kept["x"] == "foreign" and "y" not in kept


def test_evicting_a_non_mapping_queue_is_empty() -> None:
    assert pl.evict_terminal(None) == {}


def test_a_non_numeric_micro_sorts_weakest() -> None:
    """A record whose micro cannot be read must not crash the sort."""
    queue = {
        "num": _terminal("num", "integrated", micro=5.0),
        "bad": {"integration_id": "bad", "status": "integrated", "micro_speedup": "oops"},
    }
    kept = pl.evict_terminal(queue, budget=1, retention_multiple=1)
    assert set(kept) == {"num"}


# --- bundle_belongs_to: refuse a cross-sibling bundle -------------------------


def _bundle(integration_id: str | None = None) -> dict[str, Any]:
    b: dict[str, Any] = {"type": "patch_snapshot", "write_paths": ["a.py"]}
    if integration_id is not None:
        b["integration_id"] = integration_id
    return b


def test_a_matching_bundle_is_accepted() -> None:
    assert pl.bundle_belongs_to(_bundle("id-1"), "id-1") is True


def test_a_cross_sibling_bundle_is_refused() -> None:
    assert pl.bundle_belongs_to(_bundle("id-2"), "id-1") is False


def test_a_legacy_bundle_without_an_id_is_accepted() -> None:
    """No id to disagree with, so the one-patch-era fallback still holds."""
    assert pl.bundle_belongs_to(_bundle(None), "id-1") is True


def test_an_id_less_integrate_accepts_any_bundle() -> None:
    assert pl.bundle_belongs_to(_bundle("id-2"), "") is True


def test_a_non_mapping_bundle_is_refused() -> None:
    assert pl.bundle_belongs_to(None, "id-1") is False


def test_bundle_ids_are_compared_after_stripping() -> None:
    assert pl.bundle_belongs_to(_bundle(" id-1 "), "id-1") is True
