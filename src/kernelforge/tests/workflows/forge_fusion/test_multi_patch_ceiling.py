# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One lane ceiling has to cover both multi-patch pipelines."""

from __future__ import annotations

from typing import Any

import pytest

from ._multi_patch_harness import nominate, spy_pipelines


@pytest.fixture
def spied(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return spy_pipelines(monkeypatch)


def test_the_ceiling_is_shared_between_claims_and_authored_recipes(tmp_path, spied):
    """Five claims spend the whole ceiling, so the autoloop is not run at all."""
    nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3", "c4", "c5"], max_recipes=3)

    assert spied["claims"] == ["c1", "c2", "c3"]
    assert spied["autoloop_ceiling"] is None


def test_claims_take_the_whole_ceiling_when_nothing_was_authored(tmp_path, spied):
    nominate(tmp_path, authored=[], claims=["c1", "c2", "c3", "c4"], max_recipes=3)

    assert spied["autoloop_ceiling"] is None, "the autoloop is not run without authored recipes"
    assert spied["claims"] == ["c1", "c2", "c3"]


def test_no_ceiling_withholds_neither_kind(tmp_path, spied):
    """Zero means none could be derived, never \"run nothing\"."""
    nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=0)

    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2
    assert spied["claims"] == ["c1", "c2", "c3"]


def test_a_ceiling_wider_than_both_kinds_withholds_nothing(tmp_path, spied):
    nominate(tmp_path, authored=["a1"], claims=["c1"], max_recipes=9)

    assert spied["autoloop_ceiling"] == 1
    assert spied["claims"] == ["c1"]
