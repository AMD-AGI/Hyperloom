# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared lane budget has to fund compile-pass claims before authoring."""

from __future__ import annotations

from typing import Any

import pytest

from ._multi_patch_harness import nominate, spy_pipelines

#: Every real Hyperloom session caps the fusion lane at three targets
#: (``FUSION_MAX_TARGETS``), so this is the mainline ceiling, not an edge case.
LANE_CEILING = 3


@pytest.fixture
def spied(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return spy_pipelines(monkeypatch)


def test_a_full_authoring_slate_does_not_withhold_the_claim(tmp_path, spied):
    """Three authored recipes must not consume the ceiling ahead of a claim."""
    nominate(tmp_path, authored=["a1", "a2", "a3"], claims=["c1"], max_recipes=LANE_CEILING)

    assert spied["claims"] == ["c1"]


def test_a_funded_claim_reports_its_verdict_on_the_compile_pass_slot(tmp_path, spied):
    """A null ``compile_pass`` reads as "no claim ran", so the claim must fill it."""
    outcome = nominate(tmp_path, authored=["a1", "a2", "a3"], claims=["c1"], max_recipes=LANE_CEILING)[1]

    assert outcome is not None
    assert outcome.flag == "FLAG_C1"


def test_authored_recipes_take_what_the_claims_leave(tmp_path, spied):
    """The remainder still funds the autoloop, ranked order preserved."""
    nominate(tmp_path, authored=["a1", "a2", "a3"], claims=["c1"], max_recipes=LANE_CEILING)

    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2


def test_an_exhausted_ceiling_runs_no_authored_recipe(tmp_path, spied):
    """Zero remaining means the loop is skipped, never re-read as uncapped."""
    nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=LANE_CEILING)

    assert spied["claims"] == ["c1", "c2", "c3"]
    assert spied["autoloop_recipes"] == []
    assert spied["autoloop_ceiling"] is None


def test_the_round_reports_how_many_targets_it_withheld(tmp_path, spied):
    """Starvation is otherwise indistinguishable from "everything ran and failed"."""
    _patches, _outcome, _loop, withheld = nominate(
        tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3", "c4"], max_recipes=LANE_CEILING
    )

    assert withheld == 3


def test_an_uncapped_round_withholds_nothing(tmp_path, spied):
    """A non-positive ceiling means none was derived, never "run nothing"."""
    _patches, _outcome, _loop, withheld = nominate(
        tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=0
    )

    assert spied["claims"] == ["c1", "c2", "c3"]
    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2
    assert withheld == 0
