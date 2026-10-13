# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behaviour of the ``custom`` adapter's declared lever grid.

A ``custom`` workload has no framework source for the model to read, so
``_default_grid_for_framework`` returns ``[]`` and EXPLORE has nothing to sweep.
The adapter declares its knob space in ``<scripts-dir>/levers.json`` instead.

These cover what the declaration *does* to the grid, not just how it parses.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.explore import (
    DECLARED_LEVERS_FILENAME,
    DeclaredLeversError,
    load_declared_levers,
)

_LEVERS = [
    {"name": "workers_1", "extra_envs": {"GPU_CHAIN_WORKERS": "1"}},
    {"name": "workers_2", "extra_envs": {"GPU_CHAIN_WORKERS": "2"}},
]


@pytest.fixture
def scripts_dir(tmp_path, monkeypatch):
    """A bypass scripts dir, exported the way the launcher exports it."""
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    return tmp_path


def _declare(scripts_dir, payload) -> None:
    (scripts_dir / DECLARED_LEVERS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _merge(framework: str, grid_payload: list[dict], shared_state=None) -> list[dict]:
    """The executor's grid merge for ``framework``, as ``ExploreExecutor`` runs it."""
    if framework != "custom":
        return list(grid_payload)
    search = getattr(shared_state, "explore_search", None)
    tested = set((search or {}).get("name_index") or {}) if isinstance(search, dict) else set()
    declared = [v for v in load_declared_levers() if str(v.get("name") or "") not in tested]
    existing = {str(v.get("name") or "") for v in grid_payload if isinstance(v, dict)}
    fresh = [v for v in declared if str(v.get("name") or "") not in existing]
    return fresh + list(grid_payload)


# --- behaviour -------------------------------------------------------------


def test_declarations_sweep_when_the_model_proposes_nothing(scripts_dir):
    """The whole point: an empty model grid still sweeps the adapter's knobs."""
    _declare(scripts_dir, _LEVERS)
    grid = _merge("custom", [])
    assert [v["name"] for v in grid] == ["workers_1", "workers_2"]
    assert all(v["provenance"] == "default_grid" for v in grid)


@pytest.mark.parametrize("framework", ["vllm", "xdit", "sglang", "atom"])
def test_other_frameworks_are_untouched(scripts_dir, framework):
    """Only ``custom`` reads the declaration -- ``xdit`` is scriptable but excluded."""
    _declare(scripts_dir, _LEVERS)
    assert _merge(framework, [{"name": "model_variant"}]) == [{"name": "model_variant"}]


def test_already_tested_declarations_are_not_reseeded(scripts_dir):
    """A benched declaration must not re-spend the next round's budget."""
    _declare(scripts_dir, _LEVERS)
    shared_state = SimpleNamespace(explore_search={"name_index": {"workers_1": "fp-1"}})
    grid = _merge("custom", [], shared_state=shared_state)
    assert [v["name"] for v in grid] == ["workers_2"]


def test_model_proposal_wins_a_name_collision(scripts_dir):
    """A declaration never displaces the model's variant of the same name."""
    _declare(scripts_dir, _LEVERS)
    proposed = {"name": "workers_1", "extra_envs": {"GPU_CHAIN_WORKERS": "8"}}
    grid = _merge("custom", [proposed])
    assert [v["name"] for v in grid] == ["workers_2", "workers_1"]
    assert grid[-1]["extra_envs"]["GPU_CHAIN_WORKERS"] == "8"


# --- the declaration is loud when it is wrong ------------------------------


def test_no_declaration_is_not_an_error(scripts_dir):
    assert load_declared_levers() == []


def test_unset_scripts_dir_is_not_an_error(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", raising=False)
    assert load_declared_levers() == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('{"levers": []}', "must be a JSON list"),
        ("[42]", "must be a variant object"),
        ('[{"extra_envs": {"A": "1"}}]', "has no 'name'"),
        ('[{"name": "noop"}]', "declares nothing a restart could apply"),
        ("{not json", "not valid JSON"),
    ],
)
def test_malformed_declaration_raises(scripts_dir, payload, expected):
    """Every malformed shape is loud -- silence surfaces later as ``empty_grid``."""
    (scripts_dir / DECLARED_LEVERS_FILENAME).write_text(payload, encoding="utf-8")
    with pytest.raises(DeclaredLeversError, match=expected):
        load_declared_levers()
