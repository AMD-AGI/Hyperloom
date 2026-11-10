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
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.bypass_scriptable import (
    DECLARED_LEVERS_FILENAME,
    DeclaredLeversError,
    declared_lever_payload,
    load_declared_levers,
)
from hyperloom.orchestrator.actions.executors.explore import _lever_seeded_payload

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
    """The executor's grid merge: real ``declared_lever_payload``, then ``_lever_seeded_payload``.

    Drives the shipped functions rather than restating them, so a change in either
    is caught here instead of passing against a copy of the old behaviour.
    """
    declared = declared_lever_payload(shared_state, framework)
    return _lever_seeded_payload(list(grid_payload), declared)


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
    shared_state = SimpleNamespace(explore_search={"tested": {"fp-1": {"name": "workers_1", "outcome": "REVERT"}}})
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


# --- the wiring, not a restatement of it ----------------------------------
#
# The tests above drive load_declared_levers / declared_lever_payload directly.
# These drive the two places the feature is actually wired in: the executor's
# grid resolution, and the launch-time exit code. Delete either wiring and
# these fail; that is the point of them.


def _resolve_grid_on(
    framework: str, grid: list[dict], shared_state=None, *, params_extra: dict | None = None
) -> list[str]:
    """Run the real ``_ExploreRound._resolve_grid`` and return the variant names it built.

    ``__init__`` resolves a model path and a benchmark config, neither of which this
    contract depends on, so the round is built bare and given only the attributes
    ``_resolve_grid`` reads.
    """
    from hyperloom.orchestrator.actions.executors.explore import _ExploreRound

    rnd = _ExploreRound.__new__(_ExploreRound)
    rnd.params = {"grid": list(grid), **(params_extra or {})}
    rnd.ss = shared_state
    rnd.framework = framework
    rnd.yaml_envs = {}
    rnd.output_root = Path(".")
    rnd.base_remove_args = []
    rnd.base_unset_envs = []
    rnd.args_mode = "append"
    rnd._resolve_grid()
    return [gv.name for gv in rnd.grid]


def test_resolve_grid_seeds_declarations_for_custom(scripts_dir):
    _declare(scripts_dir, _LEVERS)
    names = _resolve_grid_on("custom", [])
    assert names == ["workers_1", "workers_2"]


def test_resolve_grid_leaves_other_frameworks_alone(scripts_dir):
    _declare(scripts_dir, _LEVERS)
    model_grid = [{"name": "model_pick", "extra_envs": {"A": "1"}}]
    assert _resolve_grid_on("vllm", model_grid) == ["model_pick"]


def test_resolve_grid_skips_names_the_real_ledger_has_benched(scripts_dir):
    """The ledger bucket ``apply_explore_search_update`` actually writes is ``tested``."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    _declare(scripts_dir, _LEVERS)
    ss = SharedState()
    ss.apply_explore_search_update({"tested": {"fp-workers-1": {"name": "workers_1", "outcome": "REVERT"}}})
    assert _resolve_grid_on("custom", [], ss) == ["workers_2"]


@pytest.mark.parametrize(
    "bad_envs",
    [
        "OMP_NUM_THREADS=4",  # a string normalizes to {}
        ["OMP_NUM_THREADS=4"],  # the --extra-env-style list does too
    ],
    ids=["string", "list"],
)
def test_launch_rejects_extra_envs_that_normalize_away(scripts_dir, bad_envs, capsys):
    """These pass a raw truthiness check but carry no envs once normalized."""
    from hyperloom.inference_optimizer import cli

    _declare(scripts_dir, [{"name": "threads_4", "extra_envs": bad_envs}])
    with pytest.raises(SystemExit) as excinfo:
        cli._require_custom_entrypoint("custom", gpu_type="mi355x")
    assert excinfo.value.code == 2
    # _require_custom_entrypoint also exits 2 when it cannot resolve the entrypoint,
    # so the code alone does not prove this came from the lever check.
    assert DECLARED_LEVERS_FILENAME in capsys.readouterr().err


def test_launch_exits_2_on_a_malformed_declaration(scripts_dir, capsys):
    from hyperloom.inference_optimizer import cli

    (scripts_dir / DECLARED_LEVERS_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli._require_custom_entrypoint("custom", gpu_type="mi355x")
    assert excinfo.value.code == 2
    assert "is not valid JSON" in capsys.readouterr().err


def test_resolve_grid_keeps_declarations_out_of_a_stack_revalidation(scripts_dir):
    """A revalidation re-measures the recorded stack; a seeded lever would validate stack+lever."""
    from hyperloom.orchestrator.actions.executors.explore import STACK_REVALIDATE_SOURCE

    _declare(scripts_dir, _LEVERS)
    model_grid = [{"name": "restated_stack", "extra_envs": {"A": "1"}}]
    names = _resolve_grid_on("custom", model_grid, params_extra={"source": STACK_REVALIDATE_SOURCE})
    assert names == ["restated_stack"]
