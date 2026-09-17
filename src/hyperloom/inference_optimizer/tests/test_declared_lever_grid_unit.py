# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for adapter-declared explore levers.

A scriptable / bypass workload has no framework source for the model to infer
levers from, so ``_default_grid_for_framework`` returns ``[]`` and EXPLORE has
nothing to sweep. These cover the declaration the adapter supplies instead.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.actions.executors.explore import _declared_lever_grid

_LEVERS = [
    {"name": "workers_1", "extra_envs": {"MINIMAP2_GPU_CHAIN_WORKERS": "1"}},
    {"name": "workers_2", "extra_envs": {"MINIMAP2_GPU_CHAIN_WORKERS": "2"}},
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HYPERLOOM_DECLARED_LEVERS", "HLHPC_LEVERS", "HYPERLOOM_BYPASS_SCRIPTS_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_nothing_declared_returns_empty() -> None:
    assert _declared_lever_grid() == []


def test_inline_env_json_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DECLARED_LEVERS", json.dumps(_LEVERS))
    grid = _declared_lever_grid()
    assert [v["name"] for v in grid] == ["workers_1", "workers_2"]
    assert all(v["provenance"] == "declared_lever" for v in grid)


def test_legacy_env_alias_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HLHPC_LEVERS", json.dumps(_LEVERS))
    assert len(_declared_lever_grid()) == 2


def test_inline_env_wins_over_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / "declared_levers.json").write_text(json.dumps([{"name": "from_file"}]), encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    monkeypatch.setenv("HYPERLOOM_DECLARED_LEVERS", json.dumps([{"name": "from_env"}]))
    assert [v["name"] for v in _declared_lever_grid()] == ["from_env"]


def test_file_in_scripts_dir_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / "declared_levers.json").write_text(json.dumps(_LEVERS), encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    assert len(_declared_lever_grid()) == 2


def test_legacy_filename_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / "hlhpc_levers.json").write_text(json.dumps(_LEVERS), encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    assert len(_declared_lever_grid()) == 2


def test_object_form_with_levers_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DECLARED_LEVERS", json.dumps({"levers": _LEVERS}))
    assert len(_declared_lever_grid()) == 2


def test_malformed_json_seeds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DECLARED_LEVERS", "{not json")
    assert _declared_lever_grid() == []


def test_non_list_seeds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DECLARED_LEVERS", json.dumps(42))
    assert _declared_lever_grid() == []


def test_unnamed_variants_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "HYPERLOOM_DECLARED_LEVERS",
        json.dumps([{"name": "keep"}, {"extra_envs": {"X": "1"}}, {"name": "   "}]),
    )
    assert [v["name"] for v in _declared_lever_grid()] == ["keep"]


def test_missing_scripts_dir_file_seeds_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    assert _declared_lever_grid() == []


def test_explicit_provenance_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "HYPERLOOM_DECLARED_LEVERS",
        json.dumps([{"name": "custom", "provenance": "operator", "note": "hand-picked"}]),
    )
    grid = _declared_lever_grid()
    assert grid[0]["provenance"] == "operator"
    assert grid[0]["note"] == "hand-picked"
