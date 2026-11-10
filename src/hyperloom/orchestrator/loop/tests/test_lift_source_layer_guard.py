# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stack entry that left a source layer on a framework tree is reported when it carries no snapshot."""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


def _coord(tmp_path: Path) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=1000.0,
        current_best={"tput": 1000.0, "extra_server_args": "--tp 8"},
        model_path="/models/m",
    )
    return coord


def _winner(**source: object) -> dict:
    return {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8", **source}


def _guard_errors(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and "source snapshot" in r.getMessage()
    ]


def test_a_source_patch_lifted_without_a_snapshot_is_reported(tmp_path, caplog):
    coord = _coord(tmp_path)

    with caplog.at_level(logging.ERROR):
        assert coord.writeback.lift_to_current_best(
            "integrate_patch", 1200.0, _winner(scope="source_patch", source_layer_landed=True)
        )

    assert len(_guard_errors(caplog)) == 1
    assert "integrate_patch" in _guard_errors(caplog)[0]


def test_a_recipe_patch_landed_without_a_snapshot_is_reported(tmp_path, caplog):
    coord = _coord(tmp_path)
    landed = {"recipe_patch_statuses": [{"patch_ref": "p.patch", "status": "applied", "target_repo": "/sgl"}]}

    with caplog.at_level(logging.ERROR):
        assert coord.writeback.lift_to_current_best("replay_warm_recipe", 1200.0, _winner(), entry_extra=landed)

    assert len(_guard_errors(caplog)) == 1


def test_a_source_layer_with_its_snapshot_is_not_reported(tmp_path, caplog):
    coord = _coord(tmp_path)
    snapshot = tmp_path / "optimization_stack" / "src" / "src-patch"
    snapshot.mkdir(parents=True)

    with caplog.at_level(logging.ERROR):
        assert coord.writeback.lift_to_current_best(
            "integrate_patch",
            1200.0,
            _winner(
                scope="source_patch",
                source_layer_landed=True,
                framework_root=str(tmp_path),
                source_snapshot=str(snapshot),
            ),
        )

    assert _guard_errors(caplog) == []


def test_a_config_only_winner_is_not_reported(tmp_path, caplog):
    coord = _coord(tmp_path)

    with caplog.at_level(logging.ERROR):
        assert coord.writeback.lift_to_current_best("explore", 1200.0, _winner())

    assert _guard_errors(caplog) == []


def test_an_env_only_integrate_keep_is_not_reported(tmp_path, caplog):
    coord = _coord(tmp_path)

    with caplog.at_level(logging.ERROR):
        assert coord.writeback.lift_to_current_best(
            "integrate_patch", 1200.0, _winner(scope="source_patch", framework_root=str(tmp_path))
        )

    assert _guard_errors(caplog) == []
