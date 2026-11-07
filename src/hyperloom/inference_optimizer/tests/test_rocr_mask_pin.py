# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which ROCR_VISIBLE_DEVICES a materialized or rebuilt benchmark config launches on."""

from __future__ import annotations

import os

import pytest

from hyperloom.orchestrator.actions.executors._benchmark_runtime import apply_runtime_benchmark_overrides
from hyperloom.orchestrator.actions.executors._gpu_pin import resolve_gpu_pin

from ._workload_envs_helpers import clear_env, materialize, write_config

_HOST_MASK = "0,1,2,3,4,5,6,7"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "8")


def _materialized_mask(tmp_path, **envs) -> str:
    src = write_config(tmp_path / "point.yaml", envs=envs)
    return str(materialize(src, tmp_path / "out")["envs"]["ROCR_VISIBLE_DEVICES"])


def test_point_mask_survives_a_wider_host_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    assert _materialized_mask(tmp_path, TP=1, ROCR_VISIBLE_DEVICES="7") == "7"


def test_point_mask_kept_without_host_mask(tmp_path):
    assert _materialized_mask(tmp_path, TP=1, ROCR_VISIBLE_DEVICES="7") == "7"


def test_host_mask_fills_a_point_that_names_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    assert _materialized_mask(tmp_path, TP=2) == "4,5,6,7"


def test_point_mask_outside_the_host_mask_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match=r"ROCR_VISIBLE_DEVICES=.*7.*outside.*0,1,2,3"):
        _materialized_mask(tmp_path, TP=1, ROCR_VISIBLE_DEVICES="3,7")


def test_widening_a_narrow_point_mask_stays_on_host_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    assert _materialized_mask(tmp_path, TP=2, ROCR_VISIBLE_DEVICES="7") == "4,5"


def test_autofill_without_any_mask_is_not_a_geak_pin(tmp_path):
    """The recipe's ``0..tp-1`` autofill must read as "unpinned" to the GEAK handoff."""
    src = write_config(tmp_path / "point.yaml", envs={"TP": 2})
    envs = materialize(src, tmp_path / "out")["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert resolve_gpu_pin(recipe_envs=envs, environ={}) == {}


def test_runtime_rebuild_keeps_the_point_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    envs = apply_runtime_benchmark_overrides({"envs": {"TP": 1, "ROCR_VISIBLE_DEVICES": "7"}})
    assert envs["ROCR_VISIBLE_DEVICES"] == "7"


def test_runtime_rebuild_refuses_a_point_mask_outside_the_host_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="outside"):
        apply_runtime_benchmark_overrides({"envs": {"TP": 1, "ROCR_VISIBLE_DEVICES": "7"}})


@pytest.mark.parametrize(("first_mask", "leased"), [(None, "4,5"), ("4,5,6,7", "4,5")])
def test_a_filled_mask_is_refilled_on_another_processs_cards(tmp_path, monkeypatch, first_mask, leased):
    """A materialized config re-run under a leased mask (``rebench``) lands on the lease, not on the old fill."""
    if first_mask:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", first_mask)
    src = write_config(tmp_path / "point.yaml", envs={"TP": 2})
    first = materialize(src, tmp_path / "o1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", leased)
    again = write_config(tmp_path / "again.yaml", envs=first["envs"])
    assert materialize(again, tmp_path / "o2")["envs"]["ROCR_VISIBLE_DEVICES"] == leased
    assert apply_runtime_benchmark_overrides({"envs": dict(first["envs"])})["ROCR_VISIBLE_DEVICES"] == leased


def test_a_point_mask_stays_the_points_own_through_rematerialization(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    first = materialize(
        write_config(tmp_path / "point.yaml", envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "7"}), tmp_path / "o1"
    )
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    again = write_config(tmp_path / "again.yaml", envs=first["envs"])
    with pytest.raises(ValueError, match="outside"):
        materialize(again, tmp_path / "o2")


def test_geak_pin_is_the_points_own_mask_over_the_session_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    envs = materialize(
        write_config(tmp_path / "point.yaml", envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "7"}), tmp_path / "o"
    )["envs"]
    pin = resolve_gpu_pin(recipe_envs=envs, environ={"ROCR_VISIBLE_DEVICES": _HOST_MASK})
    assert (pin["value"], pin["count"], pin["source"]) == ("7", 1, "baseline_recipe")


def test_geak_pin_ignores_a_session_fill(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    envs = materialize(write_config(tmp_path / "point.yaml", envs={"TP": 2}), tmp_path / "o")["envs"]
    pin = resolve_gpu_pin(recipe_envs=envs, environ={"ROCR_VISIBLE_DEVICES": "4,5"})
    assert (pin["value"], pin["source"]) == ("4,5", "process_env")


def test_the_launch_env_reindexes_an_inherited_hip_mask(tmp_path, monkeypatch):
    """The child inherits the coordinator's HIP mask, which indexes into the SESSION's cards."""
    from hyperloom.common.env_safety import build_benchmark_env
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    preflight._normalize_hip_visible_devices()
    assert os.environ["HIP_VISIBLE_DEVICES"] == _HOST_MASK
    envs = materialize(
        write_config(tmp_path / "point.yaml", envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "7"}), tmp_path / "o"
    )["envs"]
    env = build_benchmark_env(envs)
    assert (env["ROCR_VISIBLE_DEVICES"], env["HIP_VISIBLE_DEVICES"]) == ("7", "0")


def test_a_points_own_hip_mask_stays_inside_its_own_rocr_view(tmp_path, monkeypatch):
    """``ROCR=4,5,6,7`` with ``HIP=2,3`` is cards 6 and 7; the point chose that subset itself."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", _HOST_MASK)
    point = {"TP": 2, "ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "2,3"}
    envs = materialize(write_config(tmp_path / "point.yaml", envs=point), tmp_path / "o")["envs"]
    assert (envs["ROCR_VISIBLE_DEVICES"], envs["HIP_VISIBLE_DEVICES"]) == ("4,5,6,7", "2,3")
    again = materialize(write_config(tmp_path / "again.yaml", envs=envs), tmp_path / "o2")["envs"]
    assert (again["ROCR_VISIBLE_DEVICES"], again["HIP_VISIBLE_DEVICES"]) == ("4,5,6,7", "2,3")


def test_a_filled_hip_mask_is_refilled_on_another_processs_cards(tmp_path, monkeypatch):
    """The fill indexes into the ROCR view the fill was made under; re-materializing moves that view."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", _HOST_MASK)
    first = materialize(write_config(tmp_path / "point.yaml", envs={"TP": 2}), tmp_path / "o1")["envs"]
    assert first["HIP_VISIBLE_DEVICES"] == _HOST_MASK
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    again = materialize(write_config(tmp_path / "again.yaml", envs=first), tmp_path / "o2")["envs"]
    assert (again["ROCR_VISIBLE_DEVICES"], again["HIP_VISIBLE_DEVICES"]) == ("4,5", "0,1")
    rebuilt = apply_runtime_benchmark_overrides({"envs": dict(first)})
    assert (rebuilt["ROCR_VISIBLE_DEVICES"], rebuilt["HIP_VISIBLE_DEVICES"]) == ("4,5", "0,1")


def test_a_stale_hip_fill_is_reindexed_when_the_process_exports_none(tmp_path, monkeypatch):
    """The stale fill rides in the config itself, so dropping it from the process env does not disarm it."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", _HOST_MASK)
    first = materialize(write_config(tmp_path / "point.yaml", envs={"TP": 2}), tmp_path / "o1")["envs"]
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    again = materialize(write_config(tmp_path / "again.yaml", envs=first), tmp_path / "o2")["envs"]
    assert (again["ROCR_VISIBLE_DEVICES"], again["HIP_VISIBLE_DEVICES"]) == ("4,5", "0,1")


def test_an_uninherited_hip_mask_is_not_written_into_the_config(tmp_path, monkeypatch):
    """Nothing to override: a variable the process does not export cannot reach the child stale."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _HOST_MASK)
    envs = materialize(
        write_config(tmp_path / "point.yaml", envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "7"}), tmp_path / "o"
    )["envs"]
    assert (
        not {
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HYPERLOOM_HIP_AUTOFILL",
        }
        & envs.keys()
    )


def test_a_config_materialized_before_the_fill_record_is_refused_with_a_way_out(tmp_path, monkeypatch):
    """Such a config carries the mask of the process that wrote it, and nothing marks it as a fill."""
    legacy = write_config(tmp_path / "legacy.yaml", envs={"TP": 1, "ROCR_VISIBLE_DEVICES": "4,5,6,7"})
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="re-materialize it from its source config"):
        materialize(legacy, tmp_path / "o")
