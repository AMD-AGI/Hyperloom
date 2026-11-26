# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom_multibranch.scheduling import hosted_run
from hyperloom_multibranch.scheduling.hosted_run import (
    require_spec_matches,
    swept_session,
    sweep_stop_reason,
    take_over_swept_session,
)
from hyperloom_multibranch.scheduling.layout import SessionLayout


def _closed(root: Path) -> SessionLayout:
    layout = SessionLayout(root)
    layout.reports.mkdir(parents=True)
    layout.manifest.write_text("{}")
    layout.report.write_text("{}")
    return layout


def _spec(session_dir: Path | str, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(sweep_session_dir=str(session_dir), tp=2, gpu_type="MI300X", framework_name="vllm", **fields)


@pytest.fixture
def user_data(tmp_path: Path) -> Path:
    path = tmp_path / "user-data"
    path.mkdir()
    return path


def test_a_closed_session_below_the_runs_user_data_is_swept(user_data: Path):
    layout = _closed(user_data / "Model" / "MB-1")

    assert swept_session(_spec(layout.root), {"USER_DATA_PATH": str(user_data)}).root == layout.root


@pytest.mark.parametrize(
    ("named", "message"),
    [
        (lambda data, outside: outside, "is not below the run's USER_DATA_PATH"),
        (lambda data, outside: data / "link", "resolves to"),
        (lambda data, outside: data / "Model" / ".." / "Model" / "MB-1", "resolves to"),
        (lambda data, outside: data / "Model" / "empty", "holds no closed multi-branch session"),
    ],
)
def test_a_session_dir_outside_the_runs_data_through_a_link_or_not_closed_is_refused(
    user_data: Path, tmp_path: Path, named, message: str
):
    _closed(user_data / "Model" / "MB-1")
    (user_data / "Model" / "empty").mkdir()
    outside = _closed(tmp_path / "elsewhere" / "MB-2").root
    (user_data / "link").symlink_to(outside)

    with pytest.raises(hosted_run.HostedRefused, match=message):
        swept_session(_spec(named(user_data, outside)), {"USER_DATA_PATH": str(user_data)})


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"tp": 8}, "hardware.tp 8 is not the swept session's 2"),
        ({"gpu_type": "MI355X"}, "hardware.gpu_type mi355x is not the swept session's mi300x"),
        ({"framework_name": "sglang"}, "framework.name sglang is not the swept session's vllm"),
    ],
)
def test_a_spec_describing_another_workload_than_the_sessions_is_refused(spec: dict, message: str):
    workload = SimpleNamespace(tp=2, gpu_type="mi300x", framework="vllm")

    require_spec_matches(_spec("/s"), workload)  # type: ignore[arg-type]
    with pytest.raises(hosted_run.HostedRefused, match=message):
        require_spec_matches(SimpleNamespace(**{**vars(_spec("/s")), **spec}), workload)  # type: ignore[arg-type]


def test_a_restart_takes_over_the_lock_its_earlier_start_left_and_names_the_session(tmp_path: Path):
    layout = _closed(tmp_path / "MB-1")
    state = tmp_path / "state"
    state.mkdir()
    hosted_run.record_session(state, layout.root)
    layout.final_only_lock.write_text("4242\n")

    take_over_swept_session(state, layout)

    assert not layout.final_only_lock.exists()
    assert hosted_run.recorded_session(state) == layout.root


def test_a_lock_another_run_holds_is_left_to_refuse_the_sweep(tmp_path: Path):
    layout = _closed(tmp_path / "MB-1")
    state = tmp_path / "state"
    state.mkdir()
    layout.final_only_lock.write_text("4242\n")

    take_over_swept_session(state, layout)

    assert layout.final_only_lock.read_text() == "4242\n"


@pytest.mark.parametrize(
    ("summary", "reason"),
    [
        (None, None),
        ({"status": "succeeded"}, "conc_sweep_succeeded"),
        ({"status": "skipped", "skip_reason": "no_optimization"}, "conc_sweep_skipped:no_optimization"),
    ],
)
def test_the_stop_reason_is_the_sweeps_status(tmp_path: Path, summary: dict | None, reason: str | None):
    layout = _closed(tmp_path / "MB-1")
    if summary is not None:
        (layout.reports / "conc_sweep_summary.json").write_text(json.dumps(summary))

    assert sweep_stop_reason(layout) == reason
