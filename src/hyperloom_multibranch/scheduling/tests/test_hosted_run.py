# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted run reads from the host and writes back: the grant's bound on the session's time, the stop the
grant's end sends, and the files that tell the host whether to start the run again. The mapping of the spec
onto ``run`` is pinned through the command in ``test_cli``."""

from __future__ import annotations

import json
import signal
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom.hosted.campaign import CampaignSpec, parse_campaign
from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS
from hyperloom_multibranch.scheduling.hosted_run import (
    GRANT_MARGIN_SECONDS,
    GrantDeadline,
    HostedExit,
    HostedRefused,
    arm_grant_stop,
    grant_deadline,
    record_session,
    recorded_session,
    run_argv,
    write_final_exit,
)

CREATED = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _host(tmp_path: Path, **grant: int) -> HostInfo:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    env = {
        "HYPERLOOM_INPUT_FILE": str(spec),
        "HYPERLOOM_RUN_ID": "run-1",
        "HYPERLOOM_RUN_CREATED_AT": CREATED.isoformat(),
        "HYPERLOOM_STATE_DIR": str(tmp_path),
    }
    if grant:
        env["HYPERLOOM_GRANT_LIMITS"] = json.dumps(grant)
    return read_host_info(env, require=())


def test_the_session_ends_a_close_grace_and_a_margin_before_the_grant_and_is_stopped_a_grace_before_it(
    tmp_path: Path,
) -> None:
    deadline = grant_deadline(_host(tmp_path, max_runtime_sec=7200), CREATED + timedelta(seconds=3600))

    assert deadline == GrantDeadline(3600.0)
    assert deadline.stop_after_sec == 3600 - MB_CLOSE_GRACE_CAP_SECONDS
    assert deadline.session_hours(8.0) == (3600 - MB_CLOSE_GRACE_CAP_SECONDS - GRANT_MARGIN_SECONDS) / 3600
    assert deadline.session_hours(0.5) == 0.5, "a spec asking less than the grant has left keeps its hours"
    assert deadline.session_hours(0.0) == deadline.session_hours(8.0), "no limit of its own is what the grant leaves"


def test_a_grant_that_states_no_runtime_leaves_the_specs_hours_and_stops_nothing(tmp_path: Path) -> None:
    deadline = grant_deadline(_host(tmp_path, max_gpus=8), CREATED + timedelta(days=30))

    assert (deadline.session_hours(8.0), deadline.stop_after_sec) == (8.0, None)


@pytest.mark.parametrize("left", [-10.0, 0.0, MB_CLOSE_GRACE_CAP_SECONDS + GRANT_MARGIN_SECONDS])
def test_a_grant_without_time_for_a_session_refuses_it(left: float) -> None:
    with pytest.raises(HostedRefused, match="CLOSE grace"):
        GrantDeadline(left).session_hours(8.0)


def test_the_grant_stop_sends_this_process_sigterm() -> None:
    received = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: received.set())
    try:
        timer = arm_grant_stop(0.01)
        timer.join(timeout=5)
        assert received.wait(timeout=5)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_the_host_reads_a_final_ending_and_the_session_a_restart_finishes(tmp_path: Path) -> None:
    assert recorded_session(tmp_path) is None

    record_session(tmp_path, tmp_path / "MB-session")
    write_final_exit(tmp_path, HostedExit(1, stop_reason="signal"))

    assert recorded_session(tmp_path) == tmp_path / "MB-session"
    exit_document = json.loads((tmp_path / "exit.json").read_text(encoding="utf-8"))
    assert exit_document == {"final": True, "returncode": 1, "stop_reason": "signal"}
    write_final_exit(tmp_path, HostedExit(2, error="multibranch.workload is missing"))
    exit_document = json.loads((tmp_path / "exit.json").read_text(encoding="utf-8"))
    assert exit_document == {"final": True, "returncode": 2, "error": "multibranch.workload is missing"}


def _document(**multibranch: object) -> dict[str, object]:
    return {
        "schema": "hyperloom.campaign-spec/v1",
        "flow": "multibranch",
        "model": {"path": "/models/Llama-3-8B"},
        "framework": {"name": "vllm", "image": "registry.example/vllm@sha256:" + "2" * 64},
        "hardware": {"gpu_type": "mi355x", "tp": 1},
        "workload": {"conc": 16, "isl": 1024, "osl": 512, "precision": "fp8"},
        "budget": {"max_hours": 1},
        "placement": {"profile": "hosted-gpu", "measurement": "hosted"},
        "multibranch": {"workload": "inferencex", **multibranch},
    }


def _spec(**multibranch: object) -> CampaignSpec:
    return parse_campaign(_document(**multibranch))


@pytest.mark.parametrize(
    ("launch", "named"),
    [
        ({"server_args": "--mem-fraction-static 0.8"}, "does not read launch.server_args:"),
        ({"extra_env": {"AITER_BF16_FP8_MOE_BOUND": "0"}}, "does not read launch.extra_env:"),
        ({"server_args": "--enforce-eager", "extra_env": {}}, "does not read launch.server_args or launch.extra_env:"),
    ],
    ids=["server-args", "extra-env", "both"],
)
def test_a_spec_whose_launch_a_multibranch_run_would_not_read_is_refused_naming_the_field(
    launch: dict[str, object], named: str
) -> None:
    spec = parse_campaign({**_document(), "launch": launch})

    with pytest.raises(HostedRefused, match=named) as refused:
        run_argv(spec, 1.0, 1)
    assert "its base launch is resolved from the recipe sources" in str(refused.value)


def test_a_spec_without_a_launch_section_maps_onto_run() -> None:
    assert "--workload=inferencex" in run_argv(_spec(), 1.0, 1)


@pytest.mark.parametrize(
    ("given", "switches"),
    [
        ({}, []),  # run's default: on
        ({"make_before_break": True}, ["--make-before-break"]),
        ({"make_before_break": False}, ["--no-make-before-break"]),
    ],
)
def test_the_operator_turns_make_before_break_on_or_off_in_the_spec(
    given: dict[str, bool], switches: list[str]
) -> None:
    argv = run_argv(_spec(**given), 1.0, 8)
    assert [arg for arg in argv if arg.endswith("make-before-break")] == switches


@pytest.mark.parametrize("given", ["false", 0, 1])
def test_a_make_before_break_that_is_not_true_or_false_is_refused(given: object) -> None:
    with pytest.raises(HostedRefused, match="multibranch.make_before_break must be true or false"):
        run_argv(_spec(make_before_break=given), 1.0, 8)
