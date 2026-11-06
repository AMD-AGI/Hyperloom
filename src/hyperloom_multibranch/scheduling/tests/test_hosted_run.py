# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted run reads from the host and writes back: the grant's bound on the session's time, the stop the
grant's end sends, the files that tell the host whether to start the run again, the image a spec measured on this
container's cards must name, and the isolation its agents must have. The mapping of the spec onto ``run`` is pinned
through the command in ``test_cli``."""

from __future__ import annotations

import json
import logging
import signal
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom_common import bwrap
from hyperloom_common.bwrap import BwrapProbe
from hyperloom.common import agent_isolation
from hyperloom.common.env_safety import SPECIALIST_USER_ENV, RunAs
from hyperloom.hosted.campaign import CampaignSpec, parse_campaign
from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS
from hyperloom_multibranch.scheduling.hosted_run import (
    GRANT_MARGIN_SECONDS,
    HOST_LAYER_ENV,
    GrantDeadline,
    HostedExit,
    HostedRefused,
    arm_grant_stop,
    grant_deadline,
    layer_bwrap,
    record_session,
    recorded_session,
    require_hosted_isolation,
    require_local_image,
    run_argv,
    write_final_exit,
)
from hyperloom_multibranch.scheduling.peers import PeersUnavailable

CREATED = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
DIGEST = "sha256:" + "2" * 64
#: The measurement pins of a container that runs the image ``registry.example/vllm:v1`` at :data:`DIGEST`.
PINS = {"HYPERLOOM_ACTIVE_IMAGE_REF": "registry.example/vllm:v1", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": DIGEST}
LAYER = Path("/opt/layer")
LAYER_BWRAP = str(LAYER / "bin" / "bwrap")
ACCOUNT = RunAs(user="hl-specialist", uid=1500, gid=1500, groups=(1500,), home="/home/hl", drop_tool="/usr/bin/setpriv")
DENIED = "bwrap: setting up uid map: Permission denied"


def _spec(measurement: str, image: str = f"registry.example/vllm@{DIGEST}") -> CampaignSpec:
    return parse_campaign(
        {
            "schema": "hyperloom.campaign-spec/v1",
            "flow": "multibranch",
            "model": {"path": "/models/Llama-3-8B"},
            "framework": {"name": "vllm", "image": image},
            "hardware": {"gpu_type": "mi355x", "tp": 1},
            "workload": {"conc": 16, "isl": 1024, "osl": 512, "precision": "fp8"},
            "budget": {"max_hours": 1},
            "placement": {"profile": "hosted-gpu", "measurement": measurement},
            "multibranch": {"workload": "inferencex"},
        }
    )


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


@pytest.mark.parametrize("measurement", ["local", "hosted"])
def test_the_run_command_line_says_where_the_spec_measures(measurement: str) -> None:
    assert f"--measurement={measurement}" in run_argv(_spec(measurement), 1.0, 8)


@pytest.mark.parametrize(
    "image",
    [f"registry.example/vllm@{DIGEST}", f"registry.example/vllm:v1@{DIGEST}", f"mirror.example/any/name@{DIGEST}"],
)
def test_a_spec_measured_here_names_the_image_this_container_runs_by_its_digest(image: str) -> None:
    require_local_image(_spec("local", image), PINS)


@pytest.mark.parametrize(
    "image",
    ["submitted.example/vllm:v1", f"submitted.example/vllm@sha256:{'3' * 64}", f"submitted.example/vllm@{DIGEST}0"],
    ids=["by-tag", "another-digest", "a-longer-digest"],
)
def test_a_spec_measured_here_on_another_image_is_refused_naming_the_one_this_container_runs(image: str) -> None:
    with pytest.raises(HostedRefused) as refused:
        require_local_image(_spec("local", image), PINS)

    assert str(refused.value) == (
        f"placement.measurement 'local' measures the image this container runs, registry.example/vllm:v1 at {DIGEST}:"
        " framework.image must name it by that digest"
    )
    assert image not in str(refused.value), "a string the submitter wrote is not echoed"


def test_a_spec_measured_here_without_the_pins_is_refused_naming_what_is_missing() -> None:
    with pytest.raises(PeersUnavailable, match="HYPERLOOM_ACTIVE_IMAGE_DIGEST not set"):
        require_local_image(_spec("local"), {"HYPERLOOM_ACTIVE_IMAGE_REF": "registry.example/vllm:v1"})


def test_a_spec_measured_in_the_grants_sandboxes_is_not_held_to_this_containers_image() -> None:
    require_local_image(_spec("hosted", "registry.example/vllm:v1"), {})


@pytest.mark.parametrize(("value", "found"), [(" /opt/layer ", LAYER_BWRAP), ("", None), ("  ", None), (None, None)])
def test_the_layers_bubblewrap_is_found_in_the_layer_directory_the_host_names(
    value: str | None, found: str | None
) -> None:
    assert layer_bwrap({} if value is None else {HOST_LAYER_ENV: value}) == found


class Probes:
    """A stand-in for bubblewrap's probe: each binary named in ``answers`` answers its problems, anything else is not
    there; records every binary asked about."""

    def __init__(self, answers: dict[str, tuple[str, ...]], *, transient: bool = False) -> None:
        self.answers, self.transient = answers, transient
        self.asked: list[str | None] = []

    def __call__(self, path: str | None) -> BwrapProbe:
        self.asked.append(path)
        if path not in self.answers:
            return BwrapProbe(path, None, ("bubblewrap: not found",))
        return BwrapProbe(path, (0, 13, 1), self.answers[path], transient=self.transient)


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Probes]:
    """A process no hosted run has marked and no specialist account is configured for, whose bubblewrap probe is the
    stand-in the test sets."""

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(bwrap, "_PROBES", {})
    monkeypatch.delenv(SPECIALIST_USER_ENV, raising=False)

    def install(answers: dict[str, tuple[str, ...]], **options: bool) -> Probes:
        stand_in = Probes(answers, **options)
        monkeypatch.setattr(bwrap, "probe_bwrap", stand_in)
        return stand_in

    return install


@pytest.fixture
def account(probes: Callable[..., Probes], monkeypatch: pytest.MonkeyPatch) -> None:
    """``HYPERLOOM_SPECIALIST_USER`` names an account this process can start a child as and signal."""

    monkeypatch.setenv(SPECIALIST_USER_ENV, ACCOUNT.user)
    monkeypatch.setattr(agent_isolation, "resolve_run_as", lambda name: ACCOUNT if name == ACCOUNT.user else None)
    monkeypatch.setattr(agent_isolation, "missing_launch_capabilities", list)


@pytest.mark.parametrize("measurement", ["local", "hosted"])
def test_a_hosted_run_isolates_its_agents_with_the_layers_bubblewrap_and_no_other(
    measurement: str, probes: Callable[..., Probes], caplog: pytest.LogCaptureFixture
) -> None:
    probed = probes({LAYER_BWRAP: ()})

    with caplog.at_level(logging.INFO, logger=agent_isolation.__name__):
        require_hosted_isolation(_spec(measurement), {HOST_LAYER_ENV: str(LAYER)})

    assert set(probed.asked) == {LAYER_BWRAP}, "the bubblewrap on PATH, maybe the image's own, is never probed"
    assert agent_isolation.isolation_required(), "every later launch of this process must be isolated"
    said = [record for record in caplog.records if record.name == agent_isolation.__name__]
    assert [record.levelno for record in said] == [logging.INFO]
    assert said[0].getMessage() == (
        f"agent isolation: specialists bwrap, sandboxes bwrap (bubblewrap 0.13.1 {LAYER_BWRAP}): running them as an"
        f" account: {SPECIALIST_USER_ENV} is not set; running them as an account: {agent_isolation.SANDBOX_ACCOUNT}"
    )


@pytest.mark.parametrize("measurement", ["local", "hosted"])
def test_a_hosted_run_whose_host_names_no_layer_is_refused_and_probes_no_bubblewrap(
    measurement: str, probes: Callable[..., Probes]
) -> None:
    probed = probes({LAYER_BWRAP: ()})

    with pytest.raises(PeersUnavailable) as refused:
        require_hosted_isolation(_spec(measurement), {})

    assert str(refused.value) == (
        f"agents cannot be isolated on this host: running them as an account: {SPECIALIST_USER_ENV} is not set; "
        f"{agent_isolation.NO_LAYER_BWRAP}"
    )
    assert probed.asked == []


@pytest.mark.usefixtures("account")
def test_agents_measured_in_the_grants_sandboxes_need_only_the_specialists_isolated(
    probes: Callable[..., Probes],
) -> None:
    probes({LAYER_BWRAP: (DENIED,)})

    require_hosted_isolation(_spec("hosted"), {HOST_LAYER_ENV: str(LAYER)})


@pytest.mark.usefixtures("account")
def test_agents_measured_here_run_their_sandboxes_here_and_need_bubblewrap_for_them(
    probes: Callable[..., Probes],
) -> None:
    probes({LAYER_BWRAP: (DENIED,)})

    with pytest.raises(PeersUnavailable) as refused:
        require_hosted_isolation(_spec("local"), {HOST_LAYER_ENV: str(LAYER)})

    assert str(refused.value) == (
        f"agents cannot be isolated on this host: running them as an account: {agent_isolation.SANDBOX_ACCOUNT}; "
        f"bubblewrap ({LAYER_BWRAP}, 0.13.1): {DENIED}"
    )


def test_a_bubblewrap_that_gives_no_answer_in_time_leaves_the_run_to_be_started_again(
    probes: Callable[..., Probes],
) -> None:
    probes({LAYER_BWRAP: (f"bubblewrap at {LAYER_BWRAP}: no answer in 10 s",)}, transient=True)

    with pytest.raises(TimeoutError, match="no answer in 10 s"):
        require_hosted_isolation(_spec("local"), {HOST_LAYER_ENV: str(LAYER)})
