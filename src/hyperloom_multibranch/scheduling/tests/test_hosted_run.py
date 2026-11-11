# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted run reads from the host and writes back: the grant's bound on the session's time, the stop the
grant's end sends, the files that tell the host whether to start the run again, and the isolation its agents must
have. The mapping of the spec onto ``run`` is pinned through the command in ``test_cli``."""

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
from hyperloom_common.tests._bwrap_host import usable_bwrap
from hyperloom.common import agent_isolation
from hyperloom.common.env_safety import SPECIALIST_USER_ENV, RunAs
from hyperloom.hosted.campaign import CampaignSpec, parse_campaign
from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom_multibranch.scheduling import hosted_run as hosted_run_module
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS
from hyperloom_multibranch.scheduling.launch_isolation import HOSTED, LaunchIsolation, launch_isolation
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
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


def _spec(**multibranch: object) -> CampaignSpec:
    return parse_campaign(
        {
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
    )


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


LAYER = Path("/opt/layer")
LAYER_BWRAP = str(LAYER / "bin" / "bwrap")
#: An account the chooser takes, as ``HYPERLOOM_SPECIALIST_USER`` names it.
ACCOUNT = RunAs(user="hl-specialist", uid=1500, gid=1500, groups=(1500,), home="/home/hl", drop_tool="/usr/bin/setpriv")
DENIED = "bwrap: setting up uid map: Permission denied"


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

    def __call__(self, path: str | None, *_: object) -> BwrapProbe:
        self.asked.append(path)
        if path not in self.answers:
            return BwrapProbe(path, None, ("bubblewrap: not found",))
        return BwrapProbe(path, (0, 13, 1), self.answers[path], transient=self.transient)


@pytest.fixture
def unmarked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process no hosted run has marked and no specialist account is configured for, its bubblewrap probes asked
    afresh."""

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(bwrap, "_PROBES", {})
    monkeypatch.delenv(SPECIALIST_USER_ENV, raising=False)


@pytest.fixture
def probes(unmarked: None, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Probes]:
    """An :func:`unmarked` process whose bubblewrap probe is the stand-in the test sets."""

    def install(answers: dict[str, tuple[str, ...]], **options: bool) -> Probes:
        stand_in = Probes(answers, **options)
        monkeypatch.setattr(bwrap, "_probe", stand_in)
        return stand_in

    return install


@pytest.fixture
def account(unmarked: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """``HYPERLOOM_SPECIALIST_USER`` names an account this process can start a child as and signal: the chooser takes
    it, and the agent runner starts every specialist as it."""

    monkeypatch.setenv(SPECIALIST_USER_ENV, ACCOUNT.user)
    monkeypatch.setattr(agent_isolation, "resolve_run_as", lambda name: ACCOUNT if name == ACCOUNT.user else None)
    monkeypatch.setattr(agent_isolation, "missing_launch_capabilities", list)


@pytest.mark.usefixtures("account")
def test_a_hosted_run_marks_its_process_with_the_layers_bubblewrap_and_no_other(
    probes: Callable[..., Probes], caplog: pytest.LogCaptureFixture
) -> None:
    probed = probes({LAYER_BWRAP: ()})

    with caplog.at_level(logging.INFO):
        require_hosted_isolation({HOST_LAYER_ENV: str(LAYER)})

    assert caplog.records == [], "the session says how it launches its agents once, when its peers are built"
    assert probed.asked == [LAYER_BWRAP], "the account is chosen, and the layer's bubblewrap probed for the Ray path"
    assert agent_isolation.isolation_required(), "every later launch of this process must be isolated"
    later = agent_isolation.choose_sandbox_isolation()
    assert isinstance(later, agent_isolation.BwrapIsolation) and set(probed.asked) == {LAYER_BWRAP}, (
        "a later launch under bubblewrap uses the layer's, never one on PATH, maybe the image's own"
    )


@pytest.mark.usefixtures("account")
def test_a_hosted_run_with_an_account_needs_only_its_specialists_isolated_here(probes: Callable[..., Probes]) -> None:
    probes({LAYER_BWRAP: (DENIED,)})

    require_hosted_isolation({HOST_LAYER_ENV: str(LAYER)})

    assert launch_isolation(local=False) == LaunchIsolation(agent_isolation.ACCOUNT, HOSTED, HOSTED, True, ())


def test_a_hosted_run_with_no_account_starts_with_its_specialists_under_the_layers_bubblewrap(
    probes: Callable[..., Probes],
) -> None:
    probed = probes({LAYER_BWRAP: ()})

    require_hosted_isolation({HOST_LAYER_ENV: str(LAYER)})

    assert launch_isolation(local=False) == LaunchIsolation(agent_isolation.BWRAP, HOSTED, HOSTED, True, ())
    assert set(probed.asked) == {LAYER_BWRAP}, "the runner's chooser takes the layer's bubblewrap for the specialists"


def test_a_hosted_run_whose_host_names_no_layer_is_refused_and_probes_no_bubblewrap(
    probes: Callable[..., Probes],
) -> None:
    probed = probes({LAYER_BWRAP: ()})

    with pytest.raises(PeersUnavailable) as refused:
        require_hosted_isolation({})

    assert str(refused.value) == (
        f"agents cannot be isolated on this host: running them as an account: {SPECIALIST_USER_ENV} is not set; "
        f"{agent_isolation.NO_LAYER_BWRAP}"
    )
    assert probed.asked == []


def test_a_hosted_run_where_no_tier_can_isolate_a_specialist_is_refused_naming_both(
    probes: Callable[..., Probes],
) -> None:
    probes({LAYER_BWRAP: (DENIED,)})

    with pytest.raises(PeersUnavailable) as refused:
        require_hosted_isolation({HOST_LAYER_ENV: str(LAYER)})

    assert str(refused.value) == (
        f"agents cannot be isolated on this host: running them as an account: {SPECIALIST_USER_ENV} is not set; "
        f"bubblewrap ({LAYER_BWRAP}, 0.13.1): {DENIED}"
    )


@pytest.mark.usefixtures("account")
def test_a_hosted_run_whose_runner_would_start_its_specialists_unconfined_is_refused_naming_them(
    probes: Callable[..., Probes], monkeypatch: pytest.MonkeyPatch
) -> None:
    probes({LAYER_BWRAP: ()})
    unconfined = LaunchIsolation(agent_isolation.NONE, HOSTED, HOSTED, True, ("the runner asks no chooser",))
    monkeypatch.setattr(hosted_run_module, "launch_isolation", lambda *, local: unconfined)

    with pytest.raises(PeersUnavailable) as refused:
        require_hosted_isolation({HOST_LAYER_ENV: str(LAYER)})

    assert str(refused.value) == (
        "a hosted run launches nothing unconfined beside its credentials, and here specialists run unconfined on this"
        " host (the runner asks no chooser)"
    )


def test_a_bubblewrap_that_gives_no_answer_in_time_leaves_the_run_to_be_started_again(
    probes: Callable[..., Probes],
) -> None:
    probes({LAYER_BWRAP: (f"bubblewrap at {LAYER_BWRAP}: no answer in 10 s",)}, transient=True)

    with pytest.raises(TimeoutError, match="no answer in 10 s"):
        require_hosted_isolation({HOST_LAYER_ENV: str(LAYER)})


def _layer(tmp_path: Path, bwrap_target: str | None) -> Path:
    """A hosting layer directory whose ``bin/bwrap`` links to ``bwrap_target`` (none when ``None``), and a directory
    ``image-bin`` whose ``bwrap`` cannot run a sandbox, as a framework image's own might be."""

    layer = tmp_path / "layer"
    (layer / "bin").mkdir(parents=True)
    if bwrap_target is not None:
        (layer / "bin" / "bwrap").symlink_to(bwrap_target)
    image_bin = tmp_path / "image-bin"
    image_bin.mkdir()
    (image_bin / "bwrap").write_text("#!/bin/sh\necho 'not the layer bubblewrap' >&2\nexit 1\n")
    (image_bin / "bwrap").chmod(0o755)
    return layer


@pytest.mark.agent_isolation
@pytest.mark.usefixtures("unmarked")
def test_a_hosted_run_with_no_account_and_a_real_layer_bubblewrap_starts_its_specialists_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = usable_bwrap()
    layer = _layer(tmp_path, str(real.path))
    monkeypatch.setenv("PATH", f"{tmp_path / 'image-bin'}:/usr/bin:/bin")

    require_hosted_isolation({HOST_LAYER_ENV: str(layer)})

    assert launch_isolation(local=False) == LaunchIsolation(agent_isolation.BWRAP, HOSTED, HOSTED, True, ())
    later = agent_isolation.choose_sandbox_isolation()
    assert isinstance(later, agent_isolation.BwrapIsolation), later
    version = ".".join(map(str, real.version or ()))
    assert later.shown == f"{version} {layer / 'bin' / 'bwrap'}"


@pytest.mark.usefixtures("unmarked")
def test_a_hosted_run_whose_layer_has_no_bubblewrap_is_refused_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = _layer(tmp_path, None)
    monkeypatch.setenv("PATH", f"{tmp_path / 'image-bin'}:/usr/bin:/bin")

    with pytest.raises(PeersUnavailable) as refused:
        require_hosted_isolation({HOST_LAYER_ENV: str(layer)})

    assert f"bubblewrap at {layer / 'bin' / 'bwrap'}: No such file or directory" in str(refused.value)
    assert "not the layer bubblewrap" not in str(refused.value)
