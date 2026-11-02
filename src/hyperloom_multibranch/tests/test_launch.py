# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launcher of one multi-branch session per operating point, run against a fake ``hyperloom-multibranch`` and a
fake ``docker`` that runs each container's commands on this machine."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch import launch
from hyperloom_multibranch.launch import (
    DONE,
    FAILED,
    PID_FILE,
    REVIEW,
    Containers,
    Host,
    Launch,
    LaunchRefused,
    Point,
    Sessions,
    main,
    plan,
    run_arg_refusals,
    session_argv,
    session_env,
    tp_of,
)
from hyperloom_multibranch.scheduling.layout import REPORT_FILE, REPORTS_DIR, REVIEW_DIR, session_dir_name
from hyperloom_multibranch.scheduling.review import REVIEW_FILE, REVIEW_PAGE

RUN_ARGS = ("--workload", "inferencex", "--tp", "4", "--max-hours", "3.0")
#: A stand-in for ``hyperloom-multibranch run``: it prints its session directory, keeps what it was given there,
#: waits for its review when FAKE_REVIEW names its conc, and ends with a report, failing when FAKE_FAIL names it.
FAKE_MB = """\
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[2:]
conc = int(args[args.index("--conc") + 1])
session = Path(os.environ["FAKE_ROOT"]) / f"MB-20261102T1200{conc % 100:02d}Z-{conc:08x}"
(session / "MB-reports").mkdir(parents=True)
print(session, flush=True)
given = {name: os.environ.get(name) for name in ("ROCR_VISIBLE_DEVICES", "PORT", "INFERENCE_OPTIMIZER_RAY_EXEC")}
(session / "given.json").write_text(json.dumps({"argv": sys.argv[1:], "env": given}))
def end(reason, code):
    report = {"stop_reason": reason, "error": "", "final": {"chosen": {"branch": "b01"}, "reason": "best"}}
    (session / "MB-reports" / "MB-report.json").write_text(json.dumps(report))
    sys.exit(code)
signal.signal(signal.SIGTERM, lambda *_: end("signal", 1))
if str(conc) == os.environ.get("FAKE_REVIEW"):
    review = session / "MB-review" / "review.json"
    review.parent.mkdir()
    review.write_text(json.dumps({"status": "pending", "open_by_iso": "2026-11-02T13:00:00Z"}))
    while json.loads(review.read_text())["status"] == "pending":
        time.sleep(0.05)
if str(conc) == os.environ.get("FAKE_HANG"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN) if os.environ.get("FAKE_DEAF") else None
    time.sleep(600)
end("time_exhausted", 1 if str(conc) == os.environ.get("FAKE_FAIL") else 0)
"""
#: A stand-in for ``docker``: it records each call, and runs an ``exec``'s command here with its ``--env`` values.
FAKE_DOCKER = """\
import json, os, subprocess, sys
from pathlib import Path
with open(os.environ["FAKE_CALLS"], "a") as calls:
    calls.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] != "exec":
    sys.exit(0)
rest, env = sys.argv[2:], dict(os.environ)
while rest[0].startswith("--"):
    flag = rest.pop(0)
    if flag.startswith("--env="):
        key, _, value = flag[len("--env="):].partition("=")
        env[key] = value
sys.exit(subprocess.run(rest[1:], env=env).returncode)
"""


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _env(tmp_path: Path, **fake: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    _script(bin_dir / "docker", FAKE_DOCKER)
    path = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    return {
        **os.environ,
        "PATH": path,
        "FAKE_ROOT": str(tmp_path / "sessions"),
        "FAKE_CALLS": str(tmp_path / "calls"),
        **fake,
    }


def _launch(tmp_path: Path, containers: Containers | None = None, **fake: str) -> Launch:
    points = plan([8, 64], {64: 3.25}, range(8), 0, 4)
    program = str(_script(tmp_path / "fake-mb", FAKE_MB))
    return Launch(points, Sessions(RUN_ARGS, tmp_path / "out", "", program, containers), Host(_env(tmp_path, **fake)))


def _until(launch_: Launch, done: Callable[[], object], seconds: float = 30.0) -> None:
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        launch_.poll()
        if done():
            return
        time.sleep(0.05)
    raise AssertionError(f"not reached in {seconds} s: {[(s.point.name, s.state) for s in launch_.states]}")


def _given(launch_: Launch, conc: int) -> dict[str, object]:
    state = next(state for state in launch_.states if state.point.conc == conc)
    return json.loads((Path(state.session) / "given.json").read_text())


def test_each_point_takes_the_next_share_of_the_cards_and_ports_one_step_on() -> None:
    points = plan([8, 64], {64: 3.25}, range(8), 0, 4)

    assert points == (
        Point(8, None, (0, 1, 2, 3), 8888, 31100),
        Point(64, 3.25, (4, 5, 6, 7), 8988, 31200),
    )
    assert [point.cards for point in plan([4, 8, 12], {}, range(8), 2, 2)] == [(0, 1), (2, 3), (4, 5)]


def test_a_plan_the_cards_cannot_hold_is_refused_with_every_reason() -> None:
    with pytest.raises(LaunchRefused) as refused:
        plan([8, 8, 64], {32: 2.0}, range(8), 0, 4)

    assert str(refused.value) == (
        "each --point once: 8 given twice; --point-hours names 32, which no --point gives;"
        " 2 cards a point is fewer than --tp 4"
    )
    with pytest.raises(LaunchRefused, match="3 points of 4 cards need 12; 8 are given"):
        plan([4, 8, 16], {}, range(8), 4, 4)


def test_the_run_arguments_may_not_set_a_point_s_own_options_and_must_give_the_tp() -> None:
    assert run_arg_refusals(["--tp", "4", "--conc", "8", "--agent-port=31100"]) == [
        "--agent-port is set by the launcher, one per point",
        "--conc is set by the launcher, one per point",
    ]
    assert run_arg_refusals(["--workload", "agentx"]) == ["the run arguments give no --tp"]
    assert (tp_of(["--tp", "1", "--tp=4"]), tp_of(["--tp", "x"])) == (4, None)


def test_a_session_runs_at_its_point_s_conc_ports_and_hours_on_its_cards() -> None:
    point = Point(64, 3.25, (4, 5, 6, 7), 8988, 31200)

    argv = session_argv(point, RUN_ARGS)

    assert argv == [
        "hyperloom-multibranch",
        "run",
        *RUN_ARGS,
        "--conc",
        "64",
        "--agent-port",
        "31200",
        "--max-hours",
        "3.25",
    ]
    assert session_argv(Point(8, None, (0,), 8888, 31100), RUN_ARGS)[-4:] == ["--conc", "8", "--agent-port", "31100"]
    assert session_env(point) == {
        "ROCR_VISIBLE_DEVICES": "4,5,6,7",
        "PORT": "8988",
        "INFERENCE_OPTIMIZER_RAY_EXEC": "0",
    }


def test_a_container_forwards_only_the_variables_set_by_name_and_mounts_at_the_same_path() -> None:
    containers = Containers("img:1", "77", ("/apps", "/data:/mnt/data"), ("HF_TOKEN", "UNSET_ONE"), ("--user=1:1",))
    point = Point(8, None, (0, 1), 8888, 31100)

    started = containers.start(point, {"HF_TOKEN": "secret"})

    assert started[:4] == ["docker", "run", "--detach", "--name=mbl-77-c8"]
    assert started[-9:] == [
        *("--volume=/apps:/apps", "--volume=/data:/mnt/data", "--env=HF_TOKEN", "--user=1:1"),
        *("--entrypoint", "tail", "img:1", "-f", "/dev/null"),
    ]
    assert "secret" not in " ".join(started) and "--env=UNSET_ONE" not in started
    assert containers.exec(point, ["true"], root=True) == ["docker", "exec", "--user=0", "mbl-77-c8", "true"]
    assert containers.remove(point) == ["docker", "rm", "--force", "mbl-77-c8"]
    assert containers.stop(point)[-1] == f'kill -TERM "$(cat {PID_FILE})"'


def test_two_points_run_their_sessions_at_once_and_each_one_s_ending_is_kept(tmp_path: Path) -> None:
    run = _launch(tmp_path, FAKE_FAIL="64")

    run.start()
    _until(run, lambda: not run.poll())
    run.close()

    assert [(s.point.conc, s.state, s.exit_code, s.stop_reason) for s in run.states] == [
        (8, DONE, 0, "time_exhausted"),
        (64, FAILED, 1, "time_exhausted"),
    ]
    assert run.exit_code() == 1
    assert _given(run, 8)["env"] == {
        "ROCR_VISIBLE_DEVICES": "0,1,2,3",
        "PORT": "8888",
        "INFERENCE_OPTIMIZER_RAY_EXEC": "0",
    }
    assert _given(run, 64)["argv"][-2:] == ["--max-hours", "3.25"]
    status = (tmp_path / "out" / "STATUS.md").read_text()
    assert "| C64 | 4,5,6,7 | 8988, 31200 | failed |" in status and "| b01: best |" in status
    document = json.loads((tmp_path / "out" / "launch.json").read_text())
    assert [point["state"] for point in document["points"]] == [DONE, FAILED]
    assert (tmp_path / "out" / "c8.log").read_text().startswith(str(tmp_path / "sessions"))


def test_a_session_waiting_for_its_review_shows_the_commands_until_it_is_decided(tmp_path: Path) -> None:
    run = _launch(tmp_path, FAKE_REVIEW="8")
    run.start()

    _until(run, lambda: run.states[0].state == REVIEW)
    session = run.states[0].session
    status = (tmp_path / "out" / "STATUS.md").read_text()
    assert f"read {session}/{REVIEW_DIR}/{REVIEW_PAGE}" in status
    assert f"`hyperloom-multibranch review --session {session} approve`; decide by about 2026-11-02T13:00:00Z" in status
    (Path(session) / REVIEW_DIR / REVIEW_FILE).write_text(json.dumps({"status": "approved"}))
    _until(run, lambda: not run.poll())
    run.close()

    assert [state.state for state in run.states] == [DONE, DONE]
    assert "Waiting for a review" not in (tmp_path / "out" / "STATUS.md").read_text()


def test_a_stop_ends_each_session_with_sigterm_and_kills_one_that_does_not_end_in_the_grace(tmp_path: Path) -> None:
    run = _launch(tmp_path, FAKE_HANG="8")
    run.start()
    _until(run, lambda: run.states[0].session and run.states[1].state == DONE)
    run.stop(grace=60.0)
    _until(run, lambda: not run.poll())
    assert (run.states[0].state, run.states[0].stop_reason) == (FAILED, "signal")

    deaf = _launch(tmp_path / "deaf", FAKE_HANG="8", FAKE_DEAF="1")
    deaf.start()
    _until(deaf, lambda: deaf.states[0].session and deaf.states[1].state == DONE)
    time.sleep(0.3)  # the session has ignored SIGTERM by now
    deaf.stop(grace=0.5)
    _until(deaf, lambda: not deaf.poll())
    assert (deaf.states[0].state, deaf.states[0].exit_code) == (FAILED, -9)


def test_containers_are_prepared_one_at_a_time_then_run_their_sessions_and_are_removed(tmp_path: Path) -> None:
    containers = Containers("img:1", "77", workdir=str(tmp_path), root_prepare="true", prepare="echo prepared")
    run = _launch(tmp_path, containers)

    run.start()
    _until(run, lambda: not run.poll())
    run.close()

    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    steps = [(call[0], next(arg for arg in call if arg.startswith(("mbl-", "--name=mbl-")))) for call in calls]
    assert steps[:6] == [
        ("run", "--name=mbl-77-c8"),
        ("exec", "mbl-77-c8"),
        ("exec", "mbl-77-c8"),
        ("run", "--name=mbl-77-c64"),
        ("exec", "mbl-77-c64"),
        ("exec", "mbl-77-c64"),
    ]
    # The sessions start together, in either order, once every container is prepared.
    assert sorted(steps[6:8]) == [("exec", "mbl-77-c64"), ("exec", "mbl-77-c8")]
    assert steps[8:] == [("rm", "mbl-77-c8"), ("rm", "mbl-77-c64")]
    assert "--env=SKIP_RAY_START=1" in calls[2] and "--user=0" in calls[1]
    assert [state.state for state in run.states] == [DONE, DONE]
    assert _given(run, 64)["env"]["ROCR_VISIBLE_DEVICES"] == "4,5,6,7"
    assert "prepared" in (tmp_path / "out" / "c8.log").read_text()


def test_a_container_whose_preparation_fails_runs_no_session_and_the_others_go_on(tmp_path: Path) -> None:
    containers = Containers("img:1", "77", prepare=f"mkdir {tmp_path / 'once'} 2>/dev/null || exit 3")
    run = _launch(tmp_path, containers)

    run.start()
    _until(run, lambda: not run.poll())

    assert [(state.state, state.session != "") for state in run.states] == [(DONE, True), (FAILED, False)]
    assert run.states[1].error == f"the preparation exited 3; see {tmp_path / 'out' / 'c64.log'}"
    assert "the preparation exited 3" in (tmp_path / "out" / "STATUS.md").read_text()


def test_the_command_refuses_a_launch_it_cannot_start_before_starting_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "STATUS.md").write_text("")

    code = main(
        [
            "--point",
            "8",
            "--point",
            "64",
            "--out",
            str(tmp_path / "used"),
            "--cards",
            "0-7",
            "--",
            "--tp",
            "4",
            "--conc",
            "8",
        ]
    )

    assert code == 2
    said = capsys.readouterr().err
    assert "--conc is set by the launcher" in said and "--out" in said and "give --image, or --same-host" in said
    assert not (tmp_path / "used" / "launch.json").exists()


def test_a_dry_run_prints_each_point_s_steps_and_starts_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--point",
            "8",
            "--point",
            "64",
            "--out",
            str(tmp_path / "out"),
            "--cards",
            "0-3,6,7",
            "--cards-per-point",
            "2",
            "--image",
            "img:1",
            "--tag",
            "9",
            "--prepare",
            "bash install.sh",
            "--dry-run",
            "--",
            "--tp",
            "2",
        ]
    )

    steps = [line for line in capsys.readouterr().out.splitlines() if line.startswith("C")]
    assert code == 0 and not (tmp_path / "out").exists()
    assert [line.split(":")[0] for line in steps] == [
        "C8, starting the container",
        "C8, the preparation",
        "C8, the session",
        "C64, starting the container",
        "C64, the preparation",
        "C64, the session",
    ]
    assert "--env=ROCR_VISIBLE_DEVICES=2,3" in steps[-1] and "mbl-9-c64" in steps[-1]


def test_what_the_launcher_reads_is_where_mb_writes_it() -> None:
    assert launch.REVIEW_FILE == f"{REVIEW_DIR}/{REVIEW_FILE}"
    assert launch.REVIEW_PAGE == f"{REVIEW_DIR}/{REVIEW_PAGE}"
    assert launch.REPORT_FILE == f"{REPORTS_DIR}/{REPORT_FILE}"
    assert launch.SESSION_LINE.fullmatch(f"/data/qwen/{session_dir_name(1793500000.0)}")
