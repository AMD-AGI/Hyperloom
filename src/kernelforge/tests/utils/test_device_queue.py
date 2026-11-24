# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The queue a round's lanes wait in for one device, and the runner a lane's commands go through."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kernelforge.utils import device_queue
from kernelforge.utils.device_queue import (
    DEVICE_VISIBILITY_VARIABLES,
    HOLD_LIMIT_EXIT_STATUS,
    RUNNER_CONFIG_NAME,
    RUNNER_NAME,
    DeviceQueue,
    hidden_device_environment,
    runner_config,
    visible_device_environment,
    write_runner_config,
)


def _install_runner(directory: Path, *, queue_root: Path, hold_limit_sec: int = 60, environment=None) -> Path:
    """Put the runner in ``directory`` the way a round puts it in a lane: a copy of the module beside its config."""
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / RUNNER_NAME
    shutil.copyfile(device_queue.__file__, runner)
    write_runner_config(
        directory / RUNNER_CONFIG_NAME,
        runner_config(
            queue_root=queue_root,
            usage_log=directory / "usage.jsonl",
            lane="1",
            hold_limit_sec=hold_limit_sec,
            environment=environment if environment is not None else {"HIP_VISIBLE_DEVICES": "3"},
        ),
    )
    return runner


def _run(runner: Path, *command: str, env: dict | None = None) -> subprocess.CompletedProcess:
    hidden = {**os.environ, **hidden_device_environment(), **(env or {})}
    return subprocess.run(
        [sys.executable, str(runner), *command], capture_output=True, text=True, env=hidden, timeout=60, check=False
    )


def _usage(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "usage.jsonl").read_text().splitlines()]


def test_the_device_is_granted_in_the_order_the_waiters_arrived(tmp_path):
    queue = DeviceQueue(tmp_path / "queue")
    order: list[int] = []

    def wait_for_it(number: int) -> None:
        with queue.held():
            order.append(number)

    with queue.held():
        waiters = []
        for number in (1, 2, 3):
            waiter = threading.Thread(target=wait_for_it, args=(number,))
            waiter.start()
            waiters.append(waiter)
            # Each ticket exists before the next waiter arrives.
            while len(list(queue.root.iterdir())) < number + 1:
                time.sleep(0.01)
        assert order == []
    for waiter in waiters:
        waiter.join(timeout=30)

    assert order == [1, 2, 3]
    assert list(queue.root.iterdir()) == []


def test_a_waiter_whose_process_died_does_not_hold_the_queue(tmp_path):
    queue = DeviceQueue(tmp_path / "queue")
    queue.root.mkdir()
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    (queue.root / f"{1:012d}").write_text(str(exited.pid))

    with queue.held() as waited:
        assert waited < 5.0

    assert list(queue.root.iterdir()) == []


def test_a_ticket_whose_owners_id_went_to_another_process_does_not_hold_the_queue(tmp_path):
    """A process id is reused; the start time recorded beside it is not."""
    queue = DeviceQueue(tmp_path / "queue")
    queue.root.mkdir()
    (queue.root / f"{1:012d}").write_text(f"{os.getpid()} 1")

    with queue.held() as waited:
        assert waited < 5.0

    assert list(queue.root.iterdir()) == []


async def test_a_cancelled_wait_gives_its_place_up(tmp_path):
    """The loop's own measurements wait as coroutines; one that is cancelled must not leave a ticket nobody releases."""
    queue = DeviceQueue(tmp_path / "queue")

    async def wait_for_it() -> None:
        async with queue.held_async():
            raise AssertionError("the device was never free")

    async with queue.held_async():
        waiting = asyncio.create_task(wait_for_it())
        await asyncio.sleep(0.3)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert len(list(queue.root.iterdir())) == 1

    assert list(queue.root.iterdir()) == []


def test_the_runner_makes_the_device_visible_to_the_command_and_to_nothing_else(tmp_path):
    """A session's own environment hides every device; only what goes through the runner sees the campaign's."""
    runner = _install_runner(
        tmp_path / "lane",
        queue_root=tmp_path / "queue",
        environment={"HIP_VISIBLE_DEVICES": "3", "CUDA_VISIBLE_DEVICES": None},
    )
    show = "import os; print(os.environ.get('HIP_VISIBLE_DEVICES'), os.environ.get('CUDA_VISIBLE_DEVICES'))"

    direct = subprocess.run(
        [sys.executable, "-c", show],
        capture_output=True,
        text=True,
        env={**os.environ, **hidden_device_environment()},
        check=True,
    )
    through = _run(runner, sys.executable, "-c", show)

    assert direct.stdout.split() == []
    assert through.stdout.split() == ["3", "None"]
    assert through.returncode == 0


def test_the_runner_passes_arguments_input_and_exit_status_through(tmp_path):
    runner = _install_runner(tmp_path / "lane", queue_root=tmp_path / "queue")
    echo = "import sys; print(sys.argv[1:], sys.stdin.read().strip()); sys.exit(7)"

    result = subprocess.run(
        [sys.executable, str(runner), sys.executable, "-c", echo, "--bench-mode", "a b"],
        input="from stdin",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.stdout.strip() == "['--bench-mode', 'a b'] from stdin"
    assert result.returncode == 7


def test_the_runner_records_how_long_it_waited_and_held(tmp_path):
    lane = tmp_path / "lane"
    runner = _install_runner(lane, queue_root=tmp_path / "queue")

    _run(runner, sys.executable, "-c", "import time; time.sleep(0.3)")

    [record] = _usage(lane)
    assert record["lane"] == "1"
    assert record["hold_sec"] >= 0.3
    assert record["wait_sec"] < record["hold_sec"]
    assert (record["status"], record["hold_limit_reached"]) == (0, False)
    assert "time.sleep(0.3)" in record["command"]


def test_a_command_that_outlasts_the_hold_limit_is_stopped_with_everything_it_started(tmp_path):
    """One command may not keep the other lanes waiting past the limit, and what it started may not keep the device."""
    lane = tmp_path / "lane"
    runner = _install_runner(lane, queue_root=tmp_path / "queue", hold_limit_sec=1)
    child_pid = tmp_path / "child.pid"
    spawns_a_child = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"open({str(child_pid)!r}, 'w').write(str(child.pid)); time.sleep(120)"
    )

    result = _run(runner, sys.executable, "-c", spawns_a_child)

    assert result.returncode == HOLD_LIMIT_EXIT_STATUS
    assert "one command may hold the device no longer" in result.stderr
    assert _usage(lane)[0]["hold_limit_reached"] is True
    grandchild = int(child_pid.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and Path(f"/proc/{grandchild}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{grandchild}").exists()
    assert list((tmp_path / "queue").iterdir()) == []


def test_two_runners_never_hold_the_device_at_once(tmp_path):
    queue_root = tmp_path / "queue"
    runners = [_install_runner(tmp_path / f"lane{number}", queue_root=queue_root) for number in (1, 2, 3)]
    stamp = "import time; print(time.time()); time.sleep(0.4); print(time.time())"

    processes = [
        subprocess.Popen([sys.executable, str(runner), sys.executable, "-c", stamp], stdout=subprocess.PIPE, text=True)
        for runner in runners
    ]
    spans = sorted(tuple(float(value) for value in process.communicate(timeout=60)[0].split()) for process in processes)

    assert all(earlier[1] <= later[0] for earlier, later in zip(spans, spans[1:]))


def test_the_visibility_a_runner_restores_is_what_the_campaign_started_with():
    environ = {"HIP_VISIBLE_DEVICES": "5", "PATH": "/usr/bin"}

    assert visible_device_environment(environ) == {"HIP_VISIBLE_DEVICES": "5", "CUDA_VISIBLE_DEVICES": None}
    assert hidden_device_environment() == dict.fromkeys(DEVICE_VISIBILITY_VARIABLES, "")
