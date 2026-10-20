# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The E2B stand-in itself: that it behaves like the service the provider talks to, and that it cleans up after itself.

The provider is the client here, so these tests are also its first run against real envd processes. Where the machine
cannot run the stand-in the ``standin`` fixture skips the test with the reason.
"""

from __future__ import annotations

import contextlib
import http.client
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import pytest

from hyperloom.sandbox import CreateOptions
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy
from hyperloom.sandbox.socket import IDEMPOTENCY_KEY, SandboxDenied, SandboxError, SandboxGone, SandboxUnavailable
from hyperloom.sandbox.tests.e2b_standin import ENVD_ENV, StandIn, StandInTemplate, unavailable_reason
from hyperloom.sandbox.tests.e2b_standin.envd_host import RunningSandbox, SandboxPlan
from hyperloom.sandbox.tests.e2b_standin.service import ServiceError

FAST = RetryPolicy(budget_sec=5.0, first_delay_sec=0.02, max_delay_sec=0.1)


def _provider(standin: StandIn) -> E2BSandboxProvider:
    return E2BSandboxProvider(standin.keys, api_url=standin.api_url, sandbox_url=standin.sandbox_url, retry=FAST)


def _processes_matching(marker: str) -> list[int]:
    """PIDs whose command line contains ``marker`` (a read-only scan of /proc; nothing is signalled)."""
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                cmdline = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if marker.encode() in cmdline:
                found.append(int(entry.name))
    return found


def test_the_skip_reason_names_what_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    assert ENVD_ENV in str(unavailable_reason(None))
    assert "not an executable file" in str(unavailable_reason(str(tmp_path / "nope")))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert "unshare, chroot and timeout" in str(unavailable_reason(None))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert "needs root" in str(unavailable_reason(None))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "needs Linux" in str(unavailable_reason(None))


def test_a_sandbox_is_a_real_envd_behind_the_router(standin: StandIn) -> None:
    provider = _provider(standin)
    sandbox = provider.create(
        "dev", role="dev", timeout_sec=60, metadata={"k": "v"}, options=CreateOptions(envs={"CALLER": "mine"})
    )
    result = sandbox.run("echo $CALLER $E2B_SANDBOX_ID $E2B_TEMPLATE_ID $NODE_NAME; id -un")
    assert result.stdout == f"mine {sandbox.sandbox_id} dev standin-node\nroot\n"
    info = sandbox.info()
    assert (info.template, dict(info.metadata)) == ("dev", {"k": "v"})
    sandbox.kill()


def test_sandboxes_are_isolated_from_each_other_and_from_this_machine(standin: StandIn) -> None:
    provider = _provider(standin)
    first = provider.create("dev", role="dev", timeout_sec=60, metadata={})
    second = provider.create("dev", role="dev", timeout_sec=60, metadata={})
    marker = f"leftover-{first.sandbox_id}"
    first.write_file(f"/work/{marker}", b"x")
    first.write_file(f"/tmp/{marker}", b"x")
    first.write_file(f"/dev/shm/{marker}", b"x")
    assert second.list_dir("/work") == [] and second.list_dir("/tmp") == [] and second.list_dir("/dev/shm") == []
    assert not any(Path(d, marker).exists() for d in ("/work", "/tmp", "/dev/shm"))
    assert first.run("ls /in /out /work").exit_code == 0
    first.kill()
    second.kill()


def test_a_sandbox_shows_none_of_this_machines_etc_beyond_what_programs_need(standin: StandIn) -> None:
    sandbox = _provider(standin).create("dev", role="dev", timeout_sec=60, metadata={})
    shown = set(sandbox.list_dir("/etc"))
    assert shown <= {
        "passwd",
        "group",
        "nsswitch.conf",
        "hosts",
        "resolv.conf",
        "ld.so.cache",
        "ld.so.conf",
        "ld.so.conf.d",
    }
    assert sandbox.run("test -e /etc/shadow -o -e /etc/ssh -o -e /root/.ssh").exit_code != 0
    assert set(sandbox.list_dir("/dev")) <= {"null", "zero", "full", "random", "urandom", "tty", "shm"}
    sandbox.kill()


def test_the_templates_start_command_runs_inside_the_sandbox(standin: StandIn) -> None:
    sandbox = _provider(standin).create("sealed", role="measurement", timeout_sec=60, metadata={})
    deadline = time.monotonic() + 10
    while "started" not in sandbox.list_dir("/out") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sandbox.read_file("/out/started") == b"ready\n"
    sandbox.kill()


def test_keys_templates_and_sealed_environment_are_enforced(standin: StandIn) -> None:
    provider = _provider(standin)
    sandbox = provider.create("dev", role="dev", timeout_sec=60, metadata={"who": "dev"})
    assert provider.connect(sandbox.sandbox_id, role="dev").sandbox_id == sandbox.sandbox_id
    with pytest.raises(SandboxGone):
        provider.connect(sandbox.sandbox_id, role="subrun")
    assert provider.find({"who": "dev"}, role="subrun") == []
    for template, role in (("sub", "dev"), ("dev", "measurement"), ("nope", "dev")):
        with pytest.raises(SandboxDenied):
            provider.create(template, role=role, timeout_sec=60, metadata={})
    with pytest.raises(SandboxDenied):
        provider.create(
            "sealed", role="measurement", timeout_sec=60, metadata={}, options=CreateOptions(envs={"X": "1"})
        )
    forged = E2BSandboxProvider(
        {"dev": "e2b_not_a_key"}, api_url=standin.api_url, sandbox_url=standin.sandbox_url, retry=FAST
    )
    with pytest.raises(SandboxDenied):
        forged.create("dev", role="dev", timeout_sec=60, metadata={})
    sandbox.kill()


def test_one_idempotency_key_makes_one_sandbox_even_when_creates_race(standin: StandIn) -> None:
    provider = _provider(standin)
    before = len(standin.host.pids)
    made: list[str] = []
    lock = threading.Lock()

    def create() -> None:
        sandbox = provider.create("dev", role="dev", timeout_sec=60, metadata={IDEMPOTENCY_KEY: "race-1"})
        with lock:
            made.append(sandbox.sandbox_id)

    threads = [threading.Thread(target=create) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(made)) == 1 and len(made) == 4
    assert len(standin.host.pids) == before + 1
    provider.connect(made[0], role="dev").kill()


def test_a_listing_is_paged_and_a_killed_sandbox_leaves_it(e2b_envd: Path, tmp_path: Path) -> None:
    with StandIn(e2b_envd, tmp_path, [StandInTemplate("dev", "dev")], page_size=2) as small:
        provider = _provider(small)
        made = [provider.create("dev", role="dev", timeout_sec=60, metadata={"set": "a"}) for _ in range(5)]
        assert [i.sandbox_id for i in provider.find({"set": "a"}, role="dev")] == sorted(s.sandbox_id for s in made)
        made[0].kill()
        assert len(provider.find({}, role="dev")) == 4


def test_the_host_refuses_to_stop_a_process_it_did_not_start(standin: StandIn) -> None:
    foreign = subprocess.Popen(["sleep", "7403.5"])
    try:
        with pytest.raises(ValueError, match="not started by this host"):
            standin.host.stop(
                RunningSandbox(SandboxPlan(Path("/nonexistent"), (), (), {}, Path("/nonexistent"), 0), foreign)
            )
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait()


def test_a_role_over_its_cap_is_refused_and_injected_failures_are_retried(e2b_envd: Path, tmp_path: Path) -> None:
    with StandIn(e2b_envd, tmp_path, [StandInTemplate("dev", "dev")], max_sandboxes={"dev": 1}) as capped:
        provider = _provider(capped)
        one = provider.create("dev", role="dev", timeout_sec=60, metadata={})
        with pytest.raises(SandboxUnavailable, match="already has 1"):
            provider.create("dev", role="dev", timeout_sec=60, metadata={})
        capped.plane.fail_next("kill", 503, times=2)
        one.kill()
        with pytest.raises(SandboxGone):
            one.info()
        capped.plane.fail_next("create", 500, times=2)
        again = provider.create("dev", role="dev", timeout_sec=60, metadata={IDEMPOTENCY_KEY: "retry-1"})
        assert again.info().sandbox_id == again.sandbox_id


def test_a_slow_create_outlasts_wait_sec_and_the_same_key_finds_it_later(e2b_envd: Path, tmp_path: Path) -> None:
    with StandIn(e2b_envd, tmp_path, [StandInTemplate("dev", "dev")]) as slow:
        provider = _provider(slow)
        slow.plane.create_delay_sec = 1.0
        meta = {IDEMPOTENCY_KEY: "slow-1"}
        with pytest.raises(SandboxUnavailable):
            provider.create("dev", role="dev", timeout_sec=60, metadata=meta, options=CreateOptions(wait_sec=0.3))
        slow.plane.create_delay_sec = 0.0
        sandbox = provider.create("dev", role="dev", timeout_sec=60, metadata=meta, options=CreateOptions(wait_sec=20))
        assert [i.sandbox_id for i in provider.find(meta, role="dev")] == [sandbox.sandbox_id]


def test_killing_a_sandbox_ends_everything_in_it_even_a_detached_child(standin: StandIn) -> None:
    sandbox = _provider(standin).create("dev", role="dev", timeout_sec=60, metadata={})
    sandbox.run("setsid sleep 7401.5 >/dev/null 2>&1 < /dev/null &")
    deadline = time.monotonic() + 5
    while not [p for p in _processes_matching("sleep\x007401.5") if p] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _processes_matching("sleep\x007401.5")
    sandbox.kill()
    deadline = time.monotonic() + 5
    while _processes_matching("sleep\x007401.5") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _processes_matching("sleep\x007401.5") == []


def test_the_router_answers_for_a_sandbox_that_does_not_exist_like_e2b(standin: StandIn) -> None:
    url = urlparse(standin.sandbox_url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    conn.request("GET", "/health", headers={"E2b-Sandbox-Id": "sbx-nope"})
    response = conn.getresponse()
    assert response.status == 502 and b"was not found" in response.read()
    conn.close()


def test_large_files_stream_through_the_router(standin: StandIn, tmp_path: Path) -> None:
    sandbox = _provider(standin).create("dev", role="dev", timeout_sec=120, metadata={})
    payload = os.urandom(1 << 20) * 24
    sandbox.write_file("/work/large.bin", payload)
    assert sandbox.copy_out("/work/large.bin", tmp_path / "large.bin") == len(payload)
    assert (tmp_path / "large.bin").read_bytes() == payload
    sandbox.kill()


def test_closing_waits_for_a_sandbox_still_starting_and_refuses_new_ones(e2b_envd: Path, tmp_path: Path) -> None:
    standin = StandIn(e2b_envd, tmp_path, [StandInTemplate("dev", "dev")])
    standin.plane.create_delay_sec = 1.0
    provider = _provider(standin)

    def create() -> None:
        with contextlib.suppress(SandboxError):
            provider.create("dev", role="dev", timeout_sec=60, metadata={})

    creating = threading.Thread(target=create)
    creating.start()
    assert _wait_until(lambda: standin.plane.starting() == 1)
    standin.close()
    creating.join()
    assert len(standin.host.pids) == 1 and standin.host.alive() == []
    with pytest.raises(ServiceError) as refused:
        standin.plane.create("dev", {"templateID": "dev"})
    assert refused.value.status == 503


def test_the_standin_ends_only_the_processes_it_started(e2b_envd: Path, tmp_path: Path) -> None:
    decoy = subprocess.Popen(["sleep", "7402.5"])
    try:
        with StandIn(e2b_envd, tmp_path, [StandInTemplate("dev", "dev")]) as private:
            provider = _provider(private)
            for _ in range(3):
                provider.create("dev", role="dev", timeout_sec=60, metadata={})
            pids = private.host.pids
            ports = [str(p.plan.port) for p in private.plane.live_sandboxes()]
            assert len(pids) == 3 and set(private.host.alive()) == set(pids)
        assert private.host.alive() == []
        assert not any(_processes_matching(f"-port\x00{port}") for port in ports)
        assert list(tmp_path.iterdir()) == []
        assert decoy.poll() is None
    finally:
        decoy.kill()
        decoy.wait()


def _wait_until(predicate: Callable[[], bool], seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_a_sandbox_ends_when_its_timeout_passes(standin: StandIn) -> None:
    provider = _provider(standin)
    sandbox = provider.create("dev", role="dev", timeout_sec=1, metadata={})

    def listed() -> bool:
        return sandbox.sandbox_id in {i.sandbox_id for i in provider.find({}, role="dev")}

    assert _wait_until(lambda: not listed())
    with pytest.raises(SandboxGone):
        sandbox.info()
