# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``ray start`` daemons must not stay in the caller's process group or hold its pipes.

``ray start`` returns while gcs_server, raylet and the dashboard keep running. Started from a background command in a
shell that tracks the job by its process group (an agent's sandbox bash), daemons left in that group make the job look
alive for as long as Ray lives, and a daemon that inherited the job's stdout/stderr pipe keeps the shell's output
stream open. Each launch site is exercised here with a stand-in ``ray`` whose ``start`` leaves a ``sleep`` behind that
inherits ray start's session, group and stdio exactly as a careless daemon would.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _ray_runtime as ray_runtime

INSTALL_SH = Path(__file__).resolve().parents[4] / "inference_optimizer" / "assets" / "install_kernel_tools.sh"
_STDERR_NOTE = "fake-ray-start: note on stderr"
_SERVING_SLOT_ARG = '--resources={"serving_slot": 1}'

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups and sessions are POSIX")


def _write_fake_ray(bin_dir: Path, state: Path) -> None:
    """Install a ``ray`` whose ``start`` leaves a stdio-inheriting ``sleep`` daemon and records its pid."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "ray"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            state={str(state)!r}
            case "$1" in
              status) [ -f "$state/up" ] ;;
              stop) rm -f "$state/up"; exit 0 ;;
              start)
                printf '%s\\n' "$@" > "$state/start.argv"
                sleep 60 &
                echo "$!" > "$state/daemon.pid"
                touch "$state/up"
                echo {_STDERR_NOTE!r} >&2
                exit "${{FAKE_RAY_START_RC:-0}}"
                ;;
              *) exit 2 ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)


def _daemon_pid(state: Path) -> int:
    path = state / "daemon.pid"
    assert path.is_file(), "the stand-in ray start never ran"
    return int(path.read_text(encoding="utf-8").strip())


def _kill_daemon(state: Path) -> None:
    """Kill only the sleep the stand-in recorded, never anything else."""
    path = state / "daemon.pid"
    if not path.is_file():
        return
    pid = int(path.read_text(encoding="utf-8").strip())
    # The daemon may already have exited; only a live one needs killing.
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _extract_shell_function(name: str) -> str:
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


#: The ``ray`` package the installer's own driver connect imports: ``init`` answers as ``FAKE_RAY_HEAD`` says, and
#: ``gilhold`` waits the way a raylet that never registers the driver makes Ray wait, in native code with the GIL held,
#: after starting a helper process.
_FAKE_RAY_PACKAGE = """
import ctypes, os, subprocess, sys


def _note(line):
    with open(os.environ["FAKE_RAY_LOG"], "a", encoding="utf-8") as log:
        log.write(line + "\\n")


def init(address=None, **_kwargs):
    _note(f"init {os.getpid()}")
    if os.environ["FAKE_RAY_HEAD"] == "gilhold":
        helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        _note(f"helper {helper.pid}")
        ctypes.PyDLL(None).sleep(3600)


def cluster_resources():
    return {"GPU": 1.0, "serving_slot": 1.0} if os.environ["FAKE_RAY_HEAD"] == "slot" else {"GPU": 1.0}


def shutdown():
    pass
"""
# The installer's bound on that connect in these tests, in seconds.
_DRIVER_CONNECT_BOUND_SEC = 2


def _driver_pids(state: Path) -> dict[str, int]:
    """The pids the installer's driver connect noted: its own (``init``) and its helper's."""
    path = state / "driver.log"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return {kind: int(pid) for kind, pid in (line.split() for line in lines)}


def _kill_driver_helper(state: Path) -> None:
    """Kill only the helper the stand-in driver recorded, and only while it is still that sleep."""
    pid = _driver_pids(state).get("helper")
    with contextlib.suppress(OSError):
        if pid is not None and b"time.sleep(3600)" in Path(f"/proc/{pid}/cmdline").read_bytes():
            os.kill(pid, signal.SIGKILL)


def _running(pid: int) -> bool:
    """Whether ``pid`` is a live process; a zombie is not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def _run_install_ensure_ray_started(tmp_path: Path, extra_env: dict[str, str] | None = None, head: str | None = None):
    """Run install.sh's real ``ensure_ray_started`` as a background job of its own group, like a sandbox shell does.

    With ``head`` set, a head is already running and the real ``ray_head_has_serving_slot`` asks it through a driver
    connect that :data:`_FAKE_RAY_PACKAGE` answers as ``head`` says (``slot``, ``noslot`` or ``gilhold``).

    Returns ``(proc, stdout, stderr, state)``; the stand-in daemon is left for the caller to inspect and kill.
    """
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    _write_fake_ray(bin_dir, state)
    slot_probe = "ray_head_has_serving_slot() { return 0; }"
    env = {k: v for k, v in os.environ.items() if k not in ("HL_RAY_HEAD_PORT", "RAY_NUM_GPUS")}
    if head is not None:
        (state / "up").touch()
        package = tmp_path / "driver" / "ray"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(_FAKE_RAY_PACKAGE, encoding="utf-8")
        # ``timeout`` runs the python3 on PATH, never a shell function.
        python3 = bin_dir / "python3"
        python3.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
        python3.chmod(0o755)
        slot_probe = _extract_shell_function("ray_head_has_serving_slot")
        env.update(
            PYTHONPATH=str(package.parent),
            FAKE_RAY_HEAD=head,
            FAKE_RAY_LOG=str(state / "driver.log"),
            HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC=str(_DRIVER_CONNECT_BOUND_SEC),
        )
    body = "\n".join(
        [
            "set -euo pipefail",
            "CHECK_ONLY=0; DRY_RUN=0; SKIP_RAY_START=0",
            'log() { echo "[kernel-agent] $*"; }',
            'warn() { echo "[kernel-agent WARN] $*" >&2; }',
            # torch / free-port probes: no python needed for this test. A driver connect goes to the python3 on
            # PATH, so one that lost its bound would still connect, and hang, rather than read as a stub's answer.
            'python3() { local src; src="$(cat)"; case "$src" in *"import ray"*) command python3 "$@" <<<"$src" ;; '
            "*) echo 0 ;; esac; }",
            "ensure_fd_limit_for_ray() { :; }",
            slot_probe,
            _extract_shell_function("_free_tcp_port"),
            _extract_shell_function("_ray_start_detached"),
            _extract_shell_function("ensure_ray_started"),
            "ensure_ray_started",
            'echo "installer-done"',
        ]
    )
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["TMPDIR"] = str(tmp_path)
    env.update(extra_env or {})
    proc = subprocess.Popen(
        ["bash", "-c", body],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        _kill_daemon(state)
        # The installer leads its own group: everything it still runs goes with it.
        os.killpg(proc.pid, signal.SIGKILL)
        _kill_driver_helper(state)
        stdout, stderr = proc.communicate()
        pytest.fail(
            "install.sh did not finish within 20s: a Ray daemon inherited the installer's pipe, or a driver connect "
            f"went unbounded (stdout={stdout!r}, stderr={stderr!r})"
        )
    return proc, stdout, stderr, state


def test_install_sh_ray_start_daemons_leave_the_installers_group_and_pipes(tmp_path):
    """The installer's output ends when it does, and the daemon is in another session and group."""
    proc, stdout, stderr, state = _run_install_ensure_ray_started(tmp_path)
    try:
        pid = _daemon_pid(state)
        assert proc.returncode == 0, stderr
        assert "installer-done" in stdout
        # The installer is its own group leader (start_new_session), so its pid is its group and session id.
        assert os.getpgid(pid) != proc.pid, "Ray daemon is still in the installer's process group"
        assert os.getsid(pid) != proc.pid, "Ray daemon is still in the installer's session"
        assert os.getsid(pid) == os.getpgid(pid), "Ray daemon should sit in the new session's own group"
        # ray start's own stderr is still relayed to the installer's stderr after it exits.
        assert _STDERR_NOTE in stderr
        # The arguments reach ray start unchanged.
        argv = (state / "start.argv").read_text(encoding="utf-8").splitlines()
        assert argv[:2] == ["start", "--head"]
        assert "--disable-usage-stats" in argv and "--include-dashboard=false" in argv
        assert _SERVING_SLOT_ARG in argv
        assert "--num-gpus=0" in argv
        # No temp file is left behind for the relayed stderr.
        assert not list(tmp_path.glob("hl-ray-start.*"))
    finally:
        _kill_daemon(state)


def test_install_sh_failed_ray_start_still_warns_and_relays_its_stderr(tmp_path):
    """A non-zero ray start exit status survives the detaching wrapper."""
    proc, stdout, stderr, state = _run_install_ensure_ray_started(tmp_path, {"FAKE_RAY_START_RC": "3"})
    try:
        assert proc.returncode == 0, stderr
        assert "ray start failed; kernel optimization will hang" in stderr
        assert _STDERR_NOTE in stderr
        assert "installer-done" in stdout
    finally:
        _kill_daemon(state)


# ---- install.sh's driver connect to a running head ------------------------------------------------------------------


def test_install_sh_restarts_a_head_that_never_answers_its_driver_within_the_bound(tmp_path):
    """A head whose raylet never registers a driver costs the bound, not the install, and is replaced like one
    without serving_slot; nothing the bounded connect started survives it."""
    started = time.monotonic()
    proc, stdout, stderr, state = _run_install_ensure_ray_started(tmp_path, head="gilhold")
    elapsed = time.monotonic() - started
    try:
        assert proc.returncode == 0, stderr
        assert "installer-done" in stdout
        assert f"ray head did not answer a driver within {_DRIVER_CONNECT_BOUND_SEC}s" in stderr
        assert "HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC" in stderr
        assert "without serving_slot" not in stderr
        assert _SERVING_SLOT_ARG in (state / "start.argv").read_text(encoding="utf-8").splitlines()
        # The bound, plus timeout's SIGKILL grace at most, plus start-up on a loaded host.
        assert elapsed < _DRIVER_CONNECT_BOUND_SEC + 5 + 10
        pids = _driver_pids(state)
        assert set(pids) == {"init", "helper"}, "the stand-in driver did not get as far as starting its helper"
        assert not [kind for kind, pid in pids.items() if _running(pid)], "the bounded connect left processes behind"
    finally:
        _kill_daemon(state)
        _kill_driver_helper(state)


@pytest.mark.parametrize(
    ("head", "restarted", "message"),
    [
        ("slot", False, "ray head already running with serving_slot"),
        ("noslot", True, "ray head already running without serving_slot; restarting local Ray head"),
    ],
)
def test_install_sh_driver_connect_that_answers_decides_on_the_slot_as_before(tmp_path, head, restarted, message):
    proc, stdout, stderr, state = _run_install_ensure_ray_started(tmp_path, head=head)
    try:
        assert proc.returncode == 0, stderr
        assert "installer-done" in stdout
        assert message in stdout + stderr
        assert "did not answer a driver" not in stderr
        assert (state / "start.argv").is_file() is restarted
    finally:
        _kill_daemon(state)


# ---- ray_runtime (orchestrator / kernel backends) ------------------------------------------------------------------


class _Proc:
    returncode = 0
    stdout = ""
    stderr = ""


class _FakeResource:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def getrlimit(self, which):
        return (1048576, 1048576)

    def setrlimit(self, which, limits):  # pragma: no cover - never reached at a high soft limit
        raise AssertionError("fd limit already high enough")


def _record_ray_start_kwargs(monkeypatch):
    seen: list[dict] = []
    started = False
    monkeypatch.setattr(ray_runtime, "ray_status_ok", lambda: started)
    monkeypatch.setattr(ray_runtime, "resource", _FakeResource(), raising=False)

    def _fake_run(cmd, **kwargs):
        nonlocal started
        if cmd[:2] == ["ray", "start"]:
            seen.append(kwargs)
            started = True
        return _Proc()

    monkeypatch.setattr(ray_runtime.subprocess, "run", _fake_run)
    return seen


def _assert_detached_kwargs(kwargs: dict) -> None:
    assert kwargs.get("start_new_session") is True, kwargs
    assert kwargs.get("stdin") is subprocess.DEVNULL, kwargs
    assert "capture_output" not in kwargs, kwargs
    assert kwargs.get("stdout") is not subprocess.PIPE, kwargs
    assert kwargs.get("stderr") is not subprocess.PIPE, kwargs


@pytest.mark.parametrize("entry", ["ensure_ray_cluster", "force_restart_local_cluster"])
def test_ray_runtime_launches_ray_start_detached(monkeypatch, entry):
    seen = _record_ray_start_kwargs(monkeypatch)

    getattr(ray_runtime, entry)(num_gpus=1)

    assert len(seen) == 1, seen
    _assert_detached_kwargs(seen[0])


def test_ensure_ray_cluster_real_daemon_leaves_caller_group_and_does_not_block(monkeypatch, tmp_path):
    """A real stand-in daemon ends up outside this process's group and session, and the call returns promptly."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    _write_fake_ray(bin_dir, state)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("HL_RAY_HEAD_PORT", raising=False)
    monkeypatch.setattr(ray_runtime, "resource", _FakeResource(), raising=False)
    errors: list[Exception] = []

    def _call():
        try:
            ray_runtime.ensure_ray_cluster(num_gpus=1)
        except Exception as exc:  # noqa: BLE001 - any failure is surfaced by the assertion below
            errors.append(exc)

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    try:
        worker.join(timeout=20)
        assert not worker.is_alive(), "ensure_ray_cluster blocked on a pipe the Ray daemon inherited"
        assert not errors, errors
        pid = _daemon_pid(state)
        assert os.getpgid(pid) != os.getpgid(0), "Ray daemon is still in the caller's process group"
        assert os.getsid(pid) != os.getsid(0), "Ray daemon is still in the caller's session"
    finally:
        _kill_daemon(state)
        worker.join(timeout=5)


def test_ensure_ray_cluster_reports_ray_start_output(monkeypatch, tmp_path):
    """The anonymous-file capture still carries a real failed ray start's output into the error."""
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    _write_fake_ray(bin_dir, state)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_RAY_START_RC", "5")
    monkeypatch.delenv("HL_RAY_HEAD_PORT", raising=False)
    monkeypatch.setattr(ray_runtime, "resource", _FakeResource(), raising=False)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            ray_runtime.ensure_ray_cluster(num_gpus=1)
        assert "rc=5" in str(excinfo.value)
        assert _STDERR_NOTE in str(excinfo.value)
    finally:
        _kill_daemon(state)
