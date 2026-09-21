# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for Ray version-mismatch recovery before GEAK dispatch."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import ray_runtime  # noqa: E402


_VERSION_MISMATCH_MSG = (
    "Version mismatch: The cluster was started with:\n"
    "    Ray: 2.44.1\n"
    "    Python: 3.10.12\n"
    "This process on node 192.0.2.63 was started with:\n"
    "    Ray: 2.44.1\n"
    "    Python: 3.12.13\n"
)


def _make_fake_ray(init_side_effects):
    """Build a stand-in ``ray`` module whose ``init`` pops side effects in order: a ``BaseException`` instance is raised, ``None`` succeeds."""
    fake = types.ModuleType("ray")
    calls = {"init": 0, "shutdown": 0}
    effects = list(init_side_effects)

    def _init(*args, **kwargs):
        calls["init"] += 1
        eff = effects.pop(0)
        if isinstance(eff, BaseException):
            raise eff
        return None

    def _shutdown(*args, **kwargs):
        calls["shutdown"] += 1

    fake.init = _init
    fake.shutdown = _shutdown
    fake._calls = calls
    return fake


@pytest.fixture(autouse=True)
def _clear_stale_connect_flag():
    """``_STALE_CONNECT_POSSIBLE`` is process-wide by design, so tests must isolate it.

    A timeout in one test otherwise makes the next one's first ``ray.init``
    clear a session it never left behind -- the coupling is real in production
    too, where it is exactly the intended behaviour across legs.
    """
    ray_runtime._STALE_CONNECT_POSSIBLE.clear()
    yield
    ray_runtime._STALE_CONNECT_POSSIBLE.clear()


def test_is_version_mismatch_detects_banner():
    assert ray_runtime._is_ray_version_mismatch(_VERSION_MISMATCH_MSG)
    assert ray_runtime._is_ray_version_mismatch("ray Version Mismatch: foo")
    assert not ray_runtime._is_ray_version_mismatch("ConnectionError: GCS down")
    assert not ray_runtime._is_ray_version_mismatch("")
    assert not ray_runtime._is_ray_version_mismatch(None)  # type: ignore[arg-type]


def test_quiet_ray_init_recovers_from_version_mismatch(tmp_path, monkeypatch):
    """First init raises Version mismatch -> restart local cluster -> retry OK."""
    fake_ray = _make_fake_ray([RuntimeError(_VERSION_MISMATCH_MSG), None])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    restart_calls = []

    def _fake_restart(num_gpus=None, log_path=None):
        restart_calls.append({"num_gpus": num_gpus, "log_path": log_path})

    monkeypatch.setattr(ray_runtime, "force_restart_local_cluster", _fake_restart)

    log_path = tmp_path / "ray_lifecycle.log"
    runtime_env = ray_runtime.quiet_ray_init(num_gpus=2, log_path=log_path)

    assert fake_ray._calls["init"] == 2, "should retry init exactly once after restart"
    assert fake_ray._calls["shutdown"] == 1, "should shutdown stale driver before retry"
    assert len(restart_calls) == 1
    assert restart_calls[0]["num_gpus"] == 2
    assert restart_calls[0]["log_path"] == log_path
    assert "env_vars" in runtime_env


def test_quiet_ray_init_propagates_non_mismatch_error(monkeypatch):
    """A non-version-mismatch failure must NOT trigger a restart; it raises."""
    fake_ray = _make_fake_ray([ConnectionError("GCS handshake failed")])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    restart_calls = []
    monkeypatch.setattr(
        ray_runtime,
        "force_restart_local_cluster",
        lambda **kw: restart_calls.append(kw),
    )

    with pytest.raises(ConnectionError):
        ray_runtime.quiet_ray_init(num_gpus=1)

    assert fake_ray._calls["init"] == 1, "must not retry on non-mismatch errors"
    assert restart_calls == [], "must not restart cluster on non-mismatch errors"


def test_quiet_ray_init_no_mismatch_succeeds_first_try(monkeypatch):
    """Happy path: init succeeds immediately, no restart, no retry."""
    fake_ray = _make_fake_ray([None])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    restart_calls = []
    monkeypatch.setattr(
        ray_runtime,
        "force_restart_local_cluster",
        lambda **kw: restart_calls.append(kw),
    )
    ray_runtime.quiet_ray_init()
    assert fake_ray._calls["init"] == 1
    assert restart_calls == []


def test_force_restart_local_cluster_runs_stop_then_start(tmp_path):
    """``force_restart_local_cluster`` must ``ray stop --force`` then ``ray start --head`` with the requested num_gpus, logging to the audit file."""
    log_path = tmp_path / "ray_lifecycle.log"
    runs = []

    class _Proc:
        returncode = 0

    def _fake_run(cmd, **kwargs):
        runs.append(cmd)
        return _Proc()

    with mock.patch.object(ray_runtime.subprocess, "run", _fake_run):
        ray_runtime.force_restart_local_cluster(num_gpus=4, log_path=log_path)

    assert runs[0] == ["ray", "stop", "--force"]
    # The fresh head binds a probed free port (not the fixed 6379) so co-located host-network sessions never collide
    # on Ray's default GCS port.
    assert runs[1][:3] == ["ray", "start", "--head"]
    assert any(tok.startswith("--port=") for tok in runs[1])
    assert "--num-gpus=4" in runs[1]
    assert log_path.exists()


def test_force_restart_raises_when_start_fails(tmp_path):
    """A non-zero ``ray start`` exit must raise so ``submit``'s except can record it as a backend-dispatch failure."""
    log_path = tmp_path / "ray_lifecycle.log"

    class _Proc:
        returncode = 1

    def _fake_run(cmd, **kwargs):
        return _Proc()

    with mock.patch.object(ray_runtime.subprocess, "run", _fake_run):
        with pytest.raises(RuntimeError, match="restart local Ray"):
            ray_runtime.force_restart_local_cluster(num_gpus=1, log_path=log_path)


# ---- ray.init is bounded -----------------------------------------------------


def test_quiet_ray_init_times_out_instead_of_hanging(monkeypatch):
    """A raylet that accepts the socket and never replies must not block forever.

    Observed live: a coordinator sat inside this call for 44 minutes holding
    tick 1, with no child process, no server log and no failure to read -- the
    stall was indistinguishable from a slow model load.
    """
    import threading as _threading

    release = _threading.Event()

    class _WedgedRay:
        def init(self, **_kwargs):
            release.wait(30)  # never released within the test's timeout

        def shutdown(self):
            pass

    monkeypatch.setattr(ray_runtime, "safe_runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setitem(__import__("sys").modules, "ray", _WedgedRay())
    monkeypatch.setenv("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", "0.5")

    with pytest.raises(TimeoutError, match="did not complete within"):
        ray_runtime.quiet_ray_init(num_gpus=1)
    release.set()


def test_quiet_ray_init_timeout_is_configurable(monkeypatch):
    """The bound is a knob, so a genuinely slow cold start can be waited out."""
    monkeypatch.delenv("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", raising=False)
    assert ray_runtime._ray_init_timeout_sec() == ray_runtime.DEFAULT_RAY_INIT_TIMEOUT_SEC
    monkeypatch.setenv("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", "12.5")
    assert ray_runtime._ray_init_timeout_sec() == 12.5


def test_a_timed_out_init_leaves_stdout_usable(monkeypatch, capsys):
    """The abandoned thread must not take the process's stdout with it.

    ``sys.stdout`` is process-global. A redirection held on the thread the caller
    abandons would never unwind, so every later write from every thread would
    disappear into a buffer nobody reads -- trading a bounded stall for a
    permanent one.
    """
    import threading as _threading

    release = _threading.Event()

    class _WedgedRay:
        def init(self, **_kwargs):
            release.wait(30)

        def shutdown(self):
            pass

    monkeypatch.setattr(ray_runtime, "safe_runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setitem(sys.modules, "ray", _WedgedRay())
    monkeypatch.setenv("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", "0.5")

    with pytest.raises(TimeoutError):
        ray_runtime.quiet_ray_init(num_gpus=1)

    # The wedged connect is still alive; the caller's stream must still work.
    assert _threading.active_count() >= 1
    print("visible-after-timeout")
    assert "visible-after-timeout" in capsys.readouterr().out
    release.set()


def test_an_abandoned_runner_never_tears_down_a_session(monkeypatch):
    """The late thread is no longer the owner, so it must not clean up.

    ``ignore_reinit_error=True`` means a late ``ray.init`` on an
    already-connected process is a no-op that attaches to whatever session is
    current. A shutdown from that thread would therefore tear down a LATER
    leg's working connection rather than its own -- a cross-leg teardown in a
    long-lived coordinator, which is worse than the stale session it was meant
    to clear.
    """
    import threading
    import types

    release = threading.Event()
    shutdowns = []

    def _slow_init(**_kw):
        release.wait(5)

    fake_ray = types.SimpleNamespace(
        init=_slow_init, shutdown=lambda: shutdowns.append(1), is_initialized=lambda: False
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", "0.2")

    with pytest.raises(TimeoutError):
        ray_runtime.quiet_ray_init(num_gpus=1)

    release.set()
    threading.Event().wait(0.3)
    assert shutdowns == [], "the abandoned runner must not shut anything down"


def test_a_second_attempt_waits_out_an_unresolved_first(monkeypatch):
    """Two connects may not overlap, because ``ray.init`` is process-global.

    The interleaving that matters: the second attempt starts while the first
    runner is still inside ``ray.init``. If it were allowed to run, its shutdown
    would happen before that runner connects, the connect would land in the gap,
    and its own ``ray.init`` -- ``ignore_reinit_error=True`` -- would silently
    attach to the OLD cluster. Which is exactly what the version-mismatch retry
    exists to escape.
    """
    import threading
    import types

    release = threading.Event()
    order = []

    def _init(**_kw):
        order.append("init")
        if len(order) == 1:
            release.wait(5)

    fake_ray = types.SimpleNamespace(
        init=_init, shutdown=lambda: order.append("shutdown"), is_initialized=lambda: False
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("HYPERLOOM_RAY_INIT_TIMEOUT_SEC", "0.2")

    with pytest.raises(TimeoutError):
        ray_runtime.quiet_ray_init(num_gpus=1)

    # The first runner is still blocked, so the gate is still held and the second
    # attempt cannot even begin its connect.
    second: dict = {}

    def _second():
        try:
            ray_runtime.quiet_ray_init(num_gpus=1)
            second["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            second["error"] = exc

    t = threading.Thread(target=_second, daemon=True)
    t.start()
    t.join(5)
    assert order == ["init"], f"the second attempt started while the first was unresolved: {order}"
    assert isinstance(second.get("error"), TimeoutError), second
    assert "still outstanding" in str(second["error"])

    # Once the first resolves, the gate frees and a later attempt proceeds --
    # and its cleanup now runs AFTER that late connect rather than before it.
    release.set()
    for _ in range(100):
        if ray_runtime._INIT_GATE.acquire(blocking=False):
            ray_runtime._INIT_GATE.release()
            break
        threading.Event().wait(0.05)
    ray_runtime.quiet_ray_init(num_gpus=1)
    assert order[1:] == ["shutdown", "init"], order
