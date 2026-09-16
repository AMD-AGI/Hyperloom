# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-host port isolation for single-round serving benchmarks."""

from __future__ import annotations

import errno
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import pytest
import yaml

from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors import _multi_node_env, _ray_backend, _server_lifecycle
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
)


@pytest.fixture(autouse=True)
def local_execution(monkeypatch):
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    for key in ("BENCHMARK_BASE_URL", "MAGPIE_RUN_PHASE", "SERVER_REUSE", "FORCE_SERVER_REUSE", "EXTRA_SGLANG_ARGS"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def occupied_default_port():
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", 8888))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                pytest.skip("port 8888 is already occupied outside this test")
            raise
        listener.listen()
        yield


def _configs(tmp_path: Path, *, framework="sglang", profile=True, overrides=None) -> tuple[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bench = {
        "framework": framework,
        "benchmark_script": f"{framework}_mi355x.sh",
        "model": "/models/example",
        "envs": {"PORT": 8888, "ROCR_VISIBLE_DEVICES": "7", "TP": 1},
        "profiler": {"torch_profiler": {"enabled": profile}},
    }
    bench.update(overrides or {})
    original = tmp_path / "materialized.yaml"
    original.write_text(yaml.safe_dump({"benchmark": bench}))
    launch = _ray_backend.strip_visible_devices_from_config(original)
    return str(launch), str(original)


@pytest.mark.parametrize("backend", ["magpie", "bypass"])
@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
@pytest.mark.parametrize("profile", [False, True])
def test_local_server_and_recipe_change_only_port(tmp_path, backend, framework, profile, occupied_default_port):
    paths = _configs(tmp_path, framework=framework, profile=profile)
    before = [yaml.safe_load(Path(path).read_text()) for path in paths]
    env = {"HYPERLOOM_BENCHMARK_BACKEND": backend, "PORT": "8888", "ROCR_VISIBLE_DEVICES": "2"}

    _server_lifecycle.prepare_single_round_port(paths, env)

    port = int(env["PORT"])
    assert port != 8888
    assert env["ROCR_VISIBLE_DEVICES"] == "2"
    for path, expected in zip(paths, before):
        expected["benchmark"]["envs"]["PORT"] = port
        assert yaml.safe_load(Path(path).read_text()) == expected
    assert "ROCR_VISIBLE_DEVICES" not in yaml.safe_load(Path(paths[0]).read_text())["benchmark"]["envs"]
    assert yaml.safe_load(Path(paths[1]).read_text())["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "7"


@pytest.mark.parametrize(
    "overrides,extra_env,multi_node",
    [
        ({"benchmark_script": "custom.sh"}, {}, False),
        ({"benchmark_script": "/custom/sglang_mi355x.sh"}, {}, False),
        ({"workload_kind": "scriptable"}, {}, False),
        ({"framework": "xdit"}, {"HYPERLOOM_BENCHMARK_BACKEND": "bypass"}, False),
        ({"server_lifecycle": {"enabled": True, "cleanup": False}}, {}, False),
        ({}, {"MAGPIE_RUN_PHASE": "client"}, False),
        ({"envs": {"MAGPIE_RUN_PHASE": "server", "PORT": 8888}}, {}, False),
        ({}, {"BENCHMARK_BASE_URL": "http://remote.example:8888"}, False),
        ({"envs": {"BENCHMARK_BASE_URL": "http://remote.example:8888"}}, {}, False),
        ({}, {"SERVER_REUSE": "1"}, False),
        ({"envs": {"FORCE_SERVER_REUSE": "true"}}, {}, False),
        ({"envs": {"EXTRA_SGLANG_ARGS": "--port 9123"}}, {}, False),
        ({}, {"EXTRA_SGLANG_ARGS": "--port=9123"}, False),
        ({}, {"EXTRA_SGLANG_ARGS": "--host remote.example"}, False),
        ({}, {}, True),
    ],
)
def test_unowned_endpoints_are_untouched(tmp_path, monkeypatch, overrides, extra_env, multi_node):
    paths = _configs(tmp_path, overrides=overrides)
    before = [Path(path).read_bytes() for path in paths]
    env = {"HYPERLOOM_BENCHMARK_BACKEND": "magpie", "PORT": "8888", **extra_env}
    before_env = dict(env)
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: multi_node)
    monkeypatch.setattr(_server_lifecycle, "_pick_free_port", lambda: pytest.fail("unowned endpoint was allocated"))

    _server_lifecycle.prepare_single_round_port(paths, env)

    assert [Path(path).read_bytes() for path in paths] == before
    assert env == before_env


def test_bypass_owns_server_without_a_magpie_script(tmp_path, monkeypatch):
    paths = _configs(tmp_path, overrides={"benchmark_script": "ignored-by-bypass.sh"})
    env = {"HYPERLOOM_BENCHMARK_BACKEND": "bypass", "PORT": "8888"}
    monkeypatch.setattr(_server_lifecycle, "_pick_free_port", lambda: 41001)
    _server_lifecycle.prepare_single_round_port(paths, env)
    assert env["PORT"] == "41001"


def test_port_allocation_failure_does_not_fall_back_or_mutate(tmp_path, monkeypatch):
    paths = _configs(tmp_path)
    before = [Path(path).read_bytes() for path in paths]
    env = {"PORT": "8888"}

    def unavailable():
        raise OSError("no available local port")

    monkeypatch.setattr(_server_lifecycle, "_pick_free_port", unavailable)
    with pytest.raises(OSError, match="no available local port"):
        _server_lifecycle.prepare_single_round_port(paths, env)
    assert [Path(path).read_bytes() for path in paths] == before
    assert env == {"PORT": "8888"}


@pytest.mark.parametrize("stop", ["cancelled", "expired"])
def test_stopped_round_does_not_read_configs_or_allocate(monkeypatch, stop):
    monkeypatch.setattr(_server_lifecycle, "_pick_free_port", lambda: pytest.fail("stopped round allocated a port"))
    scope = CancelScope()
    if stop == "cancelled":
        scope.cancel(reason="test cancellation")
    env = {"PORT": "8888"}
    with use_cancel_scope(scope):
        _server_lifecycle.prepare_single_round_port(
            ("missing-launch-config", "missing-materialized-config"),
            env,
            session_deadline_sec=time.monotonic() - 1 if stop == "expired" else None,
        )
    assert env == {"PORT": "8888"}


def _worker(cmd, paths, *, remaining=30, ready=False):
    return _ray_backend._run_subprocess_worker(
        cmd=cmd,
        env={"HYPERLOOM_BENCHMARK_BACKEND": "bypass", "PORT": "8888", "ROCR_VISIBLE_DEVICES": "99"},
        cwd=None,
        timeout_s=20,
        soft_deadline_sec=None,
        server_log_path=None,
        server_already_ready=ready,
        session_remaining_sec=remaining,
        single_round_configs=paths,
    )


@pytest.mark.parametrize("stop", ["cancelled", "expired"])
def test_worker_preserves_stop_verdict_without_port_preparation(tmp_path, monkeypatch, stop):
    paths = _configs(tmp_path)
    before = [Path(path).read_bytes() for path in paths]
    monkeypatch.setattr(_server_lifecycle, "_pick_free_port", lambda: pytest.fail("stopped worker allocated a port"))
    scope = CancelScope()
    if stop == "cancelled":
        scope.cancel(reason="test cancellation")
    with use_cancel_scope(scope):
        rc, _, _ = _worker(
            [sys.executable, "-c", "import time; time.sleep(30)"], paths, remaining=-1 if stop == "expired" else 30
        )
    assert rc == (SESSION_TIME_EXHAUSTED_RETURNCODE if stop == "expired" else ORCHESTRATOR_CANCELLED_RETURNCODE)
    assert [Path(path).read_bytes() for path in paths] == before


def test_worker_does_not_retarget_a_ready_server(tmp_path, monkeypatch):
    paths = _configs(tmp_path)
    before = [Path(path).read_bytes() for path in paths]
    monkeypatch.setattr(_server_lifecycle, "_pick_free_port", lambda: pytest.fail("ready server was retargeted"))
    rc, _, _ = _worker([sys.executable, "-c", "pass"], paths, ready=True)
    assert rc == 0
    assert [Path(path).read_bytes() for path in paths] == before


_HTTP_CHILD = """
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import yaml
config, ready, stop, marker = sys.argv[1:]
port = int(os.environ['PORT'])
assert yaml.safe_load(Path(config).read_text())['benchmark']['envs']['PORT'] == port
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(marker.encode())
    def log_message(self, *args):
        pass
server = HTTPServer(('127.0.0.1', port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
Path(ready).write_text(json.dumps({'port': port, 'mask': os.environ.get('ROCR_VISIBLE_DEVICES')}))
deadline = time.monotonic() + 15
while not Path(stop).exists() and time.monotonic() < deadline:
    time.sleep(.02)
server.shutdown()
server.server_close()
"""


def test_two_actual_worker_endpoints_avoid_occupied_default(tmp_path, monkeypatch, occupied_default_port):
    monkeypatch.setenv("PORT", "8888")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    futures = []
    stops = []
    endpoints = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            for marker in ("first", "second"):
                root = tmp_path / marker
                paths = _configs(root)
                ready, stop = root / "ready.json", root / "stop"
                stops.append(stop)
                future = pool.submit(
                    _worker, [sys.executable, "-c", _HTTP_CHILD, paths[0], str(ready), str(stop), marker], paths
                )
                futures.append(future)
                deadline = time.monotonic() + 10
                while not ready.exists() and not future.done() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert ready.exists(), future.result(timeout=1) if future.done() else "HTTP worker did not start"
                observed = json.loads(ready.read_text())
                port = observed["port"]
                assert observed["mask"] == "2"
                assert port != 8888
                endpoints.append(port)
                for path in paths:
                    assert yaml.safe_load(Path(path).read_text())["benchmark"]["envs"]["PORT"] == port
            assert len(set(endpoints)) == 2
            for port, marker in zip(endpoints, ("first", "second")):
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    assert response.read().decode() == marker
        finally:
            for stop in stops:
                stop.touch()
            for future in futures:
                assert future.result(timeout=10)[0] == 0
