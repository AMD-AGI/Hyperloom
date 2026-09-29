# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The loopback Experience service is started on demand, reused, and verified before a client writes to it."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import hyperloom_kb
from hyperloom_kb import (
    PACKAGED_DECLARATION,
    ExperienceDeclaration,
    ExperienceHTTPService,
    FieldDeclaration,
    HTTPServiceConfig,
    LocalService,
    LocalServiceError,
    ObjectiveDeclaration,
    ObjectiveDirection,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    ServiceSettings,
    create_http_server,
    ensure_local_service,
    is_loopback,
    load_declaration,
)

TOKEN = "local-service-token"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _reachable(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        time.sleep(0.1)
        return False


def _config(port: int, tmp_path: Path) -> RemoteConfig:
    return RemoteConfig(f"http://127.0.0.1:{port}", TOKEN, timeout_seconds=5, spool_root=tmp_path / "spool")


def _env_without_planner_gateway(tmp_path: Path) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}


@contextmanager
def _serving(tmp_path: Path, declaration: ExperienceDeclaration, token: str) -> Iterator[int]:
    app = ExperienceHTTPService(HTTPServiceConfig(tmp_path / "other-service", token), declaration, None)
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_missing_service_is_started_once_and_then_reused(tmp_path: Path) -> None:
    config = _config(_free_port(), tmp_path)
    home = tmp_path / "home"
    started = ensure_local_service(config, home, env=_env_without_planner_gateway(tmp_path))
    assert started.process is not None
    try:
        assert started.health["status"] == "ok"
        assert started.health["schema_ref"] == load_declaration(PACKAGED_DECLARATION).schema_ref
        assert "reads are unavailable" in (home / "service.log").read_text(encoding="utf-8")

        reused = ensure_local_service(config, home, env=_env_without_planner_gateway(tmp_path))

        assert reused.process is None
        assert reused.health["schema_ref"] == started.health["schema_ref"]
    finally:
        started.process.terminate()
        started.process.wait(timeout=10)


def test_a_listener_with_another_token_is_refused_without_starting_a_service(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _serving(tmp_path, load_declaration(PACKAGED_DECLARATION), "another-workspace") as port:
        with pytest.raises(LocalServiceError, match="did not answer as this Experience service"):
            ensure_local_service(_config(port, tmp_path), home, env=_env_without_planner_gateway(tmp_path))
    assert not home.exists()


def _other_declaration() -> ExperienceDeclaration:
    return ExperienceDeclaration(
        identity=(FieldDeclaration("model", "Model."),),
        baseline_identity=(FieldDeclaration("config", "Config."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(ObjectiveDeclaration("throughput@v1", ObjectiveDirection.HIGHER_IS_BETTER, "T."),),
        decisions=("keep", "revert", "failed"),
    )


def test_a_loopback_service_is_reached_directly_despite_an_environment_proxy(monkeypatch, tmp_path: Path) -> None:
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{_free_port()}")

    with _serving(tmp_path, load_declaration(PACKAGED_DECLARATION), TOKEN) as port:
        # ``urlopen`` caches its first opener; a fresh one reads the proxy the way a process started with it does.
        with pytest.raises(RemoteClientError):
            RemoteClient(_config(port, tmp_path), opener=urllib.request.build_opener().open).health()
        assert RemoteClient(_config(port, tmp_path)).health()["status"] == "ok"


def test_a_stale_service_inside_this_process_is_refused_rather_than_signalled(tmp_path: Path) -> None:
    with _serving(tmp_path, _other_declaration(), TOKEN) as port:
        with pytest.raises(LocalServiceError, match="no process to restart"):
            ensure_local_service(_config(port, tmp_path), tmp_path / "home", env=_env_without_planner_gateway(tmp_path))


def _stop(service: LocalService) -> None:
    if service.process is not None:
        service.process.terminate()
        service.process.wait(timeout=10)


def test_a_service_started_with_other_settings_is_restarted_with_the_launch_settings(tmp_path: Path) -> None:
    config = _config(_free_port(), tmp_path)
    home = tmp_path / "home"
    before = _env_without_planner_gateway(tmp_path)
    after = {**before, "HYPERLOOM_GLOBAL_KB_URL": "https://global.example", "HYPERLOOM_GLOBAL_KB_TOKEN": "global"}
    first = ensure_local_service(config, home, env=before)
    second = LocalService({})
    try:
        second = ensure_local_service(config, home, env=after)

        assert first.process is not None and first.process.wait(timeout=10) is not None
        assert second.restarted and second.process is not None
        assert second.health["pid"] == second.process.pid != first.process.pid
        assert second.health["config_digest"] == ServiceSettings.from_env(after).digest()
    finally:
        _stop(first)
        _stop(second)


def test_the_same_settings_spelled_differently_reuse_the_running_service(tmp_path: Path) -> None:
    config = _config(_free_port(), tmp_path)
    gateway = {"ANTHROPIC_BASE_URL": "https://gateway.example", "ANTHROPIC_API_KEY": "key", "CLAUDE_MODEL": "m"}
    launched = {**_env_without_planner_gateway(tmp_path), **gateway}
    aliased = {**launched, "ANTHROPIC_AUTH_TOKEN": "key", "ANTHROPIC_MODEL": "m"}
    first = ensure_local_service(config, tmp_path / "home", env=launched)
    try:
        assert ensure_local_service(config, tmp_path / "home", env=aliased).process is None
    finally:
        _stop(first)


def test_a_service_serving_an_older_declaration_is_restarted_with_the_packaged_one(tmp_path: Path) -> None:
    port = _free_port()
    declaration = tmp_path / "older.yaml"
    declaration.write_text(yaml.safe_dump(_other_declaration().to_dict(), sort_keys=False), encoding="utf-8")
    env = _env_without_planner_gateway(tmp_path)
    older = subprocess.Popen(
        [sys.executable, "-m", "hyperloom_kb", "--declaration", str(declaration)]
        + ["--home", str(tmp_path / "older"), "--port", str(port)],
        env={**env, "HYPERLOOM_KB_TOKEN": TOKEN, "PYTHONPATH": str(Path(hyperloom_kb.__file__).parent.parent)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    current = LocalService({})
    try:
        while not _reachable(port):
            assert older.poll() is None
        current = ensure_local_service(_config(port, tmp_path), tmp_path / "home", env=env)

        assert older.wait(timeout=10) is not None
        assert current.restarted
        assert current.health["schema_ref"] == load_declaration(PACKAGED_DECLARATION).schema_ref
    finally:
        older.kill()
        older.wait(timeout=10)
        _stop(current)


def test_a_service_that_cannot_start_is_reported_with_its_log(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "kb.sqlite3").mkdir(parents=True)

    with pytest.raises(LocalServiceError, match="exited with status"):
        ensure_local_service(_config(_free_port(), tmp_path), home, env=_env_without_planner_gateway(tmp_path))

    assert "Traceback" in (home / "service.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("url", ["http://kb.example:8787", "https://127.0.0.1:8787"])
def test_only_a_plain_http_loopback_url_names_a_local_service(tmp_path: Path, url: str) -> None:
    with pytest.raises(LocalServiceError, match="loopback"):
        ensure_local_service(RemoteConfig(url, TOKEN), tmp_path / "home")
    assert not (tmp_path / "home").exists()


def test_loopback_hosts() -> None:
    assert is_loopback("http://127.0.0.1:8787")
    assert is_loopback("http://localhost:8787")
    assert is_loopback("http://[::1]:8787")
    assert not is_loopback("http://10.235.192.67:8787")
    assert not is_loopback("http://kb.example:8787")
