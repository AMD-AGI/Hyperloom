# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The loopback Experience service is started on demand, reused, and verified before a client writes to it."""

from __future__ import annotations

import os
import shutil
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
from hyperloom_kb import local_service
from hyperloom_kb.http_service import code_digest
from hyperloom_kb.tests.conftest import fresh_database

# Spawned services run their own embedded database under ``tmp_path``.
pytestmark = pytest.mark.usefixtures("reachable_tmp_path")

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
    app = ExperienceHTTPService(
        HTTPServiceConfig(tmp_path / "other-service", token), declaration, None, database=fresh_database()
    )
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


def test_a_log_past_its_size_is_kept_once_and_the_next_service_starts_a_new_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(local_service, "LOG_ROTATE_BYTES", 64)
    home = tmp_path / "home"
    home.mkdir()
    old = "an earlier service's log line\n" * 4
    (home / "service.log").write_text(old, encoding="utf-8")
    started = ensure_local_service(_config(_free_port(), tmp_path), home, env=_env_without_planner_gateway(tmp_path))
    assert started.process is not None
    try:
        kept = (home / "service.log.1").read_text(encoding="utf-8")
        current = (home / "service.log").read_text(encoding="utf-8")
    finally:
        started.process.terminate()
        started.process.wait(timeout=10)

    assert kept == old
    assert "an earlier service" not in current and '"event": "listening"' in current


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
            ensure_local_service(
                _config(port, tmp_path), tmp_path / "other-service", env=_env_without_planner_gateway(tmp_path)
            )


def test_a_service_holding_another_workspaces_data_is_refused_and_left_serving(tmp_path: Path) -> None:
    with _serving(tmp_path, load_declaration(PACKAGED_DECLARATION), TOKEN) as port:
        with pytest.raises(LocalServiceError, match="give this workspace its own port"):
            ensure_local_service(_config(port, tmp_path), tmp_path / "home", env=_env_without_planner_gateway(tmp_path))
        assert RemoteClient(_config(port, tmp_path)).health()["status"] == "ok"
    assert not (tmp_path / "home").exists()


def test_a_caller_that_must_not_restart_gets_the_stale_service_as_it_runs(tmp_path: Path) -> None:
    config = _config(_free_port(), tmp_path)
    home = tmp_path / "home"
    before = _env_without_planner_gateway(tmp_path)
    after = {**before, "HYPERLOOM_GLOBAL_KB_URL": "https://global.example", "HYPERLOOM_GLOBAL_KB_TOKEN": "global"}
    first = ensure_local_service(config, home, env=before)
    try:
        kept = ensure_local_service(config, home, env=after, restart=False)

        assert (kept.process, kept.restarted, kept.stale) == (None, False, "it was started with other settings")
        assert first.process is not None and first.process.poll() is None
        assert kept.health["pid"] == first.process.pid
    finally:
        _stop(first)


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
        + ["--home", str(tmp_path / "home"), "--port", str(port)],
        env={**env, "HYPERLOOM_KB_TOKEN": TOKEN, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
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


def test_a_service_started_from_other_code_is_restarted_by_a_launch_and_kept_by_a_side_command(
    tmp_path: Path,
) -> None:
    older_code = tmp_path / "older"
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        older_code / "hyperloom_kb",
        ignore=shutil.ignore_patterns("tests", "__pycache__"),
    )
    with (older_code / "hyperloom_kb" / "remote.py").open("a", encoding="utf-8") as source:
        source.write("\n# as released before an upgrade\n")
    port = _free_port()
    env = _env_without_planner_gateway(tmp_path)
    older = subprocess.Popen(
        [sys.executable, "-m", "hyperloom_kb", "--home", str(tmp_path / "home"), "--port", str(port)],
        env={**env, "HYPERLOOM_KB_TOKEN": TOKEN, "PYTHONPATH": str(older_code)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    current = LocalService({})
    try:
        while not _reachable(port):
            assert older.poll() is None
        kept = ensure_local_service(_config(port, tmp_path), tmp_path / "home", env=env, restart=False)
        assert (kept.stale, older.poll()) == ("it runs other Experience KB code than this client", None)

        current = ensure_local_service(_config(port, tmp_path), tmp_path / "home", env=env)

        assert older.wait(timeout=10) is not None
        assert current.restarted and current.health["code_digest"] == code_digest()
    finally:
        older.kill()
        older.wait(timeout=10)
        _stop(current)


def test_a_data_home_another_service_serves_is_never_served_twice(tmp_path: Path) -> None:
    home = tmp_path / "shared-home"
    env = _env_without_planner_gateway(tmp_path)
    first_port = _free_port()
    first = ensure_local_service(_config(first_port, tmp_path), home, env=env)
    try:
        other = RemoteConfig(f"http://127.0.0.1:{_free_port()}", "other-workspace", timeout_seconds=5)
        with pytest.raises(
            LocalServiceError, match=rf"another Experience service \(pid \d+, port {first_port}\) already"
        ):
            ensure_local_service(other, home, env=env)

        assert first.process is not None and first.process.poll() is None
        assert RemoteClient(_config(first_port, tmp_path)).health()["status"] == "ok"
    finally:
        _stop(first)


def test_a_service_that_cannot_start_is_reported_with_its_log(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "postgres").write_text("not a database directory", encoding="utf-8")

    with pytest.raises(LocalServiceError, match="exited with status"):
        ensure_local_service(_config(_free_port(), tmp_path), home, env=_env_without_planner_gateway(tmp_path))

    assert "Traceback" in (home / "service.log").read_text(encoding="utf-8")


def test_a_home_that_cannot_hold_the_service_is_a_local_service_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.write_text("not a directory", encoding="utf-8")

    with pytest.raises(LocalServiceError, match="cannot start the Experience service"):
        ensure_local_service(_config(_free_port(), tmp_path), home, env=_env_without_planner_gateway(tmp_path))


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
