# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The loopback Experience service is started on demand, reused, and verified before a client writes to it."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from hyperloom_kb import (
    PACKAGED_DECLARATION,
    ExperienceDeclaration,
    ExperienceHTTPService,
    FieldDeclaration,
    HTTPServiceConfig,
    LocalServiceError,
    ObjectiveDeclaration,
    ObjectiveDirection,
    RemoteConfig,
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


def test_a_service_with_another_declaration_is_refused(tmp_path: Path) -> None:
    other = ExperienceDeclaration(
        identity=(FieldDeclaration("model", "Model."),),
        baseline_identity=(FieldDeclaration("config", "Config."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(ObjectiveDeclaration("throughput@v1", ObjectiveDirection.HIGHER_IS_BETTER, "T."),),
        decisions=("keep", "revert", "failed"),
    )
    with _serving(tmp_path, other, TOKEN) as port:
        with pytest.raises(LocalServiceError, match="stop that service"):
            ensure_local_service(_config(port, tmp_path), tmp_path / "home", env=_env_without_planner_gateway(tmp_path))


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
