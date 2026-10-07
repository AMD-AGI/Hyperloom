# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A service reports its health and metrics without a token, logs every request it answers with the request's id and
client, records which KB every write came from, and drains the requests in flight when it is stopped."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import (
    ExperienceHTTPService,
    HTTPServiceConfig,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    file_ref,
)
from hyperloom_kb.database import Database, PostgresDatabase, SqliteDatabase
from hyperloom_kb.http_service import create_http_server, serve_until_stopped
from hyperloom_kb.observability import JsonLogFormatter, event
from hyperloom_kb.tests.conftest import fresh_database
from hyperloom_kb.tests.database_fixtures import TEST_DATABASE_URL_ENV
from hyperloom_kb.tests.test_http_service import TOKEN, RunningServer, _declaration, _experience, _http

SCHEMA = _declaration()


def _lose(database: Database) -> None:
    """Take the database away from a service still serving it: its file is removed, or its server drops it."""

    if isinstance(database, SqliteDatabase):
        for suffix in ("", "-wal", "-shm"):
            database.path.with_name(database.path.name + suffix).unlink(missing_ok=True)
        return
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    assert isinstance(database, PostgresDatabase)
    with psycopg.connect(os.environ[TEST_DATABASE_URL_ENV], autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{conninfo_to_dict(database.conninfo)["dbname"]}" WITH (FORCE)')


def _service(home: Path, *, global_url: str | None = None, name: str = "") -> ExperienceHTTPService:
    global_kb = None if global_url is None else RemoteClient(RemoteConfig(global_url, TOKEN))
    return ExperienceHTTPService(
        HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=fresh_database(), global_kb=global_kb, name=name
    )


def _get(url: str, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(f"{url}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, dict(exc.headers), exc.read()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _metrics(text: str) -> dict[str, float]:
    return {
        series: float(value)
        for series, value in (line.rsplit(" ", 1) for line in text.splitlines() if line and not line.startswith("#"))
    }


def test_probes_answer_without_a_token_and_name_only_their_checks(tmp_path: Path) -> None:
    app = _service(tmp_path / "home")
    with RunningServer(app) as url:
        live = _get(url, "/livez")
        ready = _get(url, "/readyz")
        _lose(app._database)
        unready = _get(url, "/readyz")

    assert (live[0], json.loads(live[2])) == (200, {"status": "alive"})
    checks = {"database": "ok", "home_writable": "ok", "disk_space": "ok", "records": "ok", "files": "ok"}
    assert (ready[0], json.loads(ready[2])) == (200, {"ready": True, "checks": checks})
    assert (unready[0], json.loads(unready[2])) == (503, {"ready": False, "checks": {**checks, "database": "failed"}})


def test_probes_and_metrics_answer_promptly_while_the_database_is_gone(tmp_path: Path) -> None:
    database = fresh_database()
    app = ExperienceHTTPService(HTTPServiceConfig(tmp_path / "home", TOKEN), SCHEMA, None, database=database)
    app.write(_experience(SCHEMA))
    with RunningServer(app) as url:
        before = _metrics(_get(url, "/metrics")[2].decode())
        _lose(database)
        started = time.monotonic()
        ready = _get(url, "/readyz")
        status, _, body = _get(url, "/metrics")
        elapsed = time.monotonic() - started

    after = _metrics(body.decode())
    assert (ready[0], json.loads(ready[2])["checks"]["database"]) == (503, "failed")
    assert status == 200
    assert (before['hyperloom_kb_ready{check="database"}'], after['hyperloom_kb_ready{check="database"}']) == (1, 0)
    assert any(series.startswith("hyperloom_kb_experiences") for series in before)
    assert not any(series.startswith("hyperloom_kb_experiences") for series in after)
    assert any(series.startswith("hyperloom_kb_http_requests_total") for series in after)
    assert elapsed < 10


def test_a_record_file_lost_since_the_last_start_makes_the_next_start_unready(tmp_path: Path) -> None:
    home, database = tmp_path / "home", fresh_database()
    first = ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=database)
    experience = _experience(SCHEMA)
    first.write(experience)
    (home / first.kb_id / "records" / f"{experience.id}.json").unlink()

    restarted = ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=database)
    with RunningServer(restarted) as url:
        status, _, body = _get(url, "/readyz")
        metrics = _metrics(_get(url, "/metrics")[2].decode())

    assert (status, json.loads(body)["checks"]["records"]) == (503, "failed")
    assert metrics["hyperloom_kb_records_missing"] == 1


def test_a_file_lost_since_the_last_start_makes_the_next_start_unready(tmp_path: Path) -> None:
    home, database = tmp_path / "home", fresh_database()
    first = ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=database)
    artifact = tmp_path / "trace.json"
    artifact.write_bytes(b"[0.41]\n")
    ref = file_ref(artifact)
    with artifact.open("rb") as stream:
        first.put_file(ref.sha256, ref.bytes, stream)
    first.write(replace(_experience(SCHEMA), change={**_experience(SCHEMA).change, "artifact": ref}))
    first.file_path(ref).unlink()

    restarted = ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=database)
    with RunningServer(restarted) as url:
        status, _, body = _get(url, "/readyz")
        metrics = _metrics(_get(url, "/metrics")[2].decode())

    assert (status, json.loads(body)["checks"]["files"]) == (503, "failed")
    assert (metrics["hyperloom_kb_files_missing"], metrics["hyperloom_kb_files"]) == (1, 1)
    assert metrics["hyperloom_kb_file_bytes"] == ref.bytes


def test_metrics_count_requests_writes_and_refusals_and_sample_what_the_kb_holds(tmp_path: Path) -> None:
    app = _service(tmp_path / "home", name="team hub")
    experience = _experience(SCHEMA)
    path = f"/v1/experiences/{experience.id}"
    with RunningServer(app) as url:
        for _ in range(2):
            _http(url, "PUT", path, {"experience": experience.to_dict(), "declaration": SCHEMA.to_dict()})
        _http(url, "GET", "/health", token="wrong")
        status, headers, body = _get(url, "/metrics")
    metrics = _metrics(body.decode())
    schema = SCHEMA.schema_ref

    assert status == 200 and headers["Content-Type"].startswith("text/plain")
    put = 'method="PUT",route="/v1/experiences/{experience_id}"'
    assert metrics[f'hyperloom_kb_http_requests_total{{{put},status="200"}}'] == 2
    assert metrics[f'hyperloom_kb_http_request_duration_seconds_bucket{{{put},le="+Inf"}}'] == 2
    assert metrics[f'hyperloom_kb_writes_total{{result="created",schema_ref="{schema}"}}'] == 1
    assert metrics[f'hyperloom_kb_writes_total{{result="unchanged",schema_ref="{schema}"}}'] == 1
    assert metrics["hyperloom_kb_http_unauthorized_total"] == 1
    assert metrics[f'hyperloom_kb_experiences{{schema_ref="{schema}",state="visible"}}'] == 1
    assert (
        metrics[
            f'hyperloom_kb_build_info{{kb_id="{app.kb_id}",name="team hub",code_digest="{app.health()["code_digest"]}"}}'
        ]
        == 1
    )
    assert metrics['hyperloom_kb_ready{check="database"}'] == 1
    assert metrics["hyperloom_kb_record_bytes"] > 0
    assert "# TYPE hyperloom_kb_http_request_duration_seconds histogram" in body.decode()


def test_every_answer_carries_its_request_id(tmp_path: Path) -> None:
    with RunningServer(_service(tmp_path / "home")) as url:
        named = _get(url, "/livez", {"X-Request-ID": "req-from-client"})
        unnamed = _get(url, "/livez")
        unusable = _get(url, "/livez", {"X-Request-ID": "has spaces"})

    assert named[1]["X-Request-ID"] == "req-from-client"
    assert re.fullmatch(r"[0-9a-f]{32}", unnamed[1]["X-Request-ID"])
    assert re.fullmatch(r"[0-9a-f]{32}", unusable[1]["X-Request-ID"])


def _logged(caplog: pytest.LogCaptureFixture, name: str) -> dict[str, Any]:
    """The first ``name`` event captured; a request is logged on the server's thread after its answer is sent."""

    deadline = time.monotonic() + 5
    while True:
        for record in caplog.records:
            fields = getattr(record, "fields", {})
            if fields.get("event") == name:
                return fields
        assert time.monotonic() < deadline, f"no {name} event was logged"
        time.sleep(0.01)


def test_a_request_and_the_state_it_changes_are_logged_with_its_id_and_client(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = _service(tmp_path / "home")
    experience = _experience(SCHEMA)
    with RunningServer(app) as url:
        client = RemoteClient(RemoteConfig(url, TOKEN))
        client.identify("kb-0123456789abcdef0123456789abcdef", "laptop")
        client.write(experience, declaration=SCHEMA)
        with caplog.at_level(logging.INFO, logger="hyperloom_kb.http_service"):
            client.exclude(experience.id, reason="measured on a noisy node")
            request = _logged(caplog, "http_request")
    audit = _logged(caplog, "audit")

    assert audit["action"] == "exclude" and audit["experience_id"] == experience.id
    assert audit["reason"] == "measured on a noisy node"
    assert (request["method"], request["route"], request["status"]) == ("POST", "/v1/exclusions", 200)
    assert audit["request_id"] == request["request_id"] and re.fullmatch(r"[0-9a-f]{32}", request["request_id"])
    assert (request["client_kb_id"], request["client_name"]) == ("kb-0123456789abcdef0123456789abcdef", "laptop")


def test_a_log_line_is_one_json_object_with_its_event_fields() -> None:
    logger = logging.getLogger("hyperloom_kb.tests.json")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        event(logger, "audit", action="label", name="before tuning")
    finally:
        logger.removeHandler(handler)
    line = json.loads(JsonLogFormatter().format(records[0]))

    assert {key: line[key] for key in ("level", "logger", "message", "event", "action", "name")} == {
        "level": "info",
        "logger": "hyperloom_kb.tests.json",
        "message": "audit",
        "event": "audit",
        "action": "label",
        "name": "before tuning",
    }
    assert line["ts"].endswith("Z")


def _writes(service: ExperienceHTTPService) -> list[dict[str, Any]]:
    with service._database.transaction() as connection:
        return connection.execute(
            "SELECT experience_id, result, source_kb_id, source_name, request_id FROM writes ORDER BY write_id"
        ).fetchall()


def test_every_write_is_recorded_with_the_kb_that_sent_it(tmp_path: Path) -> None:
    hub = _service(tmp_path / "global", name="hub")
    with RunningServer(hub) as global_url:
        teammate = RemoteClient(RemoteConfig(global_url, TOKEN))
        teammate.write(_experience(SCHEMA, run_id="teammate"), declaration=SCHEMA)
        local = _service(tmp_path / "local", global_url=global_url, name="laptop")
        local.write(_experience(SCHEMA, run_id="mine"))
        local.push()
        local.pull(SCHEMA.schema_ref)
    on_global = {row["experience_id"]: row for row in _writes(hub)}
    on_local = {row["experience_id"]: row for row in _writes(local)}
    mine, theirs = _experience(SCHEMA, run_id="mine").id, _experience(SCHEMA, run_id="teammate").id

    assert (on_global[mine]["source_kb_id"], on_global[mine]["source_name"]) == (local.kb_id, "laptop")
    assert re.fullmatch(r"[0-9a-f]{32}", on_global[mine]["request_id"])
    assert on_global[theirs]["source_kb_id"] == ""
    assert (on_local[theirs]["source_kb_id"], on_local[theirs]["result"]) == (hub.kb_id, "created")


def test_a_stopped_service_finishes_the_request_in_flight(tmp_path: Path) -> None:
    app = _service(tmp_path / "home")
    started, answer = threading.Event(), {}
    health = app.health

    def slow_health() -> dict[str, Any]:
        started.set()
        time.sleep(0.5)
        return health()

    app.health = slow_health  # type: ignore[method-assign]
    server = create_http_server(app, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    stop = threading.Event()
    serving = threading.Thread(target=serve_until_stopped, args=(server, stop, 5.0))
    serving.start()
    asking = threading.Thread(target=lambda: answer.update(status=_http(url, "GET", "/health")[0]))
    asking.start()
    started.wait(timeout=5)
    stop.set()
    serving.join(timeout=10)
    asking.join(timeout=10)
    server.server_close()

    assert not serving.is_alive()
    assert answer == {"status": 200}


def test_sigterm_stops_the_service_once_it_drained(tmp_path: Path) -> None:
    home, port = tmp_path / "home", _free_port()
    env = {key: value for key, value in os.environ.items() if not key.startswith(("ANTHROPIC_", "HYPERLOOM_"))} | {
        "HYPERLOOM_KB_TOKEN": TOKEN,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    log_path = tmp_path / "service.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "hyperloom_kb", "--home", str(home), "--port", str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    client = RemoteClient(RemoteConfig(f"http://127.0.0.1:{port}", TOKEN))
    deadline = time.monotonic() + 60
    while True:
        try:
            client.health()
            break
        except RemoteClientError:
            assert process.poll() is None and time.monotonic() < deadline, log_path.read_text(encoding="utf-8")
            time.sleep(0.2)
    process.send_signal(signal.SIGTERM)
    returncode = process.wait(timeout=60)
    events = [
        json.loads(line).get("event")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]

    assert returncode == 0
    assert (home / "kb.sqlite3").is_file()
    assert "listening" in events and "stopped" in events
