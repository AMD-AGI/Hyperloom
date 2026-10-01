# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local service pushes the Experiences written to it to its global KB and pulls the global KB's into itself."""

from __future__ import annotations

import io
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import (
    Change,
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    HTTPServiceConfig,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Outcome,
    Provenance,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    ServiceSettings,
    create_http_server,
    derive_experience_id,
    sync,
)

LOCAL_TOKEN = "local-token"
GLOBAL_TOKEN = "global-token"
NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


def _declaration(objective: str = "throughput@v1") -> ExperienceDeclaration:
    return ExperienceDeclaration(
        identity=(FieldDeclaration("model", "Model."),),
        baseline_identity=(FieldDeclaration("config", "Baseline."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(ObjectiveDeclaration(objective, ObjectiveDirection.HIGHER_IS_BETTER, "Throughput."),),
        decisions=("keep", "revert"),
    )


def _experience(schema: ExperienceDeclaration, seq: int, *, run_id: str = "local-run") -> Experience:
    return Experience(
        id=derive_experience_id("sync-test", run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": "qwen3"},
        objective=schema.objectives[0].id,
        baseline_identity={"config": "default"},
        baseline_value=100.0,
        provenance=Provenance("sync-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        reasoning=f"Try knob {seq}.",
        change=Change({"knob": f"knob_{seq}"}, f"Set knob {seq}.", kind="config"),
        outcome=Outcome("keep", 110.0 + seq),
        reflection="Measured.",
    )


def _service(
    home: Path,
    schema: ExperienceDeclaration,
    token: str,
    global_url: str | None = None,
    **client_options: Any,
) -> ExperienceHTTPService:
    global_kb = None if global_url is None else RemoteClient(RemoteConfig(global_url, GLOBAL_TOKEN), **client_options)
    return ExperienceHTTPService(HTTPServiceConfig(home, token), schema, None, global_kb=global_kb)


@contextmanager
def _serving(app: ExperienceHTTPService) -> Iterator[str]:
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _client(url: str, token: str, tmp_path: Path) -> RemoteClient:
    return RemoteClient(RemoteConfig(url, token, spool_root=tmp_path / "spool"))


def _ids(client: RemoteClient) -> set[str]:
    return {str(item["experience_id"]) for item in client.list_experiences().items}


def test_push_sends_each_local_write_to_the_global_kb_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sync, "SYNC_BATCH", 2)
    schema = _declaration()
    with _serving(_service(tmp_path / "global", schema, GLOBAL_TOKEN)) as global_url:
        with _serving(_service(tmp_path / "local", schema, LOCAL_TOKEN, global_url)) as local_url:
            local = _client(local_url, LOCAL_TOKEN, tmp_path)
            for seq in range(3):
                local.write(_experience(schema, seq))
            first = local.push()
            local.write(_experience(schema, 3))
            second = local.push()
        pushed = _ids(_client(global_url, GLOBAL_TOKEN, tmp_path))

    assert (first["status"], first["created"], first["unchanged"], first["has_more"]) == ("completed", 3, 0, False)
    assert (second["status"], second["created"], second["unchanged"]) == ("completed", 1, 0)
    assert first["global_url"] == global_url
    assert pushed == {_experience(schema, seq).id for seq in range(4)}


def test_pull_stores_the_global_kb_records_and_never_pushes_them_back(tmp_path: Path) -> None:
    schema = _declaration()
    with _serving(_service(tmp_path / "global", schema, GLOBAL_TOKEN)) as global_url:
        shared = _client(global_url, GLOBAL_TOKEN, tmp_path)
        for seq in range(3):
            shared.write(_experience(schema, seq, run_id="teammate-run"))
        with _serving(_service(tmp_path / "local", schema, LOCAL_TOKEN, global_url)) as local_url:
            local = _client(local_url, LOCAL_TOKEN, tmp_path)
            pulled = local.pull(schema.schema_ref)
            again = local.pull(schema.schema_ref)
            local.write(_experience(schema, 0))
            pushed = local.push()
            listed = _ids(local)
            readable = local.health()["experience_count"]

    assert (pulled["status"], pulled["created"], pulled["rejected"]) == ("completed", 3, [])
    assert (again["created"], again["unchanged"]) == (0, 0)
    assert (pushed["created"], pushed["skipped"]) == (1, 3)
    # Reads see what was pulled; list and export name only what was written here.
    assert readable == 4
    assert listed == {_experience(schema, 0).id}


def test_a_push_the_global_kb_drops_resumes_where_it_stopped(tmp_path: Path) -> None:
    puts: list[str] = []

    def drops_the_second_write(request: urllib.request.Request, **options: Any) -> Any:
        if request.get_method() == "PUT":
            puts.append(request.full_url)
            if len(puts) == 2:
                raise urllib.error.URLError("global KB went away")
        return urllib.request.urlopen(request, **options)

    schema = _declaration()
    with _serving(_service(tmp_path / "global", schema, GLOBAL_TOKEN)) as global_url:
        local_app = _service(tmp_path / "local", schema, LOCAL_TOKEN, global_url, opener=drops_the_second_write)
        with _serving(local_app) as local_url:
            local = _client(local_url, LOCAL_TOKEN, tmp_path)
            for seq in range(3):
                local.write(_experience(schema, seq))
            interrupted = local.push()
            resumed = local.push()
        pushed = _ids(_client(global_url, GLOBAL_TOKEN, tmp_path))

    assert (interrupted["status"], interrupted["created"]) == ("incomplete", 1)
    assert "URLError" in str(interrupted["error"])
    assert (resumed["status"], resumed["created"], resumed["unchanged"]) == ("completed", 2, 0)
    assert pushed == {_experience(schema, seq).id for seq in range(3)}


def test_a_record_a_proxy_refuses_as_too_large_never_holds_back_the_rest(tmp_path: Path) -> None:
    puts: list[str] = []

    def refuses_the_second_write(request: urllib.request.Request, **options: Any) -> Any:
        if request.get_method() == "PUT":
            puts.append(request.full_url)
            if len(puts) == 2:
                body = io.BytesIO(b"<html>413 Request Entity Too Large</html>")
                raise urllib.error.HTTPError(request.full_url, 413, "Request Entity Too Large", Message(), body)
        return urllib.request.urlopen(request, **options)

    schema = _declaration()
    with _serving(_service(tmp_path / "global", schema, GLOBAL_TOKEN)) as global_url:
        local_app = _service(tmp_path / "local", schema, LOCAL_TOKEN, global_url, opener=refuses_the_second_write)
        with _serving(local_app) as local_url:
            local = _client(local_url, LOCAL_TOKEN, tmp_path)
            for seq in range(3):
                local.write(_experience(schema, seq))
            pushed = local.push()
            again = local.push()

    refused = puts[1].rsplit("/", 1)[-1]
    assert (pushed["status"], pushed["created"]) == ("completed", 2)
    assert [item["experience_id"] for item in pushed["rejected"]] == [refused]
    assert (again["status"], again["created"], again["rejected"]) == ("completed", 0, [])


def test_a_service_without_a_global_kb_refuses_to_sync(tmp_path: Path) -> None:
    schema = _declaration()
    with _serving(_service(tmp_path / "local", schema, LOCAL_TOKEN)) as local_url:
        local = _client(local_url, LOCAL_TOKEN, tmp_path)
        for operation in (local.push, lambda: local.pull(schema.schema_ref)):
            with pytest.raises(RemoteClientError, match="started without a global Experience KB") as refused:
                operation()
            assert refused.value.retryable is False


def test_a_global_kb_keeps_every_user_schema_and_each_user_pulls_back_its_own(tmp_path: Path) -> None:
    schemas = {user: _declaration(objective=f"{user}@v1") for user in ("alice", "bob", "carol")}
    with _serving(_service(tmp_path / "global", _declaration(), GLOBAL_TOKEN)) as global_url:
        for user, schema in schemas.items():
            with _serving(_service(tmp_path / user, schema, LOCAL_TOKEN, global_url)) as local_url:
                local = _client(local_url, LOCAL_TOKEN, tmp_path)
                local.write(_experience(schema, 0, run_id=f"{user}-run"), declaration=schema)
                assert local.push()["created"] == 1
        shared = _client(global_url, GLOBAL_TOKEN, tmp_path).health()

        # A second workspace of alice's pulls alice's schema only.
        with _serving(_service(tmp_path / "alice-2", schemas["alice"], LOCAL_TOKEN, global_url)) as local_url:
            teammate = _client(local_url, LOCAL_TOKEN, tmp_path)
            pulled = teammate.pull(schemas["alice"].schema_ref)
            held = teammate.health()["schemas"]

    assert set(shared["schemas"]) >= {schema.schema_ref for schema in schemas.values()}
    assert all(shared["schemas"][schema.schema_ref] == 1 for schema in schemas.values())
    assert pulled["created"] == 1
    assert held == {schemas["alice"].schema_ref: 1}


def test_a_workspace_that_switched_schema_keeps_and_syncs_both(tmp_path: Path) -> None:
    first, second = _declaration(objective="first@v1"), _declaration(objective="second@v1")
    with _serving(_service(tmp_path / "global", _declaration(), GLOBAL_TOKEN)) as global_url:
        teammate = _client(global_url, GLOBAL_TOKEN, tmp_path)
        teammate.write(_experience(first, 0, run_id="teammate-first"), declaration=first)
        teammate.write(_experience(second, 0, run_id="teammate-second"), declaration=second)

        with _serving(_service(tmp_path / "local", first, LOCAL_TOKEN, global_url)) as local_url:
            _client(local_url, LOCAL_TOKEN, tmp_path).write(_experience(first, 1, run_id="run-1"), declaration=first)
        # The workspace now runs the second schema; its service restarts with it as the default.
        with _serving(_service(tmp_path / "local", second, LOCAL_TOKEN, global_url)) as local_url:
            local = _client(local_url, LOCAL_TOKEN, tmp_path)
            local.write(_experience(second, 1, run_id="run-2"), declaration=second)
            pushed = local.push()
            pulled = [local.pull(schema.schema_ref) for schema in (first, second)]
            first_ids = {
                str(item["experience_id"]) for item in local.list_experiences(schema_ref=first.schema_ref).items
            }
            health = local.health()

    assert pushed["created"] == 2
    assert [report["created"] for report in pulled] == [1, 1]
    assert health["schema_ref"] == second.schema_ref
    assert health["schemas"] == {first.schema_ref: 2, second.schema_ref: 2}
    assert first_ids == {_experience(first, 1, run_id="run-1").id}


def test_a_write_of_an_unregistered_schema_needs_its_declaration(tmp_path: Path) -> None:
    other = _declaration(objective="other@v1")
    with _serving(_service(tmp_path / "local", _declaration(), LOCAL_TOKEN)) as local_url:
        local = _client(local_url, LOCAL_TOKEN, tmp_path)
        with pytest.raises(RemoteClientError, match="not registered; write it with its declaration"):
            local.write(_experience(other, 0))
        with pytest.raises(RemoteClientError, match="does not derive"):
            local.write(_experience(other, 0), declaration=_declaration())
        written = local.write(_experience(other, 0), declaration=other)
        assert written.status == "created"


def test_a_spooled_write_of_a_new_schema_is_accepted_when_the_service_returns(tmp_path: Path) -> None:
    other = _declaration(objective="other@v1")
    home = tmp_path / "local"
    app = _service(home, _declaration(), LOCAL_TOKEN)
    server = create_http_server(app, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    server.server_close()
    offline = _client(url, LOCAL_TOKEN, tmp_path)
    assert offline.publish(_experience(other, 0), declaration=other).status == "spooled"

    with _serving(_service(home, _declaration(), LOCAL_TOKEN)) as local_url:
        flushed = _client(local_url, LOCAL_TOKEN, tmp_path).flush_spool()

    assert [result.status for result in flushed] == ["created"]


def test_reads_search_the_schema_they_name(tmp_path: Path) -> None:
    other = _declaration(objective="other@v1")
    with _serving(_service(tmp_path / "local", _declaration(), LOCAL_TOKEN)) as local_url:
        local = _client(local_url, LOCAL_TOKEN, tmp_path)
        local.write(_experience(other, 0), declaration=other)
        default = local.read("Pick a knob.", {})
        named = local.read("Pick a knob.", {}, schema_ref=other.schema_ref)

    # The default schema holds nothing; the named one holds an Experience, and reads plan through a planner.
    assert (default.status, default.warnings) == ("completed", ())
    assert named.status == "unavailable"
    assert "read_service_unconfigured" in named.warnings


def test_export_pages_complete_records_in_write_order(tmp_path: Path) -> None:
    schema = _declaration()
    written = [_experience(schema, seq) for seq in (2, 0, 1)]
    with _serving(_service(tmp_path / "global", schema, GLOBAL_TOKEN)) as global_url:
        shared = _client(global_url, GLOBAL_TOKEN, tmp_path)
        for experience in written:
            shared.write(experience)
        first = shared.export_page(limit=2)
        rest = shared.export_page(after=first.next_cursor, limit=2)
        with pytest.raises(RemoteClientError, match="HTTP 400"):
            shared.export_page(limit=101)

    exported = [Experience.from_dict(item["experience"]) for item in (*first.items, *rest.items)]
    assert exported == written
    assert (first.has_more, rest.has_more) == (True, False)


def test_settings_digest_changes_with_what_the_service_does_not_with_spelling() -> None:
    gateway = {"ANTHROPIC_BASE_URL": "https://gateway.example", "ANTHROPIC_API_KEY": "key", "CLAUDE_MODEL": "m"}
    aliased = {**gateway, "ANTHROPIC_AUTH_TOKEN": "key", "ANTHROPIC_MODEL": "m"}
    shared = {**gateway, "HYPERLOOM_GLOBAL_KB_URL": "https://global.example", "HYPERLOOM_GLOBAL_KB_TOKEN": "t"}
    half = {**gateway, "HYPERLOOM_GLOBAL_KB_URL": "https://global.example"}

    assert ServiceSettings.from_env(aliased).digest() == ServiceSettings.from_env(gateway).digest()
    assert ServiceSettings.from_env(shared).digest() != ServiceSettings.from_env(gateway).digest()
    assert ServiceSettings.from_env(half).global_kb is None
    assert "HYPERLOOM_GLOBAL_KB_TOKEN must be set" in ServiceSettings.from_env(half).global_problem
