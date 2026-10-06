# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every service has one identity; a pull brings one schema to everything the global KB's state holds of it."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    RemoteConfig,
    create_http_server,
    derive_experience_id,
    sync,
)
from hyperloom_kb.tests.conftest import fresh_database

TOKEN = "pull-token"
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _declaration(objective: str = "throughput@v1") -> ExperienceDeclaration:
    return ExperienceDeclaration(
        identity=(FieldDeclaration("model", "Model."),),
        baseline_identity=(FieldDeclaration("config", "Baseline."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(ObjectiveDeclaration(objective, ObjectiveDirection.HIGHER_IS_BETTER, "Throughput."),),
        decisions=("keep", "revert"),
    )


SCHEMA = _declaration()


def _experience(seq: int, *, run_id: str = "local-run", schema: ExperienceDeclaration = SCHEMA) -> Experience:
    return Experience(
        id=derive_experience_id("pull-test", run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": "qwen3"},
        objective=schema.objectives[0].id,
        baseline_identity={"config": "default"},
        baseline_value=100.0,
        provenance=Provenance("pull-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        reasoning=f"Try knob {seq}.",
        change=Change({"knob": f"knob_{seq}"}, f"Set knob {seq}.", kind="config"),
        outcome=Outcome("keep", 110.0 + seq),
        reflection="Measured.",
    )


def _service(home: Path, global_url: str | None = None, **options: Any) -> ExperienceHTTPService:
    global_kb = None if global_url is None else RemoteClient(RemoteConfig(global_url, TOKEN))
    schema = options.pop("schema", SCHEMA)
    database = options.pop("database", None) or fresh_database()
    return ExperienceHTTPService(
        HTTPServiceConfig(home, TOKEN), schema, None, database=database, global_kb=global_kb, **options
    )


@contextmanager
def _serving(app: ExperienceHTTPService, port: int = 0) -> Iterator[RemoteClient]:
    server = create_http_server(app, "127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        yield RemoteClient(RemoteConfig(url, TOKEN, spool_root=app.config.home.parent / "spool"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _port(client: RemoteClient) -> int:
    return int(client.config.base_url.rsplit(":", 1)[1])


def _readable(client: RemoteClient, schema: ExperienceDeclaration = SCHEMA) -> int:
    return int(client.health()["schemas"].get(schema.schema_ref, 0))


def _seed(global_client: RemoteClient, count: int, *, run_id: str = "teammate-run") -> None:
    for seq in range(count):
        global_client.write(_experience(seq, run_id=run_id), declaration=SCHEMA)


def test_a_kb_keeps_its_identity_in_its_database(tmp_path: Path) -> None:
    database = fresh_database()
    first = _service(tmp_path / "hub", name="team hub", database=database)
    another_replica = _service(tmp_path / "hub", database=database)
    other_database = _service(tmp_path / "hub")
    with _serving(first) as client:
        health = client.health()

    assert (health["kb_id"], health["name"]) == (first.kb_id, "team hub")
    assert another_replica.kb_id == first.kb_id
    assert other_database.kb_id != first.kb_id


def test_a_global_kb_state_decides_what_a_pull_brings(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 3)
        global_client.exclude(_experience(1, run_id="teammate-run").id, reason="failed its release check")
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            pulled = local.pull(SCHEMA.schema_ref)
            held = _readable(local)

    assert (pulled["created"], held) == (2, 2)


def test_a_pull_brings_what_the_global_kb_shows_again_after_excluding_it(tmp_path: Path) -> None:
    hidden = _experience(1, run_id="teammate-run").id
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 3)
        global_client.exclude(hidden, reason="under review")
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            first = local.pull(SCHEMA.schema_ref)
            global_client.include(hidden)
            second = local.pull(SCHEMA.schema_ref)
            held = _readable(local)

    assert first["created"] == 2
    assert (second["created"], second["unchanged"], held) == (1, 2, 3)


def test_a_pull_brings_what_a_global_kb_restore_shows_again(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 1)
        one = global_client.create_label(name="one")
        for seq in (1, 2):
            global_client.write(_experience(seq, run_id="teammate-run"), declaration=SCHEMA)
        rolled_back = global_client.restore(str(one["label_id"]))
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            first = local.pull(SCHEMA.schema_ref)
            global_client.restore(str(rolled_back["saved"]["label_id"]))
            second = local.pull(SCHEMA.schema_ref)
            held = _readable(local)

    assert (first["created"], second["created"], held) == (1, 2, 3)


def test_a_pull_brings_only_the_schema_it_names_and_registers_it(tmp_path: Path) -> None:
    other = _declaration(objective="latency@v1")
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 2)
        global_client.write(_experience(0, run_id="latency-run", schema=other), declaration=other)
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            pulled = local.pull(other.schema_ref)
            held = local.health()["schemas"]

    assert (pulled["status"], pulled["created"], pulled["schema_ref"]) == ("completed", 1, other.schema_ref)
    assert held == {SCHEMA.schema_ref: 0, other.schema_ref: 1}


def test_a_pull_first_labels_an_unlabelled_state_and_restoring_it_undoes_the_pull(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 3)
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            for seq in range(2):
                local.write(_experience(seq))
            pulled = local.pull(SCHEMA.schema_ref)
            after_pull = _readable(local)
            local.restore(str(pulled["saved"]["label_id"]))
            undone = _readable(local)
            again = local.pull(SCHEMA.schema_ref)
            after_again = _readable(local)

    assert (pulled["created"], pulled["saved"]["reason"], pulled["saved"]["member_count"]) == (3, "before_pull", 2)
    assert pulled["saved"]["name"].startswith("before pull ")
    assert (after_pull, undone) == (5, 2)
    # Nothing changed since the restore, so the second pull labels nothing and brings the global KB's back.
    assert (again["created"], again["saved"], after_again) == (0, None, 5)


def test_a_pull_brings_back_what_was_pushed_and_leaves_exclusions_standing(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            local.write(_experience(0))
            one = local.create_label(name="one")
            for seq in (1, 2):
                local.write(_experience(seq))
            local.push()
            local.restore(str(one["label_id"]))
            local.exclude(_experience(1).id, reason="regressed on a rerun")
            local.pull(SCHEMA.schema_ref)
            seen = {str(item["experience_id"]) for item in local.list_experiences().items}

    assert seen == {_experience(0).id, _experience(2).id}


def test_a_pull_over_several_batches_labels_the_state_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sync, "SYNC_BATCH", 1)
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 3)
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            local.write(_experience(0))
            pulled = local.pull(SCHEMA.schema_ref)
            reasons = [label["reason"] for label in local.labels()["labels"]]

    assert (pulled["status"], pulled["created"], pulled["has_more"]) == ("completed", 3, False)
    assert pulled["saved"]["reason"] == "before_pull"
    assert reasons == ["before_pull"]


def test_sync_refuses_another_global_kb_at_the_same_url(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global-a")) as first:
        _seed(first, 1)
        port = _port(first)
        local_app = _service(tmp_path / "local", first.config.base_url)
        with _serving(local_app) as local:
            assert local.pull(SCHEMA.schema_ref)["status"] == "completed"
    with _serving(_service(tmp_path / "global-b"), port=port) as replacement:
        _seed(replacement, 1, run_id="other-run")
        with _serving(local_app) as local:
            local.write(_experience(0))
            refused = (local.pull(SCHEMA.schema_ref), local.push())

    for report in refused:
        assert report["status"] == "refused"
        assert "it is another global KB" in str(report["error"])


def test_a_pull_refuses_a_global_kb_that_lost_experiences_it_pulled(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        _seed(global_client, 3)
        port = _port(global_client)
        global_kb_id = str(global_client.health()["kb_id"])
        local_app = _service(tmp_path / "local", global_client.config.base_url)
        with _serving(local_app) as local:
            assert local.pull(SCHEMA.schema_ref)["created"] == 3
    # The same KB serves again from a backup taken when it held only its first Experience.
    backup = fresh_database()
    backup.resolve_kb(adopt_kb_id=global_kb_id)
    with _serving(_service(tmp_path / "backup", database=backup), port=port) as restored:
        _seed(restored, 1)
        with _serving(local_app) as local:
            refused = local.pull(SCHEMA.schema_ref)

    assert refused["status"] == "refused"
    assert "it lost Experiences" in str(refused["error"])


def test_sync_refuses_a_global_kb_that_reports_no_identity(tmp_path: Path) -> None:
    old = _service(tmp_path / "global")
    current = old.health
    old.health = lambda: {key: value for key, value in current().items() if key != "kb_id"}  # type: ignore[method-assign]
    with _serving(old) as global_client:
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as local:
            refused = local.pull(SCHEMA.schema_ref)

    assert refused["status"] == "refused"
    assert "reports no identity; upgrade it" in str(refused["error"])
