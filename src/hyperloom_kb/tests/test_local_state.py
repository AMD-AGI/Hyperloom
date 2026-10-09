# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A service's reads see one state of its corpus: labels save states, a restore brings one back, exclusions hide."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom_kb import (
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    HTTPServiceConfig,
    ObjectiveDeclaration,
    Provenance,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    create_http_server,
    derive_experience_id,
)
from hyperloom_kb.database import Database
from hyperloom_kb.tests.conftest import fresh_database

TOKEN = "state-token"
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
SCHEMA = ExperienceDeclaration(
    objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
    identity=(FieldDeclaration("model", "Model."),),
    baseline=(FieldDeclaration("config", "Baseline.", group=True),),
    change=(
        FieldDeclaration("knob", "Knob.", group=True),
        FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
    ),
    outcome=(
        FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
        FieldDeclaration("value", "Throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
    ),
    reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
)


def _experience(seq: int) -> Experience:
    return Experience(
        id=derive_experience_id("state-test", "run", seq),
        run_id="run",
        seq=seq,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": "qwen3"},
        objective="throughput@v1",
        baseline={"config": "default"},
        provenance=Provenance("state-test", "1"),
        schema_ref=SCHEMA.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": f"Try knob {seq}."},
        change={"knob": f"knob_{seq}", "summary": f"Set knob {seq}."},
        outcome={"decision": "keep", "value": 110.0 + seq},
        reflection={"text": "Measured."},
    )


def _id(seq: int) -> str:
    return _experience(seq).id


def _service(home: Path, global_url: str | None = None, *, database: Database | None = None) -> ExperienceHTTPService:
    global_kb = None if global_url is None else RemoteClient(RemoteConfig(global_url, TOKEN))
    return ExperienceHTTPService(
        HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=database or fresh_database(), global_kb=global_kb
    )


@contextmanager
def _serving(app: ExperienceHTTPService) -> Iterator[RemoteClient]:
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield RemoteClient(RemoteConfig(f"http://127.0.0.1:{server.server_address[1]}", app.config.token))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _listed(client: RemoteClient, *, include_excluded: bool = False) -> set[str]:
    return {str(item["experience_id"]) for item in client.list_experiences(include_excluded=include_excluded).items}


def _readable(app: ExperienceHTTPService) -> int:
    return int(app.read(decision="Pick the next change.", context={})["eligible_count"])


def test_a_restore_saves_the_unlabelled_state_and_later_writes_continue_from_the_restored_one(tmp_path: Path) -> None:
    app = _service(tmp_path / "local")
    with _serving(app) as client:
        for seq in range(3):
            client.write(_experience(seq))
        three = client.create_label(name="three")
        for seq in (3, 4):
            client.write(_experience(seq))
        before = client.labels()

        restored = client.restore(str(three["label_id"]))
        after_restore = (_readable(app), _listed(client))
        client.write(_experience(5))
        continued = _listed(client)

        back = client.restore(str(restored["saved"]["label_id"]))
        five = _listed(client)
        labels = client.labels()

    assert (before["current_label_id"], before["modified"]) == (three["label_id"], True)
    assert restored["restored"]["label_id"] == three["label_id"]
    assert restored["saved"]["reason"] == "before_restore"
    assert restored["saved"]["name"].startswith("before restore ")
    assert restored["saved"]["member_count"] == 5
    assert after_restore == (3, {_id(seq) for seq in range(3)})
    assert continued == {_id(seq) for seq in (0, 1, 2, 5)}
    # Writing after the restore changed the state again, so going back saved it too before replacing it.
    assert back["saved"]["member_count"] == 4
    assert five == {_id(seq) for seq in range(5)}
    assert (labels["current_label_id"], labels["modified"]) == (restored["saved"]["label_id"], False)
    assert [label["name"] for label in labels["labels"]][-1] == "three"


def test_an_include_brings_back_what_a_restore_set_outside_once_no_label_holds_it(tmp_path: Path) -> None:
    app = _service(tmp_path / "local")
    with _serving(app) as client:
        client.write(_experience(0))
        one = client.create_label(name="one")
        client.write(_experience(1))
        restored = client.restore(str(one["label_id"]))
        client.delete_label(str(restored["saved"]["label_id"]))
        stranded = (_readable(app), client.labels()["labels"])
        included = client.include(_id(1))
        shown = (_readable(app), _listed(client))

    assert stranded == (1, [one])
    assert included == {"experience_id": _id(1), "status": "included"}
    assert shown == (2, {_id(0), _id(1)})


def test_an_exclusion_hides_an_experience_and_lifting_it_restores_the_labelled_state(tmp_path: Path) -> None:
    app = _service(tmp_path / "local")
    with _serving(app) as client:
        for seq in range(2):
            client.write(_experience(seq))
        client.create_label(name="both")
        excluded = client.exclude(_id(0), reason="measured on a degraded node")
        hidden = (_readable(app), _listed(client), _listed(client, include_excluded=True), client.labels()["modified"])
        listing = client.exclusions()
        lifted = client.include(_id(0))
        again = client.include(_id(0))
        shown = (_readable(app), _listed(client), client.labels()["modified"])
        history = client.exclusions()["history"]

    assert excluded == {"experience_id": _id(0), "status": "excluded"}
    assert hidden == (1, {_id(1)}, {_id(0), _id(1)}, True)
    assert [(row["experience_id"], row["reason"]) for row in listing["exclusions"]] == [
        (_id(0), "measured on a degraded node")
    ]
    assert (lifted["status"], again["status"]) == ("included", "not_excluded")
    # The state equals its label again, so it carries no unlabelled change.
    assert shown == (2, {_id(0), _id(1)}, False)
    assert [(row["action"], row["reason"]) for row in history] == [
        ("exclude", "measured on a degraded node"),
        ("include", ""),
    ]


def test_a_label_saves_the_exclusions_with_the_state(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "local")) as client:
        for seq in range(2):
            client.write(_experience(seq))
        clean = client.create_label(name="clean")
        client.exclude(_id(1), reason="crashed the server")
        excluding = client.create_label(name="excluding")

        client.restore(str(clean["label_id"]))
        under_clean = (_listed(client), client.exclusions()["exclusions"])
        client.restore(str(excluding["label_id"]))
        under_excluding = _listed(client)
        reason = client.exclusions()["exclusions"][0]["reason"]

    assert under_clean == ({_id(0), _id(1)}, [])
    assert under_excluding == {_id(0)}
    assert reason == "crashed the server"


def test_labels_are_managed_by_id_and_their_names_may_repeat(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "local")) as client:
        client.write(_experience(0))
        first = client.create_label(name="same")
        client.write(_experience(1))
        second = client.create_label(name="same")
        client.delete_label(str(first["label_id"]))
        remaining = client.labels()["labels"]
        with pytest.raises(RemoteClientError, match="404"):
            client.restore(str(first["label_id"]))
        with pytest.raises(RemoteClientError, match="404"):
            client.exclude("exp-00000000000000000000000000000000", reason="unknown")

    assert first["label_id"] != second["label_id"]
    assert [label["label_id"] for label in remaining] == [second["label_id"]]


def test_a_push_sends_only_what_reads_see_and_the_rest_once_they_see_it(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as client:
            for seq in range(2):
                client.write(_experience(seq))
            client.exclude(_id(0), reason="not ready to share")
            first = client.push()
            shared_first = _listed(global_client)
            client.include(_id(0))
            second = client.push()
            shared_second = _listed(global_client)

    assert (first["created"], first["held_back"]) == (1, 1)
    assert shared_first == {_id(1)}
    assert (second["created"], second["held_back"]) == (1, 0)
    assert shared_second == {_id(0), _id(1)}


def test_an_experience_excluded_before_its_push_waits_for_an_include_whatever_a_restore_shows(tmp_path: Path) -> None:
    with _serving(_service(tmp_path / "global")) as global_client:
        with _serving(_service(tmp_path / "local", global_client.config.base_url)) as client:
            client.write(_experience(0))
            before = client.create_label(name="before the exclusion")
            client.exclude(_id(0), reason="not ready to share")
            client.push()
            client.restore(str(before["label_id"]))
            shown = _listed(client)
            after_restore = client.push()
            shared_after_restore = _listed(global_client)
            released = client.include(_id(0))
            after_include = client.push()
            shared = _listed(global_client)

    # The restore shows it to reads again, but only an include releases it for a push.
    assert shown == {_id(0)}
    assert (after_restore["created"], shared_after_restore) == (0, set())
    assert released["status"] == "included"
    assert (after_include["created"], shared) == (1, {_id(0)})


def test_a_kb_with_no_recorded_state_keeps_every_stored_experience_visible(tmp_path: Path) -> None:
    home, database = tmp_path / "local", fresh_database()
    with _serving(_service(home, database=database)) as client:
        for seq in range(3):
            client.write(_experience(seq))

    app = _service(home, database=database)
    with _serving(app) as client:
        visible = (_readable(app), _listed(client))
        labels = client.labels()

    assert visible == (3, {_id(seq) for seq in range(3)})
    assert (labels["current_label_id"], labels["modified"], labels["labels"]) == (None, True, [])
