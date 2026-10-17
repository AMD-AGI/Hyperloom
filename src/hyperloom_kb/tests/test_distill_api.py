# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an author does with Distills over a KB's API: look at a schema and its Experiences, explore a group, register
a form, preview a draft, create and edit it, read it back with its evidence, delete it, and see each step counted."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import (
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    ObjectiveDeclaration,
    Provenance,
    RemoteClient,
    RemoteClientError,
    derive_experience_id,
)
from hyperloom_kb.distill import FormDeclaration
from hyperloom_kb.tests.distills import (
    EXPERIENCES,
    IDS,
    NOW,
    OTHER,
    PLOTTED,
    RULE,
    SCHEMA,
    make_draft,
    make_experience,
    serving,
)
from hyperloom_kb.tests.logs import LOG, LOG_PATH
from hyperloom_kb.tests.test_observability import _logged


@pytest.fixture
def kb(tmp_path: Path) -> Iterator[tuple[ExperienceHTTPService, RemoteClient]]:
    with serving(tmp_path / "home") as served:
        yield served


def _refused(call: Any, status: int, detail: str) -> None:
    with pytest.raises(RemoteClientError, match=rf"HTTP {status}.*{detail}"):
        call()


def _members(entry: dict[str, Any]) -> dict[str, str]:
    return {member["id"]: member["class"] for member in entry["evidence"]["members"]}


# Looking at a schema and its Experiences.


def test_a_schema_view_names_the_paths_a_scope_may_use_its_roles_and_what_it_holds(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.exclude(IDS[6], reason="measured on a degraded node")

    view = client.schema(SCHEMA.schema_ref)
    held = client.experience(IDS[6])

    assert view["paths"] == [
        "objective",
        "baseline.throughput",
        "change.attention",
        "change.summary",
        "identity.model",
        "identity.tp",
        "outcome.decision",
        "outcome.throughput",
    ]
    assert view["roles"] == {
        "decision": "outcome.decision",
        "baseline_measurement": "baseline.throughput",
        "outcome_measurement": "outcome.throughput",
        "summary": "change.summary",
        "group": ["change.attention"],
    }
    assert view["experiences"] == {"stored": 7, "visible": 6}
    assert view["distills"] == {"visible": 0, "deleted": 0} and view["forms"] == []
    assert ExperienceDeclaration.from_dict(view["declaration"]).schema_ref == SCHEMA.schema_ref
    assert (held["experience"]["id"], held["visible"], held["files"]) == (IDS[6], False, {})
    _refused(lambda: client.experience("exp-unknown"), 404, "not held")
    _refused(lambda: client.schema(OTHER.schema_ref), 400, "not registered")


def test_exploring_summarizes_the_group_a_scope_selects_and_crosses_it_over_one_or_two_fields(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.exclude(IDS[5], reason="measured on a degraded node")

    group = client.explore(SCHEMA.schema_ref, scope={"identity.model": ["qwen3"]}, member_limit=2)
    crossed = client.explore(
        SCHEMA.schema_ref, scope={"identity.model": ["qwen3"]}, group_by=["change.attention", "outcome.decision"]
    )

    assert (group["count"], group["runs"], group["repeat_groups"]) == (5, 2, 2)
    assert group["decisions"] == {"keep": 2, "revert": 3}
    assert group["members"] == [IDS[4], IDS[3]] and group["members_total"] == 5
    assert group["measurement"] == {"count": 5, "min": 95.0, "p25": 96.0, "median": 97.0, "p75": 104.0, "max": 112.0}
    assert group["change_pct"]["median"] == pytest.approx(-3.0)
    cells = [(cell["values"], cell["count"]) for cell in crossed["cells"]]
    assert cells == [
        ({"change.attention": "triton", "outcome.decision": "revert"}, 3),
        ({"change.attention": "aiter", "outcome.decision": "keep"}, 1),
        ({"change.attention": "triton", "outcome.decision": "keep"}, 1),
    ]
    assert crossed["cells"][0]["members"] == [IDS[4], IDS[1], IDS[0]]
    _refused(lambda: client.explore(SCHEMA.schema_ref, group_by=["reflection.text"]), 400, "one scalar")
    _refused(
        lambda: client.explore(SCHEMA.schema_ref, group_by=["identity.model", "identity.tp", "objective"]),
        400,
        "at most 2",
    )
    _refused(lambda: client.explore(SCHEMA.schema_ref, scope={"identity.gpu": ["mi300x"]}), 400, "not a declared")
    _refused(lambda: client.explore(SCHEMA.schema_ref, member_limit=0), 400, "member_limit")


# Forms and previews.


def test_a_form_registers_once_per_schema_and_a_draft_must_use_one_registered_there(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb

    first = client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    again = client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    listed = client.distill_forms(SCHEMA.schema_ref)
    unregistered = client.preview_distill(make_draft(form_ref=PLOTTED.form_ref))

    assert (first["status"], again["status"], first["form_ref"]) == ("created", "unchanged", RULE.form_ref)
    assert [FormDeclaration.from_dict(item["form"]) for item in listed["forms"]] == [RULE]
    assert unregistered["valid"] is False and "form" in unregistered["errors"][0]
    _refused(lambda: client.register_distill_form(SCHEMA.schema_ref, {"name": "x", "claim": []}), 400, "description")


def test_a_preview_shows_what_a_create_would_store_and_stores_nothing(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())

    preview = client.preview_distill(make_draft())
    refused = client.preview_distill(make_draft(claim={"effect": "hurts"}))

    assert preview["valid"] is True and preview["errors"] == []
    assert preview["population"] == 4
    assert preview["evidence"] == {"supports": 3, "contradicts": 1, "irrelevant": 0, "runs": 2}
    assert preview["knowledge"]["claim"]["effect"] == "hurts"
    assert preview["overlaps"] == []
    assert refused == {"valid": False, "errors": ["claim is missing required field 'statement'"]}
    assert client.list_distills(SCHEMA.schema_ref)["items"] == []


# Creating, editing, and reading back.


def test_a_create_draws_the_scopes_group_applies_the_authors_edits_and_freezes_what_it_drew(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    app, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    draft = make_draft(remove=[IDS[1]], add=[IDS[3]], annotations={IDS[3]: "irrelevant"}, representatives=[IDS[4]])
    state = app.kb_state(SCHEMA.schema_ref)

    created = client.create_distill(draft)
    got = client.distill(str(created["entry_id"]))
    entry = got["entry"]

    assert re.fullmatch(r"dst-[0-9a-f]{32}", str(created["entry_id"]))
    assert (created["status"], created["version"]) == ("created", 1)
    assert created["evidence"] == {"supports": 2, "contradicts": 1, "irrelevant": 1, "runs": 2}
    assert _members(entry) == {IDS[4]: "support", IDS[2]: "contradict", IDS[0]: "support", IDS[3]: "irrelevant"}
    assert (entry["evidence"]["added"], entry["evidence"]["removed"]) == ([IDS[3]], [IDS[1]])
    assert entry["evidence"]["representatives"] == [IDS[4]]
    assert entry["derivation"] == {"kb_state": state}
    assert entry["provenance"]["producer"] == "hyperloom-kb" and entry["author"] == "zili"
    assert (got["current_version"], got["versions"], got["deleted"], got["outside"]) == (1, [1], False, False)


def test_following_the_evidence_pages_through_the_members_with_their_records(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(make_draft())["entry_id"])
    client.exclude(IDS[0], reason="measured on a degraded node")

    first = client.distill(entry_id, follow_evidence=True, limit=3)["evidence"]
    rest = client.distill(entry_id, follow_evidence=True, after=first["next"], limit=3)["evidence"]

    assert [item["id"] for item in first["items"]] == [IDS[4], IDS[2], IDS[1]]
    assert (first["next"], first["has_more"], first["total"]) == (3, True, 4)
    assert first["items"][0]["experience"]["outcome"] == {"decision": "revert", "throughput": 96.0}
    assert [(item["id"], item["class"], item["visible"]) for item in rest["items"]] == [(IDS[0], "support", False)]
    assert (rest["next"], rest["has_more"]) == (4, False)
    _refused(lambda: client.distill(entry_id, follow_evidence=True, limit=501), 400, "limit")


def test_a_versions_draft_writes_it_again_with_its_edits_and_annotations_and_drops_hidden_ids(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    written = make_draft(
        remove=[IDS[1]],
        add=[IDS[3]],
        annotations={IDS[3]: "irrelevant", IDS[2]: "support"},
        representatives=[IDS[4], IDS[3]],
        notes={"why": "aiter is the control."},
    )
    entry_id = str(client.create_distill(written)["entry_id"])
    first = client.distill(entry_id)

    client.update_distill(entry_id, first["draft"])
    again = client.distill(entry_id)
    client.exclude(IDS[3], reason="noisy")
    trimmed = client.distill(entry_id)["draft"]
    refreshed = client.update_distill(entry_id, trimmed)

    assert first["entry"]["evidence"]["annotated"] == sorted([IDS[3], IDS[2]])
    assert first["draft"] == {**written, "relations": {"entries": [], "external": []}}
    assert (again["entry"]["version"], again["entry"]["evidence"]) == (2, first["entry"]["evidence"])
    assert again["drift"]["evidence_status"] == "consistent"
    assert (trimmed["add"], trimmed["annotations"], trimmed["representatives"]) == ([], {IDS[2]: "support"}, [IDS[4]])
    assert refreshed["evidence"] == {"supports": 3, "contradicts": 0, "irrelevant": 0, "runs": 2}


def test_a_schema_that_declares_no_roles_still_explores_and_distills(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    plain = ExperienceDeclaration(
        objectives=(ObjectiveDeclaration("latency@v1", "Minimize latency."),),
        identity=(FieldDeclaration("model", "Model."),),
        change=(FieldDeclaration("knob", "Knob."),),
        outcome=(FieldDeclaration("note", "What happened.", FieldKind.TEXT),),
    )
    for seq, knob in enumerate(("a", "a", "b")):
        client.write(
            Experience(
                id=derive_experience_id("plain-test", f"run-{seq}", seq),
                run_id=f"run-{seq}",
                seq=seq,
                created_at=NOW,
                completed_at=NOW,
                identity={"model": "qwen3"},
                objective="latency@v1",
                provenance=Provenance("plain-test", "1"),
                schema_ref=plain.schema_ref,
                status=ExperienceStatus.COMPLETE,
                rationale={"reasoning": "Try it."},
                change={"knob": knob},
                outcome={"note": "Measured."},
            ),
            declaration=plain,
        )
    form = FormDeclaration("note", "A note.", (FieldDeclaration("text", "Text.", FieldKind.TEXT, required=True),))
    client.register_distill_form(plain.schema_ref, form.to_dict())

    explored = client.explore(plain.schema_ref, group_by=["change.knob"])
    view = client.schema(plain.schema_ref)
    created = client.create_distill(
        {
            "schema_ref": plain.schema_ref,
            "form_ref": form.form_ref,
            "scope": {"change.knob": ["a"]},
            "claim": {"text": "Knob a was tried twice."},
            "annotations": {derive_experience_id("plain-test", "run-1", 1): "contradict"},
        }
    )
    listed = client.list_distills(plain.schema_ref)["items"]

    assert (explored["count"], explored["runs"], explored["decisions"]) == (3, 3, {})
    assert (explored["measurement"], explored["change_pct"]) == ({"count": 0}, {"count": 0})
    assert [(cell["values"], cell["count"]) for cell in explored["cells"]] == [
        ({"change.knob": "a"}, 2),
        ({"change.knob": "b"}, 1),
    ]
    assert view["roles"] == {
        "decision": None,
        "baseline_measurement": None,
        "outcome_measurement": None,
        "summary": None,
        "group": [],
    }
    assert created["evidence"] == {"supports": 1, "contradicts": 1, "irrelevant": 0, "runs": 2}
    assert [(item["summary"], item["evidence_status"]) for item in listed] == [("", "consistent")]


def test_a_read_names_the_visible_members_that_represent_the_evidence(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    chosen = str(client.create_distill(make_draft())["entry_id"])
    designated = str(client.create_distill(make_draft(representatives=[IDS[0], IDS[1]]))["entry_id"])

    by_rule = client.distill(chosen)["representatives"]
    two = client.distill(chosen, representatives=2)["representatives"]
    client.exclude(IDS[2], reason="noisy")
    client.exclude(IDS[1], reason="noisy")
    without_hidden = (client.distill(chosen)["representatives"], client.distill(designated)["representatives"])

    # Every contradicting member first, then the newest supporting member of each run.
    assert by_rule == [IDS[2], IDS[4], IDS[1]] and two == [IDS[2], IDS[4]]
    assert without_hidden == ([IDS[4], IDS[0]], [IDS[0]])
    _refused(lambda: client.distill(chosen, representatives=51), 400, "representatives")


def test_an_update_adds_a_version_with_its_evidence_drawn_again_and_every_version_stays_readable(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(make_draft())["entry_id"])
    client.write(make_experience(7, "run-c", "qwen3", "triton", "revert", 94.0))

    preview = client.preview_distill(make_draft(), entry_id=entry_id)
    updated = client.update_distill(entry_id, make_draft(notes={"why": "A third run reverted too."}))
    latest, first = client.distill(entry_id), client.distill(entry_id, version=1)
    listed = client.list_distills(SCHEMA.schema_ref)["items"]

    assert preview["evidence"] == {"supports": 4, "contradicts": 1, "irrelevant": 0, "runs": 3}
    assert (updated["version"], updated["evidence"]) == (2, preview["evidence"])
    assert (latest["entry"]["id"], latest["current_version"], latest["versions"]) == (entry_id, 2, [1, 2])
    assert latest["entry"]["notes"] == {"why": "A third run reverted too."}
    assert first["entry"]["evidence"]["stats"]["supports"] == 3
    assert [(item["entry_id"], item["version"], item["supports"]) for item in listed] == [(entry_id, 2, 4)]
    _refused(lambda: client.update_distill("dst-" + "0" * 32, make_draft()), 404, "not held")
    _refused(
        lambda: client.update_distill(entry_id, make_draft(schema_ref=OTHER.schema_ref)), 400, f"not {OTHER.schema_ref}"
    )


def test_a_listing_shows_each_distill_at_the_version_its_state_shows_and_names_overlapping_ones(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(make_draft())["entry_id"])
    label = client.create_label(name="first rule")
    client.update_distill(entry_id, make_draft(tags=["attention", "qwen3"]))
    narrower = client.preview_distill(make_draft(scope={"identity.model": ["qwen3"]}))
    elsewhere = client.preview_distill(make_draft(scope={"identity.model": ["llama"]}, check=None))

    client.restore(str(label["label_id"]))
    restored = client.list_distills(SCHEMA.schema_ref)["items"]

    assert [item["entry_id"] for item in narrower["overlaps"]] == [entry_id]
    assert elsewhere["overlaps"] == []
    item = restored[0]
    assert (item["version"], item["tags"], item["state"]) == (1, ["attention"], "visible")
    assert item["summary"] == "Triton attention alone does not raise qwen3 throughput."
    assert item["scope"] == {"change.attention": ["triton"], "identity.model": ["qwen3"]}
    assert client.distill(entry_id)["current_version"] == 1


def test_deleting_a_distill_hides_it_from_listings_until_it_is_included_again(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(make_draft())["entry_id"])

    client.exclude(entry_id, reason="superseded by a per-model rule")
    hidden = client.list_distills(SCHEMA.schema_ref)["items"]
    shown_deleted = client.list_distills(SCHEMA.schema_ref, include_deleted=True)["items"]
    deleted = client.distill(entry_id)["deleted"]
    client.include(entry_id)

    assert hidden == [] and [(item["entry_id"], item["state"]) for item in shown_deleted] == [(entry_id, "deleted")]
    assert deleted is True
    assert [item["state"] for item in client.list_distills(SCHEMA.schema_ref)["items"]] == ["visible"]
    assert client.schema(SCHEMA.schema_ref)["distills"] == {"visible": 1, "deleted": 0}
    assert client.exclusions()["history"][-1]["experience_id"] == entry_id


@pytest.mark.parametrize(
    ("changes", "status", "detail"),
    [
        ({"remove": [IDS[3]]}, 400, "outside the scope's group"),
        ({"add": [IDS[0]]}, 400, "already selects"),
        ({"add": ["exp-unknown"]}, 400, "no visible Experience"),
        ({"annotations": {IDS[3]: "support"}}, 400, "not members"),
        ({"annotations": {IDS[0]: "strong"}}, 400, "must be one of"),
        ({"scope": {"identity.model": ["mistral"]}}, 400, "at least one Experience"),
        ({"check": {"reflection.text": ["x"]}}, 400, "does not hold one scalar"),
        ({"relations": {"entries": [{"kind": "refines", "target": "dst-" + "0" * 32}]}}, 404, "not held"),
        ({"claim": {"statement": "x", "effect": "maybe"}}, 400, "must be one of"),
        ({"mechanism": "x"}, 400, "mechanism"),
        ({"schema_ref": None}, 400, "schema_ref is required"),
    ],
)
def test_a_create_refuses_a_draft_the_kb_cannot_hold(
    kb: tuple[ExperienceHTTPService, RemoteClient], changes: dict[str, Any], status: int, detail: str
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    draft = {key: value for key, value in make_draft(**changes).items() if value is not None}

    _refused(lambda: client.create_distill(draft), status, detail)
    assert client.preview_distill(draft)["valid"] is False
    assert client.list_distills(SCHEMA.schema_ref)["items"] == []


def test_a_claim_names_only_files_the_kb_holds(kb: tuple[ExperienceHTTPService, RemoteClient]) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, PLOTTED.to_dict())
    claim = {"statement": "Reverts cluster below baseline.", "plot": LOG.to_dict()}
    draft = make_draft(form_ref=PLOTTED.form_ref, claim=claim)

    _refused(lambda: client.create_distill(draft), 400, "does not hold")
    client.put_file(LOG, LOG_PATH)
    created = client.create_distill(draft)

    assert created["status"] == "created"


def test_a_relation_names_a_distill_of_the_same_schema(kb: tuple[ExperienceHTTPService, RemoteClient]) -> None:
    app, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    broad = str(client.create_distill(make_draft(scope={"identity.model": ["qwen3"]}, check=None))["entry_id"])
    refines = {"entries": [{"kind": "refines", "target": broad}], "external": [{"uri": "wiki://attention/triton"}]}

    narrow = client.create_distill(make_draft(relations=refines))
    other_form = FormDeclaration("note", "A note.", (FieldDeclaration("text", "Text.", FieldKind.TEXT),))
    app.register(OTHER)
    client.register_distill_form(OTHER.schema_ref, other_form.to_dict())
    elsewhere = {
        "schema_ref": OTHER.schema_ref,
        "form_ref": other_form.form_ref,
        "claim": {"text": "x"},
        "relations": {"entries": [{"kind": "related", "target": broad}]},
    }

    assert client.distill(str(narrow["entry_id"]))["entry"]["relations"]["entries"] == refines["entries"]
    _refused(lambda: client.create_distill(elsewhere), 400, "another schema")


# How evidence moves after a version froze it.

NEW = make_experience(7, "run-c", "qwen3", "triton", "keep", 103.0)
NEW_REVERT = make_experience(7, "run-c", "qwen3", "triton", "revert", 94.0)


def _described(experience: Experience) -> dict[str, Any]:
    return {"id": experience.id, "summary": experience.change["summary"]}


@pytest.mark.parametrize(
    ("draft", "move", "status", "current", "detail"),
    [
        (make_draft(), lambda client: None, "consistent", (3, 1, 0, 0, 2), {}),
        (make_draft(), lambda client: client.write(NEW_REVERT), "strengthened", (4, 1, 0, 0, 3), {}),
        (make_draft(), lambda client: client.exclude(IDS[2], reason="noisy"), "strengthened", (3, 0, 0, 0, 2), {}),
        (
            make_draft(),
            lambda client: client.write(make_experience(7, "run-c", "llama", "triton", "keep", 130.0)),
            "consistent",
            (3, 1, 0, 0, 2),
            {},
        ),
        (
            make_draft(),
            lambda client: client.write(NEW),
            "contradicted",
            (3, 2, 0, 0, 3),
            {"gained_contradicting": [_described(NEW)]},
        ),
        (
            make_draft(),
            lambda client: client.exclude(IDS[0], reason="noisy"),
            "weakened",
            (2, 1, 0, 0, 2),
            {"lost_supporting": [_described(EXPERIENCES[0])]},
        ),
        (
            make_draft(check=None, annotations={IDS[2]: "contradict"}),
            lambda client: client.write(NEW),
            "undetermined",
            (3, 1, 0, 1, 3),
            {"gained_undetermined": [_described(NEW)]},
        ),
        (
            make_draft(remove=[IDS[1]], annotations={IDS[2]: "support"}),
            lambda client: client.write(NEW_REVERT),
            "strengthened",
            (4, 0, 0, 0, 3),
            {},
        ),
    ],
    ids=[
        "unchanged",
        "a-new-supporting-member",
        "a-contradicting-member-hidden",
        "a-write-outside-the-scope",
        "a-new-contradicting-member",
        "a-supporting-member-hidden",
        "a-new-member-without-a-check",
        "removed-and-annotated-members-stay-as-written",
    ],
)
def test_a_listing_and_a_read_say_how_each_distills_evidence_moved_since_its_snapshot(
    kb: tuple[ExperienceHTTPService, RemoteClient],
    draft: dict[str, Any],
    move: Any,
    status: str,
    current: tuple[int, ...],
    detail: dict[str, Any],
) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(draft)["entry_id"])
    move(client)

    item = client.list_distills(SCHEMA.schema_ref)["items"][0]
    found = client.distill(entry_id)["drift"]

    counts = dict(zip(("supports", "contradicts", "irrelevant", "undetermined", "runs"), current, strict=True))
    attention = status in ("contradicted", "weakened", "undetermined")
    assert (item["evidence_status"], item["needs_attention"], item["current"]) == (status, attention, counts)
    assert (found["evidence_status"], found["current"]) == (status, counts)
    for kind in ("gained_contradicting", "gained_undetermined", "lost_supporting"):
        assert found[kind] == detail.get(kind, [])


def test_the_changes_inbox_lists_what_needs_attention_until_an_update_takes_the_new_evidence_in(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    app, client = kb
    app.register(OTHER)
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    shaken = str(client.create_distill(make_draft())["entry_id"])
    deleted = str(client.create_distill(make_draft(tags=["copy"]))["entry_id"])
    aiter = {"identity.model": ["qwen3"], "change.attention": ["aiter"]}
    steady = str(client.create_distill(make_draft(scope=aiter))["entry_id"])
    client.exclude(deleted, reason="a duplicate")
    client.write(NEW)

    inbox = client.distill_changes()
    one_schema = client.distill_changes(schema_ref=SCHEMA.schema_ref)
    client.update_distill(shaken, make_draft())
    after = client.distill_changes()

    assert [item["entry_id"] for item in inbox["items"]] == [shaken]
    item = inbox["items"][0]
    assert (item["schema_ref"], item["version"], item["evidence_status"]) == (SCHEMA.schema_ref, 1, "contradicted")
    assert (item["supports"], item["contradicts"]) == (3, 1)
    assert item["current"] == {"supports": 3, "contradicts": 2, "irrelevant": 0, "undetermined": 0, "runs": 3}
    assert item["gained"] == {"supports": 0, "contradicts": 1, "irrelevant": 0, "undetermined": 0}
    assert item["gained_contradicting"] == [_described(NEW)]
    assert inbox["checked"][SCHEMA.schema_ref] == {
        "contradicted": 1,
        "weakened": 0,
        "undetermined": 0,
        "strengthened": 0,
        "consistent": 1,
    }
    assert set(inbox["checked"][OTHER.schema_ref].values()) == {0}
    assert one_schema["items"] == inbox["items"] and list(one_schema["checked"]) == [SCHEMA.schema_ref]
    assert after["items"] == [] and after["checked"][SCHEMA.schema_ref]["consistent"] == 2
    assert client.distill(steady)["drift"]["evidence_status"] == "consistent"
    _refused(lambda: client.distill_changes(schema_ref="schema:sha256:" + "0" * 64), 400, "not registered")


def test_an_older_version_reads_with_its_own_snapshots_drift(kb: tuple[ExperienceHTTPService, RemoteClient]) -> None:
    _, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(make_draft())["entry_id"])
    client.write(NEW_REVERT)
    client.update_distill(entry_id, make_draft())

    assert client.distill(entry_id)["drift"]["evidence_status"] == "consistent"
    assert client.distill(entry_id, version=1)["drift"]["evidence_status"] == "strengthened"


def test_health_and_metrics_report_what_the_last_check_found_without_computing_it(
    kb: tuple[ExperienceHTTPService, RemoteClient],
) -> None:
    app, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    entry_id = str(client.create_distill(make_draft())["entry_id"])
    client.write(NEW)
    unchecked = (client.health()["distills"], app.metrics.render(app.metric_gauges()))

    client.distill_changes()
    checked = (client.health()["distills"][SCHEMA.schema_ref], app.metrics.render(app.metric_gauges()))
    client.exclude(entry_id, reason="superseded")
    client.list_distills(SCHEMA.schema_ref)
    emptied = client.health()["distills"][SCHEMA.schema_ref]

    labels = f'schema_ref="{SCHEMA.schema_ref}"'
    assert unchecked[0] == {
        SCHEMA.schema_ref: {"visible": 1, "deleted": 0, "needs_attention": None, "checked_at": None}
    }
    assert "hyperloom_kb_distill_changes{" not in unchecked[1]
    assert (checked[0]["visible"], checked[0]["needs_attention"]) == (1, 1)
    assert datetime.fromisoformat(checked[0]["checked_at"]).tzinfo is not None
    assert f'hyperloom_kb_distill_changes{{{labels},status="contradicted"}} 1' in checked[1]
    assert f'hyperloom_kb_distill_changes{{{labels},status="consistent"}} 0' in checked[1]
    assert f"hyperloom_kb_distill_checked_timestamp_seconds{{{labels}}}" in checked[1]
    assert 'hyperloom_kb_distill_evidence_seconds_count{operation="check"} 1' in checked[1]
    assert (emptied["visible"], emptied["deleted"], emptied["needs_attention"]) == (0, 1, 0)


# What an operator sees.


def test_distill_writes_and_evidence_time_are_counted_audited_and_routed_by_template(
    kb: tuple[ExperienceHTTPService, RemoteClient], caplog: pytest.LogCaptureFixture
) -> None:
    app, client = kb
    client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
    with caplog.at_level(logging.INFO):
        entry_id = str(client.create_distill(make_draft())["entry_id"])
        client.update_distill(entry_id, make_draft())
        client.preview_distill(make_draft())
        client.explore(SCHEMA.schema_ref)
        client.distill(entry_id)
        audit = _logged(caplog, "audit", action="distill_update")
        request = _logged(caplog, "http_request", route="/v1/distill/entries/{entry_id}", method="PUT")
    with app._database.transaction() as connection:
        kinds = [
            row["record_kind"]
            for row in connection.execute(
                "SELECT record_kind FROM writes WHERE experience_id = %s ORDER BY write_id", (entry_id,)
            )
        ]
    metrics = app.metrics.render(app.metric_gauges())

    assert (audit["entry_id"], audit["version"], audit["result"]) == (entry_id, 2, "created")
    assert audit["request_id"] == request["request_id"]
    assert kinds == ["distill", "distill"]
    assert 'hyperloom_kb_distill_writes_total{operation="create",result="created"} 1' in metrics
    assert 'hyperloom_kb_distill_writes_total{operation="update",result="created"} 1' in metrics
    for operation in ("create", "update", "preview", "explore"):
        assert f'hyperloom_kb_distill_evidence_seconds_count{{operation="{operation}"}} 1' in metrics
    assert "# TYPE hyperloom_kb_distill_evidence_seconds histogram" in metrics
    assert 'route="/v1/distill/entries/{entry_id}"' in metrics and entry_id not in metrics
    assert f'hyperloom_kb_distill_entries{{schema_ref="{SCHEMA.schema_ref}",state="visible"}} 1' in metrics
