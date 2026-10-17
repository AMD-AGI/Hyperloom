# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Distills travel with their schema's sync: a push sends each visible Distill's latest version once its evidence is
on the global KB, a pull brings the version the global KB shows, and neither overwrites a version the other side
wrote differently."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from hyperloom_kb import ExperienceHTTPService, RemoteClient
from hyperloom_kb.tests.distills import IDS, PLOTTED, RULE, SCHEMA, make_draft, make_experience, serving
from hyperloom_kb.tests.logs import LOG, LOG_PATH

NOTHING = {"created": 0, "unchanged": 0, "skipped": 0, "held_back": 0}


class _Team(NamedTuple):
    """A global KB, and two KBs syncing with it: ``a`` holds the corpus and the rule form, ``b`` nothing yet."""

    global_app: ExperienceHTTPService
    shared: RemoteClient
    a: RemoteClient
    b: RemoteClient


@contextmanager
def _team(tmp_path: Path) -> Iterator[_Team]:
    with (
        serving(tmp_path / "global", seeded=False) as (global_app, shared),
        serving(tmp_path / "a", global_url=shared.config.base_url) as (_, a),
        serving(tmp_path / "b", global_url=shared.config.base_url, seeded=False) as (_, b),
    ):
        a.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
        yield _Team(global_app, shared, a, b)


def _shown(client: Any) -> dict[str, int]:
    return {str(item["entry_id"]): int(item["version"]) for item in client.list_distills(SCHEMA.schema_ref)["items"]}


def test_a_push_sends_each_visible_distills_latest_version_and_a_pull_brings_it_to_a_teammate(tmp_path: Path) -> None:
    with _team(tmp_path) as (_, shared, a, b):
        rule = str(a.create_distill(make_draft())["entry_id"])
        a.update_distill(rule, make_draft(tags=["attention", "qwen3"]))
        a.put_file(LOG, LOG_PATH)
        plot = {"statement": "Reverts cluster below the baseline.", "plot": LOG.to_dict()}
        a.register_distill_form(SCHEMA.schema_ref, PLOTTED.to_dict())
        plotted = str(a.create_distill(make_draft(form_ref=PLOTTED.form_ref, claim=plot))["entry_id"])
        dropped = str(a.create_distill(make_draft(tags=["copy"]))["entry_id"])
        a.exclude(dropped, reason="a duplicate")

        pushed, again = a.push(), a.push()
        written, on_global = a.distill(rule)["entry"], shared.distill(rule)
        pulled = b.pull(SCHEMA.schema_ref)
        held, files_missing = b.distill(rule), b.missing_files((LOG,))
        back = b.push()
        shown = _shown(b)

    assert (pushed["status"], pushed["created"], pushed["rejected"]) == ("completed", 7, [])
    assert pushed["distills"] == {**NOTHING, "created": 2} and again["distills"] == NOTHING
    # Only the latest version travels, exactly as it was written.
    assert (on_global["versions"], on_global["entry"]) == ([2], written)
    assert (pulled["created"], pulled["distills"]) == (7, {**NOTHING, "created": 2})
    assert shown == {rule: 2, plotted: 1} and files_missing == ()
    assert (held["current_version"], held["drift"]["evidence_status"]) == (2, "consistent")
    # What was pulled is never pushed back.
    assert (back["skipped"], back["distills"]) == (7, NOTHING)


def test_a_pull_keeps_a_later_version_here_and_neither_side_overwrites_a_version_written_differently(
    tmp_path: Path,
) -> None:
    with _team(tmp_path) as (_, _, a, b):
        entry_id = str(a.create_distill(make_draft())["entry_id"])
        a.push()
        b.pull(SCHEMA.schema_ref)
        b.update_distill(entry_id, make_draft(tags=["from b"]))
        a.update_distill(entry_id, make_draft(tags=["from a"]))
        a.push()

        clash, refused = b.pull(SCHEMA.schema_ref), b.push()
        b.update_distill(entry_id, make_draft(tags=["from b", "again"]))
        ahead, settled = b.pull(SCHEMA.schema_ref), b.push()
        caught_up = a.pull(SCHEMA.schema_ref)
        tags = a.distill(entry_id)["entry"]["tags"]

    assert clash["status"] == "completed" and clash["distills"] == NOTHING
    assert [(item["entry_id"], item["version"]) for item in clash["rejected"]] == [(entry_id, 2)]
    assert "differs from the global KB's" in clash["rejected"][0]["detail"]
    assert [(item["entry_id"], item["version"]) for item in refused["rejected"]] == [(entry_id, 2)]
    assert "HTTP 409" in refused["rejected"][0]["detail"]
    assert ahead["distills"] == {**NOTHING, "skipped": 1}
    assert (settled["distills"], settled["rejected"]) == ({**NOTHING, "created": 1}, [])
    assert caught_up["distills"] == {**NOTHING, "created": 1} and tags == ["from b", "again"]


def test_a_teammate_refreshes_a_pulled_distill_from_its_draft_under_the_authors_judgement(tmp_path: Path) -> None:
    with _team(tmp_path) as (_, _, a, b):
        entry_id = str(a.create_distill(make_draft(annotations={IDS[2]: "support"}))["entry_id"])
        a.push()
        b.pull(SCHEMA.schema_ref)
        b.write(make_experience(7, "run-c", "qwen3", "triton", "revert", 94.0))
        refreshed = b.update_distill(entry_id, b.distill(entry_id)["draft"])
        evidence = b.distill(entry_id)["entry"]["evidence"]

    # The member the author annotated as supporting stays supporting, though the check would contradict it.
    assert refreshed["evidence"] == {"supports": 5, "contradicts": 0, "irrelevant": 0, "runs": 3}
    assert evidence["annotated"] == [IDS[2]]


def test_a_push_waits_for_a_distill_to_show_its_latest_version_and_a_pull_shows_the_global_one_again(
    tmp_path: Path,
) -> None:
    with _team(tmp_path) as (_, _, a, b):
        entry_id = str(a.create_distill(make_draft())["entry_id"])
        first = a.create_label(name="first")
        a.update_distill(entry_id, make_draft(tags=["second"]))
        saved = a.restore(str(first["label_id"]))["saved"]
        waiting = a.push()
        a.restore(str(saved["label_id"]))
        sent = a.push()

        b.pull(SCHEMA.schema_ref)
        pulled = b.create_label(name="pulled")
        a.update_distill(entry_id, make_draft(tags=["third"]))
        a.push()
        b.pull(SCHEMA.schema_ref)
        b.restore(str(pulled["label_id"]))
        restored = _shown(b)
        b.pull(SCHEMA.schema_ref)
        pulled_again = _shown(b)

    assert waiting["distills"] == NOTHING and sent["distills"] == {**NOTHING, "created": 1}
    assert restored == {entry_id: 2} and pulled_again == {entry_id: 3}


def test_a_global_kb_older_than_distills_still_syncs_experiences_and_takes_distills_once_upgraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _team(tmp_path) as (global_app, _, a, b):
        health = global_app.health
        # A global KB older than Distills reports none in its health.
        monkeypatch.setattr(global_app, "health", lambda: {k: v for k, v in health().items() if k != "distills"})
        entry_id = str(a.create_distill(make_draft())["entry_id"])
        old_push, old_pull = a.push(), b.pull(SCHEMA.schema_ref)
        before_upgrade = _shown(b)
        monkeypatch.undo()
        upgraded_push, upgraded_pull = a.push(), b.pull(SCHEMA.schema_ref)
        after_upgrade = _shown(b)

    held_back = {**NOTHING, "held_back": 1}
    assert (old_push["status"], old_push["created"], old_push["distills"]) == ("completed", 7, held_back)
    assert (old_pull["status"], old_pull["created"], old_pull["distills"]) == ("completed", 7, NOTHING)
    assert before_upgrade == {}
    assert upgraded_push["distills"] == {**NOTHING, "created": 1}
    assert upgraded_pull["distills"] == {**NOTHING, "created": 1} and after_upgrade == {entry_id: 1}
