# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The review's state and page, as the session writes them and a person decides on them."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.scheduling.review import (
    APPROVED,
    PENDING,
    REJECTED,
    REVIEW_FILE,
    REVIEW_PAGE,
    decide,
    load,
    request,
    show,
    wait,
)

NOW = 1793500000.0


def _map_and_dossier(tmp_path: Path) -> tuple[Path, Path]:
    map_file = tmp_path / "pass" / "directions.json"
    map_file.parent.mkdir(parents=True)
    map_file.write_text(json.dumps({"directions": [{"id": "m-attn", "status": "open", "title": "attention | decode"}]}))
    dossier = tmp_path / "pass" / "docs" / "dossier" / "qwen3-8b"
    dossier.mkdir(parents=True)
    (dossier / "README.md").write_text("# dossier\n")
    (dossier / "07-profile.md").write_text("profile\n")
    return map_file, dossier


def test_a_request_writes_the_page_and_a_pending_state(tmp_path: Path) -> None:
    map_file, dossier = _map_and_dossier(tmp_path)
    directory, session = tmp_path / "session" / "MB-review", tmp_path / "session"

    state = request(directory, session, map_file, dossier, "dossier pass", NOW)

    assert state == load(directory)
    assert (state["status"], state["map"], state["dossier"], state["made_by"]) == (
        PENDING,
        str(map_file),
        str(dossier),
        "dossier pass",
    )
    assert set(state["requested_digests"]) == {
        str(map_file),
        str(dossier / "07-profile.md"),
        str(dossier / "README.md"),
    }
    page = (directory / REVIEW_PAGE).read_text()
    assert "| `m-attn` | open | attention / decode |" in page
    assert f"review --session {session} approve" in page and "07-profile.md, README.md" in page


def test_an_approval_names_the_files_the_review_changed_and_a_second_decision_is_refused(tmp_path: Path) -> None:
    map_file, dossier = _map_and_dossier(tmp_path)
    directory = tmp_path / "MB-review"
    request(directory, tmp_path, map_file, dossier, "dossier pass", NOW)
    (dossier / "07-profile.md").write_text("profile, corrected\n")

    assert decide(directory, True, "alice", "looks right", NOW + 60) == []

    state = load(directory)
    assert state is not None
    assert (state["status"], state["by"], state["note"]) == (APPROVED, "alice", "looks right")
    assert state["changed"] == [str(dossier / "07-profile.md")]
    assert state["decided_at_iso"] == "2026-11-01T02:27:40Z"
    assert decide(directory, False, "bob", "", NOW + 90) == ["the review is already approved"]
    assert show(directory) is not None and "Status: approved by alice" in str(show(directory))


def test_a_decision_with_no_review_pending_is_refused(tmp_path: Path) -> None:
    assert decide(tmp_path / "MB-review", True, "alice", "", NOW) == [
        f"no review is pending in {tmp_path / 'MB-review'}"
    ]
    assert show(tmp_path / "MB-review") is None


def test_a_review_with_no_map_says_each_arbor_maps_its_own(tmp_path: Path) -> None:
    request(tmp_path / "MB-review", tmp_path, None, None, "generated from the root's shares", NOW)

    page = (tmp_path / "MB-review" / REVIEW_PAGE).read_text()
    assert "none could be made: each Arbor maps its own" in page and "**Dossier:** none" in page


async def test_the_wait_ends_once_a_person_decides(tmp_path: Path) -> None:
    map_file, dossier = _map_and_dossier(tmp_path)
    directory = tmp_path / "MB-review"
    request(directory, tmp_path, map_file, dossier, "dossier pass", NOW)
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 3:
            decide(directory, False, "alice", "wrong directions", NOW)

    decided = await wait(directory, sleep, poll_seconds=15.0)

    assert decided["status"] == REJECTED and slept == [15.0, 15.0, 15.0]
    assert json.loads((directory / REVIEW_FILE).read_text())["note"] == "wrong directions"
