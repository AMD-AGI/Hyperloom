# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``points`` command: sessions at several concs on one machine, mapped, reviewed, then run where approved."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hyperloom_multibranch import cli
from hyperloom_multibranch.points import (
    REVIEW_PAGE,
    Decision,
    Stage,
    decide,
    map_points,
    place_points,
    reviewed_points,
    run_approved,
    shared_args,
)
from hyperloom_multibranch.scheduling.settings import AGENT_PORT

#: A stand-in for ``hyperloom-multibranch``: each ``run`` makes a session directory under its ``USER_DATA_PATH`` and
#: records there how it was started; one that stops after its map keeps a map and a dossier the way a session does.
#: ``FAIL_CONC`` names a conc whose map it keeps none of, exiting 1.
FAKE_SESSION = r"""
import json, os, sys, time
from pathlib import Path

argv = sys.argv[1:]
conc = argv[argv.index("--conc") + 1]
session = Path(os.environ["USER_DATA_PATH"]) / "Qwen3-8B" / f"MB-{time.time_ns()}-{conc}"
session.mkdir(parents=True)
seen = {name: os.environ.get(name) for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "PORT", "USER_DATA_PATH")}
(session / "started.json").write_text(json.dumps({"argv": argv, "env": seen}))
if "--stop-after-map" in argv:
    if conc == os.environ.get("FAIL_CONC"):
        sys.exit(1)
    kept = session / "MB-agent-map"
    (kept / "dossier" / "qwen3-8b").mkdir(parents=True)
    (kept / "dossier" / "qwen3-8b" / "README.md").write_text("# dossier\n")
    directions = [{"id": f"m-c{conc}", "kind": "increment", "title": f"what C{conc} needs", "rank": 1, "ceiling": "4%"}]
    (kept / "directions.json").write_text(json.dumps({"directions": directions}))
    summary = {"conc": int(conc), "made_by": "dossier pass", "pass": "mapped", "map": str(kept / "directions.json"),
               "dossier": str(kept / "dossier" / "qwen3-8b")}
    (kept / "agent-map.json").write_text(json.dumps(summary))
"""
RUN_ARGS = ["--model", "/models/Qwen3-8B", "--autonomous-agent", "/agents/arbor"]


@pytest.fixture
def command(tmp_path: Path) -> tuple[str, ...]:
    script = tmp_path / "fake_session.py"
    script.write_text(FAKE_SESSION, encoding="utf-8")
    return (sys.executable, str(script))


def _started(session: Path | None) -> dict:
    assert session is not None
    return json.loads((session / "started.json").read_text())


def test_points_on_one_machine_share_its_cards_and_take_ports_and_directories_of_their_own(tmp_path: Path) -> None:
    c8, c64 = place_points([8, 64], tuple(range(8)), tmp_path, None)

    assert (c8.cards, c64.cards) == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert c8.port != c64.port and c8.agent_port == AGENT_PORT and c64.agent_port > c8.agent_port
    assert (c8.root, c64.root) == (tmp_path / "c8", tmp_path / "c64")
    assert [point.cards for point in place_points([8, 64], tuple(range(8)), tmp_path, 2)] == [(0, 1), (2, 3)]


@pytest.mark.parametrize(
    ("concs", "cards", "per_point", "match"),
    [
        ([], 8, None, "at least one"),
        ([8, 8], 8, None, "once"),
        ([8, 64], 1, None, "do not give"),
        ([8], 4, 6, "6 each"),
    ],
)
def test_points_that_repeat_or_that_the_cards_do_not_go_round_are_refused(
    tmp_path: Path, concs: list[int], cards: int, per_point: int | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        place_points(concs, tuple(range(cards)), tmp_path, per_point)


def test_the_shared_run_arguments_may_not_set_what_each_point_s_stage_sets() -> None:
    assert shared_args(RUN_ARGS) == RUN_ARGS
    with pytest.raises(ValueError, match="--agent-port, --conc"):
        shared_args([*RUN_ARGS, "--conc", "8", "--agent-port=32100"])


async def test_each_point_maps_on_its_own_cards_port_and_user_data_and_one_page_reviews_them_all(
    tmp_path: Path, command: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    points = place_points([8, 64], tuple(range(8)), tmp_path, None)

    exited = await map_points(points, RUN_ARGS, command)

    assert exited == [0, 0]
    reviewed = reviewed_points(tmp_path)
    assert [point.conc for point in reviewed] == [8, 64]
    starts = [_started(point.session) for point in reviewed]
    assert [start["env"]["ROCR_VISIBLE_DEVICES"] for start in starts] == ["0,1,2,3", "4,5,6,7"]
    assert [start["env"]["HIP_VISIBLE_DEVICES"] for start in starts] == [None, None]
    assert len({start["env"]["PORT"] for start in starts}) == 2
    assert [start["env"]["USER_DATA_PATH"] for start in starts] == [str(p.root / "userdata") for p in points]
    for start, point in zip(starts, points):
        assert start["argv"] == [
            "run",
            *RUN_ARGS,
            "--conc",
            str(point.conc),
            "--agent-port",
            str(point.agent_port),
            "--stop-after-map",
        ]
    page = (tmp_path / REVIEW_PAGE).read_text()
    assert "## C8: awaiting review" in page and "what C8 needs" in page and "what C64 needs" in page
    assert f"points approve --dir {tmp_path} --conc 64" in page


async def test_only_approved_points_run_on_their_reviewed_map_and_dossier_and_the_others_say_why_not(
    tmp_path: Path, command: tuple[str, ...]
) -> None:
    await map_points(place_points([8, 16, 64], tuple(range(6)), tmp_path, None), RUN_ARGS, command)
    decide(tmp_path, 8, Decision.APPROVED, "")
    decide(tmp_path, 64, Decision.REJECTED, "the map only tunes GEMM tables")
    mapped = {point.conc: point for point in reviewed_points(tmp_path)}

    outcome = await run_approved(tmp_path, tuple(range(8)), 2, RUN_ARGS, command)

    assert outcome == {8: 0, 16: "not reviewed", 64: "rejected: the map only tunes GEMM tables"}
    [ran] = [session for session in (tmp_path / "c8" / "userdata").glob("*/MB-*") if session != mapped[8].session]
    start = _started(ran)
    summary = mapped[8].summary
    assert start["argv"][-6:] == [
        "--reuse-prelude",
        str(mapped[8].session),
        "--agent-directions",
        summary["map"],
        "--agent-dossier",
        summary["dossier"],
    ]
    assert start["env"]["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert len(list((tmp_path / "c64" / "userdata").glob("*/MB-*"))) == 1
    assert "## C64: rejected" in (tmp_path / REVIEW_PAGE).read_text()


async def test_a_point_that_kept_no_map_fails_its_stage_and_cannot_be_approved_while_the_others_can(
    tmp_path: Path, command: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAIL_CONC", "64")

    exited = await map_points(place_points([8, 64], tuple(range(8)), tmp_path, None), RUN_ARGS, command)

    assert exited == [0, 1]
    assert "No map was kept" in (tmp_path / REVIEW_PAGE).read_text()
    with pytest.raises(ValueError, match="no map to decide on"):
        decide(tmp_path, 64, Decision.APPROVED, "")
    decide(tmp_path, 8, Decision.APPROVED, "")


def test_the_command_line_names_each_points_action_and_a_rejection_needs_a_reason(tmp_path: Path) -> None:
    parser = cli.build_parser()

    mapping = parser.parse_args(
        ["points", "map", "--dir", str(tmp_path), "--conc", "8", "--conc", "64", "--", *RUN_ARGS]
    )
    assert (mapping.points_command, mapping.conc, mapping.run_args) == (Stage.MAP.value, [8, 64], ["--", *RUN_ARGS])
    assert parser.parse_args(["points", "approve", "--dir", str(tmp_path), "--conc", "8"]).points_command == "approve"
    with pytest.raises(SystemExit):
        parser.parse_args(["points", "reject", "--dir", str(tmp_path), "--conc", "8"])
