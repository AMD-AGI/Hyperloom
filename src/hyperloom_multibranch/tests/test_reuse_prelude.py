# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--reuse-prelude``: a session rooted on the child session an earlier session's PRELUDE rooted on, without starting
a child of its own, and the earlier sessions the command line refuses to reuse before it creates anything."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.contracts.measurement import Purpose
from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.pins import IMAGE_DIGEST_VAR
from hyperloom_multibranch import cli
from hyperloom_multibranch.contracts.notebook import Role
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.local_peers import AGENTX_ENV
from hyperloom_multibranch.scheduling.prelude import reused_prelude
from hyperloom_multibranch.scheduling.settings import ReusedPrelude, RunSettings
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.scheduling.tests.child_sessions import child_with_patch
from hyperloom_multibranch.scheduling.tests.peer_fakes import ScriptedCompletion
from hyperloom_multibranch.tests.test_skeleton import (
    CONVERGED_REASON,
    SETTINGS,
    WORKLOAD,
    Stage,
    _decisions,
    _reply,
    _stage,
)

#: :data:`~hyperloom_multibranch.tests.test_skeleton.WORKLOAD` as the command line spells it.
WORKLOAD_ARGS = [
    "--model", WORKLOAD.model, "--framework", "sglang", "--tp", "1", "--isl", "1024", "--osl", "1024",
    "--conc", "16", "--precision", "bf16", "--workload", "inferencex", "--gpu-type", "mi355x",
]  # fmt: skip
#: An opening judge reply that ends EXPLORE at once, so a session runs PRELUDE, the baseline and FINAL on the root.
SOFT_STOP = _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})
OTHER_IMAGE = "sha256:" + "c" * 64


@pytest.fixture(autouse=True)
def command_line_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eight cards, no AgentX switch or benchmark YAML, and no knowledge targets for CLOSE."""

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    for name in (AGENTX_ENV, BENCHMARK_CONFIG_ENV, ARCHIVE_ENV):
        monkeypatch.delenv(name, raising=False)


def _root_recipe(layout: SessionLayout) -> Recipe:
    return Recipe.from_document(json.loads(layout.root_recipe.read_text()))


async def _session(tmp_path: Path, *, child: Path | None, exit_code: int = 0) -> Stage:
    stage = _stage(tmp_path, child=child, exit_code=exit_code)
    stage.judge = ScriptedCompletion([SOFT_STOP])
    await stage.run()
    return stage


async def _earlier(tmp_path: Path) -> Stage:
    """A session in ``tmp_path`` that rooted on a completed PRELUDE child and ran its course."""

    return await _session(tmp_path / "earlier", child=child_with_patch(tmp_path))


def _rooted(stage: Stage) -> dict[str, object]:
    [rooted] = [row for row in _decisions(stage) if row["kind"] == "prelude" and row["reason"] == "root committed"]
    return rooted


async def test_a_session_reusing_an_earlier_prelude_starts_no_child_and_measures_its_own_baseline(
    tmp_path: Path,
) -> None:
    earlier = await _earlier(tmp_path)
    earlier_root = _root_recipe(earlier.layout)
    stage = _stage(tmp_path / "later", child=None, exit_code=1)
    reused = reused_prelude(earlier.layout.root, WORKLOAD, earlier_root.base_image_digest)
    stage.settings = replace(SETTINGS, reuse_prelude=reused)
    stage.judge = ScriptedCompletion([SOFT_STOP])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.GLOBAL_CONVERGED, "", 0)
    assert stage.processes.argv == ()
    child_session = _rooted(earlier)["detail"]["child_session"]
    assert reused == ReusedPrelude(str(earlier.layout.root), str(child_session), earlier_root.digest())
    assert _root_recipe(stage.layout).digest() == earlier_root.digest()
    assert (await stage.repo.read_recipe(await stage.root())).digest() == earlier_root.digest()
    prelude = [row for row in _decisions(stage) if row["kind"] == "prelude"]
    assert "child started" not in [row["reason"] for row in prelude]
    [reuse] = [row for row in prelude if row["reason"] == "child reused"]
    assert reuse["detail"] == {
        "session": str(earlier.layout.root),
        "child_session": child_session,
        "root_recipe_digest": earlier_root.digest(),
    }
    rooted = _rooted(stage)["detail"]
    assert (rooted["child_session"], rooted["child_gpu_seconds_upper_bound"]) == (child_session, 0.0)
    [baseline] = [row for row in prelude if row["reason"] == "baseline established"]
    records = (await stage.handle(Role.OBSERVER).snapshot()).records
    measured = [record.request_id for record in records if record.request.purpose is Purpose.BASELINE and record.result]
    assert baseline["detail"]["request_ids"] and set(baseline["detail"]["request_ids"]) <= set(measured)
    manifest = json.loads(stage.layout.manifest.read_text())
    assert manifest["settings"]["reuse_prelude"]["session"] == str(earlier.layout.root)


def _refused(
    argv: list[str], workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> str:
    """Run the command on ``argv`` in ``workspace``, check it exits two and creates nothing; return its stderr."""

    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *argv])
    assert exc.value.code == cli.EXIT_USAGE
    assert not workspace.exists()
    return capsys.readouterr().err


def test_a_directory_that_does_not_exist_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-session"
    monkeypatch.setenv(IMAGE_DIGEST_VAR, OTHER_IMAGE)

    err = _refused([*WORKLOAD_ARGS, "--reuse-prelude", str(missing)], tmp_path / "workspace", monkeypatch, capsys)

    assert f"{missing} is not a directory" in err


async def test_a_session_whose_prelude_committed_no_root_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failed = await _session(tmp_path / "failed", child=None, exit_code=1)
    monkeypatch.setenv(IMAGE_DIGEST_VAR, OTHER_IMAGE)

    argv = [*WORKLOAD_ARGS, "--reuse-prelude", str(failed.layout.root)]
    err = _refused(argv, tmp_path / "workspace", monkeypatch, capsys)

    assert f"{failed.layout.root} holds no completed PRELUDE" in err


def _replaced(args: list[str], option: str, value: str) -> list[str]:
    at = args.index(option)
    return [*args[: at + 1], value, *args[at + 2 :]]


@pytest.mark.parametrize(
    ("option", "value", "named"),
    [
        ("--conc", "32", "conc 16, not 32"),
        ("--precision", "fp8", "precision 'bf16', not 'fp8'"),
        ("--model", "/models/Qwen3-8B", "model '/models/Llama-3-8B', not '/models/Qwen3-8B'"),
    ],
)
async def test_an_earlier_prelude_of_another_workload_exits_two_naming_what_differs(
    option: str,
    value: str,
    named: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    earlier = await _earlier(tmp_path)
    monkeypatch.setenv(IMAGE_DIGEST_VAR, _root_recipe(earlier.layout).base_image_digest)

    argv = [*_replaced(WORKLOAD_ARGS, option, value), "--reuse-prelude", str(earlier.layout.root)]
    err = _refused(argv, tmp_path / "workspace", monkeypatch, capsys)

    assert "ran another workload" in err and named in err


async def test_an_earlier_prelude_of_an_agentx_workload_is_not_reused_for_an_inferencex_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    earlier = await _earlier(tmp_path)
    monkeypatch.setenv(IMAGE_DIGEST_VAR, _root_recipe(earlier.layout).base_image_digest)
    agentx = _replaced(WORKLOAD_ARGS, "--workload", "agentx")
    for option in ("--isl", "--osl"):
        at = agentx.index(option)
        agentx = [*agentx[:at], *agentx[at + 2 :]]

    argv = [*agentx, "--reuse-prelude", str(earlier.layout.root)]
    err = _refused(argv, tmp_path / "workspace", monkeypatch, capsys)

    assert "ran another workload" in err and "inferencex" in err


async def test_an_earlier_prelude_on_another_image_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    earlier = await _earlier(tmp_path)
    monkeypatch.setenv(IMAGE_DIGEST_VAR, OTHER_IMAGE)

    argv = [*WORKLOAD_ARGS, "--reuse-prelude", str(earlier.layout.root)]
    err = _refused(argv, tmp_path / "workspace", monkeypatch, capsys)

    assert f"ran on image {_root_recipe(earlier.layout).base_image_digest}" in err and OTHER_IMAGE in err


async def test_a_reusable_earlier_prelude_reaches_the_session_s_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    earlier = await _earlier(tmp_path)
    digest = _root_recipe(earlier.layout).base_image_digest
    monkeypatch.setenv(IMAGE_DIGEST_VAR, digest)
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path))
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)
    runs: list[RunSettings] = []

    def run_session(settings: RunSettings, peer_factory: object, services: object) -> int:
        runs.append(settings)
        return 0

    monkeypatch.setattr(cli, "load_peers", lambda workload: None)
    monkeypatch.setattr(cli, "session_services", lambda recipe_kb: None)
    monkeypatch.setattr(cli, "run_session", run_session)

    assert cli.main(["run", *WORKLOAD_ARGS, "--reuse-prelude", str(earlier.layout.root)]) == 0

    [settings] = runs
    assert settings.reuse_prelude == reused_prelude(earlier.layout.root, WORKLOAD, digest)
    fresh = cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS]))
    assert replace(settings, reuse_prelude=None) == fresh
