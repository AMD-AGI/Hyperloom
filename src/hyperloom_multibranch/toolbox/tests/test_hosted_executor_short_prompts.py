# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A succeeded test that ran the short-prompt consistency check carries what its landed outputs say (#126).

The executor is the real one over the hosted backend and the local sandbox provider; the stand-in runner leaves the
outputs of a check whose one process, the test's own server, behaves as the point says (``fake_short_prompts``) in the
files the bundle signs. What is read is what the bundle landed; the toolbox's pool compares the tests of one recipe.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.consistency import (
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
)
from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.mcp_tools import jsonable
from hyperloom_multibranch.toolbox.short_prompt_pool import POOL_FILE, ShortPromptPool
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner

PLAN = ShortPromptPlan("short-prompts-test", "a" * 64, trigger=("purpose final",))


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


class _Planner(PointPlanner):
    """Plans every request to one point file and the short-prompt check ``check`` names."""

    def __init__(self, point: Path, check: ShortPromptPlan | None) -> None:
        super().__init__(point)
        self.check = check

    def plan(self, request, parent) -> AcceptedMeasurement:  # type: ignore[no-untyped-def]
        return replace(super().plan(request, parent), short_prompts=self.check)


class _Session:
    """One toolbox over one hosted executor, that runs several tests one after another against the same session."""

    def __init__(self, tmp_path: Path, rig: Rig, check: ShortPromptPlan | None) -> None:
        self.point = tmp_path / "planned.yaml"
        self.planner = _Planner(self.point, check)
        self.executor = HostedExecutor(rig.backend, tmp_path / "session")
        pool = ShortPromptPool(tmp_path / "toolbox" / POOL_FILE)
        self.tb = build(tmp_path, executor=self.executor, planner=self.planner, policy=POLICY, short_prompt_pool=pool)
        self.tb.toolbox.trace_root = tmp_path / "session"
        self.tmp_path, self.rig, self.count = tmp_path, rig, 0
        self.started = False

    async def run(
        self,
        behaviour: str | None,
        *,
        check: ShortPromptPlan | None = None,
        server_args: str | None = None,
        **fake: object,
    ):
        fake = {"fake_accuracy": "0.64", **fake, **({} if behaviour is None else {"fake_short_prompts": behaviour})}
        point_yaml(self.tmp_path, self.point.name, **fake)
        self.planner.check = check or self.planner.check
        if not self.started:
            await self.tb.toolbox.start()
            self.started = True
        self.count += 1
        args = server_args or f"--max-num-seqs {256 + self.count}"
        request_id = await self.tb.toolbox.submit_test(req(server_args=args))
        await asyncio.wait_for(self.tb.toolbox.drain(), 60)
        entry = self.tb.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None
        return TestResult.from_document(entry.outcome["result"])


async def test_a_succeeded_test_carries_the_check_of_the_outputs_its_bundle_landed(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run("good")

    assert result.validity is Validity.SUCCEEDED
    check = result.short_prompt_consistency
    assert check is not None and check.status == SHORT_PROMPTS_COMPLETE
    assert check.processes == (0,) and check.trigger == ("purpose final",)
    assert check.inconsistency is None and check.pooled.count == 1, "its own server alone compares nothing"
    assert "reference" not in check.to_document()
    landed = tmp_path / "session" / result.output_dir / check.outputs
    assert landed.is_file()
    archived = ResultManifest.from_bytes((Path(rig.record_archive(result.request_id)) / "manifest.json").read_bytes())
    assert check.outputs in {entry.path for entry in archived.files}, "the signature covers the outputs"


async def test_a_child_test_is_held_to_its_own_recipes_pool_and_the_record_survives_reverification(
    tmp_path: Path, rig
) -> None:
    session = _Session(tmp_path, rig, PLAN)
    parent = await session.run("good")

    child = await session.run("bad")

    assert child.short_prompt_consistency.pooled.count == 1, "another recipe, another pool"
    assert not any((tmp_path / "session").rglob("short_prompt_refs")), "no parent outputs are kept for children"
    assert session.executor.difference(child) is None, "the record follows from its own bundle"
    assert session.executor.difference(parent) is None


async def test_an_edited_prompt_count_with_untouched_hashes_is_named(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)
    result = await session.run("good")
    check = result.short_prompt_consistency

    edited = replace(result, short_prompt_consistency=replace(check, prompts=63))

    assert edited.manifest_sha256 == result.manifest_sha256
    assert session.executor.difference(edited) == "short_prompt_consistency.prompts: recorded 63, the bundle has 64"


async def test_a_server_that_did_not_answer_leaves_the_check_incomplete_and_the_test_succeeded(
    tmp_path: Path, rig
) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run("fail")

    check = result.short_prompt_consistency
    assert check.status == SHORT_PROMPTS_INCOMPLETE and check.reason == "process 0: did not answer (test)"
    assert result.validity is Validity.SUCCEEDED and result.failure_class is None, "an incomplete check blocks nothing"
    assert check.processes == () and check.inconsistency is None
    assert session.executor.difference(result) is None


async def test_a_runner_that_left_no_outputs_gives_an_incomplete_check_never_a_clean_one(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run(None)

    check = result.short_prompt_consistency
    assert check.status == SHORT_PROMPTS_INCOMPLETE and "no readable short_prompts/outputs.json" in check.reason
    assert check.processes == () and check.inconsistency is None
    assert result.validity is Validity.SUCCEEDED


async def test_a_test_that_asked_for_no_check_carries_none(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, None)

    result = await session.run("good")

    assert result.validity is Validity.SUCCEEDED and result.short_prompt_consistency is None
    assert jsonable(result)["short_prompt_consistency"] is None


async def test_a_test_that_did_not_succeed_makes_no_claim_about_short_prompts(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run("good", fake_mode="fail_test")

    assert result.validity is not Validity.SUCCEEDED and result.short_prompt_consistency is None


async def test_test_result_shows_the_check_with_its_pooled_counts_and_no_reference(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)
    recipe = "--max-num-seqs 300"
    await session.run("good", server_args=recipe)
    result = await session.run("bad", server_args=recipe)

    shown = jsonable(result)["short_prompt_consistency"]

    assert shown["status"] == SHORT_PROMPTS_COMPLETE and shown["processes"] == [0]
    assert set(shown) == set(result.short_prompt_consistency.to_document()) and "reference" not in shown
    counts = shown["pooled"]["counts"]
    assert counts["mismatches"][1] > 0 and counts["first_difference"]["process"] == 1
    assert counts["first_difference"]["position"] == 1, "a defect on short prompts shows from the second token"
    assert ShortPromptPlan.parse(PLAN.to_document(), "x") == PLAN


async def test_the_tests_of_one_recipe_pool_their_own_servers_and_the_pool_survives_a_toolbox_restart(
    tmp_path: Path, rig
) -> None:
    session = _Session(tmp_path, rig, PLAN)
    recipe = "--max-num-seqs 300"

    first = await session.run("good", server_args=recipe)
    second = await session.run("bad", server_args=recipe)
    await session.tb.toolbox.close()
    restarted = _Session(tmp_path, rig, PLAN)
    restarted.count = session.count
    third = await restarted.run("good", server_args=recipe)
    other = await restarted.run("bad", server_args="--max-num-seqs 301")

    own = [r.short_prompt_consistency for r in (first, second, third)]
    assert [c.processes for c in own] == [(0,), (0,), (0,)]
    pooled = [c.pooled for c in own]
    assert [p.count for p in pooled] == [1, 2, 3], "every test of the recipe adds its server, across the restart"
    assert pooled[2].processes == (f"{first.output_dir}#0", f"{second.output_dir}#0", f"{third.output_dir}#0")
    assert pooled[0].inconsistency is None, "one process compares nothing; it is not read as 0"
    assert pooled[1].inconsistency > 0 and [n > 0 for n in pooled[2].counts.mismatches] == [False, True, False]
    assert own[2].inconsistency == pooled[2].inconsistency, "the gate's number is the pooled one"
    assert other.short_prompt_consistency.pooled.count == 1, "another recipe has its own pool"
    assert restarted.executor.difference(third) is None, "the pooled part is not claimed by one bundle"
