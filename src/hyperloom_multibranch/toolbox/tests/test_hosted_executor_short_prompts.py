# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A succeeded test that ran the short-prompt consistency check carries what its landed outputs say (#126).

The executor is the real one over the hosted backend and the local sandbox provider; the stand-in runner leaves the
outputs a check whose processes behave as the point says would (``fake_short_prompts``) in the files the bundle signs.
What is read is what the bundle landed, and the parent's outputs come from the session's reference store.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.consistency import (
    REFERENCE_ABSENT,
    REFERENCE_COMPARED,
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
)
from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.toolbox.hosted_executor import REFERENCES_DIR, HostedExecutor
from hyperloom_multibranch.toolbox.mcp_tools import jsonable
from hyperloom_multibranch.toolbox.short_prompt_cap import CAP_MARGIN_FRACTION, CAP_MARGIN_SEC
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner

PARENT_RECIPE = "sha256:" + "2" * 64
CHILD_RECIPE = "sha256:" + "1" * 64
PLAN = ShortPromptPlan(
    "short-prompts-test", "a" * 64, trigger=("purpose final",), recipe=CHILD_RECIPE, parent_recipe=PARENT_RECIPE
)
GOOD = ["good"] * 6


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
        self.tb = build(tmp_path, executor=self.executor, planner=self.planner, policy=POLICY)
        self.tmp_path, self.rig, self.count = tmp_path, rig, 0
        self.started = False

    async def run(self, behaviours: list[str] | None, *, check: ShortPromptPlan | None = None, **fake: object):
        fake = {"fake_accuracy": "0.64", **fake, **({} if behaviours is None else {"fake_short_prompts": behaviours})}
        point_yaml(self.tmp_path, self.point.name, **fake)
        self.planner.check = check or self.planner.check
        if not self.started:
            await self.tb.toolbox.start()
            self.started = True
        self.count += 1
        request_id = await self.tb.toolbox.submit_test(req(server_args=f"--max-num-seqs {256 + self.count}"))
        await asyncio.wait_for(self.tb.toolbox.drain(), 60)
        entry = self.tb.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None
        return TestResult.from_document(entry.outcome["result"])


async def test_a_succeeded_test_carries_the_check_of_the_outputs_its_bundle_landed(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run(GOOD)

    assert result.validity is Validity.SUCCEEDED
    check = result.short_prompt_consistency
    assert check is not None and check.status == SHORT_PROMPTS_COMPLETE
    assert check.processes == (0, 1, 2, 3, 4, 5) and check.trigger == ("purpose final",)
    assert check.self_consistency is not None and check.self_consistency.mismatches == (0,) * 6
    assert check.reference.status == REFERENCE_ABSENT, "the first test of a recipe has no parent output to compare with"
    landed = tmp_path / "session" / result.output_dir / check.outputs
    assert landed.is_file()
    archived = ResultManifest.from_bytes((Path(rig.record_archive(result.request_id)) / "manifest.json").read_bytes())
    assert check.outputs in {entry.path for entry in archived.files}, "the signature covers the outputs"


async def test_a_defective_launch_among_the_candidates_processes_is_counted_and_located(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run(["good", "bad", "good", "good", "bad", "good"])

    own = result.short_prompt_consistency.self_consistency
    assert [bool(n) for n in own.mismatches] == [False, True, False, False, True, False]
    assert (own.first_difference.process, own.first_difference.position) == (1, 1)
    assert result.validity is Validity.SUCCEEDED, "the check states a count, the measurement is still a measurement"


async def test_a_child_test_is_compared_with_its_parents_outputs_and_the_record_survives_reverification(
    tmp_path: Path, rig
) -> None:
    session = _Session(tmp_path, rig, replace(PLAN, recipe=PARENT_RECIPE, parent_recipe="sha256:" + "3" * 64))
    parent = await session.run(GOOD)
    assert (tmp_path / "session" / REFERENCES_DIR).is_dir() and parent.short_prompt_consistency is not None

    child = await session.run(["good", "good", "bad", "good", "good", "good"], check=PLAN)

    reference = child.short_prompt_consistency.reference
    assert reference.status == REFERENCE_COMPARED and reference.request_id == parent.request_id
    assert [bool(n) for n in reference.counts.mismatches] == [False, False, True, False, False, False]
    assert session.executor.difference(child) is None, "the record follows from its own bundle"
    assert session.executor.difference(parent) is None


async def test_an_edited_mismatch_count_with_untouched_hashes_is_named(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)
    result = await session.run(["good", "bad", "good", "good", "bad", "good"])
    own = result.short_prompt_consistency.self_consistency
    cleaned = replace(
        result.short_prompt_consistency, self_consistency=replace(own, mismatches=(0,) * 6, first_difference=None)
    )

    edited = replace(result, short_prompt_consistency=cleaned)

    assert edited.manifest_sha256 == result.manifest_sha256
    assert session.executor.difference(edited) == (
        f"short_prompt_consistency.self_consistency: recorded {cleaned.self_consistency!r}, "
        f"the bundle has {result.short_prompt_consistency.self_consistency!r}"
    )


async def test_a_process_that_did_not_come_up_leaves_the_check_incomplete_and_nothing_stored(
    tmp_path: Path, rig
) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run(["good", "fail", "good", "good", "good", "good"])

    check = result.short_prompt_consistency
    assert check.status == SHORT_PROMPTS_INCOMPLETE and "process 1: did not come up (test)" in check.reason
    assert not any((tmp_path / "session" / REFERENCES_DIR).glob("*.json")), "an incomplete check is no reference"
    assert session.executor.difference(result) is None


async def test_a_runner_that_left_no_outputs_gives_an_incomplete_check_never_a_clean_one(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run(None)

    check = result.short_prompt_consistency
    assert check.status == SHORT_PROMPTS_INCOMPLETE and "no readable short_prompts/outputs.json" in check.reason
    assert check.processes == () and check.self_consistency is None and check.reference.status == REFERENCE_ABSENT


async def test_a_test_that_asked_for_no_check_carries_none(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, None)

    result = await session.run(GOOD)

    assert result.validity is Validity.SUCCEEDED and result.short_prompt_consistency is None
    assert jsonable(result)["short_prompt_consistency"] is None


async def test_a_test_that_did_not_succeed_makes_no_claim_about_short_prompts(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)

    result = await session.run(GOOD, fake_mode="fail_test")

    assert result.validity is not Validity.SUCCEEDED and result.short_prompt_consistency is None


async def test_test_result_shows_the_check_with_its_counts_and_its_reference(tmp_path: Path, rig) -> None:
    session = _Session(tmp_path, rig, PLAN)
    result = await session.run(["good", "bad", "good", "good", "good", "good"])

    shown = jsonable(result)["short_prompt_consistency"]

    assert shown["status"] == SHORT_PROMPTS_COMPLETE and shown["reference"]["status"] == REFERENCE_ABSENT
    assert (
        shown["self_consistency"]["mismatches"][1] > 0 and shown["self_consistency"]["first_difference"]["process"] == 1
    )
    assert ShortPromptPlan.parse(PLAN.to_document(), "x") == PLAN


async def test_a_check_is_capped_by_the_pack_until_a_boot_is_measured_then_by_the_sessions_measured_times(
    tmp_path: Path, rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _Session(tmp_path, rig, PLAN)
    caps: list[float] = []
    run_test = rig.backend.run_test

    def recording(call):  # type: ignore[no-untyped-def]
        caps.append(call.short_prompts.phase_cap_sec)
        return run_test(call)

    monkeypatch.setattr(rig.backend, "run_test", recording)

    first = await session.run(GOOD)
    await session.run(GOOD)

    boot = max(80.0, first.cold_start_seconds or 0.0)  # the stand-in's processes each boot in 80 s and answer in 20 s
    assert caps == [PLAN.phase_cap_sec, round(6 * (boot + 20.0) * (1 + CAP_MARGIN_FRACTION) + CAP_MARGIN_SEC, 1)]
