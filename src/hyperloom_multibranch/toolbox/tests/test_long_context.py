# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's long-context check: which tests run it, how it is switched on, and what the result records."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.long_context import (
    LONG_CONTEXT_COMPLETE,
    LONG_CONTEXT_INCOMPLETE,
    PROMPT_ERROR,
    PROMPT_NOT_RUN,
    PROMPT_OK,
)
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE
from hyperloom.measure.long_context.rounds import run_long_context
from hyperloom.measure.long_context.tests.support import LENGTH, FakeServer, prompt_set, staged_plan
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.short_prompts.rounds import Serving
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.leased_toolbox import with_long_context
from hyperloom_multibranch.toolbox.long_context import (
    LONG_CONTEXT_ENV,
    LongContextNotStaged,
    LongContextSetup,
    decide,
    environment_long_context,
    first_difference,
    long_context_of,
    rederive,
    staged_long_context,
)
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_short_prompts import PARENT, _accepted, _Planner

SETUP = LongContextSetup("long-context-test", "a" * 64, LENGTH)
NIAH = ("niah_single", "niah_multikey")


class _Host:
    def __init__(self, server: FakeServer) -> None:
        self.server = server

    async def start(self, index: int, deadline: float) -> Serving:
        return Serving(self.server.base_url)

    async def stop(self) -> None:
        return None

    def should_stop(self) -> bool:
        return False


def _landed(tmp_path: Path, *, served: int | None = 4096, **server: object) -> tuple[Path, object]:
    """The output directory a runner leaves after a check whose server behaves as ``server`` says."""
    plan = staged_plan(tmp_path / "cache")
    with FakeServer(**server) as fake:  # type: ignore[arg-type]
        asyncio.run(run_long_context(plan, prompt_set(), _Host(fake), tmp_path / "out", served_context=served))
    return tmp_path / "out", plan


def _outputs(directory: Path) -> dict:
    return json.loads((directory / LONG_CONTEXT_DIR / OUTPUTS_FILE).read_text())


def _write(directory: Path, document: dict) -> None:
    (directory / LONG_CONTEXT_DIR / OUTPUTS_FILE).write_text(json.dumps(document))


# --- when ------------------------------------------------------------------------------------------------------------


def test_a_request_that_forces_the_check_gets_the_plan_the_setup_names_and_says_why() -> None:
    plan = decide(req(long_context_check=True), SETUP)

    assert plan is not None and (plan.prompt_set, plan.prompts_sha256, plan.length) == (*SETUP.__dict__.values(),)
    assert (plan.max_new_tokens, plan.concurrency, plan.trigger) == (128, 4, ("request long_context_check",))


@pytest.mark.parametrize("forced", [None, False])
def test_a_request_that_does_not_force_the_check_runs_none(forced) -> None:
    assert decide(req(long_context_check=forced), SETUP) is None


def test_a_toolbox_with_no_prompt_set_and_a_kind_the_check_does_not_cover_run_none() -> None:
    assert decide(req(long_context_check=True), None) is None
    assert decide(req(long_context_check=True, kind=TestKind.PROFILE), SETUP) is None
    assert decide(req(long_context_check=True, purpose=Purpose.FINAL), SETUP) is not None


class _Accept:
    @staticmethod
    def prepare(request, setup, accepted) -> object:
        acceptance = Acceptance(
            AcceptanceSetup(FakeRepo(None), _Planner(accepted), IMAGE, long_context=setup),  # type: ignore[arg-type]
            is_accepted=lambda _id: True,
        )
        return asyncio.run(acceptance.prepare(request))


def test_acceptance_hands_the_executor_the_plan_with_the_accepted_measurement(tmp_path) -> None:
    prepared = _Accept.prepare(req(long_context_check=True), SETUP, _accepted(tmp_path))

    environment = prepared.environment  # type: ignore[attr-defined]
    assert isinstance(environment, AcceptedMeasurement) and environment.long_context is not None
    assert environment.short_prompts is None and environment.points == (tmp_path / "point.yaml",)
    assert _Accept.prepare(req(), SETUP, _accepted(tmp_path)).environment.long_context is None  # type: ignore[attr-defined]
    assert _Accept.prepare(req(long_context_check=True), None, _accepted(tmp_path)).environment.long_context is None  # type: ignore[attr-defined]


def test_a_planner_whose_accepted_measurement_cannot_carry_a_plan_is_left_as_it_is() -> None:
    accepted = {"env": "x"}

    assert _Accept.prepare(req(long_context_check=True), SETUP, accepted).environment is accepted  # type: ignore[attr-defined]
    assert PARENT


# --- switched on -----------------------------------------------------------------------------------------------------


def test_the_environments_key_takes_its_hash_and_length_from_the_staged_manifest(tmp_path) -> None:
    plan = staged_plan(tmp_path)
    environ = {"HYPERLOOM_RUNNER_CACHE": str(tmp_path), LONG_CONTEXT_ENV: plan.prompt_set}

    assert environment_long_context(environ) == LongContextSetup(plan.prompt_set, plan.prompts_sha256, LENGTH)
    assert environment_long_context({**environ, LONG_CONTEXT_ENV: "  "}) is None
    assert environment_long_context({}) is None


def test_a_set_that_is_asked_for_and_not_staged_refuses_rather_than_runs_without(tmp_path) -> None:
    environ = {"HYPERLOOM_RUNNER_CACHE": str(tmp_path), LONG_CONTEXT_ENV: "long-context-missing"}

    with pytest.raises(LongContextNotStaged, match="not staged in the runner cache: .*no readable manifest"):
        environment_long_context(environ)
    with pytest.raises(LongContextNotStaged, match="prompt set long-context-test"):
        staged_long_context(SETUP, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)})


def test_a_named_set_is_checked_against_the_cache_and_one_that_differs_is_refused(tmp_path) -> None:
    plan = staged_plan(tmp_path)
    environ = {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)}
    named = LongContextSetup(plan.prompt_set, plan.prompts_sha256, LENGTH)

    assert staged_long_context(named, environ) == named
    for other in (replace(named, prompts_sha256="1" * 64), replace(named, length=LENGTH + 1)):
        with pytest.raises(LongContextNotStaged, match="other contents or another length"):
            staged_long_context(other, environ)
    assert staged_long_context(None, {**environ, LONG_CONTEXT_ENV: plan.prompt_set}) == named


def test_the_option_and_the_acceptance_setup_name_the_set_once() -> None:
    acceptance = AcceptanceSetup(FakeRepo(None), _Planner(None), IMAGE)  # type: ignore[arg-type]

    named = with_long_context(acceptance, SETUP)

    assert named.long_context == SETUP and with_long_context(named, SETUP) == named
    assert with_long_context(acceptance, None) is acceptance
    with pytest.raises(ValueError, match="name it once"):
        with_long_context(named, replace(SETUP, length=LENGTH * 2))


# --- what is recorded -----------------------------------------------------------------------------------------------


def test_a_server_that_retrieves_everything_gives_a_complete_check_with_every_task_at_one(tmp_path) -> None:
    directory, plan = _landed(tmp_path)

    check = long_context_of(plan, directory)  # type: ignore[arg-type]

    assert (check.status, check.reason) == (LONG_CONTEXT_COMPLETE, "")
    assert (check.n, check.passed, check.score) == (64, 64, 1.0)
    assert [(t.task, t.n, t.score) for t in check.tasks] == [
        ("niah_single", 16, 1.0),
        ("niah_multikey", 16, 1.0),
        ("variable_tracking", 16, 1.0),
        ("common_words", 16, 1.0),
    ]
    assert check.length == LENGTH and check.served_context == 4096 and check.trigger == plan.trigger  # type: ignore[attr-defined]
    assert all(p.passed and p.status == PROMPT_OK for p in check.prompts)
    assert check.prefill_tokens == sum(p.prompt_tokens for p in check.prompts) > 64 * LENGTH * 0.99
    assert check.seconds >= 0 and check.outputs == f"{LONG_CONTEXT_DIR}/{OUTPUTS_FILE}"


@pytest.mark.parametrize(
    ("broken", "lost"),
    [("niah", NIAH), ("tracking", ("variable_tracking",)), ("words", ("common_words",))],
)
def test_a_server_that_loses_one_family_scores_zero_on_exactly_its_tasks_and_stays_complete(
    tmp_path, broken, lost
) -> None:
    directory, plan = _landed(tmp_path, broken={broken})

    check = long_context_of(plan, directory)  # type: ignore[arg-type]

    assert check.status == LONG_CONTEXT_COMPLETE, "wrong answers are a score, not an incomplete check"
    assert {t.task: t.score for t in check.tasks} == {
        t: 0.0 if t in lost else 1.0 for t in {p.task for p in check.prompts}
    }
    assert check.passed == 64 - 16 * len(lost) and check.score == check.passed / 64
    assert sum(not p.passed for p in check.prompts) == 16 * len(lost)


def test_a_launch_that_serves_too_little_is_incomplete_with_the_reason_and_scores_nothing(tmp_path) -> None:
    directory, plan = _landed(tmp_path, served=LENGTH)

    check = long_context_of(plan, directory)  # type: ignore[arg-type]

    assert check.status == LONG_CONTEXT_INCOMPLETE and check.reason.startswith("served context too short")
    assert (check.n, check.passed, check.served_context, check.prefill_tokens) == (64, 0, LENGTH, 0)
    assert {p.status for p in check.prompts} == {PROMPT_NOT_RUN} and not any(p.passed for p in check.prompts)


def test_a_prompt_that_errored_makes_the_check_incomplete_and_counts_against_the_score(tmp_path) -> None:
    directory, plan = _landed(tmp_path, fail=lambda text: "numbered list of words" in text)

    check = long_context_of(plan, directory)  # type: ignore[arg-type]

    assert check.status == LONG_CONTEXT_INCOMPLETE
    assert check.reason.startswith("16 of 64 prompts were not answered (common_words-00: the server answered HTTP 500")
    assert (check.n, check.passed, check.score) == (64, 48, 0.75)
    assert {p.status for p in check.prompts if p.task == "common_words"} == {PROMPT_ERROR}


def test_a_bundle_with_no_usable_outputs_is_incomplete_with_no_score_not_a_score_of_zero(tmp_path) -> None:
    directory, plan = _landed(tmp_path)
    cases = {}
    (directory / LONG_CONTEXT_DIR / OUTPUTS_FILE).write_text("{not json")
    cases["no readable"] = long_context_of(plan, directory)  # type: ignore[arg-type]
    (directory / LONG_CONTEXT_DIR / OUTPUTS_FILE).unlink()
    cases["no readable "] = long_context_of(plan, directory)  # type: ignore[arg-type]
    other = replace(plan, prompts_sha256="1" * 64)  # type: ignore[type-var]
    (directory / LONG_CONTEXT_DIR / OUTPUTS_FILE).write_text("{}")
    cases["the outputs file holds the keys"] = long_context_of(other, directory)

    for reason, check in cases.items():
        assert check.status == LONG_CONTEXT_INCOMPLETE and reason in check.reason
        assert (check.n, check.score, check.prompts, check.tasks) == (0, None, (), ())


def test_outputs_for_another_prompt_set_than_the_test_asked_for_are_not_read_as_its_own(tmp_path) -> None:
    directory, plan = _landed(tmp_path)

    check = long_context_of(replace(plan, prompts_sha256="1" * 64), directory)  # type: ignore[type-var]

    assert check.reason == "the bundle's outputs are for another prompt set than the one the test asked for"
    assert check.score is None


# --- verified again from the bundle ---------------------------------------------------------------------------------


def test_a_check_derived_again_from_the_verified_bundle_agrees_with_the_one_recorded(tmp_path) -> None:
    directory, plan = _landed(tmp_path, broken={"words"})
    recorded = long_context_of(plan, directory)  # type: ignore[arg-type]

    assert first_difference(recorded, rederive(recorded, directory)) is None


def test_a_recorded_score_the_bundles_answers_do_not_support_is_named(tmp_path) -> None:
    directory, plan = _landed(tmp_path, broken={"words"})
    recorded = long_context_of(plan, directory)  # type: ignore[arg-type]
    inflated = replace(
        recorded,
        passed=64,
        score=1.0,
        tasks=tuple(replace(t, passed=t.n, score=1.0) for t in recorded.tasks),
        prompts=tuple(replace(p, passed=True) for p in recorded.prompts),
    )

    message = first_difference(inflated, rederive(recorded, directory))

    assert message == "long_context.passed: recorded 64, the bundle's outputs give 48"


def test_an_answer_changed_in_the_bundle_after_the_record_was_made_is_found(tmp_path) -> None:
    directory, plan = _landed(tmp_path)
    recorded = long_context_of(plan, directory)  # type: ignore[arg-type]
    document = _outputs(directory)
    document["prompts"][0]["text"] = "no idea"
    _write(directory, document)

    message = first_difference(recorded, rederive(recorded, directory))

    assert message == "long_context.passed: recorded 64, the bundle's outputs give 63"
