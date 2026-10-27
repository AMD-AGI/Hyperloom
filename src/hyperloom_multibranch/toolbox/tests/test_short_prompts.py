# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's short-prompt consistency check: which tests run it, what it is compared with, what the result records."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.consistency import (
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
)
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
)
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.short_prompts.plan import OUTPUTS_FILE, SHORT_PROMPTS_DIR, ShortPromptPlan
from hyperloom.measure.short_prompts.tests.support import outputs_files
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.short_prompts import (
    ShortPromptSetup,
    consistency_of,
    decide,
    short_prompt_rules,
    no_patch_files,
    touch_lines,
)
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req

SETUP = ShortPromptSetup("short-prompts-test", "a" * 64)
AITER = "https://example.invalid/aiter"


def _recipe(server_args: str = "", *, envs: dict[str, str] | None = None, **fields: object) -> Recipe:
    return Recipe(base_image_digest=IMAGE, launch=LaunchConfig(server_args, envs or {}), **fields)  # type: ignore[arg-type]


def _aiter(*patches: PatchRef) -> tuple[ComponentPatches, ...]:
    return (ComponentPatches("aiter", AITER, "1" * 40, patches),)


def _patch(path: str, digest: str = "2" * 64) -> PatchRef:
    return PatchRef(path, digest)


PARENT = _recipe("--max-running-requests 128")
#: A patch's path in a recipe: named by its hash, it says nothing of what the patch changes.
_HASHED = f"patch/sglang/{'6' * 64}.patch"


def _no_files(patch: PatchRef) -> tuple[str, ...]:
    return ()


def _plan(request, candidate: Recipe, parent: Recipe = PARENT, files=_no_files) -> ShortPromptPlan | None:
    return decide(request, parent, candidate, SETUP, files)


# --- which tests run it ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "line"),
    [
        (_recipe("--max-running-requests 128 --attention-backend triton"), "arg --attention-backend triton"),
        (_recipe("--max-running-requests 128 --kv-cache-dtype fp8_e4m3"), "arg --kv-cache-dtype fp8_e4m3"),
        (_recipe("--max-running-requests 128", envs={"SGLANG_USE_AITER_ATTN": "1"}), "env SGLANG_USE_AITER_ATTN=1"),
        (
            _recipe(
                "--max-running-requests 128",
                kernels=(KernelArtifact(KernelKind.REWRITE, "mha_batch_prefill", "kernel/mha.py", "3" * 64),),
            ),
            "kernel rewrite mha_batch_prefill",
        ),
        (_recipe("--max-running-requests 128", unexpressed=("setup_commands",)), "unexpressed setup_commands"),
    ],
)
def test_an_explore_whose_change_touches_attention_kv_cache_or_prefill_runs_the_check_and_says_which_line(
    candidate, line
):
    plan = _plan(req(), candidate)

    assert plan is not None and line in plan.trigger
    assert (plan.prompt_set, plan.prompts_sha256) == (SETUP.prompt_set, SETUP.prompts_sha256)


@pytest.mark.parametrize(
    "candidate",
    [
        _recipe("--max-running-requests 256"),
        _recipe("--max-running-requests 128 --tensor-parallel-size 8"),
        _recipe("--max-running-requests 128", envs={"HIP_FORCE_DEV_KERNARG": "1"}),
    ],
)
def test_an_explore_whose_change_touches_none_of_them_runs_no_check(candidate):
    assert _plan(req(), candidate) is None


def test_a_patch_to_a_component_the_pack_does_not_tie_to_attention_runs_no_check_unless_a_file_it_changes_does():
    sglang = ("sglang", "https://example.invalid/sglang", "4" * 40)
    parent = _recipe("--max-running-requests 128", components=(ComponentPatches(*sglang),))
    candidate = _recipe("--max-running-requests 128", components=(ComponentPatches(*sglang, (_patch(_HASHED),)),))

    assert _plan(req(), candidate, parent, lambda patch: ("python/sglang/srt/utils.py",)) is None
    plan = _plan(req(), candidate, parent, lambda patch: ("python/sglang/srt/layers/radix_attention.py",))
    assert plan is not None and plan.trigger == ("patch-file sglang python/sglang/srt/layers/radix_attention.py",)


def test_a_removed_attention_flag_or_variable_touches_attention_as_much_as_an_added_one():
    parent = _recipe("--attention-backend aiter", envs={"AITER_ENABLE_PREFILL_X": "1"})

    plan = _plan(req(), _recipe(""), parent)

    assert plan is not None
    assert "arg --attention-backend (unset)" in plan.trigger and "env AITER_ENABLE_PREFILL_X=(unset)" in plan.trigger


def test_dropping_a_patch_or_a_kernel_the_parent_had_touches_attention_as_much_as_adding_one():
    parent = _recipe(
        "--max-running-requests 128",
        components=_aiter(_patch("patch/aiter/0001-mha.patch")),
        kernels=(KernelArtifact(KernelKind.REWRITE, "mha_batch_prefill", "kernel/mha.py", "3" * 64),),
    )

    candidate = _recipe("--max-running-requests 128", components=_aiter())

    plan = _plan(req(), candidate, parent, lambda patch: ("aiter/ops/mha.py",))

    assert plan is not None
    assert plan.trigger == ("patch-file aiter aiter/ops/mha.py", "kernel rewrite mha_batch_prefill")


def test_a_component_moved_to_another_upstream_commit_touches_what_the_pack_names_and_one_it_does_not_name_does_not():
    moved = _recipe("--max-running-requests 128", components=_aiter())
    plan = _plan(req(), moved)
    assert plan is not None and plan.trigger == (f"ref aiter (unset)->{'1' * 40}",)

    transformers = ComponentPatches("transformers", "https://example.invalid/t", "5" * 40)
    assert _plan(req(), _recipe("--max-running-requests 128", components=(transformers,))) is None


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.FINAL])
def test_a_baseline_and_a_final_candidate_always_run_the_check_whatever_their_change(purpose):
    plan = _plan(req(purpose=purpose), PARENT)

    assert plan is not None and plan.trigger == (f"purpose {purpose.value}",)


def test_an_integration_test_runs_it_only_when_its_change_touches_the_pack_s_lines():
    benign, touching = (
        _recipe("--max-running-requests 256"),
        _recipe("--max-running-requests 128 --attention-backend aiter"),
    )

    assert _plan(req(purpose=Purpose.INTEGRATION), benign) is None
    assert _plan(req(purpose=Purpose.INTEGRATION), touching) is not None


def test_a_caller_can_force_the_check_but_cannot_remove_one_the_rule_demands():
    benign, touching = (
        _recipe("--max-running-requests 256"),
        _recipe("--max-running-requests 128 --attention-backend aiter"),
    )

    forced = _plan(req(purpose=Purpose.INTEGRATION, consistency_check=True), benign)
    assert forced is not None and forced.trigger == ("request consistency_check",)
    for value in (None, False):
        assert _plan(req(purpose=Purpose.INTEGRATION, consistency_check=value), benign) is None
    assert _plan(req(purpose=Purpose.FINAL, consistency_check=False), benign) is not None
    assert _plan(req(consistency_check=False), touching) is not None


def test_the_reasons_are_bounded_and_the_purpose_and_the_request_come_first():
    many = _recipe(" ".join(f"--attention-backend-{n} x" for n in range(40)) + " --kv-cache-dtype fp8_e4m3")
    request = req(purpose=Purpose.FINAL, consistency_check=True)

    plan = _plan(request, many)

    assert plan is not None and len(plan.trigger) == 32
    assert plan.trigger[:2] == ("purpose final", "request consistency_check")
    assert plan.trigger[-1] == "... and 12 more lines", "2 reasons and 41 lines, of which 31 are shown"


@pytest.mark.parametrize("kind", [TestKind.PROFILE, TestKind.REPRO, TestKind.ACCURACY])
def test_only_a_test_and_a_sweep_run_the_check(kind):
    request = replace(req(purpose=Purpose.FINAL), kind=TestKind.TEST)
    assert _plan(request, PARENT) is not None
    assert _plan(replace(request, kind=TestKind.SWEEP), PARENT) is not None
    other = req(kind=TestKind.TEST, purpose=Purpose.FINAL)
    object.__setattr__(other, "kind", kind)
    assert _plan(other, PARENT) is None


#: Every purpose but the two the pack always checks; ``confirm`` needs the test it confirms.
_UNCHECKED_PURPOSES = [p for p in Purpose if p not in (Purpose.BASELINE, Purpose.FINAL)]
_BENIGN = _recipe("--max-running-requests 256")
_TOUCHING = _recipe("--max-running-requests 128 --attention-backend aiter")


def _purposed(purpose: Purpose, **fields: object):
    return req(purpose=purpose, retest_of="r1" if purpose is Purpose.CONFIRM else None, **fields)


@pytest.mark.parametrize(
    ("purpose", "fields", "candidate", "trigger"),
    [
        (Purpose.BASELINE, {}, PARENT, "purpose baseline"),
        (Purpose.FINAL, {}, _BENIGN, "purpose final"),
        (Purpose.INTEGRATION, {"consistency_check": True}, _BENIGN, "request consistency_check"),
        (Purpose.EXPLORE, {}, _TOUCHING, "arg --attention-backend aiter"),
        (Purpose.INTEGRATION, {}, _TOUCHING, "arg --attention-backend aiter"),
    ],
    ids=["baseline", "final", "lead-adoption", "diff-list-explore", "diff-list-integration"],
)
def test_each_decided_trigger_runs_the_check_six_times_and_names_itself(purpose, fields, candidate, trigger):
    plan = _plan(_purposed(purpose, **fields), candidate)

    assert plan is not None and plan.trigger == (trigger,)
    assert plan.processes == 6


@pytest.mark.parametrize("consistency_check", [None, False])
@pytest.mark.parametrize("purpose", _UNCHECKED_PURPOSES, ids=lambda p: p.value)
def test_no_other_test_runs_it_an_integration_test_that_is_not_an_adoption_included(purpose, consistency_check):
    candidate = PARENT if purpose is Purpose.HEALTH else _BENIGN

    assert _plan(_purposed(purpose, consistency_check=consistency_check), candidate) is None


def test_a_toolbox_given_no_prompt_set_runs_no_check_and_refuses_nothing():
    assert decide(req(purpose=Purpose.FINAL), PARENT, PARENT, None, no_patch_files) is None


def test_the_rules_come_from_the_rule_pack():
    rules = short_prompt_rules()

    assert (rules.processes, rules.max_new_tokens, rules.phase_cap_sec) == (6, 32, 1200.0)
    assert rules.always_purposes == {"baseline", "final"}
    plan = _plan(req(purpose=Purpose.FINAL), PARENT)
    assert plan is not None and (plan.processes, plan.max_new_tokens, plan.phase_cap_sec) == (6, 32, 1200.0)


def test_touch_lines_render_each_kind_of_change_one_line_each():
    candidate = _recipe(
        "--max-running-requests 128 --kv-cache-dtype fp8",
        envs={"A": "1"},
        components=_aiter(_patch("patch/aiter/x.patch")),
    )

    assert touch_lines(PARENT, candidate, lambda patch: ("csrc/a.cu", "aiter/b.py")) == [
        "arg --kv-cache-dtype fp8",
        "env A=1",
        f"ref aiter (unset)->{'1' * 40}",
        "patch aiter patch/aiter/x.patch",
        "patch-file aiter csrc/a.cu",
        "patch-file aiter aiter/b.py",
    ]


def test_the_reader_of_a_patchs_files_is_required_so_no_caller_counts_every_patch_as_touching_by_leaving_it_out():
    with pytest.raises(TypeError, match="files"):
        touch_lines(PARENT, PARENT)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="files"):
        decide(req(), PARENT, PARENT, SETUP)  # type: ignore[call-arg]


# --- acceptance -----------------------------------------------------------------------------------------------------


class _Planner:
    blobs = None

    def __init__(self, accepted: object) -> None:
        self.accepted = accepted

    def plan(self, request, parent) -> object:  # type: ignore[no-untyped-def]
        return self.accepted


def _prepare(request, candidate_args: str, *, setup: ShortPromptSetup | None, accepted: object, replay: bool = False):
    repo = FakeRepo(PARENT)
    acceptance = Acceptance(
        AcceptanceSetup(repo, _Planner(accepted), IMAGE, short_prompts=setup),  # type: ignore[arg-type]
        is_accepted=lambda _id: True,
    )
    return asyncio.run(acceptance.prepare(request, replay=replay))


def _accepted(tmp_path: Path) -> AcceptedMeasurement:
    return AcceptedMeasurement(None, (tmp_path / "point.yaml",))  # type: ignore[arg-type]


def test_acceptance_hands_the_executor_the_plan_with_the_accepted_measurement(tmp_path):
    request = req(server_args="--attention-backend aiter")

    prepared = _prepare(request, "", setup=SETUP, accepted=_accepted(tmp_path))

    environment = prepared.environment
    assert isinstance(environment, AcceptedMeasurement)
    assert (
        environment.short_prompts is not None and "arg --attention-backend aiter" in environment.short_prompts.trigger
    )
    assert environment.points == (tmp_path / "point.yaml",), "the rest of the accepted measurement is untouched"


def test_a_test_with_nothing_touching_attention_is_accepted_without_a_plan(tmp_path):
    prepared = _prepare(req(server_args="--max-num-seqs 256"), "", setup=SETUP, accepted=_accepted(tmp_path))

    assert prepared.environment.short_prompts is None


def test_a_toolbox_with_no_prompt_set_accepts_a_touching_test_with_no_plan(tmp_path):
    prepared = _prepare(req(server_args="--attention-backend aiter"), "", setup=None, accepted=_accepted(tmp_path))

    assert prepared.environment.short_prompts is None


def test_a_planner_whose_accepted_measurement_cannot_carry_a_plan_is_left_as_it_is():
    accepted = {"env": "x"}

    prepared = _prepare(req(server_args="--attention-backend aiter"), "", setup=SETUP, accepted=accepted)

    assert prepared.environment is accepted


# --- what is recorded -----------------------------------------------------------------------------------------------

PROMPT_PLAN = ShortPromptPlan(
    SETUP.prompt_set,
    SETUP.prompts_sha256,
    trigger=("purpose final",),
)


def _landed(tmp_path: Path, behaviours: list[str], plan: ShortPromptPlan = PROMPT_PLAN, name: str = "landed") -> Path:
    directory = tmp_path / name
    for relative, data in outputs_files(plan, behaviours).items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(data)
    return directory


def test_six_clean_processes_state_zero_mismatches_and_no_inconsistency(tmp_path):
    check = consistency_of(PROMPT_PLAN, _landed(tmp_path, ["good"] * 6))

    assert check.status == SHORT_PROMPTS_COMPLETE and check.reason == "" and check.processes == (0, 1, 2, 3, 4, 5)
    assert check.self_consistency is not None and check.self_consistency.mismatches == (0,) * 6
    assert check.inconsistency == 0
    assert check.miss_probability_one_third == pytest.approx(0.0878, abs=1e-4)
    assert (check.prompts, check.compared_by, check.outputs) == (64, "tokens", "short_prompts/outputs.json")
    assert check.trigger == ("purpose final",)


def test_a_defect_in_two_of_six_launches_is_counted_and_located_in_the_candidates_own_processes(tmp_path):
    check = consistency_of(PROMPT_PLAN, _landed(tmp_path, ["good", "bad", "good", "good", "bad", "good"]))

    own = check.self_consistency
    assert own is not None and [bool(n) for n in own.mismatches] == [False, True, False, False, True, False]
    first = own.first_difference
    assert first is not None and (first.process, first.position) == (1, 1) and first.prompt_tokens <= 16


def test_a_process_that_did_not_come_up_makes_the_check_incomplete_and_says_which(tmp_path):
    check = consistency_of(PROMPT_PLAN, _landed(tmp_path, ["good", "fail", "good", "good", "good", "good"]))

    assert check.status == SHORT_PROMPTS_INCOMPLETE and "process 1: did not come up (test)" in check.reason
    assert check.processes == (0, 2, 3, 4, 5) and check.processes_planned == 6
    assert check.self_consistency is not None and len(check.self_consistency.mismatches) == 5


def test_a_run_that_stopped_at_the_cap_is_incomplete_with_the_cap_as_the_reason(tmp_path):
    landed = _landed(tmp_path, ["good", "good"])
    path = landed / SHORT_PROMPTS_DIR / OUTPUTS_FILE
    document = json.loads(path.read_text())
    document["stopped"] = "the phase cap leaves no room for another process"
    path.write_text(json.dumps(document))

    check = consistency_of(PROMPT_PLAN, landed)

    assert check.status == SHORT_PROMPTS_INCOMPLETE and "phase cap" in check.reason and check.processes == (0, 1)


def test_one_process_alone_has_no_self_consistency_and_is_incomplete(tmp_path):
    check = consistency_of(PROMPT_PLAN, _landed(tmp_path, ["good"]))

    assert check.self_consistency is None and check.status == SHORT_PROMPTS_INCOMPLETE


@pytest.mark.parametrize("behaviours", [["fail"] * 6, []])
def test_when_no_process_answered_nothing_was_compared_and_the_check_says_so(tmp_path, behaviours):
    check = consistency_of(PROMPT_PLAN, _landed(tmp_path, behaviours))

    assert check.status == SHORT_PROMPTS_INCOMPLETE and check.processes == () and check.self_consistency is None
    assert check.inconsistency is None, "nothing compared is not a count of zero"


def test_a_bundle_with_no_outputs_file_is_an_incomplete_check_not_a_clean_one(tmp_path):
    (tmp_path / "empty").mkdir()

    check = consistency_of(PROMPT_PLAN, tmp_path / "empty")

    assert check.status == SHORT_PROMPTS_INCOMPLETE and "no readable short_prompts/outputs.json" in check.reason
    assert check.processes == () and check.self_consistency is None


def test_outputs_of_another_prompt_set_are_not_read_as_the_asked_for_one(tmp_path):
    other = replace(PROMPT_PLAN, prompts_sha256="9" * 64)

    check = consistency_of(PROMPT_PLAN, _landed(tmp_path, ["good"] * 6, other))

    assert check.status == SHORT_PROMPTS_INCOMPLETE and "another prompt set" in check.reason


def test_a_late_drift_in_one_process_is_located_at_its_token_and_the_prompt_is_long(tmp_path):
    three = replace(PROMPT_PLAN, processes=3)

    check = consistency_of(three, _landed(tmp_path, ["good", "drift", "good"], three))

    difference = check.self_consistency.first_difference
    assert difference is not None and (difference.process, difference.position) == (1, 31)
    assert difference.prompt_tokens > 16 and check.inconsistency == 1
