# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's pool of short-prompt answers per recipe: what it keeps, what it compares, and that a restart keeps it."""

from __future__ import annotations

from pathlib import Path

from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.short_prompts.outputs import read_outputs
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.tests.support import outputs_files
from hyperloom_multibranch.toolbox.short_prompt_pool import POOL_FILE, PoolKey, ShortPromptPool, pooled_result
from hyperloom_multibranch.toolbox.short_prompts import consistency_of

PLAN = ShortPromptPlan("short-prompts-test", "a" * 64)
RECIPE = "sha256:" + "1" * 64
OTHER = "sha256:" + "2" * 64


def _landed(session: Path, name: str, behaviours: list[str]) -> TestResult:
    """A succeeded result whose test landed the outputs of processes that behave as ``behaviours`` say."""
    plan = ShortPromptPlan(PLAN.prompt_set, PLAN.prompts_sha256, processes=len(behaviours))
    directory = session / "tests" / name / "attempt-1"
    for relative, data in outputs_files(plan, behaviours).items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(data)
    check = consistency_of(plan, directory)
    return TestResult(name, Validity.SUCCEEDED, output_dir=f"tests/{name}/attempt-1", short_prompt_consistency=check)


def _key(session: Path) -> PoolKey:
    found = read_outputs(session / "tests" / "a" / "attempt-1")
    return PoolKey(RECIPE, found.prompts_sha256, found.max_new_tokens)


def test_each_test_of_a_recipe_adds_its_processes_and_is_compared_with_all_of_them(tmp_path):
    pool = ShortPromptPool(tmp_path / POOL_FILE)

    first = pooled_result(pool, RECIPE, _landed(tmp_path, "a", ["good"]), tmp_path).short_prompt_consistency
    second = pooled_result(pool, RECIPE, _landed(tmp_path, "b", ["good", "bad"]), tmp_path).short_prompt_consistency

    assert first.pooled.processes == ("tests/a/attempt-1#0",) and first.inconsistency is None
    assert second.pooled.processes == ("tests/a/attempt-1#0", "tests/b/attempt-1#0", "tests/b/attempt-1#1")
    assert [n > 0 for n in second.pooled.counts.mismatches] == [False, False, True]
    assert second.inconsistency == second.pooled.inconsistency == max(second.pooled.counts.mismatches) > 0
    assert second.self_consistency.mismatches[1] > 0, "the test's own comparison is still its bundle's"


def test_another_recipe_or_another_prompt_set_is_another_pool(tmp_path):
    pool = ShortPromptPool(tmp_path / POOL_FILE)
    pooled_result(pool, RECIPE, _landed(tmp_path, "a", ["good"]), tmp_path)

    other = pooled_result(pool, OTHER, _landed(tmp_path, "b", ["bad"]), tmp_path).short_prompt_consistency

    assert other.pooled.count == 1 and other.pooled.recipe_digest == OTHER
    key = _key(tmp_path)
    assert pool.count(key) == 1 and pool.count(PoolKey(RECIPE, "f" * 64, key.max_new_tokens)) == 0


def test_the_pool_is_read_back_after_a_restart_and_a_process_is_never_pooled_twice(tmp_path):
    pool = ShortPromptPool(tmp_path / POOL_FILE)
    landed = _landed(tmp_path, "a", ["good", "good"])
    pooled_result(pool, RECIPE, landed, tmp_path)
    pooled_result(pool, RECIPE, landed, tmp_path)

    restarted = ShortPromptPool(tmp_path / POOL_FILE)
    later = pooled_result(restarted, RECIPE, _landed(tmp_path, "b", ["bad"]), tmp_path).short_prompt_consistency

    assert restarted.count(_key(tmp_path)) == 3 and later.pooled.count == 3
    assert later.pooled.counts.mismatches[2] > 0 and later.pooled.counts.mismatches[:2] == (0, 0)
    assert len((tmp_path / POOL_FILE).read_text().splitlines()) == 3, "the same process is not appended again"


def test_a_torn_last_line_is_ignored_and_the_lines_before_it_are_kept(tmp_path):
    pool = ShortPromptPool(tmp_path / POOL_FILE)
    pooled_result(pool, RECIPE, _landed(tmp_path, "a", ["good"]), tmp_path)
    with (tmp_path / POOL_FILE).open("a", encoding="utf-8") as handle:
        handle.write('{"recipe_digest": "sha256:')

    assert ShortPromptPool(tmp_path / POOL_FILE).count(_key(tmp_path)) == 1


def test_a_result_without_a_check_a_digest_or_readable_outputs_is_left_as_it_is(tmp_path):
    pool = ShortPromptPool(tmp_path / POOL_FILE)
    landed = _landed(tmp_path, "a", ["good"])
    unchecked = TestResult("u", Validity.SUCCEEDED, output_dir="tests/a/attempt-1")
    lost = TestResult(
        "l", Validity.SUCCEEDED, output_dir="tests/nowhere", short_prompt_consistency=landed.short_prompt_consistency
    )

    assert pooled_result(pool, RECIPE, unchecked, tmp_path) is unchecked
    assert pooled_result(pool, "", landed, tmp_path) is landed, "a request logged before digests were recorded"
    assert pooled_result(pool, RECIPE, lost, tmp_path) is lost
    assert not (tmp_path / POOL_FILE).exists() or pool.count(_key(tmp_path)) == 0
