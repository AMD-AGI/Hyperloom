# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frozen prompt set, its offline entry and the plan that names it."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys

import pytest

from hyperloom.measure.docs import DocumentError
from hyperloom.measure.runner import cache_layout
from hyperloom.measure.short_prompts import build, staging
from hyperloom.measure.short_prompts.plan import ShortPromptPlan, miss_probability
from hyperloom.measure.short_prompts.prompts import (
    BATCHED,
    LENGTHS,
    SEQUENTIAL,
    PromptSetError,
    build_prompts,
    load_prompts,
    prompts_bytes,
)
from hyperloom.measure.short_prompts.staging import (
    EVALSETS_DIR,
    MANIFEST_NAME,
    cache_root,
    load_prompt_set,
    stage_entry,
)
from hyperloom.measure.short_prompts.tests.support import encode, prompt_set, staged_plan


def test_the_set_is_sixty_four_prompts_of_one_to_sixty_four_tokens_with_single_and_odd_lengths():
    prompts = prompt_set()

    assert [p.mode for p in prompts].count(SEQUENTIAL) == [p.mode for p in prompts].count(BATCHED) == 32
    assert [p.length for p in prompts if p.mode == SEQUENTIAL] == list(LENGTHS[SEQUENTIAL])
    lengths = {p.length for p in prompts}
    assert min(lengths) == 1 and max(lengths) == 64
    assert sum(1 for p in prompts if p.length == 1) >= 3, "single-token prompts, with different tokens"
    assert set(range(1, 34, 2)) <= lengths, "every odd length up to 33"


def test_no_two_prompts_share_a_first_token_so_the_prefix_cache_cannot_serve_one_from_another():
    prompts = prompt_set()

    assert len({p.token_ids[0] for p in prompts}) == len(prompts)


def test_the_same_tokenizer_and_seed_give_the_same_file_and_another_seed_another():
    assert prompts_bytes(build_prompts(encode, seed=1)) == prompts_bytes(build_prompts(encode, seed=1))
    assert prompts_bytes(build_prompts(encode, seed=1)) != prompts_bytes(build_prompts(encode, seed=2))


def test_a_tokenizer_that_starts_every_passage_with_one_token_cannot_make_the_set():
    with pytest.raises(PromptSetError, match="too few different first tokens"):
        build_prompts(lambda text: [7] * 200, seed=1)


def test_a_tokenizer_that_gives_too_few_tokens_cannot_make_the_set():
    firsts = itertools.count(100)

    with pytest.raises(PromptSetError, match="fewer than 3 tokens"):
        build_prompts(lambda text: [next(firsts), 5], seed=1)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ('{"id":"a","mode":"sequential"}', "keys id, mode and token_ids"),
        ('{"id":"a","mode":"parallel","token_ids":[1]}', "mode 'parallel'"),
        ('{"id":"a","mode":"batched","token_ids":[]}', "list of 1 to"),
        ('{"id":"a","mode":"batched","token_ids":[1,-2]}', "non-negative integers"),
        ('{"id":"a","mode":"batched","token_ids":[true]}', "non-negative integers"),
        ("not json", "not valid UTF-8 JSON"),
    ],
)
def test_a_prompts_file_that_is_not_a_prompt_set_is_refused_naming_the_line(tmp_path, line, match):
    path = tmp_path / "prompts.jsonl"
    path.write_text(line + "\n")

    with pytest.raises(PromptSetError, match=match):
        load_prompts(path)


def test_a_prompts_file_that_repeats_an_id_or_holds_none_is_refused(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id":"a","mode":"batched","token_ids":[1]}\n' * 2)
    with pytest.raises(PromptSetError, match="an id twice"):
        load_prompts(path)
    path.write_text("")
    with pytest.raises(PromptSetError, match="no prompt"):
        load_prompts(path)


def test_a_staged_entry_loads_back_the_prompts_the_plan_names(tmp_path):
    plan = staged_plan(tmp_path)

    loaded = load_prompt_set(plan, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)})

    assert loaded == prompt_set()


def test_an_entry_is_where_the_runner_cache_layout_puts_it(tmp_path):
    plan = staged_plan(tmp_path)

    entry = tmp_path / EVALSETS_DIR / plan.prompt_set
    manifest = json.loads((entry / MANIFEST_NAME).read_text())
    assert manifest["prompts"]["file"] == "data/prompts.jsonl"
    assert manifest["prompts"]["sha256"] == hashlib.sha256((entry / "data" / "prompts.jsonl").read_bytes()).hexdigest()
    assert cache_root({"HYPERLOOM_RUNNER_CACHE": str(tmp_path)}) == tmp_path
    assert str(cache_root({})) == "/runner-cache"


def test_the_cache_directory_is_the_one_the_runner_cache_layout_names():
    assert (staging.CACHE_ENV, staging.DEFAULT_CACHE_DIR) == (cache_layout.CACHE_ENV, cache_layout.DEFAULT_CACHE_DIR)


def test_the_sheet_naming_other_prompts_than_the_entry_holds_is_refused_with_both_hashes(tmp_path):
    plan = staged_plan(tmp_path)
    other = ShortPromptPlan(plan.prompt_set, "1" * 64)

    with pytest.raises(PromptSetError, match=f"holds prompts {plan.prompts_sha256}, the sheet names {'1' * 64}"):
        load_prompt_set(other, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)})


def test_an_entry_whose_prompts_file_was_changed_after_staging_is_refused(tmp_path):
    plan = staged_plan(tmp_path)
    data = tmp_path / EVALSETS_DIR / plan.prompt_set / "data" / "prompts.jsonl"
    data.write_bytes(data.read_bytes().replace(b"sequential", b"batched", 1))

    with pytest.raises(PromptSetError, match="not what its manifest recorded"):
        load_prompt_set(plan, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)})


def test_an_entry_with_no_manifest_is_unfinished_and_is_not_used(tmp_path):
    plan = staged_plan(tmp_path)
    (tmp_path / EVALSETS_DIR / plan.prompt_set / MANIFEST_NAME).unlink()

    with pytest.raises(PromptSetError, match="not staged, or unfinished"):
        load_prompt_set(plan, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)})


def test_a_cache_without_the_entry_is_refused(tmp_path):
    plan = staged_plan(tmp_path / "builder")

    with pytest.raises(PromptSetError, match="not staged"):
        load_prompt_set(plan, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path / "sandbox")})


def test_staging_the_same_entry_again_is_a_no_op_and_a_different_one_under_the_key_is_refused(tmp_path):
    inputs = {"seed": 1, "tokenizer": {"fake": "0" * 64}}
    first = stage_entry(tmp_path, prompt_set(), inputs)

    assert stage_entry(tmp_path, prompt_set(), inputs) == first
    with pytest.raises(PromptSetError, match="differs from the one that would be written"):
        stage_entry(tmp_path, build_prompts(encode, seed=2), inputs)


def test_the_builder_stages_an_entry_for_a_model_directorys_tokenizer(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")

    entry = build.build_entry(model, tmp_path / "cache", seed=3, encode=encode)

    assert entry.key.startswith("short-prompts-") and (entry.path / MANIFEST_NAME).is_file()
    assert build.tokenizer_hashes(model) == {"tokenizer.json": hashlib.sha256(b"{}").hexdigest()}


def test_the_builder_command_prints_the_two_values_a_plan_names(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(build, "_huggingface_encoder", lambda directory: encode)

    code = build.main(["--model", str(model), "--cache", str(tmp_path / "cache")])

    shown = dict(line.split("=") for line in capsys.readouterr().out.split())
    assert code == 0 and shown["prompt_set"].startswith("short-prompts-") and len(shown["prompts_sha256"]) == 64


def test_the_builder_says_so_when_the_tokenizer_library_is_missing(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert build.main(["--model", str(model), "--cache", str(tmp_path / "cache")]) == 1
    assert "transformers" in capsys.readouterr().err


def test_a_model_directory_with_no_tokenizer_files_is_refused(tmp_path):
    with pytest.raises(PromptSetError, match="none of the tokenizer files"):
        build.tokenizer_hashes(tmp_path)


def test_the_plan_round_trips_and_a_wrong_field_is_refused_naming_it(tmp_path):
    plan = ShortPromptPlan("short-prompts-abc", "a" * 64, trigger=("purpose final",))

    assert ShortPromptPlan.parse(plan.to_document(), "sheet") == plan
    with pytest.raises(DocumentError, match="prompt set key"):
        ShortPromptPlan("Not A Key", "a" * 64)
    with pytest.raises(DocumentError, match="processes must be 1 to"):
        ShortPromptPlan("k", "a" * 64, processes=0)
    with pytest.raises(DocumentError, match="unknown key"):
        ShortPromptPlan.parse({**plan.to_document(), "extra": 1}, "sheet")


def test_a_sheet_written_while_checks_were_stored_by_recipe_still_parses_and_its_digests_are_left_out():
    plan = ShortPromptPlan("short-prompts-abc", "a" * 64, trigger=("purpose final",))
    old = {**plan.to_document(), "recipe": "sha256:" + "b" * 64, "parent_recipe": "sha256:" + "c" * 64}

    assert ShortPromptPlan.parse(old, "sheet") == plan and "recipe" not in plan.to_document()


def test_six_launches_miss_a_defect_in_one_launch_in_three_with_probability_8_8_percent():
    assert miss_probability(6) == pytest.approx(0.0878, abs=1e-4)
    assert miss_probability(8) == pytest.approx(0.0390, abs=1e-4)
    assert miss_probability(0) == 1.0
