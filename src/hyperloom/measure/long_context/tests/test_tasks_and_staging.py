# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frozen prompt set: what the generators make, and the entry in the runner cache that holds it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.docs import DocumentError
from hyperloom.measure.long_context import build
from hyperloom.measure.long_context.grade import passes
from hyperloom.measure.long_context.plan import MAX_LENGTH, REPLY_HEADROOM, LongContextPlan
from hyperloom.measure.long_context.prompts import PER_TASK, TASKS, PromptSetError, load_prompts, prompts_bytes
from hyperloom.measure.long_context.resolve import staged_set
from hyperloom.measure.long_context.staging import KEY_PREFIX, load_prompt_set, stage_entry
from hyperloom.measure.long_context.tasks import build_prompts
from hyperloom.measure.long_context.tests.support import LENGTH, encode, prompt_set, staged_plan
from hyperloom.measure.short_prompts import staging as short_staging

TOKENIZER = {"fake": "0" * 64}
INPUTS = {"seed": 1, "length": LENGTH, "tokenizer": TOKENIZER}


def test_each_task_has_its_prompts_each_at_most_the_length_and_within_one_percent_of_it() -> None:
    prompts = prompt_set()

    assert [(task, sum(p.task == task for p in prompts)) for task in TASKS] == [(task, PER_TASK) for task in TASKS]
    assert len({p.id for p in prompts}) == len(prompts)
    assert all(LENGTH * 0.99 <= p.tokens <= LENGTH for p in prompts)
    assert all(p.tokens == len(encode(p.text)) for p in prompts), "the recorded length is what the tokenizer measured"


def test_the_same_seed_gives_the_same_prompts_and_another_seed_gives_others() -> None:
    again = build_prompts(encode, length=LENGTH, seed=1)

    assert prompts_bytes(again) == prompts_bytes(prompt_set())
    assert prompts_bytes(build_prompts(encode, length=LENGTH, seed=2)) != prompts_bytes(again)


def test_every_reference_is_in_its_prompt_and_the_needles_are_spread_over_the_depths() -> None:
    prompts = prompt_set()

    assert all(ref in p.text for p in prompts for ref in p.references)
    single = [p for p in prompts if p.task == "niah_single"]
    depths = [p.text.index(p.references[0]) / len(p.text) for p in single]
    assert depths == sorted(depths) and depths[0] < 0.1 and depths[-1] > 0.9, "the needle goes deeper with the index"


def test_the_multikey_question_names_one_of_several_needles_and_the_tracking_prompt_has_a_distractor_chain() -> None:
    prompts = {p.id: p for p in prompt_set()}

    multikey = prompts["niah_multikey-03"]
    assert multikey.text.count("One of the special magic numbers for") == 4
    tracking = prompts["variable_tracking-03"]
    assert tracking.text.count(" = ") == 8 and len(tracking.references) == 4, "two chains of four assignments"
    words = prompts["common_words-03"]
    assert all(words.text.count(f". {ref}\n") == 30 for ref in words.references), "the five words are the frequent ones"


def test_a_tokenizer_that_gives_no_more_tokens_for_a_longer_text_is_refused() -> None:
    with pytest.raises(PromptSetError, match="a longer text does not give more tokens"):
        build_prompts(lambda text: [1, 2, 3], length=LENGTH, seed=1)


def test_grading_is_a_case_insensitive_substring_match_of_every_reference() -> None:
    assert passes("The number is 1234567.", ["1234567"])
    assert passes("abc DEF", ["ABC", "def"])
    assert not passes("abc", ["abc", "def"]), "one missing reference fails the prompt"
    assert not passes("anything", []), "no reference is nothing to find, never a pass"


def test_the_plan_refuses_what_no_server_could_be_asked() -> None:
    for fields, match in (
        ({"prompt_set": "Bad Key"}, "not a prompt set key"),
        ({"prompts_sha256": "x"}, "not a sha256"),
        ({"length": 10}, "length must be"),
        ({"max_new_tokens": 129}, "max_new_tokens must be 1 to 128"),
        ({"concurrency": 0}, "concurrency must be"),
        ({"phase_cap_sec": float("nan")}, "phase_cap_sec"),
    ):
        arguments = {"prompt_set": "k", "prompts_sha256": "0" * 64, "length": LENGTH, **fields}
        with pytest.raises(DocumentError, match=match):
            LongContextPlan(**arguments)
    plan = LongContextPlan("k", "0" * 64, 131072, trigger=("request long_context_check",))
    assert LongContextPlan.parse(plan.to_document(), "plan") == plan
    assert plan.needed_context == 131072 + 128


def test_an_entry_is_staged_in_the_runner_cache_layout_and_read_back(tmp_path: Path) -> None:
    plan = staged_plan(tmp_path)

    assert plan.prompt_set.startswith(KEY_PREFIX)
    entry = tmp_path / short_staging.EVALSETS_DIR / plan.prompt_set
    assert sorted(p.name for p in entry.iterdir()) == ["data", "manifest"]
    manifest = json.loads((entry / "manifest").read_text())
    assert manifest["inputs"]["length"] == LENGTH and manifest["prompts"]["count"] == 64
    assert load_prompt_set(plan, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)}) == prompt_set()
    found = staged_set(plan.prompt_set, {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)})
    assert (found.prompts_sha256, found.length) == (plan.prompts_sha256, LENGTH)


def test_a_rewrite_that_is_the_same_entry_is_left_alone_and_one_that_differs_is_refused(tmp_path: Path) -> None:
    first = stage_entry(tmp_path, prompt_set(), INPUTS)

    assert stage_entry(tmp_path, prompt_set(), INPUTS) == first
    with pytest.raises(PromptSetError, match="exists and differs"):
        stage_entry(tmp_path, prompt_set()[:-1], INPUTS)


def test_a_runner_refuses_an_entry_that_is_not_the_one_the_sheet_names(tmp_path: Path) -> None:
    plan = staged_plan(tmp_path)
    environ = {"HYPERLOOM_RUNNER_CACHE": str(tmp_path)}
    entry = tmp_path / short_staging.EVALSETS_DIR / plan.prompt_set

    other = LongContextPlan(plan.prompt_set, "1" * 64, plan.length)
    with pytest.raises(PromptSetError, match="the sheet names"):
        load_prompt_set(other, environ)
    shorter = LongContextPlan(plan.prompt_set, plan.prompts_sha256, 1024)
    with pytest.raises(PromptSetError, match="the sheet's length is 1024"):
        load_prompt_set(shorter, environ)
    data = entry / "data" / "prompts.jsonl"
    data.write_bytes(data.read_bytes().replace(b"magic", b"MAGIC", 1))
    with pytest.raises(PromptSetError, match="not what its manifest recorded"):
        load_prompt_set(plan, environ)
    (entry / "manifest").unlink()
    with pytest.raises(PromptSetError, match="no readable manifest"):
        load_prompt_set(plan, environ)
    with pytest.raises(PromptSetError, match="no readable manifest"):
        staged_set(plan.prompt_set, environ)
    with pytest.raises(PromptSetError, match="not a prompt set key"):
        staged_set("../escape", environ)


def test_a_prompts_file_that_is_not_a_prompt_set_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    good = json.dumps({"id": "a", "task": "niah_single", "input": "x", "outputs": ["1"], "tokens": 3})
    for line, match in (
        (good.replace('"niah_single"', '"other"'), "is not one of"),
        (good.replace('["1"]', "[]"), "non-empty list"),
        (good.replace('"tokens": 3', '"tokens": 0'), "tokens must be"),
        (good.replace('"id"', '"name"'), "expected the keys"),
        ("not json", "prompts line 1"),
        (good + "\n" + good, "an id twice"),
    ):
        path.write_text(line + "\n")
        with pytest.raises(PromptSetError, match=match):
            load_prompts(path)


def test_the_builder_command_stages_the_set_and_prints_what_a_setup_names(tmp_path, monkeypatch, capsys) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(build, "huggingface_encoder", lambda directory: encode)

    assert build.main(["--model", str(model), "--cache", str(tmp_path / "cache"), "--length", str(LENGTH)]) == 0

    printed = dict(line.split("=") for line in capsys.readouterr().out.split())
    found = staged_set(printed["prompt_set"], {"HYPERLOOM_RUNNER_CACHE": str(tmp_path / "cache")})
    assert (found.prompts_sha256, str(found.length)) == (printed["prompts_sha256"], printed["length"])
    assert build.main(["--model", str(model), "--cache", str(tmp_path / "cache"), "--length", "10"]) == 1
    assert "the length must be" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("model_limit", "served", "length"),
    [
        (32768, 8192, 8064),  # measured: a 32K model whose launch serves 8192 left a 32K set incomplete
        (32768, None, 32640),
        (4096, 8192, 3968),
        (None, 8192, 8064),
        (1 << 24, None, MAX_LENGTH),
    ],
)
def test_the_set_is_built_at_the_context_the_launch_serves_within_the_models_limit_less_the_answer(
    model_limit, served, length
) -> None:
    assert build.set_length(model_limit, served) == length


def test_a_set_with_neither_a_model_limit_nor_a_served_context_is_refused() -> None:
    with pytest.raises(PromptSetError, match="no context limit and no served context"):
        build.set_length(None, None)


def test_the_builder_builds_for_the_served_context_and_the_check_then_fits_the_launch(
    tmp_path, monkeypatch, capsys
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    monkeypatch.setattr(build, "huggingface_encoder", lambda directory: encode)
    served = LENGTH + REPLY_HEADROOM

    assert build.main(["--model", str(model), "--cache", str(tmp_path / "cache"), "--served-context", str(served)]) == 0

    printed = dict(line.split("=") for line in capsys.readouterr().out.split())
    found = staged_set(printed["prompt_set"], {"HYPERLOOM_RUNNER_CACHE": str(tmp_path / "cache")})
    assert found.length == int(printed["length"]) == LENGTH
    assert LongContextPlan(found.prompt_set, found.prompts_sha256, found.length).needed_context == served
