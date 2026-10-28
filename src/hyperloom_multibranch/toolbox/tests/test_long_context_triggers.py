# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which tests run the long-context check (board #142 c3016): the short-prompt check's lever list and always purposes,
read from the same pack entries, and a request that forces it; and that it is on by default, with ``off`` to disable it."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.long_context.build import DEFAULT_SEED, build_entry
from hyperloom.measure.long_context.tests.support import LENGTH, encode
from hyperloom_multibranch.toolbox import long_context as long_context_module
from hyperloom_multibranch.toolbox import short_prompts
from hyperloom_multibranch.toolbox.long_context import (
    LONG_CONTEXT_ENV,
    LongContextSetup,
    decide,
    default_long_context,
    environment_long_context,
)
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup, blob_patch_files, no_patch_files
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, req
from hyperloom_multibranch.toolbox.tests.test_short_prompt_triggers import ATTENTION_FILE, MOE_FILE, _sglang

SETUP = LongContextSetup("long-context-test", "a" * 64, LENGTH)
SHORT = ShortPromptSetup("short-prompts-test", "b" * 64)
PARENT = Recipe(base_image_digest=IMAGE, launch=LaunchConfig("--max-running-requests 128", {}))
#: A context limit whose set, less the 128 tokens an answer may have, is the test sets' length.
MODEL_LIMIT = LENGTH + 128


def _launch(server_args: str = "--max-running-requests 128", **envs: str) -> Recipe:
    return Recipe(base_image_digest=IMAGE, launch=LaunchConfig(server_args, envs))


@pytest.mark.parametrize(
    ("candidate", "trigger"),
    [
        (_launch(SGLANG_USE_AITER_ATTN="1"), ("env SGLANG_USE_AITER_ATTN=1",)),
        (_launch(SGLANG_AITER_KV_CACHE_LAYOUT="vectorized_5d"), ("env SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d",)),
        (_launch("--max-running-requests 128 --attention-backend aiter"), ("arg --attention-backend aiter",)),
        (_launch(AITER_CUSTOM_AR_MAX_SIZE="1"), None),
        (_launch("--max-running-requests 256"), None),
    ],
    ids=["attention-env", "kv-layout-env", "attention-arg", "all-reduce-env", "batch-arg"],
)
def test_a_lever_on_the_short_prompt_list_runs_the_check_and_one_off_it_does_not(candidate, trigger) -> None:
    plan = decide(req(), PARENT, candidate, SETUP, no_patch_files)

    assert (plan and plan.trigger) == trigger


def test_a_patch_runs_it_by_the_files_it_changes_as_the_short_prompt_check_reads_them(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    parent, attention, _ = _sglang(blobs, tmp_path / "a", ATTENTION_FILE)
    _, moe, _ = _sglang(blobs, tmp_path / "m", MOE_FILE)

    touching = decide(req(), parent, attention, SETUP, blob_patch_files(blobs))

    assert touching is not None and touching.trigger == (f"patch-file sglang {ATTENTION_FILE}",)
    assert decide(req(), parent, moe, SETUP, blob_patch_files(blobs)) is None


@pytest.mark.parametrize(
    "candidate",
    [_launch(SGLANG_USE_AITER_ATTN="1"), _launch(AITER_CUSTOM_AR_MAX_SIZE="1"), _launch("--attention-backend aiter")],
    ids=["attention-env", "all-reduce-env", "attention-arg-replacing"],
)
@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.FINAL, Purpose.BASELINE, Purpose.HEALTH])
def test_both_checks_read_one_list_so_they_trigger_on_the_same_change_and_purpose(candidate, purpose) -> None:
    request = req(purpose=purpose)
    short = short_prompts.decide(request, PARENT, candidate, SHORT, no_patch_files)
    long = decide(request, PARENT, candidate, SETUP, no_patch_files)

    assert (long and long.trigger) == (short and short.trigger)


@pytest.mark.parametrize(
    ("purpose", "trigger"),
    [
        (Purpose.FINAL, ("purpose final",)),
        (Purpose.BASELINE, ("purpose baseline",)),
        (Purpose.EXPLORE, None),
        (Purpose.HEALTH, None),
    ],
)
def test_a_final_candidate_and_the_roots_baseline_run_it_with_no_change_and_nothing_else_does(purpose, trigger) -> None:
    plan = decide(req(purpose=purpose), PARENT, PARENT, SETUP, no_patch_files)

    assert (plan and plan.trigger) == trigger


def test_a_lead_adoption_forces_the_short_prompt_check_only_and_a_forced_long_context_check_says_so() -> None:
    adoption = req(consistency_check=True)
    forced = req(long_context_check=True)

    assert decide(adoption, PARENT, PARENT, SETUP, no_patch_files) is None
    plan = decide(forced, PARENT, _launch(SGLANG_USE_AITER_ATTN="1"), SETUP, no_patch_files)
    assert plan is not None and plan.trigger == ("request long_context_check", "env SGLANG_USE_AITER_ATTN=1")


def test_no_prompt_set_or_a_kind_the_check_does_not_cover_runs_none_whatever_the_trigger() -> None:
    final = req(purpose=Purpose.FINAL)

    assert decide(final, PARENT, PARENT, None, no_patch_files) is None
    assert decide(req(purpose=Purpose.FINAL, kind=TestKind.PROFILE), PARENT, PARENT, SETUP, no_patch_files) is None


# --- switched on ---------------------------------------------------------------------------------------------------


def _model(tmp_path: Path, limit: int | None = MODEL_LIMIT) -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "tokenizer.json").write_bytes(b'{"model": "fake"}')
    (model / "config.json").write_text(json.dumps({} if limit is None else {"max_position_embeddings": limit}))
    return model


def _environ(cache: Path, **values: str) -> dict[str, str]:
    return {"HYPERLOOM_RUNNER_CACHE": str(cache), **values}


def _staged(model: Path, cache: Path, length: int = LENGTH, seed: int = DEFAULT_SEED) -> LongContextSetup:
    entry = build_entry(model, cache, length=length, seed=seed, encode=encode)
    return LongContextSetup(entry.key, entry.prompts_sha256, length)


def test_nothing_named_turns_the_check_on_with_the_set_built_for_the_model_at_its_own_limit(tmp_path: Path) -> None:
    model, cache = _model(tmp_path), tmp_path / "cache"
    _staged(model, cache, length=LENGTH + 64)  # a set built for another served context is not the default one
    _staged(model, cache, seed=DEFAULT_SEED + 1)
    default = _staged(model, cache)

    assert environment_long_context(_environ(cache), model) == default
    assert environment_long_context(_environ(cache, MODEL_PATH=str(model))) == default, "a session's MODEL_PATH"
    assert environment_long_context(_environ(cache, **{LONG_CONTEXT_ENV: " "}), model) == default


def test_off_turns_the_check_off_where_it_would_be_on(tmp_path: Path) -> None:
    model, cache = _model(tmp_path), tmp_path / "cache"
    _staged(model, cache)

    assert environment_long_context(_environ(cache, **{LONG_CONTEXT_ENV: "off"}), model) is None


def test_a_model_whose_set_is_not_staged_leaves_the_check_off_saying_so_unless_the_toolbox_stages_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, cache = _model(tmp_path), tmp_path / "cache"
    with caplog.at_level(logging.WARNING):
        assert default_long_context(model, _environ(cache)) is None
    assert "is not staged in the runner cache: the toolbox does not stage it" in caplog.text

    def built(model_dir: Path, root: Path, *, length: int) -> object:
        return build_entry(model_dir, root, length=length, encode=encode)

    monkeypatch.setattr(long_context_module, "build_entry", built)
    staged = default_long_context(model, _environ(cache), stage=True)

    assert staged is not None and staged.length == LENGTH
    assert default_long_context(model, _environ(cache)) == staged, "staged where the runners read it"


@pytest.mark.parametrize(
    ("limit", "tokenizer", "said"),
    [
        (None, True, "the model's tokenizer or context limit cannot be read"),
        (MODEL_LIMIT, False, "the model's tokenizer or context limit cannot be read"),
    ],
    ids=["no-context-limit", "no-tokenizer"],
)
def test_a_model_whose_limit_or_tokenizer_cannot_be_read_leaves_the_check_off_saying_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, limit, tokenizer, said
) -> None:
    model = _model(tmp_path, limit)
    if not tokenizer:
        (model / "tokenizer.json").unlink()

    with caplog.at_level(logging.WARNING):
        assert default_long_context(model, _environ(tmp_path / "cache"), stage=True) is None
    assert said in caplog.text


def test_a_toolbox_of_no_model_runs_no_check(tmp_path: Path) -> None:
    assert environment_long_context(_environ(tmp_path)) is None
