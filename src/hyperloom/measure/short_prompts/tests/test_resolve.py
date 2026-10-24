# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Finding a staged prompt set by its key and showing that a runner can read it."""

from __future__ import annotations

import json

import pytest

from hyperloom.measure.short_prompts.prompts import PromptSetError
from hyperloom.measure.short_prompts.resolve import staged_entry, verified_entry
from hyperloom.measure.short_prompts.staging import EVALSETS_DIR, MANIFEST_NAME, CACHE_ENV
from hyperloom.measure.short_prompts.tests.support import staged_plan


def _environ(cache):
    return {CACHE_ENV: str(cache)}


def test_a_key_is_resolved_to_the_hash_its_own_manifest_records(tmp_path):
    plan = staged_plan(tmp_path)

    entry = staged_entry(plan.prompt_set, _environ(tmp_path))

    assert (entry.key, entry.prompts_sha256) == (plan.prompt_set, plan.prompts_sha256)
    assert entry.path == tmp_path / EVALSETS_DIR / plan.prompt_set


def test_a_key_and_the_hash_a_sheet_names_are_confirmed_against_the_entry(tmp_path):
    plan = staged_plan(tmp_path)

    entry = verified_entry(plan.prompt_set, plan.prompts_sha256, _environ(tmp_path))

    assert (entry.key, entry.prompts_sha256) == (plan.prompt_set, plan.prompts_sha256)


def test_a_hash_the_entry_does_not_hold_is_refused_with_both_hashes(tmp_path):
    plan = staged_plan(tmp_path)

    with pytest.raises(PromptSetError, match=f"holds prompts {plan.prompts_sha256}, the sheet names {'1' * 64}"):
        verified_entry(plan.prompt_set, "1" * 64, _environ(tmp_path))


def test_a_cache_without_the_entry_has_no_manifest_to_read(tmp_path):
    plan = staged_plan(tmp_path / "builder")

    with pytest.raises(PromptSetError, match="has no readable manifest .*FileNotFoundError.*not staged, or unfinished"):
        staged_entry(plan.prompt_set, _environ(tmp_path / "elsewhere"))


@pytest.mark.parametrize("written", ["not json", "[]", '{"prompts": 3}', '{"prompts": {}}'])
def test_a_manifest_that_does_not_record_a_hash_is_refused(tmp_path, written):
    plan = staged_plan(tmp_path)
    (tmp_path / EVALSETS_DIR / plan.prompt_set / MANIFEST_NAME).write_text(written)

    with pytest.raises(PromptSetError, match="has no readable manifest"):
        staged_entry(plan.prompt_set, _environ(tmp_path))


@pytest.mark.parametrize("recorded", ["short", None, 7, ["a" * 64]])
def test_a_manifest_whose_recorded_hash_is_not_a_sha256_is_refused(tmp_path, recorded):
    plan = staged_plan(tmp_path)
    manifest = tmp_path / EVALSETS_DIR / plan.prompt_set / MANIFEST_NAME
    document = json.loads(manifest.read_text())
    document["prompts"]["sha256"] = recorded
    manifest.write_text(json.dumps(document))

    with pytest.raises(PromptSetError, match="prompts_sha256 is"):
        staged_entry(plan.prompt_set, _environ(tmp_path))


def test_an_entry_whose_prompts_file_changed_after_staging_is_refused(tmp_path):
    plan = staged_plan(tmp_path)
    data = tmp_path / EVALSETS_DIR / plan.prompt_set / "data" / "prompts.jsonl"
    data.write_bytes(data.read_bytes().replace(b"sequential", b"batched", 1))

    with pytest.raises(PromptSetError, match="not what its manifest recorded"):
        staged_entry(plan.prompt_set, _environ(tmp_path))


@pytest.mark.parametrize("key", ["../outside", "a/b", "Not A Key", ""])
def test_a_key_that_is_not_a_plain_name_is_refused_before_any_path_is_read(tmp_path, key):
    with pytest.raises(PromptSetError, match="is not a prompt set key"):
        staged_entry(key, _environ(tmp_path))
    with pytest.raises(PromptSetError, match="is not a prompt set key"):
        verified_entry(key, "a" * 64, _environ(tmp_path))
