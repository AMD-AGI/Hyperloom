# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The build key: every input moves it, the record form reads back, and the native digest follows the base file."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from hyperloom.measure.compile import EnvEntry
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.jit.native.key import BuildInputs, native_digest
from hyperloom.measure.runner.native.tests._native import BUILDER_IMAGE, OTHER_IMAGE
from hyperloom.measure.tests._recipes import IMAGE

INPUTS = BuildInputs(IMAGE, BUILDER_IMAGE, "fakepkg", "gfx942", "e" * 64, "fake/1")
CHANGED = {
    "image_digest": OTHER_IMAGE,
    "builder_image_digest": OTHER_IMAGE,
    "component": "otherpkg",
    "gfx_arch": "gfx950",
    "native_digest": "f" * 64,
    "recipe_version": "fake/2",
}


def test_every_input_is_in_the_key() -> None:
    assert sorted(CHANGED) == sorted(field.name for field in fields(BuildInputs))
    keys = {replace(INPUTS, **{name: value}).key for name, value in CHANGED.items()}
    assert len(keys | {INPUTS.key}) == len(CHANGED) + 1


def test_the_record_form_reads_back_and_refuses_an_unknown_field() -> None:
    assert BuildInputs.from_document(INPUTS.document(), "inputs") == INPUTS
    with pytest.raises(DocumentError, match="unknown key 'extra'"):
        BuildInputs.from_document({**INPUTS.document(), "extra": "x"}, "inputs")


def test_a_component_that_is_not_one_path_segment_is_refused() -> None:
    with pytest.raises(ValueError, match="is not a component name"):
        replace(INPUTS, component="../escape")


def test_the_native_digest_follows_the_file_a_change_replaces(tmp_path: Path) -> None:
    entries = (EnvEntry("src/kernel.c", "upsert", "a" * 64, "0644"),)
    (tmp_path / "src").mkdir()
    absent = native_digest(entries, tmp_path)
    (tmp_path / "src/kernel.c").write_text("one")
    one = native_digest(entries, tmp_path)
    (tmp_path / "src/kernel.c").write_text("two")

    assert len({absent, one, native_digest(entries, tmp_path)}) == 3
    assert native_digest((EnvEntry("src/kernel.c", "delete"),), tmp_path) != native_digest(entries, tmp_path)
