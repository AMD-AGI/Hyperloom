# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local toolbox stages the session model's short-prompt set at its build when the runner cache lacks it.

Staging is the CPU step ``measure.short_prompts.build`` runs (the model's tokenizer and the seeded generator); the
tokenizer here is the measure tests' one-token-per-word stand-in. A staging that fails leaves the check off with one
warning that says why; a builder that does not stage (the hosted one, whose runners may not read this host's cache)
keeps the old behaviour.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from hyperloom.measure.short_prompts import build
from hyperloom.measure.short_prompts.prompts import PromptSetError
from hyperloom.measure.short_prompts.staging import CACHE_ENV, EVALSETS_DIR
from hyperloom.measure.short_prompts.tests.support import encode
from hyperloom_multibranch.toolbox import short_prompts
from hyperloom_multibranch.toolbox.short_prompts import SHORT_PROMPTS_ENV, staged_short_prompts

_LOGGER = short_prompts.__name__


@pytest.fixture(autouse=True)
def _no_ambient_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SHORT_PROMPTS_ENV, raising=False)


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_bytes(b'{"model": "glm"}')
    return model


def test_a_missing_set_is_staged_at_the_build_and_the_check_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(short_prompts, "build_entry", functools.partial(build.build_entry, encode=encode))
    model, cache = _model(tmp_path), tmp_path / "cache"

    with caplog.at_level("INFO", logger=_LOGGER):
        setup = staged_short_prompts(None, {CACHE_ENV: str(cache)}, model, stage=True)

    assert setup is not None and (cache / EVALSETS_DIR / setup.prompt_set).is_dir()
    assert [r.levelname for r in caplog.records] == ["INFO"] and "staged the model's prompt set" in caplog.text
    assert staged_short_prompts(None, {CACHE_ENV: str(cache)}, model) == setup, "the next build reads the same set"


def test_a_staging_that_fails_leaves_the_check_off_with_one_warning_that_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_tokenizer(model_dir: Path, cache: Path) -> None:
        raise PromptSetError("the transformers library is needed to read the model's tokenizer")

    monkeypatch.setattr(short_prompts, "build_entry", no_tokenizer)

    with caplog.at_level("INFO", logger=_LOGGER):
        assert staged_short_prompts(None, {CACHE_ENV: str(tmp_path / "cache")}, _model(tmp_path), stage=True) is None

    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert (
        "is not staged in the runner cache: staging it failed: PromptSetError: the transformers library" in caplog.text
    )


def test_a_builder_that_does_not_stage_leaves_the_cache_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(short_prompts, "build_entry", functools.partial(build.build_entry, encode=encode))

    assert staged_short_prompts(None, {CACHE_ENV: str(tmp_path / "cache")}, _model(tmp_path)) is None

    assert not (tmp_path / "cache").exists()
    assert "the toolbox does not stage it" in caplog.text
