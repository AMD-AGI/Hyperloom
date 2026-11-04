# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context check is on by default: a staged set named by an option of the local builder or by the host environment,
else the session model's own set; ``off`` in the environment turns it off, and a toolbox of no model and no set runs none.

The local toolbox is built for real (the sandbox provider and the backend are made), as the short-prompt option tests build
it; what is asserted is what a built toolbox does with a request that forces the check, and what the runners are told.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose
from hyperloom.measure.long_context.build import build_entry
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.long_context.tests.support import LENGTH, encode, staged_plan
from hyperloom.measure.short_prompts.staging import CACHE_ENV
from hyperloom_multibranch.toolbox import long_context as long_context_module
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.long_context import LONG_CONTEXT_ENV, LongContextNotStaged, LongContextSetup
from hyperloom_multibranch.toolbox.short_prompts import MODEL_PATH_ENV, SHORT_PROMPTS_ENV
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_long_context_triggers import _model
from hyperloom_multibranch.toolbox.tests.test_short_prompts_option import (
    _acceptance,
    _built,
    _hosted,
    _local,
    _runner_env,
    nothing_may_start,  # noqa: F401 - a fixture of that module
)


@pytest.fixture(autouse=True)
def _no_ambient_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the machine running the suite exports must not decide whether a check is on."""
    for name in (LONG_CONTEXT_ENV, SHORT_PROMPTS_ENV, CACHE_ENV, MODEL_PATH_ENV):
        monkeypatch.delenv(name, raising=False)


def _setup(cache: Path) -> LongContextSetup:
    plan = staged_plan(cache)
    return LongContextSetup(plan.prompt_set, plan.prompts_sha256, plan.length)


async def _planned(toolbox: LocalHostedToolbox, **request: object) -> LongContextPlan | None:
    """The plan acceptance hands the executor for a test with ``request``'s fields, if any."""
    prepared = await toolbox._acceptance.prepare(req(**request))  # type: ignore[arg-type]
    return prepared.environment.long_context  # type: ignore[attr-defined]


async def test_a_toolbox_built_without_a_set_runs_no_check_even_for_a_request_that_forces_it(tmp_path: Path) -> None:
    _setup(tmp_path / "cache")

    toolbox, branch_env = await _built(tmp_path)

    assert toolbox._acceptance._long_context is None
    assert CACHE_ENV not in _runner_env(toolbox) and CACHE_ENV not in branch_env
    assert await _planned(toolbox, long_context_check=True) is None


async def test_a_set_in_the_options_is_carried_by_acceptance_and_the_runner_is_told_where_it_is_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup = _setup(tmp_path / "cache")

    toolbox, branch_env = await _built(tmp_path, long_context=setup)

    assert toolbox._acceptance._long_context == setup
    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())
    assert CACHE_ENV not in branch_env, "the agent's own sandboxes are not told where the prompt set is"
    plan = await _planned(toolbox, long_context_check=True)
    assert plan is not None and (plan.prompt_set, plan.prompts_sha256, plan.length) == (*vars(setup).values(),)
    assert plan.trigger == ("request long_context_check",)
    assert await _planned(toolbox) is None and await _planned(toolbox, long_context_check=False) is None
    assert toolbox._acceptance._short_prompts is None, "the other check stays off"


async def test_a_key_in_the_host_environment_turns_the_check_on_for_a_caller_that_passes_no_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup = _setup(tmp_path / "cache")
    monkeypatch.setenv(LONG_CONTEXT_ENV, setup.prompt_set)

    toolbox, _ = await _built(tmp_path)

    assert toolbox._acceptance._long_context == setup
    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())
    assert await _planned(toolbox, long_context_check=True) is not None


def test_a_set_in_the_environment_that_is_not_staged_refuses_the_build_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nothing_may_start: None
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "empty-cache"))
    monkeypatch.setenv(LONG_CONTEXT_ENV, "long-context-0123456789abcdef")

    with pytest.raises(LongContextNotStaged, match=f"{LONG_CONTEXT_ENV}=long-context-0123456789abcdef.*not staged"):
        _local(tmp_path)


def test_a_set_in_the_options_that_is_not_staged_in_the_runner_cache_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nothing_may_start: None
) -> None:
    setup = _setup(tmp_path / "builder")
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "sandbox-cache"))

    with pytest.raises(LongContextNotStaged, match=f"{setup.prompt_set}.*not staged"):
        _local(tmp_path, long_context=setup)


def test_an_option_whose_hash_or_length_is_not_the_staged_ones_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup = _setup(tmp_path / "cache")

    for wrong in (
        LongContextSetup(setup.prompt_set, "1" * 64, LENGTH),
        LongContextSetup(setup.prompt_set, setup.prompts_sha256, LENGTH * 2),
    ):
        with pytest.raises(LongContextNotStaged, match="other contents or another length|the sheet names"):
            _local(tmp_path / "again", long_context=wrong)


def test_an_option_and_an_acceptance_setup_that_name_different_sets_refuse_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup = _setup(tmp_path / "cache")
    other = LongContextSetup("long-context-other", setup.prompts_sha256, LENGTH)

    with pytest.raises(ValueError, match="name it once"):
        _local(tmp_path, _acceptance(tmp_path, long_context=setup), long_context=other)


def _models_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LongContextSetup:
    """The session model's own set, staged in the runner cache this host names, the model exported as a session does."""
    cache, model = tmp_path / "cache", _model(tmp_path)
    monkeypatch.setenv(CACHE_ENV, str(cache))
    monkeypatch.setenv(MODEL_PATH_ENV, str(model))
    entry = build_entry(model, cache, length=LENGTH, encode=encode)
    return LongContextSetup(entry.key, entry.prompts_sha256, LENGTH)


async def test_nothing_named_runs_the_check_with_the_models_own_set_under_the_rules_triggers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _models_set(tmp_path, monkeypatch)

    toolbox, _ = await _built(tmp_path)

    assert toolbox._acceptance._long_context.at(None) == setup  # type: ignore[union-attr]
    final = await _planned(toolbox, purpose=Purpose.FINAL)
    assert final is not None and final.trigger == ("purpose final",)
    assert final.prompt_set == setup.prompt_set, "a launch that states no context gets the set at the model's limit"
    touching = await _planned(toolbox, server_args="--max-num-seqs 256 --attention-backend aiter")
    assert touching is not None and touching.trigger == ("arg --attention-backend aiter",)
    assert await _planned(toolbox) is None, "a plain explore that touches no lever runs none"


async def test_off_turns_the_check_off_though_the_models_set_is_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _models_set(tmp_path, monkeypatch)
    monkeypatch.setenv(LONG_CONTEXT_ENV, "off")

    toolbox, _ = await _built(tmp_path)

    assert toolbox._acceptance._long_context is None
    assert await _planned(toolbox, purpose=Purpose.FINAL) is None


async def test_a_hosted_toolbox_takes_the_environments_set_and_off_turns_it_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _models_set(tmp_path, monkeypatch)
    for name in ("default", "off"):
        (tmp_path / name).mkdir()

    default = _hosted(tmp_path / "default")._acceptance._long_context
    assert default.at(None) == setup, "the model's own set, found staged"  # type: ignore[union-attr]
    monkeypatch.setenv(LONG_CONTEXT_ENV, "off")
    assert _hosted(tmp_path / "off")._acceptance._long_context is None


async def test_the_local_toolbox_stages_the_models_set_when_a_test_first_needs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, model = tmp_path / "cache", _model(tmp_path)
    monkeypatch.setenv(CACHE_ENV, str(cache))
    monkeypatch.setenv(MODEL_PATH_ENV, str(model))

    def built(model_dir: Path, root: Path, *, length: int) -> object:
        return build_entry(model_dir, root, length=length, encode=encode)

    monkeypatch.setattr(long_context_module, "build_entry", built)

    toolbox, _ = await _built(tmp_path)

    assert not (cache / "evalsets").exists(), "nothing is staged before a test says what context its launch serves"
    plan = await _planned(toolbox, purpose=Purpose.FINAL)
    assert plan is not None and plan.length == LENGTH and (cache / "evalsets" / plan.prompt_set).is_dir()
