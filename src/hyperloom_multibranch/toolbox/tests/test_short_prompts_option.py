# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt consistency check is on by default in both toolbox builders: with the set a host names, else the set
staged for the session model's tokenizer; ``off`` turns it off.

What is proved here is the wiring: which set the acceptance step carries, where the runners are told to read it, and that a
set that is asked for and not staged refuses the build. That the check then runs and what it says is proved by the
short-prompt tests of the measure package and of the toolbox.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import Purpose
from hyperloom.measure.short_prompts.build import DEFAULT_SEED, tokenizer_hashes
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.staging import CACHE_ENV, EVALSETS_DIR, MANIFEST_NAME, stage_entry
from hyperloom.measure.short_prompts.tests.support import prompt_set, staged_plan
from hyperloom.measure.tests.hosted_fixtures import point_yaml, spec_context
from hyperloom.measure.tests.test_hosted_real_runner import _point
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions, HostedToolbox, build_hosted_toolbox
from hyperloom_multibranch.toolbox.leased_toolbox import with_short_prompts
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.short_prompts import (
    MODEL_PATH_ENV,
    SHORT_PROMPTS_ENV,
    SHORT_PROMPTS_OFF,
    ShortPromptSetup,
    ShortPromptsNotStaged,
    default_short_prompts,
    staged_short_prompts,
)
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.tests.test_hosted_toolbox import _inputs
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

_RUNNER = "runner-1"
#: A change the rule pack's pattern does not read as touching attention, the KV cache or prefill, a purpose it does not always check.
_QUIET = {"server_args": "--max-num-seqs 256"}
_TOUCHING = {"server_args": "--attention-backend aiter"}


@pytest.fixture(autouse=True)
def _no_ambient_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the machine running the suite exports must not decide whether the check is on."""
    monkeypatch.delenv(SHORT_PROMPTS_ENV, raising=False)
    monkeypatch.delenv(CACHE_ENV, raising=False)
    monkeypatch.delenv(MODEL_PATH_ENV, raising=False)


@pytest.fixture
def nothing_may_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build that makes the sandbox provider or starts the backend before it has refused fails the test it is in."""

    def started(*args: object, **kwargs: object) -> None:
        raise AssertionError("the provider or the backend was made before the short-prompt set was checked")

    monkeypatch.setattr(local_hosted, "LocalSandboxProvider", started)
    monkeypatch.setattr(local_hosted, "started_backend", started)


def _setup(cache: Path) -> ShortPromptSetup:
    plan = staged_plan(cache)
    return ShortPromptSetup(plan.prompt_set, plan.prompts_sha256)


def _other_setup(cache: Path) -> ShortPromptSetup:
    """A second staged entry: the same prompts under another key, as another seed's would be."""
    entry = stage_entry(cache, prompt_set(), {"seed": 2, "tokenizer": {"fake": "0" * 64}})
    return ShortPromptSetup(entry.key, entry.prompts_sha256)


def _model(tmp_path: Path, tokenizer: bytes = b'{"model": "fake"}') -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "tokenizer.json").write_bytes(tokenizer)
    return model


def _model_set(model: Path, cache: Path) -> ShortPromptSetup:
    """The set ``measure.short_prompts.build`` stages for ``model``'s tokenizer with its default seed."""
    entry = stage_entry(cache, prompt_set(), {"seed": DEFAULT_SEED, "tokenizer": tokenizer_hashes(model)})
    return ShortPromptSetup(entry.key, entry.prompts_sha256)


# --- the setup and the environment ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "sha"), [("../outside", "a" * 64), ("Not A Key", "a" * 64), ("short-prompts-x", "z")])
def test_a_setup_no_sheet_could_carry_is_refused_when_it_is_made(key: str, sha: str) -> None:
    with pytest.raises(ValueError, match="short prompts: .* is not a"):
        ShortPromptSetup(key, sha)


def test_nothing_named_and_no_model_means_no_check_and_a_blank_variable_names_nothing(tmp_path: Path) -> None:
    assert staged_short_prompts(None, {CACHE_ENV: str(tmp_path)}) is None
    assert staged_short_prompts(None, {CACHE_ENV: str(tmp_path), SHORT_PROMPTS_ENV: "  "}) is None


@pytest.mark.parametrize("by_variable", [False, True], ids=["spec-model-dir", "MODEL_PATH"])
def test_nothing_named_turns_the_check_on_with_the_set_staged_for_the_models_tokenizer(
    tmp_path: Path, by_variable: bool
) -> None:
    model, cache = _model(tmp_path), tmp_path / "cache"
    setup = _model_set(model, cache)
    _other_setup(cache)  # another tokenizer's set beside it is not taken
    environ = {CACHE_ENV: str(cache), **({MODEL_PATH_ENV: str(model)} if by_variable else {})}

    assert staged_short_prompts(None, environ, None if by_variable else model) == setup


def test_off_turns_the_check_off_though_the_models_set_is_staged(tmp_path: Path) -> None:
    model, cache = _model(tmp_path), tmp_path / "cache"
    _model_set(model, cache)

    assert staged_short_prompts(None, {CACHE_ENV: str(cache), SHORT_PROMPTS_ENV: SHORT_PROMPTS_OFF}, model) is None


def test_another_tokenizers_set_is_not_the_models_and_leaves_the_check_off_saying_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache = tmp_path / "cache"
    _model_set(_model(tmp_path / "a", b'{"model": "a"}'), cache)

    assert default_short_prompts(_model(tmp_path / "b", b'{"model": "b"}'), {CACHE_ENV: str(cache)}) is None
    assert "is not staged in the runner cache" in caplog.text


def test_a_model_directory_with_no_tokenizer_leaves_the_check_off_saying_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "model").mkdir()

    assert default_short_prompts(tmp_path / "model", {CACHE_ENV: str(tmp_path)}) is None
    assert "the model's tokenizer cannot be read" in caplog.text


def test_the_models_set_found_unfinished_refuses_rather_than_turning_the_check_off(tmp_path: Path) -> None:
    model, cache = _model(tmp_path), tmp_path / "cache"
    setup = _model_set(model, cache)
    _unfinished(cache, setup)

    with pytest.raises(ShortPromptsNotStaged, match=f"the model's prompt set {setup.prompt_set}.*unfinished"):
        default_short_prompts(model, {CACHE_ENV: str(cache)})


def test_a_key_in_the_environment_takes_its_hash_from_the_staged_manifest(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    found = staged_short_prompts(None, {CACHE_ENV: str(tmp_path), SHORT_PROMPTS_ENV: f" {setup.prompt_set} "})

    assert found == setup


def test_the_option_wins_over_the_environment_and_the_environment_is_not_even_read(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    found = staged_short_prompts(setup, {CACHE_ENV: str(tmp_path), SHORT_PROMPTS_ENV: "short-prompts-not-staged"})

    assert found == setup


def test_an_option_that_is_staged_is_returned_as_the_entry_reads(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    assert staged_short_prompts(setup, {CACHE_ENV: str(tmp_path)}) == setup


def _tampered(cache: Path, setup: ShortPromptSetup) -> None:
    data = cache / EVALSETS_DIR / setup.prompt_set / "data" / "prompts.jsonl"
    data.write_bytes(data.read_bytes().replace(b"sequential", b"batched", 1))


def _unfinished(cache: Path, setup: ShortPromptSetup) -> None:
    (cache / EVALSETS_DIR / setup.prompt_set / MANIFEST_NAME).unlink()


@pytest.mark.parametrize(
    ("damage", "reason"),
    [(_tampered, "not what its manifest recorded"), (_unfinished, "not staged, or unfinished")],
)
@pytest.mark.parametrize("by_option", [True, False], ids=["option", "environment"])
def test_a_set_that_is_asked_for_and_staged_badly_is_refused_saying_why(
    tmp_path: Path, damage: Any, reason: str, by_option: bool
) -> None:
    setup = _setup(tmp_path)
    damage(tmp_path, setup)
    environ = {CACHE_ENV: str(tmp_path), **({} if by_option else {SHORT_PROMPTS_ENV: setup.prompt_set})}

    with pytest.raises(ShortPromptsNotStaged, match=f"asked for.*{setup.prompt_set}.*not staged.*{reason}"):
        staged_short_prompts(setup if by_option else None, environ)


@pytest.mark.parametrize("by_option", [True, False], ids=["option", "environment"])
def test_a_set_that_is_asked_for_and_not_in_the_cache_at_all_is_refused_naming_the_cache(
    tmp_path: Path, by_option: bool
) -> None:
    setup = _setup(tmp_path / "builder")
    environ = {CACHE_ENV: str(tmp_path / "empty"), **({} if by_option else {SHORT_PROMPTS_ENV: setup.prompt_set})}

    with pytest.raises(ShortPromptsNotStaged, match=f"{setup.prompt_set}.*not staged in the runner cache"):
        staged_short_prompts(setup if by_option else None, environ)


def test_an_option_whose_hash_is_not_the_one_staged_is_refused_with_both_hashes(tmp_path: Path) -> None:
    staged = _setup(tmp_path)
    wrong = ShortPromptSetup(staged.prompt_set, "1" * 64)

    with pytest.raises(
        ShortPromptsNotStaged, match=f"holds prompts {staged.prompts_sha256}, the sheet names {'1' * 64}"
    ):
        staged_short_prompts(wrong, {CACHE_ENV: str(tmp_path)})


def test_a_key_in_the_environment_that_is_not_a_plain_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ShortPromptsNotStaged, match="is not a prompt set key"):
        staged_short_prompts(None, {CACHE_ENV: str(tmp_path), SHORT_PROMPTS_ENV: "../outside"})


# --- what a builder does with an acceptance setup that names one ----------------------------------------------------


def _acceptance(tmp_path: Path, **fields: Any) -> AcceptanceSetup:
    return AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX, **fields)


def test_the_options_set_is_given_to_an_acceptance_setup_that_names_none(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    acceptance = _acceptance(tmp_path)

    assert with_short_prompts(acceptance, None) is acceptance
    assert with_short_prompts(acceptance, setup).short_prompts == setup
    named = _acceptance(tmp_path, short_prompts=setup)
    assert with_short_prompts(named, setup) is named and with_short_prompts(named, None) is named


def test_two_different_sets_are_refused_so_the_set_is_named_once(tmp_path: Path) -> None:
    setup, other = _setup(tmp_path), _other_setup(tmp_path)

    with pytest.raises(ValueError, match=f"options name the short-prompt set {other.prompt_set}.*{setup.prompt_set}"):
        with_short_prompts(_acceptance(tmp_path, short_prompts=setup), other)


# --- the local toolbox ----------------------------------------------------------------------------------------------


def _local(tmp_path: Path, acceptance: AcceptanceSetup | None = None, **options: Any) -> LocalHostedToolbox:
    session = tmp_path / "session"
    spec = spec_context(tmp_path, blob_root=str(session / "hosted" / "blobs"))
    return build_local_toolbox(
        LocalSession(session, "run-1", range(8)),
        LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox")),
        acceptance or _acceptance(tmp_path),
        spec,
        options=LocalOptions(
            policy=QueuePolicy(per_branch_k=2, global_concurrency=64), poll_interval_sec=0.1, **options
        ),
    )


async def _built(
    tmp_path: Path, acceptance: AcceptanceSetup | None = None, **options: Any
) -> tuple[LocalHostedToolbox, dict[str, str]]:
    """A toolbox built with ``options``, its facts read and closed: ``(toolbox, what a branch sandbox gets)``."""
    toolbox = _local(tmp_path, acceptance, **options)
    try:
        branch_env = dict((await toolbox.lease_branch_sandbox("branch-1", 1)).env)
    finally:
        await toolbox.close()
    return toolbox, branch_env


def _runner_env(toolbox: LocalHostedToolbox) -> dict[str, str]:
    return dict(toolbox.provider._templates[_RUNNER].envs)


async def _planned(toolbox: LocalHostedToolbox | HostedToolbox, **request: Any) -> ShortPromptPlan | None:
    """The short-prompt plan acceptance hands the executor for a test with ``request``'s fields, if any."""
    prepared = await toolbox._acceptance.prepare(req(**request))
    return prepared.environment.short_prompts  # type: ignore[attr-defined]


async def test_a_toolbox_built_without_a_set_runs_no_check_even_for_a_test_the_rule_always_checks(
    tmp_path: Path,
) -> None:
    _setup(tmp_path / "cache")

    toolbox, branch_env = await _built(tmp_path)

    assert toolbox._acceptance._short_prompts is None
    assert CACHE_ENV not in _runner_env(toolbox) and CACHE_ENV not in branch_env
    assert await _planned(toolbox, purpose=Purpose.FINAL) is None
    assert await _planned(toolbox, **_TOUCHING) is None


async def test_a_set_in_the_options_is_carried_by_acceptance_and_the_runner_is_told_where_it_is_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup = _setup(tmp_path / "cache")

    toolbox, branch_env = await _built(tmp_path, short_prompts=setup)

    assert toolbox._acceptance._short_prompts == setup
    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())
    assert CACHE_ENV not in branch_env, "the agent's own sandboxes are not told where the prompt set is"
    plan = await _planned(toolbox, purpose=Purpose.FINAL)
    assert plan is not None and (plan.prompt_set, plan.prompts_sha256) == (setup.prompt_set, setup.prompts_sha256)
    assert plan.trigger == ("purpose final",)


async def test_a_key_in_the_host_environment_turns_the_check_on_for_a_caller_that_passes_no_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(tmp_path / "cache")
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(SHORT_PROMPTS_ENV, setup.prompt_set)

    toolbox, _ = await _built(tmp_path)

    assert toolbox._acceptance._short_prompts == setup, "the hash is the staged manifest's"
    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())
    plan = await _planned(toolbox, **_TOUCHING)
    assert plan is not None and "arg --attention-backend aiter" in plan.trigger


async def test_a_toolbox_with_the_check_on_still_leaves_to_the_rule_which_tests_run_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    toolbox, _ = await _built(tmp_path, short_prompts=_setup(tmp_path / "cache"))

    assert await _planned(toolbox, **_QUIET) is None, "nothing touches attention, the KV cache or prefill"
    forced = await _planned(toolbox, consistency_check=True, **_QUIET)
    assert forced is not None and forced.trigger == ("request consistency_check",), "a caller may force it"


async def test_a_relative_runner_cache_is_made_absolute_because_the_runner_starts_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(tmp_path / "cache")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CACHE_ENV, "cache")

    toolbox, _ = await _built(tmp_path, short_prompts=setup)

    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())


async def test_the_runner_reads_the_cache_the_set_was_checked_in_even_when_the_runner_environment_names_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))

    toolbox, _ = await _built(
        tmp_path, short_prompts=_setup(tmp_path / "cache"), runner_envs={CACHE_ENV: str(tmp_path / "elsewhere")}
    )

    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())


async def test_a_set_the_acceptance_setup_names_is_checked_and_its_cache_told_to_the_runner_like_an_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup = _setup(tmp_path / "cache")

    toolbox, _ = await _built(tmp_path, _acceptance(tmp_path, short_prompts=setup))

    assert toolbox._acceptance._short_prompts == setup
    assert _runner_env(toolbox)[CACHE_ENV] == str((tmp_path / "cache").resolve())
    with pytest.raises(ShortPromptsNotStaged, match="not staged"):
        _local(tmp_path / "again", _acceptance(tmp_path, short_prompts=ShortPromptSetup(setup.prompt_set, "1" * 64)))


def test_a_set_in_the_environment_that_is_not_staged_refuses_the_build_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nothing_may_start: None
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "empty-cache"))
    monkeypatch.setenv(SHORT_PROMPTS_ENV, "short-prompts-0123456789abcdef")

    with pytest.raises(ShortPromptsNotStaged, match=f"{SHORT_PROMPTS_ENV}=short-prompts-0123456789abcdef.*not staged"):
        _local(tmp_path)


def test_a_set_in_the_options_that_is_not_staged_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nothing_may_start: None
) -> None:
    setup = _setup(tmp_path / "builder")
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "sandbox-cache"))

    with pytest.raises(ShortPromptsNotStaged, match=f"{setup.prompt_set}.*not staged"):
        _local(tmp_path, short_prompts=setup)


def test_an_option_and_an_acceptance_setup_that_name_different_sets_refuse_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    setup, other = _setup(tmp_path / "cache"), _other_setup(tmp_path / "cache")

    with pytest.raises(ValueError, match="name it once"):
        _local(tmp_path, _acceptance(tmp_path, short_prompts=setup), short_prompts=other)


# --- the hosted toolbox ---------------------------------------------------------------------------------------------


def _hosted(
    tmp_path: Path, acceptance: AcceptanceSetup | None = None, options: HostedOptions | None = None
) -> HostedToolbox:
    """The hosted toolbox over plain values; no service is reached until it starts."""
    point = point_yaml(tmp_path, "planned.yaml")
    return build_hosted_toolbox(
        _inputs(None, tmp_path),
        LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox")),
        acceptance or AcceptanceSetup(FakeRepo(), PointPlanner(point), IMAGE, WorkloadKind.AGENTX),
        spec_context(tmp_path),
        options=options or HostedOptions(policy=QueuePolicy(per_branch_k=2, global_concurrency=64)),
    )


async def test_a_hosted_toolbox_built_without_a_set_runs_no_check(tmp_path: Path) -> None:
    toolbox = _hosted(tmp_path)

    assert toolbox._acceptance._short_prompts is None
    assert await _planned(toolbox, purpose=Purpose.FINAL) is None


async def test_a_set_in_the_hosted_options_is_carried_by_acceptance_as_given(tmp_path: Path) -> None:
    setup = ShortPromptSetup("short-prompts-0123456789abcdef", "a" * 64)

    toolbox = _hosted(tmp_path, options=HostedOptions(short_prompts=setup))

    assert toolbox._acceptance._short_prompts == setup, "the runner cache is the service's: nothing here reads it"
    plan = await _planned(toolbox, purpose=Purpose.FINAL)
    assert plan is not None and (plan.prompt_set, plan.prompts_sha256) == (setup.prompt_set, setup.prompts_sha256)
    assert plan.trigger == ("purpose final",)


async def test_a_key_in_the_host_environment_turns_the_check_on_for_a_hosted_caller_that_passes_no_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hosted peers build the toolbox with the default options, so the environment is how an operator turns it on."""
    setup = _setup(tmp_path / "cache")
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(SHORT_PROMPTS_ENV, setup.prompt_set)

    toolbox = _hosted(tmp_path)

    assert toolbox._acceptance._short_prompts == setup, "the hash is the staged manifest's"
    plan = await _planned(toolbox, **_TOUCHING)
    assert plan is not None and "arg --attention-backend aiter" in plan.trigger


def test_a_key_in_the_host_environment_that_is_not_staged_in_the_hosts_cache_refuses_the_hosted_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "empty-cache"))
    monkeypatch.setenv(SHORT_PROMPTS_ENV, "short-prompts-0123456789abcdef")

    with pytest.raises(ShortPromptsNotStaged, match=f"{SHORT_PROMPTS_ENV}=short-prompts-0123456789abcdef.*not staged"):
        _hosted(tmp_path)


async def test_a_hosted_option_is_carried_as_given_and_wins_over_the_environment_without_reading_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _setup(tmp_path / "cache")
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(SHORT_PROMPTS_ENV, staged.prompt_set)
    given = ShortPromptSetup("short-prompts-0123456789abcdef", "a" * 64)  # staged nowhere this host can see

    toolbox = _hosted(tmp_path, options=HostedOptions(short_prompts=given))

    assert toolbox._acceptance._short_prompts == given


async def test_a_hosted_acceptance_setup_that_names_a_set_is_not_replaced_by_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = _setup(tmp_path / "cache")
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(SHORT_PROMPTS_ENV, other.prompt_set)
    named = ShortPromptSetup("short-prompts-0123456789abcdef", "a" * 64)
    acceptance = AcceptanceSetup(
        FakeRepo(), PointPlanner(point_yaml(tmp_path, "planned.yaml")), IMAGE, WorkloadKind.AGENTX, short_prompts=named
    )

    toolbox = _hosted(tmp_path, acceptance)

    assert toolbox._acceptance._short_prompts == named


async def test_both_builders_with_no_options_run_the_check_with_the_session_models_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The peers build both toolboxes with default options and export the model's directory as ``MODEL_PATH``."""
    model, cache = _model(tmp_path), tmp_path / "cache"
    setup = _model_set(model, cache)
    monkeypatch.setenv(CACHE_ENV, str(cache))
    monkeypatch.setenv(MODEL_PATH_ENV, str(model))

    for name in ("local", "hosted", "off"):
        (tmp_path / name).mkdir()
    local, _ = await _built(tmp_path / "local")
    hosted = _hosted(tmp_path / "hosted")

    for toolbox in (local, hosted):
        assert toolbox._acceptance._short_prompts == setup
        plan = await _planned(toolbox, purpose=Purpose.FINAL)
        assert plan is not None and plan.prompt_set == setup.prompt_set and plan.processes == 6
        assert await _planned(toolbox, **_QUIET) is None, "on by default, still only under the triggers"
    monkeypatch.setenv(SHORT_PROMPTS_ENV, SHORT_PROMPTS_OFF)
    assert _hosted(tmp_path / "off")._acceptance._short_prompts is None


def test_a_hosted_option_and_an_acceptance_setup_that_name_different_sets_refuse_the_build(tmp_path: Path) -> None:
    first = ShortPromptSetup("short-prompts-0123456789abcdef", "a" * 64)
    second = dataclasses.replace(first, prompts_sha256="b" * 64)
    named = AcceptanceSetup(
        FakeRepo(), PointPlanner(point_yaml(tmp_path, "planned.yaml")), IMAGE, WorkloadKind.AGENTX, short_prompts=first
    )

    with pytest.raises(ValueError, match="name it once"):
        _hosted(tmp_path, named, HostedOptions(short_prompts=second))
