# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A hosted toolbox's tests run the short-prompt and long-context checks with the session model's own prompt sets.

A hosted session's host names its runner cache in ``HYPERLOOM_RUNNER_CACHE``: it builds the commit's entry there, in the
directory the runner sandboxes mount (maybe at another path). A hosted toolbox that sees that directory stages the
model's sets there when they are missing, as the local one does: ``<cache>/evalsets/<key>``, the key a hash of the generator and its inputs (the
tokenizer's file hashes, the seed, the long-context length) and the prompts named by their SHA-256 in every sheet. The
backend here is the hosted one over the local sandbox provider; the stand-in runner reads the sets with the runner's own
check of the call (``fake_reads_prompt_sets``), in the cache its template names, and lists what it read in the bundle.
A set the runner does not find, or finds with other prompts, refuses the test and names the set and why.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from hyperloom.contracts.consistency import SHORT_PROMPTS_COMPLETE
from hyperloom.contracts.long_context import LONG_CONTEXT_COMPLETE
from hyperloom.contracts.measurement import Purpose, TestResult, Validity
from hyperloom.measure.long_context import build as long_build
from hyperloom.measure.long_context.plan import REPLY_HEADROOM
from hyperloom.measure.long_context.tests.support import LENGTH, encode as long_encode
from hyperloom.measure.short_prompts import build as short_build
from hyperloom.measure.short_prompts.staging import CACHE_ENV, DATA_DIR, EVALSETS_DIR, PROMPTS_FILE
from hyperloom.measure.short_prompts.tests.support import encode as short_encode
from hyperloom.measure.tests.fake_runner import PROMPT_SETS_READ
from hyperloom.measure.tests.hosted_fixtures import FAKE_RUNNER, Rig, make_rig, point_yaml
from hyperloom.sandbox.templates import LocalTemplate
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import FakeClock
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox import hosted_toolbox, long_context, short_prompts
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions
from hyperloom_multibranch.toolbox.long_context import (
    LONG_CONTEXT_ENV,
    LongContextSetup,
    ServedLongContext,
    sees_runner_cache,
)
from hyperloom_multibranch.toolbox.short_prompts import MODEL_PATH_ENV, SHORT_PROMPTS_ENV
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox, ToolboxStores, ToolboxTuning

#: The model's own limit: its long-context set is built at ``LENGTH`` tokens, the test tokenizer's set.
_MODEL_LIMIT = LENGTH + REPLY_HEADROOM
_RUNS_BOTH = {
    "fake_accuracy": "0.64",
    "fake_reads_prompt_sets": True,
    "fake_short_prompts": "good",
    "fake_long_context": {},
}


@dataclass
class Host:
    """A hosted session's host: the model it serves, its view of the runner cache and the environment it runs in."""

    model: Path
    cache: Path
    environ: dict[str, str]


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    """The model (a tokenizer file and its context limit) and the host's environment, the sets built with the test
    tokenizers in place of the model's."""
    for name in (SHORT_PROMPTS_ENV, LONG_CONTEXT_ENV, MODEL_PATH_ENV, CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(short_prompts, "build_entry", functools.partial(short_build.build_entry, encode=short_encode))

    def long_built(model_dir: Path, root: Path, *, length: int) -> object:
        return long_build.build_entry(model_dir, root, length=length, encode=long_encode)

    monkeypatch.setattr(long_context, "build_entry", long_built)
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_bytes(b'{"model": "hosted"}')
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": _MODEL_LIMIT}), encoding="utf-8")
    cache = tmp_path / "shared" / "runner-cache"
    cache.mkdir(parents=True)
    return Host(model, cache, {CACHE_ENV: str(cache), MODEL_PATH_ENV: str(model)})


def _sandbox_view(tmp_path: Path, host: Host) -> Path:
    """The runner cache as the runner sandboxes mount it: the host's directory, at a path of their own."""
    view = tmp_path / "sandbox-mount"
    view.symlink_to(host.cache, target_is_directory=True)
    return view


def _rig(tmp_path: Path, cache_in_sandbox: Path) -> Rig:
    """The hosted backend over the local provider; the runner template mounts the cache at ``cache_in_sandbox``."""
    template = LocalTemplate(
        "runner-1",
        "measurement",
        command=FAKE_RUNNER,
        gpus=1,
        sealed=True,
        signing=True,
        envs={CACHE_ENV: str(cache_in_sandbox)},
    )
    return make_rig(tmp_path, templates=[template])


def _hosted_acceptance(host: Host, point: Path, workload: WorkloadKind = WorkloadKind.AGENTX) -> AcceptanceSetup:
    """The acceptance setup ``build_hosted_toolbox`` makes from the default options and the host's environment."""
    base = AcceptanceSetup(FakeRepo(), PointPlanner(point), IMAGE, workload)
    return hosted_toolbox._acceptance_with_prompts(base, HostedOptions(), host.environ, None)


def _toolbox(tmp_path: Path, rig: Rig, acceptance: AcceptanceSetup) -> RealToolbox:
    clock = FakeClock()
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    stores = ToolboxStores(
        log=SubmissionLog(tmp_path / "toolbox" / "submissions.jsonl", "campaign-1"),
        sink=blackboard,
        health=blackboard,
        claims=blackboard,
        records=blackboard,
    )
    executor = HostedExecutor(rig.backend, tmp_path / "session")
    toolbox = RealToolbox(stores, executor, acceptance, tuning=ToolboxTuning(policy=POLICY, clock=clock))
    toolbox.trace_root = tmp_path / "session"
    return toolbox


async def _final_test(toolbox: RealToolbox) -> TestResult:
    await toolbox.start()
    try:
        request_id = await toolbox.submit_test(req(purpose=Purpose.FINAL))
        await asyncio.wait_for(toolbox.drain(), 60)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
        return TestResult.from_document(entry.outcome["result"])
    finally:
        await toolbox.close()


async def test_a_hosted_test_runs_both_checks_with_the_model_sets_the_host_staged_where_the_runner_reads(
    tmp_path: Path, host: Host
) -> None:
    rig = _rig(tmp_path, _sandbox_view(tmp_path, host))
    point = point_yaml(tmp_path, "planned.yaml", **_RUNS_BOTH)
    try:
        # A fixed workload: an AgentX campaign runs no long-context check, its servers are not held up for one.
        acceptance = _hosted_acceptance(host, point, WorkloadKind.INFERENCEX)
        short, served = acceptance.short_prompts, acceptance.long_context
        assert short is not None and isinstance(served, ServedLongContext), "both checks are on with the model's sets"
        assert (host.cache / EVALSETS_DIR / short.prompt_set).is_dir(), "the short set is staged at the build"

        result = await _final_test(_toolbox(tmp_path, rig, acceptance))
    finally:
        rig.backend.close()
        rig.provider.close()

    assert result.validity is Validity.SUCCEEDED, result.failure_reason
    consistency, retrieval = result.short_prompt_consistency, result.long_context
    assert consistency is not None and consistency.status == SHORT_PROMPTS_COMPLETE
    assert consistency.prompt_set_sha256 == short.prompts_sha256
    long_set = served.at(None)
    assert isinstance(long_set, LongContextSetup) and long_set.length == LENGTH
    assert (host.cache / EVALSETS_DIR / long_set.prompt_set).is_dir(), "the long set is staged when a test plans it"
    assert retrieval is not None and retrieval.status == LONG_CONTEXT_COMPLETE
    assert retrieval.prompt_set_sha256 == long_set.prompts_sha256 and retrieval.length == LENGTH
    read = json.loads((tmp_path / "session" / result.output_dir / PROMPT_SETS_READ).read_bytes())
    assert read["cache"] == str(tmp_path / "sandbox-mount"), "the runner read its own mount, never the host's path"
    assert read["short_prompts"]["key"] == short.prompt_set and read["short_prompts"]["prompts"] > 0
    assert read["short_prompts"]["sha256"] == short.prompts_sha256
    assert (read["long_context"]["key"], read["long_context"]["sha256"]) == (
        long_set.prompt_set,
        long_set.prompts_sha256,
    )
    assert read["long_context"]["prompts"] > 0


async def test_a_set_the_runner_does_not_find_refuses_the_test_and_names_the_set(tmp_path: Path, host: Host) -> None:
    """The sandbox mounts another, empty cache: the sheet names the sets, and the runner does not find them."""
    elsewhere = tmp_path / "another-cache"
    elsewhere.mkdir()
    rig = _rig(tmp_path, elsewhere)
    point = point_yaml(tmp_path, "planned.yaml", **_RUNS_BOTH)
    try:
        acceptance = _hosted_acceptance(host, point)
        result = await _final_test(_toolbox(tmp_path, rig, acceptance))
    finally:
        rig.backend.close()
        rig.provider.close()

    assert acceptance.short_prompts is not None
    assert result.validity is Validity.MEASUREMENT_FAILED and result.short_prompt_consistency is None
    reason = result.failure_reason or ""
    assert "measurement refused: spec_rejected" in reason
    assert f"the short-prompt set {acceptance.short_prompts.prompt_set} cannot be used" in reason
    assert "not staged, or unfinished" in reason


async def test_a_set_whose_prompts_changed_in_the_cache_refuses_the_test_and_says_so(
    tmp_path: Path, host: Host
) -> None:
    rig = _rig(tmp_path, _sandbox_view(tmp_path, host))
    point = point_yaml(tmp_path, "planned.yaml", **_RUNS_BOTH)
    try:
        acceptance = _hosted_acceptance(host, point)
        assert acceptance.short_prompts is not None
        prompts = host.cache / EVALSETS_DIR / acceptance.short_prompts.prompt_set / DATA_DIR / PROMPTS_FILE
        prompts.chmod(0o644)
        prompts.write_bytes(prompts.read_bytes().replace(b"1", b"2", 1))
        result = await _final_test(_toolbox(tmp_path, rig, acceptance))
    finally:
        rig.backend.close()
        rig.provider.close()

    reason = result.failure_reason or ""
    assert result.validity is Validity.MEASUREMENT_FAILED and "measurement refused: spec_rejected" in reason
    assert f"the short-prompt set {acceptance.short_prompts.prompt_set} cannot be used" in reason
    assert "is not what its manifest recorded" in reason


@pytest.mark.parametrize("named", ["missing", "runner-cache"], ids=["absent", "relative-absent"])
def test_a_host_that_does_not_see_the_runner_cache_stages_nothing_and_says_why_the_check_is_off(
    tmp_path: Path, host: Host, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, named: str
) -> None:
    """What this host would stage where it does not see the cache would be where no runner reads it."""
    monkeypatch.chdir(tmp_path)
    cache = str(tmp_path / named) if named == "missing" else named
    environ = {MODEL_PATH_ENV: str(host.model), CACHE_ENV: cache}

    with caplog.at_level(logging.WARNING, logger=short_prompts.__name__):
        acceptance = hosted_toolbox._acceptance_with_prompts(
            AcceptanceSetup(FakeRepo(), PointPlanner(tmp_path / "p.yaml"), IMAGE, WorkloadKind.AGENTX),
            HostedOptions(),
            environ,
            None,
        )

    assert not sees_runner_cache(environ)
    assert acceptance.short_prompts is None and "the toolbox does not stage it" in caplog.text
    assert not (tmp_path / named).exists()


def test_a_host_that_sees_its_runner_cache_stages_the_short_prompt_set_there(tmp_path: Path, host: Host) -> None:
    acceptance = hosted_toolbox._acceptance_with_prompts(
        AcceptanceSetup(FakeRepo(), PointPlanner(tmp_path / "p.yaml"), IMAGE, WorkloadKind.AGENTX),
        HostedOptions(),
        host.environ,
        None,
    )

    assert sees_runner_cache(host.environ)
    assert acceptance.short_prompts is not None
    assert (host.cache / EVALSETS_DIR / acceptance.short_prompts.prompt_set / DATA_DIR / PROMPTS_FILE).is_file()
