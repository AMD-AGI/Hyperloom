# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The model's own long-context set is built at the context the test's launch serves.

Acceptance plans the test, reads the context its planned launch serves (its context flag, else ``MAX_MODEL_LEN``) and
gives it the model's set for that context, staged in the runner cache keyed by its length; the check then runs on a fake
server with the runner's own reading of that set. A launch at the model's full window keeps the set at the model's limit;
a 1M-token launch gets its set at the bound, not at 1M tokens a prompt. An AgentX campaign gets one on the tests its gate
reads (BASELINE, every integration merge TEST, FINAL's TEST) and none, staging nothing, on any other.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom.contracts.long_context import LONG_CONTEXT_COMPLETE, PROMPT_OK, LongContextResult
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.measure.long_context import build as build_module
from hyperloom.measure.long_context.build import build_entry
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.long_context.rounds import CheckServer, run_long_context
from hyperloom.measure.long_context.staging import load_prompt_set
from hyperloom.measure.long_context.tests.support import FakeServer, encode
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox import long_context as long_context_module
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.long_context import (
    ServedLongContext,
    default_long_context,
    launch_served_context,
    long_context_of,
)
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_long_context import _Host
from hyperloom_multibranch.toolbox.tests.test_short_prompts import _Planner

#: A model's context limit (a dense model's public config): the prompts a server limited to less were all ``not_run``.
MODEL_LIMIT = 40960
SERVED = 8192
HEADROOM = 128
#: A 1M-token launch (a model whose own limit is 1M), and the bound a set is built under, written out: the oracle does not
#: follow the code's constant.
MILLION = 1 << 20
BOUND = 131072
PROMPTS = 16


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The lengths the toolbox staged a set at, built with the test tokenizer in place of the model's."""
    lengths: list[int] = []

    def built(model_dir: Path, root: Path, *, length: int) -> object:
        lengths.append(length)
        return build_entry(model_dir, root, length=length, encode=encode)

    monkeypatch.setattr(long_context_module, "build_entry", built)
    return lengths


def _model(tmp_path: Path, limit: int = MODEL_LIMIT) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_bytes(b'{"model": "fake"}')
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": limit}))
    return model


def _point(tmp_path: Path, name: str, server_args: str = "", **envs: int) -> Path:
    """A rendered SGLang point whose launch carries ``envs`` and the server arguments ``server_args``."""
    point = tmp_path / f"{name}.yaml"
    carried = {"EXTRA_SGLANG_ARGS": server_args} if server_args else {}
    point.write_text(json.dumps({"benchmark": {"framework": "sglang", "envs": {**envs, **carried}}}))
    return point


class _Session:
    """A toolbox's acceptance with the model's own set (no set named), its runner cache below ``tmp_path``."""

    def __init__(self, tmp_path: Path, limit: int = MODEL_LIMIT) -> None:
        self.environ = {"HYPERLOOM_RUNNER_CACHE": str(tmp_path / "cache")}
        self.model = _model(tmp_path, limit)
        self.source = ServedLongContext(self.model, self.environ, stage=True)

    def planned(
        self,
        point: Path,
        purpose: Purpose = Purpose.BASELINE,
        workload: WorkloadKind = WorkloadKind.INFERENCEX,
        **request: object,
    ) -> LongContextPlan | None:
        """The check the test of ``point`` for ``purpose`` is planned with; the request forces it unless ``request``
        says otherwise."""
        acceptance = Acceptance(
            AcceptanceSetup(
                FakeRepo(None),  # type: ignore[arg-type]
                _Planner(AcceptedMeasurement(None, (point,))),  # type: ignore[arg-type]
                IMAGE,
                workload,
                long_context=self.source,
            ),
            is_accepted=lambda _id: True,
        )
        submitted = req(purpose=purpose, **{"long_context_check": True, **request})  # type: ignore[arg-type]
        planned = asyncio.run(acceptance.prepare(submitted)).environment
        assert isinstance(planned, AcceptedMeasurement)
        return planned.long_context

    def plan(self, point: Path, purpose: Purpose = Purpose.BASELINE) -> LongContextPlan:
        planned = self.planned(point, purpose)
        assert planned is not None
        return planned


def _checked(plan: LongContextPlan, point: Path, session: _Session, out: Path):  # type: ignore[no-untyped-def]
    """The check of ``plan`` run as the runner runs it: the set read from the cache, the served context from the point."""
    prompts = load_prompt_set(plan, session.environ)
    with FakeServer() as fake:
        asyncio.run(run_long_context(plan, prompts, _Host(fake), out, server=CheckServer(launch_served_context(point))))
    return long_context_of(plan, out)


def test_a_launch_served_at_8192_gets_a_set_of_that_window_that_runs_on_its_server(
    tmp_path: Path, builds: list[int]
) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "served-8192", MAX_MODEL_LEN=SERVED)

    plan = session.plan(point)

    assert plan.length == SERVED - HEADROOM and builds == [SERVED - HEADROOM]
    check = _checked(plan, point, session, tmp_path / "out")
    assert check.status == LONG_CONTEXT_COMPLETE and check.reason == "" and check.served_context == SERVED
    assert (check.n, check.passed) == (PROMPTS, PROMPTS)
    assert {prompt.status for prompt in check.prompts} == {PROMPT_OK}, "no prompt is not_run"


def test_the_server_flag_is_the_served_context_where_the_launch_sets_one(tmp_path: Path, builds: list[int]) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "flag", "--context-length 8192", MAX_MODEL_LEN=MODEL_LIMIT)

    assert session.plan(point).length == SERVED - HEADROOM


@pytest.mark.parametrize(
    ("server_args", "envs"),
    [("", {"MAX_MODEL_LEN": MODEL_LIMIT}), ("", {}), ("--context-length 131072", {})],
    ids=["full-window", "states-none", "more-than-the-model"],
)
def test_a_launch_at_the_full_window_keeps_the_set_at_the_models_limit(
    tmp_path: Path, builds: list[int], server_args: str, envs: dict[str, int]
) -> None:
    session = _Session(tmp_path)
    today = default_long_context(session.model, session.environ, stage=True)

    plan = session.plan(_point(tmp_path, "full", server_args, **envs))

    assert today is not None and today.length == MODEL_LIMIT - HEADROOM
    assert (plan.prompt_set, plan.prompts_sha256, plan.length) == (today.prompt_set, today.prompts_sha256, today.length)
    assert builds == [MODEL_LIMIT - HEADROOM], "the set the model's limit gives, staged once"


def test_two_served_contexts_get_two_staged_sets_and_one_launch_shape_gets_one(
    tmp_path: Path, builds: list[int]
) -> None:
    session = _Session(tmp_path)
    short, longer = _point(tmp_path, "short", MAX_MODEL_LEN=SERVED), _point(tmp_path, "longer", MAX_MODEL_LEN=16384)

    root, candidate = session.plan(short), session.plan(short, Purpose.FINAL)
    other = session.plan(longer)

    assert root.prompt_set == candidate.prompt_set and root.prompts_sha256 == candidate.prompts_sha256
    assert other.prompt_set != root.prompt_set and other.prompts_sha256 != root.prompts_sha256
    assert (root.length, other.length) == (SERVED - HEADROOM, 16384 - HEADROOM)
    staged = tmp_path / "cache" / "evalsets"
    assert (staged / root.prompt_set).is_dir() and (staged / other.prompt_set).is_dir()
    assert builds == [SERVED - HEADROOM, 16384 - HEADROOM], "each context's set is staged once"


def test_a_1m_context_launch_checks_16_prompts_of_the_bound_less_the_answer_and_records_the_bound(
    tmp_path: Path, builds: list[int]
) -> None:
    session = _Session(tmp_path, limit=MILLION)
    point = _point(tmp_path, "served-1m", MAX_MODEL_LEN=MILLION)

    plan = session.plan(point)

    assert plan.length == BOUND - HEADROOM and builds == [BOUND - HEADROOM], "not 1M tokens a prompt"
    assert plan.bound == BOUND and plan.needed_context == BOUND
    prompts = load_prompt_set(plan, session.environ)
    assert len(prompts) == PROMPTS and max(prompt.tokens for prompt in prompts) <= BOUND - HEADROOM
    check = _checked(plan, point, session, tmp_path / "out")
    assert check.status == LONG_CONTEXT_COMPLETE and (check.n, check.passed) == (PROMPTS, PROMPTS)
    assert (check.served_context, check.length, check.bound) == (MILLION, BOUND - HEADROOM, BOUND)
    assert check.to_document()["bound"] == BOUND and LongContextResult.from_document(check.to_document()) == check


def test_a_32k_launch_keeps_its_length_and_only_the_count_changes(tmp_path: Path, builds: list[int]) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "served-32k", MAX_MODEL_LEN=32768)

    plan = session.plan(point)

    assert plan.length == 32768 - HEADROOM and builds == [32768 - HEADROOM], (
        "the bound is above it: the length is the same"
    )
    assert len(load_prompt_set(plan, session.environ)) == PROMPTS


def test_the_sets_key_holds_the_bound_so_another_bound_is_another_set_staged_once(
    tmp_path: Path, builds: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _Session(tmp_path)
    today = session.plan(_point(tmp_path, "served-8192", MAX_MODEL_LEN=SERVED))

    monkeypatch.setattr(build_module, "LENGTH_BOUND", 65536)
    other = ServedLongContext(session.model, session.environ, stage=True).at(SERVED)

    assert other is not None and other.length == today.length == SERVED - HEADROOM, "the bound is above both"
    assert other.prompt_set != today.prompt_set and other.bound == 65536 and today.bound == BOUND
    assert builds == [SERVED - HEADROOM, SERVED - HEADROOM], "the new bound's set is staged, once"


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.INTEGRATION, Purpose.FINAL])
def test_an_agentx_campaign_gates_every_baseline_integration_and_final_test_on_the_check(
    tmp_path: Path, builds: list[int], purpose: Purpose
) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "agentx", MAX_MODEL_LEN=SERVED)

    plan = session.planned(point, purpose, WorkloadKind.AGENTX, long_context_check=False)

    assert plan is not None and plan.trigger == (f"agentx gate: purpose {purpose.value}",), "not forced: the gate"
    assert plan.length == SERVED - HEADROOM and builds == [SERVED - HEADROOM]


def test_an_agentx_merge_that_touches_no_lever_is_planned_where_an_inferencex_one_is_not(
    tmp_path: Path, builds: list[int]
) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "agentx", MAX_MODEL_LEN=SERVED)
    unforced = {"long_context_check": False, "server_args": "--max-running-requests 128"}

    assert session.planned(point, Purpose.INTEGRATION, WorkloadKind.AGENTX, **unforced) is not None
    assert session.planned(point, Purpose.INTEGRATION, WorkloadKind.INFERENCEX, **unforced) is None


@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.CONFIRM])
def test_an_agentx_explore_or_confirm_runs_no_check_forced_or_on_a_lever_and_stages_nothing(
    tmp_path: Path, builds: list[int], purpose: Purpose
) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "agentx", MAX_MODEL_LEN=SERVED)
    lever = {"server_args": "--attention-backend aiter", **({"retest_of": "r-1"} if purpose is Purpose.CONFIRM else {})}

    assert session.planned(point, purpose, WorkloadKind.AGENTX, **lever) is None, "forced, on a lever line, too"
    assert builds == [] and not (tmp_path / "cache" / "evalsets").exists()
    assert session.planned(point, purpose, **lever) is not None, "the same test in an InferenceX campaign runs it"


def test_an_agentx_final_sweep_runs_no_check(tmp_path: Path, builds: list[int]) -> None:
    session = _Session(tmp_path)
    point = _point(tmp_path, "agentx", MAX_MODEL_LEN=SERVED)

    assert session.planned(point, Purpose.FINAL, WorkloadKind.AGENTX, kind=TestKind.SWEEP) is None
    assert builds == []
