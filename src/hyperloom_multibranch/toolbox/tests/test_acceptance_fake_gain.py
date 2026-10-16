# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance refuses a request whose gain main refuses as fake (board topic 48), before anything launches."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, build, req

COPY_CODE = "speculative_context_copy_on_synthetic"
NGRAM = '--speculative-config \'{"method": "ngram", "num_speculative_tokens": 4}\''


def _repo(server_args: str = "", envs: dict[str, str] | None = None) -> FakeRepo:
    return FakeRepo(Recipe(base_image_digest=IMAGE, launch=LaunchConfig(server_args=server_args, envs=envs or {})))


async def _refused_with(tmp_path: Path, repo: FakeRepo, kind: WorkloadKind, server_args: str) -> str | None:
    rig = build(tmp_path, repo=repo, workload_kind=kind)
    try:
        await rig.toolbox.submit_test(req(server_args=server_args))
    except RequestRejected as refused:
        assert rig.planner.calls == [], "a refused request must not reach the compiler"
        assert rig.notebook._state().requests == {}
        return refused.code
    return None


def test_the_context_copy_code_is_a_documented_rejection_code() -> None:
    assert COPY_CODE in REJECTION_CODES


@pytest.mark.parametrize(
    "server_args",
    [
        NGRAM,
        "--speculative-config.method ngram_gpu",
        "--speculative-config.method suffix",
        '--speculative-config \'{"model": "[ngram]"}\'',
        "--speculative-algorithm NGRAM",
    ],
)
async def test_an_inferencex_candidate_that_adds_a_context_copy_draft_is_refused(
    tmp_path: Path, server_args: str
) -> None:
    assert await _refused_with(tmp_path, _repo(), WorkloadKind.INFERENCEX, server_args) == COPY_CODE


async def test_an_inferencex_candidate_that_tunes_the_copy_draft_of_its_base_is_refused(tmp_path: Path) -> None:
    base = _repo("--speculative-algorithm NGRAM --speculative-num-draft-tokens 4")
    code = await _refused_with(tmp_path, base, WorkloadKind.INFERENCEX, "--speculative-num-draft-tokens 8")
    assert code == COPY_CODE


@pytest.mark.parametrize(
    "server_args",
    [
        '--speculative-config \'{"method": "eagle3", "num_speculative_tokens": 3}\'',
        "--speculative-algorithm EAGLE",
        "--max-num-seqs 128",
    ],
)
async def test_an_inferencex_model_based_draft_or_no_draft_is_accepted(tmp_path: Path, server_args: str) -> None:
    assert await _refused_with(tmp_path, _repo(), WorkloadKind.INFERENCEX, server_args) is None


async def test_an_inferencex_candidate_that_leaves_the_base_copy_draft_alone_is_accepted(tmp_path: Path) -> None:
    base = _repo("--speculative-algorithm NGRAM")
    assert await _refused_with(tmp_path, base, WorkloadKind.INFERENCEX, "--max-num-seqs 128") is None


async def test_an_agentx_campaign_does_not_refuse_a_context_copy_draft(tmp_path: Path) -> None:
    assert await _refused_with(tmp_path, _repo(), WorkloadKind.AGENTX, NGRAM) is None


# --- AgentX: a draft whose acceptance the parent simulates stays pinned ---

PINNED_CODE = "speculative_decoding_pinned"
SIMULATING = "--speculative-algorithm EAGLE --speculative-num-steps 3"
SIM_ENV = {"SGLANG_SIMULATE_ACC_LEN": "3.6"}


def test_the_pinned_code_is_a_documented_rejection_code() -> None:
    assert PINNED_CODE in REJECTION_CODES


@pytest.mark.parametrize("server_args", ["--speculative-num-steps 5", "--speculative-algorithm NEXTN"])
async def test_an_agentx_candidate_that_changes_the_pinned_draft_is_refused(tmp_path: Path, server_args: str) -> None:
    code = await _refused_with(tmp_path, _repo(SIMULATING, SIM_ENV), WorkloadKind.AGENTX, server_args)
    assert code == PINNED_CODE


async def test_an_agentx_candidate_that_leaves_the_pinned_draft_alone_is_accepted(tmp_path: Path) -> None:
    code = await _refused_with(tmp_path, _repo(SIMULATING, SIM_ENV), WorkloadKind.AGENTX, "--max-running-requests 8")
    assert code is None


async def test_an_agentx_candidate_that_starts_simulating_over_a_real_draft_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=_repo(SIMULATING), workload_kind=WorkloadKind.AGENTX)
    change = RecipeDelta(config=ConfigDelta(envs=SIM_ENV))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=change))
    assert refused.value.code == PINNED_CODE


async def test_an_agentx_candidate_may_enable_a_draft_the_parent_does_not_run(tmp_path: Path) -> None:
    assert await _refused_with(tmp_path, _repo(), WorkloadKind.AGENTX, SIMULATING) is None


async def test_an_inferencex_campaign_does_not_pin_a_simulated_draft(tmp_path: Path) -> None:
    code = await _refused_with(
        tmp_path, _repo(SIMULATING, SIM_ENV), WorkloadKind.INFERENCEX, "--speculative-num-steps 5"
    )
    assert code is None
