# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance refuses a candidate that runs TunableOp with its tuning on, before a card is held for it.

With tuning on, every GEMM shape is tuned the first time the server runs it: at boot, in the warmup and for each new
prompt length of the evaluation, so the length of the test follows the shapes it happens to meet. The results file is
written when the server exits; the measurement ends the server first, so a retry starts from nothing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose
from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, Recipe, RecipeDelta
from hyperloom.measure.tests.tunableop_table import TUNABLEOP_ENV, TUNABLEOP_TABLE
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox.acceptance import ONLINE_TUNING_CODE, Acceptance
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakePlanner, FakeRepo, build, req
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

AITER_TABLE_ENV = "AITER_CONFIG_GEMM_BF16"
#: The envs of the test that was measured, as its config recorded them.
MEASURED = {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1", "PYTORCH_TUNABLEOP_VERBOSE": "2"}
#: The head of the image's own bf16 GEMM table, the table a specialist tunes and puts.
TABLE = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw\n"
    "gfx950,256,64,256,5120,False,torch.bfloat16,torch.float32,False,False,asm,5,13,7.9191,"
    "_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE,0.0,21.19,417.92\n"
    "gfx950,256,80,256,5120,False,torch.bfloat16,torch.float32,False,False,asm,7,13,7.9676,"
    "_ZN5aiter39bf16gemm_fp32bf16_tn_80x64_splitk_cleanE,0.0,26.32,436.97\n"
).encode()


def _delta(**envs: str) -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args="--max-running-requests 64", envs=envs))


def _repo(**envs: str) -> FakeRepo:
    return FakeRepo(Recipe(base_image_digest=IMAGE, launch=dataclasses.replace(Recipe(IMAGE).launch, envs=envs)))


async def _code_of(tmp_path: Path, repo: FakeRepo, change: RecipeDelta, **fields: object) -> str | None:
    """The code a submitted request is refused with, ``None`` when it is accepted."""
    rig = build(tmp_path, repo=repo)
    try:
        await rig.toolbox.submit_test(req(change=change, **fields))
    except RequestRejected as refused:
        assert rig.planner.calls == [], "a refused request must not reach the compiler"
        assert rig.notebook._state().requests == {}, "a refused request must write nothing to the notebook"
        return refused.code
    return None


def test_the_online_tuning_code_is_a_documented_rejection_code() -> None:
    assert ONLINE_TUNING_CODE in REJECTION_CODES


async def test_the_test_that_was_measured_is_refused_and_the_reason_names_the_way_out(tmp_path: Path) -> None:
    rig = build(tmp_path)
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=_delta(**MEASURED)))
    assert refused.value.code == ONLINE_TUNING_CODE
    for named in ("PYTORCH_TUNABLEOP_TUNING=0", "put_kernel_artifact", AITER_TABLE_ENV):
        assert named in refused.value.reason, named
    for tunableop_route in ("a TunableOp table", f"env {TUNABLEOP_ENV}", "PYTORCH_TUNABLEOP_ENABLED=1 and"):
        assert tunableop_route in refused.value.reason, tunableop_route


@pytest.mark.parametrize(
    "envs",
    [
        {"PYTORCH_TUNABLEOP_ENABLED": "1"},
        {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1"},
        {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": ""},
        {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "false"},
    ],
    ids=["tuning_unset", "tuning_on", "tuning_empty", "tuning_false"],
)
@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.INTEGRATION])
async def test_a_candidate_that_tunes_online_is_refused(tmp_path: Path, envs: dict[str, str], purpose: Purpose) -> None:
    assert await _code_of(tmp_path, FakeRepo(), _delta(**envs), purpose=purpose) == ONLINE_TUNING_CODE


@pytest.mark.parametrize(
    "envs",
    [
        {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0"},
        {"PYTORCH_TUNABLEOP_TUNING": "1"},
        {"PYTORCH_TUNABLEOP_ENABLED": "0", "PYTORCH_TUNABLEOP_TUNING": "1"},
        {"PYTORCH_TUNABLEOP_ENABLED": "true"},
        {"PYTORCH_TUNABLEOP_VERBOSE": "2"},
    ],
    ids=["tuning_off", "not_enabled", "disabled", "true_enables_nothing", "verbose_only"],
)
async def test_a_candidate_that_does_not_tune_online_is_accepted(tmp_path: Path, envs: dict[str, str]) -> None:
    assert await _code_of(tmp_path, FakeRepo(), _delta(**envs)) is None


async def test_a_candidate_that_puts_an_offline_tuned_table_is_accepted(tmp_path: Path) -> None:
    table = mint_kernel_artifact(KernelKind.GEMM, "qwen3-8b-bf16", AITER_TABLE_ENV, TABLE)
    change = dataclasses.replace(_delta(), add_kernels=(KernelArtifact.from_document(table.to_document()),))
    assert await _code_of(tmp_path, FakeRepo(), change) is None


async def test_a_candidate_that_puts_a_tunableop_table_and_turns_tuning_off_is_accepted(tmp_path: Path) -> None:
    table = mint_kernel_artifact(KernelKind.GEMM, "qwen3-8b-bf16", TUNABLEOP_ENV, TUNABLEOP_TABLE)
    offline = _delta(PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="0")
    change = dataclasses.replace(offline, add_kernels=(KernelArtifact.from_document(table.to_document()),))
    assert await _code_of(tmp_path, FakeRepo(), change) is None


async def test_the_launch_the_delta_produces_is_judged_not_the_delta_alone(tmp_path: Path) -> None:
    parent = _repo(PYTORCH_TUNABLEOP_ENABLED="1")
    assert await _code_of(tmp_path / "a", parent, _delta(PYTORCH_TUNABLEOP_TUNING="0")) is None
    assert await _code_of(tmp_path / "b", parent, _delta(PYTORCH_TUNABLEOP_VERBOSE="1")) == ONLINE_TUNING_CODE
    unset = RecipeDelta(
        config=ConfigDelta(server_args="--max-running-requests 64", unset_envs=("PYTORCH_TUNABLEOP_ENABLED",))
    )
    assert await _code_of(tmp_path / "c", parent, unset) is None


@pytest.mark.parametrize("kind", [WorkloadKind.INFERENCEX, WorkloadKind.AGENTX])
async def test_the_refusal_is_the_same_for_every_workload(tmp_path: Path, kind: WorkloadKind) -> None:
    rig = build(tmp_path, workload_kind=kind)
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=_delta(**MEASURED)))
    assert refused.value.code == ONLINE_TUNING_CODE


async def test_a_request_an_earlier_toolbox_accepted_is_not_withdrawn_by_the_rule() -> None:
    acceptance = Acceptance(AcceptanceSetup(FakeRepo(), FakePlanner(), IMAGE), is_accepted=lambda _: True)
    request = req(change=_delta(**MEASURED))
    await acceptance.prepare(dataclasses.replace(request, request_id="r1"), replay=True)
    with pytest.raises(RequestRejected) as refused:
        await acceptance.prepare(request)
    assert refused.value.code == ONLINE_TUNING_CODE
