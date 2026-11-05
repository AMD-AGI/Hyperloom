# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's directions split into one share per Arbor agent (:mod:`hyperloom_multibranch.scheduling.agent_plans`),
with a fake Arbor whose split deals the open directions round-robin, and whose generate makes one direction per op of
the profile at 2 % or more."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.scheduling.agent_plans import GENERATED_MAP, agent_plans, arbor_agents
from hyperloom_multibranch.scheduling.root_shares import KERNEL_SHARES_SOURCE
from hyperloom_multibranch.specialist.agent_onboarding import PROFILE_FILE, ROOT_PROFILE_SOURCE
from hyperloom_multibranch.tests.test_specialist_arbor_adapter import CHECKOUT_FILES, MAP, _committed, _write

#: The root's op shares as PRELUDE's profile reports them, in percent.
ROOT_SHARES = OpShares("c0ffee", "prelude-1", "p1", {"attn_fwd": 41.0, "moe_gemm": 33.0, "rmsnorm": 1.5, "nccl": 9.0})


def _map(tmp_path: Path, document: object = MAP) -> str:
    path = tmp_path / "map.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _ids(plan: str) -> list[str]:
    return [entry["id"] for entry in json.loads(Path(plan).read_text(encoding="utf-8"))["directions"]]


def _agents(tmp_path: Path) -> tuple[str, str]:
    arbor = _committed(_write(tmp_path / "arbor", CHECKOUT_FILES))
    other = _committed(_write(tmp_path / "other", {"README.md": "another agent\n"}))
    return str(arbor), str(other)


async def test_each_arbor_agent_gets_its_own_share_in_order_and_another_agent_none(tmp_path: Path) -> None:
    arbor, other = _agents(tmp_path)

    plans = await agent_plans((arbor, other, arbor), _map(tmp_path), tmp_path / "work")

    assert plans[1] == ""
    assert [_ids(plans[0]), _ids(plans[2])] == [["m-attn", "m-allreduce"], ["m-moe"]]
    assert Path(plans[0]).is_absolute() and Path(plans[2]).is_absolute()
    assert len(list((tmp_path / "work" / "sources").iterdir())) == 2


async def test_no_branch_gets_a_share_without_a_map_an_arbor_enough_directions_or_a_source(tmp_path: Path) -> None:
    arbor, other = _agents(tmp_path)
    one = {"directions": [{"id": "m-attn", "kind": "increment", "title": "attention decode path"}]}

    assert await agent_plans((arbor, arbor), "", tmp_path / "none") == ("", "")
    assert await agent_plans((other,), _map(tmp_path), tmp_path / "other-only") == ("",)
    assert await agent_plans((str(tmp_path / "nowhere"), arbor), _map(tmp_path), tmp_path / "unfetched") == ("", "")
    assert await agent_plans((arbor, arbor), _map(tmp_path, one), tmp_path / "too-few") == ("", "")


async def test_without_a_map_two_arbor_agents_split_one_arbor_generates_from_the_root_s_op_shares(
    tmp_path: Path,
) -> None:
    arbor, other = _agents(tmp_path)
    work = tmp_path / "work"

    plans = await agent_plans((arbor, other, arbor), "", work, ROOT_SHARES)

    assert plans[1] == ""
    assert [_ids(plans[0]), _ids(plans[2])] == [["m-attn_fwd", "m-nccl"], ["m-moe_gemm"]]
    profile = json.loads((work / PROFILE_FILE).read_text(encoding="utf-8"))
    assert (profile["commit"], profile["unit"], profile["shares"]) == ("c0ffee", "percent", dict(ROOT_SHARES.shares))
    assert profile["source"] == ROOT_PROFILE_SOURCE
    assert (work / GENERATED_MAP).is_file()


async def test_kernel_shares_generate_the_map_too_and_the_profile_says_they_are_kernels(tmp_path: Path) -> None:
    arbor, _ = _agents(tmp_path)
    kernels = OpShares("c0ffee", "prelude:p1", "", {"paged_attention_ll4mi_QKV": 32.5, "Cijk_Alik": 27.3})

    plans = await agent_plans((arbor, arbor), "", tmp_path / "work", kernels, KERNEL_SHARES_SOURCE)

    assert [_ids(plan) for plan in plans] == [["m-paged_attention_ll4mi_QKV"], ["m-Cijk_Alik"]]
    profile = json.loads((tmp_path / "work" / PROFILE_FILE).read_text(encoding="utf-8"))
    assert profile["source"] == KERNEL_SHARES_SOURCE


async def test_the_arbor_agents_are_found_by_their_checkouts_each_source_fetched_once(tmp_path: Path) -> None:
    arbor, other = _agents(tmp_path)

    found = await arbor_agents((other, arbor, arbor), tmp_path / "work")

    assert list(found) == [1, 2] and found[1] == found[2]
    assert len(list((tmp_path / "work" / "sources").iterdir())) == 2


async def test_no_map_is_generated_for_one_arbor_or_a_root_whose_shares_arbor_cannot_use(tmp_path: Path) -> None:
    arbor, other = _agents(tmp_path)
    small = OpShares("c0ffee", "prelude-1", "p1", {"rmsnorm": 1.5, "rope": 0.5})

    assert await agent_plans((arbor, other), "", tmp_path / "one-arbor", ROOT_SHARES) == ("", "")
    assert not (tmp_path / "one-arbor" / GENERATED_MAP).exists()
    assert await agent_plans((arbor,), "", tmp_path / "alone", ROOT_SHARES) == ("",)
    assert not (tmp_path / "alone").exists()
    assert await agent_plans((arbor, arbor), "", tmp_path / "small", small) == ("", "")
    assert not (tmp_path / "small" / GENERATED_MAP).exists()


async def test_a_given_map_is_split_and_none_is_generated_even_with_the_root_s_op_shares(tmp_path: Path) -> None:
    arbor, _ = _agents(tmp_path)

    plans = await agent_plans((arbor, arbor), _map(tmp_path), tmp_path / "work", ROOT_SHARES)

    assert [_ids(plans[0]), _ids(plans[1])] == [["m-attn", "m-allreduce"], ["m-moe"]]
    assert not (tmp_path / "work" / GENERATED_MAP).exists()
