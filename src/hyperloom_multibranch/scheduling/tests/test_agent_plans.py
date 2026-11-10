# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's directions split into one share per Arbor agent (:mod:`hyperloom_multibranch.scheduling.agent_plans`),
with a fake Arbor whose split deals the open directions round-robin, and whose generate makes one direction per op of
the profile at 2 % or more; and the map kept to deal a reporting worker's slot its next order
(:class:`~hyperloom_multibranch.scheduling.agent_plans.MapDealer`), by a fake Arbor master that deals by rank the open
directions no report settled, two reports did not give back open and no held order holds."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.scheduling.agent_plans import GENERATED_MAP, agent_plans, arbor_agents, map_dealer
from hyperloom_multibranch.specialist.agent_onboarding import PROFILE_FILE, ROOT_PROFILE_SOURCE
from hyperloom_multibranch.specialist.arbor_adapter import SplitFailed
from hyperloom_multibranch.tests.agent_rig import CHECKOUT_FILES, MAP, _committed, _write, worker_report
from hyperloom_multibranch.tests.test_specialist_arbor_keys import VENDORED_ARBOR

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


async def test_kernel_names_the_toolbox_took_as_op_shares_generate_the_map_too(tmp_path: Path) -> None:
    arbor, _ = _agents(tmp_path)
    kernels = OpShares("c0ffee", "prelude:p1", "", {"paged_attention_ll4mi_QKV": 32.5, "Cijk_Alik": 27.3})

    plans = await agent_plans((arbor, arbor), "", tmp_path / "work", kernels)

    assert [_ids(plan) for plan in plans] == [["m-paged_attention_ll4mi_QKV"], ["m-Cijk_Alik"]]
    profile = json.loads((tmp_path / "work" / PROFILE_FILE).read_text(encoding="utf-8"))
    assert (profile["shares"], profile["source"]) == (dict(kernels.shares), ROOT_PROFILE_SOURCE)


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


#: A map with more open directions than the first split hands out by rank, so that a deal has some left.
DEAL_MAP = {
    "directions": [
        {"id": "m-attn", "kind": "increment", "title": "attention decode path", "rank": 1},
        {"id": "m-moe", "kind": "increment", "title": "MoE GEMM tables", "rank": 2},
        {"id": "m-allreduce", "kind": "increment", "title": "custom all-reduce", "rank": 3},
        {"id": "m-sampler", "kind": "increment", "title": "sampler", "rank": 4},
        {"id": "m-rope", "kind": "increment", "title": "fused rope", "status": "done", "why": "iteration 2, KEEP 1"},
    ]
}


def _report(tmp_path: Path, name: str, *closed: dict[str, str], left_open: tuple[str, ...] = ()) -> Path:
    return worker_report(tmp_path / f"{name}.json", *closed, left_open=left_open)


def _share(tmp_path: Path, name: str, *ids: str) -> str:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"directions": [{"id": entry} for entry in ids]}), encoding="utf-8")
    return str(path)


async def test_a_reporting_worker_s_slot_is_dealt_open_directions_no_report_settled_and_no_running_worker_holds(
    tmp_path: Path,
) -> None:
    arbor, _ = _agents(tmp_path)
    first, second = _share(tmp_path, "share-1", "m-attn"), _share(tmp_path, "share-2", "m-moe")
    dealer = map_dealer(_map(tmp_path, DEAL_MAP), (first, "", second), tmp_path / "work")
    assert dealer is not None

    dealer.hand_back(first, _report(tmp_path, "r1", {"id": "m-attn", "status": "done"}))
    order = await dealer.deal(Path(arbor))

    # m-moe is held by the running worker, m-rope was closed in the map; at most two by rank.
    assert order is not None and _ids(str(order)) == ["m-allreduce", "m-sampler"]

    dealer.hand_back(
        str(order), _report(tmp_path, "r2", {"id": "m-sampler", "status": "done"}, left_open=("m-allreduce",))
    )
    again = await dealer.deal(Path(arbor))

    # Given back open once, m-allreduce is dealt again; the closed m-attn and m-sampler stay closed.
    assert again is not None and _ids(str(again)) == ["m-allreduce"]

    dealer.hand_back(str(again), _report(tmp_path, "r3", left_open=("m-allreduce",)))

    # Given back open by a second worker, m-allreduce is exhausted: every report is passed, so it is not reopened.
    assert await dealer.deal(Path(arbor)) is None


async def test_arbor_s_own_master_settles_what_a_worker_report_closed(tmp_path: Path) -> None:
    share = _share(tmp_path, "share-1", "m-attn", "m-moe")
    dealer = map_dealer(_map(tmp_path, DEAL_MAP), (share,), tmp_path / "work")
    assert dealer is not None

    closed = ({"id": "m-allreduce", "status": "killed", "why": "no gain"}, {"id": "m-sampler", "status": "done"})
    dealer.hand_back(share, _report(tmp_path, "r1", *closed, left_open=("m-attn",)))
    order = await dealer.deal(VENDORED_ARBOR)

    # The vendored Arbor reads the report as the fake master does: m-attn back open, the two it closed settled.
    assert order is not None and _ids(str(order)) == ["m-attn", "m-moe"]


async def test_two_workers_reporting_at_once_are_never_dealt_the_same_direction(tmp_path: Path) -> None:
    arbor, _ = _agents(tmp_path)
    first, second = _share(tmp_path, "share-1", "m-attn"), _share(tmp_path, "share-2", "m-moe")
    dealer = map_dealer(_map(tmp_path, DEAL_MAP), (first, second), tmp_path / "work")
    assert dealer is not None

    dealer.hand_back(first, _report(tmp_path, "r1", {"id": "m-attn", "status": "done"}))
    dealer.hand_back(second, _report(tmp_path, "r2", {"id": "m-moe", "status": "done"}))
    orders = await asyncio.gather(dealer.deal(Path(arbor)), dealer.deal(Path(arbor)))

    # The second deal holds the order the first dealt: what is left is all in one order.
    assert [None if order is None else _ids(str(order)) for order in orders] == [["m-allreduce", "m-sampler"], None]


async def test_an_order_no_branch_runs_is_dealt_again(tmp_path: Path) -> None:
    arbor, _ = _agents(tmp_path)
    first, second = _share(tmp_path, "share-1", "m-attn"), _share(tmp_path, "share-2", "m-moe")
    dealer = map_dealer(_map(tmp_path, DEAL_MAP), (first, second), tmp_path / "work")
    assert dealer is not None

    dealer.hand_back(first, _report(tmp_path, "r1", {"id": "m-attn", "status": "done"}))
    order = await dealer.deal(Path(arbor))
    assert order is not None
    dealer.release(str(order))
    dealer.release(second)

    again = await dealer.deal(Path(arbor))

    assert again is not None and _ids(str(again)) == ["m-moe", "m-allreduce"]


async def test_with_every_open_direction_settled_or_held_the_deal_leaves_the_slot_without_an_order(
    tmp_path: Path,
) -> None:
    arbor, _ = _agents(tmp_path)
    first, second = _share(tmp_path, "share-1", "m-attn", "m-allreduce", "m-sampler"), _share(tmp_path, "s2", "m-moe")
    dealer = map_dealer(_map(tmp_path, DEAL_MAP), (first, second), tmp_path / "work")
    assert dealer is not None

    closed = ({"id": "m-attn", "status": "done"}, {"id": "m-allreduce", "status": "dropped"})
    dealer.hand_back(first, _report(tmp_path, "r1", *closed, {"id": "m-sampler", "status": "done"}))

    assert await dealer.deal(Path(arbor)) is None


async def test_an_arbor_with_no_master_cannot_deal_and_a_split_that_dealt_no_share_keeps_no_map(
    tmp_path: Path,
) -> None:
    old = _committed(_write(tmp_path / "old-arbor", {"arbor/__init__.py": ""}))
    share = _share(tmp_path, "share-1", "m-attn")
    dealer = map_dealer(_map(tmp_path, DEAL_MAP), (share,), tmp_path / "work")
    assert dealer is not None

    dealer.hand_back(share, _report(tmp_path, "r1", {"id": "m-attn", "status": "done"}))
    with pytest.raises(SplitFailed, match="arbor.roles"):
        await dealer.deal(old)
    assert map_dealer(_map(tmp_path), ("", ""), tmp_path / "none") is None
