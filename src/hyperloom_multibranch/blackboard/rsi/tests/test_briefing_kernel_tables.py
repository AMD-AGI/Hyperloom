# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Another branch's kernel table replayed on a branch head: stacked in the head's slot as the merge stacks two tables.

These tests read tables as the toolbox mints them and stack them with the repository's merge."""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import NoReturn

import pytest

from hyperloom.contracts.recipe import KernelArtifact, KernelKind, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import PeerReplay
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.knowledge.replay_tables import StackSlot
from hyperloom_multibranch.blackboard.rsi.briefing import CrossSessionBriefing
from hyperloom_multibranch.blackboard.rsi.tests.repos import FakeRepo
from hyperloom_multibranch.blackboard.tests.factories import FakeClock, request
from hyperloom_multibranch.blackboard.tests.sessions import SPLIT_KV, scripted_session
from hyperloom_multibranch.repo.merge import Stored, stack_kernel_slot
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact, union_rows
from hyperloom_multibranch.toolbox.tests.kernel_rows import (
    ALPHA_NEW,
    BETA_HEAD,
    BETA_NEW,
    BETA_SHARED,
    ENV,
    GAMMA_NEW,
    SHARED,
    csv_text,
    install,
)


def _unused(*_: object) -> NoReturn:
    raise AssertionError("stacking a kernel slot reads and keeps tables only")


def _slot(store: dict[str, bytes], configs: Path | None = None) -> StackSlot:
    """The merge's stack of two tables in one slot, reading from ``store`` and keeping a united table in it."""

    def keep(data: bytes) -> str:
        store[hashlib.sha256(data).hexdigest()] = data
        return hashlib.sha256(data).hexdigest()

    return partial(
        stack_kernel_slot, stored=Stored(_unused, lambda artifact: store[artifact.sha256], _unused, keep, configs)
    )


@pytest.mark.parametrize("target", ["a8w8", "qkv"])
async def test_a_peer_replay_takes_the_wider_table_and_names_no_loss(tmp_path: Path, target: str) -> None:
    """Also for another layer's table on the head's load variable: the service reads one table per variable, so the
    wider table takes the head's place and the replayed recipe sets the variable once."""

    root = tmp_path / "blackboard"
    blackboard = await scripted_session(root, FakeClock())
    toolbox = LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox"), clock=FakeClock())
    narrow = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED))
    wide = mint_kernel_artifact(KernelKind.GEMM, target, ENV, csv_text(SHARED, ALPHA_NEW))
    store = {narrow.sha256: csv_text(SHARED), wide.sha256: csv_text(SHARED, ALPHA_NEW)}
    await toolbox.write_request(request("r7", SPLIT_KV, RecipeDelta(add_kernels=(wide,))))
    head = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"), kernels=(narrow,))
    briefing = CrossSessionBriefing(
        blackboard, branch="s3", repo=FakeRepo({"s3": "c3"}, {"c3": head}), stack_slot=_slot(store)
    )

    replayed = await briefing.peer_replay("r7")

    assert replayed.replaces == ()
    assert replayed.delta.add_kernels == (wide,)
    assert apply(head, replayed.delta).kernels == (wide,)


async def test_a_peer_replay_drops_a_kernel_the_heads_wider_table_already_covers(tmp_path: Path) -> None:
    root = tmp_path / "blackboard"
    blackboard = await scripted_session(root, FakeClock())
    toolbox = LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox"), clock=FakeClock())
    narrow = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED))
    wide = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED, ALPHA_NEW))
    store = {narrow.sha256: csv_text(SHARED), wide.sha256: csv_text(SHARED, ALPHA_NEW)}
    await toolbox.write_request(request("r7", SPLIT_KV, RecipeDelta(add_kernels=(narrow,))))
    head = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"), kernels=(wide,))
    briefing = CrossSessionBriefing(
        blackboard, branch="s3", repo=FakeRepo({"s3": "c3"}, {"c3": head}), stack_slot=_slot(store)
    )

    with pytest.raises(ValueError, match="already has the change"):
        await briefing.peer_replay("r7")


async def _replayed_over(
    tmp_path: Path, head_data: bytes, peer_data: bytes, peer_target: str = "a8w8"
) -> tuple[PeerReplay, dict[str, bytes]]:
    """Another branch's table, for ``peer_target``, replayed on a head that holds ``head_data`` for ``a8w8`` on the same
    load variable, as the merge stacks it."""

    root = tmp_path / "blackboard"
    blackboard = await scripted_session(root, FakeClock())
    toolbox = LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox"), clock=FakeClock())
    head_art = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, head_data)
    peer = mint_kernel_artifact(KernelKind.GEMM, peer_target, ENV, peer_data)
    store = {head_art.sha256: head_data, peer.sha256: peer_data}
    await toolbox.write_request(request("r7", SPLIT_KV, RecipeDelta(add_kernels=(peer,))))
    head = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"), kernels=(head_art,))
    stack_slot = _slot(store, install(tmp_path / "image"))
    briefing = CrossSessionBriefing(
        blackboard, branch="s3", repo=FakeRepo({"s3": "c3"}, {"c3": head}), stack_slot=stack_slot
    )
    return await briefing.peer_replay("r7"), store


@pytest.mark.parametrize("target", ["a8w8", "qkv"])
async def test_a_peer_replay_unites_two_tables_each_with_rows_the_other_lacks(tmp_path: Path, target: str) -> None:
    """In the head's slot, also when the peer's table is for another layer on the head's load variable."""

    head_data, peer_data = csv_text(SHARED, BETA_NEW), csv_text(SHARED[:3], GAMMA_NEW)

    replayed, store = await _replayed_over(tmp_path, head_data, peer_data, target)

    (united,) = replayed.delta.add_kernels
    expected = union_rows(peer_data, head_data, ENV, configs=install(tmp_path / "expected"))
    assert replayed.replaces == () and replayed.delta.remove_kernels == ()
    assert united.slot == mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, head_data).slot
    assert (united.sha256, store[united.sha256]) == (hashlib.sha256(expected).hexdigest(), expected)


async def test_a_peer_replay_keeps_the_heads_row_for_a_shape_both_tuned(tmp_path: Path) -> None:
    head_data, peer_data = csv_text(SHARED, BETA_HEAD), csv_text(BETA_SHARED, BETA_NEW)

    replayed, store = await _replayed_over(tmp_path, head_data, peer_data)

    (united,) = replayed.delta.add_kernels
    expected = union_rows(csv_text(BETA_NEW), head_data, ENV, configs=install(tmp_path / "expected"))
    assert replayed.replaces == ()
    assert store[united.sha256] == expected


async def test_a_peer_replay_names_the_swap_when_the_merge_would_refuse_the_tables(tmp_path: Path) -> None:
    root = tmp_path / "blackboard"
    blackboard = await scripted_session(root, FakeClock())
    toolbox = LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox"), clock=FakeClock())
    own = KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/head.csv", "b" * 64)
    theirs = KernelArtifact(KernelKind.GEMM, "moe", "kernel/gemm/peer.csv", "d" * 64)
    store = {
        own.sha256: b"cu_num,M,N,K,gfx,us\n304,1,4096,4096,gfx950,23.1\n",
        theirs.sha256: b"M,N,K,us\n1,4096,4096,19.8\n8,4096,4096,31.0\n",
    }
    await toolbox.write_request(request("r7", SPLIT_KV, RecipeDelta(add_kernels=(theirs,))))
    head = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"), kernels=(own,))
    briefing = CrossSessionBriefing(
        blackboard, branch="s3", repo=FakeRepo({"s3": "c3"}, {"c3": head}), stack_slot=_slot(store)
    )

    replayed = await briefing.peer_replay("r7")

    assert replayed.replaces == (own,)
    assert replayed.delta.add_kernels == (theirs,)
