# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two tests of one session run side by side in local sandboxes on disjoint cards of one machine, inside a container with
a pid namespace of its own: test A leases cards 0-3, test B leases cards 4-7 and its round starts a TP4 server.

The ROCm runtime numbers the GPUs it enumerates and opens every one of them: ``ROCR_VISIBLE_DEVICES`` limits what it
enumerates, while ``HIP_VISIBLE_DEVICES`` (and ``CUDA_VISIBLE_DEVICES``) only pick among the devices it already opened. A
process that opened a card keeps its runtime context resident there, and a server rank that maps the other ranks'
buffers keeps far more. The kernel's list then charges B's server ranks to card 0, by host ids the container cannot
name, and A's card guard counts each of them as a foreign holder before A starts. So B's server must reach the
runtime with a ROCr mask of its own cards, whatever the sandbox's mask was: the round drops the inherited one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from hyperloom.orchestrator.actions.executors._suite.rounds import inherited_env_names
from hyperloom.sandbox.card_occupancy import OwnProcesses, read_holders
from hyperloom.sandbox.tests.kernel_tree import KernelTree
from inference_suite.compose import compose
from inference_suite.contracts.launch import Launch
from inference_suite.tests.test_base_launch_contract import _base
from inference_suite.tests.test_compose import _FACTS
from inference_suite.tests.test_config_contract import _config

_MACHINE_CARDS = 8
_A_CARDS = (0, 1, 2, 3)
_B_CARDS = (4, 5, 6, 7)
_B_RANKS = (3001, 3002, 3003, 3004)  # host ids: the container's /proc has no such process
_RANK_VRAM = 106 << 30
_OPENED_CONTEXT_VRAM = 3 << 30  # what a rank keeps on a card it only opened, as the kernel charged it (GiB, not KiB)
_NOBODY = OwnProcesses(frozenset())


def _rocm_opened_cards(env: Mapping[str, str]) -> tuple[int, ...]:
    """The machine cards a ROCm process started with ``env`` opens: ROCr's mask selects them, HIP's does not."""
    mask = env.get("ROCR_VISIBLE_DEVICES")
    return tuple(range(_MACHINE_CARDS)) if mask is None else tuple(int(card) for card in mask.split(","))


def _b_server_env() -> dict[str, str]:
    """The environment B's server starts with: what its sandbox exports, less what the round drops, under the launch."""
    sandbox = {"PATH": "/usr/bin", "HOME": "/root", "ROCR_VISIBLE_DEVICES": ",".join(map(str, _B_CARDS))}
    base = _base()
    server = _config().server
    server = replace(server, base=replace(server.base, digest=base.digest()), gpus=_B_CARDS)
    launch = compose(_config(server=server), base, _FACTS)
    assert isinstance(launch, Launch), launch
    inherited = {name: sandbox[name] for name in inherited_env_names(sandbox)}
    return {**inherited, **{name: str(value) for name, value in launch.env.items()}}


def _serve_b(tree: KernelTree, env: Mapping[str, str]) -> None:
    """B's four ranks as the kernel lists them: each holds its own card, and keeps a context on every other it opened."""
    opened = _rocm_opened_cards(env)
    visible = [card for card in _B_CARDS if card in opened]
    for rank, pid in enumerate(_B_RANKS):
        own = visible[rank]
        tree.holds(pid, own, vram=_RANK_VRAM, queues=6)
        for card in opened:
            if card != own:
                tree.holds(pid, card, vram=_OPENED_CONTEXT_VRAM, queues=0)


def test_bs_server_reaches_the_runtime_with_a_rocr_mask_of_its_own_cards_only(tmp_path: Path) -> None:
    env = _b_server_env()

    assert _rocm_opened_cards(env) == _B_CARDS
    assert env["HIP_VISIBLE_DEVICES"] == "0,1,2,3"


def test_a_sees_no_foreign_holder_on_its_cards_while_bs_server_runs_beside_it(tmp_path: Path) -> None:
    tree = KernelTree(tmp_path, gpus=_MACHINE_CARDS, pid_namespace="pid:[4026532001]")
    _serve_b(tree, _b_server_env())

    reading = read_holders(list(_A_CARDS), kfd=tree.kfd, procfs=tree.procfs)

    assert reading.unavailable is None
    assert reading.intrusions(_NOBODY, before_test=True) == ()


def test_a_hip_mask_alone_leaks_bs_server_onto_card_0_and_as_host_pids_as_reads_it(tmp_path: Path) -> None:
    """The control: the model and the probe do see the leak this file guards against, so its passes are not vacuous."""
    tree = KernelTree(tmp_path, gpus=_MACHINE_CARDS, pid_namespace="pid:[4026532001]")
    _serve_b(tree, {"HIP_VISIBLE_DEVICES": ",".join(map(str, _B_CARDS))})

    found = read_holders(list(_A_CARDS), kfd=tree.kfd, procfs=tree.procfs).intrusions(_NOBODY, before_test=True)

    assert {intrusion.key for intrusion in found if intrusion.key[2] == 0} == {("host-pid", pid, 0) for pid in _B_RANKS}
    assert "host pid 3001 on card 0 (3.0 GiB resident" in found[0].what
