# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One rank of a TP4 seam test, as its own process (``ranks.launch_ranks`` starts the four):

    parity WIDTH...  two back-to-back launches per width, each judged against the reference
    replay           graphs of widths 1, 4 and 16 replayed interleaved; launches issued back to
                     back; a run of launches across the epoch's wrap
    fault            back-to-back launches under the test's fault injection: they never finish
    absent RANK      every rank but RANK launches; RANK stays up holding its peer buffer

A rank prints ``RESULT {json}`` per verdict, ``LAUNCH <wall clock>`` before a fault scenario's first
launch, and ``FINISHED`` once its launches completed.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK), str(TASK / "benchmark" / "forge_mono")]

import reference  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from judge import judge  # noqa: E402
from ranks import (  # noqa: E402
    BACK_TO_BACK,
    BACK_TO_BACK_WIDTH,
    FAULT_WIDTH,
    LAUNCHES_UNDER_FAULT,
    MARGIN_S,
    REPLAY_ORDER,
    REPLAY_ROUNDS,
    REPLAY_WIDTHS,
    TP,
    WEIGHT_SEED,
    WRAP_FROM,
    WRAP_LAUNCHES,
)
from runner import DEFAULT_DEADLINE_S, SeamRunner  # noqa: E402
from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from weights import random_layer  # noqa: E402

from qwen3_seam import seam  # noqa: E402


def report(**fields) -> None:
    print("RESULT " + json.dumps(fields), flush=True)


class Rank:
    """This process's rank: its device, the process groups, the peer buffer, the layer and the
    warmed runner."""

    def __init__(self, backend: str, widths: tuple[int, ...]) -> None:
        self.rank = int(os.environ["RANK"])
        self.device = torch.device("cuda", self.rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend, rank=self.rank, world_size=TP)
        self.peer = PeerBuffer(seam.peer_bytes(TP), dist.new_group(backend="gloo"), self.rank, TP, self.device)
        self.weights = random_layer(TP, self.rank, self.device, WEIGHT_SEED)
        self.runner = SeamRunner(TP, self.rank, self.peer.local, self.peer.addresses, self.device, widths)
        self.runner.warm(self.weights, torch.cuda.get_device_properties(self.device).multi_processor_count)

    def inputs(self, s: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's attention output, and the residual every rank shares, both drawn from ``seed``."""
        own = torch.Generator(device=self.device).manual_seed(2 * TP * seed + self.rank)
        shared = torch.Generator(device=self.device).manual_seed(2 * TP * seed + TP)
        attn = torch.randn(s, seam.HIDDEN // TP, generator=own, dtype=torch.bfloat16, device=self.device)
        residual = torch.randn(s, seam.HIDDEN, generator=shared, dtype=torch.bfloat16, device=self.device)
        return attn, residual

    def launch(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.runner(*inputs, self.weights)

    def verdict(self, got, want, **where) -> None:
        passed, metrics = judge(got, want, self.runner.error_words())
        report(rank=self.rank, passed=passed, **where, **metrics)

    def finish(self) -> None:
        torch.cuda.synchronize()
        print("FINISHED", flush=True)
        dist.barrier()
        dist.destroy_process_group()


def parity(widths: tuple[int, ...]) -> None:
    me = Rank("nccl", widths)
    linear = reference.o_proj_linear()
    for s in widths:
        inputs = [me.inputs(s, seed) for seed in (1, 2)]
        got = [me.launch(x) for x in inputs]
        for call, (x, answer) in enumerate(zip(inputs, got, strict=True), start=1):
            me.verdict(answer, reference.seam(linear, *x, me.weights), check="parity", width=s, call=call)
    me.finish()


def replay() -> None:
    me = Rank("nccl", (*REPLAY_WIDTHS, BACK_TO_BACK_WIDTH))
    linear = reference.o_proj_linear()

    inputs = {s: me.inputs(s, 10 + s) for s in REPLAY_WIDTHS}
    want = {s: reference.seam(linear, *inputs[s], me.weights) for s in REPLAY_WIDTHS}
    graphs, outs = {}, {}
    for s in REPLAY_WIDTHS:
        graphs[s] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[s]):
            outs[s] = me.launch(inputs[s])
    torch.cuda.synchronize()
    for round_ in range(REPLAY_ROUNDS):
        for s in REPLAY_ORDER:
            graphs[s].replay()
        torch.cuda.synchronize()
        for s in REPLAY_WIDTHS:
            me.verdict(outs[s], want[s], check="graph", round=round_, width=s)

    pair = [me.inputs(BACK_TO_BACK_WIDTH, seed) for seed in (31, 32)]
    pair_want = [reference.seam(linear, *x, me.weights) for x in pair]
    torch.cuda.synchronize()
    start = int(me.runner.epoch[0]) & 0xFFFFFFFF
    got = [me.launch(pair[i % 2]) for i in range(BACK_TO_BACK)]
    torch.cuda.synchronize()
    for i, answer in enumerate(got):
        me.verdict(answer, pair_want[i % 2], check="back_to_back", launch=i)
    moved = ((int(me.runner.epoch[0]) & 0xFFFFFFFF) - start) % 2**32
    report(rank=me.rank, passed=moved == BACK_TO_BACK, check="epoch", moved=moved)

    dist.barrier()
    me.runner.set_epoch(WRAP_FROM)
    torch.cuda.synchronize()
    dist.barrier()
    got = [me.launch(pair[i % 2]) for i in range(WRAP_LAUNCHES)]
    torch.cuda.synchronize()
    for i, answer in enumerate(got):
        me.verdict(answer, pair_want[i % 2], check="wrap", launch=i, epoch=(WRAP_FROM + i) % 2**32)
    after = int(me.runner.epoch[0]) & 0xFFFFFFFF
    report(rank=me.rank, passed=after == (WRAP_FROM + WRAP_LAUNCHES) % 2**32, check="wrap_epoch", epoch=after)
    me.finish()


def stalled(missing: int | None) -> None:
    """Back-to-back launches that a failed wait must stop on every launching rank."""
    me = Rank("gloo", (FAULT_WIDTH,))
    inputs = me.inputs(FAULT_WIDTH, 3)
    torch.cuda.synchronize()
    dist.barrier()
    print(f"LAUNCH {time.time()}", flush=True)
    if me.rank == missing:
        time.sleep(DEFAULT_DEADLINE_S + MARGIN_S)
        return
    for _ in range(LAUNCHES_UNDER_FAULT):
        me.launch(inputs)
    me.finish()


def main(argv: list[str]) -> None:
    # a stopped launch takes its process down; a core file of it helps no test
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    scenario, args = argv[0], argv[1:]
    if scenario == "parity":
        parity(tuple(int(a) for a in args))
    elif scenario == "replay":
        replay()
    elif scenario == "fault":
        stalled(None)
    elif scenario == "absent":
        stalled(int(args[0]))
    else:
        raise SystemExit(f"unknown scenario {scenario!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
