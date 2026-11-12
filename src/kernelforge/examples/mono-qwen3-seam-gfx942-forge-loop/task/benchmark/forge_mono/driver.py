# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge driver: Qwen3-14B-FP8's attention seam (``qwen3_seam/seam.py``), one launch per layer and
rank from the attention output to the post-attention RMSNorm, measured at TP4 on gfx942.

Every case is one launch on every rank (``launches=1``) whose grid is resident (``resident=True``),
held to SGLang's path on the same tensors (``reference.py``) by ``judge.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this workspace's qwen3_seam

import torch
import torch.distributed as dist

import dist_harness as dh
from contract import check_contract
from judge import judge

TP = 4
CHECK_WIDTHS = (1, 4, 8, 13, 16)
BENCH_WIDTHS = (1, 4, 8, 16)
WEIGHT_SEED = 20261112

_rank: dict = {}


def _state(ctx: dh.RankContext) -> dict:
    """Built once per rank: the peer buffer (exchanged over a gloo group), the layer, the warmed
    runner, and the linear SGLang dispatches o_proj to. The launch's and the reference's modules
    import here: the process that re-executes under torchrun never uses them."""
    if not _rank:
        import reference
        from runner import SeamRunner
        from sgl_amd_mono.common.peer import PeerBuffer
        from weights import random_layer

        from qwen3_seam import seam

        peer = PeerBuffer(seam.peer_bytes(TP), dist.new_group(backend="gloo"), ctx.rank, TP, ctx.device)
        weights = random_layer(TP, ctx.rank, ctx.device, WEIGHT_SEED)
        runner = SeamRunner(
            TP, ctx.rank, peer.local, peer.addresses, ctx.device, sorted({*CHECK_WIDTHS, *BENCH_WIDTHS})
        )
        runner.warm(weights, torch.cuda.get_device_properties(ctx.device).multi_processor_count)
        _rank.update(
            seam=seam, reference=reference, peer=peer, weights=weights, runner=runner, linear=reference.o_proj_linear()
        )
    return _rank


def _inputs(s: int):
    def build(ctx: dh.RankContext):
        hidden = _state(ctx)["seam"].HIDDEN
        attn = torch.randn(s, hidden // TP, dtype=torch.bfloat16, device=ctx.device)
        residual = torch.randn(s, hidden, dtype=torch.bfloat16, device=ctx.device)
        dist.broadcast(residual, src=0)  # the residual stream is the same on every TP rank
        return attn, residual

    return build


def _candidate(ctx: dh.RankContext, inputs):
    state = _state(ctx)
    return state["runner"](*inputs, state["weights"])


def _reference(ctx: dh.RankContext, inputs):
    state = _state(ctx)
    return state["reference"].seam(state["linear"], *inputs, state["weights"])


def _judge(ctx: dh.RankContext, got, want) -> dh.Judgement:
    return dh.Judgement(*judge(got, want, _state(ctx)["runner"].error_words()))


def _cases(bench: bool) -> list[dh.Case]:
    return [
        dh.Case(f"s{s}", _inputs(s), _candidate, _reference, _judge, verify_replay=bench, launches=1, resident=True)
        for s in (BENCH_WIDTHS if bench else CHECK_WIDTHS)
    ]


if __name__ == "__main__":
    check_contract()
    sys.exit(dh.run(_cases("--bench-mode" in sys.argv), world_size=TP))
