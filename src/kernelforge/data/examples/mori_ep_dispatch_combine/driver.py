"""Measurement driver for the MoRI-EP dispatch/combine forge-loop task.

forge-loop treats this as a black box invoked as ``python driver.py <args>``
and communicates with it purely through stdout (see examples/README.md §2).
It implements the three modes of that contract for MoRI-EP's distributed
(EP8, all-8-GPU) dispatch/combine all-to-all:

  * Correctness  ``python driver.py --mode <smoke|stability|determinism|full|bench>``
    -> spawns 8 ranks, each does a dispatch -> identity-expert -> combine
    round trip (the exact "isolated round-trip" recipe from
    local_knowledge/framework/mori/operators/ep_dispatch_combine/: "dispatch ->
    (identity expert) -> combine round-trip must reconstruct the input
    within the dispatch dtype's tolerance"). ``smoke``/``stability``/
    ``determinism``/``full`` run cheap bf16-dispatch/bf16-combine round
    trips at <=256 tokens/rank; ``bench`` runs the SAME fp8-dispatch/
    bf16-combine, ``_BENCH_TOKENS_PER_RANK``-token shape the benchmark below
    times -- without it, a config could pass every other mode while being
    wrong (or simply untested) at the shape that's actually scored. The
    identity expert casts dispatch's output to the combine dtype (a no-op
    when they already match) before feeding combine, mirroring what a real
    expert would do between a quantized dispatch and a higher-precision
    combine. Per-token routing weights are uniform (1/topk) so a correct
    round trip reconstructs the original input exactly (modulo rounding),
    but the actual pass/fail verdict comes from mori's own exact-equality
    test-suite assertions (``check_dispatch_result`` / ``check_combine_
    result`` in ``_correctness_worker``), not a computed SNR: an invalid
    config raises inside the spawned worker (propagated by ``mp.spawn``)
    before rank 0 ever writes its "ok" marker file. Prints only
    ``allclose: <bool>`` -- the README's documented fallback for a driver
    that gates on pass/fail rather than a dB score. There is no
    ``SNR: <db> dB`` line; nothing here computes one.

  * Benchmark    ``python driver.py --bench-mode --warmup <n> --iters <n>``
    -> spawns 8 ranks running the REAL reference workload documented in
    local_knowledge/framework/mori/operators/ep_dispatch_combine/tuning.md: EP8,
    4096 tokens/rank, hidden_dim=7168, top-8 routing, fp8 (e4m3fnuz)
    dispatch + bf16 combine. Routing is fixed for the whole run (generated
    once, reused every round) so the one host sync needed to learn dispatch's
    data-dependent recv count happens ONCE before any timed round, never
    inside one -- see ``_bench_worker``. Each round: every rank times its own
    dispatch+combine with CUDA events, then ALL ranks join a MAX all_reduce
    for that round (a synchronized collective's true per-round duration is
    its slowest rank); the driver takes the median across rounds of that
    per-round max and prints one ``wall_ms`` line per round plus one
    ``case_ms`` line (the median) forge scores on.

  * Profiling    ``python driver.py --profile-run [--profile-case <id>]``
    -> runs a few dispatch+combine calls on all 8 ranks with no reference/
    correctness/timing output, then exits 0.

The tunable launch config (dispatch/combine block_num & warp_per_block,
kernel_type, combine_zero_copy) lives in ``mori_ep_config.py`` — forge edits
THAT file; this driver is protected and never edited by the agent. The fixed
workload sizes above are NOT tunable — they anchor what block_num/
warp_per_block choices are actually being optimized for.

Requires ``HSA_NO_SCRATCH_RECLAIM=1`` (mori's own hard runtime requirement on
this ROCm build) — set below before anything imports torch/HIP.

The correctness gate also requires a **git checkout of ``mori`` itself**
(not just ``pip install mori``) reachable on disk, because it reuses mori's
own test-suite reference math (``tests/python/ops/
dispatch_combine_test_utils.py``), which the installed package does not
ship. Point ``MORI_REPO_ROOT`` at that checkout if it is not at ``/work/mori``.
"""

from __future__ import annotations

import os

os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
os.environ.setdefault("MORI_GPU_ARCHS", "gfx942")

import argparse
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mori_ep_config import get_ep_launch_config

_WORLD_SIZE = 8
_MASTER_PORT = "29581"

# Fixed reference workload (see module docstring) — matches the numbers cited
# in local_knowledge/framework/mori/operators/ep_dispatch_combine/tuning.md.
# Overridable via env vars so the SAME driver contract can be pointed at a
# different shape regime (decode-ish / large-prefill / narrower hidden) for a
# generalization sweep, without hand-editing per-shape copies of this file.
_HIDDEN_DIM = int(os.environ.get("MORI_HIDDEN_DIM", "7168"))
_NUM_EXPERTS_PER_RANK = 32  # E=256 total over world_size=8
_NUM_EXPERTS_PER_TOKEN = int(os.environ.get("MORI_TOPK", "8"))  # top-8
_BENCH_TOKENS_PER_RANK = int(os.environ.get("MORI_TOKENS_PER_RANK", "4096"))

# Smaller token counts for correctness modes — same topk/hidden_dim (so the
# same launch-config knobs are exercised) but cheap enough to run many times
# per forge-loop iteration (smoke/shape-sweep/stability/determinism/full).
# These four run bf16 dispatch + bf16 combine (dispatch dtype == combine
# dtype, so the "identity expert" cast is a no-op) purely to keep them cheap.
_CORRECTNESS_TOKENS = {"smoke": 8, "stability": 64, "determinism": 128, "full": 256}
# "bench" is NOT a cheap smoke check -- it runs the EXACT dtype/token
# combination _bench_worker times (fp8 dispatch + bf16 combine,
# _BENCH_TOKENS_PER_RANK tokens/rank). Without this, a config could pass
# every other mode at bf16/<=256 tokens while being wrong (or merely
# untested) at the fp8/4096-token shape the benchmark and forge-loop's KEEP
# decision actually score -- this mode exists to close exactly that hole.
_CORRECTNESS_DTYPES = {
    "smoke": (torch.bfloat16, torch.bfloat16),
    "stability": (torch.bfloat16, torch.bfloat16),
    "determinism": (torch.bfloat16, torch.bfloat16),
    "full": (torch.bfloat16, torch.bfloat16),
    "bench": (torch.float8_e4m3fnuz, torch.bfloat16),
}
_SEED = 0
_CASE_ID = f"ep8_{_BENCH_TOKENS_PER_RANK}tok_h{_HIDDEN_DIM}_top{_NUM_EXPERTS_PER_TOKEN}"


def _dist_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = _MASTER_PORT
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)


def _dist_teardown(op=None, *, healthy: bool = True) -> None:
    """Best-effort, deadlock-safe teardown.

    ``healthy`` must be False whenever this rank is unwinding through an
    exception. dispatch()/combine() are themselves cross-rank collectives, so
    if THIS rank failed mid-round, the other 7 ranks may currently be blocked
    inside a collective this rank never issued -- calling another collective
    here (``dist.barrier()``) would just add a second deadlock on top of the
    first. Only a healthy rank (the common case: normal end-of-run cleanup)
    barriers, and even then every step is best-effort (mirrors mori's own
    "Complete Example" teardown order in docs/MORI-EP-GUIDE.md: ``del op`` ->
    ``mori.shmem.shmem_finalize()`` -> ``dist.destroy_process_group()``) so one
    failing step never blocks the next.
    """
    # shmem_finalize() on a rank that never actually initialized shmem (e.g.
    # config validation or kernel_type validation raised before
    # shmem_torch_process_group_init() ran) is not a clean Python exception
    # -- it's a native SIGABRT. ``op is not None`` is a reliable proxy for
    # "shmem was initialized": every call site initializes shmem immediately
    # before constructing the op, never after.
    had_op = op is not None
    if op is not None:
        try:
            del op
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    if healthy and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
    if had_op:
        try:
            import mori

            mori.shmem.shmem_finalize()
        except Exception:  # noqa: BLE001 - not fatal if already torn down/unavailable
            pass
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


# Both are single-node kernel families (no RDMA fabric needed) so both are
# legal on this 8-GPU box; IntraNodeLL ("low latency") shares the same
# .hsaco module as IntraNode (mori/python/mori/ops/dispatch_combine.py
# _KERNEL_TYPE_TO_HIP) and the same block_num/warp_per_block launch-config
# surface -- it is a different kernel entry point selected at construction
# time, not a per-call knob, so it is looked up here (not passed to
# dispatch()/combine() like block_num/warp_per_block are).
_KERNEL_TYPE_MAP = {
    "IntraNode": "IntraNode",
    "IntraNodeLL": "IntraNodeLL",
}

# All 6 keys are mandatory (program.md says so explicitly). Enforcing that
# HERE, with a direct error message, matters because the alternative is an
# uncaught KeyError from a bare `cfg["dispatch_block_num"]` deep inside a
# spawned rank -- opaque, and it contradicts program.md's own contract
# instead of failing loudly against it.
_REQUIRED_CFG_KEYS = (
    "dispatch_block_num", "dispatch_warp_per_block",
    "combine_block_num", "combine_warp_per_block",
    "kernel_type", "combine_zero_copy",
)


def _validated_cfg() -> dict:
    cfg = get_ep_launch_config()
    missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"get_ep_launch_config() is missing required key(s) {missing} -- "
            f"all of {list(_REQUIRED_CFG_KEYS)} must be present in the "
            "returned dict (see mori_ep_config.py's docstring)."
        )
    return cfg


def _make_config(rank: int, world_size: int, dtype, max_tokens: int, kernel_type_name: str = "IntraNode"):
    import mori

    if kernel_type_name not in _KERNEL_TYPE_MAP:
        # Fail loudly and fail here -- before any GPU/HIP work happens -- rather
        # than silently substituting a different kernel type. An agent-supplied
        # ``kernel_type`` that isn't legal on this single-node box (e.g. an
        # InterNode* family, which needs an RDMA fabric this box doesn't have,
        # or a typo) must fail the correctness gate loudly, not quietly
        # benchmark IntraNode under a mislabeled config.
        raise ValueError(
            f"invalid kernel_type {kernel_type_name!r} in mori_ep_config.py -- "
            f"must be one of {sorted(_KERNEL_TYPE_MAP)}. This single-node box "
            "has no RDMA fabric configured, so InterNode/InterNodeV1/"
            "InterNodeV1LL/AsyncLL are not selectable here."
        )
    kt_name = _KERNEL_TYPE_MAP[kernel_type_name]
    kernel_type = getattr(mori.ops.EpDispatchCombineKernelType, kt_name)
    config = mori.ops.EpDispatchCombineConfig(
        data_type=dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=_HIDDEN_DIM,
        scale_dim=0,
        scale_type_size=4,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_tokens,
        num_experts_per_rank=_NUM_EXPERTS_PER_RANK,
        num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
        max_total_recv_tokens=0,
        warp_num_per_block=8,
        block_num=80,
        # Class-level default only -- actual mode is chosen per-call in
        # _combine_with_config() via cfg["combine_zero_copy"], which is the
        # tunable the agent controls (see mori_ep_config.py).
        use_external_inp_buf=True,
        gpu_per_node=world_size,
        quant_type="none",
        kernel_type=kernel_type,
    )
    mori.shmem.shmem_torch_process_group_init("default")
    return config


def _build_op(rank: int, world_size: int, dtype, combine_dtype, max_tokens: int):
    import mori

    cfg = _validated_cfg()
    config = _make_config(rank, world_size, dtype, max_tokens, cfg["kernel_type"])
    op = mori.ops.EpDispatchCombineOp(config)
    return op, cfg


def _combine_with_config(
    op, cfg: dict, expert_output: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
    call_reset: bool = False, prime_buffer: bool = True,
):
    """Runs op.combine() honoring cfg["combine_zero_copy"] (default False).

    True  -> mori's registered zero-copy buffer path (op.get_registered_
             combine_input_buffer + use_external_inp_buf=0): the caller writes
             the expert output directly into MORI's own peer-visible buffer,
             skipping the internal copy the "external buffer" path performs.
             docs/MORI-EP-GUIDE.md §2/§3; mori's own tuner
             (tools/batch_intranode_tuning.sh) treats this as a first-class
             tuning axis with its own optimal block_num/warp_per_block, not a
             free win layered on top of the external-buffer optimum -- do not
             assume the external-buffer best config transfers.
    False -> the externally-managed buffer path (use_external_inp_buf=1),
             which is what every prior campaign in this task used and is
             mori's class-level default.
    Both paths must be wired here (not just in the benchmark) so the
    correctness gate validates the SAME combine call shape the benchmark
    times -- gating on the wrong path would validate nothing.

    ``prime_buffer`` (zero-copy path only): whether to copy ``expert_output``
    into the registered buffer before calling combine(). Correctness callers
    must leave this True (the buffer has to actually contain the real expert
    output to check anything). The benchmark's timed loop passes False after
    priming the buffer once, outside the timed region: in true zero-copy
    usage the expert GEMM writes directly into the registered buffer, so a
    ``copy_()`` inside the timed window would measure "external buffer +
    an extra manual copy", not zero-copy -- see local_knowledge KB card for
    the measured impact of getting this wrong.
    """
    block_num = cfg["combine_block_num"]
    warp_per_block = cfg["combine_warp_per_block"]
    if cfg["combine_zero_copy"]:
        buf = op.get_registered_combine_input_buffer(expert_output.dtype)
        n = expert_output.size(0)
        if prime_buffer:
            buf[:n].copy_(expert_output)
        return op.combine(
            buf[:n], weights, indices,
            block_num=block_num, warp_per_block=warp_per_block,
            use_external_inp_buf=0, call_reset=call_reset,
        )
    return op.combine(
        expert_output, weights, indices,
        block_num=block_num, warp_per_block=warp_per_block,
        use_external_inp_buf=1, call_reset=call_reset,
    )


def _make_routing(n_tokens: int, world_size: int, rank: int, device, scale: float):
    torch.manual_seed(_SEED + rank)
    total_experts = _NUM_EXPERTS_PER_RANK * world_size
    x = torch.randn(n_tokens, _HIDDEN_DIM, device=device, dtype=torch.bfloat16) * scale
    indices = torch.empty(n_tokens, _NUM_EXPERTS_PER_TOKEN, dtype=torch.int32, device=device)
    for i in range(n_tokens):
        perm = torch.randperm(total_experts, device=device)
        indices[i] = perm[: _NUM_EXPERTS_PER_TOKEN].to(torch.int32)
    # Uniform weights summing to 1 per token: an identity-expert round trip
    # then reconstructs the original token exactly (modulo bf16 rounding).
    weights = torch.full(
        (n_tokens, _NUM_EXPERTS_PER_TOKEN), 1.0 / _NUM_EXPERTS_PER_TOKEN,
        dtype=torch.float32, device=device,
    )
    return x, weights, indices


def _correctness_worker(
    rank: int, world_size: int, n_tokens: int, scale: float, dtype, combine_dtype, result_path: str,
) -> None:
    _dist_setup(rank, world_size)
    op = None
    healthy = False
    try:
        import mori

        # Reuse mori's own tested dispatch/combine reference-checking math
        # (tests/python/ops/dispatch_combine_test_utils.py) instead of a
        # hand-rolled identity-reconstruction check: combine's real contract
        # is "unique-destination-PE dedup'd sum of the expert output,
        # unweighted" (routing-weight multiply is the CALLER's job via the
        # returned combine_output_weight), which is easy to get subtly wrong
        # by hand. We still drive dispatch()/combine() with OUR OWN tunable
        # block_num/warp_per_block overrides, so this exercises exactly the
        # launch config the agent edits.
        sys.path.insert(0, os.environ.get("MORI_REPO_ROOT", "/work/mori"))
        from tests.python.ops.dispatch_combine_test_utils import EpDispatchCombineTestCase

        cfg = _validated_cfg()
        config = _make_config(rank, world_size, dtype, max(n_tokens, 1), cfg["kernel_type"])
        op = mori.ops.EpDispatchCombineOp(config)
        test_case = EpDispatchCombineTestCase(config)
        test_data = test_case.gen_test_data(num_token_override=[n_tokens] * world_size)
        _, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales = test_data
        if scale != 1.0:
            # Every rank deterministically regenerates the SAME all_rank_input
            # (fixed seed) as everyone else's view of the whole world, so the
            # scale must be applied uniformly across all ranks' entries here
            # — scaling only this rank's own slice would desync it from what
            # remote ranks (which independently recomputed the same list)
            # still believe this rank sent, and check_dispatch_result's
            # exact-equality assertion would then legitimately fail on them.
            all_rank_input = [t * scale for t in all_rank_input]
            test_data = (test_data[0], all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales)

        dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, dispatch_recv_num_token = op.dispatch(
            all_rank_input[rank], all_rank_weights[rank], all_rank_scales[rank], all_rank_indices[rank],
            block_num=cfg["dispatch_block_num"], warp_per_block=cfg["dispatch_warp_per_block"],
        )
        test_case.check_dispatch_result(
            op, test_data, dispatch_output, dispatch_weights, dispatch_scales,
            dispatch_indices, dispatch_recv_num_token,
        )

        # Identity expert: a real expert would still cast its dispatch-dtype
        # input to whatever (typically higher-precision) dtype it computes
        # in before writing an output combine() consumes; when dispatch and
        # combine dtypes match (the bf16 modes) this cast is a no-op, but for
        # the fp8-dispatch/bf16-combine shape (matching the benchmark) it is
        # a REAL cast that must happen for combine to see the dtype it
        # expects -- mirrors _bench_worker's separately-allocated bf16
        # combine_input, just derived from dispatch_output instead of being
        # random (identity expert has no other transformation to apply).
        expert_output = dispatch_output.to(combine_dtype)
        combine_output, combine_output_weight = _combine_with_config(
            op, cfg, expert_output, dispatch_weights, all_rank_indices[rank], call_reset=True,
        )
        # combine_data_type defaults to config.data_type (dispatch's dtype) --
        # must be overridden explicitly here whenever combine_dtype differs
        # (the fp8-dispatch/bf16-combine "bench" mode), or check_combine_
        # result computes its reference at the WRONG precision and every
        # token spuriously "mismatches" by exactly an fp8-rounding delta.
        test_case.check_combine_result(
            op, test_data, combine_output, combine_output_weight, combine_data_type=combine_dtype,
        )
        torch.cuda.synchronize()

        if rank == 0:
            with open(result_path, "w") as f:
                f.write("ok\n")
        healthy = True
    finally:
        _dist_teardown(op, healthy=healthy)


def _capture_bench_graphs(op, cfg, x_fp8, weights, indices, combine_input, n_recv):
    """Per-rank CUDA graph capture of one dispatch+combine round.

    Mirrors mori's own reference benchmark (tests/python/ops/
    bench_dispatch_combine.py:_capture_split_graphs) as closely as possible:
    two SEPARATE graphs (so each half's replay can be timed independently if
    ever needed), with combine's capture referencing the actual tensor
    object dispatch's capture produced -- CUDA graphs bake in fixed memory
    addresses, so combine must read from the SAME static buffer dispatch's
    replay will refill, not a fresh tensor. Deliberately no ``call_reset``
    here: mori's own reference graph-capture path doesn't reset between
    replays either, so this follows their validated pattern rather than
    guessing at graph-capture-specific reset semantics that aren't
    documented. This is why graph-mode is opt-in (--graph-mode), not the
    default this driver's every-iteration eager path uses.
    """
    dispatch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dispatch_graph):
        _do, dispatch_weights, _dsc, _di, _rn = op.dispatch(
            x_fp8, weights, None, indices,
            block_num=cfg["dispatch_block_num"], warp_per_block=cfg["dispatch_warp_per_block"],
        )

    combine_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(combine_graph):
        _combine_with_config(
            op, cfg, combine_input[:n_recv], dispatch_weights, indices, prime_buffer=False,
        )
    torch.cuda.synchronize()
    return dispatch_graph, combine_graph


def _bench_worker(
    rank: int, world_size: int, warmup: int, iters: int, graph_mode: bool, result_path: str,
) -> None:
    _dist_setup(rank, world_size)
    op = None
    healthy = False
    try:
        device = torch.device("cuda", rank)
        op, cfg = _build_op(rank, world_size, torch.float8_e4m3fnuz, torch.bfloat16, _BENCH_TOKENS_PER_RANK)
        x_fp8, weights, indices = _make_routing(_BENCH_TOKENS_PER_RANK, world_size, rank, device, 1.0)
        x_fp8 = x_fp8.to(torch.float8_e4m3fnuz)
        # Combine's expert-output input must be in combine dtype (bf16); its
        # numeric content does not matter for a bandwidth/latency benchmark.
        combine_input = torch.randn(
            _BENCH_TOKENS_PER_RANK * _NUM_EXPERTS_PER_TOKEN, _HIDDEN_DIM,
            device=device, dtype=torch.bfloat16,
        )

        # dispatch's actual recv count is data-dependent (depends on this
        # iteration's random routing) and only known device-side until a
        # host sync -- but x_fp8/weights/indices are fixed for the whole
        # run (generated once above, not re-randomized per round), so
        # recv_n is IDENTICAL on every round for a given rank. Do the one
        # unavoidable host sync ONCE here, outside any timed region,
        # instead of inside one_round() every iteration (that forced
        # per-iteration sync was destroying any real dispatch/combine
        # overlap and inflating every measured number -- see KB card).
        _ds0, dispatch_weights0, _dsc0, _di0, recv_n0 = op.dispatch(
            x_fp8, weights, None, indices,
            block_num=cfg["dispatch_block_num"], warp_per_block=cfg["dispatch_warp_per_block"],
        )
        n_recv = int(recv_n0.sum().item()) if hasattr(recv_n0, "sum") else int(recv_n0)
        n_recv = max(n_recv, 1)
        # Prime the zero-copy registered buffer (a no-op read for the
        # external-buffer path) with this fixed-content slice ONCE; the
        # timed loop below never copies into it again.
        _combine_with_config(
            op, cfg, combine_input[:n_recv], dispatch_weights0, indices, call_reset=True, prime_buffer=True,
        )
        torch.cuda.synchronize()

        if graph_mode:
            dispatch_graph, combine_graph = _capture_bench_graphs(
                op, cfg, x_fp8, weights, indices, combine_input, n_recv,
            )

            def one_round_local_ms() -> float:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                start.record()
                dispatch_graph.replay()
                combine_graph.replay()
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end)
        else:

            def one_round_local_ms() -> float:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                start.record()
                _do, dispatch_weights, _dsc, _di, _rn = op.dispatch(
                    x_fp8, weights, None, indices,
                    block_num=cfg["dispatch_block_num"], warp_per_block=cfg["dispatch_warp_per_block"],
                )
                # call_reset=True: mori's own docs require reset() "between
                # iterations" for repeated eager (non-graph) calls -- a real
                # eager-mode production serving loop pays this same cost
                # every round, so it belongs inside the timed region, not
                # around it.
                _combine_with_config(
                    op, cfg, combine_input[:n_recv], dispatch_weights, indices,
                    call_reset=True, prime_buffer=False,
                )
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end)

        def one_round() -> float:
            # Synchronized collective: the round's true duration is the
            # SLOWEST rank, taken per-round (not each rank's own median
            # across rounds, maxed afterward -- that mixes rounds together
            # and isn't a real per-round statistic). Every rank must join
            # this all_reduce every round since it is itself a collective.
            t = torch.tensor([one_round_local_ms()], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return t.item()

        for _ in range(warmup):
            one_round()

        round_ms = [one_round() for _ in range(iters)]
        if rank == 0:
            with open(result_path, "w") as f:
                f.write("\n".join(f"{v:.6f}" for v in round_ms) + "\n")
        healthy = True
    finally:
        _dist_teardown(op, healthy=healthy)


def _profile_worker(rank: int, world_size: int) -> None:
    _dist_setup(rank, world_size)
    op = None
    healthy = False
    try:
        device = torch.device("cuda", rank)
        op, cfg = _build_op(rank, world_size, torch.float8_e4m3fnuz, torch.bfloat16, _BENCH_TOKENS_PER_RANK)
        x_fp8, weights, indices = _make_routing(_BENCH_TOKENS_PER_RANK, world_size, rank, device, 1.0)
        x_fp8 = x_fp8.to(torch.float8_e4m3fnuz)
        combine_input = torch.randn(
            _BENCH_TOKENS_PER_RANK * _NUM_EXPERTS_PER_TOKEN, _HIDDEN_DIM,
            device=device, dtype=torch.bfloat16,
        )
        for _ in range(3):
            dispatch_output, dispatch_weights, _ds, _di, recv_n = op.dispatch(
                x_fp8, weights, None, indices,
                block_num=cfg["dispatch_block_num"], warp_per_block=cfg["dispatch_warp_per_block"],
            )
            n_recv = int(recv_n.sum().item()) if hasattr(recv_n, "sum") else int(recv_n)
            # call_reset=True between iterations -- see _bench_worker.
            _combine_with_config(
                op, cfg, combine_input[: max(n_recv, 1)], dispatch_weights, indices, call_reset=True,
            )
        torch.cuda.synchronize()
        healthy = True
    finally:
        _dist_teardown(op, healthy=healthy)


def _run_correctness(mode: str) -> int:
    n_tokens = _BENCH_TOKENS_PER_RANK if mode == "bench" else _CORRECTNESS_TOKENS.get(
        mode, _CORRECTNESS_TOKENS["full"],
    )
    dtype, combine_dtype = _CORRECTNESS_DTYPES.get(mode, _CORRECTNESS_DTYPES["full"])
    scale = 1000.0 if mode == "stability" else 1.0
    result_path = f"/tmp/.mori_forge_correctness_result_{os.getpid()}.txt"
    try:
        mp.spawn(
            _correctness_worker,
            args=(_WORLD_SIZE, n_tokens, scale, dtype, combine_dtype, result_path),
            nprocs=_WORLD_SIZE,
            join=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a driver failure, not a crash
        print("allclose: False")
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ok = os.path.exists(result_path)
    if ok:
        os.remove(result_path)
    # Reference-checked via mori's own dispatch/combine assertions (see
    # _correctness_worker) rather than a hand-computed SNR — the README's
    # documented fallback for drivers that gate on a pass/fail check.
    print(f"allclose: {ok}")
    return 0


def _run_bench(warmup: int, iters: int, graph_mode: bool = False) -> int:
    result_path = f"/tmp/.mori_forge_bench_result_{os.getpid()}.txt"
    try:
        mp.spawn(
            _bench_worker,
            args=(_WORLD_SIZE, warmup, iters, graph_mode, result_path),
            nprocs=_WORLD_SIZE,
            join=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        with open(result_path) as f:
            round_ms = [float(line) for line in f.read().splitlines() if line.strip()]
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)
    if not round_ms:
        print("error: bench worker produced no timing samples", file=sys.stderr)
        return 1
    # Per examples/README.md §2: one `wall_ms:` line per timed iteration (each
    # already the per-round max-across-ranks value -- see _bench_worker) plus
    # one `case_ms:` line forge scores on. Printing the samples (not just the
    # aggregate) lets forge take its own median and lets a human sanity-check
    # round-to-round variance directly from stdout.
    for v in round_ms:
        print(f"wall_ms: {v:.6f}")
    case_ms = sorted(round_ms)[len(round_ms) // 2]
    print(f"case_ms: {_CASE_ID} {case_ms:.6f}")
    return 0


def _run_profile(case_id: str) -> int:
    if case_id and case_id != _CASE_ID:
        print(f"error: unknown profile case: {case_id!r} (only {_CASE_ID!r} exists)", file=sys.stderr)
        return 2
    try:
        mp.spawn(_profile_worker, args=(_WORLD_SIZE,), nprocs=_WORLD_SIZE, join=True)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="forge-loop MoRI-EP dispatch/combine driver")
    parser.add_argument("--shape", default="default")
    parser.add_argument("--mode", default="full", help="smoke|stability|determinism|full|bench")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--graph-mode", action="store_true",
        help="bench-mode only: replay each rank's dispatch/combine from a "
             "per-rank CUDA graph instead of eager calls (closer to a "
             "production graph-captured serving loop; see driver.py's "
             "module docstring). Off by default -- the default eager path "
             "is what forge-loop scores every candidate config against.",
    )
    args, _unknown = parser.parse_known_args()

    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)")
        return 1
    if torch.cuda.device_count() < _WORLD_SIZE:
        print(f"error: need {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
        return 1

    mp.set_start_method("spawn", force=True)

    if args.profile_run:
        return _run_profile(args.profile_case)
    if args.bench_mode:
        return _run_bench(args.warmup, args.iters, args.graph_mode)
    return _run_correctness(args.mode)


if __name__ == "__main__":
    sys.exit(main())
