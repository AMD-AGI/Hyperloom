# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KernelForge driver: the LM head's cross-entropy, forward and backward, at a Primus training step's shapes.

* no flags: correctness of ``kernel.py`` against an fp32 reference, one ``allclose:`` line;
* ``--bench-mode``: one ``case_ms:`` line per case (forward plus backward, median), then ``median_ms:``;
* ``--baseline-json PATH``: the same times for Megatron's native path (the anchor to beat);
* ``--phases-json PATH``: forward and backward apart, baseline and kernel, per case (what Primus is fed),
  and Transformer Engine's fused cross-entropy (``te_ms``) where it imports;
* ``--bench-case ID``: with ``--bench-mode``, only that case;
* ``--profile-run``: one warmup plus two calls of the kernel on one case, for a profiler.

Timing replays CUDA/HIP graphs of the forward and of the backward; the logits restore is untimed.

The cases and the baseline live here, not in ``kernel.py``: the agents edit the kernel only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TASK = json.loads((HERE / "task.json").read_text())
CASES = [(f"T={c['tokens']}", int(c["tokens"]), int(c["vocab"])) for c in TASK["cases"]]
CHECK_CASES = [(f"T={c['tokens']}", int(c["tokens"]), int(c["vocab"])) for c in TASK["check_cases"]]
SEED = 0
CHUNK = 4096
DEVICE = "cuda"
INF, NEG_INF = float("inf"), float("-inf")
#: Untimed calls before a profiled one: JIT and autotuning settle in the first.
PROFILE_CALLS = 3


def _kernel():
    spec = importlib.util.spec_from_file_location("kernel", HERE / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    spec.loader.exec_module(module)
    return module


class _Baseline:
    """Megatron-LM's native fused cross-entropy at tensor-parallel size 1 (fp32 softmax kept for the backward)."""

    @staticmethod
    @torch.compile
    def _logits_max(logits):
        logits = logits.float()
        return logits, torch.max(logits, dim=-1)[0]

    @staticmethod
    @torch.compile
    def _predicted_and_exp(logits, target, logits_max):
        rows = torch.arange(logits.size(0), device=logits.device)
        predicted = logits[rows, target] - logits_max
        logits.sub_(logits_max.unsqueeze(-1))
        torch.exp(logits, out=logits)
        return predicted, logits.sum(dim=-1)

    @staticmethod
    @torch.compile
    def _loss_and_softmax(exp_logits, predicted, sum_exp):
        loss = torch.log(sum_exp) - predicted
        exp_logits.div_(sum_exp.unsqueeze(-1))
        return exp_logits, loss

    @staticmethod
    @torch.compile
    def _gradients(softmax, grad_loss, target):
        rows = torch.arange(softmax.size(0), device=softmax.device)
        softmax[rows, target] -= 1.0
        softmax.mul_(grad_loss.unsqueeze(-1))
        return softmax

    @classmethod
    def forward(cls, logits, target):
        fp32, logits_max = cls._logits_max(logits)
        predicted, sum_exp = cls._predicted_and_exp(fp32, target, logits_max)
        softmax, loss = cls._loss_and_softmax(fp32, predicted, sum_exp)
        return loss, (softmax, target, logits.dtype)

    @classmethod
    def backward(cls, ctx, grad_loss):
        softmax, target, dtype = ctx
        return cls._gradients(softmax, grad_loss, target).to(dtype)


class _TransformerEngine:
    """Transformer Engine's fused cross-entropy, what Megatron runs with ``cross_entropy_fusion_impl: te``.

    Its forward writes the loss's gradient into the logits as it goes; its backward only scales that by
    ``grad_loss``, so its time sits mostly in the forward.
    """

    @staticmethod
    def available() -> bool:
        try:
            from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy  # noqa: F401
        except (ImportError, OSError, RuntimeError):
            return False
        return True

    @staticmethod
    def forward(logits, target):
        from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

        x = logits.view(1, *logits.shape).detach().requires_grad_()
        loss = parallel_cross_entropy(x, target.view(1, -1), 0.0, False, None, -100, True)
        return loss.detach().view(-1), (loss, x)

    @staticmethod
    def backward(ctx, grad_loss):
        loss, x = ctx
        (grad,) = torch.autograd.grad(loss, x, grad_loss.view_as(loss))
        return grad.view(x.shape[1], x.shape[2])


def _inputs(tokens: int, vocab: int):
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    logits = (torch.randn(tokens, vocab, device=DEVICE, generator=gen) * 2.0).to(torch.bfloat16)
    # a few rows far from zero and one dominated by its target: the softmax must stay stable
    logits[: min(8, tokens)] += 40.0
    target = torch.randint(0, vocab, (tokens,), device=DEVICE, generator=gen)
    logits[min(8, tokens - 1), target[min(8, tokens - 1)]] = 60.0
    grad_loss = torch.rand(tokens, device=DEVICE, generator=gen) + 0.5
    return logits, target, grad_loss


def _check(module, tokens: int, vocab: int) -> tuple[bool, str]:
    logits, target, grad_loss = _inputs(tokens, vocab)
    master = logits.clone()
    loss, ctx = module.forward(logits, target)
    grad = module.backward(ctx, grad_loss)
    torch.cuda.synchronize()
    if loss.shape != (tokens,) or grad.shape != (tokens, vocab) or grad.dtype != torch.bfloat16:
        return False, f"shapes: loss {tuple(loss.shape)}, grad {tuple(grad.shape)} {grad.dtype}"
    worst_loss = 0.0
    err2 = ref2 = 0.0
    bad = 0
    for start in range(0, tokens, CHUNK):
        x = master[start : start + CHUNK].float()
        t = target[start : start + CHUNK]
        lse = torch.logsumexp(x, dim=-1)
        ref_loss = lse - x.gather(1, t[:, None])[:, 0]
        got = loss[start : start + CHUNK].float()
        worst_loss = max(worst_loss, ((got - ref_loss).abs() / (1e-3 + 1e-4 * ref_loss.abs())).max().item())
        ref = torch.exp(x - lse[:, None])
        ref[torch.arange(ref.size(0), device=DEVICE), t] -= 1.0
        ref *= grad_loss[start : start + CHUNK, None]
        g = grad[start : start + CHUNK].float()
        err2 += ((g - ref) ** 2).sum().item()
        ref2 += (ref**2).sum().item()
        bad += (~torch.isclose(g, ref, rtol=1.6e-2, atol=1e-6)).sum().item()
    rel = (err2 / ref2) ** 0.5 if ref2 else INF
    snr = 10.0 * math.log10(ref2 / err2) if err2 > 0 and math.isfinite(err2) else (INF if err2 == 0 else NEG_INF)
    ok = worst_loss <= 1.0 and rel <= 4e-3 and bad == 0
    return ok, snr, f"loss_err/tol={worst_loss:.3f} grad_rel_l2={rel:.2e} grad_out_of_tol={bad} grad_snr_db={snr:.2f}"


def _run_correctness() -> int:
    module = _kernel()
    passed = []
    snrs = []
    for case_id, tokens, vocab in CHECK_CASES:
        try:
            ok, snr, detail = _check(module, tokens, vocab)
        except Exception as error:  # noqa: BLE001 -- a candidate that raises is wrong, not a driver failure
            traceback.print_exc(file=sys.stderr)
            ok, snr, detail = False, NEG_INF, f"{type(error).__name__}: {error}"
        print(f"check: {case_id} {'ok' if ok else 'FAIL'} {detail}")
        passed.append(ok)
        snrs.append(snr)
        torch.cuda.empty_cache()
    worst = min(snrs) if snrs else NEG_INF
    # a failed tolerance check must not pass the SNR pre-filter
    print(f"SNR: {worst if all(passed) else min(worst, 0.0):.2f} dB")
    print(f"allclose: {all(passed)}")
    return 0


def _snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref = ref.float()
    err = (out.float() - ref).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    if err == 0.0:
        return INF
    if not math.isfinite(err):
        return NEG_INF
    return 10.0 * math.log10(max(sig, 1e-30) / err)


def _ref_loss(master: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = torch.empty(master.size(0), device=master.device, dtype=torch.float32)
    for start in range(0, master.size(0), CHUNK):
        x = master[start : start + CHUNK].float()
        t = target[start : start + CHUNK]
        out[start : start + CHUNK] = torch.logsumexp(x, dim=-1) - x.gather(1, t[:, None])[:, 0]
    return out


def _ref_grad_rows(master, target, grad_loss, rows: int) -> torch.Tensor:
    x = master[:rows].float()
    ref = torch.softmax(x, dim=-1)
    ref[torch.arange(rows, device=x.device), target[:rows]] -= 1.0
    return ref * grad_loss[:rows, None]


def _warmup(impl, inputs, logits, warmup: int) -> None:
    """Eager calls on a side stream: torch.compile, Triton JIT and autotuning happen here, not in capture."""
    master, target, grad_loss = inputs
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            logits.copy_(master)
            loss, ctx = impl.forward(logits, target)
            grad = impl.backward(ctx, grad_loss)
            del loss, ctx, grad
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _capture(impl, inputs, logits):
    """The forward and the backward captured into two graphs sharing one pool, checked to recompute on replay."""
    master, target, grad_loss = inputs
    logits.copy_(master)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    g_fwd = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_fwd, pool=pool):
        loss, ctx = impl.forward(logits, target)
    g_bwd = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_bwd, pool=pool):
        grad = impl.backward(ctx, grad_loss)
    torch.cuda.synchronize()

    # Capture-validity guard: poison the static outputs, replay, and check the replay recomputed them.
    ref_loss = _ref_loss(master, target)
    rows = min(64, master.size(0))
    ref_grad = _ref_grad_rows(master, target, grad_loss, rows)
    loss.fill_(float("nan"))
    grad[:rows].fill_(float("nan"))
    logits.copy_(master)
    torch.cuda.synchronize()
    g_fwd.replay()
    g_bwd.replay()
    torch.cuda.synchronize()
    snr_loss = _snr_db(ref_loss, loss)
    snr_grad = _snr_db(ref_grad, grad[:rows])
    if not (snr_loss > 30.0 and snr_grad > 30.0):
        raise RuntimeError(
            f"graph replay did not recompute a correct result (loss SNR {snr_loss:.1f} dB, "
            f"grad SNR {snr_grad:.1f} dB): the kernel was not captured on the current stream"
        )
    return g_fwd, g_bwd, (loss, ctx, grad)


def _event():
    return torch.cuda.Event(enable_timing=True)


def _replay_times(graphs, master, logits, iters: int) -> tuple[list[float], list[float]]:
    """Per timed iteration, the forward and the backward graph replays' ms, after two to reach steady state."""
    g_fwd, g_bwd = graphs
    for _ in range(2):
        logits.copy_(master)
        g_fwd.replay()
        g_bwd.replay()
    torch.cuda.synchronize()
    fwd, bwd = [], []
    start, mid, end = _event(), _event(), _event()
    for _ in range(iters):
        logits.copy_(master)  # untimed restore
        start.record()
        g_fwd.replay()
        mid.record()
        g_bwd.replay()
        end.record()
        torch.cuda.synchronize()
        fwd.append(start.elapsed_time(mid))
        bwd.append(mid.elapsed_time(end))
    return fwd, bwd


def _time(impl, tokens: int, vocab: int, warmup: int, iters: int) -> tuple[float, float, float]:
    """Graph-replayed forward and backward: median fwd ms, median bwd ms, median (fwd + bwd) ms.

    The forward and the backward are each captured once into a CUDA/HIP graph (sharing one memory
    pool, so the forward's ctx feeds the backward's static input). Before each timed iteration the
    logits are restored from a pristine copy, untimed (the kernel may consume them in place); the
    events then bracket one forward-graph replay and one backward-graph replay.
    """
    inputs = _inputs(tokens, vocab)
    logits = torch.empty_like(inputs[0])
    _warmup(impl, inputs, logits, warmup)
    g_fwd, g_bwd, outputs = _capture(impl, inputs, logits)
    fwd, bwd = _replay_times((g_fwd, g_bwd), inputs[0], logits, iters)
    del g_fwd, g_bwd, outputs, inputs, logits
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return statistics.median(fwd), statistics.median(bwd), statistics.median([f + b for f, b in zip(fwd, bwd)])


def _bench(impl, warmup: int, iters: int, cases=None) -> dict[str, tuple[float, float, float]]:
    cases = CASES if cases is None else cases
    return {case_id: _time(impl, tokens, vocab, warmup, iters) for case_id, tokens, vocab in cases}


def _run_bench(warmup: int, iters: int, bench_case: str = "") -> int:
    cases = CASES
    if bench_case:
        cases = [c for c in CASES if c[0] == bench_case]
        if not cases:
            print(f"unknown case {bench_case!r}; declared: {' '.join(c[0] for c in CASES)}", file=sys.stderr)
            return 2
    times = _bench(_kernel(), warmup, iters, cases)
    for case_id, (f, b, t) in times.items():
        print(f"phase_ms: {case_id} fwd={f:.4f} bwd={b:.4f}")
        print(f"case_ms: {case_id} {t:.6f}")
    # sum over the cases of each case's median (forward + backward) graph-replay time
    print(f"median_ms: {sum(t for _, _, t in times.values()):.6f}")
    return 0


def _write_baseline(path: str, warmup: int, iters: int) -> int:
    times = _bench(_Baseline, warmup, iters)
    case_times = {case_id: t for case_id, (_, _, t) in times.items()}
    payload = {
        "wall_ms": sum(case_times.values()),
        "case_times": case_times,
        "label": "Megatron native cross-entropy",
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


def _write_phases(path: str, warmup: int, iters: int) -> int:
    base = _bench(_Baseline, warmup, iters)
    kern = _bench(_kernel(), warmup, iters)
    te = _bench(_TransformerEngine, warmup, iters) if _TransformerEngine.available() else {}
    cases = {}
    for case_id in base:
        cases[case_id] = {}
        for i, phase in enumerate(("fwd", "bwd")):
            rec = {"baseline_ms": base[case_id][i], "kernel_ms": kern[case_id][i]}
            if te:
                rec["te_ms"] = te[case_id][i]
            cases[case_id][phase] = rec
    payload = {"op": TASK["op"], "scope": TASK.get("scope"), "cases": cases}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


PROFILE_CASE = CASES[-1]  # the smaller scored case: same kernels, half the memory


def _run_profile() -> int:
    """Kernel only (no reference, no timing output): one warmup to settle JIT, then two profiled calls."""
    module = _kernel()
    _, tokens, vocab = PROFILE_CASE
    master, target, grad_loss = _inputs(tokens, vocab)
    logits = torch.empty_like(master)
    for _ in range(PROFILE_CALLS):
        logits.copy_(master)
        loss, ctx = module.forward(logits, target)
        grad = module.backward(ctx, grad_loss)
        del loss, ctx, grad
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--phases-json", default="")
    parser.add_argument("--bench-case", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args, _ = parser.parse_known_args()
    if args.baseline_json:
        return _write_baseline(args.baseline_json, args.warmup, args.iters)
    if args.phases_json:
        return _write_phases(args.phases_json, args.warmup, args.iters)
    if args.profile_run:
        return _run_profile()
    if args.bench_mode:
        return _run_bench(args.warmup, args.iters, args.bench_case)
    return _run_correctness()


if __name__ == "__main__":
    sys.exit(main())
