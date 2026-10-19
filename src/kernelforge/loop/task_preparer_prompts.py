# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed text the task-preparation agent is given: contract, reference driver, and prompt sections."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path


#: Above this the spec is referenced by path instead of inlined in the prompt.
#: An observed document runs a few kilobytes -- it is bounded by the operand
#: count -- so this only guards against a producer that grew without warning.
_SPEC_INLINE_MAX_BYTES = 64 * 1024


def _reference_note(ref_dir: Path | None, workspace: Path) -> str:
    """Prompt block that points the agent at the on-disk reference files to Read."""
    if ref_dir is None or not ref_dir.is_dir():
        return "No reference files were available; follow the contract above."
    rel_root = os.path.relpath(ref_dir, workspace)
    lines = [
        "## Reference files to Read (real, complete — do NOT rely on memory)",
        f"A copy of KernelForge's shipped reference material is in `./{rel_root}/`.",
        # A driver authored against this directory passed validation and then crashed on the loop's first baseline
        # bench, because the directory is deleted between the two.
        f"`./{rel_root}/` is TEMPORARY authoring scaffolding: it is DELETED before "
        "your driver is validated and committed, so read it now but never read it "
        "at runtime — do not open, import, or glob anything under it from the "
        "driver or its helpers.",
        "Read the contract and ONE reference driver before writing; the others "
        "are there if the first does not match your case:",
    ]
    readme = ref_dir / "README.md"
    if readme.is_file():
        lines.append(f"- `./{rel_root}/README.md` — the full driver contract (files + rules)")
    contract = ref_dir / "CONTRACT.md"
    if contract.is_file():
        lines.append(f"- `./{rel_root}/CONTRACT.md` — the driver contract")
    # List each example task's key files (driver + any harness).
    for sub in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        drv = sub / "driver.py"
        if drv.is_file():
            rel = os.path.relpath(sub, workspace)
            extras = [f.name for f in sub.iterdir() if f.name in ("graph_harness.py", "program.md")]
            extra = f" (+ {', '.join(sorted(extras))})" if extras else ""
            lines.append(f"- `./{rel}/driver.py`{extra} — a complete working reference driver")
    tmpl = ref_dir / "driver_template.py"
    if tmpl.is_file():
        lines.append(f"- `./{rel_root}/driver_template.py` — a minimal driver skeleton to adapt")
    return "\n".join(lines)


class _SpecInline(NamedTuple):
    """The spec's text, or a statement of why it is not below."""

    text: str
    #: Completes "The specification at `./<path>` ...". Empty when ``text`` is.
    refusal: str
    #: What the agent should do instead. Empty when ``text`` is usable.
    recourse: str


_SPEC_RECOVER_FROM_SOURCE = (
    "Recover the public callable, the operand shapes and dtypes, and the "
    "deployment context from the kernel source and the tests it names; do not "
    "invent them and do not fall back to round numbers of your own choosing."
)


def _invocation_spec_text(spec_path: Path) -> _SpecInline:
    """Return the spec verbatim, or say precisely why it is not inlined."""
    try:
        text = spec_path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError) as error:
        return _SpecInline(
            "",
            f"could not be read ({type(error).__name__})",
            "Try `Read` on it yourself; if that fails too, "
            + _SPEC_RECOVER_FROM_SOURCE[0].lower()
            + _SPEC_RECOVER_FROM_SOURCE[1:],
        )
    if not text:
        return _SpecInline(
            "",
            "is empty, so it declares no invocation evidence at all",
            _SPEC_RECOVER_FROM_SOURCE,
        )
    if len(text.encode("utf-8")) > _SPEC_INLINE_MAX_BYTES:
        # Nothing observed comes close -- the document is bounded by the operand count -- but a prompt is the wrong
        # place to find out that some producer emitted a megabyte.
        return _SpecInline(
            "",
            f"is larger than the {_SPEC_INLINE_MAX_BYTES // 1024} KB inline "
            "limit, so it is referenced rather than quoted",
            "Use `Read` on it before you touch the driver.",
        )
    return _SpecInline(text, "", "")


def _invocation_spec_note(spec_path: Path | None, workspace: Path) -> str:
    """Build the highest-priority prompt instruction for invocation evidence."""
    if spec_path is None:
        return ""
    rel_path = os.path.relpath(spec_path, workspace)
    inline = _invocation_spec_text(spec_path)
    if inline.text:
        spec_block = f"\n### The specification, verbatim\n```json\n{inline.text}\n```\n"
    else:
        spec_block = f"\nThe specification at `./{rel_path}` {inline.refusal}. {inline.recourse}\n"
    return f"""\
## Invocation specification — BUILD THE DRIVER FROM THIS
The JSON below is the authoritative evidence for this task. Build the driver
from it: the public callable that executes the operator, the ordered input and
output shapes and dtypes, the deployment batch and sequence-length context, the
editable source or device symbol (which may differ from the public callable),
and the relevant tests, benchmarks and runtime paths.

Benchmark the shapes and dtypes this operator actually runs at in the deployment
the spec describes. A toy size is not a smaller version of the real measurement,
it is a different one: a kernel tuned at a sequence length the workload never
serves can report a large speedup that disappears end to end. Where the spec
states the operand dims, use exactly those. Where it does not, recover them from
the kernel source and the tests it names, and size them from the deployment
context it carries -- do not fall back to round numbers of your own choosing.

`./{rel_path}` is DURABLE: it sits beside the driver, is committed with it, and
is the ONLY task input the driver may read at runtime. If your driver loads its
case table from this file, resolve the path relative to the driver's own
directory and nowhere else. Every other path handed to you below is authoring
scaffolding that will not exist when the driver runs.

The specification is read-only evidence. Do NOT edit it. Unknown or omitted
fields must be resolved from the referenced source/tests; do not invent
signatures, tensor shapes, dtypes, or correctness rules.
{spec_block}"""


RETRY_HEADING_DEFAULT = "Your previous attempt still did NOT pass the deterministic check"
RETRY_HEADING_NO_EDIT = "Your previous attempt did not change the driver at all"


def _distributed_contract_note(nproc: int) -> str:
    """What a driver must do when the task runs on more than one rank."""
    if nproc <= 1:
        return ""
    return f"""
## This task runs on {nproc} GPUs — measure it through `dist_harness`

The kernel is a collective: it only computes the right answer when {nproc} ranks
participate. You do NOT write the launch, the timing, or the reporting.
`dist_harness.py` sits beside the driver and owns all of it — the torchrun
re-exec, one device per rank, the process group, per-rank seeding, warmup, graph
capture, barrier placement, the cross-rank reduction, rank-0 printing and the
teardown. Do NOT rewrite it, and do NOT launch ranks yourself.

Your driver supplies three functions per case and hands them over:

```python
from dist_harness import Case, run

def build_inputs(ctx):        # ctx.rank, ctx.world_size, ctx.device, ctx.case_id
    ...                       # return whatever call_candidate/reference take

def call_candidate(ctx, inputs):
    ...                       # the kernel under optimization

def reference(ctx, inputs):
    ...                       # the matching torch.distributed collective

if __name__ == "__main__":
    raise SystemExit(run(
        [Case("default", build_inputs, call_candidate, reference)],
        world_size={nproc},
    ))
```

The harness calls `build_inputs` under a seed it chooses, once per rank and
again for the second parity call, so inputs cannot be identical across ranks or
across the two calls — which is what lets a collective that silently drops a
rank pass parity. It issues both candidate calls before comparing either, so a
second call overwriting the first's registered scratch buffer is visible. It
takes the SLOWEST rank's time and the WORST rank's SNR. None of that is yours to
remember.

What is still yours:

* Build whatever context the kernel needs inside `build_inputs`. A compiled
  collective usually takes an opaque handle (IPC buffer, registered workspace,
  communicator) that must be created and exchanged across ranks first. Read the
  kernel's own Python binding to see what it expects.
* Make `reference` a real `torch.distributed` collective. This is the one
  requirement the harness cannot hold for you: a single-GPU reference is still a
  function returning a tensor, and a candidate that quietly drops a rank's
  contribution matches it. Getting this wrong costs more than a wasted
  validation — it lets such a candidate report a real speedup and pass, and the
  end-to-end gate downstream can adopt on throughput alone without ever scoring
  accuracy. Nothing after you is guaranteed to catch it.
* Declare one `Case` per case id the task's invocation specification lists.

The deterministic check confirms that {nproc} ranks launched and that every one
of them measured inside the harness. A driver that launches ranks itself is
rejected for that, whatever else it got right.
"""


@dataclass(frozen=True)
class PromptBrief:
    """What every attempt's prompt states: the evidence, the driver path, the invocation and distributed notes."""

    evidence: str
    driver_rel: str
    invocation_note: str = ""
    distributed_note: str = ""


@dataclass(frozen=True)
class RetryNote:
    """Why the previous attempt failed, under the heading that frames it; empty on the first attempt."""

    failure: str = ""
    heading: str = RETRY_HEADING_DEFAULT


def _build_prompt(brief: PromptBrief, reference_note: str, retry_note: RetryNote = RetryNote()) -> str:
    evidence, driver_rel = brief.evidence, brief.driver_rel
    invocation_note, distributed_note = brief.invocation_note, brief.distributed_note
    retry = ""
    if retry_note.failure:
        retry = (
            f"\n## {retry_note.heading}\n"
            f"{retry_note.failure}\n"
            "Fix the driver so all preflight checks pass (correctness, benchmark, "
            "graph timing, and profiling contract). Do not stop until "
            f"`python {driver_rel}` prints a correctness metric AND timing.\n"
        )
    return f"""\
Prepare this KernelForge task so forge-loop can optimize it. Author (or repair)
the measurement driver at `{driver_rel}` so it satisfies the driver contract
below. A deterministic validator will execute the driver after each attempt and
return any failures for the next attempt, so get a complete draft on disk early
and let the validator tell you what is wrong — do not try to read your way to a
perfect first version.

{invocation_note}
{distributed_note}
Rules:
- The ONLY off-limits files are the kernel and the listed source files — NEVER
  edit them (they are what forge-loop optimizes). You MAY create or modify any
  OTHER file you need (the driver, small helper modules, etc.).
- Do not create symlinks or modify task metadata, invocation specifications, or
  files outside the driver staging directory.
- The driver must still run after this preparation ends. Only the driver, the
  helpers you write beside it, and the invocation specification beside it are
  durable; the reference bundle is deleted before your driver is validated. Read
  it now, never at runtime.
- Graph timing is REQUIRED: the benchmark MUST capture the op into a CUDA/HIP
  graph and REPLAY it once per timed iteration. The deterministic check counts
  actual `torch.cuda.CUDAGraph` replays, so a printed label does NOT count and
  eager timing is REJECTED. The simplest way to pass is to bench through the
  provided `graph_harness.py` (see the note below); make capture work — allocate
  inputs once, reuse one output buffer, launch on the current stream, and pass
  `dirty`/`verify` — rather than settling for eager timing.
- The `verify` callback confirms graph replay actually ran the kernel; it is NOT
  a full correctness check. Use `_snr_db(ref, out) > 30.0` (SNR-based), NOT
  `torch.allclose` — for FP8/quantized kernels, allclose can fail after graph
  replay even though the kernel computed correctly, causing a false fallback to
  eager timing.
- Keep the driver deterministic (fixed seed) and exit 0 on success.

{reference_note}

{evidence}

{DRIVER_CONTRACT_SPEC}
{retry}
When done, ensure `python {driver_rel}` runs the complete correctness suite and
prints an `SNR:`/`allclose:` line,
`python {driver_rel} --bench-mode --warmup 3 --iters 10` runs the complete
benchmark suite and prints `wall_ms:`/`median_ms:` plus `case_ms:` lines, and
`python {driver_rel} --profile-run` runs only the driver-selected profile case's
target kernel; all commands must exit 0.
"""


_SYSTEM_PROMPT = """\
You are a measurement-harness engineer for GPU kernel optimization. Your only job
is to make a task benchmarkable by forge-loop by writing a correct measurement
driver (plus any helper files it needs). You NEVER modify the kernel or source
files under optimization — only the measurement scaffolding. Time on the GPU
(graph timing strongly preferred); the caller runs deterministic verification
after every edit attempt. The finished driver MUST also expose the complete
profiling contract: benchmark mode prints `case_ms: <case_id> <ms>` for every
case the task declares, and `--profile-run` lets the driver select one
representative case and run only its target kernel without a reference
implementation or timing output.

The driver you leave behind is committed and then re-run unchanged for hours. It
may only depend at runtime on itself, the helpers you write beside it, and the
invocation specification beside it. Reference material you are given for
authoring is deleted before validation; a driver that reads it at runtime passes
the check and then crashes on the first real measurement.

Working order — you are on a wall clock, and the caller keeps whatever is on
disk when it expires:
1. Read the invocation spec and ONE reference driver. That is enough to start.
2. WRITE a complete first draft of the driver. Complete means both modes, every
   required output line, runnable end to end — not a sketch, not a TODO.
3. Only then run it, read more, and iterate on what actually failed.
An attempt that ends with the driver file unchanged is a total loss: nothing is
salvaged and the next attempt restarts from the same broken state, while an
attempt that merely ran out of time still gets validated and can be accepted.
Never spend a whole attempt reading. If time is running short, save the best
driver you have rather than nothing.
"""


# Embedded canonical assets (examples/ is not packaged in the wheel)

DRIVER_CONTRACT_SPEC = """\
## forge-loop driver contract (what the driver MUST satisfy)

forge-loop treats the driver as a black box run as `python driver.py <args>` and
reads it purely over stdout. The driver owns all case selection:

Correctness — `python driver.py`
must run the complete correctness suite and
prints (at least one):
    SNR: 62.13 dB          # preferred; forge pre-filters on this vs the SNR threshold
    allclose: True         # optional fallback
Benchmark — `python driver.py --warmup <n> --iters <n> --bench-mode`
must run the complete benchmark suite and
prints per-iteration:
    wall_ms: 0.081920      # one line per timed iteration (forge takes the median)
  or one aggregate line: `median_ms: <ms>` / `mean_ms: <ms>` (label it honestly).
It MUST additionally print `case_ms: <case_id> <ms>` for every case the task
declares. The case_id is a no-whitespace token; when the invocation
specification lists `tests.driver_contract.case_selectors`, use each entry's
`CASE_ID` verbatim. The deterministic check compares your `case_ms` ids against
that declared set, and a driver that reports only some of them is rejected.
Single-case sweep — `python driver.py --warmup <n> --iters <n> --bench-mode --bench-case <case_id>`
SHOULD benchmark that one case and print only its lines, exiting non-zero if the
id is not one it declares. This is what makes "hold the code, vary one dispatch
constant, time one shape" cost seconds instead of a full suite; a driver that
ignores the flag still measures correctly, it just makes every such question cost
the whole suite.
Profiling — `python driver.py --profile-run`
selects one representative case inside the driver and runs only its target
kernel (no reference/correctness path), performs
only enough warmup to settle JIT selection, launches 1-3 profiled iterations,
synchronizes, and exits 0 without printing timing data.

Rules:
- Case definitions come from the task's real harness/config; do not invent
  hard-coded dimensions in the driver.
- Runtime inputs must be DURABLE. The driver is committed and then re-run
  unchanged, many times, by the optimization loop. The only non-source files
  guaranteed to exist then are the driver itself, the helper modules you write
  beside it, and the invocation specification beside it. Everything else handed
  to you for authoring — the reference example bundle above all — is deleted
  before the driver is validated and committed. Reading any of it at runtime
  produces a driver that passes validation and then crashes on the loop's first
  measurement. Resolve durable paths relative to the driver's own directory
  (`Path(__file__).resolve().parent`), never relative to the process cwd.
- Deterministic inputs use a fixed seed.
- Exit 0 on success; a non-zero exit is treated as a crash.
- REQUIRED: the benchmark MUST actually run under a CUDA/HIP graph — it must
  capture the op into a graph and REPLAY it once per timed iteration. The prepass
  verifies this for real (it counts `torch.cuda.CUDAGraph` replays during the
  bench), so printing a label is NOT enough and eager timing is REJECTED. The
  simplest way to satisfy it is to bench through the provided graph_harness.

## Why the benchmark MUST run under a CUDA/HIP graph (required)
A small kernel's wall time is dominated by host-side launch/dispatch overhead, not
GPU work. Timing eagerly makes the optimizer chase host cost it cannot change,
makes iteration-to-iteration numbers noisy and incomparable (breaking keep/revert),
and does not match production (AMD serving runs these ops under a HIP graph). So
the produced driver MUST time under a graph: capture ONE invocation and time
replays, so CUDA events bracket only GPU execution. Allocate inputs once, launch
on the CURRENT stream (so capture records the kernel), and pass dirty/verify so an
empty/invalid graph is detected instead of reported as a fake speedup. The verify
callback checks that graph replay actually ran the kernel — use SNR-based
verification (`_snr_db(ref, out) > 30.0`), NOT `torch.allclose`, because FP8 and
quantized kernels can produce results that differ enough from the torch reference
to fail allclose yet are numerically correct. Verification is by actual replay
count (via torch.cuda.CUDAGraph), not a printed label — an eager run (or a silent
eager fallback) performs no replays and is rejected. Make the capture work
(allocate once, reuse the same output buffer, launch on the current stream) rather
than settling for eager timing.
"""


REFERENCE_DRIVER_TEMPLATE = r'''
"""Measurement driver — correctness (SNR) + graph-timed benchmark + profiling."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from graph_harness import cuda_graph_bench
# Import the kernel's STABLE public entry point (adapt this import + call):
from your_kernel_module import your_entry_point  # noqa: F401

_SEED = 0

# Load every scored case from the task's existing harness or configuration. Do
# not invent dimensions here. Keys are the case IDs used in case_ms lines (the
# invocation spec's CASE_ID values when the task declares them). Resolve any file
# you read here against _HERE, and only read files that outlive preparation.
_HERE = Path(__file__).resolve().parent
CASES = {}


def _make_inputs(dims, mode, device):
    torch.manual_seed(_SEED)
    x = torch.randn(dims["M"], dims["N"], device=device, dtype=torch.float16)
    if mode == "stability":
        x = x * 50.0
    return x


def _reference(x):
    # Replace with the operator's reference (e.g. torch.softmax(x, dim=-1)).
    raise NotImplementedError


def _snr_db(ref, test):
    ref = ref.float(); test = test.float()
    noise = test - ref
    sp = torch.mean(ref * ref).item(); npow = torch.mean(noise * noise).item()
    if npow <= 0:
        return 100.0
    if sp <= 0:
        return 0.0
    return 10.0 * math.log10(sp / npow)


def _run_correctness(device):
    snrs = []
    close = True
    for dims in CASES.values():
        x = _make_inputs(dims, "full", device)
        out = your_entry_point(x)
        ref = _reference(x)
        snrs.append(_snr_db(ref, out))
        close = close and torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"SNR: {min(snrs):.2f} dB")
    print(f"allclose: {close}")
    return 0


def _run_bench(dims, case_id, warmup, iters, device):
    x = _make_inputs(dims, "full", device)
    ref = _reference(x)
    out = torch.empty_like(x)
    step = lambda: your_entry_point(x, out)  # noqa: E731
    res = cuda_graph_bench(
        step, warmup=warmup, iters=iters,
        dirty=lambda: out.zero_(),
        verify=lambda: _snr_db(ref, out) > 30.0,
    )
    for t in res["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(res["times_ms"])
    print(f"case_ms: {case_id} {times[len(times) // 2]:.6f}")
    return 0


def _run_profile(dims, device):
    x = _make_inputs(dims, "full", device)
    for _ in range(3):
        your_entry_point(x)
    torch.cuda.synchronize()
    for _ in range(3):
        your_entry_point(x)
    torch.cuda.synchronize()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--bench-case", default="")
    p.add_argument("--profile-run", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    args, _ = p.parse_known_args()

    if not torch.cuda.is_available():
        print("error: no GPU"); return 1
    device = "cuda"

    if args.profile_run:
        profile_dims = next(iter(CASES.values()))
        return _run_profile(profile_dims, device)

    if args.bench_mode:
        selected = CASES
        if args.bench_case:
            if args.bench_case not in CASES:
                print(f"error: unknown case {args.bench_case}"); return 1
            selected = {args.bench_case: CASES[args.bench_case]}
        for case_id, case_dims in selected.items():
            _run_bench(case_dims, case_id, args.warmup, args.iters, device)
        return 0

    return _run_correctness(device)


if __name__ == "__main__":
    sys.exit(main())
'''
