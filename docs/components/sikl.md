---
myst:
    html_meta:
        "description": "Learn about SIKL, AMD's kernel definition and solution library, and how Hyperloom optimizes one of its definitions with KernelForge."
        "keywords": "SIKL, AITER, GPU kernel, definition, workload, KernelForge, Hyperloom, ROCm, Triton, AMD GPU"
---
# SIKL

SIKL is AMD's library of GPU kernel *definitions* and the solutions submitted against
them. A definition states a problem: the operator, which axes are fixed and which vary,
the inputs and outputs, and the callbacks that decide whether an answer is correct —
`initialize`, `reference` and `compare`. A solution is one implementation measured over
every workload the definition declares.

- **Source**: <https://github.com/AMD-AGI/SIKL>
- **License**: MIT

## Role in Hyperloom

`hyperloom optimize` starts from a model and a serving configuration and finds its own
kernels to optimize. A SIKL definition is the other direction: the problem is given, and
the shapes that count are enumerated up front. `hyperloom sikl` is the entry point for
that case. It materializes a forge-loop run from one definition and dispatches
KernelForge over it.

SIKL is not part of Hyperloom's default install. The command needs a SIKL checkout on
the host, named with `--sikl-root`, and an exported schema collection (a *bundle*),
named with `--bundle`.

```bash
hyperloom sikl \
  --bundle /path/to/bundle \
  --definition gemm_a16w16_nt_n1536_k7168 \
  --sikl-root /path/to/SIKL \
  --kernel-backend flydsl --gpu-target gfx950 --max-hours 4
```

The harness times only the workloads that pass correctness, and it checks correctness under
graph capture, so the loop's starting kernel has to pass every workload there. Before the
first commit the command runs one correctness pass on the GPU over the definition's
reference and keeps it if it passes; otherwise it tries the baseline. A reference that
syncs with the host (a data-dependent branch, `.tolist()`) cannot run under capture; a
baseline that accumulates in lower precision can miss the reference's tolerance. If
neither passes, the command stops before dispatching KernelForge.

`--kernel-backend` names the language a candidate is written in: `flydsl`, `triton` or
`gluon`. Every GPU kernel `run` launches has to be one `kernel.py` defines with that
language's kernel decorator. Once the harness passes every workload, the driver profiles
one call of `run` per workload and fails each case that launched anything else — a
vendor library such as hipBLASLt, an aiter kernel, or a torch kernel. Without that check
the fastest move on a GEMM definition is to pick a library solution per scored shape,
which beats the baseline without being a kernel in the declared language. The starting
kernel is exempt only while it is the definition's reference or baseline byte for byte.

The run directory holds the loop's workspace, a private copy of the bundle, the handoff
documents, and `summary.md` — the delivered kernel's per-case result, including the
workloads it loses and the ones that could not be scored. Give each run its own
directory: the command refuses one that already holds files, because the starting
commit takes the whole workspace and a leftover file would become part of what every
candidate is measured from.

## Who owns the measurement

Numerical correctness and timing come from `sikl.benchmarks.api`, which applies the
definition's own callbacks. The generated driver reports what that harness returns and
measures nothing of its own, so the numbers the loop optimizes against are produced by the same
code SIKL's acceptance gate runs. A driver that re-implemented the measurement would
eventually disagree with the gate, and a candidate tuned against that disagreement is
tuned against nothing.

The loop's measurements time the candidate only. Timing the baseline beside it doubles a
measurement, which on definitions with many workloads runs past forge-loop's limit on a
single measurement. The final score times the baseline in the same window as the
delivered kernel.

Read `summary.md` rather than SIKL's acceptance rule. That rule asks for one workload
above 1.02x, which a kernel with a geometric mean below 1.0 can still satisfy.

## SIKL documentation

For the definition schema, the solution format and the acceptance gates, see the
`harness/` directory of the SIKL repository.
