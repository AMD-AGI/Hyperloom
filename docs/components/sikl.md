---
myst:
    html_meta:
        "description": "Learn about SIKL, AMD's kernel library served in front of AITER, and how Hyperloom optimizes the kernel of one SIKL schema with KernelForge."
        "keywords": "SIKL, AITER, GPU kernel, schema, workload, KernelForge, Hyperloom, ROCm, FlyDSL, Triton, AMD GPU"
---
# SIKL

SIKL is AMD's library of GPU kernels served in front of AITER. Each kernel is one *schema*
file: an op contract narrowed to one scenario (its dtypes, attributes and constant sizes),
the workloads the kernel serves, and the kernel's sources. A schema with no sources states
a problem: the op and shapes a kernel is wanted for. SIKL serves a kernel workload by
workload, where its measured speedup over AITER reaches `SIKL_MIN_SPEEDUP` (1.03 by
default) and its output is correct; every other workload keeps running AITER.

- **Source**: <https://github.com/AMD-AGI/SIKL>
- **License**: MIT

## Role in Hyperloom

`hyperloom optimize` starts from a model and a serving configuration and finds its own
kernels to optimize. A SIKL schema is the other direction: the problem is given, and the
shapes that count are enumerated up front. `hyperloom sikl` is the entry point for that
case. It materializes a forge-loop run from one schema and dispatches KernelForge over it.

SIKL is not part of Hyperloom's default install. The command needs a SIKL checkout on the
host, named with `--sikl-root`. The schema is either a kernel-less schema in that checkout,
named with `--schema`, or a schema file from elsewhere, named with `--schema-file` and read
with the checkout's own contracts.

```bash
hyperloom sikl \
  --sikl-root /path/to/SIKL \
  --schema gemm_n6016_k7168 \
  --kernel-backend flydsl --gpu-target gfx950 --max-hours 4
```

The loop starts from a kernel SIKL passes on every workload. For a gemm-contract schema the
command writes one in torch; for any other op, pass one with `--starting-kernel`. Before the
first commit the command has SIKL check it on the GPU, and stops if any workload fails.

`--kernel-backend` names the language a candidate is written in: `flydsl`, `triton` or
`gluon`. Every GPU kernel `run` launches has to be one `kernel.py` defines with that
language's kernel decorator. Once SIKL passes every workload, the driver profiles one call
of `run` per workload and fails each workload that launched anything else — a vendor
library such as hipBLASLt, an aiter kernel, or a torch kernel. Without that check the
fastest move on a GEMM schema is to call a library per scored shape, which beats AITER
without being a kernel in the declared language. The starting kernel is exempt only while
it is unchanged.

The run directory holds the loop's workspace, a copy of the schema, SIKL's validation
reports, the handoff documents, and `summary.md` — the delivered kernel's per-workload
result, including the workloads it loses, the ones SIKL would serve, and the ones that
could not be scored. Give each run its own directory: the command refuses one that already
holds files, because the starting commit takes the whole workspace and a leftover file
would become part of what every candidate is measured from.

## Who owns the measurement

Correctness and timing come from `python -m sikl validate`, the check SIKL's CI runs: its
audit of the source, its float64 reference and error bounds, and its graph-replay timing
against AITER in the same window. The generated driver publishes `kernel.py` as the
schema's kernel, reports what `validate` returns and measures nothing of its own, so the
numbers the loop optimizes against are produced by the same code that decides whether
SIKL serves the kernel. A driver that re-implemented the measurement would eventually
disagree with it, and a candidate tuned against that disagreement is tuned against nothing.

One `validate` run checks and times a kernel, which takes longer than forge-loop allows a
single benchmark, so the driver runs it once per kernel and every mode reads that report.
The published file leaves the checkout when the run ends: another kernel's file left
there would be served beside the next one.

## SIKL documentation

For the schema format, the kernel ABI and the audit rules a kernel must pass, see
`sikl/ops/README.md` in the SIKL repository.
