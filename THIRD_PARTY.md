<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Third-party content

Hyperloom is MIT. A handful of files inside it are not AMD's work, and they
ship in the wheel because forge's knowledge base and its runnable examples are
packaged data. `REUSE.toml` carries the machine-readable version of this table
and `reuse lint` enforces it; this file records *why* each entry is there, which
the annotations cannot.

| Content | Origin | Licence | Why it ships |
|---|---|---|---|
| `src/kernelforge/data/local_knowledge/languages/flydsl/API_docs/examples/0{1,2,3,4}-*.py` | FlyDSL project | Apache-2.0 | Working reference kernels the agent reads when authoring FlyDSL. Four files. `04-preshuffle_gemm.py` carries no upstream header — see the note in `REUSE.toml`. |
| `src/kernelforge/data/examples/flydsl-softmax-forge-loop/softmax_kernel.py` | FlyDSL project | Apache-2.0 | Starting point of a runnable example campaign. |
| `src/kernelforge/data/examples/triton2flydsl-mxfp8-grouped-gemm/mxfp8_grouped_gemm.py` | SGLang (`kernels/ops/moe/mxfp8_moe_amd_gfx95.py`) | Apache-2.0 | The protected Triton oracle for a rewrite example. The pipeline reads it and never edits it; the FlyDSL port it produces is AMD's. |
| `src/kernelforge/data/serving_patches/sglang/**/*.patch` | AMD, against SGLang | `Apache-2.0 AND MIT` | Added lines are AMD's; the diff context and paths are SGLang's. Dual notice for that reason. |

## Named but not vendored

`languages/flydsl/API_docs/cute_layout_algebra_guide.md` describes the CuTe
layout algebra and cites CUTLASS (BSD-3-Clause) as its origin. The prose is
AMD's own and the guide embeds no CUTLASS source, so there is no BSD-3-Clause
file under `LICENSES/` and none is needed. If CUTLASS code is ever quoted into
that guide, add `LICENSES/BSD-3-Clause.txt` and an override annotation at the
same time.

## Adding to this list

Anything copied in from another project needs three things, together: an
`SPDX-FileCopyrightText`/`SPDX-License-Identifier` header or a `REUSE.toml`
override with `precedence = "override"` (`aggregate` would assert AMD copyright
over someone else's work), the licence text under `LICENSES/`, and a row here.
`reuse lint` runs in CI and will catch a missing licence file, but it cannot
catch the blanket `**` entry quietly claiming MIT over a file that is not MIT —
that part is on the reviewer.
