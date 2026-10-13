<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Risk lenses

Consulted from Step 7 of [`../SKILL.md`](../SKILL.md). Each lens is one line in `risks.txt`. An
upstream PR is reviewed and measured on its author's platform; each lens is a way a change that is
right there turns out wrong, slower or inert on a ROCm deployment. A lens names where to look,
what makes it a `RISK`, and when it must stay `CLEAR` even though the diff touches its area.

## L1 — Semantics

**Look at** what each changed function returns, writes or raises for the inputs it already
accepted: shapes, dtypes, memory layout, output order, the defaults of new parameters, the errors
it now raises or swallows.

**RISK** when a caller on the target's path gets a different value for an input it already
passes, and no test in `testfiles.txt` pins the old one.

**CLEAR** when the change is additive: a new path behind a condition the target does not meet, or
a new parameter whose default keeps the old behaviour. Name which.

## L2 — Premise

**Look at** the problem the body says the PR fixes and the measurement it rests on, then the
comments and `linked.txt` after it.

**RISK** when the premise belongs to the author's platform (a bottleneck that only exists there, a
library version or a bug the target's tree never had), or when a later comment, a `reverted-by` or
a `superseded-by` row says it no longer holds.

**CLEAR** when the premise does not depend on the platform, or the target shows the same symptom.

## L3 — Gate or default change

**Look at** new or changed environment variables, config fields, server flags, platform, arch and
dtype checks, backend priority lists, and their defaults.

**RISK** when a default flips on a path the target runs (another backend chosen, a feature turned
on), or when the new path is off unless a flag is set that the PR documents only for its own
platform: merged and included, it never runs.

**CLEAR** when no default changes, or the changed default's condition excludes the target (cite
it).

## L4 — Regression on other paths

**Look at** every caller of the changed code beyond the path the PR measured: other backends,
dtypes and model families, prefill against decode, eager against graph capture, speculative
decoding, chunked prefill.

**RISK** when a path the target runs reaches the changed code and the PR shows no test or number
for it.

**CLEAR** when a search of the tree at `commit` finds no caller on the target's path other than
the one measured.

## L5 — Parity with the CUDA path

**Look at** changes made in a CUDA-only file, under a CUDA guard, or in a kernel or class that has
a ROCm twin: a HIP kernel, a Triton kernel with ROCm tuning, a kernel-library op, a ROCm backend or
platform class.

**RISK** when the CUDA side changes behaviour or an interface that the twin also implements and the
diff leaves the twin alone: the ROCm path keeps the old bug, diverges in output, or breaks at
import or dispatch.

**CLEAR** when the twin changes in the same diff, or the changed code has no ROCm counterpart (cite
where the ROCm path goes instead).

## L6 — Weights and quantization

**Look at** weight loading, layout transforms and shuffles done at load, scale handling,
quantization methods and the checkpoint formats each accepts.

**RISK** when a load-time layout or scale change reaches the quantization method the target's
checkpoint selects, above all where the ROCm path preshuffles or re-lays out weights for its own
kernels and the PR does not know it.

**CLEAR** when the changed method is not the one the checkpoint selects (cite the selection).

## L7 — Numeric bounds

**Look at** accumulation dtypes, epsilons, clamps, scale ranges, FP8 formats, block sizes, and the
integer width of offsets and indices.

**RISK** when the change assumes a range or encoding that differs on the target arch (FP8 comes in
variants whose maximum value and NaN encoding differ, and the arch decides which one runs), or
widens a shape so that an offset can overflow.

**CLEAR** when the dtype and range are the same on every arch, or the diff handles the variant
(cite where).

## L8 — Collectives and parallelism

**Look at** all-reduce, all-gather, reduce-scatter and all-to-all calls, custom communication
kernels, process groups, collectives under graph capture, and the tensor, data, expert and
pipeline parallel sizes the change assumes.

**RISK** when it changes which communication backend runs, the count or order of collectives per
step, or assumes a topology, an interconnect, a library feature or a world size the target lacks.

**CLEAR** when no collective or group changes, or the change is backend-neutral and the target's
parallel layout exercises it unchanged.

## L9 — Tests, and whether ROCm CI runs them

**Look at** `testfiles.txt`, `ci.txt`, and the skip conditions and markers on the new tests.

**RISK** when the new tests skip on ROCm or are marked for one platform, or `ci.txt` shows no ROCm
job ran at the head: the PR's green says nothing about the target.

**CLEAR** when a ROCm job in `ci.txt` ran the changed tests at the head and passed. Name the job.

## L10 — Blast radius

**Look at** `numstat.txt`, and how many call sites and model families reach the changed code on
the target: once at load, once per request, or on every forward pass.

**RISK** when a small diff sits on a hot path every request crosses, or a large one spreads over
subsystems the target uses. Size it by the callers on the target's path, not by lines.

**CLEAR** when the target reaches the changed code through one isolated call site that Step 8 can
exercise, or not at all.

## L11 — Upstream fit

**Look at** `meta.txt` (state, mergeable), the `draft` line of `inclusion.txt`, the reviews in
`comments_untrusted.txt` and their authors' roles, `commits.txt`, `linked.txt`, and
`conflicts.txt`.

**RISK** when maintainers asked for another design, the PR is a draft or has gone quiet, it
conflicts with an open PR likely to land first, or it closed unmerged: what you carry is not what
upstream will land, and the next rebase meets a different change.

**CLEAR** when a maintainer approved with no change request open, or the PR merged and no merged
`reverted-by` row undid it.
