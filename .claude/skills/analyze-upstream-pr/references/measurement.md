<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Measurement plan

Consulted from Step 8 of [`../SKILL.md`](../SKILL.md) when the recommendation hinges on a number,
or the user asks for one. `measure.md` is a plan the user runs in their own environment; the skill
never starts a server, a benchmark or a GPU job. Each part below is a heading in the plan.

## Control and change

Two arms that differ by the PR alone: the same tree or image at `commit`, model, weights, flags,
environment and workload. The change arm has the PR's diff applied, or the tree moved past its
merge commit. Anything else that moves between the arms (an image rebuilt on another day, a
dependency that floats, a different node) is a second change, and the result cannot be assigned
to the PR.

## Noise first

Run the control arm twice. The gap between the two runs is the noise floor: an effect inside it is
no effect, however it reads.

## Correctness before speed

When the diff touches numerics, weights, kernels or sampling, run an accuracy check on a fixed
evaluation set for both arms before any timing. A faster arm with different outputs is a
regression, not a win.

## Operating point

The target's model, batch size or concurrency and sequence lengths, from `target.txt`. Add the
author's operating point from `operating_point.txt` when a `GAP` line leaves the sign of the effect
open, so a difference between the two is visible as such.

## Metric and decision rule

Step 5's metric, the number of repetitions, and the rule that turns the numbers into a verdict,
all written before the first run. A rule chosen after reading the numbers fits the numbers.

## Path proof

A profile, a trace or a log line showing the change arm ran the changed code. A run that never
reached it measures nothing, and says nothing about Step 3.

## Results

The user's numbers go under this heading with their source (where and when they ran), labeled
`measured`. Steps 9 and 10 are then rerun with them.
