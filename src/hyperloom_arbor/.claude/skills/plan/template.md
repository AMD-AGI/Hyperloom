# <Model> <Workload> on <GPU>: <Subject> (round <N> plan)

> What this round takes over from (the dossier, earlier rounds and their
> settlements) and what it leaves to others.

Status: **draft**, written <date> by <campaign or person>; not yet read by the operator

<!--
The skeleton of a round plan (the plan skill, SKILL.md). Keep the section
numbers: iterations, reviews and settlements cite them, and work items are
cited by their number (4.1, 4.2) in DONE lines. Delete every comment and
replace every <placeholder>.
-->

## 1. Objective

<!-- One paragraph: what will be true when this round succeeds, at which
operating point (concurrency, input and output lengths), measured how. -->

## 2. Evidence

<!-- What says this round is worth running, each claim with a source a reader
can open: the dossier's parts, ledger ids, iterations, profiles, job ids. -->

### 2.1 The workload this targets

<!-- From the dossier's parts 5 and 7: what the targeted code sees at the
operating point (rows per step, context, step kinds), the share of step time
the targeted costs have, and what in the benchmark is synthetic. -->

### 2.2 The expected gain against the noise floor

<!-- Per work item: its largest gain in the metric's unit (the Amdahl ceiling
from the dossier's parts 7 and 11), beside the measured floor (part 6). An item
whose best case is below the floor is set aside here, with the number, or
changes the measurement. -->

## 3. Fixed baseline

<!-- The pins (framework, kernel library, image digest), the benchmark and its
seed, the server flags that may not change, and what the round may change. -->

## 4. Work items

### 4.1 <Work item>

<!-- One item per direction, in the order they run, the cheapest decisive one
first. -->

**What**: <the change or measurement, naming the kernel or code path the evidence shows runs>

**Produces**: <the artifact: a patch, an evaluator, a profile, a record>

**Expected**: <the gain in the metric's unit and its source, against the floor in 2.2; or "enabling: <the item it unblocks>">

**Done when**: <a check someone else can run, with its threshold>

**Sources**: <dossier part, ledger id, iteration, profile>

### 4.2 Correctness

**What**: <what must stay identical, and when it runs: before any timed run>

**Done when**: <the check and its threshold>

## 5. Gates

<!-- Declared before any run, each with its threshold. -->

- **Correctness.** <check and threshold>
- **Performance.** <metric, estimator, threshold against the floor>

## 6. Stop conditions

<!-- Conditions under which the round ends without a candidate, each
checkable. -->

- <condition>

## 7. Artifacts

<!-- What is kept and where; what stays node-local. -->

## 8. Execution setup

### 8.1 Campaign, node and trees

### 8.2 Existing work to reuse

### 8.3 The report

### 8.4 Hazards

## 9. Settlement

Not settled: written after the round, per work item (the plan skill, "After
the round").
