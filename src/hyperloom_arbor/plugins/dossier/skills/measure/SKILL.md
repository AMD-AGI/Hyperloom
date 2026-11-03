---
name: measure
description: "Write the dossier's measured parts on the target node: 6 the baseline and noise floor, 7 the profile (plan, then results, with the profiler skill), 8 kernel usage (what ran, why, and what could run instead), and 12 the fast bench, which does not block. Then complete parts 1 and 3 where they wait on the profile and the kernels. Run by Arbor in the map.measure chapter, or by hand when these parts are missing or stale (after a KEEP, or a change of flags or parallelism)."
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: map.measure
    after: ["dossier:discover"]
    produces: ["{dossier}/06-baseline.md", "{dossier}/07-profile.md", "{dossier}/08-kernels.md"]
    reuse_key: [identity, framework]
    gpu: true
---

# Measure: the dossier's measured parts

**In a campaign, Arbor runs this skill in a turn of its own.** Its argument is the path of a brief: read the brief
first. It names the dossier's directory, the files to write, and the campaign's program.
**By hand,** the directory is `docs/dossier/<model>-<workload>-<gpu>/`.

**What it needs:**
- a node, for parts 6 and 7, and the `profiler` skill, for part 7;
- the rules in [the rules](../../shared/rules.md), relative to this skill's directory. Above all: the profile is taken
  while the dossier is written, and a part 7 with a plan and no results leaves the dossier unwritten.

**After parts 7 and 8:**
- complete part 1's kernel column and part 3's kernel-to-method mapping, which `dossier:discover` left `partial`;
- update the index's rows.

**End your reply** with one `LEDGER:` line per part written, a FACT naming its file:
`LEDGER: {"id":"dossier-07","type":"FACT","claim":"part 07 written: <the top shares>","evidence":"<file>"}`.

## The parts

### 6. Baseline and noise floor

- **Answers:** what the system does today, and how much it moves when nothing
  changes.
- **Contains:** the pins (framework, kernel library, image digest, harness);
  the node and its health (foreign processes, containers, a connect census);
  the metric at each operating point; the noise floor from repeated control
  runs on the node the campaign will use, with the same estimator as a
  treatment; the validity gate (for AgentX, the duration band).
- **Done when:** the floor is measured on the campaign's node, in the unit
  part 11 compares gains against.

### 7. Profile: plan, then results

Use the `profiler` skill (TraceLens, rocprofv3).

- **Plan, written before anything is captured:** the question each capture
  answers; the mode (the profiler skill's route, and whether it is validated
  here); windows on the client's phase clock, per regime, with the running
  count inside each; the cells of rows and context to cover, from part 5;
  what else must be idle; where raw traces stay.
- **Results:** step time by component per regime and step kind, from the
  stratified tables; collectives by transfer; overlap and host gaps
  (host-side numbers from untraced runs only); outliers, and whether the
  tracer caused them; the checks and what each left open; what the capture
  could not see.
- **Done when:** a capture of the served tree exists (a plan alone is not
  done); the shares add up to the measured step time within a stated
  residual; every share carries its window, regime, mode and validation
  status; and the profiler's checks have run, or each one not run says why.
- **Stale when:** the tree changes (a KEEP) or the deployment does (flags,
  parallelism).

### 8. Kernel usage: what runs, and what could run instead

The procedure is [kernels.md](kernels.md).

- **Answers:** which kernels serve each op at the served shapes, why those
  were chosen, and whether better ones exist: unused in the libraries already
  shipped, documented in AITER and KernelForge, or picked by the reference
  platform.
- **Contains:** one row per op family: the kernels that ran (symbol,
  provider, calls and share per step, shapes, step kinds); the dispatch
  decision (call site, flag, tuned-table hit or fallback); alternatives in the
  codebase that do not run at these shapes, why not, and how to try each;
  KernelForge's pages; what B200 runs for the same op; the gfx950 gap that may
  block a port; the lead, with its ceiling from part 7. The serving playbook's
  `knobs.json` says whether a vendor knob changes numerics, depends on the
  operating point, or belongs to a bundle.
  The `kernels` skill holds the kernel notes (AITER, FlyDSL, CK, HIP, Triton)
  and the examples a "how to try it" column draws on.
- **Done when:** every kernel with at least 1% of stratified GPU time in any
  regime has a row, with its dispatch decision and its alternatives checked,
  or "none found" and where the search looked.


### 12. Fast bench (does not block the campaign)

Whether a cheaper bench can screen changes, and how often it disagrees with
the long one: candidates tied to part 5 (a kernel bench at the served shapes,
a step bench at the weighted cells, a compressed replay, a shortened run,
each with what it cannot see); ground truths of known effect (an A/A pair, a
change the long bench already judged, injected delays proven on the device);
per candidate, its cost, its own floor, sign agreement with the long bench and
false positives. Until a candidate is validated, its readings may order work
but never keep or reject a change; validated, it screens, and a claim still
needs the long bench. The runner's Plan 21 is the worked example.

## Done when

- **Each part you wrote meets its own "Done when",** above, or is marked partial or not available, with what would fill
  it.
- **Its file holds real content:** `python3 -m arbor.dossier.check <dossier dir> --json` lists it `ok`. That means at
  least 300 words and 10 numbers outside headings, template labels and placeholders. Parts the other dossier skills
  write may fail until they run.
- **Every number has its source and label.**
- **The index lists each part you wrote** with its status, sources, date and staleness key, and the open asks.
