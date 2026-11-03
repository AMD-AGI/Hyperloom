---
name: study
description: "Study the benchmark a campaign is judged on, once, for every campaign on it: the rules a change could break (what it fixes, forbids and requires, each with its source and who checks it, as rules.yaml), its reference configuration and published number and how to measure it on the session's node, and what to measure (the official metric and units, the secondary metrics, the operating points, how many runs a claim needs). Writes benchmarks/<name>/. Run by Arbor in the benchmark chapter, before the dossier, or by hand when a benchmark is new or its rules change."
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: benchmark
    produces: ["{benchmark}/README.md", "{benchmark}/rules.yaml", "{benchmark}/reference.md", "{benchmark}/metric.md"]
    reuse_key: [benchmark]
    gpu: false
---

# Study the benchmark

A campaign that does not know its benchmark keeps the wrong changes, and they are expensive. Each of these cost a campaign
its result:
- **Under the benchmark's correct rules,** a tuned config ran slower than the reference.
- **Natural speculative acceptance** looked like a free P90 gain, where the benchmark fixes golden acceptance.
- **A campaign scored output throughput** while the chart plots P90 interactivity, and its one adopted change
  lowered P90.

This skill writes down, once per benchmark, what every campaign on it must respect and measure. The dossier's part 4
copies the rules in, and the gates check them.

**In a campaign, Arbor runs this skill in a turn of its own,** before the dossier. Its argument is the path of a brief:
read the brief first. It names the benchmark's directory (`benchmarks/<name>/`) and the files to write.
**By hand,** the directory is `benchmarks/<benchmark>-<workload>/`, for example `benchmarks/inferencex-agentx/`.

**Look first.** When the directory exists, read its `README.md`, then refresh only what is stale: a new revision of the
benchmark's rules, a new reference recipe, or a new metric. A benchmark's study belongs to everyone who runs it, and
its owners review changes to it.

## What to read

- **The benchmark's own documents, at a pinned revision:** its README and rules, its fairness or submission guidelines,
  its validation scripts, its PR checklist. For InferenceX: the recipe scripts (`benchmarks/single_node/...`), the
  AgentX trace and client settings, `perf-changelog.yaml`, and the golden acceptance curve for speculative decoding.
- **The host's rules,** when a host runs the campaign: the config's directives that start `Host rule`, copied
  verbatim.
- **What campaigns on it learned:** [contract.md](contract.md), beside this file.
  - **Its source:** earlier serving campaigns, and what failed in them.
  - **What to do with them:** each says what it guards against. Check whether it applies to this benchmark.

## What to write

### `rules.yaml`: the rules a change could break

One entry per rule, in the form a gate can check:

```yaml
benchmark: inferencex-agentx
revision: <the documents' commit or date>
rules:
  - id: golden-acceptance
    kind: requires                 # fixes | forbids | requires
    what: speculative decoding's acceptance follows the benchmark's golden curve, checked on the live server
    setting: SGLANG_SIMULATE_ACC_LEN   # the flag, variable or knob it is about, when there is one
    value: "<from the golden curve at the configured draft depth>"
    source: "<document, section, revision>"
    checked_by: host-gate          # host-gate | arbor-gate | validator | nobody
```

- **`fixes`:** the workload, dataset, ISL/OSL, concurrency, request count, seed, client, duration, chat template.
- **`forbids`:** a flag, a variable or a technique.
- **`requires`:** a precision, golden acceptance, an accuracy floor, or a valid run.
- **`checked_by: nobody`:** says so plainly. It is the gap the next gate closes.
- **Check the file:** `python3 -m arbor.benchmarks check <dir>`.

### `reference.md`: what a campaign is judged against

- **The configuration:** the benchmark's reference for this model and GPU, with its image (by digest), flags and
  environment.
- **Its number:** the published one, with its source and date.
- **How to measure it on the session's node,** under the rules above: the exact commands. A campaign measures it first
  and states its gain over it, not only over its own start (#16).
- **The same for the reference platform** (B200), when one is published: a lead, never evidence for an AMD claim.

### `metric.md`: what to measure

- **The official metric:** its units, its estimator, and what the chart plots. Score that, with a cap on the other
  axis.
- **The secondary metrics** the benchmark reports.
- **The operating points:** concurrencies, ISL/OSL, and the regime at each when known.
- **How many runs a claim needs:** the noise first (three replays at least), and bands set from it. Where an axis is
  noisy, the primary moves to the stable one.
- **The run's validity:** a non-zero client exit, or a window under 90% of the duration, is not a run.
- **The rungs:** a component eval, then a screen, then a confirm at the benchmark's full duration, outside the loop.
- **What the benchmark cannot show:** a synthetic acceptance rate, say. And what a campaign must check besides
  throughput: accuracy, and real acceptance at the confirm.

### `README.md`: the index

- **The header:** the benchmark, its documents and revision, who reviewed the study, and its date.
- **One line per file,** with its status: written, partial or not available.
- **Open asks:** what only the benchmark's owners can answer.

## Done when

- `python3 -m arbor.benchmarks check <dir>` passes: the four files exist, and `rules.yaml` parses, with every rule
  carrying its kind, source and checker.
- Every rule a change could break is in `rules.yaml`, with its source. The ones nobody checks say so.
- The reference has a number with its source, and the commands to measure it here.
- A reader can class a made-up result as kept or rejected, from `metric.md` and `rules.yaml` alone.

**End your reply** with one `LEDGER:` line per file written, a FACT naming it:
`LEDGER: {"id":"benchmark-rules","type":"FACT","claim":"<n> rules, <m> checked by nobody","evidence":"<file>"}`.
