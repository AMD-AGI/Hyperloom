---
name: draft
description: "Draft a new Arbor pack and its campaign config for a person to review: start from the serving template (packs/serving, configs/serving-template.json) or the nearest existing pack; take the objective, its bands and the constraints from the benchmark's study (benchmarks/<name>/); keep the model's facts in its dossier, not the pack; write the skills in the lifecycle form (plugins:, lifecycle:); check it with a dry run; and end with a review note of what was copied, chosen and left open. Use when a new model, framework, GPU, benchmark or operating point needs a pack, or when asked to derive one pack from another."
metadata:
  arbor:
    modes: [use]
---

# Draft a pack

A pack is the contract every campaign of one kind shares: the objective and its bands, the protected validators, the
workload, the policy, the constraints the agent reads, and where skills take part (`docs/packs.md`). A new kind of
campaign gets a new pack. Today that means copying the nearest one by hand, and it shows: 30 `pack.yaml` files, and
the DSV4.1 line alone has 12+ variants of about 820 lines that differ in hundreds. This skill drafts one instead.

**It drafts; a person decides.** It writes the draft, checks it and explains it. It does not run a campaign, commit, or
change another pack.

**By hand:** load the family with `claude --plugin-dir plugins/pack` from the Arbor tree, then ask for a pack, or run
`/pack:draft <what the campaign is>`.

## 1. Settle what the campaign is

Ask for what is missing, in one message, then go on with what you have:
- **the model:** its path or Hugging Face id, and its precision;
- **the framework:** vLLM or SGLang, and the image;
- **the GPU:** MI355X, MI350X, ..., and the TP;
- **the benchmark and workload:** InferenceX random 1k/1k, an AgentX trace replay, a document campaign;
- **the operating point:** concurrency, ISL/OSL, seed;
- **standalone, or under a host:** Hyperloom's MB, for example (docs/hosts.md).

The pack's name says what it is for, not a version: `agentx-qwen38`, `serving-sglang-1k1k`. A variant of an existing
pack takes that pack's name and a suffix that says what differs (`-c8`, `-tp4`), never a bare `-2`.

## 2. Pick the parent

Start from the nearest pack, not from nothing: [parents.md](parents.md) says which, by workload. The default is the
serving template, `packs/serving`: `balanced`'s method with nothing model-specific in it.

**Copy the parent whole, then change it:**

```bash
cp -r packs/<parent> packs/<name>          # pack.yaml, knowledge/, validator/, eval/
rm -rf packs/<name>/results                # another campaign's numbers are not this one's evidence
```

**What stays with the parent's text:**
- the validators, unless the benchmark's study says the bench or the correctness check differs;
- the standing directives and the harness rules.

Say which parent, and why, in the review note.

## 3. Take what the benchmark decides from its study

Read `benchmarks/<benchmark>/` (the `benchmark` plugin's `study` writes it). If it is missing, say so, and either run
`/benchmark:study` first or leave the objective as the parent's, marked for review.

| The pack's | From the study |
| --- | --- |
| `objective.metric`, `direction`, `unit` | `metric.md`: the official metric, what the chart plots |
| `objective.measurement`: `threshold_rel`, `noise_band_pct`, `secondary_metric` and its band | `metric.md`: the noise, the bands, the second axis |
| `constraints` on what is fixed, forbidden or required | `rules.yaml`: one line per rule a change could break, with its source and who checks it |
| the config's operating point (`concurrency`, `num_prompts`, `ISL`, `OSL`, `BENCH_SEED`) | `metric.md` and `rules.yaml`'s `fixes` |

**Never invent a number.**
- A band comes from a measured noise floor, or stays the parent's, marked for review.
- An accuracy floor (`GSM8K_MIN`) comes from the model's measured reference score, less about three standard errors
  of the subset. Without one, leave the template's 0.30 sanity floor and say so.

## 4. Keep the model in its dossier

The pack should hold nothing a different model of the same kind would not share. The model's layers, quantization,
code paths and directions belong to its dossier (`docs/dossier/<name>/`; `/dossier:read` finds it).
- **Knowledge the parent carries about another model:** move it out or drop it; never keep it.
- **A fact the campaign needs on its first iteration:** cite the dossier part that holds it, instead of copying it.

## 5. Write where its skills take part

Use the lifecycle form (`docs/packs.md`, "Where skills take part"; `docs/skills.md`), not `skills:`:

```yaml
plugins: [benchmark, dossier]
lifecycle:
  benchmark:      {run: ["benchmark:study"]}
  map.discover:   {run: ["dossier:discover"]}
  map.measure:    {run: ["dossier:measure"], use: [profiler]}
  map.directions: {run: ["dossier:directions"]}
  iterate:        {use: ["dossier:read", profiler]}
  round_end:      {run: ["dossier:update"], use: [profiler]}
```

- **Extra skills for the iteration:** add `hardware`, `serving-playbook` and `kernels` to `iterate`'s `use` list when
  the campaign may write kernels.
- **A contributed family:** add it only when the person asks.
- **Under a host:** add `component.lifecycle` for what the host owns. An empty `benchmark` chapter, for example, when
  the host's rules stand in for the study.

## 6. Write the config

Start from `configs/serving-template.json`, or the parent's config:
- **The campaign:** `identity` (workload:shape:gpu:framework:precision:model:conc:tp), `goal` (one sentence: what
  rises, at which operating point, without losing what), `pack`, `dossier` and `benchmark` (the names under
  `docs/dossier/` and `benchmarks/`).
- **The serving:** `model_path`, `model_name`, `framework`, `tp`, `gpu_ids`, `server_flags` and `server_env`. Copy
  them from the benchmark's reference recipe (`reference.md`): the arms must start from what the benchmark measures.
- **Under a host:** the host writes `docker_container`, `session_root`, the scripts and the component block. Draft
  only what the host takes from the pack.

## 7. Check it

```bash
python3 -m arbor.skillcheck                                # the skills the pack names exist and check
python3 -m arbor run --config configs/<...>/<name>.json --dry-run   # validate_pack, the config, the program it renders
```

Fix what fails. A dry run that passes says the pack loads. It does not say the pack is right: that is the person's
review.

## 8. End with the review note

In your reply:
- **What:** the pack's name and path, the config's path, and the parent.
- **The diff against the parent:** `diff -ru packs/<parent> packs/<name>`, summarized by section.
- **What you chose,** each with its source: the study, the dossier, the person, or the parent's text kept.
- **What is left for the person:** every `<...>` value, every band or floor marked for review, the open asks.
- **What you did not do:** no campaign run, no commit, no other pack changed.
