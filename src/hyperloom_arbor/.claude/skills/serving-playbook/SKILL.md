---
name: serving-playbook
description: What 1,258 InferenceX serving changes teach an optimizing agent — 22 proven big directions (MegaMoE, persistent kernels, collective fusion, two-batch overlap, layouts, disaggregation, KV tiers, speculative policy) with their preconditions and AMD status; the settings that win per concurrency band, per model, GPU and engine; every AMD knob (vLLM-ROCm, SGLang, ATOM, AITER, quick-reduce, MoRI, RCCL, HSA) and what it needs; the knobs that change numerics; and which image first carried each capability or fix. Use when ranking directions in a dossier, before a knob search, when a change shows no effect, or when a knob may change accuracy.
---

# Serving playbook: what other stacks already learned

InferenceX tunes vLLM, SGLang, ATOM and TRT-LLM recipes for AMD and NVIDIA GPUs at every
concurrency band. Its history holds the big wins that an agent changing one thing at a time rarely
reaches. Our study read all 1,258 of its changes one by one and explains why
agents miss them. These files are that knowledge, organized for a campaign agent.

**Built 2026-10-20** from InferenceX `origin/main` at `6d6d296c6` (2026-08-30) and the study's records.
Facts about images expire. A recipe that changed after that date is not reflected here.

## Which file, when

| You are | Read |
| --- | --- |
| Writing or updating dossier part 11: which large changes apply here | `big-directions.md`: start with its quick-lookup table at the end, then the entries that apply |
| Choosing a setting for an operating point (TP/EP/DP, MTP depth, TBO, chunking, caps) | `regime-tables.md`, the model and GPU nearest yours; §8 has the cross-model rules |
| About to set or search an AMD env var or flag | `amd-knobs.md`: §0 first, then the layer you touch |
| Changing anything that might alter outputs (quantization, KV dtype, collectives, speculative decoding) | `numerics-knobs.md`: §1 has the golden scores on record |
| A change showed no effect, a feature crashes, or a flag looks stale | `image-manifest.md`: the "blocked before" table, the default changes and silent no-ops, the workaround inventory |
| Pricing a direction against what the hardware can do | the `hardware` skill (`.claude/skills/hardware/SKILL.md`), not these files |

## Evidence

- **`#655`** is a record id of that study, one per InferenceX change. Its records are not carried in this copy;
  the id names the change for whoever has them.
- **`main:<path>`** is a file on InferenceX `origin/main` at the commit above.
- **"Missed by 2.x"** names the mechanism the study found kept agents away from a
  direction.
- **Disagreements** between records and recipes are flagged where they occur (↔ in the regime
  tables, "unsure" notes elsewhere). Treat a flagged fact as a lead to check, not a fact.

## Rules that hold across the files

- **Price every direction on this node.** A gain these files report was measured on another model,
  GPU, image or operating point.
  - Use part 7 of the dossier and the hardware skill's measured table to price it here.
  - NVIDIA numbers never transfer as numbers, only as mechanisms. MI300X/MI325X (gfx942) facts are
    not facts about MI350X/MI355X (gfx950).
- **Prove the arm differed.** Many recipe changes were silent no-ops: a renamed env var, a size gate
  that kept quick-reduce off, a launcher that dropped a flag.
  - Check the kernel names in a trace, or the server's own log of what it selected, per arm.
  - "No effect" is a finding only after that check.
- **Never kill a policy on one operating point.** Most big settings flip sign with concurrency,
  sequence length or layout (`regime-tables.md`).
  - Before a DEAD_END, measure on the other side of the stated crossover.
  - Record the image digest, concurrency and lengths with the kill.
- **"Blocked by image X" is not a dead end.** A feature that fails in the pinned image may work in
  a newer one (`image-manifest.md` §A7). Record the blocker and the version that fixes it.
- **Numerics changes need the accuracy gate before a KEEP** (`numerics-knobs.md`). A throughput
  gate alone has kept broken changes.
- **Re-test workarounds on every image change.** Removing a stale one is a candidate optimization
  (`image-manifest.md` §W).
- **Bundles:** some wins are a set of settings whose members fail or sit under the gate alone
  (`big-directions.md` §0).
  - Apply the documented set in one iteration, cite the entry, and prove each member took effect.
- **Layout changes** (TP width, replicas, DP attention, EP) are judged per GPU, at more than one
  band. Most fixed-TP packs cannot run them: say so in the dossier rather than dropping them.

## How this reaches a campaign

- **The dossier** (part 11) checks every big direction whose preconditions this workload meets.
  Each one is either carried, usually as a rewrite with a kernel plan, or set aside with its
  reason. See the dossier skill.
- **A knob search** reads `amd-knobs.md` for what each knob does, what it needs and whether it
  changes numerics, before trying it. Arbor's always-on knowledge (`arbor/knowledge/mi355x-optimization.md`)
  points here.
- **These files are reference.** Read the entry you need; they are never pasted into a prompt
  whole.
- **`knobs.json` is their machine-readable index, for the harness.** Every knob in `amd-knobs.md`
  §1–§9 and `numerics-knobs.md` (env or flag, NUM, OP), and `amd-knobs.md` §0's bundles with
  their members. Arbor reads it: a declared `BUNDLE:` is gated as one change, a change to a NUM
  knob calls for the accuracy gate, and a dead end on an OP knob is marked as holding at one
  operating point. Edit the md files first, then the index; `arbor/tests/test_knobs_json.py`
  checks that every name in it is quoted from them.
