---
name: serving-playbook
description: The rules that hold for serving changes on AMD GPUs (price on this node, prove each arm differed, judge a policy at more than one operating point, numerics changes need the accuracy gate, bundles go in as one change) and knobs.json, the knob catalog Arbor's harness reads. Use before a knob search, when a change shows no effect, or when a change may alter outputs.
---

# Serving playbook: rules and the knob catalog

This copy of the playbook holds its rules and its knob catalog. The upstream skill's reference pages (big
directions, regime tables, the AMD knob reference, numerics knobs, the image manifest) are not carried here. A file
that names one of them keeps the name; the page is not in this copy.

## The knob catalog: `knobs.json`

The harness reads it (`arbor/knobs.py`):

- **`knobs`:** every AMD env var or server flag it records, with its framework, and two marks:
  - **`numerics`:** a change to this knob can alter outputs, so it calls for the accuracy gate before a KEEP;
  - **`op_dependent`:** its effect flips with the operating point, so a dead end on it holds at that point only.
- **`bundles`:** documented sets of settings that pay only together. A declared `BUNDLE:` is gated as one change.
- **`source`:** the reference page and section the entry was taken from, and record ids (`#655`) of the study behind
  it. Neither is carried in this copy; they name the origin for whoever has them.

## Rules that hold for every serving change

- **Price every direction on this node.** A gain reported elsewhere was measured on another model, GPU, image or
  operating point: it is a lead, never this node's number. Use the dossier's part 7 and the `hardware` skill's
  measured table to price it here.
- **Prove the arm differed.** Many settings are silent no-ops: a renamed env var, a size gate that keeps a path off,
  a launcher that drops a flag.
  - Check the kernel names in a trace, or the server's own log of what it selected, per arm.
  - "No effect" is a finding only after that check.
- **Never kill a policy on one operating point.** Many settings flip sign with concurrency, sequence length or
  layout (`op_dependent` in the catalog).
  - Before a DEAD_END, measure on the other side of the expected crossover.
  - Record the image digest, concurrency and lengths with the kill.
- **"Blocked by the image" is not a dead end.** A feature that fails in the pinned image may work in a newer one.
  Record the blocker and the version that fixes it.
- **Numerics changes need the accuracy gate before a KEEP** (`numerics` in the catalog). A throughput gate alone has
  kept broken changes.
- **Re-test workarounds on every image change.** Removing a stale one is a candidate optimization.
- **Bundles:** some wins are a set of settings whose members fail or sit under the gate alone.
  - Apply the catalogued set in one iteration, name the bundle, and prove each member took effect.
- **Layout changes** (TP width, replicas, DP attention, EP) are judged per GPU, at more than one concurrency. Most
  fixed-TP packs cannot run them: say so in the dossier rather than dropping them.

## How this reaches a campaign

- **The dossier** weighs structural changes whose preconditions this workload meets, each carried or set aside with
  its reason. See the dossier skill.
- **A knob search** checks the catalog before trying a knob: whether it changes numerics, whether it depends on the
  operating point, and whether it belongs to a bundle.
- **This file is reference.** Read the part you need; it is never pasted into a prompt whole.
