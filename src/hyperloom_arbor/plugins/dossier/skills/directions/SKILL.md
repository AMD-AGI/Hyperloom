---
name: directions
description: "Write the dossier's part 11, its directions: what to try, in what order, and what each could give at most, increments and kernel rewrites ranked apart (a kernel plan per rewrite), from parts 1 to 10 only, ending in the map Arbor reads. Run by Arbor in the map.directions chapter, last; or by hand when the directions must be ranked again."
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: map.directions
    produces: ["{dossier}/11-directions.md"]
    reuse_key: [identity, framework]
    gpu: false
---

# Directions: the dossier's part 11, and the map

**In a campaign, Arbor runs this skill in a turn of its own,** after the other parts. Its argument is the path of a
brief: read the brief first. It names the dossier's directory, the file to write, and the campaign's program.
**By hand,** the directory is `docs/dossier/<model>-<workload>-<gpu>/`.

**Before ranking:**
- Read [the rules](../../shared/rules.md), relative to this skill's directory.
- Read parts 1 to 10 (`dossier:read`). Without part 7's results, the whole part is a draft and says so at its top.
- Write each rewrite's kernel plan from [kernel-plan.md](kernel-plan.md).

**End your reply with the map,** as ONE line, exactly the `## Map` block's: `DIRECTIONS: [...]`. Arbor sets the
campaign's map from it, and a dossier pass ends there. Check it first:
`python3 -m arbor.directions check <list.json>`. Then add a `LEDGER:` FACT naming part 11's file.

### 11. Directions

- **Answers:** what to try, in what order, and what each could give at most.
- **Contains:** per direction: the component and kernel it targets (parts 7,
  8); its largest gain in the target's unit (part 4) beside the floor (part
  6); what it depends on, what it costs, what would stop it; whether it was
  tried before (part 10) or the reference platform shows it (part 9); the
  rules of part 4's table it touches, and how it stays within each, or
  "none". A direction that only pays by breaking a rule is not a direction:
  list it with the rule, and set it aside.
  Directions whose best case is below the floor are listed, with the number,
  and set aside. Ranked by expected gain over the floor, then by cost.
- **From:** parts 1 to 10 only, ranked by the ceilings part 7 measured. A
  direction that needs a missing fact names the missing part; without part 7
  results the whole part is a draft and says so at its top.
- **Two kinds of direction, ranked apart.**
  - **An increment** changes one thing that a campaign iteration can build and
    measure: a flag, a table row, a dispatch rule, a fusion of two adjacent
    launches. Its ceiling is what its component costs now.
  - **A rewrite** rebuilds a kernel or a fused region: a decode MoE from route
    sort to finalize, an attention block from projection to output norm, a
    layer as one persistent kernel. It takes many iterations before anything
    can be served. Its ceiling is the region's time now minus its floor at the
    served shapes (bytes over bandwidth, FLOPs over peak, the launches it must
    keep), both from part 7.
  - Part 11 lists rewrites in their own table, each with a kernel plan
    ([kernel-plan.md](kernel-plan.md)), even while increments lead.
- **When rewrites go first: the increments are running out.** Watch for these
  signs. They're calibrated on Model M campaigns B4 and C4 (MI350X, 0.5% gate,
  2026-10-13); revisit the numbers after the first rewrite campaign.
  - **The KEEPs are small:** the median of the last five KEEPs is under 3×
    the gate. B4's was 1.27%, C4's 1.39%.
  - **The probes kill their own leads:** five or more probes in a row each
    closed the lead it set out to build, at or below the gate. That was B4's
    iterations 42–46, and C4's 38–43, the last of which "kills the
    campaign's last big build".
  - **The board is thin:** no increment left has a measured ceiling of 3× the
    gate, or what's left is launch-bound or already at its roofline. B4's
    mixed-step all-gather already ran at the XGMI roofline; 86% of C4's
    mixed-step idle time was a dependency stall.
  - **Arbor's `barren_streak` has reached its notice (8).** Any KEEP resets it,
    so on its own it misses a campaign that still keeps a gate-sized change
    now and then. The first three signs don't.

  Two of these signs make the case. A rewrite still needs something to win,
  though: part 7 must show a region whose time now is well above its floor,
  several percent of the step on paper. Without one, the dossier says so: the
  remaining gap is in the system (collectives, scheduling, the host), not in
  a kernel, or the target is met.
- **Then the operator decides; the dossier informs.** An agent that sees two
  signs says so in its verdict and carries on. The usual next step:
  - an update round (`dossier:update`) whose 11.uK ranks rewrites first, with a kernel
    plan for each of the top ones;
  - then a campaign spent on one rewrite:
    - its config sets `"space": {"levels": ["kernel"]}`;
    - its pack raises `policy.max_consecutive_probes`, so build stages that
      can't serve yet aren't cut off;
    - each stage is judged by the plan's gate for it, until the last stage,
      where the campaign's own A/B gate decides.
- **Check structural directions before ranking.** A campaign that works one
  iteration at a time rarely reaches large structural changes on its own
  (fused MoE and mega kernels, persistent kernels, collective fusion, layout
  changes), so part 11 has to bring them in.
  - **For each one whose preconditions this workload meets** at the target's
    operating point (parts 4 and 5): part 11 carries it, usually as a rewrite
    with a kernel plan, or lists it as set aside with the reason. Reasons: the
    image lacks it, the operating point doesn't favour it, it is already on,
    or its best case is below the floor.
  - **Price it on this node,** from part 7 and the hardware skill's measured
    table, never from a gain reported elsewhere.
- **The map: what the campaign tracks.** Every direction has a map ID,
  `m-<component>`, which stays the same across update rounds; a direction
  that changes so much it needs a new ID says which one it replaces.
  - **The `## Map` block.** The part ends with one, holding the exact line an
    Arbor campaign reads (`arbor/directions.py`):
    `DIRECTIONS: [{"id": "m-…", "kind": "increment|rewrite", "title": "…", "rank": 1, "ceiling": "…", "status": "open"}, …]`
  - **Statuses:** open, done, killed, below_floor, or plan (a rewrite whose
    kernel plan is written). Every closed direction carries a `why`.
  - **What a parallel run needs** (optional; a map without them is read as
    before):
    - `lane`: increment, kernel-plan, architectural or upstream. By default
      an increment is an increment lane and a rewrite a kernel-plan lane.
    - `plan`: the kernel plan, relative to the dossier. A kernel-plan lane
      needs one, with its `## Items` block (`kernel-plan.md` §5).
    - `depends_on`: the map IDs it builds on, from the same list. Cycles are
      refused; a closed dependency counts as met.
    - `nodes`: the nodes its lane holds at once (default 1).
    - `upstream`: `{"repo": …, "pr": …, "commit": …, "image": …}`, which an
      upstream lane (a port of work done elsewhere) needs.
    - `reopens`: `{"round": K, "why": …}`, for a direction an earlier round
      closed and this one brings back on evidence: 10.uK marks the kill
      reopenable, or the new tree removes the reason it died.
    - `evidence` on a close: `{"iterations": [N], "ledger": [ids],
      "artifacts": [paths]}`. A lane needs it, and the harness checks it.
  - **What the campaign does with it:** it keeps the map as its checklist and
    closes directions with `DIRECTION:` lines. When every one is closed, it
    runs this part again (the `dossier:update` skill, "In the campaign, by the harness").
  - **Check the line before it goes in:** `python3 -m arbor.directions check`.
    With `--dossier <this dossier>` it also checks each lane's plan and
    prints the order the dependencies allow.
- **Done when:** each direction can become a plan whose evidence section cites
  the dossier, each rewrite has a kernel plan, every big direction from the
  playbook that applies is carried or set aside with its reason, and the
  `## Map` line passes `arbor.directions check`.

## Done when

- **Each part you wrote meets its own "Done when",** above, or is marked partial or not available, with what would fill
  it.
- **Its file holds real content:** `python3 -m arbor.dossier.check <dossier dir> --json` lists it `ok`. That means at
  least 300 words and 10 numbers outside headings, template labels and placeholders. Parts the other dossier skills
  write may fail until they run.
- **Every number has its source and label.**
- **The index lists each part you wrote** with its status, sources, date and staleness key, and the open asks.
