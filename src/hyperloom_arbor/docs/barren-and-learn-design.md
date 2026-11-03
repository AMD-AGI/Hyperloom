# The LEARN knob and the BARREN knob

> A design record (checked 2026-10-28: its constants match `arbor/state.py`; the map rounds came
> later, see [how-it-works.md](how-it-works.md)). Test counts below are from its date.

**Status: IMPLEMENTED 2026-09-25.** `arbor/state.py`, `arbor/pack/render.py`,
`arbor/pack/loader.py`, `arbor/policy/claude_loop.py`; tests in
`arbor/tests/test_barren.py` (17) with the full suite at 460 passing. Two rules
below were CHANGED during implementation because real campaign data contradicted
them — see §3 and §4.2, both marked **REVISED**.

Written while 2b/2c/2d/2e were live. 2b and 2c were cancelled on 2026-09-25;
2d and 2e run on unchanged deployed trees and are NOT affected by this change.
It ships to the new 2f/2g pair.

---

## 1. The problem, in one table

Reconstructing "iterations since the last booked advance on either axis" from the
four live sessions:

| run | iters | longest drought | current | `null_streak` reads |
|---|---|---|---|---|
| **2b** | 60 | **31** | **31** | **0** |
| 2c | 40 | 9 | 4 | 0 |
| 2d | 38 | 4 | 2 | 0 |
| 2e | 18 | 4 | 2 | 0 |

2b has gone **31 iterations — over half its campaign — without moving either
bar**, and the counter that exists to detect that reads zero. Its last 22 rows
before the drought deepened:

```
31 WIN   REVERT  13 findings     40 LEARN KEEP  8      47 FIX   REVERT  9
32 LEARN KEEP    12              41 LEARN KEEP  8      48 LEARN KEEP    7
33 LEARN KEEP    10              42 WIN   REVERT 10    49 LEARN KEEP    7
34 WIN   REVERT   7              43 LEARN KEEP  8      50 FIX   REVERT 11
35 LEARN KEEP     9              44 WIN   REVERT 13    51 FIX   REVERT  9
36 FIX   REVERT  11              45 LEARN KEEP 10      52 LEARN KEEP   12
```

A REVERT raises `null_streak` to 1. The next LEARN closes 7–13 real ledger
findings, which resets it to 0. It never reaches `ESCALATE_NULL_STREAK = 3`, so
escalation never fires, the LEARN mandate never fires, and **nothing at all is
rendered into the prompt**. The agent is not gaming anything — every one of those
LEARNs closed genuine findings.

### Root cause

`null_streak` is wired to three consumers that want different things:

1. escalation (`check_escalation`) — "is this tier dry?"
2. the LEARN mandate (`probe_mandated`) — "should we buy knowledge?"
3. a reset from LEARN-with-findings (`record_probe`) — "foundation work is progress"

Wire 3 was a deliberate and correct fix; `record_probe`'s docstring records that
without it the old dsv41flash campaign sat at `null_streak 7` with 30 unresolved
findings and never probed, "because every counter it could see treated probing as
failure." That fix converted an **absorbing** failure into an **oscillating**
one. Both are live failure modes and we now have one clean instance of each.

One counter cannot answer both "is this tier dry?" (which foundation work
legitimately answers) and "are we converting effort into wins?" (which it must
not).

---

## 2. The two knobs

### The LEARN knob — a rate limiter

- **Counts:** consecutive LEARN iterations (`probe_streak`, exists today).
- **Threshold:** `MAX_CONSECUTIVE_PROBES = 2`.
- **Acts on:** the next iteration. At the cap, `claude_loop.py:1108` demotes a
  declared LEARN to class B so it faces the throughput bar instead of skipping the
  server run.
- **Resets on:** any non-LEARN iteration.
- **Answers:** *has this campaign stopped shipping and started surveying?*

Fast, local, and it can only ever say **no**.

### The BARREN knob — a regime detector

- **Counts:** iterations since the last *booked* ratchet on either axis
  (`barren_streak`, **new**).
- **Thresholds:** `BARREN_NOTICE = 8`, `BARREN_ACT = 16`.
- **Acts on:** what the prompt asks for (§4), plus the LEARN budget (§5).
- **Resets on:** a real advance only.
- **Answers:** *is the campaign's picture of where the cost is still correct?*

Slow, global, and on its own it owns no lever.

### How they complement

Each is blind exactly where the other sees. The LEARN knob cannot distinguish
productive exploration from stalling — it caps both at 2, identically. The BARREN
knob can tell the difference but cannot act. So **BARREN sets the LEARN budget.**
That one wire is the entire coupling.

| | **not barren** | **barren** |
|---|---|---|
| **LEARN available** | normal — explore freely, it's cheap | drought — explore *more*, budget widens |
| **LEARN exhausted** | correct: exploitation is working, go ship | **today's bug** — forced to guess while blind |

The bottom-right cell is not hypothetical: 2b at iteration 53 (`probe_streak 2`,
29 iterations barren) and 2e at iteration 16. **The system rations exploration
hardest at the moment exploitation has proven least productive.**

### Why this is not circular

If a drought grants more LEARNs and LEARNs counted as progress, the drought would
feed itself. It does not, because **LEARN resets `null_streak` but never
`barren_streak`**. A campaign may spend all four granted LEARNs while the drought
counter keeps climbing. Exploration is made *cheap*, never *self-justifying* —
only a booked ratchet clears a drought.

---

## 3. `barren_streak` — exact rules

Add one field to `CampaignState`, defaulting to `0` so an old `session.json` loads
unchanged and starts counting from where it resumes.

| iteration outcome | effect on `barren_streak` |
|---|---|
| KEEP that ratcheted `best_metric` or `best_secondary` | **reset to 0** |
| KEEP recorded `frontier=True` | **reset to 0** (see §8, open question) |
| KEEP that booked nothing (`advanced_metric=False`) **and set no new high-water** | `+= 1` |
| REVERT | `+= 1` |
| LEARN → KEEP, any number of findings | `+= 1` |
| `FIX-CORRECTNESS`, any verdict | **unchanged** (exempt) |
| NOT_MEASURED | **unchanged** (frozen, as every other counter is) |

`FIX-CORRECTNESS` is exempt because a correctness fix that costs throughput is
still the right move, and counting it as drought would pressure the agent away
from fixing bugs — half of what this design is for. No per-drought cap on the
exemption: the intent is rare in practice (v2 used it 3 times in 70 iterations,
2d once in 38) and a cap is complexity we have no evidence we need.

**REVISED during implementation — the ratchet is authoritative, not
`advanced_metric`.** As first written, rows 1 and 3 of that table conflict, and
they conflict on a *real* row: **2d iteration 32 shipped `FIX → KEEP` with
`booked=False` and still moved both bars** (291.44 over 290.31, 53.43 over
52.93), because `_record_keep` ratchets whatever it is handed regardless of the
gate's verdict. Treating that as drought would have the prompt assert "no advance
in 11 iterations" while the monitor showed a bar moving 7 iterations earlier. The
drought is a statement about **the bars**; if a bar moved, it is over. So the
reset condition is `frontier or _would_ratchet(...)`, with `advanced_metric`
ignored. A neutral KEEP normally reads below the incumbent and still counts as
drought — that is the common case row 3 was written for.

`null_streak` keeps its current semantics and its current LEARN reset, untouched.

**ADDED 2026-10-20: a new map round also resets it** (`arbor/directions.py`,
`CampaignState.apply_map_round`).
- **The rule:** a new map round sets `barren_streak`, `null_streak` and
  `probe_streak` to 0.
  - That breaks this section's principle that only a moved bar ends a drought.
  - It's deliberate: once every direction is closed and a dossier update round
    has re-profiled and re-ranked, the campaign continues on a new map
    "like new" (operator).
- **While the map is written or updated,** the drought blocks and the LEARN cap
  are off, because that round is LEARNs by design.
- **The backstop:** a drought can't hide behind repeated resets. Two map rounds
  in a row that end without a KEEP stop the campaign for the operator, as does a
  new map with no open increment.

---

## 4. The prompt blocks

These render in `render_program`, in the same slot and the same voice as the
existing `secondary_bar_suspect` block — which is the model to follow: *state the
statistical inference, make the condition visible, name the cheap experiment that
settles it, never auto-correct.*

Placeholders in `{braces}` are filled from state; everything else is literal.

### 4.1 BARREN NOTICE — renders when `8 <= barren_streak < 16`

```
## No booked advance in {n} iterations

`{metric_name}` has sat at {best_metric} and `{secondary_name}` at
{best_secondary} since iteration {k}. Those {n} iterations were
{intent_mix} — they were not idle, and they closed {f} ledger findings
between them, but none of them cleared a bar.

Read that as information, not as a verdict: the cheap moves adjacent to
the current tree are spent, and the next advance is probably not adjacent
to the last one.

Three things this campaign has available and has not used during this
drought:

**{j} open findings nothing in this drought has cited:**
{finding_list}

**{r} changes REVERTed before iteration {k}.** A change that lost against
the incumbent of {k} is a different experiment against this one:
{revert_list}

**Your LEARN budget is {b} consecutive this iteration, not {default}.** A
LEARN touching only `eval/` and the ledger skips the server run entirely —
minutes instead of ~45. While this drought lasts, knowledge is the
cheapest thing on the menu.

You are NOT required to change direction. Going deeper on a seam that is
already paying is frequently the right move. But say which it is: name
the one finding or the one seam the next advance comes from, and name
what would tell you it is the wrong one.
```

Design notes on this text:

- It offers, it does not scold. The block exists to widen the option set at the
  moment the agent's own search has narrowed.
- The closing paragraph deliberately refuses to mandate breadth. **2d's entire
  lead came from going deeper on one seam it had already touched** (iterations 22,
  24, 28, 32 are all the same prefill gather). A forced-diversity rule would have
  pulled 2d off the exact thing that was working. Breadth is the right prior when
  you do not know where the cost is; it is the wrong rule once you do.
- The re-land list is the highest-yield move in the corpus and nothing currently
  asks for it: 2c re-landed previously-REVERTed work at iterations 15, 17, 19 and
  24 — organically, unprompted — and it cleared a bar every time.

### 4.2 BARREN ACT — renders when `barren_streak >= 16`

```
## THE FRAME IS SUSPECT — {n} iterations, no advance

{n} consecutive iterations produced no booked advance on either axis.
Across that many UNRELATED changes, competently executed and measured,
that is better explained by this campaign optimizing components that are
not the cost than by {n} independent failures.

This is the same inference the harness already makes about a suspect
secondary bar, applied one level up. When a long run of unrelated changes
all read null, the thing they have in common is not the changes. It is
the DENOMINATOR.

Every one of those {n} iterations assumed a particular answer to "where
does the time go". {denominator_provenance}

The cheap experiment that settles it: **price the WHOLE RUN.** Not another
component evaluator — an instrument that accounts for wall-clock end to
end, and that is capable of telling you the component you have been
tuning is a small share of it. Build it to BRACKET rather than to agree:
two independent estimators that should disagree with each other if your
model of the run is wrong. Build it so it CAN come back and say no.

Declare `INTENT: LEARN`. State the decision rule BEFORE you see the
number: which split would send you somewhere else entirely, and where
would you go.
```

**REVISED during implementation.** `{denominator_provenance}` was specified as one
of two factual claims:

- ~~`"Nothing in this campaign has measured it."`~~
- ~~`"The last whole-run measurement was iteration {x}, {d} iterations and {c} landed changes ago."`~~

Rendering the first against the real 2b session produced a **flat falsehood** —
2b holds roughly 20 denominator/census ledger entries — and a block the agent can
catch lying is a block it discounts, which costs more than the sentence buys.
Detecting the second reliably means classifying an arbitrary instrument, which is
fiddly and can be wrong in both directions. Shipped instead as a question, which
is always true and strictly more actionable, since after N landed changes
staleness is the live issue anyway:

> `Name the measurement that answer rests on, and when it was last taken against the CURRENT tree.`

One further change to §4.1: the re-land list filters out `LEARN → REVERT` rows. A
probe that closed no finding is not a change sitting on the shelf, and the first
draft listed six of them for 2b, pointing at nothing.

Design notes on this text:

- It names **no component**. This matters: the same leak-check that governs the
  2e pack constraints governs this block. It must never say prefill, gather,
  `NUM_WORKERS`, or any other hint at a known answer, in any campaign.
- It is a deliberate **promotion of 2e's pack constraint to harness level**. 2e is
  currently testing, via three generic pack constraints, whether an agent can be
  steered to author a whole-run instrument without being told where the win is.
  Signal (a) says yes — by iteration 18 it had authored
  `eval_decode_step_denominator.py`, `eval_decode_prologue_census.py` and
  `eval_prologue_gap_attribution.py`. This block asks for the same thing, but
  *only when a measured condition says it is warranted* rather than in every
  campaign from iteration 1. If 2e ends up ON PACE or better, that is the
  evidence this block is worth rendering at all.
- **One mandate, and it is the cheapest one available.** Exploration without a
  re-measured denominator is random search, and random search over a ~100-component
  surface will not find a gather kernel in 16 iterations. v2 and 2d both reached
  the winning seam by measurement, not by breadth.

---

## 5. The LEARN budget widening

The one wire between the knobs:

```
effective_max_consecutive_probes =
    LEARN_BUDGET_BARREN  if barren_streak >= BARREN_NOTICE  else  MAX_CONSECUTIVE_PROBES
```

with `MAX_CONSECUTIVE_PROBES = 2` (today's value, unchanged) and
`LEARN_BUDGET_BARREN = 4`.

Read by both `probes_exhausted()` and the demotion check at
`claude_loop.py:1108`, so the cap and its enforcement stay consistent.

Nearly free in wall-clock: a LEARN touching only `eval/` and the ledger skips the
server run (~2 min vs ~45). This is a single integer and it inverts the current
perverse rationing.

Mandated LEARNs — `probe_mandated`, `secondary_bar_suspect`, BARREN ACT — should
not consume budget at all. A harness that mandates a LEARN and then refuses it is
incoherent.

---

## 6. Cleanup of what is already there

Not new features; debt this change should not step over.

1. **Eight integers into one `Policy` dataclass.** `CONFIG_MAX_ITERS`,
   `ESCALATE_NULL_STREAK`, `MAX_CONSECUTIVE_PROBES`, `PROBE_MANDATE_NULL_STREAK`,
   `MAX_CONSECUTIVE_INVALID`, `SECONDARY_SUSPECT_STREAK`, plus the two new BARREN
   thresholds. **Five of the six existing ones are the integer `3`** and mean five
   unrelated things; none is per-pack configurable. Defaults = today's values,
   overridable from `pack.yaml`, **persisted into `session.json`** so a resume
   reproduces its regime and the monitor can show which regime a run was under.
   That also makes A/B-ing the *harness* possible, the way 2c/2d currently A/B the
   pack.

2. **Drop the tier half of `probe_mandated`.** It requires
   `at_terminal_tier AND null_streak >= 3`. All four live packs have
   `space.levels: None`, so the tier half rarely fires and the mandate is mostly
   dead code. Keep the streak half.

3. **Delete `CampaignState.branches`.** Declared, serialized into every
   `session.json`, written by nothing. All four runs: `0`. Either it becomes a real
   backtrack lever or it goes; it should not stay as a field that looks like a
   feature. See §8.

4. **Add `regime` to each lesson row** and a `state.diagnose()` line in the prompt
   header. The reason 2b's drought was invisible is that the monitor faithfully
   reported `null_streak=2`; both the agent and the monitor should see the ladder
   stage.

### Explicitly out of scope

The gate — `threshold_rel` 1.01, `noise_band_pct` 1.0,
`secondary_noise_band_pct` 1.7, `DEFAULT_CORRECTNESS_CAP_PCT` 5.0, per-intent
`books`/`requires_threshold`, the frontier no-ratchet rule. That is what counts as
a win, not pace, and it is the part that is behaving. Do not touch it.

Also out of scope: `HISTORY_WINDOW`, `DEFAULT_COST_MIN`, the hard stops
(`time_budget_minutes`, `--max-iters`).

---

## 7. Thresholds and calibration

`BARREN_NOTICE = 8`, `BARREN_ACT = 16`, `LEARN_BUDGET_BARREN = 4`.

**Verified by replaying all four real sessions through the implemented state
machine** (not hand-calculated):

| run | final `barren_streak` | stage | NOTICE first at | ACT first at | last iter |
|---|---|---|---|---|---|
| 2d | 7 | clear | — | — | 39 |
| 2e | 0 | clear | — | — | 18 |
| 2c | 0 | clear | iter 13 | — | 38 |
| 2b | 35 | **act** | iter 32 | **iter 40** | 59 |

The two healthy runs never fire. 2c fires NOTICE once during its 9-iteration
drought and then recovers on its own, ending clear — which is the block working
as intended, or the drought being self-limiting; one run cannot distinguish
those. 2b reaches ACT at iteration 40 with **19 iterations of budget still
unspent**.

Fit to four runs. Revisit with more.

---

## 8. Open questions

1. **Does `frontier=True` reset the drought?** Above says yes — it is a genuine
   measured Pareto advance. But it ratchets no bar, so a campaign could in
   principle trade back and forth along the frontier indefinitely without ever
   tripping the ladder. v2 had 1 frontier point in 70 iterations and 2c has 3, so
   this has never bitten. If it ever does, the fix is to require that a frontier
   point also be *new* — not a return to a previously-visited point.

2. **Is there a missing backtrack lever?** Right now the campaign has exactly one
   incumbent tree and only ever moves forward from it. The textbook escape from a
   barren state is to restart from an earlier tree, and `branches` is the field for
   it, unimplemented. Left out of this design on purpose — it is a much larger
   change than one counter — but it is the one genuinely absent capability, and
   §6.3 deletes the field rather than leaving a stub that implies otherwise.

3. **Are 8 and 16 right?** Fit to four campaigns. In particular, 2c's 9-iteration
   drought resolving on its own is weak evidence that 8 is slightly early.

---

## 9. Summary of the change

- **+1 counter** (`barren_streak`)
- **+3 constants** (`BARREN_NOTICE`, `BARREN_ACT`, `LEARN_BUDGET_BARREN`)
- **+2 render blocks** (§4.1, §4.2)
- **+3 computed lists** for the NOTICE block (uncited open findings, pre-drought
  REVERTs, untouched components vs. phase-budget share where available)
- **−1 condition** (`at_terminal_tier` in `probe_mandated`)
- **−1 dead field** (`branches`)
- **8 constants relocated** into a per-pack-overridable, persisted `Policy`

No schema break. `null_streak` semantics unchanged. The gate untouched.
