# Re-review

Read from Step 1 of [`../SKILL.md`](../SKILL.md) when `comments.txt` holds an earlier card for this PR: a
`## PR #<n>` heading in any review or comment it carries. [`../check-card.sh`](../check-card.sh) matches that heading
alone, so an author filter here would skip `prior_verdicts.txt` and the gate would withhold the verdict. Settle that card's
findings first, then review the whole diff anyway: an earlier card, the PR author's own included, can have missed
anything.

## Order of work

1. **The latest card's findings first.** The latest card is the one posted last (each `comments.txt` entry names its
   time); its `Head:` is the head it reviewed. Decide each finding at this head, one line each into
   `$WORK/prior_verdicts.txt`:

   ```
   <KEY> FIXED|HOLDS|MOVED -- <path:line, and what you read there>
   ```

   `FIXED` cites the fixing line, `HOLDS` the line that still has it, `MOVED` the same defect elsewhere. A `HOLDS` or
   `MOVED` finding returns to the card under its earlier key — not new, not renamed — its `Problem:` opening with
   `held since <sha>.`, the head of the first card that raised it. A finding the author answered with a reason in
   `comments.txt` holds only where this head shows the reason wrong; cite where. When the latest card had no
   finding, the file is the one line `none -- the latest card reported no finding`.
2. **Then the whole diff, as a first review.** Steps 1b-7 as written; Step 1b adds the rules of the `HOLDS` and
   `MOVED` findings.
3. **Code an earlier card already passed.** A new finding on a line the latest card's head already had was in front
   of that card too (read the file at that head with
   `gh api -H 'Accept: application/vnd.github.raw' 'repos/<repo>/contents/<path>?ref=<sha>'`). Its summary closes
   with `(missed at <sha>)`: the miss is named, not presented as new.
4. **A class that comes back.** A blocking finding of the class a card raised on an earlier head — the author changed
   the code and the class survived — has an `Action:` naming the shared cause and asking the author to reconsider
   the approach: one site patched per push is the loop this breaks.

## A human's open objection

`comments.txt` carries more than earlier cards. A blocking objection a human reviewer raised on this PR, which the
author has neither answered in code nor the reviewer withdrawn, is settled exactly like a card's finding: it goes in
`prior_verdicts.txt` under the key `human:<slug>`, and a `HOLDS` there cannot sit beside `Blocking issues: none`. A
card reporting no finding while an open human objection stands on the same page is not a second opinion but a review
that did not read the page: in PR #2020 one was published two hours after a CHANGES_REQUESTED whose "+361 production
lines against *Simplify by removing a mechanism*" was still unanswered, and the PR landed at net -134.

Withdrawn means the reviewer said so. An author's reply that disagrees is no withdrawal, and neither is a later push
that touches the file without addressing the point.

## The same head

When the latest card's `Head:` is this head, this review is a second, full reading: every new finding carries
`(missed at <sha>)`, and `Blocking issues: none` cannot stand beside a `HOLDS` line.

## On the card

`Checked` carries `Prior:`, `prior_verdicts.txt` in order as `<KEY> fixed|holds|moved`, comma-separated (for example
`Prior: C1 fixed, T2 holds`), or `Prior: none` when the latest card had no finding; a PR with no earlier card writes
no `Prior`. A `FIXED` finding leaves the card, and `Prior` is where the author reads that it was fixed.
[`../check-card.sh`](../check-card.sh) refuses a card whose `Prior` and `prior_verdicts.txt` disagree.
