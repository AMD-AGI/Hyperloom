# Re-review

Consulted from Step 1 of [`../SKILL.md`](../SKILL.md) whenever `comments.txt` holds an earlier card for
this PR: a `## PR #<n>` heading in any review or comment `comments.txt` carries.
`pr-review-publish.sh` matches that heading alone, so an author filter here would skip
`prior_verdicts.txt` and the gate would withhold the verdict. This review settles
that card's findings before its own, and still reviews the whole diff: an earlier card can have missed
anything, the PR author's own included.

## Order of work

1. **The latest card's findings first.** The latest card is the one posted last (each entry's header
   in `comments.txt` names its time); its `Head:` is the head it reviewed. Decide each of its findings
   at this head, one line each into `$WORK/prior_verdicts.txt`:

   ```
   <KEY> FIXED|HOLDS|MOVED -- <path:line, and what you read there>
   ```

   `FIXED` cites the line that fixes it, `HOLDS` the line that still has it, `MOVED` the same defect
   at another place. A `HOLDS` or `MOVED` finding goes back on the card under its earlier key, its
   `Problem:` opening with `held since <sha>.`, `<sha>` being the head of the first card that raised
   it: it is not new, and it is not renamed. A finding the author answered in `comments.txt` with a
   reason holds only where this head shows the reason wrong; cite where. When the latest card reported
   no finding, the file holds the one line `none -- the latest card reported no finding`.
2. **Then the whole diff, as a first review does.** Steps 1b-7 run as written; Step 1b adds the rules
   of the `HOLDS` and `MOVED` findings.
3. **Code an earlier card already passed.** A new finding on a line the latest card's head already had
   was in front of that card too; read the file at that head with
   `gh api -H 'Accept: application/vnd.github.raw' 'repos/<repo>/contents/<path>?ref=<sha>'`. Its
   summary closes with `(missed at <sha>)`: the miss is named rather than presented as new.
4. **A class that comes back.** When a blocking finding is the same class as one a card raised on an
   earlier head, so that the author changed the code in between and the class survived, its `Action:`
   names the cause they share and asks the author to reconsider the approach: patching one site per
   push is the loop this rule exists to break.

## A human's open objection

`comments.txt` carries more than earlier cards. Where a human reviewer has raised a blocking
objection on this PR that the author has neither answered in code nor the reviewer withdrawn, it is
settled exactly like a card's finding: it goes in `prior_verdicts.txt` under the key
`human:<slug>`, and a `HOLDS` there cannot sit beside `Blocking issues: none`. A card that reports
no finding while an open human objection stands on the same page is not a second opinion, it is a
review that did not read the page: in PR #2020 one was published two hours after a
CHANGES_REQUESTED whose "+361 production lines against *Simplify by removing a mechanism*" was
still unanswered, and the PR ultimately landed at net -134.

Withdrawn means the reviewer said so. An author's reply that disagrees is not a withdrawal, and
neither is a later push that touches the file without addressing the point.

## The same head

When the latest card's `Head:` is this head, this review is a second reading of it, in full: every
new finding carries `(missed at <sha>)`, and `Blocking issues: none` cannot stand beside a `HOLDS`
line.

## On the card

The `Checked` line carries `Prior:`, listing `prior_verdicts.txt` in its order as
`<KEY> fixed|holds|moved`, comma-separated, for example `Prior: C1 fixed, T2 holds`, or `Prior: none`
when the latest card reported no finding. A review of a PR with no earlier card writes no `Prior`. A
`FIXED` finding leaves the card; `Prior` is where the author reads that it was fixed.
`pr-review-publish.sh` applies no verdict when `Prior` and `prior_verdicts.txt` disagree.
