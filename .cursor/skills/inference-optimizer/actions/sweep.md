# sweep — quasi-random scheduling sweep

**Family**: `shallow` · **Cost**: ~8‑20 min · **Risk**: zero accuracy

Last‑ditch shallow exploration when no other action's score exceeds 1.0
(DESIGN §9.3 update rule #7). Picks 3‑5 random parameter combos within
the policy‑allowed safe set and bench‑runs each.

Outputs:

- `report` follows once sweep completes — exit through `report` even if
  no winner (so the user gets a final summary).
