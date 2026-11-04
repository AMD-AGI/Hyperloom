---
name: read
description: "Find the campaign's dossier and read it: look it up (find-dossier.sh), match it to the model, workload and GPU, check each part's staleness against the current pins, and cite its parts by file and section. Read only; it writes nothing. Use during a campaign's iterations to ground a change or a ranking in the dossier, or to learn which dossier part says what."
metadata:
  arbor:
    modes: [use]
---

# Reading the dossier

A dossier is what to know before optimizing one model, workload and GPU target. It is written once, then refreshed
part by part ([the rules](../../shared/rules.md), relative to this skill's directory). This skill finds it and reads
it. The dossier plugin's other skills write it:
- `dossier:discover`: parts 1-5 and 10;
- `dossier:measure`: 6-8 and 12;
- `dossier:reference`: 9;
- `dossier:directions`: 11;
- `dossier:update`: update rounds.

Where the steps below say a part is refreshed or written, that is their work, not this skill's.

**Cite what you use:** the file and section (`07-profile.md`, "Results"), and the part's status and date from the
index. A stale part is a lead, not evidence. Say so when you rely on one.

## First: look it up

A dossier is written once per model, workload and GPU type, then refreshed
part by part. It is not rewritten for each campaign or each question. Before
anything else:

1. **Find it.** Use a path you were given (by the user, or by the campaign's
   pack or config) first. Otherwise run `bash ../../shared/find-dossier.sh`.
   It lists `docs/dossier/*/README.md` in this repository and, when this
   repository is a submodule (Arbor inside a project), in the project around
   it. Then it lists the dossiers earlier campaigns shared,
   `/shared/arbor/projects/<project>/<user>/<campaign>/knowledge/dossier/`
   (the `knowledge-collector` skill), read in place where the share is
   mounted and through `ssh arbor-share` elsewhere. Each comes with its
   header: project, pins, platform, workload, date.
2. **Match it** on model, workload (benchmark, dataset, seed, concurrencies)
   and GPU type. Another GPU type means another dossier: MI350 never stands
   in for MI355X. Other pins or another node class make parts stale; they do
   not make the dossier missing.
3. **Found: use it.** Read the index and the parts your question needs. Check
   each of those parts' "Stale when" against the current pins, node class,
   workload and harness. Use the fresh parts as they are. Refresh only the
   parts that are stale, missing or partial, and only the ones the work needs;
   then update their status and date in the index. Never rewrite a part that
   still holds. A dossier found only on the share is the copy one campaign
   worked from, as of its collection: copy it into the project's
   `docs/dossier/` and refresh it there, never on the share. When the
   repository has the same dossier, the repository's copy is the one that is
   kept up to date. When the served tree itself has moved (a long
   campaign's KEEPs, or merges from other branches), refreshing parts one at
   a time is not enough: see the `dossier:update` skill.
4. **Missing: write it**, from the parts below. Say so before starting: it
   takes a node (parts 6 and 7) and the user's answers (the asks in [the rules](../../shared/rules.md)).
   The profile run (part 7) is part of writing it, not a later step: see
   "The profile is taken while the dossier is written", below.
   Material that is not in dossier form, such as the runner's `research/` and
   `learn/code/`, is a source to write from, not a dossier.


## Evaluation: the files exist and hold real content

Before a dossier or a round is called done, and before a campaign ranks
directions from it, run:

```bash
python3 -m arbor.dossier.check <dossier dir>                # parts 01-11 and the index
python3 -m arbor.dossier.check <dossier dir> --round uK     # 03, 06, 07, 10, 11 .uK and the index
python3 -m arbor.dossier.check <reports dir> --round uK --no-index   # a round staged outside the dossier
```

- **What it requires:** every needed file exists and holds real content. That means at least 300 words and 10 numbers outside headings, template labels, `<placeholders>`, table rules and empty cells. Kernel plans (`11.uK-plan-*.md`) are checked the same way when present. A whole dossier's part 4 also needs its `## The benchmark's rules` table, with at least one rule.
- **What fails a file:** missing, a copied template, a status of "missing", or a round the index doesn't list.
- **What it doesn't judge:** whether the content is right. That's what each part's "Done when" is for.

Calibrated on the Model M MI350X dossier: every real part has at least 635 words and 131 numbers, while a template skeleton with one line filled in has 17 words. A FAIL names the file and why. Write or finish that file. A part that genuinely can't be filled says so in its own status line ("not available", with the reason); it isn't left out.
