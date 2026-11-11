---
name: worker-summary
description: "Write a worker's own account of its stretch: the iterations it ran on the map its work order gave, from the order to its report. What it tried, what held and why, what did not and what that showed, which directions it closed or left open, and what the master should weigh for the next map, every claim with its evidence. Reads the session, and writes WORKER_SUMMARY.md beside the mechanical report. Arbor runs it when a stretch ends (the stretch_end event)."
disable-model-invocation: true
metadata:
  arbor:
    modes: [trigger]
    triggers: [stretch_end]
    blocking: true
    produces: ["{session}/WORKER_SUMMARY.md"]
    gpu: false
---

# A worker's account of its stretch

A session is a master and its workers (Arbor's `docs/master-worker.md`). A worker's stretch starts with a work order
(`work_order.json`) and ends with a report (`worker_report.json`, `WORKER_REPORT.md`). That report is mechanical: counts,
KEEPs, the bar, the directions closed and open. This skill adds the worker's own account, which the report then names
(`account`), for the master (its next map, its round end) or the host that reads it.

Arbor runs it once per stretch, in a turn of its own, before the worker hands over. The session waits for it, within the
time the run has left, so be brief.

## What you read

Your argument is the path of a brief. Read it first: it names the session, and the event's journal entry gives the
report (`report`), the order's id (`order`), the stretch's iterations (`first` to the entry's `iteration`) and why it
ended (`ended`).

In the session directory:

| File | What it gives you |
| --- | --- |
| `worker_report.json`, `WORKER_REPORT.md` | the mechanical report: decisions, KEEPs, the bar and its scale, the directions |
| `work_order.json` | what the stretch started from: the map's directions, the bar, the target, the limits |
| `session.json` | `lessons`: one row per verdict. Read the rows of the stretch's iterations: `change`, `decision`, `metric`, `reason`, `findings` |
| `ledger.jsonl` | the findings the agents recorded; read those of the stretch's iterations |
| `directions.json` | the map: each direction's status and `why`, and `proposals` (directions an agent proposed to the master) |
| `iter_N/` | one iteration's evidence (`scope.json`, `patch.diff` on a KEEP or PROBE); open it only to settle a claim |

Read nothing else, and change nothing: not the served tree, not the session's records, not the map.

## What you write

`{session}/WORKER_SUMMARY.md` (the brief's session), at most about 120 lines:

```markdown
# Worker summary: round <round>, iterations <first>-<last>

<One paragraph: why the stretch ended; the best result against the baseline, with the scale the report gives; how many
KEEPs; the one thing the master should know first.>

## What held
- <each KEEP: the change, its gain, why it worked; iteration N, metric>

## What did not, and what it showed
- <by direction: what was tried (REVERT, PROBE, NOT MEASURED), what was learned; iterations, ledger ids>

## The directions
- **Closed:** <each, with why>
- **Left open:** <each, with where it stands>
- **Proposed:** <the proposals, with why>

## For the next map
- <what to drop, extend or add, and why; measurement concerns, such as noise or a bench that did not run>
```

**Rules:**
- **Every claim carries its evidence:** an iteration number, a metric from `session.json`, or a ledger id. Numbers come
  from the session's records only; never estimate one.
- **Say what is not known:** a direction with one inconclusive try is open, not exhausted. A NOT MEASURED iteration
  measured nothing.
- **Keep the report's numbers:** where you and `worker_report.json` disagree, the report is right; say what differs.

End your reply with one line: `SUMMARY: <the path you wrote>`.
