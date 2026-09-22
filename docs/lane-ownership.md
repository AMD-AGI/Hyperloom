---
myst:
    html_meta:
        "robots": "noindex"
orphan: true
---

# Lane ownership: move the mutex back into the process

A proposal, not a change. It rests on one measurement that has not been taken
yet, named at the end.

## What happened

2026-09-21, no operator involved. A GPU specialist ended at 07:04 with its
cleanup unconfirmed, holding six lanes. The release is skipped on that path on
purpose — its process tree may still be running, and the lane is what keeps
conflicting work off the cards. Nothing afterwards could establish that those
processes had gone, so the rows stayed.

Twenty-one `integrate_patch` tasks then sat in `queued` for two hours behind
them while the coordinator's log went silent. Deleting six rows by hand released
it in ninety seconds.

The cards were idle the whole time: 0% VRAM, no server process. **The lock
outlived its holder, and nothing else was wrong.**

## What the lane was protecting

Measured from that session's own database:

| tasks | requires_lanes |
| --- | --- |
| 21 `integrate_patch` | `benchmark_lane`, `server_lifecycle`, `workspace_mutation` |
| 1 `baseline` | `benchmark_lane`, `server_lifecycle` |
| 1 `specialist` | `gpu_research_lane`, `research_lane` |
| 11 `specialist` | `research_lane` (pure research — never touches a GPU) |
| 1 `target_analysis` | none |

Eleven of the twelve specialists never go near a GPU. The serving lanes conflict
with each other and with `gpu_research_lane`, so **at most one task may serve at
a time by design**. GPU work in an ENABLEMENT session is already serial; the
lanes describe that, they do not enable concurrency.

And there is exactly one coordinator: `leases` lives in
`$SESSION_DIR/storage/coordinator.db`, one session, one process, one event loop.

So the mutex has a single user, and it is enforcing "do not dispatch two GPU
tasks at once" — a property that process holds entirely on its own.

## The actual defect: one table, two jobs

`leases` serves two unrelated purposes.

**A — in-process mutual exclusion.** `dispatcher.py` (8 call sites) takes lanes
before dispatch and `sub_agent_runner.py` (5) holds them for the execution.
Single process, single loop. Persisting this buys nothing, and it creates a
failure mode that cannot otherwise exist: **a lock can outlive its holder.**

**B — a cross-session safety gate.** `resume_guard.py` reads the table before
resuming an old session, to find out whether the previous run's processes are
still alive:

```python
for row in db.execute("SELECT lane, holder_id, task_id, pid, owner_scope FROM leases"):
    pid = int(row["pid"] or 0)      # is the previous coordinator still there?
```

This one genuinely needs durability. But what it records is *"the last run left
this behind"*, which is not mutual exclusion at all — it is a crash record that
happens to be stored in the mutex table.

The incident is entirely within A. A is the half that does not need to be
durable.

## The proposal

Split them.

**A becomes in-memory.** Lanes are held by the asyncio task that owns the work,
acquired and released through a context manager. When that task ends — returns,
raises, or is cancelled — the lane is gone with it. When the process ends, every
lane ends. *"Leaked lane"* stops being a state the system can reach.

**B stays durable, and says what it means.** Resume keeps reading a record of
what the previous run left running, keyed by pid and `owner_scope`. That record
can be the `tasks` table it already maintains — a task in `running` with a
recorded pid is the same evidence, in the place that already owns it — so the
lease table can go entirely.

What falls away with A: `expires_at`, `heartbeat_at`, `heartbeat_by_task`,
`reap_dead_holders`, `owner_scope` on the mutex path, the acquire-time sweep,
and the operator diagnostic that exists because a lane can strand. Roughly
speaking this proposal **deletes** a subsystem rather than adding one.

## What it gives up

One thing, and it should be stated plainly: **the deliberate retention on the
cleanup-unconfirmed path.** Today that retention means "its processes may still
be alive, so keep the cards reserved." Under this proposal the lane is released
when the task ends and the next task may start immediately.

That is a real reduction in safety, in exchange for removing a failure mode that
has actually fired. Whether the trade is right depends on a fact nobody has
measured.

## The measurement that decides it

`cleanup_unconfirmed` now rides the maintenance summary (PR #1595). It counts
ended tasks whose teardown was not confirmed, against all ended tasks.

What is needed is one step further: of the tasks that end unconfirmed, **how
often is anything actually still running?** In the incident the answer was never
— 0% VRAM, no server process, the retention protected nothing. If that
generalises, A's retention is pure cost and this proposal is straightforwardly
correct. If real survivors are common, releasing on task end would put two
rounds on the same cards, and the honest answer is to keep today's behaviour and
accept the operator in the loop.

Gather it over a handful of ENABLEMENT runs before writing any code:

- the `cleanup_unconfirmed` ratio per session;
- for each unconfirmed ending, whether the GPUs were busy a minute later;
- how many lanes a session strands, and whether any run starves as 2026-09-21
  did.

## Alternatives, and why this one

| | nature | new privileges | fixes the incident | survivor processes |
| --- | --- | --- | --- | --- |
| containment (cgroup / PID ns) | add a subsystem | **required, unobtainable** | yes | eliminated |
| pre-emptive cleanup before start | add a mechanism | none | yes | killed on sight |
| **in-memory mutex (this)** | **delete a subsystem** | none | **yes** | **not handled** |
| shipped today (PR #1595) | add a diagnostic | none | no — visible, manual | conservatively retained |

Containment is the only one that eliminates survivors, and it is undeployable:
claw mode mounts cgroupfs read-only and refuses `mount()`, so neither a per-task
cgroup nor a usable PID namespace can be built from inside. See
`task-containment.md` for those measurements.

Pre-emptive cleanup was considered and set aside: it solves survivors, but no
survivor has been observed. Designing for an unobserved failure is what produced
three refuted designs already.

This proposal is the only one that makes the incident structurally impossible by
removing code, and the only cost it carries is a protection whose value is
measurable — so measure it.

## If it is adopted

1. Land PR #1595 and collect `cleanup_unconfirmed` across several real runs.
2. If retention proves to be protecting nothing, give `resume_guard` its own
   durable record (or point it at `tasks`), so the lease table has one remaining
   user.
3. Move acquisition into a context manager owned by the executing task; delete
   the SQLite backend, the TTL fields and the reaper along with it.
4. Keep one test that pins the property worth keeping: **a lane cannot outlive
   the task that holds it**, including when that task raises or is cancelled.
