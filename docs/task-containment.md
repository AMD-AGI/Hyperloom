---
myst:
    html_meta:
        "robots": "noindex"
orphan: true
---

# Task containment

A lane is the mutex that keeps two rounds off the same GPUs. When a specialist
ends without confirming its teardown, the dispatcher deliberately keeps that
lane: the process tree may still be running, and the lane is the only thing
holding conflicting work back. Nothing afterwards can establish that those
processes are gone, so the lane is kept for the life of the session.

Observed 2026-09-21, with no operator involvement: a specialist ended with
cleanup unconfirmed at 07:04 holding six lanes. Its process tree was in fact
already gone — the GPUs read 0% VRAM and no server process survived. Nineteen
queued tasks starved behind those lanes for two hours and the session made no
progress. Deleting the six rows by hand released it within ninety seconds.

This page describes the mechanism that lets such a lane be released
automatically, and why the cheaper answers do not work.

## Why process identities cannot answer this

Three proofs were implemented and each was refuted by probe.

**The holder's task is terminal.** Terminal says the task stopped, not that its
children did. This is the retained-lane contract, inverted.

**The process group recorded at spawn is empty.** Both launch sites spawn with
`start_new_session=True`, so the root's pid is also its group and session id and
keeps naming the group after the root exits. A survivor that stays in the dead
root's group is caught. A descendant that calls `setsid()` leaves the group, the
tree — its parent is rewritten to the reaper — and the session, and nothing in
`/proc` ties it back. `_server_lifecycle` states the problem plainly where it
reads a pidfile: *"Server is setsid'd, so pgid == pid unless the pid file gave
one."* The served inference servers, which are exactly what a lane protects, are
the processes that escape. This is the main path, not an edge case.

**No pidfile under the session's `runs/` names a live server.** The pidfile is
written only after `wait_for_server_ready`, so for the whole model-load window —
measured at 4m20s for a 599 GB checkpoint — a live server maps VRAM with nothing
naming it. `teardown_lifecycle_server` unlinks it unconditionally, including on
the branch where it explicitly refuses to signal. And identifying "our" servers
by command line is the same class of proxy: a live `/bin/bash …/sglang_mi300x.sh`
was classified foreign and its lane released.

The pattern is the point. Each identity is an *observation of expected
behaviour*, and a process that daemonises is free to stop matching it. Asking
better questions about processes does not fix that; the identity has to be one a
descendant cannot leave.

## The invariant

> Every process able to use a task's GPUs is born into one containment domain,
> inherits it across `fork`, `exec`, daemonisation, reparenting and `setsid()`,
> and cannot move itself or its descendants out. The lane is coupled to that
> domain and released only on a positive attestation that the domain is empty.

A cgroup v2 subtree is such a domain. `setsid()` changes PID-session and
process-group membership only; it does not change cgroup membership. Forked and
exec'd descendants stay in the parent's cgroup, reparenting does not affect it,
and PID namespaces do not erase the cgroup's node-local identity.

Two properties make it structural rather than likely:

- **No window.** The first workload process is created *directly in* the cgroup
  with `clone3(CLONE_INTO_CGROUP)`. There is no interval in which specialist
  code runs outside it. Adding a cgroup from a Python `preexec_fn`, or moving
  the pid in after `Popen` returns, both leave that window open and do not close
  the hole.
- **No exit.** The workload runs without `CAP_SYS_ADMIN` and without a writable
  `cgroupfs`, so it cannot migrate itself or a descendant out.

Emptiness is then a fact rather than a failure to find anything:
`cgroup.events` reports `populated 0`, and `cgroup.kill` drains the domain in one
operation.

### Verified

A proof of concept on the target pod (`/wekafs/yihan/cgroup_poc.py`) clones a
root into a fresh cgroup; the root calls `setsid()`, forks, and exits, and the
grandchild calls `setsid()` again — reproducing every escape that defeated the
earlier designs:

```
cloned root pid   : 193141 (exited)
cgroup members    : [193142]
survivor 193142   : pgid=193142 sid=193142   <- left group and session
after cgroup.kill : members=[] populated=0
```

The survivor is invisible to the process group, to the session, to the pidfile
and to command-line matching. It is still in the cgroup.

## Design

A node-local containment service, owned by the coordinator deployment rather
than by specialist or Ray-actor lifetimes.

For each GPU-holding task it mints a never-reused UUID and creates
`<delegated-root>/hyperloom/<session-uuid>/<task-uuid>`. The lane acquisition and
a containment record commit together in SQLite, carrying the task UUID, node
identity, node `boot_id`, cgroup version, cgroup path and id, the assigned GPU
ids, and a lifecycle state. The service keeps its own durable ledger, so the two
sides can be reconciled after either one crashes.

Release runs through that state machine and nowhere else: the lane is freed only
after the node owning the domain attests it is empty. Process-group, pidfile,
task-terminal and command-line identities are removed from release decisions
entirely.

A domain is never identified by a reusable path alone. Paths are reused; UUIDs
and `boot_id` are not.

## Boundaries

Ray-placed actors on other nodes are **out of scope** until the same node-local
boundary exists there. Killing an actor releases Ray's logical GPUs while its
daemonised subprocesses may survive, so actor lifetime is specifically not proof
of process-tree death. Until each eligible worker runs the service, those lanes
stay fail-closed — held, and reported.

Multi-tenant Ray is also out of scope. Containment proves when *our* work is
finished; it does not stop an unrelated Ray client from being scheduled onto the
same physical cards. That needs either exclusive workers or a node-level
reservation keyed by physical GPU identity, used by every client.

## When not to build this

Containment becomes another escapable proxy — and this design should be
abandoned for one Kubernetes pod per specialist, where the cluster runtime owns
containment and fencing — if the deployment cannot:

- remove `CAP_SYS_ADMIN` and cgroup-write authority from specialist and server
  processes, or
- provide a delegated cgroup v2 subtree on every execution node, or
- prevent unrelated GPU clients from scheduling onto the same cards without a
  shared reservation boundary.

Startup validation treats each of these as unsupported and **disables automatic
reclamation**, keeping the fail-closed lane. It must never fall back to
inspection: a silent downgrade to a refuted proxy is worse than a lane an
operator can see and clear.

## Failure modes

Every one of them retains the lane. None releases on an inference.

| Situation | Behaviour |
| --- | --- |
| Containment preparation or atomic spawn fails | No specialist code ran. Fail the task; release only after the service attests the prepared domain is empty or was never activated. |
| Node service unreachable | Retain and retry. Visible stranding, never overlap. |
| `cgroup.kill` fails, or `populated` never reaches 0 | Retain, report the remaining membership, escalate. Never substitute GPU-utilisation or process-name inspection. |
| Coordinator crashes | The cgroup and the node ledger survive it. The reconciler resumes from SQLite and still requires an attestation. |
| Node service crashes | Kernel membership survives. It rebuilds active domains from its ledger and cgroupfs; an ambiguous same-boot absence stays fail-closed. |
| Worker reboots | A changed `boot_id` proves the old processes cannot survive, but release still waits for the restarted service to confirm recovery. |
| Workload retains `CAP_SYS_ADMIN` or cgroup write | The structural guarantee is broken. Startup validation rejects this rather than degrading. |
| SQLite and the node ledger disagree | Retain, reconcile by UUID and `boot_id`. Never infer safety from a missing row, path, actor or pid. |

## Costs

- A privileged deployment component on every execution node, with a narrowly
  delegated writable subtree and permission to kill within it.
- Specialist workloads can no longer be fully privileged. Scripts expecting
  `CAP_SYS_ADMIN`, writable `cgroupfs`, nested containers or arbitrary namespace
  and mount operations will break and must move those operations to a controlled
  service.
- A small native launcher, because ordinary Python spawning cannot place a child
  in a cgroup atomically before user code runs.
- Two durable ledgers and a state machine, with the operational complexity that
  follows around crashes and boot identity.
- Forced cleanup kills every process in the domain, so a per-task cgroup must
  contain only that task's processes. Placing a shared helper there becomes a
  correctness bug.

## Steps

1. Startup capability detection: cgroup v2, a writable delegated subtree,
   `clone3(CLONE_INTO_CGROUP)`, and the ability to launch without
   `CAP_SYS_ADMIN` or cgroup write. Any failure disables automatic reclamation
   and preserves the current fail-closed behaviour.
2. Containment and node-ledger schemas, without changing release behaviour yet.
   UUID identities and `boot_id`; never a bare path.
3. The node service and the atomic launcher, used first for one local
   GPU-specialist spawn path, asserting that deliberately daemonised and
   `setsid`'d descendants appear under the task cgroup.
4. Containment made mandatory for every local path that can launch GPU work,
   including Magpie scripts. Dispatch is rejected before specialist code runs if
   preparation or atomic spawn fails.
5. `release_resources()` releases GPU lanes only through the state machine, after
   an empty attestation. Process-group, pidfile, task-terminal and command-line
   identities are removed from release decisions.
6. The restart reconciler, tested against crashes at every transition: before and
   after the SQLite commit, cgroup creation, spawn, terminal write, kill, empty
   observation and release. A `PREPARING` record with no workload requires a
   ledger attestation, not a bare missing-path check.
7. The service on every eligible Ray worker, returning its attested identity at
   actor startup, with actor subprocess creation routed through the launcher.
   Lane ownership stays with the coordinator, not inside the actor.
8. Before multi-tenant Ray: either a node-level reservation keyed by physical GPU
   identity that all clients consult, or exclusive workers. Until then the
   guarantee is limited to Hyperloom-controlled submissions.
9. Optionally a cgroup v1 backend. Its presence must not weaken the v2 contract;
   partially delegated v1 hosts stay fail-closed.

## Rejected

- **Process groups, sessions, tree traversal, subreapers, pidfds.** A pidfd
  identifies one process reliably, not a descendant that daemonises before being
  registered. `setsid` and reparenting defeat the rest.
- **Pidfiles, command lines, open files, ROCm utilisation, VRAM at zero.**
  Observations of expected behaviour, with startup gaps, deletion races and
  false negatives — not ownership boundaries.
- **Namespaces alone.** A PID namespace improves visibility but does not stop a
  privileged descendant creating nested namespaces, and emptiness is less direct
  to establish than a cgroup's.
- **Ray actor lifetime or `ray.kill` acknowledgement.** Actor death releases
  Ray's logical GPUs while daemonised subprocesses may remain.
- **Holding a module-scope actor handle.** Prevents accidental collection; says
  nothing about descendants and does not survive coordinator failure.
- **systemd transient scopes.** Viable only where systemd is the node-local
  service; typically unavailable inside these containers, and the same privilege
  restrictions still apply.
- **GPU device locks without containment.** A durable physical reservation can
  prevent overlap but cannot decide when abandoned work is dead. Complementary
  for multi-tenant Ray, not a substitute.
- **One Kubernetes pod or Job per specialist.** A genuinely strong boundary with
  cluster-owned cleanup, but a larger execution-model change and slow for
  iterative specialists. The reasonable alternative if the constraints above
  cannot be met.
