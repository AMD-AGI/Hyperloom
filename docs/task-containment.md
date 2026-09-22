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

## Three designs, measured against the real environments

Hyperloom runs in three shapes, and the mechanism has to hold in all of them.

| | claw mode (production) | local / baremetal | privileged pod (testing only) |
| --- | --- | --- | --- |
| how it starts | Claw -> SaFE -> PyTorchJob | `pip install`, run on a GPU host | hand-made |
| `CAP_SYS_ADMIN` | **no** | sometimes | yes |
| `/sys/fs/cgroup` | **read-only** | varies | writable |
| `mount()` | **EPERM** | varies | permitted |
| Kubernetes | yes | **may not exist** | yes |

`local` is not one environment. The install guide tells users to pick
`baremetal` *even inside someone else's Docker container*, and notes Hyperloom
may run as non-root — so it spans bare-metal root and containers measured to be
as locked down as claw mode.

### Per-task cgroup — undeployable

Needs a delegated writable subtree. Claw mode and the locked-down half of local
mount cgroupfs read-only and cannot create one at all. Only the privileged pod
can — and it is also the one environment where a cgroup cannot secure anything:
a root process there leaves its cgroup with a single write to `cgroup.procs`,
measured.

### PID namespace — undeployable, and failed in a way worth recording

`unshare(CLONE_NEWUSER|CLONE_NEWPID)` succeeds everywhere, including claw mode
without `CAP_SYS_ADMIN`, and ROCm sees every GPU from inside. A survivor that
calls `setsid` twice and outlives its root — the exact shape that defeated the
process group, the pidfile and the command line — cannot leave the namespace and
is killed when its init exits. The boundary itself is sound.

It still does not work, because a real server will not start in one:

```
detokenizer_manager.py:546
    parent_process = psutil.Process().parent()
psutil.NoSuchProcess: process PID not found (pid=173)
```

A PID namespace renumbers pids while `/proc` still shows the host, so the
server's children cannot find their own parent. Fixing that means mounting a
fresh procfs, which needs a mount namespace — and `mount()` returns `EPERM` in
claw mode and in the locked-down half of local. It works only in the privileged
pod.

### One Kubernetes pod per specialist — not portable

Sound where Kubernetes exists, and it puts containment where it belongs: with
the cluster runtime. But local mode may have no Kubernetes at all.

### Why they fail for the same reason

The deployment deliberately removes exactly the privileges needed to construct
an isolation domain from inside a container. Being unable to build one is the
security model working, not a misconfiguration to route around.

## Conclusion: retention is the mechanism

There is no portable way to prove a lane is free. Quoting the design review:

> A plain pipe, socketpair, flock, OFD lock, UID scan, PID scan, or any
> combination of them does not meet the bar and must not trigger release.

So the shipped behaviour is not a stopgap, it is the answer: **retain the lane
unless teardown positively confirms completion**, and make the retention
visible. That is what PR #1595 does — one operator diagnostic per
`(lane, holder)`, naming the lane, the holder, why it cannot be verified, and a
paste-ready statement that clears it.

Overlap is structurally impossible in every supported environment. The cost is
an operator on an ordinary path.

## The one mechanism that could automate a subset

A **sealed execution**: the workload inherits one end of a pipe with
`FD_CLOEXEC` cleared, a supervisor keeps the other, and the kernel emits EOF only
once every reference is gone. That survives `setsid`, double-forking,
reparenting and `exec`, because the open-file reference is copied through
`fork` and outlives `exec`.

A bare descriptor is not enough: a workload can `close`, `close_range`, `dup2`
or re-set `FD_CLOEXEC` and produce EOF while it is still GPU-capable. What makes
it sound is an **unprivileged seccomp filter** — available in all three shapes,
inherited by every descendant, and impossible to weaken — that refuses every
syscall route which could discard or replace the sentinel.

Conditions, none of them optional:

- The filter must cover *every* fd-destruction mechanism of the running kernel.
  If it does not, the execution **must not** be classified as sealed.
- `SCM_RIGHTS` can only prolong retention, never release early; allowing
  `sendmsg` means accepting hidden queued references as conservative retention.
- Only the sealed subset may be released automatically. Everything else keeps
  the behaviour above.

This is worth building only when the numbers justify it. The comprehensiveness
of that filter is safety-critical: getting it wrong is more dangerous than not
having it, because it would release lanes on a guarantee that does not hold.

## Deciding whether to build it

`leases_unverifiable` already rides the maintenance summary each tick, and the
diagnostic names every retained lane. What that does not answer is how often an
ordinary run ends up there. Measure first:

- How many specialists end with cleanup unconfirmed, as a share of all that end?
- How many lanes does a typical session strand, and for how long?
- Does a session ever strand enough to starve, as 2026-09-21 did, or is it
  usually one lane nobody needed?

If stranding is rare, the diagnostic is sufficient and the sealed backend is not
worth its risk. If it is routine, these numbers are also the argument for
building it.
