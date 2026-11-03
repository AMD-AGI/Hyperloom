---
name: realtime-collector
description: Push an Arbor campaign's live state (phase, iteration in flight, verdicts, baseline and best, the log's tail) to the shared volume every minute, where the dashboard's global mode and other people read it, and set up the ssh access to the share that it needs. Use when starting or restarting a campaign, when a campaign is missing from the share or its snapshot is stale, or when asked to make a campaign visible to others.
---

# Realtime collector: a campaign's live state, on the share

Audience: whoever starts an Arbor campaign, or keeps one running. The
collector is `python3 -m arbor.collect realtime` (`arbor/collect/`, standard
library only). What a campaign learned, as opposed to where it stands, is the
`knowledge-collector` skill; reading either is the `dashboard` skill.

## What it does

Every `--interval` seconds (60 by default) it reads, from files on the host the
campaign runs on:

- `session.json`: iteration, baseline, best, every verdict, STOP and ABORTED
  files;
- the campaign's stdout log (`--log`): the phase (agent working, warmup,
  measured replay, critic, gate), the iteration in flight, gate lines,
  supervisor restarts, and whether an `arbor run` process is alive on this host;
- baseline records (`--results`, `*baseline*.json`), an event log
  (`--events`), and the target the config states (`--config`).

It writes `status.json` and `meta.json` to `<session>/collector/realtime/`, and
pushes both to the share:

```text
/shared/arbor/projects/<project>/<user>/<campaign>/realtime/{status.json,meta.json}
```

It never samples a GPU node: a probe during a measured replay perturbs it. It
never raises either: a section that fails is recorded in the snapshot's `errors`.

## Once per person: reach the share

The share is `/shared/arbor` on the Primus-SaFE pod
`arbor-pipeline-b9kcp-master-0` (workspace `control-plan-dev`), reached
through the Primus-SaFE ssh gateway. The gateway takes ssh commands, scp and
sftp, but not rsync and not piped stdin (tested 2026-10-06). That is why an
upload is one sftp batch (make the directories, put each file under a temporary
name, rename it into place), so a reader never sees half a file. The gateway
also returns exit status 0 for every command and puts its stderr on stdout
(tested 2026-10-07): in anything you script against it, print a marker or
JSON and read that, never `$?`.

1. Register your public key on Primus-SaFE, and add the pod to `~/.ssh/config`
   on every host that will run a collector (your workstation, WSL, a Cluster B
   login node):

   ```text
   Host arbor-share
       HostName <ssh gateway host>
       Port 2222
       User <your Primus-SaFE user id>.arbor-pipeline-b9kcp-master-0.pytorch.bash.control-plan-dev
       IdentityFile ~/.ssh/<the key you registered>
       IdentitiesOnly yes
       ServerAliveInterval 60
   ```

   The user is `<user id>.<pod>.<container>.<shell>.<workspace>`. If the pod
   is replaced, its name changes; Primus-SaFE shows the current one.
   Cluster B's login nodes reach the gateway (port 2222 open from `<login node>`,
   2026-10-07).
2. Check it from an Arbor checkout: `python3 -m arbor.collect check`. It runs
   three steps (ssh, the share root, an upload of a probe file it then removes)
   and, for the first that fails, says what to do.

The share root and its `projects/` directory are made once, by a person, never
by a collector. They were made on 2026-10-07 to match `/shared` itself:
owner root, group 10008 (`allusers` on Cluster B), mode 1775, so every member of
the group can add a project and no one can remove another person's:

```bash
ssh arbor-share 'mkdir -p /shared/arbor/projects && chgrp 10008 /shared/arbor /shared/arbor/projects && chmod 1775 /shared/arbor /shared/arbor/projects'
```

The pod's `/shared` is the same NFS volume Cluster B mounts (both see
`/shared/runner`; checked 2026-10-07), so on a Cluster B node the
share can be read at `/shared/arbor` directly. Collectors still write
through the pod. The volume was 98% full that day (12 TB free); the collectors
push summaries, never traces.

## Names: project, user, campaign

The path is `projects/<project>/<user>/<campaign>/`, and the index groups by
it, so get the names right before the first push:

```bash
python3 -m arbor.collect project --session <session> --config <config>
```

- **project**: one name per model family, from
  [`arbor/collect/projects.json`](../../../arbor/collect/projects.json).
  `--project`, else `ARBOR_PROJECT`, else the config's `project` chooses it
  outright (normalized through the table); otherwise the config's
  `model_name`, the pack, the session's name and its identity are tried in
  turn. A name the table does not know becomes a slug of
  itself, and the output says so. For a new model family, add an entry with
  examples (`arbor/tests/test_collect.py` checks every example resolves).
- **user**: `--user`, else `ARBOR_USER`, else the login name. Use your ldap,
  the same on every host, or your campaigns split across two users.
- **campaign**: the session directory's name, or `--campaign`. Keep it across
  restarts and chunks; a new name is a new campaign on the share.

## Run it beside the campaign

On the host that holds the session and the log, from an Arbor checkout:

```bash
python3 -m arbor.collect realtime --session <session> --config <config> \
    --log <the campaign's stdout log> [--slurm-job <job name>] [--results <dir>] [--events <file>] --interval 60
```

- **A driver inside a Slurm job** (a chain of chunks on GPU nodes, while the
  collector runs on a login node) needs `--slurm-job <job name>`: without it
  the collector looks for an `arbor run` process on its own host, finds none,
  and reports a running campaign as not running.

- **It must outlive your shell.** A Cluster B login node kills what an ssh
  session leaves behind; run the collector as a user unit, from a script file
  (sent from PowerShell, the quotes are stripped):

  ```bash
  loginctl enable-linger "$(id -un)"     # once per host
  systemd-run --user --collect --unit=arbor-rt-<campaign> --setenv=PATH="$PATH" \
    /bin/bash -lc "mkdir -p <session>/collector && cd <Arbor checkout> && exec python3 -m arbor.collect realtime --session <session> --config <config> --log <log> >> <session>/collector/realtime.out 2>&1"
  systemctl --user is-active arbor-rt-<campaign>
  ```

  User units are per host; check it on the host that started it.
- **One collector per campaign.** Two race on the same files on the share.
- `--once` takes one snapshot and exits (at the end of a campaign, say);
  `--no-push` writes only the local copy.

## When the share cannot be reached

The collector runs `check` when it starts. If any step fails, it prints which
step and what to do, keeps writing its local copy every interval, and retries a
push every 10 intervals. Each snapshot's `share` block records `share_ok`,
`last_error` and `pushed_at`, and the dashboard turns a failure into a banner.
On the share, the campaign's snapshot ages, and global mode marks it "last
seen ..." after 15 minutes. Exit status (with `--once`): 0 when pushed, 1 when
the share could not be reached or written (the local copy is still written),
2 for bad arguments.

## Reading it

- Every campaign, or one live: the `dashboard` skill.
- By hand: `ssh arbor-share cat /shared/arbor/projects/<project>/<user>/<campaign>/realtime/status.json`,
  or read the path directly on a Cluster B node.

## Settings

| Variable | Default | |
| --- | --- | --- |
| `ARBOR_SHARE_HOST` | `arbor-share` | ssh destination, an alias from `~/.ssh/config` |
| `ARBOR_SHARE_ROOT` | `/shared/arbor` | the share root on that host; a scratch root (under `/tmp`) for tests |
| `ARBOR_SHARE_SSH_OPTS` | none | extra options for both ssh and sftp; use `-F` and `-o` forms (`-o Port=2222`), since ssh and sftp spell `-p` differently |
| `ARBOR_PROJECT`, `ARBOR_USER` | none | the names above |
