---
name: dashboard
description: Watch Arbor campaigns in a browser. Local mode shows one campaign live, collected over ssh where its agents run; global mode shows every campaign on the shared volume by project, user and campaign, with each one's last snapshot and what it learned. Use when asked to watch, monitor or show a campaign or all campaigns, to see a campaign's phase, iteration and verdicts at a glance, or to browse what other campaigns pushed.
---

# Dashboard: one campaign live, or every campaign on the share

Audience: whoever runs or follows Arbor campaigns. The server is
[`arbor/dashboard/server.mjs`](../../../arbor/dashboard/server.mjs) (Node 16 or
later, no dependencies); its README has the files, routes, what changed from
the earlier dashboard it was ported from, and troubleshooting. The data comes from
the collectors (`realtime-collector`, `knowledge-collector` skills).

The server runs on your machine, binds to `127.0.0.1` only, and runs
read-only commands over ssh with `BatchMode`. It never writes on a host or the
share, and never samples a GPU node.

## Which mode

| You want | Mode |
| --- | --- |
| every campaign, anyone's, by project; a campaign whose host you cannot reach; what a campaign learned | global |
| your campaign, live, straight from its host (before its realtime collector is set up, or when the share is down) | local |

## Global mode: everything on the share

```bash
node arbor/dashboard/server.mjs global          # http://localhost:8767
```

- Needs the `arbor-share` ssh alias (the `realtime-collector` skill, "Once per
  person"). `--share-host`, `--share-root` (or `ARBOR_SHARE_HOST`,
  `ARBOR_SHARE_ROOT`) point it elsewhere, such as a scratch root under `/tmp`.
- **The index** (`/`) lists every campaign under `<root>/projects/`, grouped by
  project: user, state, the iteration in flight, baseline and best with the
  gain, keep and revert counts, when it was last updated and from which host,
  and when its knowledge was collected. Filter by project, user and state;
  the filters stay in the URL, so a filtered view can be shared as a link.
  It refreshes every 30 s (`--refresh`); each refresh is one ssh command.
- **States**: `live` (a snapshot from the last 15 minutes, with no sign the
  campaign stopped); `quiet` (no snapshot for 15 minutes: the collector stopped, or
  the campaign ended without saying so); `stopped` (the log names the end);
  `not running` (no driver on the campaign's host and no end named: it may
  have been killed); `aborted` (an `ABORTED.md`); `knowledge only`.
- **A campaign** (`/c/<project>/<user>/<campaign>/`) is the campaign page
  below, from the snapshot its collector last pushed, plus a knowledge panel:
  the summary, outcomes, findings, each profile's report and the dossier's
  index, and every file on the share, each a link that fetches it. It is
  fetched while the page is open, at most once per refresh interval.

## Local mode: one campaign, from where it runs

```bash
node arbor/dashboard/server.mjs local --host <host with the session> \
    --session <session dir> --config <config> --log <campaign stdout log> \
    [--slurm-job <job name>] [--results <dir>] [--events <file>] [--label <tab name>]   # http://localhost:8766
```

- Each refresh (every 15 s; `--refresh`) sends `arbor/collect/realtime.py`
  from this checkout inside one ssh command and runs it there. The host needs
  only `python3` and the campaign's files; no Arbor checkout there, and
  nothing someone changes in one, affects the page. It writes nothing, so it
  can run beside the campaign's realtime collector. Paths are absolute.
- **A driver inside a Slurm job** (a chain of chunks on GPU nodes) is
  invisible from the login node: give `--slurm-job <job name>`, and the job's
  state decides running, between chunks (only a queued successor) or unknown
  (`squeue` failed). Without it, the page looks for an `arbor run` process on
  the host and reports a chained campaign as not running.
- `--host` is an ssh alias with key authentication. `--host local` runs the
  collector on this machine (it needs `python3` here: WSL, not Windows).
- **Run it from WSL with a reused connection** when the host is slow to reach
  (`<login node>`: 7–15 s per ssh connection): `--ssh-opts "-o ControlMaster=auto
  -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=10m"`. Windows'
  OpenSSH cannot reuse connections. On Windows the server uses
  `C:\Windows\System32\OpenSSH\ssh.exe` when present, because Git for Windows'
  ssh cuts a command at about 8 KB. The Model M campaigns' pair runs this way:
  `scripts/model-m-campaign/dashboards.sh start|stop|status`.
- On a Cluster B login node, use one that accepts remote commands (`<login node>`)
  and keep the refresh at 15 s or more: every refresh is a
  process on a host with a 4 GiB per-user cap, and when the host refuses a
  session, the page keeps the last snapshot and says so.
- Several campaigns: one server each, with its own `--port` and `--label`.

## The campaign page

- **Top bar**: the phase chip (the iteration in flight and what it is doing,
  from the log's last marker: agent working, warmup, measured replay, critic,
  gate), and two clocks: when the snapshot was taken, and when the campaign
  last wrote its log. A stale snapshot is never shown as live.
- **Banners**: a snapshot over 15 minutes old (global); no driver running; a
  STOP, CHAIN_STOP or `ABORTED.md` file; supervisor restarts; a collector
  that cannot reach the share; a refresh that failed (the last good snapshot
  stays, labelled).
- **Cards**: baseline, best and gain, the iteration (in flight vs. last
  booked), keep and revert counts. **Progress** shows only when the config
  states a target.
- **History**: every booked iteration, newest first, with its verdict, metric
  and change. Then baselines (with `--results`), A/B (when present), events
  (with `--events`), the agent's latest tool calls, the log's tail, and the
  collector's own errors.

## Keep it running

In a terminal is enough. To keep one up in WSL as a user unit:

```bash
systemd-run --user --collect --unit=arbor-dash-global --setenv=PATH="$PATH" \
  /bin/bash -lc "exec node <Arbor checkout>/arbor/dashboard/server.mjs global >> ~/arbor-dash-global.log 2>&1"
```

WSL forwards `localhost`, so the Windows browser opens it. Where dashboards
are collected in a hub (the runner's, at <http://localhost:8790>), link it
there instead of passing the port around:
`hub-register --link <id> http://localhost:8767 "Arbor share"`.
