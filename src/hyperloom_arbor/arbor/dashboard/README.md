# Arbor campaign dashboard

One page per campaign, in two modes, and a third for a parallel run:

- **local**: one campaign, live. Each refresh runs the realtime collector
  (`arbor/collect/realtime.py`, sent from this checkout inside the command)
  on the host that has the campaign's files, over ssh, or on this machine
  with `--host local`. Nothing there is used but `python3` and those files.
  A campaign whose driver runs inside a Slurm job gets `--slurm-job <name>`,
  and the job's state says whether it runs.
- **global**: every campaign on the share. Each refresh runs one `ssh` command
  to the share host that lists every campaign under
  `<share root>/projects/<project>/<user>/<campaign>/`. Opening a campaign
  fetches its `status.json` and knowledge files the same way.
- **parallel**: one parallel run (`arbor/parallel/`). Each refresh runs its
  status payload (`arbor/parallel/status.py`, sent the same way as the realtime
  collector) on the host that has the run's store: the phase and round, every
  lane's timeline and clocks, the round's map (the campaign page's checklist,
  `directions.js`), the integration table.

How to run it, and when: [`.claude/skills/dashboard/SKILL.md`](../../.claude/skills/dashboard/SKILL.md).
The collectors that feed it: `arbor/collect/`, and the `realtime-collector` and
`knowledge-collector` skills.

```text
local:   browser ── localhost:8766 ── server.mjs ── ssh <host> ── python3 -c <realtime.py> --session ... [--slurm-job ...]
global:  browser ── localhost:8767 ── server.mjs ── ssh arbor-share ── python3 -c <share_query.py> index|campaign|file
parallel: browser ── localhost:8768 ── server.mjs ── ssh <host> ── python3 -c <status.py> --store <store>/<run_id>
```

The browser never talks to a cluster. The server runs the commands, caches
their output, and serves it.

## Files

| File | What |
| --- | --- |
| `server.mjs` | the server, every mode; Node 16 or later, no dependencies |
| `share_query.py` | what global mode runs on the share host, sent compressed and base64-encoded inside the command (the gateway takes commands, not piped stdin; Git for Windows' ssh cuts a command at about 8 KB); Python 3.8 or later there, nothing installed. It always answers with one JSON object, `{"error": ...}` on failure, because through the gateway every command exits 0 and its stderr arrives on stdout (tested 2026-10-07) |
| `public/index.html`, `app.js` | one campaign: `/` in local mode, `/c/<project>/<user>/<campaign>/` in global mode, which adds the knowledge panel |
| `public/global.html`, `global.js` | the share's index, by project, with filters by project, user and state in the query string |
| `public/parallel.html`, `parallel.js` | a parallel run: lanes on the round's clock, their clocks and items, the round's map, the integration table |
| `public/directions.js` | what the campaign page and the parallel page share: escaping, wall times, the map's checklist |
| `public/styles.css` | every page |
| `smoke.mjs` | renders a page against a payload in a stub DOM; `arbor/tests/test_dashboard.py` runs it |

## Routes

| Mode | Route | Serves |
| --- | --- | --- |
| local | `/`, `/api/status` | the page; `{snapshot, meta}` |
| global | `/`, `/api/index` | the index; `{index, meta}` |
| global | `/c/<p>/<u>/<c>/`, `.../api/status` | a campaign's page; `{snapshot, knowledge, share, meta}` |
| global | `.../api/file?path=knowledge/...` | one file from the share, as text, up to 8 MB |
| parallel | `/`, `/api/status` | the run's page; `{status, meta}` |

## Ported from an earlier dashboard

The source is an earlier campaign dashboard, which served the v2 and v2.1
campaigns. Kept: the page and its panels, the snapshot's keys (`arbor.collect`'s realtime
collector writes the same ones), the server's caching, and its rules:

- **The collector never raises.** Every section records its own failure in
  `errors`, which the page shows as a panel; a partial page beats a blank one.
- **The last good snapshot is kept** when a refresh fails, and labelled stale.
- **ssh runs with `BatchMode=yes`**, so a password or host-key prompt fails
  fast instead of hanging the poll loop.
- **Everything put into a remote command is checked** against
  `[A-Za-z0-9._~/+-]`; anything else is refused.
- **The phase comes from Arbor's real stdout markers.** The verdict line is
  `[arbor] Gate: KEEP (B) — reason`, not `[arbor] KEEP`.

Changed:

- The collector is `arbor/collect/realtime.py`, sent inside each command,
  not a `collect.py` uploaded to the cluster. A cluster checkout, which others
  can change, never decides what the page shows. It writes nothing, so a
  dashboard polling beside a running realtime collector never replaces that
  collector's `status.json`.
- **Liveness from the Slurm job** (`--slurm-job`), as the Cluster A page's last
  fix did for chained campaigns: a driver inside a job on a GPU node is
  invisible to a process check on the login node, and the first port showed
  two running campaigns as "not running". Only a queued successor is "between
  chunks"; a `squeue` that fails is "unknown".
- **No node panel by default.** The realtime collector never samples GPU nodes:
  a probe during a measured replay perturbs it. The panel shows only when a
  snapshot carries node data.
- No hard-coded model, operating point or target. The KPIs show the
  campaign's own metric and unit; the progress bar shows only when the
  campaign config states a target (the Cluster A page fell back to a B200 number).
- The events panel shows only when the collector is given `--events`;
  `event_recorder.sh`, which watched one Cluster A campaign's paths, is not ported.
- Global mode, the knowledge panel, and the share's index are new.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Every value is `—` | A script failed to parse. `node --check public/app.js`, or `node smoke.mjs campaign <payload>` with a saved `api/status` |
| `ssh exited 255` | No key or alias for the host. `ssh -o BatchMode=yes <host> true` shows why |
| "Not running: no arbor run process on <login node>" over a live campaign | Its driver runs inside a Slurm job; start the dashboard with `--slurm-job <job name>` |
| `the collector command is N characters, over the 8000 ...` | Git for Windows' ssh is the one found (it cuts commands at about 8 KB); on Windows the server uses `C:\Windows\System32\OpenSSH\ssh.exe` when it exists |
| global: "projects does not exist" | The share root has not been created; the realtime-collector skill says how |
| global: a campaign reads "last seen ..." | Its realtime collector stopped pushing (its loop died, or the share check fails; the campaign's local `collector/realtime/status.json` says which) |
| `share_query: not a campaign file` | `api/file` takes paths under `realtime/` or `knowledge/` only, with no `..` |
