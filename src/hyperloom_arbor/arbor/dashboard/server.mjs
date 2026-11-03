#!/usr/bin/env node
// Arbor campaign dashboard: one campaign live (local mode), or every campaign
// on the share (global mode). Ported from the Cluster A campaign dashboard
// (scripts/arbor_dashboard/ on <fork>/dsv41-v2.1-ruppell); README.md here
// says what changed.
//
//   node server.mjs local    --host <host with the session> --session <dir> [--slurm-job <name>] ...
//   node server.mjs global   [--share-host arbor-share] [--share-root /shared/arbor]
//   node server.mjs parallel --host <host with the run's store> --store <store>/<run_id>
//
// local:  each refresh runs the realtime collector (arbor/collect/realtime.py,
//         sent from here inside the command) where the campaign's files are,
//         over ssh, or here with --host local.
// global: each refresh runs one `ssh <share host>` command that lists every
//         campaign under <share root>/projects; opening a campaign fetches its
//         status.json and knowledge files the same way, and a file link fetches
//         that file.
// parallel: a parallel run (arbor/parallel/): each refresh runs its status
//         payload (arbor/parallel/status.py, sent the same way) on the store.
//
// The browser never talks to a cluster, and the cluster runs only code this
// checkout sends: nothing installed or checked out there can change what the
// page shows. Commands run with BatchMode, so a prompt fails fast; the last
// good result is kept, and labelled, on failure.

import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { shipped } from "./ship.mjs";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
// The Arbor checkout this file is in: the collector it sends comes from here.
const ARBOR_ROOT = path.resolve(MODULE_DIR, "..", "..");

const USAGE = `usage:
  node server.mjs local  --host <ssh host | local> --session <dir> [--slurm-job <job name>]
                         [--config F] [--log F] [--results D] [--events F]
                         [--campaign C] [--python python3]
  node server.mjs global [--share-host H] [--share-root D] [--python python3]
  node server.mjs parallel --host <ssh host | local> --store <store>/<run_id> [--python python3]
common: [--port N] [--refresh ms] [--timeout ms] [--label L] [--ssh-opts "..."]`;

// Everything interpolated into a remote shell is restricted to characters that
// cannot break out of it; a path with a semicolon fails loudly instead.
const SAFE_ARG = /^[A-Za-z0-9._~/+-]*$/;
// The collectors' share path components (arbor/collect/project.py).
const COMPONENT = /^[a-z0-9][a-z0-9._-]*$/;

function safe(value, label) {
  if (!SAFE_ARG.test(String(value))) {
    throw new Error(`${label} may only contain letters, digits and . _ ~ / + -`);
  }
  return String(value);
}

function requireInt(value, label, min, max) {
  if (!/^\d+$/.test(String(value))) throw new Error(`${label} must be an integer`);
  const n = Number(value);
  if (n < min || n > max) throw new Error(`${label} must be between ${min} and ${max}`);
  return n;
}

function parseArgs(argv) {
  const mode = argv[0];
  if (!["local", "global", "parallel"].includes(mode)) throw new Error(USAGE);
  const cfg = {
    mode,
    port: { local: 8766, global: 8767, parallel: 8768 }[mode],
    refreshMs: mode === "global" ? 30000 : 15000,
    timeoutMs: 90000,
    label: "",
    python: "python3",
    sshOpts: (process.env.ARBOR_SHARE_SSH_OPTS || "").split(/\s+/).filter(Boolean),
    // local
    host: "", session: "", config: "", log: "", results: "", events: "", slurmJob: "", campaign: "",
    // parallel
    store: "",
    // global
    shareHost: process.env.ARBOR_SHARE_HOST || "arbor-share",
    shareRoot: process.env.ARBOR_SHARE_ROOT || "/shared/arbor",
    maxOutputBytes: 32 * 1024 * 1024,
  };
  const flags = {
    "--port": (v) => { cfg.port = requireInt(v, "--port", 1, 65535); },
    "--refresh": (v) => { cfg.refreshMs = requireInt(v, "--refresh", 2000, 3600000); },
    "--timeout": (v) => { cfg.timeoutMs = requireInt(v, "--timeout", 5000, 600000); },
    // Shown in the page title so two dashboards are distinguishable in tabs.
    "--label": (v) => { cfg.label = safe(v, "--label"); },
    "--python": (v) => { cfg.python = safe(v, "--python"); },
    "--ssh-opts": (v) => { cfg.sshOpts = v.split(/\s+/).filter(Boolean); },
  };
  if (mode === "local") {
    for (const k of ["host", "session", "config", "log", "results", "events", "campaign"]) {
      flags[`--${k}`] = (v) => { cfg[k] = safe(v, `--${k}`); };
    }
    // A driver inside a Slurm job is invisible from a login node: name the job.
    flags["--slurm-job"] = (v) => { cfg.slurmJob = safe(v, "--slurm-job"); };
  } else if (mode === "parallel") {
    for (const k of ["host", "store"]) flags[`--${k}`] = (v) => { cfg[k] = safe(v, `--${k}`); };
  } else {
    flags["--share-host"] = (v) => { cfg.shareHost = safe(v, "--share-host"); };
    flags["--share-root"] = (v) => { cfg.shareRoot = safe(v, "--share-root"); };
  }
  for (let i = 1; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!flags[key]) throw new Error(`unknown option ${key}\n${USAGE}`);
    if (value === undefined) throw new Error(`missing value for ${key}`);
    flags[key](value);
  }
  if (mode === "local") {
    if (!cfg.host || !cfg.session) throw new Error(`local mode needs --host and --session\n${USAGE}`);
  } else if (mode === "parallel") {
    if (!cfg.host || !cfg.store) throw new Error(`parallel mode needs --host and --store\n${USAGE}`);
  } else {
    safe(cfg.shareHost, "ARBOR_SHARE_HOST");
    safe(cfg.shareRoot, "ARBOR_SHARE_ROOT");
  }
  return cfg;
}

let cfg;
try {
  cfg = parseArgs(process.argv.slice(2));
} catch (err) {
  console.error(err.message);
  process.exit(2);
}

// -- running commands --------------------------------------------------------

// Run a command; resolve { ok, stdout (Buffer), durationMs, error }. Never rejects.
function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const started = Date.now();
    let child;
    try {
      child = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"], ...opts });
    } catch (err) {
      resolve({ ok: false, error: `failed to start ${cmd}: ${err.message}` });
      return;
    }
    const chunks = [];
    let bytes = 0;
    let errText = "";
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ durationMs: Date.now() - started, ...result });
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({ ok: false, error: `timed out after ${cfg.timeoutMs} ms` });
    }, cfg.timeoutMs);
    child.stdout.on("data", (chunk) => {
      bytes += chunk.length;
      if (bytes > cfg.maxOutputBytes) {
        child.kill("SIGKILL");
        finish({ ok: false, error: "output exceeded the size limit" });
        return;
      }
      chunks.push(chunk);
    });
    child.stderr.on("data", (chunk) => { errText += chunk.toString("utf8"); });
    child.on("error", (err) => finish({ ok: false, error: `failed to start ${cmd}: ${err.message}` }));
    child.on("close", (code) => {
      if (code !== 0) {
        finish({ ok: false, error: `${cmd} exited ${code}: ${errText.trim().slice(-600)}` });
      } else {
        finish({ ok: true, stdout: Buffer.concat(chunks) });
      }
    });
  });
}

// Git for Windows' ssh, first on PATH in Git Bash, cuts a command at about
// 8 KB (measured 2026-10-07: 8,000 characters arrive, 8,200 do not); Windows'
// own OpenSSH and Linux's pass 24,000. On Windows, use Windows' own.
const WINDOWS_SSH = "C:\\Windows\\System32\\OpenSSH\\ssh.exe";
const SSH = process.platform === "win32" && existsSync(WINDOWS_SSH) ? WINDOWS_SSH : "ssh";
const MAX_COMMAND = process.platform === "win32" && SSH !== WINDOWS_SSH ? 8000 : 24000;

function ssh(host, command) {
  return run(SSH, ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", ...cfg.sshOpts, host, command]);
}

// The command's answer is the JSON object on its stdout. Through the
// Primus-SaFE gateway every command exits 0 and its stderr arrives on stdout
// (tested 2026-10-07), so a failure there is either share_query's
// {"error": ...} or output that is not JSON, never an exit status.
function parseJson(result) {
  if (!result.ok) return result;
  const out = result.stdout.toString("utf8");
  let data;
  try {
    data = JSON.parse(out);
  } catch (err) {
    // A stray banner or motd line ahead of the JSON is the usual cause.
    const start = out.indexOf("{");
    try {
      if (start <= 0) throw err;
      data = JSON.parse(out.slice(start));
    } catch {
      const head = out.trim().slice(0, 300) || "(no output)";
      return { ok: false, durationMs: result.durationMs, error: `not JSON: ${head}` };
    }
  }
  if (data && typeof data.error === "string" && Object.keys(data).length === 1) {
    return { ok: false, durationMs: result.durationMs, error: data.error };
  }
  return { ...result, data };
}

// -- Python sent inside the command --------------------------------------------

// Both modes run a Python file from this checkout on the far host, inside the
// command, compressed (and the collector minified) by ship.mjs so that even Git
// for Windows' ssh passes it (MAX_COMMAND above; arbor/tests/test_dashboard.py
// holds both files under its limit).

// Run a shipped file with arguments, here (host "local") or over ssh.
function runShipped(host, code, args, what) {
  for (const a of args) safe(a, `${what} argument`);
  if (host === "local") return run(cfg.python, ["-c", code, ...args]);
  const command = `${cfg.python} -c '${code}' ${args.join(" ")}`;
  if (command.length > MAX_COMMAND) {
    return Promise.resolve({ ok: false, error: `the ${what} command is ${command.length} characters, over the ${MAX_COMMAND} some ssh clients pass` });
  }
  return ssh(host, command);
}

// -- local mode: one campaign, collected where its files are -------------------

const REALTIME = shipped(path.join(ARBOR_ROOT, "arbor", "collect", "realtime.py"), { minify: true });

function collectorArgs() {
  const args = ["--session", cfg.session];
  for (const [k, flag] of [["config", "--config"], ["log", "--log"], ["results", "--results"],
                           ["events", "--events"], ["slurmJob", "--slurm-job"], ["campaign", "--campaign"]]) {
    if (cfg[k]) args.push(flag, cfg[k]);
  }
  return args;
}

function collectorDescription() {
  return `${cfg.host === "local" ? "here" : `ssh ${cfg.host}`}: ${cfg.python} -c <arbor/collect/realtime.py from ` +
    `${ARBOR_ROOT}, ${REALTIME.length} characters> ${collectorArgs().join(" ")}`;
}

async function collectLocal() {
  return parseJson(await runShipped(cfg.host, REALTIME, collectorArgs(), "collector"));
}

// -- global mode: the share ---------------------------------------------------

const QUERY = shipped(path.join(MODULE_DIR, "share_query.py"));

function shareQuery(args) {
  return runShipped(cfg.shareHost, QUERY, args, "share query");
}

// -- parallel mode: one parallel run's store ---------------------------------------

const STATUS = shipped(path.join(ARBOR_ROOT, "arbor", "parallel", "status.py"), { minify: true });

async function collectParallel() {
  return parseJson(await runShipped(cfg.host, STATUS, ["--store", cfg.store], "status"));
}

// -- caches --------------------------------------------------------------------

// One cached, refreshable value: the last good data is kept on failure.
function makeCache(fetcher) {
  const c = { data: null, fetchedAt: 0, durationMs: null, error: null, failures: 0, inFlight: null,
              requestedAt: Date.now() };
  c.refresh = () => {
    if (c.inFlight) return c.inFlight;
    c.inFlight = (async () => {
      const r = await fetcher();
      c.durationMs = r.durationMs ?? null;
      if (r.ok) {
        Object.assign(c, { data: r.data, fetchedAt: Date.now(), error: null, failures: 0 });
      } else {
        c.error = r.error;
        c.failures += 1;
      }
      c.inFlight = null;
      return r;
    })();
    return c.inFlight;
  };
  c.fresh = async () => {
    c.requestedAt = Date.now();
    if (!c.data || Date.now() - c.fetchedAt > cfg.refreshMs) await c.refresh();
    return c;
  };
  c.meta = (extra = {}) => ({
    mode: cfg.mode,
    fetchedAt: c.fetchedAt ? new Date(c.fetchedAt).toISOString() : null,
    ageMs: c.fetchedAt ? Date.now() - c.fetchedAt : null,
    collectMs: c.durationMs,
    error: c.error,
    consecutiveFailures: c.failures,
    refreshMs: cfg.refreshMs,
    label: cfg.label,
    ...extra,
  });
  return c;
}

const local = cfg.mode === "local" ? makeCache(collectLocal) : null;
const parallelRun = cfg.mode === "parallel" ? makeCache(collectParallel) : null;
const shareIndex = cfg.mode === "global" ? makeCache(async () => parseJson(await shareQuery(["index", cfg.shareRoot]))) : null;
const campaigns = new Map();   // "p/u/c" -> cache, fetched only while someone looks

function campaignCache(names) {
  const key = names.join("/");
  if (!campaigns.has(key)) {
    campaigns.set(key, makeCache(async () => parseJson(await shareQuery(["campaign", cfg.shareRoot, ...names]))));
  }
  return campaigns.get(key);
}

// -- HTTP ------------------------------------------------------------------------

// The page is served from this origin only and pulls no remote assets.
const BASE_HEADERS = Object.freeze({
  "Content-Security-Policy":
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
});

const STATIC = new Map([
  ["/app.js", ["app.js", "text/javascript; charset=utf-8"]],
  ["/directions.js", ["directions.js", "text/javascript; charset=utf-8"]],
  ["/global.js", ["global.js", "text/javascript; charset=utf-8"]],
  ["/parallel.js", ["parallel.js", "text/javascript; charset=utf-8"]],
  ["/styles.css", ["styles.css", "text/css; charset=utf-8"]],
]);

function send(res, status, body, headers = {}) {
  res.writeHead(status, { ...BASE_HEADERS, "Cache-Control": "no-store", ...headers });
  res.end(body);
}

function sendJson(res, obj) {
  send(res, 200, JSON.stringify(obj), { "Content-Type": "application/json; charset=utf-8" });
}

async function serveStatic(res, file, type) {
  try {
    send(res, 200, await readFile(path.join(MODULE_DIR, "public", file)), { "Content-Type": type });
  } catch {
    send(res, 404, "not found", { "Content-Type": "text/plain" });
  }
}

async function handleLocal(url, res) {
  if (url.pathname === "/" || url.pathname === "/index.html") return serveStatic(res, "index.html", "text/html; charset=utf-8");
  if (url.pathname === "/api/status") {
    await local.fresh();
    return sendJson(res, { snapshot: local.data, meta: local.meta({ host: cfg.host, session: cfg.session }) });
  }
  return send(res, 404, "not found", { "Content-Type": "text/plain" });
}

async function handleParallel(url, res) {
  if (url.pathname === "/" || url.pathname === "/index.html") return serveStatic(res, "parallel.html", "text/html; charset=utf-8");
  if (url.pathname === "/api/status") {
    await parallelRun.fresh();
    return sendJson(res, { status: parallelRun.data, meta: parallelRun.meta({ host: cfg.host, store: cfg.store }) });
  }
  return send(res, 404, "not found", { "Content-Type": "text/plain" });
}

async function handleGlobal(url, res) {
  const where = { host: cfg.shareHost, root: cfg.shareRoot };
  if (url.pathname === "/" || url.pathname === "/index.html") return serveStatic(res, "global.html", "text/html; charset=utf-8");
  if (url.pathname === "/api/index") {
    await shareIndex.fresh();
    return sendJson(res, { index: shareIndex.data, meta: shareIndex.meta(where) });
  }
  const m = url.pathname.match(/^\/c\/([^/]+)\/([^/]+)\/([^/]+)(\/.*)?$/);
  if (!m || ![m[1], m[2], m[3]].every((x) => COMPONENT.test(x))) {
    return send(res, 404, "not found", { "Content-Type": "text/plain" });
  }
  const names = [m[1], m[2], m[3]];
  const rest = m[4] || "";
  if (rest === "") return send(res, 302, "", { Location: `/c/${names.join("/")}/` });
  if (rest === "/") return serveStatic(res, "index.html", "text/html; charset=utf-8");
  if (rest === "/api/status") {
    const c = await campaignCache(names).fresh();
    const d = c.data || {};
    return sendJson(res, {
      snapshot: d.status ?? null,
      knowledge: d.knowledge ?? null,
      share: { project: names[0], user: names[1], campaign: names[2], dir: d.dir, meta: d.meta,
               status_mtime: d.status_mtime, errors: d.errors || [] },
      meta: c.meta({ ...where, names }),
    });
  }
  if (rest === "/api/file") {
    const p = url.searchParams.get("path") || "";
    if (!SAFE_ARG.test(p) || !/^(realtime|knowledge)\//.test(p) || p.split("/").some((s) => s === ".." || s === "")) {
      return send(res, 400, "bad path", { "Content-Type": "text/plain" });
    }
    const r = parseJson(await shareQuery(["file", cfg.shareRoot, ...names, p]));
    if (!r.ok) return send(res, 502, r.error, { "Content-Type": "text/plain; charset=utf-8" });
    return send(res, 200, Buffer.from(r.data.b64 || "", "base64"), { "Content-Type": "text/plain; charset=utf-8" });
  }
  return send(res, 404, "not found", { "Content-Type": "text/plain" });
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (req.method !== "GET") return send(res, 405, "method not allowed", { "Content-Type": "text/plain" });
    const entry = STATIC.get(url.pathname);
    if (entry) return serveStatic(res, entry[0], entry[1]);
    return { local: handleLocal, parallel: handleParallel, global: handleGlobal }[cfg.mode](url, res);
  } catch (err) {
    send(res, 500, `dashboard error: ${err.message}`, { "Content-Type": "text/plain" });
  }
});

server.listen(cfg.port, "127.0.0.1", () => {
  console.log(`[dashboard] ${cfg.mode} mode: http://localhost:${cfg.port}`);
  const first = { local, parallel: parallelRun, global: shareIndex }[cfg.mode];
  console.log({ local: () => `[dashboard] collector: ${collectorDescription()}`,
                parallel: () => `[dashboard] run: ${cfg.host}:${cfg.store} (arbor/parallel/status.py, ${STATUS.length} characters)`,
                global: () => `[dashboard] share: ${cfg.shareHost}:${cfg.shareRoot}/projects` }[cfg.mode]());
  console.log(`[dashboard] refresh every ${cfg.refreshMs} ms`);
  first.refresh().then((r) => {
    if (!r.ok) console.error(`[dashboard] first collection failed: ${r.error}`);
    else console.log(`[dashboard] first collection ok in ${r.durationMs} ms`);
  });
});

// Poll in the background so the page is served from a warm cache. A campaign
// opened in global mode is fetched while someone has it open, and dropped ten
// minutes after the last look.
setInterval(() => {
  ({ local, parallel: parallelRun, global: shareIndex }[cfg.mode]).refresh().catch(() => {});
  for (const [key, c] of campaigns) {
    if (Date.now() - c.requestedAt > 600000) campaigns.delete(key);
  }
}, cfg.refreshMs);
