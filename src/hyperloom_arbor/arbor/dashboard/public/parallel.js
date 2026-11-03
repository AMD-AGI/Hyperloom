"use strict";

// A parallel run's page (arbor/parallel/, runner decision 0003): its phase and
// round, every lane's timeline and clocks, the round's map, the integration
// table. The data is `arbor.parallel status --json` (arbor/parallel/status.py),
// which the server runs where the run's store is; esc, durText and the map's
// table come from directions.js.

const $ = (id) => document.getElementById(id);

const LANE_STATE = { CLOSED: "good", BROKEN: "bad", FAILED: "bad", STOPPED: "idle", WAITING_DEPS: "idle" };
const ROUND_PHASES = { INIT: "starting", MAP: "mapping", FANOUT: "fanning out", DRAIN: "draining",
                       INTEGRATE: "integrating", CLOSED: "round closed", PAUSED: "paused",
                       STOPPING: "stopping", STOPPED: "stopped" };

function pill(text, cls) {
  return `<span class="pill ${esc(cls || "")}">${esc(text)}</span>`;
}

// One bar per lane, on the round's clock: waiting on its dependencies, waiting
// for a node, then on a node (chunks, requeues and handoffs inside it). SVG, so
// the page's CSP (no inline styles) holds; the colours are styles.css's seg-*.
function timelineHtml(lanes, now) {
  const starts = lanes.map((l) => l.clocks.created).filter((t) => typeof t === "number");
  if (!starts.length) return "";
  const t0 = Math.min(...starts);
  const span = Math.max(1, now - t0);
  const seg = (a, b, cls, title) => {
    if (typeof a !== "number") return "";
    const end = typeof b === "number" ? b : now;
    const x = ((a - t0) / span) * 1000, w = Math.max(4, ((end - a) / span) * 1000);
    return `<rect class="${cls}" x="${x.toFixed(1)}" y="0" width="${w.toFixed(1)}" height="14"><title>${esc(title)}</title></rect>`;
  };
  const rows = lanes.map((l) => {
    const c = l.clocks;
    const end = c.closed;
    return `<tr><td><code>${esc(l.lane)}</code></td><td class="lanebar">
      <svg viewBox="0 0 1000 14" preserveAspectRatio="none" width="100%" height="14">
        <rect class="seg-bg" x="0" y="0" width="1000" height="14"></rect>
        ${seg(c.created, c.deps_met || end, "seg-deps", `waiting on dependencies ${durText(c.wait_deps_s)}`)}
        ${seg(c.deps_met, c.started || end, "seg-wait", `waiting for a node ${durText(c.wait_node_s)}`)}
        ${seg(c.started, end, l.state === "CLOSED" ? "seg-closed" : "seg-run", `on a node ${durText(c.on_node_s)}`)}
      </svg></td><td>${pill(l.state, LANE_STATE[l.state])}</td></tr>`;
  }).join("");
  return `<table><tbody>${rows}</tbody></table>` +
    `<p class="muted">Grey: waiting on dependencies · amber: queued or pending · blue: on a node · green: closed. ` +
    `Round clock ${esc(durText(span))}.</p>`;
}

function itemsHtml(items) {
  return (items || []).map((i) =>
    `<div><code>${esc(i.id)}</code> ${i.stage ? esc(i.stage) + " " : ""}` +
    `<span class="dstatus ${esc(i.status || "open")}">${esc((i.status || "open").replace("_", " "))}</span>` +
    ` <span class="muted">${esc(durText(i.wall_s))}</span></div>`).join("");
}

function lanesHtml(lanes) {
  if (!lanes.length) return `<p class="muted">No lanes in this round yet.</p>`;
  const rows = lanes.map((l) => {
    const c = l.clocks, o = l.outcome || {};
    return `<tr><td><code>${esc(l.lane)}</code></td>
      <td><code>${esc(l.direction)}</code><div class="muted">${esc((l.title || "").slice(0, 80))}</div></td>
      <td>${esc(l.kind)}${l.nodes > 1 ? ` (${esc(l.nodes)} nodes)` : ""}</td>
      <td>${pill(l.state, LANE_STATE[l.state])}${l.node ? `<div class="muted">${esc(l.node)} · job ${esc(l.head)}</div>` : ""}</td>
      <td>${esc((l.depends_on || []).join(", ") || "—")}</td>
      <td class="num">${esc(durText(c.wait_deps_s))}</td><td class="num">${esc(durText(c.wait_node_s))}</td>
      <td class="num">${esc(durText(c.on_node_s))}</td><td class="num">${esc(durText(c.node_s))}</td>
      <td class="num">${esc(c.chunks ?? 0)}</td>
      <td>${o.status ? `<span class="dstatus ${esc(o.status)}">${esc(o.status)}</span>` : "—"}
        ${o.keeps && o.keeps.length ? `<div class="muted">KEEPs at ${esc(o.keeps.join(", "))}</div>` : ""}</td>
      <td>${itemsHtml(l.items)}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>lane</th><th>direction</th><th>kind</th><th>state</th><th>after</th>` +
    `<th class="num">wait deps</th><th class="num">wait node</th><th class="num">on node</th>` +
    `<th class="num">node time</th><th class="num">chunks</th><th>outcome</th><th>items</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>`;
}

// The round's map as the campaign page shows a map, each direction's lane filling it in.
function mapDirs(round) {
  const byDir = {};
  for (const l of round.lanes || []) byDir[l.direction] = l;
  return ((round.map || {}).directions || []).map((d) => {
    const l = byDir[d.id];
    const o = (l && l.outcome) || {};
    return { ...d, status: o.status && o.status !== "closed" ? o.status : d.status,
             why: o.why || d.why || (d.no_lane ? `no lane: ${d.no_lane}` : ""),
             keeps: o.keeps || [], worked: [], wall_s: l ? l.clocks.on_node_s : undefined };
  });
}

function mapStepHtml(ms) {
  const jobs = (ms && ms.jobs) || {};
  const names = Object.keys(jobs);
  if (!names.length) return ms && ms.result && ms.result.seeded ? `<p class="muted">Seeded: ${esc(ms.result.source)}</p>` : "";
  return `<p class="muted">Map step (u${esc(ms.uk)}): ` + names.map((j) =>
    `${esc(j === "profile" ? "profile" : "dossier " + j)} ${esc(jobs[j].state)} (attempt ${esc(jobs[j].attempt)}` +
    `${jobs[j].node ? ", " + esc(jobs[j].node) : ""})`).join(" · ") +
    ((ms.errors || []).length ? `<br>refused: ${esc(ms.errors.slice(0, 3).join("; "))}` : "") + `</p>`;
}

function integrationHtml(integ) {
  if (!integ) return `<p class="muted">Not yet: the round integrates once every lane has closed.</p>`;
  const rows = (integ.additions || []).map((a) => {
    const ab = a.ab || {};
    const cls = a.state === "KEEP" ? "good" : a.state === "AB" || a.state === "PENDING" ? "" : "bad";
    return `<tr><td class="num">${esc(a.k)}</td><td><code>${esc(a.lane)}</code></td><td><code>${esc(a.direction)}</code></td>
      <td>${pill(a.state, cls)}${a.reopenable ? ` ${pill("reopenable", "idle")}` : ""}</td>
      <td class="num">${typeof ab.delta_pct === "number" ? esc(ab.delta_pct.toFixed(2)) + " %" : "—"}</td>
      <td class="num">${typeof ab.base_spread_pct === "number" ? esc(ab.base_spread_pct.toFixed(2)) + " %" : "—"}</td>
      <td>${esc(ab.node || "—")}</td><td class="muted">${esc((a.why || ab.reason || "").slice(0, 160))}</td></tr>`;
  }).join("");
  const unchanged = (integ.unchanged || []).map((u) =>
    `<tr><td class="num">—</td><td><code>${esc(u.lane)}</code></td><td><code>${esc(u.direction)}</code></td>` +
    `<td>${pill("nothing to add", "idle")}</td><td></td><td></td><td></td><td class="muted">${esc((u.why || "").slice(0, 160))}</td></tr>`).join("");
  const res = integ.result;
  return `<table><thead><tr><th class="num">k</th><th>lane</th><th>direction</th><th>state</th><th class="num">delta</th>` +
    `<th class="num">base spread</th><th>node</th><th>why</th></tr></thead><tbody>${rows}${unchanged}</tbody></table>` +
    (res ? `<p>Round result: <strong>${esc(res.base_next)}</strong>, ${esc(res.kept)} kept, claim ` +
      `<strong>${esc(Number(res.claim_pct).toFixed(2))} %</strong> (the same-node A/Bs, compounded).</p>` : "") +
    (integ.failed ? `<div class="banner bad">The integration failed: ${esc(integ.failed)}</div>` : "");
}

function roundsHtml(rounds) {
  const earlier = rounds.slice(0, -1);
  if (!earlier.length) return `<p class="muted">This is the run's first round.</p>`;
  return earlier.map((r) => {
    const res = (r.integration || {}).result || {};
    return `<div>r${String(r.round).padStart(2, "0")}: ${esc((r.lanes || []).length)} lane(s), kept ${esc(res.kept ?? 0)}, ` +
      `claim ${esc(typeof res.claim_pct === "number" ? res.claim_pct.toFixed(2) : "0.00")} %` +
      `${res.base_next ? ` → ${esc(res.base_next)}` : ""}</div>`;
  }).join("");
}

function render(payload) {
  const s = payload.status || {};
  const meta = payload.meta || {};
  const st = s.state || {};
  const rounds = s.rounds || [];
  const round = rounds[rounds.length - 1] || { lanes: [] };
  const m = s.metrics || {};
  const now = typeof s.collected_at === "number" ? s.collected_at : Date.now() / 1000;
  document.title = `${s.run_id || "run"} · parallel · Arbor`;
  $("identity").textContent = `${s.run_id || "?"} · ${s.store || ""} · Arbor ${(s.arbor_sha || "").slice(0, 10)}`;
  $("phase-chip").textContent = `${ROUND_PHASES[st.phase] || st.phase || "—"}${st.round ? ` · round ${st.round}` : ""}`;
  $("live-dot").className = meta.error ? "dot dead" : "dot live";
  $("freshness").textContent = meta.error ? `collector failed: ${meta.error}` :
    `collected ${typeof meta.ageMs === "number" ? Math.round(meta.ageMs / 1000) + "s ago" : "—"}`;

  $("kpi-round").textContent = st.round ? `r${String(st.round).padStart(2, "0")}` : "—";
  $("kpi-round-foot").textContent = `${st.phase || "—"} · ${durText(m.round_wall_s)} in this round`;
  $("kpi-base").textContent = s.base ? `base_${String(s.base.index).padStart(3, "0")}` : "—";
  $("kpi-base-foot").textContent = s.base ? `${(s.base.sha || "").slice(0, 10)} · ${s.base.source || ""}` : "—";
  $("kpi-nodes").textContent = `${m.nodes_busy ?? 0} / ${s.max_lanes ?? "—"}`;
  $("kpi-nodes-foot").textContent = Object.entries(m.lanes || {}).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(" · ") || "no lanes";
  $("kpi-kept").textContent = `${m.kept ?? 0}`;
  $("kpi-kept-foot").textContent = `compounded ${typeof m.compounded_gain_pct === "number" ? m.compounded_gain_pct.toFixed(2) : "0.00"} % over base_000`;

  const banners = [];
  if (st.phase === "STOPPED" && st.stop) {
    banners.push(`<div class="banner bad"><strong>Stopped (${esc(st.stop.rule)}):</strong> ${esc(st.stop.why)}` +
      `${st.stop.resume_to ? " · `arbor.parallel resume` continues it once you have acted." : ""}</div>`);
  }
  if (st.phase === "PAUSED") banners.push(`<div class="banner warn"><strong>Paused.</strong> Lanes are observed; nothing new starts.</div>`);
  if ((s.switches || {}).STOP && st.phase !== "STOPPED") banners.push(`<div class="banner warn">STOP is set: the lanes stop at their next boundary.</div>`);
  if (s.attention) banners.push(`<div class="banner warn"><pre class="log">${esc(s.attention)}</pre></div>`);
  $("banner-area").innerHTML = banners.join("");

  const lanes = round.lanes || [];
  $("lanes-sub").textContent = `· ${lanes.length} in round ${round.round ?? "—"}`;
  $("timeline").innerHTML = timelineHtml(lanes, now);
  $("lanes").innerHTML = lanesHtml(lanes);

  const dirs = mapDirs(round);
  const nClosed = dirs.filter((d) => DIR_CLOSED.has(d.status)).length;
  $("map-round").textContent = round.map ? `· u${round.map.uk ?? "?"} · ${nClosed} of ${dirs.length} closed` : "· not yet";
  $("mapstep").innerHTML = mapStepHtml(round.map_step);
  $("map").innerHTML = dirs.length ? mapTableHtml(dirs) : `<p class="muted">No map yet.</p>`;
  $("integration").innerHTML = integrationHtml(round.integration);
  $("rounds").innerHTML = roundsHtml(rounds);
  $("events").innerHTML = `<table><tbody>${[...(s.events || [])].reverse().slice(0, 25).map((e) =>
    `<tr><td class="num">${esc(new Date(e.ts * 1000).toISOString().slice(5, 19).replace("T", " "))}</td>` +
    `<td><code>${esc(e.kind)}</code></td><td class="muted">${esc(Object.entries(e).filter(([k]) => k !== "ts" && k !== "kind")
      .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ").slice(0, 200))}</td></tr>`).join("")}</tbody></table>`;
}

let last = null;

async function tick() {
  try {
    const res = await fetch("api/status", { cache: "no-store" });
    last = await res.json();
    render(last);
  } catch (err) {
    $("live-dot").className = "dot dead";
    $("freshness").textContent = `dashboard server unreachable: ${err.message}`;
  }
}

tick();
setInterval(tick, 15000);
