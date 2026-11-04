"use strict";

// Every campaign on the share, by project, user and campaign: what each
// campaign's realtime collector last pushed (meta.json, and a summary of
// status.json) and when its knowledge was last collected. A campaign opens at
// /c/<project>/<user>/<campaign>/. Filters live in the query string
// (?project=&user=&state=), so a filtered view can be linked.

const $ = (id) => document.getElementById(id);

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const num = (v, digits = 2) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

function agoSecs(iso) {
  return iso ? Math.max(0, (Date.now() - Date.parse(iso)) / 1000) : null;
}

function agoText(iso) {
  const secs = agoSecs(iso);
  if (secs === null || Number.isNaN(secs)) return "—";
  if (secs < 90) return `${Math.round(secs)}s ago`;
  if (secs < 5400) return `${Math.round(secs / 60)}m ago`;
  if (secs < 172800) return `${(secs / 3600).toFixed(1)}h ago`;
  return `${(secs / 86400).toFixed(1)}d ago`;
}

// The same line the campaign page draws: a snapshot older than this is not live.
const STALE_SECS = 15 * 60;

function stateOf(c) {
  const s = c.summary;
  if (!s) return c.knowledge ? "knowledge only" : "empty";
  if (s.aborted_file) return "aborted";
  if (s.running === false) return s.terminal ? "stopped" : "not running";
  const age = agoSecs(s.collected_at);
  return age !== null && age < STALE_SECS ? "live" : "quiet";
}

const STATE_CLASS = { live: "good", quiet: "idle", stopped: "idle", "not running": "bad",
                      aborted: "bad", "knowledge only": "idle", empty: "idle" };

const params = new URLSearchParams(location.search);
const filters = { project: params.get("project") || "", user: params.get("user") || "",
                  state: params.get("state") || "" };

function setOptions(select, values, current) {
  const opts = ["", ...values];
  if (current && !values.includes(current)) opts.push(current);
  select.innerHTML = opts.map((v) =>
    `<option value="${esc(v)}"${v === current ? " selected" : ""}>${esc(v || "all")}</option>`).join("");
}

let last = null;

function render(payload) {
  const idx = payload.index;
  const meta = payload.meta || {};
  $("where").textContent = `${meta.host}:${meta.root}/projects`;
  $("live-dot").className = "dot " + (meta.error ? "dead" : idx ? "live" : "stale");
  $("freshness").textContent = meta.error
    ? `share error · ${meta.consecutiveFailures} consecutive`
    : `listed ${agoText(idx?.listed_at)} on ${idx?.host || "?"}`;

  const banners = [];
  if (meta.error) banners.push(`<div class="banner bad"><strong>Share:</strong> ${esc(meta.error)}${idx ? " — showing the last good listing." : ""}</div>`);
  if (idx && !idx.projects_present) {
    banners.push(`<div class="banner warn"><strong>${esc(meta.root)}/projects does not exist.</strong>
      A person creates it once (the realtime-collector skill says how); nothing has been pushed yet.</div>`);
  }
  $("banner-area").innerHTML = banners.join("");
  if (!idx) {
    $("projects").innerHTML = `<p class="muted">no listing yet</p>`;
    return;
  }

  const all = idx.campaigns || [];
  for (const c of all) c.state = stateOf(c);
  const uniq = (xs) => [...new Set(xs)].sort();
  setOptions($("f-project"), uniq(all.map((c) => c.project)), filters.project);
  setOptions($("f-user"), uniq(all.map((c) => c.user)), filters.user);
  setOptions($("f-state"), uniq(all.map((c) => c.state)), filters.state);

  const shown = all.filter((c) => (!filters.project || c.project === filters.project)
    && (!filters.user || c.user === filters.user) && (!filters.state || c.state === filters.state));
  const live = shown.filter((c) => c.state === "live").length;
  $("counts").textContent = `${shown.length} of ${all.length} campaigns · ${live} live · `
    + `${uniq(shown.map((c) => c.project)).length} projects · ${uniq(shown.map((c) => c.user)).length} users`;

  if (!shown.length) {
    $("projects").innerHTML = `<p class="muted">${all.length ? "no campaign matches the filters" : "no campaign has pushed to the share yet"}</p>`;
    return;
  }
  const byProject = new Map();
  for (const c of shown) {
    if (!byProject.has(c.project)) byProject.set(c.project, []);
    byProject.get(c.project).push(c);
  }
  const updated = (c) => c.summary?.collected_at || c.meta?.updated_at || c.knowledge?.collected_at || "";
  $("projects").innerHTML = [...byProject.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([project, cs]) => {
    cs.sort((a, b) => updated(b).localeCompare(updated(a)));
    const rows = cs.map((c) => {
      const s = c.summary || {};
      const href = `/c/${encodeURIComponent(c.project)}/${encodeURIComponent(c.user)}/${encodeURIComponent(c.campaign)}/`;
      const iter = typeof s.current_iteration === "number" && c.state === "live" && s.current_iteration > (s.iteration ?? -1)
        ? `${s.current_iteration} <span class="muted">running</span>` : esc(s.iteration ?? "—");
      const gain = typeof s.gain_pct === "number" ? ` <span class="${s.gain_pct >= 0 ? "up" : "down"}">${s.gain_pct >= 0 ? "+" : ""}${s.gain_pct.toFixed(2)}%</span>` : "";
      const counts = s.counts ? `${s.counts.KEEP || 0}/${s.counts.REVERT || 0}` : "—";
      const k = c.knowledge;
      const kText = k ? `${agoText(k.collected_at)} · ${Object.entries(k.files || {}).map(([t, n]) => `${t} ${n}`).join(", ")}` : "—";
      return `<tr>
        <td><a href="${href}">${esc(c.campaign)}</a>${c.errors?.length ? ` <span class="verdict INVALID" title="${esc(c.errors.join("; "))}">!</span>` : ""}
          <div class="muted">${esc(s.pack || s.identity || "")}</div></td>
        <td>${esc(c.user)}</td>
        <td><span class="pill ${STATE_CLASS[c.state] || "idle"}">${esc(c.state)}</span>
          <div class="muted">${esc(c.state === "live" ? s.phase || "" : s.terminal || "")}</div></td>
        <td class="num">${iter}</td>
        <td class="num">${num(s.baseline)} → ${num(s.best_metric)}${gain}<div class="muted">${esc(s.metric_name || "")}</div></td>
        <td class="num">${counts}</td>
        <td>${esc(agoText(updated(c)))}<div class="muted">${esc(c.meta?.host || s.host || "")}</div></td>
        <td class="muted">${esc(kText)}</td>
      </tr>`;
    }).join("");
    return `<section class="panel"><h2>${esc(project)} <span class="muted">${cs.length} campaign${cs.length === 1 ? "" : "s"}</span></h2>
      <table><thead><tr><th>campaign</th><th>user</th><th>state</th><th class="num">iteration</th>
      <th class="num">baseline → best</th><th class="num">keep/revert</th><th>updated</th><th>knowledge</th></tr></thead>
      <tbody>${rows}</tbody></table></section>`;
  }).join("");
}

for (const key of ["project", "user", "state"]) {
  $(`f-${key}`).addEventListener("change", (e) => {
    filters[key] = e.target.value;
    const q = new URLSearchParams(Object.entries(filters).filter(([, v]) => v));
    history.replaceState(null, "", q.toString() ? `?${q}` : location.pathname);
    if (last) render(last);
  });
}

async function tick() {
  try {
    const res = await fetch("/api/index", { cache: "no-store" });
    last = await res.json();
    render(last);
  } catch (err) {
    $("live-dot").className = "dot dead";
    $("freshness").textContent = `dashboard server unreachable: ${err.message}`;
  }
}

tick();
setInterval(tick, 15000);
