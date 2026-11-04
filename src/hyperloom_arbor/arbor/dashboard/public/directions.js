"use strict";

// What the campaign page (app.js) and the parallel run's page (parallel.js)
// share: escaping, wall times, and the map's checklist (arbor/directions.py),
// one row per direction, the open ones first. Loaded before either page's
// script.

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// A wall time in seconds as 3h05m, 42m or 50s; "—" when unknown.
function durText(secs) {
  if (typeof secs !== "number" || !(secs > 0)) return "—";
  const s = Math.round(secs);
  if (s < 60) return `${s}s`;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m`;
}

const DIR_CLOSED = new Set(["done", "killed", "below_floor", "plan"]);

// The checklist as a table: worked is every iteration whose DONE line cited the
// direction, KEEPs the kept ones, time their wall time (a parallel run fills
// these from the direction's lane).
function mapTableHtml(dirs) {
  const order = (d) => [DIR_CLOSED.has(d.status) ? 1 : 0, d.kind === "increment" ? 0 : 1,
                        typeof d.rank === "number" ? d.rank : 999];
  const rows = [...dirs].sort((a, b) => {
    const x = order(a), y = order(b);
    return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
  }).map((d) => {
    const worked = (d.worked || []).join(", ");
    const keeps = (d.keeps || []).join(", ");
    return `<tr>
        <td>${DIR_CLOSED.has(d.status) ? "&#9745;" : "&#9744;"}</td>
        <td><code>${esc(d.id)}</code>${d.flagged ? ` <span class="pill bad" title="worked ${(d.worked || []).length} times, never closed">flagged</span>` : ""}
          <div class="muted">${esc((d.title || "").slice(0, 120))}</div></td>
        <td>${esc(d.kind || "")}</td>
        <td class="num">${esc(d.rank ?? "")}</td>
        <td>${esc(d.ceiling || "")}</td>
        <td><span class="dstatus ${esc(d.status || "open")}">${esc((d.status || "open").replace("_", " "))}</span>
          ${d.why ? `<div class="muted">${esc(d.why.slice(0, 160))}</div>` : ""}</td>
        <td class="num">${esc(worked || "—")}</td>
        <td class="num">${esc(keeps || "—")}</td>
        <td class="num" title="the wall time of the iterations that worked it">${esc(durText(d.wall_s))}</td>
      </tr>`;
  }).join("");
  return `<table><thead><tr><th></th><th>direction</th><th>kind</th><th class="num">rank</th><th>ceiling</th>` +
    `<th>status</th><th class="num">worked</th><th class="num">KEEPs</th><th class="num">time</th></tr></thead><tbody>${rows}</tbody></table>`;
}
