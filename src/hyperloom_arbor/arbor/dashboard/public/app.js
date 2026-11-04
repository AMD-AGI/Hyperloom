"use strict";

// One campaign's page. Local mode serves it at / with a snapshot the server
// collected where the campaign runs; global mode serves it at
// /c/<project>/<user>/<campaign>/ with the snapshot the campaign's realtime
// collector last pushed to the share, plus its knowledge. Either way the data
// comes from api/status, relative to the page. esc, durText and the map's
// table come from directions.js, loaded first.

const $ = (id) => document.getElementById(id);

const num = (v, digits = 3) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

const pct = (v, digits = 2) =>
  typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : "—";

const bytes = (n) =>
  typeof n !== "number" ? "—" : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;

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

// A snapshot pushed to the share this long ago no longer describes a live campaign.
const STALE_SECS = 15 * 60;

// Which knowledge text is shown; kept across refreshes.
let knowledgeTab = null;

function render(payload) {
  const snap = payload.snapshot;
  const meta = payload.meta || {};
  const global = meta.mode === "global";
  const share = payload.share || {};

  // ---- freshness -----------------------------------------------------------
  // Two different clocks, and conflating them hides real problems: how fresh
  // the SNAPSHOT is, and how long since the CAMPAIGN last wrote anything. In
  // global mode the snapshot's age is when its collector took it, not when
  // this server last read the share.
  const dot = $("live-dot");
  const ageMs = meta.ageMs ?? Infinity;
  const snapAge = global ? agoSecs(snap?.collected_at) : ageMs / 1000;
  dot.className = "dot " + (meta.error ? "dead"
    : global ? (snapAge !== null && snapAge < STALE_SECS ? "live" : "stale")
    : ageMs < meta.refreshMs * 3 ? "live" : "stale");
  const campaignBeat = snap?.log?.log_mtime;
  $("freshness").textContent = meta.error
    ? `collector error · ${meta.consecutiveFailures} consecutive`
    : global
      ? `${meta.host}:${share.dir || "?"} · snapshot taken ${agoText(snap?.collected_at)} on ${snap?.host || "?"}`
        + (campaignBeat ? ` · campaign log ${agoText(campaignBeat)}` : "")
      : `${meta.host} · snapshot ${agoText(meta.fetchedAt)}`
        + (campaignBeat ? ` · campaign log ${agoText(campaignBeat)}` : "");
  $("back-link").hidden = !global;

  renderKnowledge(payload.knowledge);

  if (!snap) {
    const why = meta.error || (global
      ? (payload.knowledge ? "this campaign pushed its knowledge but no realtime snapshot"
        : "nothing pushed for this campaign yet")
      : "waiting for the collector");
    $("banner-area").innerHTML = `<div class="banner ${global && payload.knowledge ? "warn" : "bad"}">No snapshot. ${esc(why)}</div>`;
    if (global) $("identity").textContent = [share.project, share.user, share.campaign].filter(Boolean).join(" / ");
    renderErrors([], share.errors);
    return;
  }

  const c = snap.campaign || {};
  const log = snap.log || {};
  const sup = log.supervisor || {};
  const names = snap.share || {};

  $("identity").textContent = c.identity || snap.session_dir || snap.root || "—";
  // Distinguish browser tabs when more than one campaign dashboard is open.
  const label = meta.label || names.campaign || share.campaign || (c.identity || "").split(":").pop() || "";
  if (label) {
    document.title = `${label} · Arbor`;
    const h = document.querySelector(".brand h1");
    if (h) h.textContent = `Arbor · ${label}`;
  }

  // ---- phase ---------------------------------------------------------------
  // The phase comes from the last log marker, which keeps describing whatever
  // was happening when the log stopped. Claiming "benchmark running" off an
  // hour-old line is worse than saying nothing.
  const phase = log.phase;
  const chip = $("phase-chip");
  const stale = global && (snapAge === null || snapAge > STALE_SECS);
  if (sup.between_chunks) {
    chip.textContent = "between chunks · next job queued";
    chip.className = "chip warmup";
  } else if (sup.running === false) {
    chip.textContent = sup.terminal ? `stopped · ${sup.terminal}` : "not running";
    chip.className = "chip idle";
  } else if (stale) {
    chip.textContent = `last seen ${agoText(snap.collected_at)}`;
    chip.className = "chip idle";
  } else {
    // Between iterations (a chunk's start, its bar, a round end's A/B) no iteration runs;
    // the A/B's chip is its last line.
    const between = ["chunk", "rebase", "driver", "round_ab"].includes(phase?.key);
    const abLine = /^\[round-ab\] \S+ /.test(phase?.line || "")
      ? `round-end A/B: ${phase.line.replace(/^\[round-ab\] \S+ /, "").slice(0, 90)}` : null;
    chip.textContent = phase
      ? (typeof log.current_iteration === "number" && !between
          ? `iter ${log.current_iteration}: ${phase.label}` : abLine || phase.label)
      : "unknown";
    chip.className = "chip " +
      (["measuring", "round_ab", "rebase"].includes(phase?.key) ? "measuring"
        : phase?.key === "warmup" ? "warmup"
        : phase?.key === "agent" || phase?.key === "probe" ? "agent"
        : "idle");
  }

  // ---- banners -------------------------------------------------------------
  const banners = [];
  if (stale) {
    banners.push(`<div class="banner warn"><strong>This snapshot was taken ${esc(agoText(snap.collected_at))}.</strong>
      Its collector stopped pushing, or the campaign ended; nothing below is live.</div>`);
  }
  // Only claim the campaign stopped when the supervisor is actually gone. The
  // log is append-only across restarts, so an earlier run's terminal line lingers.
  // What "running" was decided by: a Slurm job's state, or a process on the
  // collector's host (which cannot see a driver inside a job on a GPU node).
  const liveness = sup.liveness || `arbor run process on ${snap.host}`;
  if (sup.between_chunks) {
    banners.push(`<div class="banner good"><strong>Between chunks:</strong> no ${esc(liveness)} is running and the
      next one is queued; it resumes from session.json when it starts. No action needed.</div>`);
  } else if (sup.terminal && sup.running === false) {
    banners.push(`<div class="banner warn"><strong>Campaign stopped:</strong> ${esc(sup.terminal)}.
      Nothing will restart. State is preserved and resumable.</div>`);
  } else if (sup.running === false && log.present) {
    banners.push(`<div class="banner bad"><strong>Not running: no ${esc(liveness)}</strong>, and the log names no end.
      It may have been killed, or it runs where this check cannot see it (a driver inside a Slurm job:
      give the dashboard --slurm-job). State is preserved; relaunch to resume.</div>`);
  } else if (sup.running === null && sup.liveness) {
    banners.push(`<div class="banner warn">Could not check ${esc(liveness)} (see the collector errors):
      whether the campaign runs is unknown.</div>`);
  }
  const mapDoc = c.directions || null;
  if (c.stop_file && c.stop_kind === "chunk") {
    banners.push(`<div class="banner good"><strong>This chunk is ending</strong> (its job's soft stop): the current
      iteration finishes, then the next chunk's job resumes from session.json. No action needed.</div>`);
  } else if (c.stop_file && c.stop_kind === "map") {
    banners.push(`<div class="banner warn"><strong>Stopped for the operator by the map:</strong>
      ${esc(mapDoc?.stop?.reason || "see MAP_STOP.md")}. Read MAP_STOP.md in the session before resuming.</div>`);
  } else if (c.stop_file && c.stop_kind === "round") {
    banners.push(`<div class="banner warn"><strong>Stopped for the operator at a round end:</strong>
      ${esc(c.round_end?.stop?.reason || "see ROUND_STOP.md")}. Read ROUND_STOP.md in the session before resuming.</div>`);
  } else if (c.stop_file && c.stop_kind === "lane") {
    banners.push(`<div class="banner good"><strong>The lane is closed:</strong>
      ${esc(mapDoc?.stop?.reason || "every item has its verdict")}. Its record is LANE_CLOSED.json in the session.</div>`);
  } else if (c.stop_file) {
    banners.push(`<div class="banner warn"><strong>STOP file present.</strong> The campaign finishes the current iteration, then exits.</div>`);
  }
  // A preempted job is requeued under its own id and starts over on another node.
  for (const j of (snap.jobs?.ours || []).filter((x) => x.restarts > 0)) {
    banners.push(`<div class="banner warn"><strong>Job ${esc(j.id)} was preempted and requeued ${esc(j.restarts)}
      time${j.restarts === 1 ? "" : "s"}</strong> (${esc(String(j.state).toLowerCase())}${j.node ? ` on ${esc(j.node)}` : ""}).
      Each start sets up on a new node before the driver resumes from session.json; the campaign log
      last moved ${esc(agoText(log.log_mtime))}.</div>`);
  }
  // A round end (arbor/rounds.py): a reached target or an explored map, A/B first. Its
  // attempts are the A/B panel's (realtime.py collect_round_ab); one a preemption cut
  // short has no verdict.
  const roundPending = c.round_end?.pending;
  const abTries = c.round_ab || [];
  if (roundPending && !c.stop_file) {
    const cut = abTries.filter((a) => !a.verdict).length - 1;
    banners.push(`<div class="banner good"><strong>Round end ${esc(roundPending.end)}</strong>
      (${esc(roundPending.trigger)}): ${esc(roundPending.detail)}. A same-node A/B of the campaign's base
      against the current tree runs before the next iteration${abTries.length
        ? ` (attempt ${abTries.length}${cut > 0 ? `, after ${cut} cut short` : ""}: see the A/B panel)` : ""}.</div>`);
  }
  if (mapDoc?.phase === "updating") {
    const lastRound = (mapDoc.rounds || []).slice(-1)[0];
    const why = lastRound?.ended_by === "target" ? "ended at its target" : "closed every direction";
    banners.push(`<div class="banner good"><strong>Dossier update round in progress</strong> (iteration
      ${esc(mapDoc.phase_iters)}): map round ${esc(mapDoc.map_round)} ${why}, so the campaign is
      re-profiling and re-ranking before it continues on a new map.</div>`);
  }
  // An update round's map with nothing open starts afresh, then takes a rewrite (arbor/directions.py).
  if (mapDoc?.phase === "fresh") {
    banners.push(`<div class="banner good"><strong>Fresh start in progress</strong> (iteration
      ${esc(mapDoc.phase_iters)}): the last map had no open increment, so the campaign is writing a new
      dossier as a new campaign would: the code read again, a new profile, a new map.</div>`);
  }
  const rewriteOpen = mapDoc?.phase === "active" && (mapDoc.directions || [])
    .find((x) => x.id === mapDoc.rewrite && x.status === "open");
  if (rewriteOpen) {
    banners.push(`<div class="banner good"><strong>Rewrite round:</strong> the campaign carries out
      <code>${esc(rewriteOpen.id)}</code> (${esc(rewriteOpen.title || "")}), the top planned rewrite,
      after a fresh start found no increment.</div>`);
  }
  if (c.chain_stop) banners.push(`<div class="banner warn"><strong>CHAIN_STOP present.</strong> No further chunk will start.</div>`);
  if (c.aborted_file) banners.push(`<div class="banner bad"><strong>ABORTED.md present.</strong> Arbor halted itself; read the file before relaunching.</div>`);
  if (sup.restarts > 0) banners.push(`<div class="banner warn">Supervisor restarted Arbor <strong>${sup.restarts}</strong> time(s) across ${sup.attempts} attempt(s) this run. Arbor resumes from session.json, so iterations are not lost.</div>`);
  if (sup.runs_in_log > 1) banners.push(`<div class="banner good">This is supervisor run <strong>${sup.runs_in_log}</strong> in the log. Counts above are scoped to the current run.</div>`);
  if (names.share_ok === false) {
    banners.push(`<div class="banner warn"><strong>The collector is not reaching the share</strong>${names.pushed_at ? ` (last push ${esc(agoText(names.pushed_at))})` : ""}:
      ${esc(names.last_error || "unknown error")}. It keeps writing its local copy.</div>`);
  }
  if (!c.present) banners.push(`<div class="banner bad"><strong>No session.json</strong> in ${esc(c.session_root || snap.session_dir)}.</div>`);
  if (meta.error) banners.push(`<div class="banner bad"><strong>Collector:</strong> ${esc(meta.error)} — showing the last good snapshot.</div>`);
  $("banner-area").innerHTML = banners.join("");

  // ---- KPIs ----------------------------------------------------------------
  const unit = c.metric_unit || c.metric_name || "";
  const blEntries = Object.entries(snap.baselines || {}).filter(([k]) => !/replay/.test(k));
  const bl = blEntries[0]?.[1];
  $("kpi-baseline").textContent = num(c.baseline);
  $("kpi-baseline-foot").textContent = bl ? `${unit} · spread ${pct(bl.relative_spread)} (${blEntries[0][0]})` : unit || "—";

  // A rolling campaign rebases its bar on each chunk's node. Its numbers are shown on
  // the first node's scale (where baseline and target were set), chained across the
  // rebases, with this node's own reading beside them: a slower node is not a regression.
  // Each lesson carries the factors in force when it was written (arbor/rebaseline.py);
  // the newest one that has them is the current node's.
  const rebases = (c.counts || {}).REBASELINE || 0;
  const stamped = (c.lessons || []).find((l) => typeof l.scale === "number");
  const rebased = rebases > 0 && !!stamped;
  // The collector's warm chain (realtime.py warm_chain): each rebased bar read on its
  // replays after the container's first, which reads low, as the candidates run. The
  // driver's chain (the lessons' `scale`) counted that first replay: Model M day-0 b0's
  // read 1.438x its Day-0 bar where its round-end A/B measured 1.24x.
  const warm = rebased && typeof c.warm_scale === "number" && typeof c.warm_best === "number";
  const scale = rebased ? (warm ? c.warm_scale : stamped.scale) : 1;
  const scale2 = rebased && typeof stamped.secondary_scale === "number" ? stamped.secondary_scale : 1;
  const times = (v, f) => typeof v === "number" ? v * f : v;
  const gainPct = (best, base, fallback) => rebased
    ? (typeof best === "number" && base ? (best / base - 1) * 100 : null) : fallback;
  const ownBest = warm ? c.warm_best : c.best_metric;
  const bestShown = times(ownBest, scale);
  const driverBest = rebased ? times(c.best_metric, stamped.scale) : null;
  const driverDiffers = warm && typeof driverBest === "number" && Math.abs(driverBest / bestShown - 1) > 0.02;
  const secBest = times(c.best_secondary, scale2);
  const gain = gainPct(bestShown, c.baseline, c.gain_pct);
  const secGain = gainPct(secBest, c.baseline_secondary, c.secondary_gain_pct);
  const onNode = (v, d) => rebased ? ` · ${num(v, d)} on this node` : "";
  $("kpi-best").textContent = num(bestShown);
  $("kpi-best-foot").textContent = (typeof gain === "number"
    ? `${gain >= 0 ? "+" : ""}${gain.toFixed(2)}% vs baseline`
    : unit || "—") + onNode(ownBest)
    + (rebased ? ` · chained across ${rebases} rebase${rebases === 1 ? "" : "s"}${warm ? " on warm replays" : ""}` : "")
    + (driverDiffers ? ` · the driver's chain: ${num(driverBest)}` : "");

  // The second Pareto axis (a serving campaign's p90 interactivity), when the
  // session declares one: a gain bought by spending it is a frontier point.
  const sec = c.secondary_name;
  const secLabel = sec ? sec.replace(/_/g, " ") : "";
  $("kpi-secondary-card").hidden = !sec;
  if (sec) {
    $("kpi-secondary-label").textContent = `${secLabel} at best`;
    $("kpi-secondary").textContent = num(secBest, 2);
    const better = c.secondary_direction === "min" ? "lower" : "higher";
    $("kpi-secondary-foot").textContent = (typeof secGain === "number"
      ? `${secGain >= 0 ? "+" : ""}${secGain.toFixed(2)}% vs baseline ${num(c.baseline_secondary, 2)} · ${better} is better`
      : `baseline ${num(c.baseline_secondary, 2)} · ${better} is better`) + onNode(c.best_secondary, 2);
  }

  // The in-flight iteration is what someone actually wants to know. It is NOT
  // session.json's `iteration`, which is the last one the gate counted: a
  // not-measured iteration leaves that number behind while work continues.
  const cur = log.current_iteration;
  const booked = c.iteration;
  const inFlight = sup.running !== false && !stale
    && typeof cur === "number" && (booked == null || cur > booked);
  $("kpi-iter").innerHTML = inFlight
    ? `${cur} <span class="running-badge">running</span>`
    : `${esc(booked ?? cur ?? "—")}${sup.running ? "" : " <span class=\"muted\">idle</span>"}`;
  const bits = [];
  if (inFlight) bits.push(`last booked ${booked ?? "none"}`);
  bits.push(`tier ${c.tier ?? "—"}`);
  if (log.not_measured_count) bits.push(`${log.not_measured_count} not measured`);
  if (c.null_streak) bits.push(`null streak ${c.null_streak}`);
  if (log.iteration_dir_mtime && inFlight) bits.push(`started ${agoText(log.iteration_dir_mtime)}`);
  $("kpi-iter-foot").textContent = bits.join(" · ");

  const counts = c.counts || {};
  $("kpi-verdicts").textContent = `${counts.KEEP || 0} / ${counts.REVERT || 0}`;
  $("kpi-verdicts-foot").textContent = `KEEP / REVERT${counts.PROBE ? ` · ${counts.PROBE} probe` : ""}${counts.INVALID ? ` · ${counts.INVALID} invalid` : ""}`;

  // ---- progress ------------------------------------------------------------
  // Scale runs from 0 to the target so the baseline mark shows how much of the
  // gap was already closed before the campaign started. No target, no bar:
  // a made-up one reads like a commitment.
  const tg = snap.target || {};
  $("progress-panel").hidden = typeof tg.value !== "number";
  if (typeof tg.value === "number") {
    const target = tg.value;
    const targetNote = tg.set_by === "operator" ? "set by the operator"
      : tg.set_by === "round" ? `set by ${tg.source || "a round-end A/B"}` : "from the campaign config";
    const driverNote = (typeof tg.driver === "number" && tg.driver !== target)
      ? ` · <span class="muted">the running driver still uses ${num(tg.driver)} until it restarts</span>` : "";
    const base = c.baseline || 0;
    const best = bestShown || base;
    $("progress-fill").style.width = `${Math.min(100, (best / target) * 100)}%`;
    $("progress-baseline").style.left = `${Math.min(100, (base / target) * 100)}%`;
    $("progress-legend").innerHTML =
      `baseline <strong>${num(base)}</strong> → best <strong>${num(best)}</strong> ` +
      `· target <strong>${num(target)}</strong> ${esc(unit)} (${targetNote}) ` +
      `· gap remaining <strong>${num(target - best, 2)}</strong>${driverNote}` +
      (driverDiffers ? `<br><span class="muted">The driver's chain reads ${num(driverBest)}: it counts each ` +
        `rebased bar's first replay in a fresh container, which reads low. Its target check uses that ` +
        `number; a round end's A/B measures the gain on one node.</span>` : "") +
      (sec ? `<br>${esc(secLabel)}: baseline <strong>${num(c.baseline_secondary, 2)}</strong> → at best ` +
        `<strong>${num(secBest, 2)}</strong>` +
        (typeof tg.secondary_value === "number" ? ` · target <strong>${num(tg.secondary_value, 2)}</strong>` : "") : "");
  }

  // ---- directions: the map's checklist (arbor/directions.py) ---------------
  // Which of the dossier's directions this map round has explored. A direction
  // is explored when the agent closes it; "worked" is every iteration whose DONE
  // line cited it. When all are closed, the campaign re-profiles and re-ranks.
  const mapDirs = (mapDoc && mapDoc.directions) || [];
  $("map-panel").hidden = !mapDoc;
  if (mapDoc) {
    const closedSet = DIR_CLOSED;
    const nClosed = mapDirs.filter((d) => closedSet.has(d.status)).length;
    const phaseText = { bootstrap: "the map is being written", unmapped: "no map yet", active: "in progress",
                        verifying: "round end: the A/B runs",
                        updating: "update round: re-profiling and re-ranking", stopped: "stopped for the operator",
                        fresh: "fresh start: a new dossier from the code and a new profile",
                        closed: "closed: every item has its verdict" };
    // The wall clock of what is running: the round, or the update round.
    const since = (t) => typeof t === "number" ? Date.now() / 1000 - t : null;
    const clock = mapDoc.phase === "active" && typeof mapDoc.round_started_ts === "number"
      ? ` · ${durText(since(mapDoc.round_started_ts))} in this round`
      : mapDoc.phase === "updating" && typeof mapDoc.phase_started_ts === "number"
        ? ` · ${durText(since(mapDoc.phase_started_ts))} so far` : "";
    const lane = mapDoc.scope === "lane" ? (mapDoc.lane || {}) : null;
    $("map-round").textContent = (lane
      ? `· lane ${lane.direction || ""} (${lane.kind || "kernel-plan"}) · ${nClosed} of ${mapDirs.length} items closed · ${phaseText[mapDoc.phase] || mapDoc.phase}`
      : mapDirs.length
        ? `· round ${mapDoc.map_round} · ${nClosed} of ${mapDirs.length} explored · ${phaseText[mapDoc.phase] || mapDoc.phase}`
        : `· ${phaseText[mapDoc.phase] || mapDoc.phase}`) + clock;
    const rounds = (mapDoc.rounds || []).map((r) =>
      `round ${esc(r.round)}: iterations ${esc(r.started)}–${esc(r.ended)}, ${esc(r.keeps)} KEEP${r.keeps === 1 ? "" : "s"}` +
      (typeof r.wall_s === "number" ? `, ${esc(durText(r.wall_s))}` : "") +
      (r.ended_by === "target" ? ", ended at the target" : "")).join(" · ");
    const stray = (mapDoc.unmapped_ids || []).length
      ? `<p class="muted">Cited but not on the map: ${mapDoc.unmapped_ids.map((t) => `<code>${esc(t)}</code>`).join(", ")}</p>` : "";
    // A lane's stage records (arbor/stages.py): each item's attempts, as the harness judged them.
    const st = c.stages || null;
    const stageRows = st ? mapDirs.map((d) => {
      const tries = ((st.items || {})[d.id] || {}).attempts || [];
      const last = tries[tries.length - 1];
      return `<tr><td><code>${esc(d.id)}</code></td><td>${esc(d.stage || "")}</td>
        <td><span class="dstatus ${esc(d.status || "open")}">${esc((d.status || "open").replace("_", " "))}</span></td>
        <td class="num">${tries.length}</td>
        <td>${last ? `iteration ${esc(last.iteration)}: ${esc(last.declared || "verdict")} → <strong>${esc(last.harness)}</strong>` +
          ((last.why || []).length ? `<div class="muted">${esc(last.why.join("; ").slice(0, 160))}</div>` : "") : "—"}</td></tr>`;
    }).join("") : "";
    const waiting = st && (st.pending || []).length
      ? `<p class="muted">Waiting to be judged: ${st.pending.map((p) => `<code>${esc(p.item)}</code> ${esc(p.result)} (iteration ${esc(p.iteration)})`).join(", ")}</p>` : "";
    const stagesTable = st
      ? `<h3>Stages</h3><table><thead><tr><th>item</th><th>stage</th><th>status</th><th class="num">attempts</th>` +
        `<th>last attempt</th></tr></thead><tbody>${stageRows}</tbody></table>${waiting}` : "";
    $("map").innerHTML = mapDirs.length
      ? mapTableHtml(mapDirs) + stray + (rounds ? `<p class="muted">Earlier rounds: ${rounds}</p>` : "") + stagesTable
      : `<p class="muted">No map yet: the agent is asked to write the dossier's directions as a map ` +
        `(iteration ${esc(mapDoc.phase_iters)} of the bootstrap).</p>`;
  }

  // ---- history -------------------------------------------------------------
  const lessons = c.lessons || [];
  if (lessons.length) {
    // On the first node's scale once the bar has been rebased; the node's own value on
    // hover. A row written before lessons carried their factor shows its node's value.
    const cell = (own, f, d) => (rebased && typeof own === "number" && typeof f === "number")
      ? `<span title="${esc(num(own, d))} on its node">${num(own * f, d)}</span>` : num(own, d);
    const rows = lessons.map((l) => {
      const d = String(l.decision || "?");
      return `<tr>
        <td class="num">${esc(l.iteration ?? "")}</td>
        <td><span class="verdict ${esc(d)}">${esc(d)}</span></td>
        <td class="num">${cell(l.warm_metric ?? l.metric, l.warm_scale ?? l.scale)}</td>
        ${sec ? `<td class="num">${l.metric ? cell(l.secondary, l.secondary_scale, 2) : "—"}</td>` : ""}
        <td>${esc(l.tier ?? "")}</td>
        <td>${esc((l.change || "").slice(0, 150))}</td>
      </tr>`;
    }).join("");
    $("history").innerHTML =
      (rebased ? `<p class="muted">Numbers are on the first node's scale, chained across ${rebases} ` +
        `rebase${rebases === 1 ? "" : "s"} of the bar${warm ? ", each bar read on its warm replays (a container's " +
          "first replay reads low and is left out)" : ""}; hover for the value on the node that measured it.</p>` : "") +
      `<table><thead><tr><th class="num">#</th><th>verdict</th><th class="num">${esc(unit || "metric")}</th>` +
      `${sec ? `<th class="num">${esc(secLabel)}</th>` : ""}<th>tier</th><th>change</th></tr></thead><tbody>${rows}</tbody></table>`;
  } else {
    $("history").innerHTML = `<p class="muted">No iterations booked yet. Verdicts appear here once the gate decides.</p>`;
  }

  // ---- node ----------------------------------------------------------------
  // The realtime collector never samples GPU nodes (a probe during a measured
  // replay perturbs it), so this panel only shows when a snapshot carries node
  // data, as the Cluster A collector's did.
  const node = snap.node || {};
  const gpus = node.gpus || [];
  const containers = node.containers || [];
  const jobs = snap.jobs?.ours || [];
  const showNode = gpus.length > 0 || containers.length > 0 || jobs.length > 0;
  $("node-panel").hidden = !showNode;
  $("history-panel").classList.toggle("wide", !showNode);
  if (showNode) {
    const campaignJob = jobs.find((j) => j.role === "campaign");
    $("node-name").textContent = campaignJob?.node || "—";
    $("gpus").innerHTML = gpus.length
      ? gpus.map((g) => {
          const vram = g.vram_pct ?? 0;
          const serving = g.id <= 3;
          const cls = vram > 2 ? (serving ? "serving" : "busy") : "";
          return `<div class="gpu ${cls}"><div class="id">GPU ${esc(g.id)}${serving ? " · TP" : ""}</div>
            <div class="pct">${esc(vram)}%</div><div class="use">use ${esc(g.use_pct ?? "—")}%</div></div>`;
        }).join("")
      : `<p class="muted">${esc(node.error || "no GPU data")}</p>`;
    $("containers").innerHTML = containers.length
      ? `<table><tbody>${containers.map((x) =>
          `<tr><td>${esc(x.name)}</td><td class="muted">${esc(x.status)}</td></tr>`).join("")}</tbody></table>`
      : `<p class="muted">none</p>`;
    $("jobs").innerHTML = jobs.length
      ? `<table><thead><tr><th>job</th><th>role</th><th>node</th><th class="num">elapsed</th><th class="num">limit</th></tr></thead><tbody>${
          jobs.map((j) => `<tr><td>${esc(j.id)}</td><td>${esc(j.role)}</td><td>${esc(j.node || j.reason)}</td>
            <td class="num">${esc(j.elapsed)}</td><td class="num">${esc(j.limit)}</td></tr>`).join("")
        }</tbody></table>`
      : `<p class="muted">no jobs found</p>`;
  }

  // ---- baselines -----------------------------------------------------------
  const blRows = blEntries.map(([k, b]) => `<tr>
      <td>${esc(k)}${b.concurrency ? ` <span class="muted">C${esc(b.concurrency)}</span>` : ""}</td>
      <td class="num">${num(b.mean)}</td>
      <td class="num">${pct(b.relative_spread)}</td>
      <td class="num">${num(b.mean_secondary, 2)}</td>
      <td class="num">${num(b.min_credible_gain, 4)}</td>
      <td class="muted">${(b.runs || []).map((r) => num(r, 1)).join(", ")}</td>
    </tr>`).join("");
  $("baselines").innerHTML = blRows
    ? `<table><thead><tr><th>record</th><th class="num">mean</th><th class="num">spread</th>
       <th class="num">secondary</th><th class="num">min credible</th><th>replays</th></tr></thead><tbody>${blRows}</tbody></table>`
    : `<p class="muted">no baseline records (the collector reads them with --results)</p>`;

  // ---- A/B -----------------------------------------------------------------
  // Not every campaign has a validation A/B; an empty panel is noise.
  const ab = snap.ab || {};
  const abJob = jobs.find((j) => j.role === "ab");
  const abSection = $("ab")?.closest(".panel");
  const hasAb = Boolean(ab.job || ab.results_present || ab.report_present || abJob || abTries.length);
  if (abSection) abSection.hidden = !hasAb;
  $("baselines-panel").classList.toggle("wide", !hasAb);
  if (hasAb) $("ab").innerHTML = abTries.length ? renderRoundAb(abTries, c, jobs) : renderAb(ab, abJob);

  // ---- logs ----------------------------------------------------------------
  const ev = snap.events || [];
  $("events-panel").hidden = ev.length === 0;
  const evEl = $("events");
  const evPinned = evEl.scrollTop + evEl.clientHeight >= evEl.scrollHeight - 30;
  evEl.textContent = ev.slice().reverse().join("\n");
  if (evPinned) evEl.scrollTop = 0;

  $("agent-log").textContent = (log.agent_tail || []).join("\n") || "—";
  const tailEl = $("log-tail");
  const pinned = tailEl.scrollTop + tailEl.clientHeight >= tailEl.scrollHeight - 30;
  tailEl.textContent = (log.tail || []).join("\n")
    || (log.present ? "—" : "no campaign log (the collector reads it with --log)");
  if (pinned) tailEl.scrollTop = tailEl.scrollHeight;

  renderErrors(snap.errors, share.errors);
}

function renderAb(ab, abJob) {
  const parts = [
    `<dl class="kv">
      <dt>job</dt><dd>${esc(ab.job)} ${abJob ? `· ${esc(abJob.state)} on ${esc(abJob.node)}` : "· not running"}</dd>
      <dt>replays</dt><dd>${esc(ab.replay_count ?? 0)} recorded${ab.run_dirs ? ` · ${ab.run_dirs.length} run dirs` : ""}</dd>
      <dt>report</dt><dd>${ab.report_present ? `present · ${esc(ab.report_mtime || "")}` : "not yet"}</dd>
    </dl>`,
  ];
  const bop = ab.summary?.by_operating_point;
  if (bop && typeof bop === "object") {
    const rows = Object.entries(bop).map(([point, v]) => {
      // Two producers nest the arms differently: under `arms.A` or directly under `A`.
      const armA = v?.arms?.A ?? v?.A ?? {};
      const armB = v?.arms?.B ?? v?.B ?? {};
      const at = armA.output_throughput || {};
      const bt = armB.output_throughput || {};
      const ratio = (typeof at.mean === "number" && typeof bt.mean === "number" && at.mean) ? bt.mean / at.mean : null;
      // A ratio from one replay per arm is not a result yet; label it.
      const thin = (at.n ?? 0) < 2 || (bt.n ?? 0) < 2;
      return `<tr><td>${esc(point)}${thin ? ' <span class="muted">(partial)</span>' : ""}</td>
        <td class="num">${num(at.mean)}</td><td class="num">${num(bt.mean)}</td>
        <td class="num"><strong>${typeof ratio === "number" ? ratio.toFixed(4) + "×" : "—"}</strong></td>
        <td class="num">${esc(at.n ?? 0)}/${esc(bt.n ?? 0)}</td><td>${esc(v?.verdict ?? "")}</td></tr>`;
    }).join("");
    parts.push(`<h3>Arms by operating point</h3><table><thead><tr><th>point</th><th class="num">A</th>
      <th class="num">B</th><th class="num">B/A</th><th class="num">n A/B</th><th>verdict</th></tr></thead><tbody>${rows}</tbody></table>`);
  }
  return parts.join("");
}

// A round end's A/B (scripts/model-m-campaign/round_ab.py), newest attempt first: per round,
// the base and current arms at each point, the campaign's own (judged) point first. A row
// is [round, arm, concurrency, metric, secondary]; the current arm's full GSM8K is a dict.
function renderRoundAb(tries, c, jobs) {
  const running = jobs.find((j) => j.role === "campaign" && j.state === "RUNNING");
  const end = c.round_end?.pending?.end ?? (c.round_end?.ends || []).slice(-1)[0]?.end;
  const sec = (c.secondary_name || "secondary").replace(/_/g, " ");
  const head = `<p class="muted">Round end ${esc(end ?? "")}: the campaign's base tree against its current ` +
    `tree on one node, the arms alternating. Round 0 warms both up and is not judged.</p>`;
  return head + tries.slice().reverse().map((a, i) => {
    const job = a.dir.split(".")[0];
    const live = !a.verdict && running?.id === job;
    const node = a.node || (live ? running.node : "");
    const gsm = a.rows.find((r) => !Array.isArray(r));
    const own = a.rows.find(Array.isArray)?.[2];
    const pts = {};
    for (const r of a.rows.filter(Array.isArray)) {
      const [round, arm, conc, v, s] = r;
      (pts[`${round}|${conc}`] ||= { round, conc })[arm] = { v, s };
    }
    const ok = (p) => typeof p.base?.v === "number" && typeof p.current?.v === "number" && p.base.v;
    const list = Object.values(pts).sort((x, y) =>
      x.round - y.round || (x.conc === own ? -1 : y.conc === own ? 1 : x.conc - y.conc));
    const rows = list.map((p) => `<tr${p.conc === own ? "" : ' class="muted"'}>
        <td class="num">${esc(p.round)}${p.round === 0 ? " warm-up" : ""}</td><td>C${esc(p.conc)}</td>
        <td class="num">${num(p.base?.v)}</td><td class="num">${num(p.current?.v)}</td>
        <td class="num"><strong>${ok(p) ? (p.current.v / p.base.v).toFixed(4) + "×" : "—"}</strong></td>
        <td class="num">${num(p.base?.s, 2)}</td><td class="num">${num(p.current?.s, 2)}</td></tr>`).join("");
    const pairs = list.filter((p) => p.round >= 1 && p.conc === own && ok(p));
    const mean = (arm) => pairs.reduce((t, p) => t + p[arm].v, 0) / pairs.length;
    const sofar = pairs.length
      ? `<p>${pairs.length} measured pair${pairs.length === 1 ? "" : "s"} at C${esc(own)}: current/base ` +
        `<strong>${(mean("current") / mean("base")).toFixed(4)}×</strong>.</p>` : "";
    return `<h3>Attempt ${tries.length - i} · job ${esc(job)}${node ? ` on ${esc(node)}` : ""} · ` +
      `${esc(a.verdict || (live ? "running" : "cut short"))}` +
      `${gsm ? ` · GSM8K ${num(gsm.score, 3)} ${gsm.passed ? "passed" : "failed"}` : ""}</h3>` + sofar +
      `<table><thead><tr><th class="num">round</th><th>point</th><th class="num">base</th>` +
      `<th class="num">current</th><th class="num">current/base</th><th class="num">${esc(sec)} base</th>` +
      `<th class="num">${esc(sec)} current</th></tr></thead><tbody>${rows}</tbody></table>`;
  }).join("");
}

function renderErrors(snapErrors, shareErrors) {
  const errs = [...(snapErrors || []), ...(shareErrors || []).map((e) => `share: ${e}`)];
  $("errors-panel").hidden = errs.length === 0;
  $("errors").innerHTML = errs.map((e) => `<div class="banner warn">${esc(e)}</div>`).join("");
}

// ---- knowledge (global mode) -------------------------------------------------

function tabLabel(p) {
  const m = p.match(/^knowledge\/profiles\/([^/]+)\//);
  if (m) return `Profile · ${m[1]}`;
  return { "knowledge/summary/SUMMARY.md": "Summary", "knowledge/outcomes/OUTCOMES.md": "Outcomes",
           "knowledge/findings/FINDINGS.md": "Findings", "knowledge/dossier/README.md": "Dossier index" }[p] || p;
}

function renderKnowledge(k) {
  $("knowledge-panel").hidden = !k;
  if (!k) return;
  const texts = k.texts || {};
  const order = ["knowledge/summary/SUMMARY.md", "knowledge/outcomes/OUTCOMES.md", "knowledge/findings/FINDINGS.md"];
  const paths = [...order.filter((p) => p in texts),
                 ...Object.keys(texts).filter((p) => !order.includes(p)).sort()];
  if (!paths.includes(knowledgeTab)) knowledgeTab = paths[0] || null;
  $("knowledge-when").textContent = k.manifest
    ? `(collected ${agoText(k.manifest.collected_at)} on ${k.manifest.host || "?"})` : "(no manifest)";
  $("knowledge-tabs").innerHTML = paths.map((p) =>
    `<button type="button" class="tab${p === knowledgeTab ? " on" : ""}" data-path="${esc(p)}">${esc(tabLabel(p))}</button>`).join("");
  $("knowledge-text").textContent = knowledgeTab ? texts[knowledgeTab] : "no summary, outcomes or findings were collected";
  const files = k.files || [];
  $("knowledge-files").innerHTML = files.length
    ? `<table><thead><tr><th>file</th><th class="num">size</th><th>written</th></tr></thead><tbody>${
        files.map((f) => `<tr><td><a href="api/file?path=${encodeURIComponent(f.path)}" target="_blank" rel="noopener">${esc(f.path.replace(/^knowledge\//, ""))}</a>${
          f.in_manifest ? "" : ' <span class="muted" title="not in the latest manifest">(from an earlier collection)</span>'}</td>
          <td class="num">${bytes(f.bytes)}</td><td class="muted">${esc(agoText(f.mtime))}</td></tr>`).join("")
      }</tbody></table>`
    : `<p class="muted">none</p>`;
}

let last = null;

$("knowledge-tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-path]");
  if (!b) return;
  knowledgeTab = b.dataset.path;
  if (last) render(last);
});

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
setInterval(tick, 6000);
