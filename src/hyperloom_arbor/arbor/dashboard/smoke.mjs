#!/usr/bin/env node
// Render a page's script against a payload in a stub DOM, and print what it
// drew. A classic-script error leaves every value at "—" with nothing said;
// this makes it throw instead.
//
//   node smoke.mjs campaign <payload.json>   public/app.js, an api/status payload
//   node smoke.mjs index    <payload.json>   public/global.js, an api/index payload
//   node smoke.mjs parallel <payload.json>   public/parallel.js, a parallel run's api/status payload
//
// Used by arbor/tests/test_dashboard.py; exits non-zero on any exception.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const [kind, payloadPath] = process.argv.slice(2);
// Each page's scripts, in the order its HTML loads them.
const scripts = { campaign: ["directions.js", "app.js"], index: ["global.js"],
                  parallel: ["directions.js", "parallel.js"] }[kind];
if (!scripts || !payloadPath) {
  console.error("usage: node smoke.mjs campaign|index|parallel <payload.json>");
  process.exit(2);
}
const payload = JSON.parse(readFileSync(payloadPath, "utf8"));

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      id, innerHTML: "", textContent: "", className: "", hidden: false, value: "",
      style: {}, dataset: {}, scrollTop: 0, clientHeight: 0, scrollHeight: 0,
      classList: { toggle() {}, add() {}, remove() {} },
      closest() { return element(`${id}:panel`); },
      addEventListener() {},
    });
  }
  return elements.get(id);
}

const context = vm.createContext({
  document: { getElementById: element, querySelector: (s) => element(s), title: "" },
  location: { search: "", pathname: "/" },
  history: { replaceState() {} },
  URLSearchParams,
  fetch: async () => ({ json: async () => payload }),
  setInterval: () => 0,
  console,
});
for (const script of scripts) {
  vm.runInContext(readFileSync(path.join(here, "public", script), "utf8"), context, { filename: script });
}
// Render directly, so an exception surfaces instead of becoming the page's
// "server unreachable" line.
context.render(payload);

const ids = {
  campaign: ["phase-chip", "freshness", "identity", "kpi-baseline", "kpi-best", "kpi-best-foot", "kpi-iter",
             "kpi-verdicts", "banner-area", "history", "knowledge-tabs", "knowledge-text", "knowledge-files",
             "progress-legend", "map", "map-round", "ab"],
  index: ["where", "freshness", "banner-area", "counts", "projects"],
  parallel: ["identity", "phase-chip", "freshness", "kpi-round", "kpi-round-foot", "kpi-base", "kpi-nodes",
             "kpi-nodes-foot", "kpi-kept", "kpi-kept-foot", "banner-area", "lanes-sub", "timeline", "lanes",
             "map-round", "mapstep", "map", "integration", "rounds", "events"],
}[kind];
const out = { title: context.document.title };
for (const id of ids) {
  const e = element(id);
  out[id] = { text: e.textContent, html: e.innerHTML, hidden: e.hidden, className: e.className };
}
for (const id of ["progress-panel", "node-panel", "knowledge-panel", "events-panel", "errors-panel", "map-panel"]) {
  out[id] = { hidden: element(id).hidden };
}
console.log(JSON.stringify(out));
