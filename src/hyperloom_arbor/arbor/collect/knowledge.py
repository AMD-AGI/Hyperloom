"""Knowledge collector: what a campaign learned, in a form someone else can reuse.

Stages one directory per type, pushed to ``knowledge/<type>/`` on the share:

    summary    SUMMARY.md and manifest.json (every file, its size and sha256, what was skipped)
    outcomes   outcomes.jsonl and OUTCOMES.md: each iteration kept, reverted, probed or
               not measured, with what changed and why
    findings   findings.jsonl and FINDINGS.md: the ledger's current facts, assumptions,
               unknowns and dead ends
    profiles   each profile's report (what was profiled, what it found) and its small
               summary files; never raw traces
    dossier    the dossier the campaign worked from
    artifacts  session.json, ledger.jsonl, landed.patch, ABORTED.md, REPORT.md, every
               iteration's patch.diff, and the bench output of the iterations kept

Files over MAX_FILE_BYTES are listed in the manifest and left out.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
from pathlib import Path

from arbor.collect.realtime import now_iso

TYPES = ("summary", "outcomes", "findings", "profiles", "dossier", "artifacts")
MAX_FILE_BYTES = 5_000_000
# What a profile directory contributes: its report and the small files a reader
# needs to check it. Raw records and traces stay where they were made.
PROFILE_FILES = ("PROFILE.md", "profile.md", "profile.json", "census_summary.json",
                 "LIMITATIONS.md", "*_check.json", "*.csv")
ANNOTATIONS = ("BASELINE", "ESCALATE", "NOTE", "REBASELINE", "MAP_ROUND", "ROUND_END", "ROUND_AB")


def outcome_kind(lesson: dict) -> str | None:
    """keep, revert, probe or not_measured; None for a row no gate produced."""
    decision = str(lesson.get("decision") or "").upper()
    verdict = str(lesson.get("verdict") or "").upper()
    intent = str(lesson.get("intent") or "").upper()
    if decision in ANNOTATIONS:
        return None
    if verdict == "NOT_MEASURED" or decision == "INVALID":
        return "not_measured"
    if verdict == "REVERT" or decision == "REVERT":
        return "revert"
    if intent == "LEARN" or decision == "PROBE":
        return "probe"
    if verdict == "KEEP" or decision == "KEEP":
        return "keep"
    return None


def outcomes(lessons: list[dict]) -> list[dict]:
    rows = []
    for lesson in lessons:
        kind = outcome_kind(lesson)
        if kind is None:
            continue
        findings = [str(f) for f in (lesson.get("findings") or [])]
        why = str(lesson.get("reason") or "").strip() or "; ".join(findings)
        if not why and kind == "keep":
            why = ("a frontier point (one axis traded for the other)" if lesson.get("frontier")
                   else "moved the bar" if lesson.get("booked", True)
                   else "kept on evidence without moving the bar")
        rows.append({
            "iteration": lesson.get("iteration"),
            "kind": kind,
            "intent": lesson.get("intent") or lesson.get("claim_class"),
            "what": str(lesson.get("change") or "").strip(),
            "why": why,
            "metric": lesson.get("metric"),
            "secondary": lesson.get("secondary"),
            "tier": lesson.get("tier"),
            "findings": findings,
        })
    return rows


_SECTIONS = (("keep", "Kept"), ("revert", "Reverted"), ("probe", "Probes"),
             ("not_measured", "Not measured (nothing learned about the change)"))


def render_outcomes(rows: list[dict], title: str) -> str:
    lines = [f"# Outcomes: {title}", ""]
    lines.append(", ".join(f"{label.split(' (')[0].lower()} {sum(r['kind'] == k for r in rows)}"
                           for k, label in _SECTIONS))
    for kind, label in _SECTIONS:
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        lines += ["", f"## {label} ({len(group)})", ""]
        for r in group:
            metric = f", {r['metric']}" if r["metric"] not in (None, 0, 0.0) else ""
            lines.append(f"- **Iteration {r['iteration']}** ({r['intent']}{metric}): {r['what'] or '(no change text)'}")
            lines.append(f"  Why: {r['why'] or '(no reason recorded)'}")
    return "\n".join(lines) + "\n"


def current_findings(ledger: Path) -> list[dict]:
    """The ledger's current view: a later record with the same id supersedes."""
    latest: dict[str, dict] = {}
    if not ledger.is_file():
        return []
    for line in ledger.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("id"):
            latest[rec["id"]] = rec
    return list(latest.values())


def render_findings(records: list[dict], title: str) -> str:
    lines = [f"# Findings: {title}", ""]
    for kind in ("FACT", "DEAD_END", "ASSUMPTION", "UNKNOWN"):
        group = [r for r in records if r.get("type") == kind]
        if not group:
            continue
        lines += [f"## {kind} ({len(group)})", ""]
        for r in group:
            lines.append(f"- **{r.get('id')}** [{r.get('status', 'open')}]: {r.get('claim', '')}")
            for key in ("evidence", "falsifier", "revisit_if"):
                if r.get(key):
                    lines.append(f"  {key.replace('_', ' ').capitalize()}: {r[key]}")
        lines.append("")
    return "\n".join(lines) + "\n"


class _Stage:
    def __init__(self, out: Path):
        self.out = out
        self.files: list[dict] = []
        self.skipped: list[dict] = []

    def reset(self, kind: str) -> Path:
        d = self.out / kind
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        return d

    def copy(self, src: Path, kind: str, rel: str) -> None:
        try:
            size = src.stat().st_size
        except OSError:
            return
        if size > MAX_FILE_BYTES:
            self.skipped.append({"type": kind, "path": str(src), "bytes": size, "why": "over the size limit"})
            return
        dst = self.out / kind / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def write(self, kind: str, rel: str, text: str) -> None:
        dst = self.out / kind / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)


def _profile_report_head(report: Path, max_lines: int = 12) -> list[str]:
    """The report's title and its candidates or findings, for the summary."""
    try:
        text = report.read_text(errors="replace").splitlines()
    except OSError:
        return []
    head = [l for l in text if l.startswith("# ")][:1]
    for i, l in enumerate(text):
        if l.startswith("## ") and any(w in l.lower() for w in ("candidate", "finding")):
            body = [x for x in text[i + 1:] if x.strip()]
            body = body[:next((j for j, x in enumerate(body) if x.startswith("## ")), len(body))]
            return head + body[:max_lines]
    return head


def stage(session: Path, out: Path, meta: dict, dossier: Path | None = None,
          profile_dirs: list[Path] | None = None, types: tuple[str, ...] = TYPES) -> dict:
    """Stage the collected knowledge under ``out/<type>/``; returns the manifest."""
    st = _Stage(out)
    state: dict = {}
    try:
        state = json.loads((session / "session.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    title = f"{meta.get('campaign')} ({meta.get('project')}, {meta.get('user')})"
    rows = outcomes(state.get("lessons") or [])
    records = current_findings(session / "ledger.jsonl")
    profiles = [p for p in sorted((session / "profiles").glob("*")) if p.is_dir()] + list(profile_dirs or [])

    if "outcomes" in types:
        st.reset("outcomes")
        st.write("outcomes", "outcomes.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        st.write("outcomes", "OUTCOMES.md", render_outcomes(rows, title))
    if "findings" in types:
        st.reset("findings")
        st.write("findings", "findings.jsonl", "".join(json.dumps(r) + "\n" for r in records))
        st.write("findings", "FINDINGS.md", render_findings(records, title))
    if "profiles" in types:
        st.reset("profiles")
        for p in profiles:
            for f in sorted(p.rglob("*")):
                if f.is_file() and any(fnmatch.fnmatch(f.name, pat) for pat in PROFILE_FILES):
                    st.copy(f, "profiles", f"{p.name}/{f.relative_to(p).as_posix()}")
    if "dossier" in types:
        st.reset("dossier")
        if dossier and dossier.is_dir():
            for f in sorted(dossier.rglob("*.md")):
                st.copy(f, "dossier", f.relative_to(dossier).as_posix())
    if "artifacts" in types:
        st.reset("artifacts")
        for name in ("session.json", "ledger.jsonl", "landed.patch", "ABORTED.md", "REPORT.md",
                     "directions.json", "MAP_STOP.md"):
            if (session / name).is_file():
                st.copy(session / name, "artifacts", name)
        kept = {r["iteration"] for r in rows if r["kind"] == "keep"}
        for d in sorted(session.glob("iter_*")):
            if (d / "patch.diff").is_file():
                st.copy(d / "patch.diff", "artifacts", f"{d.name}/patch.diff")
            n = d.name[len("iter_"):]
            if n.isdigit() and int(n) in kept and (d / "bench_output.json").is_file():
                st.copy(d / "bench_output.json", "artifacts", f"{d.name}/bench_output.json")

    manifest = {"collected_at": now_iso(), **meta, "session_dir": str(session),
                "types": [t for t in types if t != "summary"], "files": [], "skipped": st.skipped}
    if "summary" in types:
        st.reset("summary")
        counts = {k: sum(r["kind"] == k for r in rows) for k, _ in _SECTIONS}
        lines = [f"# {title}", "",
                 f"- Pack: {state.get('pack_name')}; goal: {state.get('goal')}",
                 f"- Metric: {state.get('metric_name')}: baseline {state.get('baseline')}, "
                 f"best {state.get('best_metric')}, iteration {state.get('iteration')}",
                 "- Outcomes: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in counts.items()),
                 f"- Findings: {len(records)} current "
                 f"({sum(r.get('type') == 'DEAD_END' for r in records)} dead ends)",
                 f"- Dossier: {'included' if dossier and dossier.is_dir() else 'none given'}",
                 f"- Collected {manifest['collected_at']} on {meta.get('host')}"]
        kept_rows = [r for r in rows if r["kind"] == "keep"]
        if kept_rows:
            lines += ["", "## Kept", ""] + [f"- {r['iteration']}: {r['what']}" for r in kept_rows[-10:]]
        if profiles:
            lines += ["", "## Profiles", ""]
            for p in profiles:
                report = next((p / n for n in ("PROFILE.md", "profile.md") if (p / n).is_file()), None)
                lines.append(f"- `{p.name}`" + ("" if report else ": no PROFILE.md"))
                lines += [f"  {l}" for l in (_profile_report_head(report) if report else [])]
        st.write("summary", "SUMMARY.md", "\n".join(lines) + "\n")
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name != "manifest.json":
            manifest["files"].append({"path": f.relative_to(out).as_posix(), "bytes": f.stat().st_size,
                                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    if "summary" in types:
        st.write("summary", "manifest.json", json.dumps(manifest, indent=1) + "\n")
    return manifest


def files_for_push(out: Path, remote_knowledge: str) -> list[tuple[Path, str]]:
    return [(f, f"{remote_knowledge}/{f.relative_to(out).as_posix()}")
            for f in sorted(out.rglob("*")) if f.is_file()]
