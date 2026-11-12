"""Export an Arbor session directory as the ``arbor-kb-export.v1`` document the ``arbor-v1`` mapping collects.

The export writes ``<out>/arbor-kb-export.json`` and, under ``<out>/files/``, the files a record links that the
session does not hold as they are: each iteration's manifest (``arbor-iteration.json``), its hook-trace slice
(``events.jsonl``), its campaign-log segment, the reduced server metrics of a detailed run, and the campaign's launch
record (``arbor-campaign.json``). Every other linked file is the session's own, or its pack copy's. The KB stores
each content once by SHA-256 and names a file by its basename; the manifest gives each file's place in the session
directory, so a session directory rebuilt from the KB opens in Overseer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arbor.collect.kb import records
from arbor.collect.kb.files import MAX_HITS, Screen, iteration_links
from arbor.collect.kb.layout import (
    CAMPAIGN_FORMAT,
    CAMPAIGN_NAME,
    FORMAT,
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    META_STATIC,
    SESSION_STATIC,
    Link,
    Stage,
    as_dict,
    number,
    read_jsonl,
    sha256_file,
)
from arbor.collect.kb.session import (
    Bars,
    Session,
    arbor_version,
    find_pack,
    load,
    pack_links,
    pack_measurement,
    replay_bars,
)

@dataclass
class Campaign:
    """What every record of a campaign shares: its files, its pack's gate settings, and its first measured run."""

    links: list[Link] = field(default_factory=list)
    omitted: list[dict] = field(default_factory=list)
    screen: Screen = field(default_factory=Screen)
    gate: dict = field(default_factory=dict)
    pack_note: str = ""
    first_agg: dict = field(default_factory=dict)
    first_measured: int | None = None


# ---------------------------------------------------------------- the campaign's files
def gate_settings(pack: Path) -> dict:
    """The pack copy's gate settings, as provenance.extra names them."""

    measurement = pack_measurement(pack)
    return {name: number(measurement.get(key)) for key, name in records.GATE_SETTINGS.items()}


def campaign(s: Session, stage: Stage, pack_roots: list[Path]) -> Campaign:
    launch = {
        "format": CAMPAIGN_FORMAT,
        "session_name": s.name,
        "session": {k: s.session[k] for k in SESSION_STATIC if k in s.session},
        "meta": {k: s.meta[k] for k in META_STATIC if k in s.meta},
        "baseline_row": next((r for r in s.lessons if r.get("decision") == "BASELINE"), None),
    }
    c = Campaign(links=[Link(stage.write_json(f"campaign/{CAMPAIGN_NAME}", launch), CAMPAIGN_NAME, "campaign")])
    pack = find_pack(s.session.get("pack_name") or s.config.get("pack") or "", s.dir, pack_roots)
    if pack is None:
        c.pack_note = "no pack copy of the session's pack was found"
    else:
        links, c.pack_note = pack_links(pack, s.session.get("validator_digests"))
        c.links += links
        c.gate = gate_settings(pack) if links else {}
    c.links = c.screen.keep(c.links, c.omitted)
    c.first_measured = next((it for it in s.owners.gated
                             if s.agg(it) is not None and s.main(it).get("decision") != "INVALID"), None)
    agg = s.agg(c.first_measured) if c.first_measured is not None else None
    c.first_agg = s.agg_doc(c.first_measured) if agg else {}
    return c


# ---------------------------------------------------------------- one record
#: Claude's usage counts, as provenance.extra names them (a key that names tokens reads as a credential to the KB).
USAGE = {
    "input_tokens": "claude_usage_input",
    "output_tokens": "claude_usage_output",
    "cache_creation_input_tokens": "claude_usage_cache_creation",
    "cache_read_input_tokens": "claude_usage_cache_read",
}


def _transcript_usage(links: list[Link]) -> dict:
    """The agent model and the four usage counts of the transcripts the iteration owns, when they are readable."""

    counts = dict.fromkeys(USAGE.values(), 0)
    model = None
    for link in links:
        if link.kind != "transcript":
            continue
        for _, _, row in read_jsonl(link.source):
            message = as_dict(row.get("message"))
            usage = as_dict(message.get("usage"))
            model = message.get("model") or model
            for key, name in USAGE.items():
                value = usage.get(key)
                counts[name] += value if isinstance(value, int) and not isinstance(value, bool) else 0
    return {"agent_model": model, **{name: float(n) if any(counts.values()) else None for name, n in counts.items()}}


def provenance_extra(s: Session, it: int, main: dict, c: Campaign, extras: dict) -> dict:
    config, env = s.config, as_dict(s.config.get("server_env"))
    lesson = as_dict(main.get("provenance"))
    gpu_id = s.iteration_json(it, "agentic_run/gpu_metrics_identity.json")
    first_gpu = as_dict((gpu_id.get("gpu_data") or [{}])[0])
    segment = s.log.get(it)
    usage = extras.pop("usage")
    return {
        "arbor_identity": s.session.get("identity"),
        "pack": s.session.get("pack_name"),
        "claim_class": main.get("claim_class"),
        **extras.pop("ledger"),
        "bench_script": config.get("bench_script"),
        "image_tag": config.get("_image") or env.get("IMAGE") or c.first_agg.get("image"),
        "gpu_reported": as_dict(first_gpu.get("asic")).get("market_name"),
        "commit": lesson.get("arbor_commit") or (segment.commit if segment else None),
        "agent_model": lesson.get("agent_model") or usage.pop("agent_model"),
        **usage,
        **records.phase_seconds(s, it),
        "session_format": float(s.format),
        "scale": number(main.get("scale")),
        "secondary_scale": number(main.get("secondary_scale")),
        **extras,
        **c.gate,
    }


def unit(s: Session, it: int, bars: Bars, c: Campaign, kept: list[str], stage: Stage) -> dict:
    main = s.main(it)
    detailed = main.get("decision") == "KEEP" or it == c.first_measured
    omitted: list[dict] = []
    links = c.screen.keep(iteration_links(s, it, stage, detailed), omitted) + c.links
    owned = sorted(s.own_lessons.get(it, []), key=lambda r: r["index"])
    manifest = {
        "format": MANIFEST_FORMAT, "session_name": s.name, "run_id": s.run_id, "iteration": it,
        "lessons": owned,
        "ledger": sorted(s.own_ledger.get(it, []), key=lambda r: r["line"]),
        "trace_lines": [line for line, _ in sorted(s.own_events.get(it, []))],
        "claude_sessions": s.own_claude.get(it, []),
        "files": [link.entry() for link in links],
        "omitted": omitted + c.omitted,
    }
    staged = stage.write_json(f"iter_{it}/{MANIFEST_NAME}", manifest)
    in_rows = c.screen.hits(staged)
    cond = records.conditions(s, it, main, c.first_agg)
    base = records.baseline(s, it, bars, records.fingerprint_inputs(s, kept, cond))
    ledger = records.ledger_view(s, it)
    rationale, reasoning_clipped = records.rationale(s, ledger, str(main.get("change") or ""))
    change = records.change(s, it, main)
    outcome = records.outcome(s, it, main, base["best_before"])
    extras = {"ledger": records.ledger_counts(records.first_ids(s), it), "usage": _transcript_usage(links),
              **cond, **records.secondary_bars(s, bars), "reasoning_clipped": True if reasoning_clipped else None}
    return {
        "unit_id": f"{s.name}/iter_{it}",
        "kind": "framework",
        "excluded": _excluded(s, it, in_rows),
        "workload_kind": s.session.get("workload_kind"),
        "run_id": s.run_id,
        "seq": it,
        "created_at": records.created_at(s, main),
        "completed_at": records.iso_of(main.get("ts")),
        "objective": records.objective_of(s.session),
        "identity": records.identity(s, main, c.first_agg),
        "baseline": base,
        "rationale": rationale,
        "change": change,
        "outcome": outcome,
        "reflection": records.reflection(s, it, main, ledger),
        "notes": records.notes(s, it, main),
        "raw_logs": [str(staged)] + [str(link.source) for link in links],
        "provenance": {"source_ref": f"arbor:{s.name}/iter_{it}",
                       "extra": provenance_extra(s, it, main, c, extras)},
    }


def _excluded(s: Session, it: int, in_rows: list[str]) -> str | None:
    """Why the record is not collected: its rows hold a credential-shaped value, or it is not final yet."""

    if in_rows:
        return (f"the iteration's own rows (its manifest, {MANIFEST_NAME}) hold credential-shaped values: "
                f"{'; '.join(in_rows[:MAX_HITS])}")
    return None if s.owners.final(it) else "no campaign end follows this iteration yet"


# ---------------------------------------------------------------- the document
def export(session_dir: Path, out: Path, pack_roots: list[Path] | None = None,
           campaign_logs: list[Path] | None = None) -> dict:
    """The export document for ``session_dir``, its linked files staged under ``out/files``."""

    session_dir, out = session_dir.resolve(), out.resolve()
    stage = Stage(out)
    (out / "files").mkdir(parents=True, exist_ok=True)
    s = load(session_dir, list(campaign_logs or []))
    if isinstance(s, str):
        return {"format": FORMAT, "session": {"name": session_dir.name, "dir": str(session_dir), "read_error": s},
                "units": []}
    c = campaign(s, stage, list(pack_roots or []))
    before, after = replay_bars(s.lessons, s.session)
    kept: list[tuple[int, str]] = []
    retracted: set[Any] = set()
    units = []
    for it in s.owners.gated:
        units.append(unit(s, it, before.get(it, Bars()), c, [sha for k, sha in kept if k not in retracted], stage))
        for item in s.own_lessons.get(it, []):
            retracts = item["row"].get("retracts")
            if item["row"].get("decision") == "KEEP" and isinstance(retracts, dict):
                retracted.add(retracts.get("iteration"))
        patch = s.patch(it)
        if patch and s.main(it).get("decision") == "KEEP":
            kept.append((it, sha256_file(patch)))
    return {
        "format": FORMAT,
        "session": {"name": s.name, "dir": str(session_dir), "run_id": s.run_id, "read_error": None,
                    "ended": s.owners.end is not None, "pack_note": c.pack_note,
                    "session_format": s.format, **arbor_version()},
        "units": units,
        "checks": checks(s, before, after, c),
    }


def checks(s: Session, before: dict[int, Bars], after: Bars, c: Campaign) -> dict:
    """What the export confirms against Arbor's own records, for whoever reads it; the KB never sees it."""

    shown = [(it, s.prompt_best(it)) for it in s.owners.gated]
    compared = [(it, best) for it, best in shown if best is not None and it in before]
    replayed, recorded = after.best, number(s.session.get("best_metric"))
    agree = replayed is None or recorded is None or abs(replayed - recorded) <= 1e-6 * max(1.0, abs(recorded))
    return {
        "final_best_metric": {"replayed": replayed, "session": recorded, "agree": agree},
        "credential_omissions": dict(sorted(c.screen.flagged.items())),
        "final_best_secondary": {"replayed": after.best_secondary, "session": s.session.get("best_secondary")},
        "worker_prompt_best_lines": {
            "checked": len(compared),
            "match": sum(1 for it, best in compared
                         if before[it].best is not None and abs(best - before[it].best) < 1e-6),
        },
        "lessons": {"total": len(s.session.get("lessons") or []),
                    "owned": sum(len(v) for v in s.own_lessons.values())},
        "ledger_rows_owned": sum(len(v) for v in s.own_ledger.values()),
        "trace_events": {"total": len(s.events), "owned": sum(len(v) for v in s.own_events.values())},
        "iterations": {"gated": len(s.owners.gated), "final": sum(map(s.owners.final, s.owners.gated))},
        "campaign_log_segments": len(s.log),
        "pack": c.pack_note or "linked",
    }
