"""One framework campaign iteration as an ``arbor-v1`` unit: each category's values, by the schema's Source column.

Every value is read from the session's own records, each from its sources in a fixed order; a value no source holds
is left out (``None``), and the mapping drops it from the record.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from arbor.collect.kb.layout import (
    ANNOTATIONS,
    BENCH_OUTCOMES,
    INTENT_ALIASES,
    as_dict,
    canonical,
    clip,
    epoch,
    iso,
    measured,
    number,
    sha256_file,
    sha256_text,
)
from arbor.collect.kb.session import PHASES, Bars, Session

#: (metric_name, metric_unit or None for any unit) to the objective, the order the schema's objective table gives.
OBJECTIVES = (
    ("output_throughput", "tok/s/chip", "output_throughput_per_gpu@v1"),
    ("output_throughput", "tok/s/GPU", "output_throughput_per_gpu@v1"),
    ("total_throughput_per_gpu", None, "total_throughput_per_gpu@v1"),
    ("total_throughput", "tok/s/GPU", "total_throughput_per_gpu@v1"),
    ("p50_interactivity", None, "p50_interactivity@v1"),
    ("mean_interactivity", None, "mean_interactivity@v1"),
)
#: The outcome field each objective's reading is in.
READING = {
    "output_throughput_per_gpu@v1": "output_throughput",
    "total_throughput_per_gpu@v1": "total_throughput_per_gpu",
    "p50_interactivity@v1": "p50_interactivity",
    "mean_interactivity@v1": "mean_interactivity",
}
#: The pack copy's objective.measurement keys, as provenance.extra names them.
GATE_SETTINGS = {
    "threshold_rel": "gate_threshold_rel",
    "noise_band_pct": "gate_noise_band_pct",
    "secondary_noise_band_pct": "gate_secondary_noise_band_pct",
    "secondary_regression_cap_pct": "gate_secondary_regression_cap_pct",
    "correctness_regression_cap_pct": "gate_correctness_regression_cap_pct",
}
_AGENT_ERROR = re.compile(r"^iteration \d+ \(agent error: (.*)\)\s*$", re.S)
_ROADMAP_TAGS = re.compile(r"^\s*((?:\[[^\]]+\]\s*)+)")
_DIRECTION_HEAD = re.compile(r"^\s*(?:\[[^\]]+\]\s*)*([a-z][a-z0-9.-]*(?:\s*/\s*[a-z][a-z0-9.-]*)*)")
_CRITIC_OVERRIDE = re.compile(r"critic override:\s*(.*)$", re.S)


def objective_of(session: dict) -> str | None:
    """The objective the campaign's metric reads as, or ``None`` when the declaration names none for it."""

    name, unit = session.get("metric_name"), session.get("metric_unit")
    return next((objective for metric, want, objective in OBJECTIVES
                 if metric == name and (want is None or want == unit)), None)


def _identity_part(s: Session, index: int) -> str | None:
    """Field ``index`` (from 1) of the Arbor identity, ``inference:<bench>:<gpu>:<framework>:<precision>:…``."""

    parts = str(s.session.get("identity") or s.config.get("identity") or "").split(":")
    return (parts[index - 1] or None) if len(parts) >= index else None


def identity(s: Session, main: dict, first_agg: dict) -> dict:
    config = s.config
    env = as_dict(config.get("server_env"))
    gpu_ids = [g for g in str(config.get("gpu_ids") or "").split(",") if g.strip()]
    image = as_dict(main.get("scope")).get("image")
    gpu = config.get("gpu_type") or _identity_part(s, 3) or first_agg.get("hw")
    return {
        "gpu": str(gpu).lower() if gpu else None,
        "num_gpus": float(len(gpu_ids)) if gpu_ids else None,
        "image": image if isinstance(image, str) and image.startswith("sha256:") else None,
        "model": config.get("model_name"),
        "precision": env.get("PRECISION") or _identity_part(s, 5),
        "tp": number(config.get("tp")),
        "framework": config.get("framework"),
        "bench": _identity_part(s, 2),
        "conc": number(config.get("concurrency")),
    }


def effective_flags(s: Session, iteration: int) -> dict:
    """The server flags the iteration ran: its effective_config.json, else the nearest earlier one, else launch."""

    return s.effective_flags(iteration) or _parse_flags(str(s.config.get("server_flags") or ""))


def _parse_flags(text: str) -> dict:
    try:
        words = shlex.split(text)
    except ValueError:
        return {}
    flags: dict[str, Any] = {}
    for at, word in enumerate(words):
        if word.startswith("--"):
            has_value = at + 1 < len(words) and not words[at + 1].startswith("--")
            flags[word] = words[at + 1] if has_value else True
    return flags


def conditions(s: Session, iteration: int, main: dict, first_agg: dict) -> dict:
    """The measurement conditions, which ``base_fingerprint`` hashes and provenance.extra holds for now."""

    digests = as_dict(s.session.get("validator_digests"))
    scope = as_dict(main.get("scope"))
    flags = effective_flags(s, iteration)
    spec = _json_flag(flags.get("--speculative-config"))
    duration = as_dict(s.config.get("server_env")).get("DURATION")
    return {
        "bench_sha256": next((v for k, v in digests.items() if k.endswith("/validator/bench.sh")), None),
        "replay_duration_s": number(float(duration)) if _numeric(duration) else None,
        "dataset": as_dict(first_agg.get("dataset")).get("hf_dataset_name"),
        "scenario": first_agg.get("scenario_type"),
        "isl": number(scope.get("isl")),
        "osl": number(scope.get("osl")),
        "max_model_len": number(float(flags["--max-model-len"])) if _numeric(flags.get("--max-model-len")) else None,
        "speculative_method": spec.get("method"),
        "speculative_draft_count": number(spec.get("num_speculative_tokens")),
        "rejection_sample_method": spec.get("rejection_sample_method"),
        "synthetic_acceptance_length": number(spec.get("synthetic_acceptance_length")),
    }


def _numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return not isinstance(value, bool)


def _json_flag(value: Any) -> dict:
    try:
        return as_dict(json.loads(value)) if isinstance(value, str) else {}
    except ValueError:
        return {}


def baseline(s: Session, iteration: int, bars: Bars, fingerprint_inputs: dict) -> dict:
    """The bar before the decision: the campaign log's ``best=``, else the worker prompt's ``- Best:``, else replay."""

    segment = s.log.get(iteration)
    logged = segment.best if segment else None
    best = next((v for v in (logged, s.prompt_best(iteration), bars.best) if v is not None), None)
    return {
        "best_before": best,
        "campaign_baseline": number(s.session.get("baseline")),
        "base_tree": s.config.get("_base_tree") or None,
        "base_fingerprint": sha256_text(canonical(fingerprint_inputs)),
        "best_set_by": bars.set_by,
    }


def secondary_bars(s: Session, bars: Bars) -> dict:
    """The secondary metric's bars, kept in provenance.extra until the schema review promotes them."""

    return {"best_secondary_before": bars.best_secondary,
            "campaign_baseline_secondary": number(s.session.get("baseline_secondary"))}


def fingerprint_inputs(s: Session, kept: list[str], cond: dict) -> dict:
    """What base_fingerprint hashes: the launch flags, earlier KEEPs' patches, the base tree and the conditions."""

    keys = ("bench_sha256", "replay_duration_s", "dataset", "scenario", "isl", "osl")
    return {
        "server_flags": _parse_flags(str(s.config.get("server_flags") or "")),
        "kept_patches": kept,
        "base_tree": s.config.get("_base_tree") or None,
        **{key: cond.get(key) for key in keys},
    }


def ledger_view(s: Session, iteration: int) -> dict[str, dict]:
    """The ledger records the iteration wrote, the last row per id."""

    records: dict[str, dict] = {}
    for item in sorted(s.own_ledger.get(iteration, []), key=lambda r: r["line"]):
        records[str(item["row"].get("id"))] = item["row"]
    return records


def _ledger_text(record: dict) -> str:
    lines = [f"{record.get('type')} {record.get('id')} [{record.get('status') or 'open'}]: "
             f"{record.get('claim') or ''}".strip()]
    for key, label in (("evidence", "Evidence"), ("instrument", "Instrument"), ("falsifier", "Falsifier"),
                       ("revisit_if", "Revisit if")):
        if record.get(key):
            lines.append(f"{label}: {record[key]}")
    return "\n".join(lines)


def rationale(s: Session, records: dict[str, dict], change: str) -> tuple[dict, bool]:
    hypotheses = [_ledger_text(r) for r in records.values() if r.get("type") in ("ASSUMPTION", "UNKNOWN")]
    reasoning, clipped = clip("\n\n".join(hypotheses))
    tags = _ROADMAP_TAGS.match(change or "")
    heads = _DIRECTION_HEAD.match(change or "")
    ids = [p.strip() for p in heads.group(1).split("/")] if heads else []
    return {
        "reasoning": reasoning or None,
        "roadmap_items": re.findall(r"\[([^\]]+)\]", tags.group(1)) if tags else [],
        "directions": [i for i in ids if i in s.directions],
    }, clipped


def change(s: Session, iteration: int, main: dict) -> dict:
    text = str(main.get("change") or "").strip()
    summary = None if _AGENT_ERROR.match(text) else text or None
    patch = s.patch(iteration)
    claim = main.get("claim_class")
    return {
        "summary": clip(summary)[0] if summary else None,
        "tier": main.get("tier"),
        "change_fingerprint": sha256_file(patch) if patch else (sha256_text(text) if text else None),
        "intent": main.get("intent") or INTENT_ALIASES.get(str(claim)),
        "patch": str(patch) if patch and main.get("decision") == "KEEP" else None,
    }


def outcome(s: Session, iteration: int, main: dict, best: float | None) -> dict:
    bench = s.iteration_json(iteration, "bench_output.json")
    agg = s.agg_doc(iteration)
    correctness = s.iteration_json(iteration, "correctness_output.json")
    readings = {key: measured(bench.get(key)) for key in BENCH_OUTCOMES}
    ttft = as_dict(as_dict(as_dict(agg.get("request_metrics")).get("latency")).get("ttft"))
    reading = readings.get(READING.get(objective_of(s.session) or "", ""))
    error = _AGENT_ERROR.match(str(main.get("change") or "").strip())
    segment = s.log.get(iteration)
    reason = main.get("reason") or (segment.gate_reason if segment else None) or (error.group(1) if error else None)
    return {
        "decision": main.get("decision"),
        "delta_pct": round((reading / best - 1) * 100, 4) if reading is not None and best else None,
        "reason": clip(str(reason).strip())[0] if reason else None,
        **readings,
        "p50_ttft_s": measured(ttft.get("p50")),
        "booked": main.get("booked") if isinstance(main.get("booked"), bool) else None,
        "frontier": main.get("frontier") if isinstance(main.get("frontier"), bool) else None,
        "correctness_passed": correctness.get("passed") if isinstance(correctness.get("passed"), bool) else None,
        "correctness_score": number(correctness.get("score")),
        "wall_s": number(main.get("wall_s")),
    }


def reflection(s: Session, iteration: int, main: dict, records: dict[str, dict]) -> dict:
    findings = [clip(_ledger_text(r))[0] for r in records.values() if r.get("type") in ("FACT", "DEAD_END")]
    segment = s.log.get(iteration)
    override = _CRITIC_OVERRIDE.search(str(main.get("reason") or ""))
    critic = (segment.critic if segment else None) or (override.group(1).strip() if override else None)
    return {"findings": findings, "critic": clip(critic)[0] if critic else None}


def notes(s: Session, iteration: int, main: dict) -> dict:
    """Each annotation row's text the iteration owns, by kind; several of one kind joined by newlines."""

    found: dict[str, list[str]] = {}
    for item in s.own_lessons.get(iteration, []):
        row = item["row"]
        kind = ANNOTATIONS.get(str(row.get("decision")))
        if kind and row is not main and str(row.get("change") or "").strip():
            found.setdefault(kind, []).append(str(row["change"]).strip())
    return {kind: clip("\n".join(texts))[0] for kind, texts in found.items()}


def iso_of(value: Any) -> str | None:
    seconds = epoch(value)
    return iso(seconds) if seconds is not None else None


def created_at(s: Session, main: dict) -> str | None:
    """The gate row's ``started``, else the ``ts`` of the lesson before it."""

    started = epoch(main.get("started"))
    if started is not None:
        return iso(started)
    rows = s.lessons
    at = next((i for i, row in enumerate(rows) if row is main), 0)
    previous = epoch(rows[at - 1].get("ts")) if at > 0 else None
    return iso(previous) if previous is not None else None


def phase_seconds(s: Session, iteration: int) -> dict:
    """Seconds each agent role spent in the iteration's hook-trace sessions, as Overseer sums them."""

    base = number(s.session.get("start_epoch")) or 0.0
    totals = dict.fromkeys(PHASES, 0.0)
    for trace in s.own_traces.get(iteration, []):
        if trace.role in totals:
            totals[trace.role] += round(trace.t1 - base, 1) - round(trace.t0 - base, 1)
    return {f"{role}_seconds": round(total, 1) if total else None for role, total in totals.items()}


def first_ids(s: Session) -> dict[str, int]:
    """Each ledger id's owner, where the id first appears: an id counts as new there, as Overseer counts."""

    owner: dict[str, int] = {}
    for it in s.owners.gated:
        for item in sorted(s.own_ledger.get(it, []), key=lambda r: r["line"]):
            owner.setdefault(f"{item['row'].get('type')}:{item['row'].get('id')}", it)
    return owner


def ledger_counts(first: dict[str, int], iteration: int) -> dict:
    counts = dict.fromkeys(("FACT", "ASSUMPTION", "DEAD_END", "UNKNOWN"), 0)
    for key, owner in first.items():
        kind = key.split(":", 1)[0]
        if owner == iteration and kind in counts:
            counts[kind] += 1
    return {f"ledger_{kind.lower()}": float(count) for kind, count in counts.items()}
