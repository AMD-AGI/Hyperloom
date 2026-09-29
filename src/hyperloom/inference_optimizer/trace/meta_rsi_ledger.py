"""Per-session ledger for the meta-RSI outer loop: which harness ran, what it cost, what it scored.

The ledger is derived only from files a session already writes (``manifest.json``,
``state.json``, ``reports/optimization_journal.json``, ``reports/trace/llm_calls.jsonl``),
so it can be rebuilt for any past session. It never reads or writes the scoring outlet;
gain and verdicts are copied from the journal the measurement path produced.

Stdlib only: ``scripts/meta_rsi`` imports it on hosts without a Hyperloom install.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = "meta_rsi_ledger/1"
LEDGER_FILENAME = "meta_rsi_ledger.json"

# Relative price of each token class against an uncached input token (Anthropic list
# pricing ratios). Used only for the cost-weighted total, never for any decision.
COST_WEIGHTS = {
    "input": 1.0,
    "cache_read": 0.1,
    "cache_write": 1.25,
    "output": 5.0,
}

_TOKEN_FIELDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
    "reasoning": "reasoning_output_tokens",
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a JSONL file, skipping blank or corrupt lines."""
    try:
        fh = path.open()
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def call_tokens(row: dict[str, Any]) -> dict[str, int]:
    """Token classes of one ``llm_calls.jsonl`` record."""
    return {name: _int(row.get(field)) for name, field in _TOKEN_FIELDS.items()}


def context_tokens(row: dict[str, Any]) -> int:
    """Prompt-side tokens the model read for one call (fresh + cached)."""
    t = call_tokens(row)
    return t["input"] + t["cache_read"] + t["cache_write"]


def billed_tokens(tokens: dict[str, int]) -> int:
    return tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_write"]


def weighted_tokens(tokens: dict[str, int]) -> float:
    return sum(COST_WEIGHTS[k] * tokens[k] for k in COST_WEIGHTS)


def component_of(row: dict[str, Any]) -> str:
    return str(row.get("component") or row.get("role") or "unknown")


class _Bucket:
    __slots__ = ("calls", "errors", "tokens", "contexts")

    def __init__(self) -> None:
        self.calls = 0
        self.errors = 0
        self.tokens: Counter[str] = Counter()
        self.contexts: list[int] = []

    def add(self, row: dict[str, Any]) -> None:
        self.calls += 1
        if str(row.get("status") or "ok") != "ok":
            self.errors += 1
        self.tokens.update(call_tokens(row))
        self.contexts.append(context_tokens(row))

    def summary(self) -> dict[str, Any]:
        tokens = {k: int(self.tokens.get(k, 0)) for k in _TOKEN_FIELDS}
        ctx = sorted(self.contexts)
        return {
            "calls": self.calls,
            "errors": self.errors,
            **tokens,
            "billed": billed_tokens(tokens),
            "weighted": round(weighted_tokens(tokens)),
            "ctx_p50": int(statistics.median(ctx)) if ctx else 0,
            "ctx_p90": ctx[int(0.9 * (len(ctx) - 1))] if ctx else 0,
            "ctx_max": ctx[-1] if ctx else 0,
        }


def _role_model_hint(manifest: dict[str, Any]) -> dict[str, str]:
    """Model per component recorded by the harness fingerprint, if any."""
    fp = manifest.get("harness_fingerprint") or {}
    models = (fp.get("model") or {}).get("role_models") or {}
    return {str(k): str(v.get("model") if isinstance(v, dict) else v) for k, v in models.items() if v}


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def journal_outcome(journal: Any) -> dict[str, Any]:
    """Verdict counts and gain copied from ``optimization_journal.json``."""
    if not isinstance(journal, dict):
        return {}
    entries = [e for e in journal.get("entries") or [] if isinstance(e, dict)]
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for e in entries:
        by_kind[str(e.get("kind") or "other")][str(e.get("outcome") or "unknown")] += 1
    # "other" rows are bookkeeping (target_analysis, roofline, specialist) rather than
    # a serving change; config KEEPs are what the session actually bought.
    config_kinds = {"param", "env", "backend", "kernel", "patch", "integrate_patch"}
    config = Counter()
    for kind, counts in by_kind.items():
        if kind in config_kinds:
            config.update(counts)
    return {
        "baseline_throughput": journal.get("baseline_throughput"),
        "final_throughput": journal.get("final_throughput"),
        "total_gain_pct": journal.get("total_gain_pct"),
        "entries": len(entries),
        "keep": sum(c.get("KEEP", 0) for c in by_kind.values()),
        "revert": sum(c.get("REVERT", 0) for c in by_kind.values()),
        "config_keep": config.get("KEEP", 0),
        "config_revert": config.get("REVERT", 0),
        "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
    }


def build_ledger(session_dir: str | Path, calls: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the ledger of one session directory."""
    sdir = Path(session_dir)
    manifest = _load_json(sdir / "manifest.json") or {}
    state = _load_json(sdir / "state.json") or {}
    journal = _load_json(sdir / "reports" / "optimization_journal.json")
    rows = list(calls) if calls is not None else list(iter_jsonl(sdir / "reports" / "trace" / "llm_calls.jsonl"))

    model_hint = _role_model_hint(manifest)
    total = _Bucket()
    by_component: dict[str, _Bucket] = defaultdict(_Bucket)
    by_phase: dict[str, _Bucket] = defaultdict(_Bucket)
    by_model: dict[str, _Bucket] = defaultdict(_Bucket)
    by_component_phase: dict[str, _Bucket] = defaultdict(_Bucket)
    first = last = None
    for row in rows:
        comp = component_of(row)
        phase = str(row.get("phase") or "none")
        model = str(row.get("model") or model_hint.get(comp) or "unknown")
        total.add(row)
        by_component[comp].add(row)
        by_phase[phase].add(row)
        by_model[model].add(row)
        by_component_phase[f"{comp}@{phase}"].add(row)
        ts = _parse_ts(row.get("ts"))
        if ts is not None:
            first = ts if first is None or ts < first else first
            last = ts if last is None or ts > last else last

    workload = manifest.get("workload") or {}
    outcome = journal_outcome(journal)
    tot = total.summary()
    gain = outcome.get("total_gain_pct")
    keeps = outcome.get("config_keep") or 0
    return {
        "schema": SCHEMA,
        "session_dir": str(sdir),
        "session_id": manifest.get("session_id") or sdir.name,
        "model": Path(str(manifest.get("model_path") or manifest.get("model_name") or "")).name or None,
        "framework": manifest.get("framework"),
        "gpu_type": manifest.get("gpu_type"),
        "tp": manifest.get("tp"),
        "workload": {k: workload.get(k) for k in ("conc", "isl", "osl") if k in workload} or workload,
        "max_minutes": manifest.get("max_minutes"),
        "objective": manifest.get("objective"),
        "code_revision": manifest.get("code_revision"),
        "harness_fingerprint": manifest.get("harness_fingerprint"),
        "stop_reason": state.get("stop_reason"),
        "orchestration_idle_skips": state.get("orchestration_idle_skips"),
        "llm_span_min": round((last - first).total_seconds() / 60, 1) if first and last else None,
        "outcome": outcome,
        "tokens": {
            "total": tot,
            "by_component": {k: v.summary() for k, v in sorted(by_component.items())},
            "by_phase": {k: v.summary() for k, v in sorted(by_phase.items())},
            "by_model": {k: v.summary() for k, v in sorted(by_model.items())},
            "by_component_phase": {k: v.summary() for k, v in sorted(by_component_phase.items())},
        },
        "efficiency": {
            "billed_per_config_keep": round(tot["billed"] / keeps) if keeps else None,
            "weighted_per_config_keep": round(tot["weighted"] / keeps) if keeps else None,
            "billed_per_gain_pct": round(tot["billed"] / gain) if isinstance(gain, (int, float)) and gain > 0 else None,
        },
    }


def write_ledger(session_dir: str | Path) -> Path:
    """Write ``reports/meta_rsi_ledger.json`` for a session and return its path."""
    sdir = Path(session_dir)
    out = sdir / "reports" / LEDGER_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(build_ledger(sdir), indent=1, sort_keys=True))
    tmp.replace(out)
    return out


__all__ = [
    "COST_WEIGHTS",
    "LEDGER_FILENAME",
    "SCHEMA",
    "billed_tokens",
    "build_ledger",
    "call_tokens",
    "component_of",
    "context_tokens",
    "iter_jsonl",
    "journal_outcome",
    "weighted_tokens",
    "write_ledger",
]
