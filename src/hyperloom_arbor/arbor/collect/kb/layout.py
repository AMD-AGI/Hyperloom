"""What an Arbor session directory holds, and the small readers every part of the export shares.

Arbor's own names are kept as Arbor writes them (``arbor/state.py``, ``arbor/driver.py``, ``arbor/triggers.py``); a
constant here that copies an Arbor table names the module it copies.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arbor import directions as _directions
from arbor import retraction as _retraction
from arbor.measure import intents as _intents

FORMAT = "arbor-kb-export.v1"
MANIFEST_FORMAT = "arbor-kb-iteration.v1"
CAMPAIGN_FORMAT = "arbor-kb-campaign.v1"
SERVER_METRICS_REDUCED = "overseer-server-metrics.v1"
MANIFEST_NAME = "arbor-iteration.json"
CAMPAIGN_NAME = "arbor-campaign.json"

#: The gate's rows, as Arbor writes them (``arbor.measure.intents.legacy_decision``); Overseer takes an iteration's last.
GATE = tuple(sorted({_intents.legacy_decision(i, v) for i in _intents.REGISTRY for v in _intents.VERDICTS}))
#: Rows that are no record (``arbor.directions.ANNOTATIONS``): kept verbatim in the manifest, their text in notes under
#: the row's name in lower case; BASELINE is campaign_baseline instead.
ANNOTATIONS = {row: row.lower() for row in sorted(_directions.ANNOTATIONS - {"BASELINE"})}
#: Rows that set both bars to their own readings (``arbor.retraction``).
SETS_BAR = tuple(_retraction._SETS_BAR)  # noqa: SLF001 - Arbor's own table, read where it is kept
#: Intents whose non-frontier KEEP rebases both bars to its readings (``pareto_exempt`` in ``arbor.measure.intents``).
REBASING_INTENTS = tuple(i.name for i in _intents.REGISTRY if i.pareto_exempt)
#: The claim class a lesson without an intent names, to its intent (the intents' aliases).
INTENT_ALIASES = {alias: i.name for i in _intents.REGISTRY for alias in i.aliases}

#: session.json keys fixed when a campaign starts; the rest move every iteration.
SESSION_STATIC = (
    "identity", "pack_name", "goal", "baseline", "baseline_secondary", "start_epoch", "metric_name",
    "metric_direction", "metric_unit", "secondary_name", "secondary_direction", "workload_kind",
    "validator_digests", "mode", "policy", "review",
)
#: meta.json keys fixed at a campaign's first launch. Each launch rewrites the others and keeps only its own.
META_STATIC = ("arbor_session_dir", "started", "thinking_capture", "host_writable_session_root", "config")

#: Every iteration's own files under iter_N/, beside its agg_*.json.
ITERATION_FILES = ("bench_output.json", "correctness_output.json", "scope.json", "patch.diff")
#: Where a bench writes its run: Arbor's own (agentic_run/), or a host's (host_bench/[runK/], since Arbor v0.9.6).
BENCH_ROOT = "agentic_run"
HOST_BENCH_ROOT = "host_bench"
#: Each bench root's own files, on every measured iteration.
BENCH_FILES = ("power_validation.json", "gpu_metrics_identity.json", "benchmark_command.txt", "report.json")
#: Where each bench root keeps aiperf's artifacts.
ARTIFACTS = {BENCH_ROOT: "aiperf_artifacts", HOST_BENCH_ROOT: "client/aiperf_artifacts"}
#: A detailed iteration's files (each KEEP and the first measured one): the runs Overseer draws a result from.
DETAIL_FILES = ("metrics_plots.png", "workload_distribution_plots.png")
DETAIL_ARTIFACTS = ("profile_export.jsonl", "profile_export_console.txt")
SERVER_METRICS = "server_metrics_export.json"
REQUESTS = "profile_export.jsonl"
REQUESTS_REDUCED = "overseer-requests.v1"
#: What Overseer reads of each request (overseer: per-request charts and the P90 after warmup).
REQUEST_METADATA = ("benchmark_phase", "request_start_ns", "was_cancelled")
REQUEST_METRICS = (
    "input_sequence_length", "output_sequence_length", "time_to_first_token", "inter_token_latency",
    "request_latency", "output_token_throughput_per_user",
)
#: The server metrics Overseer reads, per engine for its charts.
SERVER_METRIC_NAMES = frozenset({
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:generation_tokens",
    "vllm:prompt_tokens", "vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
    "vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_drafts", "vllm:num_preemptions",
    "sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage", "sglang:generation_tokens",
    "sglang:prompt_tokens", "sglang:cached_tokens", "sglang:spec_accept_length",
})
#: Readings bench_output.json gives under the outcome field of the same name.
BENCH_OUTCOMES = (
    "output_throughput", "total_throughput_per_gpu", "p50_interactivity", "p90_interactivity", "mean_interactivity",
    "p90_ttft_s", "p90_tpot_s", "kv_cache_hit_rate",
)
TEXT_MAX_BYTES = 32 * 1024
CLIP_MARK = "\n… [clipped to the KB's 32 KiB text limit]"


def epoch(value: Any) -> float | None:
    """Seconds since the epoch from Arbor's float stamps or ISO 8601 text; ``None`` for anything else."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    return None


def iso(seconds: float) -> str:
    """UTC ISO 8601 with milliseconds and a ``Z``, as the schema's examples write it."""

    stamp = dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> list[tuple[int, str, dict]]:
    """Each line that holds a JSON object, as (line number, the line as written, the object)."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for number, line in enumerate(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append((number, line, value))
    return rows


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def measured(value: Any) -> float | None:
    """A reading Arbor took; it writes 0.0 or null for an iteration that measured nothing."""

    reading = number(value)
    return reading if reading else None


def clip(text: str) -> tuple[str, bool]:
    """``text`` cut to fit a KB text field, and whether it was cut."""

    data = text.encode()
    if len(data) <= TEXT_MAX_BYTES:
        return text, False
    keep = TEXT_MAX_BYTES - len(CLIP_MARK.encode())
    return data[:keep].decode(errors="ignore") + CLIP_MARK, True


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


@dataclass
class Link:
    """One file a record links: where the export reads it, and where a rebuilt session directory puts it."""

    source: Path
    path: str
    kind: str = "file"
    extra: dict = field(default_factory=dict)

    def entry(self) -> dict:
        return {"path": self.path, "as": self.kind, "sha256": sha256_file(self.source),
                "bytes": self.source.stat().st_size, **self.extra}


@dataclass
class Stage:
    """The export's own directory, where it writes the files a record links that the session does not hold as is."""

    root: Path

    def write(self, rel: str, data: bytes) -> Path:
        path = self.root / "files" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
        return path

    def write_json(self, rel: str, value: Any) -> Path:
        return self.write(rel, (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode())
