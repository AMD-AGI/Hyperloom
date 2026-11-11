# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KV-cache readings sampled from the engine's ``/metrics`` endpoint, and aiperf's phases.

The engine is the only thing that knows how full its KV pool is, and until now nothing read it: a run could spend its
whole budget retracting requests and the breakdown would show only "throughput low". This module is the reader; the
round-level recorder that tags its samples with phases and writes ``kv_metrics.json`` is :mod:`.kv_record`.

``/metrics`` is the primary source rather than ``server.log`` because it wins on every axis that matters here -- full
float precision instead of the log's two decimals, stable metric names instead of names that branch on the model's pool
type, explicit units, and ``kv_evictable_tokens``, which the log cannot express at all and which is the only way to
separate the two occupancy readings this module reports (see :class:`KvSample`). On an AgentX round aiperf scrapes the
same endpoint and exports what it read; :func:`read_aiperf_server_metrics` brings that export into the same shape.

Two rules run through everything below and through :mod:`.kv_record`:

* **Never raise.** This samples a benchmark that is being measured; a scrape failure must cost the run nothing. Every
  entry point returns a value.
* **Never coerce a missing reading to zero.** A pool nobody sampled and a pool that is genuinely empty are different
  findings, and collapsing them is how an optimizer ends up steering on a number that was never measured. Absent means
  ``None``, all the way out to the breakdown.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


__all__ = [
    "KvMetricsPoller",
    "KvSample",
    "RUNNING_REQUEST_GAUGES",
    "aggregate_series",
    "canonical_label_key",
    "families_from_aiperf_record",
    "json_number",
    "parse_prometheus_text",
    "read_aiperf_phases",
    "read_aiperf_server_metrics",
    "running_requests",
    "sample_from_families",
    "server_metrics_exports",
]


#: Scrape timeout. Deliberately well under the 0.5s poll interval of the loop this runs inside: a slow endpoint must not
#: stretch the interval that the stall and soft-deadline gates are measured on. The engine is on loopback, so anything
#: approaching this budget is already pathological.
_SCRAPE_TIMEOUT_SEC = 0.4

_ENCODING = "utf-8"
#: Logs and HTTP bodies are another process's bytes: a stray undecodable byte must not cost the reading.
_DECODE_ERRORS = "ignore"
_NS_PER_SEC = 1e9

#: Opener that ignores the ambient proxy configuration. ``urlopen`` honours ``http_proxy`` by default, which on a
#: corporate host routes a loopback scrape through an external proxy: wrong by construction, and slow enough that the
#: blocking call visibly delays the watchdog loop it runs inside.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

#: Consecutive failures after which the poller stops trying. A server that never exposes ``/metrics`` (SGLang without
#: ``--enable-metrics``) would otherwise pay a connection refusal every couple of seconds for the whole round.
_MAX_CONSECUTIVE_FAILURES = 3

#: ...but not before this much time has passed since the first attempt. The count alone gives up after roughly six
#: seconds, which an engine can easily still be inside after logging that it is ready.
_GIVE_UP_GRACE_SEC = 60.0


# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------

# Occupancy, SGLang. ``token_usage`` is the active reading -- its numerator excludes blocks the prefix cache is holding
# but would hand back under pressure. The per-subpool gauges are always exposed and read 0 on a model that has no such
# subpool, so taking the max over all of them is both the hybrid-correct answer (SGLang's own scheduler judges pressure
# that way) and a no-op on an ordinary KV pool. ``token_usage`` is already max(full, swa, mamba) on current SGLang, so
# it is authoritative on its own. The per-subpool gauges are a fallback for a build that predates it, where reading only
# the full pool would understate a hybrid model's real pressure.
_SGL_USAGE_PRIMARY = "sglang:token_usage"
_SGL_USAGE_FALLBACK = ("sglang:full_token_usage", "sglang:swa_token_usage", "sglang:mamba_usage")
_SGL_USED = "sglang:kv_used_tokens"
_SGL_AVAILABLE = "sglang:kv_available_tokens"
# Only the main KV pool's evictable count. The SWA and Mamba pools have their own capacities, and ``kv_used_tokens`` /
# ``kv_available_tokens`` describe the main pool alone -- folding the other pools' evictable tokens into a ratio built
# from main-pool numerators mixes two different denominators.
_SGL_EVICTABLE = "sglang:kv_evictable_tokens"
# Pool capacity as the engine reports it. Preferred over deriving it, because used + available + evictable can fall
# short of the true size: the gap is tokens held in reserve (protected / session-held), and deriving would both
# understate capacity and overstate the physical occupancy computed from it.
_SGL_CAPACITY_TOKENS = "sglang:max_total_num_tokens"
_SGL_CAPACITY_GB = "sglang:kv_cache_memory_usage_gb"
# Cumulative retract counter. NOT ``sglang:num_retracted_reqs``, which is the most recent batch's instantaneous gauge
# and carries a ``pid`` label; the two names differ by one word and reading the wrong one turns 48 retracts into 1.
_SGL_RETRACT_TOTAL = "sglang:num_retracted_requests_total"
_SGL_CACHED_TOKENS = "sglang:cached_tokens_total"

# Occupancy, vLLM. ``gpu_cache_usage_perc`` is the pre-rename spelling, kept so an older image still reports.
_VLLM_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
# The only workable preemption count: vLLM's log line for it is dead code (``log()`` resets the counter before reading
# it), so this has no log fallback.
_VLLM_PREEMPT_TOTAL = "vllm:num_preemptions_total"
_VLLM_PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
_VLLM_PREFIX_HITS = "vllm:prefix_cache_hits_total"

# Deliberately not read: ``sglang:cache_hit_rate``. Observed reading 0.0 on a server whose ``cached_tokens_total`` had
# already reached 4.6M, so it is not a cumulative rate -- whether it is instantaneous or windowed is unresolved, and a
# field nobody can interpret is worse than an absent one. The raw families stay available to whoever settles it.


ParsedFamilies = dict[str, list[tuple[dict[str, str], float]]]


def canonical_label_key(labels: dict[str, str]) -> str:
    """Render a label set as a stable string usable as a dict key.

    Counters are diffed per series, not in aggregate: an engine restart resets a series to zero, and a naive total would
    read that as a negative delta or, worse, silently absorb it. Sorting makes the key independent of the order the
    exporter happened to emit.
    """
    return ",".join(f'{k}="{labels[k]}"' for k in sorted(labels))


def _split_labels(raw: str) -> dict[str, str]:
    """Parse the inside of a Prometheus label brace into a mapping.

    Values may contain escaped quotes and commas, so the scan is character-wise rather than a naive ``split(",")``.
    """
    labels: dict[str, str] = {}
    index = 0
    length = len(raw)
    while index < length:
        eq = raw.find("=", index)
        if eq < 0:
            break
        name = raw[index:eq].strip().strip(",").strip()
        quote = raw.find('"', eq)
        if quote < 0:
            break
        cursor = quote + 1
        chars: list[str] = []
        while cursor < length:
            char = raw[cursor]
            if char == "\\" and cursor + 1 < length:
                nxt = raw[cursor + 1]
                chars.append({"n": "\n", "t": "\t"}.get(nxt, nxt))
                cursor += 2
                continue
            if char == '"':
                break
            chars.append(char)
            cursor += 1
        if name:
            labels[name] = "".join(chars)
        index = cursor + 1
    return labels


#: aiperf's own export of the engine's ``/metrics``, written into the round's artifact dir.
_SERVER_METRICS_RELPATHS = (
    "aiperf_artifacts/server_metrics_export.json",
    "*/aiperf_artifacts/server_metrics_export.json",
)

#: aiperf's ``CreditPhase`` name for its warmup.
_AIPERF_WARMUP = "warmup"
#: aiperf's ``CreditPhase`` values in its server-metrics export, mapped onto ours. It has no notion of the accuracy eval
#: or of boot, which is why the round's own scrapes still own those.
_AIPERF_PHASE_NAMES = {_AIPERF_WARMUP: "warmup", "profiling": "measured"}


def families_from_aiperf_record(metrics: Any) -> ParsedFamilies:
    """Rebuild the parser's internal shape from one aiperf scrape record.

    aiperf stores ``{name: [{"labels": {...}, "value": x}]}``, which is the same information
    :func:`parse_prometheus_text` produces from the raw exposition -- crucially including the labels, without which the
    per-rank assembly below could not be done at all. Rebuilding rather than re-deriving means both sources go through
    one normalisation, so an aiperf-fed round and a live-scraped one cannot disagree about what a number means.
    """
    families: ParsedFamilies = {}
    if not isinstance(metrics, dict):
        return families
    for name, samples in metrics.items():
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            value = json_number(sample.get("value"))
            if value is None:  # histograms carry buckets instead, and are not read here
                continue
            labels = sample.get("labels")
            families.setdefault(str(name), []).append(
                ({str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {}, value)
            )
    # aiperf names a counter by its Prometheus *family*, which drops the ``_total`` the exposition writes on the sample
    # itself: the text carries ``vllm:num_preemptions_total`` and the export says ``vllm:num_preemptions``. Every
    # cumulative counter this module reads is spelled the exposition way, so without the alias all three -- preemptions,
    # SGLang retractions, cached tokens -- come back empty on this path, and an empty counter is indistinguishable from
    # a round that never retracted. Aliasing here rather than widening every constant keeps one spelling in the lookups
    # and confines the difference to the source that has it. The lists are shared, not copied.
    for name, samples in list(families.items()):
        if not name.endswith("_total"):
            families.setdefault(f"{name}_total", samples)
    return families


def server_metrics_exports(workspace: Any) -> list[Path]:
    """Locate aiperf server-metrics exports in preference order."""
    found: list[Path] = []
    try:
        root = Path(workspace)
        for pattern in _SERVER_METRICS_RELPATHS:
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
    except OSError:
        return []
    return found


def _ns(raw: Any) -> int:
    return int(json_number(raw) or 0)


def _phase_blocks(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], tuple[int, int]]]:
    summary = payload.get("summary")
    ranges = (summary if isinstance(summary, dict) else {}).get("phase_time_ranges") or {}

    def window(name: str) -> tuple[int, int]:
        value = ranges.get(name)
        if not isinstance(value, dict):
            return 0, 0
        return _ns(value.get("start_ns")), _ns(value.get("end_ns"))

    blocks: list[tuple[str, dict[str, Any], tuple[int, int]]] = []
    warmup = payload.get("warmup_metrics")
    if isinstance(warmup, dict):
        blocks.append((_AIPERF_PHASE_NAMES[_AIPERF_WARMUP], warmup, window(_AIPERF_WARMUP)))
    measured = payload.get("metrics")
    if isinstance(measured, dict):
        metrics_phase = str(payload.get("metrics_phase") or "").lower()
        blocks.append((_AIPERF_PHASE_NAMES.get(metrics_phase, "measured"), measured, window(metrics_phase)))
    return blocks


def _timeslices_in_window(series: dict[str, Any], window: tuple[int, int]) -> list[dict[str, Any]]:
    start, end = window
    return [
        item
        for item in (series.get("timeslices") or [])
        if isinstance(item, dict)
        and (not start or _ns(item.get("end_ns")) > start)
        and (not end or _ns(item.get("start_ns")) < end)
    ]


def _series_readings(
    series: dict[str, Any], metric_type: str, window: tuple[int, int], running: float
) -> tuple[list[tuple[int, int, float]], float]:
    counter = metric_type == "counter"
    slices = _timeslices_in_window(series, window)
    rows: list[tuple[int, int, float]] = []
    if counter and slices:
        opening = _ns(slices[0].get("start_ns"))
        if opening > 0:
            # AIPerf stores per-timeslice increments. Emit the opening cumulative snapshot so
            # phase deltas include the first interval instead of starting after it.
            rows.append((opening, opening, running))
    for item in slices:
        end = _ns(item.get("end_ns"))
        value = json_number(item.get("total" if counter else "avg"))
        if end <= 0 or value is None:
            continue
        if counter:
            running += value
            value = running
        rows.append((_ns(item.get("start_ns")), end, value))
    return rows, running


def _metric_readings(
    metric: dict[str, Any], window: tuple[int, int], name: str, cumulative: dict[tuple[str, str], float]
) -> list[tuple[int, int, dict[str, str], float]]:
    rows: list[tuple[int, int, dict[str, str], float]] = []
    metric_type = str(metric.get("type") or "").lower()
    for series in metric.get("series") or []:
        if not isinstance(series, dict):
            continue
        labels = series.get("labels")
        label_map = {str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {}
        key = (name, canonical_label_key(label_map))
        readings, cumulative[key] = _series_readings(series, metric_type, window, cumulative.get(key, 0.0))
        rows += [(start, end, label_map, value) for start, end, value in readings]
    return rows


def read_aiperf_server_metrics(path: Path) -> list[tuple[KvSample, dict[str, Any], str | None]]:
    """Read AIPerf aggregate JSON as ``(sample, timing, phase)`` rows."""
    try:
        payload = json.loads(path.read_text(encoding=_ENCODING))
    except (OSError, ValueError) as exc:
        log.debug("kv_metrics: could not read %s (%s)", path, exc)
        return []
    if not isinstance(payload, dict):
        return []
    cumulative: dict[tuple[str, str], float] = {}
    grouped: dict[tuple[int, int, str], dict[str, list[dict[str, Any]]]] = {}
    for phase, metrics, window in _phase_blocks(payload):
        for name, metric in metrics.items():
            if not isinstance(metric, dict):
                continue
            for start, end, label_map, value in _metric_readings(metric, window, str(name), cumulative):
                group = grouped.setdefault((start, end, phase), {})
                group.setdefault(str(name), []).append({"labels": label_map, "value": value})

    out: list[tuple[KvSample, dict[str, Any], str | None]] = []
    for (start, end, phase), record in sorted(grouped.items()):
        ts = end / _NS_PER_SEC
        sample = sample_from_families(families_from_aiperf_record(record), ts=ts, mono=ts)
        if not sample.has_readings():
            continue
        out.append(
            (
                sample,
                {
                    "scrape_start_unix": start / _NS_PER_SEC,
                    "scrape_end_unix": end / _NS_PER_SEC,
                    "scrape_sec": round(max(0, end - start) / _NS_PER_SEC, 4),
                },
                phase,
            )
        )
    return out


def parse_prometheus_text(text: str) -> ParsedFamilies:
    """Parse a Prometheus text exposition into metric families.

    Handles the parts of the format an engine actually emits: comments, labels, an optional trailing timestamp, and the
    ``NaN`` / ``+Inf`` / ``-Inf`` literals. Non-finite values are dropped -- they carry no occupancy meaning and would
    poison any max or mean taken over them.
    """
    families: ParsedFamilies = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "{" in stripped:
            brace = stripped.index("{")
            close = stripped.rfind("}")
            if close < brace:
                continue
            name = stripped[:brace].strip()
            labels = _split_labels(stripped[brace + 1 : close])
            rest = stripped[close + 1 :].strip()
        else:
            parts = stripped.split(None, 1)
            if len(parts) < 2:
                continue
            name = parts[0]
            labels = {}
            rest = parts[1].strip()
        if not name or not rest:
            continue
        try:
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(value):
            continue
        families.setdefault(name, []).append((labels, value))
    return families


#: Labels that identify a shard of one engine rather than an independent one. Shards schedule in lockstep, so each
#: reports the same event; anything else (``dp_rank``, ``engine``, ``model_name``, ``pid``) marks a unit that retracts
#: on its own account.
_REPLICA_LABELS = frozenset({"tp_rank", "pp_rank"})


def _series(families: ParsedFamilies, name: str) -> dict[str, dict[str, float]]:
    """Read a counter grouped by independent unit, then by shard.

    Two levels because the two label kinds need opposite treatment and a flat map cannot express that: shards of one
    engine duplicate each other's counts, while separate engines contribute their own.
    """
    out: dict[str, dict[str, float]] = {}
    for labels, value in families.get(name, []):
        group = canonical_label_key({k: v for k, v in labels.items() if k not in _REPLICA_LABELS})
        out.setdefault(group, {})[canonical_label_key(labels)] = value
    return out


def aggregate_series(grouped: dict[str, dict[str, float]]) -> float | None:
    """Collapse a grouped counter into one number.

    Max within a group, sum across groups. Eight tensor-parallel ranks carrying 48 retracts describe 48 events, not 384;
    two data-parallel engines carrying 48 each describe 96. Applying one rule to both label kinds is wrong in one
    direction or the other, which is why the grouping exists.
    """
    if not grouped:
        return None
    return sum(max(shards.values()) for shards in grouped.values() if shards)


#: Each engine's gauge of the requests in its running batch.
RUNNING_REQUEST_GAUGES = ("sglang:num_running_reqs", "vllm:num_requests_running")


def running_requests(families: ParsedFamilies) -> float | None:
    """The requests running across the engine's independent units; ``None`` when it exposes no running gauge."""
    for name in RUNNING_REQUEST_GAUGES:
        total = aggregate_series(_series(families, name))
        if total is not None:
            return total
    return None


def _rank_keys(families: ParsedFamilies, names: tuple[str, ...]) -> list[str]:
    """Label keys the given gauges were reported under."""
    keys: list[str] = []
    for name in names:
        for labels, _ in families.get(name, []):
            key = canonical_label_key(labels)
            if key not in keys:
                keys.append(key)
    return keys or [""]


def _read(families: ParsedFamilies, name: str, key: str) -> float | None:
    """One gauge's value for one rank.

    Falls back to an unlabelled series so a metric the engine reports once globally still resolves for every rank.
    """
    series = families.get(name) or []
    for labels, value in series:
        if canonical_label_key(labels) == key:
            return value
    if len(series) == 1 and not series[0][0]:
        return series[0][1]
    return None


@dataclass(frozen=True)
class KvSample:
    """One ``/metrics`` scrape, normalised across engines.

    Every metric is ``None`` when the engine did not expose it. Read them with ``is None``, never with truthiness: an
    idle pool reports a true 0.0, and an engine that has not been asked for a token yet reports 0.0 for minutes.

    The two occupancy readings are not interchangeable. ``active_pool_usage`` is the pressure reading; a pool at 100%
    ``physical_pool_usage`` can be under no pressure at all, because the difference is prefix cache holding blocks it
    would hand straight back. Reporting either one as "the" occupancy is how a healthy run gets read as a saturated one.
    """

    ts: float  # wall clock, for lining up with externally collected data
    mono: float  # monotonic clock, for intervals within the run
    engine: str = ""  # "sglang", "vllm", or "" when unrecognised
    active_pool_usage: float | None = None
    physical_pool_usage: float | None = None  # SGLang only; vLLM cannot express it
    used_tokens: float | None = None
    evictable_tokens: float | None = None
    available_tokens: float | None = None
    capacity_tokens: float | None = None
    # Set when no capacity gauge was exposed and capacity had to be summed from used + available + evictable. That sum
    # can fall short of the real pool -- tokens held in reserve belong to none of the three -- which makes
    # ``physical_pool_usage`` an upper bound rather than a measurement.
    capacity_derived: bool = False
    # Label series the occupancy gauge carried. Above one means a sharded engine reported per-rank views that were
    # collapsed to their maximum; see :func:`_scalar`.
    series_count: int = 0
    capacity_gb: float | None = None  # the engine's "GB" is 1024-based; do not rescale
    # Cumulative counters, grouped by independent unit then by shard. Kept per series so an engine restart shows as one
    # series resetting rather than as a total going backwards, and so shards can be collapsed while separate engines are
    # added. Combine with :func:`aggregate_series`.
    retract_total: dict[str, dict[str, float]] = field(default_factory=dict)  # SGLang
    preempt_total: dict[str, dict[str, float]] = field(default_factory=dict)  # vLLM
    prefix_cache_queries: dict[str, dict[str, float]] = field(default_factory=dict)  # vLLM
    prefix_cache_hits: dict[str, dict[str, float]] = field(default_factory=dict)  # vLLM
    # SGLang. Empty when prefix caching is off -- the metric is not emitted at all, rather than emitted as zero.
    cached_tokens_total: dict[str, dict[str, float]] = field(default_factory=dict)

    def has_readings(self) -> bool:
        """Whether this scrape carried any KV signal at all.

        A reachable endpoint that exposes no KV metric -- an engine started without them, or one whose exposition this
        module does not recognise -- must not be recorded as a row of nothing.
        """
        return any(
            value is not None
            for value in (
                self.active_pool_usage,
                self.physical_pool_usage,
                self.used_tokens,
                self.evictable_tokens,
                self.available_tokens,
                self.capacity_tokens,
                self.capacity_gb,
            )
        ) or bool(
            self.retract_total
            or self.preempt_total
            or self.prefix_cache_queries
            or self.prefix_cache_hits
            or self.cached_tokens_total
        )


def _outranks(candidate: dict[str, Any], best: dict[str, Any]) -> bool:
    for metric in ("active", "physical", "used"):
        mine, theirs = candidate.get(metric), best.get(metric)
        if mine is None and theirs is None:
            continue
        return theirs is None or (mine is not None and mine > theirs)
    return False


def _detect_engine(families: ParsedFamilies) -> str:
    """Identify the engine from its metric-name prefix."""
    for name in families:
        if name.startswith("sglang:"):
            return "sglang"
        if name.startswith("vllm:"):
            return "vllm"
    return ""


def sample_from_families(
    families: ParsedFamilies,
    *,
    ts: float | None = None,
    mono: float | None = None,
) -> KvSample:
    """Build a normalised sample from a parsed exposition."""
    engine = _detect_engine(families)
    occupancy_names = (_SGL_USAGE_PRIMARY,) + _SGL_USAGE_FALLBACK + _VLLM_USAGE + (_SGL_USED,)
    keys = _rank_keys(families, occupancy_names)

    # Assemble each rank in full before choosing one. Taking a per-field max across ranks silently welds numbers from
    # different sides of the engine together: rank A's capacity against rank B's used tokens produced a physical
    # occupancy of 1.7 on two individually valid readings.
    best: dict[str, Any] = {}
    for key in keys:
        used = _read(families, _SGL_USED, key)
        available = _read(families, _SGL_AVAILABLE, key)
        evictable = _read(families, _SGL_EVICTABLE, key)

        # Engine-reported capacity wins; the sum is a fallback for builds that do not expose it, flagged so a consumer
        # knows it may run short of the true pool size.
        capacity = _read(families, _SGL_CAPACITY_TOKENS, key)
        evicted = evictable or 0.0
        capacity_derived = False
        if capacity is None and used is not None and available is not None:
            capacity = used + available + evicted
            capacity_derived = True

        physical: float | None = None
        if capacity and capacity > 0 and used is not None:
            physical = (used + evicted) / capacity

        # ``token_usage`` is already the maximum across the full, SWA and Mamba subpools on current SGLang, so it is
        # read directly. The per-subpool gauges are only consulted when it is missing, which is what an older build
        # looks like.
        active = _read(families, _SGL_USAGE_PRIMARY, key)
        for names in (_SGL_USAGE_FALLBACK, _VLLM_USAGE):
            if active is None:
                active = max((v for n in names if (v := _read(families, n, key)) is not None), default=None)

        candidate = {
            "active": active,
            "physical": physical,
            "used": used,
            "available": available,
            "evictable": evictable,
            "capacity": capacity,
            "capacity_derived": capacity_derived,
            "capacity_gb": _read(families, _SGL_CAPACITY_GB, key),
        }
        # The most pressured rank is the one that will retract, so it is the one worth reporting. Fall back to physical,
        # then to having read anything.
        if not best or _outranks(candidate, best):
            best = candidate

    return KvSample(
        ts=time.time() if ts is None else ts,
        mono=time.monotonic() if mono is None else mono,
        engine=engine,
        active_pool_usage=best.get("active"),
        physical_pool_usage=best.get("physical"),
        used_tokens=best.get("used"),
        evictable_tokens=best.get("evictable"),
        available_tokens=best.get("available"),
        capacity_tokens=best.get("capacity"),
        capacity_derived=bool(best.get("capacity_derived")),
        series_count=len(keys) if keys != [""] else 0,
        # From the rank that was assembled, not re-read: a fresh lookup would be free to land on a different rank than
        # every field above it.
        capacity_gb=best.get("capacity_gb"),
        retract_total=_series(families, _SGL_RETRACT_TOTAL),
        preempt_total=_series(families, _VLLM_PREEMPT_TOTAL),
        prefix_cache_queries=_series(families, _VLLM_PREFIX_QUERIES),
        prefix_cache_hits=_series(families, _VLLM_PREFIX_HITS),
        cached_tokens_total=_series(families, _SGL_CACHED_TOKENS),
    )


class KvMetricsPoller:
    """Scrapes one engine's ``/metrics``, degrading quietly when it cannot.

    Availability is tri-state and sticky. Until the first successful scrape the poller is ``unknown``; a scrape that
    lands makes it available for good; a run of consecutive failures parks it as unavailable and it stops issuing
    requests, because the common cause is an engine started without ``--enable-metrics``, and that will not fix itself
    mid-round.
    """

    def __init__(self, url: str) -> None:
        """Bind the poller to the engine's ``/metrics`` ``url`` without contacting it."""
        self.url = url
        self._failures = 0
        self._succeeded = False
        self._gave_up = False
        self._warned = False
        self._first_attempt_mono: float | None = None

    @property
    def available(self) -> bool | None:
        """Tri-state reachability of the endpoint."""
        if self._succeeded:
            return True
        if self._gave_up:
            return False
        return None

    def fetch(self) -> str | None:
        """Scrape once."""
        if self._gave_up:
            return None
        now = time.monotonic()
        if self._first_attempt_mono is None:
            self._first_attempt_mono = now
        try:
            with _OPENER.open(self.url, timeout=_SCRAPE_TIMEOUT_SEC) as response:
                body = response.read().decode(_ENCODING, _DECODE_ERRORS)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            self._failures += 1
            # Both a count and a clock. On the count alone, three misses inside six seconds park the poller for the rest
            # of the round -- and an engine that has logged "ready" can still be a few seconds from accepting a request,
            # so a round would go dark for the whole of it over a start that was merely slow. The grace makes giving up
            # mean "nothing answered here for a minute", which is the case the sticky give-up was written for.
            waited = now - (self._first_attempt_mono or now)
            if self._failures >= _MAX_CONSECUTIVE_FAILURES and waited >= _GIVE_UP_GRACE_SEC and not self._succeeded:
                self._gave_up = True
                if not self._warned:
                    self._warned = True
                    log.info(
                        "kv_metrics: %s unreachable after %d attempts over %.0fs (%s); KV metrics "
                        "recorded as unavailable for this round. SGLang needs --enable-metrics (vLLM exposes it "
                        "by default).",
                        self.url,
                        self._failures,
                        waited,
                        exc,
                    )
            return None
        self._failures = 0
        self._succeeded = True
        return body

    def sample(self) -> KvSample | None:
        """Scrape and normalise in one step."""
        body = self.fetch()
        if body is None:
            return None
        sample = sample_from_families(parse_prometheus_text(body))
        return sample if sample.has_readings() else None


def json_number(raw: Any) -> float | None:
    """Coerce a JSON value to a float, or ``None`` when it is not one.

    ``bool`` is excluded on purpose: it is a subclass of ``int``, and a ``True`` where a timestamp belongs would
    otherwise become 1.0 -- a Unix time in 1970 that the sanity check would then reject for the wrong reason.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) if math.isfinite(raw) else None


def read_aiperf_phases(url: str) -> dict[str, Any] | None:
    """The ``phases`` mapping aiperf's progress API at ``url`` reports; ``None`` when its answer carries none.

    Raises:
        urllib.error.URLError, http.client.HTTPException, OSError, ValueError: The API could not be read.
    """
    with _OPENER.open(f"{url.rstrip('/')}/api/progress", timeout=_SCRAPE_TIMEOUT_SEC) as response:
        payload = json.loads(response.read().decode(_ENCODING, _DECODE_ERRORS))
    phases = payload.get("phases") if isinstance(payload, dict) else None
    return phases if isinstance(phases, dict) else None
