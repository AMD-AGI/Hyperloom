# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KV-cache observability sampled from the engine's ``/metrics`` endpoint.

The engine is the only thing that knows how full its KV pool is, and until now nothing read it: a run could spend its
whole budget retracting requests and the breakdown would show only "throughput low". This module is the reader.

``/metrics`` is the primary source rather than ``server.log`` because it wins on every axis that matters here -- full
float precision instead of the log's two decimals, stable metric names instead of names that branch on the model's pool
type, explicit units, and ``kv_evictable_tokens``, which the log cannot express at all and which is the only way to
separate the two occupancy readings this module reports (see :class:`KvSample`).

There are two collectors, and the artifact says which one produced its rows in ``sample_source``. On an AgentX round
aiperf is already scraping the same endpoint for its own purposes, so its export is adopted wholesale: it samples on a
tighter cadence, takes a reading at each phase boundary of its own accord, and stamps every record with the phase from
the process that owns the transition -- which removes the boundary lag rather than correcting for it. Everything else
-- synthetic benchmarks, the accuracy eval, boot, and any round killed before aiperf flushed -- has no such export, and
is covered by scraping from the watchdog loop below.

Phase attribution has two sources, and the artifact always says which one it used. Preferred is aiperf's progress API,
whose ``phases.<name>.start_ns`` is stamped when the phase actually begins; the fallback is the ``aiperf.log`` line the
watchdog greps for, which is written after the transition and read on the next poll. Both lags land on the boundary, so
under the fallback the samples either side of it are systematically credited to the phase that just ended. When the
authoritative stamps are available the rows are re-labelled and the counter windows re-bracketed from them at close --
which is why every row carries its raw per-series counters, and not just a phase-level first/last pair.

Below the phase, the workload timeline comes from aiperf's per-request export; see :mod:`._agentx_timeline`. That gives
each row the trajectories, turns and requests in flight during its scrape window, and writes the event stream beside
this artifact.

What this does not do is stop the workload at a boundary. An exact counter total needs the client to quiesce, take one
snapshot, and resume, and aiperf exposes no such control: the boundary here is exact to aiperf's own stamp, and the
counter attribution to the scrape interval either side of it.

Two rules run through everything below:

* **Never raise.** This samples a benchmark that is being measured; a scrape failure must cost the run nothing. Every
  entry point returns a value.
* **Never coerce a missing reading to zero.** A pool nobody sampled and a pool that is genuinely empty are different
  findings, and collapsing them is how an optimizer ends up steering on a number that was never measured. Absent means
  ``None``, all the way out to the breakdown.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_METRICS_PORT",
    "KV_ARTIFACT_NAME",
    "PHASES",
    "AiperfProgressPoller",
    "KvMetricsPoller",
    "KvMetricsRecorder",
    "KvSample",
    "canonical_label_key",
    "counter_delta",
    "families_from_aiperf_record",
    "find_server_metrics_export",
    "parse_prometheus_text",
    "read_aiperf_server_metrics",
    "resolve_metrics_port",
    "sample_from_families",
]


#: Port the persistent server binds when ``benchmark.envs.PORT`` is unset. Mirrors
#: ``_server_lifecycle.REUSE_PORT_DEFAULT``; duplicated rather than imported because that module imports
#: ``_subprocess_kill``, which is where this one gets wired in.
DEFAULT_METRICS_PORT = 8888

#: Scrape timeout. Deliberately well under the 0.5s poll interval of the loop this runs inside: a slow endpoint must not
#: stretch the interval that the stall and soft-deadline gates are measured on. The engine is on loopback, so anything
#: approaching this budget is already pathological.
_SCRAPE_TIMEOUT_SEC = 0.4

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
_VLLM_PREFIX_QUERIES = "vllm:prefix_cache_queries"
_VLLM_PREFIX_HITS = "vllm:prefix_cache_hits"

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
    "aiperf_artifacts/server_metrics_export.jsonl",
    "*/aiperf_artifacts/server_metrics_export.jsonl",
)

#: aiperf's ``CreditPhase`` values, mapped onto ours. It has no notion of the accuracy eval or of boot, which is why the
#: live path still covers those.
_CREDIT_PHASE_NAMES = {"warmup": "warmup", "profiling": "measured"}


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
            value = _number(sample.get("value"))
            if value is None:  # histograms carry buckets instead, and are not read here
                continue
            labels = sample.get("labels")
            families.setdefault(str(name), []).append(
                ({str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {}, value)
            )
    return families


def find_server_metrics_export(workspace: Any) -> Path | None:
    """Locate aiperf's server-metrics export under a round's directory."""
    try:
        root = Path(workspace)
        for pattern in _SERVER_METRICS_RELPATHS:
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_file():
                    return candidate
    except OSError:
        return None
    return None


def read_aiperf_server_metrics(path: Path) -> list[tuple[KvSample, dict[str, Any], str | None]]:
    """Read aiperf's scrape records as ``(sample, timing, phase)``, oldest first.

    Every field the live path measures for itself is already on the record: ``timestamp_ns`` for the reading,
    ``request_sent_ns`` and ``endpoint_latency_ns`` for the round trip, and ``benchmark_phase`` for the phase. That last
    one is why this source is preferred where it exists -- aiperf stamps the phase at collection time, from the process
    that owns the transition, so there is no lag between the boundary and the label and nothing to re-attribute.
    """
    out: list[tuple[KvSample, dict[str, Any], str | None]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                stamp = _number(record.get("timestamp_ns"))
                if stamp is None or stamp <= 0:
                    continue
                families = families_from_aiperf_record(record.get("metrics"))
                if not families:
                    continue
                ts = stamp / 1e9
                sample = sample_from_families(families, ts=ts, mono=ts)
                if not sample.has_readings():
                    continue
                started = _number(record.get("request_sent_ns"))
                latency = _number(record.get("endpoint_latency_ns")) or 0.0
                start_unix = started / 1e9 if started and started > 0 else ts
                out.append(
                    (
                        sample,
                        {
                            # Widened to the enclosing millisecond for the same reason the live path does it: these
                            # bound the interval workload records are joined against.
                            "scrape_start_unix": math.floor(start_unix * 1000) / 1000,
                            "scrape_end_unix": math.ceil((start_unix + latency / 1e9) * 1000) / 1000,
                            "scrape_sec": round(latency / 1e9, 4),
                        },
                        _CREDIT_PHASE_NAMES.get(str(record.get("benchmark_phase") or "").lower()),
                    )
                )
    except OSError as exc:
        log.debug("kv_metrics: could not read %s (%s)", path, exc)
        return []
    out.sort(key=lambda item: item[0].ts)
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
                self.capacity_gb,
            )
        ) or bool(self.retract_total or self.preempt_total)


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
    best: dict[str, Any] | None = None
    for key in keys:
        used = _read(families, _SGL_USED, key)
        available = _read(families, _SGL_AVAILABLE, key)
        evictable = _read(families, _SGL_EVICTABLE, key)

        # Engine-reported capacity wins; the sum is a fallback for builds that do not expose it, flagged so a consumer
        # knows it may run short of the true pool size.
        capacity = _read(families, _SGL_CAPACITY_TOKENS, key)
        capacity_derived = False
        if capacity is None and used is not None and available is not None:
            capacity = used + available + (evictable or 0.0)
            capacity_derived = True

        physical: float | None = None
        if capacity and capacity > 0 and used is not None:
            physical = (used + (evictable or 0.0)) / capacity

        # ``token_usage`` is already the maximum across the full, SWA and Mamba subpools on current SGLang, so it is
        # read directly. The per-subpool gauges are only consulted when it is missing, which is what an older build
        # looks like.
        active = _read(families, _SGL_USAGE_PRIMARY, key)
        if active is None:
            active = max(
                (v for n in _SGL_USAGE_FALLBACK if (v := _read(families, n, key)) is not None),
                default=None,
            )
        if active is None:
            active = max(
                (v for n in _VLLM_USAGE if (v := _read(families, n, key)) is not None),
                default=None,
            )

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
        if best is None:
            best = candidate
            continue
        for metric in ("active", "physical", "used"):
            mine, theirs = candidate.get(metric), best.get(metric)
            if mine is None and theirs is None:
                continue
            if theirs is None or (mine is not None and mine > theirs):
                best = candidate
            break

    assembled = best or {}
    used = assembled.get("used")
    available = assembled.get("available")
    evictable = assembled.get("evictable")
    capacity = assembled.get("capacity")
    capacity_derived = bool(assembled.get("capacity_derived"))
    physical = assembled.get("physical")
    active = assembled.get("active")

    return KvSample(
        ts=time.time() if ts is None else ts,
        mono=time.monotonic() if mono is None else mono,
        engine=engine,
        active_pool_usage=active,
        physical_pool_usage=physical,
        used_tokens=used,
        evictable_tokens=evictable,
        available_tokens=available,
        capacity_tokens=capacity,
        capacity_derived=capacity_derived,
        series_count=len(keys) if keys != [""] else 0,
        # From the rank that was assembled, not re-read: a fresh lookup would be free to land on a different rank than
        # every field above it.
        capacity_gb=assembled.get("capacity_gb"),
        retract_total=_series(families, _SGL_RETRACT_TOTAL),
        preempt_total=_series(families, _VLLM_PREEMPT_TOTAL),
        prefix_cache_queries=_series(families, _VLLM_PREFIX_QUERIES),
        prefix_cache_hits=_series(families, _VLLM_PREFIX_HITS),
        cached_tokens_total=_series(families, _SGL_CACHED_TOKENS),
    )


def _port_value(raw: Any) -> int | None:
    """One PORT reading as a usable port number, or ``None``."""
    if raw in (None, ""):
        return None
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return port if port > 0 else None


def port_from_workspace(workspace: Any) -> int | None:
    """Read ``benchmark.envs.PORT`` out of the round's materialized config.

    This is the only place the port is reliably written. The subprocess environment does not carry it -- the port is
    pinned in the YAML that Magpie reads, not exported to the parent -- so resolving from the environment alone lands
    on the default and scrapes a port nothing is listening on. Observed on a real session: two of three rounds bound an
    ephemeral 34407 while collection sat on 8888 and recorded the engine as unavailable, and the one round that worked
    did so only because it happened to bind the default.
    """
    try:
        root = Path(workspace)
        # The round's own config first, then the benchmark's copy, which only exists once the client has started.
        for pattern in ("*.yaml", "*/config.yaml"):
            for candidate in sorted(root.glob(pattern)):
                try:
                    import yaml

                    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - a config we cannot read is not a reason to fail a round
                    continue
                if not isinstance(payload, dict):
                    continue
                envs = (
                    (payload.get("benchmark") or {}).get("envs") if isinstance(payload.get("benchmark"), dict) else None
                )
                port = _port_value(envs.get("PORT")) if isinstance(envs, dict) else None
                if port is not None:
                    return port
    except OSError:
        return None
    return None


def resolve_metrics_port(config_envs: dict[str, Any] | None = None, workspace: Any = None) -> int:
    """Resolve the port the engine serves ``/metrics`` on.

    The server binds whatever ``benchmark.envs.PORT`` the materialized YAML pins -- an ephemeral port assigned per
    session, not a constant. The YAML is therefore the authority; the caller's env and the ambient env are fallbacks for
    paths that never materialize one, and the default is a last resort that is only ever right by coincidence.
    """
    for source in (config_envs or {}, os.environ):
        port = _port_value(source.get("PORT"))
        if port is not None:
            return port
    if workspace is not None:
        port = port_from_workspace(workspace)
        if port is not None:
            return port
    return DEFAULT_METRICS_PORT


class KvMetricsPoller:
    """Scrapes one engine's ``/metrics``, degrading quietly when it cannot.

    Built to sit inside the executors' existing 0.5s watchdog loop rather than in a process of its own: a sampler racing
    the engine for CPU would show up in the very latency numbers the round exists to measure.

    Availability is tri-state and sticky. Until the first successful scrape the poller is ``unknown``; a scrape that
    lands makes it available for good; a run of consecutive failures parks it as unavailable and it stops issuing
    requests, because the common cause is an engine started without ``--enable-metrics``, and that will not fix itself
    mid-round.
    """

    def __init__(
        self,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        config_envs: dict[str, Any] | None = None,
        timeout_sec: float = _SCRAPE_TIMEOUT_SEC,
        workspace: Any = None,
    ) -> None:
        """Bind the poller to an endpoint without contacting it."""
        self.port = int(port) if port else resolve_metrics_port(config_envs, workspace)
        self.url = f"http://{host}:{self.port}/metrics"
        self.timeout_sec = float(timeout_sec)
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
            with _OPENER.open(self.url, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8", "ignore")
        except (urllib.error.URLError, OSError, ValueError) as exc:
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
                        "recorded as unavailable for this round. Check the port against the round's "
                        "benchmark.envs.PORT; SGLang also needs --enable-metrics (vLLM exposes it by default).",
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


#: Address file ``aiperf_client.sh`` publishes so this process can find the progress API. Searched under the round's own
#: directory first, then one level down, matching how the watchdog resolves a nested benchmark's logs.
_PROGRESS_ADDRESS_RELPATHS = ("aiperf_artifacts/progress_api.json", "*/aiperf_artifacts/progress_api.json")

#: aiperf's phase names, mapped onto ours. It has no notion of the accuracy eval, which is why ``eval`` is not here and
#: stays owned by the log markers.
_AIPERF_PHASE_NAMES = {"warmup": "warmup", "profiling": "measured"}

#: How far an authoritative timestamp may sit from this process's clock before it is rejected. Wide, because it is only
#: meant to catch a clock domain that is not the Unix epoch at all -- a monotonic ``start_ns`` is off by decades, not by
#: hours -- rather than to police skew.
_PROGRESS_CLOCK_SANITY_SEC = 86400.0


def _number(raw: Any) -> float | None:
    """Coerce a JSON value to a float, or ``None`` when it is not one.

    ``bool`` is excluded on purpose: it is a subclass of ``int``, and a ``True`` where a timestamp belongs would
    otherwise become 1.0 -- a Unix time in 1970 that the sanity check would then reject for the wrong reason.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) if math.isfinite(raw) else None


class AiperfProgressPoller:
    """Reads AIPerf's phase timeline from its local progress API.

    This is the authority on when a phase actually began. The alternative -- and what the recorder falls back to -- is
    grepping ``aiperf.log`` for a human-readable line, which aiperf writes after the transition and the watchdog then
    reads on its next pass. Those two lags land squarely on the boundary, so warmup traffic gets counted as measured.

    Everything here degrades to ``None``: the endpoint only exists when the round is an AgentX one whose client managed
    to publish an address, and a round that cannot reach it must still produce KV metrics.
    """

    def __init__(self, workspace: Any, *, timeout_sec: float = _SCRAPE_TIMEOUT_SEC) -> None:
        """Bind to a round's directory without reading anything from it."""
        self._workspace = workspace
        self.timeout_sec = float(timeout_sec)
        self.url: str | None = None
        self._resolved = False
        self._failures = 0
        self._succeeded = False
        self._gave_up = False

    def _resolve_url(self) -> str | None:
        """Find the published address, once the client has written it.

        Resolution is retried until it succeeds because the address file appears when the benchmark client starts, which
        is after the server is up and therefore after the first scrapes have already happened.
        """
        if self.url is not None:
            return self.url
        try:
            root = Path(self._workspace)
            for pattern in _PROGRESS_ADDRESS_RELPATHS:
                for candidate in sorted(root.glob(pattern)):
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    url = payload.get("url") if isinstance(payload, dict) else None
                    if isinstance(url, str) and url.startswith("http"):
                        self.url = url
                        return url
        except (OSError, ValueError, TypeError):
            return None
        return None

    def poll(self) -> dict[str, Any] | None:
        """Return the raw ``phases`` mapping, or ``None`` when it cannot be read."""
        if self._gave_up:
            return None
        url = self._resolve_url()
        if url is None:
            return None
        try:
            with _OPENER.open(f"{url.rstrip('/')}/api/progress", timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8", "ignore"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._failures += 1
            # Only give up on an endpoint that never worked. One that answered before and is failing now is an aiperf
            # that has exited, and the timeline already collected is still the authority for this round.
            if self._failures >= _MAX_CONSECUTIVE_FAILURES and not self._succeeded:
                self._gave_up = True
                log.debug("kv_metrics: aiperf progress API unreachable at %s (%s)", url, exc)
            return None
        self._failures = 0
        self._succeeded = True
        phases = payload.get("phases") if isinstance(payload, dict) else None
        return phases if isinstance(phases, dict) else None


#: Artifact the recorder writes, alongside the round's ``server.log``.
KV_ARTIFACT_NAME = "kv_metrics.json"

#: Collection phases. An engine process outlives the window that is actually being measured by a wide margin, and mixing
#: them makes every statistic meaningless: ``boot`` has no traffic at all, ``warmup`` runs a deliberately cold cache,
#: and ``eval`` drives accuracy traffic whose shape has nothing to do with the throughput benchmark. Only ``measured``
#: may enter a comparison.
PHASES = ("boot", "warmup", "measured", "eval")

#: The one phase whose numbers may be compared against another round's.
_COMPARABLE_PHASE = "measured"

#: Row cap before stride downsampling, mirroring ``_MN_GPU_SAMPLE_CAP``. A three-hour round at the scrape interval below
#: lands well over this.
_MAX_STORED_ROWS = 5000

#: Floor between scrapes, enforced on the monotonic clock -- the loop this runs in iterates faster than its nominal
#: period when a deadline bounds the slice, so counting passes would sample fastest exactly when the run is most loaded.
#:
#: Not the loop's 0.5s. A scrape is a synchronous HTTP round trip plus a parse of the engine's whole exposition, which
#: is not free at that rate, and the engine's own gauges do not refresh anywhere near it -- most of those samples would
#: be the same numbers read again. Two seconds keeps the trend visible at a fraction of the cost. Override with
#: ``INFERENCE_OPTIMIZER_KV_SCRAPE_INTERVAL_SEC``.
_SCRAPE_INTERVAL_ENV = "INFERENCE_OPTIMIZER_KV_SCRAPE_INTERVAL_SEC"
_DEFAULT_SCRAPE_INTERVAL_SEC = 2.0


def resolve_scrape_interval_sec() -> float:
    """Seconds between scrapes, from the environment or the default."""
    raw = os.environ.get(_SCRAPE_INTERVAL_ENV, "").strip()
    if not raw:
        return _DEFAULT_SCRAPE_INTERVAL_SEC
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SCRAPE_INTERVAL_SEC
    return value if value >= 0 else _DEFAULT_SCRAPE_INTERVAL_SEC


def counter_delta(first: dict[str, dict[str, float]], last: dict[str, dict[str, float]]) -> float | None:
    """Increment of a labelled counter between two observations.

    Diffed per label series rather than on a flat total, because an engine restart zeroes its series: a flat subtraction
    would go negative, and clamping that to zero would quietly discard the whole window. When a series ends below where
    it started it is treated as having restarted, and only the post-restart count is credited -- the increments before
    the restart are genuinely unrecoverable, and inventing them would be worse than losing them.
    """
    if not first and not last:
        return None
    deltas: dict[str, dict[str, float]] = {}
    for group, shards in last.items():
        opened = first.get(group) or {}
        for key, end in shards.items():
            start = opened.get(key)
            deltas.setdefault(group, {})[key] = end if start is None or end < start else end - start
    return aggregate_series(deltas) or 0.0


class KvMetricsRecorder:
    """Collects phase-tagged KV samples for one benchmark round.

    Owns everything stateful about collection so the watchdog loop it hangs off keeps a single call per pass and one
    ``finally``. Like the poller, nothing here raises: a recorder that fails must cost the round nothing.

    The phase machine is driven by markers the loop already detects, plus aiperf's own phase lines under AgentX. It is
    never inferred from elapsed time, because the boundaries move by tens of minutes between rounds -- an AgentX warmup
    alone was measured at nearly 18 of them.
    """

    def __init__(
        self,
        *,
        poller: KvMetricsPoller,
        output_path: str | None = None,
        scope: dict[str, Any] | None = None,
        min_interval_sec: float | None = None,
        progress: AiperfProgressPoller | None = None,
        workspace: Any = None,
    ) -> None:
        """Prepare a recorder without contacting anything."""
        self._poller = poller
        self._output_path = output_path
        # The round's own directory, which is where both aiperf's export and our artifacts live. Derived from the
        # output path when not given, so the two can never point at different rounds.
        self._workspace = workspace if workspace is not None else (Path(output_path).parent if output_path else None)
        self._scope = dict(scope or {})
        self._min_interval = resolve_scrape_interval_sec() if min_interval_sec is None else float(min_interval_sec)
        self._progress = progress
        self._phase_timeline: dict[str, dict[str, Any]] = {}
        self._phase = "boot"
        self._rows: list[dict[str, Any]] = []
        self._last_scrape_mono: float | None = None
        self._phase_marks: list[dict[str, Any]] = []
        self._first_counters: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        self._last_counters: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        self._capacity_tokens: float | None = None
        self._capacity_derived = False
        self._capacity_gb: float | None = None
        self._series_count = 0
        self._prefix_first: dict[str, float] = {}
        self._prefix_last: dict[str, float] = {}
        self._closed = False

    @property
    def phase(self) -> str:
        """Current collection phase."""
        return self._phase

    def note_phase(self, phase: str, mono: float) -> None:
        """Record a phase transition."""
        if phase not in PHASES or phase == self._phase:
            return
        # One reading taken at the boundary closes the phase that is ending and opens the one beginning. Deriving a
        # phase total from its own first and last periodic samples instead leaves a gap at each end -- up to a full
        # interval of activity credited to neither phase -- and the gap lands exactly where a phase change makes the
        # engine's behaviour change most.
        boundary = None if self._closed else self._scrape(mono)
        self._phase = phase
        self._phase_marks.append({"phase": phase, "mono": round(float(mono), 3), "ts": time.time()})
        if boundary is not None:
            self._record_counters(boundary)

    def tick(self, mono: float) -> None:
        """Scrape if the interval has elapsed, tagging the sample with the phase."""
        if self._closed:
            return
        if self._last_scrape_mono is not None and (mono - self._last_scrape_mono) < self._min_interval:
            return
        self._scrape(mono)

    def _scrape(self, mono: float) -> KvSample | None:
        """Take one reading unconditionally, timing the round trip.

        Both ends of the round trip are recorded, not just the duration. A gauge read over a 300 ms scrape describes
        some instant inside that window and the consumer cannot say which, so a bracket is the honest representation;
        a single stamp would invite lining KV samples up against a workload timeline to a precision that was never
        measured.
        """
        self._last_scrape_mono = mono
        # Polled first, and regardless of whether the engine answers: the phase timeline is what makes any of the rows
        # comparable, and an engine that is briefly unreachable must not cost us the boundary.
        workload = self._poll_progress()
        start_mono = time.monotonic()
        start_unix = time.time()
        try:
            sample = self._poller.sample()
        except Exception:  # noqa: BLE001 - collection must never fail a round
            log.debug("kv_metrics: sample failed", exc_info=True)
            return None
        end_mono = time.monotonic()
        if sample is None:
            return None
        # Widened to the enclosing millisecond, not rounded to the nearest one. These two bound the interval that
        # workload records are joined against, so the stored window has to contain the real one: nearest-rounding
        # shrinks it by up to half a millisecond at each end, and a request that began inside a sub-millisecond scrape
        # then falls outside the window that observed it. Measured at ~13% of scrapes on a host fast enough for the
        # rounding to bite, and invisible on a coarse clock -- which is exactly how it reached CI unnoticed.
        elapsed = end_mono - start_mono
        self._absorb(
            sample,
            timing={
                "scrape_start_unix": math.floor(start_unix * 1000) / 1000,
                "scrape_end_unix": math.ceil((start_unix + elapsed) * 1000) / 1000,
                "scrape_start_mono": round(start_mono, 3),
                "scrape_end_mono": round(end_mono, 3),
                "scrape_sec": round(elapsed, 4),
            },
            workload=workload,
        )
        return sample

    def _poll_progress(self) -> dict[str, Any] | None:
        """Fold one progress reading into the timeline and return the phase in flight.

        The timeline accumulates rather than replaces: aiperf drops a phase from its report once the next one begins, so
        keeping only the latest response would lose the warmup boundary the moment profiling starts.
        """
        if self._progress is None:
            return None
        try:
            phases = self._progress.poll()
        except Exception:  # noqa: BLE001 - collection must never fail a round
            log.debug("kv_metrics: progress poll failed", exc_info=True)
            return None
        if not phases:
            return None
        active: dict[str, Any] | None = None
        active_start = -1.0
        for name, stats in phases.items():
            if not isinstance(stats, dict):
                continue
            self._phase_timeline.setdefault(name, {}).update(stats)
            start_ns = _number(stats.get("start_ns"))
            if start_ns is None or stats.get("requests_end_ns") is not None:
                continue
            if start_ns > active_start:
                active_start = start_ns
                # Timestamps live in the timeline; a row only needs to say which phase it fell in and how much work was
                # in flight, which is the granularity aiperf actually exposes.
                active = {"aiperf_phase": name, "stats": {k: v for k, v in stats.items() if not k.endswith("_ns")}}
        return active

    def _absorb(
        self,
        sample: KvSample,
        *,
        timing: dict[str, float] | None = None,
        workload: dict[str, Any] | None = None,
    ) -> None:
        """Fold one sample into the row buffer and the counter windows."""
        # Pool capacity is only observable while the engine is up; latch the first non-null reading so the artifact
        # still carries it after a round that ended with the server gone.
        if self._capacity_tokens is None and sample.capacity_tokens is not None:
            self._capacity_tokens = sample.capacity_tokens
            self._capacity_derived = sample.capacity_derived
        if self._capacity_gb is None and sample.capacity_gb is not None:
            self._capacity_gb = sample.capacity_gb
        self._series_count = max(self._series_count, sample.series_count)
        # Every cumulative counter gets the same treatment: bracket the round and diff. The prefix-cache ones need it as
        # much as the pressure ones, because under warm reuse the engine outlives the round and its absolute totals
        # carry the previous round's cache warming. Bracketed per phase, not per round. An engine keeps retracting
        # through warmup and the accuracy eval, and a round-wide difference silently folds both into the one number that
        # is supposed to describe the measured window alone. The gauge rows carry their phase and can be re-sliced
        # later; counters cannot, so the split has to happen here.
        self._record_counters(sample)
        # Every row carries the gauges *and* the cumulative counters, in raw per-series form. Recording counters only at
        # phase boundaries would leave the interior of a phase blind: an engine that retracted in one burst and one that
        # retracted steadily produce the same phase total, and an engine restart mid-phase is invisible without the
        # series. With the raw maps present a consumer can difference any two adjacent rows and does not have to trust
        # this module's aggregation to do it.
        row = {
            "phase": self._phase,
            "ts": round(sample.ts, 3),
            "mono": round(sample.mono, 3),
            **(timing or {"scrape_sec": 0.0}),
        }
        if workload is not None:
            row["workload"] = workload
        self._rows.append(
            {
                **row,
                "active_pool_usage": sample.active_pool_usage,
                "physical_pool_usage": sample.physical_pool_usage,
                "used_tokens": sample.used_tokens,
                "evictable_tokens": sample.evictable_tokens,
                "available_tokens": sample.available_tokens,
                "capacity_tokens": sample.capacity_tokens,
                # Same aggregation rule the summary uses. Rows and summary disagreeing inside one artifact is worse than
                # either rule being wrong, because nothing on the page says which is which.
                "retract_total": aggregate_series(sample.retract_total),
                "preempt_total": aggregate_series(sample.preempt_total),
                "counters_by_series": {
                    "retract": sample.retract_total,
                    "preempt": sample.preempt_total,
                    "prefix_cache_queries": sample.prefix_cache_queries,
                    "prefix_cache_hits": sample.prefix_cache_hits,
                    "cached_tokens_total": sample.cached_tokens_total,
                },
            }
        )

    def _record_counters(self, sample: KvSample) -> None:
        """Bracket every cumulative counter under the phase in force."""
        for name, series in (
            ("retract", sample.retract_total),
            ("preempt", sample.preempt_total),
            ("prefix_cache_queries", sample.prefix_cache_queries),
            ("prefix_cache_hits", sample.prefix_cache_hits),
            ("cached_tokens_total", sample.cached_tokens_total),
        ):
            if not series:
                continue
            snapshot = {g: dict(s) for g, s in series.items()}
            self._first_counters.setdefault(self._phase, {}).setdefault(name, snapshot)
            self._last_counters.setdefault(self._phase, {})[name] = snapshot

    def _counter_deltas(self, name: str) -> tuple[float | None, dict[str, float | None]]:
        """Increment of one counter, attributed to the phase that earned it."""
        by_phase: dict[str, float | None] = {}
        for phase in PHASES:
            last = self._last_counters.get(phase, {}).get(name)
            if not last:
                continue
            by_phase[phase] = counter_delta(self._first_counters.get(phase, {}).get(name, {}), last)
        return by_phase.get(_COMPARABLE_PHASE), by_phase

    def _prefix_cache_window(self) -> dict[str, Any]:
        """Prefix-cache increments attributable to this round.

        Each counter is diffed per label series and combined by the shared grouping rule, so two data-parallel engines
        that served 200 lookups each report 400 rather than 200. Reporting the round's delta rather than the running
        total is what makes a hit rate belong to the round that earned it.
        """
        window: dict[str, Any] = {}
        for name in ("prefix_cache_queries", "prefix_cache_hits", "cached_tokens_total"):
            measured, by_phase = self._counter_deltas(name)
            if not by_phase:
                continue
            window[f"{name}_delta"] = measured
            window[f"{name}_delta_by_phase"] = by_phase
        return window

    def _adopt_aiperf_samples(self) -> str | None:
        """Replace the live rows with aiperf's own scrapes when it collected any.

        aiperf drives the AgentX workload, so on those rounds it is the better source in every way that matters: it
        scrapes the same endpoint on a tighter cadence, takes its own reading at each phase boundary, and stamps each
        record with the phase from the process that owns the transition -- so there is no boundary lag to correct and
        nothing to re-attribute afterwards.

        The live rows are discarded rather than merged. Two collectors on one endpoint produce interleaved readings of
        the same counters at slightly different instants, and a window bracketed across both would attribute an
        increment to whichever happened to be sampled either side of it.
        """
        if self._workspace is None:
            return None
        try:
            export = find_server_metrics_export(self._workspace)
            if export is None:
                return None
            records = read_aiperf_server_metrics(export)
            if not records:
                return None
            # Rebuilt from scratch so no live reading survives into the counter windows below.
            self._rows = []
            self._first_counters = {}
            self._last_counters = {}
            previous_phase = self._phase
            for sample, timing, phase in records:
                self._phase = phase or "measured"
                self._absorb(sample, timing=timing)
            self._phase = previous_phase
            # Adjacent phases share the reading at their boundary here too. aiperf takes its own scrape at each
            # transition, but tags it with the phase that is ending, so without this the next phase's window would open
            # at its first periodic sample and the increment in between would be credited to neither.
            self._rebuild_counter_windows()
            return str(export)
        except Exception:  # noqa: BLE001 - a better source that cannot be read is not a reason to fail a round
            log.debug("kv_metrics: aiperf server metrics unavailable", exc_info=True)
            return None

    def _authoritative_bounds(self) -> dict[str, tuple[float, float | None]] | None:
        """Phase windows in Unix seconds, from aiperf's own stamps, or ``None`` when unusable.

        ``start_ns`` is rejected rather than trusted blindly when it does not land near this process's wall clock. The
        API reports nanoseconds but does not say from which epoch, and a monotonic reading would silently place every
        boundary in 1970 and re-attribute the whole round to one phase.
        """
        if not self._phase_timeline:
            return None
        now = time.time()
        bounds: dict[str, tuple[float, float | None]] = {}
        for name, stats in self._phase_timeline.items():
            phase = _AIPERF_PHASE_NAMES.get(name)
            start_ns = _number(stats.get("start_ns"))
            if phase is None or start_ns is None or start_ns <= 0:
                continue
            start = start_ns / 1e9
            if abs(start - now) > _PROGRESS_CLOCK_SANITY_SEC:
                return None
            end_ns = _number(stats.get("requests_end_ns"))
            end = end_ns / 1e9 if end_ns and end_ns > 0 else None
            if end is not None and end < start:
                return None
            bounds[phase] = (start, end)
        return bounds or None

    def _reattribute_phases(self) -> dict[str, Any]:
        """Re-label rows from the authoritative timeline, reporting what was used.

        Re-labelling after the fact is what makes the boundary exact. A row is stamped with whatever phase the watchdog
        believed at the time, and the watchdog only learns of a transition once aiperf has written a log line and the
        next poll has read it -- so the rows straddling a boundary are systematically attributed to the phase that just
        ended. The stamps say when the phase actually began, so the correction is applied to the stored rows.

        ``eval`` is never overwritten. It is the accuracy run, which aiperf has no part in and no opinion about.
        """
        bounds = self._authoritative_bounds()
        if bounds is None:
            return {"phase_source": "log_markers", "phase_timeline": self._phase_timeline or None}
        # Latest-starting window wins, so a row inside profiling is not also claimed by warmup, whose end the API leaves
        # null until its requests drain.
        ordered = sorted(bounds.items(), key=lambda item: item[1][0], reverse=True)
        moved = 0
        for row in self._rows:
            if row.get("phase") == "eval":
                continue
            ts = _number(row.get("ts"))
            if ts is None:
                continue
            for phase, (start, end) in ordered:
                if ts >= start and (end is None or ts <= end):
                    if row["phase"] != phase:
                        row["phase"] = phase
                        moved += 1
                    break
        self._rebuild_counter_windows()
        return {
            "phase_source": "aiperf_progress_api",
            "phase_timeline": self._phase_timeline,
            "phase_bounds_unix": {p: {"start": s, "requests_end": e} for p, (s, e) in bounds.items()},
            "rows_reattributed": moved,
        }

    def _rebuild_counter_windows(self) -> None:
        """Re-bracket every counter from the re-attributed rows.

        Re-labelling the rows alone would leave the phase totals wrong, which is the whole point of the exercise: the
        windows were closed when the watchdog *noticed* a transition, so the increments earned in the lag were already
        booked to the phase that had ended. Every row carries its raw per-series counters precisely so the windows can
        be rebuilt from them once the true boundary is known.

        Adjacent phases share the snapshot at their boundary -- the reading that closes one opens the next -- so this
        keeps the gap-free property the boundary scrapes give, rather than trading it for accuracy.
        """
        ordered = [r for r in self._rows if isinstance(r.get("counters_by_series"), dict)]
        if not ordered:
            return
        first: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        last: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

        def _apply(phase: str, row: dict[str, Any], *, opening: bool) -> None:
            """Record one row's counters as a window edge for ``phase``."""
            for name, series in (row.get("counters_by_series") or {}).items():
                if not series:
                    continue
                snapshot = {g: dict(s) for g, s in series.items()}
                if opening:
                    first.setdefault(phase, {}).setdefault(name, snapshot)
                else:
                    last.setdefault(phase, {})[name] = snapshot

        _apply(ordered[0]["phase"], ordered[0], opening=True)
        for previous, row in zip(ordered, ordered[1:]):
            if row["phase"] != previous["phase"]:
                _apply(previous["phase"], row, opening=False)
                _apply(row["phase"], row, opening=True)
            _apply(row["phase"], row, opening=False)
        self._first_counters = first
        self._last_counters = last

    def _build_workload_timeline(self) -> dict[str, Any] | None:
        """Emit the AgentX event stream and fold the in-flight work into the rows.

        Read from aiperf's own per-request export rather than asked of it: ``profile_export.jsonl`` is written at the
        default export level, so every AgentX round already has one. A round that has none -- any synthetic benchmark,
        or an AgentX round killed before aiperf flushed -- simply reports ``None``.
        """
        if self._workspace is None:
            return None
        try:
            from ._agentx_timeline import (
                TIMELINE_ARTIFACT_NAME,
                build_events,
                correlate_rows,
                find_profile_export,
                parse_profile_export,
                write_timeline,
            )

            export = find_profile_export(self._workspace)
            if export is None:
                return None
            records = parse_profile_export(export)
            if not records:
                return None
            events, summary = build_events(records, self._authoritative_bounds())
            # Correlation runs over every record, so the counts on a row stay exact even when the event stream above
            # had to be sampled.
            summary["rows_correlated"] = correlate_rows(self._rows, records)
            path = Path(self._workspace) / TIMELINE_ARTIFACT_NAME
            summary["path"] = path.name if write_timeline(path, events) else None
            summary["source"] = str(export)
            return summary
        except Exception:  # noqa: BLE001 - a timeline must never fail a round
            log.debug("kv_metrics: workload timeline unavailable", exc_info=True)
            return None

    def rows(self) -> list[dict[str, Any]]:
        """Collected rows, stride-downsampled to the row cap."""
        if len(self._rows) <= _MAX_STORED_ROWS:
            return list(self._rows)
        # Round the stride up. Integer division gives 1 for anything under twice the cap, so 5001 rows would have
        # downsampled to 5001.
        stride = -(-len(self._rows) // _MAX_STORED_ROWS)
        return self._rows[::stride]

    def summary(self, *, aborted: bool = False) -> dict[str, Any]:
        """Build the artifact payload."""
        # Before attribution, because adopting aiperf's rows replaces the very rows attribution would re-label -- and
        # makes re-labelling unnecessary, since those rows already carry the phase aiperf stamped at collection time.
        adopted = self._adopt_aiperf_samples()
        attribution = (
            {"phase_source": "aiperf_server_metrics", "phase_timeline": self._phase_timeline or None}
            if adopted
            else self._reattribute_phases()
        )
        # After attribution, so the phase events in the timeline are the same boundaries the rows were labelled by.
        workload = self._build_workload_timeline()
        retract_measured, retract_by_phase = self._counter_deltas("retract")
        preempt_measured, preempt_by_phase = self._counter_deltas("preempt")
        return {
            **attribution,
            "workload_timeline": workload,
            "schema_version": 1,
            "source": "metrics",
            # Which collector produced the rows. The two differ in cadence and in how the phase was decided, so a
            # consumer comparing rounds has to be able to tell them apart.
            "sample_source": "aiperf_server_metrics" if adopted else "watchdog_scrape",
            "aiperf_server_metrics_path": adopted,
            "url": self._poller.url,
            "available": self._poller.available,
            "aborted": bool(aborted),
            "scope": self._scope,
            "capacity_tokens": self._capacity_tokens,
            "capacity_derived": self._capacity_derived,
            "capacity_gb": self._capacity_gb,
            "series_count": self._series_count,
            "prefix_cache": self._prefix_cache_window(),
            "phase_marks": self._phase_marks,
            # The headline figure is the measured phase alone, because that is the only phase the plan lets into a
            # comparison. The breakdown is kept beside it so a round that retracted hard during warmup is still visible
            # rather than rounded away.
            "retract_delta": retract_measured,
            "retract_delta_by_phase": retract_by_phase,
            "preempt_delta": preempt_measured,
            "preempt_delta_by_phase": preempt_by_phase,
            "sample_count": len(self._rows),
            "samples": self.rows(),
        }

    def close(self, *, aborted: bool = False) -> dict[str, Any]:
        """Finish collection and write the artifact when a path was given.

        Idempotent: the loop's ``finally`` may run after a caller has already closed explicitly.
        """
        # Final boundary reading, for the same reason the phase transitions take one: without it the last phase ends at
        # whenever its last periodic sample happened to land, and everything after that is lost.
        if not self._closed:
            self._scrape(time.monotonic())
        payload = self.summary(aborted=aborted)
        if self._closed or not self._output_path:
            self._closed = True
            return payload
        self._closed = True
        try:
            from pathlib import Path

            from hyperloom.common.io import atomic_write_json

            atomic_write_json(Path(self._output_path), payload)
        except Exception:  # noqa: BLE001 - an unwritten artifact must not fail a round
            # Warning, not debug. Not failing the round is the requirement; being quiet about it is not. A round that
            # collected samples and then dropped them on the floor looks identical afterwards to one that never
            # collected any, and nobody goes looking for a file they were never told was missing.
            log.warning(
                "kv_metrics: could not write %s (%d samples collected this round are lost)",
                self._output_path,
                len(self._rows),
                exc_info=True,
            )
        return payload
