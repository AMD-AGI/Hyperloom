# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KV-cache observability sampled from the engine's ``/metrics`` endpoint.

The engine is the only thing that knows how full its KV pool is, and until now
nothing read it: a run could spend its whole budget retracting requests and the
breakdown would show only "throughput low". This module is the reader.

``/metrics`` is the primary source rather than ``server.log`` because it wins on
every axis that matters here -- full float precision instead of the log's two
decimals, stable metric names instead of names that branch on the model's pool
type, explicit units, and ``kv_evictable_tokens``, which the log cannot express
at all and which is the only way to separate the two occupancy readings this
module reports (see :class:`KvSample`).

Two rules run through everything below:

* **Never raise.** This samples a benchmark that is being measured; a scrape
  failure must cost the run nothing. Every entry point returns a value.
* **Never coerce a missing reading to zero.** A pool nobody sampled and a pool
  that is genuinely empty are different findings, and collapsing them is how an
  optimizer ends up steering on a number that was never measured. Absent means
  ``None``, all the way out to the breakdown.
"""

from __future__ import annotations

import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_METRICS_PORT",
    "KV_ARTIFACT_NAME",
    "PHASES",
    "KvMetricsPoller",
    "KvMetricsRecorder",
    "KvSample",
    "canonical_label_key",
    "counter_delta",
    "parse_prometheus_text",
    "resolve_metrics_port",
    "sample_from_families",
]


#: Port the persistent server binds when ``benchmark.envs.PORT`` is unset.
#: Mirrors ``_server_lifecycle.REUSE_PORT_DEFAULT``; duplicated rather than
#: imported because that module imports ``_subprocess_kill``, which is where
#: this one gets wired in.
DEFAULT_METRICS_PORT = 8888

#: Scrape timeout. Deliberately well under the 0.5s poll interval of the loop
#: this runs inside: a slow endpoint must not stretch the interval that the
#: stall and soft-deadline gates are measured on. The engine is on loopback, so
#: anything approaching this budget is already pathological.
_SCRAPE_TIMEOUT_SEC = 0.4

#: Opener that ignores the ambient proxy configuration. ``urlopen`` honours
#: ``http_proxy`` by default, which on a corporate host routes a loopback scrape
#: through an external proxy: wrong by construction, and slow enough that the
#: blocking call visibly delays the watchdog loop it runs inside.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

#: Consecutive failures after which the poller stops trying. A server that
#: never exposes ``/metrics`` (SGLang without ``--enable-metrics``) would
#: otherwise pay a connection refusal twice a second for the whole round.
_MAX_CONSECUTIVE_FAILURES = 3


# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------

# Occupancy, SGLang. ``token_usage`` is the active reading -- its numerator
# excludes blocks the prefix cache is holding but would hand back under
# pressure. The per-subpool gauges are always exposed and read 0 on a model
# that has no such subpool, so taking the max over all of them is both the
# hybrid-correct answer (SGLang's own scheduler judges pressure that way) and a
# no-op on an ordinary KV pool.
_SGL_USAGE = ("sglang:token_usage", "sglang:full_token_usage", "sglang:swa_token_usage")
_SGL_USED = "sglang:kv_used_tokens"
_SGL_AVAILABLE = "sglang:kv_available_tokens"
# Only the main KV pool's evictable count. The SWA and Mamba pools have their
# own capacities, and ``kv_used_tokens`` / ``kv_available_tokens`` describe the
# main pool alone -- folding the other pools' evictable tokens into a ratio
# built from main-pool numerators mixes two different denominators.
_SGL_EVICTABLE = "sglang:kv_evictable_tokens"
# Pool capacity as the engine reports it. Preferred over deriving it, because
# used + available + evictable can fall short of the true size: the gap is
# tokens held in reserve (protected / session-held), and deriving would both
# understate capacity and overstate the physical occupancy computed from it.
_SGL_CAPACITY_TOKENS = "sglang:max_total_num_tokens"
_SGL_CAPACITY_GB = "sglang:kv_cache_memory_usage_gb"
# Cumulative retract counter. NOT ``sglang:num_retracted_reqs``, which is the
# most recent batch's instantaneous gauge and carries a ``pid`` label; the two
# names differ by one word and reading the wrong one turns 48 retracts into 1.
_SGL_RETRACT_TOTAL = "sglang:num_retracted_requests_total"
_SGL_CACHED_TOKENS = "sglang:cached_tokens_total"

# Occupancy, vLLM. ``gpu_cache_usage_perc`` is the pre-rename spelling, kept so
# an older image still reports.
_VLLM_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
# The only workable preemption count: vLLM's log line for it is dead code
# (``log()`` resets the counter before reading it), so this has no log fallback.
_VLLM_PREEMPT_TOTAL = "vllm:num_preemptions_total"
_VLLM_PREFIX_QUERIES = "vllm:prefix_cache_queries"
_VLLM_PREFIX_HITS = "vllm:prefix_cache_hits"

# Deliberately not read: ``sglang:cache_hit_rate``. Observed reading 0.0 on a
# server whose ``cached_tokens_total`` had already reached 4.6M, so it is not a
# cumulative rate -- whether it is instantaneous or windowed is unresolved, and
# a field nobody can interpret is worse than an absent one. The raw families
# stay available to whoever settles it.


ParsedFamilies = dict[str, list[tuple[dict[str, str], float]]]


def canonical_label_key(labels: dict[str, str]) -> str:
    """Render a label set as a stable string usable as a dict key.

    Counters are diffed per series, not in aggregate: an engine restart resets
    a series to zero, and a naive total would read that as a negative delta or,
    worse, silently absorb it. Sorting makes the key independent of the order
    the exporter happened to emit.

    Args:
        labels (dict[str, str]): Label name to value.

    Returns:
        str: ``k="v",k2="v2"`` with names sorted; ``""`` for no labels.
    """
    return ",".join(f'{k}="{labels[k]}"' for k in sorted(labels))


def _split_labels(raw: str) -> dict[str, str]:
    """Parse the inside of a Prometheus label brace into a mapping.

    Values may contain escaped quotes and commas, so the scan is character-wise
    rather than a naive ``split(",")``.

    Args:
        raw (str): The text between ``{`` and ``}``, exclusive.

    Returns:
        dict[str, str]: Label name to unescaped value. Malformed fragments are
        skipped rather than raising -- a label this module does not understand
        must not cost it the sample's value.
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


def parse_prometheus_text(text: str) -> ParsedFamilies:
    """Parse a Prometheus text exposition into metric families.

    Handles the parts of the format an engine actually emits: comments, labels,
    an optional trailing timestamp, and the ``NaN`` / ``+Inf`` / ``-Inf``
    literals. Non-finite values are dropped -- they carry no occupancy meaning
    and would poison any max or mean taken over them.

    Args:
        text (str): The response body of a ``/metrics`` GET.

    Returns:
        ParsedFamilies: Metric name to a list of ``(labels, value)``. A metric
        that appears with several label sets keeps one entry per set.
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


def _scalar(families: ParsedFamilies, name: str) -> float | None:
    """Read a gauge, taking the largest reading when several series exist.

    **Never sums.** A sharded engine exposes one series per rank
    (``tp_rank`` / ``pp_rank`` / ``dp_rank``), and every one of them describes
    the same pool from its own side: the ranks hold different head slices of
    the same tokens, so their capacities and occupancies are views, not parts.
    Adding them turns an 85%-full pool on TP=8 into an occupancy of 6.8 and
    multiplies capacity eightfold -- the same mistake the log path guards
    against by de-duplicating capacity lines per rank.

    Max rather than first-rank so an imbalanced deployment reports its most
    pressured rank, which is the one that will retract.

    Args:
        families (ParsedFamilies): Parsed exposition.
        name (str): Metric name.

    Returns:
        float | None: The value, or ``None`` when the metric is absent. Absent
        is never 0.0: several of these gauges legitimately read zero.
    """
    series = families.get(name)
    if not series:
        return None
    return max(value for _, value in series)


def _max_scalar(families: ParsedFamilies, names: tuple[str, ...]) -> float | None:
    """Largest reading across several gauges, ignoring the ones not present.

    Args:
        families (ParsedFamilies): Parsed exposition.
        names (tuple[str, ...]): Metric names to consider.

    Returns:
        float | None: The max, or ``None`` when none of the names appeared.
    """
    values = [v for name in names if (v := _scalar(families, name)) is not None]
    return max(values) if values else None


def _series_count(families: ParsedFamilies, names: tuple[str, ...]) -> int:
    """How many label series the largest of these metrics carried.

    Surfaced in the artifact so a consumer can tell a single-rank reading from
    a collapsed multi-rank one, and judge for itself whether the aggregation
    rule above was the right one for its deployment.

    Args:
        families (ParsedFamilies): Parsed exposition.
        names (tuple[str, ...]): Metric names to inspect.

    Returns:
        int: The largest series count among the named metrics; 0 when absent.
    """
    return max((len(families.get(name) or ()) for name in names), default=0)


def _series(families: ParsedFamilies, name: str) -> dict[str, float]:
    """Read a counter as a per-label-set mapping.

    Args:
        families (ParsedFamilies): Parsed exposition.
        name (str): Metric name.

    Returns:
        dict[str, float]: Canonical label key to value; empty when absent.
    """
    return {canonical_label_key(labels): value for labels, value in families.get(name, [])}


@dataclass(frozen=True)
class KvSample:
    """One ``/metrics`` scrape, normalised across engines.

    Every metric is ``None`` when the engine did not expose it. Read them with
    ``is None``, never with truthiness: an idle pool reports a true 0.0, and an
    engine that has not been asked for a token yet reports 0.0 for minutes.

    Attributes:
        ts (float): Wall clock, for lining up with externally collected data.
        mono (float): Monotonic clock, for intervals within the run.
        engine (str): ``"sglang"``, ``"vllm"``, or ``""`` when unrecognised.
        active_pool_usage (float | None): Fraction of the pool held by running
            requests. This is the pressure reading.
        physical_pool_usage (float | None): Fraction physically occupied,
            including blocks the prefix cache holds but would release. A pool
            at 100% physical can be under no pressure at all -- that is prefix
            cache doing its job -- which is why this is reported separately and
            never as "the" occupancy. SGLang only; vLLM cannot express it.
        used_tokens (float | None): Tokens held by running requests.
        evictable_tokens (float | None): Tokens the prefix cache holds and would
            hand back under pressure.
        available_tokens (float | None): Tokens free outright.
        capacity_tokens (float | None): Pool size in tokens, read from the
            engine when it reports one.
        capacity_derived (bool): True when ``capacity_tokens`` had to be summed
            from used + available + evictable because no capacity gauge was
            exposed. That sum can fall short of the real pool -- tokens held in
            reserve belong to none of the three -- so a derived capacity makes
            ``physical_pool_usage`` an upper bound rather than a measurement.
        series_count (int): Label series the occupancy gauge carried. Greater
            than one means a sharded engine reported per-rank views that were
            collapsed to their maximum; see :func:`_scalar`.
        capacity_gb (float | None): Pool size in GiB as the engine reports it.
            The engine's "GB" is 1024-based; do not rescale it.
        retract_total (dict[str, float]): SGLang cumulative retracts, per label
            series. Kept per series so an engine restart shows up as a series
            resetting rather than as a total going backwards.
        preempt_total (dict[str, float]): vLLM cumulative preemptions, likewise.
        prefix_cache_queries (float | None): vLLM cumulative prefix lookups.
        prefix_cache_hits (float | None): vLLM cumulative prefix hits.
        cached_tokens_total (float | None): SGLang cumulative prefix-cached
            tokens. Absent entirely when prefix caching is off -- the metric is
            not emitted at all, rather than emitted as zero.
    """

    ts: float
    mono: float
    engine: str = ""
    active_pool_usage: float | None = None
    physical_pool_usage: float | None = None
    used_tokens: float | None = None
    evictable_tokens: float | None = None
    available_tokens: float | None = None
    capacity_tokens: float | None = None
    capacity_derived: bool = False
    series_count: int = 0
    capacity_gb: float | None = None
    retract_total: dict[str, float] = field(default_factory=dict)
    preempt_total: dict[str, float] = field(default_factory=dict)
    prefix_cache_queries: float | None = None
    prefix_cache_hits: float | None = None
    cached_tokens_total: float | None = None

    def has_readings(self) -> bool:
        """Whether this scrape carried any KV signal at all.

        A reachable endpoint that exposes no KV metric -- an engine started
        without them, or one whose exposition this module does not recognise --
        must not be recorded as a row of nothing.

        Returns:
            bool: True when at least one occupancy or pressure field is set.
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
    """Identify the engine from its metric-name prefix.

    Args:
        families (ParsedFamilies): Parsed exposition.

    Returns:
        str: ``"sglang"``, ``"vllm"``, or ``""`` when neither prefix appears.
    """
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
    """Build a normalised sample from a parsed exposition.

    Args:
        families (ParsedFamilies): Output of :func:`parse_prometheus_text`.
        ts (float | None): Wall clock to stamp; defaults to now.
        mono (float | None): Monotonic clock to stamp; defaults to now.

    Returns:
        KvSample: The normalised reading. Fields the engine did not expose stay
        ``None``.
    """
    engine = _detect_engine(families)
    used = _scalar(families, _SGL_USED)
    available = _scalar(families, _SGL_AVAILABLE)
    evictable = _scalar(families, _SGL_EVICTABLE)

    # Engine-reported capacity wins; the sum is a fallback for builds that do
    # not expose it, and is marked as derived so a consumer knows it may run
    # short of the true pool size.
    capacity = _scalar(families, _SGL_CAPACITY_TOKENS)
    capacity_derived = False
    if capacity is None and used is not None and available is not None:
        capacity = used + available + (evictable or 0.0)
        capacity_derived = True

    physical: float | None = None
    if capacity and capacity > 0 and used is not None:
        physical = (used + (evictable or 0.0)) / capacity

    active = _max_scalar(families, _SGL_USAGE)
    if active is None:
        active = _max_scalar(families, _VLLM_USAGE)

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
        series_count=_series_count(families, _SGL_USAGE + _VLLM_USAGE),
        capacity_gb=_scalar(families, _SGL_CAPACITY_GB),
        retract_total=_series(families, _SGL_RETRACT_TOTAL),
        preempt_total=_series(families, _VLLM_PREEMPT_TOTAL),
        prefix_cache_queries=_scalar(families, _VLLM_PREFIX_QUERIES),
        prefix_cache_hits=_scalar(families, _VLLM_PREFIX_HITS),
        cached_tokens_total=_scalar(families, _SGL_CACHED_TOKENS),
    )


def resolve_metrics_port(config_envs: dict[str, Any] | None = None) -> int:
    """Resolve the port the engine serves ``/metrics`` on.

    The server binds whatever ``benchmark.envs.PORT`` the materialized YAML
    pins -- an ephemeral port assigned per session, not a constant -- so the
    caller's config is the authoritative source and the ambient env is only a
    fallback for paths that never materialize a YAML.

    Args:
        config_envs (dict[str, Any] | None): The materialized config's
            ``benchmark.envs`` mapping, when the caller has it.

    Returns:
        int: The resolved port, falling back to :data:`DEFAULT_METRICS_PORT`.
    """
    for source in (config_envs or {}, os.environ):
        raw = source.get("PORT")
        if raw in (None, ""):
            continue
        try:
            port = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if port > 0:
            return port
    return DEFAULT_METRICS_PORT


class KvMetricsPoller:
    """Scrapes one engine's ``/metrics``, degrading quietly when it cannot.

    Built to sit inside the executors' existing 0.5s watchdog loop rather than
    in a process of its own: a sampler racing the engine for CPU would show up
    in the very latency numbers the round exists to measure.

    Availability is tri-state and sticky. Until the first successful scrape the
    poller is ``unknown``; a scrape that lands makes it available for good; a
    run of consecutive failures parks it as unavailable and it stops issuing
    requests, because the common cause is an engine started without
    ``--enable-metrics``, and that will not fix itself mid-round.
    """

    def __init__(
        self,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        config_envs: dict[str, Any] | None = None,
        timeout_sec: float = _SCRAPE_TIMEOUT_SEC,
    ) -> None:
        """Bind the poller to an endpoint without contacting it.

        Args:
            port (int | None): Explicit port; resolved from config/env when
                omitted.
            host (str): Host to scrape. The engine is process-local on every
                path that reaches here.
            config_envs (dict[str, Any] | None): Materialized ``benchmark.envs``
                used for port resolution.
            timeout_sec (float): Per-scrape timeout.
        """
        self.port = int(port) if port else resolve_metrics_port(config_envs)
        self.url = f"http://{host}:{self.port}/metrics"
        self.timeout_sec = float(timeout_sec)
        self._failures = 0
        self._succeeded = False
        self._gave_up = False
        self._warned = False

    @property
    def available(self) -> bool | None:
        """Tri-state reachability of the endpoint.

        Returns:
            bool | None: ``True`` once a scrape has landed, ``False`` once the
            poller has given up, ``None`` while still unknown. Callers must test
            with ``is True`` / ``is False``; treating ``None`` as ``False`` would
            record "no metrics" for a round that simply had not booted yet.
        """
        if self._succeeded:
            return True
        if self._gave_up:
            return False
        return None

    def fetch(self) -> str | None:
        """Scrape once.

        Returns:
            str | None: The response body, or ``None`` when the endpoint could
            not be read (including when the poller has already given up). Never
            raises: this runs alongside a measured benchmark.
        """
        if self._gave_up:
            return None
        try:
            with _OPENER.open(self.url, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8", "ignore")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._failures += 1
            if self._failures >= _MAX_CONSECUTIVE_FAILURES and not self._succeeded:
                self._gave_up = True
                if not self._warned:
                    self._warned = True
                    log.info(
                        "kv_metrics: %s unreachable after %d attempts (%s); KV metrics "
                        "recorded as unavailable for this round. SGLang needs "
                        "--enable-metrics; vLLM exposes it by default.",
                        self.url,
                        self._failures,
                        exc,
                    )
            return None
        self._failures = 0
        self._succeeded = True
        return body

    def sample(self) -> KvSample | None:
        """Scrape and normalise in one step.

        Returns:
            KvSample | None: The reading, or ``None`` when the endpoint was
            unreadable or exposed no KV metric at all.
        """
        body = self.fetch()
        if body is None:
            return None
        sample = sample_from_families(parse_prometheus_text(body))
        return sample if sample.has_readings() else None


#: Artifact the recorder writes, alongside the round's ``server.log``.
KV_ARTIFACT_NAME = "kv_metrics.json"

#: Collection phases. An engine process outlives the window that is actually
#: being measured by a wide margin, and mixing them makes every statistic
#: meaningless: ``boot`` has no traffic at all, ``warmup`` runs a deliberately
#: cold cache, and ``eval`` drives accuracy traffic whose shape has nothing to
#: do with the throughput benchmark. Only ``measured`` may enter a comparison.
PHASES = ("boot", "warmup", "measured", "eval")

#: Row cap before stride downsampling, mirroring ``_MN_GPU_SAMPLE_CAP``. A
#: three-hour round at the scrape interval below lands well over this.
_MAX_STORED_ROWS = 5000

#: Floor between scrapes. The loop this runs in can iterate faster than its
#: nominal 0.5s when a deadline bounds the slice, so the interval is enforced
#: on the monotonic clock rather than by counting passes.
_MIN_SCRAPE_INTERVAL_SEC = 0.5


def counter_delta(first: dict[str, float], last: dict[str, float]) -> float | None:
    """Increment of a labelled counter between two observations.

    Diffed per label series rather than on a flat total, because an engine
    restart zeroes its series: a flat subtraction would go negative, and
    clamping that to zero would quietly discard the whole window. When a series
    ends below where it started it is treated as having restarted, and only the
    post-restart count is credited -- the increments before the restart are
    genuinely unrecoverable, and inventing them would be worse than losing them.

    Args:
        first (dict[str, float]): Series readings at window open.
        last (dict[str, float]): Series readings at window close.

    Series are combined by **max, not sum**, for the same reason gauges are: a
    sharded engine reports one series per rank and the ranks retract in
    lockstep, so eight series carrying 48 describe 48 events, not 384. This
    under-counts a deployment whose series really are independent engines;
    that trade is deliberate, because an inflated pressure count would send the
    optimizer chasing a bottleneck that is not there, while an under-count only
    understates one that is. ``series_count`` on the sample says which case a
    reading came from.

    Args:
        first (dict[str, float]): Series readings at window open.
        last (dict[str, float]): Series readings at window close.

    Returns:
        float | None: The increment, or ``None`` when the counter was never
        observed at all -- which is not the same as an increment of zero.
    """
    if not first and not last:
        return None
    deltas: list[float] = []
    for key, end in last.items():
        start = first.get(key)
        deltas.append(end if start is None or end < start else end - start)
    return max(deltas) if deltas else 0.0


class KvMetricsRecorder:
    """Collects phase-tagged KV samples for one benchmark round.

    Owns everything stateful about collection so the watchdog loop it hangs off
    keeps a single call per pass and one ``finally``. Like the poller, nothing
    here raises: a recorder that fails must cost the round nothing.

    The phase machine is driven by markers the loop already detects, plus
    aiperf's own phase lines under AgentX. It is never inferred from elapsed
    time, because the boundaries move by tens of minutes between rounds -- an
    AgentX warmup alone was measured at nearly 18 of them.
    """

    def __init__(
        self,
        *,
        poller: KvMetricsPoller,
        output_path: str | None = None,
        scope: dict[str, Any] | None = None,
        min_interval_sec: float = _MIN_SCRAPE_INTERVAL_SEC,
    ) -> None:
        """Prepare a recorder without contacting anything.

        Args:
            poller (KvMetricsPoller): The endpoint reader.
            output_path (str | None): Where :meth:`close` writes its artifact.
                ``None`` collects in memory only, which is what the tests and
                any caller without a workspace want.
            scope (dict[str, Any] | None): Identity carried into the artifact so
                a consumer never has to join against another file to learn which
                variant and round produced it.
            min_interval_sec (float): Floor between scrapes.
        """
        self._poller = poller
        self._output_path = output_path
        self._scope = dict(scope or {})
        self._min_interval = float(min_interval_sec)
        self._phase = "boot"
        self._rows: list[dict[str, Any]] = []
        self._last_scrape_mono: float | None = None
        self._phase_marks: list[dict[str, Any]] = []
        self._first_counters: dict[str, dict[str, float]] = {}
        self._last_counters: dict[str, dict[str, float]] = {}
        self._capacity_tokens: float | None = None
        self._capacity_derived = False
        self._capacity_gb: float | None = None
        self._series_count = 0
        self._prefix_cache: dict[str, float] = {}
        self._closed = False

    @property
    def phase(self) -> str:
        """Current collection phase."""
        return self._phase

    def note_phase(self, phase: str, mono: float) -> None:
        """Record a phase transition.

        Args:
            phase (str): One of :data:`PHASES`. Anything else is ignored rather
                than raising -- an unrecognised marker must not end collection.
            mono (float): Monotonic instant of the transition.
        """
        if phase not in PHASES or phase == self._phase:
            return
        self._phase = phase
        self._phase_marks.append({"phase": phase, "mono": round(float(mono), 3), "ts": time.time()})

    def tick(self, mono: float) -> None:
        """Scrape if the interval has elapsed, tagging the sample with the phase.

        Args:
            mono (float): The loop's current monotonic instant.
        """
        if self._closed:
            return
        if self._last_scrape_mono is not None and (mono - self._last_scrape_mono) < self._min_interval:
            return
        self._last_scrape_mono = mono
        try:
            sample = self._poller.sample()
        except Exception:  # noqa: BLE001 - collection must never fail a round
            log.debug("kv_metrics: sample failed", exc_info=True)
            return
        if sample is None:
            return
        self._absorb(sample)

    def _absorb(self, sample: KvSample) -> None:
        """Fold one sample into the row buffer and the counter windows.

        Args:
            sample (KvSample): The reading to record.
        """
        # Pool capacity is only observable while the engine is up; latch the
        # first non-null reading so the artifact still carries it after a round
        # that ended with the server gone.
        if self._capacity_tokens is None and sample.capacity_tokens is not None:
            self._capacity_tokens = sample.capacity_tokens
            self._capacity_derived = sample.capacity_derived
        if self._capacity_gb is None and sample.capacity_gb is not None:
            self._capacity_gb = sample.capacity_gb
        self._series_count = max(self._series_count, sample.series_count)
        # Prefix-cache counters are cumulative, so the newest reading is the
        # whole story; keeping the latest non-null avoids a column of repeats
        # in every row.
        for attr in ("prefix_cache_queries", "prefix_cache_hits", "cached_tokens_total"):
            value = getattr(sample, attr)
            if value is not None:
                self._prefix_cache[attr] = value
        for name, series in (("retract", sample.retract_total), ("preempt", sample.preempt_total)):
            if not series:
                continue
            self._first_counters.setdefault(name, dict(series))
            self._last_counters[name] = dict(series)
        self._rows.append(
            {
                "phase": self._phase,
                "ts": round(sample.ts, 3),
                "mono": round(sample.mono, 3),
                "active_pool_usage": sample.active_pool_usage,
                "physical_pool_usage": sample.physical_pool_usage,
                "used_tokens": sample.used_tokens,
                "evictable_tokens": sample.evictable_tokens,
                "retract_total": sum(sample.retract_total.values()) if sample.retract_total else None,
                "preempt_total": sum(sample.preempt_total.values()) if sample.preempt_total else None,
            }
        )

    def rows(self) -> list[dict[str, Any]]:
        """Collected rows, stride-downsampled to the row cap.

        Returns:
            list[dict[str, Any]]: Phase-tagged samples in collection order.
        """
        if len(self._rows) <= _MAX_STORED_ROWS:
            return list(self._rows)
        # Round the stride up. Integer division gives 1 for anything under
        # twice the cap, so 5001 rows would have downsampled to 5001.
        stride = -(-len(self._rows) // _MAX_STORED_ROWS)
        return self._rows[::stride]

    def summary(self, *, aborted: bool = False) -> dict[str, Any]:
        """Build the artifact payload.

        Args:
            aborted (bool): Whether the round left through an exception path.
                Recorded rather than inferred so a consumer can tell a window
                that closed from one that was cut short.

        Returns:
            dict[str, Any]: The artifact. ``available`` is tri-state: ``None``
            means the endpoint was never reached one way or the other, which is
            not the same as reaching it and finding no pressure.
        """
        return {
            "schema_version": 1,
            "source": "metrics",
            "url": self._poller.url,
            "available": self._poller.available,
            "aborted": bool(aborted),
            "scope": self._scope,
            "capacity_tokens": self._capacity_tokens,
            "capacity_derived": self._capacity_derived,
            "capacity_gb": self._capacity_gb,
            "series_count": self._series_count,
            "prefix_cache": dict(self._prefix_cache),
            "phase_marks": self._phase_marks,
            "retract_delta": counter_delta(
                self._first_counters.get("retract", {}),
                self._last_counters.get("retract", {}),
            ),
            "preempt_delta": counter_delta(
                self._first_counters.get("preempt", {}),
                self._last_counters.get("preempt", {}),
            ),
            "sample_count": len(self._rows),
            "samples": self.rows(),
        }

    def close(self, *, aborted: bool = False) -> dict[str, Any]:
        """Finish collection and write the artifact when a path was given.

        Idempotent: the loop's ``finally`` may run after a caller has already
        closed explicitly.

        Args:
            aborted (bool): Whether the round is unwinding through an exception.

        Returns:
            dict[str, Any]: The artifact payload, written or not.
        """
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
            # Warning, not debug. Not failing the round is the requirement; being
            # quiet about it is not. A round that collected samples and then
            # dropped them on the floor looks identical afterwards to one that
            # never collected any, and nobody goes looking for a file they were
            # never told was missing.
            log.warning(
                "kv_metrics: could not write %s (%d samples collected this round are lost)",
                self._output_path,
                len(self._rows),
                exc_info=True,
            )
        return payload
