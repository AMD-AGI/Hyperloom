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
    "KvMetricsPoller",
    "KvSample",
    "canonical_label_key",
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
#: stall and soft-deadline gates are measured on.
_SCRAPE_TIMEOUT_SEC = 1.5

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
_SGL_EVICTABLE = (
    "sglang:kv_evictable_tokens",
    "sglang:swa_evictable_tokens",
    "sglang:mamba_evictable_tokens",
)
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
    """Read a single-series gauge, summing across label sets when several exist.

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
    return sum(value for _, value in series)


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


def _sum_scalar(families: ParsedFamilies, names: tuple[str, ...]) -> float | None:
    """Sum of several gauges, ignoring the ones not present.

    Args:
        families (ParsedFamilies): Parsed exposition.
        names (tuple[str, ...]): Metric names to add.

    Returns:
        float | None: The sum, or ``None`` when none of the names appeared.
    """
    values = [v for name in names if (v := _scalar(families, name)) is not None]
    return sum(values) if values else None


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
        capacity_tokens (float | None): Pool size, derived as used + available +
            evictable. Not read from a gauge because none reports it directly.
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
    evictable = _sum_scalar(families, _SGL_EVICTABLE)

    capacity: float | None = None
    if used is not None and available is not None:
        capacity = used + available + (evictable or 0.0)

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
            with urllib.request.urlopen(self.url, timeout=self.timeout_sec) as response:  # noqa: S310
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
