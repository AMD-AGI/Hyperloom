# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase-tagged KV-cache recording for one benchmark round, written to ``kv_metrics.json``.

Samples come from :mod:`.kv_poll`, which also states the two rules this module keeps: never raise, and never coerce a
missing reading to zero.

There are two collectors, and the artifact says which one produced its rows in ``sample_source``. On an AgentX round
aiperf is already scraping the same endpoint for its own purposes, so its export is adopted wholesale: it samples on a
tighter cadence, takes a reading at each phase boundary of its own accord, and stamps every record with the phase from
the process that owns the transition. Everything else -- synthetic benchmarks, the accuracy eval, boot, and any round
killed before aiperf flushed -- has no such export, and is covered by the round's own scrapes, each tagged with the
phase the round's phase clock last marked (``phase_source``).

Below the phase, the workload timeline comes from aiperf's per-request export; see :mod:`.agentx_timeline`. That gives
each row the trajectories, turns and requests in flight during its scrape window, and writes the event stream beside
this artifact.

What this does not do is stop the workload at a boundary. An exact counter total needs the client to quiesce, take one
snapshot, and resume, and aiperf exposes no such control: the boundary here is exact to aiperf's own stamp, and the
counter attribution to the scrape interval either side of it.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from hyperloom_common.io import atomic_write_json

from .agentx_timeline import (
    TIMELINE_ARTIFACT_NAME,
    build_events,
    correlate_rows,
    find_profile_export,
    parse_profile_export,
    write_timeline,
)
from .kv_poll import (
    KvMetricsPoller,
    KvSample,
    aggregate_series,
    read_aiperf_server_metrics,
    server_metrics_exports,
)


log = logging.getLogger(__name__)


__all__ = [
    "PHASES",
    "KvMetricsRecorder",
    "counter_delta",
]


_BOOT = "boot"
#: The one phase whose numbers may be compared against another round's.
_COMPARABLE_PHASE = "measured"
#: Collection phases. An engine process outlives the window that is actually being measured by a wide margin, and mixing
#: them makes every statistic meaningless: ``boot`` has no traffic at all, ``warmup`` runs a deliberately cold cache,
#: and ``eval`` drives accuracy traffic whose shape has nothing to do with the throughput benchmark. Only ``measured``
#: may enter a comparison.
PHASES = (_BOOT, "warmup", _COMPARABLE_PHASE, "finished", "eval")

#: Row cap before stride downsampling, mirroring ``_MN_GPU_SAMPLE_CAP``. A three-hour round at the scrape interval below
#: lands well over this.
_MAX_STORED_ROWS = 5000
#: Decimal places kept on recorded clock readings: millisecond resolution.
_CLOCK_DIGITS = 3
_CLOCK_STEPS_PER_SEC = 10**_CLOCK_DIGITS

#: Floor between scrapes, enforced on the monotonic clock -- the loop this runs in iterates faster than its nominal
#: period when a deadline bounds the slice, so counting passes would sample fastest exactly when the run is most loaded.
#:
#: Not the loop's 0.5s. A scrape is a synchronous HTTP round trip plus a parse of the engine's whole exposition, which
#: is not free at that rate, and the engine's own gauges do not refresh anywhere near it -- most of those samples would
#: be the same numbers read again. Two seconds keeps the trend visible at a fraction of the cost. Override with
#: ``INFERENCE_OPTIMIZER_KV_SCRAPE_INTERVAL_SEC``.
_SCRAPE_INTERVAL_ENV = "INFERENCE_OPTIMIZER_KV_SCRAPE_INTERVAL_SEC"
_DEFAULT_SCRAPE_INTERVAL_SEC = 2.0

#: The live rows' phase comes from the round's phase clock, which marks each transition as the round makes it.
_PHASE_MARKS_SOURCE = "phase_marks"
_AIPERF_SOURCE = "aiperf_server_metrics"


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
    ``finally``. Like the poller, a failed scrape costs the round one sample: connection and truncated-response errors
    are a miss.

    The phase is the one the round's phase clock last marked. It is never inferred from elapsed time, because the
    boundaries move by tens of minutes between rounds -- an AgentX warmup alone was measured at nearly 18 of them.
    """

    def __init__(self, *, poller: KvMetricsPoller, output_path: Path, min_interval_sec: float | None = None) -> None:
        """Prepare a recorder without contacting anything; aiperf's exports are read beside ``output_path``."""
        self._poller = poller
        self._output_path = output_path
        self._workspace = output_path.parent
        self._min_interval = resolve_scrape_interval_sec() if min_interval_sec is None else float(min_interval_sec)
        self._phase = _BOOT
        self._rows: list[dict[str, Any]] = []
        self._last_scrape_mono: float | None = None
        self._phase_marks: list[dict[str, Any]] = []
        self._first_counters: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        self._last_counters: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        self._capacity_tokens: float | None = None
        self._capacity_derived = False
        self._capacity_gb: float | None = None
        self._series_count = 0
        self._closed = False

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
        self._phase_marks.append({"phase": phase, "mono": round(float(mono), _CLOCK_DIGITS), "ts": time.time()})
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
        start_mono = time.monotonic()
        start_unix = time.time()
        sample = self._poller.sample()
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
                "scrape_start_unix": math.floor(start_unix * _CLOCK_STEPS_PER_SEC) / _CLOCK_STEPS_PER_SEC,
                "scrape_end_unix": math.ceil((start_unix + elapsed) * _CLOCK_STEPS_PER_SEC) / _CLOCK_STEPS_PER_SEC,
                "scrape_start_mono": round(start_mono, _CLOCK_DIGITS),
                "scrape_end_mono": round(end_mono, _CLOCK_DIGITS),
                "scrape_sec": round(elapsed, 4),
            },
        )
        return sample

    def _absorb(self, sample: KvSample, *, timing: dict[str, float] | None = None) -> None:
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
            "ts": round(sample.ts, _CLOCK_DIGITS),
            "mono": round(sample.mono, _CLOCK_DIGITS),
            **(timing or {"scrape_sec": 0.0}),
        }
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
        record with the phase from the process that owns the transition, so there is no boundary lag.

        Only the window aiperf covers is replaced. It starts after the server is up and exits before the accuracy eval,
        so readings outside that span -- boot, and the whole eval phase -- are ours and are kept. Inside it, its
        records replace ours outright rather than merging: two collectors reading the same counters at slightly
        different instants would let a window bracket across both and credit an increment to whichever was sampled
        either side of it. Across the boundary there is no such risk, because the two sets are disjoint in time.
        """
        export = None
        records = []
        for candidate in server_metrics_exports(self._workspace):
            records = read_aiperf_server_metrics(candidate)
            if records:
                export = candidate
                break
        if export is None:
            return None
        # Kept by time, not by phase label. aiperf covers one contiguous window -- it starts after the server is up
        # and exits before the accuracy eval -- so a live row outside that span is a reading nothing else took,
        # while one inside it is redundant by definition. Judging on the label instead would keep a row whose
        # timestamp lands mid-window and let it close a counter window it has no business closing.
        covered_from = records[0][0].ts
        covered_to = records[-1][0].ts
        kept = [
            r
            for r in self._rows
            if isinstance(r.get("ts"), (int, float)) and not (covered_from <= r["ts"] <= covered_to)
        ]
        # Rebuilt from scratch so no superseded live reading survives into the counter windows below.
        self._rows = []
        self._first_counters = {}
        self._last_counters = {}
        previous_phase = self._phase
        for sample, timing, phase in records:
            # An unstamped record is one aiperf took before any phase began -- its baseline capture, of an idle
            # pool. That is boot. Defaulting it to "measured" put readings of a pool under no load into the one
            # phase allowed into a comparison: observed on a live round, where three such rows produced a measured
            # prefix-cache delta of 34,395 describing nothing that happened.
            self._phase = phase or _BOOT
            self._absorb(sample, timing=timing)
        self._phase = previous_phase
        if kept:
            # Time order, because the counter windows below are bracketed by walking the rows in sequence.
            self._rows = sorted([*self._rows, *kept], key=lambda r: r.get("ts") or 0.0)
        # Adjacent phases share the reading at their boundary here too. aiperf takes its own scrape at each
        # transition, but tags it with the phase that is ending, so without this the next phase's window would open
        # at its first periodic sample and the increment in between would be credited to neither.
        self._rebuild_counter_windows()
        return str(export)

    def _rebuild_counter_windows(self) -> None:
        """Re-bracket every counter from the rows, in time order.

        Adjacent phases share the snapshot at their boundary -- the reading that closes one opens the next -- so this
        keeps the gap-free property the boundary scrapes give.
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
        export = find_profile_export(self._workspace)
        if export is None:
            return None
        records = parse_profile_export(export)
        if not records:
            return None
        events, summary = build_events(records)
        # Correlation runs over every record, so the counts on a row stay exact even when the event stream above
        # had to be sampled.
        summary["rows_correlated"] = correlate_rows(self._rows, records)
        path = self._workspace / TIMELINE_ARTIFACT_NAME
        summary["path"] = path.name if write_timeline(path, events) else None
        summary["source"] = str(export)
        return summary

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
        adopted = self._adopt_aiperf_samples()
        source = _AIPERF_SOURCE if adopted else None
        workload = self._build_workload_timeline()
        retract_measured, retract_by_phase = self._counter_deltas("retract")
        preempt_measured, preempt_by_phase = self._counter_deltas("preempt")
        return {
            "phase_source": source or _PHASE_MARKS_SOURCE,
            "workload_timeline": workload,
            "schema_version": 1,
            "source": "metrics",
            # Which collector produced the rows. The two differ in cadence and in how the phase was decided, so a
            # consumer comparing rounds has to be able to tell them apart.
            "sample_source": source or "watchdog_scrape",
            "aiperf_server_metrics_path": adopted,
            "url": self._poller.url,
            # Whether this round obtained readings at all, by whichever collector: aiperf can collect a round whose
            # engine port the round's own poller never reached.
            "available": True if adopted else self._poller.available,
            "aborted": bool(aborted),
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
        """Finish collection and write the artifact.

        Idempotent: the loop's ``finally`` may run after a caller has already closed explicitly.
        """
        # Final boundary reading, for the same reason the phase transitions take one: without it the last phase ends at
        # whenever its last periodic sample happened to land, and everything after that is lost.
        if not self._closed:
            self._scrape(time.monotonic())
        payload = self.summary(aborted=aborted)
        if self._closed:
            return payload
        self._closed = True
        try:
            atomic_write_json(self._output_path, payload)
        except (OSError, TypeError, ValueError):
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
