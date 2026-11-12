# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for KV collection and for the checkpoint judge riding the watchdog loop.

Covers the phase split, the recorder itself, and the ``finally`` that closes the judge on every exit path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk
from inference_suite.telemetry.kv_poll import KvSample
from inference_suite.telemetry.kv_record import (
    KV_ARTIFACT_NAME,
    KvMetricsRecorder,
    counter_delta,
)


class _StubPoller:
    """Hands out canned samples without touching the network."""

    def __init__(self, samples, available=True):
        self.url = "http://127.0.0.1:1/metrics"
        self.available = available
        self._samples = list(samples)
        self.calls = 0

    def sample(self):
        self.calls += 1
        return self._samples.pop(0) if self._samples else None


def _sample(**kwargs) -> KvSample:
    base = {"ts": 1.0, "mono": 1.0, "engine": "sglang", "active_pool_usage": 0.5}
    base.update(kwargs)
    return KvSample(**base)


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
def test_rows_are_tagged_with_the_phase_they_were_taken_in(tmp_path):
    poller = _StubPoller([_sample() for _ in range(6)])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)

    rec.tick(0.0)
    rec.note_phase("warmup", 1.0)  # boundary reading closes boot
    rec.tick(1.0)
    rec.note_phase("measured", 2.0)  # boundary reading closes warmup
    rec.tick(2.0)

    assert [r["phase"] for r in rec.rows()] == ["boot", "boot", "warmup", "warmup", "measured"]


def test_a_boundary_reading_closes_one_phase_and_opens_the_next(tmp_path):
    """Deriving a phase total from its own first and last periodic samples
    leaves up to a full interval at each end credited to neither phase."""
    poller = _StubPoller([_sample(retract_total=_grouped(a=v)) for v in (7.0, 7.0, 9.0)])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    rec.note_phase("measured", 1.0)
    rec.tick(2.0)

    # The single boundary read is both boot's endpoint and measured's baseline.
    assert rec.summary()["retract_delta_by_phase"]["measured"] == 2.0


def test_every_row_carries_gauges_and_raw_counters(tmp_path):
    """Counters only at boundaries would blind the interior of a phase: one
    burst and a steady trickle produce the same total, and a mid-phase engine
    restart is invisible without the series."""
    poller = _StubPoller([_sample(capacity_tokens=32768.0, retract_total=_grouped(a=48.0))])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    row = rec.rows()[0]
    assert row["capacity_tokens"] == 32768.0
    assert row["counters_by_series"]["retract"] == _grouped(a=48.0)
    assert "scrape_sec" in row and "mono" in row and "ts" in row


def test_scrape_interval_is_enforced_on_the_monotonic_clock(tmp_path):
    """The loop's slice shrinks below its nominal period near a deadline, so
    counting passes would sample fastest exactly when the run is most loaded."""
    poller = _StubPoller([_sample() for _ in range(5)])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.5)

    rec.tick(0.0)
    rec.tick(0.1)
    rec.tick(0.2)
    rec.tick(0.6)

    assert poller.calls == 2


def test_unknown_phase_is_ignored_rather_than_raising(tmp_path):
    rec = KvMetricsRecorder(
        output_path=tmp_path / KV_ARTIFACT_NAME, poller=_StubPoller([_sample()]), min_interval_sec=0.0
    )
    rec.note_phase("nonsense", 1.0)
    rec.tick(2.0)
    assert rec.rows()[0]["phase"] == "boot"
    assert rec.summary()["phase_marks"] == []


def test_capacity_is_latched_from_the_first_reading_that_has_it(tmp_path):
    """Capacity is only observable while the engine is up; a round that ends
    with the server gone must still carry it."""
    poller = _StubPoller([_sample(capacity_tokens=32768.0, capacity_gb=180.0), _sample()])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    summary = rec.summary()
    assert summary["capacity_tokens"] == 32768.0
    assert summary["capacity_gb"] == 180.0


def _grouped(**shards) -> dict[str, dict[str, float]]:
    """One independent unit carrying the given shard readings."""
    return {"": dict(shards)}


def test_counter_delta_credits_only_the_post_restart_count():
    """An engine restart zeroes its series; a flat subtraction would go negative."""
    assert counter_delta(_grouped(a=10.0), _grouped(a=48.0)) == 38.0
    assert counter_delta(_grouped(a=100.0), _grouped(a=3.0)) == 3.0
    assert counter_delta({}, {}) is None
    assert counter_delta({}, _grouped(a=5.0)) == 5.0


def test_counter_delta_does_not_multiply_lockstep_ranks():
    """Eight ranks reporting 48 describe 48 retracts, not 384."""
    first = {"": {f'tp_rank="{i}"': 0.0 for i in range(8)}}
    last = {"": {f'tp_rank="{i}"': 48.0 for i in range(8)}}

    assert counter_delta(first, last) == 48.0


def test_counter_deltas_are_attributed_to_the_phase_that_earned_them(tmp_path):
    """An engine retracts through warmup and the accuracy eval too.

    A round-wide difference folds both into the number that is supposed to
    describe the measured window alone, and unlike the gauge rows -- which carry
    their phase and can be re-sliced -- a counter difference cannot be taken
    apart afterwards.
    """
    # Reads in scrape order. The boundary scrapes are the 0, 5, 7 and the final
    # 10; each closes one phase and opens the next, so the windows meet.
    reads = (0.0, 2.0, 5.0, 5.0, 6.0, 7.0, 7.0, 9.0, 10.0, 10.0)
    poller = _StubPoller([_sample(retract_total=_grouped(a=v)) for v in reads])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)

    rec.note_phase("warmup", 0.0)
    rec.tick(1.0)
    rec.tick(2.0)
    rec.note_phase("measured", 3.0)
    rec.tick(4.0)
    rec.tick(5.0)
    rec.note_phase("eval", 6.0)
    rec.tick(7.0)
    rec.tick(8.0)
    rec.close()

    summary = rec.summary()
    assert summary["retract_delta"] == 2.0
    by_phase = summary["retract_delta_by_phase"]
    assert (by_phase["warmup"], by_phase["measured"], by_phase["eval"]) == (5.0, 2.0, 3.0)


def test_measured_delta_is_none_when_no_sample_landed_there(tmp_path):
    """Not an increment of zero: nothing was ever measured."""
    poller = _StubPoller([_sample(retract_total=_grouped(a=5.0))])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.note_phase("warmup", 0.0)
    rec.tick(0.0)

    assert rec.summary()["retract_delta"] is None


def test_counter_delta_adds_independent_engines():
    first = {'engine="0"': {'engine="0"': 0.0}, 'engine="1"': {'engine="1"': 0.0}}
    last = {'engine="0"': {'engine="0"': 48.0}, 'engine="1"': {'engine="1"': 48.0}}

    assert counter_delta(first, last) == 96.0


def test_rows_and_summary_use_the_same_counter_rule(tmp_path):
    """Two rules inside one artifact is worse than either being wrong: nothing
    on the page says which number was computed which way."""
    shards = {"": {f'tp_rank="{i}"': 48.0 for i in range(8)}}
    poller = _StubPoller([_sample(retract_total=shards)])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    assert rec.rows()[0]["retract_total"] == 48.0


def test_downsampling_respects_the_cap(tmp_path):
    """Integer division gave a stride of 1 for anything under twice the cap, so
    5001 rows downsampled to 5001."""
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=_StubPoller([]))
    rec._rows = [{"i": i} for i in range(5001)]

    assert len(rec.rows()) <= 5000


def test_prefix_cache_counters_are_bracketed_not_snapshotted(tmp_path):
    """Cumulative counters, and the engine outlives the round under warm reuse.

    The latest absolute value therefore carries whatever the previous round
    warmed the cache with, which is not attributable to this one.
    """
    poller = _StubPoller(
        [
            _sample(prefix_cache_queries=_grouped(a=1000.0), prefix_cache_hits=_grouped(a=800.0)),
            _sample(prefix_cache_queries=_grouped(a=1400.0), prefix_cache_hits=_grouped(a=1100.0)),
        ]
    )
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    window = rec.summary()["prefix_cache"]
    assert window["prefix_cache_queries_delta"] == 400.0
    assert window["prefix_cache_hits_delta"] == 300.0
    assert window["prefix_cache_queries_delta_by_phase"]["measured"] == 400.0


def test_prefix_cache_counters_add_across_independent_engines(tmp_path):
    """Two vLLM engines serving 200 lookups each did 400, not 200. Taking the
    max across series before diffing halved every cache figure on a DP
    deployment."""
    first = {'engine="0"': {'engine="0"': 0.0}, 'engine="1"': {'engine="1"': 0.0}}
    last = {'engine="0"': {'engine="0"': 200.0}, 'engine="1"': {'engine="1"': 200.0}}
    poller = _StubPoller([_sample(prefix_cache_queries=first), _sample(prefix_cache_queries=last)])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    assert rec.summary()["prefix_cache"]["prefix_cache_queries_delta"] == 400.0


def test_prefix_cache_restart_credits_only_the_post_restart_count(tmp_path):
    poller = _StubPoller(
        [_sample(cached_tokens_total=_grouped(a=900.0)), _sample(cached_tokens_total=_grouped(a=12.0))]
    )
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    assert rec.summary()["prefix_cache"]["cached_tokens_total_delta"] == 12.0


def test_capacity_provenance_and_series_count_reach_the_artifact(tmp_path):
    poller = _StubPoller([_sample(capacity_tokens=32768.0, capacity_derived=True, series_count=8)])
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    summary = rec.summary()
    assert summary["capacity_derived"] is True
    assert summary["series_count"] == 8


def test_summary_availability_is_tristate(tmp_path):
    """Never reached is not the same as reached and found quiet."""
    unknown = KvMetricsRecorder(
        output_path=tmp_path / KV_ARTIFACT_NAME, poller=_StubPoller([], available=None)
    ).summary()
    assert unknown["available"] is None

    off = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=_StubPoller([], available=False)).summary()
    assert off["available"] is False


def test_close_writes_the_artifact_and_is_idempotent(tmp_path):
    out = tmp_path / KV_ARTIFACT_NAME
    poller = _StubPoller([_sample(retract_total={"": {"": 48.0}})])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0, output_path=out)
    rec.tick(0.0)

    rec.close()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["sample_count"] == 1
    assert payload["aborted"] is False

    rec.close(aborted=True)  # the loop's finally may run after an explicit close
    assert json.loads(out.read_text(encoding="utf-8"))["aborted"] is False


def test_close_never_raises_on_an_unwritable_path(tmp_path):
    rec = KvMetricsRecorder(poller=_StubPoller([]), output_path=tmp_path / "nope" / "x" / KV_ARTIFACT_NAME)
    # Parent dirs are created by the writer; make that impossible instead.
    (tmp_path / "nope").write_text("not a directory", encoding="utf-8")

    payload = rec.close()
    assert payload["schema_version"] == 1


# --------------------------------------------------------------------------
# the round's checkpoint judge in the watchdog loop
# --------------------------------------------------------------------------
class _Judge:
    """Minimal stand-in for the judge a round's checkpoint builds."""

    def __init__(self):
        self.ticks = 0
        self.closed_aborted: bool | None = None

    def tick(self, mono):
        self.ticks += 1

    def close(self, *, aborted=False):
        self.closed_aborted = aborted


class _Checkpoint:
    def __init__(self, judge):
        self._judge = judge

    def judge(self):
        return self._judge


class _DoneProc:
    args = ["x"]

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


class _HangingProc:
    args = ["x"]

    def poll(self):
        return None

    def wait(self, timeout=None):
        raise sk.subprocess.TimeoutExpired(self.args, timeout)


class _Capture:
    """A capture holding what the child wrote, with no pipe activity of its own."""

    last_activity_at = None

    def finish(self):
        return ("out", "err")


def _watch(proc, watch):
    return sk._communicate_with_watchdog(proc, sk._WatchedRound(watch), capture=_Capture())


def test_the_judge_rules_from_the_first_poll_and_is_closed_on_the_normal_return_path():
    judge = _Judge()
    out = _watch(_DoneProc(), sk.RoundWatch(timeout=5, checkpoint=_Checkpoint(judge)))

    assert out == ("out", "err")
    assert judge.ticks >= 1
    assert judge.closed_aborted is False


def test_the_judge_is_closed_as_aborted_when_a_gate_raises():
    """The failure a consumer cannot recover from is a window that never closed."""
    judge = _Judge()
    with pytest.raises(sk.subprocess.TimeoutExpired):
        _watch(_HangingProc(), sk.RoundWatch(timeout=0.01, checkpoint=_Checkpoint(judge)))

    assert judge.closed_aborted is True


def test_a_failing_judge_fails_the_round():
    """A judge absorbs its own read and write failures, so anything else it raises is a defect that surfaces."""

    class _Exploding(_Judge):
        def tick(self, mono):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="^boom$"):
        _watch(_DoneProc(), sk.RoundWatch(timeout=5, checkpoint=_Checkpoint(_Exploding())))


def test_artifact_name_is_stable():
    assert Path(KV_ARTIFACT_NAME).suffix == ".json"


def test_artifact_is_in_the_package_globs():
    """It lives in the round workspace, which the bundle does not otherwise reach."""
    from hyperloom.inference_optimizer.breakdown.session_package import PACKAGE_GLOBS

    assert "runs/**/kv_metrics.json" in PACKAGE_GLOBS


# ---------------------------------------------------------------------------
# aiperf's own server-metrics export
# ---------------------------------------------------------------------------


def test_a_round_that_nobody_reached_is_still_unavailable(tmp_path):
    """No export and a poller that gave up: the honest answer is still false."""

    class _GaveUp:
        url = "http://127.0.0.1:8888/metrics"
        available = False

        def sample(self):
            return None

    rec = KvMetricsRecorder(poller=_GaveUp(), min_interval_sec=0, output_path=tmp_path / KV_ARTIFACT_NAME)
    rec.tick(1.0)
    payload = rec.summary()

    assert payload["available"] is False
    assert payload["sample_count"] == 0


def test_aiperf_aggregate_json_timeslices_are_adopted(tmp_path):
    from inference_suite.telemetry.kv_poll import (
        aggregate_series,
        read_aiperf_server_metrics,
    )

    artifact_dir = tmp_path / "aiperf_artifacts"
    artifact_dir.mkdir()
    unusable = artifact_dir / "server_metrics_export.json"
    unusable.write_text("{not json\n", encoding="utf-8")
    nested = tmp_path / "benchmark_vllm_1" / "aiperf_artifacts"
    nested.mkdir(parents=True)
    path = nested / "server_metrics_export.json"
    gauge = {
        "type": "gauge",
        "series": [
            {
                "labels": {"engine": "0"},
                "timeslices": [
                    {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000, "avg": 0.25},
                ],
            }
        ],
    }
    counter = {
        "type": "counter",
        "series": [
            {
                "labels": {"engine": "0"},
                "timeslices": [
                    {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000, "total": 10},
                ],
            }
        ],
    }
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "phase_time_ranges": {
                        "warmup": {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000},
                        "profiling": {"start_ns": 3_000_000_000, "end_ns": 4_000_000_000},
                    }
                },
                "metrics_phase": "profiling",
                "warmup_metrics": {
                    "vllm:kv_cache_usage_perc": gauge,
                    "vllm:prefix_cache_hits": counter,
                },
                "metrics": {
                    "vllm:kv_cache_usage_perc": {
                        **gauge,
                        "series": [
                            {
                                "labels": {"engine": "0"},
                                "timeslices": [
                                    {
                                        "start_ns": 1_000_000_000,
                                        "end_ns": 2_000_000_000,
                                        "avg": 0.99,
                                    },
                                    {
                                        "start_ns": 3_000_000_000,
                                        "end_ns": 4_000_000_000,
                                        "avg": 0.5,
                                    },
                                ],
                            }
                        ],
                    },
                    "vllm:prefix_cache_hits": {
                        **counter,
                        "series": [
                            {
                                "labels": {"engine": "0"},
                                "timeslices": [
                                    {
                                        "start_ns": 1_000_000_000,
                                        "end_ns": 2_000_000_000,
                                        "total": 999,
                                    },
                                    {
                                        "start_ns": 3_000_000_000,
                                        "end_ns": 4_000_000_000,
                                        "total": 20,
                                    },
                                ],
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rows = read_aiperf_server_metrics(path)
    assert [phase for _sample, _timing, phase in rows] == ["warmup", "warmup", "measured", "measured"]
    assert [sample.active_pool_usage for sample, _timing, _phase in rows] == [None, 0.25, None, 0.5]
    assert [aggregate_series(sample.prefix_cache_hits) for sample, _timing, _phase in rows] == [0, 10, 10, 30]
    assert aggregate_series(rows[-1][0].prefix_cache_hits) == 30.0

    payload = KvMetricsRecorder(
        poller=_StubPoller([]),
        min_interval_sec=0,
        output_path=tmp_path / KV_ARTIFACT_NAME,
    ).summary()
    assert payload["aiperf_server_metrics_path"] == str(path)
    assert payload["prefix_cache"]["prefix_cache_hits_delta_by_phase"] == {
        "warmup": 10.0,
        "measured": 20.0,
    }


def test_a_round_without_an_aiperf_export_keeps_the_watchdog_rows(tmp_path):
    """Synthetic benchmarks, the accuracy eval and any round killed before aiperf
    flushed have no export, and are still the majority of rounds."""
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(active_pool_usage=0.42)]),
        min_interval_sec=0,
        output_path=tmp_path / KV_ARTIFACT_NAME,
    )
    rec.tick(1.0)
    payload = rec.summary()

    assert payload["sample_source"] == "watchdog_scrape"
    assert payload["aiperf_server_metrics_path"] is None
    assert [r["active_pool_usage"] for r in payload["samples"]] == [0.42]


def test_counters_resolve_under_aiperfs_family_naming(tmp_path):
    """aiperf names a counter by its Prometheus family, dropping the ``_total``.

    Observed on a live AgentX round: of 47 families in one record, exactly one
    ended in ``_total``, and vLLM's preemption counter -- written
    ``vllm:num_preemptions_total`` in the exposition this module reads everywhere
    else -- appears as ``vllm:num_preemptions``. Without the alias all three
    cumulative counters come back empty on this path, and an empty counter looks
    exactly like a round that never retracted.
    """
    from inference_suite.telemetry.kv_poll import families_from_aiperf_record

    families = families_from_aiperf_record(
        {
            "vllm:num_preemptions": [{"labels": {"engine": "0"}, "value": 7.0}],
            "sglang:num_retracted_requests": [{"labels": {"tp_rank": "0"}, "value": 48.0}],
            "sglang:cached_tokens": [{"labels": {}, "value": 4600439.0}],
        }
    )

    assert families["vllm:num_preemptions_total"] == [({"engine": "0"}, 7.0)]
    assert families["sglang:num_retracted_requests_total"] == [({"tp_rank": "0"}, 48.0)]
    assert families["sglang:cached_tokens_total"] == [({}, 4600439.0)]
    # The name aiperf actually used still resolves too.
    assert families["vllm:num_preemptions"] == [({"engine": "0"}, 7.0)]


def test_histogram_samples_are_skipped_not_mistaken_for_gauges(tmp_path):
    """aiperf carries histograms as buckets with no ``value``."""
    from inference_suite.telemetry.kv_poll import families_from_aiperf_record

    families = families_from_aiperf_record(
        {
            "sglang:token_usage": [{"labels": {}, "value": 0.5}],
            "sglang:e2e_latency_seconds": [{"labels": {}, "buckets": {"0.1": 3.0}, "sum": 1.0, "count": 3.0}],
        }
    )

    assert families["sglang:token_usage"] == [({}, 0.5)]
    # Carried no scalar, so it is absent entirely rather than present and empty.
    assert not any(name.startswith("sglang:e2e_latency_seconds") for name in families)


def _refuse(monkeypatch):
    """Make every scrape fail the way a closed port does."""
    from inference_suite.telemetry import kv_poll as km

    def _boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(km._OPENER, "open", _boom)


def test_poller_does_not_give_up_on_a_slow_start(monkeypatch):
    """Three misses inside six seconds is a slow engine, not an absent endpoint.

    Giving up on the count alone parked collection for the rest of a round over a
    server that was merely still binding.
    """
    from inference_suite.telemetry.kv_poll import KvMetricsPoller

    _refuse(monkeypatch)
    poller = KvMetricsPoller("http://127.0.0.1:1/metrics")
    for _ in range(5):
        poller.fetch()

    assert poller.available is None  # still unknown, still trying


def test_poller_gives_up_once_the_grace_has_also_passed(monkeypatch):
    """An endpoint that answered nothing for a minute is not coming back."""
    import time

    from inference_suite.telemetry.kv_poll import _GIVE_UP_GRACE_SEC, KvMetricsPoller

    _refuse(monkeypatch)
    poller = KvMetricsPoller("http://127.0.0.1:1/metrics")
    poller.fetch()
    # Age the first attempt rather than the process clock: patching time.monotonic
    # globally lets any other caller consume the fake readings.
    poller._first_attempt_mono = time.monotonic() - (_GIVE_UP_GRACE_SEC + 1)
    poller.fetch()
    poller.fetch()

    assert poller.available is False


def _now() -> float:
    """A base near the real wall clock, which the epoch sanity check requires."""
    import time

    return time.time()


def test_rows_carry_scrape_start_and_end_not_just_a_duration(tmp_path):
    """A gauge read across a 300ms window describes some instant inside it, and
    a single stamp would invite an alignment precision nobody measured."""
    rec = KvMetricsRecorder(
        output_path=tmp_path / KV_ARTIFACT_NAME, poller=_StubPoller([_sample()]), min_interval_sec=0
    )
    rec.tick(1.0)
    row = rec.summary()["samples"][0]

    for key in ("scrape_start_unix", "scrape_end_unix", "scrape_start_mono", "scrape_end_mono", "scrape_sec"):
        assert key in row, key
    assert row["scrape_end_mono"] >= row["scrape_start_mono"]
    assert row["scrape_end_unix"] >= row["scrape_start_unix"]


def test_the_stored_scrape_window_contains_the_real_one(tmp_path):
    """Widened to the enclosing millisecond, never rounded to the nearest.

    These bounds are what workload records are joined against. Nearest-rounding
    shrinks the window by up to half a millisecond at each end, so a request that
    began inside a sub-millisecond scrape falls outside the window that observed
    it -- silently, and only on a host whose clock is fine enough to notice.
    """
    import time

    before = time.time()
    rec = KvMetricsRecorder(
        output_path=tmp_path / KV_ARTIFACT_NAME, poller=_StubPoller([_sample()]), min_interval_sec=0
    )
    rec.tick(1.0)
    row = rec.summary()["samples"][0]

    # The scrape began after this test did, and the floor can only move that earlier.
    assert row["scrape_start_unix"] <= time.time()
    assert row["scrape_start_unix"] >= before - 0.001
    # The stored window is never narrower than the round trip it brackets. Asserted
    # against the measured duration rather than a clock read after ``tick`` returns:
    # that read includes building the row, which the window is not claiming to cover.
    assert row["scrape_end_unix"] - row["scrape_start_unix"] >= row["scrape_sec"]
    # Still milliseconds, not full float noise.
    assert row["scrape_start_unix"] == round(row["scrape_start_unix"], 3)
    assert row["scrape_end_unix"] == round(row["scrape_end_unix"], 3)


def test_a_request_starting_at_the_scrape_instant_is_still_correlated(tmp_path):
    """The regression: a sub-millisecond scrape must not lose the work it saw."""
    from inference_suite.telemetry.agentx_timeline import (
        correlate_rows,
        parse_profile_export,
    )

    art = tmp_path / "aiperf_artifacts"
    art.mkdir()
    # A request beginning a quarter of a millisecond after the window opens: inside
    # it, but below the resolution the window is stored at.
    start_ns = 1_789_000_000_144_255_000
    (art / "profile_export.jsonl").write_text(
        json.dumps(
            {
                "metadata": {
                    "x_request_id": "req-1",
                    "x_correlation_id": "traj-1",
                    "turn_index": 0,
                    "request_start_ns": start_ns,
                    "request_end_ns": start_ns + 1_000_000_000,
                },
                "metrics": {},
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = parse_profile_export(art / "profile_export.jsonl")
    exact = start_ns / 1e9
    rows = [
        {
            "scrape_start_unix": math.floor((exact - 0.00025) * 1000) / 1000,
            "scrape_end_unix": math.ceil((exact - 0.00020) * 1000) / 1000,
        }
    ]

    assert correlate_rows(rows, records) == 1


def test_recorder_writes_the_workload_timeline_beside_the_kv_artifact(tmp_path):
    """End to end: aiperf's export in, event stream plus correlated rows out.

    The per-request export is written at aiperf's default export level, so this
    needs nothing added to the invocation and nothing from upstream.
    """
    now = _now()
    art = tmp_path / "aiperf_artifacts"
    art.mkdir()
    start_ns = int(now * 1e9)
    (art / "profile_export.jsonl").write_text(
        json.dumps(
            {
                "metadata": {
                    "x_request_id": "req-1",
                    "x_correlation_id": "traj-1",
                    "conversation_id": "conv-1",
                    "turn_index": 0,
                    "request_start_ns": start_ns,
                    # Generously long, so a loaded CI box cannot end the request
                    # before the scrape below happens. What this test is about is
                    # the wiring, not the width of the window.
                    "request_end_ns": start_ns + 600_000_000_000,
                    "benchmark_phase": "profiling",
                },
                "metrics": {"time_to_first_token": {"value": 250.0, "unit": "ms"}},
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(ts=now + 5)]),
        min_interval_sec=0,
        output_path=tmp_path / KV_ARTIFACT_NAME,
    )
    rec.tick(1.0)
    payload = rec.close()

    timeline = payload["workload_timeline"]
    assert timeline["requests"] == 1
    assert timeline["trajectories"] == 1
    assert timeline["path"] == "agentx_timeline.jsonl"
    assert timeline["rows_correlated"] == 1

    events = [
        json.loads(line) for line in (tmp_path / "agentx_timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {e["event"] for e in events} >= {"trajectory_start", "turn_start", "request_start"}

    # The sample taken 5s in saw that trajectory running; the timeline names it.
    assert payload["samples"][0]["workload"]["in_flight"]["trajectories"] == 1
    assert {e["trajectory_id"] for e in events if e["event"] == "trajectory_start"} == {"traj-1"}


def test_a_round_without_an_aiperf_export_reports_no_timeline(tmp_path):
    """Every synthetic benchmark is this case, and it is not a failure."""
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample()]),
        min_interval_sec=0,
        output_path=tmp_path / KV_ARTIFACT_NAME,
    )
    rec.tick(1.0)
    payload = rec.close()

    assert payload["workload_timeline"] is None
    assert not (tmp_path / "agentx_timeline.jsonl").exists()


class _RaisingPoller(_StubPoller):
    def sample(self):
        raise RuntimeError("sample bug")


def test_a_failing_poller_sample_propagates_from_the_scrape(tmp_path):
    rec = KvMetricsRecorder(output_path=tmp_path / KV_ARTIFACT_NAME, poller=_RaisingPoller([]), min_interval_sec=0.0)

    with pytest.raises(RuntimeError, match="sample bug"):
        rec.tick(0.0)

    assert rec.rows() == []


def test_a_failing_aiperf_read_propagates_from_the_summary(tmp_path, monkeypatch):
    from inference_suite.telemetry import kv_record as kv

    def _boom(_path):
        raise RuntimeError("aiperf read bug")

    monkeypatch.setattr(kv, "server_metrics_exports", lambda _workspace: [tmp_path / "export.json"])
    monkeypatch.setattr(kv, "read_aiperf_server_metrics", _boom)
    rec = KvMetricsRecorder(poller=_StubPoller([]), min_interval_sec=0.0, output_path=tmp_path / KV_ARTIFACT_NAME)

    with pytest.raises(RuntimeError, match="aiperf read bug"):
        rec.summary()


def test_a_failing_timeline_build_propagates_from_the_summary(tmp_path, monkeypatch):
    from inference_suite.telemetry import kv_record as kv

    def _boom(_workspace):
        raise RuntimeError("timeline bug")

    monkeypatch.setattr(kv, "find_profile_export", _boom)
    rec = KvMetricsRecorder(poller=_StubPoller([]), min_interval_sec=0.0, output_path=tmp_path / KV_ARTIFACT_NAME)

    with pytest.raises(RuntimeError, match="timeline bug"):
        rec.summary()
