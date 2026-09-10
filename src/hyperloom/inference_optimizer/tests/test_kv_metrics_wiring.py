# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for KV collection riding the watchdog loop.

Covers the phase split, the recorder itself, and the ``finally`` that closes a
window on every exit path. Also covers the scanner's truncation branch, which
has been reachable and untested since it was written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk
from hyperloom.orchestrator.actions.executors._kv_metrics import (
    KV_ARTIFACT_NAME,
    KvMetricsRecorder,
    KvSample,
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
# scanner
# --------------------------------------------------------------------------
def test_scan_survives_truncation_by_rescanning_from_the_top(tmp_path):
    """A rotated log is shorter than the consumed offset; markers in the new
    content must still be seen rather than skipped past."""
    log_path = tmp_path / "server.log"
    log_path.write_text("x" * 500 + "\n", encoding="utf-8")
    first = sk._scan_server_log_increment(str(log_path), 0)
    assert first.offset == log_path.stat().st_size

    log_path.write_text("Application startup complete\n", encoding="utf-8")
    after = sk._scan_server_log_increment(str(log_path), first.offset)

    assert after.saw_ready is True
    assert after.offset == log_path.stat().st_size


def test_scan_reports_agentx_phase_boundaries(tmp_path):
    """aiperf prints these itself; we add no marker of our own."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Phase warmup (warmup) started | target: 44 requests\n", encoding="utf-8")
    warm = sk._scan_server_log_increment(str(log_path), 0)
    assert warm.saw_warmup_begin is True and warm.saw_measured_begin is False

    with log_path.open("a", encoding="utf-8") as f:
        f.write("Phase profiling (profiling) started | phase_index=0\n")
    measured = sk._scan_server_log_increment(str(log_path), warm.offset)
    assert measured.saw_measured_begin is True and measured.saw_warmup_begin is False


def test_resolve_scan_logs_includes_the_agentx_client_log(tmp_path):
    """An AgentX workspace has no benchmark_stderr.log, so without this the
    phase lines are unreachable."""
    bench = tmp_path / "benchmark_sglang_20260903"
    (bench / "aiperf_artifacts" / "logs").mkdir(parents=True)
    (bench / "server.log").write_text("up\n", encoding="utf-8")
    (bench / "aiperf_artifacts" / "logs" / "aiperf.log").write_text("phases\n", encoding="utf-8")

    resolved = sk._resolve_scan_logs(str(tmp_path / "server.log"))

    assert any(p.endswith("aiperf.log") for p in resolved)


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
def test_rows_are_tagged_with_the_phase_they_were_taken_in():
    poller = _StubPoller([_sample() for _ in range(6)])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)

    rec.tick(0.0)
    rec.note_phase("warmup", 1.0)  # boundary reading closes boot
    rec.tick(1.0)
    rec.note_phase("measured", 2.0)  # boundary reading closes warmup
    rec.tick(2.0)

    assert [r["phase"] for r in rec.rows()] == ["boot", "boot", "warmup", "warmup", "measured"]


def test_a_boundary_reading_closes_one_phase_and_opens_the_next():
    """Deriving a phase total from its own first and last periodic samples
    leaves up to a full interval at each end credited to neither phase."""
    poller = _StubPoller([_sample(retract_total=_grouped(a=v)) for v in (7.0, 7.0, 9.0)])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    rec.note_phase("measured", 1.0)
    rec.tick(2.0)

    # The single boundary read is both boot's endpoint and measured's baseline.
    assert rec.summary()["retract_delta_by_phase"]["measured"] == 2.0


def test_every_row_carries_gauges_and_raw_counters():
    """Counters only at boundaries would blind the interior of a phase: one
    burst and a steady trickle produce the same total, and a mid-phase engine
    restart is invisible without the series."""
    poller = _StubPoller([_sample(capacity_tokens=32768.0, retract_total=_grouped(a=48.0))])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    row = rec.rows()[0]
    assert row["capacity_tokens"] == 32768.0
    assert row["counters_by_series"]["retract"] == _grouped(a=48.0)
    assert "scrape_sec" in row and "mono" in row and "ts" in row


def test_scrape_interval_is_enforced_on_the_monotonic_clock():
    """The loop's slice shrinks below its nominal period near a deadline, so
    counting passes would sample fastest exactly when the run is most loaded."""
    poller = _StubPoller([_sample() for _ in range(5)])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.5)

    rec.tick(0.0)
    rec.tick(0.1)
    rec.tick(0.2)
    rec.tick(0.6)

    assert poller.calls == 2


def test_unknown_phase_is_ignored_rather_than_raising():
    rec = KvMetricsRecorder(poller=_StubPoller([]), min_interval_sec=0.0)
    rec.note_phase("nonsense", 1.0)
    assert rec.phase == "boot"


def test_capacity_is_latched_from_the_first_reading_that_has_it():
    """Capacity is only observable while the engine is up; a round that ends
    with the server gone must still carry it."""
    poller = _StubPoller([_sample(capacity_tokens=32768.0, capacity_gb=180.0), _sample()])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
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


def test_counter_deltas_are_attributed_to_the_phase_that_earned_them():
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
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)

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


def test_measured_delta_is_none_when_no_sample_landed_there():
    """Not an increment of zero: nothing was ever measured."""
    poller = _StubPoller([_sample(retract_total=_grouped(a=5.0))])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.note_phase("warmup", 0.0)
    rec.tick(0.0)

    assert rec.summary()["retract_delta"] is None


def test_counter_delta_adds_independent_engines():
    first = {'engine="0"': {'engine="0"': 0.0}, 'engine="1"': {'engine="1"': 0.0}}
    last = {'engine="0"': {'engine="0"': 48.0}, 'engine="1"': {'engine="1"': 48.0}}

    assert counter_delta(first, last) == 96.0


def test_rows_and_summary_use_the_same_counter_rule():
    """Two rules inside one artifact is worse than either being wrong: nothing
    on the page says which number was computed which way."""
    shards = {"": {f'tp_rank="{i}"': 48.0 for i in range(8)}}
    poller = _StubPoller([_sample(retract_total=shards)])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    assert rec.rows()[0]["retract_total"] == 48.0


def test_downsampling_respects_the_cap():
    """Integer division gave a stride of 1 for anything under twice the cap, so
    5001 rows downsampled to 5001."""
    rec = KvMetricsRecorder(poller=_StubPoller([]))
    rec._rows = [{"i": i} for i in range(5001)]

    assert len(rec.rows()) <= 5000


def test_prefix_cache_counters_are_bracketed_not_snapshotted():
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
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    window = rec.summary()["prefix_cache"]
    assert window["prefix_cache_queries_delta"] == 400.0
    assert window["prefix_cache_hits_delta"] == 300.0
    assert window["prefix_cache_queries_delta_by_phase"]["measured"] == 400.0


def test_prefix_cache_counters_add_across_independent_engines():
    """Two vLLM engines serving 200 lookups each did 400, not 200. Taking the
    max across series before diffing halved every cache figure on a DP
    deployment."""
    first = {'engine="0"': {'engine="0"': 0.0}, 'engine="1"': {'engine="1"': 0.0}}
    last = {'engine="0"': {'engine="0"': 200.0}, 'engine="1"': {'engine="1"': 200.0}}
    poller = _StubPoller([_sample(prefix_cache_queries=first), _sample(prefix_cache_queries=last)])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    assert rec.summary()["prefix_cache"]["prefix_cache_queries_delta"] == 400.0


def test_prefix_cache_restart_credits_only_the_post_restart_count():
    poller = _StubPoller(
        [_sample(cached_tokens_total=_grouped(a=900.0)), _sample(cached_tokens_total=_grouped(a=12.0))]
    )
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.note_phase("measured", 0.0)
    rec.tick(0.0)
    rec.tick(1.0)

    assert rec.summary()["prefix_cache"]["cached_tokens_total_delta"] == 12.0


def test_capacity_provenance_and_series_count_reach_the_artifact():
    poller = _StubPoller([_sample(capacity_tokens=32768.0, capacity_derived=True, series_count=8)])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)
    rec.tick(0.0)

    summary = rec.summary()
    assert summary["capacity_derived"] is True
    assert summary["series_count"] == 8


def test_summary_availability_is_tristate():
    """Never reached is not the same as reached and found quiet."""
    unknown = KvMetricsRecorder(poller=_StubPoller([], available=None)).summary()
    assert unknown["available"] is None

    off = KvMetricsRecorder(poller=_StubPoller([], available=False)).summary()
    assert off["available"] is False


def test_close_writes_the_artifact_and_is_idempotent(tmp_path):
    out = tmp_path / KV_ARTIFACT_NAME
    poller = _StubPoller([_sample(retract_total={"": {"": 48.0}})])
    rec = KvMetricsRecorder(poller=poller, output_path=str(out), min_interval_sec=0.0)
    rec.tick(0.0)

    rec.close()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["sample_count"] == 1
    assert payload["aborted"] is False

    rec.close(aborted=True)  # the loop's finally may run after an explicit close
    assert json.loads(out.read_text(encoding="utf-8"))["aborted"] is False


def test_close_never_raises_on_an_unwritable_path(tmp_path):
    rec = KvMetricsRecorder(poller=_StubPoller([]), output_path=str(tmp_path / "nope" / "x" / KV_ARTIFACT_NAME))
    # Parent dirs are created by the writer; make that impossible instead.
    (tmp_path / "nope").write_text("not a directory", encoding="utf-8")

    payload = rec.close()
    assert payload["schema_version"] == 1


def test_sampling_failure_does_not_propagate():
    class _Boom:
        url = "http://x/metrics"
        available = None

        def sample(self):
            raise RuntimeError("scrape exploded")

    rec = KvMetricsRecorder(poller=_Boom(), min_interval_sec=0.0)
    rec.tick(0.0)

    assert rec.rows() == []


# --------------------------------------------------------------------------
# loop integration
# --------------------------------------------------------------------------
class _Recorder:
    """Minimal stand-in matching the duck-typed contract the loop expects."""

    def __init__(self):
        self.phases: list[str] = []
        self.ticks = 0
        self.closed_aborted: bool | None = None

    def note_phase(self, phase, mono):
        self.phases.append(phase)

    def tick(self, mono):
        self.ticks += 1

    def close(self, *, aborted=False):
        self.closed_aborted = aborted


class _DoneProc:
    args = ["x"]

    def communicate(self, timeout=None):
        return ("out", "err")


class _HangingProc:
    args = ["x"]

    def communicate(self, timeout=None):
        raise sk.subprocess.TimeoutExpired(self.args, timeout)


def test_recorder_is_closed_on_the_normal_return_path():
    rec = _Recorder()
    sk._communicate_with_soft_deadline(
        _DoneProc(),
        hard_timeout=5,
        soft_deadline_sec=5,
        kv_recorder=rec,
    )

    assert rec.closed_aborted is False


def test_no_scraping_before_the_server_is_up():
    """The scrape blocks, and before the engine binds its port every attempt is
    a refused connection paid for inside the watchdog loop -- during boot, when
    the ready marker and the death gate most need it responsive. Boot has no
    traffic to measure anyway."""
    rec = _Recorder()
    sk._communicate_with_soft_deadline(
        _DoneProc(),
        hard_timeout=5,
        soft_deadline_sec=5,
        kv_recorder=rec,
    )

    assert rec.ticks == 0


def test_warm_reuse_rounds_scrape_immediately_and_are_tagged_measured():
    """A round re-attaching to a live server writes no ready marker.

    Gating on that marker alone collected nothing; gating only the *scrape* on
    it was worse -- samples were taken but left in ``boot``, the one phase that
    never enters a comparison, so the whole round aggregated to empty.
    """
    rec = _Recorder()
    sk._communicate_with_soft_deadline(
        _DoneProc(),
        hard_timeout=5,
        soft_deadline_sec=5,
        server_already_ready=True,
        kv_recorder=rec,
    )

    assert rec.ticks >= 1
    assert rec.phases == ["measured"]


def test_marker_split_across_a_read_boundary_is_still_seen(tmp_path):
    """Half a marker is in neither chunk, so without a carried tail the phase
    boundary is lost outright -- not seen late, never seen."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Phase profiling (prof", encoding="utf-8")

    residuals: dict[str, str] = {}
    first = sk._scan_server_log_increment(str(log_path), 0, residuals.get("k", ""))
    residuals["k"] = first.residual
    assert first.saw_measured_begin is False

    with log_path.open("a", encoding="utf-8") as f:
        f.write("iling) started | phase_index=0\n")
    second = sk._scan_server_log_increment(str(log_path), first.offset, residuals["k"])

    assert second.saw_measured_begin is True


def test_residual_is_held_per_path(tmp_path):
    """The resolved logs have different writers; splicing one's tail into
    another's next read would invent a line neither of them wrote."""
    bench = tmp_path / "benchmark_sglang"
    bench.mkdir()
    (bench / "server.log").write_text("Phase profiling (prof", encoding="utf-8")
    (bench / "benchmark_stderr.log").write_text("iling) started\n", encoding="utf-8")

    offsets: dict[str, int] = {}
    residuals: dict[str, str] = {}
    scan = sk._scan_logs_increment(str(tmp_path / "server.log"), offsets, residuals)

    assert scan.saw_measured_begin is False
    assert len(residuals) >= 2


def test_truncation_drops_the_carried_tail(tmp_path):
    """The tail belonged to a file that no longer exists."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Phase profiling (prof", encoding="utf-8")
    first = sk._scan_server_log_increment(str(log_path), 0)
    assert first.residual

    log_path.write_text("iling) started\n", encoding="utf-8")
    after = sk._scan_server_log_increment(str(log_path), 10_000, first.residual)

    assert after.saw_measured_begin is False


def test_scope_identifies_the_round_without_another_file(tmp_path):
    """A consumer must not have to join against something else to know which
    variant and action produced the artifact."""
    workspace = tmp_path / "runs" / "explore" / "task-abc" / "variant_03_fp8" / "benchmark_sglang"
    workspace.mkdir(parents=True)

    rec = sk._build_kv_recorder(str(workspace / "server.log"), {})

    assert rec._scope["workspace"] == "benchmark_sglang"
    assert rec._scope["run_path"] == "explore/task-abc/variant_03_fp8/benchmark_sglang"


def test_recorder_is_closed_as_aborted_when_a_gate_raises():
    """The failure a consumer cannot recover from is a window that never closed."""
    rec = _Recorder()
    with pytest.raises(sk.subprocess.TimeoutExpired):
        sk._communicate_with_soft_deadline(
            _HangingProc(),
            hard_timeout=0.01,
            soft_deadline_sec=5,
            kv_recorder=rec,
        )

    assert rec.closed_aborted is True


def test_phase_transitions_reach_the_recorder(tmp_path, monkeypatch):
    """Ready opens the measured window; AgentX's own lines refine it."""
    rec = _Recorder()
    scans = iter(
        [
            sk._LogScan(True, False, False, True, False, False, False),
            sk._LogScan(False, False, False, True, False, True, False),
            sk._LogScan(False, False, False, True, False, False, True),
            sk._LogScan(False, False, True, True, False, False, False),
        ]
    )
    monkeypatch.setattr(sk, "_scan_logs_increment", lambda *_a, **_k: next(scans, sk._LogScan(*([False] * 7))))
    monkeypatch.setattr(sk, "stamp_server_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(sk, "_server_log_shows_death", lambda *_a, **_k: None)
    log_path = tmp_path / "server.log"
    log_path.write_text("x\n", encoding="utf-8")

    class _SlowProc:
        args = ["x"]
        calls = 0

        def communicate(self, timeout=None):
            _SlowProc.calls += 1
            if _SlowProc.calls <= 4:
                raise sk.subprocess.TimeoutExpired(self.args, timeout)
            return ("out", "err")

    sk._communicate_with_soft_deadline(
        _SlowProc(),
        hard_timeout=60,
        soft_deadline_sec=60,
        server_log_path=str(log_path),
        server_dead_grace_sec=60,
        kv_recorder=rec,
    )

    assert rec.phases == ["measured", "warmup", "measured", "eval"]
    assert rec.closed_aborted is False


def test_no_recorder_leaves_the_loop_unchanged():
    """The default must be a strict no-op: this is a production watchdog."""
    out = sk._communicate_with_soft_deadline(
        _DoneProc(),
        hard_timeout=5,
        soft_deadline_sec=5,
    )
    assert out == ("out", "err")


def test_artifact_name_is_stable():
    assert Path(KV_ARTIFACT_NAME).suffix == ".json"


# --------------------------------------------------------------------------
# call-site wiring
# --------------------------------------------------------------------------
def test_no_recorder_without_a_server_log_path():
    """A helper subprocess has no engine to scrape and no round to scope to."""
    assert sk._build_kv_recorder(None, {}) is None
    assert sk._build_kv_recorder("", {}) is None


def test_kill_switch_disables_collection(tmp_path, monkeypatch):
    """A loop this central needs a way out that does not require a redeploy."""
    log_path = str(tmp_path / "server.log")
    assert sk._build_kv_recorder(log_path, {}) is not None

    monkeypatch.setenv(sk._KV_METRICS_ENV, "0")
    assert sk._build_kv_recorder(log_path, {}) is None


def test_recorder_targets_the_round_workspace_and_the_bound_port(tmp_path):
    """The port is the per-session ephemeral one the config pinned, not 8888."""
    rec = sk._build_kv_recorder(str(tmp_path / "server.log"), {"PORT": "31234"})

    assert rec._output_path == str(tmp_path / KV_ARTIFACT_NAME)
    assert ":31234/metrics" in rec._poller.url


def test_run_with_session_kill_produces_the_artifact(tmp_path):
    """End to end: the artifact has to exist on disk after a real round.

    The wiring this covers was the gap between "the loop accepts a recorder" and
    "a recorder is ever built" -- with it missing, every piece below had tests
    that passed while nothing was ever collected.
    """
    import sys

    log_path = tmp_path / "server.log"
    log_path.write_text("Application startup complete\n", encoding="utf-8")

    sk.run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(1.2)"],
        timeout=30,
        server_log_path=str(log_path),
        server_dead_grace_sec=30.0,
    )

    artifact = tmp_path / KV_ARTIFACT_NAME
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    # No engine was listening, and that is recorded rather than reported as an
    # idle pool: every metric stays absent instead of reading zero.
    assert payload["available"] is not True
    assert payload["capacity_tokens"] is None
    assert payload["retract_delta"] is None


def test_artifact_is_in_the_package_globs():
    """It lives in the round workspace, which the bundle does not otherwise reach."""
    from hyperloom.inference_optimizer.breakdown.session_package import PACKAGE_GLOBS

    assert "runs/**/kv_metrics.json" in PACKAGE_GLOBS


# ---------------------------------------------------------------------------
# aiperf's own server-metrics export
# ---------------------------------------------------------------------------
def _slim_record(*, ts_ns: int, usage: float, retracts: float, phase: str = "profiling") -> dict:
    """One line of aiperf's ``server_metrics_export.jsonl``, in its documented shape."""
    return {
        "endpoint_url": "http://localhost:34407/metrics",
        "timestamp_ns": ts_ns,
        "endpoint_latency_ns": 4_000_000,
        "request_sent_ns": ts_ns - 4_000_000,
        "benchmark_phase": phase,
        "metrics": {
            "sglang:token_usage": [{"labels": {"tp_rank": "0"}, "value": usage}],
            "sglang:num_retracted_requests_total": [{"labels": {"tp_rank": "0"}, "value": retracts}],
        },
    }


def _write_server_metrics(tmp_path, records) -> Path:
    """Write the export where aiperf would put it."""
    art = tmp_path / "aiperf_artifacts"
    art.mkdir(exist_ok=True)
    path = art / "server_metrics_export.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_aiperf_records_rebuild_the_same_normalised_sample(tmp_path):
    """Labels survive aiperf's export, so one normalisation serves both sources.

    ``SlimRecord.metrics`` is ``{name: [{labels, value}]}`` -- the same information
    ``parse_prometheus_text`` produces -- so the per-rank assembly does not have to
    be reimplemented for this path.
    """
    from hyperloom.orchestrator.actions.executors._kv_metrics import read_aiperf_server_metrics

    path = _write_server_metrics(tmp_path, [_slim_record(ts_ns=1_789_000_000_000_000_000, usage=0.62, retracts=4)])
    (sample, timing, phase) = read_aiperf_server_metrics(path)[0]

    assert sample.engine == "sglang"
    assert sample.active_pool_usage == 0.62
    assert phase == "measured"  # aiperf's "profiling"
    assert timing["scrape_sec"] == 0.004
    assert timing["scrape_start_unix"] <= sample.ts <= timing["scrape_end_unix"]


def test_aiperf_phase_stamp_is_used_verbatim(tmp_path):
    """No re-attribution: aiperf stamps the phase at collection time, from the
    process that owns the transition, so there is no boundary lag to correct."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import read_aiperf_server_metrics

    base = 1_789_000_000_000_000_000
    path = _write_server_metrics(
        tmp_path,
        [
            _slim_record(ts_ns=base, usage=0.1, retracts=0, phase="warmup"),
            _slim_record(ts_ns=base + 2_000_000_000, usage=0.5, retracts=3, phase="profiling"),
        ],
    )

    assert [phase for _s, _t, phase in read_aiperf_server_metrics(path)] == ["warmup", "measured"]


def test_recorder_prefers_aiperfs_scrapes_over_its_own(tmp_path):
    """On an AgentX round aiperf is already scraping the same endpoint; its rows
    win, and the live ones are discarded rather than interleaved."""
    base = 1_789_000_000_000_000_000
    _write_server_metrics(
        tmp_path,
        [
            _slim_record(ts_ns=base, usage=0.10, retracts=1, phase="warmup"),
            _slim_record(ts_ns=base + 2_000_000_000, usage=0.70, retracts=5, phase="profiling"),
            _slim_record(ts_ns=base + 4_000_000_000, usage=0.80, retracts=9, phase="profiling"),
        ],
    )
    # A live reading taken inside aiperf's window, which is the redundant case.
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(ts=base / 1e9 + 1.0, active_pool_usage=0.99)]),
        output_path=str(tmp_path / KV_ARTIFACT_NAME),
        min_interval_sec=0,
    )
    rec.tick(1.0)
    payload = rec.summary()

    assert payload["sample_source"] == "aiperf_server_metrics"
    assert payload["phase_source"] == "aiperf_server_metrics"
    assert [r["phase"] for r in payload["samples"]] == ["warmup", "measured", "measured"]
    # The live reading of 0.99 is gone: two collectors on one endpoint would bracket
    # counters across interleaved readings of the same series.
    assert [r["active_pool_usage"] for r in payload["samples"]] == [0.10, 0.70, 0.80]
    # Warmup opened at 1 and is closed by the first measured reading; measured runs 5 -> 9.
    assert payload["retract_delta_by_phase"]["warmup"] == 4.0
    assert payload["retract_delta"] == 4.0


def test_live_scraping_backs_off_once_aiperf_is_collecting(tmp_path):
    """The point of reading aiperf's export is to stop scraping the same endpoint twice.

    Adopting its records while still polling at the full rate doubles the load on
    the engine to produce rows that are then discarded -- worse on the very axis
    the refactor set out to improve.
    """
    (tmp_path / "aiperf_artifacts").mkdir()
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(), _sample(), _sample()]),
        workspace=tmp_path,
        min_interval_sec=2.0,
    )
    rec.tick(0.0)  # first tick always scrapes
    rec.tick(10.0)  # ten seconds on: would scrape at the 2s rate, must not here
    rec.tick(30.0)

    assert rec._poller.calls == 1


def test_backoff_still_leaves_a_coarse_trace(tmp_path):
    """Backing off rather than stopping: if aiperf collected nothing after all, a
    sparse series is a far better artifact than the empty one this shipped once."""
    (tmp_path / "aiperf_artifacts").mkdir()
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(), _sample()]),
        workspace=tmp_path,
        min_interval_sec=2.0,
    )
    rec.tick(0.0)
    rec.tick(120.0)  # past the suspended interval

    assert rec._poller.calls == 2


def test_a_synthetic_round_is_unaffected_by_the_backoff(tmp_path):
    """No aiperf directory, so nothing else is scraping and the full rate stands."""
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(), _sample(), _sample()]),
        workspace=tmp_path,
        min_interval_sec=2.0,
    )
    rec.tick(0.0)
    rec.tick(3.0)
    rec.tick(6.0)

    assert rec._poller.calls == 3


def test_eval_resumes_the_full_rate(tmp_path):
    """aiperf has exited by the accuracy eval, so that window is ours alone."""
    (tmp_path / "aiperf_artifacts").mkdir()
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample() for _ in range(5)]),
        workspace=tmp_path,
        min_interval_sec=2.0,
    )
    rec.tick(0.0)
    rec.note_phase("eval", 1.0)  # takes a boundary reading of its own
    before = rec._poller.calls
    rec.tick(10.0)
    rec.tick(20.0)

    assert rec._poller.calls == before + 2


def test_eval_rows_survive_the_aiperf_adoption(tmp_path):
    """aiperf covers warmup and profiling and nothing else, so discarding every
    live row would throw away the only readings the eval phase will ever have."""
    base = 1_789_000_000_000_000_000
    _write_server_metrics(
        tmp_path,
        [
            _slim_record(ts_ns=base, usage=0.10, retracts=1, phase="warmup"),
            _slim_record(ts_ns=base + 2_000_000_000, usage=0.70, retracts=5, phase="profiling"),
        ],
    )
    rec = KvMetricsRecorder(
        poller=_StubPoller(
            [
                # The boundary reading note_phase takes, still inside aiperf's window.
                _sample(ts=base / 1e9 + 1.0, active_pool_usage=0.99),
                # The eval reading proper, after aiperf has exited.
                _sample(ts=base / 1e9 + 60.0, active_pool_usage=0.33),
            ]
        ),
        workspace=tmp_path,
        output_path=str(tmp_path / KV_ARTIFACT_NAME),
        min_interval_sec=0,
    )
    rec.note_phase("eval", 1.0)
    rec.tick(2.0)
    payload = rec.summary()

    phases = [r["phase"] for r in payload["samples"]]
    assert phases == ["warmup", "measured", "eval"]
    # The eval reading is the live one, kept in time order after aiperf's window;
    # the redundant one taken inside that window is gone.
    assert payload["samples"][-1]["active_pool_usage"] == 0.33
    assert payload["sample_source"] == "aiperf_server_metrics"


def test_a_round_without_an_aiperf_export_keeps_the_watchdog_rows(tmp_path):
    """Synthetic benchmarks, the accuracy eval and any round killed before aiperf
    flushed have no export, and are still the majority of rounds."""
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(active_pool_usage=0.42)]),
        output_path=str(tmp_path / KV_ARTIFACT_NAME),
        min_interval_sec=0,
    )
    rec.tick(1.0)
    payload = rec.summary()

    assert payload["sample_source"] == "watchdog_scrape"
    assert payload["aiperf_server_metrics_path"] is None
    assert [r["active_pool_usage"] for r in payload["samples"]] == [0.42]


def test_an_unreadable_export_falls_back_rather_than_failing(tmp_path):
    """A better source that cannot be read is not a reason to lose the round."""
    art = tmp_path / "aiperf_artifacts"
    art.mkdir()
    (art / "server_metrics_export.jsonl").write_text("{not json\n", encoding="utf-8")
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(active_pool_usage=0.42)]),
        output_path=str(tmp_path / KV_ARTIFACT_NAME),
        min_interval_sec=0,
    )
    rec.tick(1.0)
    payload = rec.summary()

    assert payload["sample_source"] == "watchdog_scrape"
    assert payload["sample_count"] == 1


def test_histogram_samples_are_skipped_not_mistaken_for_gauges(tmp_path):
    """aiperf carries histograms as buckets with no ``value``."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import families_from_aiperf_record

    families = families_from_aiperf_record(
        {
            "sglang:token_usage": [{"labels": {}, "value": 0.5}],
            "sglang:e2e_latency_seconds": [{"labels": {}, "buckets": {"0.1": 3.0}, "sum": 1.0, "count": 3.0}],
        }
    )

    assert families == {"sglang:token_usage": [({}, 0.5)]}


# ---------------------------------------------------------------------------
# port resolution
# ---------------------------------------------------------------------------
_ROUND_YAML = """\
benchmark:
  framework: vllm
  model: /models/Qwen3-0.6B
  envs:
    TP: 1
    CONC: 64
    PORT: 34407
"""


def test_port_comes_from_the_round_config_not_the_default(tmp_path):
    """The regression that cost a real session its measured round.

    The port is pinned in the materialized ``benchmark.envs.PORT`` and never
    exported into the subprocess environment, so resolving from the environment
    alone lands on 8888 and scrapes a port nothing is listening on.
    """
    from hyperloom.orchestrator.actions.executors._kv_metrics import port_from_workspace, resolve_metrics_port

    (tmp_path / "baseline_lifecycle.yaml").write_text(_ROUND_YAML, encoding="utf-8")

    assert port_from_workspace(tmp_path) == 34407
    assert resolve_metrics_port({}, tmp_path) == 34407


def test_port_is_read_from_the_benchmark_config_too(tmp_path):
    """A warm-reuse round carries it here rather than in a lifecycle YAML."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import port_from_workspace

    nested = tmp_path / "benchmark_vllm_1"
    nested.mkdir()
    (nested / "config.yaml").write_text(_ROUND_YAML, encoding="utf-8")

    assert port_from_workspace(tmp_path) == 34407


def test_explicit_env_still_outranks_the_config(tmp_path):
    """An operator pin has to win over a file we merely found."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import resolve_metrics_port

    (tmp_path / "baseline_lifecycle.yaml").write_text(_ROUND_YAML, encoding="utf-8")

    assert resolve_metrics_port({"PORT": "9001"}, tmp_path) == 9001


def test_a_round_without_a_pinned_port_falls_back_to_the_default(tmp_path):
    """Which is what the one round that worked was relying on."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import DEFAULT_METRICS_PORT, resolve_metrics_port

    (tmp_path / "baseline_config.with_envs.yaml").write_text(
        "benchmark:\n  framework: vllm\n  envs:\n    TP: 1\n", encoding="utf-8"
    )

    assert resolve_metrics_port({}, tmp_path) == DEFAULT_METRICS_PORT


def test_unreadable_config_does_not_raise(tmp_path):
    """Malformed YAML in a round directory is not a reason to fail a round."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import port_from_workspace

    (tmp_path / "broken.yaml").write_text("benchmark: [unclosed\n", encoding="utf-8")

    assert port_from_workspace(tmp_path) is None


def _refuse(monkeypatch):
    """Make every scrape fail the way a closed port does."""
    from hyperloom.orchestrator.actions.executors import _kv_metrics as km

    def _boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(km._OPENER, "open", _boom)


def test_poller_does_not_give_up_on_a_slow_start(monkeypatch):
    """Three misses inside six seconds is a slow engine, not an absent endpoint.

    Giving up on the count alone parked collection for the rest of a round over a
    server that was merely still binding.
    """
    from hyperloom.orchestrator.actions.executors._kv_metrics import KvMetricsPoller

    _refuse(monkeypatch)
    poller = KvMetricsPoller(port=1)
    for _ in range(5):
        poller.fetch()

    assert poller.available is None  # still unknown, still trying


def test_poller_gives_up_once_the_grace_has_also_passed(monkeypatch):
    """An endpoint that answered nothing for a minute is not coming back."""
    import time

    from hyperloom.orchestrator.actions.executors._kv_metrics import _GIVE_UP_GRACE_SEC, KvMetricsPoller

    _refuse(monkeypatch)
    poller = KvMetricsPoller(port=1)
    poller.fetch()
    # Age the first attempt rather than the process clock: patching time.monotonic
    # globally lets any other caller consume the fake readings.
    poller._first_attempt_mono = time.monotonic() - (_GIVE_UP_GRACE_SEC + 1)
    poller.fetch()
    poller.fetch()

    assert poller.available is False


# ---------------------------------------------------------------------------
# authoritative phase boundaries
# ---------------------------------------------------------------------------
class _StubProgress:
    """Serves canned ``/api/progress`` phase payloads."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def poll(self):
        self.calls += 1
        return self._responses.pop(0) if self._responses else None


def _ns(unix_seconds: float) -> int:
    """Unix seconds as the nanosecond stamp aiperf reports."""
    return int(unix_seconds * 1e9)


def _now() -> float:
    """A base near the real wall clock, which the epoch sanity check requires."""
    import time

    return time.time()


def test_progress_address_is_read_from_the_round_directory(tmp_path):
    """The client publishes it; the watchdog is a different process."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import AiperfProgressPoller

    artifacts = tmp_path / "aiperf_artifacts"
    artifacts.mkdir()
    (artifacts / "progress_api.json").write_text(json.dumps({"url": "http://127.0.0.1:19090"}), encoding="utf-8")

    assert AiperfProgressPoller(tmp_path)._resolve_url() == "http://127.0.0.1:19090"


def test_progress_address_is_found_in_a_nested_benchmark_directory(tmp_path):
    """An AgentX round nests its artifacts one level below the server log."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import AiperfProgressPoller

    artifacts = tmp_path / "benchmark_agentx_1" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "progress_api.json").write_text(json.dumps({"url": "http://127.0.0.1:20001"}), encoding="utf-8")

    assert AiperfProgressPoller(tmp_path)._resolve_url() == "http://127.0.0.1:20001"


def test_missing_progress_address_is_not_an_error(tmp_path):
    """Every non-AgentX round is this case."""
    from hyperloom.orchestrator.actions.executors._kv_metrics import AiperfProgressPoller

    poller = AiperfProgressPoller(tmp_path)
    assert poller._resolve_url() is None
    assert poller.poll() is None


def test_authoritative_boundary_moves_rows_the_log_marker_mislabelled():
    """The correction this exists for.

    aiperf starts profiling at T, but writes the line the watchdog greps for
    afterwards, and the watchdog reads it on its next pass. Samples taken in
    that lag are stamped ``warmup`` while the measured phase is already running.
    """
    now = _now()
    progress = _StubProgress(
        [
            {"warmup": {"start_ns": _ns(now), "requests_end_ns": None}},
            {
                "warmup": {"start_ns": _ns(now), "requests_end_ns": _ns(now + 10)},
                "profiling": {"start_ns": _ns(now + 10), "requests_end_ns": None},
            },
            {"profiling": {"start_ns": _ns(now + 10), "requests_end_ns": None}},
        ]
    )
    rec = KvMetricsRecorder(
        poller=_StubPoller(
            [
                _sample(ts=now + 1),
                _sample(ts=now + 12),  # after the real boundary, before the marker was read
                _sample(ts=now + 20),
            ]
        ),
        progress=progress,
        min_interval_sec=0,
    )
    # No note_phase call: every row is stamped "boot", so the phases below come
    # entirely from the authoritative timeline.
    rec.tick(1.0)
    rec.tick(2.0)
    rec.tick(3.0)
    payload = rec.summary()

    assert payload["phase_source"] == "aiperf_progress_api"
    phases = [row["phase"] for row in payload["samples"]]
    assert phases[0] == "warmup"
    # Both later samples belong to the measured window, whatever the watchdog believed.
    assert phases[1] == "measured"
    assert phases[2] == "measured"
    assert payload["rows_reattributed"] == 3


def test_counter_totals_follow_the_authoritative_boundary():
    """Re-labelling the rows alone would leave the phase totals wrong.

    The retract that happened after the real transition has to be booked to the
    measured phase, not to the warmup window the watchdog had not yet closed.
    """
    now = _now()
    progress = _StubProgress(
        [
            {"warmup": {"start_ns": _ns(now), "requests_end_ns": None}},
            {
                "warmup": {"start_ns": _ns(now), "requests_end_ns": _ns(now + 10)},
                "profiling": {"start_ns": _ns(now + 10), "requests_end_ns": None},
            },
            {"profiling": {"start_ns": _ns(now + 10), "requests_end_ns": None}},
        ]
    )
    rec = KvMetricsRecorder(
        poller=_StubPoller(
            [
                _sample(ts=now + 1, retract_total=_grouped(a=1.0)),
                _sample(ts=now + 12, retract_total=_grouped(a=5.0)),
                _sample(ts=now + 20, retract_total=_grouped(a=9.0)),
            ]
        ),
        progress=progress,
        min_interval_sec=0,
    )
    rec.tick(1.0)
    rec.tick(2.0)
    rec.tick(3.0)
    payload = rec.summary()

    # Warmup opened at 1 and is closed by the first measured reading (5): 4.
    assert payload["retract_delta_by_phase"]["warmup"] == 4.0
    # Measured runs 5 -> 9, and shares the boundary reading rather than starting after it.
    assert payload["retract_delta"] == 4.0


def test_a_non_epoch_timestamp_is_rejected_rather_than_trusted():
    """A monotonic ``start_ns`` would place every boundary in 1970 and sweep the
    whole round into one phase. Fall back to the markers instead."""
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(ts=_now()), _sample(ts=_now())]),
        progress=_StubProgress(
            [
                {"profiling": {"start_ns": 12_345_678, "requests_end_ns": None}},
                {"profiling": {"start_ns": 12_345_678, "requests_end_ns": None}},
            ]
        ),
        min_interval_sec=0,
    )
    rec.note_phase("warmup", 1.0)  # consumes the boundary reading
    rec.tick(2.0)
    payload = rec.summary()

    assert payload["phase_source"] == "log_markers"
    assert payload["samples"][-1]["phase"] == "warmup"


def test_eval_is_never_overwritten_by_the_workload_timeline():
    """The accuracy run is not aiperf's traffic and it has no opinion about it."""
    now = _now()
    snapshot = {"profiling": {"start_ns": _ns(now), "requests_end_ns": None}}
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(ts=now + 30), _sample(ts=now + 31)]),
        progress=_StubProgress([snapshot, snapshot]),
        min_interval_sec=0,
    )
    rec.note_phase("eval", 1.0)  # consumes the boundary reading
    rec.tick(2.0)
    payload = rec.summary()

    assert payload["phase_source"] == "aiperf_progress_api"
    assert payload["samples"][-1]["phase"] == "eval"


def test_rows_carry_scrape_start_and_end_not_just_a_duration():
    """A gauge read across a 300ms window describes some instant inside it, and
    a single stamp would invite an alignment precision nobody measured."""
    rec = KvMetricsRecorder(poller=_StubPoller([_sample()]), min_interval_sec=0)
    rec.tick(1.0)
    row = rec.summary()["samples"][0]

    for key in ("scrape_start_unix", "scrape_end_unix", "scrape_start_mono", "scrape_end_mono", "scrape_sec"):
        assert key in row, key
    assert row["scrape_end_mono"] >= row["scrape_start_mono"]
    assert row["scrape_end_unix"] >= row["scrape_start_unix"]


def test_the_stored_scrape_window_contains_the_real_one():
    """Widened to the enclosing millisecond, never rounded to the nearest.

    These bounds are what workload records are joined against. Nearest-rounding
    shrinks the window by up to half a millisecond at each end, so a request that
    began inside a sub-millisecond scrape falls outside the window that observed
    it -- silently, and only on a host whose clock is fine enough to notice.
    """
    import time

    before = time.time()
    rec = KvMetricsRecorder(poller=_StubPoller([_sample()]), min_interval_sec=0)
    rec.tick(1.0)
    after = time.time()
    row = rec.summary()["samples"][0]

    assert row["scrape_start_unix"] <= before
    assert row["scrape_end_unix"] >= after
    # Still milliseconds, not full float noise.
    assert row["scrape_start_unix"] == round(row["scrape_start_unix"], 3)
    assert row["scrape_end_unix"] == round(row["scrape_end_unix"], 3)


def test_a_request_starting_at_the_scrape_instant_is_still_correlated(tmp_path):
    """The regression: a sub-millisecond scrape must not lose the work it saw."""
    from hyperloom.orchestrator.actions.executors._agentx_timeline import (
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


def test_rows_carry_the_workload_in_flight_at_the_scrape():
    """The correlation the timeline supports: engine-wide KV against the active
    phase's own counters, at the granularity aiperf actually exposes."""
    now = _now()
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(ts=now + 1)]),
        progress=_StubProgress(
            [{"profiling": {"start_ns": _ns(now), "requests_end_ns": None, "sent": 12, "completed": 7}}]
        ),
        min_interval_sec=0,
    )
    rec.tick(1.0)
    row = rec.summary()["samples"][0]

    assert row["workload"]["aiperf_phase"] == "profiling"
    assert row["workload"]["stats"] == {"sent": 12, "completed": 7}


def test_timeline_survives_a_phase_aiperf_stops_reporting():
    """aiperf drops a finished phase from its report; keeping only the latest
    response would lose the warmup boundary the moment profiling starts."""
    now = _now()
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample(ts=now + 1), _sample(ts=now + 12)]),
        progress=_StubProgress(
            [
                {"warmup": {"start_ns": _ns(now), "requests_end_ns": None}},
                {"profiling": {"start_ns": _ns(now + 10), "requests_end_ns": None}},
            ]
        ),
        min_interval_sec=0,
    )
    rec.tick(1.0)
    rec.tick(2.0)
    payload = rec.summary()

    assert set(payload["phase_timeline"]) == {"warmup", "profiling"}
    assert payload["phase_bounds_unix"]["warmup"]["start"] == pytest.approx(now, abs=1e-3)


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
        progress=_StubProgress([{"profiling": {"start_ns": start_ns, "requests_end_ns": None}}]),
        output_path=str(tmp_path / KV_ARTIFACT_NAME),
        min_interval_sec=0,
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
    assert {e["event"] for e in events} >= {"phase_start", "trajectory_start", "turn_start", "request_start"}

    # The sample taken 5s in saw that trajectory running.
    assert payload["samples"][0]["workload"]["in_flight"]["trajectory_ids"] == ["traj-1"]


def test_a_round_without_an_aiperf_export_reports_no_timeline(tmp_path):
    """Every synthetic benchmark is this case, and it is not a failure."""
    rec = KvMetricsRecorder(
        poller=_StubPoller([_sample()]),
        output_path=str(tmp_path / KV_ARTIFACT_NAME),
        min_interval_sec=0,
    )
    rec.tick(1.0)
    payload = rec.close()

    assert payload["workload_timeline"] is None
    assert not (tmp_path / "agentx_timeline.jsonl").exists()


def test_progress_failures_never_reach_the_round():
    """Collection is observational; a broken endpoint costs nothing."""

    class _Exploding:
        def poll(self):
            raise RuntimeError("progress API on fire")

    rec = KvMetricsRecorder(poller=_StubPoller([_sample()]), progress=_Exploding(), min_interval_sec=0)
    rec.tick(1.0)
    payload = rec.summary()

    assert payload["phase_source"] == "log_markers"
    assert payload["sample_count"] == 1
