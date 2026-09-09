# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for KV collection riding the watchdog loop.

Covers the phase split, the recorder itself, and the ``finally`` that closes a
window on every exit path. Also covers the scanner's truncation branch, which
has been reachable and untested since it was written.
"""

from __future__ import annotations

import json
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


# â”€â”€ scanner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


# â”€â”€ recorder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


# â”€â”€ loop integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    monkeypatch.setattr(sk, "_stamp_server_ready", lambda *_a, **_k: None)
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


# â”€â”€ call-site wiring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
