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


# ── scanner ────────────────────────────────────────────────────────────────
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


# ── recorder ───────────────────────────────────────────────────────────────
def test_rows_are_tagged_with_the_phase_they_were_taken_in():
    poller = _StubPoller([_sample(), _sample(), _sample()])
    rec = KvMetricsRecorder(poller=poller, min_interval_sec=0.0)

    rec.tick(0.0)
    rec.note_phase("warmup", 1.0)
    rec.tick(1.0)
    rec.note_phase("measured", 2.0)
    rec.tick(2.0)

    assert [r["phase"] for r in rec.rows()] == ["boot", "warmup", "measured"]


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
    rec.tick(0.0)
    rec.tick(1.0)

    summary = rec.summary()
    assert summary["capacity_tokens"] == 32768.0
    assert summary["capacity_gb"] == 180.0


def test_counter_delta_credits_only_the_post_restart_count():
    """An engine restart zeroes its series; a flat subtraction would go negative."""
    assert counter_delta({"a": 10.0}, {"a": 48.0}) == 38.0
    assert counter_delta({"a": 100.0}, {"a": 3.0}) == 3.0
    assert counter_delta({}, {}) is None
    assert counter_delta({}, {"a": 5.0}) == 5.0


def test_summary_availability_is_tristate():
    """Never reached is not the same as reached and found quiet."""
    unknown = KvMetricsRecorder(poller=_StubPoller([], available=None)).summary()
    assert unknown["available"] is None

    off = KvMetricsRecorder(poller=_StubPoller([], available=False)).summary()
    assert off["available"] is False


def test_close_writes_the_artifact_and_is_idempotent(tmp_path):
    out = tmp_path / KV_ARTIFACT_NAME
    poller = _StubPoller([_sample(retract_total={"": 48.0})])
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

    assert rec.close()["schema_version"] == 1


def test_sampling_failure_does_not_propagate():
    class _Boom:
        url = "http://x/metrics"
        available = None

        def sample(self):
            raise RuntimeError("scrape exploded")

    rec = KvMetricsRecorder(poller=_Boom(), min_interval_sec=0.0)
    rec.tick(0.0)

    assert rec.rows() == []


# ── loop integration ───────────────────────────────────────────────────────
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


def test_warm_reuse_rounds_scrape_immediately():
    """A round re-attaching to a live server writes no ready marker, so gating on
    that marker alone would collect nothing for the entire round."""
    rec = _Recorder()
    sk._communicate_with_soft_deadline(
        _DoneProc(),
        hard_timeout=5,
        soft_deadline_sec=5,
        server_already_ready=True,
        kv_recorder=rec,
    )

    assert rec.ticks >= 1


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


# ── call-site wiring ───────────────────────────────────────────────────────
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
