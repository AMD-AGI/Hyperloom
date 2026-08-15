# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the two things measured from the server-ready marker.

When a ``server.log`` is available the explore overtime soft deadline measures
only the post-ready phase: the clock starts at the server-ready marker,
excluding pre-ready boot / weight load / first-request recompile. Opt out via
``INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY=0``.

The same marker is recorded so a round can be *priced* by its two parts rather
than its total, which is what lets later work be charged for what it will
actually spend: a variant boots its own server and pays both parts, a pass that
re-attaches pays only the second.
"""

from __future__ import annotations

import sys
import time

from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    OVERTIME_KILL_RETURNCODE,
    clear_server_ready_stamp,
    post_ready_runtime_sec,
    run_with_session_kill,
    server_ready_unix,
)

# Long stall grace so the detok-stall watchdog never interferes here.
_LONG_STALL_GRACE = 3600.0


def test_from_ready_excludes_pre_ready_phase(tmp_path):
    """Pre-ready time is NOT counted: a child that spends > deadline BEFORE the
    ready marker but only a little AFTER it finishes normally."""
    log_path = tmp_path / "server.log"
    # 3s pre-ready boot, then ready, then ~1s post-ready client.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('INFO loading weights\\n'); f.flush()\n"
        "time.sleep(3)\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "time.sleep(1)\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        soft_deadline_sec=2.0,
        server_log_path=str(log_path),
        detok_stall_grace_sec=_LONG_STALL_GRACE,
    )
    elapsed = time.monotonic() - start
    # Post-ready (~1s) < 2.0s deadline -> not killed despite ~4s wall-clock.
    assert cp.returncode == 0, f"unexpected returncode={cp.returncode}"
    assert elapsed >= 3.0, f"child should have run its full pre-ready sleep, got {elapsed:.2f}s"


def test_from_ready_fires_after_ready(tmp_path):
    """Post-ready overrun IS killed: once ready, exceeding the deadline in the
    client phase reaps the tree with the overtime sentinel."""
    log_path = tmp_path / "server.log"
    # Ready immediately, then a post-ready run that overruns the deadline.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "time.sleep(30)\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        soft_deadline_sec=1.0,
        server_log_path=str(log_path),
        detok_stall_grace_sec=_LONG_STALL_GRACE,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    # Killed shortly after ready, not at 30s.
    assert elapsed < 15.0, f"from-ready soft deadline took {elapsed:.2f}s"


def test_opt_out_reverts_to_from_spawn(tmp_path, monkeypatch):
    """With INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY=0 the legacy from-spawn
    clock applies even with a server.log: pre-ready time counts and trips."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY", "0")
    log_path = tmp_path / "server.log"
    # Long pre-ready phase; from-spawn overruns the 1s deadline.
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('INFO loading weights\\n'); f.flush()\n"
        "time.sleep(30)\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "raise SystemExit(0)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        soft_deadline_sec=1.0,
        server_log_path=str(log_path),
        detok_stall_grace_sec=_LONG_STALL_GRACE,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 15.0, f"from-spawn opt-out took {elapsed:.2f}s"


def test_no_server_log_uses_from_spawn(tmp_path):
    """Without a server.log the soft deadline is the legacy from-spawn clock."""
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=60,
        soft_deadline_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 15.0, f"from-spawn (no server.log) took {elapsed:.2f}s"


class TestARoundIsPricedByItsTwoParts:
    """What a round spent booting and what it spent benchmarking, told apart.

    The whole point of separating them is that they are spent by different
    things. Charging a re-attaching pass for a boot it never pays is what makes
    a budget gate refuse work that fits.
    """

    def test_the_boot_is_not_charged_to_the_benchmark(self, tmp_path):
        """A round that boots for 3s and benchmarks for 1s reports 1s, not 4s."""
        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(3)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        started_unix = time.time()
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            timeout=30,
            server_log_path=str(log_path),
            detok_stall_grace_sec=_LONG_STALL_GRACE,
        )
        runtime_sec = time.time() - started_unix
        assert cp.returncode == 0

        post_ready = post_ready_runtime_sec(
            str(log_path),
            started_unix=started_unix,
            runtime_sec=runtime_sec,
        )
        assert post_ready is not None, "the round reported ready but nothing recorded when"
        # The benchmark's own second, found without the three the boot took. The
        # windows are wide because the poll interval and process spawn are inside
        # them; what is being pinned is that the two parts are told apart at all.
        assert 0.5 <= post_ready <= 2.5, f"benchmark share read as {post_ready:.2f}s, expected ~1s"
        boot_sec = runtime_sec - post_ready
        assert 2.5 <= boot_sec <= 4.5, f"boot share read as {boot_sec:.2f}s, expected ~3s"

    def test_a_previous_attempts_log_does_not_time_this_rounds_boot(self, tmp_path):
        """Only bytes this round writes may say when its server came up.

        Magpie writes into a ``benchmark_*/`` workspace under the round's output
        dir, and a reused dir can still hold one from an earlier attempt. Scanned
        from byte zero, its ready line latches on the first poll -- seconds after
        spawn -- and the round reports a boot that took minutes as one that took
        none. Every variant is then admitted at a benchmark's price and reaped.
        """
        stale = tmp_path / "benchmark_vllm_20200101" / "server.log"
        stale.parent.mkdir(parents=True)
        stale.write_text("Application startup complete\n", encoding="utf-8")

        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(3)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        started_unix = time.time()
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            timeout=30,
            server_log_path=str(log_path),
            detok_stall_grace_sec=_LONG_STALL_GRACE,
        )
        runtime_sec = time.time() - started_unix
        assert cp.returncode == 0

        post_ready = post_ready_runtime_sec(
            str(log_path),
            started_unix=started_unix,
            runtime_sec=runtime_sec,
        )
        assert post_ready is not None
        boot_sec = runtime_sec - post_ready
        assert boot_sec >= 2.5, (
            f"the stale log timed the boot: read {boot_sec:.2f}s of boot for a round "
            f"that spent 3s on it"
        )

    def test_the_split_does_not_depend_on_an_unrelated_watchdog(self, tmp_path):
        """The stall grace is a hang backstop, not a switch for the cost model.

        Turning it off used to withdraw the ready timestamp with it, and a session
        run that way prices every round at its whole cold wall-clock without
        anything saying so.
        """
        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(2)\n"
            "f.write('Application startup complete\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(0)\n"
        )
        started_unix = time.time()
        cp = run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            timeout=30,
            server_log_path=str(log_path),
            detok_stall_grace_sec=0.0,
            session_deadline_sec=time.monotonic() + 30.0,
        )
        assert cp.returncode == 0
        assert server_ready_unix(str(log_path)) is not None, (
            "the boot/benchmark split was withdrawn along with the stall watchdog"
        )

    def test_a_round_that_never_came_up_is_not_priced(self, tmp_path):
        """No ready marker means no split, reported as unknown rather than guessed."""
        log_path = tmp_path / "server.log"
        script = (
            "import sys, time\n"
            "f = open(sys.argv[1], 'w')\n"
            "f.write('INFO loading weights\\n'); f.flush()\n"
            "time.sleep(1)\n"
            "raise SystemExit(1)\n"
        )
        started_unix = time.time()
        run_with_session_kill(
            [sys.executable, "-c", script, str(log_path)],
            timeout=30,
            server_log_path=str(log_path),
            detok_stall_grace_sec=_LONG_STALL_GRACE,
        )
        assert server_ready_unix(str(log_path)) is None
        assert (
            post_ready_runtime_sec(
                str(log_path),
                started_unix=started_unix,
                runtime_sec=time.time() - started_unix,
            )
            is None
        )

    def test_an_earlier_rounds_stamp_is_not_read_as_this_ones(self, tmp_path):
        """A stamp predating the round is unknown, not "it never booted".

        The clamp alone would report such a stamp as a whole-round benchmark,
        which is the reading that would price a cold round as a warm one.
        """
        log_path = tmp_path / "server.log"
        log_path.write_text("INFO loading weights\n", encoding="utf-8")
        (tmp_path / "server_ready_at").write_text(f"{time.time() - 600.0:.3f}\n", encoding="utf-8")

        assert (
            post_ready_runtime_sec(
                str(log_path),
                started_unix=time.time(),
                runtime_sec=90.0,
            )
            is None
        )

    def test_a_stamp_is_cleared_so_the_next_round_starts_blind(self, tmp_path):
        """Clearing is what keeps the case above from arising in a reused dir."""
        log_path = tmp_path / "server.log"
        stamp = tmp_path / "server_ready_at"
        stamp.write_text(f"{time.time():.3f}\n", encoding="utf-8")
        assert server_ready_unix(str(log_path)) is not None

        clear_server_ready_stamp(str(log_path))

        assert not stamp.exists()
        assert server_ready_unix(str(log_path)) is None
        # Idempotent: a round whose dir never had one must not fail to start.
        clear_server_ready_stamp(str(log_path))

    def test_a_clock_that_disagrees_cannot_produce_a_negative_price(self, tmp_path):
        """The writer and the reader can be different hosts, so skew is possible.

        A stamp after the round's own end would price the benchmark at less than
        nothing; it is floored instead, and the round's total caps the other end.
        """
        log_path = tmp_path / "server.log"
        started_unix = time.time()
        (tmp_path / "server_ready_at").write_text(f"{started_unix + 500.0:.3f}\n", encoding="utf-8")

        priced = post_ready_runtime_sec(
            str(log_path),
            started_unix=started_unix,
            runtime_sec=100.0,
        )

        assert priced == 0.0
