# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the bounded torch.profiler window in the HY-WorldPlay bench.

The roofline leg had never once succeeded, across four sessions and two
container images, because the profiled capture was unbounded: it recorded a
whole generation, every AR chunk and every denoise step, with Python stacks on.

That single defect surfaced as two unrelated-looking failures. On ROCm, kineto
collects roctracer records into one buffer shared by HIP API calls and GPU
kernel ops, capped at ``Config::maxEvents_``; a generation issues roughly eight
times that cap, so the API calls exhausted the budget and every GPU op record
was dropped — traces came out holding exactly 1,000,000 ``cuda_runtime`` events
and zero ``kernel`` events, which TraceLens rejects outright. On the earlier
build, before that cap existed, the same capture instead produced a 2.7 GB trace
that took TraceLens 287 GB of RSS and over an hour, and the leg timed out.

So these tests pin the property that fixes both: what gets recorded is bounded
by construction and does not grow with the workload. A capture that scales with
chunk count or step count is the bug, whatever it happens to break that week.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_BENCH_PATH = (
    Path(__file__).parents[1] / "assets" / "benchmark_scripts" / "bench_fps.py"
)


@pytest.fixture(scope="module")
def bench():
    """The loaded ``bench_fps`` module.

    It lives under ``assets/`` and is run by torchrun, so it is not importable by
    package path.
    """
    spec = importlib.util.spec_from_file_location("bench_fps", _BENCH_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_device_sync(monkeypatch):
    """The window syncs before stopping; there is no device in a unit test."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


class _FakeBar:
    """Stands in for the tqdm the pipeline opens once per AR chunk."""

    def __init__(self):
        self.updates = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def update(self, *a, **k):
        self.updates += 1


class _FakePipe:
    def __init__(self):
        self.bars = []

    def progress_bar(self, *a, **k):
        bar = _FakeBar()
        self.bars.append(bar)
        return bar


class _FakeProfiler:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1


def _generate(pipe, chunks: int, steps: int) -> None:
    """Drive the hook the way ``ar_rollout`` does: one bar per chunk, one
    ``update()`` per denoise step."""
    for _ in range(chunks):
        with pipe.progress_bar(total=steps) as bar:
            for _ in range(steps):
                bar.update()


def _window(bench, pipe, prof, *, skip=1, steps=6):
    return bench._ProfileWindow(pipe, prof, skip, steps)


# --------------------------------------------------------------------------
# the capture is bounded by construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "steps"),
    [(8, 12), (8, 50), (32, 50)],
)
def test_recorded_steps_do_not_grow_with_the_workload(bench, chunks, steps):
    """The whole defect, stated directly.

    The unbounded capture recorded ``chunks * steps`` denoise steps, so a longer
    video or a finer schedule silently enlarged the trace until it blew the
    roctracer budget or the TraceLens memory ceiling. The window must record the
    same bounded amount of work no matter how big the generation is.
    """
    pipe, prof = _FakePipe(), _FakeProfiler()
    with _window(bench, pipe, prof, steps=6) as window:
        _generate(pipe, chunks=chunks, steps=steps)

    assert window.recorded_steps == 6
    assert window.recorded_steps < chunks * steps


def test_window_skips_the_warm_up_chunk(bench):
    """Chunk 0 runs a different path, so recording it would misrepresent the
    steady state the roofline is meant to describe."""
    pipe, prof = _FakePipe(), _FakeProfiler()
    with _window(bench, pipe, prof, skip=1, steps=6) as window:
        assert not window.started
        with pipe.progress_bar(total=12) as bar:  # chunk 0
            for _ in range(12):
                bar.update()
        assert not window.started, "profiler opened during the warm-up chunk"
        with pipe.progress_bar(total=12) as bar:  # chunk 1
            assert window.started
            for _ in range(12):
                bar.update()

    assert window.recorded_steps == 6


def test_profiler_is_started_and_stopped_exactly_once(bench):
    """A generation continues long after the window closes; the profiler must
    not be restarted by later chunks, nor stopped twice on exit."""
    pipe, prof = _FakePipe(), _FakeProfiler()
    with _window(bench, pipe, prof, steps=6):
        _generate(pipe, chunks=8, steps=12)

    assert (prof.starts, prof.stops) == (1, 1)


def test_short_generation_still_stops_the_profiler(bench):
    """If the generation ends before the window fills, exiting must close it —
    otherwise the export would run against a profiler that is still recording."""
    pipe, prof = _FakePipe(), _FakeProfiler()
    with _window(bench, pipe, prof, skip=1, steps=6) as window:
        _generate(pipe, chunks=2, steps=2)

    assert window.started
    assert window.recorded_steps == 2
    assert prof.stops == 1


def test_window_never_opens_on_a_single_chunk_generation(bench):
    """Fewer chunks than the skip count leaves nothing to record. The caller
    reports that rather than exporting an empty trace, so the profiler must
    stay shut and must not be stopped."""
    pipe, prof = _FakePipe(), _FakeProfiler()
    with _window(bench, pipe, prof, skip=1, steps=6) as window:
        _generate(pipe, chunks=1, steps=12)

    assert not window.started
    assert (prof.starts, prof.stops) == (0, 0)


# --------------------------------------------------------------------------
# the hook leaves the pipeline as it found it
# --------------------------------------------------------------------------


def test_progress_bar_is_restored_and_still_drives_the_pipeline(bench):
    """The bench reuses the pipeline after the capture, and the pipeline relies
    on the bar it was handed, so the wrapper must forward faithfully and then
    get out of the way."""
    pipe, prof = _FakePipe(), _FakeProfiler()
    original = pipe.progress_bar

    with _window(bench, pipe, prof, steps=6):
        assert pipe.progress_bar != original
        _generate(pipe, chunks=3, steps=4)

    assert pipe.progress_bar == original
    assert [bar.updates for bar in pipe.bars] == [4, 4, 4]
    assert all(bar.closed for bar in pipe.bars)


def test_defaults_stay_inside_the_roctracer_record_budget(bench):
    """kineto's shared roctracer buffer holds 1,000,000 records on this build,
    and a denoise step cost about 84k of them when measured on the build before
    the cap existed. Keep the default window well under that: the failure it
    causes is silent, and the sessions it costs are 24 hours each.
    """
    measured_records_per_step = 84_000
    budget = 1_000_000

    assert bench._PROFILE_WINDOW_SKIP_CHUNKS >= 1
    assert bench._PROFILE_WINDOW_STEPS >= 1
    projected = bench._PROFILE_WINDOW_STEPS * measured_records_per_step
    assert projected <= budget // 2, (
        f"default window projects to {projected} roctracer records, which "
        f"leaves no margin under the {budget} cap"
    )
