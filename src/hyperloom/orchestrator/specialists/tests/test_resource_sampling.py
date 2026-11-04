# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resource sampling against a fake /proc and a fake cgroup directory."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from hyperloom.common import proctree
from hyperloom.orchestrator.specialists import resources
from hyperloom.orchestrator.specialists.resources import (
    ProcessTreeSampler,
    ResourceLog,
    dir_usage_bytes,
    read_container_cgroup,
)


def add_proc(
    root: Path,
    pid: int,
    ppid: int,
    *,
    rss_kb: int,
    utime: int,
    stime: int = 0,
    cutime: int = 0,
    cstime: int = 0,
    start_time: int | None = None,
) -> None:
    pdir = root / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "status").write_text(f"Name:\tx\nVmRSS:\t{rss_kb} kB\n", encoding="utf-8")
    started = pid if start_time is None else start_time
    # comm contains a space and a parenthesis on purpose: fields are counted after the LAST ')'.
    stat = f"{pid} (a b) c) S {ppid} {pid} {pid} 0 0 0 0 0 0 0 {utime} {stime} {cutime} {cstime} 20 0 1 0 {started}"
    (pdir / "stat").write_text(stat, encoding="utf-8")


def remove_proc(root: Path, pid: int) -> None:
    shutil.rmtree(root / str(pid))


@pytest.fixture
def proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``/proc`` that ``proctree.descendants`` walks and the sampler reads."""
    root = tmp_path / "proc"
    root.mkdir()

    def identity(pid: int) -> tuple[int, int] | None:
        try:
            fields = (root / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[-1].split()
        except OSError:
            return None
        return int(fields[1]), int(fields[19])

    monkeypatch.setattr(proctree, "_PROC", root)
    monkeypatch.setattr(proctree, "proc_identity", identity)
    return root


def sampler_of(proc: Path, root_pid: int = 100) -> ProcessTreeSampler:
    return ProcessTreeSampler(root_pid, proc_root=proc, clock_ticks=100)


def test_peak_memory_and_cpu_seconds_over_the_tree(proc: Path) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=200)
    add_proc(proc, 101, 100, rss_kb=3000, utime=100, stime=100)
    add_proc(proc, 102, 100, rss_kb=2000, utime=0)
    add_proc(proc, 103, 101, rss_kb=500, utime=50, stime=50)  # a grandchild
    add_proc(proc, 999, 1, rss_kb=777777, utime=9999)  # unrelated process
    sampler = sampler_of(proc)
    first = sampler.sample()
    assert first.rss_bytes == (1000 + 3000 + 2000 + 500) * 1024
    assert first.live_pids == 4
    assert sampler.cpu_seconds == 5.0  # (200 + 200 + 0 + 100) ticks / 100
    assert sampler.peak_rss_bytes == 6500 * 1024


def test_exited_child_stops_counting_memory_and_its_cpu_arrives_through_the_parent(proc: Path) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=100)
    add_proc(proc, 101, 100, rss_kb=5000, utime=100)
    sampler = sampler_of(proc)
    assert sampler.sample().cpu_seconds == 2.0
    remove_proc(proc, 101)  # it ran 3.0 s in all, 2.0 of them after the first sample, and the parent reaped it
    add_proc(proc, 100, 1, rss_kb=1500, utime=150, cutime=200, cstime=100)
    after = sampler.sample()
    assert after.rss_bytes == 1500 * 1024  # the dead child is no longer counted
    assert after.live_pids == 1
    assert sampler.peak_rss_bytes == 6000 * 1024  # the peak keeps the earlier reading
    assert sampler.cpu_seconds == 1.5 + 3.0


def test_a_child_that_starts_and_exits_between_two_samples_counts(proc: Path) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=100)
    sampler = sampler_of(proc)
    sampler.sample()
    add_proc(proc, 100, 1, rss_kb=1000, utime=150, cutime=300)  # a 3.0 s compiler came and went
    assert sampler.sample().cpu_seconds == 1.5 + 3.0


def test_a_reused_pid_does_not_hide_either_process(proc: Path) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=0)
    add_proc(proc, 101, 100, rss_kb=1000, utime=500)
    sampler = sampler_of(proc)
    sampler.sample()
    add_proc(proc, 100, 1, rss_kb=1000, utime=0, cutime=500)  # reaped the first 101 ...
    add_proc(proc, 101, 100, rss_kb=1000, utime=200)  # ... and started another that got its pid
    assert sampler.sample().cpu_seconds == 5.0 + 2.0


def test_a_pid_reused_between_the_scan_and_the_read_is_not_counted(proc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=100)
    add_proc(proc, 101, 100, rss_kb=1000, utime=100)
    scan = resources.descendants

    def scan_then_reuse(root_pid: int) -> list[tuple[int, int]]:
        found = scan(root_pid)
        add_proc(proc, 100, 1, rss_kb=1000, utime=100, cutime=100)  # 100 reaped 101 right after the scan ...
        add_proc(proc, 101, 1, rss_kb=9000, utime=5000, start_time=7777)  # ... and an unrelated process got 101
        return found

    monkeypatch.setattr(resources, "descendants", scan_then_reuse)
    sample = sampler_of(proc).sample()
    assert sample.cpu_seconds == 2.0  # 101's own second, through 100's cutime; none of the newcomer's 50
    assert sample.rss_bytes == 1000 * 1024
    assert sample.live_pids == 1


def test_the_cpu_total_never_goes_back_when_a_process_leaves_the_tree_unreaped(proc: Path) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=100)
    add_proc(proc, 101, 100, rss_kb=1000, utime=300)
    sampler = sampler_of(proc)
    sampler.sample()
    add_proc(proc, 101, 1, rss_kb=1000, utime=300)  # detached: re-parented, never waited for by the tree
    assert sampler.sample().cpu_seconds == 4.0


def test_a_child_reaped_in_the_middle_of_a_sample_is_not_counted_twice(
    proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_proc(proc, 100, 1, rss_kb=1000, utime=100)
    add_proc(proc, 101, 100, rss_kb=1000, utime=100)
    add_proc(proc, 102, 101, rss_kb=1000, utime=300)
    read = ProcessTreeSampler._read_process
    reaped: list[int] = []

    def read_then_reap(self: ProcessTreeSampler, pid: int, start_time: int | None) -> tuple[int, float] | None:
        reading = read(self, pid, start_time)
        if pid in (101, 102) and not reaped:  # 101 waits for 102 right after the first of the two is read
            reaped.append(pid)
            remove_proc(proc, 102)
            add_proc(proc, 101, 100, rss_kb=1000, utime=100, cutime=300)
        return reading

    monkeypatch.setattr(ProcessTreeSampler, "_read_process", read_then_reap)
    sampler = sampler_of(proc)
    sampler.sample()
    sampler.sample()
    assert sampler.cpu_seconds == 5.0  # 102 counted once, through 101, never both as itself and in 101's cutime


def test_vanished_root_reads_as_zero_not_an_error(proc: Path) -> None:
    sampler = sampler_of(proc, 4242)
    sample = sampler.sample()
    assert (sample.rss_bytes, sample.live_pids, sampler.peak_rss_bytes) == (0, 0, 0)


def write_cgroup(root: Path, *, memory_max: str, peak: str | None, oom: int | None, cpu_max: str | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cgroup.controllers").write_text("cpu memory\n", encoding="utf-8")
    (root / "memory.max").write_text(memory_max + "\n", encoding="utf-8")
    if peak is not None:
        (root / "memory.peak").write_text(peak + "\n", encoding="utf-8")
    if oom is not None:
        (root / "memory.events").write_text(f"low 0\nhigh 0\nmax 3\noom 1\noom_kill {oom}\n", encoding="utf-8")
    if cpu_max is not None:
        (root / "cpu.max").write_text(cpu_max + "\n", encoding="utf-8")


def test_cgroup_v2_values(tmp_path: Path) -> None:
    write_cgroup(tmp_path, memory_max="8589934592", peak="1073741824", oom=2, cpu_max="200000 100000")
    snap = read_container_cgroup(tmp_path)
    assert snap.memory_max_bytes == 8589934592
    assert snap.memory_peak_bytes == 1073741824
    assert snap.oom_kill == 2
    assert (snap.cpu_quota_us, snap.cpu_period_us) == (200000, 100000)


def test_cgroup_unlimited_and_missing_files_are_null(tmp_path: Path) -> None:
    write_cgroup(tmp_path, memory_max="max", peak=None, oom=None, cpu_max="max 100000")
    snap = read_container_cgroup(tmp_path)
    assert snap.memory_max_bytes is None  # "max" is unlimited, not a number
    assert snap.memory_peak_bytes is None  # kernels before 5.19 have no memory.peak
    assert snap.oom_kill is None
    assert (snap.cpu_quota_us, snap.cpu_period_us) == (None, 100000)


def test_cgroup_without_cpu_max_leaves_both_cpu_fields_null(tmp_path: Path) -> None:
    write_cgroup(tmp_path, memory_max="8589934592", peak=None, oom=0, cpu_max=None)
    snap = read_container_cgroup(tmp_path)
    assert (snap.cpu_quota_us, snap.cpu_period_us) == (None, None)
    assert (snap.memory_max_bytes, snap.oom_kill) == (8589934592, 0)  # the other readings are unaffected


def test_cgroup_v1_or_unmounted_is_all_null(tmp_path: Path) -> None:
    (tmp_path / "memory").mkdir()  # a v1 hierarchy has controller subdirectories and no cgroup.controllers
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("123\n", encoding="utf-8")
    assert set(read_container_cgroup(tmp_path).as_fields().values()) == {None}
    assert set(read_container_cgroup(tmp_path / "missing").as_fields().values()) == {None}


def test_dir_usage_counts_files_and_reports_missing_as_null(tmp_path: Path) -> None:
    (tmp_path / "w" / "sub").mkdir(parents=True)
    (tmp_path / "w" / "a.bin").write_bytes(b"x" * 10000)
    (tmp_path / "w" / "sub" / "b.bin").write_bytes(b"y" * 10000)
    (tmp_path / "w" / "link").symlink_to(tmp_path / "w" / "a.bin")
    used = dir_usage_bytes(tmp_path / "w")
    assert used is not None and used >= 20000
    assert dir_usage_bytes(tmp_path / "absent") is None


BLOCK = 512


@pytest.fixture
def one_block_each(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weigh every directory entry at one block, so a count says how many entries were visited, on any filesystem."""

    def weigh(entry: os.DirEntry[str]) -> tuple[int, Path | None]:
        return BLOCK, Path(entry.path) if entry.is_dir(follow_symlinks=False) else None

    monkeypatch.setattr(resources, "_entry_blocks", weigh)


def test_dir_usage_visits_every_entry_at_any_depth_and_does_not_follow_a_link_to_a_directory(
    tmp_path: Path, one_block_each: None
) -> None:
    root = tmp_path / "w"
    (root / "sub" / "deeper").mkdir(parents=True)
    for name in ("a.bin", "sub/b.bin", "sub/deeper/c.bin"):
        (root / name).write_bytes(b"x")
    (root / "link").symlink_to(root / "sub", target_is_directory=True)

    assert dir_usage_bytes(root) == 6 * BLOCK  # a.bin, sub, b.bin, deeper, c.bin, and the link as one entry


class _ScanFailingAfter:
    """What ``os.scandir`` returns for a directory that yields ``entries`` and then fails."""

    def __init__(self, entries: list[os.DirEntry[str]]) -> None:
        self.entries = entries

    def __enter__(self) -> "_ScanFailingAfter":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        yield from self.entries
        raise PermissionError("the scan failed")


@pytest.mark.parametrize("read_before_failing", [0, 1])
def test_dir_usage_counts_what_was_read_of_a_directory_that_fails_and_goes_on_with_the_rest(
    tmp_path: Path, one_block_each: None, monkeypatch: pytest.MonkeyPatch, read_before_failing: int
) -> None:
    root = tmp_path / "w"
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"x")
    for name in ("b.bin", "c.bin"):
        (root / "sub" / name).write_bytes(b"x")
    real_scandir = os.scandir

    def scandir(path: str | os.PathLike[str]):
        if Path(path) != root / "sub":
            return real_scandir(path)
        return _ScanFailingAfter(sorted(real_scandir(path), key=lambda entry: entry.name)[:read_before_failing])

    monkeypatch.setattr(os, "scandir", scandir)

    assert (
        dir_usage_bytes(root) == (2 + read_before_failing) * BLOCK
    )  # a.bin and sub, then what sub gave before failing


def test_resource_log_appends_one_json_object_per_line(tmp_path: Path) -> None:
    log = ResourceLog(tmp_path / "agents" / "resources.jsonl")
    log.append({"kind": "a", "n": 1})
    log.append({"kind": "b", "n": 2})
    lines = (tmp_path / "agents" / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["a", "b"]
    assert oct((tmp_path / "agents" / "resources.jsonl").stat().st_mode & 0o777) == "0o600"
