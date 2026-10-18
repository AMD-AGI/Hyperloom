# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resource sampling against a fake /proc and a fake cgroup directory."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.specialists.resources import (
    ProcessTreeSampler,
    ResourceLog,
    dir_usage_bytes,
    read_container_cgroup,
)


def add_proc(
    root: Path, pid: int, ppid: int, *, rss_kb: int, utime: int, stime: int = 0, children: list[int] | None = None
) -> None:
    pdir = root / str(pid)
    (pdir / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
    (pdir / "status").write_text(f"Name:\tx\nVmRSS:\t{rss_kb} kB\n", encoding="utf-8")
    # comm contains a space and a parenthesis on purpose: fields are counted after the LAST ')'.
    stat = f"{pid} (a b) c) S {ppid} {pid} {pid} 0 0 0 0 0 0 0 {utime} {stime} 0 0 20 0 1 0 1"
    (pdir / "stat").write_text(stat, encoding="utf-8")
    if children is not None:
        (pdir / "task" / str(pid) / "children").write_text(" ".join(map(str, children)), encoding="utf-8")


def remove_proc(root: Path, pid: int) -> None:
    import shutil

    shutil.rmtree(root / str(pid))


def test_peak_memory_and_cpu_seconds_over_the_tree(tmp_path: Path) -> None:
    add_proc(tmp_path, 100, 1, rss_kb=1000, utime=200, children=[101, 102])
    add_proc(tmp_path, 101, 100, rss_kb=3000, utime=100, stime=100, children=[])
    add_proc(tmp_path, 102, 100, rss_kb=2000, utime=0, children=[])
    add_proc(tmp_path, 999, 1, rss_kb=777777, utime=9999, children=[])  # unrelated process
    sampler = ProcessTreeSampler(100, proc_root=tmp_path, clock_ticks=100)
    first = sampler.sample()
    assert first.rss_bytes == (1000 + 3000 + 2000) * 1024
    assert first.live_pids == 3
    assert sampler.cpu_seconds == 4.0  # (200 + 200 + 0) ticks / 100
    assert sampler.peak_rss_bytes == 6000 * 1024


def test_exited_child_stops_counting_memory_but_keeps_its_cpu(tmp_path: Path) -> None:
    add_proc(tmp_path, 100, 1, rss_kb=1000, utime=100, children=[101])
    add_proc(tmp_path, 101, 100, rss_kb=5000, utime=300, children=[])
    sampler = ProcessTreeSampler(100, proc_root=tmp_path, clock_ticks=100)
    sampler.sample()
    remove_proc(tmp_path, 101)
    add_proc(tmp_path, 100, 1, rss_kb=1500, utime=150, children=[])
    after = sampler.sample()
    assert after.rss_bytes == 1500 * 1024  # the dead child is no longer counted
    assert after.live_pids == 1
    assert sampler.peak_rss_bytes == 6000 * 1024  # the peak keeps the earlier reading
    assert sampler.cpu_seconds == 1.5 + 3.0  # the child's last-seen CPU stays in the total


def test_falls_back_to_parent_pid_scan_without_children_files(tmp_path: Path) -> None:
    add_proc(tmp_path, 100, 1, rss_kb=1000, utime=0)
    add_proc(tmp_path, 101, 100, rss_kb=2000, utime=0)
    add_proc(tmp_path, 102, 101, rss_kb=4000, utime=0)  # grandchild
    add_proc(tmp_path, 103, 1, rss_kb=9999, utime=0)
    sampler = ProcessTreeSampler(100, proc_root=tmp_path, clock_ticks=100)
    assert sampler.sample().rss_bytes == 7000 * 1024


def test_vanished_root_reads_as_zero_not_an_error(tmp_path: Path) -> None:
    sampler = ProcessTreeSampler(4242, proc_root=tmp_path, clock_ticks=100)
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


def test_resource_log_appends_one_json_object_per_line(tmp_path: Path) -> None:
    log = ResourceLog(tmp_path / "agents" / "resources.jsonl")
    log.append({"kind": "a", "n": 1})
    log.append({"kind": "b", "n": 2})
    lines = (tmp_path / "agents" / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["a", "b"]
    assert oct((tmp_path / "agents" / "resources.jsonl").stat().st_mode & 0o777) == "0o600"
