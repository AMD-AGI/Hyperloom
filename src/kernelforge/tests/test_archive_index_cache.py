# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the in-memory index cache on loop/archive.py.

The candidate archive treats each iter_NNN/meta.json as the on-disk authority
and memoizes the reconciled index so hot readers (render_digest / load_index /
max_iteration) stop re-parsing every meta.json on each call — turning a per-
campaign O(N^2) scan into O(N). These tests pin the cache contract:

  * a warm cache serves load_index() without re-parsing meta.json,
  * record() folds its new line into the cache without a rescan,
  * an unexpected external change (mtime bump) invalidates the cache,
  * a degraded op invalidates the cache (self-heal on next read),
  * a transient scan error is NOT cached (best-effort view is re-checked),
  * a fresh instance (resume) rebuilds the index from disk.

Filesystem via tmp_path; no LLM / GPU."""

from __future__ import annotations

from kernelforge.loop.archive import CandidateArchive, CandidateRecord


def _rec(iteration: int, decision: str = "KEEP", **kw) -> CandidateRecord:
    base = dict(
        iteration=iteration,
        commit_hash=f"c{iteration}",
        decision=decision,
        kept=(decision == "KEEP"),
        validation_passed=True,
        wall_ms=float(10 - iteration),
        snr_db=40.0,
        baseline_wall_ms=10.0,
        best_wall_ms_before=10.0,
        kernel_source=f"# iter {iteration}\n",
        plan=f"plan {iteration}",
    )
    base.update(kw)
    return CandidateRecord(**base)


class _MetaSpy:
    """Count _inspect_complete_meta calls without changing its behavior."""

    def __init__(self, archive: CandidateArchive):
        self.archive = archive
        self.calls = 0
        self._orig = archive._inspect_complete_meta

    def install(self):
        def wrapper(directory, iteration):
            self.calls += 1
            return self._orig(directory, iteration)

        self.archive._inspect_complete_meta = wrapper
        return self


def test_warm_cache_serves_without_reparsing_meta(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    archive.record(_rec(2, decision="REVERT_PERF"))

    # Warm the cache.
    first = archive.load_index()
    assert [e["iter"] for e in first] == [1, 2]

    spy = _MetaSpy(archive).install()
    again = archive.load_index()
    assert [e["iter"] for e in again] == [1, 2]
    assert spy.calls == 0  # served from cache, no meta.json re-parse


def test_load_index_returns_fresh_list_each_call(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    a = archive.load_index()
    b = archive.load_index()
    assert a == b
    assert a is not b  # mutating the returned list must not corrupt the cache
    a.append({"iter": 999})
    assert [e["iter"] for e in archive.load_index()] == [1]


def test_record_folds_new_entry_without_rescan(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    archive.load_index()  # warm

    spy = _MetaSpy(archive).install()
    archive.record(_rec(2))
    # record's internal load_index is a cache hit and the target dir does not
    # pre-exist, so no meta.json is parsed during the record itself.
    assert spy.calls == 0

    idx = archive.load_index()
    assert [e["iter"] for e in idx] == [1, 2]
    assert spy.calls == 0  # new line was folded into the cache


def test_external_change_invalidates_cache(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    assert [e["iter"] for e in archive.load_index()] == [1]  # warm

    # A second writer on the same root (what a stray/concurrent process would
    # look like) bumps root's mtime; the first instance must notice and rescan.
    other = CandidateArchive(str(tmp_path), kernel_file="k.py")
    other.record(_rec(2))

    spy = _MetaSpy(archive).install()
    idx = archive.load_index()
    assert [e["iter"] for e in idx] == [1, 2]
    assert spy.calls > 0  # mtime changed → full reconcile happened


def test_degraded_op_invalidates_cache(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    archive.load_index()  # warm
    assert archive._index_cache is not None

    archive._mark_degraded("synthetic", OSError("boom"))
    assert archive._index_cache is None

    spy = _MetaSpy(archive).install()
    idx = archive.load_index()
    assert [e["iter"] for e in idx] == [1]
    assert spy.calls > 0  # reconciled from disk after degradation


def test_transient_scan_error_is_not_cached(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    archive.record(_rec(2))

    # Make the very next reconcile look transient: one dir reports "unavailable"
    # and marks degraded, exactly like a transient I/O error mid-scan.
    orig = archive._inspect_complete_meta
    state = {"tripped": False}

    def flaky(directory, iteration):
        if not state["tripped"]:
            state["tripped"] = True
            archive._mark_degraded(f"stat {directory}", OSError("transient"))
            return "unavailable", None
        return orig(directory, iteration)

    archive._invalidate_cache()
    archive._inspect_complete_meta = flaky
    archive.load_index()  # hits the transient branch
    assert archive._index_cache is None  # best-effort view was NOT cached

    # Recovery: clean scan now caches normally.
    archive._inspect_complete_meta = orig
    idx = archive.load_index()
    assert [e["iter"] for e in idx] == [1, 2]
    assert archive._index_cache is not None


def test_fresh_instance_rebuilds_index_from_disk(tmp_path):
    writer = CandidateArchive(str(tmp_path), kernel_file="k.py")
    writer.record(_rec(1))
    writer.record(_rec(2, decision="REVERT_PERF"))

    # Simulate --resume: a brand-new instance with an empty in-memory cache must
    # reconstruct the full index by scanning meta.json on disk.
    resumed = CandidateArchive(str(tmp_path), kernel_file="k.py")
    assert resumed._index_cache is None
    idx = resumed.load_index()
    assert [e["iter"] for e in idx] == [1, 2]
    assert resumed.max_iteration() == 2


def test_missing_index_line_recovered_via_reconcile(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="k.py")
    archive.record(_rec(1))
    archive.record(_rec(2))

    # Drop index.jsonl entirely (meta.json remains authoritative) and clear the
    # in-memory cache — reconcile must rebuild the index from the dirs.
    archive.index_path.unlink()
    archive._invalidate_cache()
    idx = archive.load_index()
    assert [e["iter"] for e in idx] == [1, 2]
    assert archive.index_path.exists()  # index rebuilt on disk
