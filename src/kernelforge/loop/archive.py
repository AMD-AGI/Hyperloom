# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Candidate archive for the forge-loop — persist every iteration's full solution.

The forge-loop keeps only the current *best* kernel on disk, so losing attempts
are ``git revert``-ed away. Without an archive, a later iteration cannot inspect
the actual code of a prior attempt.

This module fixes that by archiving, per iteration, the WHOLE solution and its
measurements into a self-contained directory, so a later iteration (or a human)
can read back the complete kernel, its full profile, and its outcome:

    <workspace>/forge_experiments/candidates/
        index.jsonl              # one compact JSON line per iteration (global view)
        iter_001/
            <kernel>.py          # full kernel snapshot (self-contained)
            change.diff          # full git diff of this iteration's commit
            meta.json            # structured measurement + decision + agent info (incl. profile{})
            profile.txt          # full profiling summary (rocprof-compute SoL or PMC; see meta.profile.backend)
            validation.txt       # full-suite validation report / failure tail
        iter_002/
            ...

Storage is deliberately full-fidelity: kernels are small on disk, and only the
*prompt* (a separate concern) needs to be token-frugal. What we inject into the
next iteration's prompt is decided elsewhere; this module's job is only to make
sure nothing is lost.

Pre-publication failures return ``None``; post-publication index failures raise
so durability-sensitive callers can retain their recovery journal.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from kernelforge.durable_io import atomic_write_text, fsync_directory

log = logging.getLogger(__name__)


@dataclass
class CandidateRecord:
    """Everything worth persisting about one iteration's solution."""

    iteration: int
    commit_hash: str = ""
    decision: str = ""  # KEEP / REVERT_PERF / REVERT_VALIDATION* / BUILD_FAILED
    kept: bool = False
    validation_passed: bool = False

    # Measurement
    wall_ms: float | None = None
    mean_case_speedup: float | None = None
    bench_detail: dict | None = None  # raw bench_wallclock dict
    snr_db: float | None = None
    vgpr: int | None = None
    pmc_diagnosis: str = ""
    # Structured profile metadata (profile_backend, bottleneck, target_kernels,
    # roofline dtype/AI, HBM/compute pct, SoL metrics) — consumed from meta.json.
    profile_meta: dict | None = None

    # Comparison anchors
    baseline_wall_ms: float | None = None
    best_wall_ms_before: float | None = None
    best_mean_case_speedup_before: float | None = None

    # Agent context
    plan: str = ""
    rationale: str = ""

    # Why the agent session ended + turns spent (for end-reason analysis).
    session_end_reason: str = ""
    turns: int | None = None

    # Context
    kernel_file: str = ""
    shape: dict | None = None

    # Full blobs written to their own files (kept out of meta.json to keep it small)
    kernel_source: str = ""
    change_diff: str = ""
    pmc_full: str = ""
    validation_text: str = ""


class CandidateArchive:
    """Per-run store of full iteration solutions + measurements.

    One instance per campaign; ``record`` is called once per iteration that
    produced a commit.
    """

    def __init__(self, workspace_dir: str, kernel_file: str = ""):
        self.root = Path(workspace_dir) / "forge_experiments" / "candidates"
        self.index_path = self.root / "index.jsonl"
        # Snapshot file basename — use the real kernel filename so the archived
        # copy is instantly recognizable (e.g. flash_attn_kernel.py).
        self.kernel_basename = Path(kernel_file).name if kernel_file else "kernel.py"
        self.degraded = False
        self.persistence_errors: list[str] = []
        # In-memory index cache. meta.json stays the on-disk authority; this
        # memoizes the last reconciled view so hot readers (render_digest,
        # load_index, max_iteration) avoid re-scanning + re-parsing every
        # iter_NNN/meta.json on each call. Valid only while THIS process is the
        # sole writer and root's mtime matches what we saw after our last own
        # write; any degradation or unexpected external change drops it back to
        # a full _reconcile_storage(). None means "cold — full reconcile next".
        self._index_cache: list[dict] | None = None
        self._cache_sig: tuple | None = None
        # Bumped on every _mark_degraded; lets a reconcile tell whether it hit
        # any transient trouble mid-scan (→ don't cache that best-effort view).
        self._degrade_seq: int = 0
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._mark_degraded(f"create {self.root}", e)

    def _iter_dir(self, iteration: int) -> Path:
        return self.root / f"iter_{iteration:03d}"

    def _mark_degraded(self, operation: str, error: Exception) -> None:
        self.degraded = True
        self._degrade_seq += 1
        self.persistence_errors.append(f"{operation}: {error}")
        self.persistence_errors = self.persistence_errors[-10:]
        # Any degraded op may have left the on-disk archive inconsistent with the
        # cache; force the next read through a full reconcile so it self-heals.
        self._invalidate_cache()
        log.warning("archive: %s failed: %s", operation, error)

    def _invalidate_cache(self) -> None:
        self._index_cache = None
        self._cache_sig = None

    def _fs_signature(self) -> tuple:
        """Cheap change signal for the archive.

        Combines root's mtime (catches dir-entry add/remove/rename — i.e. a new
        iter_NNN/) with index.jsonl's (mtime, size) (catches in-place rewrites of
        the index, which do NOT bump the parent dir's mtime). Any external change
        moves at least one component, so a stale cache is never served.
        """
        try:
            root_mtime = self.root.stat().st_mtime_ns
        except OSError:
            root_mtime = None
        try:
            st = self.index_path.stat()
            index_sig: tuple | None = (st.st_mtime_ns, st.st_size)
        except OSError:
            index_sig = None
        return (root_mtime, index_sig)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        with open(path, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _iteration_from_dir(path: Path) -> int | None:
        suffix = path.name.removeprefix("iter_")
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _inspect_complete_meta(
        self,
        directory: Path,
        iteration: int,
    ) -> tuple[str, dict | None]:
        """Classify metadata without treating transient I/O as corruption."""
        try:
            meta = json.loads((directory / "meta.json").read_text())
        except FileNotFoundError:
            return "incomplete", None
        except OSError as error:
            self._mark_degraded(f"read {directory / 'meta.json'}", error)
            return "unavailable", None
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return "incomplete", None
        if not isinstance(meta, dict):
            return "incomplete", None
        required = {"iteration", "decision", "kept", "validation_passed", "files"}
        if not required.issubset(meta):
            return "incomplete", None
        if meta.get("iteration") != iteration or not isinstance(meta.get("files"), dict):
            return "incomplete", None
        archive_format = meta.get("archive_format", 1)
        if not isinstance(archive_format, int):
            return "incomplete", None
        if archive_format >= 2 and meta.get("complete") is not True:
            return "incomplete", None
        for filename in meta["files"].values():
            if filename is None:
                continue
            if not isinstance(filename, str) or not filename or Path(filename).name != filename:
                return "incomplete", None
            candidate_path = directory / filename
            try:
                mode = candidate_path.stat().st_mode
            except FileNotFoundError:
                return "incomplete", None
            except OSError as error:
                self._mark_degraded(f"stat {candidate_path}", error)
                return "unavailable", None
            if not stat.S_ISREG(mode):
                return "incomplete", None
        return "complete", meta

    def _complete_meta(self, directory: Path, iteration: int) -> dict | None:
        status, meta = self._inspect_complete_meta(directory, iteration)
        return meta if status == "complete" else None

    def _quarantine_incomplete(self, directory: Path) -> bool:
        quarantine = self.root / (f".{directory.name}.incomplete-{os.getpid()}-{time.time_ns()}")
        try:
            os.replace(directory, quarantine)
            fsync_directory(self.root)
            return True
        except OSError as error:
            self._mark_degraded(f"quarantine incomplete {directory}", error)
            return False

    @staticmethod
    def _index_entry_from_meta(meta: dict, directory: Path) -> dict:
        return {
            "iter": meta["iteration"],
            "decision": meta.get("decision", ""),
            "kept": bool(meta.get("kept")),
            "wall_ms": meta.get("wall_ms"),
            "mean_case_speedup": meta.get("mean_case_speedup"),
            "snr_db": meta.get("snr_db"),
            "delta_vs_best_pct": meta.get("delta_vs_best_pct"),
            "plan": meta.get("plan", ""),
            "session_end_reason": meta.get("session_end_reason", ""),
            "turns": meta.get("turns"),
            "dir": directory.name,
        }

    def _read_index_file(self) -> list[dict]:
        entries: list[dict] = []
        try:
            text = self.index_path.read_text()
        except FileNotFoundError:
            return entries
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _write_index(self, entries: list[dict]) -> None:
        atomic_write_text(
            self.index_path,
            "".join(json.dumps(entry) + "\n" for entry in entries),
        )

    def _reconcile_storage(self) -> list[dict]:
        """Use complete candidate metadata as the canonical archive index."""
        complete: list[tuple[int, Path, dict]] = []
        uncertain_iterations: set[int] = set()
        scan_unavailable = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            directories = sorted(self.root.iterdir())
        except OSError as error:
            self._mark_degraded(f"scan {self.root}", error)
            directories = []
            scan_unavailable = True

        for directory in directories:
            iteration = self._iteration_from_dir(directory)
            if iteration is None:
                continue
            try:
                mode = directory.stat().st_mode
            except FileNotFoundError:
                continue
            except OSError as error:
                self._mark_degraded(f"stat {directory}", error)
                uncertain_iterations.add(iteration)
                continue
            if not stat.S_ISDIR(mode):
                continue
            status, meta = self._inspect_complete_meta(directory, iteration)
            if status == "unavailable":
                uncertain_iterations.add(iteration)
                continue
            if status == "incomplete":
                self._quarantine_incomplete(directory)
                continue
            if meta is not None:
                complete.append((iteration, directory, meta))

        expected = [self._index_entry_from_meta(meta, directory) for _iteration, directory, meta in sorted(complete)]
        try:
            current = self._read_index_file()
        except UnicodeDecodeError:
            current = []
        except OSError as error:
            self._mark_degraded(f"read index {self.index_path}", error)
            return expected

        if scan_unavailable:
            preserved = current
        else:
            preserved = [
                entry
                for entry in current
                if isinstance(entry.get("iter"), int) and entry["iter"] in uncertain_iterations
            ]
        reconciled_by_iteration = {
            entry["iter"]: entry for entry in preserved + expected if isinstance(entry.get("iter"), int)
        }
        reconciled = [reconciled_by_iteration[iteration] for iteration in sorted(reconciled_by_iteration)]

        if not scan_unavailable and not uncertain_iterations and current != expected:
            try:
                self._write_index(expected)
            except OSError as error:
                self._mark_degraded(f"rebuild index {self.index_path}", error)
        return reconciled

    def max_iteration(self) -> int:
        """Highest iteration backed by a complete candidate directory."""
        return max([int(entry["iter"]) for entry in self.load_index() if isinstance(entry.get("iter"), int)], default=0)

    def reconcile_next_iteration(self, state_next_iteration: int) -> int:
        """Return a monotonic cursor that cannot collide with archived attempts."""
        return max(1, state_next_iteration, self.max_iteration() + 1)

    @staticmethod
    def _mean_case_speedup_delta_pct(
        mean_case_speedup: float | None,
        best: float | None,
    ) -> float | None:
        """Signed mean case speedup change vs best (positive = better)."""
        if mean_case_speedup is None or not best:
            return None
        return round((mean_case_speedup / best - 1.0) * 100.0, 3)

    def record(self, rec: CandidateRecord) -> Path | None:
        """Atomically persist one iteration's full solution + measurements.

        Returns the iteration directory path, or None on a pre-publication
        failure. An index append failure is raised after the complete directory
        is published so the caller sees the degraded write and reconciliation can
        recover the missing line later.
        """
        temp_dir: Path | None = None
        try:
            # Warm-up + crash-residue quarantine on the first call; a cheap
            # cache hit afterwards. The target-dir collision check below stats
            # the specific dir directly, so it does not depend on this.
            self.load_index()
            d = self._iter_dir(rec.iteration)
            try:
                existing_mode = d.stat().st_mode
            except FileNotFoundError:
                existing_mode = None
            except OSError as error:
                self._mark_degraded(f"stat {d}", error)
                return None
            if existing_mode is not None:
                if stat.S_ISDIR(existing_mode):
                    status, _meta = self._inspect_complete_meta(d, rec.iteration)
                else:
                    status = "incomplete"
                if status == "unavailable":
                    return None
                if status == "complete":
                    log.warning(
                        "archive: refusing to overwrite complete iteration directory %s",
                        d,
                    )
                    return None
                if not self._quarantine_incomplete(d):
                    return None
            temp_dir = Path(
                tempfile.mkdtemp(
                    dir=str(self.root),
                    prefix=f".iter_{rec.iteration:03d}.tmp-",
                )
            )

            # 1) Full kernel snapshot — self-contained, readable as-is.
            if rec.kernel_source:
                self._write_text(temp_dir / self.kernel_basename, rec.kernel_source)
            # 2) Full diff of the commit (captures sibling-file edits too).
            if rec.change_diff:
                self._write_text(temp_dir / "change.diff", rec.change_diff)
            # 3) Full profiling summary (backend-aware name; may be rocprof-compute
            #    SoL or the legacy PMC summary — profile_meta.backend says which).
            if rec.pmc_full:
                self._write_text(temp_dir / "profile.txt", rec.pmc_full)
            # 4) Validation report / failure tail.
            if rec.validation_text:
                self._write_text(temp_dir / "validation.txt", rec.validation_text)

            # 5) Structured metadata.
            delta_vs_best = self._mean_case_speedup_delta_pct(
                rec.mean_case_speedup,
                rec.best_mean_case_speedup_before,
            )
            meta = {
                "archive_format": 2,
                "complete": True,
                "iteration": rec.iteration,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "commit_hash": rec.commit_hash,
                "decision": rec.decision,
                "kept": rec.kept,
                "validation_passed": rec.validation_passed,
                "wall_ms": rec.wall_ms,
                "mean_case_speedup": rec.mean_case_speedup,
                "bench": rec.bench_detail or {},
                "snr_db": rec.snr_db,
                "vgpr": rec.vgpr,
                "pmc_diagnosis": rec.pmc_diagnosis,
                "profile": rec.profile_meta or {},
                "baseline_wall_ms": rec.baseline_wall_ms,
                "best_wall_ms_before": rec.best_wall_ms_before,
                "best_mean_case_speedup_before": rec.best_mean_case_speedup_before,
                "delta_vs_best_pct": delta_vs_best,
                "plan": rec.plan,
                "rationale": rec.rationale,
                "session_end_reason": rec.session_end_reason,
                "turns": rec.turns,
                "kernel_file": self.kernel_basename,
                "shape": rec.shape or {},
                "files": {
                    "kernel": self.kernel_basename if rec.kernel_source else None,
                    "diff": "change.diff" if rec.change_diff else None,
                    "profile": "profile.txt" if rec.pmc_full else None,
                    "validation": "validation.txt" if rec.validation_text else None,
                },
            }
            self._write_text(temp_dir / "meta.json", json.dumps(meta, indent=2))
            fsync_directory(temp_dir)
            os.rename(temp_dir, d)
            temp_dir = None
            fsync_directory(self.root)
        except Exception as e:
            self._mark_degraded(f"record iteration {rec.iteration}", e)
            return None
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # The complete directory is now durable and authoritative. Do not hide an
        # index failure: load_index() can reconstruct this line from meta.json.
        entry = self._index_entry_from_meta(meta, d)
        try:
            self._append_index(entry)
        except OSError as error:
            self._mark_degraded(f"append index for iteration {rec.iteration}", error)
            raise
        # Disk and cache are now in sync; fold the new line in so the next reader
        # keeps hitting the cache instead of triggering a full rescan.
        self._cache_add_entry(entry)
        return d

    def _append_index(self, entry: dict) -> None:
        with open(self.index_path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ── reading / prompt digest ──────────────────────────────────────────────
    def load_index(self) -> list[dict]:
        """All archived iteration records (compact index lines), oldest first.

        Memoized: the first call (and any call after the archive changed on disk
        or a degraded op) runs the full _reconcile_storage() that re-parses every
        iter_NNN/meta.json; subsequent calls with an unchanged root return the
        cached view in O(1). Returns a fresh list each call so a caller mutating
        the list cannot corrupt the cache (entries themselves are read-only).
        """
        cached = self._cached_index()
        if cached is not None:
            return list(cached)
        seq_before = self._degrade_seq
        reconciled = self._reconcile_storage()
        # Only cache a clean scan; a reconcile that hit transient I/O returns a
        # best-effort view we must re-check next time.
        if self._degrade_seq == seq_before:
            self._store_cache(reconciled)
        else:
            self._invalidate_cache()
        return list(reconciled)

    def _cached_index(self) -> list[dict] | None:
        """Return the cached index iff it is still trustworthy, else None."""
        if self._index_cache is None or self._cache_sig is None:
            return None
        if self._fs_signature() != self._cache_sig:
            return None
        return self._index_cache

    def _store_cache(self, entries: list[dict]) -> None:
        # Capture the signature AFTER reconcile (which may have rewritten
        # index.jsonl) so only a later change moves it.
        self._index_cache = entries
        self._cache_sig = self._fs_signature()

    def _cache_add_entry(self, entry: dict) -> None:
        """Fold one freshly-recorded entry into the cache without a rescan.

        Called after record() has published the dir and appended the index line,
        so the cache stays coherent with disk. If the cache is cold (never built,
        or dropped by a degraded op), leave it cold — the next load_index() will
        reconcile from disk, which now includes this entry.
        """
        if self._index_cache is None:
            return
        by_iter = {
            existing["iter"]: existing for existing in self._index_cache if isinstance(existing.get("iter"), int)
        }
        if isinstance(entry.get("iter"), int):
            by_iter[entry["iter"]] = entry
        self._index_cache = [by_iter[key] for key in sorted(by_iter)]
        self._cache_sig = self._fs_signature()

    def load_meta(self, iteration: int) -> dict:
        """Structured metadata for one archived iteration (best-effort)."""
        directory = self._iter_dir(iteration)
        status, meta = self._inspect_complete_meta(directory, iteration)
        if status == "incomplete":
            try:
                mode = directory.stat().st_mode
            except FileNotFoundError:
                return {}
            except OSError as error:
                self._mark_degraded(f"stat {directory}", error)
                return {}
            if stat.S_ISDIR(mode):
                self._quarantine_incomplete(directory)
        if status == "unavailable":
            return {}
        return meta or {}

    def read_candidate_file(self, iteration: int, filename: str) -> str:
        """Raw content of one file inside an iteration dir (best-effort)."""
        try:
            return (self._iter_dir(iteration) / filename).read_text()
        except Exception as e:
            log.debug("archive: failed to read %s for iter %s: %s", filename, iteration, e)
            return ""

    # Short, prompt-friendly label per decision.
    _OUTCOME_LABEL = {
        "KEEP": "KEEP*",
        "REVERT_PERF": "slow",  # correct but not faster than best
        "REVERT_VALIDATION": "wrong",  # failed correctness
        "REVERT_VALIDATION_TIMEOUT": "validation-timeout",
        "REVERT_VALIDATION_ERROR": "validation-error",
        "BUILD_FAILED": "build-fail",
        "CRASH": "crash",  # raised an exception during the iteration
    }

    @classmethod
    def _label(cls, decision: str) -> str:
        return cls._OUTCOME_LABEL.get(decision or "", decision or "?")

    @staticmethod
    def _fmt_num(v, fmt: str, suffix: str = "") -> str:
        try:
            return format(v, fmt) + suffix
        except (ValueError, TypeError):
            return "-"

    def _row(self, e: dict) -> str:
        """One trajectory-table row for an index entry."""
        it = e.get("iter", "?")
        dec = self._label(e.get("decision", ""))
        wtxt = self._fmt_num(e.get("wall_ms"), ".4f")
        sptxt = self._fmt_num(e.get("mean_case_speedup"), ".4f", "x")
        dtxt = self._fmt_num(e.get("delta_vs_best_pct"), "+.1f", "%")
        plan = (e.get("plan") or "").replace("\n", " ").strip()[:52]
        return f"{it:>4}  {dec:<10} {wtxt:>9} {sptxt:>7} {dtxt:>8}  {plan}"

    def _select_for_diffs(
        self,
        index: list[dict],
        max_full_diffs: int,
        near_miss_count: int,
        recent_count: int,
    ) -> list[dict]:
        """Pick which iterations get a full diff in the prompt (AVO-style Sample).

        Priority: KEPT versions (the winning "lineage" jumps) > closest correct-
        but-not-faster near-misses (promising directions) > most recent attempts
        (what just happened). De-duplicated by iteration, capped, sorted by iter.
        """
        keeps = [e for e in index if e.get("decision") == "KEEP"]
        near = sorted(
            [e for e in index if e.get("decision") == "REVERT_PERF" and e.get("mean_case_speedup") is not None],
            key=lambda e: e["mean_case_speedup"],
            reverse=True,
        )[:near_miss_count]
        recent = index[-recent_count:] if recent_count else []

        prioritized: list[dict] = []
        seen: set = set()
        for e in list(keeps) + list(near) + list(recent):
            it = e.get("iter")
            if it in seen:
                continue
            seen.add(it)
            prioritized.append(e)
        selected = prioritized[:max_full_diffs]
        selected.sort(key=lambda e: e.get("iter", 0))
        return selected

    def _table_entries(self, index: list[dict], max_rows: int) -> tuple[list[dict], bool]:
        """Trajectory rows to show: all if within budget, else KEEPs + latest."""
        if len(index) <= max_rows:
            return index, False
        keeps = [e for e in index if e.get("decision") == "KEEP"]
        tail_n = max(0, max_rows - len(keeps))
        tail = index[-tail_n:] if tail_n else []
        seen: set = set()
        rows: list[dict] = []
        for e in keeps + tail:
            it = e.get("iter")
            if it not in seen:
                seen.add(it)
                rows.append(e)
        rows.sort(key=lambda e: e.get("iter", 0))
        return rows, True

    def render_digest(
        self,
        max_full_diffs: int = 5,
        max_diff_lines: int = 80,
        near_miss_count: int = 3,
        recent_count: int = 2,
        max_table_rows: int = 60,
    ) -> str:
        """Build the prompt digest of the solution lineage (Layers 1-3).

        Layer 1: a compact trajectory table of every attempt + its score.
        Layer 2: full change diffs for a curated few (KEPT + near-misses + recent).
        Layer 3: a pointer to the on-disk archive so the agent can Read/compare
                 any prior kernel's FULL source on demand.
        Returns "" when nothing has been archived yet (e.g. iteration 1).
        """
        index = self.load_index()
        if not index:
            return ""

        # Best-so-far + baseline anchors for the header.
        baseline = None
        last_meta = self.load_meta(index[-1].get("iter", 0))
        if last_meta:
            baseline = last_meta.get("baseline_wall_ms")
        kept_speedups = [
            entry["mean_case_speedup"]
            for entry in index
            if entry.get("decision") == "KEEP" and entry.get("mean_case_speedup") is not None
        ]
        best = max(kept_speedups) if kept_speedups else 1.0

        out: list[str] = []
        # Layer 3 — pointer to the full archive.
        out.append("## Solution archive — your lineage so far")
        out.append("Every prior attempt's FULL kernel + measurements are saved under:")
        out.append(f"  {self.root}/iter_NNN/")
        out.append("    kernel.py  change.diff  meta.json  validation.txt")
        out.append("Read any of them (Read tool, or `git show <commit>`) to study, reuse, or")
        out.append("COMBINE prior approaches — the file on disk is only the current best.")
        out.append("")

        # Layer 1 — trajectory table.
        anchor = ""
        if best is not None:
            anchor += f" best mean case speedup={self._fmt_num(best, '.6f')}x"
        if baseline is not None:
            anchor += f", baseline={self._fmt_num(baseline, '.4f')} ms"
        out.append(f"### Trajectory ({len(index)} attempts;{anchor})")
        out.append(
            "legend: KEEP*=new best · slow=correct-but-not-faster · "
            "wrong=failed correctness · build-fail=didn't compile · "
            "crash=raised an exception"
        )
        rows, capped = self._table_entries(index, max_table_rows)
        out.append(f"{'iter':>4}  {'outcome':<10} {'raw_ms':>9} {'mean×':>7} {'Δvs_best':>8}  plan")
        if capped:
            out.append("  (older rows omitted — showing KEPT versions + most recent)")
        out.extend(self._row(e) for e in rows)

        # Layer 2 — curated full diffs.
        selected = self._select_for_diffs(index, max_full_diffs, near_miss_count, recent_count)
        if selected:
            out.append("")
            out.append("### Notable prior solutions (full change diff vs the state each built on)")
            for e in selected:
                it = e.get("iter", "?")
                dec = self._label(e.get("decision", ""))
                wtxt = self._fmt_num(e.get("wall_ms"), ".4f")
                sptxt = self._fmt_num(e.get("mean_case_speedup"), ".4f", "x")
                dtxt = self._fmt_num(e.get("delta_vs_best_pct"), "+.1f", "%")
                plan = (e.get("plan") or "").replace("\n", " ").strip()[:80]
                out.append("")
                out.append(f'#### iter {it} — {dec} — wall={wtxt} ms, {sptxt}, Δvs_best={dtxt} — "{plan}"')
                diff = self.read_candidate_file(it, "change.diff")
                if not diff.strip():
                    out.append(f"(diff unavailable — full kernel at iter_{it:03d}/kernel.py)")
                    continue
                dlines = diff.splitlines()
                trunc = ""
                if len(dlines) > max_diff_lines:
                    dlines = dlines[:max_diff_lines]
                    trunc = (
                        f"\n... (truncated to {max_diff_lines} lines — Read "
                        f"iter_{it:03d}/kernel.py for the full kernel)"
                    )
                out.append("```diff")
                out.append("\n".join(dlines) + trunc)
                out.append("```")

        return "\n".join(out).strip()
