# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An aiter dense tuner run as chunks of its untuned CSV (see ``_chunks``), merged into the files one run writes."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .. import tune_robustness as _tr
from ._chunks import CHUNK_DIR, ChunkBudget, MergedCsv, csv_lines, decode_band_units, run_in_chunks, write_rows

#: ``run_subprocess``'s signature, passed in so the tuner module's own binding is the one called.
RunSubprocess = Callable[..., tuple[int, str, str]]


@dataclasses.dataclass(frozen=True)
class DenseRun:
    """A tuner run: rc 0 when a chunk succeeded, else the last chunk's; the merged candidate CSV when aiter wrote any;
    the shapes it was given (0 when unknown) and, when part of them never got a finished chunk, why."""

    rc: int
    stdout: str
    stderr: str
    candidate: Path | None
    expected_shapes: int = 0
    shortfall: str = ""


def _newest(directory: Path, stem: str, since: float, suffix: str) -> Path | None:
    """The newest ``<stem>.<pid><suffix>`` aiter wrote under ``directory`` after ``since``."""
    if not directory.is_dir():
        return None
    found = [path for path in directory.glob(f"{stem}.*{suffix}") if path.stat().st_mtime > since]
    return max(found, key=lambda path: path.stat().st_mtime, default=None)


class _DenseChunks:
    """Each chunk tunes into CSVs of its own; its rows, profile, candidate and compare table join the run's."""

    def __init__(self, invocation: _tr.TunerInvocation, header: str, run: RunSubprocess) -> None:
        self.invocation, self.header, self.run_subprocess = invocation, header, run
        args = invocation.base_args
        self.profile_at = args.index("-o2") + 1 if "-o2" in args[:-1] else -1
        self.tuned = MergedCsv(invocation.tuned_csv)
        self.profiles = MergedCsv(Path(args[self.profile_at])) if self.profile_at > 0 else None
        self.candidates = MergedCsv(invocation.work_dir / f"candidate_{invocation.tuned_csv.stem}.merged.csv")
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.codes: list[int] = []
        self.failures: list[str] = []

    def run(self, index: int, rows: list[str], timeout_s: int) -> bool:
        work, stem = self.invocation.work_dir, self.invocation.tuned_csv.stem
        chunk_dir = work / CHUNK_DIR
        chunk_dir.mkdir(exist_ok=True)
        args = list(self.invocation.base_args)
        profile, tuned = chunk_dir / f"{stem}_{index}.profile.csv", chunk_dir / f"{stem}_{index}.csv"
        # aiter merges into an -o that exists, so a previous run's chunk would come back as this one's rows.
        for stale in (profile, tuned):
            stale.unlink(missing_ok=True)
        if self.profiles is not None:
            args[self.profile_at] = str(profile)
        chunk = dataclasses.replace(
            self.invocation,
            base_args=args,
            input_csv=write_rows(chunk_dir / f"un{stem}_{index}.csv", self.header, rows),
            tuned_csv=tuned,
        )
        env = _tr.compare_temp_env()
        compare_dir = Path(env["TMPDIR"]) / "aiter_compare"
        started = time.time()
        rc, stdout, stderr = self.run_subprocess(
            chunk.command(),
            cwd=self.invocation.aiter_root,
            timeout_s=timeout_s,
            log_file=work / f"tune.{index}.log",
            env_override=env,
        )
        # Past 30 shapes aiter writes its compare table to a file instead of stdout.
        report = _newest(compare_dir, chunk.tuned_csv.stem, started, ".compare.txt")
        self.stdout.append(stdout + ("\n" + report.read_text(encoding="utf-8", errors="replace") if report else ""))
        self.stderr.append(stderr)
        self.codes.append(rc)
        if rc != 0:
            self.failures.append(f"chunk {index} of {len(rows)} shape(s) exited {rc}: {stderr.strip()[-300:]}")
        self.tuned.absorb(csv_lines(chunk.tuned_csv))
        if self.profiles is not None:
            self.profiles.absorb(csv_lines(profile))
        candidate = _newest(compare_dir, chunk.tuned_csv.stem, started, ".candidate.csv")
        if candidate is not None:
            self.candidates.absorb(csv_lines(candidate))
        # aiter exits 1 when any shape in its input found no valid candidate; the next chunk's shapes are not at fault.
        return True


def run_dense_chunks(
    invocation: _tr.TunerInvocation, budget: ChunkBudget, decode_m: Sequence[int], run: RunSubprocess
) -> DenseRun:
    """Tune ``invocation``'s shapes chunk by chunk."""
    header, rows = _tr.read_untuned_csv(invocation.input_csv)
    if not rows:
        return DenseRun(1, "", f"no data rows in {invocation.input_csv}", None)
    chunks = _DenseChunks(invocation, header, run)
    tally = run_in_chunks(decode_band_units(rows, header, decode_m), budget, chunks.run)
    shortfall = list(chunks.failures)
    if tally.shapes_left:
        shortfall.append(f"{tally.shapes_left} shape(s) not reached before the deadline")
    return DenseRun(
        rc=0 if 0 in chunks.codes else chunks.codes[-1],
        stdout="\n".join(chunks.stdout),
        stderr="\n".join(chunks.stderr),
        candidate=chunks.candidates.path if chunks.candidates.header is not None else None,
        expected_shapes=len(rows),
        shortfall="; ".join(shortfall),
    )
