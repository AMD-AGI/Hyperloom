# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness helpers for aiter tuner invocation (Forge-side; no aiter changes)."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.env import env_bool

from .utils import check_gpu_status, run_subprocess

log = logging.getLogger(__name__)

#: Timeout (seconds) passed to aiter ``--timeout``. NOTE: aiter's mp_tuner
#: applies this per *task group* (one group == all candidates of a single shape,
#: e.g. ~53 candidates), NOT per individual candidate. The first group of a fresh
#: run also pays first-time JIT compilation (~44s per kernel module) plus aiter's
#: serial baton-lock builds, so a small value (the old 120s) makes every shape's
#: first group blow the limit and get falsely flagged as "GPU hang" -> 0 tuned.
#: (This supersedes the earlier gfx950 bump to 600s, which was sized as if the
#: timeout were per-candidate; it is per-group, so 600s still starved big shapes.)
#: 3600s (1h) already gave the first-run JIT + a whole group's benchmark ~2x
#: headroom (measured gfx950: the largest-K shape's group extrapolates to ~1750s
#: because candidates JIT-build serially behind a single baton-lock, so 1800s was
#: cutting it too close). Bumped to 7200s (2h) so large models -- more shapes and
#: bigger K per shape, where a cold-JIT group (and queued groups, whose clock runs
#: from submission) can push a single group well past 1h -- stop tripping false
#: "GPU hang" timeouts. It MUST stay meaningfully below the outer per-tuner cap
#: (_DEFAULT_GEMM_TUNING_TIMEOUT_SEC=5h in Hyperloom) so mp_tuner keeps its
#: per-group fault isolation: a genuinely hung group is killed here and the tuner
#: salvages the remaining shapes, instead of one bad group eating the whole outer
#: budget and losing every shape. Overridable via env FORGE_TUNE_TASK_TIMEOUT.
DEFAULT_TASK_TIMEOUT_S = int(os.environ.get("FORGE_TUNE_TASK_TIMEOUT", "7200") or "7200")

#: Opt-in switch for per-shape process isolation + blocklist.
ISOLATE_ENV = "FORGE_ISOLATE_SHAPES"

#: Default blocklist location (provenance-keyed JSON), overridable via env.
#: The directory name predates this package moving under ``kernelforge`` and is
#: deliberately left alone: it is a user-home cache, so renaming it would orphan
#: every blocklist an operator has already accumulated.
_DEFAULT_BLOCKLIST = os.path.expanduser(
    os.environ.get("FORGE_FAULTED_BLOCKLIST", "~/.forge_gemm_tune/faulted_shapes.json")
)

# Fault signatures in tuner output.
_HARD_FAULT_RE = re.compile(
    r"Memory access fault|GPU core ?dump|coredump|HIP error|hipError|"
    r"illegal memory access|Segmentation fault",
    re.IGNORECASE,
)
# Recovered-but-noted: a candidate task timed out or a respawned worker lost its GPU map.
_SOFT_FAULT_RE = re.compile(
    r"\[!\] Task \d+ timed out|Mapping Error|Process PID not in GPU map",
    re.IGNORECASE,
)


def is_isolation_enabled() -> bool:
    """Whether per-shape isolation is opted in via env."""
    return env_bool(ISOLATE_ENV)


def with_task_timeout(cmd: list[str], task_timeout_s: int = DEFAULT_TASK_TIMEOUT_S) -> list[str]:
    """Return ``cmd`` with an aiter ``--timeout`` appended if not already present."""
    if "--timeout" in cmd:
        return cmd
    return [*cmd, "--timeout", str(int(task_timeout_s))]


def classify_fault(rc: int, stdout: str, stderr: str) -> str | None:
    """Classify a per-shape tuner run outcome."""
    if rc == 124:
        return "outer_timeout"
    blob = f"{stdout}\n{stderr}"
    # Only a NON-ZERO exit means the run itself failed on a hard fault.
    if rc != 0 and _HARD_FAULT_RE.search(blob):
        return "hard_fault"
    return None


def count_soft_faults(stdout: str, stderr: str) -> int:
    """Count survivable per-candidate faults (timed-out tasks / mapping errors)."""
    return len(_SOFT_FAULT_RE.findall(f"{stdout}\n{stderr}"))


def read_untuned_csv(path: str | Path) -> tuple[str, list[str]]:
    """Read an untuned CSV into (header_line, data_lines)."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", []
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < 2:
        return (lines[0] if lines else ""), []
    return lines[0], lines[1:]


def shape_signature(row: str) -> str:
    """Stable short signature for one untuned-CSV data row (order-preserving)."""
    norm = ",".join(tok.strip() for tok in row.split(","))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class FaultBlocklist:
    """Provenance-keyed record of shapes that fault even with ``--timeout``."""

    def __init__(self, path: str | Path | None, key: dict[str, Any]):
        self.path = Path(path) if path else Path(_DEFAULT_BLOCKLIST)
        self.key = ":".join(str(key.get(k, "")) for k in ("gpu_type", "quant_type", "tp", "tuner"))
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except (OSError, ValueError):
            self._data = {}

    def _bucket(self) -> dict[str, Any]:
        return self._data.setdefault(self.key, {})

    def is_blocked(self, sig: str) -> bool:
        return sig in self._bucket()

    def record(self, sig: str, reason: str, row: str = "") -> None:
        self._bucket()[sig] = {"reason": reason, "row": row, "ts": int(time.time())}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:  # non-fatal: blocklist is best-effort
            log.warning("could not persist fault blocklist to %s: %s", self.path, exc)

    def filter_rows(self, rows: list[str]) -> tuple[list[str], list[str]]:
        """Return (kept, skipped) partitioning rows by blocklist membership."""
        kept, skipped = [], []
        for r in rows:
            (skipped if self.is_blocked(shape_signature(r)) else kept).append(r)
        return kept, skipped


def gpu_healthy(gpu_ids: str = "") -> bool:
    """Best-effort: rocm-smi responds and the target GPU(s) are not wedged."""
    gpus = check_gpu_status()
    if not gpus:
        return False
    wanted = {g.strip() for g in gpu_ids.split(",") if g.strip()}
    if not wanted:
        return True
    return any(str(g.gpu_id) in wanted for g in gpus)


def compare_temp_env() -> dict[str, str]:
    """Pin producer and collector to one temp root, independent of tempfile's cache."""
    return {"TMPDIR": str(Path(tempfile.gettempdir()).resolve())}


def _read_csv_lines(path: Path) -> list[str]:
    """One per-shape CSV's lines, best-effort: an unreadable file contributes none."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


class _MergedCsv:
    """Rows gathered from many per-shape CSVs, under the first header seen."""

    def __init__(self) -> None:
        self.header: str | None = None
        self.rows: list[str] = []

    def absorb(self, lines: list[str]) -> None:
        """Take one shape's lines; a file with none contributes neither header nor rows."""
        if not lines:
            return
        if self.header is None:
            self.header = lines[0]
        self.rows.extend(lines[1:])

    def write(self, path: Path) -> Path | None:
        """Write the merged CSV, header first, to ``path``; nothing is written when no file contributed."""
        if self.header is None:
            return None
        path.write_text(f"{self.header}\n" + "\n".join(self.rows) + "\n", encoding="utf-8")
        return path


def _rows_to_tune(
    input_csv: str | Path,
    blocklist: FaultBlocklist,
) -> tuple[str, list[str], tuple[int, str] | None]:
    """The header and shape rows to tune, or the exit code and message that says there are none.

    An input with no rows is a failure; an input whose every shape is
    blocklisted is not -- the shapes were skipped on purpose.
    """
    header, rows = read_untuned_csv(input_csv)
    if not rows:
        return header, [], (1, f"no data rows in {input_csv}")
    rows, skipped = blocklist.filter_rows(rows)
    if skipped:
        log.warning("skipping %d blocklisted shape(s) on this regime", len(skipped))
    if not rows:
        return header, [], (0, "all shapes blocklisted (skipped)")
    return header, rows, None


@dataclass(frozen=True)
class TunerInvocation:
    """One aiter tuner script run: the script and its shared flags, the shapes it reads, and where it writes."""

    script: str | Path
    base_args: list[str]
    input_csv: str | Path
    tuned_csv: Path
    work_dir: Path
    aiter_root: Path | None

    def command(self) -> list[str]:
        """The tuner script over ``input_csv`` into ``tuned_csv``, with aiter's per-group ``--timeout`` injected."""
        return with_task_timeout(
            ["python3", str(self.script), "-i", str(self.input_csv), "-o", str(self.tuned_csv), *self.base_args]
        )


@dataclass(frozen=True)
class IsolationGuard:
    """How each isolated shape run is bounded, and where a fault is checked and recorded."""

    outer_timeout_s: int
    gpu_ids: str
    blocklist: FaultBlocklist


def run_isolated(invocation: TunerInvocation, guard: IsolationGuard) -> tuple[int, str, str, Path | None]:
    """Run the aiter tuner once per shape (process isolation), merge results."""
    base_args = invocation.base_args
    work_dir = invocation.work_dir
    tuned_stem = invocation.tuned_csv.stem
    header, rows, nothing_to_tune = _rows_to_tune(invocation.input_csv, guard.blocklist)
    if nothing_to_tune is not None:
        return nothing_to_tune[0], "", nothing_to_tune[1], None

    merged_out: list[str] = []
    merged_err: list[str] = []
    candidates = _MergedCsv()
    n_ok = 0
    env_override = compare_temp_env()
    compare_dir = Path(env_override["TMPDIR"]) / "aiter_compare"

    # ``base_args`` carries a single shared ``-o2`` profile path.
    profile_idx = base_args.index("-o2") if "-o2" in base_args[:-1] else -1
    profiles = _MergedCsv()

    for idx, row in enumerate(rows):
        shape_csv = work_dir / f"_iso_{tuned_stem}_{idx}.csv"
        shape_out = work_dir / f"_iso_{tuned_stem}_{idx}_tuned.csv"
        shape_csv.write_text(f"{header}\n{row}\n", encoding="utf-8")

        # Per-shape profile so shapes don't overwrite each other's -o2 output.
        shape_args = list(base_args)
        shape_profile = work_dir / f"_iso_{tuned_stem}_{idx}_profile.csv"
        if profile_idx >= 0:
            shape_args[profile_idx + 1] = str(shape_profile)
        shape = dataclasses.replace(invocation, base_args=shape_args, input_csv=shape_csv, tuned_csv=shape_out)
        start = time.time()
        rc, out, err = run_subprocess(
            shape.command(),
            cwd=invocation.aiter_root,
            timeout_s=guard.outer_timeout_s,
            log_file=work_dir / f"_iso_{tuned_stem}_{idx}.log",
            env_override=env_override,
        )
        merged_out.append(out)
        merged_err.append(err)

        # Accumulate this shape's profile candidates (best-effort) so the merged profile downstream carries every
        # shape, not just the last.
        if profile_idx >= 0:
            profiles.absorb([line for line in _read_csv_lines(shape_profile) if line.strip()])

        fault = classify_fault(rc, out, err)
        if fault is not None:
            log.warning("shape %d/%d faulted (%s); recording to blocklist", idx + 1, len(rows), fault)
            guard.blocklist.record(shape_signature(row), fault, row)
            if not gpu_healthy(guard.gpu_ids):
                log.error("GPU unhealthy after fault; aborting remaining shapes")
                merged_err.append("gpu_unhealthy_abort")
                break
            continue
        if rc != 0:
            # Non-zero exit with no recognized fault (bad args / Python traceback): a real failure, not a tuned shape
            # -- do not count it as ok (otherwise a whole run of these reports final_rc=0).
            log.warning("shape %d/%d exited rc=%d with no recognized fault; treating as failed", idx + 1, len(rows), rc)
            continue
        n_ok += 1
        if soft := count_soft_faults(out, err):
            log.info("shape %d/%d tuned with %d recovered candidate fault(s)", idx + 1, len(rows), soft)
        # aiter writes this shape's compare candidate under compare_dir.
        cand = _latest_candidate(compare_dir, shape_out.stem, start)
        if cand is not None:
            candidates.absorb(_read_csv_lines(cand))

    guard.blocklist.save()

    # Merge every shape's profile back into the shared -o2 path so the serve-safe split-K cap sees candidates for ALL
    # shapes, not just the last.
    if profile_idx >= 0:
        with contextlib.suppress(OSError):
            profiles.write(Path(base_args[profile_idx + 1]))
    merged_candidate_path = candidates.write(work_dir / f"candidate_{tuned_stem}.merged.csv")

    # rc: 0 if at least one shape tuned; 1 only if all faulted/failed.
    return (0 if n_ok > 0 else 1), "\n".join(merged_out), "\n".join(merged_err), merged_candidate_path


def _latest_candidate(compare_dir: Path, tuned_stem: str, start: float) -> Path | None:
    """Newest ``*.candidate.csv`` under compare_dir for this stem, newer than start."""
    if not compare_dir.is_dir():
        return None
    boundary = re.compile(r"^" + re.escape(tuned_stem) + r"(?:\.|_\d)")
    cands = [p for p in compare_dir.glob("*.candidate.csv") if boundary.search(p.name) and p.stat().st_mtime > start]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


__all__ = [
    "DEFAULT_TASK_TIMEOUT_S",
    "ISOLATE_ENV",
    "is_isolation_enabled",
    "with_task_timeout",
    "classify_fault",
    "count_soft_faults",
    "read_untuned_csv",
    "shape_signature",
    "FaultBlocklist",
    "IsolationGuard",
    "TunerInvocation",
    "gpu_healthy",
    "run_isolated",
]
