# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness helpers for aiter tuner invocation (Forge-side; no aiter changes).

Two problems this addresses, entirely from the Forge side:

1. **Hang on a faulting candidate.** aiter's ``mp_tuner`` only activates its
   GPU-fault isolation when the tuner is invoked with ``--timeout``. Forge did
   not pass it, so a single faulting kernel candidate (e.g. an asm fmoe tile on
   gfx950) hung the whole run until the outer subprocess cap (up to an hour),
   losing every shape. We now always inject ``--timeout`` (see
   :func:`with_task_timeout`) so aiter's per-candidate recovery kicks in.

2. **No isolation / no record of faulting shapes.** Per-candidate isolation
   lives inside ``mp_tuner`` (aiter) and is out of scope. What we *can* do from
   Forge is **per-shape** process isolation: split the untuned CSV and invoke
   the tuner once per shape, so one shape's fault storm cannot disrupt another's
   benchmark, and record shapes that fail even with ``--timeout`` into a
   provenance-keyed blocklist so future runs skip them (see
   :class:`FaultBlocklist` and :func:`run_isolated`).

Per-shape isolation is **opt-in** (env ``FORGE_ISOLATE_SHAPES=1``); the default
path is unchanged except for the always-on ``--timeout`` injection. Pure stdlib;
reuses ``utils.run_subprocess`` / ``utils.check_gpu_status``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

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

# Fault signatures in tuner output. A run that trips these on a shape means that
# shape could not be tuned even with per-candidate recovery -> blocklist it.
_HARD_FAULT_RE = re.compile(
    r"Memory access fault|GPU core ?dump|coredump|HIP error|hipError|"
    r"illegal memory access|Segmentation fault",
    re.IGNORECASE,
)
# Recovered-but-noted: a candidate task timed out or a respawned worker lost its
# GPU map. These are survivable (aiter continues); we count them, not blocklist.
_SOFT_FAULT_RE = re.compile(
    r"\[!\] Task \d+ timed out|Mapping Error|Process PID not in GPU map",
    re.IGNORECASE,
)


def is_isolation_enabled() -> bool:
    """Whether per-shape isolation is opted in via env."""
    return os.environ.get(ISOLATE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def with_task_timeout(cmd: list[str], task_timeout_s: int = DEFAULT_TASK_TIMEOUT_S) -> list[str]:
    """Return ``cmd`` with an aiter ``--timeout`` appended if not already present.

    This is the one always-on fix: it activates aiter mp_tuner's per-candidate
    GPU-fault isolation. Idempotent (no double ``--timeout``).
    """
    if "--timeout" in cmd:
        return cmd
    return [*cmd, "--timeout", str(int(task_timeout_s))]


def classify_fault(rc: int, stdout: str, stderr: str) -> str | None:
    """Classify a per-shape tuner run outcome.

    Returns ``"outer_timeout"`` (rc 124 -> whole run killed by the subprocess
    cap, i.e. hung even with ``--timeout``), ``"hard_fault"`` (GPU memory fault /
    coredump), or ``None`` when the run completed acceptably (soft, recovered
    faults do not count). Soft faults are logged by the caller, not returned.
    """
    if rc == 124:
        return "outer_timeout"
    blob = f"{stdout}\n{stderr}"
    # Only a NON-ZERO exit means the run itself failed on a hard fault. With
    # aiter --timeout (mp_tuner), per-candidate memory-access / HIP faults are
    # recovered (rc==0) and merely printed under -v; classifying those as a
    # hard fault would blocklist a shape that actually tuned fine -- exactly the
    # "recovered faults do not count" contract this function documents.
    if rc != 0 and _HARD_FAULT_RE.search(blob):
        return "hard_fault"
    return None


def count_soft_faults(stdout: str, stderr: str) -> int:
    """Count survivable per-candidate faults (timed-out tasks / mapping errors)."""
    return len(_SOFT_FAULT_RE.findall(f"{stdout}\n{stderr}"))


def read_untuned_csv(path: str | Path) -> tuple[str, list[str]]:
    """Read an untuned CSV into (header_line, data_lines). Never raises on a
    missing/short file -> returns ("", [])."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", []
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < 2:
        return (lines[0] if lines else ""), []
    return lines[0], lines[1:]


def shape_signature(row: str) -> str:
    """Stable short signature for one untuned-CSV data row (order-preserving).

    Works for both dense (``M,N,K[,q_dtype_w]``) and MoE
    (``token,model_dim,inter_dim,expert,...``) rows: the whole normalized row is
    the shape identity.
    """
    norm = ",".join(tok.strip() for tok in row.split(","))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class FaultBlocklist:
    """Provenance-keyed record of shapes that fault even with ``--timeout``.

    Keyed by (gpu_type, quant_type, tp, tuner) so a blocklist entry only applies
    to the exact regime it was observed in -- a tile that faults on gfx950 fp8
    must not suppress tuning on a different arch/quant.
    """

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
    """Best-effort: rocm-smi responds and the target GPU(s) are not wedged.

    A GPU memory fault can transiently wedge the device; we probe before moving
    to the next shape. Returns True when rocm-smi returns any GPU data (we treat
    an unreadable rocm-smi as unhealthy). ``gpu_ids`` (comma list) narrows the
    check; empty means any GPU.
    """
    gpus = check_gpu_status(skip=False)
    if not gpus:
        return False
    wanted = {g.strip() for g in gpu_ids.split(",") if g.strip()}
    if not wanted:
        return True
    return any(str(g.gpu_id) in wanted for g in gpus)


def run_isolated(
    *,
    script: str,
    base_args: list[str],
    input_csv: str | Path,
    tuned_stem: str,
    work_dir: Path,
    aiter_root: Path | None,
    outer_timeout_s: int,
    task_timeout_s: int,
    gpu_ids: str,
    blocklist: FaultBlocklist | None,
) -> tuple[int, str, str, Path | None]:
    """Run the aiter tuner once per shape (process isolation), merge results.

    ``base_args`` are the flags shared by every shape (everything except
    ``-i``/``-o``; must NOT already contain them). Returns
    ``(rc, merged_stdout, merged_stderr, merged_candidate_path)`` shaped exactly
    like a single :func:`run_subprocess` call so the caller's existing stdout /
    candidate-CSV parsing is unchanged. ``rc`` is 0 unless *every* shape faulted.
    """
    header, rows = read_untuned_csv(input_csv)
    if not rows:
        return 1, "", f"no data rows in {input_csv}", None

    if blocklist is not None:
        rows, skipped = blocklist.filter_rows(rows)
        if skipped:
            log.warning("skipping %d blocklisted shape(s) on this regime", len(skipped))
        if not rows:
            return 0, "", "all shapes blocklisted (skipped)", None

    merged_out: list[str] = []
    merged_err: list[str] = []
    merged_candidate_rows: list[str] = []
    candidate_header: str | None = None
    n_ok = 0
    compare_dir = Path("/tmp/aiter_compare")

    # ``base_args`` carries a single shared ``-o2`` profile path. Reusing it
    # verbatim for every shape makes each shape overwrite the same file, so only
    # the LAST shape's candidates survive -> the serve-safe split-K cap
    # downstream then drops (falls back to default) every other shape's
    # over-cap splitK rows, silently losing the split-K gain. Give each shape
    # its own profile and merge them all back into the shared path.
    try:
        profile_idx = base_args.index("-o2")
        shared_profile: Path | None = Path(base_args[profile_idx + 1])
    except (ValueError, IndexError):
        profile_idx, shared_profile = -1, None
    merged_profile_rows: list[str] = []
    profile_header: str | None = None

    for idx, row in enumerate(rows):
        sig = shape_signature(row)
        shape_csv = work_dir / f"_iso_{tuned_stem}_{idx}.csv"
        shape_out = work_dir / f"_iso_{tuned_stem}_{idx}_tuned.csv"
        shape_csv.write_text(f"{header}\n{row}\n", encoding="utf-8")

        # Per-shape profile so shapes don't overwrite each other's -o2 output.
        shape_args = list(base_args)
        shape_profile = work_dir / f"_iso_{tuned_stem}_{idx}_profile.csv"
        if profile_idx >= 0:
            shape_args[profile_idx + 1] = str(shape_profile)
        cmd = with_task_timeout(
            ["python3", str(script), "-i", str(shape_csv), "-o", str(shape_out), *shape_args],
            task_timeout_s,
        )
        start = time.time()
        rc, out, err = run_subprocess(
            cmd,
            cwd=aiter_root,
            timeout_s=outer_timeout_s,
            log_file=work_dir / f"_iso_{tuned_stem}_{idx}.log",
        )
        merged_out.append(out)
        merged_err.append(err)

        # Accumulate this shape's profile candidates (best-effort) so the merged
        # profile downstream carries every shape, not just the last.
        if profile_idx >= 0 and shape_profile.is_file():
            try:
                plines = [ln for ln in shape_profile.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except OSError:
                plines = []
            if plines:
                if profile_header is None:
                    profile_header = plines[0]
                merged_profile_rows.extend(plines[1:])

        fault = classify_fault(rc, out, err)
        soft = count_soft_faults(out, err)
        if fault is not None:
            log.warning("shape %d/%d faulted (%s); recording to blocklist", idx + 1, len(rows), fault)
            if blocklist is not None:
                blocklist.record(sig, fault, row)
            if not gpu_healthy(gpu_ids):
                log.error("GPU unhealthy after fault; aborting remaining shapes")
                merged_err.append("gpu_unhealthy_abort")
                break
            continue
        if rc != 0:
            # Non-zero exit with no recognized fault (bad args / Python
            # traceback): a real failure, not a tuned shape -- do not count it
            # as ok (otherwise a whole run of these reports final_rc=0).
            log.warning(
                "shape %d/%d exited rc=%d with no recognized fault; treating as failed",
                idx + 1,
                len(rows),
                rc,
            )
            continue
        n_ok += 1
        if soft:
            log.info("shape %d/%d tuned with %d recovered candidate fault(s)", idx + 1, len(rows), soft)
        # Collect this shape's compare candidate (aiter writes it under compare_dir).
        cand = _latest_candidate(compare_dir, tuned_stem, start)
        if cand is not None:
            try:
                clines = cand.read_text(encoding="utf-8").splitlines()
            except OSError:
                clines = []
            if clines:
                if candidate_header is None:
                    candidate_header = clines[0]
                merged_candidate_rows.extend(clines[1:])

    if blocklist is not None:
        blocklist.save()

    # Merge every shape's profile back into the shared -o2 path so the
    # serve-safe split-K cap sees candidates for ALL shapes, not just the last.
    if profile_idx >= 0 and shared_profile is not None and profile_header is not None:
        try:
            shared_profile.write_text(profile_header + "\n" + "\n".join(merged_profile_rows) + "\n", encoding="utf-8")
        except OSError:
            pass

    merged_candidate_path: Path | None = None
    if candidate_header is not None:
        merged_candidate_path = work_dir / f"candidate_{tuned_stem}.merged.csv"
        merged_candidate_path.write_text(
            candidate_header + "\n" + "\n".join(merged_candidate_rows) + "\n", encoding="utf-8"
        )

    # rc: 0 if at least one shape tuned; 1 only if all faulted/failed.
    final_rc = 0 if n_ok > 0 else 1
    return final_rc, "\n".join(merged_out), "\n".join(merged_err), merged_candidate_path


def _latest_candidate(compare_dir: Path, tuned_stem: str, start: float) -> Path | None:
    """Newest ``*.candidate.csv`` under compare_dir for this stem, newer than start.

    The stem must appear as a whole token, not a substring: the dense tuner names
    nest by prefix (``tuned_a8w8_blockscale`` is a prefix of
    ``tuned_a8w8_blockscale_bpreshuffle``), so a plain ``in`` test would let a
    shorter tuner steal a longer sibling's candidate. Require the stem to be
    followed by ``.`` (extension) or ``_<digit>`` (the per-shape index).
    """
    if not compare_dir.is_dir():
        return None
    boundary = re.compile(re.escape(tuned_stem) + r"(?:\.|_\d)")
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
    "gpu_healthy",
    "run_isolated",
]
