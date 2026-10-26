# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a serving log proves: which aiter kernels ran, and the shapes the model actually saw."""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


#: An aiter dispatch line, hit or miss. Either one proves the process actually
#: routed a GEMM through aiter, which is what makes a server log usable as a
#: shape source; a log without one is silent about shapes no matter how recent
#: or how well its workspace matches.
_AITER_DISPATCH_MARKER = "shape is M:"

#: The MoE half of the same question. The resolved log is not only a dense-shape
#: source: ``kernelforge.gemm_tune.router`` reads it for MoE stage coverage and
#: 1-stage ASM detection, and those parse ``[fused_moe]`` dispatch lines, which
#: aiter prints from a different code path than the dense ``shape is M:`` ones.
#: Requiring the dense marker alone would reject a log that is fully informative
#: about MoE routing -- on a fleet where every model under tuning is MoE, that
#: is the common case, and the router would silently fall back to "tune CK 2-stage
#: unconditionally".
_AITER_MOE_DISPATCH_MARKERS = (b"[fused_moe]", b"Mxfp4 MoE backend")

#: Only the M of a dispatch line. Reading M straight off the serving log is the
#: one token source grounded in what the model actually ran -- forge's fallback
#: derives ``--tokens`` from ``conc`` alone, and real fleet logs reach M=15842,
#: far outside anything that derivation produces.
_AITER_M_RE = re.compile(rb"shape is M:(\d+),")

#: Read logs in chunks: a fleet server.log is ~17MB and the evidence question
#: is usually answered in the first few KB.
_LOG_SCAN_CHUNK = 1 << 20

#: Longest a dispatch prefix can be, so a match straddling a chunk boundary is
#: carried into the next read. ``shape is M:`` plus its digits is far shorter.
_LOG_SCAN_OVERLAP = 64


def _scan_serving_log_m(path) -> dict[int, int]:
    """Count the M values of the dense aiter dispatch lines in a serving log.

    Chunks overlap by ``_LOG_SCAN_OVERLAP`` so a match spanning a boundary is
    still seen, but the carry starts after the last match already counted:
    re-feeding a fixed tail would count any match landing in it twice, which
    skews the frequency ranking that picks ``--tokens``.

    Any read error yields no counts -- an unreadable candidate is not a usable
    shape source either way.
    """
    counts: dict[int, int] = {}
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_LOG_SCAN_CHUNK)
                if not chunk:
                    break
                buf = carry + chunk
                last_end = 0
                for match in _AITER_M_RE.finditer(buf):
                    last_end = match.end()
                    value = int(match.group(1))
                    if value > 0:
                        counts[value] = counts.get(value, 0) + 1
                carry = buf[max(last_end, len(buf) - _LOG_SCAN_OVERLAP) :]
    except (OSError, ValueError):
        return {}
    return counts


def _log_has_aiter_evidence(path) -> bool:
    """True when the log carries at least one aiter dispatch line, dense or MoE.

    Kept separate from :func:`_scan_serving_log_m` rather than folded into it as
    a ``first_only`` flag. Two reasons, both of which cost a real behaviour bug
    when the two were one function:

    * The M counter skips ``M:0``, so a log whose first dispatch line carried
      one read as "no evidence at all".
    * Evidence is not dense-only. ``[fused_moe]`` lines make a log fully usable
      for the MoE routing decisions that consume the same path.

    Stops at the first marker: a log with evidence usually proves it in the
    first few KB, and only a silent log is read to the end.
    """
    markers = (_AITER_DISPATCH_MARKER.encode(), *_AITER_MOE_DISPATCH_MARKERS)
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_LOG_SCAN_CHUNK)
                if not chunk:
                    return False
                buf = carry + chunk
                if any(marker in buf for marker in markers):
                    return True
                carry = buf[-_LOG_SCAN_OVERLAP:]
    except OSError:
        return False


def _tokens_from_serving_log(path, limit: int = 16, reserve_largest: int = 4) -> str:
    """Derive forge's ``--tokens`` from the M values the server actually saw.

    Returns up to ``limit`` distinct M values, smallest first, as a
    comma-separated string -- empty when the log carries no dispatch lines.

    Selection is frequency-ranked, because tuning the M values the model spends
    its time at beats tuning the largest one it ever reached. But frequency
    alone is not enough: a serving warmup sweeps every M about equally often,
    so on real logs the counts come out uniform and the ranking degenerates
    into its tie-break. Measured on two fleet sessions, every distinct M
    carried an identical count (17 values x4, and 44 values x40), so a plain
    frequency cut kept the smallest M and dropped exactly the large prefill
    shapes -- 16384/24576/32768 and 57344/65536 -- that the runtime then
    missed. Reserve slots for the largest observed M so the prefill end
    survives the cut; GEMM time scales with M, so those are also where the
    end-to-end time actually is.
    """
    counts = _scan_serving_log_m(path)
    if not counts:
        return ""
    # Never let the reservation crowd out the frequency ranking: at most a
    # quarter of the budget goes to "largest", and always at least one slot.
    reserve = min(max(reserve_largest, 0), max(1, limit // 4))
    picked: list[int] = sorted(counts, reverse=True)[:reserve]
    for value, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(picked) >= limit:
            break
        if value not in picked:
            picked.append(value)
    return ",".join(str(m) for m in sorted(picked))


def resolve_forge_server_log(state, session_dir: Path) -> str:
    """Find the server log matching the current runtime configuration.

    Priority: current_best workspace (matches the resolved server args)
    → baseline workspace → most recent server.log under runs/.

    Every candidate must carry aiter dispatch evidence. Picking on existence
    alone made the first *present* log win, so a ``current_best`` workspace
    whose log never routed a GEMM through aiter ended the search and the
    ``runs/`` fallback became unreachable -- the tuner was then handed a log
    that had no shapes in it at all.

    The server log is written by the benchmark server at startup and lives in
    the warmup_round benchmark directory (where the server process was first
    launched). When ``current_best.workspace`` points to the measure_round
    benchmark directory (one level sibling), the log is not there — so we also
    check sibling ``warmup_round/`` dirs and walk up to the parent run
    directory.
    """

    def _find_server_log_near(workspace_str: str) -> str | None:
        if not workspace_str:
            return None
        ws = Path(workspace_str)
        # Direct hit (server started in this exact dir).
        direct = ws / "server.log"
        if direct.is_file() and _log_has_aiter_evidence(direct):
            return str(direct)
        # Sibling warmup_round — benchmark dirs sit under
        # {run_hash}/{warmup_round|measure_round}/{benchmark_dir}/
        parent = ws.parent  # e.g. measure_round/
        if parent.name in ("warmup_round", "measure_round"):
            run_hash_dir = parent.parent
        else:
            run_hash_dir = parent
        warmup = run_hash_dir / "warmup_round"
        if warmup.is_dir():
            candidates: list[tuple[float, str]] = []
            for child in warmup.iterdir():
                sl = child / "server.log"
                if sl.is_file():
                    try:
                        candidates.append((sl.stat().st_mtime, str(sl)))
                    except OSError:
                        continue
            candidates.sort(reverse=True)
            for _mtime, candidate in candidates:
                if _log_has_aiter_evidence(candidate):
                    return candidate
        return None

    current_best = getattr(state, "current_best", None) or {}
    found = _find_server_log_near(str(current_best.get("workspace") or "").strip())
    if found:
        return found

    last_baseline = getattr(state, "last_baseline", None) or {}
    if isinstance(last_baseline, dict):
        found = _find_server_log_near(str(last_baseline.get("workspace") or "").strip())
        if found:
            return found

    # Fallback: the whole runs/ tree, newest first. Restricting this to a fixed
    # (baseline, explore, gemm_tuning, roofline) tuple skipped runs/integrate/,
    # which is where the GEMM validation runs put their logs -- those sessions
    # got "" plus a warning telling them to enable a flag that was already on.
    # Newest-first with an early return also means only the logs newer than the
    # winner are scanned, instead of every log in the tree.
    runs_dir = session_dir / "runs"
    candidates_by_age: list[tuple[float, Path]] = []
    if runs_dir.is_dir():
        for candidate_log in runs_dir.glob("**/server.log"):
            try:
                candidates_by_age.append((candidate_log.stat().st_mtime, candidate_log))
            except OSError:
                continue
        candidates_by_age.sort(key=lambda item: item[0], reverse=True)
        for _mtime, candidate_log in candidates_by_age:
            if _log_has_aiter_evidence(candidate_log):
                return str(candidate_log)

    # Separate the two ways this fails. No server.log at all is an upstream
    # gap; logs that exist but never dispatched through aiter means the serving
    # run had AITER_LOG_TUNED_CONFIG off. Both return "", but only the second is
    # actionable, and one silent "" hid it. Reuse the listing above rather than
    # walking the tree a second time.
    if candidates_by_age:
        log.warning(
            "GEMM: %d server.log file(s) under %s but none contain aiter dispatch "
            "lines (dense %r or MoE %s), so there is no runtime shape source. "
            "Serving runs need AITER_LOG_TUNED_CONFIG enabled for shapes to be "
            "observable",
            len(candidates_by_age),
            runs_dir,
            _AITER_DISPATCH_MARKER,
            " / ".join(repr(m.decode()) for m in _AITER_MOE_DISPATCH_MARKERS),
        )
    return ""


#: Markers that identify which kernels aiter is serving, read off a server log.
#: ``bf16_tuned_gemm.csv`` means dense linears resolve through
#: ``aiter/tuned_gemm.py`` (Forge's ``sglang_dense_bf16`` writes that table via
#: ``AITER_CONFIG_GEMM_BF16``); the fused-MoE markers mean the MoE layers run on
#: aiter's CK kernels (Forge's ``fmoe_ck``, via ``AITER_CONFIG_FMOE``) rather
#: than vLLM's Triton ``fused_moe``.
_AITER_SERVING_MARKERS = {
    "bf16_dense": ("bf16_tuned_gemm.csv",),
    "fused_moe": ("[aiter] [fused_moe]", "Mxfp4 MoE backend"),
}


def _aiter_serving_evidence(server_log: str) -> set[str]:
    """Return which aiter kernel families a server log shows in use.

    Routing is driven by the log rather than by precision alone because only the
    log says which backend the model actually got: the same checkpoint runs on
    aiter or on the native path depending on the recipe's env.
    """
    found: set[str] = set()
    if not server_log:
        return found
    try:
        with open(server_log, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for family, markers in _AITER_SERVING_MARKERS.items():
                    if family not in found and any(marker in line for marker in markers):
                        found.add(family)
                if len(found) == len(_AITER_SERVING_MARKERS):
                    break
    except OSError:
        return set()
    return found
