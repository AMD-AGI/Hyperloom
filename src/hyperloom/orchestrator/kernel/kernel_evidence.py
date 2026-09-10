# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only discovery of the artifacts a KERNEL lane reads.

Every scanner here answers one question about what a session already produced:
which serving log carries dispatch evidence, which trace the fusion discover
stage can use, which untuned CSV belongs to this model. They are reads, so
calling one can never cost a GPU or mutate a workspace.

Materialization -- capturing shapes from a live server, writing a CSV, re-keying
a shape table -- is not discovery and stays in the lane that needs it.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import subprocess
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

# Map the resolved (precision, quant_type) to the aiter untuned-GEMM CSV the
# specialist phase records; fp8 "auto" resolves to blockscale (forge default).
_FORGE_UNTUNED_CSV_BY_QUANT: dict[str, str] = {
    "auto": "a8w8_blockscale_untuned_gemm.csv",
    "blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "block_scale": "a8w8_blockscale_untuned_gemm.csv",
    "a8w8_blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "fp8_blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "per_token": "a8w8_untuned_gemm.csv",
    "per_tensor": "a8w8_untuned_gemm.csv",
    "a8w8": "a8w8_untuned_gemm.csv",
    "w8a8": "a8w8_untuned_gemm.csv",
    "w8a8_fp8": "a8w8_untuned_gemm.csv",
    "fp8_w8a8": "a8w8_untuned_gemm.csv",
    "bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "a8w8_bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "a8w8_blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "blockscale+bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "fp4": "a4w4_blockscale_untuned_gemm.csv",
    "mxfp4": "a4w4_blockscale_untuned_gemm.csv",
    "a4w4": "a4w4_blockscale_untuned_gemm.csv",
    "a4w4_blockscale": "a4w4_blockscale_untuned_gemm.csv",
}

_GFX950_GPU_TYPES = frozenset({"mi355x", "gfx950"})


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


def tokens_from_serving_log(path, limit: int = 16, reserve_largest: int = 4) -> str:
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


def resolve_trace_shape_manifest(state, session_dir: Path) -> str:
    """Find the newest TraceShapeManifest this session produced.

    ``bypass_trace_analysis`` writes ``trace_shape_manifest.json`` next to its
    other bypass artifacts; forge calls it the preferred dense-shape source but
    nothing forwarded it, so the file was written and never read. Newest wins:
    a later trace reflects the currently resolved server args.
    """
    # Deduplicate: state.session_dir is usually the same path we were handed,
    # and an empty session then paid for two full-tree walks to find nothing.
    seen_roots: set[str] = set()
    roots: list[Path] = []
    for raw in (session_dir, Path(str(getattr(state, "session_dir", "") or session_dir))):
        if raw is None or not Path(raw).is_dir():
            continue
        key = str(Path(raw).resolve())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        roots.append(Path(raw))
    for root in roots:
        best: tuple[float, str] | None = None
        for found in Path(root).glob("**/trace_shape_manifest.json"):
            try:
                mtime = found.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, str(found))
        if best is not None:
            return best[1]
    return ""


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
    if isinstance(current_best, dict):
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


def resolve_fusion_decode_trace(state, payload: dict) -> str:
    """Reuse the PRELUDE/roofline decode trace for fusion discovery.

    forge-fusion's discover stage needs a CUDA-graph-disabled decode kineto trace,
    already captured in PRELUDE (``state.last_profile_trace``); reuse it instead of
    re-profiling. Explicit ``payload['trace_path']`` wins.
    """

    def _trace_file(path_str: str) -> str:
        path = Path(path_str)
        if path.is_file():
            return str(path)
        if not path.is_dir():
            return ""
        candidates = sorted(
            list(path.glob("*.trace.json.gz")) + list(path.glob("*.trace.json")) + list(path.glob("*.json.gz")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(candidates[0]) if candidates else ""

    explicit = str(payload.get("trace_path") or "").strip()
    if explicit:
        resolved = _trace_file(explicit)
        if resolved:
            return resolved
    trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    if trace:
        resolved = _trace_file(trace)
        if resolved:
            return resolved
    return ""


def _csv_has_data_rows(path: Path) -> bool:
    """Return True when ``path`` is a CSV carrying at least one data row.

    The aiter recorder leaves header-only or empty files for quant types the
    server never exercised; those must not be passed to forge as a real shape
    source.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if "M" not in header.upper():
                return False
            for line in f:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _csv_k_values(path: Path) -> set[int]:
    """Return the distinct integer ``K`` (contraction-dim) values in a CSV.

    The aiter recorder writes a header containing ``M,N,K`` (optionally with
    extra columns such as ``q_dtype_w``). ``K`` is the GEMM contraction dim,
    which for a transformer layer equals its input dim (``hidden_size`` for
    QKV/gate-up/o projections, ``intermediate_size`` for the down projection).
    """
    ks: set[int] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split(",")
            cols = {name.strip().upper(): i for i, name in enumerate(header)}
            kidx = cols.get("K")
            if kidx is None:
                return ks
            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= kidx:
                    continue
                try:
                    ks.add(int(float(parts[kidx])))
                except ValueError:
                    continue
    except OSError:
        return ks
    return ks


def _read_model_config(model_path: str) -> dict | None:
    """Load a HF ``config.json`` as a dict; ``None`` when unavailable/unreadable."""
    if not model_path:
        return None
    # ``model_path`` may be an HF repo id; resolve to the local weights dir
    # (shared resolver) so the config read works for repo-id launches.
    from hyperloom.inference_optimizer.model_config_utils import (
        resolve_local_model_dir,
    )

    cfg = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _model_hidden_size(model_path: str) -> int | None:
    """Read ``hidden_size`` from a HF ``config.json``; ``None`` when unavailable."""
    data = _read_model_config(model_path)
    if data is None:
        return None
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg_dict in candidates:
        for key in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
            val = cfg_dict.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


def resolve_fp8_quant_type(model_path: str, gpu_type: str = "", framework: str = "") -> str:
    """Pick the fp8 dense tuner quant_type from the checkpoint's static format.

    forge accepts an explicit ``quant_type``; rather than letting it fall back to
    its internal blockscale default, hand it the path the model actually runs:

    - ``blockscale_bpreshuffle`` when the checkpoint uses block-quantized
      weights AND the target GPU is gfx950 (MI355X) AND framework is sglang --
      sglang/aiter automatically upgrades blockscale to the bpreshuffle kernel
      on CDNA4. vLLM does NOT use this path (it reads
      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE).
    - ``blockscale`` when the checkpoint uses block-quantized weights on gfx942,
      on vllm, or when GPU type is unknown.
    - ``per_token`` for a plain fp16/bf16 checkpoint served under dynamic
      ``--quantization fp8`` (the a8w8 per-token path).
    - ``auto`` when ``config.json`` cannot be read, so forge sniffs the
      ``kernel_signature_log`` itself (preserves the legacy behaviour and keeps
      the no-readable-config case unchanged).
    """
    data = _read_model_config(model_path)
    if data is None:
        return "auto"
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    is_blockscale = False
    for cfg_dict in candidates:
        qc = cfg_dict.get("quantization_config")
        if isinstance(qc, dict):
            if qc.get("weight_block_size"):
                is_blockscale = True
                break
            method = str(qc.get("quant_method") or qc.get("fmt") or "").lower()
            if "block" in method:
                is_blockscale = True
                break
    if is_blockscale:
        if _is_gfx950(gpu_type) and framework.lower() == "sglang":
            return "blockscale_bpreshuffle"
        return "blockscale"
    return "per_token"


def _is_gfx950(gpu_type: str) -> bool:
    """True when gpu_type resolves to gfx950 (CDNA4 / MI355X)."""
    key = (gpu_type or "").strip().lower()
    if key in _GFX950_GPU_TYPES:
        return True
    if not key or key == "auto":
        return _is_gfx950_rocminfo()
    return False


@functools.lru_cache(maxsize=1)
def _is_gfx950_rocminfo() -> bool:
    """Cached rocminfo probe for gfx950 arch."""
    try:
        out = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
        return "gfx950" in out.lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _csv_matches_model(csv_path: Path, model_path: str) -> bool:
    """Return True when an untuned CSV plausibly belongs to ``model_path``.

    A real per-model dense untuned CSV always contains GEMMs whose ``K`` equals
    the model ``hidden_size``. When ``hidden_size`` is known and absent from the
    CSV's ``K`` column, the CSV was recorded for a different model and is
    rejected so forge derives shapes from the model config instead.

    Returns True when validation is not possible (``hidden_size`` unreadable or
    the CSV exposes no ``K`` column) to avoid false rejections.
    """
    hidden = _model_hidden_size(model_path)
    if hidden is None:
        return True
    k_values = _csv_k_values(csv_path)
    if not k_values:
        return True
    return hidden in k_values


def resolve_forge_untuned_csv(session_dir: Path, precision: str, quant_type: str, model_path: str = "") -> str:
    """Find an aiter untuned-GEMM CSV in a specialist worktree.

    Specialist runs may materialize or modify these files under
    ``runs/specialist/<hash>/worktree/aiter/configs/*_untuned_gemm.csv``; this
    resolver picks the newest non-empty CSV matching the resolved quant type.
    Because an unchanged checkout can also contain static upstream rows, this is
    a fallback behind explicit benchmark input and the latest runtime profile.

    When ``model_path`` is given, candidate CSVs whose GEMM shapes do not match
    the model are rejected so forge derives per-model shapes from ``config.json``.
    Returns the CSV path, or "" when none is available.
    """
    precision = (precision or "").strip().lower()
    quant_type = (quant_type or "").strip().lower()

    fname = _FORGE_UNTUNED_CSV_BY_QUANT.get(quant_type)
    if fname is None:
        log.warning(
            "Forge GEMM shapes: unknown quant_type=%r for precision=%r; not guessing an untuned CSV",
            quant_type,
            precision,
        )
        return ""

    from hyperloom.inference_optimizer.session.session_paths import runs_root

    specialist_dir = runs_root(session_dir) / "specialist"
    if not specialist_dir.is_dir():
        return ""

    best: Path | None = None
    best_mtime = -1.0
    for csv_path in specialist_dir.glob(f"*/worktree/aiter/configs/{fname}"):
        if not _csv_has_data_rows(csv_path):
            continue
        if not _csv_matches_model(csv_path, model_path):
            continue
        try:
            mtime = csv_path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = csv_path

    return str(best) if best is not None else ""


__all__ = [
    "resolve_forge_server_log",
    "resolve_forge_untuned_csv",
    "resolve_fp8_quant_type",
    "resolve_fusion_decode_trace",
    "resolve_trace_shape_manifest",
    "tokens_from_serving_log",
]
