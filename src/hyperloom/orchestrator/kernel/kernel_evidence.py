# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only discovery of KERNEL session artifacts.

Every scanner here is called from
:func:`hyperloom.orchestrator.kernel.kernel_context.build_kernel_context` so
each lane sees the same evidence index. Materialization stays in the lane that
needs it.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

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
_AITER_MOE_DISPATCH_MARKERS = (b"[fused_moe]", b"Mxfp4 MoE backend")
_AITER_M_RE = re.compile(rb"shape is M:(\d+),")
_LOG_SCAN_CHUNK = 1 << 20
_LOG_SCAN_OVERLAP = 64

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
    counts = _scan_serving_log_m(path)
    if not counts:
        return ""
    reserve = min(max(reserve_largest, 0), max(1, limit // 4))
    picked: list[int] = sorted(counts, reverse=True)[:reserve]
    for value, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(picked) >= limit:
            break
        if value not in picked:
            picked.append(value)
    return ",".join(str(m) for m in sorted(picked))


def _resolve_trace_shape_manifest(state, session_dir: Path) -> str:
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


def _resolve_forge_server_log(state, session_dir: Path) -> str:
    def _find_server_log_near(workspace_str: str) -> str | None:
        if not workspace_str:
            return None
        ws = Path(workspace_str)
        direct = ws / "server.log"
        if direct.is_file() and _log_has_aiter_evidence(direct):
            return str(direct)
        parent = ws.parent
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


def _resolve_fusion_decode_trace(state, payload: dict) -> str:
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


def _read_model_config(model_path: str) -> dict | None:
    if not model_path:
        return None
    from hyperloom.inference_optimizer.model_config_utils import resolve_local_model_dir

    cfg = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _model_hidden_size(model_path: str) -> int | None:
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


def _csv_has_data_rows(path: Path) -> bool:
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


def _csv_matches_model(csv_path: Path, model_path: str) -> bool:
    hidden = _model_hidden_size(model_path)
    if hidden is None:
        return True
    k_values = _csv_k_values(csv_path)
    if not k_values:
        return True
    return hidden in k_values


@functools.lru_cache(maxsize=1)
def _is_gfx950_rocminfo() -> bool:
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


def _is_gfx950(gpu_type: str) -> bool:
    key = (gpu_type or "").strip().lower()
    if key in _GFX950_GPU_TYPES:
        return True
    if not key or key == "auto":
        return _is_gfx950_rocminfo()
    return False


def resolve_fp8_quant_type(model_path: str, gpu_type: str = "", framework: str = "") -> str:
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


def _resolve_forge_untuned_csv(session_dir: Path, precision: str, quant_type: str, model_path: str = "") -> str:
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
    "_log_has_aiter_evidence",
    "_resolve_forge_server_log",
    "_resolve_forge_untuned_csv",
    "_resolve_fusion_decode_trace",
    "_resolve_trace_shape_manifest",
    "_tokens_from_serving_log",
    "resolve_fp8_quant_type",
]
