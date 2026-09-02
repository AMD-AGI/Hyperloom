# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``profile`` ActionRunner — Magpie run with torch profiler on.

Reuses the BaselineExecutor shell-out machinery; only the YAML differs
(``profiler.torch_profiler.enabled: true``), so Magpie writes trace files
under ``torch_trace/`` (or ``capture_traces/`` for TraceLens vLLM capture).

Result schema (delivered on the bus as ``delegated_result``)::

    status:        "succeeded" | "failed"
    framework:     "sglang" | "vllm" | "atom" | "xdit" | "custom"
    model:         path
    request/output/total_token_throughput, latency stats (same as baseline)
    workspace:     absolute path of the Magpie workspace
    trace_dir:     absolute path of the torch_trace dir (or None)
    trace_files:   absolute paths of the selected trace files
    main_trace_path: absolute path of the chosen main trace
    primary_rank / rank_trace_paths / merged_trace_paths: trace topology
    trace_health:  structure-check dict (see _validate_trace_structure)
    trace_capture_status: AgentX capture lifecycle status, when available
    trace_capture: full AgentX capture status sidecar payload
    profile_trace_selection_reason: why that main trace was picked
    report_path:   absolute path of benchmark_report.json
    error_class / error: set on the failure path (e.g. "no_trace_files")

In-repo consumers (roofline.py, loop/writeback.py) prefer ``main_trace_path``
and fall back to ``trace_files[0]``; the rest is surfaced so the baseline
SharedState promotion works unchanged.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from hyperloom.agents.kernel.tools._capture_shapes import (
    is_capture_fragment as _shared_is_capture_fragment,
)
from hyperloom.common.io import safe_mtime
from hyperloom.common.profile_args import sanitize_profile_server_args as _sanitize_profile_server_args
from hyperloom.inference_optimizer.session.paths import asset_root, mn_profile_trace_root
from ._inferencex_patcher import (
    ensure_benchmark_lib_patched,
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_serving_patched,
)
from ._xdit_patcher import verify_xdit_profiler_baked
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


# Leading bytes of a trace to sample for sentinel substrings.
_TRACE_INSPECT_BYTES = 2_000_000

# Cap for the confirmation streaming scan used when the leading-window sample
# finds zero of a sentinel. Override via ``INFERENCE_OPTIMIZER_TRACE_CONFIRM_BYTES``.
_TRACE_CONFIRM_BYTES = 64_000_000

# Min fraction of ``cpu_op`` events carrying ``Input Dims`` for a healthy
# ``capture_traces/`` file (Deval ref 99.97%; gated low to avoid false-positives).
_INPUT_DIMS_FRACTION_FLOOR = 0.90


def _trace_contains(path: Path, substring: str, max_bytes: int | None = None) -> bool:
    """Stream-decompress ``path`` for ``substring``, reading at most
    ``max_bytes`` (default :data:`_TRACE_CONFIRM_BYTES`).

    Confirmation pass when :func:`_sample_trace_text` finds zero
    occurrences. Returns ``False`` on any IO/decode error (never raises).

    Args:
        path: The gzipped trace file to scan.
        substring: The marker substring to search for.
        max_bytes: Maximum decompressed bytes to read; defaults to the
            ``INFERENCE_OPTIMIZER_TRACE_CONFIRM_BYTES`` env value or
            :data:`_TRACE_CONFIRM_BYTES`.

    Returns:
        True if ``substring`` is found within ``max_bytes``, False otherwise
        (including on any IO/decode error).
    """
    if not substring:
        return False
    if max_bytes is None:
        try:
            max_bytes = int(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_TRACE_CONFIRM_BYTES",
                    _TRACE_CONFIRM_BYTES,
                )
            )
        except (TypeError, ValueError):
            max_bytes = _TRACE_CONFIRM_BYTES
    read = 0
    carry = ""
    chunk_size = 4_000_000
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            while read < max_bytes:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                read += len(chunk)
                if substring in (carry + chunk):
                    return True
                # Carry tail to catch a sentinel split across the chunk boundary.
                carry = chunk[-(len(substring)) :]
    except (OSError, EOFError, UnicodeDecodeError) as e:
        log.debug("_trace_contains: cannot stream %s: %s", path, e)
        return False
    return False


def _sample_trace_text(path: Path) -> str | None:
    """Read up to ``_TRACE_INSPECT_BYTES`` of decompressed text from a
    gzipped trace. Returns ``None`` (debug-logged) on IO/decode error so the
    check is skipped rather than failing the profile path.

    Args:
        path: The gzipped trace file to sample.

    Returns:
        The decompressed leading text, or ``None`` on IO/decode error.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read(_TRACE_INSPECT_BYTES)
    except (OSError, EOFError, UnicodeDecodeError) as e:
        # Best-effort: a malformed sample must not fail the profile path.
        log.debug(
            "_validate_trace_structure: cannot sample %s: %s",
            path,
            e,
        )
        return None


def _count_substring_occurrences(text: str, substring: str) -> int:
    """Count non-overlapping ``substring`` occurrences as a cheap
    lower-bound event count (avoids full JSON parsing).

    Args:
        text: The text to scan.
        substring: The substring to count.

    Returns:
        The number of non-overlapping occurrences (0 when ``substring`` is
        empty).
    """
    if not substring:
        return 0
    return text.count(substring)


def _validate_trace_structure(
    trace_dir: Path,
    framework: str,
) -> dict[str, Any]:
    """Post-profile sanity check on the produced trace structure.

    Logs warnings (never raises) when the trace structure suggests
    TraceLens features didn't reach the framework:

    1. ``capture_traces/`` exists with files (graph capture fired)
    2. capture files contain ``cpu_op`` events with ``Input Dims``
       (shape-discovery instrumentation recording per-event shapes)
    3. main-directory trace has ``user_annotation`` events including
       ``execute_*`` annotations (InferenceX per-step instrumentation fired)
    4. ``trace_split/`` per-file ``execute_*`` user_annotations
       counted (splitter ran AND each split is non-empty)
    5. (sglang only) main trace contains ``kernel_shape_profiler``
       substring (server-side patch landed)
    6. (Hyperloom-specific) ``trace_split/`` has ``_steady_state_*``
       files, NOT ``_extend_*`` / ``_decode_*`` (detects profile_by_stage
       leaking through PROFILE_EXTRA_BODY)
    7. main trace has ``cpu_op`` / ``kernel`` events (a metadata-only trace
       means the profiler active window recorded nothing; triggers a roofline
       re-profile)

    Read-only; each check warns independently so partial signals stay actionable.

    Args:
        trace_dir: The profile workspace trace directory to inspect.
        framework: The framework name (e.g. ``"sglang"``) gating
            framework-specific checks.

    Returns:
        A ``trace_health`` dict with ``issues`` (logged warning strings),
        ``per_kernel_attribution_degraded`` (no ``execute_*``/
        ``user_annotation`` events -> per-kernel time folded, triggers an eager
        re-profile), ``capture_traces_present``, and ``zero_ops``
        (metadata-only trace, triggers a roofline re-profile).
    """
    issues: list[str] = []
    per_kernel_attribution_degraded = False
    capture_traces_present = False
    zero_ops = False

    # Scriptable image frameworks (xDiT diffusion) produce a plain torch-
    # profiler trace, so checks 1-6 (LLM/serving-specific) would emit spurious
    # warnings; only check 7 (zero_ops) is meaningful and always runs below.
    from hyperloom.inference_optimizer import framework_registry as _fw_reg

    # A roofline-composite ctx carries no framework, and an empty name resolves
    # to the serving default — which fires every serving-only check below against
    # a scriptable trace. $FRAMEWORK is the session-wide lock, so fall back to it.
    framework = str(framework or os.environ.get("FRAMEWORK", "") or "")
    scriptable = _fw_reg.is_scriptable(framework)

    # --- Check 1: capture_traces/ presence (LLM/serving only) ---
    capture = trace_dir / "capture_traces"
    capture_files: list[Path] = []
    if not scriptable:
        if not capture.is_dir():
            issues.append(
                "[1] capture_traces/ subdirectory missing — graph capture "
                "didn't fire. Verify EXTRA_VLLM_ARGS / EXTRA_SGLANG_ARGS "
                "include the TraceLens flag and the server-side patch landed."
            )
        else:
            capture_files = sorted(p for p in capture.iterdir() if p.is_file())
            capture_traces_present = bool(capture_files)
            if not capture_files:
                issues.append(
                    "[1] capture_traces/ exists but is empty — graph capture path fired but produced no files."
                )

    # --- Check 2 (Deval): capture file has cpu_op + Input Dims ---
    # Sample the heaviest capture file; gate cpu_op-with-Input-Dims fraction
    # at _INPUT_DIMS_FRACTION_FLOOR.
    if capture_files:
        target = max(capture_files, key=lambda p: p.stat().st_size)
        text = _sample_trace_text(target)
        if text is not None:
            cpu_op_count = _count_substring_occurrences(text, '"name": "cpu_op"')
            input_dims_count = _count_substring_occurrences(text, '"Input Dims"')
            if cpu_op_count == 0:
                # ROCm/SGLang often log graph-capture kernels under other names,
                # so zero cpu_op isn't itself a capture failure (cross-check [5]).
                issues.append(
                    f"[2] capture file {target.name} has no literal "
                    f"'cpu_op' events in the first "
                    f"{_TRACE_INSPECT_BYTES // 1_000_000} MB — on ROCm/SGLang "
                    "this is often just an event-naming difference (kernels "
                    "logged under 'sglang_profiler::*'); cross-check Check "
                    "[5] (kernel_shape_profiler) and the server log before "
                    "treating it as a capture regression."
                )
            elif input_dims_count / max(cpu_op_count, 1) < _INPUT_DIMS_FRACTION_FLOOR:
                pct = 100.0 * input_dims_count / cpu_op_count
                issues.append(
                    f"[2] capture file {target.name}: only {pct:.1f}% of "
                    f"cpu_op events carry 'Input Dims' (expected ≥ "
                    f"{int(_INPUT_DIMS_FRACTION_FLOOR * 100)}%). Shape-"
                    "discovery instrumentation may not be fully active — "
                    "verify TraceLens server patch and capture flag."
                )

    # --- Check 3 (Deval): main trace has user_annotation + execute_* ---
    # execute_* annotations = InferenceX per-step writes when
    # detailed_annotations is honoured (distinct from check 5).
    main_traces = sorted(
        (p for p in trace_dir.glob("*.trace.json.gz") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    main_text: str | None = None
    if main_traces:
        main_text = _sample_trace_text(main_traces[0])
        if main_text is not None:
            user_ann_count = _count_substring_occurrences(
                main_text,
                '"name": "user_annotation"',
            )
            execute_count = _count_substring_occurrences(main_text, '"execute_')
            # ``execute_*`` labels are the real health signal; ``user_annotation``
            # presence is profiler-version-dependent. Only warn when both are
            # absent, confirmed via a streaming scan (the 2 MB window can miss
            # markers on large traces).
            confirmed_absent = (
                not scriptable
                and execute_count == 0
                and user_ann_count == 0
                and not (
                    _trace_contains(main_traces[0], '"execute_')
                    or _trace_contains(main_traces[0], '"name": "user_annotation"')
                )
            )
            if confirmed_absent:
                per_kernel_attribution_degraded = True
                issues.append(
                    f"[3] main trace {main_traces[0].name} has no "
                    "execute_* / user_annotation events — InferenceX "
                    "per-step annotations didn't fire. Verify "
                    "detailed_annotations reached the framework "
                    "(PROFILE_EXTRA_BODY consumed; see #210)."
                )

    # --- Check 4 (Deval): per-file execute_* in trace_split/ ---
    # An empty split means the splitter ran but got no usable events.
    split = trace_dir / "trace_split"
    split_files: list[Path] = []
    if split.is_dir() and not scriptable:
        split_files = sorted(p for p in split.iterdir() if p.is_file())
        empty_splits: list[str] = []
        for sp in split_files:
            if not sp.name.endswith(".json.gz"):
                continue
            text = _sample_trace_text(sp)
            if text is None:
                continue
            if _count_substring_occurrences(text, '"execute_') == 0:
                empty_splits.append(sp.name)
        if split_files and empty_splits:
            issues.append(
                f"[4] {len(empty_splits)} trace_split/ file(s) have NO "
                "execute_* user_annotations: "
                f"{', '.join(empty_splits[:3])}"
                + (f" (and {len(empty_splits) - 3} more)" if len(empty_splits) > 3 else "")
                + " — splitter ran but the chunks are empty. Likely the "
                "trace lacks the per-step annotations needed for splitting "
                "(see check [3])."
            )

    # --- Check 6 (Hyperloom): _extend_* / _decode_* without ---
    # _steady_state_* in trace_split/.
    if split.is_dir() and not scriptable:
        names = [p.name for p in split_files]
        has_extend = any("_extend_" in n or "extend_only_" in n for n in names)
        has_decode = any("_decode_" in n or "decode_only_" in n for n in names)
        has_steady_state = any("steady_state" in n for n in names)
        if (has_extend or has_decode) and not has_steady_state:
            issues.append(
                "[6] trace_split/ has _extend_* / _decode_* files but NO "
                "_steady_state_* — profile_by_stage=True leaked through, "
                "PROFILE_EXTRA_BODY env was not consumed by the framework. "
                "Confirm _inferencex_patcher patched Magpie's bundled "
                "InferenceX (#210; check $MAGPIE_PATH/InferenceX/utils/"
                "bench_serving/benchmark_serving.py)."
            )

    # --- Check 7 (Hyperloom): torch-profiler captured zero ops ---
    # A metadata-only trace (no ``cpu_op`` / ``kernel`` events) means the
    # profiler active window never recorded real execution; flag it so roofline
    # re-profiles rather than caching an empty snapshot. ``"Op count"`` is 0 even
    # on healthy traces, so key on the presence of ``cpu_op`` / ``kernel`` events.
    if main_traces:
        has_ops = _trace_contains(main_traces[0], '"cat": "cpu_op"') or _trace_contains(
            main_traces[0], '"cat": "kernel"'
        )
        if not has_ops:
            zero_ops = True
            issues.append(
                f"[7] main trace {main_traces[0].name} has NO cpu_op / kernel "
                "events — the torch-profiler active window recorded nothing "
                "(metadata-only trace). On xDiT/diffusion this is the "
                "torch.profiler.schedule repeat=0 discard (the active window is "
                "dropped when the schedule restarts after the last active "
                "step). The trace is unusable for roofline; re-profile needed."
            )

    # --- Check 5 (Deval): sglang kernel_shape_profiler presence ---
    if framework.lower() == "sglang" and main_text is not None:
        if "kernel_shape_profiler" not in main_text:
            issues.append(
                f"[5] sglang main trace ({main_traces[0].name}, sampled "
                f"first {_TRACE_INSPECT_BYTES // 1_000_000} MB) lacks "
                "kernel_shape_profiler events — shape-discovery "
                "patch didn't reach the live SGLang. Verify "
                "_server_patcher (PR #207) succeeded for the "
                "deployed SGLang version (check log warnings)."
            )

    if issues:
        for issue in issues:
            log.warning("trace structure check: %s", issue)
        log.warning(
            "trace structure check: %d issue(s) detected — TraceLens "
            "downstream analysis may be degraded. See per-issue messages "
            "above for the actionable check.",
            len(issues),
        )
    return {
        "issues": issues,
        "per_kernel_attribution_degraded": per_kernel_attribution_degraded,
        "capture_traces_present": capture_traces_present,
        "zero_ops": zero_ops,
    }


# sglang profile yaml, used by tests/fixtures; runtime selection goes through
# `_default_profile_config()`.
PROFILE_DEFAULT_CONFIG = asset_root() / "assets" / "configs" / "profile_sglang.yaml"
PROFILE_DEFAULT_TIMEOUT_SEC = 14400  # 4 h wall cap


def _is_capture_trace(path: Path, root: Path | None = None) -> bool:
    """True when ``path`` is a CUDA-graph capture sidecar rather than a trace.

    Capture sidecars carry no per-iteration annotations, so the steady-state
    splitter cannot use them, yet a vLLM ``graph_capture_*.pt.trace.json.gz``
    also ends in ``.trace.json.gz`` and would otherwise be promoted as the
    primary annotated trace.

    Delegates to the shared classifier so this executor recognises the same
    layouts the kernel-agent routes do. The exact-``capture_traces`` test this
    replaced missed an unpatched SGLang's ``graph_capture_profile/``.

    Args:
        path: The candidate path.
        root: Directory to judge the path relative to, so an unrelated ancestor
            named after graph capture does not condemn everything beneath it.
    """
    return _shared_is_capture_fragment(path, root)


def _trace_size_bytes(path: Path) -> int:
    """Size in bytes, or 0 when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _is_split_chunk(path: Path, root: Path) -> bool:
    """True when ``path`` is steady-state splitter output below ``root``.

    Same reasoning as :func:`_is_capture_trace`, one directory along: the
    splitter's per-phase chunks also end in ``.trace.json.gz`` and would
    otherwise be mistaken for a real annotated trace. They are a few hundred
    bytes covering one phase of one iteration, and they sort ahead of
    ``rank_0.trace.json.gz`` alphabetically, so a caller falling back to
    ``trace_files[0]`` would analyse a sliver of the run.

    Relative to ``root`` rather than over the whole absolute path, so a capture
    that happens to live under some ancestor named ``trace_split`` does not have
    every one of its traces excluded.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part == "trace_split" for part in relative.parts)


def _trace_files_for_dir(trace_dir: Path) -> list[Path]:
    """Return annotated ``*.trace.json.gz`` files under ``trace_dir``.

    Excludes CUDA-graph capture sidecars (see :func:`_is_capture_trace`, which
    matches them by shape rather than by one directory name) so a vLLM
    ``graph_capture_*.pt.trace.json.gz`` or an unpatched SGLang's
    ``graph_capture_profile/cuda_graph_capture-*`` is never promoted as the
    primary annotated trace, and steady-state chunks under ``trace_split/`` for
    the same reason.

    Ordered largest first. Consumers fall back to ``trace_files[0]`` when
    ``main_trace_path`` is absent, and at that point a 938-byte chunk and a
    910 KB capture are indistinguishable by name -- alphabetical order picked the
    chunk. Size needs no naming rule to get this right.

    Args:
        trace_dir: The directory to scan recursively.

    Returns:
        ``*.trace.json.gz`` paths largest first, capture sidecars and split
        chunks removed.
    """
    candidates = [
        p
        for p in trace_dir.rglob("*.trace.json.gz")
        if not _is_capture_trace(p, trace_dir) and not _is_split_chunk(p, trace_dir)
    ]
    return sorted(candidates, key=lambda p: (-_trace_size_bytes(p), str(p)))


def _capture_sidecar_traces_for_dir(trace_dir: Path) -> list[Path]:
    """Return CUDA-graph capture sidecars under ``trace_dir`` (fallback only).

    Whatever :func:`_is_capture_trace` classifies as a sidecar: SGLang ``bs_*``
    under ``capture_traces/``, vLLM ``graph_capture_*``, and an unpatched
    SGLang's ``graph_capture_profile/cuda_graph_capture-*``.

    Args:
        trace_dir: The directory to scan recursively.

    Returns:
        Sorted capture sidecar paths found under ``trace_dir``.
    """
    return sorted(p for p in trace_dir.rglob("*.json.gz") if _is_capture_trace(p, trace_dir))


_TRACE_RANK_RE = re.compile(
    r"(?:^|[-_.])(?:tp|rank)[-_](\d+)(?=[-_.]|$)",
    re.IGNORECASE,
)


def _trace_rank(path: Path) -> int | None:
    """Return the rank encoded in a framework trace path, when present."""
    match = _TRACE_RANK_RE.search(path.name)
    if match:
        return int(match.group(1))
    parent_match = _TRACE_RANK_RE.fullmatch(path.parent.name)
    if parent_match:
        return int(parent_match.group(1))
    return None


def _record_trace_topology(result: dict[str, Any], trace_files: list[Path]) -> None:
    """Attach per-rank and merged trace paths to a profile result."""
    rank_paths: dict[str, list[str]] = {}
    for path in trace_files:
        rank = _trace_rank(path)
        if rank is not None:
            rank_paths.setdefault(str(rank), []).append(str(path))
    result["rank_trace_paths"] = rank_paths
    result["merged_trace_paths"] = [str(path) for path in trace_files if path.name.startswith("merged-")]


def _preferred_main_trace_path(
    trace_dir: Path,
    trace_files: list[Path],
    *,
    require_single_rank: bool = False,
    preferred_rank: int = 0,
    tensor_parallel_size: int | None = None,
) -> Path | None:
    """Trace path to pass downstream to TraceLens.

    AgentX inference splitting requires one rank timeline, so its path selects
    the largest raw trace for ``preferred_rank`` and never silently substitutes
    a multi-rank merge. Other workloads retain the existing merged/directory
    behavior.

    Args:
        trace_dir: The trace directory, returned as the fallback path.
        trace_files: Candidate trace files discovered under ``trace_dir``.

    Returns:
        The selected trace path, ``trace_dir`` for the legacy directory path,
        or ``None`` when AgentX requires a rank trace that is unavailable.
    """
    if require_single_rank:
        preferred = [path for path in trace_files if _trace_rank(path) == preferred_rank]
        if preferred:
            return max(preferred, key=_trace_size_bytes)

        non_merged = [path for path in trace_files if not path.name.startswith("merged-")]
        if tensor_parallel_size == 1 and len(non_merged) == 1:
            return non_merged[0]
        if tensor_parallel_size == 1 and not non_merged:
            merged = [path for path in trace_files if path.name.startswith("merged-")]
            if len(merged) == 1:
                return merged[0]
        return None

    merged = sorted(p for p in trace_files if p.name.startswith("merged-"))
    return merged[0] if merged else trace_dir


def _candidate_trace_dirs(workspace: Path) -> list[Path]:
    """Trace directories to probe for a Magpie profile workspace.

    Args:
        workspace (Path): The Magpie profile workspace directory.

    Returns:
        list[Path]: Candidate trace directories, in probe order.
    """
    return [
        workspace / "torch_trace",
        workspace / "capture_traces",
        workspace.parent / "capture_traces",
    ]


def _default_profile_config() -> Path:
    """Resolve default profile YAML from $FRAMEWORK (atom / vllm / sglang;
    unknown falls back to ``profile_sglang.yaml``).

    The atom branch is explicit because the materializer resolves Magpie's
    wrapper script from the YAML's ``benchmark.framework`` (not $FRAMEWORK);
    falling through to the sglang yaml on FRAMEWORK=atom would launch the
    wrong wrapper.

    Returns:
        The path to the framework-specific profile YAML config.
    """
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    if fw == "atom":
        name = "profile_atom.yaml"
    elif fw == "vllm":
        name = "profile_vllm.yaml"
    elif fw == "xdit":
        name = "profile_xdit.yaml"
    elif fw == "custom":
        name = "profile_custom.yaml"
    else:
        name = "profile_sglang.yaml"
    return asset_root() / "assets" / "configs" / name


class ProfileExecutor(BaselineExecutor):
    """Subclass that swaps the default config + extracts trace_dir."""

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_timeout_sec: int = PROFILE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str | None = None,
    ):
        """Initialize the profile executor with profile-specific defaults.

        Args:
            magpie_python (str | None): Python interpreter for the Magpie
                shell-out; ``None`` uses the base resolver.
            default_config_path (Path | str | None): Override config path;
                ``None`` defers to :meth:`_resolve_default_config`.
            session_dir (Path | str | None): Session output directory.
            default_timeout_sec (int): Wall-clock cap for the profile run.
                Defaults to :data:`PROFILE_DEFAULT_TIMEOUT_SEC`.
            cwd (Path | str): Working directory for the subprocess.
                Defaults to ``"/tmp"``.
        """
        super().__init__(
            magpie_python=magpie_python,
            default_config_path=default_config_path,
            session_dir=session_dir,
            default_timeout_sec=default_timeout_sec,
            cwd=cwd if cwd is not None else tempfile.gettempdir(),
        )
        # Set by ``_after_materialize_config`` once the probe is armed, read
        # after the run to aggregate the per-rank reports.
        self._host_probe_dir: str = ""
        # Non-empty only when arming failed, and then it carries why: an empty
        # probe dir alone cannot say whether the probe was never asked for or
        # could not be installed.
        self._host_probe_status: str = ""

    def _resolve_default_config(self) -> Path:
        """Override BaselineExecutor's resolver to pick the profile yaml.

        Returns:
            Path: The framework-specific profile YAML config path.
        """
        return _default_profile_config()

    def _resolve_mn_round_trace_root(self, ctx) -> str:
        """Return the shared torch-trace base dir for multi-node, or ''.

        Same base dir for every profile round (sglang's
        ``SGLANG_TORCH_PROFILER_DIR`` is pinned to it on first launch and
        never re-injected under the resume path); the ``__call__`` mtime gate
        isolates the current round's traces from earlier leftovers.

        Three-tier resolution (first non-empty wins):
        1. ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` env (in-process provision).
        2. State-file ``rayjob_id`` →
           ``<mn_profile_trace_root>/<rayjob>/torch_trace`` (out-of-band launches).
        3. ``<mn_profile_trace_root>/default-<pid>/torch_trace`` — pid-scoped
           last-resort so concurrent sandboxes never share a dir.

        The resolved dir is mkdir'd best-effort.

        Args:
            ctx: Action context (unused beyond multi-node detection).

        Returns:
            The resolved shared torch-trace base dir, or ``""`` when not
            running multi-node.
        """
        from ._multi_node_env import is_multi_node, rayjob_id_from_state

        if not is_multi_node():
            return ""
        provisioned = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
        if provisioned:
            return provisioned
        # Tier 2: derive from state-file rayjob_id (out-of-band launches).
        rid = rayjob_id_from_state()
        if rid:
            scoped = mn_profile_trace_root() / rid / "torch_trace"
        else:
            # Tier 3: pid-scoped last-resort.
            scoped = mn_profile_trace_root() / f"default-{os.getpid()}" / "torch_trace"
        try:
            scoped.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning(
                "cannot mkdir multi-node profile fallback dir %s: %s; downstream readers may FileNotFoundError",
                scoped,
                exc,
            )
        return str(scoped)

    def _inject_host_probe(self, config_path: Path, output_dir: Path) -> str:
        """Arm the host-side evidence probe in the materialized profile config.

        The probe is delivered by prepending its asset directory to the
        benchmark process's ``PYTHONPATH``, so CPython's ``sitecustomize``
        auto-import installs it without the framework's entrypoint knowing it
        exists. Editing the materialized YAML (rather than passing ``extra_envs``)
        is what makes the ``PYTHONPATH`` a *prefix*: ``extra_envs`` overrides, and
        replacing a framework's ``PYTHONPATH`` would break its imports.

        Args:
            config_path: The materialized profile YAML to edit in place.
            output_dir: The run workspace; the probe writes its per-rank reports
                into a subdirectory of it.

        Returns:
            The probe report directory, or ``""`` when the probe was not armed.
        """
        from . import _framework_rewrite_evidence as _evidence

        if not _evidence.probe_enabled():
            return ""
        asset_dir = _evidence.probe_asset_dir()
        if not asset_dir.is_dir():
            log.warning(
                "profile_executor: host-probe assets missing at %s; host-side rewrite evidence disabled",
                asset_dir,
            )
            return ""
        probe_dir = output_dir / _evidence.PROBE_SUBDIR
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("profile_executor: cannot create host-probe dir %s: %s", probe_dir, exc)
            return ""

        from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist

        try:
            roots = list(resolve_source_file_allowlist())
        except Exception:  # noqa: BLE001 - attribution is advisory
            roots = []
        probe_env = _evidence.build_probe_env(
            probe_dir=probe_dir,
            source_roots=roots,
            deep=_evidence.deep_probe_enabled(),
        )
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning("profile_executor: cannot read %s to arm the host probe: %s", config_path, exc)
            return ""
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else None
        if not isinstance(bench, dict):
            return ""
        envs = bench.setdefault("envs", {})
        if not isinstance(envs, dict):
            return ""
        current = str(envs.get("PYTHONPATH", "") or "").strip()
        entry = str(asset_dir)
        if entry not in current.split(os.pathsep):
            envs["PYTHONPATH"] = f"{entry}{os.pathsep}{current}" if current else entry
        for key, value in probe_env.items():
            envs[key] = value
        try:
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except OSError as exc:
            log.warning("profile_executor: cannot write %s after arming the host probe: %s", config_path, exc)
            return ""
        log.info(
            "profile_executor: host-side rewrite evidence probe armed (deep=%s), reports -> %s",
            bool(probe_env.get("HYPERLOOM_HOST_PROBE_DEEP")),
            probe_dir,
        )
        return str(probe_dir)

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Arm the host probe, then patch the InferenceX checkout Magpie will execute.

        `$INFERENCEX_PATH` alone is insufficient (Magpie resolves an empty
        `benchmark.inferencex_path` to its own sibling checkout); patch the
        path resolved from the materialized YAML so NUM_PROMPTS /
        PROFILE_EXTRA_BODY aren't applied to a different checkout.

        Args:
            config_path: The materialized profile YAML config to read.
            output_dir: The run output directory, also the probe report root.

        Returns:
            ``None`` when the InferenceX checkout is patched correctly,
            otherwise a failure result dict describing the patch gap.
        """
        try:
            self._host_probe_dir = self._inject_host_probe(config_path, output_dir)
            self._host_probe_status = ""
        except Exception as exc:  # noqa: BLE001 - evidence collection is never fatal
            log.warning("profile_executor: host-probe injection failed: %s", exc, exc_info=True)
            self._host_probe_dir = ""
            self._host_probe_status = f"probe_injection_failed: {exc}"
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "error_class": "profile_config_unreadable",
                "error": f"cannot read materialized profile config {config_path}: {exc}",
            }
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
        framework = ""
        if isinstance(bench, dict):
            framework = str(bench.get("framework") or "").strip().lower()
        # Scriptable diffusion (xDiT) has no InferenceX server; it profiles via
        # its own torch.profiler schedule. Patch it to retain the active window
        # (upstream default repeat=0 discards it -> empty trace) and skip the
        # InferenceX NUM_PROMPTS / PROFILE_EXTRA_BODY validation below.
        from hyperloom.inference_optimizer import framework_registry

        if framework_registry.is_scriptable(framework):
            # The baked-profiler verifier is xDiT/xfuser-specific (it inspects
            # xfuser's base_model.py). Other scriptable frameworks (e.g. an
            # operator's ``custom`` workload) share the server-less early-return
            # but must not trigger the xDiT check.
            if str(framework or "").strip().lower() == "xdit":
                verify_xdit_profiler_baked()
            return None
        inferencex_path = ""
        if isinstance(bench, dict):
            inferencex_path = str(bench.get("inferencex_path") or "").strip()
        if not inferencex_path:
            inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
        if not inferencex_path:
            log.warning(
                "profile_executor: no benchmark.inferencex_path / "
                "INFERENCEX_PATH configured; skipping InferenceX profile "
                "patch validation"
            )
            return None

        ix_root = Path(inferencex_path)
        lib_ok = ensure_benchmark_lib_patched(ix_root)
        ensure_benchmark_lib_eval_dest_patched(ix_root)
        serving_ok = ensure_benchmark_serving_patched(ix_root)
        lib_path = ix_root / "benchmarks" / "benchmark_lib.sh"
        serving_path = ix_root / "utils" / "bench_serving" / "benchmark_serving.py"

        def _contains(path: Path, needle: str) -> bool:
            """Check whether ``needle`` appears in ``path``'s text.

            Args:
                path (Path): File to read.
                needle (str): Substring to search for.

            Returns:
                bool: ``True`` if found; ``False`` on miss or read error.
            """
            try:
                return needle in path.read_text(encoding="utf-8")
            except OSError:
                return False

        lib_valid = _contains(lib_path, "${NUM_PROMPTS:-$max_concurrency}")
        serving_valid = _contains(serving_path, "PROFILE_EXTRA_BODY")
        if not (lib_ok and serving_ok and lib_valid and serving_valid):
            return {
                "status": "failed",
                "error_class": "profile_inferencex_patch_failed",
                "error": (
                    "profile requires InferenceX to honour NUM_PROMPTS and "
                    "PROFILE_EXTRA_BODY, but the checkout Magpie will use is "
                    f"not patched: inferencex_path={ix_root}, "
                    f"benchmark_lib_ok={lib_ok}/{lib_valid}, "
                    f"benchmark_serving_ok={serving_ok}/{serving_valid}"
                ),
                "inferencex_path": str(ix_root),
                "benchmark_lib": str(lib_path),
                "benchmark_serving": str(serving_path),
            }
        return None

    def _collect_rewrite_evidence(self, result: dict[str, Any]) -> None:
        """Merge the per-rank host-probe reports onto ``result``.

        Adds ``framework_rewrite_evidence`` (the document path) and
        ``framework_rewrite_candidate_count`` when candidates were found, and
        always sets ``framework_rewrite_evidence_status``. Never raises:
        evidence is an input to the next optimization decision, not a
        precondition for reporting this profile. It does not fail *silently*
        either — "the probe broke" and "this workload has nothing left to
        rewrite" both end with no document, and the phase that consumes the
        evidence has to be able to tell those apart.

        Args:
            result: The profile result dict, mutated in place.
        """
        probe_dir = str(self._host_probe_dir or "").strip()
        if not probe_dir:
            result["framework_rewrite_evidence_status"] = self._host_probe_status or "probe_not_armed"
            return
        from . import _framework_rewrite_evidence as _evidence

        try:
            out_path = Path(probe_dir).parent / _evidence.EVIDENCE_FILENAME
            document = _evidence.aggregate_probe_dir(probe_dir, out_path)
        except Exception as exc:  # noqa: BLE001 - aggregation is best-effort
            log.warning(
                "profile_executor: rewrite-evidence aggregation failed for %s: %s",
                probe_dir,
                exc,
                exc_info=True,
            )
            result["framework_rewrite_evidence_status"] = f"aggregation_failed: {exc}"
            return
        candidates = document.get("candidates") or []
        if not candidates:
            log.info(
                "profile_executor: host probe produced no rewrite candidates (ranks_merged=%s); see %s for why",
                document.get("ranks_merged"),
                out_path,
            )
            result["framework_rewrite_evidence_status"] = "no_candidates"
            return
        result["framework_rewrite_evidence"] = str(out_path)
        result["framework_rewrite_candidate_count"] = len(candidates)
        result["framework_rewrite_evidence_status"] = "ok"
        log.info(
            "profile_executor: %d host-side rewrite candidate(s) from %s rank(s) -> %s",
            len(candidates),
            document.get("ranks_merged"),
            out_path,
        )

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the profiling action for the given context.

        Launches a profiling run (sglang/vllm or the Magpie atom path),
        merging the current-best server args with caller params, and
        returns the captured trace artifacts.

        Args:
            ctx: Action context carrying the task and its parameters.

        Returns:
            A result dict describing the profiling outcome and artifacts.
        """
        # atom: the Magpie atom wrapper bridges PROFILE=1 to atom's
        # --torch-profiler-dir and writes standard *.pt.trace.json.gz, so the
        # executor falls through to the sglang/vllm path.
        params = ctx.task.params or {}
        # Merge current_best.extra_server_args (stamped into base_extra_args) with
        # caller args, dropping compile flags that break profiling.
        base_args = _sanitize_profile_server_args(
            str(params.get("base_extra_args") or "").strip(),
        )
        caller_args = _sanitize_profile_server_args(str(params.get("extra_server_args") or ""))
        from ._grid_runner import merge_server_args

        merged_args = merge_server_args(base_args, caller_args)
        if merged_args:
            params["extra_server_args"] = merged_args
        else:
            params.pop("extra_server_args", None)
        base_envs = params.get("base_extra_envs")
        caller_envs = params.get("extra_envs")
        merged_envs: dict[str, Any] = {}
        if isinstance(base_envs, dict):
            merged_envs.update(base_envs)
        if isinstance(caller_envs, dict):
            merged_envs.update(caller_envs)
        if merged_envs:
            params["extra_envs"] = merged_envs
        else:
            params.pop("extra_envs", None)
        if params.get("base_remove_args") and "remove_args" not in params:
            raw_remove = params.get("base_remove_args")
            params["remove_args"] = [raw_remove] if isinstance(raw_remove, str) else list(raw_remove or [])
        if params.get("base_unset_envs") and "unset_envs" not in params:
            raw_unset = params.get("base_unset_envs")
            params["unset_envs"] = [raw_unset] if isinstance(raw_unset, str) else list(raw_unset or [])
        if str(params.get("base_args_mode") or "").strip().lower() == "replace":
            params.setdefault("args_mode", "replace")
        extra = getattr(ctx, "extra", None) or {}
        if not (params.get("output_dir") or extra.get("workspace")):
            output_dir = self._resolve_workspace(ctx, "profile")
            output_dir.mkdir(parents=True, exist_ok=True)
            # Stash so BaselineExecutor.__call__ picks it up via ctx.extra.
            if extra is None:
                ctx.extra = {"workspace": str(output_dir)}
                extra = ctx.extra
            else:
                extra["workspace"] = str(output_dir)

        # Mtime gate for the multi-node shared-trace-dir layout: captured before
        # super().__call__ so this round's traces are newer than the watermark.
        import time as _time

        task_started_unix = _time.time()

        # Multi-node banner (silent for single-node) surfacing the round's dir.
        from ._multi_node_env import log_mn_banner

        log_mn_banner(
            "profile_executor",
            log,
            trace_dir=self._resolve_mn_round_trace_root(ctx),
        )

        # Multi-node only: pre-restart the server with this round's profiler
        # dir, marking ``ctx.extra['mn_round_restarted']`` so BaselineExecutor
        # skips a second restart. No-op in single-node.
        round_trace_root = self._resolve_mn_round_trace_root(ctx)
        if round_trace_root:
            from ._multi_node_server_lifecycle import (
                ServerRestartFailed,
                restart_server_for_round,
            )

            try:
                # PD knobs auto-resolved from $PD_* env (see baseline.py).
                await restart_server_for_round(
                    extra_server_args=str(params.get("extra_server_args") or ""),
                    torch_profiler_dir=round_trace_root,
                    framework=os.environ.get("FRAMEWORK") or None,
                    model_path=(str(params.get("model_path") or "").strip() or os.environ.get("MODEL_PATH") or None),
                    tp=int(os.environ.get("TP") or 0) or None,
                    ep=int(os.environ.get("EP") or 0) or None,
                )
            except ServerRestartFailed as exc:
                return {
                    "status": "failed",
                    "error_class": "mn_server_restart_failed",
                    "error": str(exc),
                    "trace_dir": round_trace_root,
                }
            if isinstance(extra, dict):
                extra["mn_round_restarted"] = True

        # InferenceX patching (``ensure_benchmark_lib_patched`` /
        # ``ensure_benchmark_serving_patched``) happens in the
        # ``_after_materialize_config`` hook, which covers the exact InferenceX
        # checkout Magpie will execute.
        # Multi-node infera only: the Infera frontend does not propagate
        # /start_profile to the SSH-launched disagg workers, so torch
        # profiling must be triggered directly on each worker's system
        # server (/engine/start_profile). Bracket the Magpie benchmark with
        # start/stop so traces land in the shared-FS round trace dir for
        # TraceLens. The helper no-ops for RayJob / single-node and is
        # fail-soft per worker. ``PROFILE_EXTRA_BODY`` (start_step/num_steps
        # computed by _workload_envs) is the start_profile payload.
        from ._multi_node_server_lifecycle import trigger_infera_engine_profile

        prof_body: dict[str, Any] = {}
        try:
            import json as _json

            parsed = _json.loads(os.environ.get("PROFILE_EXTRA_BODY") or "{}")
            if isinstance(parsed, dict):
                prof_body = parsed
        except (ValueError, TypeError):
            prof_body = {}
        # The sglang disaggregated scheduler crashes
        # (``TypeError: unsupported operand type(s) for +=: 'NoneType' and
        # 'int'`` -> SIGQUIT, server disconnects, no trace) when
        # start_profile carries the step-window / stage-split params that
        # the single-node InferenceX PROFILE_EXTRA_BODY normally sets
        # (``profile_by_stage`` / ``merge_profiles`` / ``num_steps`` /
        # ``start_step``). Isolated reproduction: ``output_dir``-only =
        # 8 traces written cleanly; any of the step/stage params = scheduler
        # crash. So the infera engine-route path forwards ONLY ``output_dir``
        # and bounds the trace by the start/stop wall-clock window instead.
        _SAFE_PROFILE_KEYS = ("output_dir",)
        prof_body = {k: v for k, v in prof_body.items() if k in _SAFE_PROFILE_KEYS}
        # Pin the trace output dir explicitly: the disagg workers may not carry
        # SGLANG_TORCH_PROFILER_DIR, so without output_dir sglang writes nowhere
        # the sandbox can read. ``round_trace_root`` is the shared-FS dir the
        # post-bench trace scan reads.
        if round_trace_root:
            prof_body.setdefault("output_dir", round_trace_root)
        # Bounded profiling window. Open-ended profiling for the whole
        # Magpie run overflows the disagg scheduler's in-memory trace and
        # crashes stop_profile ("Server disconnected", no trace) — verified:
        # a 4s window writes 8 traces cleanly, a ~10min full-run window
        # crashes the worker. num_steps would bound it but crashes the
        # disagg scheduler too. So bound by WALL-CLOCK: after a warmup delay
        # (let load reach steady state) profile a short fixed window
        # concurrently with the full Magpie run (which still produces the
        # throughput number), then stop. ``trigger_infera_engine_profile``
        # no-ops for RayJob / single-node, so this is multi-node-infera only.
        import asyncio as _asyncio

        warmup_s = float(os.environ.get("HYPERLOOM_MN_PROFILE_WARMUP_S", "60") or 60)
        window_s = float(os.environ.get("HYPERLOOM_MN_PROFILE_WINDOW_S", "8") or 8)
        _prof_started = {"v": False}

        async def _bounded_profile_window() -> None:
            """Run a warmup-then-bounded engine profiling window.

            Sleeps for the warmup period, starts engine profiling, holds it
            open for the configured window, then stops it; updates the shared
            ``_prof_started`` flag around the active window.
            """
            await _asyncio.sleep(warmup_s)
            await trigger_infera_engine_profile("start", prof_body)
            _prof_started["v"] = True
            await _asyncio.sleep(window_s)
            await trigger_infera_engine_profile("stop")
            _prof_started["v"] = False

        prof_task = _asyncio.create_task(_bounded_profile_window())
        try:
            result = await super().__call__(ctx)
        finally:
            # Magpie ended (or raised): wind the window task down so profiling
            # is never left running open-ended.
            if not prof_task.done():
                prof_task.cancel()
                try:
                    await prof_task
                except (_asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if _prof_started.get("v"):
                # start fired but stop didn't (window task cancelled mid-run
                # because Magpie finished first) -> ensure a matching stop.
                await trigger_infera_engine_profile("stop")

        # Merge the per-rank host-probe reports into the rewrite-evidence
        # document. Independent of the trace path below: the two answer different
        # questions (which kernel is hot vs. which host-side work is redundant),
        # and a run that produced no trace can still have produced good evidence.
        self._collect_rewrite_evidence(result)

        # Augment with trace_dir. Multi-node: traces live at the round-scoped
        # wekafs dir we restarted with. Single-node uses workspace/torch_trace.
        workspace_str = result.get("workspace")
        capture_status: dict[str, Any] | None = None
        if workspace_str:
            capture_status_path = Path(workspace_str) / "agentx_profile_capture.json"
            if capture_status_path.is_file() and safe_mtime(capture_status_path) >= int(task_started_unix):
                try:
                    parsed_capture_status = json.loads(capture_status_path.read_text(encoding="utf-8"))
                    if isinstance(parsed_capture_status, dict):
                        capture_status = parsed_capture_status
                        result["trace_capture_status"] = str(parsed_capture_status.get("status") or "")
                        result["trace_capture"] = parsed_capture_status
                        result["trace_capture_status_path"] = str(capture_status_path)
                except (OSError, ValueError, TypeError) as exc:
                    capture_status = {
                        "status": "failed",
                        "reason": "capture_status_unreadable",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    result["trace_capture_status"] = "failed"
                    result["trace_capture"] = capture_status
            elif capture_status_path.is_file():
                log.warning(
                    "profile_executor: ignoring stale AgentX capture status %s (mtime %.0f < round start second %d)",
                    capture_status_path,
                    safe_mtime(capture_status_path),
                    int(task_started_unix),
                )
        from ._workload_envs import agentx_enabled

        agentx_profile = capture_status is not None or "submission_valid" in result or agentx_enabled()
        try:
            tensor_parallel_size = int(os.environ.get("TP") or 0) or None
        except (TypeError, ValueError):
            tensor_parallel_size = None
        if round_trace_root:
            # Multi-node: traces land at the shared wekafs base dir (not the
            # workspace-local ``_candidate_trace_dirs``). Mtime-gate to files
            # at-or-after this round's start, else we pick up round 1's trace.
            trace_dir = Path(round_trace_root)
            if trace_dir.is_dir():
                all_files = sorted(trace_dir.glob("*.trace.json.gz"))
                trace_files = [p for p in all_files if safe_mtime(p) >= task_started_unix]
                result["trace_dir"] = str(trace_dir)
                result["trace_files"] = [str(p) for p in trace_files]
                _record_trace_topology(result, trace_files)
                if trace_files:

                    def _safe_size(p: Path) -> int:
                        """Return ``p``'s size in bytes, or 0 on stat() failure.

                        Args:
                            p (Path): Path to stat.

                        Returns:
                            int: The file size in bytes, or ``0`` if ``stat()``
                            fails.
                        """
                        try:
                            return p.stat().st_size
                        except OSError:
                            return 0

                    if agentx_profile:
                        main_trace = _preferred_main_trace_path(
                            trace_dir,
                            trace_files,
                            require_single_rank=True,
                            tensor_parallel_size=tensor_parallel_size,
                        )
                        if main_trace is not None:
                            result["main_trace_path"] = str(main_trace)
                            result["primary_rank"] = _trace_rank(main_trace)
                            result["profile_trace_selection_reason"] = "primary_rank_trace"
                    else:
                        # The shared round dir can contain a small warmup
                        # capture beside the real GPU-rich trace.
                        main_trace = max(trace_files, key=_safe_size)
                        result["main_trace_path"] = str(main_trace)
                    log.info(
                        "profile_executor: multi-node main trace selected: %s (%d candidate traces this round)",
                        main_trace.name if main_trace is not None else "<none>",
                        len(trace_files),
                    )
                elif all_files:
                    log.warning(
                        "profile_executor: multi-node trace dir %s has "
                        "%d historical trace(s) but none with mtime >= "
                        "%.0f (this round's start); sglang may have "
                        "skipped /start_profile or the trace flush is "
                        "lagging",
                        trace_dir,
                        len(all_files),
                        task_started_unix,
                    )
                else:
                    log.warning(
                        "profile_executor: multi-node trace dir %s exists "
                        "but no .trace.json.gz files found yet (server "
                        "pods may still be flushing)",
                        trace_dir,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                log.warning(
                    "profile_executor: round trace dir %s does not exist "
                    "after magpie completed; check sglang server logs for "
                    "torch profiler errors",
                    round_trace_root,
                )
        elif workspace_str:
            # Single-node branch: multi-candidate trace discovery.
            workspace = Path(workspace_str)
            selected_trace_dir: Path | None = None
            selected_trace_files: list[Path] = []
            existing_empty_dirs: list[Path] = []
            capture_only = False
            for trace_dir in _candidate_trace_dirs(workspace):
                if not trace_dir.is_dir():
                    continue
                trace_files = _trace_files_for_dir(trace_dir)
                if trace_files:
                    selected_trace_dir = trace_dir
                    selected_trace_files = trace_files
                    break
                existing_empty_dirs.append(trace_dir)

            # SGLang can emit only capture sidecars without a top-level
            # *.trace.json.gz; fall back to those so roofline analyzes the
            # available trace instead of failing with no_trace_files.
            if selected_trace_dir is None:
                for trace_dir in _candidate_trace_dirs(workspace):
                    if not trace_dir.is_dir():
                        continue
                    sidecars = _capture_sidecar_traces_for_dir(trace_dir)
                    if sidecars:
                        selected_trace_dir = trace_dir
                        selected_trace_files = sidecars
                        capture_only = True
                        break

            if selected_trace_dir is not None:
                result["trace_dir"] = str(selected_trace_dir)
                result["trace_files"] = [str(p) for p in selected_trace_files]
                _record_trace_topology(result, selected_trace_files)
                if capture_only:
                    # Pass the dir so TraceLens picks its own ingest order over
                    # the capture sidecars (it accepts *.json.gz).
                    main_trace = selected_trace_dir
                    result["profile_trace_selection_reason"] = "capture_only_fallback"
                    log.info(
                        "profile_executor: no *.trace.json.gz; falling back to "
                        "%d SGLang capture sidecar(s) in %s (#575)",
                        len(selected_trace_files),
                        selected_trace_dir,
                    )
                else:
                    main_trace = _preferred_main_trace_path(
                        selected_trace_dir,
                        selected_trace_files,
                        require_single_rank=agentx_profile,
                        tensor_parallel_size=tensor_parallel_size,
                    )
                    if main_trace is not None:
                        if agentx_profile:
                            result["primary_rank"] = _trace_rank(main_trace)
                            result["profile_trace_selection_reason"] = "primary_rank_trace"
                        else:
                            result["profile_trace_selection_reason"] = (
                                "merged_trace_preferred"
                                if main_trace.name.startswith("merged-")
                                else "trace_dir_preferred"
                            )
                if main_trace is not None:
                    result["main_trace_path"] = str(main_trace)
                # Warn if the trace shape suggests PROFILE_EXTRA_BODY leaked /
                # shape-discovery missing. Read-only; never blocks.
                try:
                    framework = str(
                        getattr(ctx, "framework", "")
                        or (extra.get("framework") if isinstance(extra, dict) else "")
                        or ""
                    )
                    health = _validate_trace_structure(selected_trace_dir, framework)
                    if isinstance(health, dict):
                        result["trace_health"] = health
                except Exception as e:  # noqa: BLE001 - validator is best-effort
                    log.debug(
                        "profile_executor: trace structure validator failed: %s",
                        e,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                result["status"] = "failed"
                result["error_class"] = "no_trace_files"
                probed = ", ".join(str(p) for p in _candidate_trace_dirs(workspace))
                result["error"] = f"no .trace.json.gz or capture sidecar under {workspace_str} (probed: {probed})"
                if existing_empty_dirs:
                    log.warning(
                        "profile_executor: trace dirs exist but no .trace.json.gz "
                        "or capture sidecar (bs_*_rank*.json.gz) files in %s",
                        ", ".join(str(p) for p in existing_empty_dirs),
                    )
                else:
                    log.warning(
                        "profile_executor: workspace=%s has no trace dir (checked: %s)",
                        workspace_str,
                        ", ".join(str(p) for p in _candidate_trace_dirs(workspace)),
                    )
        if result.get("main_trace_path"):
            result["trace_input_ready"] = True
        if capture_status is not None and str(capture_status.get("status") or "") != "succeeded":
            result["trace_input_ready"] = False
            result["status"] = "failed"
            result["error_class"] = "profile_capture_failed"
            result["error"] = (
                f"AgentX trace capture failed: {capture_status.get('reason') or 'unknown capture failure'}"
            )
        elif agentx_profile and result.get("trace_files") and not result.get("main_trace_path"):
            result["trace_input_ready"] = False
            result["status"] = "failed"
            result["error_class"] = "primary_rank_trace_missing"
            result["error"] = (
                "AgentX profiling produced trace files but no rank-0 raw trace "
                "could be identified; refusing a merged multi-rank fallback"
            )
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_CONFIG",
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]
