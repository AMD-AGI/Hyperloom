# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Real ``profile`` ActionRunner — Magpie run with torch profiler on.

Reuses the BaselineExecutor shell-out machinery; only the YAML differs
(``profiler.torch_profiler.enabled: true``), so Magpie writes trace files
under ``torch_trace/`` (or ``capture_traces/`` for TraceLens vLLM capture).

Result schema (delivered on the bus as ``delegated_result``)::

    status:        "succeeded" | "failed"
    framework:     "sglang"
    model:         path
    request/output/total_token_throughput, latency stats (same as baseline)
    workspace:     absolute path of the Magpie workspace
    trace_dir:     absolute path of the torch_trace dir (or None)
    report_path:   absolute path of benchmark_report.json

Downstream (Kernel agent → tracelens_analysis.py) needs ``trace_dir``; the
rest is surfaced so the baseline SharedState promotion works unchanged.
"""

from __future__ import annotations

import gzip
import logging
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

from ...compat.payload_aliases import read_extra_server_args
from ...paths import asset_root, mn_profile_trace_root
from ._inferencex_patcher import (
    ensure_benchmark_lib_patched,
    ensure_benchmark_serving_patched,
)
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


# Leading bytes of a trace to sample for sentinel substrings (full
# decompress would burn tens of seconds on 100 MB+ traces; 2 MB catches
# marker presence/absence).
_TRACE_INSPECT_BYTES = 2_000_000

# Cap for the confirmation streaming scan used when the leading-window
# sample finds zero of a sentinel; ``execute_*`` annotations land past the
# 2 MB window on 600 MB+ traces, so confirm absence before warning. Override
# via ``INFERENCE_OPTIMIZER_TRACE_CONFIRM_BYTES``.
_TRACE_CONFIRM_BYTES = 64_000_000

# Min fraction of ``cpu_op`` events carrying ``Input Dims`` for a healthy
# ``capture_traces/`` file (Deval ref 99.97%; gated low to avoid false-positives).
_INPUT_DIMS_FRACTION_FLOOR = 0.90

_PROFILE_UNSAFE_BOOL_FLAGS = frozenset({
    "--enable-torch-compile",
})
_PROFILE_UNSAFE_VALUE_FLAGS = frozenset({
    "--torch-compile-max-bs",
})


def _sanitize_profile_server_args(args: str) -> str:
    """Drop server flags known to conflict with profiler/shape discovery."""
    raw = str(args or "").strip()
    if not raw:
        return ""
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    out: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in _PROFILE_UNSAFE_BOOL_FLAGS:
            continue
        if token in _PROFILE_UNSAFE_VALUE_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in _PROFILE_UNSAFE_VALUE_FLAGS):
            continue
        out.append(token)
    return shlex.join(out)


def _trace_contains(path: Path, substring: str, max_bytes: int | None = None) -> bool:
    """Stream-decompress ``path`` for ``substring``, reading at most
    ``max_bytes`` (default :data:`_TRACE_CONFIRM_BYTES`).

    Confirmation pass when :func:`_sample_trace_text` finds zero
    occurrences. Returns ``False`` on any IO/decode error (never raises).
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
                # Tail to catch a sentinel split across the chunk boundary.
                carry = chunk[-(len(substring)):]
    except (OSError, EOFError, UnicodeDecodeError) as e:
        log.debug("_trace_contains: cannot stream %s: %s", path, e)
        return False
    return False


def _sample_trace_text(path: Path) -> str | None:
    """Read up to ``_TRACE_INSPECT_BYTES`` of decompressed text from a
    gzipped trace. Returns ``None`` (debug-logged) on IO/decode error so the
    check is skipped rather than failing the profile path."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read(_TRACE_INSPECT_BYTES)
    except (OSError, EOFError, UnicodeDecodeError) as e:
        # Best-effort: a malformed sample must not fail the profile path.
        log.debug(
            "_validate_trace_structure: cannot sample %s: %s", path, e,
        )
        return None


def _count_substring_occurrences(text: str, substring: str) -> int:
    """Count non-overlapping ``substring`` occurrences as a cheap
    lower-bound event count (avoids full JSON parsing)."""
    if not substring:
        return 0
    return text.count(substring)


def _validate_trace_structure(
    trace_dir: Path, framework: str, expected_pieces: int = 1,
) -> dict[str, Any]:
    """Post-profile sanity check (#210 / Deval's ``check_torch_trace.py``).

    Logs warnings (never raises) when the trace structure suggests
    TraceLens features didn't reach the framework. Mirrors the 5
    check_torch_trace.py checks plus one Hyperloom-specific check:

    1. ``capture_traces/`` exists with files (graph capture fired)
    2. capture files contain ``cpu_op`` events with ``Input Dims``
       (shape-discovery instrumentation recording per-event shapes)
    3. main-directory trace has ``user_annotation`` events including
       ``execute_*`` annotations (InferenceX per-step instrumentation
       fired — relates to ``roofline_annotations`` flag landing)
    4. ``trace_split/`` per-file ``execute_*`` user_annotations
       counted (splitter ran AND each split is non-empty)
    5. (sglang only) main trace contains ``kernel_shape_profiler``
       substring (server-side patch from PR #207 landed)
    6. (Hyperloom-specific) ``trace_split/`` has ``_steady_state_*``
       files, NOT ``_extend_*`` / ``_decode_*`` (the #210 smoking-
       gun: profile_by_stage leaked through PROFILE_EXTRA_BODY)

    Read-only; each check warns independently so partial signals stay actionable.

    Returns a structured ``trace_health`` dict (#431): ``per_kernel_attribution_degraded`` (no execute_*/user_annotation events -> cuda-graph folds per-kernel time, 0 hot kernels -> triggers eager re-profile), ``capture_traces_present``, and ``issues`` (logged warning strings).
    """
    issues: list[str] = []
    per_kernel_attribution_degraded = False
    capture_traces_present = False

    # --- Check 1: capture_traces/ presence ---
    capture = trace_dir / "capture_traces"
    capture_files: list[Path] = []
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
                "[1] capture_traces/ exists but is empty — graph capture "
                "path fired but produced no files."
            )

    # --- Check 2 (Deval): capture file has cpu_op + Input Dims ---
    # Sample the heaviest capture file; gate cpu_op-with-Input-Dims fraction
    # at _INPUT_DIMS_FRACTION_FLOOR (Deval ref 99.97%).
    if capture_files:
        target = max(capture_files, key=lambda p: p.stat().st_size)
        text = _sample_trace_text(target)
        if text is not None:
            cpu_op_count = _count_substring_occurrences(text, '"name": "cpu_op"')
            input_dims_count = _count_substring_occurrences(text, '"Input Dims"')
            if cpu_op_count == 0:
                # ROCm/SGLang often log graph-capture kernels under other
                # names (e.g. ``sglang_profiler::*``), so zero cpu_op isn't
                # itself a capture failure — informational, cross-check [5].
                issues.append(
                    f"[2] capture file {target.name} has no literal "
                    f"'cpu_op' events in the first "
                    f"{_TRACE_INSPECT_BYTES//1_000_000} MB — on ROCm/SGLang "
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
    # roofline_annotations is honoured; a separate failure mode from
    # kernel_shape_profiler absence (check 5).
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
                main_text, '"name": "user_annotation"',
            )
            execute_count = _count_substring_occurrences(main_text, '"execute_')
            # ``execute_*`` labels are the real health signal (the splitter +
            # roofline consume them); ``user_annotation`` wrapper presence is
            # profiler-version-dependent. Only warn when both are absent, and
            # confirm via a streaming scan (the 2 MB window misses markers on
            # 600 MB+ traces) to avoid false "annotations didn't fire" warnings.
            confirmed_absent = (
                execute_count == 0
                and user_ann_count == 0
                and not (
                    _trace_contains(main_traces[0], '"execute_')
                    or _trace_contains(
                        main_traces[0], '"name": "user_annotation"'
                    )
                )
            )
            if confirmed_absent:
                per_kernel_attribution_degraded = True
                issues.append(
                    f"[3] main trace {main_traces[0].name} has no "
                    "execute_* / user_annotation events — InferenceX "
                    "per-step annotations didn't fire. Verify "
                    "roofline_annotations reached the framework "
                    "(PROFILE_EXTRA_BODY consumed; see #210)."
                )

    # --- Check 4 (Deval): per-file execute_* in trace_split/ ---
    # An empty split means the splitter ran but got no usable events.
    split = trace_dir / "trace_split"
    split_files: list[Path] = []
    if split.is_dir():
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

    # --- Check 6 (Hyperloom-specific #210 smoking-gun): _extend_* / ---
    # _decode_* without _steady_state_* in trace_split/.
    if split.is_dir():
        names = [p.name for p in split_files]
        has_extend = any("_extend_" in n or "extend_only_" in n for n in names)
        has_decode = any("_decode_" in n or "decode_only_" in n for n in names)
        has_steady_state = any("steady_state" in n for n in names)
        # Failure mode is per-step ``_extend_*`` / ``_decode_*`` files
        # without any ``steady_state`` marker.
        if (has_extend or has_decode) and not has_steady_state:
            issues.append(
                "[6] trace_split/ has _extend_* / _decode_* files but NO "
                "_steady_state_* — profile_by_stage=True leaked through, "
                "PROFILE_EXTRA_BODY env was not consumed by the framework. "
                "Confirm _inferencex_patcher patched Magpie's bundled "
                "InferenceX (#210; check $MAGPIE_DIR/InferenceX/utils/"
                "bench_serving/benchmark_serving.py)."
            )

    # --- Check 5 (Deval): sglang kernel_shape_profiler presence ---
    if framework.lower() == "sglang" and main_text is not None:
        if "kernel_shape_profiler" not in main_text:
            issues.append(
                f"[5] sglang main trace ({main_traces[0].name}, sampled "
                f"first {_TRACE_INSPECT_BYTES//1_000_000} MB) lacks "
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
            "above for the actionable check.", len(issues),
        )
    return {
        "issues": issues,
        "per_kernel_attribution_degraded": per_kernel_attribution_degraded,
        "capture_traces_present": capture_traces_present,
    }


# Legacy constant kept pointing at the sglang profile yaml for fixture use.
# Runtime sglang/vllm selection goes through `_default_profile_config()`.
PROFILE_DEFAULT_CONFIG = (
    asset_root() / "scripts" / "configs" / "profile_sglang.yaml"
)
PROFILE_DEFAULT_TIMEOUT_SEC = 14400    # 4 h wall cap; Qwen3-32B TP=1 profile needs ~3 h with steady-state window


def _trace_files_for_dir(trace_dir: Path) -> list[Path]:
    """Return ``*.trace.json.gz`` files under ``trace_dir`` (recursive,
    stable order)."""
    return sorted(trace_dir.rglob("*.trace.json.gz"))


def _preferred_main_trace_path(trace_dir: Path, trace_files: list[Path]) -> Path:
    """Trace path to pass downstream to TraceLens.

    Prefer the ``merged-*`` trace (the large annotated trace the splitter
    wants); otherwise pass the trace dir so kernel-agent picks its own order
    rather than pinning a tiny single-rank slice.
    """
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


def _safe_mtime(p: Path) -> float:
    """Return st_mtime, or 0 on stat() failure (e.g. NFS stale handle).

    Args:
        p (Path): Path to stat.

    Returns:
        float: The modification time, or ``0.0`` if ``stat()`` fails.
    """
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _default_profile_config() -> Path:
    """Resolve default profile YAML from $FRAMEWORK (atom / vllm / sglang;
    unknown falls back to ``profile_sglang.yaml``).

    The atom branch is explicit because the materializer resolves Magpie's
    wrapper script from the YAML's ``benchmark.framework`` (not $FRAMEWORK);
    falling through to the sglang yaml on FRAMEWORK=atom would launch the
    wrong wrapper.
    """
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    if fw == "atom":
        name = "profile_atom.yaml"
    elif fw == "vllm":
        name = "profile_vllm.yaml"
    else:
        name = "profile_sglang.yaml"
    return asset_root() / "scripts" / "configs" / name


class ProfileExecutor(BaselineExecutor):
    """Subclass that swaps the default config + extracts trace_dir."""

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_timeout_sec: int = PROFILE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
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
            cwd=cwd,
        )

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
        """
        from ._multi_node_env import is_multi_node, rayjob_id_from_state
        if not is_multi_node():
            return ""
        provisioned = os.environ.get(
            "HYPERLOOM_MN_PROFILE_TRACE_DIR", ""
        ).strip()
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
                "cannot mkdir multi-node profile fallback dir %s: %s; "
                "downstream readers may FileNotFoundError",
                scoped, exc,
            )
        return str(scoped)

    def _after_materialize_config(
        self, config_path: Path, output_dir: Path,
    ) -> dict[str, Any] | None:
        """Patch the exact InferenceX checkout Magpie will execute.

        `$INFERENCEX_PATH` alone is insufficient (Magpie resolves an empty
        `benchmark.inferencex_path` to its own sibling checkout); patch the
        path resolved from the materialized YAML so NUM_PROMPTS /
        PROFILE_EXTRA_BODY aren't applied to a different checkout.
        """
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "error_class": "profile_config_unreadable",
                "error": f"cannot read materialized profile config {config_path}: {exc}",
            }
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
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

        lib_valid = _contains(lib_path, '${NUM_PROMPTS:-$max_concurrency}')
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
        # --torch-profiler-dir; atom writes standard *.pt.trace.json.gz that
        # _candidate_trace_dirs + TraceLens consume unchanged, so the executor
        # falls through to the sglang/vllm path. (atom's TraceLens-flag guard
        # lives in _workload_envs.py.)
        params = ctx.task.params or {}
        # Merge current_best.extra_server_args (stamped into base_extra_args
        # by the Coordinator) with caller args so profile reflects stable
        # workload tweaks without carrying compile flags that break profiling.
        base_args = _sanitize_profile_server_args(
            str(params.get("base_extra_args") or "").strip(),
        )
        caller_args = _sanitize_profile_server_args(read_extra_server_args(params))
        from ._grid_runner import merge_server_args
        merged_args = merge_server_args(base_args, caller_args)
        if merged_args:
            params["extra_server_args"] = merged_args
        else:
            params.pop("extra_server_args", None)
        params.pop("extra_sglang_args", None)
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

        # Mtime gate for the multi-node shared-trace-dir layout: captured
        # before super().__call__ so this round's traces are newer than the
        # watermark and earlier rounds' traces are filtered out below.
        import time as _time
        task_started_unix = _time.time()

        # Multi-node banner (silent for single-node) surfacing the round's dir.
        from ._multi_node_env import log_mn_banner
        log_mn_banner(
            "profile_executor", log,
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
                    model_path=(
                        str(params.get("model_path") or "").strip()
                        or os.environ.get("MODEL_PATH") or None
                    ),
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

        # NOTE: InferenceX patches (``ensure_benchmark_lib_patched`` and
        # ``ensure_benchmark_serving_patched``) used to live here on the
        # multi-node feature branch. Main moved them into the
        # ``_after_materialize_config`` hook (run earlier in the
        # materialize step) so the patch always covers the exact
        # InferenceX checkout Magpie will execute, regardless of which
        # subdirectory benchmark.inferencex_path resolves to. Removing
        # the duplicate calls here keeps the patch idempotent (same
        # behaviour) while letting the single-source-of-truth in
        # ``_after_materialize_config`` carry the resolved-path
        # validation.
        # Multi-node dynamo only: the Dynamo frontend does not propagate
        # /start_profile to the SSH-launched disagg workers, so torch
        # profiling must be triggered directly on each worker's system
        # server (/engine/start_profile). Bracket the Magpie benchmark with
        # start/stop so traces land in the shared-FS round trace dir for
        # TraceLens. The helper no-ops for RayJob / single-node and is
        # fail-soft per worker. ``PROFILE_EXTRA_BODY`` (start_step/num_steps
        # computed by _workload_envs) is the start_profile payload.
        from ._multi_node_server_lifecycle import trigger_dynamo_engine_profile
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
        # crash. So the dynamo engine-route path forwards ONLY ``output_dir``
        # and bounds the trace by the start/stop wall-clock window instead.
        _SAFE_PROFILE_KEYS = ("output_dir",)
        prof_body = {
            k: v for k, v in prof_body.items() if k in _SAFE_PROFILE_KEYS
        }
        # Pin the trace output dir explicitly. The disagg workers may not
        # carry SGLANG_TORCH_PROFILER_DIR (baseline-time launches never set
        # it and per-round restarts can resume rather than relaunch), so
        # without output_dir sglang writes nowhere the sandbox can read ->
        # roofline profile_no_trace. ``round_trace_root`` is the same
        # shared-FS dir the post-bench trace scan reads (wekafs, mounted on
        # the worker pods; verified writable from the worker).
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
        # throughput number), then stop. ``trigger_dynamo_engine_profile``
        # no-ops for RayJob / single-node, so this is multi-node-dynamo only.
        import asyncio as _asyncio
        warmup_s = float(
            os.environ.get("HYPERLOOM_MN_PROFILE_WARMUP_S", "60") or 60
        )
        window_s = float(
            os.environ.get("HYPERLOOM_MN_PROFILE_WINDOW_S", "8") or 8
        )
        _prof_started = {"v": False}

        async def _bounded_profile_window() -> None:
            await _asyncio.sleep(warmup_s)
            await trigger_dynamo_engine_profile("start", prof_body)
            _prof_started["v"] = True
            await _asyncio.sleep(window_s)
            await trigger_dynamo_engine_profile("stop")
            _prof_started["v"] = False

        prof_task = _asyncio.create_task(_bounded_profile_window())
        try:
            result = await super().__call__(ctx)
        finally:
            # Magpie ended (or raised): wind the window task down and make
            # sure profiling is never left running open-ended.
            if not prof_task.done():
                prof_task.cancel()
                try:
                    await prof_task
                except (_asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if _prof_started.get("v"):
                # start fired but stop didn't (window task cancelled mid-run
                # because Magpie finished first) -> ensure a matching stop.
                await trigger_dynamo_engine_profile("stop")

        # Augment with trace_dir. Multi-node: traces live at the round-scoped
        # wekafs dir we restarted with (not $HYPERLOOM_MN_PROFILE_TRACE_DIR,
        # which the helper reset). Single-node uses workspace/torch_trace below.
        workspace_str = result.get("workspace")
        if round_trace_root:
            # Multi-node: traces land at the shared wekafs base dir (not the
            # workspace-local ``_candidate_trace_dirs``). Mtime-gate to files
            # at-or-after this round's start, else we pick up round 1's trace.
            trace_dir = Path(round_trace_root)
            if trace_dir.is_dir():
                all_files = sorted(trace_dir.glob("*.trace.json.gz"))
                trace_files = [
                    p for p in all_files
                    if _safe_mtime(p) >= task_started_unix
                ]
                result["trace_dir"] = str(trace_dir)
                result["trace_files"] = [str(p) for p in trace_files]
                if trace_files:
                    # Multi-node only: the shared round dir can hold more
                    # than one profiling batch (e.g. a tiny warmup-window
                    # capture that recorded CPU-only activity PLUS the real
                    # GPU-rich capture). Handing TraceLens the CPU-only one
                    # fails with "Trace contains zero GPU kernel events".
                    # GPU-rich traces are orders of magnitude larger
                    # (hundreds of MB vs ~250 KB CPU-only), so select the
                    # LARGEST file as the main trace rather than the
                    # earliest-by-name. Single-node (elif below) is
                    # untouched — it never produces a competing CPU-only
                    # batch.
                    def _safe_size(p: Path) -> int:
                        try:
                            return p.stat().st_size
                        except OSError:
                            return 0
                    main_trace = max(trace_files, key=_safe_size)
                    result["main_trace_path"] = str(main_trace)
                    log.info(
                        "profile_executor: multi-node main trace selected by "
                        "size: %s (%d bytes; %d candidate traces this round)",
                        main_trace.name, _safe_size(main_trace),
                        len(trace_files),
                    )
                elif all_files:
                    log.warning(
                        "profile_executor: multi-node trace dir %s has "
                        "%d historical trace(s) but none with mtime >= "
                        "%.0f (this round's start); sglang may have "
                        "skipped /start_profile or the trace flush is "
                        "lagging", trace_dir, len(all_files),
                        task_started_unix,
                    )
                else:
                    log.warning(
                        "profile_executor: multi-node trace dir %s exists "
                        "but no .trace.json.gz files found yet (server "
                        "pods may still be flushing)", trace_dir,
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
            for trace_dir in _candidate_trace_dirs(workspace):
                if not trace_dir.is_dir():
                    continue
                trace_files = _trace_files_for_dir(trace_dir)
                if trace_files:
                    selected_trace_dir = trace_dir
                    selected_trace_files = trace_files
                    break
                existing_empty_dirs.append(trace_dir)

            if selected_trace_dir is not None:
                result["trace_dir"] = str(selected_trace_dir)
                result["trace_files"] = [str(p) for p in selected_trace_files]
                main_trace = _preferred_main_trace_path(
                    selected_trace_dir, selected_trace_files,
                )
                result["main_trace_path"] = str(main_trace)
                result["profile_trace_selection_reason"] = (
                    "merged_trace_preferred"
                    if main_trace.name.startswith("merged-")
                    else "trace_dir_preferred"
                )
                # #210: warn if the trace shape suggests PROFILE_EXTRA_BODY
                # leaked / shape-discovery missing. Read-only; never blocks.
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
                        "profile_executor: trace structure validator failed: %s", e,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                result["status"] = "failed"
                result["error_class"] = "no_trace_files"
                probed = ", ".join(
                    str(p) for p in _candidate_trace_dirs(workspace)
                )
                result["error"] = (
                    f"no .trace.json.gz under {workspace_str} (probed: {probed})"
                )
                if existing_empty_dirs:
                    log.warning(
                        "profile_executor: trace dirs exist but no "
                        ".trace.json.gz files in %s",
                        ", ".join(str(p) for p in existing_empty_dirs),
                    )
                else:
                    log.warning(
                        "profile_executor: workspace=%s has no trace dir "
                        "(checked: %s)",
                        workspace_str,
                        ", ".join(str(p) for p in _candidate_trace_dirs(workspace)),
                    )
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_CONFIG",
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]
