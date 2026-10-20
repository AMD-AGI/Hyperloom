# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``profile`` ActionRunner — Magpie run with torch profiler on."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, NamedTuple

import yaml

from hyperloom.orchestrator.trace_analysis._capture_shapes import (
    is_capture_fragment as _shared_is_capture_fragment,
)
from hyperloom.orchestrator.trace_analysis._trace_rank import (
    select_primary_trace,
    trace_rank as _trace_rank,
)
from hyperloom.common.io import safe_mtime
from hyperloom.common.profile_args import sanitize_profile_server_args as _sanitize_profile_server_args
from hyperloom.inference_optimizer.session.paths import asset_root, mn_profile_trace_root
from hyperloom.inference_optimizer import framework_registry
from ._inferencex_patcher import (
    benchmark_serving_path_in,
    ensure_benchmark_lib_patched,
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_serving_patched,
)
from ._profile_trace_checks import (
    CHECK_SGLANG_SHAPE_PROFILER,
    CHECK_STEP_ANNOTATIONS,
    _build_trace_validate,
    _certify_trace_dir,
    _instrumentation_preflight_row,
    _TraceProbe,
    _validate_trace_structure,
    _write_trace_certificate,
)
from ._xdit_patcher import verify_xdit_profiler_baked
from .baseline import BenchmarkRunExecutor


log = logging.getLogger(__name__)


PROFILE_DEFAULT_TIMEOUT_SEC = 14400  # 4 h wall cap

#: Keys of ``$PROFILE_EXTRA_BODY`` the Infera engine profiling API may be driven with.
_SAFE_PROFILE_KEYS = ("output_dir",)


def _is_capture_trace(path: Path, root: Path | None = None) -> bool:
    """True when ``path`` is a CUDA-graph capture sidecar rather than a trace."""
    return _shared_is_capture_fragment(path, root)


def _trace_size_bytes(path: Path) -> int:
    """Size in bytes, or 0 when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _is_split_chunk(path: Path, root: Path) -> bool:
    """True when ``path`` is steady-state splitter output below ``root``."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part == "trace_split" for part in relative.parts)


def _trace_files_for_dir(trace_dir: Path) -> list[Path]:
    """Return annotated ``*.trace.json.gz`` files under ``trace_dir``."""
    candidates = [
        p
        for p in trace_dir.rglob("*.trace.json.gz")
        if not _is_capture_trace(p, trace_dir) and not _is_split_chunk(p, trace_dir)
    ]
    return sorted(candidates, key=lambda p: (-_trace_size_bytes(p), str(p)))


def _capture_sidecar_traces_for_dir(trace_dir: Path) -> list[Path]:
    """Return CUDA-graph capture sidecars under ``trace_dir`` (fallback only)."""
    return sorted(p for p in trace_dir.rglob("*.json.gz") if _is_capture_trace(p, trace_dir))


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
    """Trace path to pass downstream to TraceLens."""
    if require_single_rank:
        return select_primary_trace(
            trace_files,
            file_size=_trace_size_bytes,
            preferred_rank=preferred_rank,
            tensor_parallel_size=tensor_parallel_size,
        )

    merged = sorted(p for p in trace_files if p.name.startswith("merged-"))
    return merged[0] if merged else trace_dir


def _candidate_trace_dirs(workspace: Path) -> list[Path]:
    """Trace directories to probe for a Magpie profile workspace."""
    return [
        workspace / "torch_trace",
        workspace / "capture_traces",
        workspace.parent / "capture_traces",
    ]


def _default_profile_config() -> Path:
    """Resolve the shipped profile YAML for ``$FRAMEWORK``, or for the default framework when it is unset."""
    fw = os.environ.get("FRAMEWORK") or framework_registry.DEFAULT_FRAMEWORK
    return asset_root() / "assets" / "configs" / framework_registry.shipped_config_name("profile", fw)


def _merged_server_args(params: Mapping[str, Any]) -> str:
    """The server args this round runs with: the action's base args, overlaid by the caller's."""
    from ._grid_runner import merge_server_args

    base_args = _sanitize_profile_server_args(str(params.get("base_extra_args") or "").strip())
    caller_args = _sanitize_profile_server_args(str(params.get("extra_server_args") or ""))
    return merge_server_args(base_args, caller_args)


def _merged_envs(params: Mapping[str, Any]) -> dict[str, Any]:
    """The environment overrides this round runs with: the action's base envs under the caller's."""
    merged: dict[str, Any] = {}
    for source in (params.get("base_extra_envs"), params.get("extra_envs")):
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _as_list(raw: Any) -> list[Any]:
    """One string, or an iterable of them, as a list."""
    return [raw] if isinstance(raw, str) else list(raw or [])


def _apply_round_defaults(params: dict[str, Any]) -> None:
    """Fold this action's ``base_*`` defaults into the params the benchmark round will run with."""
    merged_args = _merged_server_args(params)
    if merged_args:
        params["extra_server_args"] = merged_args
    else:
        params.pop("extra_server_args", None)
    merged_envs = _merged_envs(params)
    if merged_envs:
        params["extra_envs"] = merged_envs
    else:
        params.pop("extra_envs", None)
    if params.get("base_remove_args") and "remove_args" not in params:
        params["remove_args"] = _as_list(params.get("base_remove_args"))
    if params.get("base_unset_envs") and "unset_envs" not in params:
        params["unset_envs"] = _as_list(params.get("base_unset_envs"))
    if str(params.get("base_args_mode") or "").strip().lower() == "replace":
        params.setdefault("args_mode", "replace")


async def _restart_mn_server_for_profile(params: Mapping[str, Any], round_trace_root: str) -> dict[str, Any] | None:
    """Restart the multi-node server onto this round's trace dir; its failure result, or None once it came up."""
    from ._multi_node_server_lifecycle import ServerRestartFailed, restart_server_for_round

    try:
        await restart_server_for_round(
            extra_server_args=str(params.get("extra_server_args") or ""),
            torch_profiler_dir=round_trace_root,
            framework=os.environ.get("FRAMEWORK") or None,
            model_path=str(params.get("model_path") or "").strip() or os.environ.get("MODEL_PATH") or None,
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
    return None


def _engine_profile_body(round_trace_root: str) -> dict[str, Any]:
    """The body the Infera engine profiling API is started with for this round."""
    try:
        parsed = json.loads(os.environ.get("PROFILE_EXTRA_BODY") or "{}")
    except (ValueError, TypeError):
        parsed = {}
    body = {k: v for (k, v) in parsed.items() if k in _SAFE_PROFILE_KEYS} if isinstance(parsed, dict) else {}
    if round_trace_root:
        body.setdefault("output_dir", round_trace_root)
    return body


@asynccontextmanager
async def _bounded_engine_profile(round_trace_root: str) -> AsyncIterator[None]:
    """Capture an Infera engine profile over a warmup-then-bounded window while the block runs."""
    from ._multi_node_server_lifecycle import trigger_infera_engine_profile

    body = _engine_profile_body(round_trace_root)
    warmup_s = float(os.environ.get("HYPERLOOM_MN_PROFILE_WARMUP_S", "60") or 60)
    window_s = float(os.environ.get("HYPERLOOM_MN_PROFILE_WINDOW_S", "8") or 8)
    capturing = {"v": False}

    async def _window() -> None:
        """Run a warmup-then-bounded engine profiling window."""
        await asyncio.sleep(warmup_s)
        await trigger_infera_engine_profile("start", body)
        capturing["v"] = True
        await asyncio.sleep(window_s)
        await trigger_infera_engine_profile("stop")
        capturing["v"] = False

    window = asyncio.create_task(_window())
    try:
        yield
    finally:
        if not window.done():
            window.cancel()
            try:
                await window
            except (asyncio.CancelledError, OSError, RuntimeError, ValueError) as exc:
                log.debug("profile_executor: bounded profile task ended during cleanup: %r", exc)
        if capturing.get("v"):
            await trigger_infera_engine_profile("stop")


def _tensor_parallel_size(result: Mapping[str, Any], params: Mapping[str, Any], shared_state: Any) -> int | None:
    """The tensor-parallel width this round ran at, from the first source that states one."""
    for raw_tp in (result.get("tp"), params.get("tp"), getattr(shared_state, "tp", None), os.environ.get("TP")):
        try:
            parsed_tp = int(raw_tp or 0)
        except (TypeError, ValueError):
            continue
        if parsed_tp > 0:
            return parsed_tp
    return None


def _log_missing_round_traces(trace_dir: Path, historical_count: int, task_started_unix: float) -> None:
    """Say why this round's multi-node trace dir yielded nothing the round itself produced."""
    if historical_count:
        log.warning(
            "profile_executor: multi-node trace dir %s has %d historical trace(s) but none with mtime >= %.0f (this round's start); sglang may have skipped /start_profile or the trace flush is lagging",
            trace_dir,
            historical_count,
            task_started_unix,
        )
    else:
        log.warning(
            "profile_executor: multi-node trace dir %s exists but no .trace.json.gz files found yet (server pods may still be flushing)",
            trace_dir,
        )


def _record_mn_round_traces(result: dict[str, Any], round_trace_root: str, task_started_unix: float) -> None:
    """Attach the traces this round's multi-node server flushed to ``result``."""
    trace_dir = Path(round_trace_root)
    if not trace_dir.is_dir():
        result["trace_dir"] = None
        result["trace_files"] = []
        log.warning(
            "profile_executor: round trace dir %s does not exist after magpie completed; check sglang server logs for torch profiler errors",
            round_trace_root,
        )
        return
    all_files = sorted(trace_dir.glob("*.trace.json.gz"))
    trace_files = [p for p in all_files if safe_mtime(p) >= task_started_unix]
    result["trace_dir"] = str(trace_dir)
    result["trace_files"] = [str(p) for p in trace_files]
    _record_trace_topology(result, trace_files)
    if not trace_files:
        _log_missing_round_traces(trace_dir, len(all_files), task_started_unix)
        return
    main_trace = max(trace_files, key=_trace_size_bytes)
    result["main_trace_path"] = str(main_trace)
    log.info(
        "profile_executor: multi-node main trace selected: %s (%d candidate traces this round)",
        main_trace.name,
        len(trace_files),
    )


class _TraceSelection(NamedTuple):
    """The trace directory a profile workspace produced, the files worth reading in it, and what was ruled out."""

    trace_dir: Path | None
    files: list[Path]
    capture_only: bool
    empty_dirs: list[Path]


def _select_workspace_traces(candidate_dirs: list[Path]) -> _TraceSelection:
    """Pick the trace dir a profile workspace produced, falling back to SGLang capture sidecars (#575)."""
    empty_dirs: list[Path] = []
    for trace_dir in candidate_dirs:
        if not trace_dir.is_dir():
            continue
        trace_files = _trace_files_for_dir(trace_dir)
        if trace_files:
            return _TraceSelection(trace_dir, trace_files, False, empty_dirs)
        empty_dirs.append(trace_dir)
    for trace_dir in candidate_dirs:
        if not trace_dir.is_dir():
            continue
        sidecars = _capture_sidecar_traces_for_dir(trace_dir)
        if sidecars:
            return _TraceSelection(trace_dir, sidecars, True, empty_dirs)
    return _TraceSelection(None, [], False, empty_dirs)


def _record_selected_traces(
    result: dict[str, Any],
    selection: _TraceSelection,
    tensor_parallel_size: int | None,
) -> None:
    """Attach the selected trace dir, its files and the main trace TraceLens should read to ``result``."""
    result["trace_dir"] = str(selection.trace_dir)
    result["trace_files"] = [str(p) for p in selection.files]
    _record_trace_topology(result, selection.files)
    if selection.capture_only:
        result["profile_trace_selection_reason"] = "capture_only_fallback"
        main_trace = selection.trace_dir
        log.info(
            "profile_executor: no *.trace.json.gz; falling back to %d SGLang capture sidecar(s) in %s (#575)",
            len(selection.files),
            selection.trace_dir,
        )
    else:
        main_trace = _preferred_main_trace_path(
            selection.trace_dir,
            selection.files,
            require_single_rank=False,
            tensor_parallel_size=tensor_parallel_size,
        )
        if main_trace is not None:
            result["profile_trace_selection_reason"] = (
                "merged_trace_preferred" if main_trace.name.startswith("merged-") else "trace_dir_preferred"
            )
    if main_trace is not None:
        result["main_trace_path"] = str(main_trace)


def _record_no_trace_failure(
    result: dict[str, Any],
    workspace_str: str,
    candidate_dirs: list[Path],
    empty_dirs: list[Path],
) -> None:
    """Fail the profile result because its workspace produced no trace, naming where one was looked for."""
    result["trace_dir"] = None
    result["trace_files"] = []
    result["status"] = "failed"
    result["error_class"] = "no_trace_files"
    probed = ", ".join((str(p) for p in candidate_dirs))
    result["error"] = f"no .trace.json.gz or capture sidecar under {workspace_str} (probed: {probed})"
    if empty_dirs:
        log.warning(
            "profile_executor: trace dirs exist but no .trace.json.gz or capture sidecar (bs_*_rank*.json.gz) files in %s",
            ", ".join((str(p) for p in empty_dirs)),
        )
    else:
        log.warning(
            "profile_executor: workspace=%s has no trace dir (checked: %s)",
            workspace_str,
            ", ".join((str(p) for p in candidate_dirs)),
        )


def _record_trace_validation(
    result: dict[str, Any],
    trace_dir: Path,
    framework: str,
    preflight: dict[str, Any] | None,
) -> None:
    """Attach the post-profile structure checks and the self-certification certificate to ``result``."""
    health = _validate_trace_structure(trace_dir, framework)
    if not isinstance(health, dict):
        return
    result["trace_health"] = health
    certificate: dict[str, Any] = {}
    probe_error = ""
    try:
        certificate = _certify_trace_dir(trace_dir, framework)
    except (OSError, RuntimeError, ValueError, TypeError) as e:
        probe_error = f"{type(e).__name__}: {e}"
        log.debug("profile_executor: trace self-certification failed: %s", probe_error)
    result["trace_validate"] = _build_trace_validate(
        health,
        _TraceProbe(trace_dir, framework, certificate=certificate, error=probe_error),
        preflight=preflight,
    )
    result["trace_validate_path"] = _write_trace_certificate(trace_dir, result["trace_validate"])


class ProfileExecutor(BenchmarkRunExecutor):
    """Benchmark round with the torch profiler on; extracts and certifies the trace_dir."""

    runs_accuracy_eval = False

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_timeout_sec: int = PROFILE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str | None = None,
    ):
        """Initialize the profile executor with profile-specific defaults."""
        super().__init__(
            magpie_python=magpie_python,
            default_config_path=default_config_path,
            session_dir=session_dir,
            default_timeout_sec=default_timeout_sec,
            cwd=cwd if cwd is not None else tempfile.gettempdir(),
        )
        # Set by ``_after_materialize_config`` once the probe is armed; read after the run to merge per-rank reports.
        self._host_probe_dir: str = ""
        # Non-empty only when arming failed, and then says why (an empty probe dir cannot tell never-asked from failed).
        self._host_probe_status: str = ""
        # Set by ``_after_materialize_config`` from the config the run will actually execute, so the post-hoc trace
        # checks ship alongside the pre-run statement of whether their subject could have been produced.
        self._instrumentation_preflight: dict[str, Any] | None = None

    def _resolve_default_config(self) -> Path:
        """Use the session workload for native AgentX diagnostics."""
        from hyperloom.common.benchmark_config import benchmark_config_path

        return benchmark_config_path() or _default_profile_config()

    def _resolve_mn_round_trace_root(self, ctx) -> str:
        """Return the shared torch-trace base dir for multi-node, or ''."""
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
        """Put the profile start-up shim on the materialized config's ``PYTHONPATH`` and arm the host probe.

        The shim always goes in, since it keeps GPU events in the traces of spawned engines; the
        probe's environment only when the probe is enabled. Returns the probe's report directory,
        or ``""`` when the probe is not armed.
        """
        from . import _framework_rewrite_evidence as _evidence
        from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path

        asset_dir = _evidence.probe_asset_dir()
        if not asset_dir.is_dir():
            log.warning(
                "profile_executor: start-up shim assets missing at %s; host-side rewrite evidence disabled",
                asset_dir,
            )
            return ""
        probe_env = self._host_probe_env(output_dir) if _evidence.probe_enabled() else {}
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning("profile_executor: cannot read %s to arm the start-up shim: %s", config_path, exc)
            return ""
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else None
        if not isinstance(bench, dict):
            return ""
        envs = bench.setdefault("envs", {})
        if not isinstance(envs, dict):
            return ""
        # First on the path even when already present: Python imports only the first sitecustomize, and the
        # shim's chains to the next one.
        prepend_launch_path(envs, "PYTHONPATH", str(asset_dir))
        for key, value in probe_env.items():
            envs[key] = value
        try:
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except OSError as exc:
            log.warning("profile_executor: cannot write %s after arming the start-up shim: %s", config_path, exc)
            return ""
        if not probe_env:
            return ""
        log.info(
            "profile_executor: host-side rewrite evidence probe armed (deep=%s), reports -> %s",
            bool(probe_env.get("HYPERLOOM_HOST_PROBE_DEEP")),
            probe_env["HYPERLOOM_HOST_PROBE_DIR"],
        )
        return probe_env["HYPERLOOM_HOST_PROBE_DIR"]

    def _host_probe_env(self, output_dir: Path) -> dict[str, str]:
        """Return the environment that arms the host probe, or ``{}`` when its report directory cannot be made."""
        from . import _framework_rewrite_evidence as _evidence

        probe_dir = output_dir / _evidence.PROBE_SUBDIR
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("profile_executor: cannot create host-probe dir %s: %s", probe_dir, exc)
            return {}

        from hyperloom.inference_optimizer.framework_paths import resolve_kernel_search_roots

        return _evidence.build_probe_env(
            probe_dir=probe_dir,
            source_roots=list(resolve_kernel_search_roots()),
            deep=_evidence.deep_probe_enabled(),
        )

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Arm the host probe, then patch the InferenceX checkout Magpie will execute."""
        from .magpie_agentx import enable_profile, native_config

        if native_config(config_path):
            enable_profile(config_path)
            return None
        try:
            self._host_probe_dir = self._inject_host_probe(config_path, output_dir)
            self._host_probe_status = ""
        except Exception as exc:
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
        patchers: dict[str, Any] = {}

        def _note_instrumentation() -> None:
            """Re-state the instrumentation report. Called before every exit below, so no path leaves it unset."""
            self._instrumentation_preflight = _instrumentation_preflight_row(bench, patchers)
            detail = self._instrumentation_preflight["detail"]
            if self._instrumentation_preflight["status"] != "failed":
                return
            log.warning(
                "profile_executor: instrumentation preflight — degraded_reason=%r failed_patchers=%s; "
                "annotation-dependent trace checks (%s, %s) cannot pass for this run",
                detail.get("degraded_reason"),
                detail.get("failed_patchers"),
                CHECK_STEP_ANNOTATIONS,
                CHECK_SGLANG_SHAPE_PROFILER,
            )

        _note_instrumentation()
        framework = ""
        if isinstance(bench, dict):
            framework = str(bench.get("framework") or "").strip().lower()
        from hyperloom.inference_optimizer import framework_registry

        if framework_registry.is_scriptable(framework):
            if str(framework or "").strip().lower() == "xdit":
                patchers["xdit_profiler_baked"] = verify_xdit_profiler_baked()
                _note_instrumentation()
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
            patchers["inferencex"] = None
            _note_instrumentation()
            return None

        ix_root = Path(inferencex_path)
        lib_ok = ensure_benchmark_lib_patched(ix_root)
        patchers["benchmark_lib"] = lib_ok
        patchers["benchmark_lib_eval_dest"] = ensure_benchmark_lib_eval_dest_patched(ix_root)
        serving_ok = ensure_benchmark_serving_patched(ix_root)
        patchers["benchmark_serving"] = serving_ok
        lib_path = ix_root / "benchmarks" / "benchmark_lib.sh"
        # path as a forwarding shim, which never carries the sentinel however well the patch landed.
        # Scoped to ``ix_root`` like ``lib_path`` above: this gate speaks for the tree Magpie runs.
        serving_path = benchmark_serving_path_in(ix_root)

        def _contains(path: Path, needle: str) -> bool:
            """Check whether ``needle`` appears in ``path``'s text."""
            try:
                return needle in path.read_text(encoding="utf-8")
            except OSError:
                return False

        lib_valid = _contains(lib_path, "${NUM_PROMPTS:-$max_concurrency}")
        serving_valid = _contains(serving_path, "PROFILE_EXTRA_BODY")
        # The sentinels are a separate fact from the patcher's return: a patcher can report success against a
        # checkout whose anchors have since moved, and only reading the file back tells them apart.
        patchers["benchmark_lib_sentinel"] = lib_valid
        patchers["benchmark_serving_sentinel"] = serving_valid
        patchers["inferencex_path"] = str(ix_root)
        _note_instrumentation()
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

    def drain_instrumentation_report(self) -> dict[str, Any] | None:
        """Take the instrumentation report this executor produced, clearing it.

        Draining rather than reading: the caller runs several profile attempts against this module-level singleton,
        and an attempt that dies before materializing a config must report nothing rather than inherit the
        previous attempt's report. Exposed because the report has to survive the paths where no result dict does
        -- an executor that raises still patched, or still failed to.
        """
        report, self._instrumentation_preflight = self._instrumentation_preflight, None
        return report

    def _collect_rewrite_evidence(self, result: dict[str, Any]) -> None:
        """Merge the per-rank host-probe reports onto ``result``."""
        probe_dir = str(self._host_probe_dir or "").strip()
        if not probe_dir:
            result["framework_rewrite_evidence_status"] = self._host_probe_status or "probe_not_armed"
            return
        from . import _framework_rewrite_evidence as _evidence

        try:
            out_path = Path(probe_dir).parent / _evidence.EVIDENCE_FILENAME
            document = _evidence.aggregate_probe_dir(probe_dir, out_path)
        except Exception as exc:
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

    def _ensure_workspace(self, ctx, extra: dict[str, Any], params: Mapping[str, Any]) -> None:
        """Give this round a workspace to write into when neither the caller nor the context named one."""
        if params.get("output_dir") or extra.get("workspace"):
            return
        output_dir = self._resolve_workspace(ctx, "profile")
        output_dir.mkdir(parents=True, exist_ok=True)
        extra["workspace"] = str(output_dir)

    def _record_workspace_traces(
        self,
        ctx,
        result: dict[str, Any],
        extra: Mapping[str, Any],
        tensor_parallel_size: int | None,
    ) -> None:
        """Attach the traces a single-node profile workspace produced, with their structure checks."""
        workspace_str = result["workspace"]
        candidate_dirs = _candidate_trace_dirs(Path(workspace_str))
        selection = _select_workspace_traces(candidate_dirs)
        if selection.trace_dir is None:
            _record_no_trace_failure(result, workspace_str, candidate_dirs, selection.empty_dirs)
            return
        _record_selected_traces(result, selection, tensor_parallel_size)
        framework = str(getattr(ctx, "framework", "") or extra.get("framework") or "")
        try:
            _record_trace_validation(result, selection.trace_dir, framework, self._instrumentation_preflight)
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            log.debug("profile_executor: trace structure validator failed: %s", e)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the profiling action for the given context."""
        params = ctx.task.params or {}
        _apply_round_defaults(params)
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or getattr(self, "shared_state", None)
        from hyperloom.common.benchmark_config import configured_benchmark, is_magpie_agentx

        if is_magpie_agentx(configured_benchmark()):
            return await self._run_native_profile(ctx)
        self._ensure_workspace(ctx, extra, params)
        task_started_unix = time.time()
        round_trace_root = self._resolve_mn_round_trace_root(ctx)
        from ._multi_node_env import log_mn_banner

        log_mn_banner("profile_executor", log, trace_dir=round_trace_root)
        if round_trace_root:
            restart_failure = await _restart_mn_server_for_profile(params, round_trace_root)
            if restart_failure is not None:
                return restart_failure
            extra["mn_round_restarted"] = True
        async with _bounded_engine_profile(round_trace_root):
            result = await super().__call__(ctx)
        self._collect_rewrite_evidence(result)
        tensor_parallel_size = _tensor_parallel_size(result, params, shared_state)
        if round_trace_root:
            _record_mn_round_traces(result, round_trace_root, task_started_unix)
        elif result.get("workspace"):
            self._record_workspace_traces(ctx, result, extra, tensor_parallel_size)
        if result.get("main_trace_path"):
            result["trace_input_ready"] = True
        return result

    def _measurement_succeeded(self, measurement: dict[str, Any]) -> bool:
        return measurement.get("round_succeeded", measurement.get("valid_measurement")) is True

    async def _run_native_profile(self, ctx) -> dict[str, Any]:
        """Collect Magpie-owned captures and certify each time window separately."""
        result = await super().__call__(ctx)
        if result.get("status") != "succeeded":
            return result
        capture = result.get("agentx_metrics", {}).get("profile_capture", {})
        if capture.get("status") != "complete":
            return {
                **result,
                "status": "failed",
                "error_class": "profile_capture_failed",
                "trace_capture": capture,
                "trace_input_ready": False,
            }
        result["trace_capture"] = capture
        result["trace_capture_status"] = capture["status"]
        result["trace_input_ready"] = False
        result["profile_captures"] = []
        for window in capture.get("profiles", [capture]):
            mapped = await asyncio.to_thread(_map_native_capture, window, result)
            result["profile_captures"].append(mapped)
        primary = next((window for window in result["profile_captures"] if window.get("trace_input_ready")), None)
        if primary is None:
            result.update(status="failed", error_class="primary_rank_trace_missing")
        else:
            result.update(primary)
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]


def _map_native_capture(capture: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    """Select and check one capture, without combining ranks from different windows."""
    files = [Path(name) for name in capture.get("trace_files", [])]
    result: dict[str, Any] = {"capture_id": capture["capture_id"], "trace_input_ready": False}
    if not files or not all(path.is_file() for path in files):
        return result
    trace_dir = files[0].parent
    main = _preferred_main_trace_path(
        trace_dir,
        files,
        require_single_rank=True,
        tensor_parallel_size=measurement.get("tp"),
    )
    result.update(trace_dir=str(trace_dir), trace_files=[str(path) for path in files])
    _record_trace_topology(result, files)
    if main is None:
        return result
    result.update(main_trace_path=str(main), primary_rank=_trace_rank(main), trace_input_ready=True)
    framework = measurement["framework"]
    health = _validate_trace_structure(trace_dir, framework)
    certificate = _certify_trace_dir(trace_dir, framework)
    result["trace_health"] = health
    result["trace_validate"] = _build_trace_validate(health, _TraceProbe(trace_dir, framework, certificate=certificate))
    result["trace_validate_path"] = _write_trace_certificate(trace_dir, result["trace_validate"])
    return result
