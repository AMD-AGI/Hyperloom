# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Real ``baseline`` ActionRunner — runs Magpie SGLang benchmark.

Runs the Magpie CLI as a subprocess, parses ``benchmark_report.json``,
and returns the result on the bus as a ``delegated_result`` event.

RunnerContext.task.params keys (all optional; defaults from
default_baseline_config()): ``config_path``, ``output_dir``, ``timeout_sec``.

Returns ``error_class`` on failure so the coordinator can route to
Robustness RCA later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from hyperloom.inference_optimizer.compat import read_extra_server_args
from hyperloom.inference_optimizer.session_paths import runs_dir
from ...loop.sub_agent_runner import RunnerContext
from . import _server_lifecycle as _lifecycle
from ._file_lock import best_effort_file_lock
from ._aiter_jit import (
    AITER_JIT_PROBE_PATHS,
    COLD_START_KERNEL_THRESHOLD,
    _resolve_aiter_jit_dir_dynamic,
    sweep_stale_aiter_locks_if_dead,
)
from ._grid_runner import (
    _kill_stale_servers,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._subprocess_kill import (
    DETOKENIZER_STALL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    run_with_session_kill,
    server_log_death_excerpt,
)
from ._workload_envs import (
    _RUN_EVAL_FALSE_VALUES,
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)
from .benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
)


log = logging.getLogger(__name__)


# Markers that identify an InferenceX ``run_eval`` (lm-eval) failure as the
# root cause of a benchmark non-zero exit. ``run_eval`` echoes the first marker
# on ANY eval failure (benchmark_lib.sh), so it is the most general signal; the
# others catch the specific redundant-flag breakage even when the generic
# message scrolled out of a truncated tail.
_EVAL_FAILURE_MARKERS = (
    "run_eval failed with exit code",
    "ERROR: run_eval failed",
    "Unknown parameter: --concurrent-requests",
)
# Bounded per-file read so scanning a failed run's logs for eval markers never
# slurps a multi-GB server.log.
_EVAL_SCAN_MAX_BYTES = 262_144


def _is_truthy(value: Any) -> bool:
    """Return whether ``value`` represents an affirmative flag.

    Accepts bools and the usual truthy strings (``true``/``1``/``yes``/``on``);
    everything else (including ``None`` and ``false``/``0``) is False.

    Args:
        value: The task-param value to interpret.

    Returns:
        ``True`` for an affirmative bool/string, else ``False``.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# Fast-exit arg errors (vLLM/sglang exits in <30s on bad CLI args)
# should not consume the slow-baseline retry budget.
FAST_EXIT_THRESHOLD_SEC = 30.0
_ARG_ERROR_PATTERNS = (
    "unrecognized arguments",
    "invalid choice",
    "Unknown attention backend",
    "not a valid",
)
_ARG_ERROR_CONTEXT_PATTERNS = (
    "argument",
    "argparse",
    "backend",
    "choice",
    "cli",
    "invalid",
    "option",
    "flag",
    "unknown",
)


# Strong cuda-graph capture markers: stream-capture incompatibility, reliably
# recoverable by disabling cuda-graph. Markers live in server.log, not the
# Magpie stdout/stderr tail.
_CUDA_GRAPH_STRONG_MARKERS = (
    "operation not permitted when stream is capturing",
    "hiperrorstreamcaptureunsupported",
)
# Weak marker: bare "Capture cuda graph failed" carries no root cause. Trust it
# only when the nearby context is neither OOM nor a compile/lowering error,
# both of which disabling cuda-graph cannot recover.
_CUDA_GRAPH_WEAK_MARKER = "capture cuda graph failed"

# Profile-cuda-graph shape discovery feeds a device seq_lens into sglang's
# get_num_new_pages (which asserts CPU) -> AssertionError -> SIGQUIT. Disabling
# cuda-graph skips that path, so this specific assert IS recoverable and must
# win over the generic assertionerror non-recoverable gate below. Both markers
# are required so a generic AssertionError never matches.
_CUDA_GRAPH_PROFILE_ASSERT_MARKERS = (
    "get_num_new_pages",
    "seq_lens.device == cpu_device",
)

# OOM-rooted capture failures are NOT recoverable by disabling cuda-graph
# (eager peaks can be higher); compile/lowering errors are not either.
_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
)
_NON_RECOVERABLE_MARKERS = (
    "loweringexception",
    "assertionerror",
    "compilationerror",
)
# Strong markers are high-confidence stream-capture incompatibilities, so OOM
# exclusion is scoped tight (±1 line): only an OOM on/adjacent to the marker
# line demotes it, an unrelated startup OOM warning nearby does not (false
# negative guard). The bare weak marker is unreliable, so it keeps the
# whole-blob OOM/compile exclusion to avoid wasting the one-shot retry.
_STRONG_OOM_CONTEXT_RADIUS = 1


def _is_cuda_graph_capture_failure(*texts: str) -> bool:
    """True when a cuda-graph capture marker is recoverable by disabling graph.

    A strong stream-capture marker arms the fallback unless an OOM sits on its
    ±1-line context (tight, since the marker itself is high confidence). The
    bare ``Capture cuda graph failed`` is a weak signal: the WHOLE blob must
    carry neither OOM nor a compile/lowering error, both unrecoverable by
    disabling cuda-graph. Strong wins on a line that also matches weak, so the
    compile/OOM whole-blob gate never demotes a genuine stream-capture failure.

    Args:
        *texts: Log / stdout / stderr blobs to scan for cuda-graph markers.

    Returns:
        ``True`` when a cuda-graph capture failure recoverable by disabling
        cuda-graph capture is detected, else ``False``.
    """
    lines = "\n".join(t for t in texts if t).splitlines()
    lowered = [ln.lower() for ln in lines]
    blob = "\n".join(lowered)
    # Specific profile-cuda-graph assert wins over the assertionerror gate.
    if all(m in blob for m in _CUDA_GRAPH_PROFILE_ASSERT_MARKERS):
        return True
    blob_has_oom = any(m in blob for m in _OOM_MARKERS)
    blob_has_non_recoverable = any(m in blob for m in _NON_RECOVERABLE_MARKERS)
    saw_pure_weak = False
    for idx, line in enumerate(lowered):
        is_strong = any(m in line for m in _CUDA_GRAPH_STRONG_MARKERS)
        if is_strong:
            lo = max(0, idx - _STRONG_OOM_CONTEXT_RADIUS)
            hi = min(len(lowered), idx + _STRONG_OOM_CONTEXT_RADIUS + 1)
            if not any(m in "\n".join(lowered[lo:hi]) for m in _OOM_MARKERS):
                return True
            continue
        if _CUDA_GRAPH_WEAK_MARKER in line:
            saw_pure_weak = True
    if saw_pure_weak and not blob_has_oom and not blob_has_non_recoverable:
        return True
    return False


# Disable cuda-graph capture per framework: sglang uses --disable-cuda-graph
# (it rejects vLLM's --enforce-eager), vllm uses --enforce-eager.
_DISABLE_CUDA_GRAPH_FLAGS = {
    "sglang": "--disable-cuda-graph",
    "vllm": "--enforce-eager",
}


def _disable_cuda_graph_flag(framework: str) -> str:
    """Return the framework-correct flag that disables cuda-graph capture.

    Args:
        framework: Framework name (e.g. ``"sglang"`` or ``"vllm"``); matched
            case-insensitively.

    Returns:
        The disable-cuda-graph flag for ``framework``, defaulting to
        ``--disable-cuda-graph`` for unknown frameworks.
    """
    return _DISABLE_CUDA_GRAPH_FLAGS.get(
        (framework or "").strip().lower(),
        "--disable-cuda-graph",
    )


def _with_cuda_graph_disabled(extra_server_args: str, framework: str) -> str:
    """Append the framework-correct disable-cuda-graph flag once (idempotent).

    Token-level dedup so a longer flag (e.g. ``--disable-cuda-graph-extra``)
    is not mistaken for an existing ``--disable-cuda-graph``.

    Args:
        extra_server_args: Existing extra server args string (may be empty).
        framework: Framework name used to pick the correct disable flag.

    Returns:
        ``extra_server_args`` with the framework-correct disable-cuda-graph
        flag appended once; unchanged when the flag is already present.
    """
    flag = _disable_cuda_graph_flag(framework)
    if flag in (extra_server_args or "").split():
        return extra_server_args or ""
    return f"{extra_server_args} {flag}".strip()


def _classify_subprocess_error(
    elapsed_sec: float,
    stderr_tail: str,
) -> str:
    """Return 'fast_exit_arg_error' when the subprocess died fast on an arg
    validation error, else 'subprocess_nonzero'.

    Args:
        elapsed_sec: Subprocess wall-clock runtime in seconds.
        stderr_tail: Tail of the subprocess stderr used for marker matching.

    Returns:
        ``"fast_exit_arg_error"`` for a fast exit caused by argument
        validation, else ``"subprocess_nonzero"``.
    """
    if elapsed_sec >= FAST_EXIT_THRESHOLD_SEC:
        return "subprocess_nonzero"
    tail = stderr_tail.lower()
    if any(p.lower() in tail for p in _ARG_ERROR_PATTERNS):
        return "fast_exit_arg_error"
    if "valueerror:" in tail and any(p in tail for p in _ARG_ERROR_CONTEXT_PATTERNS):
        return "fast_exit_arg_error"
    return "subprocess_nonzero"


BASELINE_DEFAULT_TIMEOUT_SEC = (
    7800  # WARM-start cap, 130 min (raised for Qwen3-32B TP=1 CONC=64 ISL/OSL=1024 NUM_PROMPTS=320 ~82 min workload)
)
BASELINE_DEFAULT_TIMEOUT_SEC = 7800           # WARM-start cap, 130 min (raised for Qwen3-32B TP=1 CONC=64 ISL/OSL=1024 NUM_PROMPTS=320 ~82 min workload)
BASELINE_COLD_START_TIMEOUT_SEC = 9000        # COLD-start cap, 150 min (includes ~20 min cuda graph capture)
# COLD_START_KERNEL_THRESHOLD and AITER_JIT_PROBE_PATHS now live in
# ``_aiter_jit`` (shared with cli.py's startup sweep); re-exported below for
# callers/tests that import them from this module.


# Underscore-prefixed aliases re-exported for callers/tests; canonical
# names live in `_workload_envs`.
_default_baseline_config = default_baseline_config
_materialize_config_with_envs = materialize_config_with_envs


def _should_establish_quality_ref(task_kind: str | None) -> bool:
    """Only a genuine ``baseline`` task may establish/overwrite the quality reference.

    ``replay_warm_recipe`` reuses this executor but is an optimization
    candidate, so it must compare against the pure baseline reference rather
    than redefine it (otherwise the gate would mask the warm recipe's own
    deviation from the baseline output).

    Args:
        task_kind: The task kind (``ctx.task.kind``); ``None`` is treated as
            "not a baseline".

    Returns:
        bool: ``True`` only when the task kind is exactly ``"baseline"``.
    """
    return str(task_kind or "") == "baseline"


def _resolve_reference_base(
    session_dir: Path, *, model_path: str,
) -> tuple[str, dict[str, str]]:
    """Read the model-gated reference base server args/envs from SharedState.

    Baseline is the single choke point every run funnels through (initial /
    restart / stack-rebaseline), so reading the reference here makes it seed
    EVERY baseline — including resume restarts (the reference is a persisted
    fact-layer field). Model-gated: a recipe captured for a different model is
    skipped (a near-name mismatch could apply flags that break this model).

    Returns ``("", {})`` on any failure or when no reference is set — the
    caller then materializes exactly as before (0 degrade).
    """
    try:
        from ...state.shared_state import SharedState
        from hyperloom.inference_optimizer.reference_script import models_compatible
        # The baseline executor is a module-level singleton instantiated at
        # import time — before $INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR is
        # pinned — so its cached session_dir can resolve to the workspace root
        # instead of the live session dir whose state.json carries the
        # reference_* fields. Prefer the live pin when present; otherwise honor
        # the caller-supplied path (direct-instantiation tests, explicit calls).
        from hyperloom.inference_optimizer.paths import ENV_CURRENT_SESSION_DIR
        _pinned = os.environ.get(ENV_CURRENT_SESSION_DIR)
        if _pinned:
            session_dir = Path(_pinned)
        state = SharedState.load_or_init(session_dir)
        ref_args = str(getattr(state, "reference_server_args", "") or "").strip()
        ref_envs = dict(getattr(state, "reference_envs", {}) or {})
        if not ref_args and not ref_envs:
            return ("", {})
        ref_model = str(getattr(state, "reference_model", "") or "").strip()
        # Same normalized, version-aware gate discovery uses (no drift).
        if not models_compatible(ref_model, str(model_path or "")):
            log.warning(
                "reference recipe is for model %r but run model is %r; "
                "skipping reference base (flags may not apply).",
                ref_model, Path(str(model_path or "")).name,
            )
            return ("", {})
        return (ref_args, ref_envs)
    except Exception as exc:  # noqa: BLE001 — fail-soft, never block baseline
        log.warning("reference base lookup failed: %s", exc)
        return ("", {})


# Filesystem types that can be revoked / unmounted mid-run (e.g. a
# wekafs/NFS mount flap). A process whose cwd lives on such a mount sees its
# working directory "vanish underneath it" and any RELATIVE-path write hits
# ``FileNotFoundError``. SGLang's cuda-graph profiling
# (``--enable-profile-cuda-graph``, injected by profile_sglang.yaml) dumps
# ``cuda_graph_runner_memory_usage.pickle`` to a bare relative path, and
# Magpie launches the server via ``cd <inferencex> && bash <script>`` — so
# the server's cwd IS the InferenceX checkout. When that checkout is on
# wekafs and the mount flaps, the dump ENOENTs and the scheduler sigquits
# before any ``.trace.json.gz`` is produced. Such FS types trigger local
# mirroring.
_NETWORK_FS_TYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "lustre",
        "glusterfs",
        "ceph",
        "fuse.weka",
        "wekafs",
        "wekafsgw",
        "fuse.juicefs",
        "fuse.s3fs",
        "fuse.sshfs",
        "9p",
    }
)


def _path_fstype(path: str) -> str:
    """Return the filesystem type backing ``path`` per ``/proc/mounts``.

    Picks the longest mountpoint that is a prefix of the resolved path.
    Returns ``""`` when it cannot be determined (non-Linux, unreadable
    ``/proc/mounts``, ...), which callers treat as "assume local".

    Args:
        path: Filesystem path whose backing mount type is resolved.

    Returns:
        The filesystem type backing ``path``, or ``""`` when it cannot be
        determined.
    """
    try:
        rp = os.path.realpath(path)
    except OSError:
        return ""
    best_mp = ""
    best_type = ""
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                # /proc/mounts octal-escapes spaces etc. in the mountpoint.
                try:
                    mp = parts[1].encode("latin-1").decode("unicode_escape")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    mp = parts[1]
                fstype = parts[2]
                norm = mp.rstrip("/") or "/"
                if norm == "/":
                    is_under = True  # root matches everything (lowest priority)
                else:
                    is_under = rp == norm or rp.startswith(norm + "/")
                if is_under and len(norm) >= len(best_mp):
                    best_mp = norm
                    best_type = fstype
    except OSError:
        return ""
    return best_type


def _is_network_fs(path: str) -> bool:
    """True when ``path`` is backed by a revocable network filesystem.

    Args:
        path: Filesystem path to classify.

    Returns:
        ``True`` when ``path`` lives on a known network filesystem type.
    """
    return _path_fstype(path).lower() in _NETWORK_FS_TYPES


def _ensure_local_inferencex(src: str, *, mirror_key: str = "") -> str:
    """Mirror an InferenceX checkout onto stable local disk.

    #523: returns a local-disk path Magpie can ``cd`` into so the sglang
    server's relative-path cuda-graph snapshot dump survives a network-mount
    (wekafs/NFS) flap. No-op (returns ``src`` unchanged) when:

    * relocation is disabled via
      ``INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX=1``,
    * ``src`` already lives on a local filesystem, or
    * the copy fails for any reason.

    Best-effort — never raises; on failure it falls back to ``src`` so the
    run proceeds (degraded to the pre-fix behaviour) rather than aborting.
    ``mirror_key`` lets callers isolate long-running tasks that share the same
    source checkout: Baseline/Profile pass their task output dir, so a later
    overlapping task cannot ``rmtree`` a mirror another server is still
    ``cd``-ed into.

    Ordering: the caller relocates BEFORE config materialization and passes
    the returned path explicitly into materialization, so the subsequent
    ProfileExecutor patch step (``_after_materialize_config`` reads the
    materialized YAML first) patches the LOCAL mirror in place — the mirror
    therefore ends up carrying the NUM_PROMPTS / PROFILE_EXTRA_BODY patches,
    and Magpie ``cd``-s into the patched local copy.

    Args:
        src: Source InferenceX checkout path (typically on a network mount).
        mirror_key: Optional key isolating concurrent tasks that share the
            same source checkout; folded into the mirror destination name.

    Returns:
        A local-disk mirror path when relocation succeeds, otherwise ``src``
        unchanged (relocation disabled, already local, or copy failed).
    """
    src = str(src)
    if (
        os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX",
            "",
        ).strip()
        == "1"
    ):
        return src
    try:
        if not _is_network_fs(src):
            return src
    except Exception:  # noqa: BLE001 — detection is best-effort
        return src

    real_src = os.path.realpath(src)
    local_root = Path(
        os.environ.get("INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT", "")
        or os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "hyperloom",
            "inferencex_local",
        )
    )
    src_hash = hashlib.sha1(real_src.encode("utf-8")).hexdigest()[:16]
    key_hash = hashlib.sha1(str(mirror_key or "").encode("utf-8")).hexdigest()[:16]
    dest_name = src_hash if not mirror_key else f"{src_hash}-{key_hash}"
    dest = local_root / dest_name
    try:
        local_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "baseline_executor: could not create local InferenceX root %s (%s); using the network-mount checkout.",
            local_root,
            exc,
        )
        return src
    # Lock keyed on dest so concurrent tasks mirroring the same source
    # serialize their rmtree/replace instead of racing (see _mirror_lock).
    lock_path = str(local_root / f".{dest.name}.lock")
    staging: Path | None = None
    try:
        with best_effort_file_lock(lock_path, label="baseline_executor: InferenceX mirror lock"):
            staging = Path(tempfile.mkdtemp(dir=str(local_root)))
            staged_ix = staging / "InferenceX"
            # Copy the tree fresh; re-copy every run because the per-task
            # patch step (_after_materialize_config) rewrites the mirror in
            # place. 8-9 MB onto local disk is sub-second. Holding the lock
            # across rmtree+replace stops a concurrent task swapping ``dest``
            # out from under this copy.
            shutil.copytree(real_src, staged_ix, symlinks=True)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            os.replace(staged_ix, dest)
    except OSError as exc:
        log.warning(
            "baseline_executor: could not mirror InferenceX %s to local disk "
            "(%s); using the network-mount checkout. The #523 cuda-graph "
            "pickle dump may ENOENT if the mount flaps mid-run.",
            real_src,
            exc,
        )
        return src
    finally:
        # Always clear the staging dir: empty after a successful os.replace,
        # or holding a half-finished copy after a failure. Either way it must
        # not accumulate under local_root across runs.
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if not (dest / "benchmarks" / "benchmark_lib.sh").is_file():
        log.warning(
            "baseline_executor: local InferenceX mirror at %s is incomplete; using original %s",
            dest,
            real_src,
        )
        shutil.rmtree(dest, ignore_errors=True)
        return src
    log.info(
        "baseline_executor: #523 — mirrored InferenceX from network mount %s "
        "to local disk %s so the server cwd (cuda-graph pickle dump target) "
        "survives a wekafs/NFS flap.",
        real_src,
        dest,
    )
    return str(dest)


def _probe_aiter_jit_cache() -> dict[str, Any]:
    """Inspect aiter's ``jit/`` dir to decide cold vs warm start.

    Read-only filesystem probe (no subprocess / GPU). Resolution order:
    env override → dynamic find_spec → legacy AITER_JIT_PROBE_PATHS. First
    existing dir wins; counts ``.so`` recursively. Any IO error degrades
    to ``probe_status="error"`` (callers fall back to the WARM timeout).

    Returns a dict with keys:
        path           Path that was probed, or None if nothing found.
        kernel_count   Number of `.so` files under `path` (recursive).
        size_mb        Total size of those `.so` files, in MiB (int).
        is_cold        True iff kernel_count < COLD_START_KERNEL_THRESHOLD;
                       None when probe failed.
        probe_status   "found" | "not_found" | "error".

    Returns:
        dict[str, Any]: Probe info with keys ``path``, ``kernel_count``,
            ``size_mb``, ``is_cold`` and ``probe_status``.
    """
    info: dict[str, Any] = {
        "path": None,
        "kernel_count": 0,
        "size_mb": 0,
        "is_cold": None,
        "probe_status": "not_found",
    }
    candidates: list[str] = []
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(_resolve_aiter_jit_dir_dynamic())
    candidates.extend(AITER_JIT_PROBE_PATHS)

    try:
        chosen: Path | None = None
        for raw in candidates:
            p = Path(raw)
            if p.exists() and p.is_dir():
                chosen = p
                break
        if chosen is None:
            return info
        info["path"] = str(chosen)

        total_bytes = 0
        kernel_count = 0
        for so_path in chosen.rglob("*.so"):
            try:
                total_bytes += so_path.stat().st_size
                kernel_count += 1
            except OSError:
                continue
        info["kernel_count"] = kernel_count
        info["size_mb"] = total_bytes // (1024 * 1024)
        info["is_cold"] = kernel_count < COLD_START_KERNEL_THRESHOLD
        info["probe_status"] = "found"
        return info
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "baseline_executor: aiter jit cache probe failed: %s",
            exc,
        )
        info["probe_status"] = "error"
        info["is_cold"] = None
        return info


def _git_head_sha(repo_path: str) -> str:
    """Return the current HEAD sha of a git repo, or empty string on failure."""
    if not repo_path:
        return ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, capture_output=True, timeout=5, check=True,
        )
        return result.stdout.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def _revert_patches(repo_path: str, pre_sha: str) -> None:
    """Revert the repo to pre_sha after warm-replay patches were applied.

    Prevents patch residue from leaking into subsequent tasks that reuse
    the same InferenceX checkout mirror.
    """
    if not repo_path or not pre_sha:
        return
    try:
        subprocess.run(
            ["git", "reset", "--hard", pre_sha],
            cwd=repo_path, capture_output=True, timeout=15, check=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path, capture_output=True, timeout=15, check=False,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("baseline_executor: patch revert failed: %s", exc)


def _apply_warm_patches(
    params: dict[str, Any],
    target_repo: str,
    output_dir: Path,
) -> list[dict[str, str]]:
    """Apply warm-replay code patches (Phase 0+1) to InferenceX checkout.

    Reads ``params["patches"]`` (list of dicts with patch_file/patch_content/
    patch_ref) and ``params["blocked_patches"]`` (blocklist). Applies each patch
    via ``git apply`` in the target repo. Skips patches that appear in the
    blocklist. Returns list of successfully applied patch metadata dicts.

    If target_repo is empty or no patches are present, returns [].
    """
    patches = params.get("patches") or []
    if not patches or not target_repo:
        return []

    blocked = {
        p.get("patch_file", "") for p in (params.get("blocked_patches") or [])
    }

    applied: list[dict[str, str]] = []
    patch_log_dir = output_dir / "warm_patches"
    patch_log_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch in enumerate(patches):
        patch_file = patch.get("patch_file") or ""
        patch_content = patch.get("patch_content") or ""
        patch_ref = patch.get("patch_ref") or ""

        if patch_file in blocked:
            log.info(
                "baseline_executor: skipping blocked patch %s", patch_file,
            )
            continue

        if not patch_content and not patch_ref:
            log.warning(
                "baseline_executor: patch entry has no content/ref, skipping: %s",
                patch_file,
            )
            continue

        # Resolve patch content: prefer inline content, fallback to patch_ref file.
        if not patch_content and patch_ref:
            ref_path = Path(patch_ref)
            if ref_path.is_file():
                try:
                    patch_content = ref_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    log.warning(
                        "baseline_executor: cannot read patch_ref %s: %s",
                        patch_ref, exc,
                    )
                    continue
            else:
                log.warning(
                    "baseline_executor: patch_ref not found: %s", patch_ref,
                )
                continue

        # Write patch to temp file then apply.
        patch_path = patch_log_dir / f"{idx:03d}_{Path(patch_file).stem or 'patch'}.diff"
        patch_path.write_text(patch_content, encoding="utf-8")

        try:
            subprocess.run(
                ["git", "apply", "--stat", "--check", str(patch_path)],
                cwd=target_repo,
                capture_output=True,
                timeout=30,
                check=True,
            )
            subprocess.run(
                ["git", "apply", str(patch_path)],
                cwd=target_repo,
                capture_output=True,
                timeout=30,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            log.warning(
                "baseline_executor: git apply failed for patch %s: %s",
                patch_file, exc.stderr.decode(errors="replace")[:500] if exc.stderr else str(exc),
            )
            continue
        except (subprocess.TimeoutExpired, OSError) as exc:  # pragma: no cover
            log.warning(
                "baseline_executor: patch apply error for %s: %s",
                patch_file, exc,
            )
            continue

        applied.append({"patch_file": patch_file, "idx": str(idx)})

    return applied


class BaselineExecutor:
    """Class form for tests / DI; ``baseline_executor`` is the bare callable.

    ``session_dir`` is the session root for the per-task workspace
    (``<sd>/runs/baseline/<task_id>/``); used only as a fallback when the
    SubAgentRunner injects a pre-created workspace via ``ctx.extra``.
    """

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        shared_state: Any | None = None,
        default_timeout_sec: int = BASELINE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
    ):
        """Initialize the baseline executor with launch defaults.

        Args:
            magpie_python (str | None): Python interpreter used to invoke
                Magpie; resolved automatically when ``None``.
            default_config_path (Path | str | None): Default Magpie YAML config
                path; resolved from ``$FRAMEWORK`` at call time when ``None``.
            session_dir (Path | str | None): Canonical session root for
                per-task workspaces; resolved automatically when ``None``.
            shared_state: Optional live SharedState object. When provided, the
                eager-fallback one-shot is consumed in memory before saving so
                Coordinator cannot later re-persist a stale True value.
            default_timeout_sec (int): Default (warm-start) subprocess timeout.
            cwd (Path | str): Working directory for the Magpie subprocess.
        """
        from ._grid_runner import _resolve_magpie_python, _resolve_session_dir

        self.magpie_python = magpie_python or _resolve_magpie_python()
        # None = resolve from $FRAMEWORK at call time; explicit fixture path wins.
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.shared_state = shared_state
        self.default_timeout_sec = default_timeout_sec
        self.cwd = Path(cwd)

    def _resolve_default_config(self) -> Path:
        """Hook for subclasses (ProfileExecutor) to swap the resolver.

        Returns:
            Path: The default baseline Magpie YAML config path.
        """
        return _default_baseline_config()

    def _resolve_workspace(self, ctx: RunnerContext, action: str) -> Path:
        """Pick the per-task workspace dir.

        Order: ``task.params['output_dir']`` → ``ctx.extra['workspace']``
        → ``runs_dir(...)`` (direct-instantiation fallback for tests).

        Args:
            ctx: Runner context carrying ``task.params`` and ``extra``.
            action: Action name used when falling back to ``runs_dir(...)``.

        Returns:
            The resolved per-task workspace directory.
        """
        params = ctx.task.params or {}
        if params.get("output_dir"):
            return Path(params["output_dir"])
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("workspace"):
            return Path(extra["workspace"])
        return runs_dir(self.session_dir, action, ctx.task.task_id)

    def _eager_fallback_armed(self, shared_state: Any | None = None) -> bool:
        """Peek the one-shot eager fallback flag WITHOUT consuming it.

        Used to keep the flag armed when the framework is unknown (cannot pick
        a safe disable-cuda-graph flag), so the one-shot is not wasted.
        Best-effort: missing/unreadable state reads as not armed.

        Args:
            shared_state: Optional live SharedState; falls back to
                ``self.shared_state`` and then a loaded session state.

        Returns:
            ``True`` when the one-shot eager-fallback flag is currently armed.
        """
        try:
            state = shared_state or self.shared_state
            if state is None:
                from ...state.shared_state import SharedState

                state = SharedState.load_or_init(self.session_dir)
            return bool(getattr(state, "baseline_eager_fallback", False))
        except Exception:  # noqa: BLE001 — fallback must never break baseline
            log.debug(
                "baseline_executor: eager-fallback flag peek failed",
                exc_info=True,
            )
            return False

    def _consume_eager_fallback(self, shared_state: Any | None = None) -> bool:
        """Consume the one-shot cuda-graph eager fallback flag from SharedState.

        Returns True (and clears the flag) when a prior baseline armed it.
        Best-effort: missing/unreadable state reads as no fallback.

        Args:
            shared_state: Optional live SharedState; falls back to
                ``self.shared_state`` and then a loaded session state.

        Returns:
            ``True`` when the flag was armed (and is now cleared), else
            ``False``.
        """
        try:
            state = shared_state or self.shared_state
            if state is None:
                from ...state.shared_state import SharedState

                state = SharedState.load_or_init(self.session_dir)
            if not getattr(state, "baseline_eager_fallback", False):
                return False
            state.baseline_eager_fallback = False
            state.save(self.session_dir)
            return True
        except Exception:  # noqa: BLE001 — fallback must never break baseline
            log.debug(
                "baseline_executor: eager-fallback flag check failed",
                exc_info=True,
            )
            return False

    def _resolve_timeout(self, params: dict[str, Any]) -> int:
        """Pick the subprocess timeout for this baseline launch.

        Order: explicit ``task.params['timeout_sec']`` → cold-start cap
        when the aiter jit probe reports COLD (env-overridable via
        ``INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC``) → warm default.
        Every path emits one log line for greppability.

        Args:
            params: Task params; an explicit ``timeout_sec`` overrides the
                probe-based selection.

        Returns:
            The subprocess timeout in seconds for this baseline launch.
        """
        explicit = params.get("timeout_sec")
        if explicit:
            timeout_sec = int(explicit)
            log.info(
                "baseline_executor: timeout=%ds (explicit task param)",
                timeout_sec,
            )
            return timeout_sec

        cache = _probe_aiter_jit_cache()
        cold_cap = int(
            os.environ.get(
                "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
                BASELINE_COLD_START_TIMEOUT_SEC,
            )
        )
        if cache["probe_status"] == "found" and cache["is_cold"]:
            # Before paying the cold-start compile, reap aiter JIT locks left
            # by a killed hipcc (else this build spins forever in FileBaton's
            # untimed wait). Gated on "no live compiler" so a concurrent
            # benchmark's in-flight compile on the node-global jit dir is never
            # disturbed; ninja resumes incrementally from any surviving .o.
            sweep = sweep_stale_aiter_locks_if_dead()
            if sweep.get("skipped_live"):
                log.info(
                    "baseline_executor: aiter lock sweep skipped — live "
                    "compiler process present (jit dir node-shared).",
                )
            elif sweep.get("deleted"):
                log.warning(
                    "baseline_executor: reaped %d stale aiter JIT lock(s) "
                    "under %s (compiler_alive=%s) before cold start.",
                    sweep["deleted"], sweep.get("dir"),
                    sweep.get("compiler_alive"),
                )
                # Locks gone — re-probe so the log line below reflects reality.
                cache = _probe_aiter_jit_cache()
        if cache["probe_status"] == "found" and cache["is_cold"]:
            log.warning(
                "baseline_executor: COLD_START detected — aiter jit/build/ "
                "at %s has %d .so (< %d threshold), %d MB. Bumping timeout "
                "%ds -> %ds. First-time JIT compile on a new "
                "(model, dtype, TP, max_model_len) signature can take 30+ "
                "minutes for large FP8 / MoE models.",
                cache["path"],
                cache["kernel_count"],
                COLD_START_KERNEL_THRESHOLD,
                cache["size_mb"],
                self.default_timeout_sec,
                cold_cap,
            )
            return cold_cap
        if cache["probe_status"] == "found":
            log.info(
                "baseline_executor: WARM start — aiter jit/build/ at %s has %d .so, %d MB. Using default timeout=%ds.",
                cache["path"],
                cache["kernel_count"],
                cache["size_mb"],
                self.default_timeout_sec,
            )
            return self.default_timeout_sec
        log.warning(
            "baseline_executor: aiter jit cache not located "
            "(probe_status=%s). Using default timeout=%ds. Cold-start "
            "auto-bump disabled for this run.",
            cache["probe_status"],
            self.default_timeout_sec,
        )
        return self.default_timeout_sec

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Hook for subclasses after YAML materialization, before launch.

        ProfileExecutor uses this to patch/validate the InferenceX checkout
        named by the rendered YAML. No-op default keeps baseline unchanged.

        Args:
            config_path: The materialized Magpie YAML config path.
            output_dir: The per-task workspace directory.

        Returns:
            An early-return result dict to short-circuit the launch, or
            ``None`` to proceed with the baseline run.
        """
        return None

    @staticmethod
    def _is_eval_rooted_failure(result: dict[str, Any]) -> bool:
        """Whether a failed baseline result was caused by the accuracy eval.

        Scans the result's ``error`` tail + ``nonfatal_warnings`` and then any
        benchmark stdout/stderr + ``server.log`` under the run's ``output_dir``
        for ``run_eval``-failure markers. Recursive so the double-run path is
        covered (the warmup round's logs carry the marker even when the measure
        round failed for a downstream reason). Never raises.

        Args:
            result: A ``status="failed"`` baseline result dict.

        Returns:
            ``True`` when an eval-failure marker is found, else ``False``.
        """

        def _hit(text: str) -> bool:
            return any(m in text for m in _EVAL_FAILURE_MARKERS)

        if _hit(str(result.get("error") or "")):
            return True
        for w in result.get("nonfatal_warnings") or []:
            if _hit(str(w)):
                return True
        out_dir = result.get("output_dir")
        if not out_dir:
            return False
        root = Path(out_dir)
        # Double-run: the returned result is the measure round, but the eval
        # failure (and its markers) live in the sibling warmup round. Climb to
        # the shared task root so the scan covers both rounds — bounded to the
        # named round subdirs so we never scan a sibling baseline task.
        if root.name in ("warmup_round", "measure_round"):
            root = root.parent
        if not root.exists():
            return False
        log_names = ("benchmark_stderr.log", "benchmark_stdout.log", "server.log")
        seen = 0
        try:
            for path in root.rglob("*.log"):
                if path.name not in log_names:
                    continue
                seen += 1
                if seen > 64:  # bound the scan on pathological trees
                    break
                try:
                    with path.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - _EVAL_SCAN_MAX_BYTES))
                        chunk = f.read().decode("utf-8", "replace")
                except OSError:
                    continue
                if _hit(chunk):
                    return True
        except OSError:
            return False
        return False

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the Magpie baseline, with a one-shot eval-failure fallback.

        Delegates to :meth:`_run_once`. When the run fails for an
        eval-rooted reason (InferenceX ``run_eval`` aborted the benchmark even
        though throughput was healthy — e.g. the redundant
        ``--concurrent-requests`` flag, or any lm-eval breakage) AND accuracy
        eval was active, it re-runs **once** with ``RUN_EVAL=false`` so the
        throughput baseline is salvaged. The retried result is tagged
        ``accuracy_source="eval_unavailable"`` and carries a
        ``eval_failed_fallback_no_accuracy`` warning. This is the in-executor
        complement to the install-time flag strip in ``_magpie_patcher.py``:
        even if the upstream script breaks again, the run degrades gracefully
        instead of terminating after 3 baseline attempts.

        Args:
            ctx (RunnerContext): The runner context carrying ``task.params``
                (config / model / timeout knobs) and ``extra`` (workspace).

        Returns:
            dict[str, Any]: The baseline result dict (see :meth:`_run_once`).

        Raises:
            FileNotFoundError: If the resolved baseline config does not exist.
        """
        result = await self._run_once(ctx)
        params = ctx.task.params or {}
        # "Already off" only when the operator explicitly disabled eval — via
        # the param, or an extra_envs RUN_EVAL that is PRESENT and falsey. An
        # absent RUN_EVAL must NOT count (the empty string is in
        # _RUN_EVAL_FALSE_VALUES for the materialize-time default, not here).
        _extra_envs = params.get("extra_envs") or {}
        _explicit_run_eval = "RUN_EVAL" in _extra_envs and str(
            _extra_envs["RUN_EVAL"]
        ).strip().lower() in _RUN_EVAL_FALSE_VALUES
        eval_already_off = _is_truthy(params.get("disable_run_eval")) or _explicit_run_eval
        if (
            result.get("status") != "succeeded"
            and not eval_already_off
            and self._is_eval_rooted_failure(result)
        ):
            log.warning(
                "baseline_executor: failure looks eval-rooted (InferenceX "
                "run_eval aborted the benchmark); retrying once with "
                "RUN_EVAL=false to salvage the throughput baseline without "
                "the accuracy gate."
            )
            retry = await self._run_once(ctx, force_disable_eval=True)
            retry.setdefault("nonfatal_warnings", [])
            retry["nonfatal_warnings"].append("eval_failed_fallback_no_accuracy")
            if retry.get("status") == "succeeded":
                retry["accuracy_source"] = "eval_unavailable"
            return retry
        return result

    async def _run_once(
        self,
        ctx: RunnerContext,
        *,
        force_disable_eval: bool = False,
    ) -> dict[str, Any]:
        """Run the Magpie baseline benchmark and parse its result.

        Materializes the workload config, resolves the timeout (with cold-start
        detection), restarts the multi-node server when required, launches
        Magpie via ``run_with_session_kill``, harvests leaked artifacts, parses
        ``benchmark_report.json`` and the accuracy eval, and returns a result
        dict the Coordinator promotes into SharedState.

        Args:
            ctx (RunnerContext): The runner context carrying ``task.params``
                (config / model / timeout knobs) and ``extra`` (workspace).
            force_disable_eval: When True, force ``RUN_EVAL=false`` into the
                materialized config (the eval-failure fallback path); also set
                by the ``disable_run_eval`` task param.

        Returns:
            dict[str, Any]: On success, a ``status="succeeded"`` dict with
                throughput / latency / accuracy measurements and artifact
                paths; on failure, a ``status="failed"`` dict with an
                ``error_class`` (``timeout``, ``subprocess_nonzero``,
                ``no_workspace``, ``no_report``, ``invalid_measurement`` ...).

        Raises:
            FileNotFoundError: If the resolved baseline config does not exist.
        """
        params = ctx.task.params or {}
        # Only a genuine ``baseline`` task may establish/overwrite the quality
        # reference image. ``replay_warm_recipe`` reuses this executor but is an
        # optimization candidate, so it must compare against the pure baseline
        # reference rather than redefine it (otherwise the gate would mask the
        # warm recipe's own deviation from the baseline output).
        is_genuine_baseline = _should_establish_quality_ref(getattr(ctx.task, "kind", ""))
        config_path = Path(params.get("config_path") or self.default_config_path or self._resolve_default_config())
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")

        # One-shot cuda-graph eager fallback: a prior baseline hit a cuda-graph
        # capture failure and armed state.baseline_eager_fallback. Inject the
        # framework-correct disable-cuda-graph flag for this retry and consume
        # the flag so it fires once. Resolve framework FIRST: an unknown
        # framework cannot pick a safe flag, so we leave the flag armed (do not
        # consume) and let a later baseline with a known framework apply it,
        # instead of burning the one-shot on a no-op retry.
        effective_extra_server_args = read_extra_server_args(params)
        extra = getattr(ctx, "extra", None) or {}
        live_shared_state = extra.get("shared_state") or self.shared_state
        fw = str(params.get("framework") or "").strip() or os.environ.get("FRAMEWORK", "").strip()
        if not fw and self._eager_fallback_armed(live_shared_state):
            log.warning(
                "baseline_executor: eager fallback is armed but framework is "
                "unknown; leaving the one-shot armed (not consuming) so a "
                "later baseline with a known framework can apply it",
            )
        elif fw and self._consume_eager_fallback(live_shared_state):
            cg_flag = _disable_cuda_graph_flag(fw)
            effective_extra_server_args = _with_cuda_graph_disabled(
                effective_extra_server_args,
                fw,
            )
            log.warning(
                "baseline_executor: retrying with %s after a prior cuda-graph capture failure (framework=%s)",
                cg_flag,
                fw,
            )

        output_dir = self._resolve_workspace(ctx, "baseline")
        output_dir.mkdir(parents=True, exist_ok=True)

        # #523: keep the InferenceX checkout Magpie will ``cd`` into on stable
        # local disk. Magpie launches the server via ``cd <inferencex> && bash
        # <script>`` (see Magpie ``_build_local_command``), so that checkout is
        # the server's cwd — and SGLang's cuda-graph profiling dumps
        # ``cuda_graph_runner_memory_usage.pickle`` there via a RELATIVE path.
        # On wekafs/NFS a mid-run mount flap makes the cwd vanish and the dump
        # ENOENTs (scheduler sigquit, no trace). Relocate BEFORE materialize so
        # the rendered ``benchmark.inferencex_path`` (which Magpie actually
        # honours — the MAGPIE_INFERENCEX_PATH env is only a fallback), the
        # ProfileExecutor patch step, and Magpie all use the local mirror.
        #
        # Keep this value task-local. Mutating process-wide $INFERENCEX_PATH
        # would let two overlapping asyncio tasks race between relocation,
        # materialization and Magpie env export. Passing the same explicit path
        # through all three call sites also documents the invariant directly.
        ix_env = os.environ.get("INFERENCEX_PATH", "").strip()
        effective_inferencex_path = _ensure_local_inferencex(ix_env, mirror_key=str(output_dir)) if ix_env else ""

        # Phase 0+1: apply warm-replay code patches before server launch.
        # Record pre-apply HEAD so patches are reverted after the benchmark
        # completes (or fails), preventing residue in the shared checkout.
        patch_target = effective_inferencex_path or ix_env
        _pre_patch_sha = _git_head_sha(patch_target)
        applied_patches = _apply_warm_patches(params, patch_target, output_dir)
        if applied_patches:
            log.info(
                "baseline_executor: applied %d warm-replay code patches (pre_sha=%s): %s",
                len(applied_patches), _pre_patch_sha[:8],
                [p["patch_file"] for p in applied_patches],
            )

        timeout_sec = self._resolve_timeout(params)
        # Model path: task.params['model_path'] > $MODEL_PATH > SharedState;
        # if none, leave the YAML's hardcoded `model:` for fixture-based tests.
        # Read live state from ctx.extra (Coordinator path: the executor is a
        # module-level singleton with self.shared_state=None), mirroring the
        # eager-fallback resolution above; the fallback stops a real run (params
        # + env both unset) from leaking the YAML bare model name into
        # --model-path, which sglang treats as an HF repo id.
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
            or str(getattr(live_shared_state, "model_path", "") or "").strip()
        )
        # gpu_type: task.params > $GPU_TYPE (cli.py canonicalizes mi325x->mi300x).
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        # Orchestration-supplied script + result_dir overrides (route around
        # scripts that hardcode ``--result-dir /workspace/``). Sanitization
        # turns a malformed override into ``error_class=bad_param``.
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        # Reference recipe base (lowest priority): seeds every baseline incl.
        # resume restarts. Task params may also pass it through explicitly;
        # prefer the explicit param, else read the model-gated SharedState value.
        ref_args = str(params.get("reference_server_args") or "").strip()
        ref_envs = dict(params.get("reference_envs") or {})
        if not ref_args and not ref_envs:
            ref_args, ref_envs = _resolve_reference_base(
                self.session_dir, model_path=resolved_model,
            )
        # Accuracy eval (GSM8K) opt-out: the ``disable_run_eval`` task param
        # (documented to the LLM + fingerprinted) and the in-executor
        # eval-failure fallback both force ``RUN_EVAL=false`` via extra_envs,
        # which materialize honors over the default-true. An explicit
        # extra_envs RUN_EVAL still loses to the deliberate disable.
        base_extra_envs = dict(params.get("extra_envs") or {})
        if force_disable_eval or _is_truthy(params.get("disable_run_eval")):
            base_extra_envs["RUN_EVAL"] = "false"
        try:
            config_path = materialize_config_with_envs(
                config_path,
                output_dir,
                extra_server_args=effective_extra_server_args,
                extra_envs=base_extra_envs,
                model_path=resolved_model,
                gpu_type=resolved_gpu,
                inferencex_path=effective_inferencex_path,
                benchmark_script=override_script,
                reference_server_args=ref_args,
                reference_envs=ref_envs,
                establish_quality_ref=is_genuine_baseline,
            )
        except FrameworkScriptMismatchError as exc:
            # Cross-framework script override (e.g. sglang_*.sh on a vllm run):
            # return a structured failure instead of bubbling to coordinator.
            return {
                "status": "failed",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        # Stash for the result so Coordinator can reuse it downstream
        # (workload-contract reuse).
        materialized_config_path = config_path
        hook_result = self._after_materialize_config(config_path, output_dir)
        if hook_result is not None:
            hook_result.setdefault("materialized_config", str(config_path))
            hook_result.setdefault("output_dir", str(output_dir))
            return hook_result

        # Cold-start "warmup artifact" guard: the freshly-booted server's
        # first benchmark window pays one-time cold costs (JIT, graph
        # capture, KV alloc, clock warmup) that the client-side warmups
        # can't absorb, inflating later gains into fictitious "improvements".
        # Fix: run TWICE against the SAME persistent server via Magpie's
        # ``server_lifecycle`` reuse — round 1 boots + pays cold costs,
        # round 2 re-attaches to the hot server and is the clean baseline.
        # Eligibility (else single round): double-run env enabled,
        # single-node, benchmark script is a Magpie built-in, profiler off.
        lifecycle = self._resolve_lifecycle_params(materialized_config_path)
        double_run = self._double_run_enabled() and lifecycle["eligible"]

        common = {
            "timeout_sec": timeout_sec,
            "override_result_dir": override_result_dir,
            "resolved_model": resolved_model,
            "materialized_config_path": materialized_config_path,
            "inferencex_path": effective_inferencex_path,
            "effective_extra_server_args": effective_extra_server_args,
            "params": params,
            "ctx": ctx,
        }

        if not double_run:
            if self._double_run_enabled() and not lifecycle["eligible"]:
                log.info(
                    "baseline_executor: cold-start double-run not eligible (%s); running single round.",
                    lifecycle["reason"],
                )
            return await self._run_single_benchmark(
                config_path=config_path,
                output_dir=output_dir,
                **common,
            )

        framework = lifecycle["framework"]
        port = lifecycle["port"]
        # pid_dir is SHARED across both rounds (Magpie keys the persistent
        # server by ``<pid_dir>/<framework>_<port>.{pid,json}``) so round 2
        # discovers round 1's server. Task root keeps it per-task isolated.
        pid_dir = output_dir
        try:
            # Deep-clean zombie listeners + stale pid/meta BEFORE round 1
            # boots its server. Runs once here (not per round) so round 1's
            # persistent server survives for round 2's re-attach.
            self._pre_start_cleanup(
                pid_dir=pid_dir,
                framework=framework,
                port=port,
            )
            # Round 1 (warmup): boot + run, leave running (cleanup=false) so
            # round 2 can re-attach. Throughput discarded (cold-contaminated).
            warmup_dir = output_dir / "warmup_round"
            warmup_cfg = self._write_lifecycle_config(
                materialized_config_path,
                warmup_dir,
                cleanup=False,
                pid_dir=pid_dir,
                port=port,
            )
            log.info(
                "baseline_executor: cold-start guard — warmup round (discarded, boots persistent server) in %s",
                warmup_dir,
            )
            warmup_result = await self._run_single_benchmark(
                config_path=warmup_cfg,
                output_dir=warmup_dir,
                **common,
            )
            if warmup_result.get("status") != "succeeded":
                # Warmup failure almost certainly recurs, so skip the
                # measured round; the finally block tears down any leak.
                warmup_result.setdefault("nonfatal_warnings", [])
                warmup_result["nonfatal_warnings"].append(
                    "baseline_warmup_round_failed",
                )
                log.warning(
                    "baseline_executor: warmup round failed (error_class=%s); skipping measured round",
                    warmup_result.get("error_class"),
                )
                return warmup_result
            warmup_tput = warmup_result.get("output_throughput")
            warmup_runtime = warmup_result.get("subprocess_runtime_sec")

            # Round 2 (measured): re-attach to the hot server (client only).
            # cleanup=true tears it down on the happy path; finally is the net.
            measure_dir = output_dir / "measure_round"
            measure_cfg = self._write_lifecycle_config(
                materialized_config_path,
                measure_dir,
                cleanup=True,
                pid_dir=pid_dir,
                port=port,
            )
            log.info(
                "baseline_executor: cold-start guard — measured baseline "
                "round in %s (warmup tput=%.1f tok/s discarded, reusing "
                "hot server)",
                measure_dir,
                warmup_tput or 0.0,
            )
            result = await self._run_single_benchmark(
                config_path=measure_cfg,
                output_dir=measure_dir,
                **common,
            )
            if result.get("status") == "succeeded":
                result.setdefault("nonfatal_warnings", [])
                result["nonfatal_warnings"].append(
                    "baseline_double_run_discarded_first",
                )
                result["warmup_round_tput"] = warmup_tput
                # Overtime-kill anchor fix: the Coordinator promotes
                # ``subprocess_runtime_sec`` into baseline_runtime_sec, the
                # explore soft-kill anchor. Explore variants restart the
                # server, so report round 1's FULL boot+client wall-clock
                # (matches their profile); round 2's client-only time stays
                # under a separate key.
                if isinstance(warmup_runtime, (int, float)) and warmup_runtime > 0:
                    result["measure_round_runtime_sec"] = result.get(
                        "subprocess_runtime_sec",
                    )
                    result["subprocess_runtime_sec"] = round(
                        float(warmup_runtime),
                        2,
                    )
                _hot = result.get("output_throughput") or 0.0
                _cold = warmup_tput or 0.0
                log.info(
                    "baseline_executor: cold-start guard — measured "
                    "baseline=%.1f tok/s (warmup=%.1f tok/s discarded; "
                    "artifact would have been +%.0f%%)",
                    _hot,
                    _cold,
                    ((_hot / _cold - 1.0) * 100.0) if _cold > 0 else 0.0,
                )
            return result
        finally:
            # Defensive teardown so no persistent server leaks regardless of
            # which round failed. Idempotent (no-op on the happy path).
            self._teardown_lifecycle_server(
                pid_dir=pid_dir,
                framework=framework,
                port=port,
            )
            # Revert warm-replay patches to prevent state leakage into
            # subsequent tasks that reuse the same InferenceX checkout.
            if applied_patches and _pre_patch_sha:
                _revert_patches(patch_target, _pre_patch_sha)

    @staticmethod
    def _double_run_enabled() -> bool:
        """Whether baseline double-run is enabled.

        Controlled by ``INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN``.

        Returns:
            ``True`` unless the env var is set to a falsey value.
        """
        return os.environ.get(
            "INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN",
            "1",
        ).strip().lower() not in ("0", "false", "no", "")

    def _resolve_lifecycle_params(
        self,
        materialized_config_path: Path,
    ) -> dict[str, Any]:
        """Inspect the materialized YAML for server_lifecycle eligibility.

        Args:
            materialized_config_path: The materialized Magpie YAML config path.

        Returns:
            Lifecycle params including eligibility, framework, port and the
            reason a run is ineligible.
        """
        return _lifecycle.resolve_lifecycle_params(materialized_config_path)

    def _write_lifecycle_config(
        self,
        base_config_path: Path,
        dest_dir: Path,
        *,
        cleanup: bool,
        pid_dir: Path,
        port: int,
    ) -> Path:
        """Render a per-round YAML injecting ``benchmark.server_lifecycle``.

        Both rounds share ``pid_dir`` + ``port`` so round 2 re-attaches;
        only ``cleanup`` differs (round 1 persists, round 2 tears down).

        Args:
            base_config_path: Source materialized YAML to clone and patch.
            dest_dir: Directory the per-round YAML is written into.
            cleanup: Whether the server should be torn down after the round.
            pid_dir: Shared pid/metadata directory keying the persistent
                server across both rounds.
            port: Server port shared across both rounds.

        Returns:
            Path to the written per-round lifecycle YAML.
        """
        with Path(base_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bench = cfg.setdefault("benchmark", {})
        _lifecycle.inject_lifecycle(
            bench,
            cleanup=cleanup,
            pid_dir=pid_dir,
            port=port,
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = Path(dest_dir) / "baseline_lifecycle.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out

    @staticmethod
    def _port_healthy(port: int, timeout: float = 3.0) -> bool:
        """Return True when localhost:{port}/health responds HTTP 200.

        Args:
            port: Local server port to probe.
            timeout: Per-request timeout in seconds.

        Returns:
            ``True`` when the health endpoint responds HTTP 200, else
            ``False``.
        """
        import urllib.request

        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=timeout,
            )
            return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _pre_start_cleanup(
        self,
        *,
        pid_dir: Path,
        framework: str,
        port: int,
    ) -> None:
        """Best-effort startup pre-clean for the double-run path.

        Only acts when there is concrete evidence of a zombie: the reuse
        port responds to /health but the matching metadata file is absent
        (the exact "Reuse metadata mismatch" trigger). In that case it
        calls _kill_stale_servers() to reap the orphan listener. Stale
        pid/json files are always unlinked (without sending signals to
        potentially-recycled PIDs). Never raises.

        Args:
            pid_dir: Directory holding the server pid/metadata files.
            framework: Framework name used to build the server tag.
            port: Server port used to build the server tag.
        """
        base = Path(pid_dir)
        tag = f"{framework}_{port}"
        pid_file = base / f"{tag}.pid"
        meta_file = base / f"{tag}.json"
        meta_exists = meta_file.exists()
        try:
            port_healthy = self._port_healthy(port)
        except Exception as exc:  # noqa: BLE001 — best-effort pre-clean
            log.warning(
                "baseline_executor: pre-start port probe failed (%s); proceeding.",
                exc,
            )
            port_healthy = False
        if meta_exists and port_healthy:
            # A healthy reuse target with metadata is not a zombie; keep the
            # files so Magpie can reattach instead of creating a mismatch.
            return
        # Unlink stale metadata/pid files only (no signal to possibly-
        # recycled PIDs — _teardown_lifecycle_server is too aggressive here).
        for p in (pid_file, meta_file):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        # Narrow trigger: only deep-clean when the port is occupied by a
        # zombie (healthy endpoint, no metadata). Avoids killing unrelated
        # servers sharing the pod.
        try:
            if not meta_exists and port_healthy:
                _kill_stale_servers()
        except Exception as exc:  # noqa: BLE001 — best-effort pre-clean
            log.warning(
                "baseline_executor: pre-start _kill_stale_servers failed (%s); proceeding.",
                exc,
            )

    def _teardown_lifecycle_server(
        self,
        *,
        pid_dir: Path,
        framework: str,
        port: int,
    ) -> None:
        """Best-effort teardown of a persistent server left by the
        double-run rounds. Idempotent and never raises (safe in finally).

        Args:
            pid_dir: Directory holding the server pid/metadata files.
            framework: Framework name used to build the server tag.
            port: Server port used to build the server tag.
        """
        _lifecycle.teardown_lifecycle_server(
            pid_dir=pid_dir,
            framework=framework,
            port=port,
        )

    async def _run_single_benchmark(
        self,
        *,
        config_path: Path,
        output_dir: Path,
        timeout_sec: int,
        override_result_dir: str | None,
        resolved_model: str,
        materialized_config_path: Path,
        inferencex_path: str,
        effective_extra_server_args: str,
        params: dict[str, Any],
        ctx: RunnerContext,
    ) -> dict[str, Any]:
        """Run one Magpie benchmark subprocess and parse its result.

        Single-round core extracted from ``__call__`` so the cold-start
        guard can invoke it twice. ``output_dir`` is the per-round slot.

        Args:
            config_path: The materialized Magpie YAML config for this round.
            output_dir: The per-round workspace slot.
            timeout_sec: Subprocess timeout in seconds.
            override_result_dir: Optional ``$RESULT_DIR`` override for the
                benchmark wrapper.
            resolved_model: Resolved model path for the run.
            materialized_config_path: The canonical materialized YAML, echoed
                into the result for downstream reuse.
            inferencex_path: Task-local InferenceX checkout path pinned via
                ``MAGPIE_INFERENCEX_PATH``.
            effective_extra_server_args: Extra server args passed to the
                multi-node restart helper.
            params: Task params for this launch.
            ctx: Runner context carrying ``extra`` (e.g. multi-node round
                state).

        Returns:
            A result dict: ``status="succeeded"`` with measurements on
            success, or ``status="failed"`` with an ``error_class`` on
            failure.
        """
        cmd = [
            self.magpie_python,
            "-m",
            "Magpie",
            "-v",
            "benchmark",
            "--benchmark-config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-mode",
            "local",
        ]
        env = os.environ.copy()
        # Put the venv first in PATH so the benchmark script's `python3`
        # resolves to one with torch+rocm (defense in depth vs Magpie YAML).
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
        # Pin Magpie's InferenceX resolution to the same per-task
        # checkout rendered into benchmark.inferencex_path and patched by
        # ProfileExecutor. Do not re-read process env here; the task-local
        # explicit value avoids cross-task races on $INFERENCEX_PATH.
        if inferencex_path:
            env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Always-on ``$RESULT_DIR`` default for scripts that respect it
        # (else they fall back to ``/workspace/``); scripts that ignore it
        # are caught by the ``extract_benchmark_measurement`` salvage pass.
        env["RESULT_DIR"] = override_result_dir or str(output_dir)
        # Pin SERVER_LOG / GPU_METRICS_CSV per-task so wrappers write into
        # the task workspace instead of leaking to ``/workspace/``;
        # ``harvest_leaked_artifacts`` below is the defense-in-depth net.
        env["SERVER_LOG"] = str(output_dir / "server.log")
        env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")

        # Multi-node (--nodes >= 2): inject MAGPIE_RUN_PHASE=client +
        # BENCHMARK_BASE_URL so Magpie skips its server launch and targets
        # the RayJob head. No-op ({}) in single-node.
        from ._multi_node_env import magpie_remote_env

        env.update(magpie_remote_env())

        # Multi-node only: restart sglang/vllm per round for a fresh server
        # (parity with single-node PHASE=all). No-op in single-node. Profile
        # rounds set ctx.extra["mn_round_restarted"] to claim the restart so
        # each Magpie spawn maps to exactly one server boot.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )

        ctx_extra = getattr(ctx, "extra", None) or {}
        if not ctx_extra.get("mn_round_restarted"):
            try:
                # Merge the reference base UNDER the per-task args (lowest
                # priority, last-wins) so a multi-node per-round restart carries
                # the same reference flags the single-node materialized YAML
                # does — else exotic-arch models re-fail on every MN round.
                # Resolve here (separate method from __call__): prefer explicit
                # params, else the model-gated SharedState value.
                from ._grid_runner import merge_server_args
                _mn_ref_args = str(params.get("reference_server_args") or "").strip()
                _mn_ref_envs = dict(params.get("reference_envs") or {})
                if not _mn_ref_args and not _mn_ref_envs:
                    _mn_ref_args, _mn_ref_envs = _resolve_reference_base(
                        self.session_dir, model_path=resolved_model,
                    )
                # Base on effective_extra_server_args (carries the one-shot
                # cuda-graph eager-fallback flag when armed) rather than raw
                # params, so the MN per-round restart keeps that fallback too.
                _mn_task_args = effective_extra_server_args
                _mn_server_args = (
                    merge_server_args(_mn_ref_args, _mn_task_args)
                    if _mn_ref_args else _mn_task_args
                )
                _mn_env = {str(k): str(v) for k, v in _mn_ref_envs.items()}
                # PD knobs auto-resolved by the helper from $PD_* env (set
                # by cli.py), falling back to state.json — call site is
                # identical between colocated and disaggregated runs.
                await restart_server_for_round(
                    extra_server_args=_mn_server_args,
                    extra_env=_mn_env or None,
                    framework=os.environ.get("FRAMEWORK") or None,
                    model_path=resolved_model or None,
                    tp=int(os.environ.get("TP") or 0) or None,
                    ep=int(os.environ.get("EP") or 0) or None,
                )
            except ServerRestartFailed as exc:
                return {
                    "status": "failed",
                    "error_class": "mn_server_restart_failed",
                    "error": str(exc),
                    "output_dir": str(output_dir),
                }

        from ._multi_node_env import log_mn_banner

        log_mn_banner("baseline_executor", log, output_dir=str(output_dir))
        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s", cmd, output_dir)

        # Magpie launched via ``run_with_session_kill`` (subprocess.run-like
        # but tears down the whole descendant tree on every exit path).
        # Plain subprocess.run leaks daemonized server processes. See
        # ``_subprocess_kill.py``.
        subprocess_started_unix = time.time()
        # Anchor the Magpie *parent* process cwd to the stable per-task
        # output_dir instead of the default ``/tmp`` (defence-in-depth for any
        # relative-path writes Magpie itself makes). NOTE: this does NOT keep
        # the server's cuda-graph dump safe on its own — Magpie re-roots the
        # actual server via ``cd <inferencex>`` (see ``_build_local_command``
        # in Magpie), so the server's cwd (where SGLang dumps the cuda-graph
        # pickle) is the InferenceX checkout, not this output_dir.
        # ``_ensure_local_inferencex`` above keeps that checkout on stable
        # local disk.
        output_dir.mkdir(parents=True, exist_ok=True)
        # A reused output_dir (explicit ``params['output_dir']``
        # on a retry, or a re-run of the same task slot) may still hold a
        # PRIOR attempt's server.log. Its terminal engine/worker-init markers
        # would otherwise misclassify THIS attempt as ``server_init_dead`` even
        # when the current server booted fine — the post-run scan
        # (``server_log_death_excerpt`` below) can't tell a stale marker from a
        # fresh one. Clear it so classification only sees this attempt's log.
        stale_server_log = output_dir / "server.log"
        try:
            stale_server_log.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning(
                "baseline_executor: could not clear stale server.log %s (%s); "
                "a prior attempt's markers may bias failure classification.",
                stale_server_log,
                exc,
            )
        try:
            proc = await asyncio.to_thread(
                run_with_session_kill,
                cmd,
                env=env,
                cwd=str(output_dir),
                timeout=timeout_sec,
                server_log_path=str(output_dir / "server.log"),
            )
            subprocess_runtime_sec = max(
                0.0,
                time.time() - subprocess_started_unix,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_candidates = sorted(output_dir.glob("benchmark_*"))
            timeout_destination = timeout_candidates[-1] if timeout_candidates else output_dir
            timeout_harvested = harvest_leaked_artifacts(
                timeout_destination,
                subprocess_started_unix=subprocess_started_unix,
            )
            return {
                "status": "failed",
                "error_class": "timeout",
                "error": f"baseline benchmark exceeded {timeout_sec}s: {exc}",
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in timeout_harvested],
                "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in timeout_harvested],
            }
        proc_returncode = proc.returncode
        proc_stdout = proc.stdout
        proc_stderr = proc.stderr

        # Detokenizer-stall watchdog reap: the server came up healthy (ready
        # marker logged) but then went completely silent for the stall grace
        # window — a hung engine / wedged detokenizer. The clock runs from the
        # ready marker, so a long cold start never trips it. Short-circuit here:
        # a stall reap leaves no benchmark_* workspace, so without this it would
        # misclassify as ``no_workspace``. Distinct ``error_class`` lets the
        # coordinator fast-fail this baseline / variant instead of burning the
        # full hard timeout. Harvest whatever the wrapper wrote first.
        if proc_returncode == DETOKENIZER_STALL_RETURNCODE:
            stall_candidates = sorted(output_dir.glob("benchmark_*"))
            stall_destination = stall_candidates[-1] if stall_candidates else output_dir
            stall_harvested = harvest_leaked_artifacts(
                stall_destination,
                subprocess_started_unix=subprocess_started_unix,
            )
            log.warning(
                "baseline_executor: detokenizer-stall watchdog reaped run "
                "(server ready but log went silent); error_class=detokenizer_stall."
            )
            return {
                "status": "failed",
                "error_class": "detokenizer_stall",
                "returncode": proc_returncode,
                "error": (
                    "server reported ready but emitted no log output (hung "
                    "engine / detokenizer stall); reaped by the "
                    "detokenizer-stall watchdog. See server.log."
                ),
                "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in stall_harvested],
                "nonfatal_warnings": [f"harvested_leaked_artifact:{src}" for src, _ in stall_harvested],
            }

        # When the inference server's engine/worker bootstrap dies (e.g.
        # vLLM ``RuntimeError: Engine core initialization failed``), the real
        # root cause is in server.log, not in Magpie's stdout/stderr tail — and
        # the liveness watchdog may have reaped the hung parent with
        # ``SERVER_DEAD_RETURNCODE``. Detect that once here and reuse it across
        # the failure branches below so the failure is classified
        # ``server_init_dead`` and the operator sees the actual server fault
        # instead of a generic ``subprocess_nonzero``. Backend-agnostic: the
        # markers cover both vLLM and SGLang engine/worker init failures.
        server_death_excerpt = server_log_death_excerpt(str(output_dir / "server.log"))
        server_init_dead = server_death_excerpt is not None or proc_returncode == SERVER_DEAD_RETURNCODE
        server_init_dead_error = server_death_excerpt or (
            "server engine/worker init failed (reaped by liveness watchdog); see server.log"
        )

        # Detect cuda-graph capture failures (recoverable by disabling
        # cuda-graph capture; OOM-rooted ones are excluded).
        # Markers live in server.log; read a bounded tail for classification.
        server_log_tail = ""
        try:
            slog = output_dir / "server.log"
            if slog.exists():
                with open(slog, "rb") as f:
                    f.seek(0, 2)
                    sz = f.tell()
                    f.seek(max(0, sz - 65536))
                    server_log_tail = f.read().decode("utf-8", "replace")
        except OSError:
            server_log_tail = ""
        cuda_graph_capture_failed = _is_cuda_graph_capture_failure(
            server_log_tail,
            proc_stderr or "",
            proc_stdout or "",
        )

        # Locate the workspace Magpie created (benchmark_<framework>_<ts>/).
        candidates = sorted(output_dir.glob("benchmark_*"))
        # Always-on artifact harvest: copy wrapper-side leaks hardcoded
        # under ``/workspace/`` into the task workspace (see
        # ``harvest_leaked_artifacts``). Runs unconditionally so failure-path
        # diagnostics survive; mtime gating rejects stale prior-run leaks.
        harvest_destination = candidates[-1] if candidates else output_dir
        harvested = harvest_leaked_artifacts(
            harvest_destination,
            subprocess_started_unix=subprocess_started_unix,
        )
        if harvested:
            log.info(
                "baseline_executor: harvested %d leaked artifact(s) into workspace: %s",
                len(harvested),
                ", ".join(str(src.name) for src, _ in harvested),
            )
        if not candidates:
            failure_extras = {
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in harvested],
            }
            # Magpie never created a benchmark_* workspace, so the wrapper
            # never wrote server.log. Persist the captured stderr/stdout to
            # a file so the failure survives the NFS clone and S3 archive
            # (without this, no_workspace failures leave zero on-disk logs).
            captured = (proc_stderr or "") + (proc_stdout or "")
            stderr_log_path: str | None = None
            if captured.strip():
                try:
                    log_file = output_dir / "baseline_stderr.log"
                    log_file.write_text(captured, encoding="utf-8")
                    stderr_log_path = str(log_file)
                except OSError as exc:
                    log.warning(
                        "baseline_executor: failed to persist stderr log: %s",
                        exc,
                    )
            if stderr_log_path:
                failure_extras["stderr_log_path"] = stderr_log_path
            # cuda-graph capture failures take priority over server_init_dead:
            # the marker may co-occur with a server-death marker, but only this
            # class arms the one-shot disable-cuda-graph retry.
            if cuda_graph_capture_failed:
                return {
                    "status": "failed",
                    "error_class": "cuda_graph_capture_failed",
                    "returncode": proc_returncode,
                    "error": server_init_dead_error if server_init_dead else (proc_stderr or proc_stdout or "")[-2000:],
                    **failure_extras,
                }
            if server_init_dead:
                return {
                    "status": "failed",
                    "error_class": "server_init_dead",
                    "returncode": proc_returncode,
                    "error": server_init_dead_error,
                    **failure_extras,
                }
            if proc_returncode != 0:
                tail = (proc_stderr or proc_stdout or "")[-2000:]
                err_class = _classify_subprocess_error(
                    subprocess_runtime_sec,
                    tail,
                )
                return {
                    "status": "failed",
                    "error_class": err_class,
                    "returncode": proc_returncode,
                    "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                    "error": tail,
                    **failure_extras,
                }
            return {
                "status": "failed",
                "error_class": "no_workspace",
                "error": "Magpie completed but produced no benchmark_* workspace",
                **failure_extras,
            }
        workspace = candidates[-1]
        report_path = workspace / "benchmark_report.json"
        report: dict[str, Any] | None = None
        if report_path.exists():
            try:
                with report_path.open(encoding="utf-8") as f:
                    loaded = json.load(f)
                report = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                report = None

        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=subprocess_started_unix,
        )
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        if proc_returncode != 0:
            warnings.append("magpie_nonzero_after_valid_measurement")
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")

        if not measurement.get("valid_measurement"):
            # cuda-graph capture failure wins over server_init_dead so the
            # one-shot disable-cuda-graph retry is armed even when both markers
            # co-occur (see the no_workspace branch above).
            if cuda_graph_capture_failed:
                error_class = "cuda_graph_capture_failed"
                error = server_init_dead_error if server_init_dead else ((proc_stderr or proc_stdout or "")[-2000:])
            elif server_init_dead:
                error_class = "server_init_dead"
                error = server_init_dead_error
            elif proc_returncode != 0:
                tail = (proc_stderr or proc_stdout or "")[-2000:]
                error_class = _classify_subprocess_error(
                    subprocess_runtime_sec,
                    tail,
                )
                error = tail
            elif not report_path.exists():
                error_class = "no_report"
                error = f"benchmark_report.json missing under {workspace}"
            else:
                error_class = "invalid_measurement"
                error = "benchmark report did not contain positive throughput and completed requests"
            return {
                "status": "failed",
                "error_class": error_class,
                "returncode": proc_returncode,
                "error": error,
                "output_dir": str(output_dir),
                "workspace": str(workspace),
                "report_path": str(report_path) if report_path.exists() else None,
                "reported_success": measurement.get("reported_success"),
                "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
                "nonfatal_warnings": warnings,
            }

        result = {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": proc_returncode,
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(workspace),
            # Materialized YAML for THIS baseline. Coordinator promotes it
            # into SharedState.baseline_config_path so downstream tasks reuse
            # it as `config_path` (else variants render from the YAML's smoke
            # defaults and produce ~10x lower throughput). See `_workload_envs.py`.
            "materialized_config": str(materialized_config_path),
            # Magpie subprocess wall-clock (success path only). Coordinator
            # promotes into ``SharedState.baseline_runtime_sec``, the explore
            # overtime-kill anchor. Omitted on failure paths so a botched
            # baseline can't seed a bad deadline.
            "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
        }

        # Parse accuracy eval results (GSM8K for serving, or the image-quality
        # gate for scriptable frameworks); RUN_EVAL=true ran lm-eval while the
        # server was up. Pass the framework so scriptable runs (xDiT) fail
        # closed on a missing quality gate instead of falling back to GSM8K.
        from ._accuracy_gate import parse_eval_results

        eval_framework = (report or {}).get("framework") or os.environ.get("FRAMEWORK") or None
        eval_data = parse_eval_results(workspace, framework=eval_framework)
        if eval_data.get("accuracy") is not None:
            result["accuracy"] = eval_data["accuracy"]
            result["accuracy_task"] = eval_data.get("task", "gsm8k")
            result["accuracy_metric"] = eval_data.get("metric", "")
            result["accuracy_source"] = eval_data.get("source_file", "")
            log.info("baseline_executor: accuracy=%.4f (%s)", result["accuracy"], result["accuracy_task"])
        else:
            log.warning("baseline_executor: accuracy eval not found: %s", eval_data.get("error", "unknown"))

        from hyperloom.inference_optimizer import framework_registry

        log.info(
            "baseline_executor: %s %s (output) e2el=%.1fms",
            "success_with_warning" if warnings else "success",
            framework_registry.format_primary_metric(eval_framework, result["output_throughput"]),
            result["e2el_mean_ms"] or 0.0,
        )
        return result


# Module-level callable for ``register_executor("baseline", baseline_executor)``.
baseline_executor = BaselineExecutor()


__all__ = [
    "AITER_JIT_PROBE_PATHS",
    "BASELINE_COLD_START_TIMEOUT_SEC",
    "BASELINE_DEFAULT_TIMEOUT_SEC",
    "BaselineExecutor",
    "COLD_START_KERNEL_THRESHOLD",
    "baseline_executor",
]
