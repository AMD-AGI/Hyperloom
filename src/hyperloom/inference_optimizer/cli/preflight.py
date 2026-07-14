# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI ``_preflight`` cluster — auto-install/env-hygiene checks run before ``optimize`` starts.

Extracted from ``cli/__init__.py`` (tree-reform.MD P2.4 follow-up). Four
functions in this cluster (``_load_dotenv_fallback``,
``_load_kernel_agent_env_fallback``, ``_clone_inferencex``, and — still in
``__init__.py`` — ``_probe_llm_catalog``) are directly monkeypatched by name
in ``test_preflight_auth_override.py`` via
``monkeypatch.setattr(cli, "<name>", ...)``. Per tree-reform-lessons.MD §3.2/
§3.3, a bare-name call from inside this module would resolve THIS module's
own binding and silently bypass such a patch — even for a function defined in
this same file (verified empirically; see the inline comments at each call
site below). Every call to one of those four names is therefore a lazy,
package-qualified ``from . import <name>`` read at call time instead of a
bare-name reference, so the patched value on the ``cli`` package is always
picked up (same technique used for coordinator.py's cross-collaborator calls).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .credentials import (  # noqa: F401 - re-exported for callers/tests
    _is_stale_proxy_url,
    _resolve_llm_endpoints,
    _reset_claude_config_to_upstream,
    _sync_geak_config_base_url,
    _validate_credentials,
)
from ..session.paths import (
    DEFAULT_SESSION_DIR,
    ENV_USER_DATA_PATH,
    session_dir as _session_dir_resolve,
    workspace_root as _workspace_root_resolve,
)

log = logging.getLogger("hyperloom.inference_optimizer.cli")

_PROVIDER_FALLBACK_KEYS: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "SAFE_API_KEY",
    "LLM_GATEWAY_KEY",
    "DEEPSEEK_BASE_URL",
    "GEAK_BASE_URL",
    "LLM_API_BASE",
)


def _provider_only_mode_before_fallback() -> str:
    """Detect explicit single-provider intent before installer env fallback runs."""
    has_anthropic = bool(
        os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_BASE_URL")
    )
    has_openai = bool(os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_KEY"))
    has_gateway = bool(os.environ.get("SAFE_API_KEY") or os.environ.get("LLM_GATEWAY_KEY"))
    if has_anthropic and not has_openai and not has_gateway:
        return "anthropic"
    if has_openai and not has_anthropic and not has_gateway:
        return "openai"
    return ""


def _restore_provider_only_mode(provider_mode: str, snapshot: dict[str, str | None]) -> None:
    """Undo stale cross-provider credentials loaded from installer env fallback."""
    if provider_mode != "anthropic":
        return
    for key in _PROVIDER_FALLBACK_KEYS:
        original = snapshot.get(key)
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original

# /dev/shm threshold: below this, next launch collides with stale vLLM/NCCL shm segments and hangs in zmq.
_DEV_SHM_MIN_FREE_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB


def _is_placeholder_tracelens_path(value: str) -> bool:
    """Treat unedited .env.template placeholders as unset.

    Covers the bare ``\\`` / whitespace-only values plus common literal
    placeholders (``/path/to/your/TraceLens``, ``<your-...>``) an operator
    forgot to replace, so the pod-local fallback / installer value wins.

    Args:
        value (str): The candidate TraceLens path value.

    Returns:
        bool: ``True`` when the value is blank or an unedited placeholder.
    """
    stripped = value.strip()
    if stripped in ("", "\\"):
        return True
    low = stripped.lower()
    if "/path/to/" in low or "path/to/your" in low:
        return True
    if "<" in stripped and ">" in stripped:
        return True
    return False


def _load_dotenv_fallback() -> None:
    """Source missing vars from ``$REPO_ROOT/.env``; env always wins (no-clobber).

    Always parses ``.env`` and loads any key that is not already present in the
    environment, regardless of whether LLM credentials are already set. Removing
    the former URL+KEY early-return fixes silent misconfiguration: exporting only
    ``OPENAI_BASE_URL`` and ``SAFE_API_KEY`` previously skipped unrelated
    operational vars (e.g. ``TRACELENS_ROOT``, ``FORGE_PATH``,
    ``KERNEL_OPT_BACKEND_ORDER``) that are also stored in ``.env``.
    """
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    if not env_file.exists():
        return
    loaded = 0
    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key in ("TRACELENS_ROOT", "TRACELENS_INTERNAL_ROOT") and _is_placeholder_tracelens_path(value):
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        print(f"Preflight: loaded {loaded} missing var(s) from {env_file} (env wins)")


_KERNEL_AGENT_PATH_VARS: tuple[str, ...] = ("TRACELENS_ROOT",)


def _parse_env_assignments(text: str) -> dict[str, str]:
    """Parse ``[export] KEY=VALUE`` shell assignments into a dict (first wins)."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out.setdefault(key, value)
    return out


def _correct_kernel_agent_path_vars(file_vars: dict[str, str], env_path: Path) -> None:
    """Overwrite invalid inherited path-class vars with the env file's value.

    Only fires when the inherited value is unset/non-existent AND the file value
    points at an existing dir; a valid inherited value keeps env-wins semantics.
    """
    for key in _KERNEL_AGENT_PATH_VARS:
        file_val = file_vars.get(key)
        if not file_val or not Path(file_val).is_dir():
            continue
        current = os.environ.get(key, "")
        if current == file_val or (current and Path(current).is_dir()):
            continue
        print(
            f"Preflight: WARNING — {key}={current or '(unset)'} does not point "
            f"at an existing checkout; correcting to {file_val} from {env_path} "
            f"(installer-written value wins for path vars).",
            file=sys.stderr,
        )
        os.environ[key] = file_val


def _load_kernel_agent_env_fallback() -> None:
    """Auto-source the installer-written kernel-agent env file
    (``$KERNEL_AGENT_ENV`` or ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``).

    Must source before any orchestrator import (trace_analyze reads
    HYPERLOOM_KERNEL_AGENT_ROOT at module load). When HYPERLOOM_KERNEL_AGENT_ROOT
    is already set, bootstrapping is skipped but the env file is still consulted
    to CORRECT a stale/invalid inherited TRACELENS_ROOT — the fix for issue
    #722 where a bad inherited TRACELENS_ROOT survived. Hard-fail contract
    (root unset only): sys.exit(2) if missing/0-vars/still-unset.
    """
    candidate = os.environ.get("KERNEL_AGENT_ENV")
    if not candidate:
        user_data = os.environ.get("USER_DATA_PATH")
        if user_data:
            candidate = str(Path(user_data) / "runtime" / "kernel-agent.env.sh")

    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        # Root is set: no bootstrap needed, but still correct invalid path vars
        # from the env file when resolvable. Silent no-op otherwise.
        if not candidate:
            return
        env_path = Path(candidate)
        if not env_path.is_file():
            return
        try:
            text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        _correct_kernel_agent_path_vars(_parse_env_assignments(text), env_path)
        return

    if not candidate:
        print(
            "Preflight: ERROR — neither $HYPERLOOM_KERNEL_AGENT_ROOT "
            "nor $KERNEL_AGENT_ENV nor $USER_DATA_PATH is set. Cannot "
            "resolve kernel-agent.env.sh. Run "
            "src/hyperloom/inference_optimizer/assets/install.sh and export "
            "USER_DATA_PATH=/path/to/sessions first.",
            file=sys.stderr,
        )
        sys.exit(2)
    env_path = Path(candidate)
    if not env_path.is_file():
        print(
            f"Preflight: ERROR — kernel-agent env file not found at "
            f"{env_path}. USER_DATA_PATH must be the workspace root "
            f"(parent of <model>/<ts>/ per-session subdirs); runtime/ "
            f"is workspace-shared, not per-session. Either "
            f"(a) re-run src/hyperloom/inference_optimizer/assets/install.sh under "
            f"USER_DATA_PATH={os.environ.get('USER_DATA_PATH', '?')}, "
            f"(b) set $KERNEL_AGENT_ENV to point at an existing file, or "
            f"(c) set $HYPERLOOM_KERNEL_AGENT_ROOT directly to skip this "
            f"fallback entirely. Aborting now (was: silently warning and "
            f"letting trace_analyze fail 10h in).",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(
            f"Preflight: ERROR — failed to read {env_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    file_vars = _parse_env_assignments(text)
    loaded = 0
    for key, value in file_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    _correct_kernel_agent_path_vars(file_vars, env_path)
    if "HYPERLOOM_KERNEL_AGENT_ROOT" not in os.environ:
        print(
            f"Preflight: ERROR — sourced {env_path} ({loaded} vars) but "
            f"HYPERLOOM_KERNEL_AGENT_ROOT is still unset. The env file is "
            f"malformed or stale. Re-run src/hyperloom/inference_optimizer/assets/"
            f"install.sh to regenerate it.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"Preflight: loaded {loaded} kernel-agent var(s) from "
        f"{env_path} (env wins, HYPERLOOM_KERNEL_AGENT_ROOT="
        f"{os.environ['HYPERLOOM_KERNEL_AGENT_ROOT']})"
    )


def _ensure_python_sdks(python_exe: str, pip_extra: list[str]) -> None:
    """Probe-then-install runtime-imported Python SDKs using the same interpreter that imports them.

    Avoids first-tick BackendError after baseline burns wall time; same-interpreter install avoids
    cross-interpreter install failures.

    Args:
        python_exe (str): The interpreter that will import the SDKs (and run
            the probe / install).
        pip_extra (list[str]): Extra arguments threaded into the ``pip
            install`` invocation (e.g. index flags).
    """
    candidates = (
        ("claude_agent_sdk", "claude-agent-sdk>=0.1.65"),
        ("openai", "openai>=1.50"),
        ("httpx", "httpx>=0.27"),
    )
    for module_name, pip_spec in candidates:
        check = subprocess.run(
            [python_exe, "-c", f"import {module_name}"],
            capture_output=True,
        )
        if check.returncode == 0:
            print(f"Preflight: {module_name} OK")
            continue
        print(f"Preflight: {module_name} not importable, installing {pip_spec} ...")
        subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet", *pip_extra, pip_spec],
            check=True,
        )
        print(f"Preflight: installed {pip_spec}")


def _unset_hip_visible_devices() -> None:
    """Drop ``HIP_VISIBLE_DEVICES`` if ``ROCR_VISIBLE_DEVICES`` is set (SKILL.md §"GPU Runner Type").

    ROCm gotcha: both set can make torch.cuda.is_available() false in Magpie; ROCR_VISIBLE_DEVICES is canonical.
    """
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        return
    if "ROCR_VISIBLE_DEVICES" not in os.environ:
        return
    value = os.environ.pop("HIP_VISIBLE_DEVICES")
    print(
        f"Preflight: WARNING — unset HIP_VISIBLE_DEVICES={value!r} "
        f"(ROCR_VISIBLE_DEVICES wins on ROCm; HIP_VISIBLE_DEVICES can "
        f"make torch.cuda.is_available() false inside Magpie subprocess)"
    )


def _check_gpu_visibility() -> None:
    """Best-effort informational check of visible GPU count vs ``$TP`` (silent when rocm-smi is absent)."""
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return
    if proc.returncode != 0:
        return
    # rocm-smi --showid emits multiple GPU[ lines per GPU (~6x overcount); deduplicate by GPU index.
    visible_indices: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU["):
            idx, _, _ = stripped[4:].partition("]")
            if idx:
                visible_indices.add(idx)
    visible = len(visible_indices)
    try:
        wanted = int(os.environ.get("TP", "1") or "1")
    except ValueError:
        wanted = 1
    if visible == 0:
        print("Preflight: WARNING — rocm-smi sees 0 GPUs; benchmark will fail")
        return
    if wanted > visible:
        print(
            f"Preflight: WARNING — TP={wanted} but rocm-smi sees {visible} "
            f"GPU(s); sglang/vllm may fail to load weights. Lower TP or "
            f"adjust ROCR_VISIBLE_DEVICES."
        )


def _check_shm_disk() -> None:
    """Warn (not fail-fast) on tight ``/dev/shm`` (vLLM/NCCL IPC needs headroom)."""
    try:
        usage = shutil.disk_usage("/dev/shm")
    except (FileNotFoundError, OSError):
        return
    if usage.free < _DEV_SHM_MIN_FREE_BYTES:
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        print(
            f"Preflight: WARNING — /dev/shm has {free_gb:.1f} GiB free of "
            f"{total_gb:.1f} GiB total (< 16 GiB threshold). vLLM IPC + "
            f"NCCL shm segments may collide with stale entries; if the "
            f"first server launch hangs >5min, clear /dev/shm/{{vllm,nccl,cuda}}*"
        )


_TRACELENS_REQUIRED_CLIS: tuple[str, ...] = ("TraceLens_generate_perf_report_pytorch_inference",)


def _tracelens_required_at_preflight(no_kernel: bool, enable_roofline: bool) -> bool:
    """Return whether the TraceLens CLI must be present at preflight (hard-fail).

    TraceLens is reached both by the Kernel-agent AND by the PRELUDE/auto
    roofline (roofline -> trace_analyze -> TraceLens). ``enable_roofline``
    defaults True and is NOT disabled by ``--no-kernel``, so under ``--no-kernel``
    alone the PRELUDE roofline still invokes TraceLens. It is only truly unused
    when the kernel_agent role is off (``--no-kernel``) AND roofline is disabled; only
    then may preflight degrade to WARN. Otherwise keep the hard-fail so a missing
    CLI fails fast at preflight instead of mid-run at the first roofline.

    Args:
        no_kernel: Whether the run is started with ``--no-kernel``.
        enable_roofline: Whether the auto/PRELUDE roofline is enabled.

    Returns:
        bool: ``True`` when TraceLens must hard-gate at preflight.
    """
    return not (no_kernel and not enable_roofline)


def _check_tracelens_cli() -> None:
    """Hard-gate TraceLens CLI presence — abort before Coordinator starts (SKILL IR-2).

    Pod-local /opt/venv/bin/TraceLens_* console_scripts don't persist across pod restarts, so install.sh
    must run before every launch (carve-out: --resume in the same shell). Fail-fast beats a delayed
    tracelens_cli_missing strike at tick ~6 after baseline burned setup time.
    """
    missing = [name for name in _TRACELENS_REQUIRED_CLIS if shutil.which(name) is None]
    if not missing:
        return
    session_dir = str(_workspace_root_resolve())
    print(
        f"ERROR: TraceLens CLI(s) not on PATH: {missing}. The pod-local "
        f"/opt/venv/bin/TraceLens_* console_scripts are installed by "
        f"src/hyperloom/agents/kernel/scripts/install.sh (chained from "
        f"src/hyperloom/inference_optimizer/assets/install.sh) and do NOT persist "
        f"across pod restarts. SKILL IR-2 requires running install.sh "
        f"before every launch (carve-out applies only to --resume in "
        f"the same shell that earlier ran install.sh). Re-run:\n"
        f"  bash $REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh\n"
        f"  . {session_dir}/runtime/kernel-agent.env.sh\n"
        f"then retry `inference_optimizer optimize`. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_tracelens_root_exists() -> None:
    """Hard-gate an explicitly set ``TRACELENS_ROOT`` at preflight (issue #722).

    An operator-supplied TRACELENS_ROOT that points at a missing checkout (stale
    path or unedited template placeholder) otherwise only surfaces ~10h later in
    trace_analyze. Unset is fine (pod-local fallback handled downstream).
    """
    override = os.environ.get("TRACELENS_ROOT")
    if not override or Path(override).is_dir():
        return
    print(
        f"ERROR: TRACELENS_ROOT={override} does not point at an existing "
        f"TraceLens checkout. It was likely inherited from a stale shell or an "
        f"unedited .env template. Re-run src/hyperloom/inference_optimizer/assets/install.sh "
        f"and source $KERNEL_AGENT_ENV, point TRACELENS_ROOT at a real checkout, "
        f"or unset it to use the pod-local default. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_node_claude_cli() -> None:
    """WARN-only presence check for bundled agent CLIs (node/claude/codex).

    SDKs fall back to direct HTTP when CLIs are absent, so this is informational.
    """
    missing = [t for t in ("node", "claude", "codex") if shutil.which(t) is None]
    if missing:
        print(
            f"Preflight: WARNING — CLI(s) not on PATH: {missing}. "
            f"ClaudeBackend / CodexBackend may fall back to direct HTTP. "
            f"Run src/hyperloom/agents/kernel/scripts/install.sh to bring them in."
        )


def _emit_preflight_diagnostics(
    *,
    magpie_python: str,
    anthropic_base_url: str | None,
    args: argparse.Namespace | None = None,
) -> None:
    """One canonical, grep-friendly diagnostics block at the end of preflight.

    Args:
        magpie_python (str): The Magpie interpreter path to report.
        anthropic_base_url (str | None): The resolved Anthropic base URL, or
            ``None`` when unset.
        args (argparse.Namespace | None): Parsed CLI args; when present, KB /
            PR-monitor status lines are added.
    """
    from hyperloom.orchestrator.actions.executors.baseline import (
        BASELINE_COLD_START_TIMEOUT_SEC,
        BASELINE_DEFAULT_TIMEOUT_SEC,
        _probe_aiter_jit_cache,
    )
    from ..session.paths import asset_root

    probe = _probe_aiter_jit_cache()
    cold_cap = os.environ.get(
        "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
        str(BASELINE_COLD_START_TIMEOUT_SEC),
    )
    if probe["probe_status"] == "found":
        kind = "COLD" if probe["is_cold"] else "WARM"
        cache_line = f"{probe['kernel_count']} .so / {probe['size_mb']} MB ({kind}) at {probe['path']}"
    else:
        cache_line = f"<probe_status={probe['probe_status']}>"

    print("Preflight diagnostics:")
    print(f"  asset_root          = {asset_root()}")
    print(
        f"  session_dir         = {_session_dir_resolve()}  "
        f"({ENV_USER_DATA_PATH}="
        f"{os.environ.get(ENV_USER_DATA_PATH, '<unset>')}, "
        f"default={DEFAULT_SESSION_DIR})"
    )
    print(f"  magpie_python       = {magpie_python}")
    print(f"  INFERENCEX_PATH     = {os.environ.get('INFERENCEX_PATH', '<unset>')}")
    print(f"  aiter jit cache     = {cache_line}")
    print(f"  cold_start_timeout  = {cold_cap}s")
    print(f"  warm_timeout        = {BASELINE_DEFAULT_TIMEOUT_SEC}s")
    # Surface the hard GPU-reset arming state up front: `recover` (robustness-
    # delegated) may shell out to `rocm-smi --gpureset` on gpu_memory_leaked.
    # It is opt-in and scoped to ROCR_VISIBLE_DEVICES (never implicit --gpu=all).
    _gpureset_on = os.environ.get(
        "HYPERLOOM_RECOVER_ALLOW_GPU_RESET",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    _rocr_scope = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if _gpureset_on and _rocr_scope:
        print(
            f"  recover_gpureset    = ARMED — robustness may auto "
            f"`rocm-smi --gpureset --gpu={_rocr_scope}` on gpu_memory_leaked; "
            f"WARNING: confirm this session exclusively owns those cards"
        )
    elif _gpureset_on:
        print(
            "  recover_gpureset    = ARMED but UNSCOPED (ROCR_VISIBLE_DEVICES "
            "unset) — hard reset will be SKIPPED (refuses implicit --gpu=all)"
        )
    else:
        print(
            "  recover_gpureset    = disabled (opt-in; set "
            "HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1 to enable, scoped to "
            "ROCR_VISIBLE_DEVICES)"
        )
    if anthropic_base_url:
        print(f"  ANTHROPIC_BASE_URL  = {anthropic_base_url}")
    else:
        print("  ANTHROPIC_BASE_URL  = <unset> — no LLM base URL resolved; Claude SDK will fail")
    if args is not None:
        kb_enabled = bool(getattr(args, "cortex_enabled", True))
        pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
        kb_reason = getattr(args, "kb_degraded_reason", None) or "-"
        pr_reason = getattr(args, "pr_degraded_reason", None) or "-"
        kb_status = "OK" if kb_enabled else f"DEGRADED ({kb_reason})"
        pr_status = "OK" if pr_enabled else f"DEGRADED ({pr_reason})"
        print(f"  kb_status           = {kb_status}")
        print(f"  pr_monitor_status   = {pr_status}")
        print(f"  kb_degraded_reason  = {kb_reason}")
        print(f"  pr_degraded_reason  = {pr_reason}")

    # Surface Cortex KB offline-queue state; dead-letter pile-up signals a cold-start session.
    try:
        _print_cortex_kb_queue_status()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  cortex_kb_queue     = <probe_failed: {exc!r}>")


def _print_cortex_kb_queue_status() -> None:
    """Emit a one-line summary of the Cortex KB offline NDJSON queue (dead-letter = permanent-reject signal).

    Note:
        Side-effecting: writes the queue status summary to stdout and returns
        nothing.
    """
    from ..session.session_paths import (
        cortex_dead_letter_ndjson,
        cortex_flushed_ndjson,
        cortex_pending_ndjson,
    )

    sd = _session_dir_resolve()
    pending = cortex_pending_ndjson(sd)
    dead = cortex_dead_letter_ndjson(sd)
    flushed = cortex_flushed_ndjson(sd)

    def _count(p: Path) -> int:
        """Count non-blank lines (NDJSON rows) in a queue file.

        Args:
            p (Path): Path to the NDJSON file to count.

        Returns:
            int: The number of non-empty lines, or 0 when the file is missing
            or unreadable.
        """
        if not p.exists():
            return 0
        try:
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    p_n, d_n, f_n = _count(pending), _count(dead), _count(flushed)
    print(f"  cortex_kb_queue     = pending={p_n} dead_letter={d_n} flushed={f_n} (root={pending.parent})")
    if d_n > 0:
        print(
            f"                        ⚠ {d_n} dead-letter row(s) — "
            f"prior KB writes permanently rejected (4xx schema). "
            f"Specialists for affected anchors will start cold "
            f"(no priors). See {dead}."
        )


_INFERENCEX_REPO_DEFAULT = "https://github.com/SemiAnalysisAI/InferenceX.git"
_INFERENCEX_REF_DEFAULT = "2035a2117ad22403376359be0064dfa2c078c59b"


def _inferencex_checkout_ok(path: Path | str) -> bool:
    """True when ``path`` is a usable InferenceX checkout, not a stub.

    A bare ``is_dir()`` check accepts a half-cloned dir left behind by a
    ``git init`` that then failed to fetch/checkout. Magpie sources
    ``benchmarks/benchmark_lib.sh`` at runtime, so require that file to
    exist — a complete checkout always has it, a stub never does.

    Args:
        path (Path | str): The candidate InferenceX checkout directory.

    Returns:
        bool: ``True`` when the checkout contains
            ``benchmarks/benchmark_lib.sh``.
    """
    return (Path(path) / "benchmarks" / "benchmark_lib.sh").is_file()


def _clone_inferencex(dest: Path) -> str | None:
    """Clone InferenceX into ``dest`` (writable), pinned to INFERENCEX_REF.

    Mirrors install.sh ``git_fetch_pinned``: a 7-40 hex ref triggers a
    shallow fetch-checkout (GitHub serves SHA fetches), otherwise a
    ``--branch`` clone. Returns the path on success, ``None`` on failure
    (caller decides how to surface it). Never raises out.

    On any failure the partial ``dest`` (e.g. a bare ``git init`` with no
    fetched tree) is removed so a later preflight's detection does not
    mistake the stub for a valid checkout and skip re-cloning.

    Args:
        dest (Path): The writable destination directory for the checkout.

    Returns:
        str | None: The checkout path string on success, or ``None`` on
            failure.
    """
    repo = os.environ.get("INFERENCEX_REPO") or _INFERENCEX_REPO_DEFAULT
    ref = os.environ.get("INFERENCEX_REF") or _INFERENCEX_REF_DEFAULT
    dest_str = str(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", ref):
            subprocess.run(["git", "init", "-q", dest_str], check=True, timeout=60)
            subprocess.run(
                ["git", "-C", dest_str, "fetch", "-q", "--depth", "1", repo, ref],
                check=True,
                timeout=600,
            )
            subprocess.run(
                ["git", "-C", dest_str, "checkout", "-q", "FETCH_HEAD"],
                check=True,
                timeout=120,
            )
        else:
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", "--branch", ref, repo, dest_str],
                check=True,
                timeout=600,
            )
        if not _inferencex_checkout_ok(dest):
            raise OSError(f"clone reported success but {dest_str} is missing benchmarks/benchmark_lib.sh")
        return dest_str
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("InferenceX clone into %s failed: %s", dest_str, exc)
        shutil.rmtree(dest, ignore_errors=True)
        return None


def _preflight(
    args: argparse.Namespace | None = None,
) -> tuple[str, str] | None:
    """Auto-install missing runtime deps and export auth aliases.

    Credentials fallback → auth aliases → SDK install → Anthropic/OpenAI base URL resolve + ~/.claude reset →
    ROCm hygiene → ray/Magpie/InferenceX install → CLI presence checks → diagnostics. Returns
    ``(anthropic_base_url, openai_base_url)`` or ``None`` when no LLM base URL is configured.

    Args:
        args (argparse.Namespace | None): Parsed CLI args, used for the
            diagnostics block; optional.

    Returns:
        tuple[str, str] | None: ``(anthropic_base_url, openai_base_url)``, or
            ``None`` when neither base URL is configured.
    """
    # Lazy, package-qualified lookups (not bare-name calls): tests monkeypatch
    # these by name on the ``cli`` package (``monkeypatch.setattr(cli,
    # "_load_dotenv_fallback", ...)``), and since this function now lives in a
    # sibling module, a bare-name call would resolve this module's own
    # (un-patched) binding and silently bypass the patch. Re-reading the
    # current package attribute at call time picks up the patched version.
    from . import _load_dotenv_fallback as _load_dotenv_fallback_current
    from . import _load_kernel_agent_env_fallback as _load_kernel_agent_env_fallback_current

    provider_mode = _provider_only_mode_before_fallback()
    provider_snapshot = {key: os.environ.get(key) for key in _PROVIDER_FALLBACK_KEYS}
    _load_dotenv_fallback_current()
    _load_kernel_agent_env_fallback_current()
    _restore_provider_only_mode(provider_mode, provider_snapshot)

    # Fail fast on missing credentials after the fallback loaders, before any cycle-burning work.
    _validate_credentials()

    # --- Auth alias export ---
    # SAFE_API_KEY only FILLS gaps now: an operator who set a provider-specific
    # key (OPENAI_API_KEY / ANTHROPIC_API_KEY) for split entrypoints keeps it.
    safe_key = os.environ.get("SAFE_API_KEY", "")
    if safe_key:
        for alias in (
            "OPENAI_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "GEAK_API_KEY",
            "LLM_API_KEY",
            "AMD_LLM_API_KEY",
        ):
            if not os.environ.get(alias):
                os.environ[alias] = safe_key
                print(f"Preflight: filled {alias} from SAFE_API_KEY")
    # --- Resolve install interpreters ---
    from hyperloom.orchestrator.actions.executors._grid_runner import _resolve_magpie_python

    magpie_python = _resolve_magpie_python()

    # Outside a venv, add --break-system-packages so pip installs on bare-metal Debian/Ubuntu.
    pip_extra: list[str] = []
    if not (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        pip_extra = ["--break-system-packages"]

    # --- Python SDK auto-install (claude-agent-sdk / openai / httpx) ---
    # Must precede Coordinator import (ClaudeBackend lazy-imports the SDK); sys.executable matches imports.
    _ensure_python_sdks(sys.executable, pip_extra)

    # --- Resolve Anthropic + OpenAI base URLs (split entrypoints) ---
    # Explicit operator values on each side are preserved; a missing side falls
    # back to the other (legacy single-gateway stays one URL).
    resolved_urls: tuple[str, str] | None = None
    anthropic_url, openai_url = _resolve_llm_endpoints()
    if anthropic_url or openai_url:
        for var, want in (
            ("ANTHROPIC_BASE_URL", anthropic_url),
            ("OPENAI_BASE_URL", openai_url),
        ):
            if not want:
                continue
            prev = os.environ.get(var, "")
            if prev != want:
                os.environ[var] = want
                print(f"Preflight: {var} {prev or '<unset>'} -> {want} (resolved endpoint)")
        # Claude CLI primary key: prefer the explicit Anthropic-side key so a
        # split-entrypoint deploy auths Claude with its own key; SAFE_API_KEY
        # (single-gateway) is the fallback.
        claude_primary_key = (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or safe_key
        )
        _reset_claude_config_to_upstream(claude_primary_key, anthropic_url)
        if anthropic_url and not openai_url and not os.environ.get("GEAK_CLAUDE_MODEL"):
            geak_claude_model = (
                os.environ.get("CLAUDE_MODEL", "").strip()
                or os.environ.get("DEEPSEEK_MODEL", "").strip()
                or ("deepseek-chat" if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "")
                or "claude-opus-4-8"
            )
            os.environ["GEAK_CLAUDE_MODEL"] = geak_claude_model
            print(f"Preflight: GEAK_CLAUDE_MODEL <unset> -> {geak_claude_model} (GEAKv4 Claude workflow)")
        resolved_urls = (anthropic_url, openai_url)

        # GEAK / LLM_API_BASE default to the resolved OpenAI-compatible gateway
        # URL, but an INTENTIONAL operator override is preserved (#521:
        # GEAK runs in a separate network namespace reached via a host-local
        # reverse tunnel).
        gateway_url = openai_url or anthropic_url
        if gateway_url:
            for alias in ("GEAK_BASE_URL", "LLM_API_BASE"):
                current = os.environ.get(alias, "").strip()
                if current and current != gateway_url:
                    # A genuine operator override is preserved, but a leftover
                    # install-time 127.0.0.1:4002 proxy is unreachable and must
                    # be force-rewritten to the gateway.
                    if _is_stale_proxy_url(current):
                        os.environ[alias] = gateway_url
                        print(
                            f"Preflight: {alias} {current} -> {gateway_url} "
                            "(stale install-time proxy; force-rewritten to gateway)"
                        )
                        continue
                    print(f"Preflight: {alias} kept at {current} (operator override; not forced to gateway)")
                    continue
                if os.environ.get(alias) != gateway_url:
                    prev = os.environ.get(alias, "")
                    os.environ[alias] = gateway_url
                    print(f"Preflight: {alias} {prev or '<unset>'} -> {gateway_url} (direct to gateway)")

        # #521: GEAK reads its endpoint from $GEAK_CONFIG (written at install
        # time), not from $GEAK_BASE_URL at runtime. Sync the yaml so the
        # resolved GEAK_BASE_URL above actually reaches the kernel agent.
        geak_cfg = os.environ.get("GEAK_CONFIG", "").strip()
        geak_url = os.environ.get("GEAK_BASE_URL", "").strip()
        if geak_cfg and geak_url and _sync_geak_config_base_url(geak_cfg, geak_url):
            print(f"Preflight: synced GEAK config base_url -> {geak_url} ({geak_cfg})")
    else:
        print("Preflight: WARNING — no LLM base URL set; Claude/Codex SDKs will fail at first call")

    # --- ROCm env hygiene + GPU/shm sanity (defensive WARN-only) ---
    _unset_hip_visible_devices()
    _check_gpu_visibility()
    _check_shm_disk()

    # --- Runtime dep install ---
    # 1. Ray — needed by Magpie for task scheduling even without kernel-agent.
    if shutil.which("ray") is None:
        print("Preflight: ray not found, installing ray[default]==2.44.1 + click<8.3.0 ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet", *pip_extra, "ray[default]==2.44.1", "click<8.3.0"],
            check=True,
        )
        print("Preflight: ray installed OK")

    # 2. Magpie — the benchmark engine all executors shell out to ($MAGPIE_PATH override; auto-clones if missing).
    check = subprocess.run(
        [magpie_python, "-c", "import Magpie"],
        capture_output=True,
    )
    if check.returncode != 0:
        magpie_env = os.environ.get("MAGPIE_PATH")
        magpie_env_explicit = bool(magpie_env)
        if magpie_env:
            magpie_dir = Path(magpie_env)
        else:
            from ..session.paths import magpie_dir as _magpie_default

            magpie_dir = _magpie_default(_session_dir_resolve())
        magpie_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (magpie_dir / "setup.py").exists() and not (magpie_dir / "pyproject.toml").exists():
            # Refuse-to-clobber: don't clone Magpie main over an explicit $MAGPIE_PATH (would destroy local work).
            if magpie_env_explicit:
                print(
                    f"Preflight: ERROR — $MAGPIE_PATH={magpie_dir} has no "
                    f"setup.py/pyproject.toml; refusing to clone Magpie "
                    f"main on top of an operator-supplied path. Fix the "
                    f"env or unset $MAGPIE_PATH to fall back to the "
                    f"session-default location.",
                    file=sys.stderr,
                )
                raise FileNotFoundError(f"$MAGPIE_PATH={magpie_dir} is not a valid Magpie checkout")
            print(f"Preflight: Magpie not importable and not found at {magpie_dir}; cloning ...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/AMD-AGI/Magpie.git", str(magpie_dir)],
                check=True,
            )
        print(f"Preflight: installing Magpie from {magpie_dir} ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet", *pip_extra, "-e", str(magpie_dir)],
            check=True,
        )
        print("Preflight: Magpie installed OK")

    # 3. InferenceX — required for GSM8K accuracy eval; lm-eval deps auto-install at runtime via benchmark_lib.sh.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if not inferencex_path:
        from ..session.paths import (
            magpie_dir as _magpie_default,
            open_source_root as _open_source_default,
        )

        open_source_root = _open_source_default()
        _magpie_env = os.environ.get("MAGPIE_PATH")
        magpie_root = Path(_magpie_env) if _magpie_env else _magpie_default(_session_dir_resolve())
        # InferenceX detection order: Magpie submodule (canonical post-install.sh) → standalone pod-local checkout. Legacy read-only host mounts removed (caused mkstemp [Errno 30]); clone a fresh writable checkout instead.
        for candidate in (
            magpie_root / "InferenceX",
            open_source_root / "InferenceX",
        ):
            if _inferencex_checkout_ok(candidate):
                if os.access(candidate, os.W_OK):
                    inferencex_path = str(candidate)
                    break
                print(
                    "Preflight: skipping non-writable auto-detected "
                    f"InferenceX checkout at {candidate}; cloning a "
                    "writable checkout instead."
                )
    # When no writable checkout was found (e.g. a brain-launched run that
    # skipped install.sh's ensure_inferencex), clone one ourselves rather
    # than falling back to a read-only host mount. baseline cannot run
    # without InferenceX, so a clone failure is a hard error.
    if not (inferencex_path and _inferencex_checkout_ok(inferencex_path)):
        from ..session.paths import open_source_root as _open_source_default

        dest = _open_source_default() / "InferenceX"
        print(f"Preflight: InferenceX not found; cloning into {dest} ...")
        # Lazy package-qualified lookup (see the comment above the
        # _load_dotenv_fallback/_load_kernel_agent_env_fallback calls): tests
        # monkeypatch ``cli._clone_inferencex`` directly, and a bare-name call
        # -- even to a function defined in this same module -- resolves this
        # module's own binding and misses that patch (verified empirically).
        from . import _clone_inferencex as _clone_inferencex_current

        inferencex_path = _clone_inferencex_current(dest)
        if not (inferencex_path and _inferencex_checkout_ok(inferencex_path)):
            print(
                "Preflight: ERROR — InferenceX checkout missing and clone "
                "failed. baseline cannot run without it. Set INFERENCEX_PATH "
                "to a writable checkout or re-run "
                "src/hyperloom/inference_optimizer/assets/install.sh.",
                file=sys.stderr,
            )
            sys.exit(2)
    # Guard against a read-only selection (shared mount handed to us via
    # INFERENCEX_PATH): Magpie stages benchmark scripts there, so a
    # non-writable tree fails the run before server boot.
    if not os.access(inferencex_path, os.W_OK):
        print(
            f"Preflight: ERROR — INFERENCEX_PATH={inferencex_path} is not "
            f"writable. Magpie stages benchmark scripts into it and will "
            f"fail with [Errno 30] Read-only file system. Point "
            f"INFERENCEX_PATH at a writable checkout (unset it to let "
            f"Hyperloom clone a fresh one).",
            file=sys.stderr,
        )
        sys.exit(2)
    # Always overwrite (not setdefault): a stale/broken INFERENCEX_PATH that
    # triggered the clone above must not survive into the child env, or Magpie
    # still reads the bad path. The validated value wins.
    os.environ["INFERENCEX_PATH"] = inferencex_path

    # --- node / claude / codex CLI presence (WARN-only) ---
    _check_node_claude_cli()

    # --- TraceLens CLI presence (HARD-FAIL unless --no-kernel AND roofline off) ---
    # Catches launchers that skip install.sh, else missing-CLI only surfaces at the tick ~6 robustness probe.
    no_kernel = getattr(args, "no_kernel", False) if args else False
    enable_roofline = getattr(args, "enable_roofline", True) if args else True
    if _tracelens_required_at_preflight(no_kernel, enable_roofline):
        _check_tracelens_cli()
        # Fail fast on a stale/placeholder TRACELENS_ROOT before the Coordinator
        # starts, rather than ~10h later in trace_analyze (issue #722).
        _check_tracelens_root_exists()
    else:
        _missing_tl = [n for n in _TRACELENS_REQUIRED_CLIS if shutil.which(n) is None]
        if _missing_tl:
            print(
                f"Preflight: WARNING — TraceLens CLI(s) not on PATH: {_missing_tl} "
                f"(skipped; --no-kernel + roofline disabled)"
            )

    # --- IR-3: Cortex KB + PR Monitor reachability (soft degrade) ---
    if args is not None:
        _run_ir3_preflight(args)

    # --- Single canonical diagnostics block ---
    _emit_preflight_diagnostics(
        magpie_python=magpie_python,
        anthropic_base_url=(resolved_urls[0] if resolved_urls is not None else None),
        args=args,
    )

    return resolved_urls


def _run_ir3_preflight(args: argparse.Namespace) -> None:
    """IR-3 — Cortex KB + PR Monitor reachability probe (soft degrade); never raises/exits.

    Mutates args: ``cortex_enabled``/``pr_monitor_enabled`` plus
    ``kb_degraded_reason``/``pr_degraded_reason`` (None|"explicit_flag"|"ir3_auto").

    Args:
        args (argparse.Namespace): The parsed CLI namespace; mutated in place
            with the resolved KB / PR-monitor enable flags and reasons.
    """
    explicit_kb = bool(getattr(args, "degraded_kb", False))
    explicit_pr = bool(getattr(args, "degraded_pr", False))

    args.cortex_enabled = True
    args.pr_monitor_enabled = True
    args.kb_degraded_reason = None
    args.pr_degraded_reason = None

    if explicit_kb and explicit_pr:
        args.cortex_enabled = False
        args.kb_degraded_reason = "explicit_flag"
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "explicit_flag"
        return

    user_data = _workspace_root_resolve()
    marker_path = user_data / "runtime" / "cortex" / ".kb_preflight.json"
    script = Path(__file__).resolve().parent.parent / "assets" / "preflight_kb.sh"
    env = os.environ.copy()
    # Inject --cortex-kb-url into env so the probe script sees it; empty URL means skip the KB branch.
    cortex_url = (getattr(args, "cortex_kb_url", None) or "").strip()
    if cortex_url:
        env["CORTEX_KB_URL"] = cortex_url
    if explicit_kb:
        env["SKIP_KB_PROBE"] = "1"
    if explicit_pr:
        env["SKIP_PR_PROBE"] = "1"

    try:
        subprocess.run(
            ["bash", str(script)],
            env=env,
            check=False,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Script died — treat both branches as unreachable so soft-degrade kicks in.
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False,
            "pr_reachable": False,
            "kb_skipped": explicit_kb,
            "pr_skipped": explicit_pr,
        }
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("IR-3 marker unreadable: %s", exc)
            marker = {
                "kb_reachable": False,
                "pr_reachable": False,
                "kb_skipped": explicit_kb,
                "pr_skipped": explicit_pr,
            }

    if explicit_kb:
        args.cortex_enabled = False
        args.kb_degraded_reason = "explicit_flag"
    elif not marker.get("kb_reachable", False) and not marker.get("kb_skipped", False):
        args.cortex_enabled = False
        args.kb_degraded_reason = "ir3_auto"

    if explicit_pr:
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "explicit_flag"
    elif not marker.get("pr_reachable", False) and not marker.get("pr_skipped", False):
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "ir3_auto"
