# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cli_executors import (  # noqa: F401 - re-exported for callers/tests
    _NOOP_KINDS_KERNEL_ONLY,
    _REAL_EXECUTORS_FULL,
    _build_specialist_executor,
    _noop_prep,
    _register_executors,
)
from .cli_kb import (  # noqa: F401 - re-exported for callers/tests
    _bootstrap_cortex_kb,
    _bootstrap_knowledge_plane,
    _build_recipe_kb_dispatcher,
    _resolve_local_kb_root,
)
from .cli_backends import (  # noqa: F401 - re-exported for callers/tests
    _MULTI_NODE_WORKLOAD_UID_ENV_KEYS,
    _build_backends,
    _build_proposal_scorer,
    _build_robustness_options,
    _robustness_server_configured,
)
from .orchestrator.backends import ClaudeBackend
from .manifest import load_manifest, write_manifest
from .orchestrator.action_registry import ActionRegistry
from .orchestrator.coordinator import Coordinator
from .orchestrator.proposal_scorer import DEFAULT_SCORER_MODELS
from .orchestrator.framework_paths import resolve_source_file_allowlist
from .orchestrator.objective import Objective, build_objective
from .orchestrator.shared_state import SharedState
from .orchestrator.system_prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from .paths import (
    DEFAULT_SESSION_DIR,
    ENV_USER_DATA_PATH,
    _SESSION_SKELETON,
    asset_system_prompts_dir,
    make_session_dir,
    mn_profile_trace_root,
    session_dir as _session_dir_resolve,
    workspace_root as _workspace_root_resolve,
)
from .session_paths import (
    agent_prompt_snapshot,
)


log = logging.getLogger("inference_optimizer.cli")


class _RetiredFlag(argparse.Action):
    """Argparse action that hard-fails on a retired CLI flag with a migration hint (exits 2)."""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        hint: str,
        **kwargs: Any,
    ) -> None:
        """Register the retired flag as a zero-argument, hidden action.

        Args:
            option_strings (list[str]): The flag spellings this action handles.
            dest (str): The argparse destination name (unused; suppressed).
            hint (str): One-line migration hint shown in the error message when
                the retired flag is used.
            **kwargs (Any): Passed through to :class:`argparse.Action`; ``nargs``,
                ``default``, and ``help`` are defaulted so the flag takes no
                value and stays out of ``--help``.
        """
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("default", argparse.SUPPRESS)
        kwargs.setdefault("help", argparse.SUPPRESS)
        self._hint = hint
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        """Abort parsing with a migration hint when the retired flag is seen.

        Args:
            parser (argparse.ArgumentParser): The parser invoking this action.
            namespace (argparse.Namespace): The in-progress parse namespace.
            values (Any): Parsed values for the flag (always empty; ``nargs=0``).
            option_string (str | None): The exact flag spelling that triggered this.

        Raises:
            SystemExit: Always — ``parser.error`` prints the message and exits 2.
        """
        parser.error(f"{option_string} was removed. {self._hint}")


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only ``orchestration.md`` fragment consumed by ``prompt_builder``."""
    return asset_system_prompts_dir() / "orchestration.md"


def _objective_summary_for_prompt(objective: Objective) -> tuple[str, float | str | None]:
    """Summarise an objective into the ``(kind, value)`` pair the prompt expects.

    Inspects the objective for the first recognised target attribute
    (``target_gain_pct`` → float, ``target_tput_per_gpu`` → float,
    ``baseline_dir`` → str) and pairs it with the objective's ``kind()``.

    Args:
        objective (Objective): The run objective to summarise.

    Returns:
        tuple[str, float | str | None]: ``(kind, value)`` where ``value`` is the
        objective's numeric / string target, or ``None`` when none is present.
    """
    kind = objective.kind()
    value: float | str | None = None
    if hasattr(objective, "target_gain_pct"):
        value = float(getattr(objective, "target_gain_pct"))
    elif hasattr(objective, "target_tput_per_gpu"):
        value = float(getattr(objective, "target_tput_per_gpu"))
    elif hasattr(objective, "baseline_dir"):
        value = str(getattr(objective, "baseline_dir"))
    return kind, value


def _build_orchestration_prompt(
    *,
    no_kernel: bool,
    framework: str,
    objective: Objective,
    max_minutes: int,
    action_registry: ActionRegistry | None = None,
) -> str:
    """Compose the Orchestration system prompt from typed inputs (``--orch-prompt`` overrides)."""
    registry = action_registry or ActionRegistry().load()
    enabled = default_enabled_actions(no_kernel=no_kernel)
    kind, value = _objective_summary_for_prompt(objective)
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework=framework,
        kernel_enabled=not no_kernel,
        objective_kind=kind,
        objective_value=value,
        max_minutes=int(max_minutes),
        rules_fragment_path=_orchestration_rules_fragment_path(),
        framework_source_roots=resolve_source_file_allowlist(),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``system_prompts/critic.md``.

    Returns:
        str: The contents of ``critic.md``.
    """
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


_DEFAULT_KERNEL_PROMPT = (
    "You are the Kernel agent — responder-only. You receive `request`\n"
    "events from Orchestration in your inbox.\n\n"
    "For every un-answered request, emit ONE `response` intent in reply.\n"
    "Schema:\n"
    "  intent_type: response\n"
    "  payload: {\n"
    "    in_reply_to: <request msg_id>,\n"
    "    kind:        '<request.kind>_done',\n"
    "    status:      'ok' | 'failed' | 'needs_review',\n"
    "    result:      { /* whatever the request asked for */ }\n"
    "  }\n\n"
    "Native-only rule: run_optimization must refuse runtime-generated\n"
    "torch.compile/Inductor/Triton cache kernels. Only reusable framework\n"
    "sources under stable repos (aiter/sglang/vllm source trees) are valid\n"
    "kernel-opt targets; otherwise return status='failed' with a clear reason.\n\n"
    "SESSION_DIR contract: every path you emit in result.* must be either\n"
    "verbatim from the request payload, prefixed by SESSION_DIR (injected\n"
    "per tick), or under one of `/sgl-workspace/aiter/`, `/sgl-workspace/\n"
    "sglang/`, `/sgl-workspace/vllm/` (the framework source allowlists).\n"
    "PolicyGate rejects responses whose path fields escape this set.\n\n"
    "If your inbox has no requests, emit one send_message{topic='heartbeat',\n"
    "body_md='ok'}. You may NOT propose, delegate, or initiate REQUESTs."
)


_GFX_TO_RUNNER: dict[str, str] = {
    # Mirror Magpie/modes/benchmark/image_selector.py:138-140 so we can log resolved value at session start.
    "gfx942":  "mi300x",
    "gfx950":  "mi355x",
}


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type.

    MI308X and MI325X share the gfx942 / CDNA3 die with MI300X and reuse
    the same Magpie benchmark scripts (sglang_mi300x.sh / vllm_mi300x.sh).
    """
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi325x", "mi308x"):
        return "mi300x"
    return normalized


# Hard model allowlist (_CLAUDE_ALLOWED_MODELS): orchestration MUST resolve to Opus 4-7 (preferred)
# or 4-6 (fallback) before Coordinator boots; other models drifted behaviour measurably (operator 2026-05-09).
_CLAUDE_PREFERRED_MODEL = "claude-opus-4-7"
_CLAUDE_FALLBACK_MODEL  = "claude-opus-4-6"
_CLAUDE_ALLOWED_MODELS  = (_CLAUDE_PREFERRED_MODEL, _CLAUDE_FALLBACK_MODEL)

# Catalog probe retry contract: gateway is documented-flaky. Sleep N seconds before attempt i+1;
# len(_CATALOG_RETRY_DELAYS_SEC) is the retry count after the initial attempt.
_CATALOG_RETRY_DELAYS_SEC = (1.0, 3.0, 5.0)
_CATALOG_REQUEST_TIMEOUT_SEC = 5.0

# /dev/shm threshold: below this, next launch collides with stale vLLM/NCCL shm segments and hangs in zmq.
_DEV_SHM_MIN_FREE_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB


# Critic-agent skill root resolution. Env wins; else sibling ``critic-agent/`` next to repo root.
_CRITIC_AGENT_ROOT_ENV = "CRITIC_AGENT_ROOT"


def _resolve_critic_agent_root() -> Path | None:
    """Return the critic-agent skill root (``$CRITIC_AGENT_ROOT`` else sibling ``critic-agent/``), or ``None``."""
    override = os.environ.get(_CRITIC_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "runtime" / "cli.py").is_file() else None
    from .paths import PACKAGE_ROOT
    candidate = PACKAGE_ROOT.parent / "critic-agent"
    return candidate if (candidate / "runtime" / "cli.py").is_file() else None


def _validate_critic_agent_runtime(root: Path) -> None:
    """Fail fast (SystemExit) if ``python -m runtime.cli --help`` doesn't work."""
    cmd = [sys.executable, "-m", "runtime.cli", "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: critic-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix CRITIC_AGENT_ROOT, install critic-agent at "
            f"$REPO_ROOT/critic-agent, or pass --critic-mock to bypass "
            f"critic-agent.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: critic-agent runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


# Robustness-agent runtime location resolution; mirrors critic-agent helpers above.
_ROBUSTNESS_AGENT_ROOT_ENV = "ROBUSTNESS_AGENT_ROOT"


def _resolve_robustness_agent_root() -> Path | None:
    """Return robustness-agent skill root (``$ROBUSTNESS_AGENT_ROOT`` else sibling), or ``None``."""
    override = os.environ.get(_ROBUSTNESS_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "src" / "robustness_agent" / "runtime" / "cli.py").is_file() else None
    from .paths import PACKAGE_ROOT
    candidate = PACKAGE_ROOT.parent / "robustness-agent"
    cli_module = candidate / "src" / "robustness_agent" / "runtime" / "cli.py"
    return candidate if cli_module.is_file() else None


def _validate_robustness_agent_runtime(root: Path) -> None:
    """Fail fast if ``python -m robustness_agent.runtime.cli --help`` doesn't work.

    Runs the runtime's ``--help`` with ``cwd=root`` and ``PYTHONPATH`` extended
    by ``<root>/src`` so the subprocess resolves the module the same way the
    real backend will. Any launch failure or non-zero exit prints an
    operator-facing message and aborts.

    Args:
        root (Path): The robustness-agent skill root to validate.

    Raises:
        SystemExit: With code 2 when the runtime cannot start or exits non-zero.
    """
    src = str(root / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    cmd = [sys.executable, "-m", "robustness_agent.runtime.cli", "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: robustness-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix ROBUSTNESS_AGENT_ROOT, install robustness-agent at "
            f"$REPO_ROOT/robustness-agent, or pass --robustness-mock to bypass.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: robustness-agent runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


def _apply_atom_auto_tighten(args: argparse.Namespace) -> list[str]:
    """Validate atom-specific CLI knobs: sole job is the ``--nodes>=2`` fail-fast guard (IR-8).

    No auto-tightening is applied; kernel/framework/profile all work on atom. Multi-node TP wiring
    is unimplemented so ``--nodes>=2`` exits 2. Returns the list of auto-disabled flags (always empty).
    """
    auto_disabled: list[str] = []
    if int(getattr(args, "nodes", 1) or 1) >= 2:
        print(
            "ERROR: --framework atom does not support multi-node "
            "(--nodes >= 2). atom multi-node TP wiring is deferred; "
            "drop to --nodes 1 or pick --framework sglang/vllm.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        "  framework=atom: no auto-disable applied (kernel-agent + "
        "framework-agent + profile / roofline / TraceLens all wired "
        "for atom); --nodes>=2 guard active — see SKILL.md IR-8"
    )
    return auto_disabled


# Forward-looking alias for _apply_atom_auto_tighten (old name kept for tests/SKILL refs; same callable).
_assert_atom_single_node = _apply_atom_auto_tighten


def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve effective gpu_type from a user hint and a hardware probe; pure for unit testing.

    Probe always wins on disagreement (wrong --gpu-type corrupts baseline+KB rows); user value kept
    only on probe failure. Returns ``(effective_gpu_type, warnings)``; warnings go to stderr to keep
    the ``HYPERLOOM_LAUNCH`` stdout sentinel clean.
    """
    warnings: list[str] = []
    if probed and user_specified and probed != user_specified:
        warnings.append(
            f"WARN: --gpu-type={user_specified!r} disagrees with probed "
            f"{probed!r}; using probed {probed!r}. The probe wins because "
            f"Magpie runner_type + KB recipe rows must match the actual "
            f"hardware to keep baseline numbers comparable across sessions."
        )
        return probed, warnings
    return (probed or user_specified), warnings


def _emit_launch_info(
    *,
    pid: int,
    session_dir: Path,
    session_id: str,
    run_log: str,
    gpu_type: str,
    framework: str,
    model: str,
    launch_info_file: str | None,
) -> dict[str, Any]:
    """Print the machine-readable HYPERLOOM_LAUNCH stdout line; optionally JSON-dump to ``launch_info_file``.

    Returns the launch_info dict for callers/tests.
    """
    launch_info: dict[str, Any] = {
        "event": "launch",
        "pid": pid,
        "session_dir": str(session_dir),
        "session_id": session_id,
        "run_log": run_log,
        "manifest": str(session_dir / "manifest.json"),
        "gpu_type": gpu_type,
        "framework": framework,
        "model": model,
    }
    kv_body = " ".join(
        f"{k}={shlex.quote(str(v))}" for k, v in launch_info.items()
    )
    print(f"HYPERLOOM_LAUNCH {kv_body}")
    if launch_info_file:
        path = Path(launch_info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(launch_info, indent=2))
        print(f"Launch info file: {path}")
    return launch_info


def _clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = 5,
) -> dict[str, Any]:
    """Sweep aiter's JIT build dir for stale plain-file locks left by killed runs (else next run hangs).

    Only deletes locks with mtime older than ``stale_minutes`` (default 5; above cold-start MoE build
    time, below the hang-suspicion cliff). Build dir resolution: caller arg → $INFERENCE_OPTIMIZER_AITER_JIT_DIR
    → dynamic <aiter>/jit/build → legacy fallbacks. Returns a stats dict; never raises (errors counted).
    """
    stats: dict[str, Any] = {
        "dir": None,
        "scanned": 0,
        "deleted": 0,
        "skipped_fresh": 0,
        "errors": 0,
    }

    if aiter_jit_dir is None:
        candidates: list[str] = []
        override = os.environ.get(
            "INFERENCE_OPTIMIZER_AITER_JIT_DIR", "",
        ).strip()
        if override:
            override_path = Path(override)
            candidates.extend([str(override_path), str(override_path / "build")])
        try:
            import importlib.util as _il_util
            spec = _il_util.find_spec("aiter")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            aiter_root = Path(spec.origin).parent
            candidates.append(str(aiter_root / "jit" / "build"))
        candidates.extend([
            "/sgl-workspace/aiter/aiter/jit/build",
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/build",
            "/usr/local/lib/python3.12/dist-packages/aiter/jit/build",
            "/opt/venv/lib/python3.10/site-packages/aiter/jit/build",
            "/opt/venv/lib/python3.12/site-packages/aiter/jit/build",
        ])
        chosen: Path | None = None
        for cand in candidates:
            p = Path(cand)
            if p.is_dir():
                chosen = p
                break
        if chosen is None:
            return stats
        aiter_jit_dir = chosen

    stats["dir"] = str(aiter_jit_dir)

    threshold_seconds = float(stale_minutes) * 60.0
    now = time.time()
    lock_names = {"lock", ".ninja_lock"}
    try:
        walker = os.walk(str(aiter_jit_dir))
    except OSError:
        stats["errors"] += 1
        return stats

    for root, _dirs, files in walker:
        for fname in files:
            if not (fname in lock_names or fname.startswith("lock_")):
                continue
            stats["scanned"] += 1
            fpath = Path(root) / fname
            try:
                age = now - fpath.stat().st_mtime
            except OSError:
                stats["errors"] += 1
                continue
            if age < threshold_seconds:
                stats["skipped_fresh"] += 1
                continue
            try:
                fpath.unlink()
                stats["deleted"] += 1
            except OSError:
                stats["errors"] += 1

    return stats


def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi308x|mi325x|mi355x or None if undetectable (rocm-smi then torch gcnArchName, best-effort)."""
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        ).stdout.upper()
        for tag in ("MI355X", "MI325X", "MI308X", "MI300X"):
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        pass
    try:
        import torch
        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except Exception:  # noqa: BLE001
        return None


# AMD/ROCm runner types (gfx9). dual_chunk_flash_attn (sm90+) is unsupported
# here, and some upstream archs (DSA) are not adapted to AMD yet.
_AMD_GPU_TYPES = frozenset({"mi300x", "mi308x", "mi325x", "mi355x"})


def _resolve_amd_gpu_type(explicit: str | None = None) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown.

    Resolution order (most authoritative first): an explicit ``gpu_type``
    argument, the ``GPU_TYPE`` env, then a best-effort runtime autodetect.
    Returning the resolved value only when it names a known AMD runner lets
    callers gate AMD-specific behaviour on real hardware while still honouring
    a launcher/CI-supplied ``gpu_type`` even if ``rocm-smi``/torch probing is
    unavailable at the call site.
    """
    for cand in (explicit, os.environ.get("GPU_TYPE")):
        norm = str(cand or "").strip().lower()
        if norm in _AMD_GPU_TYPES:
            return norm
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in _AMD_GPU_TYPES else None


def _resume_safe_flag(
    args: argparse.Namespace,
    arg_name: str,
    manifest: dict | None,
    manifest_key: str,
    *,
    default: bool,
    invert: bool = False,
) -> bool:
    """Resolve a boolean CLI flag with resume-safe manifest fallback: explicit arg → manifest → default.

    ``invert=True`` handles the ``--no-*`` pattern (args.no_X True == disable; manifest stores positive form).
    Lets robustness_monitor.sh resume preserve original intent without re-passing the flag.
    """
    raw_arg = getattr(args, arg_name, None)
    if isinstance(raw_arg, bool) and raw_arg:
        return (not raw_arg) if invert else raw_arg
    if manifest is not None and manifest_key in manifest:
        stored = manifest.get(manifest_key)
        if isinstance(stored, bool):
            return stored
    return default


def _resume_safe_numeric(
    args: argparse.Namespace,
    arg_name: str,
    manifest: dict | None,
    manifest_key: str,
    *,
    default: float,
) -> float:
    """Float-valued analog of :func:`_resume_safe_flag`: explicit non-default arg → manifest → default."""
    raw_arg = getattr(args, arg_name, None)
    if raw_arg is not None:
        try:
            v = float(raw_arg)
        except (TypeError, ValueError):
            v = None
        if v is not None and v != default:
            return v
    if manifest is not None and manifest_key in manifest:
        try:
            return float(manifest.get(manifest_key) or default)
        except (TypeError, ValueError):
            pass
    return default


def _load_model_arch(workspace_root: Path, model_name: str) -> dict:
    """Best-effort loader for the advisory ``<workspace_root>/model_arch.json`` profile (prompts only).

    Soft-degrades to ``{}`` (never blocks launch) on missing/unreadable/invalid file. Stale-file guard:
    require ``data["model_name"]`` basename to match launched ``--model`` basename, else WARN + ``{}``.
    """
    arch_path = workspace_root / "model_arch.json"
    try:
        raw = arch_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logging.warning("model_arch_unreadable: %s (%s)", arch_path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_arch_invalid_json: %s (%s)", arch_path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning(
            "model_arch_not_a_dict: %s (got %s)", arch_path, type(data).__name__
        )
        return {}
    declared = str(data.get("model_name") or "").strip()
    if not declared:
        logging.warning(
            "model_arch_missing_model_name: %s (cannot verify freshness)", arch_path
        )
        return {}
    if Path(declared).name != Path(model_name).name:
        logging.warning(
            "model_arch_stale_or_mismatch: %s declares model_name=%r but "
            "launching %r — ignoring",
            arch_path,
            declared,
            model_name,
        )
        return {}
    return data


def _load_model_config_dict(model_path: str) -> dict | None:
    """Best-effort parse of ``<model_path>/config.json`` into a dict; returns ``None`` on any failure."""
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logging.warning("model_config_unreadable: %s (%s)", cfg_path, exc)
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_config_invalid_json: %s (%s)", cfg_path, exc)
        return None
    if not isinstance(data, dict):
        logging.warning(
            "model_config_not_a_dict: %s (got %s)", cfg_path, type(data).__name__,
        )
        return None
    return data


def _config_architectures(config: dict) -> list[str]:
    """Normalise ``config["architectures"]`` to a list of non-empty strings (scalar wrapped; absent -> [])."""
    arches_raw = config.get("architectures")
    if isinstance(arches_raw, list):
        return [str(a).strip() for a in arches_raw if str(a or "").strip()]
    if isinstance(arches_raw, str) and arches_raw.strip():
        return [arches_raw.strip()]
    return []


def _load_model_config_tags(model_path: str) -> dict:
    """Best-effort loader for KB architecture-identity tags (``architectures`` + ``model_type``) from config.json.

    Soft-degrades to ``{}`` (never blocks launch); normalised fields are omitted when empty so callers can .get().
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return {}
    out: dict = {}
    arches = _config_architectures(data)
    if arches:
        out["architectures"] = arches
    model_type = str(data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    return out


# Unsupported-model (multimodal / vision) preflight gate
# Hyperloom only supports text-generation (decoder-only causal LM). Multimodal models leak past upstream
# and fail ~5min in with a cryptic image-processor error; these constants drive a fail-fast whitelist classifier.

# Supported text-generation architecture markers (decoder-only causal LM); ForCausalLM is an infix
# because some variants append a suffix (e.g. DeepseekV3ForCausalLMNextN / LlamaForCausalLMEagle3).
_SUPPORTED_ARCH_MARKERS = (
    "ForCausalLM",
    "LMHeadModel",
    "ForCausalLMWithValueHead",
)

# Explicit allowlist of supported model_type values (fallback when architectures is empty/missing).
_SUPPORTED_MODEL_TYPES = frozenset({
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "gemma", "gemma2", "phi", "phi3", "phimoe",
    "starcoder2", "codellama", "deepseek_v2", "deepseek_v3",
    "falcon", "gpt_neox", "gpt2", "opt", "bloom",
    "internlm", "internlm2", "yi", "baichuan",
    "chatglm", "glm", "glm4",
    "command-r", "cohere", "cohere2", "dbrx",
    "mpt", "olmo", "olmo2", "jamba", "arctic",
    "exaone", "granite", "granitemoeshared",
    "stablelm", "persimmon",
})

# Explicit multimodal / vision signals that win even if an architecture ends with ForCausalLM (e.g. Phi3VForCausalLM).
_UNSUPPORTED_MODEL_TYPES = frozenset({
    "gemma3",
    "mllama",
    "llava",
    "llava_next",
    "qwen2_vl",
    "qwen2_5_vl",
    "idefics",
    "idefics2",
    "idefics3",
    "paligemma",
    "pixtral",
    "internvl_chat",
    "phi3_v",
})

_UNSUPPORTED_ARCHITECTURES = frozenset({
    # RWKV6/Qwen2 hybrid linear-attention arch: not in sglang's supported list
    # (only plain RwkvForCausalLM is), fails ModelConfig validation at boot.
    "RWKV6Qwen2ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "InternVLChatModel",
    "Phi3VForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "PixtralForConditionalGeneration",
})

_UNSUPPORTED_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "image_token_index",
    "mm_config",
    "multi_modal_config",
    "vision_tower",
    "vision_tower_cfg",
    "image_processor_type",
    "projector_config",
    "mm_projector_type",
)


_TEXT_COMPAT_MULTIMODAL_EXCEPTIONS = frozenset({
    ("kimi_k25", "KimiK25ForConditionalGeneration"),
    ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
})


def _arch_is_supported_text_generation(arch: str) -> bool:
    """True when an architecture class name denotes a supported text-generation
    (decoder-only causal LM) model."""
    a = (arch or "").strip()
    if not a:
        return False
    return any(marker in a for marker in _SUPPORTED_ARCH_MARKERS)


def _is_text_compatible_multimodal_exception(
    model_type_l: str,
    architectures: list[str],
) -> bool:
    """Allow known multimodal configs whose text path is benchmark-compatible.

    Kimi-K2.6 and Qwen3.6 MoE ship with ``vision_config`` even when used as
    text-only checkpoints. Their serving stacks can exercise the text-generation
    path for our benchmark; the generic multimodal gate was too broad and
    rejected them before baseline could start. Keep this list exact so ordinary
    VLMs remain fail-fast.
    """
    return any(
        (model_type_l, arch) in _TEXT_COMPAT_MULTIMODAL_EXCEPTIONS
        for arch in architectures
    )


def _detect_unsupported_model(model_path: str) -> dict | None:
    """Best-effort classify a model as unsupported (non-text-generation): multimodal/vision rejected, else whitelist.

    Returns ``{"architecture", "model_type", "signal"}`` when unsupported, else ``None`` (also for an
    unreadable config.json — we don't hard-block on a config we cannot read).
    """
    config = _load_model_config_dict(model_path)
    if config is None:
        return None
    architectures = _config_architectures(config)
    # Wrapper models may nest the real arch under text_config; merge so the
    # unsupported-arch blocklist still matches (e.g. RWKV6Qwen2ForCausalLM).
    nested = config.get("text_config")
    if isinstance(nested, dict):
        for a in _config_architectures(nested):
            if a not in architectures:
                architectures.append(a)
    model_type = str(config.get("model_type") or "").strip()
    model_type_l = model_type.lower()

    if _is_text_compatible_multimodal_exception(model_type_l, architectures):
        return None

    for arch in architectures:
        if arch in _UNSUPPORTED_ARCHITECTURES:
            return {
                "architecture": arch,
                "model_type": model_type,
                "signal": f"unsupported architecture '{arch}'",
            }
    if model_type_l in _UNSUPPORTED_MODEL_TYPES:
        return {
            "architecture": architectures[0] if architectures else "",
            "model_type": model_type,
            "signal": f"unsupported model_type '{model_type}'",
        }
    for key in _UNSUPPORTED_CONFIG_KEYS:
        if key in config:
            return {
                "architecture": architectures[0] if architectures else "",
                "model_type": model_type,
                "signal": f"unsupported multimodal config key '{key}'",
            }

    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return None

    if model_type_l in _SUPPORTED_MODEL_TYPES:
        return None

    if architectures:
        return {
            "architecture": architectures[0],
            "model_type": model_type,
            "signal": (
                f"architecture '{architectures[0]}' does not match any "
                f"supported text-generation pattern "
                f"({', '.join(_SUPPORTED_ARCH_MARKERS)})"
            ),
        }

    if model_type:
        return {
            "architecture": "",
            "model_type": model_type,
            "signal": (
                f"model_type '{model_type}' is not in the supported "
                f"text-generation allowlist"
            ),
        }

    return {
        "architecture": "",
        "model_type": "",
        "signal": "config.json has neither architectures nor model_type",
    }


# config.json keys carrying max sequence length, priority order (legacy/alt configs use aliases).
_MAXPOS_CONFIG_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
)

# Safety headroom (tokens) above ISL+OSL: covers BOS/chat-template tokens and dataset length jitter.
_CONTEXT_HEADROOM_ENV = "HYPERLOOM_CONTEXT_HEADROOM_TOKENS"
_CONTEXT_HEADROOM_DEFAULT = 512

# Extra context budget on top of ISL+OSL for server-facing MAX_MODEL_LEN; always clamped to native
# window by _resolve_max_model_len (vllm wires it into --max-model-len, so an unclamped value crashes).
_MAX_MODEL_LEN_HEADROOM = 4096


def _load_model_max_position_embeddings(model_path: str) -> int | None:
    """Best-effort read of max sequence length from config.json (first positive among known keys, incl. nested ``text_config``), or None."""
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        for key in _MAXPOS_CONFIG_KEYS:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 0:
                return val
    return None


def _model_has_dual_chunk_attention(model_path: str) -> bool:
    """Best-effort detect a ``dual_chunk_attention_config`` in config.json.

    Qwen 1M long-context models ship this block; sglang then rejects the
    default aiter attention backend and demands ``dual_chunk_flash_attn``.
    Checks the top level and a nested ``text_config``. Soft-degrades to
    False on any missing / unreadable / invalid config.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    if data.get("dual_chunk_attention_config"):
        return True
    nested = data.get("text_config")
    return isinstance(nested, dict) and bool(
        nested.get("dual_chunk_attention_config")
    )


def _model_is_moe(model_path: str) -> bool:
    """Best-effort detect a Mixture-of-Experts model from config.json.

    MoE checkpoints declare an expert count (``num_experts`` /
    ``num_local_experts`` / ``n_routed_experts``), a ``moe_intermediate_size``,
    or carry a ``moe`` marker in ``architectures`` / ``model_type`` (e.g.
    Qwen3MoeForCausalLM / qwen3_moe). On ROCm/aiter, sglang's default
    ``--moe-runner-backend auto`` routes these through aiter's CK 2-stage
    fused-MoE kernel, whose first-request JIT build is broken in some images;
    callers use this to switch to a ROCm-capable MoE runner. Checks the top
    level and a nested ``text_config``. Soft-degrades to False on any missing
    / unreadable / invalid config.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    expert_keys = ("num_experts", "num_local_experts", "n_routed_experts")
    for cfg in candidates:
        for key in expert_keys:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 1:
                return True
        if cfg.get("moe_intermediate_size"):
            return True
        if "moe" in str(cfg.get("model_type") or "").lower():
            return True
        if any("moe" in arch.lower() for arch in _config_architectures(cfg)):
            return True
    return False


def _context_headroom_tokens() -> int:
    """Resolve the context headroom (tokens); env override, else default."""
    raw = os.environ.get(_CONTEXT_HEADROOM_ENV, "").strip()
    if not raw:
        return _CONTEXT_HEADROOM_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        return _CONTEXT_HEADROOM_DEFAULT
    return val if val >= 0 else _CONTEXT_HEADROOM_DEFAULT


def _resolve_max_model_len(isl: int, osl: int, model_path: str) -> int:
    """Resolve ``MAX_MODEL_LEN`` = ISL+OSL+headroom, clamped to ``max_position_embeddings`` (never stretch context)."""
    desired = int(isl) + int(osl) + _MAX_MODEL_LEN_HEADROOM
    maxpos = _load_model_max_position_embeddings(model_path)
    if maxpos:
        return min(desired, maxpos)
    return desired


def _preflight_context_window(args: argparse.Namespace, session_dir: Path) -> bool:
    """Fail fast when ``max_position_embeddings < ISL+OSL+headroom`` (no --context-length stretch by policy).

    Persists a stop reason and returns True (caller should exit) when the workload does NOT fit; False
    when it fits or the model's max length is unknown.
    """
    isl = int(getattr(args, "isl", 0) or 0)
    osl = int(getattr(args, "osl", 0) or 0)
    if isl <= 0 or osl <= 0:
        return False
    maxpos = _load_model_max_position_embeddings(str(getattr(args, "model", "") or ""))
    if not maxpos:
        return False
    headroom = _context_headroom_tokens()
    required = isl + osl + headroom
    if maxpos >= required:
        return False

    reason = (
        f"model max_position_embeddings={maxpos} < required {required} "
        f"(ISL={isl} + OSL={osl} + headroom={headroom}). The workload exceeds "
        f"the model context window; every request would 400. Refusing to run "
        f"(no --context-length override by policy). Lower ISL/OSL for this "
        f"model, or lower {_CONTEXT_HEADROOM_ENV} if the headroom is too "
        f"conservative (it is added to `required`, so raising it makes "
        f"admission stricter, not looser)."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json instead of the log.
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3 (term registered in STOP_REASON_VOCAB).
        state.set_stop_reason("model_context_window_too_small")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist context-window stop report: {exc!r}",
            file=sys.stderr,
        )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before
    # coordinator.run()'s finally, so CI's delivery contract sees a clean skip not "Missing artifacts".
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on context "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )
    print(f"ERROR: {reason}", file=sys.stderr)
    return True


# RoPE-config signals: their presence means the model uses extended/scaled
# positions, so transformers/vLLM read a max-position field during rope init.
# A config that ships these but no max-position key crashes with
# "'PreTrainedConfig' object has no attribute 'max_position_embeddings'" deep
# in engine init.
_ROPE_CONFIG_KEYS = ("rope_scaling", "rope_parameters", "rope_theta")

# Architectures whose runtime path is not adapted to AMD/ROCm yet.
# Matched case-insensitively against model_type and architectures.
_AMD_UNSUPPORTED_MODEL_TYPES = frozenset({"deepseek_v32"})
_AMD_UNSUPPORTED_ARCHITECTURES = frozenset({"deepseekv32forcausallm"})

# model_type values that ship a custom AutoConfig (auto_map) but aren't
# registered in sglang/vLLM's config mapping. sglang falls back to
# PreTrainedConfig (base class), which lacks max_position_embeddings etc.,
# causing AttributeError deep in engine init.
_UNREGISTERED_CUSTOM_CONFIG_TYPES = frozenset({"kimi_k2"})

# Quantization formats with no ROCm/AMD runtime path. NVIDIA ModelOpt FP8/NVFP4
# use vendor-specific scale packing (no sglang ROCm loader); bitsandbytes ships
# CUDA-only kernels; NVFP4/FP4 need Blackwell hardware. AMD-native fp8 (Quark /
# compressed-tensors), gptq, awq are NOT listed so they keep running.
_AMD_UNSUPPORTED_QUANT_ALGOS = frozenset({"nvfp4", "fp4"})
_AMD_UNSUPPORTED_QUANT_METHODS = frozenset({"bitsandbytes", "bnb"})


def _detect_amd_unsupported_quant(model_path: str) -> str | None:
    """Return a reason when the model ships a quant format unsupported on ROCm.

    Reads both ``config.json:quantization_config`` (standard HF) and the
    separate ``hf_quant_config.json`` (NVIDIA ModelOpt). Returns None when the
    format is ROCm-runnable or absent.
    """
    if not model_path:
        return None
    cfg = _load_model_config_dict(model_path) or {}
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method") or "").strip().lower()
        if method in _AMD_UNSUPPORTED_QUANT_METHODS:
            return (
                f"quantization_config.quant_method '{method}' ships CUDA-only "
                f"kernels with no ROCm equivalent; it crashes in engine init "
                f"on AMD."
            )
    # NVIDIA ModelOpt writes a separate hf_quant_config.json, not config.json.
    hq_path = Path(model_path) / "hf_quant_config.json"
    if hq_path.is_file():
        try:
            hq = json.loads(hq_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            hq = None
        if isinstance(hq, dict):
            producer = str(
                (hq.get("producer") or {}).get("name") or "",
            ).strip().lower()
            algo = str(
                (hq.get("quantization") or {}).get("quant_algo") or "",
            ).strip().lower()
            if producer == "modelopt" and algo:
                return (
                    f"NVIDIA ModelOpt '{algo.upper()}' quantization "
                    f"(hf_quant_config.json) uses vendor-specific scale packing "
                    f"with no sglang ROCm loader (e.g. 'modelopt_fp8 ... not "
                    f"supported in ROCm'); use an AMD-native (Quark) checkpoint."
                )
            if algo in _AMD_UNSUPPORTED_QUANT_ALGOS:
                return (
                    f"'{algo.upper()}' quantization needs NVIDIA Blackwell "
                    f"hardware; no AMD/ROCm runtime path exists."
                )
    return None


def _detect_incompatible_model_config(
    model_path: str, gpu_type: str | None = None,
) -> str | None:
    """Detect a statically-knowable model-config incompatibility.

    Returns a human-readable reason string when the model's ``config.json``
    will crash vLLM/transformers at load time, else ``None``. Two cases,
    both conservative (no false positives on healthy configs):

    * ``config.json`` is present but corrupt / not a JSON object — the loader
      soft-degrades to ``None``, but a present-yet-unparseable file means the
      framework will fail at config load, so block early.
    * the config (top level or ``text_config``) declares a RoPE block but has
      no max-position key at all — the rope init then dereferences a missing
      ``max_position_embeddings``.

    A fully absent ``config.json`` is NOT blocked (kept soft-degrade): the
    upstream submission filter + downstream loader still apply.
    """
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return None
    data = _load_model_config_dict(model_path)
    if data is None:
        # File exists but did not parse into a dict (corrupt / non-object).
        return (
            f"config.json at {cfg_path} is present but unparseable "
            f"(corrupt JSON or not a JSON object); the framework would crash "
            f"at config load."
        )
    # Reject DSA-like architectures only on AMD/ROCm.
    # The same model can still run on vendor-supported NVIDIA engines.
    if _resolve_amd_gpu_type(gpu_type):
        quant_reason = _detect_amd_unsupported_quant(model_path)
        if quant_reason is not None:
            return quant_reason
        model_type = str(data.get("model_type") or "").strip().lower()
        arches = {a.lower() for a in _config_architectures(data)}
        if (
            model_type in _AMD_UNSUPPORTED_MODEL_TYPES
            or arches & _AMD_UNSUPPORTED_ARCHITECTURES
        ):
            label = model_type or (next(iter(arches), "") if arches else "?")
            return (
                f"model architecture '{label}' has no AMD/ROCm runtime path "
                f"(needs a vendor engine on NVIDIA Hopper/Blackwell, e.g. "
                f"DeepSeek Sparse Attention); it crashes in engine init on "
                f"this hardware."
            )
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    has_rope = any(
        s.get(k) for s in scopes for k in _ROPE_CONFIG_KEYS
    )
    has_maxpos = any(
        isinstance(s.get(k), int) and not isinstance(s.get(k), bool)
        and s.get(k) > 0
        for s in scopes for k in _MAXPOS_CONFIG_KEYS
    )
    if has_rope and not has_maxpos:
        return (
            "config.json declares a RoPE block "
            f"({', '.join(_ROPE_CONFIG_KEYS)}) but no max-position field "
            f"({', '.join(_MAXPOS_CONFIG_KEYS)}); transformers/vLLM rope "
            "init dereferences a missing max_position_embeddings and crashes "
            "in engine init (DeepSeek-V3.2-Exp class)."
        )
    # Custom AutoConfig with unregistered model_type: sglang/vLLM fall
    # back to PreTrainedConfig (no max_position_embeddings attr) → crash.
    auto_map = data.get("auto_map")
    model_type = str(data.get("model_type") or "").strip().lower()
    if (
        isinstance(auto_map, dict)
        and auto_map.get("AutoConfig")
        and model_type in _UNREGISTERED_CUSTOM_CONFIG_TYPES
    ):
        return (
            f"model_type '{model_type}' ships a custom AutoConfig "
            f"({auto_map['AutoConfig']}) but is not registered in sglang/"
            f"vLLM's config mapping; the engine falls back to "
            f"PreTrainedConfig which lacks key attributes "
            f"(max_position_embeddings) and crashes in init."
        )
    # Dual-chunk attention on AMD/ROCm: sglang hard-requires
    # dual_chunk_flash_attn (sm90+ only) and rejects all other backends.
    if _resolve_amd_gpu_type(gpu_type) and _model_has_dual_chunk_attention(
        model_path
    ):
        return (
            "model declares dual_chunk_attention_config but sglang requires "
            "the dual_chunk_flash_attn backend which only builds on sm90+ "
            "(NVIDIA Hopper); no compatible backend exists for AMD/ROCm."
        )
    return None


def _preflight_model_config_compat(
    args: argparse.Namespace, session_dir: Path,
) -> bool:
    """Fail fast when the model config is statically known to be incompatible.

    Catches configs that crash vLLM/transformers at load (corrupt config.json,
    or a RoPE block without any max-position field) so we persist a clear stop
    reason instead of booting a server that dies cryptically in engine init.

    Returns True when incompatible (caller should exit); False otherwise.
    """
    model = str(getattr(args, "model", "") or "")
    detail = _detect_incompatible_model_config(
        model, str(getattr(args, "gpu_type", "") or "") or None,
    )
    if detail is None:
        return False
    name = Path(model).name or model
    reason = (
        f"Model '{name}' has an incompatible config: {detail} Refusing to run "
        f"before the heavy server bring-up. Upgrade the framework/transformers "
        f"to a version that supports this model, or skip it on this hardware."
    )
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        state.set_stop_reason("model_config_incompatible")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist model-config stop report: {exc!r}",
            file=sys.stderr,
        )
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on config "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )
    print(f"ERROR: {reason}", file=sys.stderr)
    return True


def _preflight_unsupported_model_arch(
    args: argparse.Namespace, session_dir: Path,
) -> bool:
    """Fail fast on positively-identified unsupported multimodal/vision models before expensive bring-up.

    Best-effort (an unreadable config.json is not a hard block). Persists a stop reason and returns True
    (caller should exit) when unsupported; False when supported or unclassifiable.
    """
    model = str(getattr(args, "model", "") or "")
    hit = _detect_unsupported_model(model)
    if hit is None:
        return False

    name = Path(model).name or model
    arch = hit.get("architecture") or "<unknown>"
    mt = hit.get("model_type") or "<unknown>"
    reason = (
        f"Unsupported model '{name}': architecture '{arch}' (model_type "
        f"'{mt}') is not a supported text-generation model. Hyperloom only "
        f"supports decoder-only causal LM models (architectures containing "
        f"ForCausalLM or LMHeadModel). Rejected because: "
        f"{hit.get('signal', 'unknown architecture')}. Submit a "
        f"text-generation checkpoint instead."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json instead of the log.
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3 (term registered in STOP_REASON_VOCAB).
        state.set_stop_reason("unsupported_model_arch")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist unsupported-model stop report: {exc!r}",
            file=sys.stderr,
        )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before
    # coordinator.run()'s finally, so CI's delivery contract sees a clean skip not "Missing artifacts".
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on unsupported-"
            f"model fail-fast: {exc!r}",
            file=sys.stderr,
        )
    print(f"ERROR: {reason}", file=sys.stderr)
    return True


def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
) -> SharedState:
    # research_lane capacity is locked for the session; clamp to [0, ceiling] (2×GPU) to protect quota/PR-Monitor.
    from inference_optimizer.orchestrator.policy import (
        research_lane_ceiling,
    )
    research_lane_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 1
    )
    research_lane_capacity = max(
        0, min(research_lane_ceiling(), research_lane_capacity),
    )
    gpu_specialist_capacity_raw = getattr(
        args, "gpu_specialist_capacity", None,
    )
    try:
        gpu_specialist_capacity = max(
            0,
            int(gpu_specialist_capacity_raw)
            if gpu_specialist_capacity_raw is not None else 0,
        )
    except (TypeError, ValueError):
        gpu_specialist_capacity = 0
    # Collect plateau threshold overrides; absent keys fall through to DEFAULT_PLATEAU_* at compute time.
    plateau_overrides: dict[str, Any] = {}
    if getattr(args, "plateau_explore_keep_gain", None) is not None:
        plateau_overrides["explore_keep_gain_pct"] = float(args.plateau_explore_keep_gain)
    if getattr(args, "plateau_explore_empty_streak", None) is not None:
        plateau_overrides["explore_empty_streak"] = int(args.plateau_explore_empty_streak)
    if getattr(args, "plateau_explore_lookback", None) is not None:
        plateau_overrides["explore_lookback"] = int(args.plateau_explore_lookback)
    if getattr(args, "plateau_kernel_revert_streak", None) is not None:
        plateau_overrides["kernel_revert_streak"] = int(args.plateau_kernel_revert_streak)
    if getattr(args, "plateau_kernel_keep_gain", None) is not None:
        plateau_overrides["kernel_keep_gain_pct"] = float(args.plateau_kernel_keep_gain)
    if getattr(args, "plateau_kernel_lookback", None) is not None:
        plateau_overrides["kernel_lookback"] = int(args.plateau_kernel_lookback)
    # EXPLORE HARD force-exit thresholds; either fires an explore_force_exit_low_budget exit (overrides all).
    if getattr(args, "explore_force_exit_hours_remaining", None) is not None:
        plateau_overrides["force_exit_hours_remaining"] = float(
            args.explore_force_exit_hours_remaining
        )
    if getattr(args, "explore_force_exit_budget_pct", None) is not None:
        plateau_overrides["force_exit_budget_pct"] = float(
            args.explore_force_exit_budget_pct
        )
    # Resolve workload metadata from CLI flags then env; parse duplicated here to avoid re-reading manifest.json.
    def _int_env_or_arg(arg_name: str, env_name: str) -> int:
        """Resolve an int workload knob from a CLI arg, falling back to env.

        Args:
            arg_name (str): Attribute name to read off ``args``.
            env_name (str): Environment variable consulted when the arg is unset/0.

        Returns:
            int: The resolved value, or 0 when neither source yields a valid int.
        """
        val = getattr(args, arg_name, None)
        if val is None or val == 0:
            raw = (os.environ.get(env_name, "") or "").strip()
            return int(raw) if raw.isdigit() else 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def _resolve_framework_version(args_in: Any) -> str:
        """Resolve ``framework_version`` for the recipe-snapshot canonical id.

        Ladder: explicit CLI/$FRAMEWORK_VERSION → auto-detect package __version__ → "" (canonical_id
        substitutes unknown_version). Auto-detect runs only when both CLI and env are empty.
        """
        explicit = (
            (getattr(args_in, "framework_version", None) or "").strip()
            or (os.environ.get("FRAMEWORK_VERSION", "") or "").strip()
        )
        if explicit:
            return explicit
        framework = (
            (getattr(args_in, "framework", None) or "").strip()
            or (os.environ.get("FRAMEWORK", "") or "").strip()
        )
        if not framework:
            return ""
        from .recipe_snapshot_constants import (
            DEFAULT_FRAMEWORK_VERSION_SLUG,
            detect_framework_version,
        )

        detected = detect_framework_version(framework)
        # Treat the failure-slug as "no info"; canonical_id redoes the fallback at use time.
        return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected

    # --explore-overtime-kill-ratio: mirror into fresh SharedState for ExploreExecutor; <=0 disables the gate.
    explore_overtime_kill_ratio_raw = getattr(
        args, "explore_overtime_kill_ratio", None,
    )
    try:
        explore_overtime_kill_ratio = (
            float(explore_overtime_kill_ratio_raw)
            if explore_overtime_kill_ratio_raw is not None else 1.10
        )
    except (TypeError, ValueError):
        explore_overtime_kill_ratio = 1.10

    # --explore-variant-timeout-sec mirror; 0 (default) auto-derives the cap, positive pins it.
    explore_variant_timeout_raw = getattr(
        args, "explore_variant_timeout_sec", None,
    )
    try:
        explore_variant_timeout_sec_override = max(
            0,
            int(explore_variant_timeout_raw)
            if explore_variant_timeout_raw is not None else 0,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_sec_override = 0

    # --explore-variant-timeout-safety-margin mirror: auto-derive headroom over the soft kill ratio (neg -> 0).
    explore_variant_timeout_safety_margin_raw = getattr(
        args, "explore_variant_timeout_safety_margin", None,
    )
    try:
        explore_variant_timeout_safety_margin = max(
            0.0,
            float(explore_variant_timeout_safety_margin_raw)
            if explore_variant_timeout_safety_margin_raw is not None else 0.5,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_safety_margin = 0.5

    # KB architecture tags from config.json (architectures + model_type); fresh-launch only (resume rehydrates).
    _cfg_tags = _load_model_config_tags(str(args.model))

    state = SharedState(
        session_id=session_id,
        claw_session_id=(os.environ.get("CLAW_SESSION_ID") or "").strip(),
        sandbox_user_id=(os.environ.get("SANDBOX_USER_ID") or "").strip(),
        model_name=Path(args.model).name,
        model_path=str(args.model),
        model_class=args.model_class or "",
        # Advisory architecture profile; fresh-launch only (resume rehydrates, must not clobber). Soft-degrade to {}.
        model_arch=_load_model_arch(
            _workspace_root_resolve(), Path(args.model).name
        ),
        # Architecture-identity tags from config.json stamped into recipe-snapshot extras (fine-tune carries base identity).
        model_architectures=_cfg_tags.get("architectures", []),
        model_type=_cfg_tags.get("model_type", ""),
        framework=os.environ.get("FRAMEWORK", "sglang"),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI/env so downstream prompts see real values (else TP defaults to 1).
        tp=_int_env_or_arg("tp", "TP"),
        # ``ep`` mirrors EP env so fresh-shell resume recovers it for the KB warm-start same-shape filter.
        ep=_int_env_or_arg("ep", "EP"),
        precision=(
            str(getattr(args, "precision", None) or os.environ.get("PRECISION", "") or "").strip()
        ),
        framework_version=_resolve_framework_version(args),
        conc=_int_env_or_arg("conc", "CONC"),
        isl=_int_env_or_arg("isl", "ISL"),
        osl=_int_env_or_arg("osl", "OSL"),
        max_model_len=_int_env_or_arg("max_model_len", "MAX_MODEL_LEN"),
        kernel_enabled=not getattr(args, "no_kernel", False),
        continue_kernel_after_gemm=bool(
            getattr(args, "continue_kernel_after_gemm", True)
        ),
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain=0.0,
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        gpu_specialist_capacity=gpu_specialist_capacity,
        plateau_overrides=plateau_overrides,
        explore_overtime_kill_ratio=explore_overtime_kill_ratio,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_PR phase; --no-framework skips it (mirrors --no-kernel/kernel_enabled).
        framework_phase_enabled=not bool(getattr(args, "no_framework", False)),
        # --no-explore skips the EXPLORE phase entirely.
        explore_enabled=not bool(getattr(args, "no_explore", False)),
        explore_variant_timeout_sec_override=explore_variant_timeout_sec_override,
        explore_variant_timeout_safety_margin=explore_variant_timeout_safety_margin,
        research_scout_enabled=bool(getattr(args, "research_scout", True)),
        research_scout_interval=max(
            1, int(getattr(args, "research_scout_interval", 3) or 3)
        ),
        target_advisory_enabled=bool(getattr(args, "target_advisory", True)),
        recipe_sediment_enabled=bool(getattr(args, "recipe_sediment", True)),
        # SWEEP-phase post-sweep concurrency sweep flags (on by default); see orchestrator/conc_sweep.py.
        conc_sweep_enabled=bool(getattr(args, "enable_conc_sweep", True)),
        conc_sweep_concs=_parse_conc_sweep_concs(args),
        conc_sweep_total_budget_sec=int(
            getattr(args, "conc_sweep_total_budget_sec", 9000) or 0,
        ),
        conc_sweep_variant_timeout_sec=int(
            getattr(args, "conc_sweep_timeout_sec", 1800) or 1800,
        ),
    )
    state.save(session_dir)
    return state


def _parse_conc_sweep_concs(args: argparse.Namespace) -> list[int]:
    """Parse ``--conc-sweep-concs '1,2,4,8'`` into a list[int]; non-integers warned+dropped, empty -> 1..128 ladder."""
    raw = str(getattr(args, "conc_sweep_concs", "") or "").strip()
    if not raw:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    out: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            log.warning("conc_sweep: ignoring non-integer CONC token %r", t)
    return out or [1, 2, 4, 8, 16, 32, 64, 128]


def _print_session_skeleton(session_dir: Path) -> None:
    """Echo the freshly-created skeleton so launchers see the exact layout.

    Args:
        session_dir (Path): The session root directory whose skeleton
            subdirectories are listed.
    """
    print(f"Session layout under {session_dir}:")
    for sub in _SESSION_SKELETON:
        marker = "ok" if (session_dir / sub).is_dir() else "MISSING"
        print(f"  [{marker}] {sub}/")
    print("  [ok] manifest.json (written first)")


def _snapshot_system_prompts(
    session_dir: Path,
    *,
    prompts: dict[str, str],
) -> None:
    """Persist each agent's effective system prompt to ``agents/<role>/system_prompt.snapshot.md``."""
    for role, body in prompts.items():
        target = agent_prompt_snapshot(session_dir, role)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")


def _default_target_summary(args: argparse.Namespace) -> str:
    """Compose a human-readable objective summary from the CLI target flags.

    Used as the fallback ``target_summary`` when the operator did not pass an
    explicit ``--target-summary``. The phrasing depends on which target flag is
    set: ``--target-gain`` (percentage), ``--target-tput`` (tok/s/GPU), or
    neither (open-ended optimization within the time budget).

    Args:
        args (argparse.Namespace): Parsed ``optimize`` arguments (reads ``model``,
            ``target_gain``, ``target_tput``, ``max_hours``).

    Returns:
        str: A one-sentence description of the run's objective.
    """
    if args.target_gain:
        return (
            f"Establish baseline on {Path(args.model).name} then drive "
            f"cumulative_gain to >= {args.target_gain}% within "
            f"{args.max_hours}h."
        )
    if args.target_tput:
        return (
            f"Establish baseline on {Path(args.model).name} then reach "
            f"{args.target_tput} tok/s/GPU within {args.max_hours}h."
        )
    return f"Optimize {Path(args.model).name} for up to {args.max_hours}h (no target)."


def _print_final_summary(state: SharedState, stop_reason: str) -> None:
    """Print the end-of-run summary block to stdout.

    Reports the stop reason, session id, model, baseline throughput, the
    per-round (informational) cumulative gain, the validated cumulative gain
    (with a staleness warning when the optimization stack grew after the last
    validation), the current best config, pruned families, and crash count.

    Args:
        state (SharedState): The final shared state after the run completes.
        stop_reason (str): Why the run stopped (e.g. ``"target_reached"``).

    Returns:
        None
    """
    print()
    print("================ Final summary ================")
    print(f"  stop_reason          : {stop_reason}")
    print(f"  session_id           : {state.session_id}")
    print(f"  model                : {state.model_name}")
    print(f"  baseline_tput        : {state.baseline_tput:.1f} tok/s/GPU")
    print(
        f"  cumulative_gain      : {state.cumulative_gain:.2f}% "
        f"(per-round sum — informational)"
    )
    if state.cumulative_gain_validated_ts:
        stale = (
            " ⚠ stack changed since validation"
            if len(state.optimization_stack) > state.cumulative_gain_validated_stack_len
            else ""
        )
        print(
            f"  cumulative_gain_val  : {state.cumulative_gain_validated:.2f}% "
            f"(validated_at_stack_len={state.cumulative_gain_validated_stack_len}, "
            f"ts={state.cumulative_gain_validated_ts}){stale}"
        )
    else:
        print(
            "  cumulative_gain_val  : 0.00% "
            "⚠ never validated — no `explore` stack-rebench has succeeded yet"
        )
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    _print_kernel_opt_summary_line(state)
    print("===============================================")


def _reconcile_crash_count(state: SharedState, session_dir: Path) -> None:
    """Reconcile persisted ``crash_count`` (state.json + final.json) up to the live in-memory value.

    Only ever raises the persisted value (max), never lowers it; best-effort, never fatal.
    """
    live = int(getattr(state, "crash_count", 0) or 0)

    # 1) state.json — reload, bump if stale, atomic re-save.
    try:
        disk_state = SharedState.load_or_init(session_dir)
        if int(disk_state.crash_count or 0) < live:
            disk_state.crash_count = live
            disk_state.save(session_dir)
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (state.json) failed (non-fatal)")

    # 2) reports/final.json — patch the single field in place if present.
    try:
        from .session_paths import reports_dir
        final_json = reports_dir(session_dir) / "final.json"
        if final_json.exists():
            data = json.loads(final_json.read_text(encoding="utf-8"))
            if int(data.get("crash_count") or 0) < live:
                data["crash_count"] = live
                final_json.write_text(
                    json.dumps(data, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (final.json) failed (non-fatal)")


def _print_kernel_opt_summary_line(state: SharedState) -> None:
    """One-line forensic readout of kernel_opt attempts at session end (matches the on-disk report; best-effort)."""
    try:
        from .orchestrator.kernel_attempt_summary import (
            build_kernel_optimization_summary,
        )
        session_dir = _resolve_session_dir_for_summary(state)
        if session_dir is None:
            return
        summary = build_kernel_optimization_summary(state, session_dir)
        totals = summary.get("totals") or {}
        attempted = int(totals.get("attempted") or 0)
        if attempted == 0 and int(totals.get("unattempted") or 0) == 0:
            return
        integrated = int(totals.get("integrated") or 0)
        rejected = int(totals.get("rejected") or 0)
        unattempted = int(totals.get("unattempted") or 0)
        print(
            f"  kernel_opt           : {attempted} attempted "
            f"({integrated} integrated, {rejected} rejected), "
            f"{unattempted} unattempted in top candidates"
        )
        takeaways = summary.get("top_takeaways") or []
        if len(takeaways) >= 2:
            print(f"  kernel_opt_top_cause : {takeaways[1]}")
        report_path = (
            Path(session_dir) / "reports" / "kernel_optimization_summary.json"
        )
        if report_path.is_file():
            print(f"  kernel_opt_report    : {report_path}")
    except Exception:  # noqa: BLE001 — stdout print must never fail the run
        pass


def _resolve_session_dir_for_summary(state: SharedState) -> Path | None:
    """Best-effort session_dir lookup ($HYPERLOOM_SESSION_DIR) for the stdout kernel_opt line; ``None`` if unresolved."""
    env_sd = os.environ.get("HYPERLOOM_SESSION_DIR", "").strip()
    if env_sd:
        p = Path(env_sd).expanduser()
        if p.is_dir():
            return p
    return None


def _derive_anthropic_base_url(openai_base_url: str) -> str:
    """Derive ``ANTHROPIC_BASE_URL`` from ``OPENAI_BASE_URL`` by stripping a trailing ``/v1`` (SDK re-appends it)."""
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(openai_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunparse(parsed._replace(path=path))


def _reset_claude_config_to_upstream(
    safe_key: str, anthropic_base_url: str
) -> None:
    """Point ``~/.claude/config.json`` ``customApiUrl`` at the upstream gateway (stale 127.0.0.1:4002 would fail)."""
    import json as _json

    if not anthropic_base_url:
        return
    claude_config_path = Path.home() / ".claude" / "config.json"
    config_data: dict = {}
    if claude_config_path.exists():
        try:
            config_data = _json.loads(
                claude_config_path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            config_data = {}
        current_url = config_data.get("customApiUrl", "")
        if current_url == anthropic_base_url:
            print("Preflight: ~/.claude/config.json already points at upstream")
            return

    config_data.setdefault("theme", "dark")
    config_data.setdefault("hasCompletedOnboarding", True)
    if safe_key:
        config_data["primaryApiKey"] = safe_key
    elif "primaryApiKey" not in config_data:
        config_data["primaryApiKey"] = ""
    config_data["customApiUrl"] = anthropic_base_url
    claude_config_path.parent.mkdir(parents=True, exist_ok=True)
    claude_config_path.write_text(
        _json.dumps(config_data, indent=2) + "\n", encoding="utf-8",
    )
    claude_config_path.chmod(0o600)
    print(
        f"Preflight: updated ~/.claude/config.json customApiUrl -> "
        f"{anthropic_base_url}"
    )


def _validate_credentials() -> None:
    """Fail fast when SAFE_API_KEY or OPENAI_BASE_URL is missing; strict by design (no bypass)."""
    missing: list[str] = []
    if not os.environ.get("SAFE_API_KEY"):
        missing.append("SAFE_API_KEY")
    if not os.environ.get("OPENAI_BASE_URL"):
        missing.append("OPENAI_BASE_URL")
    if not missing:
        return
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    env_status = "present" if env_file.exists() else "not found"
    print(
        "\nERROR: Missing required credential(s): "
        f"{', '.join(missing)}\n\n"
        "Tried loading from:\n"
        "  - shell environment\n"
        f"  - $REPO_ROOT/.env  ({env_status}: {env_file})\n\n"
        "Fix one of:\n"
        "  1. Copy .env from a working worktree into this one:\n"
        f"       cp /path/to/main-worktree/.env {env_file}\n"
        "  2. Export directly into the shell before re-running:\n"
        "       export SAFE_API_KEY=sk-xxxxx\n"
        "       export OPENAI_BASE_URL=https://gateway.example.com/v1",
        file=sys.stderr,
    )
    sys.exit(2)


def _is_placeholder_tracelens_path(value: str) -> bool:
    """Treat .env.template's bare ``\\`` and whitespace-only values as unset."""
    stripped = value.strip()
    return stripped in ("", "\\")


def _load_dotenv_fallback() -> None:
    """Source missing keys from ``$REPO_ROOT/.env`` when SAFE_API_KEY/OPENAI_BASE_URL absent; env always wins."""
    if os.environ.get("SAFE_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        return
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
            line = line[len("export "):].lstrip()
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


def _load_kernel_agent_env_fallback() -> None:
    """If ``HYPERLOOM_KERNEL_AGENT_ROOT`` is unset, auto-source the
    kernel-agent env file produced by ``inference_optimizer/scripts/
    install.sh`` at ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``
    (overridable via ``$KERNEL_AGENT_ENV``).

    Must source before any orchestrator import (trace_analyze reads HYPERLOOM_KERNEL_AGENT_ROOT at module load).
    Hard-fail contract: looks only at $KERNEL_AGENT_ENV or $USER_DATA_PATH/runtime/kernel-agent.env.sh (no
    parent-dir fallback); sys.exit(2) if missing/0-vars/still-unset; skip when the var is already set.
    """
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    candidate = os.environ.get("KERNEL_AGENT_ENV")
    if not candidate:
        user_data = os.environ.get("USER_DATA_PATH")
        if not user_data:
            print(
                "Preflight: ERROR — neither $HYPERLOOM_KERNEL_AGENT_ROOT "
                "nor $KERNEL_AGENT_ENV nor $USER_DATA_PATH is set. Cannot "
                "resolve kernel-agent.env.sh. Run "
                "inference_optimizer/scripts/install.sh and export "
                "USER_DATA_PATH=/path/to/sessions first.",
                file=sys.stderr,
            )
            sys.exit(2)
        candidate = str(Path(user_data) / "runtime" / "kernel-agent.env.sh")
    env_path = Path(candidate)
    if not env_path.is_file():
        print(
            f"Preflight: ERROR — kernel-agent env file not found at "
            f"{env_path}. USER_DATA_PATH must be the workspace root "
            f"(parent of <model>/<ts>/ per-session subdirs); runtime/ "
            f"is workspace-shared, not per-session. Either "
            f"(a) re-run inference_optimizer/scripts/install.sh under "
            f"USER_DATA_PATH={os.environ.get('USER_DATA_PATH','?')}, "
            f"(b) set $KERNEL_AGENT_ENV to point at an existing file, or "
            f"(c) set $HYPERLOOM_KERNEL_AGENT_ROOT directly to skip this "
            f"fallback entirely. Aborting now (was: silently warning and "
            f"letting trace_analyze fail 10h in).",
            file=sys.stderr,
        )
        sys.exit(2)
    loaded = 0
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(
            f"Preflight: ERROR — failed to read {env_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if "HYPERLOOM_KERNEL_AGENT_ROOT" not in os.environ:
        print(
            f"Preflight: ERROR — sourced {env_path} ({loaded} vars) but "
            f"HYPERLOOM_KERNEL_AGENT_ROOT is still unset. The env file is "
            f"malformed or stale. Re-run inference_optimizer/scripts/"
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
    """
    candidates = (
        ("claude_agent_sdk", "claude-agent-sdk>=0.1.65"),
        ("openai",           "openai>=1.50"),
        ("httpx",             "httpx>=0.27"),
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
            [python_exe, "-m", "pip", "install", "--quiet",
             *pip_extra, pip_spec],
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
            capture_output=True, text=True, timeout=5,
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
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        print(
            f"Preflight: WARNING — /dev/shm has {free_gb:.1f} GiB free of "
            f"{total_gb:.1f} GiB total (< 16 GiB threshold). vLLM IPC + "
            f"NCCL shm segments may collide with stale entries; if the "
            f"first server launch hangs >5min, clear /dev/shm/{{vllm,nccl,cuda}}*"
        )


_TRACELENS_REQUIRED_CLIS: tuple[str, ...] = (
    "TraceLens_generate_perf_report_pytorch_inference",
)


def _check_tracelens_cli() -> None:
    """Hard-gate TraceLens CLI presence — abort before Coordinator starts (SKILL IR-2).

    Pod-local /opt/venv/bin/TraceLens_* console_scripts don't persist across pod restarts, so install.sh
    must run before every launch (carve-out: --resume in the same shell). Fail-fast beats a delayed
    tracelens_cli_missing strike at tick ~6 after baseline burned setup time.
    """
    missing = [
        name for name in _TRACELENS_REQUIRED_CLIS
        if shutil.which(name) is None
    ]
    if not missing:
        return
    session_dir = str(_workspace_root_resolve())
    print(
        f"ERROR: TraceLens CLI(s) not on PATH: {missing}. The pod-local "
        f"/opt/venv/bin/TraceLens_* console_scripts are installed by "
        f"kernel-agent/scripts/install.sh (chained from "
        f"inference_optimizer/scripts/install.sh) and do NOT persist "
        f"across pod restarts. SKILL IR-2 requires running install.sh "
        f"before every launch (carve-out applies only to --resume in "
        f"the same shell that earlier ran install.sh). Re-run:\n"
        f"  bash $REPO_ROOT/inference_optimizer/scripts/install.sh\n"
        f"  . {session_dir}/runtime/kernel-agent.env.sh\n"
        f"then retry `inference_optimizer optimize`. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_node_claude_cli() -> None:
    """WARN-only presence check for bundled agent CLIs (node/claude/codex) and ``@cursor/sdk``.

    SDKs fall back to direct HTTP when CLIs are absent, so this is informational. @cursor/sdk is a Node
    library probed via ``require.resolve`` against ``$(npm root -g)`` since ``shutil.which`` would miss it.
    """
    missing = [t for t in ("node", "claude", "codex") if shutil.which(t) is None]
    if missing:
        print(
            f"Preflight: WARNING — CLI(s) not on PATH: {missing}. "
            f"ClaudeBackend / CodexBackend may fall back to direct HTTP. "
            f"Run kernel-agent/scripts/install.sh to bring them in."
        )
    # @cursor/sdk presence — probe via Node since it's a library, not a CLI.
    if shutil.which("node") is not None and shutil.which("npm") is not None:
        try:
            npm_root = subprocess.run(
                ["npm", "root", "-g"], capture_output=True, text=True, timeout=10,
            )
            global_modules = (npm_root.stdout or "").strip()
            probe = subprocess.run(
                ["node", "-e", "require.resolve('@cursor/sdk')"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "NODE_PATH": global_modules} if global_modules else None,
            )
            if probe.returncode != 0:
                print(
                    "Preflight: WARNING — @cursor/sdk not resolvable; cursor "
                    "backend will fail to start. Run kernel-agent/scripts/"
                    "install.sh to install it globally via npm."
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            print(
                "Preflight: WARNING — could not probe @cursor/sdk presence; "
                "cursor backend may be unavailable."
            )


def _emit_preflight_diagnostics(
    *,
    magpie_python: str,
    anthropic_base_url: str | None,
    args: argparse.Namespace | None = None,
) -> None:
    """One canonical, grep-friendly diagnostics block at the end of preflight."""
    from .orchestrator.action_executors.baseline import (
        BASELINE_COLD_START_TIMEOUT_SEC,
        BASELINE_DEFAULT_TIMEOUT_SEC,
        _probe_aiter_jit_cache,
    )
    from .paths import asset_root

    probe = _probe_aiter_jit_cache()
    cold_cap = os.environ.get(
        "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
        str(BASELINE_COLD_START_TIMEOUT_SEC),
    )
    if probe["probe_status"] == "found":
        kind = "COLD" if probe["is_cold"] else "WARM"
        cache_line = (
            f"{probe['kernel_count']} .so / {probe['size_mb']} MB "
            f"({kind}) at {probe['path']}"
        )
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
    print(
        f"  INFERENCEX_PATH     = "
        f"{os.environ.get('INFERENCEX_PATH', '<unset>')}"
    )
    print(f"  aiter jit cache     = {cache_line}")
    print(f"  cold_start_timeout  = {cold_cap}s")
    print(f"  warm_timeout        = {BASELINE_DEFAULT_TIMEOUT_SEC}s")
    if anthropic_base_url:
        print(f"  ANTHROPIC_BASE_URL  = {anthropic_base_url} (direct to gateway)")
    else:
        print(
            "  ANTHROPIC_BASE_URL  = <unset> — OPENAI_BASE_URL missing; "
            "Claude SDK will fail"
        )
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

    # Issue-H: surface Cortex KB offline-queue state; dead-letter pile-up signals a cold-start session.
    try:
        _print_cortex_kb_queue_status()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  cortex_kb_queue     = <probe_failed: {exc!r}>")


def _print_cortex_kb_queue_status() -> None:
    """Emit a one-line summary of the Cortex KB offline NDJSON queue (dead-letter = permanent-reject signal)."""
    from .session_paths import (
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
    print(
        f"  cortex_kb_queue     = pending={p_n} dead_letter={d_n} "
        f"flushed={f_n} (root={pending.parent})"
    )
    if d_n > 0:
        print(
            f"                        ⚠ {d_n} dead-letter row(s) — "
            f"prior KB writes permanently rejected (4xx schema). "
            f"Specialists for affected anchors will start cold "
            f"(no priors). See {dead}."
        )


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | None:
    """Probe ``<base_url>/models`` with retry (gateway flakes); return set of model ids or None.

    TLS verification is on by default; ``INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1`` skips it (warns).
    """
    import time

    if not base_url:
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # _ensure_python_sdks should have installed httpx; return None so the caller decides.
        print(
            "Preflight: WARNING — httpx not importable, skipping catalog "
            "probe. _ensure_python_sdks should have installed it."
        )
        return None

    insecure = os.environ.get(
        "INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE", "",
    ).strip().lower() in ("1", "true", "yes")
    if insecure:
        print(
            "Preflight: WARNING — INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1 "
            "is set; catalog probe will skip TLS verification while sending "
            "an Authorization: Bearer header. Use only against trusted internal "
            "gateways with self-signed certs."
        )
        try:
            import urllib3  # type: ignore[import-not-found]
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass

    probe_url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    delays = (0.0, *_CATALOG_RETRY_DELAYS_SEC)
    last_err: str = ""
    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.get(
                probe_url,
                headers=headers,
                timeout=_CATALOG_REQUEST_TIMEOUT_SEC,
                verify=not insecure,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"failed: {last_err}"
            )
            continue
        if resp.status_code != 200:
            last_err = (
                f"HTTP {resp.status_code}: "
                f"{(resp.text or '')[:200]}"
            )
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"got {last_err}"
            )
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_err = f"JSON decode: {exc}"
            print(
                f"Preflight: catalog probe attempt {i + 1}/{len(delays)} "
                f"returned non-JSON: {last_err}"
            )
            continue
        ids = {
            m["id"] for m in data.get("data") or []
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        }
        if not ids:
            last_err = "empty data[]"
            continue
        return ids

    print(
        f"Preflight: catalog probe exhausted {len(delays)} attempts "
        f"({last_err}); cannot validate model availability"
    )
    return None


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
) -> set[str] | None:
    """Hard-gate Claude model selection (must be in _CLAUDE_ALLOWED_MODELS); mutates ``args.claude_model``.

    Probes the gateway catalog (retries); falls back 4-7→4-6 with a WARN, else sys.exit(2). Returns the
    catalog id set on success (reused by the codex smoke-test).
    """
    chosen = (args.claude_model or "").strip()
    if chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} is not allowed. "
            f"Orchestration model must be one of {list(_CLAUDE_ALLOWED_MODELS)} "
            f"(preferred: {_CLAUDE_PREFERRED_MODEL}, "
            f"fallback: {_CLAUDE_FALLBACK_MODEL}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Catalog probe GETs <base>/models; INFERENCE_OPTIMIZER_CATALOG_PROBE_URL overrides the host.
    base_url = (
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "")
    )
    if not base_url and resolved_urls is not None:
        base_url = resolved_urls[1]

    api_key = (
        os.environ.get("SAFE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )

    catalog_ids = _probe_llm_catalog(base_url=base_url, api_key=api_key)
    if catalog_ids is None:
        print(
            "ERROR: gateway catalog unreachable after retries; cannot "
            "verify Claude model availability. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if chosen in catalog_ids:
        print(f"Preflight: Claude model {chosen!r} confirmed in gateway catalog")
        return catalog_ids

    if _CLAUDE_FALLBACK_MODEL in catalog_ids:
        print(
            f"Preflight: WARNING — {chosen!r} not in gateway catalog; "
            f"falling back to {_CLAUDE_FALLBACK_MODEL!r}"
        )
        args.claude_model = _CLAUDE_FALLBACK_MODEL
        return catalog_ids

    print(
        f"ERROR: neither {_CLAUDE_PREFERRED_MODEL!r} nor "
        f"{_CLAUDE_FALLBACK_MODEL!r} present in gateway catalog "
        f"(catalog has {sorted(m for m in catalog_ids if m.startswith('claude-'))}). "
        f"Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    catalog_ids: set[str] | None,
) -> None:
    """WARN-only catalog check for ``--codex-model`` (no hard gate); flags typos before Coordinator starts."""
    if catalog_ids is None:
        return
    # Codex is needed by the Kernel agent (kernel-codex on) and the critic-agent review path.
    critic_uses_codex = args.critic_backend == "agent"
    needs_codex = critic_uses_codex or (
        args.kernel_codex and not getattr(args, "no_kernel", False)
    )
    if not needs_codex:
        return
    chosen = (args.codex_model or "").strip()
    if chosen in catalog_ids:
        print(f"Preflight: Codex model {chosen!r} confirmed in gateway catalog")
        return
    print(
        f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog "
        f"({sorted(m for m in catalog_ids if m.startswith('gpt-'))}); "
        f"CodexBackend will fail at first turn. Pass --codex-model with a "
        f"value in the catalog or use --critic-mock / --kernel-claude to "
        f"avoid the Codex path entirely."
    )


# InferenceX clone defaults — kept in sync with
# inference_optimizer/scripts/install.sh (INFERENCEX_REPO / INFERENCEX_REF).
_INFERENCEX_REPO_DEFAULT = "https://github.com/SemiAnalysisAI/InferenceX.git"
_INFERENCEX_REF_DEFAULT = "2035a2117ad22403376359be0064dfa2c078c59b"


def _inferencex_checkout_ok(path: Path | str) -> bool:
    """True when ``path`` is a usable InferenceX checkout, not a stub.

    A bare ``is_dir()`` check accepts a half-cloned dir left behind by a
    ``git init`` that then failed to fetch/checkout. Magpie sources
    ``benchmarks/benchmark_lib.sh`` at runtime, so require that file to
    exist — a complete checkout always has it, a stub never does.
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
                check=True, timeout=600,
            )
            subprocess.run(
                ["git", "-C", dest_str, "checkout", "-q", "FETCH_HEAD"],
                check=True, timeout=120,
            )
        else:
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", "--branch", ref, repo, dest_str],
                check=True, timeout=600,
            )
        if not _inferencex_checkout_ok(dest):
            raise OSError(
                f"clone reported success but {dest_str} is missing "
                "benchmarks/benchmark_lib.sh"
            )
        return dest_str
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("InferenceX clone into %s failed: %s", dest_str, exc)
        shutil.rmtree(dest, ignore_errors=True)
        return None


def _preflight(
    args: argparse.Namespace | None = None,
) -> tuple[str, str] | None:
    """Auto-install missing runtime deps and export auth aliases.

    Credentials fallback → auth aliases → SDK install → ANTHROPIC_BASE_URL resolve + ~/.claude reset →
    ROCm hygiene → ray/Magpie/InferenceX install → CLI presence checks → diagnostics. Returns
    ``(anthropic_base_url, openai_base_url)`` or ``None`` when ``OPENAI_BASE_URL`` is missing.
    """
    _load_dotenv_fallback()
    _load_kernel_agent_env_fallback()

    # Fail fast on missing credentials after the fallback loaders, before any cycle-burning work.
    _validate_credentials()

    # --- Auth alias export ---
    safe_key = os.environ.get("SAFE_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if safe_key:
        for alias in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                      "ANTHROPIC_API_KEY", "OOB_API_KEY", "GEAK_API_KEY",
                      "LLM_API_KEY", "AMD_LLM_API_KEY"):
            if os.environ.get(alias) != safe_key:
                os.environ[alias] = safe_key
                print(f"Preflight: refreshed {alias} from SAFE_API_KEY")
    # OOB / GEAK / LLM_API_BASE inherit the upstream URL verbatim.
    if base_url:
        for alias in ("OOB_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE"):
            if os.environ.get(alias) != base_url:
                prev = os.environ.get(alias, "")
                os.environ[alias] = base_url
                print(
                    f"Preflight: {alias} {prev or '<unset>'} -> {base_url} "
                    f"(direct to gateway)"
                )

    # --- Resolve install interpreters ---
    from .orchestrator.action_executors._grid_runner import _resolve_magpie_python
    magpie_python = _resolve_magpie_python()

    # Outside a venv, add --break-system-packages so pip installs on bare-metal Debian/Ubuntu.
    pip_extra: list[str] = []
    if not (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        pip_extra = ["--break-system-packages"]

    # --- Python SDK auto-install (claude-agent-sdk / openai / httpx) ---
    # Must precede Coordinator import (ClaudeBackend lazy-imports the SDK); sys.executable matches imports.
    _ensure_python_sdks(sys.executable, pip_extra)

    # --- Resolve ANTHROPIC_BASE_URL + reset ~/.claude/config.json ---
    # Force-override both URL vars so stale 127.0.0.1:4002 leftovers can't reach the CLIs.
    resolved_urls: tuple[str, str] | None = None
    if base_url:
        anthropic_url = _derive_anthropic_base_url(base_url)
        orig_anthropic = os.environ.get("ANTHROPIC_BASE_URL", "")
        orig_openai = os.environ.get("OPENAI_BASE_URL", "")
        for var, want, prev in (
            ("ANTHROPIC_BASE_URL", anthropic_url, orig_anthropic),
            ("OPENAI_BASE_URL", base_url, orig_openai),
        ):
            if os.environ.get(var) != want:
                os.environ[var] = want
                print(
                    f"Preflight: {var} {prev or '<unset>'} -> {want} "
                    f"(direct to gateway)"
                )
        _reset_claude_config_to_upstream(safe_key, anthropic_url)
        resolved_urls = (anthropic_url, base_url)
    else:
        print(
            "Preflight: WARNING — OPENAI_BASE_URL unset; "
            "Claude/Codex SDKs will fail at first call"
        )

    # --- ROCm env hygiene + GPU/shm sanity (defensive WARN-only) ---
    _unset_hip_visible_devices()
    _check_gpu_visibility()
    _check_shm_disk()

    # --- Runtime dep install ---
    # 1. Ray — needed by Magpie for task scheduling even without kernel-agent.
    if shutil.which("ray") is None:
        print("Preflight: ray not found, installing ray[default]==2.44.1 + click<8.3.0 ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet",
             *pip_extra, "ray[default]==2.44.1", "click<8.3.0"],
            check=True,
        )
        print("Preflight: ray installed OK")

    # 2. Magpie — the benchmark engine all executors shell out to ($MAGPIE_DIR override; auto-clones if missing).
    check = subprocess.run(
        [magpie_python, "-c", "import Magpie"],
        capture_output=True,
    )
    if check.returncode != 0:
        magpie_env = os.environ.get("MAGPIE_DIR")
        magpie_env_explicit = bool(magpie_env)
        if magpie_env:
            magpie_dir = Path(magpie_env)
        else:
            from .paths import magpie_dir as _magpie_default
            magpie_dir = _magpie_default(_session_dir_resolve())
        magpie_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (magpie_dir / "setup.py").exists() and not (magpie_dir / "pyproject.toml").exists():
            # Refuse-to-clobber: don't clone Magpie main over an explicit $MAGPIE_DIR (would destroy local work).
            if magpie_env_explicit:
                print(
                    f"Preflight: ERROR — $MAGPIE_DIR={magpie_dir} has no "
                    f"setup.py/pyproject.toml; refusing to clone Magpie "
                    f"main on top of an operator-supplied path. Fix the "
                    f"env or unset $MAGPIE_DIR to fall back to the "
                    f"session-default location.",
                    file=sys.stderr,
                )
                raise FileNotFoundError(
                    f"$MAGPIE_DIR={magpie_dir} is not a valid Magpie checkout"
                )
            print(f"Preflight: Magpie not importable and not found at {magpie_dir}; cloning ...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/AMD-AGI/Magpie.git", str(magpie_dir)],
                check=True,
            )
        print(f"Preflight: installing Magpie from {magpie_dir} ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet",
             *pip_extra, "-e", str(magpie_dir)],
            check=True,
        )
        print("Preflight: Magpie installed OK")

    # 3. InferenceX — required for GSM8K accuracy eval; lm-eval deps auto-install at runtime via benchmark_lib.sh.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if not inferencex_path:
        from .paths import (
            magpie_dir as _magpie_default,
            open_source_root as _open_source_default,
        )
        open_source_root = _open_source_default()
        magpie_root = (
            Path(os.environ["MAGPIE_DIR"])
            if os.environ.get("MAGPIE_DIR")
            else _magpie_default(_session_dir_resolve())
        )
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
        from .paths import open_source_root as _open_source_default
        dest = _open_source_default() / "InferenceX"
        print(f"Preflight: InferenceX not found; cloning into {dest} ...")
        inferencex_path = _clone_inferencex(dest)
        if not (inferencex_path and _inferencex_checkout_ok(inferencex_path)):
            print(
                "Preflight: ERROR — InferenceX checkout missing and clone "
                "failed. baseline cannot run without it. Set INFERENCEX_PATH "
                "to a writable checkout or re-run "
                "inference_optimizer/scripts/install.sh.",
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

    # --- TraceLens CLI presence (HARD-FAIL; SKILL Step 2 step 8.5) ---
    # Catches launchers that skip install.sh, else missing-CLI only surfaces at the tick ~6 robustness probe.
    _check_tracelens_cli()

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
    script = (
        Path(__file__).resolve().parent / "scripts" / "preflight_kb.sh"
    )
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
            ["bash", str(script)], env=env,
            check=False, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Script died — treat both branches as unreachable so soft-degrade kicks in.
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False, "pr_reachable": False,
            "kb_skipped": explicit_kb, "pr_skipped": explicit_pr,
        }
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("IR-3 marker unreadable: %s", exc)
            marker = {
                "kb_reachable": False, "pr_reachable": False,
                "kb_skipped": explicit_kb, "pr_skipped": explicit_pr,
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


# Default critic backend ("agent" since Step D); override via env or --critic-mock/--critic-agent.
DEFAULT_CRITIC_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND", "agent",
)
_VALID_CRITIC_BACKENDS = ("mock", "agent")


def _resolve_critic_choice(args: argparse.Namespace) -> str:
    """Resolve the active critic backend choice (arg → DEFAULT_CRITIC_BACKEND); hard-fails on invalid."""
    chosen = args.critic_backend
    if chosen is None:
        chosen = DEFAULT_CRITIC_BACKEND
    if chosen not in _VALID_CRITIC_BACKENDS:
        print(
            f"ERROR: critic backend {chosen!r} not in {_VALID_CRITIC_BACKENDS!r} "
            f"(set by --critic-mock / --critic-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    return chosen


# Default robustness backend ("agent"); force heartbeat-only mock via --robustness-mock or env.
DEFAULT_ROBUSTNESS_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND", "agent",
)
_VALID_ROBUSTNESS_BACKENDS = ("mock", "agent")


def _resolve_robustness_choice(args: argparse.Namespace) -> str:
    """Resolve the active robustness backend choice (arg → DEFAULT_ROBUSTNESS_BACKEND); hard-fails on invalid.

    Multi-node policy: on ``nodes>=2`` the agent's LocalProbe targets sandbox-local resources that live in
    separate pods (HIGH false positives). Keep ``agent`` only when a robustness-server is configured; else
    auto-downgrade to ``mock`` (explicit --robustness-agent gets a WARN).
    """
    chosen = getattr(args, "robustness_backend", None)
    explicit = chosen is not None
    if chosen is None:
        chosen = DEFAULT_ROBUSTNESS_BACKEND
    if chosen not in _VALID_ROBUSTNESS_BACKENDS:
        print(
            f"ERROR: robustness backend {chosen!r} not in "
            f"{_VALID_ROBUSTNESS_BACKENDS!r} (set by --robustness-mock / "
            f"--robustness-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    nodes = int(getattr(args, "nodes", 1) or 1)
    if nodes >= 2 and chosen == "agent" and not _robustness_server_configured(args):
        if explicit:
            print(
                f"WARN: --robustness-agent selected but nodes={nodes} and "
                f"no robustness-server configured — the agent's LocalProbe "
                f"family targets sandbox-local resources (ray, inference "
                f"server, GPU, ...) that all live in separate pods on "
                f"multi-node and surface as HIGH false positives. "
                f"Auto-downgrading to --robustness-mock; configure "
                f"--robustness-server-url / ROBUSTNESS_SERVER_URL to keep "
                f"the agent backend, or pass --robustness-mock explicitly "
                f"to suppress this warning. See "
                f"inference_optimizer/multi_node/SKILL.md "
                f"(Robustness limitation in multi-node mode).",
                file=sys.stderr,
            )
        chosen = "mock"
    return chosen


def _reset_state_file(session_dir: Path) -> None:
    """Back up ``state.json`` to ``state.json.preReset.<unix_ts>`` and start fresh (Cortex KB untouched)."""
    state_path = session_dir / "state.json"
    if not state_path.exists():
        return
    import time as _time
    ts = int(_time.time())
    backup_path = session_dir / f"state.json.preReset.{ts}"
    try:
        state_path.replace(backup_path)
    except OSError as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "v0.8 §3.10 --reset-state: could not move %s → %s: %s",
            state_path, backup_path, exc,
        )
        return
    import logging as _logging
    _logging.getLogger(__name__).info(
        "v0.8 §3.10 --reset-state: backed up state.json to %s; "
        "session starts blank.", backup_path.name,
    )


def _gc_old_profile_traces(
    root: str | None = None,
    retention_days: int = 7,
    keep: str | None = None,
) -> None:
    """Best-effort GC of stale per-RayJob profile-trace dirs older than ``retention_days`` (``keep`` name-guarded).

    Never blocks startup (errors swallowed). Env knobs: HYPERLOOM_MN_TRACE_RETENTION_DAYS,
    HYPERLOOM_MN_TRACE_GC_DISABLE.
    """
    if os.environ.get("HYPERLOOM_MN_TRACE_GC_DISABLE", "").strip() in (
        "1", "true", "yes",
    ):
        return
    try:
        retention_days = int(
            os.environ.get("HYPERLOOM_MN_TRACE_RETENTION_DAYS") or retention_days
        )
    except ValueError:
        retention_days = 7
    base = Path(root) if root is not None else mn_profile_trace_root()
    if not base.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    keep_name = Path(keep).name if keep else ""
    removed = 0
    kept = 0
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if keep_name and child.name == keep_name:
                kept += 1
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                kept += 1
                continue
            try:
                shutil.rmtree(child)
                removed += 1
            except OSError as exc:
                print(
                    f"WARN multi-node GC: failed to rm {child}: {exc}",
                    file=sys.stderr,
                )
    except OSError as exc:
        print(f"WARN multi-node GC: scan failed under {base}: {exc}", file=sys.stderr)
        return
    if removed or kept:
        print(
            f"multi-node: GC profile-traces removed={removed} kept={kept} "
            f"retention={retention_days}d root={base}"
        )


def _provision_multi_node_rayjob_stack(args: argparse.Namespace) -> None:
    """Create/reuse the SaFE RayJob stack for a multi-node run.

    No-op when ``--nodes < 2``. Otherwise resolves the RayJob container image
    (CLI flag → env → prior state file), creates or reuses the RayJob, runs the
    one-time bootstrap if it hasn't run yet, exports ``RAY_ADDRESS`` for
    kernel-agent Ray tasks, sets ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` to a
    cluster-shared trace directory namespaced by ``rayjob_id`` (GC'ing older
    sibling dirs), and replays previously-applied kernel patches onto the
    (possibly fresh) pods.

    Args:
        args (argparse.Namespace): Parsed ``optimize`` arguments (reads
            ``nodes``, ``rayjob_image``, ``rayjob_gpus_per_node``,
            ``rayjob_extra_env``).

    Raises:
        SystemExit: With code 2 when ``--nodes >= 2`` but no RayJob image is
            configured, or with the create/bootstrap return code on failure.
    """
    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    from .multi_node.cli import cmd_bootstrap, cmd_create_rayjob, _load_state
    from .orchestrator.action_executors._multi_node_env import export_ray_address_to_os

    state_path = Path(os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json"))
    image = (
        (getattr(args, "rayjob_image", None) or "").strip()
        or os.environ.get("INFERENCE_OPTIMIZER_RAYJOB_IMAGE", "").strip()
    )
    if not image and state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            image = str((prior.get("last_create_request") or {}).get("image") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            image = ""
    if not image:
        print(
            "ERROR: --nodes >= 2 requires a RayJob container image. Pass "
            "--rayjob-image <harbor/...> or set INFERENCE_OPTIMIZER_RAYJOB_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "rayjob_gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    # Forward agent-supplied prompt env verbatim; no-op on RayJob reuse (see multi_node/SKILL.md).
    rayjob_extra_env = list(getattr(args, "rayjob_extra_env", None) or [])

    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=96,
        mem_per_node=1024,
        ephemeral_per_node=400,
        display_name=None,
        description=None,
        owner_id=None,
        extra_env=rayjob_extra_env,
        extra_label=[],
        no_wait=False,
        recreate=False,
        poll_interval=6,
        poll_timeout=int(
            os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110
        ),
    )
    rc = cmd_create_rayjob(ns_create)
    if rc != 0:
        sys.exit(rc)

    state = _load_state()
    if not state.get("last_bootstrap_submission_id"):
        ns_boot = argparse.Namespace(
            script=None,
            force=False,
            print_logs=False,
            poll_interval=6,
            poll_timeout=int(
                os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110
            ),
        )
        rc_boot = cmd_bootstrap(ns_boot)
        if rc_boot != 0:
            sys.exit(rc_boot)

    export_ray_address_to_os()
    ra = os.environ.get("RAY_ADDRESS", "")
    if ra:
        print(f"multi-node: exported RAY_ADDRESS={ra} for kernel-agent Ray tasks")

    # Multi-node: server pods must write torch traces to a sandbox-readable wekafs path, namespaced by rayjob_id.
    state_after = _load_state()
    rid = (state_after.get("rayjob_id") or "").strip()
    if rid:
        # Anchor torch-profile shared root on $USER_DATA_PATH so sandbox and RayJob pods see the same path.
        trace_root_path = mn_profile_trace_root() / rid / "torch_trace"
        trace_root = str(trace_root_path)
        try:
            trace_root_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"WARN multi-node: cannot mkdir {trace_root}: {exc}; "
                f"server traces will fall back to per-pod /tmp",
                file=sys.stderr,
            )
        else:
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = trace_root
            print(
                f"multi-node: exported HYPERLOOM_MN_PROFILE_TRACE_DIR={trace_root}"
            )
            # Best-effort GC of older sibling RayJob trace dirs (active rayjob_id name-guarded).
            _gc_old_profile_traces(keep=rid)

    # RayJob recreate path: replay promoted patches from optimization_stack since fresh pods lost them.
    # Best-effort; failures degrade to warnings (orchestrator re-runs kernel-agent on missing speedups).
    _replay_kernel_patches_for_multi_node(args)


def _replay_kernel_patches_for_multi_node(args: argparse.Namespace) -> None:
    """Replay every applied kernel-agent patch (manifest status=applied + multinode block) onto RayJob pods.

    Idempotent ``apply-patch`` fan-out, run only when ``--nodes>=2``. Best-effort: per-patch failures warn.
    """
    nodes = max(1, int(getattr(args, "nodes", 1) or 1))
    if nodes < 2:
        return
    session_dir = _session_dir_resolve()
    workspace_root = session_dir / "kernel-agent-workspace"
    if not workspace_root.is_dir():
        return

    manifests: list[Path] = sorted(workspace_root.rglob("manifest.json"))
    if not manifests:
        return

    replayed = 0
    skipped = 0
    failed = 0
    for mpath in manifests:
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARN multi-node patch replay: skipping unreadable "
                f"manifest {mpath}: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        if str(data.get("status", "")).lower() != "applied":
            continue
        mn = data.get("multinode") or {}
        if not mn:
            continue
        target_file = data.get("target_file") or ""
        patch_path = data.get("patch_path") or ""
        kernel_id = data.get("kernel_id") or ""
        backup_dir_on_pod = mn.get("backup_dir_on_pod") or "/var/kernel_patch_backups"
        if not target_file or not patch_path:
            skipped += 1
            continue
        if not Path(patch_path).is_file():
            print(
                f"WARN multi-node patch replay: source patch missing for "
                f"{target_file} (manifest={mpath} patch_path={patch_path}); "
                f"skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue
        cmd = [
            sys.executable, "-m", "inference_optimizer.multi_node",
            "apply-patch",
            "--patch-file", str(patch_path),
            "--target-path", str(target_file),
            "--backup-dir", str(backup_dir_on_pod),
            "--kernel-id", str(kernel_id),
        ]
        print(
            f"multi-node patch replay: target={target_file} kernel_id={kernel_id!r} "
            f"(from {mpath})",
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            failed += 1
            print(
                f"WARN multi-node patch replay failed for {target_file} "
                f"rc={proc.returncode} stderr={(proc.stderr or '')[-1000:]!r}",
                file=sys.stderr,
            )
            continue
        replayed += 1
    if replayed or failed or skipped:
        print(
            f"multi-node patch replay: applied={replayed} "
            f"skipped={skipped} failed={failed} "
            f"(scanned {len(manifests)} manifest(s) under {workspace_root})"
        )


async def _run_quantization_prelude(args: argparse.Namespace) -> None:
    """Run the quantization-agent once before the optimization loop.

    No-op unless ``--quantize "<prompt>"`` was passed. When set, this drives
    AMD Quark PTQ from the prompt via the ``quantization_request_handlers``
    adapter, then rewrites ``args.model`` (+ ``$MODEL_PATH``) to the exported
    quantized model so every downstream phase (baseline / profile / sweep /
    kernel) optimizes the quantized model instead of the source.

    Contract:
      * Skipped on ``--resume`` (a resumed session already has its model
        pinned in the manifest; re-quantizing would diverge from it).
      * On a failed/blocked quantization the process exits with code 3 —
        we must not silently fall through and optimize the un-quantized
        source model when the user explicitly asked for quantization.
      * On a scheme/GPU mismatch (e.g. an MI355X-only scheme on an mi300x
        target), the structured ``--quantize-scheme`` path reports the error
        and *skips* quantization, then continues optimizing the un-quantized
        model. The mismatch is a config error caught before any Quark work
        runs, not a mid-run failure, so the run proceeds rather than aborting.
        The skip is made **detectable** so a launcher / UI never mistakes the
        run for quantized: a ``QUANTIZATION_SKIPPED:`` marker line on stdout
        plus the ``$HYPERLOOM_QUANTIZATION_SKIPPED`` env var (set to the reason).
    """
    # Free-text --quantize wins; otherwise resolve the structured
    # --quantize-scheme enum (the UI/backend path) to a prompt.
    prompt = getattr(args, "quantize", None)
    if not prompt:
        from .orchestrator.quantization_schemes import (
            SchemeNotSupportedError,
            resolve_scheme_prompt,
            validate_scheme,
        )

        scheme = getattr(args, "quantize_scheme", None)
        # Constrain the scheme by the target GPU. The real GPU is probed later;
        # use the --gpu-type / $GPU_TYPE hint here (empty => no enforcement).
        gpu_hint = (
            getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")
        ).strip().lower()
        try:
            validate_scheme(scheme, gpu_hint)
        except SchemeNotSupportedError as exc:
            # Pre-flight config error (caught before any Quark work): per the
            # documented contract we SKIP quantization and continue on the
            # un-quantized model rather than hard-stopping. Make the skip
            # explicit + machine-detectable (stdout marker + env var) so a
            # launcher / UI surfaces "requested quantization was skipped"
            # instead of silently believing the run is quantized.
            reason = str(exc)
            os.environ["HYPERLOOM_QUANTIZATION_SKIPPED"] = reason
            print(
                f"QUANTIZATION_SKIPPED: {reason}; continuing optimization on the "
                "un-quantized model. Pick a scheme supported by this GPU TYPE "
                "(or change GPU_TYPE) to actually quantize."
            )
            print(f"ERROR: quantization skipped — {reason}", file=sys.stderr)
            return
        prompt = resolve_scheme_prompt(scheme)
    if not prompt:
        return
    if getattr(args, "resume", False):
        print("Quantization prelude: skipped (--resume); using model from manifest.")
        return

    from .paths import workspace_root

    source_model = str(args.model)
    workspace = workspace_root() / "quantization" / Path(source_model).name
    workspace.mkdir(parents=True, exist_ok=True)

    # Adapter lives in the orchestrator package; lazy-import so the CLI keeps
    # importing cleanly even in environments without the quantization deps.
    # _run_optimize already runs under asyncio.run, so await the async form
    # directly (the sync wrapper would call asyncio.run inside a live loop).
    from .orchestrator.quantization_request_handlers import (
        run_quantization_prelude_async,
    )

    quantized_model_dir = await run_quantization_prelude_async(
        prompt=prompt,
        source_model=source_model,
        workspace=workspace,
    )

    args.model = Path(quantized_model_dir)
    os.environ["MODEL_PATH"] = str(quantized_model_dir)
    print(f"Quantization prelude: model -> {quantized_model_dir}")


def _argv_has_option(argv: list[str], option: str) -> bool:
    """Report whether ``argv`` explicitly carries a given option.

    Matches both the bare flag (``--tp``) and the ``=``-joined form
    (``--tp=8``).

    Args:
        argv (list[str]): The argument vector to scan.
        option (str): The long-option flag to look for (e.g. ``"--tp"``).

    Returns:
        bool: ``True`` when the option appears in ``argv``, else ``False``.
    """
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def _export_workload_envs_for_optimize(
    args: argparse.Namespace,
    *,
    nodes_resolved: int,
    tp_resolved: int,
    ep_resolved: int,
    argv: list[str] | None = None,
) -> None:
    """Mirror explicit workload CLI flags (--tp/--conc/--ep) into env so executors' Magpie YAMLs honor them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if nodes_resolved >= 2 or _argv_has_option(argv, "--tp"):
        os.environ["TP"] = str(tp_resolved)
    if nodes_resolved >= 2 or _argv_has_option(argv, "--conc"):
        os.environ["CONC"] = str(max(1, int(getattr(args, "conc", 8) or 8)))
    if nodes_resolved >= 2 or _argv_has_option(argv, "--ep"):
        os.environ["EP"] = str(ep_resolved)


async def _run_optimize(args: argparse.Namespace) -> int:
    # Surface --nodes (CLI flag wins) before _preflight runs.
    nodes_resolved = max(1, int(args.nodes))
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    # Resolve gpus_per_node with the same CLI > env > 8 chain _provision_multi_node_rayjob_stack uses.
    gpn_attr = getattr(args, "rayjob_gpus_per_node", None)
    if gpn_attr is not None:
        gpus_per_node_resolved = int(gpn_attr)
    else:
        try:
            gpus_per_node_resolved = int(
                os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8,
            )
        except ValueError:
            gpus_per_node_resolved = 8
    total_gpus = nodes_resolved * gpus_per_node_resolved

    # Topology sanity gates — multi-node only (nodes>=2); fail fast vs a cryptic launcher crash mid-cold-start.
    if nodes_resolved >= 2:
        # Gate 1: total cluster GPUs (nodes*gpus_per_node) must hold the model's TP shards.
        if total_gpus < tp_resolved:
            print(
                f"ERROR: TP={tp_resolved} exceeds total GPU count "
                f"({nodes_resolved} nodes * {gpus_per_node_resolved} "
                f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
                "--nodes, or use a larger --rayjob-gpus-per-node pod "
                "template.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Gate 2: EP cannot exceed TP (can't place more expert shards than ranks); fail before bootstrap.
        if ep_resolved > tp_resolved:
            print(
                f"ERROR: EP={ep_resolved} > TP={tp_resolved}. Expert-parallel "
                "size must be <= tensor-parallel size. Either lower --ep or "
                "raise --tp.",
                file=sys.stderr,
            )
            sys.exit(2)

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    # Re-export $TP/$CONC/$EP when explicitly supplied (always for multi-node); skip defaults in single-node.
    _export_workload_envs_for_optimize(
        args,
        nodes_resolved=nodes_resolved,
        tp_resolved=tp_resolved,
        ep_resolved=ep_resolved,
    )
    # User-declared grid skip list; re-export so subprocess executors inherit it (empty clears stale values).
    skip_variants_resolved = (getattr(args, "skip_variants", "") or "").strip()
    os.environ["SKIP_VARIANTS"] = skip_variants_resolved
    # Surface PD_* knobs for executors; empty means "resolve from state.json", pd_mode always exported.
    pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
    if pd_mode == "disaggregated" and nodes_resolved < 2:
        # PD disaggregation needs >=2 nodes (separate prefill + decode pods); fail at parse time.
        print(
            f"ERROR: --pd-mode disaggregated requires --nodes >= 2 "
            f"(got --nodes {nodes_resolved}). PD splits the cluster "
            "into prefill + decode groups; a single pod cannot host "
            "both. Either drop --pd-mode (defaults to colocated) or "
            "raise --nodes.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.environ["PD_MODE"] = pd_mode
    if pd_mode == "disaggregated":
        for cli_attr, env_key in (
            ("pd_prefill_nodes", "PD_PREFILL_NODES"),
            ("pd_decode_nodes", "PD_DECODE_NODES"),
            ("pd_prefill_tp", "PD_PREFILL_TP"),
            ("pd_decode_tp", "PD_DECODE_TP"),
        ):
            v = int(getattr(args, cli_attr, 0) or 0)
            if v > 0:
                os.environ[env_key] = str(v)
        for cli_attr, env_key in (
            ("pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            ("pd_ib_device", "PD_IB_DEVICE"),
        ):
            v = (getattr(args, cli_attr, "") or "").strip()
            if v:
                os.environ[env_key] = v

    await asyncio.to_thread(_provision_multi_node_rayjob_stack, args)

    # Stale aiter JIT lock sweep: killed runs leave locks that block subsequent starts (locks <5min preserved).
    aiter_sweep = _clean_stale_aiter_locks()
    if aiter_sweep["dir"] and aiter_sweep["deleted"]:
        print(
            f"Stale aiter locks cleared: "
            f"dir={aiter_sweep['dir']} "
            f"deleted={aiter_sweep['deleted']} "
            f"skipped_fresh={aiter_sweep['skipped_fresh']} "
            f"errors={aiter_sweep['errors']}"
        )

    resolved_urls = _preflight(args)

    # Hard-gate Claude model before any session work (mutates args.claude_model on fallback; sys.exit(2) on failure).
    catalog_ids = _validate_and_resolve_claude_model(args, resolved_urls)
    _smoke_test_codex_model(args, catalog_ids)

    # `--resume-from <path>` implies `--resume` (operator convenience).
    if args.resume_from and not args.resume:
        args.resume = True

    if args.resume:
        # Resume mode: USER_DATA_PATH stays at workspace level; pick the per-session subdir via --resume-from
        # or auto-pick the latest under <model>/<ts>/ (legacy flat layout fallback). Pin
        # INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR so paths/subprocesses resolve consistently.
        from .paths import (
            ENV_CURRENT_SESSION_DIR, find_latest_per_session_dir,
            workspace_root,
        )
        ws = workspace_root()
        if args.resume_from:
            session_dir = Path(args.resume_from).expanduser().resolve()
            try:
                session_dir.relative_to(ws.resolve())
            except ValueError:
                print(
                    f"ERROR: --resume-from {session_dir!r} is not under "
                    f"$USER_DATA_PATH={ws}. Move USER_DATA_PATH to the "
                    f"workspace root (the parent of the per-session subdirs) "
                    f"and pass the per-session subdir via --resume-from.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not session_dir.is_dir():
                print(
                    f"ERROR: --resume-from {session_dir!r} does not exist.",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            picked = find_latest_per_session_dir()
            if picked is not None:
                session_dir = picked
                print("  --resume: auto-picked latest per-session subdir")
            else:
                # Legacy flat layout — workspace_root itself is the session_dir.
                session_dir = ws
                print(
                    f"  --resume: no per-session subdir found under "
                    f"{ws}/<model>/<ts>/; falling back to flat layout "
                    f"({ws})"
                )
        # Pin before Coordinator/SharedState load so paths/subprocesses inherit the resolved location.
        os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
        # Ensure per-session skeleton exists (idempotent mkdir -p).
        for sub in __import__(
            "inference_optimizer.paths", fromlist=["_SESSION_SKELETON"]
        )._SESSION_SKELETON:
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        try:
            manifest = load_manifest(session_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: --resume failed: {exc}", file=sys.stderr)
            sys.exit(2)
        if not (session_dir / "state.json").exists():
            print(
                f"ERROR: --resume failed: {session_dir}/state.json missing "
                f"(manifest exists but Coordinator never wrote SharedState)",
                file=sys.stderr,
            )
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        prior_stop = state.stop_reason
        print(f"Resuming session: {session_dir}")
        print(f"  manifest.session_id    : {manifest.get('session_id')}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain:.2f}%")
        print(f"  prior current_best    : "
              f"{(state.current_best or {}).get('action')}/"
              f"{(state.current_best or {}).get('tput')}")
        print(f"  prior stop_reason     : {prior_stop or '(none)'}")

        # Re-export session-level env from persisted state so a fresh-shell resume doesn't fall back to YAML defaults.
        if state.model_path:
            os.environ["MODEL_PATH"] = state.model_path
            print(f"  re-exported MODEL_PATH: {state.model_path}")
        if state.framework:
            os.environ["FRAMEWORK"] = state.framework
            print(f"  re-exported FRAMEWORK : {state.framework}")
        if state.gpu_type:
            runner_gpu_type = _gpu_runner_type(state.gpu_type)
            os.environ["TARGET_GPU_TYPE"] = state.gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
            if runner_gpu_type != state.gpu_type:
                print(f"  Magpie runner GPU_TYPE: {runner_gpu_type}")
        # Re-export workload metadata from SharedState so resume sees the same workload contract (not YAML defaults).
        for state_attr, env_name in (
            ("tp", "TP"),
            ("conc", "CONC"),
            ("isl", "ISL"),
            ("osl", "OSL"),
            ("max_model_len", "MAX_MODEL_LEN"),
        ):
            val = getattr(state, state_attr, 0) or 0
            if val:
                os.environ[env_name] = str(val)
                print(f"  re-exported {env_name:<14s}: {val}")
        if state.precision:
            os.environ["PRECISION"] = state.precision
            print(f"  re-exported PRECISION     : {state.precision}")
        if getattr(state, "framework_version", ""):
            os.environ["FRAMEWORK_VERSION"] = state.framework_version
            print(
                f"  re-exported FRAMEWORK_VERSION: "
                f"{state.framework_version}"
            )
        # Honour persisted kernel_enabled on resume; CLI --no-kernel can still override.
        if not state.kernel_enabled:
            args.no_kernel = True
            print("  kernel agent          : DISABLED (persisted from original run)")
        # Same persistence contract for the FRAMEWORK_PR phase toggle.
        if not bool(getattr(state, "framework_phase_enabled", True)):
            args.no_framework = True
            print("  framework phase       : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_framework", False)):
            # Inverse (P2.d): honour --no-framework on resume only before FRAMEWORK_PR is entered.
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if cur_phase in ("", "PRELUDE"):
                state.framework_phase_enabled = False
                # Persist immediately; the later conditional save only runs on prior stop_reason/crash.
                state.save(session_dir)
                print(
                    "  framework phase       : DISABLING for resume "
                    "(--no-framework + phase=PRELUDE)"
                )
            else:
                print(
                    f"  framework phase       : WARN --no-framework ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )
        # Same persistence contract for the EXPLORE phase toggle.
        if not bool(getattr(state, "explore_enabled", True)):
            args.no_explore = True
            print("  explore phase         : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_explore", False)):
            # Honour --no-explore on resume only before EXPLORE is entered.
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if cur_phase in ("", "PRELUDE", "FRAMEWORK_PR"):
                state.explore_enabled = False
                print(
                    "  explore phase         : DISABLING for resume "
                    f"(--no-explore + phase={cur_phase or 'PRELUDE'})"
                )
            else:
                print(
                    f"  explore phase         : WARN --no-explore ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )

        # CRITICAL: clear leftover stop_reason or Orchestration heartbeats forever thinking work is done.
        prior_crash = state.crash_count

        # Issue-G: no_more_leverage / target_reached are intentional terminal states (SKILL Run-time signals);
        # require --force-resume to push past them. Other reasons (time_exhausted, max_ticks, crash) auto-clear.
        force_resume = bool(getattr(args, "force_resume", False))
        gated_terminal = {"no_more_leverage", "target_reached"}
        if prior_stop in gated_terminal and not force_resume:
            print(
                f"\nERROR: --resume blocked by terminal stop_reason="
                f"{prior_stop!r}.\n"
                f"\n"
                f"  SKILL.md (Run-time signals): {prior_stop!r} is a "
                f"deliberate terminal state.\n"
                f"  The optimizer will not auto-resume past it because "
                f"the prior run\n"
                f"  declared exhaustion — picking up where it left off "
                f"only repeats\n"
                f"  the same exhaustion verdict.\n"
                f"\n"
                f"  Override paths:\n"
                f"  1. Pass ``--force-resume`` if you have changed the "
                f"workload /\n"
                f"     search space / model / strategy and want to "
                f"continue regardless.\n"
                f"  2. Start a fresh session (different "
                f"$USER_DATA_PATH) for a clean run.\n"
                f"\n"
                f"  Reports for the prior run live under "
                f"{session_dir}/reports/.\n",
                file=sys.stderr,
            )
            sys.exit(2)

        if prior_stop or prior_crash >= 3:
            state.stop_reason = ""
            state.closing_phase = False
            state.closing_started_unix = 0.0
            state.closing_report_task_id = ""
            # Reset persisted crash_count so a fresh resume isn't immediately tripped into "emergency".
            state.crash_count = 0
            # Reset start_ts to now so resume budget isn't seen as already-over-budget by the LLM.
            from datetime import datetime, timezone
            state.start_ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            state.save(session_dir)
            override_note = (
                " (--force-resume override)"
                if force_resume and prior_stop in gated_terminal
                else ""
            )
            print(
                f"  → cleared stop_reason and reset crash_count "
                f"(was {prior_crash}) for fresh resume{override_note}"
            )
            print(f"  → reset start_ts to {state.start_ts} (resume budget)")
        # Re-bootstrap the Cortex KB client (recreates client + reruns T0 warm-start); resume=True is banner-only.
        cortex_client = _bootstrap_cortex_kb(
            args, session_dir=session_dir, manifest=manifest, resume=True,
        )
        # KnowledgePlane facade (fail-soft degrades when PR Monitor/Cortex unreachable); None only when --degraded-kb.
        knowledge_plane = (
            None if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )
        # No resume backfill needed for roofline (PR #321: roofline_snapshots restored by SharedState.from_dict).
    else:
        # Resolve model path: --model > $MODEL_PATH; fail fast rather than silently use the YAML hardcoded model.
        if not args.model:
            args.model = os.environ.get("MODEL_PATH") or ""
        if not args.model:
            print(
                "ERROR: model is required. Pass --model <path> or set "
                "MODEL_PATH env (or use --resume to continue an existing "
                "session at the canonical session_dir).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Re-export so subprocess executors inject the resolved model into the Magpie YAML, not its hardcoded model.
        os.environ["MODEL_PATH"] = str(args.model)

        # Quantization prelude (one-shot, before any session/baseline work):
        # if --quantize was passed, quantize the source model now and rewrite
        # args.model to the exported quantized model. No-op otherwise.
        await _run_quantization_prelude(args)

        # Resolve framework: --framework > $FRAMEWORK > "sglang" (session-wide; no framework mixing).
        framework = (
            (args.framework or os.environ.get("FRAMEWORK", "")).strip().lower()
            or "sglang"
        )
        if framework not in ("sglang", "vllm", "atom"):
            print(
                f"ERROR: --framework must be sglang, vllm, or atom "
                f"(got {framework!r}); set $FRAMEWORK accordingly or pass "
                "--framework",
                file=sys.stderr,
            )
            sys.exit(2)
        os.environ["FRAMEWORK"] = framework
        print(f"Framework       : {framework}")

        # B3: --framework atom auto-tightens incompatible phases (see _apply_atom_auto_tighten).
        if framework == "atom":
            _apply_atom_auto_tighten(args)

        # Resolve real target GPU: probe > --gpu-type hint; probe wins to catch wrong-host typos that corrupt KB.
        user_specified = (
            (args.gpu_type or os.environ.get("GPU_TYPE", "")).strip().lower()
        )
        probed = _autodetect_gpu_type() or ""
        gpu_type, gpu_warnings = _resolve_gpu_type(
            user_specified=user_specified,
            probed=probed,
        )
        for line in gpu_warnings:
            print(line, file=sys.stderr)
        if probed and not user_specified:
            print(f"GPU type        : {gpu_type} (auto-detected)")
        runner_gpu_type = _gpu_runner_type(gpu_type)
        if gpu_type and runner_gpu_type != gpu_type:
            print(
                f"WARN: {gpu_type} uses {runner_gpu_type} as Magpie "
                f"runner_type (same gfx942/CDNA3 arch; Magpie has no "
                f"sglang_{gpu_type}.sh / vllm_{gpu_type}.sh yet)",
                file=sys.stderr,
            )
        args.gpu_type = gpu_type or None
        if runner_gpu_type:
            os.environ["TARGET_GPU_TYPE"] = gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"GPU type        : {gpu_type}")
            print(f"Magpie runner   : {runner_gpu_type} (will inject runner_type into Magpie YAML)")
        else:
            os.environ.pop("TARGET_GPU_TYPE", None)
            os.environ.pop("GPU_TYPE", None)
            args.gpu_type = None
            print("GPU type        : <unset> (Magpie will auto-detect)")

        # MAX_MODEL_LEN = ISL+OSL+headroom clamped to native window (see _resolve_max_model_len); exported for YAML.
        max_model_len = _resolve_max_model_len(
            args.isl, args.osl, str(args.model or ""),
        )
        os.environ["MAX_MODEL_LEN"] = str(max_model_len)
        os.environ["ISL"] = str(args.isl)
        os.environ["OSL"] = str(args.osl)
        os.environ["PRECISION"] = args.precision
        # Mirror resolved framework_version into env (explicit > auto-detect > unset; see _resolve_framework_version).
        _fw_version_for_env = (
            (getattr(args, "framework_version", None) or "").strip()
            or (os.environ.get("FRAMEWORK_VERSION", "") or "").strip()
        )
        if not _fw_version_for_env:
            from .recipe_snapshot_constants import (
                DEFAULT_FRAMEWORK_VERSION_SLUG,
                detect_framework_version,
            )

            _detected = detect_framework_version(
                (getattr(args, "framework", None) or "").strip()
                or os.environ.get("FRAMEWORK", "")
            )
            if _detected and _detected != DEFAULT_FRAMEWORK_VERSION_SLUG:
                _fw_version_for_env = _detected
        if _fw_version_for_env:
            os.environ["FRAMEWORK_VERSION"] = _fw_version_for_env
        print(f"Workload        : ISL={args.isl} OSL={args.osl} "
              f"MAX_MODEL_LEN={max_model_len} PRECISION={args.precision} "
              f"FRAMEWORK_VERSION={_fw_version_for_env or '<unset>'}")

        # session_dir defaults to <workspace_root>/<model>/<UTC ts>/ (INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat for legacy).
        session_dir = make_session_dir(model_name=args.model)
        manifest = write_manifest(session_dir, args=args)
        print(f"Session dir     : {session_dir}")
        print(f"Session id      : {manifest['session_id']}  (manifest label only)")
        _print_session_skeleton(session_dir)

        # Machine-readable launch info: stable point for launcher scripts to harvest pid/session_dir/run_log.
        _emit_launch_info(
            pid=os.getpid(),
            session_dir=session_dir,
            session_id=str(manifest["session_id"]),
            run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
            gpu_type=gpu_type or "",
            framework=args.framework or "",
            model=str(args.model) if args.model else "",
            launch_info_file=getattr(args, "launch_info_file", None),
        )
        _seed_shared_state(
            session_dir, args, session_id=manifest["session_id"],
        )
        # Unsupported-model preflight: reject multimodal/vision configs (runs after seed, before heavy bring-up).
        if _preflight_unsupported_model_arch(args, session_dir):
            sys.exit(2)
        # Model-config compatibility preflight: reject statically-broken
        # configs before the heavy server bring-up.
        if _preflight_model_config_compat(args, session_dir):
            sys.exit(2)
        # Context-window preflight: reject when ISL+OSL+headroom exceeds max_position_embeddings (no stretch by policy).
        if _preflight_context_window(args, session_dir):
            sys.exit(2)
        # Cortex KB T0 anchor (after seed for recipe_canonical_id, before Coordinator); fails fast unless --degraded-kb.
        cortex_client = _bootstrap_cortex_kb(
            args, session_dir=session_dir, manifest=manifest, resume=False,
        )
        # KnowledgePlane facade for specialists (fail-soft both sides; always non-None for dispatch).
        knowledge_plane = (
            None if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )

    objective = build_objective({
        "MAX_HOURS": str(args.max_hours),
        "TARGET_GAIN_PCT": str(args.target_gain) if args.target_gain else "",
        "TARGET_TPUT_PER_GPU": str(args.target_tput) if args.target_tput else "",
        "TARGET_DIR": args.target_baseline_dir or "",
    })
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    no_kernel = getattr(args, "no_kernel", False)
    no_explore = getattr(args, "no_explore", False)
    if no_explore and no_kernel:
        print(
            "WARNING: --no-explore and --no-kernel are both set; the run "
            "collapses to baseline -> SWEEP over an empty optimization_stack "
            "(no EXPLORE param search, no KERNEL rewrites). SWEEP only "
            "re-validates the baseline recipe. Continuing as requested.",
            file=sys.stderr,
        )
    elif no_explore:
        print("Explore phase   : DISABLED (--no-explore); "
              f"{'baseline -> SWEEP' if no_kernel else 'baseline -> KERNEL -> SWEEP'}")
    if bool(getattr(args, "research_scout", True)):
        print(
            "Research scout  : ENABLED at PRELUDE (re-dispatch every "
            f"{max(1, int(getattr(args, 'research_scout_interval', 3) or 3))} "
            "explore rounds)"
        )
    else:
        print("Research scout  : DISABLED (--no-research-scout)")
    if bool(getattr(args, "target_advisory", True)):
        print("Target advisory : ENABLED (External target gap injected into "
              "prompts; advisory-only)")
    else:
        print("Target advisory : DISABLED (--no-target-advisory)")
    if bool(getattr(args, "recipe_sediment", True)):
        print("Recipe sediment : ENABLED (KEEP/REVERT provenance written to "
              "persistent recipe)")
    else:
        print("Recipe sediment : DISABLED (--no-recipe-sediment)")
    if bool(getattr(args, "allow_empty_kernel_shape", False)):
        os.environ["HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE"] = "1"
        print("Kernel shape    : empty-shape dispatch ALLOWED "
              "(--allow-empty-kernel-shape)")
    else:
        os.environ.pop("HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", None)
        print("Kernel shape    : non-empty trace shape REQUIRED for "
              "kernel-opt dispatch")

    # Resolve critic backend + runtime root before _build_backends; abort rc=2 if --critic-agent runtime unreachable.
    critic_choice = _resolve_critic_choice(args)
    critic_agent_root: Path | None = None
    critic_kb_mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    if critic_kb_mode not in ("inmemory", "live"):
        print(
            f"ERROR: CRITIC_KB_CLIENT_MODE={critic_kb_mode!r} not in "
            "{'inmemory','live'}",
            file=sys.stderr,
        )
        sys.exit(2)
    if critic_choice == "agent":
        critic_agent_root = _resolve_critic_agent_root()
        if critic_agent_root is None:
            print(
                f"ERROR: --critic-agent selected but critic-agent runtime not "
                f"found.\n"
                f"  Set ${_CRITIC_AGENT_ROOT_ENV} to the directory containing "
                f"runtime/cli.py, or install critic-agent at "
                f"$REPO_ROOT/critic-agent/.\n"
                f"  Bypass with --critic-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_critic_agent_runtime(critic_agent_root)
        if critic_kb_mode == "live" and not os.environ.get("KB_BASE_URL"):
            print(
                "ERROR: CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not "
                "set. Either export KB_BASE_URL or unset "
                "CRITIC_KB_CLIENT_MODE to fall back to inmemory.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Default WORKSPACE_PATH for critic-agent runtime: SKILL static-asset root (repo root), not artefact dir.
        os.environ.setdefault("WORKSPACE_PATH", str(Path(__file__).resolve().parents[1]))

    # Resolve robustness backend choice + runtime root, mirroring critic.
    robustness_choice = _resolve_robustness_choice(args)
    robustness_agent_root: Path | None = None
    robustness_options = _build_robustness_options(args)
    if robustness_choice == "agent":
        robustness_agent_root = _resolve_robustness_agent_root()
        if robustness_agent_root is None:
            print(
                f"ERROR: --robustness-agent selected but robustness-agent "
                f"runtime not found.\n"
                f"  Set ${_ROBUSTNESS_AGENT_ROOT_ENV} to the directory "
                f"containing src/robustness_agent/runtime/cli.py, or install "
                f"robustness-agent at $REPO_ROOT/robustness-agent/.\n"
                f"  Bypass with --robustness-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_robustness_agent_runtime(robustness_agent_root)

    backends = _build_backends(
        claude_model=args.claude_model,
        codex_model=args.codex_model,
        kernel_codex=args.kernel_codex,
        critic_choice=critic_choice,
        session_dir=session_dir,
        critic_agent_root=critic_agent_root,
        critic_kb_mode=critic_kb_mode,
        robustness_choice=robustness_choice,
        robustness_agent_root=robustness_agent_root,
        robustness_options=robustness_options,
        no_kernel=no_kernel,
    )
    # Bug A: expose active session_dir to in-process executors (read in report.py::_resolve_session_dir).
    os.environ["USER_DATA_PATH"] = str(session_dir)
    # Production: enable strict PolicyGate path-containment (escaping intents land as policy_denied).
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"
    # PolicyGate R1 phase_incompatible enforcement for production runs (env affects cli boot path only).
    if getattr(args, "strict_phase", True):
        os.environ["INFERENCE_OPTIMIZER_STRICT_PHASE"] = "1"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_STRICT_PHASE", None)
    # Propagate --legacy-action-scores so SharedState.from_dict handles drop/warn uniformly (default drop).
    legacy_mode = str(
        getattr(args, "legacy_action_scores", "drop") or "drop",
    ).strip().lower()
    if legacy_mode == "warn":
        os.environ["INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES"] = "warn"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", None)
    # Propagate --migration-mode: SharedState.from_dict treats fact-layer discrepancy as fatal (strict) or WARN (lenient).
    migration_mode = str(
        getattr(args, "migration_mode", "strict") or "strict",
    ).strip().lower()
    if migration_mode == "lenient":
        os.environ["INFERENCE_OPTIMIZER_MIGRATION_MODE"] = "lenient"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_MIGRATION_MODE", None)
    # --reset-state backs up state.json and starts blank, before Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir)
    # Propagate --breakdown-include-transcripts (inline / path-only choice) to end-of-session breakdown.
    transcripts_flag = str(
        getattr(args, "breakdown_include_transcripts", "false") or "false",
    ).strip().lower()
    if transcripts_flag == "true":
        os.environ["INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS"] = "1"
    else:
        os.environ.pop(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", None,
        )

    # Build phase budget pct dict from CLI flags; absent values fall back to Coordinator library defaults.
    phase_budget_pct: dict[str, float] = {}
    for cli_field, phase_name in (
        ("phase_budget_prelude_pct", "PRELUDE"),
        ("phase_budget_explore_pct", "EXPLORE"),
        ("phase_budget_kernel_pct",  "KERNEL"),
        ("phase_budget_sweep_pct",   "SWEEP"),
        ("phase_budget_close_pct",   "CLOSE"),
    ):
        val = getattr(args, cli_field, None)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)

    # When kernel is disabled, strip it from the role registry (no tick / no backend expectation).
    role_registry = None
    if no_kernel:
        from .orchestrator.agent_role import default_role_registry
        role_registry = {
            k: v for k, v in default_role_registry().items() if k != "kernel"
        }

    coordinator = Coordinator(
        session_dir, backends=backends, role_registry=role_registry,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        model_class=(
            getattr(args, "model_class", None)
            or os.environ.get("MODEL_CLASS")
            or ""
        ),
        cortex_kb=cortex_client,
        phase_budget_pct=phase_budget_pct or None,
        # KnowledgePlane facade (None when --degraded-kb).
        knowledge_plane=knowledge_plane,
        # Advisory multi-model specialist-proposal scorer. ``None`` when
        # --no-proposal-scoring or an empty model list; otherwise scores
        # each proposal_set and surfaces the results to Orchestration as
        # one reference among many (never gates anything). ``session_dir``
        # is forwarded so the scorer can append its per-model token usage
        # to the full-trace ledger (component=proposal_scorer).
        proposal_scorer=_build_proposal_scorer(args, session_dir),
        # Warm-recipe replay controls. Default ON, fires when
        # warm_start_recipe.confidence >= min_confidence and the
        # measured gain reproduces at least min_reproduce_pct of the
        # recipe's historical claim. Manifest is the persistent
        # authority across restarts (resume-safe).
        warm_replay_enabled=_resume_safe_flag(
            args, "no_warm_replay", manifest, "warm_replay_enabled",
            default=True, invert=True,
        ),
        warm_replay_min_confidence=_resume_safe_numeric(
            args, "warm_replay_min_confidence", manifest,
            "warm_replay_min_confidence", default=0.7,
        ),
        warm_replay_min_reproduce_pct=_resume_safe_numeric(
            args, "warm_replay_min_reproduce_pct", manifest,
            "warm_replay_min_reproduce_pct", default=0.8,
        ),
    )
    framework_for_prompt = (
        os.environ.get("FRAMEWORK", "").strip().lower() or "sglang"
    )
    max_minutes_for_prompt = int(round(float(args.max_hours) * 60))
    prompts: dict[str, str] = {
        "orchestration": args.orch_prompt or _build_orchestration_prompt(
            no_kernel=no_kernel,
            framework=framework_for_prompt,
            objective=objective,
            max_minutes=max_minutes_for_prompt,
        ),
        "critic":        args.critic_prompt or _load_critic_prompt(),
    }
    if not no_kernel:
        prompts["kernel"] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT
    coordinator.system_prompt_overrides = prompts
    # ``fa phase-discover`` timeout override (falsy -> DEFAULT_FA_PHASE_TIMEOUT_SEC 180s).
    try:
        coordinator.framework_pr_discover_timeout_sec = float(
            getattr(args, "framework_pr_discover_timeout_sec", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        coordinator.framework_pr_discover_timeout_sec = 0.0
    # Build specialist executor only when research_lane capacity > 0 (0 degrades to LLM-direct grid).
    specialist_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 0
    )
    specialist_executor: "Any" = None
    if specialist_capacity > 0:
        specialist_executor = _build_specialist_executor(
            args,
            session_dir=session_dir,
            knowledge_plane=knowledge_plane,
        )
    _register_executors(
        coordinator,
        no_kernel=no_kernel,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        session_dir=session_dir,
        specialist_executor=specialist_executor,
    )
    # Persist effective system prompts for resume / drift inspection.
    _snapshot_system_prompts(session_dir, prompts=prompts)

    kernel_str = "DISABLED" if no_kernel else (
        f"{'Codex' if args.kernel_codex else 'Claude'}"
    )
    if critic_choice == "mock":
        critic_str = "mock"
    else:  # "agent"
        critic_str = (
            f"critic-agent(kb={critic_kb_mode}, codex={args.codex_model}, "
            f"root={critic_agent_root})"
        )
    if robustness_choice == "mock":
        robustness_str = "mock"
    else:
        robustness_str = f"robustness-agent(root={robustness_agent_root})"
        if robustness_options:
            kvs = ",".join(f"{k}={v!r}" for k, v in sorted(robustness_options.items()))
            robustness_str += f"[{kvs}]"
    print(f"Backends        : "
          f"orchestration=Claude({args.claude_model}), "
          f"kernel={kernel_str}, "
          f"critic={critic_str}, "
          f"robustness={robustness_str}")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} "
          f"(budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()

    if not (getattr(args, "compare_against_gpu", None) or "").strip():
        print(
            "[target_analysis] no --compare-against-gpu set; will write a "
            "marker JSON at $SESSION_DIR/target_analysis/target_baseline.json "
            "(reason=no_target_gpu_configured) — set --compare-against-gpu "
            "to fetch real InferenceX reference data.",
            file=sys.stderr,
        )

    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
            closing_grace_sec=args.closing_grace_sec,
        )
    finally:
        await coordinator.stop()
        # End-of-session safety net: always materialize session_breakdown.json (best-effort; never mask stop_reason).
        # Skip when the CLOSE sequencer already wrote it (close_sequence_done is locked in CORE_STATE_FIELDS).
        sequencer_done = getattr(
            coordinator.shared_state, "close_sequence_done", False,
        )
        if sequencer_done:
            print(
                "Session breakdown : (already written by CLOSE phase "
                "sequencer; skipping cli.finally safety-net write)"
            )
        else:
            try:
                from .breakdown import write_breakdown_json
                breakdown_path = write_breakdown_json(session_dir)
                print(f"Session breakdown : {breakdown_path}")
            except Exception:  # noqa: BLE001
                log.exception(
                    "session_breakdown finalize failed (non-fatal)"
                )
            # Issue-I: safety-net reports/final.md (no-op when the sequencer's final.md already exists).
            try:
                from .breakdown import write_minimal_final_report
                final_md = write_minimal_final_report(session_dir)
                print(f"Final report      : {final_md}")
            except Exception:  # noqa: BLE001
                log.exception(
                    "emergency final report write failed (non-fatal)"
                )

    _reconcile_crash_count(coordinator.shared_state, session_dir)
    # NOTE: conc_sweep is now a SWEEP-phase action auto-enqueued by the Coordinator, not a post-hook here.

    _print_final_summary(coordinator.shared_state, stop_reason)
    return 0 if stop_reason in (
        "target_reached",
        "no_more_leverage",
        "time_exhausted",
        "max_ticks",
    ) else 1


def _default_research_lane_capacity() -> int:
    """Default ``--research-lane-capacity``: $INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY else the GPU ceiling (2×GPU)."""
    env = os.environ.get("INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    from inference_optimizer.orchestrator.policy import research_lane_ceiling
    return research_lane_ceiling()


def _build_parser() -> argparse.ArgumentParser:
    from inference_optimizer.orchestrator.specialist_domains import (
        DEFAULT_SPECIALIST_MAX_TURNS as _DEFAULT_SPECIALIST_MAX_TURNS,
    )
    p = argparse.ArgumentParser(
        prog="inference_optimizer",
        description="Inference Optimizer v0.6 — multi-agent SGLang/vLLM optimization",
    )
    p.add_argument("--verbose", "-v", action="count", default=0,
                    help="Verbose logging (-v INFO, -vv DEBUG)")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize",
                          help="Drive a multi-agent optimization run on a model")
    opt.add_argument("--model", "-m", type=Path, default=None,
                      help="Model path (required for new runs; ignored when "
                           "--resume is set — model is read from manifest.json/"
                           "state.json)")
    opt.add_argument(
        "--quantize", type=str, default=None, metavar="PROMPT",
        help="Optional natural-language quantization request. When set, the "
             "quantization-agent runs ONCE as a prelude before the "
             "optimization loop: it drives AMD Quark PTQ from this prompt, "
             "then rewrites --model to the exported quantized model so the "
             "rest of the run optimizes the quantized model. Ignored on "
             "--resume.",
    )
    from .orchestrator.quantization_schemes import QUANT_SCHEME_CHOICES
    opt.add_argument(
        "--quantize-scheme", choices=QUANT_SCHEME_CHOICES, default=None,
        metavar="SCHEME",
        help="Structured alternative to --quantize for UI/backends: pick a "
             "curated quantization scheme (resolved to a prompt internally). "
             f"Choices: {', '.join(QUANT_SCHEME_CHOICES)}. 'none' or omit = no "
             "quantization. Ignored if --quantize (free text) is also given.",
    )
    opt.add_argument(
        "--gpu-type", choices=["mi300x", "mi308x", "mi325x", "mi355x"], default=None,
        help="Hint for the real target GPU. The rocm-smi probe always "
             "wins when both are present and disagree; a WARN is "
             "emitted to stderr so the operator sees the typo. Used "
             "verbatim only when the probe fails (CPU sandbox / no "
             "rocm-smi). Magpie runner_type is derived separately; "
             "mi325x currently runs with mi300x runner scripts because "
             "Magpie does not yet ship sglang_mi325x.sh / vllm_mi325x.sh.",
    )
    opt.add_argument(
        "--framework", choices=["sglang", "vllm", "atom"], default=None,
        help="Inference framework to benchmark / optimize. Resolution order: "
             "--framework > $FRAMEWORK env > sglang (default). Selection is "
             "session-wide; mixing frameworks in a single session is not "
             "supported. NOTE: --framework atom is single-node-only "
             "(``--nodes>=2`` fails fast); profile / roofline, "
             "kernel-agent, and framework-agent are all enabled on atom. "
             "The auto-tighten guard only enforces ``--nodes 1``.",
    )
    opt.add_argument(
        "--nodes", type=int,
        # Resolution: --nodes > $INFERENCE_OPTIMIZER_NODES > $NODES > 1 ($NODES fallback for SaFE optimizer.env).
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_NODES")
            or os.environ.get("NODES")
            or "1"
        ),
        help="Total number of GPU nodes for the inference RayJob. "
             "1 (default) keeps the legacy single-pod path. "
             ">=2: `optimize` provisions the SaFE RayJob before preflight "
             "(unless already in /tmp/multi_node_state.json), runs bootstrap "
             "once, and exports RAY_ADDRESS for kernel-agent. Does not stop the "
             "RayJob on exit; run `python3 -m inference_optimizer.multi_node "
             "stop-rayjob` when you want to release it. Requires "
             "--rayjob-image or INFERENCE_OPTIMIZER_RAYJOB_IMAGE. "
             "Resolution: --nodes > $INFERENCE_OPTIMIZER_NODES > $NODES > 1.",
    )
    opt.add_argument(
        "--rayjob-image",
        default=None,
        help="Container image for the multi-node RayJob (head+workers). "
             "Required when --nodes>=2 unless INFERENCE_OPTIMIZER_RAYJOB_IMAGE "
             "is set or state file last_create_request.image is present.",
    )
    opt.add_argument(
        "--rayjob-gpus-per-node",
        type=int,
        default=None,
        help="GPUs per RayJob pod (default: INFERENCE_OPTIMIZER_GPUS_PER_NODE "
             "or 8). Passed to multi_node create-rayjob.",
    )
    # --rayjob-extra-env is a prompt-driven pass-through forwarded verbatim to workload_spec.env; the CLI
    # invents no keys. Reserved RAY_JOB_ENTRYPOINT stripped downstream; credential keys auto-injected elsewhere.
    opt.add_argument(
        "--rayjob-extra-env",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra env entries to inject into the multi-node RayJob "
             "(repeatable). Agent maps each line of the user prompt's "
             "`env:` block into one --rayjob-extra-env K=V; the CLI "
             "does not own any default. Skip *_API_KEY / *_BASE_URL "
             "(auto-injected by _credential_fanout) and RAY_JOB_ENTRYPOINT "
             "(reserved by workload_spec). Only takes effect when "
             "--nodes>=2 and this run actually creates the RayJob; "
             "idempotent reuse of an existing rayjob_id keeps the env "
             "set at original create time.",
    )
    opt.add_argument(
        "--tp", type=int,
        default=int(os.environ.get("TP", "1") or 1),
        help="Tensor parallel size. Resolution: --tp > $TP env > 1. "
             "Symmetric with --ep — historically TP only flowed in via "
             "$TP env (read by _workload_envs); the CLI flag was added "
             "so the agent can pass `--tp N` directly from the prompt's "
             "Environment block instead of having to `export TP=N` "
             "first. Either path still works.",
    )
    opt.add_argument(
        "--conc", type=int,
        default=int(os.environ.get("CONC", "8") or 8),
        help="Magpie client concurrency cap (max in-flight requests). "
             "Resolution: --conc > $CONC env > 8. Symmetric with --tp; "
             "agent can pass `--conc N` directly from the prompt.",
    )
    opt.add_argument(
        "--ep", type=int,
        default=int(os.environ.get("EP", "1") or 1),
        help="Expert-parallel size for MoE inference. 1 (default) keeps "
             "experts sharded by TP (legacy behaviour). >=2 enables true "
             "expert parallelism: sglang adds `--expert-parallel-size N`, "
             "vllm adds `--enable-expert-parallel`. Typical: EP=TP for "
             "DSr1/DSv3 on multi-node. Resolution: --ep > $EP env > 1. "
             "EP > TP is rejected at server-restart time.",
    )
    opt.add_argument(
        "--pd-mode",
        choices=("colocated", "disaggregated"),
        default="colocated",
        help="Prefill-Decode disaggregation mode. ALWAYS defaults to "
             "`colocated` regardless of any inherited $PD_MODE env, so "
             "PD only turns on when the agent explicitly passes "
             "`--pd-mode disaggregated` (driven by the prompt's "
             "Environment block having a PD_MODE=disaggregated line). "
             "Stale env from a previous restart cannot accidentally "
             "re-enable PD.",
    )
    opt.add_argument(
        "--pd-prefill-nodes", type=int,
        default=int(os.environ.get("PD_PREFILL_NODES", "0") or 0),
        help="Number of prefill nodes (disaggregated only); pn+dn=nodes",
    )
    opt.add_argument(
        "--pd-decode-nodes", type=int,
        default=int(os.environ.get("PD_DECODE_NODES", "0") or 0),
        help="Number of decode nodes (disaggregated only)",
    )
    opt.add_argument(
        "--pd-prefill-tp", type=int,
        default=int(os.environ.get("PD_PREFILL_TP", "0") or 0),
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-decode-tp", type=int,
        default=int(os.environ.get("PD_DECODE_TP", "0") or 0),
        help="TP for decode group (disaggregated only); default = --tp",
    )
    opt.add_argument(
        "--pd-transfer-backend", type=str,
        default=os.environ.get("PD_TRANSFER_BACKEND", ""),
        help="sglang: mooncake|nixl ; vllm: NixlConnector|...; empty = default",
    )
    opt.add_argument(
        "--pd-ib-device", type=str,
        default=os.environ.get("PD_IB_DEVICE", ""),
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). "
             "Empty = use $NCCL_IB_HCA from RayJob pod env at server-launch time.",
    )
    opt.add_argument(
        "--skip-variants", type=str,
        default=os.environ.get("SKIP_VARIANTS", ""),
        help="Comma/whitespace-separated list of variant names or fnmatch "
             "globs to drop from the backends/params grids before launch. "
             "Examples: `attn_aiter` (exact), `attn_aiter,sched_dfs` (two "
             "exacts), `attn_*,vllm_aiter_*` (globs). Resolution: "
             "--skip-variants > $SKIP_VARIANTS > empty. Exported back into "
             "$SKIP_VARIANTS so all executors and the multi-node orchestrator "
             "subprocess see the same value. Dropped variants surface in "
             "state.json under each action's `dropped_variants` field tagged "
             "`source=user_skip`.",
    )
    opt.add_argument("--max-hours", type=float, default=2.0,
                      help="Wall-clock budget in hours (default 2.0)")
    opt.add_argument(
        "--closing-grace-sec",
        type=float,
        default=None,
        help=(
            "Extra seconds after the wall-clock deadline for Coordinator to "
            "flush a deterministic report task (no LLM). Default: "
            "min(120, max_hours * 60 * 0.02). Pass 0 to disable closing phase."
        ),
    )
    opt.add_argument("--isl", type=int, default=int(os.environ.get("ISL", "256")),
                      help="Input sequence length (default $ISL or 256)")
    opt.add_argument("--osl", type=int, default=int(os.environ.get("OSL", "256")),
                      help="Output sequence length (default $OSL or 256)")
    opt.add_argument("--precision", type=str,
                      default=os.environ.get("PRECISION", "bf16"),
                      help="Model precision (default $PRECISION or bf16)")
    opt.add_argument(
        "--framework-version",
        dest="framework_version",
        type=str,
        default=None,
        help=(
            "Framework version slug for the recipe-snapshot canonical id "
            "(scopes recipes to a specific framework release — sglang 0.4.5 "
            "and sglang 0.5.x have different scheduler defaults so they "
            "deserve separate KB rows). When omitted, auto-detected via "
            "importing the framework's top-level package and reading "
            "``__version__`` (sglang/vllm/atom supported); auto-detect "
            "failure degrades to 'unknown_version'. Override with "
            "--framework-version=0.4.5 to pin a specific tag for the run."
        ),
    )
    opt.add_argument(
        "--continue-kernel-after-gemm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After FP8 GEMM tuning succeeds, continue into source-level "
            "kernel_opt. Use --no-continue-kernel-after-gemm for a "
            "GEMM-only validation run."
        ),
    )
    grp = opt.add_mutually_exclusive_group()
    grp.add_argument("--target-gain", type=float, default=None,
                      help="Stop when cumulative_gain >= N%% over baseline")
    grp.add_argument("--target-tput", type=float, default=None,
                      help="Stop when current best tok/s/GPU >= N")
    grp.add_argument("--target-baseline-dir", type=str, default=None,
                      help="Stop when current best matches the baseline in DIR")
    opt.add_argument("--resume", action="store_true", default=False,
                      help="Resume an existing session. Without --resume-from, "
                           "auto-picks the latest per-session subdir under "
                           "$USER_DATA_PATH/<model>/<UTC ts>/ (N17 layout) or "
                           "falls back to $USER_DATA_PATH (legacy flat layout). "
                           "USER_DATA_PATH MUST stay at workspace level "
                           "(/wekafs/.../sessions/, not the per-session subdir) "
                           "so runtime/ resolution works. Skips the SharedState "
                           "seed and lets the Coordinator replay the prior "
                           "event log + state.json.")
    opt.add_argument("--resume-from", type=str, default=None,
                      help="Explicit per-session subdir to resume from. Use "
                           "when multiple per-launch ts dirs exist under the "
                           "same model and the latest is not what you want. "
                           "Must be an absolute path under $USER_DATA_PATH "
                           "(workspace_root). Implies --resume.")
    opt.add_argument(
        "--force-resume", action="store_true", default=False,
        help=(
            "Allow ``--resume`` to push past a terminal "
            "``stop_reason='no_more_leverage'`` or ``'target_reached'``. "
            "Without this flag the resume aborts (Issue-G guard, per "
            "SKILL.md 'Run-time signals': those terminals require an "
            "operator-side workload / strategy change before resuming). "
            "No-op outside ``--resume``."
        ),
    )
    opt.add_argument(
        "--model-class", type=str,
        default=os.environ.get("MODEL_CLASS", None),
        help=(
            "Categorical model-class key. It is the deterministic key for "
            "several consumers: the atom explore seed grid "
            "(action_executors/explore.py), the framework-agent gap search "
            "token (action_executors/_framework_gap_composer.py), the recipe "
            "key, and the orchestration prompt label. Recognised values "
            "(case-insensitive, with -/+/space tolerated): dense / moe_mla / "
            "moe_swa / moe_mla_nsa. When unset, Coordinator boot infers and "
            "persists it from config.json (num_experts / architectures) or "
            "model-path family keywords, replacing the deleted ``classify`` "
            "action's lightweight state-write role. For richer *advisory* "
            "model context (attention variant, KV/token, experts, MTP, ...) "
            "the SKILL launcher writes $USER_DATA_PATH/model_arch.json, which "
            "is injected into prompts but drives no gating."
        ),
    )
    opt.add_argument("--target-summary", type=str, default=None,
                      help="Free-text goal summary surfaced in prompts")
    opt.add_argument(
        "--compare-against-gpu", type=str, default=None,
        help=(
            "Reference GPU hardware key for external baseline comparison "
            "(e.g. b300 / mi355x / h200). target_analysis ALWAYS runs as "
            "TODO 0 and always writes "
            "$SESSION_DIR/target_analysis/target_baseline.json + a short "
            "MD report. When this flag is set, the JSON carries the "
            "matching InferenceX (https://inferencex.semianalysis.com) "
            "reference data point; when unset, the JSON carries a "
            "structured reason='no_target_gpu_configured' marker so the "
            "report still has a deterministic 'External baseline' "
            "section. The data is REPORT-ONLY: it does not influence "
            "Objective, scoring, or any agent prompt. Other dimensions "
            "(model / framework / precision / ISL / OSL) are derived "
            "from --model and the standard FRAMEWORK / PRECISION / ISL / "
            "OSL env vars."
        ),
    )
    opt.add_argument("--max-ticks", type=int, default=None,
                      help="Hard tick cap (None = unlimited; mostly for tests)")
    opt.add_argument("--tick-interval-sec", type=float, default=0.0,
                      help="Sleep between ticks (0 = no sleep)")
    opt.add_argument("--claude-model", type=str,
                      default=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"))
    opt.add_argument("--codex-model", type=str,
                      default=os.environ.get("CODEX_MODEL", "gpt-5.4"))
    opt.add_argument("--no-kernel", action="store_true", default=False,
                      help="Disable the Kernel agent entirely. The run will "
                           "only do baseline + params + backends + sweep (pure "
                           "parameter search). Useful when GEAK/OOB/GPU "
                           "compile env is unavailable or you just want the "
                           "quick-win parameter path. Default: kernel enabled.")
    opt.add_argument("--no-explore", action="store_true", default=False,
                      help="Skip the EXPLORE phase entirely. PRELUDE (and "
                           "FRAMEWORK_PR, if enabled) route straight to KERNEL "
                           "— or to SWEEP when --no-kernel is also set. Useful "
                           "for a baseline -> kernel-only run, or to validate "
                           "the current recipe via SWEEP without a serving-"
                           "param search. Default: explore enabled.")
    opt.add_argument(
        "--launch-info-file", type=str, default=None,
        help="Write a JSON file with the launched session's pid, "
             "session_dir, session_id, run_log, manifest path, gpu_type, "
             "framework and model. Launcher scripts can ``jq -r .pid`` / "
             "``jq -r .session_dir`` instead of grepping stdout or "
             "pgrep'ing. Always emitted alongside the ``HYPERLOOM_LAUNCH "
             "<key=value> ...`` single-line sentinel that is printed to "
             "stdout for stream-based parsers.",
    )
    opt.add_argument("--framework-pr-discover-timeout-sec", type=float,
                      default=0.0,
                      help="Override the per-call timeout for "
                           "``fa phase-discover``. 0 (the default) uses "
                           "framework_agent_client.DEFAULT_FA_PHASE_TIMEOUT_SEC "
                           "(180s). The Coordinator retries discover up to "
                           "DISCOVER_FAILURE_RETRY_LIMIT (3) consecutive "
                           "failures before marking FRAMEWORK_PR done.")
    opt.add_argument("--no-framework", action="store_true",
                      default=os.environ.get(
                          "INFERENCE_OPTIMIZER_NO_FRAMEWORK", "0",
                      ).strip() in ("1", "true", "True", "TRUE", "yes"),
                      help="Skip the FRAMEWORK_PR phase (PRELUDE → EXPLORE "
                           "directly). The phase pre-scans upstream sglang/"
                           "vllm PRs via framework-agent and lands KEPT "
                           "patches before EXPLORE starts. Disable when "
                           "the framework-agent toolchain is unavailable "
                           "or you want a faster cold start. Also read from "
                           "$INFERENCE_OPTIMIZER_NO_FRAMEWORK=1. "
                           "Default: framework phase enabled.")
    opt.add_argument("--kernel-codex", action="store_true", default=True,
                      help="Use Codex backend for Kernel agent (default — faster). "
                           "Pass --kernel-claude to switch.")
    opt.add_argument("--kernel-claude", action="store_false", dest="kernel_codex",
                      help="Use Claude backend for Kernel agent")
    # Critic backend selection; flags are aliases setting the same dest, default/conflicts resolved in _resolve_critic_choice.
    opt.add_argument(
        "--critic-mock",
        dest="critic_backend",
        action="store_const",
        const="mock",
        default=None,
        help="Force the always-approve mock Critic (offline / smoke tests).",
    )
    opt.add_argument(
        "--critic-agent",
        dest="critic_backend",
        action="store_const",
        const="agent",
        help="Force the critic-agent runtime backend (KB + session memory + "
             "review_constraints). Requires CRITIC_AGENT_ROOT or a sibling "
             "$REPO_ROOT/critic-agent/ directory.",
    )
    # Robustness backend selection (mirrors critic)
    opt.add_argument(
        "--robustness-mock",
        dest="robustness_backend",
        action="store_const",
        const="mock",
        default=None,
        help="Force the heartbeat-only mock Robustness backend.",
    )
    opt.add_argument(
        "--robustness-agent",
        dest="robustness_backend",
        action="store_const",
        const="agent",
        help="Force the robustness-agent runtime backend (subprocess + JSON, "
             "mirrors critic-agent transport). Requires ROBUSTNESS_AGENT_ROOT "
             "or a sibling $REPO_ROOT/robustness-agent/ directory.",
    )
    opt.add_argument(
        "--robustness-server-url",
        dest="robustness_server_url",
        type=str,
        default=None,
        help="Override the robustness-server base URL forwarded into "
             "request.options. Honoured only when --robustness-agent is "
             "selected.",
    )
    opt.add_argument(
        "--robustness-llm-rca",
        dest="robustness_llm_rca",
        action="store_true",
        default=None,
        help="Forward llm_rca_enabled=true into request.options. The agent "
             "still falls back to NoopRcaEngine when LLM credentials aren't "
             "set in the runtime env.",
    )
    opt.add_argument(
        "--no-robustness-llm-rca",
        dest="robustness_llm_rca",
        action="store_false",
        help="Forward llm_rca_enabled=false into request.options.",
    )
    opt.add_argument(
        "--robustness-workload-uid",
        dest="robustness_workload_uid",
        type=str,
        default=None,
        help="Forward workload_uid into request.options. The robustness-server "
             "resolves it to every pod (head + workers) backing the RayJob via "
             "the cluster/workloads/{uid}/hierarchy endpoint. Falls back to "
             "$CLAW_WORKLOAD_UID / $WORKLOAD_UID / $RAY_JOB_ID when unset.",
    )
    opt.add_argument(
        "--robustness-disable-local-probe",
        dest="robustness_disable_local_probe",
        action="store_true",
        default=None,
        help="Force disable_local_probe=true. The robustness-agent silences "
             "its LocalProbe fallback so per-pod sandbox checks (ps, rocm-smi, "
             "local HTTP) cannot emit false-positive symptoms.",
    )
    opt.add_argument(
        "--no-robustness-disable-local-probe",
        dest="robustness_disable_local_probe",
        action="store_false",
        help="Force disable_local_probe=false (keep the LocalProbe fallback "
             "even in multi-node mode).",
    )
    opt.add_argument(
        "--robustness-enable-cluster-pod-metrics",
        dest="robustness_enable_cluster_pod_metrics",
        action="store_true",
        default=None,
        help="Force enable_cluster_pod_metrics=true so the robustness-agent "
             "fans out per-pod metrics through robustness-server and feeds "
             "the local_health rules with cluster-decoded GPU snapshots.",
    )
    opt.add_argument(
        "--no-robustness-enable-cluster-pod-metrics",
        dest="robustness_enable_cluster_pod_metrics",
        action="store_false",
        help="Force enable_cluster_pod_metrics=false.",
    )
    opt.add_argument(
        "--robustness-pod-metrics-categories",
        dest="robustness_pod_metrics_categories",
        type=str,
        default=None,
        help="Comma-separated metric categories forwarded into "
             "pod_metrics_categories (e.g. 'gpu,memory'). Default 'gpu' is "
             "applied by the runtime when this flag is omitted.",
    )
    opt.add_argument("--orch-prompt", type=str, default=None,
                      help="Override Orchestration system prompt (file path or inline)")
    opt.add_argument("--critic-prompt", type=str, default=None,
                      help="Override Critic system prompt")
    opt.add_argument("--kernel-prompt", type=str, default=None,
                      help="Override Kernel system prompt")
    # Cortex KB integration flags
    # Defaults wire Cortex on. --degraded-kb bypasses KB hooks; --cortex-kb-url overrides $CORTEX_KB_URL;
    # --cortex-strict-fingerprint requires the manifest stack_fingerprint to match before warm_start.
    opt.add_argument(
        "--cortex-kb-url",
        dest="cortex_kb_url",
        type=str,
        default=None,
        help="Remote recipe-snapshot KB URL (read-only) for this run; "
             "also settable via $CORTEX_KB_URL. Leave it UNSET to run "
             "fully local — there is no default endpoint, so the "
             "optimizer never connects to a remote KB unless you pass "
             "this explicitly. Writes always go to --local-kb-root "
             "regardless; an explicitly-configured but unreachable URL "
             "degrades the dispatcher to local-only transparently (no "
             "need to also pass --degraded-kb).",
    )
    opt.add_argument(
        "--local-kb-root",
        dest="local_kb_root",
        type=str,
        default=None,
        help="Filesystem root for the local recipe-snapshot KB store. "
             "All writes (put_recipe / append_attempt / delete_recipe) go "
             "here regardless of --cortex-kb-url. Defaults to "
             "$HYPERLOOM_LOCAL_KB_ROOT, then $USER_DATA_PATH/kb, "
             "then /workspace/hyperloom/kb. Layout is a 5-level "
             "directory tree keyed by canonical_id components "
             "(model -> hardware -> framework -> framework_version -> "
             "precision); each leaf holds recipe.json + history/ + "
             "attempts.ndjson + .lock. See "
             "inference_optimizer/recipe_kb/local_store.py for the "
             "on-disk contract.",
    )
    opt.add_argument(
        "--degraded-kb",
        dest="degraded_kb",
        action="store_true",
        default=False,
        help="Skip the Cortex KB integration entirely (T0/T2/T3/T4 become "
             "no-ops). Also short-circuits the IR-3 KB probe. IR-3 sets "
             "this automatically when kb-service is unreachable (soft "
             "degrade); manifest records the reason as ``explicit_flag`` "
             "vs ``ir3_auto``.",
    )
    opt.add_argument(
        "--cortex-strict-fingerprint",
        dest="cortex_strict_fingerprint",
        action="store_true",
        default=False,
        help="When set, T0 refuses warm_start_recipe rows whose "
             "stack_fingerprint does not match the current pod (recorded "
             "in manifest.json). Default: lenient (M1 records the flag "
             "in manifest only; consumed by M5 specialist assembly).",
    )
    # Warm-recipe replay (PRELUDE auto-applies KB best_config before EXPLORE): --no-warm-replay disables;
    # --warm-replay-min-confidence (0.7) gates trigger tier; --warm-replay-min-reproduce-pct (0.8) gates reproduction.
    opt.add_argument(
        "--no-warm-replay",
        dest="no_warm_replay",
        action="store_true",
        default=False,
        help="Disable the PRELUDE auto-replay of KB warm-start "
             "``best_config``. The warm_start_recipe is still rendered "
             "into the specialist prompt as priors, but the Coordinator "
             "will NOT auto-run the historical best_config. Use this "
             "for cold debugging / ablation runs.",
    )
    opt.add_argument(
        "--warm-replay-min-confidence",
        dest="warm_replay_min_confidence",
        type=float,
        default=0.7,
        help="Minimum ``warm_start_recipe.confidence`` required to "
             "trigger the auto-replay. Default 0.7 means an ``exact`` "
             "5-tuple hit (conf 1.0) and a server-returned ``relative`` "
             "match (conf 0.7) both fire, while a ``miss`` (conf 0.0) "
             "does not. Raise it above 0.7 to require an exact hit "
             "before spending a verify on the warm config.",
    )
    opt.add_argument(
        "--warm-replay-min-reproduce-pct",
        dest="warm_replay_min_reproduce_pct",
        type=float,
        default=0.8,
        help="Minimum fraction of the recipe's recorded gain we need "
             "to reproduce to count as ``status=reproduced`` and push "
             "the warm config onto the optimization stack. Default "
             "0.8 — a recipe claiming +25%% counts if we measure "
             "+20%% or more. Below the threshold we record "
             "``status=drift`` and continue with the regular EXPLORE "
             "flow without inheriting the warm config.",
    )
    # PR Monitor REST + MCP
    # --pr-monitor-url overrides the in-cluster default (port-forward when outside the primus-cortex namespace);
    # --degraded-pr makes pr_feed_warm a no-op and strips mcp__pr_monitor__* from the specialist whitelist.
    opt.add_argument(
        "--pr-monitor-url",
        dest="pr_monitor_url",
        type=str,
        default=None,
        help="Override PR Monitor REST URL for this run. Default: "
             "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local"
             "/v1 (env: PR_MONITOR_URL). Pair with --pr-monitor-mcp-url "
             "when port-forwarding for local debug.",
    )
    opt.add_argument(
        "--pr-monitor-mcp-url",
        dest="pr_monitor_mcp_url",
        type=str,
        default=None,
        help="Override PR Monitor MCP URL handed to specialist LLM "
             "backends. Default mirrors --pr-monitor-url with /mcp/ "
             "suffix; the trailing slash is mandatory.",
    )
    opt.add_argument(
        "--degraded-pr",
        dest="degraded_pr",
        action="store_true",
        default=False,
        help="Disable the PR Monitor integration entirely. "
             "pr_feed_warm returns empty; the specialist tool "
             "whitelist drops mcp__pr_monitor__* tools. Short-circuits "
             "the IR-3 PR Monitor probe; IR-3 sets this automatically "
             "when PR Monitor is unreachable (soft degrade).",
    )
    opt.add_argument(
        "--pr-feed-window-days",
        dest="pr_feed_window_days",
        type=int,
        default=int(
            os.environ.get("PR_FEED_WINDOW_DAYS", "30") or "30"
        ),
        help="Look-back window for the PR feed warmup (days). "
             "Default: 30.",
    )
    # specialist research_lane capacity
    # --research-lane-capacity locks concurrent LLM specialists (0=no dispatch, ceiling=2×GPU, clamped).
    # Locked at session start (manifest + SharedState); PolicyGate denies mid-flight mutation.
    opt.add_argument(
        "--research-lane-capacity",
        dest="research_lane_capacity",
        type=int,
        default=_default_research_lane_capacity(),
        help="Max concurrent LLM specialist sub-agents on the "
             "research_lane. 0 disables specialist "
             "dispatch entirely (degrades to LLM-direct grid). The "
             "default is the research-lane ceiling (2 x visible GPU "
             "count, falling back to a conservative value when no GPU "
             "is detected); values above the ceiling are silently "
             "clamped down. Locked at session start.",
    )
    opt.add_argument(
        "--gpu-specialist-capacity",
        dest="gpu_specialist_capacity",
        type=int,
        default=int(
            os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "0")
            or "0"
        ),
        help="Number of GPUs available to specialists that request "
             "needs_gpu=true. 0 disables GPU specialists (default). "
             "Set INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES to a "
             "comma-separated GPU id pool when the specialist pool should "
             "not use device ids 0..N-1. Locked at session start.",
    )
    # Advisory specialist-proposal scorer (ProposalScorer): scores each proposal_set with gateway models
    # (0-10 + reason) as one reference for Orchestration; never gates. Add a model by appending its slug.
    opt.add_argument(
        "--proposal-scorer-models",
        dest="proposal_scorer_models",
        type=str,
        default=",".join(DEFAULT_SCORER_MODELS),
        help="Comma-separated gateway model slugs that independently "
             "score each specialist proposal_set (advisory only; never "
             "gates; rater identities are anonymized in the orchestration "
             "prompt). Default 'claude-opus-4-8,gpt-5.5,"
             "dvue-aoai-005-Kimi-K2.6,gemini/gemini-3.1-pro-preview'. "
             "Add a model by "
             "appending its slug. Empty list disables scoring.",
    )
    opt.add_argument(
        "--no-proposal-scoring",
        dest="no_proposal_scoring",
        action="store_true",
        help="Disable the advisory specialist-proposal scorer entirely.",
    )
    # specialist sub-agent backend selection: Claude (default), inherits orchestration model; per-task caps bound LLM use.
    opt.add_argument(
        "--specialist-model",
        dest="specialist_model",
        type=str,
        default=os.environ.get("INFERENCE_OPTIMIZER_SPECIALIST_MODEL", "")
        or None,
        help="Claude model used for specialist sub-agents (defaults to "
             "the orchestration --claude-model). KB_design §3.5 §6.",
    )
    opt.add_argument(
        "--specialist-max-turns",
        dest="specialist_max_turns",
        type=int,
        default=int(
            os.environ.get(
                "INFERENCE_OPTIMIZER_SPECIALIST_MAX_TURNS",
                str(_DEFAULT_SPECIALIST_MAX_TURNS),
            )
            or _DEFAULT_SPECIALIST_MAX_TURNS
        ),
        help="Hard cap on LLM turns per specialist task (KB_design "
             "§3.5 §6). On exhaustion the runner synthesises an empty "
             "specialist_done (Inv-5.3).",
    )
    opt.add_argument(
        "--specialist-per-turn-max-seconds",
        dest="specialist_per_turn_max_seconds",
        type=float,
        default=float(
            os.environ.get(
                "INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS", "600"
            )
            or "600"
        ),
        help="Wall-clock cap per specialist turn (default 600s). Used "
             "by the robustness stale-scan to detect stuck specialists "
             ".",
    )
    # specialist dispatch shape
    opt.add_argument(
        "--specialist-dispatch-mode",
        dest="specialist_dispatch_mode",
        type=str,
        choices=("subprocess", "inprocess"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_DISPATCH_MODE", "subprocess",
        ).strip() or "subprocess",
        help="Specialist execution shape. 'subprocess' (default) spawns "
             "a fresh `claude` CLI per task inside a per-task git worktree "
             "for isolation (PR-A2). 'inprocess' keeps the legacy M5 path "
             "(claude-agent-sdk in the orchestrator process) for tests / "
             "environments without the claude binary.",
    )
    opt.add_argument(
        "--specialist-mcp-config",
        dest="specialist_mcp_config",
        type=str,
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_MCP_CONFIG", "",
        ).strip() or None,
        help="Optional path to an MCP config JSON forwarded to the "
             "subprocess claude (`--mcp-config`). Used to wire kb / pr "
             "MCP servers into specialists. Default: None.",
    )
    # Integration toggles. Roofline refresh is unconditional now (fires at PRELUDE and every 10%
    # cumulative_gain_validated crossing); the legacy composite/deny profile toggles are gone.
    def _env_default_on(env_var: str) -> bool:
        """Resolve a default-on boolean toggle from an environment variable.

        Args:
            env_var (str): The environment variable name to read.

        Returns:
            bool: ``False`` only when the variable is explicitly set to ``"0"``;
            ``True`` otherwise (including when unset).
        """
        return os.environ.get(env_var, "1").strip() != "0"

    opt.add_argument(
        "--allow-empty-kernel-shape",
        dest="allow_empty_kernel_shape",
        action="store_true",
        default=os.environ.get(
            "HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", "0",
        ).strip().lower() in {"1", "true", "yes", "on"},
        help="Escape hatch (default off): allow kernel optimization to "
             "dispatch a candidate with no trace-anchored shape. Normally "
             "a shapeless candidate is rejected with a structured error so "
             "the run returns to ``trace_analyze`` instead of burning a "
             "GEAK / OOB budget on an unanchored kernel. Env: "
             "HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE=1.",
    )
    opt.add_argument(
        "--enable-roofline",
        dest="enable_roofline",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_ENABLE_ROOFLINE"),
        help="Select which analysis action the Coordinator enqueues at "
             "PRELUDE bootstrap and on every +10%% watermark crossing. "
             "Default on: ``roofline`` (composite profile + "
             "trace_analyze + analysis.md). Pass ``--no-enable-roofline`` "
             "to use plain ``profile`` instead (lighter — captures the "
             "trace only, skips trace_analyze). Behaviour is otherwise "
             "identical (same idempotency keys, same pending-task "
             "dispatch gate, same watermark anchor update). Env: "
             "INFERENCE_OPTIMIZER_ENABLE_ROOFLINE=0.",
    )
    opt.add_argument(
        "--research-scout",
        dest="research_scout",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_RESEARCH_SCOUT"),
        help="Auto-dispatch a read-only research scout at PRELUDE (and "
             "every --research-scout-interval EXPLORE rounds) that "
             "collects proven priors — reference launch scripts, model "
             "config.json architecture features, and cross-framework / "
             "NVIDIA research — into ``research_hints.md`` and seeds "
             "high-priority gaps. Default on; pass ``--no-research-scout`` "
             "to disable the whole feature. Env: "
             "INFERENCE_OPTIMIZER_RESEARCH_SCOUT=0.",
    )
    opt.add_argument(
        "--research-scout-interval",
        dest="research_scout_interval",
        type=int,
        default=3,
        help="Re-dispatch the research scout every N EXPLORE rounds with "
             "the current bottleneck context (append-only). Default 3. "
             "Ignored when ``--no-research-scout`` is set.",
    )
    opt.add_argument(
        "--recipe-sediment",
        dest="recipe_sediment",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_RECIPE_SEDIMENT"),
        help="Sediment KEEP/REVERT provenance into the persistent recipe: "
             "KEEP optimizations traceable to a research hint carry their "
             "source + measured gain into ``what_worked``; REVERTs land in "
             "``what_failed`` so the next warm-start avoids re-testing them. "
             "Default on; pass ``--no-recipe-sediment`` to keep the recipe "
             "purely ephemeral. Env: INFERENCE_OPTIMIZER_RECIPE_SEDIMENT=0.",
    )
    opt.add_argument(
        "--target-advisory",
        dest="target_advisory",
        action=argparse.BooleanOptionalAction,
        default=_env_default_on("INFERENCE_OPTIMIZER_TARGET_ADVISORY"),
        help="Inject an advisory 'External target gap' block (throughput / "
             "TPOT / interactivity gap vs the LLM-authored competitor "
             "target) into the orchestration and specialist prompts; when "
             "the TPOT ratio dominates it nudges toward latency-reducing "
             "directions. Advisory only — never gates Objective or scoring. "
             "Default on; pass ``--no-target-advisory`` to disable. Env: "
             "INFERENCE_OPTIMIZER_TARGET_ADVISORY=0.",
    )
    # Post-optimization concurrency sweep (on by default): a baseline-vs-optimized Magpie grid across CONC
    # values, output to reports/conc_sweep_summary.json (see orchestrator/conc_sweep.py). Bounded by
    # --conc-sweep-total-budget-sec; skip conditions short-circuit. Disable via --no-enable-conc-sweep.
    opt.add_argument(
        "--enable-conc-sweep",
        dest="enable_conc_sweep",
        action=argparse.BooleanOptionalAction,
        default=(
            os.environ.get(
                "INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP", "",
            ).strip().lower() not in ("0", "false", "no", "off")
        ),
        help="Run a post-optimization concurrency sweep (baseline vs "
             "current_best across CONC) and write "
             "reports/conc_sweep_summary.json + conc_sweep_raw.csv. "
             "On by default; disable with --no-enable-conc-sweep or "
             "INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP=0.",
    )
    opt.add_argument(
        "--conc-sweep-concs",
        dest="conc_sweep_concs",
        type=str,
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS", "1,2,4,8,16,32,64,128",
        ),
        help="Comma-separated CONC ladder for --enable-conc-sweep. "
             "Default 1,2,4,8,16,32,64,128.",
    )
    opt.add_argument(
        "--conc-sweep-timeout-sec",
        dest="conc_sweep_timeout_sec",
        type=int,
        default=int(os.environ.get(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_TIMEOUT_SEC", "1800",
        )),
        help="Per-variant timeout (seconds) for --enable-conc-sweep. "
             "Default 1800 (~30 min). Per-variant cap is also clamped "
             "by the remaining --conc-sweep-total-budget-sec.",
    )
    opt.add_argument(
        "--conc-sweep-total-budget-sec",
        dest="conc_sweep_total_budget_sec",
        type=int,
        default=int(os.environ.get(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_TOTAL_BUDGET_SEC", "9000",
        )),
        help="Total wall-clock budget (seconds) for the whole conc-sweep "
             "action, independent of the per-variant Magpie timeout. "
             "Once exhausted, remaining variants are recorded as "
             "status=skipped / error_class=budget_exhausted and the "
             "JSON envelope carries budget_exhausted=true. Default 9000 "
             "(~2.5h); set to 0 to disable. Also bounded above by the "
             "main session wall-clock deadline since conc_sweep runs as "
             "a SWEEP-phase action.",
    )
    # Retired flags operator scripts may still pass; hard-fail at argparse with a migration hint, not a silent alias.
    _retired_hint = (
        "Use ``--enable-roofline`` (default on) / ``--no-enable-roofline`` "
        "instead. The PRELUDE-initial analysis task is unconditional and "
        "the composite/direct-profile bifurcation has been removed."
    )
    for _retired in (
        "--use-roofline-composite",
        "--no-use-roofline-composite",
        "--deny-direct-profile",
        "--no-deny-direct-profile",
        "--force-roofline-after-baseline",
        "--no-force-roofline-after-baseline",
    ):
        opt.add_argument(
            _retired,
            action=_RetiredFlag,
            hint=_retired_hint,
        )
    # Per-variant explore overtime kill ratio (mirrored to SharedState.explore_overtime_kill_ratio).
    # Default 1.10: kill a single-variant run once wall-clock exceeds baseline by +10% (outcome=KILLED_OVERTIME).
    # 0 disables (legacy variant_timeout_sec hard cap still applies); gate skips the inlined stack rebench (Q4).
    def _env_float_or(default: float, env_var: str) -> float:
        """Resolve a float CLI default from an environment variable.

        Args:
            default (float): Value to use when the variable is unset or invalid.
            env_var (str): The environment variable name to read.

        Returns:
            float: The parsed env value, or ``default`` on absence / parse error.
        """
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _env_int_or(default: int, env_var: str) -> int:
        """Resolve an int CLI default from an environment variable.

        Args:
            default (int): Value to use when the variable is unset or invalid.
            env_var (str): The environment variable name to read.

        Returns:
            int: The parsed env value, or ``default`` on absence / parse error.
        """
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)
    opt.add_argument(
        "--explore-overtime-kill-ratio",
        dest="explore_overtime_kill_ratio",
        type=float,
        default=_env_float_or(
            1.10, "INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO",
        ),
        help="Per-variant explore overtime kill: each single-variant "
             "Magpie run in the explore loop is reaped once its "
             "wall-clock exceeds ``baseline_runtime_sec * RATIO``. The "
             "variant is recorded with outcome=KILLED_OVERTIME + "
             "runtime_sec + wall_clock_ratio_vs_baseline (no tput) so "
             "the LLM can distinguish it from a hard timeout / crash. "
             "Default 1.10 (kill at +10%% over baseline wall-clock). "
             "Pass 0 to disable. Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO.",
    )
    # Explore variant hard timeout — operator override for the auto-derived cap.
    # ExploreExecutor auto-derives from baseline_runtime_sec*(kill_ratio+0.5); 0 (default) keeps auto-derive.
    # Mirrored to SharedState.explore_variant_timeout_sec_override (injected as params['variant_timeout_sec']).
    opt.add_argument(
        "--explore-variant-timeout-sec",
        dest="explore_variant_timeout_sec",
        type=int,
        default=_env_int_or(
            0, "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC",
        ),
        help="Pin the per-variant hard timeout (seconds) inside the "
             "EXPLORE phase. ``0`` (default) auto-derives from "
             "``baseline_runtime_sec * (--explore-overtime-kill-ratio + "
             "--explore-variant-timeout-safety-margin)`` once baseline "
             "lands, with a 2400-14400 s range guard. Set to a positive "
             "integer to pin (CI smoke runs / debugging). Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC.",
    )
    opt.add_argument(
        "--explore-variant-timeout-safety-margin",
        dest="explore_variant_timeout_safety_margin",
        type=float,
        default=_env_float_or(
            0.5, "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN",
        ),
        help="Headroom (as a fraction of baseline_runtime_sec) added on "
             "top of --explore-overtime-kill-ratio when the EXPLORE hard "
             "cap is auto-derived. Default 0.5 (≈ 50%% of baseline as "
             "buffer for variant cold starts: torch.compile AOTI compile, "
             "fresh aiter shapes, spec-decoding draft load). Bump for "
             "workloads with heavy compile cost; lower to tighten the "
             "backstop. No effect when --explore-variant-timeout-sec is "
             "set to a positive value. Env: "
             "INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN.",
    )
    # drop scoreboard
    # Legacy action_scores is retired; flag controls a resumed session's leftover scoreboard:
    # drop (default) strips the fields silently; warn additionally logs + adds a breakdown.warnings entry.
    opt.add_argument(
        "--legacy-action-scores",
        dest="legacy_action_scores",
        type=str,
        choices=("drop", "warn"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "drop",
        ).strip() or "drop",
        help="Resume-mode handling of the legacy scoreboard "
             "(``action_scores`` and friends). 'drop' (default) "
             "silently discards. 'warn' logs a WARNING + adds a "
             "breakdown.warnings entry. KB_design §3.9 §7.",
    )
    # SharedState evolution
    # --migration-mode: strict (default) makes a missing fact-layer field in a non-empty state.json fatal (exit 1);
    # lenient downgrades to WARNING. --reset-state backs up state.json and starts fresh (Cortex KB untouched).
    opt.add_argument(
        "--migration-mode",
        dest="migration_mode",
        type=str,
        choices=("strict", "lenient"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_MIGRATION_MODE", "strict",
        ).strip() or "strict",
        help="Strictness of the legacy → v0.8 state.json migration. "
             "'strict' (default) aborts on fact-layer field loss; "
             "'lenient' logs WARNING and continues. KB_design §3.10 §5.3.",
    )
    opt.add_argument(
        "--reset-state",
        dest="reset_state",
        action="store_true",
        default=False,
        help="Back up the existing ``state.json`` (if any) to "
             "``state.json.preReset.<unix_ts>`` and start the session "
             "from a blank SharedState. Cortex KB is NOT touched. "
             "KB_design §3.10 §5.3.",
    )
    # observability
    # --breakdown-include-transcripts: inline specialist transcript bodies (true) or path-only (false, default).
    opt.add_argument(
        "--breakdown-include-transcripts",
        dest="breakdown_include_transcripts",
        type=str,
        choices=("true", "false"),
        default=os.environ.get(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "false",
        ).strip().lower() or "false",
        help="Inline specialist transcript bodies into "
             "``specialist_runs`` (true) or reference them by path "
             "only (false, default). KB_design §3.12 §7.",
    )
    # plateau threshold tuning
    # Swap library default plateau thresholds; land in SharedState.plateau_overrides, locked at session start.
    opt.add_argument(
        "--plateau-explore-keep-gain",
        dest="plateau_explore_keep_gain",
        type=float,
        default=None,
        help="EXPLORE plateau: max cumulative KEEP-gain (%%) across the "
             "lookback window below which the AND condition fires. "
             "Default 0.5.",
    )
    opt.add_argument(
        "--plateau-explore-empty-streak",
        dest="plateau_explore_empty_streak",
        type=int,
        default=None,
        help="EXPLORE plateau: required count of *consecutive* specialist "
             "rounds with empty proposal_set before the AND condition "
             "fires. Default 3.",
    )
    opt.add_argument(
        "--plateau-explore-lookback",
        dest="plateau_explore_lookback",
        type=int,
        default=None,
        help="EXPLORE plateau: number of trailing rounds the gain sum is "
             "computed over. Default 5.",
    )
    opt.add_argument(
        "--plateau-kernel-revert-streak",
        dest="plateau_kernel_revert_streak",
        type=int,
        default=None,
        help="KERNEL plateau: consecutive REVERT / NEEDS_REVIEW integrate "
             "attempts to count as plateau (one half of the OR). "
             "Default 3.",
    )
    opt.add_argument(
        "--plateau-kernel-keep-gain",
        dest="plateau_kernel_keep_gain",
        type=float,
        default=None,
        help="KERNEL plateau: max cumulative KEEP-gain (%%) across the "
             "lookback window below which the OR fires. Default 0.5.",
    )
    opt.add_argument(
        "--plateau-kernel-lookback",
        dest="plateau_kernel_lookback",
        type=int,
        default=None,
        help="KERNEL plateau: number of trailing integrate attempts the "
             "gain sum is computed over. Default 5.",
    )
    # IR-6 — EXPLORE HARD force-exit thresholds
    # Either condition fires explore_force_exit_low_budget (EXPLORE→KERNEL/SWEEP); non-negotiable, locked at start.
    opt.add_argument(
        "--explore-force-exit-hours-remaining",
        dest="explore_force_exit_hours_remaining",
        type=float,
        default=None,
        help="EXPLORE force-exit: total wall-clock remaining (hours) "
             "below which EXPLORE exits immediately to the next phase, "
             "regardless of plateau / steward. Default 3.0 (IR-6).",
    )
    opt.add_argument(
        "--explore-force-exit-budget-pct",
        dest="explore_force_exit_budget_pct",
        type=float,
        default=None,
        help="EXPLORE force-exit: phase-budget remaining fraction "
             "(0..1) below which EXPLORE exits immediately. Default "
             "0.20 (IR-6).",
    )
    # phase budget percentages
    # Each phase claims a fraction of the wall-clock budget (caps; may exit earlier). Sum need not equal 1.0.
    opt.add_argument(
        "--max-minutes-prelude-pct",
        dest="phase_budget_prelude_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for PRELUDE as a fraction of "
             "--max-hours. Default: 0.03.",
    )
    opt.add_argument(
        "--max-minutes-explore-pct",
        dest="phase_budget_explore_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for EXPLORE. Default: 0.45.",
    )
    opt.add_argument(
        "--max-minutes-kernel-pct",
        dest="phase_budget_kernel_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for KERNEL. Default: 0.38.",
    )
    opt.add_argument(
        "--max-minutes-sweep-pct",
        dest="phase_budget_sweep_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for SWEEP. Default: 0.12.",
    )
    opt.add_argument(
        "--max-minutes-close-pct",
        dest="phase_budget_close_pct",
        type=float,
        default=None,
        help="Wall-clock budget cap for CLOSE. Default: 0.02.",
    )
    opt.add_argument(
        "--strict-phase",
        dest="strict_phase",
        action="store_true",
        default=True,
        help="(v0.8 M2 default) Enforce PolicyGate R1 phase_incompatible. "
             "Action proposals outside the current phase's allowlist "
             "return policy_denied so the LLM self-corrects.",
    )
    opt.add_argument(
        "--no-strict-phase",
        dest="strict_phase",
        action="store_false",
        help="Disable R1 enforcement (warn-only). Useful for "
             "back-compat smoke tests; production should stay strict.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments and dispatch the requested subcommand.

    Configures logging from the ``-v`` count, resolves any ``--*-prompt`` flag
    that points at a file (reading its contents in place), and runs the
    ``optimize`` subcommand via :func:`asyncio.run`. Prints help and returns a
    non-zero code for unknown commands.

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        int: The process exit code (``optimize`` result, or ``2`` for no/unknown
        command).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.command == "optimize":
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt", "kernel_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        return asyncio.run(_run_optimize(args))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
