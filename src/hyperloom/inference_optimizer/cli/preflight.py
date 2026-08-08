# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI ``_preflight`` cluster — auto-install/env-hygiene checks run before ``optimize`` starts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_dotenv_key,
    is_allowed_kernel_agent_env_key,
)
from hyperloom.common.llm_config import (
    LEGACY_DEEPSEEK_ENV_KEYS,
    deepseek_compat_env,
    provider_model_defaults,
)

from .credentials import (
    _is_stale_proxy_url,
    _resolve_llm_endpoints,
    _reset_claude_config_to_upstream,
    _sync_geak_config_base_url,
    _validate_credentials,
)
from ..session.paths import (
    DEFAULT_SESSION_DIR,
    ENV_USER_DATA_PATH,
    find_latest_per_session_dir,
    session_dir as _session_dir_resolve,
    workspace_root as _workspace_root_resolve,
)

log = logging.getLogger("hyperloom.inference_optimizer.cli")

_PROVIDER_FALLBACK_KEYS: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "LLM_GATEWAY_KEY",
    "GEAK_BASE_URL",
    "LLM_API_BASE",
    # Legacy: not consumed anymore, still stripped if present.
    "SAFE_API_KEY",
    # A retired DeepSeek config normalizes to BOTH protocol sides, so it is
    # stripped in either single-provider mode: neither an Anthropic-only nor an
    # OpenAI-only shell may acquire the other side from a stale .env.
    *LEGACY_DEEPSEEK_ENV_KEYS,
)

_ANTHROPIC_FALLBACK_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    *LEGACY_DEEPSEEK_ENV_KEYS,
)


def _resolve_dotenv_file() -> Path | None:
    """Resolve the trusted repo ``.env`` file without trusting arbitrary cwd."""
    explicit_root = os.environ.get("REPO_ROOT", "").strip()
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    else:
        # Development checkout: preflight.py -> cli -> inference_optimizer ->
        # hyperloom -> src -> repo root.
        package_root = Path(__file__).resolve().parents[4]
        candidates.append(package_root)
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").is_file() and (cwd / "src" / "hyperloom").is_dir():
            candidates.append(cwd)
    for root in candidates:
        env_file = root / ".env"
        if env_file.is_file():
            return env_file
    return None


def _provider_only_mode() -> str:
    """Detect explicit single-provider intent from the current environment.

    Runs ahead of :func:`_normalize_legacy_deepseek_env`, so a retired
    ``DEEPSEEK_*`` shell export is still read here and counts as Anthropic-side
    intent.
    """
    has_anthropic = bool(
        os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_BASE_URL")
    )
    has_openai = bool(os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_KEY"))
    has_gateway = bool(os.environ.get("LLM_GATEWAY_KEY"))
    if has_anthropic and not has_openai and not has_gateway:
        return "anthropic"
    if has_openai and not has_anthropic and not has_gateway:
        return "openai"
    return ""


def _normalize_legacy_deepseek_env() -> None:
    """Rewrite a retired ``DEEPSEEK_*`` configuration into the standard variables.

    DeepSeek serves the Anthropic protocol on ``/anthropic`` and the OpenAI
    protocol on ``/v1`` from one gateway with one key, so it is expressed with
    ``ANTHROPIC_*`` + ``OPENAI_*`` like any other dual-protocol gateway. This is
    the only place in the runtime that reads the retired variables; everything
    downstream sees just the two protocol sides.
    """
    updates = deepseek_compat_env()
    if updates:
        for key, value in updates.items():
            os.environ[key] = value
        print(
            "Preflight: DEEPSEEK_* is deprecated; normalized to "
            f"{', '.join(sorted(updates))}. Re-run setup to migrate your .env."
        )
    # A gateway that serves only its own models supplies the model ids too.
    # Exported (not just resolved) so subprocesses, GEAKv4 and the kernel-agent
    # installer inherit them instead of falling back to an AMD Claude id.
    for key, value in provider_model_defaults().items():
        os.environ[key] = value
        print(f"Preflight: {key} <unset> -> {value} (implied by the configured gateway)")


def _restore_provider_only_mode(provider_mode: str, snapshot: dict[str, str | None]) -> None:
    """Undo cross-provider credentials injected by the installer env file.

    Symmetric: a single-provider run keeps the other side exactly as the shell
    and ``.env`` left it, so a stale installer env file cannot turn a valid
    configuration into a mispaired one.
    """
    if provider_mode == "anthropic":
        keys: tuple[str, ...] = _PROVIDER_FALLBACK_KEYS
    elif provider_mode == "openai":
        keys = _ANTHROPIC_FALLBACK_KEYS
    else:
        return
    for key in keys:
        original = snapshot.get(key)
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


# /dev/shm threshold: below this, a launch can collide with stale vLLM/NCCL shm segments.
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

    Always parses ``.env`` and loads any key not already present in the
    environment, regardless of whether LLM credentials are already set (so
    operational vars like ``TRACELENS_ROOT`` / ``FORGE_PATH`` are also picked up).
    """
    env_file = _resolve_dotenv_file()
    if env_file is None:
        return
    parsed: dict[str, str] = {}
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
        parsed[key] = value
    safe_vars, dropped_vars = filter_untrusted_env_mapping(
        parsed,
        allow_predicate=is_allowed_dotenv_key,
    )
    for key in dropped_vars:
        print(f"Preflight: WARNING — ignoring unsupported .env key {key} from {env_file}", file=sys.stderr)
    for key, value in safe_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    if loaded:
        print(f"Preflight: loaded {loaded} missing var(s) from {env_file} (env wins)")


def _prepend_path(var: str, entry: str) -> None:
    """Prepend ``entry`` to a ``:``-separated env var, skipping if already leading."""
    if not entry:
        return
    current = os.environ.get(var, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if parts and parts[0] == entry:
        return
    parts = [entry] + [p for p in parts if p != entry]
    os.environ[var] = os.pathsep.join(parts)


def _derive_runtime_paths() -> None:
    """Rebuild PATH / LD_LIBRARY_PATH from .env-loaded roots (replaces hyperloom.env.sh).

    The bare-metal installer no longer writes a sourceable combined env; PATH-class
    values are derived here from VIRTUAL_ENV / ROCM_PATH / VLLM_VENV_ROOT so the
    single .env source stays authoritative. Order mirrors the former script:
    venv, then ROCm, then the isolated vLLM venv (last prepended wins).
    """
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv:
        _prepend_path("PATH", str(Path(venv) / "bin"))
    rocm = os.environ.get("ROCM_PATH", "")
    if rocm:
        _prepend_path("PATH", str(Path(rocm) / "bin"))
        _prepend_path("LD_LIBRARY_PATH", str(Path(rocm) / "lib"))
    vllm_root = os.environ.get("VLLM_VENV_ROOT", "")
    if vllm_root:
        _prepend_path("PATH", str(Path(vllm_root) / "bin"))


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
    to correct a stale/invalid inherited TRACELENS_ROOT. Hard-fail contract
    (root unset only): sys.exit(2) if missing/0-vars/still-unset.
    """
    candidate = os.environ.get("KERNEL_AGENT_ENV")
    if not candidate:
        user_data = os.environ.get("USER_DATA_PATH")
        if user_data:
            candidate = str(Path(user_data) / "runtime" / "kernel-agent.env.sh")

    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        # Root is set: no bootstrap, but still correct invalid path vars from the
        # env file when resolvable.
        if not candidate:
            return
        env_path = Path(candidate)
        if not env_path.is_file():
            return
        try:
            text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        file_vars, dropped_file_vars = filter_untrusted_env_mapping(
            _parse_env_assignments(text),
            allow_predicate=is_allowed_kernel_agent_env_key,
        )
        for key in dropped_file_vars:
            print(
                f"Preflight: WARNING — ignoring unsupported kernel-agent env key {key} from {env_path}",
                file=sys.stderr,
            )
        _correct_kernel_agent_path_vars(file_vars, env_path)
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
    parsed_file_vars = _parse_env_assignments(text)
    file_vars, dropped_file_vars = filter_untrusted_env_mapping(
        parsed_file_vars,
        allow_predicate=is_allowed_kernel_agent_env_key,
    )
    for key in dropped_file_vars:
        print(
            f"Preflight: WARNING — ignoring unsupported kernel-agent env key {key} from {env_path}",
            file=sys.stderr,
        )
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
        ("claude_agent_sdk", "claude-agent-sdk>=0.2.110"),
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


_RAY_VERSION = "2.44.1"
# Ray 2.44.1's CLI currently fails during import with click >= 8.3.0.
_RAY_CLI_CLICK_MAX_VERSION = "8.3.0"
_RAY_INSTALL_SPEC = f"ray[default]=={_RAY_VERSION}"
_CLICK_INSTALL_SPEC = f"click<{_RAY_CLI_CLICK_MAX_VERSION}"
_RAY_INSTALL_SPECS = (_RAY_INSTALL_SPEC, _CLICK_INSTALL_SPEC)


_RAY_SMOKE_TEMPLATE = r"""
import importlib.metadata as md
import re
import sys

RAY_VERSION = "__RAY_VERSION__"
RAY_CLI_CLICK_MAX_VERSION = "__RAY_CLI_CLICK_MAX_VERSION__"
RAY_CLI_CLICK_MAX_VERSION_TUPLE = __RAY_CLI_CLICK_MAX_VERSION_TUPLE__

def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3])

try:
    import ray
except Exception as exc:
    print(f"ray import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if ray.__version__ != RAY_VERSION:
    print(f"ray version mismatch: {ray.__version__} != {RAY_VERSION}", file=sys.stderr)
    raise SystemExit(1)

try:
    click_version = md.version("click")
except md.PackageNotFoundError:
    print("click is not installed", file=sys.stderr)
    raise SystemExit(1)

if _version_tuple(click_version) >= RAY_CLI_CLICK_MAX_VERSION_TUPLE:
    print(
        f"click version incompatible with Ray CLI: {click_version} >= {RAY_CLI_CLICK_MAX_VERSION}",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    from ray.scripts.scripts import main as _ray_cli_main  # noqa: F401
except Exception as exc:
    print(f"ray CLI import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
"""


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3])


_RAY_SMOKE = (
    _RAY_SMOKE_TEMPLATE.replace("__RAY_VERSION__", _RAY_VERSION)
    .replace("__RAY_CLI_CLICK_MAX_VERSION__", _RAY_CLI_CLICK_MAX_VERSION)
    .replace("__RAY_CLI_CLICK_MAX_VERSION_TUPLE__", repr(_version_tuple(_RAY_CLI_CLICK_MAX_VERSION)))
)


def _ray_smoke(python_exe: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python_exe, "-c", _RAY_SMOKE],
        capture_output=True,
        text=True,
    )


def _ensure_ray(python_exe: str, pip_extra: list[str]) -> None:
    """Probe-then-install Ray using the interpreter that will import it.

    Ray is used broadly (multi-node scheduling, kernel/profile/recover
    executors), not only by Magpie. The smoke test imports ``ray`` with
    ``python_exe``, checks the pinned Ray/click compatibility contract, and
    imports Ray's CLI module. A PATH-only ``which ray`` lookup, or even a bare
    ``import ray``, can false-positive when the CLI is broken by an incompatible
    click version.

    Args:
        python_exe (str): The interpreter that will import Ray (and run the
            probe / install).
        pip_extra (list[str]): Extra arguments threaded into the ``pip
            install`` invocation (e.g. ``--break-system-packages``).
    """
    check = _ray_smoke(python_exe)
    if check.returncode == 0:
        print("Preflight: ray OK")
        return
    reason = (check.stderr or check.stdout or "unknown Ray smoke failure").strip().splitlines()[-1]
    print(f"Preflight: ray/click invalid ({reason}), installing {_RAY_INSTALL_SPEC} + {_CLICK_INSTALL_SPEC} ...")
    subprocess.run(
        [python_exe, "-m", "pip", "install", "--quiet", *pip_extra, *_RAY_INSTALL_SPECS],
        check=True,
    )
    check = _ray_smoke(python_exe)
    if check.returncode != 0:
        reason = (check.stderr or check.stdout or "unknown Ray smoke failure").strip()
        raise RuntimeError(f"Ray install completed but smoke test still failed: {reason}")
    print("Preflight: ray installed OK")


# InferenceX benchmark_serving client-side deps. Mirrors the ``_BENCH_SERVING_DEPS``
# list in assets/install.sh (keep in sync). ``benchmark_serving.py`` lives under
# InferenceX (not Magpie's pyproject), so installing Magpie never pulls
# these; every bypass/Magpie client launch imports them before hitting the server.
_BENCH_SERVING_DEPS = (
    "aiohttp",
    "tqdm",
    "numpy",
    "requests",
    "transformers",
    "huggingface_hub",
    "datasets",
    "pandas",
)


def _ensure_bench_serving_deps(python_exe: str, pip_extra: list[str]) -> None:
    """Probe-then-install the InferenceX benchmark_serving client deps in python_exe.

    ``assets/install.sh:ensure_bench_serving_deps`` installs these into the
    install-time ``$PYTHON`` (often ``/opt/venv``). The bypass runner, however,
    launches ``benchmark_serving.py`` with the ACTIVE benchmark interpreter
    (``sys.executable``); when that differs from the install-time interpreter the
    client dies with a missing-module error even though Ray reported OK. Route the
    ensure through ``python_exe`` (the resolved benchmark interpreter) so the two
    stay aligned. Detection uses ``importlib.util.find_spec`` (no heavy import) in
    a single probe subprocess, then pip-installs only the missing modules.

    Args:
        python_exe (str): The interpreter that will import the client deps.
        pip_extra (list[str]): Extra ``pip install`` arguments (e.g.
            ``--break-system-packages``).
    """
    mods = list(_BENCH_SERVING_DEPS)
    probe = (
        "import importlib.util, sys; print('\\n'.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
    )
    result = subprocess.run([python_exe, "-c", probe, *mods], capture_output=True, text=True)
    if result.returncode != 0:
        # Probe itself failed unexpectedly; fall back to attempting all so a
        # genuinely missing client is not silently left uninstalled.
        missing = mods
    else:
        missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if not missing:
        print("Preflight: benchmark_serving client deps OK")
        return
    print(f"Preflight: installing benchmark_serving client deps: {' '.join(missing)} ...")
    subprocess.run(
        [python_exe, "-m", "pip", "install", "--quiet", "--no-cache-dir", *pip_extra, *missing],
        check=True,
    )
    print("Preflight: benchmark_serving client deps installed OK")


def _ensure_framework_deps(args, python_exe: str, pip_extra: list[str]) -> None:
    """Install the selected framework's declared runtime deps into python_exe.

    Resolution mirrors the CLI's own order (``--framework`` > ``$FRAMEWORK`` >
    default) rather than reading ``os.environ["FRAMEWORK"]``, which the CLI only
    pins after preflight has run.

    Args:
        args: Parsed CLI namespace; only ``framework`` is read.
        python_exe (str): Interpreter the benchmark imports these from.
        pip_extra (list[str]): Extra ``pip install`` arguments.
    """
    from hyperloom.inference_optimizer import framework_deps, framework_registry

    framework = (
        getattr(args, "framework", None) or os.environ.get("FRAMEWORK", "")
    ).strip().lower() or framework_registry.DEFAULT_FRAMEWORK
    try:
        outcome = framework_deps.ensure(
            framework, python_exe=python_exe, pip_extra=tuple(pip_extra)
        )
    except framework_deps.TorchClobberedError as exc:
        print(f"Preflight: FATAL {exc}", file=sys.stderr)
        sys.exit(2)
    # Frameworks that ship no manifest are the common case; stay quiet unless
    # the manifest actually asked for something or was partly rejected.
    if outcome.skipped_reason and not (outcome.refused or outcome.invalid):
        return
    framework_deps.report(outcome, prefix="Preflight: framework deps")


# RUN_EVAL values that disable the accuracy gate (mirrors _workload_envs).
_RUN_EVAL_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})

# Probed one subprocess each: the base package and the [api] extra can arrive
# from different places (image vs pip), and only the truly absent one is
# installed. Order matters only for the log line.
_LM_EVAL_DEPS = ("lm_eval", "tenacity")


def _probe_missing_lm_eval_deps(python_exe: str) -> list[str] | None:
    """Report which accuracy-gate modules ``python_exe`` cannot import.

    One subprocess per module, on purpose. Importing ``lm_eval`` pulls in torch,
    which on a broken ROCm install can take the interpreter down with a signal
    rather than an exception; a shared probe would lose the verdict for every
    other module along with it. Separate probes also distinguish "no lm_eval at
    all" from "the image ships lm_eval and only the extra is absent", which
    decides whether the image's own build gets reinstalled over.

    A real import rather than ``find_spec``, so a half-installed package that
    fails at exec time counts as missing.

    Args:
        python_exe (str): The interpreter to probe.

    Returns:
        list[str] | None: The modules that failed to import, or ``None`` when
            the interpreter could not run the probe at all, leaving absence
            unproven.
    """
    try:
        # A missing or non-executable interpreter raises instead of returning a
        # code, and preflight must not die on it: absence stays unproven, which
        # the caller reports without touching anything.
        liveness = subprocess.run([python_exe, "-c", "pass"], capture_output=True)
        if liveness.returncode != 0:
            return None
        missing: list[str] = []
        for module in _LM_EVAL_DEPS:
            probe = subprocess.run([python_exe, "-c", f"import {module}"], capture_output=True)
            if probe.returncode != 0:
                missing.append(module)
    except OSError:
        return None
    return missing


# The harness the single-node path ends up on: InferenceX's benchmark_lib.sh
# force-reinstalls this commit over whatever pip resolved. Pinning the same one
# keeps a multi-node accuracy number comparable with a single-node one instead
# of silently measuring against whatever PyPI happens to serve that day.
_LM_EVAL_PINNED_REF = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
_LM_EVAL_REPO = "github.com/EleutherAI/lm-evaluation-harness"
# git first, then the archive, because the sandbox may not ship a git binary.
_LM_EVAL_PINNED_SPECS = (
    ("git", f"lm_eval[api] @ git+https://{_LM_EVAL_REPO}.git@{_LM_EVAL_PINNED_REF}"),
    ("archive", f"lm_eval[api] @ https://{_LM_EVAL_REPO}/archive/{_LM_EVAL_PINNED_REF}.tar.gz"),
)
# Settled by install.sh (or the image) and load-bearing elsewhere in the stack:
# pandas for rocprof-compute's CSV converter, torch/triton for the ROCm build
# PyPI has no equivalent of, numpy because both pin against it.
_LM_EVAL_FROZEN_DEPS = ("torch", "pandas", "numpy", "triton")


def _frozen_constraints(python_exe: str) -> list[str]:
    """Pin the packages this install must not move, as ``pip -c`` arguments.

    ``install.sh`` orders itself so the ``pandas<3`` pin is the last pip step,
    on the stated grounds that "no later pip install can re-pull pandas>=3".
    This install runs at ``optimize`` time, which is after that last word, in
    the same interpreter, and resolves a dependency closure that reaches pandas
    through ``datasets`` and torch directly. Left free, pip may satisfy those
    from PyPI: pandas>=3 makes rocprof-compute drop every counter so forge
    degrades to PMC with no roofline, and a PyPI torch is a CUDA build on a ROCm
    box.

    Constraining rather than passing ``--no-deps`` keeps the rest of the closure
    resolvable -- lm_eval needs far more than the bench-serving set already
    installed -- while making an incompatible requirement a loud pip failure
    instead of a silently broken profiler.

    Args:
        python_exe (str): The interpreter being installed into.

    Returns:
        list[str]: ``["-c", <path>]``, or ``[]`` when nothing could be read.
    """
    pins: list[str] = []
    for name in _LM_EVAL_FROZEN_DEPS:
        probe = subprocess.run(
            [python_exe, "-c", f"import importlib.metadata as m; print(m.version({name!r}))"],
            capture_output=True,
            text=True,
            check=False,
        )
        version = probe.stdout.strip()
        if probe.returncode == 0 and version:
            pins.append(f"{name}=={version}")
    if not pins:
        print("Preflight: WARNING — could not read installed versions; lm_eval install is unconstrained")
        return []
    # The name deliberately carries no package name: this path is spliced into a
    # pip command line that callers assert does not mention a package spec.
    handle, path = tempfile.mkstemp(prefix="hyperloom_pip_constraints_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write("\n".join(pins) + "\n")
    print(f"Preflight: constraining the lm_eval install to {' '.join(pins)}")
    return ["-c", path]


def _install_pinned_lm_eval(python_exe: str, pip_extra: list[str]) -> None:
    """Install the pinned ``lm_eval[api]``, falling back to the source archive.

    Raises:
        subprocess.CalledProcessError: If every spec fails; the last error wins.
    """
    constraints = _frozen_constraints(python_exe)
    for i, (source, spec) in enumerate(_LM_EVAL_PINNED_SPECS):
        is_last = i == len(_LM_EVAL_PINNED_SPECS) - 1
        proc = subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet", "--no-cache-dir", *constraints, *pip_extra, spec],
            check=is_last,
        )
        if proc.returncode == 0:
            return
        print(f"Preflight: WARNING — pinned lm_eval via {source} failed; falling back")


def _resolved_eval_disabled(args: argparse.Namespace) -> bool:
    """Effective ``--no-eval`` for this launch, flag or persisted.

    Preflight runs before the ``--resume`` block reads ``state.json``, so a
    resume that inherits the flag instead of re-passing it must be read here.

    Args:
        args (argparse.Namespace): The parsed ``optimize`` args.

    Returns:
        bool: ``True`` when no accuracy eval will run in this session.
    """
    if bool(getattr(args, "no_eval", False)):
        return True
    if not bool(getattr(args, "resume", False)):
        return False
    raw = str(getattr(args, "resume_from", "") or "").strip()
    resumed = Path(raw).expanduser() if raw else find_latest_per_session_dir()
    if resumed is None:
        return False
    try:
        state = json.loads((resumed / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(state.get("eval_disabled"))


def _ensure_lm_eval_dep(python_exe: str, pip_extra: list[str], *, eval_disabled: bool = False) -> None:
    """Probe-then-install ``lm_eval`` in python_exe when the accuracy gate is on.

    ``install.sh`` defers ``lm_eval`` to InferenceX's ``benchmark_lib.sh`` runtime
    shim, but the multi-node ``magpie_bench_remote_compat`` client path runs
    ``python -m lm_eval`` directly without that shim. On those runs ``lm_eval`` is
    never installed, the eval subprocess dies with ``No module named lm_eval``,
    and every ``RUN_EVAL=true`` baseline aborts with ``baseline_accuracy_failed``.
    Ensure it here (preflight runs for all paths) with the benchmark interpreter.
    Skipped under ``--no-eval`` or an explicitly disabled RUN_EVAL.

    Multi-node only. Single-node reaches lm_eval through ``run_eval`` ->
    InferenceX ``benchmark_lib.sh::run_lm_eval``, which installs the harness on
    first use and then force-reinstalls its own pinned commit over whatever is
    there. Installing ahead of it therefore cannot change a single-node outcome,
    while ``check=True`` below would give a working run a new way to die, so
    single-node is left exactly as it was before this ensure existed.

    Even where it does run, it is a no-op on any interpreter that can already
    import the modules, so an image that ships ``lm_eval`` is left untouched
    rather than reinstalled over. When it does install, it pins
    :data:`_LM_EVAL_PINNED_REF` -- the commit InferenceX force-reinstalls on the
    single-node path -- so the two paths measure with the same harness.

    Args:
        python_exe (str): The interpreter that will run ``python -m lm_eval``.
        pip_extra (list[str]): Extra ``pip install`` arguments.
        eval_disabled (bool): ``--no-eval``; skip the ensure entirely.

    Raises:
        subprocess.CalledProcessError: If the install fails. Preflight aborts
            rather than continuing, since letting a failed install through
            reproduces the exact ``baseline_accuracy_failed`` this prevents,
            only hours later and with the pip diagnostics long gone.
    """
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if not is_multi_node():
        return  # single-node: InferenceX installs lm_eval itself, see above
    if eval_disabled:
        return  # --no-eval: no eval anywhere, so the harness is never loaded
    run_eval = os.environ.get("RUN_EVAL")
    if run_eval is not None and run_eval.strip().lower() in _RUN_EVAL_FALSE_VALUES:
        return  # accuracy gate disabled; lm_eval not required
    missing = _probe_missing_lm_eval_deps(python_exe)
    if missing is None:
        # Absence is unproven, so installing would be a guess that could replace
        # an lm_eval the image ships. An interpreter that cannot run ``python -c``
        # already breaks the benchmark far more loudly than a missing accuracy
        # gate would, so this warns and changes nothing.
        print("Preflight: WARNING — cannot run the lm_eval probe; leaving the interpreter untouched")
        return
    if not missing:
        print("Preflight: lm_eval[api] OK")
        return
    if "lm_eval" in missing:
        print(f"Preflight: installing lm_eval[api]@{_LM_EVAL_PINNED_REF[:12]} (accuracy gate) ...")
        _install_pinned_lm_eval(python_exe, pip_extra)
        print("Preflight: lm_eval[api] installed OK")
        return
    # The image already ships lm_eval. Install only the absent extra so pip
    # cannot resolve a different lm_eval build over the one baked in, which
    # would silently swap out a version the image pinned on purpose -- the same
    # reason the pinned spec above is not force-reinstalled here.
    targets = missing
    print(f"Preflight: installing {' '.join(targets)} (accuracy gate; missing: {' '.join(missing)}) ...")
    subprocess.run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            *_frozen_constraints(python_exe),
            *pip_extra,
            *targets,
        ],
        check=True,
    )
    print(f"Preflight: {' '.join(targets)} installed OK")


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
    """Best-effort informational check of visible GPU count vs ``$TP`` (silent when rocm-smi is absent).

    Skipped in external multi-node mode: there the server runs on remote GPU
    pods and the benchmark drives the frontend over HTTP, so this
    orchestrator-only sandbox legitimately has no local GPUs. Probing rocm-smi
    here would emit a misleading "0 GPUs; benchmark will fail" warning.

    The node count is part of that test, not just the hand-off URL. The platform
    exports one env block for single- and multi-node runs alike, so a stray
    ``HYPERLOOM_MN_EXT_SERVICE_URL`` must not silently disarm this check on a
    single-node run that really does need local GPUs.
    """
    # External multi-node: GPUs are on remote pods, not this sandbox.
    from hyperloom.inference_optimizer.multi_node._internal.external_state import external_service_url
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if is_multi_node() and external_service_url():
        print("Preflight: external multi-node mode; skipping local GPU visibility check (GPUs are on remote pods)")
        return
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
    # rocm-smi --showid emits multiple GPU[ lines per GPU; deduplicate by GPU index.
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
        usage = shutil.disk_usage("/dev/shm")  # nosec B108 - mountpoint probe, not temp file creation.
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
        f"then retry `python -m hyperloom.inference_optimizer.cli optimize`. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_tracelens_root_exists() -> None:
    """Hard-gate an explicitly set ``TRACELENS_ROOT`` at preflight.

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
    if anthropic_base_url:
        print(f"  ANTHROPIC_BASE_URL  = {anthropic_base_url}")
    else:
        print("  ANTHROPIC_BASE_URL  = <unset> — no LLM base URL resolved; Claude SDK will fail")
    if args is not None:
        kb_enabled = bool(getattr(args, "recipe_kb_enabled", True))
        pr_enabled = bool(getattr(args, "pr_monitor_enabled", True))
        kb_reason = getattr(args, "kb_degraded_reason", None) or "-"
        pr_reason = getattr(args, "pr_degraded_reason", None) or "-"
        kb_status = "OK" if kb_enabled else f"DEGRADED ({kb_reason})"
        pr_status = "OK" if pr_enabled else f"DEGRADED ({pr_reason})"
        print(f"  kb_status           = {kb_status}")
        print(f"  pr_monitor_status   = {pr_status}")
        print(f"  kb_degraded_reason  = {kb_reason}")
        print(f"  pr_degraded_reason  = {pr_reason}")

    # Surface Recipe KB offline-queue state; dead-letter pile-up signals a cold start.
    try:
        _print_recipe_kb_queue_status()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"  recipe_kb_queue     = <probe_failed: {exc!r}>")


def _print_recipe_kb_queue_status() -> None:
    """Emit a one-line summary of the Recipe KB offline NDJSON queue (dead-letter = permanent-reject signal).

    Note:
        Side-effecting: writes the queue status summary to stdout and returns
        nothing.
    """
    from ..session.session_paths import (
        recipe_kb_dead_letter_ndjson,
        recipe_kb_flushed_ndjson,
        recipe_kb_pending_ndjson,
    )

    sd = _session_dir_resolve()
    pending = recipe_kb_pending_ndjson(sd)
    dead = recipe_kb_dead_letter_ndjson(sd)
    flushed = recipe_kb_flushed_ndjson(sd)

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
    print(f"  recipe_kb_queue     = pending={p_n} dead_letter={d_n} flushed={f_n} (root={pending.parent})")
    if d_n > 0:
        print(
            f"                        ⚠ {d_n} dead-letter row(s) — "
            f"prior KB writes permanently rejected (4xx schema). "
            f"Specialists for affected anchors will start cold "
            f"(no priors). See {dead}."
        )


_INFERENCEX_REPO_DEFAULT = "https://github.com/SemiAnalysisAI/InferenceX.git"
_INFERENCEX_REF_DEFAULT = "a4bb43afa7fd74c1356583ed29e51421be010f0f"


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


def _ensure_eval_concurrency_compat(magpie_path: str, inferencex_path: str) -> bool:
    """Scrub the fatal ``--concurrent-requests`` eval flag from the resolved trees.

    Thin preflight wrapper around
    :func:`hyperloom.orchestrator.actions.executors._magpie_patcher.ensure_eval_concurrency_compat`,
    whose docstring is the authoritative account of why the flag must go.

    Warn-only here: an unpatchable script is re-checked (and fails loudly) by
    the baseline executor immediately before launch, where the effective
    mirrored InferenceX checkout is known.

    Args:
        magpie_path: Resolved Magpie root (``$MAGPIE_PATH``); may be empty.
        inferencex_path: Resolved InferenceX checkout root.

    Returns:
        ``True`` when every resolved target is clean / successfully patched.
    """
    try:
        from hyperloom.orchestrator.actions.executors._magpie_patcher import (
            ensure_eval_concurrency_compat,
        )

        ok = ensure_eval_concurrency_compat(magpie_path or None, inferencex_path or None)
    except Exception as exc:  # noqa: BLE001 — preflight must not die here
        print(f"Preflight: WARNING — eval-concurrency compat patch failed to run: {exc}")
        return False
    if not ok:
        print(
            "Preflight: WARNING — could not remove the redundant "
            "'--concurrent-requests' flag from a Magpie benchmark script "
            f"(MAGPIE_PATH={magpie_path or '<unset>'}, "
            f"INFERENCEX_PATH={inferencex_path}). RUN_EVAL=true baselines will "
            "abort with \"Unknown parameter: --concurrent-requests\"; eval "
            "concurrency must flow via EVAL_CONCURRENT_REQUESTS/CONC instead."
        )
    return ok


def _ensure_client_trust_compat(magpie_path: str) -> bool:
    """Assert the custom-tokenizer trust patch on the resolved Magpie tree.

    ``MAGPIE_TRUST_REMOTE_CODE=1`` is set by default for every run, but it is
    inert on an unpatched Magpie: upstream's SGLang client call sites never
    pass the ``trust`` argument, so ``benchmark_serving.py`` falls back to its
    ``--trust-remote-code`` default of False and builds the tokenizer with
    ``trust_remote_code=False``. For a model shipping custom tokenizer code
    (Kimi, some Qwen / DeepSeek / ChatGLM) transformers then refuses to execute
    it and the benchmark client dies while synthesizing prompts — before any
    request reaches the server. transformers has no environment-variable
    escape hatch, so the CLI flag is the only opt-in.

    ``install.sh`` applies the patch, but preflight pip-installs Magpie on its
    own, so a preflight-only box never gets it. Re-assert it here.

    Multi-node only, keeping the blast radius off the single-node path: the
    remote-direct client this unblocks is the multi-node one. Single-node keeps
    whatever ``install.sh`` did (or did not) leave behind.

    Warn-only: a drifted script leaves the run usable for every model that does
    not need remote code, which is the common case.

    Args:
        magpie_path: Resolved Magpie root (``$MAGPIE_PATH``); may be empty.

    Returns:
        ``True`` when every resolved SGLang script carries the trust gating,
        none was applicable, or the run is single-node.
    """
    from hyperloom.orchestrator.actions.executors._multi_node_env import is_multi_node

    if not is_multi_node():
        return True
    try:
        from hyperloom.orchestrator.actions.executors._magpie_patcher import (
            ensure_client_trust_compat,
        )

        ok = ensure_client_trust_compat(magpie_path or None)
    except Exception as exc:  # noqa: BLE001 — preflight must not die here
        print(f"Preflight: WARNING — client trust compat patch failed to run: {exc}")
        return False
    if not ok:
        print(
            "Preflight: WARNING — could not apply the custom-tokenizer trust "
            "patch to a Magpie SGLang script "
            f"(MAGPIE_PATH={magpie_path or '<unset>'}). MAGPIE_TRUST_REMOTE_CODE=1 "
            "will not reach benchmark_serving.py, so a model shipping custom "
            "tokenizer code will fail to load its tokenizer before issuing any "
            "request. Models without remote code are unaffected."
        )
    return ok


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
    _load_dotenv_fallback()
    # ``.env`` is operator configuration, so both the single-provider intent and
    # the restore baseline are taken after it loads. Only what the installer env
    # file injects on top is undone below.
    provider_mode = _provider_only_mode()
    provider_snapshot = {
        key: os.environ.get(key) for key in (*_PROVIDER_FALLBACK_KEYS, *_ANTHROPIC_FALLBACK_KEYS)
    }
    _load_kernel_agent_env_fallback()
    _derive_runtime_paths()
    _restore_provider_only_mode(provider_mode, provider_snapshot)
    _normalize_legacy_deepseek_env()

    # Fail fast on missing credentials after the fallback loaders.
    _validate_credentials()

    # --- Auth alias export (internal LLM aliases only) ---
    # These aliases feed OpenAI-protocol consumers, so they are filled from the
    # OpenAI-side key only and stay unset when that side is not configured. An
    # explicitly set alias is kept, and the primary keys are never cross-filled.
    # GEAK_API_KEY is not among them: GEAK runs on the Anthropic side, so an
    # OpenAI-side value could not start it.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        for alias in (
            "LLM_API_KEY",
            "AMD_LLM_API_KEY",
        ):
            if not os.environ.get(alias):
                os.environ[alias] = openai_key
                print(f"Preflight: filled {alias} from OPENAI_API_KEY")
    # --- Resolve install interpreters ---
    # Resolve the ACTIVE benchmark backend first so a bypass-only environment
    # (no Magpie / no /opt/venv) never routes installs through Magpie's
    # interpreter. ``resolve_benchmark_interpreter()`` returns the current
    # interpreter for bypass and the Magpie-importable venv for Magpie, so the
    # bypass path never resolves the Magpie venv.
    from hyperloom.orchestrator.actions.executors.benchmark_backend import (
        resolve_backend_name as _resolve_active_backend_name,
        resolve_benchmark_interpreter as _resolve_benchmark_interpreter,
    )

    _magpie_backend_active = _resolve_active_backend_name() == "magpie"
    # Interpreter used for benchmark-runtime installs (Ray). For bypass this is
    # sys.executable; for Magpie it's the Magpie-importable venv.
    benchmark_python = _resolve_benchmark_interpreter()

    # Outside a venv, add --break-system-packages so pip installs on bare-metal Debian/Ubuntu.
    pip_extra: list[str] = []
    if not (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        pip_extra = ["--break-system-packages"]

    # --- Python SDK auto-install (claude-agent-sdk / openai / httpx) ---
    # Must precede Coordinator import (ClaudeBackend lazy-imports the SDK).
    _ensure_python_sdks(sys.executable, pip_extra)

    # --- Resolve Anthropic + OpenAI base URLs (split entrypoints) ---
    # Explicit operator values on each side are preserved; a missing side falls
    # back to the other.
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
        # Claude CLI primary key: Anthropic-side credentials only.
        claude_primary_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        _reset_claude_config_to_upstream(claude_primary_key, anthropic_url)
        if anthropic_url and not openai_url and not os.environ.get("GEAK_CLAUDE_MODEL"):
            geak_claude_model = os.environ.get("CLAUDE_MODEL", "").strip() or "claude-opus-5"
            os.environ["GEAK_CLAUDE_MODEL"] = geak_claude_model
            print(f"Preflight: GEAK_CLAUDE_MODEL <unset> -> {geak_claude_model} (GEAKv4 Claude workflow)")
        resolved_urls = (anthropic_url, openai_url)

        # LLM_API_BASE addresses an OpenAI-protocol endpoint, so it defaults to
        # the resolved OpenAI-side URL and stays unset when that side is not
        # configured. An intentional operator override is preserved. GEAK_BASE_URL
        # is not defaulted here: GEAK runs on the Anthropic side, so an
        # OpenAI-side endpoint could not start it.
        gateway_url = openai_url
        if gateway_url:
            for alias in ("LLM_API_BASE",):
                current = os.environ.get(alias, "").strip()
                if current and current != gateway_url:
                    # A genuine operator override is preserved, but a leftover
                    # install-time proxy is unreachable and force-rewritten.
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

        # A supplied GEAK_CONFIG yaml may carry its own endpoint; sync it so an
        # operator GEAK_BASE_URL override reaches GEAK.
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
    # 1. Ray — used broadly (multi-node scheduling, kernel/profile/recover
    # executors), not only by Magpie, so it is installed regardless of backend.
    # Install it with the active backend's interpreter so a bypass-only box
    # gets Ray in its own venv instead of Magpie's.
    _ensure_ray(benchmark_python, pip_extra)

    # 1b. InferenceX benchmark_serving client deps — required by every serving
    # benchmark client launch. install.sh installs these into the install-time
    # $PYTHON, but the bypass runner launches the client with the active
    # benchmark interpreter; ensure them there too so a bypass-only box whose
    # sys.executable differs from /opt/venv can still import the client.
    _ensure_bench_serving_deps(benchmark_python, pip_extra)

    # 1c. lm_eval — GSM8K accuracy gate, multi-node only (the helper gates
    # itself). The multi-node magpie remote-compat client path runs
    # ``python -m lm_eval`` directly, with no InferenceX runtime shim to install
    # the harness, so every RUN_EVAL=true baseline there would otherwise abort
    # with baseline_accuracy_failed.
    _ensure_lm_eval_dep(benchmark_python, pip_extra, eval_disabled=_resolved_eval_disabled(args))

    # 1d. Per-framework runtime deps declared in assets/framework_deps/. This is
    # the pass that covers the documented flow: install.sh runs before
    # --framework is known, so its own attempt usually no-ops and a scriptable
    # framework would otherwise reach baseline with nothing installed.
    _ensure_framework_deps(args, benchmark_python, pip_extra)

    # 2. Magpie — the benchmark engine the Magpie backend shells out to.
    # Skipped entirely when the
    # active benchmark backend does not need Magpie (e.g. bypass): for the
    # Magpie backend ``benchmark_python`` already resolves to the
    # Magpie-importable venv (via resolve_benchmark_interpreter), so a
    # bypass-only environment never resolves the Magpie venv / /opt/venv.
    magpie_python = benchmark_python
    if not _magpie_backend_active:
        print(f"Preflight: benchmark backend is {_resolve_active_backend_name()!r}; skipping Magpie install/import")
        check = None
    else:
        check = subprocess.run([magpie_python, "-c", "import Magpie"], capture_output=True)
    if _magpie_backend_active and check is not None and check.returncode != 0:
        magpie_repo = os.environ.get("MAGPIE_REPO", "https://github.com/AMD-AGI/Magpie.git")
        magpie_ref = os.environ.get("MAGPIE_REF", "0171222c532db6fc5cb174667db66e34f1d9dd98")
        magpie_spec = os.environ.get(
            "MAGPIE_PACKAGE_SPEC",
            f"magpie-eval @ git+{magpie_repo}@{magpie_ref}",
        )
        print(f"Preflight: Magpie not importable; installing {magpie_spec} ...")
        subprocess.run(
            [magpie_python, "-m", "pip", "install", "--quiet", *pip_extra, magpie_spec],
            check=True,
        )
        print("Preflight: Magpie installed OK")
    if _magpie_backend_active and not os.environ.get("MAGPIE_PATH", "").strip():
        magpie_root = subprocess.run(
            [
                magpie_python,
                "-c",
                "from pathlib import Path; import Magpie; print(Path(Magpie.__file__).resolve().parent.parent)",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if magpie_root:
            os.environ["MAGPIE_PATH"] = magpie_root
            print(f"Preflight: MAGPIE_PATH resolved from installed package: {magpie_root}")

    # 3. InferenceX — required for GSM8K accuracy eval; lm-eval deps auto-install at runtime via benchmark_lib.sh.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if not inferencex_path:
        from ..session.paths import (
            magpie_dir as _magpie_default,
            resolve_dep_dir as _resolve_dep_dir,
        )

        _magpie_env = os.environ.get("MAGPIE_PATH")
        magpie_root = Path(_magpie_env) if _magpie_env else _magpie_default()
        # InferenceX detection order: Magpie submodule (canonical post-install.sh)
        # → installer's per-revision cache checkout (InferenceX@<sha>, resolved via
        # resolve_dep_dir so a process that did not inherit INFERENCEX_PATH still
        # finds it; falls back to the bare dir). Legacy read-only host mounts
        # removed (caused mkstemp [Errno 30]); clone a fresh writable checkout instead.
        for candidate in (
            magpie_root / "InferenceX",
            _resolve_dep_dir("InferenceX"),
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
    # When no writable checkout was found, clone one ourselves. baseline cannot
    # run without InferenceX, so a clone failure is a hard error.
    if not (inferencex_path and _inferencex_checkout_ok(inferencex_path)):
        from ..session.paths import deps_cache_root as _open_source_default

        dest = _open_source_default() / "InferenceX"
        print(f"Preflight: InferenceX not found; cloning into {dest} ...")
        inferencex_path = _clone_inferencex(dest)
        if not (inferencex_path and _inferencex_checkout_ok(inferencex_path)):
            print(
                "Preflight: ERROR — InferenceX checkout missing and clone "
                "failed. baseline cannot run without it. Set INFERENCEX_PATH "
                "to a writable checkout or re-run "
                "src/hyperloom/inference_optimizer/assets/install.sh.",
                file=sys.stderr,
            )
            sys.exit(2)
    # Guard against a read-only INFERENCEX_PATH: Magpie stages benchmark scripts
    # there, so a non-writable tree fails the run before server boot.
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
    # Always overwrite (not setdefault): a stale/broken INFERENCEX_PATH must not
    # survive into the child env. The validated value wins.
    os.environ["INFERENCEX_PATH"] = inferencex_path

    # --- Magpie/InferenceX eval-concurrency compatibility -------------------
    # Preflight installs Magpie and clones InferenceX itself (above), entirely
    # outside install.sh -- and install.sh is the ONLY place that used to apply
    # the Magpie script patches. A Magpie installed here therefore kept
    # upstream's `run_eval --framework lm-eval --port "$PORT"
    # --concurrent-requests $CONC`, which Magpie re-copies into
    # <inferencex>/benchmarks/ at run time; InferenceX's run_lm_eval rejects the
    # flag ("Unknown parameter: --concurrent-requests"), aborting every
    # RUN_EVAL=true baseline before any results*.json exists. Patch the trees we
    # just materialized, now that both paths are known.
    if _magpie_backend_active:
        # Trust patch first, mirroring install.sh: the eval-concurrency strip
        # removes the very `run_eval ... --concurrent-requests` line the legacy
        # MI300X trust patcher matches on, so the reverse order would leave a
        # tree permanently unpatchable by that path.
        _ensure_client_trust_compat(os.environ.get("MAGPIE_PATH", ""))
        _ensure_eval_concurrency_compat(os.environ.get("MAGPIE_PATH", ""), inferencex_path)

    # --- node / claude / codex CLI presence (WARN-only) ---
    _check_node_claude_cli()

    # --- TraceLens CLI presence (HARD-FAIL unless --no-kernel AND roofline off) ---
    # Catches launchers that skip install.sh before a missing CLI surfaces mid-run.
    no_kernel = getattr(args, "no_kernel", False) if args else False
    enable_roofline = getattr(args, "enable_roofline", True) if args else True
    if _tracelens_required_at_preflight(no_kernel, enable_roofline):
        _check_tracelens_cli()
        # Fail fast on a stale/placeholder TRACELENS_ROOT before the Coordinator
        # starts, rather than ~10h later in trace_analyze.
        _check_tracelens_root_exists()
    else:
        _missing_tl = [n for n in _TRACELENS_REQUIRED_CLIS if shutil.which(n) is None]
        if _missing_tl:
            print(
                f"Preflight: WARNING — TraceLens CLI(s) not on PATH: {_missing_tl} "
                f"(skipped; --no-kernel + roofline disabled)"
            )

    # --- IR-3: PR Monitor reachability probe (soft degrade) ---
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
    """IR-3 — PR Monitor reachability probe (soft degrade); never raises/exits.

    Recipe KB enablement is controlled by ``--degraded-kb`` (``recipe_kb_enabled``);
    this probe only affects ``pr_monitor_enabled``.

    Mutates args: ``recipe_kb_enabled``/``pr_monitor_enabled`` plus
    ``kb_degraded_reason``/``pr_degraded_reason`` (None|"explicit_flag"|"ir3_auto").

    Args:
        args (argparse.Namespace): The parsed CLI namespace; mutated in place
            with the resolved KB / PR-monitor enable flags and reasons.
    """
    explicit_kb = bool(getattr(args, "degraded_kb", False))
    explicit_pr = bool(getattr(args, "degraded_pr", False))

    args.recipe_kb_enabled = not explicit_kb
    args.kb_degraded_reason = "explicit_flag" if explicit_kb else None
    args.pr_monitor_enabled = not explicit_pr
    args.pr_degraded_reason = "explicit_flag" if explicit_pr else None

    if explicit_kb and explicit_pr:
        return

    user_data = _workspace_root_resolve()
    marker_path = user_data / "runtime" / "recipe_kb" / ".kb_preflight.json"
    script = Path(__file__).resolve().parent.parent / "assets" / "preflight_kb.sh"
    env = os.environ.copy()
    env.pop("PR_MONITOR_URL", None)
    pr_url = (getattr(args, "pr_monitor_url", None) or "").strip().rstrip("/")
    if pr_url:
        probe_base = pr_url if pr_url.endswith("/v1") else pr_url + "/v1"
        env["PR_MONITOR_URL"] = probe_base
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
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False,
            "pr_reachable": False,
            "kb_skipped": True,
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
                "kb_skipped": True,
                "pr_skipped": explicit_pr,
            }

    if not explicit_pr and not marker.get("pr_reachable", False) and not marker.get("pr_skipped", False):
        args.pr_monitor_enabled = False
        args.pr_degraded_reason = "ir3_auto"
