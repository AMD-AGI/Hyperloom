# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight env loading: the repo ``.env``, the kernel-agent env file, provider-only mode, and runtime paths."""

from __future__ import annotations

import importlib.util
import io
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_dotenv_key,
    is_allowed_kernel_agent_env_key,
    is_python_package_root,
)
from hyperloom.common.llm_config import (
    LEGACY_DEEPSEEK_ENV_KEYS,
    deepseek_compat_env,
    has_anthropic_credential,
    provider_model_defaults,
)


_PROVIDER_FALLBACK_KEYS: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "GEAK_BASE_URL",
    "LLM_API_BASE",
    # Legacy: nothing reads these, but they are stripped if present.
    "LLM_GATEWAY_KEY",
    "SAFE_API_KEY",
    # A retired DeepSeek config normalizes to BOTH protocol sides, so it is stripped in either single-provider mode:
    # neither an Anthropic-only nor an OpenAI-only shell may acquire the other side from a stale .env.
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
        # Development checkout: preflight_env.py -> cli -> inference_optimizer -> hyperloom -> src -> repo root.
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
    """Detect explicit single-provider intent from the current environment."""
    has_anthropic = bool(
        os.environ.get("ANTHROPIC_BASE_URL")
        or has_anthropic_credential()
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_BASE_URL")
    )
    has_openai = bool(os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_KEY"))
    if has_anthropic and not has_openai:
        return "anthropic"
    if has_openai and not has_anthropic:
        return "openai"
    return ""


def _normalize_legacy_deepseek_env() -> dict[str, Any]:
    """Rewrite a retired ``DEEPSEEK_*`` configuration into the standard variables."""
    before = dict(os.environ)
    had_legacy_config = any(os.environ.get(key) for key in LEGACY_DEEPSEEK_ENV_KEYS)
    updates = deepseek_compat_env()
    if updates:
        for key, value in updates.items():
            os.environ[key] = value
        print(
            "Preflight: DEEPSEEK_* is deprecated; normalized to "
            f"{', '.join(sorted(updates))}. Re-run setup to migrate your .env."
        )
    # A gateway that serves only its own models supplies the model ids too.
    model_defaults = provider_model_defaults()
    for key, value in model_defaults.items():
        os.environ[key] = value
        print(f"Preflight: {key} <unset> -> {value} (implied by the configured gateway)")
    changed = sorted(key for key in {*updates, *model_defaults} if before.get(key) != os.environ.get(key))
    if changed:
        status = "applied"
        skip_reason = None
    elif had_legacy_config or updates or model_defaults:
        status = "already_present"
        skip_reason = None
    else:
        status = "skipped"
        skip_reason = "legacy_env_absent"
    return {
        "status": status,
        "skip_reason": skip_reason,
        "detail": {"keys_set": changed},
    }


def _restore_provider_only_mode(provider_mode: str, snapshot: dict[str, str | None]) -> None:
    """Undo cross-provider credentials injected by the installer env file."""
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


def _is_placeholder_tracelens_path(value: str) -> bool:
    """Treat unedited .env.template placeholders as unset."""
    stripped = value.strip()
    if stripped in ("", "\\"):
        return True
    low = stripped.lower()
    if "/path/to/" in low or "path/to/your" in low:
        return True
    if "<" in stripped and ">" in stripped:
        return True
    return False


_HOST_PYTHON_ENV_KEYS = frozenset({"PYTHON", "VIRTUAL_ENV", "INFERENCE_OPTIMIZER_FORCE_PYTHON"})

# Paths install.sh derives from USER_DATA_PATH. The checkout's .env is shared by every session run from it, so when the
# caller names its own workspace these come from that root, not from whichever workspace last wrote .env (mirrors
# RUNTIME_ENV_WORKSPACE_DERIVED_KEYS in assets/runtime_env.sh).
_WORKSPACE_DERIVED_ENV_KEYS = frozenset({"HYPERLOOM_RUNTIME_DIR", "KERNEL_AGENT_ENV", "HYPERLOOM_ROOT"})


def _load_missing_env_vars(file_vars: dict[str, str]) -> int:
    """Fill gaps without importing host Python choices into an explicit Docker run."""
    mode = (os.environ.get("HYPERLOOM_RUN_MODE") or file_vars.get("HYPERLOOM_RUN_MODE", "")).strip().lower()
    loaded = 0
    if mode and not os.environ.get("HYPERLOOM_RUN_MODE"):
        os.environ["HYPERLOOM_RUN_MODE"] = mode
        loaded += 1
    for key, value in file_vars.items():
        if key in os.environ or (mode == "docker" and key in _HOST_PYTHON_ENV_KEYS):
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def _load_dotenv_fallback() -> dict[str, Any]:
    """Source missing vars from ``$REPO_ROOT/.env``; env always wins (no-clobber)."""
    env_file = _resolve_dotenv_file()
    if env_file is None:
        return {
            "status": "skipped",
            "skip_reason": "dotenv_missing",
            "detail": {"vars_loaded": 0, "source": None},
        }
    parsed = _parse_env_assignments(env_file.read_text(encoding="utf-8", errors="replace"))
    for key in ("TRACELENS_ROOT", "TRACELENS_INTERNAL_ROOT"):
        if key in parsed and _is_placeholder_tracelens_path(parsed[key]):
            del parsed[key]
    safe_vars, dropped_vars = filter_untrusted_env_mapping(
        parsed,
        allow_predicate=lambda key: key == "PYTHON" or is_allowed_dotenv_key(key),
    )
    for key in dropped_vars:
        print(f"Preflight: WARNING — ignoring unsupported .env key {key} from {env_file}", file=sys.stderr)
    if os.environ.get("USER_DATA_PATH", "").strip():
        safe_vars = {key: value for key, value in safe_vars.items() if key not in _WORKSPACE_DERIVED_ENV_KEYS}
    loaded = _load_missing_env_vars(safe_vars)
    if loaded:
        print(f"Preflight: loaded {loaded} missing var(s) from {env_file} (env wins)")
    return {
        "status": "applied" if loaded else "already_present",
        "skip_reason": None,
        "detail": {"vars_loaded": loaded, "source": str(env_file)},
    }


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


_ROCM_SDK_WHEEL_PACKAGES: tuple[str, ...] = (
    "_rocm_sdk_core",
    "_rocm_sdk_libraries",
    "_rocm_sdk_devel",
)
_ROCM_SDK_WHEEL_LIB_SUBDIRS: tuple[str, ...] = (
    "lib",
    "lib/host-math/lib",
    "lib/rocm_sysdeps/lib",
)


def _rocm_sdk_wheel_lib_dirs() -> list[str]:
    """Lib dirs for TheRock's pip-packaged ROCm (``_rocm_sdk_*`` wheels).

    TheRock splits libraries across up to three namespace packages
    (``_rocm_sdk_core``, ``_rocm_sdk_libraries``, ``_rocm_sdk_devel``); which
    ones are installed depends on the wheel's build profile. Each package can
    also nest libraries under subdirs (host-math, rocm_sysdeps) the dynamic
    loader does not search by default. Returns [] on a standard ``/opt/rocm``
    image, where none of these packages are importable.
    """
    dirs: list[str] = []
    for pkg in _ROCM_SDK_WHEEL_PACKAGES:
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.origin:
            continue
        root = Path(spec.origin).resolve().parent
        for subdir in _ROCM_SDK_WHEEL_LIB_SUBDIRS:
            candidate = root / subdir
            if candidate.is_dir():
                dirs.append(str(candidate))
    return dirs


def _derive_runtime_paths() -> None:
    """Rebuild PATH / LD_LIBRARY_PATH from .env-loaded roots (replaces hyperloom.env.sh)."""
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv:
        _prepend_path("PATH", str(Path(venv) / "bin"))
    rocm = os.environ.get("ROCM_PATH", "")
    if rocm:
        _prepend_path("PATH", str(Path(rocm) / "bin"))
        _prepend_path("LD_LIBRARY_PATH", str(Path(rocm) / "lib"))
    for lib_dir in reversed(_rocm_sdk_wheel_lib_dirs()):
        _prepend_path("LD_LIBRARY_PATH", lib_dir)
    vllm_root = os.environ.get("VLLM_VENV_ROOT", "")
    if os.environ.get("FRAMEWORK", "").strip().lower() == "vllm" and vllm_root:
        _prepend_path("PATH", str(Path(vllm_root) / "bin"))

    # Reconstruct the installer's import roots instead of accepting raw PATH/PYTHONPATH assignments from files.
    magpie = os.environ.get("MAGPIE_PATH", "")
    if magpie and not is_python_package_root(Path(magpie).as_posix()):
        _prepend_path("PYTHONPATH", magpie)
    package_parent = Path(__file__).resolve().parents[2].parent
    root = Path(os.environ.get("REPO_ROOT") or package_parent)
    for candidate in (root / "src", root):
        if (candidate / "hyperloom").is_dir() and not is_python_package_root(candidate.as_posix()):
            _prepend_path("PYTHONPATH", str(candidate))


_KERNEL_AGENT_PATH_VARS: tuple[str, ...] = ("TRACELENS_ROOT",)

_SHELL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_assignments(text: str) -> dict[str, str]:
    """Read assignments as data (first wins), decoding quoted values without expansion."""
    out: dict[str, str] = {}
    stream = io.StringIO(text)
    while True:
        start = stream.tell()
        raw = stream.readline()
        if not raw:
            break
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator:
            continue
        if not _SHELL_NAME_RE.fullmatch(key):
            if not any(char.isspace() for char in key):
                out.setdefault(key, value.strip())  # The allowlist reports invalid assignment names.
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            value_start = raw.index("=") + 1
            while value_start < len(raw) and raw[value_start] in " \t":
                value_start += 1
            stream.seek(start + value_start)
            lexer = shlex.shlex(stream, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            try:
                value = lexer.get_token()
                if stream.tell() and text[stream.tell() - 1] != "\n":
                    stream.readline()
            except ValueError as exc:
                raise ValueError(f"Invalid quoted environment assignment for {key}: {exc}") from exc
        out.setdefault(key, value)
    return out


def _correct_kernel_agent_path_vars(file_vars: dict[str, str], env_path: Path) -> list[str]:
    """Overwrite invalid inherited path-class vars with the env file's value."""
    corrected: list[str] = []
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
        corrected.append(key)
    return corrected


def _load_kernel_agent_env_fallback() -> dict[str, Any]:
    """Auto-source the installer-written kernel-agent env file (``$KERNEL_AGENT_ENV`` or ``$USER_DATA_PATH/runtime/kernel-agent.env.sh``)."""
    candidate = os.environ.get("KERNEL_AGENT_ENV")
    if not candidate:
        user_data = (os.environ.get("USER_DATA_PATH") or "").strip()
        if user_data:
            candidate = str(Path(user_data).expanduser() / "runtime" / "kernel-agent.env.sh")

    if not candidate:
        print(
            "Preflight: ERROR — neither $KERNEL_AGENT_ENV nor $USER_DATA_PATH is set. Cannot "
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
            f"USER_DATA_PATH={os.environ.get('USER_DATA_PATH', '?')}, or "
            f"(b) set $KERNEL_AGENT_ENV to point at an existing file. "
            f"Aborting now (was: silently warning and "
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
        allow_predicate=lambda key: (
            key in _HOST_PYTHON_ENV_KEYS or key == "HYPERLOOM_RUN_MODE" or is_allowed_kernel_agent_env_key(key)
        ),
    )
    for key in dropped_file_vars:
        print(
            f"Preflight: WARNING — ignoring unsupported kernel-agent env key {key} from {env_path}",
            file=sys.stderr,
        )
    if not file_vars:
        print(
            f"Preflight: ERROR — {env_path} sets no supported kernel-agent "
            f"variables. The env file is malformed or stale. Re-run "
            f"src/hyperloom/inference_optimizer/assets/install.sh to regenerate it.",
            file=sys.stderr,
        )
        sys.exit(2)
    loaded = _load_missing_env_vars(file_vars)
    corrected = _correct_kernel_agent_path_vars(file_vars, env_path)
    print(f"Preflight: loaded {loaded} kernel-agent var(s) from {env_path} (env wins)")
    return {
        "status": "applied" if loaded or corrected else "already_present",
        "skip_reason": None,
        "detail": {
            "vars_loaded": loaded,
            "env_file": str(env_path),
            "corrected_keys": corrected,
        },
    }
