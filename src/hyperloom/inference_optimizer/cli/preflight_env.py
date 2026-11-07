# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight env loading: the repo ``.env``, the kernel-agent env file, provider-only mode, and runtime paths."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import re
import shlex
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_dotenv_key,
    is_allowed_kernel_agent_env_key,
    is_python_package_root,
)
from hyperloom_common.llm.llm_config import (
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
# RUNTIME_ENV_WORKSPACE_DERIVED_KEYS in assets/runtime_env.sh). The kernel-agent.env.sh loaded next carries them.
_WORKSPACE_DERIVED_ENV_KEYS = frozenset({"HYPERLOOM_RUNTIME_DIR", "KERNEL_AGENT_ENV", "HYPERLOOM_ROOT"})
_KERNEL_AGENT_ENV_NAME = "kernel-agent.env.sh"


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


def _read_dotenv_text(env_file: Path) -> str:
    return env_file.read_text(encoding="utf-8", errors="replace")


def _load_dotenv_fallback() -> dict[str, Any]:
    """Source missing vars from ``$REPO_ROOT/.env``; env always wins (no-clobber)."""
    env_file = _resolve_dotenv_file()
    if env_file is None:
        return {
            "status": "skipped",
            "skip_reason": "dotenv_missing",
            "detail": {"vars_loaded": 0, "source": None},
        }
    parsed = _parse_env_assignments(_read_dotenv_text(env_file))
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


_AUTHORITATIVE_DOTENV_LINE = re.compile(r"^HYPERLOOM_RUN_MODE=", re.MULTILINE)


def _workspace_selection() -> Mapping[str, str]:
    """The variables that select this run's workspace and so its runtime dir.

    That is the environment, which ``_load_dotenv_fallback`` has already filled from ``.env``, except where a ``.env``
    written by ``hyperloom setup`` names the workspace itself and the shell's copies are stale. This mirrors
    ``runtime_env.sh`` (``setup_dotenv_is_authoritative`` and ``runtime_env_forget_stale_workspace``), which install.sh
    runs; ``test_install_session_paths.py`` holds the two to the same answer.
    """
    env_file = _resolve_dotenv_file()
    if env_file is None:
        return os.environ
    text = _read_dotenv_text(env_file)
    if os.environ.get("HYPERLOOM_SETUP_ENV_AUTHORITATIVE") != "1" and not _AUTHORITATIVE_DOTENV_LINE.search(text):
        return os.environ
    return _parse_env_assignments(text)


def _kernel_agent_env_candidate() -> str | None:
    """The kernel-agent env file path, derived as ``runtime_env_derive_runtime_paths`` derives it for install.sh.

    ``KERNEL_AGENT_ENV``, unless it names the file's legacy place in the user's directory (``_legacy_kernel_agent_env``);
    else ``${TMPDIR:-/tmp}/hyperloom-<uid>-<first 16 hex of sha256(runtime dir)>/kernel-agent.env.sh``, the runtime dir
    being ``HYPERLOOM_RUNTIME_DIR``, else ``USER_DATA_PATH/runtime``, each as ``_workspace_selection`` has it, and
    hashed as ``_absolute_path`` makes it absolute.
    """
    selection = _workspace_selection()
    user_data = selection.get("USER_DATA_PATH") or ""
    runtime_dir = selection.get("HYPERLOOM_RUNTIME_DIR") or (f"{user_data}/runtime" if user_data else "")
    explicit = selection.get("KERNEL_AGENT_ENV")
    if explicit and explicit not in _legacy_kernel_agent_env(runtime_dir, user_data):
        return explicit
    if not runtime_dir:
        return None
    key = hashlib.sha256(os.fsencode(_absolute_path(runtime_dir))).hexdigest()[:16]
    tmp = os.environ.get("TMPDIR") or "/tmp"  # nosec B108 - every reader refuses what another account wrote there
    return f"{tmp}/hyperloom-{os.getuid()}-{key}/{_KERNEL_AGENT_ENV_NAME}"


def _absolute_path(path: str) -> str:
    """``path`` absolute: a leading ``~`` or ``~name`` expanded, a relative path put under ``os.getcwd()`` (what
    ``cli.main`` makes ``USER_DATA_PATH`` absolute against), empty and ``.`` components dropped, nothing else resolved.

    A copy of ``runtime_env_absolute_path`` in assets/runtime_env.sh, which keys the file for install.sh and the
    documented ``source`` line; ``test_install_session_paths.py`` holds the two equal.
    """
    path = os.path.expanduser(path)
    if not path.startswith("/"):
        path = f"{os.getcwd()}/{path}"
    return "/" + "/".join(part for part in path.split("/") if part not in ("", "."))


def _legacy_kernel_agent_env(runtime_dir: str, user_data: str) -> set[str]:
    """Where install.sh wrote the file before it left the user's directory, which every pod of the user can write."""
    places = {f"{runtime_dir}/{_KERNEL_AGENT_ENV_NAME}"} if runtime_dir else set()
    if user_data:
        places.add(f"{user_data}/runtime/{_KERNEL_AGENT_ENV_NAME}")
    return places


def _kernel_agent_env_path() -> Path:
    """The kernel-agent env file this run sources; exit 2 when it cannot be resolved, does not exist, or is not private.

    Its directory and the file, where they exist, are held to ``_kernel_agent_env_problem`` before anything is read.
    """
    candidate = _kernel_agent_env_candidate()

    if not candidate:
        print(
            "Preflight: ERROR — none of $KERNEL_AGENT_ENV, $HYPERLOOM_RUNTIME_DIR, $USER_DATA_PATH is set. Cannot "
            "resolve kernel-agent.env.sh. Run "
            "src/hyperloom/inference_optimizer/assets/install.sh and export "
            "USER_DATA_PATH=/path/to/sessions first.",
            file=sys.stderr,
        )
        sys.exit(2)
    env_path = Path(candidate)
    for entry, is_dir in ((env_path.parent, True), (env_path, False)):
        try:
            entry_stat = entry.lstat()
        except OSError:
            continue
        _require_private_kernel_agent_env(env_path, entry, entry_stat, is_dir=is_dir)
    if not env_path.is_file():
        print(
            f"Preflight: ERROR — kernel-agent env file not found at "
            f"{env_path}. install.sh writes it into this container's "
            f"temporary directory, never onto shared storage, so it does "
            f"not outlive the container. Either "
            f"(a) re-run src/hyperloom/inference_optimizer/assets/install.sh from this checkout in "
            f"this container with the same USER_DATA_PATH (it writes the file where this check looks), or "
            f"(b) set $KERNEL_AGENT_ENV to point at an existing file. "
            f"Aborting now (was: silently warning and "
            f"letting trace_analyze fail 10h in).",
            file=sys.stderr,
        )
        sys.exit(2)
    return env_path


def _kernel_agent_env_problem(entry_stat: os.stat_result, *, is_dir: bool) -> str | None:
    """Why an entry with this status cannot be, or hold, the kernel-agent env file; None when it can.

    The file carries credentials and is sourced, so it is read only from a directory of this account's, not a link,
    that its group and others cannot write, as a regular file of this account's, not a link, that they cannot write.
    The rule's owner is ``runtime_env_private_kernel_agent_env`` in assets/runtime_env.sh (install.sh, the documented
    ``source`` line, the slurm job); this is its copy for preflight, in the same words, and
    ``test_install_session_paths.py`` holds every reader to the same refusals.
    """
    uid = os.geteuid()
    if stat.S_ISLNK(entry_stat.st_mode):
        return "is a symbolic link"
    if not (stat.S_ISDIR if is_dir else stat.S_ISREG)(entry_stat.st_mode):
        return f"is not a {'directory' if is_dir else 'regular file'}"
    if entry_stat.st_uid != uid:
        return f"belongs to uid {entry_stat.st_uid}, not to this account (uid {uid})"
    if entry_stat.st_mode & 0o022:
        return f"can be written by its group or others (mode {stat.S_IMODE(entry_stat.st_mode):o})"
    return None


def _require_private_kernel_agent_env(env_path: Path, entry: Path, entry_stat: os.stat_result, *, is_dir: bool) -> None:
    """Exit 2 naming ``entry`` and why when ``_kernel_agent_env_problem`` refuses it, never showing what the file holds."""
    problem = _kernel_agent_env_problem(entry_stat, is_dir=is_dir)
    if problem is None:
        return
    print(
        f"Preflight: ERROR — refusing the kernel-agent env file {env_path}: {entry} {problem}; it carries credentials "
        f"and is sourced, so it is only written and read in a directory of this account's that no other account can "
        f"write. Set KERNEL_AGENT_ENV to a file in such a directory on this container's own disk, or TMPDIR to such "
        f"a directory, and re-run install.sh.",
        file=sys.stderr,
    )
    sys.exit(2)


def _read_kernel_agent_env(env_path: Path) -> str:
    """The text of ``env_path``, checked again on the descriptors it is read through.

    The directory is opened without following a link and the file opened inside it the same way, so neither can be
    swapped between ``_kernel_agent_env_path``'s check and this read.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    dir_fd = os.open(env_path.parent, flags | os.O_DIRECTORY)
    try:
        _require_private_kernel_agent_env(env_path, env_path.parent, os.fstat(dir_fd), is_dir=True)
        file_fd = os.open(env_path.name, flags, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    with os.fdopen(file_fd, "rb") as handle:
        _require_private_kernel_agent_env(env_path, env_path, os.fstat(handle.fileno()), is_dir=False)
        return handle.read().decode("utf-8", errors="replace")


def _load_kernel_agent_env_fallback() -> dict[str, Any]:
    """Auto-source the installer-written kernel-agent env file (see ``_kernel_agent_env_candidate``)."""
    env_path = _kernel_agent_env_path()
    try:
        text = _read_kernel_agent_env(env_path)
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
