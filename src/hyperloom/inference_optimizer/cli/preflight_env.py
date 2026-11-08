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
    is_lent_not_private_dotenv_assignment,
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
# The credentials install_kernel_tools.sh write_env_file keeps in a private checkout's .env and, on a checkout another
# account can write (_dotenv_not_private), in the kernel-agent env file alone (RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS in
# assets/runtime_env.sh; test_dotenv_credentials_private.py holds the two equal).
_DOTENV_CREDENTIAL_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
# Their retired spellings, with the DeepSeek configuration that normalizes into them: a .env that keeps no credentials
# lends none of these either (_load_dotenv_fallback). RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS in
# assets/runtime_env.sh, which load_dotenv_no_clobber skips and write_env_file removes.
_DOTENV_RETIRED_CREDENTIAL_KEYS: tuple[str, ...] = (
    "ANTHROPIC_AUTH_TOKEN",
    *LEGACY_DEEPSEEK_ENV_KEYS,
    "SAFE_API_KEY",
    "AMD_LLM_API_KEY",
    "LLM_GATEWAY_KEY",
    "GEAK_API_KEY",
)
# The line write_env_file leaves in a .env it kept them out of (see _dotenv_credentials_elsewhere).
_DOTENV_CREDENTIALS_MARKER = "HYPERLOOM_DOTENV_CREDENTIALS"


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
    # A .env that keeps no credentials lends none: another account may have written them, and a gateway URL or headers
    # of its own would carry this account's key there. They come from the shell or the kernel-agent env file alone
    # (_load_moved_dotenv_credentials). The shell's copy of this skip is in load_dotenv_no_clobber.
    elsewhere = _dotenv_credentials_elsewhere(env_file, parsed)
    if elsewhere is not None:
        ignored = [key for key in (*_DOTENV_CREDENTIAL_KEYS, *_DOTENV_RETIRED_CREDENTIAL_KEYS) if key in parsed]
        for key in ignored:
            del parsed[key]
        if ignored:
            print(
                f"Preflight: WARNING — ignored {len(ignored)} credential line(s) in {env_file}, which keeps none, "
                f"as {elsewhere}",
                file=sys.stderr,
            )
    for key in ("TRACELENS_ROOT", "TRACELENS_INTERNAL_ROOT"):
        if key in parsed and _is_placeholder_tracelens_path(parsed[key]):
            del parsed[key]
    safe_vars = _allowed_dotenv_vars(env_file, parsed)
    if os.environ.get("USER_DATA_PATH", "").strip():
        safe_vars = {key: value for key, value in safe_vars.items() if key not in _WORKSPACE_DERIVED_ENV_KEYS}
    loaded = _load_missing_env_vars(safe_vars)
    if loaded:
        print(f"Preflight: loaded {loaded} missing var(s) from {env_file} (env wins)")
    detail: dict[str, Any] = {"vars_loaded": loaded, "source": str(env_file)}
    moved = None if elsewhere is None else _load_moved_dotenv_credentials(env_file, elsewhere)
    if moved is not None:
        detail["credentials"] = moved
        loaded += moved["vars_loaded"]
    return {
        "status": "applied" if loaded else "already_present",
        "skip_reason": None,
        "detail": detail,
    }


def _allowed_dotenv_vars(env_file: Path, parsed: Mapping[str, str]) -> dict[str, str]:
    """The assignments of ``env_file`` it may lend, each other one named in a warning, never its value.

    A ``.env`` another account can write (``_dotenv_not_private``) lends only the names and values
    ``is_lent_not_private_dotenv_assignment`` lists: nothing that picks what this account runs or fetches or where its
    traffic goes. The shell's copy is load_dotenv_no_clobber in assets/runtime_env.sh.
    """
    not_private = _dotenv_not_private(env_file)
    if not_private is None:
        safe_vars, dropped_vars = filter_untrusted_env_mapping(
            parsed, allow_predicate=lambda key: key == "PYTHON" or is_allowed_dotenv_key(key)
        )
        for key in dropped_vars:
            print(f"Preflight: WARNING — ignoring unsupported .env key {key} from {env_file}", file=sys.stderr)
        return safe_vars
    safe_vars = {}
    for key, value in parsed.items():
        if is_lent_not_private_dotenv_assignment(key, value):
            safe_vars[key] = str(value)
            continue
        print(
            f"Preflight: WARNING — ignoring .env key {key} from {env_file}, which another account can write "
            f"({not_private})",
            file=sys.stderr,
        )
    return safe_vars


def _load_moved_dotenv_credentials(env_file: Path, elsewhere: str) -> dict[str, Any]:
    """Fill the credentials ``env_file`` no longer carries, ``elsewhere`` saying why, from the kernel-agent env file.

    install_kernel_tools.sh ``write_env_file`` keeps them out of a ``.env`` another account can write
    (``_dotenv_credentials_elsewhere``) and writes them into this container's kernel-agent env file alone. They are
    read from there here, with ``.env``, so the provider intent preflight takes from what ``.env`` supplied
    (``_provider_only_mode``) still counts them, as it did when ``.env`` held them. install.sh's
    ``preflight_load_dotenv`` does the same in shell (``runtime_env_load_kernel_agent_credentials``).
    """
    env_path, credentials = _kernel_agent_env_credentials()
    loaded = _load_missing_env_vars(credentials)
    if loaded:
        print(
            f"Preflight: loaded {loaded} missing credential(s) from {env_path} (env wins): {env_file} keeps none, "
            f"as {elsewhere}"
        )
    return {"why": elsewhere, "vars_loaded": loaded, "source": str(env_path) if env_path else None}


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


_AUTHORITATIVE_DOTENV_LINE = re.compile(r"^HYPERLOOM_RUN_MODE=", re.MULTILINE)


def _workspace_selection() -> Mapping[str, str]:
    """The variables that select this run's workspace and so its runtime dir.

    That is the environment, which ``_load_dotenv_fallback`` has already filled from ``.env``, except where a ``.env``
    written by ``hyperloom setup`` names the workspace itself and the shell's copies are stale. This mirrors
    ``runtime_env.sh`` (``setup_dotenv_is_authoritative`` and ``runtime_env_forget_stale_workspace``), which install.sh
    runs; ``test_install_session_paths.py`` holds the two to the same answer. A ``.env`` another account can write
    never names the workspace (``_dotenv_not_private``).
    """
    env_file = _resolve_dotenv_file()
    if env_file is None or _dotenv_not_private(env_file) is not None:
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


def _kernel_agent_env_problem(entry_stat: os.stat_result, *, is_dir: bool, also_owner: int | None = None) -> str | None:
    """Why an entry with this status cannot be, or hold, the kernel-agent env file; None when it can.

    The file carries credentials and is sourced, so it is read only from a directory of this account's, not a link,
    that its group and others cannot write, as a regular file of this account's, not a link, that they cannot write.
    The rule's owner is ``runtime_env_private_kernel_agent_env`` in assets/runtime_env.sh (install.sh, the documented
    ``source`` line, the slurm job); this is its copy for preflight, in the same words, and
    ``test_install_session_paths.py`` holds every reader to the same refusals. ``_dotenv_not_private`` holds the
    checkout's ``.env`` to the same rule, with ``also_owner`` one more uid the entry may belong to.
    """
    uid = os.geteuid()
    if stat.S_ISLNK(entry_stat.st_mode):
        return "is a symbolic link"
    if not (stat.S_ISDIR if is_dir else stat.S_ISREG)(entry_stat.st_mode):
        return f"is not a {'directory' if is_dir else 'regular file'}"
    if entry_stat.st_uid not in (uid, uid if also_owner is None else also_owner):
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


def _kernel_agent_env_text(env_path: Path) -> str:
    """``_read_kernel_agent_env``, exiting 2 with why when the file cannot be read."""
    try:
        text = _read_kernel_agent_env(env_path)
    except OSError as exc:
        print(
            f"Preflight: ERROR — failed to read {env_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    return text


def _dotenv_not_private(env_file: Path) -> str | None:
    """Why another account can write the checkout's ``env_file``, or None when this account alone can.

    The checkout directory and the file, when there is one, are held to ``_kernel_agent_env_problem``'s rule for a
    directory and for a regular file: a link, another account's, or writable by its group or others. A copy of
    ``runtime_env_dotenv_not_private`` in assets/runtime_env.sh, which install_kernel_tools.sh ``write_env_file`` asks
    before it writes credentials to ``.env``, in the same words; test_dotenv_credentials_private.py holds the two to the
    same answer. The checkout is judged where it resolves to, as the shell does (``cd -P``), so one reached through a
    link gets one answer whether ``REPO_ROOT`` or this file named it; ``.env`` itself is never followed. Root accepts
    the uid that owns the checkout as its own (a container's root on the user's bind-mounted checkout): root can
    already write it, and that uid already writes the code root runs from it.
    """
    checkout = env_file.parent
    also_owner = None
    if checkout.is_dir():
        checkout = Path(os.path.realpath(checkout))
        env_file = checkout / env_file.name
        if os.geteuid() == 0:
            try:
                also_owner = checkout.stat().st_uid
            except OSError:
                also_owner = None
    for entry, is_dir in ((checkout, True), (env_file, False)):
        try:
            entry_stat = entry.lstat()
        except FileNotFoundError:
            if is_dir:
                return f"{entry} is not a directory"
            continue
        except OSError:
            return f"{entry} cannot be inspected"
        problem = _kernel_agent_env_problem(entry_stat, is_dir=is_dir, also_owner=also_owner)
        if problem is not None:
            return f"{entry} {problem}"
    return None


def _dotenv_credentials_elsewhere(env_file: Path, parsed: Mapping[str, str] | None = None) -> str | None:
    """Why the checkout's credentials are in the kernel-agent env file rather than in ``env_file``; None when it keeps them.

    Another account can write it (``_dotenv_not_private``), or it carries
    ``HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env``, which ``write_env_file`` leaves in a ``.env`` it kept them out
    of: its own writes turn a ``.env`` another account could write, in a private directory, into one of this
    account's. The marker is looked for in the environment and in ``parsed``, the file's assignments, when given, so
    the answer is known before ``.env`` is loaded (load_dotenv_no_clobber looks in both too). A copy of
    ``runtime_env_dotenv_credentials_elsewhere`` in assets/runtime_env.sh, in the same words.
    """
    not_private = _dotenv_not_private(env_file)
    if not_private is not None:
        return f"another account can write it ({not_private})"
    markers = (os.environ.get(_DOTENV_CREDENTIALS_MARKER), (parsed or {}).get(_DOTENV_CREDENTIALS_MARKER))
    if "kernel-agent-env" in markers:
        return f"it carries {_DOTENV_CREDENTIALS_MARKER}=kernel-agent-env"
    return None


def _kernel_agent_env_credentials() -> tuple[Path | None, dict[str, str]]:
    """The kernel-agent env file install.sh wrote in this container, and its ``_DOTENV_CREDENTIAL_KEYS`` read as data.

    No file and no credentials when install.sh has not written one: ``_load_kernel_agent_env_fallback``, which runs
    next, stops the run and says why. A file another account could have written is refused, unread, as that step
    refuses it (exit 2). A copy of ``runtime_env_load_kernel_agent_credentials`` in assets/runtime_env.sh.
    """
    candidate = _kernel_agent_env_candidate()
    if not candidate or not os.path.lexists(candidate):
        return None, {}
    env_path = _kernel_agent_env_path()
    parsed = _parse_env_assignments(_kernel_agent_env_text(env_path))
    return env_path, {key: parsed[key] for key in _DOTENV_CREDENTIAL_KEYS if parsed.get(key)}


def _load_kernel_agent_env_fallback() -> dict[str, Any]:
    """Auto-source the installer-written kernel-agent env file (see ``_kernel_agent_env_candidate``)."""
    env_path = _kernel_agent_env_path()
    parsed_file_vars = _parse_env_assignments(_kernel_agent_env_text(env_path))
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
