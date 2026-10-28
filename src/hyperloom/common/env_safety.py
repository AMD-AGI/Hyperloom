# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared environment-variable safety helpers."""

from __future__ import annotations

import grp
import os
import pwd
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, MutableMapping, Sequence
from hyperloom.common.env import env_str
from hyperloom.common.git_safety import trusted_admin_dir
from hyperloom.common.fs_utils import rewrite_lines_nofollow

from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES

import json
import logging
import re
from collections.abc import Mapping

log = logging.getLogger(__name__)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PYTHON_PACKAGE_ROOT_BASENAMES: frozenset[str] = frozenset({"site-packages", "dist-packages"})

BLOCKED_UNTRUSTED_ENV_NAMES: frozenset[str] = frozenset(
    {
        # Names a second variable for the agentic client to export, so it reaches a shell indirection the way the
        # loader vars below do: whoever sets it chooses which variable gets written, not merely a value.
        "AGENTX_KEEP_ALIVE_ENV",
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GCONV_PATH",
        "GIT_SSH_COMMAND",
        "IFS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "PATH",
        "PERL5OPT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        # site.py adds its site-packages to sys.path, so it loads arbitrary code.
        "PYTHONUSERBASE",
        "SHELLOPTS",
    }
)

BLOCKED_CHILD_ENV_NAMES: frozenset[str] = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "GCONV_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "SHELLOPTS",
    }
)

BENCHMARK_SECRET_ENV_NAMES: frozenset[str] = frozenset(
    {
        "AMD_API_KEY",
        "AMD_LLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAW_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEAK_API_KEY",
        "LLM_API_KEY",
        "LLM_GATEWAY_KEY",
        "LLM_PROXY_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_CUSTOM_HEADERS",
        # Hosted measurement: a sandbox API key per role, the application's MCP token and the path of a sealed
        # runner's signing key. Only a name listed here is removed from a child process's environment.
        "HYPERLOOM_MCP_TOKEN",
        "HYPERLOOM_SANDBOX_KEY_DEV",
        "HYPERLOOM_SANDBOX_KEY_MEASUREMENT",
        "HYPERLOOM_SANDBOX_KEY_SUBRUN",
        "HYPERLOOM_SIGNING_KEY_FILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        # Legacy: nothing reads it, but it is scrubbed if present.
        "SAFE_API_KEY",
    }
)

_SECRET_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(authorization\s*:\s*)"
            r"(?:(?P<authorization_scheme>[A-Za-z][A-Za-z0-9._~+/-]*[ \t]+)"
            r"(?=[^\s,;'\"\\]+))?"
            r"[^\s,;'\"\\]+"
        ),
        r"\1\g<authorization_scheme>[REDACTED]",
    ),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"\b((?:ak|sk|pk)-(?:lf-)?)[A-Za-z0-9\-_]{6,}"), r"\1[REDACTED]"),
    (re.compile(r"\b(gh[pousr]_)[A-Za-z0-9_]{3,}"), r"\1[REDACTED]"),
    (re.compile(r"\b(github_pat_)[A-Za-z0-9_]{10,}"), r"\1[REDACTED]"),
    # AWS access-key id and compact JWT.
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "[REDACTED]"),
    (
        re.compile(r"(?i)(ocp-apim-subscription-key\s*:\s*)[^\s,;'\"\\]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)(?<![A-Z0-9_./-])(authorization\s*=\s*)[^\s,;'\"\\]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*CUSTOM_HEADERS\s*[=:]\s*)"
            r"(\\?[\"'])?"
            r"(?:[A-Z0-9_-]+:[ \t]*[^\r\n,;'\"\\]+"
            r"(?:,[ \t]*[A-Z0-9_-]+:[ \t]*[^\r\n,;'\"\\]+)*|"
            r"[^\s,;'\"\\]+)"
        ),
        r"\1\2[REDACTED]",
    ),
    # Shell commands quote the value (``PASSWORD="x"``), and a command that has been through ``json.dumps`` carries an
    # escaped quote (``PASSWORD=\"x\"``).
    (
        re.compile(
            r"(?i)(?<![A-Z0-9_./-])("
            r"(?:[A-Z0-9_]*(?:API_?KEY|SECRET|PASSWORD|CREDENTIAL|HEADERS)[A-Z0-9_]*|"
            r"AUTH(?:_[A-Z0-9_]+)?|"
            r"[A-Z0-9_]+_AUTH(?:ORIZATION)?(?:_[A-Z0-9_]+)?|"
            r"(?:[A-Z0-9_]+_)?TOKEN(?:_\d+)?)"
            r"\s*[=:]\s*)"
            r"(\\?[\"'])?(?:\\(?![\"'])|[^\s,;'\"\\])+"
        ),
        r"\1\2[REDACTED]",
    ),
)

DOTENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        # ANTHROPIC_AUTH_TOKEN is deliberately absent: no installer ever writes it (they persist the API-key spelling
        # and actively remove this one), so anything read back here is a hand-written leftover -- and since it
        # outranks a subscription token, reading it would silently move the run onto API billing.
        "ANTHROPIC_BASE_URL",
        # Gateway auth header.
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        # Retired provider variables, still readable so a pre-migration .env can be normalized by
        # hyperloom.common.llm_config.deepseek_compat_env.
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DSH_MODEL",
        "FRAMEWORK",
        "GEAK_CLAUDE_MODEL",
        "HIP_PATH",
        "HF_DATASETS_OFFLINE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "HYPERLOOM_BENCHMARK_BACKEND",
        "HYPERLOOM_DOCKER_TARGET_HOST",
        "HYPERLOOM_FRAMEWORK_ENV",
        "HYPERLOOM_LLM_MODE",
        "HYPERLOOM_RUN_MODE",
        "HYPERLOOM_RUNTIME_DIR",
        "HYPERLOOM_SKILL_PATH",
        "HYPERLOOM_WHEEL_REPO",
        "HYPERLOOM_WHEEL_TAG",
        "INFERENCE_OPTIMIZER_FORCE_PYTHON",
        "KERNEL_AGENT_ENV",
        "KERNEL_OPT_BACKEND_ORDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CUSTOM_HEADERS",
        "NO_PROXY",
        "ROCM_PATH",
        "SGLANG_ROCM_EXTRA",
        "SGLANG_ROCM_INDEX_URL",
        "SGLANG_ROCM_PYPI_VERSION",
        "USER_DATA_PATH",
        "VIRTUAL_ENV",
        "VLLM_PYTHON",
        "VLLM_VENV_ROOT",
    }
)

DOTENV_PREFIX_ALLOWLIST: tuple[str, ...] = (
    "AITER_",
    "FORGE_",
    "GEAK_",
    # Not covered by "FORGE_": KernelForge's own knobs (writable-state root, rewrite handshake, mori KB opt-in) are
    # spelled KERNELFORGE_*, and now that forge ships inside this distribution an operator configures them here.
    "KERNELFORGE_",
    "HF_",
    "HYPERLOOM_",
    "INFERENCE_OPTIMIZER_",
    "KERNEL_OPT_",
    "SGLANG_",
    "VLLM_",
)

KERNEL_AGENT_ENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        # install.sh persists this next to the Anthropic URL and key, so the reader has to accept it or a
        # header-authenticated gateway loses its credential on the way back in.
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "GEAK_CLAUDE_BIN",
        "GEAK_CLAUDE_MODEL",
        "GEAK_E2E_RUNNER",
        "GEAK_MAX_BENCHMARK_SHAPES",
        "GEAK_ROOT",
        "GEAK_RUN_MODE",
        "GEAK_SCORE_TARGET",
        "GEAK_SKIP_PROFILE",
        "HYPERLOOM_ROOT",
        "HYPERLOOM_RUNTIME_DIR",
        "HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV",
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "INFERENCEX_PATH",
        "KERNEL_AGENT_ENV",
        "KERNEL_AGENT_LOG_LEVEL",
        # KernelForge's writable-state root.
        "KERNELFORGE_PROJECT_ROOT",
        "KERNEL_OPT_BACKEND_ORDER",
        "MAGPIE_PATH",
        "MAGPIE_PYTHON",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        # install.sh never writes the OpenAI side itself, but the URL and key are already accepted from an
        # operator-supplied env file; the header that authenticates the same endpoint is read on the same terms.
        "OPENAI_CUSTOM_HEADERS",
        "TRACELENS_INTERNAL_ROOT",
        "TRACELENS_ROOT",
        "USER_DATA_PATH",
    }
)

# Env names an untrusted external source (reference recipe, framework-switch manifest) may never set: shell-unsafe
# vars plus the workload/benchmark keys the optimizer's CLI flags own — setting one retargets the benchmark instead of
# toggling a knob.
BLOCKED_EXTERNAL_ENV_NAMES: frozenset[str] = (
    BLOCKED_UNTRUSTED_ENV_NAMES
    | BENCHMARK_SECRET_ENV_NAMES
    | GPU_MASK_ENV_NAMES
    | frozenset(
        {
            "HOME",
            "MODEL",
            "MODEL_PATH",
            "TP",
            "EP",
            "CONC",
            "ISL",
            "OSL",
            "MAX_MODEL_LEN",
            "PRECISION",
            "PORT",
            "NUM_PROMPTS",
            "NUM_WARMUPS",
            "RANDOM_RANGE_RATIO",
            "RUN_EVAL",
            "PROFILE",
            "RESULT_DIR",
            "RESULT_FILENAME",
            # Reroute traffic, model downloads or TLS trust.
            "CURL_CA_BUNDLE",
            "HF_ENDPOINT",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
        }
    )
)

# Env names the launcher or the server process puts in the environment itself: rank/world topology a distributed
# launch assigns per process. Nothing a round declares can own one — the launch overwrites it per rank — so a patch
# reading one is reading the launch's own state, not arming a switch.
LAUNCH_PROVIDED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "GROUP_RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "RANK",
        "ROLE_RANK",
        "WORLD_SIZE",
    }
)

# Prefixes of the same kind: everything a job scheduler or an elastic launch exports around the process it starts.
LAUNCH_PROVIDED_ENV_PREFIXES: tuple[str, ...] = (
    "OMPI_",
    "PMIX_",
    "PMI_",
    "SLURM_",
    "TORCHELASTIC_",
)

# Env names a per-variant override may never set.
BLOCKED_VARIANT_ENV_NAMES: frozenset[str] = BLOCKED_UNTRUSTED_ENV_NAMES | BENCHMARK_SECRET_ENV_NAMES


def is_launch_provided_env_name(key: object) -> bool:
    """True when the launch or the server process supplies this name, so no round can arm it."""
    upper = str(key or "").strip().upper()
    if upper in LAUNCH_PROVIDED_ENV_NAMES:
        return True
    return upper.startswith(LAUNCH_PROVIDED_ENV_PREFIXES)


# Credential-shaped name fragments, so an unlisted secret cannot be persisted into a session YAML by name alone.
_SECRET_NAME_FRAGMENTS: tuple[str, ...] = ("APIKEY", "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

# Masked out before matching, not exempted whole, so TOKENIZER_API_KEY still reads as a credential while
# TOKENIZERS_PARALLELISM does not.
_SECRET_FRAGMENT_EXEMPTIONS: tuple[str, ...] = ("TOKENIZER",)


def is_secret_shaped_env_name(key: object) -> bool:
    """True when a name looks like a credential rather than a tuning knob."""
    upper = str(key or "").strip().upper()
    for exempt in _SECRET_FRAGMENT_EXEMPTIONS:
        upper = upper.replace(exempt, "")
    # Header strings carry Authorization values, matching the ``*CUSTOM_HEADERS`` text-redaction pattern above.
    if upper.endswith("CUSTOM_HEADERS"):
        return True
    return any(fragment in upper for fragment in _SECRET_NAME_FRAGMENTS)


# Env-var name fragments whose value is masked before an env snapshot leaves this process. Shared with the Langfuse
# session_start snapshot, which matches every marker as a substring.
SENSITIVE_ENV_NAME_MARKERS: tuple[str, ...] = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "SECRET_KEY",
    "AUTH",
    "SIGNATURE",
    "HEADERS",
    "CUSTOM_HEADERS",
)

# As a substring TOKEN also hits tuning knobs (``MAX_NUM_BATCHED_TOKENS``, ``TOKENIZERS_PARALLELISM``), so a record of
# launch knobs treats it as a credential only as a whole name segment (``HF_TOKEN``, ``HF_TOKEN_2``,
# ``GITHUB_TOKEN_BACKUP``), and never in ``PER_TOKEN`` (``SGLANG_USE_AITER_FP8_PER_TOKEN``).
_SUBSTRING_MARKERS_EXCEPT_TOKEN: tuple[str, ...] = tuple(m for m in SENSITIVE_ENV_NAME_MARKERS if m != "TOKEN")
_TOKEN_SEGMENT_RE = re.compile(r"(?:^|_)TOKEN(?=_|$)")
_PER_TOKEN_SEGMENT_RE = re.compile(r"(?:^|_)PER_TOKEN(?=_|$)")


def _is_credential_env_name(key: object) -> bool:
    upper = str(key or "").strip().upper()
    if upper in BENCHMARK_SECRET_ENV_NAMES:
        return True
    if any(marker in upper for marker in _SUBSTRING_MARKERS_EXCEPT_TOKEN):
        return True
    return bool(_TOKEN_SEGMENT_RE.search(_PER_TOKEN_SEGMENT_RE.sub("", upper)))


def redact_secret_env_values(env: Mapping[str, object] | None) -> dict[str, object]:
    """Copy ``env`` with every credential's value masked, for a record that leaves this process.

    A credential-shaped name is masked whole; any other string value is still scrubbed of recognizable credentials.
    """
    return {
        key: "[REDACTED]"
        if _is_credential_env_name(key)
        else (redact_secret_values(value) if isinstance(value, str) and value else value)
        for key, value in (env or {}).items()
    }


def is_allowed_external_env_key(key: object) -> bool:
    """True when an env export from an untrusted external source is safe to carry."""
    upper = str(key or "").strip().upper()
    if not valid_env_key(upper) or upper in BLOCKED_EXTERNAL_ENV_NAMES:
        return False
    return not is_secret_shaped_env_name(upper)


def is_allowed_variant_env_key(key: object) -> bool:
    """True when a per-variant env override is safe to hand a benchmark subprocess."""
    upper = str(key or "").strip().upper()
    return valid_env_key(upper) and upper not in BLOCKED_VARIANT_ENV_NAMES


def is_python_package_root(path: object) -> bool:
    """True when ``path`` is a ``site-packages``/``dist-packages`` dir."""
    name = str(path or "").strip().rstrip("/")
    if not name:
        return False
    return name.rsplit("/", 1)[-1] in _PYTHON_PACKAGE_ROOT_BASENAMES


def valid_env_key(key: object) -> bool:
    return bool(_ENV_KEY_RE.fullmatch(str(key or "")))


def is_allowed_dotenv_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    if not valid_env_key(upper) or upper in BLOCKED_UNTRUSTED_ENV_NAMES:
        return False
    return upper in DOTENV_EXACT_ALLOWLIST or any(upper.startswith(prefix) for prefix in DOTENV_PREFIX_ALLOWLIST)


def is_allowed_kernel_agent_env_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    return valid_env_key(upper) and upper in KERNEL_AGENT_ENV_EXACT_ALLOWLIST


def filter_untrusted_env_mapping(
    envs: Mapping[str, object] | None,
    *,
    allow_predicate,
) -> tuple[dict[str, str], dict[str, str]]:
    """Filter env overrides from state, LLM output, or shared env files."""
    allowed: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for key, value in (envs or {}).items():
        name = str(key or "").strip()
        if not valid_env_key(name):
            dropped[name or "<empty>"] = "invalid_env_key"
            continue
        upper = name.upper()
        if not allow_predicate(upper):
            dropped[name] = "not_allowed"
            continue
        allowed[name] = str(value)
    return allowed, dropped


# Env vars whose value names a config file the serving stack reads unconditionally: aiter's tuned-GEMM loader skips
# its merge default and pd.read_csv's exactly the AITER_CONFIG_* path when one is set, so a missing file kills the
# server at config load instead of falling back to the default table. Values that are not file paths (e.g. geak's
# JSON-payload AITER_CONFIG) do not carry the trailing underscore and stay untouched.
FILE_REFERENCING_ENV_PREFIXES: tuple[str, ...] = ("AITER_CONFIG_",)


def _is_file(path: str) -> bool:
    # Path.is_file() raises for an unreadable parent or an overlong name (before Python 3.14); such a path is not a
    # file the server can read either, and one proposal must not crash what checks it.
    try:
        return Path(path).is_file()
    except OSError:
        return False


def ungrounded_env_file_reason(envs: Mapping[str, object] | None) -> str | None:
    """Return why ``envs`` cannot be served if one names a config file that does not exist, else ``None``.

    Grounds file-like env values the way patch targets are grounded: a proposal may point a config env at an
    artifact only a later (possibly rejected) authoring step would create, and the server would die at config load.
    The check is existence-only; anything already on disk (a promoted, durable artifact) passes. A set value is a
    ``:``-joined list of paths (see ``csvs_aiter_will_load``), so every non-empty element must exist, not only the
    first.
    """
    for raw_key, raw_value in (envs or {}).items():
        key = str(raw_key or "").strip().upper()
        value = str(raw_value or "").strip()
        if not key.startswith(FILE_REFERENCING_ENV_PREFIXES) or not value:
            continue
        missing = [p for p in (part.strip() for part in value.split(":")) if p and not _is_file(p)]
        if missing:
            return (
                f"{key} points at {value}, whose element(s) {':'.join(missing)} do not exist; the server would read it "
                "unconditionally and die at config load. Point it at a file a stage wrote (the gemm_tuning action "
                "writes tuned-GEMM CSVs) or leave it unset to use the default config table."
            )
    return None


OPERATOR_EXTRA_ENV_VAR = "INFERENCE_OPTIMIZER_EXTRA_ENV"


def operator_extra_env(raw: str | None = None) -> dict[str, str]:
    """Return the operator's ``--extra-env`` pins from the CLI handoff blob.

    The pins are also exported under their own names; this blob is what survives a state roundtrip and what a
    resume reads to tell which names the previous launch set.

    Args:
        raw: The serialized blob, or ``None`` to read it from the environment.

    Returns:
        The pins; empty when unset or unparseable, because a malformed pin must not take the run down with it.
    """
    blob = (os.environ.get(OPERATOR_EXTRA_ENV_VAR, "") if raw is None else raw).strip()
    if not blob:
        return {}
    try:
        parsed = json.loads(blob)
    except (TypeError, ValueError):
        log.warning("ignoring unparseable %s; the operator's pins are not in effect", OPERATOR_EXTRA_ENV_VAR)
        return {}
    if not isinstance(parsed, dict):
        log.warning("ignoring non-object %s; the operator's pins are not in effect", OPERATOR_EXTRA_ENV_VAR)
        return {}
    return {str(key).strip(): str(value) for key, value in parsed.items() if str(key).strip()}


def scrub_child_process_env(env: dict[str, str]) -> dict[str, str]:
    """Remove startup/preload hooks from a child-process environment in place."""
    for name in BLOCKED_CHILD_ENV_NAMES:
        env.pop(name, None)
    return env


def scrub_benchmark_process_env(env: dict[str, str]) -> dict[str, str]:
    """Remove control-plane credentials from a benchmark environment in place."""
    scrub_child_process_env(env)
    for name in BENCHMARK_SECRET_ENV_NAMES:
        env.pop(name, None)
    return env


def build_benchmark_env(*layers: Mapping[str, object] | None) -> dict[str, str]:
    """Build a benchmark subprocess env: parent env under each layer, later winning."""
    env = os.environ.copy()
    for layer in layers:
        env.update({str(key).upper(): str(value) for key, value in (layer or {}).items()})
    return scrub_benchmark_process_env(env)


def redact_secret_values(text: str) -> str:
    """Mask recognizable credential values before diagnostic text is persisted."""
    out = str(text or "")
    for pattern, replacement in _SECRET_REDACTION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact_file_in_place(path: os.PathLike[str] | str, *, mode: int = 0o600) -> None:
    """Stream-redact ``path`` with :func:`redact_secret_values` and restrict mode.

    The file sits in a directory a less-privileged account may write, so the
    redacted copy is staged and moved without following any link there
    (:func:`hyperloom.common.fs_utils.rewrite_lines_nofollow`). A failure
    leaves the file as it was.
    """
    try:
        rewrite_lines_nofollow(path, redact_secret_values, mode=mode)
    except OSError:
        return


SPECIALIST_USER_ENV = "HYPERLOOM_SPECIALIST_USER"
_GIT_NAME = ".git"

# New files made by the child are group-writable, so the app (a member of the
# specialist's group, or root) can still read and rewrite them.
_CHILD_UMASK = 0o007
_GROUP_RW = stat.S_IRGRP | stat.S_IWGRP
_GROUP_RWX_SETGID = _GROUP_RW | stat.S_IXGRP | stat.S_ISGID
_GROUP_R = stat.S_IRGRP
_GROUP_RX_SETGID = _GROUP_R | stat.S_IXGRP | stat.S_ISGID
_DROP_TOOL = "setpriv"
# What the demoted command runs under: no inheritable or ambient capability and no way to gain one by exec.
_DROP_ARGS = ("--no-new-privs", "--inh-caps=-all", "--ambient-caps=-all", "--")


class RunAsError(ValueError):
    """The requested specialist user cannot be used; the launch must not proceed."""


@dataclass(frozen=True)
class RunAs:
    """What a launcher needs to start a child as one named user."""

    user: str
    uid: int
    gid: int
    groups: tuple[int, ...]
    home: str
    drop_tool: str = ""

    def command(self, cmd: Sequence[str]) -> list[str]:
        """``cmd`` prefixed so that the demoted child holds no capability and can gain none.

        Run as the child's first program, after the uid switch and before ``cmd``:
        it clears the inheritable and ambient capability sets (which a non-root
        app's CAP_SETUID/SETGID/KILL would otherwise pass on) and sets
        ``no_new_privs``. Without a ``drop_tool`` the command is returned as it
        is; :func:`resolve_run_as` always sets one.
        """
        if not self.drop_tool:
            return list(cmd)
        return [self.drop_tool, *_DROP_ARGS, *cmd]

    def popen_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``subprocess.Popen`` that start the child as this user."""
        return {
            "user": self.uid,
            "group": self.gid,
            "extra_groups": list(self.groups),
            "umask": _CHILD_UMASK,
        }

    def apply_env(self, env: MutableMapping[str, str], *, safe_dirs: Iterable[Path] = ()) -> None:
        """Point the identity variables at this user, in place.

        The app's ``HOME`` is not readable by this user. ``safe_dirs`` are
        marked ``safe.directory`` for git (command-line scope, the only scope
        git honours for it), because the child is not the owner of a tree the
        app created.
        """
        env["HOME"] = self.home
        env["USER"] = env["LOGNAME"] = self.user
        if "USERNAME" in env:
            env["USERNAME"] = self.user
        safe = [str(p) for p in safe_dirs]
        env["GIT_CONFIG_COUNT"] = str(len(safe))
        for i, path in enumerate(safe):
            env[f"GIT_CONFIG_KEY_{i}"] = "safe.directory"
            env[f"GIT_CONFIG_VALUE_{i}"] = path

    def share_worktree_git(self, worktree: Path) -> None:
        """Let this user read the git metadata of a linked worktree, and nothing more.

        ``git worktree add`` creates the ``<worktree>/.git`` gitfile and its
        ``<repo>/.git/worktrees/<name>`` admin dir under the app's group and,
        under a restrictive app umask, without other-read, so the child's git
        would fail. Both are shared read-only, as is search (not write) on the
        ``worktrees`` directory above the admin dir. A worktree with no gitfile (a snapshot)
        needs nothing.

        The gitfile sits in a tree the child can rewrite, so its ``gitdir:``
        line is believed only when the admin dir it names points back at this
        worktree's gitfile (the admin dir is the app's); a gitfile that lies is
        refused with ``OSError`` rather than shared.
        """
        gitfile = worktree / _GIT_NAME
        if not gitfile.is_file() or gitfile.is_symlink():
            return
        try:
            admin = trusted_admin_dir(worktree)
        except OSError as exc:
            raise OSError(f"refusing to share the git metadata of {worktree}: {exc}") from exc
        self.share_tree(gitfile, write=False)
        self.share_tree(admin, write=False)
        parent = admin.parent
        os.chown(parent, -1, self.gid)
        # Search, not write: a group that could rename entries here could break every other task's git.
        parent.chmod((stat.S_IMODE(parent.stat().st_mode) | stat.S_IRGRP | stat.S_IXGRP) & ~stat.S_IWGRP)

    def share_tree(self, root: Path, *, sticky: bool = False, write: bool = True) -> None:
        """Give this user's group read-write access to everything under ``root``.

        The tree stays owned by the app. Its group becomes this user's group,
        directories get the setgid bit (new entries keep the group) and every
        entry gets group read and write (``write=False``: group read only,
        plus search on directories, for a tree the child only reads, such as
        a worktree's git metadata). ``sticky`` additionally stops the child
        deleting the app's own files in the top directory. Symlinks are
        not followed, and a ``root`` that is itself a symlink is refused.
        ``.git`` is left alone: the child does no git writes. Needs root or
        membership in the group; raises ``OSError``
        otherwise, so a launcher cannot start a child that cannot write.
        """
        if root.is_symlink():
            raise OSError(f"refusing to share {root}: it is a symlink")
        for path in _walk(root):
            self._share_entry(path, write=write, sticky=sticky and path == root)

    def _share_entry(self, path: Path, *, write: bool, sticky: bool) -> None:
        """Give one entry the account's group and the share modes, never through a link.

        The entry is pinned with ``O_PATH | O_NOFOLLOW`` and changed through
        that descriptor, so a name the account swaps for a link after the walk
        listed it is left alone instead of followed.
        """
        fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            st = os.fstat(fd)
            if stat.S_ISLNK(st.st_mode):
                return
            proc = f"/proc/self/fd/{fd}"
            if st.st_gid != self.gid:
                os.chown(proc, -1, self.gid)
            if stat.S_ISDIR(st.st_mode):
                add = _GROUP_RWX_SETGID if write else _GROUP_RX_SETGID
            else:
                add = _GROUP_RW if write else _GROUP_R
            if sticky:
                add |= stat.S_ISVTX
            mode = stat.S_IMODE(st.st_mode) | add
            if not write:
                mode &= ~stat.S_IWGRP  # read-only means the group cannot write what an earlier umask let it
            if mode != stat.S_IMODE(st.st_mode):
                os.chmod(proc, mode)
        finally:
            os.close(fd)


def _walk(root: Path) -> Iterable[Path]:
    """``root`` and everything below it, never following a link and never entering or touching ``.git``."""
    yield root
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != _GIT_NAME]
        base = Path(dirpath)
        for name in (*dirnames, *(f for f in filenames if f != _GIT_NAME)):
            yield base / name


def resolve_run_as(user: str | None) -> RunAs | None:
    """Look ``user`` up now and return how to run a child as it; ``None`` for no demote.

    Raises:
        RunAsError: The user does not exist, is uid 0, is the app's own user,
            or belongs to group 0.
    """
    name = (user or "").strip()
    if not name:
        return None
    try:
        entry = pwd.getpwnam(name)
    except KeyError:
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r}: no such user") from None
    if entry.pw_uid == 0:
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r} is uid 0; specialists must not run as root")
    if entry.pw_uid == os.geteuid():
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r} is the app's own user; no boundary would exist")
    groups = tuple(os.getgrouplist(entry.pw_name, entry.pw_gid))
    if 0 in groups:
        root_group = grp.getgrgid(0).gr_name
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r} is in group {root_group!r} (gid 0)")
    drop_tool = shutil.which(_DROP_TOOL)
    if drop_tool is None:
        raise RunAsError(
            f"{SPECIALIST_USER_ENV}={name!r}: {_DROP_TOOL} (util-linux) is needed to drop the child's capabilities"
        )
    return RunAs(
        user=entry.pw_name, uid=entry.pw_uid, gid=entry.pw_gid, groups=groups, home=entry.pw_dir, drop_tool=drop_tool
    )


def specialist_run_as() -> RunAs | None:
    """Resolve ``HYPERLOOM_SPECIALIST_USER``; ``None`` (today's behaviour) when unset or blank. No side effects."""
    return resolve_run_as(env_str(SPECIALIST_USER_ENV))


_CAPABILITIES = {"CAP_KILL": 5, "CAP_SETGID": 6, "CAP_SETUID": 7}


def missing_capabilities(effective: int) -> list[str]:
    """The launch capabilities absent from the effective capability bitmask ``effective``."""
    return [name for name, bit in _CAPABILITIES.items() if not effective >> bit & 1]


def missing_launch_capabilities() -> list[str]:
    """Capabilities the app lacks to start, and later stop, a child that runs as another user.

    ``setuid``/``setgid`` start it; ``kill`` is what lets the reap paths (deadline,
    stale heartbeat, done file, cancel) signal a process of another uid, and
    without it every reap ends unconfirmed. Root has all three unless they were
    dropped, so the effective set is read, not inferred from the uid. When it
    cannot be read, root is assumed to have them and anyone else to have none.
    """
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        effective = next(int(line.split()[1], 16) for line in status.splitlines() if line.startswith("CapEff:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return [] if os.geteuid() == 0 else list(_CAPABILITIES)
    return missing_capabilities(effective)


__all__ = [
    "BENCHMARK_SECRET_ENV_NAMES",
    "BLOCKED_CHILD_ENV_NAMES",
    "BLOCKED_EXTERNAL_ENV_NAMES",
    "BLOCKED_UNTRUSTED_ENV_NAMES",
    "BLOCKED_VARIANT_ENV_NAMES",
    "FILE_REFERENCING_ENV_PREFIXES",
    "LAUNCH_PROVIDED_ENV_NAMES",
    "LAUNCH_PROVIDED_ENV_PREFIXES",
    "OPERATOR_EXTRA_ENV_VAR",
    "SENSITIVE_ENV_NAME_MARKERS",
    "build_benchmark_env",
    "filter_untrusted_env_mapping",
    "is_allowed_dotenv_key",
    "is_allowed_external_env_key",
    "is_allowed_kernel_agent_env_key",
    "is_allowed_variant_env_key",
    "is_launch_provided_env_name",
    "is_python_package_root",
    "is_secret_shaped_env_name",
    "operator_extra_env",
    "redact_file_in_place",
    "redact_secret_env_values",
    "redact_secret_values",
    "scrub_benchmark_process_env",
    "scrub_child_process_env",
    "ungrounded_env_file_reason",
    "valid_env_key",
]
