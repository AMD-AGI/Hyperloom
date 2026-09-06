# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Credential *classes* and *channels* for replayed enablement commands.

Names which class of credential a replay operator must supply. No value, host,
path or userinfo is ever recorded, and nothing here opens a file or
authenticates: an ambient channel is observed by the presence of a variable name
or of a fixed location the admitted installers read.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

from hyperloom.common.env_safety import is_secret_shaped_env_name, redact_secret_values

#: Installer families the setup allowlist admits, keyed by the token that
#: introduces the command. Only the Python family is covered by the KEEP-time
#: distribution closure.
_INSTALLER_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pip", ("pip", "pip3", "uv", "python", "python3")),
    ("apt", ("apt", "apt-get")),
    ("npm", ("npm", "pnpm", "yarn")),
    ("conda", ("conda", "mamba")),
)

_PYTHON_INSTALLER_FAMILY = "pip"

_VCS_SCHEME_PREFIXES: tuple[str, ...] = ("git+", "hg+", "svn+")

_INDEX_URL_OPTIONS: frozenset[str] = frozenset({"--index-url", "--extra-index-url", "-i"})
_FIND_LINKS_OPTIONS: frozenset[str] = frozenset({"--find-links", "-f"})
_REGISTRY_OPTIONS: frozenset[str] = frozenset({"--registry", "--_auth", "--_authToken"})
_CHANNEL_OPTIONS: frozenset[str] = frozenset({"-c", "--channel"})

_ENV_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_TRUSTED_BIN_PREFIX_RE = re.compile(r"^(?:/opt/[^/]+|/usr(?:/local)?|/bin|/sbin)(?:/[^/]+)*/")

#: Ambient channels proved by a variable name.
_ENV_CHANNELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pip_index_env", ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_INDEX_URL")),
    ("pip_config", ("PIP_CONFIG_FILE",)),
    ("netrc", ("NETRC",)),
    ("keyring", ("PIP_KEYRING_PROVIDER",)),
    ("npm_config", ("NPM_CONFIG_REGISTRY",)),
    ("conda_config", ("CONDA_TOKEN",)),
    ("ssh_agent", ("SSH_AUTH_SOCK",)),
    ("git_ssh_command", ("GIT_SSH_COMMAND",)),
    ("git_credential_helper", ("GIT_ASKPASS", "GIT_TERMINAL_PROMPT")),
)

_HOME_RELATIVE_CHANNEL_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pip_config", (".config/pip/pip.conf", ".pip/pip.conf")),
    ("netrc", (".netrc",)),
    ("npm_config", (".npmrc",)),
    ("conda_config", (".condarc",)),
    ("git_credential_helper", (".git-credentials",)),
)

#: Locations relative to the filesystem root, so a caller can point the probe at
#: a controlled tree instead of the host's own.
_ROOT_RELATIVE_CHANNEL_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pip_config", ("etc/pip.conf",)),
    ("apt_auth", ("etc/apt/auth.conf", "etc/apt/auth.conf.d")),
)


def installer_class(cmd: str) -> str:
    """Return the installer family that introduces ``cmd`` (``""`` when none).

    ``dependency_closure_status`` may only claim a closed environment for the
    Python family; every other family mutates state the distribution map cannot
    observe.
    """
    _, tokens = split_env_assignments(cmd)
    if not tokens:
        return ""
    head = Path(_TRUSTED_BIN_PREFIX_RE.sub("", tokens[0], count=1)).name
    for family, heads in _INSTALLER_FAMILIES:
        if head in heads:
            return family
    return ""


def is_python_installer(cmd: str) -> bool:
    """True when ``cmd``'s effects are visible to the KEEP-time distribution map."""
    return installer_class(cmd) == _PYTHON_INSTALLER_FAMILY


def split_env_assignments(cmd: str) -> tuple[list[str], list[str]]:
    """Split ``cmd`` into its leading ``KEY=VALUE`` prefix and remaining tokens.

    The allowlist strips a leading ``sudo`` and those assignments before matching
    an installer, so classification has to see the same normalized stream.

    Returns:
        A ``(assignments, tokens)`` pair; both empty when ``cmd`` does not
        tokenize.
    """
    try:
        tokens = shlex.split(str(cmd or "").strip())
    except ValueError:
        return [], []
    if tokens and tokens[0] == "sudo":
        tokens = tokens[1:]
    assignments: list[str] = []
    while tokens and _ENV_ASSIGNMENT_RE.match(tokens[0]):
        assignments.append(tokens.pop(0))
    return assignments, tokens


def option_operands(tokens: Iterable[str]) -> list[tuple[str, str]]:
    """Pair each token with the option it is an operand of.

    An attached ``--opt=value`` yields one pair; a separated ``--opt value``
    yields the pair for ``value``. A bare operand pairs with ``""``. This is what
    lets a quoted, attached and separated spelling of one flag classify alike.
    """
    pairs: list[tuple[str, str]] = []
    pending = ""
    for token in tokens:
        if token.startswith("-") and "=" in token:
            option, _, operand = token.partition("=")
            pairs.append((option, operand))
            pending = ""
            continue
        if token.startswith("-"):
            pairs.append((token, ""))
            pending = token
            continue
        pairs.append((pending, token))
        pending = ""
    return pairs


def url_userinfo(token: str) -> str:
    """Return the userinfo component of ``token``, or ``""``.

    A ``git+``/``hg+``/``svn+`` prefix is stepped over first: without that the
    scheme parse sees ``git+https`` and the userinfo is never reached.
    """
    text = str(token or "")
    for prefix in _VCS_SCHEME_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if "://" not in text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    return parts.netloc.rsplit("@", 1)[0] if "@" in parts.netloc else ""


def strip_url_userinfo(url: str) -> str:
    """Return ``url`` with any userinfo component removed."""
    text = str(url or "")
    prefix = ""
    for candidate in _VCS_SCHEME_PREFIXES:
        if text.startswith(candidate):
            prefix, text = candidate, text[len(candidate) :]
            break
    if "://" not in text:
        return str(url or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return str(url or "")
    if "@" not in parts.netloc:
        return str(url or "")
    host = parts.netloc.rsplit("@", 1)[1]
    return prefix + urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _class_for_pair(option: str, operand: str, *, family: str) -> str:
    """Return the credential class an ``(option, operand)`` pair carries."""
    if not url_userinfo(operand):
        return ""
    if option in _INDEX_URL_OPTIONS:
        return "index_url"
    if option in _FIND_LINKS_OPTIONS:
        return "find_links"
    if operand.startswith(_VCS_SCHEME_PREFIXES):
        return "vcs_url"
    if family == "npm" and option in _REGISTRY_OPTIONS:
        return "registry"
    if family == "conda" and option in _CHANNEL_OPTIONS:
        return "channel"
    if family == "apt":
        return "apt_source"
    return "opaque_credential"


#: Table order; the first match wins, so classification is deterministic.
_CLASS_ORDER: tuple[str, ...] = (
    "index_url",
    "find_links",
    "vcs_url",
    "registry",
    "channel",
    "apt_source",
    "opaque_credential",
)


def classify_credential_class(cmd: str) -> str | None:
    """Name the class of credential ``cmd`` carries, or ``None``.

    Only the class is returned; the userinfo, the host and the operand it was
    found under are never recorded.
    """
    assignments, tokens = split_env_assignments(cmd)
    for assignment in assignments:
        match = _ENV_ASSIGNMENT_RE.match(assignment)
        if match and is_secret_shaped_env_name(match.group(1)):
            return "env_assignment"
    family = installer_class(cmd)
    pairs = option_operands(tokens)
    found = [_class_for_pair(option, operand, family=family) for option, operand in pairs]
    ranked = [name for name in found if name]
    if ranked:
        return sorted(ranked, key=_CLASS_ORDER.index)[0]
    for option, operand in pairs:
        if option in _REGISTRY_OPTIONS and option != "--registry":
            return "registry"
        if operand and redact_secret_values(operand) != operand:
            return "opaque_credential"
    return None


def sanitize_command_text(cmd: str, *, clip: int = 0) -> str:
    """Return ``cmd`` with credential material removed.

    A URL-aware pass runs first, because the shipped redactor matches assignment
    and header shapes and never parses a URL, so a credentialed ``--index-url``
    would otherwise pass through unchanged. The redactor then covers the literal
    token shapes a URL parse cannot see.
    """
    text = str(cmd or "").strip()
    if not text:
        return ""
    try:
        tokens = shlex.split(text)
    except ValueError:
        return _clip(redact_secret_values(text), clip)
    family = installer_class(text)
    rebuilt: list[str] = []
    for option, operand in option_operands(tokens):
        if not operand:
            rebuilt.append(option)
            continue
        safe = strip_url_userinfo(operand) if _class_for_pair(option, operand, family=family) else operand
        if option and option.startswith("-") and rebuilt and rebuilt[-1] == option:
            rebuilt[-1] = f"{option} {safe}"
        elif option and option.startswith("-"):
            rebuilt.append(f"{option}={safe}")
        else:
            rebuilt.append(safe)
    return _clip(redact_secret_values(" ".join(rebuilt)), clip)


def _clip(text: str, clip: int) -> str:
    return text if clip <= 0 or len(text) <= clip else text[:clip] + "..."


def detect_credential_channels(env: Mapping[str, str] | None, *, fs_root: str | Path = "/") -> list[str]:
    """Name the ambient credential channels ``env`` makes available.

    A plain ``pip install foo`` can succeed on an ambient index redirect or a
    stored credential while carrying no credential-shaped token at all, so a
    token-only classifier would say nothing about it.

    Args:
        env: The environment the consumer was actually given.
        fs_root: Filesystem root the fixed locations are resolved against.
    """
    environ = dict(env or {})
    found: set[str] = set()
    for channel, names in _ENV_CHANNELS:
        if any(str(environ.get(name) or "").strip() for name in names):
            found.add(channel)
    home = str(environ.get("HOME") or "").strip()
    if home:
        for channel, rels in _HOME_RELATIVE_CHANNEL_FILES:
            if any(Path(home, rel).exists() for rel in rels):
                found.add(channel)
    for channel, rels in _ROOT_RELATIVE_CHANNEL_FILES:
        if any(Path(fs_root, rel).exists() for rel in rels):
            found.add(channel)
    return sorted(found)
