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
#: introduces the command. Only the ``pip`` family is covered by the KEEP-time
#: distribution closure; every other family mutates state that map cannot see.
_INSTALLER_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pip", ("pip", "pip3", "uv", "python", "python3")),
    ("apt", ("apt", "apt-get")),
    ("npm", ("npm", "pnpm", "yarn")),
    ("conda", ("conda", "mamba")),
)

_VCS_SCHEME_PREFIXES: tuple[str, ...] = ("git+", "hg+", "svn+")

_INDEX_URL_OPTIONS: frozenset[str] = frozenset({"--index-url", "--extra-index-url", "-i"})
_FIND_LINKS_OPTIONS: frozenset[str] = frozenset({"--find-links", "-f"})
_REGISTRY_OPTIONS: frozenset[str] = frozenset({"--registry", "--_auth", "--_authToken"})
_CHANNEL_OPTIONS: frozenset[str] = frozenset({"-c", "--channel"})
_ATTACHED_SHORT_VALUE_OPTIONS: tuple[str, ...] = ("-r", "-c", "-i", "-f")

_ENV_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_CREDENTIALED_URL_RE = re.compile(
    r"(?P<scheme>(?:(?:git|hg|svn)\+)?[A-Za-z][A-Za-z0-9+.-]*://)"
    r"(?P<userinfo>[^\s/?#\"']+)@(?P<location>[^\s\"']+)"
)

#: The ambient spelling of ``--index-url``; an inline assignment of one is the
#: same flag by another name, and the allowlist admits it as readily.
_PIP_INDEX_ENV_NAMES: tuple[str, ...] = ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_INDEX_URL")
_TRUSTED_BIN_PREFIX_RE = re.compile(r"^(?:/opt/[^/]+|/usr(?:/local)?|/bin|/sbin)(?:/[^/]+)*/")

#: Ambient channels proved by a variable name.
_ENV_CHANNELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pip_index_env", _PIP_INDEX_ENV_NAMES),
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
    yields the pair for ``value``. Compact short options such as ``-rFILE`` and
    ``-iURL`` are split the same way. A bare operand pairs with ``""``. This is
    what lets quoted, attached and separated spellings classify alike.
    """
    pairs: list[tuple[str, str]] = []
    pending = ""
    for token in tokens:
        if token.startswith("-") and "=" in token:
            option, _, operand = token.partition("=")
            pairs.append((option, operand))
            pending = ""
            continue
        attached = next(
            (option for option in _ATTACHED_SHORT_VALUE_OPTIONS if token.startswith(option) and token != option),
            "",
        )
        if attached:
            pairs.append((attached, token[len(attached) :].removeprefix("=")))
            pending = ""
            continue
        if token.startswith("-"):
            pairs.append((token, ""))
            pending = token
            continue
        pairs.append((pending, token))
        pending = ""
    return pairs


def assignment_pairs(assignments: Iterable[str]) -> list[tuple[str, str, str]]:
    """Pair each leading ``KEY=VALUE`` with the option its value is written as.

    The setup allowlist strips these before matching an installer, so a
    credentialed value written this way is admitted exactly as the flag
    spelling is and has to classify and sanitize the same way.

    Returns:
        ``(name, option, value)`` triples; ``option`` is ``""`` when the name is
        not the ambient spelling of a known flag.
    """
    triples: list[tuple[str, str, str]] = []
    for assignment in assignments:
        match = _ENV_ASSIGNMENT_RE.match(str(assignment))
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        triples.append((name, "--index-url" if name in _PIP_INDEX_ENV_NAMES else "", value))
    return triples


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
        parts = None
    if parts is not None and "@" in parts.netloc:
        return parts.netloc.rsplit("@", 1)[0]
    match = _CREDENTIALED_URL_RE.search(text)
    return match.group("userinfo") if match else ""


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
        parts = None
    if parts is not None and "@" in parts.netloc:
        host = parts.netloc.rsplit("@", 1)[1]
        return prefix + urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return prefix + _CREDENTIALED_URL_RE.sub(r"\g<scheme>\g<location>", text)


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


def classify_credential_value(value: str, *, option: str = "") -> str | None:
    """Name the class of credential a single ``value`` carries, or ``None``.

    Args:
        value: The token to classify; never recorded, only classified.
        option: The flag it was written under, when it was written under one.
    """
    return _class_for_pair(option, str(value or ""), family="pip") or None


def classify_credential_class(cmd: str) -> str | None:
    """Name the class of credential ``cmd`` carries, or ``None``.

    Only the class is returned; the userinfo, the host and the operand it was
    found under are never recorded.
    """
    assignments, tokens = split_env_assignments(cmd)
    if any(is_secret_shaped_env_name(name) for name, _option, _value in assignment_pairs(assignments)):
        return "env_assignment"
    family = installer_class(cmd)
    pairs = [(option, value) for _name, option, value in assignment_pairs(assignments)]
    pairs += option_operands(tokens)
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
    token shapes a URL parse cannot see. A credentialed operand is replaced by
    its class name, so neither the userinfo nor the host it named survives.
    """
    text = str(cmd or "").strip()
    if not text:
        return ""
    try:
        shlex.split(text)
    except ValueError:
        redacted = _CREDENTIALED_URL_RE.sub("<opaque_credential>", text)
        return _clip(redact_secret_values(redacted), clip)
    assignments, tokens = split_env_assignments(text)
    family = installer_class(text)
    rebuilt: list[str] = ["sudo"] if shlex.split(text)[:1] == ["sudo"] else []
    for name, option, value in assignment_pairs(assignments):
        found = _class_for_pair(option, value, family=family)
        rebuilt.append(f"{name}=<{found}>" if found else f"{name}={value}")
    for option, operand in option_operands(tokens):
        if not operand:
            rebuilt.append(option)
            continue
        # The whole operand goes, not just its userinfo: the host it names is
        # the private index the credential unlocks, and the class alone is what
        # a replay operator needs to know.
        found = _class_for_pair(option, operand, family=family)
        safe = f"<{found}>" if found else operand
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
