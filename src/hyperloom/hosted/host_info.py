# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The inputs a host injects into a hosted run, read and checked once, at start-up.

A host (the application launcher) hands a hosted run its configuration through
environment variables.  This module is the only reader of them: it checks every
variable the run needs, names *all* the ones that are missing or malformed in a
single :class:`HostInfoError`, and returns a frozen :class:`HostInfo` with typed
fields.  A run that would otherwise fail hours in, on its first sandbox create
or its first archive upload, fails before it starts instead.

Which variables a run needs depends on what it hosts, so :func:`read_host_info`
takes the groups to require (:class:`HostGroup`).  The run, state and input
variables are always required.  A group that is not required may still be set;
it is then checked like a required one and read into its field.

Credentials (sandbox keys, the MCP token, archive credentials) are held in
:class:`Secret`, whose ``repr``, ``str`` and format output never contain the
value, and no error message here carries a value, only variable names and a
reason.  :func:`take_host_info` also removes every host-only input from the
environment it read, so a child process started without an explicit ``env=``
cannot inherit them; the credential names are also
in :data:`hyperloom.common.env_safety.BENCHMARK_SECRET_ENV_NAMES`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit


class Var:
    """Names of the host-injected variables (one place, so a name is spelled once)."""

    INPUT_FILE = "HYPERLOOM_INPUT_FILE"
    RUN_ID = "HYPERLOOM_RUN_ID"
    RUN_CREATED_AT = "HYPERLOOM_RUN_CREATED_AT"
    STATE_DIR = "HYPERLOOM_STATE_DIR"
    EVENTS_DIR = "HYPERLOOM_EVENTS_DIR"
    MCP_PORT = "HYPERLOOM_MCP_PORT"
    MCP_TOKEN = "HYPERLOOM_MCP_TOKEN"
    MCP_PUBLIC_URL = "HYPERLOOM_MCP_PUBLIC_URL"
    SANDBOX_TEMPLATES = "HYPERLOOM_SANDBOX_TEMPLATES"
    E2B_API_URL = "E2B_API_URL"
    E2B_SANDBOX_URL = "E2B_SANDBOX_URL"
    SANDBOX_KEY_MEASUREMENT = "HYPERLOOM_SANDBOX_KEY_MEASUREMENT"
    SANDBOX_KEY_SUBRUN = "HYPERLOOM_SANDBOX_KEY_SUBRUN"
    SANDBOX_KEY_DEV = "HYPERLOOM_SANDBOX_KEY_DEV"
    RESULT_ARCHIVE = "HYPERLOOM_RESULT_ARCHIVE"
    ARCHIVE_ENDPOINT_URL = "AWS_ENDPOINT_URL_S3"
    ARCHIVE_REGION = "AWS_REGION"
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
    GRANT_LIMITS = "HYPERLOOM_GRANT_LIMITS"
    SOURCE_REPO = "HYPERLOOM_SOURCE_REPO"
    SOURCE_SHA = "HYPERLOOM_SOURCE_SHA"
    NODE_NAME = "NODE_NAME"
    CLAW_SESSION_ID = "CLAW_SESSION_ID"


#: Every variable whose value is a credential.  Removed from the environment by :func:`take_host_info`.
SECRET_VARS: tuple[str, ...] = (
    Var.MCP_TOKEN,
    Var.SANDBOX_KEY_MEASUREMENT,
    Var.SANDBOX_KEY_SUBRUN,
    Var.SANDBOX_KEY_DEV,
    Var.AWS_ACCESS_KEY_ID,
    Var.AWS_SECRET_ACCESS_KEY,
    Var.AWS_SESSION_TOKEN,
)


#: Variables other code reads from the environment, so :func:`take_host_info` leaves them in place.
_KEPT_IN_ENVIRONMENT: frozenset[str] = frozenset({Var.NODE_NAME, Var.CLAW_SESSION_ID})

#: Every host-only input: removed from the environment once read, so a child process started without an
#: explicit ``env=`` does not inherit it.  The values live on :class:`HostInfo` from then on.
HOST_ONLY_VARS: tuple[str, ...] = tuple(
    value for name, value in vars(Var).items() if not name.startswith("_") and value not in _KEPT_IN_ENVIRONMENT
)


class HostGroup(str, Enum):
    """A set of inputs a run needs only when it hosts the part that reads them."""

    MCP = "mcp"
    MCP_PUBLIC_URL = "mcp-public-url"
    SANDBOX = "sandbox"
    MEASUREMENT_KEY = "measurement-key"
    SUBRUN_KEY = "subrun-key"
    DEV_KEY = "dev-key"


_ALWAYS_REQUIRED: tuple[str, ...] = (Var.INPUT_FILE, Var.RUN_ID, Var.RUN_CREATED_AT, Var.STATE_DIR)

_GROUP_VARS: Mapping[HostGroup, tuple[str, ...]] = {
    HostGroup.MCP: (Var.MCP_PORT, Var.MCP_TOKEN),
    HostGroup.MCP_PUBLIC_URL: (Var.MCP_PUBLIC_URL,),
    HostGroup.SANDBOX: (Var.SANDBOX_TEMPLATES, Var.E2B_API_URL, Var.E2B_SANDBOX_URL),
    HostGroup.MEASUREMENT_KEY: (Var.SANDBOX_KEY_MEASUREMENT,),
    HostGroup.SUBRUN_KEY: (Var.SANDBOX_KEY_SUBRUN,),
    HostGroup.DEV_KEY: (Var.SANDBOX_KEY_DEV,),
}

#: Required by default: the shape of the first hosted profile (run, state, input and the MCP server).
DEFAULT_REQUIRE: frozenset[HostGroup] = frozenset({HostGroup.MCP})

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_OR_SPACE_RE = re.compile(r"[\s\x00-\x1f\x7f]")


class HostInfoError(ValueError):
    """The host inputs are missing or malformed.  Names variables and reasons, never values."""

    def __init__(self, missing: Iterable[str], invalid: Mapping[str, str]) -> None:
        self.missing: tuple[str, ...] = tuple(missing)
        self.invalid: dict[str, str] = dict(invalid)
        parts = []
        if self.missing:
            parts.append("missing " + ", ".join(self.missing))
        if self.invalid:
            parts.append("invalid " + "; ".join(f"{name} ({why})" for name, why in self.invalid.items()))
        super().__init__("host inputs: " + " | ".join(parts))


class Secret:
    """A credential that does not print: ``repr``, ``str`` and ``format`` show a placeholder."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The credential itself, for the one call site that sends it."""
        return self._value

    def __repr__(self) -> str:
        return "Secret(<redacted>)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return format(repr(self), spec)

    def __reduce__(self) -> str:
        raise TypeError("Secret is not serializable")

    def __copy__(self) -> Secret:
        return self

    def __deepcopy__(self, memo: object) -> Secret:
        return self


@dataclass(frozen=True)
class ArchiveLocation:
    """Where results are archived: ``s3://<bucket>/<prefix>``."""

    bucket: str
    prefix: str


@dataclass(frozen=True)
class ArchiveCredentials:
    """Static credentials for the archive store; the session token is present for temporary ones."""

    access_key_id: Secret
    secret_access_key: Secret
    session_token: Secret | None


@dataclass(frozen=True)
class SandboxKeys:
    """One sandbox API key per role; a role the run does not host is ``None``."""

    measurement: Secret | None
    subrun: Secret | None
    dev: Secret | None


@dataclass(frozen=True)
class GrantLimits:
    """The totals the host granted the run; a total the host did not state is ``None``."""

    max_sandboxes: int | None
    max_gpus: int | None
    gpu_hours: float | None
    max_runtime_sec: int | None


@dataclass(frozen=True)
class HostInfo:
    """Everything the host injected, typed.  Optional fields are ``None`` when the host did not set them."""

    input_file: Path
    run_id: str
    run_created_at: datetime
    state_dir: Path
    events_dir: Path
    mcp_port: int | None
    mcp_token: Secret | None
    mcp_public_url: str | None
    sandbox_templates: dict[str, object] | None
    e2b_api_url: str | None
    e2b_sandbox_url: str | None
    sandbox_keys: SandboxKeys
    result_archive: ArchiveLocation | None  # None: results are archived on the shared storage
    archive_endpoint_url: str | None
    archive_region: str | None
    archive_credentials: ArchiveCredentials | None
    node_name: str | None
    claw_session_id: str | None
    grant_limits: GrantLimits | None = None
    source_repo: str | None = None  # what the runner sandbox's start-up script fetches
    source_sha: str | None = None  # full 40-hex commit id of the version this run executes


class _Reader:
    """Reads one variable at a time and collects, rather than raises, what is wrong with it."""

    def __init__(self, env: Mapping[str, str], needed: frozenset[str]) -> None:
        self._env = env
        self._needed = needed
        self.missing: list[str] = []
        self.invalid: dict[str, str] = {}

    def raw(self, name: str, *, force: bool = False) -> str | None:
        value = self._env.get(name, "")
        if value.strip():
            return value
        if force or name in self._needed:
            self.missing.append(name)
        return None

    def present(self, name: str) -> bool:
        return bool(self._env.get(name, "").strip())

    def _parsed(self, name: str, parse, *, force: bool = False):
        value = self.raw(name, force=force)
        if value is None:
            return None
        try:
            return parse(value)
        except ValueError as exc:
            self.invalid[name] = str(exc)
            return None

    def text(self, name: str) -> str | None:
        return self._parsed(name, _plain_text)

    def path(self, name: str) -> Path | None:
        return self._parsed(name, _absolute_path)

    def existing_file(self, name: str) -> Path | None:
        return self._parsed(name, _existing_file)

    def existing_dir(self, name: str) -> Path | None:
        return self._parsed(name, _existing_dir)

    def url(self, name: str) -> str | None:
        return self._parsed(name, _http_url)

    def port(self, name: str) -> int | None:
        return self._parsed(name, _port)

    def timestamp(self, name: str) -> datetime | None:
        return self._parsed(name, _timestamp)

    def templates(self, name: str) -> dict[str, object] | None:
        return self._parsed(name, _templates)

    def archive(self, name: str) -> ArchiveLocation | None:
        return self._parsed(name, _archive_location)

    def grant_limits(self, name: str) -> GrantLimits | None:
        return self._parsed(name, _grant_limits)

    def commit_sha(self, name: str) -> str | None:
        return self._parsed(name, _commit_sha)

    def secret(self, name: str, *, force: bool = False) -> Secret | None:
        value = self._parsed(name, _plain_text, force=force)
        return None if value is None else Secret(value)


def _plain_text(value: str) -> str:
    if _CONTROL_OR_SPACE_RE.search(value):
        raise ValueError("must not contain whitespace or control characters")
    return value


def _commit_sha(value: str) -> str:
    if not _COMMIT_SHA_RE.match(value):
        raise ValueError("must be a full lowercase 40-hex commit id")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("must be an absolute path")
    return path


def _existing_file(value: str) -> Path:
    path = _absolute_path(value)
    if not path.is_file():
        raise ValueError("is not an existing file")
    return path


def _existing_dir(value: str) -> Path:
    path = _absolute_path(value)
    if not path.is_dir():
        raise ValueError("is not an existing directory")
    return path


def _http_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        raise ValueError("is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("must be an http or https URL with a host")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("must not carry credentials, a query or a fragment")
    return value


def _port(value: str) -> int:
    if not (value.isascii() and value.isdecimal()) or not 1 <= int(value) <= 65535:
        raise ValueError("must be an integer from 1 to 65535")
    return int(value)


def _timestamp(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("must be an ISO 8601 timestamp") from None
    if stamp.tzinfo is None:
        raise ValueError("must carry a UTC offset or Z")
    return stamp


def _templates(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})") from None
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("must be a non-empty JSON object")
    if not all(_is_template_entry(entry) for entry in parsed.values()):
        raise ValueError("every entry must be a template name or an object of template names")
    return parsed


def _is_template_name(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_template_entry(entry: object) -> bool:
    if isinstance(entry, dict):
        return bool(entry) and all(_is_template_name(name) for name in entry.values())
    return _is_template_name(entry)


_GRANT_COUNTS = ("max_sandboxes", "max_gpus", "max_runtime_sec")


def _grant_total(value: object, name: str, *, whole: bool) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0 or value == float("inf"):
        raise ValueError(f"{name} must be a number >= 0")
    if whole and value != int(value):
        raise ValueError(f"{name} must be a whole number")
    return int(value) if whole else value


def _grant_limits(value: str) -> GrantLimits:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})") from None
    if not isinstance(parsed, dict):
        raise ValueError("must be a JSON object")
    totals = {
        name: _grant_total(parsed.get(name), name, whole=name in _GRANT_COUNTS)
        for name in (*_GRANT_COUNTS, "gpu_hours")
    }
    return GrantLimits(**totals)  # type: ignore[arg-type]


def _archive_location(value: str) -> ArchiveLocation:
    parts = urlsplit(value)
    if parts.scheme != "s3" or parts.query or parts.fragment:
        raise ValueError("only s3://<bucket>/<prefix> or unset (shared storage) is accepted")
    if not _BUCKET_RE.match(parts.netloc):
        raise ValueError("bucket is not a valid S3 bucket name")
    # Split the raw path: S3 keys are literal, so a path normaliser would accept a malformed prefix as another one.
    raw = parts.path.removeprefix("/").removesuffix("/")
    segments = raw.split("/") if raw else []
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError("prefix must not contain empty, '.' or '..' segments")
    return ArchiveLocation(bucket=parts.netloc, prefix=raw)


def _needed_names(require: Collection[HostGroup]) -> frozenset[str]:
    names = set(_ALWAYS_REQUIRED)
    for group in require:
        names.update(_GROUP_VARS[HostGroup(group)])
    return frozenset(names)


def _sandbox_keys(reader: _Reader) -> SandboxKeys:
    return SandboxKeys(
        measurement=reader.secret(Var.SANDBOX_KEY_MEASUREMENT),
        subrun=reader.secret(Var.SANDBOX_KEY_SUBRUN),
        dev=reader.secret(Var.SANDBOX_KEY_DEV),
    )


def _archive_credentials(reader: _Reader, archive: ArchiveLocation | None) -> ArchiveCredentials | None:
    # Required only for an s3:// archive.  Otherwise the pair is all or nothing: one half set means the host meant
    # to give credentials and got them wrong.
    pair = (Var.AWS_ACCESS_KEY_ID, Var.AWS_SECRET_ACCESS_KEY)
    force = archive is not None or any(reader.present(name) for name in pair)
    key_id, secret = (reader.secret(name, force=force) for name in pair)
    token = reader.secret(Var.AWS_SESSION_TOKEN)
    if key_id is None or secret is None:
        return None
    return ArchiveCredentials(access_key_id=key_id, secret_access_key=secret, session_token=token)


def _build(reader: _Reader) -> HostInfo:
    state_dir = reader.existing_dir(Var.STATE_DIR)
    events_dir = reader.path(Var.EVENTS_DIR)
    result_archive = reader.archive(Var.RESULT_ARCHIVE)
    return HostInfo(
        input_file=reader.existing_file(Var.INPUT_FILE),  # type: ignore[arg-type]
        run_id=reader.text(Var.RUN_ID),  # type: ignore[arg-type]
        run_created_at=reader.timestamp(Var.RUN_CREATED_AT),  # type: ignore[arg-type]
        state_dir=state_dir,  # type: ignore[arg-type]
        events_dir=events_dir or (state_dir / "events" if state_dir else None),  # type: ignore[arg-type]
        mcp_port=reader.port(Var.MCP_PORT),
        mcp_token=reader.secret(Var.MCP_TOKEN),
        mcp_public_url=reader.url(Var.MCP_PUBLIC_URL),
        sandbox_templates=reader.templates(Var.SANDBOX_TEMPLATES),
        e2b_api_url=reader.url(Var.E2B_API_URL),
        e2b_sandbox_url=reader.url(Var.E2B_SANDBOX_URL),
        sandbox_keys=_sandbox_keys(reader),
        result_archive=result_archive,
        archive_endpoint_url=reader.url(Var.ARCHIVE_ENDPOINT_URL),
        archive_region=reader.text(Var.ARCHIVE_REGION),
        archive_credentials=_archive_credentials(reader, result_archive),
        node_name=reader.text(Var.NODE_NAME),
        claw_session_id=reader.text(Var.CLAW_SESSION_ID),
        grant_limits=reader.grant_limits(Var.GRANT_LIMITS),
        source_repo=reader.text(Var.SOURCE_REPO),
        source_sha=reader.commit_sha(Var.SOURCE_SHA),
    )


def read_host_info(env: Mapping[str, str], *, require: Collection[HostGroup] = DEFAULT_REQUIRE) -> HostInfo:
    """Read and check the host-injected inputs from *env*.

    The run, state and input variables are always required; *require* adds the
    groups the run hosts (sandbox variables when the measurement or sandbox plug
    is hosted, the archive when results are archived, and so on).  Every missing
    or malformed variable is reported in one :class:`HostInfoError`.  *env* is
    not modified.
    """
    reader = _Reader(env, _needed_names(require))
    info = _build(reader)
    if reader.missing or reader.invalid:
        raise HostInfoError(reader.missing, reader.invalid)
    return info


def take_host_info(environ: MutableMapping[str, str], *, require: Collection[HostGroup] = DEFAULT_REQUIRE) -> HostInfo:
    """:func:`read_host_info`, then remove every host-only input from *environ* (:data:`HOST_ONLY_VARS`).

    ``NODE_NAME`` and ``CLAW_SESSION_ID`` stay: other code reads them.

    Nothing is removed when the read fails, so the caller can still report on the environment it was given.
    """
    info = read_host_info(environ, require=require)
    for name in HOST_ONLY_VARS:
        environ.pop(name, None)
    return info
