# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's frozen base launch, ``base_launch.json``.

A fresh launch resolves the base once and writes it here; every later read, a resume's included, takes it from this
file and never resolves it again. Reading refuses a file whose content no longer matches its recorded digest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.session_files import shared_options
from hyperloom.common.url_userinfo import strip_url_userinfo
from hyperloom_common.io import atomic_write_json
from inference_suite.contracts import BaseLaunch

BASE_LAUNCH_FILE = "base_launch.json"
#: The parts of the frozen base a session breakdown reports: where it was resolved from (with the sha256 of each file
#: read), the image it named and the topology it runs at. Its args and env stay in the file.
BASE_LAUNCH_METADATA_KEYS = ("source", "image", "topology")


def base_launch_path(session_dir: Path) -> Path:
    return Path(session_dir) / BASE_LAUNCH_FILE


def write_base_launch(session_dir: Path, base: BaseLaunch) -> Path:
    """Freeze ``base`` as the session's base launch, ``664`` below shared storage unless it holds a credential;
    returns the file written."""
    path = base_launch_path(session_dir)
    document = base.to_document()
    atomic_write_json(path, document, options=shared_options(document))  # a later leg may be another account
    return path


def read_base_launch(session_dir: Path) -> BaseLaunch | None:
    """The session's frozen base, or ``None`` for a session that froze none (a scriptable or AgentX workload)."""
    try:
        text = base_launch_path(session_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return BaseLaunch.from_document(json.loads(text))


def base_launch_metadata(session_dir: Path) -> dict[str, Any] | None:
    """The :data:`BASE_LAUNCH_METADATA_KEYS` of the base frozen in ``session_dir``, or ``None`` when it froze none.

    Every location in ``source`` is reported without a URL's userinfo or a recognizable credential value: an
    operator's ``--reference-script`` is recorded as given, and may be a URL that embeds one.

    Raises:
        OSError: The file exists and cannot be read.
        ValueError: It is no base launch, or fails its digest check (which covers what the base launches, not
            ``source`` or ``image``).
    """
    base = read_base_launch(session_dir)
    if base is None:
        return None
    document = base.to_document()
    found = {key: document[key] for key in BASE_LAUNCH_METADATA_KEYS}
    found["source"] = _redacted_source(found["source"])
    return found


#: The ``source`` fields that name a location: a repository, a file, a variant or an API response.
_SOURCE_LOCATIONS = ("repo", "path", "entry", "locator")


def _redacted_location(location: Any) -> Any:
    return redact_secret_values(strip_url_userinfo(location)) if isinstance(location, str) else location


def _redacted_source(source: Any) -> Any:
    if not isinstance(source, dict):
        return source
    redacted = {key: _redacted_location(value) if key in _SOURCE_LOCATIONS else value for key, value in source.items()}
    files = source.get("files")
    if isinstance(files, dict):
        redacted["files"] = {_redacted_location(path): sha256 for path, sha256 in files.items()}
    return redacted


def shared_state_fields(base: BaseLaunch | None) -> dict[str, Any]:
    """The ``SharedState`` fields a fresh session seeds from its frozen ``base``: the digest naming the frozen file."""
    if base is None:
        return {}
    return {"base_launch_digest": base.digest()}


__all__ = [
    "BASE_LAUNCH_FILE",
    "BASE_LAUNCH_METADATA_KEYS",
    "base_launch_metadata",
    "base_launch_path",
    "read_base_launch",
    "shared_state_fields",
    "write_base_launch",
]
