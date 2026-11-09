# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator's launch file (``--reference-script``): a shell launch script or a frozen ``base_launch.json``.

The caller reads the file once and hands over its bytes, whose sha256 identifies the file in the base's source. A
shell script must launch the session's framework, fix every flag in its text, and may no longer carry
``# hyperloom-launch-controls:``; a ``base_launch.json`` is checked against its schema and digest and must be for the session's framework.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from inference_suite.contracts.base_launch import BASE_LAUNCH_SCHEMA, BaseLaunch
from inference_suite.recipe.shell_launch import ShellLaunch, parse_shell_launch

_CONTROLS_PREFIX = "# hyperloom-launch-controls:"


class UserFileError(ValueError):
    """A user launch file cannot serve as a base launch."""


@dataclass(frozen=True)
class UserFile:
    """A parsed user launch file: where it was read from, the sha256 of its bytes, and the launch it holds."""

    location: str
    sha256: str
    launch: ShellLaunch | BaseLaunch


def parse_user_file(location: str, content: bytes, *, framework: str) -> UserFile:
    """Parse the user launch file read from ``location`` for a ``framework`` session."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UserFileError(f"{location} is not UTF-8 text: {exc}") from exc
    if text.lstrip().startswith("{"):
        launch: ShellLaunch | BaseLaunch = _base_launch(location, text, framework)
    else:
        launch = _shell_launch(location, text, framework)
    return UserFile(location=location, sha256=hashlib.sha256(content).hexdigest(), launch=launch)


def _base_launch(location: str, text: str, framework: str) -> BaseLaunch:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UserFileError(f"{location} is not valid JSON: {exc}") from exc
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema != BASE_LAUNCH_SCHEMA:
        raise UserFileError(f"{location} has schema {schema!r}; a JSON launch file must be {BASE_LAUNCH_SCHEMA!r}")
    base = BaseLaunch.from_document(document)
    if base.framework != framework:
        raise UserFileError(f"{location} is a {base.framework} launch, but the session serves {framework}")
    return base


def _shell_launch(location: str, text: str, framework: str) -> ShellLaunch:
    if any(line.startswith(_CONTROLS_PREFIX) for line in text.splitlines()):
        raise UserFileError(
            f"{location} carries {_CONTROLS_PREFIX!r}, which a launch file no longer accepts; "
            "state the flags and exports the launch needs instead"
        )
    launch = parse_shell_launch(text, framework=framework)
    if not launch.has_entrypoint:
        raise UserFileError(f"{location} has no {framework} serving entrypoint")
    if launch.unresolved_flags:
        raise UserFileError(
            f"{location} sets {', '.join(launch.unresolved_flags)} from shell variables; write the literal values"
        )
    return launch


__all__ = ["UserFile", "UserFileError", "parse_user_file"]
