# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish a session's Framework attempts as Experiences through the Hyperloom-KB mapping.

The projection from ``session_breakdown.json`` to Experiences is the packaged
``hyperloom-sbd-v6`` mapping in the ``hyperloom_kb`` SDK. Hyperloom owns only
the two ends: recording the attempt fields that mapping reads, and handing the
written breakdown to ``collect``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

URL_ENV = "HYPERLOOM_KB_URL"
MAPPING = "hyperloom-sbd-v6"
RECEIPT = Path("reports") / "experience_collect.json"


def enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return bool(str(values.get(URL_ENV) or "").strip())


def mapping_schema_ref() -> str:
    """Return the declaration the packaged mapping produces, proving the installed SDK can collect."""

    from hyperloom_kb.collect import load_mapping

    return load_mapping(MAPPING).declaration.schema_ref


def validate_config() -> None:
    """Fail before a run starts when a configured Experience KB could not accept its Experiences."""

    if not enabled():
        return
    expected = mapping_schema_ref()
    from hyperloom_kb import ConfigurationError, experience_kb_from_env

    target = experience_kb_from_env()
    if target.schema_ref != expected:
        raise ConfigurationError(
            f"{MAPPING} produces {expected}, but the configured Experience KB validates {target.schema_ref}"
        )


def collect_session(session_dir: Path, breakdown: Mapping[str, Any]) -> None:
    """Publish ``breakdown``'s Framework attempts; failures never touch the written breakdown."""

    if not enabled():
        return
    from hyperloom_kb import ConfigurationError, RemoteClientError
    from hyperloom_kb.collect import MappingError, SourceDocumentError, collect

    try:
        report = collect(MAPPING, breakdown, receipt=Path(session_dir) / RECEIPT)
    except (ConfigurationError, MappingError, RemoteClientError, SourceDocumentError, OSError):
        log.warning(
            "Experience collection failed for %s; the session breakdown remains authoritative",
            session_dir,
            exc_info=True,
        )
        return
    log.info("Experience collection for %s: %s", session_dir, report.to_dict()["counts"])


__all__ = ["MAPPING", "RECEIPT", "URL_ENV", "collect_session", "enabled", "mapping_schema_ref", "validate_config"]
