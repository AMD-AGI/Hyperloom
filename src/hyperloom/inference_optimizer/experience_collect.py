# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish a session's Framework attempts as Experiences through the Hyperloom-KB mapping.

The projection from ``session_breakdown.json`` to Experiences is the packaged
``hyperloom-sbd-v6`` mapping in ``hyperloom_kb``. The optimizer owns only the
two ends: recording the attempt fields that mapping reads, and handing the
written breakdown to ``collect``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.experience_kb_service import (
    MAPPING,
    REQUEST_TIMEOUT_SECONDS,
    auto_push,
    check_auto_push,
    spool_root,
)
from hyperloom_kb import ConfigurationError, RemoteClientError, experience_kb_from_env
from hyperloom_kb.collect import MappingError, SourceDocumentError, collect, load_mapping

log = logging.getLogger(__name__)

URL_ENV = "HYPERLOOM_KB_URL"
RECEIPT = Path("reports") / "experience_collect.json"


def enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return bool(str(values.get(URL_ENV) or "").strip())


def validate_config() -> None:
    """Raise when a configured Experience KB could not accept this run's Experiences, so launch can warn of it."""

    if not enabled():
        return
    load_mapping(MAPPING)
    experience_kb_from_env(spool_root=spool_root(), timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    # Surfaces an unusable auto-push setting at launch rather than three hours later; it never stops the run.
    check_auto_push()


def collect_session(session_dir: Path, breakdown: Mapping[str, Any]) -> None:
    """Publish ``breakdown``'s Framework attempts; failures never touch the written breakdown."""

    if not enabled():
        return
    try:
        target = experience_kb_from_env(spool_root=spool_root(), timeout_seconds=REQUEST_TIMEOUT_SECONDS)
        report = collect(MAPPING, breakdown, kb=target, receipt=Path(session_dir) / RECEIPT)
    except (ConfigurationError, MappingError, RemoteClientError, SourceDocumentError, OSError):
        log.warning(
            "Experience collection failed for %s; the session breakdown remains authoritative",
            session_dir,
            exc_info=True,
        )
        return
    log.info("Experience collection for %s: %s", session_dir, report.to_dict()["counts"])
    auto_push()


__all__ = ["RECEIPT", "URL_ENV", "collect_session", "enabled", "validate_config"]
