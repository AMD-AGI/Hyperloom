# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Structured logging setup for framework-agent.

Addresses the zhenggong "0-observability" gap noted in the merged design
(§4.4.1): long-running build/benchmark stages used to be a black box.

Public API:

* :func:`configure_logging`       — explicit init (called by ``fa`` CLI).
* :func:`get_logger`              — module-level helper that lazy-inits.
* :func:`stage_log`               — context manager emitting structured
                                    start/done/failed envelopes around a
                                    per-stage block (per-candidate build,
                                    bench, accuracy, kb-sediment, ...).

Design choices:

* No import-time side effects (safe to import from libraries / tests).
* Single root logger ``framework_agent`` with module-prefixed children;
  callers use ``get_logger(__name__)`` to inherit.
* Text format by default (human-readable); JSON Lines optional via
  ``FRAMEWORK_AGENT_LOG_JSON=1`` for machine ingest.
* Level resolution: explicit arg > ``FRAMEWORK_EXPLORER_LOG_LEVEL`` env
  > ``FRAMEWORK_AGENT_LOG_LEVEL`` env (alias) > ``INFO``.
* Optional file sink (``--log-file`` / env) appended alongside stderr.
* Re-entrant: a second ``configure_logging`` call replaces handlers but
  preserves child loggers' levels (so tests can override safely).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator


# Single root for the package; all module loggers descend from this so a
# single configure call propagates to every emitter.
_ROOT_NAME = "framework_agent"
_DEFAULT_FMT = (
    "%(asctime)s %(levelname)-5s %(name)s :: %(message)s"
)
# Env vars surfaced in the merged design §14.3.
_LEVEL_ENVS: tuple[str, ...] = (
    "FRAMEWORK_EXPLORER_LOG_LEVEL",
    "FRAMEWORK_AGENT_LOG_LEVEL",
)
_JSON_ENV = "FRAMEWORK_AGENT_LOG_JSON"
_FILE_ENV = "FRAMEWORK_AGENT_LOG_FILE"


def _resolve_level(explicit: str | int | None) -> int:
    """Pick the effective log level (explicit > env > INFO)."""
    if explicit is not None:
        if isinstance(explicit, int):
            return explicit
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
        level = logging.getLevelName(str(explicit).upper())
        if isinstance(level, int):
            return level
    for env in _LEVEL_ENVS:
        raw = os.environ.get(env)
        if not raw:
            continue
        level = logging.getLevelName(str(raw).strip().upper())
        if isinstance(level, int):
            return level
    return logging.INFO


class _JsonLineFormatter(logging.Formatter):
    """Emit one JSON object per log record (machine-friendly sink).

    Includes any ``record.extra_*`` attribute so callers can attach
    structured fields via ``logger.info("msg", extra={"extra_pr": 25748})``.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k.startswith("extra_"):
                payload[k[len("extra_"):]] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    level: str | int | None = None,
    json_output: bool | None = None,
    log_file: str | Path | None = None,
    quiet_third_party: bool = True,
) -> logging.Logger:
    """Initialise the framework-agent root logger.

    Idempotent: each call clears previously attached handlers on the
    root so re-running (e.g. in tests, or when the CLI re-parses args)
    does not stack duplicates.

    Returns the root logger so callers may chain ``.info(...)``.
    """
    use_json = (
        json_output
        if json_output is not None
        else os.environ.get(_JSON_ENV, "").strip() in ("1", "true", "yes")
    )
    effective_level = _resolve_level(level)
    resolved_file = log_file if log_file is not None else os.environ.get(_FILE_ENV)

    root = logging.getLogger(_ROOT_NAME)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(effective_level)
    root.propagate = False

    formatter: logging.Formatter
    if use_json:
        formatter = _JsonLineFormatter()
    else:
        formatter = logging.Formatter(_DEFAULT_FMT, datefmt="%Y-%m-%d %H:%M:%S")

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(effective_level)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if resolved_file:
        file_path = Path(str(resolved_file)).expanduser()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Best-effort: read-only mount / permission denied / non-dir
            # parent should not abort logging setup. The FileHandler call
            # below will raise a more specific OSError if the parent is
            # still unwritable, so report and continue rather than swallow.
            print(
                f"[framework-agent logging_setup] WARN: mkdir({file_path.parent}) "
                f"failed: {exc!r}; FileHandler may also fail",
                file=sys.stderr,
            )
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if quiet_third_party:
        for noisy in ("urllib3", "requests", "git"):
            logging.getLogger(noisy).setLevel(max(effective_level, logging.WARNING))

    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``framework_agent`` root.

    When ``name`` is a fully-qualified module name (``framework_agent.xxx``),
    it is returned as-is; otherwise it is treated as a leaf under the root.
    """
    if not name:
        return logging.getLogger(_ROOT_NAME)
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


@contextlib.contextmanager
def stage_log(
    logger: logging.Logger,
    stage: str,
    *,
    candidate: str | None = None,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Bracket a per-stage block with start/done/failed envelopes.

    Yields a mutable dict so the caller can attach result metrics (e.g.
    ``ctx["throughput"] = 1234.5``) before the ``done`` envelope fires.

    Example::

        with stage_log(log, "build", candidate="PR:25748") as ctx:
            run_build(...)
            ctx["wall_sec"] = 412.3
    """
    started = time.monotonic()
    base: dict[str, Any] = {"stage": stage}
    if candidate:
        base["candidate"] = candidate
    base.update(fields)
    extra = {f"extra_{k}": v for k, v in base.items()}
    logger.info("stage.start %s", stage, extra=extra)
    ctx: dict[str, Any] = dict(base)
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001
        wall = time.monotonic() - started
        ctx["wall_sec"] = round(wall, 3)
        ctx["error"] = type(exc).__name__
        ctx["error_msg"] = str(exc)[:240]
        logger.exception(
            "stage.failed %s wall=%.1fs %s",
            stage, wall, type(exc).__name__,
            extra={f"extra_{k}": v for k, v in ctx.items()},
        )
        raise
    else:
        wall = time.monotonic() - started
        ctx.setdefault("wall_sec", round(wall, 3))
        logger.info(
            "stage.done %s wall=%.1fs",
            stage, wall,
            extra={f"extra_{k}": v for k, v in ctx.items()},
        )


__all__ = [
    "configure_logging",
    "get_logger",
    "stage_log",
]
