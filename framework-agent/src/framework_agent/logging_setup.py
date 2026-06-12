# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Structured logging setup for framework-agent.

Public API:

* :func:`configure_logging`  — explicit init (called by ``fa`` CLI).
* :func:`get_logger`         — module-level helper that lazy-inits.
* :func:`stage_log`          — context manager emitting start/done/failed
                               envelopes around a per-stage block.

No import-time side effects. Single root logger ``framework_agent`` with
module-prefixed children. Text format by default; JSON Lines via
``FRAMEWORK_AGENT_LOG_JSON=1``. Level resolution: explicit arg >
``FRAMEWORK_EXPLORER_LOG_LEVEL`` > ``FRAMEWORK_AGENT_LOG_LEVEL`` > ``INFO``.
Optional file sink via ``--log-file`` / env. Re-entrant (replaces handlers).
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


# All module loggers descend from this root so one configure call propagates.
_ROOT_NAME = "framework_agent"
_DEFAULT_FMT = (
    "%(asctime)s %(levelname)-5s %(name)s :: %(message)s"
)
_LEVEL_ENVS: tuple[str, ...] = (
    "FRAMEWORK_EXPLORER_LOG_LEVEL",
    "FRAMEWORK_AGENT_LOG_LEVEL",
)
_JSON_ENV = "FRAMEWORK_AGENT_LOG_JSON"
_FILE_ENV = "FRAMEWORK_AGENT_LOG_FILE"


def _resolve_level(explicit: str | int | None) -> int:
    """Pick the effective log level (explicit > env > INFO).

    Args:
        explicit (str | int | None): An explicit level as an int, a numeric
            string, or a level name (e.g. ``"DEBUG"``). ``None`` defers to the
            environment variables in :data:`_LEVEL_ENVS`.

    Returns:
        int: The resolved :mod:`logging` level constant, defaulting to
            ``logging.INFO``.
    """
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
        """Serialise a log record to a single JSON line.

        Promotes any ``extra_*`` record attribute to a top-level field (with
        the ``extra_`` prefix stripped) and includes a formatted traceback when
        exception info is present.

        Args:
            record (logging.LogRecord): The record to format.

        Returns:
            str: A JSON object encoded as a single line.
        """
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
    """Initialise the framework-agent root logger and return it.

    Idempotent: each call clears previously attached handlers so re-running
    does not stack duplicates.
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
            # A read-only mount / bad parent shouldn't abort logging setup;
            # report and continue (FileHandler below raises if still unwritable).
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

    Args:
        name (str | None): Logger name. ``None`` or empty returns the root
            logger; a name already under the root is used verbatim; any other
            name becomes a leaf under ``framework_agent``.

    Returns:
        logging.Logger: The resolved child (or root) logger.
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
