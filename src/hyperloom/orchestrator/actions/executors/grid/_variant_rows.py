# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The records a grid variant's rounds pass between stages, and the rows a variant's ending is recorded as."""

from __future__ import annotations

import json
import logging
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.perf_metric import GRADED_INTVTY_P50_CHECKPOINT, constraint_fields

from ...stop_attribution import StoppedByTheRun
from .._round_folder import round_report_path
from ..benchmark_result import served_complete_protocol
from ._grid_base import GridVariant, VariantResult

log = logging.getLogger(__name__)


class _Halt(Enum):
    """What one variant's end means for the rest of the grid: walk on, stop unless told otherwise, or stop."""

    CONTINUE = auto()
    ON_FAILURE = auto()
    STOP = auto()


class _VariantAborted(Exception):
    """A variant ended before its result was assembled, carrying what the grid records for it."""

    def __init__(
        self, result: VariantResult, *, halt: _Halt = _Halt.ON_FAILURE, abandoned: StoppedByTheRun | None = None
    ) -> None:
        super().__init__(result.error_class)
        self.result = result
        self.halt = halt
        self.abandoned = abandoned


class _RoundProcess(NamedTuple):
    """What the benchmark subprocess of one grid round returned."""

    returncode: Any
    stdout: str
    stderr: str


class _VariantSlot(NamedTuple):
    """One variant's place in the grid, the directory its rounds write to, and the time they are granted.

    The cap belongs to the slot rather than to the grid because a sweep's rungs differ in
    the concurrency they measure, so they differ in the work one round has to get through.
    It is ``None`` until this variant's config has been composed, because until then nobody
    has seen the env the rounds will run under: a slot still holding ``None`` names a
    variant whose rounds never started, and was granted nothing.
    """

    idx: int
    variant: GridVariant
    path: Path
    benchmark_timeout_sec: float | None = None


class _VariantFailure(NamedTuple):
    """Why one variant ended, and what its ending means for the rest of the grid.

    ``error`` is the row's own message when it says more than the marker's ``recorded``
    summary; ``abandoned`` is why the variants a halted grid will not reach never started,
    and is set exactly when ``halt`` is ``STOP``.
    """

    error_class: str
    recorded: str
    error: str | None = None
    halt: _Halt = _Halt.ON_FAILURE
    abandoned: StoppedByTheRun | None = None


class _RoundAttempt(NamedTuple):
    """One variant's run of one round, and what it left behind.

    ``server_survivors`` are the servers the variant booted that outlived its teardown.
    """

    at: _VariantSlot
    returncode: int | None
    started_unix: float
    server_log: Path
    server_survivors: tuple[int, ...]


class _VariantRounds(NamedTuple):
    """The config one variant's rounds share, the server they were given, and the time they are granted.

    ``benchmark_timeout_sec`` is sized from ``cfg_path`` and not from the variant spec,
    because that config is where every layer of the launch env has already landed.
    """

    cfg_path: Path
    lifecycle: dict[str, Any]
    auto_warmup: bool
    mn_effective_args: str
    benchmark_timeout_sec: float


class _BenchmarkPass(NamedTuple):
    """One benchmark pass of one variant, and what it left behind.

    ``server_survivors`` are the servers the pass booted that outlived the teardown that
    followed it; a pass whose servers the next round still needs leaves it empty.
    """

    proc: _RoundProcess
    started_unix: float
    workspaces_before: Any
    server_survivors: tuple[int, ...] = ()


class _MeasuredVariant(NamedTuple):
    """One variant's measured round, the workspace and salvage it left on disk, and the warmup it discarded."""

    rounds: _VariantRounds
    round: _BenchmarkPass
    workspace: Path
    artifacts: list[Any]
    warmup_tput: float | None


class _VariantOutcome(NamedTuple):
    """A measured variant together with what its benchmark report turned out to say."""

    measured: _MeasuredVariant
    report: dict[str, Any] | None
    measurement: dict[str, Any]
    warnings: list[str]


def _abort_variant(at: _VariantSlot, failure: _VariantFailure, **fields: Any) -> NoReturn:
    _write_variant_abort_marker(
        at.path,
        variant_name=at.variant.name,
        error_class=failure.error_class,
        error_summary=failure.recorded,
        extra_args=at.variant.extra_server_args,
    )
    raise _VariantAborted(
        VariantResult(
            name=at.variant.name,
            extra_server_args=at.variant.extra_server_args,
            extra_envs=dict(at.variant.extra_envs),
            status="failed",
            error=failure.recorded if failure.error is None else failure.error,
            error_class=failure.error_class,
            note=at.variant.note,
            granted_cap_sec=at.benchmark_timeout_sec,
            **fields,
        ),
        halt=failure.halt,
        abandoned=failure.abandoned,
    )


def _write_variant_abort_marker(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str = "",
) -> None:
    """Write ``abort_reason.json`` into the variant slot directory."""
    try:
        slot.mkdir(parents=True, exist_ok=True)
        marker = {
            "variant": variant_name,
            "error_class": error_class,
            "error": (error_summary or "")[:2000],
            "extra_args": extra_args,
            "aborted_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        (slot / "abort_reason.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "_grid_runner: failed to write abort_reason.json at %s: %s",
            slot,
            exc,
        )


def _not_run_skip_result(variant: GridVariant, stopped: StoppedByTheRun) -> VariantResult:
    """Build the ``skipped`` result for a variant the run never got to."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        error=stopped.never_started,
        error_class=stopped.error_class,
        note=variant.note,
    )


def _kept_nonzero_exit(at: _VariantSlot, outcome: _VariantOutcome) -> str | None:
    """The exit code this measurement is kept despite, or None when the round exited clean."""
    returncode = outcome.measured.round.proc.returncode
    if returncode == 0:
        return None
    proc = outcome.measured.round.proc
    nonzero_error = redact_secret_values((proc.stderr or proc.stdout)[-2000:])
    measurement = outcome.measurement
    if not served_complete_protocol(measurement):
        log.warning(
            "grid_runner: variant %s aborted: round_nonzero_after_valid_measurement (rc=%d, served %s of %s requests)",
            at.variant.name,
            returncode,
            measurement.get("completed_requests"),
            measurement.get("requested_requests"),
        )
        workspace = outcome.measured.workspace
        report_path = round_report_path(workspace)
        _abort_variant(
            at,
            _VariantFailure("round_nonzero_after_valid_measurement", nonzero_error),
            workspace=str(workspace),
            report_path=str(report_path) if report_path.exists() else None,
            raw_result_path=measurement.get("raw_result_path"),
            reported_success=measurement.get("reported_success"),
            returncode=returncode,
            nonfatal_warnings=outcome.warnings,
            server_log_path=None,
        )
    # Every request the client recorded as requested was served, so the exit code came from
    # something the round had already finished with. The cause stays on the result: a reader
    # comparing this point to its neighbours is owed the reason it is not a clean zero.
    outcome.warnings.append(f"nonzero_rc_after_complete_protocol:{returncode}")
    log.warning(
        "grid_runner: variant %s exited %d after serving its whole protocol "
        "(%s of %s requests); keeping the measurement: %s",
        at.variant.name,
        returncode,
        measurement.get("completed_requests"),
        measurement.get("requested_requests"),
        nonzero_error,
    )
    return nonzero_error


def _measured_row(at: _VariantSlot, outcome: _VariantOutcome) -> VariantResult:
    nonzero_kept_error = _kept_nonzero_exit(at, outcome)
    measurement = outcome.measurement
    workspace = outcome.measured.workspace
    report_path = round_report_path(workspace)
    row = VariantResult(
        name=at.variant.name,
        extra_server_args=at.variant.extra_server_args,
        extra_envs=dict(at.variant.extra_envs),
        status="succeeded",
        output_throughput=measurement.get("output_throughput"),
        request_throughput=measurement.get("request_throughput"),
        total_token_throughput=measurement.get("total_token_throughput"),
        completed_requests=measurement.get("completed_requests"),
        duration_seconds=measurement.get("duration_seconds"),
        ttft_mean_ms=measurement.get("ttft_mean_ms"),
        **constraint_fields(measurement),
        tpot_mean_ms=measurement.get("tpot_mean_ms"),
        input_throughput=measurement.get("input_throughput"),
        tpot_p90_ms=measurement.get("tpot_p90_ms"),
        intvty_p90=measurement.get("e2e_norm_intvty_p90"),
        intvty_p50=measurement.get("e2e_norm_intvty_p50"),
        intvty_p50_checkpoint=measurement.get(GRADED_INTVTY_P50_CHECKPOINT),
        request_error_rate=measurement.get("request_error_rate"),
        workspace=str(workspace),
        report_path=str(report_path) if report_path.exists() else None,
        raw_result_path=measurement.get("raw_result_path"),
        reported_success=measurement.get("reported_success"),
        returncode=outcome.measured.round.proc.returncode,
        nonfatal_warnings=outcome.warnings,
        error=nonzero_kept_error,
        server_log_path=None,
        note=at.variant.note,
        runtime_sec=round(max(0.0, time.time() - outcome.measured.round.started_unix), 2),
        granted_cap_sec=at.benchmark_timeout_sec,
    )
    log.info("grid_runner: variant %s tput=%.1f tok/s", at.variant.name, row.output_throughput or 0.0)
    return row
