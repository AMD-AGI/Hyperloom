# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run one arm's CONC ladder: hold one persistent server across it, or restart the server per rung."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    STAGE_BOOT,
    STAGE_BOOT_ATTEMPT,
    STAGE_BUDGET_SKIP,
    STAGE_REUSE,
    STAGE_SERVER_RESTART,
    STRATEGY_SERVER_RESTART,
    STRATEGY_SINGLE_SERVER,
    ArmStrategy,
)
from ._conc_sweep_run import (
    _Arm,
    _budget_skip_result,
    _CANCEL_STOP,
    _deadline_skip_result,
    _RungNote,
    _RungServer,
    _SweepRun,
)
from ..actions.executors._grid_runner import GridVariant, VariantResult, variant_conc
from ..actions.executors._subprocess_kill import session_deadline_to_remaining_sec
from ..actions.executors.benchmark_backend import resolve_backend
from ..actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, StoppedByTheRun


log = logging.getLogger(__name__)


class _BootOutcome(Enum):
    """How an arm's attempt to hold one server across its whole ladder ended."""

    SERVED = "served"
    STOPPED = "stopped"
    ALL_FAILED = "all_failed"


def _failed_rung_result(variant: GridVariant, error: str, error_class: str) -> VariantResult:
    """The result a rung is filed under when running it raised instead of measuring."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="failed",
        error=error,
        error_class=error_class,
    )


@dataclass(frozen=True)
class _ArmServer:
    """The server one arm's ladder runs against, and the Ray lease covering its whole lifetime.

    One held Ray lease (``num_gpus=TP``) spans the arm's persistent server -- boot plus every CONC reuse round, or
    the Option B per-variant restarts -- so no GPU process outlives it.
    """

    arm_name: str
    pid_dir: Path
    port: int
    framework: str
    lifecycle_eligible: bool
    lifecycle_reason: str
    lease: Any

    @classmethod
    def resolve(cls, run: _SweepRun, arm_name: str, lease: Any) -> _ArmServer:
        """Claim the arm's shared pid dir and read its lifecycle parameters from the materialized config."""
        from ..actions.executors._server_lifecycle import resolve_lifecycle_params

        pid_dir = run.workspace / f"server_{arm_name}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        try:
            lc_params = resolve_lifecycle_params(run.base_yaml_path)
            return cls(
                arm_name=arm_name,
                pid_dir=pid_dir,
                port=int(lc_params.get("port") or 8888),
                framework=str(lc_params.get("framework") or ""),
                lifecycle_eligible=bool(lc_params.get("eligible")),
                lifecycle_reason=str(lc_params.get("reason") or ""),
                lease=lease,
            )
        except Exception as exc:
            log.warning("conc_sweep single-server: resolve_lifecycle_params failed", exc_info=True)
            if run.recorder is not None:
                # Named per arm: both arms resolve, and two faults reading alike
                # would collapse into one row.
                run.recorder.record_fault(stage=f"resolve_lifecycle_params:{arm_name}", exc=exc)
            return cls(
                arm_name=arm_name,
                pid_dir=pid_dir,
                port=8888,
                framework="",
                lifecycle_eligible=False,
                lifecycle_reason="resolve_failed",
                lease=lease,
            )

    def strategy(self) -> ArmStrategy:
        """How this arm's ladder is run, and why."""
        return ArmStrategy(
            strategy=STRATEGY_SINGLE_SERVER if self.lifecycle_eligible else STRATEGY_SERVER_RESTART,
            reason=None if self.lifecycle_eligible else "framework_not_lifecycle_eligible",
            lifecycle_eligible=self.lifecycle_eligible,
            lifecycle_reason=self.lifecycle_reason,
            port=self.port,
            framework=self.framework,
            serving_lease_held=self.lease is not None,
        )

    def lifecycle(self, *, cleanup: bool) -> dict[str, Any]:
        """The ``server_lifecycle`` block a rung launches with against this server."""
        return {"cleanup": cleanup, "pid_dir": str(self.pid_dir), "port": self.port}

    def teardown(self) -> None:
        """Stop this arm's server; idempotent, and a no-op once it is already down."""
        from ..actions.executors._server_lifecycle import teardown_lifecycle_server

        teardown_lifecycle_server(pid_dir=self.pid_dir, framework=self.framework, port=self.port)

    def release(self) -> None:
        """Give back the Ray lease that covered this arm's server."""
        if self.lease is not None:
            self.lease.close()


class _BootAttempt(NamedTuple):
    """One attempt to bring the arm's persistent server up at one CONC."""

    variant: GridVariant
    result: VariantResult | None
    started_iso: str
    elapsed_sec: float

    @property
    def failed(self) -> bool:
        """Whether the server did not come up at this CONC."""
        return self.result is None or self.result.status in {"failed", "skipped"}

    @property
    def boot_only(self) -> bool:
        """Whether this rung only booted the server and never measured its own CONC."""
        return str(getattr(self.result, "note", "") or "") == "server_lifecycle_boot_only"


class _ArmBoot(NamedTuple):
    """Where an arm's ladder resumes once the boot-retry-descend has settled."""

    outcome: _BootOutcome
    index: int
    attempt: _BootAttempt | None
    failed_boots: list[VariantResult]


@dataclass
class _BootLadder:
    """Descend an arm's CONC ladder until its persistent server comes up.

    A concurrency the server would not come up at is the finding, and whether it counts toward the curve is not
    known until a lower rung either boots or does not -- so each failed boot is carried here, recorded but
    uncommitted, until one of the three outcomes settles it.
    """

    run: _SweepRun
    server: _ArmServer
    grid: list[GridVariant]
    failed_boots: list[VariantResult] = field(default_factory=list)

    async def descend(self) -> _ArmBoot:
        """Try each CONC from highest to lowest until the server boots."""
        for index, variant in enumerate(self.grid):
            log.info(
                "conc_sweep single-server: arm=%s boot attempt %d/%d (conc=%s)",
                self.server.arm_name,
                index + 1,
                len(self.grid),
                variant.extra_envs.get("CONC", "?"),
            )
            attempt = await self._attempt(variant)
            stop = self._stop_named_by(attempt)
            if stop is not None:
                self._close_ladder(index, attempt, stop)
                return _ArmBoot(_BootOutcome.STOPPED, index, attempt, self.failed_boots)
            if attempt.failed:
                self._defer_failed_boot(attempt)
                continue
            self._commit_boot(index, attempt)
            return _ArmBoot(_BootOutcome.SERVED, index, attempt, self.failed_boots)
        return _ArmBoot(_BootOutcome.ALL_FAILED, len(self.grid), None, self.failed_boots)

    async def _attempt(self, variant: GridVariant) -> _BootAttempt:
        """Run one boot-only rung at this variant's CONC."""
        started_iso = now_iso("seconds")
        started_at = time.time()
        try:
            results = await self.run.run_rung(
                variant,
                _RungServer(
                    serving_lease=self.server.lease,
                    base_extra_envs=dict(resolve_backend().boot_phase_envs),
                    server_lifecycle=self.server.lifecycle(cleanup=False),
                    server_already_ready=False,
                    warmup_before_measure=False,
                    lifecycle_boot_only=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "conc_sweep single-server: arm=%s boot conc=%s raised %r; trying next lower conc",
                self.server.arm_name,
                variant.extra_envs.get("CONC", "?"),
                exc,
            )
            results = [_failed_rung_result(variant, f"single_server_boot: {exc}", type(exc).__name__)]
        return _BootAttempt(
            variant=variant,
            result=results[0] if results else None,
            started_iso=started_iso,
            elapsed_sec=round(time.time() - started_at, 3),
        )

    def _stop_named_by(self, attempt: _BootAttempt) -> StoppedByTheRun | None:
        """The run-side stop this boot names, if any."""
        stop = self.run.stop_named_by(attempt.result)
        if stop is None and attempt.failed and self.run.cancelled:
            return _CANCEL_STOP
        return stop

    def _close_ladder(self, index: int, attempt: _BootAttempt, stop: StoppedByTheRun) -> None:
        """File every rung from this one down under the stop that ended the arm, and close its server."""
        if stop is _CANCEL_STOP:
            log.info(
                "conc_sweep single-server: arm=%s stopped at boot conc=%s (%s); not trying lower concs",
                self.server.arm_name,
                attempt.variant.extra_envs.get("CONC", "?"),
                self.run.cancel_reason or ORCHESTRATOR_CANCELLED_CLASS,
            )
        self.run.commit_failed_boots(self.server.arm_name, self.failed_boots)
        stopped_results = [
            *([attempt.result] if attempt.result is not None else []),
            *[_deadline_skip_result(v, stop) for v in self.grid[index + 1 :]],
        ]
        for stopped in stopped_results:
            self.run.results.append(stopped)
            self.run.record_rung(
                self.server.arm_name,
                stopped,
                _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=self.run.budget.remaining_sec),
            )
        try:
            self.server.teardown()
        finally:
            self.server.release()
            self.run.close_arm(self.server.arm_name, None)

    def _defer_failed_boot(self, attempt: _BootAttempt) -> None:
        """Hold a failed boot out of the curve, tearing the server down before the next lower CONC is tried."""
        try:
            self.server.teardown()
        except Exception:  # noqa: BLE001
            pass
        failed = attempt.result or _failed_rung_result(
            attempt.variant, "single_server_boot_failed", "single_server_boot_failed"
        )
        self.failed_boots.append(failed)
        self.run.record_rung(
            self.server.arm_name,
            failed,
            _RungNote(
                stage=STAGE_BOOT_ATTEMPT,
                committed=False,
                start_time=attempt.started_iso,
                wall_duration_sec=attempt.elapsed_sec,
            ),
        )

    def _commit_boot(self, index: int, attempt: _BootAttempt) -> None:
        """Publish the boot that came up, and the higher-CONC failures it settles as genuine capacity failures."""
        result = attempt.result
        assert result is not None
        self.run.commit_failed_boots(self.server.arm_name, self.failed_boots)
        if not attempt.boot_only:
            self.run.results.append(result)
        self.run.record_rung(
            self.server.arm_name,
            result,
            _RungNote(stage=STAGE_BOOT, start_time=attempt.started_iso, wall_duration_sec=attempt.elapsed_sec),
        )
        if self.run.recorder is not None:
            self.run.recorder.record_arm_boot(
                self.server.arm_name,
                succeeded=True,
                booted_conc=variant_conc(attempt.variant),
                attempted_concs=[variant_conc(variant) for variant in self.grid[: index + 1]],
                failed_concs=[variant_conc(fb) for fb in self.failed_boots],
            )
        # Incremental flush after boot point.
        self.run.flush_progress()


def _record_arm_grid(run: _SweepRun, arm_name: str, grid: list[GridVariant]) -> None:
    """Announce the ladder this arm is about to run."""
    if run.recorder is None:
        return
    run.recorder.record_arm_grid(
        arm_name,
        rungs=[
            {
                "name": variant.name,
                "conc": variant.extra_envs.get("CONC"),
                "num_prompts": variant.extra_envs.get("NUM_PROMPTS"),
            }
            for variant in grid
        ],
    )


def _record_all_boots_failed(run: _SweepRun, server: _ArmServer, grid: list[GridVariant], boot: _ArmBoot) -> None:
    """Report that no CONC booted the persistent server and that the ladder is being retried per rung."""
    log.warning(
        "conc_sweep single-server: arm=%s all boot attempts failed; "
        "falling back to Option B (per-variant restart) for the full ladder",
        server.arm_name,
    )
    if run.recorder is None:
        return
    run.recorder.record_arm_boot(
        server.arm_name,
        succeeded=False,
        attempted_concs=[variant_conc(variant) for variant in grid],
        failed_concs=[variant_conc(fb) for fb in boot.failed_boots],
    )
    run.recorder.record_arm_strategy(
        server.arm_name,
        replace(server.strategy(), strategy=STRATEGY_SERVER_RESTART, reason="all_boot_attempts_failed"),
    )


async def _sweep_ladder_by_restart(run: _SweepRun, server: _ArmServer, grid: list[GridVariant]) -> None:
    """Run the whole ladder with a server restart per rung."""
    arm_failure: BaseException | None = None
    try:
        await _sweep_arm_option_b(run, server.arm_name, grid, serving_lease=server.lease)
    except BaseException as exc:
        arm_failure = exc
        raise
    finally:
        server.release()
        run.close_arm(server.arm_name, arm_failure)


def _reuse_grid(grid: list[GridVariant], boot: _ArmBoot) -> list[GridVariant]:
    """The rungs left to run on the booted server; a boot-only rung never measured its own CONC, so it runs again."""
    booted = boot.attempt is not None and boot.attempt.boot_only
    return grid[boot.index :] if booted else grid[boot.index + 1 :]


def _skip_rungs_left(run: _SweepRun, arm_name: str, remaining: list[GridVariant]) -> bool:
    """File every rung left under whatever stopped the ladder; false while the ladder may continue."""
    if run.observe_cancel():
        for variant in remaining:
            skipped = _deadline_skip_result(variant, _CANCEL_STOP)
            run.results.append(skipped)
            run.record_rung(arm_name, skipped, _RungNote(stage=STAGE_BUDGET_SKIP))
        return True
    if run.session_closing():
        run.budget.spend_on_session_reserve()
        for variant in remaining:
            skipped = _budget_skip_result(variant)
            run.results.append(skipped)
            run.record_rung(arm_name, skipped, _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=0.0))
        return True
    return False


async def _reuse_rung_results(
    run: _SweepRun,
    server: _ArmServer,
    variant: GridVariant,
    server_lifecycle: dict[str, Any],
) -> list[VariantResult]:
    """Run one rung against the server already up."""
    try:
        return await run.run_rung(
            variant,
            _RungServer(
                serving_lease=server.lease,
                server_lifecycle=server_lifecycle,
                server_already_ready=True,
                warmup_before_measure=False,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "conc_sweep single-server: arm=%s reuse conc=%s raised %r",
            server.arm_name,
            variant.extra_envs.get("CONC", "?"),
            exc,
        )
        return [_failed_rung_result(variant, f"single_server_reuse: {exc}", type(exc).__name__)]


async def _run_reuse_rungs(run: _SweepRun, server: _ArmServer, reuse_grid: list[GridVariant]) -> None:
    """Sweep the remaining CONCs on the booted server, checking the session deadline before each."""
    for index, variant in enumerate(reuse_grid):
        remaining_sec = session_deadline_to_remaining_sec(run.session_deadline_sec)
        if _skip_rungs_left(run, server.arm_name, reuse_grid[index:]):
            break
        started_iso = now_iso("seconds")
        started_at = time.time()
        results = await _reuse_rung_results(
            run, server, variant, server.lifecycle(cleanup=index == len(reuse_grid) - 1)
        )
        elapsed = round(time.time() - started_at, 3)
        for result in results:
            run.results.append(result)
            stopped = run.stop_named_by(result) is not None
            run.record_rung(
                server.arm_name,
                result,
                _RungNote(
                    stage=STAGE_BUDGET_SKIP if stopped else STAGE_REUSE,
                    start_time=started_iso,
                    wall_duration_sec=elapsed,
                    budget_remaining_sec=run.budget.remaining_sec if stopped else remaining_sec,
                ),
            )
        # Incremental flush after each reuse point.
        run.flush_progress()


def _teardown_after_reuse(run: _SweepRun, server: _ArmServer) -> None:
    """Safety teardown -- idempotent, no-op if already torn down."""
    try:
        server.teardown()
    except Exception as exc:
        log.warning("conc_sweep single-server: arm=%s teardown failed", server.arm_name, exc_info=True)
        if run.recorder is not None:
            run.recorder.record_fault(stage=f"teardown_lifecycle_server:{server.arm_name}", exc=exc)


async def _sweep_ladder_by_reuse(
    run: _SweepRun,
    server: _ArmServer,
    grid: list[GridVariant],
    boot: _ArmBoot,
) -> None:
    """Sweep the remaining CONCs against the server the boot left up, then close the arm on it."""
    reuse_failure: BaseException | None = None
    try:
        await _run_reuse_rungs(run, server, _reuse_grid(grid, boot))
    except BaseException as exc:
        reuse_failure = exc
        raise
    finally:
        _teardown_after_reuse(run, server)
        server.release()
        run.close_arm(server.arm_name, reuse_failure)


async def _sweep_one_arm_single_server(run: _SweepRun, arm: _Arm) -> None:
    """Sweep one arm across all CONC values reusing a single persistent server."""
    from ..actions.executors._grid_runner import _num_gpus_for_config
    from ..actions.executors._ray_serving import maybe_serving_lease

    grid = run.arm_grid(arm)
    if not grid:
        return
    _record_arm_grid(run, arm.name, grid)
    server = _ArmServer.resolve(run, arm.name, maybe_serving_lease(num_gpus=_num_gpus_for_config(run.base_yaml_path)))
    if run.recorder is not None:
        run.recorder.record_arm_strategy(arm.name, server.strategy())
    if not server.lifecycle_eligible:
        log.info(
            "conc_sweep single-server: arm=%s not lifecycle-eligible (%s); using per-variant server restart (Option B)",
            arm.name,
            server.lifecycle_reason,
        )
        await _sweep_ladder_by_restart(run, server, grid)
        return
    boot = await _BootLadder(run=run, server=server, grid=grid).descend()
    if boot.outcome is _BootOutcome.STOPPED:
        return
    if boot.outcome is _BootOutcome.ALL_FAILED:
        _record_all_boots_failed(run, server, grid, boot)
        await _sweep_ladder_by_restart(run, server, grid)
        return
    await _sweep_ladder_by_reuse(run, server, grid, boot)


async def _sweep_arm_option_b(
    run: _SweepRun,
    arm_name: str,
    grid: list[GridVariant],
    *,
    serving_lease: Any = None,
) -> None:
    """Option B fallback: run each variant with its own server (legacy behaviour)."""
    for variant in grid:
        _ob_rem = session_deadline_to_remaining_sec(run.session_deadline_sec)
        if run.observe_cancel():
            skip_r = _deadline_skip_result(variant, _CANCEL_STOP)
            run.results.append(skip_r)
            run.record_rung(arm_name, skip_r, _RungNote(stage=STAGE_BUDGET_SKIP))
            continue
        if run.session_closing():
            run.budget.spend_on_session_reserve()
            skip_r = _budget_skip_result(variant)
            run.results.append(skip_r)
            run.record_rung(arm_name, skip_r, _RungNote(stage=STAGE_BUDGET_SKIP, budget_remaining_sec=0.0))
            continue
        rung_started_iso = now_iso("seconds")
        rung_started_at = time.time()
        try:
            sub = await run.run_rung(variant, _RungServer(serving_lease=serving_lease))
        except Exception as exc:  # noqa: BLE001
            log.warning("conc_sweep option B: arm=%s variant=%s raised %r", arm_name, variant.name, exc)
            sub = [_failed_rung_result(variant, f"option_b: {exc}", type(exc).__name__)]
        rung_elapsed = round(time.time() - rung_started_at, 3)
        for r in sub:
            run.results.append(r)
            stopped = run.stop_named_by(r) is not None
            run.record_rung(
                arm_name,
                r,
                _RungNote(
                    stage=STAGE_BUDGET_SKIP if stopped else STAGE_SERVER_RESTART,
                    start_time=rung_started_iso,
                    wall_duration_sec=rung_elapsed,
                    budget_remaining_sec=run.budget.remaining_sec if stopped else _ob_rem,
                ),
            )
        run.flush_progress()
