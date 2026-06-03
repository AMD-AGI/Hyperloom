"""Core Robustness Agent — legacy event-driven daemon (pre-M1).

This loop wires monitors -> checks -> RCA -> ``IntentEmitter`` directly
against the Coordinator's SQLite database.  M1 introduces a reactor
pipeline (see :mod:`robustness_agent.role.reactor`) driven through the
subprocess CLI in :mod:`robustness_agent.runtime.cli`; this file is
kept so the ``--mode legacy`` CLI flag still works for environments
that haven't migrated yet.

Prefer ``python -m robustness_agent.runtime.cli tick`` (or its host-
side wrapper :class:`inference_optimizer.orchestrator.backends.RobustnessAgentBackend`)
for new deployments.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .checks.disk_check import DiskCheck
from .checks.event_check import EventCheck
from .checks.stall_check import StallCheck
from .conductor import ConductorReader, IntentEmitter
from .config import Config
from .models import Alert
from .monitors.gpu_monitor import GpuMonitor
from .monitors.log_tailer import LogTailer
from .monitors.process_monitor import ProcessMonitor
from .monitors.server_health import ServerHealthMonitor
from .providers import create_provider
from .rca import RcaEngine

log = logging.getLogger(__name__)


class RobustnessAgent:
    """Main agent class — orchestrates monitors, checks, and RCA."""

    def __init__(self, config: Config):
        """Initialise the legacy agent and its conductor handles.

        Args:
            config (Config): Resolved configuration providing the
                conductor DB path and check thresholds.
        """
        self.config = config
        self._provider: Any = None
        self._conductor_reader = ConductorReader(config.conductor_db_path)
        self._intent_emitter = IntentEmitter(config.conductor_db_path)
        self._rca_engine = RcaEngine(config)

        self._alert_history: list[Alert] = []
        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []

    async def start(self) -> None:
        """Initialize monitors/checks and spawn monitoring tasks."""
        log.info("Robustness Agent starting (session_dir=%s)", self.config.session_dir)

        self._provider = await create_provider(self.config)

        self._process_monitor = ProcessMonitor(self.config, self._provider)
        self._gpu_monitor = GpuMonitor(self.config, self._provider)
        self._server_health = ServerHealthMonitor()
        self._log_tailer = LogTailer()
        self._disk_check = DiskCheck(self.config, self._provider)
        self._stall_check = StallCheck(self.config, self._conductor_reader)
        self._event_check = EventCheck(self.config)

        self._conductor_reader.connect()
        self._intent_emitter.connect()

        self._running = True
        self._tasks = [
            asyncio.create_task(self._loop_process_check(), name="process_check"),
            asyncio.create_task(self._loop_gpu_check(), name="gpu_check"),
            asyncio.create_task(self._loop_health_check(), name="health_check"),
            asyncio.create_task(self._loop_log_check(), name="log_check"),
            asyncio.create_task(self._loop_disk_check(), name="disk_check"),
            asyncio.create_task(self._loop_event_poll(), name="event_poll"),
            asyncio.create_task(self._loop_stall_check(), name="stall_check"),
            asyncio.create_task(self._loop_rca(), name="rca"),
            asyncio.create_task(self._loop_heartbeat(), name="heartbeat"),
        ]
        log.info("Robustness Agent started with %d monitoring tasks", len(self._tasks))

    async def stop(self) -> None:
        """Stop all tasks and close Conductor connections."""
        log.info("Robustness Agent stopping")
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._conductor_reader.close()
        self._intent_emitter.close()
        log.info("Robustness Agent stopped")

    async def run_forever(self) -> None:
        """Start the agent and run until cancelled."""
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # -- monitoring loops --

    async def _loop_process_check(self) -> None:
        """Poll process health and emit alerts at configured cadence."""
        while self._running:
            try:
                alerts = await self._process_monitor.check()
                self._handle_alerts(alerts)
            except Exception as exc:
                log.error("Process check failed: %s", exc)
            await asyncio.sleep(self.config.process_check_interval)

    async def _loop_gpu_check(self) -> None:
        """Poll GPU health and emit alerts at configured cadence."""
        while self._running:
            try:
                alerts = await self._gpu_monitor.check()
                self._handle_alerts(alerts)
            except Exception as exc:
                log.error("GPU check failed: %s", exc)
            await asyncio.sleep(self.config.gpu_check_interval)

    async def _loop_health_check(self) -> None:
        """Poll server health endpoints and emit alerts."""
        while self._running:
            try:
                _, alerts = await self._server_health.check()
                self._handle_alerts(alerts)
            except Exception as exc:
                log.error("Health check failed: %s", exc)
            await asyncio.sleep(self.config.health_check_interval)

    async def _loop_log_check(self) -> None:
        """Tail logs for errors and emit alerts."""
        while self._running:
            try:
                alerts = await self._log_tailer.check()
                self._handle_alerts(alerts)
            except Exception as exc:
                log.error("Log check failed: %s", exc)
            await asyncio.sleep(5.0)

    async def _loop_disk_check(self) -> None:
        """Monitor disk usage and emit alerts."""
        while self._running:
            try:
                alerts = await self._disk_check.check()
                self._handle_alerts(alerts)
            except Exception as exc:
                log.error("Disk check failed: %s", exc)
            await asyncio.sleep(self.config.disk_check_interval)

    async def _loop_event_poll(self) -> None:
        """Poll Conductor events and turn them into alerts."""
        while self._running:
            try:
                events = self._conductor_reader.poll_events()
                if events:
                    alerts = self._event_check.process_events(events)
                    self._handle_alerts(alerts)
            except Exception as exc:
                log.error("Event poll failed: %s", exc)
            await asyncio.sleep(self.config.event_poll_interval)

    async def _loop_stall_check(self) -> None:
        """Detect orchestration stalls via Conductor state."""
        while self._running:
            try:
                alerts = await self._stall_check.check()
                self._handle_alerts(alerts)
            except Exception as exc:
                log.error("Stall check failed: %s", exc)
            await asyncio.sleep(60.0)

    async def _loop_rca(self) -> None:
        """Trigger RCA when thresholds are met and execute actions."""
        while self._running:
            try:
                if self._rca_engine.should_trigger():
                    log.info("RCA triggered — running diagnosis")
                    extra_context = self._gather_rca_context()
                    finding = await self._rca_engine.run_rca(extra_context)
                    if finding:
                        log.info(
                            "RCA result: root_cause=%s action=%s confidence=%.2f",
                            finding.root_cause, finding.action_type, finding.confidence,
                        )
                        self._execute_rca_action(finding)
            except Exception as exc:
                log.error("RCA loop failed: %s", exc)
            await asyncio.sleep(30.0)

    async def _loop_heartbeat(self) -> None:
        """Emit periodic heartbeat events for liveness tracking."""
        while self._running:
            self._intent_emitter._write_event("send_message", {
                "topic": "heartbeat",
                "body_md": f"ok (robustness agent, {len(self._alert_history)} alerts total)",
            })
            await asyncio.sleep(60.0)

    # -- alert handling --

    def _handle_alerts(self, alerts: list[Alert]) -> None:
        """Persist and forward alerts to Coordinator and RCA engine.

        Records each alert in the bounded history, emits it to the
        Coordinator, and feeds it to the RCA engine.

        Args:
            alerts (list[Alert]): Alerts produced during a check cycle.
        """
        for alert in alerts:
            self._alert_history.append(alert)
            log.warning("[%s] %s: %s", alert.severity.value, alert.check_name, alert.summary)
            self._intent_emitter.emit_alert(alert)
            self._rca_engine.ingest_alerts([alert])

        # keep history bounded
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

    def _gather_rca_context(self) -> dict[str, Any]:
        """Collect supplementary context for an RCA run.

        Returns:
            dict[str, Any]: Optional ``agent_last_activity`` ages and
            ``active_leases`` pulled from the conductor reader.
        """
        context: dict[str, Any] = {}
        activity = self._conductor_reader.get_agent_last_activity()
        if activity:
            context["agent_last_activity"] = {
                k: time.time() - v for k, v in activity.items()
            }
        leases = self._conductor_reader.get_active_leases()
        if leases:
            context["active_leases"] = leases
        return context

    def _execute_rca_action(self, finding: Any) -> None:
        """Emit the Coordinator intent recommended by an RCA finding.

        Dispatches on ``finding.action_type`` to emit a kill-task,
        prune-branch, escalate, or no-op intent.

        Args:
            finding (Any): The RCA finding carrying ``action_type``,
                ``action_payload``, and ``root_cause``.
        """
        if finding.action_type == "kill_task":
            task_id = finding.action_payload.get("task_id", "")
            if task_id:
                self._intent_emitter.emit_kill_task(task_id, finding.root_cause)
        elif finding.action_type == "prune_branch":
            family = finding.action_payload.get("family", "")
            if family:
                self._intent_emitter.emit_prune_branch(family, finding.root_cause)
        elif finding.action_type == "escalate_strategy_change":
            hint = finding.action_payload.get("next_action_hint", "")
            self._intent_emitter.emit_escalate(finding.root_cause, hint)
        elif finding.action_type == "no_action":
            log.info("RCA recommends no action: %s", finding.root_cause)
