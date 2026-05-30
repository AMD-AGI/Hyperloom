"""Server health monitor — checks inference server HTTP endpoints."""

from __future__ import annotations

import logging
import time

import httpx

from ..models import Alert, ServerHealthStatus, Severity

log = logging.getLogger(__name__)


class ServerHealthMonitor:
    """Poll inference server /health endpoint."""

    def __init__(self, server_url: str = "", timeout_s: float = 5.0):
        self._server_url = server_url
        self._timeout = timeout_s
        self._last_healthy: float = 0
        self._consecutive_failures: int = 0

    def set_server_url(self, url: str) -> None:
        self._server_url = url
        self._consecutive_failures = 0

    async def check(self) -> tuple[ServerHealthStatus, list[Alert]]:
        alerts: list[Alert] = []

        if not self._server_url:
            return ServerHealthStatus(url="", reachable=False, error="no server URL configured"), alerts

        status = await self._probe()

        if status.reachable:
            self._last_healthy = time.time()
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            severity = Severity.WARNING if self._consecutive_failures < 3 else Severity.CRITICAL
            alerts.append(Alert(
                check_name="server_health_fail",
                severity=severity,
                summary=f"Server health check failed ({self._consecutive_failures} consecutive): "
                        f"{status.error}",
                evidence={
                    "url": status.url,
                    "consecutive_failures": self._consecutive_failures,
                    "last_healthy_ago_s": time.time() - self._last_healthy if self._last_healthy else None,
                    "error": status.error,
                },
                timestamp=time.time(),
            ))

        if status.reachable and status.response_time_ms > 5000:
            alerts.append(Alert(
                check_name="server_slow_response",
                severity=Severity.WARNING,
                summary=f"Server responded in {status.response_time_ms:.0f}ms (>5000ms)",
                evidence={"response_time_ms": status.response_time_ms},
                timestamp=time.time(),
            ))

        return status, alerts

    async def _probe(self) -> ServerHealthStatus:
        health_url = self._server_url.rstrip("/") + "/health"
        try:
            async with httpx.AsyncClient() as client:
                start = time.time()
                resp = await client.get(health_url, timeout=self._timeout)
                elapsed_ms = (time.time() - start) * 1000
                return ServerHealthStatus(
                    url=health_url,
                    reachable=resp.status_code == 200,
                    response_time_ms=elapsed_ms,
                    status_code=resp.status_code,
                    error="" if resp.status_code == 200 else f"HTTP {resp.status_code}",
                )
        except httpx.TimeoutException:
            return ServerHealthStatus(url=health_url, reachable=False, error="timeout")
        except httpx.ConnectError as e:
            return ServerHealthStatus(url=health_url, reachable=False, error=f"connect error: {e}")
        except Exception as e:
            return ServerHealthStatus(url=health_url, reachable=False, error=str(e))
