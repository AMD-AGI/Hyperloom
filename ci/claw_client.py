# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Claw API client for CI/CD orchestration."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Generator
from urllib.parse import quote

import requests
import sseclient

log = logging.getLogger(__name__)


class ClawClient:
    def __init__(self, endpoint: str, api_key: str | None = None,
                 timeout: int = 14400, agent_id: str = "agent_default",
                 sandbox_workspace: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.agent_id = agent_id
        self.sandbox_workspace = sandbox_workspace
        self.default_tools: list[int] = []
        self._last_event_id: str | None = None
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.verify = os.environ.get("SSL_CERT_FILE", os.environ.get("REQUESTS_CA_BUNDLE", True))

    @classmethod
    def from_config(cls, claw_cfg: dict) -> ClawClient:
        endpoint = claw_cfg["endpoint"]
        api_key_env = claw_cfg.get("api_key_env", "CLAW_API_KEY")
        api_key = os.environ.get(api_key_env)
        timeout = claw_cfg.get("sandbox_timeout", 14400)
        agent_id = claw_cfg.get("agent_id", "agent_default")
        sandbox_ws_env = claw_cfg.get("sandbox_workspace_env", "SANDBOX_WORKSPACE")
        sandbox_workspace = os.environ.get(sandbox_ws_env)
        client = cls(endpoint, api_key, timeout, agent_id, sandbox_workspace)
        client.default_tools = claw_cfg.get("tools", [])
        return client

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path}"

    def _check(self, resp: requests.Response) -> dict:
        resp.raise_for_status()
        return resp.json()

    # ── Session CRUD ──

    def create_session(
        self,
        name: str,
        agent_id: str | None = None,
        sandbox_image: str | None = None,
    ) -> dict:
        agent_id = agent_id or self.agent_id
        body: dict = {"name": name, "agent_id": agent_id}
        if sandbox_image:
            body["sandbox_image"] = sandbox_image
        data = self._check(self._session.post(
            self._url("/sessions"),
            json=body,
        ))
        log.info("Created session %s (name=%s)", data["data"]["session_id"], name)
        return data["data"]

    def get_session(self, session_id: str) -> dict:
        return self._check(self._session.get(
            self._url(f"/sessions/{session_id}"),
        ))["data"]

    def list_sessions(self) -> list[dict]:
        return self._check(self._session.get(
            self._url("/sessions"),
        ))["data"]

    # ── Messages ──

    def send_message(
        self,
        session_id: str,
        content: str,
        task_mode: str = "agent",
        tools: list[int] | None = None,
        plugin_id: int | None = 4,
        resource: dict | None = None,
    ) -> dict:
        body = {
            "content": content,
            "contents": [{"type": "text", "value": content}],
            "messageType": "text",
            "taskMode": task_mode,
            "attachments": [],
            "tools": tools if tools is not None else self.default_tools,
            "workspaceId": self.sandbox_workspace or os.environ.get("SANDBOX_WORKSPACE", ""),
        }
        # Omit pluginId when None so the Claw backend uses the agent_id default
        # (matches GUI behavior for remote-mode multi-node sessions).
        if plugin_id is not None:
            body["pluginId"] = plugin_id
        if resource:
            body["resource"] = resource
        resp = self._session.post(
            self._url(f"/sessions/{session_id}/messages"),
            json=body,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except requests.exceptions.JSONDecodeError:
            return {"status": "sent", "raw": resp.text[:200]}

    # ── Files ──

    def list_files(self, session_id: str) -> list[dict]:
        return self._check(self._session.get(
            self._url(f"/sessions/{session_id}/files"),
        ))["data"]

    def download_file(self, session_id: str, file_path: str) -> bytes:
        # Percent-encode the full path including any leading slash.
        encoded = quote(file_path, safe="")
        resp = self._session.get(
            self._url(f"/sessions/{session_id}/files/{encoded}/stream"),
        )
        resp.raise_for_status()
        return resp.content

    def download_file_to(self, session_id: str, file_path: str, local_path: str) -> str:
        content = self.download_file(session_id, file_path)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        log.info("Downloaded %s → %s (%d bytes)", file_path, local_path, len(content))
        return local_path

    # ── SSE streaming ──

    def subscribe_sse(
        self,
        session_id: str,
        timeout: int | None = None,
        last_event_id: str | None = None,
    ) -> Generator[dict, None, None]:
        """Subscribe to SSE event stream for a session.

        Must be called BEFORE send_message to avoid missing events.
        Yields parsed event dicts until agent stops or timeout.
        When ``last_event_id`` is provided, the server should resume from that
        point instead of replaying from the beginning (standard SSE protocol).
        """
        effective_timeout = timeout or self.timeout
        url = self._url(f"/chat/sessions/{session_id}/messages")
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        read_timeout = min(effective_timeout, 600) if effective_timeout > 0 else None
        resp = self._session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, read_timeout),
        )
        resp.raise_for_status()

        client = sseclient.SSEClient(resp)
        start = time.time()

        for event in client.events():
            if time.time() - start > effective_timeout:
                log.warning("SSE timeout after %ds for session %s", effective_timeout, session_id)
                break

            if not event.data:
                continue

            if event.data.strip() == "[DONE]":
                log.info("SSE [DONE] received for session %s", session_id)
                break

            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                log.debug("Non-JSON SSE event: %s", event.data[:100])
                continue

            if event.id:
                self._last_event_id = event.id

            yield data

    def _sse_background(self, session_id: str, on_event: Any, stop_event: threading.Event):
        """Background thread: stream SSE events for real-time logging.

        Runs until the SSE stream closes or ``stop_event`` is set.
        Does NOT reconnect — when the stream ends, the thread exits silently.
        The main polling loop handles completion detection independently.
        """
        try:
            for event_data in self.subscribe_sse(session_id, timeout=self.timeout):
                if stop_event.is_set():
                    break
                elapsed_label = ""
                try:
                    event_type = event_data.get("type", "") if isinstance(event_data, dict) else ""
                except Exception:
                    continue

                if on_event:
                    try:
                        on_event(event_data)
                    except Exception:
                        pass

                if event_type == "toolUsed":
                    tool = event_data.get("tool", "")
                    status = event_data.get("status", "")
                    brief = event_data.get("brief", "")
                    desc = event_data.get("description", "")[:200]
                    log.info("[SSE] Tool %s [%s]: %s %s", tool, status, brief, desc[:100] if status == "success" else "")
                elif event_type == "chatDelta":
                    content = event_data.get("delta", {}).get("content", "")[:150]
                    stats = event_data.get("token_stats", {})
                    if content and len(content) > 10:
                        log.info("[SSE] Agent [turn=%s]: %s", stats.get("turn", "?"), content)
                elif event_type == "statusUpdate":
                    log.info("[SSE] Status: %s - %s",
                             event_data.get("agentStatus", ""), event_data.get("brief", ""))
                elif event_type in ("sandboxStatus", "error"):
                    log.info("[SSE] %s: %s", event_type, json.dumps(event_data, default=str)[:300])
                else:
                    sub = event_data.get("subagent_id", "") if isinstance(event_data, dict) else ""
                    if sub:
                        tool = event_data.get("tool", "")
                        sdelta = event_data.get("delta", {}).get("content", "")[:100] if isinstance(event_data.get("delta"), dict) else ""
                        if tool:
                            log.info("[SSE] Sub[%s] %s: %s", sub, tool, sdelta)

        except Exception as e:
            log.info("[SSE] Stream ended: %s", e)

        log.info("[SSE] Background thread exiting")

    def monitor_session(
        self,
        session_id: str,
        timeout: int | None = None,
        heartbeat_interval: int = 300,
        on_event: Any = None,
        reconnect_retries: int = 3,
        reconnect_wait_s: int = 180,
    ) -> str:
        """Monitor a session using SSE (background, best-effort) + polling (reliable).

        SSE runs in a daemon thread for real-time tool event logging.
        Polling runs in the main thread for reliable completion detection.
        When SSE disconnects, it stops — no reconnect, no blocking.

        Returns final status: 'completed', 'failed', 'timeout'.
        """
        effective_timeout = timeout or self.timeout
        start = time.time()
        poll_interval = 30

        stop_sse = threading.Event()
        sse_thread = threading.Thread(
            target=self._sse_background,
            args=(session_id, on_event, stop_sse),
            daemon=True,
        )
        sse_thread.start()
        log.info("Session %s monitoring started (SSE background + polling every %ds)",
                 session_id, poll_interval)

        poll_count = 0
        agent_ever_ran = False
        try:
            while True:
                elapsed = time.time() - start
                if elapsed >= effective_timeout:
                    log.warning("Session %s global timeout reached (%.0fs)", session_id, elapsed)
                    return "timeout"

                try:
                    sess = self.get_session(session_id)
                    sess_status = sess.get("status", "active")
                    agent_status = sess.get("agent_status", "running")
                    poll_count += 1

                    if agent_status == "running":
                        agent_ever_ran = True

                    if poll_count <= 3 or poll_count % 10 == 0:
                        log.info("Session %s [%.0fs] poll #%d: status=%s agent=%s",
                                 session_id, elapsed, poll_count, sess_status, agent_status)

                    # Agent is done when: explicitly stopped, session completed,
                    # or agent returned to idle after having run (normal completion path)
                    if agent_status == "stopped" or sess_status in ("completed", "stopped"):
                        log.info(">>> SESSION COMPLETED <<< %s after %.0fs (status=%s, agent=%s)",
                                 session_id, elapsed, sess_status, agent_status)
                        return "completed"
                    elif agent_ever_ran and agent_status == "idle":
                        log.info(">>> SESSION COMPLETED (agent idle) <<< %s after %.0fs",
                                 session_id, elapsed)
                        return "completed"
                    elif agent_status == "failed" or sess_status == "failed":
                        log.error(">>> SESSION FAILED <<< %s after %.0fs (status=%s, agent=%s)",
                                  session_id, elapsed, sess_status, agent_status)
                        return "failed"

                except Exception as e:
                    log.warning("Session %s poll error: %s", session_id, e)

                if elapsed > 0 and int(elapsed) % heartbeat_interval < poll_interval:
                    log.info("Session %s still running... (%.0f min)", session_id, elapsed / 60)

                time.sleep(poll_interval)
        finally:
            stop_sse.set()

        return "timeout"
