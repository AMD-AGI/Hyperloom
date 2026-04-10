"""Claw API client for CI/CD orchestration."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Generator

import requests
import sseclient

log = logging.getLogger(__name__)


class ClawClient:
    def __init__(self, endpoint: str, api_key: str | None = None,
                 timeout: int = 14400, agent_id: str = "agent_default"):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.agent_id = agent_id
        self.default_tools: list[int] = []
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
        client = cls(endpoint, api_key, timeout, agent_id)
        client.default_tools = claw_cfg.get("tools", [])
        return client

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path}"

    def _check(self, resp: requests.Response) -> dict:
        resp.raise_for_status()
        return resp.json()

    # ── Session CRUD ──

    def create_session(self, name: str, agent_id: str | None = None) -> dict:
        agent_id = agent_id or self.agent_id
        data = self._check(self._session.post(
            self._url("/sessions"),
            json={"name": name, "agent_id": agent_id},
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
    ) -> dict:
        body = {
            "content": content,
            "contents": [{"type": "text", "value": content}],
            "messageType": "text",
            "taskMode": task_mode,
            "attachments": [],
            "tools": tools if tools is not None else self.default_tools,
            "workspaceId": "control-plane-sandbox",
        }
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
        resp = self._session.get(
            self._url(f"/sessions/{session_id}/files/{file_path}/stream"),
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
    ) -> Generator[dict, None, None]:
        """Subscribe to SSE event stream for a session.

        Must be called BEFORE send_message to avoid missing events.
        Yields parsed event dicts until agent stops or timeout.
        """
        effective_timeout = timeout or self.timeout
        url = self._url(f"/chat/sessions/{session_id}/messages")
        resp = self._session.get(
            url,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=(30, effective_timeout),
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

            yield data

    def monitor_session(
        self,
        session_id: str,
        timeout: int | None = None,
        heartbeat_interval: int = 300,
        on_event: Any = None,
        reconnect_retries: int = 3,
        reconnect_wait_s: int = 180,
    ) -> str:
        """Monitor a session's SSE stream until completion.

        On transient disconnects (Claw service restart, network blip), waits
        ``reconnect_wait_s`` seconds and retries up to ``reconnect_retries``
        times before declaring failure.

        Returns final status: 'completed', 'failed', 'timeout'.
        """
        effective_timeout = timeout or self.timeout
        start = time.time()
        last_heartbeat = start
        status = "running"
        got_agent_response = False
        retries_left = reconnect_retries

        while True:
            try:
                remaining = effective_timeout - (time.time() - start)
                if remaining <= 0:
                    status = "timeout"
                    log.warning("Session %s global timeout reached (%.0fs)",
                                session_id, time.time() - start)
                    break

                for event_data in self.subscribe_sse(session_id, int(remaining)):
                    elapsed = time.time() - start
                    retries_left = reconnect_retries

                    try:
                        event_type = event_data.get("type", "") if isinstance(event_data, dict) else ""
                    except Exception:
                        log.warning("Session %s [%.0fs] unparseable event: %s",
                                    session_id, elapsed, repr(event_data)[:500])
                        continue

                    if on_event:
                        try:
                            on_event(event_data)
                        except Exception as cb_err:
                            log.warning("Session %s on_event callback error: %s", session_id, cb_err)

                    try:
                        log.info("Session %s [%.0fs] SSE %s: %s",
                                 session_id, elapsed, event_type,
                                 json.dumps(event_data, default=str))
                    except Exception:
                        log.info("Session %s [%.0fs] SSE %s: (unserializable event)",
                                 session_id, elapsed, event_type)

                    if event_type == "chatDelta":
                        got_agent_response = True

                    if event_type in ("sandboxStatus", "error", "statusUpdate"):
                        try:
                            _dump = json.dumps(event_data, indent=2, default=str)
                        except Exception:
                            _dump = repr(event_data)

                        if event_type == "sandboxStatus":
                            sb_status = event_data.get("status", "")
                            if sb_status == "failed":
                                log.error(">>> SANDBOX FAILED <<< session=%s\n%s", session_id, _dump)
                            continue

                        if event_type == "error":
                            log.error(">>> ERROR EVENT <<< session=%s\n%s", session_id, _dump)
                            status = "failed"
                            break

                        if event_type == "statusUpdate":
                            agent_status = event_data.get("agentStatus", "")
                            if agent_status == "stopped":
                                brief = event_data.get("brief", "")
                                status = "failed" if "failed" in brief.lower() else "completed"
                                log.info(">>> SESSION %s <<< after %.0fs (%s)",
                                         status.upper(), elapsed, brief)
                                break

                    if time.time() - last_heartbeat > heartbeat_interval:
                        log.info("Session %s still running... (%.0f min)", session_id, elapsed / 60)
                        last_heartbeat = time.time()

                if status != "running":
                    break

                elapsed = time.time() - start
                if elapsed >= effective_timeout * 0.8:
                    status = "timeout"
                    log.warning("Session %s SSE stream ended after %.0fs with no terminal event, marking as timeout",
                                session_id, elapsed)
                    break

                log.warning("Session %s SSE stream ended after %.0fs with status still 'running', "
                            "will attempt reconnect", session_id, elapsed)

            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                elapsed = time.time() - start
                if elapsed >= effective_timeout * 0.9:
                    status = "timeout"
                    log.warning("Session %s timeout after %.0fs", session_id, elapsed)
                    break

                retries_left -= 1
                if retries_left <= 0:
                    status = "failed"
                    log.error("Session %s connection lost after %.0fs (exhausted %d reconnect attempts): %s",
                              session_id, elapsed, reconnect_retries, e)
                    break

                log.warning("Session %s SSE disconnected after %.0fs: %s — "
                            "waiting %ds then reconnecting (attempt %d/%d)",
                            session_id, elapsed, e, reconnect_wait_s,
                            reconnect_retries - retries_left + 1, reconnect_retries)
                time.sleep(reconnect_wait_s)
                continue

            except Exception as e:
                elapsed = time.time() - start
                log.error("Session %s monitoring error after %.0fs: %s: %s",
                          session_id, elapsed, type(e).__name__, e)
                if elapsed >= effective_timeout * 0.9:
                    status = "timeout"
                else:
                    retries_left -= 1
                    if retries_left <= 0:
                        status = "failed"
                        log.error("Session %s exhausted %d reconnect attempts after unrecoverable errors",
                                  session_id, reconnect_retries)
                    else:
                        log.warning("Session %s waiting %ds before reconnect (attempt %d/%d)",
                                    session_id, reconnect_wait_s,
                                    reconnect_retries - retries_left + 1, reconnect_retries)
                        time.sleep(reconnect_wait_s)
                        continue
                break

        return status
