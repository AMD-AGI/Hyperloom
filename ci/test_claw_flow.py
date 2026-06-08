#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Smoke test: validate the full Claw API call sequence with a trivial prompt."""

import json
import logging
import os
import sys
import threading
import time

sys.path.insert(0, ".")
from claw_client import ClawClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

ENDPOINT = os.environ.get("SAFE_BASE_URL", "") + "/claw-api/v1"
TEST_PROMPT = "Reply with exactly: HELLO_FROM_CI_TEST. Do not run any tools or commands. Just reply with that text."
TIMEOUT = 120


def main():
    api_key = os.environ.get("CLAW_API_KEY")
    claw = ClawClient(ENDPOINT, api_key=api_key)
    if not api_key:
        log.warning("CLAW_API_KEY not set. POST messages may fail with 401.")

    # Step 1: Create session
    log.info("Step 1: Creating session...")
    session = claw.create_session("ci-smoke-test", "agent_default")
    sid = session["session_id"]
    log.info("  Session created: %s", sid)

    # Step 2: Subscribe SSE in background
    log.info("Step 2: Subscribing to SSE...")
    events_received = []
    event_types_seen = set()
    final_status = {"value": None}

    def _monitor():
        try:
            for evt in claw.subscribe_sse(sid, timeout=TIMEOUT):
                etype = evt.get("type", "unknown")
                events_received.append(evt)
                event_types_seen.add(etype)

                if etype == "statusUpdate":
                    agent_status = evt.get("agentStatus", "")
                    log.info("  SSE [statusUpdate] agentStatus=%s brief=%s",
                             agent_status, evt.get("brief", ""))
                    if agent_status == "stopped":
                        final_status["value"] = "completed"
                        break
                elif etype == "chat":
                    role = evt.get("role", "")
                    content = evt.get("content", "")
                    if not isinstance(content, str):
                        content = json.dumps(content)[:100]
                    else:
                        content = content[:100]
                    log.info("  SSE [chat] role=%s content=%s", role, content)
                elif etype == "sandboxStatus":
                    log.info("  SSE [sandboxStatus] phase=%s status=%s",
                             evt.get("phase", "?"), evt.get("status", "?"))
                elif etype == "chatDelta":
                    delta = evt.get("delta", {})
                    log.info("  SSE [chatDelta] content=%s", (delta.get("content") or "")[:60])
                elif etype == "toolUsed":
                    log.info("  SSE [toolUsed] tool=%s status=%s",
                             evt.get("tool", "?"), evt.get("status", "?"))
                elif etype == "liveStatus":
                    log.info("  SSE [liveStatus] %s", evt.get("text", ""))
                elif etype == "error":
                    log.error("  SSE [error] %s", evt.get("message", ""))
                    final_status["value"] = "failed"
                    break
                elif etype == "eventsNotifyEventsAfter":
                    log.info("  SSE [history] %d events replayed", len(evt.get("events", [])))
                else:
                    log.info("  SSE [%s] %s", etype, json.dumps(evt)[:100])
        except Exception as e:
            log.error("  SSE error: %s", e)
            final_status["value"] = "error"

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    time.sleep(2)

    # Step 3: Send message
    log.info("Step 3: Sending test prompt...")
    resp = claw.send_message(sid, TEST_PROMPT)
    log.info("  Send response: %s", json.dumps(resp)[:200])

    # Step 4: Wait for completion
    log.info("Step 4: Waiting for agent to finish (max %ds)...", TIMEOUT)
    t.join(timeout=TIMEOUT)

    # Results
    log.info("=" * 50)
    log.info("RESULTS")
    log.info("=" * 50)
    log.info("Events received: %d", len(events_received))
    log.info("Event types seen: %s", sorted(event_types_seen))
    log.info("Final status: %s", final_status["value"] or "timeout")

    expected_types = {"statusUpdate", "chat"}
    missing = expected_types - event_types_seen
    if missing:
        log.warning("Missing expected event types: %s", missing)
    else:
        log.info("All expected event types received")

    if final_status["value"] == "completed":
        log.info("PASS: Full flow validated successfully")
        return 0
    else:
        log.warning("WARN: Flow ended with status=%s", final_status["value"])
        return 1


if __name__ == "__main__":
    sys.exit(main())
