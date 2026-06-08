#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Kill the previous vllm/sglang server on the RayJob head pod.
#
# Submitted via Ray Dashboard REST by ``multi_node restart-server`` as
# the first phase of a server swap. Idempotent: missing PID file means
# "no previous server", which is fine.
#
# IR-5: NEVER use ``pkill -f sglang`` (would also kill Ray workers).
# We rely on the PID file written by ``launch_server.sh``.
#
# Usage: kill_server.sh PID_FILE [GRACE_SECONDS]
#   PID_FILE       — same path that launch_server.sh used in the prior cycle
#   GRACE_SECONDS  — wait this many seconds after SIGTERM before SIGKILL (default 5)
#
# Exits 0 even if there was nothing to kill; only fails if the PID file
# exists but is unreadable.

set -euo pipefail

PID_FILE="${1:-}"
GRACE="${2:-5}"

if [ -z "$PID_FILE" ]; then
    echo "kill_server.sh: usage: $0 PID_FILE [GRACE_SECONDS]" >&2
    exit 2
fi

if [ ! -e "$PID_FILE" ]; then
    echo "kill_server.sh: no PID file at $PID_FILE; nothing to kill"
    exit 0
fi

if [ ! -s "$PID_FILE" ]; then
    echo "kill_server.sh: PID file $PID_FILE is empty; removing and exiting"
    rm -f "$PID_FILE"
    exit 0
fi

OLD_PID=$(cat "$PID_FILE")
if ! [[ "$OLD_PID" =~ ^[0-9]+$ ]]; then
    echo "kill_server.sh: PID file $PID_FILE contains non-numeric '$OLD_PID'; removing" >&2
    rm -f "$PID_FILE"
    exit 0
fi

if ! kill -0 "$OLD_PID" 2>/dev/null; then
    echo "kill_server.sh: pid=$OLD_PID is not alive; removing stale PID file"
    rm -f "$PID_FILE"
    exit 0
fi

echo "kill_server.sh: SIGTERM pid=$OLD_PID (grace=${GRACE}s)"
kill "$OLD_PID" || true
sleep "$GRACE"

if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "kill_server.sh: still alive after grace; SIGKILL pid=$OLD_PID"
    kill -9 "$OLD_PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "kill_server.sh: OK"
