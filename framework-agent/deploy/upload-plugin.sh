#!/usr/bin/env bash
# Two-stage upload: prompt tool first, then plugin (idempotent upsert).
#
# Mirrors the Hyperloom plugin (id=4) shape: same image + same wekafs +
# a single type=prompt tool. See deploy/README.md for the full flow.
#
# Stages:
#   1. POST <api>/v1/tools/prompt  with claw-tool-prompt.json  -> tool_id
#   2. POST <api>/v1/plugins/upsert with claw-plugin.json (tools[0].id = tool_id)
#
# Required env:
#   CLAW_API_BASE   - Full API root including any ingress path prefix, e.g.
#                       https://core42.example-internal-host.invalid/claw-api
#                       (external higress; /claw-api prefix is part of ingress)
#                     or in-cluster (port 80, no prefix):
#                       http://primus-claw-api.primus-claw.svc.cluster.local
#                     or port-forwarded:
#                       http://127.0.0.1:19080
#                     The script appends "/v1/tools/prompt" and
#                     "/v1/plugins/upsert" directly.
#   CLAW_API_TOKEN  - Bearer token (ak-...) for an authenticated Claw user
#
# Optional env:
#   DRY_RUN         - if non-empty, print the curl commands without executing
#
# Exit codes:
#   0  - both upserts succeeded
#   2  - missing env / file / malformed JSON
#   3  - non-2xx response from either endpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_JSON="${TOOL_JSON:-$SCRIPT_DIR/claw-tool-prompt.json}"
PLUGIN_JSON="${PLUGIN_JSON:-$SCRIPT_DIR/claw-plugin.json}"

if [[ -z "${CLAW_API_BASE:-}" ]]; then
    echo "ERROR: CLAW_API_BASE not set" >&2
    exit 2
fi
if [[ -z "${CLAW_API_TOKEN:-}" ]]; then
    echo "ERROR: CLAW_API_TOKEN not set" >&2
    exit 2
fi
for f in "$TOOL_JSON" "$PLUGIN_JSON"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: JSON not found at $f" >&2
        exit 2
    fi
done

TOOL_URL="${CLAW_API_BASE%/}/v1/tools/prompt"
PLUGIN_URL="${CLAW_API_BASE%/}/v1/plugins/upsert"

echo "=== framework-agent plugin upload ==="
echo "  CLAW_API_BASE = $CLAW_API_BASE"
echo "  TOOL_URL      = $TOOL_URL"
echo "  PLUGIN_URL    = $PLUGIN_URL"
echo

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN=1 - showing payloads, not invoking curl"
    echo
    echo "---tool payload---"
    cat "$TOOL_JSON"
    echo
    echo "---plugin payload (id is placeholder; will be patched after stage 1)---"
    cat "$PLUGIN_JSON"
    exit 0
fi

# Stage 1: create / update the prompt tool ------------------------------
echo "--- stage 1: POST $TOOL_URL ---"
tool_resp="$(mktemp)"
trap 'rm -f "$tool_resp" "$plugin_resp" "$plugin_body" 2>/dev/null || true' EXIT
plugin_resp="$(mktemp)"
plugin_body="$(mktemp)"

http_code=$(curl -sS -o "$tool_resp" -w "%{http_code}" \
    -X POST "$TOOL_URL" \
    -H "Authorization: Bearer ${CLAW_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data-binary "@$TOOL_JSON")

echo "  HTTP $http_code"
if [[ ! "$http_code" =~ ^2 ]]; then
    echo "FAIL stage 1: non-2xx response" >&2
    cat "$tool_resp" >&2
    exit 3
fi
TOOL_ID=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print((d.get('data') or {}).get('id', ''))" "$tool_resp")
if [[ -z "$TOOL_ID" || "$TOOL_ID" == "None" ]]; then
    echo "FAIL stage 1: could not parse tool id from response" >&2
    cat "$tool_resp" >&2
    exit 3
fi
echo "  tool_id = $TOOL_ID"
echo

# Stage 2: patch plugin body with real tool_id, then upsert ------------
python3 - "$PLUGIN_JSON" "$TOOL_ID" "$plugin_body" <<'PY'
import json, sys
src, tool_id, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = json.load(open(src, encoding="utf-8"))
tools = body.get("tools") or []
if not tools:
    raise SystemExit("plugin.tools[] empty")
tools[0]["id"] = tool_id
body["tools"] = tools
with open(dst, "w", encoding="utf-8") as f:
    json.dump(body, f, indent=2)
print(f"patched tools[0].id = {tool_id}")
PY

echo "--- stage 2: POST $PLUGIN_URL ---"
http_code=$(curl -sS -o "$plugin_resp" -w "%{http_code}" \
    -X POST "$PLUGIN_URL" \
    -H "Authorization: Bearer ${CLAW_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data-binary "@$plugin_body")

echo "  HTTP $http_code"
if [[ ! "$http_code" =~ ^2 ]]; then
    echo "FAIL stage 2: non-2xx response" >&2
    cat "$plugin_resp" >&2
    exit 3
fi

PLUGIN_ID=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print((d.get('data') or {}).get('id', ''))" "$plugin_resp")
PLUGIN_ACTION=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print((d.get('data') or {}).get('upsert_action', 'unknown'))" "$plugin_resp")

echo
echo "OK: plugin_id=$PLUGIN_ID  action=$PLUGIN_ACTION  tool_id=$TOOL_ID"
