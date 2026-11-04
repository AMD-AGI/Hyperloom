#!/usr/bin/env bash
# Correctness check: sends a single request and verifies the server responds coherently.
# Called via: docker exec <container> bash <this_script>
# Env vars: PORT, MODEL_PATH, MODEL_NAME, CORRECTNESS_OUTPUT

set -euo pipefail

: "${PORT:?PORT env var not set}"
: "${MODEL_PATH:?MODEL_PATH env var not set}"
: "${CORRECTNESS_OUTPUT:?CORRECTNESS_OUTPUT env var not set}"

# Capture both response body and HTTP status separately so we can distinguish
# server-unreachable from server-reachable-but-wrong.
HTTP_CODE=$(curl -s -o /tmp/_correctness_body.json -w "%{http_code}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_PATH\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer with just the number.\"}],
        \"max_tokens\": 512,
        \"temperature\": 0
    }" 2>/dev/null) || HTTP_CODE="000"

RESPONSE=$(cat /tmp/_correctness_body.json 2>/dev/null || echo "")

python3 - <<PYEOF
import json, sys, os

http_code = "$HTTP_CODE"
response_raw = r"""$RESPONSE"""
output_path = "$CORRECTNESS_OUTPUT"

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

def write(result):
    with open(output_path, "w") as f:
        json.dump(result, f)

if http_code == "000" or not response_raw.strip():
    write({"passed": False, "score": 0.0,
           "error": f"server unreachable or no response (HTTP {http_code})"})
    sys.exit(0)

try:
    data = json.loads(response_raw)
except json.JSONDecodeError as e:
    write({"passed": False, "score": 0.0,
           "error": f"server returned non-JSON (HTTP {http_code}): {response_raw[:200]}"})
    sys.exit(0)

if http_code != "200":
    write({"passed": False, "score": 0.0,
           "error": f"HTTP {http_code}: {response_raw[:200]}"})
    sys.exit(0)

choices = data.get("choices", [])
if not choices:
    write({"passed": False, "score": 0.0, "error": "response had no choices"})
    sys.exit(0)

msg = choices[0].get("message") or {}
content = msg.get("content", "").strip()
reasoning = msg.get("reasoning_content", "").strip()

# Accept reasoning models: content OR reasoning_content must be non-empty.
# GLM-5.2 is a thinking model — reasoning_content carries the response when
# max_tokens is hit before the final answer token.
if not content and not reasoning:
    write({"passed": False, "score": 0.0, "error": "response had no content or reasoning"})
    sys.exit(0)

response = content or reasoning[:200]
print(f"content: {response[:100]}", file=sys.stderr)
write({"passed": True, "score": 1.0, "response": response})
PYEOF
