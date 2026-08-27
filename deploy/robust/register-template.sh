#!/usr/bin/env bash
# Register deploy/robust/kernelforge.yaml on Crusoe primus-robust workload-manager.
#
# Usage:
#   export ROBUST_API_KEY=ak-...
#   export ROBUST_API_BASE=https://crusoe.primus-safe.amd.com/robust-api   # optional
#   export ROBUST_INSECURE=0   # set when a CA bundle is available (default 1 for Crusoe ingress)
#   ./deploy/robust/register-template.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YAML="${ROOT}/deploy/robust/kernelforge.yaml"
API_BASE="${ROBUST_API_BASE:-https://crusoe.primus-safe.amd.com/robust-api}"
API_PREFIX="${ROBUST_API_PREFIX:-/v1}"
KEY="${ROBUST_API_KEY:?ROBUST_API_KEY is required}"

TLS=()
if [ "${ROBUST_INSECURE:-1}" = "1" ]; then
  TLS=(-k)
fi

PAYLOAD="$(python3 - "$YAML" <<'PY'
import json, sys
try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML required: pip install pyyaml\n")
    sys.exit(1)
doc = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
payload = {
    "infra_type": doc.get("infraType", "kubernetes"),
    "kind": doc["kind"],
    "validation": doc["validation"],
    "status_mapping": doc["status"],
    "body": doc["body"],
}
json.dump(payload, sys.stdout)
PY
)"

RESP="$(printf '%s' "$PAYLOAD" | curl -sS "${TLS[@]}" -w $'\n%{http_code}' -X PUT \
  "${API_BASE%/}${API_PREFIX}/orchestration/templates/kubernetes/kernelforge" \
  -H "Authorization: Bearer ${KEY}" \
  -H "Content-Type: application/json" \
  -d @-)"
CODE="$(printf '%s' "$RESP" | tail -n1)"
BODY="$(printf '%s' "$RESP" | sed '$d')"
printf '%s' "$BODY" | python3 -m json.tool 2>/dev/null || printf '%s\n' "$BODY"

if [ "$CODE" -lt 200 ] || [ "$CODE" -ge 300 ]; then
  echo "register failed (HTTP ${CODE}) for ${YAML}" >&2
  exit 1
fi
echo "registered kubernetes/kernelforge from ${YAML} (HTTP ${CODE})"
