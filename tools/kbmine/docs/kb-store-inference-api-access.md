<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->
# Accessing Inference Data from KB Store

## Endpoint configuration

The endpoint and the token are environment settings; neither is written in
this repository. Ask the KB Store owners for the URL that applies to you.

From inside the Kubernetes cluster, the in-cluster Service address:

```bash
export KB_STORE_URL=http://<kb-service>.<kb-namespace>.svc.cluster.local
export KB_STORE_TOKEN=<read-scoped-token>
```

Outside the cluster or through VPN, the external endpoint:

```bash
export KB_STORE_URL=https://<kb-store-host>/knowledge-base
export KB_STORE_TOKEN=<read-scoped-token>
```

Verify connectivity:

```bash
curl -sS "$KB_STORE_URL/health"
```

The expected response is `{"status":"ok"}`.

## Authentication

KB Store APIs require bearer-token authentication:

```bash
-H "Authorization: Bearer $KB_STORE_TOKEN"
```

Use a read-only token authorized for the `inference` scheme. Do not retrieve or
share the server-side `KB_STORE_TOKENS` secret.

## Complete retrieval flow

There is no single endpoint that returns every Inference record. The complete
flow is:

```text
Search all identities
    -> list Sessions for each identity
    -> retrieve each Session document
    -> retrieve or download associated artifacts
```

## 1. Search Inference identities

```bash
curl -sS -X POST "$KB_STORE_URL/v1/kb/search" \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "scheme": "inference",
    "match": {},
    "offset": 0,
    "limit": 100
  }'
```

Example response:

```json
{
  "items": [
    {
      "canonical_id": "inference:model:hardware:framework:model_type:architecture:version:precision",
      "dimensions": {
        "model": "example-model",
        "hardware": "mi355x",
        "framework_name": "sglang"
      },
      "source": "native",
      "updated_at": "2026-09-03T00:00:00Z"
    }
  ],
  "total": 235,
  "next_offset": 100
}
```

The maximum page size is 100. Continue with `offset=next_offset` until
`next_offset` is `null`.

Search dimensions use exact matching. For example:

```bash
curl -sS -X POST "$KB_STORE_URL/v1/kb/search" \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "scheme": "inference",
    "match": {
      "hardware": "mi355x",
      "framework_name": "sglang"
    },
    "offset": 0,
    "limit": 100
  }'
```

## 2. List Sessions for an identity

```bash
export CANONICAL_ID='inference:model:hardware:framework:model_type:architecture:version:precision'

curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/sessions"
```

This returns all recorded Sessions and the current champion pointer.

## 3. Retrieve the recommended record

```bash
curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID"
```

This normally returns the champion. If no champion has been promoted, KB Store
returns an available Session record and includes the selection reason.

## 4. Retrieve a complete Session document

```bash
export SESSION_ID=<session-id>

curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/sessions/$SESSION_ID"
```

The response includes the canonical identity, Session and record IDs, schema
version, revision, producers, knowledge payload, and artifact manifest.

## 5. Retrieve artifacts

List files for one Session:

```bash
curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/sessions/$SESSION_ID/files"
```

List deduplicated files across every Session under an identity:

```bash
curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/files"
```

Optionally filter by artifact kind:

```bash
curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/files?kind=patch"
```

Download a complete Session archive:

```bash
curl -fSL \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/sessions/$SESSION_ID/archive" \
  -o "$SESSION_ID.tar.gz"
```

## Hyperloom Recipe View

A replay-ready Hyperloom Recipe View requires the complete replay scope:

```bash
curl -sS \
  -H "Authorization: Bearer $KB_STORE_TOKEN" \
  "$KB_STORE_URL/v1/kb/$CANONICAL_ID/views/hyperloom-recipe?kernel_optimizer=forge&tp=8&conc=64&isl=1024&osl=1024"
```

The following parameters must be supplied together:

```text
kernel_optimizer
tp
conc
isl
osl
```

An incomplete scope returns HTTP `422`.

## Python client

This tool ships a blocking, standard-library Python client:

```python
from kbmine.kb_store_client import KBStoreClient

client = KBStoreClient.from_env()
```

It reads `KB_STORE_URL` and `KB_STORE_TOKEN` from the environment.

## Port forwarding

If direct Service DNS is unavailable:

```bash
kubectl --kubeconfig <kubeconfig> \
  -n <kb-namespace> port-forward svc/<kb-service> 18080:80

export KB_STORE_URL=http://127.0.0.1:18080
```

## Related PR Monitor endpoints

PR Monitor is hosted by the same KB Service:

```text
REST: ${KB_STORE_URL}/pr-monitor/v1
MCP:  ${KB_STORE_URL}/pr-monitor/mcp/
```

Health check:

```bash
curl -sS "$KB_STORE_URL/pr-monitor/v1/healthz"
```

## API summary

```text
POST /v1/kb/search
    Search and enumerate Inference identities

GET /v1/kb/{canonical_id}
    Retrieve the recommended record

GET /v1/kb/{canonical_id}/sessions
    List every Session under an identity

GET /v1/kb/{canonical_id}/sessions/{session_id}
    Retrieve a complete Session document

GET /v1/kb/{canonical_id}/files
    List artifacts across Sessions

GET /v1/kb/{canonical_id}/sessions/{session_id}/files
    List artifacts for one Session

GET /v1/kb/{canonical_id}/sessions/{session_id}/archive
    Download the complete Session bundle

GET /v1/kb/{canonical_id}/views/hyperloom-recipe
    Retrieve a scope-specific replay-ready Recipe View
```
