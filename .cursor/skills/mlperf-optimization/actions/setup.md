# Action: Environment Setup

## Overview

Initializes the MLPerf optimization environment: sources config, creates symlinks,
validates data/Primus paths, verifies trial infrastructure, and checks MCP connectivity.

## Inputs
- User-specified config shell script path (default: `config_MI355X_1x8x1_fp8.sh`)
- MLPerf code directory (default: `/root/Hyperloom-plus-mlperf/training_optimization/mlperf`)

## Procedure

### Step 1: Auto-detect environment

```bash
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 8)
GPU_TYPE=$(rocm-smi --showproductname 2>/dev/null | grep -o "MI[0-9]*[A-Za-z]*" | head -1 || echo "MI355X")
```

### Step 2: Source config and set paths

```bash
MLPERF_DIR="${MLPERF_DIR:-/root/Hyperloom-plus-mlperf/training_optimization/mlperf}"
CONFIG_SH="${CONFIG_SH:-$MLPERF_DIR/config_MI355X_1x8x1_fp8.sh}"

cd "$MLPERF_DIR"
source "$CONFIG_SH"
```

### Step 3: Create necessary directories

```bash
mkdir -p "$LOGDIR" "$(dirname $MLLOG_OUTPUT_FILE)" /root/mlperf_primus/conf
cp "$MLPERF_DIR/conf/gpt_oss_20B-pretrain-fp8.yaml" /root/mlperf_primus/conf/
```

### Step 4: Setup container symlinks

```bash
bash "$MLPERF_DIR/setup_container_symlinks.sh"
```

This creates:
- `/workspace/code` → `$MLPERF_DIR`
- `/data` → `$DATADIR`
- `/model` → `$MODELDIR`
- `/results` → `$LOGDIR`

### Step 4.5: Validate trial infrastructure

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/mlperf-optimization}"

# Verify trial_monitor.py
[ -f "$SKILL_ROOT/scripts/trial_monitor.py" ] || { echo "ERROR: trial_monitor.py not found"; exit 1; }
python3 "$SKILL_ROOT/scripts/trial_monitor.py" --help >/dev/null 2>&1 || { echo "ERROR: trial_monitor.py not executable"; exit 1; }

# Verify quiet config functions
source "$SKILL_ROOT/scripts/apply_quiet_config.sh"
quiet_yaml "$EXP" && restore_yaml "$EXP" || { echo "ERROR: quiet_yaml/restore_yaml failed"; exit 1; }

# Verify common.sh loads
source "$SKILL_ROOT/scripts/common.sh" || { echo "ERROR: common.sh failed to source"; exit 1; }
```

### Step 5: Validate prerequisites

```bash
# Check data exists
[ -f /data/c4-train.en_6_text_document.bin ] || { echo "ERROR: Training data not found"; exit 1; }
[ -f /data/c4-validation-91205-samples.en_text_document.bin ] || { echo "ERROR: Validation data not found"; exit 1; }

# Check Primus
[ -d "$PRIMUS_PATH" ] || { echo "ERROR: Primus not found at $PRIMUS_PATH"; exit 1; }

# Check config
[ -f "$EXP" ] || { echo "ERROR: Config not found at $EXP"; exit 1; }

# Kill any lingering training processes
source "$SKILL_ROOT/scripts/common.sh"
kill_training
```

### Step 6: Apply runtime tunables (optional)

```bash
# Only if running on bare metal or have sudo access
if [ -w /proc/sys/vm/drop_caches ]; then
    bash "$MLPERF_DIR/runtime_tunables.sh"
fi
```

### Step 7: Create results directory

```bash
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/mlperf-optimization}"
RESULT_DIR="${RESULT_DIR:-/root/mlperf_results/${TIMESTAMP}}"
mkdir -p "$RESULT_DIR"
```

### Step 8: MCP Connectivity — Probe, Heal, Verify

Run the following Python script via Shell. It probes each required MCP server
using the URL **as configured in mcp.json** — first via Streamable HTTP (POST),
then via SSE (GET) — and reports connectivity. The only auto-heal applied is
**auth propagation** from sibling servers when a server has no Authorization
header. PS: different servers use different
transports: SSE vs Streamable HTTP, and the URL in mcp.json reflects each
server's own transport.

```python
import subprocess, json, re, sys

MCP_JSON = ".cursor/mcp.json"

REQUIRED_SERVERS = {
    "oci-traceLens-agent": "TraceLens profiling",
    "oob-optimizer-dev":   "OOB Agent (Codex/Claude kernel opt)",
    "oci-geak-agent":      "GEAK (GPU kernel opt)",
}

def _post_jsonrpc(url, method, params, auth, req_id=1):
    """POST a JSON-RPC request and return parsed result."""
    headers = ["-H", f"Authorization: {auth}"] if auth else []
    r = subprocess.run(
        ["curl", "-s", "--max-time", "10", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-H", "Accept: application/json, text/event-stream"] + headers +
        ["-d", json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})],
        capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise ConnectionError(f"curl failed (exit {r.returncode}): {r.stderr[:120]}")
    body = r.stdout.strip()
    if not body:
        raise ConnectionError("empty response")
    # Streamable HTTP may return bare JSON or SSE-wrapped JSON
    if body.startswith("{"):
        return json.loads(body)
    # Try extracting JSON from SSE data: lines
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload and payload.startswith("{"):
                return json.loads(payload)
    raise ConnectionError(f"unexpected response format: {body[:120]}")

def handshake_streamable_http(url, auth):
    """Streamable HTTP transport: POST initialize + tools/list directly to the URL."""
    init = _post_jsonrpc(url, "initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "mlperf-setup", "version": "1.0"}
    }, auth, req_id=1)
    info = init["result"]["serverInfo"]
    tools_resp = _post_jsonrpc(url, "tools/list", {}, auth, req_id=2)
    tools = [t["name"] for t in tools_resp["result"]["tools"]]
    return info, tools, "streamable_http"

def handshake_sse(url, auth):
    """SSE transport: GET to obtain message endpoint, then POST initialize + tools/list."""
    headers = ["-H", f"Authorization: {auth}"] if auth else []
    r = subprocess.run(
        ["curl", "-s", "--max-time", "4", url] + headers,
        capture_output=True, text=True, timeout=6)
    m = re.search(r"^data:\s*(.+)$", r.stdout, re.MULTILINE)
    if not m:
        raise ConnectionError(f"no SSE data line (body: {r.stdout[:80]})")
    msg_path = m.group(1).strip()
    if msg_path.startswith("http"):
        msg_url = msg_path
    else:
        base = url.rsplit("/sse", 1)[0] if "/sse" in url else url.rsplit("/", 1)[0]
        msg_url = f"{base}/{msg_path.lstrip('/')}"
    init = _post_jsonrpc(msg_url, "initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "mlperf-setup", "version": "1.0"}
    }, auth, req_id=1)
    info = init["result"]["serverInfo"]
    tools_resp = _post_jsonrpc(msg_url, "tools/list", {}, auth, req_id=2)
    tools = [t["name"] for t in tools_resp["result"]["tools"]]
    return info, tools, "sse"

def mcp_handshake(url, auth):
    """Try Streamable HTTP first, then SSE. Returns (info, tools, transport)."""
    errors = []
    for fn in (handshake_streamable_http, handshake_sse):
        try:
            return fn(url, auth)
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
    raise ConnectionError(" | ".join(errors))

# --- Load config ---
with open(MCP_JSON) as f:
    cfg = json.load(f)
servers = cfg.get("mcpServers", {})

# --- Collect auth tokens by domain for healing ---
tokens = {}
for s in servers.values():
    url, auth = s.get("url", ""), s.get("headers", {}).get("Authorization", "")
    if auth and "://" in url:
        tokens.setdefault(url.split("://")[1].split("/")[0], auth)

# --- Probe each required server; only heal missing auth, never touch URLs ---
results = {}
changed = False

for name, role in REQUIRED_SERVERS.items():
    srv = servers.get(name)
    if not srv:
        results[name] = {"status": "missing", "role": role}
        continue

    url = srv.get("url", "")
    auth = srv.get("headers", {}).get("Authorization", "")

    for attempt in range(2):
        try:
            info, tool_names, transport = mcp_handshake(url, auth)
            results[name] = {
                "status": "ok", "server": f"{info['name']} v{info['version']}",
                "tools": tool_names, "transport": transport, "role": role,
            }
            break
        except Exception as e:
            err = str(e)
            if attempt == 0 and not auth:
                # Only heal: propagate auth from a sibling on the same domain
                domain = url.split("://")[1].split("/")[0] if "://" in url else ""
                if domain in tokens:
                    srv.setdefault("headers", {})["Authorization"] = tokens[domain]
                    auth = tokens[domain]
                    changed = True
                    continue  # retry with auth
            results[name] = {"status": "down", "error": err, "role": role}
            break

# --- Write back only if auth was healed ---
if changed:
    with open(MCP_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

# --- Print summary ---
print("=== MCP Connectivity Summary ===")
for name, r in results.items():
    if r["status"] == "ok":
        tp = r.get("transport", "?")
        print(f"  {name}: OK ({tp}) — {r['server']} — {len(r['tools'])} tools: {', '.join(r['tools'])}")
    else:
        print(f"  {name}: {r['status'].upper()} ({r.get('error', 'not in mcp.json')}) [{r['role']}]")
if changed:
    print("\nmcp.json auth was patched. Cursor should auto-reload; if not: Cmd+Shift+P → Developer: Reload Window")

# --- Output as JSON for state ---
json.dump(results, open("/tmp/mcp_probe_results.json", "w"), indent=2)
```

After the script completes, set state from the results:

```python
import json
r = json.load(open("/tmp/mcp_probe_results.json"))
state["mcp_status"] = {name: v["status"] for name, v in r.items()}
state["mcp_tools"] = {name: v.get("tools", []) for name, v in r.items()}
state["tracelens_available"] = r.get("oci-traceLens-agent", {}).get("status") == "ok"
state["oob_available"]       = r.get("oob-optimizer-dev", {}).get("status") == "ok"
state["geak_available"]      = r.get("oci-geak-agent", {}).get("status") == "ok"
state["kernel_opt_available"] = state["oob_available"] or state["geak_available"]
```

## Outputs
- Environment variables set, symlinks verified, `$RESULT_DIR` created
- `trial_monitor.py`, `quiet_yaml`/`restore_yaml`, `common.sh` validated
- `state["mcp_status"]` — per-server status (`ok` / `down` / `missing`)
- `state["mcp_tools"]` — per-server tool name list (for `CallMcpTool` reference)
- `state["tracelens_available"]`, `state["oob_available"]`, `state["geak_available"]` — boolean flags
- `mcp.json` auto-patched if fixable issues detected

## Heuristic Update

N/A — setup is a prerequisite, not an optimization action.

## Failure Handling

- Data/GPU/Primus not found: check paths, ROCm installation
- MCP server `down`: the probe tried both Streamable HTTP and SSE transports on the
  configured URL. The only auto-heal is auth propagation from sibling servers on the
  same domain. Server URLs are **never modified** — each server's URL in `mcp.json`
  reflects its own transport (e.g., TraceLens uses Streamable HTTP at `.../mcp`,
  GEAK/OOB use SSE at `.../sse`). If still down after auth heal, the service is
  genuinely offline — downstream actions fall back gracefully (TraceLens → local
  parse_trace.py; GEAK/OOB → whichever is available; both down → kernel-opt skipped)
