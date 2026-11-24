---
myst:
    html_meta:
        "description": "Multi-node and prefill-decode cluster hand-off variables for Hyperloom: topology hand-off file, service URL, SSH keys, pod IPs, and Ray Dashboard connection details."
        "keywords": "Hyperloom, environment variables, multi-node, prefill-decode, TOPOLOGY_HANDOFF_FILE, HYPERLOOM_MN_EXT_SERVICE_URL, HYPERLOOM_MN_EXT_SSH_KEY, infera, rayjob"
---
# Multi-node / prefill-decode (PD)

Use CLI flags for multi-node topology and prefill-decode configuration:

`--nodes`, `--mn-backend`, `--gpus-per-node`, `--tp`, `--ep`,
`--pd-mode`, `--pd-prefill-nodes`, `--pd-decode-nodes`, `--pd-prefill-tp`,
`--pd-decode-tp`, `--pd-transfer-backend`, and `--pd-ib-device`.

`optimize` never creates or releases a multi-node cluster. The provisioning
platform (for example, Primus-Claw) creates the RayJob or InferaDeployment and hands it
over through the variables below; without a hand-off `--nodes >= 2` exits 2.

## Cluster hand-off variables

`TOPOLOGY_HANDOFF_FILE` is read first. The file carries `backend` (`rayjob` or
`infera`), `nodes`, `gpus_per_node`, `service_url`, and the block of that
backend: `ray` (`head_ip`, `dashboard_token_file`) or `infera` (`ssh_key_file`,
`ssh_port_base`, `known_hosts_file`, `prefill_ips`, `decode_ips`, `worker_ips`).
Credentials are file paths; the Ray Dashboard token is read from its file. A set
variable naming a file that is unreadable, not valid JSON, of an unsupported
schema major, or missing a required field is an error that names the file and
the problem; Hyperloom does not fall back to the legacy variables. If legacy
variables are also set and differ from the file, the file wins and a warning
names the differing fields. The exceptions are `--nodes`, `--gpus-per-node` and
`--mn-backend`: a launch flag always wins over the file, so a `--nodes 1` run is
single-node and a `--nodes 3` run is stored as three nodes whatever the file says.

Without `TOPOLOGY_HANDOFF_FILE`, `HYPERLOOM_MN_EXT_SERVICE_URL` is the only variable
that tells the optimizer a cluster is ready; the rest describe how to reach it.

| Variable | Backend | Required | Description |
|----------|---------|----------|-------------|
| `TOPOLOGY_HANDOFF_FILE` | both | No | Path to a TopologyHandoff v1 JSON file (`"schema": "topology-handoff/v1"`) the platform wrote. When set (non-empty) it is the hand-off and the `HYPERLOOM_MN_EXT_*` and `INFERENCE_OPTIMIZER_MN_BACKEND` variables are not read for the fields it carries. The launch flags win over it: `--mn-backend` over `backend`, `--gpus-per-node` over `gpus_per_node`, and `--nodes` over `nodes` (`optimize` exports `INFERENCE_OPTIMIZER_NODES` and `INFERENCE_OPTIMIZER_GPUS_PER_NODE` from the flags, and a value set there is never replaced by the file's); the file fills in `backend` and `gpus_per_node` when their flag is omitted. When unset, the variables below apply unchanged. |
| `HYPERLOOM_MN_EXT_SERVICE_URL` | both | **yes** | Benchmark frontend URL (`http(s)://…`; infera frontend typically `:8000`). Its presence triggers external mode. |
| `HYPERLOOM_MN_EXT_SSH_KEY` | infera | **yes** | Private SSH key already authorized on the pods (the platform installs the public half at create time). |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` / `_DECODE_IPS` | infera | PD | Prefill / decode pod IPs (comma-separated) for PD-disaggregated runs. |
| `HYPERLOOM_MN_EXT_WORKER_IPS` | infera | aggregated | Worker pod IPs (comma-separated) for aggregated (non-PD) runs. At least one of `_PREFILL_IPS` / `_DECODE_IPS` / `_WORKER_IPS` is required. |
| `HYPERLOOM_MN_EXT_SSH_PORT` | infera | No (default `2233`) | SSH base port; decode role is offset `+10`. |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | infera | No | `known_hosts` path; else a relaxed host-key check is used. |
| `HYPERLOOM_MN_EXT_HEAD_IP` | rayjob | No (recommended) | Ray head IP (Dashboard `:8265`, GCS `:6379`). Enables per-round restarts; omit for benchmark-only. |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | rayjob | No | Ray Dashboard auth token, only if the dashboard is authenticated. |

Infera external mode requires `HYPERLOOM_MN_EXT_SSH_KEY` plus at least one
`*_IPS` list, or the run fails fast at startup. RayJob external mode ignores
the SSH / IP vars and uses `HYPERLOOM_MN_EXT_HEAD_IP` for restarts.
