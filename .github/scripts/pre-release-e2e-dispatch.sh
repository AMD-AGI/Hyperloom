#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Pre-release E2E: create the SaFE PyTorchJob workloads that run the packaged wheel
# through the real user path (Claude CLI + setup skill + demo skill). Unlike the PR
# smoke test (.github/scripts/ci-e2e-dispatch.sh), which uses the orchestration
# endpoint (POST /api/v1/orchestration/workloads, kind=hyperloom) to dispatch a git
# SHA, this dispatches GENERIC pods (POST /api/v1/workloads) whose entrypoint is the
# bootstrap script. kind=PyTorchJob, NOT Authoring: the Authoring mutating webhook
# rewrites EntryPoints to `sleep infinity`, so an Authoring pod would never run our
# bootstrap; PyTorchJob honors the submitted entrypoint. See
# hyperloom-pre-release-e2e-ci-design.md §7.
#
# It creates 5 workloads for the 8 legs:
#   * 4x non-privileged 1-GPU PyTorchJob  (one per baremetal leg)
#   * 1x privileged   8-GPU PyTorchJob    (docker host; 4 nested containers, GPU 0-3)
# and writes a dispatch map (leg -> workloadId) to $DISPATCH_MAP for the poll step.
#
# Requires: bash, curl, jq on the (self-hosted, in-network) runner.
#
# Inputs (env):
#   SAFE_API_BASE     SaFE API base url                         (required)
#   SAFE_API_KEY      bearer token; privileged pod needs an
#                     ADMIN token (privileged=true is admin-only) (required)
#   SAFE_WORKSPACE_ID workspace that mounts the shared NFS       (required)
#   CI_VERSION        wheel/run version, e.g. 1.0.0b3.dev...+ci  (required)
#   AUTHORING_IMAGE   Authoring base image ref                   (required)
#   NFS_ROOT          pre-release test root on shared NFS
#                     (default /shared_nfs/hyperloom-pre-release-e2e-test)
#   MODEL_3H          local path to the 3h model (Qwen3-8B)      (required)
#   MODEL_12H         local path to the 12h model (Qwen3-14B-FP8)(required)
#   TARGET_GAIN       release gate gain %% for every leg         (default 100)
#   CLAUDE_MODEL      model for the Agent turns                  (required)
#   CLAUDE_CLI_VERSION pinned Claude CLI version                 (required)
#   ANTHROPIC_API_KEY Claude CLI auth; injected here as base64
#                     into the workload env (never written to NFS)(required)
#   ANTHROPIC_BASE_URL optional proxy / base url                 (optional)
#   TASKS             comma-separated leg subset (default: all 8)
#   DISPATCH_MAP      output file: JSON {leg: workloadId}
#                     (default $RUNNER_TEMP/pre_release_dispatch.json)
#   HOST_CPU / HOST_MEM / HOST_SHM / HOST_EPHEMERAL  privileged host resource request
#                     (default 128 / 2048Gi / 256Gi / 1792Gi -- ref 8-GPU Authoring pod)
#   LEG_CPU  / LEG_MEM              baremetal leg resource request
#                     (default 32 / 128Gi)
#   DEADLINE_3H_S / DEADLINE_12H_S pod hard-timeout per duration
#                     (default 14400 = 3+1h / 46800 = 12+1h). The docker host
#                     pod uses the MAX over its legs. SaFE kills the pod at the
#                     deadline; poll then judges that leg FAIL. Timing starts when
#                     the workload is DISPATCHED, not when it is queued.
#   DEADLINE_FIELD    SaFE payload field for the deadline (default `timeout`, the
#                     authoritative WorkloadSpec.Timeout field, integer seconds,
#                     top-level in the create-workload body; set "" to omit).
#   SAFE_CACERT / SAFE_INSECURE    TLS to the API (CA bundle / skip-verify)
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/shared_nfs/hyperloom-pre-release-e2e-test}"
TARGET_GAIN="${TARGET_GAIN:-100}"
# Sized to a proven Running 8-GPU Authoring pod (ref: sglang-kimik3-2): CPU 128,
# mem 2048Gi, ephemeral 1792Gi. The privileged DinD host runs nested docker with the
# vfs storage driver (no layer dedup), so 8 ROCm images x full copies blow past a small
# ephemeral limit -- our first run was EVICTED at ephemeralStorage 200Gi. 1792Gi matches
# the reference host and leaves headroom for vfs image blowup + model/wheel scratch.
HOST_CPU="${HOST_CPU:-128}"; HOST_MEM="${HOST_MEM:-2048Gi}"; HOST_SHM="${HOST_SHM:-256Gi}"
HOST_EPHEMERAL="${HOST_EPHEMERAL:-1792Gi}"
LEG_CPU="${LEG_CPU:-32}";    LEG_MEM="${LEG_MEM:-128Gi}"
DISPATCH_MAP="${DISPATCH_MAP:-${RUNNER_TEMP:-/tmp}/pre_release_dispatch.json}"

# Pod hard-timeout (design: 3h leg -> 3+1h, 12h leg -> 12+1h). SaFE terminates the
# workload at the deadline; the poll then sees a non-Succeeded terminal / missing
# report and judges that leg FAIL. The deadline is counted from DISPATCH (not queue)
# time, so the +1h buffer absorbs bootstrap/setup/agent overhead. Given per duration:
DEADLINE_3H_S="${DEADLINE_3H_S:-14400}"    # 3h + 1h buffer  = 4h
DEADLINE_12H_S="${DEADLINE_12H_S:-46800}"  # 12h + 1h buffer = 13h
# The SaFE API field that carries the pod deadline. Confirmed against the Primus-SaFE
# codebase: the create-workload body embeds WorkloadSpec inline, whose `timeout`
# (integer seconds, top-level, from dispatch time) is enforced by WorkloadTTLController
# for ALL workload kinds incl. Authoring. Set DEADLINE_FIELD="" to omit (then the pod
# survival cap falls back to the workspace's per-scope maxRuntime, or the poll-side
# GLOBAL_TIMEOUT_S if none). Ref: apis/pkg/apis/amd/v1/workload_types.go WorkloadSpec.Timeout.
DEADLINE_FIELD="${DEADLINE_FIELD:-timeout}"
leg_deadline_s() { case "$1" in *-3h) echo "$DEADLINE_3H_S" ;; *-12h) echo "$DEADLINE_12H_S" ;; esac; }

# SaFE caps the derived k8s object name at 44 chars (see create_workload), so we can
# NOT embed the full CI_VERSION (e.g. 1.0.0.dev202608270954+ci) in every workload name
# -- it would blow the limit and, once truncated, collide across legs (the leg suffix
# gets cut). Instead build "e2e-<leg>-<short version hash>": the human-readable leg
# stays intact up front, and a 6-hex digest of CI_VERSION disambiguates across runs
# without length risk. All legs of one run share the same VERSION_TAG.
VERSION_TAG="$(printf '%s' "$CI_VERSION" | sha1sum | cut -c1-6)"
workload_name() { printf 'e2e-%s-%s' "$1" "$VERSION_TAG"; }  # $1 = leg (or "docker-host")

: "${SAFE_API_BASE:?SAFE_API_BASE is required}"
: "${SAFE_API_KEY:?SAFE_API_KEY is required}"
: "${SAFE_WORKSPACE_ID:?SAFE_WORKSPACE_ID is required}"
: "${CI_VERSION:?CI_VERSION is required}"
: "${AUTHORING_IMAGE:?AUTHORING_IMAGE is required}"
: "${MODEL_3H:?MODEL_3H is required}"
: "${MODEL_12H:?MODEL_12H is required}"
: "${CLAUDE_MODEL:?CLAUDE_MODEL is required}"
: "${CLAUDE_CLI_VERSION:?CLAUDE_CLI_VERSION is required}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"

API="${SAFE_API_BASE%/}/api/v1/workloads"
auth=(-H "Authorization: Bearer ${SAFE_API_KEY}")
tls=()
if [ -n "${SAFE_CACERT:-}" ]; then
  tls=(--cacert "$SAFE_CACERT")
elif [ "${SAFE_INSECURE:-0}" = "1" ]; then
  tls=(-k)
fi

summary() { echo "$*" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"; }

# All 8 legs. Fields: mode backend hours model_path -- gpu index within the docker host
ALL_LEGS="baremetal-vllm-3h baremetal-vllm-12h baremetal-sglang-3h baremetal-sglang-12h \
docker-vllm-3h docker-vllm-12h docker-sglang-3h docker-sglang-12h"
REQ_TASKS="${TASKS:-$ALL_LEGS}"
REQ_TASKS="${REQ_TASKS//,/ }"

leg_model_path() { case "$1" in *-3h) echo "$MODEL_3H" ;; *-12h) echo "$MODEL_12H" ;; esac; }
leg_hours()      { case "$1" in *-3h) echo "3"       ;; *-12h) echo "12"       ;; esac; }
leg_backend()    { case "$1" in *-vllm-*) echo "vllm" ;; *-sglang-*) echo "sglang" ;; esac; }

# GPU index a docker leg binds inside the privileged host (design §3).
docker_gpu_index() {
  case "$1" in
    docker-vllm-3h)    echo 0 ;;
    docker-vllm-12h)   echo 1 ;;
    docker-sglang-3h)  echo 2 ;;
    docker-sglang-12h) echo 3 ;;
    *) echo "" ;;
  esac
}

# Common env for every workload. The API key is passed base64 so it is not visible in
# plaintext in the API payload log; bootstrap decodes it and writes it only to the
# pod-local .env (never to NFS). See design §9 (point D).
common_env_json() {
  local model_path="$1" hours="$2" backend="$3"
  jq -n \
    --arg civ "$CI_VERSION" --arg nfs "$NFS_ROOT" \
    --arg model "$model_path" --arg hours "$hours" --arg backend "$backend" \
    --arg tgain "$TARGET_GAIN" \
    --arg cmodel "$CLAUDE_MODEL" --arg cver "$CLAUDE_CLI_VERSION" \
    --arg keyb64 "$(printf '%s' "$ANTHROPIC_API_KEY" | base64 | tr -d '\n')" \
    --arg baseurl "${ANTHROPIC_BASE_URL:-}" \
    --arg cheaders "${ANTHROPIC_CUSTOM_HEADERS:-}" \
    '{
      CI_VERSION: $civ,
      NFS_ROOT: $nfs,
      HYPERLOOM_MODEL_PATH: $model,
      DEMO_HOURS: $hours,
      HYPERLOOM_BACKEND: $backend,
      TARGET_GAIN: $tgain,
      CLAUDE_MODEL: $cmodel,
      CLAUDE_CLI_VERSION: $cver,
      ANTHROPIC_API_KEY_B64: $keyb64
    }
    + (if $baseurl  == "" then {} else {ANTHROPIC_BASE_URL: $baseurl} end)
    + (if $cheaders == "" then {} else {ANTHROPIC_CUSTOM_HEADERS: $cheaders} end)'
}

# POST one workload; echo the workloadId.
# Args: displayName resourcesJson envJson privileged(true|false) entry_b64 deadline_s
create_workload() {
  local name="$1" resources="$2" env="$3" privileged="$4" entry_b64="$5" deadline_s="${6:-}"
  local body resp code json wid dl_json="{}"
  # SaFE derives the k8s object name from displayName and enforces (via the
  # vworkload admission webhook, STRICTER than plain RFC 1123): 1-44 chars, lower
  # case alphanumerics or '-', MUST start with an ALPHABETIC char and end with an
  # alphanumeric. So a leading digit or a '.'/'+' (both in CI_VERSION, e.g.
  # 1.0.0.dev...+ci) is illegal, and the full "e2e-<CI_VERSION>-<leg>" easily
  # exceeds 44. Fold every illegal char to '-', lowercase, collapse/trim dashes,
  # then cap at 44 chars re-trimming any trailing dash the cut may leave.
  name="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/-+/-/g; s/^[^a-z]+//; s/-+$//')"
  name="${name:0:44}"; name="${name%%-}"
  [ -n "$name" ] || name="e2e"
  # Attach the pod hard-deadline when both a field name and a value are set.
  if [ -n "$DEADLINE_FIELD" ] && [ -n "$deadline_s" ]; then
    dl_json="$(jq -n --arg k "$DEADLINE_FIELD" --argjson v "$deadline_s" '{($k): $v}')"
  fi
  # kind PyTorchJob, NOT Authoring: SaFE's mutating webhook (mutateAuthoring)
  # unconditionally overwrites an Authoring workload's EntryPoints to `sleep infinity`,
  # so our bootstrap entrypoint would never auto-run -- every leg would idle until
  # something exec'd in. PyTorchJob is not in that mutate switch, so it HONORS the
  # submitted entryPoints (run via launcher.sh) and bootstrap runs as the pod command.
  # version stays "v1"; NO `group` field (webhook clears group, workload_webhook.go:260).
  # privileged / 8-GPU / useWorkspaceStorage / timeout are all kind-agnostic (driven by
  # request fields, not kind) -- confirmed against Primus-SaFE source. Pod name is
  # <workloadId>-master-0, main container `pytorch`.
  body="$(jq -n \
    --arg name "$name" --arg ws "$SAFE_WORKSPACE_ID" --arg img "$AUTHORING_IMAGE" \
    --arg entry "$entry_b64" --argjson res "$resources" --argjson env "$env" \
    --argjson priv "$privileged" --argjson dl "$dl_json" \
    '{
      displayName: $name,
      workspaceId: $ws,
      groupVersionKind: {kind:"PyTorchJob", version:"v1"},
      resources: [$res],
      images: [$img],
      entryPoints: [$entry],
      env: $env,
      useWorkspaceStorage: true
    } + (if $priv then {privileged:true} else {} end) + $dl')"
  resp="$(curl -sS "${tls[@]}" -w $'\n%{http_code}' -X POST "$API" \
    "${auth[@]}" -H "Content-Type: application/json" -d "$body")"
  code="$(printf '%s' "$resp" | tail -n1)"
  json="$(printf '%s' "$resp" | sed '$d')"
  if [ "$code" -lt 200 ] || [ "$code" -ge 300 ]; then
    # Report to stderr: this runs inside wid="$(create_workload ...)" command
    # substitution, so a stdout message would be captured into $wid and never
    # reach the CI log. stderr surfaces the real HTTP status + API body.
    echo "❌ create '$name' failed (HTTP $code): $(printf '%s' "$json" | head -c 400)" >&2
    return 1
  fi
  wid="$(printf '%s' "$json" | jq -r '.workloadId // empty')"
  if [ -z "$wid" ]; then
    echo "❌ create '$name' returned no workloadId: $(printf '%s' "$json" | head -c 400)" >&2
    return 1
  fi
  printf '%s' "$wid"
}

# Base64 the bootstrap entrypoint (SaFE requires base64-encoded entryPoints).
# The bootstrap script is staged to NFS by the build job (from .github/pre-release/)
# and read by the pod at ${NFS_ROOT}/bootstrap/${CI_VERSION}/bootstrap-pre-release.sh.
bootstrap_entry_b64() {
  local extra="$1"  # extra shell prepended (e.g. E2E_DOCKER_HOST=1)
  local cmd
  cmd="set -e; ${extra} exec bash \"\${NFS_ROOT}/bootstrap/${CI_VERSION}/bootstrap-pre-release.sh\""
  printf '%s' "$cmd" | base64 | tr -d '\n'
}

echo "[dispatch] CI_VERSION=$CI_VERSION tasks='$REQ_TASKS'"
declare -A DISPATCH   # leg -> workloadId

# ---- baremetal legs: one non-privileged 1-GPU workload each ----------------
leg_resources_1gpu="$(jq -n --arg cpu "$LEG_CPU" --arg mem "$LEG_MEM" \
  '{replica:1, gpu:"1", cpu:$cpu, memory:$mem, ephemeralStorage:"100Gi"}')"

want_docker_host=0
for leg in $REQ_TASKS; do
  case "$leg" in
    baremetal-*)
      env_json="$(common_env_json "$(leg_model_path "$leg")" "$(leg_hours "$leg")" "$(leg_backend "$leg")" \
        | jq --arg leg "$leg" '. + {LEG_ID:$leg, HYPERLOOM_RUN_MODE:"baremetal"}')"
      entry="$(bootstrap_entry_b64 "")"
      dl="$(leg_deadline_s "$leg")"
      wid="$(create_workload "$(workload_name "$leg")" "$leg_resources_1gpu" "$env_json" false "$entry" "$dl")"
      DISPATCH["$leg"]="$wid"
      summary "• \`$leg\` → workloadId \`$wid\` (baremetal, 1 GPU, deadline $((dl/3600))h)"
      ;;
    docker-*)
      want_docker_host=1
      ;;
    *)
      summary "⚠️  unknown leg '$leg' ignored"
      ;;
  esac
done

# ---- docker legs: one privileged 8-GPU host running all requested docker legs ----
if [ "$want_docker_host" = 1 ]; then
  host_resources="$(jq -n --arg cpu "$HOST_CPU" --arg mem "$HOST_MEM" --arg shm "$HOST_SHM" \
    --arg eph "$HOST_EPHEMERAL" \
    '{replica:1, gpu:"8", cpu:$cpu, memory:$mem, sharedMemory:$shm, ephemeralStorage:$eph}')"
  # The host env carries the per-leg GPU map so the host bootstrap launches the right
  # nested containers via docker-run-hyperloom.sh <gpu_index> <leg>.
  docker_legs=""; gpu_map="{}"
  for leg in $REQ_TASKS; do
    case "$leg" in
      docker-*)
        idx="$(docker_gpu_index "$leg")"
        docker_legs="${docker_legs}${docker_legs:+ }${leg}"
        gpu_map="$(printf '%s' "$gpu_map" | jq --arg l "$leg" --arg i "$idx" '. + {($l): $i}')"
        ;;
    esac
  done
  # Host env: model paths for both durations, plus the leg->gpu map. Per-leg model/
  # backend are resolved inside the host bootstrap from the leg id.
  host_env="$(jq -n \
    --arg civ "$CI_VERSION" --arg nfs "$NFS_ROOT" \
    --arg m3 "$MODEL_3H" --arg m12 "$MODEL_12H" \
    --arg tgain "$TARGET_GAIN" --arg cmodel "$CLAUDE_MODEL" --arg cver "$CLAUDE_CLI_VERSION" \
    --arg keyb64 "$(printf '%s' "$ANTHROPIC_API_KEY" | base64 | tr -d '\n')" \
    --arg baseurl "${ANTHROPIC_BASE_URL:-}" \
    --arg cheaders "${ANTHROPIC_CUSTOM_HEADERS:-}" \
    --arg legs "$docker_legs" --argjson gpumap "$gpu_map" \
    '{
      CI_VERSION:$civ, NFS_ROOT:$nfs,
      MODEL_3H:$m3, MODEL_12H:$m12,
      TARGET_GAIN:$tgain, CLAUDE_MODEL:$cmodel, CLAUDE_CLI_VERSION:$cver,
      ANTHROPIC_API_KEY_B64:$keyb64,
      HYPERLOOM_RUN_MODE:"docker",
      E2E_DOCKER_HOST:"1",
      DOCKER_LEGS:$legs, DOCKER_GPU_MAP:($gpumap|tostring)
    }
    + (if $baseurl  == "" then {} else {ANTHROPIC_BASE_URL:$baseurl} end)
    + (if $cheaders == "" then {} else {ANTHROPIC_CUSTOM_HEADERS:$cheaders} end)')"
  # E2E_DOCKER_HOST=1 travels in the workload `env` (above), NOT as a command prefix:
  # under PyTorchJob the entrypoint actually runs, and a `VAR=1;` prefix followed by a
  # separate `exec bash` would NOT export VAR into the bootstrap's environment.
  entry="$(bootstrap_entry_b64 "")"
  # The one host pod runs a mix of 3h and 12h nested legs, so its deadline must be
  # the MAX over the legs it hosts (a 3h deadline would kill a still-running 12h leg).
  host_dl="$DEADLINE_3H_S"
  for leg in $docker_legs; do
    case "$leg" in *-12h) host_dl="$DEADLINE_12H_S" ;; esac
  done
  wid="$(create_workload "$(workload_name "docker-host")" "$host_resources" "$host_env" true "$entry" "$host_dl")"
  # Every docker leg shares the one host workloadId; the poll distinguishes them by
  # reading each leg's own session dir on NFS.
  for leg in $docker_legs; do
    DISPATCH["$leg"]="$wid"
    summary "• \`$leg\` → workloadId \`$wid\` (docker host, GPU $(docker_gpu_index "$leg"), deadline $((host_dl/3600))h)"
  done
fi

# ---- write the dispatch map for the poll step ------------------------------
map_json="{}"
for leg in "${!DISPATCH[@]}"; do
  map_json="$(printf '%s' "$map_json" | jq --arg l "$leg" --arg w "${DISPATCH[$leg]}" '. + {($l):$w}')"
done
printf '%s\n' "$map_json" > "$DISPATCH_MAP"
echo "dispatch_map=$DISPATCH_MAP" >> "${GITHUB_OUTPUT:-/dev/null}"
summary ""
summary "**dispatched $(printf '%s' "$map_json" | jq 'length') legs** → \`$DISPATCH_MAP\`"
echo "[dispatch] wrote $DISPATCH_MAP"
printf '%s\n' "$map_json" | jq .
