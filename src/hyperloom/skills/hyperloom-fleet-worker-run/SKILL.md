---
name: hyperloom-fleet-worker-run
description: Runs one Hyperloom Fleet demo job after the Slack KB service is already healthy. Installs the matching branches, injects Worker-only Fleet settings, validates connectivity, launches a recommended Qwen3-8B example workload, monitors it, and verifies Experience reads and writes. Use when the Slack bot is asked to start Hyperloom on a Worker it already knows how to reach.
---

# Run Hyperloom with Fleet KB

This is the single entry point for starting a Hyperloom demo Run. Assume:

- the Slack central service and Fleet KB are already running;
- the Slack bot already knows how to execute commands on the target Worker;
- Slack Catalog/verification tools are already connected;
- credentials come from the central secret store.

Do not redeploy the service and do not ask the user to call APIs or export
variables manually.

## Required Slack state

Resolve these values before touching the Worker:

- a fresh `scope_id`, normally the Slack job ID;
- Slack `thread_id`;
- Fleet service URL and Fleet ID;
- Worker token reference;
- Worker repository path;
- model path and framework, using the recommendation below when unset.

The Slack toolbox decides whether the scope has zero or more
`verified_for_scope` Experiences before this Skill runs. Use the exact scope it
hands off. Never substitute a different scope and never send the Bot token to
the Worker.

## Recommended workload template

These values are a starting template, not Fleet KB protocol:

```text
model      = Qwen/Qwen3-8B local checkpoint
framework  = vllm
image      = docker.io/vllm/vllm-openai-rocm:v0.29.0
gpu        = MI300X
precision  = bf16
tp         = 1
conc       = 64
isl        = 1024
osl        = 1024
max_hours  = 3
```

Use the environment's existing model path when valid. These values are only a
recommended example. If the user supplies model or workload overrides, use
those values consistently for this Run.

## Prepare the Worker checkout

Run inside the same host/container that will execute Hyperloom:

```bash
: "${HYPERLOOM_REPO_ROOT:?Hyperloom checkout path is required}"
cd "$HYPERLOOM_REPO_ROOT"
test -z "$(git status --porcelain --untracked-files=no)" || {
  echo "Hyperloom checkout is dirty; refusing demo launch" >&2
  exit 2
}
git fetch origin demo/fleet-kb-integration
git switch demo/fleet-kb-integration
git pull --ff-only origin demo/fleet-kb-integration

python3 -m pip install -e .
python3 -m pip install --upgrade \
  "git+https://github.com/zili-amd/Hyperloom-KB.git@demo/fleet-kb-service"
```

Start from a clean checkout or fresh container. Do not inherit framework
mutations from an unrelated completed Run.

## Inject the Worker environment

Set these in the actual Hyperloom process environment:

```bash
: "${SLACK_JOB_ID:?Slack job/scope ID is required}"
: "${SLACK_THREAD_ID:?Slack thread ID is required}"
: "${FLEET_KB_URL:?Fleet KB URL is required}"
: "${FLEET_KB_WORKER_TOKEN:?Worker token is required}"
: "${FLEET_KB_ID:?Fleet ID is required}"
: "${MODEL_PATH:?Model path is required}"
export FRAMEWORK="${FRAMEWORK:-vllm}"

export REPO_ROOT="$HYPERLOOM_REPO_ROOT"
export HYPERLOOM_KB_ENABLE=true
export HYPERLOOM_KB_DECL="$REPO_ROOT/examples/hyperloom-kb-inference.yaml"

export HYPERLOOM_FLEET_KB_URL="$FLEET_KB_URL"
export HYPERLOOM_FLEET_KB_WORKER_TOKEN="$FLEET_KB_WORKER_TOKEN"
export HYPERLOOM_FLEET_KB_ID="$FLEET_KB_ID"
export HYPERLOOM_FLEET_KB_WORKER_ID="$(hostname)"
export HYPERLOOM_FLEET_KB_SCOPE_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_JOB_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_THREAD_ID="$SLACK_THREAD_ID"
export HYPERLOOM_FLEET_KB_SPOOL="${USER_DATA_PATH:?}/fleet-kb-spool/$SLACK_JOB_ID"
```

Validate:

```bash
test "$HYPERLOOM_FLEET_KB_SCOPE_ID" = "$SLACK_JOB_ID"
test -f "$HYPERLOOM_KB_DECL"
```

Only `HYPERLOOM_FLEET_KB_WORKER_TOKEN` belongs on the Worker.

## Install and validate runtime

Run the repository installer in this same process/container:

```bash
: "${USER_DATA_PATH:?USER_DATA_PATH is required}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
ulimit -Sn 65536 || true

INSTALL_SH="$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
fi
bash "$INSTALL_SH"
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
```

Then run:

```bash
python3 -m hyperloom.inference_optimizer.tools.cold_start_check \
  --model "$MODEL_PATH" \
  --framework "$FRAMEWORK" \
  --require-experience-kb \
  --output "$HYPERLOOM_FLEET_KB_SPOOL/cold-start.json"
```

Require `cold_start_ready=true`. On failure, report the failing check in Slack
and stop. A healthy central service alone is not sufficient.

## Launch

Use this concrete template:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-vllm}" \
  --tp "${TP:-1}" \
  --conc "${CONC:-64}" \
  --isl "${ISL:-1024}" \
  --osl "${OSL:-1024}" \
  --precision "${PRECISION:-bf16}" \
  --target-gain "${TARGET_GAIN:-30}" \
  --max-hours "${MAX_HOURS:-3}" \
  --max-minutes-framework-pct 0.50 \
  --max-minutes-sweep-pct 0.01 \
  --no-kernel \
  --no-enable-conc-sweep \
  --no-enable-roofline \
  --degraded-kb \
  --degraded-pr
```

`--degraded-kb` disables Recipe KB only; it does not disable Fleet Experience
read/write. `--degraded-pr` keeps this example isolated from changing PR
Monitor inputs.

Use the Slack runtime's supported background-process mechanism. Do not launch a
second fresh process after an unexpected stop; resume the same session with
`optimize --resume-from`.

## Monitor and report

Immediately report to the Slack thread:

- scope ID;
- branch commits and framework image/version;
- model/workload parameters;
- cold-start result;
- optimizer PID, log, and session paths.

Poll every 300 seconds and report phase, baseline throughput, current best,
validated gain, latest KEEP/REVERT, and stop reason without exposing secrets.

## Completion acceptance

At CLOSE:

1. Require the optimizer to finish or report a terminal failure.
2. Query Fleet events for this exact scope/thread.
3. Count however many `kb.experience.cataloged` events the measured attempts
   produced; there is no fixed Experience count.
4. Require every new record to be `unverified` and unavailable outside an
   explicitly verified Run scope.
5. If the scope has verified Experiences, require `kb.read.completed` with
   non-empty `rendered_refs` and verify those refs reach the resulting measured
   Experience. If the scope is empty by explicit user choice, require an empty
   read instead.
6. Report baseline, final `current_best`, elapsed time, and Experience IDs.

Every successful Run proves unverified writes. A Run with scope verifications
additionally proves that human-approved evidence reached the Hyperloom decision
prompt and write trace.
