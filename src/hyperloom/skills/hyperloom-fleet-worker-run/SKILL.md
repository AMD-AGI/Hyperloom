---
name: hyperloom-fleet-worker-run
description: Applies the Fleet KB overlay to a Hyperloom run chosen and configured by an existing Hyperloom workload Skill. Use when Slack launches Hyperloom with a Fleet scope and needs KB SDK installation, Worker credentials, scope-safe reads, cold-start validation, and read/write acceptance checks.
---

# Fleet KB Worker overlay

This is not a standalone Hyperloom run Skill. First select and follow the
appropriate Hyperloom workload Skill, such as
`hyperloom-qwen3-8b-3h`. That Skill remains the only source for model,
framework, workload, image, runtime installation, launch, budget, and
monitoring behavior.

Apply only the Fleet-specific additions below.

## Match the demo branches

The Worker checkout must use Hyperloom branch
`demo/fleet-kb-integration`:

```bash
cd "$HYPERLOOM_REPO_ROOT"
git fetch origin demo/fleet-kb-integration
git switch demo/fleet-kb-integration
git pull --ff-only origin demo/fleet-kb-integration

python3 -m pip install --upgrade \
  "git+https://github.com/zili-amd/Hyperloom-KB.git@demo/fleet-kb-service"
```

Do not modify or clean an unrelated dirty checkout automatically.

## Inject the Fleet Worker environment

The Slack toolbox owns the scope and secret references. Set these values in
the actual process/container that launches Hyperloom:

```bash
: "${SLACK_JOB_ID:?Slack job/scope ID is required}"
: "${SLACK_THREAD_ID:?Slack thread ID is required}"
: "${FLEET_KB_URL:?Fleet KB URL is required}"
: "${FLEET_KB_WORKER_TOKEN:?Worker token is required}"
: "${FLEET_KB_ID:?Fleet ID is required}"

export HYPERLOOM_KB_ENABLE=true
export HYPERLOOM_KB_DECL="$HYPERLOOM_REPO_ROOT/examples/hyperloom-kb-inference.yaml"
export HYPERLOOM_FLEET_KB_URL="$FLEET_KB_URL"
export HYPERLOOM_FLEET_KB_WORKER_TOKEN="$FLEET_KB_WORKER_TOKEN"
export HYPERLOOM_FLEET_KB_ID="$FLEET_KB_ID"
export HYPERLOOM_FLEET_KB_WORKER_ID="$(hostname)"
export HYPERLOOM_FLEET_KB_SCOPE_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_JOB_ID="$SLACK_JOB_ID"
export HYPERLOOM_FLEET_KB_THREAD_ID="$SLACK_THREAD_ID"
export HYPERLOOM_FLEET_KB_SPOOL="${USER_DATA_PATH:?}/fleet-kb-spool/$SLACK_JOB_ID"
```

Only the Worker token belongs on the Worker. Never inject the Bot token.

Do not derive `GPU_TYPE` from a Skill example. Preserve a platform-provided
`TARGET_GPU_TYPE` when present and let Hyperloom's hardware probe resolve the
real board. Hyperloom may set `GPU_TYPE=mi300x` as a Magpie runner label for
MI325X while persisting `mi325x` as the Experience identity.

## Add the Experience-KB cold-start gate

After the selected workload Skill completes its normal runtime install and
resolves `MODEL_PATH` and `FRAMEWORK`, run:

```bash
python3 -m hyperloom.inference_optimizer.tools.cold_start_check \
  --model "$MODEL_PATH" \
  --framework "$FRAMEWORK" \
  --require-experience-kb \
  --output "$HYPERLOOM_FLEET_KB_SPOOL/cold-start.json"
```

Require `cold_start_ready=true`. Stop before a long run if Fleet bootstrap or
health fails.

## Add one launch flag

Append this flag to the workload Skill's existing `optimize` command:

```text
--degraded-kb
```

This disables Recipe KB so it cannot independently warm-start the demo. It
does not disable Fleet Experience reads or writes.

Do not duplicate or override the workload Skill's model, framework, GPU,
TP/EP, concurrency, ISL/OSL, precision, target, phase, budget, Docker, launch,
resume, or monitoring instructions. Do not force `--degraded-pr`; that is not
required by Fleet KB.

## Fleet acceptance

At run completion:

1. Query Fleet events for the exact scope and Slack thread.
2. Count every `kb.experience.cataloged` event; the Experience count is not
   fixed.
3. Require new records to remain Catalog `unverified`.
4. If the scope contains `verified_for_scope` Experiences, require a completed
   read with non-empty `rendered_refs` and verify those refs reach the measured
   Experience.
5. If the user explicitly chose an empty scope, require an empty read instead.
6. Report Experience IDs and any spooled writes without printing credentials.
