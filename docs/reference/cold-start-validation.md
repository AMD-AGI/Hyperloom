# Cold-start validation

A fresh Hyperloom host can pass package imports and keep an optimizer process
alive while being unable to perform any useful work. Long campaigns therefore
need two gates:

1. a bounded setup check that proves the installed environment and the actual
   orchestration transport work;
2. a short canary that proves the serving baseline and Framework attempt path
   work.

Neither gate may be replaced by checking only that a PID, `manifest.json`, or
`state.json` exists.

## Run the bounded setup check

Run the full installer first, then source the same environment the optimizer
will inherit:

```bash
export REPO_ROOT="$(pwd -P)"
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"

python3 "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/cold_start_check.py" \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --output "${USER_DATA_PATH}/optimizer_runs/cold_start_$(date -u +%Y%m%dT%H%M%SZ).json"
```

Add `--require-experience-kb` when the campaign is expected to collect
canonical Experiences. The check exits `2` if any required gate fails.

The report covers:

- active placeholder values in `.env`;
- `install.sh --check-only`;
- model path, torch/HIP visibility, GPU occupancy, and stale serving processes;
- serving-framework import and version;
- Experience KB declaration and local-store bootstrap when enabled;
- verified TLS to the configured LLM endpoint;
- one real, tool-free request through the production orchestration backend.

The last check is deliberately a model request, not only `GET /models`.
Authentication, Node/Claude Code trust, and the Messages endpoint can fail
after a catalog probe succeeds.

The check does not load the checkpoint into a serving engine or benchmark it.
That belongs to the canary.

## Run a canary before a long campaign

Use the intended model, framework, precision, TP, concurrency, ISL, and OSL,
but cap the first session to a short budget. Disable unrelated phases only when
the campaign is specifically validating Framework Experience collection.

Do not promote the environment to a long campaign until all of these are true:

- the optimizer has no terminal `stop_reason`;
- a `baseline` task exists and reaches a measured result;
- `state.json` has a positive `baseline_tput`;
- the run enters `FRAMEWORK_AGENT`;
- at least one Framework attempt reaches a terminal KEEP, REVERT, or FAILED
  outcome;
- CLOSE writes `session_breakdown.json` and
  `reports/experience_v1_publish.json`;
- the publish receipt accounts for every terminal Framework attempt as either
  selected or skipped.

For a live status check, resolve the session only from the launch-info JSON:

```bash
SESSION_DIR="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_dir"])' \
    "$LAUNCH_INFO_FILE"
)"
python3 "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/read_optimizer_state.py" \
  "$SESSION_DIR"
python3 "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/event_counts.py" \
  "$SESSION_DIR"
```

## MI300X cold-start findings, 2026-09-17

Two independent Qwen3-14B-FP8 / vLLM 0.29.0 sessions ran for approximately
twelve hours on MI300X. Both closed cleanly enough to write reports, but neither
created a baseline or entered Framework optimization.

The Experience receipts were identical:

```text
framework_attempts=0
selected=0
published=0
skipped=0
publish_errors=0
```

This was not an Experience quality-gate rejection. No upstream attempt existed
for the producer to project.

### ROCm compiler disappeared behind PATH reconstruction

The container initially resolved `/usr/bin/hipcc` even though its matching ROCm
7.2 toolchain was under `/opt/rocm`. A launcher had rebuilt `PATH` with
`/usr/bin` before the image's existing ROCm entries. The setup base check
correctly refused the mismatch.

Preserve the image's existing path order and prepend only the selected Python:

```bash
export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="${ROCM_PATH:-/opt/rocm}/llvm/bin:${ROCM_PATH:-/opt/rocm}/bin:$(dirname "$PYTHON"):${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
```

### `.env.template` activated credentials that were documentation only

On a source tree without `.env`, setup copied `.env.template`. Its OpenAI
example values were active assignments, including an angle-bracket hostname.
The next shell `source` interpreted the hostname as shell syntax; after the URL
was removed, the remaining example key formed a conflicting OpenAI side beside
the real Anthropic side.

Template credential examples are now comments. A generated `.env` contains
only values selected by setup or supplied by the operator.

### Later dependencies invalidated the Ray/Click pair

Ray 2.44.1 requires Click older than 8.3 for its CLI import path. The installer
established that pair, then installed GEAK and the Claude SDK; their dependency
resolution upgraded Click. The full installer returned success, but an
immediate `install.sh --check-only` failed with:

```text
click version incompatible with Ray CLI
```

The kernel installer now reasserts and verifies the certified Ray/Click pair
after GEAK dependencies are installed. This is why the bounded setup check runs
`--check-only` after the full installer rather than trusting its exit code
alone.

### The gateway certificate chain was not trusted

Both Python and the Node-based Claude CLI rejected the AMD-internal issuing CA:

```text
SSL certificate verification failed (UNABLE_TO_VERIFY_LEAF_SIGNATURE)
```

For the AMD Primus-SaFE gateway, install its documented CA bundle inside the
container:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/setup.sh \
  | bash
export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

When `NODE_EXTRA_CA_CERTS` is unset, Hyperloom's Claude SDK environment
inherits the first existing file named by `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, or `CURL_CA_BUNDLE`. Setting the Node variable
explicitly remains clearer for launch scripts and takes precedence.

For another private gateway, install its own CA instead. Never disable TLS
verification.

After installing the CA, both nodes passed Python TLS, an authenticated model
catalog request, Node TLS, the Claude CLI, and the production
`ClaudeBackend` round trip. The URL and token were not the failure.

### A catalog warning allowed a non-functional launch

The catalog probe retried and reported the certificate failure, but custom
model support converted an unreachable gateway into a warning. That policy was
incorrect: a custom model id changes model selection, not transport
reachability. An unreachable catalog now refuses to launch. A gateway that
explicitly returns 404/405 for an unsupported `/models` route remains a
distinct case.

### Backend health was observable but not terminal

Each session emitted `backend_unhealthy` after five failed orchestration turns,
but the coordinator continued retrying:

- each run recorded 513 TLS failures;
- the unread error history then enlarged the prompt;
- the remaining approximately 790 turns failed with `Prompt is too long`;
- the only tasks ever created were CLOSE report and session-breakdown tasks.

PRELUDE now stops with `prelude_orchestration_unavailable` when the
orchestration backend crosses its error threshold before it has either produced
a baseline or placed baseline work in flight. A baseline already running is
allowed to finish.

### The initial health check was too shallow

The first check declared both sessions healthy because their PIDs,
`manifest.json`, and `state.json` existed and `stop_reason` was empty. The first
orchestration failure was already in the event log.

A cold-start launch is not healthy until the bounded setup check passes and the
canary shows task progress. Process liveness remains useful for crash recovery,
but it is not evidence of optimization progress.

### Container exit and session completion are separate

One container later exited with code 255 after the optimizer had completed
CLOSE and the campaign finalizer had persisted its results. Container state
must still be monitored, but it did not cause the zero-Experience result.
Session artifacts under the host-mounted campaign directory remained
authoritative.
