# Inference Optimization — Addendum (Agent Execution Rules)

These rules are derived from observed Claw-driven session failures. They are
**non-negotiable** for agent-driven runs of the inference-optimization skill,
and supplement (not replace) `SKILL.md`, `modes/REMOTE.md`, etc.

---

## ADDENDUM-01: Never block a single bash on long polling/waiting

A single `bash` tool call must not be used to wait for a remote condition
(RayJob phase, server `/health`, Ray Job submission status, model loading).
Long blocking calls are killed by the sandbox/MCP layer with SIGTERM /
`Request timed out`, and that error often consumes the remaining turn budget
of the message.

Required pattern:

1. Start the long-running thing asynchronously and return an identifier
   immediately:
   - For SaFE workloads: `workload_create` returns `workload_id`.
   - For Ray Dashboard jobs: `POST /api/jobs/` returns `submission_id`.
2. Poll until the work actually finishes — do not give up after a few checks.
   Use **many short bash calls in sequence**, each one:
   - Has bash `timeout` ≤ 60 s.
   - Inside bash, no long `sleep`. Do at most a few quick checks and return
     immediately with the latest status.
   - Prints the latest progress signal (phase, last log line, tail of pod log,
     Ray Job status), so each turn shows real progress instead of
     "still waiting".
3. Between bash calls, the agent should briefly judge progress, then issue the
   next short poll on the next turn. There is no fixed cap on number of polls —
   keep going until the work is `Succeeded` / `Failed` / `Stopped`, or the
   user cancels.
4. Never write `for i in $(seq 1 N); do ... sleep 30; done` inside one bash to
   "wait it out".

---

## ADDENDUM-02: Do not depend on a Ray client inside the Claw sandbox

The Claw sandbox is a control plane (issue commands, poll status, read logs).
It is **not** a Ray client host. Trying to use `ray://<head>:10001` from the
sandbox is fragile because the sandbox image and the RayJob image evolve
independently — Ray and Python `major.minor` will drift, and
`ray.init(address="ray://...")` will fail with `Version mismatch`.

Required pattern:

1. Do not install Ray inside the sandbox. Do not call
   `ray.init(address="ray://...")` from sandbox bash.
   `scripts/ray_submit.py` (which uses Ray client) is therefore not a reliable
   path from the sandbox.
2. Do all remote execution via HTTP / structured tools, in this order:
   a. Submit ad-hoc work via Ray Dashboard REST:
      `POST http://<head_ip>:8265/api/jobs/` with
      `{"entrypoint": "<bash command>"}`. Read status/logs at
      `GET /api/jobs/<submission_id>` and `/api/jobs/<submission_id>/logs`.
      This bypasses Ray client version checks entirely.
   b. For non-trivial scheduling (node pinning, multi-actor placement), put a
      Python driver script that uses `ray.init()` (no address) **inside** a
      Ray Job entrypoint. Inside the cluster image, the Ray and Python
      versions match by construction.
   c. For workload-level operations, use the structured MCP tools (e.g.
      `workload_get`, `workload_pod_logs`).
3. Never spend turns trying to upgrade/downgrade Python or Ray inside the
   sandbox (`apt install python3.X-venv` etc.). It consistently fails.

---

## ADDENDUM-03: RAY_JOB_ENTRYPOINT — choose by workflow shape

`RAY_JOB_ENTRYPOINT` (base64) defines what the RayJob will execute when it
starts. The job's lifecycle ends when this entrypoint exits, so the choice
has to match the workflow shape:

**Mode A — Self-contained entrypoint (one-shot batch):**

- Use only when the entire pipeline can run as one non-interactive script
  (e.g. a fixed benchmark + report).
- The script must do everything end-to-end and exit. Once it exits, the
  RayJob enters Stopped and the pods are torn down.
- Not suitable for iterative agent-driven flows
  (baseline → profile → kernel-opt → integrate → re-baseline), because the
  agent has nothing to come back to.

**Mode B — Idle entrypoint + Ray Dashboard REST (default for agent-driven runs):**

- Set `RAY_JOB_ENTRYPOINT` to `tail -f /dev/null` (base64) so the job stays
  alive for the whole session and the agent can submit successive steps.
- Submit each step via Ray Dashboard REST:
  `POST http://<head_ip>:8265/api/jobs/` with
  `{"entrypoint": "<bash command>"}`. This is the supported control channel.
  - `POST /api/jobs/` returns immediately with a `submission_id`; it does
    not block.
  - Step status: `GET /api/jobs/<submission_id>` (returns `status` ∈
    {PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED}).
  - Step logs: `GET /api/jobs/<submission_id>/logs` (returns
    `{"logs": "..."}`).
  - The HTTP calls themselves are short and never wait for the underlying job
    to finish — the agent polls (per ADDENDUM-01).
- Do **not** try to drive the cluster via Ray client (`ray://...`), SSH, or
  by adding a `Service`. None of these are reliable from the Claw sandbox.

In this flow, all GPU work runs inside the RayJob image where
Ray/Python/SGLang/vLLM versions match by construction. The Claw sandbox stays
on the control path: submit, poll, read logs.

---

## ADDENDUM-04: Multi-process GPU servers must be pinned to distinct nodes

If the workload requires multiple GPU server processes that each consume a
full node's GPUs (e.g. PD disaggregation: prefill + decode), they MUST be
placed on different nodes. The default behavior of
`nohup python3 -m sglang.launch_server ... &` from a Ray Dashboard driver
runs everything on the head node and OOMs the second process.

Required pattern:

1. Submit a Python driver as one Ray Job (`POST /api/jobs/`) that uses
   `ray.init()` to talk to the in-cluster Ray runtime.
2. Inside that driver, enumerate cluster nodes via `ray.nodes()`. Record each
   `NodeManagerAddress` and which one is head
   (`Resources["node:__internal_head__"] > 0`) vs worker.
3. Place each GPU server with explicit node affinity:
   - Wrap the server launch in a Ray actor and schedule with
     `NodeAffinitySchedulingStrategy(node_id=<that node>, soft=False)`.
   - For PD disaggregation: prefill on one node, decode on the other. Which
     one is head vs worker does not matter functionally; the rule is
     "different nodes".
4. Verify placement before launching by logging `socket.gethostname()` from
   inside each actor.
5. Per-instance `--mem-fraction-static` (or vLLM equivalent) must assume the
   full node, not be split between co-located processes.
6. When the cluster has more than one node candidate per role, choose
   deterministically (e.g. sort node IDs) so reruns don't relocate processes.

---

## ADDENDUM-05: Use structured tool calls or robust SSE parsing for SaFE/MCP responses

SaFE MCP (`/api/v1/safe-mcp/mcp`) responses are SSE
(`event: message\ndata: {...}\n\n`). Plain `grep '"phase":"..."'` misses
fields and `json.load` on the raw stream throws `Invalid control character`.

Required pattern:

1. Prefer the structured MCP tools available to the agent (e.g.
   `workload_get`, `workload_pod_logs`, `opsjob_get`) instead of hand-rolled
   `curl` to `/safe-mcp/mcp`.
2. If `curl` is unavoidable: send `Accept: application/json, text/event-stream`,
   then parse only `data:` lines:

   ```python
   data = "\n".join(l[6:] for l in resp.splitlines() if l.startswith("data: "))
   payload = json.loads(data)
   inner = json.loads(payload["result"]["content"][0]["text"])
   phase = inner.get("phase")
   ```

3. Never compare unparsed status with `[ "$STATUS" = "Running" ]` derived from
   `grep`. Parse JSON explicitly and check the field.

---

## ADDENDUM-06: Treat /wekafs as read-only from the Claw sandbox; do not probe unknown SaFE endpoints

From the Claw sandbox side:

- `/wekafs/...` is **read-only**. Do not attempt `mkdir`, `touch`, `cp` into
  `/wekafs`. As soon as a write attempt to `/wekafs/...` returns
  `Read-only file system`, switch to `/workspace/...` for any sandbox-side
  scratch / generated files. `/workspace/` is writable and Claw persists it
  (e.g. to S3) across the session.
- The RayJob pods can read AND write `/wekafs`, so use `/wekafs` only for
  files produced inside the RayJob, and read-only references from the sandbox.
- There is no public REST endpoint for SSH-key upload exposed by SaFE. Do not
  iterate over guessed paths (`/api/v1/users/.../ssh-keys`, etc.); they all
  404 and waste turns. The SSH gateway is not a supported execution path for
  the agent.
- Do not try to add a `Service` on a RayJob to expose internal ports; the
  admission webhook only allows specific `service.type` values and most
  attempts will be rejected. Use Ray Dashboard REST (`:8265/api/jobs/`) or
  the structured MCP tools instead.

---

## ADDENDUM-07: Always bootstrap OOB / TraceLens inside the RayJob before kernel-opt

The default RayJob image (e.g. `sglang:*`) is a vanilla framework image. It
does **not** ship the OOB CLI, claude CLI, codex CLI, TraceLens, the
AMD CA bundle, or the OOB auth-proxy. These are installed BYOI-style by
`scripts/bootstrap.sh` (script is idempotent; the marker
`/opt/hyperloom/.bootstrap_done` short-circuits subsequent runs).

If bootstrap is skipped, `which oob` returns nothing inside the RayJob pods.
The agent must NOT silently treat this as "OOB unavailable, fall back to
direct provider API calls" — that violates IR-7b
("orchestrator must use the configured `KERNEL_OPT_BACKENDS` toolchain").
The right fix is always to run bootstrap inside the RayJob.

Required pattern:

1. Right after the RayJob enters `Running` (per ADDENDUM-01 polling) and
   BEFORE any kernel-opt / TraceLens / OOB action, submit a Ray Dashboard
   REST job that runs the BYOI bootstrap inside the cluster:

   ```bash
   POST http://<head_ip>:8265/api/jobs/
   {"entrypoint":
     "bash $SKILL_ROOT/scripts/bootstrap.sh"}
   ```

   `$SKILL_ROOT` defaults to
   `/wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization`. Poll
   the submission to terminal status (per ADDENDUM-01) — do not block on it.
2. Verify with one short follow-up REST job before continuing:

   ```bash
   {"entrypoint":
     "source /etc/profile.d/hyperloom-env.sh && \
      which oob && which claude && which codex && which ray && \
      oob --help | head -5"}
   ```

   If any of `oob` / `claude` / `codex` / `ray` is missing, re-run
   bootstrap with `--force` and inspect `/var/log/hyperloom/*.log`. Do not
   try to install dependencies ad-hoc with `pip install ray ...` /
   `apt install ...` — that is what bootstrap is for.
3. Every later REST job that needs the Hyperloom toolchain must
   `source /etc/profile.d/hyperloom-env.sh` first, so it picks up the
   bootstrap-written values for `PATH`, `OOB_RAY_CLI`, `OOB_CLI`
   (`/opt/venv/bin/oob`), `ANTHROPIC_BASE_URL` (auth-proxy on
   `127.0.0.1:4002`), `OPENAI_BASE_URL`, `AMD_LLM_API_KEY`, `SKILL_ROOT`,
   `SCRIPTS_DIR`, `KERNEL_OPT_BACKENDS`, `INFERENCEX_PATH`, `TRACELENS_ROOT`.
   Do not redefine these by hand; the env file is the single source of
   truth that matches the bootstrap that actually ran.
4. OOB / claude / codex are installed inside the **RayJob pods**, not
   the Claw sandbox. Do not try to install or invoke `oob` / `claude` /
   `codex` from sandbox bash. All such calls go via Ray Dashboard REST so
   they execute inside the RayJob image.
5. Do not silently fall back to direct provider API calls
   because `oob` is missing. Per IR-7b the orchestrator must use the
   configured `KERNEL_OPT_BACKENDS`. Either fix bootstrap, or change
   `KERNEL_OPT_BACKENDS` explicitly with the user's confirmation.

---

## ADDENDUM-08: Don't reinvent skill scripts in the sandbox

The skill ships official tools under `$SKILL_ROOT/scripts/` and `actions/*`,
and the agent has structured MCP tools (e.g. `workload_create`,
`workload_get`, `workload_pod_logs`) plus the Ray Dashboard REST API. Do
not write parallel re-implementations in `/workspace/` (such as
`mcp_call.js`, `ray_submit.js`, `ray_poll.js`, ad-hoc provider-call scripts,
`sweep.py`, `launch_pd_servers.py`, `parse_profile.py`, etc.).
Re-implementations fragment the workflow, drift from the skill (versioning
as `*_v2.py` / `*_v3.py`), and break inspector / IR-2 / IR-3 traceability.

Required pattern:

1. Before writing any helper into `/workspace/`, check whether the same
   thing is already covered by:
   - A script under `$SKILL_ROOT/scripts/` (e.g. `run_baseline.sh`,
     `run_profile.sh`, `run_sweep.sh`, `oob_ray_submit.py`,
     `trace_action.py`, `bootstrap.sh`).
   - A documented action in `$SKILL_ROOT/actions/*.md`.
   - A structured MCP tool (`workload_*`, `opsjob_*`, `node_*`,
     `cluster_*`).
   - The Ray Dashboard REST API (`POST/GET :8265/api/jobs/...`).
   If so, use that. Do not duplicate it in `/workspace/`.
2. Sandbox-side `/workspace/` is for genuinely new artefacts (kernel
   files, prompts, intermediate results, the final report). It is not for
   shadow copies of `oob_ray_submit.py`, `mcp_call.js`, `ray_submit.js`,
   etc.
3. If a needed helper truly does not exist in the skill, add it under
   `$SKILL_ROOT/scripts/` (or extend the action doc) instead of dropping a
   one-off in `/workspace/`. Otherwise the next session will not see it.
4. Never call out to provider gateways from sandbox bash. All OOB-backed kernel
   work goes through the configured
   `KERNEL_OPT_BACKENDS` (OOB Codex / Claude CLI) inside the
   RayJob, per IR-7b and ADDENDUM-07.

---

## ADDENDUM-09: Hard cap on `sleep` inside a bash call

Every bash tool call has a sandbox-enforced ceiling around 120 s. Any
single `sleep` longer than that will cause the bash to be timed out with
empty stdout/stderr, wasting a full turn and producing no progress signal.

Required pattern:

1. Inside one bash call, the **sum of all `sleep` durations MUST be
   ≤ 30 s** (rule of thumb; leave headroom for the actual work the bash
   also does).
2. Never write `sleep 60` / `sleep 120` / `sleep 180` / `sleep 240` /
   `sleep 300` in a bash. If a longer wait is needed, return from the
   bash, then issue the next short bash on the next turn (per
   ADDENDUM-01).
3. For "wait for X to be ready" patterns, prefer fast probes over fixed
   sleeps:
   - Probe a Ray Job once with `GET /api/jobs/<submission_id>` (or
     `workload_get` for SaFE).
   - If still pending, return immediately and re-probe on the next turn.
   - Do not bury the probe behind a long pre-`sleep`.
4. Do not chain `sleep 30 && sleep 30 && sleep 30` either; the cap is on
   total wall time inside the bash, not on individual `sleep`s.

---

## ADDENDUM-10: Profiling must go through the skill's profile flow

`actions/profile.md` defines the supported profiling pipeline:
`run_baseline.sh` (or `run_profile.sh`) on the cluster → TraceLens
analysis → kernel candidate list. The pipeline also requires
`scripts/trace_action.py --component ... --action start|end` for cost
attribution. Hand-rolled `urllib.request` calls to
`http://<server>:<port>/start_profile` / `/stop_profile` from sandbox
bash are not a substitute — they skip TraceLens, skip the candidate
extraction format, and skip tracing.

Required pattern:

1. Trigger profiling exactly as `actions/profile.md` (and the
   mode-specific `modes/REMOTE.md` "Profile" section) describe — usually
   by submitting a Ray Job that runs `bash $SCRIPTS_DIR/run_profile.sh`
   (or relying on `run_baseline.sh`'s built-in profile phase).
2. Wrap the external call with
   `python3 $SCRIPTS_DIR/trace_action.py --component tracelens --action start`
   before and `--action end` after, per SKILL.md "Common Pitfalls" #6.
3. Run TraceLens on the produced trace (or use the kernel_summary fallback
   path documented in the skill). Do not parse `*.json.gz` traces by hand
   in `/workspace/parse_profile*.py`.
4. The kernel candidate list passed to `kernel-opt` MUST come from this
   pipeline (per IR-1, "submit ALL kernel candidates"). Do not invent
   candidates from a custom inline profile.

---

## ADDENDUM-11: Don't run `find /` / `grep -r /` anywhere

`actions/kernel-opt.md`, `kernel-opt/claude.md`, and `kernel-opt/codex.md`
already say: "Do NOT search the filesystem with `find /` or `grep -r /`."
This rule applies to BOTH sides — Claw sandbox AND RayJob pods. Whole-disk
searches routinely run into the sandbox/MCP 120 s ceiling and are killed
with SIGTERM after wasting an entire turn (or, when given longer in a Ray
Job, can run for 60+ minutes scanning network filesystems).

Required pattern:

1. Never write `find / ...`, `grep -r / ...`, `find / -path "*/foo/*"`,
   etc. in either sandbox bash or Ray Job entrypoints.
2. To locate kernel source files, use the known roots only:
   `/sgl-workspace/`, `/opt/venv/`, `/opt/hyperloom/`,
   `/wekafs/InferenceX/`, `/wekafs/fully-local/`, `/tmp/torchinductor_root/`.
   Always pass an explicit root and a small `-maxdepth` (≤ 4).
3. To find a file by Python module name, prefer (inside the RayJob, where
   `python3` is available):
   ```python
   import importlib, os
   m = importlib.import_module("aiter")
   print(os.path.dirname(m.__file__))
   ```
   over `find`.
4. If the path really isn't predictable, use a single, bounded
   `find <known-root> -maxdepth 4 -name "<file>"` rather than `find /`.

---

## ADDENDUM-12: Don't assume `python3` is present in the Claw sandbox

The Claw sandbox image is a control-plane image and is **not** guaranteed
to have `python3`. Common case: `which python3` returns nothing,
`python3 -m json.tool` exits 127, etc. Node.js is generally available in
the sandbox.

This rule is sandbox-only. Inside the RayJob (where commands submitted via
`POST /api/jobs/` actually run), `python3` is present in the framework
image, and after `bootstrap.sh` `/opt/venv/bin/python3` is on `PATH`. The
rule is purely about which side of the boundary the script runs on.

Required pattern:

1. In sandbox bash:
   - Use `node -e '...'` (or pipe through `node`) for JSON / SSE parsing
     and small glue logic.
   - Do not call `python3` from sandbox bash unless a previous probe in
     the same session has already shown `which python3` succeeds.
   - Do not try to `apt install python3`, `pip install ...`, etc. The
     sandbox is not the place to install language runtimes (same spirit
     as ADDENDUM-02 for Ray).
2. In a Ray Job entrypoint (executes inside the RayJob image): `python3`
   is fine and expected. After `bootstrap.sh`, prefer
   `/opt/venv/bin/python3` (or just `python3` after sourcing
   `/etc/profile.d/hyperloom-env.sh`).
3. Don't mix the two sides: never paste a `python3 -c "..."` that was
   meant for the RayJob into a sandbox bash command.

---

## ADDENDUM-13: Propagate API credentials from sandbox to RayJob before bootstrap/OOB

The Claw sandbox and the RayJob pods do not automatically share environment
variables. A key that exists in sandbox bash (for example `SAFE_API_KEY`) is
NOT automatically visible inside the RayJob. `bootstrap.sh` only renders
`/etc/profile.d/hyperloom-env.sh` from environment variables that are present
inside the RayJob at bootstrap time.

Required pattern:

1. When creating the RayJob via `workload_create`, propagate the current
   available sandbox credentials into the RayJob `env` field. Prefer this
   mapping:

   - `SAFE_API_KEY` = current sandbox `SAFE_API_KEY`
   - `OOB_API_KEY` = current sandbox `SAFE_API_KEY` unless explicitly provided
   - `AMD_LLM_API_KEY` = current sandbox `SAFE_API_KEY` unless explicitly provided
   - `LLM_API_KEY` = current sandbox `SAFE_API_KEY` unless explicitly provided
   - `ANTHROPIC_API_KEY` = current sandbox `SAFE_API_KEY` unless explicitly provided
   - `OPENAI_API_KEY` = current sandbox `SAFE_API_KEY` unless explicitly provided
   - `OOB_BASE_URL` = `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1`

2. After RayJob is `Running` and bootstrap has run, verify inside the RayJob:

   ```bash
   source /etc/profile.d/hyperloom-env.sh
   test -n "$ANTHROPIC_API_KEY"
   test -n "$OPENAI_API_KEY"
   test -n "$AMD_LLM_API_KEY"
   test -n "$ANTHROPIC_BASE_URL"
   ```

   Also verify `claude -p "Reply OK only" --model claude-opus-4-7 --print`
   or an equivalent minimal OOB smoke test.

3. If the RayJob was already created without credentials and the env file is
   empty, do not proceed with OOB. Submit a short Ray Dashboard REST job that
   injects the currently available sandbox key into the entrypoint, then reruns
   bootstrap with `--force`:

   ```bash
   SAFE_KEY="<current sandbox SAFE_API_KEY>"
   POST http://<head_ip>:8265/api/jobs/
   {
     "entrypoint": "bash -lc 'export SAFE_API_KEY=\"$SAFE_KEY\" OOB_API_KEY=\"$SAFE_KEY\" AMD_LLM_API_KEY=\"$SAFE_KEY\" LLM_API_KEY=\"$SAFE_KEY\" ANTHROPIC_API_KEY=\"$SAFE_KEY\" OPENAI_API_KEY=\"$SAFE_KEY\" OOB_BASE_URL=\"https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1\"; bash /wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization/scripts/bootstrap.sh --force'"
   }
   ```

4. When an OOB / Claude command reports a missing API key
   (`no --api-key`, `ANTHROPIC_API_KEY not set`, `apiKeySource: none`, empty
   auth config), the agent must immediately use the currently available
   sandbox key as above. Do not continue with unauthenticated OOB,
   and do not silently fall back to direct provider API calls.