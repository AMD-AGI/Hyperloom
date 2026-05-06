# Action: Parameter Sweep (Sweep-Only Skill)

> **TL;DR.** Generate a single Magpie YAML with `sweep_matrix`, run one
> `magpie benchmark` call. Server is launched once and reused across all
> CONC/ISL/OSL cases.

## Inputs

Provided by `actions/setup.md`:
- `MODEL`, `TP`, `FRAMEWORK`, `RUNNER_TYPE`, `INFERENCEX_PATH`, `RESULT_DIR`
- `EXTRA_SGLANG_ARGS` or `EXTRA_VLLM_ARGS` (caller-supplied; mandatory)

Optional caller override:
- `SWEEP_CASES_YAML` — multi-line YAML body listing extra `cases` entries.
  When unset, the default matrix is used (see below).

## Default Sweep Matrix

5 cases (~ 1 server startup + 5 client runs):

| CONC | ISL | OSL |
|-----:|----:|----:|
|    4 | 1024 | 1024 |
|   16 | 1024 | 1024 |
|   64 | 1024 | 1024 |
|   16 | 8192 | 1024 |
|   16 | 1024 | 8192 |

## Procedure

**Claw mode:** wrap the single `magpie benchmark` call below with
`exec_on_gpu`.

```bash
EXTRA_ARGS_KEY="EXTRA_$(echo "$FRAMEWORK" | tr '[:lower:]' '[:upper:]')_ARGS"
SWEEP_DIR="$RESULT_DIR/sweep_$(date +%Y-%m-%d-%H-%M)"
mkdir -p "$SWEEP_DIR"

CASES_YAML="${SWEEP_CASES_YAML:-$(cat <<'YAML'
      - { CONC: 4,  ISL: 1024, OSL: 1024 }
      - { CONC: 16, ISL: 1024, OSL: 1024 }
      - { CONC: 64, ISL: 1024, OSL: 1024 }
      - { CONC: 16, ISL: 8192, OSL: 1024 }
      - { CONC: 16, ISL: 1024, OSL: 8192 }
YAML
)}"

cat > "$SWEEP_DIR/sweep_config.yaml" <<EOF
benchmark:
  framework: $FRAMEWORK
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: ${FRAMEWORK}_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    RANDOM_RANGE_RATIO: 0.5
    $EXTRA_ARGS_KEY: "${!EXTRA_ARGS_KEY}"
  timeout_seconds: 7200
  profiler:
    torch_profiler:
      enabled: false
  sweep_matrix:
    cases:
$CASES_YAML
    on_failure: continue
    inter_client_sleep_s: 5
EOF

# Do NOT run magpie in a foreground bash — a sweep with server-startup can
# take 20-90 min; a proxy/ingress typically kills idle HTTP connections after
# 60s, dropping the MCP response even though hands-binary sent it correctly.
# Validated regression: session ed26388c 2026-05-06, sweep finished but brain
# never received the result due to idle-connection teardown.
#
# Use the Background Runner Recipe (run_in_background=true + bash_output):
```

```text
# Step 1 — launch (returns shell_id immediately):
shell_id ← bash(
  command = "export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $SWEEP_DIR/sweep_config.yaml -o $SWEEP_DIR 2>&1",
  run_in_background = true
)

# Step 2 — poll every 120s (sweep server startup + N cases takes longer than baseline):
while True:
    bash(command = "sleep 120", timeout = 130)
    out ← bash_output(shell_id = shell_id)
    if out matches "Benchmark Result|sweep_report\.json|Execution time"  : break (DONE)
    if out matches "Traceback|FATAL|exit [1-9]|signal=SIG|OOM|server failed": break (ERROR)
    if elapsed > 7200 : kill_shell(shell_id); break (TIMEOUT)

# Step 3 — collect results:
bash(command = "cat $SWEEP_DIR/benchmark_*/sweep_report.json 2>/dev/null \
                || cat $SWEEP_DIR/benchmark_*/benchmark_report.json 2>/dev/null")
bash(command = "cat $SWEEP_DIR/benchmark_*/results.tsv 2>/dev/null | head -20")
```

## Constraints (enforced by Magpie at parse time)

- `sweep_matrix.cases` may only override client-side env vars: `CONC`, `ISL`,
  `OSL`, `NUM_PROMPTS`, `RANDOM_RANGE_RATIO`.
- Server-side params (`TP`, framework backend, memory fraction, `EXTRA_*_ARGS`)
  must remain fixed for the whole sweep.
- `profiler.torch_profiler.enabled` must be `false`. Profiling and sweep are
  mutually exclusive in Magpie.
- `run_mode: local` is required (or `ray` indirectly via `exec_on_gpu` in
  claw mode). `docker` is rejected.

## Outputs

- `$SWEEP_DIR/<timestamp>/sweep_report.json` — aggregated case results + best case
- `$SWEEP_DIR/<timestamp>/results.tsv` — `CONC ISL OSL output_tput tput_per_gpu TPOT_mean TTFT_mean success`
- `$SWEEP_DIR/<timestamp>/case_*/inferencex_result.json` — raw per-case results
- `$SWEEP_DIR/<timestamp>/server.log` — single shared server log

## Quick Pareto Pretty-Print

```bash
SWEEP_RESULT=$(ls -td "$SWEEP_DIR"/sweep_* | head -1)
TSV="$SWEEP_RESULT/results.tsv"
echo
echo "=== Top 3 by tput_per_gpu ==="
{ head -1 "$TSV"; tail -n +2 "$TSV" | sort -k5 -nr | head -3; } \
    | column -t -s $'\t'
echo
echo "Full results: $TSV"
echo "Sweep report: $SWEEP_RESULT/sweep_report.json"
```

## Failure Handling

- Individual case fails: logged in `sweep_report.json`, sweep continues
  (`on_failure: continue`).
- Server fails to start: entire sweep fails; inspect
  `$SWEEP_RESULT/server.log` and the runner stderr.
- Server crashes mid-sweep: detected by Magpie's `/health` poll between
  cases; remaining cases recorded as failed.
