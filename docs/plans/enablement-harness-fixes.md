# ENABLEMENT harness fix plan (feat/zgong/enablement-fix)

Scope: H1, H4, H5, H6, H8, H9 from the 2026-09-22 root-cause report.
H2/H3/H7 are excluded — verified against current HEAD and found to already
be correctly implemented; no code changes needed there.

## H1 — AgentX recipe tree is not a patch-root candidate

`resolve_kernel_search_roots()` only discovers framework Python packages
(`FRAMEWORK_SOURCE_PACKAGES`) plus a handful of env-named checkouts. The
InferenceX recipe tree — the actual patch target in AgentX sessions — is
never a candidate, so `integrate_patch` always resolves `no_matching_root`
for a recipe-header patch.

Fix: add `$INFERENCEX_PATH` as one more discovered root in
`paths.py`, merged the same way `_env_source_roots()` is. No new
indirection through `manifest.json` — the orchestrator process already has
`INFERENCEX_PATH` in its own environment by the time a specialist or
integrate_patch runs.

## H4 — `extra_server_args` channel liveness is never checked

A recipe with no `$@`/`EXTRA_ARGS` sink silently no-ops every
`extra_server_args` proposal. The harness already parses the *observed*
server launch flags from the log (`launch_log_evidence.py`) and compares
them against the *requested* ones for identity verification
(`writeback.py::_identity_verification_status`). Reuse that comparison: a
measurement whose `requested_server_args` tokens are absent from
`observed_server_launch_flags` gets flagged, instead of silently scoring as
a normal measurement.

## H5 — recipe env overwrites make `extra_envs` a silent no-op for those names

No mechanism currently tells the harness which env names a recipe
re-exports unconditionally. Fix: statically scan the recipe script text (the
harness already reads/reference it at variant build time) for unguarded
`export NAME=...` lines and treat those names the same way `env_safety.py`
already treats `BLOCKED_VARIANT_ENV_NAMES` — reject them from
`extra_envs` up front instead of dropping the value silently downstream.

## H6 — `env_grant_requests` is a documented, unimplemented contract

No schema field, no parser, no consumer anywhere in the tree; the guidance
text also asserts a "prepend" merge semantic that contradicts the actual
last-wins `build_benchmark_env`. Fix: delete
`ENABLEMENT_ENV_GRANT_GUIDANCE` and its injection into the specialist
prompt. Nothing consumes it, so nothing else needs to change.

## H8 — research hints and derived constraints never go stale

`_coerce_hint` carries no timestamp, so a constraint derived from a
transient measurement lives forever in `research_hints.json`. Fix: stamp
each hint with `observed_at` (via the existing `now_iso` helper) at
ingestion, and drop hints past a fixed staleness window when hints are
loaded for prompting.

## H9 — a zero-measurement session still reports success

`outcome_status()` / `_exit_code_for_stop_reason()` only look at the
`stop_reason` string, so `time_exhausted` with `baseline_tput == 0` still
reports `completed`/exit 0. Fix: thread `baseline_tput` into both
functions and downgrade a success-shaped stop reason to `failed` when no
baseline measurement was ever produced.
