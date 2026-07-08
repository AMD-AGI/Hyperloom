# Stage 4 - Isolate and run (`isolate_and_run`)

> LLM-facing skill prompt for the final stage: drop audit material,
> optionally create an isolated worktree + venv, run build / bench /
> accuracy commands, evaluate the winner gate, and contribute the
> verdict to the KB.

## Intent

For every kept candidate:

1. Drop `pr.patches` + `pr_files.json` under
   `${work_dir}/candidates/<index>_<slug>/`.
2. If `execute=True` and `prepare_candidate_env=True`, create a
   detached git worktree at the PR head and a per-candidate venv.
3. If `execute=True`, run the configured `build` -> `benchmark` ->
   `accuracy` -> `cleanup` commands (any subset may be omitted).
4. Read `outputs.benchmark_json` + `outputs.accuracy_json`, apply
   the three-gate winner check (`min_throughput_ratio`,
   `max_accuracy_drop`, `completed` parity).
5. On the first winner, **early-break** the per-candidate loop.
6. If `kb_domain` is set and a winner exists, append a Finding to
   `${KB}/<domain>/empirical_kb.md` via the explorer's KB hook.

## Inputs

| Field | Type | Notes |
|---|---|---|
| `request` | ExploreRequest | Includes `commands`, `outputs`, `baseline`, `thresholds`, `kb_domain`, ... |
| `candidates` | list[Candidate] | Output of Stage 3 (filtered). |
| `execute` | bool | False = plan mode (only audit material drop). True = full pipeline. |

## Tool surface

```python
from framework_agent.explorer import explore
# or for stateless reuse:
from framework_agent.runtime.tools_api import (
    fetch_pr_audit_material,
    evaluate_candidate_outcome,
)
```

For driving the full pipeline at once, `framework_agent.explorer.explore(req,
execute=...)` is the canonical entry. For piecewise control, mix the
`tools_api` helpers above.

## Procedure

### Plan mode (`execute=False`)

1. For each candidate, materialise `candidate_dir` and call
   `_write_pr_artifacts` (or `fetch_pr_audit_material` from
   `tools_api`).
2. Emit a `planned` result per candidate; no commands run; no winner
   selected; promotion_policy stays `manual_only`.

### Execute mode (`execute=True`)

1. Materialise `candidate_dir` as above.
2. If `prepare_candidate_env=True`, prepare the worktree + venv via
   `_prepare_candidate_workspace`. Otherwise skip and run commands
   against the existing global environment.
3. Render command templates with `render_template(spec.command,
   variables, shell_quote=True)`. The variable bag includes
   `candidate_ref`, `candidate_repo`, `candidate_dir`, `worktree_dir`,
   `venv_dir`, `venv_bin`, `framework`, `repo_url`, `work_dir`.
4. Run `build` -> `benchmark` -> `accuracy` -> `cleanup` in that fixed
   order. A non-zero rc on any `required` command shorts the
   candidate to `status="failed"`.
5. Apply `_winner_decision` to throughput / accuracy / completed.
   The first winner ends the loop early.
6. If `kb_domain` is set and a winner exists, append a synthesised
   Finding to `${KB}/<domain>/empirical_kb.md` via the explorer's
   KB hook.

## Output contract

`explore_summary.json` schema (top-level keys, ordered as emitted):

```jsonc
{
  "ok": true,
  "mode": "plan" | "execute",
  "framework": "sglang",
  "repo_url": "...",
  "work_dir": "...",
  "baseline": { "throughput": ..., "accuracy": ..., "completed": "..." },
  "thresholds": { "min_throughput_ratio": ..., "max_accuracy_drop": ... },
  "winner_ref": "PR:22918" | null,
  "winner_dir": "/.../candidates/01_pr-22918" | null,
  "promotion_policy": "manual_only",
  "promotion_hint": "...",
  "pr_filter_applied": { ... },
  "skipped_candidates": [ ... ],
  "audit_materials": {
    "patch_files_present": 3,
    "files_json_present": 3,
    "policy": "patches_and_files_only"
  },
  "kb_contribution": {
    "status": "appended" | "skipped" | "failed",
    "domain": "framework",
    "path": "...",
    "finding_title": "..."
  },
  "candidates": [ CandidateResult, ... ]
}
```

## Failure modes

| Symptom | Resolution |
|---|---|
| `build` rc != 0 | `status=failed`, `winner=False`, candidate dropped from contention; rest of pipeline still runs for remaining candidates. |
| `benchmark.json` missing | `_evaluate_candidate` returns `throughput=None` -> winner gate rejects with `"missing throughput"`. |
| `accuracy.json` missing while `baseline.accuracy` set | Reject with `"missing accuracy while baseline accuracy is set"`. Operators wanting "throughput only" should leave `baseline.accuracy` null. |
| `completed != "N/N"` | Reject with `"benchmark completed=<...> is incomplete"`. Set `completed` to an exact `N/N` string in your benchmark command output JSON. |
| KB hook OSError (read-only KB) | `kb_contribution.status="failed"`, but the rest of the run is unaffected. |

## Promotion

`framework-agent` is **manual-promotion only**. `winner_dir`
identifies the candidate worktree / venv combination ready for
promotion; the operator (not the agent) decides whether to merge,
revert, or shelve. Never auto-merge.
