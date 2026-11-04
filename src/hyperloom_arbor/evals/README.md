# Trajectory traces and evals

This directory records what Arbor's agents actually did in a campaign (every prompt, response,
tool call and decision) and turns it into a readable log and eval numbers: which iterations
earned their GPU time, how much went into REVERTs, whether a prompt edit changed behaviour.

## How it records

- **Hooks:** `.claude/settings.json` runs `python3 -S evals/arbor_trace.py hook` on every Claude
  Code event in this repository, so the agents of a campaign (which run with this repository's
  project settings) are recorded with no instruction added to them.
- **Marks:** the loop records what the filesystem cannot recover (the iteration, its hypothesis,
  the gate's decision) with `arbor_trace.py mark` (`arbor/cli/iterate.py`). A missing mark lowers
  fidelity; nothing is lost.
- The trace goes to `<session>/reports/trace/`; `python3 evals/arbor_trace.py where` prints where.

## After a session

```bash
python3 evals/arbor_trace.py collect --session-dir <session>
```

writes, idempotently (re-run it any time, mid-session included):

```
<session>/
  trajectory.jsonl        one JSON row per step, reward-annotated
  EVAL_REPORT.md          metrics and a verdict
  reports/trace/
    SESSION_LOG.md        the whole run, in order, readable
    meta.json             config snapshot, Arbor SHA, transcripts
```

A session recorded before tracing existed can be collected from its Claude transcript alone:

```bash
python3 evals/arbor_trace.py collect --session-dir /tmp/out \
  --transcript ~/.claude/projects/<slug>/<session-id>.jsonl
```

Compare sessions:

```bash
python3 evals/compare.py <sessions-dir> --group-by model
python3 evals/compare.py <sessions-dir> --group-by arbor_git_sha        # A/B a prompt edit
```

## Sources

| Source | Authoritative for |
|---|---|
| `reports/trace/events.jsonl` (hooks) | order, wall-clock timing, tool duration |
| the Claude transcript JSONL | prompt and response text, token usage, sub-agent steps |
| `iter_*/bench_output.json`, `session.json` | outcomes: the ground truth a self-report is checked against |

`evals/collect.py` reads `session.json` in the format `arbor/state.py` keeps compatible with it.

## Known gaps

- Some rules in `evals/report.py` and `evals/metrics.py` still cite the deleted slash commands
  (`hyperloom-optimize.md`) as their source.
- `evals/tests/test_pipeline.py`'s `CommandIntegrationTest` reads those deleted files: its four
  tests are the suite's four known failures.
- `arbor_trace.py` calls `os.uname()` at import, so it does not run on Windows Python.

## Tests

```bash
python3 -m unittest discover -s evals/tests -t evals/tests -v
```
