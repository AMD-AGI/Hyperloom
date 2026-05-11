# Action: Fake Baseline (test fixture)

This is a synthetic action `.md` used by inspector self-tests. It is NOT a
real action and is not referenced by any production skill. Edit only with
matching updates to [RUN_TESTS.md](RUN_TESTS.md) and
[fixture_transcript.jsonl](fixture_transcript.jsonl), since the test
expectations are tied to specific line numbers and quotes here.

## Procedure

You MUST run the warm-up script before everything else:

```bash
python3 $SKILL_ROOT/kb/kb_query.py --model "$MODEL_NAME" --top-k 5
```

Then MUST run the baseline benchmark:

```bash
bash "$SCRIPTS_DIR/run_fake_baseline.sh"
```

You MAY optionally run a smoke probe:

```bash
curl -s http://localhost:$PORT/v1/health
```

Mandatory accuracy gate (Iron Rule): `eval_accuracy.sh` MUST be invoked.
Violation = invalidation.

```bash
bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
```

## Outputs

- `$RESULT_DIR/fake_baseline.json` — main throughput record
- `$RESULT_DIR/eval_fake/eval_summary_fake.json` — accuracy summary
- `$RESULT_DIR/server_fake.log` — server log

## State

- Set `fake_baseline_tput` from the throughput record.
- Set `fake_baseline_accuracy` from the eval summary.
