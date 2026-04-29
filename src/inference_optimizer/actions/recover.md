# recover — resume from latest checkpoint

**Family**: `resilience` · **Cost**: ~5‑15 min · **Risk**: low

Restore from `<session_dir>/checkpoints/<latest>/conductor.db.bak` via
`storage.backup.restore_from_backup`, replay events past the recorded
cursors, and run `evidence_check_matrix` on every in‑flight task.

Verdicts:

- `succeeded` — task evidence intact, mark task `succeeded`
- `safely_failed` — evidence partially intact but result not usable
- `evidence_insufficient` → `needs_manual_review`
