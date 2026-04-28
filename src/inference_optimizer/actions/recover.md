# Action: `recover` (STUB)

> Family: **resilience** · marathon-only · accuracy_risk=0.0.

Crash-recovery action. Reverts workspace to the latest checkpoint
backup, kills servers, restarts from the last KEEP-snapshot, and emits a
`recover_succeeded` event.

## TODO (IMPL-CHECKLIST §4.41)

- [ ] Read latest checkpoint via `Checkpoint.load_latest`
- [ ] `storage.backup.restore_from_backup` if local DB corrupt
- [ ] Hard requirement: `process_management.safe_kill_server` for both frameworks
