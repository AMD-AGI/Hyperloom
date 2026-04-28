# report — final write-up

**Family**: `shallow` · **Cost**: ~2‑5 min · **Risk**: zero

Compose a markdown summary at `<session_dir>/report.md` plus a structured
`report.json` with: baseline_tput, best_tput, cumulative_gain, all KEEP
decisions, accuracy_gate verdicts, and a chronological action history.

Always runs at end of session via `_graceful_stop` (DESIGN §7.2).

Outputs:

- artifacts: `report.md`, `report.json`
- `send_message` topic=`event` "session report ready"
