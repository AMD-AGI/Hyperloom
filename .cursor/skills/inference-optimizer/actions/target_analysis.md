# target_analysis — read user-provided target

**Family**: `prep` · **Cost**: ~2‑5 min · **Risk**: zero

When the user supplies `TARGET_DIR` (a directory containing prior results
or an explicit numeric goal file), parse it into the `Objective`:

- `target_tput.json` → numeric `tok/s/GPU` goal
- `target_gain_pct.json` → relative gain target
- markdown notes inside `notes.md` are read into the prompt context but
  never become hard goals

Outputs:

- `update_state` with refined `current_best` / target metadata
- `send_message` topic=`event` summarizing the target.
