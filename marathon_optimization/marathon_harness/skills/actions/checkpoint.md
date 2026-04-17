# Action: Checkpoint — State Persistence

**DFS role:** Runs after every KEEP decision, at tier boundaries, and before risky
operations (Strategy D/E framework patches). Not scored — triggered automatically.

## When to Trigger

| Trigger | Priority |
|---------|----------|
| After every KEEP decision | Mandatory (IR-7) |
| At tier boundary (Sprint→Standard, etc.) | Mandatory (IR-9) |
| Before Strategy D/E framework patches | Recommended |
| Before server restart after crash | Recommended |
| Every 30 minutes of wall clock | Automatic safety net |

## Procedure

### Save Checkpoint

```bash
python3 $SKILL_ROOT/scripts/checkpoint.py save \
    --state-json "$RESULT_DIR/state.json" \
    --output "$RESULT_DIR/checkpoints/checkpoint_$(date +%s).json" \
    --metadata "tier=$CURRENT_TIER,action=$LAST_ACTION,gain=$CUMULATIVE_GAIN"
```

The checkpoint captures:
- Full `state` dict (see SKILL.md State Schema)
- Current server config (saved as `server_config.sh`)
- List of applied patches (kernel patches, framework edits)
- KB entries added during this run
- Winning backends and params
- Profile tier breakdown

### Restore from Checkpoint

```bash
# List available checkpoints
python3 $SKILL_ROOT/scripts/checkpoint.py list \
    --checkpoint-dir "$RESULT_DIR/checkpoints"

# Restore most recent
python3 $SKILL_ROOT/scripts/checkpoint.py restore \
    --checkpoint "$RESULT_DIR/checkpoints/checkpoint_XXXXX.json" \
    --output "$RESULT_DIR/state.json"
```

After restore:
1. Reload `state.json` into agent's state dict
2. Re-apply patches listed in checkpoint (or verify they're still applied)
3. Verify server config matches checkpoint
4. Resume from the step after the last completed action

## Outputs
- Checkpoint file at `$RESULT_DIR/checkpoints/checkpoint_<timestamp>.json`
- Symlink `$RESULT_DIR/checkpoints/latest` → most recent checkpoint

## Failure Handling
- Write failure: retry once with alternate path; if persistent, log and continue without checkpoint
- Restore failure: fall back to full restart (Step 1 of protocol)
- State corruption: validate JSON structure before restore; reject and try previous checkpoint
