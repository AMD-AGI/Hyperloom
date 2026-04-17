# Action: Dream — Knowledge Consolidation (Marathon)

**DFS role:** Mandatory every 3-4 hours wall clock. Also triggered at tier boundaries,
plateau detection, and Sprint→Marathon transition. Every dream MUST contribute to KB.

Consolidates learnings from the current run and cross-run history into actionable
KB entries and model-specific playbooks. Marathon runs can span hours to days — dream
prevents knowledge loss and enables future runs to benefit.

## When to Trigger

| Trigger | Context |
|---------|---------|
| **Sprint→Marathon transition** | Before Marathon DFS starts — consolidate Sprint findings |
| **Every 3-4h wall clock** | Mandatory cadence — `DREAM_CADENCE_MIN` (210 min default) |
| **Tier boundary** (Tier 1→2→3→4) | Before deepening optimization scope |
| **3+ consecutive discards** (plateau) | Re-examine assumptions, challenge "already tuned" claims |
| **After major success** (single action >5% gain) | Capture and generalize the insight |
| **End of Marathon run** | Full consolidation before final report |
| **Explicit user request** | Any time |

**Non-negotiable: every dream MUST write at least 1 new KB entry.** Dreams that only
read without contributing are violations.

## Procedure

### Phase 1: Orient

Assess what happened since the last dream (or since Marathon start).

```python
import time, json

completed = state["completed_actions"]
last_dream = state.get("last_dream_ts", 0)
since_last = [a for a in completed if a.get("timestamp", 0) > last_dream]

gains = [a for a in since_last if a.get("gain_pct", 0) > 0]
losses = [a for a in since_last if a.get("gain_pct", 0) < 0]
crashes = [a for a in since_last if a.get("status") == "crash"]

# Marathon-specific metrics
dispatch_bugs = state.get("dispatch_bugs_found", [])
untuned_shapes = state.get("untuned_shapes", [])
frameworks_rebuilt = state.get("frameworks_rebuilt", [])
strategies_tested = set(state.get("strategies_tested", []))
all_strategies = {"A", "B", "B'", "C", "D", "E", "F", "G"}
untested = all_strategies - strategies_tested

print(f"=== Dream #{state.get('dream_count', 0) + 1} ===")
print(f"Since last dream: {len(since_last)} actions")
print(f"  Gains: {len(gains)}, Losses: {len(losses)}, Crashes: {len(crashes)}")
print(f"Cumulative Marathon gain: {state['cumulative_gain_pct']:.1f}%")
print(f"Strategies tested: {sorted(strategies_tested)}")
print(f"Strategies untested: {sorted(untested)}")
print(f"Dispatch bugs found: {len(dispatch_bugs)}")
print(f"Untuned shapes found: {len(untuned_shapes)}")
print(f"Libraries rebuilt: {frameworks_rebuilt}")
print(f"Wall time: {state['total_wall_minutes']:.0f} min")
```

### Phase 2: Gather Signal

```bash
# 1. Query recent KB entries from this run
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME" --top-k 50 --compact | \
    grep "$(date +%Y-%m-%d)" > /tmp/dream_today_entries.txt

# 2. Check for contradictions with existing KB
python3 $SKILL_ROOT/kb/kb_query.py --category pitfall --top-k 20 --compact

# 3. Check for model-class patterns
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_CLASS kernel optimization" --top-k 10 --compact

# 4. Check deep analysis findings not yet acted on
python3 -c "
import json
dispatch_map = json.loads('$KERNEL_DISPATCH_MAP')
for k, v in dispatch_map.items():
    if v.get('dispatch_bug') and v.get('status') != 'fixed':
        print(f'UNFIXED DISPATCH BUG: {k}')
    if v.get('config_status') == 'generic-default' and v.get('status') != 'tuned':
        print(f'UNTUNED SHAPE: {k}')
"
```

### Phase 3: Consolidate + Contribute (MANDATORY)

This phase MUST produce at least 1 new KB entry. No exceptions.

```bash
# 1. Programmatic consolidation
python3 $SKILL_ROOT/scripts/dream.py consolidate \
    --kb-path "$SKILL_ROOT/kb/entries.jsonl" \
    --model "$MODEL_NAME" \
    --run-id "$RUN_ID" \
    --output "$RESULT_DIR/dream_$(date +%s).json"

# 2. MANDATORY: Contribute new findings to KB
# Categories of contributions:

# a) New optimization result (gain or loss)
for action in since_last_dream_with_results:
    python3 $SKILL_ROOT/kb/kb_ingest.py \
        --category "$ACTION_CATEGORY" \
        --model "$MODEL_NAME" \
        --action "$ACTION_DESCRIPTION" \
        --lesson "$KEY_TAKEAWAY" \
        --tags "$TAGS" \
        --gain $GAIN_PCT --status $STATUS

# b) Dispatch bug discovery (high-value for future runs)
if dispatch_bugs_found:
    python3 $SKILL_ROOT/kb/kb_ingest.py \
        --category dispatch_bug \
        --model "$MODEL_NAME" \
        --action "Dispatch bug: $BUG_DESCRIPTION" \
        --lesson "Framework was routing to suboptimal kernel path" \
        --tags "dispatch-bug,$FRAMEWORK,$GPU_TYPE" \
        --gain $FIX_GAIN_PCT --status KEEP

# c) Untested strategy identification
for strategy in untested_strategies:
    python3 $SKILL_ROOT/kb/kb_ingest.py \
        --category strategy_gap \
        --model "$MODEL_NAME" \
        --action "Strategy $STRATEGY not yet tested" \
        --lesson "Marathon has not attempted $STRATEGY_DESCRIPTION" \
        --tags "strategy-untested,$STRATEGY" \
        --gain 0 --status PENDING

# d) Cross-model insight (if applicable)
# If this model's findings generalize to a model class, contribute a
# class-level entry rather than model-specific
```

### Phase 4: Re-score Action Stack

After consolidation, re-evaluate the entire action stack:

```python
for i, (score, action_name, params) in enumerate(state['action_stack']):
    new_score = score

    # Boost untested strategies
    if action_name in untested_strategies:
        new_score *= 1.5
        print(f"  Boosted {action_name}: {score:.1f} → {new_score:.1f} (untested)")

    # Boost actions related to dispatch bugs not yet fixed
    if action_name == 'framework-rebuild' and unfixed_dispatch_bugs:
        new_score *= 2.0

    # Boost operator-tuning if untuned shapes discovered
    if action_name == 'operator-tuning' and untuned_shapes:
        new_score *= 1.5

    # Reduce strategies consistently failing across models (from KB)
    if strategy_consistently_fails(action_name, model_class):
        new_score *= 0.5

    # Apply cross-model insights
    if playbook_recommends(action_name, model_class):
        new_score *= 1.3

    state['action_stack'][i] = (new_score, action_name, params)

# Re-sort by score
state['action_stack'].sort(reverse=True, key=lambda x: x[0])
```

### Phase 5: Prune and Index

```bash
# 1. Generate/update model-specific playbook
python3 $SKILL_ROOT/scripts/kb_summary.py \
    --model "$MODEL_NAME" \
    --kb-path "$SKILL_ROOT/kb/entries.jsonl" \
    --output "$SKILL_ROOT/kb/playbooks/${MODEL_NAME_SAFE}.md"

# 2. Prune low-confidence superseded entries
python3 $SKILL_ROOT/scripts/dream.py prune \
    --kb-path "$SKILL_ROOT/kb/entries.jsonl" \
    --min-confidence 0.3 \
    --dry-run  # Always dry-run first, then apply

# 3. Update KNOWLEDGE-BASE.md if this model has new findings
# Check if model section exists and update, or add new section
```

### Phase 6: Checkpoint

Always checkpoint after dream:
```bash
# actions/checkpoint.md
python3 -c "
import json
state['last_dream_ts'] = $(date +%s)
state['dream_count'] = state.get('dream_count', 0) + 1
json.dump(state, open('$RESULT_DIR/state.json', 'w'), indent=2)
"
echo "Dream #${DREAM_COUNT} complete. Next dream at ~$((TOTAL_WALL_MIN + DREAM_CADENCE_MIN)) min"
```

## Sprint→Marathon Transition Dream

Special dream that runs when Marathon starts from a Sprint handoff:

```python
# Orient: summarize Sprint results
sprint_config = json.load(open(f"{SPRINT_HANDOFF_DIR}/handoff/config.json"))
sprint_opps = json.load(open(f"{SPRINT_HANDOFF_DIR}/handoff/opportunities.json"))

print("=== Sprint→Marathon Transition Dream ===")
print(f"Sprint achieved: {sprint_config['cumulative_gain_pct']:.1f}% gain")
print(f"Sprint throughput: {sprint_config['optimized_tput_per_gpu']:.2f} tok/s/GPU")
print(f"Marathon opportunities: {len(sprint_opps)}")

# Consolidate: which Sprint findings need deeper investigation?
for opp in sprint_opps:
    if 'register-pressure-fixable' in opp.get('tags', []):
        print(f"  HIGH PRIORITY: {opp['kernel_name']} — register-constrained OOB needed")
    if 'shape-tuning-untested' in opp.get('tags', []):
        print(f"  MEDIUM PRIORITY: {opp['kernel_name']} — GEMM shape tuning")
    if 'oob-untested' in opp.get('tags', []):
        print(f"  STANDARD: {opp['kernel_name']} — OOB agents not yet tried")

# Contribute: KB entry for Sprint→Marathon transition
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category sprint_marathon_transition \
    --model "$MODEL_NAME" \
    --action "Sprint complete, Marathon starting" \
    --lesson "Sprint achieved ${SPRINT_GAIN}%. Marathon targets: ${N_OPPS} deep opportunities" \
    --tags "sprint-complete,marathon-start,$MODEL_CLASS" \
    --gain ${SPRINT_GAIN} --status KEEP
```

## Outputs

- `dream_report`: summary of consolidation actions taken
- **At least 1 new KB entry** (mandatory)
- Updated `kb/entries.jsonl` with resolved contradictions and merged duplicates
- Model-specific playbook at `kb/playbooks/<model>.md`
- Re-scored action stack
- Updated `state.last_dream_ts` and `state.dream_count`
- Checkpoint saved

## Accuracy Validation
N/A — dream is read-only analysis + KB updates, no model changes.

## Heuristic Update

After dream, ALL actions on the stack are re-scored (see Phase 4):
- Boost untested strategies (1.5×)
- Boost actions related to unfixed dispatch bugs (2.0×)
- Boost operator-tuning when untuned shapes exist (1.5×)
- Reduce consistently failing strategies (0.5×)
- Apply cross-model playbook insights (1.3×)

## Failure Handling
- KB parse error: skip consolidation, log warning, continue run
- Playbook generation fails: skip, KB entries are still the source of truth
- Contribution fails: retry once, then log to RESULT_DIR as fallback
