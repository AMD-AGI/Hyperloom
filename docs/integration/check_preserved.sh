#!/usr/bin/env bash
# F0/F1/F2/F3 — preserved-asset verification.
#
# Run after every cherry-pick / rebase to make sure no v0.8-only asset
# was accidentally clobbered by an upstream change.
#
# Exit codes:
#   0  all checks passed
#   1+ first failed check (message points at the offending check)

set -euo pipefail

FEAT_ROOT="${FEAT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$FEAT_ROOT"

echo "[check_preserved] FEAT_ROOT=$FEAT_ROOT"

# ---- Section 1: deleted files must stay deleted ---------------------------
DELETED=(
  "inference_optimizer/orchestrator/action_executors/backends.py"
  "inference_optimizer/orchestrator/action_executors/params.py"
  "inference_optimizer/orchestrator/action_executors/validate_stack.py"
  "inference_optimizer/orchestrator/scoring.py"
  "inference_optimizer/actions/_meta/backends.yaml"
  "inference_optimizer/actions/_meta/params.yaml"
  "inference_optimizer/actions/_meta/validate_stack.yaml"
  "inference_optimizer/actions/validate_stack.md"
  "inference_optimizer/tests/test_p3_search_space_expansion.py"
  "inference_optimizer/tests/test_validate_stack.py"
  "inference_optimizer/tests/test_validate_stack_gate_skip.py"
)
echo "[1/6] Deleted files (must stay deleted)"
for f in "${DELETED[@]}"; do
  if [[ -f "$f" ]]; then
    echo "  FAIL: $f reappeared"
    exit 1
  fi
done
echo "  OK (${#DELETED[@]} files confirmed absent)"

# ---- Section 2: required v0.8 files must exist ----------------------------
REQUIRED=(
  "inference_optimizer/orchestrator/phase_state.py"
  "inference_optimizer/orchestrator/specialist_runner.py"
  "inference_optimizer/orchestrator/specialist_subprocess.py"
  "inference_optimizer/orchestrator/specialist_domains.py"
  "inference_optimizer/orchestrator/system_prompts/specialist_prompt_builder.py"
)
echo "[2/6] Required v0.8 files"
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  FAIL: $f missing"
    exit 2
  fi
done
echo "  OK (${#REQUIRED[@]} files present)"

# ---- Section 3: Iron rules in SKILL.md ------------------------------------
echo "[3/6] Iron rules in SKILL.md"
N=$(grep -cE "^### IR-[0-9]+" inference_optimizer/SKILL.md || echo 0)
if (( N < 6 )); then
  echo "  FAIL: only $N IR rules found, expected >= 6 (IR-1..4, IR-6, IR-7)"
  exit 3
fi
echo "  OK ($N IR rules present)"

# ---- Section 4: PolicyGate / Coordinator rule ids -------------------------
echo "[4/6] PolicyGate + Coordinator rules"
P_RULES=$(grep -cE "rule=['\"]?(action_deprecated|explore_requires_specialist_provenance)" \
            inference_optimizer/orchestrator/policy.py || echo 0)
C_RULES=$(grep -cE "rule=['\"]?assess_remaining_gaps_throttle" \
            inference_optimizer/orchestrator/coordinator.py || echo 0)
if (( P_RULES < 2 )); then
  echo "  FAIL: only $P_RULES policy.py rules out of {action_deprecated, explore_requires_specialist_provenance}"
  exit 4
fi
if (( C_RULES < 1 )); then
  echo "  FAIL: assess_remaining_gaps_throttle missing from coordinator.py"
  exit 4
fi
echo "  OK ($P_RULES policy rules, $C_RULES coordinator rules)"

# ---- Section 5: SharedState fields ----------------------------------------
echo "[5/6] SharedState v0.8 fields"
SS_FIELDS=$(grep -cE "(last_remaining_gaps_assessment|remaining_gaps_assessments|steward_continuation_used|specialist_domain_empty_streak)" \
            inference_optimizer/orchestrator/shared_state.py || echo 0)
if (( SS_FIELDS < 4 )); then
  echo "  FAIL: only $SS_FIELDS / 4 v0.8 SharedState fields present"
  exit 5
fi
echo "  OK ($SS_FIELDS v0.8 fields confirmed)"

# ---- Section 6: v0.8 test files exist -------------------------------------
#
# Commit 247888e (Sun May 24 2026) renamed the milestone-prefixed
# files; the table below maps each pre-rename name to its post-rename
# home so the check passes either way.
echo "[6/6] v0.8 test files"
V08_TESTS=(
  "inference_optimizer/tests/test_phase_force_exit.py"
  "inference_optimizer/tests/test_assess_remaining_gaps.py"
  "inference_optimizer/tests/test_phase_state_machine.py"        # was test_v08_m2_phase_machine.py
  "inference_optimizer/tests/test_per_domain_prompts.py"          # absorbed test_v08_m5_specialist.py
  "inference_optimizer/tests/test_specialist_subprocess.py"
)
for f in "${V08_TESTS[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  FAIL: $f missing"
    exit 6
  fi
done
echo "  OK (${#V08_TESTS[@]} test files present)"

echo ""
echo "[check_preserved] ALL PRESERVED OK"
