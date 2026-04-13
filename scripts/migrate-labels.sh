#!/usr/bin/env bash
#
# Label & Milestone migration script for AMD-AGI/Hyperloom
# Prerequisites: gh CLI authenticated with repo + project access
#
# Usage:
#   chmod +x scripts/migrate-labels.sh
#   ./scripts/migrate-labels.sh
#
set -euo pipefail

REPO="AMD-AGI/Hyperloom"

echo "=== Step 1: Create new labels ==="

declare -A LABELS=(
  # kind:
  ["kind:bug"]="d73a4a"
  ["kind:feature"]="a2eeef"
  ["kind:question"]="d876e3"
  ["kind:task"]="0075ca"
  ["kind:docs"]="0075ca"
  # domain:
  ["domain:inference"]="c5def5"
  ["domain:training"]="c5def5"
  ["domain:mcp"]="c5def5"
  ["domain:ui"]="c5def5"
  # priority:
  ["priority:critical"]="b60205"
  ["priority:high"]="d93f0b"
  ["priority:medium"]="fbca04"
  ["priority:low"]="0e8a16"
  # release:
  ["release:v0.2"]="5319e7"
  ["release:v0.3"]="5319e7"
  ["release:v0.4"]="5319e7"
  # status:
  ["needs-triage"]="e4e669"
  ["auto-answered"]="bfdadc"
)

declare -A DESCRIPTIONS=(
  ["kind:bug"]="Something is broken or not working as expected"
  ["kind:feature"]="New capability or improvement"
  ["kind:question"]="Further information is requested"
  ["kind:task"]="Internal task or chore"
  ["kind:docs"]="Documentation improvements"
  ["domain:inference"]="Related to inference optimization"
  ["domain:training"]="Related to training optimization"
  ["domain:mcp"]="Related to MCP tools"
  ["domain:ui"]="Related to UI / PrimusClaw"
  ["priority:critical"]="Production issue or blocker"
  ["priority:high"]="Important, should be addressed soon"
  ["priority:medium"]="Normal priority"
  ["priority:low"]="Nice to have, no rush"
  ["release:v0.2"]="Release v0.2"
  ["release:v0.3"]="Release v0.3"
  ["release:v0.4"]="Release v0.4"
  ["needs-triage"]="New issue awaiting classification"
  ["auto-answered"]="Answered automatically by QA bot"
)

for label in "${!LABELS[@]}"; do
  color="${LABELS[$label]}"
  desc="${DESCRIPTIONS[$label]:-}"
  echo "  Creating: $label"
  gh label create "$label" --repo "$REPO" --color "$color" --description "$desc" --force 2>/dev/null || true
done

echo ""
echo "=== Step 2: Migrate old labels to new labels ==="

migrate_label() {
  local old="$1" new="$2"
  echo "  Migrating: $old -> $new"
  issues=$(gh issue list --repo "$REPO" --label "$old" --state all --json number --jq '.[].number' 2>/dev/null || true)
  for num in $issues; do
    echo "    Issue #$num: adding $new, removing $old"
    gh issue edit "$num" --repo "$REPO" --add-label "$new" --remove-label "$old" 2>/dev/null || true
  done
}

migrate_label "Bug"           "kind:bug"
migrate_label "Feature"       "kind:feature"
migrate_label "Enhancement"   "kind:feature"
migrate_label "documentation" "kind:docs"
migrate_label "question"      "kind:question"
migrate_label "Task"          "kind:task"

echo ""
echo "=== Step 3: Migrate Release labels to new format ==="

for version in "v0.2" "v0.3"; do
  old_label="Release-$version"
  new_label="release:$version"
  echo "  Migrating: $old_label -> $new_label"
  migrate_label "$old_label" "$new_label"
done

echo ""
echo "=== Step 4: Delete old labels ==="

OLD_LABELS=("Bug" "Feature" "Enhancement" "documentation" "question" "Task" "Release-v0.2" "Release-v0.3")
for label in "${OLD_LABELS[@]}"; do
  echo "  Deleting: $label"
  gh label delete "$label" --repo "$REPO" --yes 2>/dev/null || true
done

echo ""
echo "=== Migration complete! ==="
echo "Run 'gh label list --repo $REPO' to verify."
