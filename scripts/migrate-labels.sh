#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
  # type:
  ["type:bug"]="d73a4a"
  ["type:feature"]="a2eeef"
  ["type:question"]="d876e3"
  ["type:task"]="0075ca"
  ["type:docs"]="0075ca"
  # domain:
  ["domain:inference"]="c5def5"
  ["domain:training"]="c5def5"
  ["domain:mcp"]="c5def5"
  ["domain:ui"]="c5def5"
  # release:
  ["release:v0.2"]="5319e7"
  ["release:v0.3"]="5319e7"
  ["release:v0.4"]="5319e7"
)

declare -A DESCRIPTIONS=(
  ["type:bug"]="Something is broken or not working as expected"
  ["type:feature"]="New capability or improvement"
  ["type:question"]="Further information is requested"
  ["type:task"]="Internal task or chore"
  ["type:docs"]="Documentation improvements"
  ["domain:inference"]="Related to inference optimization"
  ["domain:training"]="Related to training optimization"
  ["domain:mcp"]="Related to MCP tools"
  ["domain:ui"]="Related to UI / PrimusClaw"
  ["release:v0.2"]="Release v0.2"
  ["release:v0.3"]="Release v0.3"
  ["release:v0.4"]="Release v0.4"
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

migrate_label "Bug"           "type:bug"
migrate_label "Feature"       "type:feature"
migrate_label "Enhancement"   "type:feature"
migrate_label "documentation" "type:docs"
migrate_label "question"      "type:question"
migrate_label "Task"          "type:task"

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

OLD_LABELS=("Bug" "Feature" "Enhancement" "documentation" "question" "Task" "Release-v0.2" "Release-v0.3" "duplicate" "invalid" "wontfix")
for label in "${OLD_LABELS[@]}"; do
  echo "  Deleting: $label"
  gh label delete "$label" --repo "$REPO" --yes 2>/dev/null || true
done

echo ""
echo "=== Migration complete! ==="
echo "Run 'gh label list --repo $REPO' to verify."
